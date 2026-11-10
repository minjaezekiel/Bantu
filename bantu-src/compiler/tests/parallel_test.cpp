// ════════════════════════════════════════════════════════════════════════
//  parallel_test.cpp — the pool's LIFETIME and ERROR properties, at the C++
//  level, because they are not reachable from Bantu.
//
//  Two properties here have no Bantu-visible caller yet:
//
//    * a body that THROWS. Every ufunc that can throw (nd_left_shift on a
//      count outside 0..63) is in the CHEAP cost class and so runs
//      sequentially, and no threaded kernel throws. But an exception escaping
//      a worker thread is std::terminate -- a crash where the sequential path
//      raises a catchable error -- so the pool catches per chunk, keeps the
//      lowest-indexed exception and rethrows it on the caller's thread. The
//      next threaded kernel (bkit's histogram builder, ecosystem stage 3) is
//      the first that will rely on it, and untested is untrue.
//    * which exception wins. Kept by chunk INDEX, not arrival, so the message
//      a user sees does not depend on the thread count.
//
//  Build and run (not part of build-mac.sh: it links nothing else):
//    g++ -std=c++17 -O2 -I src tests/parallel_test.cpp -o /tmp/ptest && /tmp/ptest
//  And under the sanitizer that found the lifetime bug (DECISIONS N33):
//    g++ -std=c++17 -O1 -g -fsanitize=thread -I src tests/parallel_test.cpp \
//        -o /tmp/ptest-tsan && /tmp/ptest-tsan
// ════════════════════════════════════════════════════════════════════════

#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// parallel.hpp is self-contained; nothing else in numba is needed.
namespace numba {}
#include "parallel.hpp"

static int failures = 0;
static void ok(bool cond, const char* name) {
    std::printf(cond ? "  ok    %s\n" : "  FAIL  %s\n", name);
    if (!cond) failures++;
}

int main() {
    using namespace numba;
    std::printf("\n  numba pool: lifetime and errors\n");
    std::printf("  ════════════════════════════════\n\n");

    // ── 1. every chunk runs exactly once, at every thread count ─────────────
    for (size_t threads : {1u, 2u, 3u, 4u, 8u, 16u}) {
        setThreads(threads);
        const size_t n = 1000;
        std::vector<std::atomic<int>> seen(n);
        for (auto& c : seen) c.store(0);
        parallelFor(n, [&](size_t i) { seen[i].fetch_add(1); });
        bool exactlyOnce = true;
        for (size_t i = 0; i < n; i++) if (seen[i].load() != 1) exactlyOnce = false;
        ok(exactlyOnce, ("every chunk ran exactly once, " +
                         std::to_string(threads) + " threads").c_str());
    }

    // ── 2. a throwing body ──────────────────────────────────────────────────
    // The caller must see the exception, not std::terminate, and must see the
    // one from the LOWEST chunk index however the chunks were scheduled.
    for (size_t threads : {1u, 2u, 4u, 8u, 16u}) {
        setThreads(threads);
        std::string msg;
        bool caught = false;
        try {
            parallelFor(64, [](size_t i) {
                // Several chunks fail. 17 is the lowest, so 17 must win.
                if (i == 17 || i == 23 || i == 60) {
                    throw std::runtime_error("chunk " + std::to_string(i));
                }
                // make the later chunks slower, so arrival order would differ
                if (i > 17) std::this_thread::yield();
            });
        } catch (const std::exception& e) {
            caught = true;
            msg = e.what();
        }
        ok(caught, ("a throwing body reaches the caller, " +
                    std::to_string(threads) + " threads").c_str());
        ok(msg == "chunk 17",
           ("and it is the lowest-indexed failure (got \"" + msg + "\"), " +
            std::to_string(threads) + " threads").c_str());
    }

    // ── 3. the pool survives the error ──────────────────────────────────────
    // If a failure left busy_ set, every later region in the process would
    // silently run sequentially -- a performance bug that no test of the
    // ANSWERS could ever see.
    setThreads(8);
    {
        std::atomic<int> ran{0};
        parallelFor(200, [&](size_t) { ran.fetch_add(1); });
        ok(ran.load() == 200, "a region after an error still runs every chunk");

        // Each chunk has to do real work for this one. With a body that is a
        // single fetch_add, the CALLING thread drains all 200 chunks in about
        // a microsecond -- long before a worker can be woken -- and the region
        // legitimately runs on one thread. That is the pool working, not the
        // pool left claimed, and asserting otherwise is a test that fails for
        // the wrong reason (it did).
        std::vector<std::thread::id> ids(64);
        parallelFor(64, [&](size_t i) {
            volatile double sink = 0.0;
            for (int k = 0; k < 20000; k++) sink += (double)k * 1.000001;   // ~50us at -O2
            (void)sink;
            ids[i] = std::this_thread::get_id();
        });
        size_t distinct = 0;
        for (size_t i = 0; i < ids.size(); i++) {
            bool first = true;
            for (size_t j = 0; j < i; j++) if (ids[j] == ids[i]) first = false;
            if (first) distinct++;
        }
        ok(distinct > 1, "and it still runs on more than one thread "
                         "(the pool was not left claimed)");
    }

    // ── 4. nesting ──────────────────────────────────────────────────────────
    // An inner region must run sequentially rather than claim the pool again.
    setThreads(8);
    {
        std::atomic<int> inner{0};
        parallelFor(16, [&](size_t) {
            parallelFor(4, [&](size_t) { inner.fetch_add(1); });
        });
        ok(inner.load() == 64, "a nested region runs every chunk, sequentially");
    }

    // ── 5. concurrent callers ───────────────────────────────────────────────
    // sua runs each connection's handler on its own thread (N18). A caller
    // that finds the pool busy runs the region itself; nobody blocks and
    // nobody loses a chunk.
    setThreads(6);
    {
        std::atomic<int> bad{0};
        std::vector<std::thread> callers;
        for (int c = 0; c < 12; c++) {
            callers.emplace_back([&] {
                for (int r = 0; r < 60; r++) {
                    std::vector<std::atomic<int>> seen(97);
                    for (auto& s : seen) s.store(0);
                    parallelFor(97, [&](size_t i) { seen[i].fetch_add(1); });
                    for (size_t i = 0; i < 97; i++) if (seen[i].load() != 1) bad++;
                }
            });
        }
        for (auto& t : callers) t.join();
        ok(bad.load() == 0, "12 concurrent callers x 60 regions, no chunk lost or doubled");
    }

    // ── 6. a worker created INSIDE a region ─────────────────────────────────
    // This is the shape of the bug TSan found (N33): the first region after a
    // worker is born is the one where the completion counter can go wrong. So
    // grow the pool one thread at a time and check each first region.
    {
        bool allRan = true;
        for (size_t threads = 2; threads <= 12; threads++) {
            setThreads(threads);                 // creates threads-1 workers
            std::vector<std::atomic<int>> seen(500);
            for (auto& s : seen) s.store(0);
            parallelFor(500, [&](size_t i) { seen[i].fetch_add(1); });
            for (size_t i = 0; i < 500; i++) if (seen[i].load() != 1) allRan = false;
        }
        ok(allRan, "the first region after each new worker is born is complete");
    }

    setThreads(1);
    std::printf("\n  %s\n", failures == 0 ? "RESULT: ALL GREEN"
                                          : "RESULT: FAILURES PRESENT");
    return failures == 0 ? 0 : 1;
}
