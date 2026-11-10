#pragma once
// ═══════════════════════════════════════════════════════════════════════════
//  numba — deterministic parallelism (ecosystem stage 1, item D3)
//
//  One primitive: parallelFor(nChunks, body) runs body(0..nChunks-1).
//
//  THE DETERMINISM CONTRACT, which is the whole point of this file:
//
//    1. CHUNK BOUNDARIES are a function of the data's shape ALONE -- never of
//       the thread count. The caller computes them; see chunkCount() below.
//    2. CHUNK -> WORKER ASSIGNMENT is arbitrary and dynamic. It does not
//       matter who runs chunk i, only that chunk i always computes the same
//       thing.
//    3. The caller either writes to storage no other chunk touches, or
//       produces one partial per chunk and COMBINES THEM IN ASCENDING INDEX
//       ORDER.
//
//  (1) + (3) are what make the answer bit-identical at any thread count, which
//  is the property that makes this stack worth choosing (DECISIONS N25). (2)
//  is then free, and buys load balance for nothing -- an earlier draft of this
//  design fixed the assignment too, which was a cost with no benefit.
//
//  Floating-point addition is not associative, so a work-stealing pool that
//  let completion order decide summation order would change results run to
//  run. That is the thing being avoided, and it is avoided by (3), not by
//  pinning threads to chunks.
//
//  CONCURRENCY: sua runs each connection's Bantu handler on a detached thread
//  (N18), so two requests can enter a threaded kernel at the same instant. A
//  caller that finds the pool already in use RUNS SEQUENTIALLY rather than
//  blocking. That is sound precisely because of the contract above: the
//  sequential answer is the same answer, so there is nothing to lose. A
//  work-stealing design would have had to choose between blocking an unrelated
//  HTTP request and giving up reproducibility.
//
//  LIFETIME: the body is a std::function owned by the CALLER'S STACK and the
//  pool holds only a pointer to it, so run() may not return until every worker
//  that touched it has finished. The counter that enforces that -- active_ --
//  counts exactly the workers a region asked to take chunks, and each of those
//  reports back exactly once. Getting that count wrong by one in either
//  direction is either a hang or a use-after-free, so the two rules are
//  written out where they are implemented (see ensureWorkers).
//
//  ALLOCATION: the sequential path -- threads == 1, pool busy, nested call, or
//  one chunk -- takes no lock and allocates nothing. It is a plain loop. Given
//  the default of 1 thread that is what almost every call does. Phase 9 found
//  nditer heap-allocating three vectors per call, which cost ~450,000
//  allocations in one pooling layer; a pool that allocated per region would
//  reintroduce that defect one level up.
// ═══════════════════════════════════════════════════════════════════════════

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace numba {

// ── the thread count ────────────────────────────────────────────────────────
// Process-global and atomic, NOT thread_local -- a deliberate departure from
// N18's PRNG precedent, recorded as N32. N18 made the PRNG thread-local
// because a shared stream lets one request silently reshape another's results.
// A shared thread count cannot: every thread count produces identical output.
// So the objection does not apply, and a global is far more useful -- set once
// at startup instead of in every sua handler.
inline std::atomic<size_t>& threadCountCell() {
    static std::atomic<size_t> n{1};        // opt-in; 1 until asked otherwise
    return n;
}

inline size_t hardwareThreads() {
    const unsigned h = std::thread::hardware_concurrency();
    return h == 0 ? 1 : static_cast<size_t>(h);
}

inline size_t getThreads() {
    return threadCountCell().load(std::memory_order_relaxed);
}

// n == 0 means "as many as the hardware reports". Capped so a bad argument
// cannot spawn thousands of threads.
inline size_t setThreads(size_t n) {
    if (n == 0) n = hardwareThreads();
    if (n > 256) n = 256;
    threadCountCell().store(n, std::memory_order_relaxed);
    return n;
}

// Already inside a parallel region on THIS thread? A threaded kernel calling
// another threaded kernel must not try to claim the pool again -- it runs
// sequentially instead, so nesting can neither deadlock nor oversubscribe.
inline bool& inRegionFlag() {
    static thread_local bool v = false;
    return v;
}

// ── the pool ────────────────────────────────────────────────────────────────
class Pool {
public:
    static Pool& instance() {
        static Pool p;
        return p;
    }

    // Returns true if the region was run in parallel, false if the caller must
    // run it itself. Never throws, never blocks on another caller.
    bool run(size_t nChunks, const std::function<void(size_t)>& body, size_t want) {
        if (want <= 1 || nChunks <= 1) return false;
        if (inRegionFlag())            return false;     // nested
        // One claimant at a time. A loser runs sequentially; see the header.
        if (busy_.test_and_set(std::memory_order_acquire)) return false;

        inRegionFlag() = true;
        ensureWorkers(want - 1);                         // the caller is a worker too

        {
            std::lock_guard<std::mutex> lk(m_);
            body_    = &body;
            nChunks_ = nChunks;
            limit_   = want - 1;                         // how many WORKERS take chunks
            next_.store(0, std::memory_order_relaxed);
            // Only the workers that will TAKE chunks are counted, and only
            // those report back. A region that asks for fewer threads than the
            // pool has already created leaves the surplus parked without
            // touching the counter at all.
            active_  = std::min(limit_, workers_.size());
            generation_++;
        }
        cvWork_.notify_all();

        drain(body, nChunks);                            // the calling thread works too

        {
            std::unique_lock<std::mutex> lk(m_);
            cvDone_.wait(lk, [this] { return active_ == 0; });
            body_ = nullptr;
        }

        inRegionFlag() = false;
        busy_.clear(std::memory_order_release);
        return true;
    }

    ~Pool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
            generation_++;
        }
        cvWork_.notify_all();
        for (auto& t : workers_) if (t.joinable()) t.join();
    }

private:
    Pool() = default;
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    // Chunks are claimed with an atomic counter. This is dynamic assignment and
    // it is deliberate: see contract point (2). Determinism comes from the
    // chunk BOUNDARIES and the caller's combine order, not from who runs what.
    void drain(const std::function<void(size_t)>& body, size_t nChunks) {
        for (;;) {
            const size_t i = next_.fetch_add(1, std::memory_order_relaxed);
            if (i >= nChunks) return;
            body(i);
        }
    }

    // Workers are created once and never retired. A later region that wants
    // fewer threads simply tells the surplus not to take chunks (limit_), which
    // is cheaper than tearing threads down and standing them back up.
    // A new worker is given the generation that is current AT ITS BIRTH, so
    // its first wake-up is for the next region -- the one whose `active_` will
    // count it. Starting it at generation 0 instead was a real bug, caught by
    // ThreadSanitizer on the integrated build: the worker would find
    // `generation_ != 0` true the instant it started, consume the region that
    // had ALREADY FINISHED (body_ == nullptr, so it ran nothing), and then
    // decrement `active_` a second time for the region that was just being set
    // up. One extra decrement means the counter reaches zero while a real
    // worker is still inside the body: the caller stops waiting, its
    // `std::function` on the stack dies under the straggler's feet, and the
    // next region's chunks get claimed by a thread still running the last
    // one's body. Nothing in the results showed it -- the window is a few
    // microseconds wide -- which is precisely why the sanitizer gate exists.
    void ensureWorkers(size_t want) {
        std::unique_lock<std::mutex> lk(m_);
        while (workers_.size() < want) {
            const size_t id    = workers_.size();
            const uint64_t gen = generation_;
            workers_.emplace_back([this, id, gen] { workerLoop(id, gen); });
        }
    }

    void workerLoop(size_t id, uint64_t startGen) {
        uint64_t seen = startGen;
        for (;;) {
            const std::function<void(size_t)>* body = nullptr;
            size_t nChunks = 0;
            bool   take    = false;
            {
                std::unique_lock<std::mutex> lk(m_);
                cvWork_.wait(lk, [this, &seen] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
                // Copied under the lock rather than read after unlocking: the
                // values are stable for the region's duration either way, but
                // an unsynchronised read of them is still a data race by the
                // letter of the model and TSan is right to say so.
                body    = body_;
                nChunks = nChunks_;
                take    = (id < limit_);
            }
            if (!take) continue;        // parked: not counted, does not report
            if (body) drain(*body, nChunks);
            {
                std::lock_guard<std::mutex> lk(m_);
                if (--active_ == 0) cvDone_.notify_one();
            }
        }
    }

    std::mutex                               m_;
    std::condition_variable                  cvWork_;
    std::condition_variable                  cvDone_;
    std::vector<std::thread>                 workers_;
    bool                                     stop_       = false;
    uint64_t                                 generation_ = 0;
    const std::function<void(size_t)>*       body_       = nullptr;
    size_t                                   nChunks_    = 0;
    size_t                                   limit_      = 0;
    size_t                                   active_     = 0;
    std::atomic<size_t>                      next_{0};
    std::atomic_flag                         busy_ = ATOMIC_FLAG_INIT;
};

// ── the primitive ───────────────────────────────────────────────────────────
// Runs body(0), body(1), ... body(nChunks-1), in parallel when that is possible
// and sequentially when it is not. Identical results either way.
template <class Body>
inline void parallelFor(size_t nChunks, Body&& body) {
    const size_t want = getThreads();
    if (want <= 1 || nChunks <= 1 || inRegionFlag()) {       // the common path:
        for (size_t i = 0; i < nChunks; i++) body(i);        // no lock, no alloc
        return;
    }
    const std::function<void(size_t)> fn(std::forward<Body>(body));
    if (!Pool::instance().run(nChunks, fn, want)) {
        for (size_t i = 0; i < nChunks; i++) fn(i);          // pool was busy
    }
}

// How many chunks to split `n` items into: a function of `n` and a minimum
// useful chunk size ONLY -- never of the thread count, which is contract point
// (1) and the reason results do not move when the core count does.
inline size_t chunkCount(size_t n, size_t minPerChunk) {
    if (n <= minPerChunk) return 1;
    return (n + minPerChunk - 1) / minPerChunk;
}

}   // namespace numba
