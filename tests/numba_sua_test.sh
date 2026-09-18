#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  numba_sua_test.sh — numba inside concurrent sua request handlers.
#
#  WHY THIS EXISTS
#  sua runs Bantu handlers on its event loop, one at a time; a handler that
#  opts into suspension continues on a pooled task thread that is handed a
#  baton, so handlers do run on DIFFERENT OS threads, never concurrently.
#  (This comment used to say every connection got its own detached
#  std::thread. That describes server.hpp's SuaServer, which nothing in the
#  tree constructs.) numba is the first thing in the tree to put MUTABLE
#  process-global state behind a builtin, so it is the first to care:
#
#    1. The allocation counters are shared across every request. They are
#       atomic (N18); if they were not, concurrent writes would be a data
#       race, and the live-byte total would drift until numba refused to
#       allocate anything and nothing said why.
#    2. The PRNG is thread_local (N18). A shared stream would mean nd_seed()
#       in one request silently reshaping every other in-flight request's
#       random arrays -- which is the exact objection that made numba carry
#       its own generator instead of using the global random(), one level up.
#    3. A big allocation inside a handler must RAISE, not OOM-kill the
#       worker. A per-call ceiling never bounded a loop, which is how a
#       handler actually exhausts a server (N17).
#
#  Each request computes something only it can know, so a wrong answer is
#  detectable rather than merely suspicious -- the same design as
#  sua_concurrency_test.sh, whose bug was silent and returned confidently
#  wrong answers rather than crashing.
#
#  Run:  BANTU=./bantu-src/compiler/build/bantu bash tests/numba_sua_test.sh
# ════════════════════════════════════════════════════════════════════════
set -u
BANTU="${BANTU:-bantu}"
PORT="${PORT:-39914}"
TMP="$(mktemp -d)"
PASS=0
FAIL=0

cleanup() { [ -n "${PID:-}" ] && kill "$PID" 2>/dev/null; rm -rf "$TMP"; }
trap cleanup EXIT

ok()   { PASS=$((PASS+1)); echo "  ok    $1"; }
bad()  { FAIL=$((FAIL+1)); echo "  FAIL  $1"; }
check(){ if [ "$1" = "1" ]; then ok "$2"; else bad "$2"; fi; }

if ! "$BANTU" -q run /dev/stdin <<'PROBE' 2>/dev/null | grep -q true
print(has_native("ndarray"));
PROBE
then
    echo "  --    numba not built in; skipping"
    exit 0
fi

cat > "$TMP/server.b" <<BEOF
// /work?n=K  — build a K-element array, do real numba work, and return a
// value that depends only on K. Any cross-talk between concurrent handlers
// shows up as a wrong number rather than as a crash.
sua.server.get("/work", def(\$req, \$res) {
    \$n = 1 + num(\$req.query["n"]);
    \$a = nd_arange(0, \$n, null);
    \$sq = nd_multiply(\$a, \$a, null);
    // sum of squares 0..n-1, which has a closed form we check on this side.
    \$got = nd_get(nd_sum(\$sq, null, null), []);
    \$want = (\$n - 1) * \$n * (2 * \$n - 1) / 6;
    \$res.json({"n": \$n, "got": \$got, "want": \$want, "ok": \$got == \$want});
});

// /seeded?s=K — seed THIS thread's generator and return the first draw.
// A shared generator would make these interfere; a thread_local one makes
// each request's answer depend only on its own seed.
sua.server.get("/seeded", def(\$req, \$res) {
    \$s = num(\$req.query["s"]);
    nd_seed(\$s);
    \$v = nd_get(nd_random_uniform([1], 0, 1), [0]);
    \$res.json({"s": \$s, "v": \$v});
});

// /huge — a deliberately impossible allocation inside a handler. It must
// raise catchably and the worker must keep serving.
sua.server.get("/huge", def(\$req, \$res) {
    \$caught = false;
    try { nd_zeros([1000000000000000], null); }
    catch (\$e) { \$caught = true; }
    \$res.json({"caught": \$caught, "alive": nd_get(nd_sum(nd_arange(0, 10, null), null, null), [])});
});

// /live — the live-byte counter, to check it returns to baseline.
sua.server.get("/live", def(\$req, \$res) {
    \$res.json({"live": nd_live_bytes()});
});

sua.server.listen($PORT);
BEOF

"$BANTU" -q run "$TMP/server.b" > "$TMP/server.log" 2>&1 &
PID=$!

# Wait for the port rather than sleeping a guessed amount.
i=0
while [ $i -lt 100 ]; do
    curl -s -o /dev/null "http://127.0.0.1:$PORT/live" 2>/dev/null && break
    sleep 0.1
    i=$((i+1))
done
if [ $i -ge 100 ]; then
    bad "the server never came up"
    cat "$TMP/server.log"
    exit 1
fi

BASE=$(curl -s "http://127.0.0.1:$PORT/live" | sed 's/.*"live"[: ]*\([0-9]*\).*/\1/')

echo "-- 60 concurrent requests, each doing real numba work --"
# Only the request PIDs: a bare `wait` would also wait for the server, which
# never exits, and the suite would hang rather than fail.
PIDS=""
for n in $(seq 1 60); do
    curl -s --max-time 30 "http://127.0.0.1:$PORT/work?n=$((n * 137))" > "$TMP/r$n.json" 2>/dev/null &
    PIDS="$PIDS $!"
done
for p in $PIDS; do wait "$p" 2>/dev/null; done

GOOD=0
MISSING=0
WRONG=0
for n in $(seq 1 60); do
    if [ ! -s "$TMP/r$n.json" ]; then MISSING=$((MISSING+1)); continue; fi
    if grep -q '"ok"[: ]*true' "$TMP/r$n.json"; then GOOD=$((GOOD+1)); else WRONG=$((WRONG+1)); fi
done
echo "        $GOOD correct, $WRONG wrong, $MISSING missing"
check "$([ "$GOOD" = "60" ] && echo 1 || echo 0)" \
      "all 60 concurrent handlers computed the right answer"
check "$([ "$WRONG" = "0" ] && echo 1 || echo 0)" \
      "none returned a confidently wrong number (the failure mode that matters)"
check "$([ "$MISSING" = "0" ] && echo 1 || echo 0)" "and none dropped its reply"

echo ""
echo "-- the PRNG is per-thread, so seeds do not leak between requests --"
# The same seed must give the same draw no matter what else is in flight.
PIDS=""
for r in 1 2 3; do
    for s in 11 22 33; do
        curl -s --max-time 30 "http://127.0.0.1:$PORT/seeded?s=$s" > "$TMP/s_${r}_${s}.json" 2>/dev/null &
        PIDS="$PIDS $!"
    done
done
for p in $PIDS; do wait "$p" 2>/dev/null; done
SEED_OK=1
for s in 11 22 33; do
    a=$(sed 's/.*"v"[: ]*\([0-9.e-]*\).*/\1/' "$TMP/s_1_${s}.json" 2>/dev/null)
    b=$(sed 's/.*"v"[: ]*\([0-9.e-]*\).*/\1/' "$TMP/s_2_${s}.json" 2>/dev/null)
    c=$(sed 's/.*"v"[: ]*\([0-9.e-]*\).*/\1/' "$TMP/s_3_${s}.json" 2>/dev/null)
    if [ -z "$a" ] || [ "$a" != "$b" ] || [ "$a" != "$c" ]; then SEED_OK=0; fi
done
check "$SEED_OK" "the same seed gives the same draw under concurrency"
# Different seeds must still differ, or the check above would pass trivially.
V11=$(sed 's/.*"v"[: ]*\([0-9.e-]*\).*/\1/' "$TMP/s_1_11.json" 2>/dev/null)
V22=$(sed 's/.*"v"[: ]*\([0-9.e-]*\).*/\1/' "$TMP/s_1_22.json" 2>/dev/null)
check "$([ -n "$V11" ] && [ "$V11" != "$V22" ] && echo 1 || echo 0)" \
      "and different seeds still give different draws"

echo ""
echo "-- an impossible allocation raises instead of killing the worker --"
PIDS=""
for n in 1 2 3 4 5 6 7 8 9 10; do
    curl -s --max-time 30 "http://127.0.0.1:$PORT/huge" > "$TMP/h$n.json" 2>/dev/null &
    PIDS="$PIDS $!"
done
for p in $PIDS; do wait "$p" 2>/dev/null; done
CAUGHT=0
for n in 1 2 3 4 5 6 7 8 9 10; do
    grep -q '"caught"[: ]*true' "$TMP/h$n.json" 2>/dev/null && CAUGHT=$((CAUGHT+1))
done
check "$([ "$CAUGHT" = "10" ] && echo 1 || echo 0)" \
      "all 10 impossible allocations were caught inside the handler"
# The worker has to still be serving, which is the whole point.
STILL=$(curl -s "http://127.0.0.1:$PORT/work?n=100" 2>/dev/null)
check "$(printf '%s' "$STILL" | grep -q '"ok"[: ]*true' && echo 1 || echo 0)" \
      "and the server is still serving correct answers afterwards"

echo ""
echo "-- the live-byte counter returns to its baseline --"
# Concurrent increments and decrements on a non-atomic counter would drift,
# and the drift is one-way: numba would slowly refuse to allocate.
AFTER=$(curl -s "http://127.0.0.1:$PORT/live" | sed 's/.*"live"[: ]*\([0-9]*\).*/\1/')
echo "        live bytes: $BASE -> $AFTER after ~80 requests"
check "$([ "$AFTER" = "$BASE" ] && echo 1 || echo 0)" \
      "byte accounting is exact under concurrency (an atomic counter, not a racy one)"

# Silence the shell's own "Terminated" notice for a kill we asked for.
kill "$PID" 2>/dev/null
PID=""
wait 2>/dev/null

echo ""
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || { echo "── server log ──"; tail -20 "$TMP/server.log"; }
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
