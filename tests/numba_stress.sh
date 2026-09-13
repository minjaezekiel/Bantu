#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  numba_stress.sh — size, repetition and abuse.
#
#  The feature suite (tests/numba_array_test.b) proves each builtin does what
#  it claims. This proves the layer survives being used hard, which is a
#  different question and the one that catches leaks:
#
#    1. 200,000 arrays created and dropped -> RSS must stay flat. Every array
#       is a shared_ptr<Buffer>, so a single missed decrement anywhere leaks
#       80 bytes a time and nothing fails -- the process just grows. No
#       assertion inside Bantu can see this; only RSS can.
#    2. 200,000 VIEWS created and dropped, where the base outlives them and
#       then does not. A view holds its base's buffer alive on purpose, so
#       this is where a reference cycle would show up.
#    3. 30,000 deliberately-bad calls -> every one raises catchably and the
#       process is still alive and correct afterwards. The sua.udp review is
#       the precedent: two of its four defects killed the process outright
#       because one unvalidated argument reached an allocation.
#    4. Allocation and view timing at 10M elements.
#
#  RSS is sampled from OUTSIDE the process, so the script runs in the
#  background and pauses at marks while the shell reads ps.
#
#  Run:  BANTU=./bantu-src/compiler/build/bantu bash tests/numba_stress.sh
# ════════════════════════════════════════════════════════════════════════
set -u
BANTU="${BANTU:-bantu}"
TMP="$(mktemp -d)"
PASS=0
FAIL=0

cleanup() { [ -n "${PID:-}" ] && kill "$PID" 2>/dev/null; rm -rf "$TMP"; }
trap cleanup EXIT

ok()   { PASS=$((PASS+1)); echo "  ok    $1"; }
bad()  { FAIL=$((FAIL+1)); echo "  FAIL  $1"; }
check(){ if [ "$1" = "1" ]; then ok "$2"; else bad "$2"; fi; }

rss() { ps -o rss= -p "$PID" | tr -d ' '; }

# Wait for the running script to write a mark file (it pauses there).
wait_mark() {
    local f="$1" i=0
    while [ ! -f "$f" ] && [ $i -lt 600 ]; do sleep 0.1; i=$((i+1)); done
    [ -f "$f" ]
}

cat > "$TMP/stress.b" <<BEOF
def mark(\$name) { writefile("$TMP/" + \$name, "1"); sleep(1200); }

// ── warm up: let the allocator reach a steady state first, so the number we
// compare against is not dominated by first-touch growth ──────────────────
\$i = 0;
while (\$i < 20000) {
    \$a = nd_zeros([10], null);
    \$i = \$i + 1;
}
mark("m1");

// ── 1. 200,000 arrays created and dropped ─────────────────────────────────
\$i = 0;
while (\$i < 200000) {
    \$a = nd_zeros([10], null);
    \$i = \$i + 1;
}
mark("m2");

// ── 2. 200,000 views, base dropped at the end of each iteration ───────────
\$i = 0;
while (\$i < 200000) {
    \$base = nd_arange(0, 64, null);
    \$v = nd_T(nd_reshape(\$base, [8, 8]));
    \$w = nd_slice(\$v, [[1, 5, null], null]);
    \$i = \$i + 1;
}
mark("m3");

// ── 3. 30,000 bad calls, every one caught ────────────────────────────────
\$raised = 0;
\$i = 0;
while (\$i < 30000) {
    try { nd_zeros([-1], null); }            catch (\$e) { \$raised = \$raised + 1; }
    try { nd_zeros([1000000000000000], null);} catch (\$e) { \$raised = \$raised + 1; }
    try { nd_get(nd_zeros([2], null), 99); } catch (\$e) { \$raised = \$raised + 1; }
    try { nd_reshape(nd_zeros([3], null), [4]); } catch (\$e) { \$raised = \$raised + 1; }
    try { nd([[1,2],[3]], null); }           catch (\$e) { \$raised = \$raised + 1; }
    try { nd_zeros([4194304,4194304,4194304], null); } catch (\$e) { \$raised = \$raised + 1; }
    \$i = \$i + 1;
}
writefile("$TMP/raised", str(\$raised));
mark("m4");

// ── 4. still correct after all that ──────────────────────────────────────
\$chk = nd_reshape(nd_arange(0, 12, null), [3, 4]);
nd_set(\$chk, [2, 3], 42);
writefile("$TMP/final", str(nd_get(\$chk, [2,3])) + "," + str(nd_size(\$chk)));

// ── 5. timing at 10M ─────────────────────────────────────────────────────
\$t0 = clock();
\$big = nd_zeros([10000000], null);
\$alloc = clock() - \$t0;
\$t0 = clock();
\$view = nd_T(nd_reshape(\$big, [1000, 10000]));
\$vms = clock() - \$t0;
\$t0 = clock();
\$s = nd_slice(\$big, [[0, 10000000, 2]]);
\$sms = clock() - \$t0;
writefile("$TMP/timing", str(\$alloc) + "," + str(\$vms) + "," + str(\$sms) + "," + str(nd_size(\$s)));
writefile("$TMP/done", "1");
BEOF

"$BANTU" -q run "$TMP/stress.b" > "$TMP/out.log" 2>&1 &
PID=$!

echo "-- 200,000 arrays created and dropped --"
wait_mark "$TMP/m1" || { bad "script did not reach the warm-up mark"; cat "$TMP/out.log"; exit 1; }
BEFORE=$(rss)
wait_mark "$TMP/m2" || { bad "script did not finish the allocation loop"; cat "$TMP/out.log"; exit 1; }
AFTER=$(rss)
GROWTH=$((AFTER - BEFORE))
echo "        RSS ${BEFORE}KB -> ${AFTER}KB over 200,000 arrays (${GROWTH}KB)"
check "$([ "$GROWTH" -lt 5120 ] && echo 1 || echo 0)" \
      "RSS grew less than 5MB over 200,000 arrays (a leak would be ~16MB+)"

echo ""
echo "-- 200,000 views, each holding a base alive --"
wait_mark "$TMP/m3" || { bad "script did not finish the view loop"; cat "$TMP/out.log"; exit 1; }
AFTER2=$(rss)
GROWTH2=$((AFTER2 - AFTER))
echo "        RSS ${AFTER}KB -> ${AFTER2}KB over 200,000 view chains (${GROWTH2}KB)"
check "$([ "$GROWTH2" -lt 5120 ] && echo 1 || echo 0)" \
      "a view keeps its base alive without leaking it"

echo ""
echo "-- 180,000 deliberately-bad calls --"
wait_mark "$TMP/m4" || { bad "the process did not survive the bad-argument loop"; cat "$TMP/out.log"; exit 1; }
RAISED=$(cat "$TMP/raised" 2>/dev/null || echo 0)
AFTER3=$(rss)
GROWTH3=$((AFTER3 - AFTER2))
echo "        ${RAISED} errors raised and caught; RSS ${AFTER2}KB -> ${AFTER3}KB (${GROWTH3}KB)"
check "$([ "$RAISED" = "180000" ] && echo 1 || echo 0)" \
      "every one of 180,000 bad calls raised a catchable error"
check "$([ "$GROWTH3" -lt 5120 ] && echo 1 || echo 0)" \
      "and the failed allocations leaked nothing"

echo ""
echo "-- still correct, and still fast --"
i=0
while [ ! -f "$TMP/done" ] && [ $i -lt 900 ]; do sleep 0.1; i=$((i+1)); done
if [ ! -f "$TMP/done" ]; then bad "the script did not finish"; cat "$TMP/out.log"; exit 1; fi

FINAL=$(cat "$TMP/final")
check "$([ "$FINAL" = "42,12" ] && echo 1 || echo 0)" \
      "arrays still behave correctly after the whole run (got $FINAL)"

IFS=, read -r ALLOC VMS SMS SSIZE < "$TMP/timing"
echo "        10M f64: allocate ${ALLOC}ms, reshape+transpose ${VMS}ms, strided slice ${SMS}ms"
check "$([ "$ALLOC" -lt 300 ] && echo 1 || echo 0)" "10M-element allocation under 300ms"
check "$([ "$VMS" -lt 50 ] && echo 1 || echo 0)"    "a view of 10M elements is free (no copy)"
check "$([ "$SMS" -lt 50 ] && echo 1 || echo 0)"    "a strided slice of 10M elements is free too"
check "$([ "$SSIZE" = "5000000" ] && echo 1 || echo 0)" "and the slice has the right length"

wait "$PID" 2>/dev/null
echo ""
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || { echo "── script log ──"; tail -20 "$TMP/out.log"; }
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
