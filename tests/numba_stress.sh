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

// ── 4b. byte accounting is EXACT, which RSS can never show ───────────────
// RSS is noisy: the allocator keeps freed pages, so a small genuine leak hides
// inside normal variation. nd_live_bytes() is the number the ceiling is
// actually tested against, so it must return to its baseline exactly -- if it
// drifts upward, numba slowly refuses to allocate and nothing says why.
\$baseline = nd_live_bytes();
\$i = 0;
while (\$i < 50000) {
    \$t = nd_zeros([64], null);
    \$u = nd_T(nd_reshape(\$t, [8, 8]));
    \$i = \$i + 1;
}
// The last iteration's array is still bound to \$t/\$u, and it is accounted
// because it is genuinely still live. Release it, or the check measures scope
// rather than leakage.
\$t = 0;
\$u = 0;
\$afterLoop = nd_live_bytes() - \$baseline;

// Failed allocations must return their accounted bytes on the way out.
\$i = 0;
while (\$i < 20000) {
    try { nd_zeros([1000000000000000], null); } catch (\$e) { }
    \$i = \$i + 1;
}
\$afterFail = nd_live_bytes() - \$baseline;

// The ceiling must bound a hostile LOOP, not just one call. This is the case a
// per-allocation limit misses entirely, and it is how a request handler
// actually exhausts a server.
\$prevCap = nd_max_bytes(null);
nd_max_bytes(104857600);                       // 100 MB
\$hoard = [];
\$i = 0;
while (\$i < 200) {
    try { \$hoard[len(\$hoard)] = nd_zeros([1250000], null); } catch (\$e) { }  // 10 MB
    \$i = \$i + 1;
}
\$heldMB = len(\$hoard) * 10;
\$hoard = [];
nd_max_bytes(\$prevCap);
\$afterHoard = nd_live_bytes() - \$baseline;
writefile("$TMP/account", str(\$afterLoop) + "," + str(\$afterFail) + "," +
                          str(\$heldMB) + "," + str(\$afterHoard));

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

// ── 6. ufuncs at scale, and the out= aliasing gate under repetition ──────
\$u = nd_random_uniform([10000000], 0, 1);
\$v = nd_random_uniform([10000000], 0, 1);
\$w = nd_empty([10000000], null);
nd_add(\$u, \$v, \$w);                       // warm
\$t0 = clock();
nd_add(\$u, \$v, \$w);
\$addMs = clock() - \$t0;
\$t0 = clock();
nd_sqrt(\$u, \$w);
\$sqrtMs = clock() - \$t0;

// nd_empty must NOT pay for touching pages it is about to overwrite (N20).
\$t0 = clock();
\$e = nd_empty([10000000], null);
\$emptyMs = clock() - \$t0;
\$t0 = clock();
\$z = nd_zeros([10000000], null);
\$zeroMs = clock() - \$t0;
\$e = 0; \$z = 0;

// 100,000 ufunc calls with out= must not leak: each one runs the aliasing
// gate, and the exact-overlap path must NOT quietly allocate a temporary.
\$small = nd_random_uniform([64], 0, 1);
\$acc   = nd_zeros([64], null);
// Baseline AFTER the operands exist: they are legitimately live, so counting
// them as drift would measure scope rather than leakage.
\$baseU = nd_live_bytes();
\$i = 0;
while (\$i < 100000) {
    nd_add(\$acc, \$small, \$acc);          // the exact-overlap in-place idiom
    \$i = \$i + 1;
}
\$driftU = nd_live_bytes() - \$baseU;

// Every shape-mismatched call must raise rather than corrupt, 60,000 times.
\$badRaised = 0;
\$i = 0;
while (\$i < 20000) {
    try { nd_add(nd_zeros([3], null), nd_zeros([4], null), null); } catch (\$e2) { \$badRaised = \$badRaised + 1; }
    try { nd_add(\$small, \$small, nd_broadcast_to(nd([0], null), [64])); } catch (\$e2) { \$badRaised = \$badRaised + 1; }
    try { nd_add(\$small, \$small, nd_zeros([9], null)); } catch (\$e2) { \$badRaised = \$badRaised + 1; }
    \$i = \$i + 1;
}
\$driftB = nd_live_bytes() - \$baseU;
writefile("$TMP/ufunc", str(\$addMs) + "," + str(\$sqrtMs) + "," + str(\$driftU) + "," +
                        str(\$badRaised) + "," + str(\$driftB) + "," +
                        str(\$emptyMs) + "," + str(\$zeroMs));

// ── 7. reductions, scans, sorting at scale ──────────────────────────────
\$t0 = clock();
\$sm = nd_get(nd_sum(\$u, null, null), []);
\$sumMs = clock() - \$t0;

// Accuracy at scale, which is the whole reason for pairwise accumulation:
// a naive accumulator loses about n*eps, ~2e-12 at ten million elements.
\$tenth = nd_full([10000000], 0.1, null);
\$exact = nd_get(nd_sum(\$tenth, null, null), []);
\$relerr = (\$exact - 1000000.0) / 1000000.0;
if (\$relerr < 0) { \$relerr = 0 - \$relerr; }
\$tenth = 0;

\$mid = nd_random_uniform([1000000], 0, 1);
\$t0 = clock();
\$as = nd_argsort(\$mid, null);
\$argMs = clock() - \$t0;
// The permutation must actually sort: applying it reproduces nd_sort's output.
\$sortOk = nd_array_equal(nd_take(\$mid, \$as, null), nd_sort(\$mid, null));

// Degenerate inputs at scale: already sorted, reverse sorted, all equal.
\$asc  = nd_arange(0, 1000000, null);
\$desc = nd_flip(\$asc, null);
\$same = nd_full([1000000], 7, null);
\$degenOk = 0;
if (nd_array_equal(nd_sort(\$asc, null), \$asc))            { \$degenOk = \$degenOk + 1; }
if (nd_array_equal(nd_sort(\$desc, null), \$asc))           { \$degenOk = \$degenOk + 1; }
if (nd_array_equal(nd_sort(\$same, null), \$same))          { \$degenOk = \$degenOk + 1; }
\$asc = 0; \$desc = 0; \$same = 0; \$as = 0; \$mid = 0;

// 40,000 bad reduction and indexing calls: every one raises, nothing leaks.
\$small2 = nd_arange(0, 16, null);
// Baseline AFTER every operand exists. Anything allocated between the baseline
// and the measurement is legitimately live and would read as drift -- this is
// the third time that has caught a check in this file.
\$baseR = nd_live_bytes();
\$redBad = 0;
\$i = 0;
while (\$i < 10000) {
    try { nd_min(nd_zeros([0], null), null, null); } catch (\$e3) { \$redBad = \$redBad + 1; }
    try { nd_sum(\$small2, 5, null); }               catch (\$e3) { \$redBad = \$redBad + 1; }
    try { nd_take(\$small2, nd([99], null), null); } catch (\$e3) { \$redBad = \$redBad + 1; }
    try { nd_compress(nd_zeros([3], null), \$small2); } catch (\$e3) { \$redBad = \$redBad + 1; }
    \$i = \$i + 1;
}
\$driftR = nd_live_bytes() - \$baseR;
writefile("$TMP/reduce", str(\$sumMs) + "," + str(\$relerr) + "," + str(\$argMs) + "," +
                         str(\$sortOk) + "," + str(\$degenOk) + "," + str(\$redBad) + "," +
                         str(\$driftR));
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

IFS=, read -r ACCT_LOOP ACCT_FAIL HELD_MB ACCT_HOARD < "$TMP/account"
echo "        live-byte drift: ${ACCT_LOOP}B over 50,000 arrays+views, ${ACCT_FAIL}B over 20,000 failures"
check "$([ "$ACCT_LOOP" = "0" ] && echo 1 || echo 0)" \
      "byte accounting returns to its baseline EXACTLY after 50,000 arrays"
check "$([ "$ACCT_FAIL" = "0" ] && echo 1 || echo 0)" \
      "and after 20,000 refused allocations (a drift here slowly bricks numba)"
echo "        a hoarding loop under a 100MB ceiling held ${HELD_MB}MB"
check "$([ "$HELD_MB" -le 100 ] && echo 1 || echo 0)" \
      "the ceiling bounds a LOOP in aggregate, not just one allocation"
check "$([ "$ACCT_HOARD" = "0" ] && echo 1 || echo 0)" \
      "and every hoarded byte comes back when the arrays are dropped"

IFS=, read -r ALLOC VMS SMS SSIZE < "$TMP/timing"
echo "        10M f64: allocate ${ALLOC}ms, reshape+transpose ${VMS}ms, strided slice ${SMS}ms"
check "$([ "$ALLOC" -lt 300 ] && echo 1 || echo 0)" "10M-element allocation under 300ms"
check "$([ "$VMS" -lt 50 ] && echo 1 || echo 0)"    "a view of 10M elements is free (no copy)"
check "$([ "$SMS" -lt 50 ] && echo 1 || echo 0)"    "a strided slice of 10M elements is free too"
check "$([ "$SSIZE" = "5000000" ] && echo 1 || echo 0)" "and the slice has the right length"

echo ""
echo "-- ufuncs at scale --"
IFS=, read -r ADDMS SQRTMS DRIFTU BADRAISED DRIFTB EMPTYMS ZEROMS < "$TMP/ufunc"
# 10M f64 add moves 3 x 80MB. Reported as GB/s so the number is comparable
# across machines; the threshold is deliberately per-platform (N22), because a
# single core sustains only ~10 concurrent L1 misses and cannot saturate DRAM.
echo "        10M nd_add ${ADDMS}ms, nd_sqrt ${SQRTMS}ms"
check "$([ "$ADDMS" -lt 120 ] && echo 1 || echo 0)" \
      "10M nd_add under 120ms (>= 2 GB/s effective even on a slow shared vCPU)"
check "$([ "$SQRTMS" -lt 200 ] && echo 1 || echo 0)" "10M nd_sqrt under 200ms"
echo "        nd_empty ${EMPTYMS}ms vs nd_zeros ${ZEROMS}ms for 10M f64"
check "$([ "$EMPTYMS" -le "$ZEROMS" ] && echo 1 || echo 0)" \
      "nd_empty does not pay to touch pages it is about to overwrite (N20)"
echo "        100,000 in-place nd_add(a,b,a): live-byte drift ${DRIFTU}B"
check "$([ "$DRIFTU" = "0" ] && echo 1 || echo 0)" \
      "the exact-overlap out= path allocates NO temporary, 100,000 times"
echo "        ${BADRAISED} rejected ufunc calls; drift ${DRIFTB}B"
check "$([ "$BADRAISED" = "60000" ] && echo 1 || echo 0)" \
      "every one of 60,000 bad ufunc calls raised (shape, read-only out, wrong-size out)"
check "$([ "$DRIFTB" = "0" ] && echo 1 || echo 0)" \
      "and none of them leaked a partially-built result"

echo ""
echo "-- reductions, scans and sorting at scale --"
IFS=, read -r SUMMS RELERR ARGMS SORTOK DEGENOK REDBAD DRIFTR < "$TMP/reduce"
echo "        10M nd_sum ${SUMMS}ms; 1M nd_argsort ${ARGMS}ms"
check "$([ "$SUMMS" -lt 150 ] && echo 1 || echo 0)" "10M nd_sum under 150ms"
check "$([ "$ARGMS" -lt 2000 ] && echo 1 || echo 0)" "1M nd_argsort under 2000ms"
echo "        10M x 0.1 relative error: ${RELERR}"
# A naive accumulator lands near 2e-12 here, so this threshold is a real gate.
check "$(awk -v e="$RELERR" 'BEGIN{print (e < 1e-12) ? 1 : 0}')" \
      "pairwise summation holds 10M x 0.1 under 1e-12 (naive fails by construction)"
check "$([ "$SORTOK" = "true" ] && echo 1 || echo 0)" \
      "argsort's permutation on 1M elements reproduces sort's output exactly"
check "$([ "$DEGENOK" = "3" ] && echo 1 || echo 0)" \
      "1M already-sorted, reverse-sorted and all-equal inputs all sort correctly"
echo "        ${REDBAD} rejected reduction/indexing calls; drift ${DRIFTR}B"
check "$([ "$REDBAD" = "40000" ] && echo 1 || echo 0)" \
      "every one of 40,000 bad reduction and indexing calls raised"
check "$([ "$DRIFTR" = "0" ] && echo 1 || echo 0)" "and none of them leaked"

wait "$PID" 2>/dev/null
echo ""
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || { echo "── script log ──"; tail -20 "$TMP/out.log"; }
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
