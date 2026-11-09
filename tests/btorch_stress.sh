#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  btorch_stress.sh — size, repetition and memory.
#
#  tests/btorch_autograd_test.b proves every gradient is right. This proves
#  the engine survives being used the way training actually uses it, which is
#  a different question and the one that catches leaks.
#
#  The autograd tape is the part of btorch most likely to leak, because every
#  op allocates a Tensor, a Node and a closure that captures saved arrays. If
#  backward() fails to drop those, a training loop grows without bound and
#  NOTHING fails -- the numbers stay right and the process dies after an hour.
#  No assertion inside Bantu can see that; only RSS can, sampled from outside.
#
#    1. 10,000 full forward+backward+update steps -> RSS flat after warm-up.
#       This is the gate from the architecture doc section 9.
#    2. 20,000 tensors created and dropped -> RSS flat.
#    3. A 5,000-node graph: built, backwarded, freed, repeatedly.
#    4. retain_graph NOT set must actually release: 2,000 graphs in sequence.
#    5. Deliberately-bad calls all raise catchably, process correct afterwards.
#    6. Step timing, for the autograd-overhead budget.
#
#  Run:  BANTU=./bantu-src/compiler/build/bantu bash tests/btorch_stress.sh
# ════════════════════════════════════════════════════════════════════════
set -u
BANTU="${BANTU:-bantu}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
TMP="$(mktemp -d)"
PASS=0
FAIL=0

cleanup() { [ -n "${PID:-}" ] && kill "$PID" 2>/dev/null; rm -rf "$TMP"; }
trap cleanup EXIT

ok()   { PASS=$((PASS+1)); echo "  ok    $1"; }
bad()  { FAIL=$((FAIL+1)); echo "  FAIL  $1"; }
check(){ if [ "$1" = "1" ]; then ok "$2"; else bad "$2"; fi; }

rss() { ps -o rss= -p "$PID" | tr -d ' '; }

wait_mark() {
    local f="$1" i=0
    while [ ! -f "$f" ] && [ $i -lt 1200 ]; do sleep 0.1; i=$((i+1)); done
    [ -f "$f" ]
}

cat > "$TMP/stress.b" <<BEOF
include "$ROOT/btorch/btorch.b" as bt;

def mark(\$name) { writefile("$TMP/" + \$name, "1"); sleep(1200); }

bt.manual_seed(7);
mark("ready");

// ── 1. a real training loop: 10,000 steps, one linear layer ────────────
// forward, backward, and an SGD update under no_grad. If the tape is not
// freed, this is where it shows.
\$W = bt.randn([8, 4], true);
\$b = bt.zeros([4], true);
\$x = bt.randn([16, 8], false);
\$y = bt.randn([16, 4], false);

def step(\$W, \$b, \$x, \$y) {
    \$W.zero_grad();
    \$b.zero_grad();
    \$pred = \$x.matmul(\$W).add(\$b).tanh();
    \$diff = \$pred.sub(\$y);
    \$loss = \$diff.mul(\$diff).mean(null, null);
    \$loss.backward(null);
    bt.no_grad(def() {
        nd_subtract(\$W.data, nd_multiply(\$W.grad, 0.01, null), \$W.data);
        nd_subtract(\$b.data, nd_multiply(\$b.grad, 0.01, null), \$b.data);
        return null;
    });
    return \$loss.item();
}

// warm up so the first-touch page faults are not counted as growth
\$i = 0;
while (\$i < 200) { \$l = step(\$W, \$b, \$x, \$y); \$i = \$i + 1; }
mark("warm");

\$t0 = clock();
\$i = 0;
while (\$i < 10000) { \$l = step(\$W, \$b, \$x, \$y); \$i = \$i + 1; }
\$t1 = clock();
writefile("$TMP/steptime", str((\$t1 - \$t0) / 10000));
writefile("$TMP/finalloss", str(\$l));
mark("trained");

// ── 2. 20,000 tensors created and dropped ──────────────────────────────
\$i = 0;
while (\$i < 20000) { \$t = bt.randn([32], true); \$i = \$i + 1; }
mark("tensors");

// ── 3. a 5,000-node graph ──────────────────────────────────────────────
// A graph this size is LEGITIMATELY large while it is alive: 5,000 Nodes,
// ~15,000 Tensors and 5,000 closures over saved arrays, all live at once.
// So peak RSS here is not interesting. The question is whether it is RELEASED,
// and the way to ask that is to build one, measure, then build twenty more and
// measure again. A leak grows with every round; a high-water mark does not.
def deepRound() {
    \$leaf = bt.tensor([0.0], true);
    \$acc = \$leaf;
    \$j = 0;
    while (\$j < 5000) { \$acc = \$acc.add(bt.tensor([1.0], false)); \$j = \$j + 1; }
    \$acc.backward(null);
    return nd_get(\$leaf.grad, [0]);
}
\$dg = deepRound();
mark("deep1");

\$i = 0;
while (\$i < 20) { \$dg = deepRound(); \$i = \$i + 1; }
mark("deep");

// A third point, because two cannot tell a slow leak from an allocator
// high-water mark settling. A leak adds the SAME amount every round; drift
// tails off. Rounds 22-41 are compared against rounds 2-21.
\$i = 0;
while (\$i < 20) { \$dg = deepRound(); \$i = \$i + 1; }
writefile("$TMP/deepgrad", str(\$dg));
mark("deep3");

// ── 4. 2,000 graphs in sequence, none retained ─────────────────────────
\$i = 0;
while (\$i < 2000) {
    \$a = bt.randn([64], true);
    \$o = \$a.mul(\$a).tanh().exp().sum(null, null);
    \$o.backward(null);
    \$i = \$i + 1;
}
mark("graphs");

// ── 5. abuse: every bad call must raise catchably ──────────────────────
\$raised = 0;
\$i = 0;
while (\$i < 5000) {
    try { \$bad = bt.tensor([[1.0, 2.0]], true); \$bad.backward(null); }
    catch (\$e) { \$raised = \$raised + 1; }
    try { \$n = bt.tensor([1.0], false); \$n.backward(null); }
    catch (\$e) { \$raised = \$raised + 1; }
    try { \$m1 = bt.randn([3, 4], true); \$m2 = bt.randn([5, 6], true); \$m1.matmul(\$m2); }
    catch (\$e) { \$raised = \$raised + 1; }
    try { \$t = bt.tensor([1.0, 2.0], false); \$t.item(); }
    catch (\$e) { \$raised = \$raised + 1; }
    \$i = \$i + 1;
}
writefile("$TMP/raised", str(\$raised));
// and the process is still correct afterwards
\$sane = bt.tensor([3.0], true);
\$sane.mul(\$sane).backward(null);
writefile("$TMP/sane", str(nd_get(\$sane.grad, [0])));
mark("abused");
BEOF

echo "════════════════════════════════════════════════════════════════"
echo "  btorch stress — memory, repetition, abuse"
echo "════════════════════════════════════════════════════════════════"

"$BANTU" --quiet run "$TMP/stress.b" > "$TMP/out.txt" 2>&1 &
PID=$!

wait_mark "$TMP/ready" || { bad "the script never started"; echo "$(cat "$TMP/out.txt")"; exit 1; }
R_START=$(rss); kill -CONT $PID 2>/dev/null
echo "  RSS at start:        ${R_START} KB"

wait_mark "$TMP/warm"    || bad "warm-up never finished"
R_WARM=$(rss)
echo "  RSS after warm-up:   ${R_WARM} KB  (200 steps)"

wait_mark "$TMP/trained" || bad "the 10,000-step run never finished"
R_TRAIN=$(rss)
STEP=$(cat "$TMP/steptime" 2>/dev/null || echo "?")
echo "  RSS after 10k steps: ${R_TRAIN} KB  (${STEP} ms/step)"

wait_mark "$TMP/tensors" || bad "the tensor churn never finished"
R_TENS=$(rss)
echo "  RSS after 20k tensors: ${R_TENS} KB"

wait_mark "$TMP/deep1"   || bad "the first deep graph never finished"
R_DEEP1=$(rss)
echo "  RSS after ONE 5,000-node graph: ${R_DEEP1} KB  (peak: it is legitimately live)"

wait_mark "$TMP/deep"    || bad "the deep-graph run never finished"
R_DEEP=$(rss)
echo "  RSS after rounds 2-21:          ${R_DEEP} KB"

wait_mark "$TMP/deep3"   || bad "the third deep-graph run never finished"
R_DEEP3=$(rss)
echo "  RSS after rounds 22-41:         ${R_DEEP3} KB"

wait_mark "$TMP/graphs"  || bad "the 2000-graph run never finished"
R_GRAPH=$(rss)
echo "  RSS after 2000 graphs: ${R_GRAPH} KB"

wait_mark "$TMP/abused"  || bad "the abuse run never finished"
R_ABUSE=$(rss)
echo "  RSS after 20k bad calls: ${R_ABUSE} KB"
echo ""

# ── the gates ───────────────────────────────────────────────────────────
# 10,000 training steps after a 200-step warm-up must not grow the process.
# A tape that is not freed leaks a Tensor + Node + closure per op per step;
# at ~6 ops a step that is 60,000 live objects and would be unmistakable.
GROWTH=$(( R_TRAIN - R_WARM ))
echo "  growth across 10,000 training steps: ${GROWTH} KB"
check "$([ "$GROWTH" -lt 4096 ] && echo 1 || echo 0)" \
      "10,000 forward+backward+update steps grow RSS by < 4 MB (got ${GROWTH} KB)"

TGROWTH=$(( R_TENS - R_TRAIN ))
check "$([ "$TGROWTH" -lt 4096 ] && echo 1 || echo 0)" \
      "20,000 tensors created and dropped grow RSS by < 4 MB (got ${TGROWTH} KB)"

# The real question: does a 5,000-node graph get RELEASED after backward?
# Peak RSS for one such graph is expected to be tens of MB -- 5,000 Nodes,
# ~15,000 Tensors and 5,000 saved-array closures are all genuinely alive at
# once. What must NOT happen is that the next twenty rounds add to it. A leak
# would grow by the same amount every round; reuse of a high-water mark will
# not move at all.
DGROWTH=$(( R_DEEP - R_DEEP1 ))
DGROWTH2=$(( R_DEEP3 - R_DEEP ))
echo "  growth over rounds 2-21:  ${DGROWTH} KB"
echo "  growth over rounds 22-41: ${DGROWTH2} KB"
# Each round allocates ~20 MB of live graph. A leak of even a fraction of that
# would show as tens of MB per 20 rounds, and would NOT tail off. Two gates:
check "$([ "$DGROWTH" -lt 8192 ] && echo 1 || echo 0)" \
      "rounds 2-21 add < 8 MB, not the ~400 MB a per-round leak would (got ${DGROWTH} KB)"
# The decisive one: the second batch of 20 must not add as much as the first.
# Equal growth means a leak; tailing off means the allocator's high-water mark.
check "$([ "$DGROWTH2" -lt "$DGROWTH" ] || [ "$DGROWTH2" -lt 512 ] && echo 1 || echo 0)" \
      "rounds 22-41 add LESS than rounds 2-21 (${DGROWTH2} < ${DGROWTH} KB) -- drift, not a leak"

GGROWTH=$(( R_GRAPH - R_DEEP ))
check "$([ "$GGROWTH" -lt 4096 ] && echo 1 || echo 0)" \
      "2,000 unretained graphs grow RSS by < 4 MB (got ${GGROWTH} KB)"

AGROWTH=$(( R_ABUSE - R_GRAPH ))
check "$([ "$AGROWTH" -lt 4096 ] && echo 1 || echo 0)" \
      "20,000 raising calls grow RSS by < 4 MB (got ${AGROWTH} KB)"

# correctness survived all of it
RAISED=$(cat "$TMP/raised" 2>/dev/null || echo 0)
check "$([ "$RAISED" = "20000" ] && echo 1 || echo 0)" \
      "all 20,000 deliberately-bad calls raised catchably (got ${RAISED})"
SANE=$(cat "$TMP/sane" 2>/dev/null || echo "")
check "$([ "$SANE" = "6" ] && echo 1 || echo 0)" \
      "the engine is still correct after the abuse (d(x^2)/dx at 3 = 6, got ${SANE})"
DEEPG=$(cat "$TMP/deepgrad" 2>/dev/null || echo "")
check "$([ "$DEEPG" = "1" ] && echo 1 || echo 0)" \
      "a 5,000-node chain's gradient is 1 (got ${DEEPG})"

# the process must still be alive
check "$(kill -0 $PID 2>/dev/null && echo 1 || echo 0)" \
      "the process survived the whole run"

echo ""
echo "  (step time ${STEP} ms for an 8->4 layer at batch 16; final loss $(cat "$TMP/finalloss" 2>/dev/null))"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  PASS: $PASS   FAIL: $FAIL"
echo "════════════════════════════════════════════════════════════════"
if [ "$FAIL" -gt 0 ]; then echo "  RESULT: FAILURES PRESENT"; exit 1; fi
echo "  RESULT: ALL GREEN"
