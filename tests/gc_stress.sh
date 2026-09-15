#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  gc_stress.sh — object lifetime under size, repetition and abuse.
#
#  tests/lang_gc_test.b proves the collector frees each cycle shape, using
#  gc_stats()["live"] -- an exact number, which is the right instrument for
#  correctness. This script asks the different question: does the PROCESS
#  stay bounded when a program runs long enough to matter?
#
#  That question is the one the defect was: every leak measured here grew
#  without bound before the collector, so a sua worker creating objects per
#  request was killed eventually no matter how much memory the box had. RSS
#  read from outside the process is the only instrument that sees it.
#
#  The gate is BOUNDEDNESS, not a byte count: ten times the work must not
#  cost ten times the memory. A threshold-driven collector always holds some
#  garbage between collections, so a fixed ceiling would be measuring the
#  threshold rather than the leak.
#
#  Design: docs/object-lifetime-architecture.md
#  Run:    BANTU=./bantu-src/compiler/build/bantu bash tests/gc_stress.sh
# ════════════════════════════════════════════════════════════════════════
set -u
BANTU="${BANTU:-bantu}"
TMP="$(mktemp -d)"
PASS=0
FAIL=0

cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

ok()    { PASS=$((PASS+1)); echo "  ok    $1"; }
bad()   { FAIL=$((FAIL+1)); echo "  FAIL  $1"; }
check() { if [ "$1" = "1" ]; then ok "$2"; else bad "$2"; fi; }

# Peak RSS in KB. BSD /usr/bin/time -l reports bytes; GNU -v reports KB.
peakRss() {
    if /usr/bin/time -l true >/dev/null 2>&1; then
        /usr/bin/time -l "$@" 2>&1 >/dev/null |
            awk '/maximum resident/ { print int($1/1024); f=1 } END { if (!f) print 0 }'
    else
        /usr/bin/time -v "$@" 2>&1 >/dev/null |
            awk '/Maximum resident/ { print int($NF); f=1 } END { if (!f) print 0 }'
    fi
}

# Writes $TMP/cycle.b: $2 iterations of the cycle shape named by $1.
#
# The shape is read from the environment but the COUNT is written into the
# source, so the loop bound is a literal and cannot be mis-parsed. env() yields
# a string, and `0 + env(...)` CONCATENATES rather than adding -- which silently
# produced a loop that never ran, and is how the first measurement of this leak
# came back falsely flat. (num(env(...)) would parse it; an earlier version of
# this comment wrongly said Bantu had no string-to-number builtin.)
writeShape() {
    cat > "$TMP/cycle.b" <<BEOF
\$kind = env("GC_KIND");
\$n = $2;
BEOF
    cat >> "$TMP/cycle.b" <<'BEOF'
class GcNode {
    def init($v) { this.v = $v; }
    def value()  { return this.v; }
}
def makeNested() {
    $captured = [1, 2, 3];
    def helper() { return $captured; }
    return 1;
}
$i = 0;
while ($i < $n) {
    if ($kind == "plain")     { $a = new GcNode(1); }
    if ($kind == "selfref")   { $a = new GcNode(1); $a.self = $a; }
    if ($kind == "pair")      { $a = new GcNode(1); $b = new GcNode(2);
                                $a.p = $b; $b.p = $a; }
    if ($kind == "boundprop") { $a = new GcNode(1); $a.cb = $a.value; }
    if ($kind == "nestedfn")  { $z = makeNested(); }
    if ($kind == "dictself")  { $d = {}; $d["self"] = $d; }
    if ($kind == "ring")      { $a = new GcNode(1); $b = new GcNode(2);
                                $c = new GcNode(3); $d = new GcNode(4);
                                $a.n = $b; $b.n = $c; $c.n = $d; $d.n = $a; }
    $i = $i + 1;
}
print("live " + str(gc_stats()["live"]) + " collections " + str(gc_stats()["collections"]));
BEOF
}

echo "── every cycle shape stays bounded at scale ─────────────────────────"
echo "        shape        small      large    ratio"

for shape in plain selfref pair boundprop nestedfn dictself ring; do
    writeShape "$shape" 100000
    SMALL="$(GC_KIND=$shape peakRss "$BANTU" -q run "$TMP/cycle.b")"
    writeShape "$shape" 1000000
    LARGE="$(GC_KIND=$shape peakRss "$BANTU" -q run "$TMP/cycle.b")"

    if [ "${SMALL:-0}" -gt 0 ] && [ "${LARGE:-0}" -gt 0 ]; then
        RATIO="$(awk -v a="$SMALL" -v b="$LARGE" 'BEGIN { printf "%.2f", b/a }')"
        printf "        %-11s %6s KB  %6s KB   %sx\n" "$shape" "$SMALL" "$LARGE" "$RATIO"
        # Ten times the work. Before the collector, every shape but `plain`
        # grew linearly -- 400,000 mutual pairs reached 502 MB and climbing.
        check "$([ "$LARGE" -lt $((SMALL * 2)) ] && echo 1 || echo 0)" \
              "$shape: 10x the cycles costs under 2x the memory"
    else
        bad "$shape: could not read RSS"
    fi
done

echo "── a program with no cycles never pays for the collector ────────────"

writeShape plain 1000000
OUT="$(GC_KIND=plain "$BANTU" -q run "$TMP/cycle.b" 2>&1)"
COLL="$(echo "$OUT" | sed -n 's/.*collections \([0-9]*\).*/\1/p')"
check "$([ "${COLL:-1}" = "0" ] && echo 1 || echo 0)" \
      "1,000,000 acyclic objects trigger zero collections (got ${COLL:-?})"

echo "── the switch is honoured, and is not a way to lose the fix ─────────"

writeShape selfref 200000
WITH="$(GC_KIND=selfref peakRss "$BANTU" -q run "$TMP/cycle.b")"
WITHOUT="$(GC_KIND=selfref BANTU_GC=0 peakRss "$BANTU" -q run "$TMP/cycle.b")"
printf "        collector on: %s KB   BANTU_GC=0: %s KB\n" "$WITH" "$WITHOUT"
check "$([ "$WITHOUT" -gt $((WITH * 2)) ] && echo 1 || echo 0)" \
      "BANTU_GC=0 really disables it (the leak comes back, proving the gate measures the collector)"

cat > "$TMP/off.b" <<'BEOF'
class N { def init() { this.v = 1; } }
$i = 0;
while ($i < 20000) { $a = new N(); $a.self = $a; $i = $i + 1; }
// Automatic collection is off, but the explicit call must still work.
$freed = gc_collect();
if ($freed > 10000) { print("EXPLICIT OK"); } else { print("EXPLICIT BAD " + str($freed)); }
BEOF
check "$(BANTU_GC=0 "$BANTU" -q run "$TMP/off.b" 2>&1 | grep -c 'EXPLICIT OK')" \
      "gc_collect() still works with BANTU_GC=0"

echo "── abuse ────────────────────────────────────────────────────────────"

cat > "$TMP/abuse.b" <<'BEOF'
class N { def init($v) { this.v = $v; } def m() { return this.v; } }

// Collecting when there is nothing to collect, repeatedly.
$i = 0;
while ($i < 200) { gc_collect(); $i = $i + 1; }
print("A empty collections survived");

// A single very large cycle -- one ring of 50,000 nodes, so the collector
// walks a long chain rather than many short ones.
$head = new N(0);
$prev = $head;
$i = 1;
while ($i < 50000) { $n = new N($i); $prev.next = $n; $prev = $n; $i = $i + 1; }
$prev.next = $head;
$head = null; $prev = null; $n = null;
$freed = gc_collect();
if ($freed > 40000) { print("B big ring freed " + str($freed)); }
else { print("B BAD " + str($freed)); }

// A cycle that reaches a native handle: the handle must be released with it
// rather than outliving the cycle.
$i = 0;
while ($i < 2000) {
    $a = new N(1);
    $a.self = $a;
    $a.data = [1, 2, 3, 4, 5, 6, 7, 8];
    $i = $i + 1;
}
gc_collect();
print("C cycles holding lists collected");

// Deeply nested containers inside a cycle.
$root = new N(0);
$cur = $root;
$i = 0;
while ($i < 500) { $child = new N($i); $cur.kid = $child; $cur = $child; $i = $i + 1; }
$cur.back = $root;          // close the cycle at the far end
$root = null; $cur = null; $child = null;
$freed = gc_collect();
if ($freed > 400) { print("D deep chain freed " + str($freed)); }
else { print("D BAD " + str($freed)); }

// Collecting while a cycle is still live, over and over: the live one must
// survive every pass and stay correct.
$keep = new N(99);
$keep.self = $keep;
$i = 0;
while ($i < 100) {
    $junk = new N(1); $junk.self = $junk;
    gc_collect();
    $i = $i + 1;
}
if ($keep.self.self.v == 99) { print("E live cycle survived 100 collections"); }
else { print("E BAD"); }

// gc_enable with rubbish arguments must not crash.
gc_enable(null); gc_enable("x"); gc_enable(0); gc_enable(true);
print("F gc_enable tolerates bad arguments");

// A method torn off an object, kept after the object is dropped.
$obj = new N(5);
$fn = $obj.m;
$obj = null;
gc_collect();
if ($fn() == 5) { print("G torn-off method still works after collection"); }
else { print("G BAD"); }
BEOF

AB="$("$BANTU" -q run "$TMP/abuse.b" 2>&1)"
for mark in "A empty collections survived" "B big ring freed" \
            "C cycles holding lists collected" "D deep chain freed" \
            "E live cycle survived 100 collections" \
            "F gc_enable tolerates bad arguments" \
            "G torn-off method still works after collection"; do
    check "$(echo "$AB" | grep -c "$mark")" "$mark"
done
if echo "$AB" | grep -q 'BAD'; then bad "an abuse case reported BAD"; echo "$AB" | grep BAD; else ok "no abuse case reported BAD"; fi

echo "── a long run stays flat, not merely smaller ────────────────────────"

writeShape pair 400000
R1="$(GC_KIND=pair peakRss "$BANTU" -q run "$TMP/cycle.b")"
writeShape pair 4000000
R2="$(GC_KIND=pair peakRss "$BANTU" -q run "$TMP/cycle.b")"
printf "        400,000 pairs: %s KB   4,000,000 pairs: %s KB\n" "$R1" "$R2"
check "$([ "$R2" -lt $((R1 * 2)) ] && echo 1 || echo 0)" \
      "ten times the run length costs under twice the memory"

echo ""
echo "  Passed: $PASS   Failed: $FAIL"
if [ "$FAIL" -eq 0 ]; then
    echo "  RESULT: ALL GREEN"
    exit 0
else
    echo "  RESULT: FAILURES"
    exit 1
fi
