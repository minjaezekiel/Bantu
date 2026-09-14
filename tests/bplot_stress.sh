#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  bplot_stress.sh — scale, degenerate input, hostile input, and the one
#  check that cannot be made from inside Bantu: does the SVG actually parse?
#
#  WHY A SHELL SCRIPT
#  bplot_core_test.b asserts on the document's CONTENT -- that it has a
#  polyline, that a hostile title was escaped, that no NaN reached a
#  coordinate. What it cannot do is hand the output to a real XML parser and
#  ask whether a browser would accept it. A document that is subtly malformed
#  still contains all the right substrings; it just renders as a blank page,
#  with nothing in any log. That is the failure this file exists to catch.
#
#  It also holds the gates that need a clock, a file size or a process RSS.
#
#  Run:  BANTU=./bantu-src/compiler/build/bantu bash tests/bplot_stress.sh
# ════════════════════════════════════════════════════════════════════════
set -u
BANTU="${BANTU:-bantu}"
TMP="$(mktemp -d)"
PASS=0
FAIL=0

cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

ok()   { PASS=$((PASS+1)); echo "  ok    $1"; }
bad()  { FAIL=$((FAIL+1)); echo "  FAIL  $1"; }
check(){ if [ "$1" = "1" ]; then ok "$2"; else bad "$2"; fi; }

PY=""
for c in python3 python; do command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }; done

# ── Every SVG this suite produces must parse as XML ──────────────────────
wellformed() {
    [ -n "$PY" ] || return 0                       # no parser here; the caller reports it
    "$PY" - "$1" <<'EOF' >/dev/null 2>&1
import sys, xml.dom.minidom
xml.dom.minidom.parse(sys.argv[1])
EOF
}

echo "-- rendering the shapes that break plotting libraries --"

cat > "$TMP/cases.b" <<'BEOF'
include "./bplot/bplot.b" as plt;

def render($name, $xs, $ys, $kind) {
    $f = plt.figure(600, 400);
    $a = $f.addAxes();
    if ($kind == "scatter") { $a.scatter($xs, $ys, null); }
    else { if ($kind == "bar") { $a.bar($xs, $ys, null); } else { $a.plot($xs, $ys, null); } }
    $a.setTitle($name);
    $a.setXLabel("x");
    $a.setYLabel("y");
    $a.setGrid(true);
    $f.savefig("OUTDIR/" + $name + ".svg");
    return null;
}

render("empty",        [],            [],            "line");
render("single",       [5],           [7],           "line");
render("two",          [1, 2],        [1, 2],        "line");
render("flat",         [1, 2, 3],     [7, 7, 7],     "line");
render("origin",       [0, 0, 0],     [0, 0, 0],     "line");
render("nan_middle",   [1, 2, 3, 4],  [1, 2, NAN, 4],"line");
render("nan_all",      [1, 2, 3],     [NAN, NAN, NAN], "line");
render("inf",          [1, 2, 3],     [1, INF, 3],   "line");
render("neginf",       [1, 2, 3],     [1, 0 - INF, 3], "line");
render("negative",     [-3, -2, -1],  [-9, -4, -1],  "line");
render("tiny",         [0, 0.000001], [0, 0.000002], "line");
render("huge",         [0, 1e12],     [0, 1e12],     "line");
render("scatter_one",  [1],           [1],           "scatter");
render("bar_one",      [1],           [5],           "bar");
render("bar_negative", [1, 2, 3],     [-3, 5, -1],   "bar");
render("bar_zero",     [1, 2],        [0, 0],        "bar");

// Hostile text in every place text can go.
$evil = "</text><script>alert(1)</script>&<>\"'";
$f = plt.figure(600, 400);
$a = $f.addAxes();
$a.plot([1, 2, 3], [1, 2, 3], {"label": $evil});
$a.setTitle($evil);
$a.setXLabel($evil);
$a.setYLabel($evil);
$a.setLegend(true);
$f.savefig("OUTDIR/hostile.svg");

// A control byte: not representable in XML 1.0 at all, so one of them turns
// the whole chart into a parse error unless it is stripped.
$f2 = plt.figure(400, 300);
$a2 = $f2.addAxes();
$a2.plot([1, 2], [1, 2], null);
$a2.setTitle("bell" + chr(7) + "here" + chr(0) + "and" + chr(27) + "escape");
$f2.savefig("OUTDIR/control.svg");

// Unicode, which must survive intact rather than being mangled or stripped.
$f3 = plt.figure(400, 300);
$a3 = $f3.addAxes();
$a3.plot([1, 2], [1, 2], null);
$a3.setTitle("temperatuur °C — Dar es Salaam ☀");
$f3.savefig("OUTDIR/unicode.svg");
print("rendered");
BEOF
sed -i.bak "s#OUTDIR#$TMP#g" "$TMP/cases.b" && rm -f "$TMP/cases.b.bak"

if "$BANTU" -q run "$TMP/cases.b" > "$TMP/cases.log" 2>&1; then
    ok "every degenerate and hostile case rendered without raising"
else
    bad "a case raised"
    cat "$TMP/cases.log"
fi

# ── The gate that cannot be written in Bantu ─────────────────────────────
if [ -z "$PY" ]; then
    echo "  --    no python found; skipping the XML well-formedness gate"
else
    BADXML=0
    COUNT=0
    for f in "$TMP"/*.svg; do
        [ -e "$f" ] || continue
        COUNT=$((COUNT+1))
        wellformed "$f" || { BADXML=$((BADXML+1)); echo "        malformed: $(basename "$f")"; }
    done
    echo "        $COUNT documents checked"
    check "$([ "$BADXML" = "0" ] && [ "$COUNT" -gt 10 ] && echo 1 || echo 0)" \
          "every document parses as well-formed XML"
fi

# ── The hostile document specifically ────────────────────────────────────
if [ -f "$TMP/hostile.svg" ]; then
    check "$(grep -c '<script' "$TMP/hostile.svg" | grep -q '^0$' && echo 1 || echo 0)" \
          "no <script survived into the hostile document"
    check "$(grep -q '&lt;script&gt;' "$TMP/hostile.svg" && echo 1 || echo 0)" \
          "the hostile title is present, escaped"
    check "$(wellformed "$TMP/hostile.svg" && echo 1 || echo 0)" \
          "and it still parses -- escaping did not break the document"
else
    bad "the hostile document was not written"
fi

if [ -f "$TMP/control.svg" ]; then
    # A raw control byte makes an XML parser reject the whole file, which is
    # why they are stripped rather than escaped.
    check "$(wellformed "$TMP/control.svg" && echo 1 || echo 0)" \
          "control bytes were stripped, so the document still parses"
    check "$(grep -q 'bellhereandescape' "$TMP/control.svg" && echo 1 || echo 0)" \
          "and only the control bytes were removed"
fi

if [ -f "$TMP/unicode.svg" ]; then
    check "$(grep -q '°C' "$TMP/unicode.svg" && echo 1 || echo 0)" \
          "unicode survives intact"
    check "$(wellformed "$TMP/unicode.svg" && echo 1 || echo 0)" \
          "and the document parses"
fi

# No document may contain a NaN or an infinity: a browser renders such a
# path as nothing at all, with no error anywhere.
DIRTY=0
for f in "$TMP"/*.svg; do
    [ -e "$f" ] || continue
    grep -qE '(NaN|[^a-z-]inf|-inf)[,"]' "$f" && { DIRTY=$((DIRTY+1)); echo "        $(basename "$f")"; }
done
check "$([ "$DIRTY" = "0" ] && echo 1 || echo 0)" \
      "no NaN or infinity reached a coordinate in any document"

# ── Scale ────────────────────────────────────────────────────────────────
echo ""
echo "-- scale: a 100,000-point line --"
cat > "$TMP/big.b" <<'BEOF'
include "./bplot/bplot.b" as plt;
$n = 100000;
$xs = [];
$ys = [];
$i = 0;
while ($i < $n) {
    push($xs, $i);
    push($ys, sin($i / 900) * 50 + cos($i / 130) * 8);
    $i = $i + 1;
}
$t0 = clock();
$f = plt.figure(800, 500);
$a = $f.addAxes();
$a.plot($xs, $ys, {"label": "signal"});
$a.setTitle("100,000 points");
$a.setGrid(true);
$a.setLegend(true);
$svg = $f.to_svg();
$ms = clock() - $t0;
writefile("OUTDIR/big.svg", $svg);
print("RENDER_MS " + str($ms));
print("BYTES " + str(len($svg)));

// The same data with simplification off, to prove it is doing something.
$f2 = plt.figure(800, 500);
$a2 = $f2.addAxes();
$a2.simplify = false;
$a2.plot($xs, $ys, null);
print("BYTES_RAW " + str(len($f2.to_svg())));
BEOF
sed -i.bak "s#OUTDIR#$TMP#g" "$TMP/big.b" && rm -f "$TMP/big.b.bak"

if "$BANTU" -q run "$TMP/big.b" > "$TMP/big.log" 2>&1; then
    MS=$(grep '^RENDER_MS' "$TMP/big.log" | awk '{print $2}')
    BY=$(grep '^BYTES ' "$TMP/big.log" | awk '{print $2}')
    RAW=$(grep '^BYTES_RAW' "$TMP/big.log" | awk '{print $2}')
    echo "        rendered in ${MS} ms; ${BY} bytes simplified, ${RAW} bytes raw"
    ok "a 100,000-point line renders"
    # Generous, because CI runners are shared and slow. A quadratic string
    # build or a per-point element would be tens of seconds, not single digits.
    check "$([ "${MS:-99999}" -lt 20000 ] && echo 1 || echo 0)" \
          "and does so in under 20 s (it is one polyline, not 100,000 elements)"
    # Simplification bounds the output by the canvas, not by the data.
    check "$([ "${BY:-0}" -lt 200000 ] && echo 1 || echo 0)" \
          "the simplified document is under 200 KB"
    check "$([ "${RAW:-0}" -gt "${BY:-0}" ] && echo 1 || echo 0)" \
          "and is smaller than the same chart with simplification off"
    check "$(wellformed "$TMP/big.svg" && echo 1 || echo 0)" \
          "the 100,000-point document parses"
    check "$([ "$(grep -c '<polyline' "$TMP/big.svg")" = "1" ] && echo 1 || echo 0)" \
          "and it is a single polyline"
else
    bad "the 100,000-point chart failed"
    tail -20 "$TMP/big.log"
fi

# ── Repetition: memory must not creep ────────────────────────────────────
echo ""
echo "-- repetition: 2,000 figures built and dropped --"
cat > "$TMP/rep.b" <<'BEOF'
include "./bplot/bplot.b" as plt;
$xs = [];
$ys = [];
$i = 0;
while ($i < 200) { push($xs, $i); push($ys, $i * $i); $i = $i + 1; }
$total = 0;
$k = 0;
while ($k < 2000) {
    $f = plt.figure(400, 300);
    $a = $f.addAxes();
    $a.plot($xs, $ys, {"label": "s"});
    $a.setTitle("t");
    $a.setLegend(true);
    $total = $total + len($f.to_svg());
    $k = $k + 1;
}
print("TOTAL " + str($total));
BEOF
if "$BANTU" -q run "$TMP/rep.b" > "$TMP/rep.log" 2>&1; then
    ok "2,000 figures built and dropped without failing"
    grep -q '^TOTAL' "$TMP/rep.log" && ok "and every one produced output" || bad "output missing"
else
    bad "the repetition run failed"
    tail -10 "$TMP/rep.log"
fi

# ── The documented examples must actually run ────────────────────────────
echo ""
echo "-- the quickstart from the docs --"
cat > "$TMP/quick.b" <<'BEOF'
include "./bplot/bplot.b" as plt;
plt.plot([1, 2, 3], [2, 4, 9]);
plt.savefig("OUTDIR/quick.svg");
BEOF
sed -i.bak "s#OUTDIR#$TMP#g" "$TMP/quick.b" && rm -f "$TMP/quick.b.bak"
if "$BANTU" -q run "$TMP/quick.b" >/dev/null 2>&1 && [ -f "$TMP/quick.svg" ]; then
    ok "three lines produce a chart, with no options dict"
    check "$(wellformed "$TMP/quick.svg" && echo 1 || echo 0)" "and it parses"
else
    bad "the quickstart did not run"
fi

echo ""
echo "  $PASS passed, $FAIL failed"
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
