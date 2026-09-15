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

# ── B4: data from arctic and numba, at scale and with holes in it ────────
# Rendered BEFORE the XML gate below, so every one of these documents is also
# handed to a real XML parser.
#
# The numbers this section exists for, measured before B4 on one 1,000,000-
# point line through the convert-to-a-list boundary: ~20 s and 3.1 GB of
# resident memory (docs/bplot-architecture.md §13).
echo "-- B4: arctic and numba data, at scale and with holes in it --"
cat > "$TMP/b4.b" <<'BEOF'
include "./bplot/bplot.b" as plt;
include "./arctic/arctic.b" as ac;

// 1,000,000 rows, built natively, so the frame is arctic columns end to end.
$x = nd_linspace(0, 100, 1000000, null);
$y = nd_add(nd_sin($x, null), nd_multiply(nd_cos(nd_multiply($x, 7.3, null), null), 0.2, null), null);
$df = ac.from_columns({"t": nd_to_column($x, null), "v": nd_to_column($y, null)});

$t0 = clock();
$f = plt.figure(900, 500);
$a = $f.addAxes();
$a.plot($df.get("t"), $df.get("v"), {"label": "v"});
$a.setTitle("1M rows");
$f.savefig("OUTDIR/b4_million_line.svg");
print("LINE_MS " + str(clock() - $t0));

$t0 = clock();
$f = plt.figure(900, 500); $a = $f.addAxes();
$a.hist($df.get("v"), {"bins": 60});
$f.savefig("OUTDIR/b4_million_hist.svg");
print("HIST_MS " + str(clock() - $t0));

$t0 = clock();
$f = plt.figure(900, 500); $a = $f.addAxes();
$a.boxplot([$df.get("v"), $df.get("t")], {"labels": ["v", "t"]});
$f.savefig("OUTDIR/b4_million_box.svg");
print("BOX_MS " + str(clock() - $t0));

$t0 = clock();
$f = plt.figure(900, 500); $a = $f.addAxes();
$a.violin($df.get("v"), null);
$f.savefig("OUTDIR/b4_million_violin.svg");
print("VIOLIN_MS " + str(clock() - $t0));

// A column that is one-third holes.
$holes = [];
$i = 0;
while ($i < 5000) {
    if ($i % 3 == 0) { push($holes, null); } else { push($holes, sin($i * 0.01)); }
    $i = $i + 1;
}
$f = plt.figure(600, 400); $a = $f.addAxes();
$a.plot(nd_arange(0, 5000, 1), ac.series("h", $holes, "f64"), null);
$f.savefig("OUTDIR/b4_nulls.svg");
print("NULLS_OK");

// A column that is ALL nulls: nothing to draw, and still a valid chart.
$f = plt.figure(600, 400); $a = $f.addAxes();
$a.plot([1, 2, 3], ac.series("z", [null, null, null], "f64"), null);
$a.hist(ac.series("z", [null, null, null], "f64"), null);
$f.savefig("OUTDIR/b4_all_null.svg");
print("ALLNULL_OK");

// A frame with a text column selected by name raises, catchably -- and the
// process carries on and still draws the frame correctly afterwards.
$mixed = ac.dataframe({"name": ["a", "b", "c"], "score": [3, 1, 2]}, null);
$msg = "";
try { plt.figure(400, 300).addAxes().plot_frame($mixed, {"y": "name"}); } catch ($e) { $msg = str($e); }
if (contains($msg, "column 'name' is utf8")) { print("NONNUMERIC_RAISED"); }
$f = plt.figure(600, 400); $a = $f.addAxes();
$a.plot_frame($mixed, {"kind": "bar", "x": "name"});
$f.savefig("OUTDIR/b4_frame_bar.svg");
print("FRAME_OK");

// Hostile column NAMES reach the legend and the axis labels.
$evil = "</text><script>alert(1)</script>";
$d = {};
$d[$evil] = [1, 2, 3];
$d["ok"] = [3, 2, 1];
$f = plt.figure(600, 400); $a = $f.addAxes();
$a.plot_frame(ac.dataframe($d, null), null);
$f.savefig("OUTDIR/b4_hostile_names.svg");
print("HOSTILE_OK");

// A 1,000,000-point scatter, batched into one path.
$t0 = clock();
$f = plt.figure(900, 500); $a = $f.addAxes();
$a.scatter($x, $y, {"size": 1});
$f.savefig("OUTDIR/b4_million_scatter.svg");
print("SCATTER_MS " + str(clock() - $t0));
BEOF
sed -i.bak "s#OUTDIR#$TMP#g" "$TMP/b4.b" && rm -f "$TMP/b4.b.bak"
if "$BANTU" -q run "$TMP/b4.b" > "$TMP/b4.log" 2>&1; then
    ok "every B4 case rendered"
else
    bad "a B4 case raised"
    tail -20 "$TMP/b4.log"
fi
msOf() { sed -n "s/^$1 //p" "$TMP/b4.log" | head -1; }
LINE_MS="$(msOf LINE_MS)"; HIST_MS="$(msOf HIST_MS)"; BOX_MS="$(msOf BOX_MS)"
VIOLIN_MS="$(msOf VIOLIN_MS)"; SCATTER_MS="$(msOf SCATTER_MS)"
echo "        1,000,000 rows: line ${LINE_MS} ms, hist ${HIST_MS} ms, box ${BOX_MS} ms, violin ${VIOLIN_MS} ms, scatter ${SCATTER_MS} ms"
# Gates carry roughly ten times headroom over this machine for a loaded CI box;
# each is still an order of magnitude under what the list boundary cost.
check "$([ -n "$LINE_MS" ] && [ "$LINE_MS" -lt 5000 ] && echo 1 || echo 0)" \
      "a 1,000,000-row arctic column draws as a line in under 5 s (was ~20 s)"
check "$([ -n "$HIST_MS" ] && [ "$HIST_MS" -lt 3000 ] && echo 1 || echo 0)" \
      "a 1,000,000-row histogram in under 3 s"
check "$([ -n "$BOX_MS" ] && [ "$BOX_MS" -lt 8000 ] && echo 1 || echo 0)" \
      "two 1,000,000-row boxes in under 8 s"
check "$([ -n "$VIOLIN_MS" ] && [ "$VIOLIN_MS" -lt 8000 ] && echo 1 || echo 0)" \
      "a 1,000,000-row violin in under 8 s"
check "$([ -n "$SCATTER_MS" ] && [ "$SCATTER_MS" -lt 20000 ] && echo 1 || echo 0)" \
      "a 1,000,000-point scatter in under 20 s"
LINE_BYTES="$(wc -c < "$TMP/b4_million_line.svg" 2>/dev/null | tr -d ' ')"
echo "        the 1,000,000-row line document: ${LINE_BYTES} bytes"
check "$([ "${LINE_BYTES:-0}" -gt 0 ] && [ "$LINE_BYTES" -lt 100000 ] && echo 1 || echo 0)" \
      "and its size is set by the canvas, not the data"
for mark in NULLS_OK ALLNULL_OK NONNUMERIC_RAISED FRAME_OK HOSTILE_OK; do
    check "$(grep -c "^$mark\$" "$TMP/b4.log")" "B4 $mark"
done
check "$([ -f "$TMP/b4_hostile_names.svg" ] && ! grep -q '<script' "$TMP/b4_hostile_names.svg" && echo 1 || echo 0)" \
      "a hostile column name is escaped wherever plot_frame writes it"

cat > "$TMP/b4rss.b" <<'BEOF'
include "./bplot/bplot.b" as plt;
include "./arctic/arctic.b" as ac;
$x = nd_linspace(0, 100, 1000000, null);
$df = ac.from_columns({"t": nd_to_column($x, null), "v": nd_to_column(nd_sin($x, null), null)});
$f = plt.figure(900, 500);
$a = $f.addAxes();
$a.plot($df.get("t"), $df.get("v"), null);
print(len($f.to_svg()));
BEOF
if /usr/bin/time -l true >/dev/null 2>&1; then
    B4_RSS="$(/usr/bin/time -l "$BANTU" -q run "$TMP/b4rss.b" 2>&1 | awk '/maximum resident/ { print int($1/1024) }')"
else
    B4_RSS="$(/usr/bin/time -v "$BANTU" -q run "$TMP/b4rss.b" 2>&1 | awk '/Maximum resident/ { print int($NF) }')"
fi
if [ -n "${B4_RSS:-}" ] && [ "$B4_RSS" -gt 0 ]; then
    echo "        peak RSS plotting 1,000,000 rows: ${B4_RSS} KB (was ~3.1 GB)"
    check "$([ "$B4_RSS" -lt 400000 ] && echo 1 || echo 0)" \
          "plotting 1,000,000 rows peaks under 400 MB"
else
    echo "  --    could not measure RSS on this platform; skipping the B4 memory gate"
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

# ════════════════════════════════════════════════════════════════════════
#  B2 — the chart types, the scales, and the date axis
# ════════════════════════════════════════════════════════════════════════
echo ""
echo "-- B2: every chart type, fed the input that breaks it --"

B2="$TMP/b2"
mkdir -p "$B2"
cat > "$TMP/b2.b" <<'BEOF'
include "./bplot/bplot.b" as plt;

def fig() { return plt.figure(600, 400); }
def save($f, $name) { $f.savefig("OUTDIR/" + $name + ".svg"); return null; }

// ── Degenerate input to each new chart type ──────────────────────────────
// A single-bin histogram: the range collapses, so the widening rule is the
// only thing standing between this and a division by zero.
$f = fig(); $a = $f.addAxes(); $a.hist([3, 3, 3], {"bins": 1});           save($f, "hist_onebin");
$f = fig(); $a = $f.addAxes(); $a.hist([1], {"bins": 10});                save($f, "hist_onevalue");
$f = fig(); $a = $f.addAxes(); $a.hist([1, NAN, 2, NAN], {"bins": 4});    save($f, "hist_nan");
$f = fig(); $a = $f.addAxes(); $a.hist([1, 2, 3], {"bins": 500});         save($f, "hist_manybins");
$f = fig(); $a = $f.addAxes(); $a.hist([-5, -4, -3, 0, 3], {"density": true}); save($f, "hist_density");

// A boxplot of one point: every quartile is that point and both whiskers
// have zero length.
$f = fig(); $a = $f.addAxes(); $a.boxplot([7], null);                     save($f, "box_one");
$f = fig(); $a = $f.addAxes(); $a.boxplot([[1],[2],[3]], null);           save($f, "box_ones");
$f = fig(); $a = $f.addAxes(); $a.boxplot([[5,5,5,5]], null);             save($f, "box_identical");
$f = fig(); $a = $f.addAxes(); $a.boxplot([[1,2,3,4,5,6,7,8,9,1000]], null); save($f, "box_outlier");

// A violin with no spread has no bandwidth; it must not divide by zero.
$f = fig(); $a = $f.addAxes(); $a.violin([[4,4,4]], null);                save($f, "violin_flat");
$f = fig(); $a = $f.addAxes(); $a.violin([[1,2],[3,4]], null);            save($f, "violin_tiny");

$f = fig(); $a = $f.addAxes(); $a.errorbar([1,2],[1,2],{"yerr":0});       save($f, "err_zero");
$f = fig(); $a = $f.addAxes(); $a.errorbar([1],[1],{"yerr":1,"xerr":1});  save($f, "err_one");
$f = fig(); $a = $f.addAxes(); $a.fill_between([1,2,3],[1,NAN,3],null,null); save($f, "band_nan");
$f = fig(); $a = $f.addAxes(); $a.fill_between([1],[1],null,null);        save($f, "band_one");
$f = fig(); $a = $f.addAxes(); $a.step([1],[1],{"where":"mid"});          save($f, "step_one");
$f = fig(); $a = $f.addAxes(); $a.step([1,2,3],[1,NAN,3],{"where":"post"}); save($f, "step_nan");
$f = fig(); $a = $f.addAxes(); $a.stem([1,2,3],[0,0,0],null);             save($f, "stem_zero");
$f = fig(); $a = $f.addAxes(); $a.pie([1],null);                          save($f, "pie_one");
$f = fig(); $a = $f.addAxes(); $a.pie([1,0,1],{"percent":true});          save($f, "pie_zeroslice");
$f = fig(); $a = $f.addAxes(); $a.pie([0.0001, 99999],{"percent":true});  save($f, "pie_sliver");

// ── Scales ───────────────────────────────────────────────────────────────
$f = fig(); $a = $f.addAxes(); $a.setScale("y","log",null);
$a.plot([1,2,3,4],[0.001,1,1000,1000000],null); $a.setGrid(true);         save($f, "log_wide");
$f = fig(); $a = $f.addAxes(); $a.setScale("y","log",null);
$a.plot([1,2],[3,7],null);                                               save($f, "log_narrow");
$f = fig(); $a = $f.addAxes(); $a.setScale("x","log",null); $a.setScale("y","log",null);
$a.scatter([1,10,100],[1,10,100],null);                                  save($f, "log_both");
$f = fig(); $a = $f.addAxes(); $a.setScale("y","log",null);
$a.bar([1,2,3],[0,10,1000],null);                                        save($f, "log_bar_zero");
$f = fig(); $a = $f.addAxes(); $a.setScale("y","symlog",null);
$a.plot([1,2,3,4,5],[0-1000,0-1,0,1,1000],null); $a.setGrid(true);        save($f, "symlog");
$f = fig(); $a = $f.addAxes(); $a.setScale("y","symlog",0.001);
$a.plot([1,2,3],[0-0.0001,0,0.0001],null);                               save($f, "symlog_tiny");

// ── Date axis, including a span short enough to fall off the ladder ──────
$f = fig(); $a = $f.addAxes(); $a.setDateAxis(true);
$a.plot([1704067200000,1704153600000,1704240000000],[1,2,3],null);        save($f, "date_days");
$f = fig(); $a = $f.addAxes(); $a.setDateAxis(true);
$a.plot([1704067200000,1704067200100,1704067200200],[1,2,3],null);        save($f, "date_sub_second");
$f = fig(); $a = $f.addAxes(); $a.setDateAxis(true);
$a.plot([0, 1704067200000],[1,2],null);                                   save($f, "date_epoch_span");

// ── A hostile annotation, which is a text node reached by a new route ────
$evil = "</text><script>alert(1)</script><foreignObject>";
$f = fig(); $a = $f.addAxes();
$a.plot([1,2,3],[1,2,3],null);
$a.addText(2, 2, $evil, {"rotate": 30});
$a.annotate($evil, 1.5, 2.5, {"to": [3, 3]});
$a.boxplot([[1,2],[3,4]], {"labels": [$evil, $evil]});
save($f, "hostile_b2");

// A hostile string through pie slice labels, which the legend never sees.
$f = fig(); $a = $f.addAxes();
$a.pie([1,1], {"labels": [$evil, "ok"], "percent": true});
save($f, "hostile_pie");

print("rendered");
BEOF
sed -i.bak "s#OUTDIR#$B2#g" "$TMP/b2.b" && rm -f "$TMP/b2.b.bak"

if "$BANTU" -q run "$TMP/b2.b" > "$TMP/b2.log" 2>&1; then
    ok "every B2 chart type rendered on degenerate input without raising"
else
    bad "a B2 case raised"
    tail -20 "$TMP/b2.log"
fi

if [ -z "$PY" ]; then
    echo "  --    no python found; skipping the B2 XML gate"
else
    BADXML=0; COUNT=0
    for f in "$B2"/*.svg; do
        [ -e "$f" ] || continue
        COUNT=$((COUNT+1))
        wellformed "$f" || { BADXML=$((BADXML+1)); echo "        malformed: $(basename "$f")"; }
    done
    echo "        $COUNT B2 documents checked"
    check "$([ "$BADXML" = "0" ] && [ "$COUNT" -gt 25 ] && echo 1 || echo 0)" \
          "every B2 document parses as well-formed XML"
fi

DIRTY=0
for f in "$B2"/*.svg; do
    [ -e "$f" ] || continue
    grep -qE '(NaN|[^a-z-]inf|-inf)[,"]' "$f" && { DIRTY=$((DIRTY+1)); echo "        $(basename "$f")"; }
done
check "$([ "$DIRTY" = "0" ] && echo 1 || echo 0)" \
      "no NaN or infinity reached a coordinate in any B2 document"

if [ -f "$B2/hostile_b2.svg" ]; then
    check "$(grep -c '<script' "$B2/hostile_b2.svg" | grep -q '^0$' && echo 1 || echo 0)" \
          "no <script survived an annotation, a text node or a group label"
    check "$(grep -c '<foreignObject' "$B2/hostile_b2.svg" | grep -q '^0$' && echo 1 || echo 0)" \
          "and no <foreignObject either"
    check "$(wellformed "$B2/hostile_b2.svg" && echo 1 || echo 0)" \
          "and the document still parses"
fi
if [ -f "$B2/hostile_pie.svg" ]; then
    check "$(grep -c '<script' "$B2/hostile_pie.svg" | grep -q '^0$' && echo 1 || echo 0)" \
          "pie slice labels are escaped too"
fi

# ── The raise a log axis exists to produce ───────────────────────────────
echo ""
echo "-- B2: a log axis must REFUSE non-positive data, not draw nothing --"
logfail() {
    cat > "$TMP/logbad.b" <<BEOF
include "./bplot/bplot.b" as plt;
\$f = plt.figure(400, 300);
\$a = \$f.addAxes();
\$a.setScale("$1", "log", null);
\$a.plot([1, 2, 3], [$2]);
\$f.to_svg();
print("NO_RAISE");
BEOF
    "$BANTU" -q run "$TMP/logbad.b" 2>&1
}
OUT="$(logfail y '1, 0, 3')"
check "$(echo "$OUT" | grep -q 'NO_RAISE' && echo 0 || echo 1)" "a zero in the data raises"
check "$(echo "$OUT" | grep -q 'cannot show 0' && echo 1 || echo 0)" "and the message names the value"
check "$(echo "$OUT" | grep -q 'symlog' && echo 1 || echo 0)" "and suggests symlog"
OUT="$(logfail y '1, -8, 3')"
check "$(echo "$OUT" | grep -q 'NO_RAISE' && echo 0 || echo 1)" "a negative in the data raises"
check "$(echo "$OUT" | grep -q -- '-8' && echo 1 || echo 0)" "and the message names it"

# ── Scale: a violin and a histogram over a lot of points ─────────────────
echo ""
echo "-- B2: 200,000 points through hist, boxplot and violin --"
cat > "$TMP/b2big.b" <<'BEOF'
include "./bplot/bplot.b" as plt;
$n = 200000;
$v = [];
$i = 0;
$s = 12345;
while ($i < $n) {
    // A cheap deterministic spread; the point is the volume, not the shape.
    $s = ($s * 1103515245 + 12345) - floor(($s * 1103515245 + 12345) / 2147483648) * 2147483648;
    push($v, $s / 2147483648 * 100);
    $i = $i + 1;
}
$t0 = clock();
$f = plt.figure(700, 450);
$a = $f.addAxes();
$a.hist($v, {"bins": 40});
$hs = $f.to_svg();
$t1 = clock();
print("HIST_MS " + str($t1 - $t0));
print("HIST_BYTES " + str(len($hs)));

$t0 = clock();
$f2 = plt.figure(700, 450);
$a2 = $f2.addAxes();
$a2.violin([$v], null);
$vs = $f2.to_svg();
$t1 = clock();
print("VIOLIN_MS " + str($t1 - $t0));
print("VIOLIN_BYTES " + str(len($vs)));
writefile("OUTDIR/b2big_hist.svg", $hs);
writefile("OUTDIR/b2big_violin.svg", $vs);
BEOF
sed -i.bak "s#OUTDIR#$TMP#g" "$TMP/b2big.b" && rm -f "$TMP/b2big.b.bak"
if "$BANTU" -q run "$TMP/b2big.b" > "$TMP/b2big.log" 2>&1; then
    HMS=$(grep '^HIST_MS' "$TMP/b2big.log" | awk '{print $2}')
    VMS=$(grep '^VIOLIN_MS' "$TMP/b2big.log" | awk '{print $2}')
    HBY=$(grep '^HIST_BYTES' "$TMP/b2big.log" | awk '{print $2}')
    VBY=$(grep '^VIOLIN_BYTES' "$TMP/b2big.log" | awk '{print $2}')
    echo "        hist ${HMS} ms / ${HBY} bytes; violin ${VMS} ms / ${VBY} bytes"
    ok "200,000 points bin and estimate without failing"
    # A histogram is 40 rects however many points went in: output must not
    # scale with the data.
    check "$([ "${HBY:-999999}" -lt 30000 ] && echo 1 || echo 0)" \
          "the histogram document is under 30 KB — it is 40 bars, not 200,000"
    # The binned KDE is why this is possible: a per-point kernel would be
    # 200,000 x 128 evaluations.
    check "$([ "${VMS:-999999}" -lt 60000 ] && echo 1 || echo 0)" \
          "the violin's binned KDE finishes in under 60 s at 200,000 points"
    check "$([ "${VBY:-999999}" -lt 30000 ] && echo 1 || echo 0)" \
          "and its document is under 30 KB"
    check "$(wellformed "$TMP/b2big_hist.svg" && echo 1 || echo 0)" "the big histogram parses"
    check "$(wellformed "$TMP/b2big_violin.svg" && echo 1 || echo 0)" "the big violin parses"
else
    bad "the 200,000-point B2 run failed"
    tail -20 "$TMP/b2big.log"
fi

# ════════════════════════════════════════════════════════════════════════
#  B3 — layout, 2-D, and the memory gate
# ════════════════════════════════════════════════════════════════════════
echo ""
echo "-- B3: layout and 2-D, on the input that breaks them --"

B3="$TMP/b3"
mkdir -p "$B3"
cat > "$TMP/b3.b" <<'BEOF'
include "./bplot/bplot.b" as plt;

def grid($rows, $cols) {
    $z = [];
    $r = 0;
    while ($r < $rows) {
        $row = [];
        $c = 0;
        while ($c < $cols) { push($row, sin($r / 7) * cos($c / 9) * 50); $c = $c + 1; }
        push($z, $row);
        $r = $r + 1;
    }
    return $z;
}

// An 8x8 grid, every panel carrying labels long enough to collide.
$f = plt.figure(1600, 1200);
$axs = $f.subplots(8, 8);
$i = 0;
while ($i < 64) {
    $axs[$i].plot([1, 2, 3], [100000, 250000, 175000], null);
    $axs[$i].setTitle("panel " + str($i));
    $axs[$i].setXLabel("a long x axis label");
    $axs[$i].setYLabel("a long y axis label");
    $i = $i + 1;
}
$f.tight_layout(true);
writefile("OUTDIR/grid8.svg", $f.to_svg());

// Degenerate layouts.
$f = plt.figure(200, 150); $f.subplots(4, 4); $f.tight_layout(true);
writefile("OUTDIR/tiny_grid.svg", $f.to_svg());
$f = plt.figure(400, 300); $a = $f.subplots(1, 1)[0]; $a.plot([1], [1], null);
writefile("OUTDIR/grid_1x1.svg", $f.to_svg());

// Twin axes on a log scale, with a shared group.
$f = plt.figure(700, 450);
$a = $f.addAxes();
$a.setScale("y", "log", null);
$a.plot([1, 2, 3], [1, 100, 10000], null);
$a.setYLabel("log left");
$b = $f.twinx($a);
$b.bar([1, 2, 3], [5, 3, 8], null);
$b.setYLabel("linear right");
$f.tight_layout(true);
writefile("OUTDIR/twin_log.svg", $f.to_svg());

// 2-D degenerate cases.
$f = plt.figure(500, 400); $f.addAxes().imshow([[1]], null);
writefile("OUTDIR/im_1x1.svg", $f.to_svg());
$f = plt.figure(500, 400); $f.addAxes().imshow([[5, 5], [5, 5]], null);
writefile("OUTDIR/im_flat.svg", $f.to_svg());
$f = plt.figure(500, 400); $f.addAxes().imshow([[NAN, NAN], [NAN, NAN]], null);
writefile("OUTDIR/im_allnan.svg", $f.to_svg());
$f = plt.figure(500, 400); $f.addAxes().imshow([[0 - INF, 1], [2, INF]], null);
writefile("OUTDIR/im_inf.svg", $f.to_svg());
$f = plt.figure(500, 400); $a = $f.addAxes(); $a.imshow(grid(3, 3), null); $a.colorbar(null);
writefile("OUTDIR/im_cbar.svg", $f.to_svg());
$f = plt.figure(600, 450); $f.addAxes().heatmap(grid(12, 12), null);
writefile("OUTDIR/heat.svg", $f.to_svg());
$f = plt.figure(500, 400); $f.addAxes().contour(grid(2, 2), {"levels": 3});
writefile("OUTDIR/ct_min.svg", $f.to_svg());
$f = plt.figure(500, 400); $f.addAxes().contour([[1, 1], [1, 1]], {"levels": 3});
writefile("OUTDIR/ct_flat.svg", $f.to_svg());
$f = plt.figure(500, 400); $f.addAxes().pcolormesh([0, 1, 5], [0, 2, 3], grid(2, 2), null);
writefile("OUTDIR/mesh.svg", $f.to_svg());

// Every style, so a theme change cannot break a document.
each ($s in ["default", "dark", "print"]) {
    plt.style($s);
    $f = plt.figure(500, 380);
    $a = $f.addAxes();
    $a.plot([1, 2, 3], [1, 3, 2], {"label": "a"});
    $a.hist([1, 2, 2, 3], {"bins": 3, "label": "h"});
    $a.setGrid(true);
    $a.setLegend(true);
    $a.setTitle($s);
    writefile("OUTDIR/style_" + $s + ".svg", $f.to_svg());
}
plt.style("default");

print("rendered");
BEOF
sed -i.bak "s#OUTDIR#$B3#g" "$TMP/b3.b" && rm -f "$TMP/b3.b.bak"

if "$BANTU" -q run "$TMP/b3.b" > "$TMP/b3.log" 2>&1; then
    ok "every B3 layout and 2-D case rendered without raising"
else
    bad "a B3 case raised"
    tail -20 "$TMP/b3.log"
fi

if [ -z "$PY" ]; then
    echo "  --    no python found; skipping the B3 XML gate"
else
    BADXML=0; COUNT=0
    for f in "$B3"/*.svg; do
        [ -e "$f" ] || continue
        COUNT=$((COUNT+1))
        wellformed "$f" || { BADXML=$((BADXML+1)); echo "        malformed: $(basename "$f")"; }
    done
    echo "        $COUNT B3 documents checked"
    check "$([ "$BADXML" = "0" ] && [ "$COUNT" -gt 13 ] && echo 1 || echo 0)" \
          "every B3 document parses as well-formed XML"
fi

DIRTY=0
for f in "$B3"/*.svg; do
    [ -e "$f" ] || continue
    grep -qE '(NaN|[^a-z-]inf|-inf)[,"]' "$f" && { DIRTY=$((DIRTY+1)); echo "        $(basename "$f")"; }
done
check "$([ "$DIRTY" = "0" ] && echo 1 || echo 0)" \
      "no NaN or infinity reached a coordinate in any B3 document"

# A dark figure must contain no hardcoded white — the legend card used to be
# one, which put near-invisible light text on a white box.
if [ -f "$B3/style_dark.svg" ]; then
    check "$(grep -c 'fill="#ffffff"' "$B3/style_dark.svg" | grep -q '^0$' && echo 1 || echo 0)" \
          "the dark style leaves no hardcoded white in the document"
fi

# ── The gate BP26 exists for ────────────────────────────────────────────
echo ""
echo "-- B3: a 1000x1000 imshow, which is where the naive encoding dies --"
cat > "$TMP/b3big.b" <<'BEOF'
include "./bplot/bplot.b" as plt;
$n = 1000;
$z = [];
$r = 0;
while ($r < $n) {
    $row = [];
    $c = 0;
    while ($c < $n) { push($row, ($r - 500) * ($r - 500) + ($c - 500) * ($c - 500)); $c = $c + 1; }
    push($z, $row);
    $r = $r + 1;
}
$t0 = clock();
$f = plt.figure(800, 600);
$a = $f.addAxes();
$a.imshow($z, null);
$a.colorbar(null);
$svg = $f.to_svg();
$ms = clock() - $t0;
writefile("OUTDIR/big_image.svg", $svg);
print("IMG_MS " + str($ms));
print("IMG_BYTES " + str(len($svg)));
print("IMG_PATHS " + str(len(split($svg, "<path")) - 1));
BEOF
sed -i.bak "s#OUTDIR#$TMP#g" "$TMP/b3big.b" && rm -f "$TMP/b3big.b.bak"
if "$BANTU" -q run "$TMP/b3big.b" > "$TMP/b3big.log" 2>&1; then
    IMS=$(grep '^IMG_MS' "$TMP/b3big.log" | awk '{print $2}')
    IBY=$(grep '^IMG_BYTES' "$TMP/b3big.log" | awk '{print $2}')
    IPA=$(grep '^IMG_PATHS' "$TMP/b3big.log" | awk '{print $2}')
    echo "        ${IMS} ms, ${IBY} bytes, ${IPA} path elements (1,000,000 input cells)"
    ok "a 1,000,000-cell image renders"
    # One <rect> per input cell would be ~55 MB. The cell budget plus
    # per-colour path batching is what keeps this openable.
    check "$([ "${IBY:-99999999}" -lt 2000000 ] && echo 1 || echo 0)" \
          "and the document is under 2 MB, not the 55 MB of one rect per cell"
    # At most 256 colours by construction, plus one bucket for no-data.
    check "$([ "${IPA:-99999}" -le 257 ] && echo 1 || echo 0)" \
          "and at most 257 elements, however many cells went in"
    check "$([ "${IMS:-999999}" -lt 300000 ] && echo 1 || echo 0)" \
          "and it finishes inside five minutes on a shared runner"
    check "$(wellformed "$TMP/big_image.svg" && echo 1 || echo 0)" \
          "the big image parses"
else
    bad "the 1000x1000 imshow failed"
    tail -20 "$TMP/b3big.log"
fi

# ── Memory: figures must be RECLAIMED, not merely survived ───────────────
echo ""
echo "-- B3: 6,000 figures built and dropped, with RSS measured --"
cat > "$TMP/rss.b" <<'BEOF'
include "./bplot/bplot.b" as plt;
$N = 1;
each ($a in split(env("BPLOT_RSS_N"), ",")) { $N = num($a); }
$xs = [];
$ys = [];
$i = 0;
while ($i < 60) { push($xs, $i); push($ys, $i * $i); $i = $i + 1; }
$k = 0;
$acc = 0;
while ($k < $N) {
    $f = plt.figure(420, 320);
    $a = $f.addAxes();
    $a.plot($xs, $ys, {"label": "s"});
    $a.setTitle("t");
    $a.setLegend(true);
    $acc = $acc + len($f.to_svg());
    $k = $k + 1;
}
print("ACC " + str($acc));
BEOF
rssOf() {
    # Portable-ish peak RSS. GNU time reports KB; BSD /usr/bin/time -l bytes.
    if /usr/bin/time -l true >/dev/null 2>&1; then
        BPLOT_RSS_N="$1" /usr/bin/time -l "$BANTU" -q run "$TMP/rss.b" 2>&1 |
            awk '/maximum resident/ { print int($1/1024); found=1 } END { if (!found) print 0 }'
    else
        BPLOT_RSS_N="$1" /usr/bin/time -v "$BANTU" -q run "$TMP/rss.b" 2>&1 |
            awk '/Maximum resident/ { print int($NF); found=1 } END { if (!found) print 0 }'
    fi
}
R_SMALL="$(rssOf 300)"
R_BIG="$(rssOf 6000)"
if [ "${R_SMALL:-0}" -gt 0 ] && [ "${R_BIG:-0}" -gt 0 ]; then
    echo "        300 figures: ${R_SMALL} KB;  6,000 figures: ${R_BIG} KB"
    # Twenty times the work must not cost twenty times the memory. Before class
    # instances were given an owner this grew linearly -- 8,000 figures reached
    # 372 MB and climbing, so a sua handler drawing a chart per request would
    # have been OOM-killed.
    check "$([ "$R_BIG" -lt $((R_SMALL * 2)) ] && echo 1 || echo 0)" \
          "RSS is flat: 20x the figures costs under 2x the memory"
else
    echo "  --    could not measure RSS on this platform; skipping"
fi

echo ""
echo "  $PASS passed, $FAIL failed"
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
