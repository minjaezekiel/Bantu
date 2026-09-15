// ════════════════════════════════════════════════════════════════════════
//  bplot_data_test.b — B4: data integration.
//
//  THE GATE: the same data renders to the SAME DOCUMENT whether it arrives as
//  a Bantu list, a numba ndarray, an arctic column or an arctic Series. Every
//  check below that compares documents compares them byte for byte.
//
//  And the second gate that makes the first meaningful: the native path
//  (ndarrays and C++ kernels, BP32/BP33) and the pure-Bantu path
//  (_useNative(false)) render the same data to the same bytes. The pure path
//  is B1-B3's code, so this holds the native path to everything B1-B3 pinned.
//
//  Design: docs/bplot-architecture.md §13, decisions BP31-BP36.
// ════════════════════════════════════════════════════════════════════════

include "./bplot/bplot.b" as plt;
include "./arctic/arctic.b" as ac;

$R = {"pass": 0, "fail": 0};
def ok($cond, $what) {
    if ($cond) { $R["pass"] = $R["pass"] + 1; }
    else { $R["fail"] = $R["fail"] + 1; print("  FAIL: " + $what); }
}
def eq($got, $want, $what) {
    if ($got == $want) { $R["pass"] = $R["pass"] + 1; }
    else { $R["fail"] = $R["fail"] + 1; print("  FAIL: " + $what + " -- got " + str($got) + ", wanted " + str($want)); }
}
def countOf($s, $needle) { return len(split($s, $needle)) - 1; }
def has($s, $needle) { return countOf($s, $needle) > 0; }

// Raises, and the message contains every one of the given fragments.
def raises($fn, $fragments, $what) {
    $msg = null;
    try { $fn(); } catch ($e) { $msg = str($e); }
    if ($msg == null) { ok(false, $what + " (did not raise)"); return null; }
    $all = true;
    each ($f in $fragments) { if (!has($msg, $f)) { $all = false; print("    missing '" + $f + "' in: " + $msg); } }
    ok($all, $what);
    return null;
}

// One chart, drawn on a fresh figure by `draw`.
def render($draw) {
    $f = plt.figure(640, 400);
    $ax = $f.addAxes();
    $draw($ax);
    return $f.to_svg();
}

// The same data four ways.
def fourWays($name, $list) {
    return {
        "list":    $list,
        "ndarray": nd($list, "f64"),
        "column":  ac.series($name, $list, "f64").col,
        "series":  ac.series($name, $list, "f64")
    };
}

// Render with every input type and with the pure path; all must equal the
// native list rendering. `draw2` receives ($ax, x, y).
def sameEverywhere($label, $xs, $ys, $draw2) {
    $X = fourWays("x", $xs);
    $Y = fourWays("y", $ys);
    $ref = render(def($ax) { $draw2($ax, $X["list"], $Y["list"]); });
    ok(len($ref) > 200, $label + ": renders a document");
    each ($k in ["ndarray", "column", "series"]) {
        $doc = render(def($ax) { $draw2($ax, $X[$k], $Y[$k]); });
        ok($doc == $ref, $label + ": " + $k + " renders the same bytes as a list");
    }
    $was = plt._useNative(false);
    $pure = render(def($ax) { $draw2($ax, $X["list"], $Y["list"]); });
    plt._useNative($was);
    ok($pure == $ref, $label + ": the pure-Bantu path renders the same bytes as the native one");
    return $ref;
}

// ── data ──────────────────────────────────────────────────────────────────
$n = 240;
$xs = [];
$ys = [];
$i = 0;
while ($i < $n) {
    push($xs, $i * 0.25);
    push($ys, sin($i * 0.11) * 40 + $i * 0.3);
    $i = $i + 1;
}
$ys[57] = NAN;            // a gap
$ys[58] = NAN;

// Big enough to engage path simplification (>= 1,000 points in one run) and
// scatter batching (>= 1,000 finite points).
$bx = [];
$by = [];
$i = 0;
while ($i < 5000) {
    push($bx, $i);
    push($by, sin($i * 0.013) * 100 + sin($i * 0.9) * 7);
    $i = $i + 1;
}
$by[2500] = INF;          // splits the run in two

print("── the gate: four input types, one document ───────────────────────");

sameEverywhere("line", $xs, $ys, def($ax, $x, $y) { $ax.plot($x, $y, {"label": "s"}); });
$bigLine = sameEverywhere("line, 5,000 points (simplified)", $bx, $by,
    def($ax, $x, $y) { $ax.plot($x, $y, null); });
ok(countOf($bigLine, "<polyline") == 2, "an infinity splits a long line into two runs");
sameEverywhere("scatter", $xs, $ys, def($ax, $x, $y) { $ax.scatter($x, $y, null); });
$bigScatter = sameEverywhere("scatter, 5,000 points (batched)", $bx, $by,
    def($ax, $x, $y) { $ax.scatter($x, $y, {"size": 2}); });
eq(countOf($bigScatter, "<circle"), 0, "a 5,000-point scatter emits no <circle> elements");
ok(has($bigScatter, "a2.00,2.00 0 1,0 4.00,0"), "it is one path of two-arc circles");
sameEverywhere("hist", $xs, $ys, def($ax, $x, $y) { $ax.hist($y, {"bins": 12}); });
sameEverywhere("hist, density", $bx, $by, def($ax, $x, $y) { $ax.hist($y, {"bins": 30, "density": true}); });
sameEverywhere("boxplot", $xs, $ys, def($ax, $x, $y) { $ax.boxplot([$x, $y], null); });
sameEverywhere("violin", $xs, $ys, def($ax, $x, $y) { $ax.violin([$x, $y], null); });
sameEverywhere("violin, 5,000 points", $bx, $by, def($ax, $x, $y) { $ax.violin($y, null); });
sameEverywhere("bar", [1, 2, 3, 4], [3, 1, 4, 1], def($ax, $x, $y) { $ax.bar($x, $y, null); });
sameEverywhere("step", $xs, $ys, def($ax, $x, $y) { $ax.step($x, $y, null); });
sameEverywhere("stem", [1, 2, 3, 4, 5], [2, 7, 1, 8, 2], def($ax, $x, $y) { $ax.stem($x, $y, null); });
sameEverywhere("fill_between", $xs, $ys, def($ax, $x, $y) { $ax.fill_between($x, $y, null, null); });
sameEverywhere("errorbar", [1, 2, 3], [4, 5, 6], def($ax, $x, $y) { $ax.errorbar($x, $y, {"yerr": 0.5}); });

$posY = [];
each ($v in $by) { if (isfinite($v)) { push($posY, abs($v) + 1); } else { push($posY, $v); } }
sameEverywhere("line on a log axis", $bx, $posY,
    def($ax, $x, $y) { $ax.setScale("y", "log", null); $ax.plot($x, $y, null); });
sameEverywhere("line on a symlog axis", $bx, $by,
    def($ax, $x, $y) { $ax.setScale("y", "symlog", 10); $ax.plot($x, $y, null); });
sameEverywhere("scatter on a symlog axis", $bx, $by,
    def($ax, $x, $y) { $ax.setScale("y", "symlog", 3); $ax.scatter($x, $y, null); });

// The log-axis error is the same one, naming the same value, on both paths.
$neg = [1, 2, 0 - 3, 4];
raises(def() { render(def($ax) { $ax.setScale("y", "log", null); $ax.plot([1, 2, 3, 4], nd($neg, "f64"), null); }); },
       ["log scale cannot show -3", "symlog"], "native log axis names the first non-positive value");
$was = plt._useNative(false);
raises(def() { render(def($ax) { $ax.setScale("y", "log", null); $ax.plot([1, 2, 3, 4], $neg, null); }); },
       ["log scale cannot show -3", "symlog"], "pure log axis names the same value");
plt._useNative($was);

print("── nulls, datetimes, text ─────────────────────────────────────────");

// A null is a missing value: the same gap a NaN makes (BP34).
$withNull = [5, 3, null, 8, 6];
$withNaN  = [5, 3, NAN, 8, 6];
$nullCol = ac.series("v", $withNull, "f64");
$a1 = render(def($ax) { $ax.plot([0, 1, 2, 3, 4], $withNaN, null); });
$a2 = render(def($ax) { $ax.plot([0, 1, 2, 3, 4], $withNull, null); });
$a3 = render(def($ax) { $ax.plot([0, 1, 2, 3, 4], $nullCol, null); });
$a4 = render(def($ax) { $ax.plot([0, 1, 2, 3, 4], $nullCol.col, null); });
ok($a2 == $a1, "a null in a list draws the same gap as a NaN");
ok($a3 == $a1, "a null in a Series draws the same gap as a NaN");
ok($a4 == $a1, "a null in a column draws the same gap as a NaN");
$intNull = ac.series("n", [1, null, 3, 4], "i64");
ok(render(def($ax) { $ax.hist($intNull, null); }) == render(def($ax) { $ax.hist([1, NAN, 3, 4], null); }),
   "an i64 column with a null bins like the same data with a NaN");
ok(render(def($ax) { $ax.boxplot($intNull, null); }) == render(def($ax) { $ax.boxplot([1, 3, 4], null); }),
   "a null is left out of box statistics");

// A datetime column is epoch ms, and makes a date axis on its own.
$days = ac.dataframe({"day": ["2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04", "2024-03-05"],
                      "sales": [12, 15, 9, 21, 17]}, null);
$dt = ac.from_column("day", col_to_datetime($days.cols["day"]));
$ms = col_to_list(col_cast($dt.col, "i64"));
$byColumn = render(def($ax) { $ax.plot($dt, $days.get("sales"), null); });
$byHand = render(def($ax) { $ax.setDateAxis(true); $ax.plot($ms, [12, 15, 9, 21, 17], null); });
ok($byColumn == $byHand, "a datetime column plots as epoch ms on a date axis, with no xdate() call");
// matplotlib labels this four-day axis '03-01 00', '03-01 12', ... (3.11.2).
ok(has($byColumn, ">03-01 12</text>") && has($byColumn, ">03-02 00</text>"),
   "and the ticks are dated, as matplotlib labels them");
$dateOnly = ac.from_column("day", col_to_date($days.cols["day"]));
ok(render(def($ax) { $ax.plot($dateOnly, $days.get("sales"), null); }) == $byHand,
   "a date column is days, scaled to the same milliseconds");
$was = plt._useNative(false);
ok(render(def($ax) { $ax.plot($dt, $days.get("sales"), null); }) == $byColumn,
   "the pure path turns a datetime column into the same date axis");
plt._useNative($was);

// Text makes a categorical axis whichever way it arrives, and a null is labelled.
$cats = ["north", "south", "east"];
$c1 = render(def($ax) { $ax.bar($cats, [3, 5, 2], null); });
$c2 = render(def($ax) { $ax.bar(ac.series("r", $cats, "utf8"), [3, 5, 2], null); });
ok($c2 == $c1, "a text Series is a categorical axis, same as a list of strings");
$c3 = render(def($ax) { $ax.bar(ac.series("r", ["north", null, "east"], "utf8"), [3, 5, 2], null); });
ok(has($c3, ">null</text>"), "a null category is labelled null");
$c4 = render(def($ax) { $ax.plot($cats, nd([3, 5, 2], "f64"), null); });
ok(has($c4, ">south</text>"), "text x with ndarray y still makes a categorical line");

print("── mistakes raise, naming what is wrong ───────────────────────────");

$sa = ac.series("date", [1, 2, 3, 4, 5], "f64");
$sb = ac.series("price", [1, 2, 3, 4], "f64");
raises(def() { render(def($ax) { $ax.plot($sa, $sb, null); }); },
       ["x ('date') has 5 points", "y ('price') has 4"], "mismatched Series name both, native");
$was = plt._useNative(false);
raises(def() { render(def($ax) { $ax.plot($sa, $sb, null); }); },
       ["x ('date') has 5 points", "y ('price') has 4"], "mismatched Series name both, pure");
plt._useNative($was);
raises(def() { render(def($ax) { $ax.scatter(nd([1, 2, 3], "f64"), [1, 2], null); }); },
       ["x has 3 points", "y has 2"], "mismatched ndarray and list");
raises(def() { render(def($ax) { $ax.plot($days, [1, 2], null); }); },
       ["is a DataFrame", "plot_frame"], "a whole DataFrame passed as x says how to plot a table");
raises(def() { render(def($ax) { $ax.plot(nd([[1, 2], [3, 4]], "f64"), [1, 2], null); }); },
       ["1-dimensional", "[2, 2]"], "a 2-D array as a line raises, naming its shape");

print("── tables: plot_frame and df.plot() ───────────────────────────────");

$df = ac.dataframe({
    "month": ["Jan", "Feb", "Mar", "Apr"],
    "rain":  [66, 61, 118, 290],
    "temp":  [28.1, 28.4, 27.9, 27.2],
    "note":  ["dry", "dry", "wet", "wet"]
}, null);

$line = render(def($ax) { $ax.plot_frame($df, null); });
ok(has($line, ">rain</text>") && has($line, ">temp</text>"), "default: every numeric column, legend on");
ok(!has($line, ">note</text>") && !has($line, ">month</text>"), "default skips the text columns nobody asked for");
eq(countOf($line, "<polyline"), 2, "two numeric columns, two lines");

$barDoc = render(def($ax) { $ax.plot_frame($df, {"kind": "bar", "x": "month", "y": ["rain", "temp"]}); });
ok(has($barDoc, ">Jan</text>") && has($barDoc, ">Apr</text>"), "grouped bars are labelled by the x column");
ok(has($barDoc, ">month</text>"), "the x column names the axis");
$one = render(def($ax) { $ax.plot_frame($df, {"kind": "bar", "x": "month", "y": "rain", "title": "Rain"}); });
ok(has($one, ">Rain</text>") && has($one, ">rain</text>"), "one series: title and the y column label");
ok(!has($one, "legend"), "one series has no legend");
ok(has(render(def($ax) { $ax.plot_frame($df, {"kind": "barh", "x": "month", "y": "rain"}); }), ">Mar</text>"),
   "barh puts the categories on y");
$sc = render(def($ax) { $ax.plot_frame($df, {"kind": "scatter", "x": "rain", "y": "temp"}); });
ok(has($sc, "<circle") && has($sc, ">rain</text>") && has($sc, ">temp</text>"), "scatter draws and labels both axes");
ok(countOf(render(def($ax) { $ax.plot_frame($df, {"kind": "hist", "y": "rain", "bins": 4}); }), "<rect") > 4,
   "hist of one column");
ok(has(render(def($ax) { $ax.plot_frame($df, {"kind": "box"}); }), ">temp</text>"),
   "box labels each box with its column");
ok(has(render(def($ax) { $ax.plot_frame($df, {"kind": "step", "y": "rain"}); }), "<polyline"), "step");
ok(has(render(def($ax) { $ax.plot_frame($df.get("rain"), null); }), "<polyline"),
   "a Series plots as a one-column table");
ok(has(render(def($ax) { $ax.heatmap($df, null); }), ">rain</text>"), "heatmap of a frame labels its columns");

// The explicit same chart, drawn by hand, is the same document.
$manual = render(def($ax) {
    $ax.plot([0, 1, 2, 3], [66, 61, 118, 290], {"label": "rain"});
    $ax.plot([0, 1, 2, 3], [28.1, 28.4, 27.9, 27.2], {"label": "temp"});
    $ax.setLegend(true);
});
ok($line == $manual, "plot_frame is exactly the plot() calls it stands for");

raises(def() { render(def($ax) { $ax.plot_frame($df, {"y": "note"}); }); },
       ["column 'note' is utf8", "not numeric", "rain (i64)"], "a text column named explicitly raises, naming it and its type");
raises(def() { render(def($ax) { $ax.plot_frame($df, {"y": "snow"}); }); },
       ["no column 'snow'", "month, rain, temp, note"], "an unknown column lists the real ones");
raises(def() { render(def($ax) { $ax.plot_frame($df, {"kind": "pie"}); }); },
       ["unknown kind \"pie\"", "line, bar"], "an unknown kind lists the real ones");
raises(def() { render(def($ax) { $ax.plot_frame($df, {"kind": "scatter"}); }); },
       ["scatter needs one x column"], "scatter without x says what it needs");
raises(def() { render(def($ax) { $ax.plot_frame(ac.dataframe({"a": ["x"], "b": ["y"]}, null), null); }); },
       ["no numeric column", "a (utf8)"], "a frame with nothing numeric says so");

// df.plot() lives in arctic, includes bplot lazily, and shares the figure.
plt.clf();
$df.plot({"kind": "bar", "x": "month", "y": "rain"});
$viaArctic = plt.to_svg();
plt.clf();
plt.plot_frame($df, {"kind": "bar", "x": "month", "y": "rain"});
$viaBplot = plt.to_svg();
ok($viaArctic == $viaBplot, "$df.plot() draws into plt's current figure, identically to plt.plot_frame");
plt.clf();
$df.get("temp").plot(null);
ok(has(plt.to_svg(), "<polyline"), "$series.plot() works");
plt.clf();

print("── the kernels, attacked directly ─────────────────────────────────");

raises(def() { bp_line_runs(nd([1, 2], "f64"), nd([1], "f64"), nd([1, 1], "f64"), true); },
       ["same length"], "bp_line_runs rejects mismatched arrays");
raises(def() { bp_line_runs([1, 2], nd([1, 2], "f64"), nd([1, 1], "f64"), true); },
       ["1-dimensional ndarray"], "bp_line_runs rejects a list");
raises(def() { bp_scatter_path(nd([1], "f64"), nd([1], "f64"), nd([1], "f64"), "big"); },
       ["r must be a number"], "bp_scatter_path rejects a non-number radius");
eq(len(bp_line_runs(nd([], "f64"), nd([], "f64"), nd([], "f64"), true)), 0, "no points, no runs");
eq(len(bp_line_runs(nd([NAN, NAN], "f64"), nd([1, 2], "f64"), nd([0, 0], "f64"), true)), 0, "all non-finite, no runs");
eq(bp_scatter_path(nd([1], "f64"), nd([1], "f64"), nd([0], "f64"), 3), "", "no finite points, empty path");
$hostile = "</text><script>alert(1)</script> & \"q\" 'a'" + chr(1) + chr(27) + "\tok ünï";
eq(bp_escape($hostile), plt._esc($hostile), "bp_escape is _esc, byte for byte, on hostile text");
$was = plt._useNative(false);
$pureEsc = plt._esc($hostile);
plt._useNative($was);
eq(plt._esc($hostile), $pureEsc, "native and pure _esc agree");
ok(!has(bp_escape($hostile), "<script"), "no markup survives escaping");

print("");
print("Passed: " + str($R["pass"]) + "   Failed: " + str($R["fail"]));
if ($R["fail"] == 0) { print("RESULT: ALL GREEN"); } else { print("RESULT: FAILURES"); }
