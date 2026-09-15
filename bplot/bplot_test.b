// ════════════════════════════════════════════════════════════════════════
//  bplot_test.b — the package's own smoke test.
//
//  Deliberately small. CI's real gates are tests/bplot_core_test.b,
//  bplot_charts_test.b, bplot_layout_test.b, bplot_data_test.b and
//  bplot_stress.sh, because CI only globs tests/*.b.
//  This one answers "did the package install and does it draw?" for anyone
//  who just ran `bantu add bplot`.
//
//  Run:  bantu run bplot/bplot_test.b
// ════════════════════════════════════════════════════════════════════════

include "./bplot.b" as plt;

$R = {"pass": 0, "fail": 0};
def ok($c, $n) {
    if ($c) { $R.pass = $R.pass + 1; print("  ok    " + $n); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $n); }
}

print("bplot smoke test");
print("");

$fig = plt.figure(400, 300);
$ax = $fig.addAxes();
$ax.plot([1, 2, 3, 4], [1, 4, 9, 16], {"label": "squares"});
$ax.setTitle("smoke");
$ax.setLegend(true);
$svg = $fig.to_svg();

ok(contains($svg, "<svg"), "a figure renders an svg");
ok(contains($svg, "</svg>"), "and closes it");
ok(contains($svg, "<polyline"), "the data is a polyline");
ok(contains($svg, ">smoke</text>"), "the title is there");
ok(contains($svg, ">squares</text>"), "and the legend");
ok(!contains($svg, "NaN"), "with no NaN in any coordinate");
ok(len(plt._ticks(0, 10, 6)) == 6, "ticks pick round numbers");
ok(plt._fmt(1 / 3, 2) == "0.33", "and labels are formatted, not stringified");
ok(!contains(plt._esc("<script>"), "<script"), "hostile text is escaped");

// numba arrays go straight in, and stay native until they become pixels.
if (has_native("ndarray")) {
    $f2 = plt.figure(400, 300);
    $a2 = $f2.addAxes();
    $a2.plot(nd_linspace(0, 1, 5000, null), nd_sin(nd_linspace(0, 20, 5000, null), null), null);
    $s2 = $f2.to_svg();
    ok(contains($s2, "<polyline"), "an ndarray draws");
    ok(len($s2) < 60000, "and 5,000 points are simplified to the canvas, not the data");
}

print("");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
