// ════════════════════════════════════════════════════════════════════════
//  bplot_test.b — the package's own smoke test.
//
//  Deliberately small. CI's real gate is tests/bplot_core_test.b (149
//  assertions) and tests/bplot_stress.sh, because CI only globs tests/*.b.
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

print("");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
