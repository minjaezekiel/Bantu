// ────────────────────────────────────────────────────────────────────────
//  02_chart_types.b — line, scatter, bar and barh, and the object API.
//
//  The stateful plt.* calls all operate on one implicit figure. When you want
//  several figures, or you are inside a sua handler where module state is
//  shared across threads, use the objects directly -- as this does.
//
//  Run:  bantu run samples/bplot/02_chart_types.b
// ────────────────────────────────────────────────────────────────────────
include "./bplot/bplot.b" as plt;

$xs = [];
$ys = [];
$i = 0;
while ($i < 60) {
    push($xs, $i / 6);
    push($ys, sin($i / 6) * 10 + $i / 12);
    $i = $i + 1;
}

// A line, through the object API: no shared state, safe inside a server.
$fig = plt.figure(680, 420);
$ax = $fig.addAxes();
$ax.plot($xs, $ys, {"label": "signal", "width": 2});
$ax.setTitle("A damped-looking wave");
$ax.setXLabel("t");
$ax.setYLabel("amplitude");
$ax.setGrid(true);
$ax.setLegend(true);
print($fig.savefig("/tmp/bplot_line.svg"));

// Scatter.
$f2 = plt.figure(520, 380);
$a2 = $f2.addAxes();
$a2.scatter($xs, $ys, {"size": 3, "color": "purple", "label": "samples"});
$a2.setTitle("The same data, as points");
$a2.setGrid(true);
$a2.setLegend(true);
print($f2.savefig("/tmp/bplot_scatter.svg"));

// Bars, including negative values, which must extend the axis downward.
$f3 = plt.figure(520, 380);
$a3 = $f3.addAxes();
$a3.bar([1, 2, 3, 4, 5], [12, -4, 7, 15, -2], {"color": "#2ca02c"});
$a3.setTitle("Monthly change");
$a3.setXLabel("month");
$a3.setYLabel("change");
$a3.setGrid(true);
print($f3.savefig("/tmp/bplot_bar.svg"));

// Horizontal bars.
$f4 = plt.figure(520, 380);
$a4 = $f4.addAxes();
$a4.barh([1, 2, 3, 4], [8, 14, 3, 11], {"color": "#17becf"});
$a4.setTitle("Totals by region");
print($f4.savefig("/tmp/bplot_barh.svg"));
