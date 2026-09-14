// ════════════════════════════════════════════════════════════════════════
//  bplot — a dashboard: subplots, twin axes, a heatmap and a colorbar.
//
//  This is the layout half of the library. Every axes rectangle in bplot,
//  whatever spelling created it, comes from one grid calculation — so a
//  panel from subplots() and the same panel from subplot(r, c, i) are the
//  same rectangle, and cannot drift apart.
//
//  tight_layout measures the text that will actually be drawn and gives each
//  panel the gutters it needs. It runs at RENDER time, not when you call it,
//  so it sees the labels you set afterwards.
//
//  Run:  bantu run samples/bplot/04_dashboard.b
// ════════════════════════════════════════════════════════════════════════

include "./bplot/bplot.b" as plt;

$months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

$rainfall    = [66, 61, 118, 290, 197, 33, 24, 22, 30, 41, 125, 118];
$temperature = [28, 28, 28, 27, 26, 24, 23, 23, 25, 26, 27, 28];

$fig = plt.figure(1100, 800);

// ── Top left: two series on one x axis, with their own y axes ────────────
// Rainfall in millimetres and temperature in degrees share nothing but the
// month. A twin axes gives each its own scale and draws the frame ONCE —
// two 1px strokes on the same rectangle is visible, and reads as a bug.
$ax = $fig.subplot(2, 2, 1);
$ax.bar($months, $rainfall, {"color": "#1f77b4", "label": "rainfall"});
$ax.setYLabel("rainfall (mm)");
$ax.setTitle("Dar es Salaam, monthly");
$ax.setXLabel("month");

$twin = $fig.twinx($ax);
$twin.plot($months, $temperature, {"color": "#d62728", "width": 2.4, "label": "temperature"});
$twin.setYLabel("temperature (C)");

// ── Top right: a cumulative view, as a step chart ────────────────────────
// "post" means the value holds until the next sample, which is what a
// running total does. Drawn with the wrong convention a step chart is off by
// one sample and still looks plausible, which is why the names match
// matplotlib's exactly.
$cum = [];
$total = 0;
$i = 0;
while ($i < len($rainfall)) {
    $total = $total + $rainfall[$i];
    push($cum, $total);
    $i = $i + 1;
}
$ax2 = $fig.subplot(2, 2, 2);
$ax2.step($months, $cum, {"where": "post", "color": "#2ca02c", "width": 2});
$ax2.fill_between($months, $cum, null, {"color": "#2ca02c", "opacity": 0.18});
$ax2.setTitle("cumulative rainfall");
$ax2.setXLabel("month");
$ax2.setYLabel("mm, year to date");
$ax2.setGrid(true);

// ── Bottom left: a heatmap with the values written in ────────────────────
// Each cell's label is black or white depending on that cell's own
// brightness. A fixed colour is unreadable over half of any colormap, and
// this is the one place where the wrong choice makes the number vanish.
$years = ["2021", "2022", "2023", "2024"];
$quarters = ["Q1", "Q2", "Q3", "Q4"];
$sales = [[12, 18, 22, 31],
          [15, 21, 19, 36],
          [18, 25, 28, 41],
          [22, 29, 34, 48]];
$ax3 = $fig.subplot(2, 2, 3);
$ax3.heatmap($sales, {"rows": $years, "cols": $quarters, "cmap": "viridis"});
$ax3.colorbar({"label": "units sold"});
$ax3.setTitle("sales by quarter");

// ── Bottom right: a pie, which turns its own frame off ───────────────────
$ax4 = $fig.subplot(2, 2, 4);
$ax4.pie([48, 27, 15, 10],
         {"labels": ["direct", "search", "social", "other"], "percent": true});
$ax4.setTitle("traffic sources");

// tight_layout last — though it would work first, because it measures during
// rendering rather than when it is called.
$fig.tight_layout(true);
$path = $fig.savefig("/tmp/bplot_dashboard.svg");
print("wrote " + $path);

// ── The same dashboard in the dark style ─────────────────────────────────
// A style is process-wide, like the current figure. Set it once at start-up;
// a sua request handler must not, because handlers run on their own threads
// and would race each other.
plt.style("dark");
$dark = plt.figure(760, 460);
$dax = $dark.addAxes();
$dax.plot($months, $temperature, {"label": "temperature", "width": 2.4});
$dax.plot($months, [26, 26, 26, 25, 24, 22, 21, 21, 23, 24, 25, 26], {"label": "average"});
$dax.setGrid(true);
$dax.setLegend(true);
$dax.setTitle("dark style");
$dax.setXLabel("month");
$dax.setYLabel("degrees C");
$dark.tight_layout(true);
print("wrote " + $dark.savefig("/tmp/bplot_dark.svg"));
plt.style("default");
