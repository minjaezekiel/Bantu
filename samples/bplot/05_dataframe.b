// ════════════════════════════════════════════════════════════════════════
//  bplot + arctic — plotting a table.
//
//  bplot takes a Bantu list, a numba ndarray, an arctic column or a Series
//  anywhere it takes a sequence, and draws the same chart from all four. A
//  whole DataFrame draws with one call: $df.plot(...) in arctic, or
//  plt.plot_frame($df, ...) in bplot -- they are the same function, and
//  arctic only loads bplot the first time you plot.
//
//  Numbers stay native until they become pixels, which is why the last
//  chart below -- a million rows -- takes well under a second.
//
//  Run:  bantu run samples/bplot/05_dataframe.b
// ════════════════════════════════════════════════════════════════════════

include "./arctic/arctic.b" as ac;
include "./bplot/bplot.b" as plt;

$df = ac.dataframe({
    "month":  ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "dar":    [66, 61, 118, 290, 197, 33, 24, 22, 30, 41, 125, 118],
    "arusha": [53, 70, 145, 330, 102, 13, 8, 10, 12, 24, 108, 85],
    "note":   ["dry", "dry", "wet", "long rains", "wet", "dry", "dry", "dry", "dry", "dry", "short rains", "wet"]
}, null);

// ── 1. A whole table, grouped bars ───────────────────────────────────────
// "note" is text, so it is not drawn -- the default is every NUMERIC column.
// Name a text column in "y" and bplot raises, naming it and its type, rather
// than silently skipping a column you asked for.
$df.plot({"kind": "bar", "x": "month", "title": "Monthly rainfall (mm)"});
print("wrote " + plt.savefig("/tmp/bplot_frame_bars.svg"));

// ── 2. Two columns against each other ────────────────────────────────────
// Series go straight in. A Series names itself, so a length mismatch says
// which series was short.
plt.scatter($df.get("dar"), $df.get("arusha"), {"size": 5});
plt.xlabel("Dar es Salaam (mm)");
plt.ylabel("Arusha (mm)");
plt.title("Same month, two stations");
print("wrote " + plt.savefig("/tmp/bplot_frame_stations.svg"));

// ── 3. A time series with a missing reading ──────────────────────────────
// A datetime column makes a date axis on its own. The null is a missing
// value, drawn as a gap: nothing is invented to fill it.
$daily = ac.dataframe({
    "date":  ["2024-04-01", "2024-04-02", "2024-04-03", "2024-04-04", "2024-04-05", "2024-04-06"],
    "rain":  [12.5, 30.1, null, 44.0, 8.2, 19.7]
}, null);
plt.plot($daily.get("date").to_datetime(), $daily.get("rain"), {"width": 2.4});
plt.title("Daily rainfall, April -- one reading missing");
plt.grid(true);
print("wrote " + plt.savefig("/tmp/bplot_frame_timeseries.svg"));

// ── 4. A million rows ────────────────────────────────────────────────────
// Built natively and plotted without ever becoming a Bantu list. Before B4
// this one chart took ~20 s and 3.1 GB; path simplification keeps the
// document the size of the canvas, not the size of the data.
$t = nd_linspace(0, 60, 1000000, null);
$signal = ac.from_columns({
    "t": nd_to_column($t, null),
    "v": nd_to_column(nd_add(nd_sin($t, null), nd_multiply(nd_sin(nd_multiply($t, 11, null), null), 0.15, null), null), null)
});
$t0 = clock();
plt.plot($signal.get("t"), $signal.get("v"), null);
plt.title("1,000,000 samples");
$path = plt.savefig("/tmp/bplot_frame_million.svg");
print("wrote " + $path + " in " + str(clock() - $t0) + " ms");
