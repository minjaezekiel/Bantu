# bplot — charts for Bantu

matplotlib's shape, in Bantu. Line, scatter, bar, histogram, box, violin, error bars, areas, steps,
stems and pies; log, symlog and date axes; several charts in one figure; heatmaps and images with
the published colormaps; three style sheets. It writes SVG, and it takes a Bantu list, a numba
ndarray, or an arctic column, Series or DataFrame — rendering exactly the same chart from each.

- **Source:** [`bplot/bplot.b`](../bplot/bplot.b) — pure Bantu — over three small native kernels in `bantu-src/compiler/src/plot_native.hpp`
- **Tests:** `tests/bplot_core_test.b`, `bplot_charts_test.b`, `bplot_layout_test.b`, `bplot_data_test.b`, `bplot_stress.sh`, `bplot_package_test.sh`, `bplot_sua_test.sh`
- **Design:** [`docs/bplot-architecture.md`](bplot-architecture.md), decisions in [`bplot-suite/DECISIONS.md`](../bplot-suite/DECISIONS.md)

Every example on this page is executed by `tests/run_doc_examples.sh`, in reading order, as one
program — so none of them can quietly stop working.

---

## Three lines to a chart

```bantu
include "bplot" as plt;

plt.plot([1, 2, 3, 4], [1, 4, 9, 16]);
plt.savefig("chart.svg");
```

Install it into a project with `bantu add bplot`. Open `chart.svg` in any browser.

---

## Why it exists

A chart is the fastest way to find out whether a number is wrong, and Bantu had no way to draw one.

bplot uses matplotlib's names — `plot`, `hist`, `xlabel`, `subplots`, `savefig` — so anyone who has
drawn a chart in Python can guess the call and be right. Where Bantu forces a difference it is the
same one everywhere: **there are no keyword arguments, so options are a dict**, `{"color": "#d62728",
"label": "rain"}`, always the last argument, always optional.

It is pure Bantu you can read, apart from three small kernels that exist for one reason: turning a
million points into the few thousand a canvas can show is too slow in an interpreted loop. They are
used only when your interpreter has them, and the library draws exactly the same document either way.

---

## A tour

### Line, bar and scatter

```bantu
$months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"];
$rain   = [66, 61, 118, 290, 197, 33];
$temp   = [28.1, 28.4, 27.9, 27.2, 26.1, 24.6];

plt.bar($months, $rain, {"color": "#1f77b4", "label": "rainfall"});
plt.title("Dar es Salaam");
plt.ylabel("mm");
plt.legend(true);
plt.savefig("rain.svg");

plt.scatter($rain, $temp, {"size": 4});
plt.xlabel("rainfall (mm)");
plt.ylabel("temperature (C)");
plt.savefig("rain_vs_temp.svg");
```

Text on an axis makes it categorical, so the months need no conversion. `savefig` writes the file
and starts a fresh figure.

### PNG

The same call writes a PNG when the name ends in `.png`. The dpi is optional: 96 draws one pixel per
unit, the size the SVG has; 192 is twice as sharp.

```bantu
include "bplot" as plt;
plt.bar(["Jan", "Feb", "Mar"], [66, 61, 118]);
plt.title("rainfall");
plt.savefig("rain.png", {"dpi": 192});
```

`to_png($dpi)` returns the bytes instead, for a response body. The PNG is drawn by bplot's own
rasteriser in an embedded font, so **the same figure is the same file on Linux, macOS and Windows**
([how](bplot-raster-architecture.md)).

### Distributions

```bantu
$scores = nd_random_normal([2000], 50, 12, 7);

plt.hist($scores, {"bins": 30});
plt.title("2,000 scores");
plt.savefig("hist.svg");

plt.boxplot([$scores, nd_random_normal([2000], 58, 6, 8)], {"labels": ["before", "after"]});
plt.savefig("box.svg");
```

Those are numba arrays, going straight in. Histogram bins follow NumPy's rules — including the
inclusive top edge, without which the largest value silently disappears — and quantiles are NumPy's
default method, so a median here is the median NumPy reports.

### Scales and dates

```bantu
plt.plot([1, 10, 100, 1000, 10000], [3, 30, 250, 2600, 31000]);
plt.xscale("log", null);
plt.yscale("log", null);
plt.savefig("loglog.svg");

// Dates are epoch milliseconds, UTC -- which is how arctic stores a datetime column.
$day = 86400000;
$april = 1711929600000;
plt.plot([$april, $april + $day, $april + 2 * $day, $april + 3 * $day], [12.5, 30.1, 8.2, 19.7]);
plt.xdate(true);
plt.savefig("dates.svg");
```

A log axis handed a zero or a negative number **raises**, naming the value and suggesting `"symlog"`
— it does not draw a chart with the point silently missing. Date ticks land where matplotlib puts
them: the 1st and 15th of each month on a three-month axis, every fourth year on a twenty-five-year
one.

### Several charts in one figure

```bantu
$fig = plt.figure(900, 400);
$left = $fig.subplot(1, 2, 1);
$left.bar($months, $rain, null);
$left.setTitle("rainfall");
$right = $fig.subplot(1, 2, 2);
$right.plot($months, $temp, {"color": "#d62728", "width": 2.4});
$right.setTitle("temperature");
$fig.tight_layout(true);
$fig.savefig("two_panels.svg");
```

Every `plt.*` call has an axes method with the same name in `camelCase` for the setters — `plt.title`
is `$ax.setTitle`. `tight_layout` measures the labels that will actually be drawn, at render time, so
it sees labels you set after calling it.

### Grids

```bantu
$sales = [[12, 18, 22, 31], [15, 21, 19, 36], [18, 25, 28, 41]];
plt.heatmap($sales, {"rows": ["2022", "2023", "2024"], "cols": ["Q1", "Q2", "Q3", "Q4"], "cmap": "viridis"});
plt.colorbar({"label": "units sold"});
plt.savefig("heatmap.svg");
```

Colormaps are `viridis`, `plasma`, `coolwarm` and `gray`, as the published 256-entry tables. Each
cell's label is black or white according to that cell's own brightness.

### Tables

```bantu
include "arctic" as ac;

$df = ac.dataframe({"month": $months, "rain": $rain, "temp": $temp}, null);

$df.plot({"kind": "bar", "x": "month", "y": "rain", "title": "From a DataFrame"});
plt.savefig("frame.svg");

plt.plot($df.get("month"), $df.get("temp"), null);
plt.savefig("series.svg");
```

`$df.plot(opts)` takes `kind` — `line`, `bar`, `barh`, `scatter`, `hist`, `box`, `step` — plus `x`,
`y` and `title`. Without `y` it draws every numeric column; naming a text column raises, naming it and
its type. It is the same function as `plt.plot_frame($df, opts)`. arctic loads bplot only the first
time you plot, and shares the figure `plt.savefig` writes.

A `null` in a column is a missing value and is drawn as a gap. A datetime column turns its axis into
a date axis on its own.

### Styles

```bantu
plt.style("dark");
plt.plot($months, $temp, {"width": 2.4});
plt.title("dark style");
plt.savefig("dark.svg");
plt.style("default");
```

`default` is matplotlib's tab10. `dark` lightens the cycle, because tab10's blue and purple vanish on
near-black. `print` orders its greys by lightness, so series stay distinguishable in a photocopy.

---

## Serving charts from sua

`samples/bplot/server.b` draws a chart per request:

```text
sua.server.get("/chart.svg", def($req, $res) {
    $fig = plt.figure(760, 420);                  // a figure of its own, not plt.*
    $ax = $fig.addAxes();
    $ax.bar($months, $rain, null);
    $ax.setTitle($req.query["title"]);            // from a stranger -- and escaped
    $res.set("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'");
    $res.set("X-Content-Type-Options", "nosniff");
    $res.type("image/svg+xml; charset=utf-8");
    $res.send($fig.to_svg());
});
```

**Use the object API in a handler.** `plt.plot` and friends draw into one current figure shared by
the whole process, so two requests using them at once would draw into each other's chart.
`tests/bplot_sua_test.sh` sends forty concurrent requests with different titles and checks every one
gets only its own.

**A PNG is one method away.** The sample also serves `/chart.png`, with `$res.type("image/png")` and
`$res.send($fig.to_png(null))`. Layout is measured by the backend doing the drawing, never by a
process-wide setting, so an SVG and a PNG rendered at once on two threads are each byte-identical to
what they are alone — the same test checks twenty-four of them.

---

## SVG is an executable document format

This is the part to read before serving a chart built from anyone else's data.

SVG is not an image in the way PNG is. It is XML that browsers execute, and it can carry scripts. A
chart title taken from a query string, a database row or an uploaded CSV is text from a stranger, and
rendered naively, `</text><script>…</script>` would run with your site's privileges when someone
opens the chart.

bplot makes that impossible **through bplot**:

- every text node and attribute value is escaped when it is emitted, with no way to turn it off;
- control characters, which would break the document, are removed;
- there is no raw-SVG escape hatch, so no caller can pass text around the escaping;
- colours and fonts are checked against what they are allowed to be, not pasted in.

What bplot cannot do is make SVG stop being an executable format. So when a chart contains anything
user-influenced, also serve it with the `Content-Security-Policy` header above, or from a separate
origin — or serve a PNG (`/chart.png` in the sample), which is pixels and carries nothing to execute.

---

## How it works, with real numbers

Measured on an Intel Core i7-9750H, macOS, with the `bantu` that `build-mac.sh` produces.

**Output is bounded by the canvas, not the data.** A canvas 800 pixels wide can show about 800 columns
of pixels, so for each column a line keeps its first, last, lowest and highest points — which keeps
every spike, where naive thinning drops them. A 100,000-point line is one polyline of **25,782 bytes**
against 1,386,925 unsimplified.

**Numbers stay native until they become pixels.** A Bantu value is about 190 bytes and a list is
copied every time a function receives it, so a million-point series held as a list is slow in a way
no drawing code can fix. bplot converts every numeric input to a numba array once, at the door:

| 1,000,000 rows from an arctic column | native | the same code as Bantu lists |
|---|---|---|
| draw as a line | **252 ms** | 15,455 ms |
| peak memory | **112 MB** | 2.69 GB |
| the document | 23,825 bytes | 23,825 bytes — identical |

A million rows through `hist` takes 61 ms, two box plots 241 ms, a violin 306 ms.

**A million-cell image is a few hundred elements.** `imshow` of 1000×1000 cells averages down to what
the canvas can show, then draws every cell of one colour as a single path: **1,473,633 bytes in 256
elements**, against about 55 MB for one rectangle per cell.

**The ticks are matplotlib's.** Tick positions match matplotlib's own choices on every range checked
— twelve linear ranges from 1e-9 to 1e9, eight date ranges from ten seconds to twenty-five years, and
the log and symlog ranges. On two log ranges bplot deliberately differs: asked for 2 to 9, matplotlib
labels nothing at all, and an axis with no labels is not a chart anyone can read.

**Memory is reclaimed.** Building and dropping 6,000 figures leaves resident memory flat.

**PNG costs a render on the server.** The four-panel dashboard in `samples/bplot/04_dashboard.b`
(1100×800, bars, a twin axis, a filled step chart, a heatmap with a colorbar, a pie):

| output | time | file |
|---|---|---|
| SVG | 37 ms | 25,386 bytes |
| PNG at 96 dpi | 71 ms | 77,041 bytes |
| PNG at 144 dpi | 118 ms | 125,327 bytes |
| PNG at 300 dpi | 368 ms | 290,196 bytes |

A tick label is 0.1 ms to draw at 96 dpi. A print-sized dashboard — 4000×3000 pixels at 300 dpi, with a
100,000-point line, a 20,000-point scatter, a 500×500 heatmap and a pie — renders in **2.2 s**, and the
whole process peaks at 220 MB; `tests/bplot_stress.sh` holds it to that.

**The same figure is the same file everywhere.** `tests/bplot_png_corpus_test.b` renders eight PNGs —
every primitive, text in every anchor and rotation, far-off coordinates, and whole figures at three
dpi values — and checks each SHA-256 against a recorded literal, in the Linux, macOS and Windows CI
jobs alike.

---

## Things worth knowing before you hit them

- **`plt.*` state is per process.** One current figure, one current style. In a server, build each
  chart with `plt.figure()` and set a style once at start-up, never in a handler.
- **`savefig` resets.** It writes the current figure and starts a new one, so the next `plt.plot`
  begins a new chart. `$fig.savefig` on a figure you hold does not reset anything.
- **`show(path)` writes a file.** Bantu has no way to open a window or launch a viewer, so `show`
  writes the SVG and prints where it is.
- **A NaN or a `null` splits a line.** It is drawn as a gap, never as a zero and never joined across.
- **Dates are UTC.** Local time needs a timezone database, which this library does not carry. Date
  axes are x axes.
- **Scatter above 1,000 points is one `<path>`**, not one element per point. It looks identical; it
  is what keeps a large scatter openable. A million-point scatter is still a 61 MB document — that
  size belongs in a PNG.
- **Labels are measured by the backend.** An SVG is laid out with Helvetica's widths, because its font
  is whatever the viewer's browser picks for `sans-serif`; a different font renders slightly wider or
  narrower. A PNG is laid out with the widths of the font it draws, DejaVu Sans, so its gutters are
  exact. The same figure's SVG and PNG can therefore place a gutter a few pixels apart.

---

## Deliberately not included

- **Interactive windows, zooming and animation** — there is nothing in Bantu to hand a window to.
- **A raw-SVG escape hatch** — see the security section; it is the one feature whose absence is the
  point.
- **`jet`** — a colormap that invents boundaries in smooth data and disappears in greyscale.
- **3-D axes, TeX-style maths in labels, PDF output, and local time zones.**

---

## Caveats, plainly

- **PNG text is one font.** DejaVu Sans, embedded: ASCII, Latin-1 and the symbols charts use
  (`° µ × − – — … ± ≤ ≥ ≈ ≠ ∞ € ™ ☀`). Anything else draws as `�`. No bold, no italic.
- **PNG rotation snaps to 1.4° steps.** 0°, 45° and 90° are exact; 30° draws at 29.5°.
- **A PNG has an opaque background.** No transparency in this release.
- **SVG fonts are estimated, not loaded.** SVG layout uses a built-in table of Helvetica widths.
- **Charts are drawn when saved.** Nothing is rendered until `savefig`, `show` or `to_svg`, so an error
  in the data — a zero on a log axis — surfaces there rather than at the `plot` call.
- **It is not matplotlib.** The names and the tick choices match; the thousands of options do not. If
  a matplotlib option is missing, the dict key is silently ignored rather than rejected.
