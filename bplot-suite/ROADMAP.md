# bplot — Roadmap

A matplotlib-shaped plotting library for Bantu. Everything from the object model to the SVG backend
is **pure Bantu**; only the raster backend (B6) is native, because rasterising a million pixels in an
interpreted language is minutes per figure. Design and rationale live in
[`docs/bplot-architecture.md`](../docs/bplot-architecture.md) and [`DECISIONS.md`](DECISIONS.md).

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done (feature **and** stress tests green)

**Every phase is gated by five tiers, all green before it advances** — the same gate numba uses:

| tier | proves | form |
|---|---|---|
| 1. feature | every call does what it claims, including its failure modes | `tests/bplot_*_test.b`, ending `RESULT: ALL GREEN` |
| 2. differential | the output is *right*, not merely stable | well-formed-XML check, golden files, ticks compared against matplotlib's own choices, libm for the maths builtins |
| 3. **stress** | it survives size, repetition and abuse | scale · degenerate input · hostile input · memory |
| 4. regression | nothing else broke | the whole existing suite (lang, scope, crypto, uuid, random, orm, arctic, **numba**, sua, webpush) on the same build |
| 5. sanitizers | no latent memory defect | ASan + UBSan over any native code the phase adds |

Results recorded in [`CHANGELOG.md`](CHANGELOG.md).

**Four rules that apply to every phase.**
1. Never assert on a stringified number — `str()` gives six significant digits and leaks scientific
   notation. The one exception is `_fmt`'s own tests, where the string *is* the contract.
2. Never build output with `$s = $s + …` — it is O(n²) (measured: 1,116 ms for 20k appends, 6,752 ms
   for 40k). Push onto a list, `join` once.
3. Never use the C-style `for` above 100,000 iterations — `evaluator.hpp` caps it at
   `safety < 100000` and exits **silently**. Use `while`.
4. Every text node and attribute value is escaped by the backend, always (decision BP7). A test that
   feeds a hostile label rides along with every phase that adds a text-bearing feature.

---

## Phase B0 — Tracking docs and the language gaps ✅

The design work found three defects and one missing primitive in the language itself. All four are
general — they are not bplot features — so they land first and benefit every Bantu program.

- [x] `docs/bplot-architecture.md`
- [x] `bplot-suite/ROADMAP.md`, `DECISIONS.md`, `CHANGELOG.md`

| Item | How | Feature test | Stress test | Status |
|---|---|---|---|---|
| **[found] `max(1,2,9)` returns 2** | `max`/`min` read `args[0]` and `args[1]` and ignore the rest. Now variadic, and a single list argument is reduced over | ✅ `tests/lang_math_test.b` | ✅ 1 to 64 arguments, a 100k-element list, NaN in every position, empty list, non-numbers | [x] |
| **[found] `str(1e21)` returns `-9223372036854775808`** | `Value::toString` casts any integral double to `long long`; out of range that is undefined behaviour (ISO C++ [conv.fpint]) | ✅ `tests/lang_number_test.b` | ✅ every power of ten to 1e308, both signs, the exact 2^63 boundary, ±inf, NaN | [x] |
| **[found] no `join`, so string building is O(n²)** | `join(list, sep)` — the missing inverse of `split`, reserving the exact total length | ✅ `tests/lang_math_test.b` | ✅ 100k parts, empty list, one element, non-string elements, a 10 MB result; linear scaling asserted | [x] |
| Scalar maths surface | `PI E TAU INF NAN` · `exp expm1 log1p log2 log10 cbrt` · `asin acos atan atan2 hypot` · `sinh cosh tanh asinh acosh atanh` · `trunc sign fmod clamp degrees radians` · `isnan isinf isfinite` | ✅ `tests/lang_math_test.b` | ✅ differential against libm incl. domain edges, ±inf, NaN, denormals, and the identities each function must satisfy | [x] |

**Gate:** `tests/lang_math_test.b` ALL GREEN; every new builtin differential-tested against libm
including domain edges, ±inf and NaN; full existing regression green (these touch shared machinery).

---

## Phase B1 — The core: Figure, Axes, scales, ticks, the SVG backend ✅

- [x] `BPlotFigure` / `BPlotAxes` / `BPlotSvg`, reached through `figure()` — `new alias.Class()`
      does not parse across a module boundary
- [x] the coordinate spaces, with the y flip in exactly one place (`coefY`'s sign)
- [x] linear scale, `MaxNLocator` ticking, `_fmt`
- [x] the SVG backend: `rect` `line` `circle` `polyline` `text` `group` `clip`, mandatory escaping,
      one `<clipPath>` per axes
- [x] path simplification (BP3)
- [x] `plot` `scatter` `bar` `barh`; `title` `xlabel` `ylabel` `legend` `grid` `xlim` `ylim`
- [x] `savefig`, `show`, `clf`, `to_svg`, `plt.help()`
- [x] `bplot/{bplot.b, bplot_test.b, package.json}`, `samples/bplot/`
- [x] `tests/bplot_core_test.b` (149), `tests/bplot_stress.sh` (20), both in CI

| gate | result |
|---|---|
| emitted SVG parses as well-formed XML | ✅ **19 documents**, every degenerate and hostile case, checked with a real XML parser |
| ticking matches matplotlib's own choices | ✅ 12 ranges asserted exactly, incl. negatives, 1e-9 and 1e9 |
| the top tick is not lost to drift | ✅ ticks computed as `k × step`, not accumulated |
| escaping has no opt-out | ✅ `</text><script>` in title, both axis labels and a legend entry |
| control bytes | ✅ stripped, so the document still parses |
| unicode | ✅ survives intact |
| NaN / ±inf | ✅ never reach a coordinate in any document; NaN splits the line into runs |
| 100k-point line | ✅ **1,136 ms**, one polyline, **25,782 bytes** against 1,386,925 unsimplified (**54×**) |
| repetition | ✅ 2,000 figures built and dropped |
| empty / single point / all-equal / all-NaN | ✅ each renders; none emits malformed SVG |
| bad input | ✅ mismatched lengths, null, wrong type, zero size, inverted limits, `.png` — all raise, naming the problem |
| full regression | ✅ 37 `.b` suites, 13 `.sh` suites |

**Two defects found while building it**, both of them Bantu's documented landmines biting their own
author:
- **Module-level state does not survive assignment from inside a function.** `$_CUR = figure()` in
  `gcf()` created a function-*local*, so every call built a new figure and the chart came out empty.
  The current figure lives in a dict now — dicts are reference-semantic, so writing through a field
  mutates the object every caller can see.
- **A list passed to a function is a copy.** `_emitCol($out, …)` pushed into a copy and the caller
  saw nothing, so path simplification silently produced no points and a 20,000-point line rendered
  as no line at all. The helper returns its points and the caller `extend`s.

**Deferred from B1 with a reason:** `tight_layout` uses the embedded Helvetica metrics but belongs
with subplots (B3), so gutters are fixed for now; the metrics table is in and used for legend boxes.

## Phase B2 — The chart types ✅

- [x] `hist` `boxplot` `violin` `errorbar` `fill_between` `step` `stem` `pie`
- [x] log and symlog scales, with decade and minor ticks
- [x] date axes (epoch milliseconds, UTC — arctic's own datetime storage)
- [x] annotations, arrows, text rotation
- [x] categorical axes (a string sequence becomes positions and tick labels — BP28)
- [x] `tests/bplot_charts_test.b` (196), B2 additions to `tests/bplot_stress.sh`

| gate | result |
|---|---|
| symlog transform vs `matplotlib.scale.SymmetricalLogTransform` | ✅ 17 points, 3 `linthresh` settings, to 9 decimals |
| log ticks vs matplotlib | ✅ 8 ranges exact; **2 ranges deliberately differ** (BP17 — matplotlib leaves the axis unlabelled) |
| symlog ticks vs matplotlib | ✅ 6 limit/linthresh combinations |
| date ticks vs `AutoDateLocator` | ✅ **8 ranges exact**, ten seconds to 25 years, including the semi-monthly and 4-year cases |
| calendar arithmetic | ✅ 1,461 consecutive days round-trip; 1900 and 2000 leap rules |
| quantiles vs numpy · boxplot vs `cbook.boxplot_stats` | ✅ exact |
| histogram binning vs `nd_histogram` / numpy | ✅ including the **inclusive top edge**, which otherwise loses the maximum silently |
| log axis fed zero or a negative | ✅ raises, names the value, suggests `symlog` |
| every B2 document parses as XML | ✅ **32 documents**, every degenerate and hostile case |
| escaping through annotations, group labels and pie slices | ✅ no `<script`, no `<foreignObject` |
| 200,000 points through `hist` and `violin` | ✅ 1,103 ms / 6,959 bytes and 1,971 ms / 5,291 bytes — output does not scale with input |
| 100k-point line, linear scale | ✅ **1,198 ms** against B1's 1,136 — the scale machinery costs a linear axis nothing (BP16) |

**One defect found and fixed while building it:** a bar's zero baseline is published as part of its
data bounds, so a log axis raised on the *baseline* before any data was projected — making `bar()`
and `hist()` unusable on a log scale. Baselines are now clamped to the view floor while real
non-positive **data** still raises, and the difference is deliberate: a line with a gap is invisible
and misleading, an absent bar already reads as zero.

---

## Phase B3 — Layout and 2-D ✅

- [x] `subplots`, `GridSpec`, `subplot(r,c,i)`, spanning cells, `twinx`/`twiny`, shared axes
- [x] `tight_layout` using the embedded Helvetica metrics (BP12)
- [x] colormaps (`viridis` `plasma` `coolwarm` `gray`), colorbars
- [x] `imshow` `contour` `pcolormesh` `heatmap`
- [x] style sheets (`default`, `dark`, `print`)
- [x] `tests/bplot_layout_test.b` (132), B3 additions to `tests/bplot_stress.sh`

| gate | result |
|---|---|
| `subplots()` and `subplot(r,c,i)` rectangles | ✅ **identical**, asserted cell by cell (BP27) |
| an 8×8 grid | ✅ 64 panels, none overlapping, all inside the figure |
| a 1×1 grid | ✅ reproduces B1's fixed gutters exactly, so B1 output did not move |
| `tight_layout` | ✅ wider labels reserve more gutter; every box stays inside its cell; **rendering twice is byte-identical** |
| colormap entries vs matplotlib | ✅ 20 sampled positions + 7 individual entries, written into the test independently |
| 1000×1000 `imshow` | ✅ **1,473,633 bytes and 256 elements** for 1,000,000 cells — against ~55 MB for one `<rect>` each |
| twin axes | ✅ same rectangle, **frame stroked once**, independent y scales |
| shared axes | ✅ union limits, and sharing is **transitive** |
| 6,000 figures built and dropped | ✅ **RSS flat** — 10.5 MB at 300 figures, 12.0 MB at 6,000 |
| every B3 document parses as XML | ✅ 16 documents, including every style |
| the dark style | ✅ no hardcoded white survives anywhere in the document |

**Two defects found and fixed while building it, both of them general:**
- **Every Bantu object leaked.** `new ClassName()` allocated an instance nothing ever deleted —
  ~45 KB per figure, 372 MB over 20,000 figures and climbing. Instances are refcounted now, and an
  Axes deliberately holds no pointer back to its Figure, because refcounting does not collect cycles
  (BP30). The gate is an RSS measurement, not an inspection.
- **`&&` and `||` did not short-circuit**, so the universal guard `if ($i < len($a) && $a[$i] == x)`
  died on exactly the boundary it was written to prevent. Fixed, measured at no cost.

**Gate:** an 8×8 subplot grid lays out without overlap; `tight_layout` on labels long enough to
collide; colormap values against reference tables.
**Stress:** a 1000×1000 `imshow` with a wall-clock gate **and a file-size gate**.

> **This gate was corrected, and the original wording is kept so the change is visible.** It read:
> "the naive encoding is one `<rect>` per pixel — a 60 MB document; the correct one is a single
> embedded image". The first half is right and measured. The second half is **not reachable in B3**:
> a single embedded image means `<image href="data:image/png;base64,…">`, which means a PNG encoder —
> CRC32, Adler-32 and deflate — which is the native work that *is* B6. B3 instead block-reduces to a
> cell budget and batches cells into one `<path>` per colour (decision BP26), which is ≤ 256 elements
> and ~14 bytes per cell rather than ~55. B6 inherits the single-image gate, where it belongs.

---

## Phase B4 — Data integration ✅

- [x] accept Bantu lists, numba `ndarray`s, arctic `Column`s, `Series` and `DataFrame` everywhere
- [x] `df.plot()` on an arctic DataFrame — plus `$series.plot()`, `plot_frame($df)` and `heatmap($df)`
- [x] native reductions for data limits and binning when numba is present — and native line
      simplification and scatter batching, which turned out to be where the time was (BP32, BP33)
- [x] `tests/bplot_data_test.b` (138), B4 additions to `tests/bplot_stress.sh` (48 → 62)

**Gate:** **identical output from all four input types for the same data.** Mismatched lengths raise
clearly, naming both.
**Stress:** a 1M-row arctic column plotted end to end; a column containing nulls; a DataFrame with a
non-numeric column selected.

| gate | result |
|---|---|
| identical output from a list, an ndarray, a column and a Series | ✅ **byte-identical** across 17 chart configurations — line, simplified line, scatter, batched scatter, hist, density hist, boxplot, violin ×2, bar, step, stem, fill_between, errorbar, log line, symlog line and symlog scatter |
| the native path against the pure-Bantu path | ✅ **byte-identical** on every one of those; and every B1–B3 assertion, written against lists, passes through the native path |
| mismatched lengths name both | ✅ `x ('date') has 5 points and y ('price') has 4` — on both paths |
| **a 1,000,000-row arctic column as a line** | ✅ **252 ms, 112 MB peak RSS** — against **15,455 ms and 2.69 GB** through the pure path on the same build. The document is **23,825 bytes both ways** |
| 1,000,000 rows through hist / two boxes / a violin | ✅ 61 ms / 241 ms / 306 ms |
| a 1,000,000-point scatter | ✅ 1,523 ms, one `<path>` of 60.9 MB — at that size the raster backend (B6) is the right format |
| a column containing nulls | ✅ a null draws exactly the gap a NaN draws, from a list, a column and a Series; an all-null column renders an empty, valid chart |
| a DataFrame with a non-numeric column selected | ✅ raises catchably, naming the column and its type, and the same process then draws the frame correctly |
| hostile column names | ✅ escaped wherever `plot_frame` writes them |
| every B4 document parses as XML | ✅ |

**Seven defects found and fixed while building it — five of them in the language, all silent:**
- **`include "bplot"` bound an empty module** when run from a folder containing a `bplot/` directory,
  because the resolver accepted a directory as a file. That is this repository's own layout.
- **`len()` answered 0 for a dict, an ndarray and a column**, so a loop bounded by it never ran.
- **`contains()` answered false for every list.**
- **Date axes at hour and minute resolution carried no date** — `00:00 12:00 00:00` — though the
  architecture doc had specified a dated label. They follow matplotlib's formats now.
- **Numeric data was materialised as Bantu lists**, costing ~20 s and 3.1 GB for one million-point
  chart. That is §13 of the architecture doc.
- A Bantu list containing `null` could not be plotted at all; it is a gap now, consistently with a
  null in a column.
- `nd_to_list` of a 2-D ndarray passed to `plot` produced nested lists that failed far from the
  cause; it raises at the boundary now, naming the shape.

---

## Phase B5 — Package, docs, gallery ✅

- [x] `bplot/{bplot.b, bplot_test.b, package.json}` at **1.1.0**, `bantu publish ./bplot`
- [x] `docs/bplot.md` — three lines to a chart, a tour, serving from sua, the SVG security section,
      measured numbers, blunt closing caveats
- [x] `samples/bplot/` — a line chart, the chart types, statistics, a dashboard with a heatmap, a
      DataFrame, and `server.b`, charts served through `sua`
- [x] `tests/run_samples.sh` picks them up; `tests/run_doc_examples.sh` executes every documented
      example

**Gate:** every documented example and every sample executed by CI; `bantu add bplot` then
`include "bplot" as plt` works from a clean project; the sua example serves a real chart.

| gate | result |
|---|---|
| every documented example executed | ✅ `tests/run_doc_examples.sh` runs each doc as **one program, in reading order**: `docs/bplot.md` (8 examples), `docs/numba.md` (9), `docs/arctic.md` (1) |
| every sample executed | ✅ `tests/run_samples.sh`, and `server.b` by `tests/bplot_sua_test.sh` |
| `bantu add bplot` → `include "bplot"` in a clean project | ✅ `tests/bplot_package_test.sh`, 17 checks, in a throwaway `HOME`: publish, add, include, a DataFrame drawn through arctic's lazy include from `bantu_modules/`, the installed copy byte-identical to its source, the installed smoke test green, and arctic without bplot naming `bantu add bplot` |
| the sua example serves a real chart | ✅ `tests/bplot_sua_test.sh`, 15 checks: `image/svg+xml` with a CSP and `nosniff`, parses as XML, a hostile query-string title escaped, **40 concurrent requests each receiving only its own chart**, and still serving afterwards |
| CI green on three platforms | ⚠️ **wired, not yet observed.** The new gates are in the Linux and macOS jobs and the Windows job now runs bplot's smoke test, but no CI run has happened on this branch — carried forward, as it has been since B1 |

**Three defects found and fixed while building it:**
- **numba's documented examples had never been run.** Its roadmap recorded "every documented example
  executed by CI" as its gate; nothing executed them, and two of nine were broken — a reduction over
  axis 2 of a two-dimensional array, and a loop over arrays its block never defined. Both are fixed,
  and the runner executes them from now on.
- **`num()` read a prefix and swallowed overflow**: `num("12abc")` was 12, `num("1e999")` was 0 and
  `num(true)` was 0 — plausible wrong numbers in exactly the code that parses a query string. Found
  while preparing the serving example.
- **A test comment described sua's threading wrongly**, saying every connection got its own detached
  thread. Handlers run one at a time on the event loop; the comment described legacy code nothing
  constructs. Corrected, because anyone reasoning about thread safety from it would reach the wrong
  answer.

---

## Phase B6 — The native raster backend

**Design:** [`docs/bplot-raster-architecture.md`](../docs/bplot-raster-architecture.md) — what byte
identity rules out, the integer rasteriser, the embedded font, the encoder, and the six steps B6a–B6f.

- [ ] `bp_*`: scanline anti-aliased polygon fill, stroke-to-path, deflate, CRC32, PNG
      - [x] **B6b** — the canvas, rectangle fills with exact fractional coverage, deflate with
            dynamic Huffman codes, CRC-32, Adler-32 and the PNG encoder
            (`tests/bplot_raster_test.b`, `tests/bplot_png_test.sh`)
      - [x] **B6c** — the rasteriser: polygons, strokes with caps and joins, dashes, circles, arcs
            and the path parser (`bp_fill_polygon`, `bp_stroke_polyline`, `bp_fill_path`)
      - [ ] B6d — text: the embedded font, glyph rendering, rotation, measurement
      - [ ] B6e — `BPlotRaster`, `savefig(".png", {dpi})`, `to_png()`, PNG from sua
- [ ] **convert the authoring-time generators to Bantu** — `scripts/gen_circle_tables.py` is a direct
      translation; `scripts/gen_font_tables.py` needs a Bantu TrueType reader (`glyf`, `loca`, `cmap`,
      `hmtx`), which binary file reads made possible in B6a. Neither is part of the build, and the
      Python in `tests/` stays: its value is being a decoder we did not write
- [x] **binary-safe file writes** — `open()`/`writefile()`/`appendfile()` accept `"wb"/"rb"/"ab"` and
      set `std::ios::binary`; today `open(path,"wb")` falls through the mode chain and silently opens
      the file for *reading* — **done as B6a**, with `readfile()` too, unknown modes raising instead of
      opening for reading, and failed writes raising instead of reporting success
      (`tests/lang_file_test.b`)
- [ ] `savefig("x.png", {"dpi": 150})`

**Gate:** PNG validated by an external decoder; **byte-identical output on Linux, macOS and
Windows** — the test that proves the binary-mode fix and that cannot pass by accident.
**Stress:** 4000×3000 at 300 dpi for memory and time; ASan over the rasteriser with degenerate
polygons, zero-width strokes and out-of-canvas coordinates.

---

## Deferred, each needing its own justification

Interactive output (there is nothing to hand a file to — BP14) · PDF backend · 3-D axes · animation ·
font file parsing · TeX-style math text · `jet` (BP13, deliberately never).
