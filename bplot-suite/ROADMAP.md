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

## Phase B1 — The core: Figure, Axes, scales, ticks, the SVG backend

- [ ] `BPlotFigure` / `BPlotAxes` / `BPlotLine` / `BPlotText` / `BPlotSvg`, factory functions
- [ ] the four coordinate spaces and the single y-flip at the backend boundary
- [ ] linear scale, `MaxNLocator` ticking, `_fmt` and the scientific/engineering formatters
- [ ] the SVG backend: nine primitives, mandatory escaping, `<clipPath>` per Axes
- [ ] path simplification (BP3)
- [ ] `plot` `scatter` `bar` `barh`; `xlabel` `ylabel` `title` `legend` `grid` `xlim` `ylim`
- [ ] `savefig`, `show`, `plt.help()`
- [ ] `tests/bplot_core_test.b`, `tests/bplot_stress.sh`

**Gate:** emitted SVG parses as well-formed XML; golden-file comparison; ticking matches
matplotlib's choices across ~20 ranges including negatives, tiny and huge spans.
**Stress:** a 100k-point line (with and without simplification, both wall-clock gated); empty series;
a single point; all-equal values; NaN and ±inf in the data; a title of
`</text><script>alert(1)</script>` — each renders or raises clearly, and **never emits malformed
SVG**.

---

## Phase B2 — The chart types

- [ ] `hist` `boxplot` `violin` `errorbar` `fill_between` `step` `stem` `pie`
- [ ] log and symlog scales, with decade and minor ticks
- [ ] date axes (arctic's datetime columns as x)
- [ ] annotations, arrows, text rotation
- [ ] `tests/bplot_charts_test.b`

**Gate:** each chart type against a golden file; histogram bin edges against numba's `nd_histogram`.
**Stress:** log scale fed zero and negative values (must raise, naming the value); a single-bin
histogram; a boxplot of one point; degenerate ranges; a hostile annotation string.

---

## Phase B3 — Layout and 2-D

- [ ] `subplots`, `GridSpec`, `twinx`/`twiny`, shared axes
- [ ] `tight_layout` using the embedded Helvetica metrics (BP12)
- [ ] colormaps (`viridis` `plasma` `coolwarm` `gray`), colorbars
- [ ] `imshow` `contour` `pcolormesh` `heatmap`, taking numba 2-D arrays
- [ ] style sheets
- [ ] `tests/bplot_layout_test.b`

**Gate:** an 8×8 subplot grid lays out without overlap; `tight_layout` on labels long enough to
collide; colormap values against reference tables.
**Stress:** a 1000×1000 `imshow` with a wall-clock gate **and a file-size gate** (the naive encoding
is one `<rect>` per pixel — a 60 MB document; the correct one is a single embedded image).

---

## Phase B4 — Data integration

- [ ] accept Bantu lists, numba `ndarray`s, arctic `Column`s, `Series` and `DataFrame` everywhere
- [ ] `df.plot()` on an arctic DataFrame
- [ ] native reductions for data limits and binning when numba is present
- [ ] `tests/bplot_data_test.b`

**Gate:** **identical output from all four input types for the same data.** Mismatched lengths raise
clearly, naming both.
**Stress:** a 1M-row arctic column plotted end to end; a column containing nulls; a DataFrame with a
non-numeric column selected.

---

## Phase B5 — Package, docs, gallery

- [ ] `bplot/{bplot.b, bplot_test.b, package.json}`, `bantu publish ./bplot`
- [ ] `docs/bplot.md` — quickstart, tour, measured numbers, the SVG security caveat, blunt closing
      caveats
- [ ] `samples/bplot/` — a line chart, a histogram, a heatmap, a dashboard served through `sua`
- [ ] `tests/run_samples.sh` picks them up

**Gate:** every documented example and every sample executed by CI; `bantu add bplot` then
`include "bplot" as plt` works from a clean project; the sua example serves a real chart.

---

## Phase B6 — The native raster backend

- [ ] `bp_*`: scanline anti-aliased polygon fill, stroke-to-path, deflate, CRC32, PNG
- [ ] **binary-safe file writes** — `open()`/`writefile()`/`appendfile()` accept `"wb"/"rb"/"ab"` and
      set `std::ios::binary`; today `open(path,"wb")` falls through the mode chain and silently opens
      the file for *reading*
- [ ] `savefig("x.png", {"dpi": 150})`

**Gate:** PNG validated by an external decoder; **byte-identical output on Linux, macOS and
Windows** — the test that proves the binary-mode fix and that cannot pass by accident.
**Stress:** 4000×3000 at 300 dpi for memory and time; ASan over the rasteriser with degenerate
polygons, zero-width strokes and out-of-canvas coordinates.

---

## Deferred, each needing its own justification

Interactive output (there is nothing to hand a file to — BP14) · PDF backend · 3-D axes · animation ·
font file parsing · TeX-style math text · `jet` (BP13, deliberately never).
