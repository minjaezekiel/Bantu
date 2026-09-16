# bplot — Changelog

Every entry is prefixed so the log can be grepped: **[design]**, **[feature]**, **[bug fix]**,
**[perf]**, **[test]**, **[docs]**.

Phases do not advance until all five gate tiers are green (see [`ROADMAP.md`](ROADMAP.md)). Measured
numbers are recorded here with the machine they were taken on, because a number without a machine is
not a measurement.

> Unless another machine is named: **Intel Core i7-9750H @ 2.60 GHz, macOS 15.7.9, Apple clang 17**,
> `bantu` built by `build-mac.sh`.

---

## Phase B0 — Tracking docs and the language gaps

- **[design]** `docs/bplot-architecture.md` — the reference design: why the object model, scales,
  ticking, layout and the SVG backend are pure Bantu while only rasterisation is native; the
  measured string-building problem and the two fixes for it; the Figure/Axes/Artist model and the
  three Bantu constraints that shape it; coordinates and the single y-flip; scales; the
  `MaxNLocator` tick algorithm with the two details that are not optional; number formatting; the
  SVG backend and **the security section that is the real teeth of this design**; text metrics; NaN
  and degenerate-input handling; the raster backend; interop; and every rejected alternative with
  its reason.
- **[design]** `bplot-suite/DECISIONS.md` — BP1–BP15.
- **[design]** `bplot-suite/ROADMAP.md` — phases B0–B6, each with its feature test, its stress test
  and its gate.

### Findings recorded during design, before any code

The design work was supposed to confirm that bplot needed **zero** interpreter changes. Measurement
disproved that, and found three defects on the way. All four are general language problems, not
bplot features, so all four are fixed here rather than worked around.

- **Building a string is O(n²).** `$s = $s + $part` in a loop: **20,000 appends 1,116 ms, 40,000
  appends 6,752 ms** — 6.05× the time for 2× the work, because `evalBinaryOp` builds a fresh
  `std::string` holding a copy of the left operand every time. Forty thousand appends to a growing
  1.16 MB buffer is ~23 GB of `memcpy`. A 100,000-element figure would have taken upwards of forty
  seconds, and the B1 stress gate would have failed on the first realistic chart.
- **`max(1, 2, 9)` answers 2.** `max` and `min` read `args[0]` and `args[1]` and ignore everything
  after. Not a crash — a **silently wrong answer**, which is the failure mode worth caring about,
  and it has been in every shipped build.
- **`str(1e21)` answers `"-9223372036854775808"`.** `Value::toString` casts any integral double to
  `long long`; converting a floating-point value outside the destination integer range is undefined
  behaviour (ISO C++ [conv.fpint]) and saturates to `INT64_MIN` on x86-64 and ARM64. Reachable from
  `pow(10, 21)`, and reachable from any numba reduction whose total exceeds 2⁶³.
- **`$a[$i]` is O(n) in the length of `$a`.** `evalNode` returns a `Value` by value, and a `Value`
  holding a list owns its elements inline — reading one element out of a 20,000-element list
  deep-copied all 20,000, each a ~190-byte struct carrying a `std::string`, a `std::vector`, a
  `std::function` and three `shared_ptr`s. **10,000 reads 1,919 ms, 20,000 reads 8,093 ms** — 4.2×
  for 2× the work, 405 µs to read one element. Every loop over a list in every Bantu program was
  quadratic. This is the same class as the `push` defect found in numba's Phase A, in the same type,
  for the same reason.

### Solved defects

- **[bug fix] `max`/`min` ignored every argument after the second.** Both are now variadic, and a
  single list argument is reduced over it. `max(1, 2, 9)` is 9; `max([3, 17, 5])` is 17. The
  two-argument form is unchanged. NaN propagates (NumPy's `max`, not its `nanmax`) because a
  primitive must not silently discard a value it was handed; bad arguments raise a catchable error
  naming the element and its type, where before a string argument was quietly read as 0.
  *Side benefit:* `max($list)` over 100,000 elements is **13 ms** natively, against a Bantu loop
  that does the same work in ~100 ms — which is how bplot takes the range of a large series.

- **[bug fix] `str()` past 2⁶³ was undefined behaviour.** The integral fast path is now range-guarded
  at 9.2e18, below 2⁶³, and larger magnitudes fall through to the stream. `str(1e21)` is `"1e+21"`,
  `str(1e308)` is `"1e+308"`, and `num(str($x))` round-trips at every magnitude. `inf`, `-inf` and
  `nan` are unaffected — they never took the integral path.

- **[feature] `join(list [, sep])`** — the missing inverse of `split()`, and the fix for the O(n²)
  problem. It reserves the exact total length for the all-strings case and appends once. Non-string
  elements stringify as `print` would; a missing separator means `""`; a non-list first argument or
  a non-string separator raises.

  | | `$s = $s + part` | `push` + `join` |
  |---|---|---|
  | 20,000 parts | 1,116 ms | **147 ms** collect + **2 ms** join |
  | 40,000 parts | 6,752 ms | **147 ms** collect + **7 ms** join |
  | 100,000 parts (2.9 MB) | ~42 s extrapolated | **327 ms** collect + **38 ms** join |

  **42× faster at 40,000 parts**, and linear rather than quadratic.

- **[bug fix] `$a[$i]` copied the whole list to read one element.** `evalIndexAccess` now borrows a
  pointer to the live container instead of evaluating it into a temporary, for the cases where that
  is provably safe: the index expression must be a literal or a variable, so if the borrow fails
  half-way the fallback path can re-evaluate it without any observable effect. Index chains
  (`$m[1][2]`) are borrowed recursively.

  | | before | after | |
  |---|---|---|---|
  | 10,000 reads | 1,919 ms | **26 ms** | 74× |
  | 20,000 reads | 8,093 ms | **52 ms** | **156×** |
  | 100,000 reads | ~3.4 min extrapolated | **282 ms** | |

  Linear, where it was quadratic. Deliberately **not** the existing `resolveLValue`, which the
  assignment paths use: its dict branch is `(*objectVal)[key]`, which *creates* the entry on lookup —
  correct for `$d["new"] = 1` and wrong for a read, where it would quietly insert a null every time
  you looked up a key that was not there. The borrowing path uses `find()` and reports failure, and
  a test asserts that a missing key is not inserted. List *semantics* are untouched: `$b = $a;` is
  still a copy.

- **[feature] The scalar maths surface.** `PI TAU E INF NAN` as constants; `exp expm1 log1p log2
  log10 cbrt asin acos atan atan2 hypot sinh cosh tanh asinh acosh atanh trunc sign fmod copysign
  degrees radians clamp isnan isinf isfinite` as functions. Domain and range behaviour is IEEE 754's
  — `acos(2)` is NaN rather than an error, `log(0)` is `-inf`, NaN propagates — and a **non-number
  argument raises**, naming the argument and its type. The older maths builtins (`sqrt`, `sin`,
  `log`, …) read `args[0].numberVal` blind, so `sqrt("hello")` answers 0; they are shipped behaviour
  and are left alone, and the difference is documented.

  `isnan`/`isinf`/`isfinite` close a real gap: these values were always *producible* (`log(0)`,
  `sqrt(-1)`, `pow(10,400)`) and there was no way to *test* for one — which for a plotting library
  means emitting a NaN coordinate, which browsers render as nothing at all.

- **[patch] `numba/numba.b` re-exports the language constants** rather than defining its own. `$INF`
  and `$NAN` were built by dividing one-element arrays, so importing numba allocated two arrays
  before doing anything.

- **[patch] `tests/arctic_lazy_test.b` renamed `$exp` to `$explain`.** Variables and functions share
  one namespace with the `$` stripped, so `$exp = …` replaced the new `exp()` builtin for the rest of
  that program. Harmless there — it never called `exp()` — and a landmine for the next edit.

### Tests

- **[test]** `tests/lang_math_test.b` — **174 assertions, ALL GREEN.** Each function is checked
  against an *identity* it must satisfy (`exp(log(x)) == x`, `sin² + cos² == 1`, `cosh² − sinh² == 1`,
  `asin(sin(x)) == x`, `degrees(radians(x)) == x`) rather than against constants copied from
  somewhere, because copied constants only prove the copying worked. Plus the exact values where
  libm implementations differ, the four quadrants of `atan2`, `hypot(1e200, 1e200)` staying finite
  where the naive form overflows, every domain edge (`asin(2)`, `acosh(0.5)`, `atanh(1)`, `log(0)`,
  `exp(±1000)`), NaN propagation through everything, and the failure modes — a string argument, an
  empty list, an inverted `clamp` range — each of which must *raise*, not answer.

  Numbers are asserted with a relative tolerance, never through `str()`, which gives six significant
  digits and would hide a 1e-7 relative error. The exception is `str()`'s own tests, where the string
  is the contract.

  The three defects each have their own block, including linearity assertions for `join` and for
  list indexing that fail if either becomes quadratic again.

- **[test]** Full regression: **33 `.b` suites + 12 `.sh` suites + the `const_bad` negative fixture,
  all green**, on the same build. These changes touch `Value::toString`, `evalIndexAccess` and two
  shipped builtins, so the whole tree is the gate, not just the new test.

- **[test] ASan + UBSan, clean.** A `-fsanitize=address,undefined -fno-sanitize-recover=undefined`
  build over the language, numba and arctic suites: **no sanitizer report anywhere**. This tier
  earns its place here more than usual, because the index fix holds a raw `Value*` across
  evaluation. A hand-written adversarial script exercises the shapes that could break it — an index
  expression with a side effect, one that reassigns the base, one that pushes 5,000 elements onto
  the base *while it is being indexed* (reallocating the very buffer being borrowed), three-deep
  chains, a dict of lists of dicts, a closure over a local list, a list inside a class instance, and
  a self-referential list. All correct, all clean.

  `numba_linalg_test.b` reports one failure under the sanitizer build — the "1000³ matmul reaches
  4 GFLOP/s" gate — because ASan instrumentation at `-O1` is several times slower than the `-O2`
  binary. It is a throughput gate, not a correctness one; all 68 correctness assertions pass.

### Findings from the verification itself

- **The borrow had to be taken *after* the index expression, not before.** `$c[grow()]`, where
  `grow()` pushes 5,000 elements onto `$c`, reallocates the list's buffer while the index is being
  computed. Borrowing first would have left a pointer into the freed buffer — a use-after-free
  reachable from ordinary script. Evaluating the index first costs nothing and cannot be wrong.
  (`unordered_map` keeps element addresses stable across a rehash, so the scope-chain half of the
  borrow was safe either way; the list buffer is not a `unordered_map` and was not.)

- **[perf] The hot-path benchmark has a ~2.7% position bias, which is larger than the ±2% gate.**
  Running the baseline binary first and the new one second showed the 1M arithmetic loop **2.7%
  slower** in all three rounds — consistent enough to look real. Reversing the order flipped it: the
  new binary measured *faster* in two of three rounds. Best-of across both orderings: **2,593 ms new
  against 2,594 ms baseline**, a difference of 0.04% on a loop neither change touches.

  This confirms, a second time and by a different route, that the ±2% benchmark gate is below this
  harness's noise floor unless the A/B ordering is controlled. **Any future hot-path gate must run
  both orderings**; a single ordering will manufacture a 2–3% result in whichever direction the
  binaries happened to be run.

  Order-controlled, and the only figure that survives it:

  | hot path | baseline | after | |
  |---|---|---|---|
  | 1M arithmetic while loop | 2,594 ms | 2,593 ms | unchanged |
  | 1M comparison loop | 2,459 ms | 2,470 ms | +0.4%, within noise |
  | 500k unary negation | 1,174 ms | 1,197 ms | +2.0%, within noise |
  | `fib(24)` recursive | 1,915 ms | 1,904 ms | −0.6%, within noise |
  | **200k list index read** | **635 ms** | **584 ms** | **8.0% faster** |
  | 100k list index write | 211 ms | 212 ms | unchanged |
  | 50k dict set | 179 ms | 176 ms | unchanged |
  | 100k string concat | 181 ms | 185 ms | +2.2%, within noise |

  (Best-of-three per ordering, then best across both orderings — the floor, not the mean, since a
  mean measures the machine's background as much as the interpreter. The rows marked "within noise"
  are inside the 2.7% position bias this harness demonstrates on a loop neither change touches.)

  The list-index gain holds in **both** orderings (635→595 and 641→584), which is what distinguishes
  it from the arithmetic figure that did not. Note this benchmark indexes a *short* list, so it
  measures only the removed double scope-chain walk and the removed cast; the 74–156× figures above
  are what the same fix does to a list long enough for the copy to dominate.

### Verification still outstanding

- **Cross-platform.** macOS only, as for every numba phase. CI runs Linux and Windows jobs and they
  pick up `tests/*.b` automatically, but no Linux or Windows machine and no Docker is available here,
  so **CI has not been observed**. Stated rather than assumed.

---

## Phase B1 — Figure, Axes, scales, ticks, the SVG backend

- **[feature] `bplot`** — a matplotlib-shaped plotting library, 100% pure Bantu, emitting SVG.
  `plot` `scatter` `bar` `barh`; `title` `xlabel` `ylabel` `legend` `grid` `xlim` `ylim`; `savefig`
  `show` `to_svg` `clf` `help`. A `BPlotFigure` / `BPlotAxes` / `BPlotSvg` object model with the
  stateful `plt.*` API over it, so a chart is three lines and the objects appear nowhere in the
  quickstart. Accepts a Bantu list, a numba ndarray or an arctic column anywhere a sequence is
  expected.

- **[feature] Ticking matches matplotlib's `MaxNLocator`** — candidate steps
  `[1, 2, 2.5, 5, 10] × 10^k`, verified against matplotlib's own choices on twelve ranges including
  negatives, `1e-9` and `1e9`. Ticks are computed as `k × step` rather than accumulated, because
  accumulation drops the *final* tick on ranges like 0–1 by 0.1 — and the top tick is the one a
  reader looks for. The epsilon is relative, not absolute.

- **[feature] Path simplification, on by default (BP3)** — at most four vertices per pixel column
  (first, last, min-y, max-y), which preserves spikes where naive decimation drops them. A
  100,000-point line: **25,782 bytes against 1,386,925 unsimplified, a 54× reduction**, still one
  `<polyline>`.

- **[feature] Escaping with no opt-out and no raw-SVG hatch (BP7)** — every text node and attribute
  value is escaped at the point of emission; `&` first, so nothing is double-escaped. Control bytes
  below 0x20 other than tab/LF/CR are **stripped**, not escaped, because they are not representable
  in XML 1.0 at all and a single one turns the chart into a parse error — a blank page with nothing
  in any log. Colours are validated against a shape (`#rgb`, `#rrggbb`, `[r,g,b]`, a named colour)
  rather than interpolated, since an attribute is the second-favourite SVG injection point after a
  text node.

- **[feature] NaN breaks the line; infinities are clipped (BP9)** — untreated, a NaN coordinate emits
  `points="NaN,12 …"`, which every browser renders as **nothing at all**. A run containing NaN is
  emitted as several polylines split at the gaps, so the valid segments draw and the gap is visible
  as a gap.

### Solved defects

Both are Bantu's own documented landmines, found by biting their author.

- **[bug fix] Module-level state did not survive assignment from inside a function.**
  `$_CUR = figure()` inside `gcf()` created a function-**local** — `Environment::assign` stops at the
  nearest function boundary — so the module binding never changed, every call built a new figure, and
  `plt.plot(); plt.savefig()` wrote a chart containing only a background rectangle. The current
  figure now lives in a dict: dicts are reference-semantic, so writing through a field mutates the
  object every caller can see.

- **[bug fix] Path simplification silently produced nothing.** `_emitCol($out, …)` pushed into a list
  it had been passed, and **a Bantu list passed to a function is a copy** — value semantics, unlike a
  dict or a class instance — so the caller saw an empty list and a 20,000-point line rendered as no
  line at all. The helper returns its points and the caller `extend`s the real list.

### Measured

> Intel Core i7-9750H @ 2.60 GHz, macOS 15.7.9, `build-mac.sh` (`-O2`).

| | |
|---|---|
| 100,000-point line, render only | **1,136 ms** |
| — output, simplified | **25,782 bytes** |
| — output, simplification off | 1,386,925 bytes |
| 2,000 figures built and dropped | no failure |
| a four-point chart with title, labels, grid and legend | 4,070 bytes, 41 elements |

The 100k figure was **3,218 ms** before the transform was hoisted out of the per-point loop:
mapping data to pixels went through `$ax.px()` / `$ax.py()`, which is 200,000 method calls. Both axes
reduce to `pixel = value × a + b`, so the coefficients are computed once per axis and the loop
multiply-adds inline — **2.8× faster, byte-identical output**. `px()` and `py()` are defined in terms
of the same coefficients, so there is still one formula and the y flip still has exactly one home.

This is the honest shape of the cost: **drawing** a 100k-point line is one polyline, but **ingesting**
100,000 points is 100,000 interpreted loop iterations. B4's numba path is where that stops being
true.

### Tests

- **[test]** `tests/bplot_core_test.b` — **149 assertions**: `_fmt` against exact strings (it is the
  formatter, so its output is the contract), twelve tick ranges, escaping including an
  already-escaped entity and a control byte, a hostile title through four different emission paths,
  colour validation including an `" onload="` injection attempt, document structure and balance,
  the transform and its single y flip, degenerate data, NaN and infinity, explicit and automatic
  limits, every chart type, the stateful API sharing one figure, `savefig` round-tripping, and path
  simplification both on and off.
- **[test]** `tests/bplot_stress.sh` — **20 checks**, including the one that cannot be written in
  Bantu: every emitted document handed to a **real XML parser**. A subtly malformed document still
  contains all the right substrings and simply renders as a blank page, so substring assertions
  cannot catch it.
- **[test]** Full regression green on the same build: 37 `.b` suites, 13 `.sh` suites.

---

## B2 — The chart types, the scales, and the date axis

Eight statistical chart types, three scales, calendar-aware date ticks, annotations, and categorical
axes. Everything still pure Bantu.

```bantu
include "bplot" as plt;
plt.hist($samples, {"bins": 30});
plt.yscale("log", null);
plt.savefig("chart.svg");
```

### Added

- **[feature]** `hist` `boxplot` `violin` `errorbar` `fill_between` `step` `stem` `pie`.
- **[feature]** `log` and `symlog` scales, with decade major ticks and 2–9 minor ticks.
- **[feature]** date axes over **epoch milliseconds, UTC** — arctic's own datetime storage, so a
  column plots without conversion.
- **[feature]** `text` and `annotate`, with arrows and text rotation.
- **[feature]** **categorical axes** — a sequence of strings becomes positions with the strings as
  tick labels (BP28). `bar(["Jan", "Feb", …], rainfall)` is the first thing anyone types, and before
  this it died inside `min()` with an error about the wrong thing.
- **[feature]** two backend primitives, `polygon` and `path` — and no more, which is what keeps B6's
  raster backend a finite job.

### Verified against the tools people will compare with

Reference values were generated once, at authoring time, from **numpy 2.5.2** and **matplotlib
3.11.2**, and pasted into the tests as literals. Nothing at run time depends on Python.

| checked against | result |
|---|---|
| `matplotlib.scale.SymmetricalLogTransform` | 17 points × 3 `linthresh` settings, to **9 decimals** |
| matplotlib's log ticks | **8 ranges exact** |
| matplotlib's symlog ticks | 6 limit/`linthresh` combinations |
| `AutoDateLocator` | **8 ranges exact**, ten seconds to 25 years |
| numpy `quantile` · `cbook.boxplot_stats` | exact |
| numpy `histogram` / numba `nd_histogram` | exact, including the **inclusive top edge** |

**Two ranges deliberately disagree with matplotlib**, and the disagreement is asserted so it cannot
decay into a bug. Asked for a log axis from 2 to 9, matplotlib returns **no major ticks at all** — an
axis with no labelled tick anywhere on it; `1..3` gets exactly one. bplot promotes the 2–9 minors to
labelled ticks when fewer than two decades fall in view (BP17), because an unlabelled axis is not a
style difference.

**Date ticks reproduce `AutoDateLocator`'s selection rule rather than approximating it.** The rule
turned out to be four lines, so exact reproduction cost less than a look-alike and cannot drift on a
range nobody checked. It gets the two cases an approximation misses: a three-month span switches to
**semi-monthly** ticks (the 1st and the 15th), and a 25-year span picks a **four**-year interval
anchored on multiples of four.

### Measured

| operation | result |
|---|---|
| 100k-point line, **linear** scale | **1,198 ms** against B1's 1,136 — the scale machinery costs a linear axis nothing |
| 100k-point line, symlog scale | 2,585 ms — the honest price of a non-linear axis |
| 200,000 points → `hist` | 1,103 ms, **6,959 bytes** (40 bars, however many points went in) |
| 200,000 points → `violin` | 1,971 ms, **5,291 bytes** |

The violin number is why the density is estimated from a 512-bin histogram rather than from every
point (BP21): the textbook form is 200,000 × 128 kernel evaluations, which is minutes.

**The transform is applied as a pre-pass over the sequence, never per point inside a drawing loop**
(BP16). The first implementation materialised a pixel list even for linear axes and cost 100,000
points **1,480 ms against 1,136** — a 30% tax on the common case for a feature it does not use. The
hot artists fuse the projection into the single loop B1 had.

### Solved defects encountered

- **A bar's zero baseline made a log axis raise before any data was projected.** `bar()` and `hist()`
  publish a baseline of zero as part of their bounds, and the log view-limit computation choked on
  it — making both unusable on a log scale. Baselines are clamped to the view floor now, while real
  non-positive **data** still raises. The difference is deliberate and documented: a line with a gap
  in it is invisible and misleading, an absent bar already reads as zero.

### Tests

- **[test]** `tests/bplot_charts_test.b` — **196 assertions**.
- **[test]** `tests/bplot_stress.sh` grew to **38 checks**; **51 documents** handed to a real XML
  parser, every one well-formed.

---

## B3 — Layout and 2-D

Subplots, twin and shared axes, measured layout, the published colormaps, colorbars, and the 2-D
chart types.

### Added

- **[feature]** `subplots(rows, cols)`, `subplot(r, c, i)`, `subplotSpan`, `twinx`, `twiny`,
  `sharex`, `sharey`, `tight_layout`.
- **[feature]** `imshow` `heatmap` `pcolormesh` `contour`, and `colorbar`.
- **[feature]** colormaps `viridis` `plasma` `coolwarm` `gray`, as the **published 256-entry tables**
  embedded as hex strings (BP25). `jet` is deliberately not shipped.
- **[feature]** style sheets: `default`, `dark`, `print` — the last ordered by **lightness** rather
  than hue, so the series stay distinguishable in a photocopy and under the common colour-vision
  deficiencies.

### Measured

| gate | result |
|---|---|
| 1000×1000 `imshow` (1,000,000 cells) | **1,473,633 bytes, 256 elements**, 30.6 s |
| — the naive encoding, one `<rect>` per cell | ~55 MB |
| 6,000 figures built and dropped | RSS **10.5 MB → 12.0 MB** — flat |
| an 8×8 grid | 64 panels, no overlap, all inside the figure |
| rendering the same figure twice | **byte-identical** |

`imshow` block-reduces to a cell budget and batches every cell of one colour into a **single
`<path>`** (BP26). A colormapped image has at most 256 colours by construction, so the element count
is bounded however many cells go in.

**The B3 roadmap gate was corrected rather than quietly met.** It asked for "a single embedded
image", which means a base64 PNG, which means CRC32, Adler-32 and deflate — the native work that *is*
B6. The original wording is kept in the roadmap alongside the correction, and B6 inherits the gate.

### Solved defects encountered — both of them general Bantu defects

- **Every Bantu object leaked.** `new ClassName()` allocated an instance that **nothing ever
  deleted**, so every object a Bantu program created leaked for the life of the process: ~300 bytes
  each, ~45 KB per figure, **372 MB over 20,000 figures and still climbing**. A `sua` handler
  creating objects per request would have grown without bound until the worker was killed. Instances
  are refcounted now — 20,000 instances is flat at 4.9 MB. Refcounting does not collect **cycles**,
  so bplot's Axes deliberately holds no pointer back to its Figure and share groups are stored as
  indices (BP30). The gate is an RSS measurement, not an inspection.
- **`&&` and `||` did not short-circuit.** The universal guard idiom
  `if ($i < len($a) && $a[$i] == x)` evaluated the right operand unconditionally and died with
  "Index out of bounds" on exactly the boundary it was written to prevent. Fixed; A/B'd on a
  1M-iteration loop with no logical operators at all, best-of-5 in both orderings: **540/541 ms
  before against 535/533 ms after**, so the cost is below the noise floor.
- **The colormap index was `round(t × 255)` where matplotlib uses `floor(t × 256)`.** The two agree
  at 0, 0.5 and 1 — so the obvious test passes — and disagree at 0.625 by a visibly different green.
  Caught by sampling the interior rather than only the endpoints (BP29).
- **The legend card, histogram bar edges and pie separators were hardcoded white**, which the dark
  style exposed as a white box carrying near-invisible light text. They follow the theme now, and
  cell and slice labels pick black or white by the fill's own luminance.

### Added to the language, because bplot could not be written without them

- **[feature] `sort` and `reverse`.** Bantu had `push`, `pop`, `insert`, `extend` and `slice` and no
  way to **order** a list, so every median, quantile, boxplot and ranking in every Bantu program was
  an interpreted sort. `NaN` sorts last in both directions — a comparator that answers `false` to
  every `NaN` comparison is not a strict weak ordering, and `std::sort` given one reads past the end
  of its range. A user comparator gets a merge sort, which cannot leave its range whatever the
  comparator answers.

### Tests

- **[test]** `tests/bplot_layout_test.b` — **132 assertions**.
- **[test]** `tests/bplot_stress.sh` — **48 checks**, including the RSS gate.
- **[test]** Full regression green on the same build: **37 `.b` suites, 13 `.sh` suites**, and all
  four `samples/bplot/` programs executed by `tests/run_samples.sh`.

---

## Phase B4 — Data integration

- **[design]** `docs/bplot-architecture.md` §13 rewritten around a measurement, and decisions
  BP31–BP36. The B0 design converted every input to a Bantu list at the boundary; measured on one
  1,000,000-point line that cost **~20 s and 3.1 GB**, because a list of a million 190-byte values is
  copied on every function call that receives it.
- **[feature]** `plot`, `scatter`, `hist`, `boxplot` and `violin` take a Bantu list, a numba ndarray,
  an arctic column or a `Series`, and render **byte-identical documents** from all four (BP31, BP32).
  bplot includes neither library; it recognises their shape.
- **[feature]** `plot_frame($df, {kind, x, y, title})` — `line`, `bar`, `barh`, `scatter`, `hist`,
  `box`, `step`. Every numeric column by default; more than one series turns the legend on and groups
  bars. A text column named explicitly raises, naming it and its type (BP35).
- **[feature]** arctic's `$df.plot(opts)` and `$series.plot(opts)`, which include bplot **lazily** —
  only on the first plot, as pandas does — and share the current figure with your own `plt`.
- **[feature]** `heatmap($df)` draws the numeric columns with their names as labels.
- **[feature]** a null is a gap; a datetime column makes a date axis on its own; a date column is
  days scaled to milliseconds (BP34).
- **[perf]** `bp_line_runs`, `bp_scatter_path` and `bp_escape` in `plot_native.hpp` (BP33). They are
  handed pixels, not data, so they contain no arithmetic a compiler could fuse into a multiply-add
  that rounds differently from the interpreter, and they reproduce the pure path byte for byte.
  Scatter at 1,000 points and above is one `<path>`; below it nothing changes.

### Measured

| | native | pure path, same build |
|---|---|---|
| 1,000,000-row column as a line | **252 ms** | 15,455 ms |
| peak RSS | **112 MB** | 2.69 GB |
| the document | 23,825 bytes | 23,825 bytes, identical |
| 1,000,000 rows: hist / two boxes / violin | 61 / 241 / 306 ms | — |
| 1,000,000-point scatter | 1,523 ms | — |

### Solved defects encountered

- **[bug fix] `include "bplot"` bound an empty module** when run from a folder that contains a
  `bplot/` directory — this repository's own layout. The resolver's existence check accepted a
  directory, a directory reads as an empty file, and the alias was bound with no error at all. A
  module must now be a regular file, and a bare name naming a directory resolves inside it.
- **[bug fix] `len()` returned 0 for a dict, an ndarray and a column**, so `while ($i < len($a))`
  never ran. An existing test had pinned the dict answer to prove an earlier performance change was
  behaviour-neutral; it is corrected in place with the reason.
- **[bug fix] `contains()` returned false for every list.**
- **[bug fix] hour and minute date labels carried no date** — a four-day axis read
  `00:00 12:00 00:00 …` — although §6.3 had specified a dated label. Now matplotlib's
  `%m-%d %H` and `%d %H:%M`, verified against matplotlib 3.11.2; the B2 assertions that pinned the
  old format are corrected in place.
- **[bug fix] a list containing `null` could not be plotted**; it is a gap now, like a null in a
  column.
- **[bug fix] a 2-D ndarray passed to `plot`** became nested lists that failed far from the cause; it
  raises at the boundary, naming its shape.

### Tests

- **[test]** `tests/bplot_data_test.b` — **138 assertions**: the four-input gate and the
  native-against-pure gate on 17 chart configurations, nulls, datetimes, text, every `plot_frame`
  kind and its errors, `df.plot()` sharing the figure, and the kernels attacked directly.
- **[test]** `tests/bplot_stress.sh` — **62 checks** (48 before): 1,000,000 rows through line, hist,
  box, violin and scatter with wall-clock gates, a document-size gate, a peak-RSS gate, a mostly-null
  and an all-null column, a non-numeric column selected, and hostile column names — every document
  parsed as XML.
- **[test]** the language fixes: `tests/lang_list_test.b`, `tests/lang_module_test.b`,
  `tests/numba_array_test.b`, `tests/arctic_api_test.b`.
- **[test]** `samples/bplot/05_dataframe.b` — a frame, a datetime column with a gap, and a million
  rows, executed by `tests/run_samples.sh`.

---

## Phase B5 — Package, docs, gallery

- **[docs]** `docs/bplot.md` — three lines to a chart; a tour of lines, bars, scatters, distributions,
  scales and dates, layouts, grids, tables from arctic and styles; serving from sua; **the SVG
  security section**, under a heading a reader will actually reach; measured numbers; things worth
  knowing; and blunt caveats. Every example on the page is executed.
- **[test]** `tests/run_doc_examples.sh` — every ```` ```bantu ```` block of a doc runs as **one
  program, in reading order**, from an empty directory that links the packages, so an example cannot
  rot and cannot write into the repository (BP37). Covers `docs/bplot.md`, `docs/numba.md` and
  `docs/arctic.md`.
- **[test]** `tests/bplot_package_test.sh` — the clean-project gate, 17 checks, run in a throwaway
  `HOME` so it never touches the real registry.
- **[feature]** `samples/bplot/server.b` — a chart drawn per request and served through sua, with the
  object API and a Content-Security-Policy (BP38); `tests/bplot_sua_test.sh`, 15 checks including 40
  concurrent requests that must each receive only their own chart.
- **[feature]** bplot **1.1.0**. `package.json` describes what the package now is; the smoke test also
  draws an ndarray.
- **[test]** CI: the three new gates in the Linux and macOS jobs, and bplot's smoke test in the
  Windows job — which is also the proof that `plot_native.hpp` compiles under MSVC.

### Solved defects encountered

- **[bug fix] numba's documented examples had never been run.** The numba roadmap recorded "every
  documented example executed by CI" as met; no test executed them. Run for the first time, two of the
  nine failed: `np.sum($m, [0, 2], null)` on a 2×3 array, and a destination-buffer loop over `$a` and
  `$b` that its block never defined, so a reader copying it hit a broadcast error from arrays built
  sections earlier. Both corrected.
- **[bug fix] `num()` read a prefix, swallowed overflow and ignored bools** — `num("12abc")` was 12,
  `num("1e999")` was 0, `num(true)` was 0. It now reads the whole string, overflow is infinity, a bool
  is 1 or 0, and `num($s, $default)` returns `$default` for input that is not a number.
- **[bug fix] `tests/numba_sua_test.sh` described sua's threading wrongly** — every connection on its
  own detached thread. Handlers run one at a time on the event loop; suspended handlers continue on
  pooled task threads under a baton, never concurrently.

### Not yet observed

- **Cross-platform CI.** Every gate above is wired into CI; no CI run has happened on this branch, so
  "green on three platforms" is still a claim about configuration, not a result.

---

## Phase B6 — The native raster backend (in progress)

- **[design]** [`docs/bplot-raster-architecture.md`](../docs/bplot-raster-architecture.md), written
  before any code. Byte-identical PNGs on three platforms rule out fused multiply-adds, libm's
  transcendental functions, the system zlib and system fonts in any step that decides a pixel; so every
  coordinate is quantised exactly as `_px` quantises it for SVG, everything after is integer
  arithmetic, circles and arcs come from embedded integer tables, text uses an embedded DejaVu Sans
  subset (211 glyphs, about 13 KB), and deflate is written here. The PNG draws the SVG's numbers.

### B6a — binary-safe file I/O

- **[bug fix]** `open()` sent every mode other than `"r"`, `"w"` and `"a"` to read mode, so
  `open($path, "wb")` silently opened for reading and every write failed. `open`, `readfile`,
  `writefile` and `appendfile` now accept `"rb"`, `"wb"` and `"ab"` with `std::ios::binary`, an unknown
  mode raises naming the ones that exist, `readfile` refuses a write mode before it can truncate
  anything, and a failed write raises. Text remains the default.
- **[bug fix] `read($f)` could not be called at all.** Found by the first test of the modes above:
  `read`, `await`, `private`, `public`, `calc`, `import` and `export` were lexed as keywords no grammar
  rule accepted. They are ordinary identifiers now.
- **[test]** `tests/lang_file_test.b` — every byte value, runs of NUL, CR, LF and 0x1A, a PNG
  signature and a megabyte round-trip in binary; text defaults unchanged; each refused mode leaves the
  file untouched.
