# bplot — Design Decisions

Rationale for the choices behind Bantu's plotting library, so they aren't re-litigated. Each entry:
**decision · why · alternatives rejected · implication.**

Decisions are numbered `BP*` so they never collide with arctic's `A*` or numba's `N*`. Phases are
lettered `B0`–`B6`; the two numbering schemes are deliberately different.

Full reasoning with measurements lives in [`docs/bplot-architecture.md`](../docs/bplot-architecture.md).

---

### BP1 — Pure Bantu for B1–B5; native only for raster (B6)
**Decision:** the object model, scales, ticking, layout, colours and the **SVG backend** are 100%
pure Bantu. Only the raster backend is native.
**Why:** the work in a vector chart is proportional to the number of *drawn elements* (~10³), not to
the number of data points. At the interpreter's measured ~0.38 µs per operation that is around a
millisecond.
Rasterisation is proportional to *pixels* (~10⁶ for a modest figure), which is minutes — a difference
of three orders of magnitude, and the line falls exactly between the two backends.
**Rejected:** a native core for B1 — it would put the tick algorithm, layout and colour handling,
the parts that change most often, behind a C++ rebuild.
**Implication:** bplot works on any Bantu build. `savefig("x.png")` is the only thing that needs a
capability probe.

### BP2 — Build output with `join`; `+=` is no longer a trap either
**Decision:** artists push strings onto a list and the document is assembled by one `join`.
`join(list, sep)` was added to the language in B0.
**Why:** measured, repeated string concatenation was **O(n²)** — 20,000 appends 1,116 ms and 40,000
appends 6,752 ms, a 6.05× cost for 2× the work, because each `+` copied the whole accumulated string.
A 100k-element figure would have taken upwards of forty seconds.

**SUPERSEDED IN PART, and the original reasoning was wrong.** This decision rejected "making `+` on
strings amortised" on the grounds that it "needs a rope or a refcounted builder inside `Value`, a far
larger change to the hottest type in the interpreter". That was the wrong conclusion: it needs
neither. When the result of `x + y` is assigned straight back to `x`, `x`'s old value is dead, so it
can be appended to in place — a peephole in `evalAssign`, no change to `Value`'s representation, and
what CPython has done since 2.4. **`$s = $s + $part` and `$s += $part` are now linear**: 40,000
appends 6,752 ms → **23 ms**, and 200,000 appends 112 ms. See
[`docs/interpreter-performance.md`](../docs/interpreter-performance.md) §3.

**What still stands:** `join` is not redundant. It is the right idiom when the pieces are already a
list, it needs no peephole to fire, and it does not depend on the accumulator being a plain local
variable — `$fig.parts` accumulated through a field takes a different node type and does not get the
in-place path. bplot keeps building through a list and joining once.
**Implication:** `split` finally has its inverse, and every Bantu program that assembles text — `sua`
HTML, CSV writing, log formatting — got both fixes, whichever idiom it already used.

### BP3 — Path simplification is on by default
**Decision:** a polyline is reduced, before emission, to at most four vertices per pixel column
(first, last, min-y, max-y), bounding output at `4 × canvas_width` vertices regardless of input size.
Switchable off.
**Why:** an 800-pixel-wide canvas cannot show more than ~800 distinct columns, so 100,000 vertices
carry no information 1,600 do not. This is ~60× less output and ~60× less build time, and it is what
matplotlib does (`path.simplify`, default-on since 1.2).
**Why min and max, not every fourth point:** keeping the extremes preserves spikes. Naive decimation
drops them, and a dropped spike is a chart that lies.
**Implication:** the "100k-point line" stress case stops being about string throughput and becomes a
correctness question — does the simplified path look like the real one — which is the question worth
gating on.

### BP4 — matplotlib's layering, reproduced exactly
**Decision:** stateful `plt.*` over Figure → Axes → Artist over a swappable Backend.
**Why:** it is the layering that let matplotlib change output targets — Agg, Cairo, PDF, SVG,
WebAgg — for twenty years without an API break, which is precisely the long-term-support property
being asked for. It is also the shape users already know.
**Rejected:** a flat immediate-mode API (`draw_line(fig, ...)`) — simpler for a week, and then
`tight_layout`, twin axes, shared axes and colorbars each need to reach into state the API has no
place to keep.
**Implication:** B6's raster backend is a new object implementing nine methods, not a rewrite.

### BP5 — Classes, not dicts, for Figure and Axes
**Decision:** `BPlotFigure`, `BPlotAxes`, `BPlotLine`, `BPlotText`, `BPlotSvg` — Bantu classes,
reached through factory functions.
**Why:** verified — class instances have **reference** semantics through assignment, function
arguments, lists and dicts; Bantu lists and dicts have **value** semantics. `$ax = plt.subplot(2,2,1);
$ax.plot(...)` mutating the figure the user holds is not optional, and only instances give it.
**Why prefixed:** `classRegistry_` is a flat global map and arctic already owns `Series`,
`DataFrame`, `LazyFrame`, `GroupBy`. An unprefixed `Figure` is a collision waiting for the first
program that includes both.
**Why factories:** `new alias.Class()` does not parse, so `plt.figure()` is the only way to reach a
class across a module boundary — the same conclusion arctic reached.

### BP6 — SVG first, raster later
**Decision:** `savefig("x.svg")` in B1; `savefig("x.png")` in B6.
**Why:** SVG is text, so it needs no rasteriser, no deflate and no binary-safe file writes — it can
ship while those are still being built. It renders in every browser, scales without loss, is what a
report or a dashboard actually wants, and `image/svg+xml` is already in `mime_types.hpp:41`, so
`sua` serves it correctly today.
**Rejected:** PNG first — it makes the entire library wait on three native subsystems before anyone
can draw a line.

### BP7 — Escaping is mandatory, unbypassable, and there is no raw-SVG hatch
**Decision:** every text node and every attribute value is escaped at the point of emission inside
the backend; control characters are stripped; colours and fonts are validated against a shape rather
than interpolated; **no `plt.raw_svg()` exists**.
**Why:** SVG is an *active document format*. It may carry `<script>`, `on*` handlers and
`<foreignObject>`, and `sua` already serves `image/svg+xml`. A chart title taken from a request
parameter is therefore a stored XSS: `</text><script>…</script>` closes the element and runs. This is
the same class as the `sua.udp` `maxBytes` defects — unvalidated input reaching a place that trusts
it.
**Why no hatch:** a library that offers one has delegated its security to every caller, and the first
caller to pass user text through it reopens the hole.
**Why strip rather than escape control bytes:** bytes below 0x20 other than tab, LF and CR are not
representable in XML 1.0 at all, so a single one turns the chart into a parse error — a blank page,
with nothing in any log.
**Implication:** injection *through bplot* is impossible. bplot cannot make SVG stop being
executable, so `docs/bplot.md` carries the deployment caveat — separate origin, or a restrictive CSP,
or PNG — under a heading users will read.

### BP8 — `max`/`min` propagate NaN; bplot filters separately
**Decision:** the B0 language `max`/`min` return NaN if any argument is NaN. bplot's data-limit
routine filters NaN itself.
**Why:** a language primitive must not silently discard a value it was handed; NumPy makes the same
split (`max` propagates, `nanmax` skips). A plot, by contrast, must draw the valid points.
**Implication:** two behaviours that look inconsistent are each correct for their level, so both are
documented in the same place to stop them being confused.

### BP9 — NaN breaks the line; infinities are clipped
**Decision:** a polyline containing NaN is emitted as several polylines split at the gaps.
Infinities are clipped to the view box. Data limits skip both.
**Why:** untreated, a NaN coordinate emits `points="NaN,12 …"`, which every browser renders as
**nothing at all**, with no error anywhere — the worst possible failure mode. Splitting draws the
valid segments and makes the gap visible as a gap, which is both matplotlib's behaviour and the
honest one.
**Implication:** `isnan` and `isfinite` are B0 builtins, not an afterthought — the language had no
way to *test for* the values it could already produce.

### BP10 — Number formatting is pure Bantu, because `str()` cannot be used
**Decision:** `_fmt(x, decimals)` in `bplot.b`; an axis picks its decimal count from its tick step.
**Why:** `str()` gives six significant digits (`1.0/3.0` → `0.333333`) and leaks scientific notation
(`0.000012345678` → `1.23457e-05`) — neither is acceptable on an axis. It also had a defect at large
magnitudes (`str(1e21)` → `-9223372036854775808`), fixed in B0.
**Implication:** every tick on an axis shares a decimal count, which is a small thing that makes a
chart look finished rather than generated. And, as in numba, **no test asserts on a stringified
number** — except `_fmt`'s own tests, where the string is the contract.

### BP11 — Ticks by `MaxNLocator`, with `2.5` in the candidates and a *relative* epsilon
**Decision:** candidate steps `[1, 2, 2.5, 5, 10] × 10^k`; the upper bound is tested against
`hi + 1e-10 × max(|lo|, |hi|)`.
**Why (`2.5`):** without it, 0–1 at six ticks gives steps of 0.2 where 0.25 is the better choice.
matplotlib includes it and charts look subtly wrong without it.
**Why relative epsilon:** floating-point accumulation otherwise drops the *final* tick on ranges like
0–1 by 0.1 — and the top tick is exactly the one a reader looks for.

### BP12 — Embedded Helvetica metrics for text measurement
**Decision:** a pure-Bantu table of AFM advance widths for ASCII 32–126, in 1/1000 em; non-ASCII
falls back to the width of `n`, CJK to one em.
**Why:** `tight_layout` and legend sizing need to know how wide a label is, and text is rendered by
whatever font the viewer resolves `sans-serif` to. Helvetica and Arial share metrics and are what
`sans-serif` resolves to on most systems, so the estimate is accurate where it matters and
approximate elsewhere — and layout pads outward, so an error produces whitespace rather than a
collision.
**Rejected:** FreeType, or parsing a font file — a new dependency and a new parser, for a table.
**Implication:** same reasoning as numba's refusal of BLAS and the tree's from-scratch P-256.

### BP13 — `viridis` by default; `jet` is not shipped
**Decision:** `tab10` for the categorical cycle, `viridis` as the default continuous colormap,
embedded as 256-entry RGB tables.
**Why:** `viridis` is perceptually uniform, survives greyscale printing, and remains readable under
the common colour-vision deficiencies. `jet` invents structure that is not in the data — a false
band at its cyan/yellow transitions — and is unreadable in greyscale. Shipping it would mean people
using it.
**Implication:** colormaps are data, not code, so adding one is a table and not a function.

### BP14 — `show()` writes or serves; it does not open a window
**Decision:** `show()` writes the file and prints the path, or serves it on `sua.server` and prints
the URL.
**Why:** Bantu has no `exec`, no `system`, no shell builtin and no GUI binding. There is nothing to
hand a file to, and inventing a process-spawning builtin for the convenience of one library would
add the largest security surface in the tree.
**Implication:** stated plainly in the docs rather than worked around.

### BP15 — Module-level current-figure state is not for `sua` handlers
**Decision:** `plt.plot(...)` without a figure creates one, held in module state. Servers use the
object API instead: `$fig = plt.figure(); …; $fig.to_svg()`.
**Why:** `sua` runs each connection's handler on its own detached thread, and module state is shared
across them — the same hazard that made numba's PRNG `thread_local` and its limits atomic. Two
concurrent requests plotting into the implicit figure would interleave into one chart.
**Implication:** the sua sample uses only the object API, and the caveat sits at the top of the
serving section, not in a footnote.

### BP16 — A scale is a pre-pass over the data, not a branch in the drawing loop
**Decision:** a non-linear scale transforms the whole sequence once (`_project`), and the artists then
run the same affine `v * a + b` loop they ran in B1.
**Why:** the obvious implementation calls `fwd(v)` per point, which at ~0.38 µs per interpreted
operation costs 200,000 extra calls on a 100k-point line — and the *branch deciding whether to call*
would sit in the loop too, so **linear axes would pay for a feature they do not use**. Linear is the
overwhelming majority of axes and it must stay exactly as fast as it was.
**Also why:** it puts the non-positive-value check in exactly one function. Four artists each doing
their own check is four chances to forget, and the one that forgets emits `NaN` coordinates.
**Rejected:** a transform object per axis with a `fwd()` method — the same per-point cost with an
extra dynamic dispatch on top.
**Implication:** limits, ticks and artists cannot disagree, because all three go through `_project`.
The B1 100k-point timing is re-run as a gate and must not move.

### BP17 — Log ticks are decades; below two decades the minors are promoted
**Decision:** major ticks at `10^k`, minor at `2..9 × 10^k`. **If fewer than two decades fall in
view, the minors become the labelled ticks and the minor set is dropped.**
**Why:** matplotlib, asked for an axis from 2 to 9, returns **no major ticks at all** — an axis with
no labelled tick anywhere on it. `1..3` gets exactly one. That is not a style difference; it is a
chart nobody can read, and matching it faithfully would be matching a wart.
**Why not always promote:** on a wide range the 2..9 multiples are 8× too many labels and they
collide.
**Implication:** the test suite asserts *both* the agreement with matplotlib on the eight sane ranges
and the divergence on the two, so the divergence stays a decision rather than decaying into a bug.

### BP18 — symlog is matplotlib's transform, constants and all
**Decision:** `linscale_adj = 1/(1 - 1/10) = 10/9`; inside the band `x * linscale_adj`, outside
`sign(x) * linthresh * (linscale_adj + log10(|x|/linthresh))`.
**Why:** symlog exists so data spanning decades *and* crossing zero can be plotted, and anyone
reaching for it knows it from matplotlib. A subtly different curve would make the same data look
different in the two tools with no way to tell which was right.
**Verified, not assumed:** checked against `matplotlib.scale.SymmetricalLogTransform` at three
`linthresh` values and ten sample points; those reference numbers are literals in
`tests/bplot_charts_test.b`.
**Implication:** `linthresh <= 0` raises naming the value — it would divide by zero and produce
exactly the `NaN` coordinates BP9 exists to prevent.

### BP19 — Histogram binning copies `nd_histogram` exactly, including the inclusive top edge
**Decision:** `floor((x - lo) / width)`, top edge inclusive, out-of-range dropped, `NaN` dropped,
`lo == hi` widened by ±0.5, empty input falling back to `0..1`.
**Why:** B4's gate is identical output from a list, an ndarray, a column and a Series. That is only
reachable if the pure-Bantu binner and numba's native one make the same decisions — so bplot copies
the native rules rather than writing the obvious ones and discovering the difference at B4.
**The one that bites:** without the inclusive top edge the largest value in the data silently
disappears, and a histogram missing its maximum lies about its range. It has its own test.
**Implication:** `density` normalises by `n × width` so bar *areas* sum to 1 (NumPy's definition),
not heights — which differ the moment bin widths are unequal.

### BP20 — Quantiles are NumPy's linear interpolation, because `nd_quantile` already is
**Decision:** `pos = q(n-1)`, linear between the bracketing order statistics (Hyndman–Fan type 7).
Boxplot is Tukey's: box Q1–Q3, median line, whiskers to the most extreme **observed** value within
1.5 × IQR, the rest drawn as outlier points.
**Why:** there are nine defensible quantile definitions and the only one that matters is the one the
rest of the tree uses. A median that moves depending on whether the data arrived as a list or an
array is worse than any of the nine.
**Why the whisker lands on a datum:** drawing it at `Q1 - 1.5·IQR` itself — the common error — puts
the whisker end in empty space where no observation exists.

### BP21 — Violin is a *binned* KDE, so its cost does not scale with n
**Decision:** bin into 512 bins in one pass, then evaluate the Gaussian kernel against the bins.
`O(n + bins × grid)` rather than `O(n × grid)`.
**Why:** the textbook form is 12.8 million interpreted operations for 100,000 points on a 128-point
grid — most of a minute, for one violin. Binned KDE is the standard approximation (R's `density()`
does it via an FFT) and its error is bounded by the bin width, which is far below the bandwidth the
kernel is already smoothing with.
**Bandwidth:** Scott's rule, `1.06 σ n^(-1/5)`, matching `scipy.stats.gaussian_kde`'s default.
**Implication:** fewer than two distinct values has no density; it degenerates to a flat line rather
than dividing by a zero bandwidth.

### BP22 — Date axes use a calendar ladder and UTC, and parity with matplotlib is not claimed
**Decision:** matplotlib's `AutoDateLocator` selection rule, reproduced exactly — walk the units
coarsest first, take the first spanning at least five of that unit, then the smallest interval with
`span <= interval × (maxticks - 1)`. Ticks are **anchored to the calendar** (days-of-month, months
from January, years on multiples of the interval), never accumulated. All UTC.
**Why:** `MaxNLocator` on epoch milliseconds proposes steps like 2,500,000,000 ms — a tick every 28.9
days, landing mid-afternoon on drifting dates. And months and years are not fixed-length, so no
amount of millisecond arithmetic can produce "the first of each month".
**Why UTC only:** local time needs a timezone database. That is not a dependency this tree will take
for an axis label, and arctic already stores epoch ms in UTC (`dataframe_native.hpp:77`).
**Calendar arithmetic:** Howard Hinnant's `civil_from_days`/`days_from_civil` — twelve lines, exact
over the proleptic Gregorian calendar, no table.
**Parity is exact** on all eight ranges checked, ten seconds to twenty-five years — including the two
a look-alike gets wrong: a three-month range switches to *semi-monthly* ticks (the 1st and the 15th),
and a twenty-five-year range picks a **four**-year interval on multiples of four, not five.
**This entry originally hedged** ("close but not claimed to be exact"), because the plan was to
approximate the ladder. Reading `AutoDateLocator.get_locator` showed the rule is four lines, so
exact reproduction cost less than approximation and cannot drift on a range nobody checked. The
hedge is withdrawn rather than left standing.

### BP23 — `sort` and `reverse` are language additions, and NaN sorts last
**Decision (found while building B2):** Bantu has `push`, `pop`, `insert`, `extend` and `slice` but
**no `sort` and no `reverse`**. Both are added as native builtins returning a new list.
**Why this is a language fix, not a bplot one:** a language shipping a dataframe library, an ORM and
now a plotting library could not order a list. Quantiles, medians, boxplots, `unique`, ranked output,
leaderboards and "top N" all need it, and every one of them was previously an interpreted sort.
**Why NaN sorts last:** a comparator that answers `false` to every NaN comparison is **not a strict
weak ordering**, and `std::sort` given one reads past the end of its range — a real out-of-bounds
write, not merely a wrong order. numba's `nd_sort` already sorts NaN last (`lessNaNLast`); `sort`
matches it, so the two agree.
**Why a merge sort for a user comparator:** a user-supplied comparator can be non-transitive, and no
amount of validation catches that. A merge sort cannot run off its range whatever the comparator
answers, so a bad comparator gives a strangely ordered list instead of memory corruption.
**Mixed types raise.** Ordering a list of numbers and strings has no correct answer, and picking one
silently is how a sort quietly produces garbage.

### BP24 — `tight_layout` measures at render time, not at call time
**Decision:** gutters are computed from the text that will actually be drawn, using the Helvetica
metrics, during rendering.
**Why at render time:** labels and titles are normally set *after* the axes exists, so measuring at
call time measures an empty axes. matplotlib has exactly this trap and answers it with "call it
last"; bplot removes the trap instead.
**Why it always pads outward:** the viewer picks the font, so the measurement is an estimate. An
overestimate is whitespace; an underestimate is a collision. The asymmetry is deliberate, and it is
why an approximate metrics table is good enough to be useful.

### BP25 — Colormaps are the published 256-entry tables, embedded as hex, never approximated
**Decision:** `viridis`, `plasma`, `coolwarm`, `gray` as 1,536-character hex strings, sliced six
characters per entry. Generated once at authoring time from the published source; **no runtime
dependency on anything.**
**Why not a polynomial fit:** there is no closed form for viridis — it is the output of an
optimisation in CAM02-UCS perceptual space. A fit called "viridis" would be a different colormap
wearing the name, and perceptual uniformity, the entire reason to use it, is exactly what the fit
loses.
**Why a hex string:** one twelfth the source size of 768 numbers, it cannot be half-edited into a
valid-but-wrong table, and parsing is four `ord` calls.
**Provenance, recorded in the source:** viridis and plasma are Smith and van der Walt's tables,
released **CC0**; coolwarm is Moreland's diverging map as matplotlib samples it.
**Implication:** the tests assert sampled entries against values written independently into the test
file, so a corrupted table fails instead of agreeing with itself.

### BP26 — `imshow` block-reduces and batches by colour; full-resolution raster is B6
**Decision:** an array larger than `maxcells` (default 256) per axis is reduced by **block mean**;
cells are then merged into horizontal runs and emitted as **one `<path>` per distinct colour**.
**Why:** one `<rect>` per pixel makes a 1000×1000 array a **55 MB** document. A colormapped image has
at most 256 colours by construction, so per-colour batching gives ≤ 256 elements and ~14 bytes per
cell instead of ~55 — and the axes box is only ~500×380 CSS pixels, so cells beyond the budget were
never visible.
**Correcting a gate I wrote.** The B3 roadmap gate said the correct encoding is "a single embedded
image". That is right, and it is **not reachable in B3**: a single image means a base64 PNG, which
means CRC32, Adler-32 and deflate — the native work that *is* B6. The gate is corrected in the
roadmap with the reason, not quietly dropped, and B6 inherits it.
**Implication:** the API does not change at B6; only the encoder behind it does. That is what the
backend boundary was for.

### BP27 — `GridSpec` is the only layout arithmetic; sharing is symmetric
**Decision:** every axes rectangle — `figure()`, `subplots`, `subplot(r,c,i)`, `twinx` — comes from
one `GridSpec` calculation. Shared axes hold a **list of peers**, not a parent pointer.
**Why one calculation:** matplotlib's `subplot` and `add_subplot` drifted apart historically; two code
paths producing "the same" rectangle is how that happens. Both spellings are asserted to produce
identical rects.
**Why peers, not a parent:** sharing is symmetric. A parent pointer makes the first axes special,
which breaks the moment it is the one removed.
**Implication:** `twinx` is a second Axes over the same rect with its x limits pinned; the frame is
drawn once by the original, because a doubled 1px stroke is visible and reads as a rendering bug.

### BP28 — A string sequence makes the axis categorical
**Decision (found by writing the gallery):** a sequence of strings on any axis becomes positions
`0..n-1` with the strings as tick labels. The mapping is kept **on the axes** and extended, never
rebuilt.
**Why:** `bar(["Jan", "Feb", …], rainfall)` is the first thing anyone types, and without this it died
inside `min()` with *"element 1 must be a number"* — an error about the wrong thing entirely, three
layers below the call the user made.
**Why the mapping is kept and extended:** two series sharing categories must line up. If the second
series were numbered from scratch its bars would sit under the first series' labels — a chart that
lies, which is worse than one that errors.
**Mixed text and numbers raise.** An axis is either categorical or numeric; guessing which would
silently misplace every point.
**Implication:** `boxplot`, `violin` and `heatmap` already set tick overrides, so this reuses the
same mechanism rather than adding a second one.

### BP29 — The colormap index is `floor(t × 256)`, not `round(t × 255)`
**Decision:** a fraction is quantised to a table entry exactly as matplotlib does it.
**Why this needed a decision at all:** the two formulas look equivalent and agree at 0, 0.5 and 1 —
so an implementation written from intuition passes every obvious test. They disagree at 0.625, where
`round` gives entry 159 and matplotlib gives 160: a visibly different green, in the middle of the
most-used colormap in science.
**How it was caught:** by checking nine sample points against values generated from the published
source rather than three. The lesson is in the test, not just the code: **sample the interior, not
only the endpoints.**

### BP30 — Class instances leaked, and that is why an Axes has no Figure pointer
**Found while building B3.** `new ClassName()` allocated an instance nothing ever deleted, so every
Bantu object leaked for the life of the process — measured at ~45 KB per bplot figure, reaching
372 MB over 20,000 figures and climbing. A `sua` handler drawing a chart per request would have been
OOM-killed. Instances are refcounted now; see
[`docs/language-features.md`](../docs/language-features.md).
**The part that is a bplot decision:** refcounting does not collect **cycles**, and
`Figure → axesList → Axes → fig → Figure` is one. So an Axes holds **no** back-reference to its
Figure, and everything that needs the figure — `twinx`, `shareX`, `setCurrent` — is a *Figure*
method. Share groups are stored as lists of axes **indices** for the same reason: an index points at
nothing.
**Implication:** the gate is an RSS measurement, not an inspection. `tests/bplot_stress.sh` builds
300 figures and 6,000 figures and requires that twenty times the work costs under twice the memory —
a test that a cycle would fail immediately.
