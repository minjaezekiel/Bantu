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
