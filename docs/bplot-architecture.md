# bplot — Architecture

**bplot** is Bantu's data-visualisation library: a matplotlib-shaped API that turns numbers into
charts. It is the second half of the numerical initiative whose first half is
[numba](numba-architecture.md), and it depends on numba the way matplotlib depends on NumPy — for
binning, percentiles, sorting and 2-D arrays.

This document is the design of record. It is written **before** the code, per the standing rule, and
every number in it was measured on the machine named beside it rather than estimated.

> Measurements in this document: **Intel Core i7-9750H @ 2.60 GHz, macOS 15.7.9, Apple clang 17**,
> `bantu` built by `build-mac.sh` (`-O2`), commit `96d28cf`.

---

## 1. What bplot is, and what it is not

**It is** a pure-Bantu rendering library that emits **SVG** — a text format, so producing it is
string work, which Bantu can do. It has matplotlib's layering (a stateful `plt.*` API over a
Figure/Axes/Artist object model over a swappable backend) because that layering is the reason
matplotlib survived twenty years of changing output targets.

**It is not** an interactive plotting window. Bantu has no `exec`, no `system`, no shell builtin and
no GUI binding, so there is nothing to hand a file to. `show()` therefore writes the file and prints
its path, or serves it over `sua.server` and prints the URL. That is a real limitation, stated in
the docs rather than worked around.

**It is not** a raster library until B6. Anti-aliased rasterisation means touching every pixel; a
1000×1000 image is a million pixels at ~0.38 µs per interpreted operation, which is still most of a
minute per figure.
That half cannot be pure Bantu and is designed as a native backend below (§10).

---

## 2. Why the whole of B1–B5 can be pure Bantu

The load-bearing question for numba was "can the inner loop be interpreted?" — and the answer was
no, which is why numba has a native core. For bplot the same question has a different answer,
because **the work is proportional to the number of drawn elements, not the number of data points**.

A chart is a few hundred SVG elements. A 100,000-point line is *one* `<polyline>`. An 800×600 figure
has perhaps 20 ticks, 40 tick labels, 5 text items, a legend and a handful of paths. At the
interpreter's measured ~0.38 µs per operation, a few thousand operations is around a millisecond. The
interpreter is fast enough for the object model, the layout, the scales and the tick algorithm.

**But it is not fast enough for the naive way of building the output string**, and that is the one
finding that shaped this design.

### 2.1 The finding: string concatenation is O(n²)

Building SVG by repeated `$svg = $svg + "<circle .../>"` is what every tutorial does and what the
first draft of this design assumed. Measured:

| appends | wall clock | output size |
|---|---|---|
| 20,000 | **1,116 ms** | 580 KB |
| 40,000 | **6,752 ms** | 1.16 MB |

Doubling the work multiplied the time by **6.05**. That is quadratic, and the cause is structural:
`evalBinaryOp` builds `Value(left.stringVal + right.stringVal)`, a fresh `std::string` holding a full
copy of the left operand. Forty thousand appends to a growing 1.16 MB buffer is ~23 GB of `memcpy`.

A 100,000-point scatter built this way would take upwards of forty seconds and the B1 stress gate
would fail outright — not slowly, but visibly, on the first realistic figure anyone drew.

**Two fixes, and bplot takes both.**

### 2.2 Fix one — `join()`, the missing inverse of `split()`

Bantu has `split(s, sep)` and has never had `join`. Its absence is why the quadratic pattern is the
only pattern available. With `join`, the idiom becomes *accumulate parts in a list, join once*:

```bantu
$parts = [];
push($parts, "<rect .../>");        // O(1) amortised — push was fixed in numba's Phase A
push($parts, "<text ...>label</text>");
$svg = join($parts, "");            // one pass, one allocation
```

`push` is already linear (80,000 pushes measured at **200 ms**, 40,000 at 131 ms — sub-quadratic,
dominated by the ~2.5 µs builtin-call overhead). `join` walks the list once and reserves the exact
total length, so the whole build becomes O(n) with a single allocation of the final size.

This is a language fix, not a bplot fix. It benefits every Bantu program that assembles text —
HTML templating in `sua`, CSV writing, log formatting — and it closes an asymmetry (`split` without
`join`) that was always a gap. It landed in **B0**.

> **Update, after B0.** This section originally went on to reject fixing `+` itself, on the grounds
> that it would need a rope or a refcounted builder inside `Value`. That was wrong, and the fix
> landed too: when the result of `x + y` is assigned straight back to `x`, the old value is dead and
> can be appended to in place — a peephole in `evalAssign`, no change to `Value`, and what CPython
> does. **`$s = $s + $part` is now linear**, 6,752 ms → **23 ms** at 40,000 appends. `join` remains
> the right idiom when the pieces are already a list, and it is what bplot uses; but a user who
> reaches for `+=` no longer falls off a cliff. See
> [`interpreter-performance.md`](interpreter-performance.md) §3.

### 2.3 Fix two — path simplification, which is the *real* answer

`join` turns forty seconds into something tolerable, but the deeper point is that **a 100,000-point
line has no business emitting 100,000 coordinates.** An 800-pixel-wide canvas can show at most ~800
distinct columns; at two vertices per column (the minimum and the maximum of the samples falling in
it) the rendered polyline is visually identical to the full one at ~1,600 vertices instead of
100,000 — a **60× reduction** in output size and build time.

matplotlib does exactly this (`path.simplify`, on by default since 1.2) for exactly this reason.
bplot does it too, on by default, switchable off:

> For each pixel column the line crosses, keep the first point, the last point, the minimum-y point
> and the maximum-y point. Emit those in x order.

This preserves every vertical extent — spikes do not disappear, which is the failure mode of naive
decimation — while bounding the output at `4 × canvas_width` vertices regardless of input size. The
binning is a single pass and, when numba is present, is a native `nd_*` reduction rather than a
Bantu loop.

**Consequence for the design:** the SVG backend never sees more elements than the canvas can
resolve. The 100k-point stress gate stops being about string throughput and becomes a correctness
question — does the simplified path look like the real one? — which is the question that actually
matters.

### 2.4 The remaining language gaps, and what B0 does about each

| gap | measured / observed | resolution |
|---|---|---|
| no `join` | string building is O(n²) — §2.1 | **add `join(list, sep)`** (B0) |
| `max(1, 2, 9)` returns **2** | silently wrong; `max`/`min` read `args[0]`/`args[1]` and ignore the rest | **make them variadic, and make them reduce a list** (B0) |
| `str(1e21)` returns `"-9223372036854775808"` | `Value::toString` casts any integral double to `long long`; out-of-range is undefined behaviour | **range-guard the cast** (B0) |
| no `PI`, `E`, `exp`, `atan2`, `asin`, `acos`, `log10`, `log2`, `hypot` | cannot draw a pie slice, a log axis or an arrowhead without them | **add them, plus the hyperbolics, `cbrt`, `trunc`, `sign`, `fmod`, `clamp`, `degrees`, `radians`** (B0) |
| no way to *test for* `inf`/`NaN` | they are producible (`log(0)` → `-inf`, `sqrt(-1)` → `nan`) but there is no `isnan` | **add `isnan`, `isinf`, `isfinite`, and the `INF`/`NAN` constants** (B0) |
| `str()` gives 6 significant digits and leaks `1.23457e-05` | axis labels would read `1.23457e-05` | **`bplot._fmt(x, decimals)` in pure Bantu** (B1) — see §7 |
| `open()`/`writefile()` never set `std::ios::binary`, and `open(p,"wb")` silently opens for *reading* | latent Windows corruption today; blocking for PNG | **fixed in B6**, where the byte-identical-PNG gate proves it |
| C-style `for` silently stops at 100,000 iterations | `evaluator.hpp`: `while (cond && safety < 100000)` | **bplot uses `while` exclusively**, as numba does |

Everything else bplot needs — `_repeat`, `_pad`, number formatting, colour interpolation, the tick
algorithm, layout — is ordinary Bantu, and `arctic.b` already hand-rolls string helpers of exactly
this shape.

**So: B1 through B5 are 100% pure Bantu on top of three small, generally useful language additions.**
That is a different claim from "zero interpreter changes", which was the original assumption and
which the measurement disproved. It is recorded here rather than quietly dropped.

---

## 3. The object model

matplotlib's layering, reproduced, because it is the reason matplotlib outlived Agg, Cairo, PDF, SVG
and WebAgg without an API break:

```
  plt.*            stateful convenience API — the only thing in the quickstart
    └── Figure     the page: size, dpi, background, a list of Axes, the layout engine
          └── Axes the plotting box: two Scales, ticks, a list of Artists, a legend
                └── Artist  one drawable: Line2D, PathPatch, Rectangle, Text, Image
                      └── Backend  turns Artists into bytes (SVG now, raster at B6)
```

**Rendering is a single depth-first walk** producing a list of strings, joined once (§2.2). Artists
never write to a shared string and never know what backend they are in; they call
`backend.line(...)`, `backend.text(...)`, `backend.path(...)`. That is what makes B6's raster backend
a drop-in rather than a rewrite.

### 3.1 Three Bantu constraints that shape it

**Class instances have reference semantics.** Verified:

```bantu
$a = make_box();  $b = $a;  $b.add(1);   // $a sees it
mutate($a);                              // passed to a function: $a sees it
$lst = [$a];  $lst[0].add(7);            // through a list: $a sees it
$d = {"ax": $a};  $d["ax"].add(8);       // through a dict: $a sees it
```

All four mutate the same object. This is what makes `$ax = plt.subplot(2,2,1); $ax.plot(...)` work —
the Axes the user holds is the Axes inside the Figure, not a copy. **Bantu lists are the opposite**
(value semantics), so the Figure holds its Axes in a list of *instances*, and that is safe, while a
list of plain dicts would not be.

**`new alias.Class()` does not parse.** So every type is reached through a factory function:
`bplot.figure()`, `bplot.subplots(2, 2)`. The classes themselves are never named by user code.

**`classRegistry_` is a flat global map**, and arctic already owns `Series`, `DataFrame`,
`LazyFrame`, `GroupBy`. Every bplot class is therefore prefixed: `BPlotFigure`, `BPlotAxes`,
`BPlotLine`, `BPlotText`, `BPlotSvg`. Unprefixed names are a collision waiting for the first user who
includes both libraries.

### 3.2 The stateful layer

`plt.plot(...)` with no Figure in hand creates one (matplotlib's `gca()`), so the quickstart is three
lines:

```bantu
include "bplot" as plt;
plt.plot([1, 2, 3], [2, 4, 9]);
plt.savefig("chart.svg");
```

The current-figure state is a module-level variable in `bplot.b`. Module state is per-module and
per-process; **a `sua` handler must not rely on it**, because handlers run on detached threads and
would share it. The documented pattern for servers is the object API:
`$fig = plt.figure(); $ax = $fig.add_axes(); ...; $svg = $fig.to_svg();` — no global state touched.
This is called out prominently in `docs/bplot.md`, and the sua sample uses only the object API.

---

## 4. Coordinates and transforms

Four spaces, composed in this order:

```
 data  --scale-->  normalised [0,1]  --axes box-->  figure px  --flip y-->  SVG user units
```

- **data → normalised** is the Scale's job (§5) and is the only non-affine step.
- **normalised → figure** is the Axes rectangle: `x_px = ax.left + u * ax.width`.
- **figure → SVG** flips y, because SVG's origin is top-left and a chart's is bottom-left:
  `y_svg = fig.height - y_fig`.

The flip is applied **once, at the backend boundary**, never inside an Artist. Artists that forget it
are the single most common bug in hand-rolled plotting code, and putting it in exactly one function
makes forgetting impossible.

Clipping uses one `<clipPath>` per Axes, referenced by every data artist in it, so out-of-range
points are handled by the renderer rather than by filtering (which would break line segments that
enter and leave the view).

---

## 5. Scales

| scale | forward | inverse | degenerate input |
|---|---|---|---|
| linear | `(x - lo) / (hi - lo)` | `lo + u*(hi-lo)` | `hi == lo` → expand by ±0.5, or ±5% of \|lo\| |
| log | `(log10 x - log10 lo) / (...)` | `10^(...)` | **non-positive data raises with the offending value named** |
| symlog | linear within ±`linthresh`, log outside | piecewise inverse | handles zero and negatives by construction |

Log with zero or negative data is the classic silent-garbage case: `log10(0)` is `-inf`, which
propagates to an `NaN` coordinate and emits `<polyline points="NaN,12 ...">`, which browsers render
as *nothing at all*, with no error anywhere. bplot raises instead, names the value, and suggests
`symlog`. That is adoption rule 7 ("errors teach") applied to the place it is most needed.

---

## 6. Ticking — nice numbers

matplotlib's `MaxNLocator`, reimplemented, because "pick ~6 round numbers spanning this range" is the
single thing that separates a chart that looks designed from one that looks generated.

```
given lo, hi, and a target count n (default 6):
  raw   = (hi - lo) / n
  mag   = 10 ^ floor(log10(raw))
  for step in [1, 2, 2.5, 5, 10]:            # the candidate multipliers
      if step * mag >= raw: chosen = step * mag; break
  first = ceil(lo / chosen) * chosen
  emit first, first+chosen, ... while <= hi + epsilon
```

Two details that are not optional:

- **`2.5` belongs in the candidate list.** Without it a 0–1 range with n=6 gives steps of 0.2
  (6 ticks) where 0.25 (5 ticks) is the better choice; matplotlib includes it and charts look wrong
  without it.
- **The `epsilon` on the upper bound must be relative**, `1e-10 * max(|lo|, |hi|)`, not absolute.
  Floating-point accumulation otherwise drops the final tick on ranges like 0–1 in steps of 0.1, and
  the missing top tick is exactly the one a reader looks for.

Ticks are computed from the *view* limits, and view limits are computed from the data limits with a
5% margin (matplotlib's default), then rounded outward to tick boundaries when the user has not set
limits explicitly. Log scales locate ticks at decades, with minor ticks at 2..9 × decade.

**Computing data limits over 100,000 points must not be a Bantu loop** (100k × ~0.38 µs = ~38 ms per
axis). It is one native call: `max($list)` / `min($list)` over a list (B0 makes them accept one), or
`nd_max` / `nd_min` when the data is already a numba array (B4). NaN is skipped for limits — the
limit routine filters explicitly rather than relying on `max`, because the *language* `max`
propagates NaN by design (§9.2).

---

## 7. Number formatting

`str()` cannot be used for tick labels. Measured:

| expression | `str()` gives |
|---|---|
| `1.0/3.0` | `0.333333` (six significant digits) |
| `0.000012345678` | `1.23457e-05` (scientific notation leaks out) |
| `0.1 + 0.2` | `0.3` (the error is hidden — fine for display, fatal for assertions) |
| `1e21` | `-9223372036854775808` — **a defect**, fixed in B0 |

So `bplot._fmt(x, decimals)` is written in pure Bantu: scale by a power of ten, round, split into
integer and fractional parts, and build the digits from `str()` of the *integer* parts, which is
exact and which the B0 range-guard fix makes reliable at every magnitude. An axis chooses its own
decimal count from its tick step (`decimals = max(0, -floor(log10(step)))`), which is why every tick
on an axis has the same number of decimals — a small thing that makes a chart look finished.

Two derived formatters: **scientific** (`1.2 × 10³`, with a shared exponent in the corner when the
range warrants it, as matplotlib does) and **engineering** (`1.2k`, `3.4M`).

**A rule for every bplot test, inherited from numba:** never assert on a stringified number.
`_fmt` itself is tested against explicit expected strings — it *is* the formatter, so its output is
the contract — but no geometric or numeric assertion anywhere else goes through `str()`.

---

## 8. The SVG backend

A backend is an object with a fixed method set — `line`, `polyline`, `path`, `rect`, `circle`,
`text`, `group`, `clip`, `image` — plus `open()`/`close()`. The SVG one appends strings to a list;
the raster one (B6) will call native primitives. Nothing above the backend knows which it is.

```xml
<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480" viewBox="0 0 640 480">
  <rect width="640" height="480" fill="#ffffff"/>
  <defs><clipPath id="c0"><rect x="80" y="40" width="500" height="380"/></clipPath></defs>
  <g clip-path="url(#c0)"><polyline points="80,420 ..." fill="none" stroke="#1f77b4" stroke-width="1.5"/></g>
  <g font-family="sans-serif" font-size="11" fill="#333333"> ... ticks and labels ... </g>
</svg>
```

Coordinates are emitted with **two decimals**. SVG user units are pixels; a hundredth of a pixel is
below any display's resolution, and it roughly halves the file size against the six digits `str()`
would otherwise produce.

### 8.1 Security: SVG is an active document, and this is the part that has teeth

**SVG is not an image format in the sense PNG is.** It is XML that browsers execute: it may contain
`<script>`, `on*` event handlers, `<foreignObject>` with arbitrary HTML, and external references.
The MIME type `image/svg+xml` is already registered in `mime_types.hpp:41`, so `sua` will happily
serve bplot output — which means **a chart title is an XSS vector** the moment any label comes from
a request parameter, a database row or a CSV:

```bantu
plt.title($req.query["name"]);     // name = </text><script>fetch('/steal?c='+document.cookie)</script>
```

Rendered naively, that closes the text element and injects a script that runs with the origin's
privileges when the SVG is opened directly or embedded via `<object>`/`<iframe>`. This is the same
class of defect as the `sua.udp` `maxBytes` crashes — unvalidated user input reaching a place that
trusts it — and it is the reason this section exists before any code is written.

**Four rules, enforced in the backend rather than left to callers:**

1. **Every text node and every attribute value is escaped**, always, with no opt-out: `&` → `&amp;`
   first, then `<` → `&lt;`, `>` → `&gt;`, `"` → `&quot;`, `'` → `&#39;`. Escaping happens in
   `_esc()`, which the backend applies at the point of emission — not at the point of assignment —
   so there is no path by which an artist can bypass it.
2. **Control characters are stripped.** Bytes below 0x20 other than tab, newline and carriage return
   are not legal in XML 1.0 at all; a single one makes the whole document fail to parse, turning a
   chart into a blank page. They are dropped rather than escaped.
3. **There is no raw-SVG escape hatch.** No `plt.raw_svg(...)`. A library that offers one has
   delegated its security to every caller, and the first caller to pass user text through it
   reintroduces the hole. Users who want custom marks get parameterised primitives.
4. **Colour and font inputs are validated, not interpolated.** A colour is a named colour, a
   `#rgb`/`#rrggbb` hex string, or an `[r,g,b]`/`[r,g,b,a]` list — checked against those shapes and
   rejected otherwise. Colours and fonts reach *attributes*, and an unvalidated attribute value is
   the second-favourite SVG injection point after text.

**And one rule for users**, documented in `docs/bplot.md` under a heading they will actually read:
serving user-influenced SVG from your own origin is serving user-influenced active content. Serve it
from a separate origin, or with `Content-Security-Policy: default-src 'none'; style-src 'unsafe-inline'`,
or as a PNG (B6). bplot's escaping makes injection through *bplot* impossible; it cannot make SVG
stop being an executable format.

The B1 test suite asserts this directly: a title of `</text><script>alert(1)</script>` produces
output containing neither `<script` nor an unescaped `</text>`, and the document still parses.

### 8.2 Fonts and text metrics

Text is emitted as `<text>` with `font-family="sans-serif"` — the renderer picks the font, which
means bplot does not know the exact advance widths. It needs approximate ones anyway, for
`tight_layout` to reserve space for tick labels and for legend boxes to fit their entries.

An **embedded Helvetica advance-width table** (the standard AFM values for ASCII 32–126, in 1/1000
em) is the answer: Helvetica and Arial share metrics, and they are what `sans-serif` resolves to on
the large majority of systems. Widths are therefore accurate where it matters and approximate
elsewhere; layout adds a small pad so an approximation errs toward whitespace rather than collision.
Non-ASCII characters fall back to the width of `n`, and CJK to one em.

This is a pure-Bantu data table. There is no font parsing, no FreeType, no new dependency — the same
reasoning that produced a from-scratch P-256 rather than linking OpenSSL.

---

## 9. Details that will otherwise be discovered the hard way

### 9.1 NaN and infinity in the data

They are producible in ordinary Bantu (`log(0)` → `-inf`, `sqrt(-1)` → `nan`, `pow(10,400)` → `inf`)
and they arrive constantly from real data. Untreated, a NaN coordinate emits
`points="NaN,12 40,NaN"`, which renders as **nothing**, silently.

bplot's rule matches matplotlib's: **NaN breaks the line.** A polyline containing NaN is split into
several polylines at the gaps, so the valid segments still draw and the gap is visible as a gap.
Infinities are clipped to the view box. Data limits skip both. This requires `isnan`/`isfinite`,
which is why they are B0 builtins and not an afterthought.

### 9.2 `max`/`min` propagate NaN, and bplot filters separately

The B0 `max`/`min` return NaN if any argument is NaN — matching NumPy's `max` (and its separate
`nanmax`), and matching what a language-level primitive should do: it must not silently hide a
value it was handed. bplot's data-limit routine therefore filters NaN explicitly rather than relying
on `max` to skip it. The two behaviours are correct for their own levels, and confusing them is how
an empty chart with plausible axes gets shipped.

### 9.3 Degenerate inputs, each with a defined answer

| input | answer |
|---|---|
| empty series | draws the axes, no line — not an error |
| one point | a point, with limits expanded to ±0.5 around it |
| all y identical | a flat line, limits expanded by ±5% of \|y\| (or ±0.5 at zero) |
| `x` and `y` different lengths | **raises**, naming both lengths |
| all values NaN | draws the axes; limits fall back to 0–1 |
| zero-size figure | **raises** |
| log scale, non-positive data | **raises**, naming the value, suggesting `symlog` |

"Renders empty" and "raises" are both acceptable; "emits malformed SVG" is not, and that is the B1
gate.

### 9.4 Colour

The default cycle is matplotlib's `tab10`, because those ten colours are the most-recognised
categorical palette in data visualisation and they are chosen for distinguishability. Colormaps
(`viridis`, `plasma`, `coolwarm`, `gray`) are embedded as 256-entry tables of RGB triples — the same
"data, not algorithm" choice as the font metrics. `viridis` is the default continuous map because it
is perceptually uniform and survives greyscale printing and the common colour-vision deficiencies,
which `jet` famously does not; `jet` is not shipped.

---

## 10. The raster backend (B6), and why it is native

| | SVG (B1–B5) | raster (B6) |
|---|---|---|
| work per figure | proportional to **elements** (~10³) | proportional to **pixels** (~10⁶) |
| at ~0.38 µs per interpreted op | ~1 ms | **most of a minute** |
| verdict | pure Bantu | must be native |

B6 adds `bp_*` primitives: a scanline anti-aliased polygon rasteriser, stroke-to-path conversion, and
a PNG encoder (CRC32 + Adler-32 + deflate). A working CRC32 + Adler-32 + stored-deflate PNG writer
already exists in the tree at `init_templates.hpp:915-1010`, private to the PWA icon generator — it
is the reference to port, and stored-deflate is a correct (if large) starting point that a real
deflate can replace behind the same interface.

B6 also carries the binary-file fix: `open()` and `writefile()` never set `std::ios::binary`, and
`open(path, "wb")` falls through the mode chain and **silently opens the file for reading**. On
Windows that corrupts every `\n` in a PNG into `\r\n`. The gate that proves the fix is
**byte-identical PNG output on Linux, macOS and Windows** — a test that cannot pass by accident.

---

## 11. Interop — one library, four input types

bplot accepts, everywhere a sequence is expected:

| input | how it is read |
|---|---|
| a Bantu list | directly |
| a numba `ndarray` | `nd_get` for small, native reductions for limits, native binning for `hist` |
| an arctic `Column` | zero-copy borrow into an ndarray via `nd_from_column`, then as above |
| an arctic `Series` / `DataFrame` | `.to_ndarray()`, added in numba's Phase 6 |

The conversion happens once, at the API boundary, into one internal representation. **Identical
output from all four input types for the same data is the B4 gate**, because "works with lists,
subtly different with arrays" is the failure this design is meant to prevent.

`df.plot()` on an arctic DataFrame is the headline convenience: it is what makes the two libraries
feel like one.

---

## 12. The six questions

**Scalable?** Yes, and in the dimension that matters. Output is bounded by canvas resolution, not by
input size, because of path simplification (§2.3) — a 10-million-point series produces the same
~1,600-vertex polyline a 100,000-point series does. Reductions over large inputs go to numba's native
kernels. The parts that scale with data are native; the parts that scale with chart complexity are
interpreted and small.

**Maintainable, long-term?** Yes. No new runtime dependency — no Cairo, no FreeType, no libpng, no
zlib. The backend boundary is one interface with nine methods, which is what lets B6 add raster
without touching a single Artist. Font metrics and colormaps are data tables, not code.

**Easy, and the Bantu way?** Yes: the arctic and numba pattern reapplied — a thin stateful façade
over an object model, factory functions instead of `new`, prefixed class names, `verb_noun` naming,
`has_native` probes for optional capability, suite docs, and `plt.help()` inside the language.
Anyone who has used matplotlib can guess the call and be right.

**Documentable and testable?** Yes, and better than usual: SVG is text, so tests assert on the
*document* — that it parses as well-formed XML, that it contains the expected elements, that a
hostile label did not survive unescaped, and that it matches a golden file. Every documented example
and every sample is executed by CI, so none can rot.

**Efficient?** Yes, with the honest caveat that this is a *renderer written in an interpreted
language*: expect single-digit to low-tens of milliseconds per figure, not the sub-millisecond a C
library would take. That is invisible for a report and fine for a web request. The numbers go in the
CHANGELOG as they are measured, per phase.

**Secure?** This is where bplot has real teeth, and §8.1 is the answer: SVG is an active document
format, `sua` already serves `image/svg+xml`, and an unescaped chart title is a stored XSS. Escaping
is mandatory and unbypassable, there is no raw-SVG hatch, colours and fonts are validated rather than
interpolated, control characters are stripped, and the deployment caveat is documented where users
will read it. Additionally: no bplot path allocates based on unvalidated input size (the figure
dimensions are checked), and no bplot path writes a file it was not asked to write.

---

## 13. Rejected alternatives

**A native rendering core for B1.** Rejected: the work is proportional to elements, not pixels
(§2), so the interpreter is fast enough, and a native core would put the tick algorithm, the layout
engine and the colour handling — all of which change often — behind a rebuild. B6 makes native
exactly the part that must be.

**Emitting SVG by string concatenation.** Rejected by measurement: O(n²), 6.75 s for 1.16 MB (§2.1).

**Plain dicts instead of classes.** Tempting — no flat-registry collisions, no factory functions.
Rejected because Bantu lists and dicts have *value* semantics while class instances have *reference*
semantics (§3.1), and `$ax = plt.subplot(...); $ax.plot(...)` mutating the figure is not optional.

**Wrapping an existing plotting library.** There is nothing to wrap; Bantu has no FFI to a C plotting
library that would not become a mandatory dependency, and the house rule that produced a from-scratch
P-256 rather than linking OpenSSL applies with more force to something as replaceable as a renderer.

**PNG first.** Rejected: it makes the whole library wait on a rasteriser, a deflate implementation
and a binary-file fix before anyone can draw a line. SVG is text, renders in every browser, scales
without loss, is what a report or a web dashboard actually wants, and `sua` already serves it.

**`jet` as the default colormap.** Rejected: it is not perceptually uniform, it invents structure
that is not in the data, and it is unreadable in greyscale and under common colour-vision
deficiencies. `viridis` is the default and `jet` is not shipped at all.

**A raw-SVG escape hatch.** Rejected — §8.1, rule 3.

---

## 14. Phases

Full table with gates in [`bplot-suite/ROADMAP.md`](../bplot-suite/ROADMAP.md). In brief:

| phase | lands |
|---|---|
| **B0** | this document, the suite docs, and the language gaps of §2.4: `join`, variadic `min`/`max`, the `str()` range fix, and the scalar maths surface |
| **B1** | Figure/Axes/Artist, linear scales, ticking, the SVG backend, `plot`/`scatter`/`bar`/`barh`, labels, legend, grid, limits, `savefig`, `help()` |
| **B2** | `hist`/`boxplot`/`violin`/`errorbar`/`fill_between`/`step`/`stem`/`pie`, log and symlog, date axes, annotations |
| **B3** | subplots and `GridSpec`, twin and shared axes, `tight_layout`, colormaps, colorbars, `imshow`/`contour`/`pcolormesh`/`heatmap` |
| **B4** | numba and arctic integration everywhere; `df.plot()` |
| **B5** | package, publish, `docs/bplot.md`, `samples/bplot/`, the sua serving example |
| **B6** | the native raster backend: rasteriser, deflate, CRC32, PNG, and binary-safe file writes |

Each phase passes the same five-tier gate numba uses — feature, differential, **stress**, regression,
sanitizers — with the numbers recorded in [`bplot-suite/CHANGELOG.md`](../bplot-suite/CHANGELOG.md).

---

## Sources

- John Hunter and Michael Droettboom, *matplotlib*, in **The Architecture of Open Source
  Applications, Volume II** — the Figure/Axes/Artist/Backend layering and why it survived.
  <https://www.aosabook.org/en/matplotlib.html>
- matplotlib `ticker.MaxNLocator` — the nice-number step candidates `[1, 2, 2.5, 5, 10]`.
  <https://matplotlib.org/stable/api/ticker_api.html>
- matplotlib `path.simplify` — path simplification on by default, and why.
  <https://matplotlib.org/stable/users/explain/artists/performance.html>
- Talbot, Lin and Hanrahan, *An Extended Work of Nice Numbers for Tick Labelling*, IEEE InfoVis 2010
  — the formal treatment of the same problem.
- **OWASP**, *XSS Filter Evasion* and *Content Security Policy* cheat sheets — why `image/svg+xml`
  from an untrusted source is active content.
  <https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html>
- W3C, *SVG 1.1 §G.1 — Scripting and SVG*, and *XML 1.0 §2.2 Characters* (why control bytes make a
  document unparseable rather than merely ugly).
- Nathaniel Smith and Stéfan van der Walt, *A Better Default Colormap for Matplotlib*, SciPy 2015 —
  the case for `viridis` over `jet`. <https://bids.github.io/colormap/>
- Adobe, *Helvetica AFM metrics* (Core 14 fonts) — the embedded advance-width table.
- ISO/IEC 14882 §7.3.10 [conv.fpint] — converting a floating-point value outside the destination
  integer range is undefined behaviour; the cause of the `str(1e21)` defect.
