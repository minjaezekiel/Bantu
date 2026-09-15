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
That half cannot be pure Bantu and is designed as a native backend below (§12).

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
| linear | `x` | `u` | `hi == lo` → expand by ±0.5, or ±5% of \|lo\| |
| log | `log10 x` | `10^u` | **non-positive data raises with the offending value named** |
| symlog | linear within ±`linthresh`, log outside | piecewise inverse | handles zero and negatives by construction |

Log with zero or negative data is the classic silent-garbage case: `log10(0)` is `-inf`, which
propagates to an `NaN` coordinate and emits `<polyline points="NaN,12 ...">`, which browsers render
as *nothing at all*, with no error anywhere. bplot raises instead, names the value, and suggests
`symlog`. That is adoption rule 7 ("errors teach") applied to the place it is most needed.

### 5.1 Where the transform is applied, and why it is not in the drawing loop

A scale is *non-affine*, so the obvious implementation calls `fwd(v)` on every point inside every
artist's loop. At the interpreter's ~0.38 µs per operation, a 100,000-point line would pay for
200,000 extra calls — and it would pay them on **linear** axes too, which are the overwhelming
majority, because the branch that decides whether to call would itself sit in the loop.

bplot applies the scale as a **pre-pass over the sequence** instead:

```
artist draws:   if scale is not linear:  vals = _project(vals, scale)     # one pass, once
                then the existing affine loop:  px = v * a + b            # unchanged
```

Three consequences, all of them wanted:

- **A linear axis costs exactly what it cost in B1** — not one extra instruction in the loop, not
  one extra branch. The B1 100k-point measurement stays valid, and it is re-run as a gate.
- **The non-positive check happens in exactly one place.** `_project` is the only function that can
  produce a log coordinate, so it is the only function that can raise — and it names the offending
  value, which a check scattered through four artists would eventually forget to do in one of them.
- Limits, ticks and artists all agree by construction, because all three go through the same
  `_project`.

The cost is one transformed copy of the data per non-linear axis, which is the same order as the
data already in hand and is bounded by path simplification immediately afterwards.

### 5.2 symlog, exactly as matplotlib defines it

Symlog exists to plot data that spans decades *and* crosses zero — residuals, profit and loss,
temperature anomalies. It is linear in a band around zero and logarithmic outside it, so zero is
representable and small values are not crushed against it.

With base 10 and `linscale = 1`, matplotlib's transform is:

```
linscale_adj = linscale / (1 - 1/base) = 10/9 = 1.111…

|x| <= linthresh :   x * linscale_adj
|x| >  linthresh :   sign(x) * linthresh * (linscale_adj + log10(|x| / linthresh))
```

It is continuous at `|x| = linthresh` (both branches give `linthresh * linscale_adj`), which is the
property that makes the axis look like one axis rather than two glued together. The constants are
not folded or simplified here: these values were checked against
`matplotlib.scale.SymmetricalLogTransform` at three `linthresh` settings and ten sample points, and
the reference numbers are embedded in `tests/bplot_charts_test.b` rather than recomputed at runtime.

`linthresh` must be positive and finite, and bplot raises naming the value if it is not — a
`linthresh` of zero would divide by zero and produce the `NaN` coordinates the whole design exists to
prevent.

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

### 6.1 Log ticks — decades, and the case matplotlib gets wrong

Major ticks are the decades `10^k` that fall in view; minor ticks are `2..9 × 10^k`. Checked against
matplotlib across ten ranges (`LogLocator` through a drawn axis, filtered to the view interval), this
agrees exactly — `1..1000` → `1, 10, 100, 1000`; `1e-9..1e-6` → the four decades; `5..5000` →
`10, 100, 1000`.

**There is one range class where matching matplotlib would be wrong, and bplot deliberately
diverges.** Asked for an axis from 2 to 9, matplotlib returns **no major ticks at all** — the axis is
drawn with no labelled tick anywhere on it, because no decade falls inside. The same happens for
`1..3` (one tick, at 1). An unlabelled axis is not a stylistic difference; it is a chart a reader
cannot read.

> **bplot's rule:** if fewer than two decades fall in view, the `2..9 × 10^k` minors are **promoted**
> to labelled major ticks and the minor set is dropped.

So `2..9` gets ticks at 2,3,4,5,6,7,8,9, and `1..3` gets 1,2,3. The test suite asserts both the
agreement on the eight ranges where matplotlib is sane *and* the divergence on the two where it is
not, so neither can regress silently and the divergence is a decision rather than an accident.

### 6.2 symlog ticks

Zero, plus `±10^k` for every decade `k ≥ floor(log10(linthresh))` that falls in view. Verified
against `matplotlib.scale`'s symlog axis for six limit/linthresh combinations, including the
`linthresh = 10` case where the `±1` ticks correctly disappear. Minor ticks are the `2..9` multiples
in each decade outside the linear band; the linear band gets `MaxNLocator` treatment, since inside it
the axis *is* linear.

### 6.3 Date ticks — a calendar ladder, not a decimal one

Time is the one axis where the nice-number algorithm of §6 is actively wrong. `MaxNLocator` on epoch
milliseconds happily proposes a step of 2,500,000,000 ms, which is 28.9 days — a tick every 28.9
days, landing mid-afternoon on drifting dates. Nobody wants that axis.

Date ticking therefore uses a **calendar ladder** of steps that are units people recognise — and
rather than invent a rule for choosing among them, bplot reproduces `AutoDateLocator`'s, which turned
out to be four lines:

```
walk the units coarsest first (year, month, day, hour, minute, second);
take the FIRST whose span is at least 5 of that unit;
within it take the SMALLEST interval satisfying  span <= interval * (maxticks - 1)

intervals   year  [1,2,4,5,10,20,40,50,100,…]   month [1,2,3,4,6]   day [1,2,3,7,14,21]
            hour  [1,2,3,4,6,12]                minute/second [1,5,10,15,30]
maxticks    year 11 · month 12 · day 11 · hour 12 · minute 11 · second 11
```

Ticks are *anchored to the calendar*, not accumulated: sub-day steps anchor to midnight UTC, day
steps to days-of-month (a fortnightly axis is the 1st and the 15th; a weekly one the 1st, 8th, 15th
and 22nd), month steps to a multiple of the interval counted from January, year steps to a multiple
of the interval. Anchoring is what makes a monthly axis land on the 1st of each month rather than on
30.44-day intervals from an arbitrary origin — and months and years are not fixed-length, so
accumulating milliseconds cannot produce them at all.

Calendar arithmetic is Howard Hinnant's `civil_from_days` / `days_from_civil` pair — twelve lines,
exact for every proleptic-Gregorian date, no lookup table and no leap-year special-casing beyond the
shifted-year trick. **All bplot date handling is UTC**, matching arctic's storage (epoch
milliseconds, UTC — `dataframe_native.hpp:77`); local-time rendering would need a timezone database,
which is not a dependency this tree is going to take for an axis label.

Label formats follow the step, and are matplotlib's defaults (`date.autoformatter.*`, checked
against 3.11.2): seconds `HH:MM:SS`, minutes `DD HH:MM`, hours `MM-DD HH`, days `YYYY-MM-DD`, months
`YYYY-MM`, years `YYYY`.

> **Corrected in B4.** This line read "`HH:MM:SS`, `HH:MM`, `MM-DD HH:MM`, `YYYY-MM-DD`, `YYYY-MM`,
> `YYYY`" — which already intended a dated hour label, but the code shipped bare `HH:MM` for both
> minutes and hours. Any axis crossing midnight read `12:00 00:00 12:00`, with no way to tell which
> day a tick belonged to. B4's first datetime column, four days long, produced exactly that. Minute
> and hour labels now carry the day, as matplotlib's do.

Parity is **exact** on the eight ranges checked, from a ten-second window to a
twenty-five-year one — including the two that a look-alike implementation gets wrong: a
three-month range, where matplotlib switches to *semi-monthly* ticks (the 1st and the 15th) rather
than to months or to 14-day intervals, and a twenty-five-year range, where it picks a **four**-year
interval anchored on multiples of four (2000, 2004, …) rather than five. The reference tick sets are
embedded in `tests/bplot_charts_test.b` as literal strings.

> An earlier draft of this section hedged — "close but not claimed to be exact" — because the
> intention was to approximate the ladder. Reading `AutoDateLocator.get_locator` instead showed the
> selection rule is four lines, so reproducing it exactly cost less than approximating it and cannot
> disagree on a range nobody thought to check. The hedge is removed rather than left standing.

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

### 7.1 Decade labels

A log axis labels `10^k`, and the readable form depends on `k`. bplot writes the plain decimal while
one exists — `0.001`, `0.01`, `0.1`, `1`, `10`, `1000`, `100000` — and switches to `1e6` / `1e-7`
outside `-4 ≤ k ≤ 5`. The alternative, a superscript `10⁶`, needs either a `<tspan>` with
`baseline-shift` (which the raster backend of B6 would then have to reimplement, for a label) or the
Unicode superscript digits (which are missing from many of the fonts `sans-serif` resolves to, and
render as boxes). `1e6` is unambiguous, is what every language's `str()` of that number produces, and
costs the backend nothing.

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

## 10. The chart types (B2), and the statistics inside them

A chart type is not a drawing routine. `hist` is a binning algorithm, `boxplot` is five order
statistics and an outlier rule, `violin` is a density estimate — and each has a definition that
*other tools already implement*, so bplot's job is to match a definition rather than invent one.
Where a definition exists in the tree already, bplot matches **that**.

### 10.1 Histogram binning matches `nd_histogram`, byte for byte

numba's `nd_histogram` is already in the tree (`ndarray_sort_reg.hpp`), and B4's gate is that the
same data plotted as a list and as an ndarray produces **identical output**. That is only achievable
if the pure-Bantu binner and the native one make the same decisions, so bplot copies `nd_histogram`'s
rules exactly rather than writing the obvious ones:

| rule | value |
|---|---|
| bin index | `floor((x - lo) / width)`, `width = (hi - lo) / bins` |
| top edge | **inclusive** — an `x` equal to `hi` lands in the last bin, not in a phantom bin past it |
| out of range | values `< lo` or `> hi` are **dropped**, not clamped |
| `NaN` | dropped |
| degenerate range (`lo == hi`) | widened to `lo - 0.5 .. hi + 0.5`, which is not an error |
| empty input | range falls back to `0..1` |

The off-by-one at the top edge is the one that bites: without the inclusive rule, the largest value
in the data silently vanishes from the chart, and a histogram missing its maximum is a histogram that
lies about its range. It has a test of its own.

`density` normalises so the bar *areas* sum to 1 (dividing by `n × width`), which is NumPy's
definition — not so the heights sum to 1, which is the common mistake and gives a different shape for
unequal bin widths.

### 10.2 Quantiles are NumPy's linear interpolation, because `nd_quantile` is

```
pos = q * (n - 1);  lo = floor(pos);  hi = ceil(pos)
result = v[lo] + (v[hi] - v[lo]) * (pos - lo)
```

This is NumPy's default (`method="linear"`, Hyndman–Fan type 7) and exactly what `nd_quantile` in
`ndarray_reduce_reg.hpp` computes. There are nine defensible quantile definitions; the one that
matters is the one the rest of the tree already uses, because a boxplot whose median moves depending
on whether the data arrived as a list or an array is worse than any of the nine.

**Boxplot**, then, is Tukey's, as matplotlib draws it: box from Q1 to Q3, line at the median, whiskers
to the most extreme *observed* value within `1.5 × IQR` of the box, and everything beyond drawn as
individual outlier points. The whisker lands on a real data point — not on `Q1 - 1.5×IQR` itself,
which is the frequent error and draws a whisker into empty space where no observation exists.

### 10.3 Violin: a binned kernel density estimate, so cost does not scale with n

The textbook Gaussian KDE evaluates `n` kernels at each of `g` grid points: `O(n × g)`. For 100,000
points on a 128-point grid that is 12.8 million interpreted operations — most of a minute, for one
violin.

bplot bins first. The data goes into a 512-bin histogram in one pass, and the grid is then evaluated
against the **bins** rather than the points:

```
cost = O(n + bins × grid)  =  O(n + 65,536)
```

— independent of `n` beyond the single binning pass. This is the standard binned-KDE approximation
(R's `density()` does the same thing, via an FFT); the error it introduces is bounded by the bin
width, which at 512 bins across the data range is far below the bandwidth the kernel is smoothing
with anyway. Bandwidth is Scott's rule, `1.06 × σ × n^(-1/5)`, matching `scipy.stats.gaussian_kde`'s
default factor.

A violin of fewer than two distinct values has no density to estimate; it degenerates to a flat line
at that value rather than dividing by a zero bandwidth.

### 10.4 The rest, and what is load-bearing in each

| type | shape | the detail that matters |
|---|---|---|
| `errorbar` | line/marker + I-bars | `yerr`/`xerr` accept a scalar, one list (symmetric) or two lists (asymmetric, `[lower, upper]`); **negative error raises** — it is always a bug, and drawn it produces an inverted bar that reads as a smaller error |
| `fill_between` | one polygon per contiguous run | NaN in either edge **breaks the band** into separate polygons, for the same reason NaN breaks a line (§9.1); `where` does the same |
| `step` | polyline with doubled vertices | `where` ∈ `pre`/`post`/`mid`, matching matplotlib's names, because a step chart drawn with the wrong convention is off by one sample and looks plausible |
| `stem` | baseline + stems + markers | the baseline is at `y = 0` by default and is drawn, so the sign of each value is readable |
| `pie` | `<path>` arcs | angles from the **fraction of the total**, negatives raise; slices are drawn from `startangle` counter-clockwise as matplotlib does; `autopct` percentages are computed before rounding so they sum to 100 |

`pie` is also the reason the Axes grows an `axisOff` flag: a pie with a frame, ticks and a grid behind
it is nobody's intent, and `plt.axis("off")` is the matplotlib spelling.

### 10.5 Categorical axes

`bar(["Jan", "Feb", …], rainfall)` is the first thing anyone types. Before B2 it died inside `min()`
with *"element 1 must be a number"* — an error about the wrong thing, three layers below the call the
user made.

A sequence of strings on any axis now becomes positions `0..n-1` with the strings as tick labels. The
mapping lives **on the Axes** and is extended rather than rebuilt, because two series sharing
categories have to line up: numbering the second series from scratch would put its bars under the
first series' labels, which is a chart that lies rather than one that errors. A sequence that mixes
text and numbers raises — an axis is either categorical or numeric, and guessing would silently
misplace every point.

### 10.6 Two new backend primitives, and no more

`polygon(points, fill, stroke, width, opacity)` and `path(d, fill, stroke, width)` join the nine from
B1. Everything in this section is expressible with those two plus what already exists — a pie slice
is an arc path, an arrowhead is a three-point polygon, a violin is a polygon, a filled band is a
polygon. Adding two primitives rather than one per chart type is what keeps B6's raster backend a
finite job: **eleven methods, not thirty**.

`path` takes a `d` string, which is the one place in the backend where a caller could in principle
inject markup. It is therefore **not** a free-text parameter: `d` is assembled by bplot's own arc and
rectangle helpers from numbers that have been through `_px`, and the backend escapes it like every
other attribute. There is still no path by which user text reaches a `d`.

---

## 11. Layout and 2-D (B3)

### 11.1 Subplots: one rectangle calculation, used by everything

`GridSpec` is the only layout arithmetic in bplot. `subplots(nrows, ncols)` builds one, and every
other entry point — a single `figure()`, a `twinx`, an inset — resolves to a rectangle through it:

```
cell(row, col) = left + col * (cellW + wspace),  top + row * (cellH + hspace)
```

with the figure's outer margins reserved first. A `subplot(r, c, index)` call in matplotlib's
1-based, row-major numbering maps onto the same grid, so both spellings produce identical
rectangles — asserted, because "the two APIs drift apart" is how matplotlib's own `subplot` /
`add_subplot` history went.

**Shared axes** (`sharex`, `sharey`) are a *list of peers* on each Axes, and view limits are the union
over the group. They are not a parent pointer: sharing is symmetric, and a parent pointer makes the
first axes special, which then breaks when it is the one removed.

**Twin axes** (`twinx`, `twiny`) are a second Axes with the *same rectangle*, its x limits pinned to
the first and its y ticks drawn on the right. The frame is drawn once, by the original, so the two do
not double-stroke the box — a doubled 1px stroke is visible and looks like a rendering bug.

### 11.2 `tight_layout` measures; the default gutters do not

B1 reserved fixed gutters (62px left, 34 top, 52 bottom, 18 right), which is right for a default and
wrong the moment a y tick label reads `-1.2345e+06` or an axis label is two words long.
`tight_layout` computes each gutter from what will actually be drawn, using the embedded Helvetica
metrics (§8.2):

```
left   = max width of the y tick labels + tick length + ylabel height + pad
bottom = x tick label height + xlabel height + pad
top    = title height + pad
right  = half the width of the last x tick label, so it cannot overhang the figure
```

The measurement is an *estimate*, because the viewer picks the font. Layout therefore always pads
**outward**: an underestimate produces a slightly tight label, an overestimate produces whitespace,
and neither produces the overlap that an exact-fit algorithm produces the first time a font differs.
That asymmetry is deliberate and is why the metrics table does not need to be exact to be useful.

`tight_layout` runs at render time, not at call time, because the labels it measures are usually set
*after* the axes is created. Running it at call time would measure an empty axes — matplotlib has
exactly this trap, and the answer there is "call it last"; bplot removes the trap instead.

### 11.3 Colormaps are data, and the data is the published data

`viridis`, `plasma`, `coolwarm` and `gray` ship as **256-entry tables of the authoritative RGB
values**, embedded as a single 1,536-character hex string per map and sliced six characters at a
time. Provenance is recorded in the source: viridis and plasma are Nathaniel Smith and Stéfan van der
Walt's tables, released into the **public domain (CC0)**; `coolwarm` is Kenneth Moreland's diverging
map as matplotlib samples it.

**They are not approximated.** There is no closed form for viridis — it is the output of an
optimisation in CAM02-UCS perceptual space — so a polynomial fit called "viridis" would be a
different colormap wearing the name, and the whole point of viridis is its perceptual uniformity,
which a fit does not preserve. The tables were generated once, at authoring time, from the published
source and pasted in; **nothing at runtime depends on Python, matplotlib or any download.** The test
suite checks sampled entries against values embedded independently in the test file, so a corrupted
table fails rather than agreeing with itself.

A hex string rather than a list of 768 numbers because it is one twelfth of the source size, it
cannot be half-edited into a valid-but-wrong table, and parsing six hex characters is four
`ord` calls.

**The index is `floor(t × 256)` clamped to 255, not `round(t × 255)`** — matplotlib's own
quantisation. The two formulas look equivalent and agree at 0, 0.5 and 1, so an implementation
written from intuition passes every obvious test; they disagree at 0.625, where `round` gives entry
159 and matplotlib gives 160, a visibly different green in the middle of the most-used colormap in
science. It was caught by checking nine sample points rather than three, and the lesson is recorded
in the test as much as in the code: **sample the interior, not only the endpoints.**

### 11.4 `imshow` — why a 1000×1000 image is downsampled, and what B6 changes

**The naive encoding is one `<rect>` per pixel**: a 1000×1000 array is a million elements at ~55
bytes each — a **55 MB** document that no browser will open happily. The roadmap's B3 gate demanded
"a single embedded image" instead. That gate was written before the encoding question was worked
through, and it is **not achievable in B3**: a single embedded image means
`<image href="data:image/png;base64,…">`, which means a PNG encoder, which means CRC32, Adler-32 and
deflate — the native work that *is* B6. The gate is corrected in the roadmap rather than quietly
dropped, alongside what B3 does instead.

What B3 does instead is two things, and together they are enough for what vector graphics is
actually for:

1. **Block-reduce to a cell budget.** An array larger than `maxcells` per axis (default 256) is
   reduced by block mean before drawing. This is not a compromise imposed by the format — the axes
   box is ~500×380 CSS pixels, so cells beyond that are *already* invisible, and averaging is a more
   honest reduction than the nearest-neighbour sampling a browser would do to the same data.
2. **Batch cells into one `<path>` per colour.** A colormapped image has at most 256 distinct
   colours by construction, so every cell of one colour goes into a single `<path>` element as
   `M x y h w v h z` subpaths, with horizontally adjacent equal-coloured cells merged into one run
   first. A 256×256 image becomes **≤ 256 elements** instead of 65,536, and ~14 bytes per cell
   instead of ~55.

Measured together: a 1000×1000 `imshow` renders as ≤ 256 elements and a file **under 1.5 MB**, in
place of 55 MB of rectangles. `heatmap` and `pcolormesh` share the encoder; `contour` marches squares
over the grid and emits polylines, which are small regardless.

At B6 the same call gains a raster path and the downsample limit goes away. The API does not change,
which is the point of having a backend boundary at all.

---

## 12. The raster backend (B6), and why it is native

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

## 13. Interop — one library, four input types (B4)

bplot accepts, everywhere a sequence is expected, a Bantu list, a numba `ndarray`, an arctic `Column`,
an arctic `Series`, and — where a table makes sense — an arctic `DataFrame`. **Identical output from
all four input types for the same data is the B4 gate**, because "works with lists, subtly different
with arrays" is the failure this design exists to prevent.

The design below is the second one. The first — convert everything to a Bantu list at the boundary,
which is what B1–B3 did for ndarrays and columns — was measured before B4 started, on one
1,000,000-point line:

| step | time |
|---|---|
| `nd_to_list` on x and y | 630 ms |
| data limits | 908 ms |
| `plot()` | 10,915 ms |
| render | 8,171 ms |
| **peak RSS** | **3.1 GB** |

Twenty seconds and three gigabytes for one chart. The cause is not the drawing: a Bantu `Value` is
~190 bytes, a list of a million is ~190 MB, and **lists have value semantics**, so every function
call that takes one as an argument copies all of it. The plotting code was fine; the representation
was wrong.

### 13.1 Recognition is structural, and bplot includes neither library (BP31)

| input | recognised by |
|---|---|
| list | `type($v) == "list"` |
| ndarray | `type($v) == "ndarray"` |
| column | `type($v) == "column"` |
| Series | an instance whose `.col` is a column |
| DataFrame | an instance whose `.names` is a list and `.cols` a dict |

bplot never `include`s arctic or numba. A plotting library that required a dataframe library would
be wrong, and one that included it would load 1,400 lines into every chart. Recognising the *shape*
instead means any future type that wraps a column works with bplot without either side changing.

### 13.2 Numeric data stays native until it becomes pixels (BP32)

For the chart kinds whose inputs are routinely large — `plot`, `scatter`, `hist`, `boxplot`,
`violin` — every numeric input, **lists included**, is normalised at the boundary into one
representation: a 1-D f64 ndarray. Columns get there by a zero-copy borrow; lists by one native
pass. From then on the data is a `shared_ptr`, so passing it between functions costs nothing, and
every reduction over it is a numba kernel.

Because all four input types then run the *same* code over the *same* representation, identical
output is **structural**, not a property that has to be tested into existence. It is tested anyway.

The pure-Bantu path B1–B3 used is kept, for two reasons. An interpreter without numba still plots.
And it is the **differential oracle**: `_useNative(false)` forces it, and the test suite requires the
same data to render byte-identically through both. B1–B3's 477 assertions, all of which pass Bantu
lists, now run through the native path — so every one of them is also a check that the native path
reproduces what they pinned.

Charts whose inputs are small by nature — bar, step, stem, errorbar, fill_between, pie, the 2-D
grids — keep materialising a list at the boundary. A 1,000,000-bar chart is not a chart.

### 13.3 The one new kernel, and why it contains no arithmetic (BP33)

Everything above composes from kernels numba already has — `nd_isfinite`, `nd_compress`,
`nd_sort`, `nd_histogram`, `nd_min`/`nd_max`, `nd_log10`, `nd_where`, `nd_multiply`, `nd_add`. One
thing does not compose: B1's path simplification (BP3), which walks the points in order, splits them
into runs at every non-finite value, and keeps first/min/max/last per pixel column.

That becomes `bp_line_runs(px, py, simplify)`, and **it is handed pixel coordinates, not data**. The
scale transform and the affine map are done first, in Bantu, as separate numba passes. That split is
deliberate: `x * a + b` evaluated by a C++ compiler may be contracted into a fused multiply-add,
which rounds differently in the last bit from the interpreter's two separate operations — and a
different last bit is a different `_px` string on a rounding boundary. Two separate kernel calls
cannot be fused. So the kernel contains nothing but comparisons and `std::round`, which is what the
interpreter's own `round()` calls, and its output is BP3's output by construction.

Scatter gets the same treatment one step further: at **1,000 finite points and above** — the same
threshold BP3 already uses — the circles are emitted as one `<path>` built natively rather than one
`<circle>` element each, reproducing `_fmt`'s arithmetic for every coordinate. Below the threshold
nothing changes, so no existing document moves.

### 13.4 Nulls, datetimes and integers (BP34)

- **A numeric null is a missing value, and becomes NaN**: a gap in a line, skipped by limits,
  histograms and box statistics. That is what pandas does, and it is the only choice that neither
  invents a value nor refuses to draw.
- **A categorical null is labelled `null`**, as `print` would show it.
- **A datetime column becomes epoch milliseconds** and turns its axis into a date axis
  automatically; a date column is days, scaled to milliseconds. That is the integration that makes
  `plot($df.get("day"), $df.get("sales"))` produce a readable time axis with no further call.
- **An i64 column is exact to 2^53**, the same caveat arctic documents as A4.

### 13.5 Frames, and `df.plot()` (BP35)

`plot_frame($df, $opts)` is the table-shaped entry point, on an Axes and at module level:

| option | meaning |
|---|---|
| `kind` | `line` (default), `bar`, `barh`, `scatter`, `hist`, `box`, `step` |
| `x` | the column for the x axis; default is the row number |
| `y` | a column name or a list of them; default is **every numeric column** except `x` |
| `title` | a title; axis labels default to the column names |

More than one series turns the legend on and, for bars, groups them side by side. Naming a column
that is not numeric **raises, naming the column and its type** — silently skipping a column the user
asked for by name is the worst of the three possible answers. Only the *default* selection skips
non-numeric columns, because nobody asked for those.

`heatmap($df)` draws the numeric columns as a matrix with their names as labels.

**`$df.plot($opts)` and `$series.plot($opts)` live in arctic, and include bplot lazily** — inside the
method, on first call. That is pandas' own design, for the same reason: the dependency points from
the convenience to the library, and only when the convenience is used. arctic loads nothing extra
for a program that never plots, and a program without bplot installed gets an error that says
`bantu add bplot` rather than an undefined name.

### 13.6 What B4 found in the language (BP36)

Four defects, each silent, each fixed rather than worked around:

- **`include "bplot"` from a folder containing a `bplot/` directory bound an empty module.** The
  resolver's existence check accepted directories, and a directory parses as an empty file. Modules
  must now be regular files, and a bare name that names a directory is resolved as a package inside
  it (`package.json` `main`, then `<name>.b`, then `index.b`) — Node's convention.
- **`len()` returned 0 for a dict, an ndarray and a column**, so `while ($i < len($a))` over an array
  never ran. A dict now reports its entries, an ndarray its first axis, a column its rows.
- **`contains()` returned false for every list.** It now tests list membership with `==`.
- **The performance wall above**, which is §13.2.

### 13.7 Results

Measured on the build that shipped B4 (Intel Core i7-9750H, macOS, Apple clang), one million rows:

| | native path | pure path, same build |
|---|---|---|
| a line from an arctic column | **252 ms** | 15,455 ms |
| peak resident memory | **112 MB** | 2.69 GB |
| the document | 23,825 bytes | 23,825 bytes — **identical** |
| histogram / two boxes / violin | 61 / 241 / 306 ms | — |
| scatter | 1,523 ms, one 60.9 MB `<path>` | — |

The byte-identical document at a million rows is the strongest form of the §13.2 claim: the native
kernels are not an approximation of the pure path that happens to agree on test-sized data.

`tests/bplot_data_test.b` holds the gate — 17 chart configurations, each rendered from a list, an
ndarray, a column and a Series and through both paths, compared byte for byte.

## 14. The six questions

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

## 15. Rejected alternatives

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

## 16. Phases

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

### 16.1 What B5 holds the package to

A package is finished when a stranger can install it and the documentation does not lie. B5 turned
both into tests rather than checklist items:

- **The documentation runs.** `tests/run_doc_examples.sh` executes every example in `docs/bplot.md`,
  `docs/numba.md` and `docs/arctic.md`, each doc as one program in reading order (BP37). Its first run
  found two broken numba examples that a roadmap had recorded as executed.
- **The package installs.** `tests/bplot_package_test.sh` publishes into a throwaway registry, runs
  `bantu add` in an empty directory, and draws — including a DataFrame through arctic's lazy include,
  which is the path that only exists once both packages live in `bantu_modules/`.
- **The server example is the safe one.** `tests/bplot_sua_test.sh` serves real charts under
  concurrency with a hostile title in the query string (BP38).

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
