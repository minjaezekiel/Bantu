# pillow — Decisions

The choices that shaped the package, each with the alternative it beat. Referenced by number from
[`docs/pillow-architecture.md`](../docs/pillow-architecture.md) and [`ROADMAP.md`](ROADMAP.md).

---

### PW1 — The pixel store is one numba array of shape `(h, w, bands)`

**Alternative:** a Bantu list of rows of pixels, or a flat byte-list.

Measured: a list index is 587 ns, so one pass over a 1080p RGBA image is 4.9 s as a list and
**0 to 150 ms as an array**. A list store of 8.3 M values also costs ~100 bytes per value in `Value`
overhead — roughly 830 MB for one 1080p image.

The trailing band axis is kept even for one-band modes (`L`, `1`, `P`, `I`, `F`) so that every
operator is written once instead of twice with a rank test, and so that broadcasting a per-band
constant is the natural expression.

### PW2 — No loop in the package may be proportional to the pixel count

This is the rule the whole design exists to satisfy, and it is enforced by a *test*, not by review:
tier 3 asserts the timing curve is linear in pixel count across five sizes. A pixel loop shows up as
a 500×-slower constant, which no eye has to catch.

Explicitly permitted loops: bands (≤4), kernel taps (≤25), palette entries (≤256), output columns in
a resize pass (hundreds), **scanlines** (thousands), and bytes of a *compressed* stream in a codec.

### PW3 — Two interpreter atoms, no more; and the package ships a Bantu twin of one of them

The instruction is a package written entirely in Bantu. It is, and this is what that means precisely:
every codec, operator, filter, rasteriser and font reader is `.b`. The interpreter supplies two
**generic byte atoms** — the same arrangement `hash` has with `band`/`rotr`/`add32` and `numba` has
with `nd_*`, where the package is the implementation and the interpreter supplies arithmetic.

Both atoms are halves of primitives the interpreter already ships, and both absences are defects
that cost every Bantu program:

- `bp_zlib` compresses; **nothing decompresses**. Bantu can write a PNG, a gzip body or a ZIP member
  and cannot read one back.
- `nd_fromfile` exists but its dtypes are all 8 bytes wide, so **no byte buffer can reach an array**
  except through a `Value` list costing 479 ms and a ~100 MB transient per megabyte.

**And `pillow/inflate.b` ships anyway** — a DEFLATE decoder in Bantu. It is not dead code: it
differential-tests the atom on every run (two independent implementations agreeing is the strongest
cheap evidence either is right, and a decompressor is a parser of hostile input), and it is the
fallback, selected by `has_native("inflate")`, so the package is complete in Bantu alone on any
build. Slowly — ~7 µs per output byte, 1.9 s for a 256×256 RGBA PNG — but completely.

### PW4 — Match Pillow's arithmetic, not the textbook's

`RGB→L` is `(R×299 + G×587 + B×114) / 1000`, integer, truncated. Not the ITU-R BT.601 float
luminance, not BT.709, not a gamma-correct luminance. Pillow's answer is the contract: a user porting
a script wants the same pixels, and a "better" greyscale is a silent behaviour change in their
output.

The same applies to `GaussianBlur` (Pillow's three-box-pass approximation, not a true Gaussian),
`RGB→CMYK` (Pillow's naive `K = 255 − max`, not a colour-managed conversion), and resize weights
(Pillow's 1/2048 fixed point, not float). Where Pillow is arguably wrong, `pillow` is wrong the same
way and says so in the docs.

### PW5 — Separable resampling, with the coefficients precomputed per output column

A `k×k` kernel is `2k` taps when separated. More important: the per-pass work becomes *one array
operation per tap* — a bicubic 1080p horizontal pass is **4 array calls**, not 2 M pixel visits. The
coefficient loop is over output columns (hundreds to thousands), which PW2 permits.

`NEAREST` is a strided `nd_slice`, which is a view, which is free.

### PW6 — `Image.open` is lazy, and the pixel budget is checked from the header

A 4-byte PNG header can declare 60,000×60,000. The check has to happen **before** any allocation, so
`open` reads only enough to answer `mode`, `size` and `info`, and `MAX_IMAGE_PIXELS` (default
89,478,485 — Pillow's number) is enforced there. Decoding happens on `load()`, on first pixel access,
or on the first operation that needs pixels. That is Pillow's own design, and it is a security
property, not an optimisation.

### PW7 — `i64` for the pixel store now; a `u8` dtype in numba later, measured

8 bytes per channel value where Pillow uses 1: a 1080p RGBA image is 66 MB instead of 8.3 MB, and a
4000×3000 RGBA is 384 MB. **Comfortable to ~16 megapixels, heavy above it.**

Adding `DType::U8` to numba is the right long-term fix and it is a change to `ndarray_native` that
every `nd_*` kernel, reduction and ufunc would have to learn — a large, separable piece of work in a
package with its own five-tier gate. Doing it inside `pillow`'s phases would mean shipping neither on
time and risking a package many things already depend on.

So: `i64` now, the limit documented in the architecture rather than discovered by a user, and P9
measures the gain on a 4000×3000 RGBA workload and decides. `pillow` picks up the dtype with **no API
change** — `mode.b` is the one file that names a store dtype.

### PW8 — `ImageDraw` fills by scanline span, and does not anti-alias

Per output row, the edge crossings are sorted and each span is a slice assignment: the loop is over
rows and spans, never pixels. A filled 1080p polygon is ~1,080 iterations.

No anti-aliasing, because **Pillow's `ImageDraw` has none**. Pillow's documented idiom — draw at 4×
and `resize` down — works here and works well, because `resize` is fast. Adding anti-aliasing would
make `pillow`'s output differ from Pillow's on every drawn edge, which fails tier 2 for a feature
nobody asked for.

### PW9 — The JPEG entropy scan stays in Bantu until measurement says otherwise

The Huffman walk is ~8 interpreted operations per coefficient, so a 12 MP JPEG's full scan is ~48 s.
Everything after it is array work and costs a fraction of that.

`draft()` is the answer, and it is Pillow's answer: DCT-domain downscaling decodes a 1/2, 1/4 or 1/8
image without the full scan, which makes a thumbnail of a 12 MP JPEG a ~0.8 s operation.
`thumbnail()` calls it automatically, exactly as Pillow does.

**Trigger for a third atom**, recorded so the decision is not re-argued from scratch: if, after P5
ships, full-size decoding of JPEGs above ~4 MP is a thing users actually do (rather than thumbnailing
them), the fix is one *generic* atom — a canonical-prefix-code scanner, which is to `jpeg.b` what
`band` is to `hash.b` — and not a native JPEG decoder. It is not added speculatively.

### PW10 — `ImageTk`, `ImageQt`, `ImageWin`, `ImageShow`, `ImageGrab` and `ImageCms` are declined, loudly

The first five bind a GUI toolkit or an OS screen-capture API that Bantu has no binding to. There is
nothing to implement; a stub that raised would be worse than an absence, because it would look like a
bug rather than a boundary.

`ImageCms` is different and worth being explicit about: ICC colour management is a library-sized
problem, and a half-implementation would silently produce wrong colour — the worst failure mode
available. So ICC profiles are **preserved** in `info["icc_profile"]` and round-trip through `save`
unchanged; they are simply not *applied*. `show()` writes the file and prints the path, the same
accommodation `bplot.show()` makes for the same reason.

All six exclusions are stated in `docs/pillow.md` beside the modules that do exist.

### PW11 — Match Pillow's arithmetic; fix Pillow's bugs

PW4 says to match Pillow's arithmetic rather than the textbook's, because a user porting a script
wants the same pixels. That stands. It is **not** a licence to reproduce upstream defects, and the
line between the two is drawn here so it is not re-argued case by case:

| | treatment |
|---|---|
| a **deliberate choice** — a legacy formula, a fixed-point scale, a rounding mode, a tie-break rule, an odd-looking default | **match it exactly.** It is the contract. `RGB→L`'s `L24`, the per-channel-rounded YCbCr tables, `getcolor`'s three-value CMYK, `convert("1")`'s `/1000` luminance where `convert("L")` rounds — all matched, all with a named test so a later "cleanup" cannot quietly drop them |
| a **defect** — code wrong by its own logic, a value computed and then discarded, a buffer read past its meaning, an overflow that wraps | **fix it**, state the deviation in the module header, and assert the *containment* of the difference rather than the difference itself |

Two fixes exist under this rule so far:

- **`Convert.c`'s dithered `topalette` ends each row with `e[0]=b0; e[1]=b1; e[2]=b2`** — blue's three
  error accumulators written into all three channel slots. Red's and green's are computed and thrown
  away, and the next row's last column is fed blue's error as if it were red's. Nothing downstream
  depends on it; it simply corrupts the final column of every row after the first. `palette.b` writes
  `r0, g0, b0`. Measured consequence, asserted by `tests/pillow_convert_test.b`: row 0 is identical to
  Pillow everywhere, and the difference is confined to a wedge **at most 4 columns** in from the right
  edge (Floyd–Steinberg sends 3/16 down-and-left, so a corrected last column necessarily changes a few
  pixels inward on later rows). 793 of 815 corpus pixels are still identical.
- **`ImagingConvertRGB2YCbCr` casts with `(UINT8)` and does not clip.** For every real RGB input the
  sums land in 0–255, so clipping and casting agree everywhere it matters — but a hand-made
  out-of-range buffer would *wrap*, turning 256 into 0, a silent corruption that propagates into
  whatever reads the image next. `convert.b` clamps: identical on all valid input, saturating rather
  than wrapping on invalid input.

**The test is the record.** A deviation that is not asserted is indistinguishable from a bug of our
own, so each one gets a test that states the property it preserves, and the differential tier counts
the affected pixels rather than excusing them.

### PW12 — Bantu keeps variables and functions in one namespace

This is a language property with teeth, and it bit twice before being written down: a parameter named
`$grey` replaced `palette.grey()` for the rest of the call, and a parameter named `$palette` replaced
the `palette` module alias, each failing with *"it holds null/a value that is not callable, not a
function"* at a call site nowhere near the cause.

Two rules follow, applied throughout the package:

1. **A module alias is never a word that could be a parameter.** `palette.b` is included as `pal`.
2. **An exported function is never a bare noun a caller would reach for as a local.** `grey()` became
   `grey_ramp()`; a `$grey` parameter became `$lum`.

The interpreter's error message already names the cause, which is why both were quick to find — but
the cure is naming, not vigilance.
