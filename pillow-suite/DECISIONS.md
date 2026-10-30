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

**What this paragraph used to say was wrong, and P5 is what showed it.** It said `draft()` "decodes a
1/2, 1/4 or 1/8 image without the full scan, which makes a thumbnail of a 12 MP JPEG a ~0.8 s
operation". Reading `jdhuff.c` and then measuring both say otherwise: scaling happens in the **IDCT**,
and the entropy-coded scan is parsed in full at every scale, because the amplitude bits of every
coefficient are interleaved into the bitstream and nothing says where the next block starts until the
current one is decoded. libjpeg does exactly the same amount of Huffman work for a 1/8 decode as for
a full one.

Measured in P5, Darwin/Apple silicon, a photographic 4:2:0 file:

| image | full decode | `draft` 1/8 | saving |
|---|---|---|---|
| 320×240 | 0.80 s | 0.73 s | 9 % |
| 640×480 | 3.06 s | 2.89 s | 6 % |
| 1024×768 | 7.77 s | 7.21 s | 7 % |

So `draft()` shrinks the IDCT, the upsampling, the colour transform and the output buffer — the cheap
stages — and the cost of reading a JPEG is the scan at every scale. It is still worth having (it is
also the only way to get libjpeg's *exact* scaled pixels), and `thumbnail()` still calls it the way
Pillow does; it is just not the answer to the scan's cost.

**Trigger for a third atom**, recorded so the decision is not re-argued from scratch: ~10 µs per
pixel means a 12 MP JPEG is about two minutes, so if full-size decoding of large JPEGs is a thing
users actually do, the fix is one *generic* atom — a canonical-prefix-code scanner, which is to
`jpeg.b` what `band` is to `hash.b` — and not a native JPEG decoder. It is not added speculatively,
and P5 ships without it.

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

### PW13 — A progressive JPEG's coefficients live in a Bantu list, and so there is a budget

**Alternative:** hold them in a numba array for every file, as the baseline decoder does.

A baseline scan is decoded once, in order, and never read back, so the decoder emits the indices and
values of the non-zero coefficients and `nd_put` scatters them into a zeroed array in 64 K chunks —
memory proportional to the chunk, not to the image.

A progressive scan cannot. Successive-approximation refinement *reads each coefficient back* to
decide whether to consume a correction bit, and `nd_get` at ~2.5 µs would cost more than the whole
scan. So progressive decoding holds the coefficients in a flat Bantu list, which costs a `Value`
each — about 100 bytes:

| progressive image | coefficients | list |
|---|---|---|
| 640×480 RGB 4:2:0 | 0.46 M | ~46 MB |
| 1600×1200 RGB 4:2:0 | 2.9 M | ~290 MB |
| 4000×3000 RGB 4:2:0 | 18 M | ~1.8 GB |

`jpeg.b` therefore checks a **coefficient budget from the frame header**, before a byte of scan is
read — the same shape as PW6's pixel budget — and refuses the third row by name, with the limit, the
arithmetic and the two ways out (re-save as baseline, or raise the budget) in the message. Baseline
files of any size are unaffected, and PW7's `u8` dtype does not fix this: a Bantu *list* is the cost,
not the array.

### PW14 — A Bantu list is copied when it crosses a function boundary or comes out of a dict

This is a language property with the same teeth as PW12, and it decides the shape of every hot loop
in `jpeg.b`. Measured on this build:

| | cost |
|---|---|
| `$local[$i]` — a list read | ~0.5 µs |
| `$state.table[$i]` — a list read **through a dict field** | ~25 µs on a 1,000-entry list, ~500 µs on a 20,000-entry one |
| `f($bigList, $i)` — a user function taking a list | the same, per call |
| `push($state.list, $v)`, `$state.list[$i] = $v` | ~0.5 µs — the *write* paths are in-place |
| `ord($s[$i])` — a string index | ~1 µs, and O(1) in the string's length |

A dict-field list read and a user-function argument both **copy the whole list**. So the entropy
coder is one long function with every table it touches flattened into a *local* list with
per-component base offsets, the byte source arrives as one local copy per scan, and no helper is
called per byte, per coefficient or per block. The alternative measured 100× slower, which is the
difference between a JPEG decoder and a demonstration of one.

Written down because the natural, readable arrangement — a `$state` dict of tables, small helpers
taking it — is the slow one, and nothing about the code says so.

### PW15 — The encoder writes libjpeg's bytes where libjpeg's choices are published, and says where it does not

The decoder can be bit-exact because every stage of a decode is spec- or table-determined. An encoder
has freedom: which quantisation tables, which Huffman tables, which scan script. So `jpeg.b` copies
libjpeg's *published* choices — Annex K's tables scaled by `jpeg_quality_scaling`, the `islow`
forward DCT, `jcsample`'s downsamplers with their alternating rounding bias, `jccolor`'s RGB→YCbCr
tables, `jcmarker`'s segment order — and the result is **byte-identical to Pillow** for every
baseline case in the test corpus: all three subsamplings, quality 1/25/75/90/100, `optimize`,
restart markers by block and by row, `dpi`, greyscale, CMYK, YCbCr and the named presets. That is a
stronger result than the P5 gate asked for (a PSNR floor), so the test asserts the stronger one.

Two places deliberately differ, both visible only in file size, both stated in the module header:

1. **`progressive` writes a spectral-selection script** — one DC scan, then one AC scan per component
   — rather than libjpeg's successive-approximation script. The file is a valid progressive JPEG and
   decodes to identical pixels (asserted); it is a few per cent larger. Successive approximation on
   the *encode* side is the upgrade path, and the decoder already implements all four of its
   refinement cases because Pillow's own files need them.
2. **End-of-band runs are emitted one at a time.** The standard Annex K AC tables contain `EOB` but
   no `EOBn` symbol, so a run longer than one cannot be spelled with them at all; accumulating runs
   is only reachable with `optimize`, for a few per cent.

### PW16 — `draft()` re-decodes, because an `Image` here is never lazy

In Pillow, `Image.open` reads only the header, so `draft()` merely tells libjpeg which scale to use
when `load()` finally runs. `pillow`'s `Image` is eager by PW1's design — `load()` is a no-op — so
`draft()` has nothing left to configure.

Rather than make the object model lazy for one format, the JPEG decoder keeps the file bytes in
`info["jpeg_src"]` and `draft()` decodes them again at the smaller scale. The *result* is identical to
Pillow's, including the mode change to `L` or `YCbCr`; the *saving* is not, because the first decode
already happened. A caller who wants the saving asks for the scale at open time:
`pillow.decode_jpeg($data, 8, null)`. The cost is the file's own size held per open JPEG, which is
stated in the docs rather than hidden.
