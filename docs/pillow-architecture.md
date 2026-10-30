# `pillow` — architecture

`pillow` is Bantu's imaging package: **Pillow's API, in Bantu**. `Image.open`, `convert`, `resize`,
`rotate`, `crop`, `paste`, `filter`, `point`, `save`; `ImageDraw`, `ImageFont`, `ImageOps`,
`ImageChops`, `ImageEnhance`, `ImageFilter`, `ImageStat`, `ImageMath`, `ImageMorph`, `ImagePalette`,
`ImageSequence`, `ImageTransform`, `ImageColor`, `ExifTags` — the same names, the same arguments, the
same results.

Every line of the package is Bantu. The codecs are Bantu, the resampling kernels are Bantu, the
rasteriser behind `ImageDraw` is Bantu, the TrueType reader behind `ImageFont` is Bantu. What the
interpreter provides is **atoms** — the same arrangement `hash` has with `band`/`rotr`/`add32` and
`numba` has with `nd_*`. §4 is the exact list, and it is two functions long.

This document is the design of record. It is written **before** the code, per the standing rule, and
every number in it was measured on the machine named below rather than estimated.

> Measurements: **Darwin 24.6.0 / macOS, Apple silicon**, `bantu` at `68d0174`, `-O2`.
> The benchmark scripts are in `tests/pillow_bench.b`.

- **Source:** `pillow/` · **Docs:** [`docs/pillow.md`](pillow.md) · **Roadmap:** [`pillow-suite/ROADMAP.md`](../pillow-suite/ROADMAP.md)
- **Depends on:** [`numba`](numba-architecture.md) for the pixel store and every per-pixel operation.

---

## 1. The load-bearing question

Pillow exists to touch every pixel. A 1920×1080 RGBA image is **8,294,400 channel values**. So the
first question — the one that decided numba's native core and bplot's native raster backend — has to
be asked again here:

**Can the inner loop be interpreted?** Measured, with `while` (the C-style `for` silently stops at
100,000 iterations, so every loop in this package is a `while`):

| 1,000,000 iterations of | wall clock | per iteration |
|---|---|---|
| `$s = $s + $i` | 487 ms | 487 ns |
| `$x = $buf[$i]` (string index) | 507 ms | 507 ns |
| `$x = ord($buf[$i])` | 977 ms | 977 ns |
| `$x = $bytes[$i]` (list index) | 587 ms | 587 ns |
| `$x = band(shr($i,3),255)` | 1,392 ms | 1,392 ns |
| `push($L, $i)` | 976 ms | 976 ns |

**No.** At ~0.5 µs for the cheapest possible loop body, one pass over a 1080p RGBA image costs
**4.1 seconds** — and `convert("L")` is four passes, a 3×3 blur is nine, a bicubic resize is sixteen.
A per-pixel `pillow` would answer `img.convert("L")` in half a minute.

### 1.1 But the question has a second answer

Pillow's own operations are **not** per-pixel loops in disguise. They are *whole-image* operations:
`point()` is a lookup table applied to every value, `convert()` is a fixed matrix, `blend()` is a
weighted sum, `filter()` is a convolution, `crop()` is a window, `transpose()` is an axis
permutation. Every one of them is an **array** operation — and Bantu has native arrays.

Measured, on a **1080p RGB** image (6,220,800 channel values) through numba's `nd_*` kernels:

| Pillow operation | expressed as | wall clock |
|---|---|---|
| `img.crop(box)` | `nd_slice` — a view | **0 ms** |
| `ImageOps.mirror(img)` | `nd_flip` — a view | **0 ms** |
| `img.point(lut)` | `nd_take(lut, nd_ravel(px))` | **92 ms** |
| `ImageEnhance.Brightness` | `nd_add` + `nd_clip` | **145 ms** |
| `img.convert("L")` | 3 × `nd_multiply`, 2 × `nd_add`, 1 × `nd_floor_divide` | **108 ms** |
| `img.histogram()` | `nd_bincount` | **74 ms** |
| `img.filter(BLUR)` 3×3 | 9 × (`nd_slice` + `nd_add`), 1 × `nd_divide` | **153 ms** |

**That is the design.** A tenth of a second for a full-frame 1080p operation is within an order of
magnitude of C Pillow, and it is reached with *no native image code at all* — only pure-Bantu
orchestration over arrays. The interpreter runs a few dozen operations per image instead of a few
million.

The rule that follows, and that every phase is held to:

> **No loop in `pillow` may be proportional to the pixel count.** Loops over bands (≤4), over
> kernel taps (≤25), over palette entries (≤256), over scanlines of a *codec* (§6) — all fine. A
> loop over pixels is a defect, and the stress tier (§10) is built to catch one by its timing curve.

## 2. The pixel store

A `pillow` image's pixels are **one numba array of shape `(height, width, bands)`**.

Three consequences, all deliberate:

- **`(h, w, bands)` is always 3-D**, even for one-band `L` and `1`. Pillow keeps single-band data
  flat; carrying the trailing axis instead means every operator — broadcast a per-band constant,
  split a channel, composite — is written once rather than twice with a rank test.
- **Row-major, so a scanline is contiguous.** Codecs read and write whole rows, and `nd_slice` on
  axis 0 is a contiguous view.
- **Writes through views work** (verified: `nd_set` and an `out=` destination both reach the base
  array), so `paste` is a slice of the destination with the source written into it — not a copy and a
  stitch.

### 2.1 Modes

Pillow's mode string is the image's type, and `pillow` carries the whole table. `mode.b` is the one
place that knows it:

| mode | bands | store dtype | value range | notes |
|---|---|---|---|---|
| `1` | 1 | `i64` | {0, 255} | bilevel; stored as 0/255 as Pillow does, packed only by the codec |
| `L` | 1 | `i64` | 0–255 | greyscale |
| `LA` | 2 | `i64` | 0–255 | greyscale + alpha |
| `La` | 2 | `i64` | 0–255 | alpha-premultiplied |
| `P` | 1 | `i64` | 0–255 | palette index; the palette lives beside it (`ImagePalette`) |
| `PA` | 2 | `i64` | 0–255 | palette + alpha |
| `RGB` | 3 | `i64` | 0–255 | |
| `RGBA` | 4 | `i64` | 0–255 | |
| `RGBa` | 4 | `i64` | 0–255 | alpha-premultiplied |
| `RGBX` | 4 | `i64` | 0–255 | 4th band is padding, preserved but ignored |
| `CMYK` | 4 | `i64` | 0–255 | inverted ink, as Pillow stores it |
| `YCbCr` | 3 | `i64` | 0–255 | JPEG's colour space, full-range JFIF |
| `LAB` | 3 | `i64` | 0–255 | L\*a\*b\* with a/b offset by +128 |
| `HSV` | 3 | `i64` | 0–255 | |
| `I` | 1 | `i64` | int32 | |
| `I;16` `I;16L` `I;16B` `I;16N` | 1 | `i64` | 0–65535 | the suffix is a *codec* byte order, not a store difference |
| `F` | 1 | `f64` | any | |

**Integer modes are stored in `i64`, not in bytes**, because numba's dtypes are `f64`/`i64`/`bool`.
The cost is memory: 8 bytes per channel value where Pillow uses 1.

| image | Pillow | `pillow` |
|---|---|---|
| 1920×1080 RGB | 6.2 MB | 50 MB |
| 1920×1080 RGBA | 8.3 MB | 66 MB |
| 4000×3000 RGBA | 48 MB | 384 MB |

That is the honest limit of this design: **comfortable to ~16 megapixels, heavy above it.** It is
recorded here rather than discovered later, `Image.open` enforces a configurable
`MAX_IMAGE_PIXELS` (Pillow has the same guard, for the same decompression-bomb reason), and
[decision PW7](../pillow-suite/DECISIONS.md) records the fix — a `u8` dtype in numba — as a
later, separable change that this package would pick up with no API change.

### 2.2 The image object

```
PillowImage
  mode       "RGBA"                     the mode string
  size       [width, height]            Pillow's order, not the array's
  px         nd array (h, w, bands)     the pixels
  palette    PillowPalette | null        for P / PA
  info       dict                       dpi, transparency, duration, exif, …
  _frames    list | null                 for multi-frame GIF/TIFF (lazy)
```

`size` is `[w, h]` because that is what Pillow's users type; `px` is `(h, w, b)` because that is
what row-major arrays want. The two orders meet in exactly one place — the constructor — and
`width`/`height` accessors exist so callers rarely index `size` at all.

## 3. Module map

One file per Pillow module, named so that a Python programmer finds it by guessing:

```
pillow/
  pillow.b        the umbrella — include this and you have everything
  mode.b          the mode table (§2.1): bands, dtype, range, conversion targets
  image.b         Image.new/open/frombytes/fromarray/merge/blend/composite/eval + PillowImage
  convert.b       the mode-conversion matrix (§5)
  resample.b      nearest, box, bilinear, hamming, bicubic, lanczos (§5.2)
  color.b         ImageColor — getrgb/getcolor, 148 CSS names, #rgb, rgb(), hsl(), hsv()
  palette.b       ImagePalette + median-cut and octree quantisation
  chops.b         ImageChops — 24 channel operations
  enhance.b       ImageEnhance — Color, Contrast, Brightness, Sharpness
  filter.b        ImageFilter — the 10 named kernels, Gaussian/Box/Unsharp, rank filters, Color3DLUT
  ops.b           ImageOps — autocontrast, equalize, fit, pad, expand, colorize, …
  stat.b          ImageStat — extrema, count, sum, sum2, mean, median, rms, var, stddev
  math.b          ImageMath — eval over band expressions
  morph.b         ImageMorph — LutBuilder, MorphOp
  draw.b          ImageDraw — the vector rasteriser (§8)
  font.b          ImageFont — TrueType and BDF/PCF readers, glyph rasterisation (§8.2)
  path.b          ImagePath
  transform.b     ImageTransform — Affine, Extent, Quad, Mesh
  sequence.b      ImageSequence — Iterator, all_frames
  exif.b          ExifTags + Exif read/write
  bitio.b         bit reader / bit writer over a byte string (§6.1)
  inflate.b       DEFLATE decoder, in Bantu (§4.1)
  deflate.b       DEFLATE encoder, in Bantu
  codecs/
    registry.b    format sniffing and the open/save dispatch table
    png.b jpeg.b gif.b bmp.b ppm.b tga.b tiff.b ico.b webp.b pcx.b xbm.b
```

`pillow.b` re-exports, so a caller writes what they would write in Python:

```bantu
include "./pillow/pillow.b" as pillow;

$img = pillow.Image_open("photo.jpg");
$img = $img.convert("L").resize([320, 240], pillow.LANCZOS);
$img.save("thumb.png");
```

## 4. What the interpreter provides

Two functions. Both are **generic byte atoms**, both are halves of primitives the interpreter already
ships, and the absence of each is a defect that costs every Bantu program, not only this one. They
are completed in the interpreter for the same reason `band` lives there and not in `hash.b`.

### 4.1 `inflate` — the decompressor the interpreter never had

The interpreter ships **`bp_zlib(s)`**, a zlib *compressor*, used by bplot's PNG writer. There is no
decompressor anywhere in `bantu-src` — grep finds `inflate` only in a GC comment. So Bantu can
*write* a PNG, a gzip body or a ZIP member and cannot *read* one back. That is a one-way primitive,
and PNG — the format `pillow` must support before any other — is defined as a zlib stream.

Pure Bantu can decode DEFLATE correctly, and `pillow/inflate.b` does — stored, fixed and dynamic
blocks, the full length/distance alphabets, back-references into the 32 KB window, the zlib and gzip
containers, Adler-32 verified. It is validated two ways: **35 vectors produced by Python's `zlib` and
`gzip` at levels 0, 1, 6 and 9 decode byte-identically**, and every stream `bp_zlib` produces
round-trips through it.

Its cost is **measured, not estimated** (`tests/pillow_inflate_test.b` prints it):

| path | measured |
|---|---|
| dynamic Huffman — what PNG uses | **3.4 µs per output byte** |
| fixed Huffman | 3.6 µs per output byte |
| stored blocks | 5.5 µs per output byte |

Which puts a PNG decode at:

| PNG | raw bytes | pure-Bantu inflate |
|---|---|---|
| 64×64 RGBA icon | 16 KB | **0.06 s** |
| 256×256 RGBA | 262 KB | **0.9 s** |
| 1024×768 RGB | 2.4 MB | **8 s** |
| 1920×1080 RGBA | 8.3 MB | **28 s** |

Usable for icons and thumbnails, too slow for photographs. And there is a second cost that the table
does not show: the decoder's output is a Bantu list, at ~100 bytes of `Value` overhead per byte, so a
1080p RGBA PNG's 8.3 MB of pixels passes through ~830 MB of list.

So: `inflate(s)` / `gunzip(s)` become interpreter atoms beside `bp_zlib`. **`pillow/inflate.b` ships
anyway**, for two reasons that are worth the duplication:

1. **It differential-tests the atom.** Every `pillow` test run decodes the same corpus both ways and
   asserts the bytes are identical. A decompressor is a parser of hostile input; two independent
   implementations agreeing is the strongest cheap evidence either is right.
2. **It is the fallback.** `pillow` sets `_HAS_NATIVE_INFLATE = has_native("inflate")` and routes
   accordingly, so the package is complete in Bantu alone on any build — slowly, but completely.

### 4.2 `nd_frombytes` / `nd_tobytes` — the bridge that is missing

Pixels arrive from a file as bytes and leave as bytes. The interpreter has `nd_fromfile(path, dtype,
shape, offset)` — but `dtype` is only `f64`/`i64`/`bool`, all 8 bytes wide, so **there is no way to
get a byte buffer into an array**. The only available route is `readfile → bytes() → nd(list)`:

| step, on 1 MB | cost |
|---|---|
| `bytes($s)` → a 1,048,576-element list | 262 ms, and ~100 bytes of `Value` per byte |
| `nd($list, "i64")` | 217 ms |
| **total** | **479 ms per megabyte**, with a ~100 MB transient |

An 8 MB PNG would spend four seconds and the better part of a gigabyte turning its own pixels into an
array. `nd_frombytes(s, dtype?, shape?)` and `nd_tobytes(a, dtype?)` widen bytes to `i64` (and narrow
back, with an explicit `"u8" | "u16be" | "u16le" | "i32" | "f32"` element type) in one pass with no
intermediate list. That is the natural completion of `nd_fromfile`, it is what `tofile`/`frombuffer`
are in NumPy, and nothing about it is image-specific.

### 4.3 Atoms already present, which `pillow` uses as-is

No change needed to any of these: `nd_*` (the whole array surface — slice, flip, transpose, take,
put, where, clip, bincount, matmul, sort, the ufuncs, the reductions); `band bor bxor bnot shl shr
rotl rotr add32 mul32`; `bytes frombytes tohex fromhex chr ord`; `readfile writefile open/read/write/
close filesize` (verified binary-safe — a NUL, a `\r\n` and a `\xff` all survive a round trip);
`bp_crc32` and `bp_adler32` (PNG's two checksums, already exposed as generic atoms by bplot);
`join split substr`; `b64encode b64decode`.

## 5. Conversion and resampling — the two operations that must match Pillow exactly

These are where a reimplementation usually drifts, so both are specified to the arithmetic.

### 5.1 `convert()`

Pillow's L is **not** a float luminance, and it is **not** the `(R×299 + G×587 + B×114) / 1000`
that every reference for BT.601 prints. It is `ImagingConvert`'s `L24` macro — fixed point, with a
rounding offset:

```
L = (R × 19595 + G × 38470 + B × 7471 + 32768) >> 16
```

The weights are 0.299, 0.587 and 0.114 scaled by 2¹⁶ (they sum to exactly 65536), and `+32768`
rounds rather than truncates. The difference is not academic: the truncated `/1000` form disagrees
with Pillow on **1,101 of 2,197** sampled triples — it answers 14 where Pillow answers 15 for
`(0, 0, 128)`. This design's first draft had the `/1000` form, and the differential tier caught it
on the day the reference was installed.

`pillow` computes the `L24` expression with `nd_multiply`/`nd_add`/`nd_floor_divide` — which is why
the measured `convert("L")` above is six array calls — and the formula above is **verified against
Pillow over all 16,777,216 RGB triples, not a sample**: zero mismatches. The P2 differential tier
re-runs that whole-cube check.

The rest of the matrix, each as a fixed sequence of array calls:

| from → to | arithmetic |
|---|---|
| `RGB → L` | the `L24` fixed-point expression above |
| `L → RGB` | the band broadcast three ways |
| `RGB → 1` | `L`, then Floyd–Steinberg — Pillow's default **is** dithered; `dither=NONE` thresholds instead. The two thresholds differ: `NONE` maps `v ≥ 128` to white, the dithered path `v > 128`. Both verified |
| `RGB → P` | median cut over the 3-D histogram, then nearest-palette assignment |
| `P → RGB` | `nd_take` of the palette — a gather, so free |
| `RGB ↔ YCbCr` | the full-range JFIF matrix, integer, with the +128 chroma offset |
| `RGB ↔ HSV` | Pillow's own 8-bit integer formulation, including its hue wrap |
| `RGB ↔ CMYK` | Pillow's naive conversion (`K = 255 − max`), not a colour-managed one — matching Pillow's behaviour matters more than being right about ink |
| `RGB ↔ LAB` | D65, the sRGB transfer function, a/b offset +128 |
| `RGBA → RGB` | drop alpha, as Pillow does — compositing is `alpha_composite`'s job |
| `RGBA ↔ RGBa` | premultiply / unpremultiply with Pillow's rounding |
| `* → F`, `* → I` | widen |

**Dithering** is Floyd–Steinberg, it is **Pillow's default** for `1` and `P` (not opt-in, which an
earlier draft of this document had wrong), and it is the one genuinely sequential operation in this
layer — each pixel's error depends on its left neighbour's. It is therefore **per-scanline**, not
per-pixel: the row is a Bantu list, the error diffusion runs along it in a loop of width `w`, and the
next row is carried in a second list. A 1080p dither is 1,080 loops of 1,920 — 2M iterations, ~2 s,
which is documented as the one slow conversion rather than silently skipped.

Three details of it are pinned by experiment, because all three are easy to get backwards and none
is documented anywhere:

- the dithered threshold is **`v > 128`**, while `dither=NONE` thresholds at **`v ≥ 128`**. Getting
  that one wrong inverts the checkerboard a flat 50% grey dithers to;
- the diffused error is carried as an **integer truncated toward zero**, not as a float. Keeping it
  in floating point makes a whole row drift;
- the weights are the classic 7/16, 3/16, 5/16, 1/16.

With those three, the implementation reproduces Pillow **exactly** for single-row and single-column
images and for an 8×8, and differs on roughly 8–12% of pixels for other 2-D sizes: how Pillow
sequences its error buffer between rows in the general case is still unknown, four rounding variants
were tried and none closed it, and it needs Pillow's C source rather than more guessing. That is
recorded in the roadmap rather than rounded off.

### 5.1.1 What is bit-exact today, and what is not

The whole point of the differential tier is that this table can be stated as fact. Measured against
Pillow 13.0.0 over the 191-pair reference matrix:

| | pairs | status |
|---|---|---|
| **exact** — every band of every probe pixel matches Pillow | **177** | verified |
| `RGBA → P`, `RGBA → PA` | 4 | need `quantize()` — see below |
| the `LAB` family | 10 | littleCMS — see below |

Beyond the matrix, and verified whole-image rather than on probe pixels:

| | result |
|---|---|
| `RGB → L` over **all 16,777,216** RGB triples | 0 mismatches |
| `RGB → YCbCr` and `RGB → HSV`, and both inverses, over 3,014 pixels | 0 mismatches |
| Floyd–Steinberg to mode `1`, 24 images from 1×1 to 64×40, `L` and `RGB` sources | **9,154 of 9,154 pixels** |
| `RGB → P` undithered, 5 images | **815 of 815 pixels** |
| `RGB → P` dithered, 5 images | 793 of 815 — the 22 are decision PW11's deliberate fix, confined to ≤4 columns from the right edge |

**The three that were hard, and why reading the source settled them.** An earlier draft fitted
YCbCr, HSV and the dither against Pillow as a black box and got each to within ±1 on 8–25% of
inputs. No amount of further fitting would have closed any of them, because in each case the
structure — not the constants — was different:

- **YCbCr** does not evaluate the JFIF matrix. `ConvertYCbCr.c` pre-multiplies each coefficient by
  every possible channel value, **rounds each product** at 6 fractional bits, and sums the three
  *rounded* values. The discarded rounding residues are the ±1, and they cannot be fitted away. The
  package carries Pillow's 13 tables (`pillow/ycbcr_tables.b`, generated by
  `tools/gen_ycbcr_tables.py`) and applies them as three `nd_take` gathers per band — exact, and
  faster than arithmetic would have been.
- **HSV** computes in C `float`, not `double`. Four intermediates are 24-bit, and the hue is folded
  as `fmod(h/6.0 + 1.0, 1.0)` — adding one *before* the fold, which costs low-order bits and is not
  the same as folding then correcting the sign. `pxbuf.to_f32` is applied at exactly the points C's
  assignments round.
- **The dither** carries its error **undivided** and divides by 16 once, at the point of use.
  Dividing each of the four weights separately — the obvious reading of "7/16, 3/16, 5/16, 1/16" —
  discards four rounding residues per pixel instead of one, and the error compounds along the row.
  `tobilevel` also uses the **truncated** `(R×299 + G×587 + B×114)/1000` luminance, where
  `convert("L")` uses the rounding `L24`: two different greys in Pillow, by design, and both matched.

**`RGBA → P` is a different feature, not a gap in the matrix.** Pillow routes an RGBA source to
`quantize()` — an *adaptive* median-cut palette — rather than to the fixed colour cube. The package
raises an error naming `quantize()`, because returning cube indices there would look almost right and
be wrong. `quantize()` is a phase of its own.

**`LAB` cannot be matched here, and the reason is structural.** There is no `{"RGB","LAB"}` row in
`Convert.c`'s converter table to copy: `Image.convert` routes LAB through **`ImageCms`, i.e.
littleCMS**, a full ICC colour-management engine. Reproducing littleCMS bit for bit is exactly the
library-sized problem [decision PW10](../pillow-suite/DECISIONS.md) declines on `ImageCms`'s behalf.
So `pillow` implements a correct, standard sRGB ↔ CIE L\*a\*b\* conversion (D65, the piecewise sRGB
transfer function, a/b offset by +128) — good colour science, and not littleCMS to the last bit.
`convert.unverified(from, to)` answers truthfully and `convert.why_unverified(from, to)` says why.

### 5.1.2 Where this deliberately differs from Pillow

Matching Pillow's arithmetic ([PW4](../pillow-suite/DECISIONS.md)) does not extend to reproducing its
defects ([PW11](../pillow-suite/DECISIONS.md)). Two fixes exist, both stated in their module headers
and both with a test asserting the *containment* of the difference:

| upstream | fix |
|---|---|
| `Convert.c`'s dithered `topalette` ends each row writing blue's three error accumulators into all three channel slots, discarding red's and green's | each channel carries its own. Row 0 is identical to Pillow; the difference is confined to ≤4 columns from the right edge |
| `ImagingConvertRGB2YCbCr` casts with `(UINT8)`, so an out-of-range input *wraps* — 256 becomes 0 | clamp. Identical on all valid input, saturating rather than wrapping on invalid input |

### 5.2 `resize()` — the six filters, and why separable### 5.2 `resize()` — the six filters, and why separable

| filter | support | Pillow's constant |
|---|---|---|
| `NEAREST` | 0 | `Image.NEAREST` = 0 |
| `LANCZOS` | 3 | 1 |
| `BILINEAR` | 1 | 2 |
| `BICUBIC` | 2 | 3 |
| `BOX` | 0.5 | 4 |
| `HAMMING` | 1 | 5 |

Pillow resizes with **two one-dimensional passes** (horizontal, then vertical), and so does
`pillow` — the same result and the same reason: a `k×k` kernel costs `2k` taps instead of `k²`.

The mechanics, per pass, with **no loop over pixels**:

1. For each of the `W_out` output columns, the coefficient support `[x0, x1)` and its normalised
   integer weights. That is a Bantu loop of `W_out` iterations — a few hundred to a few thousand, not
   a few million — and it is exactly Pillow's `precompute_coeffs`, including its `filterscale`
   clamp and its 1/2048 fixed-point weight normalisation.
2. The pass itself is `sum over taps of (input column slice × weight)`: one `nd_slice` and one
   fused multiply-add per tap. **A bicubic horizontal pass on a 1080p image is 4 array operations**,
   not 2 million.
3. `NEAREST` is a strided `nd_slice` — a *view*, costing nothing (verified: `nd_slice` with a step
   returns a view).
4. `BOX` with an integer ratio reduces by reshape-and-mean, which is how `reduce()` is also built.

`thumbnail()`, `ImageOps.fit`, `ImageOps.contain`, `ImageOps.cover` and `ImageOps.scale` are all
`resize` plus box arithmetic, as in Pillow — including `thumbnail`'s `reducing_gap` pre-reduction.

#### 5.2.1 The pass order is part of the answer

Separating the kernel means two passes, and **each pass rounds its result to the mode's precision**.
So the order they run in changes the output, not merely the cost. Pillow chooses it by cost:

```
horizontal_first = !((inH - outH) > 0 && (inH - outH) > (inW - outW) * 2)
```

— vertical first when the height shrinks more than twice as much as the width, so the horizontal
pass has fewer rows to walk. For 7×5 → 13×3 that is false (the height loses 2 rows while the width
*gains* 6 columns), and running horizontal first there disagrees with Pillow on about a quarter of
the pixels.

This was the hardest finding in the phase precisely because **every individual pass was already
bit-exact**: a pure horizontal resize matched, a pure vertical resize matched, and only the
combination did not. The lesson, recorded because it generalises: when each part is right and the
whole is wrong, the composition is the thing to look at.

#### 5.2.2 `NEAREST` accumulates, and that is observable

`ImagingScaleAffine` does not compute the source coordinate from the output index. It accumulates:

```c
xo = a[2] + a[0] * 0.5;
for each x:  xin = floor(xo);  xo += a[0];
```

Repeated addition drifts in the last bits, and the drift crosses integer boundaries: where the closed
form lands on 5.999999999999999 the accumulation lands on 6.0000000000001 — a whole source row
different. Resampling 12 rows to 11 hits it. The accumulation is therefore reproduced rather than
simplified, in both `resample.b` and `transform.b`.

## 6. Codecs

A codec's inner loop is **inherently sequential** — a bitstream has to be read in order. This is the
one part of `pillow` where the no-pixel-loop rule cannot hold as written, so it is weakened precisely
and honestly: **a codec may loop over bytes of the compressed stream; it may not loop over pixels of
the decoded image.** Everything after entropy decoding — unfiltering, dequantising, upsampling,
colour transform — goes back to array operations.

### 6.1 `bitio.b`

One reader, one writer, both over a byte string, both parameterised by bit order (`lsb` for DEFLATE,
`msb` for JPEG) — because writing that twice is how the two disagree. The reader carries a 32-bit
accumulator in a Bantu number, refilled a byte at a time, and exposes `bits(n)`, `peek(n)`,
`align()`, `skip(n)`, `pos()`. Every codec that reads bits uses it; none reimplements it.

### 6.1.1 Two interpreter facts that decide every codec's shape

Measured on this build, and they matter more to the entropy coders than the formats do:

| | cost |
|---|---|
| `$local[$i]` — a list read | ~0.5 µs |
| `$state.table[$i]` — a list read **through a dict field** | ~25 µs at 1,000 entries, ~500 µs at 20,000 |
| `f($bigList, …)` — a user function taking a list | the same, per call |
| `push($state.list, $v)` and `$state.list[$i] = $v` | ~0.5 µs — the *write* paths are in-place |
| `ord($s[$i])` — a string index | ~1 µs, O(1) in the string's length |

**A Bantu list is copied when it crosses a user function's boundary or comes out of a dict field.**
The write paths are not, which is what makes the asymmetry easy to miss. So every entropy coder in
this package is one long function whose tables are *flat locals* with per-component base offsets, the
byte source arrives as one local copy per scan, and no helper is called per byte, per coefficient or
per block. The readable arrangement — a `$state` dict of tables and small helpers over it — measured
100× slower. Decision PW14, written down because nothing about the code says so.

### 6.2 Format by format

Status after P5: **PNG, BMP, TGA, PPM and JPEG are done and differentially verified**, as is EXIF.
The JPEG corpus — 54 files covering every subsampling Pillow writes and five it cannot, baseline and
progressive, restart markers, greyscale, CMYK, YCCK, quality 1 to 100, `optimize` and draft at 1/2,
1/4 and 1/8 — decodes **byte-identically to libjpeg-turbo through Pillow**, and the encoder is
byte-identical to libjpeg on every baseline case (PW15). PngSuite — the
canonical 176-file PNG conformance corpus, checked in at `tests/fixtures/pillow/pngsuite/` — decodes
**162 of 162** readable files pixel-identically to Pillow, and all **14** intentionally-corrupt files
are refused. A further **238 images written by Pillow itself** at every option combination read back
byte-identically (PNG 147, BMP 28, TGA 63), and `tests/pillow_differential.sh` has Pillow read eleven
files these codecs wrote.

| format | read | write | how |
|---|---|---|---|
| **PNG** | ✅ all 5 colour types, 1/2/4/8/16 bits, interlaced (Adam7), `tRNS` `gAMA` `pHYs` `sRGB` `bKGD` `tEXt` `zTXt` `iTXt`, **every chunk CRC verified** | ✅ | §4.1's inflate, then the 5 filters un-applied per scanline (`None Sub Up Average Paeth` — each is one array add after the previous row is known, so the loop is over *rows*); APNG frames read |
| **JPEG** | ✅ baseline + progressive, all subsamplings, restart markers, EXIF, JFIF, Adobe APP14 | ✅ baseline, quality 1–100, `optimize`, `progressive` | Huffman scan is sequential (§6.3); dequantise, IDCT, upsample and YCbCr→RGB are array operations |
| **GIF** | ✅ including all frames, local palettes, transparency, interlace, disposal | ✅ including animation | LZW, sequential, over the compressed stream |
| **BMP** | ✅ 1/4/8/16/24/32-bit, RLE4/RLE8, `BITFIELDS`, top-down and bottom-up | ✅ 1/8/24-bit | raw rows → `nd_frombytes` → one reshape |
| **PPM/PGM/PBM** | ✅ P1–P6, 8 and 16 bit | ✅ | the trivial case, and the first codec built |
| **TGA** | ✅ 8/16/24/32-bit, RLE, all origins | ✅ | |
| **TIFF** | ✅ both byte orders, strips and tiles, `NONE` `PACKBITS` `LZW` `DEFLATE` `CCITT G3/G4`, all photometrics, multi-page | ✅ `NONE` `PACKBITS` `DEFLATE` `LZW` | the tag reader is also `exif.b`'s |
| **ICO/CUR** | ✅ all sizes, PNG- and BMP-payload | ✅ multi-size | |
| **PCX/DCX** | ✅ | ✅ | |
| **XBM/XPM** | ✅ | ✅ XBM | |
| **WebP** | ✅ lossless (VP8L) and still lossy (VP8) | ✅ lossless | the largest single codec; phased last (§11, P8) |
| **PDF** | — | ✅ | an image wrapped in a one-page document, as Pillow's `PdfImagePlugin` does |
| **EPS, PSD, DDS, SGI, SUN, IM, MSP, FITS, BLP, PALM** | ✅ read | — | the long tail; read-only, as in Pillow |

**Plugin registration matches Pillow's.** `codecs/registry.b` holds `register_open`,
`register_save`, `register_mime` and `register_extension`; `Image.open` sniffs by magic bytes, falls
back to the extension, and raises `UnidentifiedImageError` — the same name Pillow raises.

### 6.2.1 Two things P4 had to fix before a codec could exist

**Bantu could not spell a file signature.** A string literal had no `\xNN` escape, and the lexer
*silently dropped the backslash* on any unrecognised escape — so `"\x89PNG"` became `x89PNG` and
`"C:\path"` became `C:path`. A byte-oriented language with binary-safe strings, native `bytes()` and
a working inflate still could not write the eight bytes that begin a PNG without `chr(137) + "PNG"`.
`\xNN` now works and an unknown escape keeps its backslash, so an unrecognised escape is visible
rather than lossy.

**Unfiltering is the slow part of PNG, and only partly avoidable.** Of the five scanline filters,
three are whole-row array operations — `None` is nothing, `Up` is one add, and `Sub` is a strided
prefix sum — but `Average` and `Paeth` need the byte to the *left*, which needs the one to *its*
left. They are sequential by construction, the documented exception to rule PW2 alongside the
dither. Two things keep the cost down: a whole-image fast path when every row is filter 0 (which is
what this codec writes by default, so its own output re-reads at array speed), and the native
`inflate` doing the decompression. Measured: a 60,000-pixel RGB PNG writes in 13 ms and reads in
8 ms.

### 6.3 The one place the envelope is genuinely tight

JPEG's entropy-coded scan is a bit-by-bit Huffman walk over every non-zero coefficient. **Measured
in P5** on photographic 4:2:0 files, with the decoder shaped as §6.4 and PW14 describe:

| JPEG | decode | `draft` 1/8 | encode | per pixel |
|---|---|---|---|---|
| 320×240 | 0.80 s | 0.73 s | 0.72 s | 10.4 µs |
| 640×480 | 3.06 s | 2.89 s | 3.12 s | 10.0 µs |
| 1024×768 | 7.77 s | 7.21 s | 7.59 s | 9.9 µs |
| 4000×3000 (extrapolated, linear) | ~2 min | ~1.9 min | ~2 min | — |

The estimate this table replaces said 1.2 s for 640×480, from "~8 interpreted operations per
coefficient". The real figure is ~2.5× that: a coefficient costs a Huffman lookahead, a canonical
walk when the code is over 8 bits, a receive-and-extend, two pushes and the loop itself, and every
one of those is a statement at 0.2–0.6 µs (PW14's table). The curve is linear in pixels, which is
what matters for the rule in §1; the constant is what it is.

Everything *after* the scan is array work and costs a fraction of this. The scan is written in Bantu,
it is correct, and its cost is published in [`docs/pillow.md`](pillow.md) next to the API rather than
left for a user to discover. `draft()` — Pillow's own answer, DCT-domain downscaling that decodes a
1/2, 1/4 or 1/8 image — is implemented, and it shrinks everything after the scan. It does **not**
shrink the scan itself, which is what this section originally claimed; see §6.4.6 for why that was
wrong and what draft mode actually costs.

If measurement after P4 shows the full-size scan is the thing users actually hit, the answer is one
more generic atom — a canonical-prefix-code scanner, which is what `band`/`rotr` are to `hash` — and
[decision PW9](../pillow-suite/DECISIONS.md) records the trigger for it. It is not added
speculatively.

### 6.4 JPEG — the one codec whose reference is not Pillow

Every other format in §6.2 has Pillow's own C as its reference. JPEG does not: Pillow hands the file
to **libjpeg-turbo** and asks for scanlines back. So the thing `jpeg.b` has to match is
`libjpeg-turbo`'s arithmetic, and `JpegDecode.c` matters only for *which* libjpeg knobs Pillow turns.
There are four, and reading them first saved implementing three code paths that can never run:

| knob | what Pillow sets | consequence |
|---|---|---|
| `out_color_space` | from the rawmode: `JCS_GRAYSCALE`, `JCS_EXT_RGBX`, `JCS_CMYK`, `JCS_YCbCr`, `JCS_YCCK` | **libjpeg does the colour transform**, not Pillow. So `jpeg.b` must reproduce libjpeg's YCbCr→RGB, which is *not* the YCbCr of `convert()` (§5.1) — different constants, different rounding, a different table layout |
| `scale_num` / `scale_denom` | `1` / `draft`'s 1, 2, 4 or 8 | selects `_min_DCT_scaled_size` 8, 4, 2 or 1 — a *scaled IDCT*, not a resize |
| `draft` | **always 0** | `decoderconfig` is `(scale, 0)`, so the flag that would turn off fancy upsampling and switch to the fast IDCT is never set. `do_fancy_upsampling` stays `TRUE` and `dct_method` stays `JDCT_ISLOW` even in draft mode |
| `jpeg_color_space` | only when the rawmode says so | otherwise libjpeg guesses from JFIF / Adobe APP14 / component IDs, and `jpeg.b` has to guess identically |

That third row is the one worth stating plainly, because the obvious reading of `draft()` is wrong:
**draft mode in Pillow does not mean "fast and approximate".** It means a smaller output from a
*scaled* IDCT, computed with the same slow-and-exact kernel. The consequence is that `draft()` is
bit-exact too, and that `use_merged_upsample()` — which requires `do_fancy_upsampling == FALSE` —
can never fire. One upsampling family, one IDCT family, both exact.

#### 6.4.1 What must be bit-exact, and why each one can be

`libjpeg`'s decode path is five stages. Four are fixed-point integer arithmetic with no freedom in
them at all, which is why aiming at bit-exactness here is reasonable rather than optimistic:

1. **Entropy decode** — spec-determined. The coefficient values are whatever ITU T.81 says they are;
   there is nothing to match, only to get right.
2. **Dequantise** — one multiply per coefficient.
3. **IDCT** — `jpeg_idct_islow`: the Loeffler–Ligtenberg–Moschytz 12-multiply kernel in 13-bit fixed
   point, two passes, with `DESCALE` (round-half-up by an arithmetic shift) at three named points.
   Deterministic to the last bit. libjpeg-turbo's SIMD variants of *islow* are bit-exact with the C
   kernel by design, which is why this remains a single target on every machine.
4. **Upsample** — `h2v1_fancy_upsample` / `h1v2_fancy_upsample` / `h2v2_fancy_upsample`: integer
   triangle filters with an alternating ±1 bias, plus `int_upsample`'s box filter for odd ratios.
5. **Colour transform** — `ycc_rgb_convert`, three 256-entry integer tables at 16 fractional bits.

#### 6.4.2 The IDCT without a pixel loop

`jpeg_idct_islow` is written as a loop over 8 columns then 8 rows of one block. Transcribed
literally that is a loop over coefficients, which rule PW2 forbids and which would cost minutes.

The transcription that works keeps the *arithmetic* identical and changes only what each operation is
applied to. Hold the whole component's coefficients as one `(N, 8, 8)` array — `N` blocks, 8 rows, 8
columns, in natural (de-zigzagged) order, exactly libjpeg's `JBLOCK` layout. Then:

* **pass 1** reads row `r` of every block as `nd_slice(a, [null, [r, r+1], null])` — an `(N, 1, 8)`
  view, free — and so computes all 8 columns of all `N` blocks at once. The kernel's 12 multiplies
  and 32 adds become 12 `nd_multiply`s and 32 `nd_add`s.
* **pass 2** does the same with column slices.

`DESCALE(x, n)` is `nd_floor_divide(nd_add(x, 1 << (n-1)), 1 << n)` — an arithmetic right shift *is*
a floor division, and `nd_floor_divide` floors, which is what the C does for negative values too.

The whole IDCT of a component is **~70 array operations, whatever the image size**. One array
operation per arithmetic step of the kernel, not one per pixel.

Two consequences fall out of the same arrangement:

* The two "all AC terms are zero" short-circuits in the C are *not* reproduced, and do not need to
  be: they are exact algebraic identities, verified by hand (a DC-only column gives
  `DESCALE(dc << 13, 11) == dc << 2`, which is the shortcut's `dcval`), and a vectorised kernel
  cannot branch per column anyway.
* The C accumulates pass 1 into `int workspace[64]` and would **overflow and wrap** on a
  deliberately-crafted coefficient block. Our store is `i64`, so it saturates nothing and wraps
  nothing. On every valid input the two agree; on a hostile one, ours stays arithmetic. This is
  decision PW11's second case again, recorded in `jpeg.b`'s header.

The scaled IDCTs for `draft()` — `jpeg_idct_4x4`, `jpeg_idct_2x2`, `jpeg_idct_1x1` — are separate
kernels in the C with their own constants, and are transcribed the same way.

#### 6.4.3 The range-limit table is not a clamp

`libjpeg` ends the IDCT with `range_limit[x & 1023]`, where `range_limit` is the middle of a
five-segment 1,280-entry table. It is tempting to read that as `clamp(x + 128, 0, 255)` and it very
nearly is — but only for `x` in `[-512, 511]`. Outside that window the mask **wraps**, and the table
is built so that the wrap lands on deliberate values rather than on whatever was next in memory.

So `jpeg.b` builds the same 1,024-entry table, generated from the same construction by
`tools/gen_jpeg_tables.py`, and applies it as `nd_take(table, nd_mod(v, 1024))` — `nd_mod` with a
positive divisor is exactly a bitwise AND with 1,023, including for negative `v`. One gather for the
whole component. The colour converter's `range_limit` is a *different* slice of the same table and is
a genuine clamp over its reachable domain; both are generated, neither is fitted.

#### 6.4.4 Chroma is upsampled by the IDCT, not by the upsampler

This is the detail that decides the whole draft-mode shape, and it is six lines in
`jpeg_calc_output_dimensions`: libjpeg scales a subsampled component up by choosing a **larger IDCT**
for it, so the upsampler gets to run 1:1 where it can.

For a 4:2:0 image at `scale_denom = 2`, luma gets the 4×4 IDCT and chroma gets the **full 8×8** one —
and then nothing is upsampled at all, because both components land at the same size. At
`scale_denom = 1` both get 8×8 and chroma is upsampled 2×. At `scale_denom = 8` the minimum DCT size
is 1, and `do_fancy_upsampling && _min_DCT_scaled_size > 1` is false, so the *box* upsamplers run
instead of the fancy ones. Three different paths through the same file, all reachable from
`draft()`, all of which have to be selected the way libjpeg selects them.

The edge behaviour is the other half of it. The fancy upsamplers read one sample of context on each
side, and `jdmainct.c` manufactures that context by **duplicating the first or last real sample
row**; `h2v1_fancy_upsample`'s first and last column cases are the same duplication done inline. So
every fancy upsample in `jpeg.b` is written against an edge-clamped neighbour array, which is one
`nd_slice` of the plane plus one of its own edge row — and the `(cs * 4 + 8) >> 4` first-column case
is then just the general case with `lastcolsum == thiscolsum`, which is how it is tested.

#### 6.4.5 Where the coefficients live, and the one place memory bites

A baseline scan is decoded once, in order, and never read back — so the entropy decoder emits two
flat lists, the indices and the values of the **non-zero** coefficients, and one `nd_put` scatters
them into a zeroed array. Memory is proportional to the non-zero count, which for a photograph is
~5–15% of the coefficients.

A **progressive** scan cannot do that. Successive-approximation refinement reads each coefficient to
decide whether to consume a correction bit, so the store has to be randomly readable *during*
decoding, and `nd_get` at ~2.5 µs would cost more than the scan. So progressive decoding holds the
coefficients in a flat Bantu list, which costs a `Value` per coefficient — roughly 100 bytes.

| progressive image | coefficients | coefficient store |
|---|---|---|
| 640×480 RGB 4:2:0 | 0.46 M | ~46 MB |
| 1600×1200 RGB 4:2:0 | 2.9 M | ~290 MB |
| 4000×3000 RGB 4:2:0 | 18 M | ~1.8 GB — refused |

`jpeg.b` therefore enforces a coefficient budget before the first scan is read, the same way
`MAX_IMAGE_PIXELS` is enforced from the header (PW6), and the error names the limit and the
alternative rather than letting the machine swap. Baseline files of any size are unaffected. This is
decision PW13, and the `u8`/`i16` dtype of PW7 does not fix it — a Bantu *list* is the cost, not the
array.

#### 6.4.6 What `draft()` actually buys, which is not what §6.3 claimed

§6.3 said `draft()` "makes a thumbnail of a 12 MP JPEG a 0.8 s operation by never decoding the full
scan". **That is wrong, and reading `jdhuff.c` is what showed it.** Scaling happens in the IDCT;
the entropy-coded scan is parsed in full at every scale, because the amplitude bits of every
coefficient are interleaved into the bitstream and there is no way to know where the next block
starts without decoding the current one. libjpeg does exactly the same amount of Huffman work for a
1/8-scale decode as for a full one.

So what `draft()` saves is stages 3, 4 and 5 — the IDCT drops from 64 to 1 output per block, the
upsampler and the colour transform shrink with the output, and the output buffer shrinks 64×. In
Bantu those are the *cheap* stages, so the saving is real but bounded. **Measured in P5: 6–9 %**
(§6.3's table). The published cost of reading a JPEG is the scan, at every scale.

§6.3's claim is corrected rather than quietly dropped, because the reason it was wrong is
instructive: it assumed a format property — that a resolution pyramid is skippable — that JPEG does
not have. What `draft()` *is* still worth having for is correctness: it is the only way to get
libjpeg's own scaled pixels, which are a scaled IDCT and not a resize of the full-size result, and
all nine draft fixtures are byte-identical because of it.

**And it needed three kernels, not one.** `jidctred.c`'s 4×4, 2×2 and 1×1 IDCTs are separate
derivations with their own `FIX` constants, their own left shift on the DC term (`CONST_BITS+1` and
`CONST_BITS+2`, not `CONST_BITS`) and their own descale amounts (12 and 19; 13 and 20). The draft
kernels in this package were first written with the 8×8 kernel's constants — the plausible mistake —
and seven of the nine draft fixtures differed from Pillow. The difference is not subtle in kind but
it is subtle in appearance: the image is recognisable and wrong, which is exactly the failure a
differential corpus exists to catch.

### 6.5 EXIF

EXIF is a TIFF file stored inside a JPEG APP1 segment, so `exif.b` is the TIFF tag reader with a
different entry point — which is why it is written now, in P5, and why `codecs/tiff.b` in P8 will
*include* it rather than reimplementing it. Both byte orders, all 12 TIFF field types, the four
sub-IFDs Pillow exposes (`Exif`, `GPSInfo`, `Interop`, `MakerNote` as raw bytes), and
`ExifTags.TAGS` / `GPSTAGS` / `IFD` / `Base` / `GPS` / `Interop` generated from Pillow's own
dictionaries rather than retyped.

Two properties are load-bearing and tested as such:

* **An IFD offset chain can point at itself.** A `next IFD` pointer that loops, or a sub-IFD that
  points back at its parent, is a hang in a naive reader. Every offset is checked against the
  segment length and against a visited set before it is followed.
* **A tag's count times its type size can overflow the segment.** Checked before the read, not
  after — a `LONG` tag claiming 2^31 entries is a header, not an allocation.

Writing is the same structure in reverse, and round-trips through Pillow: `exif.b` writes the
segment, Pillow's `Image.getexif()` reads back every tag it was given.

## 7. The operation surface

**Status after P6: done, and four defects deeper than this section expected.** 294 whole images are
byte-identical to Pillow, all 24 channel operations are exact over every one of the 65,536 input
pairs, and the numbers — ImageStat's nine properties, `entropy`, 30 `ImageMath` expressions, nine
`ImageMorph` operators — all match. What the corpus found on the way is recorded in decision PW18:
`Image.blend` rounded where `ImagingBlend` truncates (off by one on half of all inputs, and
`ImageEnhance` is built on `blend`), mode 1 packed a bit only for 255 where `Pack.c` packs any
non-zero value, `ModeFilter` had been given `RankFilter`'s border, and `ImageMorph`'s change count
included the border the C never writes. Each was a defensible reading of the documentation and wrong
against the code.

`ImageFilter`, `ImageChops`, `ImageEnhance`, `ImageOps`, `ImageStat`, `ImageMath` and `ImageMorph`
are, between them, about ninety calls. Almost all of them are three lines once the layer underneath
is right, and the interesting content is in the handful of places where the obvious implementation is
subtly not Pillow's.

### 7.1 `ImageFilter` — four unrelated algorithms wearing one name

`image.filter(f)` dispatches to four implementations that share nothing but a signature:

| family | Pillow's implementation | our shape |
|---|---|---|
| the ten built-ins, and `Kernel` | `Filter.c`'s `ImagingFilter3x3` / `ImagingFilter5x5` | 9 or 25 shifted multiply-adds, in float32 |
| `BoxBlur`, `GaussianBlur` | `BoxBlur.c`'s sliding accumulator | one `nd_cumsum` per axis per pass |
| `RankFilter`, `MedianFilter`, `MinFilter`, `MaxFilter` | `RankFilter.c`'s per-window sort | `nd_sort` along a stacked tap axis |
| `ModeFilter` | `ModeFilter.c`'s per-window histogram | `nd_bincount` per window offset |

Five things about the convolution are contracts rather than details, and four of the five are
surprises:

1. **Only 3×3 and 5×5 exist.** `ImagingFilter` rejects every other size with `bad kernel size`, and
   mode `F` with a mode error. A 7×7 `Kernel` is not slow in Pillow, it is an exception — so it is an
   exception here, with the same message.
2. **The kernel is applied vertically flipped.** `kernel[0..2]` multiplies row `y+1`, not `y-1`.
   Pillow's docstring says so; every built-in kernel is vertically symmetric, so it is invisible until
   someone writes an asymmetric one. Tested with a deliberately asymmetric kernel, because a
   symmetric corpus cannot catch a flip.
3. **The border is copied, not filtered.** The outer one row/column (two, for 5×5) of the output is
   the *input's* pixels, unweighted. Not zero-padded, not edge-extended — copied. An image smaller
   than the kernel is returned unchanged.
4. **The accumulation is `float`, not `double`.** `ss` is a C `float`, so every multiply and every
   add rounds to 24 bits of mantissa. In f64 the result differs by one on a few pixels in a thousand,
   which is exactly the kind of difference that looks like a bug in someone's test suite. So each
   step goes through `pxbuf.to_f32`, in Pillow's order: the three taps of a row are summed into one
   float32 value, and *that* is added to the accumulator.
5. **The rounding is folded into the offset.** Pillow does `offset += 0.5` once and then truncates,
   so the operation is round-half-up for a positive sum and *toward zero* for a negative one. And the
   kernel itself is pre-divided by `scale` in float32 — `kerneldata[i] /= divisor` — so the divided
   weights, not the integer ones, are what multiply the pixels.

#### 7.1.1 The box blur is a prefix sum, and that is the whole trick

`ImagingLineBoxBlur32` is a classic sliding accumulator: one add and one subtract per output pixel,
left to right, so it costs the same for radius 1 and radius 100. Written out it is a loop over
pixels with a carried dependency — the shape rule PW2 exists to forbid.

But a sliding window sum *is* a difference of prefix sums, and `nd_cumsum` computes all the prefix
sums in one call. Reading the C closely, each output is

```
bulk(x) = ww · Σ C(i) for i in [x−r, x+r]  +  fw · (C(x−r−1) + C(x+r+1))
out(x)  = (bulk(x) + 2²³) >> 24
```

where `C(i)` is the row clamped at both edges, `r = (int)radius`, `ww = (1<<24)/(radius·2+1)`
computed in **float32 and truncated**, and `fw = ((1<<24) − (r·2+1)·ww)/2` in integers. `fw` is the
weight of the two fractional-radius pixels at the window's edges, which is how a float radius works
at all.

So the whole thing becomes: one `nd_cumsum` along the axis, two gathers for the window ends, two more
for the fractional pixels, and the integer combination — **six array operations per pass per axis**,
independent of radius. The edge clamping that the C does with its `edgeA`/`edgeB` case analysis
becomes clamped index arrays, which is also why the degenerate `radius ≥ width` case needs no special
code: the indices simply all clamp.

`GaussianBlur` is this box blur run **three times per axis** with the radius transformed by
Gwosdek et al.'s formula — `_gaussian_blur_radius`, which solves for the box length whose triple
convolution has the requested variance. It is not a Gaussian kernel and `pillow` does not substitute
one (PW4): Pillow's approximation is the contract.

`UnsharpMask` is `GaussianBlur(radius, 3)` plus a thresholded difference, where
`diff * percent / 100` is **C integer division, truncating toward zero**, which for a negative `diff`
is not the floor. `transform.b`'s `_truncToward0` is reused for it rather than rediscovered.

#### 7.1.2 Rank filters sort a stack, not a window

`RankFilter(size, rank)` takes the `rank`-th smallest of the `size²` pixels in a window. Pillow
expands the image by `size // 2` with edge replication (`ImagingExpand`), then runs
`RankFilter.c`'s per-window insertion sort.

A sort per pixel is a pixel loop. The arrangement that is not: expand once, build the `size²` shifted
views as a new axis — `nd_slice` each, which is free — stack them into one `(h, w, bands, k²)` array,
and `nd_sort` along the last axis. One sort call for the whole image; the loop is over **taps**,
which PW2 permits. `MedianFilter`, `MinFilter` and `MaxFilter` are this with rank `k²/2`, `0` and
`k²−1`, exactly as Pillow derives them.

`ModeFilter` is the odd one out twice over. It needs the most frequent value in each window, with
Pillow's tie-break — values occurring once or twice are ignored, the lowest value wins a tie, and
the *current* pixel stands if nothing occurs more than twice — and the counts come from a comparison
chain over the 256 bins, which is a loop over **bins**.

And its border is not RankFilter's. **RankFilter expands the image with edge replication**, in
Pillow's *Python*, before the C sees it; **ModeFilter filters every pixel with the window clipped to
the image**, in C, with out-of-image taps simply not counted. Two neighbouring filters in one module,
two different edge rules, and nothing in either name says so — P6 shipped RankFilter's rule in
ModeFilter first and the corpus caught it. The clipping falls out of padding with **256**: a value no
histogram bin can match, so an out-of-image tap counts for nothing without a mask of its own.

### 7.2 `ImageChops` — twenty-four operations and two rounding regimes

The twenty-four channel operations are one-liners over whole arrays. What matters is that `Chops.c`
has **two** macros and they do not round the same way:

* `CHOP` computes in `int` and clamps to `[0, 255]`.
* `CHOP2` computes in `int` and **assigns to a `UINT8`** — so it wraps modulo 256.

`add_modulo` and `subtract_modulo` use the second deliberately; that is their entire purpose.
`difference`, `soft_light`, `hard_light` and `overlay` also use it, and for those the wrap is
*unreachable* — each expression is provably within `[0, 255]` for 8-bit inputs, which is checked by
evaluating the extremes rather than assumed. They are implemented with the wrap, and the test states
the bound that makes the wrap unobservable.

Three further contracts: the output size is the **minimum** of the two inputs in each axis (not an
error, not a union); `logical_and`/`or`/`xor` require *both* images to be mode `1`; and
`add`/`subtract` divide by a **float** `scale` and then truncate toward zero before clamping.

### 7.3 `ImageOps` — look-up tables, and the three that are not

Most of `ImageOps` is a 256-entry look-up table applied with `point()`, which is one `nd_take`
(§2). `autocontrast`, `equalize`, `posterize`, `solarize`, `invert` and `colorize` build a table and
hand it over; the interesting part of each is the table's integer arithmetic, which is reproduced
statement for statement because each one has a quirk:

* `autocontrast`'s cutoff removes pixels from the histogram ends with `int(n * cutoff // 100)`, and
  its mapping is `int(ix * scale + offset)` — a truncation of a float32-free double, clamped after.
  If `hi <= lo` it returns the identity table rather than dividing by zero.
* `equalize`'s step is `(sum(histo) − histo[−1]) // 255` where `histo` is the histogram **with zero
  bins removed** — so the last *non-zero* bin is subtracted, not bin 255. Then `n` starts at
  `step // 2` and the table entry is `n // step`. Getting this wrong gives a plausible, wrong,
  slightly-too-bright image.
* `colorize` interpolates with `i * (hi − lo) // len(range_map)` — floor division per channel, and
  the three segments are built with separate ranges whose lengths differ.
* `_lut` refuses mode `P` unless the table is to be applied to the **palette** rather than the
  indexes, which is why `invert`, `posterize` and `solarize` pass `palette=True` and `autocontrast`
  and `equalize` do not. A `P` image through `autocontrast` raises `NotImplementedError` in Pillow;
  it raises here too, with Pillow's message.

The three that are not look-up tables — `contain`, `cover`, `fit`, `pad`, `scale`, `deform` — are
geometry, and they are composition over P3's `resize` and `transform`. Their content is the rounding
of the target size (`round`, which is banker's rounding in Python, not `floor(x + 0.5)`) and the
`centering` arithmetic, both of which are reproduced exactly because an off-by-one in a thumbnail
size is a visible difference.

`exif_transpose` belongs here and needs P5: it reads the orientation tag, applies one of seven
transpositions, and **removes the tag from the copy it returns**, including from `info["exif"]` so
that a subsequent save does not rotate the image twice.

### 7.4 `ImageStat` is the histogram, and nothing but

Every `Stat` property is derived from `histogram()` — including `extrema`, which therefore reports
`(0, 255)` bins for `I` and `F` images rather than true extrema. That is a documented Pillow wart
(its own docstring says to use `getextrema()` instead), so it is reproduced: `count` is the bin sum,
`sum` is `Σ j·h[j]`, `median` is the first bin where the running sum **exceeds** half the count, and
`var` is `(sum2 − sum²/count)/count` — the computational form, with its cancellation, not the
two-pass form.

### 7.4.1 `quantize` is the one call with an error bound rather than byte equality

Decision PW17 has the numbers and the reasoning; the short version is that median cut as an
*algorithm* is reproducible and is reproduced, while `Quant.c`'s colour **hash** — which merges
colours whose hashes collide and coarsens the whole colour space when its table overflows — is an
implementation detail, not an algorithm. So `quantize.b` works from the exact distinct colours and
the gate is measured error against the source rather than equality with Pillow's palette. Median cut
comes out a wash and the octree 5× better at 16 colours.

The shape is worth stating because it is PW2 again, one level down: a box is three index **arrays**,
and a split is `nd_cumsum` for the count median, `nd_compress` for the partition and `nd_min`/`nd_max`
for the new bounds — about fifteen array calls per split, whatever the box holds. The first
transcription used Bantu lists and per-colour loops, the way `Quant.c`'s linked lists read, and took
**48 seconds** on a 160×120 image; localising the dict-field reads took it to 8.7 s and vectorising
the splits to 0.3 s. Rule PW2 says no loop proportional to the pixel count; 19,000 distinct colours
is the same kind of data.

### 7.5 `ImageMath` — an expression evaluator, and six operators Bantu's arrays did not have

`ImageMath` evaluates arithmetic over whole images in mode `I` (int32) or `F` (float32), converting
`1`/`L` up to `I` and mixing modes by promoting both to `F`. The operator semantics come from
`_imagingmath.c` and three of them are not what the host language would do:

* **division by zero yields zero**, for both int and float. It is not an error and not an infinity.
* integer division and `%` are **C's**: truncating, with `%` taking the sign of the dividend.
* `pow` is `(int)(pow(x, y) + 0.5)` clamped to int32 for mode `I` — rounded, not truncated, and
  saturating rather than wrapping.

Pillow's entry points are `lambda_eval(f, **kw)`, which calls a *callable* with a dict of operands,
and `unsafe_eval(expr, **kw)`, which compiles a Python expression. The first maps onto Bantu directly
— pass a Bantu function, receive a dict. The second cannot: there is no Python to compile. So
`math.b` ships **a small expression evaluator for exactly the grammar Pillow's examples use** —
`+ - * / % ** & | ^ << >> ~` with Python's precedence, the comparisons, parentheses, numbers, names,
and the named functions `abs min max convert float int equal notequal`. An expression outside that
grammar is a parse error naming the offending token, not a silent misread. That is the honest
mapping, and `docs/pillow.md` states the grammar rather than implying that arbitrary Bantu is
evaluated.

**The six operators `numba` was missing.** `&`, `|`, `^`, `~`, `<<` and `>>` have no `nd_*` form:
NumPy has all six, `numba` had none, so there was no way to do bitwise arithmetic on an array at all
— not in `pillow`, not in a hash, not in a bitset. Doing it in Bantu means 32 array passes per
operation, one per bit. This is the same kind of gap as P1's two atoms: general, not
`pillow`-specific, and a defect in `numba` rather than a missing feature of this package. So
`nd_bitwise_and`, `nd_bitwise_or`, `nd_bitwise_xor`, `nd_bitwise_not`, `nd_left_shift` and
`nd_right_shift` were added to `numba`, integer-only (a float operand is an error, as in NumPy),
with `>>` arithmetic on negatives — and `jpeg.b`'s `& 1023` and `DESCALE` use them too.

### 7.6 `ImageMorph` — a 512-entry look-up table over the 3×3 neighbourhood

Binary morphology on mode `1` images, where the operation is defined entirely by a table indexed by
the nine neighbours as nine bits. `LutBuilder` compiles a pattern language (`4:(... .1. 111)->1`,
with rotation, mirror and negate operators) into that table; `MorphOp.apply` applies it.

Applying it is nine shifted comparisons, a weighted sum, and one `nd_take` of the 512-entry table —
no pixel loop, and the same arrangement for every pattern anyone writes. `match` and `get_on_pixels`
return coordinate lists, which `nd_nonzero` answers directly. The border is left at zero, as
`_imagingmorph.c` leaves it.

## 8. `ImageDraw` and `ImageFont`

### 8.1 The rasteriser

`ImageDraw` is vector drawing onto a raster, which is per-pixel by nature — so it is written the way
§1.1 demands:

- **`rectangle`** is a slice assignment. Free.
- **`line`** is Bresenham, but **vectorised**: the major-axis coordinates are an `nd_arange`, the
  minor-axis coordinates an integer division of it, and the pixels are written with one `nd_put`.
  One array call per line, whatever its length.
- **`polygon` / `ellipse` / `pieslice` / `chord` / `arc` / `rounded_rectangle`** are scanline
  fills: per output row, the edge crossings are sorted and each span is a slice assignment. The
  loop is over **rows**, and within a row over **spans** — not over pixels. A filled 1080p polygon
  is ~1,080 iterations.
- **`floodfill`** is a scanline flood fill, same shape: a stack of spans, each span an array write.
- **Anti-aliasing** is not in Pillow's `ImageDraw` and is not here either. Pillow's documented
  idiom — draw at 4× and `resize` down — works, because `resize` is fast.
- **`width=`** on lines and outlines, `joint="curve"`, and the `ImagePath`/`ImageDraw.Outline`
  surfaces all behave as Pillow's do.

### 8.2 Fonts

`ImageFont.load_default()` returns an embedded bitmap font, so text works with no file and no
dependency.

`ImageFont.truetype(path, size)` is a **TrueType reader written in Bantu**: `cmap` (formats 0, 4, 6,
12), `glyf`/`loca` outlines, `hmtx` advances, `kern`/`GPOS` pair kerning, and composite glyphs
decomposed. Quadratic curves are flattened by integer Bernstein weights and filled through §8.1's
scanline filler. Per-glyph cost is proportional to glyph *area*, and a glyph is small — a 24 px
label is a few thousand pixels in total, and the glyph is **cached by (font, size, codepoint)** so a
repeated character is a slice copy. `getbbox`, `getlength`, `getmask`, `font_variant`, the
`layout_engine` argument, `anchor`, `spacing`, `align`, `direction` and `stroke_width` are all
present; `variation_set_by_name` reads `fvar`.

There is prior art in this repository: bplot's raster backend already renders DejaVu glyphs from
integer outline tables generated by `scripts/gen_font_tables.b`, *a TrueType reader written in
Bantu*. `font.b` is that reader, promoted to run at load time over an arbitrary font file.

## 9. Security

An imaging library is a parser of hostile input. This is the attack surface of the whole package, and
it is designed, not patched.

| hazard | the defence |
|---|---|
| **decompression bomb** — a 4-byte PNG header declaring 60,000×60,000 | `MAX_IMAGE_PIXELS` (default 89,478,485, Pillow's number) checked **before** any allocation, from the header alone; `Image.open` is lazy and does not decode until `load()` |
| **inflate bomb** — 1 KB expanding to 10 GB | `inflate` takes a maximum-output argument; the PNG decoder passes the exact byte count its own header implies, so an over-long stream is an error, not an allocation |
| **integer overflow in dimensions** — `width × height × bands` wrapping | every dimension product goes through a checked multiply before it reaches `nd_zeros`; numba's `checkedMul` already refuses on its side |
| **malformed chunk lengths / truncated streams** | every read is bounds-checked against the buffer length and raises a catchable Bantu error naming the format and the offset. A truncated file never reads past its end, and never returns a partially-initialised image |
| **PNG CRC and zlib Adler-32 mismatch** | checked; a mismatch raises unless `LOAD_TRUNCATED_IMAGES` is set, matching Pillow |
| **cyclic / self-referential structures** — a TIFF IFD chain pointing at itself, an ICO pointing at itself | every chain has a visited set and a depth limit |
| **hostile text in metadata** reaching an SVG or HTML page | `pillow` never emits markup. Metadata is returned as Bantu strings and is the caller's to escape — stated in the docs, because the mistake is reachable |
| **path traversal in `save()`** | `save` writes exactly the path given and does nothing clever with it |

A **fuzz tier** rides along with every codec phase: the format's own test corpus, bit-flipped;
truncated at every length; with every length field replaced by 0, 1, `2^31−1` and `2^63−1`. The gate
is that **no input crashes the interpreter and no input hangs it** — every failure is a catchable
error.

## 10. The test gate

Five tiers, all green before a phase advances — the gate `numba` and `bplot` use:

| tier | proves | form |
|---|---|---|
| 1. **feature** | every call does what it claims, including its failure modes | `tests/pillow_*_test.b`, ending `RESULT: ALL GREEN` |
| 2. **differential** | the output is *right*, not merely stable — **byte-identical to Python Pillow** | `tests/pillow_differential.sh` writes the same operation from both and compares. Where Pillow is unavailable, checked-in golden files generated from it |
| 3. **stress** | it survives size, repetition and abuse | scale (and the **timing curve is asserted linear**, which is how a pixel loop gets caught) · degenerate (0×0, 1×1, 1×65535) · hostile (§9's fuzz corpus) · memory |
| 4. **regression** | nothing else broke | the whole existing suite on the same build |
| 5. **sanitizers** | no latent memory defect | ASan + UBSan over the two native atoms of §4 |

**Tier 2 is the one that matters most here**, and it is the reason this package can claim Pillow's
API rather than merely resemble it: for every operation, the same input through both libraries
produces the same bytes. Where it cannot — JPEG encoding, which depends on the encoder's own
quantisation choices — the test asserts a PSNR floor against Pillow's output and the difference is
documented.

## 11. Phases

Each is a working, tested increment; nothing is merged half-built. Detail, with per-item test
columns, in [`pillow-suite/ROADMAP.md`](../pillow-suite/ROADMAP.md).

| phase | content | why here |
|---|---|---|
| **P0** ✅ | this document, `ROADMAP.md`, `DECISIONS.md`, `CHANGELOG.md` | design before code |
| **P1** ✅ | the two atoms of §4, with `inflate.b`'s pure-Bantu twin and the differential test between them | nothing else can be built on a one-way compressor |
| **P2** ✅ | `mode.b`, `image.b`, `convert.b`, `color.b`, `palette.b` — the object model, the full mode matrix, `new`/`frombytes`/`fromarray`/`tobytes`, `copy`/`crop`/`paste`/`split`/`merge`/`getpixel`/`putpixel`/`point`/`getbbox`/`getextrema`/`histogram`, PPM read+write | the core, and PPM makes it testable end to end on the first day |
| **P3** ✅ | `resample.b` — the six filters, `resize`/`rotate`/`transpose`/`thumbnail`/`reduce`/`transform`; `transform.b`, `path.b` | every later phase wants to resize |
| **P4** ✅ | `codecs/png.b`, `codecs/bmp.b`, `codecs/tga.b` | PNG is the format that matters most and the one P1 unblocked |
| **P5** ✅ | `codecs/jpeg.b` — baseline, progressive, `draft`, encode; `exif.b`, `exif_tags.b` | the second format that matters most |
| **P6** ✅ | `filter.b`, `chops.b`, `enhance.b`, `ops.b`, `stat.b`, `math.b`, `morph.b`, `quantize.b` | the operation surface, once there is something to operate on |
| **P7** | `draw.b`, `font.b`, `path.b` | drawing and text |
| **P8** | `codecs/gif.b` (+ `sequence.b`), `tiff.b`, `ico.b`, `pcx.b`, `xbm.b`, `webp.b`, the read-only long tail, PDF write | the rest of the formats, in descending order of use |
| **P9** | `docs/pillow.md` complete, `samples/pillow/`, registry publication, and the `u8`-dtype measurement of PW7 | ship |

---

## Appendix — the API surface this package is held to

`pillow` is finished when this list is. Tracked item by item in the roadmap.

**`Image`** — `new open frombytes frombuffer fromarray merge blend composite eval alpha_composite
register_open register_save register_mime register_extension init preinit effect_noise linear_gradient
radial_gradient` · constants `NEAREST BOX BILINEAR HAMMING BICUBIC LANCZOS FLIP_LEFT_RIGHT
FLIP_TOP_BOTTOM ROTATE_90 ROTATE_180 ROTATE_270 TRANSPOSE TRANSVERSE AFFINE EXTENT PERSPECTIVE QUAD
MESH NONE ORDERED RASTERIZE FLOYDSTEINBERG WEB ADAPTIVE MEDIANCUT MAXCOVERAGE FASTOCTREE LIBIMAGEQUANT`

**`Image` instance** — `mode size width height info palette readonly` · `convert copy crop paste
resize rotate transpose thumbnail reduce transform split getchannel putalpha getbands getbbox
getcolors getdata getextrema getpalette getpixel putdata putpalette putpixel point filter histogram
entropy quantize remap_palette load save seek tell close verify tobytes tobitmap effect_spread draft
show alpha_composite getim apply_transparency`

**`ImageChops`** — `add add_modulo blend composite constant darker difference duplicate hard_light
invert lighter logical_and logical_or logical_xor multiply offset overlay screen soft_light subtract
subtract_modulo`

**`ImageColor`** — `getrgb getcolor colormap`

**`ImageDraw`** — `Draw getdraw floodfill` · instance `arc bitmap chord ellipse line pieslice point
polygon rectangle regular_polygon rounded_rectangle shape text multiline_text textbbox
multiline_textbbox textlength getfont` · `ImageDraw.Outline`

**`ImageEnhance`** — `Color Contrast Brightness Sharpness`, each with `enhance`

**`ImageFile`** — `ImageFile Parser StubImageFile raise_oserror`

**`ImageFilter`** — `BLUR CONTOUR DETAIL EDGE_ENHANCE EDGE_ENHANCE_MORE EMBOSS FIND_EDGES SHARPEN
SMOOTH SMOOTH_MORE GaussianBlur BoxBlur UnsharpMask Kernel RankFilter MedianFilter MinFilter
MaxFilter ModeFilter Color3DLUT`

**`ImageFont`** — `load_default truetype load load_path ImageFont FreeTypeFont TransposedFont` ·
instance `getbbox getlength getmask getmask2 font_variant get_variation_names
set_variation_by_name get_variation_axes set_variation_by_axes`

**`ImageMath`** — `eval lambda_eval unsafe_eval`

**`ImageMorph`** — `LutBuilder MorphOp`

**`ImageOps`** — `autocontrast colorize contain cover crop deform equalize expand fit flip grayscale
invert mirror pad posterize scale solarize exif_transpose`

**`ImagePalette`** — `ImagePalette raw load negative random sepia wedge`

**`ImagePath`** — `Path` with `compact getbbox map tolist transform`

**`ImageSequence`** — `Iterator all_frames`

**`ImageStat`** — `Stat` with `extrema count sum sum2 mean median rms var stddev`

**`ImageTransform`** — `AffineTransform ExtentTransform QuadTransform MeshTransform`

**`ExifTags`** — `TAGS GPSTAGS IFD Base GPS Interop`

**Not implemented, and why** — `ImageTk` `ImageQt` `ImageWin` `ImageShow` `ImageGrab`: each binds a
GUI toolkit or an OS screen-capture API that Bantu has no binding to. `ImageCms`: ICC colour
management is a library-sized problem of its own, and faking it is worse than declining it; ICC
profiles are *preserved* in `info["icc_profile"]` and round-trip through save, they are just not
applied. Both exclusions are stated in `docs/pillow.md`, not silently omitted.
