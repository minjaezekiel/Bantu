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
> loop over pixels is a defect, and the stress tier (§9) is built to catch one by its timing curve.

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
  draw.b          ImageDraw — the vector rasteriser (§7)
  font.b          ImageFont — TrueType and BDF/PCF readers, glyph rasterisation (§7.2)
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

### 6.2 Format by format

Status after P4: **PNG, BMP, TGA and PPM are done and differentially verified.** PngSuite — the
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
| **WebP** | ✅ lossless (VP8L) and still lossy (VP8) | ✅ lossless | the largest single codec; phased last (§10, P8) |
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

JPEG's entropy-coded scan is a bit-by-bit Huffman walk over every non-zero coefficient. Measured
cost per coefficient is ~8 interpreted operations, so:

| JPEG | coefficients | decode |
|---|---|---|
| 640×480 | ~0.3 M | ~1.2 s |
| 1600×1200 | ~1.9 M | ~7.6 s |
| 4000×3000 | ~12 M | ~48 s |

Everything *after* the scan is array work and costs a fraction of this. The scan is written in Bantu,
it is correct, and its cost is published in [`docs/pillow.md`](pillow.md) next to the API rather than
left for a user to discover. `draft()` — Pillow's own answer, DCT-domain downscaling that decodes a
1/2, 1/4 or 1/8 image — is implemented, and it makes a thumbnail of a 12 MP JPEG a **0.8 s**
operation by never decoding the full scan. That is the right fix and it is Pillow's fix.

If measurement after P4 shows the full-size scan is the thing users actually hit, the answer is one
more generic atom — a canonical-prefix-code scanner, which is what `band`/`rotr` are to `hash` — and
[decision PW9](../pillow-suite/DECISIONS.md) records the trigger for it. It is not added
speculatively.

## 7. `ImageDraw` and `ImageFont`

### 7.1 The rasteriser

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

### 7.2 Fonts

`ImageFont.load_default()` returns an embedded bitmap font, so text works with no file and no
dependency.

`ImageFont.truetype(path, size)` is a **TrueType reader written in Bantu**: `cmap` (formats 0, 4, 6,
12), `glyf`/`loca` outlines, `hmtx` advances, `kern`/`GPOS` pair kerning, and composite glyphs
decomposed. Quadratic curves are flattened by integer Bernstein weights and filled through §7.1's
scanline filler. Per-glyph cost is proportional to glyph *area*, and a glyph is small — a 24 px
label is a few thousand pixels in total, and the glyph is **cached by (font, size, codepoint)** so a
repeated character is a slice copy. `getbbox`, `getlength`, `getmask`, `font_variant`, the
`layout_engine` argument, `anchor`, `spacing`, `align`, `direction` and `stroke_width` are all
present; `variation_set_by_name` reads `fvar`.

There is prior art in this repository: bplot's raster backend already renders DejaVu glyphs from
integer outline tables generated by `scripts/gen_font_tables.b`, *a TrueType reader written in
Bantu*. `font.b` is that reader, promoted to run at load time over an arbitrary font file.

## 8. Security

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

## 9. The test gate

Five tiers, all green before a phase advances — the gate `numba` and `bplot` use:

| tier | proves | form |
|---|---|---|
| 1. **feature** | every call does what it claims, including its failure modes | `tests/pillow_*_test.b`, ending `RESULT: ALL GREEN` |
| 2. **differential** | the output is *right*, not merely stable — **byte-identical to Python Pillow** | `tests/pillow_differential.sh` writes the same operation from both and compares. Where Pillow is unavailable, checked-in golden files generated from it |
| 3. **stress** | it survives size, repetition and abuse | scale (and the **timing curve is asserted linear**, which is how a pixel loop gets caught) · degenerate (0×0, 1×1, 1×65535) · hostile (§8's fuzz corpus) · memory |
| 4. **regression** | nothing else broke | the whole existing suite on the same build |
| 5. **sanitizers** | no latent memory defect | ASan + UBSan over the two native atoms of §4 |

**Tier 2 is the one that matters most here**, and it is the reason this package can claim Pillow's
API rather than merely resemble it: for every operation, the same input through both libraries
produces the same bytes. Where it cannot — JPEG encoding, which depends on the encoder's own
quantisation choices — the test asserts a PSNR floor against Pillow's output and the difference is
documented.

## 10. Phases

Each is a working, tested increment; nothing is merged half-built. Detail, with per-item test
columns, in [`pillow-suite/ROADMAP.md`](../pillow-suite/ROADMAP.md).

| phase | content | why here |
|---|---|---|
| **P0** ✅ | this document, `ROADMAP.md`, `DECISIONS.md`, `CHANGELOG.md` | design before code |
| **P1** ✅ | the two atoms of §4, with `inflate.b`'s pure-Bantu twin and the differential test between them | nothing else can be built on a one-way compressor |
| **P2** ✅ | `mode.b`, `image.b`, `convert.b`, `color.b`, `palette.b` — the object model, the full mode matrix, `new`/`frombytes`/`fromarray`/`tobytes`, `copy`/`crop`/`paste`/`split`/`merge`/`getpixel`/`putpixel`/`point`/`getbbox`/`getextrema`/`histogram`, PPM read+write | the core, and PPM makes it testable end to end on the first day |
| **P3** ✅ | `resample.b` — the six filters, `resize`/`rotate`/`transpose`/`thumbnail`/`reduce`/`transform`; `transform.b`, `path.b` | every later phase wants to resize |
| **P4** ✅ | `codecs/png.b`, `codecs/bmp.b`, `codecs/tga.b` | PNG is the format that matters most and the one P1 unblocked |
| **P5** | `codecs/jpeg.b` — baseline, progressive, `draft`, encode; `exif.b` | the second format that matters most |
| **P6** | `filter.b`, `chops.b`, `enhance.b`, `ops.b`, `stat.b`, `math.b`, `morph.b` | the operation surface, once there is something to operate on |
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
