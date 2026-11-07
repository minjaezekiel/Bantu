# pillow — Changelog

Entries are tagged `[feature]` / `[bug fix]` / `[patch]` / `[found]`, newest first. A phase appears
here only once all five tiers of its gate are green.

---

## Phase P7 — Text ✅ (2026-11-07)

**69 checks green.** Bitmap-font text is **byte-identical to Pillow**: 71 drawings covering every
string, every mode a stencil can reach, all eighteen anchors, four alignments, the multiline
spacing, the blended path and both edges. TrueType **advances are exact**: 665 single-glyph
advances and 70 kerned pairs over seven sizes, plus 44 whole-string lengths, all to the last 1/64 of
a pixel. TrueType boxes and pixels are **measured**, in PW17's shape.

- `[feature]` **`pillow/font.b`** — `load_default` (Pillow's own embedded courB08 sheet, carried as
  base64 with its HPND notice, so text works with no file and no dependency), `truetype` (a
  TrueType reader in Bantu: `cmap` formats 0/4/6/12, `glyf`/`loca`, `hmtx`, `kern` format 0 and
  GPOS lookup type 2 pair kerning, composite glyphs with scaling and point matching, `fvar`
  introspection, `name`), `load`/`load_path`, `getbbox getlength getmask getmask2 getmetrics
  getname font_variant`, and a glyph cache keyed by (font, size, glyph) — 43 glyphs at 24 px cost
  17 ms cold and 4 ms warm.
- `[feature]` **A non-zero-winding glyph rasteriser**, sixteen sub-scanlines per pixel row with
  exact horizontal coverage. Non-zero and not even-odd, because a counter is wound against its
  outer contour and the two rules part company the moment two same-direction contours overlap.
- `[feature]` **`text multiline_text textlength textbbox multiline_textbbox`** on a `Draw`, with
  `anchor spacing align stroke_width stroke_fill` — ImageText's line splitting whole, including
  `justify` and the vertical anchors it refuses on multiline text.
- `[found]` **The advances come from two libraries, and round differently.** Pillow's default layout
  engine is libraqm, so a glyph advance is FreeType's `FT_MulFix(hmtx, x_scale)` while the GPOS pair
  adjustment is HarfBuzz's `em_scale`, `floor((v·size₆₄ + upem/2) / upem)`. Eight of DejaVu's
  seventy kern pairs land on a tie where the two rules disagree by 1/64 of a pixel. Found by
  comparing every pair at every size.
- `[found]` **A Bantu string is a byte string**, so `len` on an eight-letter Swedish word is 10.
  Every string that reaches a font is decoded to codepoints first, with a Latin-1 fallback for
  invalid UTF-8 — which is also how Pillow treats a `bytes` argument.
- `[found]` **Three more mutations applied to a copy**, in the rasteriser this time: `_quad` pushed
  its flattened segments into a copy of the caller's list, the per-row active-edge table pushed into
  a copy of a list element, and the coverage accumulator was handed to a helper by value. All three
  produced a correctly sized, entirely empty glyph. The active-edge table is now one sorted flat
  list of `(row, edge)` keys and the span accumulation is inlined.
- `[patch]` **`stroke_width` is a dilation by a disc**, not `FT_Stroker`: a round stroke of radius
  *r* is the Minkowski sum of the glyph with a disc of radius *r*, so the same shape is computed on
  the raster. The loop is over the disc's offsets, each one a whole-array maximum.
- `[patch]` `direction`, `features`, `language` and `embedded_color` **raise**, naming what each
  would take, rather than being accepted and ignored. So does a CFF `.otf`, a WOFF, and
  `set_variation_by_name`.
- `[patch]` **16 sub-scanlines, measured not chosen.** 32 moved the agreement with Pillow by 0.0003,
  because the residual is FreeType's grid fitting. The comment records it so nobody re-runs it.

---

## Phase P7 — Drawing ✅ (2026-11-07)

**75 checks green, and 694 of 694 drawings byte-identical to Pillow.** Every geometric primitive at
every width that changes its shape: 52 line slopes and signs from the centre of a 29×29 field, 56
wide lines, 3 curve joints, 56 polygons over nine shapes including a self-intersecting star and a
run of collinear horizontal edges, 70 ellipses over ten bounding-box parities, 234 arcs, chords and
pieslices over eighteen angle pairs both filled and outlined, 30 rounded rectangles with their
corner masks and three degenerate forms, 18 regular polygons, 30 bitmaps through both a `"1"`
stencil and an `"L"` matte, 30 blended RGBA-on-RGB, 17 floodfills, and one composite scene in each
of the twenty modes. `tests/pillow_differential.sh` now also draws the same scene on both sides live
against the installed Pillow.

- `[feature]` **`pillow/draw.b`** — `Draw getdraw floodfill Outline`, and on the context `point line
  rectangle polygon regular_polygon ellipse circle arc chord pieslice rounded_rectangle shape
  bitmap`, with `width=`, `joint="curve"`, `corners=` and RGBA-on-RGB blending. Ported from
  `Draw.c` rule by rule: the endpoint a line never draws, the strict `dx > dy` that sends a 45°
  tie to the y-major loop, `hline`'s refusal to swap its ends, the rectangle that normalises only
  `y`, the polygon's float32 crossings with their doubled count at an edge's last row and the
  corner-join fix-up, and the ellipse quarter-tracer whose third candidate does not refresh the
  running minimum.
- `[feature]` **A line is decomposed into its runs, not its pixels.** Inverting Bresenham's error
  recurrence gives the run boundaries in closed form, so a segment is `min(|dx|, |dy|) + 1` span
  writes rather than `max(|dx|, |dy|)` pixel writes — the same pixels, every tie included. The
  design of record said `nd_put`; `nd_put` materialises the whole destination's walk order per call,
  which a polyline cannot afford. A 1080p diagonal is 55 ms.
- `[feature]` **`floodfill` is a span fill over a one-pass fillability mask**, with each row's
  blocker positions a cached `nd_nonzero` held as an array. The timing curve is asserted: a
  1600×100 filled polygon costs 12 ms and a 100×1600 one 140 ms, so the cost follows the height and
  not the area (decision PW8).
- `[found]` **Three latent Bantu defects of one family**, all of them a mutation applied to a copy:
  the polygon corner-join adjusted a copy of the crossings list, the horizontal-edge merge widened a
  copy of the last edge, and moving the edge table into a dict to make it mutable turned
  `$T.t[$i]` quadratic. The rule is now written down in the architecture: mutable state crosses a
  call boundary in a dict, never in a list, and a list read in a hot loop stays a local.
- `[bug fix]` **`LAB`'s wire packing was missing Pillow's `+128` offset on `a` and `b`** —
  `tobytes` and `frombytes` were the one pair of calls in the package that disagreed with Pillow
  for a whole mode, and had been for five phases because nothing had ever compared `LAB` bytes.
  Fixed at the wire; `tests/pillow_core_test.b` pins all three numbers.
- `[found]` **Five upstream defects, fixed under PW11 rather than reproduced**, each published in
  `docs/pillow.md` beside the call with a test asserting what the fix preserves:
  `ImagingOutlineTransform` frees the edges it has just transformed; `fill_mask_L` and
  `fill_mask_1` blend the *bytes* of an `I`, `F` or `I;16` pixel instead of its value;
  `DIV255`'s shift form overflows past 16 bits, so a 16-bit matte of 255 over ink 40,000 lands on
  39,999; `point8` and `hline8` disagree about `I;16B`'s byte order, so Pillow's own `I;16B`
  rectangle outline is half big-endian; and a wide outline's mask is built with the *target* mode's
  ink for 1, which on `F` is a float32 whose low byte is zero, so the outline silently vanishes.

---

## Phase P6 — The operation surface ✅ (2026-10-30)

**149 checks green.** 294 whole images byte-identical to Pillow — every filter over every mode it
accepts, every `ImageOps` call, four enhancements at five factors over three modes, `Color3DLUT`,
the `convert()` matrix and both gradients. All 24 channel operations exact over **every one of the
65,536 input pairs**. ImageStat's nine properties, `entropy`, 30 `ImageMath` expressions and nine
`ImageMorph` operators exact.

- `[feature]` **`pillow/filter.b`** — the ten built-ins and `Kernel` (3×3 and 5×5 only, as
  `ImagingFilter` allows, with the border copied and the kernel applied vertically flipped),
  `BoxBlur` and `GaussianBlur` as prefix sums (six array operations per axis per pass, so radius 1
  and radius 10 cost the same — asserted), `UnsharpMask`, `RankFilter`/`Median`/`Min`/`Max` as one
  `nd_sort` over stacked taps, `ModeFilter`, and `Color3DLUT` with ColorLUT.c's fixed-point
  trilinear interpolation. The convolution accumulates in **float32**, rounding at every multiply
  and every add, because in f64 the result differs by one on a few pixels in a thousand.
- `[feature]` **`pillow/chops.b`** — 24 operations across Chops.c's two rounding regimes: `CHOP`
  clamps, `CHOP2` wraps. The four operations that use the wrap unreachably are implemented with it
  anyway, and the test states the bound that makes it unobservable over all 65,536 pairs.
- `[feature]` **`pillow/ops.b`** — the look-up-table operations with their integer quirks
  (`equalize`'s step subtracts the last NON-ZERO bin; `autocontrast`'s cutoff empties bins from both
  ends; `colorize` interpolates by floor division over segments of different lengths) and the six
  that are geometry, with Python's round-half-to-even on every target size.
- `[feature]` **`pillow/enhance.b`, `pillow/stat.b`, `pillow/morph.b`** — the four enhancements as
  `blend(degenerate, image, factor)`; every statistic from the 256-bin histogram, Pillow's `extrema`
  wart included; morphology as a 512-entry table over the 3×3 neighbourhood, with the pattern
  language (`4`, `M`, `N`) compiled the way `LutBuilder` compiles it.
- `[feature]` **`pillow/math.b`** — the operators of `_imagingmath.c` (division by zero yields
  **zero**; `%` takes the dividend's sign; `pow` in mode I rounds and saturates) plus **an
  expression evaluator** for the grammar Pillow's examples use, since `unsafe_eval` compiles Python
  and there is no Python here. An expression outside the grammar names the offending token; the
  grammar is the sandbox.
- `[feature]` **`pillow/quantize.b`** — median cut, max coverage and fast octree. This is what
  `convert("P")` needed for a source with alpha, and what GIF will need in P8. Every split is an
  array operation — `nd_cumsum` for the count median, `nd_compress` for the partition, `nd_min`/`max`
  for the bounds — so the cost is ~15 array calls per split whatever the box holds.
- `[feature]` **The Image methods that were waiting on P6** — `filter`, `quantize`, `entropy`,
  `histogram` with a **mask**, `convert` with a **matrix**, `putpalette`, `getpalette`,
  `remap_palette`, `apply_transparency`, `tobitmap`, `effect_spread`, `getim`, and
  `linear_gradient` / `radial_gradient` / `effect_noise`.
- `[bug fix]` **`Image.blend` rounded where Pillow truncates.** `ImagingBlend` is
  `(UINT8)(in1 + alpha * (in2 - in1))` — a C cast, so it truncates toward zero — and it is an 8-bit
  operation that refuses I and F with "image has wrong mode". Ours computed
  `im1*(1-a) + im2*a` and rounded, which is off by one on **half** of all inputs. The 65,536-pair
  chops corpus caught it; `ImageEnhance` depends on `blend`, so this was four modules wide.
- `[bug fix]` **Mode 1 packed a bit only for 255, where Pack.c packs any non-zero value.** Visible
  the moment a mode-1 image goes through a filter, whose interior holds the convolution's 0..255
  output: four of the corpus images differed until `_packBilevel` tested `!= 0`.
- `[bug fix]` **`ModeFilter` had RankFilter's border.** RankFilter expands the image with edge
  replication (in Pillow's *Python*, before the C sees it); ModeFilter does not — it filters every
  pixel with the window CLIPPED to the image. Two neighbouring filters, two different edge rules,
  and the fix is to pad with 256: a value no histogram bin can match.
- `[bug fix]` **`Color3DLUT`'s table conversion rounded the wrong way.** The 16-bit fixed point is
  `round(v * (255 << 6))` away from zero, not `trunc(v * (255 << 6 | 32))`; the difference moved 24
  of 429 values by one.
- `[bug fix]` **`ImageMorph`'s change count included the border.** `_imagingmorph.c` counts only the
  interior it writes, so every operator was inflated by the same number of border pixels — which is
  how it was spotted: every count was off by exactly 21.
- `[patch]` **quantize was 145× too slow, twice over, and both times for PW14's reason.** Reading
  `$h.r[$c]` inside a loop copies the whole list (a list comes out of a dict by copy), which cost 48
  seconds on a 160×120 image; localising the four columns took it to 8.7 s; vectorising the splits
  took it to 0.3 s. The rule PW2 states for pixels applies to distinct colours too.

---

## Phase P5 — JPEG and EXIF ✅ (2026-10-30)

**87 checks green.** The corpus in `tests/fixtures/pillow/reference/jpeg/` — 54 files spanning every
subsampling Pillow writes and five that it cannot, baseline and progressive, restart markers by block
and by row, greyscale, CMYK, YCCK, quality 1 to 100, `optimize`, and draft mode at 1/2, 1/4 and 1/8 —
decodes **byte-identically to Pillow**, which is to say to libjpeg-turbo. The encoder is
**byte-identical to libjpeg** on all 17 baseline cases and pixel-identical on the two progressive
ones. EXIF written here is read back tag for tag by Python Pillow's `Image.getexif()`.

- `[feature]` **`pillow/codecs/jpeg.b`** — baseline (SOF0/SOF1) and progressive (SOF2) decode: all
  four of progressive's successive-approximation paths, every subsampling including the ones only
  `cjpeg` writes (4:1:1, 2x1, 1x2, 4x2, 3x1, mixed), restart markers, JFIF, Adobe APP14, EXIF, ICC
  profiles reassembled from numbered APP2 segments, CMYK and YCCK with Adobe's inversion, and
  `draft()`'s scaled decoding. The entropy scan is one function with the bit accumulator in locals
  and an 8-bit Huffman lookahead — libjpeg's own `HUFF_LOOKAHEAD` trick — because a Bantu call is
  ~0.3 µs and a per-symbol helper would pay it on every symbol.
- `[feature]` **JPEG encode** — quality 1–100 with Annex K's tables scaled libjpeg's way, `optimize`
  (per-image Huffman tables built by T.81 Annex K.2's algorithm, including the 16-bit length limit
  and the reserved all-ones codeword), `progressive`, the three subsamplings, caller-supplied
  `qtables`, Pillow's named presets, `quality="keep"`, restart intervals by block or by row, and
  `dpi`/`exif`/`icc_profile`/`comment` written as JFIF density, APP1, APP2 and COM.
- `[feature]` **`pillow/jpegdct.b`'s encoder half** — `jpeg_fdct_islow`, `jccolor`'s RGB→YCbCr
  tables, `jcsample`'s downsamplers *with their alternating rounding bias* (0,1,0,1 for h2v1 and
  1,2,1,2 for h2v2 — drop it and the image loses half a level of brightness), and `jcdctmgr`'s
  round-to-nearest-by-magnitude quantisation. All of it vectorised the way the inverse DCT is: ~70
  array operations per component, whatever the image size.
- `[feature]` **`pillow/exif.b` and `pillow/exif_tags.b`** — both byte orders, all twelve TIFF field
  types plus BigTIFF's three, the Exif / GPSInfo / Interop sub-IFDs, MakerNote as raw bytes, the
  thumbnail IFD, read **and write**; `ExifTags.TAGS / GPSTAGS / IFD / Base / GPS / Interop`
  generated from Pillow's own dictionaries by `tools/gen_exif_tags.py` rather than retyped. Plus
  `exif_transpose`, over all eight orientations, which drops the tag from the result as Pillow does.
- `[feature]` **`Image.draft($mode, $size)`** and **`Image.save($path, $format, $options)`** — save
  keywords arrive as one dict, because Bantu has no keyword arguments.
- `[bug fix]` **The scaled inverse DCTs had the 8×8 kernel's constants.** `jpegdct.b`'s `idct4` and
  `idct2` were carrying `FIX_0_298631336` and friends from `jpeg_idct_islow`, the wrong left shift on
  the DC term (`CONST_BITS` instead of `CONST_BITS+1`/`+2`) and the wrong descale amounts. The
  scaled kernels in `jidctred.c` are separate derivations with their own constants; borrowing the
  8×8 ones produces an image that is recognisable and wrong. Seven of the nine draft fixtures
  differed from Pillow; all nine are now byte-identical.
- `[bug fix]` **A module-level variable assigned from inside a function silently stayed null.**
  `jpegdct.b`'s `_tables()` set `$_CRR` and three siblings, which bound function-locals and left the
  module variables null, so every colour transform called `nd_take(null, …)`. The cure is the one
  `codecs/registry.b` already documents — module state lives in a dict — and it is now applied here
  (decision PW12's neighbourhood).
- `[found]` **A trailing comma in a Bantu list or dict literal is a syntax error, and inside an
  `include` it fails silently.** `jpeg_tables.b`'s `$PRESETS` was generated with trailing commas;
  the include reported success and `$PRESETS` was an empty dict, so every named quality preset was
  missing with no diagnostic anywhere. The generator no longer emits them. The interpreter swallowing
  a parse error in an included file is a defect in its own right and is recorded here because the
  failure mode — a value that is quietly empty — is the worst available one.
- `[patch]` **`docs/pillow-architecture.md` §6.3 and decision PW9 claimed `draft()` skips the
  entropy scan.** It does not: scaling happens in the IDCT and the scan is parsed in full at every
  scale. Measured, 1/8 draft saves 6–9 %, not 90 %. Both are corrected in place rather than quietly
  dropped, because the reason the claim was wrong is instructive — it assumed a format property
  (that a resolution pyramid is skippable) that JPEG does not have.

---

## Phase P4 — PNG, BMP, TGA ✅ (2026-10-28)

**131 checks green.** PngSuite — the canonical 176-file PNG conformance corpus — decodes **162 of
162** readable files pixel-identically to Pillow, with all 14 intentionally-corrupt files refused;
**238 images written by Pillow itself** at every option combination read back byte-identically (PNG
147, BMP 28, TGA 63); and Pillow reads 11 files this codec wrote, all pixels identical.

- `[feature]` **`pillow/codecs/png.b`** — colour types 0/2/3/4/6, bit depths 1/2/4/8/16, Adam7
  interlace, all five scanline filters, `PLTE tRNS gAMA pHYs sRGB bKGD tEXt zTXt iTXt`, and **every
  chunk CRC verified**. Writes 1, L, LA, P, RGB, RGBA and I;16 with a chosen filter and compression
  level. The two Pillow quirks are matched: a 16-bit truecolour PNG comes back as **8-bit RGB** (the
  low byte of every sample is dropped), and 16-bit grey+alpha comes back as **RGBA**, not LA.
- `[feature]` **`pillow/codecs/bmp.b`** — 1/4/8/16/24/32-bit, `BITMAPCOREHEADER`,
  `BITMAPINFOHEADER` and the V2–V5 extensions, uncompressed, RLE4, RLE8 and BITFIELDS, bottom-up and
  top-down. Reports a grey-ramp-palette BMP as mode **L** and a black-and-white 1-bit one as **1**,
  as Pillow does, rather than leaving a palette the image does not need.
- `[feature]` **`pillow/codecs/tga.b`** — types 1/2/3 and their RLE forms, 1/8/15/16/24/32-bit, both
  origins, colour maps. A TGA has **no magic number**, so its sniffer validates the whole 18-byte
  header for self-consistency and is tried after every format that has one. Depth 1 is not in the TGA
  specification but Pillow writes mode `1` that way, so it is read and written.
- `[feature]` **`tests/fixtures/pillow/pngsuite/`** — PngSuite checked in (176 files, with Pillow's
  decoded pixels beside each and a README explaining what each prefix exercises). The `x*` files are
  the valuable ones: a decoder that reads them is not lenient, it is wrong.
- `[bug fix]` **Bantu had no way to write a non-printable byte in a string literal, and silently
  dropped the backslash on any unrecognised escape.** `"\x89PNG"` became `x89PNG`; `"C:\path"`
  became `C:path`. A byte-oriented language with binary-safe strings could not spell a single file
  signature — a PNG's `\x89PNG`, a gzip member's `\x1f\x8b`, a JPEG's `\xff\xd8`. `\xNN` now
  works, and an unknown escape keeps its backslash instead of losing data with no diagnostic.
- `[patch]` **Every PNG chunk CRC is verified, including IDAT's.** Pillow checks metadata chunks and
  not IDAT, so it accepts PngSuite's `xcsn0g01.png`. A checksum mismatch means the data cannot be
  trusted; `png.allow_truncated(true)` is the opt-out and recovers Pillow's exact pixels.

---

## Phase P3 — Geometry: resampling, transpose, rotate, reduce, thumbnail ✅ (2026-10-28)

**140 checks green.** All **378** `resize` reference images byte-identical to Pillow (9 modes × 7
shape pairs × 6 filters), plus 49/49 transpose, 45/45 transform, 32/32 rotate, 30/30 thumbnail and
84/84 `reduce` for every integer mode.

- `[feature]` **`pillow/resample.b`** — the six filters, two separable passes, Pillow's
  `precompute_coeffs` with its `filterscale` clamp and 22-bit fixed-point weights. Vectorised as
  **one array operation per tap**: 5 for bicubic at scale 1, 17 for a 4× box reduction, regardless of
  the pixel count.
- `[feature]` **`pillow/transform.b`** — `transpose` (all eight, each an axis permutation so all are
  free and exact by construction), `reduce` with its three numeric paths, and `transform` by AFFINE,
  EXTENT, PERSPECTIVE, QUAD and MESH. The per-pixel samplers vectorise through a **flat gather**:
  NEAREST is 1 gather, BILINEAR 4, BICUBIC 16.
- `[feature]` **`pillow/path.b`** — `ImagePath.Path` with `getbbox`, `tolist`, `transform`, `map` and
  `compact` (city-block distance, as Pillow uses).
- `[feature]` `resize`, `rotate`, `transpose`, `reduce`, `thumbnail` and `transform` on the image,
  with the behaviours Pillow's Python layer adds on top of its C: modes `1` and `P` force NEAREST,
  `LA`/`RGBA` are premultiplied before any interpolating operation, and `thumbnail` uses Pillow's
  `round_aspect` rather than a plain round.
- `[bug fix]` **The resize PASS ORDER is part of the answer, not just of the speed.** Pillow 13
  chooses horizontal-vs-vertical first by cost, and because each pass rounds to the mode's precision
  the order changes the result. Every single pass matched Pillow exactly; only the combination did
  not, which is how this was found.
- `[bug fix]` **NEAREST accumulates its source coordinate** (`xo += scale`) rather than computing it.
  Repeated addition drifts: where the closed form lands on 5.999999999999999 the accumulation lands
  on 6.0000000000001 — one whole source row different.
- `[bug fix]` **An interpolated sample is stored with a C cast, which truncates.** Rounding instead
  differs on about half of all interpolated pixels.
- `[bug fix]` **`Geometry.c`'s `FLOOR` macro is plain `floor()` in both branches**, so the 16.16
  fixed-point `FIX(v)` rounds down, not toward zero. Truncating moved one pixel of a rotated image to
  the wrong source column.
- `[bug fix]` **`EXTENT` is rewritten to an AFFINE before dispatch**, as Pillow's Python layer does,
  so it inherits the integer NEAREST path. Treating it as its own double map differed on 6 of 60
  pixels.
- `[patch]` **`reduce`'s 32-bit path sums each 2×2 group in float32, not in double.** The C writes
  `double ss = 0; ss += a + b + c + d`, and for a `FLOAT32` image every operand of that inner sum is
  a float, so the four values are added in float32 and only the group total is widened. Reproducing
  the grouping took mode F from 85 to 95 of 105 exact; the residual 10 are Pillow's eleven
  hand-unrolled per-factor kernels, each differing by a single ULP, and that is documented and
  bounded rather than chased.
- `[patch]` **An out-of-range transform coordinate takes the fill colour, not a clamped edge pixel**,
  which is what stops a rotated image growing a smeared border. And a negative bicubic undershoot
  CLAMPS at zero rather than wrapping as the C cast's undefined behaviour does on x86 — turning −5
  into 251, a bright speck in a dark area.

---

## Phase P2 — The core: modes, the image object, conversion, colour `[~]` (2026-10-27)

**249 feature checks green**, 111 of 111 verifiable conversion pairs byte-identical to Pillow, and a
live comparison against Pillow 13.0.0 in which all 391 pixels of a written-and-reread image match.
Two items remain open (palette quantisation, and bit-exactness for YCbCr/HSV/LAB) and are listed in
the roadmap rather than papered over.

- `[feature]` **`pillow/mode.b`** — the 20-mode table, matched against Pillow's own `ImageMode` and
  `Image.new`. Exposes Pillow's `basemode`/`basetype` faithfully *and* our own `family()`, because
  `basemode` calls RGB, CMYK, YCbCr, LAB and HSV all "RGB" and the conversion matrix must tell them
  apart.
- `[feature]` **`pillow/pxbuf.b`** — the array layer every other module builds on: band views (free),
  band assembly, broadcast splat (free), clamp, crop views (free), LUT gather, float32 quantisation,
  and the 0-d reduction unwrap. No loop in it is proportional to the pixel count.
- `[feature]` **`pillow/color.b`** — `ImageColor`: 148 CSS names, `#rgb`/`#rgba`/`#rrggbb`/
  `#rrggbbaa`, `rgb()`, `rgb(%)`, `rgba()`, `hsl()`, `hsv()`/`hsb()`, with Python's `colorsys`
  ported exactly. The table is generated from the Pillow reference, not typed from memory.
- `[feature]` **`pillow/convert.b`** — the conversion matrix, on buffers rather than Images so the
  array layer never depends on the object layer.
- `[feature]` **`pillow/image.b`** — `PillowImage` and the `Image.*` namespace: `new frombytes
  frombuffer fromarray merge blend composite alpha_composite eval`, and on the instance `copy crop
  paste split getchannel putalpha getpixel putpixel getdata putdata point histogram getbbox
  getextrema getcolors convert tobytes save show`.
- `[feature]` **`pillow/codecs/registry.b`** — the plugin registry: magic-byte sniffing first and
  extension second (a PNG named `.jpg` is a PNG), `identify`, `registered_extensions`, and Pillow's
  `UnidentifiedImageError` by name.
- `[feature]` **`pillow/codecs/ppm.b`** — all seven Netpbm formats (P1–P6), 8- and 16-bit, read and
  write. Chosen first because it needs no compression, so the whole path from file bytes to pixel
  buffer and back was testable on day one.
- `[feature]` **`pillow/pillow.b`** — the umbrella, with Pillow's spellings `pillow.new`,
  `pillow.open` and `pillow.eval` preserved by binding anonymous functions to `$new`/`$open`/`$eval`:
  `def new(...)` cannot parse (`new` is a keyword, `open` a builtin) but a member access can.
- `[feature]` **`tools/gen_pillow_reference.py`** and `tests/fixtures/pillow/reference/` — the
  differential backbone: 5,387 lines of recorded Pillow behaviour (the mode table, 148 colours, 29
  colour specs, 120 `getcolor` results, 2,744 `RGB→L` values and the 2,292-row conversion matrix),
  checked in so the suite runs on a machine with no Python.
- `[feature]` Tests: `tests/pillow_core_test.b` (168), `tests/pillow_convert_test.b` (43),
  `tests/pillow_color_test.b` (38), `tests/pillow_differential.sh` (live against Pillow).

**Seven facts the design had wrong, each found by the differential tier and each now pinned by a
named test:**

- `[bug fix]` **`RGB → L` is not `(R×299 + G×587 + B×114)/1000`.** It is Pillow's `L24` macro,
  `(R×19595 + G×38470 + B×7471 + 32768) >> 16`, which *rounds*. The truncating form every BT.601
  reference prints disagrees with Pillow on **1,101 of 2,197** sampled triples. The corrected formula
  is verified over **all 16,777,216** RGB triples: zero mismatches.
- `[bug fix]` **`Image.new("RGBA", …)` fills alpha with 0, not 255** — a new RGBA image is
  transparent black. Guessing opaque is the obvious mistake and would differ from Pillow on every
  new RGBA image.
- `[bug fix]` **`convert("1")` dithers by default.** Pillow's default is Floyd–Steinberg, not a
  threshold; the design said the opposite.
- `[bug fix]` **The two bilevel thresholds differ on purpose:** `dither=NONE` maps `v ≥ 128` to
  white, the dithered path maps `v > 128`. With them swapped, a flat 50% grey dithers to the inverse
  checkerboard.
- `[bug fix]` **The dithered error is carried as an integer truncated toward zero**, not as a float.
  In floating point a whole row drifts.
- `[bug fix]` **`RGB → RGBX` pads the fourth band with 255, not 0.**
- `[bug fix]` **grey → CMYK does not go through RGB.** It goes to the black plate: `C=M=Y=0`,
  `K = 255 − L`. Routing it through RGB got 44 of 48 probe values wrong.
- `[patch]` `getcolor(c, "CMYK")` returns **three** values, not four, because Pillow's `getcolor`
  switches on `basemode` and is a colour *parser*, not a colour-space conversion. Matched, and
  asserted by a named test so it is not "fixed" later.

**Interpreter-level work this phase needed:**

- `[bug fix]` **`indexOf` silently returned −1 for any list.** `contains($L, $x)` answered true while
  `indexOf($L, $x)` answered −1 for the same pair — a wrong answer rather than an error, and the
  opposite of what the name promises. `indexOf` is now the position-returning twin of `contains`,
  using the same element equality. 15 checks added to `tests/lang_list_test.b` (89 green).

---

## Phase P1 — The two atoms, and the Bantu twin that checks them ✅ (2026-10-27)

All five tiers green: **426 feature checks**, 40 read-side and 66 write-side differential vectors
against Python's zlib, 3,596 fuzz candidates, 38 regression suites, ASan + UBSan clean.

- `[feature]` **`inflate(src, maxOut?)` and `gunzip(src, maxOut?)`** — `bantu-src/compiler/src/inflate.hpp`,
  registered in `evaluator.hpp`. DEFLATE (RFC 1951), zlib (RFC 1950) and gzip (RFC 1952); the
  container is sniffed. `maxOut` is checked **before** each append, so an inflate bomb is refused
  rather than allocated. `src` is a byte string or byte-list; the result is a byte string, because
  that costs one byte per byte where a Bantu list costs ~100.
- `[feature]` **`nd_frombytes(src, elem?, shape?)` and `nd_tobytes(array, elem?)`** —
  `ndarray_native.cpp`. `u8 i8 u16 i16 u32 i32 f32 f64`, each with an explicit `le`/`be` suffix,
  because every format that matters specifies a byte order and the machine's is not it. A view is
  written in **logical** order, so a flipped or transposed array round-trips to what it looks like.
  Integer narrowing **wraps**, as a cast does and as NumPy's `astype` does; a caller wanting a clamp
  calls `nd_clip` first, which is explicit and free.
- `[feature]` `has_native("inflate")` and `has_native("ndbytes")`.
- `[feature]` **`pillow/inflate.b`** — DEFLATE, zlib and gzip decoding in Bantu. Ships alongside the
  native atom on purpose: it differential-tests it on every run, and it is the fallback, so the
  package is complete in Bantu alone on any build. Measured at **3.1 µs per output byte** on the
  dynamic-Huffman path PNG uses.
- `[feature]` **`pillow/deflate.b`** — DEFLATE, zlib and gzip *encoding* in Bantu: stored blocks at
  level 0, fixed-Huffman with LZ77 and lazy matching at 1–9, plus CRC-32 and Adler-32. No dynamic
  Huffman table construction, so its output is ~38% larger than `bp_zlib`'s and is still an ordinary
  zlib stream — verified by having Python's `zlib` read 66 of them.
- `[feature]` **`pillow/bitio.b`** — one bit reader and one bit writer, parameterised by bit order
  (`lsb` for DEFLATE/PNG/GIF, `msb` for JPEG/TIFF), so the two orders cannot drift apart. The hot
  decoders inline their own accumulator, since a Bantu function call is ~1 µs; this module is the
  readable reference they are tested against.
- `[bug fix]` **The pure-Bantu canonical Huffman walk shifted `code` twice per iteration** — once
  folded into the bit append at the top, again in the `else` branch. zlib's `puff.c` shifts once.
  Invisible across 35 corpus vectors because the 9-bit fast table answers almost every symbol; the
  first stream with a **distance** code longer than 9 bits failed immediately. Found by the
  differential corpus, which is what it is for.
- `[patch]` `nd_frombytes`/`nd_tobytes` now **reject an explicitly-passed empty element type**
  instead of silently defaulting to `u8`. The `u8` default is supplied at the call site when the
  argument is absent or null, so an empty string means a caller built the name and got it wrong.
- `[feature]` Tests: `tests/pillow_inflate_test.b`, `tests/pillow_deflate_test.b`,
  `tests/pillow_fuzz_test.b`, `tests/numba_bytes_test.b`,
  `tests/pillow_deflate_differential.sh`, and the checked-in corpus
  `tests/fixtures/pillow/deflate/` (8 shapes × 4 levels + gzip).

---

## Phase P0 — Design of record ✅ (2026-10-27)

- `[feature]` `docs/pillow-architecture.md` — the design, written before the code: the load-bearing
  measurements, the `(h, w, bands)` pixel store, the 20-mode table, the two interpreter atoms, the
  codec table, the security model, the five-tier test gate, and the nine phases.
- `[feature]` `pillow-suite/ROADMAP.md`, `DECISIONS.md` (PW1–PW10), `CHANGELOG.md`.
- `[found]` **`bp_zlib` compresses and nothing in the interpreter decompresses.** `grep -ri inflate
  bantu-src/compiler/src/` finds the word only in a GC comment. Bantu can *write* a PNG, a gzip body
  or a ZIP member and cannot *read* one back — a one-way primitive. Fix in P1 as `inflate`/`gunzip`.
- `[found]` **No byte buffer can reach an ndarray.** `nd_fromfile`'s dtypes are `f64`/`i64`/`bool`,
  all 8 bytes wide. The only route is `readfile → bytes() → nd(list)`, measured at **479 ms and a
  ~100 MB transient per megabyte**. Fix in P1 as `nd_frombytes`/`nd_tobytes`.
- `[patch]` Measurements recorded for the design (Darwin 24.6.0, Apple silicon, `bantu` at `68d0174`,
  `-O2`), and the finding that decided the architecture: an interpreted per-pixel pass over a 1080p
  RGBA image costs **4.1 s**, while the same operation through `nd_*` costs **0–150 ms**.
