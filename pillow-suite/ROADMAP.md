# pillow — Roadmap

Pillow's API, in Bantu. Every line of the package is Bantu; the interpreter supplies two generic byte
atoms (P1), the same arrangement `hash` has with `band` and `numba` has with `nd_*`. Design and
rationale: [`docs/pillow-architecture.md`](../docs/pillow-architecture.md) and
[`DECISIONS.md`](DECISIONS.md).

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done (feature **and** stress tests green)

**Every phase is gated by five tiers, all green before it advances:**

| tier | proves | form |
|---|---|---|
| 1. feature | every call does what it claims, including its failure modes | `tests/pillow_*_test.b`, ending `RESULT: ALL GREEN` |
| 2. differential | the output is *right* — byte-identical to Python Pillow | `tests/pillow_differential.sh`, or checked-in goldens generated from Pillow |
| 3. stress | survives size, repetition, abuse — and the **timing curve is asserted linear** | scale · degenerate · hostile/fuzz · memory |
| 4. regression | nothing else broke | the whole existing suite on the same build |
| 5. sanitizers | no latent memory defect | ASan + UBSan over P1's two native atoms |

**Five rules that apply to every phase.**
1. **No loop proportional to the pixel count.** Loops over bands (≤4), kernel taps (≤25), palette
   entries (≤256) or scanlines are fine. A pixel loop is a defect; tier 3's timing curve catches it.
2. Never use the C-style `for` above 100,000 iterations — `evaluator.hpp` caps it at
   `safety < 100000` and exits **silently**. Use `while`. Everywhere.
3. Never build output with `$s = $s + …` — it is O(n²). Push onto a list, `join` once.
4. Never assert on a stringified number — `str()` gives six significant digits.
5. Every codec read is bounds-checked against the buffer length before it happens, and every failure
   is a catchable Bantu error naming the format and the byte offset. No read ever runs off the end.

---

## Phase P0 — Design of record ✅

- [x] `docs/pillow-architecture.md` — the load-bearing measurements, the pixel store, the mode table,
      the two atoms, the codec table, the security model, the five-tier gate
- [x] `pillow-suite/ROADMAP.md`, `DECISIONS.md`, `CHANGELOG.md`
- [x] `pillow/package.json`

**Two defects found in the interpreter during design**, both general, both recorded for P1:

| found | why it is a defect | fix |
|---|---|---|
| **`bp_zlib` compresses and nothing decompresses** | Bantu can *write* a PNG, a gzip body or a ZIP member and cannot *read* one back. A one-way primitive. `grep -ri inflate bantu-src/` finds only a GC comment | `inflate(s[,maxOut])` / `gunzip(s[,maxOut])` |
| **no byte buffer can reach an ndarray** | `nd_fromfile`'s dtypes are `f64`/`i64`/`bool`, all 8 bytes wide. The only route is `readfile → bytes() → nd(list)`: **479 ms and a ~100 MB transient per megabyte** | `nd_frombytes(s, elem?, shape?)` / `nd_tobytes(a, elem?)` with `u8 u16be u16le i32 f32` |

---

## Phase P1 — The two atoms, and the Bantu twin that checks them ✅

- [x] `inflate(src, maxOut?)` — container sniffed (gzip / zlib / raw DEFLATE); `maxOut` refuses an
      inflate bomb before the memory is committed
- [x] `gunzip(src, maxOut?)`
- [x] `nd_frombytes(src, elem?, shape?)` / `nd_tobytes(array, elem?)` — `u8 i8 u16 i16 u32 i32 f32
      f64`, each with an explicit `le`/`be` suffix
- [x] `pillow/inflate.b` — DEFLATE decoder **in Bantu**: stored, fixed and dynamic blocks, the full
      length/distance alphabets, back-references into the 32 KB window, zlib and gzip containers,
      Adler-32 verified
- [x] `pillow/deflate.b` — DEFLATE encoder **in Bantu**: stored (level 0) and fixed-Huffman with
      LZ77 and lazy matching (levels 1–9), zlib and gzip containers, CRC-32 and Adler-32 in Bantu
- [x] `pillow/bitio.b` — one bit reader / writer, both bit orders (`lsb` DEFLATE, `msb` JPEG)
- [x] `has_native("inflate")` / `has_native("ndbytes")` routing

| gate | result |
|---|---|
| feature | ✅ `tests/pillow_inflate_test.b` **131**, `tests/pillow_deflate_test.b` **237**, `tests/numba_bytes_test.b` **58** — all ALL GREEN |
| differential | ✅ **40 vectors from Python's `zlib`/`gzip`** at levels 0, 1, 6, 9 (`tests/fixtures/pillow/deflate`) decode byte-identically through *both* decoders |
| differential | ✅ the write side too: **66 streams** written by `pillow/deflate.b` at every level, read back correctly by Python's `zlib` and `gzip` (`tests/pillow_deflate_differential.sh`) |
| differential | ✅ every length 0–300 round-trips through `bp_zlib` → both decoders; every length 0–200 × all 10 levels through `deflate.b` → both decoders |
| differential | ✅ `deflate.b`'s `crc32`/`adler32` equal `bp_crc32`/`bp_adler32` over all 256 byte values |
| stress | ✅ `tests/pillow_fuzz_test.b` — **3,596 hostile candidates**: random bytes, valid headers over noise, all 256 block headers, 1,200 single-byte and 400 multi-byte mutations, truncation at every length, trailing garbage, hostile gzip extras, declared lengths at every magnitude. 3,583 rejected with catchable errors, **0 crashes, 0 hangs, 0 disagreements between the two decoders** |
| stress | ✅ an inflate bomb (975:1) refused at every `maxOut` boundary, including exactly-right and one-byte-under |
| regression | ✅ **38 existing suites** green on the same build (lang, numba, arctic, bplot, crypto, hash, uuid, random, orm) |
| sanitizers | ✅ ASan + UBSan over `inflate.hpp` and `nd_frombytes`/`nd_tobytes`: all three suites green, no reports |

**Measured** (the figures `docs/pillow-architecture.md` §4.1 cites):

| | pure Bantu | native |
|---|---|---|
| inflate, dynamic Huffman (PNG's path) | 3.1 µs/byte | ~0.008 µs/byte |
| inflate, fixed Huffman | 3.6 µs/byte | — |
| inflate, stored | 5.6 µs/byte | — |
| `nd_frombytes`, 1 MiB | — | **6 ms** (was 479 ms via `bytes()` + `nd()`) |
| `nd_frombytes`, 8 MiB shaped | — | **53 ms** |

`deflate.b`'s fixed-Huffman output is ~38% larger than `bp_zlib`'s dynamic Huffman on text, which is
the documented trade (decision PW3) and still an ordinary zlib stream every decoder reads.

**One defect found and fixed in the pure-Bantu decoder**, recorded because of how it hid:
the canonical Huffman walk shifted `code` twice per iteration — once folded into the bit append at
the top, again in the `else`. zlib's `puff.c` shifts once. It was invisible across 35 corpus vectors
because the 9-bit fast table answers almost every symbol; the first stream containing a **distance**
code longer than 9 bits found it instantly. That is the differential corpus earning its place, and
the fix carries a comment saying so.

---

## Phase P2 — The core: modes, the image object, conversion, colour ✅

**Done and gated** (feature + differential + stress green; the two open items are listed after):

- [x] `pillow/mode.b` — the 20-mode table, **matched against Pillow's own** `ImageMode.getmode` and
      `Image.new`: bands, band names (including the lowercase `a` of `La`/`RGBa`), `basemode`,
      `basetype` and the default fill. Plus our own `family()`, because Pillow's `basemode` calls
      RGB, CMYK, YCbCr, LAB and HSV all "RGB" and the conversion matrix has to tell them apart
- [x] `pillow/pxbuf.b` — the array layer: band views, band assembly, broadcast splat, clamp, crop
      views, LUT gather, float32 quantisation, 0-d reduction unwrap. Nothing here loops over pixels
- [x] `pillow/color.b` — `getrgb`, `getcolor`, `colormap`: the 148 CSS names, `#rgb`, `#rgba`,
      `#rrggbb`, `#rrggbbaa`, `rgb()`, `rgb(%)`, `rgba()`, `hsl()`, `hsv()`/`hsb()`, with
      `colorsys` ported exactly
- [x] `pillow/convert.b` — the matrix, **177 of 191 pairs exact against Pillow**
- [x] `pillow/ycbcr_tables.b` + `tools/gen_ycbcr_tables.py` — Pillow's own 13 YCbCr lookup tables,
      generated from `ConvertYCbCr.c` rather than fitted
- [x] `pillow/palette.b` — `ImagePalette`, the web colour cube, the grey ramp, nearest-colour
      mapping (closed form for the cube, general search for a custom palette), Floyd–Steinberg into
      a palette
- [x] `pillow/image.b` — `PillowImage` plus `new frombytes frombuffer fromarray merge blend
      composite alpha_composite eval`; `mode size width height info palette getbands load close`;
      `copy crop paste split getchannel putalpha getpixel putpixel getdata putdata point histogram
      getbbox getextrema getcolors convert tobytes save show`
- [x] `pillow/codecs/registry.b` — `register`, magic-byte sniffing then extension, `identify`,
      `UnidentifiedImageError`, `registered_extensions`
- [x] `pillow/codecs/ppm.b` — all seven Netpbm formats (P1–P6), 8- and 16-bit, read and write
- [x] `pillow/pillow.b` — the umbrella, with Pillow's spellings `pillow.new/open/eval` restored by
      binding anonymous functions to `$new`/`$open`/`$eval` (a `def new(...)` cannot parse: `new` is
      a keyword and `open` a builtin, but a member access spells them fine)

| gate | result |
|---|---|
| feature | ✅ `tests/pillow_core_test.b` **168**, `tests/pillow_convert_test.b` **93**, `tests/pillow_color_test.b` **38** — all ALL GREEN |
| differential | ✅ all **20 modes** against `tests/fixtures/pillow/reference/modes.txt`: bands, names, basemode, basetype, default fill |
| differential | ✅ all **148 named colours**, **29 colour-spec forms** and **120 `getcolor` results** across every mode |
| differential | ✅ **111 of 111** verifiable conversion pairs byte-identical to Pillow; `RGB → L` additionally over **all 16,777,216 RGB triples** |
| differential | ✅ **live, against Pillow itself** (`tests/pillow_differential.sh`): a 23×17 RGB pattern written by `pillow` and read by Pillow — **all 391 pixels identical**; `convert("L")` and `convert("1", NONE)` equal Pillow's pixel for pixel; a 9-wide 1-bit image (row padding) and a 16-bit greymap (big-endian wire) both exact |
| stress | ✅ 0×0, 1×1, 1×1000, 1000×1; `MAX_IMAGE_PIXELS` enforced before allocation (a 60000×60000 PPM header refused); paste at every boundary and fully outside; crop fully outside |
| stress | ✅ hostile PPM: 12 malformed headers, every truncation of a valid file, every single-byte corruption — **no crash, no hang**, every failure catchable |
| stress | ✅ **rule PW2 asserted**: 4× the pixels costs 3× the time (22,500 px → 7 ms, 90,000 px → 21 ms), and a crop of a 480,000-pixel image is 19 ms |
| regression | ✅ **56 of 56** `.b` suites green (the binary is now built with `BANTU_SODIUM=1`, so the privacy suite runs too) |

**Verified facts this phase recovered, each of which the design had wrong:**

| the design said | Pillow actually does | how it was caught |
|---|---|---|
| `L = (R×299 + G×587 + B×114)/1000`, truncated | `L = (R×19595 + G×38470 + B×7471 + 32768) >> 16` — the `L24` macro, which **rounds** | the truncating form differs on **1,101 of 2,197** sampled triples |
| `Image.new("RGBA", …)` fills alpha opaque | fills **zero**, so a new RGBA image is transparent black | the mode reference's default-fill column |
| `convert("1")` is not dithered by default | **is** Floyd–Steinberg by default | `RGB → 1` gave results no threshold could explain |
| the dithered and plain thresholds are the same | dithered is `v > 128`, `dither=NONE` is `v ≥ 128` | a flat 50% grey dithered to the inverse checkerboard |
| `RGB → RGBX` pads with 0 | pads with **255** | 12 of 12 probe values wrong |
| grey → CMYK goes through RGB | goes straight to the black plate: `K = 255 − L`, `C=M=Y=0` | 44 of 48 probe values wrong |
| `getcolor(c, "CMYK")` returns 4 values | returns **3** — it switches on `basemode`, and is not a colour-space conversion | the `getcolor` reference |

### Still open, and moved to the phase that owns it

- [ ] **`quantize()`** — an adaptive palette by median cut / fast octree, which is what Pillow uses
      for `RGBA → P` (4 matrix pairs). `convert()` raises an error naming it rather than returning
      fixed-cube indices that would look almost right. Moved to **P6**, with `remap_palette` and
      `putpalette`.
- [ ] **`LAB` bit-exactness** is not open work — it is declined. Pillow has no LAB row in
      `Convert.c`; `Image.convert` routes LAB through **ImageCms (littleCMS)**, and matching an ICC
      engine bit for bit is the problem decision PW10 declines. `pillow` ships a correct standard
      D65 sRGB ↔ L\*a\*b\* conversion, `convert.unverified()` says so, and
      `convert.why_unverified()` explains it. Closing it would mean shipping littleCMS.
- [ ] Instance methods belonging to later phases are *not* stubbed: `resize`, `rotate`, `transpose`,
      `thumbnail`, `reduce`, `transform`, `filter`, `quantize`, `seek`, `tell`, `entropy`, `verify`,
      `effect_spread`, `draft`, `remap_palette`, `apply_transparency`

### What reading Pillow's C source settled

Three conversions were fitted against Pillow as a black box and reached ±1 on 8–25% of inputs. None
would have closed by further fitting, because in each case the *structure* differed — recorded here
because the lesson generalises to every remaining codec phase:

| | the structure, not the constants |
|---|---|
| **YCbCr** | sums three **pre-rounded** table lookups at 6 fractional bits; the ±1 was the discarded per-channel rounding residues |
| **HSV** | computes in C `float` (24-bit), and folds the hue as `fmod(h/6 + 1, 1)` — adding one *before* the fold |
| **the dither** | carries the error **undivided**, dividing by 16 once at the point of use, and uses the truncated `/1000` luminance where `convert("L")` rounds |

**Rule for the remaining phases: read the source before fitting.** `tools/` now holds two generators
that pull constants straight from Pillow (`gen_ycbcr_tables.py`, `gen_pillow_reference.py`).

### Deliberate deviations (decision PW11)

Two upstream defects are **fixed, not reproduced**, each stated in its module header with a test
asserting the containment of the difference:

- `Convert.c`'s dithered `topalette` writes blue's error accumulators into all three channel slots at
  each row's end, discarding red's and green's. Fixed; row 0 is identical to Pillow and the
  difference is confined to ≤4 columns from the right edge.
- `ImagingConvertRGB2YCbCr` casts with `(UINT8)`, wrapping an out-of-range input. Clamped instead.

### Measured

| operation, 120,000-pixel RGB | cost |
|---|---|
| `convert("L")` | 7 ms |
| `convert("P", dither="NONE")` | 34 ms |
| `convert("YCbCr")` | 32 ms |
| `convert("HSV")` | 40 ms |
| `convert("1")` dithered | 766 ms |
| `convert("P")` dithered | 2,260 ms |

The two dithered paths are the documented exception to rule PW2: error diffusion is sequential by
nature, so they are the one place the package loops over pixels. Both hot loops are inlined (no
per-pixel function calls, no `nd_get`), which bought 13% and 26% respectively; `dither="NONE"` is the
vectorised alternative and is ~70× faster. **If dithered output at photographic sizes turns out to be
something users do, this is where a generic native atom would pay** — the same trigger PW9 records
for JPEG's entropy scan, and not added speculatively.

---

## Phase P3 — Geometry: the six resampling filters

- [ ] `pillow/resample.b` — `NEAREST BOX BILINEAR HAMMING BICUBIC LANCZOS`, two separable passes,
      Pillow's `precompute_coeffs` including `filterscale` and 1/2048 fixed-point weights
- [ ] `resize rotate transpose thumbnail reduce transform` on `PillowImage`
- [ ] `pillow/transform.b` — `AffineTransform ExtentTransform QuadTransform MeshTransform`
- [ ] `pillow/path.b` — `ImagePath.Path`

| gate | form |
|---|---|
| differential | every filter × every mode × upscale/downscale/identity, byte-identical to Pillow |
| differential | the 8 `transpose` operations and `rotate` at 90/180/270 exact; `expand=True` boxes match |
| stress | 1 px → 10,000 px in one call and back; a 1×10,000 strip; `reducing_gap`; **a bicubic 1080p resize is ≤ 20 array operations**, asserted by instrumenting the call count |

---

## Phase P4 — PNG, BMP, TGA

- [ ] `pillow/codecs/png.b` — colour types 0/2/3/4/6, bit depths 1/2/4/8/16, Adam7 interlace, the 5
      scanline filters (loop over **rows**), `tRNS gAMA pHYs iCCP tEXt zTXt iTXt sRGB bKGD`, APNG
      frames, CRC-32 checked, write with filter selection and `compress_level`
- [ ] `pillow/codecs/bmp.b` — 1/4/8/16/24/32-bit, RLE4/RLE8, `BITFIELDS`, both row orders
- [ ] `pillow/codecs/tga.b` — 8/16/24/32-bit, RLE, all four origins

| gate | form |
|---|---|
| differential | **PngSuite** (Willem van Schaik's 175-file conformance corpus) decoded and compared pixel for pixel; every file Pillow accepts, accepted identically; every file Pillow rejects, rejected |
| differential | written PNGs re-read by Pillow and by `png.b` to the same pixels |
| stress | the fuzz tier of architecture §8 over all three formats; a 60,000×60,000 header refused before allocation; an inflate bomb refused |

---

## Phase P5 — JPEG and EXIF

- [ ] `pillow/codecs/jpeg.b` — baseline **and progressive**; all subsamplings (4:4:4, 4:2:2, 4:2:0,
      4:1:1); restart markers; the array-side pipeline (dequantise → IDCT by two `nd_matmul` over all
      blocks at once → upsample → YCbCr→RGB); CMYK and Adobe APP14 inversion; `draft()`
- [ ] JPEG encode — quality 1–100 with Pillow's quantisation tables, `optimize`, `progressive`,
      `subsampling`, `qtables`, `restart_marker_blocks`
- [ ] `pillow/exif.b` — `ExifTags.TAGS/GPSTAGS/IFD/Base/GPS/Interop`, read and write, and
      `ImageOps.exif_transpose`

| gate | form |
|---|---|
| differential | a corpus spanning every subsampling and both scan types, decoded pixel-identical to Pillow (libjpeg-turbo's IDCT is the reference; a ±1 tolerance is documented where the IDCT is not bit-specified) |
| differential | encode compared to Pillow at a **PSNR floor**, since quantisation choices are the encoder's |
| stress | truncated scans; a restart marker in every position; 2^16-wide; `draft` at 1/2, 1/4, 1/8 asserted ≤ 1 s on 12 MP |

---

## Phase P6 — The operation surface

- [ ] `pillow/filter.b` — `BLUR CONTOUR DETAIL EDGE_ENHANCE EDGE_ENHANCE_MORE EMBOSS FIND_EDGES
      SHARPEN SMOOTH SMOOTH_MORE`, `GaussianBlur BoxBlur UnsharpMask Kernel`, `RankFilter
      MedianFilter MinFilter MaxFilter ModeFilter`, `Color3DLUT`
- [ ] `pillow/chops.b` — all 24 channel operations
- [ ] `pillow/enhance.b` — `Color Contrast Brightness Sharpness`
- [ ] `pillow/ops.b` — `autocontrast colorize contain cover crop deform equalize expand fit flip
      grayscale invert mirror pad posterize scale solarize exif_transpose`
- [ ] `pillow/stat.b`, `pillow/math.b`, `pillow/morph.b`

| gate | form |
|---|---|
| differential | every filter, every chop, every op × every applicable mode, byte-identical to Pillow |
| differential | `GaussianBlur` against Pillow's three-box-pass approximation, not against a true Gaussian — Pillow's answer is the contract |
| stress | a 25×25 kernel; a rank filter at 1080p within the linear curve; `ImageMath` fed a hostile expression |

---

## Phase P7 — Drawing and text

- [ ] `pillow/draw.b` — `Draw getdraw floodfill`; `arc bitmap chord ellipse line pieslice point
      polygon rectangle regular_polygon rounded_rectangle shape text multiline_text textbbox
      multiline_textbbox textlength`; `width=`, `joint="curve"`, `ImageDraw.Outline`
- [ ] `pillow/font.b` — `load_default` (embedded bitmap font), `truetype` (a **TrueType reader in
      Bantu**: `cmap` 0/4/6/12, `glyf`/`loca`, `hmtx`, `kern`/`GPOS`, composite glyphs, `fvar`),
      `load`/`load_path` for BDF/PCF; `getbbox getlength getmask getmask2 font_variant`; glyph cache
- [ ] `anchor spacing align direction stroke_width stroke_fill embedded_color`

| gate | form |
|---|---|
| differential | every primitive at every `width` compared to Pillow pixel for pixel; text with DejaVu at 8/12/24/72 px |
| stress | a 100,000-point polygon; a self-intersecting polygon; degenerate ellipses; a 2,000-character string; a hostile/truncated font file (fuzz tier) |

---

## Phase P8 — The rest of the formats

- [ ] `pillow/codecs/gif.b` + `pillow/sequence.b` — LZW read/write, all frames, local palettes,
      transparency, interlace, the four disposal methods, `save_all` animation
- [ ] `pillow/codecs/tiff.b` — both byte orders, strips and tiles, `NONE PACKBITS LZW DEFLATE
      CCITT G3/G4`, all photometrics, multi-page
- [ ] `pillow/codecs/ico.b` (+CUR), `pcx.b` (+DCX), `xbm.b`, `xpm.b`
- [ ] `pillow/codecs/webp.b` — VP8L lossless read/write, VP8 still read
- [ ] PDF write; read-only long tail: `eps psd dds sgi sun im msp fits blp palm`

| gate | form |
|---|---|
| differential | per format, against Pillow; GIF animations frame by frame including disposal; `libtiff`'s own test corpus; `libwebp`'s conformance vectors |
| stress | the fuzz tier per format; a 1,000-frame GIF; a TIFF IFD chain pointing at itself; an ICO pointing at itself |

---

## Phase P9 — Ship

- [ ] `docs/pillow.md` — the full API reference, with the performance envelope published beside the
      calls that are slow (JPEG full scan, Floyd–Steinberg dither) rather than left to be discovered
- [ ] `samples/pillow/` — thumbnailer, watermarker, contact sheet, format converter
- [ ] registry publication; `README.md` and `CHANGELOG.md` entries
- [ ] **PW7 measurement**: a `u8` dtype in numba would cut the pixel store 8×. Measure the gain on a
      4000×3000 RGBA workload and decide; it is a separable change this package picks up with no API
      change
