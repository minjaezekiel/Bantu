# pillow — Changelog

Entries are tagged `[feature]` / `[bug fix]` / `[patch]` / `[found]`, newest first. A phase appears
here only once all five tiers of its gate are green.

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
