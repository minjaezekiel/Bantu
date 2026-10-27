# pillow — Changelog

Entries are tagged `[feature]` / `[bug fix]` / `[patch]` / `[found]`, newest first. A phase appears
here only once all five tiers of its gate are green.

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
