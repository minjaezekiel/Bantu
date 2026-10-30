# Bantu `pillow`

Imaging for Bantu, in the shape of Python's [Pillow](https://pillow.readthedocs.io): the same names,
the same arguments, and — where it has been verified — the same bytes.

Written **entirely in Bantu**, over `numba`'s native arrays. The codecs, the conversion matrix, the
compression and the colour parsing are all `.b`; what the interpreter supplies is arithmetic, the
same arrangement [`hash`](hash.md) has with `band`/`rotr` and [`numba`](numba.md) has with `nd_*`.

- **Source:** [`pillow/`](../pillow/) · **Design:** [`pillow-architecture.md`](pillow-architecture.md)
- **Roadmap:** [`pillow-suite/ROADMAP.md`](../pillow-suite/ROADMAP.md) · **Decisions:** [`DECISIONS.md`](../pillow-suite/DECISIONS.md)

```bantu
include "./pillow/pillow.b" as pillow;

$im = pillow.new("RGB", [320, 240], "cornflowerblue");
$im.putpixel([0, 0], [255, 0, 0]);
$im.convert("L").save("grey.pgm", null);

$photo = pillow.open("photo.ppm");
print($photo.mode + " " + str($photo.size));
```

```sh
bantu run tests/pillow_core_test.b      # 168 checks → RESULT: ALL GREEN
bantu run tests/pillow_convert_test.b   #  43 checks
bantu run tests/pillow_color_test.b     #  38 checks
bantu run tests/pillow_inflate_test.b   # 131 checks
bantu run tests/pillow_deflate_test.b   # 237 checks
bantu run tests/pillow_fuzz_test.b      # 3,596 hostile inputs
tests/pillow_differential.sh            # live, against Python's Pillow
```

---

## ⚠ What is in this build, and what is not

`pillow` is being built in nine phases. **This is phase P6 of 9.** The table below is the honest
state; nothing is stubbed to look present.

| | status |
|---|---|
| **The Image object** — `new open copy crop paste split merge getchannel putalpha getpixel putpixel getdata putdata point histogram getbbox getextrema getcolors blend composite alpha_composite eval tobytes frombytes fromarray save show` | ✅ |
| **All 20 modes** — `1 L LA La P PA RGB RGBA RGBa RGBX CMYK YCbCr LAB HSV I I;16 I;16L I;16B I;16N F` | ✅ |
| **The conversion matrix** | ✅ **177 of 191 pairs byte-identical to Pillow** (below) |
| **`ImageColor`** — `getrgb getcolor colormap`, 148 CSS names | ✅ |
| **`ImagePalette`** — the web cube, the grey ramp, nearest-colour mapping | ✅ |
| **Geometry** — `resize rotate transpose thumbnail reduce transform`, the six resampling filters, `ImageTransform`, `ImagePath` | ✅ |
| **PNG** — all colour types and depths, interlaced, all five filters, CRC-verified | ✅ |
| **BMP** — 1/4/8/16/24/32-bit, RLE4, RLE8, BITFIELDS, both row orders | ✅ |
| **TGA** — types 1/2/3 and their RLE forms, both origins | ✅ |
| **PPM / PGM / PBM** (P1–P6), 8- and 16-bit, read + write | ✅ |
| **DEFLATE / zlib / gzip**, both directions | ✅ |
| **The plugin registry** — `register_codec`, sniffing, `registered_extensions` | ✅ |
| **JPEG** — baseline and progressive, every subsampling, restart markers, CMYK/YCCK, `draft`, read + write | ✅ **54 of 54 corpus files byte-identical to Pillow**, and the encoder byte-identical to libjpeg |
| **`ExifTags`** — `TAGS GPSTAGS IFD Base GPS Interop`, EXIF read + write, `exif_transpose` | ✅ |
| **`ImageFilter`** — the ten built-ins, `Kernel`, `BoxBlur`, `GaussianBlur`, `UnsharpMask`, `RankFilter`, `MedianFilter`, `MinFilter`, `MaxFilter`, `ModeFilter`, `Color3DLUT` | ✅ **190 results byte-identical** over seven modes |
| **`ImageChops`** — all 24 operations | ✅ **exact over all 65,536 input pairs each** |
| **`ImageEnhance`** — `Color Contrast Brightness Sharpness` | ✅ 60 results byte-identical |
| **`ImageOps`** — `autocontrast colorize contain cover crop deform equalize expand fit flip grayscale invert mirror pad posterize scale solarize exif_transpose` | ✅ 38 results byte-identical |
| **`ImageStat`**, **`ImageMath`**, **`ImageMorph`** | ✅ nine properties, 30 expressions, nine operators, all exact |
| **`quantize`** — median cut, max coverage, fast octree; `RGBA → P`; `remap_palette`, `putpalette` | ✅ with an **error bound**, not byte equality (below) |
| `ImageDraw`, `ImageFont` | ⏳ P7 |
| GIF, TIFF, ICO, WebP, PCX, XBM, the read-only long tail, PDF write | ⏳ P8 |

**Never implemented, and why** — `ImageTk`, `ImageQt`, `ImageWin`, `ImageShow`, `ImageGrab`: each
binds a GUI toolkit or an OS screen-capture API that Bantu has no binding to, so there is nothing to
implement. `ImageCms`: ICC colour management is a library-sized problem and a half-implementation
would silently produce wrong colour, which is the worst available failure; profiles are **preserved**
in `info["icc_profile"]` and round-trip through `save`, they are just not applied.

---

## Install

The package is plain Bantu source. Include the umbrella and you have everything:

```bantu
include "./pillow/pillow.b" as pillow;
pillow.help();          // prints the surface and the current phase
```

It depends on `numba` only through the interpreter's built-in `nd_*` primitives, so there is nothing
to install.

---

## `Image`

### Creating

| call | notes |
|---|---|
| `pillow.new($mode, $size, $colour)` | `$size` is `[width, height]`. `$colour` may be `null`, a number, a list per band, or a colour string. **`null` fills every band with 0**, so a new `RGBA` image is *transparent* black — Pillow's behaviour |
| `pillow.open($path)` | sniffs the format by magic bytes, falls back to the extension, raises `UnidentifiedImageError` |
| `pillow.open_bytes($data, $hintPath)` | the same, for bytes already in memory |
| `pillow.frombytes($mode, $size, $data)` | raw pixel bytes, row-major |
| `pillow.frombuffer($mode, $size, $data)` | an alias, as in Pillow |
| `pillow.fromarray($ndarray, $mode)` | a 2-D array becomes one band; a 3-D array's last axis is bands. `$mode` may be `null` to guess. The data is **copied** |

`pillow.new`, `pillow.open` and `pillow.eval` are spelled exactly as in Pillow even though `new` is a
Bantu keyword and `open` a builtin: they are module variables holding a function, and a member access
parses where a `def` would not.

### Properties

```bantu
$im.mode       // "RGBA"
$im.size       // [width, height]
$im.width      // width
$im.height     // height
$im.info       // a dict: dpi, transparency, duration, …
$im.palette    // for P / PA, else null
$im.format     // "PPM" after open(), else null
$im.getbands() // ["R", "G", "B", "A"]
```

### Geometry

| call | notes |
|---|---|
| `$im.copy()` | an independent copy |
| `$im.crop($box)` | `$box` is `[left, upper, right, lower]`, right/lower exclusive. A box **outside** the image is legal and the outside is zero-filled, as in Pillow. Returns a copy, never an alias |
| `$im.paste($src, $box, $mask)` | mutates and returns `null`, as Pillow does. `$src` is an image or a colour; `$box` is `[left, upper]` or `[left, upper, right, lower]`; `$mask` is an image or `null`. A box partly or wholly outside is clipped, not an error. A source in another mode is converted |

### Bands

| call | notes |
|---|---|
| `$im.split()` | a list of single-band images |
| `pillow.merge($mode, $bands)` | the inverse |
| `$im.getchannel($nameOrIndex)` | `"G"` or `1` |
| `$im.putalpha($imageOrNumber)` | promotes `RGB`→`RGBA` and `L`→`LA` if needed. Mutates |

### Pixels

| call | notes |
|---|---|
| `$im.getpixel([$x, $y])` | a number for one band, a list otherwise. **Slow** — see below |
| `$im.putpixel([$x, $y], $value)` | **Slow** |
| `$im.getdata()` | every pixel, row-major |
| `$im.putdata($data, $scale, $offset)` | replaces the *leading* pixels, as in Pillow |
| `$im.point($lutOrFunction, $mode)` | a 256-entry table, a function, or one table per band concatenated. A **function is called once per possible value, never per pixel** |
| `$im.tobytes()` | raw pixel bytes |

> **`getpixel` and `putpixel` are slow on purpose.** One costs about 2.5 µs, so a loop over a 1080p
> image through them is five seconds. That is not a defect to be fixed; it is the reason `point()`,
> `paste()`, `blend()` and the operators exist, each of which is a handful of whole-array operations.
> Pillow's own `getpixel` is slow for the same reason.

### Statistics

| call | notes |
|---|---|
| `$im.getextrema()` | `[lo, hi]` for one band, `[[lo,hi], …]` otherwise |
| `$im.histogram($mask)` | 256 bins **per band, concatenated** — 768 entries for RGB, as in Pillow. `$mask` is not implemented yet (P6) |
| `$im.getbbox()` | `[l, t, r, b]` of the non-zero region, or `null` |
| `$im.getcolors($maxcolors)` | `[[count, colour], …]`, or `null` when there are more than `$maxcolors` distinct colours — Pillow's contract |

### Conversion

```bantu
$grey = $im.convert("L", null, null);                 // dither defaults to Floyd–Steinberg
$bw   = $im.convert("1", "NONE", null);               // threshold instead
```

`$im.convert($mode, $dither, $matrix)`. `$dither` is `"FLOYDSTEINBERG"` (the default, matching
Pillow) or `"NONE"`. `$matrix` is not implemented yet (P6).

### Geometry

| call | notes |
|---|---|
| `$im.resize($size, $filter, $box, $reducing_gap)` | `$filter` is `NEAREST LANCZOS BILINEAR BICUBIC BOX HAMMING`, default `BICUBIC`. Modes `1` and `P` force `NEAREST`; `LA` and `RGBA` are premultiplied first so transparent pixels do not bleed colour |
| `$im.transpose($op)` | `FLIP_LEFT_RIGHT FLIP_TOP_BOTTOM ROTATE_90 ROTATE_180 ROTATE_270 TRANSPOSE TRANSVERSE`. Each is an axis permutation, so all are exact and effectively free. `ROTATE_90` is **anticlockwise** |
| `$im.rotate($angle, $filter, $expand, $center, $translate, $fill)` | counter-clockwise degrees. 0/90/180/270 take the exact transpose path when there is no custom centre or translation |
| `$im.reduce($factor, $box)` | integer box averaging; the output size rounds **up**, and a partial edge block averages over its own smaller area. Refuses `P`, `1` and `I;16`, as Pillow does |
| `$im.thumbnail($size, $filter, $reducing_gap)` | shrinks **in place** and returns `null`. Never enlarges; preserves the aspect ratio with Pillow's `round_aspect` |
| `$im.transform($size, $method, $data, $filter, $fill)` | `AFFINE EXTENT PERSPECTIVE QUAD MESH`, with `NEAREST`, `BILINEAR` or `BICUBIC`. A source coordinate outside the image takes `$fill` rather than a clamped edge pixel |

```bantu
$thumb = $im.resize([320, 240], pillow.LANCZOS, null, null);
$flipped = $im.transpose(pillow.FLIP_LEFT_RIGHT);
$turned = $im.rotate(30, pillow.BICUBIC, true, null, null, "white");
$half = $im.reduce(2, null);
$t = $im.transform([200, 150], pillow.AFFINE, [1, 0, 10, 0, 1, 20], pillow.BILINEAR, null);
```

`pillow.AffineTransform($m)`, `ExtentTransform`, `PerspectiveTransform`, `QuadTransform` and
`MeshTransform` name the method instead of its id. `pillow.Path($points)` is `ImagePath.Path`, with
`path_getbbox`, `path_tolist`, `path_transform` and `path_compact`.

### Output

| call | notes |
|---|---|
| `$im.save($path, $format)` | `$format` may be `null` to infer from the extension |
| `$im.show($title)` | writes a file and prints its path. Bantu has no GUI binding and no shell builtin, so there is nothing to hand a file to — [`bplot.show()`](bplot.md) makes the same accommodation |

### Module-level operations

| call | notes |
|---|---|
| `pillow.blend($a, $b, $alpha)` | `a·(1−α) + b·α` |
| `pillow.composite($a, $b, $mask)` | `a` where the mask is 255, `b` where it is 0 |
| `pillow.alpha_composite($dst, $src)` | Porter–Duff "over"; both must be `RGBA` |
| `pillow.eval($im, $fn)` | `$fn` applied through a lookup table |

---

## Modes

All twenty of Pillow's modes, with Pillow's band names, `basemode`, `basetype` and default fill —
each checked against Pillow itself.

| mode | bands | range | notes |
|---|---|---|---|
| `1` | 1 | {0, 255} | bilevel, stored as 0/255 and packed only by a codec |
| `L` `LA` `La` | 1, 2, 2 | 0–255 | grey; `La` is alpha-premultiplied, and its band names are `L,a` (lowercase) |
| `P` `PA` | 1, 2 | 0–255 | palette index plus the palette beside it |
| `RGB` `RGBA` `RGBa` `RGBX` | 3, 4, 4, 4 | 0–255 | `RGBa` is premultiplied; **`RGBX`'s fourth band is padding, not alpha**, and converting to it fills that band with 255 |
| `CMYK` | 4 | 0–255 | inverted ink, with Pillow's naive conversion |
| `YCbCr` `LAB` `HSV` | 3 | 0–255 | see the exactness note below |
| `I` `I;16` `I;16L` `I;16B` `I;16N` | 1 | int32 / 0–65535 | the suffix is a *file* byte order, not a store difference |
| `F` | 1 | any | float32 semantics, so `RGB→F` gives the float32 nearest the luminance |

```bantu
pillow.getmodebands("RGBA")     // 4
pillow.getmodebase("CMYK")      // "RGB"   -- Pillow's basemode
pillow.getmodetype("F")         // "F"
pillow.getmodebandnames("La")   // ["L", "a"]
pillow.MODES                    // all 20, in Pillow's order
```

### How exact is it?

Measured against Pillow 13.0.0 and reported rather than assumed:

| | result |
|---|---|
| conversion pairs byte-identical | **177 of 191** |
| `RGB → L`, over **all 16,777,216** RGB triples | 0 mismatches |
| `RGB → YCbCr` and `HSV`, both directions, 3,014 pixels | 0 mismatches |
| Floyd–Steinberg to mode `1`, 24 images from 1×1 to 64×40 | **9,154 of 9,154** pixels |
| `RGB → P` undithered, 5 images | **815 of 815** pixels |
| `resize`, 9 modes × 7 shape pairs × 6 filters | **378 of 378** images |
| `transpose`, `transform`, `rotate`, `thumbnail` | **49/49, 45/45, 32/32, 30/30** |
| `reduce`, every integer mode | **84 of 84** |
| **PngSuite**, the canonical PNG conformance corpus | **162 of 162** readable files; all 14 corrupt files refused |
| images written by Pillow at every option (PNG/BMP/TGA) | **238 of 238** |

`RGB → L` is Pillow's `L24` macro — `(R×19595 + G×38470 + B×7471 + 32768) >> 16` — and **not** the
truncated `(R×299 + G×587 + B×114)/1000` that BT.601 references print; the two disagree on about half
of all inputs. `convert("1")` uses the truncated form, because `tobilevel` does. Both are matched.

**The three remaining gaps, stated rather than rounded off:**

- **`RGBA → P`** (4 pairs) routes through `quantize()` in Pillow — an *adaptive* median-cut palette,
  a feature of its own. `convert()` raises an error naming it rather than returning fixed-cube
  indices that would look almost right. Lands in P6.
- **`LAB`** (10 pairs) cannot be matched at all: Pillow has no LAB row in `Convert.c` and routes it
  through **ImageCms, i.e. littleCMS**. `pillow` ships a correct standard D65 sRGB ↔ L\*a\*b\*
  conversion; `convert.why_unverified()` explains why it is not bit-identical. Closing it would mean
  shipping an ICC engine.
- **`reduce` in mode F** differs by a single float32 ULP on about one output pixel in forty, because
  `Reduce.c` dispatches the common factors to eleven hand-unrolled kernels whose float addition order
  differs. Every integer mode is exact. Bounded and asserted, not chased.

### Where this deliberately differs from Pillow

Matching Pillow's arithmetic does not extend to reproducing its defects. Three fixes, each with a
test asserting the *containment* of the difference:

| upstream | fix |
|---|---|
| the dithered `topalette` writes blue's error accumulators into all three channel slots at each row's end, discarding red's and green's | each channel carries its own. Row 0 is identical to Pillow; the difference stays within 4 columns of the right edge |
| `ImagingConvertRGB2YCbCr` casts with `(UINT8)`, so an out-of-range input *wraps* — 256 becomes 0 | clamp. Identical on all valid input |
| a negative bicubic undershoot is cast to `UINT8`, which is undefined behaviour and on x86 wraps −5 to 251 — a bright speck in a dark area | clamp at zero |
| Pillow verifies metadata chunk CRCs but **not IDAT's**, so it accepts a PNG with a corrupted image checksum | verify every CRC. `png.allow_truncated(true)` is the opt-out, and it recovers Pillow's exact pixels |

---

## `ImageColor`

```bantu
pillow.getrgb("#336699")          // [51, 102, 153]
pillow.getrgb("rebeccapurple")    // [102, 51, 153]
pillow.getrgb("hsl(180,50%,50%)") // [64, 191, 191]
pillow.getcolor("red", "L")       // 76
pillow.getcolor("red", "RGBA")    // [255, 0, 0, 255]
pillow.color_names()              // all 148, sorted
```

Accepted: the 148 CSS3 names, `#rgb`, `#rgba`, `#rrggbb`, `#rrggbbaa`, `rgb(r,g,b)`,
`rgb(r%,g%,b%)`, `rgba(r,g,b,a)`, `hsl(h,s%,l%)`, `hsv(h,s%,v%)` and `hsb(…)`.

Four of Pillow's quirks are matched deliberately, and each has a named test so a later change cannot
quietly drop it:

- the spec is lowercased but **not trimmed**, so `"  red  "` is an error;
- `"transparent"` is **not** a colour name;
- there is no `#rrrgggbbb` or `#rrrrggggbbbb`, whatever other libraries accept;
- **`getcolor(c, "CMYK")` returns three values, not four**, and `getcolor(c, "1")` returns a grey
  level rather than 0 or 255. `getcolor` switches on `basemode`: it parses a colour, it does not
  convert a colour space. Use `convert()` for that.

---

## Formats

```bantu
pillow.formats()                 // ["PPM"]
pillow.registered_extensions()   // {".ppm": "PPM", ".pgm": "PPM", …}
pillow.identify($data, $path)    // the format id, or null
```

**PNG** — colour types 0, 2, 3, 4 and 6; bit depths 1, 2, 4, 8 and 16; Adam7 interlace; all five
scanline filters; `PLTE tRNS gAMA pHYs sRGB bKGD tEXt zTXt iTXt`. **Every chunk CRC is verified** —
`png.allow_truncated(true)` is the opt-out. Writes 1, L, LA, P, RGB, RGBA and I;16; pass
`png_filter` (0–4) and `png_level` (0–9) through `pillow.encode_png` for control over the output.

Two Pillow behaviours worth knowing: a **16-bit truecolour PNG comes back as 8-bit RGB** (the low
byte of each sample is dropped) and **16-bit grey+alpha comes back as RGBA**, not LA. Both are
matched.

**JPEG** — baseline (SOF0/SOF1) and **progressive** (SOF2); every subsampling, including the ones
Pillow cannot write (4:1:1, 2x1, 1x2, 4x2, 3x1 and mixed factors); restart markers; JFIF, Adobe
APP14, EXIF and ICC profiles split across APP2 segments; greyscale, RGB, CMYK and YCCK with Adobe's
inversion. Writes 1, L, RGB, RGBX, CMYK and YCbCr.

```bantu
$im = pillow.open("photo.jpg");
$im.save("out.jpg", null, {"quality": 90, "subsampling": 2, "progressive": true});
$im.save("out.jpg", null, {"quality": "web_high"});        // Pillow's presets
$im.save("out.jpg", null, {"quality": "keep"});            // the file's own tables
$im.info.jpeg_qtables                                      // and they come back
$im.info.jpeg_subsampling                                  // 0, 1, 2 or -1
```

Save options: `quality` (1–100, `"keep"`, or a preset name), `optimize`, `progressive`,
`subsampling` (0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0, or `"keep"`), `qtables` (a list of 64-entry tables
in natural order), `restart_marker_blocks`, `restart_marker_rows`, and `dpi`, `exif`, `icc_profile`
and `comment` as info keys. The defaults are libjpeg's: quality 75, 4:2:0.

**`draft($mode, $size)`** decodes through libjpeg's *scaled* IDCT at 1/2, 1/4 or 1/8 — a smaller
image with the slow exact kernel, not a resize, and bit-exact with Pillow's. It can also ask a
3-component file for `L` or `YCbCr`, which is libjpeg's `out_color_space` and not a conversion of the
RGB result. **It does not make reading faster** (below).

**BMP** — 1, 4, 8, 16, 24 and 32 bits; `BITMAPCOREHEADER`, `BITMAPINFOHEADER` and the V2–V5
extensions; uncompressed, RLE4, RLE8 and BITFIELDS; bottom-up and top-down. Writes 1, L, P and RGB.
**The writer drops alpha**, as Pillow's does — save an RGBA image and it reads back as RGB.

**TGA** — types 1 (colour-mapped), 2 (truecolour) and 3 (greyscale) and their RLE forms; 1, 8, 15,
16, 24 and 32 bits; both vertical origins. Writes 1, L, P, RGB and RGBA; pass `tga_rle` or
`tga_topdown` through `pillow.encode_tga`. A TGA has **no magic number**, so it is identified by a
self-consistency check on its 18-byte header and by its extension.

**PPM / PGM / PBM** — all seven Netpbm formats: `P1` `P2` `P3` (plain) and `P4` `P5` `P6` (raw), 8-
and 16-bit, header comments and arbitrary whitespace, read and write. Writable from modes `1`, `L`,
`I`, `I;16*` and `RGB`; other modes raise rather than writing nonsense, which is Pillow's restriction
too.

### Adding your own

```bantu
pillow.register_codec("MYFMT", ["mft"], "image/x-myfmt",
    def($head) { return substr($head, 0, 4) == "MYFM"; },
    def($data) { return {"mode": "RGB", "size": [$w, $h], "px": $buf,
                         "palette": null, "info": {}}; },
    def($spec) { return $bytes; });
```

A codec deals in **pixel buffers**, never in `Image` objects — that is what keeps the array layer
independent of the object layer. Pass `null` for the sniffer (found by extension instead) or for the
saver (read-only).

---

## The operation surface

```bantu
$im.filter(pillow.GaussianBlur(2));             // ImageFilter
$im.filter(pillow.Kernel(3, $weights, 16, 0));
$im.filter(pillow.MedianFilter(5));

pillow.chop_screen($a, $b);                     // ImageChops, 24 operations
pillow.chop_add($a, $b, 2.0, 10);

pillow.enhance(pillow.Contrast($im), 1.6);      // ImageEnhance

pillow.autocontrast($im, 2, null, null, null);  // ImageOps
pillow.equalize($im, null);
pillow.fit($im, [320, 240], null, 0.05, [0.5, 0.5]);

$st = pillow.Stat($im, null);                   // ImageStat
$st.mean $st.median $st.stddev $st.extrema

pillow.imagemath_eval("(a + b) / 2", {"a": $x, "b": $y});     // ImageMath
$op = pillow.MorphOp(null, "dilation4", null);  // ImageMorph
pillow.morph_apply($op, $bilevel);              // [changed, image]

$p = $im.quantize(64, null, null, null, null);  // an adaptive palette
```

**Three contracts worth knowing before you port code**, because each is Pillow's and not the obvious
choice:

* a convolution's **border is copied, not filtered**, the kernel is applied **vertically flipped**,
  and only 3×3 and 5×5 exist — a 7×7 `Kernel` raises `bad kernel size`, and an image smaller than
  the kernel comes back unchanged (that check comes first, so a 7×7 kernel on a 5×5 image is a copy).
* `add_modulo` and `subtract_modulo` **wrap**; `add` and `subtract` **clamp** after truncating
  `(a ± b) / scale + offset` toward zero. `difference`, `soft_light`, `hard_light` and `overlay` use
  the wrapping macro too, where it is provably unreachable.
* `ImageMath` divides by zero to **zero**, `%` takes the **dividend's** sign, and `pow` in mode I
  rounds and saturates. `pillow.imagemath_eval` parses a fixed grammar — `+ - * / // % ** & | ^ << >>
  ~`, the comparisons, parentheses, numbers, names and `abs min max convert float int equal
  notequal` — because Pillow's `unsafe_eval` compiles a *Python* expression and there is no Python
  here. Anything outside the grammar names the offending token; the grammar is the sandbox.

### quantize is the one call with an error bound

An adaptive palette is a heuristic. Pillow's `Quant.c` hashes colours into a 65,536-entry table and
merges colours whose hashes collide; `pillow` works from the exact distinct colours. The algorithm is
reproduced — the luminance-weighted split axis, the count median, the pixel-weighted mean rounded
half up, an exact nearest-entry mapping — but the palette can differ, so what is promised and tested
is the **error against the source**, measured:

| colours | median cut (pillow / Pillow) | fast octree (pillow / Pillow) |
|---|---|---|
| 4 | 2749 / 2745 | 5392 / 8263 |
| 16 | 1677 / 1845 | **1122 / 6166** |
| 64 | 396 / 390 | 261 / 273 |

(mean squared error, lower is better). `MAXCOVERAGE` is median cut with a coverage-weighted selection
rather than Pillow's own `quantize2`, a different algorithm; it stays within 2× of Pillow's error.
`LIBIMAGEQUANT` is declined by name — it is an external C library, and half of it would be worse than
none.

---

## EXIF

EXIF is a TIFF block inside a JPEG's APP1 segment, so one tag reader serves both (and `codecs/tiff.b`
will use it in P8 rather than repeat it).

```bantu
$ex = pillow.getexif($im);                       // Image.getexif()
pillow.exif_get($ex, "ifd0", 274);               // Orientation
pillow.exif_get($ex, "exif", 33434);             // ExposureTime, as [n, d]
pillow.exif_get($ex, "gps", 2);                  // GPSLatitude
pillow.exif_tag_name(271);                       // "Make"
pillow.TAGS() pillow.GPSTAGS() pillow.IFD() pillow.ExifBase() pillow.GPS() pillow.Interop()

$ex = pillow.exif_empty();
pillow.exif_put($ex, "ifd0", 274, 6);
$im.save("out.jpg", null, {"exif": pillow.exif_tobytes($ex)});

$upright = pillow.exif_transpose($im);           // ImageOps.exif_transpose
```

Values follow Pillow's rules: count 1 is a scalar, ASCII is a string with one trailing NUL stripped,
`UNDEFINED` and multi-byte `BYTE` are byte strings, a rational is `[numerator, denominator]`, and
anything else with a count above one is a list. The IFDs are `ifd0`, `exif`, `gps`, `interop` and
`ifd1` (the thumbnail directory); `makernote` is kept as raw bytes and `thumbnail` as the embedded
JPEG, if there is one.

Both hostile shapes are handled rather than hit: an **IFD chain that points at itself** raises
instead of hanging (every offset is checked against the block length and a visited set), and a tag
whose **count × type size overshoots the block** is skipped before the read, not after — a `LONG` tag
claiming 2³¹ entries is a header, not an allocation.

---

## Compression

The package needs zlib for PNG, so it is exposed:

```bantu
pillow.zlib_compress($data, 6);        // a zlib stream
pillow.zlib_decompress($data, $maxOut);
pillow.gzip_compress($data, 6);
pillow.gzip_decompress($data, $maxOut);
pillow.crc32($data);
pillow.adler32($data);
```

`$maxOut` is the defence against an inflate bomb: a 1 KB stream can legally expand to gigabytes, so
pass the expected size when you know it and an over-long stream is refused **before** the memory is
committed. `null` applies a 256 MiB ceiling.

The interpreter also gained `inflate(src, maxOut?)` and `gunzip(src, maxOut?)` as general builtins
during this work — it had shipped a zlib *compressor* with no decompressor, so every compressed
format was write-only from Bantu.

---

## Performance

Measured on a 1080p RGB image (6,220,800 channel values), Darwin/Apple silicon:

| operation | cost |
|---|---|
| `crop()`, `transpose()` | **0 ms** — a view or an axis permutation |
| `point()` with a 256-entry table | 92 ms |
| `convert("L")` | 108 ms |
| `histogram()` | 74 ms |
| `resize()` bicubic, to half size | ~530 ms |
| `reduce(4)` | ~20 ms |
| `getpixel()` | ~2.5 µs **each** |

A 60,000-pixel RGB **PNG writes in 13 ms and reads in 8 ms**.

### JPEG is the slow one, and here is the number

A JPEG's entropy-coded scan is a bit-by-bit Huffman walk over every non-zero coefficient, and it is
sequential by construction: the amplitude bits of one coefficient sit between its neighbours' Huffman
codes, so nothing says where the next block starts until the current one is decoded. That is the one
place in this package where the no-pixel-loop rule cannot hold, and the cost is published here rather
than left to be discovered. Measured on photographic 4:2:0 files:

| JPEG | decode | `draft` 1/8 | encode |
|---|---|---|---|
| 320×240 | 0.80 s | 0.73 s | 0.72 s |
| 640×480 | 3.06 s | 2.89 s | 3.12 s |
| 1024×768 | 7.77 s | 7.21 s | 7.59 s |

About **10 µs per pixel**, linear, at every scale. Two consequences worth planning around:

* **`draft()` does not make reading faster.** It saves 6–9 %, because scaling happens in the IDCT and
  the scan is read in full whatever the scale — libjpeg does the same amount of Huffman work for a
  1/8 decode as for a full one. Use it for libjpeg's exact scaled pixels and for the smaller output
  buffer, not for speed.
* **A progressive JPEG has a coefficient budget.** Successive approximation has to read coefficients
  back while decoding, so they are held in a Bantu list at ~100 bytes each. Over 6 M coefficients
  (about 2 MP at 4:2:0) the decoder refuses by name and says how to proceed —
  `pillow.jpeg_set_max_progressive_coeffs($n)` if the memory is there, or re-save as baseline.
  Baseline files of any size are unaffected.
* **An open JPEG keeps its file bytes** in `info["jpeg_src"]`, which is what `draft()` re-decodes
  from, since an `Image` here is never lazy. That is the file's own size, per open image.

### What the operation surface costs

Measured on a 160×120 RGB image, Darwin/Apple silicon:

| operation | cost |
|---|---|
| a 3×3 convolution (`SMOOTH`) | 23 ms |
| `GaussianBlur(1)` | 22 ms |
| `GaussianBlur(10)` | 22 ms — **the same**, which is what a prefix-sum blur means |
| `MedianFilter(5)` | 63 ms |
| `quantize(64)` | 314 ms |

A convolution costs ~20 whole-array passes for 3×3 and ~50 for 5×5, because each float32 rounding is
one pass and bit-exactness with Pillow needs every one of them. `ModeFilter` and the rank filters
hold `size²` taps per band at once, so a large radius on a large image is memory as well as time:
`RankFilter(5)` on 1080p is about 300 MB per band.

**The rule the whole design exists to satisfy:** no operation is proportional to the pixel count in
*interpreted* work. An interpreted per-pixel pass over a 1080p image costs 4.1 s; the same operation
through `nd_*` costs 0–150 ms. The test suite asserts the timing curve is linear across five sizes,
so a pixel loop fails the build rather than merely being slow.

### The two costs worth knowing before you start

**Memory.** Integer modes are stored in `i64` because `numba`'s dtypes are `f64`/`i64`/`bool`, so a
channel value takes 8 bytes where Pillow uses 1:

| image | Pillow | `pillow` |
|---|---|---|
| 1920×1080 RGB | 6.2 MB | 50 MB |
| 1920×1080 RGBA | 8.3 MB | 66 MB |
| 4000×3000 RGBA | 48 MB | 384 MB |

Comfortable to about **16 megapixels**, heavy above it. A `u8` dtype in `numba` would close the gap
and `mode.b` is the only file that names a store dtype, so the change would be invisible here; it is
tracked as decision PW7.

**Dithering.** `convert("1")` and `convert("P")` default to Floyd–Steinberg, which is the one
genuinely sequential operation in the package — a pixel's error depends on its neighbour's. A 1080p
dither is about 2 seconds. Pass `"NONE"` for a plain threshold, which is a single array comparison.

---

## Security

An imaging library is a parser of hostile input, so this is designed rather than patched.

| hazard | the defence |
|---|---|
| **decompression bomb** — a short header declaring 60,000×60,000 | `MAX_IMAGE_PIXELS` (default 89,478,485, Pillow's number) checked **from the header, before any allocation** |
| **inflate bomb** | `inflate`'s `maxOut`, checked before each append; a codec passes the exact byte count its own header implies |
| **integer overflow in dimensions** | every dimension product goes through a checked multiply |
| **malformed or truncated streams** | every read is bounds-checked first; failures are catchable Bantu errors naming the format and the byte offset |
| **hostile metadata** | `pillow` never emits markup. Metadata comes back as Bantu strings and is the caller's to escape — stated here because the mistake is reachable |

```bantu
pillow.max_image_pixels();          // 89478485
pillow.set_max_image_pixels($n);    // raise it only for a source you trust
```

Every codec phase carries a fuzz tier: the format's corpus bit-flipped, truncated at every length,
and with every length field replaced by 0, 1, 2³¹−1 and 2⁶³−1. The gate is that **no input crashes
or hangs the interpreter**. For P1 that was 3,596 hostile candidates, of which 3,583 were rejected
with catchable errors and none crashed — and the native and pure-Bantu decoders **agreed on every
one**.
