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

`pillow` is being built in nine phases. **This is phase P2 of 9.** The table below is the honest
state; nothing is stubbed to look present.

| | status |
|---|---|
| **The Image object** — `new open copy crop paste split merge getchannel putalpha getpixel putpixel getdata putdata point histogram getbbox getextrema getcolors blend composite alpha_composite eval tobytes frombytes fromarray save show` | ✅ |
| **All 20 modes** — `1 L LA La P PA RGB RGBA RGBa RGBX CMYK YCbCr LAB HSV I I;16 I;16L I;16B I;16N F` | ✅ |
| **The conversion matrix** | ✅ **111 pairs byte-identical to Pillow**; 60 close but not yet exact (below); 20 need quantisation |
| **`ImageColor`** — `getrgb getcolor colormap`, 148 CSS names | ✅ |
| **PPM / PGM / PBM** (P1–P6), 8- and 16-bit, read + write | ✅ |
| **DEFLATE / zlib / gzip**, both directions | ✅ |
| **The plugin registry** — `register_codec`, sniffing, `registered_extensions` | ✅ |
| `ImagePalette`, `quantize`, `* → P` conversions | ⏳ P2 remainder |
| `resize rotate transpose thumbnail reduce transform`, `ImageTransform`, `ImagePath` | ⏳ P3 |
| **PNG**, BMP, TGA | ⏳ P4 |
| **JPEG**, `ExifTags` | ⏳ P5 |
| `ImageFilter ImageChops ImageEnhance ImageOps ImageStat ImageMath ImageMorph` | ⏳ P6 |
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

### How exact is `convert()`?

Measured against Pillow 13.0.0 over a 191-pair reference matrix, and reported rather than assumed:

| | pairs |
|---|---|
| **byte-identical to Pillow** | **111** |
| need quantisation (`* → P`, `* → PA`) | 20 — raise an error naming `palette.b` |
| **close but not yet identical**: the `YCbCr`, `HSV` and `LAB` families | 60 |

`RGB → L` is additionally verified over **all 16,777,216 RGB triples**. It is Pillow's `L24` macro —
`(R×19595 + G×38470 + B×7471 + 32768) >> 16` — and **not** the truncated `(R×299 + G×587 + B×114)/1000`
that BT.601 references print; the two disagree on about half of all inputs.

**For `YCbCr`, `HSV` and `LAB`**, `pillow` uses the standard formulations, which land within ±1 of
Pillow on 8–25% of inputs. They work and they are documented as not yet bit-exact; ask
`convert.unverified($from, $to)` if your code depends on matching Pillow exactly. Finishing them is
the first item of the next phase, and the fitting evidence so far is in the roadmap.

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
| `crop()` | **0 ms** — a view |
| `point()` with a 256-entry table | 92 ms |
| `convert("L")` | 108 ms |
| `histogram()` | 74 ms |
| `getpixel()` | ~2.5 µs **each** |

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
