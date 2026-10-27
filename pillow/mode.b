// ════════════════════════════════════════════════════════════════════════
//  pillow/mode.b — the mode table.
//
//  A Pillow image's mode string IS its type: how many bands it has, what
//  range each band covers, what the bands are called, and what it can be
//  converted to. This is the ONE place that knows, so that nothing else has
//  to carry a list of mode strings in a condition.
//
//      include "./mode.b" as mode;
//      mode.bands("RGBA")     -> 4
//      mode.dtype("F")        -> "f64"
//      mode.band_names("LA")  -> ["L", "A"]
//      mode.is_valid("RGBX")  -> true
//
//  THE STORE. Pixels live in a numba array of shape (height, width, bands) --
//  always three-dimensional, even for one-band L and 1, because carrying the
//  trailing axis means every operator is written once instead of twice with
//  a rank test. Integer modes are stored in i64 and float modes in f64,
//  because numba's dtypes are f64/i64/bool. The cost is memory -- eight bytes
//  per channel value where Pillow uses one -- which is decision PW7 and is
//  the documented limit of this design: comfortable to ~16 megapixels.
// ════════════════════════════════════════════════════════════════════════

// ─── The table ──────────────────────────────────────────────────────────
//
// Each entry:
//   bands     how many channels
//   dtype     the numba store dtype
//   lo, hi    the inclusive value range a band may hold
//   names     getbands(), in order
//   pbase     Pillow's OWN basemode, reported faithfully by basemode(). It is
//             a colour-space family in Pillow's terms: L for anything
//             greyscale-ish, RGB for every colour mode INCLUDING CMYK, YCbCr,
//             LAB, HSV and (surprisingly) PA, and P for P. Verified against
//             PIL.ImageMode.getmode, not guessed.
//   ptype     Pillow's basetype: L everywhere except I and F.
//   fam       OUR colour family, which is what the conversion matrix actually
//             switches on. Pillow's basemode cannot serve: it calls CMYK,
//             YCbCr, LAB, HSV and RGB all "RGB", and converting between them
//             is the whole job. One of:
//               bilevel grey palette rgb cmyk ycbcr lab hsv int float
//   alpha     the band index of alpha, or -1
//   premul    is alpha premultiplied into the colour bands?
//   palette   does the mode index a palette?
$_M = {
    "1":     {"bands": 1, "dtype": "i64", "lo": 0, "hi": 255,   "names": ["1"],
              "pbase": "L", "ptype": "L", "fam": "bilevel", "alpha": -1, "premul": false, "palette": false},
    "L":     {"bands": 1, "dtype": "i64", "lo": 0, "hi": 255,   "names": ["L"],
              "pbase": "L", "ptype": "L", "fam": "grey", "alpha": -1, "premul": false, "palette": false},
    "LA":    {"bands": 2, "dtype": "i64", "lo": 0, "hi": 255,   "names": ["L", "A"],
              "pbase": "L", "ptype": "L", "fam": "grey", "alpha": 1,  "premul": false, "palette": false},
    "La":    {"bands": 2, "dtype": "i64", "lo": 0, "hi": 255,   "names": ["L", "a"],
              "pbase": "L",   "ptype": "L", "fam": "grey",   "alpha": 1,  "premul": true,  "palette": false},
    "P":     {"bands": 1, "dtype": "i64", "lo": 0, "hi": 255,   "names": ["P"],
              "pbase": "P", "ptype": "L", "fam": "palette", "alpha": -1, "premul": false, "palette": true},
    "PA":    {"bands": 2, "dtype": "i64", "lo": 0, "hi": 255,   "names": ["P", "A"],
              "pbase": "RGB", "ptype": "L", "fam": "palette", "alpha": 1,  "premul": false, "palette": true},
    "RGB":   {"bands": 3, "dtype": "i64", "lo": 0, "hi": 255,   "names": ["R", "G", "B"],
              "pbase": "RGB", "ptype": "L", "fam": "rgb", "alpha": -1, "premul": false, "palette": false},
    "RGBA":  {"bands": 4, "dtype": "i64", "lo": 0, "hi": 255,   "names": ["R", "G", "B", "A"],
              "pbase": "RGB", "ptype": "L", "fam": "rgb", "alpha": 3,  "premul": false, "palette": false},
    "RGBa":  {"bands": 4, "dtype": "i64", "lo": 0, "hi": 255,   "names": ["R", "G", "B", "a"],
              "pbase": "RGB", "ptype": "L", "fam": "rgb",    "alpha": 3,  "premul": true,  "palette": false},
    "RGBX":  {"bands": 4, "dtype": "i64", "lo": 0, "hi": 255,   "names": ["R", "G", "B", "X"],
              "pbase": "RGB", "ptype": "L", "fam": "rgb", "alpha": -1, "premul": false, "palette": false},
    "CMYK":  {"bands": 4, "dtype": "i64", "lo": 0, "hi": 255,   "names": ["C", "M", "Y", "K"],
              "pbase": "RGB", "ptype": "L", "fam": "cmyk", "alpha": -1, "premul": false, "palette": false},
    "YCbCr": {"bands": 3, "dtype": "i64", "lo": 0, "hi": 255,   "names": ["Y", "Cb", "Cr"],
              "pbase": "RGB", "ptype": "L", "fam": "ycbcr", "alpha": -1, "premul": false, "palette": false},
    "LAB":   {"bands": 3, "dtype": "i64", "lo": 0, "hi": 255,   "names": ["L", "A", "B"],
              "pbase": "RGB", "ptype": "L", "fam": "lab", "alpha": -1, "premul": false, "palette": false},
    "HSV":   {"bands": 3, "dtype": "i64", "lo": 0, "hi": 255,   "names": ["H", "S", "V"],
              "pbase": "RGB", "ptype": "L", "fam": "hsv", "alpha": -1, "premul": false, "palette": false},
    "I":     {"bands": 1, "dtype": "i64", "lo": -2147483648, "hi": 2147483647, "names": ["I"],
              "pbase": "L", "ptype": "I", "fam": "int", "alpha": -1, "premul": false, "palette": false},
    "I;16":  {"bands": 1, "dtype": "i64", "lo": 0, "hi": 65535, "names": ["I"],
              "pbase": "L", "ptype": "L", "fam": "int", "alpha": -1, "premul": false, "palette": false},
    "I;16L": {"bands": 1, "dtype": "i64", "lo": 0, "hi": 65535, "names": ["I"],
              "pbase": "L", "ptype": "L", "fam": "int", "alpha": -1, "premul": false, "palette": false},
    "I;16B": {"bands": 1, "dtype": "i64", "lo": 0, "hi": 65535, "names": ["I"],
              "pbase": "L", "ptype": "L", "fam": "int", "alpha": -1, "premul": false, "palette": false},
    "I;16N": {"bands": 1, "dtype": "i64", "lo": 0, "hi": 65535, "names": ["I"],
              "pbase": "L", "ptype": "L", "fam": "int", "alpha": -1, "premul": false, "palette": false},
    "F":     {"bands": 1, "dtype": "f64", "lo": null, "hi": null, "names": ["F"],
              "pbase": "L", "ptype": "F", "fam": "float", "alpha": -1, "premul": false, "palette": false}
};

// Pillow's Image.MODES -- all twenty, sorted, exactly as Pillow reports it.
// (Checked: list(PIL.Image.MODES) is this list.)
$MODES = ["1", "CMYK", "F", "HSV", "I", "I;16", "I;16B", "I;16L", "I;16N",
          "L", "LA", "La", "LAB", "P", "PA", "RGB", "RGBA", "RGBa", "RGBX",
          "YCbCr"];

// The same twenty in a declaration order that reads better for tests.
$ALL_MODES = ["1", "L", "LA", "La", "P", "PA", "RGB", "RGBA", "RGBa", "RGBX",
              "CMYK", "YCbCr", "LAB", "HSV", "I", "I;16", "I;16L", "I;16B",
              "I;16N", "F"];

// ─── Lookups ────────────────────────────────────────────────────────────

// is_valid($m) -> is this a mode this package understands?
//
// A Bantu dict has no key-presence predicate -- contains() covers lists and
// strings only -- so absence is read as null. Sound here because no row in
// $_M is null, which the mode test asserts.
def is_valid($m) {
    if (type($m) != "string") { return false; }
    return $_M[$m] != null;
}

// _get($m, $who) -> the table row, or a clear error naming the caller.
def _get($m, $who) {
    if (type($m) != "string") {
        throw $who + ": mode must be a string (got " + type($m) + ")";
    }
    $row = $_M[$m];
    if ($row == null) {
        throw $who + ": unknown mode \"" + $m +
              "\". Known modes: " + join($ALL_MODES, " ");
    }
    return $row;
}

// bands($m) -> how many channels. 1 for L, 4 for RGBA.
def bands($m) { return _get($m, "mode.bands").bands; }

// dtype($m) -> the numba store dtype: "i64" for every integer mode, "f64" for F.
def dtype($m) { return _get($m, "mode.dtype").dtype; }

// band_names($m) -> getbands(): ["R","G","B","A"] for RGBA.
// A copy, so a caller cannot mutate the table.
def band_names($m) {
    $src = _get($m, "mode.band_names").names;
    $out = [];
    $i = 0;
    while ($i < len($src)) { push($out, $src[$i]); $i = $i + 1; }
    return $out;
}

// lo($m) / hi($m) -> the inclusive value range, or null for F which is unbounded.
def lo($m) { return _get($m, "mode.lo").lo; }
def hi($m) { return _get($m, "mode.hi").hi; }

// basemode($m) / basetype($m) -> Pillow's own two, reported faithfully.
// Note that basemode calls CMYK, YCbCr, LAB and HSV all "RGB", so it is NOT
// the thing to switch a conversion on -- family() is.
def basemode($m) { return _get($m, "mode.basemode").pbase; }
def basetype($m) { return _get($m, "mode.basetype").ptype; }

// family($m) -> our colour family, which the conversion matrix switches on:
// bilevel grey palette rgb cmyk ycbcr lab hsv int float
def family($m) { return _get($m, "mode.family").fam; }

// has_alpha($m) / alpha_band($m) -> is there an alpha channel, and which index?
// RGBX's fourth band is PADDING, not alpha, which is the whole difference
// between RGBX and RGBA and a classic place to get it wrong.
def has_alpha($m) { return _get($m, "mode.has_alpha").alpha >= 0; }
def alpha_band($m) { return _get($m, "mode.alpha_band").alpha; }

// is_premultiplied($m) -> RGBa and La carry alpha multiplied into the colour.
def is_premultiplied($m) { return _get($m, "mode.is_premultiplied").premul; }

// is_palette($m) -> P and PA index an ImagePalette rather than carrying colour.
def is_palette($m) { return _get($m, "mode.is_palette").palette; }

// is_float($m) -> F only.
def is_float($m) { return _get($m, "mode.is_float").dtype == "f64"; }

// is_bilevel($m) -> mode "1", whose only legal values are 0 and 255.
def is_bilevel($m) { return $m == "1"; }

// is_grey($m) -> does the colour information reduce to one luminance band?
// True for 1, L, LA, La, I, I;16* and F.
def is_grey($m) {
    $f = _get($m, "mode.is_grey").fam;
    return $f == "grey" || $f == "bilevel" || $f == "int" || $f == "float";
}

// is_rgb_like($m) -> three or four bands in R,G,B order. RGB, RGBA, RGBa, RGBX
// only -- NOT CMYK/YCbCr/LAB/HSV, which Pillow's basemode lumps in with them.
def is_rgb_like($m) { return _get($m, "mode.is_rgb_like").fam == "rgb"; }

// colour_bands($m) -> how many bands carry colour, excluding alpha and
// padding. RGBA -> 3, RGBX -> 3, LA -> 1, CMYK -> 4.
def colour_bands($m) {
    $r = _get($m, "mode.colour_bands");
    if ($m == "RGBX") { return 3; }
    if ($r.alpha >= 0) { return $r.bands - 1; }
    return $r.bands;
}

// ─── Byte-order suffix, for the codecs ──────────────────────────────────
//
// "I;16B" is big-endian 16-bit; "I;16L" little-endian; "I;16" and "I;16N"
// native, which on every platform Bantu builds for is little-endian. The
// suffix describes the FILE, not the store -- in memory they are all i64 --
// so this is the function a codec asks, and nothing else needs to care.
def wire_elem($m) {
    if ($m == "I;16B") { return "u16be"; }
    if ($m == "I;16" || $m == "I;16L" || $m == "I;16N") { return "u16le"; }
    if ($m == "I") { return "i32le"; }
    if ($m == "F") { return "f32le"; }
    return "u8";
}

// bits_per_band($m) -> what one band occupies in a file.
def bits_per_band($m) {
    if ($m == "1") { return 1; }
    if ($m == "I;16" || $m == "I;16L" || $m == "I;16B" || $m == "I;16N") { return 16; }
    if ($m == "I") { return 32; }
    if ($m == "F") { return 32; }
    return 8;
}

// ─── Clamping ───────────────────────────────────────────────────────────
//
// clamp_spec($m) -> {lo, hi} or null, so a caller can hand it straight to
// nd_clip. F is unbounded and returns null.
def clamp_spec($m) {
    $r = _get($m, "mode.clamp_spec");
    if ($r.lo == null) { return null; }
    return {"lo": $r.lo, "hi": $r.hi};
}

// default_colour($m) -> the value Image.new uses when no colour is given.
//
// ZERO in every band, alpha INCLUDED -- so a new RGBA image is transparent
// black, not opaque black. Checked against Pillow: Image.new("RGBA",(2,2))
// has getpixel == (0, 0, 0, 0). Guessing 255 for alpha here is the obvious
// mistake and it would make every new RGBA image differ from Pillow's.
def default_colour($m) {
    $r = _get($m, "mode.default_colour");
    $out = [];
    $i = 0;
    while ($i < $r.bands) { push($out, 0); $i = $i + 1; }
    return $out;
}

// white($m) -> the all-maximum value, which "1", L and ImageOps need for
// inversion. Every band including alpha, so it is opaque white.
def white($m) {
    $r = _get($m, "mode.white");
    $top = $r.hi;
    if ($top == null) { $top = 1.0; }
    $out = [];
    $i = 0;
    while ($i < $r.bands) { push($out, $top); $i = $i + 1; }
    return $out;
}
