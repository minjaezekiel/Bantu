// ════════════════════════════════════════════════════════════════════════
//  pillow/color.b — ImageColor: colour strings to numbers.
//
//      include "./color.b" as color;
//      color.getrgb("#336699")        -> [51, 102, 153]
//      color.getrgb("rebeccapurple")  -> [102, 51, 153]
//      color.getrgb("hsl(180,50%,50%)") -> [64, 191, 191]
//      color.getcolor("red", "L")     -> 76
//      color.getcolor("red", "RGBA")  -> [255, 0, 0, 255]
//
//  WHAT IS MATCHED, AND WHY SO LITERALLY. These functions are Pillow's
//  ImageColor.getrgb and getcolor, including the parts that are surprising:
//
//    * the spec is lowercased, and NOT trimmed -- "  red  " is an error in
//      Pillow, so it is an error here
//    * "transparent" is not a colour name; Pillow has no such entry
//    * only #rgb, #rgba, #rrggbb and #rrggbbaa are accepted. There is no
//      #rrrgggbbb or #rrrrggggbbbb, whatever other libraries do
//    * rgb() and rgba() take integers only -- "rgb(1.5,0,0)" is an error
//    * getcolor("red", "CMYK") returns THREE values, not four, because
//      Pillow's getcolor switches on basemode and CMYK's basemode is "RGB".
//      getcolor is not a mode conversion; convert.b is
//    * getcolor("red", "1") returns 76, not 0 or 255 -- it does not threshold
//    * getcolor to "La" or "RGBa" premultiplies the colour by alpha
//
//  Every one of those is pinned by tests/fixtures/pillow/reference/, which
//  was produced by running Pillow, not by reading its documentation.
//
//  The 148 names below are generated from that same reference file.
// ════════════════════════════════════════════════════════════════════════

include "./mode.b" as mode;

// ─── The CSS3 named colours, exactly Pillow's ImageColor.colormap ───────
$_NAMED = {
    "aliceblue": [240, 248, 255], "antiquewhite": [250, 235, 215], "aqua": [0, 255, 255],
    "aquamarine": [127, 255, 212], "azure": [240, 255, 255], "beige": [245, 245, 220],
    "bisque": [255, 228, 196], "black": [0, 0, 0], "blanchedalmond": [255, 235, 205],
    "blue": [0, 0, 255], "blueviolet": [138, 43, 226], "brown": [165, 42, 42],
    "burlywood": [222, 184, 135], "cadetblue": [95, 158, 160], "chartreuse": [127, 255, 0],
    "chocolate": [210, 105, 30], "coral": [255, 127, 80], "cornflowerblue": [100, 149, 237],
    "cornsilk": [255, 248, 220], "crimson": [220, 20, 60], "cyan": [0, 255, 255],
    "darkblue": [0, 0, 139], "darkcyan": [0, 139, 139], "darkgoldenrod": [184, 134, 11],
    "darkgray": [169, 169, 169], "darkgreen": [0, 100, 0], "darkgrey": [169, 169, 169],
    "darkkhaki": [189, 183, 107], "darkmagenta": [139, 0, 139], "darkolivegreen": [85, 107, 47],
    "darkorange": [255, 140, 0], "darkorchid": [153, 50, 204], "darkred": [139, 0, 0],
    "darksalmon": [233, 150, 122], "darkseagreen": [143, 188, 143], "darkslateblue": [72, 61, 139],
    "darkslategray": [47, 79, 79], "darkslategrey": [47, 79, 79], "darkturquoise": [0, 206, 209],
    "darkviolet": [148, 0, 211], "deeppink": [255, 20, 147], "deepskyblue": [0, 191, 255],
    "dimgray": [105, 105, 105], "dimgrey": [105, 105, 105], "dodgerblue": [30, 144, 255],
    "firebrick": [178, 34, 34], "floralwhite": [255, 250, 240], "forestgreen": [34, 139, 34],
    "fuchsia": [255, 0, 255], "gainsboro": [220, 220, 220], "ghostwhite": [248, 248, 255],
    "gold": [255, 215, 0], "goldenrod": [218, 165, 32], "gray": [128, 128, 128],
    "green": [0, 128, 0], "greenyellow": [173, 255, 47], "grey": [128, 128, 128],
    "honeydew": [240, 255, 240], "hotpink": [255, 105, 180], "indianred": [205, 92, 92],
    "indigo": [75, 0, 130], "ivory": [255, 255, 240], "khaki": [240, 230, 140],
    "lavender": [230, 230, 250], "lavenderblush": [255, 240, 245], "lawngreen": [124, 252, 0],
    "lemonchiffon": [255, 250, 205], "lightblue": [173, 216, 230], "lightcoral": [240, 128, 128],
    "lightcyan": [224, 255, 255], "lightgoldenrodyellow": [250, 250, 210], "lightgray": [211, 211, 211],
    "lightgreen": [144, 238, 144], "lightgrey": [211, 211, 211], "lightpink": [255, 182, 193],
    "lightsalmon": [255, 160, 122], "lightseagreen": [32, 178, 170], "lightskyblue": [135, 206, 250],
    "lightslategray": [119, 136, 153], "lightslategrey": [119, 136, 153], "lightsteelblue": [176, 196, 222],
    "lightyellow": [255, 255, 224], "lime": [0, 255, 0], "limegreen": [50, 205, 50],
    "linen": [250, 240, 230], "magenta": [255, 0, 255], "maroon": [128, 0, 0],
    "mediumaquamarine": [102, 205, 170], "mediumblue": [0, 0, 205], "mediumorchid": [186, 85, 211],
    "mediumpurple": [147, 112, 219], "mediumseagreen": [60, 179, 113], "mediumslateblue": [123, 104, 238],
    "mediumspringgreen": [0, 250, 154], "mediumturquoise": [72, 209, 204], "mediumvioletred": [199, 21, 133],
    "midnightblue": [25, 25, 112], "mintcream": [245, 255, 250], "mistyrose": [255, 228, 225],
    "moccasin": [255, 228, 181], "navajowhite": [255, 222, 173], "navy": [0, 0, 128],
    "oldlace": [253, 245, 230], "olive": [128, 128, 0], "olivedrab": [107, 142, 35],
    "orange": [255, 165, 0], "orangered": [255, 69, 0], "orchid": [218, 112, 214],
    "palegoldenrod": [238, 232, 170], "palegreen": [152, 251, 152], "paleturquoise": [175, 238, 238],
    "palevioletred": [219, 112, 147], "papayawhip": [255, 239, 213], "peachpuff": [255, 218, 185],
    "peru": [205, 133, 63], "pink": [255, 192, 203], "plum": [221, 160, 221],
    "powderblue": [176, 224, 230], "purple": [128, 0, 128], "rebeccapurple": [102, 51, 153],
    "red": [255, 0, 0], "rosybrown": [188, 143, 143], "royalblue": [65, 105, 225],
    "saddlebrown": [139, 69, 19], "salmon": [250, 128, 114], "sandybrown": [244, 164, 96],
    "seagreen": [46, 139, 87], "seashell": [255, 245, 238], "sienna": [160, 82, 45],
    "silver": [192, 192, 192], "skyblue": [135, 206, 235], "slateblue": [106, 90, 205],
    "slategray": [112, 128, 144], "slategrey": [112, 128, 144], "snow": [255, 250, 250],
    "springgreen": [0, 255, 127], "steelblue": [70, 130, 180], "tan": [210, 180, 140],
    "teal": [0, 128, 128], "thistle": [216, 191, 216], "tomato": [255, 99, 71],
    "turquoise": [64, 224, 208], "violet": [238, 130, 238], "wheat": [245, 222, 179],
    "white": [255, 255, 255], "whitesmoke": [245, 245, 245], "yellow": [255, 255, 0],
    "yellowgreen": [154, 205, 50]
};

// ─── Small parsing helpers (Bantu has no regex, so these are by hand) ───

// _isDigits($s) -> is every character 0-9, and at least one?
def _isDigits($s) {
    $n = len($s);
    if ($n == 0) { return false; }
    $i = 0;
    while ($i < $n) {
        $c = ord($s[$i]);
        if ($c < 48 || $c > 57) { return false; }
        $i = $i + 1;
    }
    return true;
}

// _isNum($s) -> matches Pillow's \d+\.?\d* : digits, an optional dot, more
// digits. So "12", "12.", "12.5" pass and ".5", "-1", "1e3" do not.
def _isNum($s) {
    $n = len($s);
    if ($n == 0) { return false; }
    $i = 0;
    $seen = 0;
    while ($i < $n && ord($s[$i]) >= 48 && ord($s[$i]) <= 57) { $i = $i + 1; $seen = $seen + 1; }
    if ($seen == 0) { return false; }
    if ($i == $n) { return true; }
    if ($s[$i] != ".") { return false; }
    $i = $i + 1;
    while ($i < $n && ord($s[$i]) >= 48 && ord($s[$i]) <= 57) { $i = $i + 1; }
    return $i == $n;
}

// _isHex($s) -> every character 0-9 or a-f (the spec is already lowercased).
def _isHex($s) {
    $n = len($s);
    if ($n == 0) { return false; }
    $i = 0;
    while ($i < $n) {
        $c = ord($s[$i]);
        $okDigit = ($c >= 48 && $c <= 57);
        $okAlpha = ($c >= 97 && $c <= 102);
        if (!$okDigit && !$okAlpha) { return false; }
        $i = $i + 1;
    }
    return true;
}

def _hexVal($c) {
    $v = ord($c);
    if ($v >= 48 && $v <= 57) { return $v - 48; }
    return $v - 97 + 10;
}

// _lower($s) -> ASCII lowercase. Pillow lowercases the whole spec first.
def _lower($s) {
    $out = [];
    $i = 0;
    $n = len($s);
    while ($i < $n) {
        $c = ord($s[$i]);
        if ($c >= 65 && $c <= 90) { push($out, chr($c + 32)); }
        else { push($out, $s[$i]); }
        $i = $i + 1;
    }
    return join($out, "");
}

// _trim1($s) -> strip spaces and tabs from both ends. Used ONLY inside
// rgb(...) argument lists, where Pillow's own \s* allows them. The whole
// spec is deliberately not trimmed.
def _trim1($s) {
    $a = 0;
    $b = len($s);
    while ($a < $b && ($s[$a] == " " || $s[$a] == "\t")) { $a = $a + 1; }
    while ($b > $a) {
        $c = $s[$b - 1];
        if ($c != " " && $c != "\t") { $b = $b; break; }
        $b = $b - 1;
    }
    return substr($s, $a, $b - $a);
}

// _args($s, $head) -> the comma-separated arguments of "head( ... )", each
// trimmed, or null when the shape does not match.
def _args($s, $head) {
    $hl = len($head);
    if (len($s) < $hl + 2) { return null; }
    if (substr($s, 0, $hl) != $head) { return null; }
    if ($s[$hl] != "(") { return null; }
    if ($s[len($s) - 1] != ")") { return null; }
    $inner = substr($s, $hl + 1, len($s) - $hl - 2);
    $raw = split($inner, ",");
    $out = [];
    $i = 0;
    while ($i < len($raw)) { push($out, _trim1($raw[$i])); $i = $i + 1; }
    return $out;
}

// _pct($s) -> the number before a trailing '%', or null.
def _pct($s) {
    $n = len($s);
    if ($n < 2) { return null; }
    if ($s[$n - 1] != "%") { return null; }
    $body = substr($s, 0, $n - 1);
    if (!_isNum($body)) { return null; }
    return num($body);
}

// ─── colorsys, ported exactly ───────────────────────────────────────────
//
// Pillow calls Python's colorsys for hsl() and hsv(), so these are that
// module's algorithms, not an equivalent of our own. The rounding that
// follows is Pillow's int(f * 255 + 0.5).

$_ONE_THIRD = 0.3333333333333333;
$_ONE_SIXTH = 0.16666666666666666;
$_TWO_THIRD = 0.6666666666666666;

def _hlsV($m1, $m2, $hue) {
    $h = $hue % 1.0;
    if ($h < 0) { $h = $h + 1.0; }
    if ($h < $_ONE_SIXTH) { return $m1 + ($m2 - $m1) * $h * 6.0; }
    if ($h < 0.5) { return $m2; }
    if ($h < $_TWO_THIRD) { return $m1 + ($m2 - $m1) * ($_TWO_THIRD - $h) * 6.0; }
    return $m1;
}

// hls_to_rgb($h, $l, $s) -> [r, g, b] as floats in 0..1.
// Note the argument order: colorsys takes h, L, s -- and Pillow passes
// hsl()'s third capture as L and its second as s, which is why hsl's
// arguments arrive here swapped.
def hls_to_rgb($h, $l, $s) {
    if ($s == 0.0) { return [$l, $l, $l]; }
    if ($l <= 0.5) { $m2 = $l * (1.0 + $s); }
    else { $m2 = $l + $s - ($l * $s); }
    $m1 = 2.0 * $l - $m2;
    return [_hlsV($m1, $m2, $h + $_ONE_THIRD),
            _hlsV($m1, $m2, $h),
            _hlsV($m1, $m2, $h - $_ONE_THIRD)];
}

// hsv_to_rgb($h, $s, $v) -> [r, g, b] as floats in 0..1.
def hsv_to_rgb($h, $s, $v) {
    if ($s == 0.0) { return [$v, $v, $v]; }
    $i = floor($h * 6.0);
    $f = ($h * 6.0) - $i;
    $p = $v * (1.0 - $s);
    $q = $v * (1.0 - $s * $f);
    $t = $v * (1.0 - $s * (1.0 - $f));
    $i = $i % 6;
    if ($i == 0) { return [$v, $t, $p]; }
    if ($i == 1) { return [$q, $v, $p]; }
    if ($i == 2) { return [$p, $v, $t]; }
    if ($i == 3) { return [$p, $q, $v]; }
    if ($i == 4) { return [$t, $p, $v]; }
    return [$v, $p, $q];
}

// rgb_to_hsv($r, $g, $b) -> [h, s, v] as floats in 0..1, from 0..1 inputs.
// getcolor needs it for mode HSV, and convert.b uses the same algorithm.
def rgb_to_hsv($r, $g, $b) {
    $maxc = max($r, $g, $b);
    $minc = min($r, $g, $b);
    $v = $maxc;
    if ($minc == $maxc) { return [0.0, 0.0, $v]; }
    $rangec = $maxc - $minc;
    $s = $rangec / $maxc;
    $rc = ($maxc - $r) / $rangec;
    $gc = ($maxc - $g) / $rangec;
    $bc = ($maxc - $b) / $rangec;
    if ($r == $maxc) { $h = $bc - $gc; }
    else {
        if ($g == $maxc) { $h = 2.0 + $rc - $bc; }
        else { $h = 4.0 + $gc - $rc; }
    }
    $h = ($h / 6.0) % 1.0;
    if ($h < 0) { $h = $h + 1.0; }
    return [$h, $s, $v];
}

// Pillow's int(f * 255 + 0.5) -- truncation after adding a half, which for
// non-negative values is round-half-up.
def _q255($f) { return floor($f * 255.0 + 0.5); }

// ─── getrgb ─────────────────────────────────────────────────────────────
//
// The order of the tests below is Pillow's order, and it matters: rgba() is
// checked AFTER hsl() and hsv(), and a name is checked before everything.
def getrgb($spec) {
    if (type($spec) != "string") {
        throw "ImageColor.getrgb: expected a colour string (got " + type($spec) + ")";
    }
    if (len($spec) > 100) { throw "ImageColor.getrgb: colour specifier is too long"; }
    $c = _lower($spec);

    // a named colour
    $named = $_NAMED[$c];
    if ($named != null) { return [$named[0], $named[1], $named[2]]; }

    $n = len($c);

    // #rgb / #rgba / #rrggbb / #rrggbbaa -- and nothing else
    if ($n > 0 && $c[0] == "#") {
        $body = substr($c, 1, $n - 1);
        if (_isHex($body)) {
            $bl = len($body);
            if ($bl == 3) {
                return [_hexVal($body[0]) * 17, _hexVal($body[1]) * 17, _hexVal($body[2]) * 17];
            }
            if ($bl == 4) {
                return [_hexVal($body[0]) * 17, _hexVal($body[1]) * 17,
                        _hexVal($body[2]) * 17, _hexVal($body[3]) * 17];
            }
            if ($bl == 6) {
                return [_hexVal($body[0]) * 16 + _hexVal($body[1]),
                        _hexVal($body[2]) * 16 + _hexVal($body[3]),
                        _hexVal($body[4]) * 16 + _hexVal($body[5])];
            }
            if ($bl == 8) {
                return [_hexVal($body[0]) * 16 + _hexVal($body[1]),
                        _hexVal($body[2]) * 16 + _hexVal($body[3]),
                        _hexVal($body[4]) * 16 + _hexVal($body[5]),
                        _hexVal($body[6]) * 16 + _hexVal($body[7])];
            }
        }
        throw "ImageColor.getrgb: unknown color specifier: '" + $spec + "'";
    }

    // rgb(r, g, b) -- integers
    $a = _args($c, "rgb");
    if ($a != null && len($a) == 3) {
        if (_isDigits($a[0]) && _isDigits($a[1]) && _isDigits($a[2])) {
            return [num($a[0]), num($a[1]), num($a[2])];
        }
        // rgb(r%, g%, b%)
        $p0 = _pct($a[0]);
        $p1 = _pct($a[1]);
        $p2 = _pct($a[2]);
        if ($p0 != null && $p1 != null && $p2 != null) {
            // Pillow: int((p * 255) / 100.0 + 0.5), and its regex here is
            // \d+ so a fractional percentage is NOT accepted.
            if (_isDigits(substr($a[0], 0, len($a[0]) - 1)) &&
                _isDigits(substr($a[1], 0, len($a[1]) - 1)) &&
                _isDigits(substr($a[2], 0, len($a[2]) - 1))) {
                return [floor(($p0 * 255) / 100.0 + 0.5),
                        floor(($p1 * 255) / 100.0 + 0.5),
                        floor(($p2 * 255) / 100.0 + 0.5)];
            }
        }
    }

    // hsl(h, s%, l%)
    $a = _args($c, "hsl");
    if ($a != null && len($a) == 3) {
        $s = _pct($a[1]);
        $l = _pct($a[2]);
        if (_isNum($a[0]) && $s != null && $l != null) {
            $f = hls_to_rgb(num($a[0]) / 360.0, $l / 100.0, $s / 100.0);
            return [_q255($f[0]), _q255($f[1]), _q255($f[2])];
        }
    }

    // hsv(h, s%, v%) and its alias hsb(...)
    $a = _args($c, "hsv");
    if ($a == null) { $a = _args($c, "hsb"); }
    if ($a != null && len($a) == 3) {
        $s = _pct($a[1]);
        $v = _pct($a[2]);
        if (_isNum($a[0]) && $s != null && $v != null) {
            $f = hsv_to_rgb(num($a[0]) / 360.0, $s / 100.0, $v / 100.0);
            return [_q255($f[0]), _q255($f[1]), _q255($f[2])];
        }
    }

    // rgba(r, g, b, a) -- last, as in Pillow
    $a = _args($c, "rgba");
    if ($a != null && len($a) == 4) {
        if (_isDigits($a[0]) && _isDigits($a[1]) && _isDigits($a[2]) && _isDigits($a[3])) {
            return [num($a[0]), num($a[1]), num($a[2]), num($a[3])];
        }
    }

    throw "ImageColor.getrgb: unknown color specifier: '" + $spec + "'";
}

// ─── getcolor ───────────────────────────────────────────────────────────
//
// Returns a NUMBER for a single-band result and a LIST otherwise, which is
// what Pillow returns, so a caller can compare against Pillow directly.
//
// It switches on Pillow's basemode, so CMYK, YCbCr and LAB all come back as
// three RGB values. That looks wrong and it is what Pillow does: getcolor
// parses a colour, it does not convert a colour space.
def getcolor($spec, $m) {
    $rgb = getrgb($spec);
    $alpha = 255;
    if (len($rgb) == 4) { $alpha = $rgb[3]; $rgb = [$rgb[0], $rgb[1], $rgb[2]]; }

    if ($m == "HSV") {
        // Pillow truncates here -- int(h * 255) -- unlike the +0.5 it uses in
        // getrgb. Matched deliberately.
        $hsv = rgb_to_hsv($rgb[0] / 255, $rgb[1] / 255, $rgb[2] / 255);
        return [floor($hsv[0] * 255), floor($hsv[1] * 255), floor($hsv[2] * 255)];
    }

    if (mode.basemode($m) == "L") {
        // The same L24 fixed-point luminance convert() uses, so getcolor and
        // convert agree on grey.
        $grey = floor(($rgb[0] * 19595 + $rgb[1] * 38470 + $rgb[2] * 7471 + 32768) / 65536);
        $value = [$grey];
    } else {
        $value = $rgb;
    }

    $lastCh = $m[len($m) - 1];
    if ($lastCh == "a") {
        // Premultiply, with Python's round(). band * alpha / 255 can never
        // land exactly on a half (that would need band * alpha == 255k + 127.5,
        // which is not an integer), so round-half-up is identical here to
        // Python's round-half-to-even.
        $pm = [];
        $i = 0;
        while ($i < len($value)) {
            push($pm, floor($value[$i] * $alpha / 255 + 0.5));
            $i = $i + 1;
        }
        $value = $pm;
    }
    if ($lastCh == "A" || $lastCh == "a") {
        push($value, $alpha);
        return $value;
    }
    if (len($value) == 1) { return $value[0]; }
    return $value;
}

// colormap() -> a copy of the name table, as name -> [r, g, b].
def colormap() {
    $out = {};
    for $k, $v in $_NAMED.items() { $out[$k] = [$v[0], $v[1], $v[2]]; }
    return $out;
}

// names() -> every known colour name, sorted.
def names() { return sort(keys($_NAMED)); }
