// ════════════════════════════════════════════════════════════════════════
//  pillow/ops.b — ImageOps: look-up tables, and the six that are geometry.
//
//  Most of ImageOps is a 256-entry table applied with point(), which is one
//  nd_take. The interesting part of each is the table's INTEGER arithmetic,
//  reproduced statement for statement because every one has a quirk:
//
//    * autocontrast's cutoff removes pixels from the histogram ends with
//      int(n * cutoff // 100) and maps with int(ix * scale + offset),
//      clamped after. If hi <= lo it returns the identity rather than
//      dividing by zero.
//    * equalize's step is (sum(histo) - histo[-1]) // 255 where `histo` is
//      the histogram WITH ZERO BINS REMOVED -- so the last NON-ZERO bin is
//      subtracted, not bin 255. Then n starts at step // 2 and each entry
//      is n // step. Getting this wrong gives a plausible, wrong,
//      slightly-too-bright image.
//    * colorize interpolates with i * (hi - lo) // len(range_map), floor
//      division per channel, over segments whose lengths differ.
//    * _lut itself supports only L and RGB: mode P raises
//      NotImplementedError("mode P support coming soon") and everything
//      else raises "not supported for mode X". So a P image through
//      autocontrast raises in Pillow, and raises here, with the message.
//
//  The six that are geometry -- contain, cover, fit, pad, scale, deform --
//  are composition over P3's resize and transform. Their content is the
//  ROUNDING of the target size (Python's round, which is round-half-to-EVEN,
//  not floor(x + 0.5)) and the centering arithmetic, because an off-by-one
//  in a thumbnail size is a visible difference.
//
//  docs/pillow-architecture.md §7.3.
// ════════════════════════════════════════════════════════════════════════

include "./mode.b" as mode;
include "./pxbuf.b" as pxbuf;
include "./color.b" as colour;
include "./image.b" as image;
include "./exif.b" as exif;
include "./transform.b" as xform;

$BICUBIC = 3;
$BILINEAR = 2;

// _round($x) -> Python's round(): halves go to the EVEN integer. Used for
// every target size, because round(2.5) is 2 in Python and 3 in C.
def _round($x) {
    $f = floor($x);
    $d = $x - $f;
    if ($d > 0.5) { return $f + 1; }
    if ($d < 0.5) { return $f; }
    if ($f % 2 == 0) { return $f; }
    return $f + 1;
}

// _border($b) -> [left, top, right, bottom] from an int, a pair or a quad.
def _border($b) {
    if (type($b) == "list") {
        if (len($b) == 2) { return [$b[0], $b[1], $b[0], $b[1]]; }
        if (len($b) == 4) { return [$b[0], $b[1], $b[2], $b[3]]; }
        throw "border must be an integer, or a tuple of two or four elements";
    }
    if ($b == null) { return [0, 0, 0, 0]; }
    return [$b, $b, $b, $b];
}

def _colourOf($c, $m) {
    if (type($c) == "string") { return colour.getcolor($c, $m); }
    return $c;
}

// _lut($im, $lut) -> point(), with Pillow's mode restrictions.
def _lut($im, $lut) {
    if ($im.mode == "P") {
        throw "NotImplementedError: mode P support coming soon";
    }
    if ($im.mode == "L") { return $im.point($lut, null); }
    if ($im.mode == "RGB") {
        if (len($lut) == 256) {
            $three = [];
            $k = 0;
            while ($k < 3) {
                $i = 0;
                while ($i < 256) { push($three, $lut[$i]); $i = $i + 1; }
                $k = $k + 1;
            }
            return $im.point($three, null);
        }
        return $im.point($lut, null);
    }
    throw "not supported for mode " + $im.mode;
}

// ─── The look-up table operations ───────────────────────────────────────

def autocontrast($im, $cutoff, $ignore, $mask, $preserve_tone) {
    if ($preserve_tone == true) { $h = $im.convert("L", null, null).histogram($mask); }
    else { $h = $im.histogram($mask); }
    $cut0 = 0;
    $cut1 = 0;
    if ($cutoff != null) {
        if (type($cutoff) == "list") { $cut0 = $cutoff[0]; $cut1 = $cutoff[1]; }
        else { $cut0 = $cutoff; $cut1 = $cutoff; }
    }

    $lut = [];
    $layer = 0;
    while ($layer < len($h)) {
        // This band's 256 bins, as a mutable copy.
        $b = [];
        $i = 0;
        while ($i < 256) { push($b, $h[$layer + $i]); $i = $i + 1; }
        if ($ignore != null) {
            if (type($ignore) == "list") {
                $i = 0;
                while ($i < len($ignore)) { $b[$ignore[$i]] = 0; $i = $i + 1; }
            } else { $b[$ignore] = 0; }
        }
        if ($cut0 != 0 || $cut1 != 0) {
            $n = 0;
            $i = 0;
            while ($i < 256) { $n = $n + $b[$i]; $i = $i + 1; }
            $cut = floor($n * $cut0 / 100);
            $i = 0;
            while ($i < 256) {
                if ($cut > $b[$i]) { $cut = $cut - $b[$i]; $b[$i] = 0; }
                else { $b[$i] = $b[$i] - $cut; $cut = 0; }
                if ($cut <= 0) { break; }
                $i = $i + 1;
            }
            $cut = floor($n * $cut1 / 100);
            $i = 255;
            while ($i >= 0) {
                if ($cut > $b[$i]) { $cut = $cut - $b[$i]; $b[$i] = 0; }
                else { $b[$i] = $b[$i] - $cut; $cut = 0; }
                if ($cut <= 0) { break; }
                $i = $i - 1;
            }
        }
        $lo = 0;
        $i = 0;
        while ($i < 256) { if ($b[$i] != 0) { $lo = $i; break; } $i = $i + 1; }
        if ($i == 256) { $lo = 255; }
        $hi = 0;
        $i = 255;
        while ($i >= 0) { if ($b[$i] != 0) { $hi = $i; break; } $i = $i - 1; }
        if ($i < 0) { $hi = 0; }
        if ($hi <= $lo) {
            $i = 0;
            while ($i < 256) { push($lut, $i); $i = $i + 1; }
        } else {
            $scale = 255.0 / ($hi - $lo);
            $offset = 0 - $lo * $scale;
            $i = 0;
            while ($i < 256) {
                $v = floor($i * $scale + $offset);
                if ($i * $scale + $offset < 0) { $v = 0 - floor(0 - ($i * $scale + $offset)); }
                if ($v < 0) { $v = 0; }
                if ($v > 255) { $v = 255; }
                push($lut, $v);
                $i = $i + 1;
            }
        }
        $layer = $layer + 256;
    }
    return _lut($im, $lut);
}

def equalize($im, $mask) {
    $src = $im;
    if ($im.mode == "P") { $src = $im.convert("RGB", null, null); }
    $h = $src.histogram($mask);
    $lut = [];
    $b = 0;
    while ($b < len($h)) {
        // `histo` is the band's histogram WITH THE ZERO BINS REMOVED, which
        // is what makes histo[-1] the last non-empty bin rather than bin 255.
        $histo = [];
        $i = 0;
        while ($i < 256) {
            if ($h[$b + $i] != 0) { push($histo, $h[$b + $i]); }
            $i = $i + 1;
        }
        if (len($histo) <= 1) {
            $i = 0;
            while ($i < 256) { push($lut, $i); $i = $i + 1; }
        } else {
            $tot = 0;
            $i = 0;
            while ($i < len($histo)) { $tot = $tot + $histo[$i]; $i = $i + 1; }
            $step = floor(($tot - $histo[len($histo) - 1]) / 255);
            if ($step == 0) {
                $i = 0;
                while ($i < 256) { push($lut, $i); $i = $i + 1; }
            } else {
                $n = floor($step / 2);
                $i = 0;
                while ($i < 256) {
                    push($lut, floor($n / $step));
                    $n = $n + $h[$i + $b];
                    $i = $i + 1;
                }
            }
        }
        $b = $b + 256;
    }
    return _lut($src, $lut);
}

def invert($im) {
    $lut = [];
    $i = 255;
    while ($i >= 0) { push($lut, $i); $i = $i - 1; }
    if ($im.mode == "1") { return $im.point($lut, null); }
    return _lut($im, $lut);
}

def posterize($im, $bits) {
    if ($bits < 1 || $bits > 8) {
        throw "ImageOps.posterize: bits must be 1..8, got " + str($bits);
    }
    $maskv = 256 - pow(2, 8 - $bits);     // ~(2^(8-bits) - 1), over a byte
    $lut = [];
    $i = 0;
    while ($i < 256) { push($lut, band($i, $maskv)); $i = $i + 1; }
    return _lut($im, $lut);
}

def solarize($im, $threshold) {
    $t = $threshold;
    if ($t == null) { $t = 128; }
    $lut = [];
    $i = 0;
    while ($i < 256) {
        if ($i < $t) { push($lut, $i); } else { push($lut, 255 - $i); }
        $i = $i + 1;
    }
    return _lut($im, $lut);
}

// colorize($im, $black, $white, $mid, $blackpoint, $whitepoint, $midpoint)
//
// Three segments, each interpolated with floor division by its OWN length,
// which is why the two-colour and three-colour forms are written out
// separately rather than sharing a loop.
def colorize($im, $black, $white, $mid, $blackpoint, $whitepoint, $midpoint) {
    if ($im.mode != "L") { throw "ImageOps.colorize: the image must have mode L"; }
    $bp = $blackpoint;
    if ($bp == null) { $bp = 0; }
    $wp = $whitepoint;
    if ($wp == null) { $wp = 255; }
    $mp = $midpoint;
    if ($mp == null) { $mp = 127; }
    if ($mid == null) {
        if (!(0 <= $bp && $bp <= $wp && $wp <= 255)) {
            throw "ImageOps.colorize: need 0 <= blackpoint <= whitepoint <= 255";
        }
    } else {
        if (!(0 <= $bp && $bp <= $mp && $mp <= $wp && $wp <= 255)) {
            throw "ImageOps.colorize: need 0 <= blackpoint <= midpoint <= whitepoint <= 255";
        }
    }
    $kb = _colourOf($black, "RGB");
    $kw = _colourOf($white, "RGB");
    $km = null;
    if ($mid != null) { $km = _colourOf($mid, "RGB"); }

    $red = [];
    $green = [];
    $blue = [];
    $i = 0;
    while ($i < $bp) {
        push($red, $kb[0]); push($green, $kb[1]); push($blue, $kb[2]);
        $i = $i + 1;
    }
    if ($km == null) {
        $n = $wp - $bp;
        $i = 0;
        while ($i < $n) {
            push($red, $kb[0] + floor($i * ($kw[0] - $kb[0]) / $n));
            push($green, $kb[1] + floor($i * ($kw[1] - $kb[1]) / $n));
            push($blue, $kb[2] + floor($i * ($kw[2] - $kb[2]) / $n));
            $i = $i + 1;
        }
    } else {
        $n1 = $mp - $bp;
        $n2 = $wp - $mp;
        $i = 0;
        while ($i < $n1) {
            push($red, $kb[0] + floor($i * ($km[0] - $kb[0]) / $n1));
            push($green, $kb[1] + floor($i * ($km[1] - $kb[1]) / $n1));
            push($blue, $kb[2] + floor($i * ($km[2] - $kb[2]) / $n1));
            $i = $i + 1;
        }
        $i = 0;
        while ($i < $n2) {
            push($red, $km[0] + floor($i * ($kw[0] - $km[0]) / $n2));
            push($green, $km[1] + floor($i * ($kw[1] - $km[1]) / $n2));
            push($blue, $km[2] + floor($i * ($kw[2] - $km[2]) / $n2));
            $i = $i + 1;
        }
    }
    $i = 0;
    while ($i < 256 - $wp) {
        push($red, $kw[0]); push($green, $kw[1]); push($blue, $kw[2]);
        $i = $i + 1;
    }
    $all = [];
    $i = 0;
    while ($i < len($red)) { push($all, $red[$i]); $i = $i + 1; }
    $i = 0;
    while ($i < len($green)) { push($all, $green[$i]); $i = $i + 1; }
    $i = 0;
    while ($i < len($blue)) { push($all, $blue[$i]); $i = $i + 1; }
    return _lut($im.convert("RGB", null, null), $all);
}

// ─── The geometric ones ─────────────────────────────────────────────────

def grayscale($im) { return $im.convert("L", null, null); }
def flip($im)      { return $im.transpose(1); }         // FLIP_TOP_BOTTOM
def mirror($im)    { return $im.transpose(0); }         // FLIP_LEFT_RIGHT

def crop($im, $border) {
    $b = _border($border);
    return $im.crop([$b[0], $b[1], $im.width - $b[2], $im.height - $b[3]]);
}

def expand($im, $border, $fill) {
    $b = _border($border);
    $w = $b[0] + $im.width + $b[2];
    $h = $b[1] + $im.height + $b[3];
    $f = $fill;
    if ($f == null) { $f = 0; }
    $out = image.new_image($im.mode, [$w, $h], _colourOf($f, $im.mode));
    if ($im.palette != null) { $out.palette = $im.palette; }
    $out.paste($im, [$b[0], $b[1]], null);
    return $out;
}

def contain($im, $size, $method) {
    $m = $method;
    if ($m == null) { $m = $BICUBIC; }
    $imRatio = $im.width / $im.height;
    $destRatio = $size[0] / $size[1];
    $out = [$size[0], $size[1]];
    if ($imRatio != $destRatio) {
        if ($imRatio > $destRatio) {
            $nh = _round($im.height / $im.width * $size[0]);
            if ($nh != $size[1]) { $out = [$size[0], $nh]; }
        } else {
            $nw = _round($im.width / $im.height * $size[1]);
            if ($nw != $size[0]) { $out = [$nw, $size[1]]; }
        }
    }
    return $im.resize($out, $m, null, null);
}

def cover($im, $size, $method) {
    $m = $method;
    if ($m == null) { $m = $BICUBIC; }
    $imRatio = $im.width / $im.height;
    $destRatio = $size[0] / $size[1];
    $out = [$size[0], $size[1]];
    if ($imRatio != $destRatio) {
        if ($imRatio < $destRatio) {
            $nh = _round($im.height / $im.width * $size[0]);
            if ($nh != $size[1]) { $out = [$size[0], $nh]; }
        } else {
            $nw = _round($im.width / $im.height * $size[1]);
            if ($nw != $size[0]) { $out = [$nw, $size[1]]; }
        }
    }
    return $im.resize($out, $m, null, null);
}

def pad($im, $size, $method, $colour, $centering) {
    $c = $centering;
    if ($c == null) { $c = [0.5, 0.5]; }
    $resized = contain($im, $size, $method);
    if ($resized.width == $size[0] && $resized.height == $size[1]) { return $resized; }
    $out = image.new_image($im.mode, $size, _colourOf($colour, $im.mode));
    if ($resized.palette != null) { $out.palette = $resized.palette; }
    if ($resized.width != $size[0]) {
        $x = _round(($size[0] - $resized.width) * max(0, min($c[0], 1)));
        $out.paste($resized, [$x, 0], null);
    } else {
        $y = _round(($size[1] - $resized.height) * max(0, min($c[1], 1)));
        $out.paste($resized, [0, $y], null);
    }
    return $out;
}

def scale($im, $factor, $resample) {
    $m = $resample;
    if ($m == null) { $m = $BICUBIC; }
    if ($factor == 1) { return $im.copy(); }
    if ($factor <= 0) { throw "the factor must be greater than 0"; }
    return $im.resize([_round($factor * $im.width), _round($factor * $im.height)],
                      $m, null, null);
}

// fit($im, $size, $method, $bleed, $centering) -> Kevin Cazabon's crop-and-
// resize, which is a resize with a FLOAT box rather than a crop followed by
// a resize: the box's fractional edges are part of the result.
def fit($im, $size, $method, $bleed, $centering) {
    $m = $method;
    if ($m == null) { $m = $BICUBIC; }
    $c = $centering;
    if ($c == null) { $c = [0.5, 0.5]; }
    $cx = $c[0];
    $cy = $c[1];
    if (!(0.0 <= $cx && $cx <= 1.0)) { $cx = 0.5; }
    if (!(0.0 <= $cy && $cy <= 1.0)) { $cy = 0.5; }
    $bl = $bleed;
    if ($bl == null) { $bl = 0.0; }
    if (!(0.0 <= $bl && $bl < 0.5)) { $bl = 0.0; }

    $bpx = $bl * $im.width;
    $bpy = $bl * $im.height;
    $lw = $im.width - $bpx * 2;
    $lh = $im.height - $bpy * 2;
    $liveRatio = $lw / $lh;
    $outRatio = $size[0] / $size[1];
    if ($liveRatio == $outRatio) { $cw = $lw; $ch = $lh; }
    else if ($liveRatio >= $outRatio) { $cw = $outRatio * $lh; $ch = $lh; }
    else { $cw = $lw; $ch = $lw / $outRatio; }

    $cl = $bpx + ($lw - $cw) * $cx;
    $ct = $bpy + ($lh - $ch) * $cy;
    return $im.resize($size, $m, [$cl, $ct, $cl + $cw, $ct + $ch], null);
}

// deform($im, $deformer, $resample) -> transform(MESH) with the mesh the
// deformer returns. A deformer is anything with a getmesh(image) that
// returns Pillow's list of [box, quad] pairs, or a Bantu function of the
// image returning the same.
def deform($im, $deformer, $resample) {
    $m = $resample;
    if ($m == null) { $m = $BILINEAR; }
    if (type($deformer) == "function") { $mesh = $deformer($im); }
    else if (type($deformer) == "instance") { $mesh = $deformer.getmesh($im); }
    else if (type($deformer) == "dict" && $deformer.getmesh != null) {
        $fn = $deformer.getmesh;
        $mesh = $fn($im);
    } else { throw "ImageOps.deform: the deformer needs a getmesh(image)"; }
    return $im.transform($im.size, 4, $mesh, $m, null);     // 4 = MESH
}

// exif_transpose($im) -> the orientation applied and the tag removed.
// It lives in ImageOps in Pillow and is implemented in exif.b, because it
// is the only operation here that reads metadata.
def exif_transpose($im) { return exif.exif_transpose($im); }
