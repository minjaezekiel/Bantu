// ════════════════════════════════════════════════════════════════════════
//  pillow/draw.b — ImageDraw: vector drawing onto a raster.
//
//      include "./pillow.b" as pillow;
//      $im = pillow.new("RGB", [200, 120], "white");
//      $d  = pillow.Draw($im, null);
//      $d.rectangle([10, 10, 60, 40], "cornflowerblue", "black", 2);
//      $d.ellipse([80, 10, 150, 90], "gold", null, 1);
//      $d.line([[0, 0], [199, 119]], "red", 3, null);
//
//  WHAT THIS FILE IS. Every rule in here is Draw.c's, read out of the C and
//  reproduced — not reasoned from the documentation. Phase P7's research
//  file (pillow-suite/P7-NOTES.md) lists them; the ones that a textbook
//  implementation gets wrong by default, and that the differential tier
//  would otherwise find one at a time:
//
//    * `line` NEVER DRAWS ITS ENDPOINT. A segment emits exactly
//      max(|dx|, |dy|) pixels. The Python layer then draws the final point
//      of a polyline separately, which is why `draw.line` looks closed and
//      `polygon(width=1)` draws each vertex exactly once.
//    * `dx > dy` STRICTLY selects the x-major loop, so a 45-degree tie goes
//      to the y-major one, and the error test is `e >= 0`, so a tie steps
//      the minor axis.
//    * `hline` is inclusive of both ends and NEVER SWAPS them: x0 > x1
//      draws nothing. It clamps x0 up to 0 and x1 down to width - 1.
//    * `rectangle` normalises only y. Filled ignores `width` and clamps y1
//      to ysize (not ysize - 1); unfilled promotes width 0 to 1 and expands
//      INWARD.
//    * the polygon filler computes its edge crossings IN FLOAT32, counts a
//      crossing at an edge's ymax twice, and runs a corner-join fix-up
//      against earlier edges. Spans are drawn as
//      hline(ROUND_UP(xs[p]), y, ROUND_DOWN(xs[p+1])), where ROUND_UP is
//      half-away-from-zero and ROUND_DOWN half-toward-zero.
//    * the ellipse family is an integer quarter-tracer on a grid of step 2,
//      and its third candidate DOES NOT REFRESH the running minimum. That
//      is reproduced as written: it is what Pillow's ellipses look like.
//    * an ellipse outline of width 0 draws nothing, unlike a rectangle,
//      which promotes 0 to 1.
//    * filling with an "L" matte onto a mode with alpha writes the colour
//      bands at FULL strength wherever the destination alpha is 0, because
//      Paste.c's correction factor is an integer division that collapses to
//      0 or 1. Reproduced: see _fillMaskL.
//
//  NO ANTI-ALIASING, because Pillow's ImageDraw has none (decision PW8).
//  Pillow's documented idiom — draw at 4x and resize down — works here.
//
//  THE PIXEL BUDGET (rule PW2). Nothing in this file loops over pixels:
//
//    * a horizontal span is one array fill per band over a crop VIEW, so a
//      1920-pixel span costs the same as a 3-pixel one.
//    * `line` is decomposed into its RUNS rather than its pixels: a
//      Bresenham segment is min(|dx|, |dy|) + 1 spans of constant minor
//      coordinate, and the run boundaries are a closed form,
//      i_k = ceil(major * (2k - 1) / (2 * minor)), derived from the error
//      recurrence rather than stepped. A near-horizontal 1080p line is two
//      array calls; a 45-degree one is unavoidably one call per pixel,
//      because it touches that many pixels and so does Pillow.
//    * `polygon`, `ellipse`, `arc`, `chord` and `pieslice` loop over ROWS
//      and over spans within a row. A filled 1080p polygon is ~1,080
//      iterations.
//    * `floodfill` is a scanline span fill over a fillability mask computed
//      in one array pass, so its loop is over spans, not pixels.
//
//  STATE IS A CLASS, not a dict, because `Draw()` lives in this file and so
//  `new PillowDraw(...)` parses here — the same arrangement image.b uses.
//  A caller therefore writes `$d.ellipse(...)`, exactly as in Pillow.
//
//  BANTU, TWICE OVER (decision PW14): a list handed to a function is
//  COPIED, so every piece of mutable state crossing a call boundary in here
//  travels inside a dict — the edge table, the span cursor, the tracer
//  states. A list read out of a dict, by contrast, is a reference, which is
//  what makes that work.
// ════════════════════════════════════════════════════════════════════════

include "./mode.b" as mode;
include "./pxbuf.b" as pxbuf;
include "./color.b" as color;
include "./palette.b" as pal;
include "./font.b" as font;

// ════════════════════════════════════════════════════════════════════════
//  Arithmetic that has to match the C exactly
// ════════════════════════════════════════════════════════════════════════

// Draw.c:
//   ROUND_UP(f)   = f >= 0 ? floor(f + 0.5) : -floor(|f| + 0.5)
//   ROUND_DOWN(f) = f >= 0 ? ceil(f - 0.5)  : -ceil(|f| - 0.5)
// which is "round half AWAY from zero" and "round half TOWARD zero". Bantu's
// round() is already the former, so ROUND_UP is round(); ROUND_DOWN is not
// any builtin and is written out.
def ROUND_UP($f) { return round($f); }

def ROUND_DOWN($f) {
    if ($f >= 0) { return ceil($f - 0.5); }
    return 0 - ceil((0 - $f) - 0.5);
}

// A C `(int)` cast on a double: truncation toward zero, NOT floor. Every
// coordinate Pillow's ImageDraw accepts goes through one of these, so
// line([-1.5, ...]) starts at -1, not -2.
def _int($v) {
    if (type($v) != "number") { throw "ImageDraw: expected a number, got " + type($v); }
    return trunc($v);
}

// CLIP8 from ImagingUtils.h: <= 0 -> 0, >= 256 -> 255, else itself.
def _clip8($v) {
    $n = trunc($v);
    if ($n <= 0) { return 0; }
    if ($n >= 256) { return 255; }
    return $n;
}

// _i32($v) -> v truncated into a signed 32-bit integer, as `itmp = r` does.
def _i32($v) {
    $n = trunc($v);
    $m = $n % 4294967296;
    if ($m < 0) { $m = $m + 4294967296; }
    if ($m >= 2147483648) { $m = $m - 4294967296; }
    return $m;
}

// _u16($v) -> the low sixteen bits, as getink's SPECIAL case leaves them.
def _u16($v) {
    $m = trunc($v) % 65536;
    if ($m < 0) { $m = $m + 65536; }
    return $m;
}

// _f32($v) -> the nearest float32 to v, as an f64.
//
// WHY THIS EXISTS. The polygon filler's edge slope and every crossing it
// computes are `float` in Draw.c, and the crossing is then rounded to an
// integer. Our arithmetic is f64, so a value that lands exactly on .5 in
// float32 can land just below it in f64 and round the other way — one pixel
// wrong on one row of one polygon, which is exactly the class of difference
// the differential tier exists to catch. pxbuf.to_f32 does this for arrays,
// through four bytes; this is the scalar form, and it is pure arithmetic
// rather than two array calls because it runs per edge per row.
//
// Ties go to EVEN, as IEEE 754 requires. Values beyond the float32 exponent
// range are returned unchanged: no coordinate in an image reaches 1e38, and
// pretending to model overflow here would be a lie with no test behind it.
def _f32($v) {
    if ($v == 0) { return 0; }
    $s = 1;
    $m = $v;
    if ($m < 0) { $s = 0 - 1; $m = 0 - $m; }
    if ($m >= 3.4028235677973366e38 || $m < 1.1754943508222875e-38) { return $v; }
    $e = 0;
    while ($m >= 2) { $m = $m / 2; $e = $e + 1; }
    while ($m < 1)  { $m = $m * 2; $e = $e - 1; }
    // m is now in [1, 2); float32 keeps 24 significant bits, so 23 after the
    // leading one.
    $q = $m * 8388608;
    $fl = floor($q);
    $fr = $q - $fl;
    if ($fr > 0.5) { $fl = $fl + 1; }
    else if ($fr == 0.5 && $fl % 2 != 0) { $fl = $fl + 1; }
    $r = $fl / 8388608;
    if ($r >= 2) { $r = $r / 2; $e = $e + 1; }
    return $s * $r * pow(2, $e);
}

// The float32 crossing of an edge at row y: (y - y0) * dx + x0, every step
// rounded to float32 as the C does.
def _cross($y, $y0, $dx, $x0) {
    return _f32(_f32(($y - $y0) * $dx) + $x0);
}

// ceil(a / b) for b > 0, without a division that could round across an
// integer. Our magnitudes are coordinates, so the quotient is a rational
// with a small denominator and the f64 division is exact to well within the
// gap to the nearest integer.
def _ceilDiv($a, $b) { return 0 - floor((0 - $a) / $b); }


// ════════════════════════════════════════════════════════════════════════
//  Coordinates
// ════════════════════════════════════════════════════════════════════════

// _flatten($xy) -> [x, y, x, y, ...] as plain numbers, from any of the three
// shapes Pillow's PyPath_Flatten accepts: a flat list, a list of pairs, or
// an ImagePath.
def _flatten($xy, $who) {
    if (type($xy) == "dict") {
        if ($xy.xy == null) { throw $who + ": a path must have an xy member"; }
        return _flatten($xy.xy, $who);
    }
    if (type($xy) != "list") { throw $who + ": coordinates must be a list"; }
    $out = [];
    if (len($xy) > 0 && type($xy[0]) == "list") {
        $i = 0;
        while ($i < len($xy)) {
            if (type($xy[$i]) != "list" || len($xy[$i]) < 2) {
                throw $who + ": point " + str($i) + " needs an x and a y";
            }
            push($out, $xy[$i][0]);
            push($out, $xy[$i][1]);
            $i = $i + 1;
        }
        return $out;
    }
    if (len($xy) % 2 != 0) {
        throw $who + ": a flat coordinate list must have an even length, got " +
              str(len($xy));
    }
    $i = 0;
    while ($i < len($xy)) {
        if (type($xy[$i]) != "number") {
            throw $who + ": coordinate " + str($i) + " is not a number";
        }
        push($out, $xy[$i]);
        $i = $i + 1;
    }
    return $out;
}

// _box($xy, $who) -> [x0, y0, x1, y1] as C ints, with Pillow's two
// ValueErrors for a reversed box. Pillow 12 raises here rather than drawing
// nothing, so a reversed rectangle is a caller's bug, not a silent no-op.
def _box($xy, $who) {
    $f = _flatten($xy, $who);
    if (len($f) != 4) { throw $who + ": expected two coordinates, got " + str(floor(len($f) / 2)); }
    if ($f[2] < $f[0]) { throw $who + ": x1 must be greater than or equal to x0"; }
    if ($f[3] < $f[1]) { throw $who + ": y1 must be greater than or equal to y0"; }
    return [_int($f[0]), _int($f[1]), _int($f[2]), _int($f[3])];
}


// ════════════════════════════════════════════════════════════════════════
//  Ink
// ════════════════════════════════════════════════════════════════════════
//
//  Pillow packs an ink into four bytes (getink in _imaging.c) and then
//  writes either the low byte (8-bit single-band modes), the low two bytes
//  (I;16), the whole INT32 (I, F) or all four bytes at once (everything
//  else). Our store is one i64 or f64 per BAND, so the four bytes are kept
//  as a band list instead — plus the alpha byte, which RGBA-on-RGB blending
//  needs even though RGB has only three bands.
//
//  THE BAND MAPPING IS NOT THE OBVIOUS ONE for two-band modes. Pillow's LA,
//  La and PA are four bytes internally with the alpha at byte 3, so getink
//  fills bytes [v, v, v, a] and the image reads bands from bytes 0 and 3.
//  Verified against Pillow: ImageDraw.point in LA with (11, 22) gives the
//  pixel (11, 22), from the ink 0x16_0B_0B_0B.

// _inkBands($m, $spec) -> {"v": [one value per band], "a": the alpha byte or
// null, "key": an identity string}.
//
// `key` exists because ImageDraw compares `ink != fill_ink` to decide
// whether an outline needs a second pass over a filled shape, and that
// comparison is on the packed int in Pillow. Comparing the band list is the
// same test: two specs that pack alike map to the same bands.
def _inkBands($m, $spec) {
    $nb = mode.bands($m);

    if ($m == "F") {
        $v = $spec;
        if (type($v) == "list") {
            if (len($v) != 1) { throw "ImageDraw: an F ink must be a number"; }
            $v = $v[0];
        }
        if (type($v) != "number") { throw "ImageDraw: an F ink must be a number"; }
        $f = _f32($v);
        return {"v": [$f], "a": null, "key": "F:" + str($f)};
    }

    if ($m == "I") {
        $v = $spec;
        if (type($v) == "list") {
            if (len($v) != 1) { throw "ImageDraw: an I ink must be an integer"; }
            $v = $v[0];
        }
        if (type($v) != "number") { throw "ImageDraw: an I ink must be an integer"; }
        $n = _i32($v);
        return {"v": [$n], "a": null, "key": "I:" + str($n)};
    }

    if (mode.family($m) == "int") {
        // I;16 and its byte-order twins: getink's SPECIAL case keeps two bytes.
        $v = $spec;
        if (type($v) == "list") {
            if (len($v) != 1) { throw "ImageDraw: an " + $m + " ink must be an integer"; }
            $v = $v[0];
        }
        if (type($v) != "number") { throw "ImageDraw: an " + $m + " ink must be an integer"; }
        $n = _u16($v);
        return {"v": [$n], "a": null, "key": "U16:" + str($n)};
    }

    // The 8-bit family.
    $b = [0, 0, 0, 0];
    if ($nb == 1) {
        $v = $spec;
        if (type($v) == "list") {
            if (len($v) != 1) {
                throw "ImageDraw: a " + $m + " ink must be an integer or a " +
                      "single-element list";
            }
            $v = $v[0];
        }
        if (type($v) != "number") { throw "ImageDraw: a " + $m + " ink must be an integer"; }
        $b[0] = _clip8($v);
        return {"v": [$b[0]], "a": null, "key": "B1:" + str($b[0])};
    }

    if (type($spec) == "number") {
        // getink's documented compatibility case: a bare integer is ABGR.
        $n = trunc($spec);
        $b[3] = _shiftByte($n, 24);
        $b[2] = _shiftByte($n, 16);
        $b[1] = _shiftByte($n, 8);
        $b[0] = _shiftByte($n, 0);
    } else if (type($spec) == "list") {
        $n = len($spec);
        if ($nb == 2) {
            if ($n != 1 && $n != 2) {
                throw "ImageDraw: a " + $m + " ink must be an integer, or a list of " +
                      "one or two elements";
            }
            $v = $spec[0];
            $a = 255;
            if ($n == 2) { $a = $spec[1]; }
            $b[0] = _clip8($v);
            $b[1] = $b[0];
            $b[2] = $b[0];
            $b[3] = _clip8($a);
        } else {
            if ($n != 3 && $n != 4) {
                throw "ImageDraw: a " + $m + " ink must be an integer, or a list of " +
                      "one, three or four elements";
            }
            $b[0] = _clip8($spec[0]);
            $b[1] = _clip8($spec[1]);
            $b[2] = _clip8($spec[2]);
            $b[3] = 255;
            if ($n == 4) { $b[3] = _clip8($spec[3]); }
        }
    } else {
        throw "ImageDraw: a " + $m + " ink must be an integer or a list";
    }

    // Byte 3 is the alpha slot, so a two-band mode reads bands 0 and 3.
    if ($nb == 2) {
        return {"v": [$b[0], $b[3]], "a": $b[3],
                "key": "B:" + str($b[0]) + "," + str($b[1]) + "," + str($b[2]) + "," + str($b[3])};
    }
    $v = [];
    $i = 0;
    while ($i < $nb) { push($v, $b[$i]); $i = $i + 1; }
    return {"v": $v, "a": $b[3],
            "key": "B:" + str($b[0]) + "," + str($b[1]) + "," + str($b[2]) + "," + str($b[3])};
}

// (n >> k) & 255, for a value that may be negative.
def _shiftByte($n, $k) {
    $m = $n;
    $i = 0;
    while ($i < $k) { $m = floor($m / 2); $i = $i + 1; }
    $r = $m % 256;
    if ($r < 0) { $r = $r + 256; }
    return $r;
}


// ════════════════════════════════════════════════════════════════════════
//  The writers
// ════════════════════════════════════════════════════════════════════════
//
//  A context carries only what the writers need: the pixel buffer, the size,
//  the band count, the ink mode and whether we are blending RGBA onto RGB.

def _ctx($px, $imMode, $drawMode, $blend) {
    $d = pxbuf.dims($px);
    return {"px": $px, "h": $d[0], "w": $d[1], "nb": $d[2],
            "m": $imMode, "dm": $drawMode, "blend": $blend};
}

// BLEND(mask, in1, in2) from ImagingUtils.h, over an array:
//
//     DIV255(in1 * (255 - mask) + in2 * mask)
//
// with DIV255(a) = ((((a + 128) >> 8) + (a + 128)) >> 8), which is the
// standard exact trick for ROUND(a / 255) and not the (a + 127) / 255 that
// the obvious reading gives -- the two differ for about one value in 256.
//
// WRITTEN AS round(a / 255), which is the same number. DIV255's shift form is
// exact only while a <= 65,025, which covers every product of two bytes and
// so every 8-bit mode -- the first divergence is at a = 65,663, where the
// shifts give 257 and the true rounding 258. Mode I and I;16 reach far past
// that, so a faithful transcription of the shifts would make a 16-bit matte
// of 255 write 39,999 instead of 40,000. The rounding form is identical on
// every 8-bit input (checked over the whole range) and right on the rest.
//
// F has no integer to round into at all, so it interpolates exactly.
def _blendArr($dst, $maskArr, $inkVal) {
    if (nd_dtype($dst) == "f64") {
        return nd_add($dst,
                      nd_divide(nd_multiply(nd_subtract($inkVal, $dst), $maskArr), 255.0));
    }
    $t = nd_add(nd_multiply($dst, nd_subtract(255, $maskArr)),
                nd_multiply($maskArr, $inkVal));
    return nd_floor(nd_add(nd_divide($t, 255.0), 0.5));
}

// _writeSpan($c, $dst, $ink, $msk) -> the span written, honouring the
// context's blend flag and an optional mode-"1" mask view.
def _writeSpan($c, $dst, $ink, $msk) {
    $i = 0;
    while ($i < $c.nb) {
        $band = pxbuf.band($dst, $i);
        $new = null;
        if ($c.blend) {
            $new = _blendArr($band, nd_add(nd_multiply($band, 0), $ink.a), $ink.v[$i]);
        } else {
            $new = nd_add(nd_multiply($band, 0), $ink.v[$i]);
        }
        if ($msk != null) {
            $new = nd_where(nd_not_equal($msk, 0), $new, $band);
        }
        nd_add($new, 0, $band);
        $i = $i + 1;
    }
    return null;
}

// hline($c, $x0, $y, $x1, $ink, $mask) -> Draw.c's hline8/hline32/hline32rgba.
// INCLUSIVE of both ends, and it never swaps them.
def _hline($c, $x0, $y, $x1, $ink, $mask) {
    if ($y < 0 || $y >= $c.h) { return null; }
    if ($x0 < 0) { $x0 = 0; } else if ($x0 >= $c.w) { return null; }
    if ($x1 < 0) { return null; } else if ($x1 >= $c.w) { $x1 = $c.w - 1; }
    if ($x0 > $x1) { return null; }
    $dst = pxbuf.crop($c.px, $x0, $y, $x1 + 1, $y + 1);
    $msk = null;
    if ($mask != null) { $msk = pxbuf.crop($mask, $x0, $y, $x1 + 1, $y + 1); }
    _writeSpan($c, $dst, $ink, $msk);
    return null;
}

// A vertical run, which is what the y-major half of a Bresenham line is made
// of. Clipped exactly as a column of point() calls would be.
def _vline($c, $x, $ya, $yb, $ink) {
    if ($x < 0 || $x >= $c.w) { return null; }
    if ($ya > $yb) { $t = $ya; $ya = $yb; $yb = $t; }
    if ($ya < 0) { $ya = 0; }
    if ($yb >= $c.h) { $yb = $c.h - 1; }
    if ($ya > $yb) { return null; }
    $dst = pxbuf.crop($c.px, $x, $ya, $x + 1, $yb + 1);
    _writeSpan($c, $dst, $ink, null);
    return null;
}

// point($c, $x, $y, $ink) -> Draw.c's point8/point32/point32rgba. A one-pixel
// hline has identical clipping, so it is one.
def _point($c, $x, $y, $ink) { return _hline($c, $x, $y, $x, $ink, null); }


// ════════════════════════════════════════════════════════════════════════
//  line
// ════════════════════════════════════════════════════════════════════════
//
//  Draw.c plots max(|dx|, |dy|) pixels and NOT the endpoint. The error
//  recurrence for the x-major case is
//
//      e = 2dy - dx;  per step: if (e >= 0) { minor++; e -= 2dx } e += 2dy
//
//  so after i steps the minor coordinate has advanced
//
//      k(i) = floor((2 * dy * i + dx) / (2 * dx))
//
//  times, which is the closed form this function inverts: the run at minor
//  value k covers the major indices [ceil(dx(2k-1) / 2dy), ceil(dx(2k+1) / 2dy) - 1].
//  That turns a line into min(|dx|, |dy|) + 1 span writes instead of
//  max(|dx|, |dy|) pixel writes, and it is exact — the same pixels, in the
//  same places, including every tie.

def _line($c, $x0, $y0, $x1, $y1, $ink) {
    $dx = $x1 - $x0;
    $xs = 1;
    if ($dx < 0) { $dx = 0 - $dx; $xs = 0 - 1; }
    $dy = $y1 - $y0;
    $ys = 1;
    if ($dy < 0) { $dy = 0 - $dy; $ys = 0 - 1; }

    if ($dx == 0) {
        if ($dy == 0) { return null; }              // a zero-length line draws nothing
        _vline($c, $x0, $y0, $y0 + $ys * ($dy - 1), $ink);
        return null;
    }
    if ($dy == 0) {
        $a = $x0;
        $b = $x0 + $xs * ($dx - 1);
        if ($a > $b) { $t = $a; $a = $b; $b = $t; }
        _hline($c, $a, $y0, $b, $ink, null);
        return null;
    }
    if ($dx > $dy) {
        // x-major. k is the minor (y) step count and reaches dy when the last
        // plotted pixel has already stepped, so the loop runs to dy inclusive.
        $k = 0;
        while ($k <= $dy) {
            $lo = _ceilDiv($dx * (2 * $k - 1), 2 * $dy);
            if ($lo < 0) { $lo = 0; }
            $hi = _ceilDiv($dx * (2 * $k + 1), 2 * $dy);
            if ($hi > $dx) { $hi = $dx; }
            $hi = $hi - 1;
            if ($lo <= $hi) {
                $a = $x0 + $xs * $lo;
                $b = $x0 + $xs * $hi;
                if ($a > $b) { $t = $a; $a = $b; $b = $t; }
                _hline($c, $a, $y0 + $ys * $k, $b, $ink, null);
            }
            $k = $k + 1;
        }
        return null;
    }
    // y-major, which is also where a 45-degree tie lands.
    $j = 0;
    while ($j <= $dx) {
        $lo = _ceilDiv($dy * (2 * $j - 1), 2 * $dx);
        if ($lo < 0) { $lo = 0; }
        $hi = _ceilDiv($dy * (2 * $j + 1), 2 * $dx);
        if ($hi > $dy) { $hi = $dy; }
        $hi = $hi - 1;
        if ($lo <= $hi) {
            _vline($c, $x0 + $xs * $j, $y0 + $ys * $lo, $y0 + $ys * $hi, $ink);
        }
        $j = $j + 1;
    }
    return null;
}


// ════════════════════════════════════════════════════════════════════════
//  Edges and the polygon filler
// ════════════════════════════════════════════════════════════════════════
//
//  An edge is a list, not a dict, because the filler reads it once per row
//  per edge and a dict lookup is a hash per field:
//
//      [0] d      direction: 1 when y0 is ymin, -1 when it is ymax, 0 flat
//      [1] x0     [2] y0      the ORIGINAL start, which the crossing uses
//      [3] xmin   [4] ymin    [5] xmax   [6] ymax
//      [7] dx     (x1 - x0) / (y1 - y0), in float32

def _addEdge($x0, $y0, $x1, $y1) {
    if ($x0 <= $x1) { $xmin = $x0; $xmax = $x1; } else { $xmin = $x1; $xmax = $x0; }
    if ($y0 <= $y1) { $ymin = $y0; $ymax = $y1; } else { $ymin = $y1; $ymax = $y0; }
    $d = 0;
    $dx = 0.0;
    if ($y0 != $y1) {
        $dx = _f32(($x1 - $x0) / ($y1 - $y0));
        if ($y0 == $ymin) { $d = 1; } else { $d = 0 - 1; }
    }
    return [$d, $x0, $y0, $xmin, $ymin, $xmax, $ymax, $dx];
}

// draw_horizontal_lines: the RGBA path's pre-pass for flat edges, which
// carries the row's span cursor so two spans cannot blend over each other.
// $pos is a dict because it is an out-parameter; $edges is copied on the way
// in (PW14), which is why only the blending path pays for this call.
def _drawHorizontalLines($c, $edges, $ink, $pos, $y, $mask) {
    $n = len($edges);
    $i = 0;
    while ($i < $n) {
        $ed = $edges[$i];
        if ($ed[4] == $y && $ed[4] == $ed[6]) {
            $xmin = $ed[3];
            if ($pos.v != 0 - 1 && $pos.v < $xmin) { $i = $i + 1; continue; }
            $xmax = $ed[5];
            if ($pos.v > $xmin) {
                $xmin = $pos.v;
                if ($xmax < $xmin) { $i = $i + 1; continue; }
            }
            _hline($c, $xmin, $ed[4], $xmax, $ink, $mask);
            $pos.v = $xmax + 1;
        }
        $i = $i + 1;
    }
    return null;
}

// polygon_generic: the scanline filler every filled shape goes through.
//
// THE CORNER JOIN IS INLINED, and that is not a style choice. Reading a list
// out of a dict copies it in Bantu (PW14), and so does handing one to a
// function, so a `_cornerJoin(...)` helper would copy the whole edge table
// once per corner -- quadratic, and on a 100,000-point polygon fatal. The
// fix-up therefore lives in the row loop, where the table is a local.
def _polygonGeneric($c, $edges, $ink, $mask) {
    $n = len($edges);
    if ($n <= 0) { return null; }

    // Draw.c seeds ymin at ysize - 1 and ymax at 0, which is visible
    // behaviour for a polygon entirely below the image: the scan still
    // starts at the last row.
    $ymin = $c.h - 1;
    $ymax = 0;
    $table = [];
    $hasAlpha = $c.blend;
    $i = 0;
    while ($i < $n) {
        $ed = $edges[$i];
        if ($ymin > $ed[4]) { $ymin = $ed[4]; }
        if ($ymax < $ed[6]) { $ymax = $ed[6]; }
        if ($ed[4] == $ed[6]) {
            if (!$hasAlpha) { _hline($c, $ed[3], $ed[4], $ed[5], $ink, $mask); }
            $i = $i + 1;
            continue;
        }
        push($table, $ed);
        $i = $i + 1;
    }
    if ($ymin < 0) { $ymin = 0; }
    if ($ymax > $c.h) { $ymax = $c.h; }
    $ne = len($table);

    $y = $ymin;
    while ($y <= $ymax) {
        $xx = [];
        $i = 0;
        while ($i < $ne) {
            $cur = $table[$i];
            if ($y >= $cur[4] && $y <= $cur[6]) {
                $v = _cross($y, $cur[2], $cur[7], $cur[1]);
                push($xx, $v);
                if ($y == $cur[6] && $y < $ymax) {
                    // A crossing at an edge's last row counts TWICE, which is
                    // what keeps two polygons sharing that row consistent.
                    push($xx, $v);
                } else if (($y == $cur[4] || $y == $cur[6]) && $cur[7] != 0) {
                    // "Connect discontiguous corners": where this edge's
                    // crossing rounds to the same pixel as an EARLIER edge's,
                    // and both are at one of their own ends, the crossing is
                    // pulled back towards the adjacent row's.
                    $k = 0;
                    while ($k < $i) {
                        $other = $table[$k];
                        if (($y != $other[4] && $y != $other[6]) || $other[7] == 0) {
                            $k = $k + 1;
                            continue;
                        }
                        if (round($v) ==
                            round(_cross($y, $other[2], $other[7], $other[1]))) {
                            $off = 1;
                            if ($y == $cur[6]) { $off = 0 - 1; }
                            $adj = _cross($y + $off, $cur[2], $cur[7], $cur[1]);
                            if ($y + $off >= $other[4] && $y + $off <= $other[6]) {
                                $adjO = _cross($y + $off, $other[2], $other[7], $other[1]);
                                if ($v > $adj + 1 && $v > $adjO + 1) {
                                    $xx[len($xx) - 1] = round(max($adj, $adjO)) + 1;
                                } else if ($v < $adj - 1 && $v < $adjO - 1) {
                                    $xx[len($xx) - 1] = round(min($adj, $adjO)) - 1;
                                }
                                break;
                            }
                        }
                        $k = $k + 1;
                    }
                }
            }
            $i = $i + 1;
        }
        $xs = sort($xx);
        $j = len($xs);
        if ($hasAlpha) {
            $pos = {"v": 0 - 1};
            if ($j != 0) { $pos.v = 0; }
            $i = 1;
            while ($i < $j) {
                $xEnd = ROUND_DOWN($xs[$i]);
                if ($xEnd < $pos.v) { $i = $i + 2; continue; }
                _drawHorizontalLines($c, $edges, $ink, $pos, $y, $mask);
                if ($xEnd < $pos.v) { $i = $i + 2; continue; }
                $xStart = ROUND_UP($xs[$i - 1]);
                if ($pos.v > $xStart) {
                    $xStart = $pos.v;
                    if ($xEnd < $xStart) { $i = $i + 2; continue; }
                }
                _hline($c, $xStart, $y, $xEnd, $ink, $mask);
                $pos.v = $xEnd + 1;
                $i = $i + 2;
            }
            _drawHorizontalLines($c, $edges, $ink, $pos, $y, $mask);
        } else {
            $i = 1;
            while ($i < $j) {
                _hline($c, ROUND_UP($xs[$i - 1]), $y, ROUND_DOWN($xs[$i]), $ink, $mask);
                $i = $i + 2;
            }
        }
        $y = $y + 1;
    }
    return null;
}


// ════════════════════════════════════════════════════════════════════════
//  Wide line
// ════════════════════════════════════════════════════════════════════════

def _wideLine($c, $x0, $y0, $x1, $y1, $ink, $width, $mask) {
    $dx = $x1 - $x0;
    $dy = $y1 - $y0;
    if ($dx == 0 && $dy == 0) {
        _point($c, $x0, $y0, $ink);
        return null;
    }
    $big = hypot($dx, $dy);
    $small = ($width - 1) / 2.0;
    $rmax = ROUND_UP($small) / $big;
    $rmin = ROUND_DOWN($small) / $big;
    $dxmin = ROUND_DOWN($rmin * $dy);
    $dxmax = ROUND_DOWN($rmax * $dy);
    $dymin = ROUND_DOWN($rmin * $dx);
    $dymax = ROUND_DOWN($rmax * $dx);
    $v = [[$x0 - $dxmin, $y0 + $dymax], [$x1 - $dxmin, $y1 + $dymax],
          [$x1 + $dxmax, $y1 - $dymin], [$x0 + $dxmax, $y0 - $dymin]];
    _polygonGeneric($c, [_addEdge($v[0][0], $v[0][1], $v[1][0], $v[1][1]),
                         _addEdge($v[1][0], $v[1][1], $v[2][0], $v[2][1]),
                         _addEdge($v[2][0], $v[2][1], $v[3][0], $v[3][1]),
                         _addEdge($v[3][0], $v[3][1], $v[0][0], $v[0][1])],
                    $ink, $mask);
    return null;
}


// ════════════════════════════════════════════════════════════════════════
//  rectangle
// ════════════════════════════════════════════════════════════════════════

def _rectangle($c, $x0, $y0, $x1, $y1, $ink, $fill, $width) {
    if ($y0 > $y1) { $t = $y0; $y0 = $y1; $y1 = $t; }
    if ($fill) {
        if ($y0 < 0) { $y0 = 0; } else if ($y0 >= $c.h) { return null; }
        if ($y1 < 0) { return null; } else if ($y1 > $c.h) { $y1 = $c.h; }
        $y = $y0;
        while ($y <= $y1) {
            _hline($c, $x0, $y, $x1, $ink, null);
            $y = $y + 1;
        }
        return null;
    }
    // The outline expands INWARD, and a width of 0 is promoted to 1 — which
    // the ellipse family does NOT do.
    $w = $width;
    if ($w == 0) { $w = 1; }
    $i = 0;
    while ($i < $w) {
        _hline($c, $x0, $y0 + $i, $x1, $ink, null);
        _hline($c, $x0, $y1 - $i, $x1, $ink, null);
        _line($c, $x1 - $i, $y0 + $w, $x1 - $i, $y1 - $w + 1, $ink);
        _line($c, $x0 + $i, $y0 + $w, $x0 + $i, $y1 - $w + 1, $ink);
        $i = $i + 1;
    }
    return null;
}


// ════════════════════════════════════════════════════════════════════════
//  polygon
// ════════════════════════════════════════════════════════════════════════

def _polygon($c, $xy, $ink, $fill, $width, $mask) {
    $count = floor(len($xy) / 2);
    if ($count <= 0) { return null; }
    if ($fill) {
        $edges = [];
        $i = 0;
        while ($i < $count - 1) {
            $x0 = $xy[$i * 2];
            $y0 = $xy[$i * 2 + 1];
            $x1 = $xy[$i * 2 + 2];
            $y1 = $xy[$i * 2 + 3];
            // A horizontal edge that follows another horizontal edge in the
            // same direction is merged into it, so a staircase of flat edges
            // is one span rather than several.
            if ($y0 == $y1 && $i != 0 && $y0 == $xy[$i * 2 - 1]) {
                // PW14 again: $edges[n] hands back a COPY, so the widened
                // edge has to be written back into the list.
                $last = len($edges) - 1;
                $lastE = $edges[$last];
                if ($x1 > $x0 && $x0 > $xy[$i * 2 - 2]) {
                    $lastE[5] = $x1;
                    $edges[$last] = $lastE;
                    $i = $i + 1;
                    continue;
                } else if ($x1 < $x0 && $x0 < $xy[$i * 2 - 2]) {
                    $lastE[3] = $x1;
                    $edges[$last] = $lastE;
                    $i = $i + 1;
                    continue;
                }
            }
            push($edges, _addEdge($x0, $y0, $x1, $y1));
            $i = $i + 1;
        }
        if ($xy[$i * 2] != $xy[0] || $xy[$i * 2 + 1] != $xy[1]) {
            push($edges, _addEdge($xy[$i * 2], $xy[$i * 2 + 1], $xy[0], $xy[1]));
        }
        _polygonGeneric($c, $edges, $ink, $mask);
        return null;
    }
    if ($width == 1) {
        $i = 0;
        while ($i < $count - 1) {
            _line($c, $xy[$i * 2], $xy[$i * 2 + 1], $xy[$i * 2 + 2], $xy[$i * 2 + 3], $ink);
            $i = $i + 1;
        }
        _line($c, $xy[$i * 2], $xy[$i * 2 + 1], $xy[0], $xy[1], $ink);
        return null;
    }
    $i = 0;
    while ($i < $count - 1) {
        _wideLine($c, $xy[$i * 2], $xy[$i * 2 + 1], $xy[$i * 2 + 2], $xy[$i * 2 + 3],
                  $ink, $width, $mask);
        $i = $i + 1;
    }
    _wideLine($c, $xy[$i * 2], $xy[$i * 2 + 1], $xy[0], $xy[1], $ink, $width, $mask);
    return null;
}


// ════════════════════════════════════════════════════════════════════════
//  The ellipse family
// ════════════════════════════════════════════════════════════════════════
//
//  Draw.c approximates the top-right quarter of an ellipse with semi-axes
//  a and b using integer points from {(2x + x0, 2y + y0)} — a grid of step 2
//  whose parity is chosen so that (a, b) is on it. Working at step 2 is what
//  lets the angle clipping below stay in integers.
//
//  The tracer minimises |a^2 y^2 + b^2 x^2 - a^2 b^2| over three candidates,
//  and THE THIRD CANDIDATE DOES NOT REFRESH the running minimum: the C
//  compares the third against the second's delta but never stores its own.
//  That is reproduced, because an ellipse that corrects it is not Pillow's
//  ellipse.
//
//  THE INTEGER CEILING. The delta is a^2 y^2 + b^2 x^2, which for semi-axes
//  near n is about n^4. Bantu numbers are doubles, so the arithmetic is exact
//  while n^4 < 2^53 — semi-axes up to about 7,000, or a 14,000-pixel ellipse.
//  Beyond that a candidate comparison can tie where the C would not, and the
//  outline may differ by a pixel. Noted rather than guarded: the guard would
//  cost a test per candidate on every ellipse drawn.

def _quarterInit($a, $b) {
    if ($a < 0 || $b < 0) {
        return {"finished": true, "a": $a, "b": $b, "cx": 0, "cy": 0,
                "ex": 0, "ey": 0, "a2": 0, "b2": 0, "a2b2": 0};
    }
    return {"finished": false, "a": $a, "b": $b,
            "cx": $a, "cy": $b % 2, "ex": $a % 2, "ey": $b,
            "a2": $a * $a, "b2": $b * $b, "a2b2": ($a * $a) * ($b * $b)};
}

def _quarterDelta($s, $x, $y) {
    $v = $s.a2 * $y * $y + $s.b2 * $x * $x - $s.a2b2;
    if ($v < 0) { return 0 - $v; }
    return $v;
}

// -> [x, y], or null when the quarter is exhausted.
def _quarterNext($s) {
    if ($s.finished) { return null; }
    $rx = $s.cx;
    $ry = $s.cy;
    if ($s.cx == $s.ex && $s.cy == $s.ey) {
        $s.finished = true;
    } else {
        $nx = $s.cx;
        $ny = $s.cy + 2;
        $nd = _quarterDelta($s, $nx, $ny);
        if ($nx > 1) {
            $cand = _quarterDelta($s, $s.cx - 2, $s.cy + 2);
            if ($nd > $cand) {
                $nx = $s.cx - 2;
                $ny = $s.cy + 2;
                $nd = $cand;
            }
            $cand = _quarterDelta($s, $s.cx - 2, $s.cy);
            if ($nd > $cand) {
                // As in the C: nd is NOT updated here.
                $nx = $s.cx - 2;
                $ny = $s.cy;
            }
        }
        $s.cx = $nx;
        $s.cy = $ny;
    }
    return [$rx, $ry];
}

// Two quarter tracers — an outer at (a, b) and an inner at
// (a - 2(w-1), b - 2(w-1)) — joined into the horizontal segments of a
// complete ellipse of thickness w. Rows come out LIFO within a y.
def _ellipseInit($a, $b, $w) {
    $s = {"bufcnt": 0, "leftmost": $a % 2, "finished": false,
          "st_o": _quarterInit($a, $b), "st_i": null,
          "py": 0, "pl": 0, "pr": 0,
          "cy": [0, 0, 0, 0], "cl": [0, 0, 0, 0], "cr": [0, 0, 0, 0]};
    $first = null;
    if ($w >= 1) { $first = _quarterNext($s.st_o); }
    if ($w < 1 || $first == null) {
        $s.finished = true;
        return $s;
    }
    $s.pr = $first[0];
    $s.py = $first[1];
    $s.st_i = _quarterInit($a - 2 * ($w - 1), $b - 2 * ($w - 1));
    $s.pl = $s.leftmost;
    return $s;
}

// -> [x0, y, x1], or null when the ellipse is exhausted.
def _ellipseNext($s) {
    if ($s.bufcnt == 0) {
        if ($s.finished) { return null; }
        $y = $s.py;
        $l = $s.pl;
        $r = $s.pr;
        $cx = 0;
        $cy = 0;
        $ended = false;
        while (true) {
            $p = _quarterNext($s.st_o);
            if ($p == null) { $ended = true; break; }
            $cx = $p[0];
            $cy = $p[1];
            if ($cy > $y) { break; }
        }
        if ($ended) {
            $s.finished = true;
        } else {
            $s.pr = $cx;
            $s.py = $cy;
        }
        $iEnded = false;
        while (true) {
            $p = _quarterNext($s.st_i);
            if ($p == null) { $iEnded = true; break; }
            $cx = $p[0];
            $cy = $p[1];
            if ($cy > $y) { break; }
            $l = $cx;
        }
        if ($iEnded) { $s.pl = $s.leftmost; } else { $s.pl = $cx; }

        if (($l > 0 || $l < $r) && $y > 0) {
            $q = $s.bufcnt;
            $s.cl[$q] = $l;
            if ($l == 0) { $s.cl[$q] = 2; }
            $s.cy[$q] = $y;
            $s.cr[$q] = $r;
            $s.bufcnt = $q + 1;
        }
        if ($y > 0) {
            $q = $s.bufcnt;
            $s.cl[$q] = 0 - $r;
            $s.cy[$q] = $y;
            $s.cr[$q] = 0 - $l;
            $s.bufcnt = $q + 1;
        }
        if ($l > 0 || $l < $r) {
            $q = $s.bufcnt;
            $s.cl[$q] = $l;
            if ($l == 0) { $s.cl[$q] = 2; }
            $s.cy[$q] = 0 - $y;
            $s.cr[$q] = $r;
            $s.bufcnt = $q + 1;
        }
        $q = $s.bufcnt;
        $s.cl[$q] = 0 - $r;
        $s.cy[$q] = 0 - $y;
        $s.cr[$q] = 0 - $l;
        $s.bufcnt = $q + 1;
    }
    $s.bufcnt = $s.bufcnt - 1;
    $q = $s.bufcnt;
    return [$s.cl[$q], $s.cy[$q], $s.cr[$q]];
}


// ─── The clipping tree ──────────────────────────────────────────────────
//
//  arc, chord and pieslice are the same ring intersected with a tree of
//  half-planes A*x + B*y + C >= 0, combined with AND and OR. A horizontal
//  segment thrown into the tree comes out as an ordered set of disjoint
//  segments, as a list of [x, type] events with type 1 for a left end and
//  -1 for a right end.

def _clipNode($a, $b, $cc) { return {"t": "clip", "a": $a, "b": $b, "c": $cc, "l": null, "r": null}; }
def _combine($t, $l, $r)   { return {"t": $t, "a": 0, "b": 0, "c": 0, "l": $l, "r": $r}; }

// Mirror every clipping node about y = x.
def _transposeTree($root) {
    if ($root == null) { return null; }
    if ($root.t == "clip") {
        $t = $root.a;
        $root.a = $root.b;
        $root.b = $t;
    }
    _transposeTree($root.l);
    _transposeTree($root.r);
    return null;
}

def _doClip($root, $x0, $y, $x1) {
    if ($root == null) { return [[$x0, 1], [$x1, 0 - 1]]; }
    if ($root.t == "clip") {
        $eps = 0.000000001;
        $A = $root.a;
        $B = $root.b;
        $C = $root.c;
        if (abs($A) < $eps) {
            if ($B * $y + $C < 0 - $eps) { $x0 = 1; $x1 = 0; }
        } else {
            $ix = 0 - ($B * $y + $C) / $A;
            if ($A * $x0 + $B * $y + $C < $eps) { $x0 = round(max($x0, $ix)); }
            if ($A * $x1 + $B * $y + $C < $eps) { $x1 = round(min($x1, $ix)); }
        }
        if ($x0 <= $x1) { return [[$x0, 1], [$x1, 0 - 1]]; }
        return [];
    }
    $l1 = _doClip($root.l, $x0, $y, $x1);
    $l2 = _doClip($root.r, $x0, $y, $x1);
    $n1 = len($l1);
    $n2 = len($l2);
    $i1 = 0;
    $i2 = 0;
    $k1 = 0;
    $k2 = 0;
    $tail = 0;                                 // 0 means "no tail yet"
    $out = [];
    $isOr = $root.t == "or";
    while ($i1 < $n1 || $i2 < $n2) {
        $takeFirst = false;
        if ($i2 >= $n2) {
            $takeFirst = true;
        } else if ($i1 < $n1) {
            $p = $l1[$i1];
            $q = $l2[$i2];
            if ($p[0] < $q[0] || ($p[0] == $q[0] && $p[1] > $q[1])) { $takeFirst = true; }
        }
        if ($takeFirst) {
            $t = $l1[$i1];
            $i1 = $i1 + 1;
            $k1 = $k1 + $t[1];
        } else {
            $t = $l2[$i2];
            $i2 = $i2 + 1;
            $k2 = $k2 + $t[1];
        }
        $keep = false;
        if ($isOr) {
            if (($t[1] == 1 && ($tail == 0 || $tail == 0 - 1)) ||
                ($t[1] == 0 - 1 && $k1 == 0 && $k2 == 0)) { $keep = true; }
        } else {
            if (($t[1] == 1 && ($tail == 0 || $tail == 0 - 1) && $k1 > 0 && $k2 > 0) ||
                ($t[1] == 0 - 1 && $tail == 1 && ($k1 == 0 || $k2 == 0))) { $keep = true; }
        }
        if ($keep) {
            push($out, $t);
            $tail = $t[1];
        }
    }
    return $out;
}

// Resulting angles satisfy 0 <= al < 360 and al <= ar <= al + 360. The
// angles are `float` in the C, so each assignment is rounded to float32.
def _normalizeAngles($al, $ar) {
    $a = _f32($al);
    $b = _f32($ar);
    if ($b - $a >= 360) { return [0, 360]; }
    $na = $a;
    if ($a < 0) { $na = 360 - fmod(0 - $a, 360); }
    $na = _f32(fmod($na, 360));
    $d = $b - $na;
    if ($b < $na) { $d = 360 - fmod($na - $b, 360); }
    return [$na, _f32($na + fmod($d, 360))];
}

$_PI = 3.14159265358979323846;
def _rad($deg) { return $deg * $_PI / 180.0; }

// A clip-ellipse state: the ring plus the tree, plus the event list being
// drained.
def _clipEllipse($st, $root) {
    return {"st": $st, "root": $root, "head": [], "hi": 0, "y": 0};
}

def _clipEllipseNext($s) {
    while ($s.hi >= len($s.head)) {
        $seg = _ellipseNext($s.st);
        if ($seg == null) { return null; }
        $s.head = _doClip($s.root, $seg[0], $seg[1], $seg[2]);
        $s.hi = 0;
        $s.y = $seg[1];
    }
    $a = $s.head[$s.hi];
    $b = $s.head[$s.hi + 1];
    $s.hi = $s.hi + 2;
    return [$a[0], $s.y, $b[0]];
}

// arc_init: an arc with caps orthogonal to the ellipse curve.
def _arcInit($a, $b, $w, $al, $ar) {
    if ($a < $b) {
        // Built in swapped coordinates, then mirrored back.
        $inner = _arcInit($b, $a, $w, 90 - $ar, 90 - $al);
        _transposeTree($inner.root);
        return _clipEllipse(_ellipseInit($a, $b, $w), $inner.root);
    }
    $norm = _normalizeAngles($al, $ar);
    $al = $norm[0];
    $ar = $norm[1];
    if ($ar == $al + 360) { return _clipEllipse(_ellipseInit($a, $b, $w), null); }

    $lc = _clipNode(0 - $a * sin(_rad($al)), $b * cos(_rad($al)),
                    ($a * $a - $b * $b) * sin(_rad(2 * $al)) / 2.0);
    $rc = _clipNode($a * sin(_rad($ar)), 0 - $b * cos(_rad($ar)),
                    ($b * $b - $a * $a) * sin(_rad(2 * $ar)) / 2.0);
    $root = null;
    if (fmod($al, 180) == 0 || fmod($ar, 180) == 0) {
        $t = "or";
        if ($ar - $al < 180) { $t = "and"; }
        $root = _combine($t, $lc, $rc);
    } else if ((trunc($al / 180) + trunc($ar / 180)) % 2 == 1) {
        $lb = 0 - 1;
        if (trunc($al / 180) % 2 == 0) { $lb = 1; }
        $rb = 0 - 1;
        if (trunc($ar / 180) % 2 == 0) { $rb = 1; }
        $root = _combine("or",
                         _combine("and", _clipNode(0, $lb, 0), $lc),
                         _combine("and", _clipNode(0, $rb, 0), $rc));
    } else {
        $t = "or";
        if ($ar - $al < 180) { $t = "and"; }
        $sb = 0 - 1;
        if ($ar < 180 || $ar > 540) { $sb = 1; }
        $root = _combine($t, _combine($t, $lc, $rc), _clipNode(0, $sb, 0));
    }
    return _clipEllipse(_ellipseInit($a, $b, $w), $root);
}

// chord_line_init: the straight edge of an unfilled chord.
def _chordLineInit($a, $b, $w, $al, $ar) {
    $xl = $a * cos(_rad($al));
    $xr = $a * cos(_rad($ar));
    $yl = $b * sin(_rad($al));
    $yr = $b * sin(_rad($ar));
    $la = $yr - $yl;
    $lb = $xl - $xr;
    $lcc = 0 - ($la * $xl + $lb * $yl);
    $root = _combine("and", _clipNode($la, $lb, $lcc),
                     _clipNode(0 - $la, 0 - $lb,
                               2 * $w * sqrt($la * $la + $lb * $lb) - $lcc));
    return _clipEllipse(_ellipseInit($a, $b, $a + $b + 1), $root);
}

// pie_side_init: one radius band of an unfilled pieslice.
def _pieSideInit($a, $b, $w, $al) {
    $xl = $a * cos(_rad($al));
    $yl = $b * sin(_rad($al));
    $a1 = 0 - $yl;
    $b1 = $xl;
    $c1 = $w * sqrt($a1 * $a1 + $b1 * $b1);
    $root = _combine("and",
                     _combine("and", _clipNode($a1, $b1, $c1),
                              _clipNode(0 - $a1, 0 - $b1, $c1)),
                     _clipNode($b1, 0 - $a1, 0));
    return _clipEllipse(_ellipseInit($a, $b, $a + $b + 1), $root);
}

def _chordInit($a, $b, $w, $al, $ar) {
    $xl = $a * cos(_rad($al));
    $xr = $a * cos(_rad($ar));
    $yl = $b * sin(_rad($al));
    $yr = $b * sin(_rad($ar));
    $ca = $yr - $yl;
    $cb = $xl - $xr;
    return _clipEllipse(_ellipseInit($a, $b, $w),
                        _clipNode($ca, $cb, 0 - ($ca * $xl + $cb * $yl)));
}

def _pieInit($a, $b, $w, $al, $ar) {
    $xl = $a * cos(_rad($al));
    $xr = $a * cos(_rad($ar));
    $yl = $b * sin(_rad($al));
    $yr = $b * sin(_rad($ar));
    $lc = _clipNode(0 - $yl, $xl, 0);
    $rc = _clipNode($yr, 0 - $xr, 0);
    $t = "or";
    if ($ar - $al < 180) { $t = "and"; }
    $root = _combine($t, $lc, $rc);
    if ($ar - $al < 90) {
        // one more half-plane, to avoid a spike past the two radii
        $root = _combine("and", $root,
                         _clipNode(($xl + $xr) / 2.0, ($yl + $yr) / 2.0, 0));
    }
    return _clipEllipse(_ellipseInit($a, $b, $w), $root);
}

// ellipseNew: the ring, mapped from the step-2 grid onto pixels.
def _ellipseDraw($c, $x0, $y0, $x1, $y1, $ink, $fill, $width) {
    $a = $x1 - $x0;
    $b = $y1 - $y0;
    if ($a < 0 || $b < 0) { return null; }
    $w = $width;
    if ($fill) { $w = $a + $b; }
    $s = _ellipseInit($a, $b, $w);
    while (true) {
        $seg = _ellipseNext($s);
        if ($seg == null) { break; }
        _hline($c, $x0 + floor(($seg[0] + $a) / 2), $y0 + floor(($seg[1] + $b) / 2),
               $x0 + floor(($seg[2] + $a) / 2), $ink, null);
    }
    return null;
}

def _clipEllipseDraw($c, $x0, $y0, $x1, $y1, $ink, $s) {
    $a = $x1 - $x0;
    $b = $y1 - $y0;
    if ($a < 0 || $b < 0) { return null; }
    while (true) {
        $seg = _clipEllipseNext($s);
        if ($seg == null) { break; }
        _hline($c, $x0 + floor(($seg[0] + $a) / 2), $y0 + floor(($seg[1] + $b) / 2),
               $x0 + floor(($seg[2] + $a) / 2), $ink, null);
    }
    return null;
}

def _drawArc($c, $x0, $y0, $x1, $y1, $start, $end, $ink, $width) {
    $n = _normalizeAngles($start, $end);
    $start = $n[0];
    $end = $n[1];
    if ($start + 360 == $end) {
        return _ellipseDraw($c, $x0, $y0, $x1, $y1, $ink, false, $width);
    }
    if ($start == $end) { return null; }
    $a = $x1 - $x0;
    $b = $y1 - $y0;
    if ($a < 0 || $b < 0) { return null; }
    return _clipEllipseDraw($c, $x0, $y0, $x1, $y1, $ink,
                            _arcInit($a, $b, $width, $start, $end));
}

def _drawChord($c, $x0, $y0, $x1, $y1, $start, $end, $ink, $fill, $width) {
    $n = _normalizeAngles($start, $end);
    $start = $n[0];
    $end = $n[1];
    if ($start + 360 == $end) {
        return _ellipseDraw($c, $x0, $y0, $x1, $y1, $ink, $fill, $width);
    }
    if ($start == $end) { return null; }
    $a = $x1 - $x0;
    $b = $y1 - $y0;
    if ($a < 0 || $b < 0) { return null; }
    if ($fill) {
        return _clipEllipseDraw($c, $x0, $y0, $x1, $y1, $ink,
                                _chordInit($a, $b, $x1 - $x0 + $y1 - $y0 + 1, $start, $end));
    }
    _clipEllipseDraw($c, $x0, $y0, $x1, $y1, $ink,
                     _chordLineInit($a, $b, $width, $start, $end));
    return _clipEllipseDraw($c, $x0, $y0, $x1, $y1, $ink,
                            _chordInit($a, $b, $width, $start, $end));
}

def _drawPieslice($c, $x0, $y0, $x1, $y1, $start, $end, $ink, $fill, $width) {
    $n = _normalizeAngles($start, $end);
    $start = $n[0];
    $end = $n[1];
    if ($start + 360 == $end) {
        return _ellipseDraw($c, $x0, $y0, $x1, $y1, $ink, $fill, $width);
    }
    if ($start == $end) { return null; }
    $a = $x1 - $x0;
    $b = $y1 - $y0;
    if ($a < 0 || $b < 0) { return null; }
    if ($fill) {
        return _clipEllipseDraw($c, $x0, $y0, $x1, $y1, $ink,
                                _pieInit($a, $b, $x1 + $y1 - $x0 - $y0, $start, $end));
    }
    // Two radius bands, a centre disk of diameter `width`, then the wedge.
    _clipEllipseDraw($c, $x0, $y0, $x1, $y1, $ink, _pieSideInit($a, $b, $width, $start));
    _clipEllipseDraw($c, $x0, $y0, $x1, $y1, $ink, _pieSideInit($a, $b, $width, $end));
    $xc = round(($x0 + $x1 - $width) / 2.0);
    $yc = round(($y0 + $y1 - $width) / 2.0);
    _ellipseDraw($c, $xc, $yc, $xc + $width - 1, $yc + $width - 1, $ink, true, 0);
    return _clipEllipseDraw($c, $x0, $y0, $x1, $y1, $ink,
                            _pieInit($a, $b, $width, $start, $end));
}


// ════════════════════════════════════════════════════════════════════════
//  bitmap: ImagingFill2
// ════════════════════════════════════════════════════════════════════════
//
//  Fill a box with one ink through a mask. A mode-"1" mask is a stencil; a
//  mode-"L" mask is a matte, blended per Paste.c.

def _fillMask($c, $x0, $y0, $maskIm, $ink) {
    $mm = $maskIm.mode;
    if ($mm != "1" && $mm != "L") {
        throw "ImageDraw.bitmap: the bitmap must be mode \"1\" or \"L\", got \"" +
              $mm + "\"";
    }
    $xsize = $maskIm.width;
    $ysize = $maskIm.height;
    $sx = 0;
    $sy = 0;
    $dx = $x0;
    $dy = $y0;
    if ($dx < 0) {
        $xsize = $xsize + $dx;
        if ($xsize <= 0) { return null; }
        $sx = 0 - $dx;
        $dx = 0;
    }
    if ($dx + $xsize > $c.w) { $xsize = $c.w - $dx; }
    if ($dy < 0) {
        $ysize = $ysize + $dy;
        if ($ysize <= 0) { return null; }
        $sy = 0 - $dy;
        $dy = 0;
    }
    if ($dy + $ysize > $c.h) { $ysize = $c.h - $dy; }
    if ($xsize <= 0 || $ysize <= 0) { return null; }

    $dst = pxbuf.crop($c.px, $dx, $dy, $dx + $xsize, $dy + $ysize);
    $msk = pxbuf.crop($maskIm.px, $sx, $sy, $sx + $xsize, $sy + $ysize);
    if ($mm == "1") {
        $cond = nd_not_equal($msk, 0);
        $i = 0;
        while ($i < $c.nb) {
            $band = pxbuf.band($dst, $i);
            nd_add(nd_where($cond, $ink.v[$i], $band), 0, $band);
            $i = $i + 1;
        }
        return null;
    }
    return _fillMaskL($c, $dst, $msk, $ink);
}

// fill_mask_L. The alpha correction is Paste.c's, integer division and all:
//
//     channel_mask = 255 - (255 - m) * (1 - (255 - out[3]) / 255)
//
// is INTEGER arithmetic, so (255 - out[3]) / 255 is 1 only when the
// destination alpha is 0 and 0 otherwise. The factor is therefore 0 or 1 and
// the whole expression collapses to: where the destination is fully
// transparent, the colour bands take the ink at FULL strength; everywhere
// else they blend normally. The alpha band itself always blends normally.
def _fillMaskL($c, $dst, $msk, $ink) {
    $alphaIdx = mode.alpha_band($c.m);
    $full = null;
    if ($alphaIdx >= 0) {
        $a = pxbuf.band($dst, $alphaIdx);
        $full = nd_logical_and(nd_equal($a, 0), nd_not_equal($msk, 0));
    }
    $i = 0;
    while ($i < $c.nb) {
        $band = pxbuf.band($dst, $i);
        $m = $msk;
        if ($full != null && $i != $alphaIdx) { $m = nd_where($full, 255, $msk); }
        nd_add(_blendArr($band, $m, $ink.v[$i]), 0, $band);
        $i = $i + 1;
    }
    return null;
}


// ════════════════════════════════════════════════════════════════════════
//  floodfill
// ════════════════════════════════════════════════════════════════════════
//
//  Pillow's ImageDraw.floodfill is a Python set-based four-connected flood
//  with a per-pixel predicate: with no `border`, a pixel is in the region
//  when its 1-norm distance from the SEED's colour is at most `thresh`; with
//  a `border`, when it is neither the fill value nor the border value. Both
//  predicates depend only on the ORIGINAL pixel and two constants, so they
//  are one array pass, and the flood itself is then a scanline span fill
//  over that boolean mask — rows and spans, never pixels.
//
//  Equivalent to Pillow's, not merely similar: a filled pixel can never be
//  refillable under either predicate (the thresh case returns early when the
//  fill value is within thresh of the background), so computing fillability
//  from the original image is the same region Pillow's visited-set walk
//  reaches.

def floodfill($im, $xy, $value, $border, $thresh) {
    $th = 0;
    if ($thresh != null) { $th = $thresh; }
    $x = _int($xy[0]);
    $y = _int($xy[1]);
    if ($x < 0 || $y < 0 || $x >= $im.width || $y >= $im.height) { return null; }

    $c = _ctx($im.px, $im.mode, $im.mode, false);
    $ink = _inkBands($im.mode, _resolveSpec($im, $im.mode, $value));
    $bg = pxbuf.pixel($im.px, $x, $y);

    // Seed already the fill colour (within thresh)? Pillow returns.
    $diff = 0;
    $i = 0;
    while ($i < $c.nb) {
        $d = $ink.v[$i] - $bg[$i];
        if ($d < 0) { $d = 0 - $d; }
        $diff = $diff + $d;
        $i = $i + 1;
    }
    if ($diff <= $th) { return null; }

    // The fillability mask, in one pass.
    if ($border == null) {
        $acc = null;
        $i = 0;
        while ($i < $c.nb) {
            $t = nd_abs(nd_subtract(pxbuf.band($im.px, $i), $bg[$i]));
            if ($acc == null) { $acc = $t; } else { $acc = nd_add($acc, $t); }
            $i = $i + 1;
        }
        $fillable = nd_less_equal($acc, $th);
    } else {
        $bink = _inkBands($im.mode, _resolveSpec($im, $im.mode, $border));
        $isV = null;
        $isB = null;
        $i = 0;
        while ($i < $c.nb) {
            $band = pxbuf.band($im.px, $i);
            $a = nd_equal($band, $ink.v[$i]);
            $b = nd_equal($band, $bink.v[$i]);
            if ($isV == null) { $isV = $a; $isB = $b; }
            else { $isV = nd_logical_and($isV, $a); $isB = nd_logical_and($isB, $b); }
            $i = $i + 1;
        }
        $fillable = nd_logical_not(nd_logical_or($isV, $isB));
    }
    // The seed is filled whatever its own colour says, and the walk starts
    // from it, so it joins the runs on either side of it.
    nd_set($fillable, [$y, $x, 0], 1);

    // Row-by-row blocker positions, found with one array call per row and
    // cached, so a span lookup is a binary search over a short list.
    $rows = {};
    $w = $c.w;
    $h = $c.h;
    $stack = [[$x, $y]];
    $filled = {};
    while (len($stack) > 0) {
        $top = pop($stack);
        $sx = $top[0];
        $sy = $top[1];
        $span = _runAt($rows, $fillable, $sy, $sx, $w);
        if ($span == null) { continue; }
        $a = $span[0];
        $b = $span[1];
        // A run is maximal, so its LEFT END names it: one dict entry per run
        // rather than one per pixel, which is what keeps the loop
        // proportional to spans and not to area.
        $key = str($sy) + ":" + str($a);
        if ($filled[$key] != null) { continue; }
        $filled[$key] = true;
        _hline($c, $a, $sy, $b, $ink, null);
        each ($ny in [$sy - 1, $sy + 1]) {
            if ($ny < 0 || $ny >= $h) { continue; }
            $q = $a;
            while ($q <= $b) {
                $r = _runAt($rows, $fillable, $ny, $q, $w);
                if ($r == null) { $q = $q + 1; continue; }
                if ($filled[str($ny) + ":" + str($r[0])] == null) {
                    push($stack, [$r[0], $ny]);
                }
                $q = $r[1] + 1;
            }
        }
    }
    return null;
}

// The x positions in row y that are NOT fillable, ascending, as a numba
// array. ONE array call per row, cached -- and an ARRAY rather than a Bantu
// list because a list read out of a dict or handed to a function is copied
// (PW14), and copying a hundred blockers three times per span is what turned
// a comb-shaped flood fill from a tenth of a second into seven.
def _blockers($rows, $fillable, $y) {
    $k = str($y);
    if ($rows[$k] != null) { return $rows[$k]; }
    $row = nd_slice($fillable, [[$y, $y + 1], null, null]);
    // nd_nonzero returns FLAT indices in the array's own walk order, and a
    // single row has shape (1, w, 1), so a flat index IS the x position.
    $rows[$k] = nd_nonzero(nd_logical_not($row));
    return $rows[$k];
}

// _runAt -> the maximal fillable run of row y containing x, as [a, b]
// inclusive, or null when x itself is a blocker. Four array calls, whatever
// the row is like.
def _runAt($rows, $fillable, $y, $x, $w) {
    $blk = _blockers($rows, $fillable, $y);
    $n = nd_size($blk);
    if ($n == 0) { return [0, $w - 1]; }
    $k = pxbuf.scalar(nd_searchsorted($blk, $x, null));
    if (type($k) == "list") { $k = $k[0]; }
    $next = $w;
    if ($k < $n) {
        $next = nd_get($blk, [$k]);
        if ($next == $x) { return null; }
    }
    $prev = 0 - 1;
    if ($k > 0) { $prev = nd_get($blk, [$k - 1]); }
    return [$prev + 1, $next - 1];
}


// ════════════════════════════════════════════════════════════════════════
//  ImageDraw.Outline — the experimental "shape" surface
// ════════════════════════════════════════════════════════════════════════
//
//  An Outline is an edge list built by move/line/curve, filled through the
//  same scanline engine. Pillow flattens a cubic with 32 fixed steps and
//  truncates each step to an int, and so does this.

class PillowOutline {
    def init() {
        this.e = [];
        this.x = 0;
        this.y = 0;
        this.x0 = 0;
        this.y0 = 0;
    }

    def move($x0, $y0) {
        this.x = $x0;
        this.x0 = $x0;
        this.y = $y0;
        this.y0 = $y0;
        return null;
    }

    def line($x1, $y1) {
        push(this.e, _addEdge(_int(this.x), _int(this.y), _int($x1), _int($y1)));
        this.x = $x1;
        this.y = $y1;
        return null;
    }

    def curve($x1, $y1, $x2, $y2, $x3, $y3) {
        $xo = this.x;
        $yo = this.y;
        $sx = this.x;
        $sy = this.y;
        $i = 1;
        while ($i <= 32) {
            $t = $i / 32.0;
            $t2 = $t * $t;
            $t3 = $t2 * $t;
            $u = 1.0 - $t;
            $u2 = $u * $u;
            $u3 = $u2 * $u;
            $x = $sx * $u3 + 3 * ($x1 * $t * $u2 + $x2 * $t2 * $u) + $x3 * $t3 + 0.5;
            $y = $sy * $u3 + 3 * ($y1 * $t * $u2 + $y2 * $t2 * $u) + $y3 * $t3 + 0.5;
            push(this.e, _addEdge(_int($xo), _int($yo), _int($x), _int($y)));
            $xo = $x;
            $yo = $y;
            $i = $i + 1;
        }
        this.x = $xo;
        this.y = $yo;
        return null;
    }

    def close() {
        if (this.x == this.x0 && this.y == this.y0) { return null; }
        return this.line(this.x0, this.y0);
    }

    // The 6-element affine, applied to the edge list in place.
    //
    // DELIBERATELY NOT PILLOW'S, under decision PW11. ImagingOutlineTransform
    // writes the transformed edges into a freshly allocated tail of the same
    // buffer and then does
    //
    //     free(outline->edges);
    //     outline->edges = NULL;
    //     outline->count = outline->size = 0;
    //
    // so every edge it just computed is discarded and the outline is left
    // EMPTY. A `shape()` after a `transform()` therefore draws only the one
    // closing edge that `close()` then appends, from the untransformed cursor
    // to the untransformed start. The C marks the spot itself with
    // "/* FIXME: ugly! */". This transforms the edges in place and moves the
    // cursor and the start point with them, so close() closes the transformed
    // shape. tests/pillow_draw_test.b asserts the containment: a transformed
    // outline fills exactly the polygon of its transformed vertices.
    def transform($m) {
        if (type($m) != "list" || len($m) < 6) {
            throw "ImageDraw.Outline.transform: needs 6 coefficients";
        }
        $out = [];
        $i = 0;
        while ($i < len(this.e)) {
            $ed = this.e[$i];
            $x0 = $ed[1];
            $y0 = $ed[2];
            $x1 = $ed[5];
            if ($ed[1] != $ed[3]) { $x1 = $ed[3]; }
            $y1 = $ed[6];
            if ($ed[2] != $ed[4]) { $y1 = $ed[4]; }
            push($out, _addEdge(
                _int($m[0] * $x0 + $m[1] * $y0 + $m[2]),
                _int($m[3] * $x0 + $m[4] * $y0 + $m[5]),
                _int($m[0] * $x1 + $m[1] * $y1 + $m[2]),
                _int($m[3] * $x1 + $m[4] * $y1 + $m[5])));
            $i = $i + 1;
        }
        this.e = $out;
        $cx = this.x;
        $cy = this.y;
        this.x = $m[0] * $cx + $m[1] * $cy + $m[2];
        this.y = $m[3] * $cx + $m[4] * $cy + $m[5];
        $sx = this.x0;
        $sy = this.y0;
        this.x0 = $m[0] * $sx + $m[1] * $sy + $m[2];
        this.y0 = $m[3] * $sx + $m[4] * $sy + $m[5];
        return null;
    }
}

def Outline() { return new PillowOutline(); }


// ════════════════════════════════════════════════════════════════════════
//  Ink resolution at the API edge
// ════════════════════════════════════════════════════════════════════════

// _resolveSpec($im, $m, $spec) -> the spec as a number or band list, with a
// colour NAME resolved through ImageColor and a tuple on a palette image
// resolved to an index.
def _resolveSpec($im, $m, $spec) {
    $s = $spec;
    if (type($s) == "string") {
        $s = color.getcolor($s, $m);
    }
    if (mode.is_palette($m) && type($s) == "list" && $im != null && $im.palette != null) {
        return _paletteIndex($im, $s);
    }
    return $s;
}

// ImagePalette.getcolor, as far as this package can honestly go: an exact
// match wins, and otherwise the nearest entry does.
//
// PILLOW ALSO APPENDS. Its ImagePalette.getcolor adds an unknown colour to
// the palette when there is room, mutating the image's palette as a side
// effect of drawing. A palette here is a 256x3 buffer with no notion of how
// many entries are in use, so there is no slot to append into, and inventing
// one would change what getpalette() reports. The nearest entry is returned
// instead, and the difference is published beside the call in docs/pillow.md
// rather than hidden: draw on a P image with an INDEX for an exact result.
def _paletteIndex($im, $rgb) {
    $p = $im.palette;
    $n = pal.count($p);
    $pl = pal.to_list($p);
    $i = 0;
    while ($i < $n) {
        if ($pl[$i * 3] == $rgb[0] && $pl[$i * 3 + 1] == $rgb[1] &&
            $pl[$i * 3 + 2] == $rgb[2]) { return $i; }
        $i = $i + 1;
    }
    $one = nd_zeros([1, 1, 3], "i64");
    nd_set($one, [0, 0, 0], $rgb[0]);
    nd_set($one, [0, 0, 1], $rgb[1]);
    nd_set($one, [0, 0, 2], $rgb[2]);
    return pxbuf.scalar(nd_get(pal.nearest_index($one, $p, $n), [0, 0, 0]));
}


// ════════════════════════════════════════════════════════════════════════
//  The Draw object
// ════════════════════════════════════════════════════════════════════════

class PillowDraw {
    // mode is the ink mode. For an RGB image it may be "RGBA", which turns
    // on blending; anything else must equal the image's own mode, as in
    // Pillow.
    def init($im, $drawMode) {
        if (type($im) != "instance") { throw "ImageDraw.Draw: expected an image"; }
        $m = $drawMode;
        if ($m == null) { $m = $im.mode; }
        $blend = false;
        if ($m != $im.mode) {
            if ($m == "RGBA" && $im.mode == "RGB") { $blend = true; }
            else { throw "ImageDraw.Draw: mode mismatch (" + $m + " on " + $im.mode + ")"; }
        }
        this.im = $im;
        this.mode = $m;
        this.c = _ctx($im.px, $im.mode, $m, $blend);
        this.font = null;
        this.fill = false;
        // Pillow's default ink, from draw_ink(-1) — or draw_ink(1) for I and
        // F. It is NOT white everywhere: a one-band 8-bit mode clips -1 to 0,
        // so the default ink on L, 1 and P is BLACK, and only the packed
        // multi-band modes come out white.
        if ($m == "I" || $m == "F") { this.ink = _inkBands($m, 1); }
        else { this.ink = _inkBands($m, _defaultInk($m)); }
        if ($m == "1" || $m == "P" || $m == "I" || $m == "F") { this.fontmode = "1"; }
        else { this.fontmode = "L"; }
    }

    def __repr() { return "<PillowDraw mode=" + this.mode + ">"; }

    // ─── ink plumbing ───────────────────────────────────────────────────
    //
    // _getink mirrors Pillow's: with neither argument the object's default
    // ink is used (as a fill when this.fill is set), and otherwise each of
    // the two is resolved independently. Returns [ink, fillInk], either null.
    def _getink($inkSpec, $fillSpec) {
        if ($inkSpec == null && $fillSpec == null) {
            if (this.fill) { return [null, this.ink]; }
            return [this.ink, null];
        }
        $a = null;
        $b = null;
        if ($inkSpec != null) {
            $a = _inkBands(this.mode, _resolveSpec(this.im, this.mode, $inkSpec));
        }
        if ($fillSpec != null) {
            $b = _inkBands(this.mode, _resolveSpec(this.im, this.mode, $fillSpec));
        }
        return [$a, $b];
    }

    // ─── the primitives ─────────────────────────────────────────────────

    def point($xy, $fill) {
        $ink = this._getink($fill, null)[0];
        if ($ink == null) { return null; }
        $f = _flatten($xy, "ImageDraw.point");
        $i = 0;
        while ($i + 1 < len($f)) {
            _point(this.c, _int($f[$i]), _int($f[$i + 1]), $ink);
            $i = $i + 2;
        }
        return null;
    }

    def line($xy, $fill, $width, $joint) {
        $ink = this._getink($fill, null)[0];
        $w = $width;
        if ($w == null) { $w = 1; }
        if ($ink == null || $w == 0) { return null; }
        $f = _flatten($xy, "ImageDraw.line");
        $n = floor(len($f) / 2);
        if ($w == 1) {
            $i = 0;
            while ($i < $n - 1) {
                _line(this.c, _int($f[$i * 2]), _int($f[$i * 2 + 1]),
                      _int($f[$i * 2 + 2]), _int($f[$i * 2 + 3]), $ink);
                $i = $i + 1;
            }
            // The endpoint a line never draws, drawn once at the end — which
            // is why a polyline looks closed.
            if ($n >= 2) {
                _point(this.c, _int($f[($n - 1) * 2]), _int($f[($n - 1) * 2 + 1]), $ink);
            }
        } else {
            $i = 0;
            while ($i < $n - 1) {
                _wideLine(this.c, _int($f[$i * 2]), _int($f[$i * 2 + 1]),
                          _int($f[$i * 2 + 2]), _int($f[$i * 2 + 3]), $ink, $w, null);
                $i = $i + 1;
            }
        }
        if ($joint == "curve" && $w > 4) { this._joints($f, $fill, $w); }
        return null;
    }

    // joint="curve": a pieslice at each interior vertex, plus a 3-wide line
    // across the gap when the stroke is wider than 8.
    def _joints($f, $fill, $w) {
        $n = floor(len($f) / 2);
        $i = 1;
        while ($i < $n - 1) {
            $px = $f[$i * 2];
            $py = $f[$i * 2 + 1];
            $a0 = fmod(_deg(atan2($px - $f[($i - 1) * 2], $f[($i - 1) * 2 + 1] - $py)), 360);
            if ($a0 < 0) { $a0 = $a0 + 360; }
            $a1 = fmod(_deg(atan2($f[($i + 1) * 2] - $px, $py - $f[($i + 1) * 2 + 1])), 360);
            if ($a1 < 0) { $a1 = $a1 + 360; }
            if ($a0 == $a1) { $i = $i + 1; continue; }
            $flipped = ($a1 > $a0 && $a1 - 180 > $a0) || ($a1 < $a0 && $a1 + 180 > $a0);
            $coords = [$px - $w / 2 + 1, $py - $w / 2 + 1, $px + $w / 2 - 1, $py + $w / 2 - 1];
            if ($flipped) { $s = $a1 + 90; $e = $a0 + 90; } else { $s = $a0 - 90; $e = $a1 - 90; }
            this.pieslice($coords, $s - 90, $e - 90, $fill, null, 1);
            if ($w > 8) {
                $o = 0 - 90;
                if ($flipped) { $o = 90; }
                $g = [_coordAtAngle($px, $py, $a0 + $o, $w), [$px, $py],
                      _coordAtAngle($px, $py, $a1 + $o, $w)];
                this.line($g, $fill, 3, null);
            }
            $i = $i + 1;
        }
        return null;
    }

    def rectangle($xy, $fill, $outline, $width) {
        $b = _box($xy, "ImageDraw.rectangle");
        $r = this._getink($outline, $fill);
        $w = $width;
        if ($w == null) { $w = 1; }
        if ($r[1] != null) { _rectangle(this.c, $b[0], $b[1], $b[2], $b[3], $r[1], true, 1); }
        if ($r[0] != null && $w != 0 && _inkDiff($r[0], $r[1])) {
            _rectangle(this.c, $b[0], $b[1], $b[2], $b[3], $r[0], false, $w);
        }
        return null;
    }

    def polygon($xy, $fill, $outline, $width) {
        $f = _flatten($xy, "ImageDraw.polygon");
        if (len($f) < 4) {
            throw "ImageDraw.polygon: a polygon needs at least 2 coordinates";
        }
        $ixy = [];
        $i = 0;
        while ($i < len($f)) { push($ixy, _int($f[$i])); $i = $i + 1; }
        $r = this._getink($outline, $fill);
        $w = $width;
        if ($w == null) { $w = 1; }
        if ($r[1] != null) { _polygon(this.c, $ixy, $r[1], true, $w, null); }
        if ($r[0] != null && $w != 0 && _inkDiff($r[0], $r[1])) {
            if ($w == 1) {
                _polygon(this.c, $ixy, $r[0], false, 1, null);
            } else {
                // A wide outline would expand outwards, so Pillow masks it
                // with the polygon's own fill and draws it at width * 2 - 1.
                $mask = nd_zeros([this.c.h, this.c.w, 1], "i64");
                $mc = _ctx($mask, "1", "1", false);
                _polygon($mc, $ixy, {"v": [255], "a": null, "key": "m"}, true, 1, null);
                _polygon(this.c, $ixy, $r[0], false, $w * 2 - 1, $mask);
            }
        }
        return null;
    }

    def regular_polygon($bounding_circle, $n_sides, $rotation, $fill, $outline, $width) {
        $xy = _regularPolygonVertices($bounding_circle, $n_sides, $rotation);
        return this.polygon($xy, $fill, $outline, $width);
    }

    def ellipse($xy, $fill, $outline, $width) {
        $b = _box($xy, "ImageDraw.ellipse");
        $r = this._getink($outline, $fill);
        $w = $width;
        if ($w == null) { $w = 1; }
        if ($r[1] != null) { _ellipseDraw(this.c, $b[0], $b[1], $b[2], $b[3], $r[1], true, 1); }
        if ($r[0] != null && $w != 0 && _inkDiff($r[0], $r[1])) {
            _ellipseDraw(this.c, $b[0], $b[1], $b[2], $b[3], $r[0], false, $w);
        }
        return null;
    }

    def circle($xy, $radius, $fill, $outline, $width) {
        return this.ellipse([$xy[0] - $radius, $xy[1] - $radius,
                             $xy[0] + $radius, $xy[1] + $radius], $fill, $outline, $width);
    }

    def arc($xy, $start, $end, $fill, $width) {
        $b = _box($xy, "ImageDraw.arc");
        $ink = this._getink($fill, null)[0];
        $w = $width;
        if ($w == null) { $w = 1; }
        if ($ink == null || $w == 0) { return null; }
        return _drawArc(this.c, $b[0], $b[1], $b[2], $b[3], $start, $end, $ink, $w);
    }

    def chord($xy, $start, $end, $fill, $outline, $width) {
        $b = _box($xy, "ImageDraw.chord");
        $r = this._getink($outline, $fill);
        $w = $width;
        if ($w == null) { $w = 1; }
        if ($r[1] != null) {
            _drawChord(this.c, $b[0], $b[1], $b[2], $b[3], $start, $end, $r[1], true, 1);
        }
        if ($r[0] != null && $w != 0 && _inkDiff($r[0], $r[1])) {
            _drawChord(this.c, $b[0], $b[1], $b[2], $b[3], $start, $end, $r[0], false, $w);
        }
        return null;
    }

    def pieslice($xy, $start, $end, $fill, $outline, $width) {
        $b = _box($xy, "ImageDraw.pieslice");
        $r = this._getink($outline, $fill);
        $w = $width;
        if ($w == null) { $w = 1; }
        if ($r[1] != null) {
            _drawPieslice(this.c, $b[0], $b[1], $b[2], $b[3], $start, $end, $r[1], true, 1);
        }
        if ($r[0] != null && $w != 0 && _inkDiff($r[0], $r[1])) {
            _drawPieslice(this.c, $b[0], $b[1], $b[2], $b[3], $start, $end, $r[0], false, $w);
        }
        return null;
    }

    def rounded_rectangle($xy, $radius, $fill, $outline, $width, $corners) {
        return _roundedRectangle(this, $xy, $radius, $fill, $outline, $width, $corners);
    }

    def bitmap($xy, $bitmap, $fill) {
        $r = this._getink($fill, null);
        $ink = $r[0];
        if ($ink == null) { $ink = $r[1]; }
        if ($ink == null) { return null; }
        return _fillMask(this.c, _int($xy[0]), _int($xy[1]), $bitmap, $ink);
    }

    // ─── text ───────────────────────────────────────────────────────────
    //
    // Bantu has no keyword arguments, so Pillow's eleven text options arrive
    // as ONE dict -- the same arrangement save() uses for its keywords:
    //
    //   {"anchor": "mm", "spacing": 4, "align": "center", "direction": null,
    //    "features": null, "language": null, "stroke_width": 2,
    //    "stroke_fill": "black", "embedded_color": false, "font_size": 16}
    //
    // THE ONE DEVIATION, and it is published rather than hidden: Pillow
    // passes the FRACTIONAL part of the position into getmask2, so FreeType
    // rasterises the glyph shifted by a sub-pixel amount and text at x = 10.5
    // is a different bitmap from text at x = 10. Glyph masks here are cached
    // per (font, size, glyph) and positioned on the pixel grid, so a
    // fractional xy rounds down to it. For integer coordinates -- which is
    // what every caller writes -- there is no difference at all.
    def getfont() {
        if (this.font == null) { this.font = font.load_default(null); }
        return this.font;
    }

    def _getfont($size) {
        if ($size != null) { return font.load_default($size); }
        return this.getfont();
    }

    def text($xy, $txt, $fill, $fnt, $opts) {
        $o = $opts;
        if ($o == null) { $o = {}; }
        $f = $fnt;
        if ($f == null) { $f = this._getfont($o.font_size); }
        $sw = $o.stroke_width;
        if ($sw == null) { $sw = 0; }
        if ($o.embedded_color != null && $o.embedded_color) {
            throw "ImageDraw.text: embedded_color needs a colour font (COLR, CBDT " +
                  "or sbix), which this build does not read";
        }
        $r = this._getink($fill, null);
        $ink = $r[0];
        if ($ink == null) { $ink = $r[1]; }
        if ($ink == null) { return null; }
        $strokeInk = null;
        if ($sw != 0) {
            $strokeInk = $ink;
            if ($o.stroke_fill != null) {
                $sr = this._getink($o.stroke_fill, null);
                $strokeInk = $sr[0];
                if ($strokeInk == null) { $strokeInk = $sr[1]; }
            }
        }
        $lines = _splitText(this, $f, $txt, $xy, $o, $sw);
        $i = 0;
        while ($i < len($lines)) {
            $L = $lines[$i];
            if ($strokeInk != null) {
                _drawTextLine(this, $f, $L, $strokeInk, $sw, $o);
                if (_inkDiff($ink, $strokeInk)) {
                    _drawTextLine(this, $f, $L, $ink, 0, $o);
                }
            } else {
                _drawTextLine(this, $f, $L, $ink, 0, $o);
            }
            $i = $i + 1;
        }
        return null;
    }

    // multiline_text is text: Pillow's two entry points have been the same
    // function since the line splitting moved into ImageText.
    def multiline_text($xy, $txt, $fill, $fnt, $opts) {
        return this.text($xy, $txt, $fill, $fnt, $opts);
    }

    def textlength($txt, $fnt, $opts) {
        $o = $opts;
        if ($o == null) { $o = {}; }
        $f = $fnt;
        if ($f == null) { $f = this._getfont($o.font_size); }
        return font.getlength($f, $txt, this.fontmode, $o.direction, $o.features,
                              $o.language);
    }

    def textbbox($xy, $txt, $fnt, $opts) {
        $o = $opts;
        if ($o == null) { $o = {}; }
        $f = $fnt;
        if ($f == null) { $f = this._getfont($o.font_size); }
        $sw = $o.stroke_width;
        if ($sw == null) { $sw = 0; }
        $lines = _splitText(this, $f, $txt, $xy, $o, $sw);
        $box = null;
        $i = 0;
        while ($i < len($lines)) {
            $L = $lines[$i];
            $b = font.getbbox($f, $L[3], this.fontmode, $o.direction, $o.features,
                              $o.language, $L[2], $sw);
            $b = [$b[0] + $L[0], $b[1] + $L[1], $b[2] + $L[0], $b[3] + $L[1]];
            if ($box == null) { $box = $b; }
            else {
                $box = [min($box[0], $b[0]), min($box[1], $b[1]),
                        max($box[2], $b[2]), max($box[3], $b[3])];
            }
            $i = $i + 1;
        }
        if ($box == null) { return [$xy[0], $xy[1], $xy[0], $xy[1]]; }
        return $box;
    }

    def multiline_textbbox($xy, $txt, $fnt, $opts) {
        return this.textbbox($xy, $txt, $fnt, $opts);
    }

    def shape($outline, $fill, $outlineInk) {
        $outline.close();
        $r = this._getink($outlineInk, $fill);
        if ($r[1] != null) { _polygonGeneric(this.c, $outline.e, $r[1], null); }
        if ($r[0] != null && _inkDiff($r[0], $r[1])) {
            _polygonGeneric(this.c, $outline.e, $r[0], null);
        }
        return null;
    }
}

// _splitText -> one [x, y, anchor, text] per laid-out line.
//
// ImageText._split, whole: the line spacing is the bbox height of a capital
// "A" plus the stroke width plus `spacing`; a `t` or `b` vertical anchor is
// refused on multiline text; `m` and `d` shift the block up by half or all
// of its height; and `align` shifts each line within the widest, with
// "justify" spreading the words of every line but the last.
def _splitText($d, $f, $txt, $xy, $o, $sw) {
    $anchor = $o.anchor;
    if ($anchor == null || $anchor == "") { $anchor = "la"; }
    if (len($anchor) != 2) { throw "ImageDraw.text: anchor must be a 2 character string"; }
    $align = $o.align;
    if ($align == null || $align == "") { $align = "left"; }
    $spacing = $o.spacing;
    if ($spacing == null) { $spacing = 4; }
    $lines = split($txt, "\n");
    if (len($lines) == 1) { return [[$xy[0], $xy[1], $anchor, $lines[0]]]; }
    if ($anchor[1] == "t" || $anchor[1] == "b") {
        throw "ImageDraw.text: anchor \"" + $anchor + "\" is not supported for " +
              "multiline text";
    }
    $ls = font.getbbox($f, "A", $d.fontmode, $o.direction, $o.features, $o.language,
                       null, $sw)[3] + $sw + $spacing;
    $widths = [];
    $maxw = 0;
    each ($ln in $lines) {
        $wd = font.getlength($f, $ln, $d.fontmode, $o.direction, $o.features, $o.language);
        push($widths, $wd);
        if ($wd > $maxw) { $maxw = $wd; }
    }
    $top = $xy[1];
    if ($anchor[1] == "m") { $top = $top - (len($lines) - 1) * $ls / 2.0; }
    else if ($anchor[1] == "d") { $top = $top - (len($lines) - 1) * $ls; }
    $parts = [];
    $idx = 0;
    while ($idx < len($lines)) {
        $ln = $lines[$idx];
        $left = $xy[0];
        $diff = $maxw - $widths[$idx];
        if ($align == "center") { $left = $left + $diff / 2.0; }
        else if ($align == "right") { $left = $left + $diff; }
        else if ($align != "left" && $align != "justify") {
            throw "ImageDraw.text: align must be \"left\", \"center\", \"right\" " +
                  "or \"justify\"";
        }
        $done = false;
        if ($align == "justify" && $diff != 0 && $idx != len($lines) - 1) {
            $words = split($ln, " ");
            if (len($words) > 1) {
                if ($anchor[0] == "m") { $left = $left - $maxw / 2.0; }
                else if ($anchor[0] == "r") { $left = $left - $maxw; }
                $ww = [];
                $sum = 0;
                each ($wd in $words) {
                    $v = font.getlength($f, $wd, $d.fontmode, $o.direction, $o.features,
                                        $o.language);
                    push($ww, $v);
                    $sum = $sum + $v;
                }
                $gap = ($maxw - $sum) / (len($words) - 1);
                $wa = "l" + $anchor[1];
                $k = 0;
                while ($k < len($words)) {
                    push($parts, [$left, $top, $wa, $words[$k]]);
                    $left = $left + $ww[$k] + $gap;
                    $k = $k + 1;
                }
                $top = $top + $ls;
                $done = true;
            }
        }
        if (!$done) {
            if ($anchor[0] == "m") { $left = $left - $diff / 2.0; }
            else if ($anchor[0] == "r") { $left = $left - $diff; }
            push($parts, [$left, $top, $anchor, $ln]);
            $top = $top + $ls;
        }
        $idx = $idx + 1;
    }
    return $parts;
}

// One line, through the font's mask and then ImagingFill2.
def _drawTextLine($d, $f, $L, $ink, $sw, $o) {
    $r = font.getmask2($f, $L[3], $d.fontmode, $o.direction, $o.features,
                       $o.language, $L[2], $sw);
    $mask = $r[0];
    if ($mask.width <= 0 || $mask.height <= 0) { return null; }
    $x = _int($L[0]) + $r[1][0];
    $y = _int($L[1]) + $r[1][1];
    return _fillMask($d.c, $x, $y, $mask, $ink);
}

// Draw($im, $mode) -> a drawing context over $im, which it mutates.
def Draw($im, $drawMode) { return new PillowDraw($im, $drawMode); }

// Pillow's getdraw returns (context, ImageDraw2). ImageDraw2's pen/brush
// wrapper is declined for PW10's reason — it is a convenience layer over
// this one with no behaviour of its own — so this returns the context, which
// is the half that draws.
def getdraw($im) {
    if ($im == null) { return null; }
    return Draw($im, null);
}

// The ink a Draw starts with: draw_ink(-1) in Pillow, which is -1 for the
// packed modes and clips to 0 for a single 8-bit band.
def _defaultInk($m) {
    if (mode.bands($m) == 1 && mode.family($m) == "int" && $m != "I") { return 65535; }
    return 0 - 1;
}

def _inkDiff($a, $b) {
    if ($b == null) { return true; }
    return $a.key != $b.key;
}

def _deg($r) { return $r * 180.0 / $_PI; }

// ImageDraw's coord_at_angle: the point `width / 2 - 1` away at `angle - 90`,
// each component floored toward zero.
def _coordAtAngle($x, $y, $angle, $w) {
    $a = $angle - 90;
    $dist = $w / 2 - 1;
    $dx = $dist * cos(_rad($a));
    $dy = $dist * sin(_rad($a));
    $ox = ceil($dx);
    if ($dx > 0) { $ox = floor($dx); }
    $oy = ceil($dy);
    if ($dy > 0) { $oy = floor($dy); }
    return [$x + $ox, $y + $oy];
}

// _compute_regular_polygon_vertices, including its round-to-2-decimals.
def _regularPolygonVertices($bc, $nSides, $rotation) {
    if (type($nSides) != "number" || $nSides != floor($nSides)) {
        throw "ImageDraw.regular_polygon: n_sides should be an int";
    }
    if ($nSides < 3) { throw "ImageDraw.regular_polygon: n_sides should be an int > 2"; }
    if (type($bc) != "list") {
        throw "ImageDraw.regular_polygon: bounding_circle should be a sequence";
    }
    if (len($bc) == 3) {
        $cx = $bc[0];
        $cy = $bc[1];
        $rad = $bc[2];
    } else if (len($bc) == 2 && type($bc[0]) == "list") {
        if (len($bc[0]) != 2) {
            throw "ImageDraw.regular_polygon: bounding_circle centre should contain " +
                  "2D coordinates (e.g. [x, y])";
        }
        $cx = $bc[0][0];
        $cy = $bc[0][1];
        $rad = $bc[1];
    } else {
        throw "ImageDraw.regular_polygon: bounding_circle should contain 2D " +
              "coordinates and a radius (e.g. [x, y, r] or [[x, y], r])";
    }
    if ($rad <= 0) { throw "ImageDraw.regular_polygon: bounding_circle radius should be > 0"; }
    if (type($rotation) != "number") {
        throw "ImageDraw.regular_polygon: rotation should be a number";
    }
    $degrees = 360 / $nSides;
    $cur = (270 - 0.5 * $degrees) + $rotation;
    $out = [];
    $i = 0;
    while ($i < $nSides) {
        $phi = _rad(360 - $cur);
        push($out, [_round2($rad * cos($phi) - 0 * sin($phi) + $cx),
                    _round2(0 * cos($phi) + $rad * sin($phi) + $cy)]);
        $cur = $cur + $degrees;
        if ($cur > 360) { $cur = $cur - 360; }
        $i = $i + 1;
    }
    return $out;
}

// Python's round(v, 2): half to EVEN, on the decimal value.
def _round2($v) {
    $s = 1;
    $m = $v;
    if ($m < 0) { $s = 0 - 1; $m = 0 - $m; }
    $q = $m * 100;
    $fl = floor($q);
    $fr = $q - $fl;
    if ($fr > 0.5) { $fl = $fl + 1; }
    else if ($fr == 0.5 && $fl % 2 != 0) { $fl = $fl + 1; }
    return $s * $fl / 100;
}

// rounded_rectangle, which is ImageDraw's own composition of pieslices, arcs
// and rectangles rather than a primitive. Ported whole, including the
// degenerate cases: joined corners become half-rounds, a fully joined one is
// an ellipse, and a zero radius is a rectangle.
def _roundedRectangle($d, $xy, $radius, $fill, $outline, $width, $corners) {
    $f = _flatten($xy, "ImageDraw.rounded_rectangle");
    if (len($f) != 4) { throw "ImageDraw.rounded_rectangle: expected two coordinates"; }
    $x0 = $f[0];
    $y0 = $f[1];
    $x1 = $f[2];
    $y1 = $f[3];
    if ($x1 < $x0) { throw "ImageDraw.rounded_rectangle: x1 must be greater than or equal to x0"; }
    if ($y1 < $y0) { throw "ImageDraw.rounded_rectangle: y1 must be greater than or equal to y0"; }
    $cs = $corners;
    if ($cs == null) { $cs = [true, true, true, true]; }
    $rad = $radius;
    if ($rad == null) { $rad = 0; }
    $w = $width;
    if ($w == null) { $w = 1; }

    $dd = min(min($x1 - $x0, $y1 - $y0), $rad * 2);
    $x0 = _pyRound($x0);
    $y0 = _pyRound($y0);
    $x1 = _pyRound($x1);
    $y1 = _pyRound($y1);
    $fullX = false;
    $fullY = false;
    if ($cs[0] && $cs[1] && $cs[2] && $cs[3]) {
        $fullX = $dd >= $x1 - $x0 - 1;
        if ($fullX) { $dd = $x1 - $x0; }
        $fullY = $dd >= $y1 - $y0 - 1;
        if ($fullY) { $dd = $y1 - $y0; }
        if ($fullX && $fullY) { return $d.ellipse($xy, $fill, $outline, $w); }
    }
    $anyCorner = $cs[0] || $cs[1] || $cs[2] || $cs[3];
    if ($dd == 0 || !$anyCorner) { return $d.rectangle($xy, $fill, $outline, $w); }

    $r = floor($dd / 2);
    $res = $d._getink($outline, $fill);
    $ink = $res[0];
    $fillInk = $res[1];
    $c = $d.c;

    $parts = [];
    if ($fullX) {
        $parts = [[$x0, $y0, $x0 + $dd, $y0 + $dd, 180, 360],
                  [$x0, $y1 - $dd, $x0 + $dd, $y1, 0, 180]];
    } else if ($fullY) {
        $parts = [[$x0, $y0, $x0 + $dd, $y0 + $dd, 90, 270],
                  [$x1 - $dd, $y0, $x1, $y0 + $dd, 270, 90]];
    } else {
        $all = [[$x0, $y0, $x0 + $dd, $y0 + $dd, 180, 270],
                [$x1 - $dd, $y0, $x1, $y0 + $dd, 270, 360],
                [$x1 - $dd, $y1 - $dd, $x1, $y1, 0, 90],
                [$x0, $y1 - $dd, $x0 + $dd, $y1, 90, 180]];
        $i = 0;
        while ($i < 4) {
            if ($cs[$i]) { push($parts, $all[$i]); }
            $i = $i + 1;
        }
    }

    if ($fillInk != null) {
        each ($p in $parts) {
            _drawPieslice($c, _int($p[0]), _int($p[1]), _int($p[2]), _int($p[3]),
                          $p[4], $p[5], $fillInk, true, 1);
        }
        if ($fullX) {
            _rectangle($c, $x0, $y0 + $r + 1, $x1, $y1 - $r - 1, $fillInk, true, 1);
        } else if ($x1 - $r - 1 >= $x0 + $r + 1) {
            _rectangle($c, $x0 + $r + 1, $y0, $x1 - $r - 1, $y1, $fillInk, true, 1);
        }
        if (!$fullX && !$fullY) {
            $lt = $y0;
            if ($cs[0]) { $lt = $lt + $r + 1; }
            $lb = $y1;
            if ($cs[3]) { $lb = $lb - $r - 1; }
            _rectangle($c, $x0, $lt, $x0 + $r, $lb, $fillInk, true, 1);
            $rt = $y0;
            if ($cs[1]) { $rt = $rt + $r + 1; }
            $rb = $y1;
            if ($cs[2]) { $rb = $rb - $r - 1; }
            _rectangle($c, $x1 - $r, $rt, $x1, $rb, $fillInk, true, 1);
        }
    }
    if ($ink != null && $w != 0 && _inkDiff($ink, $fillInk)) {
        each ($p in $parts) {
            _drawArc($c, _int($p[0]), _int($p[1]), _int($p[2]), _int($p[3]),
                     $p[4], $p[5], $ink, $w);
        }
        if (!$fullX) {
            $tl = $x0;
            if ($cs[0]) { $tl = $tl + $r + 1; }
            $tr = $x1;
            if ($cs[1]) { $tr = $tr - $r - 1; }
            _rectangle($c, $tl, $y0, $tr, $y0 + $w - 1, $ink, true, 1);
            $bl = $x0;
            if ($cs[3]) { $bl = $bl + $r + 1; }
            $br = $x1;
            if ($cs[2]) { $br = $br - $r - 1; }
            _rectangle($c, $bl, $y1 - $w + 1, $br, $y1, $ink, true, 1);
        }
        if (!$fullY) {
            $lt = $y0;
            if ($cs[0]) { $lt = $lt + $r + 1; }
            $lb = $y1;
            if ($cs[3]) { $lb = $lb - $r - 1; }
            _rectangle($c, $x0, $lt, $x0 + $w - 1, $lb, $ink, true, 1);
            $rt = $y0;
            if ($cs[1]) { $rt = $rt + $r + 1; }
            $rb = $y1;
            if ($cs[2]) { $rb = $rb - $r - 1; }
            _rectangle($c, $x1 - $w + 1, $rt, $x1, $rb, $ink, true, 1);
        }
    }
    return null;
}

// Python's round() on a single value: half to EVEN.
def _pyRound($v) {
    $s = 1;
    $m = $v;
    if ($m < 0) { $s = 0 - 1; $m = 0 - $m; }
    $fl = floor($m);
    $fr = $m - $fl;
    if ($fr > 0.5) { $fl = $fl + 1; }
    else if ($fr == 0.5 && $fl % 2 != 0) { $fl = $fl + 1; }
    return $s * $fl;
}
