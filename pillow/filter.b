// ════════════════════════════════════════════════════════════════════════
//  pillow/filter.b — ImageFilter: four unrelated algorithms wearing one
//  name, all of them over pixel BUFFERS rather than Images (so image.b can
//  include this without a cycle).
//
//      the ten built-ins, and Kernel   Filter.c's ImagingFilter3x3/5x5
//      BoxBlur, GaussianBlur           BoxBlur.c's sliding accumulator
//      RankFilter/Median/Min/Max       RankFilter.c's per-window sort
//      ModeFilter                      ModeFilter.c's per-window histogram
//      Color3DLUT                      ColorLUT.c's trilinear interpolation
//
//  ─── FIVE CONTRACTS OF THE CONVOLUTION, FOUR OF THEM SURPRISES ───────
//
//  1. ONLY 3x3 AND 5x5 EXIST. ImagingFilter refuses every other size with
//     "bad kernel size", and mode F with "image has wrong mode". A 7x7
//     Kernel is not slow in Pillow, it is an exception.
//  2. AN IMAGE SMALLER THAN THE KERNEL IS RETURNED UNCHANGED, and that is
//     checked BEFORE the kernel size is -- so a 7x7 kernel on a 5x5 image
//     is a copy, not an error.
//  3. THE KERNEL IS APPLIED VERTICALLY FLIPPED. kernel[0..2] multiplies row
//     y+1, not y-1. Every built-in kernel is vertically symmetric, so this
//     is invisible until someone writes an asymmetric one -- which is why
//     the test uses one.
//  4. THE BORDER IS COPIED, NOT FILTERED. The outer row and column (two,
//     for 5x5) of the output are the INPUT's pixels, unweighted. Not
//     zero-padded, not edge-extended: copied.
//  5. THE ACCUMULATION IS float, NOT double, and the rounding is folded
//     into the offset: `offset += 0.5` once, then the cast truncates. So
//     every multiply and every add rounds to 24 bits of mantissa, the
//     kernel is pre-divided by its scale IN float32, and a row's taps are
//     summed left to right before being added to the accumulator. In f64
//     the result differs by one on a few pixels in a thousand -- exactly
//     the difference that looks like a bug in someone's test suite.
//
//  Measured consequence: a filter costs ~20 (3x3) or ~50 (5x5) whole-array
//  passes, because each of those float32 roundings is one. That is the
//  price of bit-exactness and it is still far below a pixel loop.
//
//  docs/pillow-architecture.md §7.1.
// ════════════════════════════════════════════════════════════════════════

include "./mode.b" as mode;
include "./pxbuf.b" as pxbuf;

// ─── float32, which the whole convolution is computed in ────────────────

// _f32($x) -> the float32 nearest x, as a Bantu number.
def _f32($x) {
    return nd_to_list(nd_frombytes(nd_tobytes(nd([$x], "f64"), "f32le"), "f32le", null))[0];
}

// _f32a($a) -> every value of an array rounded to float32. One pass.
def _f32a($a) {
    $s = nd_shape($a);
    $flat = nd_frombytes(nd_tobytes($a, "f32le"), "f32le", null);
    return nd_reshape($flat, $s);
}

// ─── The built-in filters, with Pillow's own kernels ────────────────────
//
// A filter is a DICT, because Bantu has no classes across module
// boundaries worth the trouble: {"kind": ..., ...}. pillow.b exposes the
// ten built-ins as module variables and the rest as constructors, so a
// caller writes pillow.BLUR or pillow.GaussianBlur(2) exactly as in Pillow.
def Kernel($size, $kern, $scale, $offset) {
    $ks = $size;
    if (type($size) == "list") {
        if (len($size) != 2 || $size[0] != $size[1]) {
            throw "ImageFilter.Kernel: only square 3x3 and 5x5 kernels exist";
        }
        $ks = $size[0];
    }
    if ($ks * $ks != len($kern)) {
        throw "not enough coefficients in kernel";
    }
    $sc = $scale;
    if ($sc == null) {
        // Pillow's default scale is the sum of the kernel.
        $sc = 0;
        $i = 0;
        while ($i < len($kern)) { $sc = $sc + $kern[$i]; $i = $i + 1; }
    }
    $off = $offset;
    if ($off == null) { $off = 0; }
    return {"kind": "kernel", "name": "Kernel", "size": $ks, "scale": $sc,
            "offset": $off, "kernel": $kern};
}

def _builtin($name, $ks, $scale, $offset, $kern) {
    return {"kind": "kernel", "name": $name, "size": $ks, "scale": $scale,
            "offset": $offset, "kernel": $kern};
}

$BLUR = _builtin("Blur", 5, 16, 0,
    [1, 1, 1, 1, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 1, 1, 1, 1]);
$CONTOUR = _builtin("Contour", 3, 1, 255, [0 - 1, 0 - 1, 0 - 1, 0 - 1, 8, 0 - 1, 0 - 1, 0 - 1, 0 - 1]);
$DETAIL = _builtin("Detail", 3, 6, 0, [0, 0 - 1, 0, 0 - 1, 10, 0 - 1, 0, 0 - 1, 0]);
$EDGE_ENHANCE = _builtin("Edge-enhance", 3, 2, 0,
    [0 - 1, 0 - 1, 0 - 1, 0 - 1, 10, 0 - 1, 0 - 1, 0 - 1, 0 - 1]);
$EDGE_ENHANCE_MORE = _builtin("Edge-enhance More", 3, 1, 0,
    [0 - 1, 0 - 1, 0 - 1, 0 - 1, 9, 0 - 1, 0 - 1, 0 - 1, 0 - 1]);
$EMBOSS = _builtin("Emboss", 3, 1, 128, [0 - 1, 0, 0, 0, 1, 0, 0, 0, 0]);
$FIND_EDGES = _builtin("Find Edges", 3, 1, 0,
    [0 - 1, 0 - 1, 0 - 1, 0 - 1, 8, 0 - 1, 0 - 1, 0 - 1, 0 - 1]);
$SHARPEN = _builtin("Sharpen", 3, 16, 0,
    [0 - 2, 0 - 2, 0 - 2, 0 - 2, 32, 0 - 2, 0 - 2, 0 - 2, 0 - 2]);
$SMOOTH = _builtin("Smooth", 3, 13, 0, [1, 1, 1, 1, 5, 1, 1, 1, 1]);
$SMOOTH_MORE = _builtin("Smooth More", 5, 100, 0,
    [1, 1, 1, 1, 1,
     1, 5, 5, 5, 1,
     1, 5, 44, 5, 1,
     1, 5, 5, 5, 1,
     1, 1, 1, 1, 1]);

def BoxBlur($radius) {
    $xy = _radiusPair($radius, "BoxBlur");
    if ($xy[0] < 0 || $xy[1] < 0) { throw "radius must be >= 0"; }
    return {"kind": "box", "name": "BoxBlur", "rx": $xy[0], "ry": $xy[1], "passes": 1};
}
def GaussianBlur($radius) {
    $xy = _radiusPair($radius, "GaussianBlur");
    return {"kind": "gauss", "name": "GaussianBlur", "rx": $xy[0], "ry": $xy[1]};
}
def UnsharpMask($radius, $percent, $threshold) {
    $r = $radius;
    if ($r == null) { $r = 2; }
    $p = $percent;
    if ($p == null) { $p = 150; }
    $t = $threshold;
    if ($t == null) { $t = 3; }
    return {"kind": "unsharp", "name": "UnsharpMask", "radius": $r,
            "percent": $p, "threshold": $t};
}
def RankFilter($size, $rank) {
    if ($size % 2 == 0) { throw "bad filter size"; }
    if ($rank < 0 || $rank >= $size * $size) { throw "bad rank value"; }
    return {"kind": "rank", "name": "Rank", "size": $size, "rank": $rank};
}
def MedianFilter($size) {
    $s = $size;
    if ($s == null) { $s = 3; }
    return RankFilter($s, floor($s * $s / 2));
}
def MinFilter($size) {
    $s = $size;
    if ($s == null) { $s = 3; }
    return RankFilter($s, 0);
}
def MaxFilter($size) {
    $s = $size;
    if ($s == null) { $s = 3; }
    return RankFilter($s, $s * $s - 1);
}
def ModeFilter($size) {
    $s = $size;
    if ($s == null) { $s = 3; }
    return {"kind": "mode", "name": "Mode", "size": $s};
}

// ─── Color3DLUT ─────────────────────────────────────────────────────────

// Color3DLUT($size, $table, $channels, $target_mode) -> the filter.
//
// $size is one number or [s1, s2, s3]; $table is a flat list of
// size1*size2*size3*channels floats in 0..1, with the CHANNEL varying
// fastest, then axis 1 (the red input), then axis 2, then axis 3.
def Color3DLUT($size, $table, $channels, $target_mode) {
    $c = $channels;
    if ($c == null) { $c = 3; }
    if ($c != 3 && $c != 4) { throw "Only 3 or 4 output channels are supported"; }
    $sz = $size;
    if (type($sz) != "list") { $sz = [$size, $size, $size]; }
    if (len($sz) != 3) { throw "Size should be either an integer or a tuple of three integers."; }
    $i = 0;
    while ($i < 3) {
        if ($sz[$i] < 2 || $sz[$i] > 65) {
            throw "Size " + str(1 + $i) + "D should be in [2, 65] range.";
        }
        $i = $i + 1;
    }
    $items = $sz[0] * $sz[1] * $sz[2];
    if (len($table) != $items * $c) {
        throw "The table should have either channels * size**3 float items " +
              "or size**3 items of channels-sized tuples with " +
              str($items * $c) + " items, got " + str(len($table));
    }
    return {"kind": "lut3d", "name": "Color 3D LUT", "size": $sz,
            "channels": $c, "table": $table, "mode": $target_mode};
}

// Color3DLUT_generate($size, $fn, $channels, $target_mode) -> the table
// filled by calling $fn(r, g, b) with each node's position in 0..1, which
// is Pillow's Color3DLUT.generate.
def Color3DLUT_generate($size, $fn, $channels, $target_mode) {
    $c = $channels;
    if ($c == null) { $c = 3; }
    $sz = $size;
    if (type($sz) != "list") { $sz = [$size, $size, $size]; }
    $table = [];
    $b = 0;
    while ($b < $sz[2]) {
        $g = 0;
        while ($g < $sz[1]) {
            $r = 0;
            while ($r < $sz[0]) {
                $v = $fn($r / ($sz[0] - 1), $g / ($sz[1] - 1), $b / ($sz[2] - 1));
                $k = 0;
                while ($k < $c) { push($table, $v[$k]); $k = $k + 1; }
                $r = $r + 1;
            }
            $g = $g + 1;
        }
        $b = $b + 1;
    }
    return Color3DLUT($sz, $table, $c, $target_mode);
}

// color_lut3d($px, $m, $f) -> ImagingColorLUT3D_linear.
//
// The arithmetic is fixed point throughout, and the three constants are the
// whole of it: a byte is scaled by (size - 1) / 255 into 18 fractional
// bits, the top bits index the cell and the next 15 are the interpolation
// weight, and the table itself is 16-bit with 255 at 255 << 6. The top end
// needs no clamp -- the truncation of the scale keeps the index below
// size - 1 for every size up to 65, which is a property of 255 and not an
// accident, and the C says so in a comment.
def color_lut3d($px, $m, $f) {
    if (mode.bands($m) < 3 || mode.bits_per_band($m) != 8) {
        throw "Color3DLUT: the image must have at least three 8-bit bands, not " + str($m);
    }
    $d = pxbuf.dims($px);
    $h = $d[0];
    $w = $d[1];
    $s1 = $f.size[0];
    $s2 = $f.size[1];
    $s3 = $f.size[2];
    $C = $f.channels;

    // The table in 16-bit fixed point: 1.0 becomes 255 << 6 = 16320, and
    // the conversion ROUNDS AWAY FROM ZERO (_imaging.c adds or subtracts
    // 0.5 before the cast). Truncating instead, or folding the
    // PRECISION_ROUNDING 32 into the multiplier, moves 24 of 429 values by
    // one -- which is how the right form was found.
    $tab = [];
    $i = 0;
    while ($i < len($f.table)) {
        $x = $f.table[$i] * 16320;
        if ($x >= 0) { $v = floor($x + 0.5); } else { $v = 0 - floor(0 - $x + 0.5); }
        if ($v < 0) { $v = 0; }
        if ($v > 16320) { $v = 16320; }
        push($tab, $v);
        $i = $i + 1;
    }
    $T = nd($tab, "i64");

    $sc1 = floor(($s1 - 1) / 255 * 262144);
    $sc2 = floor(($s2 - 1) / 255 * 262144);
    $sc3 = floor(($s3 - 1) / 255 * 262144);

    $p1 = nd_multiply(nd_clip(pxbuf.band($px, 0), 0, 255), $sc1);
    $p2 = nd_multiply(nd_clip(pxbuf.band($px, 1), 0, 255), $sc2);
    $p3 = nd_multiply(nd_clip(pxbuf.band($px, 2), 0, 255), $sc3);
    $i1 = nd_floor_divide($p1, 262144);
    $i2 = nd_floor_divide($p2, 262144);
    $i3 = nd_floor_divide($p3, 262144);
    $w1 = nd_floor_divide(nd_mod($p1, 262144), 8);
    $w2 = nd_floor_divide(nd_mod($p2, 262144), 8);
    $w3 = nd_floor_divide(nd_mod($p3, 262144), 8);

    $parts = [];
    $ch = 0;
    while ($ch < $C) {
        // The eight corners, each one gather.
        $corner = [];
        $d3 = 0;
        while ($d3 < 2) {
            $d2 = 0;
            while ($d2 < 2) {
                $d1 = 0;
                while ($d1 < 2) {
                    $off = nd_multiply(nd_add(nd_add(nd_add($i1, $d1),
                               nd_multiply(nd_add($i2, $d2), $s1)),
                               nd_multiply(nd_add($i3, $d3), $s1 * $s2)), $C);
                    push($corner, nd_reshape(
                        nd_take($T, nd_ravel(nd_add($off, $ch))), [$h, $w, 1]));
                    $d1 = $d1 + 1;
                }
                $d2 = $d2 + 1;
            }
            $d3 = $d3 + 1;
        }
        // lerp(a, b, w) = (a * (32768 - w) + b * w) >> 15, every step.
        $c00 = _lerp($corner[0], $corner[1], $w1);
        $c10 = _lerp($corner[2], $corner[3], $w1);
        $c01 = _lerp($corner[4], $corner[5], $w1);
        $c11 = _lerp($corner[6], $corner[7], $w1);
        $cA = _lerp($c00, $c10, $w2);
        $cB = _lerp($c01, $c11, $w2);
        $res = _lerp($cA, $cB, $w3);
        // clip8(x) = clamp((x + 32) >> 6)
        push($parts, nd_clip(nd_floor_divide(nd_add($res, 32), 64), 0, 255));
        $ch = $ch + 1;
    }
    // A three-channel table passes the fourth band through; a four-channel
    // one replaces it.
    if ($C == 3 && mode.bands($m) == 4) { push($parts, pxbuf.band($px, 3)); }
    while (len($parts) < mode.bands($m)) { push($parts, pxbuf.band($px, len($parts))); }
    return pxbuf.stack($parts, mode.dtype($m));
}

def _lerp($a, $b, $w) {
    return nd_floor_divide(nd_add(nd_multiply($a, nd_subtract(32768, $w)),
                                  nd_multiply($b, $w)), 32768);
}

def _radiusPair($r, $who) {
    if (type($r) == "list") {
        if (len($r) != 2) { throw "ImageFilter." + $who + ": radius must be one number or two"; }
        return [$r[0], $r[1]];
    }
    if ($r == null) { return [2, 2]; }
    return [$r, $r];
}

def filter_name($f) { return $f.name; }

// ─── apply($px, $mode, $f) -> a new buffer ──────────────────────────────
def apply($px, $m, $f) {
    if ($m == "P" || $m == "PA") { throw "cannot filter palette images"; }
    if ($f == null || type($f) != "dict" || $f.kind == null) {
        throw "Image.filter: expected a filter from ImageFilter, got " + str($f);
    }
    if ($f.kind == "kernel") { return kernel_filter($px, $m, $f); }
    if ($f.kind == "box") {
        return box_blur($px, $m, $f.rx, $f.ry, 1);
    }
    if ($f.kind == "gauss") { return gaussian_blur($px, $m, $f.rx, $f.ry); }
    if ($f.kind == "unsharp") {
        return unsharp_mask($px, $m, $f.radius, $f.percent, $f.threshold);
    }
    if ($f.kind == "rank") { return rank_filter($px, $m, $f.size, $f.rank); }
    if ($f.kind == "mode") { return mode_filter($px, $m, $f.size); }
    if ($f.kind == "lut3d") { return color_lut3d($px, $m, $f); }
    throw "Image.filter: no such filter kind '" + str($f.kind) + "'";
}

// ─── The convolution ────────────────────────────────────────────────────

// _clipFor($a, $m) -> the C's clip8 / clip32 / clip16 for this mode.
def _clipFor($a, $m) {
    if ($m == "I") {
        $t = nd_trunc(nd_clip($a, 0, 2147483647));
        return $t;
    }
    if ($m == "I;16" || $m == "I;16L" || $m == "I;16B" || $m == "I;16N") {
        return nd_trunc(nd_clip($a, 0, 65535));
    }
    return nd_trunc(nd_clip($a, 0, 255));
}

def kernel_filter($px, $m, $f) {
    if ($m == "F") { throw "image has wrong mode"; }
    $d = pxbuf.dims($px);
    $h = $d[0];
    $w = $d[1];
    $ks = $f.size;
    // Checked in Pillow's order: the small-image copy comes FIRST, so a 7x7
    // kernel on a 5x5 image is a copy and not an error.
    if ($w < $ks || $h < $ks) { return pxbuf.materialise($px, mode.dtype($m)); }
    if ($ks != 3 && $ks != 5) { throw "bad kernel size"; }

    $mm = floor($ks / 2);
    // float32(weight) / float32(scale), in float32 -- _imaging's _filter
    // divides the FLOAT32 kernel by the divisor before ImagingFilter sees it.
    $k32 = [];
    $i = 0;
    while ($i < $ks * $ks) {
        push($k32, _f32(_f32($f.kernel[$i]) / _f32($f.scale)));
        $i = $i + 1;
    }
    $off = _f32(_f32($f.offset) + 0.5);

    $ih = $h - 2 * $mm;
    $iw = $w - 2 * $mm;
    $acc = nd_full([$ih, $iw, $d[2]], $off, "f64");
    $r = 0;
    while ($r < $ks) {
        $dy = $mm - $r;                 // kernel row 0 reads the row BELOW
        $rowsum = null;
        $c = 0;
        while ($c < $ks) {
            $dx = $c - $mm;
            $tap = nd_slice($px, [[$mm + $dy, $h - $mm + $dy],
                                  [$mm + $dx, $w - $mm + $dx], null]);
            $term = _f32a(nd_multiply($tap, $k32[$r * $ks + $c]));
            if ($c == 0) { $rowsum = $term; }
            else { $rowsum = _f32a(nd_add($rowsum, $term)); }
            $c = $c + 1;
        }
        $acc = _f32a(nd_add($acc, $rowsum));
        $r = $r + 1;
    }

    // The border is the INPUT's pixels, copied.
    $out = pxbuf.materialise($px, mode.dtype($m));
    pxbuf.copy_into(nd_slice($out, [[$mm, $h - $mm], [$mm, $w - $mm], null]),
                    _clipFor($acc, $m));
    return $out;
}

// ─── The box blur, which is a prefix sum ────────────────────────────────

// _blurAxis($px, $radius, $axis) -> one horizontal (axis 1) or vertical
// (axis 0) box-blur pass, ImagingHorizontalBoxBlur's arithmetic exactly.
//
//     r  = (int) radius
//     ww = (1 << 24) / (radius * 2 + 1)      in FLOAT32, truncated
//     fw = ((1 << 24) - (r * 2 + 1) * ww) / 2    in integers
//     out(x) = (ww * Σ C(i), i in [x-r, x+r]
//               + fw * (C(x-r-1) + C(x+r+1)) + (1 << 23)) >> 24
//
// where C is the line clamped at both edges. The sliding accumulator of the
// C becomes one nd_cumsum: a window sum IS a difference of prefix sums, and
// the edge cases fall out of clamped index arrays rather than needing the
// C's edgeA/edgeB analysis. Six array operations, whatever the radius.
def _blurAxis($px, $radius, $axis) {
    $d = pxbuf.dims($px);
    $n = $d[$axis];
    $r = floor($radius);
    $ww = floor(_f32(16777216 / _f32($radius * 2 + 1)));
    $fw = floor((16777216 - (2 * $r + 1) * $ww) / 2);

    // The clamped line, padded by r+1 on each side.
    $idx = [];
    $p = 0;
    while ($p < $n + 2 * $r + 2) {
        $j = $p - $r - 1;
        if ($j < 0) { $j = 0; }
        if ($j > $n - 1) { $j = $n - 1; }
        push($idx, $j);
        $p = $p + 1;
    }
    $P = nd_take($px, nd($idx, "i64"), $axis);
    $C = nd_cumsum($P, $axis);

    if ($axis == 1) {
        $hi = nd_slice($C, [null, [2 * $r + 1, 2 * $r + 1 + $n], null]);
        $lo = nd_slice($C, [null, [0, $n], null]);
        $fa = nd_slice($P, [null, [0, $n], null]);
        $fb = nd_slice($P, [null, [2 * $r + 2, 2 * $r + 2 + $n], null]);
    } else {
        $hi = nd_slice($C, [[2 * $r + 1, 2 * $r + 1 + $n], null, null]);
        $lo = nd_slice($C, [[0, $n], null, null]);
        $fa = nd_slice($P, [[0, $n], null, null]);
        $fb = nd_slice($P, [[2 * $r + 2, 2 * $r + 2 + $n], null, null]);
    }
    $sum = nd_subtract($hi, $lo);
    $bulk = nd_add(nd_multiply($sum, $ww),
                   nd_multiply(nd_add($fa, $fb), $fw));
    return nd_floor_divide(nd_add($bulk, 8388608), 16777216);
}

// box_blur($px, $m, $rx, $ry, $n) -> ImagingBoxBlur: n passes along x,
// then n along y, each axis skipped entirely when its radius is 0.
def box_blur($px, $m, $rx, $ry, $n) {
    if ($n < 1) { throw "ImageFilter.BoxBlur: a blur needs at least one pass"; }
    if ($rx < 0 || $ry < 0) { throw "radius must be >= 0"; }
    if ($m == "F" || $m == "I" || $m == "P" || $m == "PA") {
        throw "ImageFilter.BoxBlur: cannot blur mode " + str($m);
    }
    $out = pxbuf.materialise($px, mode.dtype($m));
    if ($rx == 0 && $ry == 0) { return $out; }
    $k = 0;
    while ($k < $n) {
        if ($rx != 0) { $out = _blurAxis($out, $rx, 1); }
        $k = $k + 1;
    }
    $k = 0;
    while ($k < $n) {
        if ($ry != 0) { $out = _blurAxis($out, $ry, 0); }
        $k = $k + 1;
    }
    return pxbuf.materialise($out, mode.dtype($m));
}

// _gaussRadius($sigma, $passes) -> _gaussian_blur_radius: the box radius
// whose `passes`-fold convolution has the requested variance (Gwosdek et
// al. 2011, equations 7, 11 and 14). Computed in float32, as the C does.
def _gaussRadius($sigma, $passes) {
    if ($sigma == 0) { return 0; }
    $sigma2 = _f32(_f32($sigma) * _f32($sigma) / $passes);
    $L = _f32(sqrt(12 * $sigma2 + 1));
    $l = _f32(floor(($L - 1) / 2));
    $den = _f32(6 * ($sigma2 - ($l + 1) * ($l + 1)));
    if ($den == 0) { return $l; }
    $a = _f32(_f32((2 * $l + 1) * ($l * ($l + 1) - 3 * $sigma2)) / $den);
    return _f32($l + $a);
}

// gaussian_blur($px, $m, $rx, $ry) -> the box blur run three times per
// axis with the radius transformed. NOT a Gaussian kernel: Pillow's
// approximation is the contract (PW4), so it is reproduced and not improved.
def gaussian_blur($px, $m, $rx, $ry) {
    if ($rx == 0 && $ry == 0) { return pxbuf.materialise($px, mode.dtype($m)); }
    return box_blur($px, $m, _gaussRadius($rx, 3), _gaussRadius($ry, 3), 3);
}

// unsharp_mask($px, $m, $radius, $percent, $threshold) -> UnsharpMask.c.
//
//     blurred = GaussianBlur(radius, 3 passes)
//     diff    = original - blurred
//     out     = |diff| > threshold ? clip8(original + diff * percent / 100)
//                                  : original
//
// where `diff * percent / 100` is C integer division -- TRUNCATING TOWARD
// ZERO, which for a negative diff is not the floor.
def unsharp_mask($px, $m, $radius, $percent, $threshold) {
    if ($m != "L" && $m != "LA" && $m != "La" && $m != "RGB" && $m != "RGBA" &&
        $m != "RGBa" && $m != "RGBX" && $m != "CMYK" && $m != "YCbCr" && $m != "1") {
        throw "ImageFilter.UnsharpMask: cannot sharpen mode " + str($m);
    }
    $blur = gaussian_blur($px, $m, $radius, $radius);
    $diff = nd_subtract($px, $blur);
    $scaled = nd_multiply($diff, $percent);
    // truncate toward zero: sign(x) * floor(|x| / 100)
    $trunc = nd_multiply(nd_sign($scaled), nd_floor_divide(nd_abs($scaled), 100));
    $sharp = nd_clip(nd_add($px, $trunc), 0, 255);
    $use = nd_greater(nd_abs($diff), $threshold);
    return nd_where($use, $sharp, $px);
}

// ─── Rank filters sort a stack, not a window ────────────────────────────

// _expandEdge($px, $m2) -> the image grown by m2 on every side with EDGE
// REPLICATION, which is what ImagingExpand does and what RankFilter runs on.
def _expandEdge($px, $m2) {
    $d = pxbuf.dims($px);
    $out = $px;
    $idx = [];
    $i = 0;
    while ($i < $d[1] + 2 * $m2) {
        $j = $i - $m2;
        if ($j < 0) { $j = 0; }
        if ($j > $d[1] - 1) { $j = $d[1] - 1; }
        push($idx, $j);
        $i = $i + 1;
    }
    $out = nd_take($out, nd($idx, "i64"), 1);
    $idx = [];
    $i = 0;
    while ($i < $d[0] + 2 * $m2) {
        $j = $i - $m2;
        if ($j < 0) { $j = 0; }
        if ($j > $d[0] - 1) { $j = $d[0] - 1; }
        push($idx, $j);
        $i = $i + 1;
    }
    return nd_take($out, nd($idx, "i64"), 0);
}

// rank_filter($px, $m, $size, $rank) -> the rank-th smallest of each
// size x size window.
//
// A sort per pixel is a pixel loop. The arrangement that is not: expand
// once with edge replication (which is what Pillow's RankFilter does in
// Python, before the C sees it), take the size^2 shifted views (free),
// stack them on a new axis, and nd_sort along it -- ONE sort call per band
// for the whole image. The loop is over TAPS and BANDS, which rule PW2
// permits.
//
// The stack is built per BAND rather than for all of them at once, because
// (h, w, bands, size^2) for a 1080p image at size 5 is 1.2 GB and
// (h, w, size^2) is a quarter of that. RankFilter.c is single-band too, and
// Pillow's Image.filter splits the bands before calling it.
def rank_filter($px, $m, $size, $rank) {
    if ($m == "P" || $m == "PA") { throw "cannot filter palette images"; }
    if ($size % 2 == 0) { throw "bad filter size"; }
    if ($rank < 0 || $rank >= $size * $size) { throw "bad rank value"; }
    // RankFilter.c refuses only the "special" types, so I, I;16 and F sort
    // as happily as L does -- unlike the convolution, which refuses F.
    if (mode.is_palette($m)) { throw "cannot filter palette images"; }
    $d = pxbuf.dims($px);
    $h = $d[0];
    $w = $d[1];
    $nb = $d[2];
    $m2 = floor($size / 2);
    $big = _expandEdge($px, $m2);
    $k = $size * $size;

    $parts = [];
    $ci = 0;
    while ($ci < $nb) {
        $band = nd_reshape(pxbuf.materialise(pxbuf.band($big, $ci), "i64"),
                           [$h + 2 * $m2, $w + 2 * $m2]);
        $stack = nd_zeros([$h, $w, $k], "i64");
        $t = 0;
        $dy = 0;
        while ($dy < $size) {
            $dx = 0;
            while ($dx < $size) {
                $tap = nd_slice($band, [[$dy, $dy + $h], [$dx, $dx + $w]]);
                nd_add(nd_reshape(pxbuf.materialise2d($tap), [$h, $w, 1]), 0,
                       nd_slice($stack, [null, null, [$t, $t + 1]]));
                $t = $t + 1;
                $dx = $dx + 1;
            }
            $dy = $dy + 1;
        }
        $sorted = nd_sort($stack, 2);
        push($parts, nd_reshape(nd_slice($sorted, [null, null, [$rank, $rank + 1]]),
                                [$h, $w, 1]));
        $ci = $ci + 1;
    }
    return pxbuf.stack($parts, mode.dtype($m));
}

// mode_filter($px, $m, $size) -> the most frequent value in each window.
//
// Three contracts from ModeFilter.c, and the first two are surprises:
//
//   * EVERY pixel is filtered, with the window CLIPPED to the image -- not
//     edge-replicated like RankFilter's and not skipped like the
//     convolution's border. An edge window simply holds fewer samples.
//   * values occurring once or twice are IGNORED: the result is used only
//     if some value occurs more than twice, and otherwise the original
//     pixel is kept.
//   * the scan is strictly greater over ascending bins, so a tie goes to
//     the LOWEST value, and it starts at bin 0 rather than at the centre.
//
// The clipping falls out of padding with 256: no bin can ever equal it, so
// an out-of-image tap counts for nothing without a mask of its own.
def mode_filter($px, $m, $size) {
    if ($m == "P" || $m == "PA") { throw "cannot filter palette images"; }
    if (mode.dtype($m) != "i64" || mode.bits_per_band($m) > 8) {
        throw "ImageFilter.ModeFilter: cannot filter mode " + str($m);
    }
    $d = pxbuf.dims($px);
    $h = $d[0];
    $w = $d[1];
    $nb = $d[2];
    $m2 = floor($size / 2);
    $k = $size * $size;

    $parts = [];
    $ci = 0;
    while ($ci < $nb) {
        $centre = nd_reshape(nd_clip(pxbuf.materialise(pxbuf.band($px, $ci), "i64"),
                                     0, 255), [$h, $w]);
        // 256 outside the image: a value no bin can match.
        $pad = nd_full([$h + 2 * $m2, $w + 2 * $m2], 256, "i64");
        nd_add($centre, 0, nd_slice($pad, [[$m2, $m2 + $h], [$m2, $m2 + $w]]));
        $stack = nd_zeros([$h, $w, $k], "i64");
        $t = 0;
        $dy = 0;
        while ($dy < $size) {
            $dx = 0;
            while ($dx < $size) {
                $tap = nd_slice($pad, [[$dy, $dy + $h], [$dx, $dx + $w]]);
                nd_add(nd_reshape(pxbuf.materialise2d($tap), [$h, $w, 1]), 0,
                       nd_slice($stack, [null, null, [$t, $t + 1]]));
                $t = $t + 1;
                $dx = $dx + 1;
            }
            $dy = $dy + 1;
        }
        $best = $centre;
        $bestCount = nd_full([$h, $w], 2, "i64");
        $v = 0;
        while ($v < 256) {
            $cnt = nd_sum(nd_where(nd_equal($stack, $v), 1, 0), 2);
            $wins = nd_greater($cnt, $bestCount);
            $best = nd_where($wins, $v, $best);
            $bestCount = nd_where($wins, $cnt, $bestCount);
            $v = $v + 1;
        }
        push($parts, nd_reshape($best, [$h, $w, 1]));
        $ci = $ci + 1;
    }
    return pxbuf.stack($parts, mode.dtype($m));
}
