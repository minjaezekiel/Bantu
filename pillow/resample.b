// ════════════════════════════════════════════════════════════════════════
//  pillow/resample.b — the six resampling filters, and resize().
//
//      include "./resample.b" as resample;
//      $out = resample.resize($px, "RGB", 320, 240, resample.BICUBIC, null);
//
//  Operates on PIXEL BUFFERS, not Image objects, so image.b includes this
//  and this includes nothing of image.b.
//
//  ─── THE ALGORITHM, from Pillow's Resample.c ─────────────────────────
//
//  Two one-dimensional passes, horizontal then vertical. A k-by-k kernel
//  costs 2k taps separated instead of k-squared, and -- the reason it is
//  possible here at all -- each pass becomes ONE ARRAY OPERATION PER TAP
//  rather than one per pixel.
//
//  precompute_coeffs, for an axis of `inSize` resampled to `outSize`:
//
//      filterscale = scale = (in1 - in0) / outSize,  floored at 1.0
//      support     = filter.support * filterscale
//      ksize       = ceil(support) * 2 + 1
//      for each output index xx:
//          center = in0 + (xx + 0.5) * scale
//          xmin   = clamp((int)(center - support + 0.5), 0, inSize)
//          xmax   = clamp((int)(center + support + 0.5), 0, inSize) - xmin
//          k[t]   = filter((t + xmin - center + 0.5) / filterscale)
//          then k is normalised to sum 1, and padded to ksize with zeros
//
//  `filterscale` floored at 1 is what makes DOWNSCALING average rather than
//  alias: the kernel widens with the ratio instead of sampling through it.
//
//  ─── HOW IT IS VECTORISED ────────────────────────────────────────────
//
//  Output column xx reads input columns xmin[xx] .. xmin[xx]+ksize-1. The
//  offsets differ per column, so a slice will not do -- but for a FIXED tap
//  index t, every output column reads exactly one input column, so tap t is
//  a gather of the index vector `xmin + t`:
//
//      for t in 0 .. ksize-1:
//          acc += nd_take(in, clamp(xmin + t), axis) * kcol[t]
//
//  That is `ksize` array operations for the whole pass -- 5 for bicubic at
//  scale 1, 17 for a 4x box reduction -- regardless of the pixel count.
//  Coefficients past a column's `count` are zero, so the clamped index reads
//  a harmless in-bounds pixel and contributes nothing.
//
//  ─── EXACTNESS ──────────────────────────────────────────────────────
//
//  Three numeric paths, because Pillow has three:
//
//    8-bit modes  fixed point. Coefficients become integers at
//                 PRECISION_BITS = 22 with round-half-away-from-zero; the
//                 accumulator starts at 2^21 (the rounding offset) and the
//                 result is `acc >> 22` clamped to 0..255. All integer, so
//                 there is no floating-point drift to match.
//    mode I       double accumulation from 0, then ROUND_UP -- truncate
//                 toward zero after adding a half, which is NOT floor for
//                 negative sums.
//    mode F       double accumulation, stored back through float32.
//
//  The kernels themselves are Pillow's, including two details that look
//  like typos and are not:
//
//    * the box filter is ASYMMETRIC: `x > -0.5 && x <= 0.5`, so exactly
//      -0.5 is outside the kernel and exactly +0.5 is inside.
//    * hamming's window constants are C `float` literals -- `0.54f` and
//      `0.46f` -- promoted to double for the arithmetic. The double nearest
//      to the float nearest to 0.54 is 0.5400000214576721, not 0.54, and
//      using the latter shifts the last bit of some weights.
// ════════════════════════════════════════════════════════════════════════

include "./mode.b" as mode;
include "./pxbuf.b" as pxbuf;

// Pillow's filter ids.
$NEAREST  = 0;
$LANCZOS  = 1;
$BILINEAR = 2;
$BICUBIC  = 3;
$BOX      = 4;
$HAMMING  = 5;

$FILTER_NAMES = ["NEAREST", "LANCZOS", "BILINEAR", "BICUBIC", "BOX", "HAMMING"];

// Resample.c's PRECISION_BITS = 32 - 8 - 2: eight bits of result, two spare
// for a kernel whose coefficients sum above 1 or below 0.
$PRECISION_BITS = 22;
$PRECISION_ONE = 4194304;          // 1 << 22
$PRECISION_HALF = 2097152;         // 1 << 21

// The exact doubles C's `0.54f` and `0.46f` promote to.
$_HAMMING_A = 0.5400000214576721;
$_HAMMING_B = 0.46000000834465027;


// ─── The kernels ────────────────────────────────────────────────────────

def support($filter) {
    if ($filter == $BOX)      { return 0.5; }
    if ($filter == $BILINEAR) { return 1.0; }
    if ($filter == $HAMMING)  { return 1.0; }
    if ($filter == $BICUBIC)  { return 2.0; }
    if ($filter == $LANCZOS)  { return 3.0; }
    if ($filter == $NEAREST)  { return 0.0; }
    throw "resample: unknown filter " + str($filter) +
          ". Use NEAREST, LANCZOS, BILINEAR, BICUBIC, BOX or HAMMING (0..5)";
}

def _sinc($x) {
    if ($x == 0.0) { return 1.0; }
    $t = $x * PI;
    return sin($t) / $t;
}

// kernel($filter, $x) -> the filter weight at x.
def kernel($filter, $x) {
    if ($filter == $BOX) {
        // Asymmetric on purpose: box_filter is `x > -0.5 && x <= 0.5`.
        if ($x > -0.5 && $x <= 0.5) { return 1.0; }
        return 0.0;
    }
    if ($filter == $BILINEAR) {
        $a = $x;
        if ($a < 0.0) { $a = 0.0 - $a; }
        if ($a < 1.0) { return 1.0 - $a; }
        return 0.0;
    }
    if ($filter == $HAMMING) {
        $a = $x;
        if ($a < 0.0) { $a = 0.0 - $a; }
        if ($a == 0.0) { return 1.0; }
        if ($a >= 1.0) { return 0.0; }
        $t = $a * PI;
        return sin($t) / $t * ($_HAMMING_A + $_HAMMING_B * cos($t));
    }
    if ($filter == $BICUBIC) {
        // a = -0.5, the Catmull-Rom parameter Pillow uses.
        $a = $x;
        if ($a < 0.0) { $a = 0.0 - $a; }
        if ($a < 1.0) { return ((-0.5 + 2.0) * $a - (-0.5 + 3.0)) * $a * $a + 1; }
        if ($a < 2.0) { return (((($a - 5) * $a) + 8) * $a - 4) * -0.5; }
        return 0.0;
    }
    if ($filter == $LANCZOS) {
        if ($x >= -3.0 && $x < 3.0) { return _sinc($x) * _sinc($x / 3); }
        return 0.0;
    }
    throw "resample: filter " + str($filter) + " has no kernel";
}


// ─── precompute_coeffs ──────────────────────────────────────────────────
//
// Returns {"ksize", "xmin" (list), "count" (list), "kk" (list of lists)}.
// The loop is over OUTPUT INDICES -- hundreds to thousands -- which rule PW2
// permits; nothing here touches a pixel.
def precompute($inSize, $in0, $in1, $outSize, $filter) {
    if ($outSize < 1) { throw "resample: output size must be at least 1"; }
    $scale = ($in1 - $in0) / $outSize;
    $filterscale = $scale;
    if ($filterscale < 1.0) { $filterscale = 1.0; }
    $sup = support($filter) * $filterscale;
    $ksize = floor(ceil($sup)) * 2 + 1;
    if ($ksize < 1) { $ksize = 1; }

    $xmins = [];
    $counts = [];
    $kk = [];
    $ss = 1.0 / $filterscale;

    $xx = 0;
    while ($xx < $outSize) {
        $center = $in0 + ($xx + 0.5) * $scale;

        // (int) truncates toward zero in C; both candidates clamp to the same
        // value once bounded, so floor is used and then bounded.
        $xmin = floor($center - $sup + 0.5);
        if ($xmin < 0) { $xmin = 0; }
        $xmax = floor($center + $sup + 0.5);
        if ($xmax > $inSize) { $xmax = $inSize; }
        $count = $xmax - $xmin;
        if ($count < 0) { $count = 0; }

        $row = [];
        $ww = 0.0;
        $t = 0;
        while ($t < $count) {
            $w = kernel($filter, ($t + $xmin - $center + 0.5) * $ss);
            push($row, $w);
            $ww = $ww + $w;
            $t = $t + 1;
        }
        if ($ww != 0.0) {
            $t = 0;
            while ($t < $count) { $row[$t] = $row[$t] / $ww; $t = $t + 1; }
        }
        // Pad to ksize so every output index has the same stride.
        while (len($row) < $ksize) { push($row, 0.0); }

        push($xmins, $xmin);
        push($counts, $count);
        push($kk, $row);
        $xx = $xx + 1;
    }
    return {"ksize": $ksize, "xmin": $xmins, "count": $counts, "kk": $kk,
            "inSize": $inSize, "outSize": $outSize};
}

// _toFixed($pre) -> the coefficients as 22-bit integers, rounded away from
// zero, which is normalize_coeffs_8bpc.
def _toFixed($pre) {
    $out = [];
    $xx = 0;
    while ($xx < $pre.outSize) {
        $row = [];
        $t = 0;
        while ($t < $pre.ksize) {
            $v = $pre.kk[$xx][$t] * $PRECISION_ONE;
            if ($v < 0) { push($row, 0 - floor(0.5 - $v)); }
            else { push($row, floor(0.5 + $v)); }
            $t = $t + 1;
        }
        push($out, $row);
        $xx = $xx + 1;
    }
    return $out;
}

// _coeffColumn($rows, $t, $outSize, $axis) -> tap t's coefficient for every
// output index, shaped to broadcast along the axis being resampled.
def _coeffColumn($rows, $t, $outSize, $axis, $dtype) {
    $col = [];
    $xx = 0;
    while ($xx < $outSize) { push($col, $rows[$xx][$t]); $xx = $xx + 1; }
    $a = nd($col, $dtype);
    if ($axis == 0) { return nd_reshape($a, [$outSize, 1, 1]); }
    return nd_reshape($a, [1, $outSize, 1]);
}

// _indexVector($xmin, $t, $inSize) -> the input index each output index reads
// for tap t, clamped in bounds. Coefficients past a column's count are zero,
// so a clamped index contributes nothing.
def _indexVector($xmin, $t, $inSize) {
    $idx = [];
    $xx = 0;
    while ($xx < len($xmin)) {
        $v = $xmin[$xx] + $t;
        if ($v < 0) { $v = 0; }
        if ($v > $inSize - 1) { $v = $inSize - 1; }
        push($idx, $v);
        $xx = $xx + 1;
    }
    return nd($idx, "i64");
}


// ─── The passes ─────────────────────────────────────────────────────────
//
// $axis is 0 for the vertical pass and 1 for the horizontal one. `kind` is
// "u8", "i32" or "f32", matching Pillow's three numeric paths.
def _pass($px, $axis, $pre, $kind) {
    $d = pxbuf.dims($px);
    $bands = $d[2];
    $outSize = $pre.outSize;
    $inSize = $pre.inSize;

    if ($axis == 0) { $shape = [$outSize, $d[1], $bands]; }
    else            { $shape = [$d[0], $outSize, $bands]; }

    if ($kind == "u8") {
        $rows = _toFixed($pre);
        $acc = nd_zeros($shape, "i64");
        nd_add($acc, $PRECISION_HALF, $acc);
        $cdt = "i64";
    } else {
        $rows = $pre.kk;
        $acc = nd_zeros($shape, "f64");
        $cdt = "f64";
    }

    $src = $px;
    if ($kind != "u8") { $src = nd_astype($px, "f64"); }

    // ONE array operation per tap. This is the whole point of the design.
    $t = 0;
    while ($t < $pre.ksize) {
        $kc = _coeffColumn($rows, $t, $outSize, $axis, $cdt);
        // A tap whose coefficient is zero for every output index contributes
        // nothing; skipping it saves a gather on the padded tail.
        if (pxbuf.scalar(nd_count_nonzero($kc)) > 0) {
            $idx = _indexVector($pre.xmin, $t, $inSize);
            $g = nd_take($src, $idx, $axis);
            $acc = nd_add($acc, nd_multiply($g, $kc));
        }
        $t = $t + 1;
    }

    if ($kind == "u8") {
        // clip8: an arithmetic shift, then clamp. nd_floor_divide floors on
        // negatives, which is what `>>` does in C.
        return nd_clip(nd_floor_divide($acc, $PRECISION_ONE), 0, 255);
    }
    if ($kind == "i32") {
        // ROUND_UP: truncate toward zero after adding a half, so it is floor
        // for a non-negative sum and ceil for a negative one.
        $pos = nd_floor(nd_add($acc, 0.5));
        $neg = nd_subtract(0, nd_floor(nd_add(nd_subtract(0, $acc), 0.5)));
        return nd_astype(nd_where(nd_greater_equal($acc, 0), $pos, $neg), "i64");
    }
    return pxbuf.to_f32($acc);
}


// ─── NEAREST ────────────────────────────────────────────────────────────
//
// Not a convolution: ImagingScaleAffine samples one source pixel per output
// pixel at `floor(in0 + scale * (x + 0.5))`. Two gathers, no accumulation.
def _nearest($px, $outW, $outH, $box) {
    $d = pxbuf.dims($px);
    $inH = $d[0];
    $inW = $d[1];
    $sx = ($box[2] - $box[0]) / $outW;
    $sy = ($box[3] - $box[1]) / $outH;

    // ImagingScaleAffine ACCUMULATES the source coordinate:
    //
    //     xo = a[2] + a[0] * 0.5;
    //     for each x:  xin = floor(xo);  xo += a[0];
    //
    // and that is not the same as floor(in0 + scale * (x + 0.5)). Repeated
    // addition drifts in the last bits, so a coordinate that the closed form
    // puts at 5.999999999999999 the accumulation can put at 6.0000000000001
    // -- one whole source row different. Resampling 12 rows to 11 hits it.
    // The accumulation is therefore reproduced, not simplified.
    $xi = [];
    $xo = $box[0] + $sx * 0.5;
    $x = 0;
    while ($x < $outW) {
        $v = floor($xo);
        if ($v < 0) { $v = 0; }
        if ($v > $inW - 1) { $v = $inW - 1; }
        push($xi, $v);
        $xo = $xo + $sx;
        $x = $x + 1;
    }
    $yi = [];
    $yo = $box[1] + $sy * 0.5;
    $y = 0;
    while ($y < $outH) {
        $v = floor($yo);
        if ($v < 0) { $v = 0; }
        if ($v > $inH - 1) { $v = $inH - 1; }
        push($yi, $v);
        $yo = $yo + $sy;
        $y = $y + 1;
    }
    $g = nd_take($px, nd($xi, "i64"), 1);
    return pxbuf.materialise(nd_take($g, nd($yi, "i64"), 0), nd_dtype($px));
}


// ─── resize ─────────────────────────────────────────────────────────────
//
// $box is [left, upper, right, lower] in SOURCE coordinates, or null for the
// whole image. Returns a new buffer; $px is never modified.
//
// The caller is responsible for the premultiply dance that Pillow's Python
// layer does for LA and RGBA (convert to La/RGBa, resize, convert back) and
// for forcing NEAREST on modes 1 and P -- image.b does both, because they
// need the conversion matrix and this layer has no business knowing modes.
def resize($px, $outW, $outH, $filter, $box) {
    if ($outW < 1 || $outH < 1) {
        throw "resample.resize: width and height must be at least 1 (got " +
              str($outW) + "x" + str($outH) + ")";
    }
    $d = pxbuf.dims($px);
    $inH = $d[0];
    $inW = $d[1];

    $b = $box;
    if ($b == null) { $b = [0, 0, $inW, $inH]; }
    if (len($b) != 4) { throw "resample.resize: box must be [left, upper, right, lower]"; }
    if ($b[0] < 0 || $b[1] < 0) { throw "resample.resize: the box offset cannot be negative"; }
    if ($b[2] > $inW || $b[3] > $inH) {
        throw "resample.resize: the box " + str($b) + " exceeds the image (" +
              str($inW) + "x" + str($inH) + ")";
    }
    if ($b[2] - $b[0] < 0 || $b[3] - $b[1] < 0) {
        throw "resample.resize: the box cannot be empty";
    }

    // Pillow's shortcut: an integral box whose size already matches the
    // requested size is a crop, with no resampling at all.
    if ($b[0] == floor($b[0]) && $b[1] == floor($b[1]) &&
        $b[2] - $b[0] == $outW && $b[3] - $b[1] == $outH) {
        return pxbuf.materialise(pxbuf.crop($px, $b[0], $b[1], $b[2], $b[3]),
                                 nd_dtype($px));
    }

    if ($filter == $NEAREST) { return _nearest($px, $outW, $outH, $b); }

    $kind = "u8";
    if (nd_dtype($px) == "f64") { $kind = "f32"; }

    $needH = ($outW != $inW || $b[0] != 0 || $b[2] != $inW);
    $needV = ($outH != $inH || $b[1] != 0 || $b[3] != $inH);
    if (!$needH && !$needV) { return pxbuf.materialise($px, nd_dtype($px)); }

    // THE PASS ORDER IS PART OF THE ANSWER, not just of the cost.
    //
    // Each pass rounds its result to the mode's precision, so the order
    // decides where that rounding lands. Pillow runs the VERTICAL pass first
    // when the height shrinks more than twice as much as the width, because
    // then the horizontal pass has fewer rows to walk:
    //
    //     horizontal_first = !((inH - outH) > 0 &&
    //                          (inH - outH) > (inW - outW) * 2)
    //
    // For 7x5 -> 13x3 that is false -- the height loses 2 rows while the
    // width GAINS 6 columns -- so vertical runs first. Running horizontal
    // first there gives answers that differ by one in about a quarter of the
    // pixels, which is how this was found: every single pass matched Pillow
    // exactly, and only the combination did not.
    $horizFirst = !((($inH - $outH) > 0) && (($inH - $outH) > ($inW - $outW) * 2));

    $cur = $px;
    if ($horizFirst) {
        if ($needH) { $cur = _pass($cur, 1, precompute($inW, $b[0], $b[2], $outW, $filter), $kind); }
        if ($needV) { $cur = _pass($cur, 0, precompute($inH, $b[1], $b[3], $outH, $filter), $kind); }
    } else {
        if ($needV) { $cur = _pass($cur, 0, precompute($inH, $b[1], $b[3], $outH, $filter), $kind); }
        if ($needH) { $cur = _pass($cur, 1, precompute($inW, $b[0], $b[2], $outW, $filter), $kind); }
    }
    return $cur;
}

// resize_i32($px, ...) -> the mode-I path, which rounds rather than shifting.
def resize_i32($px, $outW, $outH, $filter, $box) {
    $d = pxbuf.dims($px);
    $inH = $d[0];
    $inW = $d[1];
    $b = $box;
    if ($b == null) { $b = [0, 0, $inW, $inH]; }
    if ($filter == $NEAREST) { return _nearest($px, $outW, $outH, $b); }
    $needH = ($outW != $inW || $b[0] != 0 || $b[2] != $inW);
    $needV = ($outH != $inH || $b[1] != 0 || $b[3] != $inH);
    if (!$needH && !$needV) { return pxbuf.materialise($px, nd_dtype($px)); }
    $horizFirst = !((($inH - $outH) > 0) && (($inH - $outH) > ($inW - $outW) * 2));
    $cur = $px;
    if ($horizFirst) {
        if ($needH) { $cur = _pass($cur, 1, precompute($inW, $b[0], $b[2], $outW, $filter), "i32"); }
        if ($needV) { $cur = _pass($cur, 0, precompute($inH, $b[1], $b[3], $outH, $filter), "i32"); }
    } else {
        if ($needV) { $cur = _pass($cur, 0, precompute($inH, $b[1], $b[3], $outH, $filter), "i32"); }
        if ($needH) { $cur = _pass($cur, 1, precompute($inW, $b[0], $b[2], $outW, $filter), "i32"); }
    }
    return $cur;
}

// filter_name($f) / filter_id($name) -> the two spellings.
def filter_name($f) {
    if ($f < 0 || $f > 5) { return null; }
    return $FILTER_NAMES[$f];
}
def filter_id($name) {
    $i = indexOf($FILTER_NAMES, $name);
    if ($i < 0) { return null; }
    return $i;
}
