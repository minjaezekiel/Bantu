// ════════════════════════════════════════════════════════════════════════
//  pillow/transform.b — Image.transform and the four transform classes.
//
//      include "./transform.b" as xform;
//      $out = xform.apply($px, 200, 150, xform.AFFINE, $matrix,
//                         resample.BILINEAR, $fillList);
//
//  Operates on PIXEL BUFFERS, so image.b includes this and this includes
//  nothing of image.b.
//
//  ─── HOW A PER-PIXEL RESAMPLER IS VECTORISED ─────────────────────────
//
//  Every method here answers one question per OUTPUT pixel: which source
//  coordinate does it come from? That is a map, not a convolution, so the
//  coefficient trick resample.b uses does not apply -- the source offsets
//  are different for every pixel and in both axes at once.
//
//  It still vectorises, through a FLAT GATHER. The source is reshaped to
//  (H*W, bands); the source coordinate of every output pixel is computed as
//  two whole arrays; those become one flat index array `iy * W + ix`; and
//  nd_take along axis 0 fetches every pixel in one call. So:
//
//      NEAREST   1 gather
//      BILINEAR  4 gathers  (the 2x2 neighbourhood) + 3 blends
//      BICUBIC  16 gathers  (the 4x4 neighbourhood) + 5 cubics
//
//  -- a fixed, small number of array operations for any image size, which is
//  what rule PW2 demands. A 640,000-pixel bicubic transform is 16 gathers,
//  not 10 million coordinate computations.
//
//  ─── EXACTNESS, from Pillow's Geometry.c ─────────────────────────────
//
//  The source coordinate is taken at the PIXEL CENTRE: every method adds 0.5
//  to the output x and y before mapping. Then:
//
//    * out-of-range is tested on the UNSHIFTED coordinate --
//      `xin < 0 || xin >= xsize` -- and such a pixel takes the fill colour
//      rather than a clamped edge pixel. Clamping there would smear the edge
//      outward and is the usual way a rotated image grows a border.
//    * BILINEAR and BICUBIC then subtract 0.5 and floor, so the weights are
//      relative to the sample centre.
//    * the neighbourhood is CLAMPED (XCLIP/YCLIP) once inside, so an edge
//      pixel repeats rather than reading out of bounds...
//    * ...except downward, where Pillow does not clamp but CARRIES the last
//      valid row: `v2 = v1` when y+1 is past the bottom, and for bicubic
//      `v3 = v2`, `v4 = v3`. That is not the same as clamping to the last
//      row, and it is the detail that decides the bottom edge.
// ════════════════════════════════════════════════════════════════════════

include "./pxbuf.b" as pxbuf;
include "./resample.b" as resample;

// Pillow's transform method ids.
$AFFINE      = 0;
$EXTENT      = 1;
$PERSPECTIVE = 2;
$QUAD        = 3;
$MESH        = 4;

$METHOD_NAMES = ["AFFINE", "EXTENT", "PERSPECTIVE", "QUAD", "MESH"];

def method_name($m) {
    if ($m < 0 || $m > 4) { return null; }
    return $METHOD_NAMES[$m];
}


// ─── Coordinate maps ────────────────────────────────────────────────────
//
// Each returns {"xin": (h,w,1) f64, "yin": (h,w,1) f64}. The loops are over
// the OUTPUT axes only to build two ramps; everything else is array work.

// _ramps($w, $h) -> x+0.5 as (1,w,1) and y+0.5 as (h,1,1), ready to broadcast.
def _ramps($w, $h) {
    $xs = nd_add(nd_astype(nd_arange(0, $w, 1), "f64"), 0.5);
    $ys = nd_add(nd_astype(nd_arange(0, $h, 1), "f64"), 0.5);
    return {"x": nd_reshape($xs, [1, $w, 1]), "y": nd_reshape($ys, [$h, 1, 1])};
}

// affine: xin = a*x + b*y + c,  yin = d*x + e*y + f
def affine_coords($w, $h, $m) {
    if (len($m) < 6) { throw "transform: AFFINE needs 6 coefficients, got " + str(len($m)); }
    $r = _ramps($w, $h);
    $xin = nd_add(nd_add(nd_multiply($r.x, $m[0]), nd_multiply($r.y, $m[1])), $m[2]);
    $yin = nd_add(nd_add(nd_multiply($r.x, $m[3]), nd_multiply($r.y, $m[4])), $m[5]);
    // broadcast both to the full (h, w, 1)
    return {"xin": nd_add($xin, nd_multiply($r.y, 0.0)),
            "yin": nd_add($yin, nd_multiply($r.x, 0.0))};
}

// perspective: the affine numerator over (a6*x + a7*y + 1)
def perspective_coords($w, $h, $m) {
    if (len($m) < 8) { throw "transform: PERSPECTIVE needs 8 coefficients, got " + str(len($m)); }
    $r = _ramps($w, $h);
    $den = nd_add(nd_add(nd_multiply($r.x, $m[6]), nd_multiply($r.y, $m[7])), 1.0);
    $xn = nd_add(nd_add(nd_multiply($r.x, $m[0]), nd_multiply($r.y, $m[1])), $m[2]);
    $yn = nd_add(nd_add(nd_multiply($r.x, $m[3]), nd_multiply($r.y, $m[4])), $m[5]);
    return {"xin": nd_divide($xn, $den), "yin": nd_divide($yn, $den)};
}

// quad: a bilinear warp -- a0 + a1*x + a2*y + a3*x*y
def quad_coords($w, $h, $m) {
    if (len($m) < 8) { throw "transform: QUAD needs 8 coefficients, got " + str(len($m)); }
    $r = _ramps($w, $h);
    $xy = nd_multiply($r.x, $r.y);
    $xin = nd_add(nd_add(nd_add($m[0], nd_multiply($r.x, $m[1])),
                         nd_multiply($r.y, $m[2])), nd_multiply($xy, $m[3]));
    $yin = nd_add(nd_add(nd_add($m[4], nd_multiply($r.x, $m[5])),
                         nd_multiply($r.y, $m[6])), nd_multiply($xy, $m[7]));
    return {"xin": $xin, "yin": $yin};
}

// extent($box) -> the affine matrix Pillow derives for EXTENT.
//
// ImagingTransform converts EXTENT to an affine before doing anything, with
// the box mapped onto the whole output:
//     a = (x1 - x0) / outW,  c = x0
//     e = (y1 - y0) / outH,  f = y0
def extent_matrix($box, $w, $h) {
    if (len($box) != 4) { throw "transform: EXTENT needs a box of 4 values"; }
    $x0 = $box[0];
    $y0 = $box[1];
    $x1 = $box[2];
    $y1 = $box[3];
    return [($x1 - $x0) / $w, 0.0, $x0, 0.0, ($y1 - $y0) / $h, $y0];
}

// quad_matrix($quad, $w, $h) -> the 8 coefficients Pillow derives for QUAD.
//
// The quad is (nw, sw, se, ne) as x,y pairs -- Pillow's order, going
// anticlockwise from the top-left -- mapped onto the output rectangle.
def quad_matrix($q, $w, $h) {
    if (len($q) != 8) { throw "transform: QUAD needs 8 values (4 corner pairs)"; }
    $nwx = $q[0]; $nwy = $q[1];
    $swx = $q[2]; $swy = $q[3];
    $sex = $q[4]; $sey = $q[5];
    $nex = $q[6]; $ney = $q[7];
    $x0 = $nwx;
    $y0 = $nwy;
    $As = 1.0 / $w;
    $At = 1.0 / $h;
    return [$x0,
            ($nex - $x0) * $As,
            ($swx - $x0) * $At,
            ($sex - $swx - $nex + $x0) * $As * $At,
            $y0,
            ($ney - $y0) * $As,
            ($swy - $y0) * $At,
            ($sey - $swy - $ney + $y0) * $As * $At];
}


// ─── The flat gather ────────────────────────────────────────────────────

// _gather($flatSrc, $iy, $ix, $H, $W, $shape) -> the source pixels at the
// given integer coordinates, as (h, w, bands).
//
// Both coordinate arrays are CLAMPED first: an index that would leave the
// buffer is a crash, and every caller has already decided what an
// out-of-range pixel means (the fill colour, applied afterwards by a mask).
def _gather($flatSrc, $iy, $ix, $H, $W, $shape) {
    $cy = nd_clip($iy, 0, $H - 1);
    $cx = nd_clip($ix, 0, $W - 1);
    $idx = nd_add(nd_multiply($cy, $W), $cx);
    $g = nd_take($flatSrc, nd_ravel($idx), 0);
    return nd_reshape($g, $shape);
}

def _blend($a, $b, $d) { return nd_add($a, nd_multiply(nd_subtract($b, $a), $d)); }

// Pillow's BICUBIC macro: a cubic through four samples, Catmull-Rom form.
def _cubic($v1, $v2, $v3, $v4, $d) {
    $p1 = $v2;
    $p2 = nd_add(nd_subtract(0, $v1), $v3);
    $p3 = nd_subtract(nd_add(nd_multiply(nd_subtract($v1, $v2), 2), $v3), $v4);
    $p4 = nd_add(nd_subtract(nd_add(nd_subtract(0, $v1), $v2), $v3), $v4);
    return nd_add($p1, nd_multiply($d,
             nd_add($p2, nd_multiply($d, nd_add($p3, nd_multiply($d, $p4))))));
}


// ─── apply ──────────────────────────────────────────────────────────────
//
// $fill is a list with one value per band, used wherever the source
// coordinate falls outside the image. Pass null for zero.
def apply($px, $outW, $outH, $method, $data, $filter, $fill) {
    if ($outW < 1 || $outH < 1) {
        throw "transform: width and height must be at least 1 (got " +
              str($outW) + "x" + str($outH) + ")";
    }
    $d = pxbuf.dims($px);
    $H = $d[0];
    $W = $d[1];
    $bands = $d[2];
    $shape = [$outH, $outW, $bands];

    if ($method == $MESH) {
        return _mesh($px, $outW, $outH, $data, $filter, $fill);
    }

    // EXTENT IS AN AFFINE. Pillow's Python layer rewrites it to one before
    // any C is reached, so it inherits the AFFINE dispatch below -- which
    // for NEAREST is an integer path, not the closed form. Treating EXTENT
    // as its own double map instead disagrees on 6 of 60 pixels, because the
    // accumulated and closed forms land on different source columns where
    // the coordinate is near a whole number.
    $meth = $method;
    $dat = $data;
    if ($meth == $EXTENT) {
        $dat = extent_matrix($data, $outW, $outH);
        $meth = $AFFINE;
    }

    // AFFINE with NEAREST does not use the double coordinate map at all.
    // ImagingTransformAffine branches twice before reaching the generic
    // engine, and both branches are integer or accumulated, so neither
    // agrees with the closed form to the last pixel:
    //
    //   a[1] == 0 and a[3] == 0  ->  ImagingScaleAffine, which ACCUMULATES
    //                                the source coordinate per output pixel
    //   otherwise                ->  affine_fixed, 16.16 FIXED POINT
    //
    // A filter other than NEAREST falls through to the generic double engine.
    if ($meth == $AFFINE && $filter == resample.NEAREST) {
        if (len($dat) < 6) { throw "transform: AFFINE needs 6 coefficients, got " + str(len($dat)); }
        if ($dat[1] == 0 && $dat[3] == 0) {
            return _scaleAffine($px, $outW, $outH, $dat, $fill);
        }
        return _affineFixed($px, $outW, $outH, $dat, $fill);
    }

    // one coordinate map per method
    if ($meth == $AFFINE) { $c = affine_coords($outW, $outH, $dat); }
    else {
        if ($meth == $PERSPECTIVE) { $c = perspective_coords($outW, $outH, $dat); }
        else {
            if ($meth == $QUAD) { $c = quad_coords($outW, $outH, quad_matrix($dat, $outW, $outH)); }
            else {
                throw "transform: unknown method " + str($method) +
                      ". Use AFFINE, EXTENT, PERSPECTIVE, QUAD or MESH (0..4)";
            }
        }
    }
    return _sample($px, $c.xin, $c.yin, $shape, $filter, $fill);
}

// _scaleAffine -> ImagingScaleAffine: nearest neighbour with the source
// coordinate ACCUMULATED per output pixel rather than computed from it.
//
//     xo = a[2] + a[0] * 0.5;  per pixel: xin = COORD(xo); xo += a[0];
//
// Repeated addition drifts in the last bits, so a coordinate the closed form
// puts at 5.999999999999999 the accumulation can put at 6.0000000000001 --
// one whole source pixel different. resample.b's NEAREST has the same shape
// and the same reason.
def _scaleAffine($px, $outW, $outH, $a, $fill) {
    $d = pxbuf.dims($px);
    $H = $d[0];
    $W = $d[1];
    $xs = [];
    $xo = $a[2] + $a[0] * 0.5;
    $x = 0;
    while ($x < $outW) { push($xs, floor($xo)); $xo = $xo + $a[0]; $x = $x + 1; }
    $ys = [];
    $yo = $a[5] + $a[4] * 0.5;
    $y = 0;
    while ($y < $outH) { push($ys, floor($yo)); $yo = $yo + $a[4]; $y = $y + 1; }
    $ix = nd_reshape(nd($xs, "i64"), [1, $outW, 1]);
    $iy = nd_reshape(nd($ys, "i64"), [$outH, 1, 1]);
    return _nearestFrom($px, $ix, $iy, $outW, $outH, $fill);
}

// _affineFixed -> affine_fixed: 16.16 fixed point, all integer.
//
//     FIX(v) = floor(v * 65536 + 0.5)
//     A2 = FIX(a2 + a0/2 + a1/2),  A5 = FIX(a5 + a3/2 + a4/2)
//     xx(x, y) = A2 + y * A1 + x * A0   ->  xin = xx >> 16
//     yy(x, y) = A5 + y * A4 + x * A3   ->  yin = yy >> 16
//
// The C walks this with running accumulators; because every step is an
// integer addition the running form and the closed form agree exactly, so
// this computes it directly as two array expressions.
// FIX(v) = FLOOR(v * 65536.0 + 0.5), and Geometry.c's FLOOR is
//
//     #define FLOOR(v) ((v) < 0.0 ? ((int)floor(v)) : ((int)(v)))
//
// which is plain floor in BOTH branches: a cast truncates, and for a
// non-negative value truncation IS floor. So this rounds DOWN, not toward
// zero. Truncating toward zero instead makes FIX(-0.3) come out 19660/65536
// too high, which moves one output pixel of a rotated image to the wrong
// source column.
def _fix($v) {
    return floor($v * 65536.0 + 0.5);
}

def _affineFixed($px, $outW, $outH, $a, $fill) {
    $A0 = _fix($a[0]);
    $A1 = _fix($a[1]);
    $A3 = _fix($a[3]);
    $A4 = _fix($a[4]);
    $A2 = _fix($a[2] + $a[0] * 0.5 + $a[1] * 0.5);
    $A5 = _fix($a[5] + $a[3] * 0.5 + $a[4] * 0.5);

    $xr = nd_reshape(nd_arange(0, $outW, 1), [1, $outW, 1]);
    $yr = nd_reshape(nd_arange(0, $outH, 1), [$outH, 1, 1]);
    // `>> 16` is an arithmetic shift, i.e. floor division, which is what
    // nd_floor_divide does -- and these values go negative.
    $xx = nd_add(nd_add($A2, nd_multiply($yr, $A1)), nd_multiply($xr, $A0));
    $yy = nd_add(nd_add($A5, nd_multiply($yr, $A4)), nd_multiply($xr, $A3));
    $ix = nd_floor_divide($xx, 65536);
    $iy = nd_floor_divide($yy, 65536);
    return _nearestFrom($px, $ix, $iy, $outW, $outH, $fill);
}

// _nearestFrom -> gather at integer coordinates, filling outside the source.
// The integer paths test the INDEX against the bounds, where the double
// paths test the coordinate before flooring; for these the index is all
// there is.
def _nearestFrom($px, $ix, $iy, $outW, $outH, $fill) {
    $d = pxbuf.dims($px);
    $H = $d[0];
    $W = $d[1];
    $bands = $d[2];
    $shape = [$outH, $outW, $bands];
    $isFloat = (nd_dtype($px) == "f64");
    $srcF = $px;
    if (!$isFloat) { $srcF = nd_astype($px, "f64"); }
    $flat = nd_reshape(nd_ascontiguous($srcF), [$H * $W, $bands]);

    $ixb = nd_add($ix, nd_multiply($iy, 0));
    $iyb = nd_add($iy, nd_multiply($ix, 0));
    $inside = nd_logical_and(
        nd_logical_and(nd_greater_equal($ixb, 0), nd_less($ixb, $W)),
        nd_logical_and(nd_greater_equal($iyb, 0), nd_less($iyb, $H)));
    $v = _gather($flat, $iyb, $ixb, $H, $W, $shape);
    return _applyFill($v, $inside, $shape, $fill, $isFloat, $px);
}

// _sample($px, $xin, $yin, $shape, $filter, $fill) -> the resampled buffer.
def _sample($px, $xin, $yin, $shape, $filter, $fill) {
    $d = pxbuf.dims($px);
    $H = $d[0];
    $W = $d[1];
    $bands = $d[2];
    $isFloat = (nd_dtype($px) == "f64");
    $srcF = $px;
    if (!$isFloat) { $srcF = nd_astype($px, "f64"); }
    $flat = nd_reshape(nd_ascontiguous($srcF), [$H * $W, $bands]);

    // Out of range is decided on the UNSHIFTED coordinate, before the
    // half-pixel adjustment, and such a pixel takes the fill colour.
    $inside = nd_logical_and(
        nd_logical_and(nd_greater_equal($xin, 0.0), nd_less($xin, $W)),
        nd_logical_and(nd_greater_equal($yin, 0.0), nd_less($yin, $H)));

    if ($filter == resample.NEAREST) {
        // COORD(v) = v < 0 ? -1 : (int)v, and a negative is already outside.
        $ix = nd_astype(nd_floor($xin), "i64");
        $iy = nd_astype(nd_floor($yin), "i64");
        $v = _gather($flat, $iy, $ix, $H, $W, $shape);
        return _applyFill($v, $inside, $shape, $fill, $isFloat, $px);
    }

    $xs = nd_subtract($xin, 0.5);
    $ys = nd_subtract($yin, 0.5);
    $x = nd_floor($xs);
    $y = nd_floor($ys);
    $dx = nd_subtract($xs, $x);
    $dy = nd_subtract($ys, $y);
    $xi = nd_astype($x, "i64");
    $yi = nd_astype($y, "i64");

    if ($filter == resample.BILINEAR) {
        // Rows: y is clamped by YCLIP, but y+1 is NOT -- past the bottom
        // Pillow carries the previous row's value instead.
        $r0 = _rowBilinear($flat, $yi, $xi, $dx, $H, $W, $shape);
        $hasNext = nd_less(nd_add($yi, 1), $H);
        $r1 = _rowBilinear($flat, nd_add($yi, 1), $xi, $dx, $H, $W, $shape);
        $r1 = nd_where($hasNext, $r1, $r0);
        $v = _blend($r0, $r1, $dy);
        return _applyFill($v, $inside, $shape, $fill, $isFloat, $px);
    }

    if ($filter == resample.BICUBIC) {
        $xi = nd_subtract($xi, 1);
        $yi = nd_subtract($yi, 1);
        $v1 = _rowBicubic($flat, $yi, $xi, $dx, $H, $W, $shape);
        $v2 = nd_where(nd_logical_and(nd_greater_equal(nd_add($yi, 1), 0),
                                      nd_less(nd_add($yi, 1), $H)),
                       _rowBicubic($flat, nd_add($yi, 1), $xi, $dx, $H, $W, $shape), $v1);
        $v3 = nd_where(nd_logical_and(nd_greater_equal(nd_add($yi, 2), 0),
                                      nd_less(nd_add($yi, 2), $H)),
                       _rowBicubic($flat, nd_add($yi, 2), $xi, $dx, $H, $W, $shape), $v2);
        $v4 = nd_where(nd_logical_and(nd_greater_equal(nd_add($yi, 3), 0),
                                      nd_less(nd_add($yi, 3), $H)),
                       _rowBicubic($flat, nd_add($yi, 3), $xi, $dx, $H, $W, $shape), $v3);
        $v = _cubic($v1, $v2, $v3, $v4, $dy);
        return _applyFill($v, $inside, $shape, $fill, $isFloat, $px);
    }

    throw "transform: filter " + str($filter) +
          " is not supported. Use NEAREST, BILINEAR or BICUBIC";
}

def _rowBilinear($flat, $yi, $xi, $dx, $H, $W, $shape) {
    $a = _gather($flat, $yi, $xi, $H, $W, $shape);
    $b = _gather($flat, $yi, nd_add($xi, 1), $H, $W, $shape);
    return _blend($a, $b, $dx);
}

def _rowBicubic($flat, $yi, $xi, $dx, $H, $W, $shape) {
    $a = _gather($flat, $yi, $xi, $H, $W, $shape);
    $b = _gather($flat, $yi, nd_add($xi, 1), $H, $W, $shape);
    $c = _gather($flat, $yi, nd_add($xi, 2), $H, $W, $shape);
    $e = _gather($flat, $yi, nd_add($xi, 3), $H, $W, $shape);
    return _cubic($a, $b, $c, $e, $dx);
}

// _applyFill -> the fill colour outside the source, and the result narrowed
// back to the source's dtype.
def _applyFill($v, $inside, $shape, $fill, $isFloat, $px) {
    $out = $v;
    if ($fill != null) {
        $bands = $shape[2];
        $parts = [];
        $i = 0;
        while ($i < $bands) {
            $band = nd_slice($out, [null, null, [$i, $i + 1]]);
            $m = nd_slice($inside, [null, null, [0, 1]]);
            push($parts, nd_where($m, $band, $fill[$i] * 1.0));
            $i = $i + 1;
        }
        $out = pxbuf.stack($parts, "f64");
    } else {
        $out = nd_where($inside, $out, 0.0);
    }
    if ($isFloat) { return pxbuf.to_f32($out); }
    // Pillow stores the sampled value with a C CAST -- `(UINT8)v1` -- which
    // TRUNCATES toward zero. Rounding instead disagrees with it on roughly
    // half of all interpolated pixels, which is how this was found: NEAREST
    // matched almost everywhere and BILINEAR differed on 23 of 60.
    //
    // For a NEGATIVE value the C cast is undefined behaviour, and bicubic
    // genuinely undershoots below zero on a high-contrast edge -- on x86 it
    // happens to wrap, so -5 becomes 251, a bright speck in a dark area.
    // That is a defect rather than a decision (PW11), so this CLAMPS at zero
    // instead of wrapping: identical wherever the cast is defined, and
    // saturating rather than wrapping where it is not.
    return nd_astype(nd_clip(_truncToward0($out), 0, 255), "i64");
}

// _truncToward0($v) -> toward zero, as a C cast does, where nd_floor rounds
// down. The two differ on every negative value.
def _truncToward0($v) {
    return nd_where(nd_less($v, 0), nd_subtract(0, nd_floor(nd_subtract(0, $v))),
                    nd_floor($v));
}


// ─── MESH ───────────────────────────────────────────────────────────────
//
// A list of [box, quad] pairs: each box of the OUTPUT is filled from its own
// quad of the source. The loop is over mesh entries -- a handful -- and each
// entry is a full vectorised quad transform over its own rectangle.
def _mesh($px, $outW, $outH, $data, $filter, $fill) {
    $d = pxbuf.dims($px);
    $bands = $d[2];
    $out = nd_zeros([$outH, $outW, $bands], nd_dtype($px));
    if ($fill != null) { pxbuf.fill($out, $fill); }
    if (type($data) != "list") { throw "transform: MESH needs a list of [box, quad] pairs"; }

    $i = 0;
    while ($i < len($data)) {
        $entry = $data[$i];
        if (type($entry) != "list" || len($entry) != 2) {
            throw "transform: MESH entry " + str($i) + " must be [box, quad]";
        }
        $box = $entry[0];
        $quad = $entry[1];
        if (len($box) != 4) { throw "transform: MESH entry " + str($i) + " has a bad box"; }
        $bx0 = floor($box[0]);
        $by0 = floor($box[1]);
        $bx1 = floor($box[2]);
        $by1 = floor($box[3]);
        if ($bx0 < 0) { $bx0 = 0; }
        if ($by0 < 0) { $by0 = 0; }
        if ($bx1 > $outW) { $bx1 = $outW; }
        if ($by1 > $outH) { $by1 = $outH; }
        $bw = $bx1 - $bx0;
        $bh = $by1 - $by0;
        if ($bw > 0 && $bh > 0) {
            // The quad maps this box's own coordinate space, so the ramps run
            // from 0 within the box and are then offset to the output.
            $m = quad_matrix($quad, $bw, $bh);
            $c = quad_coords($bw, $bh, $m);
            $sub = _sample($px, $c.xin, $c.yin, [$bh, $bw, $bands], $filter, $fill);
            pxbuf.copy_into(pxbuf.crop($out, $bx0, $by0, $bx1, $by1), $sub);
        }
        $i = $i + 1;
    }
    return $out;
}


// ─── The four transform classes, as Pillow exposes them ─────────────────
//
// Each is a {"method", "data"} pair, which Image.transform unpacks. They
// exist so a caller can name the intent instead of remembering an id.
def AffineTransform($matrix)   { return {"method": $AFFINE, "data": $matrix}; }
def ExtentTransform($box)      { return {"method": $EXTENT, "data": $box}; }
def PerspectiveTransform($m)   { return {"method": $PERSPECTIVE, "data": $m}; }
def QuadTransform($quad)       { return {"method": $QUAD, "data": $quad}; }
def MeshTransform($entries)    { return {"method": $MESH, "data": $entries}; }


// ─── transpose ──────────────────────────────────────────────────────────
//
// The eight operations, all of them views or axis permutations, so each is
// free and exact by construction rather than by arithmetic.
$FLIP_LEFT_RIGHT = 0;
$FLIP_TOP_BOTTOM = 1;
$ROTATE_90       = 2;
$ROTATE_180      = 3;
$ROTATE_270      = 4;
$TRANSPOSE       = 5;
$TRANSVERSE      = 6;

def transpose($px, $op) {
    $dt = nd_dtype($px);
    if ($op == $FLIP_LEFT_RIGHT) { return pxbuf.materialise(nd_flip($px, 1), $dt); }
    if ($op == $FLIP_TOP_BOTTOM) { return pxbuf.materialise(nd_flip($px, 0), $dt); }
    if ($op == $ROTATE_180) {
        return pxbuf.materialise(nd_flip(nd_flip($px, 0), 1), $dt);
    }
    // TRANSPOSE is the main diagonal: swap the row and column axes, leaving
    // the band axis alone.
    if ($op == $TRANSPOSE) {
        return pxbuf.materialise(nd_swapaxes($px, 0, 1), $dt);
    }
    // ROTATE_90 is anticlockwise, which is transpose then flip top-to-bottom.
    if ($op == $ROTATE_90) {
        return pxbuf.materialise(nd_flip(nd_swapaxes($px, 0, 1), 0), $dt);
    }
    // ROTATE_270 is clockwise: transpose then flip left-to-right.
    if ($op == $ROTATE_270) {
        return pxbuf.materialise(nd_flip(nd_swapaxes($px, 0, 1), 1), $dt);
    }
    // TRANSVERSE is the anti-diagonal: transpose then rotate 180.
    if ($op == $TRANSVERSE) {
        return pxbuf.materialise(nd_flip(nd_flip(nd_swapaxes($px, 0, 1), 0), 1), $dt);
    }
    throw "transpose: unknown operation " + str($op) +
          ". Use FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM, ROTATE_90, ROTATE_180, " +
          "ROTATE_270, TRANSPOSE or TRANSVERSE (0..6)";
}


// ─── reduce ─────────────────────────────────────────────────────────────
//
// Integer box averaging, from Reduce.c. The arithmetic is fixed point:
//
//     multiplier = (UINT32)(2^32 / (256 * scale))
//     ss         = scale / 2  +  sum of the block
//     out        = (ss * multiplier) >> 24
//
// where `scale` is the number of pixels in the block. A block at the right
// or bottom edge that is not full uses ITS OWN smaller scale, so the edge is
// an average of what is there rather than of zeros -- that is
// ImagingReduceCorners, and skipping it darkens the last row and column.
//
// Vectorised: a full-block region is reshaped to (h', ys, w', xs, bands) and
// summed over the two block axes, which is two reductions for the whole
// image whatever the factor.
def _multiplier($scale) {
    // 2^32 / (256 * scale), truncated, exactly as division_UINT32 computes it.
    return floor(4294967296.0 / (256.0 * $scale));
}

// _reduceBlock32($region, $ys, $xs, $isInt) -> the 32-bit path.
//
// ImagingReduceNxN_32bpc accumulates in DOUBLE from zero and multiplies by
// 1/(ys*xs) -- a plain mean, with none of the 8-bit path's fixed-point
// reciprocal or its scale/2 rounding offset.
//
// THE SUBTLETY THAT DECIDES MODE F. The C writes
//
//     double ss = 0;
//     ss += line0[xx + 0] + line0[xx + 1] + line1[xx + 0] + line1[xx + 1];
//
// and for a FLOAT32 image every operand of that inner sum is a float, so by
// C's usual arithmetic conversions the four values are added IN FLOAT32 --
// rounding at each step -- and only the group total is widened into the
// double accumulator. Summing the whole block in double instead agrees on
// most pixels and differs by one or two float32 steps on the rest (measured:
// 6 of 20 output pixels). Mode I is unaffected, because integer addition is
// exact either way, which is why it matched before this was found.
//
// So the block is walked in Pillow's own 2x2 groups: pairs of rows by pairs
// of columns, with an odd last column and an odd last row handled exactly
// where the C handles them.
//
// RESIDUAL, measured and left alone deliberately. Reduce.c does not use this
// general routine for the common factors: it dispatches to eleven
// hand-unrolled kernels (1x2, 2x1, 2x2, 1x3, 3x1, 3x3, 4x4, 5x5, 1xN, Nx1)
// whose float addition order differs from the general one. For u8 and I that
// is invisible, because integer addition is associative; for F it moves the
// last float32 step on about one output pixel in forty (measured: 10 of 105
// corpus images differ, each by one or two values of a single ULP).
//
// Reproducing eleven unrolled summation orders to chase one ULP in a mode
// that is rarely reduced is a poor trade: it is eleven more code paths to
// keep correct for a difference no viewer can see and no arithmetic depends
// on. The general kernel is used for every factor, the gap is asserted and
// bounded by tests/pillow_geometry_test.b rather than hidden, and u8 and I
// remain exact.
def _reduceBlock32($region, $ys, $xs, $isInt) {
    $d = pxbuf.dims($region);
    $oh = floor($d[0] / $ys);
    $ow = floor($d[1] / $xs);
    $b = $d[2];
    $acc = nd_zeros([$oh, $ow, $b], "f64");

    // cell($dy, $dx) -> the (dy, dx) offset pixel of every block, as an
    // (oh, ow, bands) strided view.
    $dy = 0;
    while ($dy < $ys - 1) {
        $dx = 0;
        while ($dx < $xs - 1) {
            $g = _cell($region, $dy, $dx, $oh, $ow, $ys, $xs);
            $g = _add32($g, _cell($region, $dy, $dx + 1, $oh, $ow, $ys, $xs), $isInt);
            $g = _add32($g, _cell($region, $dy + 1, $dx, $oh, $ow, $ys, $xs), $isInt);
            $g = _add32($g, _cell($region, $dy + 1, $dx + 1, $oh, $ow, $ys, $xs), $isInt);
            $acc = nd_add($acc, $g);
            $dx = $dx + 2;
        }
        if ($xs % 2 == 1) {
            $g = _cell($region, $dy, $xs - 1, $oh, $ow, $ys, $xs);
            $g = _add32($g, _cell($region, $dy + 1, $xs - 1, $oh, $ow, $ys, $xs), $isInt);
            $acc = nd_add($acc, $g);
        }
        $dy = $dy + 2;
    }
    if ($ys % 2 == 1) {
        $ly = $ys - 1;
        $dx = 0;
        while ($dx < $xs - 1) {
            $g = _cell($region, $ly, $dx, $oh, $ow, $ys, $xs);
            $g = _add32($g, _cell($region, $ly, $dx + 1, $oh, $ow, $ys, $xs), $isInt);
            $acc = nd_add($acc, $g);
            $dx = $dx + 2;
        }
        if ($xs % 2 == 1) {
            $acc = nd_add($acc, _cell($region, $ly, $xs - 1, $oh, $ow, $ys, $xs));
        }
    }

    $mean = nd_multiply($acc, 1.0 / ($ys * $xs));
    // Mode F stores the mean through a FLOAT32.
    if (!$isInt) { return pxbuf.to_f32($mean); }
    // ROUND_UP: (int)(f >= 0 ? f + 0.5 : f - 0.5), truncating toward zero.
    $pos = nd_floor(nd_add($mean, 0.5));
    $neg = nd_subtract(0, nd_floor(nd_add(nd_subtract(0, $mean), 0.5)));
    return nd_where(nd_greater_equal($mean, 0), $pos, $neg);
}

// _cell -> the (dy, dx) pixel of every block, as an (oh, ow, bands) view.
def _cell($region, $dy, $dx, $oh, $ow, $ys, $xs) {
    return nd_astype(nd_slice($region,
        [[$dy, $oh * $ys + $dy, $ys], [$dx, $ow * $xs + $dx, $xs], null]), "f64");
}

// _add32 -> add, rounding to float32 afterwards for the F path, because
// that is what adding two C floats does. Mode I adds exactly.
def _add32($a, $b, $isInt) {
    $t = nd_add($a, $b);
    if ($isInt) { return $t; }
    return pxbuf.to_f32($t);
}

def _reduceBlock($region, $ys, $xs) {
    // $region is (h'*ys, w'*xs, bands) -- an exact multiple of the factors.
    $d = pxbuf.dims($region);
    $oh = floor($d[0] / $ys);
    $ow = floor($d[1] / $xs);
    $b = $d[2];
    $scale = $ys * $xs;
    $acc = nd_zeros([$oh, $ow, $b], "i64");
    nd_add($acc, floor($scale / 2), $acc);
    $dy = 0;
    while ($dy < $ys) {
        $dx = 0;
        while ($dx < $xs) {
            $acc = nd_add($acc, nd_slice($region,
                [[$dy, $oh * $ys + $dy, $ys], [$dx, $ow * $xs + $dx, $xs], null]));
            $dx = $dx + 1;
        }
        $dy = $dy + 1;
    }
    return nd_floor_divide(nd_multiply($acc, _multiplier($scale)), 16777216);
}

// reduce($px, $xs, $ys, $box, $kind)
//
// $kind selects the arithmetic: "u8" for the 8-bit modes, "i32" for mode I,
// "f32" for mode F. image.b passes it, because this layer does not know
// about modes.
def reduce($px, $xs, $ys, $box, $kind) {
    if ($xs < 1 || $ys < 1) {
        throw "reduce: the factor must be at least 1 (got " + str($xs) + "x" + str($ys) + ")";
    }
    $k = $kind;
    if ($k == null) { $k = "u8"; }
    $d = pxbuf.dims($px);
    $b = $box;
    if ($b == null) { $b = [0, 0, $d[1], $d[0]]; }
    $bw = $b[2] - $b[0];
    $bh = $b[3] - $b[1];
    if ($bw < 0 || $bh < 0) { throw "reduce: the box cannot be empty"; }
    if ($b[0] < 0 || $b[1] < 0 || $b[2] > $d[1] || $b[3] > $d[0]) {
        throw "reduce: the box " + str($b) + " is outside the image";
    }
    if ($xs == 1 && $ys == 1) {
        return pxbuf.materialise(pxbuf.crop($px, $b[0], $b[1], $b[2], $b[3]), nd_dtype($px));
    }

    // Pillow rounds the output size UP, so a partial block gets its own cell.
    $fullW = floor($bw / $xs);
    $fullH = floor($bh / $ys);
    $remW = $bw - $fullW * $xs;
    $remH = $bh - $fullH * $ys;
    $outW = $fullW;
    $outH = $fullH;
    if ($remW > 0) { $outW = $outW + 1; }
    if ($remH > 0) { $outH = $outH + 1; }
    if ($outW < 1 || $outH < 1) { throw "reduce: that factor reduces the box to nothing"; }

    $bands = $d[2];
    $odt = "i64";
    if ($k == "f32") { $odt = "f64"; }
    $out = nd_zeros([$outH, $outW, $bands], $odt);
    $src = $px;
    if ($k == "u8") { $src = nd_astype($px, "i64"); }

    // the full blocks
    if ($fullW > 0 && $fullH > 0) {
        $region = pxbuf.crop($src, $b[0], $b[1], $b[0] + $fullW * $xs, $b[1] + $fullH * $ys);
        pxbuf.copy_into(pxbuf.crop($out, 0, 0, $fullW, $fullH), _block($region, $ys, $xs, $k));
    }
    // the right-hand column of partial blocks
    if ($remW > 0 && $fullH > 0) {
        $region = pxbuf.crop($src, $b[0] + $fullW * $xs, $b[1], $b[2], $b[1] + $fullH * $ys);
        pxbuf.copy_into(pxbuf.crop($out, $fullW, 0, $outW, $fullH),
                        _block($region, $ys, $remW, $k));
    }
    // the bottom row of partial blocks
    if ($remH > 0 && $fullW > 0) {
        $region = pxbuf.crop($src, $b[0], $b[1] + $fullH * $ys, $b[0] + $fullW * $xs, $b[3]);
        pxbuf.copy_into(pxbuf.crop($out, 0, $fullH, $fullW, $outH),
                        _block($region, $remH, $xs, $k));
    }
    // the bottom-right corner
    if ($remW > 0 && $remH > 0) {
        $region = pxbuf.crop($src, $b[0] + $fullW * $xs, $b[1] + $fullH * $ys, $b[2], $b[3]);
        pxbuf.copy_into(pxbuf.crop($out, $fullW, $fullH, $outW, $outH),
                        _block($region, $remH, $remW, $k));
    }
    return $out;
}

// _block -> the right arithmetic for the mode's storage class.
def _block($region, $ys, $xs, $kind) {
    if ($kind == "u8") { return _reduceBlock($region, $ys, $xs); }
    return _reduceBlock32($region, $ys, $xs, $kind == "i32");
}
