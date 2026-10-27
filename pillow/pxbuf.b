// ════════════════════════════════════════════════════════════════════════
//  pillow/pxbuf.b — the pixel buffer: every operation on the (h, w, bands)
//  array, and nothing that knows what an Image is.
//
//  This is the layer that makes the package possible. Rule PW2 says no loop
//  may be proportional to the pixel count, and that is only achievable if
//  band extraction, band assembly, broadcasting and clamping are all single
//  numba calls. They are, and they live here so that convert.b, resample.b,
//  filter.b, chops.b and image.b all get them for free.
//
//  Measured on a 1080p RGB buffer (6,220,800 values):
//      band()        0 ms   -- a view, not a copy
//      splat()       0 ms   -- a broadcast view
//      stack()     ~90 ms   -- one copy per band
//      clamp()     ~70 ms
//
//  Nothing here loops over pixels. The only loops are over BANDS, of which
//  there are at most four.
// ════════════════════════════════════════════════════════════════════════

include "./mode.b" as mode;

// ─── Shape ──────────────────────────────────────────────────────────────

// dims($px) -> [height, width, bands]. Array order, not Pillow's size order.
def dims($px) {
    $s = nd_shape($px);
    if (len($s) != 3) {
        throw "pxbuf: a pixel buffer must be 3-dimensional (h, w, bands), got " + str($s);
    }
    return [$s[0], $s[1], $s[2]];
}

def height($px) { return nd_shape($px)[0]; }
def width($px)  { return nd_shape($px)[1]; }
def nbands($px) { return nd_shape($px)[2]; }

// make($h, $w, $bands, $dtype) -> a zeroed buffer.
def make($h, $w, $bands, $dtype) {
    if ($h < 0 || $w < 0 || $bands < 1) {
        throw "pxbuf.make: bad shape (" + str($h) + ", " + str($w) + ", " + str($bands) + ")";
    }
    return nd_zeros([$h, $w, $bands], $dtype);
}

// ─── Bands ──────────────────────────────────────────────────────────────

// band($px, $i) -> band i as an (h, w, 1) VIEW. Free, and writable: writing
// through it reaches the base buffer, which is how paste and putalpha work.
def band($px, $i) {
    $b = nbands($px);
    if ($i < 0 || $i >= $b) {
        throw "pxbuf.band: band " + str($i) + " of a " + str($b) + "-band buffer";
    }
    return nd_slice($px, [null, null, [$i, $i + 1]]);
}

// bands_from($px, $lo, $hi) -> bands [lo, hi) as an (h, w, hi-lo) view.
def bands_from($px, $lo, $hi) {
    $b = nbands($px);
    if ($lo < 0 || $hi > $b || $lo >= $hi) {
        throw "pxbuf.bands_from: [" + str($lo) + ", " + str($hi) +
              ") of a " + str($b) + "-band buffer";
    }
    return nd_slice($px, [null, null, [$lo, $hi]]);
}

// set_band($px, $i, $src) -> write src (an (h,w,1) or broadcastable buffer,
// or a number) into band i, in place. Returns $px.
//
// A NUMBER takes the zero-then-add route, because an `out=` destination is
// not broadcast to fit: nd_add(scalar, 0, view) has result shape [] and a
// destination of (h, w, 1), which is an error rather than a splat.
def set_band($px, $i, $src) {
    $v = band($px, $i);
    if (type($src) == "number") {
        nd_multiply($v, 0, $v);
        nd_add($v, $src, $v);
        return $px;
    }
    nd_add($src, 0, $v);
    return $px;
}

// stack($list) -> a new (h, w, n) buffer from n buffers of (h, w, 1).
// One copy per band: the loop is over bands, never pixels.
def stack($parts, $dtype) {
    $n = len($parts);
    if ($n < 1) { throw "pxbuf.stack: nothing to stack"; }
    $d0 = dims($parts[0]);
    $out = nd_zeros([$d0[0], $d0[1], $n], $dtype);
    $i = 0;
    while ($i < $n) {
        set_band($out, $i, $parts[$i]);
        $i = $i + 1;
    }
    return $out;
}

// splat($px, $n) -> a one-band buffer seen as n identical bands. A VIEW, so
// it costs nothing; copy it with materialise() if it must be writable.
def splat($px, $n) {
    $d = dims($px);
    if ($d[2] != 1) { throw "pxbuf.splat: needs a one-band buffer, got " + str($d[2]); }
    return nd_broadcast_to($px, [$d[0], $d[1], $n]);
}

// materialise($px, $dtype) -> a contiguous, writable copy.
def materialise($px, $dtype) {
    $d = dims($px);
    $out = nd_zeros([$d[0], $d[1], $d[2]], $dtype);
    nd_add($px, 0, $out);
    return $out;
}

// scalar($a) -> a reduction's result as a plain NUMBER.
//
// nd_min, nd_max, nd_sum and the other reductions return a 0-dimensional
// ARRAY, not a number, so a value from one compares unequal to the integer it
// prints as and str() tacks "(shape=[], dtype=i64)" onto it. Every reduction
// result that leaves this package goes through here.
def scalar($a) {
    if (type($a) == "number") { return $a; }
    return nd_to_list($a);
}

// ─── Range ──────────────────────────────────────────────────────────────

// clamp($px, $m) -> values forced into mode m's range, in a new buffer.
// F has no range and is returned unchanged.
def clamp($px, $m) {
    $spec = mode.clamp_spec($m);
    if ($spec == null) { return $px; }
    return nd_clip($px, $spec.lo, $spec.hi);
}

// clamp_into($px, $m) -> the same, in place.
def clamp_into($px, $m) {
    $spec = mode.clamp_spec($m);
    if ($spec == null) { return $px; }
    nd_clip($px, $spec.lo, $spec.hi, $px);
    return $px;
}

// to_f32($px) -> every value rounded to the nearest float32, as f64.
//
// Pillow's F mode is float32, so RGB->F produces the float32 nearest to
// 0.299R + 0.587G + 0.114B -- 76.24500274658203, not 76.245. Our store is
// f64, so matching Pillow means passing the values through four bytes, which
// nd_tobytes/nd_frombytes do in one call each. Without this, every F-mode
// comparison against Pillow is off in the last few digits.
def to_f32($px) {
    $d = dims($px);
    $flat = nd_frombytes(nd_tobytes($px, "f32le"), "f32le", null);
    return nd_reshape($flat, [$d[0], $d[1], $d[2]]);
}

// ─── Whole-buffer helpers the operators keep needing ────────────────────

// fill($px, $colour) -> every pixel set to `colour`, a list with one entry
// per band (or a single number). In place; returns $px.
//
// Written as "zero it, then add the constant", both with the destination as
// their own `out`. An `out=` destination is NOT broadcast to fit, so handing
// nd_add a 1x1x1 source and a full-size out is an error, not a splat -- which
// is the obvious way to write this and does not work.
def fill($px, $colour) {
    $d = dims($px);
    if (type($colour) == "list") {
        if (len($colour) != $d[2]) {
            throw "pxbuf.fill: " + str(len($colour)) + " values for a " +
                  str($d[2]) + "-band buffer";
        }
        $i = 0;
        while ($i < $d[2]) {
            $v = band($px, $i);
            nd_multiply($v, 0, $v);
            nd_add($v, $colour[$i], $v);
            $i = $i + 1;
        }
        return $px;
    }
    nd_multiply($px, 0, $px);
    nd_add($px, $colour, $px);
    return $px;
}

// crop($px, $l, $t, $r, $b) -> an (h, w, bands) VIEW of the box. Free.
// The box is Pillow's: left, upper, right, lower, with right/lower exclusive.
def crop($px, $l, $t, $r, $b) {
    return nd_slice($px, [[$t, $b], [$l, $r], null]);
}

// copy_into($dst, $src) -> src written over dst, shapes matching. In place.
def copy_into($dst, $src) {
    nd_add($src, 0, $dst);
    return $dst;
}

// pixel($px, $x, $y) -> one pixel as a list of band values.
// This is getpixel, and it is SLOW by design: nd_get costs ~2.5 us, so a
// loop over pixels through here is the thing rule PW2 forbids. Pillow's own
// getpixel is slow for the same reason.
def pixel($px, $x, $y) {
    $d = dims($px);
    if ($x < 0 || $y < 0 || $x >= $d[1] || $y >= $d[0]) {
        throw "pxbuf.pixel: (" + str($x) + ", " + str($y) + ") is outside " +
              str($d[1]) + "x" + str($d[0]);
    }
    $out = [];
    $i = 0;
    while ($i < $d[2]) {
        push($out, nd_get($px, [$y, $x, $i]));
        $i = $i + 1;
    }
    return $out;
}

// put_pixel($px, $x, $y, $vals) -> one pixel written. Equally slow.
def put_pixel($px, $x, $y, $vals) {
    $d = dims($px);
    if ($x < 0 || $y < 0 || $x >= $d[1] || $y >= $d[0]) {
        throw "pxbuf.put_pixel: (" + str($x) + ", " + str($y) + ") is outside " +
              str($d[1]) + "x" + str($d[0]);
    }
    $i = 0;
    while ($i < $d[2]) {
        nd_set($px, [$y, $x, $i], $vals[$i]);
        $i = $i + 1;
    }
    return $px;
}

// lut($px, $table) -> every value replaced by table[value]. One gather, so a
// 1080p image costs ~90 ms whatever the table does. This is point().
def lut($px, $table) {
    $d = dims($px);
    $t = $table;
    if (type($table) == "list") { $t = nd($table, "i64"); }
    $g = nd_take($t, nd_ravel($px));
    return nd_reshape($g, [$d[0], $d[1], $d[2]]);
}
