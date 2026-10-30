// ════════════════════════════════════════════════════════════════════════
//  pillow/chops.b — ImageChops: twenty-four channel operations, and two
//  rounding regimes that do not agree.
//
//  Chops.c has two macros and the difference between them is the whole
//  content of this module:
//
//      CHOP   computes in int and CLAMPS to [0, 255]
//      CHOP2  computes in int and assigns to a UINT8 -- so it WRAPS mod 256
//
//  `add_modulo` and `subtract_modulo` use the second deliberately; that is
//  their entire purpose. `difference`, `soft_light`, `hard_light` and
//  `overlay` also use it, and for those the wrap is UNREACHABLE: each
//  expression is provably inside [0, 255] for 8-bit inputs. They are
//  implemented with the wrap anyway, and the test states the bound that
//  makes it unobservable -- which is how a reader learns that the wrap is
//  not an accident.
//
//  Three further contracts:
//    * the output size is the MINIMUM of the two inputs in each axis,
//      anchored top left. Not an error, not a union.
//    * logical_and / logical_or / logical_xor require BOTH images to be
//      mode "1".
//    * add and subtract divide by a FLOAT scale and then truncate toward
//      zero before clamping.
//
//  docs/pillow-architecture.md §7.2.
// ════════════════════════════════════════════════════════════════════════

include "./mode.b" as mode;
include "./pxbuf.b" as pxbuf;
include "./image.b" as image;

// _pair($a, $b, $who) -> the two buffers cropped to the overlap, and the
// mode of the result (image1's).
def _pair($a, $b, $who) {
    if (type($a) != "instance" || type($b) != "instance") {
        throw "ImageChops." + $who + ": expected two images";
    }
    if (mode.bands($a.mode) != mode.bands($b.mode)) {
        throw "ImageChops." + $who + ": images must have the same number of bands (" +
              $a.mode + " has " + str(mode.bands($a.mode)) + ", " + $b.mode +
              " has " + str(mode.bands($b.mode)) + ")";
    }
    $w = min($a.width, $b.width);
    $h = min($a.height, $b.height);
    return {"a": pxbuf.crop($a.px, 0, 0, $w, $h),
            "b": pxbuf.crop($b.px, 0, 0, $w, $h),
            "m": $a.mode, "w": $w, "h": $h};
}

def _out($p, $px) {
    return image.wrap($p.m, pxbuf.materialise($px, mode.dtype($p.m)), null, {}, null);
}

// _clamp8($v) / _wrap8($v) -- the two regimes, by name.
def _clamp8($v) { return nd_clip($v, 0, 255); }
def _wrap8($v)  { return nd_mod($v, 256); }

// _trunc0($v) -> truncation toward zero, which C's integer division does
// and nd_floor_divide does not.
def _trunc0($v) {
    return nd_multiply(nd_sign($v), nd_floor_divide(nd_abs($v), 1));
}
def _divTrunc($v, $d) {
    return nd_multiply(nd_sign($v), nd_floor_divide(nd_abs($v), $d));
}

// ─── The one- and zero-image operations ─────────────────────────────────

def constant($im, $value) { return image.new_image("L", $im.size, $value); }
def duplicate($im)        { return $im.copy(); }

def invert($im) {
    _need8($im.mode, "invert");
    return image.wrap($im.mode,
        pxbuf.materialise(nd_subtract(255, nd_clip($im.px, 0, 255)), "i64"),
        null, {}, null);
}

def _need8($m, $who) {
    if (mode.bits_per_band($m) != 8 || mode.dtype($m) != "i64") {
        throw "ImageChops." + $who + ": mode " + str($m) +
              " is not an 8-bit mode; Chops.c computes in UINT8";
    }
}

// ─── The two-image operations ───────────────────────────────────────────

def lighter($a, $b)  { $p = _pair($a, $b, "lighter");  return _out($p, nd_maximum($p.a, $p.b)); }
def darker($a, $b)   { $p = _pair($a, $b, "darker");   return _out($p, nd_minimum($p.a, $p.b)); }

def difference($a, $b) {
    $p = _pair($a, $b, "difference");
    return _out($p, nd_abs(nd_subtract($p.a, $p.b)));
}

def multiply($a, $b) {
    $p = _pair($a, $b, "multiply");
    return _out($p, _clamp8(nd_floor_divide(nd_multiply($p.a, $p.b), 255)));
}

def screen($a, $b) {
    $p = _pair($a, $b, "screen");
    $inv = nd_floor_divide(nd_multiply(nd_subtract(255, $p.a), nd_subtract(255, $p.b)), 255);
    return _out($p, _clamp8(nd_subtract(255, $inv)));
}

// soft_light, from Chops.c (the SVG compositing formula in integers):
//
//   ((255 - a) * (a * b)) / 65536 + (a * (255 - ((255 - a) * (255 - b)) / 255)) / 255
//
// with every division truncating. Provably within [0, 255], so CHOP2's
// wrap is unreachable -- asserted by the test over all 65,536 input pairs.
def soft_light($a, $b) {
    $p = _pair($a, $b, "soft_light");
    $t1 = nd_floor_divide(nd_multiply(nd_subtract(255, $p.a), nd_multiply($p.a, $p.b)), 65536);
    $scr = nd_subtract(255, nd_floor_divide(
        nd_multiply(nd_subtract(255, $p.a), nd_subtract(255, $p.b)), 255));
    $t2 = nd_floor_divide(nd_multiply($p.a, $scr), 255);
    return _out($p, _wrap8(nd_add($t1, $t2)));
}

// hard_light: b decides. b < 128 -> (a * b) / 127, else
// 255 - ((255 - b) * (255 - a)) / 127.
def hard_light($a, $b) {
    $p = _pair($a, $b, "hard_light");
    $lo = nd_floor_divide(nd_multiply($p.a, $p.b), 127);
    $hi = nd_subtract(255, nd_floor_divide(
        nd_multiply(nd_subtract(255, $p.b), nd_subtract(255, $p.a)), 127));
    return _out($p, _wrap8(nd_where(nd_less($p.b, 128), $lo, $hi)));
}

// overlay is hard_light with the roles swapped: a decides.
def overlay($a, $b) {
    $p = _pair($a, $b, "overlay");
    $lo = nd_floor_divide(nd_multiply($p.a, $p.b), 127);
    $hi = nd_subtract(255, nd_floor_divide(
        nd_multiply(nd_subtract(255, $p.a), nd_subtract(255, $p.b)), 127));
    return _out($p, _wrap8(nd_where(nd_less($p.a, 128), $lo, $hi)));
}

// add / subtract: ((a ± b) / scale + offset), the division in FLOAT, then
// truncated toward zero, then clamped.
def add($a, $b, $scale, $offset) {
    $p = _pair($a, $b, "add");
    $sc = $scale;
    if ($sc == null) { $sc = 1.0; }
    $of = $offset;
    if ($of == null) { $of = 0; }
    $v = nd_add(nd_divide(nd_add($p.a, $p.b), $sc), $of);
    return _out($p, _clamp8(_trunc0($v)));
}

def subtract($a, $b, $scale, $offset) {
    $p = _pair($a, $b, "subtract");
    $sc = $scale;
    if ($sc == null) { $sc = 1.0; }
    $of = $offset;
    if ($of == null) { $of = 0; }
    $v = nd_add(nd_divide(nd_subtract($p.a, $p.b), $sc), $of);
    return _out($p, _clamp8(_trunc0($v)));
}

def add_modulo($a, $b) {
    $p = _pair($a, $b, "add_modulo");
    return _out($p, _wrap8(nd_add($p.a, $p.b)));
}

def subtract_modulo($a, $b) {
    $p = _pair($a, $b, "subtract_modulo");
    return _out($p, _wrap8(nd_subtract($p.a, $p.b)));
}

// The logical operations, which require mode "1" on both sides. A mode-1
// buffer holds 0 and 255, so the bitwise and value-wise readings agree.
def _need1($a, $b, $who) {
    if ($a.mode != "1" || $b.mode != "1") {
        throw "ImageChops." + $who + ": both images must have mode \"1\" (got " +
              $a.mode + " and " + $b.mode + "). For other modes, multiply by a " +
              "black-and-white mask instead.";
    }
}

def logical_and($a, $b) {
    _need1($a, $b, "logical_and");
    $p = _pair($a, $b, "logical_and");
    return _out($p, nd_multiply(nd_logical_and($p.a, $p.b), 255));
}
def logical_or($a, $b) {
    _need1($a, $b, "logical_or");
    $p = _pair($a, $b, "logical_or");
    return _out($p, nd_multiply(nd_logical_or($p.a, $p.b), 255));
}
def logical_xor($a, $b) {
    _need1($a, $b, "logical_xor");
    $p = _pair($a, $b, "logical_xor");
    return _out($p, nd_multiply(nd_logical_xor($p.a, $p.b), 255));
}

// blend and composite are Image's, re-exported here because ImageChops
// exports them too.
def blend($a, $b, $alpha)    { return image.blend($a, $b, $alpha); }
def composite($a, $b, $mask) { return image.composite($a, $b, $mask); }

// offset($im, $x, $y) -> the image rolled, wrapping at the edges. With one
// argument the same offset is used for both axes, as Pillow does.
def offset($im, $xoffset, $yoffset) {
    $yo = $yoffset;
    if ($yo == null) { $yo = $xoffset; }
    $w = $im.width;
    $h = $im.height;
    if ($w == 0 || $h == 0) { return $im.copy(); }
    $xi = [];
    $i = 0;
    while ($i < $w) {
        $j = ($i - floor($xoffset)) % $w;
        if ($j < 0) { $j = $j + $w; }
        push($xi, $j);
        $i = $i + 1;
    }
    $yi = [];
    $i = 0;
    while ($i < $h) {
        $j = ($i - floor($yo)) % $h;
        if ($j < 0) { $j = $j + $h; }
        push($yi, $j);
        $i = $i + 1;
    }
    $px = nd_take(nd_take($im.px, nd($xi, "i64"), 1), nd($yi, "i64"), 0);
    return image.wrap($im.mode, pxbuf.materialise($px, mode.dtype($im.mode)),
                      null, {}, null);
}
