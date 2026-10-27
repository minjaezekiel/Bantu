// ════════════════════════════════════════════════════════════════════════
//  pillow/convert.b — the mode-conversion matrix.
//
//  Operates on PIXEL BUFFERS, not on Image objects, so image.b can include
//  this without this including image.b. That layering is deliberate: the
//  array layer (mode, pxbuf, convert, resample, filter) knows nothing about
//  the Image object, and the Image object orchestrates it.
//
//      include "./convert.b" as convert;
//      $r = convert.px($px, "RGB", "L", null, null);
//      // -> {"px": <buffer>, "mode": "L", "palette": null}
//
//  FIDELITY. Every formula here is verified against Python's Pillow, and the
//  verification is not a spot check:
//
//    RGB -> L    (R*19595 + G*38470 + B*7471 + 32768) >> 16
//                checked over ALL 16,777,216 RGB triples: 0 mismatches.
//                This is Pillow's ImagingConvert L24 macro. It is NOT the
//                (R*299 + G*587 + B*114)/1000 that references print -- that
//                truncating form disagrees with Pillow on half of all inputs.
//    RGB -> I    the same L24 value, widened
//    RGB -> F    float32(0.299R + 0.587G + 0.114B), as float32, which is why
//                pxbuf.to_f32 exists
//    RGB -> CMYK C=255-R, M=255-G, Y=255-B, K=0. Pillow's naive conversion:
//                K is ALWAYS zero. Verified over 20,010 random triples.
//
//  WHAT IS NOT YET BIT-EXACT, stated rather than quietly approximated:
//  YCbCr, HSV and LAB. Pillow's integer forms for those three were measured
//  to differ from every candidate formula tried by +/-1 on 8-25% of inputs
//  (the fitting evidence is in pillow-suite/ROADMAP.md, phase P2). They are
//  implemented here with the standard formulations and are therefore CLOSE
//  but not identical to Pillow; convert.unverified($from, $to) says so, the
//  differential test skips them loudly rather than silently, and finishing
//  them is the first item of the next phase.
// ════════════════════════════════════════════════════════════════════════

include "./mode.b" as mode;
include "./pxbuf.b" as pxbuf;
include "./ycbcr_tables.b" as ycc;
// Aliased `pal`, not `palette`: Bantu resolves a module alias and a local
// `$palette` to the same name, so a function with a $palette PARAMETER --
// which px() and _toRGB() both have -- would shadow the module and every
// call through it would fail with "it holds null, not a function".
include "./palette.b" as pal;

// LAB is the one family whose arithmetic cannot be matched to Pillow here,
// and the reason is structural rather than unfinished work: Pillow does not
// convert RGB<->LAB with a formula at all. Image.convert routes it through
// ImageCms, i.e. littleCMS, a full ICC colour-management engine -- there is
// no {"RGB","LAB"} row in Convert.c's converter table to copy. Reproducing
// littleCMS bit for bit is exactly the library-sized problem decision PW10
// declines on ImageCms's behalf, and faking it would silently produce wrong
// colour, which is the worst failure available.
//
// So pillow implements a correct, standard sRGB <-> CIE L*a*b* conversion
// (D65, the piecewise sRGB transfer function, a/b offset by +128). It is
// good colour science and it is NOT littleCMS to the last bit.
$_UNVERIFIED_FAMILIES = ["lab"];

// unverified($from, $to) -> does this pair involve arithmetic that is not
// bit-identical to Pillow? Ask before relying on exact equality.
def unverified($from, $to) {
    $ff = mode.family($from);
    $tf = mode.family($to);
    return contains($_UNVERIFIED_FAMILIES, $ff) || contains($_UNVERIFIED_FAMILIES, $tf);
}

// why_unverified($from, $to) -> a sentence a caller can print, or null.
def why_unverified($from, $to) {
    if (!unverified($from, $to)) { return null; }
    return "Pillow converts RGB<->LAB through ImageCms (littleCMS), not a formula, " +
           "so pillow's standard D65 sRGB<->L*a*b* conversion is correct colour " +
           "science but not bit-identical to Pillow. See DECISIONS.md PW10.";
}

// ─── The greyscale reduction every conversion funnels through ───────────

// _l24($rgb) -> an (h, w, 1) buffer of Pillow's L24 luminance.
def _l24($rgb) {
    $r = pxbuf.band($rgb, 0);
    $g = pxbuf.band($rgb, 1);
    $b = pxbuf.band($rgb, 2);
    $acc = nd_multiply($r, 19595);
    $acc = nd_add($acc, nd_multiply($g, 38470));
    $acc = nd_add($acc, nd_multiply($b, 7471));
    $acc = nd_add($acc, 32768);
    return nd_floor_divide($acc, 65536);
}

// _lfloat($rgb) -> the float luminance, as float32, for mode F.
def _lfloat($rgb) {
    $r = pxbuf.band($rgb, 0);
    $g = pxbuf.band($rgb, 1);
    $b = pxbuf.band($rgb, 2);
    $acc = nd_multiply(nd_astype($r, "f64"), 0.299);
    $acc = nd_add($acc, nd_multiply(nd_astype($g, "f64"), 0.587));
    $acc = nd_add($acc, nd_multiply(nd_astype($b, "f64"), 0.114));
    return pxbuf.to_f32($acc);
}

// ─── Normalising any mode to RGB, and back out ──────────────────────────
//
// The matrix is built as "in to RGB, then RGB to out", with direct paths for
// the pairs where that would lose information (grey to grey, alpha moves).
// That is how Pillow's own ImagingConvert is organised, and it is why adding
// a mode is a small change rather than an N-squared one.

// _toRGB($px, $m, $palette) -> an (h, w, 3) buffer of RGB.
def _toRGB($px, $m, $palette) {
    $fam = mode.family($m);

    if ($fam == "rgb") { return pxbuf.bands_from($px, 0, 3); }

    if ($fam == "bilevel" || $fam == "grey") {
        return pxbuf.splat(pxbuf.band($px, 0), 3);
    }

    if ($fam == "int") {
        return pxbuf.splat(nd_clip(pxbuf.band($px, 0), 0, 255), 3);
    }

    if ($fam == "float") {
        $v = nd_clip(pxbuf.band($px, 0), 0, 255);
        return pxbuf.splat(nd_astype(nd_floor($v), "i64"), 3);
    }

    if ($fam == "cmyk") {
        // Pillow's naive inverse: R = 255 - C, and K is ignored, exactly as
        // its forward conversion always writes K = 0.
        $c = nd_subtract(255, pxbuf.band($px, 0));
        $m2 = nd_subtract(255, pxbuf.band($px, 1));
        $y = nd_subtract(255, pxbuf.band($px, 2));
        return pxbuf.stack([$c, $m2, $y], "i64");
    }

    if ($fam == "palette") {
        if ($palette == null) {
            throw "convert: a " + $m + " buffer needs its palette to become RGB";
        }
        return _paletteGather(pxbuf.band($px, 0), $palette);
    }

    if ($fam == "ycbcr") { return _ycbcrToRGB($px); }
    if ($fam == "hsv")   { return _hsvToRGB($px); }
    if ($fam == "lab")   { return _labToRGB($px); }

    throw "convert: no path from " + $m + " to RGB";
}

// _paletteGather($idx, $palette) -> (h, w, 3) from a 256x3 palette.
// One nd_take, so a 1080p palette image expands in about 90 ms.
def _paletteGather($idx, $palette) {
    $d = pxbuf.dims($idx);
    $pal = $palette;
    if (type($palette) == "list") { $pal = nd($palette, "i64"); }
    $safe = nd_clip($idx, 0, nd_shape($pal)[0] - 1);
    $g = nd_take($pal, nd_ravel($safe), 0);
    return nd_reshape($g, [$d[0], $d[1], 3]);
}

// ─── YCbCr — Pillow's own tables, so this is exact ─────────────────────
//
// ConvertYCbCr.c sums THREE PRE-ROUNDED products at 6 fractional bits:
//
//     y  =  (Y_R[r]  + Y_G[g]  + Y_B[b])  >> 6
//     cb = ((Cb_R[r] + Cb_G[g] + Cb_B[b]) >> 6) + 128
//     cr = ((Cr_R[r] + Cr_G[g] + Cr_B[b]) >> 6) + 128
//
// Each table entry is round(coeff * value * 64), so the rounding happens per
// channel BEFORE the sum and no combined fixed-point multiply can reproduce
// it -- see ycbcr_tables.b's header for the measurement that proves it.
//
// Three nd_take gathers and two adds per output band. No pixel loop, no
// floating point, and the `>> 6` is nd_floor_divide because a C arithmetic
// shift floors (these tables hold negatives).
//
// Pillow casts the forward result with (UINT8) and does not clip. For every
// real RGB input the sums already land in 0..255, so clipping and casting
// agree everywhere it matters -- but a hand-made out-of-range buffer would
// WRAP under the cast, turning 256 into 0, which is a silent corruption that
// propagates into whatever reads the image next. This CLAMPS instead:
// identical on all valid input, and a saturating failure rather than a
// wrapping one on invalid input.
def _gather($table, $band) {
    $d = pxbuf.dims($band);
    $g = nd_take($table, nd_ravel($band));
    return nd_reshape($g, [$d[0], $d[1], 1]);
}

def _rgbToYCbCr($rgb) {
    $r = pxbuf.band($rgb, 0);
    $g = pxbuf.band($rgb, 1);
    $b = pxbuf.band($rgb, 2);

    $y = nd_floor_divide(nd_add(nd_add(_gather(ycc.Y_R, $r), _gather(ycc.Y_G, $g)),
                                _gather(ycc.Y_B, $b)), 64);
    $cb = nd_add(nd_floor_divide(nd_add(nd_add(_gather(ycc.Cb_R, $r), _gather(ycc.Cb_G, $g)),
                                        _gather(ycc.Cb_B, $b)), 64), 128);
    $cr = nd_add(nd_floor_divide(nd_add(nd_add(_gather(ycc.Cr_R, $r), _gather(ycc.Cr_G, $g)),
                                        _gather(ycc.Cr_B, $b)), 64), 128);
    return pxbuf.stack([nd_clip($y, 0, 255), nd_clip($cb, 0, 255),
                        nd_clip($cr, 0, 255)], "i64");
}

// And the inverse, which DOES clip, as ImagingConvertYCbCr2RGB does.
def _ycbcrToRGB($px) {
    $y  = pxbuf.band($px, 0);
    $cb = pxbuf.band($px, 1);
    $cr = pxbuf.band($px, 2);
    $r = nd_add($y, nd_floor_divide(_gather(ycc.R_Cr, $cr), 64));
    $g = nd_add($y, nd_floor_divide(nd_add(_gather(ycc.G_Cb, $cb),
                                           _gather(ycc.G_Cr, $cr)), 64));
    $b = nd_add($y, nd_floor_divide(_gather(ycc.B_Cb, $cb), 64));
    return pxbuf.stack([nd_clip($r, 0, 255), nd_clip($g, 0, 255), nd_clip($b, 0, 255)], "i64");
}

// ─── HSV — Pillow's exact float32 sequence ─────────────────────────────
//
// Convert.c's rgb2hsv_row follows colorsys.py, but in C `float` -- 32-bit --
// and that is the whole difference between close and exact. Two details:
//
//   * s, rc, gc and bc are each rounded to float32 as they are assigned, so
//     the intermediate precision is 24 bits, not 53. pxbuf.to_f32 is applied
//     at exactly the points C's assignments round.
//   * the hue is folded as `fmod(h / 6.0 + 1.0, 1.0)` -- the "+1.0 then
//     fmod" form, which is NOT the same as "fmod then add 1 if negative":
//     adding 1 first costs low-order bits, and Pillow's comment says why the
//     form was chosen ("incorrect hue happens if h/6 is negative").
//
// Both quantisations truncate: (int)(h * 255.0), (int)(s * 255.0).
def _rgbToHSV($rgb) {
    $r = nd_astype(pxbuf.band($rgb, 0), "f64");
    $g = nd_astype(pxbuf.band($rgb, 1), "f64");
    $b = nd_astype(pxbuf.band($rgb, 2), "f64");

    $maxc = nd_maximum(nd_maximum($r, $g), $b);
    $minc = nd_minimum(nd_minimum($r, $g), $b);
    $isFlat = nd_less_equal(nd_subtract($maxc, $minc), 0);

    // cr = (float)(maxc - minc); guarded so a flat pixel does not divide by 0
    $cr = nd_maximum(nd_subtract($maxc, $minc), 1.0);
    $mx = nd_maximum($maxc, 1.0);

    // s = cr / maxc, rounded to float32 as C assigns it
    $s = pxbuf.to_f32(nd_divide($cr, $mx));
    $rc = pxbuf.to_f32(nd_divide(nd_subtract($maxc, $r), $cr));
    $gc = pxbuf.to_f32(nd_divide(nd_subtract($maxc, $g), $cr));
    $bc = pxbuf.to_f32(nd_divide(nd_subtract($maxc, $b), $cr));

    $rIsMax = nd_greater_equal($r, $maxc);
    $gIsMax = nd_logical_and(nd_greater_equal($g, $maxc), nd_logical_not($rIsMax));
    $h = nd_where($rIsMax, nd_subtract($bc, $gc),
         nd_where($gIsMax, nd_subtract(nd_add(2.0, $rc), $bc),
                           nd_subtract(nd_add(4.0, $gc), $rc)));
    $h = pxbuf.to_f32($h);

    // h = fmod(h / 6.0 + 1.0, 1.0), in double, then rounded to float32
    $h = pxbuf.to_f32(nd_mod(nd_add(nd_divide($h, 6.0), 1.0), 1.0));

    $uh = nd_clip(nd_floor(nd_multiply($h, 255.0)), 0, 255);
    $us = nd_clip(nd_floor(nd_multiply($s, 255.0)), 0, 255);

    $uh = nd_where($isFlat, 0, $uh);
    $us = nd_where($isFlat, 0, $us);

    return pxbuf.stack([nd_astype($uh, "i64"), nd_astype($us, "i64"),
                        nd_astype($maxc, "i64")], "i64");
}

// hsv2rgb, also from Convert.c. `f` and `fs` are C floats, so both are
// rounded to float32; p, q and t use round() -- nearest, ties away from zero
// -- where the forward direction truncates.
def _hsvToRGB($px) {
    $h = nd_astype(pxbuf.band($px, 0), "f64");
    $s = nd_astype(pxbuf.band($px, 1), "f64");
    $v = nd_astype(pxbuf.band($px, 2), "f64");

    $h6 = nd_divide(nd_multiply($h, 6.0), 255.0);
    $i = nd_floor($h6);
    $f = pxbuf.to_f32(nd_subtract($h6, $i));
    $fs = pxbuf.to_f32(nd_divide($s, 255.0));

    $p = nd_floor(nd_add(nd_multiply($v, nd_subtract(1.0, $fs)), 0.5));
    $q = nd_floor(nd_add(nd_multiply($v, nd_subtract(1.0, nd_multiply($fs, $f))), 0.5));
    $t = nd_floor(nd_add(nd_multiply($v, nd_subtract(1.0, nd_multiply($fs,
                         nd_subtract(1.0, $f)))), 0.5));
    $p = nd_clip($p, 0, 255);
    $q = nd_clip($q, 0, 255);
    $t = nd_clip($t, 0, 255);

    $k = nd_mod($i, 6);
    $r = nd_where(nd_equal($k, 0), $v,
         nd_where(nd_equal($k, 1), $q,
         nd_where(nd_equal($k, 2), $p,
         nd_where(nd_equal($k, 3), $p,
         nd_where(nd_equal($k, 4), $t, $v)))));
    $g = nd_where(nd_equal($k, 0), $t,
         nd_where(nd_equal($k, 1), $v,
         nd_where(nd_equal($k, 2), $v,
         nd_where(nd_equal($k, 3), $q,
         nd_where(nd_equal($k, 4), $p, $p)))));
    $b = nd_where(nd_equal($k, 0), $p,
         nd_where(nd_equal($k, 1), $p,
         nd_where(nd_equal($k, 2), $t,
         nd_where(nd_equal($k, 3), $v,
         nd_where(nd_equal($k, 4), $v, $q)))));

    // s == 0 is a separate branch in C, before any of the above runs.
    $sIsZero = nd_less_equal($s, 0);
    $r = nd_where($sIsZero, $v, $r);
    $g = nd_where($sIsZero, $v, $g);
    $b = nd_where($sIsZero, $v, $b);

    return pxbuf.stack([nd_astype($r, "i64"), nd_astype($g, "i64"),
                        nd_astype($b, "i64")], "i64");
}

// ─── LAB — standard D65 sRGB <-> L*a*b*, NOT littleCMS ────────────────
// D65 white point, the sRGB transfer function, and Pillow's 0..255 packing
// with a and b offset by +128.
$_D65 = [0.95047, 1.0, 1.08883];

def _srgbToLinear($c) {
    // c is 0..1. The piecewise sRGB EOTF.
    $lo = nd_divide($c, 12.92);
    $hi = nd_power(nd_divide(nd_add($c, 0.055), 1.055), 2.4);
    return nd_where(nd_less_equal($c, 0.04045), $lo, $hi);
}

def _labF($t) {
    $lo = nd_add(nd_multiply($t, 7.787037037037035), 0.13793103448275862);
    $hi = nd_cbrt($t);
    return nd_where(nd_greater($t, 0.008856451679035631), $hi, $lo);
}

def _rgbToLAB($rgb) {
    $r = _srgbToLinear(nd_divide(nd_astype(pxbuf.band($rgb, 0), "f64"), 255.0));
    $g = _srgbToLinear(nd_divide(nd_astype(pxbuf.band($rgb, 1), "f64"), 255.0));
    $b = _srgbToLinear(nd_divide(nd_astype(pxbuf.band($rgb, 2), "f64"), 255.0));
    $X = nd_add(nd_add(nd_multiply($r, 0.4124), nd_multiply($g, 0.3576)), nd_multiply($b, 0.1805));
    $Y = nd_add(nd_add(nd_multiply($r, 0.2126), nd_multiply($g, 0.7152)), nd_multiply($b, 0.0722));
    $Z = nd_add(nd_add(nd_multiply($r, 0.0193), nd_multiply($g, 0.1192)), nd_multiply($b, 0.9505));
    $fx = _labF(nd_divide($X, $_D65[0]));
    $fy = _labF(nd_divide($Y, $_D65[1]));
    $fz = _labF(nd_divide($Z, $_D65[2]));
    $L = nd_subtract(nd_multiply($fy, 116.0), 16.0);
    $A = nd_multiply(nd_subtract($fx, $fy), 500.0);
    $B = nd_multiply(nd_subtract($fy, $fz), 200.0);
    // Pillow packs L as 0..255 (L* 0..100 scaled by 255/100) and a/b offset.
    $Lp = nd_floor(nd_add(nd_multiply($L, 2.55), 0.5));
    $Ap = nd_floor(nd_add(nd_add($A, 128.0), 0.5));
    $Bp = nd_floor(nd_add(nd_add($B, 128.0), 0.5));
    return pxbuf.stack([nd_astype(nd_clip($Lp, 0, 255), "i64"),
                        nd_astype(nd_clip($Ap, 0, 255), "i64"),
                        nd_astype(nd_clip($Bp, 0, 255), "i64")], "i64");
}

def _labToRGB($px) {
    $L = nd_divide(nd_astype(pxbuf.band($px, 0), "f64"), 2.55);
    $A = nd_subtract(nd_astype(pxbuf.band($px, 1), "f64"), 128.0);
    $B = nd_subtract(nd_astype(pxbuf.band($px, 2), "f64"), 128.0);
    $fy = nd_divide(nd_add($L, 16.0), 116.0);
    $fx = nd_add($fy, nd_divide($A, 500.0));
    $fz = nd_subtract($fy, nd_divide($B, 200.0));
    $X = nd_multiply(_labFinv($fx), $_D65[0]);
    $Y = nd_multiply(_labFinv($fy), $_D65[1]);
    $Z = nd_multiply(_labFinv($fz), $_D65[2]);
    $r = nd_add(nd_add(nd_multiply($X, 3.2406), nd_multiply($Y, -1.5372)), nd_multiply($Z, -0.4986));
    $g = nd_add(nd_add(nd_multiply($X, -0.9689), nd_multiply($Y, 1.8758)), nd_multiply($Z, 0.0415));
    $b = nd_add(nd_add(nd_multiply($X, 0.0557), nd_multiply($Y, -0.2040)), nd_multiply($Z, 1.0570));
    return pxbuf.stack([_linearToSrgb8($r), _linearToSrgb8($g), _linearToSrgb8($b)], "i64");
}

def _labFinv($f) {
    $cube = nd_multiply(nd_multiply($f, $f), $f);
    $lin = nd_divide(nd_subtract($f, 0.13793103448275862), 7.787037037037035);
    return nd_where(nd_greater($cube, 0.008856451679035631), $cube, $lin);
}

def _linearToSrgb8($c) {
    $x = nd_clip($c, 0.0, 1.0);
    $lo = nd_multiply($x, 12.92);
    $hi = nd_subtract(nd_multiply(nd_power($x, 0.4166666666666667), 1.055), 0.055);
    $s = nd_where(nd_less_equal($x, 0.0031308), $lo, $hi);
    return nd_astype(nd_clip(nd_floor(nd_add(nd_multiply($s, 255.0), 0.5)), 0, 255), "i64");
}

// ─── Dithering ──────────────────────────────────────────────────────────
//
// Floyd-Steinberg, and it is the one genuinely sequential operation in the
// whole package: a pixel's error depends on its left neighbour's. So it runs
// PER SCANLINE -- the loop is over rows and, inside a row, over columns of a
// 1-D array. A 1080p dither is 1,080 x 1,920 iterations, about two seconds,
// which is why Pillow makes it opt-in too and why the docs publish the cost.
//
// Pillow's convert("1") and convert("P") DEFAULT to this, which is easy to
// get wrong: a plain threshold looks right on a gradient and is wrong on
// every photograph.
// _cdiv($a, $b) -> C integer division: truncates TOWARD ZERO, where
// nd_floor_divide and Bantu's own floor() round down. The dither divides a
// signed accumulator, so the two differ on every negative error.
def _cdiv($a, $b) {
    $q = $a / $b;
    if ($q < 0) { return 0 - floor(0 - $q); }
    return floor($q);
}

// ─── Dithering — Convert.c's tobilevel, exactly ────────────────────────
//
// Floyd-Steinberg, and it is Pillow's DEFAULT for mode "1" (not opt-in).
// It is the one genuinely sequential operation in the package: a pixel's
// value depends on its left neighbour's error, so it runs per scanline in
// Bantu lists, where a read is ~0.6 us against nd_get's ~2.5 us.
//
// THE STRUCTURE IS NOT THE TEXTBOOK ONE, and the difference is the whole
// reason an earlier draft disagreed with Pillow on 8-12% of pixels. Pillow
// carries the error UNDIVIDED and divides by 16 exactly once, at the point
// of use:
//
//     l = CLIP8(in[x] + (l + errors[x+1]) / 16);
//     out[x] = (l > 128) ? 255 : 0;
//     l -= out[x];                 // e, the residual
//     l2 = l;  d2 = l + l;         // d2 = 2e
//     l += d2;                     // l = 3e
//     errors[x] = l + l0;          // 3e + (5e_prev + e_prevprev)
//     l += d2;                     // l = 5e
//     l0 = l + l1;
//     l1 = l2;
//     l += d2;                     // l = 7e, carried right
//
// Dividing each of the four weights separately and truncating each -- the
// obvious reading of "7/16, 3/16, 5/16, 1/16" -- throws away four rounding
// residues per pixel instead of one, and the error compounds along the row.
//
// `errors` is allocated ONCE for the whole image and persists across rows;
// `l`, `l0` and `l1` reset to 0 at the start of each row. Both matter.
//
// The threshold is strictly `> 128`, where dither=NONE uses `>= 128`.
//
// $lum is an (h, w, 1) buffer of 0..255 values. NOT named $grey: Bantu keeps
// variables and functions in one namespace, so a parameter called $grey would
// replace any function of that name for the rest of the call -- which is how
// palette.grey() came to fail with "it holds a value that is not callable".
def dither_to_bilevel($lum) {
    $d = pxbuf.dims($lum);
    $h = $d[0];
    $w = $d[1];
    $out = nd_zeros([$h, $w, 1], "i64");
    if ($h == 0 || $w == 0) { return $out; }

    // errors has w+1 entries and survives from row to row, as calloc'd in C.
    $errors = [];
    $i = 0;
    while ($i <= $w) { push($errors, 0); $i = $i + 1; }

    $y = 0;
    while ($y < $h) {
        $row = nd_to_list(nd_reshape(nd_slice($lum, [[$y, $y + 1], null, null]), [$w]));
        $outRow = [];
        $l = 0;
        $l0 = 0;
        $l1 = 0;
        $x = 0;
        while ($x < $w) {
            // _cdiv inlined: a call is ~1 us and this is the hot loop.
            $t = $l + $errors[$x + 1];
            if ($t < 0) { $l = $row[$x] - floor((0 - $t) / 16); }
            else { $l = $row[$x] + floor($t / 16); }
            if ($l < 0) { $l = 0; }
            if ($l > 255) { $l = 255; }
            if ($l > 128) { $px = 255; } else { $px = 0; }
            push($outRow, $px);

            $l = $l - $px;
            $l2 = $l;
            $d2 = $l + $l;
            $l = $l + $d2;
            $errors[$x] = $l + $l0;
            $l = $l + $d2;
            $l0 = $l + $l1;
            $l1 = $l2;
            $l = $l + $d2;
            $x = $x + 1;
        }
        $errors[$w] = $l0;
        nd_add(nd_reshape(nd($outRow, "i64"), [1, $w, 1]), 0,
               nd_slice($out, [[$y, $y + 1], null, null]));
        $y = $y + 1;
    }
    return $out;
}

// _l1000($rgb) -> the TRUNCATED luminance, (R*299 + G*587 + B*114) / 1000.
//
// Pillow's dither path uses this and NOT the L24 macro that convert("L")
// uses, so convert("1") and convert("L") disagree about grey by design:
// tobilevel computes `L(in) / 1000` where L() is the 299/587/114 sum. Found
// by reading Convert.c; no amount of black-box fitting would separate the
// two, because both are "about the same" luminance.
def _l1000($rgb) {
    $r = pxbuf.band($rgb, 0);
    $g = pxbuf.band($rgb, 1);
    $b = pxbuf.band($rgb, 2);
    $acc = nd_add(nd_add(nd_multiply($r, 299), nd_multiply($g, 587)),
                  nd_multiply($b, 114));
    return nd_floor_divide($acc, 1000);
}

// ─── The public entry point ─────────────────────────────────────────────
//
// px($buf, $from, $to, $palette, $dither) -> {px, mode, palette}
//
// `dither` is "FLOYDSTEINBERG" (the default, matching Pillow) or "NONE".
def px($buf, $from, $to, $palette, $dither) {
    if (!mode.is_valid($from)) { throw "convert: unknown source mode " + str($from); }
    if (!mode.is_valid($to))   { throw "convert: unknown target mode " + str($to); }

    // Same mode: hand back the buffer. Pillow returns a copy; the Image layer
    // decides whether to copy, because it knows whether the caller shares it.
    if ($from == $to) { return {"px": $buf, "mode": $to, "palette": $palette}; }

    $ff = mode.family($from);
    $tf = mode.family($to);
    $dith = $dither;
    if ($dith == null) { $dith = "FLOYDSTEINBERG"; }

    // ── direct grey-to-grey paths, which must not detour through RGB ──
    $greyIn = ($ff == "grey" || $ff == "bilevel" || $ff == "int" || $ff == "float");
    if ($greyIn && ($tf == "grey" || $tf == "bilevel" || $tf == "int" || $tf == "float")) {
        $v = pxbuf.band($buf, 0);
        $alpha = _alphaOf($buf, $from);

        if ($tf == "float") {
            $out = pxbuf.to_f32(nd_astype($v, "f64"));
            return _withAlpha($out, $alpha, $to, "f64");
        }
        if ($ff == "float") { $v = nd_astype(nd_floor(nd_clip($v, -2147483648, 2147483647)), "i64"); }
        if ($tf == "bilevel") {
            if ($dith == "NONE") {
                $out = nd_multiply(nd_astype(nd_greater_equal($v, 128), "i64"), 255);
            } else {
                $out = dither_to_bilevel(nd_clip($v, 0, 255));
            }
            return _withAlpha($out, $alpha, $to, "i64");
        }
        $out = pxbuf.clamp($v, $to);
        return _withAlpha($out, $alpha, $to, "i64");
    }

    // ── grey -> P / PA copies the index; there is no colour search ──
    //
    // Convert.c's topalette gives a one-band source the 256-entry grey ramp
    // and memcpy's the data. So L 137 becomes index 137, not "the nearest
    // entry of a colour cube to (137,137,137)" -- routing grey through the
    // cube would quantise it to 26 levels and look obviously wrong.
    if ($greyIn && $tf == "palette") {
        $v = pxbuf.band($buf, 0);
        if ($ff == "float") { $v = nd_astype(nd_floor($v), "i64"); }
        $r = pal.grey_to_index($v);
        $res = _withAlpha($r.idx, _alphaOf($buf, $from), $to, "i64");
        $res.palette = $r.palette;
        return $res;
    }

    // ── grey <-> YCbCr does NOT go through RGB either ──
    //
    // Convert.c's l2ycbcr writes Y = L, Cb = Cr = 128, and ycbcr2l reads Y
    // straight back: in JFIF, Y IS the luminance, so round-tripping grey
    // through the matrix would be both slower and lossy. Pillow has direct
    // rows for L/LA -> YCbCr and YCbCr -> L/LA, and so does this.
    // Convert.c has rows for L and LA only. For 1, I and F Pillow converts to
    // basemode L first and then here, which lands on the same values, so one
    // path serves both -- but the clamp and the float truncation have to
    // happen, because that is what the L step would have done.
    if ($greyIn && $tf == "ycbcr") {
        $v = pxbuf.band($buf, 0);
        if ($ff == "float") { $v = nd_astype(nd_floor($v), "i64"); }
        $v = nd_clip($v, 0, 255);
        $mid = nd_zeros(nd_shape($v), "i64");
        nd_add($mid, 128, $mid);
        return {"px": pxbuf.stack([$v, $mid, $mid], "i64"), "mode": $to, "palette": null};
    }
    // Only L and LA, because those are the only rows Convert.c has. For any
    // other target Pillow's Python falls back to "convert to basemode, then
    // convert again" -- YCbCr -> I really is YCbCr -> RGB -> I, and taking
    // the Y shortcut there would be a different answer. Restricting this to
    // the rows that exist is what keeps the two in step.
    if ($ff == "ycbcr" && ($to == "L" || $to == "LA")) {
        $y = pxbuf.band($buf, 0);
        return _withAlpha(nd_clip($y, 0, 255), null, $to, "i64");
    }

    // ── grey -> CMYK does NOT go through RGB ──
    //
    // Pillow sends grey straight to the black plate: L 76 becomes
    // (0, 0, 0, 179), not (179, 179, 179, 0). Routing it through RGB -- which
    // is what the generic path below would do -- gets all four bands wrong.
    // Verified against Pillow for L, 1, I and F sources.
    if ($greyIn && $tf == "cmyk") {
        $v = pxbuf.band($buf, 0);
        if ($ff == "float") { $v = nd_astype(nd_floor($v), "i64"); }
        $v = nd_clip($v, 0, 255);
        $zero = nd_zeros(nd_shape($v), "i64");
        $k = nd_subtract(255, $v);
        return {"px": pxbuf.stack([$zero, $zero, $zero, $k], "i64"),
                "mode": $to, "palette": null};
    }

    // ── everything else goes through RGB ──
    $alpha = _alphaOf($buf, $from);
    $rgb = _toRGB($buf, $from, $palette);

    if ($tf == "rgb") {
        $out = pxbuf.materialise($rgb, "i64");
        return _withAlpha($out, $alpha, $to, "i64");
    }
    if ($tf == "grey" || $tf == "int") {
        $out = pxbuf.clamp(_l24($rgb), $to);
        return _withAlpha($out, $alpha, $to, "i64");
    }
    if ($tf == "float") {
        return _withAlpha(_lfloat($rgb), $alpha, $to, "f64");
    }
    if ($tf == "bilevel") {
        if ($dith == "NONE") {
            // The threshold path goes through L24, like convert("L")...
            $out = nd_multiply(nd_astype(nd_greater_equal(_l24($rgb), 128), "i64"), 255);
        } else {
            // ...but the DITHER path uses the truncated /1000 luminance,
            // because Convert.c's tobilevel does. The two disagree, in
            // Pillow, by design.
            $out = dither_to_bilevel(_l1000($rgb));
        }
        return _withAlpha($out, $alpha, $to, "i64");
    }
    if ($tf == "cmyk") {
        $c = nd_subtract(255, pxbuf.band($rgb, 0));
        $m2 = nd_subtract(255, pxbuf.band($rgb, 1));
        $y = nd_subtract(255, pxbuf.band($rgb, 2));
        $k = nd_zeros(nd_shape($c), "i64");
        return {"px": pxbuf.stack([$c, $m2, $y, $k], "i64"), "mode": $to, "palette": null};
    }
    if ($tf == "ycbcr") {
        return {"px": _rgbToYCbCr($rgb), "mode": $to, "palette": null};
    }
    if ($tf == "hsv") {
        return {"px": _rgbToHSV($rgb), "mode": $to, "palette": null};
    }
    if ($tf == "lab") {
        return {"px": _rgbToLAB($rgb), "mode": $to, "palette": null};
    }
    if ($tf == "palette") {
        // Pillow sends an RGBA source to quantize() -- an ADAPTIVE
        // median-cut palette -- not to the fixed colour cube. Returning cube
        // indices here would look almost right and be wrong, so it names the
        // missing feature instead.
        if (mode.has_alpha($from)) {
            throw "convert: " + $from + " -> " + $to + " uses an adaptive palette " +
                  "(Pillow routes it through quantize()), which is not implemented " +
                  "yet. Convert to RGB first for the fixed web palette.";
        }
        $r = pal.rgb_to_index($rgb, null, $dith);
        $res = _withAlpha($r.idx, $alpha, $to, "i64");
        $res.palette = $r.palette;
        return $res;
    }
    throw "convert: no path from " + $from + " to " + $to;
}

// _alphaOf($buf, $m) -> the alpha band as (h,w,1), or null when there is none.
// RGBX's fourth band is padding, not alpha, so it does not come back here.
def _alphaOf($buf, $m) {
    $ai = mode.alpha_band($m);
    if ($ai < 0) { return null; }
    return pxbuf.band($buf, $ai);
}

// _withAlpha($colour, $alpha, $to, $dtype) -> assemble the target buffer,
// adding, dropping or premultiplying alpha as the target mode requires.
def _withAlpha($colour, $alpha, $to, $dtype) {
    $want = mode.alpha_band($to);
    $nb = mode.bands($to);
    $cb = pxbuf.nbands($colour);

    // No alpha in the target: pad to the band count (RGBX) or hand back.
    if ($want < 0) {
        if ($cb == $nb) { return {"px": pxbuf.materialise($colour, $dtype), "mode": $to, "palette": null}; }
        if ($cb < $nb) {
            $parts = [];
            $i = 0;
            while ($i < $cb) { push($parts, pxbuf.band($colour, $i)); $i = $i + 1; }
            // The padding band of RGBX is 255, not 0. Checked against Pillow:
            // RGB (0,0,0) -> RGBX (0,0,0,255). Filling it with zero is the
            // obvious guess and it is wrong on every pixel.
            while (len($parts) < $nb) {
                $pad = nd_zeros(nd_shape(pxbuf.band($colour, 0)), $dtype);
                nd_add($pad, 255, $pad);
                push($parts, $pad);
            }
            return {"px": pxbuf.stack($parts, $dtype), "mode": $to, "palette": null};
        }
        return {"px": pxbuf.materialise(pxbuf.bands_from($colour, 0, $nb), $dtype),
                "mode": $to, "palette": null};
    }

    // The target wants alpha. Absent in the source, it is fully opaque --
    // which is what Pillow does for RGB -> RGBA and L -> LA.
    $a = $alpha;
    if ($a == null) {
        $a = nd_zeros(nd_shape(pxbuf.band($colour, 0)), $dtype);
        nd_add($a, 255, $a);
    }
    $parts = [];
    $i = 0;
    $colourBands = $nb - 1;
    while ($i < $colourBands) {
        if ($i < $cb) { push($parts, pxbuf.band($colour, $i)); }
        else { push($parts, pxbuf.band($colour, $cb - 1)); }
        $i = $i + 1;
    }
    if (mode.is_premultiplied($to)) {
        $pm = [];
        $i = 0;
        while ($i < len($parts)) {
            $num = nd_add(nd_multiply($parts[$i], $a), 127);
            push($pm, nd_floor_divide($num, 255));
            $i = $i + 1;
        }
        $parts = $pm;
    }
    push($parts, $a);
    return {"px": pxbuf.stack($parts, $dtype), "mode": $to, "palette": null};
}
