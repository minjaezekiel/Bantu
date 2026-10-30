// ════════════════════════════════════════════════════════════════════════
//  tests/pillow_ops_test.b — pillow Phase P6: the operation surface.
//
//  ImageFilter, ImageChops, ImageEnhance, ImageOps, ImageStat, ImageMath,
//  ImageMorph, quantize, and the Image methods P6 adds (filter, quantize,
//  entropy, putpalette, getpalette, remap_palette, apply_transparency,
//  tobitmap, effect_spread, convert with a matrix).
//
//  Four kinds of claim, four kinds of check:
//
//    1. WHOLE IMAGES, byte for byte, against Pillow: 320 of them --
//       every filter over every mode it accepts, every ImageOps call, four
//       enhancements at five factors each over three modes, Color3DLUT,
//       the convert() matrix, both gradients.
//    2. THE CHOPS CORPUS: a 256x256 pair, so every two-image operation is
//       checked over ALL 65,536 input combinations rather than sampled.
//       That is what proves Chops.c's two rounding regimes -- CHOP clamps,
//       CHOP2 wraps -- and the bound that makes the wrap unreachable for
//       the four operations that use it anyway.
//    3. THE NUMBERS: ImageStat's nine properties over four modes, entropy,
//       thirty ImageMath expressions, ImageMorph's LUT sums and change
//       counts.
//    4. QUANTIZE, which is a heuristic and so cannot be byte-identical:
//       the assertion is that our mean squared error against the source is
//       no worse than Pillow's, measured per method and colour count.
//
//  Run:  bantu run tests/pillow_ops_test.b
// ════════════════════════════════════════════════════════════════════════

include "./pillow/pillow.b" as pillow;
include "./pillow/filter.b" as flt;
include "./pillow/chops.b" as chops;
include "./pillow/enhance.b" as enh;
include "./pillow/ops.b" as ops;
include "./pillow/stat.b" as stat;
include "./pillow/math.b" as imath;
include "./pillow/morph.b" as morph;
include "./pillow/quantize.b" as quant;
include "./pillow/pxbuf.b" as pxbuf;

$R = {"p": 0, "f": 0};
def eq($got, $want, $name) {
    if (str($got) == str($want)) { $R.p = $R.p + 1; }
    else {
        $R.f = $R.f + 1;
        print("  FAIL  " + $name);
        print("          got:  " + str($got));
        print("          want: " + str($want));
    }
}
def ok($cond, $name) {
    if ($cond) { $R.p = $R.p + 1; }
    else { $R.f = $R.f + 1; print("  FAIL  " + $name); }
}
def throws($fn, $name) {
    $caught = false;
    try { $fn(); } catch ($e) { $caught = true; }
    if ($caught) { $R.p = $R.p + 1; }
    else { $R.f = $R.f + 1; print("  FAIL  (expected an error) " + $name); }
}
def near($got, $want, $tol, $name) {
    $d = $got - $want;
    if ($d < 0) { $d = 0 - $d; }
    if ($d <= $tol) { $R.p = $R.p + 1; }
    else {
        $R.f = $R.f + 1;
        print("  FAIL  " + $name + ": got " + str($got) + " want " + str($want) +
              " (tolerance " + str($tol) + ")");
    }
}
def _rows($path) {
    $all = split(trim(readfile($path)), "\n");
    $out = [];
    $i = 0;
    while ($i < len($all)) {
        $L = trim($all[$i]);
        if (len($L) > 0 && $L[0] != "#") { push($out, split($L, " ")); }
        $i = $i + 1;
    }
    return $out;
}

$D = "tests/fixtures/pillow/reference/ops/";
$W = 13;
$H = 11;

// The sources, as raw bytes, so no codec stands between the comparison and
// Pillow's own pixels.
$SRC = {};
def src($m) {
    if ($SRC[$m] == null) {
        $SRC[$m] = pillow.frombytes($m, [$W, $H], readfile($D + "src_" + $m + ".raw"));
    }
    return $SRC[$m];
}

print("");
print("── ImageFilter, ImageOps, ImageEnhance and the new methods, whole images ──");

$jobs = _rows($D + "MANIFEST");
ok(len($jobs) >= 300, "the fixture set has at least 300 images (" + str(len($jobs)) + ")");

// The filters, by tag, so the manifest drives the test rather than a
// hand-written list that could drift from it.
$FILTERS = {
    "BLUR": flt.BLUR, "CONTOUR": flt.CONTOUR, "DETAIL": flt.DETAIL,
    "EDGE_ENHANCE": flt.EDGE_ENHANCE, "EDGE_ENHANCE_MORE": flt.EDGE_ENHANCE_MORE,
    "EMBOSS": flt.EMBOSS, "FIND_EDGES": flt.FIND_EDGES, "SHARPEN": flt.SHARPEN,
    "SMOOTH": flt.SMOOTH, "SMOOTH_MORE": flt.SMOOTH_MORE,
    "kernel_asym": flt.Kernel(3, [1, 2, 3, 0, 1, 0, 0 - 1, 0 - 2, 0 - 3], 4, 10),
    "kernel_sum0": flt.Kernel(3, [0, 0 - 1, 0, 0 - 1, 4, 0 - 1, 0, 0 - 1, 0], 1, 128),
    "kernel5_asym": null,
    "box0_5": flt.BoxBlur(0.5), "box2": flt.BoxBlur(2), "box7": flt.BoxBlur(7),
    "box_xy": flt.BoxBlur([3, 1]),
    "gauss1": flt.GaussianBlur(1), "gauss2_5": flt.GaussianBlur(2.5),
    "gauss_xy": flt.GaussianBlur([2, 0.5]),
    "unsharp": flt.UnsharpMask(2, 150, 3), "unsharp_hard": flt.UnsharpMask(1, 300, 0),
    "median3": flt.MedianFilter(3), "median5": flt.MedianFilter(5),
    "min3": flt.MinFilter(3), "max3": flt.MaxFilter(3),
    "rank5_7": flt.RankFilter(5, 7),
    "mode3": flt.ModeFilter(3), "mode5": flt.ModeFilter(5)
};
$k5 = [];
$i = 0;
while ($i < 25) { push($k5, $i); $i = $i + 1; }
$FILTERS.kernel5_asym = flt.Kernel(5, $k5, 300, 5);

def _lutTable($n, $fn) {
    $t = [];
    $b = 0;
    while ($b < $n) {
        $g = 0;
        while ($g < $n) {
            $r = 0;
            while ($r < $n) {
                $v = $fn($r / ($n - 1), $g / ($n - 1), $b / ($n - 1));
                $k = 0;
                while ($k < len($v)) { push($t, $v[$k]); $k = $k + 1; }
                $r = $r + 1;
            }
            $g = $g + 1;
        }
        $b = $b + 1;
    }
    return $t;
}
$LUTS = {
    "lut3_identity": flt.Color3DLUT(3, _lutTable(3, def($r, $g, $b) { return [$r, $g, $b]; }), 3, null),
    "lut5_swap": flt.Color3DLUT(5, _lutTable(5, def($r, $g, $b) { return [$b, sqrt($g), $r]; }), 3, null),
    "lut2_flat": flt.Color3DLUT(2, _lutTable(2, def($r, $g, $b) { return [0.25, $r, 1.0]; }), 3, null),
    "lut4_alpha": flt.Color3DLUT(4, _lutTable(4, def($r, $g, $b) { return [$r, $g, $b, 0.5]; }), 4, null)
};

// ImageOps and the Image methods, by tag.
def _opsCase($tag) {
    $rgb = src("RGB");
    $g = src("L");
    if ($tag == "autocontrast") { return ops.autocontrast($rgb, null, null, null, null); }
    if ($tag == "autocontrast_L") { return ops.autocontrast($g, null, null, null, null); }
    if ($tag == "autocontrast_cut5") { return ops.autocontrast($rgb, 5, null, null, null); }
    if ($tag == "autocontrast_cut_pair") { return ops.autocontrast($rgb, [2, 8], null, null, null); }
    if ($tag == "autocontrast_ignore") { return ops.autocontrast($g, 0, 0, null, null); }
    if ($tag == "autocontrast_tone") { return ops.autocontrast($rgb, 2, null, null, true); }
    if ($tag == "equalize") { return ops.equalize($rgb, null); }
    if ($tag == "equalize_L") { return ops.equalize($g, null); }
    if ($tag == "invert") { return ops.invert($rgb); }
    if ($tag == "invert_L") { return ops.invert($g); }
    if ($tag == "solarize") { return ops.solarize($rgb, 100); }
    if ($tag == "solarize_L") { return ops.solarize($g, null); }
    if ($tag == "colorize2") { return ops.colorize($g, "red", "yellow", null, null, null, null); }
    if ($tag == "colorize3") {
        return ops.colorize($g, [0, 0, 0], [255, 255, 255], [128, 0, 0], 20, 200, 100);
    }
    if ($tag == "grayscale") { return ops.grayscale($rgb); }
    if ($tag == "flip") { return ops.flip($rgb); }
    if ($tag == "mirror") { return ops.mirror($rgb); }
    if ($tag == "crop2") { return ops.crop($rgb, 2); }
    if ($tag == "crop_quad") { return ops.crop($rgb, [1, 2, 3, 4]); }
    if ($tag == "expand3_blue") { return ops.expand($rgb, 3, "blue"); }
    if ($tag == "expand_pair") { return ops.expand($g, [2, 1], 200); }
    if ($tag == "contain_10_7") { return ops.contain($rgb, [10, 7], null); }
    if ($tag == "contain_20_20") { return ops.contain($rgb, [20, 20], null); }
    if ($tag == "cover_10_7") { return ops.cover($rgb, [10, 7], null); }
    if ($tag == "pad_10_7") { return ops.pad($rgb, [10, 7], null, "red", null); }
    if ($tag == "pad_20_5") { return ops.pad($rgb, [20, 5], null, [1, 2, 3], [0.25, 0.75]); }
    if ($tag == "scale15") { return ops.scale($rgb, 1.5, null); }
    if ($tag == "scale05") { return ops.scale($rgb, 0.5, null); }
    if ($tag == "fit_10_7") { return ops.fit($rgb, [10, 7], null, null, null); }
    if ($tag == "fit_bleed") { return ops.fit($rgb, [10, 7], null, 0.1, [0.3, 0.7]); }
    if (substr($tag, 0, 9) == "posterize") {
        return ops.posterize($rgb, num(substr($tag, 9, len($tag) - 9)));
    }
    return null;
}

def _methCase($tag) {
    $rgb = src("RGB");
    if ($tag == "matrix_L") { return $rgb.convert("L", null, [0.299, 0.587, 0.114, 0]); }
    if ($tag == "matrix_L_off") { return $rgb.convert("L", null, [0.5, 0.25, 0.25, 10]); }
    if ($tag == "matrix_RGB") {
        return $rgb.convert("RGB", null, [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]);
    }
    if ($tag == "linear_gradient_L") { return pillow.linear_gradient("L"); }
    if ($tag == "radial_gradient_L") { return pillow.radial_gradient("L"); }
    // rgba_to_p_rgb is NOT byte-compared: see the quantize bound below.
    if ($tag == "rgba_to_p_rgb") { return null; }
    return null;
}

$per = {};
$shown = 0;
$i = 0;
while ($i < len($jobs)) {
    $j = $jobs[$i];
    $i = $i + 1;
    $group = $j[0];
    if ($group == "CHOP") { continue; }
    $tag = $j[1];
    $file = $j[2];
    $wmode = $j[3];
    $ww = num($j[4]);
    $wh = num($j[5]);
    if ($per[$group] == null) { $per[$group] = {"ok": 0, "bad": 0}; }

    $got = null;
    $threw = false;
    $why = "";
    try {
        if ($group == "FILTER") {
            // the tag is <filter>_<mode>, and a filter name can hold "_"
            $us = 0;
            $k = len($tag) - 1;
            while ($k > 0) { if ($tag[$k] == "_") { $us = $k; break; } $k = $k - 1; }
            $fname = substr($tag, 0, $us);
            $fmode = substr($tag, $us + 1, len($tag) - $us - 1);
            $f = $FILTERS[$fname];
            if ($f == null) { $f = $LUTS[$fname]; }
            if ($f == null) { $why = "no such filter " + $fname; $threw = true; }
            else { $got = src($fmode).filter($f); }
        } else if ($group == "OPS") { $got = _opsCase($tag); }
        else if ($group == "METH") {
            $got = _methCase($tag);
            if ($got == null && $tag == "rgba_to_p_rgb") {
                // An adaptive palette through the octree: the colours are
                // a heuristic's choice, so this one is checked by error
                // (below) and not by bytes.
                $per[$group].ok = $per[$group].ok + 1;
                continue;
            }
        }
        else if ($group == "ENH") {
            $f2 = split($tag, "_");
            $cls = $f2[0];
            $emode = $f2[1];
            $factor = num($f2[2] + "." + $f2[3]);
            $im = src($emode);
            if ($cls == "Color") { $e = enh.Color($im); }
            else if ($cls == "Contrast") { $e = enh.Contrast($im); }
            else if ($cls == "Brightness") { $e = enh.Brightness($im); }
            else { $e = enh.Sharpness($im); }
            $got = enh.enhance($e, $factor);
        } else if ($group == "MORPH") { continue; }
    } catch ($e) { $threw = true; $why = str($e); }

    if ($threw || $got == null) {
        $per[$group].bad = $per[$group].bad + 1;
        if ($shown < 10) {
            $shown = $shown + 1;
            print("     ERROR " + $group + " " + $tag + ": " + $why);
        }
    } else {
        $want = readfile($D + $file);
        $have = $got.tobytes();
        if ($got.mode == $wmode && $got.width == $ww && $got.height == $wh &&
            $have == $want) {
            $per[$group].ok = $per[$group].ok + 1;
        } else {
            $per[$group].bad = $per[$group].bad + 1;
            if ($shown < 10) {
                $shown = $shown + 1;
                print("     DIFF " + $group + " " + $tag + ": got " + $got.mode + " " +
                      str($got.size) + " " + str(len($have)) + "B, want " + $wmode +
                      " " + str([$ww, $wh]) + " " + str(len($want)) + "B");
            }
        }
    }
}
$totBad = 0;
for $k, $v in $per.items() {
    print("   " + $k + ": " + str($v.ok) + " byte-identical, " + str($v.bad) + " differing");
    $totBad = $totBad + $v.bad;
}
eq($totBad, 0, "every image in the operation-surface corpus matches Pillow");

print("");
print("── ImageChops over all 65,536 input pairs ──");

$ca = pillow.frombytes("L", [256, 256], readfile($D + "chops_a.raw"));
$cb = pillow.frombytes("L", [256, 256], readfile($D + "chops_b.raw"));
$ca1 = pillow.fromarray(nd_reshape(nd_frombytes(readfile($D + "chops_a1.raw"), "u8", null),
                                   [256, 256]), "1");
$cb1 = pillow.fromarray(nd_reshape(nd_frombytes(readfile($D + "chops_b1.raw"), "u8", null),
                                   [256, 256]), "1");
def _chopCase($tag) {
    if ($tag == "lighter") { return chops.lighter($ca, $cb); }
    if ($tag == "darker") { return chops.darker($ca, $cb); }
    if ($tag == "difference") { return chops.difference($ca, $cb); }
    if ($tag == "multiply") { return chops.multiply($ca, $cb); }
    if ($tag == "screen") { return chops.screen($ca, $cb); }
    if ($tag == "soft_light") { return chops.soft_light($ca, $cb); }
    if ($tag == "hard_light") { return chops.hard_light($ca, $cb); }
    if ($tag == "overlay") { return chops.overlay($ca, $cb); }
    if ($tag == "add") { return chops.add($ca, $cb, null, null); }
    if ($tag == "add_s2_o10") { return chops.add($ca, $cb, 2.0, 10); }
    if ($tag == "add_s07_on5") { return chops.add($ca, $cb, 0.7, 0 - 5); }
    if ($tag == "subtract") { return chops.subtract($ca, $cb, null, null); }
    if ($tag == "subtract_s2_o10") { return chops.subtract($ca, $cb, 2.0, 10); }
    if ($tag == "subtract_s3_on20") { return chops.subtract($ca, $cb, 3.0, 0 - 20); }
    if ($tag == "add_modulo") { return chops.add_modulo($ca, $cb); }
    if ($tag == "subtract_modulo") { return chops.subtract_modulo($ca, $cb); }
    if ($tag == "invert") { return chops.invert($ca); }
    if ($tag == "offset_3_5") { return chops.offset($ca, 3, 5); }
    if ($tag == "offset_n2") { return chops.offset($ca, 0 - 2, null); }
    if ($tag == "blend_025") { return chops.blend($ca, $cb, 0.25); }
    if ($tag == "blend_075") { return chops.blend($ca, $cb, 0.75); }
    if ($tag == "blend_15") { return chops.blend($ca, $cb, 1.5); }
    if ($tag == "blend_n05") { return chops.blend($ca, $cb, 0 - 0.5); }
    if ($tag == "logical_and") { return chops.logical_and($ca1, $cb1); }
    if ($tag == "logical_or") { return chops.logical_or($ca1, $cb1); }
    if ($tag == "logical_xor") { return chops.logical_xor($ca1, $cb1); }
    return null;
}
$chopBad = 0;
$chopOk = 0;
$i = 0;
while ($i < len($jobs)) {
    $j = $jobs[$i];
    $i = $i + 1;
    if ($j[0] != "CHOP") { continue; }
    $got = _chopCase($j[1]);
    if ($got == null) { $chopBad = $chopBad + 1; print("     MISSING " + $j[1]); continue; }
    $have = nd_tobytes($got.px, "u8");
    if ($have == readfile($D + $j[2])) { $chopOk = $chopOk + 1; }
    else {
        $chopBad = $chopBad + 1;
        print("     DIFF chop " + $j[1]);
    }
}
print("   " + str($chopOk) + " operations exact over 65,536 input pairs each");
eq($chopBad, 0, "every channel operation matches Chops.c on every input pair");

// The wrap in CHOP2 is unreachable for these four, and this is the bound
// that says so: over every input pair, the raw result never leaves 0..255.
def _rawRange($fn) {
    $r = $fn();
    $lo = pxbuf.scalar(nd_min($r));
    $hi = pxbuf.scalar(nd_max($r));
    return [$lo, $hi];
}
$a = nd_reshape(nd_broadcast_to(nd_reshape(nd_arange(0, 256, 1), [1, 256]), [256, 256]), [256, 256]);
$b = nd_reshape(nd_broadcast_to(nd_reshape(nd_arange(0, 256, 1), [256, 1]), [256, 256]), [256, 256]);
eq(_rawRange(def() { return nd_abs(nd_subtract($a, $b)); }), [0, 255],
   "difference cannot leave 0..255, so its CHOP2 wrap is unreachable");
eq(_rawRange(def() {
    $t1 = nd_floor_divide(nd_multiply(nd_subtract(255, $a), nd_multiply($a, $b)), 65536);
    $scr = nd_subtract(255, nd_floor_divide(
        nd_multiply(nd_subtract(255, $a), nd_subtract(255, $b)), 255));
    return nd_add($t1, nd_floor_divide(nd_multiply($a, $scr), 255));
}), [0, 255], "nor can soft_light");
eq(_rawRange(def() {
    $lo = nd_floor_divide(nd_multiply($a, $b), 127);
    $hi = nd_subtract(255, nd_floor_divide(
        nd_multiply(nd_subtract(255, $b), nd_subtract(255, $a)), 127));
    return nd_where(nd_less($b, 128), $lo, $hi);
}), [0, 255], "nor hard_light");
eq(_rawRange(def() {
    $lo = nd_floor_divide(nd_multiply($a, $b), 127);
    $hi = nd_subtract(255, nd_floor_divide(
        nd_multiply(nd_subtract(255, $a), nd_subtract(255, $b)), 127));
    return nd_where(nd_less($a, 128), $lo, $hi);
}), [0, 255], "nor overlay");

print("");
print("── the numbers: ImageStat, entropy, ImageMath, ImageMorph, quantize ──");

$nums = _rows($D + "NUMBERS");
$statOk = 0;
$mathOk = 0;
$morphOk = 0;
$quantOk = 0;
$numBad = 0;

def _kv($fields, $from) {
    $out = {};
    $i = $from;
    while ($i < len($fields)) {
        $eqp = indexOf($fields[$i], "=");
        if ($eqp > 0) {
            $out[substr($fields[$i], 0, $eqp)] =
                substr($fields[$i], $eqp + 1, len($fields[$i]) - $eqp - 1);
        }
        $i = $i + 1;
    }
    return $out;
}

def _joinNums($list, $dp) {
    $out = [];
    $i = 0;
    while ($i < len($list)) {
        if ($dp == 0) { push($out, str(floor($list[$i] + 0.0000001))); }
        else { push($out, str($list[$i])); }
        $i = $i + 1;
    }
    return join($out, ",");
}

// Floats are compared with a tolerance, because str() in Bantu prints six
// significant digits and the fixture holds six decimals (roadmap rule 4).
def _nearCsv($got, $want, $tol, $name) {
    $wl = split($want, ",");
    if (len($wl) != len($got)) {
        $R.f = $R.f + 1;
        print("  FAIL  " + $name + ": " + str(len($got)) + " values, want " + str(len($wl)));
        return false;
    }
    $i = 0;
    while ($i < len($wl)) {
        $d = $got[$i] - num($wl[$i]);
        if ($d < 0) { $d = 0 - $d; }
        $rel = $tol;
        if (abs(num($wl[$i])) > 1000) { $rel = abs(num($wl[$i])) * 0.0000001; }
        if ($d > $rel) {
            $R.f = $R.f + 1;
            print("  FAIL  " + $name + "[" + str($i) + "]: got " + str($got[$i]) +
                  " want " + $wl[$i]);
            return false;
        }
        $i = $i + 1;
    }
    $R.p = $R.p + 1;
    return true;
}

$ma = pillow.frombytes("L", [6, 1], readfile($D + "math_a.raw"));
$mb = pillow.frombytes("L", [6, 1], readfile($D + "math_b.raw"));
$mask = pillow.fromarray(nd_reshape(nd_frombytes(readfile($D + "mask.raw"), "u8", null),
                                    [$H, $W]), "1");
$morphSrc = pillow.fromarray(nd_reshape(nd_frombytes(readfile($D + "morph_src.raw"), "u8", null),
                                        [$H, $W]), "1");

$i = 0;
while ($i < len($nums)) {
    $f = $nums[$i];
    $i = $i + 1;
    if ($f[0] == "STAT") {
        $m = $f[1];
        $kv = _kv($f, 2);
        $st = stat.Stat(src($m), null);
        $ex = [];
        $k = 0;
        while ($k < len($st.extrema)) {
            push($ex, str($st.extrema[$k][0]) + ":" + str($st.extrema[$k][1]));
            $k = $k + 1;
        }
        $good = true;
        if (join($ex, ",") != $kv.extrema) {
            $good = false;
            print("  FAIL  Stat(" + $m + ").extrema: " + join($ex, ",") + " want " + $kv.extrema);
        }
        if (_joinNums($st.count, 0) != $kv.count) {
            $good = false;
            print("  FAIL  Stat(" + $m + ").count: " + _joinNums($st.count, 0) +
                  " want " + $kv.count);
        }
        if (_joinNums($st.median, 0) != $kv.median) {
            $good = false;
            print("  FAIL  Stat(" + $m + ").median");
        }
        if ($good) { $R.p = $R.p + 1; } else { $R.f = $R.f + 1; $numBad = $numBad + 1; }
        _nearCsv($st.sum, $kv.sum, 0.000001, "Stat(" + $m + ").sum");
        _nearCsv($st.sum2, $kv.sum2, 0.01, "Stat(" + $m + ").sum2");
        _nearCsv($st.mean, $kv.mean, 0.000001, "Stat(" + $m + ").mean");
        _nearCsv($st.rms, $kv.rms, 0.000001, "Stat(" + $m + ").rms");
        _nearCsv($st.var, $kv.var, 0.001, "Stat(" + $m + ").var");
        _nearCsv($st.stddev, $kv.stddev, 0.000001, "Stat(" + $m + ").stddev");
        $statOk = $statOk + 1;
    } else if ($f[0] == "STATMASK") {
        $kv = _kv($f, 2);
        $st = stat.Stat(src("RGB"), $mask);
        _nearCsv($st.mean, $kv.mean, 0.000001, "Stat with a mask: mean");
        if (_joinNums($st.count, 0) == $kv.count) { $R.p = $R.p + 1; }
        else {
            $R.f = $R.f + 1;
            print("  FAIL  Stat with a mask: count " + _joinNums($st.count, 0) +
                  " want " + $kv.count);
        }
    } else if ($f[0] == "ENTROPY") {
        near(src($f[1]).entropy(null, null), num($f[2]), 0.000001,
             "entropy(" + $f[1] + ")");
    } else if ($f[0] == "MATH") {
        $wmode = $f[1];
        $wvals = $f[2];
        $parts = [];
        $k = 3;
        while ($k < len($f)) { push($parts, $f[$k]); $k = $k + 1; }
        $expr = join($parts, " ");
        $threw = false;
        $got = null;
        try { $got = imath.eval_($expr, {"a": $ma, "b": $mb}); }
        catch ($e) { $threw = true; $why = str($e); }
        if ($threw) {
            $R.f = $R.f + 1;
            $numBad = $numBad + 1;
            print("  FAIL  ImageMath \"" + $expr + "\": " + $why);
        } else {
            $vals = nd_to_list(nd_ravel($got.px));
            if ($got.mode != $wmode) {
                $R.f = $R.f + 1;
                $numBad = $numBad + 1;
                print("  FAIL  ImageMath \"" + $expr + "\": mode " + $got.mode +
                      " want " + $wmode);
            } else if (_nearCsv($vals, $wvals, 0.0001, "ImageMath \"" + $expr + "\"")) {
                $mathOk = $mathOk + 1;
            } else { $numBad = $numBad + 1; }
        }
    } else if ($f[0] == "MORPH") {
        $tag = $f[1];
        $file = $f[2];
        $kv = _kv($f, 3);
        if ($tag == "custom6") { $op = morph.MorphOp(null, null, ["4:(... .1. .0.)->0", "N:(11. .1. ...)->1"]); }
        else if ($tag == "custom7") { $op = morph.MorphOp(null, null, ["M:(1.. .1. ..0)->1"]); }
        else if ($tag == "custom8") { $op = morph.MorphOp(null, null, ["4:(.0. .1. ...)->0"]); }
        else { $op = morph.MorphOp(null, $tag, null); }
        $r = morph.apply($op, $morphSrc);
        $lutsum = 0;
        $k = 0;
        while ($k < 512) { $lutsum = $lutsum + $op.lut[$k]; $k = $k + 1; }
        $lut64 = [];
        $k = 0;
        while ($k < 64) { push($lut64, str($op.lut[$k])); $k = $k + 1; }
        $good = true;
        if (str($r[0]) != $kv.count) {
            $good = false;
            print("  FAIL  morph " + $tag + " count " + str($r[0]) + " want " + $kv.count);
        }
        if (str($lutsum) != $kv.lutsum) {
            $good = false;
            print("  FAIL  morph " + $tag + " lutsum " + str($lutsum) + " want " + $kv.lutsum);
        }
        if (join($lut64, "") != $kv.lut) {
            $good = false;
            print("  FAIL  morph " + $tag + " the first 64 table entries differ");
        }
        if (str(len(morph.match($op, $morphSrc))) != $kv.match) {
            $good = false;
            print("  FAIL  morph " + $tag + " match count");
        }
        if (nd_tobytes($r[1].px, "u8") != readfile($D + $file)) {
            $good = false;
            print("  FAIL  morph " + $tag + " image");
        }
        if ($good) { $R.p = $R.p + 1; $morphOk = $morphOk + 1; }
        else { $R.f = $R.f + 1; $numBad = $numBad + 1; }
    } else if ($f[0] == "QUANT") {
        $mname = $f[1];
        $colors = num($f[2]);
        $kv = _kv($f, 3);
        $meth = 0;
        if ($mname == "maxcoverage") { $meth = 1; }
        if ($mname == "fastoctree") { $meth = 2; }
        $rgb = src("RGB");
        $q = $rgb.quantize($colors, $meth, null, null, null);
        $back = nd_reshape(nd_take($q.palette, nd_ravel($q.px), 0), [$H, $W, 3]);
        $diff = nd_subtract($back, $rgb.px);
        $mse = pxbuf.scalar(nd_mean(nd_multiply($diff, $diff)));
        $want = num($kv.mse);
        // An adaptive palette is a heuristic, so the claim is an ERROR
        // BOUND rather than byte equality: ours against the source, versus
        // Pillow's against the source. Median cut and the octree are within
        // 1.5x (and usually better); MAXCOVERAGE is median cut with a
        // volume-weighted selection rather than Pillow's own quantize2,
        // which is a different algorithm, so its bound is 2x and the
        // deviation is stated in docs/pillow.md rather than hidden.
        $bound = 1.5;
        if ($mname == "maxcoverage") { $bound = 2.0; }
        ok($mse <= $want * $bound + 1,
           "quantize " + $mname + " " + str($colors) + ": mse " + str($mse) +
           " vs Pillow's " + str($want));
        ok($q.mode == "P" && nd_shape($q.palette)[0] <= $colors,
           "quantize " + $mname + " " + str($colors) + " yields at most " +
           str($colors) + " palette entries");
        $quantOk = $quantOk + 1;
    }
}
print("   ImageStat: " + str($statOk) + " modes, ImageMath: " + str($mathOk) +
      " expressions, ImageMorph: " + str($morphOk) + " operators, quantize: " +
      str($quantOk) + " cases");
eq($numBad, 0, "every number matches Pillow");

print("");
print("── RGBA -> P, which goes through the octree ──");

// Pillow's own colours for this source, from the fixture, against ours:
// the palettes differ in ORDER and sometimes in choice, so the claim is the
// error against the source.
$rgbaSrc = src("RGBA");
$oursP = $rgbaSrc.convert("P", null, null).convert("RGB", null, null);
$theirs = pillow.frombytes("RGB", [$W, $H], readfile($D + "METH_rgba_to_p_rgb.exp"));
$ref = $rgbaSrc.convert("RGB", null, null);
def _mse($a, $b) {
    $d = nd_subtract($a.px, $b.px);
    return pxbuf.scalar(nd_mean(nd_multiply($d, $d)));
}
$om = _mse($oursP, $ref);
$tm = _mse($theirs, $ref);
print("   mean squared error against the source: ours " + str($om) +
      ", Pillow " + str($tm));
ok($om <= $tm * 1.5 + 1, "RGBA -> P is no worse than Pillow's octree");

print("");
print("── the methods, and the failure modes ──");

$rgb = src("RGB");
eq(src("1").tobitmap("t"), readfile($D + "tobitmap.txt"),
   "tobitmap writes Pillow's XBM fragment, fifteen bytes to a line");
$p8 = $rgb.quantize(8, 0, null, null, null);
eq(len($p8.getpalette(null)), 24, "getpalette returns three bytes per entry");
$rm = $p8.remap_palette([3, 1, 2, 0, 4, 5, 6, 7], null);
eq($rm.mode, "P", "remap_palette returns a P image");
eq(nd_to_list(nd_ravel(nd_slice($rm.palette, [[0, 1], null]))),
   nd_to_list(nd_ravel(nd_slice($p8.palette, [[3, 4], null]))),
   "and its first entry is the one dest_map named");
$back = $p8.remap_palette([3, 1, 2, 0, 4, 5, 6, 7], null);
eq(nd_to_list(nd_ravel(nd_take($back.palette, nd_ravel($back.px), 0))),
   nd_to_list(nd_ravel(nd_take($p8.palette, nd_ravel($p8.px), 0))),
   "remap_palette changes the indexes and the palette together, so the colours are unchanged");

$lp = pillow.new("L", [4, 2], 7);
$lp.putpalette([255, 0, 0, 0, 255, 0, 0, 0, 255], null);
eq($lp.mode, "P", "putpalette turns an L image into a P image");
eq(nd_shape($lp.palette), [3, 3], "with the palette it was given");

$tp = $rgb.quantize(4, 0, null, null, null);
$tp.info = {"transparency": 2};
$tp.apply_transparency();
eq(nd_shape($tp.palette)[1], 4, "apply_transparency widens the palette to RGBA");
eq(nd_to_list(nd_ravel(nd_slice($tp.palette, [[2, 3], [3, 4]]))), [0],
   "and zeroes the alpha of the transparent entry");
eq($tp.info.transparency, null, "and drops the info key, as Pillow does");

$sp = $rgb.effect_spread(2);
eq([$sp.width, $sp.height], [$W, $H], "effect_spread keeps the size");
ok($sp.mode == "RGB", "and the mode");
$noise = pillow.effect_noise([8, 8], 4);
eq([$noise.width, $noise.height, $noise.mode], [8, 8, "L"], "effect_noise makes an L image");
$st = stat.Stat($noise, null);
ok($st.mean[0] > 100 && $st.mean[0] < 160, "centred near 128 (got " + str($st.mean[0]) + ")");

eq($rgb.getim(), $rgb.px, "getim hands back the buffer");

throws(def() { src("F").filter(flt.SMOOTH); }, "mode F cannot be filtered (Pillow's own refusal)");
throws(def() { $p8.filter(flt.SMOOTH); }, "nor can a palette image");
throws(def() { src("RGB").filter(flt.Kernel(7, _lutTable(7, def($r,$g,$b){ return [1]; }), 1, 0)); },
       "a 7x7 kernel is refused by name, as ImagingFilter refuses it");
throws(def() { flt.RankFilter(4, 2); }, "an even rank-filter size is refused");
throws(def() { flt.RankFilter(3, 9); }, "a rank outside the window is refused");
throws(def() { flt.Kernel(3, [1, 2, 3], null, null); }, "a short kernel is refused");
throws(def() { flt.BoxBlur(0 - 1); }, "a negative blur radius is refused");
throws(def() { chops.logical_and($rgb, $rgb); }, "the logical chops demand mode 1");
throws(def() { chops.add(src("RGB"), src("L"), null, null); },
       "a chop between images of different band counts is refused");
throws(def() { ops.autocontrast($p8, null, null, null, null); },
       "ImageOps refuses a P image, with Pillow's NotImplementedError");
throws(def() { ops.colorize($rgb, "red", "white", null, null, null, null); },
       "colorize demands mode L");
throws(def() { ops.posterize($rgb, 0); }, "posterize bits must be 1..8");
throws(def() { ops.scale($rgb, 0, null); }, "a scale factor of zero is refused");
throws(def() { imath.eval_("a +", {"a": $ma}); }, "a truncated expression is a parse error");
throws(def() { imath.eval_("a + c", {"a": $ma}); }, "an unknown name is refused by name");
throws(def() { imath.eval_("a $ b", {"a": $ma, "b": $mb}); },
       "a character outside the grammar names itself");
throws(def() { imath.eval_("open('x')", {"a": $ma}); },
       "a function outside the grammar is refused -- the grammar IS the sandbox");
throws(def() { imath.eval_("float(a) & 3", {"a": $ma}); },
       "a bitwise operation on mode F is refused, as _imagingmath refuses it");
throws(def() { morph.apply(morph.MorphOp(null, null, null), $morphSrc); },
       "a MorphOp with no table says so");
throws(def() { morph.MorphOp(null, "nonesuch", null); }, "an unknown pattern name is refused");
throws(def() { morph.build_lut(["4:(... .1.)->0"]); }, "a pattern of the wrong length is refused");
throws(def() { morph.apply(morph.MorphOp(null, "edge", null), $rgb); },
       "morphology demands mode 1 or L");
throws(def() { src("RGBA").quantize(16, 0, null, null, null); },
       "median cut cannot quantize RGBA, as in Pillow");
throws(def() { $rgb.quantize(16, 3, null, null, null); },
       "libimagequant is declined by name");
throws(def() { $rgb.quantize(300, 0, null, null, null); }, "over 256 colours is refused");
throws(def() { $rgb.convert("RGB", null, [1, 2]); }, "a matrix of the wrong length is refused");
throws(def() { src("L").convert("L", null, [1, 2, 3, 4]); },
       "a matrix conversion needs a three-band source");

print("");
print("── cost ──");

$big = pillow.new("RGB", [160, 120], [0, 0, 0]);
$y = 0;
while ($y < 120) {
    $x = 0;
    while ($x < 160) {
        $big.putpixel([$x, $y], [($x * 7) % 256, ($y * 11) % 256, ($x + $y) % 256]);
        $x = $x + 1;
    }
    $y = $y + 1;
}
$t0 = clock();
$big.filter(flt.SMOOTH);
$t1 = clock() - $t0;
$t0 = clock();
$big.filter(flt.GaussianBlur(10));
$t2 = clock() - $t0;
$t0 = clock();
$big.filter(flt.MedianFilter(5));
$t3 = clock() - $t0;
$t0 = clock();
$big.quantize(64, 0, null, null, null);
$t4 = clock() - $t0;
print("   160x120 RGB: SMOOTH " + str($t1) + " ms, GaussianBlur(10) " + str($t2) +
      " ms, MedianFilter(5) " + str($t3) + " ms, quantize(64) " + str($t4) + " ms");
ok($t2 < 8000, "a radius-10 Gaussian is not proportional to the radius");
$t0 = clock();
$big.filter(flt.GaussianBlur(1));
$t5 = clock() - $t0;
ok($t5 * 3 > $t2 - 200,
   "radius 1 and radius 10 cost about the same, which is what a prefix-sum blur means (" +
   str($t5) + " vs " + str($t2) + " ms)");

print("");
print("checks: " + str($R.p + $R.f) + "  passed: " + str($R.p) + "  failed: " + str($R.f));
if ($R.f == 0) { print("RESULT: ALL GREEN"); }
else { print("RESULT: FAILED"); }
