// ════════════════════════════════════════════════════════════════════════
//  tests/pillow_resample_test.b — pillow Phase P3: resize().
//
//  378 reference images from tests/fixtures/pillow/reference/resample/:
//  nine modes, seven source/target shape pairs (downscale, upscale,
//  identity, non-square, 1x1, and two odd ones), six filters. Whole images,
//  not probe pixels, because the coefficient bounds differ per output column
//  and a single pixel proves nothing about the ones beside it.
//
//  Run:  bantu run tests/pillow_resample_test.b
// ════════════════════════════════════════════════════════════════════════

include "./pillow/pillow.b" as pillow;
include "./pillow/resample.b" as resample;
include "./pillow/mode.b" as mode;

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

// ─── the kernels, at the points where they are easy to get wrong ────────
print("── the six kernels ──");

eq(resample.support(resample.BOX), 0.5, "box support");
eq(resample.support(resample.BILINEAR), 1.0, "bilinear support");
eq(resample.support(resample.HAMMING), 1.0, "hamming support");
eq(resample.support(resample.BICUBIC), 2.0, "bicubic support");
eq(resample.support(resample.LANCZOS), 3.0, "lanczos support");
throws(def() { return resample.support(9); }, "an unknown filter id");

// the box filter is ASYMMETRIC: -0.5 is out, +0.5 is in
eq(resample.kernel(resample.BOX, 0.0), 1.0, "box at 0");
eq(resample.kernel(resample.BOX, 0.5), 1.0, "box at +0.5 is INSIDE");
eq(resample.kernel(resample.BOX, -0.5), 0.0, "box at -0.5 is OUTSIDE");
eq(resample.kernel(resample.BOX, 0.51), 0.0, "box past the support");

eq(resample.kernel(resample.BILINEAR, 0.0), 1.0, "bilinear at 0");
eq(resample.kernel(resample.BILINEAR, 0.25), 0.75, "bilinear is linear");
eq(resample.kernel(resample.BILINEAR, -0.25), 0.75, "bilinear is symmetric");
eq(resample.kernel(resample.BILINEAR, 1.0), 0.0, "bilinear at its support");

eq(resample.kernel(resample.BICUBIC, 0.0), 1.0, "bicubic at 0");
eq(resample.kernel(resample.BICUBIC, 1.0), 0.0, "bicubic at 1");
eq(resample.kernel(resample.BICUBIC, 2.0), 0.0, "bicubic at its support");
ok(resample.kernel(resample.BICUBIC, 1.5) < 0, "bicubic undershoots between 1 and 2");

eq(resample.kernel(resample.LANCZOS, 0.0), 1.0, "lanczos at 0");
eq(resample.kernel(resample.LANCZOS, 3.0), 0.0, "lanczos at its support");
ok(resample.kernel(resample.LANCZOS, 1.5) < 0, "lanczos undershoots");

eq(resample.kernel(resample.HAMMING, 0.0), 1.0, "hamming at 0");
eq(resample.kernel(resample.HAMMING, 1.0), 0.0, "hamming at its support");
// the windowed value depends on C's float literals, so pin one
$hv = resample.kernel(resample.HAMMING, 0.5);
ok($hv > 0.3401 && $hv < 0.3441, "hamming at 0.5 uses the float-promoted window (" + str($hv) + ")");

eq(resample.filter_name(0), "NEAREST", "filter_name");
eq(resample.filter_name(3), "BICUBIC", "filter_name bicubic");
eq(resample.filter_id("LANCZOS"), 1, "filter_id");
eq(resample.filter_id("nope"), null, "filter_id of nonsense");

// ─── precompute_coeffs ──────────────────────────────────────────────────
print("── the coefficient tables ──");

// identity: bilinear 1:1 has support 1, ksize 3, and every row should be a
// single unit weight on its own column
$pre = resample.precompute(8, 0, 8, 8, resample.BILINEAR);
eq($pre.ksize, 3, "bilinear 1:1 has ksize 3");
eq($pre.outSize, 8, "and one row per output column");
$bad = 0;
$i = 0;
while ($i < 8) {
    $sum = 0.0;
    $t = 0;
    while ($t < $pre.ksize) { $sum = $sum + $pre.kk[$i][$t]; $t = $t + 1; }
    if ($sum < 0.9999999 || $sum > 1.0000001) { $bad = $bad + 1; }
    $i = $i + 1;
}
eq($bad, 0, "every coefficient row sums to 1");

// downscaling widens the kernel: a 4x box reduction has filterscale 4
$pre4 = resample.precompute(32, 0, 32, 8, resample.BOX);
eq($pre4.ksize, 5, "a 4x box reduction has ksize 5");
eq($pre4.xmin[0], 0, "the first output column starts at 0");
eq($pre4.count[0], 4, "and averages 4 input columns");
$pre4b = resample.precompute(32, 0, 32, 8, resample.BICUBIC);
eq($pre4b.ksize, 17, "a 4x bicubic reduction has ksize 17");

// upscaling does not: filterscale is floored at 1
$preUp = resample.precompute(8, 0, 8, 32, resample.BICUBIC);
eq($preUp.ksize, 5, "upscaling keeps bicubic's ksize at 5");

throws(def() { return resample.precompute(8, 0, 8, 0, resample.BOX); }, "a zero output size");

// ─── the corpus: 378 images against Pillow ──────────────────────────────
print("── 378 reference images, whole-image comparison ──");

$rows = split(readfile("tests/fixtures/pillow/reference/resample_index.txt"), "\n");
$jobs = [];
$i = 0;
while ($i < len($rows)) {
    $L = trim($rows[$i]);
    if (len($L) > 0 && $L[0] != "#") { push($jobs, split($L, " ")); }
    $i = $i + 1;
}
eq(len($jobs) > 370, true, "the corpus index has over 370 jobs");

$D = "tests/fixtures/pillow/reference/resample/";
$stat = {"exact": 0, "wrong": 0, "err": 0, "px": 0, "badpx": 0};
$byMode = {};
$shown = 0;

$i = 0;
while ($i < len($jobs)) {
    $j = $jobs[$i];
    $m = $j[0];
    $sw = num($j[1]);  $sh = num($j[2]);
    $dw = num($j[3]);  $dh = num($j[4]);
    $fname = $j[5];
    $status = $j[6];
    $tag = $m + "_" + str($sw) + "x" + str($sh) + "_" + str($dw) + "x" + str($dh);

    if ($status != "OK") { $i = $i + 1; continue; }

    $fid = resample.filter_id($fname);
    $elem = mode.wire_elem($m);
    $nb = mode.bands($m);
    $src = pillow.frombytes($m, [$sw, $sh], readfile($D + $tag + ".src"));

    $threw = false;
    $got = null;
    try { $got = $src.resize([$dw, $dh], $fid, null, null); } catch ($e) { $threw = true; }
    if ($threw) {
        $stat.err = $stat.err + 1;
        if ($shown < 6) { $shown = $shown + 1; print("     ERROR " + $tag + " " + $fname); }
        $i = $i + 1;
        continue;
    }

    $wantBytes = readfile($D + $tag + "_" + $fname + ".dst");
    $gotBytes = $got.tobytes();
    if ($gotBytes == $wantBytes) {
        $stat.exact = $stat.exact + 1;
        $stat.px = $stat.px + $dw * $dh;
    } else {
        // count how many VALUES differ, so a near miss is distinguishable
        $gb = bytes($gotBytes);
        $wb = bytes($wantBytes);
        $diff = 0;
        $k = 0;
        while ($k < len($wb) && $k < len($gb)) {
            if ($gb[$k] != $wb[$k]) { $diff = $diff + 1; }
            $k = $k + 1;
        }
        if (len($gb) != len($wb)) { $diff = $diff + 1000000; }
        $stat.wrong = $stat.wrong + 1;
        $stat.badpx = $stat.badpx + $diff;
        $stat.px = $stat.px + $dw * $dh;
        if ($byMode[$m] == null) { $byMode[$m] = 0; }
        $byMode[$m] = $byMode[$m] + 1;
        if ($shown < 8) {
            $shown = $shown + 1;
            print("     DIFF " + $tag + " " + $fname + ": " + str($diff) +
                  " of " + str(len($wb)) + " bytes");
        }
    }
    $i = $i + 1;
}

print("   exact: " + str($stat.exact) + "   differing: " + str($stat.wrong) +
      "   errored: " + str($stat.err));
if ($stat.wrong > 0) {
    print("   differing byte count: " + str($stat.badpx));
    for $k, $v in $byMode.items() { print("     mode " + $k + ": " + str($v) + " jobs"); }
}
eq($stat.err, 0, "no job raised");
eq($stat.wrong, 0, "every one of " + str($stat.exact) + " resize jobs is byte-identical to Pillow");

// ─── the behaviours Pillow's Python layer adds ──────────────────────────
print("── the Python-layer behaviours ──");

// modes 1 and P force NEAREST, so every filter gives the same answer
$p1 = pillow.new("1", [8, 8], 0);
$k = 0;
while ($k < 8) { $p1.putpixel([$k, $k], 255); $k = $k + 1; }
$a = $p1.resize([4, 4], resample.NEAREST, null, null).tobytes();
$b = $p1.resize([4, 4], resample.BICUBIC, null, null).tobytes();
$c = $p1.resize([4, 4], resample.LANCZOS, null, null).tobytes();
eq($a == $b && $b == $c, true, "mode '1' forces NEAREST, so all filters agree");
$vals = $p1.resize([4, 4], resample.BICUBIC, null, null).getdata();
$onlyBilevel = true;
$k = 0;
while ($k < len($vals)) {
    if ($vals[$k] != 0 && $vals[$k] != 255) { $onlyBilevel = false; }
    $k = $k + 1;
}
ok($onlyBilevel, "and the result holds only 0 and 255, as mode '1' must");

$pp = pillow.new("P", [8, 8], 0);
// NEAREST samples the pixel CENTRE: for a 2x reduction output 0 reads
// floor(2 * 0.5) = source index 1, not 0.
$pp.putpixel([1, 1], 200);
$a = $pp.resize([4, 4], resample.BILINEAR, null, null);
eq($a.mode, "P", "a P resize stays P");
eq($a.getpixel([0, 0]), 200, "and samples an index rather than blending it");
eq($a.getpixel([1, 1]), 0, "a neighbour is untouched -- no index was averaged");
// an index that would be meaningless if blended survives intact
$pp2 = pillow.new("P", [4, 4], 0);
$pp2.putpixel([1, 1], 3);
$pp2.putpixel([3, 3], 200);
$r2 = $pp2.resize([2, 2], resample.LANCZOS, null, null).getdata();
$onlyKnown = true;
$k = 0;
while ($k < len($r2)) {
    if ($r2[$k] != 0 && $r2[$k] != 3 && $r2[$k] != 200) { $onlyKnown = false; }
    $k = $k + 1;
}
ok($onlyKnown, "every output index came from the source, none was invented (" + str($r2) + ")");

// RGBA is premultiplied, so a transparent neighbour does not bleed colour.
// Half the image is opaque white, half is TRANSPARENT red; downscaling must
// not drag red into the white.
$rgba = pillow.new("RGBA", [4, 2], [255, 255, 255, 255]);
$x = 2;
while ($x < 4) {
    $y = 0;
    while ($y < 2) { $rgba.putpixel([$x, $y], [255, 0, 0, 0]); $y = $y + 1; }
    $x = $x + 1;
}
$small = $rgba.resize([2, 1], resample.BILINEAR, null, null);
$left = $small.getpixel([0, 0]);
ok($left[1] > 200 && $left[2] > 200,
   "a transparent red neighbour does not tint the opaque white (" + str($left) + ")");

// ─── degenerate and hostile input ───────────────────────────────────────
print("── degenerate and hostile input ──");

$src = pillow.new("RGB", [8, 6], [10, 20, 30]);
eq($src.resize([1, 1], resample.BOX, null, null).size, [1, 1], "down to a single pixel");
eq($src.resize([1, 1], resample.BOX, null, null).getpixel([0, 0]), [10, 20, 30],
   "and a flat image keeps its colour");
eq($src.resize([8, 6], resample.BICUBIC, null, null).getdata(), $src.getdata(),
   "an identity resize is a copy");
eq($src.resize([200, 1], resample.LANCZOS, null, null).size, [200, 1], "a 200x1 strip");
eq($src.resize([1, 200], resample.LANCZOS, null, null).size, [1, 200], "a 1x200 strip");

// a box selects a sub-region
$box = $src.resize([4, 3], resample.BOX, [0, 0, 4, 3], null);
eq($box.size, [4, 3], "a box whose size matches is a crop");
eq($box.getpixel([0, 0]), [10, 20, 30], "with the right content");

throws(def() { return $src.resize([0, 5], resample.BOX, null, null); }, "a zero width");
throws(def() { return $src.resize([5, 0], resample.BOX, null, null); }, "a zero height");
throws(def() { return $src.resize([-1, 5], resample.BOX, null, null); }, "a negative width");
throws(def() { return $src.resize([4, 4], 99, null, null); }, "an unknown filter");
throws(def() { return $src.resize(5, resample.BOX, null, null); }, "a size that is not a pair");
throws(def() { return $src.resize([4, 4], resample.BOX, [-1, 0, 4, 3], null); }, "a negative box offset");
throws(def() { return $src.resize([4, 4], resample.BOX, [0, 0, 99, 3], null); }, "a box wider than the image");
throws(def() { return $src.resize([4, 4], resample.BOX, [3, 0, 1, 3], null); }, "a box with negative width");
throws(def() { return $src.resize([4, 4], resample.BOX, null, 2.0); }, "reducing_gap says it is unimplemented");
throws(def() { return $src.resize([100000, 100000], resample.BOX, null, null); },
       "a resize past MAX_IMAGE_PIXELS is refused before allocation");

// the source is never modified
$orig = $src.getdata();
$src.resize([3, 3], resample.LANCZOS, null, null);
eq($src.getdata(), $orig, "resize does not touch its source");

// ─── rule PW2: one array operation per tap, not per pixel ───────────────
print("── rule PW2: the cost is linear in pixel count ──");

def timeResize($n, $f) {
    $im = pillow.new("RGB", [$n, $n], [40, 80, 120]);
    $t0 = clock();
    $im.resize([floor($n / 2), floor($n / 2)], $f, null, null);
    return clock() - $t0;
}
$t1 = timeResize(200, resample.BICUBIC);
$t2 = timeResize(400, resample.BICUBIC);
print("   bicubic 40,000 px -> " + str($t1) + " ms      160,000 px -> " + str($t2) + " ms");
ok($t2 <= max($t1, 1) * 12 + 120, "4x the pixels costs well under 12x the time");

// nearest is two gathers, so it must be dramatically cheaper than bicubic
$tn = timeResize(400, resample.NEAREST);
print("   nearest 160,000 px -> " + str($tn) + " ms");
ok($tn <= max($t2, 1), "nearest is no slower than bicubic");

// a large upscale must stay proportional to the OUTPUT, not to taps x pixels
$big = pillow.new("RGB", [40, 40], [1, 2, 3]);
$t0 = clock();
$up = $big.resize([800, 800], resample.BILINEAR, null, null);
$tu = clock() - $t0;
print("   1,600 px -> 640,000 px bilinear: " + str($tu) + " ms");
eq($up.size, [800, 800], "the upscale produced the right size");
ok($tu < 4000, "a 640,000-pixel upscale is seconds, not minutes");

print("");
print("checks: " + str($R.p + $R.f) + "  passed: " + str($R.p) + "  failed: " + str($R.f));
if ($R.f == 0) { print("RESULT: ALL GREEN"); }
else { print("RESULT: FAILED"); }
