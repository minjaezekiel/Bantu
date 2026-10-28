// ════════════════════════════════════════════════════════════════════════
//  tests/pillow_geometry_test.b — pillow Phase P3: the geometry surface.
//
//  261 reference images from tests/fixtures/pillow/reference/geometry/:
//  transpose (7 operations x 7 modes), reduce (105 factor/size pairs,
//  including factors that do not divide the size), thumbnail (30 aspect
//  cases), transform (AFFINE, EXTENT, PERSPECTIVE, QUAD x 3 filters x 3
//  modes) and rotate (8 angles x expand on/off x 2 modes).
//
//  Whole images, not probe pixels: these are per-pixel maps, so one pixel
//  says nothing about its neighbours.
//
//  Run:  bantu run tests/pillow_geometry_test.b
// ════════════════════════════════════════════════════════════════════════

include "./pillow/pillow.b" as pillow;
include "./pillow/resample.b" as resample;
include "./pillow/transform.b" as xform;
include "./pillow/path.b" as path;
include "./pillow/mode.b" as mode;
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

$D = "tests/fixtures/pillow/reference/geometry/";

def _jobs() {
    $rows = split(readfile("tests/fixtures/pillow/reference/geometry_index.txt"), "\n");
    $out = [];
    $i = 0;
    while ($i < len($rows)) {
        $L = trim($rows[$i]);
        if (len($L) > 0 && $L[0] != "#") { push($out, split($L, " ")); }
        $i = $i + 1;
    }
    return $out;
}
$JOBS = _jobs();
eq(len($JOBS) > 250, true, "the geometry corpus has over 250 jobs");

$stat = {"ok": 0, "bad": 0, "err": 0};
$shown = {"n": 0};
def check($gotIm, $wantFile, $label, $wantW, $wantH) {
    if ($gotIm.width != $wantW || $gotIm.height != $wantH) {
        $stat.bad = $stat.bad + 1;
        if ($shown.n < 8) {
            $shown.n = $shown.n + 1;
            print("     SIZE " + $label + ": got " + str($gotIm.size) +
                  " want [" + str($wantW) + ", " + str($wantH) + "]");
        }
        return false;
    }
    $want = readfile($D + $wantFile);
    if ($gotIm.tobytes() == $want) { $stat.ok = $stat.ok + 1; return true; }
    $gb = bytes($gotIm.tobytes());
    $wb = bytes($want);
    $diff = 0;
    $k = 0;
    while ($k < len($wb) && $k < len($gb)) {
        if ($gb[$k] != $wb[$k]) { $diff = $diff + 1; }
        $k = $k + 1;
    }
    $stat.bad = $stat.bad + 1;
    if ($shown.n < 8) {
        $shown.n = $shown.n + 1;
        print("     DIFF " + $label + ": " + str($diff) + " of " + str(len($wb)) + " bytes");
    }
    return false;
}

// ─── transpose ──────────────────────────────────────────────────────────
print("── transpose: 7 operations x 7 modes ──");

$OPNAMES = ["FLIP_LEFT_RIGHT", "FLIP_TOP_BOTTOM", "ROTATE_90", "ROTATE_180",
            "ROTATE_270", "TRANSPOSE", "TRANSVERSE"];
$OPIDS = [0, 1, 2, 3, 4, 5, 6];

$i = 0;
while ($i < len($JOBS)) {
    $j = $JOBS[$i];
    if ($j[0] == "transpose") {
        $m = $j[1];
        $w = num($j[2]);
        $h = num($j[3]);
        $op = $j[4];
        $ow = num($j[5]);
        $oh = num($j[6]);
        $tag = "T_" + $m + "_" + str($w) + "x" + str($h);
        $src = pillow.frombytes($m, [$w, $h], readfile($D + $tag + ".src"));
        $id = $OPIDS[indexOf($OPNAMES, $op)];
        $threw = false;
        $g = null;
        try { $g = $src.transpose($id); } catch ($e) { $threw = true; }
        if ($threw) { $stat.err = $stat.err + 1; print("     ERROR " + $tag + " " + $op); }
        else { check($g, $tag + "_" + $op + ".dst", $tag + " " + $op, $ow, $oh); }
    }
    $i = $i + 1;
}
print("   " + str($stat.ok) + " exact, " + str($stat.bad) + " differing, " + str($stat.err) + " errored");
eq($stat.bad + $stat.err, 0, "all 49 transpose results are byte-identical to Pillow");

// which way round the rotations go, named so a swap cannot pass silently
$m = pillow.new("L", [3, 2], 0);
$m.putpixel([0, 0], 255);
eq($m.transpose(xform.ROTATE_90).size, [2, 3], "ROTATE_90 swaps the axes");
// anticlockwise: the top-left pixel goes to the BOTTOM-left
eq($m.transpose(xform.ROTATE_90).getpixel([0, 2]), 255, "ROTATE_90 is anticlockwise");
eq($m.transpose(xform.ROTATE_270).getpixel([1, 0]), 255, "ROTATE_270 is clockwise");
eq($m.transpose(xform.TRANSPOSE).getpixel([0, 0]), 255, "TRANSPOSE keeps the main diagonal");
eq($m.transpose(xform.TRANSVERSE).getpixel([1, 2]), 255, "TRANSVERSE is the anti-diagonal");
// the involutions really are involutions
$i = 0;
$bad = 0;
while ($i < 7) {
    $op = $OPIDS[$i];
    if ($op == 0 || $op == 1 || $op == 3 || $op == 5 || $op == 6) {
        $sq = pillow.new("L", [4, 4], 0);
        $sq.putpixel([1, 0], 90);
        $sq.putpixel([3, 2], 200);
        if ($sq.transpose($op).transpose($op).getdata() != $sq.getdata()) { $bad = $bad + 1; }
    }
    $i = $i + 1;
}
eq($bad, 0, "the five self-inverse operations are involutions");
// and the two rotations compose
$sq = pillow.new("L", [4, 4], 0);
$sq.putpixel([1, 0], 90);
eq($sq.transpose(xform.ROTATE_90).transpose(xform.ROTATE_270).getdata(), $sq.getdata(),
   "ROTATE_90 then ROTATE_270 is the identity");
throws(def() { return $sq.transpose(9); }, "an unknown transpose operation");

// ─── reduce ─────────────────────────────────────────────────────────────
print("── reduce: 105 factor and size pairs ──");
$stat.ok = 0; $stat.bad = 0; $stat.err = 0; $shown.n = 0;
// Counted per mode, because mode F has a known and bounded residual while
// every other mode is exact. See transform.b's _reduceBlock32 header: Reduce.c
// dispatches the common factors to eleven hand-unrolled kernels whose FLOAT
// addition order differs from the general one. Integer modes are unaffected
// (addition is associative); mode F moves the last float32 step on a small
// fraction of pixels.
$redMode = {};
$redBad = {};
$i = 0;
while ($i < len($JOBS)) {
    $j = $JOBS[$i];
    if ($j[0] == "reduce") {
        $m = $j[1];
        $w = num($j[2]);  $h = num($j[3]);
        $fx = num($j[4]); $fy = num($j[5]);
        $ow = num($j[6]); $oh = num($j[7]);
        $tag = "R_" + $m + "_" + str($w) + "x" + str($h);
        $src = pillow.frombytes($m, [$w, $h], readfile($D + $tag + ".src"));
        if ($redMode[$m] == null) { $redMode[$m] = 0; $redBad[$m] = 0; }
        $redMode[$m] = $redMode[$m] + 1;
        $threw = false;
        $g = null;
        try { $g = $src.reduce([$fx, $fy], null); } catch ($e) { $threw = true; }
        if ($threw) {
            $stat.err = $stat.err + 1;
            $redBad[$m] = $redBad[$m] + 1;
            if ($shown.n < 6) { $shown.n = $shown.n + 1; print("     ERROR " + $tag + " " + str($fx) + "x" + str($fy)); }
        } else {
            if (!check($g, $tag + "_" + str($fx) + "x" + str($fy) + ".dst",
                       $tag + " /" + str($fx) + "x" + str($fy), $ow, $oh)) {
                $redBad[$m] = $redBad[$m] + 1;
            }
        }
    }
    $i = $i + 1;
}
print("   " + str($stat.ok) + " exact, " + str($stat.bad) + " differing, " + str($stat.err) + " errored");
for $k, $v in $redMode.items() { print("     mode " + $k + ": " + str($v - $redBad[$k]) + "/" + str($v) + " exact"); }
eq($stat.err, 0, "no reduce job raised");
// Every integer mode must be byte-identical; no exceptions.
$intBad = 0;
for $k, $v in $redBad.items() { if ($k != "F") { $intBad = $intBad + $v; } }
eq($intBad, 0, "reduce is byte-identical to Pillow for every INTEGER mode");
// Mode F is allowed a bounded residual, and the bound is asserted so a real
// regression cannot hide behind it.
ok($redBad["F"] <= 12, "mode F differs on at most 12 of " + str($redMode["F"]) +
   " jobs (got " + str($redBad["F"]) + "), the documented float-order residual");
// and when it differs, it differs by ONE float32 step, not by a visible amount
$fsrc = pillow.frombytes("F", [12, 8], readfile($D + "R_F_12x8.src"));
$fgot = $fsrc.reduce([3, 3], null);
$fwant = nd_frombytes(readfile($D + "R_F_12x8_3x3.dst"), "f32le",
                      [$fgot.height, $fgot.width, 1]);
$rel = nd_max(nd_abs(nd_divide(nd_subtract($fgot.px, $fwant),
                               nd_maximum(nd_abs($fwant), 1.0))));
ok(pxbuf.scalar($rel) < 0.0000005,
   "and the largest relative difference is under one float32 ULP (" +
   str(pxbuf.scalar($rel)) + ")");

// the output size rounds UP, so a partial block gets its own cell
eq(pillow.new("L", [13, 7], 0).reduce(3, null).size, [5, 3], "reduce rounds the size up");
eq(pillow.new("L", [12, 6], 0).reduce(3, null).size, [4, 2], "an exact division does not");
eq(pillow.new("L", [4, 4], 7).reduce(1, null).getdata(), pillow.new("L", [4, 4], 7).getdata(),
   "a factor of 1 is a copy");
throws(def() { return pillow.new("L", [4, 4], 0).reduce(0, null); }, "a factor of 0");
throws(def() { return pillow.new("L", [4, 4], 0).reduce(-2, null); }, "a negative factor");
throws(def() { return pillow.new("L", [4, 4], 0).reduce([2], null); }, "a one-element factor list");
throws(def() { return pillow.new("L", [4, 4], 0).reduce(2, [0, 0, 9, 9]); }, "a box outside the image");

// ─── thumbnail ──────────────────────────────────────────────────────────
print("── thumbnail: 30 aspect cases ──");
$stat.ok = 0; $stat.bad = 0; $stat.err = 0; $shown.n = 0;
$i = 0;
while ($i < len($JOBS)) {
    $j = $JOBS[$i];
    if ($j[0] == "thumbnail") {
        $w = num($j[2]);  $h = num($j[3]);
        $tw = num($j[4]); $th = num($j[5]);
        $ow = num($j[6]); $oh = num($j[7]);
        $tag = "TH_RGB_" + str($w) + "x" + str($h);
        $src = pillow.frombytes("RGB", [$w, $h], readfile($D + $tag + ".src"));
        $threw = false;
        try { $src.thumbnail([$tw, $th], null, null); } catch ($e) { $threw = true; }
        if ($threw) {
            $stat.err = $stat.err + 1;
            if ($shown.n < 6) { $shown.n = $shown.n + 1; print("     ERROR " + $tag + " -> " + str($tw) + "x" + str($th)); }
        } else {
            check($src, $tag + "_" + str($tw) + "x" + str($th) + ".dst",
                  $tag + " -> " + str($tw) + "x" + str($th), $ow, $oh);
        }
    }
    $i = $i + 1;
}
print("   " + str($stat.ok) + " exact, " + str($stat.bad) + " differing, " + str($stat.err) + " errored");
eq($stat.bad + $stat.err, 0, "all 30 thumbnail results are byte-identical to Pillow");

// thumbnail only ever SHRINKS, and mutates in place
$big = pillow.new("RGB", [10, 10], [1, 2, 3]);
$big.thumbnail([200, 200], null, null);
eq($big.size, [10, 10], "a thumbnail larger than the image does nothing");
$sh = pillow.new("RGB", [40, 20], [1, 2, 3]);
$sh.thumbnail([10, 10], null, null);
eq($sh.size, [10, 5], "and the aspect ratio is preserved");
ok($sh.getpixel([0, 0])[0] == 1, "with the content resampled, not cleared");
throws(def() { return pillow.new("RGB", [4, 4], null).thumbnail([0, 4], null, null); }, "a zero thumbnail size");
throws(def() { return pillow.new("RGB", [4, 4], null).thumbnail(5, null, null); }, "a size that is not a pair");

// ─── transform ──────────────────────────────────────────────────────────
print("── transform: AFFINE, EXTENT, PERSPECTIVE, QUAD x 3 filters ──");
$stat.ok = 0; $stat.bad = 0; $stat.err = 0; $shown.n = 0;

$CASEDATA = {
    "AFFINE":      [1, 0, 0, 0, 1, 0],
    "AFFINE2":     [0.7, 0.2, 1.5, -0.3, 1.1, -0.5],
    "EXTENT":      [1, 1, 9, 6],
    "PERSPECTIVE": [1.0, 0.1, 0.0, 0.05, 1.0, 0.0, 0.002, 0.001],
    "QUAD":        [0, 0, 0, 7, 11, 6, 10, 1]
};
$CASEMETH = {"AFFINE": 0, "AFFINE2": 0, "EXTENT": 1, "PERSPECTIVE": 2, "QUAD": 3};

$i = 0;
while ($i < len($JOBS)) {
    $j = $JOBS[$i];
    if ($j[0] == "transform") {
        $m = $j[1];
        $w = num($j[2]); $h = num($j[3]);
        $cname = $j[4];
        $fname = $j[5];
        $ow = num($j[6]); $oh = num($j[7]);
        $tag = "X_" + $m + "_" + str($w) + "x" + str($h);
        $src = pillow.frombytes($m, [$w, $h], readfile($D + $tag + ".src"));
        $fid = resample.filter_id($fname);
        $threw = false;
        $g = null;
        try { $g = $src.transform([$ow, $oh], $CASEMETH[$cname], $CASEDATA[$cname], $fid, null); }
        catch ($e) { $threw = true; }
        if ($threw) {
            $stat.err = $stat.err + 1;
            if ($shown.n < 6) { $shown.n = $shown.n + 1; print("     ERROR " + $tag + " " + $cname + " " + $fname); }
        } else {
            check($g, $tag + "_" + $cname + "_" + $fname + ".dst",
                  $tag + " " + $cname + " " + $fname, $ow, $oh);
        }
    }
    $i = $i + 1;
}
print("   " + str($stat.ok) + " exact, " + str($stat.bad) + " differing, " + str($stat.err) + " errored");
eq($stat.bad + $stat.err, 0, "all 45 transform results are byte-identical to Pillow");

// an identity affine is the top-left crop of the source
$src = pillow.new("RGB", [6, 4], null);
$src.putpixel([0, 0], [9, 8, 7]);
$id = $src.transform([6, 4], xform.AFFINE, [1, 0, 0, 0, 1, 0], resample.NEAREST, null);
eq($id.getdata(), $src.getdata(), "an identity AFFINE reproduces the source");
// a coordinate outside the source takes the fill, not a clamped edge pixel
$off = $src.transform([4, 4], xform.AFFINE, [1, 0, 100, 0, 1, 100], resample.NEAREST, "red");
eq($off.getpixel([0, 0]), [255, 0, 0], "an out-of-range source coordinate takes the fill colour");
$off0 = $src.transform([4, 4], xform.AFFINE, [1, 0, 100, 0, 1, 100], resample.NEAREST, null);
eq($off0.getpixel([0, 0]), [0, 0, 0], "and zero when no fill is given");
// modes 1 and P degrade to NEAREST, since an index cannot be interpolated
$pp = pillow.new("P", [6, 4], 0);
$pp.putpixel([1, 1], 200);
$pt = $pp.transform([6, 4], xform.AFFINE, [1, 0, 0, 0, 1, 0], resample.BICUBIC, null);
eq($pt.getpixel([1, 1]), 200, "mode P transform degrades to NEAREST");
throws(def() { return $src.transform([4, 4], 9, [1, 0, 0, 0, 1, 0], resample.NEAREST, null); },
       "an unknown transform method");
throws(def() { return $src.transform([4, 4], xform.AFFINE, [1, 0, 0], resample.NEAREST, null); },
       "too few AFFINE coefficients");
throws(def() { return $src.transform([4, 4], xform.PERSPECTIVE, [1, 0, 0], resample.NEAREST, null); },
       "too few PERSPECTIVE coefficients");
throws(def() { return $src.transform([0, 4], xform.AFFINE, [1, 0, 0, 0, 1, 0], resample.NEAREST, null); },
       "a zero output width");
throws(def() { return $src.transform([4, 4], xform.AFFINE, [1, 0, 0, 0, 1, 0], resample.LANCZOS, null); },
       "LANCZOS is not a transform filter");

// MESH: two boxes, each from its own quad
$mesh = $src.transform([6, 4], xform.MESH,
    [[[0, 0, 3, 4], [0, 0, 0, 4, 3, 4, 3, 0]],
     [[3, 0, 6, 4], [3, 0, 3, 4, 6, 4, 6, 0]]], resample.NEAREST, null);
eq($mesh.size, [6, 4], "a two-box MESH produces the requested size");
eq($mesh.getdata(), $src.getdata(), "and an identity mesh reproduces the source");
throws(def() { return $src.transform([4, 4], xform.MESH, [[1, 2]], resample.NEAREST, null); },
       "a MESH entry that is not [box, quad]");

// the transform classes carry the method and the data
eq(xform.AffineTransform([1, 0, 0, 0, 1, 0]).method, xform.AFFINE, "AffineTransform");
eq(xform.ExtentTransform([0, 0, 2, 2]).method, xform.EXTENT, "ExtentTransform");
eq(xform.PerspectiveTransform([1, 0, 0, 0, 1, 0, 0, 0]).method, xform.PERSPECTIVE, "PerspectiveTransform");
eq(xform.QuadTransform([0, 0, 0, 1, 1, 1, 1, 0]).method, xform.QUAD, "QuadTransform");
eq(xform.MeshTransform([]).method, xform.MESH, "MeshTransform");
eq(xform.method_name(0), "AFFINE", "method_name");

// ─── rotate ─────────────────────────────────────────────────────────────
print("── rotate: 8 angles x expand on/off ──");
$stat.ok = 0; $stat.bad = 0; $stat.err = 0; $shown.n = 0;
$i = 0;
while ($i < len($JOBS)) {
    $j = $JOBS[$i];
    if ($j[0] == "rotate") {
        $m = $j[1];
        $w = num($j[2]); $h = num($j[3]);
        $ang = num($j[4]);
        $exp = num($j[5]);
        $ow = num($j[6]); $oh = num($j[7]);
        $tag = "RO_" + $m + "_" + str($w) + "x" + str($h);
        $src = pillow.frombytes($m, [$w, $h], readfile($D + $tag + ".src"));
        $threw = false;
        $g = null;
        try { $g = $src.rotate($ang, resample.NEAREST, $exp == 1, null, null, null); }
        catch ($e) { $threw = true; }
        if ($threw) {
            $stat.err = $stat.err + 1;
            if ($shown.n < 6) { $shown.n = $shown.n + 1; print("     ERROR " + $tag + " " + str($ang) + " exp=" + str($exp)); }
        } else {
            check($g, $tag + "_" + str($ang) + "_" + str($exp) + ".dst",
                  $tag + " " + str($ang) + " exp=" + str($exp), $ow, $oh);
        }
    }
    $i = $i + 1;
}
print("   " + str($stat.ok) + " exact, " + str($stat.bad) + " differing, " + str($stat.err) + " errored");
eq($stat.bad + $stat.err, 0, "all 32 rotate results are byte-identical to Pillow");

// the right-angle fast paths
$r = pillow.new("L", [5, 3], 0);
$r.putpixel([0, 0], 200);
eq($r.rotate(0, null, false, null, null, null).getdata(), $r.getdata(), "rotate 0 is a copy");
eq($r.rotate(180, null, false, null, null, null).getpixel([4, 2]), 200, "rotate 180 uses transpose");
eq($r.rotate(90, null, true, null, null, null).size, [3, 5], "rotate 90 with expand swaps the axes");
// a non-square rotate 90 WITHOUT expand must stay the same size
eq($r.rotate(90, null, false, null, null, null).size, [5, 3], "rotate 90 without expand keeps the size");
// 360 is 0
eq($r.rotate(360, null, false, null, null, null).getdata(), $r.getdata(), "rotate 360 is a copy");
eq($r.rotate(-90, null, true, null, null, null).size, [3, 5], "a negative angle normalises");

// ─── ImagePath ──────────────────────────────────────────────────────────
print("── ImagePath ──");

$p = path.Path([0, 0, 10, 0, 10, 10]);
eq(path.len_points($p), 3, "three points from a flat list");
eq(path.getbbox($p), [0, 0, 10, 10], "getbbox");
eq(path.tolist($p, null), [[0, 0], [10, 0], [10, 10]], "tolist gives pairs");
eq(path.tolist($p, true), [0, 0, 10, 0, 10, 10], "tolist flat");
$p2 = path.Path([[1, 2], [3, 4]]);
eq(path.tolist($p2, true), [1, 2, 3, 4], "a list of pairs works too");
eq(path.getbbox(path.Path([])), null, "an empty path has no bbox");
// transform
$p3 = path.Path([1, 1, 2, 2]);
path.transform($p3, [2, 0, 1, 0, 3, 5]);
eq(path.tolist($p3, true), [3, 8, 5, 11], "transform applies the affine");
// map
$p4 = path.Path([1, 2]);
path.map_points($p4, def($x, $y) { return [$y, $x]; });
eq(path.tolist($p4, true), [2, 1], "map_points swaps");
// compact uses the CITY-BLOCK distance
$p5 = path.Path([0, 0, 1, 0, 10, 0, 11, 0, 20, 0]);
$removed = path.compact($p5, 5);
eq($removed, 2, "compact drops the two near points");
eq(path.tolist($p5, true), [0, 0, 10, 0, 20, 0], "and keeps the endpoints");
throws(def() { return path.Path([1, 2, 3]); }, "an odd-length flat list");
throws(def() { return path.Path(5); }, "a path from a non-list");
throws(def() { return path.transform(path.Path([0, 0]), [1, 0]); }, "too few affine coefficients");

// ─── rule PW2 ───────────────────────────────────────────────────────────
print("── rule PW2: a fixed number of array operations per filter ──");

def timeX($n, $f) {
    $im = pillow.new("RGB", [$n, $n], [40, 80, 120]);
    $t0 = clock();
    $im.transform([$n, $n], xform.AFFINE, [0.9, 0.1, 1, -0.1, 0.9, 1], $f, null);
    return clock() - $t0;
}
$t1 = timeX(200, resample.BILINEAR);
$t2 = timeX(400, resample.BILINEAR);
print("   bilinear transform 40,000 px -> " + str($t1) + " ms   160,000 px -> " + str($t2) + " ms");
ok($t2 <= max($t1, 1) * 12 + 150, "4x the pixels costs well under 12x the time");

$tn = timeX(400, resample.NEAREST);
$tb = timeX(400, resample.BICUBIC);
print("   160,000 px: nearest " + str($tn) + " ms, bilinear " + str($t2) +
      " ms, bicubic " + str($tb) + " ms");
ok($tn <= $t2, "nearest (1 gather) is no slower than bilinear (4)");
ok($tb < max($tn, 1) * 40 + 400, "bicubic (16 gathers) is a small multiple of nearest, not per-pixel");

// transpose and reduce on a large image must stay cheap
$big = pillow.new("RGB", [600, 400], [1, 2, 3]);
$t0 = clock();
$big.transpose(xform.ROTATE_90);
$tt = clock() - $t0;
print("   240,000-pixel ROTATE_90: " + str($tt) + " ms");
ok($tt < 500, "transpose is an axis permutation, not a pixel loop");
$t0 = clock();
$big.reduce(4, null);
$tr = clock() - $t0;
print("   240,000-pixel reduce /4:  " + str($tr) + " ms");
ok($tr < 800, "reduce is two reductions, not a pixel loop");

print("");
print("checks: " + str($R.p + $R.f) + "  passed: " + str($R.p) + "  failed: " + str($R.f));
if ($R.f == 0) { print("RESULT: ALL GREEN"); }
else { print("RESULT: FAILED"); }
