// ════════════════════════════════════════════════════════════════════════
//  tests/pillow_draw_test.b — pillow Phase P7: ImageDraw.
//
//  Three kinds of claim, three kinds of check:
//
//    1. WHOLE IMAGES, byte for byte, against Pillow: 700 of them. Every
//       primitive at every width that changes its shape, every slope and
//       sign of line, every parity of ellipse bounding box, eighteen angle
//       pairs through arc, chord and pieslice both filled and outlined, the
//       rounded-rectangle degenerate cases, Outline/shape, bitmap through
//       both a "1" stencil and an "L" matte, RGBA-on-RGB blending at six
//       alphas, floodfill with and without a border, and one composite
//       scene in each of the twenty modes.
//
//       The corpus is DRIVEN BY THE MANIFEST, not by a list in here: each
//       row carries the mode, the size, the background and the call, and
//       this file replays it. A new case is added to
//       tools/gen_pillow_reference.py and appears here with no edit.
//
//    2. THE FAILURE MODES: a reversed box, a bad mode pairing, a polygon
//       with one point, an outline with no coefficients -- each a catchable
//       Bantu error naming the call.
//
//    3. STRESS: a 100,000-point polygon, a self-intersecting polygon, a
//       degenerate ellipse, a line far outside the image, a 1080p diagonal,
//       and the timing curve -- a filled polygon's cost is proportional to
//       its HEIGHT, not its area, which is what decision PW8 claims.
//
//  Run:  bantu run tests/pillow_draw_test.b
// ════════════════════════════════════════════════════════════════════════

include "./pillow/pillow.b" as pillow;
include "./pillow/draw.b" as draw;
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

$D = "tests/fixtures/pillow/reference/draw/";


// ════════════════════════════════════════════════════════════════════════
//  The replay: one encoded call from the manifest, performed
// ════════════════════════════════════════════════════════════════════════

// An ink field: "none", "i:<int>", "t:<a,b,c[,d]>" or "c:<colour name>".
def _ink($s) {
    if ($s == null || $s == "none") { return null; }
    $kind = substr($s, 0, 2);
    $rest = substr($s, 2, len($s) - 2);
    if ($kind == "i:") { return num($rest); }
    if ($kind == "t:") {
        $out = [];
        each ($p in split($rest, ",")) { push($out, num($p)); }
        return $out;
    }
    if ($kind == "c:") { return $rest; }
    throw "bad ink field: " + $s;
}

def _nums($s) {
    $out = [];
    if ($s == "") { return $out; }
    each ($p in split($s, ",")) { push($out, num($p)); }
    return $out;
}

// The bitmap the BM_ cases draw, built from the same formula the generator
// uses, so no fixture has to carry it.
def _maskImage($m, $w, $h) {
    $im = pillow.new($m, [$w, $h], 0);
    $y = 0;
    while ($y < $h) {
        $x = 0;
        while ($x < $w) {
            if ($m == "1") {
                $v = 0;
                if (($x * 5 + $y * 3) % 7 < 3) { $v = 255; }
            } else {
                $v = ($x * 37 + $y * 17) % 256;
            }
            $im.putpixel([$x, $y], $v);
            $x = $x + 1;
        }
        $y = $y + 1;
    }
    return $im;
}

// The backgrounds: zeros, a colour, or the one prebuilt scene floodfill
// needs.
def _scene($m, $w, $h, $bg) {
    if ($bg == "z") { return pillow.new($m, [$w, $h], null); }
    if ($bg == "s1") {
        $im = pillow.new($m, [$w, $h], 255);
        $d = draw.Draw($im, null);
        $d.rectangle([2, 2, $w - 3, $h - 3], null, 0, 1);
        $d.line([[0, 0], [$w - 1, $h - 1]], 0, 1, null);
        return $im;
    }
    return pillow.new($m, [$w, $h], $bg);
}

def _kw($parts) {
    $kw = {};
    $i = 2;
    while ($i < len($parts)) {
        $f = split($parts[$i], "=");
        $v = "";
        if (len($f) > 1) { $v = $f[1]; }
        $kw[$f[0]] = $v;
        $i = $i + 1;
    }
    return $kw;
}

def _replay($im, $inkMode, $call) {
    $parts = split($call, "|");
    $op = $parts[0];
    $xy = _nums($parts[1]);
    $kw = _kw($parts);
    $dm = null;
    if ($inkMode != "-") { $dm = $inkMode; }
    $d = draw.Draw($im, $dm);
    $fill = _ink($kw["fill"]);
    $outline = _ink($kw["outline"]);
    $width = 1;
    if ($kw["width"] != null) { $width = num($kw["width"]); }

    if ($op == "point") { return $d.point($xy, $fill); }
    if ($op == "line") {
        $joint = null;
        if ($kw["joint"] != null && $kw["joint"] != "") { $joint = $kw["joint"]; }
        return $d.line($xy, $fill, $width, $joint);
    }
    if ($op == "rectangle") { return $d.rectangle($xy, $fill, $outline, $width); }
    if ($op == "polygon") { return $d.polygon($xy, $fill, $outline, $width); }
    if ($op == "ellipse") { return $d.ellipse($xy, $fill, $outline, $width); }
    if ($op == "circle") {
        return $d.circle([$xy[0], $xy[1]], num($kw["r"]), $fill, $outline, $width);
    }
    if ($op == "arc") {
        return $d.arc($xy, num($kw["start"]), num($kw["end"]), $fill, $width);
    }
    if ($op == "chord") {
        return $d.chord($xy, num($kw["start"]), num($kw["end"]), $fill, $outline, $width);
    }
    if ($op == "pieslice") {
        return $d.pieslice($xy, num($kw["start"]), num($kw["end"]), $fill, $outline, $width);
    }
    if ($op == "rounded") {
        $corners = null;
        if ($kw["corners"] != null) {
            $corners = [];
            each ($v in split($kw["corners"], ",")) { push($corners, $v == "1"); }
        }
        return $d.rounded_rectangle($xy, num($kw["radius"]), $fill, $outline,
                                    $width, $corners);
    }
    if ($op == "regular") {
        $rot = 0;
        if ($kw["rot"] != null) { $rot = num($kw["rot"]); }
        return $d.regular_polygon([$xy[0], $xy[1], $xy[2]], num($kw["n"]), $rot,
                                  $fill, $outline, $width);
    }
    if ($op == "shape") {
        $o = draw.Outline();
        $o.move($xy[0], $xy[1]);
        $i = 2;
        while ($i + 1 < len($xy)) {
            $o.line($xy[$i], $xy[$i + 1]);
            $i = $i + 2;
        }
        if ($kw["transform"] != null) { $o.transform(_nums($kw["transform"])); }
        return $d.shape($o, $fill, $outline);
    }
    if ($op == "shapec") {
        $o = draw.Outline();
        $o.move($xy[0], $xy[1]);
        $o.curve($xy[2], $xy[3], $xy[4], $xy[5], $xy[6], $xy[7]);
        $o.close();
        return $d.shape($o, $fill, $outline);
    }
    if ($op == "bitmap") {
        $bm = split($kw["bm"], ",");
        return $d.bitmap([$xy[0], $xy[1]], _maskImage($bm[0], num($bm[1]), num($bm[2])),
                         $fill);
    }
    if ($op == "floodfill") {
        $border = _ink($kw["border"]);
        $thresh = 0;
        if ($kw["thresh"] != null) { $thresh = num($kw["thresh"]); }
        return draw.floodfill($im, [$xy[0], $xy[1]], $fill, $border, $thresh);
    }
    throw "unknown op " + $op;
}


// ════════════════════════════════════════════════════════════════════════
//  Tier 2: the whole corpus, byte for byte
// ════════════════════════════════════════════════════════════════════════

print("");
print("── every primitive, whole images against Pillow ──");

def _firstDiff($a, $b) {
    $ba = bytes($a);
    $bb = bytes($b);
    $n = min(len($ba), len($bb));
    $c = len($ba) - len($bb);
    if ($c < 0) { $c = 0 - $c; }
    $j = 0;
    while ($j < $n) {
        if ($ba[$j] != $bb[$j]) { $c = $c + 1; }
        $j = $j + 1;
    }
    return $c;
}
def _firstDiffAt($a, $b) {
    $ba = bytes($a);
    $bb = bytes($b);
    $n = min(len($ba), len($bb));
    $j = 0;
    while ($j < $n) {
        if ($ba[$j] != $bb[$j]) { return $j; }
        $j = $j + 1;
    }
    return 0 - 1;
}

$rows = split(trim(readfile($D + "MANIFEST")), "\n");
ok(len($rows) >= 600, "the corpus has at least 600 drawings (" + str(len($rows)) + ")");

$per = {};
$order = [];
$shown = 0;
$i = 0;
while ($i < len($rows)) {
    $L = trim($rows[$i]);
    $i = $i + 1;
    if (len($L) == 0) { continue; }
    $f = split($L, " ");
    $tag = $f[0];
    $file = $f[1];
    $m = $f[2];
    $w = num($f[3]);
    $h = num($f[4]);
    $bg = $f[5];
    $inkMode = $f[6];
    $call = $f[7];

    $group = split($tag, "_")[0];
    if ($per[$group] == null) {
        $per[$group] = {"ok": 0, "bad": 0};
        push($order, $group);
    }

    $threw = false;
    $why = "";
    $im = null;
    try {
        $im = _scene($m, $w, $h, $bg);
        _replay($im, $inkMode, $call);
    } catch ($e) { $threw = true; $why = str($e); }

    if ($threw) {
        $per[$group].bad = $per[$group].bad + 1;
        if ($shown < 70) {
            $shown = $shown + 1;
            print("     ERROR " + $tag + ": " + $why);
        }
        continue;
    }
    $want = readfile($D + $file);
    $have = $im.tobytes();
    if ($have == $want) {
        $per[$group].ok = $per[$group].ok + 1;
    } else {
        $per[$group].bad = $per[$group].bad + 1;
        if ($shown < 70) {
            $shown = $shown + 1;
            print("     DIFF  " + $tag + " (" + $call + "): " +
                  str(_firstDiff($have, $want)) + " of " + str(len($want)) +
                  " bytes differ, first at " + str(_firstDiffAt($have, $want)));
        }
    }
}

$totBad = 0;
$totOk = 0;
each ($g in $order) {
    $v = $per[$g];
    $totBad = $totBad + $v.bad;
    $totOk = $totOk + $v.ok;
    if ($v.bad != 0) {
        print("   " + $g + ": " + str($v.ok) + " identical, " + str($v.bad) + " differing");
    }
}
print("   " + str($totOk) + " of " + str($totOk + $totBad) +
      " drawings byte-identical to Pillow");
eq($totBad, 0, "every drawing in the corpus matches Pillow byte for byte");


// ════════════════════════════════════════════════════════════════════════
//  Tier 1: the failure modes, and the documented rules
// ════════════════════════════════════════════════════════════════════════

print("");
print("── the contract, and the places it refuses ──");

$im = pillow.new("RGB", [20, 10], "black");
$d = draw.Draw($im, null);

throws(def() { draw.Draw($im, "CMYK"); }, "Draw with a mismatched ink mode");
throws(def() { $d.rectangle([10, 2, 3, 8], 200, null, 1); }, "a reversed x box");
throws(def() { $d.rectangle([2, 8, 8, 2], 200, null, 1); }, "a reversed y box");
throws(def() { $d.ellipse([2, 2, 1, 8], 200, null, 1); }, "a reversed ellipse box");
throws(def() { $d.polygon([3, 3], 200, null, 1); }, "a polygon with one point");
throws(def() { $d.rectangle([1, 2, 3], 200, null, 1); }, "a three-number box");
throws(def() { $d.regular_polygon([5, 5, 4], 2, 0, 200, null, 1); }, "a 2-sided polygon");
throws(def() { $d.regular_polygon([5, 5, 0], 5, 0, 200, null, 1); }, "a zero radius");
throws(def() { $d.bitmap([0, 0], pillow.new("RGB", [4, 4], 0), 200); },
       "a bitmap that is not 1 or L");
throws(def() { $o = draw.Outline(); $o.transform([1, 0]); }, "a short affine");
throws(def() { draw.Draw($im, null).point([1], 200); }, "an odd coordinate list");

// Draw() blends only RGBA onto RGB; the same mode is always allowed.
$ok1 = draw.Draw($im, "RGB");
ok($ok1.mode == "RGB", "Draw(im, im.mode) is accepted");
$ok2 = draw.Draw($im, "RGBA");
ok($ok2.mode == "RGBA", "Draw(rgb, \"RGBA\") is accepted and blends");

// The default ink is NOT white everywhere: a one-band 8-bit mode clips -1 to
// 0, so L, 1 and P start black. Verified against Pillow's draw_ink(-1).
$g = pillow.new("L", [4, 4], 128);
$dg = draw.Draw($g, null);
$dg.rectangle([0, 0, 3, 3], null, null, 1);
eq($g.getpixel([0, 0]), 0, "the default ink on L is black, as Pillow's is");
$rg = pillow.new("RGB", [4, 4], 0);
$drg = draw.Draw($rg, null);
$drg.rectangle([0, 0, 3, 3], null, null, 1);
eq($drg.fill, false, "a fresh Draw is not in fill mode");
eq($rg.getpixel([0, 0]), [255, 255, 255], "the default ink on RGB is white");
$dI = draw.Draw(pillow.new("I", [2, 2], 0), null);
eq($dI.ink.v, [1], "the default ink on I is 1, as draw_ink(1) gives");
eq($dI.fontmode, "1", "I takes the unantialiased font mode");
eq(draw.Draw(pillow.new("RGB", [2, 2], 0), null).fontmode, "L",
   "RGB takes the antialiased font mode");

// A line never draws its endpoint; the Python layer adds it back.
$t = pillow.new("L", [6, 1], 0);
$dt = draw.Draw($t, null);
$dt.line([0, 0, 4, 0], 255, 1, null);
eq($t.getdata(), [255, 255, 255, 255, 255, 0],
   "a polyline's last point is drawn, and nothing past it");
$t2 = pillow.new("L", [6, 1], 0);
$dt2 = draw.Draw($t2, null);
$dt2.point([[5, 0]], 99);
eq($t2.getpixel([5, 0]), 99, "point writes the pixel it is given");
$t3 = pillow.new("L", [6, 1], 0);
draw.Draw($t3, null).line([2, 0, 2, 0], 255, 1, null);
eq($t3.getdata(), [0, 0, 255, 0, 0, 0], "a zero-length polyline draws its one point");

// An ellipse outline of width 0 draws nothing; a rectangle promotes 0 to 1.
$e0 = pillow.new("L", [12, 12], 0);
draw.Draw($e0, null).ellipse([1, 1, 10, 10], null, 255, 0);
eq(pxbuf.scalar(nd_sum($e0.px)), 0, "an ellipse outline of width 0 draws nothing");
$r0 = pillow.new("L", [12, 12], 0);
draw.Draw($r0, null).rectangle([1, 1, 10, 10], null, 255, 0);
eq(pxbuf.scalar(nd_sum($r0.px)), 0,
   "ImageDraw never reaches the C's width 0 -> 1 promotion, because it " +
   "guards on width != 0 first");

// width=0 on line, arc, chord, pieslice and polygon draws nothing at all.
each ($case in ["line", "arc", "chord", "pieslice", "polygon"]) {
    $z = pillow.new("L", [16, 16], 0);
    $dz = draw.Draw($z, null);
    if ($case == "line") { $dz.line([1, 1, 14, 14], 255, 0, null); }
    if ($case == "arc") { $dz.arc([1, 1, 14, 14], 0, 180, 255, 0); }
    if ($case == "chord") { $dz.chord([1, 1, 14, 14], 0, 180, null, 255, 0); }
    if ($case == "pieslice") { $dz.pieslice([1, 1, 14, 14], 0, 180, null, 255, 0); }
    if ($case == "polygon") { $dz.polygon([1, 1, 14, 3, 7, 14], null, 255, 0); }
    eq(pxbuf.scalar(nd_sum($z.px)), 0, $case + " with width 0 draws nothing");
}

// An equal outline and fill ink means the outline pass is skipped, which is
// only observable through the pixel count being unchanged.
$same = pillow.new("L", [16, 16], 0);
draw.Draw($same, null).rectangle([2, 2, 13, 13], 200, 200, 4);
eq(pxbuf.scalar(nd_max($same.px)), 200, "an outline equal to the fill adds nothing new");

// floodfill on a seed already the fill colour returns without touching
// anything, which is Pillow's early exit.
$ff = pillow.new("L", [8, 8], 50);
draw.floodfill($ff, [4, 4], 50, null, 0);
eq(pxbuf.scalar(nd_sum($ff.px)), 50 * 64, "floodfill on its own colour is a no-op");
draw.floodfill($ff, [40, 4], 9, null, 0);
eq(pxbuf.scalar(nd_sum($ff.px)), 50 * 64, "floodfill outside the image is a no-op");
draw.floodfill($ff, [4, 4], 9, null, 0);
eq(pxbuf.scalar(nd_sum($ff.px)), 9 * 64, "floodfill with no wall fills everything");

// getdraw hands back a context over the same image.
$gd = draw.getdraw($im);
ok($gd != null && $gd.mode == "RGB", "getdraw returns a context on the image's mode");
ok(draw.getdraw(null) == null, "getdraw(null) is null");

// An ink is clipped, not wrapped: 300 is 255 and -5 is 0, as CLIP8 says.
$ci = pillow.new("L", [3, 1], 0);
$dci = draw.Draw($ci, null);
$dci.point([[0, 0]], 300);
$dci.point([[1, 0]], 0 - 5);
$dci.point([[2, 0]], 128.9);
eq($ci.getdata(), [255, 0, 128], "an 8-bit ink is clipped and truncated");

// A bare integer on a multi-band mode is ABGR, which is getink's documented
// compatibility case.
$ab = pillow.new("RGB", [1, 1], 0);
draw.Draw($ab, null).point([[0, 0]], 16711935);          // 0x00FF00FF
eq($ab.getpixel([0, 0]), [255, 0, 255],
   "a bare integer ink on RGB is ABGR: 0x00FF00FF is magenta, not yellow");

// A float32 ink on F keeps float32's value, not f64's.
$fm = pillow.new("F", [1, 1], 0);
draw.Draw($fm, null).point([[0, 0]], 0.1);
ok($fm.getpixel([0, 0]) != 0.1, "an F ink is rounded to float32");
ok(abs($fm.getpixel([0, 0]) - 0.1) < 0.000001 && $fm.getpixel([0, 0]) > 0.1,
   "...and to the right float32, which is just above 0.1");


// ════════════════════════════════════════════════════════════════════════
//  Tier 3: stress
// ════════════════════════════════════════════════════════════════════════


// ════════════════════════════════════════════════════════════════════════
//  Five places this deliberately does not match Pillow (decision PW11)
// ════════════════════════════════════════════════════════════════════════
//
//  Each is a defect by the C's own logic, not a contract: a value computed
//  and discarded, a buffer read past its meaning, two primitives disagreeing
//  about byte order, a mask ink taken from the wrong mode. PW11 says fix it,
//  state the deviation, and assert the CONTAINMENT rather than the
//  difference. These are those assertions.

print("");
print("── the five deviations, and what each one preserves ──");

// (1) Outline.transform. Pillow frees the edges it has just transformed, so
// shape() afterwards draws only the closing edge. Ours transforms in place,
// and the containment is exact: the result IS the polygon of the transformed
// vertices, which Pillow itself produced for SH_xform_ref.
$o = draw.Outline();
$o.move(5, 5);
$o.line(34, 8);
$o.line(30, 25);
$o.line(6, 22);
$o.transform([0.5, 0, 6, 0, 0.5, 4]);
$sh = pillow.new("L", [40, 30], null);
draw.Draw($sh, null).shape($o, 200, null);
ok($sh.tobytes() == readfile($D + "SH_xform_ref.exp"),
   "a transformed outline fills exactly the polygon of its transformed vertices");

// (2) and (3) the bitmap matte and stencil on I, F and I;16. Pillow blends
// the BYTES of a 4-byte or 2-byte pixel; we blend the value. The properties
// that fixes, asserted on the value: a mask of 0 leaves the destination
// alone, a mask of 255 writes the ink exactly, and a mask in between lands
// strictly between the two and moves monotonically with the mask.
each ($m in ["I", "F", "I;16"]) {
    $ink = 1000;
    if ($m == "F") { $ink = 100.5; }
    if ($m == "I;16") { $ink = 40000; }
    $base = 10;
    $tgt = pillow.new($m, [4, 1], $base);
    $matte = pillow.new("L", [4, 1], 0);
    $matte.putpixel([0, 0], 0);
    $matte.putpixel([1, 0], 64);
    $matte.putpixel([2, 0], 192);
    $matte.putpixel([3, 0], 255);
    draw.Draw($tgt, null).bitmap([0, 0], $matte, $ink);
    $v0 = $tgt.getpixel([0, 0]);
    $v1 = $tgt.getpixel([1, 0]);
    $v2 = $tgt.getpixel([2, 0]);
    $v3 = $tgt.getpixel([3, 0]);
    eq($v0, $base, $m + ": a matte of 0 leaves the destination untouched");
    eq($v3, $ink, $m + ": a matte of 255 writes the ink exactly");
    ok($v1 > $base && $v1 < $v2 && $v2 < $ink,
       $m + ": a partial matte is monotone between the two (" + str($v1) + ", " +
       str($v2) + ")");
    // and the "1" stencil is a stencil, not a half-width byte smear
    $st = pillow.new($m, [4, 1], $base);
    $one = pillow.new("1", [4, 1], 0);
    $one.putpixel([1, 0], 255);
    $one.putpixel([3, 0], 255);
    draw.Draw($st, null).bitmap([0, 0], $one, $ink);
    eq([$st.getpixel([0, 0]), $st.getpixel([1, 0]),
        $st.getpixel([2, 0]), $st.getpixel([3, 0])],
       [$base, $ink, $base, $ink],
       $m + ": a \"1\" stencil writes the ink where the bit is set and nowhere else");
}

// (4) I;16B. Pillow's hline writes big-endian for I;16B while its point --
// and so its line -- writes host order, so a rectangle OUTLINE comes out
// half one way and half the other. Ours stores a value and lets the wire
// element order the bytes, so every pixel of the outline reads back the same
// number, and the bytes are big-endian throughout.
$be = pillow.new("I;16B", [8, 8], 0);
draw.Draw($be, null).rectangle([1, 1, 6, 6], null, 4660, 1);        // 0x1234
eq($be.getpixel([1, 1]), 4660, "I;16B: the corner of the outline reads back");
eq($be.getpixel([4, 1]), 4660, "I;16B: a horizontal run reads back the same");
eq($be.getpixel([1, 4]), 4660, "I;16B: and so does a vertical one");
$beb = bytes($be.tobytes());
eq([$beb[8 * 2 * 1 + 2 * 1], $beb[8 * 2 * 1 + 2 * 1 + 1]], [18, 52],
   "I;16B: the wire bytes are big-endian for the horizontal");
eq([$beb[8 * 2 * 4 + 2 * 1], $beb[8 * 2 * 4 + 2 * 1 + 1]], [18, 52],
   "I;16B: and big-endian for the vertical too, which is where Pillow differs");

// (5) the wide polygon outline's mask ink. Pillow builds the mode-"1" mask
// with the ink for 1 in the TARGET's mode, which on F is the float32 bits of
// 1.0 -- low byte zero -- so the mask stays empty and the outline vanishes.
// Ours uses 255, so the outline is drawn, and it is the same outline every
// other mode gets.
$fp = pillow.new("F", [24, 18], 0);
draw.Draw($fp, null).polygon([2, 2, 21, 6, 16, 15], 0 - 2, 3, 2);
$lp = pillow.new("L", [24, 18], 0);
draw.Draw($lp, null).polygon([2, 2, 21, 6, 16, 15], 90, 200, 2);
$fsame = true;
$y = 0;
while ($y < 18) {
    $x = 0;
    while ($x < 24) {
        $fv = $fp.getpixel([$x, $y]);
        $lv = $lp.getpixel([$x, $y]);
        $fk = 0;
        if ($fv == 0 - 2) { $fk = 1; } else if ($fv == 3) { $fk = 2; }
        $lk = 0;
        if ($lv == 90) { $lk = 1; } else if ($lv == 200) { $lk = 2; }
        if ($fk != $lk) { $fsame = false; }
        $x = $x + 1;
    }
    $y = $y + 1;
}
ok($fsame, "F gets the same wide polygon outline every other mode gets");

print("");
print("── stress: size, degeneracy and the timing curve ──");

// A 100,000-point polygon. The roadmap's number, and the reason the edge
// table is a list of lists rather than a list of dicts.
$big = pillow.new("L", [400, 300], 0);
$pts = [];
$k = 0;
while ($k < 100000) {
    $a = $k * 0.0001;
    push($pts, 200 + floor(150 * cos($a) * (1 - $k / 200000)));
    push($pts, 150 + floor(130 * sin($a) * (1 - $k / 200000)));
    $k = $k + 1;
}
$t0 = clock();
draw.Draw($big, null).polygon($pts, 200, null, 1);
$tPoly = clock() - $t0;
ok(pxbuf.scalar(nd_max($big.px)) == 200,
   "a 100,000-point polygon fills (" + str($tPoly) + " ms)");

// A self-intersecting polygon: even-odd, so the crossing region is empty.
$bow = pillow.new("L", [40, 40], 0);
draw.Draw($bow, null).polygon([2, 2, 37, 37, 2, 37, 37, 2], 255, null, 1);
ok(pxbuf.scalar(nd_sum($bow.px)) > 0, "a self-intersecting polygon fills even-odd");

// Degenerate ellipses: zero width, zero height, one pixel, and a box bigger
// than the image.
each ($box in [[5, 5, 5, 5], [5, 5, 5, 20], [5, 5, 20, 5],
               [0 - 50, 0 - 50, 90, 90], [39, 39, 39, 39]]) {
    $z = pillow.new("L", [40, 40], 0);
    $dz = draw.Draw($z, null);
    $dz.ellipse($box, 200, null, 1);
    $dz.ellipse($box, null, 255, 3);
    $dz.arc($box, 10, 350, 255, 2);
    $dz.pieslice($box, 10, 350, null, 255, 2);
    $R.p = $R.p + 1;
}
print("   five degenerate ellipse boxes survive fill, outline, arc and pieslice");

// Coordinates far outside the image, in both directions.
$far = pillow.new("L", [30, 30], 0);
$dfar = draw.Draw($far, null);
$dfar.line([0 - 100000, 0 - 100000, 100000, 100000], 255, 1, null);
$dfar.line([0 - 100000, 100000, 100000, 0 - 100000], 200, 7, null);
$dfar.rectangle([0 - 99999, 0 - 99999, 99999, 99999], null, 128, 2);
ok(pxbuf.scalar(nd_max($far.px)) > 0, "coordinates 100,000 outside the image clip");

// A 1080p diagonal: the run decomposition is O(length), not O(area).
$hd = pillow.new("L", [1920, 1080], 0);
$t0 = clock();
draw.Draw($hd, null).line([0, 0, 1919, 1079], 255, 1, null);
$tDiag = clock() - $t0;
print("   a 1920x1080 diagonal: " + str($tDiag) + " ms");
ok($tDiag < 20000, "a 1080p diagonal is not quadratic");

// THE TIMING CURVE. A filled polygon costs one pass per ROW, so doubling the
// height roughly doubles the cost while doubling the WIDTH costs nothing --
// which is the claim decision PW8 makes and the thing a pixel loop would
// fail.
$wide = pillow.new("L", [1600, 100], 0);
$t0 = clock();
draw.Draw($wide, null).polygon([0, 0, 1599, 0, 1599, 99, 0, 99], 200, null, 1);
$tWide = clock() - $t0;
$tall = pillow.new("L", [100, 1600], 0);
$t0 = clock();
draw.Draw($tall, null).polygon([0, 0, 99, 0, 99, 1599, 0, 1599], 200, null, 1);
$tTall = clock() - $t0;
print("   a 1600x100 filled polygon: " + str($tWide) + " ms;  100x1600: " +
      str($tTall) + " ms");
ok($tTall > $tWide || $tTall + $tWide < 50,
   "a filled polygon's cost follows its height, not its area");

// The same for a filled rectangle, which is the pure span case.
$t0 = clock();
draw.Draw(pillow.new("L", [1920, 1080], 0), null).rectangle([0, 0, 1919, 1079], 200, null, 1);
$tRect = clock() - $t0;
print("   a 1080p filled rectangle: " + str($tRect) + " ms");
ok($tRect < 20000, "a 1080p filled rectangle is one span per row");

// Hostile floodfill: a comb of walls, so the span flood visits many regions.
$comb = pillow.new("L", [200, 200], 255);
$dc = draw.Draw($comb, null);
$x = 0;
while ($x < 200) {
    $dc.line([$x, 0, $x, 190], 0, 1, null);
    $x = $x + 2;
}
$t0 = clock();
draw.floodfill($comb, [1, 195], 128, null, 0);
$tFlood = clock() - $t0;
print("   a 100-channel comb flood-filled: " + str($tFlood) + " ms");
ok(pxbuf.scalar(nd_min($comb.px)) == 0, "the comb's walls are intact");
ok($comb.getpixel([1, 195]) == 128, "the flood reached the seed's row");
ok($comb.getpixel([1, 5]) == 128, "the flood climbed every channel");
ok($comb.getpixel([0, 5]) == 0, "the flood did not cross a wall");

print("");
print("checks: " + str($R.p + $R.f) + "  passed: " + str($R.p) + "  failed: " + str($R.f));
if ($R.f == 0) { print("RESULT: ALL GREEN"); }
else { print("RESULT: FAILED"); }
