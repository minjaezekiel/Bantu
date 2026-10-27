// ════════════════════════════════════════════════════════════════════════
//  tests/pillow_core_test.b — pillow Phase P2: the object model and PPM.
//
//  The Image object, the mode table, and the first codec, end to end:
//  new -> operate -> save -> open -> the same pixels.
//
//  The mode table is checked against tests/fixtures/pillow/reference/
//  modes.txt, which Pillow itself produced, so band counts, band names,
//  basemode, basetype and Image.new's default fill are all differential
//  rather than asserted from memory.
//
//  Run:  bantu run tests/pillow_core_test.b
// ════════════════════════════════════════════════════════════════════════

include "./pillow/pillow.b" as pillow;

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

$TMP = "/tmp/pillow_core_test_";

// ════════════════════════════════════════════════════════════════════════
//  The mode table, against Pillow's own
// ════════════════════════════════════════════════════════════════════════
print("── the mode table, against Pillow ──");

$rows = split(readfile("tests/fixtures/pillow/reference/modes.txt"), "\n");
$seen = 0;
$bad = 0;
$i = 0;
while ($i < len($rows)) {
    $L = trim($rows[$i]);
    if (len($L) > 0 && $L[0] != "#") {
        $f = split($L, " ");
        if ($f[0] == "MODES") {
            // Pillow's Image.MODES, in Pillow's order
            $want = [];
            $k = 1;
            while ($k < len($f)) { push($want, $f[$k]); $k = $k + 1; }
            eq(pillow.MODES, $want, "Image.MODES matches Pillow exactly");
        } else {
            $m = $f[0];
            $wantBands = num($f[1]);
            $wantBase = $f[2];
            $wantType = $f[3];
            $wantNames = split($f[4], ",");
            $wantFill = split($f[5], ",");
            if (pillow.getmodebands($m) != $wantBands) { $bad = $bad + 1; print("  FAIL bands " + $m); }
            if (pillow.getmodebase($m) != $wantBase)   { $bad = $bad + 1; print("  FAIL basemode " + $m + " got " + pillow.getmodebase($m) + " want " + $wantBase); }
            if (pillow.getmodetype($m) != $wantType)   { $bad = $bad + 1; print("  FAIL basetype " + $m); }
            if (str(pillow.getmodebandnames($m)) != str($wantNames)) {
                $bad = $bad + 1;
                print("  FAIL bandnames " + $m + " got " + str(pillow.getmodebandnames($m)) + " want " + str($wantNames));
            }
            // Image.new's default fill, which is ZERO in every band
            $im = pillow.new($m, [1, 1], null);
            $gp = $im.getpixel([0, 0]);
            if (type($gp) != "list") { $gp = [$gp]; }
            $fillOk = true;
            $k = 0;
            while ($k < len($wantFill)) {
                if ($gp[$k] != num($wantFill[$k])) { $fillOk = false; }
                $k = $k + 1;
            }
            if (!$fillOk) {
                $bad = $bad + 1;
                print("  FAIL default fill " + $m + " got " + str($gp) + " want " + str($wantFill));
            }
            $seen = $seen + 1;
        }
    }
    $i = $i + 1;
}
eq($seen, 20, "all 20 modes present in the reference");
eq($bad, 0, "all 20 modes match Pillow on bands, names, basemode, basetype and default fill");

// RGBA's default fill is transparent black -- the mistake worth a named test
eq(pillow.new("RGBA", [1, 1], null).getpixel([0, 0]), [0, 0, 0, 0],
   "a new RGBA image is TRANSPARENT black, not opaque");

// ════════════════════════════════════════════════════════════════════════
//  Image.new
// ════════════════════════════════════════════════════════════════════════
print("── Image.new ──");

$im = pillow.new("RGB", [4, 3], "cornflowerblue");
eq($im.mode, "RGB", "mode");
eq($im.size, [4, 3], "size is [width, height]");
eq($im.width, 4, "width");
eq($im.height, 3, "height");
eq($im.getbands(), ["R", "G", "B"], "getbands");
eq($im.getpixel([0, 0]), [100, 149, 237], "cornflowerblue is (100,149,237)");
eq($im.getpixel([3, 2]), [100, 149, 237], "the far corner too");
ok($im.load() == $im, "load() returns the image");

eq(pillow.new("L", [2, 2], 128).getpixel([1, 1]), 128, "a numeric colour");
eq(pillow.new("RGB", [2, 2], [1, 2, 3]).getpixel([0, 0]), [1, 2, 3], "a list colour");
eq(pillow.new("RGB", [2, 2], "#336699").getpixel([0, 0]), [51, 102, 153], "a hex colour");
eq(pillow.new("RGBA", [2, 2], "rgba(1,2,3,4)").getpixel([0, 0]), [1, 2, 3, 4], "an rgba colour");

// degenerate sizes are legal, as in Pillow
eq(pillow.new("RGB", [0, 0], null).size, [0, 0], "a 0x0 image");
eq(pillow.new("L", [1, 1], null).size, [1, 1], "a 1x1 image");
eq(pillow.new("L", [1, 1000], null).size, [1, 1000], "a 1x1000 strip");
eq(pillow.new("L", [1000, 1], null).size, [1000, 1], "a 1000x1 strip");

throws(def() { return pillow.new("NOPE", [2, 2], null); }, "an unknown mode");
throws(def() { return pillow.new("RGB", [-1, 2], null); }, "a negative width");
throws(def() { return pillow.new("RGB", 5, null); }, "a size that is not a list");

// ════════════════════════════════════════════════════════════════════════
//  The decompression-bomb guard (decision PW6)
// ════════════════════════════════════════════════════════════════════════
print("── MAX_IMAGE_PIXELS ──");

eq(pillow.max_image_pixels(), 89478485, "the default is Pillow's number");
throws(def() { return pillow.new("RGB", [100000, 100000], null); },
       "10 billion pixels is refused before any allocation");
pillow.set_max_image_pixels(100);
throws(def() { return pillow.new("L", [11, 11], null); }, "the limit is honoured once lowered");
ok(pillow.new("L", [10, 10], null).size[0] == 10, "and exactly at the limit is allowed");
pillow.set_max_image_pixels(89478485);

// ════════════════════════════════════════════════════════════════════════
//  copy / crop / paste
// ════════════════════════════════════════════════════════════════════════
print("── copy, crop, paste ──");

$a = pillow.new("RGB", [4, 4], [10, 20, 30]);
$b = $a.copy();
$b.putpixel([0, 0], [1, 1, 1]);
eq($a.getpixel([0, 0]), [10, 20, 30], "copy() is independent of the original");
eq($b.getpixel([0, 0]), [1, 1, 1], "and the copy did change");

$c = $a.crop([1, 1, 3, 3]);
eq($c.size, [2, 2], "crop size");
eq($c.getpixel([0, 0]), [10, 20, 30], "crop content");
$c.putpixel([0, 0], [9, 9, 9]);
eq($a.getpixel([1, 1]), [10, 20, 30], "crop() does not alias the source");

// a crop outside the image is legal and zero-filled, as in Pillow
$out = $a.crop([-2, -2, 2, 2]);
eq($out.size, [4, 4], "an out-of-bounds crop keeps the requested size");
eq($out.getpixel([0, 0]), [0, 0, 0], "the outside is zero");
eq($out.getpixel([3, 3]), [10, 20, 30], "the overlap is kept");
eq($a.crop([10, 10, 12, 12]).getpixel([0, 0]), [0, 0, 0], "a fully outside crop is all zero");
eq($a.crop(null).size, [4, 4], "crop(null) copies");
throws(def() { return $a.crop([3, 0, 1, 4]); }, "a crop box with negative width");

// paste an image
$dst = pillow.new("RGB", [4, 4], [0, 0, 0]);
$src = pillow.new("RGB", [2, 2], [255, 0, 0]);
$dst.paste($src, [1, 1], null);
eq($dst.getpixel([1, 1]), [255, 0, 0], "paste lands at the box");
eq($dst.getpixel([2, 2]), [255, 0, 0], "and covers its size");
eq($dst.getpixel([0, 0]), [0, 0, 0], "and nothing outside");
eq($dst.getpixel([3, 3]), [0, 0, 0], "including the far corner");

// paste a colour
$dst2 = pillow.new("RGB", [4, 4], [0, 0, 0]);
$dst2.paste([1, 2, 3], [0, 0, 2, 2], null);
eq($dst2.getpixel([0, 0]), [1, 2, 3], "paste a colour into a box");
eq($dst2.getpixel([2, 2]), [0, 0, 0], "outside the box is untouched");

// paste partly outside, both directions
$dst3 = pillow.new("L", [4, 4], 0);
$dst3.paste(pillow.new("L", [3, 3], 200), [-1, -1], null);
eq($dst3.getpixel([0, 0]), 200, "a paste overhanging the top-left clips");
eq($dst3.getpixel([2, 2]), 0, "and does not wrap");
$dst3.paste(pillow.new("L", [3, 3], 100), [3, 3], null);
eq($dst3.getpixel([3, 3]), 100, "a paste overhanging the bottom-right clips");
$dst4 = pillow.new("L", [4, 4], 7);
$dst4.paste(pillow.new("L", [2, 2], 9), [10, 10], null);
eq($dst4.getpixel([3, 3]), 7, "a paste entirely outside changes nothing");

// paste with a mask
$dstM = pillow.new("L", [2, 1], 0);
$srcM = pillow.new("L", [2, 1], 255);
$mask = pillow.new("L", [2, 1], 0);
$mask.putpixel([1, 0], 255);
$dstM.paste($srcM, [0, 0], $mask);
eq($dstM.getpixel([0, 0]), 0, "a zero mask keeps the destination");
eq($dstM.getpixel([1, 0]), 255, "a full mask takes the source");

// a paste across modes converts
$rgbDst = pillow.new("RGB", [2, 2], [0, 0, 0]);
$rgbDst.paste(pillow.new("L", [2, 2], 128), [0, 0], null);
eq($rgbDst.getpixel([0, 0]), [128, 128, 128], "paste converts the source mode");

// ════════════════════════════════════════════════════════════════════════
//  split / merge / getchannel / putalpha
// ════════════════════════════════════════════════════════════════════════
print("── bands ──");

$rgb = pillow.new("RGB", [2, 2], [10, 20, 30]);
$parts = $rgb.split();
eq(len($parts), 3, "split gives three bands");
eq($parts[0].mode, "L", "a split band is mode L");
eq($parts[0].getpixel([0, 0]), 10, "band R");
eq($parts[1].getpixel([0, 0]), 20, "band G");
eq($parts[2].getpixel([0, 0]), 30, "band B");

$merged = pillow.merge("RGB", $parts);
eq($merged.getpixel([1, 1]), [10, 20, 30], "merge rebuilds the image");
eq($merged.mode, "RGB", "merge mode");

eq($rgb.getchannel("G").getpixel([0, 0]), 20, "getchannel by name");
eq($rgb.getchannel(2).getpixel([0, 0]), 30, "getchannel by index");
throws(def() { return $rgb.getchannel("A"); }, "getchannel for a band that is not there");
throws(def() { return $rgb.getchannel(5); }, "getchannel out of range");
throws(def() { return pillow.merge("RGB", [$parts[0], $parts[1]]); }, "merge with too few bands");
throws(def() { return pillow.merge("RGB", [$parts[0], $parts[1], $rgb]); }, "merge with a multi-band input");

$pa = pillow.new("RGB", [2, 2], [1, 2, 3]);
$pa.putalpha(128);
eq($pa.mode, "RGBA", "putalpha promotes RGB to RGBA");
eq($pa.getpixel([0, 0]), [1, 2, 3, 128], "and sets the alpha");
$pa.putalpha(pillow.new("L", [2, 2], 64));
eq($pa.getpixel([1, 1]), [1, 2, 3, 64], "putalpha from an image");
$la = pillow.new("L", [2, 2], 5);
$la.putalpha(99);
eq($la.mode, "LA", "putalpha promotes L to LA");

// ════════════════════════════════════════════════════════════════════════
//  pixels, data, point
// ════════════════════════════════════════════════════════════════════════
print("── pixels and point ──");

$p = pillow.new("L", [3, 2], 0);
$p.putpixel([0, 0], 1);
$p.putpixel([2, 1], 9);
eq($p.getpixel([0, 0]), 1, "putpixel then getpixel");
eq($p.getpixel([2, 1]), 9, "the far corner");
eq($p.getdata(), [1, 0, 0, 0, 0, 9], "getdata is row-major");
throws(def() { return $p.getpixel([3, 0]); }, "getpixel outside the image");
throws(def() { return $p.getpixel([0, 2]); }, "getpixel past the last row");
throws(def() { return $p.getpixel([-1, 0]); }, "getpixel at a negative coordinate");
throws(def() { return $p.getpixel(5); }, "getpixel with a non-pair");

$q = pillow.new("RGB", [2, 1], [1, 2, 3]);
eq($q.getdata(), [[1, 2, 3], [1, 2, 3]], "getdata groups bands per pixel");

$pd = pillow.new("L", [3, 1], 0);
$pd.putdata([7, 8, 9], null, null);
eq($pd.getdata(), [7, 8, 9], "putdata");
$pd2 = pillow.new("L", [3, 1], 5);
$pd2.putdata([1], null, null);
eq($pd2.getdata(), [1, 5, 5], "putdata replaces only the leading pixels");
$pd3 = pillow.new("L", [2, 1], 0);
$pd3.putdata([10, 20], 2, 1);
eq($pd3.getdata(), [21, 41], "putdata applies scale and offset");
throws(def() { return $pd.putdata([1, 2, 3, 4, 5], null, null); }, "putdata with too many pixels");

// point() with an explicit table
$inv = [];
$i = 0;
while ($i < 256) { push($inv, 255 - $i); $i = $i + 1; }
$src = pillow.new("L", [2, 1], 10);
eq($src.point($inv, null).getpixel([0, 0]), 245, "point() with a 256-entry table");
// point() with a function: called once per value, never per pixel
$calls = {"n": 0};
$fn = def($v) { $calls.n = $calls.n + 1; return 255 - $v; };
$big = pillow.new("L", [50, 50], 10);
eq($big.point($fn, null).getpixel([0, 0]), 245, "point() with a function");
eq($calls.n, 256, "the function ran 256 times, not 2500 (once per value, not per pixel)");
throws(def() { return $src.point([1, 2, 3], null); }, "point() with too short a table");
throws(def() { return pillow.new("F", [2, 2], null).point($inv, null); }, "point() on mode F");

// one table per band, concatenated, as Pillow allows
$triple = [];
$b = 0;
while ($b < 3) {
    $i = 0;
    while ($i < 256) { push($triple, ($i + $b) % 256); $i = $i + 1; }
    $b = $b + 1;
}
eq(pillow.new("RGB", [1, 1], [5, 5, 5]).point($triple, null).getpixel([0, 0]), [5, 6, 7],
   "point() with one table per band");

// ════════════════════════════════════════════════════════════════════════
//  statistics
// ════════════════════════════════════════════════════════════════════════
print("── statistics ──");

$s = pillow.new("L", [4, 1], 0);
$s.putpixel([0, 0], 10);
$s.putpixel([1, 0], 200);
eq($s.getextrema(), [0, 200], "getextrema on one band");
$s2 = pillow.new("RGB", [2, 1], [1, 2, 3]);
$s2.putpixel([0, 0], [250, 0, 100]);
eq($s2.getextrema(), [[1, 250], [0, 2], [3, 100]], "getextrema per band");

$hist = pillow.new("L", [4, 1], 0).histogram(null);
eq(len($hist), 256, "an L histogram has 256 bins");
eq($hist[0], 4, "all four pixels are in bin 0");
$histRGB = pillow.new("RGB", [2, 1], [1, 2, 3]).histogram(null);
eq(len($histRGB), 768, "an RGB histogram is 3 x 256, concatenated");
eq($histRGB[1], 2, "band R bin 1");
eq($histRGB[256 + 2], 2, "band G bin 2");
eq($histRGB[512 + 3], 2, "band B bin 3");
throws(def() { return pillow.new("F", [2, 2], null).histogram(null); }, "histogram on mode F");

// getbbox
eq(pillow.new("L", [4, 4], 0).getbbox(), null, "getbbox of an all-zero image is null");
$bb = pillow.new("L", [4, 4], 0);
$bb.putpixel([1, 2], 5);
eq($bb.getbbox(), [1, 2, 2, 3], "getbbox of a single pixel");
$bb.putpixel([3, 3], 5);
eq($bb.getbbox(), [1, 2, 4, 4], "getbbox spans both");
eq(pillow.new("L", [4, 4], 7).getbbox(), [0, 0, 4, 4], "getbbox of a full image");
eq(pillow.new("L", [0, 0], null).getbbox(), null, "getbbox of a 0x0 image");

// getcolors
$gc = pillow.new("L", [4, 1], 0);
$gc.putpixel([0, 0], 5);
eq($gc.getcolors(null), [[3, 0], [1, 5]], "getcolors on L");
eq(pillow.new("RGB", [2, 1], [1, 2, 3]).getcolors(null), [[2, [1, 2, 3]]], "getcolors on RGB");
$many = pillow.new("L", [300, 1], 0);
$i = 0;
while ($i < 256) { $many.putpixel([$i, 0], $i); $i = $i + 1; }
eq($many.getcolors(10), null, "getcolors returns null past maxcolors");
eq(len($many.getcolors(300)), 256, "and the list when it fits");

// ════════════════════════════════════════════════════════════════════════
//  frombytes / tobytes / fromarray
// ════════════════════════════════════════════════════════════════════════
print("── bytes and arrays ──");

$fb = pillow.frombytes("RGB", [2, 1], frombytes([1, 2, 3, 4, 5, 6]));
eq($fb.getpixel([0, 0]), [1, 2, 3], "frombytes first pixel");
eq($fb.getpixel([1, 0]), [4, 5, 6], "frombytes second pixel");
eq(bytes($fb.tobytes()), [1, 2, 3, 4, 5, 6], "tobytes round-trips");
throws(def() { return pillow.frombytes("RGB", [2, 1], frombytes([1, 2, 3])); },
       "frombytes with too few bytes");

$arr = nd([[1, 2], [3, 4]], "i64");
$fa = pillow.fromarray($arr, null);
eq($fa.mode, "L", "fromarray guesses L for a 2-D integer array");
eq($fa.size, [2, 2], "fromarray size");
eq($fa.getpixel([1, 0]), 2, "fromarray content, x then y");
$arr3 = nd_zeros([2, 2, 3], "i64");
eq(pillow.fromarray($arr3, null).mode, "RGB", "fromarray guesses RGB for 3 bands");
eq(pillow.fromarray(nd_zeros([2, 2, 4], "i64"), null).mode, "RGBA", "and RGBA for 4");
eq(pillow.fromarray(nd_zeros([2, 2], "f64"), null).mode, "F", "and F for a float array");
// the data is copied, not aliased
$src2 = nd([[1, 2], [3, 4]], "i64");
$cp = pillow.fromarray($src2, null);
nd_set($src2, [0, 0], 99);
eq($cp.getpixel([0, 0]), 1, "fromarray copies rather than aliasing");
throws(def() { return pillow.fromarray(nd_zeros([2, 2, 3], "i64"), "L"); },
       "fromarray with a mode that does not fit the band count");
throws(def() { return pillow.fromarray(nd_zeros([2], "i64"), null); }, "fromarray with a 1-D array");

// ════════════════════════════════════════════════════════════════════════
//  blend / composite / alpha_composite / eval
// ════════════════════════════════════════════════════════════════════════
print("── blend, composite, eval ──");

$z = pillow.new("L", [2, 1], 0);
$o = pillow.new("L", [2, 1], 100);
eq(pillow.blend($z, $o, 0.0).getpixel([0, 0]), 0, "blend at 0.0 is the first image");
eq(pillow.blend($z, $o, 1.0).getpixel([0, 0]), 100, "blend at 1.0 is the second");
eq(pillow.blend($z, $o, 0.5).getpixel([0, 0]), 50, "blend at 0.5 is the midpoint");
throws(def() { return pillow.blend($z, pillow.new("L", [3, 1], 0), 0.5); }, "blend with mismatched sizes");
throws(def() { return pillow.blend($z, pillow.new("RGB", [2, 1], null), 0.5); }, "blend with mismatched modes");

$m2 = pillow.new("L", [2, 1], 0);
$m2.putpixel([0, 0], 255);
$comp = pillow.composite($o, $z, $m2);
eq($comp.getpixel([0, 0]), 100, "composite takes image 1 where the mask is 255");
eq($comp.getpixel([1, 0]), 0, "and image 2 where the mask is 0");

// alpha_composite: opaque red over opaque blue is red
$red = pillow.new("RGBA", [1, 1], [255, 0, 0, 255]);
$blue = pillow.new("RGBA", [1, 1], [0, 0, 255, 255]);
eq(pillow.alpha_composite($blue, $red).getpixel([0, 0]), [255, 0, 0, 255],
   "an opaque source covers the destination");
// a fully transparent source leaves the destination
$clear = pillow.new("RGBA", [1, 1], [255, 0, 0, 0]);
eq(pillow.alpha_composite($blue, $clear).getpixel([0, 0]), [0, 0, 255, 255],
   "a transparent source changes nothing");
// half-transparent white over black
$halfw = pillow.new("RGBA", [1, 1], [255, 255, 255, 128]);
$black = pillow.new("RGBA", [1, 1], [0, 0, 0, 255]);
$ac = pillow.alpha_composite($black, $halfw);
ok($ac.getpixel([0, 0])[0] >= 127 && $ac.getpixel([0, 0])[0] <= 129,
   "half-transparent white over black is mid grey (got " + str($ac.getpixel([0, 0])[0]) + ")");
eq($ac.getpixel([0, 0])[3], 255, "and stays opaque");
throws(def() { return pillow.alpha_composite($z, $o); }, "alpha_composite needs RGBA");

eq(pillow.eval(pillow.new("L", [1, 1], 10), def($v) { return $v * 2; }).getpixel([0, 0]), 20,
   "eval applies a function through a table");

// ════════════════════════════════════════════════════════════════════════
//  PPM: every one of the seven formats, round-tripped
// ════════════════════════════════════════════════════════════════════════
print("── PPM / PGM / PBM ──");

eq(pillow.formats(), ["PPM"], "PPM is registered");
eq(pillow.registered_extensions()[".ppm"], "PPM", "the .ppm extension maps to PPM");
eq(pillow.registered_extensions()[".pgm"], "PPM", "and .pgm");
eq(pillow.registered_extensions()[".pbm"], "PPM", "and .pbm");

// P6 (raw RGB)
$rgbIm = pillow.new("RGB", [3, 2], null);
$rgbIm.putpixel([0, 0], [255, 0, 0]);
$rgbIm.putpixel([1, 0], [0, 255, 0]);
$rgbIm.putpixel([2, 1], [1, 2, 3]);
$rgbIm.save($TMP + "a.ppm", null);
$rt = pillow.open($TMP + "a.ppm");
eq($rt.mode, "RGB", "P6 reads back as RGB");
eq($rt.size, [3, 2], "P6 size");
eq($rt.getdata(), $rgbIm.getdata(), "P6 round-trips every pixel");
eq($rt.format, "PPM", "the format is recorded");

// P5 (raw grey)
$lIm = pillow.new("L", [4, 3], 0);
$lIm.putpixel([0, 0], 1);
$lIm.putpixel([3, 2], 254);
$lIm.save($TMP + "b.pgm", null);
eq(pillow.open($TMP + "b.pgm").getdata(), $lIm.getdata(), "P5 round-trips");
eq(pillow.open($TMP + "b.pgm").mode, "L", "P5 reads back as L");

// P4 (raw bilevel) -- the packed one, where an off-by-one shifts every pixel
$bIm = pillow.new("1", [9, 2], 0);
$bIm.putpixel([0, 0], 255);
$bIm.putpixel([8, 0], 255);
$bIm.putpixel([4, 1], 255);
$bIm.save($TMP + "c.pbm", null);
$brt = pillow.open($TMP + "c.pbm");
eq($brt.mode, "1", "P4 reads back as 1");
eq($brt.size, [9, 2], "P4 size, with a row that is not a byte multiple");
eq($brt.getdata(), $bIm.getdata(), "P4 round-trips across the byte boundary");

// a 1-pixel-wide bilevel image, the narrowest packing case
$n1 = pillow.new("1", [1, 3], 0);
$n1.putpixel([0, 1], 255);
$n1.save($TMP + "d.pbm", null);
eq(pillow.open($TMP + "d.pbm").getdata(), $n1.getdata(), "a 1-wide P4 round-trips");

// I;16 (16-bit grey, big-endian on the wire)
$wIm = pillow.new("I;16", [2, 1], 0);
$wIm.putpixel([0, 0], 65535);
$wIm.putpixel([1, 0], 300);
$wIm.save($TMP + "e.pgm", null);
$wrt = pillow.open($TMP + "e.pgm");
eq($wrt.getdata(), [65535, 300], "a 16-bit P5 round-trips, big-endian");

// the plain (ASCII) forms
$plainSpec = pillow.new("RGB", [2, 1], [10, 20, 30]);
writefile($TMP + "f.ppm", "P3\n2 1\n255\n10 20 30 40 50 60\n");
$p3 = pillow.open($TMP + "f.ppm");
eq($p3.getdata(), [[10, 20, 30], [40, 50, 60]], "P3 (plain RGB) decodes");
writefile($TMP + "g.pgm", "P2\n3 1\n255\n1 2 3\n");
eq(pillow.open($TMP + "g.pgm").getdata(), [1, 2, 3], "P2 (plain grey) decodes");
writefile($TMP + "h.pbm", "P1\n4 1\n0 1 0 1\n");
eq(pillow.open($TMP + "h.pbm").getdata(), [255, 0, 255, 0], "P1 decodes, and 1 means black");

// comments and odd whitespace in the header
writefile($TMP + "i.ppm", "P3\n# a comment\n2 1 # another\n255\n1 1 1   2 2 2\n");
eq(pillow.open($TMP + "i.ppm").getdata(), [[1, 1, 1], [2, 2, 2]], "comments in the header");

// a 0x0 image
pillow.new("L", [0, 0], null).save($TMP + "z.pgm", null);
eq(pillow.open($TMP + "z.pgm").size, [0, 0], "a 0x0 PPM round-trips");

// ─── hostile PPM input: every failure must be catchable ─────────────────
print("── PPM: hostile input ──");

throws(def() { writefile($TMP + "x1.ppm", "P6\n2 2\n255\n"); return pillow.open($TMP + "x1.ppm"); },
       "a header with no pixel data");
throws(def() { writefile($TMP + "x2.ppm", "P6\n2 2\n255\nAB"); return pillow.open($TMP + "x2.ppm"); },
       "truncated pixel data");
throws(def() { writefile($TMP + "x3.ppm", "P6\n"); return pillow.open($TMP + "x3.ppm"); },
       "a header that ends early");
throws(def() { writefile($TMP + "x4.ppm", "P6\n2 2\n0\nXXXXXXXXXXXX"); return pillow.open($TMP + "x4.ppm"); },
       "a zero maxval");
throws(def() { writefile($TMP + "x5.ppm", "P6\n2 2\n99999\nXXXXXXXXXXXX"); return pillow.open($TMP + "x5.ppm"); },
       "a maxval above 65535");
throws(def() { writefile($TMP + "x6.ppm", "P6\n-1 2\n255\nAAAA"); return pillow.open($TMP + "x6.ppm"); },
       "a negative width");
throws(def() { writefile($TMP + "x7.ppm", "P6\n60000 60000\n255\nAAAA"); return pillow.open($TMP + "x7.ppm"); },
       "a 3.6-billion-pixel header is refused before allocation");
throws(def() { writefile($TMP + "x8.ppm", "P7\n2 2\n255\nAAAA"); return pillow.open($TMP + "x8.ppm"); },
       "P7 (PAM) is not claimed");
throws(def() { writefile($TMP + "x9.ppm", "not an image at all"); return pillow.open($TMP + "x9.ppm"); },
       "a file that is not an image");
throws(def() { return pillow.open($TMP + "does_not_exist.ppm"); }, "a missing file");
throws(def() { writefile($TMP + "xa.ppm", "P3\n2 1\n255\n1 2 3 4 5\n"); return pillow.open($TMP + "xa.ppm"); },
       "a plain file with too few values");
throws(def() { writefile($TMP + "xb.ppm", "P3\n2 1\n255\n1 2 x 4 5 6\n"); return pillow.open($TMP + "xb.ppm"); },
       "a plain file with a non-numeric value");

// every truncation of a valid file: no crash, no hang
$valid = readfile($TMP + "a.ppm");
$k = 0;
while ($k <= len($valid)) {
    writefile($TMP + "t.ppm", substr($valid, 0, $k));
    try { pillow.open($TMP + "t.ppm"); } catch ($e) { }
    $k = $k + 1;
}
ok(true, "truncated at all " + str(len($valid) + 1) + " lengths: no crash, no hang");

// every single byte corrupted: no crash, no hang
$vb = bytes($valid);
$k = 0;
while ($k < len($vb)) {
    $mut = [];
    $j = 0;
    while ($j < len($vb)) { push($mut, $vb[$j]); $j = $j + 1; }
    $mut[$k] = ($mut[$k] + 137) % 256;
    writefile($TMP + "m.ppm", frombytes($mut));
    try { pillow.open($TMP + "m.ppm"); } catch ($e) { }
    $k = $k + 1;
}
ok(true, "all " + str(len($vb)) + " single-byte corruptions: no crash, no hang");

// save refuses a mode it cannot write, rather than writing nonsense
throws(def() { return pillow.new("CMYK", [2, 2], null).save($TMP + "bad.ppm", null); },
       "PPM refuses to write CMYK");
throws(def() { return pillow.new("RGB", [2, 2], null).save($TMP + "bad.xyz", null); },
       "save with an unknown extension");

// ════════════════════════════════════════════════════════════════════════
//  rule PW2: linear in pixel count
// ════════════════════════════════════════════════════════════════════════
print("── rule PW2: no operation is per-pixel ──");

def timeOp($n) {
    $im = pillow.new("RGB", [$n, $n], [40, 80, 120]);
    $t0 = clock();
    $im.convert("L");
    $im.crop([0, 0, $n, $n]);
    $im.copy();
    $im.histogram(null);
    $im.getextrema();
    return clock() - $t0;
}
$t1 = timeOp(150);
$t2 = timeOp(300);
print("   22,500 px: " + str($t1) + " ms      90,000 px: " + str($t2) + " ms");
ok($t2 <= max($t1, 1) * 12 + 80, "4x the pixels costs well under 12x the time");

// crop and a same-shape copy of a big image must be effectively instant
$bigIm = pillow.new("RGB", [800, 600], [1, 2, 3]);
$t0 = clock();
$cr = $bigIm.crop([10, 10, 790, 590]);
$crMs = clock() - $t0;
print("   crop of a 480,000-pixel image: " + str($crMs) + " ms");
ok($crMs < 400, "crop is a slice plus one copy, not a pixel loop");

print("");
print("checks: " + str($R.p + $R.f) + "  passed: " + str($R.p) + "  failed: " + str($R.f));
if ($R.f == 0) { print("RESULT: ALL GREEN"); }
else { print("RESULT: FAILED"); }
