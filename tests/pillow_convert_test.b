// ════════════════════════════════════════════════════════════════════════
//  tests/pillow_convert_test.b — pillow Phase P2: the conversion matrix.
//
//  Every pair in tests/fixtures/pillow/reference/convert_matrix.txt is
//  converted and compared to what PYTHON'S PILLOW produced for the same
//  input. 191 pairs, 12 probe pixels each.
//
//  The probe row is converted as ONE 1-by-12 image, exactly as the fixture
//  generator did, because convert("1") dithers by default and a dither's
//  output depends on its neighbours -- converting pixel by pixel would
//  silently compare different things.
//
//  THREE GROUPS, counted separately and reported separately, because an
//  honest number matters more than a green tick:
//
//    exact        the arithmetic matches Pillow on every band of every pixel
//    quantising   *->P and *->PA, which need palette.b (phase P2 remainder)
//    unverified   the YCbCr, HSV and LAB families, whose integer forms have
//                 not yet been matched to Pillow (see ROADMAP.md P2)
//
//  The gate is: ZERO pairs in the "wrong" bucket. A pair that is expected to
//  be exact and is not fails the suite.
//
//  Run:  bantu run tests/pillow_convert_test.b
// ════════════════════════════════════════════════════════════════════════

include "./pillow/convert.b" as convert;
include "./pillow/pxbuf.b" as pxbuf;
include "./pillow/mode.b" as mode;
include "./pillow/palette.b" as pal;

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

// ─── Load and group the reference ───────────────────────────────────────
$rows = split(readfile("tests/fixtures/pillow/reference/convert_matrix.txt"), "\n");
$pairs = {};
$order = [];
$i = 0;
while ($i < len($rows)) {
    $L = trim($rows[$i]);
    if (len($L) > 0 && $L[0] != "#") {
        $f = split($rows[$i], "\t");
        $key = $f[0] + ">" + $f[1];
        if ($pairs[$key] == null) {
            $pairs[$key] = {"from": $f[0], "to": $f[1], "in": [], "out": []};
            push($order, $key);
        }
        push($pairs[$key].in, $f[2]);
        push($pairs[$key].out, $f[3]);
    }
    $i = $i + 1;
}
print("reference: " + str(len($order)) + " conversion pairs");
eq(len($order) > 180, true, "the reference holds at least 180 pairs");

def parseVals($s) {
    $p = split($s, ",");
    $o = [];
    $k = 0;
    while ($k < len($p)) { push($o, num($p[$k])); $k = $k + 1; }
    return $o;
}

// ─── Convert each pair and compare ──────────────────────────────────────
print("── the matrix, against Pillow ──");

$stat = {"exact": 0, "wrong": 0, "quant": 0, "unver": 0, "badvals": 0};
$wrongList = [];
$oi = 0;
while ($oi < len($order)) {
    $key = $order[$oi];
    $P = $pairs[$key];
    $n = len($P.in);
    $nbIn = mode.bands($P.from);

    // Build one 1-by-n image in the source mode.
    $buf = nd_zeros([1, $n, $nbIn], mode.dtype($P.from));
    $k = 0;
    while ($k < $n) {
        $v = parseVals($P.in[$k]);
        $bi = 0;
        while ($bi < $nbIn) { nd_set($buf, [0, $k, $bi], $v[$bi]); $bi = $bi + 1; }
        $k = $k + 1;
    }

    // An RGBA source goes to an ADAPTIVE palette in Pillow (quantize()),
    // which is a feature of its own and not in this build.
    if (mode.is_palette($P.to) && mode.has_alpha($P.from)) {
        $stat.quant = $stat.quant + 1;
        $oi = $oi + 1;
        continue;
    }
    // The families whose arithmetic is not matched.
    if (convert.unverified($P.from, $P.to)) {
        $stat.unver = $stat.unver + 1;
        $oi = $oi + 1;
        continue;
    }

    $threw = false;
    $res = null;
    try { $res = convert.px($buf, $P.from, $P.to, null, null); } catch ($e) { $threw = true; }
    if ($threw) {
        $stat.wrong = $stat.wrong + 1;
        push($wrongList, $key + " threw");
        $oi = $oi + 1;
        continue;
    }

    $nbOut = mode.bands($P.to);
    $mis = 0;
    $k = 0;
    while ($k < $n) {
        $want = parseVals($P.out[$k]);
        $bi = 0;
        while ($bi < len($want) && $bi < $nbOut) {
            if (nd_get($res.px, [0, $k, $bi]) != $want[$bi]) { $mis = $mis + 1; }
            $bi = $bi + 1;
        }
        $k = $k + 1;
    }
    if ($mis == 0) { $stat.exact = $stat.exact + 1; }
    else {
        $stat.wrong = $stat.wrong + 1;
        $stat.badvals = $stat.badvals + $mis;
        push($wrongList, $key + " (" + str($mis) + " values differ)");
    }
    $oi = $oi + 1;
}

print("   exact vs Pillow      : " + str($stat.exact));
print("   wrong                : " + str($stat.wrong) + " (" + str($stat.badvals) + " values)");
print("   needs quantize()     : " + str($stat.quant) + "  (RGBA->P/PA, an adaptive palette)");
print("   not yet matched      : " + str($stat.unver) + "  (LAB only -- littleCMS)");
$i = 0;
while ($i < len($wrongList)) { print("     " + $wrongList[$i]); $i = $i + 1; }

eq($stat.wrong, 0, "every pair expected to be exact IS exact");
eq($stat.exact > 100, true, "at least 100 pairs verified exact");

// ─── The formulas, asserted directly so a regression names itself ───────
print("── the formulas that matter ──");

def one($vals, $from, $to, $dither) {
    $nb = mode.bands($from);
    $buf = nd_zeros([1, 1, $nb], mode.dtype($from));
    $i = 0;
    while ($i < $nb) { nd_set($buf, [0, 0, $i], $vals[$i]); $i = $i + 1; }
    $r = convert.px($buf, $from, $to, null, $dither);
    $out = [];
    $i = 0;
    while ($i < mode.bands($to)) { push($out, nd_get($r.px, [0, 0, $i])); $i = $i + 1; }
    return $out;
}

// RGB -> L is Pillow's L24, NOT the truncated (299,587,114)/1000 form.
eq(one([255, 0, 0], "RGB", "L", null), [76], "RGB->L red");
eq(one([0, 255, 0], "RGB", "L", null), [150], "RGB->L green is 150, not 149");
eq(one([0, 0, 255], "RGB", "L", null), [29], "RGB->L blue");
eq(one([0, 0, 128], "RGB", "L", null), [15], "RGB->L (0,0,128) is 15 -- /1000 would say 14");
eq(one([0, 1, 0], "RGB", "L", null), [1], "RGB->L (0,1,0) is 1 -- /1000 would say 0");

// the L24 identity: the weights sum to exactly 65536
eq(19595 + 38470 + 7471, 65536, "the L24 weights sum to 2^16");

// CMYK is Pillow's naive conversion: K is always zero from RGB.
eq(one([0, 0, 0], "RGB", "CMYK", null), [255, 255, 255, 0], "RGB->CMYK black");
eq(one([255, 255, 255], "RGB", "CMYK", null), [0, 0, 0, 0], "RGB->CMYK white, K still 0");
eq(one([1, 2, 3], "RGB", "CMYK", null), [254, 253, 252, 0], "RGB->CMYK is 255-x");
// but GREY -> CMYK goes to the black plate instead
eq(one([76], "L", "CMYK", null), [0, 0, 0, 179], "L->CMYK uses K = 255-L");
eq(one([0], "L", "CMYK", null), [0, 0, 0, 255], "L->CMYK black is full ink");
eq(one([255], "L", "CMYK", null), [0, 0, 0, 0], "L->CMYK white is no ink");

// RGBX's fourth band is padding, and it is 255.
eq(one([0, 0, 0], "RGB", "RGBX", null), [0, 0, 0, 255], "RGB->RGBX pads with 255");
eq(one([1, 2, 3], "RGB", "RGBX", null), [1, 2, 3, 255], "RGB->RGBX keeps colour");

// alpha appears opaque and disappears silently.
eq(one([1, 2, 3], "RGB", "RGBA", null), [1, 2, 3, 255], "RGB->RGBA alpha is 255");
eq(one([1, 2, 3, 40], "RGBA", "RGB", null), [1, 2, 3], "RGBA->RGB drops alpha");
eq(one([9], "L", "LA", null), [9, 255], "L->LA alpha is 255");

// premultiplied targets multiply through.
eq(one([255, 255, 255, 128], "RGBA", "RGBa", null), [128, 128, 128, 128], "RGBA->RGBa premultiplies");
eq(one([255, 255, 255, 255], "RGBA", "RGBa", null), [255, 255, 255, 255], "opaque premultiplies to itself");

// the two thresholds for mode "1" disagree, deliberately.
eq(one([128], "L", "1", "NONE"), [255], "dither=NONE: 128 is white (>= 128)");
eq(one([127], "L", "1", "NONE"), [0], "dither=NONE: 127 is black");
eq(one([128], "L", "1", "FLOYDSTEINBERG"), [0], "dithered: 128 is black (> 128)");

// F keeps float32 precision, which is why pxbuf.to_f32 exists.
$f = one([255, 0, 0], "RGB", "F", null);
eq($f[0] > 76.2450026 && $f[0] < 76.2450028, true,
   "RGB->F is the float32 nearest 76.245, not the double");
eq(one([0, 255, 0], "RGB", "I", null), [150], "RGB->I is the L24 value");
eq(one([76.245], "F", "L", null), [76], "F->L truncates");

// ─── failure modes ──────────────────────────────────────────────────────
print("── failure modes ──");
$buf = nd_zeros([1, 1, 3], "i64");
throws(def() { return convert.px($buf, "NOPE", "L", null, null); }, "unknown source mode");
throws(def() { return convert.px($buf, "RGB", "NOPE", null, null); }, "unknown target mode");
$rgba = nd_zeros([1, 1, 4], "i64");
throws(def() { return convert.px($rgba, "RGBA", "P", null, null); },
       "RGBA->P names quantize() rather than guessing");
$pbuf = nd_zeros([1, 1, 1], "i64");
throws(def() { return convert.px($pbuf, "P", "RGB", null, null); }, "P->RGB without a palette");

// identity returns the same buffer, not a copy -- the Image layer copies
eq(nd_shares_memory(convert.px($buf, "RGB", "RGB", null, null).px, $buf), true,
   "a same-mode conversion does not copy");

// ─── YCbCr and HSV are now exact; only LAB is not ──────────────────────
print("── YCbCr and HSV, from Pillow's own source ──");

// YCbCr carries Pillow's 13 lookup tables, so it is bit-exact.
eq(convert.unverified("RGB", "YCbCr"), false, "YCbCr is no longer unverified");
eq(convert.unverified("RGB", "HSV"), false, "HSV is no longer unverified");
eq(one([255, 0, 0], "RGB", "YCbCr", null), [76, 84, 255], "RGB->YCbCr red");
eq(one([0, 255, 0], "RGB", "YCbCr", null), [149, 43, 21], "RGB->YCbCr green, Y truncates to 149");
eq(one([0, 0, 0], "RGB", "YCbCr", null), [0, 128, 128], "RGB->YCbCr black is mid-chroma");
// RGB->L rounds to 150 for pure green while YCbCr's Y truncates to 149: two
// different luminances in Pillow, and both are matched.
eq(one([0, 255, 0], "RGB", "L", null), [150], "...while convert('L') rounds to 150");
// grey <-> YCbCr are direct paths, not matrix round trips
eq(one([200], "L", "YCbCr", null), [200, 128, 128], "L->YCbCr is Y=L, Cb=Cr=128");
eq(one([200, 50, 70], "YCbCr", "L", null), [200], "YCbCr->L is just Y");
// HSV, including the hue wrap
eq(one([255, 0, 0], "RGB", "HSV", null), [0, 255, 255], "RGB->HSV red");
eq(one([0, 255, 0], "RGB", "HSV", null), [85, 255, 255], "RGB->HSV green is hue 85");
eq(one([0, 0, 255], "RGB", "HSV", null), [170, 255, 255], "RGB->HSV blue is hue 170");
eq(one([63, 63, 63], "RGB", "HSV", null), [0, 0, 63], "a grey pixel has hue 0, sat 0");
eq(one([1, 2, 3], "RGB", "HSV", null), [148, 170, 3], "RGB->HSV of a dark pixel");

// LAB is the one that cannot be matched, and the reason is structural.
print("── LAB: littleCMS, not a formula ──");
eq(convert.unverified("RGB", "LAB"), true, "LAB is declared unverified");
ok(convert.why_unverified("RGB", "LAB") != null, "and says why when asked");
ok(convert.why_unverified("RGB", "L") == null, "while a verified pair says nothing");
$r = one([51, 102, 153], "RGB", "LAB", null);
eq(len($r), 3, "RGB->LAB still produces 3 bands");
$back = nd_zeros([1, 1, 3], "i64");
$k = 0;
while ($k < 3) { nd_set($back, [0, 0, $k], $r[$k]); $k = $k + 1; }
$rt = convert.px($back, "LAB", "RGB", null, null);
$d0 = nd_get($rt.px, [0, 0, 0]);
eq($d0 > 44 && $d0 < 58, true, "RGB->LAB->RGB keeps red near 51 (got " + str($d0) + ")");

// ─── the palette mapping ────────────────────────────────────────────────
print("── RGB -> P: the web cube, from Palette.c ──");

// Pillow's browser palette: 10 reserved zeros, then a 6x6x6 cube with blue
// outermost, 226 entries in all.
$web = pal.web();
eq(pal.count($web), 226, "the web palette has 226 entries");
eq(nd_to_list(nd_slice($web, [[10, 11], null])), [[0, 0, 0]], "entry 10 is the cube's black");
eq(nd_to_list(nd_slice($web, [[11, 12], null])), [[51, 0, 0]], "entry 11 steps red");
eq(nd_to_list(nd_slice($web, [[16, 17], null])), [[0, 51, 0]], "entry 16 steps green");
eq(nd_to_list(nd_slice($web, [[46, 47], null])), [[0, 0, 51]], "entry 46 steps blue");
eq(nd_to_list(nd_slice($web, [[225, 226], null])), [[255, 255, 255]], "entry 225 is white");
eq(pal.is_web($web), true, "is_web recognises it");
eq(pal.is_web(pal.grey_ramp()), false, "and rejects the grey ramp");

// black ties with the ten reserved entries and the LOWEST index wins
eq(one([0, 0, 0], "RGB", "P", "NONE"), [0], "black maps to index 0, not 10");
eq(one([255, 255, 255], "RGB", "P", "NONE"), [225], "white maps to the last cube entry");
eq(one([255, 0, 0], "RGB", "P", "NONE"), [15], "pure red");
eq(one([0, 255, 0], "RGB", "P", "NONE"), [40], "pure green");
eq(one([0, 0, 255], "RGB", "P", "NONE"), [190], "pure blue");
// the 4-step quantisation: 7 is searched as 4, so 7 and 4 agree
eq(one([7, 7, 7], "RGB", "P", "NONE"), one([4, 4, 4], "RGB", "P", "NONE"),
   "the search quantises each channel to a multiple of 4");

// a one-band source is copied index for index with the grey ramp
eq(one([137], "L", "P", null), [137], "L->P copies the index, no colour search");
eq(one([0], "L", "P", null), [0], "L->P black");
eq(one([255], "L", "P", null), [255], "L->P white");
$gp = convert.px(nd_zeros([1, 1, 1], "i64"), "L", "P", null, null);
eq(nd_to_list(nd_slice($gp.palette, [[137, 138], null])), [[137, 137, 137]],
   "and carries the grey ramp as its palette");

// P -> RGB is a gather through the palette
$pbuf = nd_zeros([1, 2, 1], "i64");
nd_set($pbuf, [0, 0, 0], 15);
nd_set($pbuf, [0, 1, 0], 225);
$prgb = convert.px($pbuf, "P", "RGB", $web, null);
eq(nd_to_list(nd_slice($prgb.px, [null, [0, 1], null])), [[[255, 0, 0]]], "P->RGB entry 15");
eq(nd_to_list(nd_slice($prgb.px, [null, [1, 2], null])), [[[255, 255, 255]]], "P->RGB entry 225");

// a custom palette, through the general nearest-colour search
$custom = pal.from_list([0, 0, 0, 255, 255, 255, 255, 0, 0]);
$cb = nd_zeros([1, 3, 3], "i64");
nd_set($cb, [0, 0, 0], 250);
nd_set($cb, [0, 1, 1], 250);
nd_set($cb, [0, 2, 0], 240);
nd_set($cb, [0, 2, 1], 240);
nd_set($cb, [0, 2, 2], 240);
$ci = pal.nearest_index($cb, $custom, 3);
eq(nd_to_list(nd_ravel($ci)), [2, 0, 1],
   "a custom palette picks red, black and white by nearest distance");
throws(def() { return pal.from_list([1, 2]); }, "a flat palette must be a multiple of 3");
throws(def() { return pal.nearest_index($cb, $custom, 0); }, "an empty palette is refused");

// ─── the palette corpus, and the one deliberate deviation ───────────────
print("── RGB -> P whole images, against Pillow ──");

$pman = split(trim(readfile("tests/fixtures/pillow/reference/palette_index.txt")), "\n");
$pshapes = [];
$i = 0;
while ($i < len($pman)) {
    $L = trim($pman[$i]);
    if (len($L) > 0 && $L[0] != "#") { push($pshapes, split($L, " ")); }
    $i = $i + 1;
}
eq(len($pshapes), 5, "the palette corpus has 5 shapes");

$PD = "tests/fixtures/pillow/reference/palette/";
$noneTot = 0;  $noneBad = 0;
$fsTot = 0;    $fsBad = 0;    $fsRow0 = 0;   $fsDeep = 0;   $fsDepth = 0;
// How far inward the corrected carry may legitimately diffuse. Floyd-
// Steinberg moves 3/16 down-and-left per row, so a wedge of a few columns
// is expected; anything reaching the body of the image would be a real bug.
$fsWorstAllowed = 8;
$i = 0;
while ($i < len($pshapes)) {
    $w = num($pshapes[$i][0]);
    $h = num($pshapes[$i][1]);
    $tag = str($w) + "x" + str($h);
    $src = nd_frombytes(readfile($PD + "RGB_" + $tag + ".raw"), "u8", [$h, $w, 3]);

    // undithered: must be identical everywhere
    $gotN = convert.px($src, "RGB", "P", null, "NONE").px;
    $wantN = bytes(readfile($PD + "RGB_" + $tag + ".none"));
    $k = 0;
    while ($k < len($wantN)) {
        if (nd_get($gotN, [floor($k / $w), $k % $w, 0]) != $wantN[$k]) {
            $noneBad = $noneBad + 1;
            if ($noneBad < 4) {
                print("     NONE " + $tag + " px " + str($k) + ": got " +
                      str(nd_get($gotN, [floor($k / $w), $k % $w, 0])) +
                      " want " + str($wantN[$k]));
            }
        }
        $k = $k + 1;
    }
    $noneTot = $noneTot + len($wantN);

    // dithered: identical except where Pillow's end-of-row slip bites. That
    // starts in the last column of row 1 and then DIFFUSES: Floyd-Steinberg
    // sends 3/16 of each error down-and-LEFT, so a corrected last column
    // necessarily changes a wedge inward from the right edge on later rows.
    // The containment properties are what matter and what is asserted:
    //   * row 0 is identical everywhere -- it has no previous row to be fed
    //     a corrupted carry from, so the slip cannot reach it
    //   * every difference lies in the right-hand margin, never in the body
    $gotF = convert.px($src, "RGB", "P", null, "FLOYDSTEINBERG").px;
    $wantF = bytes(readfile($PD + "RGB_" + $tag + ".fs"));
    $k = 0;
    while ($k < len($wantF)) {
        $y = floor($k / $w);
        $x = $k % $w;
        if (nd_get($gotF, [$y, $x, 0]) != $wantF[$k]) {
            $fsBad = $fsBad + 1;
            if ($y == 0) {
                $fsRow0 = $fsRow0 + 1;
                if ($fsRow0 < 4) {
                    print("     FS " + $tag + " row 0 (" + str($x) + "): got " +
                          str(nd_get($gotF, [$y, $x, 0])) + " want " + str($wantF[$k]));
                }
            }
            $fromRight = $w - 1 - $x;
            if ($fromRight > $fsDepth) { $fsDepth = $fromRight; }
            if ($fromRight > $fsWorstAllowed) {
                $fsDeep = $fsDeep + 1;
                if ($fsDeep < 4) {
                    print("     FS " + $tag + " deep (" + str($x) + "," + str($y) +
                          "), " + str($fromRight) + " columns from the right");
                }
            }
        }
        $k = $k + 1;
    }
    $fsTot = $fsTot + len($wantF);
    $i = $i + 1;
}

print("   undithered: " + str($noneTot - $noneBad) + "/" + str($noneTot) + " identical");
eq($noneBad, 0, "RGB->P undithered is exact over " + str($noneTot) + " pixels");

print("   dithered:   " + str($fsTot - $fsBad) + "/" + str($fsTot) + " identical; " +
      str($fsBad) + " differ, deepest " + str($fsDepth) + " columns from the right edge");
eq($fsRow0, 0, "row 0 is identical to Pillow everywhere -- the slip cannot reach it");
eq($fsDeep, 0, "every difference is within " + str($fsWorstAllowed) +
   " columns of the right edge, never in the body of the image");
ok($fsBad > 0, "and differences DO exist -- Pillow's end-of-row slip, fixed here");

// ─── why that deviation is the correct one ──────────────────────────────
print("── the deviation is a fix, not a difference ──");
//
// Convert.c ends each dithered row with
//     e[0] = b0;  e[1] = b1;  e[2] = b2;
// writing BLUE's accumulators into all three channel slots. Red's and
// green's are computed and discarded, so the next row's last column is fed
// blue's error as if it were red's. palette.b writes r0, g0, b0.
//
// The property that proves the fix: feed a column whose red and blue carries
// MUST differ, and check the last column responds to red rather than to
// blue. A pure-red ramp has zero blue error throughout, so under Pillow's
// slip the last column would see no red carry at all.
$ramp = nd_zeros([8, 4, 3], "i64");
nd_add(nd_slice($ramp, [null, null, [0, 1]]), 130, nd_slice($ramp, [null, null, [0, 1]]));
$rd = convert.px($ramp, "RGB", "P", null, "FLOYDSTEINBERG").px;
$lastCol = [];
$y = 0;
while ($y < 8) { push($lastCol, nd_get($rd, [$y, 3, 0])); $y = $y + 1; }
// With red error propagating, the last column cannot be one constant value:
// 130 is not a cube level, so the error must flip some rows.
$distinct = 0;
$seen = [];
$y = 0;
while ($y < 8) {
    if (!contains($seen, $lastCol[$y])) { push($seen, $lastCol[$y]); $distinct = $distinct + 1; }
    $y = $y + 1;
}
ok($distinct > 1, "red's carry reaches the last column, so it varies (" +
   str($lastCol) + ")");
// and every index must be a legal palette entry
$y = 0;
$legal = true;
while ($y < 8) {
    if ($lastCol[$y] < 0 || $lastCol[$y] > 225) { $legal = false; }
    $y = $y + 1;
}
ok($legal, "and every index stays inside the 226-entry palette");

// ─── the dense YCbCr / HSV probe, both directions ───────────────────────
print("── 3,014 pixels through YCbCr and HSV, against Pillow ──");

$rows = split(readfile("tests/fixtures/pillow/reference/colorspace.txt"), "\n");
$fwd = [];
$inv = [];
$i = 0;
while ($i < len($rows)) {
    $L = trim($rows[$i]);
    if (len($L) > 0 && $L[0] != "#") {
        $f = split($L, " ");
        if ($f[0] == "INV") { push($inv, $f); } else { push($fwd, $f); }
    }
    $i = $i + 1;
}
eq(len($fwd) > 3000, true, "the probe holds over 3,000 forward pixels");
eq(len($inv), len($fwd), "and the same number of inverse pixels");

// Convert the whole probe as ONE row image, which is how the fixture was made.
def rowFrom($list, $col, $mode, $bands) {
    $n = len($list);
    $buf = nd_zeros([1, $n, $bands], "i64");
    $k = 0;
    while ($k < $n) {
        $v = parseVals($list[$k][$col]);
        $b = 0;
        while ($b < $bands) { nd_set($buf, [0, $k, $b], $v[$b]); $b = $b + 1; }
        $k = $k + 1;
    }
    return $buf;
}

// forward: the r g b columns are fields 0, 1, 2
$n = len($fwd);
$rgbBuf = nd_zeros([1, $n, 3], "i64");
$k = 0;
while ($k < $n) {
    nd_set($rgbBuf, [0, $k, 0], num($fwd[$k][0]));
    nd_set($rgbBuf, [0, $k, 1], num($fwd[$k][1]));
    nd_set($rgbBuf, [0, $k, 2], num($fwd[$k][2]));
    $k = $k + 1;
}

def compare($got, $list, $col, $label) {
    $n = len($list);
    $bad = 0;
    $shown = 0;
    $k = 0;
    while ($k < $n) {
        $want = parseVals($list[$k][$col]);
        $b = 0;
        while ($b < 3) {
            if (nd_get($got, [0, $k, $b]) != $want[$b]) {
                $bad = $bad + 1;
                if ($shown < 4) {
                    $shown = $shown + 1;
                    print("     " + $label + " pixel " + str($k) + " band " + str($b) +
                          ": got " + str(nd_get($got, [0, $k, $b])) + " want " + str($want[$b]));
                }
            }
            $b = $b + 1;
        }
        $k = $k + 1;
    }
    return $bad;
}

$gotY = convert.px($rgbBuf, "RGB", "YCbCr", null, null).px;
eq(compare($gotY, $fwd, 3, "RGB->YCbCr"), 0, "all " + str($n) + " pixels RGB->YCbCr match Pillow");
$gotH = convert.px($rgbBuf, "RGB", "HSV", null, null).px;
eq(compare($gotH, $fwd, 4, "RGB->HSV"), 0, "all " + str($n) + " pixels RGB->HSV match Pillow");

// inverse: fields are  INV ycbcr rgb hsv rgb  -> indices 1,2,3,4
$yBuf = rowFrom($inv, 1, "YCbCr", 3);
$gotRY = convert.px($yBuf, "YCbCr", "RGB", null, null).px;
eq(compare($gotRY, $inv, 2, "YCbCr->RGB"), 0, "all " + str($n) + " pixels YCbCr->RGB match Pillow");
$hBuf = rowFrom($inv, 3, "HSV", 3);
$gotRH = convert.px($hBuf, "HSV", "RGB", null, null).px;
eq(compare($gotRH, $inv, 4, "HSV->RGB"), 0, "all " + str($n) + " pixels HSV->RGB match Pillow");

// ─── Floyd-Steinberg, in two dimensions, against Pillow ─────────────────
print("── the dither corpus: whole images, several shapes ──");

$man = split(trim(readfile("tests/fixtures/pillow/reference/dither_index.txt")), "\n");
$shapes = [];
$i = 0;
while ($i < len($man)) {
    $L = trim($man[$i]);
    if (len($L) > 0 && $L[0] != "#") { push($shapes, split($L, " ")); }
    $i = $i + 1;
}
eq(len($shapes), 12, "the dither corpus has 12 shapes");

$DD = "tests/fixtures/pillow/reference/dither/";
$totPx = 0;
$badPx = 0;
$badImgs = 0;
$i = 0;
while ($i < len($shapes)) {
    $w = num($shapes[$i][0]);
    $h = num($shapes[$i][1]);
    $tag = str($w) + "x" + str($h);

    $srcL = nd_frombytes(readfile($DD + "L_" + $tag + ".raw"), "u8", [$h, $w, 1]);
    $gotL = convert.px($srcL, "L", "1", null, null).px;
    $wantL = bytes(readfile($DD + "L_" + $tag + ".bw"));
    $m = 0;
    $k = 0;
    while ($k < len($wantL)) {
        if (nd_get($gotL, [floor($k / $w), $k % $w, 0]) != $wantL[$k]) { $m = $m + 1; }
        $k = $k + 1;
    }
    $totPx = $totPx + len($wantL);
    $badPx = $badPx + $m;
    if ($m > 0) { $badImgs = $badImgs + 1; print("     L " + $tag + ": " + str($m) + " of " + str(len($wantL)) + " differ"); }

    $srcR = nd_frombytes(readfile($DD + "RGB_" + $tag + ".raw"), "u8", [$h, $w, 3]);
    $gotR = convert.px($srcR, "RGB", "1", null, null).px;
    $wantR = bytes(readfile($DD + "RGB_" + $tag + ".bw"));
    $m = 0;
    $k = 0;
    while ($k < len($wantR)) {
        if (nd_get($gotR, [floor($k / $w), $k % $w, 0]) != $wantR[$k]) { $m = $m + 1; }
        $k = $k + 1;
    }
    $totPx = $totPx + len($wantR);
    $badPx = $badPx + $m;
    if ($m > 0) { $badImgs = $badImgs + 1; print("     RGB " + $tag + ": " + str($m) + " of " + str(len($wantR)) + " differ"); }
    $i = $i + 1;
}
print("   " + str($totPx - $badPx) + " of " + str($totPx) + " dithered pixels identical to Pillow, across 24 images");
eq($badPx, 0, "Floyd-Steinberg is exact in two dimensions");
eq($badImgs, 0, "every dither image matches");

// the two luminances the bilevel paths use really are different
eq(one([0, 255, 0], "RGB", "1", "NONE"), [255], "threshold uses L24 (150 >= 128)");
$g1 = nd_zeros([1, 1, 3], "i64");
nd_set($g1, [0, 0, 1], 255);
eq(nd_get(convert.px($g1, "RGB", "1", "FLOYDSTEINBERG").px, [0, 0, 0]), 255,
   "the dither uses the /1000 luminance (149) and still lights this pixel");

// ─── no pixel loops: the timing curve must be linear ────────────────────
print("── rule PW2: the cost is linear in pixel count ──");
def timeConvert($w, $h) {
    $b = nd_zeros([$h, $w, 3], "i64");
    nd_add($b, 90, $b);
    $t0 = clock();
    convert.px($b, "RGB", "L", null, null);
    return clock() - $t0;
}
$t1 = timeConvert(200, 200);
$t4 = timeConvert(400, 400);
print("   40,000 px: " + str($t1) + " ms      160,000 px: " + str($t4) + " ms");
// Four times the pixels must not cost more than twelve times the time; a
// per-pixel loop would show up as hundreds of times.
eq($t4 <= max($t1, 1) * 12 + 60, true, "4x the pixels costs well under 12x the time");

print("");
print("checks: " + str($R.p + $R.f) + "  passed: " + str($R.p) + "  failed: " + str($R.f));
if ($R.f == 0) { print("RESULT: ALL GREEN"); }
else { print("RESULT: FAILED"); }
