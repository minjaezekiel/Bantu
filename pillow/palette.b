// ════════════════════════════════════════════════════════════════════════
//  pillow/palette.b — ImagePalette, and the RGB -> P mapping.
//
//      include "./palette.b" as pal;
//      $p = pal.web();                  // Pillow's default for convert("P")
//      $p = pal.grey_ramp();            // the 256-entry grey ramp
//      pal.rgb_to_index($rgb, $p, $dither)
//
//  WHAT PILLOW DOES, from Convert.c's topalette and Palette.c:
//
//    * a ONE-BAND source (L, and via basemode also 1, I, F, LA) is copied
//      index for index, with the 256-entry grey ramp as the palette. There
//      is no colour search at all: L 137 becomes index 137.
//    * a COLOUR source gets ImagingPaletteNewBrowser() -- ten reserved zero
//      entries followed by a 6x6x6 colour cube, 226 entries in all -- and
//      each pixel is mapped to the nearest entry, by default WITH
//      Floyd-Steinberg dithering.
//    * an RGBA source is different: Pillow routes it to quantize(), an
//      ADAPTIVE median-cut palette, not the fixed cube. quantize() is a
//      feature of its own and is not in this build; convert() says so by
//      name rather than quietly returning cube indices that would look
//      almost right.
//
//  THREE DETAILS THAT DECIDE WHETHER THE OUTPUT MATCHES, all read out of
//  the C rather than guessed:
//
//    1. The nearest-entry search runs against the colour QUANTISED TO A
//       MULTIPLE OF 4 per channel. Palette.c caches by (r>>2, g>>2, b>>2)
//       and computes each slot's distance at the slot's lower corner, so
//       r = 7 is searched as if it were 4.
//    2. Ties go to the LOWEST index, because the scan is `bd < d[j]` with i
//       ascending. The cube's first entry is (0,0,0) and so are the ten
//       reserved entries, so black maps to index 0, not index 10.
//    3. The cube is separable -- levels 0, 51, 102, 153, 204, 255 on each
//       axis -- so the Euclidean nearest entry IS the per-channel nearest
//       level, and the whole 226-entry search collapses to three integer
//       divisions. That is what makes it vectorisable: a general search is
//       226 passes over the image, the closed form is three.
//
//  ─── ONE PLACE THIS DELIBERATELY DOES NOT MATCH PILLOW ───────────────
//
//  Convert.c's dithered topalette ends each row with
//
//      e[0] = b0;  e[1] = b1;  e[2] = b2;
//
//  writing the BLUE accumulators into all three channel slots, where the
//  single-channel tobilevel correctly writes its own. The red and green
//  accumulators are computed and then thrown away, and the next row's last
//  column reads blue's error as if it were red's and green's. That is a
//  copy-paste slip, not a design choice -- nothing downstream depends on
//  it and it corrupts the final column of every row after the first.
//
//  This module writes r0, g0, b0. The consequence is stated rather than
//  hidden: on a dithered colour image TALLER THAN ONE ROW, the last column
//  can differ from Pillow by one palette step. Everything else is identical,
//  single-row images are identical, and the output here is the one Floyd
//  and Steinberg describe. tests/pillow_convert_test.b asserts our value,
//  not Pillow's, and says why.
// ════════════════════════════════════════════════════════════════════════

include "./pxbuf.b" as pxbuf;

$CUBE_STEP = 51;
$CUBE_LEVELS = 6;
$RESERVED = 10;
$WEB_SIZE = 226;                    // 10 reserved + 6*6*6

// Built palettes are cached: web() is 648 nd_set calls, and rgb_to_index
// would otherwise pay that on every conversion. A dict, because a function
// assigning a bare $var would bind a function-local.
$_CACHE = {"web": null, "grey": null, "webList": null};


// ─── Building palettes ──────────────────────────────────────────────────

// web() -> Pillow's ImagingPaletteNewBrowser as a 256x3 i64 buffer.
//
// Entries 0..9 are zero, 10..225 are the cube in Palette.c's own loop order
// (blue outermost, then green, then red), and 226..255 are zero because a
// palette here is always 256 long so the gather needs no bounds test.
def web() {
    if ($_CACHE.web != null) { return $_CACHE.web; }
    $p = nd_zeros([256, 3], "i64");
    $i = $RESERVED;
    $b = 0;
    while ($b < 256) {
        $g = 0;
        while ($g < 256) {
            $r = 0;
            while ($r < 256) {
                nd_set($p, [$i, 0], $r);
                nd_set($p, [$i, 1], $g);
                nd_set($p, [$i, 2], $b);
                $i = $i + 1;
                $r = $r + $CUBE_STEP;
            }
            $g = $g + $CUBE_STEP;
        }
        $b = $b + $CUBE_STEP;
    }
    $_CACHE.web = $p;
    return $p;
}

// grey_ramp() -> the 256-entry grey ramp: entry i is (i, i, i).
//
// NOT named grey(): Bantu keeps variables and functions in one namespace, so
// any caller with a local $grey would silently replace this function and the
// call would fail with "it holds a value that is not callable". The suffix
// costs nothing and removes a whole class of action-at-a-distance bug.
def grey_ramp() {
    if ($_CACHE.grey != null) { return $_CACHE.grey; }
    $p = nd_zeros([256, 3], "i64");
    $ramp = nd_reshape(nd_arange(0, 256, 1), [256, 1]);
    nd_add(nd_broadcast_to($ramp, [256, 3]), 0, $p);
    $_CACHE.grey = $p;
    return $p;
}

// from_list($list) -> a 256x3 buffer from a flat [r,g,b, ...] list or a list
// of triples. Short input is zero-filled, as Pillow does.
def from_list($list) {
    if (type($list) != "list") { throw "palette.from_list: expected a list"; }
    $p = nd_zeros([256, 3], "i64");
    if (len($list) == 0) { return $p; }
    if (type($list[0]) == "list") {
        $n = len($list);
        if ($n > 256) { $n = 256; }
        $i = 0;
        while ($i < $n) {
            if (len($list[$i]) < 3) {
                throw "palette.from_list: entry " + str($i) + " has fewer than 3 channels";
            }
            $k = 0;
            while ($k < 3) { nd_set($p, [$i, $k], $list[$i][$k]); $k = $k + 1; }
            $i = $i + 1;
        }
        return $p;
    }
    if (len($list) % 3 != 0) {
        throw "palette.from_list: a flat palette must be a multiple of 3 long, got " +
              str(len($list));
    }
    $n = floor(len($list) / 3);
    if ($n > 256) { $n = 256; }
    $i = 0;
    while ($i < $n) {
        $k = 0;
        while ($k < 3) { nd_set($p, [$i, $k], $list[$i * 3 + $k]); $k = $k + 1; }
        $i = $i + 1;
    }
    return $p;
}

// to_list($p) -> the flat [r,g,b, ...] list Pillow's getpalette returns.
// One nd_to_list, not 768 nd_get calls.
def to_list($p) { return nd_to_list(nd_ravel($p)); }

// to_rgb($p) -> the 256x3 buffer itself, which is what a gather wants.
def to_rgb($p) { return $p; }

// count($p) -> how many leading entries are in use, found as "the last
// non-black entry plus one".
//
// One reduction, not a 256-step nd_get walk: any() over the rows gives a
// boolean per entry and the scan is then over a 256-element Bantu list.
def count($p) {
    $nz = nd_to_list(nd_any(nd_not_equal($p, 0), 1));
    $i = 255;
    while ($i > 0) {
        if ($nz[$i]) { return $i + 1; }
        $i = $i - 1;
    }
    return 1;
}

// is_web($p) -> is this exactly the browser palette?
//
// The closed-form mapping is only valid for the cube, so the mapper asks
// before using it. Six probes pin the reserved block, the three axis steps,
// white, and the end of the cube -- enough that no other palette in practice
// passes, and cheap enough to run per conversion.
def is_web($p) {
    if ($_CACHE.webList == null) { $_CACHE.webList = to_list(web()); }
    $w = $_CACHE.webList;
    $probes = [9 * 3, 10 * 3, 11 * 3, 16 * 3 + 1, 46 * 3 + 2,
               225 * 3, 225 * 3 + 1, 225 * 3 + 2, 226 * 3];
    $got = to_list($p);
    $i = 0;
    while ($i < len($probes)) {
        $k = $probes[$i];
        if ($got[$k] != $w[$k]) { return false; }
        $i = $i + 1;
    }
    return true;
}


// ─── The closed-form cube mapping ───────────────────────────────────────
//
// For one channel, the nearest cube level to a value v is
//     L = floor((2 * (v & ~3) + 51) / 102)
// -- the `& ~3` being Palette.c's 4-step cache quantisation, and +51 / 102
// being round(vq / 51). Levels are 51 apart and quantised values are
// multiples of 4, so a value never lands exactly on a midpoint and there is
// no tie to break within a channel.
def _level($band) {
    $q = nd_subtract($band, nd_mod($band, 4));
    return nd_floor_divide(nd_add(nd_multiply($q, 2), 51), 102);
}

// cube_index($rgb) -> an (h, w, 1) buffer of palette indices. Three
// divisions over the image, whatever the palette size.
def cube_index($rgb) {
    $lr = _level(pxbuf.band($rgb, 0));
    $lg = _level(pxbuf.band($rgb, 1));
    $lb = _level(pxbuf.band($rgb, 2));
    $idx = nd_add($RESERVED, nd_add(nd_add($lr, nd_multiply($lg, 6)),
                                    nd_multiply($lb, 36)));
    // Black ties with the ten reserved entries, and the lowest index wins.
    $isBlack = nd_logical_and(nd_logical_and(nd_equal($lr, 0), nd_equal($lg, 0)),
                              nd_equal($lb, 0));
    return nd_where($isBlack, 0, $idx);
}

// _cubeOne($r, $g, $b) -> the same mapping for ONE pixel, in plain numbers.
// The dither is sequential, so it needs a scalar form.
def _cubeOne($r, $g, $b) {
    $lr = floor((2 * ($r - ($r % 4)) + 51) / 102);
    $lg = floor((2 * ($g - ($g % 4)) + 51) / 102);
    $lb = floor((2 * ($b - ($b % 4)) + 51) / 102);
    if ($lr == 0 && $lg == 0 && $lb == 0) { return 0; }
    return $RESERVED + $lr + $lg * 6 + $lb * 36;
}


// ─── A general nearest-colour search, for a custom palette ──────────────
//
// One pass over the image per palette entry, keeping a running argmin, with
// the 4-step quantisation and lowest-index tie-breaking the C uses.
//
// COST, stated because it is the one operation here that scales with the
// palette: `entries` passes over the image, and three live image-sized
// buffers (the running index, the running distance, one temporary). For the
// 226-entry cube that is 226 passes, which is why cube_index exists and why
// convert() prefers it. This path is for a small custom palette.
def nearest_index($rgb, $p, $entries) {
    $n = $entries;
    if ($n == null) { $n = count($p); }
    if ($n < 1) { throw "palette.nearest_index: the palette is empty"; }
    if ($n > 256) { $n = 256; }

    // Quantise once, outside the loop.
    $r = nd_subtract(pxbuf.band($rgb, 0), nd_mod(pxbuf.band($rgb, 0), 4));
    $g = nd_subtract(pxbuf.band($rgb, 1), nd_mod(pxbuf.band($rgb, 1), 4));
    $b = nd_subtract(pxbuf.band($rgb, 2), nd_mod(pxbuf.band($rgb, 2), 4));

    // One nd_to_list, so the loop reads plain numbers instead of paying
    // ~2.5 us per nd_get, three times per entry.
    $pl = to_list($p);

    $best = nd_zeros(nd_shape($r), "i64");
    $bestD = null;
    $i = 0;
    while ($i < $n) {
        $dr = nd_subtract($r, $pl[$i * 3]);
        $dg = nd_subtract($g, $pl[$i * 3 + 1]);
        $db = nd_subtract($b, $pl[$i * 3 + 2]);
        $d = nd_add(nd_add(nd_multiply($dr, $dr), nd_multiply($dg, $dg)),
                    nd_multiply($db, $db));
        if ($bestD == null) {
            $bestD = $d;
        } else {
            // STRICTLY less, so an equal distance keeps the lower index.
            $better = nd_less($d, $bestD);
            $best = nd_where($better, $i, $best);
            $bestD = nd_where($better, $d, $bestD);
        }
        $i = $i + 1;
    }
    return $best;
}


// ─── Floyd-Steinberg into a palette ────────────────────────────────────
//
// Convert.c's dithered topalette: tobilevel's accumulator run three times
// over, one per channel. The error is carried UNDIVIDED and divided by 16
// once at the point of use, which is the detail that makes it match.
//
// PERFORMANCE. Every per-pixel value comes from a plain Bantu list, never
// from nd_get. An earlier draft looked the chosen palette entry up with
// three nd_get calls per pixel -- about 7.5 us each, so 15 s for a 1080p
// image and 6 million native calls. Materialising the palette and the three
// source rows into lists first costs one call per row and makes the inner
// loop pure interpreter work.
//
// This is the one genuinely sequential operation in the package: a pixel
// depends on its left neighbour's error, so there is no array formulation.
// The loop is over pixels BY NECESSITY, and it is the documented slow path
// (rule PW2's stated exception), not an oversight.
def dither_to_palette($rgb, $p) {
    $d = pxbuf.dims($rgb);
    $h = $d[0];
    $w = $d[1];
    $out = nd_zeros([$h, $w, 1], "i64");
    if ($h == 0 || $w == 0) { return $out; }
    if (!is_web($p)) {
        throw "palette: dithering into a custom palette is not supported yet. " +
              "Pass dither=\"NONE\" for a nearest-colour mapping into it.";
    }
    $pl = to_list($p);

    // errors holds 3 ints per column plus one spare, and survives row to row.
    $errors = [];
    $i = 0;
    while ($i < ($w + 1) * 3) { push($errors, 0); $i = $i + 1; }

    $y = 0;
    while ($y < $h) {
        $rowR = nd_to_list(nd_reshape(nd_slice($rgb, [[$y, $y + 1], null, [0, 1]]), [$w]));
        $rowG = nd_to_list(nd_reshape(nd_slice($rgb, [[$y, $y + 1], null, [1, 2]]), [$w]));
        $rowB = nd_to_list(nd_reshape(nd_slice($rgb, [[$y, $y + 1], null, [2, 3]]), [$w]));
        $outRow = [];

        $r = 0;  $r0 = 0;  $r1 = 0;
        $g = 0;  $g0 = 0;  $g1 = 0;
        $b = 0;  $b0 = 0;  $b1 = 0;

        $x = 0;
        while ($x < $w) {
            $e = $x * 3;
            // _cdiv and _cubeOne are INLINED here, not called. A Bantu
            // function call is ~1 us and this loop runs four of them per
            // pixel, which was a fifth of the whole cost. The named
            // functions remain for every non-hot caller and for the tests.
            $t = $r + $errors[$e + 3];
            if ($t < 0) { $r = $rowR[$x] - floor((0 - $t) / 16); }
            else { $r = $rowR[$x] + floor($t / 16); }
            $t = $g + $errors[$e + 4];
            if ($t < 0) { $g = $rowG[$x] - floor((0 - $t) / 16); }
            else { $g = $rowG[$x] + floor($t / 16); }
            $t = $b + $errors[$e + 5];
            if ($t < 0) { $b = $rowB[$x] - floor((0 - $t) / 16); }
            else { $b = $rowB[$x] + floor($t / 16); }

            if ($r < 0) { $r = 0; }
            if ($r > 255) { $r = 255; }
            if ($g < 0) { $g = 0; }
            if ($g > 255) { $g = 255; }
            if ($b < 0) { $b = 0; }
            if ($b > 255) { $b = 255; }

            $lr = floor((2 * ($r - ($r % 4)) + 51) / 102);
            $lg = floor((2 * ($g - ($g % 4)) + 51) / 102);
            $lb = floor((2 * ($b - ($b % 4)) + 51) / 102);
            if ($lr == 0 && $lg == 0 && $lb == 0) { $idx = 0; }
            else { $idx = 10 + $lr + $lg * 6 + $lb * 36; }
            push($outRow, $idx);

            $r = $r - $pl[$idx * 3];
            $g = $g - $pl[$idx * 3 + 1];
            $b = $b - $pl[$idx * 3 + 2];

            $r2 = $r;  $d2 = $r + $r;  $r = $r + $d2;
            $errors[$e] = $r + $r0;
            $r = $r + $d2;  $r0 = $r + $r1;  $r1 = $r2;  $r = $r + $d2;

            $g2 = $g;  $d2 = $g + $g;  $g = $g + $d2;
            $errors[$e + 1] = $g + $g0;
            $g = $g + $d2;  $g0 = $g + $g1;  $g1 = $g2;  $g = $g + $d2;

            $b2 = $b;  $d2 = $b + $b;  $b = $b + $d2;
            $errors[$e + 2] = $b + $b0;
            $b = $b + $d2;  $b0 = $b + $b1;  $b1 = $b2;  $b = $b + $d2;

            $x = $x + 1;
        }
        // Each channel's own carry, not blue's three times over. See the
        // header: Convert.c writes b0/b1/b2 here and that is a slip.
        $errors[$w * 3]     = $r0;
        $errors[$w * 3 + 1] = $g0;
        $errors[$w * 3 + 2] = $b0;

        nd_add(nd_reshape(nd($outRow, "i64"), [1, $w, 1]), 0,
               nd_slice($out, [[$y, $y + 1], null, null]));
        $y = $y + 1;
    }
    return $out;
}

// _cdiv($a, $b) -> C integer division, truncating TOWARD ZERO, where
// floor() and nd_floor_divide round down. The dither divides a signed
// accumulator, so the two differ on every negative error.
def _cdiv($a, $b) {
    $q = $a / $b;
    if ($q < 0) { return 0 - floor(0 - $q); }
    return floor($q);
}


// ─── The entry points convert.b calls ───────────────────────────────────

// rgb_to_index($rgb, $p, $dither) -> {"idx": buffer, "palette": 256x3}
def rgb_to_index($rgb, $p, $dither) {
    $pp = $p;
    if ($pp == null) { $pp = web(); }
    $dith = $dither;
    if ($dith == null) { $dith = "FLOYDSTEINBERG"; }
    if ($dith == "NONE") {
        if (is_web($pp)) { $idx = cube_index($rgb); }
        else { $idx = nearest_index($rgb, $pp, null); }
    } else {
        $idx = dither_to_palette($rgb, $pp);
    }
    return {"idx": $idx, "palette": $pp};
}

// grey_to_index($lum) -> a one-band source becomes its own index, with the
// grey ramp as the palette. topalette copies the data as is.
def grey_to_index($lum) {
    return {"idx": nd_clip($lum, 0, 255), "palette": grey_ramp()};
}
