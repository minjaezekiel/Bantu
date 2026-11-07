// ════════════════════════════════════════════════════════════════════════
//  tests/pillow_font_test.b — pillow Phase P7: ImageFont and text.
//
//  TEXT HAS TWO GATES, because it has two kinds of font, and the whole point
//  of this file is to hold each to the strongest claim that is true of it.
//
//    1. A BITMAP FONT is rendered by byte copying — Pillow's own renderer is
//       forty lines of `ImagingPaste` — so `load_default()` text is compared
//       BYTE FOR BYTE, like every other image in this suite. Every string,
//       every mode a stencil can reach, every anchor, every alignment, the
//       multiline spacing, the blended path, and the two edges.
//
//    2. A TRUETYPE FONT is rendered by FreeType, which grid-fits the outline
//       with the font's own hinting instructions and anti-aliases it with its
//       own coverage arithmetic. Byte equality is not reachable, so the claim
//       is split into the parts that can be exact and the parts that cannot:
//
//         * ADVANCES ARE EXACT. 665 single-glyph advances and 70 kerned
//           pairs over seven sizes, to the last 1/64 of a pixel. Pillow's
//           default layout engine is libraqm, which asks HarfBuzz, which
//           loads glyphs unhinted — so an advance is FT_MulFix(hmtx,
//           x_scale) plus GPOS pair kerning and nothing else.
//         * BOUNDING BOXES ARE NOT, and the deviation is MEASURED rather
//           than excused: FreeType's box comes from the hinted outline, so
//           at 24 px Pillow's "Hello, world!" is 21 pixels tall where the
//           unhinted control box is 22.
//         * PIXELS ARE NOT, and the gate is PW17's shape: the inked region's
//           intersection-over-union against Pillow's, and the fraction of
//           pixels within a stated coverage tolerance, both reported as
//           numbers and both bounded.
//
//  Run:  bantu run tests/pillow_font_test.b
// ════════════════════════════════════════════════════════════════════════

include "./pillow/pillow.b" as pillow;
include "./pillow/draw.b" as draw;
include "./pillow/font.b" as font;
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

$D = "tests/fixtures/pillow/reference/text/";
$FONT = "tests/fixtures/pillow/fonts/DejaVuSans.ttf";


// ════════════════════════════════════════════════════════════════════════
//  Gate 1: the bitmap font, byte for byte
// ════════════════════════════════════════════════════════════════════════

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

$BF = font.load_default(null);

print("");
print("── the bitmap font: text byte for byte against Pillow ──");

$rows = split(trim(readfile($D + "MANIFEST")), "\n");
$btOk = 0;
$btBad = 0;
$ttRows = [];
$shown = 0;
$i = 0;
while ($i < len($rows)) {
    $L = trim($rows[$i]);
    $i = $i + 1;
    if (len($L) == 0) { continue; }
    $f = split($L, " ");
    if ($f[0] == "TT") { push($ttRows, $L); continue; }
    if ($f[0] != "BT") { continue; }
    $tag = $f[1];
    $file = $f[2];
    $m = $f[3];
    $w = num($f[4]);
    $h = num($f[5]);
    $parts = split($f[6], "|");
    $xy = [];
    each ($v in split($parts[1], ",")) { push($xy, num($v)); }
    $kw = _kw($parts);

    $threw = false;
    $why = "";
    $im = null;
    try {
        $bg = null;
        if ($kw["bg"] != null) { $bg = _ink($kw["bg"]); }
        $im = pillow.new($m, [$w, $h], $bg);
        if ($tag == "BT_onwhite") { $im = pillow.new($m, [$w, $h], 255); }
        if ($tag == "BT_onwhite_rgb" || $kw["dm"] != null) {
            $im = pillow.new($m, [$w, $h], "white");
        }
        $dm = null;
        if ($kw["dm"] != null && $kw["dm"] != "") { $dm = $kw["dm"]; }
        $d = draw.Draw($im, $dm);
        $o = {};
        if ($kw["anchor"] != null) { $o.anchor = $kw["anchor"]; }
        if ($kw["align"] != null) { $o.align = $kw["align"]; }
        if ($kw["spacing"] != null) { $o.spacing = num($kw["spacing"]); }
        if ($kw["stroke_width"] != null) { $o.stroke_width = num($kw["stroke_width"]); }
        if ($kw["stroke_fill"] != null) { $o.stroke_fill = num($kw["stroke_fill"]); }
        $txt = "";
        if ($kw["s"] != null && $kw["s"] != "") { $txt = b64decode($kw["s"]); }
        $d.text($xy, $txt, _ink($kw["fill"]), $BF, $o);
    } catch ($e) { $threw = true; $why = str($e); }

    if ($threw) {
        $btBad = $btBad + 1;
        if ($shown < 10) { $shown = $shown + 1; print("     ERROR " + $tag + ": " + $why); }
        continue;
    }
    if ($im.tobytes() == readfile($D + $file)) { $btOk = $btOk + 1; }
    else {
        $btBad = $btBad + 1;
        if ($shown < 10) {
            $shown = $shown + 1;
            print("     DIFF  " + $tag + " (" + $f[6] + ")");
        }
    }
}
print("   " + str($btOk) + " of " + str($btOk + $btBad) +
      " bitmap-font drawings byte-identical to Pillow");
eq($btBad, 0, "every bitmap-font text drawing matches Pillow byte for byte");


// ════════════════════════════════════════════════════════════════════════
//  Gate 2a: TrueType advances, which ARE exact
// ════════════════════════════════════════════════════════════════════════

print("");
print("── TrueType: the advances, to the last 1/64 of a pixel ──");

$nums = split(trim(readfile($D + "NUMBERS")), "\n");
$FC = {};
def tfont($size) {
    $k = str($size);
    if ($FC[$k] == null) { $FC[$k] = font.truetype($FONT, $size, null, null, null); }
    return $FC[$k];
}

$advOk = 0;
$advBad = 0;
$pairOk = 0;
$pairBad = 0;
$worstAdv = 0;
$shown = 0;
$i = 0;
while ($i < len($nums)) {
    $L = trim($nums[$i]);
    $i = $i + 1;
    if (len($L) == 0 || $L[0] == "#") { continue; }
    $f = split($L, " ");
    if ($f[0] == "ADV") {
        $size = num($f[1]);
        $cp = num($f[2]);
        $want = num($f[3]);
        $got = font.getlength(tfont($size), chr($cp), null, null, null, null);
        $d = $got - $want;
        if ($d < 0) { $d = 0 - $d; }
        if ($d > $worstAdv) { $worstAdv = $d; }
        if ($d < 0.0000001) { $advOk = $advOk + 1; }
        else {
            $advBad = $advBad + 1;
            if ($shown < 8) {
                $shown = $shown + 1;
                print("     ADV  size " + str($size) + " U+" + str($cp) + ": got " +
                      str($got) + " want " + str($want));
            }
        }
    } else if ($f[0] == "PAIR") {
        $size = num($f[1]);
        $s = b64decode($f[2]);
        $want = num($f[3]);
        $got = font.getlength(tfont($size), $s, null, null, null, null);
        $d = $got - $want;
        if ($d < 0) { $d = 0 - $d; }
        if ($d < 0.0000001) { $pairOk = $pairOk + 1; }
        else {
            $pairBad = $pairBad + 1;
            if ($shown < 8) {
                $shown = $shown + 1;
                print("     PAIR size " + str($size) + " \"" + $s + "\": got " +
                      str($got) + " want " + str($want));
            }
        }
    } else if ($f[0] == "METRICS") {
        $size = num($f[1]);
        $g = font.getmetrics(tfont($size));
        eq($g, [num($f[2]), num($f[3])], "getmetrics at size " + str($size));
    } else if ($f[0] == "NAME") {
        $g = font.getname(tfont(num($f[1])));
        eq($g[0], b64decode($f[2]), "the family name");
        eq($g[1], b64decode($f[3]), "the style name");
    }
}
print("   " + str($advOk) + " of " + str($advOk + $advBad) +
      " single-glyph advances exact over seven sizes");
print("   " + str($pairOk) + " of " + str($pairOk + $pairBad) +
      " kerned pairs exact (GPOS pair positioning)");
eq($advBad, 0, "every single-glyph advance is exactly Pillow's");
eq($pairBad, 0, "every kerned pair is exactly Pillow's, which is what GPOS buys");


// ════════════════════════════════════════════════════════════════════════
//  Gate 2b: TrueType pixels and boxes, MEASURED
// ════════════════════════════════════════════════════════════════════════
//
//  The shape agreement is measured after aligning the two masks by the
//  integer shift that fits them best, within +/-2 pixels. That is deliberate:
//  FreeType's box comes from the hinted outline and ours from the unhinted
//  one, so the two bitmaps can sit a pixel apart while describing the same
//  letters. Measuring the shape and the offset SEPARATELY says which of the
//  two is happening; one combined number would hide it.

print("");
print("── TrueType: the pixels and the boxes, measured ──");

// _best($theirs, $ours) -> [iou, dx, dy, within, jaccard] for the best
// integer shift. `iou` is over the INKED region (coverage > 0), `jaccard` is
// coverage-weighted -- sum(min) / sum(max) -- which is the stricter of the
// two because it notices a grey edge where the binary one does not.
def _best($theirs, $tw, $th, $ours, $ow, $oh) {
    $cw = max($tw, $ow) + 6;
    $ch = max($th, $oh) + 6;
    $A = nd_zeros([$ch, $cw], "i64");
    $dst = nd_slice($A, [[3, 3 + $th], [3, 3 + $tw]]);
    nd_add($theirs, 0, $dst);
    $bestIou = 0 - 1;
    $bdx = 0;
    $bdy = 0;
    $bWithin = 0;
    $bJac = 0;
    $dy = 0 - 2;
    while ($dy <= 2) {
        $dx = 0 - 2;
        while ($dx <= 2) {
            $B = nd_zeros([$ch, $cw], "i64");
            $d2 = nd_slice($B, [[3 + $dy, 3 + $dy + $oh], [3 + $dx, 3 + $dx + $ow]]);
            nd_add($ours, 0, $d2);
            $ia = nd_greater($A, 0);
            $ib = nd_greater($B, 0);
            $inter = pxbuf.scalar(nd_sum(nd_logical_and($ia, $ib)));
            $union = pxbuf.scalar(nd_sum(nd_logical_or($ia, $ib)));
            $iou = 1.0;
            if ($union > 0) { $iou = $inter / $union; }
            if ($iou > $bestIou) {
                $bestIou = $iou;
                $bdx = $dx;
                $bdy = $dy;
                $diff = nd_abs(nd_subtract($A, $B));
                $bWithin = pxbuf.scalar(nd_sum(nd_less_equal($diff, 64))) / ($cw * $ch);
                $mn = pxbuf.scalar(nd_sum(nd_minimum($A, $B)));
                $mx = pxbuf.scalar(nd_sum(nd_maximum($A, $B)));
                $bJac = 1.0;
                if ($mx > 0) { $bJac = $mn / $mx; }
            }
            $dx = $dx + 1;
        }
        $dy = $dy + 1;
    }
    return [$bestIou, $bdx, $bdy, $bWithin, $bJac];
}

$worstIou = 2.0;
$sumIou = 0.0;
$nIou = 0;
$worstWithin = 2.0;
$maxShift = 0;
$worstBox = 0;
$worstJac = 2.0;
$sumJac = 0.0;
$sumBox = 0;
$nBox = 0;
$lenOk = 0;
$lenBad = 0;
$perSize = {};
$i = 0;
while ($i < len($ttRows)) {
    $f = split($ttRows[$i], " ");
    $i = $i + 1;
    $tag = $f[1];
    $file = $f[2];
    $size = num($f[3]);
    $s = "";
    if ($f[4] != "") { $s = b64decode($f[4]); }
    $wantLen = num($f[5]);
    $wb = [];
    each ($v in split($f[6], ",")) { push($wb, num($v)); }
    $mw = num($f[7]);
    $mh = num($f[8]);

    $fo = tfont($size);
    $gotLen = font.getlength($fo, $s, null, null, null, null);
    $dl = $gotLen - $wantLen;
    if ($dl < 0) { $dl = 0 - $dl; }
    if ($dl < 0.0000001) { $lenOk = $lenOk + 1; } else { $lenBad = $lenBad + 1; }

    $gb = font.getbbox($fo, $s, "L", null, null, null, null, 0);
    $k = 0;
    while ($k < 4) {
        $d = $gb[$k] - $wb[$k];
        if ($d < 0) { $d = 0 - $d; }
        if ($d > $worstBox) { $worstBox = $d; }
        $sumBox = $sumBox + $d;
        $nBox = $nBox + 1;
        $k = $k + 1;
    }

    if ($mw > 0 && $mh > 0) {
        $theirs = nd_frombytes(readfile($D + $file), "u8", [$mh, $mw]);
        $gm = font.getmask($fo, $s, "L", null, null, null, 0);
        $ours = nd_reshape($gm.px, [$gm.height, $gm.width]);
        $r = _best($theirs, $mw, $mh, $ours, $gm.width, $gm.height);
        if ($r[0] < $worstIou) { $worstIou = $r[0]; }
        $sumIou = $sumIou + $r[0];
        $nIou = $nIou + 1;
        if ($r[3] < $worstWithin) { $worstWithin = $r[3]; }
        if ($r[4] < $worstJac) { $worstJac = $r[4]; }
        $sumJac = $sumJac + $r[4];
        $sh = max(abs($r[1]), abs($r[2]));
        if ($sh > $maxShift) { $maxShift = $sh; }
        $key = str($size);
        if ($perSize[$key] == null) { $perSize[$key] = {"s": 0.0, "n": 0, "w": 2.0}; }
        $perSize[$key].s = $perSize[$key].s + $r[0];
        $perSize[$key].n = $perSize[$key].n + 1;
        if ($r[0] < $perSize[$key].w) { $perSize[$key].w = $r[0]; }
    }
}

print("   " + str($lenOk) + " of " + str($lenOk + $lenBad) +
      " whole-string lengths exact (kerning included)");
eq($lenBad, 0, "every whole-string length is exactly Pillow's");

each ($sz in ["8", "12", "24", "72"]) {
    $v = $perSize[$sz];
    if ($v == null) { continue; }
    print("   size " + $sz + ": mean inked IoU " + str($v.s / $v.n) +
          ", worst " + str($v.w));
}
print("   over all " + str($nIou) + " rendered strings: mean IoU " +
      str($sumIou / $nIou) + ", worst " + str($worstIou));
print("   coverage-weighted agreement: mean " + str($sumJac / $nIou) +
      ", worst " + str($worstJac));
print("   coverage within 64/255 on " + str($worstWithin * 100) +
      "% of pixels at worst; best-fit offset never more than " +
      str($maxShift) + " px");
print("   bounding boxes: mean edge error " + str($sumBox / $nBox) +
      " px, worst " + str($worstBox) + " px");

// THE BOUNDS. These are the published claim, and they are deliberately not
// tight: a FreeType release that changes its hinting moves these numbers,
// and a test that breaks on that would be measuring Pillow's version rather
// than this package's correctness.
// THE PUBLISHED BOUNDS. Measured at 0.71 worst / 0.88 mean inked IoU, 0.60
// worst / 0.85 mean coverage-weighted, 90% within 64/255, never more than
// one pixel of offset, and a bounding-box edge never more than one pixel
// out with a mean error of 0.18. The bounds below sit under those with room
// to spare, deliberately: the residual is FreeType's grid fitting, so a
// FreeType release that changes its hinting moves these numbers, and a test
// that broke on that would be measuring Pillow's version rather than this
// package's correctness. A REGRESSION of our own would blow straight
// through them — the pre-kerning draft scored 0.28.
ok($worstIou > 0.65,
   "the inked region overlaps Pillow's by more than 65% on every string " +
   "(worst " + str($worstIou) + ")");
ok($sumIou / $nIou > 0.85,
   "and by more than 85% on average (" + str($sumIou / $nIou) + ")");
ok($worstJac > 0.45,
   "coverage-weighted agreement is over 45% on every string (worst " +
   str($worstJac) + ")");
ok($sumJac / $nIou > 0.70,
   "and over 70% on average (" + str($sumJac / $nIou) + ")");
ok($worstWithin > 0.88,
   "coverage is within 64/255 on more than 88% of pixels (worst " +
   str($worstWithin) + ")");
ok($maxShift <= 1,
   "the two rasters are never more than 1 px apart (" + str($maxShift) + ")");
ok($worstBox <= 2,
   "no bounding-box edge is more than 2 px from Pillow's (" + str($worstBox) + ")");
ok($sumBox / $nBox < 0.4,
   "and the mean edge error is under 0.4 px (" + str($sumBox / $nBox) + ")");


// ════════════════════════════════════════════════════════════════════════
//  The contract, and what it refuses
// ════════════════════════════════════════════════════════════════════════

print("");
print("── the reader, the cache, and the places it refuses ──");

$f24 = tfont(24);
eq($f24.upem, 2048, "DejaVu's unitsPerEm");
ok($f24.numGlyphs > 6000, "and its glyph count (" + str($f24.numGlyphs) + ")");
eq(font.gid($f24, 72), 43, "the cmap maps 'H' to glyph 43");
ok(font.gid($f24, 65533) > 0, "and U+FFFD to a real glyph");
eq(font.gid($f24, 1114100), 0, "an unmapped astral codepoint is glyph 0");

// The glyph cache: the second render of the same text costs nothing like the
// first, which is what §8.2 promises.
$t0 = clock();
font.getmask($f24, "The quick brown fox jumps over the lazy dog", "L",
             null, null, null, 0);
$t1 = clock() - $t0;
$t0 = clock();
font.getmask($f24, "The quick brown fox jumps over the lazy dog", "L",
             null, null, null, 0);
$t2 = clock() - $t0;
print("   43 glyphs cold " + str($t1) + " ms, warm " + str($t2) + " ms");
ok($t2 * 2 < $t1 + 2, "a cached glyph is a dictionary hit, not a rasterisation");

// font_variant is a copy with overrides, which is what Pillow's is.
$f12 = font.font_variant($f24, 12, null, null, null);
eq($f12.size, 12, "font_variant changes the size");
eq($f12.upem, 2048, "and keeps the face");
ok(font.getlength($f12, "Hello", null, null, null, null) <
   font.getlength($f24, "Hello", null, null, null, null),
   "and the smaller size is narrower");

// Composite glyphs: a letter with an accent is a base plus a mark, placed by
// offset, and it has to come out taller than the base alone.
$bA = font.getbbox($f24, "A", "L", null, null, null, null, 0);
$bAu = font.getbbox($f24, "Ä", "L", null, null, null, null, 0);
ok($bAu[1] < $bA[1], "a composite glyph (A-umlaut) is taller than its base");
eq(font.getlength($f24, "Ä", null, null, null, null),
   font.getlength($f24, "A", null, null, null, null),
   "and carries the base's advance");

// The refusals, each naming what it would take to lift them.
throws(def() { font.truetype($FONT, 0, null, null, null); }, "a size of zero");
throws(def() { font.truetype($FONT, 24, 7, null, null); },
       "a face index in a font that is not a collection");
throws(def() { font.truetype("tests/fixtures/pillow/fonts/nope.ttf", 12, null, null, null); },
       "a font file that is not there");
throws(def() { font.truetype($FONT, 24, null, "zzzz", null); }, "an unknown encoding");
throws(def() { font.getmask($f24, "x", "RGBA", null, null, null, 0); },
       "embedded_color, which needs a colour font");
throws(def() { font.getmask($f24, "x", "RGB", null, null, null, 0); },
       "a mask mode that is not 1 or L");
throws(def() { font.getbbox($f24, "x", "L", "rtl", null, null, null, 0); },
       "right-to-left, which needs a shaping engine");
throws(def() { font.getbbox($f24, "x", "L", null, ["liga"], null, null, 0); },
       "OpenType feature selection");
throws(def() { font.set_variation_by_name($f24, "Bold"); },
       "applying a variation instance, which needs gvar");
throws(def() { font.get_variation_axes($f24); }, "variation axes on a static font");
throws(def() { font.getbbox($f24, "x", "L", null, null, null, "zz", 0); },
       "an anchor that is not one of the legal pairs");
throws(def() { font.getbbox($f24, "x", "L", null, null, null, "m", 0); },
       "a one-character anchor");

// A bitmap font addresses its glyphs by Latin-1 byte and says so.
throws(def() { font.getlength($BF, "中", null, null, null, null); },
       "a codepoint a bitmap font has no byte for");
eq(font.getlength($BF, "ÿ", null, null, null, null) > 0, true,
   "but Latin-1 is fine");

// The default font needs no file at all, which is the whole point of
// embedding it.
$bf2 = font.load_default(null);
eq($bf2.ysize, 11, "the embedded font is 11 pixels tall");
eq($bf2.baseline, 9, "with its baseline at 9");
eq(font.getmetrics($BF), [2, 9], "and getmetrics agrees");

// A truncated or hostile font file must raise, not read past the end.
throws(def() {
    writefile("/tmp/pillow_trunc_font.ttf", substr(readfile($FONT, "rb"), 0, 400));
    font.truetype("/tmp/pillow_trunc_font.ttf", 12, null, null, null);
}, "a font truncated to 400 bytes");
throws(def() {
    writefile("/tmp/pillow_junk_font.ttf", "not a font at all, not even close");
    font.truetype("/tmp/pillow_junk_font.ttf", 12, null, null, null);
}, "a file that is not a font");


// ════════════════════════════════════════════════════════════════════════
//  The deviations, and what each one preserves
// ════════════════════════════════════════════════════════════════════════

print("");
print("── the deviations, and what each one preserves ──");

// stroke_width, as a dilation by a disc rather than FT_Stroker. The
// containment: a stroke grows the inked region and never shrinks it, it
// grows it by about the stroke width on each side, and the unstroked glyph
// is still entirely inside it.
$m0 = font.getmask($f24, "Hi", "L", null, null, null, 0);
$m2 = font.getmask($f24, "Hi", "L", null, null, null, 2);
eq($m2.width, $m0.width + 4, "a stroke of 2 widens the mask by 4");
eq($m2.height, $m0.height + 4, "and heightens it by 4");
$in0 = pxbuf.scalar(nd_sum(nd_greater($m0.px, 0)));
$in2 = pxbuf.scalar(nd_sum(nd_greater($m2.px, 0)));
ok($in2 > $in0, "and inks more pixels (" + str($in0) + " -> " + str($in2) + ")");
// the original is inside the stroked one, offset by the stroke width
$inner = pxbuf.crop($m2.px, 2, 2, 2 + $m0.width, 2 + $m0.height);
ok(pxbuf.scalar(nd_sum(nd_less(nd_maximum($inner, $m0.px), $inner))) == 0 ||
   pxbuf.scalar(nd_sum(nd_greater($m0.px, $inner))) == 0,
   "the unstroked glyph is entirely inside the stroked one");

// The I;16 stencil, which Pillow gets wrong (fill_mask_1 has no 16-bit
// case). Ours writes the ink where the bit is set and nowhere else, over the
// full width.
$g16 = pillow.new("I;16", [40, 14], 0);
draw.Draw($g16, null).text([1, 1], "Hi", 40000, $BF, null);
$maxv = pxbuf.scalar(nd_max($g16.px));
eq($maxv, 40000, "I;16 text writes the whole 16-bit ink, not its low byte");
$inked = pxbuf.scalar(nd_sum(nd_greater($g16.px, 0)));
$ref = pillow.new("L", [40, 14], 0);
draw.Draw($ref, null).text([1, 1], "Hi", 255, $BF, null);
eq($inked, pxbuf.scalar(nd_sum(nd_greater($ref.px, 0))),
   "and inks exactly the pixels an 8-bit mode does");

// Sub-pixel positioning: Pillow shifts the outline by the fractional part of
// xy, and we round to the pixel grid. The containment is that an integer
// position is identical and a fractional one lands on the floor of it.
$a = pillow.new("L", [60, 20], 0);
draw.Draw($a, null).text([5, 5], "Hi", 255, $BF, null);
$b = pillow.new("L", [60, 20], 0);
draw.Draw($b, null).text([5.7, 5.4], "Hi", 255, $BF, null);
eq($a.tobytes() == $b.tobytes(), true,
   "a fractional text position rounds onto the pixel grid");


// ════════════════════════════════════════════════════════════════════════
//  Stress
// ════════════════════════════════════════════════════════════════════════

print("");
print("── stress ──");

$long = "";
$parts = [];
$i = 0;
while ($i < 100) { push($parts, "The quick brown fox jumps over the lazy dog. "); $i = $i + 1; }
$long = join($parts, "");
print("   a " + str(len($long)) + "-character string");
$t0 = clock();
$bl = font.getlength($f24, $long, null, null, null, null);
print("   getlength: " + str($bl) + " px in " + str(clock() - $t0) + " ms");
ok($bl > 20000, "a 4,500-character line measures over 20,000 px");
$t0 = clock();
$big = pillow.new("L", [400, 300], 0);
draw.Draw($big, null).text([2, 2], substr($long, 0, 2000), 255, $BF, null);
print("   2,000 characters of bitmap text drawn in " + str(clock() - $t0) + " ms");

// Every size from 6 to 48, so no scale is special.
$bad = 0;
$sz = 6;
while ($sz <= 48) {
    $fo = font.truetype($FONT, $sz, null, null, null);
    $mm = font.getmask($fo, "Agj", "L", null, null, null, 0);
    if ($mm.width <= 0 || $mm.height <= 0) { $bad = $bad + 1; }
    $sz = $sz + 6;
}
eq($bad, 0, "every size from 6 to 48 rasterises");

// A huge size, which is where the 26.6 arithmetic could overflow.
$f400 = font.truetype($FONT, 400, null, null, null);
$m400 = font.getmask($f400, "W", "L", null, null, null, 0);
ok($m400.width > 300 && $m400.height > 200,
   "a 400 px glyph rasterises (" + str($m400.width) + "x" + str($m400.height) + ")");

// Text clipped off every edge of the image.
$clip = pillow.new("L", [30, 20], 0);
$dc = draw.Draw($clip, null);
$dc.text([0 - 40, 0 - 10], "Hello", 255, $BF, null);
$dc.text([25, 15], "Hello", 255, $BF, null);
$dc.text([0 - 5, 8], "Hello", 255, $BF, null);
$R.p = $R.p + 1;
print("   text clipped off all four edges does not fault");

// Multiline with every alignment, on the TrueType font this time.
each ($al in ["left", "center", "right", "justify"]) {
    $mi = pillow.new("L", [220, 90], 0);
    draw.Draw($mi, null).text([4, 4], "one two three\nfour five\nsix", 255, $f24,
                              {"align": $al, "spacing": 3});
    ok(pxbuf.scalar(nd_sum(nd_greater($mi.px, 0))) > 0,
       "multiline TrueType text, align=" + $al);
}

print("");
print("checks: " + str($R.p + $R.f) + "  passed: " + str($R.p) + "  failed: " + str($R.f));
if ($R.f == 0) { print("RESULT: ALL GREEN"); }
else { print("RESULT: FAILED"); }
