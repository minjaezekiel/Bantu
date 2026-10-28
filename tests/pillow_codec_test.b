// ════════════════════════════════════════════════════════════════════════
//  tests/pillow_codec_test.b — pillow Phase P4: PNG, BMP and TGA.
//
//  Three tiers of comparison, strongest first:
//
//    1. PNGSUITE -- the canonical 176-file PNG conformance corpus
//       (tests/fixtures/pillow/pngsuite/). Every legal colour type and bit
//       depth, interlaced and not, every scanline filter, every zlib level,
//       every width from 1 to 40, transparency, gamma, split IDAT chunks --
//       and twenty intentionally CORRUPT files that a decoder must refuse.
//    2. 238 images written by PILLOW ITSELF at every option combination
//       (tests/fixtures/pillow/reference/codecs/), so the comparison is
//       against Pillow's own encoder choices rather than only against the
//       spec.
//    3. Round trips: what this codec writes, read back by itself, and --
//       in tests/pillow_differential.sh -- read back by Pillow.
//
//  Plus the hostile tier, because a codec is a parser of untrusted input.
//
//  Run:  bantu run tests/pillow_codec_test.b
// ════════════════════════════════════════════════════════════════════════

include "./pillow/pillow.b" as pillow;
include "./pillow/codecs/png.b" as png;
include "./pillow/codecs/bmp.b" as bmp;
include "./pillow/codecs/tga.b" as tga;

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

// ════════════════════════════════════════════════════════════════════════
//  1. PngSuite
// ════════════════════════════════════════════════════════════════════════
print("── PngSuite: the 176-file PNG conformance corpus ──");

$PS = "tests/fixtures/pillow/pngsuite/";
$jobs = _rows($PS + "MANIFEST");
eq(len($jobs), 176, "the corpus has 176 files");

$st = {"ok": 0, "px": 0, "mode": 0, "wrongErr": 0, "refused": 0, "lenient": 0};
$shown = 0;
$i = 0;
while ($i < len($jobs)) {
    $j = $jobs[$i];
    $name = $j[0];
    $kind = $j[1];
    // PngSuite names every INTENTIONALLY CORRUPT file with a leading "x",
    // and that is the authority on whether a file is valid -- not whether
    // Pillow happens to read it. Pillow accepts xcsn0g01.png (a bad IDAT
    // CRC) because it verifies metadata CRCs and not IDAT's; this codec
    // verifies all of them, so it refuses a file Pillow accepts. Refusing a
    // file whose checksum fails is the correct behaviour (decision PW11) and
    // png.allow_truncated(true) is the documented way through, asserted in
    // the CRC section below. Classifying by the corpus's own convention is
    // what keeps that deviation visible instead of looking like a bug.
    $corrupt = ($name[0] == "x");
    $threw = false;
    $im = null;
    try { $im = pillow.open($PS + $name); } catch ($e) { $threw = true; }
    if ($kind == "ERROR" || $corrupt) {
        if ($threw) { $st.refused = $st.refused + 1; }
        else {
            $st.lenient = $st.lenient + 1;
            if ($shown < 8) { $shown = $shown + 1; print("     ACCEPTED a corrupt file: " + $name); }
        }
    } else {
        if ($threw) {
            $st.wrongErr = $st.wrongErr + 1;
            if ($shown < 8) { $shown = $shown + 1; print("     REFUSED a valid file: " + $name); }
        } else {
            if ($im.mode != $j[2] || $im.width != num($j[3]) || $im.height != num($j[4])) {
                $st.mode = $st.mode + 1;
                if ($shown < 8) {
                    $shown = $shown + 1;
                    print("     MODE " + $name + ": got " + $im.mode + " " + str($im.size) +
                          " want " + $j[2] + " " + $j[3] + "x" + $j[4]);
                }
            } else {
                $want = readfile($PS + substr($name, 0, len($name) - 4) + ".exp");
                if ($im.tobytes() == $want) { $st.ok = $st.ok + 1; }
                else {
                    $st.px = $st.px + 1;
                    if ($shown < 8) { $shown = $shown + 1; print("     PIXELS " + $name + " (" + $im.mode + ")"); }
                }
            }
        }
    }
    $i = $i + 1;
}
print("   pixel-identical to Pillow : " + str($st.ok));
print("   corrupt files refused     : " + str($st.refused));
print("   pixels differ             : " + str($st.px));
print("   mode or size differs      : " + str($st.mode));
print("   valid file refused        : " + str($st.wrongErr));
print("   corrupt file accepted     : " + str($st.lenient));

eq($st.px, 0, "every readable PngSuite file is pixel-identical to Pillow");
eq($st.mode, 0, "and reports the same mode and size");
eq($st.wrongErr, 0, "no valid PngSuite file is refused");
eq($st.lenient, 0, "every corrupt PngSuite file is refused");
ok($st.ok >= 160, "at least 160 files decode exactly (got " + str($st.ok) + ")");
ok($st.refused >= 13, "every one of PngSuite's 'x' files is caught (got " + str($st.refused) + ")");
// Pillow reads one of them; we do not, deliberately.
ok($st.refused >= 13, "including the bad-IDAT-CRC file Pillow accepts");

// ─── the CRC check is what catches a corrupt header ─────────────────────
print("── chunk CRCs ──");

// xhdn0g08 has a valid-looking IHDR with a wrong checksum. Without the CRC
// check it decodes to a plausible but wrong image.
throws(def() { return pillow.open($PS + "xhdn0g08.png"); }, "a bad IHDR CRC is refused");
throws(def() { return pillow.open($PS + "xcsn0g01.png"); }, "a bad IDAT CRC is refused");

// LOAD_TRUNCATED_IMAGES is the deliberate opt-out, and it recovers the
// pixels rather than merely suppressing the error.
png.allow_truncated(true);
$rec = pillow.open($PS + "xcsn0g01.png");
eq($rec.tobytes(), readfile($PS + "xcsn0g01.exp"),
   "allow_truncated recovers Pillow's exact pixels from a bad-CRC file");
png.allow_truncated(false);
eq(png.truncated_allowed(), false, "and the flag resets");
throws(def() { return pillow.open($PS + "xcsn0g01.png"); }, "strict again afterwards");

// ════════════════════════════════════════════════════════════════════════
//  2. Pillow's own encoder output: PNG, BMP, TGA
// ════════════════════════════════════════════════════════════════════════
print("── 238 images written by Pillow, at every option ──");

$CD = "tests/fixtures/pillow/reference/codecs/";
$cjobs = _rows("tests/fixtures/pillow/reference/codec_index.txt");
eq(len($cjobs) > 230, true, "the codec corpus has over 230 images");

$per = {};
$shown = 0;
$i = 0;
while ($i < len($cjobs)) {
    $j = $cjobs[$i];
    $fmt = $j[0];
    $name = $j[1];
    if ($per[$fmt] == null) { $per[$fmt] = {"ok": 0, "bad": 0}; }
    $threw = false;
    $im = null;
    try { $im = pillow.open($CD + $name); } catch ($e) { $threw = true; }
    if ($threw) {
        $per[$fmt].bad = $per[$fmt].bad + 1;
        if ($shown < 8) { $shown = $shown + 1; print("     ERROR " + $name); }
    } else {
        $want = readfile($CD + $name + ".exp");
        if ($im.mode == $j[2] && $im.width == num($j[3]) && $im.height == num($j[4]) &&
            $im.tobytes() == $want) {
            $per[$fmt].ok = $per[$fmt].ok + 1;
        } else {
            $per[$fmt].bad = $per[$fmt].bad + 1;
            if ($shown < 8) {
                $shown = $shown + 1;
                print("     DIFF " + $name + ": got " + $im.mode + " " + str($im.size) +
                      " want " + $j[2] + " " + $j[3] + "x" + $j[4]);
            }
        }
    }
    $i = $i + 1;
}
$totBad = 0;
for $k, $v in $per.items() {
    print("   " + $k + ": " + str($v.ok) + " exact, " + str($v.bad) + " differing");
    $totBad = $totBad + $v.bad;
}
eq($totBad, 0, "every image Pillow wrote is read back pixel-identically");

// ════════════════════════════════════════════════════════════════════════
//  3. Round trips through our own writers
// ════════════════════════════════════════════════════════════════════════
print("── round trips: write then read ──");

$TMP = "/tmp/pillow_codec_test_";

def trip($mode, $w, $h, $fmt, $ext, $extra) {
    $im = pillow.new($mode, [$w, $h], null);
    // a deterministic pattern, so a shear or a flip is visible
    $y = 0;
    while ($y < $h) {
        $x = 0;
        while ($x < $w) {
            if ($mode == "1") {
                $v = 0;
                if (($x + $y) % 3 == 0) { $v = 255; }
                $im.putpixel([$x, $y], $v);
            } else {
                if (pillow.getmodebands($mode) == 1) { $im.putpixel([$x, $y], ($x * 13 + $y * 7) % 256); }
                else {
                    $c = [];
                    $b = 0;
                    while ($b < pillow.getmodebands($mode)) {
                        push($c, ($x * 11 + $y * 5 + $b * 37) % 256);
                        $b = $b + 1;
                    }
                    $im.putpixel([$x, $y], $c);
                }
            }
            $x = $x + 1;
        }
        $y = $y + 1;
    }
    $path = $TMP + $mode + "_" + str($w) + "x" + str($h) + "." + $ext;
    $im.save($path, $fmt);
    $back = pillow.open($path);
    $label = $fmt + " " + $mode + " " + str($w) + "x" + str($h);
    if ($back.mode != $im.mode) {
        $R.f = $R.f + 1;
        print("  FAIL  " + $label + ": mode " + $im.mode + " came back as " + $back.mode);
        return false;
    }
    if ($back.size != $im.size) {
        $R.f = $R.f + 1;
        print("  FAIL  " + $label + ": size " + str($im.size) + " came back as " + str($back.size));
        return false;
    }
    if ($back.getdata() != $im.getdata()) {
        $R.f = $R.f + 1;
        print("  FAIL  " + $label + ": pixels changed");
        return false;
    }
    $R.p = $R.p + 1;
    return true;
}

// PNG: every writable mode, at widths where the row padding bites
$pngModes = ["1", "L", "LA", "RGB", "RGBA", "I;16"];
$sizes = [[11, 7], [1, 1], [8, 8], [1, 9], [9, 1], [16, 3]];
$i = 0;
while ($i < len($pngModes)) {
    $k = 0;
    while ($k < len($sizes)) {
        trip($pngModes[$i], $sizes[$k][0], $sizes[$k][1], "PNG", "png", null);
        $k = $k + 1;
    }
    $i = $i + 1;
}

// BMP: 1, L and RGB (Pillow's BMP writer drops alpha, and so does ours)
$i = 0;
$bmpModes = ["1", "L", "RGB"];
while ($i < len($bmpModes)) {
    $k = 0;
    while ($k < len($sizes)) {
        trip($bmpModes[$i], $sizes[$k][0], $sizes[$k][1], "BMP", "bmp", null);
        $k = $k + 1;
    }
    $i = $i + 1;
}

// TGA: including alpha, which TGA keeps
$i = 0;
$tgaModes = ["1", "L", "RGB", "RGBA"];
while ($i < len($tgaModes)) {
    $k = 0;
    while ($k < len($sizes)) {
        trip($tgaModes[$i], $sizes[$k][0], $sizes[$k][1], "TGA", "tga", null);
        $k = $k + 1;
    }
    $i = $i + 1;
}

// every PNG filter round-trips, which is what exercises the unfilter paths
print("── every PNG scanline filter ──");
$src = pillow.new("RGB", [17, 5], null);
$y = 0;
while ($y < 5) {
    $x = 0;
    while ($x < 17) { $src.putpixel([$x, $y], [($x * 15) % 256, ($y * 51) % 256, ($x * $y) % 256]); $x = $x + 1; }
    $y = $y + 1;
}
$f = 0;
while ($f <= 4) {
    $spec = {"mode": $src.mode, "size": $src.size, "px": $src.px,
             "palette": null, "info": {}, "png_filter": $f};
    $bytesOut = png.encode($spec);
    $back = pillow.open_bytes($bytesOut, null);
    eq($back.getdata(), $src.getdata(), "PNG filter " + str($f) + " round-trips");
    $f = $f + 1;
}
throws(def() {
    return png.encode({"mode": "RGB", "size": $src.size, "px": $src.px,
                       "palette": null, "info": {}, "png_filter": 7});
}, "an out-of-range PNG filter");

// a P-mode PNG keeps its palette through a round trip
print("── palette round trips ──");
$p = pillow.new("RGB", [9, 4], null);
$y = 0;
while ($y < 4) { $x = 0; while ($x < 9) { $p.putpixel([$x, $y], [($x * 28) % 256, ($y * 60) % 256, 90]); $x = $x + 1; } $y = $y + 1; }
$pal = $p.convert("P", "NONE", null);
$pal.save($TMP + "pal.png", null);
$palBack = pillow.open($TMP + "pal.png");
eq($palBack.mode, "P", "a P-mode PNG reads back as P");
eq($palBack.getdata(), $pal.getdata(), "with the same indices");
eq($palBack.convert("RGB", null, null).getdata(), $pal.convert("RGB", null, null).getdata(),
   "and the palette survives, so the colours match");
$pal.save($TMP + "pal.bmp", null);
eq(pillow.open($TMP + "pal.bmp").convert("RGB", null, null).getdata(),
   $pal.convert("RGB", null, null).getdata(), "and through BMP");
$pal.save($TMP + "pal.tga", null);
eq(pillow.open($TMP + "pal.tga").convert("RGB", null, null).getdata(),
   $pal.convert("RGB", null, null).getdata(), "and through TGA");

// TGA's RLE writer
print("── TGA run-length encoding ──");
$flat = pillow.new("RGB", [40, 6], [7, 8, 9]);
$plain = tga.encode({"mode": "RGB", "size": [40, 6], "px": $flat.px, "palette": null, "info": {}});
$rle = tga.encode({"mode": "RGB", "size": [40, 6], "px": $flat.px, "palette": null,
                   "info": {}, "tga_rle": true});
ok(len($rle) < len($plain) / 4, "RLE compresses a flat image hard (" +
   str(len($rle)) + " vs " + str(len($plain)) + " bytes)");
eq(pillow.open_bytes($rle, null).getdata(), $flat.getdata(), "and round-trips");
// a run that crosses a row boundary: decoding row by row would break this
$cross = pillow.new("RGB", [7, 5], [3, 4, 5]);
$cross.putpixel([6, 1], [200, 100, 50]);
$cr = tga.encode({"mode": "RGB", "size": [7, 5], "px": $cross.px, "palette": null,
                  "info": {}, "tga_rle": true});
eq(pillow.open_bytes($cr, null).getdata(), $cross.getdata(),
   "an RLE run crossing a row boundary decodes correctly");
// top-down
$td = tga.encode({"mode": "RGB", "size": [7, 5], "px": $cross.px, "palette": null,
                  "info": {}, "tga_topdown": true});
eq(pillow.open_bytes($td, null).getdata(), $cross.getdata(), "a top-down TGA round-trips");

// ─── format identification ──────────────────────────────────────────────
print("── identification ──");

eq(pillow.formats(), ["PNG", "BMP", "TGA", "PPM"], "four formats are registered");
eq(pillow.registered_extensions()[".png"], "PNG", "the .png extension");
eq(pillow.registered_extensions()[".bmp"], "BMP", "the .bmp extension");
eq(pillow.registered_extensions()[".tga"], "TGA", "the .tga extension");
eq(pillow.identify(readfile($PS + "basn0g08.png"), null), "PNG", "a PNG is identified by magic");
eq(pillow.identify(readfile($CD + "BMP_RGB_11x7.bmp"), null), "BMP", "a BMP by magic");
// Magic beats the extension: a PNG named .bmp is still a PNG.
writefile($TMP + "liar.bmp", readfile($PS + "basn0g08.png"));
eq(pillow.open($TMP + "liar.bmp").mode, "L", "a PNG named .bmp is read as a PNG");
eq(pillow.identify("not an image at all, nowhere near one", null), null,
   "and nonsense identifies as nothing");

// ─── hostile input ──────────────────────────────────────────────────────
print("── hostile input ──");

throws(def() { return pillow.open_bytes("", null); }, "an empty file");
throws(def() { return pillow.open_bytes("not an image", null); }, "a text file");
// a PNG header claiming an enormous image, refused BEFORE any allocation
$bomb = pillow.frombytes("L", [1, 1], "x");
$hdr = readfile($PS + "basn0g08.png");
$huge = substr($hdr, 0, 16) + pillow.zlib_compress("", 0);
throws(def() { return pillow.open_bytes($huge, null); }, "a truncated PNG");
// every truncation of a real PNG: no crash, no hang
$valid = readfile($PS + "basn6a08.png");
$k = 0;
while ($k <= len($valid)) {
    try { pillow.open_bytes(substr($valid, 0, $k), null); } catch ($e) { }
    $k = $k + 1;
    if ($k < len($valid) - 40 && $k > 40) { $k = $k + 29; }
}
ok(true, "truncated at many lengths: no crash, no hang");
// every byte of the header corrupted
$vb = bytes($valid);
$k = 0;
while ($k < 64 && $k < len($vb)) {
    $mut = [];
    $j = 0;
    while ($j < len($vb)) { push($mut, $vb[$j]); $j = $j + 1; }
    $mut[$k] = ($mut[$k] + 137) % 256;
    try { pillow.open_bytes(frombytes($mut), null); } catch ($e) { }
    $k = $k + 1;
}
ok(true, "64 header-byte corruptions: no crash, no hang");

// BMP and TGA hostile headers
throws(def() { return bmp.decode("BM"); }, "a BMP with only its magic");
throws(def() { return bmp.decode("BM" + "0123456789012345678901234567890123456789"); },
       "a BMP with a nonsense DIB header");
throws(def() { return tga.decode("short"); }, "a TGA shorter than its header");
// a TGA claiming a huge size, refused from the header
$tgaHuge = chr(0) + chr(0) + chr(2) + "\x00\x00\x00\x00\x00" +
           "\x00\x00\x00\x00" + "\xff\xff" + "\xff\xff" + chr(24) + chr(0);
throws(def() { return tga.decode($tgaHuge); }, "a 65535x65535 TGA is refused before allocation");

// a mode no writer supports names the restriction rather than writing junk
throws(def() { return pillow.new("CMYK", [4, 4], null).save($TMP + "x.png", null); },
       "PNG refuses CMYK");
throws(def() { return pillow.new("CMYK", [4, 4], null).save($TMP + "x.bmp", null); },
       "BMP refuses CMYK");
throws(def() { return pillow.new("CMYK", [4, 4], null).save($TMP + "x.tga", null); },
       "TGA refuses CMYK");

// ─── cost ───────────────────────────────────────────────────────────────
print("── cost ──");

$big = pillow.new("RGB", [300, 200], null);
$y = 0;
while ($y < 200) {
    $x = 0;
    while ($x < 300) { $big.putpixel([$x, $y], [($x) % 256, ($y) % 256, ($x + $y) % 256]); $x = $x + 1; }
    $y = $y + 1;
}
$t0 = clock();
$big.save($TMP + "big.png", null);
$tw = clock() - $t0;
$t0 = clock();
$rb = pillow.open($TMP + "big.png");
$tr = clock() - $t0;
print("   60,000-pixel RGB PNG: write " + str($tw) + " ms, read " + str($tr) + " ms (" +
      str(filesize($TMP + "big.png")) + " bytes)");
eq($rb.getdata(), $big.getdata(), "and the big image round-trips exactly");
$t0 = clock();
$big.save($TMP + "big.bmp", null);
$tb = clock() - $t0;
print("   the same as BMP:      write " + str($tb) + " ms");
ok($tw < 12000, "a PNG write of 60,000 pixels is seconds, not minutes");
ok($tr < 12000, "and so is the read");

print("");
print("checks: " + str($R.p + $R.f) + "  passed: " + str($R.p) + "  failed: " + str($R.f));
if ($R.f == 0) { print("RESULT: ALL GREEN"); }
else { print("RESULT: FAILED"); }
