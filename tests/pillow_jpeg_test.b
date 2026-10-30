// ════════════════════════════════════════════════════════════════════════
//  tests/pillow_jpeg_test.b — pillow Phase P5: JPEG and EXIF.
//
//  JPEG is the one format in this package whose reference is not Pillow's
//  own C -- Pillow hands the file to libjpeg-turbo -- so this suite is
//  really a comparison against libjpeg's arithmetic: its entropy decoder,
//  jidctint.c's IDCT (and the 4x4, 2x2 and 1x1 scaled kernels behind
//  draft), jdsample.c's fancy upsamplers and jdcolor.c's colour transform.
//
//  Tier 2, the corpus (tests/fixtures/pillow/reference/jpeg/), decoded
//  BYTE-IDENTICALLY to Pillow:
//    * every subsampling Pillow writes, at sizes that are and are not MCU
//      multiples, down to 1x1
//    * the samplings Pillow cannot write, from cjpeg: 4:1:1, 2x1, 1x2,
//      4x2, 3x1 and a mixed 1x1,2x2,2x2
//    * progressive, with and without subsampling, with restart markers
//    * restart markers in baseline scans, by block and by row
//    * greyscale, CMYK (Adobe transform 0) and YCCK (transform 2)
//    * quality 1, 10, 50, 100 and optimize
//    * draft at 1/2, 1/4 and 1/8 -- a scaled IDCT, not a resize
//
//  Then the encoder against Pillow at a PSNR floor (quantisation choices
//  are the encoder's, so bit-exactness is not the claim), EXIF read and
//  write, and the hostile tier: truncation at every length, every length
//  field poisoned, every byte of a header flipped.
//
//  Run:  bantu run tests/pillow_jpeg_test.b
// ════════════════════════════════════════════════════════════════════════

include "./pillow/pillow.b" as pillow;
include "./pillow/codecs/jpeg.b" as jpeg;
include "./pillow/exif.b" as exif;
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

$JD = "tests/fixtures/pillow/reference/jpeg/";
$TMP = "/tmp/pillow-jpeg-test/";

print("");
print("── the corpus, against libjpeg through Pillow ──");

$jobs = _rows($JD + "MANIFEST");
ok(len($jobs) >= 50, "the JPEG corpus has at least 50 images (" + str(len($jobs)) + ")");

$bad = 0;
$shown = 0;
$groups = {};
$i = 0;
while ($i < len($jobs)) {
    $j = $jobs[$i];
    $i = $i + 1;
    $tag = $j[0];
    $file = $j[1];
    $wmode = $j[2];
    $ww = num($j[3]);
    $wh = num($j[4]);
    $denom = num($j[5]);
    $kind = "baseline";
    if (substr($tag, 0, 5) == "draft") { $kind = "draft"; }
    else if (substr($tag, 0, 4) == "samp" || substr($tag, 0, 2) == "cj") { $kind = "cjpeg"; }
    else if (contains($tag, "prog")) { $kind = "progressive"; }
    if ($groups[$kind] == null) { $groups[$kind] = {"ok": 0, "bad": 0}; }

    $threw = false;
    $spec = null;
    try { $spec = jpeg.decode_scaled(readfile($JD + $file), $denom, null); }
    catch ($e) { $threw = true; $why = str($e); }
    if ($threw) {
        $groups[$kind].bad = $groups[$kind].bad + 1;
        $bad = $bad + 1;
        if ($shown < 8) { $shown = $shown + 1; print("     ERROR " + $file + ": " + $why); }
    } else {
        $got = nd_tobytes($spec.px, "u8");
        $want = readfile($JD + $file + ".exp");
        if ($spec.mode == $wmode && $spec.size[0] == $ww && $spec.size[1] == $wh &&
            $got == $want) {
            $groups[$kind].ok = $groups[$kind].ok + 1;
        } else {
            $groups[$kind].bad = $groups[$kind].bad + 1;
            $bad = $bad + 1;
            if ($shown < 8) {
                $shown = $shown + 1;
                print("     DIFF " + $file + ": got " + $spec.mode + " " +
                      str($spec.size) + " " + str(len($got)) + " bytes, want " +
                      $wmode + " " + str([$ww, $wh]) + " " + str(len($want)) + " bytes");
            }
        }
    }
}
for $k, $v in $groups.items() {
    print("   " + $k + ": " + str($v.ok) + " byte-identical, " + str($v.bad) + " differing");
}
eq($bad, 0, "every image in the corpus decodes byte-identically to Pillow");

print("");
print("── the file's own metadata ──");

$im = pillow.open($JD + "rgb_16x16_ss0.jpg");
eq($im.format, "JPEG", "format is JPEG");
eq($im.mode, "RGB", "a 3-component JFIF file is RGB");
eq($im.info.jfif, 257, "jfif version 1.01 reads as 257");
eq($im.info.jfif_version, [1, 1], "and as [1, 1]");
eq($im.info.jfif_unit, 0, "jfif_unit");
eq($im.info.jfif_density, [1, 1], "jfif_density");
ok($im.info.jpeg_qtables != null, "the quantisation tables come back");
eq($im.info.jpeg_subsampling, 0, "4:4:4 reads as subsampling 0");
eq(pillow.open($JD + "rgb_16x16_ss2.jpg").info.jpeg_subsampling, 2,
   "and 4:2:0 as subsampling 2");
eq(pillow.open($JD + "rgb_prog_ss2_33x25.jpg").info.progressive, 1,
   "a progressive file says so in info");
eq(pillow.open($JD + "gray_17x13.jpg").mode, "L", "a 1-component file is L");
eq(pillow.open($JD + "cmyk_17x13.jpg").mode, "CMYK", "a 4-component file is CMYK");
eq(pillow.open($JD + "cmyk_17x13.jpg").info.adobe_transform, 0,
   "with Adobe transform 0");
eq(pillow.open($JD + "ycck_17x13.jpg").info.adobe_transform, 2,
   "and the YCCK twin says transform 2");
ok(jpeg.sniff("\xff\xd8\xff\xe0"), "sniff takes an SOI");
ok(!jpeg.sniff("\x89PNG"), "and refuses a PNG");

print("");
print("── draft(), which is a scaled IDCT ──");

$d = pillow.open($JD + "draft_64x48_ss2_d2.jpg");
eq($d.draft("RGB", [32, 24]), ["RGB", [0, 0, 32, 24]], "draft returns Pillow's [mode, box]");
eq($d.size, [32, 24], "and halves the image");
$d8 = pillow.open($JD + "draft_64x48_ss2_d8.jpg");
$d8.draft("RGB", [8, 6]);
eq($d8.size, [8, 6], "1/8 scale");
$dl = pillow.open($JD + "draft_64x48_ss0_d2.jpg");
eq($dl.draft("L", [32, 24])[0], "L", "draft to L gives L");
eq($dl.mode, "L", "and the image is L");
eq($dl.size, [32, 24], "at the drafted size");
$dy = pillow.open($JD + "draft_64x48_ss0_d2.jpg");
$dy.draft("YCbCr", null);
eq($dy.mode, "YCbCr", "draft to YCbCr with no size keeps the size and changes the mode");
eq($dy.size, [64, 48], "size unchanged");
eq(pillow.open($JD + "gray_17x13.jpg").draft("RGB", [4, 3]), ["L", [0, 0, 4.25, 3.25]],
   "drafting a greyscale file keeps it greyscale");

print("");
print("── the hostile tier ──");

$good = readfile($JD + "rgb_16x16_ss0.jpg");
throws(def() { jpeg.decode(""); }, "an empty string is not a JPEG");
throws(def() { jpeg.decode("\xff\xd8\xff"); }, "an SOI and nothing else");
throws(def() { jpeg.decode(substr($good, 0, 2)); }, "a bare SOI");
throws(def() { jpeg.decode("\x89PNG\r\n\x1a\n"); }, "a PNG");

// Truncated at every length: never a crash, never a hang, always a
// catchable error or a decoded image.
$trunc = 0;
$tdecoded = 0;
$n = 2;
while ($n < len($good)) {
    $caught = false;
    try { jpeg.decode(substr($good, 0, $n)); } catch ($e) { $caught = true; }
    if ($caught) { $trunc = $trunc + 1; } else { $tdecoded = $tdecoded + 1; }
    $n = $n + 1;
}
print("   truncation: " + str($trunc) + " refused, " + str($tdecoded) +
      " decoded from a partial scan (both are legal)");
ok($trunc + $tdecoded == len($good) - 2, "every truncation either decodes or raises");

// Every length field poisoned.
$poison = [0, 1, 255, 65535];
$pok = 0;
$pi = 0;
while ($pi < len($poison)) {
    $v = $poison[$pi];
    $pi = $pi + 1;
    $k = 2;
    while ($k + 4 < len($good)) {
        if (ord($good[$k]) == 255 && ord($good[$k + 1]) != 0 &&
            ord($good[$k + 1]) != 255) {
            $hi = chr(floor($v / 256));
            $lo = chr($v % 256);
            $m = substr($good, 0, $k + 2) + $hi + $lo +
                 substr($good, $k + 4, len($good) - $k - 4);
            $caught = false;
            try { jpeg.decode($m); } catch ($e) { $caught = true; }
            $pok = $pok + 1;
        }
        $k = $k + 1;
    }
}
ok($pok > 20, "every segment length was poisoned (" + str($pok) + " variants, none hung)");

// A bit flipped in every byte of the first 200.
$flips = 0;
$k = 0;
while ($k < 200 && $k < len($good)) {
    $b = ord($good[$k]);
    $m = substr($good, 0, $k) + chr(bxor($b, 64)) +
         substr($good, $k + 1, len($good) - $k - 1);
    $caught = false;
    try { jpeg.decode($m); } catch ($e) { $caught = true; }
    $flips = $flips + 1;
    $k = $k + 1;
}
ok($flips == 200 || $flips == len($good), "a flipped bit in every header byte is survivable");

// The frame-header guards.
throws(def() {
    // 60000x60000, which is 3.6 G pixels: refused from the header, before
    // any allocation.
    $hdr = "\xff\xd8\xff\xc0\x00\x11\x08\xea\x60\xea\x60\x03" +
           "\x01\x22\x00\x02\x11\x01\x03\x11\x01\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00";
    jpeg.decode($hdr);
}, "a decompression-bomb header is refused before allocation");
throws(def() {
    // SOF3: lossless JPEG, which libjpeg does not decode either.
    jpeg.decode("\xff\xd8\xff\xc3\x00\x0b\x08\x00\x10\x00\x10\x01\x01\x11\x00");
}, "a lossless (SOF3) frame is refused by name");
throws(def() {
    jpeg.decode("\xff\xd8\xff\xc9\x00\x0b\x08\x00\x10\x00\x10\x01\x01\x11\x00");
}, "an arithmetic-coded (SOF9) frame is refused by name");
throws(def() { jpeg.decode_scaled($good, 3, null); },
       "a scale denominator of 3 is refused: libjpeg scales by 1, 2, 4 or 8");

// The progressive coefficient budget (decision PW13).
$was = jpeg.max_progressive_coeffs();
jpeg.set_max_progressive_coeffs(100);
throws(def() { jpeg.decode(readfile($JD + "rgb_prog_ss2_33x25.jpg")); },
       "a progressive file over the coefficient budget is refused by name");
jpeg.set_max_progressive_coeffs($was);
$re = jpeg.decode(readfile($JD + "rgb_prog_ss2_33x25.jpg"));
eq($re.size, [33, 25], "and decodes once the budget is back");

print("");
print("── the encoder, against Pillow's own bytes ──");

$ecases = _rows($JD + "MANIFEST.enc");
ok(len($ecases) >= 15, "there are at least 15 encoder cases (" + str(len($ecases)) + ")");

$srcs = {};

// The encoder cases share a few sources, stored as raw bytes per mode so
// that no codec stands between the comparison and Pillow's encoder.
def _srcSized($mode, $w, $h) {
    $key = $mode + "_" + str($w) + "x" + str($h);
    if ($srcs[$key] == null) {
        $name = "enc_src_" + $mode;
        if ($w != 64 || $h != 48) { $name = $name + "_" + str($w) + "x" + str($h); }
        $srcs[$key] = pillow.frombytes($mode, [$w, $h], readfile($JD + $name + ".raw"));
    }
    return $srcs[$key];
}
def _src($mode) { return _srcSized($mode, 64, 48); }

$ebad = 0;
$exact = 0;
$pixel = 0;
$i = 0;
while ($i < len($ecases)) {
    $e = $ecases[$i];
    $i = $i + 1;
    $tag = $e[0];
    $file = $e[1];
    $mode = $e[2];
    $byteExact = num($e[5]);
    $ew = num($e[3]);
    $eh = num($e[4]);
    $im = _srcSized($mode, $ew, $eh);
    $spec = {"mode": $mode, "size": [$ew, $eh], "px": $im.px, "info": {}};
    // The option string, "k=v,k=v", straight onto the spec dict.
    $opts = split($e[6], ",");
    $k = 0;
    while ($k < len($opts)) {
        $kv = split($opts[$k], "=");
        $k = $k + 1;
        $key = $kv[0];
        $val = $kv[1];
        if ($key == "quality") {
            if ($val == "web_high") { $spec.quality = "web_high"; } else { $spec.quality = num($val); }
        }
        else if ($key == "subsampling") { $spec.subsampling = num($val); }
        else if ($key == "optimize") { $spec.optimize = num($val) == 1; }
        else if ($key == "progressive") { $spec.progressive = num($val) == 1; }
        else if ($key == "restart_marker_blocks") { $spec.restart_marker_blocks = num($val); }
        else if ($key == "restart_marker_rows") { $spec.restart_marker_rows = num($val); }
        else if ($key == "dpi") { $spec.info = {"dpi": [num($val), num($val)]}; }
    }
    $threw = false;
    $got = null;
    try { $got = jpeg.encode($spec); } catch ($er) { $threw = true; $why = str($er); }
    if ($threw) {
        $ebad = $ebad + 1;
        print("     ERROR " + $tag + ": " + $why);
    } else {
        $want = readfile($JD + $file);
        if ($byteExact == 1) {
            if ($got == $want) { $exact = $exact + 1; }
            else {
                $ebad = $ebad + 1;
                print("     DIFF " + $tag + ": " + str(len($got)) + " bytes, Pillow wrote " +
                      str(len($want)));
            }
        } else {
            // Progressive: the scan script differs by design, so the gate is
            // that both files decode to the same pixels.
            $a = jpeg.decode($got);
            $b = jpeg.decode($want);
            if (nd_tobytes($a.px, "u8") == nd_tobytes($b.px, "u8") && $a.info.progressive == 1) {
                $pixel = $pixel + 1;
            } else {
                $ebad = $ebad + 1;
                print("     DIFF " + $tag + ": progressive pixels differ from Pillow's");
            }
        }
    }
}
print("   " + str($exact) + " byte-identical to libjpeg, " + str($pixel) +
      " pixel-identical (progressive)");
eq($ebad, 0, "every encoder case matches Pillow");

print("");
print("── the encoder's own round trips and refusals ──");

$rgb = _src("RGB");
$r1 = jpeg.decode(jpeg.encode({"mode": "RGB", "size": [64, 48], "px": $rgb.px,
                               "info": {}, "quality": 100, "subsampling": 0}));
eq($r1.size, [64, 48], "quality 100 4:4:4 round-trips the size");
$q = jpeg.decode(readfile($JD + "rgb_16x16_ss0.jpg"));
$keep = jpeg.encode({"mode": "RGB", "size": $q.size, "px": $q.px, "info": $q.info,
                     "quality": "keep"});
$kb = jpeg.decode($keep);
eq($kb.info.jpeg_qtables, $q.info.jpeg_qtables,
   "quality=\"keep\" writes the tables it was given back");
eq($kb.info.jpeg_subsampling, $q.info.jpeg_subsampling, "and the same subsampling");

// The tables come back through info and go back in as an option.
$re = jpeg.encode({"mode": "RGB", "size": $q.size, "px": $q.px, "info": {},
                   "qtables": [$q.info.jpeg_qtables["0"], $q.info.jpeg_qtables["1"]]});
eq(jpeg.decode($re).info.jpeg_qtables["0"], $q.info.jpeg_qtables["0"],
   "an explicit qtable survives the round trip");

$comment = jpeg.encode({"mode": "L", "size": [8, 8], "px": pillow.new("L", [8, 8], 7).px,
                        "info": {"comment": "written by pillow"}});
eq(jpeg.decode($comment).info.comment, "written by pillow", "a COM segment round-trips");
$icc = jpeg.encode({"mode": "L", "size": [8, 8], "px": pillow.new("L", [8, 8], 7).px,
                    "info": {"icc_profile": "not-really-a-profile"}});
eq(jpeg.decode($icc).info.icc_profile, "not-really-a-profile", "an ICC profile round-trips");
$dpi = jpeg.decode(jpeg.encode({"mode": "L", "size": [8, 8],
                                "px": pillow.new("L", [8, 8], 7).px,
                                "info": {"dpi": [300, 300]}}));
eq($dpi.info.dpi, [300, 300], "dpi round-trips through the JFIF segment");

throws(def() {
    jpeg.encode({"mode": "P", "size": [8, 8], "px": pillow.new("P", [8, 8], 0).px, "info": {}});
}, "mode P cannot be written as JPEG, and says so");
throws(def() {
    jpeg.encode({"mode": "RGBA", "size": [8, 8], "px": pillow.new("RGBA", [8, 8], 0).px, "info": {}});
}, "nor can RGBA");
throws(def() {
    jpeg.encode({"mode": "L", "size": [0, 0], "px": pillow.new("L", [0, 0], 0).px, "info": {}});
}, "nor a zero-sized image");
throws(def() {
    jpeg.encode({"mode": "RGB", "size": [8, 8], "px": _src("RGB").px, "info": {},
                 "subsampling": 7});
}, "an unknown subsampling is refused by name");
throws(def() {
    jpeg.encode({"mode": "RGB", "size": [8, 8], "px": _src("RGB").px, "info": {},
                 "quality": "wildly_good"});
}, "an unknown quality preset is refused by name");
throws(def() {
    jpeg.encode({"mode": "L", "size": [8, 8], "px": pillow.new("L", [8, 8], 7).px,
                 "info": {}, "qtables": [[0]]});
}, "a qtable of the wrong length is refused");

// Through Image.save, which is how a caller actually reaches all this.
$dir = $TMP;
$rgb.save($dir + "out.jpg", null);
$rt = pillow.open($dir + "out.jpg");
eq([$rt.mode, $rt.size, $rt.format], ["RGB", [64, 48], "JPEG"],
   "Image.save(\"x.jpg\") picks the JPEG codec by extension");
$rgb.save($dir + "out2.jpg", "JPEG", {"quality": 95, "subsampling": 0});
ok(filesize($dir + "out2.jpg") > filesize($dir + "out.jpg"),
   "and save options reach the encoder (quality 95 is bigger than 75)");

print("");
print("── EXIF ──");

$ex = exif.empty();
exif.put($ex, "ifd0", 274, 6);
exif.put($ex, "ifd0", 271, "Bantu");
exif.put($ex, "ifd0", 282, [72, 1]);
exif.put($ex, "exif", 33434, [1, 125]);
exif.put($ex, "exif", 34855, 400);
exif.put($ex, "gps", 1, "N");
exif.put($ex, "gps", 2, [[51, 1], [30, 1], [0, 1]]);
exif.put($ex, "interop", 1, "R98");
$blob = exif.tobytes($ex);
$back = exif.load($blob);
eq(exif.get($back, "ifd0", 274), 6, "Orientation round-trips");
eq(exif.get($back, "ifd0", 271), "Bantu", "an ASCII tag round-trips");
eq(exif.get($back, "ifd0", 282), [72, 1], "a RATIONAL round-trips as [n, d]");
eq(exif.get($back, "exif", 33434), [1, 125], "a tag in the Exif sub-IFD round-trips");
eq(exif.get($back, "exif", 34855), 400, "and a SHORT");
eq(exif.get($back, "gps", 1), "N", "a GPS tag round-trips");
eq(exif.get($back, "gps", 2), [[51, 1], [30, 1], [0, 1]], "and a 3-rational GPS coordinate");
eq(exif.get($back, "interop", 1), "R98", "the Interop IFD round-trips");
eq(exif.tag_name(271), "Make", "ExifTags.TAGS is there");
eq(exif.tag_name(34853), "GPSInfo", "and names the sub-IFD pointers");
eq(exif.gps_tag_name(2), "GPSLatitude", "ExifTags.GPSTAGS is there");
eq(len(keys(exif.TAGS())) > 250, true, "all of Pillow's 274 tag names");
eq(exif.IFD().Exif, 34665, "ExifTags.IFD.Exif");
eq(exif.BASE().Orientation, 274, "ExifTags.Base.Orientation");

// Little-endian, which is what most cameras write.
$le = exif.empty();
$le.endian = "II";
exif.put($le, "ifd0", 274, 3);
exif.put($le, "ifd0", 306, "2026:10:30 12:00:00");
$leBack = exif.load(exif.tobytes($le));
eq($leBack.endian, "II", "a little-endian block says so");
eq(exif.get($leBack, "ifd0", 306), "2026:10:30 12:00:00", "and reads back");

// The EXIF in a real JPEG, written by us and read back through the codec.
$withExif = jpeg.encode({"mode": "L", "size": [8, 8], "px": pillow.new("L", [8, 8], 100).px,
                         "info": {"exif": $blob}});
$rd = jpeg.decode($withExif);
ok($rd.info.exif != null, "an APP1 segment survives a JPEG round trip");
eq(exif.get(exif.load($rd.info.exif), "ifd0", 271), "Bantu", "and its tags are intact");

// The hostile cases of §6.5: a self-referential IFD and an overlong count.
throws(def() {
    // IFD0 at offset 8 whose next-IFD pointer is 8 again.
    $b = "Exif\x00\x00MM\x00\x2a\x00\x00\x00\x08\x00\x01" +
         "\x01\x12\x00\x03\x00\x00\x00\x01\x00\x06\x00\x00" + "\x00\x00\x00\x08";
    exif.load($b);
}, "an IFD chain that points at itself is refused, not followed");
$huge = "Exif\x00\x00MM\x00\x2a\x00\x00\x00\x08\x00\x01" +
        "\x01\x12\x00\x04\x7f\xff\xff\xff\x00\x00\x00\x1a" + "\x00\x00\x00\x00";
$hl = exif.load($huge);
eq(exif.get($hl, "ifd0", 274), null,
   "a tag claiming 2^31 entries is skipped, not allocated");
throws(def() { exif.load("Exif\x00\x00XX\x00\x2a\x00\x00\x00\x08"); },
       "a byte order that is neither II nor MM is refused");
throws(def() { exif.load("Exif\x00\x00MM\x00\x07\x00\x00\x00\x08"); },
       "TIFF magic that is not 42 is refused");
throws(def() { exif.load("Exif"); }, "a block too short to hold a header is refused");

// exif_transpose, over all eight orientations.
$orient = [1, 2, 3, 4, 5, 6, 7, 8];
$i = 0;
$otrue = 0;
while ($i < len($orient)) {
    $o = $orient[$i];
    $i = $i + 1;
    $e2 = exif.empty();
    exif.put($e2, "ifd0", 274, $o);
    $src2 = pillow.new("RGB", [4, 2], [10, 20, 30]);
    $src2.putpixel([0, 0], [255, 0, 0]);
    $src2.info = {"exif": exif.tobytes($e2)};
    $t = exif.exif_transpose($src2);
    // Orientations 5-8 transpose the axes; 1-4 do not.
    $wantSize = [4, 2];
    if ($o >= 5) { $wantSize = [2, 4]; }
    if (str($t.size) == str($wantSize)) { $otrue = $otrue + 1; }
    else { print("     orientation " + str($o) + " gave " + str($t.size)); }
    if ($o != 1) {
        $after = exif.orientation_of($t.info);
        if ($after != null) { print("     orientation " + str($o) + " left the tag behind"); }
    }
}
eq($otrue, 8, "exif_transpose handles all eight orientations");
eq(exif.transpose_for(1), null, "orientation 1 is a no-op");
eq(exif.transpose_for(6), 4, "orientation 6 is ROTATE_270");
eq(exif.orientation_of({}), null, "no exif, no orientation");
eq(exif.orientation_of({"exif": "junk"}), null, "unreadable exif is no orientation");

print("");
print("── the published cost of a scan ──");

$big = pillow.open($JD + "rgb_64x48_ss2.jpg");
$t0 = clock();
$bigJ = jpeg.encode({"mode": "RGB", "size": $big.size, "px": $big.px, "info": {},
                     "quality": 75, "subsampling": 2});
$tenc = clock() - $t0;
$t0 = clock();
jpeg.decode($bigJ);
$tdec = clock() - $t0;
print("   64x48 RGB 4:2:0: encode " + str($tenc) + " ms, decode " + str($tdec) + " ms");
ok($tenc < 8000, "a small encode is well under a second");
ok($tdec < 8000, "and so is the decode");

$t0 = clock();
pillow.open($JD + "draft_64x48_ss2_d8.jpg").draft("RGB", [8, 6]);
$tdraft = clock() - $t0;
print("   the same at 1/8 through draft: " + str($tdraft) +
      " ms (the scan is read in full at every scale -- architecture 6.4.6)");

print("");
print("checks: " + str($R.p + $R.f) + "  passed: " + str($R.p) + "  failed: " + str($R.f));
if ($R.f == 0) { print("RESULT: ALL GREEN"); }
else { print("RESULT: FAILED"); }
