// ════════════════════════════════════════════════════════════════════════
//  tests/pillow_fuzz_test.b — the hostile-input tier, phase P1.
//
//  Decompression is where an imaging library gets attacked, so the gate is
//  stated as a property rather than a list of cases:
//
//    * no input crashes the interpreter
//    * no input hangs it
//    * every rejection is a CATCHABLE Bantu error
//    * the native decoder and the pure-Bantu decoder AGREE on every input --
//      both reject, or both return the same bytes
//
//  That last one is the useful one. A single decoder can only be checked
//  against cases someone thought of; two independent decoders check each
//  other against cases nobody thought of.
//
//  Deterministic: the generator is seeded, so a failure here is reproducible.
//
//  Run:  bantu run tests/pillow_fuzz_test.b
// ════════════════════════════════════════════════════════════════════════

include "./pillow/inflate.b" as binflate;

$R = {"p": 0, "f": 0, "rejected": 0, "decoded": 0, "disagree": 0};
def ok($cond, $name) {
    if ($cond) { $R.p = $R.p + 1; }
    else { $R.f = $R.f + 1; print("  FAIL  " + $name); }
}

$NATIVE = false;
try { $NATIVE = has_native("inflate"); } catch ($e) { $NATIVE = false; }

// A seeded 32-bit LCG, so the corpus is the same on every run and on every
// machine. random() is not used: this has to be reproducible.
$RNG = {"s": 20261027};
def rnd($n) {
    $RNG.s = (1103515245 * $RNG.s + 12345) % 2147483648;
    return floor($RNG.s / 65536) % $n;
}

// Run one candidate through both decoders and require them to agree.
def both($bytesIn, $label) {
    $bOut = null; $bFail = false;
    try { $bOut = binflate.inflate($bytesIn, 4000000); } catch ($e) { $bFail = true; }
    if ($bFail) { $R.rejected = $R.rejected + 1; } else { $R.decoded = $R.decoded + 1; }
    if ($NATIVE) {
        $nOut = null; $nFail = false;
        try { $nOut = bytes(inflate(frombytes($bytesIn), 4000000)); } catch ($e) { $nFail = true; }
        if ($nFail != $bFail) {
            $R.disagree = $R.disagree + 1;
            if ($R.disagree < 6) {
                print("  DISAGREE " + $label + ": native rejected=" + str($nFail) +
                      " bantu rejected=" + str($bFail));
            }
        } else {
            if (!$bFail && str($nOut) != str($bOut)) {
                $R.disagree = $R.disagree + 1;
                if ($R.disagree < 6) { print("  DISAGREE " + $label + ": different output"); }
            }
        }
    }
    return true;
}

// ─── 1. pure random bytes ───────────────────────────────────────────────
print("── random bytes ──");
$i = 0;
while ($i < 600) {
    $n = 1 + rnd(60);
    $b = []; $k = 0;
    while ($k < $n) { push($b, rnd(256)); $k = $k + 1; }
    both($b, "random " + str($i));
    $i = $i + 1;
}
ok(true, "600 random byte strings");

// ─── 2. plausible-looking headers with random bodies ────────────────────
// A valid zlib header followed by noise is the shape that gets furthest into
// the decoder before failing, which is where the interesting bugs live.
print("── valid headers, hostile bodies ──");
$i = 0;
while ($i < 600) {
    $b = [120, 156];                                   // 0x78 0x9C: zlib, default
    $n = 2 + rnd(80);
    $k = 0;
    while ($k < $n) { push($b, rnd(256)); $k = $k + 1; }
    both($b, "zlibhdr " + str($i));
    $i = $i + 1;
}
ok(true, "600 valid zlib headers over random bodies");

// ─── 3. every block-header bit pattern, then noise ──────────────────────
print("── every block header ──");
$h = 0;
while ($h < 256) {
    $b = [$h];
    $k = 0;
    while ($k < 30) { push($b, rnd(256)); $k = $k + 1; }
    both($b, "hdr " + str($h));
    $h = $h + 1;
}
ok(true, "all 256 first-byte values");

// ─── 4. mutations of a valid stream ─────────────────────────────────────
// Single-byte, two-byte and truncating mutations of a real stream. These
// reach deep into the Huffman tables, which random noise rarely does.
print("── mutations of a valid stream ──");
$src = "";
$parts = []; $i = 0;
while ($i < 400) { push($parts, "the quick brown fox jumps over the lazy dog "); $i = $i + 1; }
$src = join($parts, "");
$valid = bytes(bp_zlib($src));
$vn = len($valid);
print("   base stream: " + str($vn) + " bytes -> " + str(len($src)) + " bytes");

// single-byte substitution at every position, with several replacements
$i = 0;
while ($i < 1200) {
    $mut = []; $k = 0;
    while ($k < $vn) { push($mut, $valid[$k]); $k = $k + 1; }
    $mut[rnd($vn)] = rnd(256);
    both($mut, "mut1 " + str($i));
    $i = $i + 1;
}
ok(true, "1200 single-byte substitutions");

// multi-byte corruption
$i = 0;
while ($i < 400) {
    $mut = []; $k = 0;
    while ($k < $vn) { push($mut, $valid[$k]); $k = $k + 1; }
    $howMany = 2 + rnd(6);
    $k = 0;
    while ($k < $howMany) { $mut[rnd($vn)] = rnd(256); $k = $k + 1; }
    both($mut, "mutN " + str($i));
    $i = $i + 1;
}
ok(true, "400 multi-byte corruptions");

// truncation at every length
$k = 0;
while ($k <= $vn) {
    $pre = []; $j = 0;
    while ($j < $k) { push($pre, $valid[$j]); $j = $j + 1; }
    both($pre, "trunc " + str($k));
    $k = $k + 1;
}
ok(true, "truncation at all " + str($vn + 1) + " lengths");

// a valid stream with trailing garbage
$i = 0;
while ($i < 100) {
    $ext = []; $j = 0;
    while ($j < $vn) { push($ext, $valid[$j]); $j = $j + 1; }
    $extra = 1 + rnd(20); $j = 0;
    while ($j < $extra) { push($ext, rnd(256)); $j = $j + 1; }
    both($ext, "tail " + str($i));
    $i = $i + 1;
}
ok(true, "100 streams with trailing garbage");

// ─── 5. gzip headers with hostile extra fields ──────────────────────────
print("── gzip header fields ──");
$i = 0;
while ($i < 300) {
    // magic, DEFLATE, a random FLG so FEXTRA/FNAME/FCOMMENT/FHCRC all fire
    $b = [31, 139, 8, rnd(32), 0, 0, 0, 0, 0, 255];
    $n = rnd(60);
    $k = 0;
    while ($k < $n) { push($b, rnd(256)); $k = $k + 1; }
    // a plausible trailer
    $k = 0;
    while ($k < 8) { push($b, rnd(256)); $k = $k + 1; }
    both($b, "gzhdr " + str($i));
    $i = $i + 1;
}
ok(true, "300 gzip headers with hostile extras");

// ─── 6. declared-length abuse ───────────────────────────────────────────
// A stored block whose LEN claims far more than the stream holds, at every
// interesting magnitude.
print("── declared lengths ──");
$lens = [0, 1, 2, 255, 256, 32767, 32768, 65534, 65535];
$li = 0;
while ($li < len($lens)) {
    $L = $lens[$li];
    $lo = $L % 256;
    $hi = floor($L / 256) % 256;
    $nlo = (65535 - $L) % 256;
    $nhi = floor((65535 - $L) / 256) % 256;
    both([1, $lo, $hi, $nlo, $nhi], "storedlen " + str($L));
    both([1, $lo, $hi, $nlo, $nhi, 65, 66, 67], "storedlen+3 " + str($L));
    // and with a deliberately wrong NLEN
    both([1, $lo, $hi, 0, 0, 65], "badnlen " + str($L));
    $li = $li + 1;
}
ok(true, "declared stored-block lengths at every magnitude");

// ─── 7. the bomb, at every maxOut boundary ──────────────────────────────
print("── maxOut boundaries ──");
$zeros = []; $i = 0;
while ($i < 200000) { push($zeros, 0); $i = $i + 1; }
$bomb = bp_zlib(frombytes($zeros));
$caps = [0, 1, 2, 100, 199999, 200000, 200001];
$ci = 0; $wrong = 0;
while ($ci < len($caps)) {
    $cap = $caps[$ci];
    $shouldPass = ($cap >= 200000);
    $passed = false;
    try { $o = binflate.inflate($bomb, $cap); $passed = (len($o) == 200000); } catch ($e) { $passed = false; }
    if ($passed != $shouldPass) { $wrong = $wrong + 1; print("  FAIL bantu maxOut " + str($cap)); }
    if ($NATIVE) {
        $npassed = false;
        try { $o = inflate($bomb, $cap); $npassed = (len($o) == 200000); } catch ($e) { $npassed = false; }
        if ($npassed != $shouldPass) { $wrong = $wrong + 1; print("  FAIL native maxOut " + str($cap)); }
    }
    $ci = $ci + 1;
}
ok($wrong == 0, "maxOut is exact at every boundary");

// ─── the gate ───────────────────────────────────────────────────────────
print("");
print("candidates: " + str($R.rejected + $R.decoded) +
      "   rejected: " + str($R.rejected) + "   decoded: " + str($R.decoded));
if ($NATIVE) {
    ok($R.disagree == 0, "native and pure-Bantu agree on every candidate (" +
       str($R.disagree) + " disagreements)");
} else {
    print("(native inflate absent: the agreement check did not run)");
}
ok(true, "no input crashed the interpreter");
ok(true, "no input hung the interpreter");

print("");
print("checks: " + str($R.p + $R.f) + "  passed: " + str($R.p) + "  failed: " + str($R.f));
if ($R.f == 0) { print("RESULT: ALL GREEN"); }
else { print("RESULT: FAILED"); }
