// ════════════════════════════════════════════════════════════════════════
//  tests/pillow_inflate_test.b — pillow Phase P1.
//
//  Three independent implementations of DEFLATE have to agree:
//
//    1. zlib / gzip in Python, which produced tests/fixtures/pillow/deflate
//       (levels 0, 1, 6 and 9, so stored, fixed-Huffman and dynamic-Huffman
//       blocks are all represented)
//    2. the interpreter's native inflate() / gunzip()   (src/inflate.hpp)
//    3. the pure-Bantu decoder                          (pillow/inflate.b)
//
//  A decompressor is a parser of hostile input. Two implementations agreeing
//  on a corpus neither one produced is the strongest cheap evidence either is
//  right, and it is why the Bantu twin ships rather than being deleted once
//  the atom landed.
//
//  Run:  bantu run tests/pillow_inflate_test.b
// ════════════════════════════════════════════════════════════════════════

include "./pillow/inflate.b" as binflate;

$R = {"p": 0, "f": 0};
def eq($got, $want, $name) {
    if ($got == $want) { $R.p = $R.p + 1; }
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

$NATIVE = false;
try { $NATIVE = has_native("inflate"); } catch ($e) { $NATIVE = false; }
print("native inflate: " + str($NATIVE));
print("");

// ─── Tier 2: differential against Python's zlib and gzip ────────────────
print("── differential: Python zlib/gzip vs native vs pure Bantu ──");

$DIR = "tests/fixtures/pillow/deflate/";
$NAMES = ["english", "runs", "random", "allbytes", "tiny", "empty", "struct", "binary"];
$LEVELS = ["0", "1", "6", "9"];

$ni = 0;
while ($ni < len($NAMES)) {
    $nm = $NAMES[$ni];
    $want = readfile($DIR + $nm + ".raw");

    $li = 0;
    while ($li < len($LEVELS)) {
        $z = readfile($DIR + $nm + "_" + $LEVELS[$li] + ".z");
        // the pure-Bantu decoder
        eq(binflate.inflate_str($z, null), $want, "bantu  zlib " + $nm + " L" + $LEVELS[$li]);
        // the native atom, and that the two agree
        if ($NATIVE) {
            eq(inflate($z, null), $want, "native zlib " + $nm + " L" + $LEVELS[$li]);
        }
        $li = $li + 1;
    }

    $g = readfile($DIR + $nm + ".gz");
    eq(binflate.inflate_str($g, null), $want, "bantu  gzip " + $nm);
    if ($NATIVE) {
        eq(inflate($g, null), $want, "native gzip " + $nm);
        eq(gunzip($g, null), $want, "native gunzip " + $nm);
    }
    $ni = $ni + 1;
}

// ─── Tier 2: round-trip against the interpreter's own compressor ─────────
print("── round trip: bp_zlib -> both decoders ──");

def trip($s, $name) {
    $z = bp_zlib($s);
    eq(binflate.inflate_str($z, null), $s, "bantu  trip " + $name);
    if ($NATIVE) { eq(inflate($z, null), $s, "native trip " + $name); }
}
trip("", "empty");
trip("a", "one byte");
trip("ab", "two bytes");
trip("hello, hello, hello, hello, hello!", "repetitive");
$all = []; $i = 0; while ($i < 256) { push($all, $i); $i = $i + 1; }
trip(frombytes($all), "all 256 byte values");
$runl = []; $i = 0; while ($i < 70000) { push($runl, 65); $i = $i + 1; }
trip(frombytes($runl), "70k identical bytes");
trip(frombytes(randbytes(100000)), "100k CSPRNG bytes");

// Every length from 0 to 300, so no boundary in the block machinery hides.
$bad = 0; $L = 0;
while ($L <= 300) {
    $s = ""; $parts = []; $i = 0;
    while ($i < $L) { push($parts, chr(65 + ($i % 26))); $i = $i + 1; }
    $s = join($parts, "");
    if (binflate.inflate_str(bp_zlib($s), null) != $s) { $bad = $bad + 1; }
    $L = $L + 1;
}
eq($bad, 0, "every length 0..300 round-trips (pure Bantu)");

if ($NATIVE) {
    $bad = 0; $L = 0;
    while ($L <= 300) {
        $parts = []; $i = 0;
        while ($i < $L) { push($parts, chr(65 + ($i % 26))); $i = $i + 1; }
        $s = join($parts, "");
        if (inflate(bp_zlib($s), null) != $s) { $bad = $bad + 1; }
        $L = $L + 1;
    }
    eq($bad, 0, "every length 0..300 round-trips (native)");
}

// ─── Tier 1: the hand-built RFC 1951 cases ──────────────────────────────
print("── RFC 1951 structure ──");

// A stored block: BFINAL=1 BTYPE=00 -> 0x01, LEN=3, NLEN=~3, "XYZ"
eq(frombytes(binflate.raw([1, 3, 0, 252, 255, 88, 89, 90], null)), "XYZ", "stored block");
// Two stored blocks, the first not final
eq(frombytes(binflate.raw([0, 1, 0, 254, 255, 65,
                           1, 1, 0, 254, 255, 66], null)), "AB", "two stored blocks");

// ─── Tier 1 + 3: failure modes. Every one must be a catchable error ─────
print("── failure modes ──");

throws(def() { return binflate.raw([7], null); }, "bantu  reserved block type 3");
throws(def() { return binflate.raw([1, 3, 0, 0, 0, 88], null); }, "bantu  stored LEN/NLEN mismatch");
throws(def() { return binflate.raw([1, 9, 0, 246, 255, 88], null); }, "bantu  stored runs past end");
throws(def() { return binflate.raw([], null); }, "bantu  empty input");
throws(def() { return binflate.zlib([0, 0], null); }, "bantu  zlib bad method");
throws(def() { return binflate.zlib([8, 0], null); }, "bantu  zlib bad header check");
throws(def() { return binflate.gunzip([31, 139, 8], null); }, "bantu  gzip too short");
throws(def() { return binflate.gunzip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                       11, 12, 13, 14, 15, 16, 17, 18], null); }, "bantu  gzip bad magic");

if ($NATIVE) {
    throws(def() { return inflate([7], null); }, "native reserved block type 3");
    throws(def() { return inflate([1, 3, 0, 0, 0, 88], null); }, "native stored LEN/NLEN mismatch");
    throws(def() { return inflate("", null); }, "native empty input");
    throws(def() { return gunzip([31, 139, 8], null); }, "native gzip too short");
    throws(def() { return inflate("this is not compressed at all", null); }, "native garbage");
}

// Adler-32 is checked: corrupt the trailer and the stream must be refused.
$z = bytes(bp_zlib("the quick brown fox"));
$z[len($z) - 1] = ($z[len($z) - 1] + 1) % 256;
throws(def() { return binflate.zlib($z, null); }, "bantu  Adler-32 mismatch refused");
if ($NATIVE) {
    throws(def() { return inflate(frombytes($z), null); }, "native Adler-32 mismatch refused");
}

// ─── Tier 3: the inflate bomb ───────────────────────────────────────────
print("── inflate bomb: maxOut is checked before the memory is committed ──");

$zeros = []; $i = 0; while ($i < 400000) { push($zeros, 0); $i = $i + 1; }
$bomb = bp_zlib(frombytes($zeros));
print("   400,000 zero bytes compress to " + str(len($bomb)) + " bytes (ratio " +
      str(floor(400000 / len($bomb))) + ":1)");

throws(def() { return binflate.inflate($bomb, 1000); }, "bantu  bomb refused at maxOut 1000");
eq(len(binflate.inflate($bomb, 400000)), 400000, "bantu  exact maxOut accepted");
throws(def() { return binflate.inflate($bomb, 399999); }, "bantu  one byte under maxOut refused");
if ($NATIVE) {
    throws(def() { return inflate($bomb, 1000); }, "native bomb refused at maxOut 1000");
    eq(len(inflate($bomb, 400000)), 400000, "native exact maxOut accepted");
    throws(def() { return inflate($bomb, 399999); }, "native one byte under maxOut refused");
    throws(def() { return inflate($bomb, 0); }, "native maxOut 0 refuses any output");
}

// ─── Tier 3: hostile input. No crash, no hang, every failure catchable ──
print("── hostile input: truncation and bit flips ──");

$zb = bytes(readfile($DIR + "english_6.z"));
$n = len($zb);

// Truncated at every single length.
$k = 0;
while ($k <= $n) {
    $pre = []; $j = 0; while ($j < $k) { push($pre, $zb[$j]); $j = $j + 1; }
    try { binflate.inflate($pre, 200000); } catch ($e) { }
    if ($NATIVE) { try { inflate(frombytes($pre), 200000); } catch ($e) { } }
    $k = $k + 1;
}
ok(true, "truncated at all " + str($n + 1) + " lengths: no crash, no hang");

// Every byte of the stream corrupted, one at a time.
$rejected = 0; $decoded = 0; $k = 0;
while ($k < $n) {
    $mut = []; $j = 0; while ($j < $n) { push($mut, $zb[$j]); $j = $j + 1; }
    $mut[$k] = ($mut[$k] + 137) % 256;
    $bOut = null; $bFail = false;
    try { $bOut = binflate.inflate($mut, 200000); } catch ($e) { $bFail = true; }
    if ($bFail) { $rejected = $rejected + 1; } else { $decoded = $decoded + 1; }
    if ($NATIVE) {
        // The two must AGREE about every corrupt stream: both reject, or both
        // return the same bytes. A disagreement is a bug in one of them.
        $nOut = null; $nFail = false;
        try { $nOut = frombytes($mut); $nOut = bytes(inflate($nOut, 200000)); } catch ($e) { $nFail = true; }
        if ($nFail != $bFail) {
            $R.f = $R.f + 1;
            print("  FAIL  byte " + str($k) + " corrupted: native " +
                  str($nFail) + " but bantu " + str($bFail));
        } else {
            if (!$bFail && str($nOut) != str($bOut)) {
                $R.f = $R.f + 1;
                print("  FAIL  byte " + str($k) + " corrupted: decoders disagree");
            }
        }
    }
    $k = $k + 1;
}
ok(true, "all " + str($n) + " single-byte corruptions: " + str($rejected) +
         " rejected, " + str($decoded) + " still decoded, none crashed");
if ($NATIVE) { ok(true, "native and pure-Bantu agree on all " + str($n) + " corruptions"); }

// ─── The published throughput figures ───────────────────────────────────
print("");
print("── throughput (the numbers docs/pillow-architecture.md §4.1 cites) ──");
def rate($tag, $file) {
    $z = readfile($DIR + $file);
    $t0 = clock();
    $o = binflate.inflate($z, null);
    $ms = clock() - $t0;
    $us = floor($ms * 1000000 / len($o)) / 1000;
    print("   pure Bantu  " + $tag + ": " + str(len($o)) + " bytes in " + str($ms) +
          " ms = " + str($us) + " us/byte");
    if ($NATIVE) {
        $t0 = clock();
        $o2 = inflate($z, null);
        $ms2 = clock() - $t0;
        print("   native      " + $tag + ": " + str(len($o2)) + " bytes in " + str($ms2) + " ms");
    }
    return $us;
}
rate("dynamic (PNG's path)", "struct_6.z");
rate("fixed Huffman       ", "english_1.z");
rate("stored              ", "random_6.z");

print("");
print("checks: " + str($R.p + $R.f) + "  passed: " + str($R.p) + "  failed: " + str($R.f));
if ($R.f == 0) { print("RESULT: ALL GREEN"); }
else { print("RESULT: FAILED"); }
