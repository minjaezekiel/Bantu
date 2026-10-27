// ════════════════════════════════════════════════════════════════════════
//  tests/pillow_deflate_test.b — pillow Phase P1, the encoder.
//
//  pillow/deflate.b is the write side of the Bantu-only path: with no native
//  help at all, the package can still produce a valid zlib stream, which is
//  what a PNG is. It emits stored blocks (level 0) and fixed-Huffman blocks
//  with LZ77 matching (levels 1-9); it does not build dynamic Huffman
//  tables, which is why its output is larger than bp_zlib's and still a
//  perfectly ordinary zlib stream.
//
//  Checked three ways:
//    * every level round-trips through pillow/inflate.b   (Bantu -> Bantu)
//    * every level round-trips through native inflate()   (Bantu -> C++)
//    * tests/pillow_deflate_differential.sh has Python's zlib read it too
//
//  Run:  bantu run tests/pillow_deflate_test.b
// ════════════════════════════════════════════════════════════════════════

include "./pillow/deflate.b" as deflate;
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
def throws($fn, $name) {
    $caught = false;
    try { $fn(); } catch ($e) { $caught = true; }
    if ($caught) { $R.p = $R.p + 1; }
    else { $R.f = $R.f + 1; print("  FAIL  (expected an error) " + $name); }
}

$NATIVE = false;
try { $NATIVE = has_native("inflate"); } catch ($e) { $NATIVE = false; }

// ─── every level, through every decoder ─────────────────────────────────
print("── round trips: all 10 levels x 2 decoders ──");

def trip($s, $name) {
    $lv = 0;
    while ($lv <= 9) {
        $z = deflate.zlib($s, $lv);
        $viaBantu = binflate.inflate_str($z, null);
        if ($viaBantu == $s) { $R.p = $R.p + 1; }
        else {
            $R.f = $R.f + 1;
            print("  FAIL  bantu decoder, " + $name + " L" + str($lv) +
                  ": got " + str(len($viaBantu)) + " of " + str(len($s)) + " bytes");
        }
        if ($NATIVE) {
            $viaNative = inflate($z, null);
            if ($viaNative == $s) { $R.p = $R.p + 1; }
            else {
                $R.f = $R.f + 1;
                print("  FAIL  native decoder, " + $name + " L" + str($lv) +
                      ": got " + str(len($viaNative)) + " of " + str(len($s)) + " bytes");
            }
        }
        $lv = $lv + 1;
    }
}

trip("", "empty");
trip("a", "one byte");
trip("ab", "two bytes");
trip("abc", "the minimum match length");
trip("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "32 identical bytes");
trip("hello, hello, hello, hello, hello!", "repetitive");
$all = []; $i = 0;
while ($i < 256) { push($all, $i); $i = $i + 1; }
trip(frombytes($all), "all 256 byte values");
$parts = []; $i = 0;
while ($i < 200) { push($parts, "The quick brown fox jumps over the lazy dog. "); $i = $i + 1; }
$english = join($parts, "");
trip($english, "9,000 bytes of English");
$runl = []; $i = 0;
while ($i < 5000) { push($runl, 65); $i = $i + 1; }
trip(frombytes($runl), "5,000 identical bytes");
trip(frombytes(randbytes(4000)), "4,000 CSPRNG bytes");
// a long match at the maximum distance the window allows
$far = []; $i = 0;
while ($i < 300) { push($far, 88); $i = $i + 1; }
while ($i < 5300) { push($far, ($i * 31) % 256); $i = $i + 1; }
$i = 0;
while ($i < 300) { push($far, 88); $i = $i + 1; }
trip(frombytes($far), "a match 5,000 bytes back");

// ─── every length, so no boundary in the block machinery hides ──────────
print("── every length 0..200, at every level ──");
$bad = 0; $lv = 0;
while ($lv <= 9) {
    $L = 0;
    while ($L <= 200) {
        $p2 = []; $i = 0;
        while ($i < $L) { push($p2, chr(65 + ($i % 7))); $i = $i + 1; }
        $s = join($p2, "");
        if (binflate.inflate_str(deflate.zlib($s, $lv), null) != $s) { $bad = $bad + 1; }
        $L = $L + 1;
    }
    $lv = $lv + 1;
}
eq($bad, 0, "all 201 lengths x 10 levels round-trip");

// ─── containers ─────────────────────────────────────────────────────────
print("── containers ──");

eq(binflate.inflate_str(deflate.gzip("gzip round trip please", 6), null),
   "gzip round trip please", "gzip, bantu decoder");
if ($NATIVE) {
    eq(gunzip(deflate.gzip($english, 6), null), $english, "gzip 9k, native gunzip");
    eq(inflate(deflate.gzip("", 6), null), "", "gzip of nothing");
}
eq(binflate.inflate_str(deflate.raw("bare deflate body", 6), null),
   "bare deflate body", "bare DEFLATE");

// a zlib header this writes must pass the RFC 1950 check
$z = bytes(deflate.zlib("header check", 6));
eq($z[0] % 16, 8, "CMF says DEFLATE");
eq(($z[0] * 256 + $z[1]) % 31, 0, "the header check byte is valid");

// ─── checksums agree with the interpreter's own ─────────────────────────
print("── checksums ──");

eq(deflate.crc32("The quick brown fox jumps over the lazy dog"),
   bp_crc32("The quick brown fox jumps over the lazy dog"), "crc32 matches bp_crc32");
eq(deflate.crc32(""), bp_crc32(""), "crc32 of empty matches");
eq(deflate.crc32("a"), bp_crc32("a"), "crc32 of one byte matches");
eq(deflate.adler32("abc"), bp_adler32("abc"), "adler32 matches bp_adler32");
eq(deflate.adler32(""), bp_adler32(""), "adler32 of empty matches");
// over every byte value, so no table entry is wrong
eq(deflate.crc32(frombytes($all)), bp_crc32(frombytes($all)), "crc32 over all 256 values");
eq(deflate.adler32(frombytes($all)), bp_adler32(frombytes($all)), "adler32 over all 256 values");

// ─── failure modes ──────────────────────────────────────────────────────
print("── failure modes ──");

throws(def() { return deflate.raw("x", -1); }, "negative level");
throws(def() { return deflate.raw("x", 10); }, "level above 9");
throws(def() { return deflate.raw(42, 6); }, "src is not bytes");

// ─── what it costs in size, stated rather than hidden ───────────────────
print("");
print("── size, 9,000 bytes of English ──");
print("   level 0 (stored):        " + str(len(deflate.zlib($english, 0))) + " bytes");
print("   level 1 (fixed, fast):   " + str(len(deflate.zlib($english, 1))) + " bytes");
print("   level 6 (fixed, lazy):   " + str(len(deflate.zlib($english, 6))) + " bytes");
print("   level 9 (fixed, deepest):" + str(len(deflate.zlib($english, 9))) + " bytes");
print("   bp_zlib (native, dynamic Huffman): " + str(len(bp_zlib($english))) + " bytes");
print("   -> fixed Huffman costs ~" +
      str(floor((len(deflate.zlib($english, 6)) * 100 / len(bp_zlib($english))) - 100)) +
      "% more than dynamic, as documented");

print("");
print("checks: " + str($R.p + $R.f) + "  passed: " + str($R.p) + "  failed: " + str($R.f));
if ($R.f == 0) { print("RESULT: ALL GREEN"); }
else { print("RESULT: FAILED"); }
