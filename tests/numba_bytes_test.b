// ════════════════════════════════════════════════════════════════════════
//  tests/numba_bytes_test.b — nd_frombytes / nd_tobytes.
//
//  The bridge between a byte buffer and an array. nd_fromfile has existed
//  since lugha needed model weights, but its dtypes are f64/i64/bool, all
//  eight bytes wide -- so a BYTE buffer had no way into an array at all, and
//  the only route was readfile -> bytes() -> nd(list): measured at 479 ms and
//  a ~100 MB transient per megabyte. These close that.
//
//  `elem` names the WIRE type, independent of the array's dtype, and always
//  with an explicit byte order, because every format that matters specifies
//  one and the machine's is not it.
//
//  Run:  bantu run tests/numba_bytes_test.b
// ════════════════════════════════════════════════════════════════════════

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

eq(has_native("ndbytes"), true, "has_native(ndbytes)");

// ─── widths and signedness ──────────────────────────────────────────────
print("── element types ──");

eq(nd_to_list(nd_frombytes(frombytes([0, 1, 127, 128, 255]), "u8", null)),
   [0, 1, 127, 128, 255], "u8 is unsigned");
eq(nd_to_list(nd_frombytes(frombytes([0, 1, 127, 128, 255]), "i8", null)),
   [0, 1, 127, -128, -1], "i8 sign-extends");
eq(nd_dtype(nd_frombytes(frombytes([1]), "u8", null)), "i64", "integer elems give i64");

eq(nd_to_list(nd_frombytes(frombytes([1, 0, 0, 1]), "u16le", null)), [1, 256], "u16le");
eq(nd_to_list(nd_frombytes(frombytes([1, 0, 0, 1]), "u16be", null)), [256, 1], "u16be");
eq(nd_to_list(nd_frombytes(frombytes([1, 0]), "u16", null)), [1], "u16 defaults to le");
eq(nd_to_list(nd_frombytes(frombytes([255, 255]), "i16le", null)), [-1], "i16le sign-extends");
eq(nd_to_list(nd_frombytes(frombytes([128, 0]), "i16be", null)), [-32768], "i16be most negative");

eq(nd_to_list(nd_frombytes(frombytes([1, 0, 0, 0]), "u32le", null)), [1], "u32le");
eq(nd_to_list(nd_frombytes(frombytes([0, 0, 0, 1]), "u32be", null)), [1], "u32be");
eq(nd_to_list(nd_frombytes(frombytes([255, 255, 255, 255]), "u32le", null)), [4294967295], "u32 full range");
eq(nd_to_list(nd_frombytes(frombytes([255, 255, 255, 255]), "i32le", null)), [-1], "i32le sign-extends");
eq(nd_to_list(nd_frombytes(frombytes([128, 0, 0, 0]), "i32be", null)), [-2147483648], "i32be most negative");

// ─── float ──────────────────────────────────────────────────────────────
print("── float ──");

eq(nd_dtype(nd_frombytes(frombytes([0, 0, 0, 0]), "f32le", null)), "f64", "f32 gives f64 dtype");
// 1.0f is 0x3F800000
eq(nd_to_list(nd_frombytes(frombytes([0, 0, 128, 63]), "f32le", null)), [1.0], "f32le 1.0");
eq(nd_to_list(nd_frombytes(frombytes([63, 128, 0, 0]), "f32be", null)), [1.0], "f32be 1.0");
// 1.0 double is 0x3FF0000000000000
eq(nd_to_list(nd_frombytes(frombytes([0,0,0,0,0,0,240,63]), "f64le", null)), [1.0], "f64le 1.0");
eq(nd_to_list(nd_frombytes(frombytes([63,240,0,0,0,0,0,0]), "f64be", null)), [1.0], "f64be 1.0");

// ─── shape ──────────────────────────────────────────────────────────────
print("── shape ──");

$img = nd_frombytes(frombytes([1,2,3, 4,5,6, 7,8,9, 10,11,12]), "u8", [2, 2, 3]);
eq(nd_shape($img), [2, 2, 3], "declared shape");
eq(nd_get($img, [0, 0, 0]), 1, "first element");
eq(nd_get($img, [1, 1, 2]), 12, "last element");
eq(nd_get($img, [1, 0, 2]), 9, "row-major order");
eq(nd_shape(nd_frombytes(frombytes([1,2,3]), "u8", null)), [3], "no shape means one axis");

// ─── round trips ────────────────────────────────────────────────────────
print("── round trips ──");

eq(bytes(nd_tobytes($img, "u8")), [1,2,3,4,5,6,7,8,9,10,11,12], "u8 round trip");
eq(bytes(nd_tobytes(nd([1, 2], "i64"), "u16be")), [0,1,0,2], "u16be out");
eq(bytes(nd_tobytes(nd([1, 2], "i64"), "u16le")), [1,0,2,0], "u16le out");
eq(bytes(nd_tobytes(nd([-1], "i64"), "i16le")), [255,255], "i16le out");
eq(nd_to_list(nd_frombytes(nd_tobytes(nd([1.5, -2.25, 0.0], "f64"), "f32le"), "f32le", null)),
   [1.5, -2.25, 0.0], "f32 exact values survive");
eq(nd_to_list(nd_frombytes(nd_tobytes(nd([1.1, -2.7e100], "f64"), "f64be"), "f64be", null)),
   [1.1, -2.7e100], "f64 round trip is exact");

// every byte value, every integer width, both orders
$ords = ["le", "be"];
$widths = ["u8", "u16", "u32"];
$bad = 0; $wi = 0;
while ($wi < len($widths)) {
    $oi = 0;
    while ($oi < len($ords)) {
        $elem = $widths[$wi];
        if ($elem != "u8") { $elem = $elem + $ords[$oi]; }
        $vals = []; $v = 0;
        while ($v < 256) { push($vals, $v); $v = $v + 1; }
        $a = nd($vals, "i64");
        $b = nd_frombytes(nd_tobytes($a, $elem), $elem, null);
        if (str(nd_to_list($b)) != str($vals)) { $bad = $bad + 1; }
        $oi = $oi + 1;
    }
    $wi = $wi + 1;
}
eq($bad, 0, "all 256 byte values through every width and order");

// ─── views write in LOGICAL order, not buffer order ─────────────────────
print("── views ──");

$base = nd([[1,2,3],[4,5,6]], "i64");
eq(bytes(nd_tobytes($base, "u8")), [1,2,3,4,5,6], "contiguous");
eq(bytes(nd_tobytes(nd_flip($base, 1), "u8")), [3,2,1,6,5,4], "flipped view");
eq(bytes(nd_tobytes(nd_flip($base, 0), "u8")), [4,5,6,1,2,3], "flipped rows");
eq(bytes(nd_tobytes(nd_transpose($base), "u8")), [1,4,2,5,3,6], "transposed view");
eq(bytes(nd_tobytes(nd_slice($base, [[0,2],[1,3]]), "u8")), [2,3,5,6], "sliced view");
eq(bytes(nd_tobytes(nd_slice($base, [[0,2],[0,3,2]]), "u8")), [1,3,4,6], "strided view");

// ─── integer narrowing WRAPS, as a cast does; it does not clamp ─────────
print("── narrowing ──");

eq(bytes(nd_tobytes(nd([256, 257, -1], "i64"), "u8")), [0, 1, 255], "u8 narrowing wraps");
eq(bytes(nd_tobytes(nd([65536], "i64"), "u16le")), [0, 0], "u16 narrowing wraps");
// a caller that wants clamping says so, and it is free
eq(bytes(nd_tobytes(nd_clip(nd([-5, 300], "i64"), 0, 255), "u8")), [0, 255], "nd_clip clamps first");

// ─── failure modes ──────────────────────────────────────────────────────
print("── failure modes ──");

throws(def() { return nd_frombytes(frombytes([1,2,3]), "u16le", null); }, "odd byte count for u16");
throws(def() { return nd_frombytes(frombytes([1,2,3]), "u32le", null); }, "3 bytes for u32");
throws(def() { return nd_frombytes(frombytes([1,2]), "u24", null); }, "unknown elem type");
throws(def() { return nd_frombytes(frombytes([1,2]), "", null); }, "an explicit empty elem type is rejected");
eq(nd_to_list(nd_frombytes(frombytes([1,2]), null, null)), [1, 2], "omitted elem type defaults to u8");
throws(def() { return nd_frombytes(frombytes([1,2,3,4]), "u8", [3]); }, "shape too small");
throws(def() { return nd_frombytes(frombytes([1,2,3,4]), "u8", [5]); }, "shape too large");
throws(def() { return nd_frombytes(42, "u8", null); }, "src is not bytes");
throws(def() { return nd_tobytes(nd([1], "i64"), "nope"); }, "unknown elem type out");
throws(def() { return nd_frombytes(["a", "b"], "u8", null); }, "byte-list with non-numbers");

// an empty buffer is legal and gives an empty array
eq(nd_size(nd_frombytes("", "u8", null)), 0, "empty buffer");
eq(nd_tobytes(nd_frombytes("", "u8", null), "u8"), "", "empty round trip");

// a byte-list source works too
eq(nd_to_list(nd_frombytes([1, 2, 3], "u8", null)), [1, 2, 3], "byte-list source");

// ─── it is actually fast, which is the whole point ──────────────────────
print("── throughput ──");

$chunk = "0123456789abcdef";
$k = 0;
while ($k < 16) { $chunk = $chunk + $chunk; $k = $k + 1; }     // 1 MiB
$t0 = clock();
$big = nd_frombytes($chunk, "u8", null);
$ms = clock() - $t0;
print("   nd_frombytes 1 MiB: " + str($ms) + " ms  (bytes()+nd() measured 479 ms)");
eq(nd_size($big), 1048576, "1 MiB element count");
$t0 = clock();
$back = nd_tobytes($big, "u8");
print("   nd_tobytes   1 MiB: " + str(clock() - $t0) + " ms");
eq($back == $chunk, true, "1 MiB round trip is identical");
// the gate: a megabyte must cost less than a tenth of what the list route did
eq($ms < 48, true, "at least 10x faster than the list route");

// an 8 MiB buffer -- the 1080p RGBA case the list route could not do at all
$big8 = $chunk + $chunk + $chunk + $chunk + $chunk + $chunk + $chunk + $chunk;
$t0 = clock();
$a8 = nd_frombytes($big8, "u8", [1024, 1024, 8]);
print("   nd_frombytes 8 MiB shaped: " + str(clock() - $t0) + " ms");
eq(nd_size($a8), 8388608, "8 MiB element count");
eq(nd_shape($a8), [1024, 1024, 8], "8 MiB shape");

print("");
print("checks: " + str($R.p + $R.f) + "  passed: " + str($R.p) + "  failed: " + str($R.f));
if ($R.f == 0) { print("RESULT: ALL GREEN"); }
else { print("RESULT: FAILED"); }
