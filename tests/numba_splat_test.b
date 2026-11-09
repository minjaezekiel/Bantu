// ════════════════════════════════════════════════════════════════════════
//  numba_splat_test.b — tier 1 (the splat path) against tier 2.
//
//  Phase A1 of btorch wired the tier1/tier1r kernels into applyBinary. They
//  had been written, commented, and never called from anywhere: a tree-wide
//  search found exactly two mentions, the two definitions. Every scalar
//  operand therefore fell through to tier 2's strided odometer and ran 4.5x
//  slower than the same op with two full arrays -- while doing LESS memory
//  traffic.
//
//  A1 is a new numerical path, so the only thing that matters here is that it
//  is the SAME path arithmetically. Every case below computes a result two
//  ways:
//
//      op($arr, 3.0)                 <- tier 1, the new code
//      op($arr, nd_full(shape, 3.0)) <- tier 0, untouched by A1
//
//  and asserts they are EXACTLY equal, element for element. Not allclose:
//  these are the same IEEE operation on the same bits, so any difference at
//  all is a defect. Both operand orders, every dtype, and the shapes that
//  decide which tier is taken.
//
//  Run:  bantu run tests/numba_splat_test.b
// ════════════════════════════════════════════════════════════════════════

$R = {"pass": 0, "fail": 0};
def ok($cond, $name) {
    if ($cond) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $name); }
}

// Exact equality over every element, and the shapes must match too.
def same($x, $y, $name) {
    if (nd_shape($x) != nd_shape($y)) {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name + "  shape " + str(nd_shape($x)) + " vs " + str(nd_shape($y)));
        return null;
    }
    // Exact element-wise comparison -- NOT allclose, which would hide a real
    // difference between two paths that are meant to be the same IEEE
    // operation. But NaN == NaN is false, so an exact comparison alone would
    // call two identical NaN results different. The honest predicate is
    // `equal OR both-NaN`, elementwise.
    $eq      = nd_equal($x, $y, null);
    $bothnan = nd_logical_and(nd_isnan($x, null), nd_isnan($y, null), null);
    ok(nd_all(nd_logical_or($eq, $bothnan, null), null, null), $name);
    return null;
}

print("== tier 1 vs tier 0: f64, scalar on the right ==");
$a = nd_linspace(-4.0, 4.0, 1000);
$s = 3.0;
$S = nd_full([1000], 3.0);
same(nd_add($a, $s, null),           nd_add($a, $S, null),           "add(arr, 3.0)");
same(nd_subtract($a, $s, null),      nd_subtract($a, $S, null),      "subtract(arr, 3.0)");
same(nd_multiply($a, $s, null),      nd_multiply($a, $S, null),      "multiply(arr, 3.0)");
same(nd_divide($a, $s, null),        nd_divide($a, $S, null),        "divide(arr, 3.0)");
same(nd_power($a, $s, null),         nd_power($a, $S, null),         "power(arr, 3.0)");
same(nd_maximum($a, $s, null),       nd_maximum($a, $S, null),       "maximum(arr, 3.0)");
same(nd_minimum($a, $s, null),       nd_minimum($a, $S, null),       "minimum(arr, 3.0)");
same(nd_mod($a, $s, null),           nd_mod($a, $S, null),           "mod(arr, 3.0)");
same(nd_floor_divide($a, $s, null),  nd_floor_divide($a, $S, null),  "floor_divide(arr, 3.0)");
same(nd_atan2($a, $s, null),         nd_atan2($a, $S, null),         "atan2(arr, 3.0)");
same(nd_hypot($a, $s, null),         nd_hypot($a, $S, null),         "hypot(arr, 3.0)");
same(nd_copysign($a, $s, null),      nd_copysign($a, $S, null),      "copysign(arr, 3.0)");

print("");
print("== comparisons emit bool, and must agree exactly ==");
same(nd_greater($a, $s, null),       nd_greater($a, $S, null),       "greater(arr, 3.0)");
same(nd_greater_equal($a, $s, null), nd_greater_equal($a, $S, null), "greater_equal(arr, 3.0)");
same(nd_less($a, $s, null),          nd_less($a, $S, null),          "less(arr, 3.0)");
same(nd_less_equal($a, $s, null),    nd_less_equal($a, $S, null),    "less_equal(arr, 3.0)");
same(nd_equal($a, $s, null),         nd_equal($a, $S, null),         "equal(arr, 3.0)");
same(nd_not_equal($a, $s, null),     nd_not_equal($a, $S, null),     "not_equal(arr, 3.0)");
ok(nd_dtype(nd_greater($a, $s, null)) == "bool", "comparison through tier 1 still yields bool");

print("");
print("== scalar on the LEFT takes tier1r, a different kernel ==");
same(nd_subtract($s, $a, null),      nd_subtract($S, $a, null),      "subtract(3.0, arr)");
same(nd_divide($s, $a, null),        nd_divide($S, $a, null),        "divide(3.0, arr)");
same(nd_power($s, $a, null),         nd_power($S, $a, null),         "power(3.0, arr)");
same(nd_mod($s, $a, null),           nd_mod($S, $a, null),           "mod(3.0, arr)");
same(nd_floor_divide($s, $a, null),  nd_floor_divide($S, $a, null),  "floor_divide(3.0, arr)");
same(nd_atan2($s, $a, null),         nd_atan2($S, $a, null),         "atan2(3.0, arr)");
same(nd_greater($s, $a, null),       nd_greater($S, $a, null),       "greater(3.0, arr)");
same(nd_less($s, $a, null),          nd_less($S, $a, null),          "less(3.0, arr)");
ok(!nd_array_equal(nd_subtract($s, $a, null), nd_subtract($a, $s, null)),
   "tier1r is not tier1 -- subtract(3,a) differs from subtract(a,3)");

print("");
print("== i64 keeps the integer kernel and stays EXACT ==");
$ai = nd_astype(nd_arange(-500, 500, null), "i64");
$si = 7;
$Si = nd_astype(nd_full([1000], 7), "i64");
same(nd_add($ai, $si, null),      nd_add($ai, $Si, null),      "i64 add(arr, 7)");
same(nd_multiply($ai, $si, null), nd_multiply($ai, $Si, null), "i64 multiply(arr, 7)");
same(nd_subtract($ai, $si, null), nd_subtract($ai, $Si, null), "i64 subtract(arr, 7)");
same(nd_mod($ai, $si, null),      nd_mod($ai, $Si, null),      "i64 mod(arr, 7)");
same(nd_floor_divide($ai, $si, null), nd_floor_divide($ai, $Si, null), "i64 floor_divide(arr, 7)");
// A bare Bantu number is a double, so `$ai + 7` is i64+f64 and promotes to
// f64. Confirmed identical on the pre-A1 build, so it is existing promotion
// behaviour and not something tier 1 introduced. An actual i64 size-1 operand
// is the case that must stay integral:
ok(nd_dtype(nd_add($ai, $Si, null)) == "i64", "i64 arr + i64 size-1 stays i64 through tier 1");
ok(nd_dtype(nd_add($ai, $si, null)) == "f64", "i64 arr + bare number promotes to f64 (pre-existing, unchanged)");
// A number in Bantu is a double, so `$ai + 7` hands an f64 scalar to an i64
// array. The promotion rule must not change just because tier 1 took the call.
ok(nd_dtype(nd_add($ai, 7.5, null)) == "f64", "i64 array + f64 scalar promotes to f64 as before");
same(nd_add($ai, 7.5, null), nd_add($ai, nd_full([1000], 7.5), null), "i64 arr + 7.5 matches tier 0");

print("");
print("== the size-1 operand may be an array, a 0-d array, or a bare number ==");
$one   = nd_full([1], 3.0);          // size 1, ndim 1
$zerod = nd_reshape(nd_full([1], 3.0), []);   // size 1, ndim 0
same(nd_add($a, $one, null),   nd_add($a, $S, null), "add(arr, [3.0]) == add(arr, full)");
same(nd_add($a, $zerod, null), nd_add($a, $S, null), "add(arr, 0-d 3.0) == add(arr, full)");
same(nd_add($a, 3.0, null),    nd_add($a, $one, null), "bare 3.0 and [3.0] agree");
// A size-1 operand of a DIFFERENT dtype: the single element is converted, so
// this must still take tier 1 and still be right.
$onei = nd_astype(nd_full([1], 3), "i64");
same(nd_add($a, $onei, null), nd_add($a, $S, null), "f64 arr + i64 [3] matches");

print("");
print("== shapes: tier 1 must not fire where it would be wrong ==");
// A genuine broadcast (not size 1) must still go to tier 2 and be correct.
$m = nd_reshape(nd_linspace(0.0, 11.0, 12), [4, 3]);
$row = nd_reshape(nd_linspace(1.0, 3.0, 3), [3]);
$expect = nd_add($m, nd_broadcast_to(nd_reshape($row, [1, 3]), [4, 3]), null);
same(nd_add($m, $row, null), $expect, "row broadcast [4,3]+[3] unchanged (tier 2)");
ok(nd_shape(nd_add($m, 1.0, null)) == [4, 3], "2-d array + scalar keeps shape");
same(nd_add($m, 1.0, null), nd_add($m, nd_full([4,3], 1.0), null), "2-d + scalar matches tier 0");
// Both sides size 1: nothing to splat. Must still be right.
same(nd_add($one, $one, null), nd_full([1], 6.0), "size-1 + size-1 still correct");
ok(nd_shape(nd_add($zerod, $zerod, null)) == [], "0-d + 0-d stays 0-d");

print("");
print("== a NON-contiguous array side must NOT take tier 1 ==");
// tier 1 reads the array side as a raw pointer, so it may only fire when that
// side is C-contiguous. A strided view must fall through to tier 2 -- if it
// took tier 1 it would read the wrong elements, which is why this is here.
$strided = nd_slice($m, [[0, 4, 2], [0, 3, 1]]);     // every other row
ok(!nd_is_contiguous($strided), "the test's strided view really is non-contiguous");
same(nd_add($strided, 10.0, null),
     nd_add(nd_ascontiguous($strided), 10.0, null),
     "strided arr + scalar == contiguous copy + scalar");
$trans = nd_transpose($m, null);
ok(!nd_is_contiguous($trans), "the test's transposed view really is non-contiguous");
same(nd_add($trans, 10.0, null),
     nd_add(nd_ascontiguous($trans), 10.0, null),
     "transposed arr + scalar == contiguous copy + scalar");
$flipped = nd_flip($m, null);
same(nd_multiply($flipped, 2.0, null),
     nd_multiply(nd_ascontiguous($flipped), 2.0, null),
     "flipped (negative stride) arr * scalar is correct");

print("");
print("== out= still honours its gates on the tier 1 path ==");
$dst = nd_zeros([1000]);
$Z = nd_full([1000], 0.0);
$got = nd_maximum($a, 0.0, $dst);
same($dst, nd_maximum($a, $Z, null), "out= receives the tier 1 result");
ok(nd_base_id($got) == nd_base_id($dst), "out= returns the caller's own buffer");
// in-place: out IS the input, which exactlyAliases and is allowed
$inplace = nd_copy($a, null);
nd_multiply($inplace, 2.0, $inplace);
same($inplace, nd_multiply($a, 2.0, null), "in-place out=self through tier 1 is correct");
// a broadcast_to destination is read-only and must be refused, tier or no tier
$ro = nd_broadcast_to(nd_full([1], 1.0), [1000]);
try {
    nd_add($a, 1.0, $ro);
    $R.fail = $R.fail + 1; print("  FAIL  writing to a broadcast_to out= should raise");
} catch ($e) {
    ok(true, "broadcast_to as out= still raises on the tier 1 path");
}

print("");
print("== edge values: tier 1 must not reorder or round anything ==");
$DIVZ = 0.0;
$edge = nd([0.0, 0.0 - 0.0, 1e308, 0.0 - 1e308, 1e-308, 0.5, 0.0 - 0.5], null);
same(nd_multiply($edge, 2.0, null),  nd_multiply($edge, nd_full([7], 2.0), null),  "overflow to inf matches");
same(nd_divide($edge, $DIVZ, null),  nd_divide($edge, nd_full([7], 0.0), null),    "division by zero matches (inf/nan)");
same(nd_add($edge, 1e308, null),     nd_add($edge, nd_full([7], 1e308), null),     "near-overflow add matches");
// NaN as the splatted scalar: max(x, NaN) is an IEEE question, and tier 1
// must answer it exactly as tier 0 does rather than reordering the operands.
$withnan = nd([1.0, 0.0, 2.0], null);
// Bantu's scalar `/` raises on a zero divisor, so the NaN comes from an array
// op and is read back out as a number.
$nanv = nd_get(nd_divide(nd_full([1], 0.0), nd_full([1], 0.0), null), [0]);
ok($nanv != $nanv, "the test really does have a NaN to work with");
$infv = nd_get(nd_divide(nd_full([1], 1.0), nd_full([1], 0.0), null), [0]);
ok($infv > 1e308, "the test really does have an infinity to work with");
same(nd_maximum($withnan, $nanv, null), nd_maximum($withnan, nd_full([3], $nanv), null),
     "NaN as the scalar operand matches tier 0");
same(nd_add($withnan, $nanv, null), nd_add($withnan, nd_full([3], $nanv), null),
     "NaN propagates through tier 1 add as through tier 0");

print("");
print("== large: the vectorized loop and the odometer agree at scale ==");
$big = nd_linspace(0.0, 1.0, 1000000);
$bigS = nd_full([1000000], 0.25);
same(nd_add($big, 0.25, null),     nd_add($big, $bigS, null),     "1M add(arr, 0.25)");
same(nd_maximum($big, 0.25, null), nd_maximum($big, $bigS, null), "1M maximum(arr, 0.25) -- the relu shape");
same(nd_multiply($big, 0.25, null),nd_multiply($big, $bigS, null),"1M multiply(arr, 0.25)");
same(nd_greater($big, 0.25, null), nd_greater($big, $bigS, null), "1M greater(arr, 0.25)");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
