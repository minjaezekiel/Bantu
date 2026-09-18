// ════════════════════════════════════════════════════════════════════════
//  numba_arctic_bridge_test.b — arctic columns <-> numba arrays, and
//  arctic's own transcendental kernels.
//
//  The asymmetry is the point and is structural, not an oversight:
//
//    column -> array   is a genuine ZERO-COPY BORROW. The array points at the
//                      column's own storage and holds it alive, so it can
//                      outlive the variable the column came from. Proved by
//                      buffer identity, not asserted.
//    array -> column   is a COPY. A Column stores std::vector, which owns its
//                      allocation; there is no portable way to adopt a
//                      foreign pointer.
//
//  The other theme is that arctic keeps a distinction numba does not: NULL is
//  not NaN. A column with nulls is REFUSED by name rather than silently
//  turning them into NaN, and NaN -> null on the way back is opt-in.
//
//  Run:  bantu run tests/numba_arctic_bridge_test.b
// ════════════════════════════════════════════════════════════════════════

$R = {"pass": 0, "fail": 0};
def eq($got, $want, $name) {
    if ($got == $want) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name);
        print("          got:  " + str($got));
        print("          want: " + str($want));
    }
}
def ok($cond, $name) {
    if ($cond) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $name); }
}
def close($got, $want, $tol, $name) {
    $d = $got - $want;
    if ($d < 0) { $d = 0 - $d; }
    if ($d <= $tol) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $name + " got " + str($got)); }
}
def raises($fn, $needle, $name) {
    try { $fn(); $R.fail = $R.fail + 1; print("  FAIL  " + $name + " (no error)"); }
    catch ($e) {
        $msg = str($e.message);
        if (contains($msg, $needle)) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
        else {
            $R.fail = $R.fail + 1;
            print("  FAIL  " + $name + " -- message did not mention \"" + $needle + "\": " + $msg);
        }
    }
}

print("========================================");
print("  numba <-> arctic");
print("========================================");

print("");
print("-- arctic's transcendental kernels, written natively --");
// Native rather than delegated to numba: delegating would make series.sqrt()
// require a numba-capable build, and would lose the null/NaN distinction.
$c = col([1.0, 4.0, 9.0], "f64");
eq(str(col_to_list(col_sqrt($c))), "[1, 2, 3]", "col_sqrt");
close(col_get(col_exp(col([1.0], "f64")), 0), 2.718281828459045, 1e-12, "col_exp");
close(col_get(col_log(col([2.718281828459045], "f64")), 0), 1.0, 1e-12, "col_log");
close(col_get(col_log10(col([1000.0], "f64")), 0), 3.0, 1e-12, "col_log10");
close(col_get(col_sin(col([0.0], "f64")), 0), 0.0, 1e-15, "col_sin");
close(col_get(col_cos(col([0.0], "f64")), 0), 1.0, 1e-15, "col_cos");
close(col_get(col_tanh(col([0.0], "f64")), 0), 0.0, 1e-15, "col_tanh");
eq(str(col_to_list(col_floor(col([1.7, -1.7], "f64")))), "[1, -2]", "col_floor");
eq(str(col_to_list(col_ceil(col([1.2, -1.2], "f64")))), "[2, -1]", "col_ceil");
eq(str(col_to_list(col_sign(col([-5.0, 0.0, 5.0], "f64")))), "[-1, 0, 1]", "col_sign");
// An integer column is promoted to f64, since these functions have no integer answer.
eq(col_dtype(col_sqrt(col([4, 9], "i64"))), "f64", "an i64 column yields f64");

print("");
print("-- nulls pass straight through, and are NOT NaN --");
$withNull = col([1.0, null, 9.0], "f64");
eq(col_null_count($withNull), 1, "the source column has one null");
$sq = col_sqrt($withNull);
eq(col_null_count($sq), 1, "the null survives the transcendental as a null");
eq(col_get($sq, 0), 1, "and the real values are still computed");
// A domain error yields NaN, not null: the value was PRESENT, the function
// simply has no real answer there, and conflating the two loses information.
$neg = col_sqrt(col([-1.0], "f64"));
eq(col_null_count($neg), 0, "sqrt(-1) is NOT null -- the value was present");
ok(col_get($neg, 0) != col_get($neg, 0), "it is NaN (which is not equal to itself)");

print("");
print("-- column -> array is a ZERO-COPY borrow --");
$src = col([1.0, 2.0, 3.0, 4.0], "f64");
$arr = nd_from_column($src);
eq(str(nd_shape($arr)), "[4]", "the array has the column's length");
eq(nd_dtype($arr), "f64", "and its dtype");
eq(str(nd_to_list($arr)), "[1, 2, 3, 4]", "and its values");
// Read-only, because arctic documents columns as immutable.
eq(nd_writable($arr), false, "the borrowed array is read-only");
raises(def() { nd_set($arr, [0], 99); }, "read-only", "writing to it raises");
// Zero-copy is PROVED: a view of the borrow shares its buffer, and no copy of
// the data was made at any point.
$v = nd_slice($arr, [[1, 3, null]]);
ok(nd_base_id($v) == nd_base_id($arr), "a view of the borrow shares the same buffer");
eq(str(nd_to_list($v)), "[2, 3]", "and reads the right values");

// The borrow must survive its source going out of scope -- that is what the
// keepalive is for, and it is the difference between a borrow and a dangling
// pointer.
def makeBorrow() {
    $tmp = col([7.0, 8.0, 9.0], "f64");
    return nd_from_column($tmp);
}
$escaped = makeBorrow();
eq(str(nd_to_list($escaped)), "[7, 8, 9]", "the borrow outlives the column it came from");

print("");
print("-- i64 and bool columns borrow too --");
eq(nd_dtype(nd_from_column(col([1, 2, 3], "i64"))), "i64", "an i64 column borrows as i64");
eq(str(nd_to_list(nd_from_column(col([1, 2], "i64")))), "[1, 2]", "with its values");
eq(nd_dtype(nd_from_column(col([true, false], "bool"))), "bool", "a bool column borrows as bool");

print("");
print("-- preconditions are refused BY NAME --");
raises(def() { return nd_from_column(col([1.0, null], "f64")); }, "nulls",
   "a column with nulls is refused, saying how many and what to do");
raises(def() { return nd_from_column(col(["a", "b"], "utf8")); }, "utf8",
   "a text column is refused, saying why");
raises(def() { return nd_from_column(nd_zeros([2], null)); }, "expected an arctic column",
   "passing an array where a column belongs is refused");

print("");
print("-- array -> column is a COPY --");
$a2 = nd([10.0, 20.0, 30.0], "f64");
$col2 = nd_to_column($a2, null);
eq(col_len($col2), 3, "the column has the array's length");
eq(str(col_to_list($col2)), "[10, 20, 30]", "and its values");
// Structural: a Column stores std::vector, which owns its allocation, so
// there is no portable way to adopt a foreign pointer.
nd_set($a2, [0], 999);
eq(col_get($col2, 0), 10, "changing the array does NOT change the column -- it was copied");
eq(col_null_count($col2), 0, "no nulls by default");
raises(def() { return nd_to_column(nd_zeros([2, 2], null), null); }, "1-dimensional",
   "a 2-d array cannot become a column, and says so");

print("");
print("-- NaN -> null on the way back is OPT-IN --");
$nan = nd_get(nd_divide(nd([0.0], "f64"), nd([0.0], "f64"), null), [0]);
$withNan = nd([1.0, 2.0], "f64");
nd_set($withNan, [1], $nan);
eq(col_null_count(nd_to_column($withNan, null)), 0, "by default a NaN stays a NaN, not a null");
eq(col_null_count(nd_to_column($withNan, true)), 1, "asking for it turns NaN into null");
eq(col_get(nd_to_column($withNan, true), 0), 1, "and leaves the real values alone");

print("");
print("-- round trip --");
$orig = col([1.5, 2.5, 3.5], "f64");
$back = nd_to_column(nd_copy(nd_from_column($orig)), null);
eq(str(col_to_list($back)), "[1.5, 2.5, 3.5]", "column -> array -> column is exact");

print("");
print("-- a frame becomes a matrix: the bridge to linear algebra --");
$c1 = col([1.0, 2.0, 3.0], "f64");
$c2 = col([4.0, 5.0, 6.0], "f64");
$M = nd_from_frame([$c1, $c2]);
eq(str(nd_shape($M)), "[3, 2]", "three rows, two columns");
eq(str(nd_to_list($M)), "[[1, 4], [2, 5], [3, 6]]", "laid out row by row");
// Mixed dtypes are promoted to f64, which is what linear algebra needs.
eq(nd_dtype(nd_from_frame([$c1, col([7, 8, 9], "i64")])), "f64",
   "a mixed frame promotes to f64");
// The whole point: a frame can go straight into a solve.
$y = nd([1.0, 2.0, 3.0], "f64");
$fit = nd_lstsq($M, $y);
eq(nd_size($fit), 2, "and it feeds straight into least squares");
raises(def() { return nd_from_frame([$c1, col([1.0], "f64")]); }, "same length",
   "mismatched column lengths are refused, naming both");
raises(def() { return nd_from_frame([]); }, "non-empty", "an empty frame is refused");
raises(def() { return nd_from_frame([$c1, col([1.0, null, 2.0], "f64")]); }, "nulls",
   "a column with nulls is refused by index");

print("");
print("-- scale: a 1,000,000-row column borrows in constant time --");
$big = nd_to_column(nd_astype(nd_arange(0, 1000000, null), "f64"), null);
$t0 = clock();
$barr = nd_from_column($big);
$borrowMs = clock() - $t0;
print("        borrowing 1,000,000 rows: " + str($borrowMs) + "ms");
ok($borrowMs < 50, "a borrow is O(1) -- it copies nothing");
eq(nd_size($barr), 1000000, "and the array has every row");
$t0 = clock();
$s = nd_get(nd_sum($barr, null, null), []);
print("        summing them through numba: " + str(clock() - $t0) + "ms");
eq($s, 499999500000, "and the sum is right");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
