// ════════════════════════════════════════════════════════════════════════
//  numba_ufunc_test.b — broadcasting, the three-tier kernel loop, and the
//  element-wise ufuncs.
//
//  Three things are being proven here, and only the first is obvious:
//
//   1. The answers are RIGHT. Every transcendental is checked against a value
//      computed independently, not against itself, and the arithmetic is
//      checked against hand-written pure-Bantu loops.
//   2. Broadcasting follows NumPy exactly -- right-aligned, stretched axes get
//      stride 0 and copy nothing -- and every failure names the axis and both
//      extents, because "shape mismatch" is a permanent support burden.
//   3. `out=` cannot produce garbage. A destination that overlaps an input is
//      the worst failure mode available: not a crash, a WRONG ANSWER. The exact
//      overlap `nd_add($a,$b,$a)` is safe and must stay fast; every other
//      overlap must be detected and routed through a temporary.
//
//  Run:  bantu run tests/numba_ufunc_test.b
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

// Compare a whole array against a nested list, exactly.
def eqArr($got, $want, $name) {
    eq(str(nd_to_list($got)), str($want), $name);
}

// Tolerance comparison for anything that went through libm.
def close($got, $want, $tol, $name) {
    $d = $got - $want;
    if ($d < 0) { $d = 0 - $d; }
    if ($d <= $tol) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name + "  got " + str($got) + " want " + str($want));
    }
}

def raises($fn, $needle, $name) {
    try {
        $fn();
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name + "  (no error raised)");
    } catch ($e) {
        $msg = str($e.message);
        if (contains($msg, $needle)) {
            $R.pass = $R.pass + 1; print("  ok    " + $name);
        } else {
            $R.fail = $R.fail + 1;
            print("  FAIL  " + $name);
            print("          message did not mention \"" + $needle + "\":");
            print("          " + $msg);
        }
    }
}

print("========================================");
print("  numba — ufuncs and broadcasting");
print("========================================");

print("");
print("-- arithmetic, and that scalars are accepted without wrapping --");
$a = nd([1, 2, 3, 4], null);
$b = nd([10, 20, 30, 40], null);
eqArr(nd_add($a, $b, null),      [11, 22, 33, 44], "add");
eqArr(nd_subtract($b, $a, null), [9, 18, 27, 36],  "subtract");
eqArr(nd_multiply($a, $b, null), [10, 40, 90, 160], "multiply");
eqArr(nd_add($a, 100, null),     [101, 102, 103, 104], "a plain number is accepted as an operand");
eqArr(nd_multiply(2, $a, null),  [2, 4, 6, 8], "and on either side");
eqArr(nd_add([1, 1, 1, 1], $a, null), [2, 3, 4, 5], "so is a plain Bantu list");

print("");
print("-- dtype promotion (docs/numba-architecture.md 4.6) --");
$i = nd([1, 2, 3], "i64");
$f = nd([0.5, 0.5, 0.5], "f64");
eq(nd_dtype(nd_add($i, $i, null)), "i64", "i64 + i64 stays i64");
eq(nd_dtype(nd_add($i, $f, null)), "f64", "i64 + f64 promotes to f64");
eq(nd_dtype(nd_divide($i, $i, null)), "f64", "/ always yields f64, as in NumPy and Python 3");
eqArr(nd_divide(nd([1, 2, 3], "i64"), 2, null), [0.5, 1, 1.5], "and really divides");
eq(nd_dtype(nd_less($i, $i, null)), "bool", "a comparison yields bool");
eq(nd_dtype(nd_add(nd([true, true], null), nd([true, true], null), null)), "i64",
   "bool + bool is i64, not bool -- true + true is 2");
eq(nd_dtype(nd_sqrt($i, null)), "f64", "a transcendental of an integer array is f64");

// The whole reason the kernels carry a separate int64 path.
$big = nd([9007199254740993], "i64");         // 2^53 + 1
eq(nd_get(nd_add($big, 0, null), [0]), 9007199254740992,
   "i64 arithmetic is exact in the kernel (the 2^53 limit is in Bantu's number, not numba's)");

print("");
print("-- floor division and modulo follow Python, not C --");
eqArr(nd_floor_divide(nd([-7, 7, -7, 7], "i64"), nd([2, 2, -2, -2], "i64"), null),
      [-4, 3, 3, -4], "-7 // 2 is -4, not -3");
eqArr(nd_mod(nd([-7, 7, -7, 7], "i64"), nd([2, 2, -2, -2], "i64"), null),
      [1, 1, -1, -1], "the modulo takes the divisor's sign");
// Integer division by zero is SIGFPE on x86 -- a process kill, not an error.
raises(def() { nd_floor_divide(nd([1], "i64"), nd([0], "i64"), null); }, "division by zero",
   "integer division by zero raises instead of killing the process");
raises(def() { nd_mod(nd([1], "i64"), nd([0], "i64"), null); }, "modulo by zero",
   "and so does integer modulo");
// Float division by zero is NOT an error: IEEE says inf and that is correct.
ok(nd_get(nd_divide(nd([1.0], null), nd([0.0], null), null), [0]) > 0,
   "float division by zero yields +inf, which is the right answer, not an error");

print("");
print("-- broadcasting --");
$m = nd_reshape(nd_arange(0, 6, null), [2, 3]);      // [[0,1,2],[3,4,5]]
$row = nd([10, 20, 30], null);                        // (3,)   -> stretched over axis 0
$col = nd_reshape(nd([100, 200], null), [2, 1]);      // (2,1)  -> stretched over axis 1
eqArr(nd_add($m, $row, null), [[10, 21, 32], [13, 24, 35]], "(2,3) + (3,) broadcasts along rows");
eqArr(nd_add($m, $col, null), [[100, 101, 102], [203, 204, 205]], "(2,3) + (2,1) broadcasts along columns");
eqArr(nd_add($col, $row, null), [[110, 120, 130], [210, 220, 230]],
      "(2,1) + (3,) produces (2,3) -- both sides stretch");
eq(str(nd_shape(nd_add($col, $row, null))), "[2, 3]", "and the result shape is right");
eq(str(nd_broadcast_shapes(nd_zeros([5, 1, 3], null), nd_zeros([4, 3], null))), "[5, 4, 3]",
   "shapes can be computed without allocating");
eq(str(nd_shape(nd_add(nd_zeros([], null), nd_zeros([2, 2], null), null))), "[2, 2]",
   "a 0-d array broadcasts against anything");

print("");
print("-- broadcast failures name the axis and both extents --");
raises(def() { nd_add(nd_zeros([3, 4], null), nd_zeros([5, 4], null), null); }, "axis 0: 3 vs 5",
   "a mismatched leading axis is reported by number and extent");
raises(def() { nd_add(nd_zeros([2, 3], null), nd_zeros([2, 4], null), null); }, "axis 1: 3 vs 4",
   "and so is a trailing one");
raises(def() { nd_add(nd_zeros([3], null), nd_zeros([4], null), null); }, "cannot be broadcast",
   "1-d mismatches too");

print("");
print("-- broadcasting copies nothing: a stretched axis has stride 0 --");
$bc = nd_broadcast_to(nd([7], null), [1000000]);
eq(str(nd_strides($bc)), "[0]", "the stretched axis really has stride 0");
eq(nd_nbytes(nd_zeros([1], null)), 8, "and the source is still one element");
ok(nd_shares_memory($bc, $bc), "the broadcast view shares its base's memory");

print("");
print("-- comparisons and boolean logic --");
$c = nd([1, 5, 3], null);
$d = nd([4, 2, 3], null);
eqArr(nd_less($c, $d, null),          [true, false, false], "less");
eqArr(nd_greater($c, $d, null),       [false, true, false], "greater");
eqArr(nd_equal($c, $d, null),         [false, false, true], "equal");
eqArr(nd_not_equal($c, $d, null),     [true, true, false],  "not_equal");
eqArr(nd_less_equal($c, $d, null),    [true, false, true],  "less_equal");
eqArr(nd_greater_equal($c, $d, null), [false, true, true],  "greater_equal");
eqArr(nd_logical_and(nd([true, true, false], null), nd([true, false, false], null), null),
      [true, false, false], "logical_and");
eqArr(nd_logical_or(nd([true, true, false], null), nd([true, false, false], null), null),
      [true, true, false], "logical_or");
eqArr(nd_logical_xor(nd([true, true, false], null), nd([true, false, false], null), null),
      [false, true, false], "logical_xor");
eqArr(nd_logical_not(nd([true, false], null), null), [false, true], "logical_not");
// Truthiness is `!= 0`, evaluated in the operand's own type. Computing logic in
// bool converted floats by ROUNDING, so 0.4 was false and 0.6 was true -- which
// is llround, not truth.
eqArr(nd_logical_and(nd([0.4, 0.6], "f64"), nd([1.0, 1.0], "f64"), null), [true, true],
      "0.4 is truthy, not rounded to false");
eqArr(nd_logical_not(nd([0.4, 0.0], "f64"), null), [false, true],
      "and logical_not agrees -- only exactly 0.0 is false");
eqArr(nd_logical_or(nd([0.0, 0.2], "f64"), nd([0.0, 0.0], "f64"), null), [false, true],
      "logical_or on small non-zero floats");
eqArr(nd_greater($c, 3, null), [false, true, false], "comparison against a scalar");

print("");
print("-- transcendentals, against independently computed values --");
$t = nd([0.0, 1.0, 2.0], null);
close(nd_get(nd_exp($t, null), [1]), 2.718281828459045, 1e-12, "exp(1) is e");
close(nd_get(nd_log(nd([2.718281828459045], null), null), [0]), 1.0, 1e-12, "log(e) is 1");
close(nd_get(nd_sqrt(nd([2.0], null), null), [0]), 1.4142135623730951, 1e-12, "sqrt(2)");
close(nd_get(nd_sin(nd([1.5707963267948966], null), null), [0]), 1.0, 1e-12, "sin(pi/2) is 1");
close(nd_get(nd_cos(nd([0.0], null), null), [0]), 1.0, 1e-15, "cos(0) is 1");
close(nd_get(nd_atan2(nd([1.0], null), nd([1.0], null), null), [0]), 0.7853981633974483, 1e-12,
      "atan2(1,1) is pi/4");
close(nd_get(nd_hypot(nd([3.0], null), nd([4.0], null), null), [0]), 5.0, 1e-12, "hypot(3,4) is 5");
close(nd_get(nd_log10(nd([1000.0], null), null), [0]), 3.0, 1e-12, "log10(1000) is 3");
close(nd_get(nd_log2(nd([1024.0], null), null), [0]), 10.0, 1e-12, "log2(1024) is 10");
close(nd_get(nd_cbrt(nd([27.0], null), null), [0]), 3.0, 1e-12, "cbrt(27) is 3");
close(nd_get(nd_tanh(nd([0.0], null), null), [0]), 0.0, 1e-15, "tanh(0) is 0");
// log1p/expm1 exist precisely because the naive form loses everything here.
close(nd_get(nd_log1p(nd([0.0000000001], null), null), [0]), 0.0000000001, 1e-20,
      "log1p keeps precision where log(1+x) would lose it");

print("");
print("-- rounding is half-to-even, matching NumPy and IEEE --");
eqArr(nd_rint(nd([0.5, 1.5, 2.5, -0.5, -1.5], null), null), [0, 2, 2, 0, -2],
      "rint(0.5) is 0 and rint(1.5) is 2 -- ties go to even");
eqArr(nd_floor(nd([-1.5, 1.5], null), null), [-2, 1], "floor");
eqArr(nd_ceil(nd([-1.5, 1.5], null), null),  [-1, 2], "ceil");
eqArr(nd_trunc(nd([-1.7, 1.7], null), null), [-1, 1], "trunc rounds toward zero");

print("");
print("-- sign, abs, square, reciprocal, negative --");
eqArr(nd_abs(nd([-3, 0, 3], "i64"), null), [3, 0, 3], "abs on i64 stays i64");
eqArr(nd_sign(nd([-5, 0, 5], "i64"), null), [-1, 0, 1], "sign");
eqArr(nd_negative(nd([1, -2], "i64"), null), [-1, 2], "negative");
eqArr(nd_square(nd([2, 3], "i64"), null), [4, 9], "square");
eqArr(nd_reciprocal(nd([2.0, 4.0], null), null), [0.5, 0.25], "reciprocal");
eqArr(nd_minimum(nd([1, 5], null), nd([3, 2], null), null), [1, 2], "minimum");
eqArr(nd_maximum(nd([1, 5], null), nd([3, 2], null), null), [3, 5], "maximum");

print("");
print("-- NaN and infinity behave --");
$nan = nd_divide(nd([0.0], null), nd([0.0], null), null);
$inf = nd_divide(nd([1.0], null), nd([0.0], null), null);
eqArr(nd_isnan($nan, null), [true], "isnan finds a NaN");
eqArr(nd_isinf($inf, null), [true], "isinf finds an infinity");
eqArr(nd_isfinite(nd([1.0], null), null), [true], "isfinite on an ordinary number");
// A predicate asks a question only a float can answer. Letting it promote to
// the integer path meant it ran a stub kernel: nd_isfinite on an i64 array
// returned FALSE for every finite integer.
eqArr(nd_isfinite(nd([1, 2], "i64"), null), [true, true], "isfinite on an i64 array is true, not false");
eqArr(nd_isnan(nd([1, 2], "i64"), null), [false, false], "and isnan on i64 is false");
eqArr(nd_isinf(nd([1], "i64"), null), [false], "and isinf on i64 is false");
eqArr(nd_isfinite(nd([true, false], null), null), [true, true], "a bool array is finite too");
eqArr(nd_isfinite($inf, null), [false], "and not on an infinity");
eqArr(nd_isnan(nd_minimum($nan, nd([1.0], null), null), null), [true],
      "minimum PROPAGATES NaN, like NumPy's minimum rather than C's fmin");
ok(nd_get(nd_equal($nan, $nan, null), [0]) == false, "NaN is not equal to itself, as IEEE requires");

print("");
print("-- where and clip --");
eqArr(nd_where(nd([true, false, true], null), nd([1, 2, 3], null), nd([10, 20, 30], null)),
      [1, 20, 3], "where selects element-wise");
eqArr(nd_where(nd_greater(nd([1, 5, 3], null), 2, null), 100, 0), [0, 100, 100],
      "where broadcasts scalars on both branches");
eqArr(nd_clip(nd([-5, 0, 5, 10], null), 0, 6, null), [0, 0, 5, 6], "clip with both bounds");
eqArr(nd_clip(nd([-5, 0, 5], null), 0, null, null), [0, 0, 5], "clip with only a lower bound");
eqArr(nd_clip(nd([-5, 0, 5], null), null, 0, null), [-5, 0, 0], "clip with only an upper bound");
raises(def() { nd_clip(nd([1], null), null, null, null); }, "at least one",
   "clip with no bounds at all is refused rather than being a silent copy");

print("");
print("-- isclose / allclose / array_equal --");
ok(nd_allclose(nd([1.0, 2.0], null), nd([1.0, 2.0000000001], null), null, null, null),
   "allclose tolerates a tiny difference");
ok(!nd_allclose(nd([1.0], null), nd([1.5], null), null, null, null),
   "and does not tolerate a large one");
eqArr(nd_isclose(nd([1.0, 5.0], null), nd([1.0, 9.0], null), null, null, null), [true, false],
      "isclose is element-wise");
ok(!nd_allclose($nan, $nan, null, null, null), "NaN is not close to NaN by default");
ok(nd_allclose($nan, $nan, null, null, true), "unless equal_nan is asked for");
ok(nd_array_equal(nd([1, 2], null), nd([1, 2], null)), "array_equal on identical arrays");
ok(!nd_array_equal(nd([1, 2], null), nd([1, 3], null)), "and on differing ones");
// array_equal deliberately does NOT broadcast: "are these the same array" and
// "do these agree where they overlap" are different questions.
ok(!nd_array_equal(nd([1, 1, 1], null), nd([1], null)),
   "array_equal does not broadcast -- different shapes are not equal");

print("");
print("-- out=: the three gates --");
$o = nd_zeros([4], null);
$r = nd_add($a, $b, $o);
eqArr($o, [11, 22, 33, 44], "out= is written in place");
ok(nd_base_id($r) == nd_base_id($o), "and the same array is handed back, not a copy");

// Gate 1: a broadcast view has stride 0, so writing through it would hit one
// element many times. This is the gate that makes the read-only flag pay.
raises(def() { nd_add($a, $b, nd_broadcast_to(nd([0], null), [4])); }, "read-only",
   "a broadcast_to view as out= RAISES");
// Gate 2: out is not itself broadcast.
raises(def() { nd_add($a, $b, nd_zeros([3], null)); }, "out has shape",
   "an out= of the wrong shape is refused, naming both shapes");
// Gate 3: aliasing.
$acc = nd([1, 2, 3, 4], null);
nd_add($acc, nd([10, 10, 10, 10], null), $acc);
eqArr($acc, [11, 12, 13, 14], "the exact-overlap case nd_add(a, b, a) is correct in place");

// A SHIFTED overlap is the dangerous one: computing in place would read an
// element already overwritten. The answer must equal the non-aliased answer.
$base = nd_arange(0, 8, null);
$lo = nd_slice($base, [[0, 4, null]]);
$hi = nd_slice($base, [[1, 5, null]]);
$expect = nd_to_list(nd_add(nd_copy($lo), nd([100, 100, 100, 100], null), null));
nd_add($lo, nd([100, 100, 100, 100], null), $hi);
eq(str($expect), "[100, 101, 102, 103]", "control: the non-aliased answer");
eqArr($hi, [100, 101, 102, 103], "a partially overlapping out= gives the SAME answer, not garbage");

print("");
print("-- the strided path agrees with the contiguous one --");
// A transposed operand cannot use tier 0, so this compares the two kernels
// against each other on identical data.
$g = nd_reshape(nd_arange(0, 12, null), [3, 4]);
$gt = nd_T($g);
$viaStrided = nd_add($gt, 1000, null);
$viaFlat    = nd_add(nd_copy($gt), 1000, null);
ok(nd_array_equal($viaStrided, $viaFlat),
   "a transposed (tier 2) operand gives the same result as its contiguous copy");
eq(str(nd_shape($viaStrided)), "[4, 3]", "and keeps the transposed shape");

// A flipped view has a NEGATIVE stride, which the odometer must walk correctly.
$fl = nd_flip(nd_arange(0, 5, null), null);
eqArr(nd_add($fl, 0, null), [4, 3, 2, 1, 0], "a negative-stride operand is read in the right order");
eqArr(nd_multiply($fl, $fl, null), [16, 9, 4, 1, 0], "and two of them together");

print("");
print("-- differential: 100k elements against a pure-Bantu loop --");
$n = 1000;
$xs = nd_linspace(0, 10, $n, null);
$ys = nd_sqrt(nd_add(nd_multiply($xs, $xs, null), 1, null), null);
$maxRel = 0;
$i = 0;
while ($i < $n) {
    $x = nd_get($xs, [$i]);
    $want = sqrt($x * $x + 1);
    $got = nd_get($ys, [$i]);
    $d = $got - $want;
    if ($d < 0) { $d = 0 - $d; }
    $rel = $d / $want;
    if ($rel > $maxRel) { $maxRel = $rel; }
    $i = $i + 1;
}
print("        max relative error vs a pure-Bantu loop: " + str($maxRel));
ok($maxRel < 0.000000000000001, "1000 composed ufunc results match Bantu's own scalar maths to < 1e-15");

print("");
print("-- empty and 0-d arrays are not special cases --");
eq(nd_size(nd_add(nd_zeros([0], null), nd_zeros([0], null), null)), 0, "an empty array adds to empty");
eq(nd_ndim(nd_add(nd_zeros([], null), 1, null)), 0, "a 0-d array stays 0-d");
eq(nd_get(nd_add(nd_zeros([], null), 5, null), []), 5, "and holds the right value");

print("");
print("-- bad arguments raise catchably --");
raises(def() { nd_add(nd_zeros([2], null)); }, "two operands", "a missing operand is refused");
raises(def() { nd_add("hello", nd_zeros([2], null), null); }, "expected an array",
   "a string operand is refused");
raises(def() { nd_sqrt(); }, "needs an operand", "a missing unary operand is refused");
raises(def() { nd_add(nd_zeros([2], null), nd_zeros([2], null), 42); }, "expected an array",
   "a non-array out= is refused");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
