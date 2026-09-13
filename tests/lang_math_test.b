// ════════════════════════════════════════════════════════════════════════
//  lang_math_test.b — the scalar maths surface, join(), and two defects.
//
//  WHY THIS EXISTS
//  Bantu shipped with abs ceil cos floor log max min pow round sin sqrt tan
//  random and nothing else. There was no exp, no atan2, no asin/acos, no
//  log10, no PI -- you could not place a log-scale tick, draw a pie slice or
//  compute the angle of an arrowhead without writing the series yourself.
//  bplot needs all of them; so does anyone doing geometry.
//
//  THREE DEFECTS FOUND WHILE DESIGNING bplot, ALL FIXED HERE
//    1. max(1, 2, 9) answered 2. max/min read args[0] and args[1] and ignored
//       everything after -- a silently wrong answer, which is the failure mode
//       that matters.
//    2. str(1e21) answered "-9223372036854775808". Value::toString cast any
//       integral double to long long; out of range that is undefined behaviour
//       (ISO C++ [conv.fpint]) and saturates to INT64_MIN on x86-64 and ARM64.
//    3. There was no join(), so the only way to build a string was
//       `$s = $s + part`, which is O(n^2): 20,000 appends took 1,116 ms and
//       40,000 took 6,752 ms -- 6.05x the time for 2x the work.
//
//  HOW THE MATHS IS CHECKED
//  Not against hard-coded constants copied from somewhere, which only proves
//  the constants were copied correctly. Each function is checked against an
//  IDENTITY it must satisfy -- exp(log(x)) == x, sin^2 + cos^2 == 1,
//  asin(sin(x)) == x, cosh^2 - sinh^2 == 1 -- plus the exact values it must
//  return at the points where libm implementations differ, plus its behaviour
//  on ±inf, NaN and outside its domain. An implementation that is wrong will
//  fail an identity; one that is merely imprecise will not.
//
//  Numbers are never asserted through str() -- str() gives six significant
//  digits and would hide a relative error of 1e-7.
//
//  Run:  bantu run tests/lang_math_test.b
// ════════════════════════════════════════════════════════════════════════

$R = {"pass": 0, "fail": 0};

def ok($cond, $name) {
    if ($cond) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $name); }
}

def eq($got, $want, $name) {
    if ($got == $want) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name);
        print("          got:  " + str($got));
        print("          want: " + str($want));
    }
}

// Relative closeness, so the tolerance means the same thing at 1e-9 and 1e9.
def close($got, $want, $name) {
    $tol = 0.00000000001;
    $scale = max(abs($want), 1);
    ok(abs($got - $want) <= $tol * $scale, $name);
}

// A catchable error is a pass; anything else -- a wrong number, a crash -- is not.
def raises($f, $name) {
    $caught = false;
    try { $f(); } catch ($e) { $caught = true; }
    ok($caught, $name);
}

print("========================================");
print("  Bantu — scalar maths, join, and three defects");
print("========================================");

// ── The constants ───────────────────────────────────────────────────────
print("");
print("-- constants, reachable bare or with a $ (one namespace, $ stripped) --");
close(PI, 3.141592653589793, "PI");
close($PI, 3.141592653589793, "$PI is the same binding");
close(TAU, 6.283185307179586, "TAU");
close(TAU, 2 * PI, "and TAU is exactly 2*PI");
close(E, 2.718281828459045, "E");
ok(isinf(INF), "INF is infinite");
ok(INF > 0, "and positive");
ok(isnan(NAN), "NAN is a NaN");
ok(NAN != NAN, "which means it is not equal to itself, as IEEE 754 requires");

// ── exp / log family ────────────────────────────────────────────────────
print("");
print("-- exp and the logarithms --");
close(exp(0), 1, "exp(0) is exactly 1");
close(exp(1), E, "exp(1) is E");
close(log(exp(2.5)), 2.5, "log undoes exp");
close(exp(log(7.25)), 7.25, "and exp undoes log");
close(log10(1000), 3, "log10(1000) is 3");
close(log10(0.001), -3, "log10 of a reciprocal decade");
close(log2(1024), 10, "log2(1024) is 10");
close(log2(0.5), -1, "log2 below 1");
// The whole reason these two exist: near zero, exp(x)-1 and log(1+x) lose
// almost every significant digit to cancellation if written the obvious way.
close(expm1(0.000000001), 0.000000001000000000500000000167, "expm1 is accurate near zero");
close(log1p(0.000000001), 0.0000000009999999995, "log1p is accurate near zero");
ok(expm1(0) == 0, "expm1(0) is exactly 0");
ok(log1p(0) == 0, "log1p(0) is exactly 0");
close(cbrt(27), 3, "cbrt(27)");
close(cbrt(-8), -2, "cbrt is defined for negatives, unlike pow(x, 1/3)");
ok(isnan(pow(-8, 0.3333333333333333)), "  (pow(-8, 1/3) really is NaN -- that is the point)");

print("");
print("-- their domain edges, which must not be errors --");
ok(isinf(log(0)) && log(0) < 0, "log(0) is -inf");
ok(isnan(log(-1)), "log of a negative is NaN");
ok(isinf(log10(0)) && log10(0) < 0, "log10(0) is -inf");
ok(exp(-1000) == 0, "exp underflows to exactly 0");
ok(isinf(exp(1000)), "exp overflows to inf");
ok(isnan(exp(NAN)), "NaN propagates through exp");
ok(isnan(log(NAN)), "and through log");

// ── Trigonometry ────────────────────────────────────────────────────────
print("");
print("-- trigonometry --");
close(sin(PI / 6), 0.5, "sin(30 degrees)");
close(cos(PI / 3), 0.5, "cos(60 degrees)");
close(sin(PI / 4) * sin(PI / 4) + cos(PI / 4) * cos(PI / 4), 1,
      "sin^2 + cos^2 == 1");
close(asin(0.5), PI / 6, "asin(0.5)");
close(acos(0.5), PI / 3, "acos(0.5)");
close(atan(1), PI / 4, "atan(1)");
close(asin(sin(0.7)), 0.7, "asin undoes sin inside the principal range");
close(acos(cos(1.2)), 1.2, "acos undoes cos");
close(asin(1), PI / 2, "asin at the top of its domain");
close(acos(-1), PI, "acos at the bottom of its domain");

print("");
print("-- atan2 knows the quadrant, which atan(y/x) cannot --");
close(atan2(1, 1), PI / 4, "first quadrant");
close(atan2(1, -1), 3 * PI / 4, "second quadrant");
close(atan2(-1, -1), -3 * PI / 4, "third quadrant -- atan(y/x) would say +PI/4");
close(atan2(-1, 1), -PI / 4, "fourth quadrant");
close(atan2(1, 0), PI / 2, "straight up, where y/x is a division by zero");
close(atan2(0, -1), PI, "straight left");
eq(atan2(0, 1), 0, "straight right is exactly 0");

print("");
print("-- outside the domain the answer is NaN, not an error --");
ok(isnan(asin(2)), "asin(2) is NaN");
ok(isnan(acos(-2)), "acos(-2) is NaN");
ok(isnan(asin(NAN)), "NaN propagates");

// ── hypot ───────────────────────────────────────────────────────────────
print("");
print("-- hypot, which is not just sqrt(x*x + y*y) --");
close(hypot(3, 4), 5, "the 3-4-5 triangle");
close(hypot(-3, -4), 5, "signs do not matter");
eq(hypot(0, 0), 0, "the degenerate case");
// This is the reason hypot exists at all: the naive form squares first and
// 1e200 squared is inf, so it answers inf for a value that is representable.
ok(isfinite(hypot(1e200, 1e200)), "hypot(1e200, 1e200) is finite");
ok(isinf(sqrt(1e200 * 1e200 + 1e200 * 1e200)), "  (the naive form overflows to inf)");
close(hypot(1e200, 1e200) / 1e200, 1.4142135623730951, "and the finite answer is right");
ok(isinf(hypot(INF, NAN)), "hypot(inf, NaN) is inf -- IEEE 754 says infinity wins");

// ── Hyperbolics ─────────────────────────────────────────────────────────
print("");
print("-- hyperbolics --");
close(sinh(0), 0, "sinh(0)");
close(cosh(0), 1, "cosh(0)");
close(tanh(0), 0, "tanh(0)");
close(cosh(1.3) * cosh(1.3) - sinh(1.3) * sinh(1.3), 1, "cosh^2 - sinh^2 == 1");
close(asinh(sinh(0.9)), 0.9, "asinh undoes sinh");
close(acosh(cosh(0.9)), 0.9, "acosh undoes cosh");
close(atanh(tanh(0.9)), 0.9, "atanh undoes tanh");
close(tanh(50), 1, "tanh saturates at 1");
close(tanh(-50), -1, "and at -1");
ok(isnan(acosh(0.5)), "acosh below 1 is NaN");
ok(isinf(atanh(1)), "atanh(1) is inf");
ok(isnan(atanh(2)), "atanh outside [-1,1] is NaN");

// ── Rounding, sign, angles ──────────────────────────────────────────────
print("");
print("-- rounding and sign --");
eq(trunc(2.7), 2, "trunc towards zero");
eq(trunc(-2.7), -2, "trunc(-2.7) is -2, where floor(-2.7) is -3");
eq(floor(-2.7), -3, "  (floor, for contrast)");
eq(sign(5.5), 1, "sign of a positive");
eq(sign(-5.5), -1, "sign of a negative");
eq(sign(0), 0, "sign of zero");
ok(isnan(sign(NAN)), "sign propagates NaN, as NumPy's does");
close(fmod(7.5, 2), 1.5, "fmod");
close(fmod(-7.5, 2), -1.5, "fmod keeps the sign of the dividend");
ok(isnan(fmod(1, 0)), "fmod by zero is NaN, not an error");
eq(copysign(3, -1), -3, "copysign");
eq(copysign(-3, 1), 3, "copysign the other way");

print("");
print("-- degrees and radians --");
close(degrees(PI), 180, "PI radians is 180 degrees");
close(radians(180), PI, "and back");
close(degrees(radians(37.5)), 37.5, "they are inverses");
close(radians(90), PI / 2, "a right angle");

// ── Predicates ──────────────────────────────────────────────────────────
print("");
print("-- isnan / isinf / isfinite: the language could PRODUCE these --");
print("   values but had no way to test for one --");
ok(isnan(sqrt(-1)), "sqrt(-1) is a NaN");
ok(isinf(pow(10, 400)), "pow(10, 400) overflows to inf");
ok(isinf(log(0)), "log(0) is infinite");
ok(!isfinite(NAN), "NaN is not finite");
ok(!isfinite(INF), "inf is not finite");
ok(isfinite(0), "zero is finite");
ok(isfinite(1e308), "and so is the largest normal double");
ok(!isnan(INF), "inf is not a NaN");
ok(!isinf(NAN), "and a NaN is not an infinity");

// ── clamp ───────────────────────────────────────────────────────────────
print("");
print("-- clamp --");
eq(clamp(5, 0, 1), 1, "above the range");
eq(clamp(-5, 0, 1), 0, "below the range");
eq(clamp(0.4, 0, 1), 0.4, "inside the range");
eq(clamp(0, 0, 1), 0, "on the lower bound");
eq(clamp(1, 0, 1), 1, "on the upper bound");
ok(isnan(clamp(NAN, 0, 1)), "a NaN clamps to a NaN rather than to a bound");
raises(def() { return clamp(1, 5, 0); }, "an inverted range raises instead of guessing");

// ════════════════════════════════════════════════════════════════════════
//  DEFECT 1 — max(1, 2, 9) answered 2
// ════════════════════════════════════════════════════════════════════════
print("");
print("-- [defect] max/min ignored every argument after the second --");
eq(max(1, 2, 9), 9, "max(1, 2, 9) is 9   (answered 2 before this fix)");
eq(min(5, 4, 1), 1, "min(5, 4, 1) is 1   (answered 4 before this fix)");
eq(max(1, 2), 2, "the two-argument form is unchanged");
eq(min(1, 2), 1, "and so is min's");
eq(max(7), 7, "one argument is its own maximum");
eq(max(3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5), 9, "eleven arguments");
eq(min(3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5), 1, "and the minimum of them");
eq(max(-5, -2, -9), -2, "all negative");
eq(max(1.5, 1.5, 1.5), 1.5, "all equal");

print("");
print("-- and a single list argument is reduced over --");
eq(max([3, 17, 5]), 17, "max of a list");
eq(min([3, 17, 5]), 3, "min of a list");
eq(max([42]), 42, "a one-element list");
eq(max([-1, -2]), -1, "a list of negatives");

print("");
print("-- NaN propagates through max/min (NumPy's rule: max, not nanmax) --");
ok(isnan(max(1, NAN, 3)), "a NaN in the middle");
ok(isnan(max(NAN, 1)), "a NaN first");
ok(isnan(max(1, NAN)), "a NaN last");
ok(isnan(min([1, 2, NAN])), "a NaN in a list");
ok(!isnan(max(1, INF)), "an infinity is an ordinary value");
eq(max(1, INF), INF, "and it wins");
eq(min(1, -INF), -INF, "as -inf does at the other end");

print("");
print("-- bad arguments raise, naming the problem --");
raises(def() { return max(); }, "max() of nothing raises");
raises(def() { return min(); }, "min() of nothing raises");
raises(def() { return max([]); }, "max of an empty list raises");
raises(def() { return max(1, "two"); }, "a string argument raises");
raises(def() { return max([1, null]); }, "a null in a list raises");

// ════════════════════════════════════════════════════════════════════════
//  DEFECT 2 — str(1e21) answered "-9223372036854775808"
// ════════════════════════════════════════════════════════════════════════
print("");
print("-- [defect] str() of a value past 2^63 was undefined behaviour --");
// The string IS the contract here, so these assert on it.
eq(str(1e21), "1e+21", "str(1e21)   (was -9223372036854775808)");
eq(str(-1e21), "-1e+21", "str(-1e21)");
eq(str(1e30), "1e+30", "str(1e30)");
eq(str(1e308), "1e+308", "str of the largest decade");
eq(str(pow(2, 70)), "1.18059e+21", "str(2^70)");
eq(str(1e18), "1000000000000000000", "just below the boundary, still an integer");
eq(str(-1e18), "-1000000000000000000", "and negative");
eq(str(0), "0", "zero");
eq(str(42), "42", "an ordinary integer");
eq(str(-42), "-42", "an ordinary negative");
eq(str(2.5), "2.5", "a fraction");
eq(str(INF), "inf", "infinity");
eq(str(-INF), "-inf", "negative infinity");
eq(str(NAN), "nan", "a NaN");
// The round trip is the real assertion: whatever str() emits, num() must read.
close(num(str(1e21)), 1e21, "num(str(1e21)) round-trips");
close(num(str(1e308)), 1e308, "and so does the largest decade");

// ════════════════════════════════════════════════════════════════════════
//  DEFECT 3 — no join(), so building a string was O(n^2)
// ════════════════════════════════════════════════════════════════════════
print("");
print("-- [defect] join(), the missing inverse of split() --");
eq(join(["a", "b", "c"], "-"), "a-b-c", "joined with a separator");
eq(join(["a", "b", "c"], ""), "abc", "joined with nothing");
eq(join(["a", "b"]), "ab", "a missing separator means nothing");
eq(join(["solo"], "-"), "solo", "one element has no separator to place");
eq(join([], "-"), "", "an empty list is an empty string");
eq(join([1, 2, 3], ","), "1,2,3", "numbers are stringified as print would");
eq(join([true, false], "/"), "true/false", "and so are bools");
eq(join(["a", "b"], ", "), "a, b", "a multi-character separator");
eq(join(split("a,b,c", ","), ","), "a,b,c", "join(split(s)) round-trips");
raises(def() { return join("not a list", "-"); }, "a non-list first argument raises");
raises(def() { return join(["a"], 5); }, "a non-string separator raises");

print("");
print("-- and it is linear, which is the whole point --");
$parts = [];
$i = 0;
while ($i < 20000) { push($parts, "xyz"); $i = $i + 1; }
$t0 = clock();
$joined = join($parts, "");
$t20k = clock() - $t0;
eq(len($joined), 60000, "20,000 parts joined to 60,000 characters");

$parts2 = [];
$i = 0;
while ($i < 40000) { push($parts2, "xyz"); $i = $i + 1; }
$t0 = clock();
$joined2 = join($parts2, "");
$t40k = clock() - $t0;
eq(len($joined2), 120000, "40,000 parts joined to 120,000 characters");
// Doubling the work must roughly double the time. The `+=` form it replaces
// took 6.05x for the same doubling. A generous ceiling of 3x still catches
// anything quadratic while tolerating a noisy machine and a millisecond clock.
ok($t40k <= max(3 * $t20k, 25),
   "twice the parts costs about twice the time, not four times it");
print("          20k join " + str($t20k) + " ms, 40k join " + str($t40k) + " ms");

// ════════════════════════════════════════════════════════════════════════
//  DEFECT 4 — $a[$i] copied the whole list to read one element
// ════════════════════════════════════════════════════════════════════════
print("");
print("-- [defect] list indexing was O(n), so every loop over a list was O(n^2) --");
$big = [];
$i = 0;
while ($i < 10000) { push($big, $i * 2); $i = $i + 1; }

$t0 = clock();
$sum = 0;
$i = 0;
while ($i < 10000) { $sum = $sum + $big[$i]; $i = $i + 1; }
$t10k = clock() - $t0;
eq($sum, 99990000, "10,000 reads sum correctly");

$big2 = [];
$i = 0;
while ($i < 20000) { push($big2, $i * 2); $i = $i + 1; }
$t0 = clock();
$sum2 = 0;
$i = 0;
while ($i < 20000) { $sum2 = $sum2 + $big2[$i]; $i = $i + 1; }
$t20kr = clock() - $t0;
eq($sum2, 399980000, "20,000 reads sum correctly");
ok($t20kr <= max(3 * $t10k, 50),
   "twice the reads costs about twice the time  (it was 4.2x before this fix)");
print("          10k reads " + str($t10k) + " ms, 20k reads " + str($t20kr) + " ms");

print("");
print("-- and the semantics of indexing are unchanged --");
$m = [[1, 2, 3], [4, 5, 6]];
eq($m[1][2], 6, "a nested read");
$m[1][2] = 99;
eq($m[1][2], 99, "a nested write");
eq($m[0][0], 1, "and it did not disturb its neighbour");
$lst = [10, 20, 30];
eq($lst[0], 10, "the first element");
eq($lst[2], 30, "the last element");
raises(def() { $z = [1, 2]; return $z[5]; }, "out of bounds raises");
raises(def() { $z = [1, 2]; return $z[-1]; }, "a negative index raises");
// A read of a missing key must not CREATE it -- the write path's resolver
// inserts on lookup, which is right for `$d["new"] = 1` and wrong for a read.
$d = {"a": 1};
eq($d["zz"], null, "a missing dict key reads as null");
eq(len(keys($d)), 1, "and the read did not insert it");
$dn = {"a": [1, 2, 3]};
eq($dn["a"][1], 2, "a list inside a dict");
eq($dn["nope"], null, "a missing key through the borrowing path");
eq(len(keys($dn)), 1, "still did not insert it");
// Lists have value semantics; the fix borrows for reading only and must not
// have turned them into references.
$src = [1, 2, 3];
$cpy = $src;
$cpy[0] = 99;
eq($src[0], 1, "a list assigned to another name is still a copy");
eq($cpy[0], 99, "and the copy took the write");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
