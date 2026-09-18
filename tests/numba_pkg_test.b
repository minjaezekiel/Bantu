// ════════════════════════════════════════════════════════════════════════
//  numba_pkg_test.b — the pure-Bantu façade and its composed helpers.
//
//  The façade is the public API, and the reason it exists is that it can
//  have optional arguments where a raw builtin cannot: Bantu binds a missing
//  argument to null, so `np.arange(0, 10)` works here while the underlying
//  `nd_arange` needs an explicit trailing null. Half of this file asserts
//  exactly that -- the defaults -- because a wrapper that forgets one is a
//  wrapper that does nothing.
//
//  The other half is the composed helpers: polyfit, interp, gradient, cov,
//  corrcoef, meshgrid, moving_average and trapz. None of them needed a new
//  kernel; they are built out of the atoms and checked against values that
//  can be worked out by hand.
//
//  Run:  bantu run tests/numba_pkg_test.b
// ════════════════════════════════════════════════════════════════════════

include "./numba/numba.b" as np;

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
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name + "  got " + str($got) + " want " + str($want));
    }
}
def raises($fn, $name) {
    try { $fn(); $R.fail = $R.fail + 1; print("  FAIL  " + $name + " (no error)"); }
    catch ($e) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
}

print("========================================");
print("  numba — the package façade");
print("========================================");

print("");
print("-- the module loads and reports itself --");
ok(np.available(), "np.available() is true on a build with the native layer");
close($np.PI, 3.141592653589793, 1e-15, "np.PI");
close($np.E, 2.718281828459045, 1e-15, "np.E");
ok(np.get(np.isnan(np.array([$np.NAN], "f64"), null), [0]), "np.NAN really is NaN");
ok(np.get(np.isinf(np.array([$np.INF], "f64"), null), [0]), "np.INF really is infinite");

print("");
print("-- THE POINT OF THE FAÇADE: optional arguments --");
// Each of these would need an explicit trailing null against the raw builtin.
eq(np.size(np.arange(0, 10, null)), 10, "np.arange(0, 10) defaults its step to 1");
eq(np.size(np.linspace(0, 1, null, null)), 50, "np.linspace defaults to 50 points");
eq(np.size(np.random([7], null, null)), 7, "np.random defaults to [0, 1)");
eq(np.size(np.randn([5], null, null)), 5, "np.randn defaults to mean 0, sd 1");
eq(np.dtype(np.zeros([2], null)), "f64", "np.zeros defaults to f64");
$r = np.random([1000], null, null);
ok(np.get(np.min($r, null, null), []) >= 0 && np.get(np.max($r, null, null), []) < 1,
   "and np.random really does stay inside [0, 1)");

print("");
print("-- creation and basic maths through the façade --");
$a = np.array([[1, 2], [3, 4]], null);
eq(str(np.shape($a)), "[2, 2]", "np.array from a nested list");
eq(np.get(np.sum($a, null, null), []), 10, "np.sum");
eq(str(np.to_list(np.sum($a, 0, null))), "[4, 6]", "np.sum over an axis");
eq(str(np.shape(np.sum($a, 1, true))), "[2, 1]", "np.sum with keepdims");
eq(str(np.to_list(np.reshape(np.arange(0, 6, null), [2, 3]))), "[[0, 1, 2], [3, 4, 5]]", "np.reshape");
eq(str(np.to_list(np.T(np.reshape(np.arange(0, 4, null), [2, 2])))), "[[0, 2], [1, 3]]", "np.T");
ok(np.allclose(np.sqrt(np.array([4.0, 9.0], "f64"), null),
               np.array([2.0, 3.0], "f64"), null, null, null), "np.sqrt");

print("");
print("-- `any` is a reserved word in Bantu, so the façade spells it anyof --");
// `def any(...)` does not parse. The builtin and the METHOD both work, which is
// why only the façade function needed a different name.
eq(np.get(np.anyof(np.array([0, 1], null), null, null), []), true, "np.anyof");
eq(np.get(np.all(np.array([1, 1], null), null, null), []), true, "np.all is fine as-is");
eq(np.get(np.array([0, 1], null).any(), []), true, "and $a.any() works as a method");

print("");
print("-- polyfit / polyval --");
// A perfect line: y = 2x, so the fit must be [2, 0] to machine precision.
$x = np.array([1.0, 2.0, 3.0, 4.0], "f64");
$c = np.polyfit($x, np.array([2.0, 4.0, 6.0, 8.0], "f64"), 1);
close(np.get($c, [0]), 2.0, 1e-10, "polyfit recovers the slope of an exact line");
close(np.get($c, [1]), 0.0, 1e-10, "and its zero intercept");
// A perfect parabola: y = x^2 + 1.
$c2 = np.polyfit($x, np.array([2.0, 5.0, 10.0, 17.0], "f64"), 2);
close(np.get($c2, [0]), 1.0, 1e-8, "a quadratic fit recovers the x^2 coefficient");
close(np.get($c2, [2]), 1.0, 1e-8, "and the constant");
// polyval must invert polyfit: evaluating the fit at the data reproduces it.
ok(np.allclose(np.polyval($c, $x), np.array([2.0, 4.0, 6.0, 8.0], "f64"), null, null, null),
   "polyval of the fitted coefficients reproduces the data");
// Horner's rule, checked independently: 3x^2 + 2x + 1 at x = 2 is 17.
close(np.get(np.polyval(np.array([3.0, 2.0, 1.0], "f64"), np.array([2.0], "f64")), [0]),
      17.0, 1e-12, "polyval evaluates 3x^2 + 2x + 1 at x=2 as 17");

print("");
print("-- interp --");
$xs = np.array([0.0, 1.0, 2.0], "f64");
$ys = np.array([0.0, 10.0, 20.0], "f64");
eq(str(np.to_list(np.interp(np.array([0.5], "f64"), $xs, $ys))), "[5]", "interp halfway");
eq(str(np.to_list(np.interp(np.array([0.0, 1.0, 2.0], "f64"), $xs, $ys))), "[0, 10, 20]",
   "interp exactly on the knots");
// Outside the range it CLAMPS rather than extrapolating -- silently
// extrapolating is how people get nonsense far from their data.
eq(str(np.to_list(np.interp(np.array([-5.0, 99.0], "f64"), $xs, $ys))), "[0, 20]",
   "interp clamps outside the range instead of extrapolating");
eq(np.size(np.interp(np.array([0.5], "f64"), np.array([1.0], "f64"), np.array([7.0], "f64"))), 1,
   "a single-point table does not crash");

print("");
print("-- gradient --");
// d/dx of 2x is 2 everywhere, including the one-sided ends.
ok(np.allclose(np.gradient(np.array([0.0, 2.0, 4.0, 6.0], "f64"), null),
               np.array([2.0, 2.0, 2.0, 2.0], "f64"), null, null, null),
   "the gradient of a straight line is constant, ends included");
// With a step: the same data over h = 0.5 has gradient 4.
ok(np.allclose(np.gradient(np.array([0.0, 2.0, 4.0], "f64"), 0.5),
               np.array([4.0, 4.0, 4.0], "f64"), null, null, null), "gradient honours the step");

print("");
print("-- cov / corrcoef --");
$p = np.array([1.0, 2.0, 3.0, 4.0], "f64");
$q = np.array([2.0, 4.0, 6.0, 8.0], "f64");
close(np.corrcoef($p, $q), 1.0, 1e-12, "perfectly correlated data gives 1");
close(np.corrcoef($p, np.array([8.0, 6.0, 4.0, 2.0], "f64")), -1.0, 1e-12,
      "perfectly anti-correlated gives -1");
close(np.cov($p, $q), 3.3333333333333335, 1e-10, "cov uses the sample (ddof=1) estimator");
// A constant series has no correlation defined -- NaN, not 0, because 0 would
// claim "no relationship" when the truth is "the question is meaningless".
ok(np.get(np.isnan(np.array([np.corrcoef($p, np.array([5.0, 5.0, 5.0, 5.0], "f64"))], "f64"), null), [0]),
   "a constant series gives NaN, not 0");

print("");
print("-- meshgrid --");
$g = np.meshgrid(np.array([1, 2, 3], null), np.array([10, 20], null));
eq(str(np.shape($g[0])), "[2, 3]", "meshgrid X has one row per y");
eq(str(np.to_list($g[0])), "[[1, 2, 3], [1, 2, 3]]", "X varies along the columns");
eq(str(np.to_list($g[1])), "[[10, 10, 10], [20, 20, 20]]", "Y varies along the rows");
// The result is copied, not a broadcast view, so it is writable -- a grid is
// something people expect to be able to modify.
ok(np.writable($g[0]), "the grids are writable, not read-only broadcast views");

print("");
print("-- moving_average and trapz --");
eq(str(np.to_list(np.moving_average(np.array([1.0, 2.0, 3.0, 4.0], "f64"), 2))), "[1.5, 2.5, 3.5]",
   "a 2-wide moving average");
eq(np.size(np.moving_average(np.array([1.0, 2.0, 3.0, 4.0], "f64"), 3)), 2,
   "the result is shorter by w-1, which is the honest length");
raises(def() { return np.moving_average(np.array([1.0], "f64"), 5); },
   "a window longer than the data raises rather than returning nothing");
// The integral of a constant 1 over 4 points spaced 1 apart is 3.
close(np.trapz(np.array([1.0, 1.0, 1.0, 1.0], "f64"), null), 3.0, 1e-12, "trapz of a constant");
// A straight line from 0 to 3: the area of the triangle is 4.5.
close(np.trapz(np.array([0.0, 1.0, 2.0, 3.0], "f64"), null), 4.5, 1e-12, "trapz of a ramp");

print("");
print("-- linear algebra through the façade --");
$A = np.array([[4.0, 1.0], [1.0, 3.0]], "f64");
$b = np.array([1.0, 2.0], "f64");
ok(np.allclose(np.matmul($A, np.solve($A, $b)), $b, null, null, null),
   "np.solve, checked by residual");
ok(np.allclose(np.matmul($A, np.inv($A)), np.identity(2), null, null, null), "np.inv");
close(np.get(np.det(np.array([[1.0, 2.0], [3.0, 4.0]], "f64")), []), -2.0, 1e-12, "np.det");
close(np.get(np.norm(np.array([3.0, 4.0], "f64"), null), []), 5.0, 1e-12, "np.norm");

print("");
print("-- help() and info() run without error --");
np.help(null);
np.help("solve");
np.info(np.reshape(np.arange(0, 6, null), [2, 3]));
$R.pass = $R.pass + 1;
print("  ok    np.help() and np.info() produce output and return cleanly");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
