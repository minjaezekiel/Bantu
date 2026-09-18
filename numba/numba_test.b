// Quick smoke test for the numba package. The real gates live in tests/:
// numba_array_test.b, numba_ufunc_test.b, numba_reduce_test.b,
// numba_linalg_test.b and lang_native_ops_test.b -- CI only globs tests/*.b,
// so anything that must gate a release belongs there, not here.
include "./numba.b" as np;

$R = {"pass": 0, "fail": 0};
def ok($c, $n) {
    if ($c) { $R.pass = $R.pass + 1; print("  ok    " + $n); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $n); }
}

print("=== numba package smoke test ===");
ok(np.available(), "the ndarray native layer is present");

$a = np.array([[1, 2], [3, 4]], null);
ok(str(np.shape($a)) == "[2, 2]", "np.array builds a 2x2");
ok(np.get(np.sum($a, null, null), []) == 10, "np.sum");

// The façade's whole purpose: optional arguments the raw builtins cannot have.
ok(np.size(np.arange(0, 10, null)) == 10, "np.arange with the step defaulted");
ok(np.size(np.linspace(0, 1, null, null)) == 50, "np.linspace defaults to 50 points");
ok(np.size(np.random([4], null, null)) == 4, "np.random defaults its bounds");

$x = np.array([1.0, 2.0, 3.0, 4.0], "f64");
ok(np.get(np.mean($x, null, null), []) == 2.5, "np.mean");
ok(np.allclose(np.sqrt(np.array([4.0, 9.0], "f64"), null), np.array([2.0, 3.0], "f64"),
               null, null, null), "np.sqrt");

// Composed helpers.
$c = np.polyfit(np.array([1.0, 2.0, 3.0], "f64"), np.array([2.0, 4.0, 6.0], "f64"), 1);
ok(np.get($c, [0]) > 1.99 && np.get($c, [0]) < 2.01, "np.polyfit finds the slope");
ok(np.corrcoef(np.array([1.0, 2.0, 3.0], "f64"), np.array([2.0, 4.0, 6.0], "f64")) > 0.999,
   "np.corrcoef on perfectly correlated data");

// Operators and chaining.
ok(str(np.to_list($x + $x)) == "[2, 4, 6, 8]", "operators work on the façade's arrays");
ok(np.get($x.multiply($x).sum(), []) == 30, "methods chain");

print("");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
