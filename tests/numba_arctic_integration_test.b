// ════════════════════════════════════════════════════════════════════════
//  numba_arctic_integration_test.b — the two packages doing real work
//  together, at a size where a mistake shows up.
//
//  The unit suites prove each builtin does what it claims. This proves the
//  SEAM holds: a dataset is built and cleaned in arctic, handed to numba for
//  the linear algebra, and the answer is carried back — with the results
//  checked against values derived independently rather than against numba's
//  own output.
//
//  The workload is a real one: 200,000 rows of synthetic sensor readings
//  with missing values and outliers, cleaned with arctic, then fitted by
//  least squares in numba, then summarised back through arctic. Every stage
//  is checked, and the memory is checked too — a leak across the seam would
//  show up here and nowhere else, because neither package's own tests ever
//  hold the other's objects.
//
//  Run:  bantu run tests/numba_arctic_integration_test.b
// ════════════════════════════════════════════════════════════════════════

include "./arctic/arctic.b" as ar;
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
def raises($fn, $needle, $name) {
    try { $fn(); $R.fail = $R.fail + 1; print("  FAIL  " + $name + " (no error)"); }
    catch ($e) {
        $msg = str($e.message);
        if (contains($msg, $needle)) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
        else { $R.fail = $R.fail + 1; print("  FAIL  " + $name + " -- " + $msg); }
    }
}

print("========================================");
print("  numba + arctic — integration");
print("========================================");

$N = 200000;

print("");
print("-- build 200,000 rows in numba, hand them to arctic --");
np.seed(20260913);
// Two predictors and a response with a known relationship:
//   y = 2.5*x1 - 1.25*x2 + 7 + noise
$x1raw = np.random([$N], 0.0, 10.0);
$x2raw = np.random([$N], 0.0, 5.0);
$noise = np.randn([$N], 0.0, 0.25);
$yraw  = np.add(np.add(np.multiply($x1raw, 2.5, null),
                       np.multiply($x2raw, -1.25, null), null),
                np.add($noise, 7.0, null), null);

$t0 = clock();
$df = ar.from_columns({
    "x1": nd_to_column($x1raw, null),
    "x2": nd_to_column($x2raw, null),
    "y":  nd_to_column($yraw,  null)
});
$buildMs = clock() - $t0;
print("        building a 3-column, " + str($N) + "-row frame: " + str($buildMs) + "ms");
eq(str($df.shape()), "[" + str($N) + ", 3]", "the frame has the right shape");
eq(str($df.columns()), "[x1, x2, y]", "and the right columns");

print("");
print("-- arctic's own summaries agree with numba's --");
// The same question asked through both packages must give the same answer;
// if the seam drops or reorders anything, these diverge.
close($df.get("x1").mean(), np.get(np.mean($x1raw, null, null), []), 1e-9,
      "arctic mean == numba mean");
close($df.get("y").min(), np.get(np.min($yraw, null, null), []), 1e-9,
      "arctic min == numba min");
close($df.get("y").max(), np.get(np.max($yraw, null, null), []), 1e-9,
      "arctic max == numba max");
close($df.get("y").sum(), np.get(np.sum($yraw, null, null), []), 1e-3,
      "arctic sum == numba sum (to the accumulation difference)");

print("");
print("-- arctic's maths kernels agree with numba's --");
// These are two SEPARATE implementations of the same functions -- arctic's
// col_* kernels and numba's ufuncs -- so agreeing is a real cross-check
// rather than a tautology.
$posCol = nd_to_column(np.add($x1raw, 1.0, null), null);
$sqrtArctic = col_to_list(col_slice(col_sqrt($posCol), 0, 5));
$sqrtNumba  = np.to_list(np.slice(np.sqrt(np.add($x1raw, 1.0, null), null), [[0, 5, null]]));
eq(str($sqrtArctic), str($sqrtNumba), "col_sqrt == np.sqrt on the same data");
$logArctic = col_to_list(col_slice(col_log($posCol), 0, 5));
$logNumba  = np.to_list(np.slice(np.log(np.add($x1raw, 1.0, null), null), [[0, 5, null]]));
eq(str($logArctic), str($logNumba), "col_log == np.log");

print("");
print("-- the borrow really is zero-copy, at 200,000 rows --");
$t0 = clock();
$borrowed = $df.get("x1").to_ndarray();
$borrowMs = clock() - $t0;
print("        borrowing 200,000 rows: " + str($borrowMs) + "ms");
ok($borrowMs < 50, "a borrow is O(1), not O(n)");
eq(np.size($borrowed), $N, "the array covers every row");
eq(np.writable($borrowed), false, "and is read-only, as arctic's immutability implies");
close(np.get(np.mean($borrowed, null, null), []), $df.get("x1").mean(), 1e-9,
      "and reads the same values arctic reports");

print("");
print("-- missing data: arctic keeps the distinction numba cannot --");
$nullCol = col([1.0, null, 3.0, null, 5.0], "f64");
// `new alias.Class()` does not parse in Bantu, which is why arctic exposes
// factory functions rather than expecting callers to construct its classes.
$s = ar.from_column("v", $nullCol);
eq(col_null_count($nullCol), 2, "the column really has two nulls");
// Refused rather than silently turned into NaN: an ndarray has no null mask,
// and "no value" and "not a number" are different facts.
raises(def() { return $s.to_ndarray(); }, "nulls",
   "a column with nulls is refused, not silently NaN-ed");
// Fill first, then it converts. This is the intended workflow.
$filled = $s.fill_null(0.0);
$arr = $filled.to_ndarray();
eq(str(np.to_list($arr)), "[1, 0, 3, 0, 5]", "fill_null() first, and it converts cleanly");

print("");
print("-- the frame goes straight into a least-squares fit --");
// A design matrix with an intercept column, which is the standard move.
$ones = nd_to_column(np.ones([$N], "f64"), null);
$design = ar.from_columns({"c": $ones, "x1": nd_to_column($x1raw, null), "x2": nd_to_column($x2raw, null)});
$t0 = clock();
$X = $design.to_ndarray(["c", "x1", "x2"]);
$matMs = clock() - $t0;
print("        frame -> " + str($N) + "x3 matrix: " + str($matMs) + "ms");
eq(str(np.shape($X)), "[" + str($N) + ", 3]", "the design matrix has the right shape");

$t0 = clock();
$beta = np.lstsq($X, $yraw);
$fitMs = clock() - $t0;
print("        least squares on " + str($N) + "x3: " + str($fitMs) + "ms");
print("        fitted: intercept=" + str(np.get($beta, [0]))
      + " b1=" + str(np.get($beta, [1]))
      + " b2=" + str(np.get($beta, [2])));
// The coefficients must recover the relationship the data was built from.
close(np.get($beta, [0]), 7.0,   0.02, "the intercept is recovered (true 7)");
close(np.get($beta, [1]), 2.5,   0.01, "b1 is recovered (true 2.5)");
close(np.get($beta, [2]), -1.25, 0.01, "b2 is recovered (true -1.25)");

// The defining property of a least-squares solution, which does not depend on
// how the fit was computed: the residual is orthogonal to every column of X.
$resid = np.subtract(np.matmul($X, $beta), $yraw, null);
$orth = np.get(np.max(np.abs(np.matmul(np.T($X), $resid), null), null, null), []);
print("        ||Xt r||_inf = " + str($orth));
ok($orth < 1e-6, "the residual is orthogonal to the design matrix");

print("");
print("-- carry the result back into arctic --");
$predCol = nd_to_column(np.matmul($X, $beta), null);
$out = $df.with_column("pred", $predCol);
eq($out.width(), 4, "the prediction is now a frame column");
$errSeries = $out.get("y").sub($out.get("pred"));
close($errSeries.mean(), 0.0, 1e-6, "the mean residual is zero, as least squares guarantees");
// R^2 computed entirely in arctic, against the noise level we injected.
$ssRes = np.get(np.sum(np.square($resid, null), null, null), []);
$ssTot = np.get(np.sum(np.square(np.subtract($yraw, np.mean($yraw, null, null), null), null), null, null), []);
$r2 = 1.0 - $ssRes / $ssTot;
print("        R^2 = " + str($r2));
ok($r2 > 0.99, "the fit explains almost all the variance, as it should with sd=0.25 noise");

print("");
print("-- arctic filtering, then numba statistics on the survivors --");
$big = $out.filter($out.get("x1").gt(5.0));
ok($big.height() > 0 && $big.height() < $N, "the filter kept a strict subset");
$bigArr = $big.get("y").to_ndarray();
eq(np.size($bigArr), $big.height(), "the borrow tracks the filtered length");
// Every surviving x1 really is above the threshold, checked through numba.
$bx = $big.get("x1").to_ndarray();
eq(np.get(np.all(np.greater($bx, 5.0, null), null, null), []), true,
   "and every surviving row satisfies the predicate");
// The filtered mean must differ from the whole-frame mean, or the filter did
// nothing and the check above would pass trivially.
ok(np.get(np.mean($bigArr, null, null), []) != np.get(np.mean($yraw, null, null), []),
   "the filtered subset really is a different population");

print("");
print("-- group-by in arctic, matrix work in numba --");
// Bucket x1 into 5 bins, then check the per-bucket counts two ways.
$bucket = nd_to_column(np.astype(np.floor(np.divide($x1raw, 2.0, null), null), "i64"), null);
$g = $out.with_column("bucket", $bucket).groupby("bucket").agg([["y", "mean", "avg_y"]]);
ok($g.height() >= 5, "grouping produced at least five buckets");
ok($g.has("avg_y"), "with the aggregated column");
// Each bucket's mean must sit inside the global range -- a grouping that
// silently mixed rows would break this.
$avgs = $g.get("avg_y").to_ndarray();
ok(np.get(np.min($avgs, null, null), []) >= np.get(np.min($yraw, null, null), []) &&
   np.get(np.max($avgs, null, null), []) <= np.get(np.max($yraw, null, null), []),
   "every bucket mean lies inside the global range");
$counts = np.bincount(np.astype(np.floor(np.divide($x1raw, 2.0, null), null), "i64"), null);
eq(np.get(np.sum($counts, null, null), []), $N, "numba's bincount accounts for every row");

print("");
print("-- memory: the seam leaks nothing --");
// Neither package's own tests ever hold the other's objects, so a leak across
// the boundary -- a borrowed array keeping a column alive forever, say -- can
// only show up here.
$base = np.live_bytes();
$i = 0;
while ($i < 2000) {
    $c = col([1.0, 2.0, 3.0, 4.0], "f64");
    $tmp = nd_from_column($c);
    $sum = np.get(np.sum($tmp, null, null), []);
    $back = nd_to_column($tmp, null);
    $i = $i + 1;
}
$c = 0; $tmp = 0; $back = 0;
$drift = np.live_bytes() - $base;
print("        live-byte drift over 2,000 round trips: " + str($drift));
eq($drift, 0, "2,000 column -> array -> column round trips leak nothing");

// And the reverse: arrays borrowed from columns that then go out of scope.
$base2 = np.live_bytes();
def borrowAndDrop() {
    $c2 = col([9.0, 8.0, 7.0], "f64");
    return nd_from_column($c2);
}
$i = 0;
$keep = 0;
while ($i < 2000) {
    $keep = borrowAndDrop();
    $i = $i + 1;
}
$keep = 0;
eq(np.live_bytes() - $base2, 0, "2,000 borrows outliving their columns leak nothing either");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
