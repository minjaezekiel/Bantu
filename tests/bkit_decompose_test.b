// ════════════════════════════════════════════════════════════════════════
//  bkit_decompose_test.b — PCA and TruncatedSVD.
//
//  The two assertions that carry the most:
//
//   1. **The components agree with numba's SVD to 1e-10**, computed here from
//      `np.svd` on the centred matrix rather than by calling the estimator a
//      second way. A reference that goes through the code it is checking
//      shares the code's errors ([K48]).
//   2. **The sign convention is checked through the PUBLIC surface.** `U` is
//      not exposed, but `transform` returns `U·S` and every singular value is
//      non-negative, so "the largest-magnitude entry of each left singular
//      vector is positive" becomes "the largest-magnitude entry of each
//      transformed column is positive" — the same claim, observable.
//
//  Every section is wrapped in its own `try` ([K56]), and helper `def`s are
//  hoisted above their section because a `def` inside a `try` is scoped to it
//  (language trap 33).
//
//  Run:  bantu run tests/bkit_decompose_test.b
// ════════════════════════════════════════════════════════════════════════

include "../numba/numba.b" as np;
include "../bkit/base.b" as base;
include "../bkit/decompose.b" as dec;

$R = {"pass": 0, "fail": 0};
def ok($cond, $name) {
    if ($cond) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $name); }
}
def eq($got, $want, $name) {
    if (str($got) == str($want)) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name);
        print("          got:  " + str($got));
        print("          want: " + str($want));
    }
}
def close($got, $want, $tol, $name) {
    if (abs($got - $want) <= $tol) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name + "  got " + str($got) + " want " + str($want));
    }
}
def raises($fn, $name) {
    try { $fn(); $R.fail = $R.fail + 1; print("  FAIL  " + $name + " (no error raised)"); }
    catch ($e) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    return null;
}
def section_threw($name, $e) {
    $R.fail = $R.fail + 1;
    print("  FAIL  [section aborted] " + $name);
    print("          threw: " + str($e));
    return null;
}
def sc($x) { return np.get(np.reshape($x, [1]), [0]); }
def maxAbsDiff($a, $b) {
    return sc(np.max(np.abs(np.subtract($a, $b, null), null), null, false));
}
def bits($a) { return nd_tobytes($a, "f64"); }
// The column of a 2-d array, as a vector.
def col($A, $j) {
    return np.reshape(np.take($A, np.astype(nd([1.0 * $j], "f64"), "i64"), 1),
                      [np.shape($A)[0]]);
}

print("========================================");
print("  bkit/decompose.b — PCA and TruncatedSVD");
print("========================================");

np.seed(3);
$X  = np.random([50, 4], 0.0, 1.0);
$mu = np.reshape(np.mean($X, 0, false), [4]);
$Xc = np.subtract($X, $mu, null);

print("");
print("== the components ARE numba's SVD of the centred matrix ==");
try {
// The reference is computed here, from np.svd, with the same sign convention
// applied independently -- not by calling the estimator a second way.
$ref  = np.svd($Xc);
$refU = $ref[0];
$refS = $ref[1];
// numba's svd returns V, not Vt (its own linalg test says "V is n-by-k").
$refV = np.transpose($ref[2], null);
// Apply the convention to the reference by hand, so the comparison tests the
// COMPONENTS and not the flip helper against itself.
$signs = [];
$j = 0;
while ($j < 4) {
    $c = col($refU, $j);
    $at = np.get(np.astype(np.argmax(np.abs($c, null), 0, false), "i64"), [0]);
    $s = 1.0;
    if (np.get($c, [$at]) < 0.0) { $s = 0.0 - 1.0; }
    append($signs, $s);
    $j = $j + 1;
}
$refVf = np.multiply($refV, np.reshape(np.array($signs, "f64"), [4, 1]), null);

$p4 = dec.PCA(null);
$p4.fit($X, null);
eq(np.shape($p4.components()), [4, 4], "all four components for a 50x4 matrix");
ok(maxAbsDiff($p4.components(), $refVf) < 1.0e-10,
   "and they match numba's SVD of the centred matrix to 1e-10 ("
   + str(maxAbsDiff($p4.components(), $refVf)) + ")");
ok(maxAbsDiff($p4.singular_values(), $refS) < 1.0e-12,
   "the singular values match too");
// explained_variance is S^2/(n-1), checked against the formula rather than
// against the field.
ok(maxAbsDiff($p4.explained_variance(),
              np.divide(np.multiply($refS, $refS, null), 49.0, null)) < 1.0e-12,
   "explained_variance is S^2/(n-1), with the unbiased denominator");
close(sc(np.sum($p4.explained_variance_ratio(), null, false)), 1.0, 1.0e-12,
      "and the ratios sum to 1 when every component is kept");
// Descending, which is what makes "the first k" mean anything.
$desc = true;
$j = 1;
while ($j < 4) {
    if (np.get($p4.explained_variance(), [$j]) > np.get($p4.explained_variance(), [$j - 1])) {
        $desc = false;
    }
    $j = $j + 1;
}
ok($desc, "the variances are in descending order");
// The mean is learned, and it is the training mean.
ok(maxAbsDiff($p4.mean(), $mu) < 1.0e-15, "and the stored mean is X's mean");
} catch ($e) { section_threw("the components are numba's SVD", $e); }

print("");
print("== the sign convention, through the public surface ==");
try {
// U is not exposed, but transform returns U*S and every singular value is
// non-negative -- so "the largest-magnitude entry of each left singular vector
// is positive" is observable as "the largest-magnitude entry of each
// transformed column is positive". Without svd_flip this is a coin toss per
// component, and the whole point is that it is not.
$Z4 = $p4.transform($X);
$allPos = true;
$j = 0;
while ($j < 4) {
    $c  = col($Z4, $j);
    $at = np.get(np.astype(np.argmax(np.abs($c, null), 0, false), "i64"), [0]);
    if (np.get($c, [$at]) <= 0.0) { $allPos = false; }
    $j = $j + 1;
}
ok($allPos,
   "every transformed column's largest-magnitude entry is POSITIVE -- the sign "
   + "of each component is pinned, not left to whichever way LAPACK oriented it");
// Two fits of the same data must agree to the bit. That is weaker than the
// line above (it would also hold if LAPACK were merely consistent), which is
// why both are here.
$again = dec.PCA(null);
$again.fit($X, null);
ok(bits($again.components()) == bits($p4.components()),
   "two fits of the same data give bit-identical components");
// And a NEGATED input must give components that differ only in the rows whose
// orientation genuinely changed -- the convention is about U, so flipping X
// flips the data, not the rule. What must hold either way is that the rule is
// still satisfied.
$pn = dec.PCA(null);
$pn.fit(np.multiply($X, 0.0 - 1.0, null), null);
$Zn = $pn.transform(np.multiply($X, 0.0 - 1.0, null));
$negOk = true;
$j = 0;
while ($j < 4) {
    $c  = col($Zn, $j);
    $at = np.get(np.astype(np.argmax(np.abs($c, null), 0, false), "i64"), [0]);
    if (np.get($c, [$at]) <= 0.0) { $negOk = false; }
    $j = $j + 1;
}
ok($negOk, "and the rule still holds on a negated input");
} catch ($e) { section_threw("the sign convention", $e); }

print("");
print("== reconstruction: exact with all of them, best-rank-k without ==");
try {
$Z4b = $p4.transform($X);
ok(maxAbsDiff($p4.inverse_transform($Z4b), $X) < 1.0e-12,
   "all four components reconstruct X to 1e-12 ("
   + str(maxAbsDiff($p4.inverse_transform($Z4b), $X)) + ")");
// Error must fall monotonically as components are added. That is the Eckart-
// Young property in the form this suite can check: a rank-k truncation of the
// SVD is the best rank-k approximation, so adding a component cannot hurt.
$prev = 0.0 - 1.0;
$k = 1;
$mono = true;
while ($k <= 4) {
    $pk = dec.PCA($k);
    $pk.fit($X, null);
    $err = maxAbsDiff($pk.inverse_transform($pk.transform($X)), $X);
    if ($prev >= 0.0 && $err > $prev + 1.0e-12) { $mono = false; }
    $prev = $err;
    $k = $k + 1;
}
ok($mono, "and the error never rises as a component is added");
// The transform of the training data is U*S, so its columns are orthogonal.
$Zt = np.matmul(np.transpose($Z4b, null), $Z4b);
$offDiag = 0.0;
$a = 0;
while ($a < 4) {
    $b = 0;
    while ($b < 4) {
        if ($a != $b) {
            if (abs(np.get($Zt, [$a, $b])) > $offDiag) {
                $offDiag = abs(np.get($Zt, [$a, $b]));
            }
        }
        $b = $b + 1;
    }
    $a = $a + 1;
}
ok($offDiag < 1.0e-10,
   "the transformed columns are orthogonal (largest off-diagonal "
   + str($offDiag) + ") -- which is what a principal component IS");
} catch ($e) { section_threw("reconstruction", $e); }

print("");
print("== n_components as a fraction is a VARIANCE TARGET ==");
try {
// Built so the answer is known: three directions with variances in a 100:10:1
// ratio, so one component carries ~90% and two carry ~99%.
np.seed(11);
$S3 = np.concat([np.multiply(np.random([200, 1], 0.0 - 1.0, 1.0), 10.0, null),
                 np.multiply(np.random([200, 1], 0.0 - 1.0, 1.0), 3.1623, null),
                 np.random([200, 1], 0.0 - 1.0, 1.0)], 1);
$pv = dec.PCA(null);
$pv.fit($S3, null);
$r = $pv.explained_variance_ratio();
print("        ratios: " + str(np.to_list($r)));
$c1 = np.get($r, [0]);
$c2 = $c1 + np.get($r, [1]);
$pA = dec.PCA($c1 - 0.01);  $pA.fit($S3, null);
eq($pA.n_components(), 1, "a target just under the first ratio takes ONE component");
$pB = dec.PCA($c1 + 0.01);  $pB.fit($S3, null);
eq($pB.n_components(), 2, "a target just over it takes two");
$pC = dec.PCA($c2 + 0.001); $pC.fit($S3, null);
eq($pC.n_components(), 3, "and one over the first two takes three");
// The resolved count is a FIT-TIME quantity: the request is 0.9, the answer is
// a number of components, and asking before a fit is an error rather than a
// guess.
raises(def() { return dec.PCA(0.9).n_components(); },
       "n_components() before a fit is refused -- it is learned, not requested");
eq($pB.get_params().n_components, $c1 + 0.01,
   "while get_params still reports what was ASKED for");
} catch ($e) { section_threw("n_components as a fraction", $e); }

print("");
print("== TruncatedSVD does not centre, and that is the whole difference ==");
try {
// On already-centred data the two agree exactly...
$tc = dec.TruncatedSVD(null);
$tc.fit($Xc, null);
ok(maxAbsDiff($tc.components(), $p4.components()) < 1.0e-12,
   "on CENTRED data TruncatedSVD and PCA give the same components");
// ...and on uncentred data they must NOT, or the centring step is not really
// absent. The first singular value is the giveaway: uncentred, the leading
// direction points at the mean.
$tu = dec.TruncatedSVD(null);
$tu.fit($X, null);
ok(maxAbsDiff($tu.components(), $p4.components()) > 0.1,
   "on UNCENTRED data they differ substantially -- only this says the centring "
   + "is really absent");
ok(np.get($tu.singular_values(), [0]) > 2.0 * np.get($p4.singular_values(), [0]),
   "and its leading singular value is far larger ("
   + str(np.get($tu.singular_values(), [0])) + " vs "
   + str(np.get($p4.singular_values(), [0])) + "), because it points at the mean");
eq($tu.mean(), null, "TruncatedSVD stores no mean at all");
ok($p4.mean() != null, "while PCA does");
// Round-trip without a mean to add back.
ok(maxAbsDiff($tu.inverse_transform($tu.transform($X)), $X) < 1.0e-12,
   "and it still reconstructs exactly with every component");
} catch ($e) { section_threw("TruncatedSVD does not centre", $e); }

print("");
print("== PCA centres test data on the TRAINING mean ==");
try {
// Recomputing the mean at transform would centre each batch on its own mean,
// which is a different transformation and the same leak shape as fitting a
// scaler on the test set. Shifting the input by a constant must move the
// output; if the mean were relearned, it would not.
$shift = np.add($X, 5.0, null);
$Zs = $p4.transform($shift);
$Z0 = $p4.transform($X);
ok(maxAbsDiff($Zs, $Z0) > 1.0,
   "a shifted input gives a shifted transform -- the mean is NOT relearned");
// And the shift is exactly the projection of the shift vector, which is a
// statement about linearity that a relearned mean would also break.
$expect = np.matmul(np.reshape(np.add(np.zeros([4], "f64"), 5.0, null), [1, 4]),
                    np.transpose($p4.components(), null));
ok(maxAbsDiff(np.subtract($Zs, $Z0, null),
              np.add(np.zeros([50, 4], "f64"), $expect, null)) < 1.0e-12,
   "and it is exactly the projection of the shift");
} catch ($e) { section_threw("PCA centres on the training mean", $e); }

print("");
print("== what decompose refuses ==");
try {
raises(def() { return dec.PCA(0); },        "n_components 0 is refused");
raises(def() { return dec.PCA(0.0 - 1.0); },"a negative n_components is refused");
raises(def() { return dec.PCA(1.5); },      "1.5 is neither a count nor a fraction");
raises(def() { return dec.PCA("two"); },    "a string n_components is refused");
raises(def() {
    $m = dec.PCA(9);
    return $m.fit($X, null);
}, "more components than min(n_samples, n_features) is refused at fit");
raises(def() {
    $m = dec.PCA(2);
    return $m.fit(np.reshape($X, [200]), null);
}, "a 1-d X is refused");
raises(def() {
    $m = dec.PCA(1);
    return $m.fit(np.reshape(np.array([1.0, 2.0, 3.0, 4.0], "f64"), [1, 4]), null);
}, "a single row is refused -- there is no variance to decompose");
raises(def() { return dec.PCA(2).transform($X); }, "transform before fit is refused");
raises(def() {
    $m = dec.PCA(2);
    $m.fit($X, null);
    return $m.transform(np.random([5, 7], 0.0, 1.0));
}, "transform with the wrong feature count is refused");
raises(def() {
    $m = dec.PCA(2);
    $m.fit($X, null);
    return $m.inverse_transform(np.random([5, 3], 0.0, 1.0));
}, "inverse_transform with the wrong component count is refused");
// [K68]: set_params re-checks the value.
raises(def() {
    $m = dec.PCA(2);
    return $m.set_params({"n_components": 0});
}, "and set_params refuses an invalid n_components, not just the constructor");
raises(def() {
    $m = dec.PCA(2);
    return $m.set_params({"n_componets": 2});
}, "a mistyped key is refused (K24)");
} catch ($e) { section_threw("what decompose refuses", $e); }

print("");
print("== a constant matrix has no variance, and says so with zeros ==");
try {
// Every singular value zero. The ratios are 0/0; reporting NaN would propagate
// into any downstream sum, while zeros state the truth about this matrix.
$K = np.add(np.zeros([20, 3], "f64"), 7.0, null);
$pk = dec.PCA(null);
$pk.fit($K, null);
eq(sc(np.sum($pk.explained_variance_ratio(), null, false)), 0,
   "a constant matrix reports ZERO explained variance, not NaN");
eq(sc(np.sum(np.astype(np.isnan($pk.explained_variance_ratio(), null), "f64"),
             null, false)), 0,
   "and no ratio is NaN");
ok(maxAbsDiff($pk.inverse_transform($pk.transform($K)), $K) < 1.0e-12,
   "while the round trip still returns the constant");
} catch ($e) { section_threw("a constant matrix", $e); }

print("");
print("== tier 4: a real-sized decomposition ==");
try {
np.seed(19);
$big = np.random([4000, 40], 0.0, 1.0);
$t0 = clock();
$pb = dec.PCA(10);
$pb.fit($big, null);
$tf = clock() - $t0;
$Zb = $pb.transform($big);
print("        4000x40 -> 10 components: fit " + str($tf) + " ms");
eq(np.shape($Zb), [4000, 10], "the transform has the requested shape");
ok(sc(np.sum($pb.explained_variance_ratio(), null, false)) < 1.0,
   "ten of forty components explain less than everything");
ok(maxAbsDiff($pb.inverse_transform($Zb), $big) > 0.0,
   "and a rank-10 reconstruction of full-rank noise is not exact");
} catch ($e) { section_threw("tier 4", $e); }

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
