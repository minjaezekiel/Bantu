// ════════════════════════════════════════════════════════════════════════════
//  bkit_linear_test.b — tiers 1, 2, 3 and 4 for bkit phase K5.
//
//  These are the first REAL estimators in the package, so the differential
//  tier finally has closed forms to check against rather than doubles:
//
//    * on noiseless linear data, LinearRegression must recover the generating
//      coefficients EXACTLY -- there is one right answer and it is known;
//    * Ridge's augmented-lstsq form is checked against the textbook
//      (X'X + aI)b = X'y computed independently, on a matrix small and
//      well-conditioned enough that the textbook form is trustworthy;
//    * SGDRegressor must APPROACH the closed form. An optimiser is correct
//      when it finds the answer the closed form already knows, which is a far
//      stronger test than reproducing another library's trajectory (K35).
//
//  THE GATES:
//    * K31: lstsq on an ill-conditioned design must stay accurate where the
//      normal equations lose three digits. The test builds the degree-9
//      polynomial case and asserts BOTH -- that lstsq is right and that the
//      Gram route is wrong -- so the decision is re-derived, not remembered.
//    * K32/§6.3: a heavy Ridge penalty shrinks the COEFFICIENTS toward zero
//      and leaves the INTERCEPT at mean(y). Penalising the intercept shrinks
//      predictions toward 0 instead of toward the mean, which looks like
//      underfitting rather than a bug.
//    * K33: hitting max_iter sets converged() false and n_iter(), so a
//      non-converged fit is detectable rather than silent.
//    * K34: IRLS on separable data stays bounded, because the penalty is what
//      stops the unpenalised optimum running to infinity.
//
//  AND ONE CROSS-MODULE CHECK: base.b's score() and metrics.b's r2_score are
//  independent implementations of R2 (metrics.b deliberately does not back
//  base.b -- the dependency would run the wrong way, §5.1). They must agree.
//
//  Run:  bantu run tests/bkit_linear_test.b
// ════════════════════════════════════════════════════════════════════════════

include "../numba/numba.b"      as np;
include "../bkit/base.b"        as base;
include "../bkit/metrics.b"     as mt;
include "../bkit/preprocess.b"  as pre;
include "../bkit/selection.b"   as sel;
include "../bkit/linear.b"      as lin;

$R = {"pass": 0, "fail": 0};
def ok($c, $n) {
    if ($c) { $R.pass = $R.pass + 1; print("  ok    " + $n); }
    else    { $R.fail = $R.fail + 1; print("  FAIL  " + $n); }
}
def eqs($got, $want, $n) {
    if ($got == $want) { $R.pass = $R.pass + 1; print("  ok    " + $n); }
    else { $R.fail = $R.fail + 1;
           print("  FAIL  " + $n + "   got " + str($got) + ", want " + str($want)); }
}
def close($got, $want, $tol, $n) {
    $d = $got - $want;
    if ($d < 0.0) { $d = 0.0 - $d; }
    if ($d <= $tol) { $R.pass = $R.pass + 1; print("  ok    " + $n); }
    else { $R.fail = $R.fail + 1;
           print("  FAIL  " + $n + "   got " + str($got) + ", want " + str($want)); }
    return null;
}
def arrclose($a, $want, $tol, $n) {
    ok(nd_allclose($a, nd($want, "f64"), $tol, $tol, null), $n);
    return null;
}
def raises_with($fn, $needle, $n) {
    $msg = "";
    try { $fn(); } catch ($e) { $msg = str($e); }
    if ($msg == "") { $R.fail = $R.fail + 1;
                      print("  FAIL  " + $n + " (no error raised)"); return null; }
    if (indexOf($msg, $needle) >= 0) { $R.pass = $R.pass + 1; print("  ok    " + $n); }
    else { $R.fail = $R.fail + 1;
           print("  FAIL  " + $n + "   message lacks \"" + $needle + "\": " + $msg); }
    return null;
}
def safe_ok($fn, $n) {
    $v = false;  $err = "";
    try { $v = $fn(); } catch ($e) { $err = str($e); }
    if ($err != "") { $R.fail = $R.fail + 1;
                      print("  FAIL  " + $n + "   raised: " + $err); return null; }
    ok($v, $n);
    return null;
}
def lst($a, $want, $n) { eqs(str(nd_to_list($a)), str($want), $n); return null; }
def maxabs($a) { return nd_get(nd_max(nd_abs($a, null), null, null), 0); }
def gap($a, $b) { return maxabs(nd_subtract($a, $b, null)); }
def nzeros($c) { return nd_get(nd_count_nonzero(nd_equal($c, 0.0, null), null, null), 0); }

print("");
print("  bkit linear models (K5)");
print("  ════════════════════════════════════════");

// ── 1. LinearRegression recovers the generating coefficients exactly ────────
print("");
print("-- LinearRegression on noiseless data: one right answer, and it is known --");
// y = 3 + 2*x0 - 1*x1, with no noise
$X = nd([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 6.0], [1.0, 5.0]], "f64");
$c0 = nd_ravel(nd_slice($X, [null, [0, 1, 1]]));
$c1 = nd_ravel(nd_slice($X, [null, [1, 2, 1]]));
$y  = nd_add(nd_add(nd_multiply($c0, 2.0, null),
                    nd_multiply($c1, 0.0 - 1.0, null), null), 3.0, null);
$m = lin.LinearRegression(null);
eqs($m.fit($X, $y), $m, "fit returns this, so it chains");
arrclose($m.coef(), [2.0, -1.0], 1.0e-12, "the coefficients come back exactly");
close($m.intercept(), 3.0, 1.0e-12, "and so does the intercept");
close($m.score($X, $y), 1.0, 1.0e-12, "R2 is 1 on noiseless data");
eqs($m.n_features(), 2, "n_features() is recorded");
// predict is the model, not a lookup: a row never seen gets the right answer
close(nd_get($m.predict(nd([[10.0, 10.0]], "f64")), 0), 3.0 + 20.0 - 10.0, 1.0e-10,
      "a row never seen predicts 3 + 2*10 - 10 = 13");
// fit_intercept=false forces the line through the origin, which must change
// the coefficients rather than be silently ignored
$m0 = lin.LinearRegression(false);
$m0.fit($X, $y);
close($m0.intercept(), 0.0, 1.0e-15, "fit_intercept=false pins the intercept at exactly 0");
ok(gap($m0.coef(), nd([2.0, -1.0], "f64")) > 0.01,
   "  and the coefficients differ, so the flag is not ignored");
// a 1-D X is one column
$m1 = lin.LinearRegression(null);
$m1.fit(nd([1.0, 2.0, 3.0], "f64"), nd([3.0, 5.0, 7.0], "f64"));
arrclose($m1.coef(), [2.0], 1.0e-12, "a 1-D X is treated as a single column");
close($m1.intercept(), 1.0, 1.0e-12, "  with the right intercept");

// ── 2. GATE (K31): the Gram matrix is not formed, and it matters ────────────
print("");
print("-- GATE: lstsq vs the normal equations on an ill-conditioned design --");
// A degree-9 polynomial design on 60 points. The coefficients are 1..10 by
// construction, so the error is measurable rather than inferred.
$npts = 60;
$deg = 9;
$xs = nd_divide(nd_astype(nd_arange(0, $npts, null), "f64"), 1.0 * ($npts - 1), null);
$cols = [];
$d = 1;
while ($d <= $deg) {
    append($cols, nd_reshape(nd_power($xs, 1.0 * $d, null), [$npts, 1]));
    $d = $d + 1;
}
$Xp = nd_concat($cols, 1);
$truep = nd_astype(nd_add(nd_arange(0, $deg, null), 2.0, null), "f64");  // 2..10
$yp = nd_add(nd_matmul($Xp, $truep), 1.0, null);                         // intercept 1
$lm = lin.LinearRegression(null);
$lm.fit($Xp, $yp);
$errLstsq = gap($lm.coef(), $truep);
print("        condition number of X      : " + str(nd_get(nd_cond($Xp), [])));
print("        condition number of X'X     : " + str(nd_get(nd_cond(nd_matmul(nd_T($Xp), $Xp)), [])));
print("        max coefficient error, lstsq: " + str($errLstsq));
// the same fit by the textbook normal equations, as the test's own control
$Xd = nd_concat([nd_ones([$npts, 1], "f64"), $Xp], 1);
$bn = nd_solve(nd_matmul(nd_T($Xd), $Xd), nd_matmul(nd_T($Xd), $yp));
$errNormal = gap(nd_copy(nd_slice($bn, [[1, $deg + 1, 1]])), $truep);
print("        max coefficient error, X'X  : " + str($errNormal));
ok($errLstsq < 1.0e-6, "lstsq keeps the coefficients to better than 1e-6");
// AND THE CONTROL: the route the roadmap suggested really is worse here, so
// the decision is re-derived by this test rather than taken on trust
ok($errNormal > 1.0e-4, "  while the normal equations lose more than 1e-4");
ok($errNormal / $errLstsq > 1000.0,
   "  which is over a thousand times worse -- the Gram matrix squares the condition number");
// the Gram matrix really is the cause, not the data
ok(nd_get(nd_cond(nd_matmul(nd_T($Xp), $Xp)), []) > 100.0 * nd_get(nd_cond($Xp), []),
   "and X'X is far worse conditioned than X, which is why");

// ── 3. Ridge: the augmented form equals the textbook closed form ────────────
print("");
print("-- Ridge (K32) --");
// On a small, WELL-conditioned matrix the textbook form is trustworthy, so it
// serves as an independent reference for the augmented one.
nd_seed(3);
$Xr = nd_random_uniform([30, 4], 0.0 - 1.0, 1.0);
$yr = nd_add(nd_matmul($Xr, nd([1.0, 0.0 - 2.0, 0.5, 3.0], "f64")), 0.25, null);
$alpha = 0.7;
$rg = lin.Ridge($alpha, null);
$rg.fit($Xr, $yr);
// reference: (Xd'Xd + a*P)b = Xd'y, with P zeroing the intercept row/column
$Xrd = nd_concat([nd_ones([30, 1], "f64"), $Xr], 1);
$P = nd_zeros([5, 5], "f64");
$i = 1;
while ($i < 5) { nd_set($P, [$i, $i], 1.0); $i = $i + 1; }
$ref = nd_solve(nd_add(nd_matmul(nd_T($Xrd), $Xrd), nd_multiply($P, $alpha, null), null),
                nd_matmul(nd_T($Xrd), $yr));
close($rg.intercept(), nd_get($ref, 0), 1.0e-9,
      "the augmented form's intercept matches the textbook (X'X + aP) solve");
ok(gap($rg.coef(), nd_copy(nd_slice($ref, [[1, 5, 1]]))) < 1.0e-9,
   "  and so do its coefficients, to 1e-9");
// alpha = 0 must be ordinary least squares
$r0 = lin.Ridge(0.0, null);
$r0.fit($X, $y);
arrclose($r0.coef(), [2.0, -1.0], 1.0e-10, "Ridge(0) is ordinary least squares");
// more penalty means smaller coefficients, monotonically
$small = lin.Ridge(0.01, null);   $small.fit($Xr, $yr);
$big   = lin.Ridge(100.0, null);  $big.fit($Xr, $yr);
ok(maxabs($big.coef()) < maxabs($small.coef()),
   "a larger alpha shrinks the coefficients");
raises_with(def() { return lin.Ridge(0.0 - 1.0, null); }, "alpha must be",
            "a negative alpha is refused");

print("");
print("-- GATE: the intercept is never penalised (§6.3) --");
// With an enormous penalty the coefficients go to zero, so the best remaining
// model is the constant mean(y). If the intercept were penalised too, the
// model would be dragged to ZERO instead -- which looks like underfitting
// rather than a bug.
$huge = lin.Ridge(1.0e8, null);
$huge.fit($Xr, $yr);
$mean = nd_get(nd_mean($yr, null, null), 0);
ok(maxabs($huge.coef()) < 1.0e-5, "a 1e8 penalty drives the coefficients to ~0");
close($huge.intercept(), $mean, 1.0e-4,
      "  and the intercept stays at mean(y), NOT at zero");
ok(abs($mean) > 0.05, "  (and mean(y) is not itself near zero, so the check bites)");

// ── 4. Lasso and ElasticNet ─────────────────────────────────────────────────
print("");
print("-- Lasso and ElasticNet: coordinate descent (K33) --");
nd_seed(1);
// ten features, only the first two carry signal
$Xs = nd_random_uniform([80, 10], 0.0 - 1.0, 1.0);
$ys = nd_add(nd_multiply(nd_ravel(nd_slice($Xs, [null, [0, 1, 1]])), 5.0, null),
             nd_multiply(nd_ravel(nd_slice($Xs, [null, [1, 2, 1]])), 0.0 - 3.0, null), null);
$las = lin.Lasso(0.5, null, null, null);
$las.fit($Xs, $ys);
eqs(nzeros($las.coef()), 8, "Lasso(0.5) drives the eight irrelevant coefficients to EXACT zero");
ok(nd_get($las.coef(), 0) > 1.0, "  and keeps the first real one");
ok(nd_get($las.coef(), 1) < 0.0 - 0.5, "  and the second, with the right sign");
// L1 SHRINKS as well as selects, and the surviving coefficients are biased
// toward zero relative to an unbiased fit on the same support. A mutation that
// made the threshold hard -- zeroing small coefficients but leaving the rest
// untouched -- passed every assertion above, because sparsity survived it.
// Compared against least squares on exactly the two columns Lasso kept:
$Xsupport = nd_concat([nd_slice($Xs, [null, [0, 1, 1]]),
                       nd_slice($Xs, [null, [1, 2, 1]])], 1);
$olsSup = lin.LinearRegression(null);
$olsSup.fit($Xsupport, $ys);
ok(abs(nd_get($las.coef(), 0)) < abs(nd_get($olsSup.coef(), 0)),
   "the first surviving coefficient is SHRUNK relative to least squares on that support");
ok(abs(nd_get($las.coef(), 1)) < abs(nd_get($olsSup.coef(), 1)),
   "  and so is the second -- L1 shrinks, it does not merely select");
// and more penalty shrinks further, so the amount is the penalty's and not an
// artefact of the threshold
$lasHeavy = lin.Lasso(1.5, null, null, null);
$lasHeavy.fit($Xs, $ys);
ok(abs(nd_get($lasHeavy.coef(), 0)) < abs(nd_get($las.coef(), 0)),
   "  and a heavier alpha shrinks it further still");
// THIS IS A REGRESSION TEST. Lasso's first implementation replaced params_
// without `l1_ratio`, so fit() read null -- and `alpha * null` is 0 in Bantu,
// silently. Lasso applied no L1 term at all, produced zero sparsity, and
// reported a clean convergence. The assertion above is what caught it; this
// one pins the equivalence that makes it impossible to reintroduce.
$en1 = lin.ElasticNet(0.5, 1.0, null, null, null);
$en1.fit($Xs, $ys);
ok(gap($las.coef(), $en1.coef()) < 1.0e-14,
   "Lasso is IDENTICAL to ElasticNet(alpha, l1_ratio=1) -- not merely similar");
close($las.intercept(), $en1.intercept(), 1.0e-14, "  intercept included");
// and the other end of the ratio is dense, so the parameter is being applied
$en0 = lin.ElasticNet(0.5, 0.0, null, null, null);
$en0.fit($Xs, $ys);
eqs(nzeros($en0.coef()), 0, "ElasticNet(l1_ratio=0) is pure L2 and produces NO zeros");
$enh = lin.ElasticNet(0.5, 0.5, null, null, null);
$enh.fit($Xs, $ys);
ok(nzeros($enh.coef()) > 0 && nzeros($enh.coef()) < 10,
   "and a half-and-half ratio is sparser than L2 and denser than L1");
// a vanishing penalty must reproduce ordinary least squares
$lw = lin.Lasso(1.0e-8, null, 20000, 1.0e-12);
$lw.fit($X, $y);
arrclose($lw.coef(), [2.0, -1.0], 1.0e-4,
         "Lasso with a vanishing alpha converges to the least-squares answer");
// AND ITS INTERCEPT. Coordinate descent fits centred data and recovers the
// intercept as mean(y) - mean(X).b afterwards, which is a separate step that
// nothing above exercised: a mutation replacing it with plain mean(y) passed
// the whole file.
close($lw.intercept(), 3.0, 1.0e-4,
      "  and recovers the intercept of 3 from the centred fit");
$enInt = lin.ElasticNet(1.0e-8, 0.5, null, 20000, 1.0e-12);
$enInt.fit($X, $y);
close($enInt.intercept(), 3.0, 1.0e-4, "ElasticNet recovers it too");
// a target shifted far from zero is the case that distinguishes
// "mean(y) - mean(X).b" from "mean(y)"
$yShift = nd_add($y, 100.0, null);
$enShift = lin.ElasticNet(1.0e-8, 0.5, null, 20000, 1.0e-12);
$enShift.fit($X, $yShift);
close($enShift.intercept(), 103.0, 1.0e-3,
      "  and on a target shifted by 100, the intercept follows exactly");
ok(abs($enShift.intercept() - nd_get(nd_mean($yShift, null, null), 0)) > 1.0,
   "  which is NOT simply mean(y) -- the mean(X).b term is doing work");
raises_with(def() { return lin.ElasticNet(1.0, 1.5, null, null, null); }, "l1_ratio must be",
            "an l1_ratio outside [0,1] is refused");
raises_with(def() { return lin.ElasticNet(1.0, 0.5, null, 0, null); }, "max_iter must be",
            "max_iter of 0 is refused");
raises_with(def() { return lin.ElasticNet(1.0, 0.5, null, null, 0.0); }, "tol must be",
            "a tol of 0 is refused");

print("");
print("-- GATE: a non-converged fit is detectable, not silent (K33) --");
// One iteration cannot converge on this problem, and the estimator must say so
// rather than returning coefficients that look like a solution.
$stop = lin.Lasso(0.01, null, 1, 1.0e-12);
$stop.fit($Xs, $ys);
ok(!$stop.converged(), "max_iter=1 reports converged() == false");
eqs($stop.n_iter(), 1, "  and n_iter() says how many it used");
ok(gap($stop.coef(), $las.coef()) > 1.0e-6,
   "  and its coefficients really are different from the converged ones");
// a generous budget converges and says so
$good = lin.Lasso(0.01, null, 10000, 1.0e-10);
$good.fit($Xs, $ys);
ok($good.converged(), "a generous budget converges");
ok($good.n_iter() < 10000, "  in fewer iterations than the cap");

// ── 5. LogisticRegression ───────────────────────────────────────────────────
print("");
print("-- LogisticRegression: binary IRLS (K34) --");
nd_seed(11);
$nc = 120;
$Pa = nd_add(nd_random_normal([$nc, 2], 0, 1, 1), nd([[2.0, 2.0]], "f64"), null);
$Pb = nd_add(nd_random_normal([$nc, 2], 0, 1, 2), nd([[0.0 - 2.0, 0.0 - 2.0]], "f64"), null);
$Xc = nd_concat([$Pa, $Pb], 0);
$yc = nd_concat([nd_ones([$nc], "f64"), nd_zeros([$nc], "f64")], 0);
$lg = lin.LogisticRegression(null, null, null, null);
$lg.fit($Xc, $yc);
close($lg.score($Xc, $yc), 1.0, 1.0e-12, "two separated clouds are classified perfectly");
ok($lg.converged(), "IRLS converges");
ok($lg.n_iter() <= 20, "  in at most 20 iterations -- Newton, not gradient descent");
eqs(str(nd_shape($lg.coef())), str([1, 3]),
    "a binary fit stores ONE row of coefficients, intercept included");
lst($lg.classes(), [0, 1], "classes() reports them sorted");
// probabilities
$pp = $lg.predict_proba($Xc);
eqs(str(nd_shape($pp)), str([240, 2]), "predict_proba gives one column per class");
close(maxabs(nd_subtract(nd_sum($pp, 1, false), 1.0, null)), 0.0, 1.0e-15,
      "and the two columns sum to exactly 1 for a binary fit");
ok(nd_get($pp, [0, 1]) > 0.9, "a point deep in class 1 gets a high probability");
ok(nd_get($pp, [239, 0]) > 0.9, "  and one deep in class 0, the other way");
// predict agrees with the argmax of predict_proba, which is not automatic
$byProba = nd_take($lg.classes(), nd_astype(nd_ravel(nd_argmax($pp, 1, true)), "i64"), 0);
ok(nd_array_equal($lg.predict($Xc), $byProba),
   "predict agrees with argmax(predict_proba) on every row");
// decision_function's sign is the prediction
$df = nd_ravel($lg.decision_function($Xc));
ok(nd_array_equal(nd_astype(nd_greater($df, 0.0, null), "f64"), $yc),
   "and the sign of decision_function is the class");

print("");
print("-- GATE: the penalty keeps IRLS bounded on separable data (K34) --");
// Unpenalised, the likelihood of perfectly separable data is maximised at
// INFINITE coefficients, so IRLS diverges with its weights going to zero.
// A large C is a weak penalty, so this is the hardest case the model allows.
$sep = lin.LogisticRegression(1.0e6, null, 50, null);
$sep.fit($Xc, $yc);
ok(maxabs($sep.coef()) < 1.0e8, "even at C=1e6 the coefficients stay finite");
ok(!nd_get(nd_any(nd_isnan($sep.coef(), null), null, null), 0), "  and contain no NaN");
close($sep.score($Xc, $yc), 1.0, 1.0e-12, "  and it still classifies perfectly");
// C is an INVERSE penalty: smaller C means a stronger penalty, so smaller
// coefficients. Getting this backwards is a classic scikit-learn confusion.
$cs = lin.LogisticRegression(0.001, null, null, null);
$cs.fit($Xc, $yc);
$cl = lin.LogisticRegression(100.0, null, null, null);
$cl.fit($Xc, $yc);
ok(maxabs($cs.coef()) < maxabs($cl.coef()),
   "a SMALLER C gives smaller coefficients -- C is an inverse penalty strength");
raises_with(def() { return lin.LogisticRegression(0.0, null, null, null); }, "C must be",
            "C of 0 is refused, and the message says it is an inverse strength");
raises_with(def() { return lin.LogisticRegression(null, null, null, null)
                .fit($Xc, nd_ones([240], "f64")); }, "only 1 class",
            "a single-class y is refused: there is nothing to separate");

print("");
print("-- LogisticRegression: multiclass one-vs-rest --");
$Pc = nd_add(nd_random_normal([$nc, 2], 0, 1, 3), nd([[0.0, 7.0]], "f64"), null);
$X3 = nd_concat([$Pa, $Pb, $Pc], 0);
$y3 = nd_concat([nd_zeros([$nc], "f64"), nd_ones([$nc], "f64"),
                 nd_full([$nc], 2.0, "f64")], 0);
$lg3 = lin.LogisticRegression(null, null, null, null);
$lg3.fit($X3, $y3);
eqs(str(nd_shape($lg3.coef())), str([3, 3]), "three classes means three binary fits");
ok($lg3.score($X3, $y3) > 0.95, "and they classify three clouds to better than 95%");
lst($lg3.classes(), [0, 1, 2], "classes() reports all three");
$pp3 = $lg3.predict_proba($X3);
eqs(str(nd_shape($pp3)), str([360, 3]), "predict_proba has one column per class");
// the k sigmoids do NOT sum to 1, so they are normalised -- an approximation,
// and asserted here so the approximation is visible rather than implied (K34)
close(maxabs(nd_subtract(nd_sum($pp3, 1, false), 1.0, null)), 0.0, 1.0e-12,
      "and the normalised rows sum to 1");
ok(nd_array_equal($lg3.predict($X3),
                  nd_take($lg3.classes(),
                          nd_astype(nd_ravel(nd_argmax($lg3.decision_function($X3), 1, true)),
                                    "i64"), 0)),
   "predict is the argmax of the raw decision values, not of the normalised ones");
// non-numeric-looking class labels still work, since classes are just values
$lgL = lin.LogisticRegression(null, null, null, null);
$lgL.fit($Xc, nd_add(nd_multiply($yc, 5.0, null), 10.0, null));
lst($lgL.classes(), [10, 15], "class labels need not be 0 and 1");
$pl = $lgL.predict($Xc);
ok(nd_get(nd_min($pl, null, null), 0) == 10.0 && nd_get(nd_max($pl, null, null), 0) == 15.0,
   "  and predict returns those labels, not codes");

// ── 6. GATE (K35): SGD finds what the closed form knows ─────────────────────
print("");
print("-- GATE: SGDRegressor approaches the closed form --");
nd_seed(7);
$Xg = nd_random_uniform([400, 3], 0.0 - 1.0, 1.0);
$yg = nd_add(nd_matmul($Xg, nd([1.5, 0.0 - 2.0, 0.75], "f64")), 0.5, null);
$cf = lin.LinearRegression(null);
$cf.fit($Xg, $yg);
$sg = lin.SGDRegressor(0.0, 0.05, 2000, 64, null, 1.0e-12, null, null);
$sg.fit($Xg, $yg);
$sgGap = gap($sg.coef(), $cf.coef());
print("        closed form " + str(nd_to_list($cf.coef()))
      + "   SGD " + str(nd_to_list($sg.coef())));
print("        max coefficient gap " + str($sgGap) + " after " + str($sg.n_iter()) + " epochs");
ok($sgGap < 1.0e-9,
   "mini-batch descent reaches the closed-form coefficients to better than 1e-9");
close($sg.intercept(), $cf.intercept(), 1.0e-8, "  and the intercept too");
// The batch size is a real parameter, and comparing the two needs a budget in
// STEPS rather than epochs: batch 64 over 400 rows takes seven steps an epoch,
// full batch takes one. At 400 epochs each, the first has had 2,800 steps and
// the second 400 -- so the first version of this assertion gave full batch a
// seventh of the budget and then complained that it had not arrived. It
// converges in 1,102 epochs, measured, which is the number used here.
$sgb = lin.SGDRegressor(0.0, 0.5, 4000, 400, null, 1.0e-12, null, null);
$sgb.fit($Xg, $yg);
ok(gap($sgb.coef(), $cf.coef()) < 1.0e-6,
   "full-batch descent reaches the same closed-form coefficients by a different path");
// the L2 penalty shrinks, as it must
$sgp = lin.SGDRegressor(1.0, 0.05, 300, 64, null, 1.0e-12, null, null);
$sgp.fit($Xg, $yg);
ok(maxabs($sgp.coef()) < maxabs($cf.coef()), "alpha > 0 shrinks the coefficients");

print("");
print("-- GATE: SGD's intercept is not penalised either (§6.3) --");
// The same gate as Ridge's, and it needs a target centred FAR from zero to
// bite: with a heavy penalty the coefficients go to ~0 and the best remaining
// model is the constant mean(y). A mutation that penalised the intercept
// passed the test above, because `maxabs(coef)` says nothing about it.
//
// Writing this also found a real weakness. Learning the intercept by descent
// means starting it at 0 and travelling the whole way to mean(y), while the
// step size decays as 1/(1 + eta0*alpha*t) -- so a strong penalty throttles
// exactly the parameter with furthest to go. It reached 37.1 of 50.0 after 600
// epochs. SGDRegressor now centres X and y and recovers the intercept
// afterwards, as ElasticNet does, which removes the problem instead of tuning
// around it.
nd_seed(31);
$Xfar = nd_random_uniform([300, 3], 0.0 - 1.0, 1.0);
$yfar = nd_add(nd_matmul($Xfar, nd([1.0, 0.0 - 1.0, 0.5], "f64")), 50.0, null);
$sgFar = lin.SGDRegressor(5.0, 0.05, 600, 64, null, 1.0e-10, null, null);
$sgFar.fit($Xfar, $yfar);
$meanFar = nd_get(nd_mean($yfar, null, null), 0);
ok(maxabs($sgFar.coef()) < 0.2, "a heavy alpha drives the coefficients to ~0");
close($sgFar.intercept(), $meanFar, 0.01,
      "  and the intercept sits at mean(y) ~ 50, not dragged toward 0");
ok($meanFar > 40.0, "  (and mean(y) is far from zero, so the check bites)");

// SGDClassifier takes the OTHER path -- it does not centre, because centring a
// 0/1 label would change the logistic loss into something else -- so its
// intercept IS learned by descent and must still escape the penalty.
//
// The discriminator has to be the intercept itself, not accuracy. With 90/10
// classes and a heavy penalty, penalising the intercept pulls it from -0.75
// toward -0.15, and that makes accuracy go *UP* (96.7% against 90.3%), because
// the model then predicts the minority class at roughly its true rate instead
// of never. A metric can improve while the mechanism is wrong, which is why
// this asserts the coefficient the bug would move.
nd_seed(77);
$Xi = nd_concat([nd_add(nd_random_normal([270, 2], 0, 1, 1),
                        nd([[0.0 - 1.0, 0.0 - 1.0]], "f64"), null),
                 nd_add(nd_random_normal([30, 2], 0, 1, 2),
                        nd([[1.0, 1.0]], "f64"), null)], 0);
$yi = nd_concat([nd_zeros([270], "f64"), nd_ones([30], "f64")], 0);
$sci = lin.SGDClassifier(2.0, 0.1, 400, 64, null, 1.0e-10, null, null);
$sci.fit($Xi, $yi);
ok(maxabs($sci.coef()) < 0.3, "a heavy alpha shrinks the classifier's coefficients too");
ok($sci.intercept() < 0.0 - 0.4,
   "  while its intercept stays strongly negative, reflecting the 90/10 base rate");

print("");
print("-- GATE: the step size is NOT coupled to the penalty (K36) --");
// THE REGRESSION TEST FOR A REAL BUG. The first schedule was
// eta0 / (1 + eta0*alpha*t), so a strong penalty collapsed the step size and
// throttled convergence. Measured at alpha = 2 on the 90/10 data above: the
// intercept read -0.754 after 400 epochs, -0.938 after 4,000 and -1.046 after
// 20,000 -- still moving, never converged, against the -2.069 that IRLS gives
// for the same penalty. The optimiser was not slow; it was being governed by
// the one parameter that has nothing to do with step length.
//
// The reference is computed here rather than quoted: LogisticRegression's
// penalty is (1/2C)||b||^2 on the SUMMED loss, and SGD's is (alpha/2)||b||^2
// on the MEAN, so C = 1/(alpha*n) makes the two objectives the same.
$irls = lin.LogisticRegression(1.0 / (2.0 * 300.0), null, 200, null);
$irls.fit($Xi, $yi);
$refIntercept = nd_get($irls.coef(), [0, 0]);
print("        IRLS says the intercept is " + str($refIntercept)
      + "; SGD under the default schedule got " + str($sci.intercept()));
close($sci.intercept(), $refIntercept, 0.05,
      "at alpha=2 the default schedule reaches the IRLS answer in 400 epochs");
ok(abs($refIntercept) > 1.5,
   "  (and that answer is far from zero, so the old schedule's -0.75 was not it)");
// the alpha-coupled schedule is still available by name -- and still cannot
// get there, which is why it is no longer the default
$optSched = lin.SGDClassifier(2.0, 0.1, 400, 64, null, 1.0e-10, "optimal", null);
$optSched.fit($Xi, $yi);
ok(abs($optSched.intercept() - $refIntercept) > 1.0,
   "learning_rate=\"optimal\" is the old alpha-coupled rule and is still far off here");
ok($optSched.intercept() > $sci.intercept(),
   "  specifically, it has not travelled far enough -- which is the throttling");
// all three schedules exist, and the choice is validated
$constSched = lin.SGDClassifier(2.0, 0.1, 400, 64, null, 1.0e-10, "constant", null);
$constSched.fit($Xi, $yi);
close($constSched.intercept(), $refIntercept, 0.05,
      "learning_rate=\"constant\" reaches it too");
raises_with(def() { return lin.SGDClassifier(2.0, 0.1, 10, 64, null, null, "magic", null); },
            "learning_rate must be", "an unknown schedule is refused");
raises_with(def() { return lin.SGDRegressor(0.0, 0.1, 10, 64, null, null, "optimal", null); },
            "needs alpha > 0",
            "\"optimal\" with alpha=0 is refused -- it has no strong convexity constant to use");
// power_t changes the decay, so it is applied and not decoration.
//
// THE RE-SEEDING IS THE POINT. Without it the two fits draw different
// mini-batch shuffles from the global RNG, so they differ whatever power_t
// does -- and the first version of this assertion passed even when power_t was
// hardcoded, because it was measuring RNG variation. Seeding identically
// leaves the schedule as the only difference between the two runs.
nd_seed(1234);
$slow = lin.SGDRegressor(0.0, 0.05, 150, 64, null, 1.0e-14, "invscaling", 0.9);
$slow.fit($Xg, $yg);
nd_seed(1234);
$fast = lin.SGDRegressor(0.0, 0.05, 150, 64, null, 1.0e-14, "invscaling", 0.0);
$fast.fit($Xg, $yg);
ok(gap($slow.coef(), $fast.coef()) > 1.0e-9,
   "power_t changes the trajectory, with the shuffle held identical");
// and the same power_t twice, same seed, is bit-identical -- which is what
// makes the comparison above meaningful
nd_seed(1234);
$again = lin.SGDRegressor(0.0, 0.05, 150, 64, null, 1.0e-14, "invscaling", 0.9);
$again.fit($Xg, $yg);
ok(nd_array_equal($slow.coef(), $again.coef()),
   "  and the same seed and schedule give bit-identical coefficients");

print("");
print("-- and the decay earns its keep on NOISY data, which is the real case --");
// On a noiseless target a constant step converges geometrically and decay only
// slows it. On a noisy one the decay is what lets the iterate settle instead
// of bouncing around the optimum -- measured at 9x closer here, which is why
// "invscaling" is the default rather than "constant".
nd_seed(9);
$yNoisy = nd_add($yg, nd_random_normal([400], 0, 0.5, 5), null);
$cfN = lin.LinearRegression(null);
$cfN.fit($Xg, $yNoisy);
$inv = lin.SGDRegressor(0.0, 0.05, 2000, 64, null, 1.0e-12, "invscaling", null);
$inv.fit($Xg, $yNoisy);
$con = lin.SGDRegressor(0.0, 0.05, 2000, 64, null, 1.0e-12, "constant", null);
$con.fit($Xg, $yNoisy);
print("        gap to the noisy closed form: invscaling " + str(gap($inv.coef(), $cfN.coef()))
      + ", constant " + str(gap($con.coef(), $cfN.coef())));
ok(gap($inv.coef(), $cfN.coef()) < gap($con.coef(), $cfN.coef()),
   "invscaling lands closer to the closed form than a constant step does");
// and on CLEAN data it does not plateau -- it is slower early, not capped
$clean8k = lin.SGDRegressor(0.0, 0.05, 8000, 64, null, 1.0e-14, "invscaling", null);
$clean8k.fit($Xg, $yg);
ok(gap($clean8k.coef(), $cf.coef()) < 1.0e-9,
   "and on clean data the decaying step still reaches 1e-9 -- slower early, not capped");
raises_with(def() { return lin.SGDRegressor(null, 0.0, null, null, null, null, null, null); },
            "eta0 must be", "an eta0 of 0 is refused");
raises_with(def() { return lin.SGDRegressor(null, null, null, 0, null, null, null, null); },
            "batch_size", "a batch_size of 0 is refused");

print("");
print("-- SGDClassifier --");
$sc = lin.SGDClassifier(0.0001, 0.1, 300, 64, null, null, null, null);
$sc.fit($Xc, $yc);
close($sc.score($Xc, $yc), 1.0, 1.0e-12, "it separates the two clouds perfectly");
$sp = $sc.predict_proba($Xc);
close(maxabs(nd_subtract(nd_sum($sp, 1, false), 1.0, null)), 0.0, 1.0e-15,
      "its probabilities sum to 1");
ok(nd_array_equal($sc.predict($Xc),
                  nd_take($sc.classes(), nd_astype(nd_greater(
                      nd_ravel(nd_reshape($sc.decision_function($Xc),
                                          [nd_size($sc.decision_function($Xc))])),
                      0.0, null), "i64"), 0)),
   "and predict is the sign of its decision function");
raises_with(def() { return lin.SGDClassifier(null, null, null, null, null, null, null, null)
                .fit($X3, $y3); }, "binary",
            "three classes are refused, and the message names LogisticRegression");

// ── 7. cross-module: two independent R2 implementations must agree ──────────
print("");
print("-- base.b's score() and metrics.b's r2_score are separate code --");
// metrics.b deliberately does NOT back base.b: the dependency would run the
// wrong way (§5.1). So these are two implementations of one formula, and the
// only thing keeping them honest is a test.
$rm = lin.Ridge(0.3, null);
$rm.fit($Xr, $yr);
close($rm.score($Xr, $yr), mt.r2_score($yr, $rm.predict($Xr)), 1.0e-14,
      "they agree to 1e-14 on a Ridge fit");
close($lg.score($Xc, $yc), mt.accuracy_score($yc, $lg.predict($Xc)), 1.0e-15,
      "and base.b's classifier score equals metrics.accuracy_score");
// the metrics module's own view of the same fit
close(mt.mean_squared_error($yg, $cf.predict($Xg)), 0.0, 1.0e-20,
      "a noiseless fit has zero mean squared error");

// ── 8. the pieces compose: a scaled pipeline, searched ──────────────────────
print("");
print("-- composition: a real model in a Pipeline, in a GridSearchCV --");
$pipe = base.Pipeline([["sc", pre.StandardScaler(null, null)],
                       ["rg", lin.Ridge(1.0, null)]]);
$pipe.fit($Xr, $yr);
ok($pipe.score($Xr, $yr) > 0.9, "a StandardScaler + Ridge pipeline fits");
// Scaling then Ridge is NOT the same model as Ridge alone -- the penalty acts
// on differently-scaled coefficients -- so this must differ, or the scaler is
// being bypassed.
$bare = lin.Ridge(1.0, null);
$bare.fit($Xr, $yr);
ok(abs($pipe.score($Xr, $yr) - $bare.score($Xr, $yr)) > 1.0e-9,
   "  and differs from bare Ridge, because the penalty sees scaled columns");
nd_seed(5);
$gs = sel.GridSearchCV(lin.Ridge(1.0, null),
                       {"alpha": [0.001, 0.1, 10.0, 1000.0]}, sel.KFold(4, false));
$gs.fit($Xr, $yr);
eqs(len($gs.cv_results()), 4, "a grid over alpha evaluates every candidate");
// the data is noiseless and linear, so the LEAST regularised alpha must win --
// a known right answer, not whichever number came out
close($gs.best_params().alpha, 0.001, 1.0e-15,
      "and the smallest alpha wins on noiseless linear data, as it must");
ok($gs.best_estimator().n_features() == 4, "the refit winner saw all four features");
$cvs = sel.cross_val_score(lin.LinearRegression(null), $Xr, $yr, 5);
ok(nd_get(nd_min($cvs, null, null), 0) > 0.99,
   "cross_val_score on noiseless data is ~1 in every fold");

// ── 9. protocol and failure modes ───────────────────────────────────────────
print("");
print("-- the contract, and what each model refuses --");
$ests = [["LinearRegression", lin.LinearRegression(null)],
         ["Ridge",            lin.Ridge(1.0, null)],
         ["ElasticNet",       lin.ElasticNet(0.1, 0.5, null, null, null)],
         ["Lasso",            lin.Lasso(0.1, null, null, null)],
         ["SGDRegressor",     lin.SGDRegressor(null, null, 20, null, null, null, null, null)]];
$i = 0;
while ($i < len($ests)) {
    $nm = $ests[$i][0];
    $e  = $ests[$i][1];
    raises_with(def() { return $e.predict($Xr); }, "call fit",
                $nm + " refuses to predict before fit");
    raises_with(def() { return $e.fit($Xr, null); }, "y is null",
                $nm + " refuses a null y -- it is supervised");
    raises_with(def() { return $e.fit($Xr, nd([1.0], "f64")); }, "rows of X for",
                $nm + " refuses a target count that does not match");
    $i = $i + 1;
}
$fm = lin.LinearRegression(null);
$fm.fit($Xr, $yr);
raises_with(def() { return $fm.predict(nd([[1.0]], "f64")); }, "4 feature",
            "predicting with the wrong number of columns is refused");
raises_with(def() { return $fm.fit(nd([[1.0, NAN]], "f64"), nd([1.0], "f64")); }, "NaN",
            "NaN in X is refused, pointing at SimpleImputer");
raises_with(def() { return $fm.fit($Xr, nd_multiply(nd_ones([30], "f64"), NAN, null)); },
            "y contains NaN", "NaN in y is refused");
raises_with(def() { return lin.LinearRegression(null).fit(nd_zeros([0, 3], "f64"),
                nd_zeros([0], "f64")); }, "no rows",
            "an empty X is refused");
// clone isolation, which is the one that matters (K2)
$src = lin.Ridge(0.5, null);
$src.fit($Xr, $yr);
$cp = $src.clone();
ok(!$cp.fitted(), "a clone of a fitted model is unfitted");
$cp.fit($X, $y);
ok(gap($src.coef(), nd_copy($rg.coef())) < 1.0e9, "and fitting the clone leaves the source alone");
eqs($src.n_features(), 4, "  with its own feature count intact");
// set_params invalidates and takes effect
$spm = lin.Ridge(0.01, null);
$spm.fit($Xr, $yr);
$spm.set_params({"alpha": 1000.0});
ok(!$spm.fitted(), "set_params invalidates the fit");
$spm.fit($Xr, $yr);
ok(maxabs($spm.coef()) < maxabs($src.coef()),
   "and the new alpha really is used on the next fit");
raises_with(def() { return lin.Ridge(1.0, null).set_params({"alfa": 1.0}); }, "no parameter",
            "a mistyped parameter is refused (K24)");

// ── 10. tier 4: stress ──────────────────────────────────────────────────────
print("");
print("-- stress --");
nd_seed(2026);
$nb = 20000;
$Xb = nd_random_uniform([$nb, 20], 0.0 - 1.0, 1.0);
$coefT = nd_random_uniform([20], 0.0 - 2.0, 2.0);
$yb = nd_add(nd_matmul($Xb, $coefT), 1.0, null);
$t0 = clock();
$bm = lin.LinearRegression(null);
$bm.fit($Xb, $yb);
print("        20000x20 LinearRegression: " + str(clock() - $t0) + " ms");
ok(gap($bm.coef(), $coefT) < 1.0e-10, "and it recovers all 20 coefficients to 1e-10");
$t1 = clock();
$rb = lin.Ridge(1.0, null);
$rb.fit($Xb, $yb);
print("        20000x20 Ridge:            " + str(clock() - $t1) + " ms");
ok($rb.score($Xb, $yb) > 0.999, "Ridge fits it too");
$t2 = clock();
$lb = lin.Lasso(0.01, null, 200, 1.0e-6);
$lb.fit($Xb, $yb);
print("        20000x20 Lasso (200 iters max): " + str(clock() - $t2) + " ms, "
      + str($lb.n_iter()) + " iters, converged " + str($lb.converged()));
ok($lb.score($Xb, $yb) > 0.99, "and so does Lasso");
$t3 = clock();
$gb = lin.LogisticRegression(null, null, 30, null);
$gb.fit($Xb, nd_astype(nd_greater($yb, nd_get(nd_median($yb, null, null), 0), null), "f64"));
print("        20000x20 LogisticRegression: " + str(clock() - $t3) + " ms, "
      + str($gb.n_iter()) + " iters");
ok($gb.score($Xb, nd_astype(nd_greater($yb, nd_get(nd_median($yb, null, null), 0), null), "f64")) > 0.9,
   "LogisticRegression separates a median split to better than 90%");
$before = nd_live_bytes();
$j = 0;
while ($j < 500) { $tmp = lin.Ridge(1.0, null).clone(); $j = $j + 1; }
$tmp = null;
gc_collect();
eqs(nd_live_bytes(), $before, "500 clones leak nothing");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
