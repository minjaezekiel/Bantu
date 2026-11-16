// ════════════════════════════════════════════════════════════════════════
//  bkit_ensemble_test.b — the gradient booster.
//
//  The assertion that matters most is the first one, and it is the reason
//  this file is shorter than it would otherwise have to be:
//
//    **ONE ROUND AT learning_rate = 1.0 IS BIT-IDENTICAL TO K7's TREE ON
//    THE RESIDUALS.** With `squared_error`, F0 = ȳ and g = F0 − y = −(y − ȳ),
//    h = 1 — which is exactly what `DecisionTreeRegressor` computes when it is
//    fitted to `y − ȳ`. Same grower, same bins, same g, same h, so the node
//    arrays must agree to the bit, not approximately. That ties every part of
//    the booster the tree already tests (binning, scanning, the NaN direction,
//    the threshold lookup) to a gate K7 has already broken five ways, and
//    leaves this file to test what is actually new: the loop, the objectives,
//    shrinkage, stopping and the subsamples.
//
//  Every section is wrapped in its own `try` ([K56]), so a throw is attributed
//  to its section and counts as one failure instead of ending the run. Helper
//  `def`s are hoisted ABOVE their section, because a `def` inside a `try` block
//  is scoped to that block while a variable is not (language trap 33).
//
//  Run:  bantu run tests/bkit_ensemble_test.b
// ════════════════════════════════════════════════════════════════════════

include "../numba/numba.b" as np;
include "../bkit/base.b" as base;
include "../bkit/tree.b" as tree;
include "../bkit/ensemble.b" as ens;

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
// A 0-d reduction is an ndarray handle and every handle is TRUTHY (numba N41),
// so it has to be read out rather than used as a condition.
def sc($x) { return np.get(np.reshape($x, [1]), [0]); }
def mae($a, $b) {
    return sc(np.mean(np.abs(np.subtract($a, $b, null), null), null, false));
}
def mse($a, $b) {
    $d = np.subtract($a, $b, null);
    return sc(np.mean(np.multiply($d, $d, null), null, false));
}
// Bit-for-bit, not allclose. An allclose here would pass even if the booster
// and the tree took different paths to nearly the same place, which is the one
// thing this file's first gate exists to rule out.
def bits($a) { return nd_tobytes($a, "f64"); }

print("========================================");
print("  bkit/ensemble.b — gradient boosting");
print("========================================");

// A target a depth-2 tree can fit exactly, so "it learned" is checkable rather
// than plausible: four blocks of five rows, one value each.
$Xr = np.reshape(np.astype(np.arange(0.0, 40.0, 1.0), "f64"), [20, 2]);
$step = np.astype(np.floor(np.divide(np.arange(0.0, 20.0, 1.0), 5.0, null), null), "f64");
$yr = np.add(np.multiply($step, 3.0, null), 1.0, null);

print("");
print("== one round at learning_rate 1.0 IS K7's tree on the residuals ==");
try {
// F0 = ȳ and g = F0 − y = −(y − ȳ), h = 1 -- exactly what the tree computes
// when fitted to y − ȳ. λ and min_child_weight are set to the TREE's defaults
// (0), because the booster's are 1.0: the claim is about the algorithm, not
// about the defaults coinciding.
$mean1 = sc(np.mean($yr, null, false));
$resid = np.subtract($yr, $mean1, null);
$t1 = tree.DecisionTreeRegressor(2, 1, null, null, null, null);
$t1.fit($Xr, $resid);
$b1 = ens.GradientBoostingRegressor({"n_estimators": 1, "learning_rate": 1.0,
                                     "max_depth": 2, "reg_lambda": 0.0,
                                     "min_child_weight": 0.0});
$b1.fit($Xr, $yr);
ok(bits($b1.nodes(0, null)) == bits($t1.nodes(null)),
   "the node arrays are BIT-IDENTICAL -- same grower, same g, same h");
ok(bits($b1.predict($Xr)) == bits(np.add($t1.predict($Xr), $mean1, null)),
   "and so is the prediction, F0 + 1.0 x tree");
eq($b1.n_rounds(), 1, "one round means one tree");
eq($b1.n_outputs(), 1, "and one output column");
} catch ($e) { section_threw("one round equals K7's tree on the residuals", $e); }

print("");
print("== shrinkage is exactly the learning rate, not approximately ==");
try {
$b2 = ens.GradientBoostingRegressor({"n_estimators": 1, "learning_rate": 0.25,
                                     "max_depth": 2, "reg_lambda": 0.0,
                                     "min_child_weight": 0.0});
$b2.fit($Xr, $yr);
// The SAME tree, scaled. The leaf already holds the Newton step; the learning
// rate is applied to the step, not folded into the leaf -- which is what makes
// a saved model's leaves readable as the step they are.
ok(bits($b2.nodes(0, null)) == bits($t1.nodes(null)),
   "a lower learning rate does not change the TREE");
ok(bits($b2.predict($Xr))
   == bits(np.add(np.multiply($t1.predict($Xr), 0.25, null), $mean1, null)),
   "only the step: F0 + 0.25 x tree, bit for bit");
} catch ($e) { section_threw("shrinkage is exactly the learning rate", $e); }

print("");
print("== the loop actually descends, and staged_predict shows it ==");
try {
$b3 = ens.GradientBoostingRegressor({"n_estimators": 25, "learning_rate": 0.2,
                                     "max_depth": 2});
$b3.fit($Xr, $yr);
$stg = $b3.staged_predict($Xr);
eq(len($stg), 25, "staged_predict gives one prediction per round");
ok(bits($stg[24]) == bits($b3.predict($Xr)),
   "and its last entry IS predict() -- same arithmetic, not a re-derivation");
$e0 = mse($stg[0], $yr);
$eN = mse($stg[24], $yr);
ok($eN < $e0, "the error falls: " + str($e0) + " -> " + str($eN));
// Non-increasing at every step, which "it got better overall" does not imply.
// A booster whose rounds sometimes undo each other is converging by luck.
$worst = 0.0;
$i = 1;
while ($i < 25) {
    $d = mse($stg[$i], $yr) - mse($stg[$i - 1], $yr);
    if ($d > $worst) { $worst = $d; }
    $i = $i + 1;
}
print("        worst round-over-round increase: " + str($worst));
ok($worst <= 1.0e-12, "and no single round made it worse");
} catch ($e) { section_threw("the loop descends", $e); }

print("");
print("== the target is reachable, so the fit must reach it ==");
try {
$b4 = ens.GradientBoostingRegressor({"n_estimators": 200, "learning_rate": 0.3,
                                     "max_depth": 2});
$b4.fit($Xr, $yr);
$err = mae($b4.predict($Xr), $yr);
print("        mean absolute error: " + str($err));
ok($err < 1.0e-6, "an exactly piecewise-constant target is fitted to 1e-6");
ok($b4.score($Xr, $yr) > 0.999999, "R² is 1 to six nines");
} catch ($e) { section_threw("the reachable target is reached", $e); }

print("");
print("== logistic: probabilities, not scores ==");
try {
$yb = np.astype(np.greater($yr, 6.0, null), "f64");
$cb = ens.GradientBoostingClassifier({"n_estimators": 30, "learning_rate": 0.3,
                                      "max_depth": 2});
$cb.fit($Xr, $yb);
eq($cb.get_params().objective, "logistic", "two classes selects logistic");
eq($cb.n_outputs(), 1, "which is ONE score column, not two");
$P = $cb.predict_proba($Xr);
eq(np.shape($P), [20, 2], "predict_proba is [n, 2] all the same");
ok(mae(np.sum($P, 1, false), np.ones([20], "f64")) < 1.0e-12,
   "each row sums to 1");
ok(sc(np.min($P, null, false)) > 0.0 && sc(np.max($P, null, false)) < 1.0,
   "and no probability is exactly 0 or 1 -- sigma never saturates fully");
eq(sc(np.mean(np.astype(np.equal($cb.predict($Xr), $yb, null), "f64"),
              null, false)), 1,
   "a separable target is classified perfectly");
// predict must be the argmax of predict_proba, or the two disagree about the
// same model.
ok(bits(np.astype($cb.predict($Xr), "f64"))
   == bits(np.astype(np.take($cb.classes(),
                             np.astype(np.argmax($P, 1, false), "i64"), 0), "f64")),
   "predict IS the argmax of predict_proba");
} catch ($e) { section_threw("logistic probabilities", $e); }

print("");
print("== softmax: K columns, normalised JOINTLY ==");
try {
$y3 = np.astype(np.floor(np.divide(np.arange(0.0, 20.0, 1.0), 7.0, null), null), "f64");
$c3 = ens.GradientBoostingClassifier({"n_estimators": 30, "learning_rate": 0.3,
                                      "max_depth": 2});
$c3.fit($Xr, $y3);
eq($c3.get_params().objective, "softmax", "three classes selects softmax");
eq($c3.n_outputs(), 3, "which is three score columns, one tree each per round");
eq($c3.n_rounds() * 3, 90, "so 30 rounds is 90 trees");
$P3 = $c3.predict_proba($Xr);
eq(np.shape($P3), [20, 3], "predict_proba is [n, K]");
ok(mae(np.sum($P3, 1, false), np.ones([20], "f64")) < 1.0e-12,
   "every row sums to 1 -- the columns are normalised against each other, "
   + "which one-vs-rest sigmoids are not");
eq(sc(np.mean(np.astype(np.equal($c3.predict($Xr), $y3, null), "f64"), null, false)), 1,
   "and the three classes are separated perfectly");
eq(np.size($c3.classes()), 3, "classes() reports all three");
} catch ($e) { section_threw("softmax columns", $e); }

print("");
print("== poisson models a count, and says so when y is not one ==");
try {
$yp = np.add($step, 1.0, null);
$pm = ens.GradientBoostingRegressor({"n_estimators": 40, "objective": "poisson",
                                     "learning_rate": 0.3, "max_depth": 2});
$pm.fit($Xr, $yp);
$pp = $pm.predict($Xr);
ok(sc(np.min($pp, null, false)) > 0.0,
   "every prediction is positive -- the link is exp, so it cannot be otherwise");
ok(mae($pp, $yp) < 0.05, "and it fits the counts: mae " + str(mae($pp, $yp)));
raises(def() {
    $bad = ens.GradientBoostingRegressor({"objective": "poisson"});
    return $bad.fit($Xr, np.subtract($yp, 5.0, null));
}, "a negative target is refused: poisson models a count");
} catch ($e) { section_threw("poisson", $e); }

print("");
print("== early stopping: on noise it must stop, and must not peek ==");
try {
// A MILDLY learnable target with a noisy held-out fold, chosen deliberately:
// pure noise stops at round 7 with no stall before its best round, and a
// stopping counter that never resets is then indistinguishable from one that
// does. This fixture improves, stalls, improves again -- and the last assertion
// in this section is what proves it still does.
np.seed(13);
$Xn = np.random([240, 4], 0.0, 1.0);
$yn = np.add(np.reshape(np.take($Xn, np.astype(nd([0.0], "f64"), "i64"), 1), [240]),
             np.random([240], 0.0, 0.5), null);
$es = ens.GradientBoostingRegressor({"n_estimators": 150, "n_iter_no_change": 5,
                                     "learning_rate": 0.1, "max_depth": 3,
                                     "validation_fraction": 0.25});
$es.fit($Xn, $yn);
print("        rounds run " + str($es.n_rounds()) + ", best " + str($es.best_round()));
ok($es.n_rounds() < 150, "it stops well before n_estimators");
ok($es.best_round() <= $es.n_rounds(), "the best round is one it actually ran");
eq(len($es.val_scores()), $es.n_rounds(), "one held-out score per round run");
// best_round must be the ARGMIN of the held-out scores, and predict must use
// exactly that many trees. Asserting only "best <= n_rounds" left both a
// best_round that kept the WORST round and a predict() that ignored it green:
// that inequality is true of any number in range.
$vs   = $es.val_scores();
// The rule, reimplemented here rather than read off the implementation: the
// best round is the last one that improved on the previous best by MORE than
// tol. That is NOT the raw argmin, and the difference is deliberate -- tol is
// the caller's statement that an improvement smaller than it does not count, so
// it must bind the prediction bound as well as the stopping decision. Using the
// raw argmin would contradict the parameter they set.
$bi   = 0;
$bv   = $vs[0];
$i = 1;
while ($i < len($vs)) {
    if ($vs[$i] < $bv - 0.0001) { $bv = $vs[$i]; $bi = $i; }
    $i = $i + 1;
}
eq($es.best_round(), $bi + 1,
   "best_round is the last tol-significant improvement, reimplemented here");
// And the semantics stated as a bound, so the choice above is visible rather
// than implied: the kept model is within tol of the genuine minimum.
$rawMin = sc(np.min(np.array($vs, "f64"), null, false));
ok($vs[$es.best_round() - 1] <= $rawMin + 0.0001,
   "the kept round is within tol of the best round ever seen ("
   + str($vs[$es.best_round() - 1]) + " vs " + str($rawMin) + ")");
ok($bv <= $vs[len($vs) - 1],
   "and no worse than the last round -- otherwise stopping gained nothing");
ok($es.n_rounds() > $es.best_round(),
   "and the run really did continue past its best round (" + str($es.n_rounds())
   + " > " + str($es.best_round()) + "), so the next assertion is not vacuous");
ok(bits($es.predict($Xn)) == bits($es.staged_predict($Xn)[$es.best_round() - 1]),
   "predict uses exactly best_round trees, not every tree it grew");
// Held-out scores are a LOSS, so they cannot be negative. A flipped sign in the
// loss would otherwise only show up as a stopping round nobody checked.
ok(sc(np.min(np.array($vs, "f64"), null, false)) >= 0.0,
   "every held-out score is >= 0 -- it is a loss, not a score to maximise");
// The stopping rule is "stop after n_iter_no_change rounds without a
// tol-sized improvement", which means the run length is pinned exactly. A
// counter that never reset would stop on the CUMULATIVE count of bad rounds and
// break this equality whenever an improvement followed a stall.
eq($es.n_rounds(), $es.best_round() + 5,
   "the run stopped exactly n_iter_no_change rounds after its best");
$stalls = 0;
$i = 1;
while ($i < $bi + 1) {
    if (!($vs[$i] < $vs[$i - 1])) { $stalls = $stalls + 1; }
    $i = $i + 1;
}
print("        stalls before the best round: " + str($stalls));
ok($stalls > 0,
   "and an improvement DID follow a stall here, so the reset is exercised "
   + "rather than assumed");
// THE LEAK GATE (§8.7). The fold is split BEFORE binning, so the bin edges
// cannot have been chosen using data the model is then scored on. That is a
// quieter leak than training on it -- it inflates the stopping round rather
// than the score -- so it is asserted structurally rather than by eye.
eq($es.n_train(), 180, "180 of 240 rows were binned, so 60 were held out");
} catch ($e) { section_threw("early stopping", $e); }

print("");
print("== the leak gate: the validation fold's VALUES never reach the bins ==");
try {
// One feature, 24 distinct values. Binning the whole matrix gives 24 bins;
// binning only the 18 training rows cannot give more than 18. If the fold were
// split after binning, the fitted model would carry 24.
$Xd = np.reshape(np.astype(np.arange(0.0, 24.0, 1.0), "f64"), [24, 1]);
$yd = np.multiply(np.reshape($Xd, [24]), 2.0, null);
eq(np.get(tree_bin_nbins(tree_bin($Xd, 256)), [0]), 24,
   "binning all 24 rows gives 24 bins (the control)");
$lk = ens.GradientBoostingRegressor({"n_estimators": 3, "n_iter_no_change": 2,
                                     "validation_fraction": 0.25, "max_depth": 2});
$lk.fit($Xd, $yd);
eq($lk.n_train(), 18, "the model trained on 18 rows");
ok(np.get($lk.n_bins(), [0]) <= 18,
   "and its bins came from those 18 only: " + str(np.get($lk.n_bins(), [0]))
   + " <= 18");
} catch ($e) { section_threw("the leak gate", $e); }

print("");
print("== reproducible from the seed alone (C4), and the subsamples are real ==");
try {
np.seed(11);
$Xs = np.random([150, 6], 0.0, 1.0);
$ys = np.add(np.multiply(np.reshape(np.take($Xs, np.astype(nd([0.0], "f64"), "i64"), 1),
                                   [150]), 4.0, null),
             np.random([150], 0.0, 0.3), null);
def fitSub() {
    $m = ens.GradientBoostingRegressor({"n_estimators": 12, "subsample": 0.7,
                                        "colsample": 0.5, "max_depth": 3});
    $m.fit($Xs, $ys);
    return $m.predict($Xs);
}
np.seed(99);  $r1 = fitSub();
np.seed(99);  $r2 = fitSub();
ok(bits($r1) == bits($r2), "the same seed gives a BYTE-IDENTICAL model");
// Without reseeding the draws differ -- which is what makes the assertion
// above mean something. An identical result here would prove only that the
// subsamples were never drawn.
$r3 = fitSub();
ok(bits($r1) != bits($r3),
   "and without reseeding it differs, so SOME randomness is drawn");
// That last assertion is satisfied by EITHER subsample or colsample being
// drawn, so on its own it cannot tell which. Dropping subsample entirely left
// it green. Each parameter is now pinned on its own, and deterministically --
// under one seed, turning it on must change the model. This needs no randomness
// to be present at all to mean something.
def fitWith($sub, $col) {
    $m = ens.GradientBoostingRegressor({"n_estimators": 10, "subsample": $sub,
                                        "colsample": $col, "max_depth": 3});
    np.seed(41);
    $m.fit($Xs, $ys);
    return $m.predict($Xs);
}
$base10 = fitWith(1.0, 1.0);
ok(bits(fitWith(0.5, 1.0)) != bits($base10),
   "subsample ALONE changes the model -- the row sample is really taken");
ok(bits(fitWith(1.0, 0.5)) != bits($base10),
   "colsample ALONE changes the model -- the column mask is really applied");
ok(bits(fitWith(0.5, 1.0)) != bits(fitWith(1.0, 0.5)),
   "and the two are not the same switch wearing two names");
// subsample = colsample = 1.0 draws nothing at all, so it must be stable with
// no seeding whatsoever.
def fitFull() {
    $m = ens.GradientBoostingRegressor({"n_estimators": 8, "max_depth": 3});
    $m.fit($Xs, $ys);
    return $m.predict($Xs);
}
ok(bits(fitFull()) == bits(fitFull()),
   "with no subsampling the fit is deterministic WITHOUT reseeding");
} catch ($e) { section_threw("reproducibility and the subsamples", $e); }

print("");
print("== feature importances: gain, split count, and the zero case ==");
try {
// Feature 0 carries the target; features 1 and 2 are noise.
np.seed(5);
$Xi = np.random([120, 3], 0.0, 1.0);
$yi = np.multiply(np.reshape(np.take($Xi, np.astype(nd([0.0], "f64"), "i64"), 1),
                             [120]), 5.0, null);
$mi = ens.GradientBoostingRegressor({"n_estimators": 20, "max_depth": 2,
                                     "learning_rate": 0.3});
$mi.fit($Xi, $yi);
$gi = $mi.feature_importances_(null);
$si = $mi.feature_importances_("split");
eq(np.size($gi), 3, "one importance per feature");
close(sc(np.sum($gi, null, false)), 1.0, 1.0e-12, "gain importances sum to 1");
close(sc(np.sum($si, null, false)), 1.0, 1.0e-12, "split importances sum to 1");
ok(np.get($gi, [0]) > 0.9,
   "the feature that carries the target takes >90% of the gain ("
   + str(np.get($gi, [0])) + ")");
ok(np.get($gi, [0]) > np.get($si, [0]),
   "and it holds MORE of the gain than of the split count -- one decisive "
   + "split and ten marginal ones are not the same thing");
raises(def() { return $mi.feature_importances_("permutation"); },
       "an unsupported kind is refused rather than silently defaulted");
// A model with no splits at all. Dividing by a zero total would give a vector
// of NaNs, which reads as "unknown" when the truthful answer is "this model
// uses no feature".
$mc = ens.GradientBoostingRegressor({"n_estimators": 5, "max_depth": 2});
$mc.fit($Xr, np.ones([20], "f64"));
eq(np.to_list($mc.feature_importances_(null)), [0, 0],
   "a constant target splits on nothing, and reports ZEROS, not NaN");
} catch ($e) { section_threw("feature importances", $e); }

print("");
print("== parameters are a dict, and a wrong one is refused ==");
try {
raises(def() { return ens.GradientBoostingRegressor({"n_estimater": 10}); },
       "a MISTYPED key raises -- bkit has shipped this bug once already (K24)");
raises(def() { return ens.GradientBoostingRegressor({"n_estimators": 0}); },
       "n_estimators 0 is refused");
raises(def() { return ens.GradientBoostingRegressor({"n_estimators": 10.5}); },
       "a fractional n_estimators is refused");
raises(def() { return ens.GradientBoostingRegressor({"learning_rate": 0.0}); },
       "learning_rate 0 is refused -- every tree would contribute nothing");
raises(def() { return ens.GradientBoostingRegressor({"learning_rate": 1.5}); },
       "learning_rate above 1 is refused");
raises(def() { return ens.GradientBoostingRegressor({"subsample": 0.0}); },
       "subsample 0 is refused -- it would fit a model of no rows");
raises(def() { return ens.GradientBoostingRegressor({"subsample": 1.5}); },
       "subsample above 1 is refused");
raises(def() { return ens.GradientBoostingRegressor({"colsample": 0.0}); },
       "colsample 0 is refused");
raises(def() { return ens.GradientBoostingRegressor({"max_depth": 0}); },
       "max_depth 0 is refused");
raises(def() { return ens.GradientBoostingRegressor({"reg_lambda": 0.0 - 1.0}); },
       "a negative reg_lambda is refused");
raises(def() { return ens.GradientBoostingRegressor({"objective": "hinge"}); },
       "an unknown objective is refused");
raises(def() { return ens.GradientBoostingRegressor({"objective": "logistic"}); },
       "a CLASSIFICATION objective on the regressor is refused by name");
raises(def() { return ens.GradientBoostingClassifier({"objective": "poisson"}); },
       "and a regression objective on the classifier");
raises(def() { return ens.GradientBoostingRegressor({"n_iter_no_change": 0}); },
       "n_iter_no_change 0 is refused");
} catch ($e) { section_threw("parameter validation", $e); }

print("");
print("== what the estimators refuse about the data ==");
try {
raises(def() {
    $m = ens.GradientBoostingClassifier(null);
    return $m.fit($Xr, np.ones([20], "f64"));
}, "a single-class target is refused -- it would score 100% on anything");
raises(def() {
    $m = ens.GradientBoostingClassifier({"objective": "logistic"});
    return $m.fit($Xr, np.astype(np.floor(np.divide(np.arange(0.0, 20.0, 1.0),
                                                    7.0, null), null), "f64"));
}, "objective 'logistic' with three classes is refused, not silently reused");
raises(def() {
    $m = ens.GradientBoostingRegressor(null);
    return $m.fit($Xr, np.ones([7], "f64"));
}, "a y of the wrong length is refused");
raises(def() {
    $m = ens.GradientBoostingRegressor(null);
    return $m.fit(np.reshape($Xr, [40]), $yr);
}, "a 1-d X is refused");
raises(def() {
    $m = ens.GradientBoostingRegressor(null);
    return $m.fit($Xr, null);
}, "a null y is refused");
raises(def() { return ens.GradientBoostingRegressor(null).predict($Xr); },
       "predict before fit is refused");
raises(def() {
    $m = ens.GradientBoostingRegressor({"n_estimators": 3});
    $m.fit($Xr, $yr);
    return $m.predict(np.reshape(np.arange(0.0, 9.0, 1.0), [3, 3]));
}, "predict with the wrong number of features is refused");
} catch ($e) { section_threw("data validation", $e); }

print("");
print("== clone is independent, and set_params invalidates the fit ==");
try {
$orig = ens.GradientBoostingRegressor({"n_estimators": 7, "max_depth": 2});
$orig.fit($Xr, $yr);
$cl = $orig.clone();
ok(!$cl.fitted(), "a clone is unfitted");
eq($cl.get_params().n_estimators, 7, "and carries the parameters");
$cl.set_params({"n_estimators": 99});
eq($orig.get_params().n_estimators, 7,
   "changing the clone does NOT change the original -- the dict is rebuilt, "
   + "not shared by reference");
$orig.set_params({"max_depth": 3});
ok(!$orig.fitted(),
   "set_params invalidates the fit, so predict cannot run with trees grown "
   + "under the old parameters");
raises(def() { return $orig.predict($Xr); }, "and predict refuses afterwards");
} catch ($e) { section_threw("clone independence", $e); }

print("");
print("== the model is a list of matrices, and nothing else (K11) ==");
try {
$mm = ens.GradientBoostingClassifier({"n_estimators": 4, "max_depth": 2});
$y3b = np.astype(np.floor(np.divide(np.arange(0.0, 20.0, 1.0), 7.0, null), null), "f64");
$mm.fit($Xr, $y3b);
eq(np.shape($mm.nodes(0, 0))[1], 7, "every node is 7 doubles");
$tot = 0;
$j = 0;
while ($j < 4) {
    $i = 0;
    while ($i < 3) { $tot = $tot + np.shape($mm.nodes($j, $i))[0]; $i = $i + 1; }
    $j = $j + 1;
}
eq($mm.n_nodes(), $tot, "n_nodes is the sum over every tree, all 12 of them");
raises(def() { return $mm.nodes(4, 0); },  "a round past the end is refused");
raises(def() { return $mm.nodes(0, 3); },  "an output past the end is refused");
raises(def() { return $mm.nodes(0 - 1, 0); }, "a negative round is refused");
} catch ($e) { section_threw("the model is a list of matrices", $e); }

print("");
print("== NaN in X works end to end ==");
try {
$Xnan = np.copy($Xr);
$Xnan = np.where(np.equal($Xnan, 8.0, null), np.divide(np.zeros([1], "f64"),
                                                       np.zeros([1], "f64"), null),
                 $Xnan);
ok(sc(np.sum(np.astype(np.isnan($Xnan, null), "f64"), null, false)) > 0.0,
   "the test matrix really does contain NaN");
$mn = ens.GradientBoostingRegressor({"n_estimators": 20, "max_depth": 2,
                                     "learning_rate": 0.3});
$mn.fit($Xnan, $yr);
$pn = $mn.predict($Xnan);
eq(sc(np.sum(np.astype(np.isnan($pn, null), "f64"), null, false)), 0,
   "and no prediction is NaN -- the missing value took a LEARNED direction");
} catch ($e) { section_threw("NaN end to end", $e); }

print("");
print("== tier 4: a real-sized fit ==");
try {
np.seed(21);
$Xb = np.random([3000, 20], 0.0, 1.0);
$cf = np.reshape(np.take($Xb, np.astype(nd([0.0, 1.0, 2.0], "f64"), "i64"), 1), [3000, 3]);
$yb4 = np.add(np.sum(np.multiply($cf, nd([3.0, 0.0 - 2.0, 1.0], "f64"), null), 1, false),
              np.random([3000], 0.0, 0.2), null);
$t0 = clock();
$big = ens.GradientBoostingRegressor({"n_estimators": 50, "max_depth": 4,
                                      "learning_rate": 0.1, "subsample": 0.8,
                                      "colsample": 0.8});
$big.fit($Xb, $yb4);
$tf = clock() - $t0;
$t1b = clock();
$pb = $big.predict($Xb);
print("        3000x20, 50 rounds, depth 4: fit " + str($tf) + " ms, predict "
      + str(clock() - $t1b) + " ms, " + str($big.n_nodes()) + " nodes");
eq(np.size($pb), 3000, "every row gets a prediction");
ok($big.score($Xb, $yb4) > 0.95,
   "R² over 0.95 on a linear target with noise ("
   + str($big.score($Xb, $yb4)) + ")");
// The three informative features must dominate, with 17 noise features present.
$gb = $big.feature_importances_(null);
$top3 = np.get($gb, [0]) + np.get($gb, [1]) + np.get($gb, [2]);
ok($top3 > 0.8, "the three real features hold " + str($top3) + " of the gain");
} catch ($e) { section_threw("tier 4", $e); }

print("");
print("== RandomForest: the same grower, n_estimators times ==");
try {
np.seed(23);
$fX = np.random([300, 6], 0.0, 1.0);
$fSig = np.reshape(np.take($fX, np.astype(nd([0.0], "f64"), "i64"), 1), [300]);
$fY = np.add(np.multiply($fSig, 4.0, null), np.random([300], 0.0, 0.4), null);
$rf = ens.RandomForestRegressor({"n_estimators": 30, "max_depth": 6});
$rf.fit($fX, $fY);
eq($rf.n_trees(), 30, "thirty trees");
eq($rf.n_outputs(), 1, "one output column for a regressor");
ok($rf.n_nodes() > 30, "and n_nodes sums over all of them (" + str($rf.n_nodes()) + ")");
ok($rf.score($fX, $fY) > 0.9, "R² over 0.9 (" + str($rf.score($fX, $fY)) + ")");
ok(np.get($rf.feature_importances_(null), [0]) > 0.8,
   "the one informative feature takes most of the gain");
// The prediction IS the mean of the trees' predictions, reconstructed here from
// the public node arrays rather than taken on trust. An accuracy threshold
// cannot see a divisor error: averaging over n-1 trees instead of n scales
// every prediction by 30/29, which R² > 0.9 absorbs and which the classifier's
// row normalisation cancels outright. This is exact.
$rfSum = np.zeros([300], "f64");
$t = 0;
while ($t < $rf.n_trees()) {
    $rfSum = np.add($rfSum, tree_predict($rf.nodes($t, null), $fX), null);
    $t = $t + 1;
}
ok(bits($rf.predict($fX))
   == bits(np.divide($rfSum, 1.0 * $rf.n_trees(), null)),
   "predict is the MEAN of its trees, bit for bit -- not a sum, and not a mean "
   + "over the wrong count");
} catch ($e) { section_threw("RandomForest regression", $e); }

print("");
print("== C4, the sharp version: fitting 20 gives the same first 10 ==");
try {
// §8.8's stated test, and it rules out the NATURAL WRONG IMPLEMENTATION: a
// per-tree seed derived from the tree count looks reproducible until
// n_estimators changes. Each tree draws from one stream in tree order, so the
// first ten node arrays must be bit-identical between the two fits.
def fitN($n) {
    np.seed(77);
    $m = ens.RandomForestRegressor({"n_estimators": $n, "max_depth": 5});
    $m.fit($fX, $fY);
    return $m;
}
$f10 = fitN(10);
$f20 = fitN(20);
eq($f20.n_trees(), 20, "the second fit really grew twenty");
$same = 0;
$i = 0;
while ($i < 10) {
    if (nd_tobytes($f10.nodes($i, null), "f64") == nd_tobytes($f20.nodes($i, null), "f64")) {
        $same = $same + 1;
    }
    $i = $i + 1;
}
eq($same, 10, "all ten of the first fit's trees appear unchanged in the second");
// And the extra ten are NOT copies of the first ten, or the stream is not
// advancing and the assertion above is vacuous.
ok(nd_tobytes($f20.nodes(10, null), "f64") != nd_tobytes($f20.nodes(0, null), "f64"),
   "while tree 10 differs from tree 0, so the stream really advances");
} catch ($e) { section_threw("C4: 20 gives the same first 10", $e); }

print("");
print("== bootstrap and max_features are what separate the trees ==");
try {
// With bootstrap off and every feature available, a forest draws NO randomness
// at all -- so every tree sees identical data and must be the identical tree.
// That is the cleanest possible proof that the two switches are the only
// source of variation, and it needs no seeding to hold.
$det = ens.RandomForestRegressor({"n_estimators": 5, "max_depth": 5,
                                  "bootstrap": false, "max_features": "all"});
$det.fit($fX, $fY);
$allSame = true;
$i = 1;
while ($i < 5) {
    if (nd_tobytes($det.nodes($i, null), "f64") != nd_tobytes($det.nodes(0, null), "f64")) {
        $allSame = false;
    }
    $i = $i + 1;
}
ok($allSame,
   "no bootstrap and no column sampling -> five IDENTICAL trees, bit for bit");
// Turn either switch on and they must diverge.
np.seed(31);
$bs = ens.RandomForestRegressor({"n_estimators": 5, "max_depth": 5,
                                 "bootstrap": true, "max_features": "all"});
$bs.fit($fX, $fY);
ok(nd_tobytes($bs.nodes(1, null), "f64") != nd_tobytes($bs.nodes(0, null), "f64"),
   "bootstrap ALONE makes them differ");
np.seed(31);
$cs = ens.RandomForestRegressor({"n_estimators": 5, "max_depth": 5,
                                 "bootstrap": false, "max_features": 2});
$cs.fit($fX, $fY);
ok(nd_tobytes($cs.nodes(1, null), "f64") != nd_tobytes($cs.nodes(0, null), "f64"),
   "and max_features ALONE does too");
// max_features = 1 means each SPLIT considers one feature, redrawn at every
// node -- which is the random forest's defining behaviour (Breiman 2001) and
// NOT the same as giving each tree one feature. The two are distinguishable on
// the node arrays, deterministically and without depending on which feature
// wins: per-node sampling makes a single tree reference MANY features, while
// per-tree sampling confines it to one.
//
// This file first asserted the per-tree version, and asserted it successfully,
// because the implementation was per-tree too. What exposed it was not this
// assertion but the two accuracy gates: with 6 features and k = 2, two thirds
// of the trees never saw an informative feature, and R² came out 0.66 where it
// should be 0.995 ([K63]).
np.seed(5);
$one = ens.RandomForestRegressor({"n_estimators": 8, "max_depth": 4,
                                  "bootstrap": false, "max_features": 1});
$one.fit($fX, $fY);
$maxDistinct = 0;
$t = 0;
while ($t < 8) {
    // Column 1 of the node array is the split feature; a LEAF stores 0 there as
    // filler, so a tree confined to one real feature shows at most two distinct
    // values and a tree that ignored the mask would show several.
    $nd8   = $one.nodes($t, null);
    $feats = np.reshape(np.take($nd8, np.astype(nd([1.0], "f64"), "i64"), 1),
                        [np.shape($nd8)[0]]);
    $d = np.size(np.unique($feats));
    if ($d > $maxDistinct) { $maxDistinct = $d; }
    $t = $t + 1;
}
ok($maxDistinct > 2,
   "with max_features = 1 a SINGLE tree still references several features ("
   + str($maxDistinct) + "), because the draw is per NODE -- per-tree sampling "
   + "would pin it at two (one real feature plus the leaves' filler zero)");
// And it is genuinely constrained: the same forest with every feature available
// at every split must differ, or max_features is being ignored.
np.seed(5);
$all6 = ens.RandomForestRegressor({"n_estimators": 8, "max_depth": 4,
                                   "bootstrap": false, "max_features": "all"});
$all6.fit($fX, $fY);
ok(bits($one.nodes(0, null)) != bits($all6.nodes(0, null)),
   "and max_features = 1 really does constrain the scan: the tree differs from "
   + "the one grown with every feature available");
// _nFeatures as a unit. The names are the whole surface a caller touches, and
// nothing above pins what they MAP TO -- every structural gate passes an
// explicit count, so "sqrt" resolving to f would go unnoticed.
eq(ens._nFeatures("sqrt", 16, "t"), 4,   "sqrt of 16 features is 4");
eq(ens._nFeatures("log2", 16, "t"), 4,   "log2 of 16 features is 4");
eq(ens._nFeatures("all",  16, "t"), 16,  "\"all\" is every feature");
eq(ens._nFeatures(null,   16, "t"), 16,  "and so is null");
eq(ens._nFeatures(0.5,    16, "t"), 8,   "a fraction is a fraction of them");
eq(ens._nFeatures(3,      16, "t"), 3,   "a whole count is itself");
eq(ens._nFeatures("sqrt", 2,  "t"), 1,   "and it never rounds down to zero");
raises(def() { return ens._nFeatures("cbrt", 16, "t"); }, "an unknown name raises");
raises(def() { return ens._nFeatures(17, 16, "t"); },     "a count past f raises");
raises(def() { return ens._nFeatures(0, 16, "t"); },      "a count of zero raises");

// Out-of-bag coverage. A bootstrap sample of n draws leaves about 1/e of the
// rows out, so ONE tree should cover about 63.2% of them -- a number worth
// asserting against rather than trusting a counter nobody checked.
np.seed(67);
$oob1 = ens.RandomForestRegressor({"n_estimators": 1, "max_depth": 3});
$oob1.fit($fX, $fY);
print("        one bootstrap sample covered " + str($oob1.n_in_bag()) + " of 300 rows");
ok($oob1.n_in_bag() > 165 && $oob1.n_in_bag() < 210,
   "one tree covers roughly 1 - 1/e of the rows (expected ~190)");
$oob9 = ens.RandomForestRegressor({"n_estimators": 9, "max_depth": 3});
$oob9.fit($fX, $fY);
ok($oob9.n_in_bag() > $oob1.n_in_bag(), "and nine trees cover more than one");
ok($oob9.n_in_bag() <= 300, "but never more rows than exist");
$noBag = ens.RandomForestRegressor({"n_estimators": 2, "max_depth": 3,
                                    "bootstrap": false});
$noBag.fit($fX, $fY);
eq($noBag.n_in_bag(), null,
   "and with bootstrap off there is no out-of-bag set to report, so it says so "
   + "rather than returning every row");
} catch ($e) { section_threw("bootstrap and max_features separate the trees", $e); }

print("");
print("== the forest averages in PROBABILITY space ==");
try {
np.seed(19);
$cX = np.random([240, 5], 0.0, 1.0);
$cY = np.astype(np.floor(np.multiply(np.reshape(np.take($cX,
          np.astype(nd([0.0], "f64"), "i64"), 1), [240]), 3.0, null), null), "f64");
$rfc = ens.RandomForestClassifier({"n_estimators": 25, "max_depth": 6});
$rfc.fit($cX, $cY);
eq($rfc.n_outputs(), 3, "three classes, three columns per tree");
$P = $rfc.predict_proba($cX);
eq(np.shape($P), [240, 3], "predict_proba is [n, K]");
ok(mae(np.sum($P, 1, false), np.ones([240], "f64")) < 1.0e-12,
   "every row sums to 1");
ok(sc(np.min($P, null, false)) >= 0.0, "and no probability is negative");
ok(bits(np.astype($rfc.predict($cX), "f64"))
   == bits(np.astype(np.take($rfc.classes(),
                             np.astype(np.argmax($P, 1, false), "i64"), 0), "f64")),
   "predict IS the argmax of predict_proba -- not a separate majority vote");
ok($rfc.score($cX, $cY) > 0.9,
   "and it classifies well (" + str($rfc.score($cX, $cY)) + ")");
// The same exact reconstruction the regressor gets, and it is needed for the
// same reason: thresholding each tree's probability at 0.5 before averaging
// turns the forest into a MAJORITY VOTE -- the one thing §8.8 says not to do --
// and leaves "rows sum to 1" and "it classifies well" both green, because
// normalisation hides the change and easy data survives it. Averaging in
// probability space is only checkable by reproducing the average.
$rcCols = [];
$i = 0;
while ($i < 3) {
    $acc = np.zeros([240], "f64");
    $t = 0;
    while ($t < $rfc.n_trees()) {
        $acc = np.add($acc, tree_predict($rfc.nodes($t, $i), $cX), null);
        $t = $t + 1;
    }
    append($rcCols, np.reshape(np.clip(np.divide($acc, 1.0 * $rfc.n_trees(), null),
                                       0.0, 1.0, null), [240, 1]));
    $i = $i + 1;
}
$rcP = np.concat($rcCols, 1);
$rcS = np.sum($rcP, 1, true);
$rcS = np.where(np.equal($rcS, 0.0, null), 1.0, $rcS);
ok(bits(np.divide($rcP, $rcS, null)) == bits($P),
   "predict_proba IS the mean of the trees' leaf PROBABILITIES, bit for bit -- "
   + "not a vote over thresholded labels");
} catch ($e) { section_threw("the forest averages in probability space", $e); }

print("");
print("== a forest beats one deep tree on held-out data ==");
try {
// The whole point of bagging: variance reduction. Asserted on HELD-OUT data,
// because on training data a deep single tree wins by memorising.
np.seed(43);
$trX = np.random([250, 5], 0.0, 1.0);
$trY = np.add(np.multiply(np.reshape(np.take($trX, np.astype(nd([0.0], "f64"), "i64"), 1),
                                     [250]), 3.0, null),
              np.random([250], 0.0, 0.8), null);
$teX = np.random([250, 5], 0.0, 1.0);
$teY = np.add(np.multiply(np.reshape(np.take($teX, np.astype(nd([0.0], "f64"), "i64"), 1),
                                     [250]), 3.0, null),
              np.random([250], 0.0, 0.8), null);
$single = tree.DecisionTreeRegressor(12, 1, 256, 0.0, 0.0, 0.0);
$single.fit($trX, $trY);
$forest = ens.RandomForestRegressor({"n_estimators": 40, "max_depth": 12});
$forest.fit($trX, $trY);
$eT = mse($single.predict($teX), $teY);
$eF = mse($forest.predict($teX), $teY);
print("        held-out MSE: one deep tree " + str($eT) + ", forest " + str($eF));
ok($eF < $eT, "the forest generalises better, which is what bagging is for");
} catch ($e) { section_threw("a forest beats one deep tree", $e); }

print("");
print("== what the forest refuses ==");
try {
raises(def() { return ens.RandomForestRegressor({"n_estimater": 10}); },
       "a mistyped key raises");
raises(def() { return ens.RandomForestRegressor({"n_estimators": 0}); },
       "n_estimators 0 is refused");
raises(def() { return ens.RandomForestRegressor({"bootstrap": "yes"}); },
       "a non-boolean bootstrap is refused");
raises(def() {
    $m = ens.RandomForestRegressor({"max_features": "cbrt"});
    return $m.fit($fX, $fY);
}, "an unknown max_features name is refused");
raises(def() {
    $m = ens.RandomForestRegressor({"max_features": 99});
    return $m.fit($fX, $fY);
}, "a max_features count larger than the feature count is refused");
raises(def() {
    $m = ens.RandomForestClassifier(null);
    return $m.fit($fX, np.ones([300], "f64"));
}, "a single-class target is refused");
raises(def() { return ens.RandomForestRegressor(null).predict($fX); },
       "predict before fit is refused");
raises(def() {
    $m = ens.RandomForestRegressor({"n_estimators": 3});
    $m.fit($fX, $fY);
    return $m.nodes(3, null);
}, "a tree index past the end is refused");
} catch ($e) { section_threw("what the forest refuses", $e); }


print("");
print("== the HESSIAN is used, and its value is hand-derivable ==");
try {
// Dropping the logistic hessian to 1 left the whole suite green: the model is
// then gradient descent rather than Newton, and on separable data it still
// converges. So the gate has to pin the leaf VALUE, which the hessian divides.
//
// One round, depth 1, lambda = 0, min_child_weight = 0, and a split nobody has
// to guess: x <= 4.
//   y01 = [0,0,0,0,1,1,1,1]   ->  ybar = 0.5,  F0 = logit(0.5) = 0
//   p = sigma(0) = 0.5        ->  h = p(1-p) = 0.25 for every row
//   left  (4 rows, y=0): sum g = 4(0.5-0) = +2, sum h = 1  ->  leaf = -2
//   right (4 rows, y=1): sum g = 4(0.5-1) = -2, sum h = 1  ->  leaf = +2
// With h = 1 the denominators would be 4, and the leaves +-0.5.
$hX  = np.reshape(np.astype(np.arange(1.0, 9.0, 1.0), "f64"), [8, 1]);
$hY  = nd([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], "f64");
$hC  = ens.GradientBoostingClassifier({"n_estimators": 1, "learning_rate": 1.0,
                                       "max_depth": 1, "reg_lambda": 0.0,
                                       "min_child_weight": 0.0});
$hC.fit($hX, $hY);
$hN = $hC.nodes(0, 0);
eq(np.shape($hN), [3, 7], "one depth-1 tree is a root and two leaves");
$hL = np.get($hN, [1, 5]);
$hR = np.get($hN, [2, 5]);
print("        logistic leaves: " + str($hL) + ", " + str($hR));
close($hL, 0.0 - 2.0, 1.0e-12, "the left leaf is -G/H = -2 exactly, so h = p(1-p)");
close($hR, 2.0, 1.0e-12, "and the right leaf is +2");

// Poisson, the same construction:
//   y = [1,1,1,1,3,3,3,3]  ->  ybar = 2,  F0 = log 2,  p = e^F0 = 2
//   h = e^F = 2 for every row
//   left  (y=1): sum g = 4(2-1) = +4, sum h = 8  ->  leaf = -0.5
//   right (y=3): sum g = 4(2-3) = -4, sum h = 8  ->  leaf = +0.5
// With h = 1 the leaves would be -+1.
$pY = nd([1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0], "f64");
$pR = ens.GradientBoostingRegressor({"n_estimators": 1, "learning_rate": 1.0,
                                     "max_depth": 1, "reg_lambda": 0.0,
                                     "min_child_weight": 0.0,
                                     "objective": "poisson"});
$pR.fit($hX, $pY);
$pN = $pR.nodes(0, null);
print("        poisson leaves: " + str(np.get($pN, [1, 5])) + ", "
      + str(np.get($pN, [2, 5])));
close(np.get($pN, [1, 5]), 0.0 - 0.5, 1.0e-12, "poisson's left leaf is -0.5, so h = e^F");
close(np.get($pN, [2, 5]), 0.5, 1.0e-12, "and its right leaf is +0.5");
} catch ($e) { section_threw("the hessian is used", $e); }

print("");
print("== softmax is normalised JOINTLY, and 1/K proves it ==");
try {
// Replacing the joint softmax with a per-column sigmoid left the suite green,
// because predict_proba recomputes the softmax and so hides a wrong TRAINING
// gradient. The gate is that F0 = 0 for every class makes p_k exactly 1/K --
// a number a sigmoid cannot produce for K != 2.
//   K = 3, 9 rows, three per class, split at x <= 3 for the class-0 tree
//   y_0 = [1,1,1,0,0,0,0,0,0],  p = 1/3,  h = (1/3)(2/3) = 2/9
//   left  (3 rows, y=1): sum g = 3(1/3 - 1) = -2,   sum h = 2/3  ->  leaf = 3
//   right (6 rows, y=0): sum g = 6(1/3)    = +2,    sum h = 4/3  ->  leaf = -1.5
// A per-column sigmoid (p = 0.5, h = 0.25) would give +2 and -2.
$sX = np.reshape(np.astype(np.arange(1.0, 10.0, 1.0), "f64"), [9, 1]);
$sY = nd([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0], "f64");
$sC = ens.GradientBoostingClassifier({"n_estimators": 1, "learning_rate": 1.0,
                                      "max_depth": 1, "reg_lambda": 0.0,
                                      "min_child_weight": 0.0});
$sC.fit($sX, $sY);
eq($sC.n_outputs(), 3, "three classes, three trees in round 0");
$sN = $sC.nodes(0, 0);
print("        class-0 leaves: " + str(np.get($sN, [1, 5])) + ", "
      + str(np.get($sN, [2, 5])));
close(np.get($sN, [1, 5]), 3.0, 1.0e-12,
      "the class-0 left leaf is 3, which needs p = 1/3 and h = 2/9");
close(np.get($sN, [2, 5]), 0.0 - 1.5, 1.0e-12, "and the right leaf is -1.5");
} catch ($e) { section_threw("softmax is normalised jointly", $e); }

print("");
print("== a zero hessian is left to the guards, NOT clamped to 1e-16 ==");
try {
// [K58]. This file's first version clamped h to 1e-16 "to keep the Newton step
// finite", and the measurement says that is the opposite of safe:
//   h = 1e-16, lambda = 0, min_child_weight = 0  ->  splits, leaves of +-5e15
//   h = 0,     lambda = 0, min_child_weight = 0  ->  one leaf, value 0
// A saturated logistic driven hard with lambda = 0 is exactly that
// configuration. So the model must stay BOUNDED rather than merely finite.
$zX = np.reshape(np.astype(np.arange(1.0, 41.0, 1.0), "f64"), [40, 1]);
$zY = np.astype(np.greater(np.reshape($zX, [40]), 20.0, null), "f64");
$zC = ens.GradientBoostingClassifier({"n_estimators": 150, "learning_rate": 1.0,
                                      "max_depth": 2, "reg_lambda": 0.0,
                                      "min_child_weight": 0.0});
$zC.fit($zX, $zY);
$zP = $zC.predict_proba($zX);
$zMaxLeaf = 0.0;
$j = 0;
while ($j < $zC.n_rounds()) {
    $nn = $zC.nodes($j, 0);
    $m = sc(np.max(np.abs(np.take($nn, np.astype(nd([5.0], "f64"), "i64"), 1),
                          null), null, false));
    if ($m > $zMaxLeaf) { $zMaxLeaf = $m; }
    $j = $j + 1;
}
print("        largest leaf value over 150 saturated rounds: " + str($zMaxLeaf));
ok($zMaxLeaf < 1.0e6, "no leaf runs away: every one stays under 1e6");
ok(sc(np.sum(np.astype(np.isnan($zP, null), "f64"), null, false)) == 0.0,
   "and no probability is NaN after saturation");
eq(sc(np.mean(np.astype(np.equal($zC.predict($zX), $zY, null), "f64"), null, false)), 1,
   "while still classifying the separable target perfectly");
// THE ABOVE IS NOT THE GATE on [K58], and saying so matters: reintroducing the
// 1e-16 clamp leaves all three of those assertions green. On a saturated
// logistic g -> 0 as fast as h does, so the quotient stays small and the clamp
// looks harmless; the ±5e15 case needs g to survive while h vanishes. Rather
// than hunt for a fixture that reaches it through two guards and a scan, the
// contract is pinned on the FUNCTION, which is where the decision lives.
$vhIn  = np.concat([np.divide(np.zeros([1], "f64"), np.zeros([1], "f64"), null),
                    nd([0.0 - 2.0, 0.0, 1.0e-20, 0.25], "f64")], 0);
$vhOut = ens._validH($vhIn);
eq(np.to_list($vhOut), [0, 0, 0, 1e-20, 0.25],
   "_validH maps NaN and negatives to 0, and PASSES A ZERO THROUGH");
eq(np.get($vhOut, [2]), 0,
   "a zero hessian is NOT lifted to 1e-16 -- that lift turned a split the "
   + "guards had refused into a leaf of five quadrillion (K58)");
eq(np.get($vhOut, [3]), 1e-20,
   "and a genuinely tiny hessian is left alone, so the clamp is not a floor");
ok(sc(np.min($vhOut, null, false)) >= 0.0,
   "nothing invalid survives: no negative, no NaN");
} catch ($e) { section_threw("a zero hessian is left to the guards", $e); }

print("");
print("== the classifier early-stops on LOG LOSS, not on accuracy ==");
try {
// E16's home. Nothing early-stopped a classifier, so a flipped sign in the log
// loss -- which would make stopping pick the WORST round -- was invisible.
np.seed(17);
$cX = np.random([300, 5], 0.0, 1.0);
$cY = np.astype(np.greater(np.add(np.reshape(np.take($cX, np.astype(nd([0.0], "f64"), "i64"), 1),
                                            [300]),
                                  np.random([300], 0.0, 0.6), null),
                           0.6, null), "f64");
$cE = ens.GradientBoostingClassifier({"n_estimators": 120, "n_iter_no_change": 6,
                                      "learning_rate": 0.1, "max_depth": 3,
                                      "validation_fraction": 0.25});
$cE.fit($cX, $cY);
$cv = $cE.val_scores();
print("        rounds " + str($cE.n_rounds()) + ", best " + str($cE.best_round())
      + ", first loss " + str($cv[0]) + ", best loss "
      + str($cv[$cE.best_round() - 1]));
ok(sc(np.min(np.array($cv, "f64"), null, false)) >= 0.0,
   "every held-out log loss is >= 0 -- a flipped sign would make them negative");
$cbi = 0;
$cbv = $cv[0];
$i = 1;
while ($i < len($cv)) {
    if ($cv[$i] < $cbv - 0.0001) { $cbv = $cv[$i]; $cbi = $i; }
    $i = $i + 1;
}
eq($cE.best_round(), $cbi + 1,
   "best_round is the last tol-significant improvement in the LOG LOSS");
ok($cv[$cE.best_round() - 1] < $cv[0],
   "and the held-out loss really improved over round 1");
// Accuracy is a step function of the scores: it plateaus while the model is
// still improving, so stopping on it would cut a run that had not finished.
// This asserts the criterion is the loss by showing it still moves where
// accuracy would not.
$accSame = 0;
$i = 1;
while ($i < len($cv)) {
    if ($cv[$i] != $cv[$i - 1]) { $accSame = $accSame + 1; }
    $i = $i + 1;
}
eq($accSame, len($cv) - 1,
   "the criterion changes on EVERY round, which accuracy would not");
} catch ($e) { section_threw("the classifier early-stops on log loss", $e); }


print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
