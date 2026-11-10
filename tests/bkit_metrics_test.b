// ════════════════════════════════════════════════════════════════════════════
//  bkit_metrics_test.b — tiers 1, 2 and 4 for bkit phase K4.
//
//  Tier 2 is the whole point of this file. A metric is a number nobody checks
//  by eye, so every reference value here is derived INDEPENDENTLY of the
//  implementation:
//
//    * the 3-class confusion matrix is written out cell by cell from the two
//      label lists;
//    * the tie-handled ROC AUC is cross-checked against the Mann-Whitney
//      statistic -- counting the 4 positive/negative pairs by hand, with a tie
//      counting 1/2 -- which is a completely different route to the same
//      number;
//    * `r2_score` and `explained_variance_score` use scikit-learn's own
//      documented example, whose published answers are 0.948... and 0.957...;
//    * NDCG uses the standard worked example whose published answer is
//      0.948...;
//    * `log_loss` on a clipped certainty is checked against -ln(1e-15).
//
//  THE FOUR GATES, from the roadmap:
//    * a zero-support class does not divide by zero, and the macro average
//      stays a finite number computed FROM the zeros rather than around them;
//    * ties in the score are handled, asserted against a hand case -- and
//      asserted to make the AUC INDEPENDENT OF INPUT ORDER, which is the
//      property tie handling actually buys;
//    * `log_loss` is clipped, so a confident mistake is finite;
//    * `r2_score` on a constant target raises rather than returning 0/0.
//
//  Run:  bantu run tests/bkit_metrics_test.b
// ════════════════════════════════════════════════════════════════════════════

include "../numba/numba.b"   as np;
include "../bkit/metrics.b"  as mt;
// btorch, for the C5 differential check at the end of the file. bkit does NOT
// depend on btorch -- this is the test reaching across to compare two
// independent implementations of the same metric (K25).
include "../btorch/btorch.b"   as bt;
include "../btorch/datasets.b" as ds;

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
// Never assert on a stringified number for a VALUE -- str() gives six
// significant digits (btorch rule 1). Values go through `close`.
def close($got, $want, $tol, $n) {
    $d = $got - $want;
    if ($d < 0.0) { $d = 0.0 - $d; }
    if ($d <= $tol) { $R.pass = $R.pass + 1; print("  ok    " + $n); }
    else { $R.fail = $R.fail + 1;
           print("  FAIL  " + $n + "   got " + str($got) + ", want " + str($want)); }
    return null;
}
def lst($a, $want, $n) { eqs(str(nd_to_list($a)), str($want), $n); return null; }
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
// There is no scalar isNaN in Bantu, and `$x != $x` is the usual trick; the
// array route is used instead because nd_isnan is the kernel the rest of the
// package trusts for this.
def notNaN($x) { return !nd_get(nd_isnan(nd([$x], "f64"), null), 0); }

def safe_ok($fn, $n) {
    $v = false;  $err = "";
    try { $v = $fn(); } catch ($e) { $err = str($e); }
    if ($err != "") { $R.fail = $R.fail + 1;
                      print("  FAIL  " + $n + "   raised: " + $err); return null; }
    ok($v, $n);
    return null;
}

print("");
print("  bkit metrics (K4)");
print("  ════════════════════════════════════════");

// ── 1. the confusion matrix, cell by cell ───────────────────────────────────
print("");
print("-- confusion_matrix: rows are TRUE, columns are PREDICTED --");
// true: 0 0 0 1 1 1 2 2      predicted: 0 0 1 1 1 2 2 0
//   true 0 -> predicted 0,0,1   => row 0 = [2, 1, 0]
//   true 1 -> predicted 1,1,2   => row 1 = [0, 2, 1]
//   true 2 -> predicted 2,0     => row 2 = [1, 0, 1]
$yt = nd([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0], "f64");
$yp = nd([0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.0], "f64");
lst(mt.confusion_matrix($yt, $yp, null), [[2, 1, 0], [0, 2, 1], [1, 0, 1]],
    "the matrix is written out cell by cell from the two label lists");
// THE ORIENTATION MATTERS: transposing it silently swaps precision and recall.
// Cell [0][1] is "true 0, predicted 1", of which there is exactly one.
eqs(nd_get(mt.confusion_matrix($yt, $yp, null), [0, 1]), 1.0,
    "cell [0][1] is \"true 0 predicted 1\" -- one of them, so it is not transposed");
eqs(nd_get(mt.confusion_matrix($yt, $yp, null), [1, 0]), 0.0,
    "and cell [1][0] is \"true 1 predicted 0\" -- none");
eqs(nd_get(nd_sum(mt.confusion_matrix($yt, $yp, null), null, null), 0), 8.0,
    "every sample lands in exactly one cell");
lst(mt.labels_of($yt, $yp), [0, 1, 2], "the classes come from both arrays, sorted");
// an explicit label list in a different order is still sorted, and a value
// outside it is an error rather than a silently dropped row
lst(mt.confusion_matrix($yt, $yp, [2.0, 0.0, 1.0]), [[2, 1, 0], [0, 2, 1], [1, 0, 1]],
    "an explicit labels list is sorted, so the orientation cannot change");
raises_with(def() { return mt.confusion_matrix($yt, $yp, [0.0, 1.0]); }, "no cell",
            "a value missing from the labels list is an error, not a dropped row");
raises_with(def() { return mt.confusion_matrix($yt, nd([0.0], "f64"), null); }, "BROADCAST",
            "a length mismatch is refused, and the message says why it must be");
raises_with(def() { return mt.confusion_matrix(nd_zeros([0], "f64"),
                nd_zeros([0], "f64"), null); }, "no values",
            "an empty input is refused");

// ── 2. precision / recall / F1, from that matrix ────────────────────────────
print("");
print("-- precision, recall, F1: every one a view of the same matrix (K26) --");
// tp = diag = [2,2,1]; column sums = [3,3,2] so fp = [1,1,1];
// row sums = [3,3,2] so fn = [1,1,1]. Hence p = r = [2/3, 2/3, 1/2].
$r = mt.precision_recall_fscore_support($yt, $yp, null);
lst($r.tp, [2, 2, 1], "tp is the diagonal");
lst($r.fp, [1, 1, 1], "fp is the column sum minus the diagonal");
lst($r.fn, [1, 1, 1], "fn is the row sum minus the diagonal");
lst($r.support, [3, 3, 2], "support is the row sum: how many of each class are TRUE");
arrclose($r.precision, [2.0/3.0, 2.0/3.0, 0.5], 1.0e-15, "per-class precision");
arrclose($r.recall, [2.0/3.0, 2.0/3.0, 0.5], 1.0e-15, "per-class recall");
arrclose($r.f1, [2.0/3.0, 2.0/3.0, 0.5], 1.0e-15, "per-class F1");
close(mt.accuracy_score($yt, $yp), 5.0 / 8.0, 1.0e-15, "accuracy is 5 of 8 correct");
// macro is the unweighted mean; weighted is by support; micro is the accuracy
// for single-label multiclass -- three different numbers from one matrix
close(mt.f1_score($yt, $yp, "macro"), (2.0/3.0 + 2.0/3.0 + 0.5) / 3.0, 1.0e-15,
      "macro F1 is the unweighted mean of the three");
close(mt.f1_score($yt, $yp, "weighted"), (3.0*2.0/3.0 + 3.0*2.0/3.0 + 2.0*0.5) / 8.0, 1.0e-15,
      "weighted F1 weights each class by its support");
close(mt.f1_score($yt, $yp, "micro"), 5.0 / 8.0, 1.0e-15,
      "micro F1 equals the accuracy for single-label multiclass");
close(mt.precision_score($yt, $yp, "micro"), 5.0 / 8.0, 1.0e-15, "and so does micro precision");
close(mt.recall_score($yt, $yp, "micro"), 5.0 / 8.0, 1.0e-15, "and micro recall");
// macro and weighted genuinely differ here, so the averaging is being applied
ok(abs(mt.f1_score($yt, $yp, "macro") - mt.f1_score($yt, $yp, "weighted")) > 0.01,
   "macro and weighted differ, so the average argument is not being ignored");
arrclose(mt.precision_score($yt, $yp, null), [2.0/3.0, 2.0/3.0, 0.5], 1.0e-15,
         "average=null gives the per-class array");
raises_with(def() { return mt.f1_score($yt, $yp, "mean"); }, "average must be",
            "an unknown average is refused");

// ── 3. GATE: a zero-support class does not divide by zero ───────────────────
print("");
print("-- GATE: zero support (K27) --");
// Class 1 is TRUE twice and NEVER predicted: tp=0, fp=0, so precision is 0/0.
$a = nd([0.0, 0.0, 1.0, 1.0], "f64");
$b = nd([0.0, 0.0, 0.0, 0.0], "f64");
lst(mt.confusion_matrix($a, $b, null), [[2, 0], [2, 0]], "the matrix of a never-predicted class");
$r2 = mt.precision_recall_fscore_support($a, $b, null);
arrclose($r2.precision, [0.5, 0.0], 1.0e-15, "its precision is 0, not NaN");
arrclose($r2.recall, [1.0, 0.0], 1.0e-15, "its recall is 0, not NaN");
arrclose($r2.f1, [2.0/3.0, 0.0], 1.0e-15, "its F1 is 0, not NaN");
ok(!nd_get(nd_any(nd_isnan($r2.precision, null), null, null), 0), "no NaN anywhere in precision");
ok(!nd_get(nd_any(nd_isnan($r2.f1, null), null, null), 0), "nor in F1");
// AND THE MACRO AVERAGE SURVIVES. A NaN in one class would make the whole
// model's macro score NaN, which is the silent-wrong-answer this rule avoids.
close(mt.f1_score($a, $b, "macro"), (2.0/3.0) / 2.0, 1.0e-15,
      "the macro average is finite, computed FROM the zero rather than around it");
// the mirror case: a class predicted but never true, so recall is 0/0
$c1 = nd([0.0, 0.0, 0.0, 0.0], "f64");
$c2 = nd([0.0, 0.0, 1.0, 1.0], "f64");
$r3 = mt.precision_recall_fscore_support($c1, $c2, null);
arrclose($r3.recall, [0.5, 0.0], 1.0e-15, "a class predicted but never true has recall 0");
arrclose($r3.precision, [1.0, 0.0], 1.0e-15, "  and precision 0 -- the other 0/0");
lst($r3.support, [4, 0], "  with zero support, which is what makes it visible");
// a perfect classifier, as the boundary in the other direction
close(mt.f1_score($yt, $yt, "macro"), 1.0, 1.0e-15, "a perfect prediction scores macro F1 of 1");

// ── 4. GATE: ties in the score ──────────────────────────────────────────────
print("");
print("-- GATE: tied scores give one curve point, not two (K28) --");
// scores [0.8, 0.8, 0.3, 0.1] with labels [1, 0, 1, 0].
// At threshold 0.8 BOTH tied samples are included: tp=1, fp=1. A curve that
// advanced one sample at a time would emit (0, 0.5) and (0.5, 0.5) separately,
// through a point no threshold can produce.
$t3 = nd([1.0, 0.0, 1.0, 0.0], "f64");
$s3 = nd([0.8, 0.8, 0.3, 0.1], "f64");
$cv = mt.roc_curve($t3, $s3);
lst($cv.fpr, [0, 0.5, 0.5, 1], "fpr has FOUR points for four samples: (0,0) plus three thresholds");
lst($cv.tpr, [0, 0.5, 1, 1], "and tpr matches");
arrclose($cv.thresholds, [1.8, 0.8, 0.3, 0.1], 1.0e-15,
         "the thresholds are the three DISTINCT scores, plus one above them all");
// THE NUMBER, cross-checked by a completely different route. Mann-Whitney over
// the 2x2 = 4 positive/negative pairs, a tie counting 1/2:
//   pos 0.8 vs neg 0.8  -> tie  -> 0.5
//   pos 0.8 vs neg 0.1  -> win  -> 1
//   pos 0.3 vs neg 0.8  -> loss -> 0
//   pos 0.3 vs neg 0.1  -> win  -> 1
//   total 2.5 of 4 = 0.625
close(mt.roc_auc_score($t3, $s3), 0.625, 1.0e-15,
      "the AUC is 0.625, which the Mann-Whitney count over the four pairs confirms");
// AND THE PROPERTY TIE HANDLING ACTUALLY BUYS: the answer does not depend on
// the order the equal scores arrived in. Without it the staircase's area would
// change when the input was shuffled.
$t3b = nd([0.0, 1.0, 1.0, 0.0], "f64");
$s3b = nd([0.8, 0.8, 0.3, 0.1], "f64");
close(mt.roc_auc_score($t3b, $s3b), 0.625, 1.0e-15,
      "swapping the two tied samples leaves the AUC IDENTICAL");
// a perfect ranking and a reversed one
close(mt.roc_auc_score(nd([0.0, 0.0, 1.0, 1.0], "f64"), nd([0.1, 0.2, 0.8, 0.9], "f64")),
      1.0, 1.0e-15, "a perfect ranking scores 1");
close(mt.roc_auc_score(nd([1.0, 1.0, 0.0, 0.0], "f64"), nd([0.1, 0.2, 0.8, 0.9], "f64")),
      0.0, 1.0e-15, "a perfectly reversed one scores 0");
// all scores equal: every ranking is as good as any other, so 0.5
close(mt.roc_auc_score(nd([1.0, 0.0, 1.0, 0.0], "f64"), nd([0.5, 0.5, 0.5, 0.5], "f64")),
      0.5, 1.0e-15, "all-equal scores give exactly 0.5 -- one point, no staircase");
raises_with(def() { return mt.roc_auc_score(nd([1.0, 1.0], "f64"), nd([0.3, 0.7], "f64")); },
            "only one class", "a single-class y_true is refused: there is nothing to rank against");
raises_with(def() { return mt.roc_curve(nd([0.0, 2.0], "f64"), nd([0.3, 0.7], "f64")); },
            "must be 0/1", "a non-binary y_true is refused");

print("");
print("-- average_precision is the STEP SUM, not the trapezoid (K28) --");
// precision at the three thresholds = tp/(tp+fp) = [1/2, 2/3, 2/4]
// recall                           = tp/nPos    = [1/2, 1, 1]
// AP = sum (R_n - R_{n-1}) * P_n = 0.5*0.5 + 0.5*(2/3) + 0*0.5 = 0.5833...
close(mt.average_precision_score($t3, $s3), 0.5 * 0.5 + 0.5 * (2.0 / 3.0), 1.0e-15,
      "AP is the step sum over the same three thresholds");
ok(abs(mt.average_precision_score($t3, $s3) - mt.roc_auc_score($t3, $s3)) > 0.04,
   "and it is a DIFFERENT number from the ROC AUC on the same data, as it should be");
$pr = mt.precision_recall_curve($t3, $s3);
arrclose($pr.precision, [0.5, 2.0 / 3.0, 0.5, 1.0], 1.0e-15,
         "the PR curve ends at precision 1, recall 0 -- the limit above every score");
arrclose($pr.recall, [0.5, 1.0, 1.0, 0.0], 1.0e-15, "  and its recall");
eqs(nd_size($pr.thresholds), 3,
    "thresholds is one shorter than the curves, because that last point has none");
close(mt.average_precision_score(nd([0.0, 0.0, 1.0, 1.0], "f64"),
                                 nd([0.1, 0.2, 0.8, 0.9], "f64")), 1.0, 1.0e-15,
      "a perfect ranking has average precision 1");

// ── 5. GATE: log_loss is clipped ────────────────────────────────────────────
print("");
print("-- GATE: a confident mistake is finite (§5.5) --");
// -ln(0.9) for both samples
close(mt.log_loss(nd([1.0, 0.0], "f64"), nd([0.9, 0.1], "f64"), null),
      0.0 - log(0.9), 1.0e-15, "log_loss of two 0.9-confident correct calls is -ln(0.9)");
// p = 0 for a positive label: clipped to eps, so -ln(1e-15) = 34.538776...
$conf = mt.log_loss(nd([1.0], "f64"), nd([0.0], "f64"), null);
close($conf, 0.0 - log(1.0e-15), 1.0e-9,
      "a probability of exactly 0 for a positive label gives -ln(eps), not inf");
ok($conf < 100.0, "  which is finite, so the rest of an evaluation survives it");
ok(notNaN($conf), "  and not NaN");
// a looser eps gives a smaller penalty, which proves the clip is the eps given
ok(mt.log_loss(nd([1.0], "f64"), nd([0.0], "f64"), 1.0e-3) < $conf,
   "a larger eps clips harder, so the penalty is smaller");
close(mt.log_loss(nd([1.0], "f64"), nd([0.0], "f64"), 1.0e-3), 0.0 - log(1.0e-3), 1.0e-12,
      "  and equals -ln(eps) exactly");
raises_with(def() { return mt.log_loss(nd([1.0], "f64"), nd([0.5], "f64"), 0.0); }, "eps must be",
            "eps of 0 is refused -- it would reintroduce the infinity");
// multiclass: y_true holds class indices, y_prob has one column per class
$mp = nd([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], "f64");
close(mt.log_loss(nd([0.0, 1.0], "f64"), $mp, null),
      0.0 - (log(0.7) + log(0.8)) / 2.0, 1.0e-14,
      "multiclass log_loss picks the probability of each row's TRUE class");
// rows are normalised first, so a row summing to 2 scores as the distribution
// it implies
close(mt.log_loss(nd([0.0], "f64"), nd([[1.4, 0.4, 0.2]], "f64"), null),
      0.0 - log(0.7), 1.0e-14, "a row that does not sum to 1 is normalised, not punished");
raises_with(def() { return mt.log_loss(nd([5.0, 1.0], "f64"), $mp, null); }, "whole class indices",
            "a label outside the column range is refused");
raises_with(def() { return mt.log_loss(nd([0.0], "f64"),
                nd([[0.0, 0.0, 0.0]], "f64"), null); }, "sums to zero",
            "a row of all zeros is not a distribution, and says so");
close(mt.brier_score_loss(nd([1.0, 0.0], "f64"), nd([0.9, 0.1], "f64")), 0.01, 1.0e-15,
      "brier is the mean squared probability error");
close(mt.brier_score_loss(nd([1.0], "f64"), nd([0.0], "f64")), 1.0, 1.0e-15,
      "and needs no clipping: a confident mistake costs exactly 1");

// ── 6. regression, against scikit-learn's documented example ───────────────
print("");
print("-- regression (scikit-learn's own worked example) --");
$g = nd([3.0, 0.0 - 0.5, 2.0, 7.0], "f64");
$h = nd([2.5, 0.0, 2.0, 8.0], "f64");
// residuals 0.5, -0.5, 0, -1 -> squares 0.25, 0.25, 0, 1
close(mt.mean_squared_error($g, $h), 1.5 / 4.0, 1.0e-15, "MSE is 0.375");
close(mt.root_mean_squared_error($g, $h), sqrt(0.375), 1.0e-15, "RMSE is its square root");
close(mt.mean_absolute_error($g, $h), 2.0 / 4.0, 1.0e-15, "MAE is 0.5");
// published answers for this exact example: 0.948... and 0.957...
close(mt.r2_score($g, $h), 0.9486081370449679, 1.0e-12,
      "R2 is 0.9486081..., scikit-learn's published value for this data");
close(mt.explained_variance_score($g, $h), 0.9571734475374732, 1.0e-12,
      "explained variance is 0.9571734..., likewise");
// the two differ exactly when the residuals have a non-zero mean, which is why
// both are reported
ok(mt.explained_variance_score($g, $h) > mt.r2_score($g, $h),
   "explained variance exceeds R2 here, because the residuals are biased");
$unb = nd([1.0, 2.0, 3.0], "f64");
close(mt.explained_variance_score($unb, $unb), 1.0, 1.0e-15,
      "a perfect prediction gives 1 for both");
close(mt.r2_score($unb, $unb), 1.0, 1.0e-15, "  R2 too");
// predicting the mean gives R2 of exactly 0 -- the definition's own baseline
close(mt.r2_score(nd([1.0, 2.0, 3.0], "f64"), nd([2.0, 2.0, 2.0], "f64")), 0.0, 1.0e-15,
      "predicting the mean scores exactly 0, which is R2's baseline");
close(mt.mean_absolute_percentage_error(nd([1.0, 10.0], "f64"),
                                        nd([1.1, 9.0], "f64"), null), 0.1, 1.0e-15,
      "MAPE averages the relative errors: 10% and 10%");

print("");
print("-- GATE: R2 on a constant target raises (K30) --");
raises_with(def() { return mt.r2_score(nd([5.0, 5.0, 5.0], "f64"),
                                       nd([4.0, 5.0, 6.0], "f64")); }, "constant",
            "a constant y_true raises rather than returning 0/0");
raises_with(def() { return mt.r2_score(nd([5.0, 5.0], "f64"), nd([5.0, 5.0], "f64")); },
            "mean_squared_error",
            "  and the message names the metric to use instead");
raises_with(def() { return mt.explained_variance_score(nd([5.0, 5.0], "f64"),
                                                       nd([4.0, 5.0], "f64")); }, "constant",
            "explained variance does the same");
// MSE and MAE are defined there, which is the point of the message
close(mt.mean_squared_error(nd([5.0, 5.0], "f64"), nd([4.0, 6.0], "f64")), 1.0, 1.0e-15,
      "MSE on a constant target is perfectly well defined");
// MAPE clamps instead of raising: one zero target is a data point
$mz = mt.mean_absolute_percentage_error(nd([0.0, 10.0], "f64"), nd([1.0, 10.0], "f64"), 1.0e-3);
ok($mz < 1.0e6, "MAPE at a zero target is bounded by the eps clamp, not infinite");
ok(notNaN($mz), "  and not NaN");

// ── 7. ranking ──────────────────────────────────────────────────────────────
print("");
print("-- ranking --");
// The standard worked example: relevances [3,2,3,0,1,2] already in score
// order. DCG = 7/1 + 3/log2(3) + 7/2 + 0 + 1/log2(6) + 3/log2(7);
// the ideal order is [3,3,2,2,1,0]. The published NDCG is 0.948...
$rel = nd([3.0, 2.0, 3.0, 0.0, 1.0, 2.0], "f64");
$sco = nd([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], "f64");
// Computed independently rather than remembered: with gain 2^rel - 1 and
// discount log2(i+1), DCG = 13.848263629272981 and IDCG = 14.595390756454924,
// so NDCG = 0.9488107485678985. My first expected value here was 0.9481156,
// which is the published figure for the same relevances under the LINEAR gain
// convention -- a different metric with the same name. The exponential gain is
// scikit-learn's default and the one implemented, so the test states which.
close(mt.ndcg_score($rel, $sco, null), 0.9488107485678985, 1.0e-12,
      "NDCG is 0.94881075..., derived from the DCG and IDCG term by term");
close(mt.ndcg_score($rel, $rel, null), 1.0, 1.0e-15,
      "ranking BY the relevance itself scores exactly 1");
// truncation at k changes the answer, so k is being applied
ok(mt.ndcg_score($rel, $sco, 2) != mt.ndcg_score($rel, $sco, null),
   "ndcg@2 differs from the full NDCG, so k is applied");
close(mt.ndcg_score(nd([0.0, 0.0], "f64"), nd([1.0, 2.0], "f64"), null), 0.0, 1.0e-15,
      "a query with nothing relevant scores 0 rather than dividing by zero");
// MRR: ranked by score the items are in input order, and the first relevant
// one is at position 2, so 1/2
close(mt.mrr_score(nd([0.0, 1.0, 0.0], "f64"), nd([3.0, 2.0, 1.0], "f64")), 0.5, 1.0e-15,
      "MRR is 1/2 when the first relevant item ranks second");
close(mt.mrr_score(nd([1.0, 0.0, 0.0], "f64"), nd([3.0, 2.0, 1.0], "f64")), 1.0, 1.0e-15,
      "and 1 when it ranks first");
close(mt.mrr_score(nd([0.0, 0.0], "f64"), nd([2.0, 1.0], "f64")), 0.0, 1.0e-15,
      "a query with nothing relevant contributes 0, not a skipped denominator");
// a batch of queries is the mean over rows
close(mt.mrr_score(nd([[1.0, 0.0], [0.0, 1.0]], "f64"), nd([[2.0, 1.0], [2.0, 1.0]], "f64")),
      (1.0 + 0.5) / 2.0, 1.0e-15, "a 2-query batch averages 1 and 1/2");
raises_with(def() { return mt.ndcg_score(nd([1.0, 2.0], "f64"), nd([1.0], "f64")); }, "must match",
            "mismatched shapes are refused");

// ── 8. classification_report: numbers first, text second (K29) ──────────────
print("");
print("-- classification_report --");
$rep = mt.classification_report($yt, $yp, null);
close($rep["0"].precision, 2.0 / 3.0, 1.0e-15, "the report carries per-class precision");
close($rep["2"].f1, 0.5, 1.0e-15, "  and per-class F1");
eqs($rep["2"].support, 2.0, "  and per-class support");
close($rep.accuracy, 5.0 / 8.0, 1.0e-15, "plus the overall accuracy");
close($rep["macro avg"].f1, (2.0/3.0 + 2.0/3.0 + 0.5) / 3.0, 1.0e-15, "the macro average");
close($rep["weighted avg"].f1, 5.0 / 8.0, 1.0e-15, "and the weighted average");
eqs($rep["macro avg"].support, 8.0, "whose support is the total count");
// the TEXT is checked for shape only: asserting the formatted string would
// make a column-width change a test failure (K29)
$txt = mt.classification_report_text($rep);
ok(indexOf($txt, "precision") >= 0, "the text names its columns");
ok(indexOf($txt, "macro avg") >= 0, "  and the macro average row");
ok(indexOf($txt, "weighted avg") >= 0, "  and the weighted one");
ok(indexOf($txt, "0.667") >= 0, "  and formats to three decimals");
$nl = 0;
$i = 0;
while ($i < len($txt)) {
    if (substr($txt, $i, 1) == "\n") { $nl = $nl + 1; }
    $i = $i + 1;
}
// header + 3 classes + blank + accuracy + macro + weighted = 8 lines, so 7
// separators. (My first count said 6 and forgot that 8 lines have 7 gaps.)
eqs($nl, 7, "and has eight lines: a header, three classes, a blank, and three summaries");
raises_with(def() { return mt.classification_report_text(null); }, "null",
            "formatting a null report says so");

// ── 9. NaN is refused, by name ──────────────────────────────────────────────
print("");
print("-- NaN --");
raises_with(def() { return mt.accuracy_score(nd([1.0, NAN], "f64"), nd([1.0, 1.0], "f64")); },
            "y_true contains NaN", "accuracy refuses NaN in y_true, naming which side");
raises_with(def() { return mt.mean_squared_error(nd([1.0, 2.0], "f64"),
                nd([1.0, NAN], "f64")); }, "y_pred contains NaN",
            "MSE refuses NaN in y_pred, naming which side");
raises_with(def() { return mt.roc_auc_score(nd([1.0, 0.0], "f64"),
                nd([NAN, 0.2], "f64")); }, "NaN", "the ROC curve refuses a NaN score");
raises_with(def() { return mt.r2_score(nd([1.0, NAN], "f64"), nd([1.0, 2.0], "f64")); },
            "NaN", "R2 too");

// ── 10. the input shapes a caller actually passes ───────────────────────────
print("");
print("-- input handling --");
close(mt.accuracy_score([1.0, 0.0, 1.0], [1.0, 0.0, 0.0]), 2.0 / 3.0, 1.0e-15,
      "plain Bantu lists are accepted");
close(mt.accuracy_score(nd([[1.0], [0.0]], "f64"), nd([1.0, 0.0], "f64")), 1.0, 1.0e-15,
      "an (n,1) column and a 1-D array compare correctly -- a model predicts a column");
close(mt.accuracy_score(nd([1, 0, 1], "i64"), nd([1, 0, 0], "i64")), 2.0 / 3.0, 1.0e-15,
      "and i64 labels are coerced");
raises_with(def() { return mt.accuracy_score(null, nd([1.0], "f64")); }, "null",
            "a null argument says so");
raises_with(def() { return mt.accuracy_score(nd_zeros([2, 3], "f64"),
                nd_zeros([2, 3], "f64")); }, "one column",
            "a multi-column input is refused");

// ── 11. tier 4: size ────────────────────────────────────────────────────────
print("");
print("-- stress --");
nd_seed(5150);
$n = 200000;
$bigT = nd_astype(nd_random_int([$n], 0, 10), "f64");
$bigP = nd_astype(nd_random_int([$n], 0, 10), "f64");
$t0 = clock();
$bcm = mt.confusion_matrix($bigT, $bigP, null);
print("        200000 rows, 10 classes, confusion_matrix: " + str(clock() - $t0) + " ms");
eqs(nd_get(nd_sum($bcm, null, null), 0), 1.0 * $n, "every one of 200000 rows lands in a cell");
eqs(str(nd_shape($bcm)), str([10, 10]), "and the matrix is 10x10");
// the matrix and a direct mean agree on the accuracy, which is a check on the
// bincount encoding at a size no hand case can reach
close(nd_get(nd_sum(mt.nd_diagonal_of($bcm), null, null), 0) / (1.0 * $n),
      mt.accuracy_score($bigT, $bigP), 1.0e-12,
      "trace/n equals accuracy_score -- the encoding is right at 200000 rows");
$t1 = clock();
$bs = nd_random_uniform([$n], 0.0, 1.0);
$bl = nd_astype(nd_greater($bs, 0.5, null), "f64");
$auc = mt.roc_auc_score($bl, $bs);
print("        200000-row roc_auc_score: " + str(clock() - $t1) + " ms, auc " + str($auc));
close($auc, 1.0, 1.0e-12, "a threshold-defined label is ranked perfectly, so AUC is 1");
// every score distinct -> as many curve points as samples, plus the origin
eqs(nd_size(mt.roc_curve($bl, $bs).fpr), $n + 1,
    "with distinct scores there is one point per sample, plus (0,0)");
// and all-tied scores collapse to a single point, which is the tie rule at scale
$tied = nd_full([$n], 0.5, "f64");
eqs(nd_size(mt.roc_curve($bl, $tied).fpr), 2,
    "while 200000 TIED scores collapse to one threshold plus the origin");
close(mt.roc_auc_score($bl, $tied), 0.5, 1.0e-12, "  and score exactly 0.5");

// ── 12. C5: the metric moved, and btorch's copy must not drift ──────────────
// Ecosystem C5/E14 say `ds.accuracy` moves here. The move happened; the
// DELEGATION deliberately did not (K25). btorch is 1.0.0 and gated on MNIST
// accuracy, bkit is 0.x and changes every phase, so delegating would let a
// bkit release break a btorch install -- E2's own blast-radius argument.
//
// What stops the two drifting is therefore this test rather than a shared
// implementation. They are not the same function: btorch's takes LOGITS and
// argmaxes them, this one takes labels on both sides.
print("");
print("-- C5: btorch's accuracy and bkit's accuracy_score agree --");
$logits = nd([[2.0, 1.0, 0.0],      // argmax 0
              [0.0, 5.0, 1.0],      // argmax 1
              [1.0, 0.0, 3.0],      // argmax 2
              [0.0, 1.0, 0.5]], "f64");   // argmax 1
$truth  = nd([0.0, 1.0, 1.0, 1.0], "f64");  // 3 of 4 correct
$viaBkit = mt.accuracy_score($truth, nd_astype(nd_argmax($logits, 1, null), "f64"));
close($viaBkit, 3.0 / 4.0, 1.0e-15, "bkit scores the hand case at 3 of 4");
$dsAcc = ds.accuracy(bt.tensor($logits, false, null), $truth);
close($dsAcc, $viaBkit, 1.0e-15,
      "and btorch's ds.accuracy returns the IDENTICAL number");
// over random data too, so the agreement is not a property of one small case
nd_seed(808);
$rl = nd_random_uniform([200, 10], 0.0 - 1.0, 1.0);
$ry = nd_astype(nd_random_int([200], 0, 10), "f64");
close(ds.accuracy(bt.tensor($rl, false, null), $ry),
      mt.accuracy_score($ry, nd_astype(nd_argmax($rl, 1, null), "f64")), 1.0e-15,
      "  and on 200 random rows of 10 classes");
// both refuse a length mismatch, rather than one broadcasting
raises_with(def() { return ds.accuracy(bt.tensor($logits, false, null),
                nd([0.0], "f64")); }, "predictions for",
            "btorch's still refuses a label-count mismatch");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
