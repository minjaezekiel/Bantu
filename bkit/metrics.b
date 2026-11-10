// ════════════════════════════════════════════════════════════════════════════
//  metrics.b — classification, regression and ranking metrics.
//
//      include "bkit/metrics.b" as mt;
//
//      print(mt.accuracy_score($y, $pred));
//      $r = mt.classification_report($y, $pred, null);
//      print(mt.classification_report_text($r));
//
//  Design: docs/bkit-architecture.md §5. Decisions K25-K30.
//
//  THIS FILE DEPENDS ON NOTHING BUT NUMBA -- not even on base.b. Nothing here
//  is an estimator; these are pure functions of two arrays, so the estimator
//  protocol has nothing to say about them. Keeping it standalone is the same
//  decision as K7 for base.b: a user who wants f1_score should not have to
//  import scalers, splitters and a gradient booster to get it.
//
//  FOUR THINGS WORTH KNOWING BEFORE READING:
//
//    * EVERY CLASSIFICATION METRIC IS A VIEW OF ONE CONFUSION MATRIX (K26).
//      precision, recall, f1, support and the report all come from
//      precision_recall_fscore_support, which builds the matrix once. So the
//      ZERO-DIVISION RULE LIVES IN ONE PLACE, and a class that never appears
//      cannot be handled one way by precision and another by f1.
//
//    * A ZERO-SUPPORT CLASS SCORES 0, not NaN (K27). NaN would propagate into
//      the macro average and turn one absent class into a NaN for the whole
//      model. An error was the other option and is too strict: a class missing
//      from one cross-validation fold is normal.
//
//    * THE CURVES EMIT ONE POINT PER DISTINCT THRESHOLD (K28). That is what
//      handling ties means. With scores [0.8, 0.8, 0.8] and labels [1, 0, 1],
//      advancing one sample at a time traces a staircase through points no
//      threshold can produce, and its area depends on the order the equal
//      scores arrived in -- the AUC would change when the input was shuffled.
//
//    * roc_auc_score IS THE TRAPEZOID; average_precision_score IS THE STEP
//      SUM. They are different numbers on the same curve. scikit-learn uses
//      exactly this pair, and a test asserts both against a hand case so
//      neither drifts into the other's rule.
// ════════════════════════════════════════════════════════════════════════════

// ── input handling ──────────────────────────────────────────────────────────

// A 1-D f64 array. Accepts a list, a 1-D array or an (n,1) column, because a
// prediction that came out of a model is a column and a label list typed by
// hand is a list, and refusing either would be a papercut on every call.
def _vec($v, $who) {
    if ($v == null) { throw $who + ": argument is null"; }
    $A = $v;
    if (type($A) == "list")   { $A = nd($A, "f64"); }
    if (type($A) == "number") { $A = nd([$A], "f64"); }
    if (type($A) != "ndarray") {
        throw $who + ": expected an array or a list, got " + type($v);
    }
    if (nd_dtype($A) != "f64") { $A = nd_astype($A, "f64"); }
    if (nd_ndim($A) == 2) {
        if (nd_shape($A)[1] != 1) {
            throw $who + ": expected one column, got " + str(nd_shape($A)[1]);
        }
        return nd_ravel($A);
    }
    if (nd_ndim($A) != 1) {
        throw $who + ": expected a 1-D array, got " + str(nd_ndim($A)) + " dimensions";
    }
    return $A;
}

// Two vectors of the same length, which is checked rather than assumed: a
// length mismatch broadcasts in numba when one side is length 1, and silently
// compares every prediction against a single label.
def _pair($a, $b, $who) {
    $x = _vec($a, $who);
    $y = _vec($b, $who);
    if (nd_shape($x)[0] != nd_shape($y)[0]) {
        throw $who + ": " + str(nd_shape($x)[0]) + " and " + str(nd_shape($y)[0])
              + " values -- these must be the same length (a length-1 side would "
              + "BROADCAST and compare everything against one value)";
    }
    if (nd_shape($x)[0] == 0) { throw $who + ": no values to score"; }
    return [$x, $y];
}

def _no_nan($a, $who, $what) {
    if (nd_get(nd_any(nd_isnan($a, null), null, null), 0)) {
        throw $who + ": " + $what + " contains NaN";
    }
    return null;
}

// Labels -> codes 0..k-1 against a sorted class list. The same searchsorted
// trick preprocess.b uses (K13), written here rather than imported because
// metrics.b depends on nothing (§5.1).
def _codes($v, $classes) {
    $k    = nd_shape($classes)[0];
    $pos  = nd_astype(nd_searchsorted($classes, $v, null), "f64");
    $safe = nd_clip($pos, 0.0, 1.0 * ($k - 1), null);
    return nd_astype($safe, "i64");
}

// x/y element-wise, with 0 where y is 0 (K27). nd_divide would give NaN there,
// and a NaN in one class's precision becomes a NaN for the macro average of
// the whole model.
def _safe_div($x, $y) {
    $d = nd_where(nd_equal($y, 0.0, null), 1.0, $y);
    return nd_where(nd_equal($y, 0.0, null), 0.0, nd_divide($x, $d, null));
}

// ── the confusion matrix ────────────────────────────────────────────────────
// Rows are TRUE classes, columns are PREDICTED -- the scikit-learn and
// textbook orientation. Transposing it silently swaps precision and recall,
// which is why the orientation is stated here and asserted in the test.
//
// No loop over samples: encode both sides, form true*k + pred, and bincount.
// Three kernel calls for any number of rows.
def labels_of($y_true, $y_pred) {
    $p = _pair($y_true, $y_pred, "labels_of");
    return nd_unique(nd_concat([$p[0], $p[1]], 0));
}

def confusion_matrix($y_true, $y_pred, $labels) {
    $who = "confusion_matrix";
    $p = _pair($y_true, $y_pred, $who);
    $yt = $p[0];
    $yp = $p[1];
    _no_nan($yt, $who, "y_true");
    _no_nan($yp, $who, "y_pred");
    $classes = $labels;
    if ($classes == null) { $classes = nd_unique(nd_concat([$yt, $yp], 0)); }
    else { $classes = nd_unique(_vec($classes, $who)); }
    $k = nd_shape($classes)[0];
    if ($k == 0) { throw $who + ": no class labels"; }

    // With an explicit `labels` list, a value outside it has no cell to go in.
    // Dropping such rows silently -- which clamping would do -- changes the
    // totals, so it is an error naming the value.
    $ct = _codes($yt, $classes);
    $cp = _codes($yp, $classes);
    if ($labels != null) {
        $okT = nd_equal(nd_take($classes, $ct, 0), $yt, null);
        $okP = nd_equal(nd_take($classes, $cp, 0), $yp, null);
        if (!nd_get(nd_all(nd_logical_and($okT, $okP, null), null, null), 0)) {
            throw $who + ": a value in y_true or y_pred is not in the `labels` "
                  + "list given, so it has no cell in the matrix. Pass labels=null "
                  + "to take the classes from the data.";
        }
    }
    $flat = nd_add(nd_multiply(nd_astype($ct, "f64"), 1.0 * $k, null),
                   nd_astype($cp, "f64"), null);
    $counts = nd_bincount(nd_astype($flat, "i64"), $k * $k);
    return nd_reshape(nd_astype($counts, "f64"), [$k, $k]);
}

// ── precision / recall / F1 / support ───────────────────────────────────────
// The one computation every other classification metric reads from (K26).
// Returns a dict of (k,) arrays plus the class list, so a caller can average
// it however it likes without recomputing anything.
def precision_recall_fscore_support($y_true, $y_pred, $labels) {
    $who = "precision_recall_fscore_support";
    $classes = $labels;
    if ($classes == null) {
        $p = _pair($y_true, $y_pred, $who);
        $classes = nd_unique(nd_concat([$p[0], $p[1]], 0));
    } else { $classes = nd_unique(_vec($classes, $who)); }
    $cm = confusion_matrix($y_true, $y_pred, $classes);

    $tp   = nd_ravel(nd_reshape(nd_diagonal_of($cm), [nd_shape($cm)[0]]));
    $rows = nd_ravel(nd_sum($cm, 1, false));          // true count per class
    $cols = nd_ravel(nd_sum($cm, 0, false));          // predicted count per class
    $fp   = nd_subtract($cols, $tp, null);
    $fn   = nd_subtract($rows, $tp, null);

    $prec = _safe_div($tp, nd_add($tp, $fp, null));
    $rec  = _safe_div($tp, nd_add($tp, $fn, null));
    // F1 = 2PR/(P+R), and 0 where P+R is 0 -- which happens exactly when a
    // class has no true members and none predicted, or when every prediction
    // for it is wrong both ways.
    $f1   = _safe_div(nd_multiply(nd_multiply($prec, $rec, null), 2.0, null),
                      nd_add($prec, $rec, null));
    return {"classes": $classes, "precision": $prec, "recall": $rec,
            "f1": $f1, "support": $rows,
            "tp": $tp, "fp": $fp, "fn": $fn, "confusion": $cm};
}

// The diagonal as a (k,) array. numba has no nd_diagonal, and a strided view
// would need a stride numba cannot express here, so it is a gather: the
// diagonal of a (k,k) C-contiguous array is elements 0, k+1, 2k+2, ...
def nd_diagonal_of($m) {
    $k = nd_shape($m)[0];
    $idx = nd_astype(nd_arange(0, $k * $k, $k + 1), "i64");
    return nd_take(nd_ravel($m), $idx, 0);
}

// average: null (per class), "micro", "macro" or "weighted".
def _average($per, $r, $average, $who) {
    $av = $average;
    if ($av == null) { return $per; }
    if ($av == "macro") { return nd_get(nd_mean($per, null, null), 0); }
    if ($av == "weighted") {
        $tot = nd_get(nd_sum($r.support, null, null), 0);
        if ($tot == 0.0) { return 0.0; }
        return nd_get(nd_sum(nd_multiply($per, $r.support, null), null, null), 0) / $tot;
    }
    throw $who + ": average must be null, \"micro\", \"macro\" or \"weighted\", got \""
          + str($av) + "\"";
}

def precision_score($y_true, $y_pred, $average) {
    $who = "precision_score";
    $r = precision_recall_fscore_support($y_true, $y_pred, null);
    if ($average == "micro") {
        // Summed over classes, the micro precision is total correct over total
        // predicted -- which for single-label prediction is the accuracy.
        return _micro_tp($r) / _micro_pred($r);
    }
    return _average($r.precision, $r, $average, $who);
}

def recall_score($y_true, $y_pred, $average) {
    $who = "recall_score";
    $r = precision_recall_fscore_support($y_true, $y_pred, null);
    if ($average == "micro") { return _micro_tp($r) / _micro_true($r); }
    return _average($r.recall, $r, $average, $who);
}

def f1_score($y_true, $y_pred, $average) {
    $who = "f1_score";
    $r = precision_recall_fscore_support($y_true, $y_pred, null);
    if ($average == "micro") {
        $p = _micro_tp($r) / _micro_pred($r);
        $q = _micro_tp($r) / _micro_true($r);
        if ($p + $q == 0.0) { return 0.0; }
        return 2.0 * $p * $q / ($p + $q);
    }
    return _average($r.f1, $r, $average, $who);
}

// The micro totals. A zero denominator here needs every class to have no
// predictions at all (or no true members at all), which only happens on an
// empty input -- and _pair already refuses that. The guard returns 1 so the
// quotient is 0 rather than NaN, keeping the same rule as _safe_div (K27)
// instead of inventing a second one for the micro case.
def _micro_tp($r)   { return nd_get(nd_sum($r.tp, null, null), 0); }
def _micro_pred($r) { return _nonzero_or_one(nd_get(nd_sum(nd_add($r.tp, $r.fp, null), null, null), 0)); }
def _micro_true($r) { return _nonzero_or_one(nd_get(nd_sum(nd_add($r.tp, $r.fn, null), null, null), 0)); }
def _nonzero_or_one($v) {
    if ($v == 0.0) { return 1.0; }
    return $v;
}

def accuracy_score($y_true, $y_pred) {
    $p = _pair($y_true, $y_pred, "accuracy_score");
    _no_nan($p[0], "accuracy_score", "y_true");
    _no_nan($p[1], "accuracy_score", "y_pred");
    return nd_get(nd_mean(nd_astype(nd_equal($p[0], $p[1], null), "f64"), null, null), 0);
}

// ── classification_report ───────────────────────────────────────────────────
// Returns the NUMBERS (K29). A metric whose only output is a formatted string
// can only be tested by matching the string, and then every change to a column
// width breaks the test. The shape is scikit-learn's output_dict=True: one
// entry per class, plus "accuracy", "macro avg" and "weighted avg".
def classification_report($y_true, $y_pred, $labels) {
    $r  = precision_recall_fscore_support($y_true, $y_pred, $labels);
    $k  = nd_shape($r.classes)[0];
    $out = {};
    $i = 0;
    while ($i < $k) {
        $out[str(nd_get($r.classes, $i))] =
            {"precision": nd_get($r.precision, $i),
             "recall":    nd_get($r.recall, $i),
             "f1":        nd_get($r.f1, $i),
             "support":   nd_get($r.support, $i)};
        $i = $i + 1;
    }
    $tot = nd_get(nd_sum($r.support, null, null), 0);
    $out["accuracy"]  = accuracy_score($y_true, $y_pred);
    $out["macro avg"] = {"precision": nd_get(nd_mean($r.precision, null, null), 0),
                         "recall":    nd_get(nd_mean($r.recall, null, null), 0),
                         "f1":        nd_get(nd_mean($r.f1, null, null), 0),
                         "support":   $tot};
    $wp = 0.0;  $wr = 0.0;  $wf = 0.0;
    if ($tot > 0.0) {
        $wp = nd_get(nd_sum(nd_multiply($r.precision, $r.support, null), null, null), 0) / $tot;
        $wr = nd_get(nd_sum(nd_multiply($r.recall, $r.support, null), null, null), 0) / $tot;
        $wf = nd_get(nd_sum(nd_multiply($r.f1, $r.support, null), null, null), 0) / $tot;
    }
    $out["weighted avg"] = {"precision": $wp, "recall": $wr, "f1": $wf, "support": $tot};
    return $out;
}

def _pad($s, $w) {
    $t = str($s);
    while (len($t) < $w) { $t = " " + $t; }
    return $t;
}
def _fmt3($x) {
    // Three decimals, without relying on str()'s six significant digits.
    $v = floor($x * 1000.0 + 0.5) / 1000.0;
    $t = str($v);
    if (indexOf($t, ".") < 0) { $t = $t + ".000"; }
    while (len($t) - indexOf($t, ".") < 4) { $t = $t + "0"; }
    return $t;
}

// The formatter, separate from the numbers (K29).
def classification_report_text($report) {
    if ($report == null) { throw "classification_report_text: report is null"; }
    $lines = [];
    append($lines, _pad("class", 14) + _pad("precision", 11) + _pad("recall", 9)
                   + _pad("f1", 9) + _pad("support", 9));
    for $k, $v in $report.items() {
        if ($k != "accuracy" && $k != "macro avg" && $k != "weighted avg") {
            append($lines, _pad($k, 14) + _pad(_fmt3($v.precision), 11)
                           + _pad(_fmt3($v.recall), 9) + _pad(_fmt3($v.f1), 9)
                           + _pad(str($v.support), 9));
        }
    }
    append($lines, "");
    append($lines, _pad("accuracy", 14) + _pad("", 11) + _pad("", 9)
                   + _pad(_fmt3($report.accuracy), 9)
                   + _pad(str($report["macro avg"].support), 9));
    $tags = ["macro avg", "weighted avg"];
    $i = 0;
    while ($i < len($tags)) {
        $v = $report[$tags[$i]];
        append($lines, _pad($tags[$i], 14) + _pad(_fmt3($v.precision), 11)
                       + _pad(_fmt3($v.recall), 9) + _pad(_fmt3($v.f1), 9)
                       + _pad(str($v.support), 9));
        $i = $i + 1;
    }
    $s = "";
    $i = 0;
    while ($i < len($lines)) {
        if ($i > 0) { $s = $s + "\n"; }
        $s = $s + $lines[$i];
        $i = $i + 1;
    }
    return $s;
}

// ── the curves ──────────────────────────────────────────────────────────────
// Binary only: y_true in {0, 1}, y_score any real number where larger means
// "more likely to be class 1".
//
// ONE POINT PER DISTINCT THRESHOLD (K28). After sorting by score descending
// and taking cumulative true and false positives, a point is emitted only at
// the END of each run of equal scores -- otherwise the curve passes through
// points no threshold can produce, and its area depends on the order the equal
// scores happened to arrive in.
def _binary_sorted($y_true, $y_score, $who) {
    $p = _pair($y_true, $y_score, $who);
    $yt = $p[0];
    $ys = $p[1];
    _no_nan($yt, $who, "y_true");
    _no_nan($ys, $who, "y_score");
    $u = nd_unique($yt);
    $bad = nd_get(nd_count_nonzero(nd_logical_and(
            nd_not_equal($u, 0.0, null), nd_not_equal($u, 1.0, null), null), null, null), 0);
    if ($bad > 0) {
        throw $who + ": y_true must be 0/1 for a binary curve, saw "
              + str(nd_to_list($u));
    }
    $nPos = nd_get(nd_sum($yt, null, null), 0);
    $nNeg = nd_shape($yt)[0] - $nPos;
    if ($nPos == 0.0 || $nNeg == 0.0) {
        throw $who + ": y_true has only one class (" + str($nPos) + " positive of "
              + str(nd_shape($yt)[0]) + "), so the curve is undefined -- there is "
              + "nothing to rank against";
    }
    // descending by score: argsort ascending on the negation. nd_argsort is a
    // stable sort, so equal scores keep input order -- which does not matter,
    // because the tie handling below makes the result independent of it.
    $order = nd_argsort(nd_negative($ys, null), null);
    $ss = nd_take($ys, $order, 0);
    $st = nd_take($yt, $order, 0);
    $tps = nd_cumsum($st, null);
    $fps = nd_subtract(nd_astype(nd_add(nd_arange(0, nd_shape($st)[0], null), 1.0, null), "f64"),
                       $tps, null);
    // the last index of each run of equal scores
    $n = nd_shape($ss)[0];
    $cut = nd([1.0 * ($n - 1)], "f64");
    if ($n > 1) {
        $d = nd_nonzero(nd_not_equal(nd_diff($ss, null), 0.0, null));
        if (nd_size($d) > 0) {
            $cut = nd_concat([nd_astype($d, "f64"), nd([1.0 * ($n - 1)], "f64")], 0);
        }
    }
    $ci = nd_astype($cut, "i64");
    return {"tps": nd_take($tps, $ci, 0), "fps": nd_take($fps, $ci, 0),
            "thresholds": nd_take($ss, $ci, 0), "n_pos": $nPos, "n_neg": $nNeg};
}

def roc_curve($y_true, $y_score) {
    $s = _binary_sorted($y_true, $y_score, "roc_curve");
    // (0,0) is prepended: the point where nothing is predicted positive. It is
    // not produced by any threshold in the data, and the trapezoid needs it.
    $fpr = nd_concat([nd([0.0], "f64"), nd_divide($s.fps, $s.n_neg, null)], 0);
    $tpr = nd_concat([nd([0.0], "f64"), nd_divide($s.tps, $s.n_pos, null)], 0);
    $thr = nd_concat([nd([nd_get($s.thresholds, 0) + 1.0], "f64"), $s.thresholds], 0);
    return {"fpr": $fpr, "tpr": $tpr, "thresholds": $thr};
}

// The TRAPEZOID over (fpr, tpr) -- which for a step-shaped ROC is the same as
// the Mann-Whitney statistic, and is what scikit-learn reports.
def roc_auc_score($y_true, $y_score) {
    $c = roc_curve($y_true, $y_score);
    return _trapezoid($c.fpr, $c.tpr);
}

def _trapezoid($x, $y) {
    $n = nd_shape($x)[0];
    if ($n < 2) { return 0.0; }
    $dx = nd_diff($x, null);
    $y0 = nd_slice($y, [[0, $n - 1, 1]]);
    $y1 = nd_slice($y, [[1, $n, 1]]);
    return nd_get(nd_sum(nd_multiply($dx, nd_multiply(nd_add($y0, $y1, null), 0.5, null), null),
                         null, null), 0);
}

def precision_recall_curve($y_true, $y_score) {
    $s = _binary_sorted($y_true, $y_score, "precision_recall_curve");
    $prec = nd_divide($s.tps, nd_add($s.tps, $s.fps, null), null);
    $rec  = nd_divide($s.tps, $s.n_pos, null);
    // scikit-learn appends (precision=1, recall=0) as the last point, which is
    // the limit as the threshold rises above every score. The thresholds array
    // is therefore one shorter than the curves, deliberately.
    return {"precision": nd_concat([$prec, nd([1.0], "f64")], 0),
            "recall":    nd_concat([$rec, nd([0.0], "f64")], 0),
            "thresholds": $s.thresholds};
}

// The STEP SUM, not the trapezoid (K28): sum over thresholds of
// (R_n - R_{n-1}) * P_n. This is scikit-learn's average_precision_score, and
// it differs from the area under the interpolated curve -- the trapezoid is
// optimistic on a curve that steps down, which a PR curve does.
def average_precision_score($y_true, $y_score) {
    $s = _binary_sorted($y_true, $y_score, "average_precision_score");
    $prec = nd_divide($s.tps, nd_add($s.tps, $s.fps, null), null);
    $rec  = nd_divide($s.tps, $s.n_pos, null);
    $prev = nd_concat([nd([0.0], "f64"),
                       nd_slice($rec, [[0, nd_shape($rec)[0] - 1, 1]])], 0);
    return nd_get(nd_sum(nd_multiply(nd_subtract($rec, $prev, null), $prec, null),
                         null, null), 0);
}

// ── probabilistic losses ────────────────────────────────────────────────────
// CLIPPED, so a confident mistake is finite (§5.5). log(0) is -inf, and one
// prediction of exactly 0 for a positive sample would make the loss of a
// 10,000-row evaluation `inf` -- which says nothing about the model beyond
// "at least one prediction was certain and wrong".
def log_loss($y_true, $y_prob, $eps) {
    $who = "log_loss";
    $e = $eps;
    if ($e == null) { $e = 1.0e-15; }
    if ($e <= 0.0 || $e >= 0.5) {
        throw $who + ": eps must be in (0, 0.5), got " + str($e);
    }
    $P = $y_prob;
    if (type($P) == "list") { $P = nd($P, "f64"); }
    if (type($P) != "ndarray") { throw $who + ": y_prob must be an array or a list"; }
    if (nd_dtype($P) != "f64") { $P = nd_astype($P, "f64"); }
    _no_nan($P, $who, "y_prob");

    if (nd_ndim($P) == 2 && nd_shape($P)[1] > 1) {
        // multiclass: one column per class, y_true holding class INDICES
        $yt = _vec($y_true, $who);
        $n  = nd_shape($P)[0];
        $k  = nd_shape($P)[1];
        if (nd_shape($yt)[0] != $n) {
            throw $who + ": " + str(nd_shape($yt)[0]) + " labels for " + str($n) + " rows";
        }
        $lo = nd_get(nd_min($yt, null, null), 0);
        $hi = nd_get(nd_max($yt, null, null), 0);
        if ($lo < 0.0 || $hi > 1.0 * ($k - 1) || nd_get(nd_any(
                nd_not_equal($yt, nd_floor($yt, null), null), null, null), 0)) {
            throw $who + ": with a " + str($k) + "-column y_prob, y_true must hold "
                  + "whole class indices in 0.." + str($k - 1) + " (saw "
                  + str($lo) + ".." + str($hi) + ")";
        }
        // rows are normalised first, as scikit-learn does, so a row that does
        // not quite sum to 1 is scored as the distribution it implies rather
        // than being silently rewarded or punished for its total
        $rs = nd_sum($P, 1, true);
        if (nd_get(nd_any(nd_equal($rs, 0.0, null), null, null), 0)) {
            throw $who + ": a row of y_prob sums to zero, so it is not a distribution";
        }
        $Pn = nd_clip(nd_divide($P, $rs, null), $e, 1.0 - $e, null);
        // the probability assigned to the TRUE class of each row
        $idx = nd_astype(nd_add(nd_multiply(nd_astype($yt, "f64"), 0.0, null),
                                nd_add(nd_multiply(nd_astype(
                                    nd_arange(0, $n, null), "f64"), 1.0 * $k, null),
                                    nd_astype($yt, "f64"), null), null), "i64");
        $pick = nd_take(nd_ravel($Pn), $idx, 0);
        return 0.0 - nd_get(nd_mean(nd_log($pick, null), null, null), 0);
    }

    // binary: y_prob is P(class 1)
    $p = _pair($y_true, $P, $who);
    $yt = $p[0];
    $pp = nd_clip($p[1], $e, 1.0 - $e, null);
    $u = nd_unique($yt);
    $bad = nd_get(nd_count_nonzero(nd_logical_and(
            nd_not_equal($u, 0.0, null), nd_not_equal($u, 1.0, null), null), null, null), 0);
    if ($bad > 0) {
        throw $who + ": with a 1-D y_prob, y_true must be 0/1, saw " + str(nd_to_list($u));
    }
    $term = nd_add(nd_multiply($yt, nd_log($pp, null), null),
                   nd_multiply(nd_subtract(1.0, $yt, null),
                               nd_log(nd_subtract(1.0, $pp, null), null), null), null);
    return 0.0 - nd_get(nd_mean($term, null, null), 0);
}

// mean((p - y)^2) over the positive-class probability. Unlike log_loss it is
// bounded in [0, 1], so it needs no clipping: a confident mistake costs 1.
def brier_score_loss($y_true, $y_prob) {
    $who = "brier_score_loss";
    $p = _pair($y_true, $y_prob, $who);
    _no_nan($p[0], $who, "y_true");
    _no_nan($p[1], $who, "y_prob");
    $u = nd_unique($p[0]);
    $bad = nd_get(nd_count_nonzero(nd_logical_and(
            nd_not_equal($u, 0.0, null), nd_not_equal($u, 1.0, null), null), null, null), 0);
    if ($bad > 0) { throw $who + ": y_true must be 0/1, saw " + str(nd_to_list($u)); }
    $d = nd_subtract($p[1], $p[0], null);
    return nd_get(nd_mean(nd_multiply($d, $d, null), null, null), 0);
}

// ── regression ──────────────────────────────────────────────────────────────
def mean_squared_error($y_true, $y_pred) {
    $p = _pair($y_true, $y_pred, "mean_squared_error");
    _no_nan($p[0], "mean_squared_error", "y_true");
    _no_nan($p[1], "mean_squared_error", "y_pred");
    $d = nd_subtract($p[0], $p[1], null);
    return nd_get(nd_mean(nd_multiply($d, $d, null), null, null), 0);
}

def root_mean_squared_error($y_true, $y_pred) {
    return sqrt(mean_squared_error($y_true, $y_pred));
}

def mean_absolute_error($y_true, $y_pred) {
    $p = _pair($y_true, $y_pred, "mean_absolute_error");
    _no_nan($p[0], "mean_absolute_error", "y_true");
    _no_nan($p[1], "mean_absolute_error", "y_pred");
    return nd_get(nd_mean(nd_abs(nd_subtract($p[0], $p[1], null), null), null, null), 0);
}

// 1 - SSE/SST. A CONSTANT TARGET RAISES (K30): SST is zero, so the metric's
// own denominator is gone. scikit-learn returns 0.0 with a warning, which
// makes a meaningless comparison look like a bad one; base.b's score() already
// raised before this file existed, and one package should not disagree with
// itself about the same formula.
def r2_score($y_true, $y_pred) {
    $who = "r2_score";
    $p = _pair($y_true, $y_pred, $who);
    $yt = $p[0];
    _no_nan($yt, $who, "y_true");
    _no_nan($p[1], $who, "y_pred");
    $res = nd_subtract($yt, $p[1], null);
    $sse = nd_get(nd_sum(nd_multiply($res, $res, null), null, null), 0);
    $mu  = nd_get(nd_mean($yt, null, null), 0);
    $dev = nd_subtract($yt, $mu, null);
    $sst = nd_get(nd_sum(nd_multiply($dev, $dev, null), null, null), 0);
    if ($sst == 0.0) {
        throw $who + ": y_true is constant, so the total sum of squares is zero "
              + "and R2 is undefined. Use mean_squared_error or "
              + "mean_absolute_error for a constant target.";
    }
    return 1.0 - $sse / $sst;
}

// 1 - Var(y - pred)/Var(y). Equal to R2 when the residuals have zero mean, and
// smaller than it when the prediction is biased -- which is the whole reason
// to report both.
def explained_variance_score($y_true, $y_pred) {
    $who = "explained_variance_score";
    $p = _pair($y_true, $y_pred, $who);
    $yt = $p[0];
    _no_nan($yt, $who, "y_true");
    _no_nan($p[1], $who, "y_pred");
    $vy = nd_get(nd_var($yt, null, null, 0), 0);
    if ($vy == 0.0) {
        throw $who + ": y_true is constant, so its variance is zero and the "
              + "explained fraction is undefined";
    }
    $vr = nd_get(nd_var(nd_subtract($yt, $p[1], null), null, null, 0), 0);
    return 1.0 - $vr / $vy;
}

// mean(|y - pred| / max(|y|, eps)). THE DENOMINATOR IS CLAMPED, not refused
// (§5.6): a single zero among thousands of nonzero targets is a data point,
// not a reason to reject the whole metric. The consequence is that the result
// is bounded rather than exact at a zero target, which is why it is said here.
def mean_absolute_percentage_error($y_true, $y_pred, $eps) {
    $who = "mean_absolute_percentage_error";
    $e = $eps;
    if ($e == null) { $e = 1.0e-15; }
    $p = _pair($y_true, $y_pred, $who);
    _no_nan($p[0], $who, "y_true");
    _no_nan($p[1], $who, "y_pred");
    $den = nd_maximum(nd_abs($p[0], null), $e, null);
    return nd_get(nd_mean(nd_divide(nd_abs(nd_subtract($p[0], $p[1], null), null),
                                    $den, null), null, null), 0);
}

// ── ranking ─────────────────────────────────────────────────────────────────
// One query: y_true is the relevance of each item, y_score the model's ranking
// signal. A (n, m) pair is n queries of m items, averaged.
def _rank_pair($y_true, $y_score, $who) {
    $T = $y_true;
    $S = $y_score;
    if (type($T) == "list") { $T = nd($T, "f64"); }
    if (type($S) == "list") { $S = nd($S, "f64"); }
    if (type($T) != "ndarray" || type($S) != "ndarray") {
        throw $who + ": expected arrays or lists";
    }
    if (nd_dtype($T) != "f64") { $T = nd_astype($T, "f64"); }
    if (nd_dtype($S) != "f64") { $S = nd_astype($S, "f64"); }
    if (nd_ndim($T) == 1) { $T = nd_reshape($T, [1, nd_shape($T)[0]]); }
    if (nd_ndim($S) == 1) { $S = nd_reshape($S, [1, nd_shape($S)[0]]); }
    if (str(nd_shape($T)) != str(nd_shape($S))) {
        throw $who + ": y_true is " + str(nd_shape($T)) + " and y_score is "
              + str(nd_shape($S)) + " -- they must match";
    }
    if (nd_shape($T)[1] == 0) { throw $who + ": no items to rank"; }
    _no_nan($T, $who, "y_true");
    _no_nan($S, $who, "y_score");
    return [$T, $S];
}

// Discounted cumulative gain with the standard log2(rank+1) discount and
// 2^rel - 1 gain, truncated at k. Ties in the score are broken by the order
// the items appear in, which is stated rather than hidden: NDCG is not
// tie-invariant and any implementation has to choose.
def _dcg($rel, $k) {
    $m = nd_shape($rel)[0];
    $kk = $k;
    if ($kk == null || $kk > $m) { $kk = $m; }
    $r = nd_slice($rel, [[0, $kk, 1]]);
    $gain = nd_subtract(nd_power(2.0, $r, null), 1.0, null);
    $disc = nd_log2(nd_add(nd_astype(nd_add(nd_arange(0, $kk, null), 1.0, null), "f64"),
                           1.0, null), null);
    return nd_get(nd_sum(nd_divide($gain, $disc, null), null, null), 0);
}

def ndcg_score($y_true, $y_score, $k) {
    $who = "ndcg_score";
    $p = _rank_pair($y_true, $y_score, $who);
    $T = $p[0];
    $S = $p[1];
    $n = nd_shape($T)[0];
    $tot = 0.0;
    $q = 0;
    while ($q < $n) {
        $rel = nd_ravel(nd_slice($T, [[$q, $q + 1, 1], null]));
        $sc  = nd_ravel(nd_slice($S, [[$q, $q + 1, 1], null]));
        $byScore = nd_take($rel, nd_argsort(nd_negative($sc, null), null), 0);
        $ideal   = nd_take($rel, nd_argsort(nd_negative($rel, null), null), 0);
        $idcg = _dcg($ideal, $k);
        // A query with no relevant item has IDCG 0. Scoring it 0 would punish
        // the model for a query it could not get right; scikit-learn scores it
        // 0 as well, and the count is kept in the mean either way, so the
        // choice is stated rather than left to a division.
        if ($idcg == 0.0) { $tot = $tot + 0.0; }
        if ($idcg != 0.0) { $tot = $tot + _dcg($byScore, $k) / $idcg; }
        $q = $q + 1;
    }
    return $tot / (1.0 * $n);
}

// Mean reciprocal rank: 1/(rank of the first relevant item), averaged over
// queries. `relevant` means y_true > 0.
def mrr_score($y_true, $y_score) {
    $who = "mrr_score";
    $p = _rank_pair($y_true, $y_score, $who);
    $T = $p[0];
    $S = $p[1];
    $n = nd_shape($T)[0];
    $m = nd_shape($T)[1];
    $tot = 0.0;
    $q = 0;
    while ($q < $n) {
        $rel = nd_ravel(nd_slice($T, [[$q, $q + 1, 1], null]));
        $sc  = nd_ravel(nd_slice($S, [[$q, $q + 1, 1], null]));
        $ord = nd_take($rel, nd_argsort(nd_negative($sc, null), null), 0);
        $hit = nd_nonzero(nd_greater($ord, 0.0, null));
        // A query with nothing relevant contributes 0, not a skipped
        // denominator: dropping it would make the mean depend on which queries
        // happened to have an answer.
        if (nd_size($hit) > 0) { $tot = $tot + 1.0 / (nd_get($hit, 0) + 1.0); }
        $q = $q + 1;
    }
    return $tot / (1.0 * $n);
}
