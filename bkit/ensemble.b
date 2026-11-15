// ════════════════════════════════════════════════════════════════════════════
//  bkit/ensemble.b — gradient boosting over the tree_ native family.
//
//      include "bkit/ensemble.b" as ens;
//      $m = ens.GradientBoostingRegressor({"n_estimators": 200,
//                                          "learning_rate": 0.05});
//      $m.fit($X, $y);
//      $m.predict($Xtest);
//
//  ── this file adds no kernels, and that is the design ─────────────────────
//  K7 shipped five tree_ kernels and §8.5 of the design doc is the record of
//  checking, feature by feature, that they are sufficient:
//
//    histogram subtraction  one nd_subtract per node on a [f, bins, 3] array
//    row subsampling        tree_hist already takes an index array
//    column subsampling     zero the unselected features' rows in the
//                           histogram the SCAN sees; a zeroed feature has
//                           G = H = n = 0, so every cut on it fails both
//                           min_data and the hessian guard
//    early stopping         one tree_predict per round on a held-out fold
//
//  So the new work here is the boosting LOOP — gradients, hessians, shrinkage,
//  stopping, importances — and the per-element work stays in C++.
//
//  ── ONE grower, not a second one ──────────────────────────────────────────
//  Every tree is grown by `tree.TreeGrower`, which is `_TreeBase._grow`
//  itself, unchanged. A second grower here would be a second place that has to
//  get the NaN direction, the empty-child assertion and the raw-value
//  threshold lookup right — and [K44] is the record of getting the NaN
//  direction subtly wrong in the *one* place that existed. §8.6 argues this at
//  length, including why it is the OPPOSITE call from btorch's [B34]: there,
//  sharing one normalisation cost four full tensor copies; here it costs one
//  `if` on a parameter.
//
//  ── parameters come as a DICT, and the rest of bkit is positional ─────────
//  A deliberate break with `DecisionTreeRegressor(5, 1, null, null, null,
//  null)`. A booster has fourteen parameters; a fourteen-slot positional
//  constructor is a place where transposing two of them is silent and
//  unreadable at the call site. `get_params`/`set_params` have always spoken
//  dicts, so this makes construction and reconfiguration the same shape.
//  **An unknown key raises** ([K24]): bkit has already shipped one bug where a
//  mistyped parameter scored fine.
//
//  ── the objectives ────────────────────────────────────────────────────────
//      squared_error   g = F − y            h = 1              F0 = ȳ
//      logistic        g = σ(F) − y         h = σ(1−σ)         F0 = logit(ȳ)
//      softmax         g = p_k − y_k        h = p_k(1−p_k)     F0 = 0
//      poisson         g = e^F − y          h = e^F            F0 = log(ȳ)
//
//  h is clamped at 1e-16, and that clamp is the FIRST of three layers, not the
//  only one: `tree_best_split` refuses a child with !(H + λ > 0) because G²/0
//  is +inf and wins every comparison ([K45]), and `_leafValue` applies the
//  identical guard to the leaf ([K55]). The clamp lives in the objective and a
//  custom objective can simply not have it, which is why the other two exist.
//
//  λ defaults to **1.0** here, not to the single tree's 0: a booster adds many
//  shrunken trees and wants its leaves regularised.
//
//  ── reproducibility ───────────────────────────────────────────────────────
//  No estimator carries RNG state ([C4]). Row and column subsamples come from
//  `nd_random_uniform`, so the whole fit is reproducible from `nd_seed` alone,
//  and a model fitted with subsample = 1.0 and colsample = 1.0 draws no
//  randomness at all.
// ════════════════════════════════════════════════════════════════════════════

include "../numba/numba.b" as np;
include "./base.b" as base;
include "./tree.b" as tree;
include "./selection.b" as sel;

// A full reduction returns a 0-d ndarray HANDLE and every handle is truthy
// (numba N41), so it must be read out rather than used as a number.
def _scalar($x) {
    return np.get(np.reshape($x, [1]), [0]);
}

// ── parameters ──────────────────────────────────────────────────────────────
// One table, used for defaults, for validation and for the error message that
// lists what is accepted. A parameter that is not here cannot be set.
def _defaults($objective) {
    return {
        "n_estimators":        100,
        "learning_rate":       0.1,
        "max_depth":           3,
        "min_samples_leaf":    1,
        "max_bins":            256,
        "reg_lambda":          1.0,
        "gamma":               0.0,
        "min_child_weight":    1.0,
        "subsample":           1.0,
        "colsample":           1.0,
        "objective":           $objective,
        "validation_fraction": 0.1,
        "n_iter_no_change":    null,
        "tol":                 0.0001
    };
}

def _num($v, $name) {
    if (type($v) != "number") {
        throw "bkit.ensemble: " + $name + " must be a number, got " + type($v)
              + " (" + str($v) + ")";
    }
    return $v;
}

def _whole($v, $lo, $name) {
    _num($v, $name);
    if ($v != floor($v) || $v < $lo) {
        throw "bkit.ensemble: " + $name + " must be a whole number >= " + str($lo)
              + ", got " + str($v);
    }
    return $v;
}

// A fraction in (0, 1]. Zero is refused rather than clamped: subsample = 0
// means every tree sees no rows, which would fit a model of nothing and report
// a clean convergence.
def _frac($v, $name) {
    _num($v, $name);
    if (!($v > 0.0) || $v > 1.0) {
        throw "bkit.ensemble: " + $name + " must be in (0, 1], got " + str($v);
    }
    return $v;
}

def _checkParams($p, $who) {
    _whole($p.n_estimators, 1, "n_estimators");
    _num($p.learning_rate, "learning_rate");
    if (!($p.learning_rate > 0.0) || $p.learning_rate > 1.0) {
        throw "bkit.ensemble: learning_rate must be in (0, 1], got "
              + str($p.learning_rate);
    }
    _whole($p.max_depth, 1, "max_depth");
    _whole($p.min_samples_leaf, 1, "min_samples_leaf");
    _whole($p.max_bins, 2, "max_bins");
    _num($p.reg_lambda, "reg_lambda");
    _num($p.gamma, "gamma");
    _num($p.min_child_weight, "min_child_weight");
    if ($p.reg_lambda < 0.0 || $p.gamma < 0.0 || $p.min_child_weight < 0.0) {
        throw "bkit.ensemble: reg_lambda, gamma and min_child_weight must be "
              + ">= 0";
    }
    _frac($p.subsample, "subsample");
    _frac($p.colsample, "colsample");
    _frac($p.validation_fraction, "validation_fraction");
    _num($p.tol, "tol");
    if ($p.tol < 0.0) {
        throw "bkit.ensemble: tol must be >= 0, got " + str($p.tol);
    }
    if ($p.n_iter_no_change != null) {
        _whole($p.n_iter_no_change, 1, "n_iter_no_change");
    }
    if (!contains(["squared_error", "logistic", "softmax", "poisson"],
                  $p.objective)) {
        throw $who + ": objective '" + str($p.objective) + "' is not one of "
              + "squared_error, logistic, softmax, poisson";
    }
    return $p;
}

// ── objectives ──────────────────────────────────────────────────────────────
// Dispatched on a string rather than held as closures: Bantu closures capture
// by REFERENCE (trap 1), so a dict of closures built over a loop variable is a
// trap, and these are called once per round anyway.

def _sigmoid($F) {
    return np.divide(1.0, np.add(1.0, np.exp(np.multiply($F, 0.0 - 1.0, null),
                                             null), null), null);
}

// The starting score, so round 1 does not spend itself learning the mean.
def _initScore($obj, $yv, $who) {
    if ($obj == "squared_error") { return _scalar(np.mean($yv, null, false)); }
    if ($obj == "softmax")       { return 0.0; }
    $m = _scalar(np.mean($yv, null, false));
    if ($obj == "logistic") {
        // Clipped away from 0 and 1: an all-one-class target would make
        // logit(p) infinite, and ±inf as a starting score poisons every
        // gradient afterwards with NaN. The classifier refuses a single-class
        // target anyway, so this guards a fold that happens to be pure.
        if ($m < 1.0e-6)       { $m = 1.0e-6; }
        if ($m > 1.0 - 1.0e-6) { $m = 1.0 - 1.0e-6; }
        return _scalar(np.log(np.array([$m / (1.0 - $m)], "f64"), null));
    }
    if ($obj == "poisson") {
        if (!($m > 0.0)) {
            throw $who + ": the poisson objective needs a positive mean target; "
                  + "this one is " + str($m) + ", and log of it is undefined";
        }
        return _scalar(np.log(np.array([$m], "f64"), null));
    }
    throw $who + ": no initial score for objective '" + str($obj) + "'";
}

// Hessians are made VALID here, and nothing more. This function used to clamp
// to 1e-16, on the reasoning that keeping the Newton step finite beats
// discarding it. **That was measured and it is wrong**, which is [K58]:
//
//   h = 1e-16, λ = 0, min_child_weight = 0  ->  splits, leaves of ±5e15
//   h = 0,     λ = 0, min_child_weight = 0  ->  one leaf, value 0
//   h = 1e-16, λ = 1 (the default)          ->  one leaf, value 0
//
// A saturated node's correct answer is the middle one, and the two guards
// already give it: tree_best_split refuses a child with !(H + λ > 0) ([K45])
// and _leafValue returns 0 for the same condition ([K55]). Clamping to 1e-16
// made H + λ pass both of them by a hair and turned a refused split into a step
// of five quadrillion -- the +inf failure mode [K45] exists to prevent, scaled
// down just far enough to survive a finiteness check. The clamp DEFEATED the
// guards in exactly the configuration where they are needed.
//
// So a hessian of zero is passed through, and only what is genuinely invalid is
// corrected: a NEGATIVE hessian (no convex objective produces one, and the gain
// G²/(H+λ) is meaningless for it) and a NaN (which would make the gain lose
// every comparison and the leaf value NaN). Both become 0, which the guards
// then handle. np.maximum PROPAGATES NaN rather than dropping it like IEEE
// fmax, so the NaN case needs its own where().
def _validH($h) {
    return np.where(np.isnan($h, null), 0.0, np.maximum($h, 0.0, null));
}

// (g, h) for one output column. $F and $yv are both [n].
def _gradHess($obj, $F, $yv) {
    if ($obj == "squared_error") {
        return [np.subtract($F, $yv, null), np.ones([np.size($yv)], "f64")];
    }
    // The softmax caller passes the probability column itself, because p_k is a
    // function of all K scores and is computed once per round for all of them.
    // The logistic caller passes the raw score, since p = sigma(F) needs only
    // its own column.
    if ($obj == "logistic" || $obj == "softmax") {
        $p = $F;
        if ($obj == "logistic") { $p = _sigmoid($F); }
        return [np.subtract($p, $yv, null),
                _validH(np.multiply($p, np.subtract(1.0, $p, null), null))];
    }
    if ($obj == "poisson") {
        $e = np.exp($F, null);
        return [np.subtract($e, $yv, null), _validH($e)];
    }
    throw "bkit.ensemble: no gradient for objective '" + str($obj) + "'";
}

// Row-normalised softmax over a list of K score columns, returned as a list of
// K probability columns. Shifted by the row max before exponentiating, which is
// the standard guard: a score of 800 makes e^F inf and inf/inf NaN.
def _softmaxCols($cols) {
    $k = len($cols);
    $mx = $cols[0];
    $i = 1;
    while ($i < $k) { $mx = np.maximum($mx, $cols[$i], null); $i = $i + 1; }
    $ex  = [];
    $tot = null;
    $i = 0;
    while ($i < $k) {
        $e = np.exp(np.subtract($cols[$i], $mx, null), null);
        append($ex, $e);
        if ($tot == null) { $tot = $e; }
        else { $tot = np.add($tot, $e, null); }
        $i = $i + 1;
    }
    $out = [];
    $i = 0;
    while ($i < $k) { append($out, np.divide($ex[$i], $tot, null)); $i = $i + 1; }
    return $out;
}

// ── subsampling ─────────────────────────────────────────────────────────────
// A uniformly random permutation in two kernel calls, the same construction
// selection.b uses (K18): argsort of distinct random keys is a uniform
// permutation, and nothing here carries RNG state ([C4]).
def _permutation($n) {
    return np.argsort(np.random([$n], 0.0, 1.0), null);
}

// `frac` of 0..n-1, WITHOUT replacement, as an i64 index array. Returns null
// for frac = 1.0 rather than a full index array, because null is what the
// grower treats as "every row" and building 0..n-1 to say the same thing is a
// copy per tree.
def _rowSample($n, $frac) {
    if ($frac >= 1.0) { return null; }
    $m = floor(1.0 * $n * $frac);
    if ($m < 1.0) { $m = 1.0; }
    return np.astype(np.take(_permutation($n),
                             np.astype(np.arange(0.0, $m, 1.0), "i64"), 0),
                     "i64");
}

// `frac` of 0..f-1 as a LIST of feature indices, or null for all of them.
def _colSample($f, $frac) {
    if ($frac >= 1.0) { return null; }
    $m = floor(1.0 * $f * $frac);
    if ($m < 1.0) { $m = 1.0; }
    $perm = np.to_list(_permutation($f));
    $out = [];
    $i = 0;
    while ($i < $m) { append($out, $perm[$i]); $i = $i + 1; }
    return $out;
}

// ── the booster ─────────────────────────────────────────────────────────────
class _BoostBase extends Estimator {
    def init($params, $objective, $name, $task) {
        super();
        this.name_   = $name;
        this.task_   = $task;
        this.params_ = _defaults($objective);
        this.set_params($params);
        this.fitted_ = false;
        _checkParams(this.params_, this.name_);

        this.rounds_   = null;    // list of rounds; each is a list of K node arrays
        this.init_     = null;    // the starting score, one per output column
        this.n_feat_   = 0;
        this.classes_  = null;
        this.n_out_    = 1;       // 1, except softmax where it is K
        this.gains_    = null;
        this.splits_   = null;
        this.best_round_ = 0;
        this.val_scores_ = null;  // the held-out loss per round, when stopping
        // Recorded so the §8.7 ordering is TESTABLE rather than asserted in a
        // comment: n_train_ is how many rows were binned, and nbins_ is the bin
        // count per feature that binning produced. If the validation fold were
        // binned too, a feature whose extra distinct value lives only in that
        // fold would come back with one more bin.
        this.n_train_ = 0;
        this.nbins_   = null;
    }

    // y as one column per output: [y] for a single output, a one-hot list of K
    // columns for softmax.
    def _targets($yv) { return [$yv]; }

    // Hooks every subclass must answer. Declared here and raising by name, so a
    // new objective that forgets one fails at the hook instead of somewhere
    // unrelated three calls later (the reason base.b's Estimator does the same).
    def _prepareTarget($v, $who) {
        throw this.name_ + "._prepareTarget is not implemented";
    }
    def _valLoss($Fva, $yva) {
        throw this.name_ + "._valLoss is not implemented, so this estimator "
              + "cannot early-stop";
    }
    // The labels the validation split should be stratified on, or null. A
    // classifier stratifies: a 1%-prevalence class landing wholly in the
    // validation fold by chance would make the stopping criterion measure a
    // class the model never trained on.
    def _stratifyOn($v) { return null; }

    // ── fit ────────────────────────────────────────────────────────────────
    def fit($X, $y) {
        $who = this.name_ + ".fit";
        $A = base._asarray($X);
        if (len(np.shape($A)) != 2) {
            throw $who + ": X must be 2-d [n_samples, n_features], got shape "
                  + str(np.shape($A));
        }
        if (np.shape($A)[0] == 0) { throw $who + ": X has no rows"; }
        if ($y == null) { throw $who + ": y is required"; }
        $v = base._asarray($y);
        if (len(np.shape($v)) != 1) { $v = np.reshape($v, [np.size($v)]); }
        if (np.shape($v)[0] != np.shape($A)[0]) {
            throw $who + ": y has " + str(np.shape($v)[0]) + " elements and X has "
                  + str(np.shape($A)[0]) + " rows";
        }
        _checkParams(this.params_, $who);
        this.n_feat_ = np.shape($A)[1];
        this._prepareTarget($v, $who);

        // EARLY STOPPING SPLITS BEFORE BINNING (§8.7). Binning the whole matrix
        // and then holding a fold out would choose the bin edges using data the
        // model is scored on -- a quieter leak than training on it, because it
        // inflates the stopping round rather than the score and so does not
        // show up as an implausible number.
        $Xtr = $A;   $ytr = $v;
        $Xva = null; $yva = null;
        if (this.params_.n_iter_no_change != null) {
            $parts = sel.train_test_split_indices($A, this.params_.validation_fraction,
                                                  true, this._stratifyOn($v));
            $itr = $parts[0];
            $iva = $parts[1];
            if (np.size($iva) < 1) {
                throw $who + ": validation_fraction "
                      + str(this.params_.validation_fraction) + " of "
                      + str(np.shape($A)[0]) + " rows is less than one row, so "
                      + "early stopping would score nothing";
            }
            $Xtr = np.take($A, $itr, 0);  $ytr = np.take($v, $itr, 0);
            $Xva = np.take($A, $iva, 0);  $yva = np.take($v, $iva, 0);
        }

        $n = np.shape($Xtr)[0];
        $B = tree_bin($Xtr, this.params_.max_bins);
        this.n_train_ = $n;
        this.nbins_   = tree_bin_nbins($B);
        $Y = this._targets($ytr);
        $k = len($Y);
        this.n_out_ = $k;

        // One starting score per output column.
        this.init_ = [];
        $i = 0;
        while ($i < $k) {
            append(this.init_, _initScore(this.params_.objective, $Y[$i], $who));
            $i = $i + 1;
        }

        // F, the running raw score, one column per output.
        $F = [];
        $i = 0;
        while ($i < $k) {
            append($F, np.add(np.zeros([$n], "f64"), this.init_[$i], null));
            $i = $i + 1;
        }
        $Fva = null;
        if ($Xva != null) {
            $Fva = [];
            $i = 0;
            while ($i < $k) {
                append($Fva, np.add(np.zeros([np.shape($Xva)[0]], "f64"),
                                    this.init_[$i], null));
                $i = $i + 1;
            }
        }

        // ONE grower for every tree in the model, reused rather than rebuilt:
        // it holds no per-tree state beyond the feature mask, and its
        // accumulated importances are exactly the model's totals.
        $G = tree.TreeGrower(this.params_.max_depth, this.params_.min_samples_leaf,
                             this.params_.max_bins, this.params_.reg_lambda,
                             this.params_.gamma, this.params_.min_child_weight);
        $G.known_features(this.n_feat_);
        $G.reset_importance();

        this.rounds_     = [];
        this.val_scores_ = [];
        $best     = null;
        $bestAt   = 0;
        $stale    = 0;
        $r = 0;
        while ($r < this.params_.n_estimators) {
            // The probabilities softmax needs are a function of ALL K columns,
            // so they are computed once per round rather than per class.
            $P = null;
            if (this.params_.objective == "softmax") { $P = _softmaxCols($F); }

            $rows  = _rowSample($n, this.params_.subsample);
            $cols  = _colSample(this.n_feat_, this.params_.colsample);
            $G.set_features($cols);

            $round = [];
            $i = 0;
            while ($i < $k) {
                $src = $F[$i];
                if ($P != null) { $src = $P[$i]; }
                $gh = _gradHess(this.params_.objective, $src, $Y[$i]);
                $nodes = $G.grow($B, $gh[0], $gh[1], $rows);
                append($round, $nodes);
                // Shrinkage. The leaf already holds the Newton step
                // −G/(H+λ); the learning rate is what makes a hundred of them
                // an ensemble rather than a hundred attempts at the same step.
                $F[$i] = np.add($F[$i],
                                np.multiply(tree_predict($nodes, $Xtr),
                                            this.params_.learning_rate, null),
                                null);
                if ($Fva != null) {
                    $Fva[$i] = np.add($Fva[$i],
                                      np.multiply(tree_predict($nodes, $Xva),
                                                  this.params_.learning_rate, null),
                                      null);
                }
                $i = $i + 1;
            }
            append(this.rounds_, $round);
            $r = $r + 1;

            if ($Fva == null) { $bestAt = $r; }
            else {
                $loss = this._valLoss($Fva, $yva);
                append(this.val_scores_, $loss);
                // Strictly better by more than tol, or the round does not count
                // as progress. `>=` here would let a run drift forever on
                // improvements of 1e-17.
                if ($best == null || $loss < $best - this.params_.tol) {
                    $best   = $loss;
                    $bestAt = $r;
                    $stale  = 0;
                } else {
                    $stale = $stale + 1;
                }
                if ($stale >= this.params_.n_iter_no_change) { break; }
            }
        }

        // The trees AFTER the best round are kept rather than discarded, and
        // best_round_ bounds prediction. Discarding them would make
        // staged_predict unable to show what stopping avoided, and keeping them
        // costs a node array each.
        this.best_round_ = $bestAt;
        this.gains_      = $G.gains_;
        this.splits_     = $G.splits_;
        this.fitted_     = true;
        return this;
    }

    // ── prediction ─────────────────────────────────────────────────────────
    // The raw score columns at round $upto (null = best_round_).
    def _rawCols($X, $upto, $who) {
        this._check_fitted($who);
        $A = base._asarray($X);
        if (len(np.shape($A)) != 2 || np.shape($A)[1] != this.n_feat_) {
            throw $who + ": X must have " + str(this.n_feat_) + " features, got "
                  + "shape " + str(np.shape($A));
        }
        $r = $upto;
        if ($r == null) { $r = this.best_round_; }
        if (type($r) != "number" || $r != floor($r) || $r < 0
            || $r > len(this.rounds_)) {
            throw $who + ": round " + str($upto) + " is out of range; this model "
                  + "has " + str(len(this.rounds_)) + " rounds";
        }
        $n = np.shape($A)[0];
        $F = [];
        $i = 0;
        while ($i < this.n_out_) {
            append($F, np.add(np.zeros([$n], "f64"), this.init_[$i], null));
            $i = $i + 1;
        }
        $j = 0;
        while ($j < $r) {
            $round = this.rounds_[$j];
            $i = 0;
            while ($i < this.n_out_) {
                $F[$i] = np.add($F[$i],
                                np.multiply(tree_predict($round[$i], $A),
                                            this.params_.learning_rate, null),
                                null);
                $i = $i + 1;
            }
            $j = $j + 1;
        }
        return $F;
    }

    // ── introspection ──────────────────────────────────────────────────────
    def n_rounds()    { this._check_fitted("n_rounds");    return len(this.rounds_); }
    def best_round()  { this._check_fitted("best_round");  return this.best_round_; }
    def n_outputs()   { this._check_fitted("n_outputs");   return this.n_out_; }
    def val_scores()  { this._check_fitted("val_scores");  return this.val_scores_; }
    def n_train()     { this._check_fitted("n_train");     return this.n_train_; }
    def n_bins()      { this._check_fitted("n_bins");      return this.nbins_; }

    // Every tree in the model, summed.
    def n_nodes() {
        this._check_fitted("n_nodes");
        $t = 0;
        $j = 0;
        while ($j < len(this.rounds_)) {
            $i = 0;
            while ($i < len(this.rounds_[$j])) {
                $t = $t + np.shape(this.rounds_[$j][$i])[0];
                $i = $i + 1;
            }
            $j = $j + 1;
        }
        return $t;
    }

    // The node array for one tree, by round and output. A booster is a list of
    // matrices and nothing else, which is what makes K11's save/load trivial.
    def nodes($round, $which) {
        this._check_fitted("nodes");
        $w = $which;
        if ($w == null) { $w = 0; }
        if (type($round) != "number" || $round != floor($round) || $round < 0
            || $round >= len(this.rounds_)) {
            throw this.name_ + ".nodes: round " + str($round) + " is out of "
                  + "range; this model has " + str(len(this.rounds_));
        }
        if (type($w) != "number" || $w != floor($w) || $w < 0
            || $w >= this.n_out_) {
            throw this.name_ + ".nodes: output " + str($which) + " is out of "
                  + "range; this model has " + str(this.n_out_);
        }
        return this.rounds_[$round][$w];
    }

    // Per-feature importance, NORMALISED to sum to 1 as scikit-learn's is.
    // "gain" totals the gain of every split on a feature; "split" counts them,
    // which a single decisive split and ten marginal ones look identical under
    // and which is therefore reported separately rather than averaged in.
    // A model with no splits at all returns zeros rather than dividing ([K57]).
    def feature_importances_($kind) {
        this._check_fitted("feature_importances_");
        $k = $kind;
        if ($k == null) { $k = "gain"; }
        if ($k != "gain" && $k != "split") {
            throw this.name_ + ".feature_importances_: kind must be \"gain\" or "
                  + "\"split\", got " + str($kind);
        }
        $v = this.gains_;
        if ($k == "split") { $v = this.splits_; }
        $tot = _scalar(np.sum($v, null, false));
        if (!($tot > 0.0)) { return np.zeros([np.size($v)], "f64"); }
        return np.divide($v, $tot, null);
    }

    // A fresh, unfitted copy with the same parameters. The dict is REBUILT, not
    // handed over: Bantu assigns dicts by reference, so sharing it would make
    // set_params on the clone change the original -- which is the one thing
    // GridSearchCV must never do.
    def _paramsCopy() {
        $d = {};
        for $kk, $vv in this.params_.items() { $d[$kk] = $vv; }
        return $d;
    }
}

// ── regressor ───────────────────────────────────────────────────────────────
class _GradientBoostingRegressor extends _BoostBase {
    def init($params) {
        $obj = "squared_error";
        if ($params != null && contains($params.keys(), "objective")) {
            $obj = $params.objective;
        }
        super($params, $obj, "GradientBoostingRegressor", "regressor");
        if (!contains(["squared_error", "poisson"], this.params_.objective)) {
            throw "GradientBoostingRegressor: objective '"
                  + str(this.params_.objective) + "' is a classification "
                  + "objective; use GradientBoostingClassifier";
        }
    }

    def _prepareTarget($v, $who) {
        if (this.params_.objective == "poisson") {
            if (_scalar(np.min($v, null, false)) < 0.0) {
                throw $who + ": the poisson objective needs y >= 0 (it models a "
                      + "count), and this y has a negative value";
            }
        }
        return null;
    }

    def predict($X) {
        $F = this._rawCols($X, null, "GradientBoostingRegressor.predict");
        if (this.params_.objective == "poisson") {
            return np.exp($F[0], null);
        }
        return $F[0];
    }

    // The prediction after each round, as a list. What early stopping avoided
    // is visible here rather than argued for.
    //
    // Accumulated in ONE pass. Calling _rawCols(X, r) for r = 1..R would redo
    // every earlier round's tree_predict each time -- R(R+1)/2 traversals
    // instead of R, i.e. 5,050 instead of 100 for a default booster. Measured
    // on 2000x10, 100 rounds, depth 3: the whole staged sequence costs **13 ms
    // against 12 ms for a single predict**, because both do exactly 100
    // traversals. The quadratic version would have been ~50x that.
    //
    // No defensive copy of $F: np.add returns a NEW array, so the rebind leaves
    // every entry already appended untouched. Copying would imply otherwise,
    // and the suite's "the error falls" assertion would fail if the entries
    // aliased one buffer.
    def staged_predict($X) {
        $who = "staged_predict";
        this._check_fitted($who);
        $A = base._asarray($X);
        if (len(np.shape($A)) != 2 || np.shape($A)[1] != this.n_feat_) {
            throw $who + ": X must have " + str(this.n_feat_) + " features, got "
                  + "shape " + str(np.shape($A));
        }
        $F = np.add(np.zeros([np.shape($A)[0]], "f64"), this.init_[0], null);
        $out = [];
        $j = 0;
        while ($j < len(this.rounds_)) {
            $F = np.add($F, np.multiply(tree_predict(this.rounds_[$j][0], $A),
                                        this.params_.learning_rate, null), null);
            if (this.params_.objective == "poisson") {
                append($out, np.exp($F, null));
            } else {
                append($out, $F);
            }
            $j = $j + 1;
        }
        return $out;
    }

    // Held-out loss for early stopping: mean squared error on the raw score,
    // or the poisson deviance proxy e^F − yF, which is what the objective is
    // actually minimising. Scoring MSE on a poisson model would stop on a
    // different criterion from the one the trees were grown for.
    def _valLoss($Fva, $yva) {
        if (this.params_.objective == "poisson") {
            $e = np.exp($Fva[0], null);
            return _scalar(np.mean(np.subtract($e,
                           np.multiply($yva, $Fva[0], null), null), null, false));
        }
        $d = np.subtract($Fva[0], $yva, null);
        return _scalar(np.mean(np.multiply($d, $d, null), null, false));
    }

    def clone() { return GradientBoostingRegressor(this._paramsCopy()); }
}
def GradientBoostingRegressor($params) {
    return new _GradientBoostingRegressor($params);
}

// ── classifier ──────────────────────────────────────────────────────────────
// Binary: one tree per round, logistic. Multiclass: K trees per round over the
// SOFTMAX objective -- which is a real difference from K7's one-vs-rest trees
// ([K43]), and the better one: the K columns are normalised jointly, so a
// gradient for class k knows what the other classes predicted.
class _GradientBoostingClassifier extends _BoostBase {
    def init($params) {
        $obj = "logistic";
        if ($params != null && contains($params.keys(), "objective")) {
            $obj = $params.objective;
        }
        super($params, $obj, "GradientBoostingClassifier", "classifier");
        if (!contains(["logistic", "softmax"], this.params_.objective)) {
            throw "GradientBoostingClassifier: objective '"
                  + str(this.params_.objective) + "' is a regression objective; "
                  + "use GradientBoostingRegressor";
        }
        this.classes_ = null;
        this.explicitObj_ = false;
        if ($params != null && contains($params.keys(), "objective")) {
            this.explicitObj_ = true;
        }
    }

    def _prepareTarget($v, $who) {
        $u = np.unique($v);
        $k = np.size($u);
        if ($k < 2) {
            throw $who + ": y has " + str($k) + " distinct label(s). A classifier "
                  + "needs at least two, and a model fitted on one would score "
                  + "100% on anything.";
        }
        if ($k > 1000) {
            throw $who + ": y has " + str($k) + " distinct labels, which is a "
                  + "regression target rather than a class set";
        }
        this.classes_ = $u;
        // The objective follows the class count unless the caller named one.
        // Silently running `logistic` on a 5-class target would train against
        // the label VALUES as if they were probabilities.
        if (!this.explicitObj_) {
            if ($k == 2) { this.params_.objective = "logistic"; }
            else         { this.params_.objective = "softmax"; }
        }
        if ($k > 2 && this.params_.objective == "logistic") {
            throw $who + ": objective 'logistic' needs exactly two classes, and "
                  + "y has " + str($k) + ". Use 'softmax'.";
        }
        return null;
    }

    def _targets($yv) {
        $k = np.size(this.classes_);
        if (this.params_.objective == "logistic") {
            return [np.astype(np.equal($yv, np.get(this.classes_, [1]), null), "f64")];
        }
        $out = [];
        $i = 0;
        while ($i < $k) {
            append($out, np.astype(np.equal($yv, np.get(this.classes_, [$i]),
                                            null), "f64"));
            $i = $i + 1;
        }
        return $out;
    }

    def predict_proba($X) {
        $who = "GradientBoostingClassifier.predict_proba";
        $F = this._rawCols($X, null, $who);
        $n = np.shape(base._asarray($X))[0];
        if (this.params_.objective == "logistic") {
            $p1 = _sigmoid($F[0]);
            return np.concat([np.reshape(np.subtract(1.0, $p1, null), [$n, 1]),
                              np.reshape($p1, [$n, 1])], 1);
        }
        $P = _softmaxCols($F);
        $cols = [];
        $i = 0;
        while ($i < len($P)) {
            append($cols, np.reshape($P[$i], [$n, 1]));
            $i = $i + 1;
        }
        return np.concat($cols, 1);
    }

    def predict($X) {
        $P = this.predict_proba($X);
        $am = np.argmax($P, 1, false);
        return np.take(this.classes_, np.astype($am, "i64"), 0);
    }

    // Held-out loss: the log loss the objective minimises, not accuracy.
    // Accuracy is a step function of the scores, so it plateaus while the model
    // is still improving and would stop a run that had not finished.
    def _valLoss($Fva, $yva) {
        $eps = 1.0e-15;
        if (this.params_.objective == "logistic") {
            $p = np.clip(_sigmoid($Fva[0]), $eps, 1.0 - $eps, null);
            $t = np.astype(np.equal($yva, np.get(this.classes_, [1]), null), "f64");
            $a = np.multiply($t, np.log($p, null), null);
            $b = np.multiply(np.subtract(1.0, $t, null),
                             np.log(np.subtract(1.0, $p, null), null), null);
            return 0.0 - _scalar(np.mean(np.add($a, $b, null), null, false));
        }
        $P = _softmaxCols($Fva);
        $tot = null;
        $i = 0;
        while ($i < len($P)) {
            $t = np.astype(np.equal($yva, np.get(this.classes_, [$i]), null), "f64");
            $term = np.multiply($t, np.log(np.clip($P[$i], $eps, 1.0, null), null),
                                null);
            if ($tot == null) { $tot = $term; }
            else { $tot = np.add($tot, $term, null); }
            $i = $i + 1;
        }
        return 0.0 - _scalar(np.mean($tot, null, false));
    }

    def _stratifyOn($v) { return $v; }

    def classes() { this._check_fitted("classes"); return this.classes_; }

    def clone() { return GradientBoostingClassifier(this._paramsCopy()); }
}
def GradientBoostingClassifier($params) {
    return new _GradientBoostingClassifier($params);
}

def version() { return "bkit.ensemble 0.8.0 (gradient boosting over tree_*)"; }
