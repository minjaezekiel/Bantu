// ════════════════════════════════════════════════════════════════════════════
//  bkit/tree.b — decision trees over the `tree_` native family.
//
//      include "bkit/tree.b" as tree;
//      $m = tree.DecisionTreeRegressor(5, 1, null, null, null, null);
//      $m.fit($X, $y);
//      $m.predict($Xtest);
//
//  ── one criterion, two estimators ─────────────────────────────────────────
//  Both the regressor and the classifier set `g = -y` and `h = 1` and use the
//  SAME regularised gain the gradient booster will use in K8:
//
//      gain = ½[ G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ) ] − γ
//
//  That is not a compromise in either direction, and the reason is worth
//  writing down because it is what lets one code path serve both:
//
//  **For regression** it IS variance reduction. With g = −y, h = 1 and λ = 0,
//  G = −Σy and H = n, so G²/H = n·ȳ², and the gain is
//  ½[n_L ȳ_L² + n_R ȳ_R² − n ȳ²] — scikit-learn's `squared_error` criterion
//  times ½. The leaf value −G/(H+λ) is Σy/(n+λ), i.e. the mean at λ = 0.
//
//  **For binary classification it is proportional to Gini decrease.** With
//  y ∈ {0,1}, p = Σy/n, the same expression is ½[n_L p_L² + n_R p_R² − n p²].
//  Writing the weighted Gini decrease out and using n_L p_L + n_R p_R = n p:
//
//      Σ n_m·Gini_m decrease = 2[n_L p_L² + n_R p_R² − n p²] = 4 × gain
//
//  A positive constant factor, so the **argmax is identical** — this picks the
//  same split scikit-learn's `criterion="gini"` picks. The leaf value is the
//  positive-class proportion, which is the probability. Asserted against a
//  hand-computed Gini table in the test rather than left as an argument.
//
//  λ defaults to **0** here, not to the booster's 1: a decision tree is
//  expected to report the mean of its leaf, and λ > 0 shrinks it toward zero.
//
//  ── multiclass is one-vs-rest, and that is a real difference ──────────────
//  scikit-learn grows ONE tree whose leaves hold a K-way distribution. This
//  grows K trees. The predictions differ, and the docs say so rather than
//  claiming a match. It is the consistent choice here because the ecosystem's
//  own GBDT design already uses one tree per class per round (design doc §8.4),
//  so multiclass boosting and multiclass trees share a shape. A true
//  multiclass Gini split needs the gain summed over K histograms *before* the
//  argmax, which is a `tree_best_split` that takes a list — recorded as a K8
//  item, not faked here.
//
//  Design: docs/bkit-architecture.md §8; decisions K42-K46.
// ════════════════════════════════════════════════════════════════════════════

include "../numba/numba.b" as np;
include "./base.b" as base;

// −∞ as a threshold. Bantu raises on scalar division by zero, but log(0) on an
// ARRAY is −inf, which is the one portable way to name it here. Used for the
// missing-vs-present split, where no present value may go left.
def _negInf() {
    return np.get(np.log(np.zeros([1], "f64"), null), [0]);
}

// A full reduction returns a 0-d ndarray HANDLE, and every handle is truthy
// (numba N41), so it has to be read out rather than used as a number.
def _scalar($x) {
    return np.get(np.reshape($x, [1]), [0]);
}

def _checkDepth($d) {
    if ($d == null) { return 6; }
    if (type($d) != "number" || $d != floor($d) || $d < 1) {
        throw "bkit.tree: max_depth must be a whole number >= 1, got " + str($d);
    }
    if ($d > 60) {
        throw "bkit.tree: max_depth " + str($d) + " would allow 2^" + str($d)
              + " nodes. The cap is 60; a tree that deep is memorising rows.";
    }
    return $d;
}

def _checkPos($v, $dflt, $name, $allowZero) {
    if ($v == null) { return $dflt; }
    if (type($v) != "number") {
        throw "bkit.tree: " + $name + " must be a number, got " + type($v);
    }
    if ($allowZero && $v < 0.0) {
        throw "bkit.tree: " + $name + " must be >= 0, got " + str($v);
    }
    if (!$allowZero && $v <= 0.0) {
        throw "bkit.tree: " + $name + " must be > 0, got " + str($v);
    }
    return $v;
}

// ── the shared growth loop ──────────────────────────────────────────────────
class _TreeBase extends Estimator {
    def init($max_depth, $min_samples_leaf, $max_bins, $reg_lambda, $gamma,
             $min_child_weight) {
        super();
        this.params_ = {
            "max_depth":        _checkDepth($max_depth),
            "min_samples_leaf": _checkPos($min_samples_leaf, 1, "min_samples_leaf", false),
            "max_bins":         _checkPos($max_bins, 256, "max_bins", false),
            "reg_lambda":       _checkPos($reg_lambda, 0.0, "reg_lambda", true),
            "gamma":            _checkPos($gamma, 0.0, "gamma", true),
            "min_child_weight": _checkPos($min_child_weight, 0.0, "min_child_weight", true)
        };
        if (this.params_.min_samples_leaf != floor(this.params_.min_samples_leaf)) {
            throw "bkit.tree: min_samples_leaf must be a whole number, got "
                  + str(this.params_.min_samples_leaf);
        }
        if (this.params_.max_bins != floor(this.params_.max_bins)
            || this.params_.max_bins < 2 || this.params_.max_bins > 256) {
            throw "bkit.tree: max_bins must be a whole number in 2..256 (one "
                  + "byte per cell is the point), got " + str(this.params_.max_bins);
        }
        this.nodes_  = null;    // [n_nodes, 7] once fitted
        this.trees_  = null;    // K node arrays, for multiclass one-vs-rest
        this.n_feat_ = 0;
        this.depth_  = 0;
        this.leaves_ = 0;
        // Per-feature split gain and split count for the LAST tree _grow grew.
        // They are a side output of the grower rather than a column of the node
        // array, and that is deliberate: a node is 7 doubles and tree_predict
        // strides by 7 in C++, so an 8th column would be a native change and a
        // breaking change to every saved model, to carry a number prediction
        // never reads. The caller accumulates across trees (K57).
        this.lastGains_  = null;
        this.lastSplits_ = null;
        this.gains_      = null;   // accumulated over every tree in the model
        this.splits_     = null;
        // Column subsampling, per §8.5: a [f,1,1] mask of 1s and 0s multiplied
        // into the histogram the SCAN sees. A zeroed feature has G = H = n = 0
        // in every bin, so every cut on it has n_L = 0 < min_data AND fails the
        // hessian guard -- two independent reasons to skip it, neither of which
        // is a special case in the kernel. null means every feature.
        //
        // The mask is applied to a COPY handed to tree_best_split, never to
        // $hist itself: $hist is what _totals reads and what the children are
        // derived from by subtraction, and masking it would make a child's
        // histogram the difference of a masked and an unmasked one.
        this.featMask_ = null;
        // Features considered PER SPLIT. null means "use featMask_ as given,
        // or every feature". A random forest's defining behaviour is that this
        // is redrawn at every NODE (Breiman 2001) -- per-tree sampling is a
        // random-subspace ensemble, which is a different and weaker estimator:
        // with f = 6 and k = 2, two thirds of the trees never see an
        // informative feature at all. A booster's colsample is per tree on
        // purpose (XGBoost's colsample_bytree), so the two use different hooks
        // rather than one switch pretending to serve both ([K63]).
        this.featK_ = null;
    }

    def _prepare($X, $y, $who) {
        $A = base._asarray($X);
        if (len(np.shape($A)) != 2) {
            throw $who + ": X must be 2-d [n_samples, n_features], got shape "
                  + str(np.shape($A));
        }
        if (np.shape($A)[0] == 0) {
            throw $who + ": X has no rows";
        }
        if ($y == null) { throw $who + ": y is required"; }
        $v = base._asarray($y);
        if (len(np.shape($v)) != 1) {
            $v = np.reshape($v, [np.size($v)]);
        }
        if (np.shape($v)[0] != np.shape($A)[0]) {
            throw $who + ": y has " + str(np.shape($v)[0]) + " elements and X has "
                  + str(np.shape($A)[0]) + " rows";
        }
        return [$A, $v];
    }

    // The node's own totals, read off the histogram rather than recomputed:
    // every feature's bins sum to the same (G, H, n), so feature 0 answers it
    // in one vectorised call.
    def _totals($hist) {
        $f0 = np.reshape(np.take($hist, np.astype(np.array([0], "f64"), "i64"), 0),
                         [np.shape($hist)[1], 3]);
        $s  = np.sum($f0, 0, false);
        return [np.get($s, [0]), np.get($s, [1]), np.get($s, [2])];
    }

    // THE SAME GUARD tree_best_split applies to the gain, applied to the leaf.
    // The kernel refuses a child with !(H + λ > 0) because G²/0 with G ≠ 0 is
    // +inf, which wins every comparison (K45). The leaf value is computed from
    // the same two quantities and had no guard at all, so a node whose hessians
    // sum to zero raised "Division by zero" out of the middle of _grow -- a
    // scalar /0 raises in Bantu, unlike the array case.
    //
    // Not reachable from K7's own estimators: both set h = 1 per row, so
    // H = n ≥ 1. It is reachable from K8's booster, where logistic h = p(1−p)
    // underflows to 0 across a saturated node and reg_lambda defaults to 0 —
    // i.e. the first thing K8 does with this grower would have hit it. Fixing
    // it in the grower rather than in the booster is the point: there is one
    // grower, and the asymmetry (kernel guards, caller does not) is the defect.
    //
    // Zero is the right value, not a fallback. −G/(H+λ) is the minimiser of the
    // second-order loss approximation; at H + λ = 0 that approximation has no
    // curvature and the minimiser is unbounded, which is an artefact of the
    // approximation rather than a step worth taking. XGBoost's CalcWeight
    // returns 0 in exactly this case and LightGBM does the same. Written
    // !(x > 0.0) so a negative or NaN total is caught too, matching the kernel
    // character for character.
    def _leafValue($G, $H) {
        $d = $H + this.params_.reg_lambda;
        if (!($d > 0.0)) { return 0.0; }
        return 0.0 - $G / $d;
    }

    // Grows one tree over (g, h) and returns the [n_nodes, 7] node array.
    // Breadth-first over an explicit queue: Bantu's call depth is capped at
    // 1000 and a recursive grower would also make the node indices awkward,
    // since a child's slot has to exist before its parent can point at it.
    def _grow($B, $g, $h, $rootIdx) {
        $shape = tree_bin_shape($B);
        $nrow  = np.get($shape, [0]);
        $ncol  = np.get($shape, [1]);

        // Per-feature gain and split count for THIS tree. Built as Bantu lists
        // because there are n_features of them and one += per split, not per
        // element -- a tree with 1,000 splits does 1,000 of these.
        $gainF  = [];
        $splitF = [];
        $fi = 0;
        while ($fi < $ncol) { append($gainF, 0.0); append($splitF, 0.0); $fi = $fi + 1; }

        $nodes = [];                 // each a 7-element list
        // [idx, depth, slot, hist] -- hist is this node's histogram, carried
        // from its parent by subtraction (see below) or null for the root.
        $pend  = [];
        append($nodes, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        // $rootIdx null means every row. A booster passes a row subsample and a
        // forest passes a BOOTSTRAP sample -- an index array with repeats, which
        // tree_hist counts once per occurrence, i.e. bootstrap semantics with no
        // kernel change (§8.5).
        append($pend,  [$rootIdx, 0, 0, null]);
        $qi = 0;
        $maxDepth = 0;
        $leaves = 0;

        while ($qi < len($pend)) {
            $item = $pend[$qi];
            $qi   = $qi + 1;
            $idx   = $item[0];
            $depth = $item[1];
            $slot  = $item[2];
            if ($depth > $maxDepth) { $maxDepth = $depth; }

            // HISTOGRAM SUBTRACTION. A child's histogram is its parent's minus
            // its sibling's, so only one child per split is built from the data
            // and the other is one vectorised subtract. Computing the SMALLER
            // child directly roughly halves the work per level, and it is the
            // single largest optimisation in the algorithm.
            //
            // It is NOT bit-identical to building the child directly, and the
            // design doc's first draft claimed it would be: tree_hist
            // accumulates in index order, so the parent sums rows 0..n-1 while
            // the sibling sums a subsequence, and sum(A u B) - sum(A) = sum(B)
            // does not hold exactly for floating point. The COUNT channel is
            // exact (counts are integers in doubles); the gradient and hessian
            // channels agree to a tight relative tolerance, which is what
            // LightGBM and XGBoost both accept. The error does not accumulate
            // across levels, because each child derives from its parent rather
            // than from a running total.
            $hist = $item[3];
            if ($hist == null) { $hist = tree_hist($B, $g, $h, $idx); }
            $tot  = this._totals($hist);
            $G = $tot[0];  $H = $tot[1];  $n = $tot[2];

            // Written as a leaf first, then overwritten if it splits. That way
            // a node is never left in a half-built state if a guard fires.
            $nodes[$slot] = [1.0, 0.0, 0.0, 0.0, 0.0, this._leafValue($G, $H), 0.0];

            if ($depth >= this.params_.max_depth) { $leaves = $leaves + 1; }
            if ($depth < this.params_.max_depth) {
                if ($n < 2.0 * this.params_.min_samples_leaf) {
                    $leaves = $leaves + 1;
                }
                if ($n >= 2.0 * this.params_.min_samples_leaf) {
                    // Redrawn per node when featK_ is set. Built with three
                    // native calls and no Bantu loop: the rank of a uniform
                    // draw is a uniform permutation, so "rank < k" selects k
                    // features without replacement (K18's construction).
                    if (this.featK_ != null) {
                        $rk = np.argsort(np.argsort(np.random([$ncol], 0.0, 1.0),
                                                    null), null);
                        this.featMask_ = np.reshape(
                            np.astype(np.less($rk, this.featK_, null), "f64"),
                            [$ncol, 1, 1]);
                    }
                    $scan = $hist;
                    if (this.featMask_ != null) {
                        $scan = np.multiply($hist, this.featMask_, null);
                    }
                    $bs = tree_best_split($scan, $B,
                                          this.params_.reg_lambda,
                                          this.params_.gamma,
                                          this.params_.min_child_weight,
                                          1.0 * this.params_.min_samples_leaf);
                    $feat = np.get($bs, [0]);
                    $bin  = np.get($bs, [1]);
                    $gain = np.get($bs, [2]);
                    $nanL = np.get($bs, [9]);
                    // A gain of zero is not an improvement; splitting on it
                    // doubles the tree for nothing and is how a "max_depth"
                    // tree ends up memorising singleton leaves.
                    if ($feat < 0 || $gain <= 0.0) { $leaves = $leaves + 1; }
                    if ($feat >= 0 && $gain > 0.0) {
                        $parts = tree_partition($B, this._idxOrAll($idx, $nrow),
                                                $feat, $bin, $nanL != 0.0);
                        $L = $parts[0];
                        $R = $parts[1];
                        // The guards above should make this unreachable; it is
                        // asserted anyway, because an empty child would make
                        // the child's histogram all zeros and its leaf value
                        // 0/λ -- a plausible-looking number from a node that
                        // represents nothing.
                        if (np.size($L) == 0 || np.size($R) == 0) {
                            throw "bkit.tree: the chosen split on feature "
                                  + str($feat) + " bin " + str($bin)
                                  + " left one child empty (" + str(np.size($L))
                                  + "/" + str(np.size($R)) + "). The gain scan "
                                  + "and the partition disagree about the NaN "
                                  + "direction or the bin ordering.";
                        }
                        $thr = this._threshold($B, $feat, $bin);
                        $li = len($nodes);
                        append($nodes, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
                        $ri = len($nodes);
                        append($nodes, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
                        $nodes[$slot] = [0.0, 1.0 * $feat, $thr, 1.0 * $li,
                                         1.0 * $ri, 0.0, $nanL];
                        // Recorded where the gain is still in hand. Reading it
                        // back off the node array afterwards is impossible --
                        // the node holds the threshold, not the gain.
                        $gainF[$feat]  = $gainF[$feat] + $gain;
                        $splitF[$feat] = $splitF[$feat] + 1.0;
                        // Build the smaller child, subtract for the larger.
                        $hL = null;
                        $hR = null;
                        if (np.size($L) <= np.size($R)) {
                            $hL = tree_hist($B, $g, $h, $L);
                            $hR = np.subtract($hist, $hL, null);
                        }
                        if (np.size($L) > np.size($R)) {
                            $hR = tree_hist($B, $g, $h, $R);
                            $hL = np.subtract($hist, $hR, null);
                        }
                        append($pend, [$L, $depth + 1, $li, $hL]);
                        append($pend, [$R, $depth + 1, $ri, $hR]);
                    }
                }
            }
        }
        this.depth_  = $maxDepth;
        this.leaves_ = $leaves;
        this.lastGains_  = np.array($gainF,  "f64");
        this.lastSplits_ = np.array($splitF, "f64");
        return np.array($nodes, "f64");
    }

    // tree_partition needs a real index array; null means "every row" only to
    // tree_hist.
    def _idxOrAll($idx, $nrow) {
        if ($idx != null) { return $idx; }
        return np.astype(np.arange(0.0, 1.0 * $nrow, 1.0), "i64");
    }

    // The split's threshold as a RAW X value, so prediction needs nothing but
    // X and the node array. bin b means "values <= edges[b]"; bin -1 is the
    // missing-vs-present split, where no present value may go left.
    def _threshold($B, $feat, $bin) {
        if ($bin < 0) { return _negInf(); }
        $e = tree_bin_edges($B, $feat);
        if (np.size($e) == 0) {
            throw "bkit.tree: feature " + str($feat) + " has no bin edges, so it "
                  + "is constant and cannot have been chosen as a split";
        }
        if ($bin >= np.size($e)) {
            throw "bkit.tree: split bin " + str($bin) + " on feature " + str($feat)
                  + " has no edge (" + str(np.size($e)) + " edges)";
        }
        return np.get($e, [$bin]);
    }

    def _check($who) {
        this._check_fitted($who);
        return null;
    }

    // How many trees this estimator holds: 1 for a regressor or a binary
    // classifier, K for a multiclass one-vs-rest classifier.
    def n_trees() {
        this._check("n_trees");
        if (this.trees_ == null) { return 1; }
        return len(this.trees_);
    }

    // The node array for one tree, defaulting to the first. A multiclass
    // classifier holds K of them, so this takes an index rather than pretending
    // there is one -- a fitted tree is one [n_nodes, 7] matrix and nothing
    // else, which is what makes K11's save/load trivial.
    def nodes($which) {
        this._check("nodes");
        $k = this.n_trees();
        $i = $which;
        if ($i == null) { $i = 0; }
        if (type($i) != "number" || $i != floor($i) || $i < 0 || $i >= $k) {
            throw this.name_ + ".nodes: tree index " + str($which)
                  + " is out of range; this estimator holds " + str($k);
        }
        if (this.trees_ == null) { return this.nodes_; }
        return this.trees_[$i];
    }

    // Summed over every tree. It used to report np.shape(this.nodes_)[0], and
    // nodes_ is aliased to trees_[0] for a multiclass classifier -- so a
    // 3-class model reported one third of its own size, which is a wrong
    // number rather than a missing one.
    def n_nodes() {
        this._check("n_nodes");
        if (this.trees_ == null) { return np.shape(this.nodes_)[0]; }
        $t = 0;
        $i = 0;
        while ($i < len(this.trees_)) {
            $t = $t + np.shape(this.trees_[$i])[0];
            $i = $i + 1;
        }
        return $t;
    }
    def depth()    { this._check("depth");    return this.depth_; }
    def n_leaves() { this._check("n_leaves"); return this.leaves_; }

    // Per-feature importance, NORMALISED to sum to 1 as scikit-learn's is.
    //
    //   kind = "gain"  (default) -- the total gain of every split on a feature,
    //                               which is what the model actually used it for
    //   kind = "split"           -- how many splits used it, which a single
    //                               high-gain split and ten marginal ones look
    //                               identical under, and which is therefore
    //                               reported separately rather than averaged in
    //
    // A model with NO splits (a single leaf -- a constant target, or every gain
    // non-positive) has total 0. That returns ZEROS rather than dividing: an
    // array /0 is NaN in numba and a vector of NaNs reads as "unknown" when the
    // truthful answer is "this model uses no feature at all" ([K57]).
    def feature_importances_($kind) {
        this._check("feature_importances_");
        $k = $kind;
        if ($k == null) { $k = "gain"; }
        if ($k != "gain" && $k != "split") {
            throw this.name_ + ".feature_importances_: kind must be \"gain\" or "
                  + "\"split\", got " + str($kind);
        }
        $v = this.gains_;
        if ($k == "split") { $v = this.splits_; }
        if ($v == null) {
            throw this.name_ + ".feature_importances_: this model recorded no "
                  + "importances, which should be impossible after a fit";
        }
        $tot = _scalar(np.sum($v, null, false));
        if (!($tot > 0.0)) { return np.zeros([np.size($v)], "f64"); }
        return np.divide($v, $tot, null);
    }

    // Accumulates one grown tree's side outputs into the model's totals. The
    // grower overwrites lastGains_ on every call, so a multiclass or boosted
    // model must fold them in after each _grow rather than reading them at the
    // end.
    def _accImportance() {
        if (this.gains_ == null) {
            this.gains_  = np.copy(this.lastGains_);
            this.splits_ = np.copy(this.lastSplits_);
            return null;
        }
        this.gains_  = np.add(this.gains_,  this.lastGains_,  null);
        this.splits_ = np.add(this.splits_, this.lastSplits_, null);
        return null;
    }
}

// ── regressor ───────────────────────────────────────────────────────────────
class _DecisionTreeRegressor extends _TreeBase {
    def init($max_depth, $min_samples_leaf, $max_bins, $reg_lambda, $gamma,
             $min_child_weight) {
        super($max_depth, $min_samples_leaf, $max_bins, $reg_lambda, $gamma,
              $min_child_weight);
        this.name_ = "DecisionTreeRegressor";
        this.task_ = "regressor";
    }
    def fit($X, $y) {
        $p = this._prepare($X, $y, "DecisionTreeRegressor.fit");
        $A = $p[0];  $v = $p[1];
        this.n_feat_ = np.shape($A)[1];
        $B = tree_bin($A, this.params_.max_bins);
        // g = -y, h = 1: see the header. The gain is then variance reduction
        // and the leaf value is the mean (at reg_lambda = 0).
        $g = np.multiply($v, 0.0 - 1.0, null);
        $h = np.ones([np.shape($A)[0]], "f64");
        this.gains_  = null;
        this.splits_ = null;
        this.nodes_  = this._grow($B, $g, $h, null);
        this._accImportance();
        this.fitted_ = true;
        return this;
    }
    def predict($X) {
        this._check_fitted("DecisionTreeRegressor.predict");
        $A = base._asarray($X);
        if (len(np.shape($A)) != 2 || np.shape($A)[1] != this.n_feat_) {
            throw "DecisionTreeRegressor.predict: X must have "
                  + str(this.n_feat_) + " features, got shape " + str(np.shape($A));
        }
        return tree_predict(this.nodes_, $A);
    }
    def clone() {
        return DecisionTreeRegressor(this.params_.max_depth,
            this.params_.min_samples_leaf, this.params_.max_bins,
            this.params_.reg_lambda, this.params_.gamma,
            this.params_.min_child_weight);
    }
}
def DecisionTreeRegressor($max_depth, $min_samples_leaf, $max_bins, $reg_lambda,
                          $gamma, $min_child_weight) {
    return new _DecisionTreeRegressor($max_depth, $min_samples_leaf, $max_bins,
                                      $reg_lambda, $gamma, $min_child_weight);
}

// ── classifier ──────────────────────────────────────────────────────────────
// Binary: one tree, leaf value = the positive-class proportion, which is the
// probability. Multiclass: one tree per class (one-vs-rest), argmax to predict
// -- a real difference from scikit-learn, stated in the header.
class _DecisionTreeClassifier extends _TreeBase {
    def init($max_depth, $min_samples_leaf, $max_bins, $reg_lambda, $gamma,
             $min_child_weight) {
        super($max_depth, $min_samples_leaf, $max_bins, $reg_lambda, $gamma,
              $min_child_weight);
        this.name_ = "DecisionTreeClassifier";
        this.task_ = "classifier";
        this.classes_ = null;
        this.trees_   = null;     // one node array per class when multiclass
    }

    def _labels($v, $who) {
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
        return $u;
    }

    def fit($X, $y) {
        $p = this._prepare($X, $y, "DecisionTreeClassifier.fit");
        $A = $p[0];  $v = $p[1];
        this.n_feat_  = np.shape($A)[1];
        this.classes_ = this._labels($v, "DecisionTreeClassifier.fit");
        $k = np.size(this.classes_);
        $B = tree_bin($A, this.params_.max_bins);
        $h = np.ones([np.shape($A)[0]], "f64");
        this.gains_  = null;
        this.splits_ = null;

        if ($k == 2) {
            // y mapped to 0/1 against classes_[1]; g = -y, h = 1 makes the gain
            // proportional to Gini decrease (header) and the leaf the positive
            // proportion.
            $pos = np.astype(np.equal($v, np.get(this.classes_, [1]), null), "f64");
            this.nodes_ = this._grow($B, np.multiply($pos, 0.0 - 1.0, null), $h, null);
            this._accImportance();
            this.trees_ = null;
        }
        if ($k != 2) {
            this.trees_ = [];
            $i = 0;
            $deepest = 0;
            $allLeaves = 0;
            while ($i < $k) {
                $pos = np.astype(np.equal($v, np.get(this.classes_, [$i]), null), "f64");
                append(this.trees_,
                       this._grow($B, np.multiply($pos, 0.0 - 1.0, null), $h, null));
                // Folded in per tree: _grow overwrites lastGains_ on every call,
                // so a K-class model that read them at the end would report only
                // its last class's splits.
                this._accImportance();
                if (this.depth_ > $deepest) { $deepest = this.depth_; }
                $allLeaves = $allLeaves + this.leaves_;
                $i = $i + 1;
            }
            this.depth_  = $deepest;
            this.leaves_ = $allLeaves;
            this.nodes_  = this.trees_[0];
        }
        this.fitted_ = true;
        return this;
    }

    def _probaRaw($X, $who) {
        this._check_fitted($who);
        $A = base._asarray($X);
        if (len(np.shape($A)) != 2 || np.shape($A)[1] != this.n_feat_) {
            throw $who + ": X must have " + str(this.n_feat_)
                  + " features, got shape " + str(np.shape($A));
        }
        $k = np.size(this.classes_);
        if ($k == 2) {
            $p1 = np.clip(tree_predict(this.nodes_, $A), 0.0, 1.0, null);
            return np.concat([np.reshape(np.subtract(1.0, $p1, null), [np.shape($A)[0], 1]),
                              np.reshape($p1, [np.shape($A)[0], 1])], 1);
        }
        $cols = [];
        $i = 0;
        while ($i < $k) {
            append($cols, np.reshape(np.clip(tree_predict(this.trees_[$i], $A),
                                             0.0, 1.0, null),
                                     [np.shape($A)[0], 1]));
            $i = $i + 1;
        }
        $P = np.concat($cols, 1);
        // One-vs-rest scores do not sum to 1, so they are normalised to be
        // read as probabilities. A row of all zeros -- every tree says "not
        // mine" -- would divide by zero, so it falls back to uniform rather
        // than producing NaN.
        $s = np.sum($P, 1, true);
        $s = np.where(np.equal($s, 0.0, null), 1.0, $s);
        $P = np.divide($P, $s, null);
        $allZero = np.equal(np.sum($P, 1, true), 0.0, null);
        return np.where($allZero, 1.0 / (1.0 * $k), $P);
    }

    def predict_proba($X) {
        return this._probaRaw($X, "DecisionTreeClassifier.predict_proba");
    }

    def predict($X) {
        $P = this._probaRaw($X, "DecisionTreeClassifier.predict");
        $am = np.argmax($P, 1, false);
        return np.take(this.classes_, np.astype($am, "i64"), 0);
    }

    def clone() {
        return DecisionTreeClassifier(this.params_.max_depth,
            this.params_.min_samples_leaf, this.params_.max_bins,
            this.params_.reg_lambda, this.params_.gamma,
            this.params_.min_child_weight);
    }
}
def DecisionTreeClassifier($max_depth, $min_samples_leaf, $max_bins, $reg_lambda,
                           $gamma, $min_child_weight) {
    return new _DecisionTreeClassifier($max_depth, $min_samples_leaf, $max_bins,
                                       $reg_lambda, $gamma, $min_child_weight);
}

// ── the grower, as a reusable object ────────────────────────────────────────
// K8's booster grows a tree over arbitrary (g, h) rather than over a target,
// and §8.6 of the design is explicit that it must be THE SAME grower: a second
// one inside ensemble.b would be a second place that has to get [K44]'s NaN
// direction, the empty-child assertion and the raw-value threshold lookup
// right, and one place already got the NaN direction subtly wrong.
//
// So the grower is EXPOSED rather than copied. _TreeGrower adds no growth logic
// at all -- it inherits _grow unchanged and only validates its arguments, which
// the estimators do not have to because they build g and h themselves.
//
// It is not an Estimator in the protocol sense: fit/predict are refused by name
// rather than left to fail somewhere unrelated, because a grower has no target
// and nothing to predict with on its own.
class _TreeGrower extends _TreeBase {
    def init($max_depth, $min_samples_leaf, $max_bins, $reg_lambda, $gamma,
             $min_child_weight) {
        super($max_depth, $min_samples_leaf, $max_bins, $reg_lambda, $gamma,
              $min_child_weight);
        this.name_ = "TreeGrower";
        this.task_ = "grower";
    }

    // Returns the [n_nodes, 7] node array for one tree over (g, h), and leaves
    // depth(), n_leaves(), n_nodes() and feature_importances_ reporting it.
    //
    // $rootIdx null grows over every row; an index array grows over those rows
    // only, with repeats counted once each -- which is a row subsample for a
    // booster and a bootstrap sample for a forest. g and h stay FULL LENGTH
    // either way: they are indexed by row, not by position in the subsample,
    // and re-indexing them would be a copy per tree for no gain.
    def grow($B, $g, $h, $rootIdx) {
        if ($B == null) {
            throw "TreeGrower.grow: a binned matrix from tree_bin is required";
        }
        if ($g == null || $h == null) {
            throw "TreeGrower.grow: both g and h are required; a booster that "
                  + "passes only gradients is using the wrong grower";
        }
        $gv = base._asarray($g);
        $hv = base._asarray($h);
        $sh = tree_bin_shape($B);
        $n  = np.get($sh, [0]);
        if (np.size($gv) != $n || np.size($hv) != $n) {
            throw "TreeGrower.grow: the binned matrix has " + str($n)
                  + " rows but g has " + str(np.size($gv)) + " and h has "
                  + str(np.size($hv));
        }
        if (len(np.shape($gv)) != 1 || len(np.shape($hv)) != 1) {
            throw "TreeGrower.grow: g and h must be 1-d, got "
                  + str(np.shape($gv)) + " and " + str(np.shape($hv));
        }
        this.n_feat_ = np.get($sh, [1]);
        this.nodes_  = this._grow($B, $gv, $hv, $rootIdx);
        this._accImportance();
        this.fitted_ = true;
        return this.nodes_;
    }

    // The columns this grower may split on, as a list of feature indices. null
    // restores all of them. Stored as a [f,1,1] mask so the histogram multiply
    // broadcasts (numba tier 1b), rather than as a list the scan would have to
    // consult per bin.
    def set_features($cols) {
        this.featK_ = null;
        if ($cols == null) { this.featMask_ = null; return null; }
        if (this.n_feat_ == 0) {
            throw "TreeGrower.set_features: the number of features is not known "
                  + "until the first grow; pass n_features to known_features() "
                  + "first, or call set_features after a grow";
        }
        $m = [];
        $i = 0;
        while ($i < this.n_feat_) { append($m, 0.0); $i = $i + 1; }
        $j = 0;
        while ($j < len($cols)) {
            $c = $cols[$j];
            if (type($c) != "number" || $c != floor($c) || $c < 0
                || $c >= this.n_feat_) {
                throw "TreeGrower.set_features: " + str($c) + " is not a feature "
                      + "index in 0.." + str(this.n_feat_ - 1);
            }
            if ($m[$c] != 0.0) {
                throw "TreeGrower.set_features: feature " + str($c) + " is listed "
                      + "twice; a repeated column does not weight it, it just "
                      + "hides the typo";
            }
            $m[$c] = 1.0;
            $j = $j + 1;
        }
        if (len($cols) == 0) {
            throw "TreeGrower.set_features: an empty column list would grow a "
                  + "single leaf for every tree; pass null for all features";
        }
        this.featMask_ = np.reshape(np.array($m, "f64"), [this.n_feat_, 1, 1]);
        return null;
    }

    // How many features each SPLIT may consider, redrawn per node. null turns it
    // off and leaves set_features() in charge. This is the random forest's hook;
    // set_features() is the booster's, which samples once per tree ([K63]).
    def set_features_per_split($k) {
        if ($k == null) { this.featK_ = null; return null; }
        if (this.n_feat_ == 0) {
            throw "TreeGrower.set_features_per_split: the number of features is "
                  + "not known yet; call known_features() first";
        }
        if (type($k) != "number" || $k != floor($k) || $k < 1
            || $k > this.n_feat_) {
            throw "TreeGrower.set_features_per_split: expected a whole number in "
                  + "1.." + str(this.n_feat_) + ", got " + str($k);
        }
        if ($k == this.n_feat_) { this.featK_ = null; this.featMask_ = null; return null; }
        this.featK_ = $k;
        return null;
    }

    // Declare the feature count before the first grow, so set_features can be
    // called up front.
    def known_features($f) {
        if (type($f) != "number" || $f != floor($f) || $f < 1) {
            throw "TreeGrower.known_features: expected a whole number >= 1, got "
                  + str($f);
        }
        this.n_feat_ = $f;
        return null;
    }

    // Forget the accumulated importances without rebuilding the object, so a
    // booster can reuse one grower for every round and still report per-model
    // totals.
    def reset_importance() {
        this.gains_  = null;
        this.splits_ = null;
        return null;
    }

    def fit($X, $y) {
        throw "TreeGrower is not an estimator: it grows one tree over (g, h). "
              + "Use DecisionTreeRegressor/Classifier, or ensemble.b.";
    }
    def predict($X) {
        throw "TreeGrower does not predict: pass the node array it returns to "
              + "tree_predict(nodes, X).";
    }
    def clone() {
        return TreeGrower(this.params_.max_depth, this.params_.min_samples_leaf,
            this.params_.max_bins, this.params_.reg_lambda, this.params_.gamma,
            this.params_.min_child_weight);
    }
}
def TreeGrower($max_depth, $min_samples_leaf, $max_bins, $reg_lambda, $gamma,
               $min_child_weight) {
    return new _TreeGrower($max_depth, $min_samples_leaf, $max_bins, $reg_lambda,
                           $gamma, $min_child_weight);
}

def version() { return "bkit.tree 0.7.1 (decision trees over tree_*)"; }
