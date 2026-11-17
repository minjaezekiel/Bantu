// ════════════════════════════════════════════════════════════════════════════
//  bkit/cluster.b — KMeans, DBSCAN, GaussianMixture.
//
//      include "bkit/cluster.b" as cl;
//      $k = cl.KMeans({"n_clusters": 3});  $k.fit($X, null);  $k.labels();
//      $d = cl.DBSCAN({"eps": 0.5, "min_samples": 5});
//      $g = cl.GaussianMixture({"n_components": 3});
//
//  ── one GEMM per assignment step ──────────────────────────────────────────
//  ‖x − c‖² = ‖x‖² − 2 x·c + ‖c‖², so a whole assignment step is ONE matmul
//  plus two broadcasts. That is what makes k-means viable in Bantu at all: the
//  per-element work is native and the loop body is a handful of nd_* calls per
//  ITERATION rather than per point.
//
//  ‖x‖² does not depend on c, so it is dropped from the argmin entirely and
//  added back only when the inertia itself is wanted -- once per run, not once
//  per iteration.
//
//  ── seeded, and nothing carries RNG state ─────────────────────────────────
//  k-means++ and DBSCAN's tie-breaks draw from nd_seed's stream ([C4]), so a
//  fit is reproducible from the seed alone and no estimator holds a generator.
//
//  ── the quadratic form goes through the Cholesky, never Σ⁻¹ ───────────────
//  GaussianMixture needs (x−μ)ᵀ Σ⁻¹ (x−μ) and log det Σ. Both come from the
//  Cholesky factor: the form is ‖L⁻¹(x−μ)‖² and the log-determinant is twice
//  the sum of the log diagonal. Forming Σ⁻¹ costs accuracy for no gain -- the
//  same mistake as forming the Gram matrix in §6.
//
//  numba exposes no triangular solve (`nd_solve_triangular` is listed in its
//  architecture doc and does not exist -- numba N42), so this uses `nd_solve`
//  on the triangular factor: more flops than back-substitution, exact to the
//  same precision, and it still never forms the inverse, which is the property
//  that matters.
// ════════════════════════════════════════════════════════════════════════════

include "../numba/numba.b" as np;
include "./base.b" as base;

// A full reduction returns a 0-d ndarray HANDLE and every handle is truthy
// (numba N41), so it has to be read out rather than used as a number.
def _scalar($x) {
    return np.get(np.reshape($x, [1]), [0]);
}

def _num($v, $name, $who) {
    if (type($v) != "number") {
        throw $who + ": " + $name + " must be a number, got " + type($v)
              + " (" + str($v) + ")";
    }
    return $v;
}

def _whole($v, $lo, $name, $who) {
    _num($v, $name, $who);
    if ($v != floor($v) || $v < $lo) {
        throw $who + ": " + $name + " must be a whole number >= " + str($lo)
              + ", got " + str($v);
    }
    return $v;
}

// Row norms ‖x‖² as a column, so it broadcasts against an [n, k] distance
// matrix without a transpose.
def _rowSq($A) {
    return np.sum(np.multiply($A, $A, null), 1, true);
}

// SQUARED distances from every row of A to every row of C, as [n, k].
// ‖a‖² − 2a·c + ‖c‖², with the cross term as one matmul. The result is clipped
// at zero: the identity is exact in real arithmetic and can go slightly
// negative in floating point when a row and a centre coincide, and a negative
// "squared distance" propagates into a NaN the moment anyone takes its root.
def _dist2($A, $C) {
    $cross = np.matmul($A, np.transpose($C, null));
    $d = np.add(np.add(np.multiply($cross, 0.0 - 2.0, null),
                       np.reshape(_rowSq($C), [1, np.shape($C)[0]]), null),
                _rowSq($A), null);
    // Clipped at zero. The identity is exact in real arithmetic and can go
    // slightly negative in floating point when a row and a centre coincide,
    // and a negative "squared distance" becomes a NaN the moment anyone takes
    // its root.
    //
    // THE CLIP ALSO HID A SIGN ERROR, which is worth the warning: the first
    // version of this function SUBTRACTED ‖c‖² instead of adding it, every
    // distance came out negative, the clip turned them all into zeros, and the
    // only visible symptom was an inertia of exactly 0 -- from a fit that
    // otherwise completed and returned plausible-looking centres. A defensive
    // clamp makes a wrong answer quiet, so this function is now gated against
    // a brute-force reference rather than against anything downstream of it.
    return np.maximum($d, 0.0, null);
}

// ── k-means++ ───────────────────────────────────────────────────────────────
// The first centre uniformly at random, each subsequent one with probability
// proportional to D(x)² -- the squared distance to the nearest centre already
// chosen. That is what makes k-means++ a better start than a uniform draw, and
// it is why the draw must be WEIGHTED rather than uniform over the remaining
// points.
//
// The weighted draw is cumsum-and-search, which is two native calls and no
// Bantu loop over points. A row of all-zero weights (every point already a
// centre, i.e. fewer distinct points than clusters) falls back to a uniform
// draw instead of dividing by zero.
def _weightedDraw($w) {
    $n = np.size($w);
    $tot = _scalar(np.sum($w, null, false));
    if (!($tot > 0.0)) {
        return floor(_scalar(np.random([1], 0.0, 1.0)) * (1.0 * $n - 1.0e-12));
    }
    $cum = np.cumsum($w, 0);
    $u = _scalar(np.random([1], 0.0, 1.0)) * $tot;
    // How many cumulative entries are strictly below u: that index is the draw.
    $at = _scalar(np.sum(np.astype(np.less($cum, $u, null), "f64"), null, false));
    if ($at >= 1.0 * $n) { $at = 1.0 * $n - 1.0; }
    return $at;
}

def _kmeanspp($A, $k, $who) {
    $n = np.shape($A)[0];
    $rows = [];
    $first = floor(_scalar(np.random([1], 0.0, 1.0)) * (1.0 * $n - 1.0e-12));
    append($rows, $first);
    $C = np.take($A, np.astype(np.array([1.0 * $first], "f64"), "i64"), 0);
    $i = 1;
    while ($i < $k) {
        // D² to the NEAREST chosen centre, per point.
        $d2 = np.reshape(np.min(_dist2($A, $C), 1, false), [$n]);
        $pick = _weightedDraw($d2);
        append($rows, $pick);
        $C = np.take($A, np.astype(np.array($rows, "f64"), "i64"), 0);
        $i = $i + 1;
    }
    return $C;
}

// ── KMeans ──────────────────────────────────────────────────────────────────
def _kmeansDefaults() {
    return {
        "n_clusters": 8,
        "n_init":     10,
        "max_iter":   300,
        "tol":        0.0001
    };
}

class _KMeans extends Estimator {
    def init($params) {
        super();
        this.name_   = "KMeans";
        this.task_   = "clusterer";
        this.params_ = _kmeansDefaults();
        this.set_params($params);
        this.fitted_ = false;
        this._validateParams();
        this.centers_ = null;   // [k, f]
        this.labels_  = null;   // [n] i64
        this.inertia_ = 0.0;
        this.iters_   = 0;
        this.n_feat_  = 0;
    }

    // [K68]: set_params checks the value, not only the key.
    def _validateParams() {
        $p = this.params_;
        _whole($p.n_clusters, 1, "n_clusters", "KMeans");
        _whole($p.n_init, 1, "n_init", "KMeans");
        _whole($p.max_iter, 1, "max_iter", "KMeans");
        _num($p.tol, "tol", "KMeans");
        if ($p.tol < 0.0) {
            throw "KMeans: tol must be >= 0, got " + str($p.tol);
        }
        return null;
    }

    // One Lloyd run from one start. Returns [centers, labels, inertia, iters].
    def _lloyd($A, $k, $n) {
        $C = _kmeanspp($A, $k, "KMeans.fit");
        $lab = null;
        $it = 0;
        while ($it < this.params_.max_iter) {
            $D = _dist2($A, $C);
            $lab = np.reshape(np.astype(np.argmin($D, 1, false), "i64"), [$n]);
            // The update: one mean per cluster. A cluster's rows are selected
            // by a mask rather than by an index list, so the whole step is
            // vectorised per cluster instead of per point.
            $newRows = [];
            $empty = [];
            $j = 0;
            while ($j < $k) {
                $m = np.astype(np.equal($lab, 1.0 * $j, null), "f64");
                $cnt = _scalar(np.sum($m, null, false));
                if ($cnt > 0.0) {
                    $sum = np.sum(np.multiply($A, np.reshape($m, [$n, 1]), null),
                                  0, false);
                    append($newRows, np.to_list(np.divide($sum, $cnt, null)));
                } else {
                    // AN EMPTY CLUSTER IS RELOCATED, NOT DROPPED. Dropping it
                    // silently returns fewer clusters than were asked for --
                    // a wrong answer that looks right, and a later predict
                    // would index a centre that does not exist. Relocation to
                    // the point farthest from its own centre keeps k an
                    // invariant, and is what scikit-learn does.
                    append($empty, $j);
                    // Reshaped to [f] before to_list: np.take along axis 0
                    // returns [1, f], and a list of [1, f] lists mixed with a
                    // list of [f] lists is ragged -- nd refuses it, loudly,
                    // which is the right behaviour and was the first thing
                    // this hit.
                    append($newRows, np.to_list(np.reshape(np.take($C,
                        np.astype(np.array([1.0 * $j], "f64"), "i64"), 0),
                        [np.shape($A)[1]])));
                }
                $j = $j + 1;
            }
            $Cnew = np.array($newRows, "f64");
            if (len($empty) > 0) {
                // Farthest-first, recomputed once per relocation rather than
                // once per empty cluster, so k relocations cost one scan.
                $far = np.reshape(np.min(_dist2($A, $Cnew), 1, false), [$n]);
                $ord = np.argsort(np.multiply($far, 0.0 - 1.0, null), null);
                // Patched in the ROW LIST and the array rebuilt once.
                // nd_set writes a scalar at an index, not a row, so there is no
                // in-place row assignment to reach for -- and rebuilding costs
                // one allocation on a path that fires only when a cluster
                // actually empties.
                $e = 0;
                while ($e < len($empty)) {
                    $row = np.get(np.astype($ord, "i64"), [$e]);
                    $newRows[$empty[$e]] = np.to_list(np.reshape(np.take($A,
                        np.astype(np.array([1.0 * $row], "f64"), "i64"), 0),
                        [np.shape($A)[1]]));
                    $e = $e + 1;
                }
                $Cnew = np.array($newRows, "f64");
            }
            // Convergence on centre movement, which is what Lloyd's iteration
            // actually makes monotone -- label churn can be zero while centres
            // still move on a tie.
            $shift = _scalar(np.max(np.abs(np.subtract($Cnew, $C, null), null),
                                    null, false));
            $C = $Cnew;
            $it = $it + 1;
            if ($shift <= this.params_.tol) { break; }
        }
        $D = _dist2($A, $C);
        $lab = np.reshape(np.astype(np.argmin($D, 1, false), "i64"), [$n]);
        $inertia = _scalar(np.sum(np.min($D, 1, false), null, false));
        return [$C, $lab, $inertia, $it];
    }

    def fit($X, $y) {
        $who = "KMeans.fit";
        $A = base._asarray($X);
        if (len(np.shape($A)) != 2) {
            throw $who + ": X must be 2-d [n_samples, n_features], got shape "
                  + str(np.shape($A));
        }
        $n = np.shape($A)[0];
        $k = this.params_.n_clusters;
        if ($n < $k) {
            throw $who + ": " + str($n) + " rows cannot be split into "
                  + str($k) + " clusters";
        }
        this._validateParams();
        this.n_feat_ = np.shape($A)[1];

        // n_init restarts, keeping the lowest INERTIA -- the objective Lloyd's
        // algorithm descends. Keeping the best by anything else would be
        // choosing a run by a criterion it was not optimising.
        $best = null;
        $r = 0;
        while ($r < this.params_.n_init) {
            $run = this._lloyd($A, $k, $n);
            if ($best == null || $run[2] < $best[2]) { $best = $run; }
            $r = $r + 1;
        }
        this.centers_ = $best[0];
        this.labels_  = $best[1];
        this.inertia_ = $best[2];
        this.iters_   = $best[3];
        this.fitted_  = true;
        return this;
    }

    def predict($X) {
        $who = "KMeans.predict";
        this._check_fitted($who);
        $A = base._asarray($X);
        if (len(np.shape($A)) != 2 || np.shape($A)[1] != this.n_feat_) {
            throw $who + ": X must have " + str(this.n_feat_) + " features, got "
                  + "shape " + str(np.shape($A));
        }
        return np.reshape(np.astype(np.argmin(_dist2($A, this.centers_), 1, false),
                                    "i64"), [np.shape($A)[0]]);
    }

    // The distance to every centre, which is what a caller wants for a soft
    // assignment or an outlier score.
    def transform($X) {
        $who = "KMeans.transform";
        this._check_fitted($who);
        $A = base._asarray($X);
        if (len(np.shape($A)) != 2 || np.shape($A)[1] != this.n_feat_) {
            throw $who + ": X must have " + str(this.n_feat_) + " features, got "
                  + "shape " + str(np.shape($A));
        }
        return np.sqrt(_dist2($A, this.centers_), null);
    }

    def cluster_centers() { this._check_fitted("cluster_centers"); return this.centers_; }
    def labels()          { this._check_fitted("labels");          return this.labels_; }
    def inertia()         { this._check_fitted("inertia");         return this.inertia_; }
    def n_iter()          { this._check_fitted("n_iter");          return this.iters_; }

    // A clusterer has no y, so base.b's score() has nothing to dispatch on.
    // The inertia NEGATED is the convention scikit-learn uses, so that higher
    // is better and a grid search over n_clusters is not inverted.
    def score($X, $y) {
        $who = "KMeans.score";
        this._check_fitted($who);
        $A = base._asarray($X);
        return 0.0 - _scalar(np.sum(np.min(_dist2($A, this.centers_), 1, false),
                                    null, false));
    }

    def _paramsCopy() {
        $d = {};
        for $kk, $vv in this.params_.items() { $d[$kk] = $vv; }
        return $d;
    }
    def clone() { return KMeans(this._paramsCopy()); }
}
def KMeans($params) { return new _KMeans($params); }

def version() { return "bkit.cluster 1.0.0 (KMeans over one GEMM per step)"; }
