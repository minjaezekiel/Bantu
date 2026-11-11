// ════════════════════════════════════════════════════════════════════════════
//  linear.b — linear and logistic models.
//
//      include "bkit/base.b"   as base;
//      include "bkit/linear.b" as lin;
//
//      $m = lin.Ridge(1.0, null);
//      $m.fit($X, $y);
//      print($m.score($Xte, $yte));        // R2, from base.b
//
//  Design: docs/bkit-architecture.md §6. Decisions K31-K35.
//
//  THE ONE THING TO KNOW: NO MODEL HERE FORMS X'X (K31). The roadmap asked for
//  "normal equations, with solve for well-conditioned"; the second half is
//  wrong and the measurement is the argument. On a degree-9 polynomial design
//  over 60 points -- an ordinary fit -- the condition number of X is 3.6e6 and
//  of X'X is 1.3e13, because the Gram matrix SQUARES it. The coefficients came
//  out 2.4 MILLION times less accurate: 2.8e-3 of error against 1.2e-9. The
//  cost of avoiding that was 41 ms against 7 ms on a 5000x60 problem, once per
//  fit.
//
//  And it is not made conditional, because the caller cannot know which regime
//  they are in: deciding "is this well-conditioned" needs the condition
//  number, which costs an SVD -- more than the lstsq it was avoiding.
//
//  Ridge still has its closed form, but written as an AUGMENTED least-squares
//  problem (K32), which never forms the Gram matrix and excludes the intercept
//  from the penalty by construction rather than by remembering to.
// ════════════════════════════════════════════════════════════════════════════

include "./base.b" as base;

// ── shared input handling ───────────────────────────────────────────────────
def _X2d($X, $who) {
    if ($X == null) { throw $who + ": X is null"; }
    $A = $X;
    if (type($A) == "list") { $A = nd($A, "f64"); }
    if (type($A) != "ndarray") {
        throw $who + ": expected an array or a list of rows, got " + type($X);
    }
    if (nd_dtype($A) != "f64") { $A = nd_astype($A, "f64"); }
    if (nd_ndim($A) == 1) { return nd_reshape($A, [nd_shape($A)[0], 1]); }
    if (nd_ndim($A) != 2) {
        throw $who + ": expected a 1-D or 2-D X, got " + str(nd_ndim($A)) + " dimensions";
    }
    return $A;
}

def _yvec($y, $who) {
    if ($y == null) { throw $who + ": y is null -- these models are supervised"; }
    $A = $y;
    if (type($A) == "list") { $A = nd($A, "f64"); }
    if (type($A) == "number") { $A = nd([$A], "f64"); }
    if (type($A) != "ndarray") {
        throw $who + ": expected an array or a list for y, got " + type($y);
    }
    if (nd_dtype($A) != "f64") { $A = nd_astype($A, "f64"); }
    if (nd_ndim($A) == 2) {
        if (nd_shape($A)[1] != 1) {
            throw $who + ": y must be one column, got " + str(nd_shape($A)[1]);
        }
        return nd_ravel($A);
    }
    return $A;
}

def _check_fit($X, $y, $who) {
    $A = _X2d($X, $who);
    $v = _yvec($y, $who);
    if (nd_shape($A)[0] != nd_shape($v)[0]) {
        throw $who + ": " + str(nd_shape($A)[0]) + " rows of X for "
              + str(nd_shape($v)[0]) + " targets";
    }
    if (nd_shape($A)[0] == 0) { throw $who + ": X has no rows"; }
    if (nd_get(nd_any(nd_isnan($A, null), null, null), 0)) {
        throw $who + ": X contains NaN -- impute it first (preprocess.SimpleImputer)";
    }
    if (nd_get(nd_any(nd_isnan($v, null), null, null), 0)) {
        throw $who + ": y contains NaN";
    }
    return [$A, $v];
}

// A leading column of ones, so the intercept is just another coefficient --
// and one the penalty terms below exclude on purpose (§6.3).
def _with_ones($A) {
    return nd_concat([nd_ones([nd_shape($A)[0], 1], "f64"), $A], 1);
}

// A random permutation, for the mini-batch shuffle. One line rather than a
// dependency on selection.b, which would pull splitters and two searches into
// every import of a linear model. Same RNG either way ([C4]).
def _permutation($n) {
    if ($n <= 1) { return nd_astype(nd_arange(0, $n, null), "i64"); }
    return nd_argsort(nd_random_uniform([$n], 0.0, 1.0), null);
}

// ── the augmented ridge solve (K32) ─────────────────────────────────────────
// minimise ||Xb - y||^2 + lam * ||P b||^2, where P zeroes the intercept
// column, by stacking sqrt(lam) rows under X and zeros under y:
//
//     [    X     ] b  ~  [ y ]
//     [ sqrt(lam) P ]    [ 0 ]
//
// The Gram matrix is never formed, so this inherits lstsq's accuracy (K31);
// and the intercept is excluded because its row of P is all zeros, which is a
// property of the construction rather than a step somebody has to remember.
//
// `skip` is how many leading columns go unpenalised: 1 when there is an
// intercept column, 0 when there is not.
def _ridge_solve($Xd, $yv, $lam, $skip) {
    if ($lam == null || $lam <= 0.0) { return nd_lstsq($Xd, $yv); }
    $p = nd_shape($Xd)[1];
    $np = $p - $skip;
    if ($np <= 0) { return nd_lstsq($Xd, $yv); }
    // sqrt(lam) * [0 | I]: `np` rows, zero in the unpenalised columns
    $pen = nd_zeros([$np, $p], "f64");
    $i = 0;
    while ($i < $np) {
        nd_set($pen, [$i, $i + $skip], sqrt($lam));
        $i = $i + 1;
    }
    return nd_lstsq(nd_concat([$Xd, $pen], 0),
                    nd_concat([$yv, nd_zeros([$np], "f64")], 0));
}

// The same, with per-row weights -- the inner solve of IRLS (§6.5). Scaling
// both sides by sqrt(w) turns a weighted least-squares problem into an
// ordinary one, which is the whole reason IRLS is cheap here.
def _weighted_ridge_solve($Xd, $yv, $w, $lam, $skip) {
    $sw = nd_sqrt($w, null);
    $Xw = nd_multiply($Xd, nd_reshape($sw, [nd_shape($sw)[0], 1]), null);
    $yw = nd_multiply($yv, $sw, null);
    return _ridge_solve($Xw, $yw, $lam, $skip);
}

// ── a shared base for the regressors ────────────────────────────────────────
// coef_ (p,) and intercept_ (scalar) are stored separately, so `predict` is
// one matmul and an add, and a caller reading coefficients does not have to
// know that the fit used a column of ones.
class _LinearModel extends Estimator {
    def init() {
        super();
        this.task_   = "regressor";
        this.coef_   = null;
        this.intercept_ = 0.0;
        this.nfeat_  = 0;
        this.n_iter_ = 0;
        this.converged_ = true;
    }

    def _store($b, $withIntercept, $nfeat) {
        if ($withIntercept) {
            this.intercept_ = nd_get($b, 0);
            this.coef_ = nd_copy(nd_slice($b, [[1, nd_shape($b)[0], 1]]));
        } else {
            this.intercept_ = 0.0;
            this.coef_ = nd_copy($b);
        }
        this.nfeat_  = $nfeat;
        this.fitted_ = true;
        return this;
    }

    def _prep_predict($X, $who) {
        this._check_fitted($who);
        $A = _X2d($X, $who);
        if (nd_shape($A)[1] != this.nfeat_) {
            throw $who + ": fitted on " + str(this.nfeat_) + " feature(s) but given "
                  + str(nd_shape($A)[1]) + ". A column-count mismatch would be a "
                  + "shape error here, but the message is clearer this way.";
        }
        if (nd_get(nd_any(nd_isnan($A, null), null, null), 0)) {
            throw $who + ": X contains NaN";
        }
        return $A;
    }

    def predict($X) {
        $A = this._prep_predict($X, this.name_ + ".predict");
        return nd_add(nd_matmul($A, this.coef_), this.intercept_, null);
    }

    def coef()       { return this.coef_; }
    def intercept()  { return this.intercept_; }
    def n_features() { return this.nfeat_; }
    def n_iter()     { return this.n_iter_; }
    def converged()  { return this.converged_; }
}

// ── LinearRegression ────────────────────────────────────────────────────────
class _LinearRegression extends _LinearModel {
    def init($fit_intercept) {
        super();
        this.name_ = "LinearRegression";
        $fi = $fit_intercept;  if ($fi == null) { $fi = true; }
        this.params_ = {"fit_intercept": $fi};
    }
    def fit($X, $y) {
        $d = _check_fit($X, $y, "LinearRegression.fit");
        $A = $d[0];
        $v = $d[1];
        $fi = this.params_.fit_intercept;
        $Xd = $A;
        if ($fi) { $Xd = _with_ones($A); }
        // lstsq on the design matrix -- never the Gram matrix (K31)
        return this._store(nd_lstsq($Xd, $v), $fi, nd_shape($A)[1]);
    }
    def clone() { return LinearRegression(this.params_.fit_intercept); }
}
def LinearRegression($fit_intercept) { return new _LinearRegression($fit_intercept); }

// ── Ridge ───────────────────────────────────────────────────────────────────
class _Ridge extends _LinearModel {
    def init($alpha, $fit_intercept) {
        super();
        this.name_ = "Ridge";
        $a = $alpha;  if ($a == null) { $a = 1.0; }
        if ($a < 0.0) { throw "Ridge: alpha must be >= 0, got " + str($a); }
        $fi = $fit_intercept;  if ($fi == null) { $fi = true; }
        this.params_ = {"alpha": $a, "fit_intercept": $fi};
    }
    def fit($X, $y) {
        $d = _check_fit($X, $y, "Ridge.fit");
        $A = $d[0];
        $v = $d[1];
        $fi = this.params_.fit_intercept;
        $skip = 0;
        $Xd = $A;
        if ($fi) { $Xd = _with_ones($A); $skip = 1; }
        // Augmented least squares, not (X'X + aI)b = X'y (K32): the Gram
        // matrix is never formed and the intercept is unpenalised by
        // construction, because its row of the penalty block is all zeros.
        return this._store(_ridge_solve($Xd, $v, this.params_.alpha, $skip),
                           $fi, nd_shape($A)[1]);
    }
    def clone() { return Ridge(this.params_.alpha, this.params_.fit_intercept); }
}
def Ridge($alpha, $fit_intercept) { return new _Ridge($alpha, $fit_intercept); }

// ── ElasticNet and Lasso ────────────────────────────────────────────────────
// Cyclic coordinate descent with soft thresholding, on
//
//     (1/2n)||y - Xb||^2 + alpha*rho*||b||_1 + (alpha/2)*(1-rho)*||b||^2
//
// which is scikit-learn's objective (rho is l1_ratio; rho = 1 is Lasso). The
// L1 term is not differentiable at zero, so there is no closed form -- but
// each COORDINATE has one, which is what makes coordinate descent the natural
// method rather than a fallback.
//
// The intercept is handled by centring X and y, fitting without one, and
// recovering it as mean(y) - mean(X) . b. That is also how it escapes the
// penalty (§6.3): there is no intercept column to penalise.
//
// THE ITERATION CAP IS AUDIBLE (K33). A sweep that stops at max_iter without
// converging returns coefficients that are WRONG, not approximate, and every
// number computed from them afterwards is wrong with nothing to say so. So
// `converged_` and `n_iter_` are on the estimator and the cap prints to
// STDERR, where a pipeline's stdout cannot swallow it.
def _soft_threshold($z, $g) {
    if ($z > $g)  { return $z - $g; }
    if ($z < 0.0 - $g) { return $z + $g; }
    return 0.0;
}

class _ElasticNet extends _LinearModel {
    def init($alpha, $l1_ratio, $fit_intercept, $max_iter, $tol) {
        super();
        this.name_ = "ElasticNet";
        $a = $alpha;     if ($a == null) { $a = 1.0; }
        $r = $l1_ratio;  if ($r == null) { $r = 0.5; }
        $fi = $fit_intercept;  if ($fi == null) { $fi = true; }
        $mi = $max_iter; if ($mi == null) { $mi = 1000; }
        $t = $tol;       if ($t == null) { $t = 1.0e-7; }
        if ($a < 0.0) { throw "ElasticNet: alpha must be >= 0, got " + str($a); }
        if ($r < 0.0 || $r > 1.0) {
            throw "ElasticNet: l1_ratio must be in [0, 1], got " + str($r);
        }
        if ($mi < 1 || $mi != floor($mi)) {
            throw "ElasticNet: max_iter must be a whole number >= 1, got " + str($mi);
        }
        if ($t <= 0.0) { throw "ElasticNet: tol must be > 0, got " + str($t); }
        this.params_ = {"alpha": $a, "l1_ratio": $r, "fit_intercept": $fi,
                        "max_iter": $mi, "tol": $t};
    }

    def fit($X, $y) {
        $who = this.name_ + ".fit";
        $d = _check_fit($X, $y, $who);
        $A = $d[0];
        $v = $d[1];
        $n = nd_shape($A)[0];
        $p = nd_shape($A)[1];
        $al = this.params_.alpha;
        $rho = this._rho();
        // VALIDATED, not trusted. This caught a real bug the day it was
        // written: _Lasso replaces params_ with a dict that has no
        // `l1_ratio`, so `this.params_.l1_ratio` read null -- and in Bantu
        // `alpha * null` is 0, silently. Lasso therefore applied NO L1 term at
        // all and produced no sparsity, while reporting a clean convergence.
        // A missing key plus null-as-zero arithmetic is a plausible wrong
        // answer, which is the only kind worth building machinery against.
        if (type($rho) != "number") {
            throw $who + ": l1_ratio resolved to " + type($rho) + ", not a number. "
                  + "An estimator extending ElasticNet must keep `l1_ratio` in "
                  + "params_ or override _rho().";
        }
        if (type($al) != "number") {
            throw $who + ": alpha resolved to " + type($al) + ", not a number";
        }

        // centre, so there is no intercept column to penalise (§6.3)
        $xm = nd_zeros([1, $p], "f64");
        $ym = 0.0;
        $Ac = $A;
        $vc = $v;
        if (this.params_.fit_intercept) {
            $xm = nd_mean($A, 0, true);
            $ym = nd_get(nd_mean($v, null, null), 0);
            $Ac = nd_subtract($A, $xm, null);
            $vc = nd_subtract($v, $ym, null);
        }

        // column norms, once: the denominator of every coordinate update
        $colsq = nd_ravel(nd_sum(nd_multiply($Ac, $Ac, null), 0, false));
        $b = nd_zeros([$p], "f64");
        $resid = nd_copy($vc);                 // y - X b, maintained in place
        $l1 = $al * $rho;
        $l2 = $al * (1.0 - $rho);

        $it = 0;
        $done = false;
        while ($it < this.params_.max_iter && !$done) {
            $maxChange = 0.0;
            $j = 0;
            while ($j < $p) {
                $cj = nd_ravel(nd_slice($Ac, [null, [$j, $j + 1, 1]]));
                $bj = nd_get($b, $j);
                // the partial residual: add this coordinate's own contribution
                // back before measuring its correlation
                $rho_j = (nd_get(nd_sum(nd_multiply($cj, $resid, null), null, null), 0)
                          + $bj * nd_get($colsq, $j)) / (1.0 * $n);
                $den = nd_get($colsq, $j) / (1.0 * $n) + $l2;
                $newj = 0.0;
                if ($den > 0.0) { $newj = _soft_threshold($rho_j, $l1) / $den; }
                $delta = $newj - $bj;
                if ($delta != 0.0) {
                    nd_set($b, [$j], $newj);
                    // keep the residual current, so the next coordinate sees
                    // the truth: O(n) per update instead of a fresh matmul
                    $resid = nd_subtract($resid, nd_multiply($cj, $delta, null), null);
                    $ad = $delta;
                    if ($ad < 0.0) { $ad = 0.0 - $ad; }
                    if ($ad > $maxChange) { $maxChange = $ad; }
                }
                $j = $j + 1;
            }
            $it = $it + 1;
            if ($maxChange < this.params_.tol) { $done = true; }
        }

        this.n_iter_    = $it;
        this.converged_ = $done;
        if (!$done) {
            // STDERR, not stdout: a warning that a pipeline's output can
            // swallow is a warning nobody reads (K33).
            eprint("[bkit] " + this.name_ + ".fit did not converge in "
                   + str(this.params_.max_iter) + " iterations (tol "
                   + str(this.params_.tol) + "). The coefficients are NOT the "
                   + "solution. Raise max_iter, loosen tol, or scale X "
                   + "(StandardScaler) -- coordinate descent converges far "
                   + "faster on columns of comparable magnitude.");
        }

        this.coef_ = $b;
        this.intercept_ = 0.0;
        if (this.params_.fit_intercept) {
            this.intercept_ = $ym - nd_get(nd_matmul($xm, $b), 0);
        }
        this.nfeat_  = $p;
        this.fitted_ = true;
        return this;
    }

    // Overridable, so a subclass that does not expose l1_ratio as a tunable
    // parameter can still say what its value is.
    def _rho() { return this.params_.l1_ratio; }

    def clone() {
        return ElasticNet(this.params_.alpha, this.params_.l1_ratio,
                          this.params_.fit_intercept, this.params_.max_iter,
                          this.params_.tol);
    }
}
def ElasticNet($alpha, $l1_ratio, $fit_intercept, $max_iter, $tol) {
    return new _ElasticNet($alpha, $l1_ratio, $fit_intercept, $max_iter, $tol);
}

// Lasso is ElasticNet with l1_ratio = 1, and is a class of its own only so
// that `name()` and `clone()` report it honestly -- the protocol asserts a
// clone is the same TYPE as its source (K3).
class _Lasso extends _ElasticNet {
    def init($alpha, $fit_intercept, $max_iter, $tol) {
        super($alpha, 1.0, $fit_intercept, $max_iter, $tol);
        this.name_ = "Lasso";
        this.params_ = {"alpha": this.params_.alpha,
                        "fit_intercept": this.params_.fit_intercept,
                        "max_iter": this.params_.max_iter,
                        "tol": this.params_.tol};
    }
    // Lasso IS ElasticNet at l1_ratio = 1, and it deliberately does not expose
    // l1_ratio as a tunable parameter -- a grid searching over it would be
    // searching outside the model's definition. So fit() gets the value from
    // here rather than from params_, where it is absent.
    def _rho() { return 1.0; }
    def clone() {
        return Lasso(this.params_.alpha, this.params_.fit_intercept,
                     this.params_.max_iter, this.params_.tol);
    }
}
def Lasso($alpha, $fit_intercept, $max_iter, $tol) {
    return new _Lasso($alpha, $fit_intercept, $max_iter, $tol);
}

// ── LogisticRegression ──────────────────────────────────────────────────────
// Binary: IRLS (Newton on the log-likelihood), each step a weighted ridge
// solved by the augmented lstsq above. Five to ten iterations on ordinary
// data, quadratic convergence, no learning rate.
//
// Multiclass: ONE-VS-REST, k binary fits, prediction by the largest decision
// value (K34). Multinomial softmax needs a general optimiser because its
// Hessian is a k*p x k*p block matrix. What OvR costs is CALIBRATION, not
// accuracy: argmax of k independent fits agrees with argmax of a softmax fit
// on the great majority of points, but the k sigmoids do not sum to 1, so
// predict_proba normalises them and that normalisation is an approximation.
// scikit-learn's liblinear solver does exactly this.
//
// THE L2 PENALTY IS NOT OPTIONAL. On separable data the unpenalised likelihood
// is maximised by coefficients of infinite magnitude, so IRLS diverges with
// the weights p(1-p) going to zero. C bounds b, and p is clipped away from 0
// and 1 so the weights cannot underflow to exactly zero.
def _sigmoid($z) {
    return nd_reciprocal(nd_add(nd_exp(nd_negative($z, null), null), 1.0, null), null);
}

class _LogisticRegression extends Estimator {
    def init($C, $fit_intercept, $max_iter, $tol) {
        super();
        this.name_ = "LogisticRegression";
        this.task_ = "classifier";
        $c = $C;   if ($c == null) { $c = 1.0; }
        $fi = $fit_intercept;  if ($fi == null) { $fi = true; }
        $mi = $max_iter;  if ($mi == null) { $mi = 100; }
        $t = $tol;  if ($t == null) { $t = 1.0e-8; }
        if ($c <= 0.0) {
            throw "LogisticRegression: C must be > 0 (it is an INVERSE penalty "
                  + "strength, as in scikit-learn), got " + str($c);
        }
        if ($mi < 1 || $mi != floor($mi)) {
            throw "LogisticRegression: max_iter must be a whole number >= 1, got "
                  + str($mi);
        }
        this.params_ = {"C": $c, "fit_intercept": $fi, "max_iter": $mi, "tol": $t};
        this.classes_ = null;
        this.W_ = null;          // (k, p[+1]) one row per binary fit
        this.nfeat_ = 0;
        this.n_iter_ = 0;
        this.converged_ = true;
    }

    // One binary IRLS fit of `z` (0/1) against the design matrix.
    def _fit_binary($Xd, $z, $skip) {
        $p = nd_shape($Xd)[1];
        $b = nd_zeros([$p], "f64");
        $lam = 1.0 / this.params_.C;
        $it = 0;
        $done = false;
        $eps = 1.0e-10;
        while ($it < this.params_.max_iter && !$done) {
            $eta = nd_matmul($Xd, $b);
            $pr  = nd_clip(_sigmoid($eta), $eps, 1.0 - $eps, null);
            $w   = nd_multiply($pr, nd_subtract(1.0, $pr, null), null);
            // the IRLS working response
            $zz  = nd_add($eta, nd_divide(nd_subtract($z, $pr, null), $w, null), null);
            $bn  = _weighted_ridge_solve($Xd, $zz, $w, $lam, $skip);
            $ch  = nd_get(nd_max(nd_abs(nd_subtract($bn, $b, null), null), null, null), 0);
            $b   = $bn;
            $it  = $it + 1;
            if ($ch < this.params_.tol) { $done = true; }
        }
        return [$b, $it, $done];
    }

    def fit($X, $y) {
        $who = "LogisticRegression.fit";
        $d = _check_fit($X, $y, $who);
        $A = $d[0];
        $v = $d[1];
        $fi = this.params_.fit_intercept;
        $skip = 0;
        $Xd = $A;
        if ($fi) { $Xd = _with_ones($A); $skip = 1; }

        $classes = nd_unique($v);
        $k = nd_shape($classes)[0];
        if ($k < 2) {
            throw $who + ": y has only " + str($k) + " class, so there is nothing "
                  + "to separate";
        }
        $p = nd_shape($Xd)[1];
        $rows = $k;
        if ($k == 2) { $rows = 1; }      // one fit decides a binary problem
        $W = nd_zeros([$rows, $p], "f64");
        $totIt = 0;
        $allConv = true;
        $r = 0;
        while ($r < $rows) {
            // class r vs the rest; for k == 2, "class 1 vs class 0"
            $target = nd_get($classes, $r);
            if ($k == 2) { $target = nd_get($classes, 1); }
            $z = nd_astype(nd_equal($v, $target, null), "f64");
            $res = this._fit_binary($Xd, $z, $skip);
            $j = 0;
            while ($j < $p) { nd_set($W, [$r, $j], nd_get($res[0], $j)); $j = $j + 1; }
            if ($res[1] > $totIt) { $totIt = $res[1]; }
            if (!$res[2]) { $allConv = false; }
            $r = $r + 1;
        }
        this.classes_ = $classes;
        this.W_ = $W;
        this.nfeat_ = nd_shape($A)[1];
        this.n_iter_ = $totIt;
        this.converged_ = $allConv;
        if (!$allConv) {
            eprint("[bkit] LogisticRegression.fit did not converge in "
                   + str(this.params_.max_iter) + " iterations (tol "
                   + str(this.params_.tol) + "). Scale X (StandardScaler) or "
                   + "lower C -- IRLS struggles when the classes are nearly "
                   + "separable, because the unpenalised optimum is at infinity.");
        }
        this.fitted_ = true;
        return this;
    }

    def _design($X, $who) {
        this._check_fitted($who);
        $A = _X2d($X, $who);
        if (nd_shape($A)[1] != this.nfeat_) {
            throw $who + ": fitted on " + str(this.nfeat_) + " feature(s) but given "
                  + str(nd_shape($A)[1]);
        }
        if (nd_get(nd_any(nd_isnan($A, null), null, null), 0)) {
            throw $who + ": X contains NaN";
        }
        if (this.params_.fit_intercept) { return _with_ones($A); }
        return $A;
    }

    // (n, rows) raw decision values -- one column per binary fit.
    def decision_function($X) {
        $Xd = this._design($X, "LogisticRegression.decision_function");
        return nd_matmul($Xd, nd_T(this.W_));
    }

    def predict($X) {
        $s = this.decision_function($X);
        if (nd_shape(this.W_)[0] == 1) {
            // binary: positive decision means the SECOND class
            $hit = nd_astype(nd_greater(nd_ravel($s), 0.0, null), "i64");
            return nd_take(this.classes_, $hit, 0);
        }
        return nd_take(this.classes_, nd_astype(nd_ravel(nd_argmax($s, 1, true)), "i64"), 0);
    }

    // (n, k). For k > 2 these are NORMALISED one-vs-rest sigmoids, which is an
    // approximation and not a model (K34) -- documented rather than implied.
    def predict_proba($X) {
        $s = this.decision_function($X);
        $pr = _sigmoid($s);
        if (nd_shape(this.W_)[0] == 1) {
            $p1 = nd_ravel($pr);
            return nd_concat([nd_reshape(nd_subtract(1.0, $p1, null), [nd_shape($p1)[0], 1]),
                              nd_reshape($p1, [nd_shape($p1)[0], 1])], 1);
        }
        $rs = nd_sum($pr, 1, true);
        return nd_divide($pr, nd_where(nd_equal($rs, 0.0, null), 1.0, $rs), null);
    }

    def classes()    { return this.classes_; }
    def n_features() { return this.nfeat_; }
    def n_iter()     { return this.n_iter_; }
    def converged()  { return this.converged_; }
    def coef()       { return this.W_; }
    def clone() {
        return LogisticRegression(this.params_.C, this.params_.fit_intercept,
                                  this.params_.max_iter, this.params_.tol);
    }
}
def LogisticRegression($C, $fit_intercept, $max_iter, $tol) {
    return new _LogisticRegression($C, $fit_intercept, $max_iter, $tol);
}

// ── SGDRegressor and SGDClassifier ──────────────────────────────────────────
// MINI-BATCH gradient descent, default batch 256 (K35). True per-sample SGD is
// n interpreter iterations an epoch and the interpreter costs about a
// microsecond a step, so a 100,000-row epoch would be a tenth of a second of
// pure overhead before any arithmetic. A mini-batch step is one matmul.
//
// This is an honest difference from scikit-learn's SGDClassifier, which really
// does take one sample at a time: the optimisation PATH differs, so the
// coefficients differ, though both minimise the same objective. What the test
// asserts instead is the property that matters -- on a well-conditioned
// problem these must approach LinearRegression's closed-form answer. An
// optimiser is correct when it finds what the closed form already knows.
// THE STEP SIZE IS NOT COUPLED TO THE PENALTY (K36). The first version decayed
// as eta0 / (1 + eta0*alpha*t), which is broken at BOTH ends and was measured
// to be so:
//
//   * with a strong penalty the denominator grows fast and the step collapses.
//     At alpha = 2 on a 90/10 classification the intercept read -0.754 after
//     400 epochs, -0.938 after 4,000 and -1.046 after 20,000 -- still moving,
//     never reporting convergence, against IRLS's -2.069 for the same penalty.
//     The optimiser was not converging slowly; it was being throttled by the
//     one parameter that had nothing to do with step length.
//   * with alpha = 0 the schedule never decays at all, so there is no
//     annealing precisely where gradient noise most needs it.
//
// So `learning_rate` is an explicit choice, and the default depends only on t:
//
//   "invscaling"  eta0 / (1 + t)^power_t      (default, power_t = 0.25)
//   "constant"    eta0
//   "optimal"     1 / (alpha * (t + t0)),  t0 = 1/(alpha*eta0)
//
// "optimal" is the alpha-coupled schedule, kept because it is the right one
// for a strongly convex objective -- the strong convexity constant IS alpha --
// and it is scikit-learn's name for it. It is now something a caller asks for,
// with alpha > 0 required, rather than the only behaviour available.
class _SGDBase extends _LinearModel {
    def init($alpha, $eta0, $max_epochs, $batch_size, $fit_intercept, $tol,
             $learning_rate, $power_t) {
        super();
        $a = $alpha;  if ($a == null) { $a = 0.0001; }
        $e = $eta0;   if ($e == null) { $e = 0.01; }
        $m = $max_epochs;  if ($m == null) { $m = 200; }
        $bs = $batch_size; if ($bs == null) { $bs = 256; }
        $fi = $fit_intercept;  if ($fi == null) { $fi = true; }
        $t = $tol;    if ($t == null) { $t = 1.0e-6; }
        $lr = $learning_rate;  if ($lr == null) { $lr = "invscaling"; }
        $pt = $power_t;        if ($pt == null) { $pt = 0.25; }
        if ($a < 0.0) { throw "SGD: alpha must be >= 0, got " + str($a); }
        if ($e <= 0.0) { throw "SGD: eta0 must be > 0, got " + str($e); }
        if ($m < 1 || $m != floor($m)) {
            throw "SGD: max_epochs must be a whole number >= 1, got " + str($m);
        }
        if ($bs < 1 || $bs != floor($bs)) {
            throw "SGD: batch_size must be a whole number >= 1, got " + str($bs);
        }
        if ($lr != "invscaling" && $lr != "constant" && $lr != "optimal") {
            throw "SGD: learning_rate must be \"invscaling\", \"constant\" or "
                  + "\"optimal\", got \"" + str($lr) + "\"";
        }
        if ($lr == "optimal" && $a <= 0.0) {
            throw "SGD: learning_rate=\"optimal\" is 1/(alpha*(t+t0)) and needs "
                  + "alpha > 0 -- with alpha = 0 there is no strong convexity "
                  + "constant for it to use. Use \"invscaling\" instead.";
        }
        if ($pt < 0.0) { throw "SGD: power_t must be >= 0, got " + str($pt); }
        this.params_ = {"alpha": $a, "eta0": $e, "max_epochs": $m,
                        "batch_size": $bs, "fit_intercept": $fi, "tol": $t,
                        "learning_rate": $lr, "power_t": $pt};
    }

    // The step size at step t. A function of t and the declared schedule only
    // -- never of alpha, unless the caller asked for "optimal" (K36).
    def _eta($t) {
        $lr = this.params_.learning_rate;
        $e0 = this.params_.eta0;
        if ($lr == "constant") { return $e0; }
        if ($lr == "invscaling") {
            return $e0 / pow(1.0 + (1.0 * $t), this.params_.power_t);
        }
        // "optimal": t0 chosen so that eta(0) == eta0, which keeps the
        // parameter meaning the same thing across all three schedules
        $al = this.params_.alpha;
        return 1.0 / ($al * ((1.0 * $t) + 1.0 / ($al * $e0)));
    }

    // The gradient of the loss with respect to the linear output, per row.
    // Squared loss gives (pred - y); logistic loss gives (sigmoid(z) - y).
    // That the two differ ONLY here is why one optimiser serves both.
    def _grad_wrt_output($eta, $z) { throw this.name_ + ": _grad_wrt_output missing"; }

    // `center` CENTRES X and y and fits without an intercept column, exactly as
    // ElasticNet does (§6.3), recovering the intercept as mean(y) - mean(X).b.
    //
    // This is not a tidiness choice. Learning the intercept BY DESCENT means
    // it starts at 0 and has to travel the whole distance to mean(y) -- and
    // the step size decays as 1/(1 + eta0*alpha*t), so a strong penalty
    // throttles exactly the parameter that has furthest to go. Measured on a
    // target centred at 50 with alpha=5: the intercept reached 37.1 after 600
    // epochs and was still climbing. Centring removes the problem rather than
    // tuning around it, and as a bonus the penalty cannot touch the intercept
    // because there is no intercept column for it to act on.
    //
    // It applies to the REGRESSOR only. Centring a 0/1 label would change the
    // logistic loss into something else, and a classifier's intercept lives on
    // the logit scale where it is O(1) and learns from 0 without trouble.
    def _run($X, $y, $who, $center) {
        $d = _check_fit($X, $y, $who);
        $A = $d[0];
        $v = $d[1];
        $fi = this.params_.fit_intercept;
        $ctr = ($center && $fi);
        $xm = null;
        $ym = 0.0;
        if ($ctr) {
            $xm = nd_mean($A, 0, true);
            $ym = nd_get(nd_mean($v, null, null), 0);
            $A = nd_subtract($A, $xm, null);
            $v = nd_subtract($v, $ym, null);
        }
        $Xd = $A;
        if ($fi && !$ctr) { $Xd = _with_ones($A); }
        $n = nd_shape($Xd)[0];
        $p = nd_shape($Xd)[1];
        $b = nd_zeros([$p], "f64");
        $bs = this.params_.batch_size;
        if ($bs > $n) { $bs = $n; }
        $al = this.params_.alpha;
        $t = 0;
        $ep = 0;
        $done = false;
        while ($ep < this.params_.max_epochs && !$done) {
            $perm = _permutation($n);
            $prev = nd_copy($b);
            $start = 0;
            while ($start < $n) {
                $end = $start + $bs;
                if ($end > $n) { $end = $n; }
                $idx = nd_slice($perm, [[$start, $end, 1]]);
                $Xb = nd_take($Xd, $idx, 0);
                $yb = nd_take($v, $idx, 0);
                $g  = this._grad_wrt_output(nd_matmul($Xb, $b), $yb);
                $grad = nd_divide(nd_matmul(nd_T($Xb), $g), 1.0 * ($end - $start), null);
                if ($al > 0.0) {
                    // L2 on the coefficients, never on the intercept (§6.3).
                    // When centred there IS no intercept column, so nothing to
                    // exclude; when not, column 0 is it.
                    $pen = nd_multiply($b, $al, null);
                    if ($fi && !$ctr) { nd_set($pen, [0], 0.0); }
                    $grad = nd_add($grad, $pen, null);
                }
                $b = nd_subtract($b, nd_multiply($grad, this._eta($t), null), null);
                $t = $t + 1;
                $start = $end;
            }
            $ep = $ep + 1;
            $ch = nd_get(nd_max(nd_abs(nd_subtract($b, $prev, null), null), null, null), 0);
            if ($ch < this.params_.tol) { $done = true; }
        }
        this.n_iter_ = $ep;
        this.converged_ = $done;
        if (!$done) {
            eprint("[bkit] " + this.name_ + ".fit used all "
                   + str(this.params_.max_epochs) + " epochs without the "
                   + "coefficients settling below tol " + str(this.params_.tol)
                   + ". Scale X (StandardScaler), lower eta0, or raise "
                   + "max_epochs -- gradient descent on unscaled columns "
                   + "zig-zags instead of descending.");
        }
        if (!$ctr) { return this._store($b, $fi, nd_shape($A)[1]); }
        this.coef_      = $b;
        this.intercept_ = $ym - nd_get(nd_matmul($xm, $b), 0);
        this.nfeat_     = nd_shape($A)[1];
        this.fitted_    = true;
        return this;
    }
}

class _SGDRegressor extends _SGDBase {
    def init($alpha, $eta0, $max_epochs, $batch_size, $fit_intercept, $tol,
             $learning_rate, $power_t) {
        super($alpha, $eta0, $max_epochs, $batch_size, $fit_intercept, $tol,
              $learning_rate, $power_t);
        this.name_ = "SGDRegressor";
        this.task_ = "regressor";
    }
    // squared loss: d/dz (1/2)(z - y)^2 = z - y
    def _grad_wrt_output($eta, $z) { return nd_subtract($eta, $z, null); }
    def fit($X, $y) { return this._run($X, $y, "SGDRegressor.fit", true); }
    def clone() {
        return SGDRegressor(this.params_.alpha, this.params_.eta0,
                            this.params_.max_epochs, this.params_.batch_size,
                            this.params_.fit_intercept, this.params_.tol,
                            this.params_.learning_rate, this.params_.power_t);
    }
}
def SGDRegressor($alpha, $eta0, $max_epochs, $batch_size, $fit_intercept, $tol,
                 $learning_rate, $power_t) {
    return new _SGDRegressor($alpha, $eta0, $max_epochs, $batch_size, $fit_intercept,
                             $tol, $learning_rate, $power_t);
}

class _SGDClassifier extends _SGDBase {
    def init($alpha, $eta0, $max_epochs, $batch_size, $fit_intercept, $tol,
             $learning_rate, $power_t) {
        super($alpha, $eta0, $max_epochs, $batch_size, $fit_intercept, $tol,
              $learning_rate, $power_t);
        this.name_ = "SGDClassifier";
        this.task_ = "classifier";
        this.classes_ = null;
    }
    // logistic loss: d/dz [-y log s(z) - (1-y) log(1-s(z))] = s(z) - y
    def _grad_wrt_output($eta, $z) { return nd_subtract(_sigmoid($eta), $z, null); }

    def fit($X, $y) {
        $who = "SGDClassifier.fit";
        $v = _yvec($y, $who);
        $classes = nd_unique($v);
        if (nd_shape($classes)[0] != 2) {
            throw $who + ": this classifier is binary; y has "
                  + str(nd_shape($classes)[0]) + " classes. Use "
                  + "LogisticRegression, which does one-vs-rest.";
        }
        this.classes_ = $classes;
        // fit against 0/1, so the loss above is the logistic one
        $z = nd_astype(nd_equal($v, nd_get($classes, 1), null), "f64");
        return this._run($X, $z, $who, false);
    }

    def decision_function($X) {
        $A = this._prep_predict($X, "SGDClassifier.decision_function");
        return nd_add(nd_matmul($A, this.coef_), this.intercept_, null);
    }
    def predict($X) {
        $s = this.decision_function($X);
        return nd_take(this.classes_, nd_astype(nd_greater($s, 0.0, null), "i64"), 0);
    }
    def predict_proba($X) {
        $p1 = _sigmoid(this.decision_function($X));
        $n = nd_shape($p1)[0];
        return nd_concat([nd_reshape(nd_subtract(1.0, $p1, null), [$n, 1]),
                          nd_reshape($p1, [$n, 1])], 1);
    }
    def classes() { return this.classes_; }
    def clone() {
        return SGDClassifier(this.params_.alpha, this.params_.eta0,
                             this.params_.max_epochs, this.params_.batch_size,
                             this.params_.fit_intercept, this.params_.tol,
                             this.params_.learning_rate, this.params_.power_t);
    }
}
def SGDClassifier($alpha, $eta0, $max_epochs, $batch_size, $fit_intercept, $tol,
                  $learning_rate, $power_t) {
    return new _SGDClassifier($alpha, $eta0, $max_epochs, $batch_size, $fit_intercept,
                              $tol, $learning_rate, $power_t);
}
