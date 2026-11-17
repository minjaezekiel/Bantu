// ════════════════════════════════════════════════════════════════════════════
//  bkit/decompose.b — PCA and TruncatedSVD.
//
//      include "bkit/decompose.b" as dec;
//      $p = dec.PCA(2);          $p.fit($X, null);  $p.transform($X);
//      $p = dec.PCA(0.95);       // "enough components for 95% of the variance"
//      $t = dec.TruncatedSVD(2); // no centring -- see below
//
//  ── through the SVD of the centred matrix, never the covariance ───────────
//  The textbook route forms XᵀX/(n−1) and takes its eigendecomposition. This
//  file refuses that for the same reason bkit's linear models refuse it: THE
//  GRAM MATRIX SQUARES THE CONDITION NUMBER. A design matrix at 1e8 —
//  ordinary for unscaled real data — gives a covariance at 1e16, which is the
//  whole of double precision. The singular values of X are the square roots of
//  the eigenvalues of XᵀX, so the SVD reaches the same answer at half the
//  condition number.
//
//  The cost is real and is stated rather than hidden: SVD on [n, f] is more
//  expensive than eigh on [f, f] when n >> f. A wrong component at 1e-3 is
//  worse than a slow one.
//
//  ── singular vectors have arbitrary signs, and that is pinned ─────────────
//  U S Vᵀ is unchanged if a column of U and the matching row of Vᵀ are both
//  negated, so a bare SVD gives components whose SIGN IS AN ARTEFACT of the
//  LAPACK driver: two runs on the same data can disagree, and so can one run on
//  data differing in the last bit. Each component's sign is therefore fixed by
//  scikit-learn's rule -- make the largest-magnitude entry of each left
//  singular vector positive and flip the matching component with it -- with
//  ties broken by the lowest index so the rule is total ([K69]).
//
//  ── TruncatedSVD is a different estimator, not a faster PCA ───────────────
//  It factorises X itself rather than X − x̄, because centring a sparse matrix
//  destroys its sparsity. On already-centred data the two agree exactly; on
//  uncentred data TruncatedSVD's first component points at the mean. The test
//  asserts both, because only the second says the centring is really absent.
// ════════════════════════════════════════════════════════════════════════════

include "../numba/numba.b" as np;
include "./base.b" as base;

// A full reduction returns a 0-d ndarray HANDLE and every handle is truthy
// (numba N41), so it has to be read out rather than used as a number.
def _scalar($x) {
    return np.get(np.reshape($x, [1]), [0]);
}

// ── the sign convention ─────────────────────────────────────────────────────
// Make the largest-magnitude entry of each column of U positive, and apply the
// same flip to the matching row of Vt. The product U·S·Vt is unchanged; what
// changes is that the answer no longer depends on which way LAPACK happened to
// orient each vector.
//
// Ties are broken by the LOWEST index. Without that the rule is not total --
// a column whose two largest entries are exactly ±a would flip according to
// whichever the search found first, which is a flaky test rather than a wrong
// one, and flaky is harder to notice.
def _svdFlip($U, $Vt) {
    $k = np.shape($U)[1];
    $signs = [];
    $j = 0;
    while ($j < $k) {
        $col = np.reshape(np.take($U, np.astype(np.array([1.0 * $j], "f64"), "i64"), 1),
                          [np.shape($U)[0]]);
        $abs = np.abs($col, null);
        // argmax returns the FIRST maximal index, which is the tie-break.
        $at  = np.get(np.astype(np.argmax($abs, 0, false), "i64"), [0]);
        $v   = np.get($col, [$at]);
        $s   = 1.0;
        // A column of exact zeros cannot be oriented and is left alone; it
        // carries no information either way, and flipping it on the sign of
        // zero would make the answer depend on whether the zero is negative.
        if ($v < 0.0) { $s = 0.0 - 1.0; }
        append($signs, $s);
        $j = $j + 1;
    }
    $sv = np.array($signs, "f64");
    return [np.multiply($U, np.reshape($sv, [1, $k]), null),
            np.multiply($Vt, np.reshape($sv, [$k, 1]), null)];
}

// ── how many components ─────────────────────────────────────────────────────
// An integer is a count. A float in (0, 1) is a VARIANCE TARGET, and cannot be
// resolved before the data is seen -- which is why the requested value and the
// resolved count are separate fields and n_components_ needs a fit ([K6]'s
// split between a parameter and a learned quantity).
def _resolveK($spec, $ratios, $maxK, $who) {
    if ($spec == null) { return $maxK; }
    if (type($spec) != "number") {
        throw $who + ": n_components must be a whole count, a fraction in "
              + "(0, 1), or null for all of them, got " + type($spec);
    }
    if ($spec > 0.0 && $spec < 1.0) {
        $cum = np.cumsum($ratios, 0);
        $k = 1;
        while ($k < $maxK) {
            if (np.get($cum, [$k - 1]) >= $spec) { return $k; }
            $k = $k + 1;
        }
        return $maxK;
    }
    if ($spec != floor($spec) || $spec < 1.0 || $spec > 1.0 * $maxK) {
        throw $who + ": n_components " + str($spec) + " is not a whole count in "
              + "1.." + str($maxK) + " (that is min(n_samples, n_features)), nor "
              + "a fraction in (0, 1)";
    }
    return $spec;
}

class _DecompBase extends Estimator {
    def init($n_components, $name) {
        super();
        this.name_   = $name;
        this.task_   = "transformer";
        this.params_ = {"n_components": $n_components};
        this._validateParams();
        this.components_ = null;   // [k, f]
        this.sv_         = null;   // [k] singular values
        this.evar_       = null;   // [k] explained variance
        this.eratio_     = null;   // [k] explained variance ratio
        this.mean_       = null;   // [f] or null for TruncatedSVD
        this.n_feat_     = 0;
        this.k_          = 0;
        this.nsamp_      = 0;
    }

    // [K68]: set_params checks the value, not only the key.
    def _validateParams() {
        $v = this.params_.n_components;
        if ($v == null) { return null; }
        if (type($v) != "number") {
            throw this.name_ + ": n_components must be a number or null, got "
                  + type($v);
        }
        if ($v <= 0.0) {
            throw this.name_ + ": n_components must be positive, got " + str($v);
        }
        if ($v >= 1.0 && $v != floor($v)) {
            throw this.name_ + ": n_components " + str($v) + " is neither a whole "
                  + "count nor a fraction in (0, 1)";
        }
        return null;
    }

    def _centre($A) { return $A; }      // PCA overrides
    // Called at the top of every fit. PCA uses it to forget the mean it learned
    // last time; a plain `super.fit()` would be the obvious way to express
    // this, and Bantu has no such form -- only `super(args)` for the
    // constructor works (language trap 15). A hook is what is left, and it is
    // arguably clearer anyway: the base decides WHEN, the subclass decides
    // WHAT.
    def _beforeFit() { return null; }

    def fit($X, $y) {
        $who = this.name_ + ".fit";
        this._beforeFit();
        $A = base._asarray($X);
        if (len(np.shape($A)) != 2) {
            throw $who + ": X must be 2-d [n_samples, n_features], got shape "
                  + str(np.shape($A));
        }
        $n = np.shape($A)[0];
        $f = np.shape($A)[1];
        if ($n < 2) {
            throw $who + ": need at least 2 rows to decompose, got " + str($n);
        }
        this.n_feat_ = $f;
        this.nsamp_  = $n;
        $C = this._centre($A);

        // numba's svd returns [U, S, V], NOT [U, S, Vt]. NumPy returns Vh and
        // this file first assumed the same; numba's own linalg test states the
        // convention ("V is n-by-k", reconstructing with nd_T(V)), so the
        // transpose belongs here rather than a complaint belonging there.
        // Getting it wrong is not loud: U S V had the right SHAPE and produced
        // components that were merely wrong, which showed up only as a
        // transform whose columns were not orthogonal (language/surface trap
        // 36).
        $usv = np.svd($C);
        $parts = _svdFlip($usv[0], np.transpose($usv[2], null));
        $U  = $parts[0];
        $S  = $usv[1];
        $Vt = $parts[1];

        // Variance explained. The denominator is n-1 (the unbiased estimator),
        // which is what makes explained_variance_ comparable with a sample
        // variance computed any other way.
        $var = np.divide(np.multiply($S, $S, null), 1.0 * ($n - 1), null);
        $tot = _scalar(np.sum($var, null, false));
        $ratio = $var;
        if ($tot > 0.0) { $ratio = np.divide($var, $tot, null); }
        else {
            // Every singular value zero: a constant matrix. The ratios are
            // undefined rather than zero, but reporting NaN would propagate
            // into any downstream sum; zeros say "no variance anywhere", which
            // is the truth about this matrix.
            $ratio = np.zeros([np.size($var)], "f64");
        }

        $maxK = np.size($S);
        $k = _resolveK(this.params_.n_components, $ratio, $maxK, $who);
        this.k_ = $k;
        $sel = np.astype(np.arange(0.0, 1.0 * $k, 1.0), "i64");
        this.components_ = np.take($Vt, $sel, 0);
        this.sv_         = np.take($S, $sel, 0);
        this.evar_       = np.take($var, $sel, 0);
        this.eratio_     = np.take($ratio, $sel, 0);
        this.fitted_     = true;
        return this;
    }

    def transform($X) {
        $who = this.name_ + ".transform";
        this._check_fitted($who);
        $A = base._asarray($X);
        if (len(np.shape($A)) != 2 || np.shape($A)[1] != this.n_feat_) {
            throw $who + ": X must have " + str(this.n_feat_) + " features, got "
                  + "shape " + str(np.shape($A));
        }
        return np.matmul(this._centre($A), np.transpose(this.components_, null));
    }

    // Back to the original space. Exact when every component is kept, and the
    // best rank-k approximation otherwise -- which is the Eckart-Young theorem
    // and is asserted rather than assumed.
    def inverse_transform($Z) {
        $who = this.name_ + ".inverse_transform";
        this._check_fitted($who);
        $Za = base._asarray($Z);
        if (len(np.shape($Za)) != 2 || np.shape($Za)[1] != this.k_) {
            throw $who + ": Z must have " + str(this.k_) + " columns, got shape "
                  + str(np.shape($Za));
        }
        $R = np.matmul($Za, this.components_);
        if (this.mean_ != null) { $R = np.add($R, this.mean_, null); }
        return $R;
    }

    def components()                { this._check_fitted("components");   return this.components_; }
    def singular_values()           { this._check_fitted("singular_values"); return this.sv_; }
    def explained_variance()        { this._check_fitted("explained_variance"); return this.evar_; }
    def explained_variance_ratio()  { this._check_fitted("explained_variance_ratio"); return this.eratio_; }
    def n_components()              { this._check_fitted("n_components");  return this.k_; }
    def mean()                      { this._check_fitted("mean");          return this.mean_; }
}

// ── PCA ─────────────────────────────────────────────────────────────────────
class _PCA extends _DecompBase {
    def init($n_components) { super($n_components, "PCA"); }
    // The mean is learned at fit and reused at transform. Recomputing it per
    // call would centre test data on its OWN mean, which is a different and
    // wrong transformation -- the same leak shape as fitting a scaler on the
    // test set.
    def _centre($A) {
        if (this.mean_ == null) {
            this.mean_ = np.reshape(np.mean($A, 0, false), [np.shape($A)[1]]);
        }
        return np.subtract($A, this.mean_, null);
    }
    def _beforeFit() { this.mean_ = null; return null; }   // relearned per fit
    def clone() { return PCA(this.params_.n_components); }
}
def PCA($n_components) { return new _PCA($n_components); }

// ── TruncatedSVD ────────────────────────────────────────────────────────────
class _TruncatedSVD extends _DecompBase {
    def init($n_components) { super($n_components, "TruncatedSVD"); }
    def clone() { return TruncatedSVD(this.params_.n_components); }
}
def TruncatedSVD($n_components) { return new _TruncatedSVD($n_components); }

def version() { return "bkit.decompose 1.0.0 (PCA and TruncatedSVD over numba's SVD)"; }
