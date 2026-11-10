// ════════════════════════════════════════════════════════════════════════════
//  preprocess.b — scalers, encoders, imputers, polynomial features.
//
//      include "bkit/base.b"       as base;
//      include "bkit/preprocess.b" as pre;
//
//      $sc = pre.StandardScaler(null, null);
//      $Z  = $sc.fit_transform($X, null);
//      $back = $sc.inverse_transform($Z);          // == $X, to the last bit
//
//  THE ELEVEN: StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
//  OrdinalEncoder, OneHotEncoder, LabelEncoder, SimpleImputer,
//  PolynomialFeatures, Normalizer, Binarizer -- eleven estimators, plus the
//  two base classes they share.
//
//  Design: docs/bkit-architecture.md §3. Four things decided there are load
//  bearing and are repeated at the code that implements them:
//
//    * STATISTICS ARE (1, d) ARRAYS and `transform` is two kernel calls over
//      the whole matrix (§3.1). The interpreter is a tree-walker, so a Bantu
//      loop over rows is three orders of magnitude off a kernel, and a scaler
//      sits on the hot path of every fit in the package. The only Bantu loops
//      in this file run over COLUMNS or over output features -- never rows.
//
//    * FOUR SCALERS SHARE ONE `transform` (§3.2, K11). They differ only in the
//      two numbers they compute, so `_ScalerBase` holds transform, the inverse,
//      the feature-count guard and the NaN refusal, and each subclass overrides
//      `fit` alone. The round-trip gate then tests one implementation four
//      times rather than four implementations once.
//
//    * THE FEATURE-COUNT GUARD IS LOAD BEARING (§3.4, K12). An (n, 1) matrix
//      against a stored (1, d) statistic BROADCASTS and returns a perfectly
//      well-formed (n, d) matrix of nonsense. Nothing in numba catches that,
//      so every transformer records `nfeat_` at fit and checks it.
//
//    * ONE RULE ABOUT NaN, with one exception (§3.7, K14): every estimator here
//      REFUSES NaN, naming the column; `SimpleImputer` accepts it, because that
//      is its entire job. The refusal is in fit, transform AND
//      inverse_transform so there is no third case to remember.
//
//      It is a refusal rather than scikit-learn's NaN-aware statistics because
//      numba has nansum/nanmean/nanmin/nanmax and no nanvar, nanstd,
//      nanmedian or nanquantile -- three of the four scalers need exactly
//      those. Supporting NaN in two scalers while a third quietly produced NaN
//      would be worse than refusing in all four: a surface that looks uniform
//      and is not. When those kernels land this becomes a one-line change.
//
//  Every estimator is `_Name` plus a same-named factory `def Name(...)`,
//  because a `def Name()` shadows a `class Name` and `new Name(...)` inside
//  the factory re-enters the factory until the call-depth guard fires. Fields
//  carry a trailing underscore because a field and a method cannot share a
//  name -- Bantu has one namespace per object.
// ════════════════════════════════════════════════════════════════════════════

include "./base.b" as base;

// ── shared input handling ───────────────────────────────────────────────────

// Every estimator takes its X through here: a list of rows, a 1-D array or a
// 2-D array, and always hands back 2-D f64. A 1-D input becomes (n, 1), which
// is what makes `Binarizer().fit_transform($vector, null)` work without a
// special case anywhere downstream.
def _as2d($X, $who) {
    if ($X == null) { throw $who + ": X is null"; }
    $A = $X;
    if (type($A) == "list")   { $A = nd($A, "f64"); }
    if (type($A) == "number") { $A = nd([[$A]], "f64"); }
    if (type($A) != "ndarray") {
        throw $who + ": expected an array or a list of rows, got " + type($X);
    }
    if (nd_dtype($A) != "f64") { $A = nd_astype($A, "f64"); }
    $k = nd_ndim($A);
    if ($k == 1) { return nd_reshape($A, [nd_shape($A)[0], 1]); }
    if ($k == 2) { return $A; }
    throw $who + ": expected a 1-D or 2-D array, got " + str($k) + " dimensions";
}

// A 1-D column of f64, for the estimators that work on y rather than X.
def _as1d($y, $who) {
    if ($y == null) { throw $who + ": y is null"; }
    $A = $y;
    if (type($A) == "list")   { $A = nd($A, "f64"); }
    if (type($A) == "number") { $A = nd([$A], "f64"); }
    if (type($A) != "ndarray") {
        throw $who + ": expected an array or a list, got " + type($y);
    }
    if (nd_dtype($A) != "f64") { $A = nd_astype($A, "f64"); }
    $k = nd_ndim($A);
    if ($k == 1) { return $A; }
    if ($k == 2) {
        if (nd_shape($A)[1] == 1) { return nd_ravel($A); }
        throw $who + ": expected one column, got " + str(nd_shape($A)[1])
              + " -- this estimator works on a single label column";
    }
    throw $who + ": expected a 1-D array, got " + str($k) + " dimensions";
}

// NaN refusal (K14). It names the FIRST offending column rather than saying
// "X contains NaN", because the caller's next question is always which one.
def _no_nan($A, $who) {
    $m = nd_isnan($A, null);
    if (!nd_get(nd_any($m, null, null), 0)) { return null; }
    $per = nd_astype(nd_any($m, 0, false), "f64");
    $j   = nd_get(nd_argmax($per, null, null), 0);
    throw $who + ": column " + str($j) + " contains NaN. Everything in "
          + "preprocess.b except SimpleImputer refuses NaN rather than "
          + "computing a statistic from a smaller sample (DECISIONS K14) -- "
          + "impute first.";
}

// A fit on zero rows, refused in ONE place so every estimator agrees.
// Without it the scalers raised here and the encoders did not: nd_unique of an
// empty column gives an empty category list, the encoder fits happily, and the
// failure arrives on the first real transform as
// "nd_take: index -1 is out of range for axis 0 with 0 elements" -- a kernel
// message about an index, for a mistake made in a different call. An empty
// TRANSFORM is fine and stays allowed: zero rows in, zero rows out.
def _no_empty($A, $who) {
    if (nd_shape($A)[0] == 0) {
        throw $who + ": X has no rows, so there is nothing to learn from. "
              + "(Transforming zero rows is fine; fitting on them is not.)";
    }
    return null;
}

// The guard that broadcasting makes necessary (K12).
def _check_nfeat($A, $want, $who) {
    $got = nd_shape($A)[1];
    if ($got == $want) { return null; }
    throw $who + ": fitted on " + str($want) + " feature(s) but given "
          + str($got) + ". A column-count mismatch BROADCASTS here instead of "
          + "failing -- (n,1) against a stored (1,d) statistic yields a "
          + "well-formed (n,d) matrix of nonsense -- so it is checked "
          + "explicitly (DECISIONS K12).";
}

// A scale of zero means a constant column, and dividing by it gives NaN for
// every row -- silently, because NaN is a perfectly good f64. Substituting 1
// is scikit-learn's handling and it has a property worth stating: the column
// transforms to exactly `x - center` = 0 and inverse_transform adds `center`
// back, so the round-trip stays EXACT on a constant column (K15).
def _safe_scale($s) {
    return nd_where(nd_equal($s, 0.0, null), 1.0, $s);
}

// `append` mutates and returns the new LENGTH, so a list cannot be copied by
// appending to itself in an expression. This exists because the combination
// enumerator below reuses one index vector and must snapshot it.
def _copy_list($l) {
    $out = [];
    $i = 0;
    while ($i < len($l)) { append($out, $l[$i]); $i = $i + 1; }
    return $out;
}

// ── _ScalerBase ─────────────────────────────────────────────────────────────
// transform, inverse_transform, the guard and the NaN rule, written ONCE for
// four scalers (K11). Each subclass computes `center_` and `scale_` in `fit`
// and calls `_finish`; nothing else.
//
// This shape works only because of the constraint in base.b: `super.method()`
// does not exist in Bantu, but a child never needs it here -- the shared
// methods read `this.center_` and `this.scale_`, which the child has set.
class _ScalerBase extends Estimator {
    def init() {
        super();
        this.task_   = "transformer";
        this.center_ = null;
        this.scale_  = null;
        this.nfeat_  = 0;
    }

    def _finish($center, $scale, $nfeat) {
        this.center_ = $center;
        this.scale_  = _safe_scale($scale);
        this.nfeat_  = $nfeat;
        this.fitted_ = true;
        return this;
    }

    // Shared by every subclass's fit: validate, and report (A, n, d).
    def _prep_fit($X) {
        $who = this.name_ + ".fit";
        $A = _as2d($X, $who);
        _no_nan($A, $who);
        _no_empty($A, $who);
        return $A;
    }

    def transform($X) {
        $who = this.name_ + ".transform";
        this._check_fitted($who);
        $A = _as2d($X, $who);
        _no_nan($A, $who);
        _check_nfeat($A, this.nfeat_, $who);
        return nd_divide(nd_subtract($A, this.center_, null), this.scale_, null);
    }

    def inverse_transform($Z) {
        $who = this.name_ + ".inverse_transform";
        this._check_fitted($who);
        $A = _as2d($Z, $who);
        _no_nan($A, $who);
        _check_nfeat($A, this.nfeat_, $who);
        return nd_add(nd_multiply($A, this.scale_, null), this.center_, null);
    }

    def n_features() { return this.nfeat_; }
    def center()     { return this.center_; }
    def scale()      { return this.scale_; }
}

// ── StandardScaler ──────────────────────────────────────────────────────────
// (x - mean) / std, with the POPULATION standard deviation (ddof = 0), which
// is scikit-learn's choice. ddof = 1 would make the transform of a
// two-row matrix differ from NumPy's by a factor of sqrt(2) and the difference
// would never show up as an error, only as a slightly wrong model.
class _StandardScaler extends _ScalerBase {
    def init($with_mean, $with_std) {
        super();
        this.name_ = "StandardScaler";
        $wm = $with_mean;  if ($wm == null) { $wm = true; }
        $ws = $with_std;   if ($ws == null) { $ws = true; }
        this.params_ = {"with_mean": $wm, "with_std": $ws};
    }
    def fit($X, $y) {
        $A = this._prep_fit($X);
        $d = nd_shape($A)[1];
        $center = nd_zeros([1, $d], "f64");
        $scale  = nd_ones([1, $d], "f64");
        if (this.params_.with_mean) { $center = nd_mean($A, 0, true); }
        if (this.params_.with_std)  { $scale  = nd_std($A, 0, true, 0); }
        return this._finish($center, $scale, $d);
    }
    def clone() { return StandardScaler(this.params_.with_mean, this.params_.with_std); }
    def mean_of() { return this.center_; }
}
def StandardScaler($with_mean, $with_std) {
    return new _StandardScaler($with_mean, $with_std);
}

// ── MinMaxScaler ────────────────────────────────────────────────────────────
// Expressed in the SAME centre/scale form as the others rather than as its own
// two-step formula, which is what lets it share `transform`:
//
//     want:  (x - min) * (hi - lo) / (max - min) + lo
//     so:    scale  = (max - min) / (hi - lo)
//            center = min - lo * scale
//            (x - center) / scale  ==  the line above, identically.
class _MinMaxScaler extends _ScalerBase {
    def init($lo, $hi) {
        super();
        this.name_ = "MinMaxScaler";
        $l = $lo;  if ($l == null) { $l = 0.0; }
        $h = $hi;  if ($h == null) { $h = 1.0; }
        this.params_ = {"range_low": $l, "range_high": $h};
    }
    def fit($X, $y) {
        $A = this._prep_fit($X);
        $d  = nd_shape($A)[1];
        $lo = this.params_.range_low;
        $hi = this.params_.range_high;
        if ($hi <= $lo) {
            throw "MinMaxScaler.fit: the feature range must be increasing, got ["
                  + str($lo) + ", " + str($hi) + "]";
        }
        $mn  = nd_min($A, 0, true);
        $rng = _safe_scale(nd_subtract(nd_max($A, 0, true), $mn, null));
        $scale  = nd_divide($rng, $hi - $lo, null);
        $center = nd_subtract($mn, nd_multiply($scale, $lo, null), null);
        return this._finish($center, $scale, $d);
    }
    def clone() { return MinMaxScaler(this.params_.range_low, this.params_.range_high); }
}
def MinMaxScaler($lo, $hi) { return new _MinMaxScaler($lo, $hi); }

// ── RobustScaler ────────────────────────────────────────────────────────────
// Median and inter-quartile range, so a handful of extreme values cannot move
// the statistics. That is the whole reason it exists: one outlier of 1e6 in a
// column of unit-scale values crushes everything else to near zero under
// StandardScaler, and leaves the median and the IQR untouched here.
class _RobustScaler extends _ScalerBase {
    def init($q_low, $q_high) {
        super();
        this.name_ = "RobustScaler";
        $l = $q_low;   if ($l == null) { $l = 25.0; }
        $h = $q_high;  if ($h == null) { $h = 75.0; }
        this.params_ = {"q_low": $l, "q_high": $h};
    }
    def fit($X, $y) {
        $A = this._prep_fit($X);
        $d = nd_shape($A)[1];
        $l = this.params_.q_low;
        $h = this.params_.q_high;
        if ($l < 0.0 || $h > 100.0 || $h <= $l) {
            throw "RobustScaler.fit: the quantile range must satisfy "
                  + "0 <= q_low < q_high <= 100, got [" + str($l) + ", " + str($h) + "]";
        }
        $center = nd_median($A, 0, true);
        $scale  = nd_subtract(nd_quantile($A, $h / 100.0, 0, true),
                              nd_quantile($A, $l / 100.0, 0, true), null);
        return this._finish($center, $scale, $d);
    }
    def clone() { return RobustScaler(this.params_.q_low, this.params_.q_high); }
}
def RobustScaler($q_low, $q_high) { return new _RobustScaler($q_low, $q_high); }

// ── MaxAbsScaler ────────────────────────────────────────────────────────────
// Divide by max(|x|) and do NOT centre, which is the point: a sparse column
// stays sparse, because 0 maps to 0. Subtracting a mean first would turn every
// structural zero into a nonzero and is what makes StandardScaler unusable on
// sparse data.
class _MaxAbsScaler extends _ScalerBase {
    def init() {
        super();
        this.name_   = "MaxAbsScaler";
        this.params_ = {};
    }
    def fit($X, $y) {
        $A = this._prep_fit($X);
        $d = nd_shape($A)[1];
        return this._finish(nd_zeros([1, $d], "f64"),
                            nd_max(nd_abs($A, null), 0, true), $d);
    }
    def clone() { return MaxAbsScaler(); }
}
def MaxAbsScaler() { return new _MaxAbsScaler(); }

// ── Normalizer ──────────────────────────────────────────────────────────────
// Row-wise, not column-wise, and therefore NOT a _ScalerBase: it has no fitted
// statistics and no inverse, because the norm it divided by is not kept.
// `fit` records the feature count and nothing else.
//
// A zero row has norm zero. It is left as zeros rather than raising: a
// zero-vector row is a legitimate observation (an empty bag of words, say) and
// its only sensible normalisation is itself.
class _Normalizer extends Estimator {
    def init($norm) {
        super();
        this.name_ = "Normalizer";
        this.task_ = "transformer";
        $n = $norm;  if ($n == null) { $n = "l2"; }
        if ($n != "l1" && $n != "l2" && $n != "max") {
            throw "Normalizer: norm must be \"l1\", \"l2\" or \"max\", got \"" + str($n) + "\"";
        }
        this.params_ = {"norm": $n};
        this.nfeat_  = 0;
    }
    def fit($X, $y) {
        $A = _as2d($X, "Normalizer.fit");
        _no_nan($A, "Normalizer.fit");
        _no_empty($A, "Normalizer.fit");
        this.nfeat_  = nd_shape($A)[1];
        this.fitted_ = true;
        return this;
    }
    def transform($X) {
        $who = "Normalizer.transform";
        this._check_fitted($who);
        $A = _as2d($X, $who);
        _no_nan($A, $who);
        _check_nfeat($A, this.nfeat_, $who);
        $n = this.params_.norm;
        $nrm = null;
        if ($n == "l2")  { $nrm = nd_sqrt(nd_sum(nd_square($A, null), 1, true), null); }
        if ($n == "l1")  { $nrm = nd_sum(nd_abs($A, null), 1, true); }
        if ($n == "max") { $nrm = nd_max(nd_abs($A, null), 1, true); }
        return nd_divide($A, _safe_scale($nrm), null);
    }
    def clone() { return Normalizer(this.params_.norm); }
    def n_features() { return this.nfeat_; }
}
def Normalizer($norm) { return new _Normalizer($norm); }

// ── Binarizer ───────────────────────────────────────────────────────────────
// x > threshold, STRICTLY, matching scikit-learn. The default threshold of 0
// therefore sends 0 to 0, not to 1.
class _Binarizer extends Estimator {
    def init($threshold) {
        super();
        this.name_ = "Binarizer";
        this.task_ = "transformer";
        $t = $threshold;  if ($t == null) { $t = 0.0; }
        this.params_ = {"threshold": $t};
        this.nfeat_  = 0;
    }
    def fit($X, $y) {
        $A = _as2d($X, "Binarizer.fit");
        _no_nan($A, "Binarizer.fit");
        _no_empty($A, "Binarizer.fit");
        this.nfeat_  = nd_shape($A)[1];
        this.fitted_ = true;
        return this;
    }
    def transform($X) {
        $who = "Binarizer.transform";
        this._check_fitted($who);
        $A = _as2d($X, $who);
        // NaN is refused rather than compared: `NaN > t` is false, which would
        // silently record every missing value as "below the threshold".
        _no_nan($A, $who);
        _check_nfeat($A, this.nfeat_, $who);
        return nd_astype(nd_greater($A, this.params_.threshold, null), "f64");
    }
    def clone() { return Binarizer(this.params_.threshold); }
    def n_features() { return this.nfeat_; }
}
def Binarizer($threshold) { return new _Binarizer($threshold); }

// ── the encoders' shared machinery ──────────────────────────────────────────
// Categories come from nd_unique, which returns them SORTED, and a value is
// mapped to its code by nd_searchsorted -- one native O(n log k) call for a
// whole column instead of a Bantu dict lookup per row (K13).
//
// The unseen-category test falls out of the same call rather than needing a
// second pass: clamp the returned position into range, take the category it
// landed on, and compare it to the input. A mismatch IS an unseen value.
//
// Returns [codes, hit]: codes is (n,) f64 of clamped positions, hit is (n,)
// bool, false exactly where the value is not a known category.
def _encode_col($cats, $col) {
    $k    = nd_shape($cats)[0];
    $pos  = nd_astype(nd_searchsorted($cats, $col, null), "f64");
    $safe = nd_clip($pos, 0.0, 1.0 * ($k - 1), null);
    $got  = nd_take($cats, nd_astype($safe, "i64"), 0);
    return [$safe, nd_equal($got, $col, null)];
}

// The first value in `col` whose `hit` is false, for the error message.
def _first_unseen($col, $hit) {
    $miss = nd_astype(nd_logical_not($hit, null), "f64");
    $i    = nd_get(nd_argmax($miss, null, null), 0);
    return nd_get($col, $i);
}

// ── OrdinalEncoder ──────────────────────────────────────────────────────────
// Each column's categories are numbered 0..k-1 in ascending value order.
//
// handle_unknown:
//   "error"              an unseen value raises, naming the column and the
//                        value -- the default, because a silently-coded
//                        unknown is a wrong feature, not a missing one
//   "use_encoded_value"  it maps to `unknown_value`, which must be given
class _OrdinalEncoder extends Estimator {
    def init($handle_unknown, $unknown_value) {
        super();
        this.name_ = "OrdinalEncoder";
        this.task_ = "transformer";
        $h = $handle_unknown;  if ($h == null) { $h = "error"; }
        if ($h != "error" && $h != "use_encoded_value") {
            throw "OrdinalEncoder: handle_unknown must be \"error\" or "
                  + "\"use_encoded_value\", got \"" + str($h) + "\"";
        }
        if ($h == "use_encoded_value" && $unknown_value == null) {
            throw "OrdinalEncoder: handle_unknown=\"use_encoded_value\" needs an "
                  + "unknown_value -- there is no safe default, since any code "
                  + "you pick collides with a real category unless you choose it";
        }
        this.params_ = {"handle_unknown": $h, "unknown_value": $unknown_value};
        this.cats_   = [];
        this.nfeat_  = 0;
    }
    def fit($X, $y) {
        $who = "OrdinalEncoder.fit";
        $A = _as2d($X, $who);
        _no_nan($A, $who);
        _no_empty($A, $who);
        $d = nd_shape($A)[1];
        $c = [];
        $j = 0;
        while ($j < $d) {
            append($c, nd_unique(nd_ravel(nd_slice($A, [null, [$j, $j + 1, 1]]))));
            $j = $j + 1;
        }
        this.cats_   = $c;
        this.nfeat_  = $d;
        this.fitted_ = true;
        return this;
    }
    def transform($X) {
        $who = "OrdinalEncoder.transform";
        this._check_fitted($who);
        $A = _as2d($X, $who);
        _no_nan($A, $who);
        _check_nfeat($A, this.nfeat_, $who);
        $n = nd_shape($A)[0];
        $cols = [];
        $j = 0;
        while ($j < this.nfeat_) {
            $col = nd_ravel(nd_slice($A, [null, [$j, $j + 1, 1]]));
            $r   = _encode_col(this.cats_[$j], $col);
            $code = $r[0];
            $hit  = $r[1];
            if (!nd_get(nd_all($hit, null, null), 0)) {
                if (this.params_.handle_unknown == "error") {
                    throw $who + ": column " + str($j) + " contains the unseen "
                          + "value " + str(_first_unseen($col, $hit)) + ". Pass "
                          + "handle_unknown=\"use_encoded_value\" to map unknowns "
                          + "to a code instead of raising.";
                }
                $code = nd_where($hit, $code, 1.0 * this.params_.unknown_value);
            }
            append($cols, nd_reshape($code, [$n, 1]));
            $j = $j + 1;
        }
        return nd_concat($cols, 1);
    }
    // Codes back to values. An out-of-range code raises rather than clamping:
    // clamping would hand back a real category for a code that never meant one.
    def inverse_transform($Z) {
        $who = "OrdinalEncoder.inverse_transform";
        this._check_fitted($who);
        $A = _as2d($Z, $who);
        _check_nfeat($A, this.nfeat_, $who);
        $n = nd_shape($A)[0];
        $cols = [];
        $j = 0;
        while ($j < this.nfeat_) {
            $code = nd_ravel(nd_slice($A, [null, [$j, $j + 1, 1]]));
            $k    = nd_shape(this.cats_[$j])[0];
            $lo   = nd_get(nd_min($code, null, null), 0);
            $hi   = nd_get(nd_max($code, null, null), 0);
            if ($lo < 0.0 || $hi > 1.0 * ($k - 1)) {
                throw $who + ": column " + str($j) + " has a code outside "
                      + "0.." + str($k - 1) + " (saw " + str($lo) + ".." + str($hi) + ")";
            }
            append($cols, nd_reshape(nd_take(this.cats_[$j],
                                             nd_astype($code, "i64"), 0), [$n, 1]));
            $j = $j + 1;
        }
        return nd_concat($cols, 1);
    }
    def categories()      { return this.cats_; }
    def n_features()      { return this.nfeat_; }
    def clone() {
        return OrdinalEncoder(this.params_.handle_unknown, this.params_.unknown_value);
    }
}
def OrdinalEncoder($handle_unknown, $unknown_value) {
    return new _OrdinalEncoder($handle_unknown, $unknown_value);
}

// ── OneHotEncoder ───────────────────────────────────────────────────────────
// One block of k_j columns per input feature, concatenated. The indicator block
// is built by broadcasting the column of codes against a row of 0..k-1:
//
//     equal( codes:(n,1) , arange(k):(1,k) )  ->  (n,k)
//
// so there is no loop over rows and no scatter -- one comparison kernel per
// feature.
//
// handle_unknown:
//   "error"   an unseen value raises (default)
//   "ignore"  its row is ALL ZEROS in that block, which is why the code for an
//             unseen value is set to -1: no comparison against 0..k-1 matches.
class _OneHotEncoder extends Estimator {
    def init($handle_unknown) {
        super();
        this.name_ = "OneHotEncoder";
        this.task_ = "transformer";
        $h = $handle_unknown;  if ($h == null) { $h = "error"; }
        if ($h != "error" && $h != "ignore") {
            throw "OneHotEncoder: handle_unknown must be \"error\" or \"ignore\", "
                  + "got \"" + str($h) + "\"";
        }
        this.params_ = {"handle_unknown": $h};
        this.cats_   = [];
        this.nfeat_  = 0;
        this.nout_   = 0;
    }
    def fit($X, $y) {
        $who = "OneHotEncoder.fit";
        $A = _as2d($X, $who);
        _no_nan($A, $who);
        _no_empty($A, $who);
        $d = nd_shape($A)[1];
        $c = [];
        $total = 0;
        $j = 0;
        while ($j < $d) {
            $u = nd_unique(nd_ravel(nd_slice($A, [null, [$j, $j + 1, 1]])));
            append($c, $u);
            $total = $total + nd_shape($u)[0];
            $j = $j + 1;
        }
        this.cats_   = $c;
        this.nfeat_  = $d;
        this.nout_   = $total;
        this.fitted_ = true;
        return this;
    }
    def transform($X) {
        $who = "OneHotEncoder.transform";
        this._check_fitted($who);
        $A = _as2d($X, $who);
        _no_nan($A, $who);
        _check_nfeat($A, this.nfeat_, $who);
        $n = nd_shape($A)[0];
        $blocks = [];
        $j = 0;
        while ($j < this.nfeat_) {
            $col = nd_ravel(nd_slice($A, [null, [$j, $j + 1, 1]]));
            $r   = _encode_col(this.cats_[$j], $col);
            $code = $r[0];
            $hit  = $r[1];
            if (!nd_get(nd_all($hit, null, null), 0)) {
                if (this.params_.handle_unknown == "error") {
                    throw $who + ": column " + str($j) + " contains the unseen "
                          + "value " + str(_first_unseen($col, $hit)) + ". Pass "
                          + "handle_unknown=\"ignore\" to emit an all-zero block "
                          + "for it instead of raising.";
                }
                $code = nd_where($hit, $code, 0.0 - 1.0);
            }
            $k   = nd_shape(this.cats_[$j])[0];
            $row = nd_reshape(nd_astype(nd_arange(0, $k, null), "f64"), [1, $k]);
            append($blocks, nd_astype(nd_equal(nd_reshape($code, [$n, 1]), $row, null), "f64"));
            $j = $j + 1;
        }
        return nd_concat($blocks, 1);
    }
    // argmax per block gives the category back. An all-zero block -- which is
    // what "ignore" produced for an unseen value -- becomes NaN rather than
    // category 0: argmax of all zeros is 0, and reporting the first category
    // for a row that matched nothing would be a fabricated answer.
    def inverse_transform($Z) {
        $who = "OneHotEncoder.inverse_transform";
        this._check_fitted($who);
        $A = _as2d($Z, $who);
        _check_nfeat($A, this.nout_, $who);
        $n = nd_shape($A)[0];
        $cols = [];
        $off = 0;
        $j = 0;
        while ($j < this.nfeat_) {
            $k   = nd_shape(this.cats_[$j])[0];
            $blk = nd_slice($A, [null, [$off, $off + $k, 1]]);
            $am  = nd_astype(nd_ravel(nd_argmax($blk, 1, true)), "i64");
            $val = nd_reshape(nd_take(this.cats_[$j], $am, 0), [$n, 1]);
            $any = nd_sum($blk, 1, true);
            append($cols, nd_where(nd_greater($any, 0.0, null), $val, NAN));
            $off = $off + $k;
            $j = $j + 1;
        }
        return nd_concat($cols, 1);
    }
    def categories()        { return this.cats_; }
    def n_features()        { return this.nfeat_; }
    def n_output_features() { return this.nout_; }
    def clone()             { return OneHotEncoder(this.params_.handle_unknown); }
}
def OneHotEncoder($handle_unknown) { return new _OneHotEncoder($handle_unknown); }

// ── LabelEncoder ────────────────────────────────────────────────────────────
// The one-dimensional case, for y rather than X. Its first argument is still
// the data, so it fits the protocol's fit($X, $y) shape: `le.fit($labels, null)`.
class _LabelEncoder extends Estimator {
    def init() {
        super();
        this.name_   = "LabelEncoder";
        this.task_   = "transformer";
        this.params_ = {};
        this.classes_ = null;
    }
    def fit($y, $ignored) {
        $v = _as1d($y, "LabelEncoder.fit");
        if (nd_shape($v)[0] == 0) {
            throw "LabelEncoder.fit: the label array is empty";
        }
        if (nd_get(nd_any(nd_isnan($v, null), null, null), 0)) {
            throw "LabelEncoder.fit: the labels contain NaN";
        }
        this.classes_ = nd_unique($v);
        this.fitted_  = true;
        return this;
    }
    def transform($y) {
        $who = "LabelEncoder.transform";
        this._check_fitted($who);
        $v = _as1d($y, $who);
        $r = _encode_col(this.classes_, $v);
        if (!nd_get(nd_all($r[1], null, null), 0)) {
            throw $who + ": the unseen label " + str(_first_unseen($v, $r[1]))
                  + " is not one of the " + str(nd_shape(this.classes_)[0])
                  + " labels seen at fit";
        }
        return $r[0];
    }
    def inverse_transform($codes) {
        $who = "LabelEncoder.inverse_transform";
        this._check_fitted($who);
        $c = _as1d($codes, $who);
        $k = nd_shape(this.classes_)[0];
        $lo = nd_get(nd_min($c, null, null), 0);
        $hi = nd_get(nd_max($c, null, null), 0);
        if ($lo < 0.0 || $hi > 1.0 * ($k - 1)) {
            throw $who + ": a code outside 0.." + str($k - 1)
                  + " (saw " + str($lo) + ".." + str($hi) + ")";
        }
        return nd_take(this.classes_, nd_astype($c, "i64"), 0);
    }
    def classes()   { return this.classes_; }
    def n_classes() { if (this.classes_ == null) { return 0; }
                      return nd_shape(this.classes_)[0]; }
    def clone()     { return LabelEncoder(); }
}
def LabelEncoder() { return new _LabelEncoder(); }

// ── SimpleImputer ───────────────────────────────────────────────────────────
// The one estimator here that accepts NaN, because filling it is the job.
//
// strategy: "mean", "median", "most_frequent" or "constant".
//
// An ALL-NaN column raises for the three statistical strategies, because the
// alternative is a statistic computed from no data at all. scikit-learn drops
// such a column with a warning; that silently changes the feature count for
// everything downstream, which is worse than stopping. Under "constant" it is
// fine -- the fill value does not come from the data.
class _SimpleImputer extends Estimator {
    def init($strategy, $fill_value) {
        super();
        this.name_ = "SimpleImputer";
        this.task_ = "transformer";
        $s = $strategy;  if ($s == null) { $s = "mean"; }
        if ($s != "mean" && $s != "median" && $s != "most_frequent" && $s != "constant") {
            throw "SimpleImputer: strategy must be \"mean\", \"median\", "
                  + "\"most_frequent\" or \"constant\", got \"" + str($s) + "\"";
        }
        $f = $fill_value;  if ($f == null) { $f = 0.0; }
        this.params_ = {"strategy": $s, "fill_value": $f};
        this.stats_  = null;
        this.nfeat_  = 0;
    }

    // The median and the mode of a column with NaN in it, without a nanmedian
    // kernel: nd_sort puts NaN LAST (verified, not assumed), so the first
    // `c` elements of the sorted column are exactly its non-NaN values in
    // order, and both statistics are indices into that prefix.
    def _col_stat($col, $j) {
        $s   = this.params_.strategy;
        $ok  = nd_logical_not(nd_isnan($col, null), null);
        $c   = nd_get(nd_count_nonzero($ok, null, null), 0);
        if ($s == "constant") { return 1.0 * this.params_.fill_value; }
        if ($c == 0) {
            throw "SimpleImputer.fit: column " + str($j) + " is entirely NaN, so "
                  + "a \"" + $s + "\" statistic would be computed from no data. "
                  + "Use strategy=\"constant\" if an all-missing column is "
                  + "expected, or drop the column.";
        }
        if ($s == "mean") { return nd_get(nd_nanmean($col, null, null), 0); }

        $srt = nd_sort($col, null);
        $v   = nd_slice($srt, [[0, $c, 1]]);          // the non-NaN values, sorted

        if ($s == "median") {
            $lo = floor(($c - 1) / 2);
            $hi = floor($c / 2);
            return (nd_get($v, $lo) + nd_get($v, $hi)) / 2.0;
        }

        // most_frequent, as a run-length on the sorted column (K16). The run
        // boundaries are where consecutive values differ; the run lengths are
        // the gaps between boundaries; the longest run's first element is the
        // mode. All native: O(n log n) in kernels, no per-unique loop.
        //
        // Ties go to the SMALLER value, because the column is ascending and
        // nd_argmax returns the first maximum -- the same rule scipy's mode
        // gives scikit-learn.
        if ($c == 1) { return nd_get($v, 0); }
        $cuts = nd_nonzero(nd_not_equal(nd_diff($v, null), 0.0, null));
        if (nd_size($cuts) == 0) { return nd_get($v, 0); }   // one run: all equal
        $edges = nd_concat([nd([0.0], "f64"),
                            nd_add(nd_astype($cuts, "f64"), 1.0, null),
                            nd([1.0 * $c], "f64")], 0);
        $lens  = nd_diff($edges, null);
        $best  = nd_get(nd_argmax($lens, null, null), 0);
        return nd_get($v, nd_get($edges, $best));
    }

    def fit($X, $y) {
        $who = "SimpleImputer.fit";
        $A = _as2d($X, $who);
        _no_empty($A, $who);
        $d = nd_shape($A)[1];
        $vals = [];
        $j = 0;
        while ($j < $d) {
            append($vals, this._col_stat(nd_ravel(nd_slice($A, [null, [$j, $j + 1, 1]])), $j));
            $j = $j + 1;
        }
        this.stats_  = nd_reshape(nd($vals, "f64"), [1, $d]);
        this.nfeat_  = $d;
        this.fitted_ = true;
        return this;
    }

    def transform($X) {
        $who = "SimpleImputer.transform";
        this._check_fitted($who);
        $A = _as2d($X, $who);
        _check_nfeat($A, this.nfeat_, $who);
        // One kernel over the whole matrix: the (1,d) statistics broadcast
        // down the rows, so there is no loop at all here.
        return nd_where(nd_isnan($A, null), this.stats_, $A);
    }

    def statistics() { return this.stats_; }
    def n_features() { return this.nfeat_; }
    def clone() { return SimpleImputer(this.params_.strategy, this.params_.fill_value); }
}
def SimpleImputer($strategy, $fill_value) {
    return new _SimpleImputer($strategy, $fill_value);
}

// ── PolynomialFeatures ──────────────────────────────────────────────────────
// The column ORDER is part of the contract, not an implementation detail: a
// model trained on one ordering and scored against another is wrong in a way
// that no test of column counts can detect. The order is scikit-learn's --
// combinations of feature indices, degree by ascending degree:
//
//     d = 2, degree = 2, include_bias   ->  1, x0, x1, x0^2, x0 x1, x1^2
//
// with repetition unless interaction_only, in which case x0^2 and x1^2 are
// left out and only distinct products survive.
class _PolynomialFeatures extends Estimator {
    def init($degree, $interaction_only, $include_bias) {
        super();
        this.name_ = "PolynomialFeatures";
        this.task_ = "transformer";
        $g = $degree;  if ($g == null) { $g = 2; }
        if ($g != floor($g) || $g < 1) {
            throw "PolynomialFeatures: degree must be a positive whole number, got "
                  + str($g);
        }
        $io = $interaction_only;  if ($io == null) { $io = false; }
        $ib = $include_bias;      if ($ib == null) { $ib = true; }
        this.params_ = {"degree": $g, "interaction_only": $io, "include_bias": $ib};
        this.combos_ = [];
        this.nfeat_  = 0;
    }

    def fit($X, $y) {
        $who = "PolynomialFeatures.fit";
        $A = _as2d($X, $who);
        _no_nan($A, $who);
        _no_empty($A, $who);
        $d = nd_shape($A)[1];
        this.combos_ = _combinations($d, this.params_.degree,
                                     this.params_.interaction_only,
                                     this.params_.include_bias);
        this.nfeat_  = $d;
        this.fitted_ = true;
        return this;
    }

    def transform($X) {
        $who = "PolynomialFeatures.transform";
        this._check_fitted($who);
        $A = _as2d($X, $who);
        _no_nan($A, $who);
        _check_nfeat($A, this.nfeat_, $who);
        $n = nd_shape($A)[0];
        $cols = [];
        $i = 0;
        while ($i < len(this.combos_)) {
            $c = this.combos_[$i];
            // The empty combination is the bias column: an empty product is 1.
            $p = nd_ones([$n, 1], "f64");
            $t = 0;
            while ($t < len($c)) {
                $j = $c[$t];
                $p = nd_multiply($p, nd_slice($A, [null, [$j, $j + 1, 1]]), null);
                $t = $t + 1;
            }
            append($cols, $p);
            $i = $i + 1;
        }
        return nd_concat($cols, 1);
    }

    // "1", "x0", "x0 x1", "x0^2" -- so a column can be identified by name
    // rather than by counting.
    def feature_names() {
        this._check_fitted("PolynomialFeatures.feature_names");
        $out = [];
        $i = 0;
        while ($i < len(this.combos_)) {
            $c = this.combos_[$i];
            if (len($c) == 0) { append($out, "1"); }
            else {
                $s = "";
                $t = 0;
                while ($t < len($c)) {
                    // count the run of this index, so x0 x0 prints as x0^2
                    $j = $c[$t];
                    $r = 0;
                    while ($t + $r < len($c) && $c[$t + $r] == $j) { $r = $r + 1; }
                    if ($s != "") { $s = $s + " "; }
                    $s = $s + "x" + str($j);
                    if ($r > 1) { $s = $s + "^" + str($r); }
                    $t = $t + $r;
                }
                append($out, $s);
            }
            $i = $i + 1;
        }
        return $out;
    }

    def n_output_features() { return len(this.combos_); }
    def n_features()        { return this.nfeat_; }
    def clone() {
        return PolynomialFeatures(this.params_.degree,
                                  this.params_.interaction_only,
                                  this.params_.include_bias);
    }
}
def PolynomialFeatures($degree, $interaction_only, $include_bias) {
    return new _PolynomialFeatures($degree, $interaction_only, $include_bias);
}

// Combinations of 0..n-1 of each length, in lexicographic order, with
// repetition unless `interaction_only`. This is CPython's itertools algorithm:
// hold one index vector, emit a SNAPSHOT of it, then advance the rightmost
// index that can still move and reset everything to its right.
//
// The snapshot is not optional. Bantu lists are references, so appending the
// working vector itself would give a result list in which every entry is the
// same list -- and it would hold whatever the vector ended up as.
def _combinations($n, $degree, $interaction_only, $include_bias) {
    $out = [];
    $r = 0;
    if (!$include_bias) { $r = 1; }
    while ($r <= $degree) {
        if ($r == 0) { append($out, []); }
        // With interaction_only there is no combination of r DISTINCT indices
        // once r exceeds n, so that degree contributes no columns at all.
        if ($r > 0 && !($interaction_only && $r > $n)) {
            $idx = [];
            $i = 0;
            while ($i < $r) {
                if ($interaction_only) { append($idx, $i); } else { append($idx, 0); }
                $i = $i + 1;
            }
            // The emit loop is BOUNDED by the number of combinations that
            // exist, and overrunning it raises. The advance rule and the cap
            // rule have to agree for the loop to terminate at all, and they
            // are written in two places -- break that agreement and the loop
            // runs forever. Found by mutation-testing this file: one mutant
            // of the advance rule hung the interpreter for four minutes with
            // no output, which is a far worse failure than a wrong answer.
            // The bound is the exact count, so it is also a statement of what
            // the enumeration is supposed to produce.
            $limit = _binom($n + $r - 1, $r);
            if ($interaction_only) { $limit = _binom($n, $r); }
            $seen  = 0;
            $more = ($n > 0);
            while ($more) {
                $seen = $seen + 1;
                if ($seen > $limit) {
                    throw "PolynomialFeatures: the combination enumerator for "
                          + "n=" + str($n) + ", r=" + str($r) + " passed its "
                          + "exact bound of " + str($limit) + " -- the advance "
                          + "rule and the index cap have gone out of agreement";
                }
                append($out, _copy_list($idx));
                // find the rightmost index that can still be advanced
                $p = $r - 1;
                $found = -1;
                while ($p >= 0) {
                    $cap = $n - 1;
                    if ($interaction_only) { $cap = _cap_for($p, $r, $n); }
                    if ($idx[$p] != $cap) { $found = $p; $p = -1; }
                    else { $p = $p - 1; }
                }
                if ($found < 0) { $more = false; }
                else {
                    $idx[$found] = $idx[$found] + 1;
                    $q = $found + 1;
                    while ($q < $r) {
                        if ($interaction_only) { $idx[$q] = $idx[$q - 1] + 1; }
                        else { $idx[$q] = $idx[$found]; }
                        $q = $q + 1;
                    }
                }
            }
        }
        $r = $r + 1;
    }
    return $out;
}

// The highest value position p may take in a combination of r DISTINCT indices
// drawn from 0..n-1: everything to its right must still fit above it.
def _cap_for($p, $r, $n) { return $p + $n - $r; }

// C(a, b), computed multiplicatively so the intermediate values stay small:
// a factorial of 20 already exceeds what an f64 holds exactly, and this is
// used as a loop bound, where an approximate answer is worse than none.
def _binom($a, $b) {
    if ($b < 0 || $b > $a) { return 0; }
    $k = $b;
    if ($a - $b < $k) { $k = $a - $b; }
    $r = 1;
    $i = 1;
    while ($i <= $k) {
        $r = ($r * ($a - $k + $i)) / $i;
        $i = $i + 1;
    }
    return floor($r + 0.5);
}
