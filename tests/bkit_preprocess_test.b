// ════════════════════════════════════════════════════════════════════════════
//  bkit_preprocess_test.b — tiers 1, 2 and 4 for bkit phase K2.
//
//  Tier 1 (feature): every estimator does what it claims, INCLUDING its
//  failure modes -- an unseen category, a column-count mismatch, an all-NaN
//  column, a zero-variance column.
//
//  Tier 2 (differential): the answers are checked against values worked out by
//  hand or available in closed form, never against this implementation's own
//  output. Where scikit-learn makes a specific choice -- population standard
//  deviation, strictly-greater binarisation, ties going to the smaller value,
//  the polynomial column ORDER -- that choice is asserted, because every one of
//  them is a silently-wrong-answer if it drifts.
//
//  Tier 4 (stress): a 100,000 x 20 fit, and 1,000 clones measured for leaks.
//
//  Tier 3 is `tests/bkit_protocol_test.b`, which runs the contract over every
//  estimator in the package at once.
//
//  THE TWO ASSERTIONS THAT MATTER MOST are the ones no amount of eyeballing
//  would catch:
//
//    * `inverse_transform(transform(x)) == x` to 1e-12, for all four scalers
//      and for a ZERO-VARIANCE column, where a naive implementation divides by
//      zero and returns a matrix of NaN -- silently, because NaN is a
//      perfectly good f64.
//    * `transform` on the wrong number of columns raises. In numba an (n,1)
//      matrix against a stored (1,d) statistic BROADCASTS and gives a
//      well-formed (n,d) matrix of nonsense, so this is the one guard with no
//      natural backstop (DECISIONS K12).
//
//  Run:  bantu run tests/bkit_preprocess_test.b
// ════════════════════════════════════════════════════════════════════════════

include "../numba/numba.b" as np;
include "../bkit/preprocess.b" as pre;

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
def raises($fn, $n) {
    $caught = false;
    try { $fn(); } catch ($e) { $caught = true; }
    ok($caught, $n);
    return null;
}
// The message, not just the fact: an error that does not say WHICH column or
// WHICH value is a worse error.
def raises_with($fn, $needle, $n) {
    $msg = "";
    try { $fn(); } catch ($e) { $msg = str($e); }
    if ($msg == "") { $R.fail = $R.fail + 1; print("  FAIL  " + $n + " (no error raised)"); return null; }
    if (indexOf($msg, $needle) >= 0) { $R.pass = $R.pass + 1; print("  ok    " + $n); }
    else { $R.fail = $R.fail + 1;
           print("  FAIL  " + $n + "   message lacks \"" + $needle + "\": " + $msg); }
    return null;
}
// An assertion whose own expression throws must FAIL, not abort the suite.
// Found by mutation-testing this file: removing the zero-scale guard made
// `inverse_transform` raise inside an `ok(...)` argument, which took the
// process down and hid the forty assertions after it. A suite that stops at
// the first failure reports one bug per run.
def safe_ok($fn, $n) {
    $v   = false;
    $err = "";
    try { $v = $fn(); } catch ($e) { $err = str($e); }
    if ($err != "") {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $n + "   raised: " + $err);
        return null;
    }
    ok($v, $n);
    return null;
}
// Exact list comparison through to_list, so a shape difference also fails.
def lst($a, $want, $n) { eqs(str(nd_to_list($a)), str($want), $n); return null; }
def near($a, $want, $tol, $n) {
    ok(nd_allclose($a, nd($want, "f64"), $tol, $tol, null), $n);
    return null;
}

print("");
print("  bkit preprocessing (K2)");
print("  ════════════════════════════════════════");

$X = nd([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]], "f64");

// ── 1. StandardScaler, against the hand-computed numbers ────────────────────
print("");
print("-- StandardScaler --");
$ss = pre.StandardScaler(null, null);
$Z  = $ss.fit_transform($X, null);
// mean = [2.5, 25]; POPULATION sd = sqrt(1.25) = 1.118033988749895 and ten
// times that. ddof=1 would give sqrt(5/3) instead and nothing would error.
near(nd_slice($Z, [null, [0, 1, 1]]),
     [[-1.3416407864998738], [-0.4472135954999579],
      [0.4472135954999579], [1.3416407864998738]], 1.0e-14,
     "the column is (x - mean) / POPULATION sd, to 1e-14");
near($ss.center(), [[2.5, 25.0]], 1.0e-15, "center_ is the column mean");
near($ss.scale(), [[1.118033988749895, 11.18033988749895]], 1.0e-14,
     "scale_ is the ddof=0 standard deviation, not ddof=1");
ok(nd_allclose(nd_mean($Z, 0, true), nd_zeros([1, 2], "f64"), 1.0e-15, 1.0e-15, null),
   "each transformed column has mean 0");
ok(nd_allclose(nd_std($Z, 0, true, 0), nd_ones([1, 2], "f64"), 1.0e-14, 1.0e-14, null),
   "and unit standard deviation");
eqs($ss.n_features(), 2, "n_features() is recorded at fit");

// with_mean / with_std off
$ss2 = pre.StandardScaler(false, true);
$ss2.fit($X, null);
near($ss2.center(), [[0.0, 0.0]], 1.0e-15, "with_mean=false leaves the centre at zero");
$ss3 = pre.StandardScaler(true, false);
$ss3.fit($X, null);
near($ss3.scale(), [[1.0, 1.0]], 1.0e-15, "with_std=false leaves the scale at one");

// ── 2. the round trip, for all four scalers ─────────────────────────────────
print("");
print("-- GATE: inverse_transform(transform(x)) == x to 1e-12 --");
nd_seed(4242);
$Xr = nd_random_uniform([40, 6], -50.0, 50.0);
$scalers = [["StandardScaler", pre.StandardScaler(null, null)],
            ["MinMaxScaler",   pre.MinMaxScaler(null, null)],
            ["MinMaxScaler(-3,7)", pre.MinMaxScaler(0.0 - 3.0, 7.0)],
            ["RobustScaler",   pre.RobustScaler(null, null)],
            ["MaxAbsScaler",   pre.MaxAbsScaler()]];
$i = 0;
while ($i < len($scalers)) {
    $nm = $scalers[$i][0];
    $sc = $scalers[$i][1];
    safe_ok(def() {
        $Zi = $sc.fit_transform($Xr, null);
        return nd_allclose($sc.inverse_transform($Zi), $Xr, 1.0e-12, 1.0e-12, null);
    }, $nm + " round-trips to 1e-12");
    $i = $i + 1;
}

// ── 3. GATE: a zero-variance column does not divide by zero ─────────────────
print("");
print("-- GATE: a constant column --");
// Column 0 is constant, so its sd, range, IQR and max-abs are all zero. The
// naive implementation divides by zero and returns NaN for every row of it.
$Xc = nd([[7.0, 1.0], [7.0, 2.0], [7.0, 3.0]], "f64");
$i = 0;
while ($i < len($scalers)) {
    $nm = $scalers[$i][0];
    $sc = $scalers[$i][1].clone();
    safe_ok(def() {
        $Zc = $sc.fit_transform($Xc, null);
        return !nd_get(nd_any(nd_isnan($Zc, null), null, null), 0);
    }, $nm + " on a constant column produces no NaN");
    safe_ok(def() {
        $Zc = $sc.fit_transform($Xc, null);
        return nd_allclose($sc.inverse_transform($Zc), $Xc, 1.0e-12, 1.0e-12, null);
    }, "  and still round-trips EXACTLY -- scale 1 makes x - centre vanish");
    $i = $i + 1;
}
$ssc = pre.StandardScaler(null, null);
lst($ssc.fit_transform($Xc, null)[0], [0, -1.224744871391589],
    "the constant column transforms to exactly 0, not to NaN");

// ── 4. MinMax / Robust / MaxAbs, each against a closed form ─────────────────
print("");
print("-- MinMaxScaler --");
lst(pre.MinMaxScaler(null, null).fit_transform(nd([[1.0], [3.0], [5.0]], "f64"), null),
    [[0], [0.5], [1]], "the default range is [0, 1]");
lst(pre.MinMaxScaler(0.0 - 1.0, 1.0).fit_transform(nd([[1.0], [3.0], [5.0]], "f64"), null),
    [[-1], [0], [1]], "an explicit range [-1, 1]");
raises_with(def() { return pre.MinMaxScaler(1.0, 1.0).fit($X, null); }, "increasing",
            "a non-increasing range is refused at fit");

print("");
print("-- RobustScaler --");
// median 2.5; q25 = 1.75 and q75 = 3.25 by linear interpolation (NumPy's
// default method), so the IQR is 1.5.
$rb = pre.RobustScaler(null, null);
$Zr = $rb.fit_transform(nd([[1.0], [2.0], [3.0], [4.0]], "f64"), null);
near($rb.center(), [[2.5]], 1.0e-15, "center_ is the median");
near($rb.scale(), [[1.5]], 1.0e-15, "scale_ is the inter-quartile range");
near($Zr, [[-1.0], [0.0 - 1.0 / 3.0], [1.0 / 3.0], [1.0]], 1.0e-15,
     "and the transform matches (x - median) / IQR");
// the point of the estimator: one wild value must not move the statistics
$Xo = nd([[1.0], [2.0], [3.0], [4.0], [1000000.0]], "f64");
$rb2 = pre.RobustScaler(null, null);
$rb2.fit($Xo, null);
near($rb2.center(), [[3.0]], 1.0e-15,
     "an outlier of 1e6 leaves the median where it belongs");
$ss4 = pre.StandardScaler(null, null);
$ss4.fit($Xo, null);
ok(nd_get($ss4.center(), [0, 0]) > 200000.0,
   "  where the mean is dragged to 200002 -- which is why RobustScaler exists");

print("");
print("-- MaxAbsScaler --");
lst(pre.MaxAbsScaler().fit_transform(nd([[0.0 - 4.0], [2.0]], "f64"), null),
    [[-1], [0.5]], "divide by max(|x|)");
lst(pre.MaxAbsScaler().fit_transform(nd([[0.0, 3.0], [0.0, 0.0 - 6.0]], "f64"), null),
    [[0, 0.5], [0, -1]], "and do NOT centre, so a structural zero stays zero");

// ── 5. Normalizer and Binarizer ─────────────────────────────────────────────
print("");
print("-- Normalizer --");
lst(pre.Normalizer(null).fit_transform(nd([[3.0, 4.0]], "f64"), null),
    [[0.6, 0.8]], "l2 is the default: 3,4 -> 0.6,0.8");
lst(pre.Normalizer("l1").fit_transform(nd([[1.0, 0.0 - 3.0]], "f64"), null),
    [[0.25, -0.75]], "l1 divides by the sum of absolute values");
lst(pre.Normalizer("max").fit_transform(nd([[2.0, 0.0 - 4.0]], "f64"), null),
    [[0.5, -1]], "max divides by the largest absolute value");
lst(pre.Normalizer(null).fit_transform(nd([[0.0, 0.0], [3.0, 4.0]], "f64"), null),
    [[0, 0], [0.6, 0.8]], "a zero row stays zero rather than becoming NaN");
raises_with(def() { return pre.Normalizer("l3"); }, "l1", "an unknown norm is refused");

print("");
print("-- Binarizer --");
lst(pre.Binarizer(null).fit_transform(nd([[0.0 - 1.0, 0.0, 1.0]], "f64"), null),
    [[0, 0, 1]], "the default threshold 0 is STRICT, so 0 -> 0");
lst(pre.Binarizer(0.5).fit_transform(nd([[0.4, 0.5, 0.6]], "f64"), null),
    [[0, 0, 1]], "and an explicit threshold is strict too");

// ── 6. OrdinalEncoder ───────────────────────────────────────────────────────
print("");
print("-- OrdinalEncoder --");
$Xo1 = nd([[10.0], [30.0], [10.0], [20.0]], "f64");
$oe = pre.OrdinalEncoder(null, null);
$Co = $oe.fit_transform($Xo1, null);
lst($Co, [[0], [2], [0], [1]], "categories are numbered in ASCENDING value order");
lst($oe.categories()[0], [10, 20, 30], "and categories() reports them sorted");
lst($oe.inverse_transform($Co), [[10], [30], [10], [20]], "inverse_transform recovers the values");
raises_with(def() { return $oe.transform(nd([[40.0]], "f64")); }, "unseen",
            "an unseen value raises by default");
raises_with(def() { return $oe.transform(nd([[40.0]], "f64")); }, "40",
            "  and the message names the offending value");
raises_with(def() { return $oe.transform(nd([[40.0]], "f64")); }, "column 0",
            "  and the column it was in");
$oe2 = pre.OrdinalEncoder("use_encoded_value", 0.0 - 1.0);
$oe2.fit($Xo1, null);
lst($oe2.transform(nd([[40.0], [20.0]], "f64")), [[-1], [1]],
    "handle_unknown=\"use_encoded_value\" maps an unknown to the given code");
raises_with(def() { return pre.OrdinalEncoder("use_encoded_value", null); }, "unknown_value",
            "use_encoded_value without an unknown_value is refused at construction");
raises_with(def() { return pre.OrdinalEncoder("whatever", null); }, "handle_unknown",
            "an unknown handle_unknown is refused");
// two columns at once, each with its own category set
$oe3 = pre.OrdinalEncoder(null, null);
lst($oe3.fit_transform(nd([[5.0, 100.0], [7.0, 100.0], [5.0, 200.0]], "f64"), null),
    [[0, 0], [1, 0], [0, 1]], "each column is encoded against its OWN categories");
raises_with(def() { return $oe3.inverse_transform(nd([[0.0, 9.0]], "f64")); }, "outside",
            "a code outside the category range raises instead of being clamped");

// ── 7. OneHotEncoder ────────────────────────────────────────────────────────
print("");
print("-- OneHotEncoder --");
$oh = pre.OneHotEncoder(null);
$H  = $oh.fit_transform(nd([[10.0], [30.0], [20.0]], "f64"), null);
lst($H, [[1, 0, 0], [0, 0, 1], [0, 1, 0]], "one column per category, in sorted order");
eqs($oh.n_output_features(), 3, "n_output_features() is the total category count");
$oh2 = pre.OneHotEncoder(null);
$H2  = $oh2.fit_transform(nd([[0.0, 5.0], [1.0, 5.0], [0.0, 7.0]], "f64"), null);
lst($H2, [[1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]],
    "two features give two blocks, concatenated in feature order");
eqs($oh2.n_output_features(), 4, "and four output columns in total");
lst($oh2.inverse_transform($H2), [[0, 5], [1, 5], [0, 7]], "inverse_transform recovers the values");
raises_with(def() { return $oh2.transform(nd([[9.0, 5.0]], "f64")); }, "unseen",
            "an unseen value raises by default");
$oh3 = pre.OneHotEncoder("ignore");
$oh3.fit(nd([[10.0], [20.0]], "f64"), null);
lst($oh3.transform(nd([[10.0], [99.0]], "f64")), [[1, 0], [0, 0]],
    "handle_unknown=\"ignore\" emits an ALL-ZERO block for an unknown");
// and the inverse of an all-zero block is NaN, not category 0 -- argmax of all
// zeros is 0, and reporting the first category would be a fabricated answer
$inv = $oh3.inverse_transform(nd([[1.0, 0.0], [0.0, 0.0]], "f64"));
eqs(nd_get($inv, [0, 0]), 10.0, "inverting a real block gives the category back");
ok(nd_get(nd_isnan(nd_slice($inv, [[1, 2, 1], null]), null), [0, 0]),
   "inverting an all-zero block gives NaN, not the first category");
raises_with(def() { return $oh2.inverse_transform($H); }, "4 feature",
            "inverse_transform checks the OUTPUT width, not the input width");

// ── 8. LabelEncoder ─────────────────────────────────────────────────────────
print("");
print("-- LabelEncoder --");
$le = pre.LabelEncoder();
$yc = $le.fit_transform([3.0, 1.0, 1.0, 3.0, 2.0], null);
lst($yc, [2, 0, 0, 2, 1], "labels are numbered in ascending order");
lst($le.classes(), [1, 2, 3], "classes() reports them");
eqs($le.n_classes(), 3, "n_classes() counts them");
lst($le.inverse_transform($yc), [3, 1, 1, 3, 2], "inverse_transform recovers the labels");
raises_with(def() { return $le.transform([9.0]); }, "unseen", "an unseen label raises");
raises_with(def() { return $le.fit(nd([[1.0, 2.0], [3.0, 4.0]], "f64"), null); }, "one column",
            "a two-column input is refused -- this estimator is for y");
eqs(pre.LabelEncoder().n_classes(), 0, "an unfitted LabelEncoder reports zero classes");

// ── 9. SimpleImputer ────────────────────────────────────────────────────────
print("");
print("-- SimpleImputer --");
$Xn = nd([[1.0, NAN], [NAN, 2.0], [3.0, 4.0]], "f64");
lst(pre.SimpleImputer(null, null).fit_transform($Xn, null),
    [[1, 3], [2, 2], [3, 4]], "mean: the NaN takes the column's mean of the rest");
// column 0 non-NaN = [1, 3] -> mean 2; column 1 non-NaN = [2, 4] -> mean 3
lst(pre.SimpleImputer("median", null).fit_transform(
        nd([[1.0], [2.0], [NAN], [4.0]], "f64"), null),
    [[1], [2], [2], [4]], "median of [1,2,4] is 2, with the NaN excluded");
lst(pre.SimpleImputer("median", null).fit_transform(
        nd([[1.0], [2.0], [3.0], [4.0]], "f64"), null),
    [[1], [2], [3], [4]], "an even count averages the two middle values (nothing to fill here)");
$imEven = pre.SimpleImputer("median", null);
$imEven.fit(nd([[1.0], [2.0], [3.0], [4.0]], "f64"), null);
near($imEven.statistics(), [[2.5]], 1.0e-15, "  and that average is 2.5, not 2 or 3");
lst(pre.SimpleImputer("most_frequent", null).fit_transform(
        nd([[5.0], [5.0], [2.0], [NAN], [7.0]], "f64"), null),
    [[5], [5], [2], [5], [7]], "most_frequent finds the mode by run length");
$imTie = pre.SimpleImputer("most_frequent", null);
$imTie.fit(nd([[1.0], [1.0], [2.0], [2.0]], "f64"), null);
near($imTie.statistics(), [[1.0]], 1.0e-15,
     "a tie goes to the SMALLER value, as scipy's mode does");
lst(pre.SimpleImputer("constant", 9.0).fit_transform(nd([[NAN], [1.0]], "f64"), null),
    [[9], [1]], "constant fills with the given value");
lst(pre.SimpleImputer("constant", null).fit_transform(nd([[NAN], [1.0]], "f64"), null),
    [[0], [1]], "and its default fill value is 0");

print("");
print("-- GATE: an all-NaN column --");
$Xall = nd([[NAN, 1.0], [NAN, 2.0]], "f64");
raises_with(def() { return pre.SimpleImputer("mean", null).fit($Xall, null); }, "entirely NaN",
            "mean on an all-NaN column raises rather than storing NaN");
raises_with(def() { return pre.SimpleImputer("mean", null).fit($Xall, null); }, "column 0",
            "  and names the column");
raises_with(def() { return pre.SimpleImputer("median", null).fit($Xall, null); }, "entirely NaN",
            "median likewise");
raises_with(def() { return pre.SimpleImputer("most_frequent", null).fit($Xall, null); },
            "entirely NaN", "most_frequent likewise");
lst(pre.SimpleImputer("constant", 0.0 - 1.0).fit_transform($Xall, null),
    [[-1, 1], [-1, 2]], "but \"constant\" accepts it -- the value is not from the data");
raises_with(def() { return pre.SimpleImputer("mode", null); }, "strategy",
            "an unknown strategy is refused at construction");
// a single non-NaN value is enough for every strategy
lst(pre.SimpleImputer("median", null).fit_transform(nd([[NAN], [6.0]], "f64"), null),
    [[6], [6]], "one non-NaN value is a valid median");
lst(pre.SimpleImputer("most_frequent", null).fit_transform(nd([[NAN], [6.0]], "f64"), null),
    [[6], [6]], "and a valid mode");
lst(pre.SimpleImputer("most_frequent", null).fit_transform(
        nd([[4.0], [4.0], [NAN]], "f64"), null),
    [[4], [4], [4]], "a column that is all one value has one run");

// ── 10. PolynomialFeatures ──────────────────────────────────────────────────
print("");
print("-- PolynomialFeatures --");
$pf = pre.PolynomialFeatures(null, null, null);
$P  = $pf.fit_transform(nd([[2.0, 3.0]], "f64"), null);
// THE ORDER IS THE CONTRACT: a model trained on one ordering and scored
// against another is wrong in a way no count of columns detects.
eqs(str($pf.feature_names()), str(["1", "x0", "x1", "x0^2", "x0 x1", "x1^2"]),
    "degree 2 enumerates in scikit-learn's order");
lst($P, [[1, 2, 3, 4, 6, 9]], "and the values follow that order");
eqs($pf.n_output_features(), 6, "six output features for d=2, degree=2, with bias");

$pfi = pre.PolynomialFeatures(2, true, true);
$pfi.fit(nd([[2.0, 3.0]], "f64"), null);
eqs(str($pfi.feature_names()), str(["1", "x0", "x1", "x0 x1"]),
    "interaction_only drops the squared terms");
lst($pfi.transform(nd([[2.0, 3.0]], "f64")), [[1, 2, 3, 6]], "  and its values");

$pfn = pre.PolynomialFeatures(2, false, false);
$pfn.fit(nd([[2.0, 3.0]], "f64"), null);
eqs(str($pfn.feature_names()), str(["x0", "x1", "x0^2", "x0 x1", "x1^2"]),
    "include_bias=false drops the leading 1");

$pf3 = pre.PolynomialFeatures(3, false, true);
$pf3.fit(nd([[2.0, 3.0]], "f64"), null);
eqs(str($pf3.feature_names()),
    str(["1", "x0", "x1", "x0^2", "x0 x1", "x1^2",
         "x0^3", "x0^2 x1", "x0 x1^2", "x1^3"]),
    "degree 3 adds the ten terms of two variables, degree by ascending degree");
lst($pf3.transform(nd([[2.0, 3.0]], "f64")), [[1, 2, 3, 4, 6, 9, 8, 12, 18, 27]],
    "  and every one of them is the right product");

$pf3i = pre.PolynomialFeatures(2, true, true);
$pf3i.fit(nd([[1.0, 2.0, 3.0]], "f64"), null);
eqs(str($pf3i.feature_names()), str(["1", "x0", "x1", "x2", "x0 x1", "x0 x2", "x1 x2"]),
    "three features, interactions only: the three pairs, in lexicographic order");
// degree above the feature count, interactions only: those degrees contribute
// nothing rather than producing a malformed combination
$pfhi = pre.PolynomialFeatures(4, true, true);
$pfhi.fit(nd([[1.0, 2.0]], "f64"), null);
eqs(str($pfhi.feature_names()), str(["1", "x0", "x1", "x0 x1"]),
    "degree 4 with 2 interacting features stops where the combinations run out");
eqs(pre.PolynomialFeatures(1, false, true).fit(nd([[1.0, 2.0]], "f64"), null)
        .n_output_features(), 3, "degree 1 with bias is just [1, x0, x1]");
raises_with(def() { return pre.PolynomialFeatures(0, null, null); }, "positive",
            "degree 0 is refused");
raises_with(def() { return pre.PolynomialFeatures(1.5, null, null); }, "whole number",
            "a fractional degree is refused");

// The output width against the CLOSED FORM, over twenty (features, degree)
// pairs. Asserting the list of names pins the small cases; this pins the
// enumerator itself, where an off-by-one in the advance rule shows up only at
// a size nobody writes out by hand. With repetition and bias the count is
// C(d+g, g); interactions only, it is the sum of C(d, r) for r = 0..g.
def binom($a, $b) {
    if ($b < 0 || $b > $a) { return 0; }
    $k = $b;
    if ($a - $b < $k) { $k = $a - $b; }
    $r = 1;
    $i = 1;
    while ($i <= $k) { $r = ($r * ($a - $k + $i)) / $i; $i = $i + 1; }
    return floor($r + 0.5);
}
$pcases = [[1, 1], [1, 4], [2, 2], [2, 3], [3, 2], [3, 3], [5, 3], [4, 4], [7, 2], [2, 6]];
$i = 0;
while ($i < len($pcases)) {
    $d = $pcases[$i][0];
    $g = $pcases[$i][1];
    $Xp = nd_ones([1, $d], "f64");
    $pr = pre.PolynomialFeatures($g, false, true);
    $pr.fit($Xp, null);
    eqs($pr.n_output_features(), binom($d + $g, $g),
        "d=" + str($d) + " degree=" + str($g) + ": C(d+g, g) columns with repetition");
    $pin = pre.PolynomialFeatures($g, true, true);
    $pin.fit($Xp, null);
    $want = 0;
    $r = 0;
    while ($r <= $g) { $want = $want + binom($d, $r); $r = $r + 1; }
    eqs($pin.n_output_features(), $want,
        "  and sum C(d, r) columns with interactions only");
    $i = $i + 1;
}

// ── 11. GATE: the feature-count guard ───────────────────────────────────────
print("");
print("-- GATE: a column-count mismatch --");
// Without this check an (n,1) input against a (1,2) statistic BROADCASTS and
// returns a well-formed (n,2) matrix of nonsense. There is no natural backstop.
$g = pre.StandardScaler(null, null);
$g.fit($X, null);
raises_with(def() { return $g.transform(nd([[1.0]], "f64")); }, "2 feature",
            "one column against a two-column fit raises, instead of broadcasting");
raises_with(def() { return $g.transform(nd([[1.0, 2.0, 3.0]], "f64")); }, "given 3",
            "three columns likewise, and the message says how many");
raises_with(def() { return $g.inverse_transform(nd([[1.0]], "f64")); }, "2 feature",
            "inverse_transform is guarded too");
$guarded = [["MinMaxScaler", pre.MinMaxScaler(null, null)],
            ["RobustScaler", pre.RobustScaler(null, null)],
            ["MaxAbsScaler", pre.MaxAbsScaler()],
            ["Normalizer",   pre.Normalizer(null)],
            ["Binarizer",    pre.Binarizer(null)],
            ["OrdinalEncoder", pre.OrdinalEncoder(null, null)],
            ["OneHotEncoder",  pre.OneHotEncoder(null)],
            ["SimpleImputer",  pre.SimpleImputer(null, null)],
            ["PolynomialFeatures", pre.PolynomialFeatures(null, null, null)]];
$i = 0;
while ($i < len($guarded)) {
    $nm = $guarded[$i][0];
    $e  = $guarded[$i][1];
    $e.fit($X, null);
    raises(def() { return $e.transform(nd([[1.0]], "f64")); },
           $nm + " checks the column count too");
    $i = $i + 1;
}

// ── 12. GATE: NaN is refused everywhere but the imputer ─────────────────────
print("");
print("-- GATE: NaN refused, by column, everywhere but SimpleImputer --");
$Xnan = nd([[1.0, 2.0], [3.0, NAN]], "f64");
$refusers = [["StandardScaler", pre.StandardScaler(null, null)],
             ["MinMaxScaler",   pre.MinMaxScaler(null, null)],
             ["RobustScaler",   pre.RobustScaler(null, null)],
             ["MaxAbsScaler",   pre.MaxAbsScaler()],
             ["Normalizer",     pre.Normalizer(null)],
             ["Binarizer",      pre.Binarizer(null)],
             ["OrdinalEncoder", pre.OrdinalEncoder(null, null)],
             ["OneHotEncoder",  pre.OneHotEncoder(null)],
             ["PolynomialFeatures", pre.PolynomialFeatures(null, null, null)]];
$i = 0;
while ($i < len($refusers)) {
    $nm = $refusers[$i][0];
    $e  = $refusers[$i][1];
    raises_with(def() { return $e.fit($Xnan, null); }, "column 1",
                $nm + " refuses NaN at fit and names the column");
    $i = $i + 1;
}
// and at transform, not only at fit -- a scaler fitted on clean data and given
// dirty data would otherwise emit a NaN column silently
$cl = pre.StandardScaler(null, null);
$cl.fit($X, null);
raises_with(def() { return $cl.transform($Xnan); }, "column 1",
            "and at transform, so dirty data cannot slip past a clean fit");
raises_with(def() { return $cl.inverse_transform($Xnan); }, "column 1",
            "and at inverse_transform, so there is no third case to remember");
raises_with(def() { return pre.LabelEncoder().fit([1.0, NAN], null); }, "NaN",
            "LabelEncoder refuses NaN labels");
// Binarizer is the one where propagation would be most harmful: NaN > t is
// false, so a missing value would be recorded as "below the threshold".
$bz = pre.Binarizer(0.0);
$bz.fit($X, null);
raises(def() { return $bz.transform($Xnan); },
       "Binarizer refuses NaN rather than recording it as below the threshold");

// ── 13. the input shapes a user actually passes ─────────────────────────────
print("");
print("-- input handling --");
lst(pre.Binarizer(1.5).fit_transform([[1.0], [2.0]], null), [[0], [1]],
    "a list of rows is accepted");
lst(pre.Binarizer(1.5).fit_transform(nd([1.0, 2.0], "f64"), null), [[0], [1]],
    "a 1-D array becomes a single column");
lst(pre.Binarizer(1.5).fit_transform(nd([1, 2], "i64"), null), [[0], [1]],
    "an i64 array is coerced to f64");
raises_with(def() { return pre.StandardScaler(null, null).fit(null, null); }, "null",
            "a null X says so");
raises_with(def() { return pre.StandardScaler(null, null).fit("hello", null); }, "string",
            "a string X names its type");
raises_with(def() { return pre.StandardScaler(null, null).fit(
                nd_zeros([2, 2, 2], "f64"), null); }, "3 dimensions",
            "a 3-D X is refused");
raises_with(def() { return pre.StandardScaler(null, null).fit(
                nd_zeros([0, 3], "f64"), null); }, "no rows",
            "an empty X is refused rather than producing NaN statistics");
raises_with(def() { return pre.StandardScaler(null, null).transform($X); }, "call fit",
            "transform before fit says to call fit");

// ── 13b. an empty fit is refused by EVERY estimator, and says why ───────────
// It used to be refused by the four scalers only. The encoders fitted happily
// on zero rows -- nd_unique of an empty column is an empty category list --
// and the failure arrived on the first real transform as "nd_take: index -1 is
// out of range for axis 0 with 0 elements": a kernel message about an index,
// for a mistake made in an earlier call. One rule now, in one place.
$emptyX = nd_zeros([0, 2], "f64");
$allest = [["StandardScaler", pre.StandardScaler(null, null)],
           ["MinMaxScaler",   pre.MinMaxScaler(null, null)],
           ["RobustScaler",   pre.RobustScaler(null, null)],
           ["MaxAbsScaler",   pre.MaxAbsScaler()],
           ["Normalizer",     pre.Normalizer(null)],
           ["Binarizer",      pre.Binarizer(null)],
           ["OrdinalEncoder", pre.OrdinalEncoder(null, null)],
           ["OneHotEncoder",  pre.OneHotEncoder(null)],
           ["SimpleImputer",  pre.SimpleImputer(null, null)],
           ["PolynomialFeatures", pre.PolynomialFeatures(null, null, null)]];
$i = 0;
while ($i < len($allest)) {
    $nm = $allest[$i][0];
    $e  = $allest[$i][1];
    raises_with(def() { return $e.fit($emptyX, null); }, "no rows",
                $nm + " refuses a fit on zero rows, and says so");
    $i = $i + 1;
}
raises_with(def() { return pre.LabelEncoder().fit(nd_zeros([0], "f64"), null); }, "empty",
            "LabelEncoder refuses an empty label array");
// but transforming zero rows is legitimate: zero rows in, zero rows out
$et = pre.StandardScaler(null, null);
$et.fit($X, null);
eqs(str(nd_shape($et.transform($emptyX))), str([0, 2]),
    "transforming zero rows is still allowed and gives zero rows back");

// ── 14. set_params re-fits with the new parameter ───────────────────────────
print("");
print("-- set_params --");
$sp = pre.MinMaxScaler(null, null);
$sp.fit($X, null);
$sp.set_params({"range_low": 0.0 - 1.0, "range_high": 1.0});
ok(!$sp.fitted(), "set_params invalidates the fit");
lst($sp.fit_transform(nd([[1.0], [3.0], [5.0]], "f64"), null), [[-1], [0], [1]],
    "and the next fit uses the new range -- fit reads params_, not a captured local");
eqs(str(pre.SimpleImputer("median", null).get_params()),
    str({"strategy": "median", "fill_value": 0.0}),
    "get_params reports exactly what the constructor took");

// ── 15. tier 4: stress ──────────────────────────────────────────────────────
print("");
print("-- stress --");
nd_seed(99);
$big = nd_random_uniform([100000, 20], 0.0 - 10.0, 10.0);
$t0 = clock();
$bs = pre.StandardScaler(null, null);
$Zb = $bs.fit_transform($big, null);
$ms = clock() - $t0;
print("        100000x20 fit_transform: " + str($ms) + " ms");
ok(nd_allclose(nd_mean($Zb, 0, true), nd_zeros([1, 20], "f64"), 1.0e-12, 1.0e-12, null),
   "a 100000x20 StandardScaler centres every column");
safe_ok(def() { return nd_allclose($bs.inverse_transform($Zb), $big,
                                   1.0e-10, 1.0e-10, null); },
        "and still round-trips at that size");
// One Bantu loop per COLUMN, never per row (architecture §3.1). A per-row loop
// would make this take minutes rather than milliseconds, so the budget is the
// assertion that the loops are where the design says they are.
ok($ms < 20000, "  in under 20 seconds, which a per-ROW loop could not manage");

$t1 = clock();
$oeb = pre.OrdinalEncoder(null, null);
$oeb.fit_transform(nd_astype(nd_random_int([50000, 4], 0, 200), "f64"), null);
print("        50000x4 ordinal encode: " + str(clock() - $t1) + " ms");
ok(true, "the searchsorted encoder handles 50000 rows x 200 categories");

$before = nd_live_bytes();
$j = 0;
while ($j < 1000) {
    $tmp = pre.StandardScaler(null, null).clone();
    $j = $j + 1;
}
$after = nd_live_bytes();
print("        live bytes " + str($before) + " -> " + str($after) + " over 1000 clones");
ok($after <= $before, "1000 clones do not grow live memory");

// a cloned FITTED scaler is unfitted and shares nothing (K2)
$src = pre.StandardScaler(null, null);
$src.fit($X, null);
$cp = $src.clone();
ok(!$cp.fitted(), "a clone of a fitted scaler is unfitted");
$cp.fit(nd([[100.0, 1.0], [200.0, 2.0]], "f64"), null);
near($src.center(), [[2.5, 25.0]], 1.0e-15,
     "and fitting the clone does not touch the source's statistics");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
