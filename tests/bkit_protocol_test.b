// ════════════════════════════════════════════════════════════════════════════
//  bkit_protocol_test.b — TIER 3 for the whole package.
//
//  One table, every estimator in bkit, one call to `base.protocol_check` each.
//  A protocol with eleven implementers fragments silently (DECISIONS K9): an
//  estimator that forgets `clone()`, or whose `get_params()` omits a parameter
//  that `set_params()` accepts, breaks `GridSearchCV` in a way that surfaces as
//  a confusing SCORE rather than an error -- and a plausible wrong number is
//  the only kind worth building machinery against.
//
//  WHY THIS IS ITS OWN FILE, and not a section of `bkit_base_test.b` as K1's
//  roadmap first planned (DECISIONS K17). The registry has to name estimators
//  from every module, so wherever it lives, that file includes all of bkit.
//  `base.b` is deliberately tiny because `btext` depends on it alone ([K7]);
//  making its test include eleven modules would mean a bug in `ensemble.b`
//  fails `base.b`'s test, which is exactly the coupling `base.b` exists to
//  avoid. `bkit_base_test.b` keeps its three deliberate doubles -- including
//  the one that forgets `clone()`, which proves the check can FAIL -- and this
//  file carries the real estimators.
//
//  EACH ENTRY BRINGS ITS OWN DATA. A single shared X cannot serve the whole
//  package: `LabelEncoder` works on one label column, the encoders want
//  repeated categorical values rather than 40 distinct floats, and a regressor
//  wants a y. Sharing one matrix would have meant either skipping the
//  estimators it did not suit or giving them a fit they do not represent.
//
//  Run:  bantu run tests/bkit_protocol_test.b
// ════════════════════════════════════════════════════════════════════════════

include "../numba/numba.b" as np;
include "../bkit/base.b"       as base;
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

print("");
print("  bkit protocol conformance — every estimator in the package");
print("  ════════════════════════════════════════");

nd_seed(11);
// Continuous: for the scalers and PolynomialFeatures.
$Xc = nd_random_uniform([20, 3], 0.0 - 5.0, 5.0);
$yc = nd_random_uniform([20], 0.0, 1.0);
// Categorical: small integers, repeated, so nd_unique finds real categories.
$Xk = nd_astype(nd_random_int([20, 3], 0, 4), "f64");
// One label column, for LabelEncoder.
$Xl = nd_astype(nd_random_int([20], 0, 3), "f64");

// ── the registry ────────────────────────────────────────────────────────────
// Every phase from here appends to this list. An estimator missing from it is
// an estimator whose contract nobody checked.
$registry = [
    ["StandardScaler",     pre.StandardScaler(null, null),              $Xc, $yc],
    ["StandardScaler(F,T)", pre.StandardScaler(false, true),            $Xc, $yc],
    ["MinMaxScaler",       pre.MinMaxScaler(null, null),                $Xc, $yc],
    ["MinMaxScaler(-1,1)", pre.MinMaxScaler(0.0 - 1.0, 1.0),            $Xc, $yc],
    ["RobustScaler",       pre.RobustScaler(null, null),                $Xc, $yc],
    ["RobustScaler(10,90)", pre.RobustScaler(10.0, 90.0),               $Xc, $yc],
    ["MaxAbsScaler",       pre.MaxAbsScaler(),                          $Xc, $yc],
    ["Normalizer(l2)",     pre.Normalizer(null),                        $Xc, $yc],
    ["Normalizer(l1)",     pre.Normalizer("l1"),                        $Xc, $yc],
    ["Normalizer(max)",    pre.Normalizer("max"),                       $Xc, $yc],
    ["Binarizer",          pre.Binarizer(null),                         $Xc, $yc],
    ["Binarizer(0.5)",     pre.Binarizer(0.5),                          $Xc, $yc],
    ["OrdinalEncoder",     pre.OrdinalEncoder(null, null),              $Xk, $yc],
    ["OrdinalEncoder(use)", pre.OrdinalEncoder("use_encoded_value", 0.0 - 1.0), $Xk, $yc],
    ["OneHotEncoder",      pre.OneHotEncoder(null),                     $Xk, $yc],
    ["OneHotEncoder(ign)", pre.OneHotEncoder("ignore"),                 $Xk, $yc],
    ["LabelEncoder",       pre.LabelEncoder(),                          $Xl, null],
    ["SimpleImputer(mean)", pre.SimpleImputer(null, null),              $Xc, $yc],
    ["SimpleImputer(med)", pre.SimpleImputer("median", null),           $Xc, $yc],
    ["SimpleImputer(mode)", pre.SimpleImputer("most_frequent", null),   $Xk, $yc],
    ["SimpleImputer(const)", pre.SimpleImputer("constant", 0.0),        $Xc, $yc],
    ["PolynomialFeatures", pre.PolynomialFeatures(null, null, null),    $Xc, $yc],
    ["PolynomialFeatures(int)", pre.PolynomialFeatures(2, true, false), $Xc, $yc]
];

print("");
print("-- the contract, estimator by estimator --");
$i = 0;
while ($i < len($registry)) {
    $nm  = $registry[$i][0];
    $bad = base.protocol_check($registry[$i][1], $registry[$i][2], $registry[$i][3]);
    if (len($bad) == 0) { ok(true, $nm); }
    else { ok(false, $nm + " violates the protocol: " + str($bad)); }
    $i = $i + 1;
}
eqs(len($registry), 23, "the registry names every estimator K2 added, in every mode");

// ── the check can still fail ────────────────────────────────────────────────
// A conformance suite that cannot fail proves nothing. `bkit_base_test.b` has
// the full negative case; this is the one-line reminder that the mechanism
// being run above is the same one that catches a real violation.
class _NoClone extends Estimator {
    def init() { super(); this.name_ = "NoClone"; this.task_ = "transformer"; }
    def fit($X, $y) { this.fitted_ = true; return this; }
    def transform($X) { return $X; }
}
def NoClone() { return new _NoClone(); }
$neg = base.protocol_check(NoClone(), $Xc, $yc);
ok(len($neg) > 0, "the same check still fails an estimator that forgot clone()");

// ── every transformer composes in a Pipeline ────────────────────────────────
// The protocol's purpose, not a separate feature: an estimator that satisfies
// the contract must drop into a Pipeline without the Pipeline knowing anything
// about it. A three-stage chain is compared against the hand-written
// composition, EXACTLY -- the same kernels in the same order, so there is no
// tolerance to hide behind.
print("");
print("-- the point of the contract: they compose --");
$pipe = base.Pipeline([["imp",  pre.SimpleImputer(null, null)],
                       ["sc",   pre.StandardScaler(null, null)],
                       ["bin",  pre.Binarizer(0.0)]]);
$Zp = $pipe.fit_transform($Xc, null);
$h1 = pre.SimpleImputer(null, null);
$h2 = pre.StandardScaler(null, null);
$h3 = pre.Binarizer(0.0);
$Zh = $h3.fit_transform($h2.fit_transform($h1.fit_transform($Xc, null), null), null);
ok(nd_array_equal($Zp, $Zh), "a 3-stage pipeline equals the hand-written composition, bit for bit");

// and "step__param" reaches into a step, which is what makes a grid search
// over a pipeline possible at all
$pipe.set_params({"bin__threshold": 10.0});
eqs($pipe.get_step("bin").get_params().threshold, 10.0,
    "step__param addresses a step's parameter");
ok(!$pipe.fitted(), "and invalidates the pipeline's fit");
$Z2 = $pipe.fit_transform($Xc, null);
eqs(nd_get(nd_sum($Z2, null, null), 0), 0.0,
    "with the threshold at 10 nothing survives binarisation, so the new value took effect");

// a pipeline of encoders, where the widths change between stages
$pipe2 = base.Pipeline([["ord", pre.OrdinalEncoder(null, null)],
                        ["oh",  pre.OneHotEncoder(null)]]);
$Zo = $pipe2.fit_transform($Xk, null);
eqs(nd_shape($Zo)[0], 20, "an encoder pipeline keeps the row count");
ok(nd_shape($Zo)[1] >= 3, "and widens the matrix, with each stage re-fitted on the last's output");

// a transformer-only pipeline has a transformer's task, so score() refuses
// rather than inventing a metric
eqs($pipe2.task(), "transformer", "a pipeline's task is its final step's");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
