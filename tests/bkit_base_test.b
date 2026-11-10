// ════════════════════════════════════════════════════════════════════════
//  bkit_base_test.b — the estimator protocol (bkit phase K1).
//
//  This file is tiers 1 and 3 of bkit's gate at once: it checks that the base
//  class provides what it promises, and it runs the table-driven PROTOCOL
//  check that every later phase's estimators will be added to.
//
//  Tier 3 exists because a protocol with eleven implementers fragments
//  silently (DECISIONS K9). An estimator that forgets `clone()`, or whose
//  `get_params()` omits a parameter that `set_params()` accepts, breaks
//  `GridSearchCV` in a way that surfaces as a confusing SCORE rather than an
//  error -- and a wrong number that looks plausible is the only kind worth
//  building machinery against.
//
//  The two estimators here are deliberate doubles, not real ones: K1 has no
//  real estimators yet, and the point is to prove the base class and the
//  protocol check work before eleven files depend on both.
//
//  Run:  bantu run tests/bkit_base_test.b
// ════════════════════════════════════════════════════════════════════════

include "../numba/numba.b" as np;
include "../bkit/base.b" as base;

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
def close($a, $b, $tol, $n) { ok(abs($a - $b) <= $tol, $n); return null; }

// ── two doubles ─────────────────────────────────────────────────────────────
// A transformer that adds a constant, and a regressor that predicts the
// training mean. Both are trivial on purpose: anything they get wrong is the
// base class's fault, not theirs.
class _AddConst extends Estimator {
    def init($k) {
        super();
        this.name_   = "AddConst";
        this.task_   = "transformer";
        this.params_ = {"k": $k};
        if ($k == null) { this.params_ = {"k": 1.0}; }
    }
    def fit($X, $y) { this.fitted_ = true; return this; }
    def transform($X) {
        this._check_fitted("AddConst.transform");
        return nd_add(base._asarray($X), this.params_.k, null);
    }
    def clone() { return AddConst(this.params_.k); }
}
def AddConst($k) { return new _AddConst($k); }

class _MeanRegressor extends Estimator {
    def init() {
        super();
        this.name_   = "MeanRegressor";
        this.task_   = "regressor";
        this.params_ = {};
        this.mean_   = 0.0;
    }
    def fit($X, $y) {
        this.mean_   = nd_get(nd_mean(base._asarray($y), null, null), 0);
        this.fitted_ = true;
        return this;
    }
    def predict($X) {
        this._check_fitted("MeanRegressor.predict");
        return nd_full([nd_shape(base._asarray($X))[0]], this.mean_, "f64");
    }
    def clone() { return MeanRegressor(); }
}
def MeanRegressor() { return new _MeanRegressor(); }

// A deliberately BROKEN estimator: it forgets clone(). The protocol check must
// catch it, because the whole point of tier 3 is catching this at the point of
// writing rather than inside a grid search.
class _Forgetful extends Estimator {
    def init() {
        super();
        this.name_ = "Forgetful";
        this.task_ = "regressor";
    }
    def fit($X, $y) { this.fitted_ = true; return this; }
    def predict($X) { this._check_fitted("Forgetful.predict"); return nd_zeros([1], "f64"); }
}
def Forgetful() { return new _Forgetful(); }

print("");
print("  bkit estimator protocol (K1)");
print("  ════════════════════════════════════════");

nd_seed(7);
$X = nd_random_uniform([12, 3], -1.0, 1.0);
$y = nd_random_uniform([12], 0.0, 5.0);

// ── 1. what the base class refuses to guess ─────────────────────────────────
print("");
print("-- an unimplemented method raises, by name --");
$bare = new Estimator();
raises(def() { return $bare.fit($X, $y); },    "fit raises when not implemented");
raises(def() { return $bare.predict($X); },    "predict raises when not implemented");
raises(def() { return $bare.transform($X); },  "transform raises when not implemented");
raises(def() { return $bare.clone(); },        "clone raises when not implemented");
$msg = "";
try { $bare.predict($X); } catch ($e) { $msg = str($e); }
ok(indexOf($msg, "Estimator.predict") >= 0,
   "and the message names the estimator and the method");
$msg2 = "";
try { $bare.clone(); } catch ($e) { $msg2 = str($e); }
ok(indexOf($msg2, "K3") >= 0, "clone's message points at the decision that explains it");

// ── 2. _check_fitted ────────────────────────────────────────────────────────
print("");
print("-- predict before fit --");
$mr = MeanRegressor();
ok(!$mr.fitted(), "a fresh estimator is not fitted");
raises(def() { return $mr.predict($X); }, "predict before fit raises");
$m3 = "";
try { $mr.predict($X); } catch ($e) { $m3 = str($e); }
ok(indexOf($m3, "call fit first") >= 0, "and says to call fit first");
$mr.fit($X, $y);
ok($mr.fitted(), "fit sets fitted()");
ok($mr.fit($X, $y) != null, "fit returns this, so it chains");

// ── 3. set_params invalidates the fit ───────────────────────────────────────
// Leaving fitted_ true would let predict run with coefficients belonging to
// the OLD parameters: a wrong answer rather than an error.
print("");
print("-- set_params --");
$ac = AddConst(2.0);
$ac.fit($X, null);
ok($ac.fitted(), "fitted after fit");
$ac.set_params({"k": 5.0});
ok(!$ac.fitted(), "set_params invalidates the fit");
eqs($ac.get_params().k, 5.0, "and the parameter took");
$ac.fit($X, null);
eqs(nd_get($ac.transform(nd_zeros([1], "f64")), 0), 5.0, "the new parameter is used");
eqs($ac.set_params(null), $ac, "set_params(null) is a no-op that still chains");

// AN UNKNOWN KEY IS REFUSED (K24). It used to be accepted and added to
// params_, so a grid search over a mistyped key ran every candidate with
// identical settings and reported a best configuration it never tried -- a
// plausible wrong number, which is the one failure this protocol exists to
// prevent. Found while designing K3; this is its regression test.
$typoMsg = "";
try { $ac.set_params({"kk": 1.0}); } catch ($e) { $typoMsg = str($e); }
ok($typoMsg != "", "set_params refuses a key get_params() does not hold");
ok(indexOf($typoMsg, "no parameter 'kk'") >= 0, "  naming the key it rejected");
ok(indexOf($typoMsg, "[k]") >= 0, "  and listing the ones it does take");
eqs(len($ac.get_params()), 1, "and the refused key was NOT added");
// a mix of one good and one bad key is still refused, so a grid cannot
// half-apply
$mixMsg = "";
try { $ac.set_params({"k": 3.0, "nope": 1.0}); } catch ($e) { $mixMsg = str($e); }
ok($mixMsg != "", "a dict with one unknown key among good ones is refused");

// PRESENCE IS NOT A NULL TEST. A parameter may legitimately HOLD null --
// OrdinalEncoder's unknown_value does -- so `== null` cannot distinguish
// "absent" from "set to null", and using it would make every nullable
// parameter unsettable.
class _NullParam extends Estimator {
    def init() { super(); this.name_ = "NullParam"; this.task_ = "transformer";
                 this.params_ = {"opt": null}; }
    def fit($X, $y) { this.fitted_ = true; return this; }
    def transform($X) { return $X; }
    def clone() { return NullParam(); }
}
def NullParam() { return new _NullParam(); }
$np2 = NullParam();
$nullOk = true;
try { $np2.set_params({"opt": 7.0}); } catch ($e) { $nullOk = false; }
ok($nullOk, "a parameter whose value is null is still settable");
eqs($np2.get_params().opt, 7.0, "  and the new value took");

// ── 4. clone isolation, which is the one that matters (K2) ──────────────────
print("");
print("-- clone --");
$src = AddConst(3.0);
$src.fit($X, null);
$cl = $src.clone();
ok(!$cl.fitted(), "a clone is UNFITTED even when the source is fitted");
eqs($cl.get_params().k, 3.0, "a clone carries the parameters");
$cl.set_params({"k": 99.0});
eqs($src.get_params().k, 3.0, "changing the clone does not touch the source");
$cl2 = $src.clone();
$cl2.fit($X, null);
ok($src.fitted(), "fitting a clone leaves the source fitted as it was");
// and no array is shared, so a transform through one cannot alias the other
$a1 = $src.transform(nd_zeros([4], "f64"));
$a2 = $cl2.transform(nd_zeros([4], "f64"));
ok(!nd_shares_memory($a1, $a2), "two estimators' outputs share no memory");

// ── 5. score: the sklearn defaults, computed without metrics.b ──────────────
print("");
print("-- score --");
$mr2 = MeanRegressor();
$mr2.fit($X, $y);
// predicting the mean gives R² = 0 exactly
close($mr2.score($X, $y), 0.0, 1.0e-12, "a mean predictor scores R² = 0");
// a constant target has no variance, so R² is undefined and says so rather
// than returning NaN from 0/0
$const = nd_full([12], 4.0, "f64");
$mr3 = MeanRegressor();
$mr3.fit($X, $const);
raises(def() { return $mr3.score($X, $const); },
       "R² on a constant target raises instead of returning NaN");
$m4 = "";
try { $mr3.score($X, $const); } catch ($e) { $m4 = str($e); }
ok(indexOf($m4, "constant target") >= 0, "and explains why");
raises(def() { return $ac.score($X, null); },
       "score on a transformer raises -- there is no default metric for one");

// ── 6. Pipeline ─────────────────────────────────────────────────────────────
print("");
print("-- Pipeline --");
$pipe = base.Pipeline([["a", AddConst(1.0)], ["b", AddConst(2.0)], ["m", MeanRegressor()]]);
eqs($pipe.name(), "Pipeline", "a pipeline names itself");
eqs($pipe.task(), "regressor", "and takes its task from the final step");
raises(def() { return $pipe.predict($X); }, "predict before fit raises");
$pipe.fit($X, $y);
ok($pipe.fitted(), "fit fits the pipeline");

// the composition is exactly the hand-written one
$hand = MeanRegressor();
$hand.fit(nd_add(nd_add($X, 1.0, null), 2.0, null), $y);
$pp = $pipe.predict($X);
$hh = $hand.predict($X);
ok(nd_get(nd_all(nd_equal($pp, $hh, null), null, null), 0),
   "a 3-stage pipeline equals the hand-written composition, exactly");

// transform chains every step, including the last
$tpipe = base.Pipeline([["a", AddConst(1.0)], ["b", AddConst(2.0)]]);
$tpipe.fit($X, null);
ok(nd_get(nd_all(nd_equal($tpipe.transform(nd_zeros([3], "f64")),
                          nd_full([3], 3.0, "f64"), null), null, null), 0),
   "transform applies every step");

// get_params addresses steps, and set_params reaches into them
$gp = $pipe.get_params();
ok(indexOf(str(keys($gp)), "a__k") >= 0, "get_params exposes step__param keys");
$pipe.set_params({"a__k": 10.0});
eqs($pipe.get_step("a").get_params().k, 10.0, "set_params reaches into a step");
ok(!$pipe.fitted(), "and invalidates the pipeline's fit");
raises(def() { return $pipe.set_params({"nosuch__k": 1.0}); },
       "a key naming a step that does not exist raises");
raises(def() { return $pipe.get_step("nope"); }, "get_step on an unknown name raises");

// clone is DEEP: a cloned pipeline shares no estimator with its source
$pc = $pipe.clone();
$pc.set_params({"a__k": 77.0});
eqs($pipe.get_step("a").get_params().k, 10.0,
    "a cloned pipeline's steps are clones, not the same objects");
raises(def() { return base.Pipeline([]).fit($X, $y); }, "an empty pipeline refuses to fit");
raises(def() { return base.Pipeline([["bad", AddConst(1.0), "extra"]]); },
       "a malformed step is rejected at construction");

// ── 7. the protocol check itself ────────────────────────────────────────────
// Tier 3's mechanism. Every later phase appends its estimators to this list.
print("");
print("-- the protocol check over every estimator here --");
$registry = [["AddConst",      AddConst(1.0)],
             ["MeanRegressor", MeanRegressor()],
             ["Pipeline",      base.Pipeline([["a", AddConst(1.0)], ["m", MeanRegressor()]])]];
$i = 0;
while ($i < len($registry)) {
    $nm  = $registry[$i][0];
    $bad = base.protocol_check($registry[$i][1], $X, $y);
    if (len($bad) == 0) {
        ok(true, $nm + " satisfies the protocol");
    } else {
        ok(false, $nm + " violates the protocol: " + str($bad));
    }
    $i = $i + 1;
}

// and the check must FAIL a broken estimator, or it proves nothing
$broken = base.protocol_check(Forgetful(), $X, $y);
ok(len($broken) > 0, "the check catches an estimator that forgot clone()");
ok(indexOf(str($broken), "clone") >= 0, "and says which part of the contract broke");
$nullBad = base.protocol_check(null, $X, $y);
ok(len($nullBad) == 1, "the check survives being handed null");

// ── 8. no leak across many clones ───────────────────────────────────────────
print("");
print("-- memory --");
$before = nd_live_bytes();
$j = 0;
while ($j < 1000) {
    $tmp = AddConst(1.0).clone();
    $j = $j + 1;
}
$after = nd_live_bytes();
print("        live bytes " + str($before) + " -> " + str($after) + " over 1000 clones");
ok($after <= $before, "1000 clones do not grow live memory");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
