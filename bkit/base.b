// ════════════════════════════════════════════════════════════════════════════
//  base.b — the estimator protocol, and nothing else.
//
//      include "bkit/base.b" as base;
//
//      class Ridge extends Estimator {
//          def init($alpha) {
//              super();
//              this.name_   = "Ridge";
//              this.task_   = "regressor";
//              this.params_ = {"alpha": $alpha};
//          }
//          def fit($X, $y)  { ...; this.fitted_ = true; return this; }
//          def predict($X)  { this._check_fitted("Ridge.predict"); ... }
//          def clone()      { return Ridge(this.params_.alpha); }
//      }
//
//  THIS FILE IS DELIBERATELY SMALL (DECISIONS K7). `btext` depends on it to get
//  the protocol without pulling in gradient boosting, SVMs and
//  cross-validation. A protocol defined inside a large module becomes a reason
//  not to conform to it, and the entire value of the contract is that every
//  implementer agrees.
//
//  TWO BANTU FACTS SHAPED THIS, and finding either one mid-implementation would
//  have cost a redesign (K3):
//
//    * `super.method()` DOES NOT WORK. It lexes, it parses, and then it fails
//      with `Undefined variable: super`; only `super($args)` for the
//      constructor works. So everything shared here is written in terms of
//      `this` -- `fit_transform` calls `this.fit` and `this.transform` -- and
//      a child never calls up. A design where `StandardScaler.fit` called
//      `super.fit()` would not have run at all.
//    * There is NO REFLECTIVE CONSTRUCTOR. A base-class `clone()` cannot
//      construct a fresh instance of the subclass, so every estimator writes
//      its own three-line `clone()`. That is boilerplate, and it is the kind
//      that cannot silently share fitted state.
//
//  WHY `clone()` IS NOT OPTIONAL (K2). Cross-validation refits on every fold.
//  Reusing a fitted estimator across folds leaks the test fold into training
//  and still produces a plausible, merely inflated score -- a wrong answer that
//  looks like a right one, which is the only kind worth building machinery
//  against.
// ════════════════════════════════════════════════════════════════════════════

include "../numba/numba.b" as np;

// A numba array IS the native handle -- `np.array(...)` returns what `nd(...)`
// returns, and every `nd_*` kernel takes it directly. So there is no wrapper to
// unwrap, only a coercion for callers who pass a plain Bantu list. `nd()`
// REFUSES an array it is given back ("expected a list or a number"), hence the
// type guard rather than an unconditional call.
def _asarray($x) {
    if ($x == null) { throw "bkit: expected an array or a list, got null"; }
    if (type($x) == "ndarray") { return $x; }
    if (type($x) == "list") { return nd($x, "f64"); }
    if (type($x) == "number") { return nd([$x], "f64"); }
    throw "bkit: expected an array or a list, got " + type($x);
}

// ── the contract ────────────────────────────────────────────────────────────
// Kept bare (not `_Estimator` + factory) because users EXTEND it and never call
// it, exactly like btorch's `Module`. The concrete estimators in every other
// file are `_Name` with a same-named factory, because a `def Name()` shadows a
// `class Name` and `new Name(...)` inside the factory then re-enters the
// factory and recurses until the interpreter's call-depth guard fires.
class Estimator {
    def init() {
        this.fitted_ = false;
        this.params_ = {};
        this.name_   = "Estimator";
        // "classifier", "regressor", "transformer" or "clusterer". `score()`
        // dispatches on it, and the protocol test asserts it is one of those.
        this.task_   = "estimator";
    }

    // ── what a subclass must provide ────────────────────────────────────────
    // Each raises BY NAME. An unimplemented method that returned null instead
    // would surface later as "cannot index null" somewhere unrelated.
    def fit($X, $y)    { throw this.name_ + ".fit is not implemented"; }
    def predict($X)    { throw this.name_ + ".predict is not implemented"; }
    def transform($X)  { throw this.name_ + ".transform is not implemented"; }
    def clone()        { throw this.name_ + ".clone is not implemented -- every "
                               + "estimator writes its own (DECISIONS K3)"; }

    // ── provided, in terms of `this` ────────────────────────────────────────
    def get_params() { return this.params_; }

    // Setting a parameter invalidates the fit. Leaving `fitted_` true would let
    // `predict` run with coefficients belonging to the OLD parameters, which is
    // a wrong answer rather than an error.
    //
    // AN UNKNOWN KEY IS REFUSED, and that is not pedantry (K24). It used to be
    // accepted, which meant `set_params({"with_men": false})` added a
    // parameter nobody reads: a grid search over a mistyped key then ran every
    // candidate with identical settings and reported a best score for a
    // configuration it never tried. A wrong number that looks plausible is the
    // failure mode this whole protocol exists to prevent.
    //
    // Presence is tested with `keys()` and not with `== null`, because a
    // parameter may legitimately HOLD null -- OrdinalEncoder's
    // `unknown_value` does -- so a null read cannot distinguish "absent" from
    // "set to null".
    def set_params($d) {
        if ($d == null) { return this; }
        $known = this.params_.keys();
        for $k, $v in $d.items() {
            if (!contains($known, $k)) {
                throw this.name_ + ".set_params: there is no parameter '" + str($k)
                      + "'. It takes " + str($known) + ".";
            }
            this.params_[$k] = $v;
        }
        this.fitted_ = false;
        return this;
    }

    def fit_transform($X, $y) {
        this.fit($X, $y);
        return this.transform($X);
    }

    // The scikit-learn default: accuracy for a classifier, R² for a regressor.
    // Implemented here with nd_* calls rather than by importing metrics.b,
    // because this file must stay dependency-free (K7). metrics.b is the
    // general surface; this is the one number `GridSearchCV` needs.
    def score($X, $y) {
        this._check_fitted(this.name_ + ".score");
        $pred = _asarray(this.predict($X));
        $yv   = _asarray($y);
        if (this.task_ == "classifier") {
            return nd_get(nd_mean(nd_astype(nd_equal($pred, $yv, null),
                                            "f64"), null, null), 0);
        }
        if (this.task_ == "regressor") {
            // 1 - SSE/SST. A constant target makes SST zero, and 0/0 would be
            // NaN silently; a model that cannot be scored says so.
            $resid = nd_subtract($yv, $pred, null);
            $sse   = nd_get(nd_sum(nd_multiply($resid, $resid, null), null, null, null), 0);
            $mean  = nd_get(nd_mean($yv, null, null), 0);
            $dev   = nd_subtract($yv, $mean, null);
            $sst   = nd_get(nd_sum(nd_multiply($dev, $dev, null), null, null, null), 0);
            if ($sst == 0.0) {
                throw this.name_ + ".score: R² is undefined for a constant target "
                      + "(the total sum of squares is zero) -- score it with an "
                      + "explicit metric instead";
            }
            return 1.0 - ($sse / $sst);
        }
        throw this.name_ + ".score is not defined for task '" + this.task_ + "'";
    }

    def _check_fitted($who) {
        if (!this.fitted_) {
            throw $who + ": call fit first";
        }
        return null;
    }

    def name()   { return this.name_; }
    def task()   { return this.task_; }
    def fitted() { return this.fitted_; }
}

// ── Pipeline ────────────────────────────────────────────────────────────────
// A list of [name, estimator] pairs. Every step but the last transforms; the
// last one does whatever it does.
//
// `set_params` understands "step__param", which is not decoration: it is what
// makes `GridSearchCV` over a pipeline possible at all, and a grid search that
// silently ignored the parameters it was given would report a best score for a
// configuration it never tried.
class _Pipeline extends Estimator {
    def init($steps) {
        super();
        this.name_ = "Pipeline";
        this.steps = $steps;
        if ($steps == null) { this.steps = []; }
        $i = 0;
        while ($i < len(this.steps)) {
            if (len(this.steps[$i]) != 2) {
                throw "Pipeline: step " + str($i) + " must be [name, estimator]";
            }
            $i = $i + 1;
        }
        // The pipeline's task is its final step's, so `score()` does the right
        // thing for a pipeline ending in a classifier.
        if (len(this.steps) > 0) {
            this.task_ = this.steps[len(this.steps) - 1][1].task();
        }
        this.params_ = {"n_steps": len(this.steps)};
    }

    def _last() { return this.steps[len(this.steps) - 1][1]; }

    def fit($X, $y) {
        if (len(this.steps) == 0) { throw "Pipeline.fit: no steps"; }
        $cur = $X;
        $i = 0;
        while ($i < len(this.steps) - 1) {
            $cur = this.steps[$i][1].fit_transform($cur, $y);
            $i = $i + 1;
        }
        this._last().fit($cur, $y);
        this.fitted_ = true;
        return this;
    }

    // Every step transforms, including the last. Only valid when the final
    // step is a transformer, and it says so rather than failing inside it.
    def transform($X) {
        this._check_fitted("Pipeline.transform");
        $cur = $X;
        $i = 0;
        while ($i < len(this.steps)) {
            $cur = this.steps[$i][1].transform($cur);
            $i = $i + 1;
        }
        return $cur;
    }

    def predict($X) {
        this._check_fitted("Pipeline.predict");
        $cur = $X;
        $i = 0;
        while ($i < len(this.steps) - 1) {
            $cur = this.steps[$i][1].transform($cur);
            $i = $i + 1;
        }
        return this._last().predict($cur);
    }

    // Deep: a cloned pipeline shares no estimator with its source, so fitting
    // the clone cannot touch the original's state. A shallow copy of the step
    // list would share the estimators themselves and quietly defeat K2.
    def clone() {
        $out = [];
        $i = 0;
        while ($i < len(this.steps)) {
            append($out, [this.steps[$i][0], this.steps[$i][1].clone()]);
            $i = $i + 1;
        }
        return Pipeline($out);
    }

    def _step_names() {
        $out = [];
        $i = 0;
        while ($i < len(this.steps)) { append($out, this.steps[$i][0]); $i = $i + 1; }
        return $out;
    }

    def get_step($name) {
        $i = 0;
        while ($i < len(this.steps)) {
            if (this.steps[$i][0] == $name) { return this.steps[$i][1]; }
            $i = $i + 1;
        }
        throw "Pipeline: no step named '" + $name + "'";
    }

    def get_params() {
        $out = {"n_steps": len(this.steps)};
        $i = 0;
        while ($i < len(this.steps)) {
            $nm = this.steps[$i][0];
            for $k, $v in this.steps[$i][1].get_params().items() {
                $out[$nm + "__" + $k] = $v;
            }
            $i = $i + 1;
        }
        return $out;
    }

    // "step__param" addresses a step's parameter. A key naming a step that
    // does not exist is an error: a grid search quietly ignoring a key would
    // report a best score for a configuration it never tried.
    def set_params($d) {
        if ($d == null) { return this; }
        for $k, $v in $d.items() {
            $cut = indexOf($k, "__");
            if ($cut < 0) {
                // Same refusal as the base class, for the same reason (K24).
                if (!contains(this.params_.keys(), $k)) {
                    throw "Pipeline.set_params: there is no pipeline parameter '"
                          + str($k) + "'. A step's parameter is addressed as "
                          + "\"<step>__<param>\"; this pipeline's steps are "
                          + str(this._step_names()) + ".";
                }
                this.params_[$k] = $v;
            }
            if ($cut >= 0) {
                $stepName = substr($k, 0, $cut);
                $param    = substr($k, $cut + 2, len($k) - $cut - 2);
                $target   = this.get_step($stepName);
                $one      = {};
                $one[$param] = $v;
                $target.set_params($one);
            }
        }
        this.fitted_ = false;
        return this;
    }
}

def Pipeline($steps) { return new _Pipeline($steps); }

// ── the protocol check ──────────────────────────────────────────────────────
// Tier 3 of the gate (K9). A protocol with eleven implementers fragments
// silently: an estimator that forgets `clone()`, or whose `get_params()` omits
// a parameter `set_params()` accepts, breaks `GridSearchCV` in a way that
// surfaces as a confusing score rather than an error.
//
// It lives here rather than in the test file so every later phase inherits one
// definition of the contract instead of writing its own idea of it.
//
// Returns a list of failure strings -- empty means conforming. It does NOT
// throw, because a test wants to report every estimator's result, not stop at
// the first.
def protocol_check($est, $X, $y) {
    $bad = [];
    if ($est == null) { append($bad, "the estimator is null"); return $bad; }

    // identity
    if (type($est.name()) != "string" || $est.name() == "") {
        append($bad, "name() must be a non-empty string");
    }
    $task = $est.task();
    $known = ["classifier", "regressor", "transformer", "clusterer"];
    if (!contains($known, $task)) {
        append($bad, "task() is '" + str($task) +
                            "', not one of classifier/regressor/transformer/clusterer");
    }

    // a fresh estimator is not fitted, and says so by name rather than
    // predicting from uninitialised state
    if ($est.fitted()) { append($bad, "a fresh estimator reports fitted()"); }
    $raised = false;
    try { $est.predict($X); } catch ($e) { $raised = true; }
    if (!$raised && $task != "transformer") {
        append($bad, "predict() before fit did not raise");
    }

    // get_params / set_params agree: every key set_params accepts must come
    // back from get_params, or a grid search cannot address it
    $p = $est.get_params();
    if (type($p) != "dict") { append($bad, "get_params() must return a dict"); }

    // clone() is implemented, is unfitted, and is a different object
    $cl = null;
    $cloneRaised = false;
    try { $cl = $est.clone(); } catch ($e) { $cloneRaised = true; }
    if ($cloneRaised) { append($bad, "clone() is not implemented"); }
    if (!$cloneRaised) {
        if ($cl.fitted()) { append($bad, "clone() returned a FITTED estimator"); }
        if ($cl.name() != $est.name()) {
            append($bad, "clone() returned a " + $cl.name() + ", not a " + $est.name());
        }
        if (len($cl.get_params()) != len($p)) {
            append($bad, "clone() has " + str(len($cl.get_params())) +
                                " parameters, the source has " + str(len($p)));
        }
    }

    // fit returns `this`, so it chains
    $fitted = $est.fit($X, $y);
    if ($fitted == null) { append($bad, "fit() returned null, not this"); }
    if (!$est.fitted()) { append($bad, "fit() did not set fitted()"); }

    // THE ONE THAT MATTERS (K2): fitting a clone must not touch the source.
    if (!$cloneRaised) {
        $cl2 = $est.clone();
        if ($cl2.fitted()) {
            append($bad, "clone() of a FITTED estimator is itself fitted -- "
                                + "cross-validation would leak the test fold");
        }
    }

    // set_params invalidates the fit
    if (len($p) > 0) {
        for $k, $v in $p.items() {
            $one = {};
            $one[$k] = $v;
            $est.set_params($one);
            if ($est.fitted()) {
                append($bad, "set_params() left the estimator fitted");
            }
            break;
        }
    }
    return $bad;
}
