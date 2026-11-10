# bkit — Changelog

Every entry is prefixed so the log can be grepped: **[design]**, **[feature]**, **[bug fix]**,
**[perf]**, **[test]**, **[docs]**.

Phases do not advance until all six gate tiers are green (see [`ROADMAP.md`](ROADMAP.md)).
Measured numbers are recorded with the machine they were taken on, because a number without a
machine is not a measurement.

---

## Phase K0 — Tracking docs

- **[design]** `docs/bkit-architecture.md` — the module layout, the estimator protocol, the
  gradient-boosting design, the `tree_` native family, and every rejected alternative with its
  reason.
- **[design]** `bkit-suite/ROADMAP.md` (K0–K12) and `DECISIONS.md` (K1–K10).

### Found during design, before any code

- **The binned feature matrix has no dtype to live in.** A GBDT wants one byte per cell; numba's
  `DType` is `{F64, I64, BOOL, F32}` and `BOOL` is the only one-byte variant. Holding bins as
  `I64` costs **800 MB for 1M × 100**, which defeats the entire purpose of binning. Resolved as an
  opaque native handle owning a `uint8_t` buffer, the same shape as the existing `col_*` and `bp_*`
  handles (K1 in DECISIONS). Adding `U8` to numba was rejected: it would need the full ufunc,
  reduction and linalg surface built for a dtype nobody does arithmetic in.
- **`super.method()` does not work in Bantu.** It lexes, it parses, and then it fails with
  `Undefined variable: super`; only `super($args)` for the constructor works. A protocol where
  `StandardScaler.fit` called `super.fit()` would not have run at all, so everything shared in
  `base.b` is written in terms of `this` (K3). Finding this mid-implementation would have cost a
  redesign.

---

## Phase K1 — `base.b`, the estimator protocol

- **[feature]** `Estimator`: `fit` / `predict` / `transform` / `clone` raise **by name** when a
  subclass has not implemented them, and `get_params` / `set_params` / `fit_transform` / `score` /
  `_check_fitted` are provided in terms of `this`. An unimplemented method that returned null
  instead would surface later as "cannot index null" somewhere unrelated.
- **[feature]** `set_params` **invalidates the fit**. Leaving `fitted_` true would let `predict`
  run with coefficients belonging to the old parameters — a wrong answer rather than an error.
- **[feature]** `score` implements the scikit-learn defaults (accuracy for a classifier, R² for a
  regressor) with `nd_*` calls rather than by importing `metrics.b`, because this file must stay
  dependency-free (K7). **R² on a constant target raises** instead of returning `NaN` from 0/0.
- **[feature]** `Pipeline`, with `step__param` addressing in `get_params`/`set_params`. That is not
  decoration: it is what makes `GridSearchCV` over a pipeline possible, and a grid search that
  silently ignored the parameters it was handed would report a best score for a configuration it
  never tried. A key naming a step that does not exist raises.
- **[feature]** `Pipeline.clone()` is **deep** — each step is cloned, so a cloned pipeline shares no
  estimator with its source. A shallow copy of the step list would share the estimators themselves
  and quietly defeat K2.
- **[feature]** `protocol_check($est, $X, $y)` — tier 3's mechanism, in `base.b` rather than in the
  test file so every later phase inherits one definition of the contract instead of writing its own
  idea of it. It returns a list of failures rather than throwing, because a test wants every
  estimator's result and not just the first failure.
- **[test]** `tests/bkit_base_test.b` — 46 assertions. The protocol check runs over every estimator
  in the file, **and over a deliberately broken one** that forgets `clone()`: a check that cannot
  fail proves nothing. Also that a clone of a fitted estimator is unfitted (K2, the one that
  matters), that two estimators' outputs share no memory, that a 3-stage pipeline equals the
  hand-written composition **exactly**, and that 1000 clones leak nothing.
- **[bug fix]** **In the language:** a class from an included file could be inherited from but not
  instantiated. `extends` resolves a parent name against the global class registry; `new X()`
  resolved it against the scope, and a class declaration binds its name in the environment where it
  appears — which for an included file is that file's own. So `class Ridge extends Estimator`
  worked while `new Estimator()` said `Undefined variable: Estimator`. A name written after `new`
  now resolves against the registry too. Gated on the keyword, so a bare `X()` still resolves
  through the scope alone and the factory pattern (`def Linear()` shadowing `class Linear`) is
  unchanged. Regression test: `tests/lang_class_include_test.b`.
