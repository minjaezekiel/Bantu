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

## Phase K2 — `preprocess.b`

Eleven estimators over two base classes, 1,011 lines. Design:
[`../docs/bkit-architecture.md`](../docs/bkit-architecture.md) §3; decisions K11–K17.

- **[feature]** `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler` — four scalers
  sharing **one** `transform`, `inverse_transform`, feature-count guard and `NaN` rule, because
  they differ only in the two statistics they compute (K11). `MinMaxScaler` was rewritten into the
  same centre/scale form rather than kept as its own two-step formula: with
  `scale = (max - min)/(hi - lo)` and `center = min - lo·scale`, `(x - center)/scale` is
  *identically* the two-step expression, so it shares everything. The round-trip gate then tests
  one implementation four times instead of four implementations once.
- **[feature]** `OrdinalEncoder`, `OneHotEncoder`, `LabelEncoder`, with `handle_unknown` on the
  first two. `inverse_transform` on all three.
- **[feature]** `SimpleImputer` with `mean`, `median`, `most_frequent` and `constant`.
- **[feature]** `PolynomialFeatures` with `degree`, `interaction_only`, `include_bias` and
  `feature_names()`.
- **[feature]** `Normalizer` (`l1`/`l2`/`max`, row-wise) and `Binarizer`.
- **[design]** **Statistics are `(1, d)` arrays and `transform` is two kernel calls** over the whole
  matrix, so the only Bantu loops in the file run over *columns* or over output features — never
  over rows. Measured: a 100,000 × 20 `fit_transform` takes **75 ms**; the interpreter costs about
  a microsecond an element, so a per-row loop would be minutes. The test asserts a budget for
  exactly that reason.
- **[design]** **The feature-count guard is load-bearing** (K12). `transform` on the wrong number
  of columns is a user error numba *cannot* catch: an `(n, 1)` input against a stored `(1, d)`
  statistic **broadcasts** and returns a well-formed `(n, d)` matrix of nonsense — no shape error,
  no `NaN`, no warning. It is the one guard in the file with no backstop downstream.
- **[design]** **Encoders map through `nd_searchsorted`, not a dict** (K13): one native O(n log k)
  call for a whole column, and the unseen-category test falls out of the same call rather than
  needing a second pass — clamp the positions, `nd_take` the category each landed on, compare with
  the input, and a mismatch *is* an unseen value. Measured: 50,000 × 4 with 200 categories in
  **44 ms**.
- **[design]** **A zero-variance column gets scale 1** (K15), which is scikit-learn's handling and
  has a property worth stating: the column transforms to exactly `x - centre` = 0 and the inverse
  adds the centre back, so the round-trip stays **exact** on a constant column and the gate needs
  no exemption for it. Dividing by zero would have given `NaN` for every row — silently, because
  `NaN` is a perfectly good `f64`.
- **[design]** **One rule about `NaN`, one exception** (K14): everything here refuses it, naming the
  column; `SimpleImputer` accepts it. The refusal is in `fit`, `transform` **and**
  `inverse_transform`, so there is no third case to remember — a scaler fitted on clean data and
  handed dirty data would otherwise emit a `NaN` column silently. The reason it is a refusal rather
  than scikit-learn's `NaN`-aware statistics is concrete: numba has `nanmean`/`nansum`/`nanmin`/
  `nanmax` and **no** `nanvar`, `nanstd`, `nanmedian` or `nanquantile`, which three of the four
  scalers need. Supporting `NaN` in two of them while a third quietly produced `NaN` would be worse
  than refusing in all four.
- **[design]** **The mode is a run-length on a sorted column** (K16), not a count per unique value:
  boundaries from `nd_nonzero(nd_diff(sorted) != 0)`, lengths from `nd_diff` of those, longest by
  `nd_argmax`. O(n log n) in kernels instead of O(k·n) with a Bantu loop per unique. Sorting also
  solves `NaN` for free — `nd_sort` puts `NaN` last, verified against the kernel rather than
  assumed, so the first `c` sorted elements are exactly the non-`NaN` values and `median` needs no
  `nanmedian` either. Ties go to the smaller value, matching `scipy.stats.mode`, and a test asserts
  it.
- **[design]** An **all-`NaN` column raises** for the three statistical strategies. scikit-learn
  drops such a column with a warning; that silently changes the feature count for everything
  downstream, which is worse than stopping. Under `constant` it is accepted, because the fill value
  does not come from the data.
- **[design]** **`PolynomialFeatures`' column order is part of the contract**, not an
  implementation detail: a model trained on one ordering and scored against another is wrong in a
  way no count of columns detects. The order is scikit-learn's, and the test asserts the *list* of
  names — `["1", "x0", "x1", "x0^2", "x0 x1", "x1^2"]` — not its length.
- **[test]** `tests/bkit_preprocess_test.b` — **174 assertions**: tier 1, tier 2 and tier 4.
  Every differential value is hand-computed or closed-form, never this implementation's own output.
  The gates: the round trip to 1e-12 for all four scalers *and* for a constant column; a
  column-count mismatch refused by all nine fitted transformers; `NaN` refused at fit, transform
  and inverse by all nine refusers, each naming the column; an all-`NaN` column refused by each
  statistical strategy. The polynomial output width is also checked against the **closed form** — `C(d+g, g)` with
  repetition, `sum C(d, r)` with interactions only — over twenty (features, degree) pairs, because
  an off-by-one in the advance rule shows up only at a size nobody writes out by hand. Stress:
  100,000 × 20 fit, 50,000 × 4 encode, 1,000 clones measured for leaks, and a clone of a fitted scaler asserted not to touch its source.
- **[test]** `tests/bkit_protocol_test.b` — **32 assertions**, tier 3 for the package: the contract
  over 23 estimator configurations, plus a three-stage `Pipeline` asserted equal to the
  hand-written composition **bit for bit** and `step__param` reaching into a step. It is its own
  file rather than a section of `bkit_base_test.b` (K17), because the registry must name estimators
  from every module and `base.b`'s test must not include eleven of them — `base.b` is small
  precisely so `btext` can depend on it alone.
- **[test]** **The suite was mutation-tested, not merely run.** Thirteen independent breakages of
  `preprocess.b` were each shown to turn it red: the zero-scale guard (11 assertions), the `NaN`
  refusal (12), the feature-count guard (11), the polynomial bias position (10), `ddof=1` (4), an
  unseen category coded silently (4), `>=` for `>` (2), `MinMaxScaler`'s range offset (2), a
  `clone()` returning `this` (2), mode ties (1), the median's middle index (1), a one-hot inverse
  claiming category 0 (1), and the polynomial advance rule. A suite that has never been shown to
  fail is a suite nobody has tested.

  Worth stating plainly: the **protocol** test stays green under ten of the thirteen, and that is
  correct rather than a gap. Tier 3 asserts the contract — `clone()` isolation, `get_params`
  agreement, raising before `fit` — and a scaler with the wrong arithmetic satisfies the contract
  perfectly. Tier 2 is what catches wrong numbers. The two mutations tier 3 *did* catch are the two
  that break the contract itself: `clone()` returning `this`, and the squared terms vanishing.
- **[bug fix]** **An empty fit was refused by the four scalers only.** The encoders fitted happily
  on zero rows — `nd_unique` of an empty column gives an empty category list — and the failure
  arrived on the first *real* `transform` as `nd_take: index -1 is out of range for axis 0 with 0
  elements`: a kernel message about an index, for a mistake made in an earlier call. Now one rule
  in one place, refused by all eleven and naming the estimator. Transforming zero rows stays
  allowed, because zero rows in and zero rows out is the right answer.
- **[bug fix]** **In the test harness, found by that mutation run:** an assertion whose own
  expression throws took the whole process down and hid the forty assertions after it, so one
  mutation reported one failure. Throwing assertions now report a failure and continue.
- **[bug fix]** **In the enumerator, found by that mutation run and a real hazard:** one mutant of
  `PolynomialFeatures`' advance rule made the combination loop **hang** — four minutes of silence,
  no output, no error. The advance rule and the index cap have to agree for the loop to terminate,
  and they are written in two places. The emit loop is now bounded by `C(n+r-1, r)` (or `C(n, r)`
  for `interaction_only`), computed multiplicatively so the intermediates stay exact, and
  overrunning it raises with the counts named. The same mutant now fails in milliseconds with
  *"the advance rule and the index cap have gone out of agreement"*. A hang is a worse failure than
  a wrong answer, and the bound is also a statement of what the enumeration is supposed to produce.
- **[bug fix]** **In the language, found while running this phase:** `bantu run --quiet` still
  printed a `[class] X extends Y` line, to **stdout**, for every class in every included file —
  thirteen lines for a script that includes `bkit`. `--quiet` exists to suppress exactly these
  informational lines, and `[class] Defined:` already honoured it. A program's library internals
  are not its caller's data, and for a `sua` handler or anything emitting JSON they were landing in
  the output stream. Now gated like its neighbour. The companion `WARNING: parent class not found`
  is **not** silenced — it goes to stderr instead, and says what the consequence is, because
  execution continues with no parent and the first symptom is "X.fit is not implemented" from a
  class that plainly implements `fit`.
