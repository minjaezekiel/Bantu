# bkit — Roadmap

Classical machine learning for Bantu: scikit-learn's shape with an
xgboost/lightgbm-class gradient-boosting implementation inside it. The public
API is pure Bantu; the atoms are numba's `nd_*` plus a new `tree_` family.

Design: [`../docs/bkit-architecture.md`](../docs/bkit-architecture.md).
Decisions: [`DECISIONS.md`](DECISIONS.md). Ecosystem context and ordering:
[`../docs/ecosystem-architecture.md`](../docs/ecosystem-architecture.md)
(stage 3).

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done (feature **and**
differential **and** stress green)

**Every phase is gated by six tiers, all green before it advances:**

| tier | proves | form |
|---|---|---|
| 1. feature | every estimator does what it claims, including its failure modes | `tests/bkit_*_test.b`, ending `RESULT: ALL GREEN` |
| 2. **differential** | the answers are *right* — scikit-learn / LightGBM reference values embedded in the test | hand-computed small cases; closed forms where they exist |
| 3. **protocol** | every estimator satisfies the [C1] contract, including `clone()` isolation | one table-driven test over every estimator in the package |
| 4. stress | survives size, repetition, abuse | 1M×100 fit, 10,000 trees, degenerate and all-`NaN` columns |
| 5. regression | nothing else broke | the whole existing suite on the same build |
| 6. sanitizers | no latent memory defect | ASan + UBSan over the full bkit suite |

Tier 3 is new to this project. A protocol with eleven implementers fragments
silently, so it is enumerated and asserted rather than trusted.

**Three rules carried from btorch, and they apply here too.**
1. Never assert on a stringified number — `str()` gives six significant digits.
   Use `nd_allclose` or a tolerance.
2. Never use the C-style `for` above 100,000 iterations — `evaluator.hpp` caps
   it at `safety < 100000` and exits **silently**. Use `while`.
3. **A falling training error proves nothing.** An overfit tree reaches zero
   training error with a wrong split rule. Correctness comes from tier 2.

**Prerequisite:** ecosystem stage 2 (the `bantu` engine field) must land first.
`bkit` ships the first new native family since that gap was found, and it is
the first package that would hit the trap in the wild.

---

## Phase K0 — Tracking docs
- [x] `docs/bkit-architecture.md`
- [x] `bkit-suite/ROADMAP.md`, `DECISIONS.md`
- [x] `bkit-suite/CHANGELOG.md`

## Phase K1 — `base.b`, the estimator protocol

Deliberately tiny and deliberately first: `btext` depends on this file alone
([E12]), and every later phase implements it.

| Item | How | Feature test | Stress test | Status |
|---|---|---|---|---|
| `Estimator` base | `fit`/`predict`/`get_params`/`set_params`/`clone`, plus `fit_transform`/`score`/`_check_fitted` provided in terms of `this` | each method present; calling an unimplemented one raises **by name** | — | [x] |
| `clone()` per estimator | no reflective constructor in Bantu, so each writes its own ([K3]) | a cloned estimator is unfitted and shares no array with its source | 1,000 clones leak 0 bytes | [x] |
| `Pipeline` | a list of `(name, estimator)`; `fit` chains `fit_transform`, `predict` chains `transform` | a 3-stage pipeline equals the hand-written composition exactly | 50-stage pipeline | [x] feature + exactness; the 50-stage stress belongs with K6's `ColumnTransformer` |
| **protocol test harness** | table-driven over a registry of every estimator | tier 3's mechanism, written here so later phases inherit it | — | [x] in `base.b`, so every phase inherits one definition of the contract |

**K1 found a language bug, fixed in the interpreter.** A class from an included
file could be **inherited from but not instantiated**: `extends` resolves a
parent against the global class registry, while `new X()` resolved it against
the scope — and a class declaration binds its name where it appears, which for
an included file is that file's own environment. So `class Ridge extends
Estimator` worked and `new Estimator()` said `Undefined variable`. A name after
`new` now resolves against the registry too, gated on the keyword so the
factory pattern is untouched. Regression: `tests/lang_class_include_test.b`.

## Phase K2 — `preprocess.b`

| Item | Feature test | Status |
|---|---|---|
| `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler` | `inverse_transform(transform(x)) == x` to 1e-12; zero-variance column does not divide by zero | [ ] |
| `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder` | unseen category at transform time raises by name, or maps to a documented slot under `handle_unknown` | [ ] |
| `SimpleImputer` (mean/median/most_frequent/constant) | an all-`NaN` column raises rather than producing `NaN` statistics | [ ] |
| `PolynomialFeatures` | column count and ordering match a reference enumeration | [ ] |
| `Normalizer`, `Binarizer` | — | [ ] |

## Phase K3 — `selection.b`

| Item | Feature test | Status |
|---|---|---|
| `train_test_split` (stratified option) | **seeded via `nd_seed` only** ([C4]); same seed → identical split, twice | [ ] |
| `KFold`, `StratifiedKFold`, `GroupKFold`, `TimeSeriesSplit` | folds partition exactly: every index once, no overlap; stratified preserves class ratios to ±1 | [ ] |
| `cross_val_score`, `cross_validate` | **clones before every fit**; a test asserts the source estimator's coefficients are unchanged afterwards ([K2]) | [ ] |
| `GridSearchCV`, `RandomizedSearchCV` | the reported best score equals a hand-run refit at those params | [ ] |

## Phase K4 — `metrics.b`

| Item | Feature test | Status |
|---|---|---|
| classification: accuracy, precision, recall, F1 (micro/macro/weighted), confusion matrix, `classification_report` | against hand-computed 3-class cases; zero-support class does not divide by zero | [ ] |
| `roc_auc_score`, `roc_curve`, `precision_recall_curve`, `average_precision` | ties in score handled by the trapezoid rule, asserted against a hand case | [ ] |
| `log_loss`, `brier_score` | clipped at `eps`, so a confident wrong prediction is finite, not `inf` | [ ] |
| regression: MSE, RMSE, MAE, R², MAPE, explained variance | R² on a constant target is defined (or raises) rather than `0/0` | [ ] |
| ranking: NDCG, MRR | — | [ ] |
| **move `ds.accuracy` here**, alias left in btorch ([C5], [E14]) | the btorch alias still works and warns once | [ ] |

## Phase K5 — `linear.b`

| Item | How | Status |
|---|---|---|
| `LinearRegression` | normal equations via numba's `lstsq`; `solve` for well-conditioned | [ ] |
| `Ridge` | closed form with the penalty on the Gram matrix | [ ] |
| `Lasso`, `ElasticNet` | coordinate descent with a convergence tolerance **and an iteration cap that warns when hit** | [ ] |
| `LogisticRegression` | IRLS or L-BFGS; multinomial via softmax | [ ] |
| `SGDClassifier`/`SGDRegressor` | one pass per epoch, `nd_*`-vectorised | [ ] |

## Phase K6 — `Pipeline` + `ColumnTransformer`
`ColumnTransformer` is the only part of `bkit` that touches `arctic`
([§4 of the design doc]).

## Phase K7 — `tree.b` and the `tree_` native family

The first native work in this package. Five kernels; the binned matrix is an
**opaque handle**, not an `NdArray` ([K1] in DECISIONS).

| Item | How | Feature test | Stress test | Status |
|---|---|---|---|---|
| `tree_bin` | quantile binning to ≤256 bins; `NaN` to a dedicated bin | bin edges match a reference quantile computation; constant column → 1 bin | 1M×100 binned, memory **= n×f bytes**, not 8× | [ ] |
| `tree_hist` | the hot loop: byte load + 3 float adds per cell | histogram equals a pure-Bantu reference loop exactly | — | [ ] |
| `tree_best_split` | regularised gain scan with a running prefix | best split matches a brute-force reference over all (feature, bin) | **hessian saturation drives `h→0`; must not produce a `NaN` gain that wins** | [ ] |
| `tree_partition` | stable index partition | left ∪ right = parent, disjoint, order preserved | — | [ ] |
| `tree_predict` | traversal | equals a Bantu reference traversal on every row | 10,000-node tree | [ ] |
| `DecisionTreeClassifier`/`Regressor` | over the above | against scikit-learn reference values | depth-50 tree; single-sample leaves | [ ] |
| **register `tree_` in `kNativeFamilies`** | `types.hpp:499`, so an unbound `tree_*` self-diagnoses ([B21]) | an unbound name names the family | — | [ ] |

## Phase K8 — `ensemble.b` — **the headline**

| Item | How | Status |
|---|---|---|
| `RandomForestClassifier`/`Regressor` | bootstrap + feature subsampling, seeded via `nd_seed` | [ ] |
| `ExtraTrees*` | random split thresholds | [ ] |
| **`GradientBoostingClassifier`/`Regressor`** | histogram-based, leaf-wise with `max_leaves`, **histogram subtraction** for the larger child | [ ] |
| objectives | `squared_error`, `logistic`, `softmax` (one tree per class per round), `poisson`; hessian clamped at 1e-16 | [ ] |
| missing values | `NaN` to its own bin; each split learns a default direction, used for unseen `NaN` at predict time | [ ] |
| early stopping | on a validation fold, with `n_iter_no_change` | [ ] |
| `feature_importances_` | gain, split count and permutation importance | [ ] |

## Phase K9 — `cluster.b`, `decompose.b`
`KMeans` (k-means++ init, seeded), `DBSCAN`, `GaussianMixture`; `PCA`,
`TruncatedSVD` over numba's existing SVD — asserted to agree with it to 1e-10
rather than reimplemented.

## Phase K10 — `neighbors.b`
kNN classifier/regressor, Gaussian/Multinomial/Bernoulli naive Bayes, linear
SVM. Full SMO is an open item ([§8.4 of the design doc]).

## Phase K11 — `persist.b`
`.btsd` with `kind: "bkit.estimator"` ([C3]). **A loaded model predicts
bit-identically to the model that was saved.**

## Phase K12 — The package, docs, samples, gates
`package.json` (with the `bantu` engine floor), `docs/bkit.md`,
`docs/bkit-tutorial.md`, `samples/bkit/`, and the accuracy gates:

| gate | dataset | target |
|---|---|---|
| GBDT binary | adult / census income | **≥ 87% accuracy, ≥ 0.92 ROC-AUC** |
| GBDT regression | california housing | **RMSE ≤ 0.50** |
| Random forest | covertype subset | ≥ 85% accuracy |
| Logistic regression | breast cancer | ≥ 97% accuracy |
| k-means | synthetic, known centroids | within 1e-6 |
| PCA | synthetic | agrees with numba's SVD to 1e-10 |

Datasets fetched and SHA-256 verified, **never committed** — the method
`get_mnist.sh` established ([B25]). Every gate measured from a **reloaded**
`.btsd` model.

---

## Where to resume

**K0 and K1 are done; K2 (`preprocess.b`) is next.** `base.b` is 330 lines and
`tests/bkit_base_test.b` is 46 assertions, all green. Read
[`../docs/bkit-architecture.md`](../docs/bkit-architecture.md) first, then
[`DECISIONS.md`](DECISIONS.md).

Ecosystem **stage 2** (the engine field) landed first, as required, and
`bkit/package.json` declares `"bantu": ">=1.4.0"` — verified against the floor
its own sources imply with `bantu engine bkit`.

What K2 inherits from K1 and should not re-invent:

* every estimator is `_Name` plus a same-named factory `def Name(...)`, because
  a `def Name()` shadows a `class Name`, and `new Name(...)` inside the factory
  then re-enters the factory until the call-depth guard fires;
* a field and a method cannot share a name (one namespace), hence `fitted_`,
  `params_`, `name_`, `task_`;
* `append($list, $v)` **mutates and returns the new LENGTH** — `$l = append($l, $v)`
  replaces the list with a number;
* every new estimator joins the registry in `tests/bkit_base_test.b`'s protocol
  section, which is tier 3 for the whole package.
