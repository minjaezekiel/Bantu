# Stage 3 — `bkit`: classical machine learning

A scikit-learn-shaped package with an xgboost/lightgbm-class gradient-boosting
implementation inside it. Phases **K0–K12** in
[`../bkit-suite/ROADMAP.md`](../bkit-suite/ROADMAP.md); decisions in
[`../bkit-suite/DECISIONS.md`](../bkit-suite/DECISIONS.md). Context:
[`ecosystem-architecture.md`](ecosystem-architecture.md).

**Why this package matters more than its position suggests.** Most production
tabular machine learning is gradient-boosted trees, not deep learning, and
`btorch` cannot do it at all. `bkit` is where the commercially valuable work
is, and it is also the package that defines the estimator protocol every later
package conforms to ([C1]).

---

## 1. Module layout

One package, eleven files. The split is by what a user imports together, and
`base.b` is deliberately tiny so `btext` can depend on it without pulling in
GBDT ([E12]).

| file | contains |
|---|---|
| `base.b` | the estimator protocol and nothing else |
| `preprocess.b` | scalers, encoders, imputers, polynomial features |
| `selection.b` | splits, K-fold, cross-validation, grid/randomised search |
| `metrics.b` | classification, regression and ranking metrics |
| `linear.b` | linear/ridge/lasso/elastic-net, logistic regression |
| `tree.b` | decision trees over the `tree_` family |
| `ensemble.b` | random forest, extra trees, **gradient boosting** |
| `cluster.b` | k-means, DBSCAN, Gaussian mixture |
| `decompose.b` | PCA, truncated SVD |
| `neighbors.b` | kNN classifier/regressor; naive Bayes; SVM |
| `persist.b` | `.btsd` save/load with `kind: "bkit.estimator"` |

---

## 2. The estimator protocol

Defined in `base.b`, implemented by everything in `bkit`, and by `btext`'s
vectorizers so they drop into a `Pipeline`.

```bantu
class Estimator {
    def init()                    { this.fitted_ = false; this.params_ = {}; }
    def fit($X, $y)               { throw "bkit: fit not implemented"; }
    def predict($X)               { throw "bkit: predict not implemented"; }
    def get_params()              { return this.params_; }
    def set_params($d)            { /* merge, invalidate fitted_ */ return this; }
    def clone()                   { throw "bkit: clone not implemented"; }

    // provided in terms of `this`, so subclasses inherit them for free
    def fit_transform($X, $y)     { this.fit($X, $y); return this.transform($X); }
    def score($X, $y)             { /* per task type */ }
    def _check_fitted($who)       { if (!this.fitted_) { throw $who + ": call fit first"; } }
}
```

### 2.1 Two Bantu constraints that shaped this

**`super.method()` does not work** — it lexes and parses, then fails with
`Undefined variable: super` (btorch A4, still deferred). Only `super($args)`
for the constructor works. So the base class shares code through `this`, never
through children calling up. `fit_transform` above is written in terms of
`this.fit`/`this.transform` and inherits cleanly; a design where
`StandardScaler.fit` called `super.fit()` would not have worked at all.

**`clone()` must be implemented per estimator.** A base-class `clone` would
need to construct a fresh instance of the *subclass*, and Bantu has no
reflective constructor. So each estimator writes
`def clone() { return new Ridge(this.params_.alpha, ...); }`. Three lines of
boilerplate, and it cannot silently share fitted state.

### 2.2 Why `clone()` is not optional

Cross-validation must refit from scratch on every fold. Reusing a fitted
estimator across folds leaks the test fold into training and still produces a
plausible — just inflated — score. That is the same class of failure as btorch's
B10 ("a falling loss is not evidence"): a wrong answer that looks right.

So `cross_val_score` clones before every fit, and a test asserts that
**a fitted estimator's coefficients are unchanged after being passed through
cross-validation**, which is the observable signature of the bug.

---

## 3. Gradient boosting — the headline

Histogram-based, LightGBM-shaped. The design choices and why.

### 3.1 The algorithm

1. **Bin once, up front.** Each feature is quantile-binned into at most 256
   bins. All later work reads bin indices, never raw values. This is what makes
   split finding O(bins) instead of O(sorted samples).
2. **Per node, build histograms** of `(Σgradient, Σhessian, count)` per bin per
   feature over that node's samples.
3. **Scan bins for the best split**, gain being the standard regularised form
   `½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)] − γ`.
4. **Histogram subtraction:** a child's histogram is its parent's minus its
   sibling's. Compute the smaller child, subtract for the larger. This roughly
   halves the work per level and is the single largest optimisation in the
   algorithm.
5. **Leaf-wise growth** with a `max_leaves` cap (LightGBM's choice), plus
   `max_depth` as a separate guard.
6. **Predict** by traversing the node array.

### 3.2 The `tree_` native family

Five kernels. The Bantu side runs the boosting loop, computes gradients and
hessians (vectorised over `nd_*`), handles early stopping and feature
importance — all of which are one call per *iteration*, not per element.

| kernel | signature | why native |
|---|---|---|
| `tree_bin` | `(X, max_bins) -> handle, edges` | quantile binning over `n×f` values |
| `tree_hist` | `(binned, grad, hess, idx, nbins) -> hist` | **the hot loop**: one byte load and three float adds per (sample, feature) per node |
| `tree_best_split` | `(hist, λ, γ, min_child_weight, min_data) -> (feat, bin, gain)` | O(features × bins) scan with a running prefix |
| `tree_partition` | `(binned, idx, feat, bin) -> (left, right)` | stable in-place partition of an index array |
| `tree_predict` | `(nodes, X, edges) -> y` | per-sample tree traversal, branch-heavy |

A Bantu function call costs **1.78 µs** (btorch B1). `tree_hist` on a
1M × 100 dataset touches 10⁸ cells per level — four orders of magnitude outside
what Bantu can do per element. These five are not an optimisation; without them
there is no GBDT.

### 3.3 The binned matrix is an opaque handle, not an `NdArray`

**A real blocker, found while writing this.** The binned matrix wants to be
`u8` — one byte per cell. numba's `DType` is `{F64, I64, BOOL}`: there is no
`u8`, and `BOOL` is the only one-byte variant. Holding bins as `I64` costs
**8 bytes a cell — 800 MB for 1M × 100**, which defeats the entire point of
binning.

Three options:

| option | verdict |
|---|---|
| add `U8` to numba in stage 1 | **rejected** — stage 1 is already scoped (f32, threading, GEMM); widening it to serve a package three stages later is scope creep, and `U8` would need the full ufunc/reduce/linalg surface for a dtype nobody does arithmetic in |
| hold bins as `I64` | **rejected** — 8× the memory, on the one data structure that must stay small |
| **an opaque native handle owning a `uint8_t` buffer** | **chosen** |

The binned matrix is an internal structure: users never index it, slice it or
do arithmetic on it. It needs exactly `get(row, feature)` from four kernels.
An opaque handle — the same shape as the existing `col_*` and `bp_*` handles —
gives one byte per cell with no dtype work at all, and decouples `bkit`
entirely from the dtype roadmap.

**Forward path:** if stage 6's `q8_` work adds a real `I8`/`U8` dtype for
quantization, the handle can migrate to an `NdArray` behind the same API. It is
not a dead end, just not a dependency.

### 3.4 Objectives

Gradients and hessians are pure Bantu over `nd_*` — one vectorised call per
boosting iteration.

| objective | gradient | hessian |
|---|---|---|
| `squared_error` | `p − y` | `1` |
| `logistic` | `σ(p) − y` | `σ(p)(1−σ(p))` |
| `softmax` (K-class) | `p_k − y_k` | `p_k(1−p_k)`, one tree per class per round |
| `poisson` | `e^p − y` | `e^p` |

The hessian clamp (`max(h, 1e-16)`) is not cosmetic: a saturated logistic gives
`h → 0`, and `G²/(H+λ)` with `λ=0` then divides by zero and produces a `NaN`
gain that wins every split comparison. A test drives the model to saturation
deliberately.

---

## 4. Data input: arrays, with arctic as an adapter

Resolves open question 4 in the ecosystem doc.

**Core `bkit` works on numba arrays.** Dense, `f64`, `NaN` as the missing
marker. This is the path every algorithm uses internally.

**`arctic` is a declared dependency, used by two things only:**
`ColumnTransformer` (which selects columns by name and type) and a
`from_frame` adapter (categorical → ordinal codes, column ordering, dtype
coercion). Everything else never sees a DataFrame.

**Rejected:** making the whole package DataFrame-native. A user who already has
arrays would pay for a column store they do not use, and every algorithm would
carry two code paths — the condition that B18 avoided in btorch by composing
convolution out of im2col and matmul rather than writing a second path.

### 4.1 Missing values in the GBDT

`NaN` goes to a dedicated bin, and each split learns a default direction (the
side with the higher gain) which is also used for unseen `NaN` at predict time.
This is XGBoost's sparsity-aware behaviour and it matters: imputing to the mean
first is both lossy and a different model.

---

## 5. Persistence

`.btsd` with `kind: "bkit.estimator"` ([C3]). One container, one SHA-256, one
set of twelve load-failure messages — already written for btorch, not rewritten
here.

A GBDT's payload is the node array (feature, bin, threshold, left, right, leaf
value, default direction) plus bin edges plus the objective name. A fitted
scaler's payload is two vectors. Both go in the same container with different
header bodies.

**A loaded model must predict bit-identically to the model that was saved**, on
the same input — the same gate btorch's checkpoint already meets, and the
reason the MNIST numbers are quoted from a reloaded checkpoint rather than from
memory.

---

## 6. Gates

Six tiers, as every phase in this project. Gradcheck does not apply (no
autograd here); its slot is taken by **differential testing against published
reference values**, which is the only way to know a GBDT is right.

| tier | proves | form |
|---|---|---|
| 1. feature | every estimator does what it claims, including failure modes | `tests/bkit_*_test.b` |
| 2. **differential** | the answers are right — scikit-learn/LightGBM reference values embedded in the test | hand-computed small cases; closed forms where they exist |
| 3. **protocol** | every estimator satisfies [C1], including `clone()` isolation | one table-driven test over every estimator in the package |
| 4. stress | survives size, repetition, abuse | 1M × 100 fit, 10,000 trees, degenerate columns, all-`NaN` features |
| 5. regression | nothing else broke | the whole existing suite |
| 6. sanitizers | no latent memory defect | ASan + UBSan over the full bkit suite |

Tier 3 is new to this project and exists because a protocol with eleven
implementers fragments silently. It enumerates the estimators and asserts the
contract on each, so a new estimator that forgets `clone()` fails immediately.

### 6.1 The accuracy gates

Public datasets, fetched and SHA-256 verified, never committed — the method
btorch's `get_mnist.sh` established (B25).

| gate | dataset | target |
|---|---|---|
| GBDT binary classification | adult / census income | **≥ 87% accuracy**, ≥ 0.92 ROC-AUC |
| GBDT regression | california housing | **RMSE ≤ 0.50** |
| Random forest | covertype subset | ≥ 85% accuracy |
| Logistic regression | breast cancer | ≥ 97% accuracy |
| k-means | synthetic, known centroids | centroids within 1e-6 |
| PCA | synthetic, known components | agreement with `numba`'s SVD to 1e-10 |

These are the numbers the implementation is published against, and they are
measured from a reloaded `.btsd` model, not from memory.

---

## 7. What is deliberately out

| item | why |
|---|---|
| neural networks (`MLPClassifier`) | `btorch` does this better; a second implementation would diverge |
| `t-SNE`, `UMAP` | visualisation, not modelling; `bplot`'s territory if ever |
| text vectorizers | `btext` owns them ([E12]), conforming to this protocol |
| GPU / distributed | out of scope for the whole ecosystem |
| `partial_fit` / online learning | a different protocol; revisit only on demand |
| calibration, isotonic regression | small and useful; after K12 if asked for |

---

## 8. Open items

| # | item | settled by |
|---|---|---|
| 1 | leaf-wise vs depth-wise as the default | measured on the adult gate; leaf-wise expected to win, must be shown |
| 2 | is binning reusable outside trees (own `hist_` family)? | if `bsci.stats` histograms want it in stage 6 |
| 3 | does `tree_hist` want stage 1's threading? | yes in principle; depends on D3 landing first, and the partition is over samples so it is race-free by construction |
| 4 | SVM: SMO or a linear-only approximation? | SMO is a week on its own; linear SVM via the existing solver may be enough for K10 |
