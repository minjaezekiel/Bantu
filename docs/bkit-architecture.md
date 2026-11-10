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

## 3. Preprocessing — the shape of every transformer

Eleven estimators over two shared base classes, and they are the first
implementers of the protocol, so
the decisions here set the pattern the other nine files follow.

### 3.1 Statistics are `(1, d)` arrays, and `transform` is one broadcast

Every column-wise transformer stores its fitted statistics as arrays of shape
`(1, d)` — `keepdims` true on the reduction that produced them — so `transform`
is two whole-array kernel calls:

```bantu
nd_divide(nd_subtract($X, this.center_, null), this.scale_, null)
```

No Bantu-level loop over columns and none over rows. This is not a
micro-optimisation: the interpreter is a tree-walker, so a per-element loop in
Bantu is three orders of magnitude off a kernel, and a scaler is on the hot
path of every fit in the package.

### 3.2 Four scalers, one `transform` [K11]

`StandardScaler`, `MinMaxScaler`, `RobustScaler` and `MaxAbsScaler` differ
**only in the two numbers they compute**:

| estimator | `center_` | `scale_` |
|---|---|---|
| `StandardScaler` | mean | standard deviation (population, `ddof=0`) |
| `MinMaxScaler` | `min − lo·scale` | `(max − min) / (hi − lo)` |
| `RobustScaler` | median | IQR (`q75 − q25`) |
| `MaxAbsScaler` | 0 | max(\|x\|) |

So `_ScalerBase` holds `transform`, `inverse_transform`, the feature-count
guard and the `NaN` refusal, and each subclass overrides `fit` alone. The
round-trip gate — `inverse_transform(transform(x)) == x` to 1e-12 — therefore
tests one implementation four times instead of four implementations once.

This works **because of** the constraint in §2.1: a child's method body refers
to `this.center_`, never to `super.transform()`, so inheritance in the one
direction Bantu supports is enough.

### 3.3 A zero-variance column gets scale 1 [K15]

A constant column has standard deviation, range, IQR and max-abs all zero, and
dividing by it yields `NaN` for every row — silently, because `NaN` is a
perfectly good `f64`. Every scaler therefore passes its scale through

```bantu
this.scale_ = nd_where(nd_equal($s, 0.0, null), 1.0, $s);
```

which is scikit-learn's handling and has a property worth stating: the column
transforms to `x − center` = exactly 0, and `inverse_transform` adds `center`
back, so **the round-trip is exact on a constant column too**. An error here
would have been defensible; a `NaN` would not.

### 3.4 The feature-count guard is load-bearing, not decorative [K12]

`transform` on a matrix with a different number of columns than `fit` saw must
raise. In NumPy this is caught for you often enough by accident; here it is
caught **never**, because `(n, 1)` against a stored `(1, d)` statistic
**broadcasts** and returns a perfectly well-formed `(n, d)` matrix of nonsense.
Every transformer records `n_features_` at fit and checks it, and the check is
in `_ScalerBase` so it cannot be forgotten by one of four.

### 3.5 Encoders map through `searchsorted`, not a dict [K13]

The obvious encoder keeps a Bantu dict from category to code and loops over
every row. That is O(n) interpreter steps per column and it cannot see a
missing dtype.

Instead: `nd_unique` gives the sorted categories, and `nd_searchsorted` maps a
whole column of values to codes in one native O(n log k) call. The
unseen-category check falls out of the same call rather than needing a second
pass — clamp the returned positions into range, `nd_take` the category each one
landed on, and compare it to the input. A mismatch *is* an unseen value:

```bantu
$pos  = nd_searchsorted($cats, $col, null);
$safe = nd_clip($pos, 0, len($cats) - 1, null);
$hit  = nd_equal(nd_take($cats, $safe, 0), $col, null);   // false => unseen
```

`handle_unknown` then chooses between raising (naming the column and the value)
and mapping to a documented slot, and both branches are the same two kernels.

### 3.6 The mode is a run-length on a sorted column [K16]

`SimpleImputer(strategy="most_frequent")` needs the most common value per
column. Counting each unique value costs O(k·n) with a Bantu loop per unique.
Sorting instead makes it native end to end: the run boundaries are
`nd_nonzero(nd_diff(sorted) != 0)`, the run lengths are `nd_diff` of those
boundaries, and `nd_argmax` picks the longest. O(n log n) in kernels, with the
only Bantu loop being over columns.

Ties go to the smaller value, because the column is sorted ascending and
`nd_argmax` returns the first maximum — the same rule scikit-learn gets from
`scipy.stats.mode`.

### 3.7 `NaN` statistics are an error, not a smaller sample [K14]

The four scalers **refuse** a `NaN` anywhere in `X`, naming the column and
pointing at `SimpleImputer`. scikit-learn instead computes its statistics
`NaN`-aware and leaves the `NaN` in place.

The reason for the difference is concrete rather than philosophical: numba has
`nd_nanmean`, `nd_nansum`, `nd_nanmin` and `nd_nanmax`, and **no** `nanvar`,
`nanstd`, `nanmedian` or `nanquantile`. Three of the four scalers need exactly
those. Supporting `NaN` in `StandardScaler` and `MinMaxScaler` while
`RobustScaler` quietly produced `NaN` would be the worst of the three options —
a surface that looks uniform and is not.

`SimpleImputer` of course accepts `NaN`; it is the one transformer whose job it
is. An **all**-`NaN` column raises there too, because the alternative is a
statistic computed from no data at all.

**Forward path:** `nd_nanvar`, `nd_nanstd` and `nd_nanquantile` are a small
addition to numba's reduction surface — the Welford and selection kernels are
already written and already thread. When they land, the scalers can follow
scikit-learn and this decision becomes a one-line change in `_ScalerBase`.

### 3.8 `PolynomialFeatures` enumerates in scikit-learn's order

The column order is not free: a model trained on one ordering and scored
against another is wrong in a way no test of column *counts* detects. The
enumeration is combinations-with-replacement of feature indices, degree by
ascending degree, which for `d=2, degree=2, include_bias=true` gives

```
[1, x0, x1, x0², x0·x1, x1²]
```

and the test asserts that list, not its length.

The enumerator is CPython's `itertools` algorithm — hold one index vector, emit
a snapshot, advance the rightmost index that can still move — with one addition:
**the emit loop is bounded by the exact combination count** and raises when it
is passed. The advance rule and the index cap must agree for the loop to
terminate, and they are written in two places; mutation-testing this file turned
one mutant of the advance rule into a four-minute hang with no output. A hang is
a worse failure than a wrong answer, and the bound doubles as a statement of
what the enumeration is supposed to produce.

The snapshot is not optional either. Bantu lists are references, so appending
the working vector itself gives a result in which every entry is the same list,
holding whatever the vector ended up as.

---

## 4. Model selection — splitting, and the one thing that must not leak

### 4.1 A splitter is not an estimator [K20]

`KFold` has nothing to fit and nothing to predict. Forcing it through the
estimator protocol would mean `fit`, `predict`, `transform`, `score` and
`clone` all raising on every splitter in the file, and a `protocol_check` that
has to be told to skip five of its seven assertions.

So splitters get their own three-method protocol, and their own check:

```bantu
def get_n_splits()                 // how many folds
def split($X, $y, $groups)         // a list of [train_idx, test_idx]
def partitions()                   // do the test folds cover 0..n-1 exactly?
```

`partitions()` exists because the invariant is not the same for all of them.
`KFold`, `StratifiedKFold` and `GroupKFold` partition the data exactly — every
index in exactly one test fold. `TimeSeriesSplit` does **not**: the earliest
rows are never tested, which is the point of it. A single check that asserted
exhaustiveness for everything would have to exempt `TimeSeriesSplit` by name,
and the next splitter added would be exempted by whoever adds it.

`splitter_check` lives in `selection.b` for the same reason `protocol_check`
lives in `base.b` ([K17]): one definition of the contract, inherited by
everything that comes later.

### 4.2 Splitters return index arrays, not data [K19]

`split` yields `[train_idx, test_idx]` as `i64` arrays, never `X_train`/`X_test`.

Three things follow, and the third is the one that mattered:
* a 5-fold split of a 1M × 100 matrix copies the data **zero** times instead of
  five;
* `cross_val_score` can hand the same indices to `nd_take` once per fold;
* **the invariants become checkable.** "No group appears in two folds" and
  "every index is tested exactly once" are statements about indices. Given only
  `X_train` matrices you would have to search for rows.

### 4.3 A permutation is `argsort` of random keys [K18]

```bantu
nd_argsort(nd_random_uniform([$n], 0.0, 1.0), null)
```

Two kernel calls. The alternative — Fisher–Yates in Bantu — is `n` interpreter
iterations and `n` separate RNG draws, which for a 1M-row shuffle is the
difference between milliseconds and minutes, and the interpreter's C-style
`for` silently stops at 100,000 iterations anyway.

It also delegates to the one RNG ([C4]) without touching it: `nd_random_uniform`
draws from `nd_seed`'s stream, so the same seed gives the same permutation and
nothing in `bkit` carries RNG state of its own.

*The honest caveat:* `argsort` of **distinct** keys is a uniform permutation.
Two equal f64 keys are broken by index, which is a negligible bias — at n = 1M
the chance of any collision is about 1 in 10,000, and a collision perturbs one
adjacent pair.

### 4.4 No `random_state` on a splitter, and what that forces [K21]

scikit-learn's splitters take a `random_state`. `bkit`'s do not: reproducibility
comes from seeding **once**, at the top of the program, which is already how
`numba` and `btorch` work ([C4]). A per-splitter seed would have to call
`nd_seed` to take effect, and that clobbers the global stream for everything
downstream — a model drawing its initial weights after a grid search would get a
stream determined by the splitter's seed. That is precisely the
"each package seeds separately" failure C4 exists to prevent.

The consequence is not a limitation, it is a requirement that was needed anyway:
**`cross_val_score` and `GridSearchCV` materialise their splits once and reuse
them for every candidate.** Re-splitting per candidate with a shuffled splitter
would compare candidates on *different folds*, so the winner would be partly a
draw of the RNG. With the splits fixed, a difference in score is a difference in
the parameters.

### 4.5 The fold-size rule is stated, not implied [K22]

With `n` rows and `k` folds, the first `n mod k` folds get `floor(n/k) + 1` rows
and the rest get `floor(n/k)`. This is scikit-learn's rule and it is written
down because "balanced folds" has several reasonable definitions and a different
one silently changes every cross-validated score a user compares against a
Python baseline.

`StratifiedKFold` applies the same rule **per class**, and rotates the starting
fold from one class to the next. Without the rotation every class puts its extra
sample in fold 0, so with three classes each having one extra, fold 0 is three
rows larger than the last — each class is still balanced to ±1, but the folds
are not. The rotation is a one-line accumulator and it is the difference between
"stratified" and "stratified and balanced".

### 4.6 `train_test_split` returns a named dict, not four positionals

```bantu
$s = sel.train_test_split($X, $y, 0.25, true, null);
$s.X_train    $s.X_test    $s.y_train    $s.y_test
```

scikit-learn's `X_train, X_test, y_train, y_test = train_test_split(...)` is
famously easy to get wrong — the order interleaves the two variables rather than
grouping them, and swapping two of them produces a working program that trains
on the test set. In Bantu the call would have to be indexed as `$r[2]` and
`$r[3]`, which is worse still. Named fields cost nothing and cannot be
transposed.

### 4.7 A mistyped parameter is an error [K24]

Found while designing this phase, fixed in `base.b`: `set_params` accepted any
key and added it to `params_`. So `GridSearchCV` over a grid with a mistyped
key — `with_men` for `with_mean` — ran every candidate with identical settings,
scored them identically, and reported a best configuration it had never tried.

`set_params` now refuses a key `get_params()` does not already hold, and names
the ones it takes. Presence is tested with `keys()` rather than `== null`,
because a parameter may legitimately *hold* null: `OrdinalEncoder`'s
`unknown_value` does, so a null read cannot tell "absent" from "set to null".

`GridSearchCV` validates the whole grid against the estimator **before** the
first fit, so a typo costs a message instead of a few minutes of fitting.

---

## 5. Metrics — one confusion matrix, and where the zeros are

### 5.1 `metrics.b` depends on nothing but numba

Not even on `base.b`. Nothing in the file is an estimator: they are pure
functions of two arrays, so the estimator protocol has nothing to say about
them and tier 3 does not apply.

Keeping it dependency-free is the same decision as [K7] for `base.b` and for the
same reason: a user who wants `f1_score` should not import scalers, splitters
and a gradient booster to get it. It also means `base.b`'s `score()` does **not**
delegate here — it keeps its two inlined formulas, because the dependency would
have to run the other way and `base.b` is the file everything else depends on.

### 5.2 Every classification metric is a view of one confusion matrix [K26]

`precision_recall_fscore_support` computes the matrix once and reads everything
off it:

```
tp = diag(cm)        fp = colsum(cm) − diag       fn = rowsum(cm) − diag
support = rowsum(cm)
```

`precision_score`, `recall_score`, `f1_score` and `classification_report` all
call it rather than recomputing. So the **zero-division rule lives in one
place**, and a class that never appears cannot be handled one way by `precision`
and another way by `f1`.

The matrix itself is built without a loop over samples: encode both label
arrays to codes `0..k−1`, form `true_code · k + pred_code`, and
`nd_bincount(..., k·k)` reshaped to `(k, k)`. Three kernel calls for any number
of rows.

### 5.3 A zero-support class scores 0, and the test proves it [K27]

A class present in `y_true` but never predicted has `tp + fp = 0`, so precision
is `0/0`. A class never present but sometimes predicted has `tp + fn = 0`, so
recall is. `f1` is `0/0` whenever `precision + recall = 0`.

All three yield **0**, which is scikit-learn's `zero_division=0` default. The
alternative — `NaN` — propagates into the macro average and turns one absent
class into a `NaN` for the whole model, which is the silent-wrong-answer shape
this package exists to avoid. An error was the other option and is too strict:
a class missing from one cross-validation fold is normal.

The test includes a 3-class case with a class that is in `y_true` and never
predicted, **and** a class predicted but never true, and asserts the macro
average is a finite number computed from the zeros rather than around them.

### 5.4 The curves emit one point per *distinct* threshold [K28]

This is what "handling ties" means, and it is not a detail. With scores
`[0.8, 0.8, 0.8]` and labels `[1, 0, 1]`, a curve that advanced one sample at a
time would trace a staircase through points that no threshold can actually
produce — and the area under that staircase depends on the order the equal
scores happened to arrive in. The AUC would change when the input was shuffled.

So after sorting by score descending and taking cumulative true and false
positives, a point is emitted only at the **end of each run of equal scores**:
the run boundaries are `nd_nonzero(nd_diff(sorted_scores) != 0)` plus the last
index — the same construction the mode uses in [K16].

`roc_auc_score` is then the **trapezoid** over `(fpr, tpr)`.
`average_precision_score` is deliberately **not**: it is the step sum
`Σ (R_n − R_{n−1}) · P_n`. Those two give different numbers on the same curve,
scikit-learn uses exactly this pair, and a test asserts both against a hand case
so neither drifts into the other's rule.

### 5.5 `log_loss` is clipped, so a confident mistake is finite

`log(0)` is `−∞`. One prediction of `0.0` for a sample whose label is 1 would
make the loss of an entire 10,000-row evaluation `inf`, which says nothing about
the model beyond "at least one prediction was certain and wrong". Probabilities
are clipped to `[eps, 1 − eps]` with `eps = 1e-15` by default, so that sample
contributes about 34.5 and the rest of the number survives.

### 5.6 `r2_score` raises on a constant target [K30]

`R² = 1 − SSE/SST`, and a constant target has `SST = 0`. scikit-learn returns
`0.0` with a warning (or `1.0` if the predictions are exactly right).

`bkit` raises, for consistency with `base.b`'s `score()`, which already did so
before this file existed ([K1]'s test asserts it). A model cannot be scored on a
constant target by R² — the metric's denominator is the thing it normalises
against — and returning `0.0` makes a meaningless comparison look like a bad
one. `mean_squared_error` is the metric for that case and the error message
says so.

`mean_absolute_percentage_error` is the mirror image and is handled the other
way: a zero in `y_true` makes one term infinite, and the denominator is clamped
to `eps` rather than raising, because a single zero among thousands of nonzero
targets is a data point and not a reason to refuse the whole metric. The
clamping is documented at the function, since it means the result is bounded
rather than exact there.

### 5.7 `classification_report` returns a dict; the text is a separate function [K29]

A metric whose only output is a formatted string can only be tested by matching
that string, and then every change to the column widths breaks the test. So
`classification_report` returns the numbers — one entry per class plus
`accuracy`, `macro avg` and `weighted avg`, which is scikit-learn's
`output_dict=True` shape — and `classification_report_text` formats them.

The numbers are asserted numerically; the formatter is asserted only to mention
every class and to produce the right number of lines.

### 5.8 `ds.accuracy` moves, but btorch keeps its implementation [K25]

[C5] and [E14] say the metric moves to `bkit.metrics` with a **delegating**
alias left in `btorch`. The move happens; the delegation does not, and the
reason is [E2]'s own argument.

`btorch` is **1.0.0 and gated on MNIST accuracy**. `bkit` is `0.3.0` and changes
every phase. Delegating would make the stable package depend on the churning
one, so a `bkit` release could break a `btorch` install — "blast radius becomes
release cadence", which is why these are separate packages at all.

So `bkit.metrics.accuracy_score($y_true, $y_pred)` is the home, and
`btorch.datasets.accuracy($logits, $labels)` keeps its eight lines, marked
deprecated in the docstring and the CHANGELOG. The two are *not* the same
function — btorch's takes logits and argmaxes them — so a **differential test**
asserts they agree on the same inputs. That test, not the delegation, is what
stops them drifting.

---

## 6. Gradient boosting — the headline

Histogram-based, LightGBM-shaped. The design choices and why.

### 6.1 The algorithm

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

### 6.2 The `tree_` native family

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

### 6.3 The binned matrix is an opaque handle, not an `NdArray`

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

### 6.4 Objectives

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

## 7. Data input: arrays, with arctic as an adapter

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

### 7.1 Missing values in the GBDT

`NaN` goes to a dedicated bin, and each split learns a default direction (the
side with the higher gain) which is also used for unseen `NaN` at predict time.
This is XGBoost's sparsity-aware behaviour and it matters: imputing to the mean
first is both lossy and a different model.

---

## 8. Persistence

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

## 9. Gates

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

### 9.1 The accuracy gates

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

## 10. What is deliberately out

| item | why |
|---|---|
| neural networks (`MLPClassifier`) | `btorch` does this better; a second implementation would diverge |
| `t-SNE`, `UMAP` | visualisation, not modelling; `bplot`'s territory if ever |
| text vectorizers | `btext` owns them ([E12]), conforming to this protocol |
| GPU / distributed | out of scope for the whole ecosystem |
| `partial_fit` / online learning | a different protocol; revisit only on demand |
| calibration, isotonic regression | small and useful; after K12 if asked for |

---

## 11. Open items

| # | item | settled by |
|---|---|---|
| 1 | leaf-wise vs depth-wise as the default | measured on the adult gate; leaf-wise expected to win, must be shown |
| 2 | is binning reusable outside trees (own `hist_` family)? | if `bsci.stats` histograms want it in stage 6 |
| 3 | does `tree_hist` want stage 1's threading? | yes in principle; depends on D3 landing first, and the partition is over samples so it is race-free by construction |
| 4 | SVM: SMO or a linear-only approximation? | SMO is a week on its own; linear SVM via the existing solver may be enough for K10 |
