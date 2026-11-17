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

One package, twelve files. The split is by what a user imports together, and
`base.b` is deliberately tiny so `btext` can depend on it without pulling in
GBDT ([E12]) — or, as `compose.b` shows, without pulling in `arctic` ([K37]).

| file | contains |
|---|---|
| `base.b` | the estimator protocol and nothing else |
| `preprocess.b` | scalers, encoders, imputers, polynomial features |
| `selection.b` | splits, K-fold, cross-validation, grid/randomised search |
| `metrics.b` | classification, regression and ranking metrics |
| `linear.b` | linear/ridge/lasso/elastic-net, logistic regression |
| `compose.b` | `ColumnTransformer` — the only file that knows `arctic` exists |
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

## 6. Linear models — and the Gram matrix we do not form

### 6.1 Never form `XᵀX`, measured rather than argued [K31]

The roadmap said *"normal equations via numba's `lstsq`; `solve` for
well-conditioned"*. The second half is wrong, and the measurement is the
argument.

A degree-9 polynomial design on 60 points — an ordinary, unremarkable fit:

| | value |
|---|---|
| condition number of `X` | 3.6 × 10⁶ |
| condition number of `XᵀX` | **1.3 × 10¹³** |
| max coefficient error, `lstsq` | 1.2 × 10⁻⁹ |
| max coefficient error, normal equations | **2.8 × 10⁻³** |

Forming the Gram matrix **squares the condition number**, so the normal
equations came out **2.4 million times less accurate** — a 0.3% error in
coefficients the caller has no way to notice. Their cost: `lstsq` took 41 ms on
a 5000 × 60 problem against 7 ms for the Gram route, and on a *well*-conditioned
problem the two agree to 2.4 × 10⁻¹⁶.

So: 34 ms, on a fit that happens once, against a silent three-digit error. Every
fit in this file goes through `nd_lstsq` on the design matrix.

The deeper reason not to make it conditional is that **the caller cannot know
which regime they are in**. "Use `solve` when well-conditioned" requires
computing the condition number to decide — which costs an SVD, more than the
`lstsq` it was trying to avoid.

### 6.2 `Ridge` is least squares on an augmented system [K32]

Ridge has a closed form, `(XᵀX + αI)b = Xᵀy`, and that form requires the Gram
matrix §6.1 just rejected. It is not needed:

```
minimise ‖Xb − y‖² + α‖b‖²   ==   minimise ‖ [  X        ] b − [ y ] ‖²
                                            [ √α · P     ]     [ 0 ]
```

Stack `√α` rows under the design matrix and zeros under the target, and `lstsq`
solves the penalised problem directly. Two things fall out for free:

* the Gram matrix is never formed, so Ridge inherits §6.1's accuracy;
* `P` selects the penalised columns, so **the intercept is excluded by
  construction** rather than by remembering to exclude it (§6.3).

It also explains what the penalty does numerically: the added rows raise the
smallest singular value to at least `√α`, so a rank-deficient design becomes
full rank. Ridge is better conditioned than the fit it regularises, which is
why the closed form is *safe* here and not elsewhere.

### 6.3 The intercept is a column of ones, and is never penalised

Every model fits `[1 | X]` and reports `coef_` and `intercept_` separately.

Penalising the intercept is the classic silent error: it shrinks predictions
toward **zero** instead of toward the **mean**, so a model of a quantity
centred on 1000 is dragged to 0 and reports a terrible fit that looks like
underfitting rather than a bug. `Ridge` excludes it through `P` above;
`ElasticNet` excludes it by centring `X` and `y`, fitting without an intercept,
and recovering `intercept_ = ȳ − x̄ · b`; `LogisticRegression` excludes it in the
same `P`.

### 6.4 Coordinate descent, and a cap that is *heard* [K33]

`Lasso` and `ElasticNet` have no closed form — the L1 term is not
differentiable at zero — so they use cyclic coordinate descent with soft
thresholding, minimising

```
(1/2n)‖y − Xb‖² + α·ρ·‖b‖₁ + ½·α·(1−ρ)·‖b‖²
```

(`ρ` is `l1_ratio`; `ρ = 1` is Lasso.) Each coordinate has a closed-form
update, and the residual is maintained incrementally so a sweep is `p` kernel
calls rather than `p` matrix products.

**The iteration cap must be audible.** A coordinate descent that stops at
`max_iter` without converging returns coefficients that are simply *wrong* — not
approximate, wrong — and every downstream number computed from them is wrong
too, with nothing in the output to say so. So:

* `converged_` and `n_iter_` are on the estimator and a test asserts them;
* hitting the cap prints a warning **to stderr** (`eprint`), once per fit, so it
  cannot be lost in a pipeline's stdout;
* the warning names the tolerance reached and what to change.

It does **not** raise. Hitting `max_iter` on hard data is normal and a user who
wants more iterations should get a warning and a number, not an exception in the
middle of a cross-validation.

### 6.5 `LogisticRegression`: binary IRLS, multiclass one-vs-rest [K34]

Binary uses **IRLS** — Newton's method on the log-likelihood — where each step
is a weighted ridge solved by §6.2's augmented `lstsq`. It converges in five to
ten iterations on ordinary data, quadratically, with no learning rate to tune.

Multiclass is **one-vs-rest**: `k` binary fits, prediction by the largest
decision value. The alternative, multinomial softmax, needs a general optimiser
(L-BFGS) because its Hessian is a `k·p × k·p` block matrix — and writing one is
a phase of its own, not a line of this file.

What that costs is worth stating precisely, because "one-vs-rest is worse" is
usually said without saying how: for **prediction** it is not worse — `argmax`
of `k` independent fits and `argmax` of a softmax fit agree on the great
majority of points. What it costs is **calibration**: the `k` sigmoids do not
sum to 1, so `predict_proba` normalises them, and that normalisation is an
approximation rather than a model. scikit-learn's `liblinear` solver does
exactly this and documents it the same way.

The L2 penalty is not optional here. On separable data the unpenalised
likelihood is maximised by coefficients of infinite magnitude, so IRLS diverges
with weights going to zero. `C` (inverse regularisation strength, as in
scikit-learn) keeps `b` bounded, and probabilities are clipped away from 0 and 1
so the IRLS weights `p(1−p)` cannot underflow to exactly zero.

### 6.6 `SGD*` is mini-batch, and that is a different optimiser [K35]

"One pass per epoch, `nd_*`-vectorised" and "stochastic gradient descent" are in
tension: true per-sample SGD is `n` interpreter iterations per epoch, and the
interpreter costs about a microsecond a step — so a 100,000-row epoch is a tenth
of a second of pure interpreter overhead before any arithmetic.

So these are **mini-batch** gradient descent, default batch 256, where each step
is one `matmul`. The batch is a declared parameter and not a hidden one.

This is an honest difference from scikit-learn's `SGDClassifier`, which really
does take one sample at a time: **the optimisation path differs, so the
coefficients differ**, even though both minimise the same objective. What is
asserted instead is the property that matters — on a well-conditioned problem,
`SGDRegressor` must approach `LinearRegression`'s closed-form coefficients. An
optimiser is correct when it finds the answer the closed form already knows, and
that is a far stronger test than reproducing another library's trajectory.

### 6.7 The step size is not coupled to the penalty [K36]

The first schedule was `η₀ / (1 + η₀·α·t)`, and it is broken at **both** ends:

* with a strong penalty the denominator grows fast and the step collapses. At
  `α = 2` on a 90/10 classification the intercept read **−0.754** after 400
  epochs, **−0.938** after 4,000 and **−1.046** after 20,000 — still moving,
  never reporting convergence — against the **−2.069** that IRLS gives for the
  same objective. The optimiser was not converging slowly; it was being
  governed by the one parameter that has nothing to do with step length.
* with `α = 0` it never decays at all, so there is no annealing exactly where
  gradient noise most needs it.

Centring `X` and `y` (§6.6) hid this for the regressor's *intercept*, by
computing it in closed form instead of descending to it. The coefficients, and
the whole of `SGDClassifier`, were still throttled.

`learning_rate` is now an explicit choice and the default depends only on `t`:

| | |
|---|---|
| `"invscaling"` (default) | `η₀ / (1 + t)^power_t`, `power_t = 0.25` |
| `"constant"` | `η₀` |
| `"optimal"` | `1 / (α·(t + t₀))`, `t₀ = 1/(α·η₀)` |

Under the default, the same `α = 2` fit reaches **−2.031** in 400 epochs.

`"optimal"` is the old rule, kept rather than deleted because it is the right
schedule for a strongly convex objective — the strong convexity constant *is*
`α` — and it is scikit-learn's name for it. It now has to be asked for, and
requires `α > 0`, since with `α = 0` there is no constant for it to use.

**Why `invscaling` and not `constant`** — the choice measurement had to settle
rather than taste:

| | noiseless target | noisy target (σ = 0.5) |
|---|---|---|
| `constant` | 8.4 × 10⁻⁹ at 175 epochs | 7.8 × 10⁻³ |
| `invscaling` | 6.1 × 10⁻⁴ at 400 epochs, **9.7 × 10⁻¹¹** at 2,000 | **2.1 × 10⁻³** |

A decaying step is *slower early* on a clean problem and **not capped** — given
the epochs it converges further than a constant one. On a noisy problem it is
nearly four times closer, because the decay is what lets the iterate settle
instead of bouncing around the optimum. Real targets are noisy, so that is the
default. Neither schedule diverged at `η₀ = 5`, so the usual robustness
argument for decay was tested, did not hold here, and is not claimed.

### 6.8 These are the first real estimators, so the protocol test drops its doubles

K1 through K4 had no classifier or regressor, so `bkit_protocol_test.b` wrapped
`GridSearchCV` around a hand-written `ConstRegressor` to have something
scorable. From K5 the registry carries `LinearRegression`, `Ridge`,
`ElasticNet`, `Lasso`, `LogisticRegression` and both `SGD*` estimators, and the
searches are exercised over a real model whose right answer is known in closed
form.

---

## 7. Composition — `ColumnTransformer`, and why it is not in `base.b`

`Pipeline` lives in `base.b`: the protocol harness needs it, and it depends on
nothing but the contract. `ColumnTransformer` lives in `compose.b`, because it
is the **only** part of bkit that knows `arctic` exists (§9) and `base.b` is
what `btext` imports for the contract alone. Importing the estimator protocol
must not drag a dataframe package behind it ([K37]).

`compose.b` does not `include` arctic either. It duck-types — a frame is an
instance carrying `names`, `cols` and `ncols`, and reading an undefined field
on an instance yields null rather than raising. So `arctic` is an **optional**
dependency: bkit works fully without it, and only the by-name column path needs
it.

### 7.1 The spec

```bantu
$ct = compose.ColumnTransformer([
    ["num", pre.StandardScaler(null, null), [0, 1, 2]],
    ["cat", pre.OneHotEncoder(null),        ["colour"]]
], "passthrough");
```

Each spec is `[name, estimator, columns]`. Columns are integer indices, arctic
column names, or a mix. **Both resolve to integer positions against one
materialised matrix before any transformer runs**, so there is exactly one code
path after resolution and the frame case cannot drift from the array case. A
frame is converted once, not once per spec.

### 7.2 Output order is spec order

Blocks are concatenated in the order the specs were given — **not** in column
order ([K38]). A caller can predict spec order; they cannot predict column
order, because a `OneHotEncoder` on one input column emits an arbitrary number
of outputs.

Hence `output_ranges()`, returning `[name, start, stop)` per spec plus the
remainder. A 40-column one-hot block is otherwise unattributable, and *"which
input produced feature 37"* is the first question anyone asks of a fitted
composition. The widths are **learned**, not declared, so this and
`n_features_out()` both require a fit ([K41]).

### 7.3 The two failures this is built against

Both produce a wrong answer that looks like a right one, which is the criterion
the whole K series uses.

**A reordered frame.** Names resolve to positions *at fit time*. A frame whose
columns were reordered between `fit` and `transform` would resolve the same
names to different positions and transform the wrong data, at the right shape,
with plausible numbers. So `transform` compares the fitted column **names**, not
just the count ([K40]).

**A block that changes width.** A transformer emitting a different number of
columns at transform time would shift every later block's meaning. Caught with
both widths and the offending spec's name.

### 7.4 Duplicates

A column may feed two *different* specs — scaled and binarised, say, which
sklearn also allows. The same column twice **within one spec** is refused: it
would transform the column twice and silently widen that block ([K39]).
`remainder` is the complement of the union, emitted in ascending index order, so
the passthrough block is X's own order rather than a dict's.

### 7.5 `fit` delegates to `fit_transform`, not the reverse

Fitting a composition already computes every block, so the base class's default
— `fit` then `transform` — would do all the work twice and let a stateful
transformer see the data twice per fit. `fit_transform` is the implementation
and `fit` calls it, discarding the result ([K41]).

---

## 8. Gradient boosting — the headline

Histogram-based, LightGBM-shaped. The design choices and why.

### 8.1 The algorithm

1. **Bin once, up front.** Each feature is quantile-binned into at most 256
   bins. All later work reads bin indices, never raw values. This is what makes
   split finding O(bins) instead of O(sorted samples).
2. **Per node, build histograms** of `(Σgradient, Σhessian, count)` per bin per
   feature over that node's samples.
3. **Scan bins for the best split**, gain being the standard regularised form
   `½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)] − γ`.

   **This same expression is also a decision tree's criterion**, which is why
   `tree.b` and the booster share one code path ([K42]). With `g = −y`, `h = 1`
   it IS variance reduction for regression, and for binary classification it is
   **proportional to Gini decrease** — `Σ n_m·Gini_m decrease = 4 × gain`, a
   positive constant, so the argmax is identical to `criterion="gini"`.
4. **Histogram subtraction:** a child's histogram is its parent's minus its
   sibling's. Compute the smaller child, subtract for the larger. This roughly
   halves the work per level and is the single largest optimisation in the
   algorithm.
5. **Leaf-wise growth** with a `max_leaves` cap (LightGBM's choice), plus
   `max_depth` as a separate guard.
6. **Predict** by traversing the node array.

### 8.2 The `tree_` native family

Five kernels. The Bantu side runs the boosting loop, computes gradients and
hessians (vectorised over `nd_*`), handles early stopping and feature
importance — all of which are one call per *iteration*, not per element.

| kernel | signature | why native |
|---|---|---|
| `tree_bin` | `(X, max_bins?) -> handle` | binning over `n×f` values; bins come from the **sorted distinct** values, so a low-cardinality column gets its own cardinality rather than 256 mostly-empty bins |
| `tree_hist` | `(binned, grad, hess, idx?) -> [f, bins, 3]` | **the hot loop**: one byte load and three float adds per (sample, feature) per node |
| `tree_best_split` | `(hist, binned, λ?, γ?, min_child_weight?, min_data?) -> [feat, bin, gain, G_L, H_L, n_L, G_R, H_R, n_R, nan_left]` | O(features × bins) scan with a running prefix, run **twice per feature** so the `NaN` direction is learned ([K44]) |
| `tree_partition` | `(binned, idx, feat, bin, nan_left?) -> [left, right]` | stable partition of an index array; `bin = -1` is the missing-vs-present split |
| `tree_predict` | `(nodes, X) -> y` | per-sample traversal, branch-heavy. **No edge table**: thresholds are raw values, so a fitted tree is one `[n_nodes, 7]` matrix ([K46]) |

Plus five accessors — `tree_bin_shape`, `tree_bin_nbins`, `tree_bin_edges`,
`tree_bin_at`, `tree_bin_bytes` — which exist **for the tests, not for
callers** ([K47]). An opaque structure with no window is a structure whose
contents are asserted by hope: `tree_bin_at` is what lets the reference
histogram be computed in Bantu from the same bins the kernel used, so
`tree_hist` is checked against something that is not itself.

**Where they are compiled.** Their own file (`tree_native.hpp`), their own
prefix, their own registration function — but **numba's translation unit**.
Every kernel reads an `NdArray`'s raw storage, and crossing the deliberately
narrow `ndarray_api.hpp` wall would mean exporting X as a flat
`std::vector<double>`: 800 MB of copy for 1M × 100, to feed kernels whose entire
purpose is to replace those doubles with one byte per cell. Sharing the TU also
gets them the `-O3 -ftree-vectorize` that `ndarray_native.cpp` alone is built
with, which the histogram loop is the reason for. The family is separate; the
compilation unit is shared, for a measured reason.

A Bantu function call costs **1.78 µs** (btorch B1). `tree_hist` on a
1M × 100 dataset touches 10⁸ cells per level — four orders of magnitude outside
what Bantu can do per element. These five are not an optimisation; without them
there is no GBDT.

### 8.3 The binned matrix is an opaque handle, not an `NdArray`

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

### 8.4 Objectives

Gradients and hessians are pure Bantu over `nd_*` — one vectorised call per
boosting iteration.

| objective | gradient | hessian |
|---|---|---|
| `squared_error` | `p − y` | `1` |
| `logistic` | `σ(p) − y` | `σ(p)(1−σ(p))` |
| `softmax` (K-class) | `p_k − y_k` | `p_k(1−p_k)`, one tree per class per round |
| `poisson` | `e^p − y` | `e^p` |

A saturated logistic gives `h → 0`, and `G²/(H+λ)` with `λ=0` then divides by
zero. **This paragraph used to say the result is a `NaN` gain that wins every
comparison, and that is backwards** — [K45] is the correction. A `NaN` *loses*
every `>` comparison and is therefore harmless; the danger is the other branch,
`G²/0` with `G ≠ 0`, which is `+inf` and **wins** every comparison, picking a
split on a child with no weight in it with maximum confidence. A test drives the
model to saturation deliberately.

The same paragraph also prescribed a **hessian clamp**, `max(h, 1e-16)`, and
called it "not cosmetic". It is worse than cosmetic, which [K58] measured: it
was implemented, then removed, because the number it protects against is not the
one it creates:

| h | λ | min_child_weight | result |
|---|---|---|---|
| **1e-16** | 0 | 0 | **splits; leaves of ±5e15** |
| 0 | 0 | 0 | one leaf, value 0 |
| 1e-16 | 1 (the default) | 1 | one leaf, value 0 |

The middle row is correct and two guards already produce it. Lifting `h` to
`1e-16` makes `H + λ` clear **both** of them by a hair and turns a refused split
into a step of five quadrillion — the `+inf` failure mode above, scaled down
just far enough to survive a finiteness check. It was invisible at the defaults,
which is the worst place for a hazard to hide. So the objective corrects only
what is **invalid** — a negative hessian or a `NaN`, both to 0 — and a genuine
zero is passed through to the guards that handle it:

1. `tree_best_split`'s `!(Hs + λ > 0.0)` on the **parent total**, which skips
   the whole *feature* ([K45]);
2. the same test per **candidate cut**, on `H_L` and `H_R`, which skips that
   *split*;
3. `_leafValue`'s identical guard, which bounds the *leaf* ([K55]).

Layers 1 and 2 look redundant and are not, which [K62] found the hard way:
removing layer 2 left all 156 assertions green, because the only fixture that
drove hessians to zero drove **all** of them to zero — so layer 1 skipped the
feature and layer 2 was never reached. **A fixture that trips the outer guard
cannot exercise the inner one.** Reaching layer 2 needs a healthy parent and a
starved child: `h = 1` on three rows and `h = 0` on four, so the parent total is
3 and the feature is scanned, while one cut leaves `H_R = 0` with `G_R ≠ 0` —
a `+inf` gain that wins every comparison.

Layer 3 was missing, and that was [K55]: the kernel refused the split while the
caller, computing `−G/(H+λ)` from the same two quantities, raised
`Division by zero` out of the middle of `_grow`. K7's estimators cannot reach it
(`h = 1`, so `H = n ≥ 1`); K8's logistic objective is exactly what reaches it.
The guard returns `0.0`, which is what XGBoost's `CalcWeight` and LightGBM both
do: at `H + λ = 0` the second-order approximation has no curvature and its
minimiser is unbounded, which is an artefact of the approximation and not a step
worth taking. **A guard in the kernel is not protection for the caller**, and a
clamp upstream of a guard can disarm it.

### 8.9 The forests, and the one argument ExtraTrees needs

**Status: both are built and gated.** The section is kept as written, including
the rows this work corrected, because what it got wrong is the useful part —
and in this section it got two things wrong: the feature-sampling row ([K63])
and the choice between the three ExtraTrees options ([K65]).

`RandomForest*` and `ExtraTrees*` are [§8.8](#88-forests-bootstrap-and-random-thresholds)'s
two switches over `n_estimators` independent trees, grown by the same
`tree.TreeGrower` the booster uses. Three of the four pieces need nothing new:

| piece | how |
|---|---|
| **bootstrap** | an index array **with repeats**, which `tree_hist` counts once per occurrence — that *is* bootstrap semantics, and `TreeGrower.grow` already takes a root index |
| **feature subsampling** (`max_features`) | `set_features_per_split(k)`, the `[f,1,1]` histogram mask **redrawn at every node**. This paragraph first said "`set_features(cols)` per tree", reusing the booster's hook, and [K63] is the record of why that is wrong: drawing once per tree is a random *subspace* ensemble, and with `f = 6`, `k = 2` a tree never sees the one informative feature with probability 2/3 — measured R² 0.66 against 0.995 |
| **averaging** | in **probability space**: average `predict_proba`, then `argmax`. A majority vote over hard labels throws away every tree's confidence and is a different, worse estimator |

**ExtraTrees is the one that does not fit, and the design doc should say so
before the code does.** §8.5 claimed it reduces to "pick a bin per candidate
feature and evaluate that one cut's gain in Bantu from the histogram". That is
true of the *arithmetic* and false of the *control flow*: the grower calls
`tree_best_split` itself, deep inside `_grow`, and there is no way to hand it a
cut. Nothing in §8.5 noticed, because the claim was checked against the kernels
and not against the caller.

Three ways out, and this section first recorded the middle one. Implementing it
found the third, which is better than both:

| option | verdict |
|---|---|
| a new kernel, `tree_eval_split(hist, binned, λ, γ, …, chosen_bins)` | **rejected**, and the first stated reason was *wrong*. It said the gain formula would exist in two places — but two functions in one file can share one inline helper, so that was a confusion between "two call sites" and "two copies". The real objection is smaller and still sufficient: a second entry point duplicates the *validation* — the shape checks, the `nbins` agreement, the `NaN`-bin bookkeeping — which is where [K44] and [K48] both lived |
| one virtual method on the grower | **superseded.** `_chooseSplit($hist, $B)` overridden by an extra-trees grower, evaluating the cut **in Bantu**. Workable, and it puts the regularised gain in a second *language*, where it can drift from the C++ one silently and no test would see it until the two disagreed. It also pays `O(max_features)` native calls **per node** instead of none |
| **one optional argument to `tree_best_split`** | **chosen.** `cuts`, a per-feature vector of bin indices: when present, feature `f` considers **only** bin `cuts[f]`. Five lines inside the existing loop — `if (haveCuts && b != cuts[f]) continue;` — and everything else is untouched |

The third option is better on every axis the first two were argued over. The
gain formula stays in **one** place because there is only one loop. The
validation stays in one place because there is only one entry point. The scan
stays **native**, so ExtraTrees pays nothing per node where the virtual method
would have paid `O(max_features)` dispatches. And the `NaN` double pass, the
guards, the tie-break and the 10-slot return are all inherited rather than
reimplemented — which matters more here than anywhere, because a randomly
chosen cut is *more* likely to produce an empty or zero-hessian child than a
scanned one, so [K45]'s and [K55]'s guards are on the hot path rather than the
cold one.

**The random draw stays in Bantu**, not in the kernel. `cuts` is drawn per node
from `nd_random_*`, so the forest remains reproducible from `nd_seed` alone
([C4]) with no RNG state inside a kernel — and, more importantly for the gate,
**the kernel becomes fully deterministic given `cuts`**. A test can hand it a
known vector and assert the exact split it must return, which is not possible
if the kernel draws its own.

Drawing it is one vectorised expression, because `tree_bin_nbins` already
reports each feature's admissible range:

```
cuts = floor(random(f) * (nbins - 1))      # bin b means "values <= edges[b]"
```

so the per-node cost on the Bantu side is three native calls, not
`O(max_features)`.

Two details this gets **for free** by staying inside the one loop, and both are
things a separate implementation would have had to rediscover:

- **the chosen cut is evaluated BOTH ways for `NaN`**, exactly as [K44]
  requires. The `NaN` bin is the highest index, so a prefix sum always leaves it
  right; a random cut inherits that bias unless the direction is chosen on
  merit. The existing two-pass loop already does this, and `cuts` restricts the
  bin, not the pass.
- **the same guards still apply**, and they matter *more* here. A randomly
  chosen cut is far likelier than a scanned one to leave a child empty or
  weightless, so `n_L < min_data`, `H_L < min_child_weight` and
  `!(H_L + λ > 0)` are on the hot path rather than the cold one. `feature = −1`
  comes back when every candidate fails, which the grower already treats as "no
  split" ([K55]).

One thing `cuts` must **not** do is silently clamp. A bin index outside
`0 .. nbins_f − 2` is a caller error, not a value to round into range: clamping
would make an off-by-one in the drawing expression look like a working forest
with a biased cut distribution, which is precisely [K48]'s failure mode — a
wrong answer that still produces plausible trees. It raises.

**Reproducibility** is [C4] and has a sharper test here than elsewhere: each
tree draws its index sample and its column sample from one seeded stream, so
**fitting 10 trees and then fitting 20 must give the same first 10**. That rules
out a per-tree seed derived from the tree count, which is the natural wrong
implementation and which looks reproducible until `n_estimators` changes.

---

## 9. Data input: arrays, with arctic as an adapter

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

### 8.5 K8 needs no new kernels, and that is the finding

The five `tree_` kernels K7 shipped are sufficient for the whole of `ensemble.b`.
Worth stating explicitly, because the instinct is to add a kernel per feature:

| K8 feature | how, over the existing kernels |
|---|---|
| **histogram subtraction** | `nd_subtract(parent_hist, sibling_hist)` — **one vectorised call per node**, on a `[features, bins, 3]` array. There is nothing per-element here for Bantu to be slow at |
| **bootstrap sampling** (random forest) | `tree_hist` takes an index array, and a duplicated index is counted twice — which *is* bootstrap semantics. No kernel change |
| **feature subsampling** | zero the unselected features' rows in the histogram before `tree_best_split`. A zeroed feature has `G = H = n = 0`, so every cut has `n_L = 0 < min_data` and is skipped |
| **random split thresholds** (extra trees) | pick a bin per candidate feature and evaluate that one cut's gain in Bantu from the histogram. One cut is arithmetic on six numbers |
| **leaf-wise growth** | a frontier keyed by gain, maintained in Bantu. One `tree_best_split` call per node either way |
| **early stopping** | a validation fold scored per round; one `tree_predict` per round |

So K8 is a boosting *loop*, not a kernel project. The expensive inner work —
binning once up front, one histogram per node, one scan per node — is already
native, and everything K8 adds is one call per node or per round.

### 8.6 Growth: depth-wise now, leaf-wise later, with histogram subtraction

K7's grower is **depth-wise** breadth-first, bounded by `max_depth`. The
booster uses **the same grower**, and that is a deliberate scope decision, not
an omission: depth-wise is XGBoost's default `grow_policy`, and leaf-wise with a
`max_leaves` cap is LightGBM's. Both are standard; neither is the correct one.

**Shipping with the shared depth-wise grower first** means K8's new code is the
boosting *loop* — gradients, hessians, shrinkage, early stopping, importances —
over a grower whose `NaN` direction, empty-child assertion and threshold lookup
are already tested. A second grower inside `ensemble.b` would mean two places
that have to get [K44]'s `NaN` direction right, and one place already got it
subtly wrong.

`max_leaves` and leaf-wise growth are then a follow-up *inside that same
grower*: pick the highest-gain frontier node instead of draining a queue. Note
that when `max_leaves` is unset the two policies produce the same set of splits
— every node that can split, does — so adding it changes node *ordering* and
not the model, which is what makes it a safe later change rather than a rewrite.

This is the opposite call from [B34] in btorch, where sharing one normalisation
implementation cost four full tensor copies and was split apart. The difference
is what sharing costs: there, four copies of the data; here, one `if` on a
parameter. Worth noting the two side by side so "share the implementation" is
not read as a rule in either direction.

**Histogram subtraction** is the single largest optimisation in the algorithm: a
child's histogram is its parent's minus its sibling's, so computing the
**smaller** child directly and subtracting for the larger roughly halves the
work per level.

**Its gate cannot be bit-identity, and this paragraph said it could.** The
first draft of this section asserted that a subtracted histogram must be
*bit-identical* to one built directly from the child's index set. That is false
and the arithmetic says so: `tree_hist` accumulates in index order, so the
parent sums rows `0..n-1` while the sibling sums a subsequence, and
`sum(A ∪ B) − sum(A) = sum(B)` does **not** hold exactly for floating point.
Writing the gate down before writing the code is what exposed it.

So the gate splits by channel, which is more informative than one tolerance
would be:

- the **count** channel must be bit-identical — counts are integers held in
  doubles, and every one of them is exactly representable, so any difference
  there is a real bug and not rounding;
- the **gradient and hessian** channels are compared with a tight relative
  tolerance, and the test also asserts the *maximum* observed difference so a
  drift from 1e-16 to 1e-9 shows up as a number rather than as a still-passing
  assertion.

This is what LightGBM and XGBoost both accept, and the error does not
accumulate across levels, because each child's histogram is derived from its
parent's rather than from a running total.

### 8.7 Early stopping, and what it must not see

A validation fold held out of training, scored every round, stopping after
`n_iter_no_change` rounds without a `tol`-sized improvement.

The failure to design against is the obvious one: **the validation fold must be
split before binning**, or the bin edges are chosen using data the model is
then scored on. That is a quieter leak than training on it, and it inflates the
stopping round rather than the score, so it would not show up as an implausible
accuracy. `train_test_split` from `selection.b` does the split; `tree_bin` sees
only the training part.

Reproducibility is [C4]: the split comes from `nd_seed` alone, and no estimator
carries RNG state.

### 8.8 Forests: bootstrap and random thresholds

`RandomForest*` and `ExtraTrees*` are the same loop with two switches —
bootstrap sampling on or off, best split or random split — over `n_estimators`
independent trees. Both reduce to [§8.5](#85-k8-needs-no-new-kernels-and-that-is-the-finding).

Two things to get right:

- **Averaging happens in probability space**, not in label space. A forest of
  classifiers averages `predict_proba` and then takes the argmax; a majority
  vote over hard labels throws away every tree's confidence and is a different
  (worse) estimator.
- **Each tree needs its own index sample from one seeded stream**, so the forest
  is reproducible from `nd_seed` alone and *not* dependent on how many trees
  were fitted before it. The test for this is that fitting 10 trees and then
  fitting 20 gives the same first 10.

---

### 9.1 Missing values in the GBDT

`NaN` goes to a dedicated bin, and each split learns a default direction (the
side with the higher gain) which is also used for unseen `NaN` at predict time.
This is XGBoost's sparsity-aware behaviour and it matters: imputing to the mean
first is both lossy and a different model.

---

## 10. K9 — `decompose.b` and `cluster.b`

Written before the code, as every section here is. What this phase mostly needs
is **decisions about numerical conditioning and determinism**, not kernels:
numba already exposes `svd`, `eigh`, `qr`, `solve`, `pinv` and `cholesky`.

### 10.1 PCA goes through the SVD of the centred matrix, never the covariance

The textbook route is to form `Xᵀ X / (n−1)` and take its eigendecomposition.
This package refuses that for the same reason §6 refuses it for linear models:
**the Gram matrix squares the condition number.** A design matrix with
condition number 10⁸ — ordinary for unscaled real data — yields a covariance
with 10¹⁶, which is the whole of double precision. The singular values of `X`
are the square roots of the eigenvalues of `XᵀX`, so SVD gets the same answer at
half the condition number, and the two agree to machine precision only when the
problem was easy to begin with.

The cost is that SVD on `[n, f]` is more expensive than `eigh` on `[f, f]` when
`n ≫ f`. That is accepted and stated rather than quietly traded: a wrong
component at 10⁻³ is worse than a slow one.

### 10.2 Singular vectors have arbitrary signs, and that must be pinned

`U S Vᵀ` is unchanged if a column of `U` and the matching row of `Vᵀ` are both
negated, so a bare SVD gives components whose **sign is an artefact of the
LAPACK driver**. Two runs on the same data can disagree; so can one run on data
differing in the last bit. That is not acceptable from an estimator whose
output is read by eye and compared across runs.

So each component's sign is fixed deterministically, by scikit-learn's rule:
make the entry of **largest absolute value** in each left singular vector
positive, and flip the matching component with it. A tie is broken by the lowest
index, as [K62] does for splits, so the rule is total.

**The gate is that it survives a sign flip injected on purpose**: negate a
column of `U` and the matching row of `Vᵀ`, reconstruct, and the fitted
components must come back identical. Without `svd_flip` the test is red; without
a tie-break the test is flaky, which is worse.

### 10.3 `n_components` as a float means "enough variance", and that is a fit-time quantity

`n_components = 0.95` is the useful form, and it cannot be resolved before
seeing the data. So the *requested* value and the *resolved* count are different
fields, and `n_components_` is only available after a fit — the same split
[K6] makes for `ColumnTransformer`'s learned output widths.

### 10.4 `TruncatedSVD` does not centre, and that is its entire reason to exist

Centring a sparse matrix destroys its sparsity. `TruncatedSVD` therefore
factorises `X` itself rather than `X − x̄`, which makes it a different estimator
from PCA rather than a faster one: on centred data they agree exactly, and on
uncentred data the first component of `TruncatedSVD` points at the mean. The
test asserts **both** — agreement after centring, and disagreement before —
because only the second says the centring step is really absent.

### 10.5 k-means: one GEMM per iteration, k-means++ seeded, and empty clusters relocated

`‖x − c‖² = ‖x‖² − 2 x·c + ‖c‖²`, so a whole assignment step is **one matmul**
plus two broadcasts. That is what makes k-means viable in Bantu at all: the
per-element work is native, and the loop body is a handful of `nd_*` calls per
iteration rather than per point.

The `‖x‖²` term is constant across iterations and is computed once. It can be
dropped entirely from the `argmin` — it does not depend on `c` — and it is kept
only when the inertia itself is wanted, which is once per run rather than once
per iteration.

**k-means++ init, from `nd_seed` alone** ([C4]): the first centre uniformly at
random, each subsequent one with probability proportional to `D(x)²`. Written
with the same cumulative-sum-and-search construction the rest of the package
uses for weighted draws, so nothing carries RNG state.

**An empty cluster is relocated to the farthest point**, not dropped. Dropping
it silently returns fewer clusters than asked for, which is a wrong answer that
looks like a right one — the caller asked for `k` and a later `predict` would
index a centre that does not exist. Relocation is what scikit-learn does and it
keeps `k` an invariant rather than a request.

`n_init` restarts and keeps the lowest **inertia**, which is the objective
Lloyd's algorithm actually descends; keeping the best by anything else would be
choosing a run by a criterion it was not optimising.

### 10.6 DBSCAN's constraint is memory, not speed

A full pairwise distance matrix for `n = 10,000` is 800 MB, so the neighbour
search is **chunked**: one GEMM per chunk of rows, thresholded to a boolean, and
the neighbour lists accumulated. The per-element work stays native and the peak
memory is the chunk, not the matrix. The chunk size is a parameter with a
documented default rather than a constant, because the right value depends on
`f` as well as `n`.

Expansion is **iterative over an explicit stack**. Bantu's call depth is capped
at 1000 and a cluster can easily be longer than that, so a recursive flood fill
would fail on exactly the inputs DBSCAN is for — the same reason [§8] grows
trees over a queue.

Border points are assigned to the **first** core point that reaches them, which
is order-dependent and is documented as such. scikit-learn has the same
property; stating it is the difference between a documented behaviour and a
surprise.

### 10.7 Gaussian mixtures need the Cholesky, not the inverse

EM wants `(x − μ)ᵀ Σ⁻¹ (x − μ)` and `log det Σ`. Both come from the Cholesky
factor without ever forming `Σ⁻¹`: the quadratic form is the squared norm of a
triangular solve, and the log-determinant is twice the sum of the log diagonal.
Forming the inverse costs accuracy for no gain, and is the same mistake as
forming the Gram matrix one section up.

`reg_covar` is added to the diagonal before factorising, and is **not** optional:
a component that collapses onto a single point has a singular covariance and a
likelihood that diverges to `+∞`, which EM will happily walk towards. That is
the mixture-model equivalent of [K45]'s `+inf` gain — an unbounded objective
reached by a degenerate solution — and the guard is the same shape.

The responsibilities go through **log-sum-exp with the row maximum subtracted**,
not through a ratio of exponentials. `exp(−800)` is zero and `exp(800)` is
infinity, and both occur on ordinary data once the dimension is more than a
handful.

---

## 11. Persistence

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

## 12. Gates

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

### 10.1 The accuracy gates

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

## 13. What is deliberately out

| item | why |
|---|---|
| neural networks (`MLPClassifier`) | `btorch` does this better; a second implementation would diverge |
| `t-SNE`, `UMAP` | visualisation, not modelling; `bplot`'s territory if ever |
| text vectorizers | `btext` owns them ([E12]), conforming to this protocol |
| GPU / distributed | out of scope for the whole ecosystem |
| `partial_fit` / online learning | a different protocol; revisit only on demand |
| calibration, isotonic regression | small and useful; after K12 if asked for |

---

## 14. Open items

| # | item | settled by |
|---|---|---|
| 1 | leaf-wise vs depth-wise as the default | measured on the adult gate; leaf-wise expected to win, must be shown |
| 2 | is binning reusable outside trees (own `hist_` family)? | if `bsci.stats` histograms want it in stage 6 |
| 3 | does `tree_hist` want stage 1's threading? | yes in principle; depends on D3 landing first, and the partition is over samples so it is race-free by construction |
| 4 | SVM: SMO or a linear-only approximation? | SMO is a week on its own; linear SVM via the existing solver may be enough for K10 |
