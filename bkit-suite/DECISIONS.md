# bkit — Decisions

Numbered so they can be cited from code comments, tests and the CHANGELOG.
A decision is recorded here when a different choice was genuinely available.
Ecosystem-wide decisions are `E*` in
[`../docs/ecosystem-architecture.md`](../docs/ecosystem-architecture.md);
btorch's are `B*`; numba's are `N*`.

---

### K1 — The binned matrix is an opaque native handle, not an `NdArray`
**Found while writing the architecture doc, and it would have been a late
surprise.** The binned feature matrix wants one byte per cell. numba's `DType`
is `{F64, I64, BOOL}` — there is no `u8`, and `BOOL` is the only one-byte
variant. Holding bins as `I64` costs 8 bytes a cell: **800 MB for 1M × 100**,
which defeats the entire purpose of binning.

Chosen: an opaque handle owning a `uint8_t` buffer, the same shape as the
existing `col_*` and `bp_*` handles. The binned matrix is internal — users
never index it, slice it or do arithmetic on it; four kernels need
`get(row, feature)` and nothing else.

**Rejected:** adding `U8` to numba in ecosystem stage 1. That stage is already
scoped (f32, threading, GEMM), and `U8` would need the full ufunc, reduction
and linalg surface built for a dtype nobody does arithmetic in — scope creep
to serve a package three stages later.

**Rejected:** `I64` bins, for the 8× memory above.

**Forward path:** if stage 6's `q8_` work adds a real `I8`/`U8` dtype for
quantization, the handle can migrate behind the same API. Not a dead end.

---

### K2 — `clone()` is mandatory, and cross-validation must use it
Cross-validation refits on every fold. Reusing a fitted estimator across folds
leaks the test fold into training and still produces a **plausible, merely
inflated** score — the same class of failure as btorch's B10, where a falling
loss proved nothing.

So `clone()` is part of the protocol, `cross_val_score` calls it before every
fit, and a test asserts the **source estimator's coefficients are unchanged
after cross-validation**, which is the observable signature of the bug.

---

### K3 — `clone()` is written per estimator, because Bantu has no reflective constructor
A base-class `clone` would need to construct a fresh instance of the
*subclass*. Bantu offers no way to do that, so each estimator writes
`def clone() { return new Ridge(this.params_.alpha, ...); }` — three lines of
boilerplate that cannot silently share fitted state.

Related: **`super.method()` does not work** in Bantu. It lexes, parses, then
fails with `Undefined variable: super` (btorch A4, still deferred); only
`super($args)` for the constructor works. So `base.b` shares code through
`this`, never through children calling up. `fit_transform` is written in terms
of `this.fit`/`this.transform` and inherits cleanly. A design where
`StandardScaler.fit` called `super.fit()` would not have worked at all, and
finding that out mid-implementation would have cost a redesign.

---

### K4 — numba arrays are the core; `arctic` is an adapter at two points only
Core `bkit` works on dense `f64` numba arrays with `NaN` as the missing marker.
`arctic` is a declared dependency used by exactly two things:
`ColumnTransformer` (selects columns by name and type) and a `from_frame`
adapter (categorical → ordinal codes, column ordering, dtype coercion).

**Rejected:** a DataFrame-native package. A user who already has arrays would
pay for a column store they do not use, and **every algorithm would carry two
code paths** — the condition btorch avoided in B18 by composing convolution out
of im2col and matmul rather than writing a second path.

---

### K5 — Histogram-based, leaf-wise gradient boosting
LightGBM's shape rather than XGBoost's exact-split, depth-wise one.

Histogram-based because split finding becomes O(bins) instead of O(sorted
samples), and because **histogram subtraction** — a child's histogram is its
parent's minus its sibling's — roughly halves the work per level. That is the
single largest optimisation in the algorithm and it is unavailable to an
exact-split implementation.

Leaf-wise with a `max_leaves` cap, plus `max_depth` as a separate guard, since
leaf-wise growth on noisy data will happily build a 60-deep chain.

**Open:** leaf-wise vs depth-wise as the *default* is settled by measurement on
the adult gate. Leaf-wise is expected to win and must be shown to.

---

### K6 — `NaN` gets its own bin and each split learns a default direction
XGBoost's sparsity-aware behaviour. The alternative — impute to the mean first
— is both lossy and **a different model**, and it hides from the user that
anything happened.

At predict time, a `NaN` in a feature whose training data had none follows the
learned default direction of that split rather than raising, because a model
that refuses to score real-world rows is not useful. This is a documented
behaviour, not a silent one.

---

### K7 — `base.b` holds the protocol and nothing else
So `btext` can depend on it without pulling in gradient boosting, SVMs and
cross-validation ([E12]). A protocol defined inside a large module becomes a
reason not to conform to it, and the whole value of [C1] is that eleven
implementers agree.

---

### K8 — GBDT lives in this package, not its own
Cites [E3]. In Python, `xgboost` is separate from `scikit-learn` because they
are separate *organizations* — not for architectural reasons. Metrics,
preprocessing and cross-validation are exactly what wraps a GBDT in practice,
so splitting forces co-installation anyway, and the ecosystem rule is: if A is
always installed with B, they are one package.

---

### K9 — Tier 3 is a protocol test, taking gradcheck's slot
There is no autograd here, so btorch's gradcheck tier has nothing to check. Its
slot in the six tiers is taken by a **table-driven protocol test** over a
registry of every estimator in the package.

This exists because a protocol with eleven implementers fragments silently: an
estimator that forgets `clone()`, or whose `get_params()` omits a parameter
that `set_params()` accepts, breaks `GridSearchCV` in a way that surfaces as a
confusing score rather than an error. Enumerating the implementers and
asserting the contract on each catches it at the point of writing.

---

### K10 — The hessian is clamped, and a test drives it to saturation
`G²/(H+λ)` with a saturated logistic gives `H → 0`; with `λ = 0` that divides
by zero and produces a `NaN` gain which **wins every split comparison**, since
`NaN > x` is false but so is `NaN < x`, and a naive max-tracking loop keeps
whatever it saw last.

So `H` is clamped at `1e-16`, and a test deliberately drives the model to
saturation rather than hoping ordinary data never gets there.

---

### K11 — Four scalers share one `transform`, because they differ only in two numbers
`StandardScaler`, `MinMaxScaler`, `RobustScaler` and `MaxAbsScaler` are the same
transformation with different statistics: subtract a centre, divide by a scale.
So `_ScalerBase` holds `transform`, `inverse_transform`, the feature-count
guard and the `NaN` refusal, and each subclass overrides `fit` alone — nine
lines each.

`MinMaxScaler` is the one that had to be *rewritten* to fit the form. Its usual
expression is two steps (`X_std = (X - min)/(max - min)`, then
`X_std*(hi - lo) + lo`), which looks like a different transformation. It is not:

```
scale  = (max - min) / (hi - lo)
center = min - lo * scale
```

makes `(x - center)/scale` identical to the two-step formula, so it shares
everything. Forcing it into the shared form was worth doing: the round-trip gate
now tests **one** implementation four times rather than four implementations
once.

This shape works only because of [K3]: `super.method()` does not exist in Bantu,
but a child never needs it here — the shared methods read `this.center_` and
`this.scale_`, which the child set. A design where `StandardScaler.transform`
called `super.transform()` would not have run at all.

**Rejected:** four independent implementations. The round-trip property is the
one thing a scaler can get silently wrong, and four copies is four chances.

---

### K12 — The feature-count guard is load-bearing, because broadcasting makes a mismatch *silent*
`transform` on a matrix with a different number of columns than `fit` saw is a
user error that numba **cannot** catch: an `(n, 1)` input against a stored
`(1, d)` statistic broadcasts and returns a perfectly well-formed `(n, d)`
matrix of nonsense. No shape error, no `NaN`, no warning — just wrong numbers of
the right shape, which then flow into a model and score plausibly.

Every transformer therefore records `n_features_` at `fit` and checks it in
`transform` and `inverse_transform`, and the check lives in `_ScalerBase` so it
cannot be forgotten by one of four. The error states both counts, because the
caller's next question is always "how many did it expect".

This is the only guard in `preprocess.b` with no natural backstop anywhere
downstream, which is why it is a decision rather than a detail.

---

### K13 — Encoders map through `searchsorted`, not a dict
The obvious encoder keeps a Bantu dict from category to code and loops over
rows. That is O(n) *interpreter* steps per column, and the interpreter costs
about a microsecond an element.

Chosen: `nd_unique` gives the sorted categories and `nd_searchsorted` maps a
whole column to codes in one native O(n log k) call.

The part that makes this more than an optimisation is that **the
unseen-category test falls out of the same call**. Clamp the returned positions
into range, `nd_take` the category each one landed on, and compare with the
input; a mismatch *is* an unseen value:

```bantu
$pos  = nd_searchsorted($cats, $col, null);
$safe = nd_clip($pos, 0, len($cats) - 1, null);
$hit  = nd_equal(nd_take($cats, $safe, 0), $col, null);   // false => unseen
```

`handle_unknown` then chooses between raising (naming the column and the value)
and mapping to a documented slot, and both branches are those same two kernels.
A dict-based encoder needs a second pass, or a per-row branch, to answer the
same question.

`OneHotEncoder` reuses it and builds each indicator block by broadcasting the
codes against a row of `0..k-1` — one comparison kernel per feature, no scatter
and no row loop.

---

### K14 — The scalers refuse `NaN` rather than computing a statistic from a smaller sample
scikit-learn's scalers compute `NaN`-aware statistics and leave the `NaN` in
place. `bkit`'s refuse, naming the column and pointing at `SimpleImputer`.

The reason is concrete, not philosophical: numba has `nd_nanmean`, `nd_nansum`,
`nd_nanmin` and `nd_nanmax`, and **no** `nanvar`, `nanstd`, `nanmedian` or
`nanquantile`. `StandardScaler` needs `nanstd`; `RobustScaler` needs
`nanmedian` and `nanquantile`. Supporting `NaN` in `MinMaxScaler` and
`MaxAbsScaler` — which only need the existing kernels — while `RobustScaler`
quietly produced `NaN` statistics would be worse than refusing in all four: a
surface that looks uniform and is not.

One rule, one exception, nothing else to remember: everything in
`preprocess.b` refuses `NaN`; `SimpleImputer` accepts it, because filling it is
its job. The refusal is in `fit`, `transform` **and** `inverse_transform`, so
there is no third case either — a scaler fitted on clean data and handed dirty
data would otherwise emit a `NaN` column silently.

**Forward path:** `nd_nanvar`, `nd_nanstd` and `nd_nanquantile` are a small
addition to numba's reduction surface; the Welford and selection kernels exist
and already thread. When they land this becomes a one-line change in
`_ScalerBase`, and the decision should be revisited rather than kept out of
habit.

---

### K15 — A zero-variance column gets scale 1, and that keeps the round trip exact
A constant column has standard deviation, range, IQR and max-abs all zero.
Dividing by it gives `NaN` for every row — silently, because `NaN` is a
perfectly good `f64` and nothing downstream objects to one.

Chosen: substitute 1 for a zero scale, which is scikit-learn's handling, and
which has a property worth stating rather than discovering. The column
transforms to exactly `x - center` = **0**, and `inverse_transform` adds
`center` back, so `inverse_transform(transform(x)) == x` holds **exactly** on a
constant column too — the gate does not need an exemption for it.

**Rejected:** raising on a constant column. Defensible, and it is what a
careful human would want told to them, but a single constant feature in a
hundred-column matrix is common and legitimate (a flag that happens to be
constant in this sample), and failing the whole fit for it would push users to
drop the guard rather than fix the data. A `NaN` was never an option.

---

### K16 — The mode is a run-length on a sorted column, not a count per unique value
`SimpleImputer(strategy="most_frequent")` needs the most common value per
column. Counting each unique value is O(k·n) with a Bantu loop per unique, and
k is data-dependent — a high-cardinality column makes it quadratic in practice.

Chosen: sort the column, then find the longest run entirely in kernels. The run
boundaries are `nd_nonzero(nd_diff(sorted) != 0)`, the run lengths are `nd_diff`
of those boundaries, and `nd_argmax` picks the longest. O(n log n) in native
code, with the only Bantu loop running over *columns*.

Sorting also solves the `NaN` problem for free, which is why `median` uses the
same route: `nd_sort` puts `NaN` **last** (verified against the kernel, not
assumed), so the first `c` elements of the sorted column are exactly its
non-`NaN` values in order, and both statistics are indices into that prefix.
There is no `nanmedian` kernel to call ([K14]), and this does not need one.

Ties go to the **smaller** value, because the column is ascending and
`nd_argmax` returns the first maximum — the same rule scikit-learn inherits
from `scipy.stats.mode`. A test asserts it, because a tie-break that drifts is
a silently different imputation.

---

### K17 — Tier 3 is its own file, not a section of `base.b`'s test
K1's roadmap said every new estimator joins the protocol registry in
`tests/bkit_base_test.b`. Writing K2 showed why it cannot: the registry has to
name estimators from every module, so wherever it lives, **that file includes
all of `bkit`**.

`base.b` is deliberately tiny because `btext` depends on it alone ([K7]).
Making its test include eleven modules would mean a bug in `ensemble.b` fails
`base.b`'s test — exactly the coupling `base.b` exists to avoid.

Chosen: `tests/bkit_protocol_test.b` carries the package-wide registry and is
tier 3 for every phase from K2 on. `tests/bkit_base_test.b` keeps its three
deliberate doubles, including the one that forgets `clone()`, so the file that
proves the *mechanism* stays independent of the files that use it.

Each registry entry also brings **its own X and y**. One shared matrix cannot
serve the package: `LabelEncoder` takes a single label column, the encoders need
repeated categorical values rather than forty distinct floats, and a regressor
needs a `y`. Sharing one matrix would have meant either skipping the estimators
it did not suit — the ones most likely to break the contract — or giving them a
fit that does not represent them.

---

### K18 — A permutation is `argsort` of random keys, not Fisher–Yates
```bantu
nd_argsort(nd_random_uniform([$n], 0.0, 1.0), null)
```
Two kernel calls. Fisher–Yates in Bantu is `n` interpreter iterations and `n`
separate RNG draws, which for a million-row shuffle is the difference between
milliseconds and minutes — and the C-style `for` silently stops at 100,000
iterations, so the obvious implementation would also be *wrong* above that size
without saying so.

It delegates to the one RNG ([C4]) without touching it: `nd_random_uniform`
draws from `nd_seed`'s stream, so the same seed gives the same permutation and
nothing in `bkit` holds RNG state.

**The honest caveat:** `argsort` of *distinct* keys is a uniform permutation.
Two equal `f64` keys are broken by index. At n = 1,000,000 the chance of any
collision at all is about 1 in 10,000, and a collision perturbs one adjacent
pair — so the bias exists and is negligible, which is worth stating rather than
implying that the method is exact.

---

### K19 — Splitters return index arrays, never data
`split` yields `[train_idx, test_idx]` as `i64` arrays, not `X_train`/`X_test`.

Two of the reasons are obvious — a 5-fold split of a 1M × 100 matrix copies
nothing, and `cross_val_score` can take rows once per fold. The third is the one
that decided it: **the invariants become checkable.** "No group appears in two
folds" and "every index is tested exactly once" are statements about indices.
Given only `X_train` matrices you would have to search for rows to verify
either, and `splitter_check` could not exist in the form it does.

---

### K20 — A splitter is not an estimator, and gets its own protocol
`KFold` has nothing to fit and nothing to predict. Putting it through the
estimator protocol would mean `fit`, `predict`, `transform`, `score` and
`clone` all raising on every splitter, and a `protocol_check` told to skip five
of its seven assertions.

So splitters have three methods — `get_n_splits()`, `split($X, $y, $groups)`,
`partitions()` — and their own `splitter_check`, which lives in `selection.b`
for the reason `protocol_check` lives in `base.b` ([K17]): one definition of the
contract, inherited by whatever is added later.

`partitions()` is on the protocol rather than assumed by the checker because the
invariant genuinely differs. `KFold`, `StratifiedKFold` and `GroupKFold`
partition the data exactly; `TimeSeriesSplit` does **not** — the earliest rows
are never tested, which is the point of it. A checker that assumed coverage
would have to exempt `TimeSeriesSplit` by name, and the next splitter added
would be exempted by whoever added it.

---

### K21 — No `random_state` on a splitter, and the requirement that follows
scikit-learn's splitters take a `random_state`. `bkit`'s do not. Reproducibility
comes from seeding **once**, with `nd_seed`, which is already how `numba` and
`btorch` work ([C4]).

**Rejected:** a per-splitter seed. To take effect it would have to call
`nd_seed`, which clobbers the global stream for everything downstream — a model
drawing its initial weights after a grid search would get a stream chosen by the
splitter. That is exactly the "each package seeds separately" failure C4 exists
to prevent, and it would arrive as an irreproducible model rather than as an
error.

The consequence is not a limitation, it is a requirement that was needed anyway:
**`cross_val_score` and the searches materialise their splits once and reuse
them for every candidate.** Re-splitting per candidate with a shuffled splitter
compares candidates on *different folds*, so the winner is partly a draw of the
RNG. With the folds fixed, a difference in score is a difference in the
parameters.

That property is also why the test for it needs a *shuffled* splitter. With an
unshuffled one it is unfalsifiable — two calls to `split()` return the same
folds — and a mutation that re-split per candidate passed the first version of
the test for exactly that reason.

---

### K22 — The fold-size rule is written down, and stratification rotates it
With `n` rows and `k` folds, the first `n mod k` folds get `floor(n/k) + 1` rows
and the rest get `floor(n/k)`. This is scikit-learn's rule, recorded because
"balanced folds" has several reasonable definitions and a different one silently
changes every cross-validated score a user compares against a Python baseline.

`StratifiedKFold` applies the same rule **per class**, and **rotates the
starting fold from one class to the next**. Without the rotation every class puts
its extra sample in fold 0: with three classes each having one extra, fold 0 ends
up three rows larger than the last. Each class is still balanced to ±1 either
way — so a test of class ratios alone would pass — but the folds are not, and a
fold 10% larger than its neighbour makes the per-fold scores incomparable.

The rotation is one accumulator, `rot = (rot + m_c mod k) mod k`, and it is the
difference between "stratified" and "stratified and balanced".

---

### K23 — `train_test_split` returns a named dict
```bantu
$s = sel.train_test_split($X, $y, 0.25, true, null);
$s.X_train   $s.X_test   $s.y_train   $s.y_test   $s.train_index   $s.test_index
```
scikit-learn's `X_train, X_test, y_train, y_test = train_test_split(...)`
interleaves the two variables rather than grouping them, and swapping two of
them yields a working program that trains on its test set. Bantu has no tuple
unpacking at all, so the positional form would have to be read as `$r[2]` and
`$r[3]`, which is worse still. Named fields cost nothing and cannot be
transposed.

The indices come back too, because a caller who wants to split a third array
the same way — sample weights, group labels — should not have to re-derive the
split, and re-deriving it with a shuffled splitter would give a *different* one.

---

### K24 — A mistyped parameter is an error
Found while designing K3, fixed in `base.b`: `set_params` accepted any key and
added it to `params_`. So a grid over a mistyped key — `with_men` for
`with_mean` — ran every candidate with identical settings, scored them
identically, and reported a "best" configuration it had never tried. The whole
protocol exists to prevent plausible wrong numbers, and this produced one.

`set_params` now refuses a key `get_params()` does not already hold, and names
the ones it takes. Presence is tested with `keys()` rather than `== null`,
because a parameter may legitimately *hold* null — `OrdinalEncoder`'s
`unknown_value` does — so a null read cannot tell "absent" from "set to null".

`GridSearchCV` validates the whole grid against a clone of the estimator
**before** the first fit, so a typo costs a message rather than a few minutes of
fitting. For a `Pipeline` that validation has to go through the estimator
anyway, since only the pipeline knows whether `reg__shift` names a real step.

---

### K25 — `ds.accuracy` moves to `bkit.metrics`, but btorch keeps its implementation
[C5] and [E14] say the metric moves with a **delegating alias** left in
`btorch`. The move happens; the delegation does not.

The reason is [E2]'s own argument, applied to itself. `btorch` is **1.0.0 and
gated on MNIST accuracy**; `bkit` is `0.3.0` and changes every phase.
Delegating would make the stable package depend on the churning one, so a
`bkit` release could break a `btorch` install — *"blast radius becomes release
cadence"*, which is the reason these are separate packages at all. A conditional
dependency (delegate if `bkit` is installed) was the third option and is worse:
two code paths for one metric, which is exactly what [B18] avoided in btorch by
composing convolution out of `im2col` and `matmul` rather than writing a second
path.

So `bkit.metrics.accuracy_score($y_true, $y_pred)` is the home, and
`btorch.datasets.accuracy($logits, $labels)` keeps its eight lines, marked
deprecated in the source and the CHANGELOG. They are not the same function —
btorch's takes logits and argmaxes them — so a **differential test** in
`tests/bkit_metrics_test.b` asserts they agree on the same inputs, over a hand
case and over 200 random rows. That test, not the delegation, is what stops them
drifting.

**Also rejected:** having `base.b`'s `score()` delegate to `metrics.b`. The
dependency would run the wrong way — `base.b` is the file everything else
depends on, and [K7] keeps it free of everything.

---

### K26 — Every classification metric is a view of one confusion matrix
`precision_recall_fscore_support` builds the matrix once and reads everything
off it: `tp` is the diagonal, `fp` the column sum minus it, `fn` the row sum
minus it, `support` the row sum. `precision_score`, `recall_score`, `f1_score`
and `classification_report` all call that function rather than recomputing.

The payoff is not the arithmetic saved, it is that **the zero-division rule
lives in one place** ([K27]). With four independent implementations, a class
that never appears could be handled one way by `precision` and another by `f1`,
and the macro average would disagree with the per-class array it was supposedly
averaging.

The matrix itself is built without a loop over samples: encode both label
arrays to codes, form `true_code · k + pred_code`, and `nd_bincount(..., k·k)`
reshaped to `(k, k)`. Three kernel calls for any number of rows — 200,000 rows
over 10 classes in 49 ms.

**The orientation is part of the contract:** rows are true, columns predicted.
Transposing it silently swaps precision and recall, so the test asserts cell
`[0][1]` is "true 0, predicted 1" rather than only checking the row sums.

---

### K27 — A zero-support class scores 0, not `NaN`
A class present in `y_true` but never predicted has `tp + fp = 0`, so precision
is `0/0`. A class never present but sometimes predicted has `tp + fn = 0`, so
recall is. `f1` is `0/0` whenever `precision + recall = 0`.

All three yield **0**, which is scikit-learn's `zero_division=0` default.

**Rejected: `NaN`.** It propagates into the macro average, so one absent class
turns the whole model's score into `NaN` — and `NaN` is a perfectly good `f64`
that nothing downstream objects to. That is the exact silent-wrong-answer shape
this package is built against.

**Rejected: raising.** Too strict. A class missing from one cross-validation
fold is normal, and a metric that refused to score such a fold would make
`cross_val_score` fail on ordinary data.

The test covers both directions — a class in `y_true` and never predicted, and
a class predicted but never true — and asserts the macro average is a finite
number computed *from* the zeros rather than around them.

---

### K28 — The curves emit one point per *distinct* threshold, and AP is not a trapezoid
**Tie handling.** With scores `[0.8, 0.8, 0.8]` and labels `[1, 0, 1]`, a curve
that advanced one sample at a time would trace a staircase through points no
threshold can actually produce — and the area under that staircase depends on
the order the equal scores happened to arrive in. **The AUC would change when
the input was shuffled.** So a point is emitted only at the end of each run of
equal scores, found with `nd_nonzero(nd_diff(sorted) != 0)` plus the last index
— the same construction the mode uses in [K16]. A test asserts the AUC is
unchanged when two tied samples swap places, which is the property the handling
actually buys.

**Two different rules on purpose.** `roc_auc_score` is the **trapezoid** over
`(fpr, tpr)`. `average_precision_score` is the **step sum**
`Σ (R_n − R_{n−1}) · P_n`. These give different numbers on the same data —
0.625 and 0.583 on the test's hand case — and scikit-learn uses exactly this
pair. The trapezoid is optimistic on a curve that steps *down*, which a PR
curve does, which is why AP does not use it. The test asserts both values and
asserts they differ, so neither can drift into the other's rule unnoticed.

The ROC AUC is cross-checked against the **Mann–Whitney statistic** — counting
the four positive/negative pairs by hand, a tie counting ½ — because that is a
completely different route to the same number and so cannot share a bug with
the curve code.

---

### K29 — `classification_report` returns a dict; the text is a separate function
A metric whose only output is a formatted string can only be tested by matching
that string, and then every change to a column width is a test failure — which
trains whoever hits it to update the expected string rather than to check the
numbers.

So `classification_report` returns the numbers, in scikit-learn's
`output_dict=True` shape (one entry per class plus `accuracy`, `macro avg` and
`weighted avg`), and `classification_report_text` formats them. The numbers are
asserted numerically; the formatter is asserted only to mention every class, to
format to three decimals, and to produce the right number of lines.

The formatter rounds explicitly rather than using `str()`, because `str()` gives
six significant digits — the first of the three rules carried from btorch.

---

### K30 — `r2_score` raises on a constant target; MAPE clamps instead
`R² = 1 − SSE/SST`, and a constant target has `SST = 0`: the metric's own
denominator is gone. scikit-learn returns `0.0` with a warning.

`bkit` **raises**, for two reasons. One, `base.b`'s `score()` already did so
before this file existed ([K1]'s test asserts it), and one package should not
disagree with itself about the same formula. Two, `0.0` makes a *meaningless*
comparison look like a *bad* one — a reader sees a number and concludes the
model is useless, when in fact R² had nothing to measure. The error names
`mean_squared_error` as the metric for that case.

`mean_absolute_percentage_error` is the mirror image and is handled the other
way round: a zero in `y_true` makes one term infinite, and the denominator is
**clamped** to `eps` rather than raising. The asymmetry is deliberate — a
constant target breaks R² for *every* sample, while a single zero among
thousands of nonzero targets breaks one term of a mean. Refusing the whole
metric for that would be out of proportion. The consequence is that the result
is bounded rather than exact at a zero target, which is stated at the function
because a bounded-but-wrong number is only safe if the reader knows.

---

### K31 — The Gram matrix is never formed, and the measurement is the argument
The roadmap asked for *"normal equations via numba's `lstsq`; `solve` for
well-conditioned"*. The second half is wrong.

A degree-9 polynomial design on 60 points — an ordinary fit, not a contrived
one:

| | value |
|---|---|
| condition number of `X` | 2.6 × 10⁶ |
| condition number of `XᵀX` | **6.9 × 10¹²** |
| max coefficient error, `lstsq` | 1.2 × 10⁻⁹ |
| max coefficient error, normal equations | **2.8 × 10⁻³** |

Forming `XᵀX` **squares the condition number**, and the normal equations came
out *a million times* less accurate — a 0.3% error in coefficients, with
nothing in the output to show it. Against that, `lstsq` cost 41 ms versus 7 ms
on a 5000 × 60 problem, once per fit, and on a well-conditioned problem the two
agree to 2.4 × 10⁻¹⁶.

**Rejected: making it conditional.** "Use `solve` when well-conditioned"
requires knowing the condition number, which costs an SVD — more than the
`lstsq` it was avoiding. And the caller cannot be asked to know: the whole
hazard is that an ill-conditioned design looks exactly like a well-conditioned
one until the answers are wrong.

The test re-derives this rather than trusting it: it asserts both that `lstsq`
stays under 1e-6 **and** that the Gram route exceeds 1e-4 on the same data, so
the decision is checked on every run instead of remembered.

---

### K32 — `Ridge` is least squares on an augmented system
Ridge's closed form, `(XᵀX + αI)b = Xᵀy`, needs the matrix [K31] just rejected.
It is not needed:

```
minimise ‖Xb − y‖² + α‖Pb‖²   ==   minimise ‖ [    X     ] b − [ y ] ‖²
                                              [ √α · P   ]     [ 0 ]
```

Stack `√α` rows under the design and zeros under the target, then `lstsq`.
Three things fall out:

* the Gram matrix is never formed, so Ridge inherits [K31]'s accuracy;
* `P` selects the penalised columns, so **the intercept is excluded by
  construction** rather than by remembering to exclude it;
* it explains what the penalty does numerically — the added rows raise the
  smallest singular value to at least `√α`, so a rank-deficient design becomes
  full rank. Ridge is *better* conditioned than the fit it regularises, which
  is why a closed form is safe here and not in [K31].

The test checks it against the textbook `(XᵀX + αP)` solve on a small
well-conditioned matrix, where that form is trustworthy — two independent
routes to the same coefficients, agreeing to 1e-9.

---

### K33 — The iteration cap is audible, on stderr, and does not raise
Coordinate descent that stops at `max_iter` without converging returns
coefficients that are **wrong**, not approximate — and every number computed
from them afterwards is wrong too, with nothing in the output to say so.

So: `converged_` and `n_iter_` are on the estimator and asserted by a test, and
hitting the cap prints a warning to **stderr** via `eprint`. Stderr, not
stdout, because a library's diagnostic that a pipeline's output can swallow is
a diagnostic nobody reads — the same argument that fixed `--quiet` for the
`[class]` lines.

**Rejected: raising.** Hitting `max_iter` on hard data is normal. A user who
wants more iterations should get a warning and a number, not an exception in
the middle of a cross-validation that then reports nothing at all.

**Rejected: silence plus a flag.** The flag is right and insufficient: nobody
checks a flag they do not know exists. The warning is what makes the flag
discoverable.

The warning names what to change — raise `max_iter`, loosen `tol`, or scale `X`
— because coordinate descent converges far faster on columns of comparable
magnitude, and "scale your features" is the actual fix nine times in ten.

---

### K34 — `LogisticRegression`: binary IRLS, multiclass one-vs-rest
Binary uses **IRLS** — Newton on the log-likelihood — where each step is a
weighted ridge solved by [K32]'s augmented `lstsq`. Five to ten iterations on
ordinary data, quadratic convergence, no learning rate to tune.

Multiclass is **one-vs-rest**: `k` binary fits, prediction by the largest
decision value. Multinomial softmax needs a general optimiser (L-BFGS) because
its Hessian is a `k·p × k·p` block matrix — a phase of its own, not a line of
this file.

**What one-vs-rest actually costs**, since "it is worse" is usually said
without saying how: for **prediction** it is not worse — `argmax` over `k`
independent fits and `argmax` of a softmax fit agree on the great majority of
points. What it costs is **calibration**: the `k` sigmoids do not sum to 1, so
`predict_proba` normalises them, and that normalisation is an approximation
rather than a model. scikit-learn's `liblinear` solver does exactly this and
documents it the same way. The test asserts the normalised rows sum to 1 so the
approximation is visible in the suite rather than implied by the docs.

**The L2 penalty is not optional.** On separable data the unpenalised
likelihood is maximised by coefficients of infinite magnitude, so IRLS diverges
with its weights `p(1−p)` going to zero. `C` (inverse strength, as in
scikit-learn) bounds `b`, and `p` is clipped away from 0 and 1 so the weights
cannot underflow to exactly zero. A test drives it at `C = 1e6` — the weakest
penalty the model allows — on perfectly separable data and asserts the
coefficients stay finite.

---

### K35 — `SGD*` is mini-batch, and that is a different optimiser
"One pass per epoch, `nd_*`-vectorised" and "stochastic gradient descent" are
in tension: true per-sample SGD is `n` interpreter iterations an epoch, and the
interpreter costs about a microsecond a step — a 100,000-row epoch is a tenth
of a second of pure overhead before any arithmetic.

So these are **mini-batch** gradient descent, default batch 256, where each
step is one `matmul`. The batch is a declared parameter, not a hidden one.

This is an honest difference from scikit-learn's `SGDClassifier`, which really
does take one sample at a time: **the optimisation path differs, so the
coefficients differ**, though both minimise the same objective. What is asserted
instead is the property that matters — on a well-conditioned problem
`SGDRegressor` must approach `LinearRegression`'s closed-form coefficients, and
it does, to 8.8 × 10⁻⁹. An optimiser is correct when it finds the answer the
closed form already knows; reproducing another library's trajectory would be a
weaker claim and an unmaintainable test.

**The regressor centres `X` and `y`** and recovers the intercept afterwards,
like `ElasticNet`. Writing the test for "the intercept is not penalised" found
why it has to: learning the intercept by descent starts it at 0 and makes it
travel the whole way to `mean(y)`, while the step size decays as
`1/(1 + η₀·α·t)` — so a strong penalty throttles exactly the parameter with
furthest to go. Measured on a target centred at 50 with `α = 5`: the intercept
reached 37.1 after 600 epochs and was still climbing. Centring removes the
problem instead of tuning around it.

**The classifier does not centre**, because centring a 0/1 label would change
the logistic loss into a different objective. Its intercept is learned, which
is fine — on the logit scale it is O(1). It is still excluded from the penalty,
and the test for that had to assert the **intercept** rather than accuracy:
with 90/10 classes, penalising the intercept pulls it from −0.75 toward −0.15
and *raises* accuracy from 90.3% to 96.7%, because the model then predicts the
minority class at roughly its true rate instead of never. A metric can improve
while the mechanism is wrong.

---

### K36 — The SGD step size is not coupled to the penalty
The first schedule was `η₀ / (1 + η₀·α·t)`. It is broken at **both** ends, and
both were measured rather than reasoned about:

* **Strong penalty, collapsed step.** At `α = 2` on a 90/10 classification the
  intercept read −0.754 after 400 epochs, −0.938 after 4,000 and −1.046 after
  20,000 — still moving, never reporting convergence — against the **−2.069**
  that IRLS gives for the same objective. The optimiser was not converging
  slowly; it was being governed by the one parameter that has nothing to do
  with step length. Centring `X` and `y` ([K35]) hid this for the *regressor's
  intercept* by computing it in closed form, but the coefficients and the whole
  of `SGDClassifier` were still throttled.
* **No penalty, no annealing.** At `α = 0` the schedule never decays, so there
  is no annealing precisely where gradient noise most needs it.

`learning_rate` is now explicit, and the default depends only on `t`:
`"invscaling"` = `η₀ / (1 + t)^power_t` with `power_t = 0.25`, `"constant"` =
`η₀`, `"optimal"` = `1/(α·(t + t₀))` with `t₀ = 1/(α·η₀)` so that `η(0) = η₀`
under all three. The same `α = 2` fit reaches −2.031 in 400 epochs under the
default.

`"optimal"` is the old rule, **kept rather than deleted**: it is the correct
schedule for a strongly convex objective — the strong convexity constant *is*
`α` — and it is scikit-learn's name for it. It now has to be asked for, and
requires `α > 0`.

**Why `invscaling` rather than `constant`**, which taste cannot settle:

| | noiseless target | noisy target (σ = 0.5) |
|---|---|---|
| `constant` | 8.4 × 10⁻⁹ at 175 epochs | 7.8 × 10⁻³ |
| `invscaling` | 6.1 × 10⁻⁴ at 400, **9.7 × 10⁻¹¹** at 2,000 | **2.1 × 10⁻³** |

A decaying step is *slower early* on a clean problem and **not capped** — given
the epochs it converges further than a constant one. On a noisy problem it is
nearly four times closer, because the decay is what lets the iterate settle
rather than bounce around the optimum. Real targets are noisy.

**Not claimed:** the usual robustness argument for decay. Neither schedule
diverged at `η₀ = 5` on this problem, so that reason was tested and did not
hold; the noisy-target result is the one doing the work.

**A test-design note worth keeping.** The first assertion that `power_t` is
applied passed even with the value hardcoded, because the two fits drew
different mini-batch shuffles from the global RNG — it was measuring RNG
variation, not the parameter. The comparison now re-seeds before each fit, and
a companion assertion checks that the same seed and schedule give bit-identical
coefficients, which is what makes the comparison mean anything.

---

### K37 — `ColumnTransformer` lives in `compose.b`, not `base.b`

`Pipeline` is in `base.b` because the protocol harness needs it and it depends
on nothing but the contract. `ColumnTransformer` is the **only** part of bkit
that knows `arctic` exists (design doc §8), and `base.b` is what `btext`
imports for the contract alone.

Putting them together would make importing the estimator protocol drag a
dataframe package behind it. So composition splits by dependency, not by
subject matter.

**And `compose.b` does not `include` arctic either.** It duck-types: a frame is
an instance carrying `names`, `cols` and `ncols`. Reading an undefined field on
an instance yields null rather than raising, which is the same mechanism
btorch's `is_dataset` uses. `arctic` is therefore an **optional** dependency in
`package.json` — bkit works fully without it, and only the by-name column path
needs it.

**Rejected: a marker field added to arctic's `DataFrame`.** It is the cleaner
test, and btorch does exactly that with `__bt_module`. But this is the only
consumer in the ecosystem, and asking another package to carry a flag for one
consumer inverts the dependency the rest of this decision is about. If a second
consumer appears, the marker becomes right.

---

### K38 — blocks are concatenated in SPEC order, not column order

A caller can predict spec order. They cannot predict column order, because a
`OneHotEncoder` on one input column can emit any number of outputs — so
"sorted by first input column" would still leave them unable to say where
feature 37 came from.

Hence `output_ranges()`, which returns `[name, start, stop)` per spec plus the
remainder. It is not decoration: a 40-column one-hot block is otherwise
unattributable, and *"which input produced feature 37"* is the first question
anyone asks of a fitted composition. The test asserts the ranges, not just the
total width.

---

### K39 — a column may feed two transformers; the same column twice in one spec may not

Across specs it is legal and useful — a column can be both scaled and binarised
(sklearn allows this too). Within one spec it would transform the column twice
and silently widen that block by one, which no caller ever means. So the
duplicate check is **per spec**, at fit, where the indices are resolved.

`remainder` is computed from the union of all specs' columns, so a column used
twice is still "used" once for that purpose, and the passthrough block is
emitted in **ascending index order** — X's own order, not the order a dict
iteration happened to produce.

---

### K40 — a reordered frame is refused at transform, not silently mistransformed

Column names resolve to **positions** at fit time, which is what makes the array
and frame paths one code path after resolution. The consequence is that a frame
whose columns were reordered between `fit` and `transform` would resolve the
same names to different positions and transform the wrong data — with the right
shape and plausible numbers.

So `transform` records and compares the fitted column *names*, not just the
count. This is the K-series' recurring criterion: the failure worth building
machinery against is the one that produces a wrong answer looking like a right
one.

A width change between fit and transform is caught for the same reason — a
block that emits a different number of columns would shift every later block's
meaning, so it is reported with both widths and the offending spec's name
rather than producing a misaligned matrix.

---

### K41 — `fit` is `fit_transform`, and `fit` is the one that delegates

Fitting a composition already computes every block, so `fit` followed by
`transform` would do all of the work twice. `fit_transform` is the real
implementation and `fit` calls it, discarding the result — the opposite of the
base class's default, and deliberate.

The cost is one wasted concatenation when a caller only wanted `fit`. The
alternative is running every sub-transformer's `fit_transform` and then its
`transform` again, which for a 20-spec composition over 5000 rows is measurably
worse and, more importantly, would let a stateful transformer see the data twice
per fit.

**Also why `n_features_out()` and `output_ranges()` require a fit:** the widths
are *learned*, not declared. A `OneHotEncoder`'s width is a property of the
data, so there is no honest answer before fitting, and returning a guess would
be worse than raising.

---

### K42 — one criterion serves both trees, because the gain IS Gini (up to a factor)

Both `DecisionTreeRegressor` and `DecisionTreeClassifier` set `g = -y`, `h = 1`
and use the regularised gradient-boosting gain

    gain = ½[ G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ) ] − γ

which is also exactly what K8's booster will use. That is not a compromise in
either direction:

**Regression — it is variance reduction.** With `g = −y`, `h = 1`, `λ = 0`:
`G = −Σy`, `H = n`, so `G²/H = n·ȳ²` and the gain is
`½[n_L ȳ_L² + n_R ȳ_R² − n ȳ²]` — scikit-learn's `squared_error` criterion
times ½. The leaf value `−G/(H+λ)` is `Σy/(n+λ)`, the mean at `λ = 0`.

**Binary classification — it is proportional to Gini decrease.** With
`y ∈ {0,1}` and `p = Σy/n` the same expression is `½[n_L p_L² + n_R p_R² − n p²]`.
Expanding the weighted Gini decrease and using `n_L p_L + n_R p_R = n p`:

    Σ n_m·Gini_m decrease = 2[n_L p_L² + n_R p_R² − n p²] = 4 × gain

A positive constant, so **the argmax is identical** — this picks the split
`criterion="gini"` picks, and the leaf value is the positive-class proportion,
i.e. the probability.

**The test computes the Gini table by hand** from counts, with no reference to
the gain formula, asserts the kernel picks the same cut, and asserts the ratio
is 4.0 to 1e-9. A derivation on paper is not evidence that the code implements
it.

**λ defaults to 0 for a tree**, not the booster's 1: a decision tree is
expected to report its leaf's mean, and `λ > 0` shrinks it toward zero. The
test pins that too — `Σy/(n+λ) = 12/6 = 2` against a mean of 4.

---

### K43 — multiclass is one-vs-rest, and that is a real difference from sklearn

scikit-learn grows ONE tree whose leaves hold a K-way distribution. This grows
K trees and takes the argmax. The predictions differ, and the documentation says
so rather than claiming a match.

It is the consistent choice here because the ecosystem's own GBDT design already
uses one tree per class per round (design doc §8.4), so multiclass boosting and
multiclass trees share a shape.

**What it would take to do it properly:** multiclass Gini is the per-class gain
*summed over K histograms before the argmax*, which needs a `tree_best_split`
that takes a list of histograms rather than one. That is a small kernel
extension and it is recorded as a **K8 item** — not faked here by averaging
one-vs-rest gains, which is a different criterion wearing the right name.

One-vs-rest scores do not sum to 1, so `predict_proba` normalises them. A row
where every tree says "not mine" would divide by zero, so it falls back to
uniform rather than returning `NaN`.

---

### K44 — the NaN direction is learned, and the scan and the partition agreed only after this was fixed

The NaN bin is the **highest** bin index per feature, so a plain prefix scan
leaves it in the right child for every cut — a fixed policy masquerading as an
algorithm. Worse, `tree_partition` was first written with `nan_left` defaulting
to *true*, so the partition would have sent missing rows the opposite way from
the scan that chose the split. The tree's own splits would not have reproduced
their own gains.

**Found by writing the design comment**, before any of it ran: the two defaults
were in the same file and disagreed.

Fixed properly rather than by matching the defaults: `tree_best_split` scans
each feature **twice**, once with the NaN bin's `(G,H,n)` added to the left
accumulator and once without, and returns `nan_left` alongside the split. This
is XGBoost's sparsity-aware behaviour, and it matters — imputing to the mean
first is both lossy and a different model.

The NaN-left pass also admits `bin = -1`, the cut with no value bins on the
left: the **missing-vs-present** split. `tree_partition` therefore accepts
`bin >= -1`, and its `nan_left` defaults to *false* (matching the scan's first
pass) rather than true.

The grower asserts that neither child came back empty. The guards should make
that unreachable; it is checked anyway, because an empty child's histogram is
all zeros and its leaf value is `0/λ` — a plausible-looking number from a node
that represents nothing.

---

### K45 — the hessian guard skips before the division, not after it

A saturated logistic gives `h → 0`, so with `λ = 0` the term `G²/(H+λ)` divides
by zero.

The tempting reasoning is that this is self-correcting: a `NaN` gain loses every
`>` comparison, so a `NaN` split can never win. That is true and it is not the
risk. **An infinite gain wins every comparison**, and `G²/0` with `G ≠ 0` is
`+inf`, not `NaN` — so the split scan would pick a child with no weight in it,
with maximum confidence.

So a child whose `H + λ` is not strictly positive is skipped **before** the
division. Written as `!(H + λ > 0.0)` rather than `H + λ <= 0.0` so a `NaN`
total is also caught.

The test drives it deliberately: every hessian zero with `λ = 0` must return
`feature = -1` and gain `0`, the *same* histogram with `λ = 1` must split
normally (proving the guard is about `H + λ` and not about refusing the data),
and one zero-hessian row among healthy ones must not produce an infinite gain.

---

### K46 — thresholds are raw values, so a fitted tree is one matrix

`tree_predict` takes `(nodes, X)` and nothing else. The design doc's sketch had
`(nodes, X, edges)`, with bin indices as thresholds; storing the bin's upper
**edge** as a raw value instead removes the edge table from prediction
entirely.

Consequences, all of them good:

- A fitted tree is one `[n_nodes, 7]` f64 matrix — `(is_leaf, feature,
  threshold, left, right, value, nan_left)`. K11's save/load is then trivial,
  and the binned handle is purely a *training* structure.
- Prediction is exact on values the training set never contained, where a bin
  lookup would have to extrapolate.
- The `-∞` threshold expresses the missing-vs-present split with no special
  case in the traversal: no present value satisfies `x <= -∞`.

`-∞` has to be built through the array path — Bantu raises on scalar division by
zero, while `nd_log(0.0)` on an array is `-inf`.

**Every node is validated once, before the row loop**, not per sample: a child
index outside the tree, a self-child, or a split on a feature `X` does not have
would otherwise be a per-row bounds check in the hot path, or a segfault. The
traversal is also bounded by the node count, so a cycle raises instead of
hanging — the four-minute hang in K2 is the precedent for not trusting a loop
to terminate because its data "should" be acyclic.

---

### K47 — the opaque binned matrix gets accessors, for the tests

[K1] made the binned matrix an opaque handle to get one byte per cell. The
consequence nobody wrote down is that an opaque structure with no window is a
structure whose contents are asserted by hope.

So `tree_bin_shape`, `tree_bin_nbins`, `tree_bin_edges`, `tree_bin_at` and
`tree_bin_bytes` exist **for the tests, not for callers** — and they are what
make the suite possible at all:

- `tree_bin_at` lets the reference histogram be computed in Bantu from the same
  bins the kernel used, so `tree_hist` is checked against something that is not
  itself.
- `tree_bin_bytes` asserts the K1 claim directly: 200000×50 is **10,000,000
  bytes**, not 80,000,000.
- `tree_bin_nbins` is how the test knows a 7-distinct-value column got 7 bins
  and a constant column got 1.

Bins are taken from the **sorted distinct** values, not from quantiles of the
raw array: a low-cardinality integer column then gets its own cardinality in
bins rather than 256 mostly-empty ones, and a constant column gets exactly one.
Quantiles of the raw array would place several edges between the same pair of
values, producing empty bins that every later scan still has to walk.

---

### K48 — the bin assignment was off by one, and the GAIN was still right

`tree_bin` assigned bins with `std::upper_bound`. The contract is *"bin b holds
values ≤ edges[b]"*, i.e. `bin(x) = min{b : edges[b] ≥ x}` — which is
`lower_bound`. `upper_bound` finds the first edge **strictly greater** than `x`,
so a value sitting exactly on its own edge was pushed one bin to the right.

Measured on 8 distinct values `0..7` with 8 bins and edges `[0,1,2,3,4,5,6]`:

    bin per row: [1, 2, 3, 4, 5, 6, 7, 7]      (wrong)
    bin per row: [0, 1, 2, 3, 4, 5, 6, 7]      (fixed)

Bin 0 was empty and the top two values were collapsed into one bin.

**What makes this worth a decision entry is what did *not* fail.** The split's
*gain* was correct — `tree_best_split` reported `0.5625`, exactly the Gini
argmax's `2.25 / 4`, because the same row partition was still reachable one bin
over. So:

- the brute-force reference agreed (it reads the same histogram);
- the "Gini decrease = 4 × gain" assertion **passed**;
- `tree_hist` against its Bantu reference **passed** (same bins both sides);
- only the independently computed **bin index** disagreed.

And the bin index is what `_threshold` looks up, so `edges[bin]` was one row
wide: the depth-1 regressor put five rows in a leaf that should have held four,
and failed its exactness assertion by 4.0.

**The lesson for the rest of the `tree_` family.** A test that asks "is the
gain right" cannot see this class of fault, because the gain is computed from
the histogram and the histogram is computed from the same wrong bins — both
sides of that comparison share the error. What caught it was asserting the
*index* against a value derived with no reference to the kernel at all (a Gini
table computed from counts). Every kernel in this family now has at least one
assertion of that kind, and K47's accessors exist to make them possible.

---

### K49 — `get_params()` must not return a key `set_params()` refuses

`ColumnTransformer.get_params()` returned `n_transformers` alongside
`remainder`, and `set_params()` accepted only `remainder`. The protocol test
found it immediately — `protocol_check` takes the **first** key `get_params()`
returns and feeds it straight back into `set_params()`, which is the invariant
`GridSearchCV` depends on: a search addresses a parameter by the name
`get_params` gave it.

Three ways out, and the choice matters:

| | verdict |
|---|---|
| accept `n_transformers` in `set_params` and apply it | impossible — the count is the length of the spec list |
| accept it and ignore it | **worse than the bug.** `set_params({"n_transformers": 5})` would succeed and change nothing, which is exactly the silent failure [K24] exists to prevent |
| **stop exposing it as a parameter** | **chosen** — it is derived structure, not a hyperparameter |

So it is a method, `n_transformers()`, and `get_params()` returns `remainder`
plus the `"<name>__<param>"` keys. Everything `get_params` offers is now
settable.

**`Pipeline` keeps `n_steps`** and its `set_params` accepts it, which is the
same shape and is left alone: it is a shipped surface, and setting it is
harmless because `get_params` recomputes the value from `len(this.steps)` on
every call rather than reading it back. Noted here so the difference is a
recorded decision rather than an inconsistency someone finds later.

---

### K50 — a malformed column list fails at construction, where it was written

`ColumnTransformer.init` validated spec shape, names and estimators but left
the **column list** to `_resolve` at fit time. The init comment already
promised the opposite — *"a malformed spec should fail once, where the caller
wrote it"* — and two tests written against that promise failed.

An empty column list and a *literally* repeated index need no knowledge of X, so
they are checked at construction. What still needs the width is caught at fit:
on a 4-column X, `[0, -4]` is the same column written two ways. So there are two
layers, each with its own message — and the fit-time check cannot be removed,
because the construction-time one cannot see it.

A grid search clones and refits many times; a malformed spec should raise once,
at the line that wrote it, not once per fold.

---

### K51 — `n_nodes()` reported a third of a 3-class model's size

A multiclass `DecisionTreeClassifier` holds K node arrays in `trees_`, and
`nodes_` was aliased to `trees_[0]` so that `_TreeBase`'s accessors had
something to return. `n_nodes()` read `np.shape(this.nodes_)[0]` — so a 3-class
model reported **one third of its own size**.

That is a wrong number rather than a missing one, which is the category this
package keeps finding and the reason the fix is recorded rather than quietly
applied:

- `n_nodes()` now sums over every tree.
- `nodes($which)` takes a tree index instead of pretending there is one, and
  refuses an out-of-range one by naming how many there are.
- `n_trees()` says how many: 1 for a regressor or a binary classifier, K for
  one-vs-rest.

`trees_` is also initialised to `null` in `_TreeBase.init` rather than only on
the classifier. Reading an undefined field on an instance yields `null`, which
would have made the `trees_ == null` test accidentally correct for the
regressor — and relying on that is exactly the subtlety that makes
`compose.b`'s duck-typing ([K37]) something to use deliberately and not by
accident.

---

### K52 — a mutation that is "caught by crashing" is caught, and is still worse

The `tree_` mutation battery broke `tree_hist`'s count channel (`cell[2] += 1.0`
→ `+= 0.0`). The suite did not report a failure — it **aborted**, with no
`PASS:` line at all. The battery scored that as caught, correctly: a suite that
does not print green is red.

But an uncaught throw ends the run, so every assertion after it is never
evaluated. For the battery that is harmless; for diagnosing a real regression it
is the K2 problem again — *"a throwing assertion aborted the whole preprocess
suite, hiding 40 assertions"* — which is why `safe_ok` exists. `bkit_tree_test.b`
has `safe_ok` but uses it in only two places, so most of its estimator section
is one throw away from hiding the rest.

**Recorded rather than fixed now**, deliberately: converting the suite wholesale
is a mechanical change to a file the battery is currently reading, and doing it
mid-battery would invalidate the remaining results. The improvement is to wrap
every estimator-level assertion in `safe_ok`, so a throw is attributed to the
assertion that caused it and the rest still run.

**What it already tells us about the kernel**, though, is useful: zeroing the
count channel makes `min_data` unsatisfiable for every cut, so no split is ever
found — and something downstream then threw rather than degrading. A grower that
cannot split should produce a single-leaf tree, not an exception. Worth checking
which guard fired once the battery is done; the candidates are the empty-child
assertion and `_threshold`'s "no bin edges" refusal, both of which are supposed
to be unreachable.

---

### K53 — the `tree_` battery so far: 3 of 3, and two of them by crashing

The first `tree_` mutation battery was interrupted part-way (the work was
committed at that point and the remainder re-run afterwards), so this records
what it had established:

| mutation | result |
|---|---|
| `tree_bin` back to `upper_bound` — i.e. [K48] reintroduced | **caught**, 4 assertions red |
| `tree_hist` drops the count channel (`cell[2] += 0.0`) | **caught**, by abort |
| `tree_hist` swaps the gradient and hessian channels | **caught**, by abort |

T1 is the one that matters most: it proves the off-by-one that shipped and was
fixed is now *gated*, so it cannot come back silently. The assertion that
catches it is the bin-index comparison against an independently computed Gini
argmax — exactly the one [K48] argues for.

T2 and T3 were caught by **aborting** rather than failing cleanly, which is
[K52]: `bkit_tree_test.b` needs `safe_ok` around its estimator section so a
throw is attributed instead of ending the run. Both also point at the same
unanswered question — a grower that can find no split should produce a
single-leaf tree, not an exception — and the guards that could be throwing are
both documented as unreachable. That is worth chasing on its own terms, not
just as test hygiene.

**Both halves are now resolved, and neither resolution was test hygiene.**
Chasing the throw found [K55]: a third site, `_leafValue`, divided by `H + λ`
with no guard at all, so a zero-hessian node raised `Division by zero` out of
the middle of `_grow`. A grower that can find no split now returns a single-leaf
tree with value 0 — gated four ways, and reached two different ways (a zero
hessian, and a row subsample whose target is constant). The attribution itself
is [K56]: all 33 sections are wrapped, and the same two mutations that used to
report *"(crashed)"* now report `PASS: 79  FAIL: 33` with the throwing section
named. T4–T12 were re-run against the restored kernel afterwards.

---

### K54 — histogram subtraction landed in K7's grower, and its gate is per channel

Subtraction was designed as a K8 item (§8.6) on the reasoning that the booster
is where the cost lives. It went into K7's grower instead, because the grower is
*shared* — K8 adds a boosting loop over it, so an optimisation inside it is
work K8 does not repeat, and more importantly it is work K8 does not have to
re-gate.

A child's histogram is its parent's minus its sibling's. The grower builds the
**smaller** child from the data and subtracts for the larger, which roughly
halves the histogram work per level. The queue item grew a fourth slot to carry
the histogram down from the parent; the root's is `null` and is built directly.

**The gate I designed for it was false.** §8.6's first draft asserted the
subtracted histogram would be *bit-identical* to one built from the child's own
index set. `tree_hist` accumulates in index order, so the parent sums rows
`0..n−1` while the sibling sums a subsequence, and `sum(A ∪ B) − sum(A) =
sum(B)` does not hold exactly in floating point. Writing the assertion down
before writing the code is what exposed it — the same order of operations that
caught [K44]'s `NaN` direction.

So the gate splits by **channel**, which says more than a single tolerance
would:

| channel | assertion | why |
|---|---|---|
| count | **bit-identical** via `nd_tobytes` | counts are integers held in doubles, all exactly representable, so any difference is a bug and not rounding |
| gradient, hessian | `< 1e-12`, and the **worst observed** difference is printed | a drift from 1e-16 to 1e-9 then shows up as a number rather than as a still-passing assertion |

Plus the property the grower actually depends on, which neither of those implies
on its own: both histograms must choose an **identical split**. A tolerance on
the channels permits a difference; a tie near the top of the gain ranking could
in principle turn that difference into a different cut, and the direct assertion
is the only thing that rules it out.

Observed on the test's dataset: both differences are exactly 0. That is not
evidence the identity holds — it is a seven-row matrix whose sums happen to be
exact — which is why the tolerance stays and the number is printed rather than
asserted to be zero.

The error does not accumulate across levels: each child derives from its
parent's histogram, not from a running total, so the depth of the tree does not
enter the bound. This is what LightGBM and XGBoost both accept.

`bkit_tree_test.b` 123 → 127 assertions.

---

### K55 — the kernel guarded the gain, the caller divided by zero

[K45] is the record of `tree_best_split` refusing a child with
`!(HL + λ > 0.0)`: the danger with a saturated hessian is not a `NaN` gain,
which loses every comparison, but `G²/0` with `G ≠ 0`, which is `+inf` and
**wins** every comparison. The kernel has had that guard since it was written.

`_leafValue` computes `−G / (H + λ)` **from the same two quantities** and had no
guard at all. In Bantu a scalar `/ 0` raises (an array `0/0` is `NaN`), so a
node whose hessians sum to zero did not produce a wrong leaf — it threw
`Division by zero` out of the middle of `_grow`, with no node, no tree and no
indication of which node.

**Reachability is the part worth being precise about.** It is *not* reachable
from K7's two estimators: both set `h = 1` per row, so `H = n ≥ 1` and the
denominator is at least `1 + λ`. It is reachable from **K8's booster**, where
logistic `h = p(1−p)` underflows to 0 across a saturated node and `reg_lambda`
defaults to 0 — so the first thing K8 does with this grower would have hit it.
That is precisely why it is fixed in the grower and not in the booster: there is
one grower ([§8.6]), and the defect is the *asymmetry* between a kernel that
guards and a caller that does not, not the booster's loss function.

**Zero is the correct value, not a fallback.** `−G/(H+λ)` is the minimiser of
the second-order approximation of the loss; at `H + λ = 0` that approximation
has no curvature and its minimiser is unbounded, which is an artefact of the
approximation rather than a step worth taking. XGBoost's `CalcWeight` returns 0
in exactly this case and LightGBM does the same. Written `!($d > 0.0)` so a
negative total (a non-convex loss) and a `NaN` total are caught too — character
for character the kernel's guard.

Gated four ways, and the mutation was run: removing the guard turns the section
red with 4 failures rather than aborting the file.

| assertion | what it pins |
|---|---|
| all-zero hessians, `λ = 0` → a tree comes back | it does not raise |
| that tree is `[1, 7]` | a grower that can find no split yields a **single leaf**, which is [K52]'s open question answered |
| its value is `0.0` | not `inf`, not `NaN` |
| the same `g`, `h` with `λ = 1` grow a real tree | the guard is about `H + λ`, not about refusing the data |
| hessians that **cancel** to exactly zero | the negative-hessian path reaches the same guard |

The general lesson, recorded in the language traps as well: **a guard the native
kernel applies, every Bantu caller computing from the same quantities must apply
too.** The kernel having one is not protection for the caller.

---

### K56 — a throw must name its section, and a `def` inside a `try` disappears

[K52] asked for this: two `tree_` mutations were "caught by aborting", which
counts as caught and is still worse than a clean red, because the battery
learned nothing about *which* assertion or *which* guard, and every section
after the throw went unreported.

`bkit_tree_test.b` now wraps **each of its 33 sections** in its own
`try`/`catch`, and a throw is attributed:

```
  FAIL  [section aborted] tree_hist over a subset
          threw: {message: ..., line: 212}
```

One failure, named by section, and the remaining sections still run and still
report. The whole-file alternative — one `try` around everything — gives one
attribution and loses every later section, which is most of what was wrong with
aborting in the first place.

This is cheap only because of a Bantu property that had to be verified rather
than assumed: **a top-level `try` block does not open a new scope for
variables.** Anything assigned inside it is visible after the `catch`, so the
file's straight-line flow — later sections reading `$B`, `$g`, `$hist` from
earlier ones — is unchanged. Converting every assertion to a closure-taking
`safe_ok` was the alternative and was rejected: Bantu closures capture by
reference, so 132 closures over shared script variables is a trap-rich rewrite
of a file whose job is to be trustworthy.

**But a `def` inside a `try` block *is* scoped to that block.** Four reference
helpers (`refHist`, `chan`, `bruteSplit`, `giniDecrease`) were defined
mid-section and became invisible to the sections that consumed them, failing as
`Undefined variable: refHist` from a line in a different block — the first run
after wrapping caught it, attributed to the right section by the very mechanism
being added. All four are hoisted above their `try`. Recorded as language trap
33, because the asymmetry (variables leak, functions do not) is not guessable.

One case was fixed rather than wrapped. The new [K55] section read `$lzN` in two
assertions after the call that produces it, so removing the guard aborted the
run on `nd_shape(null)` — a caught throw cascading into an uncaught one. Those
assertions now fail explicitly when `$lzN` is null. A section `try` is a
backstop, not a licence to let an assertion depend on a value that may not
exist.

---

### K57 — the gain is a grower side output, not an eighth node column

`feature_importances_` needs the gain of every split. A node is **7 doubles**
and `tree_predict` strides by 7 in C++, so an eighth column would be a native
change *and* a breaking change to the format every saved model uses — to carry
a number prediction never reads.

The gain is therefore recorded where it is still in hand, as `_grow` accepts
each split, into a per-feature vector published as `lastGains_` / `lastSplits_`.
Reading it back off the node array afterwards is not merely awkward, it is
impossible: the node holds the **threshold**, not the gain.

Two importances, reported separately rather than blended:

- **gain** — the total gain of every split on a feature, which is what the model
  actually used it for;
- **split** — how many splits used it, which one decisive split and ten marginal
  ones look identical under.

The accumulation is per tree, not per model, and that is the subtle part:
`_grow` overwrites its side outputs on every call, so a K-class one-vs-rest
classifier (or a 100-round booster) that read them at the end would report only
its **last** tree. `_accImportance()` folds each tree in as it is grown, and the
test asserts a 3-class model counts at least one split per tree.

A model with **no splits at all** — a constant target, or every gain
non-positive — has total 0. It returns **zeros**, not `total/0`: an array `0/0`
is `NaN` in numba, and a vector of `NaN`s reads as "unknown" when the truthful
answer is "this model uses no feature at all".

---

### K58 — the hessian clamp was harmful, and the measurement says so

The design doc said, in §8.4: *"The hessian clamp (`max(h, 1e-16)`) is not
cosmetic"* — reasoning that a saturated logistic gives `h → 0`, and keeping the
Newton step finite beats discarding it. `ensemble.b` shipped it that way for
about an hour. **It is the opposite of safe**, measured directly on the grower:

| h | λ | min_child_weight | result |
|---|---|---|---|
| **1e-16** | 0 | 0 | **splits; leaves of ±5e15** |
| 0 | 0 | 0 | one leaf, value 0 |
| 1e-16 | 1 (the default) | 1 | one leaf, value 0 |

The middle row is the correct answer and the two existing guards already give
it: `tree_best_split` refuses a child with `!(H + λ > 0)` ([K45]) and
`_leafValue` returns 0 for the same condition ([K55]). Lifting `h` to `1e-16`
makes `H + λ` clear **both** of them by a hair, and converts a refused split
into a step of five quadrillion — the `+inf` failure mode [K45] exists to
prevent, scaled down just far enough to survive a finiteness check. **The clamp
defeated the guards in exactly the configuration where they are needed**, and
it was invisible at the defaults (λ = 1 and `min_child_weight` = 1 both rescue
it), which is the worst place for a hazard to hide.

`_clampH` is now `_validH`, and it corrects only what is **invalid**:

- a **negative** hessian — no convex objective produces one, and `G²/(H+λ)` is
  meaningless for it;
- a **NaN** — which would make the gain lose every comparison and the leaf
  `NaN`.

Both become 0, which the guards then handle correctly. A genuine zero is
**passed through**, and a genuinely tiny hessian (1e-20) is left alone, so this
is not a floor. `np.maximum` **propagates** `NaN` rather than dropping it the
way IEEE `fmax` does, so the `NaN` case needs its own `where()` — asserted, not
assumed.

**Its gate is a unit test on the function, and that choice is the finding.**
Reintroducing the 1e-16 clamp left all 108 assertions green, including a
150-round saturated logistic at λ = 0: on a saturated logistic `g → 0` as fast
as `h` does, so the quotient stays small and the clamp looks harmless. The
±5e15 case needs `g` to survive while `h` vanishes — a *confidently wrong*
point — and even a deliberately mislabelled row did not reach it through two
guards and a scan. Rather than keep hunting for a fixture, the contract is
pinned on `_validH` directly, where the decision lives. All three mutations
(clamp reintroduced, negatives passed through, `NaN`s passed through) are now
caught.

---

### K59 — `get_params()` handed out the live dict, so a read accessor could write

Found by chasing a mutation that turned out to be **equivalent**. "Clone shares
the parameter dict" passed, and it passed for a good reason: the constructor
builds a fresh `_defaults()` dict and copies into it, so handing it the live
dict aliases nothing. Equivalent mutations are worth reporting as equivalent
rather than as weak tests — and following this one up found a real defect one
level out.

`Estimator.get_params()` returned `this.params_` itself, and Bantu hands dicts
over by reference. So the estimator was **writable through a read accessor**,
which bypassed both guards documented immediately beside it:

```
$p = $scaler.get_params();
$p.with_mean = false;     // changed a FITTED scaler, left fitted_ = true
$p.with_men  = true;      // ADDED a parameter nobody reads
```

The first is exactly the wrong answer `set_params`' fit-invalidation exists to
prevent — `transform` then runs with parameters that do not match the statistics
it learned. The second is [K24]'s mistyped-key bug, reachable through the
accessor next to the check for it. This was in `base.b`, so it applied to
**every estimator in the package**.

`get_params()` now returns a copy. Nothing in bkit mutated the result — every
use is a read — so the change removes only a capability that was never intended
and that defeats two documented guards.

**How deep the copy goes was measured, because Bantu's answer is asymmetric**
(language trap 35): the copy loop leaves **list** values independent, because
binding a dict's list value copies it, while a nested **dict** stays aliased.
That makes it deep enough for every parameter bkit has — `ColumnTransformer`'s
specs are lists, and `GridSearchCV` keeps its grid in `grid_`, not `params_` —
and a dict-valued parameter added later would still alias through. Both halves
are asserted in `bkit_base_test.b`, so the limit is recorded rather than
discovered.

---

### K60 — the booster's first battery: 11 of 17 missed, and five distinct reasons

`bkit_ensemble_test.b` passed 82 assertions on its first run, over an 800-line
module written in one pass. The session's own rule — *a gate nobody has tried to
break is a gate nobody has tested* — says that is a reason for suspicion, and it
was: **11 of the first 17 mutations survived.** Writing them down by cause
matters more than the count, because the causes repeat.

| cause | mutations | the shape of it |
|---|---|---|
| **a gate that proves a disjunction** | E10, E11 | "without reseeding the model differs" is satisfied by *either* subsample or colsample being drawn, so dropping one stayed green |
| **an inequality true of anything in range** | E8, E12 | `best_round() <= n_rounds()` holds for a `best_round` that kept the **worst** round, and for a `predict` that ignored it entirely |
| **an output recomputed on the read path** | E5 | `predict_proba` recomputes the softmax, so a wrong **training** gradient is invisible in it |
| **a quantity that only changes the rate** | E4, E17 | dropping the hessian to 1 turns Newton into gradient descent; on separable data it still converges to the same place |
| **a branch no test entered** | E7, E16, E6 | nothing early-stopped a *classifier*, so a flipped log-loss sign was unreachable; no fixture made an improvement follow a stall, so the patience reset was never exercised |

Each fix is a gate of a different kind, and the pattern is that **a weak gate
asserts something true of the right answer, while a strong one asserts something
false of every wrong answer**:

- the subsamples are pinned **individually and deterministically** — under one
  seed, turning each on must change the model, which needs no randomness to be
  present to mean something;
- `best_round` is checked against the stopping rule **reimplemented in the
  test**, and `predict` is required to equal `staged_predict[best_round − 1]`;
- the hessian is pinned by **hand-derived leaf values** — a depth-1 logistic
  tree on `y = [0,0,0,0,1,1,1,1]` has `F0 = 0`, `p = 0.5`, `h = 0.25`, so its
  leaves are exactly **−2 and +2**; with `h = 1` they would be ∓0.5;
- softmax is pinned by the fact that `F0 = 0` for every class makes
  `p_k = 1/K` **exactly** — a number a per-column sigmoid cannot produce for
  K ≠ 2, giving leaves of 3 and −1.5 against the sigmoid's 2 and −2;
- the stopping fixture was **replaced** with one that improves, stalls and
  improves again, and the test asserts that property of the fixture so it fails
  loudly if it stops holding.

Final: **19 mutations, 18 caught, 1 equivalent** (E14, which led to [K59]).
`bkit_ensemble_test.b` 82 → 112 assertions. Three of the fixes exist because
the first version of the gate was *vacuous rather than wrong*, which is the same
fault numba's `ok(nd_all(...))` had (N41) — the third time this session that the
instrument, not the thing measured, was the defect.

---

### K61 — "no new kernels" was checked against the kernels, not against the caller

[§8.5] is the record of checking each K8 feature against K7's five `tree_`
kernels and concluding none was missing. For the boosters that held exactly: the
whole of `ensemble.b` is one `nd_subtract` per node, an index array `tree_hist`
already accepts, and a `[f,1,1]` mask multiplied into the histogram the scan
sees. **For `ExtraTrees` it is true of the arithmetic and false of the control
flow**, and the gap is worth naming because of *how* it was missed.

§8.5 says random split thresholds reduce to *"pick a bin per candidate feature
and evaluate that one cut's gain in Bantu from the histogram"*. The arithmetic
is six numbers. But `_grow` calls `tree_best_split` **itself**, and there is no
way to hand it a cut — so the feature needs a change to the grower's structure,
not to its kernels. The check compared the feature against the kernel list and
never asked which code path would invoke it. **A sufficiency argument about
primitives says nothing about who gets to call them.**

Two resolutions, and the rejected one is the instructive half:

- **a new kernel** `tree_eval_split(..., chosen_bins)` — rejected. It is
  `tree_best_split` with the argmax removed, so the regularised gain formula
  would exist in two translation-unit locations. [K42]'s argument is that one
  criterion serving every estimator is what makes the ×4 Gini proof worth
  anything; two copies of the formula means the proof covers one of them.
- **one virtual method**, `_chooseSplit($hist, $B)`, returning the same 10-slot
  vector `tree_best_split` returns. `_TreeBase`'s implementation *is* that call;
  an extra-trees grower overrides it. Everything else — the partition, the `NaN`
  direction, the empty-child assertion, the threshold lookup, the subtraction,
  the leaf guard — stays the one implementation, which is [§8.6]'s argument for
  sharing the grower applied one level down.

Cost stated rather than assumed: a Bantu method call is 1.78 µs ([B1]) and this
one is **per node**, beside a `tree_hist` that touches `n_rows × n_features`
cells. A 1,000-node tree pays 1.8 ms of dispatch against histogram work measured
in hundreds of milliseconds, and the override's body is `O(max_features)`
vectorised slices, not one call per bin.

The override still owes the same two things the depth-wise scan does, and the
10-slot return type is what forces it to: the chosen cut is evaluated **both
ways** for `NaN` ([K44] — the `NaN` bin is the highest index, so a prefix sum
leaves it right unless the direction is chosen on merit), and it returns
`feature = −1` when either child fails `!(H + λ > 0)` ([K45], [K55]) rather than
leaving a zero-weight child to the caller.

Recorded as a decision rather than folded into §8.9 silently, because the
*method* that produced the wrong answer is reusable: a "needs no new X" claim
has to be checked against the call sites, not only against the inventory of X.

---

### K62 — the `tree_` battery, all twelve: a guard hidden behind a guard, and a tie nobody made

[K53] recorded the first three mutations of the `tree_` kernels before the
battery was interrupted. Re-run in full against the restored kernel and the
156-assertion suite:

| mutation | result |
|---|---|
| T1 `tree_bin` back to `upper_bound` ([K48]'s bug) | caught, 4 red |
| T2 `tree_hist` drops the count channel | caught, 35 red |
| T3 `tree_hist` swaps gradient and hessian | caught, 33 red |
| **T4 `best_split`'s per-CHILD hessian guard removed** | **MISSED** |
| **T5 ties go to the HIGHEST index** | **MISSED** |
| T6 `best_split` never tries NaN-left | caught, 4 red |
| T7 `partition` uses `<` instead of `<=` | caught |
| T8 `partition` swaps the children | caught, 10 red |
| T9 `predict` inverts the NaN direction | caught, 2 red |
| T10 `predict` uses `<` on the threshold | caught, 3 red |
| T11 `tree_bin` puts NaN in bin 0 | caught, 7 red |
| T12 `predict`'s cycle bound removed | caught, 2 red |

T2 and T3, which [K53] could only record as *"caught by aborting"*, now report
`PASS: 121 FAIL: 35` and `PASS: 123 FAIL: 33` with the throwing section named —
[K56] working as intended.

**T4 is the one worth generalising: a guard was hidden behind another guard.**
`tree_best_split` checks the hessian **twice**, and the two are not redundant:

```
if (!(Hs + lam > 0.0)) continue;            // per FEATURE, on the parent total
    ...
    if (!(HL + lam > 0.0) || !(HR + lam > 0.0)) continue;   // per CUT
```

Removing the second left all 156 assertions green. The reason is entirely about
the fixture: the only test that drove hessians to zero drove **all of them** to
zero, so the parent total was zero, the *first* guard skipped the whole feature,
and the second was never reached. **A fixture that trips the outer guard can
never exercise the inner one** — and the suite had been reading as though it
tested "the hessian guard" when it tested one of two.

The gate that reaches it keeps the parent healthy and starves one child: rows
0–2 carry `h = 1` and rows 3–6 carry `h = 0`, so the parent total is 3 and the
feature is scanned, while the cut at bin 2 gives `H_R = 0` with `G_R = 2`. Its
gain without the guard is `G²/0 = +inf`, which wins every comparison — [K45]'s
failure mode exactly, reached through the path [K45]'s own test could not take.
Asserted on the returned slots (`H_L > 0` and `H_R > 0`) rather than on the gain
alone, so it names the property instead of a symptom.

**T5 is simpler and was invisible for a simpler reason: no fixture had a tie.**
Changing `gain > bestGain` to `>=` moves the winner from the first maximal cut
to the last, and nothing in 156 assertions produced two equal gains. That is not
cosmetic. Duplicate and perfectly correlated columns are ordinary, and which one
a tree splits on decides what `feature_importances_` reports and whether two
fits of the same data agree on their structure rather than only on their
predictions. The gate uses two **identical** columns, and asserts the tie is
real — the two features' best gains compared bit for bit — before asserting the
lower index wins. Without that first half the second proves nothing.

Both causes are the same species as [K60]'s five: **a branch no fixture entered**
and **a comparison no fixture made ambiguous**. The battery is the only thing
that finds either, because both read as fully covered from the test file alone.

`bkit_tree_test.b` 156 → 166 assertions.

---

### K63 — a random forest samples features per NODE, and the accuracy gate is what said so

`RandomForest*` reuses the booster's column-sampling hook on its first draft:
draw `max_features` columns, call `set_features(cols)`, grow the tree. That is a
**random subspace ensemble**, not a random forest, and the difference is not
pedantry about a name:

> With `f = 6` features and `k = 2`, a tree that draws its columns **once**
> never sees the one informative feature with probability 2/3. Two thirds of the
> forest is then fitted to noise.

Measured, on a target one feature explains: **R² 0.66 where it should be 0.995**,
and a held-out MSE *worse* than a single deep tree — which inverts the one claim
a forest exists to make. Breiman's forests redraw the candidate set at **every
node**, and that is what lets every tree use the informative feature while still
decorrelating them.

**The gate that caught it was the accuracy gate, not the structural one.** The
structural assertion — *"with `max_features = 1` no tree references more than
one split feature"* — passed, because the implementation was per-tree and the
assertion described per-tree behaviour. A test written from the implementation
agrees with the implementation. The two assertions that failed were
`R² > 0.9` and *"the forest generalises better than one deep tree"*, both of
which are statements about the **estimator's purpose** rather than about its
mechanics. That is the argument for keeping at least one gate per component
phrased in terms of what the thing is *for*.

The structural assertion is now **inverted**, and in that form it distinguishes
the two designs rather than restating one of them: per-node sampling makes a
single tree reference *many* features (6 of 6, measured), while per-tree
sampling pins it at two — one real feature plus the leaves' filler zero.

**Two hooks, not one switch.** `set_features(cols)` samples once per tree and
stays the **booster's**, because that is XGBoost's `colsample_bytree` and is
correct there; `set_features_per_split(k)` is the **forest's**, and `_grow`
redraws the mask at each node. A single parameter serving both would have to
mean different things to different callers, which is how the first version came
to be wrong. The mask is rebuilt in three native calls and no Bantu loop —
`rank(uniform(f)) < k` selects `k` of `f` without replacement, [K18]'s
construction reused.

**The defaults differ by task, and that is scikit-learn's behaviour for a
reason**: a regressor considers every feature at each split (`max_features =
1.0`) while a classifier considers `sqrt(f)`. Using `"sqrt"` for both — the
first draft again — is precisely the configuration measured above. Each subclass
sets its own default, and `_forestDefaults()` leaves it `null` so the choice is
visible at the subclass rather than buried in a shared table.

---

### K64 — an average is only checkable by reproducing the average

The forest battery's two survivors were the same defect wearing different
clothes, and the shape is worth keeping:

| mutation | why every gate stayed green |
|---|---|
| **F4** the forest averages over `n − 1` trees | a *uniform scale error*. Every prediction is multiplied by 30/29; `R² > 0.9` absorbs it, and the classifier's row normalisation **cancels it exactly** |
| **F9** the classifier averages thresholded 0/1 labels instead of probabilities | a majority vote still classifies easy data perfectly, and the rows still sum to 1 after normalisation — so "it classifies well" and "every row sums to 1" are both true of the wrong estimator |

F9 is the sharper one, because §8.8 names it specifically as the thing not to
do — *"a majority vote over hard labels throws away every tree's confidence and
is a different, worse estimator"* — and the suite had an assertion for each of
the properties a vote preserves and none for the property it destroys.

**Normalisation is an information-destroying step, and every assertion
downstream of one is blind to anything it cancels.** Row-normalised
probabilities cannot see a scale error at all. Accuracy cannot see one either,
because argmax is invariant under it. Between them they covered the whole
classifier surface and tested nothing about the average.

The fix is the same for both and is not a threshold: **reconstruct the average
from the public node arrays and demand bit equality.**

```
sum over trees of tree_predict(nodes(t, k), X)  /  n_trees   ==  predict(X)
```

That is an independent computation in the sense [K48] requires — it goes through
`tree_predict` and the node arrays, not through `_avgCols` — and it is exact, so
it catches a wrong divisor, a sum instead of a mean, a threshold inserted before
the average, and any future change to that path, none of which a tolerance or an
accuracy bound can see. The classifier's version reproduces the clip, the
concat and the row normalisation too, which is the point: the gate has to end up
where `predict_proba` ends up, or it is comparing different things.

Both reconstructions were added, and F4, F9 and F10 (sum instead of mean) are
caught by them. The general rule: **an aggregate's gate must recompute the
aggregate, because every property that survives aggregating is by definition
insensitive to how it was aggregated.**

Final: **10 forest mutations, 10 caught.** `bkit_ensemble_test.b` 140 → 156
assertions. Two of the ten (F6 `sqrt(f)` computed as `f`, F8 the out-of-bag
counter marking every row) were caught only by the **unit** gates on
`_nFeatures` and `n_in_bag` — both added because listing the mutations *before
running them* made the holes obvious. Writing the battery is itself a review of
the suite, and the cheapest part of it happens before any mutation runs.

One process note, because it is the second time this session: the first run of
this battery was invalidated part-way because the test file it reads was being
edited while it ran. The numbers for F1–F8 remained valid (assertions were only
added, never weakened), but the battery was re-run end to end against the final
file rather than reported from a mixture. **A battery pins the file it reads;
editing that file mid-run forfeits the run.**

---

### K65 — ExtraTrees is one optional argument, and [K61]'s rejection was wrong

[K61] looked at three ways to let the grower take a *random* split instead of
the best one, and recorded the middle one. Implementing it found the third,
which is better than both — and, more usefully, found that the reason [K61]
gave for rejecting the first was **not a real reason**.

[K61] rejected a `tree_eval_split` kernel because *"the gain formula would exist
in two places"*. Two functions in one translation unit can share one inline
helper, so that confuses **two call sites** with **two copies**. The objection
that survives is smaller and still sufficient: a second entry point duplicates
the *validation* — the shape checks, the `nbins` agreement, the `NaN`-bin
bookkeeping — and that is exactly where [K44] and [K48] both lived.

The chosen answer avoids both problems by not being a second anything:

```
tree_best_split(hist, binned, λ?, γ?, min_child_weight?, min_data?, cuts?)
```

`cuts` is a per-feature vector of bin indices; when present, feature `f`
considers **only** the cut at `cuts[f]`. The change inside the loop is one line,
placed after the prefix accumulation and never before it, because the running
`(G_L, H_L, n_L)` still has to walk every bin on the way to the chosen one:

```
if (haveCuts && b != (int)cuts[f]) continue;
```

It beats the virtual method [K61] chose on every axis [K61] argued over. One
loop means one formula. One entry point means one set of validations. The scan
stays **native**, so ExtraTrees pays nothing per node, where the override would
have paid `O(max_features)` Bantu dispatches *and* put the regularised gain in a
second **language**, free to drift from the C++ one with no test able to see it
until the two disagreed. And the `NaN` double pass, the guards, the tie-break
and the 10-slot return are inherited rather than reimplemented — which matters
more here than anywhere else, because a randomly chosen cut is far likelier than
a scanned one to leave a child empty or weightless, so [K45]'s and [K55]'s
guards move from the cold path to the hot one.

**The draw stays in Bantu**, and that is a testing decision as much as a [C4]
one. No RNG inside a kernel keeps the forest reproducible from `nd_seed` alone;
more importantly it keeps the kernel **deterministic given its argument**, so a
test hands it a known vector and asserts the exact split it must return. A
kernel that drew its own cuts could only be checked statistically.

**It never clamps.** A bin index outside `0 .. vbins−2` raises, and a fractional
one raises rather than truncating. Rounding into range would turn an off-by-one
in the drawing expression into a working forest with a silently biased cut
distribution — [K48]'s failure mode exactly, a wrong answer that still produces
plausible trees. The consequence is that the *caller* has to be right about the
admissible range, which is why the grower computes each feature's value-bin
count **once per tree** (it is a property of the binned matrix, not of the node)
and emits `−1` for a feature with fewer than two of them: such a feature has no
cut to offer, and `−1` is already what the scan treats as "skip" when there is
no `NaN` bin and as the missing-vs-present cut when there is.

The gate that matters most is the one tying the two paths together:

> forcing each feature to the cut the **full scan** chose must reproduce the
> full scan's 10-slot answer **bit for bit**.

That is what says the restricted path and the scanning path share one formula,
rather than agreeing to within a tolerance. Alongside it, the gain of a forced
cut is checked against the formula recomputed from the histogram's own
`cumsum` — an independent route in [K48]'s sense, since it goes through the
histogram rather than through the kernel.

**Twelve mutations, twelve caught** — five on the kernel's `cuts` path (ignored
entirely; the filter placed *before* the prefix accumulation rather than after;
an out-of-range index clamped instead of refused; the filter inverted to keep
every bin **but** the chosen one; a fractional index truncated) and seven on the
Bantu side. Two of the seven exposed gaps nothing else would have:

- **nothing asserted the cuts vary WITHIN one tree.** A per-tree draw is still
  reproducible, still differs between seeds, and still differs from the scanned
  tree — every existing assertion holds. The one thing it cannot do is put two
  different thresholds on one feature in one tree, so that is what the gate
  measures (observed: 5). This is [K63]'s per-tree/per-node distinction again,
  arriving a second time in a different disguise.
- **no random-cut fixture contained `NaN`**, so counting the `NaN` bin into the
  value-bin total was *equivalent* on every matrix in the file. With `NaN` the
  count is overstated by one and a draw can land on the last value bin, which
  the kernel refuses. Twenty-five seeded trees over a `NaN`-bearing matrix now
  have to grow without a single out-of-range draw.

Everything else about `ExtraTrees*` is the forest with `_randomCuts()` returning
true. `bootstrap` defaults to **false**, matching Geurts et al., and not as
imitation: the randomness is already in the thresholds, so resampling the rows
as well mostly just removes training data. The test asserts the difference
structurally — with no bootstrap and every feature available, a RandomForest
grows five **identical** trees while an ExtraTrees still grows five different
ones, which isolates the switch to the one thing that changed.
