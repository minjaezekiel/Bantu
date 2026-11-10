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
