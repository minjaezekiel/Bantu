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
