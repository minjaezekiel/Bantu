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

## Phase K3 — `selection.b`

Four splitters, cross-validation and two parameter searches, 922 lines. Design:
[`../docs/bkit-architecture.md`](../docs/bkit-architecture.md) §4; decisions K18–K24.

- **[feature]** `train_test_split` (with a stratified option) and
  `train_test_split_indices`; `KFold`, `StratifiedKFold`, `GroupKFold`,
  `TimeSeriesSplit`; `cross_val_score`, `cross_validate`; `GridSearchCV`,
  `RandomizedSearchCV`; plus `param_grid`, `as_splitter`, `take_rows` and `splitter_check`.
- **[design]** **A splitter is not an estimator** (K20). `KFold` has nothing to fit and nothing to
  predict, so forcing it through the estimator protocol would mean five of seven methods raising on
  every splitter. They get their own three-method contract — `get_n_splits`, `split`, `partitions` —
  and their own `splitter_check`, in `selection.b` so later phases inherit one definition of it.
  `partitions()` is on the protocol rather than assumed by the checker because `TimeSeriesSplit`
  deliberately never tests the first rows: a checker that assumed coverage would exempt it by name,
  and the next splitter would be exempted by whoever added it.
- **[design]** **Splitters return index arrays, never data** (K19). A 5-fold split of a 1M × 100
  matrix copies nothing, and `cross_val_score` takes rows once per fold — but the deciding reason is
  that **the invariants become checkable**. "No group appears in two folds" is a statement about
  indices; given only `X_train` matrices you would have to search for rows, and `splitter_check`
  could not exist in the form it does.
- **[design]** **No `random_state` on any splitter** (K21). Reproducibility comes from seeding once
  with `nd_seed`, the ecosystem's single RNG authority ([C4]). A per-splitter seed would have to
  call `nd_seed` to take effect, clobbering the stream for everything downstream — a model drawing
  its weights after a grid search would get a stream chosen by the splitter. The consequence is a
  requirement that was needed anyway: **the searches materialise their splits once** and reuse them
  for every candidate, because re-splitting per candidate with a shuffled splitter compares
  candidates on different folds and makes the winner partly a draw of the RNG.
- **[design]** **A permutation is `nd_argsort` of random keys** (K18) — two kernel calls instead of
  `n` interpreter iterations with `n` RNG draws. The C-style `for` also stops silently at 100,000
  iterations, so a hand-written Fisher–Yates would be *wrong* above that size without saying so.
  Recorded with its caveat: `argsort` of distinct keys is uniform, and two equal `f64` keys break by
  index — about a 1-in-10,000 chance of any collision at n = 1,000,000, perturbing one adjacent pair.
- **[design]** **The fold-size rule is written down and stratification rotates it** (K22). The first
  `n mod k` folds get one extra row, which is scikit-learn's rule; a different one silently changes
  every score a user compares against a Python baseline. `StratifiedKFold` applies it per class and
  **rotates the starting fold between classes** — without that, every class puts its extra in fold 0
  and three classes make fold 0 three rows larger than the last. Each class is still within ±1
  either way, so a test of class ratios alone passes; the *folds* are what go wrong.
- **[design]** `train_test_split` returns a **named dict** (K23), not four positionals.
  scikit-learn's `X_train, X_test, y_train, y_test` interleaves the two variables, and swapping two
  gives a working program that trains on its test set. Bantu has no tuple unpacking at all, so the
  positional form would read as `$r[2]`/`$r[3]`. The indices come back too, so a caller splitting a
  third array the same way does not have to re-derive the split — which with a shuffled splitter
  would give a *different* one.
- **[bug fix]** **In `base.b`, found while designing this phase:** `set_params` accepted **any** key
  and added it to `params_`. A grid over a mistyped key — `with_men` for `with_mean` — therefore ran
  every candidate with identical settings, scored them identically, and reported a best
  configuration it had never tried (K24). It now refuses a key `get_params()` does not hold and
  names the ones it takes. Presence is tested with `keys()` rather than `== null`, because a
  parameter may legitimately *hold* null — `OrdinalEncoder`'s `unknown_value` does.
  `GridSearchCV` validates the whole grid before the first fit, so a typo costs a message instead of
  minutes of fitting. Regression test in `tests/bkit_base_test.b`, grown to **53 assertions**,
  including that a parameter whose value *is* null stays settable — the reason presence is tested
  with `keys()` and not with `== null`.
- **[test]** `tests/bkit_selection_test.b` — **117 assertions**, tiers 1–4. The gates: train and
  test never overlap in any fold of any splitter; `cross_val_score` leaves the estimator it was
  given unfitted and never shows it a row; the search's `best_score()` equals a hand-recomputed mean
  over the same folds to 1e-15, **and so does every other candidate's**; every training index
  precedes every test index in `TimeSeriesSplit`; each of four groups is tested in exactly one fold,
  re-checked at 50 rows with 7 random groups. Stress: a 50,000-row shuffled 5-fold split in **13 ms**
  with coverage and overlap verified across all 50,000 rows.
- **[test]** `tests/bkit_protocol_test.b` grown to **39 assertions**: `GridSearchCV` and
  `RandomizedSearchCV` join the estimator registry (they *are* estimators, `clone()` isolation
  included), and the **splitter** contract is checked there too, so the package-wide tier 3 covers
  both protocols and a reader asking "what conforms to what" has one file to open.
- **[test]** **Fourteen mutations, and four of them initially passed** — which is the entire case for
  running a battery rather than reading the test and nodding:
  * *re-splitting per candidate* was unfalsifiable against an **unshuffled** splitter, since two
    calls to `split()` return the same folds. The test now fits with `KFold(shuffle=true)` and
    reproduces the folds by re-seeding.
  * *sampling with replacement* in `RandomizedSearchCV` survived one lucky seed — drawing 4 of 5
    distinct happens about 19% of the time. The check now runs ten seeds, where replacement would
    be a 6-in-100-million coincidence.
  * *removing the shuffle* from `train_test_split` passed every assertion about sizes, disjointness,
    coverage and reproducibility, because all of those hold for an unshuffled split. The test now
    asserts the number of **distinct** test sets across seeds.
  * *removing the rare-class floor* from the stratified split passed, because at `test_size=0.5` a
    2-member class already rounds to 1 and the floor never fired. The test now exercises 0.2 and 0.9,
    the fractions where the floor and the cap actually do the work.
- **[perf]** `_complement` and `splitter_check` were passing `nd_put` two **Bantu lists** built by
  `nd_to_list`, where the kernel takes index arrays and a scalar directly: 29 ms against 1 ms for
  100,000 indices, best of five on an idle machine. That path runs once per fold per candidate of a
  grid search. The 50,000-row split went 17 ms → **13 ms**; the first reading after the change said
  32 ms, which was the rest of the suite running beside it — the figure here is from an idle
  machine, like every other number in this log.
- **[bug fix]** **Two of this file's own assertions were wrong, not the code.** A guessed optimum
  (`the best shift is 0`) held for a coarse grid and failed for a finer one — the true optimum on ten
  rows is wherever the per-fold train means sit relative to the test means. It is now the **argmax of
  a hand-recomputed table**, which is the property a search actually owes. And a 400-byte residue
  after 200 splits was read as a leak; it is exactly one 5-fold split of ten rows, still referenced
  by the loop variable. The test now runs 200 and 800 iterations and asserts the residue is **equal**,
  then drops the reference and asserts it reaches zero — both stronger than "did not grow".

## Phase K4 — `metrics.b`

Classification, probability, regression and ranking metrics, 701 lines. Design:
[`../docs/bkit-architecture.md`](../docs/bkit-architecture.md) §5; decisions K25–K30.

- **[feature]** classification: `confusion_matrix`, `accuracy_score`,
  `precision_recall_fscore_support`, `precision_score`, `recall_score`, `f1_score` (per class,
  `micro`, `macro`, `weighted`), `classification_report`, `classification_report_text`, `labels_of`.
- **[feature]** probability: `roc_curve`, `roc_auc_score`, `precision_recall_curve`,
  `average_precision_score`, `log_loss` (binary and multiclass), `brier_score_loss`.
- **[feature]** regression: `mean_squared_error`, `root_mean_squared_error`,
  `mean_absolute_error`, `r2_score`, `explained_variance_score`,
  `mean_absolute_percentage_error`.
- **[feature]** ranking: `ndcg_score` (with `k`), `mrr_score`, both over a single query or a batch.
- **[design]** **`metrics.b` depends on nothing but numba** — not even on `base.b`. Nothing here is
  an estimator, so the protocol has nothing to say about them and tier 3 does not apply. Same
  decision as [K7] and for the same reason: a user who wants `f1_score` should not import scalers,
  splitters and a gradient booster to get it. The corollary is that `base.b`'s `score()` does **not**
  delegate here — the dependency would run the wrong way.
- **[design]** **Every classification metric is a view of one confusion matrix** (K26). The payoff
  is not arithmetic saved, it is that the zero-division rule lives in **one place**: with four
  independent implementations a never-predicted class could be handled one way by `precision` and
  another by `f1`, and the macro average would disagree with the array it was averaging. The matrix
  is built with no loop over samples — encode, `true·k + pred`, `nd_bincount`, reshape — three
  kernel calls for any number of rows. **The orientation is contractual**: rows true, columns
  predicted, and the test asserts cell `[0][1]` rather than only the row sums, because transposing
  it silently swaps precision and recall.
- **[design]** **A zero-support class scores 0, not `NaN`** (K27). `NaN` propagates into the macro
  average, so one absent class would turn the whole model's score into `NaN` — and `NaN` is a
  perfectly good `f64` that nothing downstream objects to. Raising was the other option and is too
  strict: a class missing from one cross-validation fold is ordinary.
- **[design]** **The curves emit one point per *distinct* threshold** (K28). That is what tie
  handling means, and the property it buys is stated and tested: with scores `[0.8, 0.8, …]` a curve
  advancing one sample at a time traces a staircase through points no threshold can produce, and its
  area depends on the order the equal scores arrived in — **the AUC would change when the input was
  shuffled**. A test asserts the AUC is identical when two tied samples swap places.
- **[design]** **`roc_auc_score` is the trapezoid; `average_precision_score` is the step sum**
  (K28). Different numbers on the same data — 0.625 and 0.583 on the hand case — and scikit-learn
  uses exactly this pair. The trapezoid is optimistic on a curve that steps *down*, which a PR curve
  does. The test asserts both and asserts they differ, so neither can drift into the other's rule.
- **[design]** **`r2_score` raises on a constant target; MAPE clamps instead** (K30). The asymmetry
  is deliberate: a constant target breaks R² for *every* sample, while one zero among thousands of
  targets breaks one term of a mean. `base.b`'s `score()` already raised before this file existed,
  and one package should not disagree with itself about the same formula.
- **[design]** **`classification_report` returns a dict; the text is a separate function** (K29). A
  metric whose only output is a formatted string can only be tested by matching that string, and
  then a column-width change is a test failure — which trains whoever hits it to update the expected
  string rather than check the numbers.
- **[feature]** **C5 resolved: the metric moved, the delegation did not** (K25).
  `bkit.metrics.accuracy_score` is the home. `btorch.datasets.accuracy` keeps its implementation,
  marked deprecated, because btorch is **1.0.0 and gated on MNIST accuracy** while bkit is 0.x and
  changes every phase — delegating would let a bkit release break a btorch install, which is [E2]'s
  own blast-radius argument. They are not even the same function (btorch's takes logits), so a
  **differential test** asserts they agree on a hand case and on 200 random rows of 10 classes.
  That test, not a shared implementation, is what stops them drifting.
- **[test]** `tests/bkit_metrics_test.b` — **123 assertions**, tiers 1, 2 and 4. **Every reference
  value is derived independently of the implementation:** the 3-class confusion matrix written out
  cell by cell; the tie-handled AUC cross-checked against the **Mann–Whitney statistic** over the
  four positive/negative pairs with a tie counting ½, which is a wholly different route to 0.625;
  `r2_score` and `explained_variance_score` against scikit-learn's own documented example
  (0.9486081… and 0.9571734…); `log_loss` on a clipped certainty against −ln(1e-15); NDCG against a
  term-by-term DCG/IDCG computation. Stress: 200,000 rows over 10 classes in **49 ms**, with
  `trace/n` asserted equal to `accuracy_score` so the bincount encoding is verified at a size no
  hand case reaches, and 200,000 **tied** scores asserted to collapse to one threshold.
- **[test]** **Sixteen mutations, all sixteen caught** — a transposed matrix, `NaN` for zero
  division, precision and recall swapped, support from the column sum, a support-weighted "macro",
  a curve ignoring ties, each area rule swapped for the other, an unclipped `log_loss`, `r2`
  returning 0 on a constant target, a perturbed `r2` denominator, an unclamped MAPE, a linear NDCG
  gain, a shifted NDCG discount, a 0-based MRR rank, and a length mismatch left to broadcast.
- **[bug fix]** **Two of this file's own expected values were wrong, not the code.** The NDCG
  reference was remembered as 0.9481156 — which is the published figure for those relevances under
  the **linear** gain convention, a different metric with the same name; the exponential gain
  scikit-learn defaults to gives 0.9488107, confirmed term by term. And a line count said 6 where
  eight lines have seven separators.

## Phase K5 — `linear.b`

Seven estimators, 798 lines. Design:
[`../docs/bkit-architecture.md`](../docs/bkit-architecture.md) §6; decisions K31–K35.

- **[feature]** `LinearRegression`, `Ridge`, `ElasticNet`, `Lasso`, `LogisticRegression`
  (binary and multiclass), `SGDRegressor`, `SGDClassifier`. Each reports `coef_`, `intercept_`,
  `n_iter_` and `converged_`; the classifiers add `decision_function` and `predict_proba`.
- **[design]** **The Gram matrix is never formed, and the measurement is the argument** (K31). The
  roadmap asked for "`solve` for well-conditioned"; on a degree-9 polynomial design over 60 points
  the condition number of `X` is 2.6 × 10⁶ and of `XᵀX` is **6.9 × 10¹²**, and the normal equations
  came out **a million times** less accurate — 2.8 × 10⁻³ against 1.2 × 10⁻⁹, a 0.3% coefficient
  error with nothing in the output to show it. `lstsq` cost 41 ms against 7 ms on a 5000 × 60
  problem, once per fit. Not made conditional either: deciding "is this well-conditioned" needs the
  condition number, which costs an SVD — more than the `lstsq` it was avoiding.
- **[design]** **`Ridge` is least squares on an augmented system** (K32): stack `√α` rows under the
  design and zeros under the target. The Gram matrix is never formed, **the intercept is excluded by
  construction** rather than by remembering to, and it explains what the penalty does numerically —
  the added rows raise the smallest singular value to at least `√α`, so Ridge is *better*
  conditioned than the fit it regularises, which is why a closed form is safe here and not in K31.
- **[design]** **The iteration cap is audible, on stderr, and does not raise** (K33). Coordinate
  descent that stops at `max_iter` returns coefficients that are *wrong*, not approximate. So
  `converged_`/`n_iter_` are on the estimator **and** the cap warns through `eprint` — stderr, not
  stdout, because a diagnostic a pipeline's output can swallow is one nobody reads. Raising was
  rejected: hitting `max_iter` on hard data is normal, and an exception mid-cross-validation reports
  nothing at all. A flag alone was rejected too — nobody checks a flag they do not know exists.
- **[design]** **`LogisticRegression` is binary IRLS with multiclass one-vs-rest** (K34), and what
  OvR costs is stated precisely: not *accuracy* — `argmax` over `k` fits agrees with a softmax fit
  on the great majority of points — but **calibration**, since the `k` sigmoids do not sum to 1 and
  `predict_proba` normalises them. The L2 penalty is not optional: on separable data the unpenalised
  likelihood is maximised at infinite coefficients, so a test drives `C = 1e6` on perfectly
  separable clouds and asserts the coefficients stay finite.
- **[design]** **`SGD*` is mini-batch, and that is a different optimiser from scikit-learn's**
  (K35). True per-sample SGD is `n` interpreter iterations an epoch at ~1 µs a step. The honest
  consequence is stated — the optimisation path differs, so the coefficients differ — and what is
  asserted instead is the property that matters: `SGDRegressor` reaches `LinearRegression`'s
  closed-form coefficients to **8.8 × 10⁻⁹**. An optimiser is correct when it finds what the closed
  form already knows.
- **[bug fix]** **`Lasso` was applying no L1 penalty at all, and reporting a clean convergence.**
  `_Lasso.init` replaced `params_` with a dict that omitted `l1_ratio`, so `fit` read **null** — and
  in Bantu `alpha * null` is **0**, silently. Lasso therefore ran as pure L2 and produced **zero**
  sparsity where `ElasticNet(α, 1.0)` zeroed 8 of 10 coefficients. Fixed by reading the ratio
  through an overridable `_rho()`, **and** by validating that `alpha` and `l1_ratio` are numbers, so
  the whole class of "missing key plus null-as-zero arithmetic" becomes an error with a message
  instead of a plausible wrong answer. The test pins the equivalence `Lasso ≡ ElasticNet(α, 1)` to
  1e-14 so it cannot come back.
- **[perf]** **`SGDRegressor` now centres `X` and `y`** and recovers the intercept afterwards, like
  `ElasticNet`. Writing the "intercept is not penalised" test found why it must: learning the
  intercept by descent starts it at 0 and makes it travel to `mean(y)`, while the step decays as
  `1/(1 + η₀·α·t)` — a strong penalty throttles exactly the parameter with furthest to go. Measured
  at `α = 5` on a target centred at 50, the intercept reached **37.1** after 600 epochs and was
  still climbing; centred, it lands at 49.979 against a mean of 49.978.
- **[bug fix]** **The SGD step size was coupled to the penalty, and it throttled convergence**
  (K36). The schedule was `η₀ / (1 + η₀·α·t)`, which is broken at *both* ends. At `α = 2` on a
  90/10 classification the intercept read −0.754 after 400 epochs, **−0.938 after 4,000 and −1.046
  after 20,000** — still moving, never reporting convergence — against the **−2.069** that IRLS
  gives for the same objective. The optimiser was not slow; it was governed by the one parameter
  that has nothing to do with step length. At `α = 0` it never decayed at all. Centring `X` and `y`
  hid this for the regressor's *intercept* by computing it in closed form, but the coefficients and
  the whole of `SGDClassifier` were still throttled.

  `learning_rate` is now explicit — `"invscaling"` (default, `η₀/(1+t)^power_t`), `"constant"`,
  `"optimal"` — and the default depends only on `t`. The same `α = 2` fit reaches **−2.031 in 400
  epochs**. `"optimal"` is the old rule, kept rather than deleted because it is correct for a
  strongly convex objective and is scikit-learn's name for it; it now requires `α > 0`.
- **[design]** **`invscaling` over `constant` was settled by measurement, not taste.** Noiseless
  target: constant 8.4e-9 at 175 epochs, invscaling 6.1e-4 at 400 but **9.7e-11 at 2,000** — slower
  early, *not* capped. Noisy target (σ = 0.5): constant 7.8e-3, invscaling **2.1e-3**, nearly four
  times closer, because the decay is what lets the iterate settle instead of bouncing around the
  optimum. Real targets are noisy. The usual robustness argument for decay was also tested and
  **does not hold here** — neither schedule diverged at `η₀ = 5` — so it is not claimed.
- **[test]** `tests/bkit_linear_test.b` — **135 assertions**, tiers 1–4. Differential against closed
  forms throughout: exact coefficient recovery on noiseless data, Ridge's augmented form against the
  textbook `(XᵀX + αP)` solve to 1e-9, and the K31 gate asserting **both** that `lstsq` stays under
  1e-6 *and* that the Gram route exceeds 1e-4 on the same data, so the decision is re-derived on
  every run. Plus a cross-module check that `base.b`'s `score()` and `metrics.b`'s `r2_score` — two
  independent implementations, since `base.b` deliberately does not delegate — agree to 1e-14.
  Stress: 20,000 × 20 in 31 ms (LinearRegression), 30 ms (Ridge), 28 ms (Lasso) and 356 ms
  (LogisticRegression, 11 IRLS iterations).
- **[test]** **Twenty-three mutations, four of which initially passed** — and all three were properties
  asserted in *shape* but not in *substance*:
  * a **hard** threshold instead of a soft one kept the sparsity and lost the shrinkage, so every
    assertion passed. The test now compares the surviving coefficients against least squares **on
    the support Lasso selected** — L1 shrinks, it does not merely select.
  * `ElasticNet` returning `mean(y)` as the intercept instead of `mean(y) − x̄·b` passed, because
    nothing asserted the intercept against a known value. It now checks a target shifted by 100.
  * penalising SGD's intercept passed, because `maxabs(coef)` says nothing about it. Asserting it
    needed the **mechanism, not a metric**: with 90/10 classes the bug *raises* accuracy from 90.3%
    to 96.7%, since the model then predicts the minority class at roughly its true rate instead of
    never.
  * hardcoding `power_t` passed, because the two fits being compared drew **different mini-batch
    shuffles** from the global RNG — the assertion was measuring RNG variation, not the parameter.
    It now re-seeds before each fit, with a companion assertion that the same seed and schedule
    give bit-identical coefficients, which is what makes the comparison mean anything.
- **[test]** `tests/bkit_protocol_test.b` grown to **49 assertions**: the registry carries the seven
  real estimators and a `GridSearchCV` over `Ridge`, so the contract is checked against models with
  a right answer of their own. The `ConstRegressor` double stays, with a new job — as the simplest
  conforming estimator, it tells a fault in the *check* apart from a fault in a model's arithmetic.

---

## Phase K6 — `Pipeline` + `ColumnTransformer`

`bkit/compose.b`, 512 lines. `Pipeline` already shipped in K1; this phase adds
`ColumnTransformer` and settles the 50-stage Pipeline stress K1 deferred.

### Added

- **[feature]** **`ColumnTransformer([[name, estimator, columns]], remainder)`** —
  different preprocessing per column, concatenated into one matrix. Columns are
  integer indices, arctic column **names**, or a mix; `remainder` is `"drop"`
  (default) or `"passthrough"`.
- **[feature]** **`output_ranges()`** returns `[name, start, stop)` per spec plus
  the remainder, and **`n_features_out()`** the total. Not decoration: a
  40-column one-hot block is otherwise unattributable, and *"which input
  produced feature 37"* is the first question anyone asks of a fitted
  composition. Both require a fit, because the widths are **learned** and there
  is no honest answer before one ([K41]).
- **[feature]** **`"<name>__<param>"`** in `get_params`/`set_params`, so
  `GridSearchCV` can tune inside a composition — the same addressing `Pipeline`
  uses, with the same refusal of an unknown key ([K24]): a search that silently
  ignored one would report a best score for a configuration it never tried.
- **[feature]** `get_transformer(name)`, which names what it *does* have when it
  cannot find what was asked for.

### Design

- **[design]** **It is not in `base.b`** ([K37]). `ColumnTransformer` is the only
  part of bkit that knows `arctic` exists, and `base.b` is what `btext` imports
  for the contract alone — importing the estimator protocol must not drag a
  dataframe package behind it. Composition splits by **dependency**, not by
  subject matter.
- **[design]** **`compose.b` does not `include` arctic.** It duck-types: a frame
  is an instance carrying `names`, `cols` and `ncols`, and reading an undefined
  field on an instance yields null rather than raising — the mechanism btorch's
  `is_dataset` uses. So `arctic` is an **optional** dependency in
  `package.json`; bkit works fully without it and only the by-name path needs it.
  *Rejected:* a marker field added to arctic's `DataFrame`. It is the cleaner
  test, but this is the ecosystem's only consumer, and asking another package to
  carry a flag for one consumer inverts the dependency this decision is about.
- **[design]** **Names and indices resolve to integer positions against one
  materialised matrix** before any transformer runs. So there is exactly one code
  path after resolution and the frame case cannot drift from the array case — and
  a frame is converted **once**, not once per spec.
- **[design]** **Blocks concatenate in spec order, not column order** ([K38]). A
  caller can predict spec order; they cannot predict column order, because a
  `OneHotEncoder` on one input emits an arbitrary number of outputs.
- **[design]** **`fit` delegates to `fit_transform`**, not the reverse ([K41]).
  Fitting already computes every block, so the base class's default would do all
  the work twice and let a stateful transformer see the data twice per fit.

### Fixed before it shipped

- **[design]** **A reordered frame is refused at transform** ([K40]). Names
  resolve to positions *at fit time*, so a frame whose columns were reordered in
  between would transform the wrong data — at the right shape, with plausible
  numbers. `transform` therefore compares the fitted column **names**, not just
  the count. Same reasoning for a block whose width changes between fit and
  transform: it would shift every later block's meaning, so it is reported with
  both widths and the offending spec's name.
- **[design]** **The same column twice in one spec is refused** ([K39]) — it
  would transform the column twice and silently widen that block by one. Across
  *different* specs it is legal and useful (a column both scaled and binarised),
  which is why the check is per spec. `remainder` is the complement of the
  union, emitted in **ascending index order** so the passthrough block carries
  X's own order rather than a dict's.

### Gates

- **[test]** `tests/bkit_compose_test.b` — tiers 1, 2 and 4. The central gate is
  **exactness**: the composition is built twice, once through
  `ColumnTransformer` and once by slicing and concatenating by hand, and
  required to agree to 1e-12 with the same column order. A composition that is
  merely plausible — right shape, reasonable numbers — is the failure worth
  building against, because a block in the wrong place produces a model that
  trains fine and predicts from the wrong features.
- **[test]** The **no-refit-leak** assertion: a `ColumnTransformer` fitted on one
  matrix and applied to another with a different mean must *not* recentre it.
  The classic silent leak, asserted directly rather than assumed.
- **[test]** **50-stage Pipeline** (deferred here from K1). Standardising an
  already-standardised matrix is idempotent, so after fifty passes the output
  must still be exactly centred and unit-variance — which also proves no stage
  was skipped — with `set_params` reaching the fiftieth step by name.
- **[test]** **20 specs over `[5000, 30]`** with a passthrough tail, asserting
  the tail is X's own columns untouched and that `output_ranges()` places the
  twentieth block and the remainder correctly.
- **[test]** `tests/bkit_protocol_test.b` grown to **38 registry entries** with
  three `ColumnTransformer` modes. The protocol does not know it composes, which
  is the point of putting it there.

---

## Phase K7 — `tree.b` and the `tree_` native family

The first native work in this package. Five kernels plus five accessors in
`bantu-src/compiler/src/tree_native.hpp`, and `bkit/tree.b` on top.

### Added

- **[feature]** **`tree_bin`** — bins from the **sorted distinct** values, at
  most 256, with `NaN` getting a dedicated bin (the highest). A 7-distinct-value
  integer column gets 7 bins, not 256 mostly-empty ones; a constant column gets
  exactly 1. Quantiles of the raw array would put several edges between the same
  pair of values and leave empty bins that every later scan still walks.
- **[feature]** **`tree_hist`** — the hot loop: one byte load and three float
  adds per (sample, feature). Output `[features, bins, 3]` holding
  `(Σg, Σh, count)`.
- **[feature]** **`tree_best_split`** — a running prefix per feature, scanned
  **twice** so the `NaN` direction is learned ([K44]).
- **[feature]** **`tree_partition`** — a stable index partition; `bin = -1` is
  the missing-vs-present split.
- **[feature]** **`tree_predict`** — traversal over a `[n_nodes, 7]` array with
  **raw-value thresholds**, so a fitted tree is one matrix and prediction needs
  no bin table at all ([K46]).
- **[feature]** `DecisionTreeRegressor` and `DecisionTreeClassifier`, and
  `tree_` registered in `kNativeFamilies` so an unbound `tree_*` self-diagnoses
  as a stale binary rather than a bare `Undefined variable`.

### Design

- **[design]** **One criterion serves both estimators** ([K42]). Both set
  `g = −y`, `h = 1` and use the same regularised gain K8's booster will. For
  regression that **is** variance reduction (scikit-learn's `squared_error`
  times ½, leaf = the mean). For binary classification it is **proportional to
  Gini decrease** — `Σ n_m·Gini_m decrease = 4 × gain` — so the argmax is
  identical to `criterion="gini"`, and the leaf is the positive-class
  proportion, i.e. the probability. The derivation is in the file header; the
  test computes a Gini table from counts and asserts both the cut and the
  factor 4.0.
- **[design]** **The kernels live in numba's translation unit**, with their own
  file, prefix and registration function. Crossing the deliberately narrow
  `ndarray_api.hpp` wall would mean exporting X as a flat `vector<double>` —
  **800 MB of copy for 1M × 100**, to feed kernels whose purpose is to replace
  those doubles with one byte per cell. Sharing the TU also gets `-O3
  -ftree-vectorize`, which the histogram loop is the reason for.
- **[design]** **The `NaN` direction is learned** ([K44]), not fixed. The `NaN`
  bin is the highest index, so a plain prefix scan would always leave it in the
  right child — and `tree_partition` was first written defaulting it *left*, so
  the partition would have sent missing rows the opposite way from the scan that
  chose the split. Found by writing the design comment, before any of it ran.
- **[design]** **Multiclass is one-vs-rest**, which is a real difference from
  scikit-learn's single K-way tree, and is documented as one rather than
  claimed as a match ([K43]). True multiclass Gini needs the gain summed over K
  histograms *before* the argmax — a `tree_best_split` taking a list, recorded
  as a K8 item.

### Fixed

- **[bug fix]** **The bin assignment was off by one, and the gain was still
  right** ([K48]). `upper_bound` pushed a value sitting exactly on its own edge
  one bin right: 8 distinct values binned as `[1,2,3,4,5,6,7,7]` instead of
  `[0..7]`, bin 0 empty and the top two collapsed. The *gain* was correct —
  the same row partition was reachable one bin over — so the brute-force
  reference agreed, the `4 × gain` ratio held and `tree_hist` matched its own
  reference. Only the independently computed **bin index** disagreed, and that
  index is what the threshold lookup uses, so predictions were one row wide.
  `lower_bound` is the correct search for a "values ≤ edges[b]" contract.
- **[design]** **The hessian guard skips before the division** ([K45]). A
  saturated logistic gives `h → 0`; the tempting reasoning is that a resulting
  `NaN` gain loses every comparison and is therefore safe. The real risk is the
  other branch: `G²/0` with `G ≠ 0` is `+inf`, which **wins** every comparison
  and would pick a split on a child with no weight in it, with maximum
  confidence. Written `!(H + λ > 0.0)` so a `NaN` total is caught too.

- **[perf]** **Histogram subtraction**, the single largest optimisation in the
  algorithm: a child's histogram is its parent's minus its sibling's, so the
  grower builds the **smaller** child from the data and subtracts for the
  larger. Roughly halves the histogram work per level. It was designed as a K8
  item and moved here because the grower is *shared* — K8 adds a boosting loop
  over it, so this is work K8 neither repeats nor has to re-gate.
- **[test]** **The gate I wrote for it in the design was false, and the design
  said so before the code ran.** §8.6's first draft asserted a subtracted
  histogram would be **bit-identical** to one built directly. `tree_hist`
  accumulates in index order, so the parent sums rows `0..n−1` while the sibling
  sums a subsequence, and `sum(A ∪ B) − sum(A) = sum(B)` does not hold exactly
  in floating point. The gate is now per **channel**: the **count** channel must
  be bit-identical via `nd_tobytes` (counts are integers held in doubles, all
  exactly representable, so any difference is a bug and not rounding), while the
  gradient and hessian channels must agree to `< 1e-12` **and the worst observed
  difference is printed**, so a drift from 1e-16 to 1e-9 shows up as a number
  rather than as a still-passing assertion. Both differences are currently
  exactly 0, which is a property of a seven-row matrix and not evidence the
  identity holds. Plus the property the grower actually depends on and which
  neither channel assertion implies: both histograms must choose an
  **identical split**. (K54.)
- **[bug fix] The kernel guarded the gain; the caller divided by zero.** [K45] is
  the record of `tree_best_split` refusing a child with `!(H + λ > 0)`, because
  `G²/0` with `G ≠ 0` is `+inf` and **wins** every comparison. `_leafValue`
  computes `−G/(H + λ)` from the **same two quantities** and had no guard at all
  — and a scalar `/0` *raises* in Bantu, so a node whose hessians sum to zero
  threw `Division by zero` out of the middle of `_grow` with no node, no tree and
  no indication of which node. Not reachable from K7's estimators (both set
  `h = 1`, so `H = n ≥ 1`); **reachable from K8's booster**, where logistic
  `h = p(1−p)` underflows to 0 across a saturated node and `reg_lambda` defaults
  to 0. Fixed in the *grower*, because the grower is shared and the defect is the
  asymmetry between a kernel that guards and a caller that does not. **Zero is
  the right value, not a fallback**: at `H + λ = 0` the second-order
  approximation has no curvature and its minimiser is unbounded, which is an
  artefact of the approximation — XGBoost's `CalcWeight` returns 0 here and
  LightGBM does the same. Written `!($d > 0.0)` so a negative or `NaN` total is
  caught too. (K55.)
- **[test]** The mutation was run: removing that guard turns the section **red
  with 4 failures** instead of aborting the file. The gate also answers [K52]'s
  open question directly — a grower that can find no split returns a
  **single-leaf `[1,7]` tree** with value 0, and the same `g`/`h` with
  `reg_lambda = 1` grow a real tree, so the guard is about `H + λ` and not about
  refusing the data. Hessians that *cancel* to exactly zero are covered
  separately, since that is the path a negative-hessian loss reaches.
- **[test] A throw now names its section.** [K52]: two `tree_` mutations were
  "caught by aborting", which counts as caught and is still worse than a clean
  red — the battery learned nothing about which assertion fired and every later
  section went unreported. Each of `bkit_tree_test.b`'s **33 sections** is
  wrapped in its own `try`, and a throw prints
  `FAIL [section aborted] <section>` plus the message, as one failure, with the
  remaining sections still running. Cheap only because a top-level `try` in
  Bantu **does not open a new scope for variables** (verified, not assumed), so
  the file's straight-line flow is unchanged; converting 132 assertions to
  closure-taking `safe_ok` was rejected, because Bantu closures capture by
  reference. **A `def` inside a `try` *is* scoped to it**, though — four
  reference helpers vanished and failed as `Undefined variable` from a different
  block, caught on the first run by the mechanism being added, and all four are
  now hoisted. (K56, language trap 33.)

### Gates

- **[test]** `tests/bkit_tree_test.b` — tiers 1, 2 and 4. Every kernel's
  reference is computed **in Bantu from the bins the kernel itself produced**,
  never by calling the kernel a second way. `tree_hist` is required to be
  **bit-identical** to a pure-Bantu triple loop, not merely close.
- **[test]** The hessian saturation case is driven deliberately: every hessian
  zero with `λ = 0` must find no split and report gain `0`; the *same* histogram
  with `λ = 1` must split normally, which proves the guard is about `H + λ` and
  not about refusing the data.
- **[test]** `tree_predict` refuses a malformed tree instead of walking off it —
  wrong layout, a feature X does not have, a child index outside the tree, a
  self-child, and a two-node **cycle**, which raises rather than hanging. The
  four-minute hang in K2 is the precedent for not trusting a loop to terminate
  because its data "should" be acyclic.
- **[test]** **`tree_bin_bytes` asserts K1's whole justification directly**:
  200000×50 binned is exactly **10,000,000 bytes**, not 80,000,000.
- **[test]** A tree draws no randomness, so the suite asserts two fits are
  bit-identical **without** reseeding, not just with.
- **[test]** `tests/bkit_protocol_test.b` grown to **42 registry entries** —
  the first estimators backed by native kernels, so the protocol is also
  checking that a `tree_*` handle survives clone/get_params/set_params without
  the contract knowing a handle exists.

---

## Phase K8 — `ensemble.b`, gradient boosting

- **[feature]** `GradientBoostingRegressor` and `GradientBoostingClassifier`,
  over **four objectives**: `squared_error`, `logistic`, `softmax` (K trees per
  round, normalised jointly) and `poisson`. Shrinkage, row subsampling
  (`subsample`), column subsampling (`colsample`), early stopping on a held-out
  fold, `staged_predict`, and gain / split-count feature importances.
- **[design] It adds NO kernels, and that was the point of §8.5.** K7's five
  `tree_` kernels are sufficient, checked feature by feature: histogram
  subtraction is one `nd_subtract` per node; row subsampling is an index array
  `tree_hist` already accepts; column subsampling zeroes the unselected
  features' rows in the histogram the *scan* sees, so a zeroed feature has
  `G = H = n = 0` and fails both `min_data` and the hessian guard — two
  independent reasons, neither a special case in C++. The new work is the
  boosting *loop*, and the per-element work stayed in C++.
- **[design] ONE grower, exposed rather than copied** ([§8.6]). Every tree is
  grown by `tree.TreeGrower`, which *is* `_TreeBase._grow` unchanged. The gate
  is an identity, not a similarity: **one round at `learning_rate = 1.0` is
  bit-identical to `DecisionTreeRegressor` fitted to `y − ȳ`**, because
  `squared_error` sets `F0 = ȳ` and `g = F0 − y = −(y − ȳ)`, `h = 1`, which is
  exactly what the tree computes. That ties everything K7 already tests
  (binning, the scan, the `NaN` direction, the threshold lookup) to gates that
  have been broken twelve ways, and leaves this file testing what is new.
- **[design] Parameters arrive as a DICT**, a deliberate break with the rest of
  bkit's positional constructors. A booster has fourteen; a fourteen-slot
  positional constructor is where transposing two arguments is silent and
  unreadable at the call site, and `get_params`/`set_params` have always spoken
  dicts. **An unknown key raises** ([K24]).
- **[design]** Early stopping **splits the validation fold before binning**
  ([§8.7]), so bin edges cannot be chosen using data the model is scored on —
  a quieter leak than training on it, because it inflates the stopping round
  rather than the score. Asserted structurally, not by eye: a 24-row single
  feature bins to 24 bins, and the model trained with a 25% fold reports 18.
- **[bug fix] The hessian clamp was harmful and is gone** (K58). `max(h, 1e-16)`
  was prescribed by the design doc and shipped for about an hour. Measured: at
  `λ = 0` it turns a correctly refused split into **leaves of ±5e15**, because
  lifting `h` makes `H + λ` clear both [K45]'s and [K55]'s guards by a hair. It
  was invisible at the defaults, which is the worst place for a hazard to hide.
  Only *invalid* hessians are now corrected — negative and `NaN`, both to 0 —
  and a genuine zero is passed through to the guards. `np.maximum` propagates
  `NaN` rather than dropping it like IEEE `fmax`, so that case needs its own
  `where()`; asserted, not assumed.
- **[bug fix] `get_params()` returned the live dict**, so every bkit estimator
  was writable through a read accessor (K59): `$p = $m.get_params();
  $p.with_mean = false;` reconfigured a **fitted** model and left `fitted_`
  true, and `$p.with_men = true` added a parameter — [K24]'s bug reachable
  through the accessor beside the check for it. It returns a copy now. How deep
  that copy goes was **measured**, because Bantu is asymmetric: list values come
  out independent, nested dicts still alias (language trap 35).
- **[feature]** `feature_importances_("gain" | "split")` on the boosters **and**
  on K7's trees. The gain is recorded as a grower side output, not an eighth
  node column: a node is 7 doubles and `tree_predict` strides by 7 in C++, so an
  eighth would be a native change *and* a format break, to carry a number
  prediction never reads (K57). Folded in per tree, because `_grow` overwrites
  its side outputs — read at the end, a K-class model would report only its last
  tree.

### Gates

- **[test]** `tests/bkit_ensemble_test.b` — **112 assertions**, tiers 1, 2 and 4,
  every section wrapped in its own `try` so a throw is attributed (K56).
- **[test] The first run passed 82 of 82, and 11 of the first 17 mutations
  survived it** (K60). The five causes are written up individually, because they
  repeat: a gate proving a *disjunction* (either subsample **or** colsample being
  drawn satisfied it); an inequality true of anything in range
  (`best_round <= n_rounds` holds for a `best_round` that kept the **worst**
  round); an output **recomputed on the read path** (`predict_proba` recomputes
  the softmax, hiding a wrong training gradient); a quantity that only changes
  the **rate** (dropping the hessian to 1 is gradient descent, which still
  converges); and a **branch no test entered** (nothing early-stopped a
  classifier, so a flipped log-loss sign was unreachable).
- **[test]** The replacements pin numbers rather than directions: the hessian is
  checked against **hand-derived leaf values** (a depth-1 logistic tree on
  `[0,0,0,0,1,1,1,1]` has `F0 = 0`, `p = 0.5`, `h = 0.25`, so its leaves are
  exactly **−2 and +2**; with `h = 1` they would be ∓0.5), and softmax by the
  fact that `F0 = 0` makes `p_k = 1/K` **exactly** — a value a per-column
  sigmoid cannot produce for K ≠ 2. Final: **19 mutations, 18 caught, 1
  equivalent**, and the equivalent one led to K59.
- **[test]** The early-stopping fixture was replaced with one that improves,
  stalls and improves again, **and the test asserts that property of the
  fixture**, so it fails loudly if it stops exercising the patience reset.
- **[test]** `tests/bkit_protocol_test.b` grown to **47 registry entries** — the
  first estimators built from a parameter *dict*, so the protocol also checks
  that construction style against `protocol_check`'s round-trip.
- **[test]** `tests/bkit_tree_test.b` grown to **156**: `TreeGrower` is asserted
  bit-identical to `DecisionTreeRegressor` on the same `g`/`h`, the column mask
  is asserted to confine every split to the allowed features *and* to restore
  the unmasked tree bit-for-bit when cleared, and a row subsample over a
  constant target must grow a single leaf.

### The `tree_` kernel battery, all twelve (K62)

- **[test]** [K53] recorded three mutations before the battery was interrupted;
  re-run in full against the restored kernel and the 156-assertion suite,
  **10 of 12 were caught** — and T2/T3, which could previously only be recorded
  as *"caught by aborting"*, now report `PASS: 121 FAIL: 35` and
  `PASS: 123 FAIL: 33` with the throwing section named ([K56] working).
- **[test] T4 found a guard hidden behind a guard.** `tree_best_split` checks
  the hessian twice — once per **feature** on the parent total, once per
  **candidate cut** on `H_L`/`H_R` — and removing the second left all 156
  assertions green. The only fixture that drove hessians to zero drove *all* of
  them to zero, so the parent total was zero, the first guard skipped the whole
  feature, and the second was unreachable. **A fixture that trips the outer
  guard cannot exercise the inner one**, and the suite had read as though it
  tested "the hessian guard" when it tested one of two. The new gate keeps the
  parent healthy (`h = 1` on three rows) and starves one child (`h = 0` on
  four), so a cut with `H_R = 0` and `G_R = 2` would score `+inf` and win —
  [K45]'s failure mode reached by the path [K45]'s own test could not take.
- **[test] T5 found a tie nobody had made.** Changing the scan's
  `gain > bestGain` to `>=` moves the winner from the first maximal cut to the
  last, and nothing in 156 assertions produced two equal gains. Not cosmetic:
  duplicate and perfectly correlated columns are ordinary, and which one a tree
  splits on decides what `feature_importances_` reports and whether two fits
  agree on their *structure* and not only their predictions. The gate uses two
  identical columns and **asserts the tie is real** — the two features' best
  gains compared bit for bit — before asserting the lower index wins; without
  that first half the second proves nothing.
- **[test]** `tests/bkit_tree_test.b` 156 → **166 assertions**.

### Random forests (K63)

- **[feature]** `RandomForestRegressor` and `RandomForestClassifier`, over the
  same `tree.TreeGrower`. Bootstrap sampling, per-node feature sampling, and
  averaging **in probability space** — a classifier averages `predict_proba` and
  then takes the argmax, because a majority vote over hard labels discards every
  tree's confidence and is a different, worse estimator.
- **[bug fix] Feature sampling must be per NODE, not per tree** — the first
  draft reused the booster's `set_features()` hook, which draws once per tree.
  That is a random **subspace** ensemble: with 6 features and `k = 2`, a tree
  never sees the one informative feature with probability 2/3, so two thirds of
  the forest fits noise. Measured **R² 0.66 where it should be 0.995**, and a
  held-out MSE *worse than a single deep tree* — inverting the one claim a
  forest exists to make. `set_features_per_split(k)` redraws the mask at every
  node; the mask is three native calls and no Bantu loop
  (`rank(uniform(f)) < k`, [K18]'s construction).
- **[bug fix] `max_features` defaults differ BY TASK**, as scikit-learn's do:
  every feature for a regressor, `sqrt(f)` for a classifier. Using `"sqrt"` for
  both is exactly the configuration measured above.
- **[test] The gate that caught both was the ACCURACY gate, not the structural
  one.** The structural assertion — *"with `max_features = 1` no tree references
  more than one split feature"* — passed, because the implementation was
  per-tree and the assertion described per-tree behaviour; a test written from
  the implementation agrees with the implementation. What failed was `R² > 0.9`
  and *"the forest generalises better than one deep tree"*, both statements
  about what the estimator is **for**. The structural assertion is now inverted
  so it distinguishes the two designs instead of restating one.
- **[test]** §8.8's sharp reproducibility gate: **fitting 20 trees reproduces
  the first 10 exactly**, which rules out a per-tree seed derived from the tree
  count — the natural wrong implementation, and one that looks reproducible
  until `n_estimators` changes. Paired with an assertion that tree 10 differs
  from tree 0, so the first is not vacuous.
- **[test]** With `bootstrap: false` and every feature available a forest draws
  **no randomness at all**, so its trees must be bit-identical to one another —
  the cleanest available proof that the two switches are the only source of
  variation, and it needs no seeding to hold. Turning on either switch alone
  must then make them diverge.
- **[perf]** Out-of-bag coverage is counted with `np.bincount`, one O(n) native
  call per tree. The obvious construction — compare `arange(n)` against the
  sample broadcast the other way — builds an n×n boolean **per tree**: 10¹⁰
  cells for 10,000 rows and 100 trees. Caught by reading the code for cost
  before it was ever run.
- **[test] An average is only checkable by reproducing the average** (K64). The
  forest battery's two survivors were one defect in two costumes: averaging over
  `n − 1` trees (a *uniform scale error*, which `R² > 0.9` absorbs and which the
  classifier's row normalisation **cancels exactly**), and averaging thresholded
  0/1 labels instead of probabilities — a majority vote, the one thing §8.8
  names as wrong, which still classifies easy data perfectly and still produces
  rows summing to 1. The suite had an assertion for every property a vote
  *preserves* and none for the property it destroys. **Normalisation is an
  information-destroying step, and every assertion downstream of one is blind to
  whatever it cancels.** Both are now gated by reconstructing the average from
  the public node arrays — `Σ tree_predict(nodes(t,k), X) / n_trees` — and
  demanding **bit equality**, which no tolerance or accuracy bound can
  substitute for.
- **[test]** `tests/bkit_ensemble_test.b` 112 → **156 assertions**.

### ExtraTrees — one optional kernel argument (K65)

- **[feature]** `ExtraTreesRegressor` and `ExtraTreesClassifier`: the forest with
  one switch flipped, so each node draws **one candidate cut per feature**
  instead of scanning every bin. `bootstrap` defaults to **false**, matching
  Geurts et al. — the randomness is already in the thresholds, so resampling the
  rows as well mostly removes training data.
- **[feature] `tree_best_split` takes an optional `cuts` vector**, one bin index
  per feature; when present, feature `f` considers only `cuts[f]`. One line
  inside the existing loop. **This supersedes [K61]**, which chose a virtual
  `_chooseSplit` override — and whose stated reason for rejecting a kernel
  argument was itself wrong: *"the gain formula would exist in two places"*
  confuses two **call sites** with two **copies**, since two functions in one
  file can share one inline helper. The surviving objection (a second entry
  point duplicates the *validation*, which is where [K44] and [K48] both lived)
  is answered by not adding an entry point at all.
- **[design]** The argument wins on every axis [K61] argued over: one loop means
  one formula; one entry point means one set of validations; the scan stays
  **native**, where the override would have paid `O(max_features)` Bantu
  dispatches per node *and* put the regularised gain in a second **language**,
  free to drift silently. The `NaN` double pass, the guards, the tie-break and
  the 10-slot return are inherited — which matters most here, because a random
  cut is far likelier than a scanned one to leave a child empty or weightless,
  moving [K45]'s and [K55]'s guards onto the hot path.
- **[design] The draw stays in Bantu**, which is a testing decision as much as a
  [C4] one: no RNG in a kernel keeps the forest reproducible from `nd_seed`
  alone, and keeps the kernel **deterministic given its argument**, so a test
  hands it a known vector and asserts the exact split. A kernel drawing its own
  cuts could only be checked statistically.
- **[design] It never clamps.** An index outside `0 .. vbins−2`, or a fractional
  one, raises. Rounding into range would turn an off-by-one in the drawing
  expression into a working forest with a silently biased cut distribution —
  [K48]'s failure mode. The grower therefore computes each feature's value-bin
  count **once per tree** and emits `−1` where there are fewer than two, which
  the scan already treats as "skip".
- **[test]** The gate that ties the two paths together: **forcing each feature
  to the cut the full scan chose reproduces the full scan's 10-slot answer bit
  for bit.** Alongside it, a forced cut's gain is checked against the formula
  recomputed from the histogram's own `cumsum` — an independent route in
  [K48]'s sense, through the histogram rather than through the kernel.
- **[test]** Two gaps the battery found, both invisible beforehand: nothing
  asserted the cuts **vary within one tree** (a per-tree draw is still
  reproducible, still differs between seeds, and still differs from the scanned
  tree — it just cannot put two thresholds on one feature; measured 5), and no
  random-cut fixture contained **NaN**, so counting the NaN bin into the
  value-bin total was equivalent on every existing matrix. 25 seeded trees over
  a NaN-bearing matrix now have to grow without a single out-of-range draw.
- **[test]** `tests/bkit_tree_test.b` 166 → **188**,
  `tests/bkit_ensemble_test.b` 156 → **175**,
  `tests/bkit_protocol_test.b` 51 → **55 registry entries**.

### Permutation importance (K66)

- **[feature]** `selection.permutation_importance(est, X, y, n_repeats?,
  scorer?)` — how much a **fitted** model relies on each feature, measured by
  shuffling that column and watching the score fall. It lives beside
  `cross_val_score` rather than inside `ensemble.b` because it needs `score()`
  and nothing else, so it works on a pipeline, a linear model and a forest
  alike. Returns the per-feature mean and standard deviation over repeats, the
  raw drops, and the baseline.
- **[design] It does not refit**, and that is the definition rather than an
  optimisation: refitting on a shuffled column measures what the model *class*
  can do without the feature, while permuting measures what *this fitted model*
  depends on. An **unfitted estimator is refused** rather than fitted for the
  caller, because fitting it here would silently answer the other question.
- **[design] The column is permuted, not zeroed.** Shuffling preserves the
  feature's marginal distribution and destroys only its relationship to `y`;
  zeroing moves every row somewhere the model may never have seen, which
  measures extrapolation rather than reliance.
- **[docs] What it says about a duplicated column was measured, and the usual
  one-line warning covers the other case.** With two **identical** columns the
  split finder breaks ties toward the lower index ([K62]), so the model never
  reads the duplicate: it scores **exactly 0** while its twin scores **2.03**.
  That is correct, and is not the claim "this feature does not matter". With
  two **correlated but distinct** columns the familiar caveat applies and both
  under-report. It is the clearest demonstration that gain importance and
  permutation importance answer different questions — gain says what the model
  used, permutation says what breaking it costs — so both are reported and
  neither replaces the other.
- **[test] The restore gate is phrased in the run's own quantities**, and the
  first version of it was wrong. Each column must be put back before the next
  is permuted, or every later importance is measured against a model whose
  earlier columns are still shuffled — and all of them still look plausible.
  The obvious gate, re-running one feature in isolation, draws *different*
  permutations once it reseeds, so asserting equality would be asserting noise.
  A missing restore instead charges feature 1 the cost of breaking 0 **and** 1,
  so the gate is that features 1–3 **together** cost under a tenth of feature 0.

### Leaf-wise growth, and the parameter check that was not there (K67, K68)

- **[feature]** `max_leaves` switches the **shared** grower from depth-wise
  breadth-first to leaf-wise. It is **one swap into the queue head**: leaf-wise
  scans the live frontier for the highest gain and swaps that item into the
  position the loop is about to read, so everything below the swap is identical
  under both policies.
- **[design]** What it required was **moving the scan**. Leaf-wise compares
  gains *before* choosing, so a node's histogram, totals and best split are now
  computed when it **joins** the frontier and cached on the queue item;
  `_evalNode` holds that and depth-wise calls it identically, ignoring the
  ranking. Computing the split twice — once to rank, once to use — would have
  put the scan, the feature mask and the random-cut draw in two places.
- **[test] The equivalence gate needs both halves**: with no cap the two
  policies predict **bit-identically** *and* build **different node arrays**.
  Without the second, the first would also hold if the switch did nothing.
- **[test] Split counts are exact, gains are not** — [K54]'s per-channel
  structure again, for the same reason: counts are integers in doubles, gains
  are the same floats summed in a different order, and addition is not
  associative. They agree to **2.1e-16 relative** (one ULP).
- **[perf]** What the policy is for, measured on an asymmetric target (a
  staircase below `x = 0.25`, a plateau above) at an **identical** seven-leaf
  budget: depth-wise reaches depth 3 for a training MSE of **0.1797**,
  leaf-wise reaches depth **4** for **0.1406**. On a *smooth* target the two
  produce the same tree — the advantage is data-dependent and the gate says so.
- **[bug fix] `set_params` validated the KEY and never the VALUE** (K68), found
  by writing a gate for `max_leaves = 1` and watching it pass. Nine invalid
  values driven through it: **five were accepted** — `max_depth = 0` and
  `max_leaves = "many"` among them, each growing a **one-node tree** reported
  as a fit — and the four that were refused were refused **by the kernel**,
  three calls later, naming a kernel the caller never invoked. Every check bkit
  performs in a constructor was bypassable, through precisely the method that
  matters most: **`GridSearchCV` never calls a constructor**, so a grid
  containing `max_depth: 0` scored a degenerate model and reported it as a
  legitimate candidate. That is [K24]'s failure one level down — K24 fixed the
  mistyped key, this is the unchecked value.
- **[design]** Fixed as a **protocol hook**, not a patch per estimator:
  `Estimator.set_params` calls `this._validateParams()` after applying the
  dict, with a **no-op** default so an estimator with nothing to check is
  unaffected. `tree.b` overrides it with the routine its *constructor* already
  calls — the override **is** that routine, not a copy — and `ensemble.b`
  overrides it with `_checkParams`/`_checkForest`. Nothing in the package
  relied on the hole: every suite stayed green.
- **[bug fix]** `TreeGrower.set_max_leaves` assigned before validating and
  reset to `null` on the throw, so a rejected value silently replaced the
  previous cap with "no cap". A refused call that still changes the object is
  worse than either outcome alone.
- **[test]** `tests/bkit_tree_test.b` 188 → **229**,
  `tests/bkit_ensemble_test.b` 197 → **210**, `tests/bkit_base_test.b` 59 →
  **61**. Nine mutations on the policy and the validator, nine caught.

---

## Phase K9 (part 1) — `decompose.b`: PCA and TruncatedSVD

- **[feature]** `PCA` and `TruncatedSVD` over numba's `svd`. `n_components` as a
  count, as a **fraction** (a variance target, resolved at fit), or null for all
  of them; `transform`, `inverse_transform`, `components()`,
  `singular_values()`, `explained_variance()`, `explained_variance_ratio()`.
- **[design] Through the SVD of the centred matrix, never the covariance**
  (K69). Forming `XᵀX/(n−1)` **squares the condition number** — a design matrix
  at 1e8 gives a covariance at 1e16, the whole of double precision — and the
  singular values of `X` are the square roots of that matrix's eigenvalues, so
  the SVD reaches the same answer at half the conditioning. The cost is stated
  rather than hidden: SVD on `[n, f]` is more work than `eigh` on `[f, f]` when
  `n ≫ f`, and a wrong component at 1e-3 is worse than a slow one.
- **[bug fix] `nd_svd` returns `[U, S, V]`, not `[U, S, Vh]`** as NumPy does —
  numba's own linalg test states it (*"V is n-by-k"*, reconstructing with
  `nd_T(V)`), so the transpose belonged here. **It failed almost silently**:
  `U S V` has the right shape, the fit completed, every singular value and every
  variance ratio was correct and summed to 1, and the components looked
  plausible. The assertion that named it was **structural** — *the transformed
  columns must be orthogonal, because that is what a principal component is* —
  where the largest off-diagonal was **0.499**. A comparison against a reference
  built the same wrong way would have agreed ([K48]).
- **[design] Singular-vector signs are pinned**, by scikit-learn's rule (largest
  magnitude entry of each left singular vector positive), with ties broken by
  the lowest index so the rule is **total** — without that the suite is flaky
  rather than wrong, which is harder to notice. `U` is not exposed, so the gate
  goes through `transform`, which returns `U·S`: *the largest-magnitude entry of
  each transformed column must be positive*.
- **[design] `TruncatedSVD` is a different estimator, not a faster PCA.** It
  factorises `X` rather than `X − x̄` because centring a sparse matrix destroys
  its sparsity. The test asserts **both** halves — identical to PCA on centred
  data, and substantially different on uncentred data, where its leading
  singular value is over twice PCA's because the first component points at the
  mean. Only the second says the centring is genuinely absent.
- **[design]** A constant matrix reports **zeros**, not `NaN`: the ratios are
  `0/0`, and `NaN` would propagate into any downstream sum while zeros state the
  truth about that matrix — the same call [K57] makes for an importance vector
  with no splits.
- **[test]** `tests/bkit_decompose_test.b` — **44 assertions**, and **12
  mutations, 12 caught**, including the `V`/`Vᵀ` bug itself (five assertions
  red). `tests/bkit_protocol_test.b` 55 → **59 registry entries**.
