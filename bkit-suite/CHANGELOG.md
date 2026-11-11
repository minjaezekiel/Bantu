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
