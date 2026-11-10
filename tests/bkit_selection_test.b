// ════════════════════════════════════════════════════════════════════════════
//  bkit_selection_test.b — tiers 1, 2, 3 and 4 for bkit phase K3.
//
//  Everything in `selection.b` exists to stop one failure: a score better than
//  the model. So the assertions that matter are the ones that would still look
//  fine if they were wrong:
//
//    * TRAIN AND TEST NEVER OVERLAP, in any fold of any splitter. An overlap
//      grades the model on rows it trained on and the score comes out high.
//    * CROSS-VALIDATION CLONES BEFORE EVERY FIT. Reusing a fitted estimator
//      leaks the previous fold and still produces a plausible number -- so the
//      test asserts the SOURCE estimator is untouched afterwards.
//    * THE REPORTED BEST SCORE EQUALS A HAND-RUN REFIT at those parameters,
//      to the last bit. A search that reported a score it had not computed is
//      the whole hazard of the machinery.
//    * EVERY TRAINING INDEX IS BEFORE EVERY TEST INDEX in TimeSeriesSplit.
//      Shuffling a time series is the classic way to train on the future and
//      be graded on the past.
//    * A MISTYPED GRID KEY RAISES before the first fit, because `set_params`
//      used to invent the parameter and the search then scored every candidate
//      identically and reported a best it never tried (K24).
//
//  The protocol checks are run over a DELIBERATELY BROKEN splitter as well: a
//  check that cannot fail proves nothing.
//
//  Run:  bantu run tests/bkit_selection_test.b
// ════════════════════════════════════════════════════════════════════════════

include "../numba/numba.b"      as np;
include "../bkit/base.b"        as base;
include "../bkit/preprocess.b"  as pre;
include "../bkit/selection.b"   as sel;

$R = {"pass": 0, "fail": 0};
def ok($c, $n) {
    if ($c) { $R.pass = $R.pass + 1; print("  ok    " + $n); }
    else    { $R.fail = $R.fail + 1; print("  FAIL  " + $n); }
}
def eqs($got, $want, $n) {
    if ($got == $want) { $R.pass = $R.pass + 1; print("  ok    " + $n); }
    else { $R.fail = $R.fail + 1;
           print("  FAIL  " + $n + "   got " + str($got) + ", want " + str($want)); }
}
def safe_ok($fn, $n) {
    $v = false;  $err = "";
    try { $v = $fn(); } catch ($e) { $err = str($e); }
    if ($err != "") { $R.fail = $R.fail + 1;
                      print("  FAIL  " + $n + "   raised: " + $err); return null; }
    ok($v, $n);
    return null;
}
def raises_with($fn, $needle, $n) {
    $msg = "";
    try { $fn(); } catch ($e) { $msg = str($e); }
    if ($msg == "") { $R.fail = $R.fail + 1;
                      print("  FAIL  " + $n + " (no error raised)"); return null; }
    if (indexOf($msg, $needle) >= 0) { $R.pass = $R.pass + 1; print("  ok    " + $n); }
    else { $R.fail = $R.fail + 1;
           print("  FAIL  " + $n + "   message lacks \"" + $needle + "\": " + $msg); }
    return null;
}
def lst($a, $want, $n) { eqs(str(nd_to_list($a)), str($want), $n); return null; }
def close($a, $b, $tol, $n) { ok(abs($a - $b) <= $tol, $n); return null; }

// ── two doubles with a tunable parameter ────────────────────────────────────
// K3 has no real estimators to search over yet -- linear.b is K5 -- so these
// are deliberate doubles. `ShiftRegressor` predicts mean(y_train) + shift, so
// its R2 is a known function of `shift` with a maximum at 0: a grid search over
// it has a verifiable right answer, which is the only kind worth testing a
// search against.
class _ShiftRegressor extends Estimator {
    def init($shift) {
        super();
        this.name_ = "ShiftRegressor";
        this.task_ = "regressor";
        $s = $shift;  if ($s == null) { $s = 0.0; }
        this.params_ = {"shift": $s};
        this.mu_ = 0.0;
        this.nfit_ = 0;
    }
    def fit($X, $y) {
        // No ternary in Bantu (language traps, 26), so this is an if/else.
        $ya = [0.0];
        if ($y != null) { $ya = nd_to_list($y); }
        this.mu_   = nd_get(nd_mean(nd($ya, "f64"), null, null), 0);
        $A = $X;
        if (type($A) == "list") { $A = nd($A, "f64"); }
        this.nfit_ = nd_shape($A)[0];
        this.fitted_ = true;
        return this;
    }
    def predict($X) {
        this._check_fitted("ShiftRegressor.predict");
        $A = $X;
        if (type($A) == "list") { $A = nd($A, "f64"); }
        return nd_full([nd_shape($A)[0]], this.mu_ + this.params_.shift, "f64");
    }
    def clone() { return ShiftRegressor(this.params_.shift); }
    def rows_seen() { return this.nfit_; }
    def mu() { return this.mu_; }
}
def ShiftRegressor($shift) { return new _ShiftRegressor($shift); }

print("");
print("  bkit model selection (K3)");
print("  ════════════════════════════════════════");

nd_seed(1);
$X  = nd_reshape(nd_astype(nd_arange(0, 20, null), "f64"), [10, 2]);
$y  = nd([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], "f64");
$g  = nd([0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0], "f64");
$yr = nd([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0], "f64");

// ── 1. KFold, against the stated fold-size rule ─────────────────────────────
print("");
print("-- KFold: the first n mod k folds get one extra row (K22) --");
$kf = sel.KFold(5, false).split($X, null, null);
eqs(len($kf), 5, "5 folds for n_splits=5");
lst($kf[0][1], [0, 1], "fold 0 tests the first two rows");
lst($kf[4][1], [8, 9], "fold 4 tests the last two");
lst($kf[2][0], [0, 1, 2, 3, 6, 7, 8, 9], "and the training side is everything else, in order");
// 10 rows into 3 folds: 10 = 3*3 + 1, so fold 0 gets 4 and the rest get 3
$kf3 = sel.KFold(3, false).split($X, null, null);
lst($kf3[0][1], [0, 1, 2, 3], "10 rows into 3 folds: the first fold gets FOUR");
lst($kf3[1][1], [4, 5, 6], "  the second gets three");
lst($kf3[2][1], [7, 8, 9], "  and so does the third");
raises_with(def() { return sel.KFold(1, false).split($X, null, null); }, "at least 2",
            "n_splits=1 is refused");
raises_with(def() { return sel.KFold(11, false).split($X, null, null); }, "only 10 rows",
            "more folds than rows is refused, and the message says why");
raises_with(def() { return sel.KFold(2.5, false).split($X, null, null); }, "whole number",
            "a fractional n_splits is refused");

// ── 2. GATE: the same seed gives the same split, twice ──────────────────────
print("");
print("-- GATE: reproducibility comes from nd_seed alone (C4, K21) --");
nd_seed(123);
$a = sel.KFold(4, true).split($X, null, null);
nd_seed(123);
$b = sel.KFold(4, true).split($X, null, null);
$same = true;
$i = 0;
while ($i < len($a)) {
    if (!nd_array_equal($a[$i][1], $b[$i][1])) { $same = false; }
    $i = $i + 1;
}
ok($same, "a shuffled KFold is identical under the same seed");
nd_seed(124);
$c = sel.KFold(4, true).split($X, null, null);
$diff = false;
$i = 0;
while ($i < len($a)) {
    if (!nd_array_equal($a[$i][1], $c[$i][1])) { $diff = true; }
    $i = $i + 1;
}
ok($diff, "and different under a different seed -- so it really is shuffling");
// no splitter carries a random_state: there is nothing to set
ok(!sel.KFold(3, false).shuffles(), "an unshuffled KFold says so");
ok(sel.KFold(3, true).shuffles(), "and a shuffled one says so");

// ── 3. GATE: StratifiedKFold preserves the class ratio to ±1 ────────────────
print("");
print("-- GATE: stratification, and the rotation that keeps FOLDS balanced --");
$sk = sel.StratifiedKFold(2, false).split($X, $y, null);
lst($sk[0][1], [0, 1, 2, 8, 9], "fold 0 takes 3 of class 0 and 2 of class 1");
lst($sk[1][1], [3, 4, 5, 6, 7], "fold 1 takes 2 of class 0 and 3 of class 1");
// Without the rotation both classes would put their extra in fold 0 and the
// folds would be 6 and 4. With it they are 5 and 5.
eqs(nd_size($sk[0][1]), 5, "the folds are the SAME SIZE -- that is the rotation (K22)");
eqs(nd_size($sk[1][1]), 5, "  both of them");

// the general property, over an awkward shape: 23 rows, 3 classes, 4 folds
nd_seed(7);
$n2 = 23;
$X2 = nd_reshape(nd_astype(nd_arange(0, $n2, null), "f64"), [$n2, 1]);
$y2 = nd_astype(nd_random_int([$n2], 0, 3), "f64");
$sk2 = sel.StratifiedKFold(4, true).split($X2, $y2, null);
$worst = 0.0;
$cls = 0;
while ($cls < 3) {
    $total = nd_get(nd_count_nonzero(nd_equal($y2, 1.0 * $cls, null), null, null), 0);
    $f = 0;
    while ($f < len($sk2)) {
        $inFold = nd_get(nd_count_nonzero(
            nd_equal(nd_take($y2, $sk2[$f][1], 0), 1.0 * $cls, null), null, null), 0);
        $expect = $total / 4.0;
        $dev = abs($inFold - $expect);
        if ($dev > $worst) { $worst = $dev; }
        $f = $f + 1;
    }
    $cls = $cls + 1;
}
print("        worst per-fold deviation from the expected class count: " + str($worst));
ok($worst <= 1.0, "every class is within 1 of its expected count in every fold");
raises_with(def() { return sel.StratifiedKFold(4, false).split(
                nd_reshape(nd([1.0, 2.0, 3.0, 4.0], "f64"), [4, 1]),
                nd([0.0, 0.0, 0.0, 1.0], "f64"), null); }, "member(s) but there are 4 folds",
            "a class with fewer members than folds is refused, by name");
raises_with(def() { return sel.StratifiedKFold(2, false).split($X, null, null); }, "needs labels",
            "stratifying without labels says so");
raises_with(def() { return sel.StratifiedKFold(2, false).split(
                $X, nd([0.0, 1.0], "f64"), null); }, "2 labels for 10 rows",
            "a label count that does not match the rows is refused");

// ── 4. GATE: GroupKFold never splits a group ────────────────────────────────
print("");
print("-- GATE: no group appears in two folds --");
$gk = sel.GroupKFold(2).split($X, null, $g);
$leak = 0;
$gi = 0;
$uniq = nd_unique($g);
while ($gi < nd_size($uniq)) {
    $gv = nd_get($uniq, $gi);
    $foldsSeen = 0;
    $f = 0;
    while ($f < len($gk)) {
        $hits = nd_get(nd_count_nonzero(
            nd_equal(nd_take($g, $gk[$f][1], 0), $gv, null), null, null), 0);
        if ($hits > 0) { $foldsSeen = $foldsSeen + 1; }
        $f = $f + 1;
    }
    if ($foldsSeen != 1) { $leak = $leak + 1; }
    $gi = $gi + 1;
}
eqs($leak, 0, "each of the 4 groups is tested in exactly one fold");
// largest-first into the lightest fold: group 1 (3) and group 3 (3) are the
// big ones and land in different folds
lst($gk[0][1], [0, 1, 2, 3, 4], "the assignment is largest-first into the lightest fold");
lst($gk[1][1], [5, 6, 7, 8, 9], "  which balances the two folds at 5 rows each");
raises_with(def() { return sel.GroupKFold(2).split($X, null, null); }, "needs `groups`",
            "GroupKFold without groups points at KFold instead");
raises_with(def() { return sel.GroupKFold(5).split(
                $X, null, nd([0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0], "f64")); },
            "cannot fill", "fewer groups than folds is refused");
// a 50-row set with 7 groups, checked for leakage the same way
nd_seed(3);
$n3 = 50;
$X3 = nd_reshape(nd_astype(nd_arange(0, $n3, null), "f64"), [$n3, 1]);
$g3 = nd_astype(nd_random_int([$n3], 0, 7), "f64");
$gk3 = sel.GroupKFold(3).split($X3, null, $g3);
$leak3 = 0;
$gi = 0;
$u3 = nd_unique($g3);
while ($gi < nd_size($u3)) {
    $seen = 0;
    $f = 0;
    while ($f < len($gk3)) {
        if (nd_get(nd_count_nonzero(nd_equal(
                nd_take($g3, $gk3[$f][1], 0), nd_get($u3, $gi), null), null, null), 0) > 0) {
            $seen = $seen + 1;
        }
        $f = $f + 1;
    }
    if ($seen != 1) { $leak3 = $leak3 + 1; }
    $gi = $gi + 1;
}
eqs($leak3, 0, "50 rows in 7 random groups, 3 folds: still no group split");

// ── 5. GATE: TimeSeriesSplit never trains on the future ─────────────────────
print("");
print("-- GATE: every training index precedes every test index --");
$ts = sel.TimeSeriesSplit(3).split($X, null, null);
lst($ts[0][0], [0, 1], "fold 0 trains on the first t rows");
lst($ts[0][1], [2, 3], "  and tests the next t");
lst($ts[2][0], [0, 1, 2, 3, 4, 5], "the training window grows");
lst($ts[2][1], [6, 7, 8, 9], "  and the last fold takes the tail, so no row is wasted");
$bad = 0;
$f = 0;
while ($f < len($ts)) {
    $maxTrain = nd_get(nd_max($ts[$f][0], null, null), 0);
    $minTest  = nd_get(nd_min($ts[$f][1], null, null), 0);
    if ($maxTrain >= $minTest) { $bad = $bad + 1; }
    $f = $f + 1;
}
eqs($bad, 0, "in every fold, max(train) < min(test) -- no leakage from the future");
ok(!sel.TimeSeriesSplit(3).partitions(),
   "TimeSeriesSplit reports partitions()=false: the first rows are never tested");
raises_with(def() { return sel.TimeSeriesSplit(20).split($X, null, null); }, "at least 21",
            "too many splits for the row count is refused, with the number needed");

// ── 6. tier 3: the splitter contract, and a check that can FAIL ─────────────
print("");
print("-- the splitter contract over every splitter --");
$sps = [["KFold",              sel.KFold(5, false)],
        ["KFold(shuffle)",     sel.KFold(5, true)],
        ["KFold(3)",           sel.KFold(3, false)],
        ["StratifiedKFold",    sel.StratifiedKFold(2, false)],
        ["StratifiedKFold(sh)", sel.StratifiedKFold(2, true)],
        ["GroupKFold",         sel.GroupKFold(2)],
        ["TimeSeriesSplit",    sel.TimeSeriesSplit(3)],
        ["TimeSeriesSplit(1)", sel.TimeSeriesSplit(1)]];
$i = 0;
while ($i < len($sps)) {
    $bad = sel.splitter_check($sps[$i][1], $X, $y, $g);
    if (len($bad) == 0) { ok(true, $sps[$i][0] + " satisfies the splitter contract"); }
    else { ok(false, $sps[$i][0] + " violates it: " + str($bad)); }
    $i = $i + 1;
}

// A splitter whose folds OVERLAP. This is the failure the check exists for: the
// model is graded on rows it trained on, and nothing about the numbers looks
// wrong afterwards.
class _LeakySplitter {
    def init() { this.name_ = "LeakySplitter"; }
    def name() { return this.name_; }
    def get_n_splits() { return 2; }
    def partitions() { return true; }
    def shuffles() { return false; }
    def split($X, $y, $groups) {
        return [[nd([0, 1, 2, 3, 4, 5], "i64"), nd([5, 6, 7], "i64")],
                [nd([5, 6, 7, 8, 9], "i64"), nd([0, 1, 2, 3, 4], "i64")]];
    }
}
def LeakySplitter() { return new _LeakySplitter(); }
$leaky = sel.splitter_check(LeakySplitter(), $X, $y, $g);
ok(len($leaky) > 0, "the check catches a splitter whose train and test overlap");
ok(indexOf(str($leaky), "overlap") >= 0, "and says that is what went wrong");

// one that miscounts its own folds
class _MiscountSplitter {
    def init() { this.name_ = "MiscountSplitter"; }
    def name() { return this.name_; }
    def get_n_splits() { return 5; }
    def partitions() { return false; }
    def shuffles() { return false; }
    def split($X, $y, $groups) {
        return [[nd([0, 1], "i64"), nd([2, 3], "i64")]];
    }
}
def MiscountSplitter() { return new _MiscountSplitter(); }
$mis = sel.splitter_check(MiscountSplitter(), $X, $y, $g);
ok(len($mis) > 0, "and one whose get_n_splits() disagrees with split()");
$nullsp = sel.splitter_check(null, $X, $y, $g);
eqs(len($nullsp), 1, "and it survives being handed null");

// ── 7. train_test_split ─────────────────────────────────────────────────────
print("");
print("-- train_test_split --");
nd_seed(9);
$s = sel.train_test_split($X, $yr, 0.3, true, null);
eqs(str(nd_shape($s.X_train)), str([7, 2]), "a 0.3 test fraction of 10 rows leaves 7 training rows");
eqs(str(nd_shape($s.X_test)), str([3, 2]), "  and 3 test rows: ceil(10 * 0.3)");
eqs(nd_size($s.y_train), 7, "y is split the same way");
eqs(nd_size($s.y_test), 3, "  on both sides");
// disjoint and complete
$un = nd_sort(nd_concat([$s.train_index, $s.test_index], 0), null);
lst($un, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "the two index sets are disjoint and cover every row");
// the rows really are the rows those indices name
ok(nd_array_equal($s.X_test, nd_take($X, $s.test_index, 0)),
   "X_test is exactly the rows test_index names");
ok(nd_array_equal($s.y_test, nd_take($yr, $s.test_index, 0)),
   "and y_test is aligned with it -- not shuffled separately");
// a whole count instead of a fraction
$s2 = sel.train_test_split($X, $yr, 4, true, null);
eqs(nd_size($s2.test_index), 4, "an integer test_size is a row count");
// y may be omitted
$s3 = sel.train_test_split($X, null, 0.5, false, null);
ok($s3.y_train == null, "y_train is null when no y was given");
lst($s3.test_index, [0, 1, 2, 3, 4], "and an unshuffled split takes the first rows for test");
// reproducible
nd_seed(55);  $r1 = sel.train_test_split_indices($X, 0.3, true, null);
nd_seed(55);  $r2 = sel.train_test_split_indices($X, 0.3, true, null);
ok(nd_array_equal($r1[1], $r2[1]), "the same seed gives the same split");
// shuffle=true must actually PERMUTE. Everything asserted above -- the sizes,
// disjointness, full coverage, reproducibility -- holds just as well for a
// split that never shuffles at all, and a mutation that removed the shuffle
// passed all of it.
$unshuffled = nd_to_list(sel.train_test_split_indices($X, 0.3, false, null)[1]);
eqs(str($unshuffled), str([0, 1, 2]), "an unshuffled split takes rows 0, 1, 2");
$differs = 0;
$distinct = [];
$t = 0;
while ($t < 5) {
    nd_seed(2000 + $t);
    $got = str(nd_to_list(nd_sort(sel.train_test_split_indices($X, 0.3, true, null)[1], null)));
    if ($got != str([0, 1, 2])) { $differs = $differs + 1; }
    if (!contains($distinct, $got)) { append($distinct, $got); }
    $t = $t + 1;
}
// Not "all five": a shuffled split of ten rows coincides with the ordered one
// about 1 time in 120, and one of these five seeds does exactly that. Pinning
// it to 5 made the assertion depend on luck. The real proof of shuffling is
// the number of DISTINCT test sets, which no unshuffled implementation can
// produce at all.
ok($differs >= 4, "and a shuffled one differs from it on at least four of five seeds");
ok(len($distinct) >= 3, "  giving at least three different test sets across those seeds");

raises_with(def() { return sel.train_test_split($X, $yr, 10, true, null); }, "no training rows",
            "a test_size equal to the row count is refused");
raises_with(def() { return sel.train_test_split($X, $yr, 1.5, true, null); }, "fraction",
            "a test_size of 1.5 is neither a fraction nor a count, and is refused");

print("");
print("-- stratified train_test_split --");
nd_seed(21);
// 20 rows, 10% of one class: an unstratified split would often put none of it
// in the test half at all
$y4 = nd([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0], "f64");
$X4 = nd_reshape(nd_astype(nd_arange(0, 20, null), "f64"), [20, 1]);
$s4 = sel.train_test_split($X4, $y4, 0.5, true, $y4);
$nTestRare = nd_get(nd_count_nonzero(nd_equal($s4.y_test, 1.0, null), null, null), 0);
$nTrainRare = nd_get(nd_count_nonzero(nd_equal($s4.y_train, 1.0, null), null, null), 0);
eqs($nTestRare, 1, "a 2-member class puts one member in test");
eqs($nTrainRare, 1, "  and one in train -- it is never wholly on one side");
$un4 = nd_sort(nd_concat([$s4.train_index, $s4.test_index], 0), null);
eqs(nd_size($un4), 20, "the stratified split still covers every row exactly once");
ok(nd_get(nd_min(nd_diff($un4, null), null, null), 0) == 1.0,
   "  with no index repeated");
raises_with(def() { return sel.train_test_split($X4, $y4, 0.5, false, $y4); }, "shuffle=true",
            "an unshuffled stratified split is refused rather than silently ordered");

// THE FLOOR AND THE CAP, exercised at the fractions where they actually fire.
// The 0.5 case above does not reach either: for a 2-member class,
// round(2 * 0.5) is already 1. A mutation that removed the floor passed it.
//
// At test_size = 0.2 the rare class rounds to round(2 * 0.2) = 0, so without
// the floor it would be absent from the test half entirely -- which is the one
// thing stratifying is for.
nd_seed(31);
$sLow = sel.train_test_split($X4, $y4, 0.2, true, $y4);
eqs(nd_get(nd_count_nonzero(nd_equal($sLow.y_test, 1.0, null), null, null), 0), 1,
    "at a 0.2 fraction the 2-member class still reaches the test half (the floor)");
// and at 0.9 it rounds to 2, which would leave the class out of TRAINING
nd_seed(32);
$sHigh = sel.train_test_split($X4, $y4, 0.9, true, $y4);
eqs(nd_get(nd_count_nonzero(nd_equal($sHigh.y_train, 1.0, null), null, null), 0), 1,
    "and at 0.9 it still reaches the training half (the cap)");

// ── 8. GATE: cross-validation clones before every fit ───────────────────────
print("");
print("-- GATE: cross_val_score does not touch the estimator it was given --");
$src = ShiftRegressor(0.0);
$scores = sel.cross_val_score($src, $X, $yr, sel.KFold(5, false));
eqs(nd_size($scores), 5, "one score per fold");
ok(!$src.fitted(), "the SOURCE estimator is still unfitted afterwards -- it was cloned (K20)");
eqs($src.rows_seen(), 0, "and it never saw a row");
// the same numbers as a hand-run loop
$hand = [];
$folds = sel.KFold(5, false).split($X, null, null);
$f = 0;
while ($f < len($folds)) {
    $m = ShiftRegressor(0.0);
    $m.fit(nd_take($X, $folds[$f][0], 0), nd_take($yr, $folds[$f][0], 0));
    append($hand, $m.score(nd_take($X, $folds[$f][1], 0), nd_take($yr, $folds[$f][1], 0)));
    $f = $f + 1;
}
ok(nd_array_equal($scores, nd($hand, "f64")),
   "and they equal a hand-run fold loop BIT FOR BIT");
// a plain number means KFold(n, shuffle=false)
ok(nd_array_equal(sel.cross_val_score($src, $X, $yr, 5), $scores),
   "cv=5 means KFold(5) without shuffling");
$cvr = sel.cross_validate($src, $X, $yr, 5);
eqs($cvr.n_splits, 5, "cross_validate reports the split count");
eqs(len($cvr.estimators), 5, "  and hands back the per-fold models");
ok(!nd_array_equal($cvr.estimators[0].predict($X), $cvr.estimators[1].predict($X)),
   "  which are genuinely different models, one per fold");
eqs(nd_size($cvr.fit_time_ms), 5, "  and a timing for each");

// ── 9. GATE: the search's reported best equals a hand-run refit ─────────────
print("");
print("-- GATE: GridSearchCV's best score is one it actually computed --");
$grid = {"shift": [0.0 - 4.0, 0.0 - 2.0, 0.0, 2.0, 4.0]};
$gs = sel.GridSearchCV(ShiftRegressor(0.0), $grid, sel.KFold(5, false));
$gs.fit($X, $yr);
eqs(len($gs.cv_results()), 5, "one result per candidate");

// The whole grid, recomputed by hand over the SAME folds, and the argmax taken
// here rather than assumed. Asserting "the best shift is 0" would have been
// asserting my own guess: on ten rows the optimum is wherever the per-fold
// train means happen to sit relative to the test means, and with a finer grid
// it is NOT 0. What the search owes us is that its winner is the argmax of the
// scores it computed, and that the score it reports is the one it computed.
def cvMean($shift, $folds, $yv) {
    $tot = 0.0;
    $f = 0;
    while ($f < len($folds)) {
        $m = ShiftRegressor($shift);
        $m.fit(nd_take($X, $folds[$f][0], 0), nd_take($yv, $folds[$f][0], 0));
        $tot = $tot + $m.score(nd_take($X, $folds[$f][1], 0),
                               nd_take($yv, $folds[$f][1], 0));
        $f = $f + 1;
    }
    return $tot / (1.0 * len($folds));
}
$shifts = $grid.shift;
$bestHand = 0;
$i = 1;
while ($i < len($shifts)) {
    if (cvMean($shifts[$i], $folds, $yr) > cvMean($shifts[$bestHand], $folds, $yr)) { $bestHand = $i; }
    $i = $i + 1;
}
close($gs.best_params().shift, $shifts[$bestHand], 1.0e-15,
      "best_params() is the argmax of the hand-recomputed grid");
close($gs.best_score(), cvMean($shifts[$bestHand], $folds, $yr), 1.0e-15,
      "and best_score() equals that hand-run mean to 1e-15");
// every reported per-candidate mean, not only the winner's
$allMatch = true;
$i = 0;
while ($i < len($shifts)) {
    if (abs($gs.cv_results()[$i].mean_test_score - cvMean($shifts[$i], $folds, $yr)) > 1.0e-15) {
        $allMatch = false;
    }
    $i = $i + 1;
}
ok($allMatch, "and so does EVERY candidate's reported mean, not just the winner's");
// the refit model saw ALL the rows, not k-1/k of them
eqs($gs.best_estimator().rows_seen(), 10,
    "best_estimator() was refitted on every row, not on a fold");
ok($gs.fitted(), "the search itself reports fitted");
eqs(nd_size($gs.predict($X)), 10, "and predict() delegates to the refitted winner");
// monotone ordering: a bigger |shift| must score worse, which proves the
// search is really evaluating the parameter and not ignoring it
$r0 = $gs.cv_results();
ok($r0[2].mean_test_score > $r0[1].mean_test_score,
   "shift=0 scores better than shift=-2");
ok($r0[1].mean_test_score > $r0[0].mean_test_score,
   "and shift=-2 better than shift=-4 -- the parameter is being applied");

print("");
print("-- GATE: the splits are taken ONCE, not per candidate (K21) --");
// With an UNSHUFFLED splitter this is unfalsifiable: two calls to split()
// return the same folds, so re-splitting per candidate would change nothing.
// A mutation that re-split per candidate passed the test above for exactly
// that reason. With a shuffled splitter the folds differ every call, so the
// reported per-candidate means can only match a single fixed split if the
// implementation took one.
//
// The folds are reproduced by re-seeding: fit() consumes one split's worth of
// randomness from nd_seed's stream, so seeding the same value and calling
// split() directly reproduces the folds it used.
// $yr has repeated values, and a shuffled 2-row fold can pair two equal ones:
// the target is then constant, SST is zero, and base.b's score() rightly
// refuses to return an R2 of 0/0. So this block uses a target whose values are
// all distinct -- a property of the test data, not a workaround.
$yd = nd([10.0, 21.0, 32.0, 43.0, 54.0, 65.0, 76.0, 87.0, 98.0, 109.0], "f64");
nd_seed(777);
$gsh = sel.GridSearchCV(ShiftRegressor(0.0), $grid, sel.KFold(5, true));
$gsh.fit($X, $yd);
nd_seed(777);
$shFolds = sel.KFold(5, true).split($X, null, null);
$oneSplit = true;
$i = 0;
while ($i < len($shifts)) {
    if (abs($gsh.cv_results()[$i].mean_test_score - cvMean($shifts[$i], $shFolds, $yd)) > 1.0e-12) {
        $oneSplit = false;
    }
    $i = $i + 1;
}
ok($oneSplit,
   "every candidate was scored on the SAME shuffled folds -- re-splitting per candidate would not match");
// and those folds really are shuffled, or the check above proves nothing
nd_seed(777);
$shA = sel.KFold(5, true).split($X, null, null);
nd_seed(778);
$shB = sel.KFold(5, true).split($X, null, null);
ok(!nd_array_equal($shA[0][1], $shB[0][1]),
   "  and a shuffled KFold really does give different folds on a different seed");

print("");
print("-- GATE: a mistyped grid key raises BEFORE the first fit (K24) --");
raises_with(def() { return sel.GridSearchCV(ShiftRegressor(0.0),
                        {"shfit": [1.0, 2.0]}, 5).fit($X, $yr); }, "does not accept",
            "a typo in a grid key is an error, not a silently ignored parameter");
raises_with(def() { return sel.GridSearchCV(ShiftRegressor(0.0),
                        {"shift": 1.0}, 5).fit($X, $yr); }, "LIST",
            "a grid value that is not a list is refused");
raises_with(def() { return sel.GridSearchCV(ShiftRegressor(0.0),
                        {"shift": []}, 5).fit($X, $yr); }, "empty",
            "an empty value list is refused");
// and the underlying refusal, direct
raises_with(def() { return ShiftRegressor(0.0).set_params({"nope": 1}); }, "no parameter",
            "set_params itself refuses an unknown key, naming the ones it takes");

// ── 10. the candidate order is determined ───────────────────────────────────
print("");
print("-- param_grid --");
$c1 = sel.param_grid({"b": [1, 2], "a": [10, 20, 30]});
$c2 = sel.param_grid({"a": [10, 20, 30], "b": [1, 2]});
eqs(len($c1), 6, "the product of a 2-list and a 3-list is 6 candidates");
eqs(str($c1), str($c2),
    "and the order does not depend on how the dict literal was written -- keys are sorted");
eqs(str($c1[0]), str({"a": 10, "b": 1}), "the first candidate takes the first of each");
eqs(str($c1[1]), str({"a": 10, "b": 2}), "and the LAST key varies fastest");
eqs(len(sel.param_grid({})), 1, "an empty grid is one candidate: the defaults");
eqs(len(sel.param_grid(null)), 1, "and so is a null grid");

// ── 11. RandomizedSearchCV ──────────────────────────────────────────────────
print("");
print("-- RandomizedSearchCV --");
nd_seed(42);
$rs = sel.RandomizedSearchCV(ShiftRegressor(0.0), $grid, 5, sel.KFold(5, false));
$rs.fit($X, $yr);
eqs(len($rs.cv_results()), 5, "n_iter equal to the grid size evaluates all of it");
close($rs.best_score(), $gs.best_score(), 1.0e-15,
      "  so it finds exactly the same best score as the exhaustive search");
nd_seed(42);
$rs2 = sel.RandomizedSearchCV(ShiftRegressor(0.0), $grid, 3, sel.KFold(5, false));
$rs2.fit($X, $yr);
eqs(len($rs2.cv_results()), 3, "n_iter below the grid size evaluates that many");
// Distinct candidates: sampling WITH replacement would spend the budget on
// repeats and report a best over fewer configurations than asked for.
//
// This is checked over TEN SEEDS, not one. A single seed is luck: drawing 4 of
// 5 with replacement gives all-distinct about 19% of the time, so one seed
// would let a with-replacement implementation through four times in five. Ten
// consecutive all-distinct draws under replacement would be a 6-in-100-million
// coincidence.
$dup = 0;
$trial = 0;
while ($trial < 10) {
    nd_seed(1000 + $trial);
    $rsT = sel.RandomizedSearchCV(ShiftRegressor(0.0), $grid, 4, sel.KFold(5, false));
    $rsT.fit($X, $yr);
    $seen = [];
    $i = 0;
    while ($i < len($rsT.cv_results())) {
        $sv = $rsT.cv_results()[$i].params.shift;
        if (contains($seen, $sv)) { $dup = $dup + 1; }
        append($seen, $sv);
        $i = $i + 1;
    }
    $trial = $trial + 1;
}
eqs($dup, 0, "  4-of-5 sampling is DISTINCT on all ten seeds -- no replacement");
nd_seed(42);
$rs3 = sel.RandomizedSearchCV(ShiftRegressor(0.0), $grid, 3, sel.KFold(5, false));
$rs3.fit($X, $yr);
eqs(str($rs3.best_params()), str($rs2.best_params()),
    "the same seed samples the same candidates");
raises_with(def() { return sel.RandomizedSearchCV(ShiftRegressor(0.0), $grid, 0, 5); },
            "at least 1", "n_iter=0 is refused");

// ── 12. a search over a Pipeline, through step__param ───────────────────────
print("");
print("-- a search over a Pipeline --");
$pipe = base.Pipeline([["sc", pre.StandardScaler(null, null)],
                       ["reg", ShiftRegressor(0.0)]]);
$gp = sel.GridSearchCV($pipe, {"reg__shift": [0.0 - 1.0, 0.0, 1.0]}, sel.KFold(5, false));
$gp.fit($X, $yr);
eqs(len($gp.cv_results()), 3, "the grid addresses a step's parameter");
// The scaler cannot change this regressor's prediction -- ShiftRegressor
// ignores X except for its row count -- so the pipeline's per-candidate means
// must equal the bare estimator's over the same folds. That is a real check on
// the plumbing: if step__param were not reaching the step, every candidate
// would score identically.
$pipeShifts = [0.0 - 1.0, 0.0, 1.0];
$pipeMatch = true;
$i = 0;
while ($i < len($pipeShifts)) {
    if (abs($gp.cv_results()[$i].mean_test_score - cvMean($pipeShifts[$i], $folds, $yr)) > 1.0e-12) {
        $pipeMatch = false;
    }
    $i = $i + 1;
}
ok($pipeMatch, "and each candidate scores exactly as the bare estimator does");
$bh = 0;
$i = 1;
while ($i < len($pipeShifts)) {
    if (cvMean($pipeShifts[$i], $folds, $yr) > cvMean($pipeShifts[$bh], $folds, $yr)) { $bh = $i; }
    $i = $i + 1;
}
close($gp.best_params().reg__shift, $pipeShifts[$bh], 1.0e-15,
      "the winner is the argmax of those means");
close($gp.best_estimator().get_step("reg").get_params().shift, $pipeShifts[$bh], 1.0e-15,
    "and the refitted pipeline carries that value in its step");
// the candidates are NOT all equal, which is what proves the parameter arrived
ok(abs($gp.cv_results()[0].mean_test_score - $gp.cv_results()[2].mean_test_score) > 1.0e-9,
   "the three candidates score differently, so step__param really reached the step");
raises_with(def() { return sel.GridSearchCV($pipe,
                {"nosuch__shift": [1.0]}, 5).fit($X, $yr); }, "does not accept",
            "a grid naming a step that does not exist is refused before fitting");

// ── 13. tier 4: stress ──────────────────────────────────────────────────────
print("");
print("-- stress --");
nd_seed(2024);
$nBig = 50000;
$XB = nd_random_uniform([$nBig, 4], 0.0 - 1.0, 1.0);
$t0 = clock();
$bigFolds = sel.KFold(5, true).split($XB, null, null);
print("        50000-row shuffled 5-fold split: " + str(clock() - $t0) + " ms");
$cover = nd_zeros([$nBig], "f64");
$f = 0;
$ovl = 0;
while ($f < 5) {
    nd_put($cover, $bigFolds[$f][1],
           nd_add(nd_take($cover, $bigFolds[$f][1], 0), 1.0, null));
    $mask = nd_zeros([$nBig], "f64");
    nd_put($mask, $bigFolds[$f][0], 1.0);
    $ovl = $ovl + nd_get(nd_sum(nd_take($mask, $bigFolds[$f][1], 0), null, null), 0);
    $f = $f + 1;
}
eqs($ovl, 0.0, "at 50000 rows, no fold's train and test overlap");
ok(nd_get(nd_all(nd_equal($cover, 1.0, null), null, null), 0),
   "and every one of the 50000 rows is tested exactly once");
$t1 = clock();
$gsBig = sel.GridSearchCV(ShiftRegressor(0.0),
            {"shift": [0.0 - 2.0, 0.0 - 1.0, 0.0, 1.0, 2.0]}, sel.KFold(4, false));
$gsBig.fit(nd_reshape(nd_astype(nd_arange(0, 400, null), "f64"), [400, 1]),
           nd_random_uniform([400], 0.0, 1.0));
print("        5 candidates x 4 folds on 400 rows: " + str(clock() - $t1) + " ms");
ok(len($gsBig.cv_results()) == 5, "a 20-fit search completes");
// A LEAK IS GROWTH WITH THE ITERATION COUNT, not a non-zero residue. The first
// version of this assertion read 400 bytes after 200 splits and called it a
// leak; 400 bytes is exactly one 5-fold split of ten rows (5 x 10 i64 indices),
// still referenced by $tmp. So the test runs the loop twice at different
// lengths and asserts the residue is the SAME, then drops the reference and
// asserts it goes to zero -- both stronger statements than "did not grow".
$before = nd_live_bytes();
$j = 0;
while ($j < 200) { $tmp = sel.KFold(5, false).split($X, null, null); $j = $j + 1; }
$after200 = nd_live_bytes();
$j = 0;
while ($j < 600) { $tmp = sel.KFold(5, false).split($X, null, null); $j = $j + 1; }
$after800 = nd_live_bytes();
print("        live bytes " + str($before) + " -> " + str($after200)
      + " (200 splits) -> " + str($after800) + " (800)");
eqs($after800, $after200, "800 splits hold no more than 200 did: nothing accumulates");
$tmp = null;
gc_collect();
eqs(nd_live_bytes(), $before, "and dropping the last reference returns to the starting figure");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
