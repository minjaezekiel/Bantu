// ════════════════════════════════════════════════════════════════════════════
//  selection.b — splitting, cross-validation and parameter search.
//
//      include "bkit/base.b"      as base;
//      include "bkit/selection.b" as sel;
//
//      nd_seed(0);                              // reproducibility starts here
//      $s = sel.train_test_split($X, $y, 0.25, true, null);
//      $scores = sel.cross_val_score($est, $s.X_train, $s.y_train, 5);
//
//  Design: docs/bkit-architecture.md §4. Decisions K18-K24.
//
//  THE ONE THING THIS FILE EXISTS TO PREVENT is a score that is better than
//  the model. Every design choice below serves that:
//
//    * SPLITTERS RETURN INDICES, never data (K19). A 5-fold split of a
//      1M x 100 matrix copies nothing, `cross_val_score` takes rows once per
//      fold -- and the invariants become checkable, which is the part that
//      matters. "No group appears in two folds" is a statement about indices;
//      given only X_train matrices you would have to search for rows.
//
//    * CROSS-VALIDATION CLONES BEFORE EVERY FIT (K20, and [K2] before it).
//      Reusing a fitted estimator across folds leaks the test fold into
//      training and still produces a plausible, merely inflated score.
//
//    * THE SPLITS ARE MATERIALISED ONCE and reused for every candidate (K21).
//      Re-splitting per candidate with a shuffled splitter would compare
//      candidates on DIFFERENT folds, so the winner would be partly a draw of
//      the RNG. With the folds fixed, a difference in score is a difference in
//      the parameters.
//
//    * A MISTYPED GRID KEY IS AN ERROR (K24). It used to add a parameter
//      nobody reads, so a grid search over `with_men` ran every candidate with
//      identical settings and reported a best configuration it never tried.
//
//  NO SPLITTER TAKES A `random_state`. Reproducibility comes from seeding once,
//  with `nd_seed`, which is the ecosystem's single RNG authority ([C4]). A
//  per-splitter seed would have to call `nd_seed` to take effect, and that
//  clobbers the stream for everything downstream -- a model drawing its weights
//  after a grid search would get a stream chosen by the splitter.
// ════════════════════════════════════════════════════════════════════════════

include "./base.b" as base;

// ── shared helpers ──────────────────────────────────────────────────────────

// Rows of X, for anything that only needs the count. Accepts a list of rows,
// a 1-D array (n rows of one column) or a 2-D array.
def _nrows($X, $who) {
    if ($X == null) { throw $who + ": X is null"; }
    $A = $X;
    if (type($A) == "list") { $A = nd($A, "f64"); }
    if (type($A) != "ndarray") {
        throw $who + ": expected an array or a list of rows, got " + type($X);
    }
    if (nd_ndim($A) == 0) { throw $who + ": X is 0-dimensional"; }
    return nd_shape($A)[0];
}

// An i64 array of 0..n-1.
def _arange_idx($n) { return nd_astype(nd_arange(0, $n, null), "i64"); }

// A uniformly random permutation of 0..n-1, in TWO kernel calls (K18).
// Fisher-Yates in Bantu would be n interpreter iterations and n separate RNG
// draws; the C-style `for` also stops silently at 100,000 iterations.
//
// argsort of DISTINCT keys is a uniform permutation. Two equal f64 keys are
// broken by index, which is a negligible bias -- at n = 1,000,000 the chance of
// any collision at all is about 1 in 10,000, and a collision perturbs one
// adjacent pair. The draw comes from nd_seed's stream, so the same seed gives
// the same permutation and nothing here carries RNG state ([C4]).
def _permutation($n) {
    if ($n <= 1) { return _arange_idx($n); }
    return nd_argsort(nd_random_uniform([$n], 0.0, 1.0), null);
}

// 0..n-1 minus `test`, in ascending order. Built with a mask rather than a
// search, so it is O(n) in kernels and does not care whether `test` is
// contiguous -- which StratifiedKFold's and GroupKFold's test folds are not.
def _complement($n, $test) {
    $mask = nd_zeros([$n], "f64");
    // nd_put takes ARRAYS and a scalar value directly. Going through
    // nd_to_list first -- which the first version of this did -- builds two
    // Bantu lists per fold and costs 29x as much: 29 ms against 1 ms for
    // 100,000 indices, best of five on an idle machine. The lists were pure
    // overhead on a path that runs once per fold per candidate of a grid
    // search.
    if (nd_size($test) > 0) { nd_put($mask, $test, 1.0); }
    return nd_astype(nd_nonzero(nd_equal($mask, 0.0, null)), "i64");
}

// The fold-size rule, in one place because it is a decision and not a detail
// (K22): the first `n mod k` folds get one extra row. Returns the k+1 offsets,
// so fold f is [out[f], out[f+1]).
def _fold_bounds($n, $k) {
    $base = floor($n / $k);
    $rem  = $n - $base * $k;
    $out  = [0];
    $acc  = 0;
    $f    = 0;
    while ($f < $k) {
        $sz = $base;
        if ($f < $rem) { $sz = $base + 1; }
        $acc = $acc + $sz;
        append($out, $acc);
        $f = $f + 1;
    }
    return $out;
}

def _check_k($k, $n, $who) {
    if ($k == null || $k != floor($k) || $k < 2) {
        throw $who + ": n_splits must be a whole number of at least 2, got " + str($k);
    }
    if ($k > $n) {
        throw $who + ": n_splits is " + str($k) + " but there are only "
              + str($n) + " rows -- a fold would be empty";
    }
    return null;
}

// A list of indices -> an i64 array. An empty list has no dtype to infer, so
// it gets one explicitly rather than becoming an f64 array of nothing.
def _idx($l) {
    if (len($l) == 0) { return nd_astype(nd_zeros([0], "f64"), "i64"); }
    return nd_astype(nd($l, "f64"), "i64");
}

// Labels as a 1-D f64 array, for stratification and grouping.
def _labels($y, $who) {
    if ($y == null) { throw $who + ": this splitter needs labels, and y is null"; }
    $A = $y;
    if (type($A) == "list") { $A = nd($A, "f64"); }
    if (type($A) != "ndarray") {
        throw $who + ": expected an array or a list of labels, got " + type($y);
    }
    if (nd_dtype($A) != "f64") { $A = nd_astype($A, "f64"); }
    if (nd_ndim($A) == 2) {
        if (nd_shape($A)[1] != 1) {
            throw $who + ": labels must be one column, got " + str(nd_shape($A)[1]);
        }
        $A = nd_ravel($A);
    }
    if (nd_get(nd_any(nd_isnan($A, null), null, null), 0)) {
        throw $who + ": the labels contain NaN, which is not a class";
    }
    return $A;
}

// Rows of X by index, for callers that want the data and not the indices.
def take_rows($X, $idx) {
    $A = $X;
    if (type($A) == "list") { $A = nd($A, "f64"); }
    return nd_take($A, $idx, 0);
}

// ── train_test_split ────────────────────────────────────────────────────────
// Returns a NAMED dict, not four positionals (§4.6). scikit-learn's
// `X_train, X_test, y_train, y_test = ...` interleaves the two variables rather
// than grouping them, and swapping two of them gives a working program that
// trains on its test set. In Bantu it would have to be read as $r[2] and $r[3],
// which is worse. Named fields cannot be transposed.
//
// test_size: a fraction in (0, 1) -- ceil(n * frac), matching scikit-learn --
// or a whole count of at least 1. Either way 1 <= n_test <= n-1, so neither
// side is ever empty.
def train_test_split_indices($X, $test_size, $shuffle, $stratify) {
    $who = "train_test_split";
    $n = _nrows($X, $who);
    if ($n < 2) { throw $who + ": need at least 2 rows to split, got " + str($n); }

    $ts = $test_size;
    if ($ts == null) { $ts = 0.25; }
    $nTest = 0;
    if ($ts > 0.0 && $ts < 1.0) { $nTest = ceil($n * $ts); }
    else {
        if ($ts != floor($ts) || $ts < 1.0) {
            throw $who + ": test_size must be a fraction in (0, 1) or a whole "
                  + "count of at least 1, got " + str($ts);
        }
        $nTest = $ts;
    }
    if ($nTest >= $n) {
        throw $who + ": test_size " + str($ts) + " leaves no training rows "
              + "(" + str($nTest) + " of " + str($n) + ")";
    }

    $sh = $shuffle;
    if ($sh == null) { $sh = true; }

    if ($stratify == null) {
        $idx = _arange_idx($n);
        if ($sh) { $idx = _permutation($n); }
        $test  = nd_slice($idx, [[0, $nTest, 1]]);
        $train = nd_slice($idx, [[$nTest, $n, 1]]);
        return [nd_copy($train), nd_copy($test)];
    }

    // Stratified: each class contributes the same FRACTION to the test side,
    // so a 1%-prevalence class is represented in both halves instead of
    // landing wholly in one of them by chance.
    $lab = _labels($stratify, $who);
    if (nd_shape($lab)[0] != $n) {
        throw $who + ": stratify has " + str(nd_shape($lab)[0])
              + " labels for " + str($n) + " rows";
    }
    if (!$sh) {
        throw $who + ": stratify=true needs shuffle=true -- an unshuffled "
              + "stratified split would take each class's first rows, which is "
              + "whatever order the data arrived in";
    }
    $classes = nd_unique($lab);
    $nc = nd_shape($classes)[0];
    $frac = (1.0 * $nTest) / (1.0 * $n);
    $testL  = [];
    $trainL = [];
    $c = 0;
    while ($c < $nc) {
        $v    = nd_get($classes, $c);
        $mine = nd_astype(nd_nonzero(nd_equal($lab, $v, null)), "i64");
        $m    = nd_shape($mine)[0];
        // shuffle WITHIN the class, so which members go to test is random
        $mine = nd_take($mine, _permutation($m), 0);
        $k    = floor($m * $frac + 0.5);
        if ($k < 1 && $m >= 2) { $k = 1; }      // never drop a class from test
        if ($k > $m - 1 && $m >= 2) { $k = $m - 1; }
        $i = 0;
        while ($i < $m) {
            if ($i < $k) { append($testL, nd_get($mine, $i)); }
            else         { append($trainL, nd_get($mine, $i)); }
            $i = $i + 1;
        }
        $c = $c + 1;
    }
    if (len($testL) == 0 || len($trainL) == 0) {
        throw $who + ": a stratified split of these labels leaves one side "
              + "empty -- too few rows per class";
    }
    return [nd_sort(_idx($trainL), null), nd_sort(_idx($testL), null)];
}

def train_test_split($X, $y, $test_size, $shuffle, $stratify) {
    $parts = train_test_split_indices($X, $test_size, $shuffle, $stratify);
    $tr = $parts[0];
    $te = $parts[1];
    $out = {"train_index": $tr, "test_index": $te,
            "X_train": take_rows($X, $tr), "X_test": take_rows($X, $te),
            "y_train": null, "y_test": null};
    if ($y != null) {
        $ya = $y;
        if (type($ya) == "list") { $ya = nd($ya, "f64"); }
        $out.y_train = nd_take($ya, $tr, 0);
        $out.y_test  = nd_take($ya, $te, 0);
    }
    return $out;
}

// ── KFold ───────────────────────────────────────────────────────────────────
class _KFold {
    def init($n_splits, $shuffle) {
        this.name_ = "KFold";
        $k = $n_splits;  if ($k == null) { $k = 5; }
        $s = $shuffle;   if ($s == null) { $s = false; }
        this.k_  = $k;
        this.sh_ = $s;
    }
    def name()          { return this.name_; }
    def get_n_splits()  { return this.k_; }
    def partitions()    { return true; }
    def shuffles()      { return this.sh_; }
    def split($X, $y, $groups) {
        $n = _nrows($X, "KFold.split");
        _check_k(this.k_, $n, "KFold.split");
        $idx = _arange_idx($n);
        if (this.sh_) { $idx = _permutation($n); }
        $b = _fold_bounds($n, this.k_);
        $out = [];
        $f = 0;
        while ($f < this.k_) {
            $test = nd_copy(nd_slice($idx, [[$b[$f], $b[$f + 1], 1]]));
            append($out, [_complement($n, $test), $test]);
            $f = $f + 1;
        }
        return $out;
    }
}
def KFold($n_splits, $shuffle) { return new _KFold($n_splits, $shuffle); }

// ── StratifiedKFold ─────────────────────────────────────────────────────────
// The fold-size rule applied PER CLASS, with the starting fold rotating from
// one class to the next (K22). Without the rotation every class puts its extra
// sample in fold 0: with three classes each having one extra, fold 0 ends up
// three rows larger than the last. Each class is still balanced to +/-1 either
// way, but the FOLDS are not, and a fold 10% larger than its neighbour makes
// the per-fold scores incomparable.
class _StratifiedKFold {
    def init($n_splits, $shuffle) {
        this.name_ = "StratifiedKFold";
        $k = $n_splits;  if ($k == null) { $k = 5; }
        $s = $shuffle;   if ($s == null) { $s = false; }
        this.k_  = $k;
        this.sh_ = $s;
    }
    def name()          { return this.name_; }
    def get_n_splits()  { return this.k_; }
    def partitions()    { return true; }
    def shuffles()      { return this.sh_; }
    def split($X, $y, $groups) {
        $who = "StratifiedKFold.split";
        $n   = _nrows($X, $who);
        _check_k(this.k_, $n, $who);
        $lab = _labels($y, $who);
        if (nd_shape($lab)[0] != $n) {
            throw $who + ": y has " + str(nd_shape($lab)[0]) + " labels for "
                  + str($n) + " rows";
        }
        $classes = nd_unique($lab);
        $nc = nd_shape($classes)[0];

        // one growing list of test indices per fold
        $buckets = [];
        $f = 0;
        while ($f < this.k_) { append($buckets, []); $f = $f + 1; }

        $rot = 0;
        $c = 0;
        while ($c < $nc) {
            $v    = nd_get($classes, $c);
            $mine = nd_astype(nd_nonzero(nd_equal($lab, $v, null)), "i64");
            $m    = nd_shape($mine)[0];
            if ($m < this.k_) {
                throw $who + ": class " + str($v) + " has only " + str($m)
                      + " member(s) but there are " + str(this.k_) + " folds, "
                      + "so some fold would contain none of it";
            }
            if (this.sh_) { $mine = nd_take($mine, _permutation($m), 0); }
            $cb = _fold_bounds($m, this.k_);
            $j = 0;
            while ($j < this.k_) {
                $dest = ($j + $rot) % this.k_;
                $p = $cb[$j];
                while ($p < $cb[$j + 1]) {
                    append($buckets[$dest], nd_get($mine, $p));
                    $p = $p + 1;
                }
                $j = $j + 1;
            }
            // rotate by the number of folds that got an EXTRA row, so the
            // next class's extras start where this one's left off
            $rot = ($rot + ($m % this.k_)) % this.k_;
            $c = $c + 1;
        }

        $out = [];
        $f = 0;
        while ($f < this.k_) {
            $test = nd_sort(_idx($buckets[$f]), null);
            append($out, [_complement($n, $test), $test]);
            $f = $f + 1;
        }
        return $out;
    }
}
def StratifiedKFold($n_splits, $shuffle) {
    return new _StratifiedKFold($n_splits, $shuffle);
}

// ── GroupKFold ──────────────────────────────────────────────────────────────
// No row of a group is ever in a different fold from its siblings, which is
// what makes it the right splitter when rows are not independent -- several
// readings from one patient, several frames from one video. Splitting such a
// group across the train/test line lets the model memorise the group and
// report a score it cannot reproduce on a new one.
//
// Groups are assigned largest-first to whichever fold is currently lightest,
// which is scikit-learn's rule. Ties break on the group VALUE so the result
// does not depend on hash or scan order, and there is no shuffle at all:
// with group sizes fixed, the assignment is determined.
class _GroupKFold {
    def init($n_splits) {
        this.name_ = "GroupKFold";
        $k = $n_splits;  if ($k == null) { $k = 5; }
        this.k_ = $k;
    }
    def name()          { return this.name_; }
    def get_n_splits()  { return this.k_; }
    def partitions()    { return true; }
    def shuffles()      { return false; }
    def split($X, $y, $groups) {
        $who = "GroupKFold.split";
        $n   = _nrows($X, $who);
        _check_k(this.k_, $n, $who);
        if ($groups == null) {
            throw $who + ": GroupKFold needs `groups` -- without it there is "
                  + "nothing to keep together, and KFold is the splitter you want";
        }
        $g = _labels($groups, $who);
        if (nd_shape($g)[0] != $n) {
            throw $who + ": groups has " + str(nd_shape($g)[0]) + " entries for "
                  + str($n) + " rows";
        }
        $uniq = nd_unique($g);
        $ng   = nd_shape($uniq)[0];
        if ($ng < this.k_) {
            throw $who + ": " + str($ng) + " group(s) cannot fill "
                  + str(this.k_) + " folds without splitting one";
        }

        // sizes, then an order: size descending, group value ascending on ties
        $sizes = [];
        $i = 0;
        while ($i < $ng) {
            append($sizes, nd_get(nd_count_nonzero(
                nd_equal($g, nd_get($uniq, $i), null), null, null), 0));
            $i = $i + 1;
        }
        // nd_argsort is ascending and stable, so sorting by -size over a list
        // that is already in ascending group order gives exactly the tie rule
        $neg = [];
        $i = 0;
        while ($i < $ng) { append($neg, 0.0 - $sizes[$i]); $i = $i + 1; }
        $order = nd_argsort(nd($neg, "f64"), null);

        $load    = [];
        $buckets = [];
        $f = 0;
        while ($f < this.k_) { append($load, 0); append($buckets, []); $f = $f + 1; }

        $i = 0;
        while ($i < $ng) {
            $gi = nd_get($order, $i);
            // the lightest fold; first one wins a tie, so this is determined
            $best = 0;
            $f = 1;
            while ($f < this.k_) {
                if ($load[$f] < $load[$best]) { $best = $f; }
                $f = $f + 1;
            }
            $members = nd_astype(nd_nonzero(
                nd_equal($g, nd_get($uniq, $gi), null)), "i64");
            $p = 0;
            while ($p < nd_shape($members)[0]) {
                append($buckets[$best], nd_get($members, $p));
                $p = $p + 1;
            }
            $load[$best] = $load[$best] + $sizes[$gi];
            $i = $i + 1;
        }

        $out = [];
        $f = 0;
        while ($f < this.k_) {
            if (len($buckets[$f]) == 0) {
                throw $who + ": fold " + str($f) + " came out empty -- the "
                      + "groups are too unevenly sized for " + str(this.k_)
                      + " folds";
            }
            $test = nd_sort(_idx($buckets[$f]), null);
            append($out, [_complement($n, $test), $test]);
            $f = $f + 1;
        }
        return $out;
    }
}
def GroupKFold($n_splits) { return new _GroupKFold($n_splits); }

// ── TimeSeriesSplit ─────────────────────────────────────────────────────────
// Train on the past, test on what comes next. Fold i trains on rows
// [0, (i+1)*t) and tests on [(i+1)*t, (i+2)*t), where t = floor(n / (k+1)).
//
// This is the only splitter here that does NOT partition the data -- the first
// t rows are never tested and the last n mod (k+1) rows are never used at all.
// That is the point of it, and it is why `partitions()` exists on the splitter
// protocol rather than being assumed by the checker (§4.1).
//
// THE INVARIANT IS ORDER, not coverage: every training index is strictly less
// than every test index. Shuffling a time series before cross-validating is the
// classic way to report a score a model cannot reproduce in production -- it
// trains on the future and is graded on the past.
class _TimeSeriesSplit {
    def init($n_splits) {
        this.name_ = "TimeSeriesSplit";
        $k = $n_splits;  if ($k == null) { $k = 5; }
        this.k_ = $k;
    }
    def name()          { return this.name_; }
    def get_n_splits()  { return this.k_; }
    def partitions()    { return false; }
    def shuffles()      { return false; }
    def split($X, $y, $groups) {
        $who = "TimeSeriesSplit.split";
        $n   = _nrows($X, $who);
        if (this.k_ == null || this.k_ != floor(this.k_) || this.k_ < 1) {
            throw $who + ": n_splits must be a whole number of at least 1, got "
                  + str(this.k_);
        }
        $t = floor($n / (this.k_ + 1));
        if ($t < 1) {
            throw $who + ": " + str($n) + " rows cannot make " + str(this.k_)
                  + " splits -- each needs at least one test row, so at least "
                  + str(this.k_ + 1) + " rows are required";
        }
        $out = [];
        $f = 0;
        while ($f < this.k_) {
            $trEnd = ($f + 1) * $t;
            $teEnd = ($f + 2) * $t;
            if ($f == this.k_ - 1) { $teEnd = $n; }   // the last fold takes the tail
            append($out, [_arange_idx($trEnd),
                          nd_copy(nd_slice(_arange_idx($n), [[$trEnd, $teEnd, 1]]))]);
            $f = $f + 1;
        }
        return $out;
    }
}
def TimeSeriesSplit($n_splits) { return new _TimeSeriesSplit($n_splits); }

// ── the splitter check ──────────────────────────────────────────────────────
// Tier 3 for this file, and it lives here rather than in the test for the same
// reason protocol_check lives in base.b ([K17]): one definition of the
// contract, inherited by everything added later.
//
// Returns a list of failure strings; empty means conforming. It does not throw,
// because a test wants every splitter's result and not just the first.
def splitter_check($sp, $X, $y, $groups) {
    $bad = [];
    if ($sp == null) { append($bad, "the splitter is null"); return $bad; }
    $n = _nrows($X, "splitter_check");

    $folds = $sp.split($X, $y, $groups);
    if (len($folds) != $sp.get_n_splits()) {
        append($bad, "get_n_splits() says " + str($sp.get_n_splits())
                     + " but split() returned " + str(len($folds)));
    }

    $seen = nd_zeros([$n], "f64");       // how many folds TEST each index
    $f = 0;
    while ($f < len($folds)) {
        $tr = $folds[$f][0];
        $te = $folds[$f][1];
        $tag = "fold " + str($f) + ": ";

        if (nd_size($te) == 0) { append($bad, $tag + "the test side is empty"); }
        if (nd_size($tr) == 0) { append($bad, $tag + "the training side is empty"); }

        if (nd_size($te) > 0) {
            $lo = nd_get(nd_min($te, null, null), 0);
            $hi = nd_get(nd_max($te, null, null), 0);
            if ($lo < 0 || $hi >= $n) {
                append($bad, $tag + "a test index is outside 0.." + str($n - 1));
            }
        }
        if (nd_size($tr) > 0) {
            $lo = nd_get(nd_min($tr, null, null), 0);
            $hi = nd_get(nd_max($tr, null, null), 0);
            if ($lo < 0 || $hi >= $n) {
                append($bad, $tag + "a training index is outside 0.." + str($n - 1));
            }
        }

        // no index twice within one side
        if (nd_size(nd_unique($te)) != nd_size($te)) {
            append($bad, $tag + "a test index appears twice");
        }
        if (nd_size(nd_unique($tr)) != nd_size($tr)) {
            append($bad, $tag + "a training index appears twice");
        }

        // THE ONE THAT MATTERS: no index on both sides of the same fold.
        // An overlap means the model is graded on rows it trained on, and the
        // score comes out better than the model.
        $m = nd_zeros([$n], "f64");
        if (nd_size($tr) > 0) { nd_put($m, $tr, 1.0); }
        $hits = 0;
        if (nd_size($te) > 0) {
            $hits = nd_get(nd_sum(nd_take($m, $te, 0), null, null), 0);
        }
        if ($hits != 0.0) {
            append($bad, $tag + "train and test overlap in " + str($hits)
                         + " index/indices -- the score would be inflated");
        }

        if (nd_size($te) > 0) {
            nd_put($seen, $te, nd_add(nd_take($seen, $te, 0), 1.0, null));
        }
        $f = $f + 1;
    }

    // Exhaustiveness, but only for the splitters that claim it. TimeSeriesSplit
    // deliberately never tests the first rows, so a single check that assumed
    // coverage would have to exempt it by name -- and the next splitter added
    // would be exempted by whoever adds it.
    if ($sp.partitions()) {
        $notOnce = nd_get(nd_count_nonzero(nd_not_equal($seen, 1.0, null), null, null), 0);
        if ($notOnce != 0) {
            append($bad, "partitions() is true but " + str($notOnce)
                         + " index/indices are not tested exactly once");
        }
    }
    if (!$sp.partitions()) {
        $twice = nd_get(nd_count_nonzero(nd_greater($seen, 1.0, null), null, null), 0);
        if ($twice != 0) {
            append($bad, "an index is tested in " + str($twice)
                         + " place(s) more than once");
        }
    }
    return $bad;
}

// ── cross-validation ────────────────────────────────────────────────────────

// A number means KFold(n, shuffle=false); a splitter is used as given. Folds
// are materialised ONCE here and handed to the caller, which is what lets a
// parameter search compare candidates on identical data (K21).
def as_splitter($cv) {
    if ($cv == null) { return KFold(5, false); }
    if (type($cv) == "number") { return KFold($cv, false); }
    return $cv;
}

def _fit_score($est, $X, $y, $tr, $te) {
    // A CLONE per fold, never the estimator itself (K20, [K2]). Reusing a
    // fitted estimator across folds leaks the previous fold's fit into this
    // one and still produces a plausible, merely inflated number.
    $m  = $est.clone();
    $Xtr = take_rows($X, $tr);
    $Xte = take_rows($X, $te);
    $ytr = null;
    $yte = null;
    if ($y != null) {
        $ya = $y;
        if (type($ya) == "list") { $ya = nd($ya, "f64"); }
        $ytr = nd_take($ya, $tr, 0);
        $yte = nd_take($ya, $te, 0);
    }
    $m.fit($Xtr, $ytr);
    return [$m, $m.score($Xte, $yte)];
}

def cross_validate($est, $X, $y, $cv) {
    if ($est == null) { throw "cross_validate: the estimator is null"; }
    $sp    = as_splitter($cv);
    $folds = $sp.split($X, $y, null);
    $scores = [];
    $times  = [];
    $models = [];
    $f = 0;
    while ($f < len($folds)) {
        $t0 = clock();
        $r  = _fit_score($est, $X, $y, $folds[$f][0], $folds[$f][1]);
        append($times, clock() - $t0);
        append($models, $r[0]);
        append($scores, $r[1]);
        $f = $f + 1;
    }
    return {"test_score": nd($scores, "f64"),
            "fit_time_ms": nd($times, "f64"),
            "estimators": $models,
            "n_splits": len($folds)};
}

def cross_val_score($est, $X, $y, $cv) {
    return cross_validate($est, $X, $y, $cv).test_score;
}

// ── parameter grids ─────────────────────────────────────────────────────────

// The cartesian product of {param: [values]}, in a DETERMINED order: keys
// sorted, then the last key varying fastest. Bantu dicts are insertion
// ordered, so without the sort the candidate order -- and therefore which of
// two equally-scoring candidates is reported as best -- would depend on how
// the caller happened to write the literal.
def _sorted_keys($d) {
    $ks = $d.keys();
    $i = 1;
    while ($i < len($ks)) {               // insertion sort; a grid is tiny
        $j = $i;
        while ($j > 0 && $ks[$j] < $ks[$j - 1]) {
            $tmp = $ks[$j];
            $ks[$j] = $ks[$j - 1];
            $ks[$j - 1] = $tmp;
            $j = $j - 1;
        }
        $i = $i + 1;
    }
    return $ks;
}

def param_grid($grid) {
    if ($grid == null || len($grid) == 0) { return [{}]; }
    $keys = _sorted_keys($grid);
    $out  = [{}];
    $i = 0;
    while ($i < len($keys)) {
        $k    = $keys[$i];
        $vals = $grid[$k];
        if (type($vals) != "list") {
            throw "param_grid: '" + str($k) + "' must map to a LIST of values, got "
                  + type($vals);
        }
        if (len($vals) == 0) {
            throw "param_grid: '" + str($k) + "' has an empty list of values";
        }
        $next = [];
        $p = 0;
        while ($p < len($out)) {
            $v = 0;
            while ($v < len($vals)) {
                $one = {};
                for $ek, $ev in $out[$p].items() { $one[$ek] = $ev; }
                $one[$k] = $vals[$v];
                append($next, $one);
                $v = $v + 1;
            }
            $p = $p + 1;
        }
        $out = $next;
        $i = $i + 1;
    }
    return $out;
}

// Every grid key must be addressable on the estimator, checked BEFORE the
// first fit (K24). `set_params` refuses an unknown key now, but finding out
// after five folds of fitting is a worse way to learn it than finding out
// immediately -- and for a Pipeline the key may be "step__param", which only
// the pipeline can validate.
def _check_grid($est, $grid, $who) {
    if ($grid == null) { return null; }
    $probe = $est.clone();
    for $k, $v in $grid.items() {
        $one = {};
        $one[$k] = $v[0];
        try { $probe.set_params($one); }
        catch ($e) {
            throw $who + ": the grid names '" + str($k) + "', which this "
                  + str($est.name()) + " does not accept. " + str($e);
        }
    }
    return null;
}

// ── GridSearchCV ────────────────────────────────────────────────────────────
// Exhaustive search, then a refit of the winner on all the data.
//
// THE SPLITS ARE TAKEN ONCE, before the candidate loop (K21). With a shuffled
// splitter, re-splitting per candidate would compare candidates on different
// folds and the winner would be partly a draw of the RNG.
class _GridSearchCV extends Estimator {
    def init($est, $grid, $cv) {
        super();
        this.name_ = "GridSearchCV";
        if ($est == null) { throw "GridSearchCV: the estimator is null"; }
        this.est_   = $est;
        this.grid_  = $grid;
        this.cv_    = $cv;
        this.task_  = $est.task();
        this.params_ = {"n_candidates": len(param_grid($grid))};
        this.best_score_  = 0.0;
        this.best_params_ = {};
        this.best_        = null;
        this.results_     = [];
    }

    def fit($X, $y) {
        _check_grid(this.est_, this.grid_, "GridSearchCV.fit");
        $cands = param_grid(this.grid_);
        $sp    = as_splitter(this.cv_);
        $folds = $sp.split($X, $y, null);          // ONCE, for every candidate
        $results = [];
        $bestI   = -1;
        $bestS   = 0.0;
        $c = 0;
        while ($c < len($cands)) {
            $tot = 0.0;
            $per = [];
            $f = 0;
            while ($f < len($folds)) {
                $m = this.est_.clone();
                $m.set_params($cands[$c]);
                $r = _fit_score($m, $X, $y, $folds[$f][0], $folds[$f][1]);
                append($per, $r[1]);
                $tot = $tot + $r[1];
                $f = $f + 1;
            }
            $mean = $tot / (1.0 * len($folds));
            append($results, {"params": $cands[$c], "mean_test_score": $mean,
                              "fold_scores": $per});
            // Strictly greater, so the FIRST candidate wins a tie -- and the
            // candidate order is determined (keys sorted), so a tie resolves
            // the same way on every run and on every machine.
            if ($bestI < 0 || $mean > $bestS) { $bestI = $c; $bestS = $mean; }
            $c = $c + 1;
        }
        this.results_     = $results;
        this.best_score_  = $bestS;
        this.best_params_ = $cands[$bestI];
        // Refit on ALL the data: the whole point of a search is the model you
        // end up with, and the fold models each saw only k-1/k of the rows.
        this.best_ = this.est_.clone();
        this.best_.set_params($cands[$bestI]);
        this.best_.fit($X, $y);
        this.fitted_ = true;
        return this;
    }

    def predict($X) {
        this._check_fitted("GridSearchCV.predict");
        return this.best_.predict($X);
    }
    def transform($X) {
        this._check_fitted("GridSearchCV.transform");
        return this.best_.transform($X);
    }
    def score($X, $y) {
        this._check_fitted("GridSearchCV.score");
        return this.best_.score($X, $y);
    }
    def best_estimator() { this._check_fitted("GridSearchCV.best_estimator");
                           return this.best_; }
    def best_params()    { return this.best_params_; }
    def best_score()     { return this.best_score_; }
    def cv_results()     { return this.results_; }
    def clone()          { return GridSearchCV(this.est_.clone(), this.grid_, this.cv_); }
}
def GridSearchCV($est, $grid, $cv) { return new _GridSearchCV($est, $grid, $cv); }

// ── RandomizedSearchCV ──────────────────────────────────────────────────────
// The same machinery over a SAMPLE of the grid. Asking for at least as many
// candidates as the grid holds makes it exhaustive rather than drawing the same
// candidate twice: sampling with replacement would spend the budget on repeats
// and report a best score over fewer distinct configurations than requested.
class _RandomizedSearchCV extends Estimator {
    def init($est, $grid, $n_iter, $cv) {
        super();
        this.name_ = "RandomizedSearchCV";
        if ($est == null) { throw "RandomizedSearchCV: the estimator is null"; }
        $it = $n_iter;  if ($it == null) { $it = 10; }
        if ($it != floor($it) || $it < 1) {
            throw "RandomizedSearchCV: n_iter must be a whole number of at "
                  + "least 1, got " + str($it);
        }
        this.est_    = $est;
        this.grid_   = $grid;
        this.iter_   = $it;
        this.cv_     = $cv;
        this.task_   = $est.task();
        this.params_ = {"n_iter": $it};
        this.best_score_  = 0.0;
        this.best_params_ = {};
        this.best_        = null;
        this.results_     = [];
    }

    def fit($X, $y) {
        _check_grid(this.est_, this.grid_, "RandomizedSearchCV.fit");
        $all = param_grid(this.grid_);
        $pick = $all;
        if (this.iter_ < len($all)) {
            // A permutation, then the first n_iter of it: distinct candidates,
            // and reproducible from nd_seed alone ([C4], K18).
            $perm = _permutation(len($all));
            $pick = [];
            $i = 0;
            while ($i < this.iter_) {
                append($pick, $all[nd_get($perm, $i)]);
                $i = $i + 1;
            }
        }
        $sp    = as_splitter(this.cv_);
        $folds = $sp.split($X, $y, null);
        $results = [];
        $bestI = -1;
        $bestS = 0.0;
        $c = 0;
        while ($c < len($pick)) {
            $tot = 0.0;
            $per = [];
            $f = 0;
            while ($f < len($folds)) {
                $m = this.est_.clone();
                $m.set_params($pick[$c]);
                $r = _fit_score($m, $X, $y, $folds[$f][0], $folds[$f][1]);
                append($per, $r[1]);
                $tot = $tot + $r[1];
                $f = $f + 1;
            }
            $mean = $tot / (1.0 * len($folds));
            append($results, {"params": $pick[$c], "mean_test_score": $mean,
                              "fold_scores": $per});
            if ($bestI < 0 || $mean > $bestS) { $bestI = $c; $bestS = $mean; }
            $c = $c + 1;
        }
        this.results_     = $results;
        this.best_score_  = $bestS;
        this.best_params_ = $pick[$bestI];
        this.best_ = this.est_.clone();
        this.best_.set_params($pick[$bestI]);
        this.best_.fit($X, $y);
        this.fitted_ = true;
        return this;
    }

    def predict($X) {
        this._check_fitted("RandomizedSearchCV.predict");
        return this.best_.predict($X);
    }
    def transform($X) {
        this._check_fitted("RandomizedSearchCV.transform");
        return this.best_.transform($X);
    }
    def score($X, $y) {
        this._check_fitted("RandomizedSearchCV.score");
        return this.best_.score($X, $y);
    }
    def best_estimator() { this._check_fitted("RandomizedSearchCV.best_estimator");
                           return this.best_; }
    def best_params()    { return this.best_params_; }
    def best_score()     { return this.best_score_; }
    def cv_results()     { return this.results_; }
    def clone() {
        return RandomizedSearchCV(this.est_.clone(), this.grid_, this.iter_, this.cv_);
    }
}
def RandomizedSearchCV($est, $grid, $n_iter, $cv) {
    return new _RandomizedSearchCV($est, $grid, $n_iter, $cv);
}
