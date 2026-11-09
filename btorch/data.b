// ════════════════════════════════════════════════════════════════════════════
//  btorch/data.b — datasets and batching (phase 6)
//
//  Design: docs/btorch-architecture.md §7.5.1.
//
//  Three things worth knowing before changing anything here:
//
//  1. Bantu has no iterator protocol, so DataLoader is EXPLICIT:
//     num_batches() / batch(i) / reshuffle(). A pass over an epoch is an
//     ordinary `while` over an index, which is the form the rest of btorch
//     already uses, and it makes the batch boundaries testable one at a time.
//
//  2. Shuffling is nd_argsort(nd_random_uniform([n])). That is a genuine
//     uniform permutation for one native sort, where Fisher-Yates in Bantu
//     would be n interpreter iterations -- 60,000 of them for MNIST, every
//     epoch. It also draws from numba's seeded stream, so nd_seed makes an
//     epoch reproducible.
//
//  3. The property that actually protects training is that one epoch's
//     batches, with drop_last=false, use every index EXACTLY ONCE. A loader
//     that quietly drops or repeats samples still trains and still looks
//     fine. tests/btorch_data_test.b asserts the permutation directly.
// ════════════════════════════════════════════════════════════════════════════

include "btorch.b" as bt;

def _need() {
    if (nd_zeros == null) {
        throw "btorch.data needs numba's nd_* builtins -- the bantu binary is "
            + "older than this library. Run 'bantu doctor'.";
    }
    return null;
}

// Every index list in this file is an i64 array, never a Bantu list: a list of
// 60,000 indices is 60,000 Value boxes, and nd_take wants an array anyway.
def _perm($n) {
    if ($n == 0) { return nd_zeros([0], "i64"); }
    return nd_astype(nd_argsort(nd_random_uniform([$n], 0.0, 1.0), null), "i64");
}
def _iota($n) {
    if ($n == 0) { return nd_zeros([0], "i64"); }
    return nd_astype(nd_arange(0.0, $n * 1.0, null), "i64");
}

// ── Dataset ─────────────────────────────────────────────────────────────────
// The base contract: size() and get($i). A subclass overriding those works
// with Subset, random_split and DataLoader without knowing they exist.
class Dataset {
    def init() { this.__bt_dataset = true; }
    def size() { throw "btorch.data: this Dataset does not implement size()"; }
    def get($i) { throw "btorch.data: this Dataset does not implement get(i)"; }
    // Rows $idx (an i64 array) of every tensor, as one batch. Overridden by
    // TensorDataset with a single native gather; the generic version is the
    // honest fallback for a Dataset that only knows how to produce one row.
    def batch_of($idx) {
        $n = nd_size($idx);
        if ($n == 0) { throw "btorch.data: cannot build a batch of 0 rows"; }
        $first = this.get(nd_get($idx, [0]));
        $cols = len($first);
        $stacks = [];
        $c = 0;
        while ($c < $cols) { push($stacks, [$first[$c]]); $c = $c + 1; }
        $i = 1;
        while ($i < $n) {
            $row = this.get(nd_get($idx, [$i]));
            if (len($row) != $cols) {
                throw "btorch.data: row " + str(nd_get($idx, [$i])) + " has "
                    + str(len($row)) + " tensors but row " + str(nd_get($idx, [0]))
                    + " had " + str($cols) + " -- a Dataset must return the same "
                    + "shape of row every time";
            }
            $c = 0;
            while ($c < $cols) { push($stacks[$c], $row[$c]); $c = $c + 1; }
            $i = $i + 1;
        }
        $out = [];
        $c = 0;
        while ($c < $cols) { push($out, bt.stack($stacks[$c], 0)); $c = $c + 1; }
        return $out;
    }
}

def is_dataset($x) {
    if (type($x) != "instance") { return false; }
    return $x.__bt_dataset == true;
}

// ── TensorDataset ───────────────────────────────────────────────────────────
// N tensors sharing a first dimension. batch_of is ONE nd_take per tensor, so
// a batch costs two native gathers rather than batch_size interpreter rounds.
class _TensorDataset extends Dataset {
    def init($tensors) {
        super();
        if (type($tensors) != "list" || len($tensors) == 0) {
            throw "btorch.data.TensorDataset(tensors): expected a non-empty list of tensors";
        }
        $ts = [];
        $n = 0;
        $i = 0;
        while ($i < len($tensors)) {
            $t = $tensors[$i];
            if (!bt.is_tensor($t)) { $t = bt.tensor($t, false); }
            $s = nd_shape($t.data);
            if (len($s) == 0) {
                throw "btorch.data.TensorDataset: tensor " + str($i)
                    + " is 0-d, so it has no rows to index";
            }
            if ($i == 0) { $n = $s[0]; }
            if ($s[0] != $n) {
                throw "btorch.data.TensorDataset: tensor 0 has " + str($n)
                    + " rows but tensor " + str($i) + " has " + str($s[0])
                    + " -- every tensor must agree on the FIRST axis, which is "
                    + "the one being indexed";
            }
            push($ts, $t);
            $i = $i + 1;
        }
        this.tensors = $ts;
        this.n = $n;
    }
    def size() { return this.n; }
    def get($i) {
        if ($i < 0 || $i >= this.n) {
            throw "btorch.data: index " + str($i) + " is out of range for "
                + str(this.n) + " rows";
        }
        $out = [];
        $c = 0;
        while ($c < len(this.tensors)) {
            // squeeze the length-1 row axis off, so get(i) returns the ROW and
            // not a batch of one -- which is what batch_of then stacks.
            push($out, bt.squeeze(bt.take(this.tensors[$c], [$i], 0), 0));
            $c = $c + 1;
        }
        return $out;
    }
    def batch_of($idx) {
        $out = [];
        $c = 0;
        while ($c < len(this.tensors)) {
            push($out, bt.take(this.tensors[$c], $idx, 0));
            $c = $c + 1;
        }
        return $out;
    }
}
def TensorDataset($tensors) { _need(); return new _TensorDataset($tensors); }

// ── Subset ──────────────────────────────────────────────────────────────────
// A view by index list. random_split returns these, so a split costs no copy
// of the underlying data at all.
class _Subset extends Dataset {
    def init($dataset, $indices) {
        super();
        if (!is_dataset($dataset)) {
            throw "btorch.data.Subset(dataset, indices): the first argument is not a Dataset";
        }
        $ix = $indices;
        if (type($ix) == "list") { $ix = nd($ix, "i64"); }
        $ix = nd_astype($ix, "i64");
        $n = $dataset.size();
        if (nd_size($ix) > 0) {
            $lo = nd_get(nd_reshape(nd_min($ix, null, null), [1]), [0]);
            $hi = nd_get(nd_reshape(nd_max($ix, null, null), [1]), [0]);
            if ($lo < 0 || $hi >= $n) {
                throw "btorch.data.Subset: an index outside [0, " + str($n)
                    + ") (saw " + str($lo) + " .. " + str($hi) + ")";
            }
        }
        this.dataset = $dataset;
        this.indices = $ix;
    }
    def size() { return nd_size(this.indices); }
    def get($i) {
        if ($i < 0 || $i >= nd_size(this.indices)) {
            throw "btorch.data: index " + str($i) + " is out of range for "
                + str(nd_size(this.indices)) + " rows";
        }
        return this.dataset.get(nd_get(this.indices, [$i]));
    }
    def batch_of($idx) {
        // Compose the index maps rather than going row by row, so a Subset of
        // a TensorDataset still batches with one native gather.
        return this.dataset.batch_of(nd_take(this.indices, $idx, 0));
    }
}
def Subset($dataset, $indices) { _need(); return new _Subset($dataset, $indices); }

// ── random_split ────────────────────────────────────────────────────────────
// The sizes must sum to the dataset exactly. NOT a fraction-based API: 0.8/0.2
// of 1000 is unambiguous, of 999 is not, and silently rounding is how a test
// set ends up with one sample in it.
def random_split($dataset, $sizes, $seed) {
    _need();
    if (!is_dataset($dataset)) {
        throw "btorch.data.random_split(dataset, sizes, seed): the first argument is not a Dataset";
    }
    if (type($sizes) != "list" || len($sizes) == 0) {
        throw "btorch.data.random_split: sizes must be a non-empty list of counts";
    }
    $n = $dataset.size();
    $total = 0;
    $i = 0;
    while ($i < len($sizes)) {
        if (type($sizes[$i]) != "number" || $sizes[$i] < 0) {
            throw "btorch.data.random_split: size " + str($i) + " is not a non-negative count";
        }
        $total = $total + $sizes[$i];
        $i = $i + 1;
    }
    if ($total != $n) {
        throw "btorch.data.random_split: the sizes sum to " + str($total)
            + " but the dataset has " + str($n) + " rows -- give exact counts, "
            + "because rounding a fraction is how a split silently loses rows";
    }
    if ($seed != null) { nd_seed($seed); }
    $perm = _perm($n);
    $out = [];
    $at = 0;
    $i = 0;
    while ($i < len($sizes)) {
        $part = nd_zeros([0], "i64");
        if ($sizes[$i] > 0) {
            $part = nd_copy(nd_slice($perm, [[$at, $at + $sizes[$i], 1]]), null);
        }
        push($out, Subset($dataset, $part));
        $at = $at + $sizes[$i];
        $i = $i + 1;
    }
    return $out;
}

// ── DataLoader ──────────────────────────────────────────────────────────────
class _DataLoader extends Dataset {
    def init($dataset, $batch_size, $shuffle, $drop_last, $seed) {
        super();
        if (!is_dataset($dataset)) {
            throw "btorch.data.DataLoader(dataset, ...): the first argument is not a Dataset";
        }
        $bs = $batch_size;
        if ($bs == null) { $bs = 1; }
        if (type($bs) != "number" || $bs < 1 || $bs != nd_get(nd_floor_divide(nd_full([1], $bs * 1.0), 1.0, null), [0])) {
            throw "btorch.data.DataLoader: batch_size must be a whole number >= 1 (got "
                + str($batch_size) + ")";
        }
        this.dataset = $dataset;
        this.batch_size = $bs;
        this.shuffle = $shuffle == true;
        this.drop_last = $drop_last == true;
        this.epochs = 0;
        if ($seed != null) { nd_seed($seed); }
        this.order = this._order();
    }
    def _order() {
        $n = this.dataset.size();
        if (this.shuffle) { return _perm($n); }
        return _iota($n);
    }
    // Start the next epoch. With shuffle=true this draws a NEW permutation
    // from the stream -- it does not restart it, so epoch 2 differs from
    // epoch 1 and the two facts (reproducible from a seed; different between
    // epochs) are both true at once.
    def reshuffle() {
        this.order = this._order();
        this.epochs = this.epochs + 1;
        return this;
    }
    def size() { return this.dataset.size(); }
    def num_batches() {
        $n = nd_size(this.order);
        $full = nd_get(nd_floor_divide(nd_full([1], $n * 1.0), this.batch_size * 1.0, null), [0]);
        if (this.drop_last) { return $full; }
        if ($full * this.batch_size < $n) { return $full + 1; }
        return $full;
    }
    // The indices this batch covers, as an i64 array. Exposed because it is
    // what the permutation test needs, and because a caller doing something
    // unusual should not have to re-derive the boundaries.
    def batch_indices($b) {
        $nb = this.num_batches();
        if ($b < 0 || $b >= $nb) {
            throw "btorch.data.DataLoader: batch " + str($b) + " is out of range for "
                + str($nb) + " batches";
        }
        $from = $b * this.batch_size;
        $to = $from + this.batch_size;
        $n = nd_size(this.order);
        if ($to > $n) { $to = $n; }
        return nd_copy(nd_slice(this.order, [[$from, $to, 1]]), null);
    }
    def batch($b) { return this.dataset.batch_of(this.batch_indices($b)); }
    // The whole epoch as a list, for callers who prefer it to an index loop.
    def epoch() {
        $out = [];
        $b = 0;
        $nb = this.num_batches();
        while ($b < $nb) { push($out, this.batch($b)); $b = $b + 1; }
        return $out;
    }
    def get($i) { return this.dataset.get($i); }
}
def DataLoader($dataset, $batch_size, $shuffle, $drop_last, $seed) {
    _need();
    return new _DataLoader($dataset, $batch_size, $shuffle, $drop_last, $seed);
}
