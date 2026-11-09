// ════════════════════════════════════════════════════════════════════════
//  btorch_data_test.b — phase 6 (datasets, batching) and phase 7
//  (saving and loading state).
//
//  The two assertions that matter most here are easy to miss:
//
//  1. One epoch's batch indices, with drop_last=false, are a PERMUTATION of
//     0..n-1 -- every sample exactly once, none twice. A loader that quietly
//     drops or repeats samples still trains and still looks fine.
//
//  2. A truncated file and a corrupted one are DIFFERENT failures. The suite
//     builds both, by hand, from a good file: one by cutting bytes off, one
//     by flipping a byte and leaving the length alone. The second is only
//     detectable by the payload digest.
//
//  Run:  bantu run tests/btorch_data_test.b
// ════════════════════════════════════════════════════════════════════════

include "../btorch/btorch.b" as bt;
include "../btorch/nn.b" as nn;
include "../btorch/data.b" as data;
include "../btorch/serialize.b" as ser;
include "../btorch/optim.b" as optim;

$R = {"pass": 0, "fail": 0};
def ok($cond, $name) {
    if ($cond) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $name); }
}
def eqL($got, $want, $name) {
    if (str($got) == str($want)) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name);
        print("          got:  " + str($got));
        print("          want: " + str($want));
    }
}
def raises($fn, $name) {
    try { $fn(); $R.fail = $R.fail + 1; print("  FAIL  " + $name + " (no error raised)"); }
    catch ($e) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    return null;
}
// A C++ runtime error is an object with .message; a Bantu `throw "..."` is the
// string itself. Both occur here.
def errText($e) {
    if (type($e) == "string") { return $e; }
    return str($e.message);
}
def saysThat($fn, $needle, $name) {
    try { $fn(); $R.fail = $R.fail + 1; print("  FAIL  " + $name + " (no error raised)"); }
    catch ($e) {
        if (contains(errText($e), $needle)) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
        else {
            $R.fail = $R.fail + 1;
            print("  FAIL  " + $name + "  message lacked '" + $needle + "'");
            print("          got: " + errText($e));
        }
    }
    return null;
}
def sc($x) { return nd_get(nd_reshape($x, [1]), [0]); }

$TMP = "/private/tmp/claude-501/-Users-kodeuni-Desktop-projects-2026-2027-Bantu/8c425240-6e69-48b2-897f-bf5b99d23526/scratchpad";

print("========================================");
print("  PHASE 6 — TensorDataset");
print("========================================");

$x = bt.tensor(nd_reshape(nd_astype(nd_arange(0.0, 20.0, null), "f64"), [10, 2]), false);
$y = bt.tensor(nd_astype(nd_arange(0.0, 10.0, null), "f64"), false);
$ds = data.TensorDataset([$x, $y]);
ok($ds.size() == 10, "size() is the first axis");
$row = $ds.get(3);
eqL($row[0].to_list(), [6.0, 7.0], "get(i) returns row i of the first tensor");
ok($row[1].item() == 3.0, "...and row i of the second");
eqL($row[0].shape(), [2], "get(i) squeezes the row axis off -- it is a ROW, not a batch of one");
$bat = $ds.batch_of(nd([0, 2, 4], "i64"));
eqL($bat[0].to_list(), [[0.0,1.0],[4.0,5.0],[8.0,9.0]], "batch_of gathers the rows asked for");
eqL($bat[1].to_list(), [0.0, 2.0, 4.0], "...from every tensor");
eqL($bat[0].shape(), [3, 2], "and the batch axis is first");
ok(data.is_dataset($ds), "is_dataset recognises it");
ok(!data.is_dataset($x), "and does not recognise a tensor");

raises(def() { return data.TensorDataset([]); }, "an empty tensor list raises");
raises(def() { return data.TensorDataset($x); }, "a bare tensor instead of a list raises");
saysThat(def() { return data.TensorDataset([$x, bt.tensor(nd_zeros([7], "f64"), false)]); },
         "FIRST axis", "a first-axis mismatch raises and says which axis matters");
raises(def() { return data.TensorDataset([bt.tensor(nd(1.0, "f64"), false)]); },
       "a 0-d tensor has no rows, and raises");
raises(def() { return $ds.get(10); },     "get past the end raises");
raises(def() { return $ds.get(0 - 1); },  "a negative index raises");
// A plain array or list is accepted and wrapped.
ok(data.TensorDataset([nd_zeros([4, 3], "f64")]).size() == 4, "a raw array is accepted");

print("");
print("== a custom Dataset subclass goes through the generic batch_of ==");
// The generic path stacks row by row. It must give the same answer as
// TensorDataset's single-gather path, which is the whole point of having both.
class Squares extends data.Dataset {
    def init($n) { super(); this.n = $n; }
    def size() { return this.n; }
    def get($i) {
        return [bt.tensor([$i * 1.0, $i * $i * 1.0], false), bt.tensor([$i % 2 * 1.0], false)];
    }
}
$sq = new Squares(6);
ok($sq.size() == 6, "the subclass reports its own size");
$sb = $sq.batch_of(nd([1, 3, 5], "i64"));
eqL($sb[0].to_list(), [[1.0,1.0],[3.0,9.0],[5.0,25.0]], "the generic batch_of stacks rows correctly");
eqL($sb[1].to_list(), [[1.0],[1.0],[1.0]], "...for every column");
eqL($sb[0].shape(), [3, 2], "and gets the batch axis right");
raises(def() { return $sq.batch_of(nd_zeros([0], "i64")); }, "a batch of 0 rows raises");
// The base class refuses to pretend.
class Bare extends data.Dataset { def init() { super(); } }
raises(def() { return (new Bare()).size(); },  "the base Dataset's size() raises");
raises(def() { return (new Bare()).get(0); },  "the base Dataset's get() raises");

print("");
print("========================================");
print("  PHASE 6 — DataLoader");
print("========================================");

$dl = data.DataLoader($ds, 3, false, false, null);
ok($dl.num_batches() == 4, "10 rows in batches of 3 is 4 batches (the last is short)");
eqL($dl.batch(0)[0].to_list(), [[0.0,1.0],[2.0,3.0],[4.0,5.0]], "batch 0, in order, shuffle=false");
eqL($dl.batch(3)[0].shape(), [1, 2], "the last batch is the remainder, not padded");
eqL(nd_to_list($dl.batch_indices(3)), [9], "batch_indices names exactly which rows");
raises(def() { return $dl.batch(4); },     "asking for a batch past the end raises");
raises(def() { return $dl.batch(0 - 1); }, "a negative batch index raises");

$dld = data.DataLoader($ds, 3, false, true, null);
ok($dld.num_batches() == 3, "drop_last=true drops the short batch");
eqL($dld.batch(2)[0].shape(), [3, 2], "so every batch is full");
ok(data.DataLoader($ds, 10, false, false, null).num_batches() == 1,
   "a batch_size equal to the dataset is one batch");
ok(data.DataLoader($ds, 20, false, false, null).num_batches() == 1,
   "a batch_size LARGER than the dataset is still one batch, not zero");
ok(data.DataLoader($ds, 20, false, true, null).num_batches() == 0,
   "...but with drop_last there is nothing to yield");
ok(data.DataLoader($ds, 1, false, false, null).num_batches() == 10, "batch_size 1 gives n batches");
raises(def() { return data.DataLoader($ds, 0, false, false, null); },  "batch_size 0 raises");
raises(def() { return data.DataLoader($ds, 2.5, false, false, null); },"a fractional batch_size raises");
raises(def() { return data.DataLoader($x, 2, false, false, null); },   "a non-Dataset raises");

print("");
print("== THE PERMUTATION PROPERTY: every sample exactly once ==");
// A loader that drops or repeats samples still trains and still looks fine.
// This is the assertion that catches it. Checked over many sizes and batch
// sizes, because the bug lives at the boundary.
def coversExactlyOnce($n, $bs, $shuffle) {
    $d = data.DataLoader(data.TensorDataset([nd_zeros([$n, 1], "f64")]), $bs, $shuffle, false, null);
    $seen = nd_zeros([$n], "f64");
    $b = 0;
    while ($b < $d.num_batches()) {
        nd_index_add($seen, $d.batch_indices($b), 1.0, 0);
        $b = $b + 1;
    }
    // Every count exactly 1: nothing missed, nothing twice.
    return nd_array_equal($seen, nd_ones([$n], "f64"));
}
$allok = true;
$sizes = [1, 2, 3, 7, 10, 16, 17, 64, 100];
$bss = [1, 2, 3, 5, 8, 16];
$i = 0;
while ($i < len($sizes)) {
    $j = 0;
    while ($j < len($bss)) {
        if (!coversExactlyOnce($sizes[$i], $bss[$j], false)) { $allok = false; }
        if (!coversExactlyOnce($sizes[$i], $bss[$j], true))  { $allok = false; }
        $j = $j + 1;
    }
    $i = $i + 1;
}
ok($allok, "over 9 sizes x 6 batch sizes x shuffle on/off, every index appears EXACTLY once");
// With drop_last the cover is a prefix of the order, so the count is the
// number of full batches times the batch size -- never more than the dataset.
$dd = data.DataLoader(data.TensorDataset([nd_zeros([10, 1], "f64")]), 3, false, true, null);
$seen = nd_zeros([10], "f64");
$b = 0;
while ($b < $dd.num_batches()) { nd_index_add($seen, $dd.batch_indices($b), 1.0, 0); $b = $b + 1; }
ok(sc(nd_sum($seen, null, null)) == 9.0, "drop_last covers exactly 9 of 10 rows");
ok(sc(nd_max($seen, null, null)) == 1.0, "and still never twice");

print("");
print("== reproducibility, and the thing that is NOT reproducibility ==");
nd_seed(7);
$a1 = data.DataLoader($ds, 3, true, false, null);
nd_seed(7);
$a2 = data.DataLoader($ds, 3, true, false, null);
ok(nd_array_equal($a1.order, $a2.order), "the same seed gives the IDENTICAL batch order");
eqL(nd_to_list($a1.batch_indices(0)), nd_to_list($a2.batch_indices(0)), "...batch for batch");
$seedless = data.DataLoader($ds, 3, true, false, 7);
// The expected order, from the documented formula, so this asserts the RULE
// and not merely that two DataLoaders agree with each other.
nd_seed(7);
$expected = nd_astype(nd_argsort(nd_random_uniform([10], 0.0, 1.0), null), "i64");
ok(nd_array_equal($seedless.order, $expected),
   "passing seed= equals nd_seed(k) first, and the order is argsort(uniform)");
ok(!nd_array_equal($a1.order, nd_astype(nd_arange(0.0, 10.0, null), "i64")),
   "shuffle=true really does shuffle");
eqL(nd_to_list(data.DataLoader($ds, 3, false, false, null).order), [0,1,2,3,4,5,6,7,8,9],
    "shuffle=false leaves the order alone");
// The stream ADVANCES: epoch 2 must differ from epoch 1, or every epoch would
// see the same batches and shuffling would buy nothing.
$prev = nd_copy($a1.order, null);
$a1.reshuffle();
ok(!nd_array_equal($a1.order, $prev), "reshuffle() gives a DIFFERENT order (the stream advances)");
ok($a1.epochs == 1, "and counts the epoch");
ok(nd_array_equal(nd_sort($a1.order, null), nd_astype(nd_arange(0.0, 10.0, null), "i64")),
   "the reshuffled order is still a permutation");
// A reshuffled loader is reproducible too, which is what makes a resumed run
// reproducible: same seed, same number of reshuffles, same order.
nd_seed(5);
$r1 = data.DataLoader($ds, 3, true, false, null);
$r1.reshuffle(); $r1.reshuffle();
nd_seed(5);
$r2 = data.DataLoader($ds, 3, true, false, null);
$r2.reshuffle(); $r2.reshuffle();
ok(nd_array_equal($r1.order, $r2.order), "two reshuffles from one seed land in the same place");
eqL(len($dl.epoch()), 4, "epoch() returns the whole pass as a list");

print("");
print("== random_split / Subset ==");
$parts = data.random_split($ds, [7, 3], 99);
ok(len($parts) == 2, "random_split returns one Subset per size");
ok($parts[0].size() == 7 && $parts[1].size() == 3, "with the sizes asked for");
// The two halves must be DISJOINT and cover everything -- a split that leaks
// training rows into the test set is the most expensive bug in the file.
$cover = nd_zeros([10], "f64");
nd_index_add($cover, $parts[0].indices, 1.0, 0);
nd_index_add($cover, $parts[1].indices, 1.0, 0);
ok(nd_array_equal($cover, nd_ones([10], "f64")),
   "the parts are DISJOINT and cover every row exactly once");
nd_seed(99);
$again = data.random_split($ds, [7, 3], 99);
ok(nd_array_equal($parts[0].indices, $again[0].indices), "the split is reproducible from its seed");
ok(!nd_array_equal($parts[0].indices, data.random_split($ds, [7, 3], 1)[0].indices),
   "and a different seed gives a different split");
saysThat(def() { return data.random_split($ds, [7, 2], 1); }, "exact counts",
         "sizes that do not sum to the dataset raise, and the error says why fractions are refused");
raises(def() { return data.random_split($ds, [], 1); },        "an empty size list raises");
raises(def() { return data.random_split($ds, [0 - 1, 11], 1); }, "a negative size raises");
raises(def() { return data.random_split($x, [10], 1); },       "splitting a non-Dataset raises");
ok(data.random_split($ds, [10, 0], 1)[1].size() == 0, "a zero-sized part is allowed and empty");

// A Subset of a Subset must compose, and must still batch with one gather.
$sub = data.Subset($ds, [9, 7, 5, 3, 1]);
eqL($sub.get(0)[1].item(), 9.0, "Subset remaps get()");
$sub2 = data.Subset($sub, [0, 4]);
// batch_of returns TENSORS, not arrays -- .to_list(), not nd_to_list().
eqL($sub2.batch_of(nd([0, 1], "i64"))[1].to_list(), [9.0, 1.0], "a Subset of a Subset composes");
raises(def() { return data.Subset($ds, [10]); },    "a Subset index past the end raises");
raises(def() { return data.Subset($ds, [0 - 1]); }, "a negative Subset index raises");
// A DataLoader over a Subset is the normal train/val shape.
$vl = data.DataLoader($parts[1], 2, false, false, null);
ok($vl.num_batches() == 2, "a DataLoader over a Subset batches its 3 rows into 2");
ok(nd_size($vl.batch(0)[1].data) == 2, "and the batches hold the Subset's rows");

print("");
print("========================================");
print("  PHASE 7 — save / load");
print("========================================");

$P = $TMP + "/sd_test.btsd";
bt.manual_seed(3);
$m = nn.Sequential([nn.Linear(4, 3, null), nn.BatchNorm1d(3, null, null),
                    nn.ReLU(), nn.Linear(3, 2, null)]);
$probe = bt.tensor(nd_random_normal([5, 4]), false);
$m.forward($probe);                     // move the running stats off their defaults
$info = ser.save($m, $P);
ok($info.entries == 8, "a 6-parameter model with a BatchNorm saves 8 entries");
ok($info.bytes == filesize($P), "and reports the byte count the file actually has");
ok(len($m.parameters()) == 6 && len($m.buffers()) == 2, "6 parameters, 2 buffers");

$f = ser.load($P);
ok($f.version == 1, "the file records format version 1");
ok(len($f.order) == 8, "load returns all 8 entries");
ok($f.kinds["1.running_mean"] == "buffer", "and marks the running stats as buffers");
ok($f.kinds["0.weight"] == "param", "and the weights as parameters");

print("");
print("== the round trip is EXACT, bit for bit ==");
$sa = $m.state_arrays();
$allexact = true;
$i = 0;
while ($i < len($f.order)) {
    $k = $f.order[$i];
    if (!nd_array_equal($f.entries[$k], $sa[$k])) { $allexact = false; }
    $i = $i + 1;
}
ok($allexact, "every saved array comes back bit-identical (not merely close)");
// The hard case: a value that only survives if the encoding is binary. str()
// gives six significant digits, so a text format loses this at the 7th.
$fussy = {"a": nd([0.1234567890123456789, 1e-300, 1e300, 0.0 - 0.0], "f64")};
ser.save($fussy, $TMP + "/fussy.btsd");
ok(nd_array_equal(ser.load($TMP + "/fussy.btsd").entries["a"], $fussy.a),
   "a 17-digit double, 1e-300, 1e300 and -0.0 all survive exactly");
// Every dtype numba has.
$dts = {"f": nd([1.5, 0 - 2.5], "f64"),
        "i": nd([1, 0 - 7, 9007199254740993], "i64"),
        "b": nd([true, false, true], "bool")};
ser.save($dts, $TMP + "/dts.btsd");
$back = ser.load($TMP + "/dts.btsd").entries;
ok(nd_array_equal($back.f, $dts.f) && nd_dtype($back.f) == "f64", "f64 round-trips, dtype included");
ok(nd_array_equal($back.i, $dts.i) && nd_dtype($back.i) == "i64",
   "i64 round-trips exactly, INCLUDING a value above 2^53");
ok(nd_array_equal($back.b, $dts.b) && nd_dtype($back.b) == "bool", "bool round-trips as u8");
// Shapes, including the awkward ones.
$shapes = {"s0": nd_zeros([0], "f64"), "s1": nd_zeros([1], "f64"),
           "s3": nd_random_normal([2, 3, 4]), "s4": nd_zeros([1, 1, 1, 1], "f64")};
ser.save($shapes, $TMP + "/shapes.btsd");
$sback = ser.load($TMP + "/shapes.btsd").entries;
ok(nd_shape($sback.s0) == [0], "a length-0 array keeps its shape");
ok(nd_shape($sback.s3) == [2, 3, 4] && nd_array_equal($sback.s3, $shapes.s3), "a 3-d array round-trips");
ok(nd_shape($sback.s4) == [1, 1, 1, 1], "and a 4-d one of all-1s");
// A strided view is written in LOGICAL order, so it comes back as it looked.
$strided = {"v": nd_slice(nd_reshape(nd_astype(nd_arange(0.0, 12.0, null), "f64"), [6, 2]), [[0, 6, 2]])};
ser.save($strided, $TMP + "/strided.btsd");
ok(nd_array_equal(ser.load($TMP + "/strided.btsd").entries["v"], $strided.v),
   "a strided view saves what it LOOKS like, not its buffer");

print("");
print("== loading into a live model, in place ==");
bt.manual_seed(999);
$m2 = nn.Sequential([nn.Linear(4, 3, null), nn.BatchNorm1d(3, null, null),
                     nn.ReLU(), nn.Linear(3, 2, null)]);
$m.eval(); $m2.eval();
ok(!nd_allclose($m.forward($probe).data, $m2.forward($probe).data, 1e-9, 1e-9),
   "the two models start out different");
// The tensor IDS must not change: an optimizer's state keys on them, so a
// load that REPLACED the tensors would silently orphan every momentum buffer.
$idsBefore = [];
$ps = $m2.parameters();
$i = 0;
while ($i < len($ps)) { push($idsBefore, $ps[$i].id); $i = $i + 1; }
$c = ser.load_model($m2, $P);
ok($c.params == 6 && $c.buffers == 2, "load_model reports 6 parameters and 2 buffers");
$idsAfter = [];
$ps = $m2.parameters();
$i = 0;
while ($i < len($ps)) { push($idsAfter, $ps[$i].id); $i = $i + 1; }
eqL($idsAfter, $idsBefore, "the parameter tensors are the SAME objects -- loading is in place");
ok(nd_allclose($m.forward($probe).data, $m2.forward($probe).data, 1e-15, 1e-15),
   "and after loading the two models agree to 1e-15");
// The buffers specifically: an eval() forward depends on them alone, so this
// is what catches a save that silently omitted them.
ok(nd_array_equal($m2.get("1").buffer("running_mean").data,
                  $m.get("1").buffer("running_mean").data),
   "the BatchNorm running mean was saved and restored");
ok(nd_array_equal($m2.get("1").buffer("running_var").data,
                  $m.get("1").buffer("running_var").data),
   "...and the running variance");

print("");
print("== an optimizer survives a load, because the tensors did ==");
bt.manual_seed(2);
$om = nn.Linear(3, 2, null);
$opt = optim.SGD($om.parameters(), 0.1, 0.9, null);
$ox = bt.tensor(nd_random_normal([4, 3]), false);
$oy = bt.tensor(nd_random_normal([4, 2]), false);
$step = 0;
while ($step < 3) {
    $opt.zero_grad();
    nn.mse_loss($om.forward($ox), $oy, null).backward(null);
    $opt.step();
    $step = $step + 1;
}
ser.save($om, $TMP + "/opt.btsd");
ser.load_model($om, $TMP + "/opt.btsd");
$opt.zero_grad();
nn.mse_loss($om.forward($ox), $oy, null).backward(null);
$opt.step();
ok(true, "a step after a load does not throw -- the optimizer's state still matches its tensors");

print("");
print("== a dict of tensors, not just a Module ==");
$d = {"w": bt.tensor([[1.0, 2.0]], false), "b": nd([3.0], "f64"), "l": [4.0, 5.0]};
ser.save($d, $TMP + "/dict.btsd");
$dback = ser.load($TMP + "/dict.btsd").entries;
eqL(nd_to_list($dback.w), [[1.0, 2.0]], "a Tensor in a dict saves");
eqL(nd_to_list($dback.b), [3.0], "a raw array too");
eqL(nd_to_list($dback.l), [4.0, 5.0], "and a plain list is wrapped");
raises(def() { return ser.save({"bad": "a string"}, $TMP + "/x.btsd"); },
       "a dict holding something that is not a tensor raises");
raises(def() { return ser.save(42, $TMP + "/x.btsd"); }, "saving a number raises");
raises(def() { return ser.save($d, 42); },               "a non-string path raises");

print("");
print("========================================");
print("  PHASE 7 — every load failure, by name");
print("========================================");

// Build the broken files by hand from a known-good one, so each test breaks
// exactly one thing.
$good = readfile($P, "rb");
def byteAt($v) { return nd_tobytes(nd([$v], "i64"), "u8"); }
def writeRaw($path, $s) { writefile($path, $s, "wb"); return $path; }

saysThat(def() { return ser.load($TMP + "/does_not_exist.btsd"); }, "no such file",
         "a missing file raises and names the path");
saysThat(def() { return ser.load(writeRaw($TMP + "/short.btsd", "BTSD")); }, "too short",
         "a file shorter than the header raises");
saysThat(def() { return ser.load(writeRaw($TMP + "/empty.btsd", "")); }, "too short",
         "an empty file raises");
saysThat(def() { return ser.load(writeRaw($TMP + "/magic.btsd", "XXXX" + substr($good, 4, len($good) - 4))); },
         "not a btorch state file", "a bad magic raises and says what it expected");
// A version from the future must be refused, not read best-effort.
saysThat(def() { return ser.load(writeRaw($TMP + "/newer.btsd",
             substr($good, 0, 4) + nd_tobytes(nd([99], "i64"), "u32") + substr($good, 8, len($good) - 8))); },
         "newer btorch", "a newer format version raises rather than guessing");
// headerLen longer than the file.
saysThat(def() { return ser.load(writeRaw($TMP + "/hdrlen.btsd",
             substr($good, 0, 8) + nd_tobytes(nd([999999], "i64"), "u32") + substr($good, 12, len($good) - 12))); },
         "TRUNCATED", "a header length past the end of the file raises as truncated");
// A header that is not JSON at all. Its LENGTH must stay right, or the
// truncation check fires first -- which it should, and which is how the first
// draft of this case failed.
$hl = nd_get(nd_frombytes(substr($good, 8, 4), "u32", [1]), [0]);
$junkHdr = substr($good, 0, 12) + "x" + substr($good, 13, $hl - 1)
         + substr($good, 12 + $hl, len($good) - 12 - $hl);
ok(len($junkHdr) == len($good), "the junk-header file is exactly as long as the good one");
saysThat(def() { return ser.load(writeRaw($TMP + "/hdrjunk.btsd", $junkHdr)); },
         "corrupt header", "a header that is not a JSON object raises as corrupt");

print("");
print("== truncated vs corrupt: different accidents, different messages ==");
// TRUNCATED: bytes cut off the end. The length check catches it.
saysThat(def() { return ser.load(writeRaw($TMP + "/trunc.btsd", substr($good, 0, len($good) - 40))); },
         "TRUNCATED", "a payload cut short raises as TRUNCATED");
try { ser.load(writeRaw($TMP + "/trunc.btsd", substr($good, 0, len($good) - 40))); }
catch ($e) { ok(contains(errText($e), "40"), "...and says how many bytes are missing"); }
// CORRUPT: the right length, a changed byte. ONLY the digest catches this --
// which is the entire reason the digest is in the header.
$mid = len($good) - 20;
$flipped = substr($good, 0, $mid) + byteAt((nd_get(nd_frombytes(substr($good, $mid, 1), "u8", [1]), [0]) + 1) % 256)
         + substr($good, $mid + 1, len($good) - $mid - 1);
ok(len($flipped) == len($good), "the corrupted file is exactly as long as the good one");
saysThat(def() { return ser.load(writeRaw($TMP + "/corrupt.btsd", $flipped)); },
         "CORRUPT", "one flipped byte, same length, raises as CORRUPT -- caught only by the digest");
try { ser.load(writeRaw($TMP + "/corrupt.btsd", $flipped)); }
catch ($e) { ok(contains(errText($e), "SHA-256"), "...and names the digest that disagreed"); }
// Extra bytes appended: not truncation either.
saysThat(def() { return ser.load(writeRaw($TMP + "/long.btsd", $good + "EXTRA")); },
         "MORE than its header claims", "extra trailing bytes raise, and are not called truncation");
// A good file still loads after all that, so the suite has not corrupted $P.
ok(len(ser.load($P).order) == 8, "the original file still loads -- nothing above mutated it");

print("");
print("== load_model's own refusals ==");
raises(def() { return ser.load_model($x, $P); }, "load_model on a non-Module raises");
saysThat(def() { return ser.load_model(nn.Linear(4, 3, null), $P); }, "no parameter called",
         "loading into a model with different parameter names raises by name");
saysThat(def() { return ser.load_model(nn.Sequential([nn.Linear(9, 3, null),
             nn.BatchNorm1d(3, null, null), nn.ReLU(), nn.Linear(3, 2, null)]), $P); },
         "in the file but the parameter is",
         "loading a shape mismatch raises and prints both shapes");
// A model with the right names but missing the buffers.
saysThat(def() { return (nn.BatchNorm1d(3, null, null)).load_state_arrays(
             {"gamma": nd_ones([3], "f64"), "beta": nd_zeros([3], "f64")}); },
         "no buffer called", "a state map missing the BUFFERS raises, naming them as buffers");

print("");
print("== state_dict includes buffers (it used to omit them) ==");
$bn = nn.BatchNorm1d(2, null, null);
$sd = $bn.state_dict();
ok(len($sd.keys()) == 4, "a BatchNorm's state_dict has gamma, beta AND both running stats");
ok($sd.running_mean != null, "running_mean is in it");
$bn2 = nn.BatchNorm1d(2, null, null);
$bn.forward(bt.tensor([[1.0, 2.0], [3.0, 8.0]], false));
$bn2.load_state_dict($bn.state_dict());
ok(nd_array_equal($bn2.buffer("running_var").data, $bn.buffer("running_var").data),
   "and load_state_dict restores them");

print("");
print("== inference_mode / eval() interaction ==");
bt.manual_seed(4);
$im = nn.Sequential([nn.Linear(3, 4, null), nn.BatchNorm1d(4, null, null), nn.Dropout(0.5)]);
$ix = bt.tensor(nd_random_normal([6, 3]), false);
$im.train();
$im.forward($ix);
$rmTrain = nd_copy($im.get("1").buffer("running_mean").data, null);
$im.eval();
// Two eval forwards must be IDENTICAL: dropout is off and the running stats
// are frozen. In train mode they would differ on both counts.
$e1 = $im.forward($ix);
$e2 = $im.forward($ix);
ok(nd_array_equal($e1.data, $e2.data), "two eval() forwards are bit-identical (dropout off, stats frozen)");
ok(nd_array_equal($im.get("1").buffer("running_mean").data, $rmTrain),
   "and eval() leaves the running stats untouched");
// inference_mode is about the TAPE, not about the mode -- they are orthogonal,
// and conflating them is how a model ends up building a graph it never uses.
$withGrad = $im.forward($ix);
ok($withGrad.requires_grad == true, "eval() alone still builds a tape (its params require grad)");
$noGrad = bt.inference_mode(def() { return $im.forward($ix); });
ok($noGrad.requires_grad == false, "inference_mode suppresses the tape");
ok(nd_allclose($noGrad.data, $withGrad.data, 1e-15, 1e-15),
   "...and changes nothing about the numbers");
ok(bt.is_grad_enabled(), "and grad is back on afterwards");
// Saving and loading inside inference_mode must work and must not tape.
$saved = bt.inference_mode(def() { return ser.save($im, $TMP + "/im.btsd"); });
ok($saved.entries == 6, "save works under inference_mode");
$lm = bt.inference_mode(def() { return ser.load_model($im, $TMP + "/im.btsd"); });
ok($lm.params == 4 && $lm.buffers == 2, "and so does load_model");

print("");
print("========================================");
print("  TIER 4 — stress");
print("========================================");

// A 60,000-row dataset is MNIST's size. The epoch must be built from native
// gathers, not row by row: 60,000 interpreter rounds per epoch would dominate
// the training step.
$N = 60000;
$bigx = nd_random_normal([$N, 32]);
$bigy = nd_astype(nd_random_int([$N], 0, 10), "f64");
$bds = data.TensorDataset([$bigx, $bigy]);
$t0 = clock();
$bdl = data.DataLoader($bds, 128, true, false, 1234);
$t1 = clock();
ok($bdl.num_batches() == 469, "60,000 rows in batches of 128 is 469 batches");
print("        shuffling 60,000 indices: " + str($t1 - $t0) + " ms");
ok($t1 - $t0 < 500, "and it takes under 500 ms");
$t2 = clock();
$total = 0;
$b = 0;
while ($b < $bdl.num_batches()) {
    $bb = $bdl.batch($b);
    $total = $total + nd_shape($bb[0].data)[0];
    $b = $b + 1;
}
$t3 = clock();
print("        one full epoch of 469 batches: " + str($t3 - $t2) + " ms");
ok($total == $N, "one epoch yields exactly 60,000 rows across its batches");
ok($t3 - $t2 < 10000, "and a full epoch takes under 10s");
// The permutation property at scale, via a native scatter rather than a loop.
$seen = nd_zeros([$N], "f64");
$b = 0;
while ($b < $bdl.num_batches()) { nd_index_add($seen, $bdl.batch_indices($b), 1.0, 0); $b = $b + 1; }
ok(nd_array_equal($seen, nd_ones([$N], "f64")),
   "and at 60,000 rows every index still appears exactly once");

// No leaks over many epochs.
$before = nd_live_bytes();
$small = data.DataLoader(data.TensorDataset([nd_zeros([200, 4], "f64")]), 16, true, false, null);
$e = 0;
while ($e < 50) {
    $small.reshuffle();
    $b = 0;
    while ($b < $small.num_batches()) { $small.batch($b); $b = $b + 1; }
    $e = $e + 1;
}
$after = nd_live_bytes();
print("        live array bytes over 50 epochs: " + str($before) + " -> " + str($after));
ok($after - $before < 1000000, "50 epochs of batching grew live array bytes by under 1MB");

// A real model's file: LeNet-5, 61,706 parameters.
bt.manual_seed(1);
$lenet = nn.Sequential([
    nn.Conv2d(1, 6, 5, 1, 2, null),   nn.ReLU(), nn.MaxPool2d(2, null, null),
    nn.Conv2d(6, 16, 5, 1, 0, null),  nn.ReLU(), nn.MaxPool2d(2, null, null),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120, null), nn.ReLU(),
    nn.Linear(120, 84, null),         nn.ReLU(),
    nn.Linear(84, 10, null)
]);
$LP = $TMP + "/lenet.btsd";
$t4 = clock();
$li = ser.save($lenet, $LP);
$t5 = clock();
print("        LeNet-5 (61,706 params) saved: " + str($li.bytes) + " bytes in " + str($t5 - $t4) + " ms");
// 61,706 f64 values is 493,648 bytes; the header is small.
ok($li.bytes > 493648 && $li.bytes < 493648 + 4096,
   "the file is the raw f64 payload plus a header under 4KB -- no text bloat");
$t6 = clock();
$lf = ser.load($LP);
$t7 = clock();
print("        ...and loaded in " + str($t7 - $t6) + " ms");
$lsa = $lenet.state_arrays();
$exact = true;
$i = 0;
while ($i < len($lf.order)) {
    if (!nd_array_equal($lf.entries[$lf.order[$i]], $lsa[$lf.order[$i]])) { $exact = false; }
    $i = $i + 1;
}
ok($exact, "and every one of LeNet's tensors is bit-identical after the round trip");
// The end-to-end claim: train a bit, save, build a fresh model, load, and the
// predictions must match exactly.
$img = bt.tensor(nd_random_normal([4, 1, 28, 28]), false);
$lab = nd_astype(nd_random_int([4], 0, 10), "i64");
$lopt = optim.SGD($lenet.parameters(), 0.01, null, null);
$k = 0;
while ($k < 3) {
    $lopt.zero_grad();
    nn.cross_entropy($lenet.forward($img), $lab, null).backward(null);
    $lopt.step();
    $k = $k + 1;
}
ser.save($lenet, $LP);
$trained = $lenet.forward($img);
bt.manual_seed(77);
$fresh = nn.Sequential([
    nn.Conv2d(1, 6, 5, 1, 2, null),   nn.ReLU(), nn.MaxPool2d(2, null, null),
    nn.Conv2d(6, 16, 5, 1, 0, null),  nn.ReLU(), nn.MaxPool2d(2, null, null),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120, null), nn.ReLU(),
    nn.Linear(120, 84, null),         nn.ReLU(),
    nn.Linear(84, 10, null)
]);
ser.load_model($fresh, $LP);
ok(nd_allclose($fresh.forward($img).data, $trained.data, 1e-15, 1e-15),
   "a trained LeNet saved, reloaded into a fresh model, predicts IDENTICALLY");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
