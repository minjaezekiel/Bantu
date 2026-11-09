// ════════════════════════════════════════════════════════════════════════
//  btorch_indexing_test.b — phase 2: indexing, extremes and shape ops.
//
//  Tier 1 (feature), tier 2 (differential, against values worked out by
//  hand), tier 3 (gradcheck) and tier 4 (stress) for everything phase A5
//  unblocked: take / index_select / embedding / one_hot / index_add,
//  max / min / maximum / minimum / argmax / argmin, and
//  permute / squeeze / unsqueeze / movedim.
//
//  The gradient that matters most here is take's, because it is the one that
//  is quietly wrong if the scatter overwrites instead of accumulating. So the
//  duplicate-index cases are not an edge case in this file -- they are the
//  main case, and the hand-computed answers below all have repeats in them.
//
//  Run:  bantu run tests/btorch_indexing_test.b
// ════════════════════════════════════════════════════════════════════════

include "../btorch/btorch.b" as bt;
include "../btorch/nn.b" as nn;

$R = {"pass": 0, "fail": 0};
def ok($cond, $name) {
    if ($cond) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $name); }
}
def close($got, $want, $tol, $name) {
    $d = $got - $want;
    if ($d < 0) { $d = 0 - $d; }
    if ($d <= $tol) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name + "  got " + str($got) + " want " + str($want));
    }
}
def eqList($got, $want, $name) {
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
def gc($f, $inputs, $name) {
    $r = bt.gradcheck($f, $inputs, null, null);
    if ($r.ok) { $R.pass = $R.pass + 1; print("  ok    gradcheck " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  gradcheck " + $name + "  worst rel err " + str($r.worst)
              + " at " + str($r.where));
    }
    return null;
}

bt.manual_seed(20261109);

print("========================================");
print("  TIER 1+2 — take, and the gradient that");
print("  only accumulation gets right");
print("========================================");

$w = bt.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], true);
$g = bt.take($w, [0, 2, 0], 0);
eqList($g.to_list(), [[1.0, 2.0], [5.0, 6.0], [1.0, 2.0]], "take gathers rows, repeats included");
eqList($g.shape(), [3, 2], "the gathered shape is [len(indices), dim]");

// Row 0 was taken TWICE, so d(sum)/d(row 0) is 2 -- not 1. This single
// assertion is the one that fails if the backward scatter overwrites.
$g.sum(null, null).backward(null);
eqList(nd_to_list($w.grad), [[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]],
       "a row taken twice gets gradient 2, not 1 (the accumulate)");

// Weighted, so the two contributions are distinguishable rather than both 1.
$w2 = bt.tensor([[1.0, 2.0], [3.0, 4.0]], true);
$coef = bt.tensor([[1.0, 10.0], [100.0, 1000.0], [2.0, 20.0]], false);
bt.sum(bt.mul(bt.take($w2, [0, 1, 0], 0), $coef), null, null).backward(null);
eqList(nd_to_list($w2.grad), [[3.0, 30.0], [100.0, 1000.0]],
       "two different weights on one row add (1+2 and 10+20)");

// index_select is the same thing with PyTorch's argument order.
$w3 = bt.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], true);
eqList(bt.index_select($w3, 1, [2, 0]).to_list(), [[3.0, 1.0], [6.0, 4.0]],
       "index_select(axis=1) picks columns");
bt.index_select($w3, 1, [2, 0, 2]).sum(null, null).backward(null);
eqList(nd_to_list($w3.grad), [[1.0, 0.0, 2.0], [1.0, 0.0, 2.0]],
       "a column taken twice gets 2 on axis 1 as well");

// Negative indices survive the round trip.
$w4 = bt.tensor([1.0, 2.0, 3.0], true);
eqList(bt.take($w4, [-1, -1, 0], 0).to_list(), [3.0, 3.0, 1.0], "negative indices gather from the end");
bt.take($w4, [-1, -1, 0], 0).sum(null, null).backward(null);
eqList(nd_to_list($w4.grad), [1.0, 0.0, 2.0], "and scatter back to the end");

// A tensor of indices works as well as a list, and is not differentiated.
$ids = bt.tensor([0.0, 0.0], false);
$w5 = bt.tensor([[7.0], [8.0]], true);
eqList(bt.take($w5, $ids, 0).to_list(), [[7.0], [7.0]], "a Tensor can carry the indices");
bt.take($w5, $ids, 0).sum(null, null).backward(null);
eqList(nd_to_list($w5.grad), [[2.0], [0.0]], "and the gradient still accumulates");
ok($ids.grad == null, "the index tensor gets no gradient of its own");

// The snapshot: mutating the caller's index buffer after the forward must not
// change where the gradient goes.
$iarr = nd([0, 0], "i64");
$w6 = bt.tensor([[1.0], [2.0]], true);
$out6 = bt.take($w6, $iarr, 0);
nd_set($iarr, [0], 1);
nd_set($iarr, [1], 1);
$out6.sum(null, null).backward(null);
eqList(nd_to_list($w6.grad), [[2.0], [0.0]],
       "indices are snapshotted: a buffer reused after the forward cannot redirect the gradient");

// axis is required, deliberately.
raises(def() { return bt.take($w, [0], null); }, "take without an axis raises rather than guessing");
raises(def() { return bt.take($w, [3], 0); },    "an out-of-range index raises");
raises(def() { return bt.take($w, "no", 0); },   "a string as indices raises");

print("");
print("== embedding: the same gather, named for what it is ==");
$E = bt.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], true);
eqList(bt.embedding($E, [2, 0, 2]).to_list(), [[3.0, 3.0], [1.0, 1.0], [3.0, 3.0]],
       "embedding looks up rows");
bt.embedding($E, [2, 0, 2]).sum(null, null).backward(null);
eqList(nd_to_list($E.grad), [[1.0, 1.0], [0.0, 0.0], [2.0, 2.0]],
       "a token seen twice in one batch accumulates both gradients");

print("");
print("== one_hot is a constant, built natively ==");
eqList(bt.one_hot([2, 0, 1], 3).to_list(), [[0.0,0.0,1.0],[1.0,0.0,0.0],[0.0,1.0,0.0]],
       "one_hot places exactly one 1 per row");
ok(bt.one_hot([0], 3).requires_grad == false, "one_hot carries no gradient");
eqList(bt.one_hot([1, 1, 1], 2).to_list(), [[0.0,1.0],[0.0,1.0],[0.0,1.0]],
       "repeated classes do NOT accumulate to 2 -- each row is its own scatter");
raises(def() { return bt.one_hot([3], 3); },     "a class index == classes raises");
raises(def() { return bt.one_hot([0 - 1], 3); }, "a negative class index raises");
raises(def() { return bt.one_hot([[0, 1]], 3); },"a nested index list raises");

print("");
print("== index_add as an op: both sides carry gradient ==");
$base = bt.tensor([1.0, 2.0, 3.0], true);
$vals = bt.tensor([10.0, 20.0], true);
$r = bt.index_add($base, [0, 0], $vals, 0);
eqList($r.to_list(), [31.0, 2.0, 3.0], "index_add accumulates both values into cell 0");
ok(nd_to_list($base.data) == [1.0, 2.0, 3.0], "and leaves its input untouched (functional at this level)");
$r.sum(null, null).backward(null);
eqList(nd_to_list($base.grad), [1.0, 1.0, 1.0], "the destination's gradient passes straight through");
eqList(nd_to_list($vals.grad), [1.0, 1.0], "the values' gradient is gathered from where they landed");
// Weighted so the gather is visible.
$b2 = bt.tensor([0.0, 0.0], true);
$v2 = bt.tensor([1.0, 1.0, 1.0], true);
bt.sum(bt.mul(bt.index_add($b2, [1, 0, 1], $v2, 0),
              bt.tensor([5.0, 7.0], false)), null, null).backward(null);
eqList(nd_to_list($v2.grad), [7.0, 5.0, 7.0], "each value's gradient is the weight of the cell it hit");

print("");
print("== max / min: ties split, which is amax semantics (B16) ==");
$x = bt.tensor([[1.0, 5.0, 2.0], [7.0, 3.0, 7.0]], true);
eqList(bt.max($x, 1, null).to_list(), [5.0, 7.0], "max along axis 1");
eqList(bt.max($x, 1, true).shape(), [2, 1], "keepdims keeps the axis");
eqList(bt.min($x, 1, null).to_list(), [1.0, 3.0], "min along axis 1");
close(bt.max($x, null, null).item(), 7.0, 1e-15, "a global max reduces to one element");
bt.max($x, 1, null).sum(null, null).backward(null);
eqList(nd_to_list($x.grad), [[0.0, 1.0, 0.0], [0.5, 0.0, 0.5]],
       "row 0 routes 1 to its single max; row 1 SPLITS 1 between its two 7s");
// The split is what keeps the total gradient equal to the incoming one.
$xs = bt.tensor([[4.0, 4.0, 4.0, 4.0]], true);
bt.max($xs, 1, null).sum(null, null).backward(null);
eqList(nd_to_list($xs.grad), [[0.25, 0.25, 0.25, 0.25]], "four-way tie splits into quarters summing to 1");
$xg = bt.tensor([2.0, 9.0, 9.0], true);
bt.max($xg, null, null).backward(null);
eqList(nd_to_list($xg.grad), [0.0, 0.5, 0.5], "a global max splits its ties too");

print("");
print("== maximum / minimum: elementwise, against a tensor or a number ==");
$a = bt.tensor([1.0, 5.0, 3.0], true);
$b = bt.tensor([4.0, 2.0, 3.0], true);
eqList(bt.maximum($a, $b).to_list(), [4.0, 5.0, 3.0], "maximum picks elementwise");
eqList(bt.minimum($a, $b).to_list(), [1.0, 2.0, 3.0], "minimum picks elementwise");
bt.maximum($a, $b).sum(null, null).backward(null);
eqList(nd_to_list($a.grad), [0.0, 1.0, 1.0], "maximum sends 1 to the winner, and a tie to a");
eqList(nd_to_list($b.grad), [1.0, 0.0, 0.0], "and the loser gets 0 -- the two always sum to 1");
$c = bt.tensor([0 - 1.0, 2.0], true);
eqList(bt.maximum($c, 0.0).to_list(), [0.0, 2.0], "maximum against a bare number is relu");
bt.maximum($c, 0.0).sum(null, null).backward(null);
eqList(nd_to_list($c.grad), [0.0, 1.0], "and its gradient is relu's");
eqList($a.clamp_min(4.0).to_list(), [4.0, 5.0, 4.0], "clamp_min is maximum, as a method");

print("");
print("== argmax / argmin are constants ==");
eqList(bt.argmax($x, 1).to_list(), [1.0, 0.0], "argmax along axis 1 (first of a tie)");
eqList(bt.argmin($x, 1).to_list(), [0.0, 1.0], "argmin along axis 1");
ok(bt.argmax($x, 1).requires_grad == false, "argmax has no gradient");
ok(bt.argmax($x, null).item() == 3.0, "a global argmax is a flat index");

print("");
print("== permute / squeeze / unsqueeze / movedim ==");
$p = bt.tensor(nd_random_normal([2, 3, 4]), true);
eqList(bt.permute($p, [2, 0, 1]).shape(), [4, 2, 3], "permute reorders the axes");
bt.permute($p, [2, 0, 1]).sum(null, null).backward(null);
eqList(nd_shape($p.grad), [2, 3, 4], "and its gradient comes back in the ORIGINAL shape");
// A permutation that is not its own inverse: the backward must apply the
// INVERSE, not the same permutation again.
$p2 = bt.tensor(nd_reshape(nd_astype(nd_arange(0.0, 24.0, null), "f64"), [2, 3, 4]), true);
$coef3 = bt.tensor(nd_reshape(nd_astype(nd_arange(0.0, 24.0, null), "f64"), [4, 2, 3]), false);
bt.sum(bt.mul(bt.permute($p2, [2, 0, 1]), $coef3), null, null).backward(null);
// grad[i,j,k] = coef[k,i,j]; coef is 0..23 in C order over [4,2,3], so
// coef[k,i,j] = k*6 + i*3 + j.
$want = [];
$i = 0;
while ($i < 2) {
    $row = [];
    $j = 0;
    while ($j < 3) {
        $col = [];
        $k = 0;
        while ($k < 4) { push($col, $k * 6.0 + $i * 3.0 + $j * 1.0); $k = $k + 1; }
        push($row, $col);
        $j = $j + 1;
    }
    push($want, $row);
    $i = $i + 1;
}
eqList(nd_to_list($p2.grad), $want, "permute's backward applies the inverse permutation, not the same one");
raises(def() { return bt.permute($p, [0, 1]); }, "permute with the wrong number of axes raises");
raises(def() { return bt.permute($p, null); },   "permute without axes raises");

$u = bt.tensor([1.0, 2.0, 3.0], true);
eqList(bt.unsqueeze($u, 0).shape(), [1, 3], "unsqueeze(0) adds a leading axis");
eqList(bt.unsqueeze($u, 1).shape(), [3, 1], "unsqueeze(1) adds a trailing axis");
bt.unsqueeze($u, 0).sum(null, null).backward(null);
eqList(nd_shape($u.grad), [3], "unsqueeze's gradient is squeezed back");
$sq = bt.tensor(nd_zeros([1, 3, 1], "f64"), true);
eqList(bt.squeeze($sq, null).shape(), [3], "squeeze(null) drops every length-1 axis");
eqList(bt.squeeze($sq, 0).shape(), [3, 1], "squeeze(0) drops only that one");
bt.squeeze($sq, null).sum(null, null).backward(null);
eqList(nd_shape($sq.grad), [1, 3, 1], "squeeze's gradient is unsqueezed back");
eqList(bt.movedim($p, 0, 2).shape(), [3, 4, 2], "movedim moves one axis to the end");
eqList(bt.movedim($p, 2, 0).shape(), [4, 2, 3], "and back to the front");
eqList(bt.movedim($p, 0 - 1, 0).shape(), [4, 2, 3], "a negative source axis counts from the end");

print("");
print("== the same ops as Tensor methods ==");
eqList($w.take([1], 0).to_list(), [[3.0, 4.0]], "t.take()");
eqList($w.index_select(0, [1]).to_list(), [[3.0, 4.0]], "t.index_select()");
eqList($x.max(1, null).to_list(), [5.0, 7.0], "t.max()");
eqList($x.argmax(1).to_list(), [1.0, 0.0], "t.argmax()");
eqList($p.permute([2, 1, 0]).shape(), [4, 3, 2], "t.permute()");
eqList($u.unsqueeze(0).shape(), [1, 3], "t.unsqueeze()");
eqList($p.movedim(0, 2).shape(), [3, 4, 2], "t.movedim()");
eqList($a.maximum($b).to_list(), [4.0, 5.0, 3.0], "t.maximum()");

print("");
print("========================================");
print("  TIER 3 — gradcheck, against central");
print("  differences");
print("========================================");

// take, with duplicates in every case: the numerical gradient knows nothing
// about scatter semantics, so agreement is independent evidence that the
// accumulate is right.
gc(def($in) { return bt.sum(bt.take($in[0], [0, 2, 0], 0), null, null); },
   [bt.tensor(nd_random_normal([3, 2]), true)], "take axis 0, index 0 twice");
// EVERY coefficient tensor below is drawn ONCE, outside the closure.
// Drawing it inside makes the function non-deterministic, and gradcheck then
// compares f(x+h) and f(x-h) computed with DIFFERENT coefficients -- the
// central difference comes back in the tens of thousands and the failure
// looks like a broken gradient rather than a broken test. All five of this
// file's first-draft gradchecks failed exactly that way.
$cw1 = bt.tensor(nd_random_uniform([3, 2], 1.0, 2.0), false);
gc(def($in) { return bt.sum(bt.mul(bt.take($in[0], [1, 1, 0], 0), $cw1), null, null); },
   [bt.tensor(nd_random_normal([2, 2]), true)], "take axis 0, weighted, index 1 twice");
gc(def($in) { return bt.sum(bt.take($in[0], [2, 0, 2, 1], 1), null, null); },
   [bt.tensor(nd_random_normal([2, 3]), true)], "take axis 1 with a repeat");
gc(def($in) { return bt.sum(bt.take($in[0], [-1, -1], 0), null, null); },
   [bt.tensor(nd_random_normal([3]), true)], "take with negative indices");
gc(def($in) { return bt.sum(bt.take($in[0], [1, 1, 0, 1], 1), null, null); },
   [bt.tensor(nd_random_normal([2, 2, 2]), true)], "take axis 1 of a 3-D tensor");

// embedding, which is take composed with a loss that actually uses the rows.
$cw2 = bt.tensor(nd_random_normal([5, 3]), false);
gc(def($in) { return bt.sum(bt.mul(bt.embedding($in[0], [0, 1, 0, 1, 0]), $cw2), null, null); },
   [bt.tensor(nd_random_normal([2, 3]), true)], "embedding with every row repeated");

// index_add, differentiating BOTH inputs at once.
$cw3 = bt.tensor(nd_random_normal([3]), false);
gc(def($in) { return bt.sum(bt.mul(bt.index_add($in[0], [0, 0, 1], $in[1], 0), $cw3),
                            null, null); },
   [bt.tensor(nd_random_normal([3]), true), bt.tensor(nd_random_normal([3]), true)],
   "index_add, both destination and values");

// max/min. Random inputs have no ties, so the split rule is not exercised here
// -- it is pinned by the hand-computed cases above instead. What gradcheck
// proves is that the routing goes to the right element at all.
gc(def($in) { return bt.sum(bt.max($in[0], 1, null), null, null); },
   [bt.tensor(nd_random_normal([3, 4]), true)], "max along axis 1");
gc(def($in) { return bt.sum(bt.min($in[0], 0, null), null, null); },
   [bt.tensor(nd_random_normal([3, 4]), true)], "min along axis 0");
gc(def($in) { return bt.max($in[0], null, null); },
   [bt.tensor(nd_random_normal([5]), true)], "a global max");
$cw4 = bt.tensor(nd_random_normal([3, 1]), false);
gc(def($in) { return bt.sum(bt.mul(bt.max($in[0], 1, true), $cw4), null, null); },
   [bt.tensor(nd_random_normal([3, 2]), true)], "max with keepdims, weighted");

// maximum/minimum against a constant and against each other.
gc(def($in) { return bt.sum(bt.maximum($in[0], $in[1]), null, null); },
   [bt.tensor(nd_random_normal([6]), true), bt.tensor(nd_random_normal([6]), true)],
   "maximum of two tensors");
gc(def($in) { return bt.sum(bt.minimum($in[0], 0.0), null, null); },
   [bt.tensor(nd_random_normal([6]), true)], "minimum against a constant");

// the shape ops, which must survive being composed with something non-linear
gc(def($in) { return bt.sum(bt.exp(bt.permute($in[0], [1, 0])), null, null); },
   [bt.tensor(nd_random_normal([2, 3]), true)], "permute under exp");
gc(def($in) { return bt.sum(bt.tanh(bt.unsqueeze($in[0], 1)), null, null); },
   [bt.tensor(nd_random_normal([4]), true)], "unsqueeze under tanh");
gc(def($in) { return bt.sum(bt.sigmoid(bt.squeeze($in[0], null)), null, null); },
   [bt.tensor(nd_random_normal([1, 4]), true)], "squeeze under sigmoid");
gc(def($in) { return bt.sum(bt.exp(bt.movedim($in[0], 0, 2)), null, null); },
   [bt.tensor(nd_random_normal([2, 2, 2]), true)], "movedim under exp");

// a composition that uses most of the phase at once
gc(def($in) {
       $e = bt.embedding($in[0], [1, 0, 1]);
       $m = bt.max($e, 1, null);
       return bt.sum(bt.tanh(bt.unsqueeze($m, 1)), null, null);
   },
   [bt.tensor(nd_random_normal([2, 3]), true)], "embedding -> max -> unsqueeze -> tanh");

print("");
print("========================================");
print("  nn.Embedding");
print("========================================");
bt.manual_seed(99);
$emb = nn.Embedding(10, 4, null);
eqList($emb.forward(bt.tensor([1.0, 3.0], false)).shape(), [2, 4], "a flat batch gives [N, dim]");
eqList($emb.forward(nd([[1, 2], [3, 4], [1, 1]], "i64")).shape(), [3, 2, 4],
       "a [B, L] batch gives [B, L, dim] (PyTorch's rule)");
ok(len($emb.parameters()) == 1, "Embedding has exactly one parameter");
eqList(nd_shape($emb.param("weight").data), [10, 4], "the table is [num_embeddings, dim]");

// The batched path must give the SAME numbers as the flat one, reshaped.
$flatOut = $emb.forward(nd([1, 2, 3, 4], "i64"));
$batOut = $emb.forward(nd([[1, 2], [3, 4]], "i64"));
ok(nd_array_equal(nd_reshape($flatOut.data, [2, 2, 4]), $batOut.data),
   "the batched gather is the flat gather reshaped, element for element");

// A repeated token in a batched lookup still accumulates.
$emb.zero_grad();
$o = $emb.forward(nd([[1, 1], [1, 2]], "i64"));
$o.sum(null, null).backward(null);
$gw = $emb.param("weight").grad;
eqList(nd_to_list(nd_take($gw, [1], 0)), [[3.0, 3.0, 3.0, 3.0]],
       "token 1 appears three times in a [2,2] batch and gets gradient 3");
eqList(nd_to_list(nd_take($gw, [0], 0)), [[0.0, 0.0, 0.0, 0.0]],
       "a token never mentioned gets no gradient");

// padding_idx
$pad = nn.Embedding(5, 3, 0);
eqList(nd_to_list(nd_take($pad.param("weight").data, [0], 0)), [[0.0, 0.0, 0.0]],
       "the padding row starts at exactly zero");
$po = $pad.forward(nd([0, 1, 0], "i64"));
eqList(nd_to_list(nd_take($po.data, [0, 2], 0)), [[0.0,0.0,0.0],[0.0,0.0,0.0]],
       "padding positions come out zero");
$po.sum(null, null).backward(null);
eqList(nd_to_list(nd_take($pad.param("weight").grad, [0], 0)), [[0.0, 0.0, 0.0]],
       "and the padding row receives exactly no gradient");
eqList(nd_to_list(nd_take($pad.param("weight").grad, [1], 0)), [[1.0, 1.0, 1.0]],
       "while a real token still does");
raises(def() { return nn.Embedding(5, 3, 9); }, "an out-of-range padding_idx raises");

// Embedding inside a Module, trained by an optimizer, is the end-to-end shape
// of every NLP model: the parameter must be found and updated.
$cw5 = bt.tensor(nd_random_normal([3, 2]), false);
gc(def($in) { return bt.sum(bt.mul(bt.embedding($in[0], [0, 2, 0]), $cw5), null, null); },
   [bt.tensor(nd_random_normal([3, 2]), true)], "the Embedding gradient, numerically");

print("");
print("========================================");
print("  nll_loss after the native one-hot");
print("========================================");
// The rewrite replaced a per-row interpreter loop with one native scatter, so
// the numbers must be bit-identical to the closed form.
$logits = bt.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], true);
$lp = nn.log_softmax($logits);
$manual = 0.0 - (nd_get($lp.data, [0, 0]) + nd_get($lp.data, [1, 1])) / 2.0;
close(nn.cross_entropy($logits, [0, 1], null).item(), $manual, 1e-14,
      "cross_entropy equals -(logp[0,target0] + logp[1,target1])/2 exactly");
$uni = bt.tensor([[0.0, 0.0, 0.0, 0.0]], false);
close(nn.cross_entropy($uni, [0], null).item(),
      nd_get(nd_log(nd_full([1], 4.0), null), [0]), 1e-14,
      "uniform logits over 4 classes still give exactly log(4)");
raises(def() { return nn.cross_entropy($logits, [0, 5], null); },
       "an out-of-range target still raises after the rewrite");
raises(def() { return nn.cross_entropy($logits, [0, 0 - 1], null); },
       "a negative target still raises");
raises(def() { return nn.cross_entropy($logits, [0], null); },
       "a target count that does not match the batch raises");
gc(def($in) { return nn.cross_entropy($in[0], [2, 0], null); },
   [bt.tensor(nd_random_normal([2, 3]), true)], "cross_entropy through the native one-hot");

print("");
print("========================================");
print("  TIER 4 — stress");
print("========================================");

// A realistic embedding step: 32k vocabulary, 64 dims, a 4096-token batch
// drawn with replacement so repeats are guaranteed. The gradient must be
// exactly the count of each token, which is a thing the test can compute
// independently with nd_bincount.
bt.manual_seed(5);
$V = 32000; $D = 64; $B = 4096;
$tbl = bt.tensor(nd_random_normal([$V, $D], 0.0, 0.01), true);
$tok = nd_astype(nd_random_int([$B], 0, $V), "i64");
$t0 = clock();
$out = bt.embedding($tbl, $tok);
$loss = $out.sum(null, null);
$loss.backward(null);
$ms = clock() - $t0;
print("        32k x 64 table, 4096 tokens, forward + backward: " + str($ms) + " ms");
// d(sum)/d(w[v, :]) is the number of times token v appears.
$counts = nd_astype(nd_bincount($tok, $V), "f64");
$rowsums = nd_sum($tbl.grad, 1, null);
ok(nd_allclose($rowsums, nd_multiply($counts, $D * 1.0, null), 1e-9, 1e-9),
   "every row's gradient sums to its token count x dim -- independently via nd_bincount");
ok(nd_get(nd_reshape(nd_sum($tbl.grad, null, null), [1]), [0]) == $B * 1.0 * $D,
   "and the whole gradient sums to batch x dim, so nothing was lost or doubled");
ok($ms < 4000, "a realistic embedding step stays under 4s");

// Repetition without leaks. nd_live_bytes is the allocator's own accounting.
$small = bt.tensor(nd_random_normal([50, 8]), true);
$before = nd_live_bytes();
$n = 0;
while ($n < 2000) {
    $small.grad = null;
    bt.sum(bt.embedding($small, [0, 1, 0, 2]), null, null).backward(null);
    $n = $n + 1;
}
$after = nd_live_bytes();
eqList(nd_to_list(nd_take($small.grad, [0], 0)),
       [[2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0]], "2000 forward+backward rounds still accumulate 2 for row 0");
print("        live array bytes: " + str($before) + " -> " + str($after));
ok($after - $before < 1000000, "2000 embedding rounds grew live array bytes by under 1MB");

// A deep chain of shape ops: every backward is a relabelling, so a 200-deep
// stack of them must still produce the original shape and exactly ones.
$deep = bt.tensor(nd_random_normal([2, 3, 4]), true);
$cur = $deep;
$n = 0;
while ($n < 200) {
    $cur = bt.permute($cur, [1, 2, 0]);
    $n = $n + 1;
}
$cur.sum(null, null).backward(null);
eqList(nd_shape($deep.grad), [2, 3, 4], "200 chained permutes give back the original shape");
ok(nd_allclose($deep.grad, nd_ones([2, 3, 4], "f64"), 1e-15, 1e-15),
   "and a gradient of exactly ones");

// One token, repeated 100,000 times: the accumulate must not drift.
$hot = bt.tensor(nd_zeros([2, 1], "f64"), true);
$manyids = nd_zeros([100000], "i64");
bt.sum(bt.embedding($hot, $manyids), null, null).backward(null);
ok(nd_get($hot.grad, [0, 0]) == 100000.0, "100,000 hits on one row accumulate to exactly 1e5");
ok(nd_get($hot.grad, [1, 0]) == 0.0, "and the other row stays at exactly zero");

// A full training step through an Embedding, so the optimizer path is covered.
include "../btorch/optim.b" as optim;
bt.manual_seed(1);
$mdl = nn.Embedding(6, 3, null);
$opt = optim.SGD($mdl.parameters(), 0.1, null, null);
$w0 = nd_copy($mdl.param("weight").data, null);
$step = 0;
while ($step < 50) {
    $opt.zero_grad();
    $pred = $mdl.forward(nd([0, 1, 0], "i64"));
    $l = nn.mse_loss($pred, bt.tensor(nd_ones([3, 3], "f64"), false), null);
    $l.backward(null);
    $opt.step();
    $step = $step + 1;
}
$moved = nd_sum(nd_abs(nd_subtract($mdl.param("weight").data, $w0, null), null), null, null);
ok(nd_get(nd_reshape($moved, [1]), [0]) > 0.0, "50 SGD steps moved the embedding table");
// Rows 2..5 were never looked up, so they must be untouched -- the strongest
// statement that the scatter wrote only where it was told to.
ok(nd_array_equal(nd_take($mdl.param("weight").data, [2, 3, 4, 5], 0),
                  nd_take($w0, [2, 3, 4, 5], 0)),
   "and left every row it never looked up bit-identical");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
