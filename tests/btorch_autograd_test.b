// ════════════════════════════════════════════════════════════════════════
//  btorch_autograd_test.b — the tape, backward(), and every gradient.
//
//  Tier 1 (feature), tier 2 (differential, against values worked out by
//  hand) and tier 3 (gradcheck, against central differences) for phase 1
//  and phase 2 of btorch.
//
//  The gradcheck half is the one that matters. A gradient derived by hand
//  and a gradient estimated numerically are wrong in DIFFERENT ways, so
//  agreement between them is evidence rather than two copies of the same
//  mistake (DECISIONS B9). Note also B10: a loss going down proves nothing,
//  so there is no "it trains" assertion anywhere in this file.
//
//  Run:  bantu run tests/btorch_autograd_test.b
// ════════════════════════════════════════════════════════════════════════

include "../btorch/btorch.b" as bt;

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
// Gradcheck a function and report the worst disagreement it found.
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

bt.manual_seed(12345);

print("== tensors ==");
$t = bt.tensor([[1.0, 2.0], [3.0, 4.0]], true);
eqList($t.shape(), [2, 2], "shape");
ok($t.ndim() == 2, "ndim");
ok($t.size() == 4, "size");
ok($t.dtype() == "f64", "tensors compute in f64");
ok($t.requires_grad, "requires_grad was set");
ok($t.is_leaf, "a user-created tensor is a leaf");
ok($t.grad == null, "grad starts null, not a zero array");
close(bt.tensor([[7.5]], false).item(), 7.5, 0.0, "item() on a one-element tensor");
raises(def() { return $t.item(); }, "item() on a 4-element tensor raises");
// an integer-valued input must not stay i64 -- a gradient would truncate to 0
ok(bt.arange(0, 5, false).dtype() == "f64", "arange is converted to f64");

print("");
print("== the graph is built only when it should be ==");
$a = bt.tensor([2.0], true);
$b = bt.tensor([3.0], false);
$c = $a.mul($b);
ok($c.requires_grad, "an op with one requires_grad input produces a grad tensor");
ok(!$c.is_leaf, "a produced tensor is not a leaf");
ok($c.ctx != null, "a node was attached");
ok($c.ctx.name == "mul", "the node knows its op");
$d = $b.mul($b);
ok(!$d.requires_grad, "an op with no requires_grad input attaches nothing");
ok($d.ctx == null, "and builds no node");
$e = bt.no_grad(def() { return $a.mul($b); });
ok(!$e.requires_grad, "no_grad: no graph is built");
ok($e.ctx == null, "no_grad: no node");
close($e.item(), 6.0, 1e-12, "no_grad still computes the right value");
ok(bt.is_grad_enabled(), "grad mode is restored after no_grad");
// and restored even when the block throws
try { bt.no_grad(def() { throw "boom"; }); } catch ($err) { $x = 1; }
ok(bt.is_grad_enabled(), "grad mode is restored when no_grad's body THROWS");

print("");
print("== backward on a simple chain:  y = (a*b) + a,  a=2 b=3 ==");
// dy/da = b + 1 = 4 ;  dy/db = a = 2
$a = bt.tensor([2.0], true);
$b = bt.tensor([3.0], true);
$y = $a.mul($b).add($a);
close($y.item(), 8.0, 1e-12, "forward value");
$y.backward(null);
close(nd_get($a.grad, [0]), 4.0, 1e-12, "dy/da = b + 1 = 4");
close(nd_get($b.grad, [0]), 2.0, 1e-12, "dy/db = a = 2");

print("");
print("== THE DIAMOND:  y = x*x + x  -- the smallest graph that needs a real toposort ==");
// x is used TWICE. A engine that propagates as soon as it sees a parent, rather
// than in reverse topological order, gets this wrong or double-counts it.
// dy/dx = 2x + 1 = 9 at x = 4
$x = bt.tensor([4.0], true);
$y = $x.mul($x).add($x);
close($y.item(), 20.0, 1e-12, "diamond forward: 4*4 + 4 = 20");
$y.backward(null);
close(nd_get($x.grad, [0]), 9.0, 1e-12, "diamond backward: dy/dx = 2x+1 = 9");
// a longer shared path
$x2 = bt.tensor([3.0], true);
$t1 = $x2.mul($x2);          // x^2
$t2 = $t1.mul($x2);          // x^3
$y2 = $t2.add($t1);          // x^3 + x^2 ; dy/dx = 3x^2 + 2x = 27+6 = 33
$y2.backward(null);
close(nd_get($x2.grad, [0]), 33.0, 1e-12, "x^3 + x^2 -> 3x^2 + 2x = 33");

print("");
print("== B3: ten nodes built in a LOOP, all ten gradients distinct ==");
// This is the regression for the defect that shaped the whole design. Bantu
// closures capture by REFERENCE, so a backward closure written inline in a
// loop would see whatever its saved variable ended up as -- every node would
// report the LAST node's gradient, silently. Every backward in btorch is
// therefore built by a factory whose parameters bind once per call.
// Here y_i = x_i * c_i with c_i = i+1, so dy_i/dx_i = i+1: all different.
$xs = [];
$ys = [];
$i = 0;
while ($i < 10) {
    $xi = bt.tensor([1.0], true);
    $xs = push($xs, $xi);
    $ys = push($ys, $xi.mul(bt.tensor([$i + 1.0], false)));
    $i = $i + 1;
}
$i = 0;
while ($i < 10) { $ys[$i].backward(null); $i = $i + 1; }
$allright = true;
$i = 0;
while ($i < 10) {
    $g = nd_get($xs[$i].grad, [0]);
    if ($g != $i + 1.0) { $allright = false; }
    $i = $i + 1;
}
ok($allright, "ten loop-built nodes each report their OWN gradient (1..10)");
close(nd_get($xs[0].grad, [0]), 1.0, 1e-12, "  node 0 -> 1");
close(nd_get($xs[9].grad, [0]), 10.0, 1e-12, "  node 9 -> 10 (not 1, not 10 everywhere)");

print("");
print("== gradient accumulation is +=, as in PyTorch ==");
$p = bt.tensor([5.0], true);
$q = $p.mul(bt.tensor([2.0], false));
$q.backward_retain(null);
close(nd_get($p.grad, [0]), 2.0, 1e-12, "first backward sets grad");
$q.backward_retain(null);
close(nd_get($p.grad, [0]), 4.0, 1e-12, "second backward ACCUMULATES (2+2)");
$p.zero_grad();
ok($p.grad == null, "zero_grad() clears it");

print("");
print("== broadcasting: the gradient is SUMMED over the broadcast axes ==");
// This is where hand-written engines break. A [3] bias against a [4,3]
// activation gets a [4,3] gradient, and its own must be [3] -- summed, not
// reshaped and not sliced.
$W = bt.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [1.0, 1.0, 1.0]], true);
$bias = bt.tensor([10.0, 20.0, 30.0], true);
$out = $W.add($bias).sum(null, null);
$out.backward(null);
eqList(nd_shape($bias.grad), [3], "the bias gradient has the BIAS's shape, not the activation's");
eqList(nd_to_list($bias.grad), [4, 4, 4], "and it is the SUM over the 4 broadcast rows");
eqList(nd_shape($W.grad), [4, 3], "the matrix gradient keeps its own shape");
// a stretched (size-1) axis, not just an added one
$col = bt.tensor([[1.0], [2.0]], true);
$m2 = bt.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], true);
$o2 = $col.mul($m2).sum(null, null);
$o2.backward(null);
eqList(nd_shape($col.grad), [2, 1], "a stretched axis keeps its size-1 shape");
eqList(nd_to_list($col.grad), [[3], [3]], "and sums the 3 stretched columns");

print("");
print("== matmul, including its batched form ==");
$A = bt.tensor([[1.0, 2.0], [3.0, 4.0]], true);
$B = bt.tensor([[5.0, 6.0], [7.0, 8.0]], true);
$M = $A.matmul($B);
eqList(nd_to_list($M.data), [[19, 22], [43, 50]], "matmul forward");
$M.sum(null, null).backward(null);
// d(sum(A@B))/dA = ones @ B^T ; row of B sums: [5+6, 7+8] = [11, 15]
eqList(nd_to_list($A.grad), [[11, 15], [11, 15]], "dA = G @ B^T");
eqList(nd_to_list($B.grad), [[4, 4], [6, 6]], "dB = A^T @ G");

print("");
print("== reductions ==");
$s = bt.tensor([[1.0, 2.0], [3.0, 4.0]], true);
close($s.sum(null, null).item(), 10.0, 1e-12, "sum all");
$s.sum(null, null).backward(null);
eqList(nd_to_list($s.grad), [[1, 1], [1, 1]], "d(sum)/dx is all ones");
$s2 = bt.tensor([[1.0, 2.0], [3.0, 4.0]], true);
close($s2.mean(null, null).item(), 2.5, 1e-12, "mean all");
$s2.mean(null, null).backward(null);
eqList(nd_to_list($s2.grad), [[0.25, 0.25], [0.25, 0.25]], "d(mean)/dx is 1/n");
// with an axis
$s3 = bt.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], true);
eqList(nd_to_list($s3.sum(0, null).data), [5, 7, 9], "sum over axis 0");
eqList(nd_to_list($s3.sum(1, null).data), [6, 15], "sum over axis 1");
$s3.sum(1, null).sum(null, null).backward(null);
eqList(nd_to_list($s3.grad), [[1, 1, 1], [1, 1, 1]], "axis sum backward fills ones");

print("");
print("== shape ops carry gradients through ==");
$r = bt.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], true);
eqList($r.reshape([3, 2]).shape(), [3, 2], "reshape forward");
$r.reshape([6]).mul(bt.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], false)).sum(null, null).backward(null);
eqList(nd_to_list($r.grad), [[1, 2, 3], [4, 5, 6]], "reshape backward restores the original shape");
$tr = bt.tensor([[1.0, 2.0], [3.0, 4.0]], true);
eqList(nd_to_list($tr.t().data), [[1, 3], [2, 4]], "transpose forward");
$tr.t().mul(bt.tensor([[1.0, 2.0], [3.0, 4.0]], false)).sum(null, null).backward(null);
eqList(nd_to_list($tr.grad), [[1, 3], [2, 4]], "transpose backward transposes back");

print("");
print("== operators, which exist because a class can overload them now ==");
$u = bt.tensor([3.0], true);
$v = bt.tensor([4.0], true);
close($u.add($v).item(), 7.0, 1e-12, "method .add");
close(($u + $v).item(), 7.0, 1e-12, "$u + $v");
close(($u - $v).item(), 0.0 - 1.0, 1e-12, "$u - $v");
close(($u * $v).item(), 12.0, 1e-12, "$u * $v");
close(($u / $v).item(), 0.75, 1e-12, "$u / $v");
close((0 - $u).item(), 0.0 - 3.0, 1e-12, "unary minus");
close(($u * 2).item(), 6.0, 1e-12, "$u * number");
close((2 * $u).item(), 6.0, 1e-12, "number * $u (reflected)");
close((10 - $u).item(), 7.0, 1e-12, "number - $u (reflected, NOT $u - 10)");
// and an operator expression differentiates identically to the method form
$o1 = bt.tensor([2.0], true);
$o2 = bt.tensor([5.0], true);
$z = $o1 * $o2 + $o1;
$z.backward(null);
close(nd_get($o1.grad, [0]), 6.0, 1e-12, "operators differentiate: dz/da = b+1 = 6");
close(nd_get($o2.grad, [0]), 2.0, 1e-12, "operators differentiate: dz/db = a = 2");

print("");
print("== detach cuts the graph ==");
$dd = bt.tensor([3.0], true);
$de = $dd.mul(bt.tensor([2.0], false)).detach();
ok(!$de.requires_grad, "detach() drops requires_grad");
ok($de.ctx == null, "detach() drops the history");
close($de.item(), 6.0, 1e-12, "detach() keeps the value");

print("");
print("== the errors a user will actually hit ==");
$ns = bt.tensor([[1.0, 2.0], [3.0, 4.0]], true);
raises(def() { return $ns.backward(null); },
       "backward() on a NON-scalar without a gradient raises (and says what to do)");
$noreq = bt.tensor([1.0], false);
raises(def() { return $noreq.backward(null); },
       "backward() on a tensor that does not require grad raises");
// a gradient of the wrong shape is refused rather than broadcast
raises(def() { return $ns.backward(bt.tensor([1.0, 2.0, 3.0], false)); },
       "backward() with a wrongly-shaped gradient raises");
// an explicit gradient of the right shape works
$ns2 = bt.tensor([[1.0, 2.0], [3.0, 4.0]], true);
$ns2.mul(bt.tensor([[2.0, 2.0], [2.0, 2.0]], false)).backward(bt.tensor([[1.0, 1.0], [1.0, 1.0]], false));
eqList(nd_to_list($ns2.grad), [[2, 2], [2, 2]], "an explicit seed gradient of the right shape works");
raises(def() { $nl = bt.tensor([1.0], true); return $nl.mul(2).requires_grad_(true); },
       "requires_grad_() on a non-leaf raises");

print("");
print("== the graph is FREED after backward unless retained ==");
$f1 = bt.tensor([2.0], true);
$f2 = $f1.mul($f1);
$f2.backward(null);
ok($f2.ctx == null, "backward() dropped the node (so the saved arrays are released)");
$g1 = bt.tensor([2.0], true);
$g2 = $g1.mul($g1);
$g2.backward_retain(null);
ok($g2.ctx != null, "backward_retain() kept it");

print("");
print("== a DEEP graph: 2000 nodes, which a recursive toposort could not do ==");
// The interpreter caps recursion at 1000 calls, so the toposort is iterative
// with an explicit stack. This is the regression for that: y = x doubled 2000
// times, dy/dx = 2^2000 overflows, so it is built as +1 instead.
$deep = bt.tensor([0.0], true);
$acc = $deep;
$i = 0;
while ($i < 2000) { $acc = $acc.add(bt.tensor([1.0], false)); $i = $i + 1; }
close($acc.item(), 2000.0, 1e-9, "a 2000-node chain computes");
$acc.backward(null);
close(nd_get($deep.grad, [0]), 1.0, 1e-12, "and a 2000-node chain BACKWARDS (iterative toposort)");

print("");
print("== unary maths: values ==");
close(bt.tensor([0.0], false).exp().item(), 1.0, 1e-12, "exp(0) = 1");
close(bt.tensor([1.0], false).log().item(), 0.0, 1e-12, "log(1) = 0");
close(bt.tensor([4.0], false).sqrt().item(), 2.0, 1e-12, "sqrt(4) = 2");
close(bt.tensor([0.0], false).tanh().item(), 0.0, 1e-12, "tanh(0) = 0");
close(bt.tensor([0.0], false).sigmoid().item(), 0.5, 1e-12, "sigmoid(0) = 0.5");
close(bt.tensor([0.0 - 3.0], false).relu().item(), 0.0, 1e-12, "relu(-3) = 0");
close(bt.tensor([3.0], false).relu().item(), 3.0, 1e-12, "relu(3) = 3");
close(bt.tensor([0.0 - 3.0], false).abs().item(), 3.0, 1e-12, "abs(-3) = 3");
// sigmoid is built on tanh so it does not overflow where 1/(1+exp(-x)) would
close(bt.tensor([0.0 - 800.0], false).sigmoid().item(), 0.0, 1e-12,
      "sigmoid(-800) is 0, not NaN (the exp form overflows here)");
close(bt.tensor([800.0], false).sigmoid().item(), 1.0, 1e-12, "sigmoid(800) is 1");

print("");
print("== relu's gradient is gated on the input sign ==");
$rl = bt.tensor([0.0 - 2.0, 0.0, 3.0], true);
$rl.relu().sum(null, null).backward(null);
eqList(nd_to_list($rl.grad), [0, 0, 1], "relu backward: 0 for x<0, 0 at x=0, 1 for x>0");

print("");
print("====================================================");
print("  TIER 3 — gradcheck: every gradient against central");
print("  differences. This is the gate that actually proves");
print("  the calculus, independently of the hand derivation.");
print("====================================================");
$ga = bt.tensor([1.5, 0.0 - 2.5, 3.25], true);
$gb = bt.tensor([0.75, 1.25, 0.0 - 0.5], true);

gc(def($i) { return $i[0].add($i[1]).sum(null, null); }, [$ga, $gb], "add");
gc(def($i) { return $i[0].sub($i[1]).sum(null, null); }, [$ga, $gb], "sub");
gc(def($i) { return $i[0].mul($i[1]).sum(null, null); }, [$ga, $gb], "mul");
gc(def($i) { return $i[0].div($i[1]).sum(null, null); }, [$ga, $gb], "div");
gc(def($i) { return $i[0].neg().sum(null, null); },      [$ga], "neg");
gc(def($i) { return $i[0].tanh().sum(null, null); },     [$ga], "tanh");
gc(def($i) { return $i[0].sigmoid().sum(null, null); },  [$ga], "sigmoid");
gc(def($i) { return $i[0].abs().sum(null, null); },      [$ga], "abs");
gc(def($i) { return $i[0].mean(null, null); },           [$ga], "mean");
gc(def($i) { return $i[0].sum(null, null); },            [$ga], "sum");

// exp / log / sqrt need a positive domain, so they get their own input
$gp = bt.tensor([0.5, 1.5, 2.25], true);
gc(def($i) { return $i[0].exp().sum(null, null); },  [$gp], "exp");
gc(def($i) { return $i[0].log().sum(null, null); },  [$gp], "log");
gc(def($i) { return $i[0].sqrt().sum(null, null); }, [$gp], "sqrt");
gc(def($i) { return $i[0].pow(bt.tensor([3.0], false)).sum(null, null); }, [$gp], "pow (constant exponent)");
// and a differentiable exponent, which needs ln(a) -- hence a positive base
$gexp = bt.tensor([2.0], true);
gc(def($i) { return $i[0].pow($i[1]).sum(null, null); }, [$gp, $gexp], "pow (both differentiated)");

// relu is kinked at 0, so gradcheck is only valid away from it -- a numerical
// derivative straddling the kink is genuinely wrong, not a bug in the backward
$grelu = bt.tensor([0.0 - 2.0, 1.0, 3.0], true);
gc(def($i) { return $i[0].relu().sum(null, null); }, [$grelu], "relu (away from the kink)");
gc(def($i) { return bt.leaky_relu($i[0], 0.1).sum(null, null); }, [$grelu], "leaky_relu");

print("");
print("-- composites and shape ops --");
$gm1 = bt.tensor([[1.0, 2.0], [0.0 - 1.0, 0.5]], true);
$gm2 = bt.tensor([[0.5, 1.0], [2.0, 0.0 - 1.5]], true);
gc(def($i) { return $i[0].matmul($i[1]).sum(null, null); }, [$gm1, $gm2], "matmul");
gc(def($i) { return $i[0].t().mul($i[1]).sum(null, null); }, [$gm1, $gm2], "transpose into mul");
gc(def($i) { return $i[0].reshape([4]).mul(bt.tensor([1.0,2.0,3.0,4.0], false)).sum(null, null); },
   [$gm1], "reshape");
gc(def($i) { return $i[0].sum(0, null).sum(null, null); }, [$gm1], "sum over axis 0");
gc(def($i) { return $i[0].sum(1, null).sum(null, null); }, [$gm1], "sum over axis 1");
gc(def($i) { return $i[0].mean(0, null).sum(null, null); }, [$gm1], "mean over axis 0");

print("");
print("-- broadcasting, where the _unbroadcast rule is exercised --");
$gbig = bt.tensor([[1.0, 2.0, 3.0], [0.0 - 1.0, 0.5, 2.0]], true);
$gbias = bt.tensor([0.25, 0.0 - 0.5, 1.0], true);
gc(def($i) { return $i[0].add($i[1]).sum(null, null); }, [$gbig, $gbias], "broadcast add (bias)");
gc(def($i) { return $i[0].mul($i[1]).sum(null, null); }, [$gbig, $gbias], "broadcast mul");
gc(def($i) { return $i[0].add($i[1]).tanh().sum(null, null); }, [$gbig, $gbias],
   "broadcast add into tanh");
$gcol = bt.tensor([[2.0], [0.0 - 1.0]], true);
gc(def($i) { return $i[0].mul($i[1]).sum(null, null); }, [$gbig, $gcol], "broadcast over a size-1 axis");

print("");
print("-- the diamond, and a deeper shared graph --");
$gd = bt.tensor([1.5, 0.0 - 0.5], true);
gc(def($i) { return $i[0].mul($i[0]).add($i[0]).sum(null, null); }, [$gd], "diamond x*x + x");
gc(def($i) { return $i[0].mul($i[0]).mul($i[0]).sum(null, null); }, [$gd], "x*x*x");
gc(def($i) { return $i[0].exp().mul($i[0]).sum(null, null); }, [$gd], "exp(x)*x");
gc(def($i) { return $i[0].tanh().mul($i[0].sigmoid()).sum(null, null); }, [$gd],
   "tanh(x)*sigmoid(x) -- two paths from one leaf");

print("");
print("-- a small MLP's worth of composition, end to end --");
$mw1 = bt.tensor([[0.5, 0.0 - 0.25], [1.0, 0.75]], true);
$mb1 = bt.tensor([0.1, 0.0 - 0.2], true);
$mw2 = bt.tensor([[1.5], [0.0 - 0.5]], true);
$mx = bt.tensor([[1.0, 2.0], [0.0 - 1.0, 0.5]], false);
gc(def($i) {
       // x @ W1 + b1 -> tanh -> @ W2 -> sum   (tanh, not relu: no kink)
       return $mx.matmul($i[0]).add($i[1]).tanh().matmul($i[2]).sum(null, null);
   }, [$mw1, $mb1, $mw2], "two-layer network through W1, b1, W2");

print("");
print("== the checker itself: it must FAIL on a wrong gradient ==");
// A gradcheck that can never fail checks nothing. _wrongBackward returns 2*g
// where the truth is g, so this must be detected.
def _wrongBackward() { return def($gr) { return [nd_multiply($gr, 2.0, null)]; }; }
def wrongOp($a) { return bt._mk(nd_copy($a.data, null), [$a], "deliberately_wrong", _wrongBackward()); }
$wx = bt.tensor([1.0, 2.0], true);
$wr = bt.gradcheck(def($i) { return wrongOp($i[0]).sum(null, null); }, [$wx], null, null);
ok(!$wr.ok, "gradcheck REJECTS a deliberately doubled gradient");
// The relative error is |analytic - numerical| / max(1, |analytic|, |numerical|)
// = |2 - 1| / 2 = 0.5, which is five million times the 1e-7 tolerance.
close($wr.worst, 0.5, 0.01, "and quantifies it: 0.5 relative (analytic 2 vs numerical 1)");
ok($wr.worst > $wr.tol * 1000, "the disagreement is orders of magnitude over tolerance");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
