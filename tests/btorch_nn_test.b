// ════════════════════════════════════════════════════════════════════════
//  btorch_nn_test.b — the module system, layers, losses and optimizers.
//
//  Phases 3 and 4. The assertions that matter here are the ones a human
//  reviewer cannot make by eye:
//
//    * Linear's initializer is checked against the ANALYTIC variance of a
//      Kaiming-uniform bound. A wrong initializer does not fail -- it trains
//      slowly and looks like a bad learning rate.
//    * parameters() is checked for DEDUPLICATION. A weight shared between
//      two layers returned twice is stepped twice: a silent 2x lr.
//    * Adam is checked against its CLOSED FORM at t=1,2,3, because the bias
//      correction is the part everyone gets wrong and a broken Adam still
//      reduces a loss (DECISIONS B10).
//    * The stable softmax family is checked with logits of 1000, which turn
//      a naive implementation into NaN.
//
//  Run:  bantu run tests/btorch_nn_test.b
// ════════════════════════════════════════════════════════════════════════

include "../btorch/btorch.b" as bt;
include "../btorch/nn.b" as nn;
include "../btorch/optim.b" as optim;

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
def eq($got, $want, $name) {
    if (str($got) == str($want)) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name);
        print("          got:  " + str($got));
        print("          want: " + str($want));
    }
}
def raises($fn, $name) {
    try { $fn(); $R.fail = $R.fail + 1; print("  FAIL  " + $name + " (no error)"); }
    catch ($e) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    return null;
}
// A full nd_* reduction returns a 0-d ARRAY, not a number (as in NumPy), so a
// bare `sc(nd_min(x, null, null)) == 0` compares a handle against a number and is
// always false. sc() pulls the single value out.
def sc($x) {
    if (type($x) == "ndarray") { return nd_get(nd_reshape($x, [1]), [0]); }
    return $x;
}
def gc($f, $inputs, $name) {
    $r = bt.gradcheck($f, $inputs, null, null);
    if ($r.ok) { $R.pass = $R.pass + 1; print("  ok    gradcheck " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  gradcheck " + $name + "  worst " + str($r.worst) + " at " + str($r.where));
    }
    return null;
}

bt.manual_seed(999);

print("== Linear ==");
$fc = nn.Linear(4, 3, true);
eq($fc.param("weight").shape(), [4, 3], "weight is [in, out]");
eq($fc.param("bias").shape(), [3], "bias is [out]");
ok($fc.param("weight").requires_grad, "weight requires grad");
ok($fc.param("bias").requires_grad, "bias requires grad");
$x = bt.randn([5, 4], false);
eq($fc.forward($x).shape(), [5, 3], "forward maps [5,4] -> [5,3]");
$nb = nn.Linear(4, 3, false);
eq(len($nb.parameters()), 1, "bias=false registers only the weight");
eq($nb.forward($x).shape(), [5, 3], "and still forwards");

print("");
print("== Linear's initializer, against the ANALYTIC variance ==");
// Kaiming uniform with PyTorch's a=sqrt(5) default reduces to U(-b, b) with
// b = sqrt(1/fan_in). For a uniform distribution Var = b^2/3 = 1/(3*fan_in).
// A large sample makes this checkable; a wrong initializer trains slowly
// rather than failing, so only a test like this catches it.
$big = nn.Linear(100, 400, false);
$w = $big.param("weight").data;
$var = sc(nd_var($w, null, null));
$expected = 1.0 / (3.0 * 100.0);
close($var, $expected, $expected * 0.08, "weight variance is 1/(3*fan_in) within 8%");
$mean = sc(nd_mean($w, null, null));
close($mean, 0.0, 0.002, "weight mean is ~0 (the bound is symmetric)");
$bound = nd_get(nd_sqrt(nd_full([1], 1.0 / 100.0), null), [0]);
ok(sc(nd_max(nd_abs($w, null), null, null)) <= $bound, "no weight exceeds the Kaiming bound");
ok(sc(nd_max(nd_abs($w, null), null, null)) > $bound * 0.9, "and the bound is actually approached");

print("");
print("== the parameter tree ==");
class TwoLayer extends Module {
    def init() {
        super();
        this.add("fc1", nn.Linear(4, 8, true));
        this.add("fc2", nn.Linear(8, 2, true));
    }
    def forward($x) { return this.get("fc2").forward(bt.relu(this.get("fc1").forward($x))); }
}
$m = new TwoLayer();
eq(len($m.parameters()), 4, "two Linear layers with bias -> 4 parameters");
eq($m.num_parameters(), 4 * 8 + 8 + 8 * 2 + 2, "num_parameters counts every element");
$names = [];
$np = $m.named_parameters();
$i = 0;
while ($i < len($np)) { push($names, $np[$i][0]); $i = $i + 1; }
ok(contains(str($names), "fc1.weight"), "named_parameters uses dotted paths");
ok(contains(str($names), "fc2.bias"), "and reaches the second layer");
eq($m.forward(bt.randn([3, 4], false)).shape(), [3, 2], "the model forwards end to end");
raises(def() { return $m.get("nope"); }, "get() on a missing child raises by name");

print("");
print("== parameters() DEDUPLICATES by identity (a tied weight) ==");
// A weight shared between two layers must be returned ONCE. Returned twice,
// the optimizer steps it twice -- a silent 2x learning rate on that tensor.
class Tied extends Module {
    def init() {
        super();
        $shared = nn.Linear(4, 4, false);
        this.add("a", $shared);
        this.add("b", $shared);          // the SAME module, twice
        this.add("c", nn.Linear(4, 4, false));
    }
    def forward($x) { return this.get("c").forward(this.get("a").forward($x)); }
}
$tied = new Tied();
eq(len($tied.parameters()), 2, "a module registered twice contributes its weight ONCE");
// and the same tensor object registered directly under two names
class Tied2 extends Module {
    def init() {
        super();
        $w = bt.randn([2, 2], true);
        this.register("w1", $w);
        this.register("w2", $w);
        this.register("w3", bt.randn([2, 2], true));
    }
}
$t2 = new Tied2();
eq(len($t2.parameters()), 2, "the same tensor under two names counts once");

print("");
print("== zero_grad, train/eval recurse ==");
$m2 = new TwoLayer();
$m2.forward(bt.randn([3, 4], false)).sum(null, null).backward(null);
$ps = $m2.parameters();
ok($ps[0].grad != null, "a backward filled the parameter gradients");
$m2.zero_grad();
ok($ps[0].grad == null, "zero_grad cleared them");
ok($m2.training, "a module starts in training mode");
$m2.eval();
ok(!$m2.training, "eval() sets the parent");
ok(!$m2.get("fc1").training, "eval() recurses into children");
$m2.train();
ok($m2.get("fc2").training, "train() recurses too");

print("");
print("== Sequential ==");
$seq = nn.Sequential([nn.Linear(4, 6, true), nn.ReLU(), nn.Linear(6, 2, true)]);
eq($seq.forward(bt.randn([5, 4], false)).shape(), [5, 2], "Sequential forwards in order");
eq(len($seq.parameters()), 4, "Sequential collects its children's parameters");
// the ReLU really is applied: every output of a ReLU-terminated stack is >= 0
$relu_last = nn.Sequential([nn.Linear(4, 6, true), nn.ReLU()]);
ok(sc(nd_min($relu_last.forward(bt.randn([20, 4], false)).data, null, null)) >= 0,
   "a ReLU-terminated Sequential never outputs a negative");

print("");
print("== Dropout: scaled at train, EXACT identity at eval ==");
$dp = nn.Dropout(0.5);
$ones = bt.ones([200, 20], false);
$dp.train();
$dropped = $dp.forward($ones);
ok(sc(nd_min($dropped.data, null, null)) == 0, "at train time some activations are zeroed");
ok(sc(nd_max($dropped.data, null, null)) == 2.0, "and the survivors are scaled by 1/(1-p)");
// inverted dropout: the EXPECTED value is preserved, so eval needs no rescaling
close(sc(nd_mean($dropped.data, null, null)), 1.0, 0.05, "the mean is preserved (inverted dropout)");
$dp.eval();
$kept = $dp.forward($ones);
ok(nd_array_equal($kept.data, $ones.data), "at eval time dropout is the EXACT identity");
$dp0 = nn.Dropout(0.0);
$dp0.train();
ok(nd_array_equal($dp0.forward($ones).data, $ones.data), "p=0 is the identity even in training");
raises(def() { return nn.Dropout(1.0); }, "p=1 is refused (it would zero everything)");

print("");
print("== LayerNorm ==");
$ln = nn.LayerNorm(4, null);
$lnx = bt.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], false);
$lno = $ln.forward($lnx);
eq($lno.shape(), [2, 4], "LayerNorm preserves shape");
// each row is normalized independently: mean ~0, variance ~1
$row0 = nd_slice($lno.data, [[0, 1, 1], [0, 4, 1]]);
close(sc(nd_mean($row0, null, null)), 0.0, 1e-6, "row 0 has mean 0");
close(sc(nd_var($row0, null, null)), 1.0, 0.001, "row 0 has variance 1");
$row1 = nd_slice($lno.data, [[1, 2, 1], [0, 4, 1]]);
close(sc(nd_mean($row1, null, null)), 0.0, 1e-6, "row 1 has mean 0 (normalized independently)");
close(sc(nd_var($row1, null, null)), 1.0, 0.001, "row 1 has variance 1");

print("");
print("== losses: values worked out by hand ==");
$pred = bt.tensor([1.0, 2.0, 3.0], false);
$targ = bt.tensor([1.0, 4.0, 3.0], false);
// squared errors: 0, 4, 0 -> mean 4/3, sum 4
close(nn.mse_loss($pred, $targ, null).item(), 4.0 / 3.0, 1e-12, "mse mean");
close(nn.mse_loss($pred, $targ, "sum").item(), 4.0, 1e-12, "mse sum");
eq(nn.mse_loss($pred, $targ, "none").shape(), [3], "mse reduction='none' keeps the shape");
// absolute errors: 0, 2, 0 -> mean 2/3
close(nn.l1_loss($pred, $targ, null).item(), 2.0 / 3.0, 1e-12, "l1 mean");
raises(def() { return nn.mse_loss($pred, $targ, "banana"); }, "an unknown reduction raises");
// Huber with delta=1: |d|=2 is outside, so 1*(2 - 0.5) = 1.5; errors 0,1.5,0
close(nn.huber_loss($pred, $targ, 1.0, "sum").item(), 1.5, 1e-12, "huber sum, linear region");
// inside the quadratic region it is 0.5*d^2
$p2 = bt.tensor([0.0], false); $t2b = bt.tensor([0.5], false);
close(nn.huber_loss($p2, $t2b, 1.0, "sum").item(), 0.125, 1e-12, "huber quadratic region = 0.5*d^2");

print("");
print("== log_softmax / softmax are numerically STABLE ==");
$lg = bt.tensor([[1.0, 2.0, 3.0]], false);
$sm = nn.softmax($lg);
close(sc(nd_sum($sm.data, null, null)), 1.0, 1e-12, "softmax rows sum to 1");
// log_softmax of [1,2,3]: lse = log(e^-2 + e^-1 + 1) shifted by max=3
close(nd_get($sm.data, [0, 2]), 0.6652409557748219, 1e-12, "softmax matches the analytic value");
// THE stability case: logits of 1000 overflow exp() to +inf, and a naive
// softmax returns NaN. The max shift makes this exact.
$huge = bt.tensor([[1000.0, 1001.0]], false);
$hs = nn.softmax($huge);
$v = nd_get($hs.data, [0, 1]);
ok($v == $v, "softmax of [1000, 1001] is NOT NaN");
close($v, 0.7310585786300049, 1e-12, "and it is exactly sigmoid(1), the right answer");
$hl = nn.log_softmax($huge);
$lv = nd_get($hl.data, [0, 0]);
ok($lv == $lv, "log_softmax of [1000, 1001] is NOT NaN");
close($lv, 0.0 - 1.3132616875182228, 1e-12, "and it is the right value");

print("");
print("== cross entropy ==");
// A confident, CORRECT prediction has near-zero loss.
$logits = bt.tensor([[0.0, 100.0], [100.0, 0.0]], false);
close(nn.cross_entropy($logits, [1, 0], null).item(), 0.0, 1e-12,
      "a confident correct prediction has ~0 loss");
// A confident WRONG prediction has a large FINITE loss, not inf.
$lw = nn.cross_entropy($logits, [0, 1], null).item();
ok($lw > 50, "a confident wrong prediction has a large loss");
ok($lw < 1e30, "and it is FINITE, not inf (the fused form never materializes a 0 probability)");
// uniform logits over k classes -> loss = log(k)
$uni = bt.tensor([[0.0, 0.0, 0.0, 0.0]], false);
close(nn.cross_entropy($uni, [0], null).item(), nd_get(nd_log(nd_full([1], 4.0), null), [0]), 1e-12,
      "uniform logits over 4 classes give exactly log(4)");
raises(def() { return nn.cross_entropy($logits, [0, 5], null); },
       "an out-of-range target class raises by name");
raises(def() { return nn.cross_entropy($logits, [0, 0 - 1], null); },
       "a negative target class raises");

print("");
print("== bce_with_logits is stable too ==");
$bl = nn.bce_with_logits(bt.tensor([0.0], false), bt.tensor([1.0], false), null).item();
close($bl, nd_get(nd_log(nd_full([1], 2.0), null), [0]), 1e-12, "bce_with_logits(0, 1) = log 2");
$bh = nn.bce_with_logits(bt.tensor([0.0 - 500.0], false), bt.tensor([1.0], false), null).item();
ok($bh == $bh, "bce_with_logits(-500, 1) is not NaN");
close($bh, 500.0, 1e-9, "and equals 500, the right answer");

print("");
print("== gradcheck the layers and losses ==");
$gw = bt.tensor([[0.5, 0.0 - 0.25, 1.0], [0.75, 0.5, 0.0 - 1.0]], true);
$gb = bt.tensor([0.1, 0.0 - 0.2, 0.3], true);
$gx = bt.tensor([[1.0, 2.0], [0.0 - 1.0, 0.5]], false);
$gt = bt.tensor([[0.3, 0.1, 0.0 - 0.4], [0.2, 0.0 - 0.1, 0.5]], false);
gc(def($i) { return nn.mse_loss(bt.add(bt.matmul($gx, $i[0]), $i[1]), $gt, null); },
   [$gw, $gb], "mse over a linear layer");
gc(def($i) { return nn.l1_loss(bt.add(bt.matmul($gx, $i[0]), $i[1]), $gt, null); },
   [$gw, $gb], "l1 over a linear layer");
gc(def($i) { return nn.huber_loss(bt.add(bt.matmul($gx, $i[0]), $i[1]), $gt, 1.0, null); },
   [$gw, $gb], "huber over a linear layer");
gc(def($i) { return nn.cross_entropy(bt.add(bt.matmul($gx, $i[0]), $i[1]), [2, 0], null); },
   [$gw, $gb], "cross_entropy over a linear layer");
gc(def($i) { return bt.sum(nn.log_softmax(bt.matmul($gx, $i[0])), null, null); },
   [$gw], "log_softmax");
gc(def($i) { return bt.sum(nn.softmax(bt.matmul($gx, $i[0])), null, null); },
   [$gw], "softmax");
gc(def($i) { return nn.bce_with_logits(bt.matmul($gx, $i[0]), $gt, null); },
   [$gw], "bce_with_logits");
$gln = nn.LayerNorm(3, null);
gc(def($i) { return bt.sum($gln.forward(bt.add(bt.matmul($gx, $i[0]), $i[1])), null, null); },
   [$gw, $gb], "LayerNorm");
// a whole model's parameters at once
$gmodel = nn.Sequential([nn.Linear(2, 4, true), nn.Tanh(), nn.Linear(4, 3, true)]);
gc(def($i) { return nn.cross_entropy($gmodel.forward($gx), [1, 2], null); },
   $gmodel.parameters(), "cross_entropy through a 2-layer Sequential (all 4 parameters)");

print("");
print("== state_dict round-trip ==");
$sm1 = nn.Sequential([nn.Linear(3, 4, true), nn.ReLU(), nn.Linear(4, 2, true)]);
$sm2 = nn.Sequential([nn.Linear(3, 4, true), nn.ReLU(), nn.Linear(4, 2, true)]);
$inp = bt.randn([6, 3], false);
ok(!nd_allclose($sm1.forward($inp).data, $sm2.forward($inp).data, 1e-9, 1e-9),
   "two freshly initialized models differ (the seed is doing something)");
$sd = $sm1.state_dict();
eq(len($sd.keys()), 4, "state_dict has one entry per parameter");
ok(contains(str($sd.keys()), "0.weight"), "and PyTorch-style dotted keys");
$sm2.load_state_dict($sd);
ok(nd_allclose($sm1.forward($inp).data, $sm2.forward($inp).data, 1e-12, 1e-12),
   "after load_state_dict the two models agree EXACTLY");
raises(def() { return $sm2.load_state_dict({}); }, "load_state_dict with a missing key raises by name");
$badsd = $sm1.state_dict();
$badsd["0.weight"] = [[1.0, 2.0]];
raises(def() { return $sm2.load_state_dict($badsd); }, "load_state_dict with a wrong shape raises by name");

print("");
print("== SGD, against the closed form ==");
$p = bt.tensor([1.0, 2.0], true);
$p.grad = nd([0.5, 0.0 - 1.0], "f64");
$sgd = optim.SGD([$p], 0.1, null, null, null);
$sgd.step();
// p -= lr*g  ->  1 - 0.05 = 0.95 ;  2 + 0.1 = 2.1
eq(nd_to_list($p.data), [0.95, 2.1], "plain SGD: p -= lr*g");
// momentum: buf = m*buf + g, so the second step is larger than the first
$pm = bt.tensor([0.0], true);
$sgdm = optim.SGD([$pm], 0.1, 0.9, null, null);
$pm.grad = nd([1.0], "f64"); $sgdm.step();
close(nd_get($pm.data, [0]), 0.0 - 0.1, 1e-12, "momentum step 1: buf = g, so -lr*g");
$pm.grad = nd([1.0], "f64"); $sgdm.step();
// buf = 0.9*1 + 1 = 1.9 -> p -= 0.1*1.9
close(nd_get($pm.data, [0]), 0.0 - 0.29, 1e-12, "momentum step 2: buf = 0.9*1+1 = 1.9");
// weight decay adds wd*p to the gradient
$pw = bt.tensor([2.0], true);
$sgdw = optim.SGD([$pw], 0.1, null, 0.5, null);
$pw.grad = nd([0.0], "f64"); $sgdw.step();
close(nd_get($pw.data, [0]), 2.0 - 0.1 * 0.5 * 2.0, 1e-12, "weight decay: g += wd*p");
raises(def() { return optim.SGD([$p], 0.0 - 1.0, null, null, null); }, "a negative lr is refused");
raises(def() { return optim.SGD([], 0.1, null, null, null); }, "an empty parameter list is refused");
raises(def() { return optim.SGD([$p], null, null, null, null); }, "a missing lr is refused");
raises(def() { return optim.SGD([$p], 0.1, 0.0, null, true); }, "Nesterov without momentum is refused");

print("");
print("== Adam's BIAS CORRECTION, asserted at t = 1, 2, 3 ==");
// This is the part everyone gets wrong, and a broken Adam still trains.
// At t=1: m = (1-b1)g, mhat = m/(1-b1) = g ; v = (1-b2)g^2, vhat = g^2.
// So the update is lr*g/(|g| + eps) = lr for a positive g, INDEPENDENT of g.
$pa = bt.tensor([0.0], true);
$adam = optim.Adam([$pa], 0.1, null, null, null);
$pa.grad = nd([5.0], "f64");            // a big gradient
$adam.step();
close(nd_get($pa.data, [0]), 0.0 - 0.1, 1e-7,
      "t=1: the step is exactly -lr regardless of gradient size (bias correction)");
// and with a tiny gradient the first step is the SAME size -- the signature
// of correct bias correction
$pa2 = bt.tensor([0.0], true);
$adam2 = optim.Adam([$pa2], 0.1, null, null, null);
$pa2.grad = nd([0.001], "f64");
$adam2.step();
close(nd_get($pa2.data, [0]), 0.0 - 0.1, 1e-4,
      "t=1 with a 5000x smaller gradient: still -lr");
// t=2 and t=3 with a constant gradient stay at -lr each
$pa.grad = nd([5.0], "f64"); $adam.step();
close(nd_get($pa.data, [0]), 0.0 - 0.2, 1e-6, "t=2: a constant gradient gives another -lr");
$pa.grad = nd([5.0], "f64"); $adam.step();
close(nd_get($pa.data, [0]), 0.0 - 0.3, 1e-6, "t=3: and another");
ok($adam.t == 3, "the step counter advanced to 3");

print("");
print("== AdamW's decoupled decay really is different from Adam's ==");
$pd1 = bt.tensor([1.0], true);
$pd2 = bt.tensor([1.0], true);
$a_coupled = optim.Adam([$pd1], 0.1, null, null, null);
$a_coupled.weight_decay = 0.1;              // coupled: folded into g
$a_decoupled = optim.AdamW([$pd2], 0.1, 0.1, null, null);
$pd1.grad = nd([1.0], "f64"); $a_coupled.step();
$pd2.grad = nd([1.0], "f64"); $a_decoupled.step();
ok(nd_get($pd1.data, [0]) != nd_get($pd2.data, [0]),
   "coupled and decoupled weight decay give DIFFERENT updates (they are different algorithms)");

print("");
print("== RMSprop and Adagrad ==");
$pr = bt.tensor([0.0], true);
$rms = optim.RMSprop([$pr], 0.1, 0.99, null);
$pr.grad = nd([1.0], "f64"); $rms.step();
// v = 0.99*0 + 0.01*1 = 0.01 ; upd = 1/(0.1+eps) ~ 10 ; p -= 0.1*10 = 1
close(nd_get($pr.data, [0]), 0.0 - 1.0, 1e-5, "RMSprop step 1 matches its closed form");
$pg = bt.tensor([0.0], true);
$ada = optim.Adagrad([$pg], 0.1, null);
$pg.grad = nd([2.0], "f64"); $ada.step();
// s = 4 ; upd = 2/(2+eps) ~ 1 ; p -= 0.1
close(nd_get($pg.data, [0]), 0.0 - 0.1, 1e-5, "Adagrad step 1 matches its closed form");

print("");
print("== the optimizer updates WITHOUT recording a graph ==");
$pn = bt.tensor([1.0], true);
$pn.grad = nd([1.0], "f64");
$o = optim.SGD([$pn], 0.1, null, null, null);
$o.step();
ok($pn.ctx == null, "a step leaves no tape node on the parameter");
ok($pn.is_leaf, "the parameter is still a leaf");
ok(bt.is_grad_enabled(), "and grad mode is back on afterwards");
$o.zero_grad();
ok($pn.grad == null, "the optimizer's zero_grad clears gradients");

print("");
print("== gradient clipping, over ALL parameters jointly ==");
$c1 = bt.tensor([3.0], true); $c1.grad = nd([3.0], "f64");
$c2 = bt.tensor([4.0], true); $c2.grad = nd([4.0], "f64");
// joint norm = sqrt(9+16) = 5
$n = nn.clip_grad_norm([$c1, $c2], 1.0);
close($n, 5.0, 1e-12, "the reported norm is the JOINT norm, sqrt(3^2+4^2) = 5");
close(nd_get($c1.grad, [0]), 3.0 / 5.0, 1e-12, "each gradient is scaled by max_norm/norm");
close(nd_get($c2.grad, [0]), 4.0 / 5.0, 1e-12, "  and the other one too");
// under the threshold, nothing changes
$c3 = bt.tensor([1.0], true); $c3.grad = nd([0.1], "f64");
nn.clip_grad_norm([$c3], 10.0);
close(nd_get($c3.grad, [0]), 0.1, 1e-12, "a gradient under the threshold is untouched");

print("");
print("== learning-rate schedules ==");
$sp = bt.tensor([1.0], true);
$so = optim.SGD([$sp], 1.0, null, null, null);
$sl = optim.StepLR($so, 2, 0.1);
$sl.step(); close($so.lr, 1.0, 1e-12, "StepLR: unchanged before step_size");
$sl.step(); close($so.lr, 0.1, 1e-12, "StepLR: x gamma at epoch 2");
$sl.step(); close($so.lr, 0.1, 1e-12, "StepLR: still at epoch 3");
$sl.step(); close($so.lr, 0.01, 1e-12, "StepLR: x gamma^2 at epoch 4");
$eo = optim.SGD([$sp], 1.0, null, null, null);
$el = optim.ExponentialLR($eo, 0.5);
$el.step(); close($eo.lr, 0.5, 1e-12, "ExponentialLR epoch 1");
$el.step(); close($eo.lr, 0.25, 1e-12, "ExponentialLR epoch 2");
$co = optim.SGD([$sp], 1.0, null, null, null);
$cl = optim.CosineAnnealingLR($co, 10, 0.0);
$cl.step();
ok($co.lr < 1.0 && $co.lr > 0.9, "CosineAnnealingLR starts by decaying slowly");
$i = 0; while ($i < 9) { $cl.step(); $i = $i + 1; }
close($co.lr, 0.0, 1e-12, "CosineAnnealingLR reaches eta_min at T_max");

print("");
print("== a module hierarchy THREE deep (what segfaulted before phase A) ==");
// Module -> TwoLayer -> Deeper is three levels of super(), which crashed the
// interpreter with no diagnostic until A3 was fixed. nn.Module hierarchies
// reach this immediately, which is how it was found.
class Deeper extends TwoLayer {
    def init() { super(); this.add("fc3", nn.Linear(2, 1, true)); }
    def forward($x) { return this.get("fc3").forward(super_forward_hack($x, this)); }
}
def super_forward_hack($x, $self) {
    return $self.get("fc2").forward(bt.relu($self.get("fc1").forward($x)));
}
$deep = new Deeper();
eq(len($deep.parameters()), 6, "a 3-level module hierarchy collects all 6 parameters");
eq($deep.forward(bt.randn([4, 4], false)).shape(), [4, 1], "and forwards through all three layers");
$deep.eval();
ok(!$deep.get("fc3").training, "eval() reaches a child added at the third level");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
