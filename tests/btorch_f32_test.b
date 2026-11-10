// ════════════════════════════════════════════════════════════════════════
//  btorch_f32_test.b — training in f32 (ecosystem stage 1, item D2c).
//
//  f32 was useless to btorch until nd_matmul preserved it: a Linear layer
//  turned an f32 model back into f64 at the first layer (numba DECISIONS N31).
//  With f32 GEMM in place -- f32 in, f64 ACCUMULATE, f32 out -- this suite
//  checks the whole path: construction, a forward chain, backward, Module.to,
//  an optimizer step, and a checkpoint round-trip.
//
//  What is deliberately NOT tested in f32: gradcheck. Central differences need
//  a step small enough to approximate a derivative and large enough to survive
//  subtraction, and f32 has ~7 digits to spend. The gradients are verified in
//  f64 (btorch_autograd_test) and f32 inherits those code paths unchanged --
//  there is no separate f32 backward. Asserting a 1e-7 gradcheck tolerance in
//  f32 would be asserting something arithmetic forbids.
//
//  Run:  bantu run tests/btorch_f32_test.b
// ════════════════════════════════════════════════════════════════════════

include "../btorch/btorch.b" as bt;
include "../btorch/nn.b" as nn;
include "../btorch/optim.b" as optim;
include "../btorch/serialize.b" as ser;

$R = {"pass": 0, "fail": 0};
def ok($cond, $name) {
    if ($cond) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $name); }
}
def eqs($got, $want, $name) {
    if ($got == $want) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name + "   got " + str($got) + ", want " + str($want));
    }
}
def errText($e) { if ($e == null) { return ""; } if (type($e) == "string") { return $e; } return str($e); }

print("");
print("  btorch f32 (D2c)");
print("  ════════════════════════════════════════");

// ── 1. the default dtype ────────────────────────────────────────────────────
print("");
print("-- the default dtype --");
eqs(bt.get_default_dtype(), "f64", "the default is f64, and stays f64 (E19)");
$threw = false;
try { bt.set_default_dtype("i64"); } catch ($e) { $threw = true; }
ok($threw, "set_default_dtype refuses a non-float dtype");
eqs(bt.get_default_dtype(), "f64", "and a refused call leaves the default alone");

$threw2 = false;
try { bt.zeros([2], false, "bool"); } catch ($e) { $threw2 = true; }
ok($threw2, "a constructor refuses a non-float dtype by name");

// ── 2. constructors ─────────────────────────────────────────────────────────
print("");
print("-- constructors --");
eqs(nd_dtype(bt.zeros([4], false, "f32").data),  "f32", "bt.zeros(..., \"f32\")");
eqs(nd_dtype(bt.ones([4], false, "f32").data),   "f32", "bt.ones(..., \"f32\")");
eqs(nd_dtype(bt.randn([4], false, "f32").data),  "f32", "bt.randn(..., \"f32\")");
eqs(nd_dtype(bt.rand([4], false, "f32").data),   "f32", "bt.rand(..., \"f32\")");
eqs(nd_dtype(bt.full([4], 2.0, false, "f32").data), "f32", "bt.full(..., \"f32\")");
eqs(nd_dtype(bt.arange(0.0, 4.0, false, "f32").data), "f32", "bt.arange(..., \"f32\")");
// omitting the dtype is exactly what it was
eqs(nd_dtype(bt.zeros([4], false).data), "f64", "omitting the dtype still gives f64");
eqs(nd_dtype(bt.tensor([1.0, 2.0], false).data), "f64", "a list with no dtype gives f64");
// an f32 ARRAY passes through rather than being widened back
$f32arr = nd_astype(nd_full([3], 1.5, "f64"), "f32");
eqs(nd_dtype(bt.tensor($f32arr, false).data), "f32",
    "bt.tensor(f32_array, false) STAYS f32, it is not widened");

// ── 3. an op chain stays f32, and so does the gradient ──────────────────────
print("");
print("-- forward and backward keep the dtype --");
$x = bt.tensor(nd_astype(nd_full([2, 3], 0.5, "f64"), "f32"), true);
$w = bt.tensor(nd_astype(nd_full([3, 4], 0.25, "f64"), "f32"), true);
$y = bt.matmul($x, $w);
eqs(nd_dtype($y.data), "f32", "matmul of two f32 tensors is f32 (this is what N31 unblocked)");
$z = bt.relu(bt.add($y, bt.tensor(nd_astype(nd_full([4], 0.1, "f64"), "f32"), true)));
eqs(nd_dtype($z.data), "f32", "add + relu keep f32");
$l = bt.sum($z, null, false);
eqs(nd_dtype($l.data), "f32", "sum keeps f32");
$l.backward(null);
eqs(nd_dtype($x.grad), "f32", "the gradient arriving at x is f32");
eqs(nd_dtype($w.grad), "f32", "the gradient arriving at w is f32");
ok(nd_shape($w.grad) == nd_shape($w.data), "and it has the parameter's shape");

// values: the same computation in f64 must agree to f32 resolution
$x6 = bt.tensor(nd_full([2, 3], 0.5, "f64"), true);
$w6 = bt.tensor(nd_full([3, 4], 0.25, "f64"), true);
$l6 = bt.sum(bt.relu(bt.add(bt.matmul($x6, $w6),
                            bt.tensor(nd_full([4], 0.1, "f64"), true))), null, false);
$l6.backward(null);
ok(nd_allclose(nd_astype($w.grad, "f64"), $w6.grad, 1.0e-6, 1.0e-6),
   "the f32 gradient agrees with the f64 one to 1e-6");

// ── 4. Module.to ────────────────────────────────────────────────────────────
print("");
print("-- Module.to --");
bt.manual_seed(4242);
$net = nn.Sequential([nn.Linear(6, 8, null), nn.ReLU(), nn.Linear(8, 3, null)]);
eqs($net.dtype(), "f64", "a freshly built module is f64");
$same = $net.to("f32");
eqs($net.dtype(), "f32", "to(\"f32\") casts every parameter");
ok($same.num_parameters() == $net.num_parameters(), "to() returns the module, for chaining");
$ps = $net.parameters();
$allf32 = true;
$i = 0;
while ($i < len($ps)) {
    if (nd_dtype($ps[$i].data) != "f32") { $allf32 = false; }
    if ($ps[$i].grad != null) { $allf32 = false; }
    $i = $i + 1;
}
ok($allf32, "every parameter is f32 and every stale gradient was dropped");
eqs($net.float().dtype(),  "f32", "float() is to(\"f32\")");
eqs($net.double().dtype(), "f64", "double() is to(\"f64\")");

// a shared weight must stay ONE object through a cast, or the optimizer would
// step it twice (the dedup in named_parameters is by identity)
$shared = nn.Linear(5, 5, null);
$tied = nn.Sequential([$shared, nn.ReLU(), $shared]);
$before = $tied.num_parameters();
$tied.to("f32");
eqs($tied.num_parameters(), $before,
    "a tied weight is still deduplicated after a cast (identity preserved)");

// a BatchNorm's integer buffer must NOT be turned into a float
$bn = nn.BatchNorm1d(4, null, null);
$bn.to("f32");
$bufs = $bn.buffers();
$intKept = false;
$k = 0;
while ($k < len($bufs)) {
    if (nd_dtype($bufs[$k].data) == "i64") { $intKept = true; }
    $k = $k + 1;
}
ok(nd_dtype($bn.buffer("running_mean").data) == "f32",
   "BatchNorm's running_mean casts to f32");
ok($intKept || len($bufs) == 2, "an integer buffer is left alone by to()");

// ── 5. a real training loop in f32 ──────────────────────────────────────────
print("");
print("-- training in f32 --");
bt.manual_seed(7);
$model = nn.Sequential([nn.Linear(8, 16, null), nn.ReLU(), nn.Linear(16, 1, null)]).to("f32");
$opt = optim.Adam($model.parameters(), 0.01, null, null, null);
$xs = bt.tensor(nd_astype(nd_random_uniform([64, 8], -1.0, 1.0), "f32"), false);
$ys = bt.tensor(nd_astype(nd_random_uniform([64, 1], -1.0, 1.0), "f32"), false);
$first = 0.0;
$last = 0.0;
$step = 0;
while ($step < 60) {
    $opt.zero_grad();
    $pred = $model.forward($xs);
    $loss = nn.mse_loss($pred, $ys, null);
    $loss.backward(null);
    $opt.step();
    if ($step == 0) { $first = $loss.item(); }
    $last = $loss.item();
    $step = $step + 1;
}
print("        loss " + str($first) + " -> " + str($last));
ok($last < $first, "the f32 loss goes down (an integration check, not a correctness one)");
eqs($model.dtype(), "f32", "the model is still f32 after 60 Adam steps");
ok($last == $last, "and the loss is finite, not NaN");

// ── 6. the checkpoint ───────────────────────────────────────────────────────
print("");
print("-- the checkpoint --");
$p32 = env("TMPDIR") + "bt_f32.btsd";
$p64 = env("TMPDIR") + "bt_f64.btsd";
ser.save($model, $p32);
$reload = nn.Sequential([nn.Linear(8, 16, null), nn.ReLU(), nn.Linear(16, 1, null)]).to("f32");
ser.load_model($reload, $p32);
eqs($reload.dtype(), "f32", "an f32 model reloads as f32, not widened to f64");
// bit-identical prediction, the same gate the MNIST numbers are quoted under
$a = $model.forward($xs);
$b = $reload.forward($xs);
ok(nd_all(nd_equal($a.data, $b.data, null), null, null),
   "the reloaded f32 model predicts BIT-IDENTICALLY");

// and the file is about half the size of the f64 equivalent
$model64 = nn.Sequential([nn.Linear(8, 16, null), nn.ReLU(), nn.Linear(16, 1, null)]);
ser.save($model64, $p64);
$s32 = len(readfile($p32, "rb"));
$s64 = len(readfile($p64, "rb"));
print("        checkpoint: f32 " + str($s32) + " bytes, f64 " + str($s64) + " bytes");
ok($s32 < $s64, "an f32 checkpoint is smaller than the f64 one");
ok($s32 * 2 > $s64 - 2000 && $s32 * 2 < $s64 + 2000,
   "and it is about half, not merely smaller (the payload halved)");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
