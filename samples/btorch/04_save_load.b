// 04_save_load.b — checkpointing, including the part that is easy to forget.
//
// A BatchNorm layer's running statistics are BUFFERS, not parameters. A
// checkpoint that saved only parameters would reload a model whose eval() is
// wrong -- and it would train fine and score badly, with nothing pointing at
// the loader. So this example asserts the buffers came back too.
include "../../btorch/btorch.b" as bt;
include "../../btorch/nn.b" as nn;
include "../../btorch/serialize.b" as ser;

$PATH = "/tmp/btorch_sample_model.btsd";

bt.manual_seed(5);
$net = nn.Sequential([
    nn.Linear(6, 8, null), nn.BatchNorm1d(8, null, null), nn.ReLU(),
    nn.Linear(8, 3, null)
]);
$x = bt.tensor(nd_random_normal([10, 6]), false);

// One training-mode pass, so the running statistics are no longer the defaults.
$net.train();
$net.forward($x);

$net.eval();
$before = $net.forward($x);

$info = ser.save($net, $PATH);
print("  saved " + str($info.entries) + " entries (" + str(len($net.parameters()))
    + " parameters + " + str(len($net.buffers())) + " buffers) in "
    + str($info.bytes) + " bytes");

// What is actually in the file.
$f = ser.load($PATH);
print("  file contains: " + str($f.order));
print("");

// A fresh model with DIFFERENT random weights.
bt.manual_seed(999);
$fresh = nn.Sequential([
    nn.Linear(6, 8, null), nn.BatchNorm1d(8, null, null), nn.ReLU(),
    nn.Linear(8, 3, null)
]);
$fresh.eval();
$diff = nd_allclose($fresh.forward($x).data, $before.data, 1e-9, 1e-9);
print("  before loading, the fresh model agrees with the saved one: " + str($diff));

$c = ser.load_model($fresh, $PATH);
print("  loaded " + str($c.params) + " parameters and " + str($c.buffers) + " buffers");
$after = $fresh.forward($x);
$same = nd_allclose($after.data, $before.data, 1e-15, 1e-15);
print("  after loading, identical to 1e-15: " + str($same));

$bufOk = nd_array_equal($fresh.get("1").buffer("running_var").data,
                        $net.get("1").buffer("running_var").data);
print("  the BatchNorm running variance matched too: " + str($bufOk));
print("");
if (!$diff && $same && $bufOk) { print("RESULT: ALL GREEN"); }
else { print("RESULT: FAILURES PRESENT"); }
