// ════════════════════════════════════════════════════════════════════════════
//  samples/btorch/mnist_mlp.b — the phase-10 gate: MNIST MLP >= 97%.
//
//  Every unit test in btorch-suite can pass while the library is still unable
//  to train anything. This is the run that cannot.
//
//  Usage:  bantu run samples/btorch/mnist_mlp.b
//          BTORCH_EPOCHS=8 bantu run samples/btorch/mnist_mlp.b
//  Data:   samples/btorch/get_mnist.sh, or set BANTU_MNIST_DIR
//
//  Epochs come from the environment, not a command-line argument: `bantu run
//  <file> <arg>` treats the extra argument as another FILE to run, so there is
//  no argv for a script to read.
// ════════════════════════════════════════════════════════════════════════════

include "../../btorch/btorch.b" as bt;
include "../../btorch/nn.b" as nn;
include "../../btorch/optim.b" as optim;
include "../../btorch/data.b" as btdata;
include "../../btorch/datasets.b" as ds;

$GATE = 0.97;
$EPOCHS = 4;
$envEpochs = env("BTORCH_EPOCHS");
if ($envEpochs != "") { $EPOCHS = num($envEpochs); }

print("");
print("  MNIST MLP — the phase-10 gate (>= " + str($GATE * 100.0) + "% test accuracy)");
print("  ────────────────────────────────────────────────────────────");

$t0 = clock();
$m = ds.mnist(null, "standard");
print("  data:      60,000 train / 10,000 test, loaded in " + str(clock() - $t0) + " ms");

bt.manual_seed(20261109);
$net = nn.Sequential([
    nn.Linear(784, 256, null), nn.ReLU(),
    nn.Linear(256, 128, null), nn.ReLU(),
    nn.Linear(128, 10, null)
]);
print("  model:     784-256-128-10 MLP, " + str($net.num_parameters()) + " parameters");

$train = btdata.TensorDataset([$m.train_x, $m.train_y]);
$loader = btdata.DataLoader($train, 128, true, false, 20261109);
$opt = optim.Adam($net.parameters(), 0.001, null, null, null);
print("  optimizer: Adam(lr=1e-3), batch 128, " + str($loader.num_batches()) + " batches/epoch");
print("");

// Evaluated in eval() mode and under inference_mode: no dropout here, but the
// tape would otherwise build 10,000 rows of graph for nothing.
def evaluate($net, $x, $y) {
    $net.eval();
    $acc = bt.inference_mode(def() {
        $correct = 0.0;
        $n = nd_shape($x)[0];
        $at = 0;
        while ($at < $n) {
            $to = $at + 2000;
            if ($to > $n) { $to = $n; }
            $xb = bt.tensor(nd_copy(nd_slice($x, [[$at, $to, 1]]), null), false);
            $yb = nd_copy(nd_slice($y, [[$at, $to, 1]]), null);
            $correct = $correct + ds.accuracy($net.forward($xb), $yb) * ($to - $at);
            $at = $to;
        }
        return $correct / ($n * 1.0);
    });
    $net.train();
    return $acc;
}

$e = 0;
while ($e < $EPOCHS) {
    $net.train();
    $t1 = clock();
    $lossSum = 0.0;
    $b = 0;
    $nb = $loader.num_batches();
    while ($b < $nb) {
        $batch = $loader.batch($b);
        $opt.zero_grad();
        $loss = nn.cross_entropy($net.forward($batch[0]), $batch[1].data, null);
        $loss.backward(null);
        $opt.step();
        $lossSum = $lossSum + $loss.item();
        $b = $b + 1;
    }
    $loader.reshuffle();
    $trainMs = clock() - $t1;
    $acc = evaluate($net, $m.test_x, $m.test_y);
    print("  epoch " + str($e + 1) + "/" + str($EPOCHS)
        + "   loss " + str($lossSum / ($nb * 1.0))
        + "   test acc " + str($acc * 100.0) + "%"
        + "   (" + str($trainMs) + " ms)");
    $e = $e + 1;
}

$final = evaluate($net, $m.test_x, $m.test_y);
$trainAcc = evaluate($net, $m.train_x, $m.train_y);
print("");
print("  final test accuracy:  " + str($final * 100.0) + "%");
print("  final train accuracy: " + str($trainAcc * 100.0) + "%");
print("  total: " + str(clock() - $t0) + " ms");
print("");
if ($final >= $GATE) {
    print("  RESULT: ALL GREEN  (gate " + str($GATE * 100.0) + "%, got "
        + str($final * 100.0) + "%)");
} else {
    print("  RESULT: FAILURES PRESENT  (gate " + str($GATE * 100.0) + "%, got "
        + str($final * 100.0) + "%)");
}
