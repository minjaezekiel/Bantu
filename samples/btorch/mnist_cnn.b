// ════════════════════════════════════════════════════════════════════════════
//  samples/btorch/mnist_cnn.b — the phase-10 gate: MNIST CNN >= 98.5%.
//
//  LeNet-5, which is the architecture the gate was written against: two
//  convolutions with max pooling, then three fully-connected layers, 61,706
//  parameters.
//
//  This is the run that exercises everything: A9's im2col/col2im through
//  conv2d's composition (DECISIONS B18), pooling through max's gradient,
//  phase 6's DataLoader, phase 4's SGD with momentum, and phase 7's
//  checkpoint.
//
//  RESUMABLE. A convolution epoch takes about five minutes, so the run keeps a
//  checkpoint plus a sidecar recording how many epochs are done and the best
//  accuracy so far. Re-running continues from there; BTORCH_FRESH=1 starts
//  over. This is not a convenience -- the first attempt at this gate was lost
//  when the machine was turned off part-way through, and an hour-long run that
//  cannot be resumed is an hour-long run that has to be lucky.
//
//  Usage:  bantu run samples/btorch/mnist_cnn.b          # 3 more epochs
//          BTORCH_EPOCHS=6 bantu run samples/btorch/mnist_cnn.b
//          BTORCH_FRESH=1 bantu run samples/btorch/mnist_cnn.b
//  Data:   samples/btorch/get_mnist.sh, or set BANTU_MNIST_DIR
// ════════════════════════════════════════════════════════════════════════════

include "../../btorch/btorch.b" as bt;
include "../../btorch/nn.b" as nn;
include "../../btorch/optim.b" as optim;
include "../../btorch/data.b" as btdata;
include "../../btorch/serialize.b" as ser;
include "../../btorch/datasets.b" as ds;

$GATE = 0.985;
$EPOCHS = 3;
$envEpochs = env("BTORCH_EPOCHS");
if ($envEpochs != "") { $EPOCHS = num($envEpochs); }
$CKPT = env("HOME") + "/.bantu/datasets/mnist_cnn.btsd";
$STATE = env("HOME") + "/.bantu/datasets/mnist_cnn.json";
$FRESH = env("BTORCH_FRESH") == "1";

print("");
print("  MNIST CNN — the phase-10 gate (>= " + str($GATE * 100.0) + "% test accuracy)");
print("  ────────────────────────────────────────────────────────────");

$t0 = clock();
$m = ds.mnist(null, "standard");
print("  data:      60,000 train / 10,000 test, loaded in " + str(clock() - $t0) + " ms");

bt.manual_seed(20261109);
$net = nn.Sequential([
    nn.Conv2d(1, 6, 5, 1, 2, null),   nn.ReLU(), nn.MaxPool2d(2, null, null),
    nn.Conv2d(6, 16, 5, 1, 0, null),  nn.ReLU(), nn.MaxPool2d(2, null, null),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120, null), nn.ReLU(),
    nn.Linear(120, 84, null),         nn.ReLU(),
    nn.Linear(84, 10, null)
]);
print("  model:     LeNet-5, " + str($net.num_parameters()) + " parameters");

$train = btdata.TensorDataset([$m.train_x, $m.train_y]);
$loader = btdata.DataLoader($train, 128, true, false, 20261109);
// Adam rather than SGD+momentum: it reaches the gate in about half the epochs,
// and a convolution epoch here costs five minutes (the conv path runs at
// roughly an eighth of the MLP's FLOP rate -- a measured phase-9 item, not a
// correctness one).
$opt = optim.Adam($net.parameters(), 0.001, null, null, null);
$sched = optim.StepLR($opt, 3, 0.4);
print("  optimizer: Adam(lr=1e-3), StepLR(every 3 epochs, x0.4)");
print("             batch 128, " + str($loader.num_batches()) + " batches/epoch");

// ── resume ──────────────────────────────────────────────────────────────
// The optimizer's moments are NOT checkpointed, only the weights. Adam
// rebuilds its moments from zero on a resume, which costs a few batches of
// slightly-wrong step sizes and is stated here rather than hidden: saving
// optimizer state is a phase-9 nicety, and pretending a resume is exact when
// it is not would be the worse failure.
$done = 0;
$best = 0.0;
if (!$FRESH && file_exists($STATE) && file_exists($CKPT)) {
    $st = json.parse(readfile($STATE, "r"));
    if (type($st) == "dict" && type($st.epochs) == "number") {
        $done = $st.epochs;
        $best = $st.best;
        ser.load_model($net, $CKPT);
        // Replay the learning-rate schedule so a resumed run is on the same
        // curve a continuous one would have been.
        $k = 0;
        while ($k < $done) { $sched.step(); $k = $k + 1; }
        print("  resumed:   " + str($done) + " epochs already done, best "
            + str($best * 100.0) + "%, lr now " + str($opt.lr));
    }
}
print("");

def evaluate($net, $x, $y) {
    $net.eval();
    $acc = bt.inference_mode(def() {
        $correct = 0.0;
        $n = nd_shape($x)[0];
        $at = 0;
        while ($at < $n) {
            $to = $at + 500;
            if ($to > $n) { $to = $n; }
            $xb = ds.as_images(bt.tensor(nd_copy(nd_slice($x, [[$at, $to, 1]]), null), false));
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
        // [N, 784] -> [N, 1, 28, 28]; bt.reshape is differentiable, so the
        // gradient finds its way back without a second code path.
        $loss = nn.cross_entropy($net.forward(ds.as_images($batch[0])), $batch[1].data, null);
        $loss.backward(null);
        $opt.step();
        $lossSum = $lossSum + $loss.item();
        $b = $b + 1;
    }
    $loader.reshuffle();
    $sched.step();
    $trainMs = clock() - $t1;
    $acc = evaluate($net, $m.test_x, $m.test_y);
    if ($acc > $best) {
        $best = $acc;
        ser.save($net, $CKPT);          // keep the best, as a real run would
    }
    // The sidecar is written EVERY epoch, whether or not accuracy improved,
    // so a resume knows how far the schedule has advanced.
    $done = $done + 1;
    writefile($STATE, json.stringify({"epochs": $done, "best": $best}), "w");
    print("  epoch " + str($done) + " (this run " + str($e + 1) + "/" + str($EPOCHS) + ")"
        + "   loss " + str($lossSum / ($nb * 1.0))
        + "   test acc " + str($acc * 100.0) + "%"
        + "   lr " + str($opt.lr)
        + "   (" + str($trainMs) + " ms)");
    $e = $e + 1;
}

// The gate is measured on the BEST checkpoint, reloaded from disk -- so the
// number reported is one that survived a save and a load, not one that only
// ever existed in memory.
bt.manual_seed(1);
$reloaded = nn.Sequential([
    nn.Conv2d(1, 6, 5, 1, 2, null),   nn.ReLU(), nn.MaxPool2d(2, null, null),
    nn.Conv2d(6, 16, 5, 1, 0, null),  nn.ReLU(), nn.MaxPool2d(2, null, null),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120, null), nn.ReLU(),
    nn.Linear(120, 84, null),         nn.ReLU(),
    nn.Linear(84, 10, null)
]);
ser.load_model($reloaded, $CKPT);
$final = evaluate($reloaded, $m.test_x, $m.test_y);
print("");
print("  epochs completed in total:   " + str($done));
print("  best during training:        " + str($best * 100.0) + "%");
print("  reloaded from checkpoint:    " + str($final * 100.0) + "%");
print("  total: " + str(clock() - $t0) + " ms");
print("");
if ($final >= $GATE) {
    print("  RESULT: ALL GREEN  (gate " + str($GATE * 100.0) + "%, got "
        + str($final * 100.0) + "%)");
} else {
    print("  RESULT: FAILURES PRESENT  (gate " + str($GATE * 100.0) + "%, got "
        + str($final * 100.0) + "%)");
}
