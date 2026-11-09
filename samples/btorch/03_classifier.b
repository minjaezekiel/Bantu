// 03_classifier.b — an MLP on two interleaving spirals.
//
// A problem a linear model cannot solve, so reaching high accuracy is
// evidence the non-linearity and its gradient both work.
include "../../btorch/btorch.b" as bt;
include "../../btorch/nn.b" as nn;
include "../../btorch/optim.b" as optim;
include "../../btorch/data.b" as btdata;

bt.manual_seed(11);
$PER = 300;

// Two spirals, one per class.
$xs = nd_zeros([$PER * 2, 2], "f64");
$ys = nd_zeros([$PER * 2], "f64");
$i = 0;
while ($i < $PER) {
    $k = 0;
    while ($k < 2) {
        $t = ($i * 1.0 / $PER) * 4.0;
        $ang = $t * 2.5 + $k * 3.14159265358979;
        $r = 0.3 + $t * 0.25;
        $row = $i * 2 + $k;
        nd_set($xs, [$row, 0], $r * nd_get(nd_cos(nd_full([1], $ang), null), [0]));
        nd_set($xs, [$row, 1], $r * nd_get(nd_sin(nd_full([1], $ang), null), [0]));
        nd_set($ys, [$row], $k * 1.0);
        $k = $k + 1;
    }
    $i = $i + 1;
}

$ds = btdata.TensorDataset([$xs, $ys]);
$parts = btdata.random_split($ds, [500, 100], 11);
$loader = btdata.DataLoader($parts[0], 32, true, false, 11);

$net = nn.Sequential([
    nn.Linear(2, 64, null), nn.Tanh(),
    nn.Linear(64, 64, null), nn.Tanh(),
    nn.Linear(64, 2, null)
]);
$opt = optim.Adam($net.parameters(), 0.01, null, null, null);
print("  model: 2-64-64-2 MLP, " + str($net.num_parameters()) + " parameters");
print("  data:  600 points on two spirals, split 500 train / 100 test");
print("");

def accuracy($net, $subset) {
    $all = $subset.batch_of(nd_astype(nd_arange(0.0, $subset.size() * 1.0, null), "i64"));
    return bt.inference_mode(def() {
        $pred = nd_argmax($net.forward($all[0]).data, 1, null);
        $hit = nd_sum(nd_astype(nd_equal(nd_astype($pred, "i64"),
                                         nd_astype($all[1].data, "i64"), null), "f64"), null, null);
        return nd_get(nd_reshape($hit, [1]), [0]) / ($subset.size() * 1.0);
    });
}

$e = 0;
while ($e < 60) {
    $b = 0;
    while ($b < $loader.num_batches()) {
        $batch = $loader.batch($b);
        $opt.zero_grad();
        nn.cross_entropy($net.forward($batch[0]), $batch[1].data, null).backward(null);
        $opt.step();
        $b = $b + 1;
    }
    $loader.reshuffle();
    if (($e + 1) % 20 == 0) {
        print("  epoch " + str($e + 1) + "   train " + str(accuracy($net, $parts[0]) * 100.0)
            + "%   test " + str(accuracy($net, $parts[1]) * 100.0) + "%");
    }
    $e = $e + 1;
}

$acc = accuracy($net, $parts[1]);
print("");
print("  final test accuracy: " + str($acc * 100.0) + "%");
if ($acc >= 0.95) { print("RESULT: ALL GREEN  (>= 95% on held-out spirals)"); }
else { print("RESULT: FAILURES PRESENT  (got " + str($acc * 100.0) + "%)"); }
