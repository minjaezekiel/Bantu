// 02_linear_regression.b — fit y = 3x + 2 from noisy samples.
//
// The smallest end-to-end training loop: a parameter, a loss, an optimizer,
// and a number you can check against the truth.
include "../../btorch/btorch.b" as bt;
include "../../btorch/nn.b" as nn;
include "../../btorch/optim.b" as optim;

bt.manual_seed(7);
$TRUE_W = 3.0;
$TRUE_B = 2.0;

$x = nd_random_uniform([200, 1], 0.0 - 1.0, 1.0);
$noise = nd_multiply(nd_random_normal([200, 1]), 0.05, null);
$y = nd_add(nd_add(nd_multiply($x, $TRUE_W, null), $TRUE_B, null), $noise, null);

$model = nn.Linear(1, 1, null);
$opt = optim.SGD($model.parameters(), 0.5, 0.9, null);

$tx = bt.tensor($x, false);
$ty = bt.tensor($y, false);

$step = 0;
while ($step < 200) {
    $opt.zero_grad();
    $loss = nn.mse_loss($model.forward($tx), $ty, null);
    $loss.backward(null);
    $opt.step();
    if ($step % 50 == 0) {
        print("  step " + str($step) + "   loss " + str($loss.item()));
    }
    $step = $step + 1;
}

$w = nd_get(nd_reshape($model.param("weight").data, [1]), [0]);
$b = nd_get(nd_reshape($model.param("bias").data, [1]), [0]);
print("");
print("  learned  w = " + str($w) + "   b = " + str($b));
print("  truth    w = " + str($TRUE_W) + "   b = " + str($TRUE_B));

$we = $w - $TRUE_W; if ($we < 0.0) { $we = 0.0 - $we; }
$be = $b - $TRUE_B; if ($be < 0.0) { $be = 0.0 - $be; }
print("");
if ($we < 0.05 && $be < 0.05) { print("RESULT: ALL GREEN  (both within 0.05 of the truth)"); }
else { print("RESULT: FAILURES PRESENT  (w off by " + str($we) + ", b off by " + str($be) + ")"); }
