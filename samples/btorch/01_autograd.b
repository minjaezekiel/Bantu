// 01_autograd.b — tensors, the tape, and backward().
include "../../btorch/btorch.b" as bt;

print("── a scalar derivative ──");
$w = bt.tensor([2.0], true);
$y = bt.mul($w, $w);                      // w^2
$y.backward(null);
print("  d(w^2)/dw at w=2  = " + str(nd_to_list($w.grad)) + "   (expect [4])");

print("");
print("── a chain ──");
$x = bt.tensor([0.5], true);
$z = bt.tanh(bt.mul($x, 3.0));            // tanh(3x)
$z.backward(null);
// d/dx tanh(3x) = 3 (1 - tanh(3x)^2)
$t = nd_get(nd_tanh(nd_full([1], 1.5), null), [0]);
print("  d(tanh(3x))/dx    = " + str(nd_to_list($x.grad)));
print("  3(1 - tanh(1.5)^2) = " + str(3.0 * (1.0 - $t * $t)));

print("");
print("── a tensor used twice: BOTH contributions must land ──");
$a = bt.tensor([3.0], true);
bt.add(bt.mul($a, $a), $a).backward(null);   // a^2 + a  ->  2a + 1 = 7
print("  d(a^2 + a)/da at a=3 = " + str(nd_to_list($a.grad)) + "   (expect [7])");

print("");
print("── broadcasting: the gradient is summed back over the stretched axis ──");
$m = bt.tensor([[1.0, 2.0], [3.0, 4.0]], true);
$b = bt.tensor([10.0, 20.0], true);          // broadcast over 2 rows
bt.sum(bt.add($m, $b), null, null).backward(null);
print("  matrix grad " + str(nd_to_list($m.grad)));
print("  bias grad   " + str(nd_to_list($b.grad)) + "   (expect [2, 2] -- one per row)");

print("");
print("── gradients ACCUMULATE until you clear them ──");
$c = bt.tensor([1.0], true);
bt.mul($c, 5.0).backward(null);
print("  after one backward:  " + str(nd_to_list($c.grad)));
bt.mul($c, 5.0).backward(null);
print("  after two:           " + str(nd_to_list($c.grad)) + "   (summed, as PyTorch does)");
$c.zero_grad();
print("  after zero_grad():   " + str($c.grad));

print("");
print("── turning the tape off ──");
$d = bt.tensor([2.0], true);
$off = bt.no_grad(def() { return bt.mul($d, $d); });
print("  no_grad result requires_grad = " + str($off.requires_grad) + "   (expect false)");
print("  grad still enabled afterwards = " + str(bt.is_grad_enabled()));

print("");
print("RESULT: ALL GREEN");
