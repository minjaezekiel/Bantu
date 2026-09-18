// Fitting a line to noisy data, and checking the fit.
// Run: bantu run samples/numba/02_linear_fit.b
include "../../numba/numba.b" as np;

np.seed(7);
$n = 200;
$x = np.linspace(0, 10, $n, null);
// The truth we are trying to recover: y = 3x + 2, plus noise.
$y = np.add(np.add(np.multiply($x, 3.0, null), 2.0, null),
            np.randn([$n], 0.0, 0.8), null);

$c = np.polyfit($x, $y, 1);
print("fitted slope     " + str(np.get($c, [0])) + "   (true 3)");
print("fitted intercept " + str(np.get($c, [1])) + "   (true 2)");

// R^2: how much of the variance the fit explains.
$pred = np.polyval($c, $x);
$ssRes = np.get(np.sum(np.square(np.subtract($y, $pred, null), null), null, null), []);
$ssTot = np.get(np.sum(np.square(np.subtract($y, np.mean($y, null, null), null), null), null, null), []);
print("R^2              " + str(1.0 - $ssRes / $ssTot));
print("correlation      " + str(np.corrcoef($x, $y)));
