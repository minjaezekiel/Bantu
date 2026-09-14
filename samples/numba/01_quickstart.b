// numba quickstart — three lines to a result, then a short tour.
// Run: bantu run samples/numba/01_quickstart.b
include "../../numba/numba.b" as np;

$a = np.array([[1, 2], [3, 4]], null);
print($a);
print("sum:  " + str(np.get($a.sum(), [])));
print("mean: " + str(np.get($a.mean(null, null), [])));

print("");
print("-- operators work directly --");
$x = np.linspace(0, 1, 5, null);
print("x       " + str(np.to_list($x)));
print("x*x + x " + str(np.to_list(($x * $x) + $x)));
print("x > 0.5 " + str(np.to_list($x > 0.5)));

print("");
print("-- views cost nothing --");
$m = np.reshape(np.arange(0, 12, null), [3, 4]);
$t = np.T($m);
print("m is " + str(np.shape($m)) + ", its transpose is " + str(np.shape($t)));
print("same memory? " + str(np.shares_memory($m, $t)));
$m[0][0] = 99;
print("writing m[0][0]=99 changes the transpose too: " + str(np.get($t, [0, 0])));

print("");
print("-- reductions over an axis --");
$g = np.reshape(np.arange(1, 7, null), [2, 3]);
print("g            " + str(np.to_list($g)));
print("sum axis 0   " + str(np.to_list(np.sum($g, 0, null))));
print("sum axis 1   " + str(np.to_list(np.sum($g, 1, null))));
print("row-centred  " + str(np.to_list(np.subtract($g, np.mean($g, 1, true), null))));
