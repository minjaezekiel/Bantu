// Working with large arrays without running out of memory.
// Run: bantu run samples/numba/04_large_arrays.b
include "../../numba/numba.b" as np;

$n = 5000000;
np.seed(3);
$a = np.random([$n], null, null);
$b = np.random([$n], null, null);

print("two " + str($n) + "-element arrays: " + str(np.live_bytes() / 1048576) + " MB live");

// The naive way allocates a fresh result every time.
$t0 = clock();
$c = np.add($a, $b, null);
print("add, allocating a result : " + str(clock() - $t0) + " ms");

// A reused destination allocates nothing at all, which is what lets a loop
// over large arrays run in constant memory.
$out = np.empty([$n], null);
$t0 = clock();
$before = np.live_bytes();
$i = 0;
while ($i < 20) {
    np.add($a, $b, $out);
    $i = $i + 1;
}
print("20 adds into a destination: " + str(clock() - $t0) + " ms");
print("memory grew by            : " + str(np.live_bytes() - $before) + " bytes");

// Accuracy at scale: ten million copies of 0.1.
$tenth = np.full([10000000], 0.1, null);
$s = np.get(np.sum($tenth, null, null), []);
print("");
print("sum of 10M x 0.1 = " + str($s));
print("relative error   = " + str(($s - 1000000.0) / 1000000.0) + "  (a naive loop loses ~2e-12)");
