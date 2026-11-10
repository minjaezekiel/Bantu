// ════════════════════════════════════════════════════════════════════════
//  numba_pool_tsan_test.b — the pool's LIFETIME behaviour, sized for a
//  sanitizer.
//
//  numba_threading_test.b proves the answers do not depend on the thread
//  count. It cannot prove the pool's threads are done touching memory before
//  the caller moves on -- that is a lifetime property, and a test that
//  compares answers is blind to it: the answers were right while the pool had
//  a use-after-free in it (DECISIONS N33).
//
//  So this file is the one to run under ThreadSanitizer, on the FULL
//  interpreter rather than a standalone harness -- the standalone harness was
//  clean while the bug was live, because it warmed the pool before measuring
//  and so never created a worker inside a region under test.
//
//  Everything here is therefore sized for ~100x instrumentation overhead:
//  shapes just past the parallel work threshold (M*K*N >= 2^20) instead of
//  comfortably past it, and region COUNT rather than region size, since what
//  races is the per-region wake/report handshake, not the arithmetic. It runs
//  in ~30 ms uninstrumented, so it also sits in the normal suite.
//
//  Run:  bantu run tests/numba_pool_tsan_test.b
//  Gate: g++ -fsanitize=thread ... && TSAN_OPTIONS=halt_on_error=0 \
//          bantu-tsan run tests/numba_pool_tsan_test.b
// ════════════════════════════════════════════════════════════════════════

$R = {"pass": 0, "fail": 0};
def ok($c, $n) {
    if ($c) { $R.pass = $R.pass + 1; print("  ok    " + $n); }
    else    { $R.fail = $R.fail + 1; print("  FAIL  " + $n); }
}
def exact($x, $y, $n) {
    ok(nd_get(nd_all(nd_equal($x, $y, null), null, null), 0), $n);
    return null;
}

print("");
print("  TSan: pool under the interpreter");
nd_seed(31337);

// 1. both gemm kernels, ragged and aligned, over a thread-count sweep.
//    129x66x129 = 1,098,306 MACs: just past the threshold, partial last block.
$shapes = [[128, 64, 128], [129, 66, 129]];
$si = 0;
while ($si < len($shapes)) {
    $sh = $shapes[$si];
    $a  = nd_random_uniform([$sh[0], $sh[1]], -1.0, 1.0);
    $b  = nd_random_uniform([$sh[1], $sh[2]], -1.0, 1.0);
    $at = nd_transpose(nd_random_uniform([$sh[1], $sh[0]], -1.0, 1.0), null);
    nd_set_threads(1);
    $ref  = nd_matmul($a, $b);
    $refT = nd_matmul($at, $b);
    $tag  = str($sh[0]) + "x" + str($sh[1]) + "x" + str($sh[2]);
    for $t in [2, 4, 9] {
        nd_set_threads($t);
        exact(nd_matmul($a, $b),  $ref,  $tag + " contiguous @" + str($t));
        exact(nd_matmul($at, $b), $refT, $tag + " transposed-A @" + str($t));
    }
    nd_set_threads(1);
    $si = $si + 1;
}

// 2. many back-to-back regions: worker wake/sleep across generations is the
//    part most likely to race, and it is per-REGION, not per-FLOP.
print("");
$x = nd_random_uniform([128, 64], -1.0, 1.0);
$y = nd_random_uniform([64, 128], -1.0, 1.0);
nd_set_threads(4);
$r1 = nd_matmul($x, $y);
$i = 0;  $same = 1;
while ($i < 40) {
    if (!nd_get(nd_all(nd_equal(nd_matmul($x, $y), $r1, null), null, null), 0)) { $same = 0; }
    $i = $i + 1;
}
ok($same == 1, "40 consecutive regions, same bits, workers reused");

// 3. the thread count changing BETWEEN regions (shrink and grow)
for $t in [2, 8, 3, 16, 1, 5] {
    nd_set_threads($t);
    exact(nd_matmul($x, $y), $r1, "count changed to " + str($t) + " mid-stream");
}

// 4. interpreter-side work interleaved with regions: allocation, refcounting
//    and the GC running while workers are parked.
nd_set_threads(6);
$j = 0;  $acc = nd_zeros([128, 128], "f64");
while ($j < 12) {
    $acc = nd_add($acc, nd_matmul($x, $y), null);
    $junk = nd_random_uniform([200, 200], -1.0, 1.0);   // garbage, left to die
    $j = $j + 1;
}
ok(nd_allclose($acc, nd_multiply($r1, 12.0, null), 1.0e-9, 1.0e-9),
   "regions interleaved with allocation and collection");

// 5. f32 storage through the same pool
$a32 = nd_astype(nd_random_uniform([128, 128], -1.0, 1.0), "f32");
$b32 = nd_astype(nd_random_uniform([128, 128], -1.0, 1.0), "f32");
nd_set_threads(1);
$f1 = nd_matmul($a32, $b32);
nd_set_threads(7);
exact(nd_matmul($a32, $b32), $f1, "f32 through the pool");
nd_set_threads(1);

print("");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
