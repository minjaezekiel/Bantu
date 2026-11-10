// ════════════════════════════════════════════════════════════════════════
//  numba_threading_test.b — deterministic threading (stage 1, item D3).
//
//  The whole value of this phase rests on ONE property: the answer does not
//  depend on how many threads computed it. Not "agrees to a tolerance" --
//  BIT-IDENTICAL. Floating-point addition is not associative, so a pool that
//  let completion order decide summation order would change results run to
//  run, and reproducibility is the thing this stack is being built around
//  (DECISIONS N25).
//
//  So every assertion here is exact equality against the single-threaded
//  answer, over several shapes and both gemm paths, at thread counts above and
//  below the number of cores.
//
//  Determinism comes from two things, and the tests are shaped to catch either
//  breaking:
//    * chunk BOUNDARIES are a function of shape alone, never of the thread
//      count -- so sweeping the thread count must change nothing;
//    * each chunk writes only its own output rows -- so a shape with an odd
//      number of blocks, where the last block is partial, must also agree.
//
//  Run:  bantu run tests/numba_threading_test.b
// ════════════════════════════════════════════════════════════════════════

$R = {"pass": 0, "fail": 0};
def ok($cond, $name) {
    if ($cond) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $name); }
}
def eqs($got, $want, $name) {
    if ($got == $want) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name + "   got " + str($got) + ", want " + str($want));
    }
}
def exact($x, $y, $name) {
    if (nd_shape($x) != nd_shape($y)) {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name + "  shape " + str(nd_shape($x)) + " vs " + str(nd_shape($y)));
        return null;
    }
    ok(nd_get(nd_all(nd_equal($x, $y, null), null, null), 0), $name);
    return null;
}

print("");
print("  numba deterministic threading (D3)");
print("  ════════════════════════════════════════");

// ── 1. the knob ─────────────────────────────────────────────────────────────
print("");
print("-- nd_set_threads / nd_get_threads --");
eqs(nd_get_threads(), 1, "the default is 1 thread: threading is OPT-IN");
ok(nd_hardware_threads() >= 1, "nd_hardware_threads reports at least one");
eqs(nd_set_threads(4), 4, "nd_set_threads returns the count it adopted");
eqs(nd_get_threads(), 4, "and nd_get_threads reports it");
eqs(nd_set_threads(0), nd_hardware_threads(), "0 means \"as many as the hardware reports\"");
eqs(nd_set_threads(1), 1, "and 1 restores fully sequential execution");

$bad = 0;
try { nd_set_threads(-1); }  catch ($e) { $bad = $bad + 1; }
try { nd_set_threads(2.5); } catch ($e) { $bad = $bad + 1; }
try { nd_set_threads(99999); } catch ($e) { $bad = $bad + 1; }
try { nd_set_threads("4"); } catch ($e) { $bad = $bad + 1; }
try { nd_set_threads(null); } catch ($e) { $bad = $bad + 1; }
eqs($bad, 5, "a negative, fractional, absurd, non-numeric or missing count each raise");
eqs(nd_get_threads(), 1, "and a refused call leaves the count alone");

// ── 2. bit-identity across thread counts ────────────────────────────────────
// Shapes chosen around the 64-row block size: one exactly on a block boundary,
// one with a partial last block, one smaller than a single block, and one
// skinny. 257 is deliberate -- 4 full blocks plus a block of ONE row.
print("");
print("-- the core property: identical bits at any thread count --");
nd_seed(31337);
$shapes = [[256, 192, 128], [257, 193, 65], [64, 64, 64], [300, 40, 7]];
$si = 0;
while ($si < len($shapes)) {
    $sh = $shapes[$si];
    $M = $sh[0];  $K = $sh[1];  $N = $sh[2];
    $a  = nd_random_uniform([$M, $K], -1.0, 1.0);
    $b  = nd_random_uniform([$K, $N], -1.0, 1.0);
    // a transposed left operand takes the OTHER gemm kernel
    $at = nd_transpose(nd_random_uniform([$K, $M], -1.0, 1.0), null);

    nd_set_threads(1);
    $ref  = nd_matmul($a, $b);
    $refT = nd_matmul($at, $b);

    $tag = str($M) + "x" + str($K) + "x" + str($N);
    for $t in [2, 3, 4, 8, 16] {
        nd_set_threads($t);
        exact(nd_matmul($a, $b),   $ref,  $tag + " contiguous, " + str($t) + " threads");
        exact(nd_matmul($at, $b),  $refT, $tag + " transposed-A, " + str($t) + " threads");
    }
    nd_set_threads(1);
    $si = $si + 1;
}

// ── 3. f32 too ──────────────────────────────────────────────────────────────
// f32 has a quarter of f64's mantissa, so if a thread count ever did perturb
// the summation order it would show here first.
print("");
print("-- f32 is identical across thread counts as well --");
$a32 = nd_astype(nd_random_uniform([256, 256], -1.0, 1.0), "f32");
$b32 = nd_astype(nd_random_uniform([256, 256], -1.0, 1.0), "f32");
nd_set_threads(1);
$r32 = nd_matmul($a32, $b32);
eqs(nd_dtype($r32), "f32", "the f32 matmul is still f32 under threading");
for $t in [2, 4, 8] {
    nd_set_threads($t);
    exact(nd_matmul($a32, $b32), $r32, "f32 256^3, " + str($t) + " threads");
}
nd_set_threads(1);

// ── 4. the answer did not move from the pre-D3 one ──────────────────────────
// Threading must not change the result AT ALL, not merely be self-consistent.
// A four-loop reference convolution-style check: matmul against a hand-written
// accumulation, which the generic path also has to match.
print("");
print("-- and it still matches a reference computed without matmul --");
$p = nd_random_uniform([40, 24], -1.0, 1.0);
$q = nd_random_uniform([24, 16], -1.0, 1.0);
nd_set_threads(8);
$mm = nd_matmul($p, $q);
// reference: sum over k of outer products, built with ufuncs only
$acc = nd_zeros([40, 16], "f64");
$k = 0;
while ($k < 24) {
    $col = nd_reshape(nd_copy(nd_slice($p, [null, [$k, $k + 1, 1]]), null), [40, 1]);
    $row = nd_reshape(nd_copy(nd_slice($q, [[$k, $k + 1, 1], null]), null), [1, 16]);
    $acc = nd_add($acc, nd_multiply($col, $row, null), null);
    $k = $k + 1;
}
ok(nd_allclose($mm, $acc, 1.0e-12, 1.0e-12),
   "a threaded matmul equals an independent ufunc-only reference");
nd_set_threads(1);

// ── 5. no leak, no per-call allocation growth ───────────────────────────────
print("");
print("-- memory --");
$x = nd_random_uniform([128, 128], -1.0, 1.0);
$y = nd_random_uniform([128, 128], -1.0, 1.0);
nd_set_threads(4);
nd_matmul($x, $y);                        // warm: create the workers
$before = nd_live_bytes();
$i = 0;
while ($i < 300) { nd_matmul($x, $y); $i = $i + 1; }
$after = nd_live_bytes();
print("        live bytes " + str($before) + " -> " + str($after) + " over 300 threaded matmuls");
// Does not GROW, rather than exactly equal. A leak is monotonic growth; a
// DECREASE is legitimate and is what actually happens here -- the warm-up
// matmul's own result was still live at the `before` reading and is collected
// during the loop, which showed up as exactly 131072 bytes, one 128x128 f64
// array. Asserting equality made the test fail on the collector doing its job.
ok($after <= $before, "300 threaded regions do not grow live memory (no leak)");
// and the pool itself is not an array allocation at all
ok($after > 0, "the arrays under test are still live, so the reading is meaningful");
nd_set_threads(1);

// ── 6. small work still runs, and still agrees ──────────────────────────────
// Below the work threshold a region runs sequentially. That is a performance
// decision, so it must be invisible in the answer.
print("");
print("-- below the parallel threshold --");
$s1 = nd_random_uniform([8, 5], -1.0, 1.0);
$s2 = nd_random_uniform([5, 3], -1.0, 1.0);
nd_set_threads(1);
$sref = nd_matmul($s1, $s2);
nd_set_threads(16);
exact(nd_matmul($s1, $s2), $sref, "a tiny matmul is identical with 16 threads requested");
// a 1x1 result, and an empty one
exact(nd_matmul(nd_ones([1, 4], "f64"), nd_ones([4, 1], "f64")),
      nd_full([1, 1], 4.0, "f64"), "a 1x1 result is right under threading");
nd_set_threads(1);

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
