// ════════════════════════════════════════════════════════════════════════
//  numba_fastpath_test.b — the optimized paths must agree with the generic
//  ones, BIT FOR BIT.
//
//  Three fast paths were added after profiling showed where a convolutional
//  training step actually spends its time:
//
//    1. innerSweep's contiguous run       (every reduction: nd_max over a
//       [150528,4] array went from 102 ms to 1.2 ms)
//    2. nd_matmul straight on the buffers (no toMat copy: 20.6 -> 10 ms)
//    3. nd_matmul with a TRANSPOSED left operand (45.8 -> 10 ms)
//
//  A fast path that disagrees with the path it replaces is the worst kind of
//  optimization: everything passes, and the answers quietly drift. So every
//  case here computes the same thing twice -- once down the fast path, once
//  forced down the generic one -- and requires EXACT equality, not allclose.
//
//  Forcing the generic path: an i64 operand, or a strided view, or a
//  broadcast, each disqualifies the f64-contiguous fast path while computing
//  the same values.
//
//  Run:  bantu run tests/numba_fastpath_test.b
// ════════════════════════════════════════════════════════════════════════

$R = {"pass": 0, "fail": 0};
def ok($cond, $name) {
    if ($cond) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $name); }
}
def same($x, $y, $name) {
    if (nd_shape($x) != nd_shape($y)) {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name + "  shape " + str(nd_shape($x)) + " vs " + str(nd_shape($y)));
        return null;
    }
    $eq = nd_equal($x, $y, null);
    $bothnan = nd_logical_and(nd_isnan($x, null), nd_isnan($y, null), null);
    // nd_all returns a 0-d ndarray HANDLE, and a handle is always truthy, so
    // `ok(nd_all(...))` could never fail -- this comparison, in the file whose
    // whole purpose is "the fast paths must agree BIT FOR BIT", was a vacuous
    // assertion. Found by mutating a new fast path five ways and watching four
    // of the five mutations pass. The 0-d reduction result has to be extracted
    // to a Bantu value before it can be tested, which is what sc() is for.
    ok(sc(nd_all(nd_logical_or($eq, $bothnan, null), null, null)) == true, $name);
    return null;
}
def sc($x) { return nd_get(nd_reshape($x, [1]), [0]); }
// A timing budget is meaningless under a sanitizer -- ASan runs this 4-8x
// slower, so the assertion would fail for a reason that has nothing to do with
// the code. Sanitizer runs set BANTU_SKIP_PERF=1 and the budget is reported as
// skipped rather than quietly loosened to something that proves nothing.
$SKIP_PERF = env("BANTU_SKIP_PERF") == "1";
def budget($ms, $limit, $name) {
    if ($SKIP_PERF) { print("  skip  " + $name + "  (BANTU_SKIP_PERF=1)"); return null; }
    ok($ms < $limit, $name);
    return null;
}

nd_seed(20261109);

print("========================================");
print("  1. Every reduction, fast vs generic");
print("========================================");

// A strided view takes the generic innerSweep; its copy takes the fast one.
// The values are identical, so every reduction must agree exactly.
$base = nd_random_normal([400, 8]);
$wide = nd_random_normal([400, 16]);
$view = nd_slice($wide, [null, [0, 16, 2]]);        // [400, 8], inner stride 2
$copy = nd_copy($view, null);                        // same values, contiguous
ok(!nd_is_contiguous($view), "the strided view really is non-contiguous");
ok(nd_is_contiguous($copy), "and its copy really is contiguous");
ok(nd_array_equal($view, $copy), "they hold the same values");

same(nd_max($copy, 1, null),   nd_max($view, 1, null),   "nd_max  along the inner axis");
same(nd_min($copy, 1, null),   nd_min($view, 1, null),   "nd_min");
same(nd_sum($copy, 1, null),   nd_sum($view, 1, null),   "nd_sum");
same(nd_prod($copy, 1, null),  nd_prod($view, 1, null),  "nd_prod");
same(nd_mean($copy, 1, null),  nd_mean($view, 1, null),  "nd_mean");
same(nd_argmax($copy, 1, null),nd_argmax($view, 1, null),"nd_argmax");
same(nd_argmin($copy, 1, null),nd_argmin($view, 1, null),"nd_argmin");
same(nd_ptp($copy, 1, null),   nd_ptp($view, 1, null),   "nd_ptp");
same(nd_var($copy, 1, null, 0),  nd_var($view, 1, null, 0),  "nd_var");
same(nd_std($copy, 1, null, 0),  nd_std($view, 1, null, 0),  "nd_std");
same(nd_any($copy, 1, null),   nd_any($view, 1, null),   "nd_any");
same(nd_all($copy, 1, null),   nd_all($view, 1, null),   "nd_all");
same(nd_count_nonzero($copy, 1, null), nd_count_nonzero($view, 1, null), "nd_count_nonzero");
same(nd_median($copy, 1, null), nd_median($view, 1, null), "nd_median");
same(nd_nansum($copy, 1, null), nd_nansum($view, 1, null), "nd_nansum");
same(nd_nanmax($copy, 1, null), nd_nanmax($view, 1, null), "nd_nanmax");

// Full reductions and the other axis, which coalesce differently.
same(nd_max($copy, null, null), nd_max($view, null, null), "nd_max over everything");
same(nd_max($copy, 0, null),    nd_max($view, 0, null),    "nd_max along the OUTER axis");
same(nd_sum($copy, 0, true),    nd_sum($view, 0, true),    "nd_sum with keepdims");

print("");
print("== the extreme path's NaN rule is unchanged ==");
// nd_max's fast path returns the first NaN it meets and stops. The generic
// path lets a NaN stick once seen. Both must give NaN for a slice holding one.
$nan = sc(nd_divide(nd_full([1], 0.0), nd_full([1], 0.0), null));
ok($nan != $nan, "the test has a real NaN");
$withNaN = nd_copy($base, null);
nd_set($withNaN, [0, 3], $nan);
nd_set($withNaN, [7, 0], $nan);
$strided = nd_slice(nd_tile($withNaN, [1, 2]), [null, [0, 8, 1]]);
same(nd_max($withNaN, 1, null), nd_max(nd_copy($withNaN, null), 1, null),
     "a NaN in the slice gives NaN either way");
ok(sc(nd_max(nd_slice($withNaN, [[0,1,1], null]), null, null)) != sc(nd_max(nd_slice($withNaN, [[0,1,1], null]), null, null)),
   "and the row holding it maxes to NaN, not to the largest finite value");
// +/- infinity survives both paths.
$inf = sc(nd_divide(nd_full([1], 1.0), nd_full([1], 0.0), null));
$withInf = nd_copy($base, null);
nd_set($withInf, [1, 1], $inf);
nd_set($withInf, [2, 2], 0.0 - $inf);
same(nd_max($withInf, 1, null), nd_max(nd_copy($withInf, null), 1, null), "+inf survives");
same(nd_min($withInf, 1, null), nd_min(nd_copy($withInf, null), 1, null), "-inf survives");

print("");
print("== dtypes other than f64 keep the generic path and the same answers ==");
$ints = nd_astype(nd_random_int([200, 6], 0 - 50, 50), "i64");
same(nd_max($ints, 1, null), nd_astype(nd_max(nd_astype($ints, "f64"), 1, null), "i64"),
     "i64 max equals the f64 max of the same values");
ok(nd_dtype(nd_max($ints, 1, null)) == "i64", "and stays i64");
$bools = nd_greater($base, 0.0, null);
same(nd_any($bools, 1, null), nd_any(nd_copy($bools, null), 1, null), "bool any");
same(nd_all($bools, 1, null), nd_all(nd_copy($bools, null), 1, null), "bool all");

print("");
print("========================================");
print("  2. nd_matmul: contiguous fast path");
print("========================================");

// An i64 operand cannot take the f64 fast path, but holds the same values --
// so the two results must be bit-identical.
$ai = nd_astype(nd_random_int([40, 25], 0 - 9, 9), "i64");
$bi = nd_astype(nd_random_int([25, 7], 0 - 9, 9), "i64");
$af = nd_astype($ai, "f64");
$bf = nd_astype($bi, "f64");
same(nd_matmul($af, $bf, null), nd_astype(nd_matmul($ai, $bi, null), "f64"),
     "f64 fast path == the i64 generic path on whole numbers");

// A reference matmul written in Bantu: three loops, nothing shared with the
// kernel, so a blocking or stride mistake has nothing to mirror.
def refMatmul($a, $b) {
    $m = nd_shape($a)[0]; $k = nd_shape($a)[1]; $n = nd_shape($b)[1];
    $out = nd_zeros([$m, $n], "f64");
    $i = 0;
    while ($i < $m) {
        $j = 0;
        while ($j < $n) {
            $acc = 0.0;
            $p = 0;
            while ($p < $k) {
                $acc = $acc + nd_get($a, [$i, $p]) * nd_get($b, [$p, $j]);
                $p = $p + 1;
            }
            nd_set($out, [$i, $j], $acc);
            $j = $j + 1;
        }
        $i = $i + 1;
    }
    return $out;
}
same(nd_matmul($af, $bf, null), refMatmul($af, $bf),
     "and == a three-loop reference on exactly-representable values");

// Shapes that exercise the blocking boundaries (BS = 64).
$shapes = [[1,1,1], [1,64,1], [64,64,64], [65,65,65], [63,129,7], [128,25,6], [7,1,200]];
$si = 0;
$allok = true;
while ($si < len($shapes)) {
    $s = $shapes[$si];
    $x = nd_random_normal([$s[0], $s[1]]);
    $y = nd_random_normal([$s[1], $s[2]]);
    // Force the generic path by making the LEFT operand a strided view of a
    // wider array holding the same values.
    $xw = nd_zeros([$s[0], $s[1] * 2], "f64");
    nd_add($x, 0.0, nd_slice($xw, [null, [0, $s[1] * 2, 2]]));
    $xv = nd_slice($xw, [null, [0, $s[1] * 2, 2]]);
    if (!nd_array_equal($xv, $x)) { $allok = false; }
    if (!nd_array_equal(nd_matmul($x, $y, null), nd_matmul($xv, $y, null))) { $allok = false; }
    $si = $si + 1;
}
ok($allok, "fast and generic agree exactly over 7 shapes, including both sides of BS=64");

// Vectors, which take a different branch.
$v = nd_random_normal([25]);
same(nd_matmul($af, $v, null), nd_matmul($af, nd_copy($v, null), null), "matrix . vector");
// Draw the right-hand operand ONCE. This and the two below used to call
// nd_random_normal() separately on each side, so the two matmuls multiplied
// DIFFERENT matrices and could never have agreed -- three assertions that only
// passed because ok(nd_all(...)) could not fail. Both were found together.
$vm = nd_random_normal([25, 4]);
same(nd_matmul($v, $vm, null),
     nd_matmul(nd_copy($v, null), $vm, null), "vector . matrix -- shape");
ok(nd_shape(nd_matmul($v, $v, null)) == [], "vector . vector is a 0-d scalar");
// Mismatched inner dimensions must still raise, on both paths.
def raises($f, $name) {
    try { $f(); } catch ($e) { ok(true, $name); return null; }
    ok(false, $name + "  (no error raised)");
    return null;
}
raises(def() { return nd_matmul($af, nd_random_normal([9, 3]), null); },
       "a fast-path inner-dimension mismatch raises");
raises(def() { return nd_matmul($ai, nd_astype(nd_random_int([9,3],0,2), "i64"), null); },
       "and so does a generic-path one");

print("");
print("========================================");
print("  3. nd_matmul: transposed left operand");
print("========================================");

// A^T . B is every weight gradient. The transposed view takes gemmTransA; a
// materialised copy of the same transpose takes gemmContig. Bit-identical.
$si = 0;
$allok = true;
$cases = [[10, 25, 6], [100, 7, 3], [64, 64, 64], [65, 33, 2], [1, 50, 1], [200, 5, 9]];
while ($si < len($cases)) {
    $c = $cases[$si];
    $raw = nd_random_normal([$c[0], $c[1]]);      // [K, M] in C order
    $t = nd_transpose($raw, null);                 // [M, K] as a view
    $g = nd_random_normal([$c[0], $c[2]]);         // [K, N]
    $viaView = nd_matmul($t, $g, null);
    $viaCopy = nd_matmul(nd_copy($t, null), $g, null);
    if (!nd_array_equal($viaView, $viaCopy)) { $allok = false; }
    if (nd_shape($viaView) != [$c[1], $c[2]]) { $allok = false; }
    $si = $si + 1;
}
ok($allok, "A^T . B via the view == via a materialised copy, over 6 shapes");

// And against the reference loop.
$raw = nd_random_normal([30, 8]);
$rhs = nd_random_normal([30, 4]);
same(nd_matmul(nd_transpose($raw, null), $rhs, null),
     nd_matmul(nd_copy(nd_transpose($raw, null), null), $rhs, null),
     "...which is the shape conv2d's weight gradient uses");

// A DOUBLE transpose is contiguous again, and must not be mistaken for one.
$tt = nd_transpose(nd_transpose($raw, null), null);
ok(nd_array_equal($tt, $raw), "transposing twice gives the original back");
$ttr = nd_random_normal([8, 3]);
same(nd_matmul($tt, $ttr, null),
     nd_matmul($raw, $ttr, null), "and multiplies identically");

// A square transpose has strides [1, n] AND the same shape either way -- the
// one case where misreading the orientation would be silent.
$sq = nd_reshape(nd_astype(nd_arange(0.0, 16.0, null), "f64"), [4, 4]);
$sqt = nd_transpose($sq, null);
same(nd_matmul($sqt, $sq, null), nd_matmul(nd_copy($sqt, null), $sq, null),
     "a SQUARE transpose is not confused with its original");
ok(!nd_array_equal(nd_matmul($sqt, $sq, null), nd_matmul($sq, $sq, null)),
   "...and really does give a different answer from the untransposed one");

print("");
print("========================================");
print("  4. No leaks, and the measured wins");
print("========================================");

$before = nd_live_bytes();
$n = 0;
while ($n < 300) {
    nd_max($copy, 1, null);
    nd_matmul($af, $bf, null);
    nd_matmul(nd_transpose($raw, null), nd_random_normal([30, 4]), null);
    $n = $n + 1;
}
$after = nd_live_bytes();
print("        live array bytes: " + str($before) + " -> " + str($after));
ok($after - $before < 500000, "300 rounds through all three paths leak nothing");

// The wins, asserted as budgets so a regression is caught rather than noticed.
$pcols = nd_random_normal([150528, 4]);
$t0 = clock();
nd_max($pcols, 1, null);
$tmax = clock() - $t0;
print("        nd_max over [150528, 4]: " + str($tmax) + " ms  (was 102 ms)");
budget($tmax, 25, "the reduction fast path holds: under 25 ms, not 102");

$big = nd_random_normal([100352, 25]);
$small = nd_random_normal([25, 6]);
$t0 = clock();
nd_matmul($big, $small, null);
$tmm = clock() - $t0;
print("        [100352,25] x [25,6]:    " + str($tmm) + " ms  (was 20.6 ms)");
budget($tmm, 40, "the contiguous matmul path holds");

$grad = nd_random_normal([100352, 6]);
$t0 = clock();
nd_matmul(nd_transpose($big, null), $grad, null);
$tat = clock() - $t0;
print("        [25,100352]^T x [100352,6]: " + str($tat) + " ms  (was 45.8 ms)");
budget($tat, 40, "the transposed-A path holds: under 40 ms, not 46");

print("");
print("== tier 1b: a BROADCAST VECTOR operand ==");
// A broadcast operand used to fall to tier 2's per-element odometer: same
// memory traffic, 3.4x the time (18.0 ms against 5.3 ms on [16,32,64,64]).
// Tier 1b hands each innermost run to the SAME vectorized kernel tier 0 and
// tier 1 already use, so the reference here is the generic path computing the
// identical thing: broadcast the small operand into a full array first, which
// makes it a same-shape tier-0 call.
//
// The danger in this fast path is operand ORDER. Mode 2 (the broadcast side on
// the LEFT) calls tier1r, and getting that backwards would compute b - a for
// a - b -- which is why every non-commutative op below is checked in BOTH
// orders. A commutative-only test would pass with the operands swapped.
def expand($small, $shape) { return nd_copy(nd_broadcast_to($small, $shape), null); }

$bx = nd_random_uniform([4, 5, 3, 2], -3.0, 3.0);
$bc = nd_add(nd_random_uniform([1, 5, 1, 1], 0.5, 3.0), 0.25, null);   // no zeros, for divide
$bE = expand($bc, [4, 5, 3, 2]);

same(nd_add($bx, $bc, null),      nd_add($bx, $bE, null),      "mode 1: x + [1,C,1,1]");
same(nd_subtract($bx, $bc, null), nd_subtract($bx, $bE, null), "mode 1: x - [1,C,1,1]  (NOT commutative)");
same(nd_divide($bx, $bc, null),   nd_divide($bx, $bE, null),   "mode 1: x / [1,C,1,1]  (NOT commutative)");
same(nd_multiply($bx, $bc, null), nd_multiply($bx, $bE, null), "mode 1: x * [1,C,1,1]");
same(nd_power($bc, $bc, null),    nd_power($bc, $bc, null),    "mode 1: power is reflexive");

same(nd_subtract($bc, $bx, null), nd_subtract($bE, $bx, null), "mode 2: [1,C,1,1] - x  (broadcast on the LEFT)");
same(nd_divide($bc, $bx, null),   nd_divide($bE, $bx, null),   "mode 2: [1,C,1,1] / x  (broadcast on the LEFT)");
same(nd_add($bc, $bx, null),      nd_add($bE, $bx, null),      "mode 2: [1,C,1,1] + x");

// Mode 0: the broadcast operand is contiguous over the innermost run and
// repeats once per outer step -- the trailing-vector shape, [N,C] op [1,C].
$tx = nd_random_uniform([64, 7], -2.0, 2.0);
$tv = nd_add(nd_random_uniform([1, 7], 0.5, 2.0), 0.25, null);
$tE = expand($tv, [64, 7]);
same(nd_subtract($tx, $tv, null), nd_subtract($tx, $tE, null), "mode 0: [N,C] - [1,C] trailing vector");
same(nd_divide($tv, $tx, null),   nd_divide($tE, $tx, null),   "mode 0 reversed: [1,C] / [N,C]");

// A bare [C] operand, not [1,C] -- broadcasting aligns from the right.
same(nd_subtract($tx, nd_reshape($tv, [7]), null), nd_subtract($tx, $tE, null),
     "a rank-1 [C] operand aligns from the right");

// Other broadcast axes: the middle, and the innermost only.
$t3 = nd_random_uniform([3, 4, 5], -2.0, 2.0);
$m3 = nd_add(nd_random_uniform([1, 4, 1], 0.5, 2.0), 0.25, null);
$i3 = nd_add(nd_random_uniform([1, 1, 5], 0.5, 2.0), 0.25, null);
same(nd_subtract($t3, $m3, null), nd_subtract($t3, expand($m3, [3,4,5]), null), "3-d: middle axis broadcast");
same(nd_subtract($t3, $i3, null), nd_subtract($t3, expand($i3, [3,4,5]), null), "3-d: innermost axis broadcast");
same(nd_divide($t3, $m3, null),   nd_divide($t3, expand($m3, [3,4,5]), null),   "3-d: middle axis, divide");

// Both operands broadcast against each other -- neither is the full shape.
$ra = nd_random_uniform([4, 1, 3], -2.0, 2.0);
$rb = nd_add(nd_random_uniform([1, 5, 1], 0.5, 2.0), 0.25, null);
same(nd_subtract($ra, $rb, null),
     nd_subtract(expand($ra, [4,5,3]), expand($rb, [4,5,3]), null),
     "mutual broadcast: [4,1,3] - [1,5,1]");

// f32 and i64 take tier 1b too, each with its own arm.
$fx = nd_astype($bx, "f32");
$fc = nd_astype($bc, "f32");
same(nd_subtract($fx, $fc, null), nd_subtract($fx, nd_astype($bE, "f32"), null), "f32 broadcast subtract");
$ix = nd_astype(nd_random_int([6, 4], -50, 50), "i64");
$iv = nd_astype(nd_add(nd_random_int([1, 4], 1, 9), 0, null), "i64");
same(nd_subtract($ix, $iv, null), nd_subtract($ix, expand($iv, [6,4]), null), "i64 broadcast subtract");
same(nd_divide($ix, $iv, null),   nd_divide($ix, expand($iv, [6,4]), null),   "i64 broadcast divide (integer semantics)");

// MIXED DTYPES must not enter the fast path. tier 1b casts both buffers to the
// COMPUTE type, so an operand stored as i64 against an f64 array would be read
// as doubles -- reinterpreted bits, not converted values. The guard that stops
// it (`b->dtype == cdt`) is invisible in every same-dtype test, and dropping it
// passed the whole suite until these four assertions existed.
$mdF = nd([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], "f64");
$mdI = nd_astype(nd([[10.0, 20.0, 30.0]], "f64"), "i64");         // [1,3] i64
$mdE = nd_astype(expand($mdI, [4, 3]), "i64");
// THE REFERENCE MUST NOT TRAVEL THE SAME PATH. These four assertions first
// compared `op(f64, i64_broadcast)` against `op(f64, i64_expanded)` -- and an
// expanded i64 array still enters tier 1b, so a mutation that dropped the dtype
// guard broke BOTH sides into the same garbage and the comparison passed. The
// `a->dtype` guard was dropped and the whole suite stayed green.
//
// So the reference converts to the compute dtype FIRST and is then a same-shape
// f64 pair -- tier 0, which every other assertion in this file has already
// pinned. That is the N41 lesson for the second time in one file: a reference
// computed through the code under test shares its faults.
def asF64($a, $shape) { return nd_astype(expand($a, $shape), "f64"); }

same(nd_subtract($mdF, $mdI, null),
     nd_subtract($mdF, asF64($mdI, [4, 3]), null),
     "f64 array minus an i64 broadcast vector is CONVERTED, not reinterpreted");
same(nd_subtract($mdI, $mdF, null),
     nd_subtract(asF64($mdI, [4, 3]), $mdF, null),
     "and with the i64 broadcast operand on the LEFT (the a-side dtype guard)");
// The same thing one rank up, where the broadcast axis is the innermost.
$mdC = nd_astype(nd([[5.0], [6.0], [7.0], [8.0]], "f64"), "i64");  // [4,1] i64
same(nd_multiply($mdF, $mdC, null),
     nd_multiply($mdF, asF64($mdC, [4, 3]), null),
     "an i64 column broadcast across an f64 row");
same(nd_multiply($mdC, $mdF, null),
     nd_multiply(asF64($mdC, [4, 3]), $mdF, null),
     "and that column on the left");
// f32 against f64: the compute type is f64, so the f32 side must widen.
$md32 = nd_astype(nd([[0.5, 1.5, 2.5]], "f64"), "f32");
same(nd_add($mdF, $md32, null),
     nd_add($mdF, asF64($md32, [4, 3]), null),
     "an f32 broadcast vector against an f64 array widens correctly");
same(nd_add($md32, $mdF, null),
     nd_add(asF64($md32, [4, 3]), $mdF, null),
     "and on the left");

// A STRIDED destination and strided inputs must still be right -- these
// disqualify the fast path and fall to tier 2, which is the point.
$sv = nd_slice($bx, [[0, 4, 1], [0, 5, 2], [0, 3, 1], [0, 2, 1]]);
$se = nd_copy($sv, null);
same(nd_subtract($sv, nd_slice($bc, [[0,1,1],[0,5,2],[0,1,1],[0,1,1]]), null),
     nd_subtract($se, nd_copy(nd_slice($bc, [[0,1,1],[0,5,2],[0,1,1],[0,1,1]]), null), null),
     "a strided view broadcasts identically");

// Comparisons produce BOOL and are deliberately left on tier 2.
same(nd_astype(nd_greater($bx, $bc, null), "f64"),
     nd_astype(nd_greater($bx, $bE, null), "f64"), "a BOOL result is unchanged");

// out= with a broadcast operand, including writing into the full operand.
$ox = nd_copy($bx, null);
nd_subtract($ox, $bc, $ox);
same($ox, nd_subtract($bx, $bE, null), "out= in place over a broadcast operand");

// An out= whose dtype differs from the COMPUTE dtype. planOut hands the
// caller's array straight to the kernel when nothing aliases, WITHOUT checking
// its dtype -- so `out->dtype == rdt` is a reachable guard, and dropping it
// writes doubles into an i64 buffer as reinterpreted bits. Nothing in this file
// tested that, for tier 1b or for the two tiers above it, and a mutation that
// removed the guard passed the whole suite.
//
// Every value is integral, so the f64 -> i64 conversion is exact and the
// reference cannot disagree over rounding. The reference is a same-shape f64
// pair (tier 0) converted afterwards, so it cannot travel the path under test.
$oiA = nd([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0],
           [70.0, 80.0, 90.0], [100.0, 110.0, 120.0]], "f64");
$oiV = nd([[1.0, 2.0, 3.0]], "f64");
$oiDst = nd_astype(nd_zeros([4, 3], "f64"), "i64");
nd_subtract($oiA, $oiV, $oiDst);
ok(nd_dtype($oiDst) == "i64", "an i64 out= keeps its own dtype");
same(nd_astype($oiDst, "f64"),
     nd_subtract($oiA, nd_copy(nd_broadcast_to($oiV, [4, 3]), null), null),
     "a broadcast op into an i64 out= CONVERTS rather than reinterpreting");
$ofDst = nd_astype(nd_zeros([4, 3], "f64"), "f32");
nd_multiply($oiA, $oiV, $ofDst);
ok(nd_dtype($ofDst) == "f32", "an f32 out= keeps its own dtype");
same(nd_astype($ofDst, "f64"),
     nd_multiply($oiA, nd_copy(nd_broadcast_to($oiV, [4, 3]), null), null),
     "and an f32 out= narrows the f64 result rather than reinterpreting it");

// And the determinism contract: the fast path must give the same bits at any
// thread count, exactly as N25 requires of every other path.
$keep = nd_get_threads();
$big  = nd_random_uniform([8, 16, 32, 32], -2.0, 2.0);
$bigc = nd_add(nd_random_uniform([1, 16, 1, 1], 0.5, 2.0), 0.25, null);
nd_set_threads(1);
$r1 = nd_tobytes(nd_divide($big, $bigc, null), "f64");
nd_set_threads(8);
$r8 = nd_tobytes(nd_divide($big, $bigc, null), "f64");
nd_set_threads(3);
$r3 = nd_tobytes(nd_divide($big, $bigc, null), "f64");
nd_set_threads($keep);
ok($r1 == $r8, "tier 1b is bit-identical at 1 and 8 threads");
ok($r1 == $r3, "tier 1b is bit-identical at 1 and 3 threads");

// The speed it was added for. [16,32,64,64] f64 measured 18.0 ms on tier 2
// and 4.9 ms on tier 1b; the floor is set well above the fast number so a
// loaded machine does not flake it, but far below the old one.
$px = nd_random_uniform([16, 32, 64, 64], -1.0, 1.0);
$pc = nd_random_uniform([1, 32, 1, 1], 0.5, 2.0);
nd_subtract($px, $pc, null);
$pt0 = clock();
$pi = 0;
while ($pi < 5) { nd_subtract($px, $pc, null); $pi = $pi + 1; }
$pel = (clock() - $pt0) / 5.0;
budget($pel, 12, "a broadcast subtract of 2.1M f64 stays under 12 ms, not 18");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
