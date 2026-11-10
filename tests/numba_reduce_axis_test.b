// ════════════════════════════════════════════════════════════════════════
//  numba_reduce_axis_test.b — reducing a STRIDED axis gives the same answer
//  as reducing a contiguous one, after the loop inversion that made it 13-17x
//  faster.
//
//  `nd_sum($m, 0)` on a 2048x2048 array took 66 ms while axis 1 took 2.4 ms --
//  a 27x asymmetry for the same data and the same number of adds, because the
//  loops were nested the wrong way round: one output at a time, each walking a
//  column with stride 2048, so every load was its own cache line and the array
//  was re-streamed once per output. The fix streams the input in memory order
//  with one accumulator per output.
//
//  THE WHOLE POINT IS THAT NOTHING CHANGED. Each output still sees its
//  elements in the same order, so:
//
//    * `Pairwise` builds the same tree with the same roundings (sum, mean);
//    * a running product multiplies in the same order (prod);
//    * first-NaN-wins still returns the same NaN (min, max).
//
//  So every assertion here is EXACT equality against the old code path -- and
//  the old path is still reachable, which is what makes that testable: the
//  inverted loops need at least 8 outputs to pay for themselves, so a narrow
//  column slice of the same array takes the old path while the full array
//  takes the new one. Same data, same order, two implementations.
//
//  Run:  bantu run tests/numba_reduce_axis_test.b
// ════════════════════════════════════════════════════════════════════════

$R = {"pass": 0, "fail": 0};
def ok($c, $n) {
    if ($c) { $R.pass = $R.pass + 1; print("  ok    " + $n); }
    else    { $R.fail = $R.fail + 1; print("  FAIL  " + $n); }
}
// Exact, and NaN-aware: nd_equal says false for NaN == NaN, and a propagated
// NaN is exactly what several of these assertions are about.
def same($x, $y, $n) {
    if (nd_shape($x) != nd_shape($y)) {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $n + "  shape " + str(nd_shape($x)) + " vs " + str(nd_shape($y)));
        return null;
    }
    ok(nd_get(nd_all(nd_logical_or(nd_equal($x, $y, null),
        nd_logical_and(nd_isnan($x, null), nd_isnan($y, null), null), null), null, null), 0), $n);
    return null;
}

print("");
print("  numba strided-axis reductions (D3c)");
print("  ════════════════════════════════════════");
nd_seed(9091);
$nan = nd_get(nd_divide(nd_zeros([1], "f64"), 0.0, null), 0);
$inf = nd_get(nd_divide(nd_ones([1], "f64"), 0.0, null), 0);

// ── 1. sum and mean, against the old path ───────────────────────────────────
// Shapes chosen around the group-of-eight the sum path injects at level 3:
// 2048 rows is 256 whole groups, 2047 leaves a ragged tail of seven, 129 is
// sixteen groups plus one, and 13 has no whole group at all.
print("");
print("-- sum and mean: new path == old path, bit for bit --");
$shapes = [[2048, 2048], [2047, 2048], [129, 64], [128, 32], [13, 2048], [1000, 1000]];
$si = 0;
while ($si < len($shapes)) {
    $sh = $shapes[$si];
    $m  = nd_random_uniform($sh, -1.0, 1.0);
    $tag = str($sh);
    $cols = 7;
    if ($sh[1] < 7) { $cols = $sh[1]; }
    $narrow = nd_slice($m, [null, [0, $cols, 1]]);        // fewer than 8 outputs: old path
    same(nd_slice(nd_sum($m, 0, null, null), [[0, $cols, 1]]),
         nd_sum($narrow, 0, null, null), "sum axis 0 " + $tag);
    same(nd_slice(nd_mean($m, 0, null), [[0, $cols, 1]]),
         nd_mean($narrow, 0, null), "mean axis 0 " + $tag);
    // a single column also takes the old path, and must agree element-wise
    $one = nd_sum(nd_slice($m, [null, [5, 6, 1]]), 0, null, null);
    ok(nd_get($one, 0) == nd_get(nd_sum($m, 0, null, null), 5), "sum column 5 " + $tag);
    ok(nd_shape(nd_sum($m, 0, null, null)) == [$sh[1]], "sum axis 0 shape " + $tag);
    ok(nd_shape(nd_sum($m, 0, true, null)) == [1, $sh[1]], "keepdims " + $tag);
    $si = $si + 1;
}

// ── 2. the simple accumulators ──────────────────────────────────────────────
print("");
print("-- min, max, prod, any, all: new path == old path, every dtype --");
$dts = ["f64", "f32", "i64"];
$di = 0;
while ($di < len($dts)) {
    $dt = $dts[$di];
    $m   = nd_astype(nd_multiply(nd_random_uniform([300, 97], -1.0, 1.0), 100.0, null), $dt);
    $nar = nd_slice($m, [null, [0, 7, 1]]);
    same(nd_slice(nd_max($m, 0, null), [[0, 7, 1]]),  nd_max($nar, 0, null),  "max " + $dt);
    same(nd_slice(nd_min($m, 0, null), [[0, 7, 1]]),  nd_min($nar, 0, null),  "min " + $dt);
    same(nd_slice(nd_prod($m, 0, null, null), [[0, 7, 1]]),
         nd_prod($nar, 0, null, null), "prod " + $dt);
    same(nd_slice(nd_any($m, 0, null), [[0, 7, 1]]),  nd_any($nar, 0, null),  "any " + $dt);
    same(nd_slice(nd_all($m, 0, null), [[0, 7, 1]]),  nd_all($nar, 0, null),  "all " + $dt);
    ok(nd_dtype(nd_max($m, 0, null)) == nd_dtype(nd_max($nar, 0, null)), "max dtype " + $dt);
    ok(nd_dtype(nd_any($m, 0, null)) == "bool", "any is bool " + $dt);
    $di = $di + 1;
}
$b = nd_greater(nd_random_uniform([200, 50], -1.0, 1.0), 0.0, null);
same(nd_slice(nd_any($b, 0, null), [[0, 7, 1]]),
     nd_any(nd_slice($b, [null, [0, 7, 1]]), 0, null), "any on a bool array");
same(nd_slice(nd_all($b, 0, null), [[0, 7, 1]]),
     nd_all(nd_slice($b, [null, [0, 7, 1]]), 0, null), "all on a bool array");

// ── 3. NaN, inf and the first-NaN-wins rule ─────────────────────────────────
// The rule is "the first NaN in index order wins, with its payload". Inverting
// the loops must not turn that into "whichever NaN the tile saw first", so the
// NaNs here sit in the first row, in the middle, and in the last row.
print("");
print("-- NaN placement survives the inversion --");
$z = nd_full([40, 24], 2.0, "f64");
nd_set($z, [0, 3], $nan);
nd_set($z, [17, 5], $nan);
nd_set($z, [39, 7], $nan);
nd_set($z, [12, 9], $inf);
nd_set($z, [4, 11], 0.0 - $inf);
same(nd_slice(nd_max($z, 0, null), [[0, 7, 1]]),
     nd_max(nd_slice($z, [null, [0, 7, 1]]), 0, null), "max with NaNs in rows 0, 17 and 39");
same(nd_slice(nd_min($z, 0, null), [[0, 7, 1]]),
     nd_min(nd_slice($z, [null, [0, 7, 1]]), 0, null), "min with the same NaNs");
ok(nd_get(nd_isnan(nd_max($z, 0, null), null), 3), "a NaN in the FIRST row propagates");
ok(nd_get(nd_isnan(nd_max($z, 0, null), null), 5), "a NaN in the middle propagates");
ok(nd_get(nd_isnan(nd_max($z, 0, null), null), 7), "a NaN in the LAST row propagates");
ok(nd_get(nd_max($z, 0, null), 9) == $inf, "inf is the max");
ok(nd_get(nd_min($z, 0, null), 11) == 0.0 - $inf, "-inf is the min");
// and the contiguous extreme path, which was rewritten in the same increment:
// eight accumulators plus a NaN flag, with a rescan for the first NaN.
$cz = nd_full([24, 1000], 2.0, "f64");
nd_set($cz, [3, 0], $nan);              // first element of the row
nd_set($cz, [5, 999], $nan);            // last element
nd_set($cz, [7, 517], $nan);            // in the middle, past the blocked part
nd_set($cz, [9, 3], $inf);
$cm = nd_max($cz, 1, null);
ok(nd_get(nd_isnan($cm, null), 3), "contiguous max: a NaN at the start propagates");
ok(nd_get(nd_isnan($cm, null), 5), "contiguous max: a NaN at the end propagates");
ok(nd_get(nd_isnan($cm, null), 7), "contiguous max: a NaN in the middle propagates");
ok(nd_get($cm, 9) == $inf, "contiguous max: inf wins");
ok(nd_get($cm, 0) == 2.0, "contiguous max: a row with no NaN is unaffected");
ok(nd_get(nd_min($cz, 1, null), 0) == 2.0, "contiguous min agrees");
// every length around the block of eight, so the tail is covered
$k = 1;
$tailok = 1;
while ($k <= 20) {
    $row = nd_random_uniform([3, $k], -1.0, 1.0);
    if (nd_get(nd_max($row, 1, null), 0) != nd_get(nd_max(nd_slice($row, [[0, 1, 1], null]), null, null), 0)) {
        $tailok = 0;
    }
    $k = $k + 1;
}
ok($tailok == 1, "contiguous max agrees with the full reduction for every length 1..20");

// ── 4. more than two dimensions, and strided views ──────────────────────────
print("");
print("-- 3-D, strided views, reversed axes --");
$c = nd_random_uniform([9, 11, 37], -1.0, 1.0);
for $ax in [0, 1] {
    same(nd_slice(nd_sum($c, $ax, null, null), [null, [0, 4, 1]]),
         nd_sum(nd_slice($c, [null, null, [0, 4, 1]]), $ax, null, null),
         "3-D sum axis " + str($ax));
    same(nd_slice(nd_max($c, $ax, null), [null, [0, 4, 1]]),
         nd_max(nd_slice($c, [null, null, [0, 4, 1]]), $ax, null),
         "3-D max axis " + str($ax));
}
$v = nd_slice(nd_random_uniform([400, 400], -1.0, 1.0), [[10, 300, 3], [5, 390, 2]]);
same(nd_slice(nd_sum($v, 0, null, null), [[0, 7, 1]]),
     nd_sum(nd_slice($v, [null, [0, 7, 1]]), 0, null, null), "a strided view's sum");
same(nd_slice(nd_max($v, 0, null), [[0, 7, 1]]),
     nd_max(nd_slice($v, [null, [0, 7, 1]]), 0, null), "a strided view's max");
same(nd_slice(nd_prod($v, 0, null, null), [[0, 7, 1]]),
     nd_prod(nd_slice($v, [null, [0, 7, 1]]), 0, null, null), "a strided view's prod");
// a reversed axis has a negative stride, which the new path declines; it must
// still produce the old answer through the old code.
$rev = nd_slice(nd_random_uniform([100, 40], -1.0, 1.0), [[99, 0 - 1, 0 - 1], null]);
same(nd_slice(nd_sum($rev, 0, null, null), [[0, 7, 1]]),
     nd_sum(nd_slice($rev, [null, [0, 7, 1]]), 0, null, null), "a reversed axis still agrees");

// ── 5. reducing everything, and the degenerate shapes ───────────────────────
print("");
print("-- whole-array and degenerate shapes --");
$w = nd_random_uniform([64, 64], -1.0, 1.0);
ok(nd_get(nd_sum($w, null, null, null), 0) == nd_get(nd_sum(nd_reshape($w, [4096]), null, null, null), 0),
   "reducing every axis is the flat sum");
ok(nd_shape(nd_sum(nd_random_uniform([1, 50], -1.0, 1.0), 0, null, null)) == [50],
   "one row");
ok(nd_shape(nd_sum(nd_random_uniform([50, 1], -1.0, 1.0), 0, null, null)) == [1],
   "one column");
ok(nd_get(nd_sum(nd_zeros([0, 9], "f64"), 0, null, null), 0) == 0.0,
   "an empty axis sums to the identity");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
