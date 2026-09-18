// ════════════════════════════════════════════════════════════════════════
//  numba_reduce_test.b — reductions with axis/keepdims, scans, sorting,
//  search and indexing.
//
//  The gate that matters most here is ACCURACY, and it is a real gate rather
//  than a formality: summing 10,000,000 copies of 0.1 with a naive
//  accumulator loses about n*eps, which lands around 2e-12 relative -- so a
//  naive implementation fails the assertion below BY CONSTRUCTION. Pairwise
//  accumulation makes the error grow as log n instead, and comes out clean.
//  People will diff numba against NumPy, which does pairwise; matching it is
//  correctness, not polish.
//
//  The second theme is empty and degenerate input. sum of nothing is 0 and
//  all of nothing is true, because those are the operations' identities --
//  but min of nothing must RAISE, since there is no identity and returning 0
//  or -inf would be a silently wrong answer rather than an error.
//
//  Run:  bantu run tests/numba_reduce_test.b
// ════════════════════════════════════════════════════════════════════════

$R = {"pass": 0, "fail": 0};

def eq($got, $want, $name) {
    if ($got == $want) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name);
        print("          got:  " + str($got));
        print("          want: " + str($want));
    }
}
def ok($cond, $name) {
    if ($cond) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $name); }
}
def eqArr($got, $want, $name) { eq(str(nd_to_list($got)), str($want), $name); }
def close($got, $want, $tol, $name) {
    $d = $got - $want;
    if ($d < 0) { $d = 0 - $d; }
    if ($d <= $tol) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name + "  got " + str($got) + " want " + str($want));
    }
}
def raises($fn, $needle, $name) {
    try {
        $fn();
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name + "  (no error raised)");
    } catch ($e) {
        $msg = str($e.message);
        if (contains($msg, $needle)) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
        else {
            $R.fail = $R.fail + 1;
            print("  FAIL  " + $name);
            print("          message did not mention \"" + $needle + "\": " + $msg);
        }
    }
}

print("========================================");
print("  numba — reductions, scans, sorting, indexing");
print("========================================");

// A (2,3) with known values, used throughout.
//   [[1, 2, 3],
//    [4, 5, 6]]
$m = nd_reshape(nd_add(nd_arange(0, 6, null), 1, null), [2, 3]);

print("");
print("-- reducing everything --");
eq(nd_get(nd_sum($m, null, null), []), 21, "sum of the whole array");
eq(nd_get(nd_prod($m, null, null), []), 720, "prod");
eq(nd_get(nd_min($m, null, null), []), 1, "min");
eq(nd_get(nd_max($m, null, null), []), 6, "max");
eq(nd_get(nd_mean($m, null, null), []), 3.5, "mean");
eq(nd_get(nd_ptp($m, null, null), []), 5, "ptp");
eq(nd_ndim(nd_sum($m, null, null)), 0, "reducing everything gives a 0-d array");

print("");
print("-- reducing one axis --");
eqArr(nd_sum($m, 0, null), [5, 7, 9],  "sum over axis 0 (down the columns)");
eqArr(nd_sum($m, 1, null), [6, 15],    "sum over axis 1 (along the rows)");
eqArr(nd_min($m, 0, null), [1, 2, 3],  "min over axis 0");
eqArr(nd_max($m, 1, null), [3, 6],     "max over axis 1");
eqArr(nd_mean($m, 1, null), [2, 5],    "mean over axis 1");
eqArr(nd_prod($m, 0, null), [4, 10, 18], "prod over axis 0");
eq(str(nd_shape(nd_sum($m, 0, null))), "[3]", "the reduced axis is removed");
eqArr(nd_sum($m, -1, null), [6, 15], "a negative axis counts from the end");

print("");
print("-- keepdims is what makes the result broadcast back --");
eq(str(nd_shape(nd_sum($m, 1, true))), "[2, 1]", "keepdims leaves the axis as extent 1");
eqArr(nd_sum($m, 1, true), [[6], [15]], "with the right values");
// The reason keepdims exists: centring each row in one line.
$centred = nd_subtract($m, nd_mean($m, 1, true), null);
eqArr($centred, [[-1, 0, 1], [-1, 0, 1]], "so a row-centring subtraction just works");

print("");
print("-- reducing several axes at once --");
$c = nd_reshape(nd_arange(0, 24, null), [2, 3, 4]);
eq(nd_get(nd_sum($c, [0, 1, 2], null), []), 276, "a list of every axis equals reducing all");
eq(str(nd_shape(nd_sum($c, [0, 2], null))), "[3]", "reducing two of three axes leaves one");
eqArr(nd_sum($c, [0, 2], null), [60, 92, 124], "and the values are right");
eq(str(nd_shape(nd_sum($c, [0, 1], true))), "[1, 1, 4]", "keepdims across several axes");
raises(def() { nd_sum($c, [0, 0], null); }, "more than once",
   "the same axis given twice is refused");
raises(def() { nd_sum($c, 7, null); }, "out of range", "an out-of-range axis is refused");
raises(def() { nd_sum($c, "x", null); }, "must be a number", "a non-numeric axis is refused");

print("");
print("-- against reference values for every axis of a (7,5,3) --");
// Built so each element is distinct and the sums are checkable by hand:
// element [i,j,k] = i*15 + j*3 + k, so the total is sum(0..104) = 5460.
$b = nd_reshape(nd_arange(0, 105, null), [7, 5, 3]);
eq(nd_get(nd_sum($b, null, null), []), 5460, "the (7,5,3) total");
eq(str(nd_shape(nd_sum($b, 0, null))), "[5, 3]", "axis 0 leaves (5,3)");
eq(str(nd_shape(nd_sum($b, 1, null))), "[7, 3]", "axis 1 leaves (7,3)");
eq(str(nd_shape(nd_sum($b, 2, null))), "[7, 5]", "axis 2 leaves (7,5)");
// Column 0 of the axis-0 sum: elements [i,0,0] = i*15, i=0..6 -> 15*21 = 315.
eq(nd_get(nd_sum($b, 0, null), [0, 0]), 315, "axis-0 sum, first element");
// Row 0 of the axis-2 sum: [0,0,k] for k=0,1,2 -> 0+1+2 = 3.
eq(nd_get(nd_sum($b, 2, null), [0, 0]), 3, "axis-2 sum, first element");
// Every axis-pair sum must equal the total when the third is then reduced.
eq(nd_get(nd_sum(nd_sum($b, 0, null), null, null), []), 5460, "axis 0 then the rest");
eq(nd_get(nd_sum(nd_sum($b, 1, null), null, null), []), 5460, "axis 1 then the rest");
eq(nd_get(nd_sum(nd_sum($b, 2, null), null, null), []), 5460, "axis 2 then the rest");
eq(nd_get(nd_sum($b, [1, 2], null), [0]), 105, "the (1,2) axis-pair, first element");

print("");
print("-- a strided input reduces the same as its contiguous copy --");
$t = nd_T($b);                                   // (3,5,7), non-contiguous
ok(nd_array_equal(nd_sum($t, 0, null), nd_sum(nd_copy($t), 0, null)),
   "a transposed array reduces identically to its copy");
$fl = nd_flip(nd_arange(0, 10, null), null);     // negative stride
eq(nd_get(nd_sum($fl, null, null), []), 45, "a negative-stride array sums correctly");

print("");
print("-- ACCURACY: the gate a naive accumulator fails by construction --");
$tenth = nd_full([10000000], 0.1, null);
$s = nd_get(nd_sum($tenth, null, null), []);
$relerr = ($s - 1000000.0) / 1000000.0;
if ($relerr < 0) { $relerr = 0 - $relerr; }
print("        sum of 10,000,000 copies of 0.1 = " + str($s));
print("        relative error = " + str($relerr));
ok($relerr < 0.000000000001, "10M x 0.1 sums to within 1e-12 relative (naive loses ~2e-12)");
$mn = nd_get(nd_mean($tenth, null, null), []);
close($mn, 0.1, 0.000000000000001, "and the mean is 0.1 to 1e-15");
$tenth = 0;

print("");
print("-- variance and standard deviation --");
$v = nd([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], null);
close(nd_get(nd_mean($v, null, null), []), 5.0, 1e-14, "mean of the textbook sample");
close(nd_get(nd_var($v, null, null), []), 4.0, 1e-12, "population variance is 4");
close(nd_get(nd_std($v, null, null), []), 2.0, 1e-12, "population sd is 2");
// ddof=1 is the sample estimator: 32/7 rather than 32/8.
close(nd_get(nd_var($v, null, null, 1), []), 4.571428571428571, 1e-12, "ddof=1 gives the sample variance");
eqArr(nd_var(nd([[1.0, 1.0], [1.0, 1.0]], null), 1, null), [0, 0], "zero variance where all equal");
// Welford's reason for existing: a huge offset destroys the naive formula.
$off = nd_add(nd([1.0, 2.0, 3.0, 4.0], null), 1000000000.0, null);
close(nd_get(nd_var($off, null, null), []), 1.25, 1e-6,
      "variance survives a 1e9 offset (the naive sum-of-squares formula does not)");

print("");
print("-- argmin / argmax --");
eq(nd_get(nd_argmax(nd([3, 9, 2], null), null, null), []), 1, "argmax");
eq(nd_get(nd_argmin(nd([3, 9, 2], null), null, null), []), 2, "argmin");
eqArr(nd_argmax($m, 1, null), [2, 2], "argmax along axis 1");
eqArr(nd_argmin($m, 0, null), [0, 0, 0], "argmin along axis 0");
// The index is within the reduced sweep, in C order.
eq(nd_get(nd_argmax($m, null, null), []), 5, "argmax over everything is the flat C-order index");

print("");
print("-- any / all / count_nonzero --");
eq(nd_get(nd_any(nd([0, 0, 1], null), null, null), []), true, "any");
eq(nd_get(nd_any(nd([0, 0, 0], null), null, null), []), false, "any of all-zero");
eq(nd_get(nd_all(nd([1, 1, 1], null), null, null), []), true, "all");
eq(nd_get(nd_all(nd([1, 0, 1], null), null, null), []), false, "all with a zero");
eq(nd_get(nd_count_nonzero(nd([0, 3, 0, 5], null), null, null), []), 2, "count_nonzero");
eqArr(nd_any(nd([[0, 1], [0, 0]], null), 1, null), [true, false], "any along an axis");

print("");
print("-- empty slices: identities where they exist, errors where they do not --");
$empty = nd_zeros([0], null);
eq(nd_get(nd_sum($empty, null, null), []), 0, "sum of nothing is 0, the identity");
eq(nd_get(nd_prod($empty, null, null), []), 1, "prod of nothing is 1, the identity");
eq(nd_get(nd_any($empty, null, null), []), false, "any of nothing is false");
eq(nd_get(nd_all($empty, null, null), []), true, "all of nothing is true");
eq(nd_get(nd_count_nonzero($empty, null, null), []), 0, "count_nonzero of nothing is 0");
// No identity exists for these, so returning anything would be a wrong answer.
raises(def() { nd_min($empty, null, null); }, "empty slice", "min of nothing RAISES");
raises(def() { nd_max($empty, null, null); }, "empty slice", "max of nothing raises");
raises(def() { nd_argmax($empty, null, null); }, "empty slice", "argmax of nothing raises");
ok(nd_get(nd_isnan(nd_mean($empty, null, null), null), []), "mean of nothing is NaN, as in NumPy");

print("");
print("-- NaN: propagating by default, skipped by the nan* forms --");
// Bantu has a single number type, so nd([1.0, 2.0]) infers i64 -- 1.0 and 1 are
// the same value. Anything that will hold a NaN must ask for f64 explicitly.
$nan = nd_get(nd_divide(nd([0.0], "f64"), nd([0.0], "f64"), null), [0]);
$wn = nd([1.0, 2.0, 4.0], "f64");
nd_set($wn, [1], $nan);
ok(nd_get(nd_isnan(nd_sum($wn, null, null), null), []), "sum propagates NaN");
ok(nd_get(nd_isnan(nd_max($wn, null, null), null), []), "max propagates NaN, like NumPy's max");
eq(nd_get(nd_nansum($wn, null, null), []), 5, "nansum skips it");
close(nd_get(nd_nanmean($wn, null, null), []), 2.5, 1e-12, "nanmean averages what is left");
eq(nd_get(nd_nanmax($wn, null, null), []), 4, "nanmax");
eq(nd_get(nd_nanmin($wn, null, null), []), 1, "nanmin");
$allNan = nd([0.0, 0.0], "f64");
nd_set($allNan, [0], $nan);
nd_set($allNan, [1], $nan);
ok(nd_get(nd_isnan(nd_nanmean($allNan, null, null), null), []),
   "an all-NaN slice gives NaN, not 0 -- 0 would be a real value that is wrong");

// NaN and +-inf have no integer representation and the conversion is silent:
// llround(NaN) is INT64_MIN and (int64_t)inf is 0 -- the second being worse,
// because 0 looks like a real answer rather than obvious garbage.
raises(def() { nd_set(nd([1, 2], "i64"), [0], $nan); }, "no integer representation",
   "storing NaN in an i64 array RAISES instead of writing INT64_MIN");
raises(def() { nd_set(nd([1, 2], "i64"), [0], nd_get(nd_divide(nd([1.0],"f64"), nd([0.0],"f64"), null), [0])); },
   "no integer representation", "and storing infinity raises instead of writing 0");
$okf = nd([1.0, 2.0], "f64");
nd_set($okf, [0], $nan);
ok(nd_get(nd_isnan($okf, null), [0]), "an f64 array accepts NaN normally");
ok(nd_get(nd_set(nd([true, false], null), [0], $nan), [0]), "a bool array takes NaN as truthy, like NumPy");

print("");
print("-- median and quantile --");
close(nd_get(nd_median(nd([3.0, 1.0, 2.0], null), null, null), []), 2.0, 1e-12, "median, odd count");
close(nd_get(nd_median(nd([4.0, 1.0, 2.0, 3.0], null), null, null), []), 2.5, 1e-12, "median, even count");
close(nd_get(nd_quantile(nd([1.0, 2.0, 3.0, 4.0], null), 0.0, null, null), []), 1.0, 1e-12, "q=0 is the min");
close(nd_get(nd_quantile(nd([1.0, 2.0, 3.0, 4.0], null), 1.0, null, null), []), 4.0, 1e-12, "q=1 is the max");
close(nd_get(nd_quantile(nd([1.0, 2.0, 3.0, 4.0], null), 0.5, null, null), []), 2.5, 1e-12, "q=0.5 is the median");
// Linear interpolation between order statistics, NumPy's default method.
close(nd_get(nd_quantile(nd([1.0, 2.0, 3.0, 4.0], null), 0.25, null, null), []), 1.75, 1e-12,
      "q=0.25 interpolates linearly between order statistics");
eqArr(nd_median(nd([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], null), 1, null), [2, 20], "median along an axis");
raises(def() { nd_quantile(nd([1.0], null), 1.5, null, null); }, "between 0 and 1",
   "a q outside [0,1] is refused");

print("");
print("-- scans --");
eqArr(nd_cumsum(nd([1, 2, 3, 4], null), null), [1, 3, 6, 10], "cumsum");
eqArr(nd_cumprod(nd([1, 2, 3, 4], null), null), [1, 2, 6, 24], "cumprod");
eqArr(nd_cummax(nd([1, 3, 2, 5], null), null), [1, 3, 3, 5], "cummax");
eqArr(nd_cummin(nd([5, 3, 4, 1], null), null), [5, 3, 3, 1], "cummin");
eqArr(nd_cumsum($m, 1), [[1, 3, 6], [4, 9, 15]], "cumsum along axis 1");
eqArr(nd_cumsum($m, 0), [[1, 2, 3], [5, 7, 9]], "cumsum along axis 0");
eqArr(nd_cumsum($m, null), [1, 3, 6, 10, 15, 21], "cumsum with no axis flattens first");
eqArr(nd_diff(nd([1, 4, 9, 16], null), null), [3, 5, 7], "diff");
eq(str(nd_shape(nd_diff($m, 1))), "[2, 2]", "diff shortens the axis by one");
eqArr(nd_diff($m, 0), [[3, 3, 3]], "diff along axis 0");

print("");
print("-- sorting --");
eqArr(nd_sort(nd([3, 1, 2], null), null), [1, 2, 3], "sort");
eqArr(nd_argsort(nd([3, 1, 2], null), null), [1, 2, 0], "argsort");
eqArr(nd_sort(nd([[3, 1], [2, 4]], null), 1), [[1, 3], [2, 4]], "sort along axis 1");
eqArr(nd_sort(nd([[3, 1], [2, 4]], null), 0), [[2, 1], [3, 4]], "sort along axis 0");
eqArr(nd_sort(nd([1, 2, 3], null), null), [1, 2, 3], "an already-sorted array is unchanged");
eqArr(nd_sort(nd([3, 2, 1], null), null), [1, 2, 3], "a reverse-sorted array");
eqArr(nd_sort(nd([2, 2, 2], null), null), [2, 2, 2], "an all-equal array");
// argsort is STABLE, so equal elements keep their input order and the answer
// is reproducible run to run rather than depending on sort internals.
eqArr(nd_argsort(nd([5, 1, 5, 1], null), null), [1, 3, 0, 2], "argsort is stable on ties");
// NaN sorts LAST. Some rule has to be imposed -- every comparison with NaN is
// false -- and this is NumPy's.
$sn = nd([3.0, 1.0, 2.0], "f64");
nd_set($sn, [0], $nan);
ok(nd_get(nd_isnan(nd_get(nd_sort($sn, null), [2]), null), []), "NaN sorts to the end");
eq(nd_get(nd_sort($sn, null), [0]), 1, "and the real values sort normally around it");

print("");
print("-- searchsorted, unique, bincount, histogram --");
$sorted = nd([1, 3, 5, 7], null);
eqArr(nd_searchsorted($sorted, nd([0, 4, 8], null), null), [0, 2, 4], "searchsorted");
eqArr(nd_searchsorted($sorted, nd([3], null), null), [1], "searchsorted, left side of a tie");
eqArr(nd_searchsorted($sorted, nd([3], null), "right"), [2], "searchsorted, right side");
eqArr(nd_unique(nd([3, 1, 3, 2, 1], null)), [1, 2, 3], "unique sorts and de-duplicates");
eqArr(nd_unique(nd([5], null)), [5], "unique of one element");
eqArr(nd_bincount(nd([0, 1, 1, 3], null), null), [1, 2, 0, 1], "bincount");
eqArr(nd_bincount(nd([0, 1], null), 5), [1, 1, 0, 0, 0], "bincount honours minlength");
raises(def() { nd_bincount(nd([-1], null), null); }, "non-negative",
   "bincount refuses a negative value");
eqArr(nd_histogram(nd([1.0, 2.0, 3.0, 4.0], null), 2, 1.0, 5.0), [2, 2], "histogram with a range");
eq(nd_size(nd_histogram(nd([1.0, 2.0], null), 4, null, null)), 4, "histogram infers its range");
// The top edge is inclusive, so the largest value lands in the last bin rather
// than being dropped.
eqArr(nd_histogram(nd([0.0, 1.0], null), 2, 0.0, 1.0), [1, 1], "the top edge is inclusive");
raises(def() { nd_histogram(nd([1.0], null), 0, null, null); }, "at least 1",
   "a zero-bin histogram is refused");

print("");
print("-- fancy and boolean indexing --");
$src = nd([10, 20, 30, 40], null);
eqArr(nd_take($src, nd([0, 2], null), null), [10, 30], "take");
eqArr(nd_take($src, nd([-1], null), null), [40], "take with a negative index");
eqArr(nd_take($m, nd([1], null), 0), [[4, 5, 6]], "take a row");
eqArr(nd_take($m, nd([0, 2], null), 1), [[1, 3], [4, 6]], "take columns");
raises(def() { nd_take($src, nd([9], null), null); }, "out of range",
   "an out-of-range take index RAISES");
raises(def() { nd_take($src, nd([-9], null), null); }, "out of range",
   "and so does one that is too negative");

$mask = nd_greater($src, 15, null);
eqArr(nd_compress($mask, $src), [20, 30, 40], "compress with a boolean mask");
eqArr(nd_compress(nd_zeros([4], null), $src), [], "an all-false mask selects nothing");
raises(def() { nd_compress(nd_zeros([3], null), $src); }, "wrong length",
   "a mask of the wrong length is an error, not a silent truncation");
eqArr(nd_nonzero(nd([0, 5, 0, 7], null)), [1, 3], "nonzero gives flat C-order indices");
eqArr(nd_nonzero(nd([[0, 1], [1, 0]], null)), [1, 2], "nonzero on a 2-d array");

$dst = nd_zeros([4], null);
nd_put($dst, nd([0, 3], null), nd([9, 9], null));
eqArr($dst, [9, 0, 0, 9], "put scatters values");
nd_put($dst, nd([1], null), nd([7], null));
eqArr($dst, [9, 7, 0, 9], "put with one index");
nd_put($dst, nd([0, 1], null), nd([5], null));
eqArr($dst, [5, 5, 0, 9], "put broadcasts a single value across the indices");
raises(def() { nd_put($dst, nd([99], null), nd([1], null)); }, "out of range",
   "an out-of-range put index raises");
raises(def() { nd_put(nd_broadcast_to(nd([0], null), [4]), nd([0], null), nd([1], null)); },
   "read-only", "put through a read-only broadcast view raises");

print("");
print("-- performance --");
$big = nd_random_uniform([10000000], 0, 1);
$t0 = clock();
$bs = nd_sum($big, null, null);
$sumMs = clock() - $t0;
print("        10M nd_sum: " + str($sumMs) + "ms");
ok($sumMs < 120, "10M nd_sum is under 120ms");

$mid = nd_random_uniform([1000000], 0, 1);
$t0 = clock();
$as = nd_argsort($mid, null);
$argMs = clock() - $t0;
print("        1M nd_argsort: " + str($argMs) + "ms");
ok($argMs < 1500, "1M nd_argsort is under 1500ms");
eq(nd_size($as), 1000000, "and returns one index per element");
// A sorted result must actually be sorted -- checked at the ends and by the
// fact that applying argsort's permutation reproduces sort's output.
$srt = nd_sort($mid, null);
ok(nd_get($srt, [0]) <= nd_get($srt, [999999]), "the sorted array really is ordered");
ok(nd_array_equal(nd_take($mid, $as, null), $srt),
   "argsort's permutation applied to the input reproduces sort's output");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
