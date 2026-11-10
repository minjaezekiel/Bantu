// ════════════════════════════════════════════════════════════════════════
//  numba_f32_test.b — the f32 dtype (ecosystem stage 1, item D2).
//
//  f32 exists for two reasons: half the memory, and -- eventually -- twice the
//  SIMD width. The first is delivered here. The second needs the ufunc functors
//  to be generic rather than `double`-typed, which is D2b; until then f32
//  arithmetic is COMPUTED in double and stored in float, so this suite checks
//  memory and correctness, not arithmetic throughput.
//
//  Four things are easy to get wrong and silent when wrong, so each has its own
//  section:
//
//    1. The PROMOTION lattice. f32 + i64 must widen to f64, not narrow to f32
//       -- an i64 holds values no f32 can represent, and narrowing loses them
//       without a word.
//    2. The RESULT dtype of every reduction. An f32 sum landing in i64
//       truncates every value; an f32 mean landing in f64 promotes a whole f32
//       pipeline back to f64 one call in, undoing the point of f32.
//    3. The ACCUMULATOR. An f32 array must reduce through an f64 accumulator
//       (numba DECISIONS N24). Summed in f32, a million values carry ~1e-2
//       relative error against this library's ~2e-16 guarantee.
//    4. OVERFLOW on narrowing. double -> float out of range is undefined
//       behaviour in C++, not "gives inf", so it has to be explicit.
//
//  Run:  bantu run tests/numba_f32_test.b
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
// numba has no nd_item; nd_get reads a 0-d array (what an axis-less reduction
// returns) or the first element of a 1-element one.
def item($x) { return nd_get($x, 0); }
// f32 carries ~7 decimal digits, so 1e-6 relative is the right band.
def close32($x, $y, $name) {
    ok(nd_allclose(nd_astype($x, "f64"), nd_astype($y, "f64"), 1.0e-6, 1.0e-6), $name);
}

print("");
print("  numba f32 (D2)");
print("  ════════════════════════════════════════");

// ── 1. the dtype exists, and costs half ─────────────────────────────────────
print("");
print("-- the dtype itself --");
$z = nd_zeros([10], "f32");
eqs(nd_dtype($z), "f32", "nd_zeros(..., \"f32\") makes an f32 array");
eqs(nd_itemsize($z), 4, "an f32 element is 4 bytes, not 8");
eqs(nd_nbytes(nd_zeros([1000], "f32")), 4000, "1000 f32 = 4000 bytes");
eqs(nd_nbytes(nd_zeros([1000], "f64")), 8000, "1000 f64 = 8000 bytes (unchanged)");
ok(nd_nbytes(nd_zeros([1000], "f32")) * 2 == nd_nbytes(nd_zeros([1000], "f64")),
   "f32 is EXACTLY half of f64");
eqs(nd_dtype(nd_zeros([4], "float32")), "f32", "\"float32\" is accepted as a spelling");
eqs(nd_dtype(nd_zeros([4], "single")),  "f32", "\"single\" is accepted as a spelling");

// ── 2. the promotion lattice, exhaustively ──────────────────────────────────
print("");
print("-- promotion (the whole lattice, no gaps) --");
$f32 = nd_full([3], 2.0, "f32");
$f64 = nd_full([3], 2.0, "f64");
$i64 = nd_full([3], 2.0, "i64");
$bl  = nd_astype(nd_full([3], 1.0, "f64"), "bool");

eqs(nd_dtype(nd_add($f32, $f32, null)), "f32", "f32 + f32 -> f32");
eqs(nd_dtype(nd_add($f32, $f64, null)), "f64", "f32 + f64 -> f64 (widens)");
eqs(nd_dtype(nd_add($f64, $f32, null)), "f64", "f64 + f32 -> f64 (order-free)");
eqs(nd_dtype(nd_add($f32, $bl,  null)), "f32", "f32 + bool -> f32");
eqs(nd_dtype(nd_add($bl,  $f32, null)), "f32", "bool + f32 -> f32 (order-free)");
// The one that matters: an i64 carries values no f32 can represent.
eqs(nd_dtype(nd_add($f32, $i64, null)), "f64", "f32 + i64 -> f64, NOT f32");
eqs(nd_dtype(nd_add($i64, $f32, null)), "f64", "i64 + f32 -> f64 (order-free)");
// and the pre-f32 lattice is untouched
eqs(nd_dtype(nd_add($i64, $i64, null)), "i64", "i64 + i64 -> i64 (unchanged)");
eqs(nd_dtype(nd_add($f64, $i64, null)), "f64", "f64 + i64 -> f64 (unchanged)");

// why f32 + i64 -> f32 would be wrong, shown rather than asserted abstractly
$big = nd_full([1], 9007199254740993.0, "i64");   // 2^53 + 1
$sum = nd_add($big, nd_full([1], 0.0, "f32"), null);
eqs(nd_dtype($sum), "f64", "2^53+1 as i64 plus f32 stays f64 (no silent narrowing)");

// ── 3. every op class agrees with the f64 answer ─────────────────────────────
print("");
print("-- f32 agrees with f64 to 1e-6, op class by op class --");
nd_seed(424242);
$a64 = nd_random_uniform([512], 0.25, 4.0);
$b64 = nd_random_uniform([512], 0.25, 4.0);
$a32 = nd_astype($a64, "f32");
$b32 = nd_astype($b64, "f32");

close32(nd_add($a32, $b32, null),      nd_add($a64, $b64, null),      "add");
close32(nd_subtract($a32, $b32, null), nd_subtract($a64, $b64, null), "subtract");
close32(nd_multiply($a32, $b32, null), nd_multiply($a64, $b64, null), "multiply");
close32(nd_divide($a32, $b32, null),   nd_divide($a64, $b64, null),   "divide");
close32(nd_maximum($a32, $b32, null),  nd_maximum($a64, $b64, null),  "maximum");
close32(nd_exp($a32, null),            nd_exp($a64, null),            "exp (Kind::REAL)");
close32(nd_log($a32, null),            nd_log($a64, null),            "log");
close32(nd_sqrt($a32, null),           nd_sqrt($a64, null),           "sqrt");
close32(nd_sin($a32, null),            nd_sin($a64, null),            "sin");
close32(nd_abs($a32, null),            nd_abs($a64, null),            "abs");
// tier 1: a scalar splatted over an array
close32(nd_add($a32, 1.5, null),       nd_add($a64, 1.5, null),       "scalar splat (tier 1)");
close32(nd_subtract(1.5, $a32, null),  nd_subtract(1.5, $a64, null),  "reversed scalar (tier 1)");
// tier 2: a strided view disqualifies every contiguous fast path
$m32 = nd_transpose(nd_reshape($a32, [16, 32]), null);
$m64 = nd_transpose(nd_reshape($a64, [16, 32]), null);
close32(nd_multiply($m32, 2.0, null),  nd_multiply($m64, 2.0, null),  "strided (tier 2)");
close32(nd_exp($m32, null),            nd_exp($m64, null),            "strided unary (tier 2)");

// a comparison on f32 still yields bool, and the same bools as f64
eqs(nd_dtype(nd_greater($a32, $b32, null)), "bool", "f32 comparison yields bool");
ok(nd_all(nd_equal(nd_greater($a32, $b32, null),
                   nd_greater($a64, $b64, null), null), null, null),
   "f32 and f64 comparisons agree element for element");

// ── 4. reduction RESULT dtypes ───────────────────────────────────────────────
print("");
print("-- what dtype each reduction returns --");
eqs(nd_dtype(nd_sum($a32, null, null)),    "f32", "sum of f32 is f32, not i64");
eqs(nd_dtype(nd_prod($a32, null, null)),   "f32", "prod of f32 is f32, not i64");
eqs(nd_dtype(nd_mean($a32, null, null)),   "f32", "mean of f32 stays f32");
eqs(nd_dtype(nd_var($a32, null, null)),    "f32", "var of f32 stays f32");
eqs(nd_dtype(nd_std($a32, null, null)),    "f32", "std of f32 stays f32");
eqs(nd_dtype(nd_median($a32, null, null)), "f32", "median of f32 stays f32");
eqs(nd_dtype(nd_max($a32, null, null)),    "f32", "max of f32 stays f32");
eqs(nd_dtype(nd_min($a32, null, null)),    "f32", "min of f32 stays f32");
eqs(nd_dtype(nd_ptp($a32, null, null)),    "f32", "ptp of f32 stays f32");
// the integer and f64 rules are untouched
eqs(nd_dtype(nd_sum($i64, null, null)),    "i64", "sum of i64 is still i64");
eqs(nd_dtype(nd_sum($a64, null, null)),    "f64", "sum of f64 is still f64");
eqs(nd_dtype(nd_mean($i64, null, null)),   "f64", "mean of i64 is still f64");
// argmax/argmin index, so they stay integers whatever the input
eqs(nd_dtype(nd_argmax($a32, null, null)), "i64", "argmax of f32 is i64 (it is an index)");

// and the VALUES agree
close32(nd_sum($a32, null, null),  nd_sum($a64, null, null),  "sum value agrees");
close32(nd_mean($a32, null, null), nd_mean($a64, null, null), "mean value agrees");
close32(nd_max($a32, null, null),  nd_max($a64, null, null),  "max value agrees");
close32(nd_std($a32, null, null),  nd_std($a64, null, null),  "std value agrees");

// axis reductions, both the contiguous fast path and the strided one
$g32 = nd_reshape(nd_astype(nd_random_uniform([4096], 0.5, 1.5), "f32"), [64, 64]);
$g64 = nd_astype($g32, "f64");
close32(nd_sum($g32, 1, null), nd_sum($g64, 1, null), "sum over axis 1 (inner contiguous)");
close32(nd_sum($g32, 0, null), nd_sum($g64, 0, null), "sum over axis 0 (strided inner)");
close32(nd_max($g32, 1, null), nd_max($g64, 1, null), "max over axis 1 (extremeFast<float>)");
close32(nd_max($g32, 0, null), nd_max($g64, 0, null), "max over axis 0 (generic sweep)");

// ── 5. the accumulator is f64, not f32 (DECISIONS N24) ───────────────────────
print("");
print("-- the accumulator split: f32 storage, f64 reduction --");
// 1,000,000 copies of 0.1. In f32 arithmetic the running total stops being able
// to represent its own next increment and the error grows to ~1e-2 relative.
// Through an f64 accumulator it stays near machine epsilon.
// Named without an `nd_` prefix on purpose: that prefix is reserved for the
// native family, and an unbound `nd_*` is what triggers the stale-binary
// diagnosis (DECISIONS B21). A test helper must not look like a kernel.
def absd($v) { if ($v < 0.0) { return 0.0 - $v; } return $v; }
def relerr($gotArr, $want) {
    $g = item(nd_astype($gotArr, "f64"));
    return absd($g - $want) / absd($want);
}

// Case A: a million copies of 0.1f.
//
// Note what the achievable bound actually is. The RESULT is f32 (sum of f32 is
// f32), and f32 near 100000 has an ulp of ~0.0078, so the stored answer cannot
// be better than ~1e-7 relative however perfectly it was accumulated. That is
// the f32 result's resolution, not the accumulator's error -- conflating the
// two is how this assertion was first written with an impossible 1e-12 bound.
//
// The discrimination is still enormous: f64 accumulation lands at ~1e-8, while
// f32 accumulation of this series drifts to ~1e-2. Four orders of margin.
$N   = 1000000;
$ten = nd_full([$N], 0.1, "f32");
$got = nd_sum($ten, null, null);
$one32 = nd_astype(nd_full([1], 0.1, "f64"), "f32");
$want  = item(nd_astype($one32, "f64")) * ($N * 1.0);
$relA  = relerr($got, $want);
print("        1M x 0.1f summed: relative error " + str($relA));
ok($relA < 1.0e-6, "a 1M-element f32 sum sits at f32 RESULT resolution, not f32 accumulator drift");

// Case B: the sharp one. 2^25 followed by 999,999 ones.
//
// f32's ulp at 2^25 is 4.0, so in f32 arithmetic `33554432 + 1.0 == 33554432`
// exactly -- a naive f32 accumulator cannot move at all and returns the
// sentinel unchanged. Through an f64 accumulator every one of those ones lands.
// This is a case f32 arithmetic CANNOT represent, so it tests the accumulator
// rather than the storage.
$sent = nd_full([1], 33554432.0, "f32");           // 2^25
$ones = nd_full([999999], 1.0, "f32");
$mix  = nd_concat([$sent, $ones], 0);
eqs(nd_dtype($mix), "f32", "the sentinel array is f32");
$gotB  = nd_sum($mix, null, null);
$wantB = 33554432.0 + 999999.0;                    // 34554431
$relB  = relerr($gotB, $wantB);
print("        2^25 + 999999x1.0f: relative error " + str($relB)
    + "   (an f32 accumulator would give ~2.9e-2)");
ok($relB < 1.0e-6, "f32 reduces through an f64 accumulator (N24), not an f32 one");

// ── 6. inf, NaN, subnormals, and narrowing overflow ──────────────────────────
print("");
print("-- the edges --");
$inf32 = nd_astype(nd_full([1], 1.0e308, "f64"), "f32");
ok(item(nd_isinf($inf32, null)), "1e308 -> f32 saturates to +inf, not UB");
$ninf32 = nd_astype(nd_full([1], -1.0e308, "f64"), "f32");
ok(item(nd_isinf($ninf32, null)), "-1e308 -> f32 saturates to -inf");
ok(item(nd_less($ninf32, nd_zeros([1], "f32"), null)), "and it is the NEGATIVE one");

$nan32 = nd_astype(nd_divide(nd_zeros([1], "f64"), nd_zeros([1], "f64"), null), "f32");
ok(item(nd_isnan($nan32, null)), "NaN survives the narrowing to f32");

// f32's own limits round-trip exactly
$fmax = nd_full([1], 3.4028234663852886e38, "f64");
ok(item(nd_isinf(nd_astype($fmax, "f32"), null)) == false,
   "FLT_MAX itself does NOT overflow to inf");
// a value just inside f32's subnormal range survives as nonzero
$sub = nd_astype(nd_full([1], 1.0e-44, "f64"), "f32");
ok(item(nd_astype($sub, "f64")) > 0.0, "a subnormal stays nonzero, not flushed to 0");
// underflow below the subnormal range is 0, not an error
$under = nd_astype(nd_full([1], 1.0e-60, "f64"), "f32");
eqs(item(nd_astype($under, "f64")), 0.0, "1e-60 underflows to exactly 0");

// round-trip: f32 -> f64 -> f32 is the identity
$rt = nd_astype(nd_astype($a32, "f64"), "f32");
ok(nd_all(nd_equal($rt, $a32, null), null, null), "f32 -> f64 -> f32 is EXACTLY the identity");

// a bad dtype name names the valid ones
$threw = false;
try { nd_zeros([2], "f16"); } catch ($e) { $threw = true; }
ok($threw, "an unknown dtype raises rather than guessing");

// ── 7. f32 through the rest of the library ───────────────────────────────────
print("");
print("-- f32 reaches the non-ufunc kernels too --");
eqs(nd_dtype(nd_reshape($a32, [16, 32])), "f32", "reshape preserves f32");
eqs(nd_dtype(nd_transpose(nd_reshape($a32, [16, 32]), null)), "f32", "transpose preserves f32");
eqs(nd_dtype(nd_concat([$f32, $f32], 0)), "f32", "concat of f32 stays f32");
eqs(nd_dtype(nd_sort($a32, null)), "f32", "sort preserves f32");
close32(nd_sort($a32, null), nd_astype(nd_sort($a64, null), "f32"), "sort gives the same order");
close32(nd_cumsum($a32, null), nd_cumsum($a64, null), "cumsum agrees");
$mm32 = nd_matmul(nd_reshape($a32, [16, 32]), nd_reshape($b32, [32, 16]));
$mm64 = nd_matmul(nd_reshape($a64, [16, 32]), nd_reshape($b64, [32, 16]));
close32($mm32, $mm64, "matmul agrees with the f64 answer");

// ── linear algebra is still an f64 operation (DECISIONS N31) ────────────────
// These assertions PIN a known limitation rather than describing a desired
// state. nd_matmul and the decompositions compute on `double*` buffers and
// return f64 whatever they were given. The values are right -- what is missing
// is dtype preservation, and for matmul that matters: until it is fixed, a
// Linear layer turns an f32 model back into f64 at the first layer, which is
// why btorch f32 (item D2c) is not reachable yet.
//
// True f32 GEMM (f32 in, f64 accumulator, f32 out, as every real BLAS does)
// lands in D4, where these kernels are being rewritten anyway. When it does,
// these four assertions are the ones to change -- deliberately, not by
// accident.
$sq32 = nd_astype(nd_add(nd_random_uniform([4,4], 0.5, 1.5),
                         nd_multiply(nd_eye(4,4,0,null), 4.0, null), null), "f32");
eqs(nd_dtype(nd_matmul($sq32, $sq32)), "f64", "matmul(f32,f32) returns f64 today (D4 fixes this)");
eqs(nd_dtype(nd_inv($sq32)),           "f64", "inv of f32 returns f64 (decompositions stay f64)");
eqs(nd_dtype(nd_det($sq32)),           "f64", "det of f32 returns f64");
eqs(nd_dtype(nd_dot(nd_astype(nd_full([4], 2.0, "f64"), "f32"),
                    nd_astype(nd_full([4], 3.0, "f64"), "f32"))),
    "f64", "dot of f32 returns f64");
eqs(nd_dtype(nd_clip($a32, 1.0, 2.0, null)), "f32", "clip preserves f32");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
