// ════════════════════════════════════════════════════════════════════════
//  numba_elemwise_threads_test.b — threaded element-wise ufuncs (D3b).
//
//  Element i of a tier-0/1 loop reads only input element i and writes only
//  output element i, so there is no summation order to preserve and nothing
//  about the ANSWER can depend on the chunking. That makes the assertion
//  simple and absolute: EXACT equality against the single-threaded result,
//  for every op, at every thread count.
//
//  Shapes are deliberately not multiples of the 4096-element chunk: 16384 is
//  exactly four chunks, 20000 is four chunks plus 3616 elements, and 16383 is
//  one element short of the threshold, so it must not thread at all and must
//  still agree.
//
//  Only the transcendentals are threaded -- the measurements are in
//  ndarray_kernels.hpp -- so the cheap ops here are testing the opposite
//  property: that turning threading on changes neither their answers nor
//  (measurably) their speed.
//
//  Run:  bantu run tests/numba_elemwise_threads_test.b
// ════════════════════════════════════════════════════════════════════════

$R = {"pass": 0, "fail": 0};
def ok($c, $n) {
    if ($c) { $R.pass = $R.pass + 1; print("  ok    " + $n); }
    else    { $R.fail = $R.fail + 1; print("  FAIL  " + $n); }
}
def eqs($got, $want, $n) {
    if ($got == $want) { $R.pass = $R.pass + 1; print("  ok    " + $n); }
    else { $R.fail = $R.fail + 1;
           print("  FAIL  " + $n + "   got " + str($got) + ", want " + str($want)); }
}
// Exact, and NaN-aware: a threaded exp of a NaN must produce the same NaN
// pattern, and nd_equal says false for NaN == NaN, so compare the bits of
// "equal OR both-nan".
def exact($x, $y, $n) {
    if (nd_shape($x) != nd_shape($y)) {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $n + "  shape " + str(nd_shape($x)) + " vs " + str(nd_shape($y)));
        return null;
    }
    $same = nd_logical_or(nd_equal($x, $y, null),
                          nd_logical_and(nd_isnan($x, null), nd_isnan($y, null), null), null);
    ok(nd_get(nd_all($same, null, null), 0), $n);
    return null;
}

print("");
print("  numba threaded element-wise ufuncs (D3b)");
print("  ════════════════════════════════════════");

nd_seed(4242);
$THREADS = [2, 3, 4, 8, 16];

// ── 1. every threaded op, every thread count, three shapes ──────────────────
print("");
print("-- the threaded (transcendental) ops are bit-identical --");
$sizes = [16384, 20000, 131072];
$si = 0;
while ($si < len($sizes)) {
    $n  = $sizes[$si];
    $x  = nd_random_uniform([$n], 0.2, 3.0);     // positive: safe for log, acosh
    $y  = nd_random_uniform([$n], 0.5, 2.0);
    $u  = nd_random_uniform([$n], -0.9, 0.9);    // in -1..1 for asin, atanh

    nd_set_threads(1);
    $ref = {"exp":   nd_exp($y, null),   "expm1": nd_expm1($y, null),
            "log":   nd_log($x, null),   "log1p": nd_log1p($x, null),
            "log2":  nd_log2($x, null),  "log10": nd_log10($x, null),
            "sin":   nd_sin($y, null),   "cos":   nd_cos($y, null),
            "tan":   nd_tan($u, null),   "asin":  nd_asin($u, null),
            "acos":  nd_acos($u, null),  "atan":  nd_atan($y, null),
            "sinh":  nd_sinh($u, null),  "cosh":  nd_cosh($u, null),
            "tanh":  nd_tanh($y, null),  "asinh": nd_asinh($y, null),
            "acosh": nd_acosh($x, null), "atanh": nd_atanh($u, null),
            "cbrt":  nd_cbrt($y, null),
            "power": nd_power($x, $y, null),
            "atan2": nd_atan2($x, $y, null),
            "hypot": nd_hypot($x, $y, null)};

    $ti = 0;
    while ($ti < len($THREADS)) {
        $t = $THREADS[$ti];
        nd_set_threads($t);
        $tag = " n=" + str($n) + " @" + str($t);
        exact(nd_exp($y, null),       $ref.exp,   "exp" + $tag);
        exact(nd_expm1($y, null),     $ref.expm1, "expm1" + $tag);
        exact(nd_log($x, null),       $ref.log,   "log" + $tag);
        exact(nd_log1p($x, null),     $ref.log1p, "log1p" + $tag);
        exact(nd_log2($x, null),      $ref.log2,  "log2" + $tag);
        exact(nd_log10($x, null),     $ref.log10, "log10" + $tag);
        exact(nd_sin($y, null),       $ref.sin,   "sin" + $tag);
        exact(nd_cos($y, null),       $ref.cos,   "cos" + $tag);
        exact(nd_tan($u, null),       $ref.tan,   "tan" + $tag);
        exact(nd_asin($u, null),      $ref.asin,  "asin" + $tag);
        exact(nd_acos($u, null),      $ref.acos,  "acos" + $tag);
        exact(nd_atan($y, null),      $ref.atan,  "atan" + $tag);
        exact(nd_sinh($u, null),      $ref.sinh,  "sinh" + $tag);
        exact(nd_cosh($u, null),      $ref.cosh,  "cosh" + $tag);
        exact(nd_tanh($y, null),      $ref.tanh,  "tanh" + $tag);
        exact(nd_asinh($y, null),     $ref.asinh, "asinh" + $tag);
        exact(nd_acosh($x, null),     $ref.acosh, "acosh" + $tag);
        exact(nd_atanh($u, null),     $ref.atanh, "atanh" + $tag);
        exact(nd_cbrt($y, null),      $ref.cbrt,  "cbrt" + $tag);
        exact(nd_power($x, $y, null), $ref.power, "power" + $tag);
        exact(nd_atan2($x, $y, null), $ref.atan2, "atan2" + $tag);
        exact(nd_hypot($x, $y, null), $ref.hypot, "hypot" + $tag);
        $ti = $ti + 1;
    }
    nd_set_threads(1);
    $si = $si + 1;
}

// ── 2. the cheap ops are untouched, and still agree ─────────────────────────
// They are deliberately NOT threaded (measured: add is 0.83x on six threads at
// n=1M, one core already saturates the bus). This asserts the answers, which
// must hold whatever the policy.
print("");
print("-- the unthreaded (cheap) ops agree too --");
$a = nd_random_uniform([200000], 0.5, 3.0);
$b = nd_random_uniform([200000], 0.5, 3.0);
nd_set_threads(1);
$cref = {"add": nd_add($a, $b, null), "mul": nd_multiply($a, $b, null),
         "sub": nd_subtract($a, $b, null), "div": nd_divide($a, $b, null),
         "sqrt": nd_sqrt($a, null), "abs": nd_abs($a, null),
         "floor": nd_floor($a, null), "lt": nd_less($a, $b, null),
         "neg": nd_negative($a, null)};
for $t in [2, 8] {
    nd_set_threads($t);
    exact(nd_add($a, $b, null),      $cref.add,   "add @" + str($t));
    exact(nd_multiply($a, $b, null), $cref.mul,   "multiply @" + str($t));
    exact(nd_subtract($a, $b, null), $cref.sub,   "subtract @" + str($t));
    exact(nd_divide($a, $b, null),   $cref.div,   "divide @" + str($t));
    exact(nd_sqrt($a, null),         $cref.sqrt,  "sqrt @" + str($t));
    exact(nd_abs($a, null),          $cref.abs,   "abs @" + str($t));
    exact(nd_floor($a, null),        $cref.floor, "floor @" + str($t));
    exact(nd_less($a, $b, null),     $cref.lt,    "less @" + str($t));
    exact(nd_negative($a, null),     $cref.neg,   "negative @" + str($t));
}
nd_set_threads(1);

// ── 3. below the threshold ──────────────────────────────────────────────────
// 16383 is one element short of kElemMin, so it must run sequentially -- a
// performance decision, so it has to be invisible in the answer.
print("");
print("-- below the threshold, and at exactly one chunk --");
$small = nd_random_uniform([16383], 0.5, 2.0);
$tiny  = nd_random_uniform([7], 0.5, 2.0);
nd_set_threads(1);
$sref = nd_exp($small, null);
$tref = nd_exp($tiny, null);
nd_set_threads(16);
exact(nd_exp($small, null), $sref, "n=16383 (one short of the threshold) @16");
exact(nd_exp($tiny, null),  $tref, "n=7 @16");
exact(nd_exp(nd_zeros([0], "f64"), null), nd_zeros([0], "f64"), "an empty array @16");
nd_set_threads(1);

// ── 4. f32 ──────────────────────────────────────────────────────────────────
// f32 has a quarter of f64's mantissa, so a perturbation would show here
// first. It also goes through a different tier arm (storage f32, compute f64).
print("");
print("-- f32 is identical across thread counts --");
$f = nd_astype(nd_random_uniform([131072], 0.5, 2.0), "f32");
nd_set_threads(1);
$fe = nd_exp($f, null);
$fl = nd_log($f, null);
eqs(nd_dtype($fe), "f32", "an f32 exp is still f32 when threaded");
for $t in [2, 4, 16] {
    nd_set_threads($t);
    exact(nd_exp($f, null), $fe, "f32 exp @" + str($t));
    exact(nd_log($f, null), $fl, "f32 log @" + str($t));
}
nd_set_threads(1);

// ── 5. the other paths through the same ops ────────────────────────────────
// tier 1 (one side a scalar), tier 2 (strided/broadcast, NOT threaded), and
// the out= destination, which writes into storage the caller already owns.
print("");
print("-- scalar, strided and out= paths --");
$m = nd_random_uniform([400, 500], 0.5, 2.0);
nd_set_threads(1);
$p1   = nd_power($m, 2.5, null);              // tier 1
$strd = nd_exp(nd_slice($m, [null, [0, 500, 2]]), null);   // tier 2: strided
$dest = nd_zeros([400, 500], "f64");
nd_exp($m, $dest);
$oref = nd_copy($dest, null);
for $t in [3, 8] {
    nd_set_threads($t);
    exact(nd_power($m, 2.5, null), $p1, "power by a scalar @" + str($t));
    exact(nd_exp(nd_slice($m, [null, [0, 500, 2]]), null), $strd,
          "a strided (tier 2) exp @" + str($t));
    $d2 = nd_zeros([400, 500], "f64");
    nd_exp($m, $d2);
    exact($d2, $oref, "exp into an out= destination @" + str($t));
}
nd_set_threads(1);

// ── 6. the special values ──────────────────────────────────────────────────
// A chunk boundary must not change how inf, -inf, NaN or a subnormal is
// handled, and log of a negative must still be NaN in the same places.
print("");
print("-- inf, NaN and subnormals across a chunk boundary --");
$sp = nd_full([20000], 1.0, "f64");
nd_set($sp, [4095], 0.0);                    // last element of chunk 0
nd_set($sp, [4096], -1.0);                   // first of chunk 1 -> log is NaN
nd_set($sp, [8191], 1.0e308);
nd_set($sp, [8192], 5.0e-324);               // subnormal
nd_set($sp, [12287], -0.0);
nd_set($sp, [16383], 1.0e-320);
nd_set_threads(1);
$spl = nd_log($sp, null);
$spe = nd_exp(nd_multiply($sp, 1000.0, null), null);
ok(nd_get(nd_any(nd_isnan($spl, null), null, null), 0), "the reference log has NaNs in it");
ok(nd_get(nd_any(nd_isinf($spe, null), null, null), 0), "the reference exp has infinities in it");
for $t in [2, 6, 16] {
    nd_set_threads($t);
    exact(nd_log($sp, null), $spl, "log of inf/NaN/subnormal @" + str($t));
    exact(nd_exp(nd_multiply($sp, 1000.0, null), null), $spe,
          "exp overflowing to inf @" + str($t));
}
nd_set_threads(1);

// ── 7. errors still arrive as errors ───────────────────────────────────────
// An exception thrown inside a parallel region would be std::terminate if the
// pool let it escape a worker. The ufuncs that can throw are all CHEAP, so
// they run sequentially -- but the message must be unchanged whatever the
// thread count, and the pool must survive it.
print("");
print("-- a failing op raises the same error at any thread count --");
$big = nd_astype(nd_full([200000], 4.0, "f64"), "i64");
$bad = nd_astype(nd_full([200000], 99.0, "f64"), "i64");
$msg1 = "";
$msg6 = "";
nd_set_threads(1);
try { nd_left_shift($big, $bad, null); } catch ($e) { $msg1 = str($e); }
nd_set_threads(6);
try { nd_left_shift($big, $bad, null); } catch ($e) { $msg6 = str($e); }
ok($msg1 != "", "a bad shift count raises single-threaded");
eqs($msg6, $msg1, "and raises the identical message on six threads");
// and the pool is not left claimed by the failure
$after = nd_exp(nd_random_uniform([131072], 0.5, 2.0), null);
ok(nd_get(nd_all(nd_isfinite($after, null), null, null), 0),
   "a threaded region still runs after an error");
nd_set_threads(1);

// ── 8. no leak ─────────────────────────────────────────────────────────────
print("");
print("-- memory --");
$lx = nd_random_uniform([131072], 0.5, 2.0);
nd_set_threads(4);
nd_exp($lx, null);                            // warm the pool
$before = nd_live_bytes();
$i = 0;
while ($i < 500) { nd_exp($lx, null); $i = $i + 1; }
$after2 = nd_live_bytes();
print("        live bytes " + str($before) + " -> " + str($after2) + " over 500 threaded ufuncs");
ok($after2 <= $before, "500 threaded element-wise regions do not grow live memory");
nd_set_threads(1);

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
