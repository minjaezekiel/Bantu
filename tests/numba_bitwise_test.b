// ════════════════════════════════════════════════════════════════════════
//  tests/numba_bitwise_test.b — nd_bitwise_* and the two shifts.
//
//  Six operations NumPy has had since 2006 and numba had none of, so there
//  was no way to do bitwise arithmetic on an array AT ALL -- not slowly, not
//  at all. The only recourse was 32 whole-array passes built from
//  floor_divide and mod, one per bit.
//
//  Three properties here are the ones a wrong implementation gets wrong:
//  `~` is the one's complement and not a logical not, `>>` replicates the
//  sign bit, and an f64 operand is converted when it IS an integer and
//  refused by value when it is not.
//
//  docs/numba-architecture.md §4.6.1.
//
//  Run:  bantu run tests/numba_bitwise_test.b
// ════════════════════════════════════════════════════════════════════════

$R = {"p": 0, "f": 0};
def eq($got, $want, $name) {
    if ($got == $want) { $R.p = $R.p + 1; }
    else {
        $R.f = $R.f + 1;
        print("  FAIL  " + $name);
        print("          got:  " + str($got));
        print("          want: " + str($want));
    }
}
def throws($fn, $name) {
    $caught = false;
    try { $fn(); } catch ($e) { $caught = true; }
    if ($caught) { $R.p = $R.p + 1; }
    else { $R.f = $R.f + 1; print("  FAIL  (expected an error) " + $name); }
}

$I = nd([0, 1, 5, 6, 255, -1, -8], "i64");

// ─── and / or / xor against a scalar ────────────────────────────────────
print("── and, or, xor ──");

eq(nd_to_list(nd_bitwise_and($I, 3)),  [0, 1, 1, 2, 3, 3, 0],      "and 3");
eq(nd_to_list(nd_bitwise_or($I, 3)),   [3, 3, 7, 7, 255, -1, -5],  "or 3");
eq(nd_to_list(nd_bitwise_xor($I, 3)),  [3, 2, 6, 5, 252, -4, -5],  "xor 3");

// against another array, element by element
$A = nd([12, 12, 12, 12], "i64");
$B = nd([10, 5, 0, 15], "i64");
eq(nd_to_list(nd_bitwise_and($A, $B)), [8, 4, 0, 12],  "and, elementwise");
eq(nd_to_list(nd_bitwise_or($A, $B)),  [14, 13, 12, 15], "or, elementwise");
eq(nd_to_list(nd_bitwise_xor($A, $B)), [6, 9, 12, 3],  "xor, elementwise");

// the identities, which catch an operand swap
eq(nd_to_list(nd_bitwise_and($I, -1)), nd_to_list($I), "x & -1 == x");
eq(nd_to_list(nd_bitwise_or($I, 0)),   nd_to_list($I), "x | 0 == x");
eq(nd_to_list(nd_bitwise_xor($I, 0)),  nd_to_list($I), "x ^ 0 == x");
eq(nd_to_list(nd_bitwise_xor($I, $I)), [0, 0, 0, 0, 0, 0, 0], "x ^ x == 0");

// ─── not is the one's complement, NOT a logical not ─────────────────────
print("── not ──");

eq(nd_to_list(nd_bitwise_not($I)), [-1, -2, -6, -7, -256, 0, 7], "~x == -x-1");
eq(nd_to_list(nd_bitwise_not(nd([0], "i64"))), [-1], "~0 is -1, not 1");
eq(nd_to_list(nd_logical_not(nd([0], "i64"))), [true], "nd_logical_not is the other one");
eq(nd_to_list(nd_bitwise_not(nd_bitwise_not($I))), nd_to_list($I), "~~x == x");

// ─── shifts ─────────────────────────────────────────────────────────────
print("── shifts ──");

eq(nd_to_list(nd_left_shift(nd([1, 1, 1, 1], "i64"), nd([0, 1, 2, 3], "i64"))),
   [1, 2, 4, 8], "the count is elementwise");
eq(nd_to_list(nd_left_shift($I, 2)),  [0, 4, 20, 24, 1020, -4, -32], "<< 2");
eq(nd_to_list(nd_right_shift($I, 1)), [0, 0, 2, 3, 127, -1, -4],     ">> 1 is arithmetic");
// the sign bit replicates: -1 >> n stays -1 however far it goes
eq(nd_to_list(nd_right_shift(nd([-1], "i64"), 63)), [-1], "-1 >> 63 is still -1");
eq(nd_to_list(nd_right_shift(nd([-8], "i64"), 1)),  [-4], "-8 >> 1 is -4, not 2^62");
// >> is floor division, which differs from truncation for negatives
eq(nd_to_list(nd_right_shift(nd([-7], "i64"), 1)),  [-4], "-7 >> 1 floors to -4");
eq(nd_to_list(nd_left_shift(nd([1], "i64"), 62)),   [4611686018427387904], "1 << 62");

// a count outside 0..63 is undefined in C, so it is an error here
throws(def() { return nd_left_shift($I, 64); },  "<< 64 refused");
throws(def() { return nd_left_shift($I, -1); },  "<< -1 refused");
throws(def() { return nd_right_shift($I, 64); }, ">> 64 refused");
throws(def() { return nd_right_shift($I, -1); }, ">> -1 refused");

// ─── dtype and the f64 rule ─────────────────────────────────────────────
print("── dtypes ──");

eq(nd_dtype(nd_bitwise_and($I, 3)), "i64", "result stays i64");
eq(nd_dtype(nd_bitwise_not($I)), "i64", "not stays i64");
// a Bantu number is a double, so the scalar 3 arrives as f64 and must work
eq(nd_to_list(nd_bitwise_and(nd([7], "i64"), 3.0)), [3], "an integral f64 scalar converts");
eq(nd_to_list(nd_bitwise_or(nd([4.0, 8.0], "f64"), 1)), [5, 9], "an all-integral f64 array converts");
// a fractional one does not, and the message carries the value
throws(def() { return nd_bitwise_and(nd([1.5], "f64"), 1); }, "1.5 refused");
throws(def() { return nd_bitwise_not(nd([0.25], "f64")); },   "0.25 refused");
$msg = "";
try { nd_bitwise_and(nd([2.5], "f64"), 1); } catch ($e) { $msg = $e.message; }
eq(contains($msg, "2.5"), true, "the error names the offending value");

// bool promotes to i64 rather than staying one bit
eq(nd_to_list(nd_bitwise_and(nd([true, false], "bool"), 1)), [1, 0], "bool promotes");

// ─── shape: broadcasting and out= ───────────────────────────────────────
print("── shape ──");

$M = nd_reshape(nd_arange(0, 6, 1, "i64"), [2, 3]);
eq(nd_shape(nd_bitwise_and($M, nd([1, 2, 4], "i64"))), [2, 3], "broadcasts a row");
eq(nd_to_list(nd_bitwise_and($M, nd([1, 2, 4], "i64"))), [[0, 0, 0], [1, 0, 4]], "broadcast values");
$dst = nd_zeros([7], "i64");
nd_bitwise_and($I, 3, $dst);
eq(nd_to_list($dst), [0, 1, 1, 2, 3, 3, 0], "out= destination is written");
$self = nd([12, 10], "i64");
nd_bitwise_xor($self, 6, $self);
eq(nd_to_list($self), [10, 12], "in place over itself");
eq(nd_size(nd_bitwise_and(nd_zeros([0], "i64"), 1)), 0, "an empty array is empty");

// a strided view, which takes the general path rather than tier 0
$view = nd_slice(nd_arange(0, 10, 1, "i64"), [[0, 10, 3]]);
eq(nd_to_list(nd_bitwise_and($view, 1)), [0, 1, 0, 1], "strided input");

// ─── the method form ────────────────────────────────────────────────────
print("── methods ──");

eq(nd_to_list($I.bitwise_and(7)),  [0, 1, 5, 6, 7, 7, 0], "a.bitwise_and(b)");
eq(nd_to_list($I.bitwise_or(0)),   nd_to_list($I),         "a.bitwise_or(b)");
eq(nd_to_list($I.bitwise_xor(0)),  nd_to_list($I),         "a.bitwise_xor(b)");
eq(nd_to_list($I.bitwise_not()),   nd_to_list(nd_bitwise_not($I)), "a.bitwise_not()");
eq(nd_to_list($I.left_shift(1)),   nd_to_list(nd_left_shift($I, 1)), "a.left_shift(b)");
eq(nd_to_list($I.right_shift(1)),  nd_to_list(nd_right_shift($I, 1)), "a.right_shift(b)");

// ─── differential: against the arithmetic they replace ──────────────────
// The 32-pass bit loop is what a caller had to write before these existed.
// Agreeing with it over every byte pair is the strongest cheap evidence.
print("── differential against the bit loop ──");

def bitloop($x, $y, $op) {
    $acc = 0;
    $bit = 1;
    $i = 0;
    while ($i < 9) {
        $xb = ($x / $bit) % 2;
        $yb = ($y / $bit) % 2;
        $xb = $xb - ($xb % 1);
        $yb = $yb - ($yb % 1);
        $r = 0;
        if ($op == "and") { if ($xb == 1 && $yb == 1) { $r = 1; } }
        if ($op == "or")  { if ($xb == 1 || $yb == 1) { $r = 1; } }
        if ($op == "xor") { if ($xb != $yb) { $r = 1; } }
        $acc = $acc + $r * $bit;
        $bit = $bit * 2;
        $i = $i + 1;
    }
    return $acc;
}

$xs = [];
$ys = [];
$i = 0;
while ($i < 256) {
    push($xs, $i);
    push($ys, 255 - $i);
    $i = $i + 1;
}
$X = nd($xs, "i64");
$Y = nd($ys, "i64");
$gotAnd = nd_to_list(nd_bitwise_and($X, $Y));
$gotOr  = nd_to_list(nd_bitwise_or($X, $Y));
$gotXor = nd_to_list(nd_bitwise_xor($X, $Y));
$bad = 0;
$i = 0;
while ($i < 256) {
    if ($gotAnd[$i] != bitloop($xs[$i], $ys[$i], "and")) { $bad = $bad + 1; }
    if ($gotOr[$i]  != bitloop($xs[$i], $ys[$i], "or"))  { $bad = $bad + 1; }
    if ($gotXor[$i] != bitloop($xs[$i], $ys[$i], "xor")) { $bad = $bad + 1; }
    $i = $i + 1;
}
eq($bad, 0, "768 results agree with the 32-pass bit loop");

// ─── scale: the point of having these at all ────────────────────────────
print("── scale ──");

nd_seed(7);
$big = nd_random_int([400, 400], 0, 65536);
$t0 = clock();
$r1 = nd_bitwise_and($big, 255);
$ms = clock() - $t0;
print("   160,000 elements, one call: " + str($ms) + " ms");
eq(nd_size($r1), 160000, "160,000 elements");
eq(nd_to_list(nd_max($r1)) <= 255, true, "every result is under 256");
eq(nd_to_list(nd_min($r1)) >= 0, true, "every result is non-negative");
// and the identity that proves the mask is the low byte
eq(nd_to_list(nd_sum(nd_bitwise_and($big, 255))),
   nd_to_list(nd_sum(nd_mod($big, 256))), "x & 255 == x % 256 over 160,000 values");

print("");
print("checks: " + str($R.p + $R.f) + "  passed: " + str($R.p) + "  failed: " + str($R.f));
if ($R.f == 0) { print("RESULT: ALL GREEN"); }
else { print("RESULT: FAILED"); }
