// ════════════════════════════════════════════════════════════════════════
//  numba_array_test.b — the NdArray core: buffers, zero-copy views, creation,
//  introspection, shape operations, element access, and the security limits.
//
//  The point of this phase is the VIEW. reshape, transpose, slice, flip and
//  broadcast all hand back a new array that shares the original's buffer, so
//  they cost nothing regardless of size -- that property is what separates an
//  array library from a list of lists, and it is what the `nd_base_id` and
//  write-through assertions below actually prove.
//
//  The other half is that a bad argument must never take the process down.
//  A shape product that overflows size_t would allocate a tiny buffer that
//  every later kernel writes past -- a heap overflow reachable from one line of
//  script -- and an unbounded allocation inside a sua handler kills a server.
//  Both are gated here, along with every builtin's wrong-type/negative/NaN path.
//
//  Run:  bantu run tests/numba_array_test.b
// ════════════════════════════════════════════════════════════════════════

$R = {"pass": 0, "fail": 0};

def eq($got, $want, $name) {
    if ($got == $want) {
        $R.pass = $R.pass + 1;
        print("  ok    " + $name);
    } else {
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

// Assert that $fn raises, and that the message names the thing that was wrong.
// A message is part of the contract here: "index out of range" with no axis,
// no extent and no offending value is a support burden, not an error report.
def raises($fn, $needle, $name) {
    $threw = false;
    $msg = "";
    try {
        $fn();
    } catch ($e) {
        $threw = true;
        $msg = str($e);
    }
    if (!$threw) {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name + "  (it did NOT raise)");
    } else {
        if (contains($msg, $needle)) {
            $R.pass = $R.pass + 1;
            print("  ok    " + $name);
        } else {
            $R.fail = $R.fail + 1;
            print("  FAIL  " + $name);
            print("          message did not mention \"" + $needle + "\":");
            print("          " + $msg);
        }
    }
}

print("-- the layer is present --");
ok(has_native("ndarray"), "has_native(\"ndarray\") is true");

print("");
print("-- creation --");
$a = nd_arange(0, 12, null);
eq(nd_size($a), 12, "arange produces the right count");
eq(nd_dtype($a), "i64", "an integral arange infers i64");
eq(nd_get($a, 0), 0, "first element");
eq(nd_get($a, 11), 11, "last element");
eq(nd_dtype(nd_arange(0, 2, 0.5)), "f64", "a fractional step gives f64");
eq(nd_size(nd_arange(0, 10, 3)), 4, "a step of 3 over [0,10) gives 4");
eq(nd_size(nd_arange(5, 0, null)), 0, "an empty range gives an empty array");

$ls = nd_linspace(0, 1, 5, null);
eq(nd_get($ls, 0), 0, "linspace starts at start");
eq(nd_get($ls, 4), 1, "linspace lands exactly on stop");
eq(nd_get($ls, 2), 0.5, "and is evenly spaced");
eq(nd_size(nd_linspace(0, 1, null, null)), 50, "linspace defaults to 50 points");

eq(str(nd_zeros([2, 2], null)), "[[0, 0], [0, 0]]  (shape=[2,2], dtype=f64)", "zeros");
eq(str(nd_ones([3], null)), "[1, 1, 1]  (shape=[3], dtype=f64)", "ones");
eq(str(nd_full([2], 7, null)), "[7, 7]  (shape=[2], dtype=f64)", "full");
eq(str(nd_eye(2, null, null)), "[[1, 0], [0, 1]]  (shape=[2,2], dtype=f64)", "eye");
eq(str(nd_identity(2, null)), "[[1, 0], [0, 1]]  (shape=[2,2], dtype=f64)", "identity");
eq(nd_dtype(nd_zeros([2], "i64")), "i64", "an explicit dtype is honoured");

print("");
print("-- from a nested Bantu list, and back --");
$m2 = nd([[1, 2], [3, 4]], null);
eq(str(nd_shape($m2)), "[2, 2]", "shape is inferred from the nesting");
eq(nd_dtype($m2), "i64", "whole numbers infer i64");
eq(nd_dtype(nd([[1.5, 2]], null)), "f64", "any fractional value makes it f64");
eq(nd_dtype(nd([true, false], null)), "bool", "all booleans infer bool");
eq(str(nd_to_list($m2)), "[[1, 2], [3, 4]]", "to_list round-trips the nesting");
eq(nd_ndim(nd(5, null)), 0, "a bare number makes a 0-d array");
eq(nd_get(nd(5, null), []), 5, "and it holds its value");

print("");
print("-- introspection --");
$m = nd_reshape($a, [3, 4]);
eq(str(nd_shape($m)), "[3, 4]", "shape");
eq(nd_ndim($m), 2, "ndim");
eq(nd_size($m), 12, "size");
eq(str(nd_strides($m)), "[4, 1]", "C-contiguous strides");
eq(nd_itemsize($m), 8, "i64 is 8 bytes");
eq(nd_nbytes($m), 96, "nbytes = size x itemsize");
ok(nd_is_contiguous($m, null), "a fresh reshape is C-contiguous");
ok(nd_writable($m), "and writable");

print("");
print("-- views share memory: the whole point --");
$t = nd_T($m);
eq(str(nd_shape($t)), "[4, 3]", "transpose swaps the shape");
eq(str(nd_strides($t)), "[1, 4]", "and the strides, without touching data");
ok(nd_base_id($m) == nd_base_id($t), "transpose returns a VIEW of the same buffer");
ok(nd_base_id($m) == nd_base_id(nd_reshape($m, [12])), "so does reshape");
ok(nd_base_id($m) == nd_base_id(nd_ravel($m)), "so does ravel");
ok(nd_base_id($m) == nd_base_id(nd_slice($m, [[1, 3, null], null])), "so does slice");
ok(nd_base_id($m) == nd_base_id(nd_flip($m, 0)), "so does flip");
ok(!nd_is_contiguous($t, null), "a transposed view is not C-contiguous");
ok(nd_is_contiguous($t, "F"), "but it is F-contiguous");

// The assertion that matters: a write through one handle is visible through
// another. If any of these silently copied, this would fail.
nd_set($m, [1, 2], 99);
eq(nd_get($t, [2, 1]), 99, "a write through the base is seen through the transpose");
nd_set($t, [0, 0], 77);
eq(nd_get($m, [0, 0]), 77, "and a write through the view is seen in the base");
ok(nd_is_view($t), "the view reports itself as a view");
ok(!nd_is_view($m2), "a freshly built array does not");

// A copy must NOT share.
$c = nd_copy($m);
ok(nd_base_id($c) != nd_base_id($m), "nd_copy allocates its own buffer");
nd_set($c, [0, 0], 1234);
eq(nd_get($m, [0, 0]), 77, "so writing to the copy leaves the original alone");

print("");
print("-- a view keeps its base alive --");
// The base handle goes out of scope; the view must still be readable. If the
// buffer were freed with the base this would read freed memory.
def makeView() {
    $base = nd_arange(0, 100, null);
    return nd_slice($base, [[10, 20, null]]);
}
$kept = makeView();
eq(nd_get($kept, 0), 10, "a view outliving its base still reads correctly");
eq(nd_get($kept, 9), 19, "all the way to its end");

print("");
print("-- shape operations --");
eq(str(nd_reshape(nd_arange(0, 6, null), [2, -1])), "[[0, 1, 2], [3, 4, 5]]  (shape=[2,3], dtype=i64)",
   "reshape infers a single -1");
eq(str(nd_shape(nd_expand_dims(nd_arange(0, 3, null), 0))), "[1, 3]", "expand_dims");
eq(str(nd_shape(nd_squeeze(nd_zeros([1, 3, 1], null), null))), "[3]", "squeeze drops every 1-axis");
eq(str(nd_shape(nd_squeeze(nd_zeros([1, 3, 1], null), 0))), "[3, 1]", "squeeze of one named axis");
eq(str(nd_shape(nd_swapaxes(nd_zeros([2, 3, 4], null), 0, 2))), "[4, 3, 2]", "swapaxes");
eq(str(nd_shape(nd_moveaxis(nd_zeros([2, 3, 4], null), 0, 2))), "[3, 4, 2]", "moveaxis");
eq(str(nd_flip(nd_arange(0, 4, null), null)), "[3, 2, 1, 0]  (shape=[4], dtype=i64)", "flip reverses");
eq(str(nd_slice(nd_arange(0, 10, null), [[1, 8, 2]])), "[1, 3, 5, 7]  (shape=[4], dtype=i64)",
   "a strided slice");
eq(str(nd_slice(nd_arange(0, 5, null), [[-2, null, null]])), "[3, 4]  (shape=[2], dtype=i64)",
   "a negative slice start counts from the end");
eq(str(nd_slice(nd_arange(0, 5, null), [[null, null, -1]])), "[4, 3, 2, 1, 0]  (shape=[5], dtype=i64)",
   "a negative step walks backwards");
eq(nd_get(nd_arange(0, 5, null), -1), 4, "a negative index counts from the end");

// flatten always copies; ravel is a view when it can be.
$f = nd_flatten($m);
ok(nd_base_id($f) != nd_base_id($m), "flatten always copies");
ok(nd_base_id(nd_ravel(nd_T($m))) != nd_base_id($m),
   "ravel of a non-contiguous view has to copy");

print("");
print("-- broadcasting is a stride of 0, and is read-only --");
$col = nd([[1], [2], [3]], null);
$b = nd_broadcast_to($col, [3, 4]);
eq(str(nd_shape($b)), "[3, 4]", "broadcast_to reshapes without copying");
eq(str(nd_strides($b)), "[1, 0]", "the stretched axis gets a stride of 0");
ok(nd_base_id($b) == nd_base_id($col), "and it shares the original buffer");
eq(nd_get($b, [2, 3]), 3, "every column reads the same element");
ok(!nd_writable($b), "the result is NOT writable");
raises(def() { nd_set($b, [0, 0], 5); }, "read-only",
   "writing through a broadcast view raises rather than corrupting");

print("");
print("-- casts --");
eq(nd_dtype(nd_astype($m, "f64")), "f64", "astype changes dtype");
eq(nd_get(nd_astype(nd([1.7, 2.2], null), "i64"), 0), 2, "f64 -> i64 rounds to nearest");
eq(nd_get(nd_astype(nd([0, 3], null), "bool"), 1), true, "nonzero -> true");
ok(nd_base_id(nd_ascontiguous($m)) == nd_base_id($m),
   "ascontiguous on an already-contiguous array does not copy");
ok(nd_base_id(nd_ascontiguous(nd_T($m))) != nd_base_id($m),
   "but it does copy a transposed view");

print("");
print("-- random is reproducible --");
nd_seed(42);
$r1 = nd_to_list(nd_random_uniform([4], null, null));
nd_seed(42);
$r2 = nd_to_list(nd_random_uniform([4], null, null));
eq(str($r1), str($r2), "the same seed gives the same stream");
nd_seed(43);
$r3 = nd_to_list(nd_random_uniform([4], null, null));
ok(str($r1) != str($r3), "a different seed gives a different stream");
$u = nd_random_uniform([1000], 5, 6);
ok(nd_get($u, 0) >= 5 && nd_get($u, 0) < 6, "uniform respects its bounds");
eq(nd_dtype(nd_random_int([4], 0, 10)), "i64", "random_int is i64");
eq(nd_size(nd_random_normal([7], 0, 1)), 7, "normal fills an odd count correctly");

print("");
print("-- SECURITY: a bad argument raises, it never corrupts or kills --");
// A shape product that overflows size_t would wrap to a small number, allocate
// a tiny buffer, and let every later kernel write past it.
raises(def() { nd_zeros([4194304, 4194304, 4194304], null); }, "overflow",
   "a shape whose product overflows size_t is refused");
// An unbounded allocation inside a sua handler kills a server.
raises(def() { nd_zeros([1000000000000000], null); }, "exceeds the limit",
   "an allocation past the ceiling is refused");
raises(def() { nd_zeros([-1], null); }, "negative",
   "a negative dimension is refused");
raises(def() { nd_zeros([2.5], null); }, "whole number",
   "a fractional dimension is refused");
raises(def() { nd_zeros(["x"], null); }, "must be a number",
   "a non-numeric dimension is refused");
raises(def() { nd_zeros([2], "f128"); }, "unknown dtype",
   "an unknown dtype is refused by name");
raises(def() { nd_get($m, [99, 0]); }, "out of range",
   "an out-of-range index names the axis and its extent");
raises(def() { nd_get($m, [0]); }, "dimensional",
   "too few indices is refused");
raises(def() { nd_reshape($m, [5, 5]); }, "cannot reshape",
   "a reshape that does not conserve elements is refused");
raises(def() { nd_reshape($m, [-1, -1]); }, "only one",
   "two inferred dimensions are refused");
raises(def() { nd_transpose($m, [0, 0]); }, "repeated",
   "a repeated transpose axis is refused");
raises(def() { nd_broadcast_to(nd_zeros([3], null), [4]); }, "cannot broadcast",
   "an impossible broadcast names both shapes");
raises(def() { nd_arange(0, 10, 0); }, "step cannot be zero",
   "a zero step is refused");
raises(def() { nd([[1, 2], [3]], null); }, "ragged",
   "a ragged nested list is refused");
raises(def() { nd_slice($m, [[0, 2, 0]]); }, "step cannot be zero",
   "a zero slice step is refused");
raises(def() { nd_squeeze(nd_zeros([2, 3], null), 0); }, "cannot be squeezed",
   "squeezing a non-unit axis is refused");
raises(def() { nd_random_int([2], 5, 5); }, "greater than",
   "an empty random_int range is refused");
raises(def() { nd_get(nd_zeros([2], null), null); }, "must be a number",
   "a null index is refused");
raises(def() { nd_shape(42); }, "expected an array",
   "passing a non-array where an array is required is refused");

print("");
print("-- boundaries that are legal, not errors --");
eq(nd_size(nd_zeros([0], null)), 0, "a zero-length array is fine");
eq(str(nd_shape(nd_zeros([0], null))), "[0]", "and keeps its shape");
eq(nd_ndim(nd_zeros([], null)), 0, "an empty shape is a 0-d scalar");
eq(nd_size(nd_zeros([], null)), 1, "which holds exactly one element");
eq(nd_size(nd_slice(nd_arange(0, 5, null), [[3, 1, null]])), 0,
   "a backwards slice with a positive step is empty, not an error");

print("");
print("-- large allocations are fast --");
$t0 = clock();
$big = nd_zeros([10000000], null);
$alloc = clock() - $t0;
print("        nd_zeros([10,000,000]): " + str($alloc) + "ms");
eq(nd_size($big), 10000000, "10M elements allocated");
eq(nd_nbytes($big), 80000000, "80 MB");
ok($alloc < 300, "allocation of 10M f64 is under 300ms");

$t0 = clock();
$v = nd_T(nd_reshape($big, [1000, 10000]));
$viewMs = clock() - $t0;
print("        reshape + transpose of 10M: " + str($viewMs) + "ms");
ok($viewMs < 50, "a view of 10M elements is effectively free (no copy)");
ok(nd_base_id($v) == nd_base_id($big), "and really is the same buffer");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
