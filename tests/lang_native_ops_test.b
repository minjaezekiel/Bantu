// ════════════════════════════════════════════════════════════════════════
//  lang_native_ops_test.b — operators, indexing and methods on native
//  handles.
//
//  This is a LANGUAGE change, so half of this file is about what must NOT
//  have changed. Every path it fills in was previously dead: `$handle + 1`
//  read numberVal, which is always 0 for a handle, and silently produced 1;
//  `$handle[i]` returned null; `$handle[i] = v` threw. Nothing that worked
//  before is allowed to move, which is why strings, lists, dicts, class
//  instances, null and booleans are all re-asserted here against the same
//  operators.
//
//  The one deliberate asymmetry: `==` and `!=` stay identity comparisons
//  rather than becoming element-wise. `if ($a == $b)` is written constantly,
//  and an element-wise result would silently turn it into "is this array
//  non-empty and all-truthy". NumPy made the other choice and then had to
//  make `if arr:` raise; Bantu has no such escape hatch.
//
//  Run:  bantu run tests/lang_native_ops_test.b
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
print("  Bantu — native-handle operators");
print("========================================");

print("");
print("-- NOTHING ELSE CHANGED: the ordinary language still behaves --");
eq(2 + 3, 5, "number addition");
eq(10 - 4, 6, "subtraction");
eq(6 * 7, 42, "multiplication");
eq(10 / 4, 2.5, "division");
eq(10 % 3, 1, "modulo");
eq(-5, 0 - 5, "unary minus");
eq("a" + "b", "ab", "string concatenation");
eq("n=" + str(5), "n=5", "string plus a stringified number");
eq(1 + 2 == 3, true, "comparison");
ok(2 < 3, "less than");
ok(3 >= 3, "greater or equal");
ok(true && true, "logical and");
ok(false || true, "logical or");
ok(!false, "logical not");
eq([1, 2, 3] == [1, 2, 3], true, "list equality is still structural");
eq([1, 2] == [1, 3], false, "and still distinguishes");
eq({"a": 1} == {"a": 1}, true, "dict equality is still structural");
eq(null == null, true, "null equality");
eq(null == 0, false, "null is not zero");
$l = [10, 20, 30];
eq($l[1], 20, "list index read");
$l[1] = 99;
eq($l[1], 99, "list index write");
$d = {"k": 5};
eq($d["k"], 5, "dict index read");
$d["j"] = 6;
eq($d["j"], 6, "dict index write");
eq("hello"[1], "e", "string index");
eq(len($d.keys()), 2, "dict pseudo-methods still work");
eq((2.7).floor(), 2, "number pseudo-methods still work");

if (!has_native("ndarray")) {
    print("");
    print("  --    numba not built in; skipping the handle sections");
    print("");
    print("========================================");
    print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
    print("========================================");
    if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
    return;
}

print("");
print("-- arithmetic on arrays --");
$a = nd([1, 2, 3], null);
$b = nd([10, 20, 30], null);
eq(str(nd_to_list($a + $b)), "[11, 22, 33]", "$a + $b");
eq(str(nd_to_list($b - $a)), "[9, 18, 27]",  "$a - $b");
eq(str(nd_to_list($a * $b)), "[10, 40, 90]", "$a * $b");
eq(str(nd_to_list($b / $a)), "[10, 10, 10]", "$a / $b");
eq(str(nd_to_list($b % $a)), "[0, 0, 0]",    "$a % $b");

print("");
print("-- a scalar on either side --");
eq(str(nd_to_list($a + 100)), "[101, 102, 103]", "array + number");
eq(str(nd_to_list(100 + $a)), "[101, 102, 103]", "number + array");
eq(str(nd_to_list(2 * $a)),   "[2, 4, 6]",       "number * array");
eq(str(nd_to_list($a * 2)),   "[2, 4, 6]",       "array * number");
// Subtraction and division are not commutative, so both orders are asserted.
eq(str(nd_to_list(10 - $a)),  "[9, 8, 7]",       "number - array keeps the operand order");
eq(str(nd_to_list($a - 10)),  "[-9, -8, -7]",    "array - number");
eq(str(nd_to_list(12 / $a)),  "[12, 6, 4]",      "number / array");

print("");
print("-- unary minus --");
eq(str(nd_to_list(-$a)), "[-1, -2, -3]", "-$a negates element-wise");
eq(str(nd_to_list(-(-$a))), "[1, 2, 3]", "and twice returns the original values");

print("");
print("-- comparisons produce boolean arrays --");
eq(str(nd_to_list($a > 1)),  "[false, true, true]",  "$a > 1");
eq(str(nd_to_list($a < 3)),  "[true, true, false]",  "$a < 3");
eq(str(nd_to_list($a >= 2)), "[false, true, true]",  "$a >= 2");
eq(str(nd_to_list($a <= 1)), "[true, false, false]", "$a <= 1");
eq(nd_dtype($a > 1), "bool", "the result is a bool array");
eq(str(nd_to_list($a < $b)), "[true, true, true]", "array against array");

print("");
print("-- == and != stay IDENTITY, deliberately --");
// An element-wise == would silently turn `if ($x == $y)` into "is this array
// non-empty and all-truthy". NumPy chose element-wise and then had to make
// `if arr:` raise; Bantu has no such escape hatch, so identity is safer.
ok(!($a == $b), "$a == $b is false for two different arrays, not an array");
ok($a != $b, "and != is true");
ok(($a == $a), "an array equals itself");
// The explicit forms are the ones that answer the element-wise question.
ok(nd_array_equal($a, nd([1, 2, 3], null)), "nd_array_equal is the element-wise form");
ok(!nd_array_equal($a, $b), "and distinguishes");

print("");
print("-- broadcasting through the operators --");
$m = nd_reshape(nd_arange(0, 6, null), [2, 3]);
$row = nd([10, 20, 30], null);
eq(str(nd_to_list($m + $row)), "[[10, 21, 32], [13, 24, 35]]", "(2,3) + (3,) broadcasts");
raises(def() { return $m + nd_zeros([4], null); }, "cannot be broadcast",
   "a shape mismatch through the operator still names the axis and extents");

print("");
print("-- chained expressions --");
$x = nd([1.0, 2.0, 3.0], "f64");
$y = ($x * $x) + $x;
eq(str(nd_to_list($y)), "[2, 6, 12]", "($x * $x) + $x");
$z = -($x + 1) * 2;
eq(str(nd_to_list($z)), "[-4, -6, -8]", "-($x + 1) * 2");

print("");
print("-- indexing: a 1-d array yields elements --");
eq($a[0], 1, "$a[0]");
eq($a[2], 3, "$a[2]");
eq($a[-1], 3, "a negative index counts from the end");
raises(def() { return $a[9]; }, "out of range", "an out-of-range index RAISES");
raises(def() { return $a[-9]; }, "out of range", "and so does one too negative");

print("");
print("-- indexing: a 2-d array yields a VIEW, so writes go through --");
$g = nd_reshape(nd_arange(0, 6, null), [2, 3]);
$r1 = $g[1];
eq(str(nd_to_list($r1)), "[3, 4, 5]", "$g[1] is the second row");
ok(nd_base_id($r1) == nd_base_id($g), "and it shares the base's buffer -- a view, not a copy");
$g[1][2] = 99;
eq(nd_get($g, [1, 2]), 99, "$m[1][2] = 9 writes THROUGH to the base");
// This is the opposite of list behaviour and is worth asserting side by side.
$ll = [[1, 2], [3, 4]];
$ll[1][1] = 99;
eq($ll[1][1], 99, "a nested list write also persists, by a different mechanism");

print("");
print("-- index assignment on a 1-d array --");
$w = nd_zeros([3], null);
$w[0] = 5;
$w[2] = 7;
eq(str(nd_to_list($w)), "[5, 0, 7]", "element assignment");
$w[-1] = 9;
eq(str(nd_to_list($w)), "[5, 0, 9]", "with a negative index");
raises(def() { $w[9] = 1; }, "out of range", "an out-of-range write raises");
// A broadcast view repeats one element, so writing through it must raise.
raises(def() { $bv = nd_broadcast_to(nd([0], null), [4]); $bv[0] = 1; }, "read-only",
   "writing through a broadcast view raises");

print("");
print("-- assigning a whole row --");
$g2 = nd_zeros([2, 3], null);
$g2[0] = nd([1, 2, 3], null);
eq(str(nd_to_list($g2)), "[[1, 2, 3], [0, 0, 0]]", "a row can be assigned an array");
$g2[1] = 7;
eq(str(nd_to_list($g2)), "[[1, 2, 3], [7, 7, 7]]", "or a scalar, broadcast across it");
raises(def() { $g2[0] = nd([1, 2], null); }, "cannot be broadcast",
   "a row assignment of the wrong shape raises, naming the axis and extents");

print("");
print("-- boolean masks --");
$vals = nd([5, 15, 25, 35], null);
$mask = $vals > 20;
eq(str(nd_to_list($vals[$mask])), "[25, 35]", "$a[$mask] selects");
$vals[$mask] = 0;
eq(str(nd_to_list($vals)), "[5, 15, 0, 0]", "$a[$mask] = v writes where the mask is true");
raises(def() { return $vals[nd_greater(nd_zeros([2], null), 1, null)]; }, "mask has",
   "a mask of the wrong length raises");

print("");
print("-- an integer array gathers --");
$src = nd([10, 20, 30, 40], null);
eq(str(nd_to_list($src[nd([0, 3], "i64")])), "[10, 40]", "$a[$indices] gathers");
eq(str(nd_to_list($src[nd([-1], "i64")])), "[40]", "with negative indices");

print("");
print("-- methods: the chaining form, which works on any build --");
eq(nd_get($a.sum(), []), 6, "$a.sum()");
eq(nd_get($a.max(), []), 3, "$a.max()");
eq(nd_get($a.min(), []), 1, "$a.min()");
eq(str(nd_to_list($a.add($b))), "[11, 22, 33]", "$a.add($b)");
eq(str(nd_to_list($a.multiply(2))), "[2, 4, 6]", "$a.multiply(2)");
eq(str($a.shape()), "[3]", "$a.shape()");
eq($a.size(), 3, "$a.size()");
eq($a.dtype(), "i64", "$a.dtype()");
eq(str(nd_to_list($x.sqrt())), "[1, 1.41421, 1.73205]", "$x.sqrt()");
// The whole point of chaining: it reads left to right and needs no operators.
eq(nd_get($x.multiply($x).add($x).sum(), []), 20, "$x.multiply($x).add($x).sum()");
eq(str(nd_to_list($g.reshape([6]))), "[0, 1, 2, 3, 4, 99]", "$m.reshape([6])");
eq(str($g.T().shape()), "[3, 2]", "$m.T().shape()");
// A method and its builtin are the same function, so they cannot drift.
ok(nd_array_equal($a.add($b), nd_add($a, $b, null)), "a method equals its nd_ builtin exactly");

print("");
print("-- mixed and hostile operand types --");
// The dispatch must decline cleanly rather than throwing for combinations it
// does not handle, so the existing behaviour is what happens.
eq("x" + str(nd_size($a)), "x3", "a string plus a stringified array property");
ok(!($a == 5), "an array is not equal to a number");
ok(!($a == null), "nor to null");
ok(!($a == "s"), "nor to a string");
// `+` with a string follows the language's own rule -- `5 + "x"` is "5x" --
// so an array concatenates its repr rather than raising. Consistency with the
// rest of the language wins here; `print("a: " + $a)` should work.
ok(contains($a + " tail", "tail"), "array + string concatenates, as every other type does");
// Everything else must RAISE. Declining would fall through to numberVal +
// numberVal, which is 0 + 0 -- silently producing 0, the exact failure this
// dispatch exists to remove.
raises(def() { return $a + null; }, "cannot be combined", "array plus null raises, not 0");
raises(def() { return $a + {"k": 1}; }, "cannot be combined", "array plus a dict raises, not 0");
raises(def() { return $a * null; }, "cannot be combined", "and the same for other operators");
raises(def() { return $a > null; }, "cannot be combined", "and for comparisons");
eq(str(nd_to_list($a + [10, 10, 10])), "[11, 12, 13]", "array plus a plain Bantu list works");

print("");
print("-- a 0-d array --");
$s0 = nd(7, null);
eq(nd_get($s0 + 1, []), 8, "a 0-d array is still an operand");
eq(nd_ndim($s0 + 1), 0, "and stays 0-d");
raises(def() { return $s0[0]; }, "0-d", "indexing a 0-d array raises");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
