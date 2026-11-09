// ════════════════════════════════════════════════════════════════════════
//  lang_operator_overload_test.b — operators on class instances.
//
//  Before this, `$a + $b` on two class instances returned 0. Not an error --
//  zero. There is no CLASS_INSTANCE arm in evalBinaryOp, so instances fell
//  through to `case PLUS: return Value(left.numberVal + right.numberVal)`,
//  and numberVal is always 0 for an instance. `($a + $b).x` then read as null,
//  which is how it was found.
//
//  That is the third occurrence of this exact shape in evaluator.hpp: the
//  comments there record `$handle + 1` silently giving 1 and `[1] + [2,3]`
//  silently giving 0, both since fixed. Instances were the case still open,
//  and it affected every class already shipped -- arctic's Series and
//  DataFrame, pillow's Image.
//
//  Two behaviours are deliberately preserved and asserted here, because real
//  code depends on them:
//    * `$obj + "str"` still concatenates, so printing keeps working
//    * `==` / `!=` still compare IDENTITY unless the class defines __eq__
//
//  Run:  bantu run tests/lang_operator_overload_test.b
// ════════════════════════════════════════════════════════════════════════

$R = {"pass": 0, "fail": 0};
def ok($cond, $name) {
    if ($cond) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $name); }
}
def eq($got, $want, $name) {
    if ($got == $want) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name + "  got " + str($got) + " want " + str($want));
    }
}
def raises($fn, $name) {
    try { $fn(); $R.fail = $R.fail + 1; print("  FAIL  " + $name + " (no error raised)"); }
    catch ($e) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    return null;
}

// A vector with the full operator set, including the reflected forms.
class Vec {
    def init($x) { this.x = $x; }
    def __add__($o)      { return new Vec(this.x + this._v($o)); }
    def __sub__($o)      { return new Vec(this.x - this._v($o)); }
    def __mul__($o)      { return new Vec(this.x * this._v($o)); }
    def __truediv__($o)  { return new Vec(this.x / this._v($o)); }
    def __mod__($o)      { return new Vec(this.x % this._v($o)); }
    def __neg__()        { return new Vec(0 - this.x); }
    // reflected: the instance was on the RIGHT, as in `10 - $v`
    def __radd__($o)     { return new Vec(this._v($o) + this.x); }
    def __rsub__($o)     { return new Vec(this._v($o) - this.x); }
    def __rmul__($o)     { return new Vec(this._v($o) * this.x); }
    def __rtruediv__($o) { return new Vec(this._v($o) / this.x); }
    def __lt__($o)       { return this.x <  this._v($o); }
    def __le__($o)       { return this.x <= this._v($o); }
    def __gt__($o)       { return this.x >  this._v($o); }
    def __ge__($o)       { return this.x >= this._v($o); }
    def _v($o) { if (type($o) == "instance") { return $o.x; } return $o; }
}

print("== arithmetic on two instances (used to be 0) ==");
$a = new Vec(7); $b = new Vec(3);
eq(($a + $b).x, 10, "instance + instance calls __add__");
eq(($a - $b).x,  4, "instance - instance calls __sub__");
eq(($a * $b).x, 21, "instance * instance calls __mul__");
eq(($a / $b).x, 7 / 3, "instance / instance calls __truediv__");
eq(($a % $b).x,  1, "instance % instance calls __mod__");
eq((0 - $a).x, 0 - 7, "0 - instance calls __rsub__");
eq((-$a).x,    0 - 7, "unary minus calls __neg__ (used to give -0)");

print("");
print("== instance with a plain number, both orders ==");
eq(($a + 5).x, 12, "instance + number calls __add__");
eq(($a - 5).x,  2, "instance - number calls __sub__");
eq(($a * 2).x, 14, "instance * number calls __mul__");
eq((5 + $a).x, 12, "number + instance calls __radd__");
eq((10 - $a).x, 3, "number - instance calls __rsub__ (NOT __sub__)");
eq((2 * $a).x, 14, "number * instance calls __rmul__");
eq((14 / $a).x, 2, "number / instance calls __rtruediv__");
ok((10 - $a).x != ($a - 10).x, "reflected subtraction is not the forward one");

print("");
print("== comparisons, and their reflection is the SWAPPED operator ==");
ok($a > $b,  "instance > instance calls __gt__");
ok($b < $a,  "instance < instance calls __lt__");
ok($a >= $a, "instance >= itself calls __ge__");
ok($a <= $a, "instance <= itself calls __le__");
ok($a > 5,   "instance > number");
ok(!($a > 9), "instance > number, false case");
// `5 < $a` has no __rlt__; it asks $a the same question as `$a > 5`, which is
// Python's rule and the only sane one.
ok(5 < $a,   "number < instance reflects to __gt__");
ok(9 > $a,   "number > instance reflects to __lt__");
ok(7 <= $a,  "number <= instance reflects to __ge__");

print("");
print("== chained and nested expressions evaluate left to right ==");
eq(($a + $b + $a).x, 17, "a + b + a");
eq((($a + $b) * new Vec(2)).x, 20, "(a + b) * 2");
eq(($a * 2 + $b).x, 17, "a * 2 + b");

print("");
print("== PRESERVED: a class with no dunder keeps old string behaviour ==");
class Plain { def init($v) { this.v = $v; } }
$p = new Plain(1);
// Concatenation with a string still works, so print()/str() keep working. A
// class with no __add__ must not break the moment someone prints it.
ok(type($p + "!") == "string", "instance + string still concatenates");
ok(type("!" + $p) == "string", "string + instance still concatenates");

print("");
print("== PRESERVED: == and != still compare identity without __eq__ ==");
$p2 = new Plain(1);
ok($p == $p,   "an instance equals itself");
ok(!($p == $p2), "two distinct instances with equal fields are NOT == (identity)");
ok($p != $p2,  "!= is identity too");
ok(!($p != $p), "an instance is not != itself");

print("");
print("== but __eq__ / __ne__ are honoured when defined ==");
class Money {
    def init($c) { this.c = $c; }
    def __eq__($o) { return this.c == $o.c; }
    def __ne__($o) { return this.c != $o.c; }
}
$m1 = new Money(100); $m2 = new Money(100); $m3 = new Money(5);
ok($m1 == $m2, "__eq__ makes two distinct instances with the same value equal");
ok(!($m1 == $m3), "__eq__ distinguishes different values");
ok($m1 != $m3, "__ne__ is used when defined");
ok(!($m1 != $m2), "__ne__ false for equal values");

print("");
print("== a missing dunder RAISES rather than silently computing 0 ==");
// This is the whole point: the old behaviour was `$p + $p` == 0.
raises(def() { return $p + $p; },  "instance + instance without __add__ raises");
raises(def() { return $p - $p; },  "instance - instance without __sub__ raises");
raises(def() { return $p * 2; },   "instance * number without __mul__ raises");
raises(def() { return 2 * $p; },   "number * instance without __rmul__ raises");
raises(def() { return $p / 2; },   "instance / number without __truediv__ raises");
raises(def() { return -$p; },      "unary minus without __neg__ raises");
raises(def() { return $p > $p; },  "instance > instance without __gt__ raises");
// and the message names the class and the method the author needs to add
try { $z = $p + $p; }
catch ($e) {
    ok(str($e.message) != "" , "the error has a message");
    // The message must name what to do about it, not just fail.
    $msg = str($e.message);
    ok(contains($msg, "Plain"), "the message names the class ('Plain')");
    ok(contains($msg, "__add__"), "the message names the missing method ('__add__')");
}

print("");
print("== inherited dunders work through the class chain ==");
class Vec2 extends Vec { def init($x) { super($x); } }
$v2 = new Vec2(4);
eq(($v2 + new Vec2(6)).x, 10, "a subclass inherits __add__");
eq(($v2 * 3).x, 12, "a subclass inherits __mul__");
eq((1 + $v2).x, 5, "a subclass inherits __radd__");

print("");
print("== nothing else about operators changed ==");
eq(1 + 2, 3, "number + number");
eq("a" + "b", "ab", "string + string");
eq(str([1] + [2, 3]), str([1, 2, 3]), "list + list still concatenates");
ok(true && true, "&& unchanged");
ok(true || false, "|| unchanged");
ok(!false, "! unchanged");
eq(7 % 3, 1, "modulo unchanged");
ok(1 == 1, "number == number");
ok("x" == "x", "string == string");
ok(null == null, "null == null");
// &&/|| on instances is truthiness, not a dunder -- an instance is truthy
ok($p && true, "an instance is truthy in &&");
// handles still dispatch to numba, which shares this slow path
$nd = nd_full([3], 2.0);
eq(str(nd_to_list($nd + $nd)), str([4, 4, 4]), "ndarray handle + handle still dispatches");
eq(str(nd_to_list($nd * 3)), str([6, 6, 6]), "ndarray handle * number still dispatches");
eq(str(nd_to_list(-$nd)), str([0 - 2, 0 - 2, 0 - 2]), "unary minus on a handle still dispatches");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
