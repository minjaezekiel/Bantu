// ════════════════════════════════════════════════════════════════════════
//  lang_class_include_test.b — a class from an included file can be both
//  INHERITED FROM and INSTANTIATED.
//
//  It could not. `extends` resolves a parent name against the global class
//  registry, but `new X()` resolved it against the SCOPE -- and a class
//  declaration binds its name in the environment where the declaration
//  appears, which for an included file is that file's own. So:
//
//      include "fixtures/classlib.b" as lib;
//      class Mine extends Shape { ... }    // worked
//      $s = new Shape(3);                  // Undefined variable: Shape
//
//  A class visible enough to inherit from, and not visible enough to build.
//  Found while writing bkit's estimator protocol, where users are expected to
//  do exactly both: `class Ridge extends Estimator` and, in a test,
//  `new Estimator()`.
//
//  The fix gives both one rule: a name written after `new` is resolved against
//  the class registry as well as the scope. It is gated on the keyword, so a
//  bare `X()` still resolves through the scope alone and the factory pattern
//  (`def Linear()` shadowing `class Linear`) behaves exactly as before.
//
//  Run:  bantu run tests/lang_class_include_test.b
// ════════════════════════════════════════════════════════════════════════

include "fixtures/classlib.b" as lib;

$R = {"pass": 0, "fail": 0};
def ok($cond, $name) {
    if ($cond) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $name); }
}
def eq($got, $want, $name) {
    if ($got == $want) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else { $R.fail = $R.fail + 1;
           print("  FAIL  " + $name + "   got " + str($got) + ", want " + str($want)); }
}

print("");
print("  classes across an include");
print("  ════════════════════════════════════════");

// ── 1. the route that always worked ─────────────────────────────────────────
print("");
print("-- inheriting --");
class Triangle extends Shape {
    def init() { super(3); }
    def describe() { return "Triangle"; }
}
$t = new Triangle();
eq($t.describe(), "Triangle", "a local class can extend an included one");
eq($t.sides(), 3, "and inherits its state through super()");

// ── 2. the route that did not ───────────────────────────────────────────────
print("");
print("-- instantiating --");
$s = new Shape(7);
eq($s.describe(), "Shape(7)", "an included class can be instantiated directly");
eq($s.sides(), 7, "with its constructor arguments");
$sq = new Square();
eq($sq.describe(), "Square", "including a SUBCLASS defined in the same file");
eq($sq.sides(), 4, "and its chained super() ran");

// ── 3. the factory route, which must be unchanged ───────────────────────────
print("");
print("-- the factory pattern still works --");
$f = lib.makeShape(9);
eq($f.describe(), "Shape(9)", "a module function returning an instance");
eq($f.sides(), 9, "with its argument");

// ── 4. what must NOT have changed ───────────────────────────────────────────
// The fix is gated on the `new` keyword. A bare call to a name that is only a
// class must still be an undefined-variable error, because the factory pattern
// depends on a function of the same name shadowing the class -- and if a bare
// `Shape(3)` silently constructed, a typo'd function name would build an
// object instead of failing.
print("");
print("-- a bare call is still resolved through the scope alone --");
$bare = false;
try { $x = Shape(3); } catch ($e) { $bare = true; }
ok($bare, "a bare Shape(3), with no `new`, does not construct");

// An unknown class after `new` still raises, and names the problem.
$unknown = false;
$msg = "";
try { $y = new NoSuchClass(); } catch ($e) { $unknown = true; $msg = str($e); }
ok($unknown, "new on an unknown name still raises");
ok(indexOf($msg, "NoSuchClass") >= 0, "and names it");

// A local class is unaffected by any of this.
class Local { def init() { this.v_ = 42; } def get() { return this.v_; } }
eq(new Local().get(), 42, "a class in THIS file is unaffected");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
