// ════════════════════════════════════════════════════════════════════════
//  lang_inherit_test.b — class inheritance and chained super().
//
//  A three-level hierarchy with chained super() SEGFAULTED the interpreter:
//
//      class Base       { def init($n) { this.n = $n; } }
//      class Child      extends Base  { def init($n) { super($n); } }
//      class GrandChild extends Child { def init($n) { super($n); } }
//      $g = new GrandChild(3);        // exit 139, no output at all
//
//  Two levels worked, which is how it survived. evalSuper resolved the parent
//  from currentClassName_ but never SET it while running the parent
//  constructor, so super() inside Child::init still resolved "GrandChild",
//  found its parent Child, and called Child::init again -- unbounded
//  recursion until the stack died. No error, no message, no partial output.
//
//  nn.Module hierarchies are naturally three deep, which is why btorch found
//  this. The depth cases below are the regression.
//
//  Run:  bantu run tests/lang_inherit_test.b
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

print("== two levels (this always worked -- guarding it) ==");
class A1 { def init($n) { this.n = $n; this.from_a = "a"; } def who() { return "A1"; } }
class B1 extends A1 { def init($n) { super($n); this.from_b = "b"; } def who() { return "B1"; } }
$b = new B1(5);
eq($b.n, 5, "2-level: the parent constructor ran and set n");
eq($b.from_a, "a", "2-level: the parent's own field is present");
eq($b.from_b, "b", "2-level: the child's field is present");
eq($b.who(), "B1", "2-level: the override wins");

print("");
print("== THREE levels: this is the case that segfaulted ==");
class A2 { def init($n) { this.n = $n; this.a = 1; } def who() { return "A2"; } }
class B2 extends A2 { def init($n) { super($n); this.b = 2; } def who() { return "B2"; } }
class C2 extends B2 { def init($n) { super($n); this.c = 3; } def who() { return "C2"; } }
$c = new C2(9);
eq($c.n, 9, "3-level: the grandparent constructor ran");
eq($c.a, 1, "3-level: the grandparent's field is present");
eq($c.b, 2, "3-level: the parent's field is present");
eq($c.c, 3, "3-level: the child's field is present");
eq($c.who(), "C2", "3-level: the deepest override wins");

print("");
print("== FOUR levels, and every constructor in the chain runs exactly once ==");
// Each level appends to a list, so running twice (the old recursion) or not at
// all would both show up here.
class A3 { def init() { this.trace = ["A3"]; } }
class B3 extends A3 { def init() { super(); this.trace = push(this.trace, "B3"); } }
class C3 extends B3 { def init() { super(); this.trace = push(this.trace, "C3"); } }
class D3 extends C3 { def init() { super(); this.trace = push(this.trace, "D3"); } }
$d = new D3();
eq(str($d.trace), str(["A3", "B3", "C3", "D3"]),
   "4-level: base-first order, each constructor exactly once");
eq(len($d.trace), 4, "4-level: four constructors ran, not five and not three");

print("");
print("== a deep chain: 8 levels ==");
class L0 { def init() { this.depth = 0; } }
class L1 extends L0 { def init() { super(); this.depth = this.depth + 1; } }
class L2 extends L1 { def init() { super(); this.depth = this.depth + 1; } }
class L3 extends L2 { def init() { super(); this.depth = this.depth + 1; } }
class L4 extends L3 { def init() { super(); this.depth = this.depth + 1; } }
class L5 extends L4 { def init() { super(); this.depth = this.depth + 1; } }
class L6 extends L5 { def init() { super(); this.depth = this.depth + 1; } }
class L7 extends L6 { def init() { super(); this.depth = this.depth + 1; } }
$l = new L7();
eq($l.depth, 7, "8-level chain: every super() ran once, depth is 7");

print("");
print("== virtual dispatch reaches the deepest override from any level ==");
class V0 {
    def init() { this.tag = "v"; }
    def name() { return "V0"; }
    def describe() { return "I am " + this.name(); }     // defined at the TOP
}
class V1 extends V0 { def init() { super(); } def name() { return "V1"; } }
class V2 extends V1 { def init() { super(); } def name() { return "V2"; } }
class V3 extends V2 { def init() { super(); } }          // no override here
$v3 = new V3();
eq($v3.name(), "V2", "a level with no override inherits the nearest one");
eq($v3.describe(), "I am V2",
   "a method defined on the base calling this.name() reaches the deepest override");
$v1 = new V1();
eq($v1.describe(), "I am V1", "same base method, different instance, different override");

print("");
print("== a child that does NOT call super() still gets the parent's methods ==");
class P0 { def init($n) { this.n = $n; } def twice() { return this.n * 2; } }
class P1 extends P0 { def init($n) { this.n = $n * 10; } }    // no super()
$p1 = new P1(3);
eq($p1.n, 30, "no super(): the child's own constructor body ran");
eq($p1.twice(), 60, "no super(): inherited methods still work");

print("");
print("== a child with NO constructor inherits the parent's ==");
class Q0 { def init($n) { this.n = $n; this.set_by = "Q0"; } }
class Q1 extends Q0 { def extra() { return 42; } }
$q = new Q1(11);
eq($q.n, 11, "no child constructor: the parent's ran with the arguments");
eq($q.set_by, "Q0", "no child constructor: the parent set its field");
eq($q.extra(), 42, "no child constructor: the child's methods still exist");
// and three levels of that: the constructor is TWO levels up. This read
// `parentClass->methods.find(...)` -- the immediate parent's own table only --
// so it found nothing and every field stayed null, silently.
class Q2 extends Q1 { }
$q2 = new Q2(12);
eq($q2.n, 12, "3 levels, ctor 2 levels up: the base's constructor ran");
eq($q2.set_by, "Q0", "3 levels, ctor 2 levels up: its field is set");
eq($q2.extra(), 42, "3 levels: the middle class's methods are still inherited");
class Q3 extends Q2 { }
eq((new Q3(13)).n, 13, "4 levels, ctor 3 levels up");

// The inherited constructor may itself call super(). It has to run under the
// name of the class that DEFINES it, or it would find its own class as the
// parent and recurse -- the same crash this release fixed, reached by a
// different route.
class S0 { def init($n) { this.base = $n; } }
class S1 extends S0 { def init($n) { super($n); this.mid = $n * 2; } }
class S2 extends S1 { }                 // no ctor: inherits S1's, which supers
$s2 = new S2(5);
eq($s2.base, 5, "an INHERITED constructor's own super() reaches the grandparent");
eq($s2.mid, 10, "and the inherited constructor's own body ran");
class S3 extends S2 { }                 // two levels of no-ctor over a super()
eq((new S3(6)).base, 6, "two no-ctor levels above a super()-calling constructor");
eq((new S3(6)).mid, 12, "and its body still ran exactly once");

print("");
print("== super() passes arguments through, and they are the CHILD's ==");
class R0 { def init($a, $b) { this.sum = $a + $b; } }
class R1 extends R0 { def init($a, $b) { super($a * 2, $b * 3); } }
class R2 extends R1 { def init($a, $b) { super($a + 1, $b + 1); } }
$r = new R2(1, 1);
// R2(1,1) -> R1(2,2) -> R0(4,6) -> sum 10
eq($r.sum, 10, "arguments transform correctly down a 3-level chain");

print("");
print("== currentClassName_ is restored, so super() still resolves afterwards ==");
// The bug was a stale class name. After constructing a deep hierarchy, a
// SIBLING hierarchy must still resolve its own super() correctly -- if the
// name leaked, this would build the wrong parent or recurse.
$c2 = new C2(1);
$b2 = new B2(2);
eq($b2.n, 2, "a 2-level build after a 3-level build is still correct");
ok($b2.b == 2, "and its own field is set");
$c3 = new C2(7);
eq($c3.n, 7, "a 3-level build after a 2-level build is still correct");
eq($c3.who(), "C2", "and its dispatch is unaffected");

print("");
print("== a constructor that THROWS does not leave the resolver broken ==");
class T0 { def init($n) { this.n = $n; } }
class T1 extends T0 { def init($n) { super($n); if ($n < 0) { throw "negative"; } } }
class T2 extends T1 { def init($n) { super($n); this.ok = true; } }
$caught = false;
try { $bad = new T2(0 - 5); } catch ($e) { $caught = true; }
ok($caught, "a throwing constructor in the chain is catchable");
// and the very next construction must still work
$good = new T2(4);
eq($good.n, 4, "after a throw mid-chain, the next build is still correct");
ok($good.ok, "and it completed fully");

print("");
print("== repetition: many instances of a deep class, no leak of state ==");
$i = 0;
$allok = true;
while ($i < 2000) {
    $x = new D3();
    if (len($x.trace) != 4) { $allok = false; }
    $i = $i + 1;
}
ok($allok, "2000 instances of a 4-level class each ran its chain exactly once");

print("");
print("== instances of deep classes are independent ==");
$d1 = new D3();
$d2 = new D3();
$d1.trace = push($d1.trace, "mutated");
eq(len($d2.trace), 4, "mutating one instance does not touch another");

print("");
print("== type() and field access on deep instances ==");
eq(type($c), "instance", "a 3-level instance reports type 'instance'");
ok($c.who() == "C2", "method call on a deep instance");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
