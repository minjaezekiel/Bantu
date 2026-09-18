// ════════════════════════════════════════════════════════════════════════
//  lang_gc_test.b — object lifetime: reference counting and the cycle
//  collector.
//
//  Reference counting frees an object the moment its last reference drops.
//  What it cannot free is a cycle, and the shapes that make one are ordinary:
//  a tree node that knows its parent, a handler stored on the object it
//  belongs to, a helper function defined inside another function. Two of
//  those the USER never writes -- the interpreter builds the cycle itself
//  when it binds a method or defines a nested function.
//
//  Every assertion here is on gc_stats()["live"], a number, rather than on
//  RSS. RSS was too coarse to show the dict cycle at all.
//
//  Design: docs/object-lifetime-architecture.md
// ════════════════════════════════════════════════════════════════════════

$R = {"pass": 0, "fail": 0};

def ok($cond, $what) {
    if ($cond) { $R["pass"] = $R["pass"] + 1; }
    else { $R["fail"] = $R["fail"] + 1; print("  FAIL: " + $what); }
}

def eq($got, $want, $what) {
    if ($got == $want) { $R["pass"] = $R["pass"] + 1; }
    else {
        $R["fail"] = $R["fail"] + 1;
        print("  FAIL: " + $what + " -- got " + str($got) + ", wanted " + str($want));
    }
}

def hasKey($dict, $key) {
    each ($k in keys($dict)) { if ($k == $key) { return true; } }
    return false;
}

class GcNode {
    def init($v) { this.v = $v; }
    def value()  { return this.v; }
}

def makeNested() {
    $captured = [1, 2, 3];
    def helper() { return $captured; }
    return 1;
}

// Collect twice: the first pass can free objects that were only reachable
// from something the first pass itself freed.
def settle() { gc_collect(); gc_collect(); return gc_stats()["live"]; }

print("── the surface ──────────────────────────────────────────────────");

$s = gc_stats();
ok(hasKey($s, "live"),        "gc_stats has live");
ok(hasKey($s, "collections"), "gc_stats has collections");
ok(hasKey($s, "freed"),       "gc_stats has freed");
ok(hasKey($s, "threshold"),   "gc_stats has threshold");
ok(hasKey($s, "enabled"),     "gc_stats has enabled");
ok($s["live"] > 0,           "something is alive to begin with");
ok($s["threshold"] >= 10000, "the threshold has a floor");

$was = gc_enable(false);
eq(gc_stats()["enabled"], false, "gc_enable(false) turns it off");
gc_enable($was);
eq(gc_stats()["enabled"], $was, "gc_enable returns the previous setting");
// Explicit collection works whether or not the automatic one is enabled --
// the switch is a latency escape hatch, not a way to lose the fix.
gc_enable(false);
$n = new GcNode(1); $n.self = $n; $n = null;
ok(gc_collect() >= 1, "gc_collect() works with automatic collection off");
gc_enable(true);

print("── what reference counting already handles ──────────────────────");

$base = settle();
$i = 0;
while ($i < 3000) { $a = new GcNode($i); $i = $i + 1; }
// No cycle: every one of those was freed the moment the next assignment
// dropped it, without the collector being involved at all.
ok(gc_stats()["live"] - $base <= 5, "acyclic objects are freed by refcounting, not by the collector");

print("── the five cycle shapes ────────────────────────────────────────");

// 1. An object that refers to itself.
$base = settle();
$i = 0;
while ($i < 3000) { $a = new GcNode(1); $a.self = $a; $i = $i + 1; }
ok(gc_stats()["live"] - $base > 2000, "self-reference: the cycles pile up before a collection");
$freed = gc_collect();
ok($freed > 2000, "self-reference: the collector frees them");
ok(settle() - $base <= 5, "self-reference: live returns to baseline");

// 2. Two objects referring to each other -- a tree node and its parent.
$base = settle();
$i = 0;
while ($i < 3000) {
    $a = new GcNode(1); $b = new GcNode(2);
    $a.peer = $b; $b.peer = $a;
    $i = $i + 1;
}
ok(gc_collect() > 4000, "mutual pair: both halves are freed");
ok(settle() - $base <= 6, "mutual pair: live returns to baseline");

// 3. A method bound to its own object. THE USER WRITES NO BACK-REFERENCE:
//    binding a method builds a scope holding `this` and a function closing
//    over that scope, so the cycle is the interpreter's own doing.
$base = settle();
$i = 0;
while ($i < 3000) { $a = new GcNode(7); $a.callback = $a.value; $i = $i + 1; }
ok(gc_collect() > 2000, "bound method stored on its own object: freed");
ok(settle() - $base <= 6, "bound method: live returns to baseline");

// 4. A function defined inside a function -- a private helper, the most
//    ordinary thing there is. The call frame holds the helper and the helper
//    closes over the call frame.
$base = settle();
$i = 0;
while ($i < 3000) { $z = makeNested(); $i = $i + 1; }
ok(gc_collect() > 2000, "nested function: the leaked call frames are freed");
ok(settle() - $base <= 6, "nested function: live returns to baseline");

// 5. A dict that contains itself.
$base = settle();
$i = 0;
while ($i < 3000) { $d = {}; $d["self"] = $d; $i = $i + 1; }
ok(gc_collect() > 2000, "dict self-reference: freed");
ok(settle() - $base <= 6, "dict self-reference: live returns to baseline");

// 6. A longer ring, so the collector is not just handling the two-node case.
$base = settle();
$i = 0;
while ($i < 500) {
    $a = new GcNode(1); $b = new GcNode(2); $c = new GcNode(3); $d = new GcNode(4);
    $a.next = $b; $b.next = $c; $c.next = $d; $d.next = $a;
    $i = $i + 1;
}
ok(gc_collect() > 1500, "a four-node ring is freed");
ok(settle() - $base <= 6, "four-node ring: live returns to baseline");

print("── what must NOT be collected ───────────────────────────────────");

// A cycle that is still reachable must survive, and keep its contents.
$keep = new GcNode(42);
$keep.self = $keep;
$keepPair = new GcNode(1);
$other = new GcNode(2);
$keepPair.peer = $other;
$other.peer = $keepPair;
$keepDict = {"n": 5};
$keepDict["self"] = $keepDict;

gc_collect();
gc_collect();

eq($keep.self.v, 42,              "a reachable self-cycle survives collection");
eq($keep.self.self.self.v, 42,    "and is still traversable");
eq($keepPair.peer.peer.v, 1,      "a reachable mutual pair survives");
eq($other.peer.peer.v, 2,         "both halves of it survive");
eq($keepDict["self"]["self"]["n"], 5, "a reachable dict cycle survives");

// A closure returned out of its defining scope is NOT garbage: the returned
// function is the only thing keeping that frame alive, and it must work.
def makeAdder($n) { return def($x) { return $x + $n; }; }
$add5 = makeAdder(5);
gc_collect();
gc_collect();
eq($add5(10), 15, "a returned closure still works after a collection");

// The same, held only inside a collected-through container.
$holder = {"fn": makeAdder(100)};
gc_collect();
$f = $holder["fn"];
eq($f(1), 101, "a closure held in a dict survives collection");

// An object reachable only from a list inside another object.
$outer = new GcNode(0);
$outer.kids = [new GcNode(11), new GcNode(12)];
gc_collect();
eq($outer.kids[1].v, 12, "objects reachable through a list survive");

print("── the automatic collection ─────────────────────────────────────");

// Enough garbage to cross the threshold without anyone calling gc_collect().
$before = gc_stats()["collections"];
$i = 0;
while ($i < 60000) { $a = new GcNode(1); $a.self = $a; $i = $i + 1; }
ok(gc_stats()["collections"] > $before, "automatic collection runs on its own");
ok(gc_stats()["live"] < 40000, "and keeps the live set bounded while it runs");

// A program with no cycles must never trigger one. This is the "costs
// nothing when you don't need it" property, asserted rather than assumed.
gc_collect();
$before = gc_stats()["collections"];
$i = 0;
while ($i < 60000) { $a = new GcNode($i); $i = $i + 1; }
eq(gc_stats()["collections"], $before, "acyclic work triggers no collection at all");

print("── sort: the key function ───────────────────────────────────────");

$rows = [{"name": "ada", "age": 36}, {"name": "bo", "age": 19},
         {"name": "cy",  "age": 36}, {"name": "di", "age": 7}];

def byAge($r)  { return $r["age"]; }
def byName($r) { return $r["name"]; }

def names($list) {
    $out = [];
    each ($r in $list) { push($out, $r["name"]); }
    return join($out, ",");
}

eq(names(sort($rows, {"key": byAge})), "di,bo,ada,cy", "key sorts ascending");
eq(names(sort($rows, {"key": byAge, "desc": true})), "ada,cy,bo,di", "key sorts descending");
// Stability: ada and cy both have age 36 and must keep their input order in
// BOTH directions. Reversing the finished list would swap them.
eq(names(sort($rows, {"key": byAge})), "di,bo,ada,cy", "ties keep input order ascending");
eq(names(sort($rows, {"key": byAge, "desc": true})), "ada,cy,bo,di", "ties keep input order descending");
eq(names(sort($rows, {"key": byName})), "ada,bo,cy,di", "string keys sort lexicographically");
eq(names(sort($rows, {"key": byName, "desc": true})), "di,cy,bo,ada", "string keys sort descending");

// The input is untouched: Bantu lists have value semantics.
eq(names($rows), "ada,bo,cy,di", "sort does not modify its argument");

def identity($x) { return $x; }
eq(join(sort([3, 1, 2], {"key": identity}), ","), "1,2,3", "key on a plain number list");
// NaN last in both directions, for the same reason the no-key path documents:
// a < b with NaN present is not a strict weak ordering.
eq(join(sort([3, NAN, 1, 2], {"key": identity}), ","), "1,2,3,nan", "NaN keys sort last ascending");
eq(join(sort([3, NAN, 1, 2], {"key": identity, "desc": true}), ","), "3,2,1,nan", "NaN keys sort last descending");

eq(len(sort([], {"key": identity})), 0, "empty list with a key");
eq(join(sort([5], {"key": identity}), ","), "5", "single element with a key");

// "desc" alone in the options dict, with no key.
eq(join(sort([3, 1, 2], {"desc": true}), ","), "3,2,1", "options dict with desc but no key");

// The older spellings must keep working unchanged.
eq(join(sort([3, 1, 2]), ","), "1,2,3", "sort(list) still works");
eq(join(sort([3, 1, 2], "desc"), ","), "3,2,1", "sort(list, \"desc\") still works");
eq(join(sort([3, 1, 2], def($a, $b) { return $b - $a; }), ","), "3,2,1", "sort(list, cmp) still works");

$threw = false;
try { sort([1, 2], {"key": def($x) { if ($x == 1) { return "a"; } return 2; }}); }
catch ($e) { $threw = true; }
ok($threw, "a key function returning mixed types raises");

$threw = false;
try { sort([1, 2], {"key": identity, "cmp": def($a, $b) { return 0; }}); }
catch ($e) { $threw = true; }
ok($threw, "passing both key and cmp raises");

$threw = false;
try { sort([1, 2], {"key": 7}); } catch ($e) { $threw = true; }
ok($threw, "a non-function key raises");

$threw = false;
try { sort([1, 2], 7); } catch ($e) { $threw = true; }
ok($threw, "a number as the second argument still raises");

print("");
print("Passed: " + str($R["pass"]) + "   Failed: " + str($R["fail"]));
if ($R["fail"] == 0) { print("RESULT: ALL GREEN"); }
else { print("RESULT: FAILURES"); }
