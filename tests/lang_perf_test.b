// ════════════════════════════════════════════════════════════════════════
//  lang_perf_test.b — the two performance fixes, and the semantics they
//  must not have changed.
//
//  WHY THIS EXISTS
//  Two changes to the interpreter's core, both of which trade a general path
//  for a faster special case. That is exactly the kind of change that is easy
//  to get subtly wrong, so the semantics get more assertions here than the
//  speed does.
//
//  1. DISPATCH. evalNode used to try dynamic_cast against each node type in
//     turn -- 38 of them, ordered by when each was added. A profile of a 20M
//     iteration arithmetic loop put that dispatcher at 79.6% of ALL interpreter
//     time: four times everything else in the process combined, including
//     Value's size, the allocator and the environment's string hashing. It is
//     a switch on a tag now.
//
//  2. IN-PLACE APPEND. `$s = $s + $part` was O(n^2) -- every + copied the whole
//     accumulated string. When the result of x + y is assigned straight back to
//     x, x's old value is dead, so it can be appended to in place. This is
//     CPython's fix (unicode_concatenate) and it covers `$s += $part` too,
//     since the parser desugars one into the other.
//
//  THE SEMANTICS THAT MATTER, and why each is a real hazard:
//    - Inside a function, `$s = $s + "x"` on a global must create a LOCAL, the
//      way every other assignment does. Appending in place would mutate the
//      global instead -- a silent action at a distance.
//    - `$t = $s` must keep its own copy. Strings are value-semantic.
//    - `$s = $s + $s` must not read the buffer it is writing.
//    - A const target must still raise.
//    - Numbers, lists, dicts and native handles must go down the ordinary
//      operator path untouched.
//
//  Run:  bantu run tests/lang_perf_test.b
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
        print("  FAIL  " + $name);
        print("          got:  " + str($got));
        print("          want: " + str($want));
    }
}

def raises($f, $name) {
    $caught = false;
    try { $f(); } catch ($e) { $caught = true; }
    ok($caught, $name);
}

print("========================================");
print("  Bantu — dispatch and in-place append");
print("========================================");

// ── The append itself ───────────────────────────────────────────────────
print("");
print("-- the ordinary shapes --");
$s = "a";
$s = $s + "b";
eq($s, "ab", "$s = $s + literal");
$s += "c";
eq($s, "abc", "$s += literal  (the parser desugars this into the form above)");
$p = "d";
$s = $s + $p;
eq($s, "abcd", "$s = $s + variable");
$s = $s + "e" + "f" + "g";
eq($s, "abcdefg", "a multi-piece chain, applied left to right");
$s = $s + 1;
eq($s, "abcdefg1", "a number operand stringifies");
$s = $s + true;
eq($s, "abcdefg1true", "and so does a bool");
$empty = "";
$empty = $empty + "";
eq($empty, "", "appending nothing to nothing");
eq(len($empty), 0, "and the length agrees");

print("");
print("-- order is preserved exactly --");
$o = "";
$i = 0;
while ($i < 5) { $o = $o + str($i); $i = $i + 1; }
eq($o, "01234", "a loop appends in order");
$o2 = "";
$i = 0;
while ($i < 3) { $o2 = $o2 + "<" + str($i) + ">"; $i = $i + 1; }
eq($o2, "<0><1><2>", "and so does a three-piece chain in a loop");

// ── Aliasing ────────────────────────────────────────────────────────────
print("");
print("-- aliasing: appending in place must not be visible anywhere else --");
$src = "keep";
$copy = $src;
$src = $src + "-changed";
eq($copy, "keep", "a copy taken before the append is untouched");
eq($src, "keep-changed", "and the original took the append");

$self = "ab";
$self = $self + $self;
eq($self, "abab", "$s = $s + $s reads its own old value, not the buffer it writes");
// ((self + self) + self) with self = "abab": 4 -> 8 -> 12 characters, all
// three pieces being the PRE-append value. Getting 16 here would mean a piece
// had seen the buffer mid-append.
$self = $self + $self + $self;
eq($self, "abababababab", "  (three deep: every piece is the pre-append value)");
eq(len($self), 12, "  ...so 4 characters become 12, not 16");

// A function whose body reads the target while the append is in flight.
$g = "start";
def readsG() { return "-" + $g; }
$g = $g + readsG();
eq($g, "start-start", "an operand that reads the target sees its pre-append value");

// ── Scoping: the hazard that matters most ───────────────────────────────
print("");
print("-- scoping: an assignment inside a function still creates a local --");
$outer = "global";
def touchesOuter() {
    $outer = $outer + "-local";     // assignment, so this makes a LOCAL
    return $outer;
}
$inner = touchesOuter();
eq($inner, "global-local", "the function sees the global's value and builds on it");
eq($outer, "global", "but the GLOBAL is untouched -- appending in place must not leak out");
$again = touchesOuter();
eq($again, "global-local", "and it is the same every call, not an accumulation");

def nested() {
    $acc = "";
    $j = 0;
    while ($j < 4) { $acc = $acc + "x"; $j = $j + 1; }
    return $acc;
}
eq(nested(), "xxxx", "a function-local accumulator works normally");
eq(nested(), "xxxx", "and does not carry over between calls");

// ── const ───────────────────────────────────────────────────────────────
print("");
print("-- const still raises --");
raises(def() { const $c = "fixed"; $c = $c + "more"; return $c; },
       "appending to a const raises rather than mutating it");

// ── Types the peephole must decline ─────────────────────────────────────
print("");
print("-- numbers and other types take the ordinary operator path --");
$n = 1;
$n = $n + 2;
eq($n, 3, "numeric + is still numeric");
$n += 4;
eq($n, 7, "and so is numeric +=");
$fl = 0.5;
$fl = $fl + 0.25;
eq($fl, 0.75, "floats too");
$mixed = 5;
$mixed = $mixed + "x";
eq($mixed, "5x", "number + string still concatenates");
$notyet = "";
$lst = [1, 2];
$lst2 = $lst;
$lst2 = $lst2 + $lst;         // whatever + means for lists, it is unchanged
eq(len($lst), 2, "a list operand does not disturb the list");

print("");
print("-- the target on the RIGHT is not this optimisation --");
$r = "tail";
$r = "head-" + $r;
eq($r, "head-tail", "$s = literal + $s prepends correctly");
$r = "a" + $r + "z";
eq($r, "ahead-tailz", "and a chain with the target in the middle");

print("");
print("-- the assignment's own value is still available when used --");
$u = "u";
$v = ($u = $u + "w");
eq($v, "uw", "a used assignment yields the new value");
eq($u, "uw", "and the target holds it");

print("");
print("-- containers are unaffected --");
$d = {"k": "v"};
$d.k = $d.k + "2";
eq($d["k"], "v2", "a dict field appends (a different node type entirely)");
$arr = ["a", "b"];
$arr[0] = $arr[0] + "!";
eq($arr[0], "a!", "and so does a list element");
eq($arr[1], "b", "without touching its neighbour");

// ── Linearity ───────────────────────────────────────────────────────────
print("");
print("-- and it is linear, which is the whole point --");
// Sizes big enough that the times dominate the noise, and each buffer is
// released before the next is timed -- a few megabytes still live makes the
// allocator, not the interpreter, the thing being measured.
$frag = "0123456789";
$t0 = clock();
$b1 = "";
$i = 0;
while ($i < 50000) { $b1 = $b1 + $frag; $i = $i + 1; }
$t50k = clock() - $t0;
eq(len($b1), 500000, "50,000 appends produce 500,000 characters");
$b1 = "";

$t0 = clock();
$b2 = "";
$i = 0;
while ($i < 100000) { $b2 = $b2 + $frag; $i = $i + 1; }
$t100k = clock() - $t0;
eq(len($b2), 1000000, "100,000 appends produce 1,000,000 characters");
$b2 = "";
// Before the fix, doubling the work cost 6.05x, and 100,000 appends took about
// forty seconds. A 3x ceiling catches anything quadratic with enormous room to
// spare while tolerating a slow shared CI runner.
ok($t100k <= max(3 * $t50k, 100),
   "twice the appends costs about twice the time, not four times it");
print("          50k " + str($t50k) + " ms, 100k " + str($t100k) + " ms");

// The same content built the two supported ways must be identical.
$parts = [];
$i = 0;
while ($i < 500) { push($parts, $frag); $i = $i + 1; }
$viaJoin = join($parts, "");
$viaPlus = "";
$i = 0;
while ($i < 500) { $viaPlus = $viaPlus + $frag; $i = $i + 1; }
eq($viaPlus, $viaJoin, "+= and join() produce byte-identical output");

// ── Dispatch: every node kind still evaluates ───────────────────────────
// The tag-and-switch dispatcher is only correct if every node type reaches its
// own arm. These exercise the kinds a Bantu program can produce, so that under
// the -DBANTU_CHECK_NODEKIND build (which verifies every tag against RTTI)
// each one is actually checked.
print("");
print("-- every node kind still reaches its own evaluator --");
eq(1, 1, "number literal");
eq("s", "s", "string literal");
ok(true, "bool literal");
eq(null, null, "null literal");
eq(len([1, 2, 3]), 3, "list literal");
eq(len(keys({"a": 1})), 1, "dict literal");
number $typed = 5;
eq($typed, 5, "typed declaration");
eq(-$typed, -5, "unary minus");
eq(!false, true, "unary not");
$sw = 0;
switch (2) { case 1 { $sw = 10; } case 2 { $sw = 20; } default { $sw = 30; } }
eq($sw, 20, "switch/case");
$fr = 0;
for ($k = 0; $k < 3; $k = $k + 1) { $fr = $fr + 1; }
eq($fr, 3, "C-style for");
$ea = 0;
each ($x in [1, 2, 3]) { $ea = $ea + $x; }
eq($ea, 6, "each");
$wh = 0;
while ($wh < 2) { $wh = $wh + 1; if ($wh == 1) { continue; } }
eq($wh, 2, "while with continue");
$br = 0;
while (true) { $br = 1; break; }
eq($br, 1, "break");
def fnode($a) { return $a * 2; }
eq(fnode(4), 8, "function declaration, call and return");
$anon = def($a) { return $a + 1; };
eq($anon(1), 2, "anonymous function");
$thrown = "none";
try { throw "boom"; } catch ($e) { $thrown = $e; }
eq($thrown, "boom", "throw and try/catch");
class PerfBox { def init($v) { $this.v = $v; } def get() { return $this.v; } }
def mkBox($v) { return new PerfBox($v); }
eq(mkBox(9).get(), 9, "class declaration, instantiation and method dispatch");
$dd = {"a": {"b": 7}};
eq($dd["a"]["b"], 7, "nested index access");
$dd["a"]["b"] = 8;
eq($dd["a"]["b"], 8, "nested index assignment");
$oo = {"p": 1};
$oo.p = 2;
eq($oo.p, 2, "dot access and dot assignment");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
