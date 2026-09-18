// ════════════════════════════════════════════════════════════════════════
//  lang_control_flow_test.b — return, break and continue through every
//  construct that can hold them.
//
//  These used to be C++ exceptions and are now a pending signal checked
//  between statements (docs/control-flow-architecture.md). The failure mode
//  that design has to rule out is a signal left pending, which would silently
//  stop LATER statements -- so nearly every case below also checks that the
//  code after it still runs.
// ════════════════════════════════════════════════════════════════════════

$R = {"pass": 0, "fail": 0};
def eq($got, $want, $what) {
    if ($got == $want) { $R["pass"] = $R["pass"] + 1; }
    else { $R["fail"] = $R["fail"] + 1; print("  FAIL: " + $what + " -- got " + str($got) + ", wanted " + str($want)); }
}

print("── return ────────────────────────────────────────────────────────");

def early($x) {
    if ($x > 0) { return "positive"; }
    return "not positive";
}
eq(early(1), "positive", "return inside if");
eq(early(0 - 1), "not positive", "return after an if that did not return");

def elseReturn($x) {
    if ($x > 0) { $y = 1; } else { return "else"; }
    return "after";
}
eq(elseReturn(0), "else", "return inside else");
eq(elseReturn(1), "after", "and past it when else is not taken");

def implicit($x) { $y = $x * 2; $y + 1; }
eq(implicit(5), 11, "no return: the last statement's value, as before");
def emptyReturn() { return; }
eq(emptyReturn(), null, "a bare return is null");
def noBody() { }
eq(noBody(), null, "an empty body is null");

def fromWhile() {
    $i = 0;
    while (true) { $i = $i + 1; if ($i == 7) { return $i; } }
    return 0 - 1;
}
eq(fromWhile(), 7, "return from inside a while");

def fromFor() {
    for ($i = 0; $i < 100; $i = $i + 1) { if ($i == 4) { return $i * 10; } }
    return 0 - 1;
}
eq(fromFor(), 40, "return from inside a C-style for");

def fromEach($xs) {
    each ($x in $xs) { if ($x > 2) { return $x; } }
    return null;
}
eq(fromEach([1, 2, 3, 4]), 3, "return from inside each");
eq(fromEach([1]), null, "and past it when nothing matched");

def fromDictEach($d) {
    each ($k, $v in $d) { if ($v == 2) { return $k; } }
    return null;
}
eq(fromDictEach({"a": 1, "b": 2}), "b", "return from inside each over a dict");

def fromNested() {
    $i = 0;
    while ($i < 5) {
        $j = 0;
        while ($j < 5) {
            if ($i * $j == 6) { return [$i, $j]; }
            $j = $j + 1;
        }
        $i = $i + 1;
    }
    return null;
}
eq(fromNested(), [2, 3], "return from two loops deep");

def fromSwitch($x) {
    switch ($x) {
        case 1 { return "one"; }
        case 2 { return "two"; }
        default { return "many"; }
    }
    return "unreachable";
}
eq(fromSwitch(2), "two", "return from a switch case");
eq(fromSwitch(9), "many", "and from its default");

def fromTry() {
    try { return "try"; } catch ($e) { return "catch"; }
    return "after";
}
eq(fromTry(), "try", "return inside try is not caught");
def fromCatch() {
    try { throw "boom"; } catch ($e) { return "caught " + $e; }
    return "after";
}
eq(fromCatch(), "caught boom", "return inside catch");
def afterTry() {
    try { $x = 1; } catch ($e) { return "catch"; }
    return "after";
}
eq(afterTry(), "after", "a try that did not return leaves nothing pending");

def recurse($n) { if ($n <= 1) { return 1; } return $n * recurse($n - 1); }
eq(recurse(10), 3628800, "recursion returns through every frame");

def outer() {
    $inner = def($x) { return $x + 1; };
    $a = $inner(1);
    $b = $inner(10);
    return $a + $b;
}
eq(outer(), 13, "a closure's return is consumed by its own call");

def twoCalls() { return early(1) + " and " + early(0); }
eq(twoCalls(), "positive and not positive", "two returning calls in one expression");

class CfBox {
    def init($v) { $this.v = $v; }
    def get() { if ($this.v == null) { return "none"; } return $this.v; }
    def loop() { $i = 0; while (true) { $i = $i + 1; if ($i == 3) { return $i; } } }
}
$box = new CfBox(5);
eq($box.get(), 5, "return from a method");
eq($box.loop(), 3, "return from a loop in a method");
eq(new CfBox(null).get(), "none", "an early return in a method");

print("── break and continue ────────────────────────────────────────────");

$out = [];
$i = 0;
while ($i < 10) {
    $i = $i + 1;
    if ($i % 2 == 0) { continue; }
    if ($i > 7) { break; }
    push($out, $i);
}
eq($out, [1, 3, 5, 7], "continue and break in a while");
eq($i, 9, "and the loop stopped where break said");

$out = [];
for ($k = 0; $k < 10; $k = $k + 1) {
    if ($k % 3 != 0) { continue; }
    push($out, $k);
}
eq($out, [0, 3, 6, 9], "continue in a for still runs the update");

$out = [];
each ($v in [1, 2, 3, 4, 5, 6]) {
    if ($v == 2) { continue; }
    if ($v == 5) { break; }
    push($out, $v);
}
eq($out, [1, 3, 4], "continue and break in each");

$out = [];
$i = 0;
while ($i < 3) {
    $i = $i + 1;
    $j = 0;
    while ($j < 3) {
        $j = $j + 1;
        if ($j == 2) { break; }
        push($out, [$i, $j]);
    }
}
eq($out, [[1, 1], [2, 1], [3, 1]], "break leaves only the inner loop");

$out = [];
$i = 0;
while ($i < 5) {
    $i = $i + 1;
    switch ($i) {
        case 2 { continue; }
        case 4 { break; }
        default { push($out, $i); }
    }
}
eq($out, [1, 3], "continue and break inside a switch act on the loop around it");

$out = [];
$i = 0;
while ($i < 5) {
    $i = $i + 1;
    try {
        if ($i == 2) { continue; }
        if ($i == 4) { break; }
        push($out, $i);
    } catch ($e) { push($out, "caught"); }
}
eq($out, [1, 3], "continue and break inside try are not caught");

$n = 0;
$i = 0;
while ($i < 1000) { $i = $i + 1; if ($i % 2 == 0) { continue; } $n = $n + 1; }
eq($n, 500, "a thousand iterations, half of them continued");

// Legacy behaviour, kept on purpose: a break in a function but outside any
// loop in it acts on the CALLER's loop (docs/control-flow-architecture.md).
def breaker() { break; }
$i = 0;
while ($i < 10) { $i = $i + 1; if ($i == 3) { breaker(); } }
eq($i, 3, "a break escaping a function still ends the caller's loop");

print("── nothing is left pending ───────────────────────────────────────");

// After each construct above, the statements that follow ran -- that is what
// every "after" and every second assertion checks. These check it at the top
// level and across calls in one statement.
$after = 0;
early(1);
$after = $after + 1;
fromWhile();
$after = $after + 1;
fromSwitch(1);
$after = $after + 1;
eq($after, 3, "top-level statements after returning calls all run");
eq([early(1), early(0), fromEach([5])], ["positive", "not positive", 5], "returning calls inside a list literal");

print("");
print("Passed: " + str($R["pass"]) + "   Failed: " + str($R["fail"]));
if ($R["fail"] == 0) { print("RESULT: ALL GREEN"); } else { print("RESULT: FAILURES"); }
