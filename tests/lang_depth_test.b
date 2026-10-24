// ════════════════════════════════════════════════════════════════════════
//  lang_depth_test.b — recursion and JSON nesting end in an error, not a crash.
//
//  Two places recursed without a bound, and both ended the process instead of
//  raising something a program could catch.
//
//  1. Bantu function calls. Every call is a real C++ recursion through
//     evalNode, so `def f($n) { return 1 + f($n - 1); }` walked off the native
//     stack: measured here it survived 3,000 frames and died at 3,200 with
//     SIGSEGV -- exit 139, no message, no catch. Windows is far worse, since
//     its default thread stack is 1 MB against 8 MB here.
//
//  2. json.parse. The parser recursed once per `{` or `[` with no limit, so a
//     request body of nothing but `[[[[...` did the same thing -- and that one
//     was reachable from any client. Against the shipped build, a 200 KB POST
//     to a route that only reads $req.body killed the server outright: empty
//     reply, then connection refused, process exit 139.
//
//  Both now stop at a limit WE choose rather than one the stack imposes, and
//  both raise an ordinary Bantu error. That is the whole point of the fix: a
//  program can try/catch around a runaway recursion or a hostile body and keep
//  running. This test pins that, because the failure it guards against is
//  silent -- a regression here does not fail loudly, it segfaults.
//
//  Run:  bantu run tests/lang_depth_test.b
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

// A string of $n copies of $c. Built with `while`, not the C-style `for`,
// which silently stops at 100,000 iterations.
def rep($c, $n) {
    $s = ""; $i = 0;
    while ($i < $n) { $s = $s + $c; $i = $i + 1; }
    return $s;
}

// ── 1. call depth ───────────────────────────────────────────────────────
def countdown($n) {
    if ($n <= 0) { return 0; }
    return 1 + countdown($n - 1);
}

print("-- recursion below the limit is untouched --");
eq(countdown(900), 900, "900 frames still returns 900");
eq(countdown(1), 1, "a one-deep call is unaffected");
eq(countdown(0), 0, "a zero-deep call is unaffected");

print("");
print("-- past the limit it raises, and the raise is catchable --");
$caught = "";
try {
    countdown(100000);
    $caught = "NO RAISE";
} catch ($e) {
    $caught = $e.message;
}
ok($caught != "NO RAISE", "100,000 frames raises instead of crashing");
ok(contains($caught, "Maximum call depth"), "the error says what was exceeded");
ok(contains($caught, "1000"), "the error names the actual limit");
ok(contains($caught, "recursion"), "the error names the likely cause");
ok(contains($caught, "loop"), "the error names the fix");

// The guard has to UNWIND, not just fire. If the counter were not decremented
// on the way out, the first runaway recursion would poison every later call in
// the program -- which is a worse bug than the one being fixed.
eq(countdown(900), 900, "the depth counter unwinds after a catch");

print("");
print("-- mutual recursion is bounded too --");
// Defined in this order because the body is only resolved when it runs, so
// pong may name ping before ping exists.
def pong($n) { return ping($n - 1); }
def ping($n) { return pong($n - 1); }
$caught2 = "";
try { ping(100000); $caught2 = "NO RAISE"; } catch ($e) { $caught2 = $e.message; }
ok(contains($caught2, "Maximum call depth"), "a mutually recursive pair raises");

print("");
print("-- a method and a constructor are bounded on the same counter --");
class DepthProbe {
    def init($n) {
        if ($n > 0) { $this.child = new DepthProbe($n - 1); }
    }
}
$caught3 = "";
try { $p = new DepthProbe(100000); $caught3 = "NO RAISE"; } catch ($e) { $caught3 = $e.message; }
ok(contains($caught3, "Maximum call depth"), "a recursive constructor raises");

// ── 2. json.parse nesting ───────────────────────────────────────────────
print("");
print("-- ordinary JSON is unaffected --");
$flat = json.parse("{\"a\": [1, 2, {\"b\": true}], \"c\": \"x\"}");
eq(json.stringify($flat), "{\"a\":[1,2,{\"b\":true}],\"c\":\"x\"}",
   "nested objects and lists round-trip");
eq($flat.a[2].b, true, "a value three levels down reads back");

// 100 levels is far past any real document and well under the limit.
$deep100 = rep("[", 100) + rep("]", 100);
$v100 = json.parse($deep100);
ok($v100 != null, "100 levels of nesting still parses");

print("");
print("-- past the limit it raises, and the server survives it --");
$jcaught = "";
try {
    json.parse(rep("[", 5000) + rep("]", 5000));
    $jcaught = "NO RAISE";
} catch ($e) {
    $jcaught = $e.message;
}
ok($jcaught != "NO RAISE", "5,000 levels raises instead of crashing");
ok(contains($jcaught, "nesting"), "the error says nesting was the problem");
ok(contains($jcaught, "256"), "the error names the actual limit");
ok(contains($jcaught, "try/catch"), "the error names the fix");

// Unbalanced is the real attack shape -- 20 KB of "[" with no closing bracket
// at all, which is what killed the shipped build.
$jcaught2 = "";
try { json.parse(rep("[", 20000)); $jcaught2 = "NO RAISE"; } catch ($e) { $jcaught2 = $e.message; }
ok($jcaught2 != "NO RAISE", "20,000 unbalanced brackets raise, not segfault");

// Deep objects, not just deep lists -- the other recursive arm.
$jcaught3 = "";
try { json.parse(rep("{\"a\":", 5000)); $jcaught3 = "NO RAISE"; } catch ($e) { $jcaught3 = $e.message; }
ok($jcaught3 != "NO RAISE", "deeply nested objects raise too");

// And the interpreter is still healthy afterwards.
eq(json.parse("{\"ok\": 1}").ok, 1, "json.parse still works after a refusal");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
