// ════════════════════════════════════════════════════════════════════════
//  lang_test.b — regression tests for the v1.3.0 language features.
//  Run:  bantu run tests/lang_test.b
//  Prints a PASS/FAIL summary; "ALL GREEN" when everything works.
// ════════════════════════════════════════════════════════════════════════

$R = {"pass": 0, "fail": 0};

def eq($got, $want, $name) {
    if ($got == $want) {
        $R.pass += 1;
        print("  ok    " + $name);
    } else {
        $R.fail += 1;
        print("  FAIL  " + $name + "  (got " + str($got) + ", want " + str($want) + ")");
    }
}
def ok($cond, $name) {
    if ($cond) { $R.pass += 1; print("  ok    " + $name); }
    else { $R.fail += 1; print("  FAIL  " + $name); }
}

print("");
print("-- compound assignment --");
$x = 10;
$x += 5; $x *= 2; $x -= 3; $x /= 2;
eq($x, 13.5, "compound += *= -= /=");
$sum = 0;
for (number $i = 0; $i < 5; $i += 1) { $sum += $i; }
eq($sum, 10, "for-loop with += terminates");

print("");
print("-- break / continue --");
$found = -1; $j = 0;
while ($j < 100) {
    $j += 1;
    if ($j == 3) { continue; }
    if ($j == 6) { break; }
    $found = $j;
}
eq($found, 5, "break + continue");

print("");
print("-- switch --");
def label($n) {
    switch ($n) {
        case 1 { return "one"; }
        case 2 { return "two"; }
        default { return "many"; }
    }
    return "?";
}
eq(label(2), "two", "switch matches case");
eq(label(9), "many", "switch falls to default");

print("");
print("-- throw / try / catch --");
$caught = "";
try { throw {"code": 42}; } catch ($e) { $caught = str($e.code); }
eq($caught, "42", "throw value is caught");
$msg = "";
try { $bad = 1 / 0; } catch ($e) { $msg = $e.message; }
eq($msg, "Division by zero", "runtime error is catchable");

print("");
print("-- const is final --");
const $PI = 3;
eq($PI, 3, "const holds its value");
// Reassigning a const is a COMPILE error (blocked by the gate / flagged by the
// linter), so it cannot be exercised at runtime here — see tests/const_bad.b.

print("");
print("-- anonymous functions --");
$h = {"double": def($n) { return $n * 2; }};
eq($h.double(21), 42, "anonymous function in dict");

print("");
print("-- for-in / dict iteration --");
$acc = 0;
for $v in [1, 2, 3, 4] { $acc += $v; }
eq($acc, 10, "for-in over list");
$d = {"a": 1, "b": 2, "c": 3};
$dsum = 0;
for $k, $v in $d.items() { $dsum += $v; }
eq($dsum, 6, "for k,v in dict.items()");
eq(len($d.keys()), 3, "dict.keys()");
eq(len($d.values()), 3, "dict.values()");
eq($d.size(), 3, "dict.size()");
eq(len(keys($d)), 3, "keys() builtin");

print("");
print("-- list mutators --");
$l = [1, 2, 3];
append($l, 4); append($l, 5);
$last = pop($l);
insert($l, 0, 0);
remove($l, 2);
extend($l, [7, 8]);
eq(str($l), "[0, 1, 3, 4, 7, 8]", "append/pop/insert/remove/extend");
eq($last, 5, "pop returns last");

// push() now mutates in place (it used to be a silent no-op) and returns the
// list, so the older `$x = push($x, v)` idiom also works.
$p = [1, 2];
push($p, 3);
eq(str($p), "[1, 2, 3]", "push() mutates in place");
$p2 = [1];
$p2 = push($p2, 2);
eq(str($p2), "[1, 2]", "$x = push($x, v) idiom");
// method form mutates too, including a list nested in a dict
$m = [1, 2];
$m.push(3);
$popped = $m.pop();
eq(str($m), "[1, 2]", "$l.push() / $l.pop() mutate");
eq($popped, 3, "$l.pop() returns last");
$nest = {"r": [10]};
$nest.r.push(20);
eq(str($nest.r), "[10, 20]", "nested $d.list.push() mutates");

print("");
print("-- file I/O --");
writefile("/tmp/bantu_lang_test.txt", "alpha\nbeta\ngamma\n");
appendfile("/tmp/bantu_lang_test.txt", "delta\n");
$content = readfile("/tmp/bantu_lang_test.txt");
eq(len(split($content, "\n")), 5, "readfile/writefile/appendfile");
$fh = open("/tmp/bantu_lang_test.txt", "r");
$firstLine = readline($fh);
close($fh);
eq($firstLine, "alpha", "open + readline + close");

print("");
print("-- parameterized SQL --");
sua.sqlite.open(":memory:");
sua.sqlite.exec("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)");
sua.sqlite.exec("INSERT INTO t (name, age) VALUES (?, ?)", ["Ada", 36]);
sua.sqlite.exec("INSERT INTO t (name, age) VALUES (?, ?)", ["Bob", 41]);
$rows = sua.sqlite.query("SELECT * FROM t WHERE age > ?", [30]);
eq(len($rows), 2, "parameterized query");
$inj = sua.sqlite.query("SELECT * FROM t WHERE name = ?", ["x' OR '1'='1"]);
eq(len($inj), 0, "injection attempt neutralized");

print("");
print("-- FFI (libffi) --");
$ffiOk = true;
$m = null;
try { $m = loadlib("libm.dylib"); } catch ($e1) {
    try { $m = loadlib("libm.so.6"); } catch ($e2) { $ffiOk = false; }
}
if ($ffiOk) {
    $sqrt = func($m, "sqrt", "double", ["double"]);
    ok($sqrt(16.0) == 4.0, "FFI sqrt(16) = 4");
} else {
    print("  skip  FFI (libm not found at known paths)");
}

print("");
print("-- reserved-word variables --");
$db = 5; $list = [1, 2, 3]; $create = "ok";
eq($db + len($list), 8, "$db and $list usable as variables");

print("");
print("-- && and || short-circuit (fixed) --");
// They did not, and the universal guard idiom died on exactly the boundary
// it was written to prevent.
$SC = {"hits": 0};
def sideEffect() { $SC["hits"] = $SC["hits"] + 1; return true; }

$SC["hits"] = 0;
$r = false && sideEffect();
eq($r, false, "false && x is false");
eq($SC["hits"], 0, "and the right side was NOT evaluated");

$SC["hits"] = 0;
$r = true || sideEffect();
eq($r, true, "true || x is true");
eq($SC["hits"], 0, "and the right side was NOT evaluated");

$SC["hits"] = 0;
$r = true && sideEffect();
eq($r, true, "true && x still evaluates the right side");
eq($SC["hits"], 1, "exactly once");

$SC["hits"] = 0;
$r = false || sideEffect();
eq($r, true, "false || x still evaluates the right side");
eq($SC["hits"], 1, "exactly once");

// The whole point: these two lines must not raise.
$arr = [1, 2];
$i = 2;
$guardOk = true;
try {
    if ($i < len($arr) && $arr[$i] == 9) { $guardOk = false; }
} catch ($e) { $guardOk = false; }
ok($guardOk, "if ($i < len($a) && $a[$i] == x) is safe at the boundary");

$maybe = null;
$nullOk = true;
try {
    if ($maybe != null && $maybe["k"] == 1) { $nullOk = false; }
} catch ($e) { $nullOk = false; }
ok($nullOk, "if ($d != null && $d[\"k\"] == 1) is safe on a null");

// Truthiness and chaining are unchanged.
eq(1 && 1, true, "&& still yields a boolean");
eq(0 || 0, false, "|| still yields a boolean");
eq(true && true && false, false, "chained && short-circuits at the first false");
eq(false || false || true, true, "chained || stops at the first true");

print("");
print("-- objects are freed (fixed: instances used to leak forever) --");
// The observable half of the fix. `new` used to allocate an instance nothing
// ever deleted; the RSS gate lives in tests/bplot_stress.sh, and this asserts
// that the semantics did not change when ownership did.
class LifeTest { def init($v) { $this.v = $v; } def get() { return $this.v; } }
$objA = new LifeTest(7);
$objB = $objA;
$objB.v = 9;
eq($objA.get(), 9, "instances still have reference semantics through assignment");
$held = [$objA];
$held[0].v = 11;
eq($objA.get(), 11, "and through a list");
$bag = {"o": $objA};
$bag["o"].v = 13;
eq($objA.get(), 13, "and through a dict");
def mutate($o) { $o.v = 15; return null; }
mutate($objA);
eq($objA.get(), 15, "and through a function argument");
$k = 0;
while ($k < 5000) { $tmp = new LifeTest($k); $k = $k + 1; }
eq($k, 5000, "5,000 instances created and dropped without incident");

print("");
print("-- words the lexer reserved for nothing (fixed) --");
// read, await, private, public, calc, import and export were lexed as keywords
// that no grammar rule accepted: the documented read() builtin could not be
// called anywhere, and no function could take one of these names.
def calc($a, $b) { return $a + $b; }
def export($x) { return "exported " + str($x); }
def import($x) { return "imported " + str($x); }
def await($x) { return $x * 2; }
def private() { return "private"; }
def public() { return "public"; }
eq(calc(2, 3), 5, "a function can be named calc");
eq(export(1), "exported 1", "or export");
eq(import(2), "imported 2", "or import");
eq(await(21), 42, "or await");
eq(private() + " " + public(), "private public", "or private and public");
$rf = open("/tmp/bantu_lang_test_read.txt", "w");
write($rf, "abc");
close($rf);
$rf = open("/tmp/bantu_lang_test_read.txt", "r");
eq(read($rf), "abc", "read($f) can be called at all, including as an argument");
close($rf);
$rf = open("/tmp/bantu_lang_test_read.txt", "r");
$whole = read($rf);
close($rf);
eq($whole, "abc", "and in the documented form, $rest = read($f)");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
print("========================================");
