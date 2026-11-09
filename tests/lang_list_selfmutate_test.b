// ════════════════════════════════════════════════════════════════════════
//  lang_list_selfmutate_test.b — `$x = push($x, v)` and friends.
//
//  The in-place list mutators edit the caller's list through resolveLValue,
//  so when the assignment target IS the list being mutated, the returned
//  copy was being written straight back over the original. That copy is the
//  whole list, which made a loop of appends QUADRATIC:
//
//      20,000 x  $l = push($l, $i);     12,781 ms
//      20,000 x  push($l, $i);               22 ms
//
//  A 581x gap between two forms the documentation presents as equivalent --
//  and the slow one is the form the README teaches. After the fix the
//  assigned form is 29 ms and scales linearly.
//
//  The optimisation skips the store, so the risk is an append that happens
//  twice, or not at all, or a shadowing user function that stops working.
//  That is what this file is for.
//
//  Run:  bantu run tests/lang_list_selfmutate_test.b
// ════════════════════════════════════════════════════════════════════════

$R = {"pass": 0, "fail": 0};
def ok($cond, $name) {
    if ($cond) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $name); }
}
def eq($got, $want, $name) {
    if (str($got) == str($want)) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name);
        print("          got:  " + str($got));
        print("          want: " + str($want));
    }
}

print("== the element lands exactly ONCE ==");
$a = [];
$a = push($a, 1);
eq($a, [1], "one push -> one element");
$a = push($a, 2);
eq($a, [1, 2], "two pushes -> two elements, in order");
$a = push($a, 3);
eq(len($a), 3, "three pushes -> length 3 (not 6, not 0)");
eq($a, [1, 2, 3], "and the contents are right");

print("== append is the same builtin ==");
$b = [];
$b = append($b, "x");
$b = append($b, "y");
eq($b, ["x", "y"], "append twice");
eq(len($b), 2, "append length");

print("== extend ==");
$c = [1, 2];
$c = extend($c, [3, 4]);
eq($c, [1, 2, 3, 4], "extend concatenates once");
eq(len($c), 4, "extend length");
$c2 = [];
$c2 = extend($c2, []);
eq(len($c2), 0, "extend with an empty list");

print("== insert ==");
$d = [1, 3];
$d = insert($d, 1, 2);
eq($d, [1, 2, 3], "insert in the middle");
$d = insert($d, 0, 0);
eq($d, [0, 1, 2, 3], "insert at the front");

print("");
print("== pop and remove must NOT take the fast path ==");
// They return the removed ELEMENT, not the list, so `$x = pop($x)` is a
// legitimate rebinding of $x to that element and must keep working.
$e = [10, 20, 30];
$e = pop($e);
eq($e, 30, "`$x = pop($x)` rebinds $x to the popped ELEMENT, not the list");
ok(type($e) == "number", "and it really is the element, not a list");
$e2 = [1, 2, 3];
$popped = pop($e2);
eq($popped, 3, "pop into a different variable returns the element");
eq($e2, [1, 2], "and mutates the original");

print("");
print("== a DIFFERENT target still gets the list back ==");
$f = [1, 2];
$g = push($f, 3);
eq($g, [1, 2, 3], "`$g = push($f, v)` gives $g the list");
eq($f, [1, 2, 3], "and $f was mutated in place too (existing behaviour)");
ok(len($g) == 3, "the returned list is complete");

print("");
print("== the value is still produced where it is actually used ==");
$h = [1];
eq(len(push($h, 2)), 2, "push() inside len() returns the list");
$i2 = [1];
eq(str(push($i2, 2)), str([1, 2]), "push() inside str() returns the list");
$j = [5];
$k = [];
$k = push($k, push($j, 6));
eq(len($k), 1, "a push used as an ARGUMENT to another push still returns its list");
eq($k[0], [5, 6], "and that inner list is correct");

print("");
print("== scale: the assigned form is now linear ==");
def timed($n) {
    $l = [];
    $t0 = clock();
    $i = 0;
    while ($i < $n) { $l = append($l, $i); $i = $i + 1; }
    $t1 = clock();
    return [$t1 - $t0, len($l)];
}
$r1 = timed(5000);
$r2 = timed(20000);
eq($r1[1], 5000, "5,000 appends produced 5,000 elements");
eq($r2[1], 20000, "20,000 appends produced 20,000 elements");
// 4x the work must not cost anything like 16x the time. The pre-fix numbers
// were 632 ms and 12,781 ms -- a ratio of 20. Allow generous headroom for a
// slow machine while still failing loudly if it is quadratic again.
$ratio = ($r2[0] + 1) / ($r1[0] + 1);
ok($ratio < 10, "4x the appends cost well under 16x the time (ratio " + str($ratio) + ")");
print("          (5,000: " + str($r1[0]) + " ms   20,000: " + str($r2[0]) + " ms)");

print("");
print("== the accumulator pattern this exists for ==");
// The shape that appears throughout hash.b, crypto.b and btorch's own tape.
def collect($n) {
    $out = [];
    $i = 0;
    while ($i < $n) { $out = push($out, $i * 2); $i = $i + 1; }
    return $out;
}
$coll = collect(1000);
eq(len($coll), 1000, "a collect() helper builds 1,000 elements");
eq($coll[0], 0, "  first element");
eq($coll[999], 1998, "  last element");
// and inside a nested function, where the list is a local
def outer() {
    $acc = [];
    $acc = push($acc, "a");
    $acc = push($acc, "b");
    return $acc;
}
eq(outer(), ["a", "b"], "the idiom works on a function-local list");

print("");
print("== lists of lists, and lists of other values ==");
$ll = [];
$ll = push($ll, [1, 2]);
$ll = push($ll, [3, 4]);
eq(len($ll), 2, "pushing lists into a list");
eq($ll[1][0], 3, "nested access after the fast path");
$mixed = [];
$mixed = push($mixed, "s");
$mixed = push($mixed, 1);
$mixed = push($mixed, null);
$mixed = push($mixed, true);
eq(len($mixed), 4, "mixed types all land");
ok($mixed[2] == null, "a null element survives");
ok($mixed[3] == true, "a bool element survives");
$dicts = [];
$dicts = push($dicts, {"k": 1});
eq($dicts[0].k, 1, "a dict element survives and is readable");

print("");
print("== an ndarray handle in a list (btorch's tape does this) ==");
$hs = [];
$hs = push($hs, nd_full([2], 1.0));
$hs = push($hs, nd_full([2], 2.0));
eq(len($hs), 2, "two handles pushed");
eq(nd_to_list($hs[1]), [2, 2], "the second handle is intact and usable");

print("");
print("== a user-defined function of the same name still WINS ==");
// The parser cannot see shadowing, so the evaluator re-checks at run time: a
// user function cannot mutate in place, so its return value must still be
// stored. This block is LAST on purpose -- a top-level `def push` is global
// for the remainder of the file, so anywhere earlier it would shadow the
// builtin for every later section (it did, while this file was being written:
// every subsequent length came back as 8, the length of "shadowed").
def push($lst, $v) { return "shadowed"; }
$m = [1, 2];
$m = push($m, 3);
eq($m, "shadowed", "a user-defined push() shadows the builtin and its result IS stored");
ok(type($m) == "string", "the shadowing function's own return value is what lands");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
