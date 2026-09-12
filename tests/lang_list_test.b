// ════════════════════════════════════════════════════════════════════════
//  lang_list_test.b — appending to a list is O(1), and stays that way.
//
//  `$l.push(x)` returned the mutated list. A Bantu list is a
//  std::vector<Value> with value semantics, and a Value is a ~190-byte struct
//  carrying a std::string, a std::vector<Value>, a std::function and three
//  shared_ptrs -- so returning the list deep-copied every element. An O(1)
//  append was O(n), and building a list in a loop was O(n^2).
//
//  Measured before the fix, on this machine:
//
//        2,500 pushes     122 ms
//        5,000 pushes     420 ms
//       10,000 pushes   1,853 ms
//       20,000 pushes   9,616 ms      <- 137x the cost of append($l, x)
//
//  and 200,000 pushes would have taken roughly a quarter of an hour. It now
//  takes about 0.6 s. This was NOT introduced by any recent work: the shipped
//  release binary measured 9,314 ms for the same 20,000 pushes.
//
//  The fix distinguishes the two call forms. `push($l, x)` still returns the
//  list, because `$x = push($x, v)` is a documented idiom (see lang_test.b).
//  `$l.push(x)` returns the new length, as in JavaScript -- it is the form
//  that appears inside loops, and nothing assigns from it.
//
//  Run:  bantu run tests/lang_list_test.b
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

print("-- both call forms mutate in place --");
$a = [1, 2];
$a.push(3);
eq(str($a), "[1, 2, 3]", "$l.push(x) appends");
$b = [1, 2];
push($b, 3);
eq(str($b), "[1, 2, 3]", "push($l, x) appends");
$c = [1, 2];
append($c, 3);
eq(str($c), "[1, 2, 3]", "append($l, x) appends");

print("");
print("-- what each form returns --");
$d = [1, 2];
eq($d.push(3), 3, "$l.push(x) returns the new length (JavaScript's convention)");
$e = [1];
$e = push($e, 2);
eq(str($e), "[1, 2]", "$x = push($x, v) still returns the list -- the documented idiom");
$f = [1, 2];
eq(append($f, 3), 3, "append($l, x) returns the new length");

print("");
print("-- pushing several values at once --");
$g = [];
$g.push(1);
$g.push(2);
$g.push(3);
eq(str($g), "[1, 2, 3]", "repeated pushes accumulate");

print("");
print("-- values survive the move into the list --");
// push moves its arguments into the vector. A moved-from argument must not
// leave a hollow Value behind in the list.
$h = [];
$s = "hello";
$h.push($s);
$h.push([1, 2]);
$h.push({"k": 9});
eq($h[0], "hello", "a string pushed keeps its contents");
eq(str($h[1]), "[1, 2]", "a nested list pushed keeps its contents");
eq($h[2].k, 9, "a dict pushed keeps its contents");
eq($s, "hello", "and the source variable is untouched");

print("");
print("-- appending is LINEAR, not quadratic --");
// The real gate. If push ever goes back to returning a copy of the list, the
// ratio below blows up: at these sizes it was 8.6x before the fix.
def timePush($n) {
    $t0 = clock();
    $l = [];
    $i = 0;
    while ($i < $n) { $l.push($i); $i = $i + 1; }
    return clock() - $t0;
}

// The bare form as a STATEMENT discards its result, so it must be just as
// cheap. It stayed quadratic when only the method form was fixed.
def timeBarePush($n) {
    $t0 = clock();
    $l = [];
    $i = 0;
    while ($i < $n) { push($l, $i); $i = $i + 1; }
    return clock() - $t0;
}

$small = timePush(25000);
$large = timePush(100000);
$bare  = timeBarePush(100000);
print("        25,000 $l.push(x): " + str($small) + "ms");
print("       100,000 $l.push(x): " + str($large) + "ms");
print("       100,000 push($l,x): " + str($bare) + "ms");
eq(len([]), 0, "(sanity) an empty list has length 0");
ok($bare < 15000, "the bare push($l, x) statement form is linear too");

// 4x the work should cost roughly 4x the time. Allow 3x headroom for timer
// noise and allocator growth on a loaded CI box; quadratic would be ~16x.
$budget = $small * 12;
if ($budget < 250) { $budget = 250; }   // a floor, so a fast machine's ~20ms baseline cannot make this flaky
ok($large < $budget, "4x the elements costs well under 12x the time (quadratic would be ~16x)");

// An absolute ceiling as well: before the fix this took about 4 minutes.
ok($large < 15000, "100,000 pushes complete in under 15s (was ~4 minutes)");

print("");
print("-- len($var) does not copy the container --");
// Bantu lists have value semantics, so passing one to a function copies it.
// len() is the builtin routinely called on the container being built, which
// made `$out[len($out)] = v` quadratic. It reads the real storage now.
eq(len([1, 2, 3]), 3, "len of a list literal");
$L = [1, 2, 3];
eq(len($L), 3, "len of a list variable");
$S = "hello";
eq(len($S), 5, "len of a string variable");
$D = {"a": 1, "b": 2};
eq(len($D), 0, "len of a dict is still 0, exactly as before");
$Z = [];
eq(len($Z), 0, "len of an empty list");
$N = 42;
eq(len($N), 0, "len of a non-container is still 0");

$t0 = clock();
$out = [];
$i = 0;
while ($i < 20000) { $out[len($out)] = $i; $i = $i + 1; }
$idiom = clock() - $t0;
print("        20,000 x $out[len($out)] = v: " + str($idiom) + "ms");
eq(len($out), 20000, "the idiom still builds the right list");
ok($idiom < 3000, "and is linear now (was 7,027ms at this size)");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
