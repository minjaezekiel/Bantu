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
// This line used to read `eq(len($D), 0, "len of a dict is still 0, exactly as
// before")`. It pinned the old answer to prove the copy-free change above did
// not alter behaviour -- which it did not. But 0 was never RIGHT: it made
// `while ($i < len($d))` silently skip every dict. len() now counts entries
// (bplot decision BP36), so the pin moves with the fix.
eq(len($D), 2, "len of a dict counts its entries");
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
print("-- sort and reverse (new: the language could not ORDER a list) --");
$src = [3, 1, 2];
eq(join(sort($src), ","), "1,2,3", "sort ascending");
eq(join($src, ","), "3,1,2", "and the argument is untouched — value semantics");
eq(join(sort([5, 2, 9, 1], "desc"), ","), "9,5,2,1", "sort descending");
eq(join(sort([5, 2, 9, 1], "asc"), ","), "1,2,5,9", "\"asc\" is accepted explicitly");
eq(join(sort(["pear", "apple", "fig"]), ","), "apple,fig,pear", "strings sort lexicographically");
eq(join(sort([]), ","), "", "an empty list sorts to an empty list");
eq(join(sort([7]), ","), "7", "one element");
eq(join(sort([2, 2, 1, 1]), ","), "1,1,2,2", "duplicates survive");
eq(join(sort([-3, 0, 2, -1]), ","), "-3,-1,0,2", "negatives order correctly");

// NaN sorts last, and that is a correctness requirement: every comparison
// with NaN is false, so `a < b` is not a strict weak ordering when one is
// present, and std::sort given such a comparator reads past the end of its
// range. numba's nd_sort already orders NaN last, so the two agree.
$withNan = sort([3, NAN, 1, NAN, 2]);
eq(len($withNan), 5, "NaN entries are kept, not dropped");
eq($withNan[0], 1, "the finite values sort first");
eq($withNan[2], 3, "in order");
ok(isnan($withNan[3]) && isnan($withNan[4]), "and both NaNs land at the end");
$descNan = sort([3, NAN, 1], "desc");
eq($descNan[0], 3, "descending puts the largest first");
ok(isnan($descNan[2]), "and NaN is STILL last — it is not a large value, it is an absent one");

// A comparator gets a merge sort, which is stable and cannot run off its
// range whatever the comparator answers.
def byLen($a, $b) { return len($a) - len($b); }
eq(join(sort(["aaa", "b", "cc"], byLen), ","), "b,cc,aaa", "a comparator orders by any rule");
def tie($a, $b) { return 0; }
eq(join(sort(["x", "y", "z"], tie), ","), "x,y,z", "a comparator that always ties keeps the order (stable)");

$caught = false;
try { sort([1, "a"]); } catch ($e) { $caught = true; }
ok($caught, "a list mixing numbers and strings raises rather than inventing an order");
$caught = false;
try { sort([1, 2], "sideways"); } catch ($e) { $caught = true; }
ok($caught, "an unknown direction raises");
$caught = false;
try { sort("not a list"); } catch ($e) { $caught = true; }
ok($caught, "sort of a non-list raises");

eq(join(reverse([1, 2, 3]), ","), "3,2,1", "reverse a list");
eq(reverse("abc"), "cba", "reverse a string");
eq(join(reverse([]), ","), "", "reverse an empty list");
$r = [1, 2, 3];
reverse($r);
eq(join($r, ","), "1,2,3", "reverse does not mutate its argument either");

// Sorting has to be usable on real data, not just on three elements.
$big = [];
$i = 0;
$seed = 12345;
while ($i < 20000) {
    $seed = ($seed * 1103515245 + 12345) - floor(($seed * 1103515245 + 12345) / 2147483648) * 2147483648;
    push($big, $seed);
    $i = $i + 1;
}
$t0 = clock();
$sorted = sort($big);
$sortMs = clock() - $t0;
print("        20,000 elements sorted: " + str($sortMs) + "ms");
eq(len($sorted), 20000, "20,000 elements come back");
$ordered = true;
$i = 1;
while ($i < 20000) {
    if ($sorted[$i] < $sorted[$i - 1]) { $ordered = false; }
    $i = $i + 1;
}
ok($ordered, "and they are in order");
ok($sortMs < 3000, "in well under three seconds");

print("");
print("-- contains() on a list (fixed: it answered false for EVERY list) --");
// contains() was string-only and fell through to `false` for anything else,
// so contains([1, 2], 1) was false and a membership guard silently took the
// wrong branch. Equality is exactly the one == uses.
eq(contains([1, 2, 3], 2), true, "a number that is there");
eq(contains([1, 2, 3], 9), false, "a number that is not");
eq(contains(["a", "b"], "b"), true, "a string element");
eq(contains(["ab", "cd"], "b"), false, "membership, not a substring search inside elements");
eq(contains([1, null, 3], null), true, "null is a value like any other");
eq(contains([[1, 2], [3]], [1, 2]), true, "a nested list, compared structurally as == does");
eq(contains([{"k": 1}], {"k": 1}), true, "a dict, compared structurally as == does");
eq(contains([], 1), false, "an empty list contains nothing");
eq(contains([1, 2], true), true, "true == 1 in Bantu, so contains agrees with ==");
eq(contains("hello", "ell"), true, "the string form is unchanged");
eq(contains("hello", "xyz"), false, "and still answers false when absent");
eq(contains(42, 4), false, "a non-container still answers false");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
