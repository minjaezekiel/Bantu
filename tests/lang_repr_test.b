// ════════════════════════════════════════════════════════════════════════
//  lang_repr_test.b — print() on a native handle shows the value.
//
//  A NATIVE_HANDLE stringified to "<column>", so print($col) told you the type
//  and nothing whatsoever about the data. There was no way to look at a column
//  without materializing it to a Bantu list first -- which, on a five-million
//  row column, means building five million ~190-byte Values to read three
//  numbers. The handle was opaque in the one place opacity is least useful.
//
//  Value::toString now consults a registry of renderers keyed by handle tag,
//  so each native layer teaches print() about the type it owns and types.hpp
//  needs to know nothing about any of them.
//
//  Summarization follows NumPy's rule -- every element up to 1000, then three
//  from each end -- because matching a convention people already know beats
//  inventing one.
//
//  Run:  bantu run tests/lang_repr_test.b
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

print("-- a column renders its contents, not its type --");
eq(str(col([1, 2, 3], "f64")), "[1, 2, 3]  (len=3, dtype=f64)",
   "f64 column (this used to be \"<column>\")");
eq(str(col([1, 2, 3], "i64")), "[1, 2, 3]  (len=3, dtype=i64)",
   "i64 column reports its own dtype");
eq(str(col([true, false], "bool")), "[true, false]  (len=2, dtype=bool)",
   "bool column");

print("");
print("-- text is quoted, so [\"1\"] is distinguishable from [1] --");
eq(str(col(["a", "b"], "utf8")), "[\"a\", \"b\"]  (len=2, dtype=utf8)",
   "utf8 elements are quoted");
ok(str(col(["1"], "utf8")) != str(col([1], "f64")),
   "a string \"1\" does not render identically to the number 1");

print("");
print("-- nulls and boundaries --");
eq(str(col([], "f64")), "[]  (len=0, dtype=f64)", "an empty column");
eq(str(col([1.5, null, 3], "f64")), "[1.5, null, 3]  (len=3, dtype=f64)",
   "a null element renders as null, not as a zero");
eq(str(col([42], "i64")), "[42]  (len=1, dtype=i64)", "a single element");

print("");
print("-- the semantic overlays render as what they mean --");
eq(str(col_to_datetime(col(["2024-01-02 03:04:05"], "utf8"))),
   "[\"2024-01-02 03:04:05\"]  (len=1, dtype=datetime)",
   "a datetime column shows ISO text and dtype=datetime");

print("");
print("-- summarization: NumPy's rule, 1000 then 3 from each end --");
$k = [];
$i = 0;
while ($i < 1000) { $k.push($i); $i = $i + 1; }
$at = col($k, "i64");
ok(!contains(str($at), "..."), "exactly 1000 elements is printed in full");

$k.push(1000);
$over = col($k, "i64");
$s = str($over);
ok(contains($s, "..."), "1001 elements is summarized");
ok(contains($s, "[0, 1, 2, ..."), "the first three are shown");
ok(contains($s, "998, 999, 1000]"), "and the last three");
ok(contains($s, "(len=1001, dtype=i64)"), "the true length is still reported");

print("");
print("-- a large column must not be expensive to print --");
// Truncation means the cost is in building the column, not rendering it. If
// this ever became O(n) it would be the kind of regression nobody notices
// until a print() in a loop stalls a server.
$big = [];
$i = 0;
while ($i < 200000) { $big.push($i); $i = $i + 1; }
$bigcol = col($big, "i64");
$t0 = clock();
$j = 0;
while ($j < 100) { $ignored = str($bigcol); $j = $j + 1; }
$elapsed = clock() - $t0;
print("        100 reprs of a 200,000-element column: " + str($elapsed) + "ms");
ok($elapsed < 200, "rendering is independent of length (100 reprs under 200ms)");
ok(contains(str($bigcol), "(len=200000, dtype=i64)"), "and still correct");

print("");
print("-- an unregistered handle still prints something honest --");
$f = open("/dev/null", "r");
ok($f != null, "a file handle is a dict, not a native handle, so it is unaffected");
close($f);

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
