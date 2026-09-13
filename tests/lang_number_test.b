// ════════════════════════════════════════════════════════════════════════
//  lang_number_test.b — numeric literals, including scientific notation.
//
//  The defect this was written for: str() emitted "1.23457e-05" and the lexer
//  could not read it back. `$y = 1.23457e-05;` lexed as the number 1.23457
//  followed by an identifier `e`, so the error was "Undefined variable: e" --
//  a baffling message for what looks like an ordinary number. The JSON parser
//  accepted exponents all along, so the two halves of the language disagreed
//  about what a number is.
//
//  The round-trip is the real assertion here: anything str() can produce, the
//  lexer must be able to read.
//
//  Run:  bantu run tests/lang_number_test.b
// ════════════════════════════════════════════════════════════════════════

$R = {"pass": 0, "fail": 0};

def eq($got, $want, $name) {
    if ($got == $want) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
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

print("========================================");
print("  Bantu — numeric literals");
print("========================================");

print("");
print("-- ordinary literals still lex exactly as before --");
eq(1, 1, "an integer");
eq(3.5, 3.5, "a decimal");
eq(0.25, 0.25, "a leading-zero decimal");
eq(100, 100, "a round number");

print("");
print("-- scientific notation --");
eq(1e3, 1000, "1e3");
eq(1E3, 1000, "1E3 -- capital E too");
eq(1e+3, 1000, "an explicit + in the exponent");
eq(2.5e2, 250, "a decimal mantissa");
eq(1e-3, 0.001, "a negative exponent");
eq(1.23457e-05, 0.0000123457, "the exact form str() emits");
eq(5e0, 5, "a zero exponent");
eq(0e5, 0, "a zero mantissa");

print("");
print("-- the round trip that was broken --");
$x = 0.000012345678;
$s = str($x);
print("        str(0.000012345678) = " + $s);
ok(contains($s, "e"), "str() really does emit scientific notation for small numbers");
// The point of the fix: whatever str() produced above is now lexable. This
// literal IS that output, written back into the source.
eq(1.23457e-05, 0.0000123457, "and the lexer reads that exact text back");

print("");
print("-- very large and very small --");
ok(1e300 > 0, "1e300 lexes");
ok(1e-300 > 0, "1e-300 lexes");
ok(1e308 > 1e307, "near the top of the double range");

print("");
print("-- 'e' is still an identifier when it is not an exponent --");
// This is the case the fix must NOT break: only consume the e when what
// follows really is an exponent.
$e = 7;
eq($e, 7, "a variable named e still works");
$two = 2;
eq($two * $e, 14, "and can be multiplied by a number");

print("");
print("-- method calls on numbers still parse --");
eq(2.5.round(), 3, "a method call on a decimal literal");
eq(7.floor(), 7, "and on an integer literal");

print("");
print("-- arithmetic with exponent literals --");
eq(1e3 + 1, 1001, "addition");
eq(2e2 * 2, 400, "multiplication");
eq(1e3 - 1e2, 900, "subtraction between two of them");
eq(1e-2 * 1e2, 1, "a negative and a positive exponent cancel");

print("");
print("-- and they reach numba, which is where this surfaced --");
if (has_native("ndarray")) {
    $a = nd([1e-5, 2e-5], null);
    ok(nd_get($a, [0]) == 0.00001, "an exponent literal inside an array");
    ok(nd_allclose(nd_multiply($a, 1e5, null), nd([1, 2], null), null, null, null),
       "and as a scalar operand to a ufunc");
} else {
    print("  --    numba not built in; skipping");
}

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
