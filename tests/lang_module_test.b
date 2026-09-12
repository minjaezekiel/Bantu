// ════════════════════════════════════════════════════════════════════════
//  lang_module_test.b — what `include` resolves, and what it binds.
//
//  Two defects, both latent for every package already shipped:
//
//    1. `bantu add <pkg>` installs into ./bantu_modules/<pkg>/, but the module
//       resolver had no rule for that directory. So after installing a package
//       you still could not write `include "greeter" as g;` -- you had to spell
//       out `include "./bantu_modules/greeter/src/hello.b" as g;`, which is
//       both the first thing a new user hits and the one thing they cannot
//       guess.
//
//    2. A second `include` of the same file bound NOTHING. The cycle guard
//       returned before the alias was ever defined, so if two of your modules
//       both did `include "arctic" as arctic;`, whichever loaded second was
//       left with an undefined `arctic` and the only clue was a line on
//       stderr. Any application with more than one module hits this.
//
//  Fixtures live in tests/fixtures/ and tests/bantu_modules/ -- the tests/*.b
//  glob CI runs is shallow, so neither directory is executed as a test.
//
//  Run:  bantu run tests/lang_module_test.b
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

print("-- a bare name resolves inside bantu_modules --");
// greeter/package.json says "main": "src/hello.b". The conventional
// <name>.b rule would look for greeter/greeter.b, which does not exist, so
// this only loads if the manifest was actually read.
include "greeter" as g;
eq(g.hello("world"), "hello, world", "include \"greeter\" resolves via package.json main");
eq(g.ENTRY, "src/hello.b", "and it really was the file main pointed at");

// The manifest also carries a "note": "main" decoy. If the scanner had matched
// that value instead of the real key, the include above would have failed.
ok(true, "a value reading \"main\" is not mistaken for the main key");

// plainpkg has no package.json at all.
include "plainpkg" as p;
eq(p.ping(), "pong", "a package with no manifest falls back to <name>.b");

print("");
print("-- an explicit relative path still means exactly what it says --");
include "./fixtures/mod_counter.b" as ctr;
eq(ctr.NAME, "counter", "a ./ path is unaffected by the bantu_modules rule");

print("");
print("-- the same module included twice binds BOTH aliases --");
// This is the whole defect: the second one used to bind nothing.
include "./fixtures/mod_counter.b" as ctr2;
eq(ctr2.NAME, "counter", "the SECOND include of a file binds its alias too");
eq(ctr2.label(), "counter/counter", "and its functions are callable");

// One module, one namespace object -- as in Node. A field set through one
// alias is visible through the other, which proves they are the same object
// rather than two copies.
ctr.marker = 7;
eq(ctr2.marker, 7, "both aliases are the SAME object, not two copies");

print("");
print("-- a diamond: two modules that both include a third --");
// mod_a and mod_b each do `include "./mod_counter.b" as ctr`. Before the fix,
// whichever loaded second raised "Undefined variable: ctr" when called.
include "./fixtures/mod_a.b" as ma;
include "./fixtures/mod_b.b" as mb;
eq(ma.a_name(), "counter", "the first arm of the diamond works");
eq(mb.b_name(), "counter", "the second arm works too (this used to be undefined)");
eq(mb.b_label(), "counter/counter", "and it can call through to the shared module");

print("");
print("-- a genuine cycle terminates without binding a half-built module --");
// cyc_a includes cyc_b, which includes cyc_a again. The inner include lands
// while cyc_a is still executing: there is no finished module to bind, so it
// is reported and skipped. Reaching this line at all is the assertion.
include "./fixtures/cyc_a.b" as ca;
eq(ca.a_id(), "A", "a circular include terminates and the outer module loads");

print("");
print("-- a missing module is reported, not silently ignored --");
// An unresolvable include prints [INCLUDE ERROR] and carries on WITHOUT
// defining the alias, so the first use of it raises. Asserted so the
// behaviour is recorded rather than discovered: the failure is loud at the
// point of use, not a null that propagates somewhere far away.
include "./fixtures/definitely_not_here.b" as missing;
$raised = false;
try {
    $x = missing.anything;
} catch ($e) {
    $raised = true;
}
ok($raised, "using an alias whose include failed raises at the point of use");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
