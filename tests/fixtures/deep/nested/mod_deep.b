// Fixture for tests/lang_module_test.b — a module three levels BELOW the
// directory that holds the packages it imports. Nothing here is a test; the
// tests/*.b glob does not reach this directory.
//
// Neither bare name below exists beside this file, or beside the one that
// includes it: `dirpkg/` and `bantu_modules/greeter/` both live in tests/,
// three directories up. They resolve only because a bare name is searched up
// the whole tree, which is the point of the fixture.

include "greeter" as g;   // tests/bantu_modules/greeter/, via package.json main
include "dirpkg" as dp;   // tests/dirpkg/, by the directory rule

$NAME = "deep";

def deep_installed() { return g.ENTRY; }
def deep_dirpkg()    { return dp.NAME; }
