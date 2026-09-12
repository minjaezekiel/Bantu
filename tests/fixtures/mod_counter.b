// Fixture for tests/lang_module_test.b — a module included from several places.
// Nothing here is a test; the tests/*.b glob does not reach this directory.

$NAME = "counter";

def label() { return "counter/" + $NAME; }
