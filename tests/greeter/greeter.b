// Fixture for tests/lang_module_test.b: a FOLDER sharing its name with the
// installed tests/bantu_modules/greeter package. `include "greeter"` must keep
// resolving to the installed package -- if this file ever loads instead, the
// directory rule has been placed ahead of bantu_modules.
def hello($who) { return "the folder, not the package"; }
$ENTRY = "tests/greeter/greeter.b";
