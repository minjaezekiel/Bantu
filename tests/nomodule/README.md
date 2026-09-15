Fixture for `tests/lang_module_test.b`: a folder with **no module in it**.

`include "nomodule"` used to open this directory as if it were a file, read nothing, and bind the
alias to an empty module without any error. It must now be reported as not found.
