// Fixture: one half of a diamond. Both mod_a and mod_b include mod_counter.
// Before the repeat-include fix, whichever of the two loaded SECOND got an
// unbound `ctr`, so calling this function raised "Undefined variable: ctr".

include "./mod_counter.b" as ctr;

def a_name() { return ctr.NAME; }
