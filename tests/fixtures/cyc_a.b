// Fixture: a genuine circular include, cyc_a <-> cyc_b. This must TERMINATE
// and must not bind a half-built module -- when cyc_b includes cyc_a, cyc_a is
// still executing and has exported nothing yet.

include "./cyc_b.b" as b;

def a_id() { return "A"; }
