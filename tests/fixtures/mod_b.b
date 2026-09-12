// Fixture: the other half of the diamond. See mod_a.b.

include "./mod_counter.b" as ctr;

def b_name() { return ctr.NAME; }
def b_label() { return ctr.label(); }
