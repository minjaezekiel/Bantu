// ════════════════════════════════════════════════════════════════════════
//  hotpath.b — the interpreter paths that operator dispatch touches.
//
//  Written for the Phase 4 gate: adding a branch to evalBinaryOp,
//  evalUnaryOp, evalIndexAccess and evalIndexAssign has to be paid for by
//  every arithmetic expression in every Bantu program, not just by numba. So
//  the phase is gated on a measurement of exactly those paths, before and
//  after, on the same machine, and the numbers go in the CHANGELOG.
//
//  Best-of-N rather than a mean: a mean measures the machine's background
//  noise as much as the interpreter, and we want the floor.
//
//  Run:  bantu run benchmarks/hotpath.b
// ════════════════════════════════════════════════════════════════════════

def best($name, $reps, $fn) {
    $b = 99999999;
    $i = 0;
    while ($i < $reps) {
        $t0 = clock();
        $fn();
        $ms = clock() - $t0;
        if ($ms < $b) { $b = $ms; }
        $i = $i + 1;
    }
    print("  " + $name + ": " + str($b) + " ms");
    return $b;
}

def fib($n) {
    if ($n < 2) { return $n; }
    return fib($n - 1) + fib($n - 2);
}

print("=== interpreter hot paths ===");

// The headline gate: a 1M-iteration arithmetic loop. Every iteration runs
// evalBinaryOp several times on two plain numbers, which is the case the
// dispatch branch must not slow down.
best("1M arithmetic while loop   ", 5, def() {
    $i = 0;
    $acc = 0;
    while ($i < 1000000) {
        $acc = $acc + $i * 2 - 1;
        $i = $i + 1;
    }
});

// Comparison operators, same path, different opcode.
best("1M comparison loop         ", 5, def() {
    $i = 0;
    $n = 0;
    while ($i < 1000000) {
        if ($i > 500000) { $n = $n + 1; }
        $i = $i + 1;
    }
});

// Unary minus, the second dispatch site.
best("500k unary negation        ", 5, def() {
    $i = 0;
    $s = 0;
    while ($i < 500000) {
        $s = $s + -$i;
        $i = $i + 1;
    }
});

// The call path, which the dispatch does not touch but which would show a
// regression from anything accidentally added to Value construction.
best("fib(24) recursive          ", 3, def() { fib(24); });

// Index read and write on a list: evalIndexAccess and evalIndexAssign, the
// other two dispatch sites.
best("200k list index read       ", 5, def() {
    $l = [1, 2, 3, 4, 5, 6, 7, 8];
    $i = 0;
    $s = 0;
    while ($i < 200000) {
        $s = $s + $l[$i % 8];
        $i = $i + 1;
    }
});

best("100k list index write      ", 5, def() {
    $l = [0, 0, 0, 0, 0, 0, 0, 0];
    $i = 0;
    while ($i < 100000) {
        $l[$i % 8] = $i;
        $i = $i + 1;
    }
});

// Dict set, and string concatenation -- the `+` that must keep working on
// strings after the dispatch branch is added.
best("50k dict set               ", 5, def() {
    $d = {};
    $i = 0;
    while ($i < 50000) {
        $d["k" + str($i % 100)] = $i;
        $i = $i + 1;
    }
});

best("100k string concat         ", 5, def() {
    $i = 0;
    while ($i < 100000) {
        $s = "a" + "b";
        $i = $i + 1;
    }
});

print("=== done ===");
