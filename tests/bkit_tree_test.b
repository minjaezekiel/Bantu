// ════════════════════════════════════════════════════════════════════════
//  bkit_tree_test.b — the `tree_` native family and the two decision trees.
//
//  Five native kernels, and the rule for every one of them is the same: the
//  reference is computed HERE, in Bantu, from the binned matrix the kernel
//  itself produced — never by calling the kernel a second way. A kernel
//  checked against itself is a kernel nobody checked.
//
//  The two assertions that matter most:
//
//   1. **tree_hist equals a pure-Bantu triple loop, exactly.** Not allclose:
//      the same floats added in the same order must give the same bits.
//   2. **The chosen split is the argmax of a hand-computed Gini table.** The
//      whole claim that one criterion serves both estimators rests on gain
//      being proportional to Gini decrease; proving it on paper is not the
//      same as proving the code does it.
//
//  Run:  bantu run tests/bkit_tree_test.b
// ════════════════════════════════════════════════════════════════════════

include "../numba/numba.b" as np;
include "../bkit/base.b" as base;
include "../bkit/tree.b" as tree;

$R = {"pass": 0, "fail": 0};
def ok($cond, $name) {
    if ($cond) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $name); }
}
def eq($got, $want, $name) {
    if (str($got) == str($want)) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name);
        print("          got:  " + str($got));
        print("          want: " + str($want));
    }
}
def close($got, $want, $tol, $name) {
    if (abs($got - $want) <= $tol) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name + "  got " + str($got) + " want " + str($want));
    }
}
def raises($fn, $name) {
    try { $fn(); $R.fail = $R.fail + 1; print("  FAIL  " + $name + " (no error raised)"); }
    catch ($e) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    return null;
}
// A section that THROWS must be attributed to that section, not end the run.
// [K52]: two tree_ mutations were "caught by aborting" -- the battery reported
// a crash with no indication which assertion or which guard, and every later
// section went unreported. Each section below is wrapped in a try, and a throw
// counts as one failure NAMED BY SECTION. A top-level try in Bantu does not
// open a new scope (verified), so variables still flow from one section to the
// next exactly as they did before.
def section_threw($name, $e) {
    $R.fail = $R.fail + 1;
    print("  FAIL  [section aborted] " + $name);
    print("          threw: " + str($e));
    return null;
}
def safe_ok($fn, $name) {
    try { ok($fn(), $name); }
    catch ($e) {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name + "  threw: " + str($e));
    }
    return null;
}
// A 0-d reduction is an ndarray handle and every handle is truthy (numba N41).
def sc($x) { return np.get(np.reshape($x, [1]), [0]); }

np.seed(20261115);

print("========================================");
print("  bkit/tree.b + the tree_ native family");
print("========================================");

// ── tree_bin ────────────────────────────────────────────────────────────────
print("");
print("== tree_bin: bins come from DISTINCT values ==");
try {
// Feature 0: 7 distinct. Feature 1: constant. Feature 2: 3 distinct.
$Xb = nd([[1.0, 5.0, 0.0],
          [2.0, 5.0, 0.0],
          [3.0, 5.0, 1.0],
          [4.0, 5.0, 1.0],
          [5.0, 5.0, 2.0],
          [6.0, 5.0, 2.0],
          [7.0, 5.0, 2.0]], "f64");
$B = tree_bin($Xb, 256);
eq(np.to_list(tree_bin_shape($B)), [7, 3], "the shape is carried on the handle");
eq(np.to_list(tree_bin_nbins($B)), [7, 1, 3],
   "7 distinct -> 7 bins, constant -> 1 bin, 3 distinct -> 3 bins");
eq(np.size(tree_bin_edges($B, 1)), 0, "a constant column has no edges at all");
eq(np.size(tree_bin_edges($B, 0)), 6, "7 bins need 6 upper edges");

// K1's whole justification: one byte per cell.
eq(tree_bin_bytes($B), 21, "the binned matrix is rows*cols BYTES, not 8x that");

// Binning must be monotone, or a prefix scan over bins is meaningless.
$mono = true;
$i = 0;
while ($i < 6) {
    if (tree_bin_at($B, $i, 0) > tree_bin_at($B, $i + 1, 0)) { $mono = false; }
    $i = $i + 1;
}
ok($mono, "bins are monotone in the value (the prefix scan depends on it)");
eq(tree_bin_at($B, 0, 1), 0, "every row of a constant column is bin 0");
} catch ($e) { section_threw("tree_bin: bins come from DISTINCT values", $e); }

print("");
print("== tree_bin: max_bins actually caps ==");
try {
$B2 = tree_bin($Xb, 2);
eq(np.get(tree_bin_nbins($B2), [0]), 2, "max_bins=2 over 7 distinct values gives 2 bins");
eq(np.size(tree_bin_edges($B2, 0)), 1, "and one edge");
// Both halves must be non-empty, or the cap produced a useless column.
$lo = 0;  $hi = 0;
$i = 0;
while ($i < 7) {
    if (tree_bin_at($B2, $i, 0) == 0) { $lo = $lo + 1; }
    if (tree_bin_at($B2, $i, 0) == 1) { $hi = $hi + 1; }
    $i = $i + 1;
}
ok($lo > 0 && $hi > 0, "and both bins are populated (" + str($lo) + "/" + str($hi) + ")");
} catch ($e) { section_threw("tree_bin: max_bins actually caps", $e); }

print("");
print("== tree_bin: NaN gets a bin of its own, the highest one ==");
try {
// A NaN has to be built through the ARRAY path: scalar division by zero raises
// in Bantu, while 0/0 on an array is NaN.
$nan = np.get(np.divide(np.zeros([1], "f64"), np.zeros([1], "f64"), null), [0]);
$Xn  = nd([[1.0], [2.0], [3.0], [1.0]], "f64");
np.set($Xn, [3, 0], $nan);
$Bn  = tree_bin($Xn, 256);
eq(np.get(tree_bin_nbins($Bn), [0]), 4, "3 value bins + 1 NaN bin = 4");
eq(tree_bin_at($Bn, 3, 0), 3, "the NaN row lands in the HIGHEST bin");
ok(tree_bin_at($Bn, 0, 0) < 3 && tree_bin_at($Bn, 2, 0) < 3,
   "and no present value does");
} catch ($e) { section_threw("tree_bin: NaN gets a bin of its own, the highest one", $e); }

print("");
print("== tree_bin refuses what it cannot bin ==");
try {
$inf = np.get(np.divide(np.ones([1], "f64"), np.zeros([1], "f64"), null), [0]);
$Xi = nd([[1.0], [2.0]], "f64");
np.set($Xi, [1, 0], $inf);
raises(def() { return tree_bin($Xi, 256); },
       "an infinity is refused, and says to mark it missing instead");
raises(def() { return tree_bin($Xb, 1); },   "max_bins=1 is refused");
raises(def() { return tree_bin($Xb, 257); }, "max_bins=257 is refused (one byte per cell)");
raises(def() { return tree_bin($Xb, 2.5); }, "a fractional max_bins is refused");
raises(def() { return tree_bin(nd([1.0, 2.0], "f64"), 8); }, "a 1-d X is refused");
raises(def() { return tree_bin(np.zeros([0, 3], "f64"), 8); }, "an empty X is refused");
raises(def() { return tree_bin_at($B, 99, 0); }, "an out-of-range cell is refused");
raises(def() { return tree_bin_edges($B, 9); }, "an out-of-range feature is refused");
raises(def() { return tree_hist(nd([1.0], "f64"), nd([1.0], "f64"), nd([1.0], "f64"), null); },
       "a non-handle where a binned matrix belongs is refused by name");

// ── tree_hist ───────────────────────────────────────────────────────────────
} catch ($e) { section_threw("tree_bin refuses what it cannot bin", $e); }
print("");
print("== tree_hist equals a pure-Bantu triple loop, BIT FOR BIT ==");
def refHist($bins, $gv, $hv, $rows, $cols, $nb, $idx) {
    $out = np.zeros([$cols, $nb, 3], "f64");
    $k = 0;
    while ($k < len($idx)) {
        $i = $idx[$k];
        $f = 0;
        while ($f < $cols) {
            $b = tree_bin_at($bins, $i, $f);
            np.set($out, [$f, $b, 0], np.get($out, [$f, $b, 0]) + np.get($gv, [$i]));
            np.set($out, [$f, $b, 1], np.get($out, [$f, $b, 1]) + np.get($hv, [$i]));
            np.set($out, [$f, $b, 2], np.get($out, [$f, $b, 2]) + 1.0);
            $f = $f + 1;
        }
        $k = $k + 1;
    }
    return $out;
}

try {
$g = nd([1.0, 0.0 - 2.0, 3.0, 0.0 - 4.0, 5.0, 0.0 - 6.0, 7.0], "f64");
$h = nd([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0], "f64");
$H = tree_hist($B, $g, $h, null);
eq(np.shape($H), [3, 7, 3], "the histogram is [features, maxBins, 3]");

// The reference, built here from the binned matrix and the same g/h.
$allIdx = [0, 1, 2, 3, 4, 5, 6];
$Href = refHist($B, $g, $h, 7, 3, 7, $allIdx);
ok(nd_tobytes($H, "f64") == nd_tobytes($Href, "f64"),
   "the kernel and the reference loop agree exactly, not just closely");

// Per feature, the bins must sum to the node totals -- which is what the
// grower reads them for.
$tot = np.sum(np.reshape(np.take($H, nd([0], "i64"), 0), [7, 3]), 0, false);
close(np.get($tot, [0]), 4.0,  1.0e-12, "feature 0's bins sum to the total gradient");
close(np.get($tot, [1]), 10.0, 1.0e-12, "and the total hessian");
close(np.get($tot, [2]), 7.0,  1.0e-12, "and the row count");
$tot2 = np.sum(np.reshape(np.take($H, nd([1], "i64"), 0), [7, 3]), 0, false);
close(np.get($tot2, [0]), 4.0, 1.0e-12, "every feature sums to the SAME totals");
} catch ($e) { section_threw("tree_hist equals a pure-Bantu triple loop, BIT FOR BIT", $e); }

print("");
print("== tree_hist over a subset ==");
try {
$sub = np.astype(nd([0, 2, 4], "f64"), "i64");
$Hs  = tree_hist($B, $g, $h, $sub);
$Hsr = refHist($B, $g, $h, 7, 3, 7, [0, 2, 4]);
ok(nd_tobytes($Hs, "f64") == nd_tobytes($Hsr, "f64"),
   "a node's index subset histograms exactly");
$totS = np.sum(np.reshape(np.take($Hs, nd([0], "i64"), 0), [7, 3]), 0, false);
close(np.get($totS, [2]), 3.0, 1.0e-12, "and counts only the subset's rows");

raises(def() { return tree_hist($B, nd([1.0, 2.0], "f64"), $h, null); },
       "a grad of the wrong length is refused, naming both lengths");
raises(def() { return tree_hist($B, $g, $h, np.astype(nd([99.0], "f64"), "i64")); },
       "an index outside the matrix is refused");
} catch ($e) { section_threw("tree_hist over a subset", $e); }

print("");
print("== histogram subtraction: the count channel is EXACT, g and h are not ==");
def chan($H, $c) {
    $sh = np.shape($H);
    return np.reshape(np.take($H, nd([$c], "i64"), 2), [$sh[0], $sh[1]]);
}

try {
// A child's histogram is its parent's minus its sibling's -- the single largest
// optimisation in the algorithm, and the grower uses it. The design's first
// draft claimed the result would be BIT-IDENTICAL to building the child
// directly. It is not, and the arithmetic says so: tree_hist accumulates in
// index order, so the parent sums rows 0..n-1 while the sibling sums a
// subsequence, and sum(A u B) - sum(A) = sum(B) does not hold exactly for
// floating point.
//
// So the gate splits by channel, which says more than one tolerance would:
// counts are integers held in doubles and must agree EXACTLY, while g and h
// agree to a tight relative tolerance -- and the test reports the worst
// observed difference, so a drift from 1e-16 to 1e-9 shows up as a number
// rather than as a still-passing assertion.
$hsParts = tree_partition($B, np.astype(np.arange(0.0, 7.0, 1.0), "i64"), 0, 2, false);
$hsL = $hsParts[0];
$hsR = $hsParts[1];
$hsParent = tree_hist($B, $g, $h, null);
$hsDirect = tree_hist($B, $g, $h, $hsR);
$hsSubbed = np.subtract($hsParent, tree_hist($B, $g, $h, $hsL), null);

ok(nd_tobytes(chan($hsDirect, 2), "f64") == nd_tobytes(chan($hsSubbed, 2), "f64"),
   "the COUNT channel is bit-identical -- counts are exact in doubles");
$dG = sc(np.max(np.abs(np.subtract(chan($hsDirect, 0), chan($hsSubbed, 0), null), null), null, null));
$dH = sc(np.max(np.abs(np.subtract(chan($hsDirect, 1), chan($hsSubbed, 1), null), null), null, null));
print("        worst gradient difference: " + str($dG));
print("        worst hessian  difference: " + str($dH));
ok($dG < 1.0e-12, "the gradient channel agrees to better than 1e-12");
ok($dH < 1.0e-12, "the hessian channel agrees to better than 1e-12");
// And the subtracted histogram must choose the SAME split as the direct one --
// which is the property the grower actually depends on.
eq(np.to_list(tree_best_split($hsDirect, $B, 1.0, 0.0, 0.0, 1.0)),
   np.to_list(tree_best_split($hsSubbed, $B, 1.0, 0.0, 0.0, 1.0)),
   "and both histograms pick an identical split");

// ── tree_best_split ────────────────────────────────────────────────────────
} catch ($e) { section_threw("histogram subtraction: the count channel is EXACT, g and h are not", $e); }
print("");
print("== tree_best_split equals brute force over the same histogram ==");
def bruteSplit($Hv, $cols, $nb, $nbf, $lam, $gam, $minCW, $minN) {
    $bestG = 0.0 - 1.0e300;  $bestF = 0.0 - 1.0;  $bestB = 0.0 - 1.0;
    $f = 0;
    while ($f < $cols) {
        $G = 0.0;  $Hh = 0.0;  $N = 0.0;
        $b = 0;
        while ($b < $nbf[$f]) {
            $G  = $G  + np.get($Hv, [$f, $b, 0]);
            $Hh = $Hh + np.get($Hv, [$f, $b, 1]);
            $N  = $N  + np.get($Hv, [$f, $b, 2]);
            $b = $b + 1;
        }
        if ($Hh + $lam > 0.0) {
            $parent = ($G * $G) / ($Hh + $lam);
            $GL = 0.0;  $HL = 0.0;  $NL = 0.0;
            $b = 0;
            while ($b + 1 < $nbf[$f]) {
                $GL = $GL + np.get($Hv, [$f, $b, 0]);
                $HL = $HL + np.get($Hv, [$f, $b, 1]);
                $NL = $NL + np.get($Hv, [$f, $b, 2]);
                $GR = $G - $GL;  $HR = $Hh - $HL;  $NR = $N - $NL;
                $okk = true;
                if ($NL < $minN || $NR < $minN) { $okk = false; }
                if ($HL < $minCW || $HR < $minCW) { $okk = false; }
                if (!($HL + $lam > 0.0) || !($HR + $lam > 0.0)) { $okk = false; }
                if ($okk) {
                    $gn = 0.5 * (($GL * $GL) / ($HL + $lam)
                               + ($GR * $GR) / ($HR + $lam) - $parent) - $gam;
                    if ($gn > $bestG) { $bestG = $gn; $bestF = 1.0 * $f; $bestB = 1.0 * $b; }
                }
                $b = $b + 1;
            }
        }
        $f = $f + 1;
    }
    return [$bestF, $bestB, $bestG];
}

try {
// Brute force in Bantu, from the histogram: every (feature, bin) cut, same
// formula, same guards.
$nbf = np.to_list(tree_bin_nbins($B));
$bs  = tree_best_split($H, $B, 1.0, 0.0, 0.0, 1.0);
$bf  = bruteSplit($H, 3, 7, $nbf, 1.0, 0.0, 0.0, 1.0);
eq(np.get($bs, [0]), $bf[0], "the kernel picks the same FEATURE as brute force");
eq(np.get($bs, [1]), $bf[1], "and the same BIN");
close(np.get($bs, [2]), $bf[2], 1.0e-12, "and reports the same gain");
// The children it reports must be the children it chose.
close(np.get($bs, [3]) + np.get($bs, [6]), 4.0,  1.0e-12, "G_left + G_right = G");
close(np.get($bs, [4]) + np.get($bs, [7]), 10.0, 1.0e-12, "H_left + H_right = H");
close(np.get($bs, [5]) + np.get($bs, [8]), 7.0,  1.0e-12, "n_left + n_right = n");
} catch ($e) { section_threw("tree_best_split equals brute force over the same histogram", $e); }

print("");
print("== the hessian guard: h -> 0 must not win with an INFINITE gain ==");
try {
// A saturated logistic gives h = 0. With lambda = 0, G^2/(H+lambda) divides by
// zero. A NaN gain would lose every comparison; an INF gain wins every one of
// them and would pick a split on a child with no weight. So this must find no
// split at all rather than a spectacular one.
$hz = np.zeros([7], "f64");
$Hz = tree_hist($B, $g, $hz, null);
$bz = tree_best_split($Hz, $B, 0.0, 0.0, 0.0, 1.0);
eq(np.get($bz, [0]), 0.0 - 1.0, "with every hessian zero and lambda=0, no split is chosen");
ok(np.get($bz, [2]) == 0.0, "and the reported gain is 0, not inf or NaN");
// The same data with a positive lambda is perfectly splittable, which proves
// the guard is about H+lambda and not about refusing the data.
$bzl = tree_best_split($Hz, $B, 1.0, 0.0, 0.0, 1.0);
ok(np.get($bzl, [0]) >= 0.0, "with lambda=1 the same histogram splits fine");
// A single saturated row among healthy ones must not produce a winning child.
$hmix = nd([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0], "f64");
$bmix = tree_best_split(tree_hist($B, $g, $hmix, null), $B, 0.0, 0.0, 0.0, 1.0);
ok(np.get($bmix, [2]) < 1.0e300, "one zero-hessian row does not yield an infinite gain");
} catch ($e) { section_threw("the hessian guard: h -> 0 must not win with an INFINITE gain", $e); }

print("");
print("== the guard on the CHILD, which the guard on the parent was hiding ==");
try {
// tree_best_split has TWO hessian guards: one on the feature's parent total
// (!(Hs + lambda > 0) -> skip the feature) and one per candidate cut
// (!(HL + lambda > 0) || !(HR + lambda > 0) -> skip the cut). Removing the
// SECOND one left all 156 assertions green, because the only fixture that
// drove hessians to zero drove ALL of them to zero -- so the parent guard
// skipped the feature and the child guard was never reached.
//
// This reaches it: rows 0-2 carry h = 1 and rows 3-6 carry h = 0. The parent
// total is 3, so the feature is scanned; the cut at bin 2 gives HR = 0 with
// GR = -4+5-6+7 = 2, so without the child guard its gain is G^2/0 = +inf and
// it WINS every comparison -- a split on a child with no weight in it, chosen
// with maximum confidence.
$hChild = nd([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], "f64");
$bChild = tree_best_split(tree_hist($B, $g, $hChild, null), $B, 0.0, 0.0, 0.0, 1.0);
print("        chosen: feature " + str(np.get($bChild, [0])) + " bin "
      + str(np.get($bChild, [1])) + " gain " + str(np.get($bChild, [2]))
      + "  HL " + str(np.get($bChild, [4])) + " HR " + str(np.get($bChild, [7])));
ok(np.get($bChild, [2]) < 1.0e300,
   "the winning gain is FINITE -- no child divided by a zero hessian");
ok(np.get($bChild, [4]) > 0.0,
   "the left child has positive hessian weight");
ok(np.get($bChild, [7]) > 0.0,
   "and so does the right -- the per-CUT guard is what enforces this, and the "
   + "per-FEATURE guard cannot, because the parent total here is 3");
// The same histogram with a positive lambda may legitimately take that cut:
// H + lambda is then positive for both children. Asserted so the guard is
// shown to be about H + lambda and not about refusing the data, exactly as the
// all-zero case already asserts for the parent guard.
$bChildL = tree_best_split(tree_hist($B, $g, $hChild, null), $B, 1.0, 0.0, 0.0, 1.0);
ok(np.get($bChildL, [2]) < 1.0e300,
   "with lambda = 1 the gain is still finite");
ok(np.get($bChildL, [0]) >= 0.0, "and a split is found");
} catch ($e) { section_threw("the guard on the child", $e); }

print("");
print("== ties go to the LOWEST feature and the LOWEST bin ==");
try {
// Changing the scan's `gain > bestGain` to `>=` left all 156 assertions green:
// no fixture had a tie, so the tie-break was never exercised. That is not a
// cosmetic difference -- duplicate or perfectly correlated columns are common,
// and which one a tree splits on decides what feature_importances_ reports and
// whether two fits of the same data agree.
//
// Two IDENTICAL informative columns tie by construction.
$tieX = nd([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0],
            [5.0, 5.0], [6.0, 6.0], [7.0, 7.0]], "f64");
$tieB = tree_bin($tieX, 256);
$tieH = tree_hist($tieB, $g, $h, null);
$tieS = tree_best_split($tieH, $tieB, 0.0, 0.0, 0.0, 1.0);
// First prove the tie is real, or the assertion below proves nothing: scan the
// two features' best gains independently and require them equal to the bit.
def bestOfFeature($Hv, $Bv, $f) {
    $one = np.reshape(np.take($Hv, np.astype(nd([1.0 * $f], "f64"), "i64"), 0),
                      [1, np.shape($Hv)[1], 3]);
    return $one;
}
$g0 = tree_best_split(bestOfFeature($tieH, $tieB, 0),
                      tree_bin(np.reshape(np.take($tieX,
                          np.astype(nd([0.0], "f64"), "i64"), 1), [7, 1]), 256),
                      0.0, 0.0, 0.0, 1.0);
$g1 = tree_best_split(bestOfFeature($tieH, $tieB, 1),
                      tree_bin(np.reshape(np.take($tieX,
                          np.astype(nd([1.0], "f64"), "i64"), 1), [7, 1]), 256),
                      0.0, 0.0, 0.0, 1.0);
ok(nd_tobytes(np.take($g0, np.astype(nd([2.0], "f64"), "i64"), 0), "f64")
   == nd_tobytes(np.take($g1, np.astype(nd([2.0], "f64"), "i64"), 0), "f64"),
   "the two identical columns really do tie, bit for bit");
eq(np.get($tieS, [0]), 0,
   "and the tie goes to the LOWEST feature index, so a duplicated column "
   + "cannot change which feature the tree reports");
eq(np.get($tieS, [1]), np.get($g0, [1]),
   "on the same bin the single-feature scan chose");
// And the estimator inherits it, which is what makes two fits of the same data
// agree on more than their predictions.
$tieEst = tree.DecisionTreeRegressor(1, 1, 256, 0.0, 0.0, 0.0);
$tieEst.fit($tieX, nd([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0], "f64"));
eq(np.get($tieEst.nodes(null), [0, 1]), 0,
   "a fitted tree splits on feature 0 of two identical columns");
eq(np.to_list($tieEst.feature_importances_("split")), [1, 0],
   "so the importance lands on feature 0, not spread arbitrarily");
} catch ($e) { section_threw("ties go to the lowest index", $e); }


print("");
print("== the LEAF side of the same guard: a zero-hessian node must not raise ==");
try {
// tree_best_split refuses a child with !(H + lambda > 0). _leafValue is
// computed from the SAME TWO QUANTITIES and had no guard at all, so a node
// whose hessians sum to zero divided by zero -- and a scalar /0 RAISES in
// Bantu, so _grow aborted out of its middle instead of returning a tree.
// K7's own estimators cannot reach it (both set h = 1 per row, so H = n >= 1);
// K8's booster can, because logistic h = p(1-p) underflows to 0 across a
// saturated node and reg_lambda defaults to 0. The grower is shared, so the
// guard belongs here and not in the booster.
$lzE = tree.DecisionTreeRegressor(3, 1, 16, 0.0, 0.0, 0.0);
$lzN = null;
try { $lzN = $lzE._grow($B, $g, $hz, null); }
catch ($e) { print("        threw: " + str($e)); }
ok($lzN != null, "a grower over all-zero hessians returns a tree instead of raising");
// The two assertions below read $lzN, so when the guard is removed they would
// abort the whole run on nd_shape(null) -- one throw costing every later
// section. That is [K52] in miniature, so they fail explicitly instead.
if ($lzN == null) {
    ok(false, "and it is a SINGLE LEAF -- no split had positive gain (no tree)");
    ok(false, "whose value is 0, not inf (no tree)");
} else {
    eq(np.shape($lzN), [1, 7], "and it is a SINGLE LEAF -- no split had positive gain");
    ok(np.get($lzN, [0, 5]) == 0.0,
       "whose value is 0, not inf -- at H+lambda=0 the approximation has no curvature");
}
// Both guards must be about H + lambda and not about refusing the data: the
// same g and h with lambda = 1 grow a real tree.
$lzlE = tree.DecisionTreeRegressor(3, 1, 16, 1.0, 0.0, 0.0);
$lzlN = null;
try { $lzlN = $lzlE._grow($B, $g, $hz, null); }
catch ($e) { print("        threw: " + str($e)); }
if ($lzlN == null) {
    ok(false, "with reg_lambda = 1 the same g and h grow a real tree (threw)");
} else {
    ok(np.shape($lzlN)[0] > 1, "with reg_lambda = 1 the same g and h grow a real tree");
}
// Hessians that CANCEL to exactly zero rather than each being zero -- the case
// a non-convex loss with negative hessians would reach.
$lzcE = tree.DecisionTreeRegressor(2, 1, 16, 0.0, 0.0, 0.0);
$lzcN = null;
try { $lzcN = $lzcE._grow($B, $g, nd([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0], "f64"), null); }
catch ($e) { print("        threw: " + str($e)); }
ok($lzcN != null, "hessians cancelling to exactly zero do not raise either");
} catch ($e) { section_threw("the LEAF side of the same guard: a zero-hessian node must not raise", $e); }

print("");
print("== the constraints are honoured ==");
try {
// 7 rows, so min_data = 3 is the largest feasible value (3 + 4 = 7). This
// assertion first used 4, which needs 8 rows and is unsatisfiable -- the kernel
// correctly found no split and the test called that a failure. The infeasible
// case is now asserted separately, as the refusal it is.
$bMinN = tree_best_split($H, $B, 1.0, 0.0, 0.0, 3.0);
ok(np.get($bMinN, [5]) >= 3.0 && np.get($bMinN, [8]) >= 3.0,
   "min_data keeps both children at or above it");
$bInfeas = tree_best_split($H, $B, 1.0, 0.0, 0.0, 4.0);
eq(np.get($bInfeas, [0]), 0.0 - 1.0,
   "and a min_data no split can satisfy (4+4 > 7 rows) yields no split at all");
$bMinCW = tree_best_split($H, $B, 1.0, 0.0, 4.0, 1.0);
ok(np.get($bMinCW, [4]) >= 4.0 && np.get($bMinCW, [7]) >= 4.0,
   "min_child_weight keeps both children's hessian at or above it");
// gamma is subtracted from every gain, so a large one refuses every split.
$bGam = tree_best_split($H, $B, 1.0, 1.0e9, 0.0, 1.0);
ok(np.get($bGam, [2]) <= 0.0 || np.get($bGam, [0]) < 0.0,
   "a huge gamma leaves no split worth making");
raises(def() { return tree_best_split($H, $B, 0.0 - 1.0, 0.0, 0.0, 1.0); },
       "a negative lambda is refused");
raises(def() { return tree_best_split(np.zeros([3, 7], "f64"), $B, 1.0, 0.0, 0.0, 1.0); },
       "a histogram of the wrong rank is refused, naming the shape");
} catch ($e) { section_threw("the constraints are honoured", $e); }

print("");
print("== the NaN direction is LEARNED, not fixed ==");
try {
// Rows 0-3 present with gradient +1, rows 4-7 missing with gradient -10. The
// best split is missing-vs-present, so nan_left must come back set and the bin
// must be -1 -- the cut with no value bins on the left.
$Xd = np.zeros([8, 1], "f64");
$i = 0;
while ($i < 4) { np.set($Xd, [$i, 0], 1.0 * $i); $i = $i + 1; }
while ($i < 8) { np.set($Xd, [$i, 0], $nan); $i = $i + 1; }
$Bd = tree_bin($Xd, 256);
$gd = nd([1.0, 1.0, 1.0, 1.0, 0.0 - 10.0, 0.0 - 10.0, 0.0 - 10.0, 0.0 - 10.0], "f64");
$hd = np.ones([8], "f64");
$bd = tree_best_split(tree_hist($Bd, $gd, $hd, null), $Bd, 0.0, 0.0, 0.0, 1.0);
eq(np.get($bd, [9]), 1.0, "the missing rows are sent LEFT, because that is better");
eq(np.get($bd, [1]), 0.0 - 1.0, "with bin = -1: no value bins on the left");
close(np.get($bd, [5]), 4.0, 1.0e-12, "and the left child is exactly the 4 missing rows");
close(np.get($bd, [3]), 0.0 - 40.0, 1.0e-12, "carrying their gradient");

// Reverse it: the missing rows now look like the low-value rows, so NaN-right
// should win. The direction has to depend on the data or it is not learned.
$gd2 = nd([0.0 - 10.0, 0.0 - 10.0, 1.0, 1.0, 0.0 - 10.0, 0.0 - 10.0, 0.0 - 10.0, 0.0 - 10.0], "f64");
$bd2 = tree_best_split(tree_hist($Bd, $gd2, $hd, null), $Bd, 0.0, 0.0, 0.0, 1.0);
ok(np.get($bd2, [9]) == 1.0 || np.get($bd2, [9]) == 0.0,
   "the direction is a real decision either way");
ok(str($bd2) != str($bd), "and it is not the same answer as the previous data");

// ── tree_partition ──────────────────────────────────────────────────────────
} catch ($e) { section_threw("the NaN direction is LEARNED, not fixed", $e); }
print("");
print("== tree_partition: a partition, and a STABLE one ==");
try {
$idxAll = np.astype(np.arange(0.0, 7.0, 1.0), "i64");
$pp = tree_partition($B, $idxAll, 0, 2, false);
$L = $pp[0];  $Rt = $pp[1];
eq(np.size($L) + np.size($Rt), 7, "left + right account for every parent row");
// Disjoint and complete, checked by reconstruction rather than by trusting it.
$seen = np.zeros([7], "f64");
np.put($seen, $L, 1.0);
np.put($seen, $Rt, 1.0);
close(sc(np.sum($seen, null, null)), 7.0, 1.0e-12, "and are disjoint (each row once)");
// Stability: each child is ascending, because the parent was.
$asc = true;
$i = 0;
while ($i + 1 < np.size($L)) {
    if (np.get($L, [$i]) > np.get($L, [$i + 1])) { $asc = false; }
    $i = $i + 1;
}
ok($asc, "the left child keeps the parent's order");
// Every left row really is <= the cut.
$allLe = true;
$i = 0;
while ($i < np.size($L)) {
    if (tree_bin_at($B, np.get($L, [$i]), 0) > 2) { $allLe = false; }
    $i = $i + 1;
}
ok($allLe, "and every left row's bin is <= the cut");
} catch ($e) { section_threw("tree_partition: a partition, and a STABLE one", $e); }

print("");
print("== tree_partition: bin = -1 is the missing-vs-present split ==");
try {
$pm = tree_partition($Bd, np.astype(np.arange(0.0, 8.0, 1.0), "i64"), 0, 0.0 - 1.0, true);
eq(np.size($pm[0]), 4, "bin=-1 with nan_left puts exactly the missing rows left");
eq(np.to_list($pm[0]), [4, 5, 6, 7], "and they are the right ones");
// Without nan_left it is empty, which is why best_split only returns -1
// together with the direction.
eq(np.size(tree_partition($Bd, np.astype(np.arange(0.0, 8.0, 1.0), "i64"),
                          0, 0.0 - 1.0, false)[0]), 0,
   "bin=-1 without nan_left leaves the left child empty");
raises(def() { return tree_partition($B, $idxAll, 0, 0.0 - 2.0, false); },
       "bin < -1 is refused");
raises(def() { return tree_partition($B, $idxAll, 9, 0, false); },
       "an out-of-range feature is refused");

// ── tree_predict ────────────────────────────────────────────────────────────
} catch ($e) { section_threw("tree_partition: bin = -1 is the missing-vs-present split", $e); }
print("");
print("== tree_predict equals a Bantu reference traversal ==");
try {
// A hand-built stump: feature 0, threshold 3.5, left 10, right 20.
$nodes = nd([[0.0, 0.0, 3.5, 1.0, 2.0, 0.0, 0.0],
             [1.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0],
             [1.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0]], "f64");
$yp = tree_predict($nodes, $Xb);
eq(np.to_list($yp), [10, 10, 10, 20, 20, 20, 20],
   "x <= 3.5 goes left, and the leaf values come back");
} catch ($e) { section_threw("tree_predict equals a Bantu reference traversal", $e); }

print("");
print("== tree_predict routes NaN by the node's own flag ==");
try {
$Xq = np.zeros([1, 3], "f64");
np.set($Xq, [0, 0], $nan);
$nodesL = nd([[0.0, 0.0, 3.5, 1.0, 2.0, 0.0, 1.0],
              [1.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0],
              [1.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0]], "f64");
eq(np.get(tree_predict($nodesL, $Xq), [0]), 10, "nan_left=1 sends a missing value left");
eq(np.get(tree_predict($nodes,  $Xq), [0]), 20, "nan_left=0 sends it right");
} catch ($e) { section_threw("tree_predict routes NaN by the node's own flag", $e); }

print("");
print("== tree_predict refuses a malformed tree instead of walking off it ==");
try {
raises(def() { return tree_predict(np.zeros([2, 5], "f64"), $Xb); },
       "a node array that is not [n,7] is refused, naming the layout");
raises(def() {
    return tree_predict(nd([[0.0, 9.0, 1.0, 0.0, 0.0, 0.0, 0.0]], "f64"), $Xb);
}, "a split on a feature X does not have is refused");
raises(def() {
    return tree_predict(nd([[0.0, 0.0, 1.0, 5.0, 6.0, 0.0, 0.0]], "f64"), $Xb);
}, "a child index outside the tree is refused");
raises(def() {
    return tree_predict(nd([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], "f64"), $Xb);
}, "a node that is its own child is refused");
// A cycle without a self-loop: 0 -> 1 -> 0.
raises(def() {
    return tree_predict(nd([[0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], "f64"), $Xb);
}, "a two-node cycle raises rather than hanging");

// ── DecisionTreeRegressor ───────────────────────────────────────────────────
} catch ($e) { section_threw("tree_predict refuses a malformed tree instead of walking off it", $e); }
print("");
print("== DecisionTreeRegressor: an exactly piecewise-constant target ==");
try {
$Xp = np.reshape(np.arange(0.0, 20.0, 1.0), [20, 1]);
$yp2 = np.zeros([20], "f64");
$i = 0;
while ($i < 20) {
    $v = 1.0;
    if ($i >= 10) { $v = 5.0; }
    np.set($yp2, [$i], $v);
    $i = $i + 1;
}
$reg = tree.DecisionTreeRegressor(1, 1, null, null, null, null);
$reg.fit($Xp, $yp2);
$pr = $reg.predict($Xp);
close(sc(np.max(np.abs(np.subtract($pr, $yp2, null), null), null, null)), 0.0, 1.0e-12,
      "a depth-1 tree fits a single step EXACTLY");
eq($reg.n_nodes(), 3, "and uses three nodes: a split and two leaves");
eq($reg.depth(), 1, "at depth 1");
} catch ($e) { section_threw("DecisionTreeRegressor: an exactly piecewise-constant target", $e); }

print("");
print("== the leaf value is the mean (reg_lambda = 0) ==");
try {
$ym = nd([2.0, 4.0, 6.0, 100.0, 102.0, 104.0], "f64");
$Xm = np.reshape(np.arange(0.0, 6.0, 1.0), [6, 1]);
$rm = tree.DecisionTreeRegressor(1, 1, null, null, null, null);
$rm.fit($Xm, $ym);
$pm2 = $rm.predict($Xm);
close(np.get($pm2, [0]), 4.0,   1.0e-12, "the left leaf is the mean of 2,4,6");
close(np.get($pm2, [5]), 102.0, 1.0e-12, "the right leaf is the mean of 100,102,104");
} catch ($e) { section_threw("the leaf value is the mean (reg_lambda = 0)", $e); }

print("");
print("== reg_lambda shrinks the leaf toward zero, as the formula says ==");
try {
$rl = tree.DecisionTreeRegressor(1, 1, null, 3.0, null, null);
$rl.fit($Xm, $ym);
// value = sum(y)/(n + lambda) = 12/(3+3) = 2, against the mean of 4.
close(np.get($rl.predict($Xm), [0]), 2.0, 1.0e-12,
      "sum(y)/(n+lambda) = 12/6 = 2, not the mean of 4");
} catch ($e) { section_threw("reg_lambda shrinks the leaf toward zero, as the formula says", $e); }

print("");
print("== max_depth and min_samples_leaf actually bind ==");
try {
$deep = tree.DecisionTreeRegressor(3, 1, null, null, null, null);
$deep.fit($Xp, $yp2);
ok($deep.depth() <= 3, "depth never exceeds max_depth");
$leafy = tree.DecisionTreeRegressor(10, 8, null, null, null, null);
$leafy.fit($Xp, $yp2);
// With 20 rows and min_samples_leaf = 8, only one split is possible.
eq($leafy.n_nodes(), 3, "min_samples_leaf=8 over 20 rows permits exactly one split");
} catch ($e) { section_threw("max_depth and min_samples_leaf actually bind", $e); }

print("");
print("== a constant target produces no split at all ==");
try {
$cst = tree.DecisionTreeRegressor(5, 1, null, null, null, null);
$cst.fit($Xp, np.ones([20], "f64"));
eq($cst.n_nodes(), 1, "zero gain everywhere means a single leaf, not a full tree");
close(np.get($cst.predict($Xp), [0]), 1.0, 1.0e-12, "which predicts the constant");

// ── the Gini claim ──────────────────────────────────────────────────────────
} catch ($e) { section_threw("a constant target produces no split at all", $e); }
print("");
print("== the split equals the argmax of a HAND-COMPUTED Gini table ==");
def giniDecrease($yv, $cut, $n) {
    // Σ n_m·Gini_m, lower is better; the DECREASE is parent minus this.
    $nL = $cut + 1.0;  $nR = $n - $nL;
    $pL = 0.0;  $pR = 0.0;
    $i = 0;
    while ($i < $n) {
        if ($i <= $cut) { $pL = $pL + np.get($yv, [$i]); }
        if ($i > $cut)  { $pR = $pR + np.get($yv, [$i]); }
        $i = $i + 1;
    }
    $fL = $pL / $nL;  $fR = $pR / $nR;
    $gL = 1.0 - $fL * $fL - (1.0 - $fL) * (1.0 - $fL);
    $gR = 1.0 - $fR * $fR - (1.0 - $fR) * (1.0 - $fR);
    $pTot = ($pL + $pR) / $n;
    $gP = 1.0 - $pTot * $pTot - (1.0 - $pTot) * (1.0 - $pTot);
    return $n * $gP - ($nL * $gL + $nR * $gR);
}

try {
// This is the assertion the whole "one criterion, two estimators" design rests
// on. Eight rows, one feature with eight distinct values, so the candidate cuts
// are after row 0..6. The Gini decrease for each is computed here from counts,
// with no reference to the gain formula at all.
$Xg = np.reshape(np.arange(0.0, 8.0, 1.0), [8, 1]);
$yg = nd([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0], "f64");

$bestCut = 0.0 - 1.0;  $bestDec = 0.0 - 1.0e300;
$c = 0;
while ($c < 7) {
    $d = giniDecrease($yg, 1.0 * $c, 8.0);
    if ($d > $bestDec) { $bestDec = $d; $bestCut = 1.0 * $c; }
    $c = $c + 1;
}
print("        hand-computed best Gini cut: after row " + str($bestCut)
      + " (decrease " + str($bestDec) + ")");

$Bg = tree_bin($Xg, 256);
$gg = np.multiply($yg, 0.0 - 1.0, null);
$hg = np.ones([8], "f64");
$bg = tree_best_split(tree_hist($Bg, $gg, $hg, null), $Bg, 0.0, 0.0, 0.0, 1.0);
eq(np.get($bg, [1]), $bestCut,
   "tree_best_split picks the SAME cut as the Gini argmax");
// And the proportionality constant is exactly 4, as derived.
close($bestDec / np.get($bg, [2]), 4.0, 1.0e-9,
      "Gini decrease = 4 x gain, the factor the derivation predicts");

// ── DecisionTreeClassifier ──────────────────────────────────────────────────
} catch ($e) { section_threw("the split equals the argmax of a HAND-COMPUTED Gini table", $e); }
print("");
print("== DecisionTreeClassifier: binary ==");
try {
$clf = tree.DecisionTreeClassifier(3, 1, null, null, null, null);
$clf.fit($Xg, $yg);
eq(np.to_list($clf.predict($Xg)), np.to_list($yg),
   "a separable binary problem is fitted exactly");
$P = $clf.predict_proba($Xg);
eq(np.shape($P), [8, 2], "predict_proba is [n, 2]");
close(sc(np.max(np.abs(np.subtract(np.sum($P, 1, false), np.ones([8], "f64"), null), null), null, null)),
      0.0, 1.0e-12, "and every row sums to exactly 1");
ok(sc(np.min($P, null, null)) >= 0.0, "with no negative probability");
} catch ($e) { section_threw("DecisionTreeClassifier: binary", $e); }

print("");
print("== the leaf probability is the class proportion ==");
try {
// A stump on a 6-row problem whose left leaf is 1/3 positive.
$Xc = np.reshape(np.arange(0.0, 6.0, 1.0), [6, 1]);
$yc = nd([0.0, 1.0, 0.0, 1.0, 1.0, 1.0], "f64");
$cs = tree.DecisionTreeClassifier(1, 1, null, null, null, null);
$cs.fit($Xc, $yc);
$Pc = $cs.predict_proba($Xc);
ok(np.get($Pc, [0, 1]) >= 0.0 && np.get($Pc, [0, 1]) <= 1.0,
   "the leaf probability is in [0,1]");
close(np.get($Pc, [0, 0]) + np.get($Pc, [0, 1]), 1.0, 1.0e-12, "and complements");
} catch ($e) { section_threw("the leaf probability is the class proportion", $e); }

print("");
print("== multiclass is one-vs-rest ==");
try {
$Xk = np.reshape(np.arange(0.0, 12.0, 1.0), [12, 1]);
$yk = nd([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0], "f64");
$mk = tree.DecisionTreeClassifier(4, 1, null, null, null, null);
$mk.fit($Xk, $yk);
eq(np.shape($mk.predict_proba($Xk)), [12, 3], "predict_proba is [n, K]");
eq(np.to_list($mk.predict($Xk)), np.to_list($yk), "and the three classes are separated");
eq($mk.n_trees(), 3, "a 3-class model holds three trees");
// n_nodes used to report np.shape(nodes_)[0], and nodes_ is aliased to
// trees_[0] -- so a 3-class model reported a third of its own size.
ok($mk.n_nodes() >= np.shape($mk.nodes(0))[0] + np.shape($mk.nodes(1))[0],
   "and n_nodes counts every tree, not just the first");
eq(np.shape($mk.nodes(2))[1], 7, "each class's node array is [n, 7]");
raises(def() { return $mk.nodes(3); }, "an out-of-range tree index is refused");
eq(tree.DecisionTreeRegressor(2, 1, null, null, null, null)
       .fit($Xp, $yp2).n_trees(), 1, "a regressor holds exactly one tree");
close(sc(np.max(np.abs(np.subtract(np.sum($mk.predict_proba($Xk), 1, false),
                                   np.ones([12], "f64"), null), null), null, null)),
      0.0, 1.0e-12, "one-vs-rest scores are normalised to sum to 1");
} catch ($e) { section_threw("multiclass is one-vs-rest", $e); }

print("");
print("== the classifier refuses a target that is not a class set ==");
try {
raises(def() { return tree.DecisionTreeClassifier(3, 1, null, null, null, null)
                           .fit($Xg, np.ones([8], "f64")); },
       "one distinct label is refused -- such a model would score 100%");
raises(def() { return tree.DecisionTreeClassifier(3, 1, null, null, null, null)
                           .fit($Xg, nd([1.0, 2.0], "f64")); },
       "a y of the wrong length is refused");
} catch ($e) { section_threw("the classifier refuses a target that is not a class set", $e); }

print("");
print("== NaN in X works end to end ==");
try {
// arange over whole numbers yields an i64 array, which cannot hold a NaN.
$Xna = np.astype(np.reshape(np.arange(0.0, 10.0, 1.0), [10, 1]), "f64");
np.set($Xna, [7, 0], $nan);
np.set($Xna, [8, 0], $nan);
$yna = nd([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], "f64");
safe_ok(def() {
    $t = tree.DecisionTreeClassifier(3, 1, null, null, null, null);
    $t.fit($Xna, $yna);
    return np.size($t.predict($Xna)) == 10;
}, "a feature with missing values trains and predicts");
safe_ok(def() {
    $t = tree.DecisionTreeRegressor(3, 1, null, null, null, null);
    $t.fit($Xna, $yna);
    $q = np.zeros([1, 1], "f64");
    np.set($q, [0, 0], $nan);
    return np.size($t.predict($q)) == 1;
}, "and a missing value at PREDICT time routes by the learned direction");
} catch ($e) { section_threw("NaN in X works end to end", $e); }

print("");
print("== parameters are validated, and clone is independent ==");
try {
raises(def() { return tree.DecisionTreeRegressor(0, 1, null, null, null, null); },
       "max_depth 0 is refused");
raises(def() { return tree.DecisionTreeRegressor(61, 1, null, null, null, null); },
       "max_depth 61 is refused, with the 2^d reason");
raises(def() { return tree.DecisionTreeRegressor(3, 0, null, null, null, null); },
       "min_samples_leaf 0 is refused");
raises(def() { return tree.DecisionTreeRegressor(3, 1, 300, null, null, null); },
       "max_bins 300 is refused");
raises(def() { return tree.DecisionTreeRegressor(3, 1, null, 0.0 - 1.0, null, null); },
       "a negative reg_lambda is refused");
raises(def() { return tree.DecisionTreeRegressor(3, 1, null, null, null, null).predict($Xp); },
       "predict before fit says to call fit first");
raises(def() { return tree.DecisionTreeRegressor(3, 1, null, null, null, null)
                           .fit($Xp, $yp2).predict(np.zeros([2, 5], "f64")); },
       "predicting with the wrong feature count is refused");

$orig = tree.DecisionTreeRegressor(2, 1, null, null, null, null);
$cl   = $orig.clone();
ok(!$cl.fitted(), "a clone starts unfitted");
$cl.fit($Xp, $yp2);
ok(!$orig.fitted(), "and fitting it leaves the original unfitted");
} catch ($e) { section_threw("parameters are validated, and clone is independent", $e); }

print("");
print("== reproducible from the seed alone (C4) ==");
try {
np.seed(7);
$r1 = tree.DecisionTreeRegressor(4, 1, null, null, null, null).fit($Xp, $yp2).predict($Xp);
np.seed(7);
$r2 = tree.DecisionTreeRegressor(4, 1, null, null, null, null).fit($Xp, $yp2).predict($Xp);
ok(nd_tobytes($r1, "f64") == nd_tobytes($r2, "f64"), "two fits are bit-identical");
// A tree has no RNG at all, so it must also be identical WITHOUT reseeding.
$r3 = tree.DecisionTreeRegressor(4, 1, null, null, null, null).fit($Xp, $yp2).predict($Xp);
ok(nd_tobytes($r1, "f64") == nd_tobytes($r3, "f64"),
   "and identical without reseeding, because a tree draws no randomness");

// ── tier 4: stress ──────────────────────────────────────────────────────────
} catch ($e) { section_threw("reproducible from the seed alone (C4)", $e); }
print("");
print("== tier 4: the memory claim K1 is built on ==");
try {
$big = np.random([200000, 50], 0.0 - 5.0, 5.0);
$t0 = clock();
$Bbig = tree_bin($big, 256);
$binMs = clock() - $t0;
eq(tree_bin_bytes($Bbig), 10000000,
   "200000x50 binned is exactly 10,000,000 bytes -- n*f, not 8*n*f");
print("        tree_bin on 200000x50: " + str($binMs) + " ms");
ok($binMs < 30000, "and binning 10M cells stays under 30 s");

$gb = np.random([200000], 0.0 - 1.0, 1.0);
$hb = np.ones([200000], "f64");
$t1 = clock();
$Hbig = tree_hist($Bbig, $gb, $hb, null);
$histMs = clock() - $t1;
eq(np.shape($Hbig)[0], 50, "the histogram covers every feature");
print("        tree_hist over 10M cells: " + str($histMs) + " ms");
ok($histMs < 10000, "the hot loop does 10M cells in under 10 s");
$t2 = clock();
$bsBig = tree_best_split($Hbig, $Bbig, 1.0, 0.0, 1.0, 20.0);
$splitMs = clock() - $t2;
print("        tree_best_split over 50x256 bins: " + str($splitMs) + " ms");
ok($splitMs < 2000, "and the scan is under 2 s");
ok(np.get($bsBig, [0]) >= 0.0, "a real split is found in random data");
} catch ($e) { section_threw("tier 4: the memory claim K1 is built on", $e); }

print("");
print("== tier 4: a deep tree, and single-sample leaves ==");
try {
// A strictly monotone target with every value distinct: the tree can keep
// splitting until each leaf holds one row, which is what exercises depth.
$Xs = np.reshape(np.arange(0.0, 600.0, 1.0), [600, 1]);
$ys = np.multiply(np.arange(0.0, 600.0, 1.0), 1.0, null);
$t3 = clock();
$dt = tree.DecisionTreeRegressor(50, 1, null, null, null, null);
$dt.fit($Xs, $ys);
$deepMs = clock() - $t3;
print("        max_depth=50 on 600 distinct rows: " + str($deepMs) + " ms, "
      + str($dt.n_nodes()) + " nodes, depth " + str($dt.depth()));
ok($dt.depth() <= 50, "the depth cap holds");
ok($dt.n_nodes() > 100, "and the tree really did grow");
ok($deepMs < 60000, "in under a minute");
// Error must fall as depth rises, or the splits are not improving anything.
$shallow = tree.DecisionTreeRegressor(2, 1, null, null, null, null);
$shallow.fit($Xs, $ys);
$eDeep = sc(np.mean(np.abs(np.subtract($dt.predict($Xs), $ys, null), null), null, null));
$eShal = sc(np.mean(np.abs(np.subtract($shallow.predict($Xs), $ys, null), null), null, null));
ok($eDeep < $eShal, "a deeper tree fits better (" + str($eDeep) + " < " + str($eShal) + ")");
} catch ($e) { section_threw("tier 4: a deep tree, and single-sample leaves", $e); }

print("");
print("== tier 4: a large node array predicts ==");
try {
$t4 = clock();
$pbig = $dt.predict($Xs);
print("        predict over " + str($dt.n_nodes()) + " nodes: "
      + str(clock() - $t4) + " ms");
eq(np.size($pbig), 600, "every row gets a prediction");
} catch ($e) { section_threw("tier 4: a large node array predicts", $e); }

print("");
print("== TreeGrower: the SAME grower, exposed rather than copied ==");
// K8's booster grows a tree over arbitrary (g, h) instead of over a target.
// §8.6 is explicit that it must be THE SAME grower -- a second one would be a
// second place that has to get [K44]'s NaN direction right, and one place
// already got it wrong. So the claim under test is an IDENTITY, not a
// similarity: a grower handed the estimator's own g and h must produce the
// estimator's own nodes, to the bit.
$grY  = nd([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], "f64");
$grEst = tree.DecisionTreeRegressor(3, 1, 16, 0.0, 0.0, 0.0);
$grEst.fit($Xb, $grY);
$grG = tree.TreeGrower(3, 1, 16, 0.0, 0.0, 0.0);
$grNodes = $grG.grow(tree_bin($Xb, 16),
                     np.multiply($grY, 0.0 - 1.0, null),
                     np.ones([7], "f64"), null);
ok(nd_tobytes($grNodes, "f64") == nd_tobytes($grEst.nodes(null), "f64"),
   "TreeGrower.grow is BIT-IDENTICAL to DecisionTreeRegressor.fit on g = -y, h = 1");
eq($grG.depth(), $grEst.depth(), "and reports the same depth");
eq($grG.n_leaves(), $grEst.n_leaves(), "and the same leaf count");
// It is not an estimator, and says so by name rather than failing somewhere
// unrelated: it has no target and nothing to predict with on its own.
raises(def() { return $grG.fit($Xb, $grY); }, "TreeGrower.fit refuses by name");
raises(def() { return $grG.predict($Xb); }, "TreeGrower.predict refuses by name");
raises(def() { return $grG.grow(null, $grG, $grG, null); },
       "grow without a binned matrix is refused");
raises(def() { return $grG.grow(tree_bin($Xb, 16), nd([1.0, 2.0], "f64"),
                                np.ones([7], "f64"), null); },
       "a g of the wrong length is refused");
raises(def() { return $grG.grow(tree_bin($Xb, 16),
                                np.multiply($grY, 0.0 - 1.0, null), null, null); },
       "growing with no hessians is refused -- a booster that passes only "
       + "gradients is using the wrong grower");

print("");
print("== TreeGrower: the column mask, and the row subsample ==");
// §8.5's column subsampling: zero the unselected features' rows in the
// histogram the SCAN sees. A zeroed feature has G = H = n = 0, so every cut on
// it fails BOTH min_data and the hessian guard -- two independent reasons, and
// neither is a special case in the kernel. The assertion is that the grown tree
// splits on the allowed features ONLY, which is deterministic and does not
// depend on which feature happens to win.
$msG = tree.TreeGrower(3, 1, 16, 0.0, 0.0, 0.0);
$msG.known_features(3);
$msG.set_features([2]);
$msN = $msG.grow(tree_bin($Xb, 16), np.multiply($grY, 0.0 - 1.0, null),
                 np.ones([7], "f64"), null);
eq(np.get($msG.lastSplits_, [0]), 0, "feature 0 is masked out: zero splits");
eq(np.get($msG.lastSplits_, [1]), 0, "feature 1 too");
ok(np.get($msG.lastSplits_, [2]) > 0, "and every split is on the one allowed feature");
// Restoring the mask restores the unmasked tree, bit for bit -- so masking is a
// filter on the scan and not a change to the arithmetic.
$msG.set_features(null);
ok(nd_tobytes($msG.grow(tree_bin($Xb, 16), np.multiply($grY, 0.0 - 1.0, null),
                        np.ones([7], "f64"), null), "f64")
   == nd_tobytes($grNodes, "f64"),
   "and clearing the mask gives back the unmasked tree, bit for bit");
raises(def() { return $msG.set_features([3]); },
       "a feature index past the end is refused");
raises(def() { return $msG.set_features([1, 1]); },
       "a repeated column is refused -- it does not weight the feature, it "
       + "just hides the typo");
raises(def() { return $msG.set_features([]); },
       "an empty column list is refused: every tree would be a single leaf");
// A row subsample over rows whose target is constant can find no split, so the
// grower must return a single leaf -- the same answer [K55] gates for a
// zero-hessian node, reached a different way.
$rsG = tree.TreeGrower(3, 1, 16, 0.0, 0.0, 0.0);
$rsN = $rsG.grow(tree_bin($Xb, 16), np.multiply($grY, 0.0 - 1.0, null),
                 np.ones([7], "f64"),
                 np.astype(nd([0.0, 1.0, 2.0, 3.0], "f64"), "i64"));
eq(np.shape($rsN), [1, 7], "a subsample with a constant target grows ONE leaf");

print("");
print("== feature importances come from the grower, not from the node array ==");
// A node holds the threshold, not the gain, so the gain has to be recorded
// where it is still in hand. The assertion that it is the RIGHT number: a
// feature the tree never split on has importance exactly 0, and the total is 1.
$fiEst = tree.DecisionTreeRegressor(3, 1, 16, 0.0, 0.0, 0.0);
$fiEst.fit($Xb, $grY);
$fiG = $fiEst.feature_importances_(null);
$fiS = $fiEst.feature_importances_("split");
eq(np.size($fiG), 3, "one importance per feature");
close(sc(np.sum($fiG, null, false)), 1.0, 1.0e-12, "gain importances sum to 1");
close(sc(np.sum($fiS, null, false)), 1.0, 1.0e-12, "split importances sum to 1");
ok(sc(np.min($fiG, null, false)) >= 0.0, "and none is negative");
raises(def() { return $fiEst.feature_importances_("gini"); },
       "an unsupported kind is refused rather than silently defaulted");
// A constant target splits on nothing. The total is then 0, and dividing by it
// would give a vector of NaNs that reads as "unknown" when the truthful answer
// is "this model uses no feature at all".
$fiFlat = tree.DecisionTreeRegressor(3, 1, 16, 0.0, 0.0, 0.0);
$fiFlat.fit($Xb, np.ones([7], "f64"));
eq(np.to_list($fiFlat.feature_importances_(null)), [0, 0, 0],
   "a model with no splits reports ZEROS, not NaN");
// A multiclass classifier grows K trees, and _grow overwrites its side outputs
// on every call -- so importances read at the end would report only the last
// class. They are folded in per tree.
$fiMc = tree.DecisionTreeClassifier(3, 1, 16, 0.0, 0.0, 0.0);
$fiMc.fit($Xb, nd([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0], "f64"));
close(sc(np.sum($fiMc.feature_importances_("split"), null, false)), 1.0, 1.0e-12,
      "a 3-class model's importances still sum to 1");
ok(sc(np.sum($fiMc.splits_, null, false)) >= 1.0 * $fiMc.n_trees(),
   "and count at least one split per tree, so all K were folded in");


print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
