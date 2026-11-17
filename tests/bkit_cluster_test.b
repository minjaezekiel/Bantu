// ════════════════════════════════════════════════════════════════════════
//  bkit_cluster_test.b — KMeans.
//
//  THE FIRST SECTION IS THE ONE THAT MATTERS, and it is first because of how
//  this file's first bug behaved. `_dist2` computes
//  ‖a−c‖² = ‖a‖² − 2a·c + ‖c‖² so that an assignment step is one matmul; the
//  first version SUBTRACTED ‖c‖². Every distance came out negative, the
//  defensive `maximum(d, 0)` clamp turned them all into zeros, and the fit
//  completed and returned plausible-looking centres. The only visible symptom
//  was an inertia of exactly 0.
//
//  So the identity is gated against a BRUTE-FORCE double loop -- the
//  definition, not a rearrangement of it -- and nothing about distances is
//  inferred from the behaviour of anything downstream. A defensive clamp makes
//  a wrong answer quiet ([K70]).
//
//  Run:  bantu run tests/bkit_cluster_test.b
// ════════════════════════════════════════════════════════════════════════

include "../numba/numba.b" as np;
include "../bkit/base.b" as base;
include "../bkit/cluster.b" as cl;

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
def section_threw($name, $e) {
    $R.fail = $R.fail + 1;
    print("  FAIL  [section aborted] " + $name);
    print("          threw: " + str($e));
    return null;
}
def sc($x) { return np.get(np.reshape($x, [1]), [0]); }
def maxAbsDiff($a, $b) {
    return sc(np.max(np.abs(np.subtract($a, $b, null), null), null, false));
}
def bits($a) { return nd_tobytes($a, "f64"); }

print("========================================");
print("  bkit/cluster.b — KMeans");
print("========================================");

print("");
print("== the distance identity, against the DEFINITION ==");
try {
// Brute force: one Bantu loop over every (row, centre) pair, summing squared
// differences. Slow by design and small by design -- it is the definition, and
// the whole point is that it shares no arithmetic with the matmul form.
def bruteDist2($A, $C) {
    $n = np.shape($A)[0];
    $k = np.shape($C)[0];
    $f = np.shape($A)[1];
    $out = [];
    $i = 0;
    while ($i < $n) {
        $row = [];
        $j = 0;
        while ($j < $k) {
            $s = 0.0;
            $d = 0;
            while ($d < $f) {
                $delta = np.get($A, [$i, $d]) - np.get($C, [$j, $d]);
                $s = $s + $delta * $delta;
                $d = $d + 1;
            }
            append($row, $s);
            $j = $j + 1;
        }
        append($out, $row);
        $i = $i + 1;
    }
    return np.array($out, "f64");
}
np.seed(2);
$dA = np.random([9, 3], 0.0 - 2.0, 2.0);
$dC = np.random([4, 3], 0.0 - 2.0, 2.0);
$fast = cl._dist2($dA, $dC);
$slow = bruteDist2($dA, $dC);
eq(np.shape($fast), [9, 4], "the distance matrix is [n_rows, n_centres]");
print("        worst difference from brute force: " + str(maxAbsDiff($fast, $slow)));
ok(maxAbsDiff($fast, $slow) < 1.0e-12,
   "and every entry matches the brute-force definition to 1e-12 -- the matmul "
   + "form is an optimisation, so it is checked against the definition and not "
   + "against a rearrangement of itself");
// Properties the identity must satisfy, which catch a sign error even without
// a reference to compare against.
$selfD = cl._dist2($dA, $dA);
$diagMax = 0.0;
$i = 0;
while ($i < 9) {
    if (abs(np.get($selfD, [$i, $i])) > $diagMax) {
        $diagMax = abs(np.get($selfD, [$i, $i]));
    }
    $i = $i + 1;
}
ok($diagMax < 1.0e-12,
   "_dist2(A, A) has a ZERO diagonal -- a point is at no distance from itself");
ok(sc(np.min($fast, null, false)) >= 0.0,
   "and no squared distance is negative");
// THE CLAMP IS LOAD-BEARING, and the first fixture here was too benign to show
// it: at f = 3 and magnitude 1, the identity cancels exactly and removing the
// clamp changes nothing. Cancellation needs dimension AND magnitude --
// measured, at 1e7 across 50 features, 15 of 1600 entries go negative with a
// worst value of -1. Without the clamp those become NaN the moment transform
// takes a root.
np.seed(1);
$cl2 = np.multiply(np.random([40, 50], 0.0 - 1.0, 1.0), 1.0e7, null);
$selfBig = cl._dist2($cl2, $cl2);
print("        f=50 at 1e7, smallest squared distance: "
      + str(sc(np.min($selfBig, null, false))));
ok(sc(np.min($selfBig, null, false)) >= 0.0,
   "at f = 50 and magnitude 1e7 -- where the identity DOES cancel into the "
   + "negatives -- every squared distance is still >= 0");
$kBig = cl.KMeans({"n_clusters": 3, "n_init": 1, "max_iter": 5});
$kBig.fit($cl2, null);
eq(sc(np.sum(np.astype(np.isnan($kBig.transform($cl2), null), "f64"), null, false)), 0,
   "and transform yields no NaN there -- which is what the clamp is for");
// A hand-computable case, so the test does not rest entirely on one reference.
$hA = nd([[0.0, 0.0], [3.0, 4.0]], "f64");
$hC = nd([[0.0, 0.0]], "f64");
eq(np.to_list(cl._dist2($hA, $hC)), [[0], [25]],
   "and (3,4) is at squared distance 25 from the origin");
} catch ($e) { section_threw("the distance identity", $e); }

print("");
print("== three blobs, three clusters, sixty points each ==");
try {
// Centres known in advance, so "it clustered" is checkable rather than
// plausible.
np.seed(5);
$b1 = np.add(np.random([60, 2], 0.0 - 0.3, 0.3), nd([0.0, 0.0], "f64"), null);
$b2 = np.add(np.random([60, 2], 0.0 - 0.3, 0.3), nd([5.0, 0.0], "f64"), null);
$b3 = np.add(np.random([60, 2], 0.0 - 0.3, 0.3), nd([0.0, 5.0], "f64"), null);
$X  = np.concat([$b1, $b2, $b3], 0);
$km = cl.KMeans({"n_clusters": 3, "n_init": 5});
$km.fit($X, null);
eq(np.shape($km.cluster_centers()), [3, 2], "three centres in two dimensions");
eq(np.size(np.unique($km.labels())), 3, "and three distinct labels");
eq(np.to_list(np.bincount(np.astype($km.labels(), "i64"), 3)), [60, 60, 60],
   "with exactly sixty points each -- every blob is one cluster");
// Each true centre must be matched by some fitted centre.
$truth = nd([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]], "f64");
$pair  = cl._dist2($truth, $km.cluster_centers());
$worst = sc(np.max(np.min($pair, 1, false), null, false));
print("        worst squared distance from a true centre: " + str($worst));
ok($worst < 0.01, "every true centre is recovered to within 0.1");
// predict on the training data must agree with the stored labels, or the two
// are reporting different models.
ok(bits(np.astype($km.predict($X), "f64")) == bits(np.astype($km.labels(), "f64")),
   "predict reproduces the stored labels exactly");
// The inertia is the sum of minimum squared distances, recomputed here.
close($km.inertia(),
      sc(np.sum(np.min(cl._dist2($X, $km.cluster_centers()), 1, false), null, false)),
      1.0e-10, "the inertia is the sum of minimum squared distances");
close($km.score($X, null), 0.0 - $km.inertia(), 1.0e-12,
      "and score is its NEGATION, so higher is better and a grid search over "
      + "n_clusters is not inverted");
// transform is the distance to every centre, and its argmin is predict.
eq(np.shape($km.transform($X)), [180, 3], "transform gives a distance per centre");
ok(bits(np.astype(np.reshape(np.argmin($km.transform($X), 1, false), [180]), "f64"))
   == bits(np.astype($km.predict($X), "f64")),
   "whose argmin is predict -- the two cannot disagree about the same model");
ok(maxAbsDiff($km.transform($X),
              np.sqrt(cl._dist2($X, $km.cluster_centers()), null)) < 1.0e-12,
   "and it is the square ROOT of the squared distances, not the squares");
} catch ($e) { section_threw("three blobs", $e); }

print("");
print("== reproducible from the seed alone (C4) ==");
try {
def fitOnce($ni) {
    $m = cl.KMeans({"n_clusters": 3, "n_init": $ni});
    $m.fit($X, null);
    return $m;
}
np.seed(77);  $r1 = fitOnce(3);
np.seed(77);  $r2 = fitOnce(3);
ok(bits($r1.cluster_centers()) == bits($r2.cluster_centers()),
   "the same seed gives byte-identical centres");
// Without reseeding the k-means++ draws differ, which is what makes the
// assertion above mean something: an identical result here would prove only
// that nothing random was drawn.
$r3 = fitOnce(3);
ok(bits($r1.labels()) != bits($r3.labels())
   || bits($r1.cluster_centers()) != bits($r3.cluster_centers())
   || $r1.inertia() != $r3.inertia(),
   "and without reseeding the run differs, so k-means++ really is drawing");
// n_init keeps the LOWEST inertia, and this is exact rather than statistical:
// under one seed, an n_init = 5 run's first restart IS the n_init = 1 run, so
// its kept inertia cannot be worse.
np.seed(91);  $one  = fitOnce(1);
np.seed(91);  $five = fitOnce(5);
print("        n_init 1: " + str($one.inertia()) + "   n_init 5: "
      + str($five.inertia()));
ok($five.inertia() <= $one.inertia() + 1.0e-12,
   "n_init = 5 is never worse than n_init = 1 from the same seed -- its first "
   + "restart is that run");
// THAT INEQUALITY IS TRUE OF A RUN THAT KEEPS THE LAST RESTART TOO, and on
// well-separated blobs every restart finds the same optimum, so it cannot tell
// "keep the best" from "keep whichever finished last". The exact statement is
// available: Lloyd itself draws nothing, so five sequential n_init = 1 fits
// consume exactly the draws one n_init = 5 fit does, and the kept inertia must
// EQUAL the minimum over them.
//
// The fixture is deliberately awkward -- unequal sizes and spreads, so the
// restarts genuinely disagree -- because on easy data the minimum and the last
// are the same number and the assertion proves nothing again.
np.seed(44);
$hard = np.concat([np.add(np.random([80, 2], 0.0 - 1.4, 1.4), nd([0.0, 0.0], "f64"), null),
                   np.add(np.random([12, 2], 0.0 - 0.1, 0.1), nd([3.0, 0.2], "f64"), null),
                   np.add(np.random([12, 2], 0.0 - 0.1, 0.1), nd([3.4, 0.0], "f64"), null),
                   np.add(np.random([40, 2], 0.0 - 0.9, 0.9), nd([8.0, 6.0], "f64"), null)], 0);
def fitHard($ni) {
    $m = cl.KMeans({"n_clusters": 4, "n_init": $ni});
    $m.fit($hard, null);
    return $m.inertia();
}
np.seed(123);
$kept = fitHard(5);
np.seed(123);
$each = [];
$r = 0;
while ($r < 5) { append($each, fitHard(1)); $r = $r + 1; }
$lo = $each[0];
$hi = $each[0];
$r = 1;
while ($r < 5) {
    if ($each[$r] < $lo) { $lo = $each[$r]; }
    if ($each[$r] > $hi) { $hi = $each[$r]; }
    $r = $r + 1;
}
print("        restarts: " + str($each));
ok($hi - $lo > 1.0e-9,
   "the restarts genuinely disagree on this fixture (spread "
   + str($hi - $lo) + "), so the next assertion can distinguish best from last");
close($kept, $lo, 1.0e-9,
   "and the kept inertia IS the minimum over the restarts, not the last one");
} catch ($e) { section_threw("reproducible from the seed", $e); }

print("");
print("== an empty cluster is RELOCATED, not dropped ==");
try {
// Four clusters asked of three distinct points. One cluster must come up empty
// on some iteration, and the caller still asked for four: dropping it would
// return fewer centres than requested, which is a wrong answer that looks like
// a right one, and a later predict would index a centre that does not exist.
$few = nd([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
           [9.0, 9.0], [9.0, 9.0],
           [0.0, 9.0], [0.0, 9.0]], "f64");
np.seed(13);
$ke = cl.KMeans({"n_clusters": 4, "n_init": 3});
$ke.fit($few, null);
eq(np.shape($ke.cluster_centers()), [4, 2],
   "four centres come back for four requested, even with three distinct points");
ok(sc(np.sum(np.astype(np.isnan($ke.cluster_centers(), null), "f64"), null, false)) == 0.0,
   "and none of them is NaN -- an empty cluster's mean would be 0/0");
// Every label must index a real centre.
ok(sc(np.max(np.astype($ke.labels(), "f64"), null, false)) < 4.0,
   "every label indexes one of the four");
ok(sc(np.min(np.astype($ke.labels(), "f64"), null, false)) >= 0.0, "and none is negative");
// predict must still work, which is the practical consequence.
eq(np.size($ke.predict($few)), 7, "and predict works on all seven rows");
// WHAT THESE ASSERTIONS DO AND DO NOT SHOW, stated because the obvious reading
// is wrong. They pin the INVARIANT -- k centres come back, none is NaN, every
// label indexes a real centre, predict works -- which is what the relocation
// branch exists to protect. They do NOT distinguish relocating the stranded
// centre from leaving it where it was, and no fixture here does:
//
//   * k-means++ places every initial centre ON a data row and spreads them by
//     D-squared, so a stranded centre is already on a row either way;
//   * measured, over 30 seeds with k = 3 across four separated blobs -- more
//     groups than clusters, so an empty cluster is never forced -- NO cluster
//     ends empty, with or without relocation. k-means++ simply does not strand
//     one when there is somewhere better to be;
//   * the only fixture that reliably empties a cluster is this one, where
//     k exceeds the number of distinct points -- and there the farthest point
//     is at distance zero from an existing centre, so relocation moves the
//     centre (traced: it does fire) to another row that is just as good.
//
// So the branch is reachable, correct, and unobservable in its effect -- the
// same shape as K7's empty-child assertion. It is kept because it protects k as
// an invariant and prevents a 0/0 mean, and this comment is here so nobody
// later reads the assertions below as evidence that it was tested ([K70]).
$keLab = $ke.labels();
$keCnt = np.bincount(np.astype($keLab, "i64"), 4);
$toRows = cl._dist2($ke.cluster_centers(), $few);
$strayOk = true;
$strays  = 0;
$j = 0;
while ($j < 4) {
    if (np.get($keCnt, [$j]) == 0) {
        $strays = $strays + 1;
        if (sc(np.min(np.reshape(np.take($toRows,
                np.astype(nd([1.0 * $j], "f64"), "i64"), 0), [7]), null, false)) > 1.0e-18) {
            $strayOk = false;
        }
    }
    $j = $j + 1;
}
print("        clusters with no points: " + str($strays));
ok($strays > 0,
   "four clusters over three distinct points really does strand one, so the "
   + "next assertion is not vacuous");
ok($strayOk,
   "and every stranded centre sits exactly on a data row -- true whether it "
   + "was relocated or left there, which is why this is an invariant and not a "
   + "test of the relocation itself");
} catch ($e) { section_threw("an empty cluster is relocated", $e); }

print("");
print("== convergence is on CENTRE movement, and it converges ==");
try {
// Label churn can be zero while centres still move on a tie, so the stopping
// test is on the centres. On well-separated blobs it must stop early rather
// than running max_iter.
np.seed(5);
$kc = cl.KMeans({"n_clusters": 3, "n_init": 1, "max_iter": 300});
$kc.fit($X, null);
print("        iterations to converge: " + str($kc.n_iter()));
ok($kc.n_iter() < 300, "three separated blobs converge well before max_iter");
ok($kc.n_iter() >= 1, "and at least one iteration ran");
// A max_iter of 1 must stop after one, which is what says the cap is a cap.
$k1 = cl.KMeans({"n_clusters": 3, "n_init": 1, "max_iter": 1});
$k1.fit($X, null);
eq($k1.n_iter(), 1, "max_iter = 1 stops after exactly one iteration");
// A tol large enough to accept anything stops immediately.
$kt = cl.KMeans({"n_clusters": 3, "n_init": 1, "tol": 1000.0});
$kt.fit($X, null);
eq($kt.n_iter(), 1, "and an enormous tol stops after one too");
// "FEWER THAN max_iter" IS TRUE OF A LOOP THAT ALWAYS STOPS AFTER ONE, which is
// exactly what a broken convergence test does -- and `n_iter >= 1` cannot tell
// them apart. The real claim is that the result is a FIXED POINT: giving the
// same seeded run more iterations must not move the centres.
np.seed(5);
$fa = cl.KMeans({"n_clusters": 3, "n_init": 1});
$fa.fit($X, null);
np.seed(5);
$fb = cl.KMeans({"n_clusters": 3, "n_init": 1, "max_iter": 300});
$fb.fit($X, null);
ok(bits($fa.cluster_centers()) == bits($fb.cluster_centers()),
   "the converged centres are a FIXED POINT -- more iterations move nothing");
// And it takes TWO passes to get there, which is what catches a convergence
// test that always reports success. On separated blobs the first pass already
// finds the final assignment, so the second moves nothing -- but the loop
// cannot know that until it has run and measured a zero shift, and a broken
// test would stop at one without ever measuring.
//
// So the claim is about the ITERATION COUNT, not about the centres: a
// one-iteration fit lands on the same centres here (measured -- the first pass
// is already correct), and only n_iter distinguishes a loop that proved
// convergence from one that assumed it.
ok($fa.n_iter() > 1,
   "reaching it took " + str($fa.n_iter()) + " passes -- the second measures a "
   + "zero shift, which is how the loop learns it is done");
np.seed(5);
$fc = cl.KMeans({"n_clusters": 3, "n_init": 1, "max_iter": 1});
$fc.fit($X, null);
ok(bits($fc.cluster_centers()) == bits($fa.cluster_centers()),
   "and a one-pass fit reaches the same centres, which is exactly why the "
   + "count rather than the centres is the thing to assert here");
} catch ($e) { section_threw("convergence", $e); }

print("");
print("== what KMeans refuses ==");
try {
raises(def() { return cl.KMeans({"n_clusters": 0}); },  "n_clusters 0 is refused");
raises(def() { return cl.KMeans({"n_clusters": 2.5}); },"a fractional n_clusters is refused");
raises(def() { return cl.KMeans({"n_init": 0}); },      "n_init 0 is refused");
raises(def() { return cl.KMeans({"max_iter": 0}); },    "max_iter 0 is refused");
raises(def() { return cl.KMeans({"tol": 0.0 - 1.0}); }, "a negative tol is refused");
raises(def() { return cl.KMeans({"n_clustrs": 3}); },   "a mistyped key is refused (K24)");
raises(def() {
    $m = cl.KMeans({"n_clusters": 3});
    return $m.set_params({"n_clusters": 0});
}, "and set_params refuses the value too, not just the key (K68)");
raises(def() {
    $m = cl.KMeans({"n_clusters": 5});
    return $m.fit(nd([[1.0, 2.0], [3.0, 4.0]], "f64"), null);
}, "more clusters than rows is refused");
raises(def() {
    $m = cl.KMeans({"n_clusters": 2});
    return $m.fit(np.reshape($X, [360]), null);
}, "a 1-d X is refused");
raises(def() { return cl.KMeans({"n_clusters": 3}).predict($X); },
       "predict before fit is refused");
raises(def() {
    $m = cl.KMeans({"n_clusters": 3});
    $m.fit($X, null);
    return $m.predict(np.random([4, 7], 0.0, 1.0));
}, "predict with the wrong feature count is refused");
} catch ($e) { section_threw("what KMeans refuses", $e); }

print("");
print("== clone is independent ==");
try {
$orig = cl.KMeans({"n_clusters": 3, "n_init": 2});
$orig.fit($X, null);
$cl2 = $orig.clone();
ok(!$cl2.fitted(), "a clone is unfitted");
eq($cl2.get_params().n_clusters, 3, "and carries the parameters");
$cl2.set_params({"n_clusters": 5});
eq($orig.get_params().n_clusters, 3,
   "changing the clone does not change the original");
$orig.set_params({"n_init": 4});
ok(!$orig.fitted(), "set_params invalidates the fit");
} catch ($e) { section_threw("clone is independent", $e); }

print("");
print("== tier 4: a real-sized clustering ==");
try {
np.seed(23);
$cen = np.multiply(np.random([8, 6], 0.0, 1.0), 10.0, null);
$parts = [];
$i = 0;
while ($i < 8) {
    append($parts, np.add(np.random([500, 6], 0.0 - 0.5, 0.5),
                          np.reshape(np.take($cen,
                              np.astype(nd([1.0 * $i], "f64"), "i64"), 0), [6]),
                          null));
    $i = $i + 1;
}
$big = np.concat($parts, 0);
$t0 = clock();
$kb = cl.KMeans({"n_clusters": 8, "n_init": 3});
$kb.fit($big, null);
print("        4000x6 into 8 clusters, 3 restarts: " + str(clock() - $t0)
      + " ms, " + str($kb.n_iter()) + " iterations");
eq(np.size(np.unique($kb.labels())), 8, "all eight clusters are populated");
$pairB = cl._dist2($cen, $kb.cluster_centers());
ok(sc(np.max(np.min($pairB, 1, false), null, false)) < 0.05,
   "and every planted centre is recovered");
} catch ($e) { section_threw("tier 4", $e); }

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
