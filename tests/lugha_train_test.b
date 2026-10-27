// ════════════════════════════════════════════════════════════════════════
//  lugha_train_test.b — the backward pass, against finite differences.
//
//  A transformer with a sign error in one gradient still trains. It trains to
//  something worse, converges more slowly, and produces a model that looks
//  plausible and classifies badly — and no loss curve, no accuracy number and
//  no eyeball test reliably catches it. The only thing that does is comparing
//  every analytic gradient to a numerical one.
//
//  Central differences, (f(t+h) - f(t-h)) / 2h with h = 1e-5: one-sided
//  differences carry O(h) truncation error, central ones O(h^2), which is the
//  difference between a 1e-5 agreement and a 1e-9 one.
//
//  The model here is deliberately tiny (1 layer, 2 heads, d=8) so the whole
//  check runs in seconds, and every parameter group is sampled — including the
//  ones that are easy to get wrong: the per-head slices of wq/wk/wv/wo, the
//  LayerNorm scale and shift, and the embedding scatter.
//
//  Run:  bantu run tests/lugha_train_test.b
// ════════════════════════════════════════════════════════════════════════

include "numba" as np;
include "../lugha/encoder.b" as enc;
include "../lugha/init.b" as mk;
include "../lugha/trainer.b";

$R = {"pass": 0, "fail": 0};

def ok($cond, $name) {
    if ($cond) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $name); }
}

def close($got, $want, $tol, $name) {
    $d = $got - $want;
    if ($d < 0) { $d = 0 - $d; }
    if ($d <= $tol) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name + "  got " + str($got) + " want " + str($want)
              + " (diff " + str($d) + ", tol " + str($tol) + ")");
    }
}

// ── the fixture ─────────────────────────────────────────────────────────────
const $V = 12;
const $D = 8;
const $F = 16;
const $L = 1;
const $H = 2;
const $T = 6;
const $C = 7;

$model = mk.build($V, $D, $F, $L, $H, $T, $C, 1234);
$names = enc.PARAM_NAMES($L);

$ids = [2, 5, 7, 3];
$n = 4;
$y = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
// Non-trivial positive weights: a weighting bug cancels out when every weight
// is 1, which is exactly when a test would not notice it.
$w = [2.0, 1.0, 3.0, 1.0, 1.0, 4.0, 1.0];

def lossAt($m) {
    $fw = enc.forward($m, $ids, $n, false);
    return bceLossAndGrad($fw["scores"], $y, $w)["loss"];
}

print("-- forward sanity --");
$fw = enc.forward($model, $ids, $n, true);
$sc = np.to_list(np.reshape($fw["scores"], [-1]));
$allIn = true;
each ($s in $sc) { if ($s <= 0.0 || $s >= 1.0) { $allIn = false; } }
ok($allIn, "every sigmoid score is strictly inside (0, 1)");
ok(len($sc) == $C, "the head emits one score per label");
ok(np.shape($fw["pooled"])[1] == $D, "mean pooling gives a d_model vector");

// Softmax rows must sum to exactly 1, or attention silently rescales its
// output by however much padding the text happened to need.
$P0 = $fw["blocks"][0]["heads"][0]["P"];
$rowSums = np.to_list(np.sum($P0, 1, false));
$maxDev = 0.0;
each ($rs in $rowSums) {
    $d = $rs - 1.0;
    if ($d < 0) { $d = 0 - $d; }
    if ($d > $maxDev) { $maxDev = $d; }
}
close($maxDev, 0.0, 0.000000001, "attention rows sum to 1");

// LayerNorm must standardise before it scales. g starts at 1 and b at 0, so
// the normalised output should have zero mean and a variance of var/(var+eps).
//
// NOT exactly 1, and the difference is worth stating: the first LayerNorm sees
// the embedding sum, whose variance is about 2*0.02^2 = 8e-4, so an eps of
// 1e-5 is a bit over 1% of it and shaves that much off the normalisation.
// GPT-2 has the same property with the same constants, and it is harmless
// because ln1.g is learnable — but asserting "variance == 1" here would be
// asserting something false and then loosening the tolerance until it passed.
$lnY = $fw["blocks"][0]["ln1"]["Y"];
$lnX = np.add(np.take($model["tok_embed"], $ids, 0),
              np.slice($model["pos_embed"], [[0, $n, null], null]), null);
// Per row, then averaged: var/(var+eps) is concave, so the mean of the
// ratios is not the ratio of the means and the two differ in the third
// decimal place — which is exactly the size of the effect being measured.
$varRows = np.var($lnX, 1, false, 0);
$expectVar = np.get(np.mean(np.divide($varRows, np.add($varRows, 0.00001, null), null),
                            null, false), []);
close(np.get(np.mean($lnY, null, false), []), 0.0, 0.000000001, "layernorm output has zero mean");
close(np.get(np.mean(np.var($lnY, 1, false, 0), null, false), []), $expectVar, 0.002,
      "layernorm output variance is var/(var+eps)");
ok($expectVar > 0.95 && $expectVar < 1.0, "eps costs under 5% of the normalisation here");

print("-- GELU against known values --");
$gx = np.array([[-3.0, -1.0, 0.0, 1.0, 3.0]], "f64");
$gy = np.to_list(np.reshape(enc.gelu($gx)["Y"], [-1]));
close($gy[2], 0.0,        0.0000001, "gelu(0) = 0");
close($gy[3], 0.841192,   0.00001,   "gelu(1) = 0.8412");
close($gy[1], -0.158808,  0.00001,   "gelu(-1) = -0.1588");
close($gy[4], 2.9963627,  0.00001,   "gelu(3) = 2.9964");
ok($gy[0] > -0.01 && $gy[0] < 0.0, "gelu(-3) is a small negative number");

print("-- the per-head sum equals concat-then-project --");
// The claim docs/lugha-architecture.md 3.1 rests on. Build two heads of
// context and project both ways; the answers must agree to rounding.
$ctxA = np.randn([3, 4], 0.0, 1.0);
$ctxB = np.randn([3, 4], 0.0, 1.0);
$Wo = np.randn([8, 5], 0.0, 1.0);
$sumWay = np.add(np.matmul($ctxA, np.slice($Wo, [[0, 4, null], null])),
                 np.matmul($ctxB, np.slice($Wo, [[4, 8, null], null])), null);
// The concatenated form, built by hand: [ctxA | ctxB] as one [3, 8].
$cat = np.zeros([3, 8], "f64");
$lhs = np.slice($cat, [null, [0, 4, null]]);
np.add($lhs, $ctxA, $lhs);
$rhs = np.slice($cat, [null, [4, 8, null]]);
np.add($rhs, $ctxB, $rhs);
$catWay = np.matmul($cat, $Wo);
ok(np.allclose($sumWay, $catWay, 0.000000001, 0.000000001),
   "sum of per-head projections == concat then project");

// ── the gradient check ──────────────────────────────────────────────────────
print("-- analytic gradients vs central finite differences --");
$grads = zerosLike($model, $names);
$fw2 = enc.forward($model, $ids, $n, true);
$lg = bceLossAndGrad($fw2["scores"], $y, $w);
backward($model, $fw2, $lg["dz"], $grads);

const $HSTEP = 0.00001;

// A deterministic spread of indices per tensor, so the test is reproducible
// and still covers more than element [0,0] — which is the one element a
// transposition bug leaves correct.
def probeIndices($shape, $count) {
    $out = [];
    $k = 0;
    while ($k < $count) {
        $idx = [];
        $d = 0;
        while ($d < len($shape)) {
            // a different stride per axis, so the probes are not all diagonal
            append($idx, fmod($k * (3 + 2 * $d) + $d, $shape[$d]));
            $d = $d + 1;
        }
        append($out, $idx);
        $k = $k + 1;
    }
    return $out;
}

$checked = 0;
$worst = 0.0;
$worstName = "";
each ($name in $names) {
    if ($name == "meta") { continue; }
    $tensor = $model[$name];
    $shape = np.shape($tensor);
    $bad = 0;
    each ($idx in probeIndices($shape, 4)) {
        $analytic = np.get($grads[$name], $idx);
        $orig = np.get($tensor, $idx);

        np.set($tensor, $idx, $orig + $HSTEP);
        $lossPlus = lossAt($model);
        np.set($tensor, $idx, $orig - $HSTEP);
        $lossMinus = lossAt($model);
        np.set($tensor, $idx, $orig);

        $numeric = ($lossPlus - $lossMinus) / (2 * $HSTEP);
        $diff = $analytic - $numeric;
        if ($diff < 0) { $diff = 0 - $diff; }
        $scaleRef = $analytic;
        if ($scaleRef < 0) { $scaleRef = 0 - $scaleRef; }
        $nref = $numeric;
        if ($nref < 0) { $nref = 0 - $nref; }
        if ($nref > $scaleRef) { $scaleRef = $nref; }
        $tol = 0.0000001 + 0.0002 * $scaleRef;
        if ($diff > $tol) {
            $bad = $bad + 1;
            print("        at " + str($idx) + ": analytic " + str($analytic)
                  + " numeric " + str($numeric));
        }
        $rel = $diff;
        if ($scaleRef > 0.000001) { $rel = $diff / $scaleRef; }
        if ($rel > $worst) { $worst = $rel; $worstName = $name; }
        $checked = $checked + 1;
    }
    ok($bad == 0, "d(loss)/d(" + $name + ")");
}
print("  checked " + str($checked) + " elements; worst relative error "
      + str($worst) + " on " + $worstName);

// ── the embedding scatter ───────────────────────────────────────────────────
// A row of tok_embed that no token in the sequence referenced must have a zero
// gradient. If it does not, the scatter wrote to the wrong row — which trains
// the wrong words and is invisible in a loss curve.
print("-- embedding gradient touches only the rows it should --");
$touched = { "2": true, "5": true, "7": true, "3": true };
$leak = 0;
$r = 0;
while ($r < $V) {
    $rowSum = np.get(np.sum(np.abs(np.slice($grads["tok_embed"], [[$r, $r + 1, null], null]), null),
                            null, false), []);
    if ($touched[str($r)] == null) {
        if ($rowSum != 0.0) { $leak = $leak + 1; }
    }
    $r = $r + 1;
}
ok($leak == 0, "untouched embedding rows have exactly zero gradient");

$usedNonZero = true;
each ($rowKey in ["2", "5", "7", "3"]) {
    $rr = floor(num($rowKey));
    $s = np.get(np.sum(np.abs(np.slice($grads["tok_embed"], [[$rr, $rr + 1, null], null]), null),
                       null, false), []);
    if ($s == 0.0) { $usedNonZero = false; }
}
ok($usedNonZero, "every referenced embedding row has a non-zero gradient");

// Position rows beyond the sequence length must likewise be untouched.
$posLeak = 0;
$r = $n;
while ($r < $T) {
    $s = np.get(np.sum(np.abs(np.slice($grads["pos_embed"], [[$r, $r + 1, null], null]), null),
                       null, false), []);
    if ($s != 0.0) { $posLeak = $posLeak + 1; }
    $r = $r + 1;
}
ok($posLeak == 0, "positional rows past the sequence have zero gradient");

// ── loss behaviour ──────────────────────────────────────────────────────────
print("-- loss is a loss --");
$perfect = np.array([[0.999, 0.001, 0.999, 0.001, 0.001, 0.999, 0.001]], "f64");
$wrong   = np.array([[0.001, 0.999, 0.001, 0.999, 0.999, 0.001, 0.999]], "f64");
$lp = bceLossAndGrad($perfect, $y, $w)["loss"];
$lw = bceLossAndGrad($wrong, $y, $w)["loss"];
ok($lp < 0.01, "a correct prediction has near-zero loss");
ok($lw > $lp * 100, "a confidently wrong prediction costs far more");
ok($lp > 0.0, "loss is never negative");

print("");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); } else { print("  RESULT: FAILED"); }
