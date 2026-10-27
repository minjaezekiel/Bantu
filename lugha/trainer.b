// ============================================================================
// lugha/trainer.b — loss, backward pass, Adam, data loading, evaluation
//
// A module rather than part of train.b, so tests/lugha_train_test.b can import
// the backward pass and check it against finite differences WITHOUT running a
// training session. The script that actually trains is train.b.
//
// numba has every operation the backward pass needs, because a transformer's
// gradients are the same matmuls transposed.
//
// Correctness is not assumed: a transformer with one sign error in its
// backward pass still trains -- to something worse -- and nothing but the
// gradient test notices.
// ============================================================================

include "numba" as np;
include "./normalise.b" as nz;
include "./tokenizer.b" as tk;
include "./encoder.b" as enc;
include "./init.b" as mk;
include "./weights.b" as W;

const $D_MODEL  = 128;
const $D_FF     = 512;
const $N_LAYERS = 2;
const $N_HEADS  = 4;
const $MAX_LEN  = 64;
const $N_LABELS = 7;
const $GELU_C   = 0.7978845608028654;

$LABELS = ["harassment", "sexual", "sexual_minor", "grooming", "threat",
           "self_harm", "slur"];

// ── gradient containers ─────────────────────────────────────────────────────
def zerosLike($m, $names) {
    $g = {};
    each ($name in $names) {
        if ($name == "meta") { continue; }
        $g[$name] = np.zeros(np.shape($m[$name]), "f64");
    }
    return $g;
}

def zeroFill($g, $names) {
    each ($name in $names) {
        if ($name == "meta") { continue; }
        np.multiply($g[$name], 0.0, $g[$name]);
    }
}

// ── LayerNorm backward ──────────────────────────────────────────────────────
// dX = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat)), the means
// being over the feature axis. Derived rather than copied; the test checks it.
def lnBackward($dY, $ln, $g) {
    $xhat = $ln["xhat"];
    $rstd = $ln["rstd"];
    $dg = np.sum(np.multiply($dY, $xhat, null), 0, false);
    $db = np.sum($dY, 0, false);
    $dxhat = np.multiply($dY, $g, null);
    $m1 = np.mean($dxhat, 1, true);
    $m2 = np.mean(np.multiply($dxhat, $xhat, null), 1, true);
    $inner = np.subtract(np.subtract($dxhat, $m1, null),
                         np.multiply($xhat, $m2, null), null);
    return { "dX": np.multiply($rstd, $inner, null), "dg": $dg, "db": $db };
}

// ── GELU backward ───────────────────────────────────────────────────────────
// y = 0.5x(1+t), t = tanh(c(x + ax^3))
// dy/dx = 0.5(1+t) + 0.5x(1-t^2)c(1 + 3a x^2)
def geluBackward($dY, $X, $G) {
    $t = $G["t"];
    $sech2 = np.subtract(1.0, np.multiply($t, $t, null), null);
    $poly = np.add(1.0, np.multiply(np.multiply($X, $X, null), 0.134145, null), null);
    $d = np.add(np.multiply(np.add($t, 1.0, null), 0.5, null),
                np.multiply(np.multiply(np.multiply($X, 0.5, null), $sech2, null),
                            np.multiply($poly, $GELU_C, null), null), null);
    return np.multiply($dY, $d, null);
}

// ── the backward pass ───────────────────────────────────────────────────────
// Accumulates into $grads. $dLogits is [1, C].
def backward($m, $fw, $dLogits, $grads) {
    $cfg = $fw["cfg"];
    $D = $cfg["D"];
    $dh = $cfg["dh"];
    $n = $fw["n"];
    $scale = 1.0 / sqrt($dh);

    // head: logits = pooled . Wh + bh
    np.add($grads["head.w"], np.matmul(np.transpose($fw["pooled"], null), $dLogits),
           $grads["head.w"]);
    np.add($grads["head.b"], np.reshape($dLogits, [-1]), $grads["head.b"]);
    $dPooled = np.matmul($dLogits, np.transpose($m["head.w"], null));       // [1, D]

    // pooled = mean over rows of Z  ->  every row gets dPooled / n
    $dZ = np.multiply(np.broadcast_to($dPooled, [$n, $D]), 1.0 / $n, null);

    // final LayerNorm
    $bf = lnBackward($dZ, $fw["lnf"], $m["ln_f.g"]);
    np.add($grads["ln_f.g"], $bf["dg"], $grads["ln_f.g"]);
    np.add($grads["ln_f.b"], $bf["db"], $grads["ln_f.b"]);
    $dX = $bf["dX"];

    $l = $cfg["L"] - 1;
    while ($l >= 0) {
        $p = "blk" + str($l) + ".";
        $bc = $fw["blocks"][$l];

        // X_{l+1} = Xmid + ff
        $dff = $dX;
        $dXmid = np.copy($dX);

        // ff = Gy . Ff2 + b2
        np.add($grads[$p + "ff2"], np.matmul(np.transpose($bc["G"]["Y"], null), $dff),
               $grads[$p + "ff2"]);
        np.add($grads[$p + "ff2b"], np.sum($dff, 0, false), $grads[$p + "ff2b"]);
        $dGy = np.matmul($dff, np.transpose($m[$p + "ff2"], null));          // [n, F]

        $dHpre = geluBackward($dGy, $bc["Hpre"], $bc["G"]);                  // [n, F]

        // Hpre = A2 . Ff1 + b1
        np.add($grads[$p + "ff1"], np.matmul(np.transpose($bc["A2"], null), $dHpre),
               $grads[$p + "ff1"]);
        np.add($grads[$p + "ff1b"], np.sum($dHpre, 0, false), $grads[$p + "ff1b"]);
        $dA2 = np.matmul($dHpre, np.transpose($m[$p + "ff1"], null));        // [n, D]

        $b2 = lnBackward($dA2, $bc["ln2"], $m[$p + "ln2.g"]);
        np.add($grads[$p + "ln2.g"], $b2["dg"], $grads[$p + "ln2.g"]);
        np.add($grads[$p + "ln2.b"], $b2["db"], $grads[$p + "ln2.b"]);
        np.add($dXmid, $b2["dX"], $dXmid);

        // Xmid = X_l + attn
        $dattn = $dXmid;
        $dXl = np.copy($dXmid);

        $dA1 = np.zeros([$n, $D], "f64");
        $h = 0;
        while ($h < $cfg["H"]) {
            $hd = $bc["heads"][$h];
            $WoH = enc.headRows($m[$p + "wo"], $h, $dh);

            // attn += ctx . Wo_h
            np.add(enc.headRows($grads[$p + "wo"], $h, $dh),
                   np.matmul(np.transpose($hd["ctx"], null), $dattn),
                   enc.headRows($grads[$p + "wo"], $h, $dh));
            $dctx = np.matmul($dattn, np.transpose($WoH, null));             // [n, dh]

            // ctx = P . V
            $dP = np.matmul($dctx, np.transpose($hd["V"], null));            // [n, n]
            $dV = np.matmul(np.transpose($hd["P"], null), $dctx);            // [n, dh]

            // softmax, row-wise: dS = P * (dP - rowsum(dP * P))
            $rowDot = np.sum(np.multiply($dP, $hd["P"], null), 1, true);
            $dS = np.multiply($hd["P"], np.subtract($dP, $rowDot, null), null);

            // S = (Q . K^T) * scale
            $dQK = np.multiply($dS, $scale, null);
            $dQ = np.matmul($dQK, $hd["K"]);                                  // [n, dh]
            $dK = np.matmul(np.transpose($dQK, null), $hd["Q"]);              // [n, dh]

            $A1 = $bc["A1"];
            $A1t = np.transpose($A1, null);
            np.add(enc.headCols($grads[$p + "wq"], $h, $dh), np.matmul($A1t, $dQ),
                   enc.headCols($grads[$p + "wq"], $h, $dh));
            np.add(enc.headCols($grads[$p + "wk"], $h, $dh), np.matmul($A1t, $dK),
                   enc.headCols($grads[$p + "wk"], $h, $dh));
            np.add(enc.headCols($grads[$p + "wv"], $h, $dh), np.matmul($A1t, $dV),
                   enc.headCols($grads[$p + "wv"], $h, $dh));

            np.add($dA1, np.matmul($dQ, np.transpose(enc.headCols($m[$p + "wq"], $h, $dh), null)), $dA1);
            np.add($dA1, np.matmul($dK, np.transpose(enc.headCols($m[$p + "wk"], $h, $dh), null)), $dA1);
            np.add($dA1, np.matmul($dV, np.transpose(enc.headCols($m[$p + "wv"], $h, $dh), null)), $dA1);
            $h = $h + 1;
        }

        $b1 = lnBackward($dA1, $bc["ln1"], $m[$p + "ln1.g"]);
        np.add($grads[$p + "ln1.g"], $b1["dg"], $grads[$p + "ln1.g"]);
        np.add($grads[$p + "ln1.b"], $b1["db"], $grads[$p + "ln1.b"]);
        np.add($dXl, $b1["dX"], $dXl);

        $dX = $dXl;
        $l = $l - 1;
    }

    // X0 = tok_embed[ids] + pos_embed[0:n]
    $dPos = np.slice($grads["pos_embed"], [[0, $n, null], null]);
    np.add($dPos, $dX, $dPos);

    $ids = $fw["ids"];
    $i = 0;
    while ($i < $n) {
        $row = floor($ids[$i]);
        $dst = np.slice($grads["tok_embed"], [[$row, $row + 1, null], null]);
        np.add($dst, np.slice($dX, [[$i, $i + 1, null], null]), $dst);
        $i = $i + 1;
    }
}

// ── loss ────────────────────────────────────────────────────────────────────
// Per-label binary cross-entropy with a positive weight, because the corpus is
// 46% negative overall and far more skewed per label (self_harm is ~6%).
// Unweighted BCE on that data produces a model that predicts "nothing wrong"
// and scores well doing it.
//   dL/dz_c = [ (1-y)p - w*y*(1-p) ] / C
def bceLossAndGrad($scores, $y, $w) {
    $p = np.to_list(np.reshape($scores, [-1]));
    $loss = 0.0;
    $dz = [];
    $c = 0;
    while ($c < $N_LABELS) {
        $pc = $p[$c];
        if ($pc < 0.0000001) { $pc = 0.0000001; }
        if ($pc > 0.9999999) { $pc = 0.9999999; }
        if ($y[$c] > 0.5) {
            $loss = $loss - $w[$c] * log($pc);
            append($dz, (0.0 - $w[$c] * (1.0 - $pc)) / $N_LABELS);
        } else {
            $loss = $loss - log(1.0 - $pc);
            append($dz, $pc / $N_LABELS);
        }
        $c = $c + 1;
    }
    return { "loss": $loss / $N_LABELS, "dz": np.array([$dz], "f64") };
}

// ── Adam ────────────────────────────────────────────────────────────────────
$OPT = { "m": null, "v": null, "t": 0 };

def adamInit($model, $names) {
    $OPT["m"] = zerosLike($model, $names);
    $OPT["v"] = zerosLike($model, $names);
    $OPT["t"] = 0;
}

def gradNorm($grads, $names) {
    $sq = 0.0;
    each ($name in $names) {
        if ($name == "meta") { continue; }
        $g = $grads[$name];
        $sq = $sq + np.get(np.sum(np.multiply($g, $g, null), null, false), []);
    }
    return sqrt($sq);
}

def adamStep($model, $grads, $names, $lr, $clipScale) {
    $OPT["t"] = $OPT["t"] + 1;
    $t = $OPT["t"];
    $b1 = 0.9;
    $b2 = 0.999;
    $c1 = 1.0 - pow($b1, $t);
    $c2 = 1.0 - pow($b2, $t);
    each ($name in $names) {
        if ($name == "meta") { continue; }
        $g = $grads[$name];
        if ($clipScale != 1.0) { np.multiply($g, $clipScale, $g); }
        $mm = $OPT["m"][$name];
        $vv = $OPT["v"][$name];
        // m = b1*m + (1-b1)*g ; v = b2*v + (1-b2)*g^2, both in place
        np.multiply($mm, $b1, $mm);
        np.add($mm, np.multiply($g, 1.0 - $b1, null), $mm);
        np.multiply($vv, $b2, $vv);
        np.add($vv, np.multiply(np.multiply($g, $g, null), 1.0 - $b2, null), $vv);
        $mhat = np.multiply($mm, 1.0 / $c1, null);
        $vhat = np.multiply($vv, 1.0 / $c2, null);
        $step = np.divide($mhat, np.add(np.sqrt($vhat, null), 0.00000001, null), null);
        np.subtract($model[$name], np.multiply($step, $lr, null), $model[$name]);
    }
}

// ── data ────────────────────────────────────────────────────────────────────
def loadRows($path) {
    $raw = readfile($path);
    $out = [];
    each ($line in split($raw, "\n")) {
        if (len($line) < 9) { continue; }
        $bits = substr($line, 0, 7);
        $text = substr($line, 8, len($line));
        $y = [];
        $c = 0;
        while ($c < $N_LABELS) {
            if (substr($bits, $c, 1) == "1") { append($y, 1.0); } else { append($y, 0.0); }
            $c = $c + 1;
        }
        $e = tk.encode($text, $MAX_LEN);
        append($out, { "ids": $e["ids"], "n": $e["n"], "y": $y, "text": $text });
    }
    return $out;
}

// ── evaluation ──────────────────────────────────────────────────────────────
// Score every row once, then measure at whatever thresholds we like. Running
// the model again per threshold would make calibration cost 19 passes over the
// held-out set instead of one.
def collectScores($model, $rows, $limit) {
    $P = [];
    $Y = [];
    $loss = 0.0;
    $seen = 0;
    $ones = [1, 1, 1, 1, 1, 1, 1];
    each ($r in $rows) {
        if ($seen >= $limit) { break; }
        $fw = enc.forward($model, $r["ids"], $r["n"], false);
        $loss = $loss + bceLossAndGrad($fw["scores"], $r["y"], $ones)["loss"];
        append($P, np.to_list(np.reshape($fw["scores"], [-1])));
        append($Y, $r["y"]);
        $seen = $seen + 1;
    }
    return { "P": $P, "Y": $Y, "loss": $loss / $seen, "n": $seen };
}

def countAt($coll, $thresholds) {
    $tp = [0, 0, 0, 0, 0, 0, 0];
    $fp = [0, 0, 0, 0, 0, 0, 0];
    $fn = [0, 0, 0, 0, 0, 0, 0];
    $P = $coll["P"];
    $Y = $coll["Y"];
    $i = 0;
    while ($i < len($P)) {
        $p = $P[$i];
        $y = $Y[$i];
        $c = 0;
        while ($c < $N_LABELS) {
            $pred = $p[$c] >= $thresholds[$c];
            $truth = $y[$c] > 0.5;
            if ($pred && $truth) { $tp[$c] = $tp[$c] + 1; }
            else if ($pred && !$truth) { $fp[$c] = $fp[$c] + 1; }
            else if (!$pred && $truth) { $fn[$c] = $fn[$c] + 1; }
            $c = $c + 1;
        }
        $i = $i + 1;
    }
    return { "tp": $tp, "fp": $fp, "fn": $fn, "loss": $coll["loss"], "n": $coll["n"] };
}

// Per-label threshold calibration.
//
// Training weights the positive class by up to 4x, which is deliberate -- a
// moderation classifier should miss less and over-flag more -- but it leaves
// every score shifted upward, so a fixed 0.5 reads as poor precision. The fix
// is not to un-weight the loss; it is to pick each label's operating point on
// held-out data, which is what any deployed classifier does.
//
// F-beta with beta = 2 weights recall twice as heavily as precision.
//
// The precision floor is a PREFERENCE, not a veto, and that distinction cost a
// measured 6 macro-F1 points to learn. Treated as a hard constraint it picks,
// for a label the model has not yet separated, the threshold where precision
// is 1.0 because almost nothing crosses it -- self_harm came back with
// precision 1.0 and recall 0.073, which is a classifier that has stopped
// working while reporting perfect scores. So: maximise F2 among the thresholds
// that meet the floor AND keep at least half the positives; if no threshold
// manages both, drop the floor and maximise F2 outright. Missing 90% of
// self-harm disclosures is not a trade worth any amount of precision.
const $MIN_RECALL = 0.5;

// A threshold floor, and the reason it exists.
//
// Without it the fallback path -- "no threshold met the precision floor, so
// maximise F2" -- chooses 0.05 for any label the model has not learned to
// separate. At 0.05 nearly everything crosses, F2 looks respectable because
// recall is 1.0, and the deployed effect is that a 15-year-old posting "I play
// football and write small games" is blocked for grooming. That is not a
// tuning mistake, it is a model with no discriminating power being given
// authority anyway.
//
// So two things, together:
//   - the sweep may not choose a threshold below $MIN_THRESHOLD;
//   - each label gets a RELIABILITY flag, from its precision at the chosen
//     threshold. A label the model cannot call precisely is advisory: it
//     routes a post to a human, and it is never allowed to block one.
const $MIN_THRESHOLD = 0.35;
const $MAX_THRESHOLD = 0.95;
const $RELIABLE_PRECISION = 0.6;

def _prf($coll, $c, $th) {
    $tp = 0;
    $fp = 0;
    $fn = 0;
    $i = 0;
    while ($i < len($coll["P"])) {
        $pred = $coll["P"][$i][$c] >= $th;
        $truth = $coll["Y"][$i][$c] > 0.5;
        if ($pred && $truth) { $tp = $tp + 1; }
        else if ($pred && !$truth) { $fp = $fp + 1; }
        else if (!$pred && $truth) { $fn = $fn + 1; }
        $i = $i + 1;
    }
    $prec = 0.0;
    if ($tp + $fp > 0) { $prec = $tp / ($tp + $fp); }
    $rec = 0.0;
    if ($tp + $fn > 0) { $rec = $tp / ($tp + $fn); }
    $f2 = 0.0;
    if ($prec + $rec > 0) { $f2 = 5 * $prec * $rec / (4 * $prec + $rec); }
    return { "prec": $prec, "rec": $rec, "f2": $f2 };
}

def sweepThresholds($coll, $floors) {
    $out = [];
    $c = 0;
    while ($c < $N_LABELS) {
        $floor = $floors[$c];
        $bestConstrained = 0.0 - 1.0;
        $thConstrained = 0.5;
        $bestFree = 0.0 - 1.0;
        $thFree = 0.5;
        $th = $MIN_THRESHOLD;
        while ($th <= $MAX_THRESHOLD + 0.0001) {
            $m = _prf($coll, $c, $th);
            if ($m["f2"] > $bestFree) { $bestFree = $m["f2"]; $thFree = $th; }
            if ($m["prec"] >= $floor && $m["rec"] >= $MIN_RECALL) {
                if ($m["f2"] > $bestConstrained) {
                    $bestConstrained = $m["f2"];
                    $thConstrained = $th;
                }
            }
            $th = $th + 0.05;
        }
        if ($bestConstrained >= 0.0) { append($out, $thConstrained); }
        else { append($out, $thFree); }
        $c = $c + 1;
    }
    return $out;
}

// Which labels the model is allowed to ACT on, as opposed to merely mention.
// 1.0 = precise enough at its operating point to block a post; 0.0 = advisory,
// so it reaches the review queue and nothing else.
def reliabilityOf($coll, $thresholds) {
    $out = [];
    $c = 0;
    while ($c < $N_LABELS) {
        $m = _prf($coll, $c, $thresholds[$c]);
        if ($m["prec"] >= $RELIABLE_PRECISION && $m["rec"] > 0.0) { append($out, 1.0); }
        else { append($out, 0.0); }
        $c = $c + 1;
    }
    return $out;
}

def evaluate($model, $rows, $limit) {
    $coll = collectScores($model, $rows, $limit);
    return countAt($coll, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
}

def reportEval($ev) {
    $lines = [];
    $c = 0;
    $f1sum = 0.0;
    while ($c < $N_LABELS) {
        $tp = $ev["tp"][$c];
        $fp = $ev["fp"][$c];
        $fn = $ev["fn"][$c];
        $prec = 0.0;
        if ($tp + $fp > 0) { $prec = $tp / ($tp + $fp); }
        $rec = 0.0;
        if ($tp + $fn > 0) { $rec = $tp / ($tp + $fn); }
        $f1 = 0.0;
        if ($prec + $rec > 0) { $f1 = 2 * $prec * $rec / ($prec + $rec); }
        $f1sum = $f1sum + $f1;
        append($lines, { "label": $LABELS[$c], "prec": $prec, "rec": $rec,
                         "f1": $f1, "support": $tp + $fn });
        $c = $c + 1;
    }
    return { "rows": $lines, "macroF1": $f1sum / $N_LABELS, "loss": $ev["loss"] };
}

def fmt3($x) {
    $r = floor($x * 1000 + 0.5) / 1000;
    return str($r);
}

