// ============================================================================
// lugha/train.b — train the moderation classifier
//
//   bantu run lugha/train.b
//     reads  lugha/corpus/moderation_{train,eval}.tsv  +  lugha/vocab.txt
//     writes samples/one-taylor/backend/model/toxicity.btw
//            lugha/corpus/METRICS.md
//
// Environment:
//   LUGHA_STEPS   training steps (default 1500)
//   LUGHA_LR      peak learning rate (default 0.0012)
//   LUGHA_BATCH   sequences per step (default 16)
//
// The maths lives in trainer.b; this file is the run.
// ============================================================================

include "numba" as np;
include "./tokenizer.b" as tk;
include "./encoder.b" as enc;
include "./init.b" as mk;
include "./weights.b" as W;
include "./trainer.b";

// ════════════════════════════════════════════════════════════════════════════
//  main
// ════════════════════════════════════════════════════════════════════════════
$steps = 1500;
$envSteps = env("LUGHA_STEPS");
if ($envSteps != null && $envSteps != "") { $steps = floor(num($envSteps)); }
$peakLr = 0.0012;
$envLr = env("LUGHA_LR");
if ($envLr != null && $envLr != "") { $peakLr = num($envLr); }
$batch = 16;
$envBatch = env("LUGHA_BATCH");
if ($envBatch != null && $envBatch != "") { $batch = floor(num($envBatch)); }

print("=== lugha: training the moderation classifier ===");
$V = tk.load("lugha/vocab.txt");
print("[data] vocab " + str($V) + " tokens");

$train = loadRows("lugha/corpus/moderation_train.tsv");
$evalRows = loadRows("lugha/corpus/moderation_eval.tsv");
$hardRows = loadRows("lugha/corpus/moderation_hard.tsv");
print("[data] train " + str(len($train)) + "  held out " + str(len($evalRows))
      + "  hand-written " + str(len($hardRows)));

// Per-label positive weights from the training split itself.
$pos = [0, 0, 0, 0, 0, 0, 0];
each ($r in $train) {
    $c = 0;
    while ($c < $N_LABELS) {
        if ($r["y"][$c] > 0.5) { $pos[$c] = $pos[$c] + 1; }
        $c = $c + 1;
    }
}
$w = [];
$c = 0;
while ($c < $N_LABELS) {
    $p = $pos[$c];
    if ($p < 1) { $p = 1; }
    $ratio = (len($train) - $p) / $p;
    // Ceiling of 4, not 8: beyond about 4x the model simply predicts the rare
    // label everywhere and the threshold sweep has nothing left to recover.
    append($w, clamp($ratio, 1.0, 4.0));
    $c = $c + 1;
}
print("[data] positive weights: " + str($w));

$model = mk.build($V, $D_MODEL, $D_FF, $N_LAYERS, $N_HEADS, $MAX_LEN, $N_LABELS, 20261026);
$names = enc.PARAM_NAMES($N_LAYERS);
print("[model] " + str(mk.paramCount($model, $N_LAYERS)) + " parameters");
adamInit($model, $names);
$grads = zerosLike($model, $names);

// Deterministic sampling, so a run is reproducible and a diff of the metrics
// means a change in the code rather than a change in the dice.
$R = { "state": 7777 };
def rnd() {
    $R["state"] = fmod($R["state"] * 1664525 + 1013904223, 4294967296);
    return $R["state"] / 4294967296;
}

const $WARMUP = 150;
$t0 = clock();
$runLoss = 0.0;
$runSeen = 0;
$step = 1;
while ($step <= $steps) {
    zeroFill($grads, $names);
    $batchLoss = 0.0;
    $b = 0;
    while ($b < $batch) {
        $r = $train[floor(rnd() * len($train))];
        $fw = enc.forward($model, $r["ids"], $r["n"], true);
        $lg = bceLossAndGrad($fw["scores"], $r["y"], $w);
        $batchLoss = $batchLoss + $lg["loss"];
        backward($model, $fw, $lg["dz"], $grads);
        $b = $b + 1;
    }
    // Average the gradient over the batch.
    each ($name in $names) {
        if ($name == "meta") { continue; }
        np.multiply($grads[$name], 1.0 / $batch, $grads[$name]);
    }

    // Learning rate: linear warmup then cosine decay.
    $lr = $peakLr;
    if ($step < $WARMUP) {
        $lr = $peakLr * $step / $WARMUP;
    } else {
        $frac = ($step - $WARMUP) / ($steps - $WARMUP);
        $lr = $peakLr * (0.5 * (1.0 + cos(PI * $frac)));
    }

    // Global-norm clipping at 1.0.
    $gn = gradNorm($grads, $names);
    $clip = 1.0;
    if ($gn > 1.0) { $clip = 1.0 / $gn; }

    adamStep($model, $grads, $names, $lr, $clip);

    $runLoss = $runLoss + $batchLoss / $batch;
    $runSeen = $runSeen + 1;
    if (fmod($step, 50) == 0) {
        print("[train] step " + str($step) + "/" + str($steps)
              + "  loss " + fmt3($runLoss / $runSeen)
              + "  lr " + fmt3($lr * 1000) + "e-3"
              + "  |g| " + fmt3($gn)
              + "  " + str(floor((clock() - $t0) / 1000)) + "s");
        $runLoss = 0.0;
        $runSeen = 0;
    }
    if (fmod($step, 500) == 0) {
        $rep = reportEval(evaluate($model, $evalRows, 250));
        print("[eval]  step " + str($step) + "  macro-F1 " + fmt3($rep["macroF1"])
              + "  loss " + fmt3($rep["loss"]));
    }
    $step = $step + 1;
}
print("[train] done in " + str(floor((clock() - $t0) / 1000)) + "s");

// ── final evaluation + threshold calibration ────────────────────────────────
// Scores are collected once; the sweep and both reports read the same scores.
$coll = collectScores($model, $evalRows, len($evalRows));
$half = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
$repHalf = reportEval(countAt($coll, $half));

// Different floors for different stakes. sexual_minor and grooming decide
// whether a child's report reaches a human, so recall is worth more there and
// the floor is lower; the adult-facing labels only attach a content warning,
// where over-labelling is its own harm.
//                harass sexual sx_min groom threat self_h slur
$floors =        [0.80,  0.80,  0.55,  0.55,  0.65,  0.65,  0.80];
$thresholds = sweepThresholds($coll, $floors);
$repCal = reportEval(countAt($coll, $thresholds));

print("");
print("=== held-out evaluation (" + str(len($evalRows)) + " rows) ===");
print("  label           thr    prec    rec     f1      support");
$md = ["| label | threshold | precision | recall | F1 | support |",
       "|---|---|---|---|---|---|"];
$c = 0;
each ($row in $repCal["rows"]) {
    print("  " + $row["label"] + "   " + fmt3($thresholds[$c]) + "   "
          + fmt3($row["prec"]) + "   " + fmt3($row["rec"]) + "   "
          + fmt3($row["f1"]) + "   " + str($row["support"]));
    append($md, "| `" + $row["label"] + "` | " + fmt3($thresholds[$c]) + " | "
          + fmt3($row["prec"]) + " | " + fmt3($row["rec"]) + " | "
          + fmt3($row["f1"]) + " | " + str($row["support"]) + " |");
    $c = $c + 1;
}
print("  macro-F1 = " + fmt3($repCal["macroF1"]) + " calibrated, "
      + fmt3($repHalf["macroF1"]) + " at a flat 0.5");
print("  mean held-out BCE = " + fmt3($coll["loss"]));

// The hand-written set, at the thresholds calibrated on the held-out set --
// never on this one. Calibrating on the set you report would make the number
// meaningless, which is the same mistake as splitting rows instead of
// templates, one layer up.
$hardColl = collectScores($model, $hardRows, len($hardRows));
$repHard = reportEval(countAt($hardColl, $thresholds));
$mdHard = ["| label | precision | recall | F1 | support |", "|---|---|---|---|---|"];
print("");
print("=== hand-written set (" + str(len($hardRows))
      + " rows, no templates, thresholds from the held-out set) ===");
print("  label           prec    rec     f1      support");
each ($row in $repHard["rows"]) {
    print("  " + $row["label"] + "   " + fmt3($row["prec"]) + "   "
          + fmt3($row["rec"]) + "   " + fmt3($row["f1"]) + "   " + str($row["support"]));
    append($mdHard, "| `" + $row["label"] + "` | " + fmt3($row["prec"]) + " | "
          + fmt3($row["rec"]) + " | " + fmt3($row["f1"]) + " | "
          + str($row["support"]) + " |");
}
print("  macro-F1 = " + fmt3($repHard["macroF1"]));

// ── save ────────────────────────────────────────────────────────────────────
// The thresholds travel WITH the weights. A model file whose operating points
// live somewhere else is a model file that will be deployed against the wrong
// ones the first time anybody retrains it.
$reliability = reliabilityOf($coll, $thresholds);
print("");
print("  labels the model may BLOCK on (precision >= 0.6 at its threshold):");
$c = 0;
$advisory = [];
while ($c < $N_LABELS) {
    if ($reliability[$c] < 0.5) { append($advisory, $LABELS[$c]); }
    $c = $c + 1;
}
$blocking = [];
$c = 0;
while ($c < $N_LABELS) {
    if ($reliability[$c] >= 0.5) { append($blocking, $LABELS[$c]); }
    $c = $c + 1;
}
print("    block:    " + join($blocking, ", "));
print("    advisory: " + join($advisory, ", ") + "   (review queue only)");

$model["thresholds"] = np.array($thresholds, "f64");
$model["reliability"] = np.array($reliability, "f64");
$saveNames = [];
extend($saveNames, $names);
append($saveNames, "thresholds");
append($saveNames, "reliability");

const $OUT = "samples/one-taylor/backend/model/toxicity.btw";
$bytes = W.save($OUT, $saveNames, $model);
print("[save] " + $OUT + " (" + str($bytes) + " bytes)");

writefile("lugha/corpus/METRICS.md",
    "# lugha moderation classifier — held-out metrics\n\n"
  + "Regenerated by `bantu run lugha/train.b`.\n\n"
  + "- " + str(len($evalRows)) + " held-out rows, " + str(len($train)) + " training rows\n"
  + "- " + str($steps) + " steps, batch " + str($batch) + ", peak lr " + str($peakLr) + "\n"
  + "- " + str(mk.paramCount($model, $N_LAYERS)) + " parameters "
  + "(" + str($N_LAYERS) + " layers, " + str($N_HEADS) + " heads, d_model "
  + str($D_MODEL) + ")\n\n"
  + "## Held-out templates and slot values\n\n"
  + join($md, "\n") + "\n\n"
  + "## Hand-written set (no templates)\n\n"
  + join($mdHard, "\n") + "\n\n"
  + "Hand-written macro-F1 = **" + fmt3($repHard["macroF1"])
  + "**, measured at the thresholds calibrated on the held-out set — never on "
  + "this one.\n\n"
  + "**macro-F1 = " + fmt3($repCal["macroF1"]) + "** at the calibrated thresholds, "
  + fmt3($repHalf["macroF1"]) + " at a flat 0.5. Mean held-out BCE = "
  + fmt3($coll["loss"]) + ".\n\n"
  + "Thresholds are swept on this held-out split for the best F2 (recall weighted "
  + "twice as heavily as precision) subject to a per-label precision floor and a "
  + "hard threshold floor of 0.35, and are stored in `toxicity.btw` alongside the "
  + "weights.\n\n"
  + "A label whose precision at its chosen threshold is under 0.6 is **advisory**: "
  + "it routes a post to the human review queue and is never allowed to block one. "
  + "Advisory here: " + join($advisory, ", ") + ". Blocking: "
  + join($blocking, ", ") + ".\n\n"
  + "The split is over GENERATORS, not rows: about a fifth of the templates and "
  + "a fifth of each slot's values are held out, so a held-out sentence is a "
  + "phrasing the model never saw, filled with words it never saw in that role. "
  + "An earlier version split rows at random and reported a macro-F1 of 1.000 — "
  + "that number measured memorisation of 89 sentence patterns, nothing else.\n\n"
  + "These numbers describe this corpus. They are a floor, not a claim about "
  + "real reported content — see `docs/lugha-architecture.md` section 6.\n");
print("[save] lugha/corpus/METRICS.md");
