// ============================================================================
// lugha/calibrate.b — re-derive thresholds and reliability, without retraining
//
//   bantu run lugha/calibrate.b
//
// Loads the saved model, scores the held-out and hand-written splits, sweeps
// each label's operating point again, recomputes which labels are precise
// enough to BLOCK on, and rewrites the .btw in place.
//
// This is separate from train.b because the two answer different questions and
// cost two orders of magnitude apart. "What are the right thresholds?" is a
// policy question that changes whenever the floors change, the corpus grows or
// a reviewer says the queue is too noisy -- and re-answering it took twelve
// minutes of gradient descent that could not change the answer, because the
// weights were not what was being reconsidered.
// ============================================================================

include "numba" as np;
include "./tokenizer.b" as tk;
include "./encoder.b" as enc;
include "./init.b" as mk;
include "./weights.b" as W;
include "./trainer.b";

const $MODEL = "samples/one-taylor/backend/model/toxicity.btw";
const $VOCAB = "lugha/vocab.txt";

print("=== lugha: recalibrating " + $MODEL + " ===");
$V = tk.load($VOCAB);
$model = W.load($MODEL);
$cfg = enc.config($model);
if ($cfg["V"] != $V) {
    throw "calibrate: the model expects a " + str($cfg["V"]) + "-token vocabulary but "
        + $VOCAB + " has " + str($V);
}
print("[model] " + str($cfg["L"]) + " layers, d_model " + str($cfg["D"])
      + ", vocab " + str($cfg["V"]));

$evalRows = loadRows("lugha/corpus/moderation_eval.tsv");
$hardRows = loadRows("lugha/corpus/moderation_hard.tsv");
print("[data] held out " + str(len($evalRows)) + ", hand-written " + str(len($hardRows)));

$coll = collectScores($model, $evalRows, len($evalRows));
//                harass sexual sx_min groom threat self_h slur
$floors =        [0.80,  0.80,  0.55,  0.55,  0.65,  0.65,  0.80];
$thresholds = sweepThresholds($coll, $floors);
$reliability = reliabilityOf($coll, $thresholds);
$rep = reportEval(countAt($coll, $thresholds));

print("");
print("  label           thr    prec    rec     f1     may block?");
$c = 0;
each ($row in $rep["rows"]) {
    $act = "advisory";
    if ($reliability[$c] >= 0.5) { $act = "BLOCK"; }
    print("  " + $row["label"] + "   " + fmt3($thresholds[$c]) + "   "
          + fmt3($row["prec"]) + "   " + fmt3($row["rec"]) + "   "
          + fmt3($row["f1"]) + "   " + $act);
    $c = $c + 1;
}
print("  macro-F1 = " + fmt3($rep["macroF1"]));

$hardColl = collectScores($model, $hardRows, len($hardRows));
$repHard = reportEval(countAt($hardColl, $thresholds));
print("  hand-written macro-F1 = " + fmt3($repHard["macroF1"]));

$model["thresholds"] = np.array($thresholds, "f64");
$model["reliability"] = np.array($reliability, "f64");
$saveNames = [];
extend($saveNames, enc.PARAM_NAMES($cfg["L"]));
append($saveNames, "thresholds");
append($saveNames, "reliability");
$bytes = W.save($MODEL, $saveNames, $model);
print("[save] " + $MODEL + " (" + str($bytes) + " bytes)");
