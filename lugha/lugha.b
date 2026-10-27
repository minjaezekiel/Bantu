// ============================================================================
// lugha — Bantu's NLP package. Public surface.
//
//   include "lugha" as lugha;
//   lugha.init("model/toxicity.btw", "model/vocab.txt");
//   $r = lugha.score("you are such an idiot");
//   if ($r["flagged"]) { ... }
//
// Design: docs/lugha-architecture.md
//
// Exposes:
//   init($modelPath, $vocabPath) -> {"params", "vocab", "labels"}
//   ready()                      -> bool
//   labels()                     -> list of label names
//   thresholds()                 -> dict label -> operating point
//   reliable()                   -> dict label -> may this label BLOCK?
//   score($text)                 -> {"scores", "flags", "flagged", "top", "normalised"}
//   normalise($text)             -> the canonical form the model sees
//   tokens($text)                -> the pieces, for debugging a score
// ============================================================================

include "numba" as np;
include "./normalise.b" as nz;
include "./tokenizer.b" as tk;
include "./encoder.b" as enc;
include "./weights.b" as W;

const $MAX_LEN = 64;
const $STRIDE  = 48;

$LABELS = ["harassment", "sexual", "sexual_minor", "grooming", "threat",
           "self_harm", "slur"];

// A dict, so the functions below can fill it in place. A list assigned to a
// global name inside a function would write a local and silently vanish.
$S = { "model": null, "thr": null, "rel": null, "ready": false };

def init($modelPath, $vocabPath) {
    $vocab = tk.load($vocabPath);
    $m = W.load($modelPath);
    if ($m["thresholds"] == null) {
        throw "lugha.init: '" + $modelPath + "' carries no thresholds tensor; "
            + "it was written by a build older than threshold calibration";
    }
    $thr = {};
    $t = np.to_list($m["thresholds"]);
    $i = 0;
    while ($i < len($LABELS)) {
        $thr[$LABELS[$i]] = $t[$i];
        $i = $i + 1;
    }
    // Reliability is per label: can the model act on this label, or only
    // mention it? A label whose measured precision is poor routes a post to a
    // human and is never allowed to block one. A model file written before
    // this existed is refused rather than defaulted to "yes", because
    // defaulting to "yes" is how a 15-year-old gets blocked for grooming by a
    // classifier with 0.44 precision.
    if ($m["reliability"] == null) {
        throw "lugha.init: '" + $modelPath + "' carries no reliability tensor; "
            + "recalibrate it with: bantu run lugha/calibrate.b";
    }
    $rel = {};
    $rl = np.to_list($m["reliability"]);
    $i = 0;
    while ($i < len($LABELS)) {
        $rel[$LABELS[$i]] = $rl[$i] >= 0.5;
        $i = $i + 1;
    }

    $cfg = enc.config($m);
    if ($cfg["V"] != $vocab) {
        throw "lugha.init: the model expects a " + str($cfg["V"])
            + "-token vocabulary but '" + $vocabPath + "' has " + str($vocab)
            + ". Model and vocab must come from the same training run.";
    }
    $S["model"] = $m;
    $S["thr"] = $thr;
    $S["rel"] = $rel;
    $S["ready"] = true;
    $blocking = [];
    each ($l in $LABELS) { if ($rel[$l]) { append($blocking, $l); } }
    return { "vocab": $vocab, "labels": $LABELS, "layers": $cfg["L"],
             "d_model": $cfg["D"], "blocking": $blocking };
}

def ready() { return $S["ready"]; }
def labels() { return $LABELS; }
def thresholds() { return $S["thr"]; }
def reliable() { return $S["rel"]; }
def normalise($text) { return nz.normalise($text); }
def tokens($text) { return tk.tokenize(nz.normalise($text)); }

// ── score ───────────────────────────────────────────────────────────────────
// A text longer than the window is scored in overlapping windows and the
// per-label MAXIMUM is taken. Truncating instead would let a paragraph of
// pleasantries followed by one threat score as pleasantries, which is how a
// moderation classifier gets evaded in practice.
def score($text) {
    if (!$S["ready"]) { throw "lugha.score: call lugha.init(model, vocab) first"; }
    $norm = nz.normalise($text);
    $scores = {};
    each ($l in $LABELS) { $scores[$l] = 0.0; }

    if ($norm == "") {
        return { "scores": $scores, "flags": {}, "flagged": false,
                 "top": null, "topScore": 0.0, "normalised": "", "windows": 0 };
    }

    $wins = tk.windows($norm, $MAX_LEN, $STRIDE);
    each ($win in $wins) {
        $p = np.to_list(np.reshape(
                enc.forward($S["model"], $win["ids"], $win["n"], false)["scores"], [-1]));
        $i = 0;
        while ($i < len($LABELS)) {
            if ($p[$i] > $scores[$LABELS[$i]]) { $scores[$LABELS[$i]] = $p[$i]; }
            $i = $i + 1;
        }
    }

    // A flag is a score at or over that label's own calibrated operating
    // point, not over a shared 0.5 -- the points differ by label because the
    // stakes do. See lugha/corpus/METRICS.md.
    $flags = {};
    $flagged = false;
    $top = null;
    $topScore = 0.0;
    each ($l in $LABELS) {
        $over = $scores[$l] >= $S["thr"][$l];
        $flags[$l] = $over;
        if ($over) { $flagged = true; }
        // "top" is ranked by how far over the line a label is, not by raw
        // score: 0.6 against a 0.4 threshold is a stronger signal than 0.7
        // against 0.75, and a raw ranking would report the wrong one.
        $margin = $scores[$l] - $S["thr"][$l];
        if ($top == null || $margin > $topScore) { $top = $l; $topScore = $margin; }
    }
    // Split the flags by what the caller is entitled to do with them. A
    // moderation pipeline needs to know the difference between "the model is
    // confident and measurably right about this" and "the model mentioned it".
    $blocking = {};
    $advisory = {};
    each ($l in $LABELS) {
        if (!$flags[$l]) { continue; }
        if ($S["rel"][$l]) { $blocking[$l] = $scores[$l]; }
        else { $advisory[$l] = $scores[$l]; }
    }
    return { "scores": $scores, "flags": $flags, "flagged": $flagged,
             "blocking": $blocking, "advisory": $advisory,
             "top": $top, "topScore": $topScore, "normalised": $norm,
             "windows": len($wins) };
}

// How close a text came to its nearest threshold without crossing it. The
// moderation queue uses this to route borderline posts to a human instead of
// deciding on its own.
def margin($result) {
    $worst = 0.0 - 1.0;
    each ($l in $LABELS) {
        $m = $result["scores"][$l] - $S["thr"][$l];
        if ($m > $worst) { $worst = $m; }
    }
    return $worst;
}
