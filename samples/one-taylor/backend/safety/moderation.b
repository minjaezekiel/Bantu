// ============================================================================
// one-taylor/backend/safety/moderation.b — one decision, at write time
//
// normalise -> lexicon -> lugha (7 scores) -> decision table -> audit row
//
// Two kinds of evidence reach the decision table, and they are not
// interchangeable:
//
//   a LEXICON hit is an exact match on a term an operator chose to list. It is
//   certain, and it blocks.
//
//   a MODEL flag is a probability over a threshold. It blocks only for a label
//   whose measured precision earned that right (lugha's reliability tensor);
//   otherwise it is advisory and routes the post to a human.
//
// That distinction is not fussiness. Before it existed, the threshold sweep
// gave grooming an operating point of 0.05 at 0.44 precision, and a
// 15-year-old posting "I play football and write small games" was blocked for
// grooming. A classifier with no discriminating power on a label must not be
// given authority over it.
//
//   author band   outcome
//   ------------  ------------------------------------------------------------
//   minor         lexicon hit, or a BLOCKING model flag -> BLOCKED + queued
//   minor         an ADVISORY model flag                -> published + queued
//   minor         within MARGIN below a threshold       -> published + queued
//   adult         lexicon hit on sexual_minor | grooming | threat -> BLOCKED
//   adult         a BLOCKING model flag in that set              -> BLOCKED
//   adult         anything else -> published WITH A CONTENT LABEL
//
// Adults are free, as specified: what the classifier does for an adult is
// LABEL, so the opt-in gate has something to work with. Minors get the strict
// table, which is the entire reason the model exists.
//
// This never runs on a read. sua.sqlite and the handler share one thread, so a
// 15 ms model call on the feed path would be 15 ms added to every other
// client's latency on that worker. Scores are computed once, at write, and
// stored.
//
// Exposes:
//   init($modelPath, $vocabPath, $lexiconDir) -> {"ready", ...}
//   ready()
//   assess($text, $authorBand)  -> decision
//   record($subjectType, $subjectId, $decision, $actor)
// ============================================================================

include "lugha" as lugha;
include "orm" as orm;
include "../models.b" as models;
include "./lexicon.b" as lexicon;
include "./labels.b" as labels;

// How far below its threshold a score still counts as borderline. A borderline
// post by a minor is published and queued for a human, rather than decided by
// a number that was one rounding away from the other answer.
const $MARGIN = 0.15;

// For an adult author, only these three are refusals. The rest become labels.
$ADULT_BLOCKING = ["sexual_minor", "grooming", "threat"];

$S = { "ready": false, "reason": "not initialised" };

def init($modelPath, $vocabPath, $lexiconDir) {
    $lexCounts = lexicon.load($lexiconDir);
    try {
        $info = lugha.init($modelPath, $vocabPath);
        $S["ready"] = true;
        $S["reason"] = null;
        return { "ready": true, "model": $info, "lexicon": $lexCounts };
    } catch ($e) {
        // The lexicon still works, and it is the stage with perfect recall on
        // what it lists. Running degraded and saying so is better than
        // refusing to start -- but "degraded" must be visible, so it goes in
        // the health endpoint and the startup banner.
        $S["ready"] = false;
        $m = $e;
        if (type($e) == "dict" && $e.message != null) { $m = $e.message; }
        $S["reason"] = str($m);
        return { "ready": false, "error": str($m), "lexicon": $lexCounts };
    }
}

def ready() { return $S["ready"]; }
def reason() { return $S["reason"]; }
def allLabels() { return lugha.labels(); }

// Which labels may block, and which are advisory. Reported by /api/health so
// the policy currently in force is observable rather than inferred.
def blockingLabels() {
    if (!$S["ready"]) { return []; }
    $out = [];
    $rel = lugha.reliable();
    each ($l in lugha.labels()) { if ($rel[$l]) { append($out, $l); } }
    return $out;
}

def advisoryLabels() {
    if (!$S["ready"]) { return []; }
    $out = [];
    $rel = lugha.reliable();
    each ($l in lugha.labels()) { if (!$rel[$l]) { append($out, $l); } }
    return $out;
}

def _has($list, $v) {
    each ($x in $list) { if ($x == $v) { return true; } }
    return false;
}

// ── assess ──────────────────────────────────────────────────────────────────
// Returns:
//   verdict  "allow" | "label" | "watch" | "hold" | "block"
//
//     allow   published, nothing recorded beyond the audit row
//     label   published, carrying a content warning
//     watch   published, and queued for a human to look at
//     hold    saved but NOT visible to anyone except its author, pending review
//     block   never visible to anyone, pending appeal
//
//   labels   list of {category, confidence, source} for post_labels
//   reasons  human-readable, shown to the author
//   scores   the model's 7 scores (or {} when it is unavailable)
//   lexicon  categories the deterministic stage matched
//
// WHO IS ALLOWED TO BLOCK, and why it is only the lexicon
// ──────────────────────────────────────────────────────────────────────────
// The measured precision of this classifier at its operating points is 0.44
// to 0.92 depending on the label (lugha/corpus/METRICS.md). On the labels that
// matter most for a minor it is around two thirds. A policy of "the model
// blocks" therefore deletes roughly one innocent post in three — and it was
// not a thought experiment: the API test caught a 15-year-old's "Scored twice
// at practice today" being blocked outright.
//
// So the authority is split by the quality of the evidence:
//
//   a LEXICON hit BLOCKS. It is an exact match, on a word boundary, against a
//   term an operator deliberately listed. There is no probability in it.
//
//   a RELIABLE model flag HOLDS. The post is saved and the author can see it,
//   nobody else can, and a human decides. A hold is reversible in one click
//   and the author is told it is pending; a block is not something to do on
//   two-thirds confidence.
//
//   an ADVISORY model flag WATCHES. The post is published and queued. A minor
//   who is actually being groomed is better served by a reviewer reading the
//   thread than by their own message vanishing — and the person grooming them
//   learns nothing from a post that silently fails to appear.
//
// Adults keep the freedom the brief asked for: for an adult author the model
// produces content LABELS, and only the three categories in $ADULT_BLOCKING
// can hold or block at all.
def assess($text, $authorBand) {
    $norm = lugha.normalise($text);
    $lex = lexicon.scan($norm);

    $scores = {};
    $modelUsed = false;
    $r = null;
    if ($S["ready"]) {
        $r = lugha.score($text);
        $scores = $r["scores"];
        $modelUsed = true;
    }

    // Three buckets of evidence, by what each is entitled to do.
    $certain = $lex["categories"];                      // lexicon: may block
    $strong = [];                                       // reliable model flags
    $weak = [];                                         // advisory model flags
    if ($modelUsed && $r != null) {
        for $l, $v in $r["blocking"].items() { append($strong, $l); }
        for $l, $v in $r["advisory"].items() { append($weak, $l); }
    }

    $contentLabels = [];
    if ($modelUsed) {
        // Content WARNINGS come from every flag, reliable or not: a warning is
        // a courtesy to the reader, not a sanction on the author, so the cost
        // of a wrong one is a blurred image rather than a deleted post.
        each ($cl in labels.fromModelScores($scores, $r["flags"])) {
            append($contentLabels, { "category": $cl["category"],
                                     "confidence": $cl["confidence"],
                                     "source": "model" });
        }
    }
    each ($cat in $certain) {
        if ($cat == "sexual") {
            append($contentLabels, { "category": "suggestive", "confidence": 1.0,
                                     "source": "model" });
        }
    }

    def _result($verdict, $reasons, $withLabels) {
        $out = { "verdict": $verdict, "reasons": $reasons,
                 "scores": $scores, "lexicon": $certain,
                 "strong": $strong, "advisory": $weak,
                 "modelUsed": $modelUsed, "normalised": $norm, "labels": [] };
        if ($withLabels) { $out["labels"] = $contentLabels; }
        return $out;
    }

    if ($authorBand == "minor") {
        if (len($certain) > 0)  { return _result("block", $certain, false); }
        if (len($strong) > 0)   { return _result("hold", $strong, false); }
        if (len($weak) > 0)     { return _result("watch", $weak, false); }
        // Borderline: nothing crossed, but something came close enough that a
        // rounding decided it. A human decides instead.
        if ($modelUsed && $r != null) {
            $m = lugha.margin($r);
            if ($m > 0.0 - $MARGIN) {
                return _result("watch", ["borderline: " + $r["top"]], false);
            }
        }
        return _result("allow", [], false);
    }

    // ── adult author ────────────────────────────────────────────────────────
    $certainSerious = [];
    each ($f in $certain) { if (_has($ADULT_BLOCKING, $f)) { append($certainSerious, $f); } }
    if (len($certainSerious) > 0) { return _result("block", $certainSerious, false); }

    $strongSerious = [];
    each ($f in $strong) { if (_has($ADULT_BLOCKING, $f)) { append($strongSerious, $f); } }
    if (len($strongSerious) > 0) { return _result("hold", $strongSerious, false); }

    $weakSerious = [];
    each ($f in $weak) { if (_has($ADULT_BLOCKING, $f)) { append($weakSerious, $f); } }
    if (len($weakSerious) > 0) { return _result("watch", $weakSerious, true); }

    if (len($contentLabels) > 0) { return _result("label", [], true); }
    return _result("allow", [], false);
}

// ── audit ───────────────────────────────────────────────────────────────────
// Append-only, with the scores that produced the decision. A disputed takedown
// has to be reconstructable, and a threshold change has to be evaluable
// against what already happened.
def record($subjectType, $subjectId, $decision, $actor, $detail, $scores) {
    $enc = "";
    if ($scores != null && len(keys($scores)) > 0) {
        $parts = [];
        for $k, $v in $scores.items() {
            append($parts, $k + "=" + str(floor($v * 1000) / 1000));
        }
        $enc = join(sort($parts), " ");
    }
    orm.insert(models.M["moderation_events"], {
        "subject_type": $subjectType,
        "subject_id": $subjectId,
        "actor": $actor,
        "decision": $decision,
        "detail": $detail,
        "scores": $enc
    });
}
