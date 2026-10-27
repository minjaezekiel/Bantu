// ============================================================================
// lugha/corpus/generate.b — build the labelled moderation corpus
//
//   bantu run lugha/corpus/generate.b
//     -> lugha/corpus/moderation_train.tsv
//     -> lugha/corpus/moderation_eval.tsv
//
// Each line is "<7 label bits>\t<text>". The text is stored ALREADY NORMALISED,
// which is the point: the model trains on exactly the surface form
// lugha.score() will hand it at inference. Any mismatch there is a silent
// accuracy loss that no test would catch.
//
// Generation is deterministic (a fixed LCG seed), so the corpus is a
// reproducible artefact and a diff of it means something.
//
// ── THE SPLIT, and why it is not a random row split ────────────────────────
//
// The first version of this file shuffled all the generated rows and took 15%
// as the held-out set. It reported a macro-F1 of 1.000, which is how the
// mistake was caught: train and eval were drawn from the SAME templates, so
// an eval row was a training row with different words in the slots. The model
// had learned 89 sentence patterns, and the evaluation asked it to recognise
// those 89 sentence patterns.
//
// The split is therefore over the GENERATORS, not the rows:
//
//   - about a fifth of the TEMPLATES are held out, stratified by primary
//     label so every label has both training and evaluation phrasings;
//   - about a fifth of each SLOT's values are held out too, so an evaluation
//     sentence contains words the model never trained on.
//
// A held-out row is then a phrasing the model has never seen, filled with
// vocabulary it has never seen in that role. That is the number worth
// reporting, and it is much lower than 1.000.
//
// moderation_hard.tsv is a third set: hand-written sentences that follow no
// template at all. It is small, it is not used for calibration, and it is the
// closest thing here to a real test.
// ============================================================================

include "./slots.b";
include "./templates.b";
include "../normalise.b" as nz;

const $PER_TEMPLATE = 150;
const $EVAL_FRACTION = 0.15;
const $AUGMENT_RATE = 0.28;

// ── deterministic PRNG ──────────────────────────────────────────────────────
// Numbers are float64, so a 64-bit LCG would lose its low bits. This is the
// 32-bit Numerical Recipes constant set, kept in exact float range.
$R = { "state": 20261026 };

def rnd() {
    $R["state"] = fmod($R["state"] * 1664525 + 1013904223, 4294967296);
    return $R["state"] / 4294967296;
}

def rndInt($n) { return floor(rnd() * $n); }

def pick($list) { return $list[rndInt(len($list))]; }

// ── the generator-level split ───────────────────────────────────────────────
// Slot values first: each list with enough entries reserves its tail for
// evaluation. A list too short to split is shared, and said so below.
$SLOT_TRAIN = {};
$SLOT_EVAL  = {};
$SHARED_SLOTS = [];

def _splitSlots() {
    for $name, $values in $SLOTS.items() {
        $n = len($values);
        if ($n < 5) {
            // Too few to divide without leaving one side empty. Shared, and
            // reported, rather than silently halving the vocabulary.
            $SLOT_TRAIN[$name] = $values;
            $SLOT_EVAL[$name] = $values;
            append($SHARED_SLOTS, $name);
            continue;
        }
        $cut = $n - floor($n / 5);
        $tr = [];
        $ev = [];
        $i = 0;
        while ($i < $n) {
            if ($i < $cut) { append($tr, $values[$i]); }
            else { append($ev, $values[$i]); }
            $i = $i + 1;
        }
        $SLOT_TRAIN[$name] = $tr;
        $SLOT_EVAL[$name] = $ev;
    }
}
_splitSlots();

// Templates: grouped by primary label, then every fifth one in each group is
// held out. Grouping first is what guarantees that "grooming" has evaluation
// phrasings at all -- a flat every-fifth over a label-ordered list can skip a
// whole small group.
def _primaryLabel($bits) {
    $c = 0;
    while ($c < 7) {
        if (substr($bits, $c, 1) == "1") { return str($c); }
        $c = $c + 1;
    }
    return "benign";
}

$TPL_TRAIN = [];
$TPL_EVAL = [];

def _splitTemplates() {
    $seen = {};
    each ($tpl in $TEMPLATES) {
        $g = _primaryLabel($tpl["l"]);
        $k = 0;
        if ($seen[$g] != null) { $k = $seen[$g]; }
        $seen[$g] = $k + 1;
        if (fmod($k, 5) == 4) { append($TPL_EVAL, $tpl); }
        else { append($TPL_TRAIN, $tpl); }
    }
}
_splitTemplates();

// ── template expansion ──────────────────────────────────────────────────────
// One slot at a time, left to right, so a template using {hobby} twice gets
// two different hobbies rather than the same one repeated. $pool selects which
// half of each slot's values may be drawn.
def expand($template, $pool) {
    $t = $template;
    $guard = 0;
    while ($guard < 32) {
        $open = indexOf($t, "{");
        if ($open < 0) { break; }
        $close = indexOf($t, "}");
        if ($close < $open) { break; }
        $name = substr($t, $open + 1, $close - $open - 1);
        $choices = $pool[$name];
        $value = "";
        if ($choices == null || len($choices) == 0) {
            print("[generate] WARNING: unknown or empty slot {" + $name + "}");
        } else {
            $value = pick($choices);
        }
        $t = substr($t, 0, $open) + $value + substr($t, $close + 1, len($t));
        $guard = $guard + 1;
    }
    return $t;
}

// ── surface augmentation ────────────────────────────────────────────────────
// Applied BEFORE normalisation, so what survives into the corpus is only the
// residue the normaliser cannot undo: "sht" from "sh*t", "fuuck" from
// "fuuuuck". That residue is what the model has to learn; the rest the
// normaliser already handles, and training on it would teach nothing.
$_LEETABLE = { "a": "@", "e": "3", "i": "1", "o": "0", "s": "$", "t": "7" };

def augment($s) {
    $words = split($s, " ");
    $n = len($words);
    if ($n == 0) { return $s; }
    $i = rndInt($n);
    $w = $words[$i];
    if (len($w) < 4) { return $s; }

    $mode = rndInt(4);
    $out = $w;
    if ($mode == 0) {
        // mask one interior letter
        $at = 1 + rndInt(len($w) - 2);
        $out = substr($w, 0, $at) + "*" + substr($w, $at + 1, len($w));
    } else if ($mode == 1) {
        // stretch one letter
        $at = rndInt(len($w));
        $ch = substr($w, $at, 1);
        $out = substr($w, 0, $at) + $ch + $ch + $ch + substr($w, $at + 1, len($w));
    } else if ($mode == 2) {
        // dot-separate the whole word
        $chars = [];
        $k = 0;
        while ($k < len($w)) { append($chars, substr($w, $k, 1)); $k = $k + 1; }
        $out = join($chars, ".");
    } else {
        // leetspeak one letter
        $at = rndInt(len($w));
        $ch = substr($w, $at, 1);
        $sub = $_LEETABLE[$ch];
        if ($sub == null) { return $s; }
        $out = substr($w, 0, $at) + $sub + substr($w, $at + 1, len($w));
    }
    $words[$i] = $out;
    return join($words, " ");
}

// ── build ───────────────────────────────────────────────────────────────────
def buildRows($templates, $pool, $perTemplate, $seen) {
    $rows = [];
    $dupes = 0;
    each ($tpl in $templates) {
        $k = 0;
        while ($k < $perTemplate) {
            $raw = expand($tpl["t"], $pool);
            if (rnd() < $AUGMENT_RATE) { $raw = augment($raw); }
            $text = nz.normalise($raw);
            $k = $k + 1;
            if ($text == "") { continue; }
            $key = $tpl["l"] + "\t" + $text;
            if ($seen[$key] != null) { $dupes = $dupes + 1; continue; }
            $seen[$key] = true;
            append($rows, $key);
        }
    }
    return { "rows": $rows, "dupes": $dupes };
}

print("[generate] " + str(len($TEMPLATES)) + " templates -> "
      + str(len($TPL_TRAIN)) + " train / " + str(len($TPL_EVAL)) + " held out");
if (len($SHARED_SLOTS) > 0) {
    print("[generate] slots too short to split (shared): "
          + join(sort($SHARED_SLOTS), ", "));
}

$seen = {};
$trainBuild = buildRows($TPL_TRAIN, $SLOT_TRAIN, $PER_TEMPLATE, $seen);
$evalBuild  = buildRows($TPL_EVAL,  $SLOT_EVAL,  $PER_TEMPLATE, $seen);
$trainRows = $trainBuild["rows"];
$evalRows  = $evalBuild["rows"];

print("[generate] train " + str(len($trainRows)) + " rows ("
      + str($trainBuild["dupes"]) + " duplicates dropped)");
print("[generate] eval  " + str(len($evalRows)) + " rows ("
      + str($evalBuild["dupes"]) + " duplicates dropped)");

// ── shuffle each set independently (Fisher-Yates) ───────────────────────────
def shuffled($rows) {
    $n = len($rows);
    $i = $n - 1;
    while ($i > 0) {
        $j = rndInt($i + 1);
        $tmp = $rows[$i];
        $rows[$i] = $rows[$j];
        $rows[$j] = $tmp;
        $i = $i - 1;
    }
    return $rows;
}

$trainRows = shuffled($trainRows);
$evalRows = shuffled($evalRows);

writefile("lugha/corpus/moderation_train.tsv", join($trainRows, "\n") + "\n");
writefile("lugha/corpus/moderation_eval.tsv", join($evalRows, "\n") + "\n");

// ── label balance, so a skew is visible rather than discovered later ───────
$names = ["harassment", "sexual", "sexual_minor", "grooming", "threat",
          "self_harm", "slur"];

def balance($rows) {
    $counts = [0, 0, 0, 0, 0, 0, 0];
    $clean = 0;
    each ($r in $rows) {
        $bits = substr($r, 0, 7);
        $any = false;
        $c = 0;
        while ($c < 7) {
            if (substr($bits, $c, 1) == "1") { $counts[$c] = $counts[$c] + 1; $any = true; }
            $c = $c + 1;
        }
        if (!$any) { $clean = $clean + 1; }
    }
    return { "counts": $counts, "clean": $clean };
}

$bt = balance($trainRows);
$be = balance($evalRows);
print("[generate] label balance (train / held out):");
$c = 0;
while ($c < 7) {
    print("    " + $names[$c] + ": " + str($bt["counts"][$c]) + " / "
          + str($be["counts"][$c]));
    $c = $c + 1;
}
print("    (no label): " + str($bt["clean"]) + " / " + str($be["clean"])
      + "   = " + str(floor(100 * $bt["clean"] / len($trainRows))) + "% / "
      + str(floor(100 * $be["clean"] / len($evalRows))) + "%");

// ── the hard set ────────────────────────────────────────────────────────────
// hard_seed.tsv is hand-written: no template, no slot list, deliberately
// awkward. It is normalised here so it reaches the model in the same form as
// everything else, and it is NEVER used for training or for calibrating a
// threshold -- only for reporting. A number measured on data that shaped the
// model is not a measurement.
$hardIn = readfile("lugha/corpus/hard_seed.tsv");
$hardOut = [];
$hardSeen = {};
each ($line in split($hardIn, "\n")) {
    if (len($line) < 9) { continue; }
    $bits = substr($line, 0, 7);
    $tab = indexOf($line, "\t");
    if ($tab < 0) { continue; }
    $text = nz.normalise(substr($line, $tab + 1, len($line)));
    if ($text == "") { continue; }
    $key = $bits + "\t" + $text;
    if ($hardSeen[$key] != null) { continue; }
    $hardSeen[$key] = true;
    append($hardOut, $key);
}
writefile("lugha/corpus/moderation_hard.tsv", join($hardOut, "\n") + "\n");
print("[generate] hard " + str(len($hardOut)) + " hand-written rows");

// Leakage check. A hand-written line that happens to match a generated one
// would quietly turn the hard set into part of the training set.
$genKeys = {};
each ($r in $trainRows) { $genKeys[$r] = true; }
$overlap = 0;
each ($r in $hardOut) { if ($genKeys[$r] != null) { $overlap = $overlap + 1; } }
if ($overlap > 0) {
    print("[generate] WARNING: " + str($overlap)
          + " hard rows are identical to training rows");
} else {
    print("[generate] no overlap between the hard set and training rows");
}
