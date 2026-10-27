// ============================================================================
// one-taylor/backend/safety/lexicon.b — the deterministic stage
//
// A lexicon and a classifier are good at opposite things. A lexicon has
// perfect recall on the exact strings it lists and no understanding of
// context; the classifier reads context and will always have a miss rate.
// Running both and taking the union means the unambiguous cases never depend
// on a probability.
//
// The word lists live in text files under safety/lexicons/, one term per line,
// so an operator can extend them without touching code and without a rebuild.
// Missing files are not an error: the directory ships with the obvious terms,
// and the slurs an operator cares about are theirs to add. Publishing an
// exhaustive list of ethnic and sexual-orientation slurs in a public
// repository is itself a harm, and it would be a worse one than the gap it
// closes, because the classifier and the report system both cover that ground.
//
// Matching runs on NORMALISED text (lugha's normaliser), so "f.u.c.k",
// "ƒuck" and "FUUUCK" are all matched by the single entry "fuck".
//
// Exposes:
//   load($dir)        -> {category: termCount}
//   scan($normText)   -> {"hits": {category: [terms]}, "categories": [...]}
// ============================================================================

$CATEGORIES = ["slur", "sexual", "sexual_minor", "grooming", "threat", "self_harm"];

$LEX = { "terms": {}, "loaded": false, "counts": {} };

// Shipped defaults. Deliberately small and unambiguous: a term here fires with
// no appeal, so anything context-dependent belongs to the classifier instead.
$DEFAULTS = {
    "slur": ["retard", "tranny", "faggot", "fag"],
    "sexual": ["send nudes", "dick pic", "sext me", "nudes"],
    "sexual_minor": ["cp", "jailbait", "loli", "underage nudes",
                     "schoolgirl nudes", "child porn"],
    "grooming": ["dont tell your parents", "do not tell your parents",
                 "this is our secret", "keep this between us",
                 "delete these messages", "are you home alone",
                 "mature for your age", "dont tell anyone we talked"],
    "threat": ["i know where you live", "i will kill you", "ill kill you",
               "i will find you", "watch your back"],
    "self_harm": ["kill myself", "want to die", "end my life", "hurt myself",
                  "better off dead"]
};

def _readTerms($path) {
    $out = [];
    try {
        $raw = readfile($path);
        each ($line in split($raw, "\n")) {
            $t = trim($line).lower();
            if ($t == "" ) { continue; }
            if (substr($t, 0, 1) == "#") { continue; }
            append($out, $t);
        }
    } catch ($e) {
        // No file for this category: the defaults stand. Not an error.
        return $out;
    }
    return $out;
}

def load($dir) {
    $terms = {};
    $counts = {};
    each ($cat in $CATEGORIES) {
        $list = [];
        $d = $DEFAULTS[$cat];
        if ($d != null) { extend($list, $d); }
        extend($list, _readTerms($dir + "/" + $cat + ".txt"));
        // Dedupe, and index by first character so a scan does not test every
        // term against every position.
        $seen = {};
        $kept = [];
        each ($t in $list) {
            if ($seen[$t] == null) { $seen[$t] = true; append($kept, $t); }
        }
        $terms[$cat] = $kept;
        $counts[$cat] = len($kept);
    }
    $LEX["terms"] = $terms;
    $LEX["counts"] = $counts;
    $LEX["loaded"] = true;
    return $counts;
}

def counts() { return $LEX["counts"]; }

// A term matches on a WORD BOUNDARY, not as a bare substring. "fag" inside
// "fagott" is a bassoon, and "cp" inside "cpu" is a processor; a substring
// match would flag both, and a moderation system that flags musicians loses
// the authority it needs for the cases that matter.
def _containsWord($hay, $needle) {
    $hl = len($hay);
    $nl = len($needle);
    if ($nl == 0 || $nl > $hl) { return false; }
    $from = 0;
    while ($from <= $hl - $nl) {
        $at = indexOf(substr($hay, $from, $hl), $needle);
        if ($at < 0) { return false; }
        $abs = $from + $at;
        $beforeOk = true;
        if ($abs > 0) {
            $b = substr($hay, $abs - 1, 1);
            if (($b >= "a" && $b <= "z") || ($b >= "0" && $b <= "9")) { $beforeOk = false; }
        }
        $afterOk = true;
        $endAt = $abs + $nl;
        if ($endAt < $hl) {
            $a = substr($hay, $endAt, 1);
            if (($a >= "a" && $a <= "z") || ($a >= "0" && $a <= "9")) { $afterOk = false; }
        }
        if ($beforeOk && $afterOk) { return true; }
        $from = $abs + 1;
    }
    return false;
}

def scan($normText) {
    if (!$LEX["loaded"]) { throw "lexicon.scan: call lexicon.load(dir) first"; }
    $hits = {};
    $cats = [];
    if ($normText == null || $normText == "") {
        return { "hits": $hits, "categories": $cats };
    }
    each ($cat in $CATEGORIES) {
        $found = [];
        each ($term in $LEX["terms"][$cat]) {
            if (_containsWord($normText, $term)) { append($found, $term); }
        }
        if (len($found) > 0) {
            $hits[$cat] = $found;
            append($cats, $cat);
        }
    }
    return { "hits": $hits, "categories": $cats };
}
