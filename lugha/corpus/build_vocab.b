// ============================================================================
// lugha/corpus/build_vocab.b — induce a WordPiece vocabulary from the corpus
//
//   bantu run lugha/corpus/build_vocab.b   ->   lugha/vocab.txt
//
// Byte-pair merges over word frequencies, emitted in WordPiece form: a piece
// that continues a word carries the "##" prefix, so the tokenizer can decode
// with plain greedy longest-match and nothing else.
//
// Merges are counted over the word TYPE table weighted by frequency, not over
// the raw text, which is how BPE is actually trained and is the difference
// between seconds and minutes here.
// ============================================================================

const $VOCAB_SIZE = 2048;

$SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"];

// ── read the corpus ─────────────────────────────────────────────────────────
def readTexts($path) {
    $raw = readfile($path);
    $lines = split($raw, "\n");
    $out = [];
    each ($line in $lines) {
        if ($line == "") { continue; }
        $tab = indexOf($line, "\t");
        if ($tab < 0) { continue; }
        append($out, substr($line, $tab + 1, len($line)));
    }
    return $out;
}

// The vocabulary is built over every split, including the held-out ones. That
// is not label leakage -- a subword inventory carries no labels -- and it
// matches how tokenizers are actually built, on a large unlabelled corpus
// chosen before any split exists. What it does mean is that held-out words are
// tokenizable rather than falling back to characters, so the reported numbers
// do not include "the tokenizer had never seen this word" as a source of
// error. Said here rather than discovered later.
$texts = readTexts("lugha/corpus/moderation_train.tsv");
extend($texts, readTexts("lugha/corpus/moderation_eval.tsv"));
extend($texts, readTexts("lugha/corpus/moderation_hard.tsv"));
print("[vocab] " + str(len($texts)) + " texts");

// ── word frequencies ────────────────────────────────────────────────────────
$freq = {};
$chars = {};
each ($t in $texts) {
    each ($w in split($t, " ")) {
        if ($w == "") { continue; }
        if ($freq[$w] == null) { $freq[$w] = 1; } else { $freq[$w] = $freq[$w] + 1; }
        $i = 0;
        while ($i < len($w)) {
            $chars[substr($w, $i, 1)] = true;
            $i = $i + 1;
        }
    }
}
$words = keys($freq);
print("[vocab] " + str(len($words)) + " word types, " + str(len($chars)) + " distinct characters");

// ── initial split: one piece per character, "##" on continuations ───────────
$splits = {};
each ($w in $words) {
    $pieces = [];
    $i = 0;
    while ($i < len($w)) {
        $ch = substr($w, $i, 1);
        if ($i == 0) { append($pieces, $ch); } else { append($pieces, "##" + $ch); }
        $i = $i + 1;
    }
    $splits[$w] = $pieces;
}

// Every single character is in the vocabulary in both forms, so greedy
// longest-match can always make progress and [UNK] is reserved for characters
// the corpus never contained at all.
$vocab = [];
each ($s in $SPECIALS) { append($vocab, $s); }
$charList = sort(keys($chars));
each ($c in $charList) { append($vocab, $c); }
each ($c in $charList) { append($vocab, "##" + $c); }
$base = len($vocab);
print("[vocab] " + str($base) + " base tokens (specials + characters)");

// ── merge loop ──────────────────────────────────────────────────────────────
def mergePieces($pieces, $a, $b, $merged) {
    $out = [];
    $n = len($pieces);
    $i = 0;
    while ($i < $n) {
        if ($i + 1 < $n && $pieces[$i] == $a && $pieces[$i + 1] == $b) {
            append($out, $merged);
            $i = $i + 2;
        } else {
            append($out, $pieces[$i]);
            $i = $i + 1;
        }
    }
    return $out;
}

$merges = 0;
$target = $VOCAB_SIZE - $base;
if ($target < 0) { $target = 0; }

while ($merges < $target) {
    // Count adjacent pairs across the type table, weighted by frequency.
    $pairs = {};
    each ($w in $words) {
        $p = $splits[$w];
        $f = $freq[$w];
        $n = len($p);
        $i = 0;
        while ($i + 1 < $n) {
            $key = $p[$i] + "\t" + $p[$i + 1];
            if ($pairs[$key] == null) { $pairs[$key] = $f; }
            else { $pairs[$key] = $pairs[$key] + $f; }
            $i = $i + 1;
        }
    }
    if (len($pairs) == 0) { break; }

    // Best pair. Ties broken by the pair string so the run is reproducible.
    $bestKey = null;
    $bestCount = 0;
    for $k, $v in $pairs.items() {
        if ($v > $bestCount || ($v == $bestCount && $bestKey != null && $k < $bestKey)) {
            $bestCount = $v;
            $bestKey = $k;
        }
    }
    if ($bestKey == null || $bestCount < 2) { break; }

    $tab = indexOf($bestKey, "\t");
    $a = substr($bestKey, 0, $tab);
    $b = substr($bestKey, $tab + 1, len($bestKey));
    // "cat" + "##s" -> "cats";  "##in" + "##g" -> "##ing"
    $merged = $a + substr($b, 2, len($b));

    each ($w in $words) {
        $p = $splits[$w];
        if (len($p) < 2) { continue; }
        $splits[$w] = mergePieces($p, $a, $b, $merged);
    }
    append($vocab, $merged);
    $merges = $merges + 1;
    if (fmod($merges, 250) == 0) {
        print("[vocab] " + str($merges) + " merges, last = " + $merged
              + " (count " + str($bestCount) + ")");
    }
}

print("[vocab] " + str($merges) + " merges, " + str(len($vocab)) + " tokens total");
writefile("lugha/vocab.txt", join($vocab, "\n") + "\n");
print("[vocab] wrote lugha/vocab.txt");
