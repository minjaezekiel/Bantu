// ============================================================================
// lugha/tokenizer.b — WordPiece tokenizer
//
// Greedy longest-match over a vocabulary where a piece continuing a word
// carries a "##" prefix. Input is expected to be ALREADY NORMALISED
// (normalise.b): the tokenizer deliberately does no folding of its own, so
// there is exactly one place where surface form is decided.
//
// Exposes:
//   load($vocabPath)              -> token count
//   vocabSize()                   -> token count
//   id($token) / tokenAt($id)
//   tokenize($text)               -> list of piece strings
//   encode($text, $maxLen)        -> {"ids", "mask", "n"}
//   windows($text, $maxLen, $stride) -> list of encodings covering long text
// ============================================================================

const $PAD = 0;
const $UNK = 1;
const $CLS = 2;
const $SEP = 3;

// Dicts are reference types, so a function body may fill these in place; a
// LIST assigned to a global name inside a function would not propagate.
$V = { "tok2id": {}, "id2tok": [], "size": 0, "maxPiece": 1, "loaded": false };

def load($vocabPath) {
    $raw = readfile($vocabPath);
    $lines = split($raw, "\n");
    $tok2id = {};
    $id2tok = [];
    $maxPiece = 1;
    $i = 0;
    each ($line in $lines) {
        if ($line == "") { continue; }
        $tok2id[$line] = $i;
        append($id2tok, $line);
        $plain = $line;
        if (substr($line, 0, 2) == "##") { $plain = substr($line, 2, len($line)); }
        if (len($plain) > $maxPiece && substr($line, 0, 1) != "[") {
            $maxPiece = len($plain);
        }
        $i = $i + 1;
    }
    $V["tok2id"] = $tok2id;
    $V["id2tok"] = $id2tok;
    $V["size"] = $i;
    $V["maxPiece"] = $maxPiece;
    $V["loaded"] = true;
    return $i;
}

def vocabSize() { return $V["size"]; }

def id($token) {
    $v = $V["tok2id"][$token];
    if ($v == null) { return $UNK; }
    return $v;
}

def tokenAt($i) {
    $l = $V["id2tok"];
    if ($i < 0 || $i >= len($l)) { return "[UNK]"; }
    return $l[$i];
}

// ── one word -> pieces ──────────────────────────────────────────────────────
// Longest-match from the left. A word no prefix of which is in the vocabulary
// becomes a single [UNK] rather than a run of them: a run would tell the model
// how LONG the unknown word was, which is noise, and would also let a long
// unknown string push real tokens out of the 64-token window.
def piecesOf($word) {
    $tok2id = $V["tok2id"];
    $n = len($word);
    $out = [];
    $start = 0;
    while ($start < $n) {
        $end = $n;
        if ($end - $start > $V["maxPiece"]) { $end = $start + $V["maxPiece"]; }
        $found = null;
        while ($end > $start) {
            $sub = substr($word, $start, $end - $start);
            if ($start > 0) { $sub = "##" + $sub; }
            if ($tok2id[$sub] != null) { $found = $sub; break; }
            $end = $end - 1;
        }
        if ($found == null) { return ["[UNK]"]; }
        append($out, $found);
        $start = $end;
    }
    return $out;
}

// ── punctuation is its own token ────────────────────────────────────────────
def _isAlnum($ch) {
    if ($ch >= "a" && $ch <= "z") { return true; }
    if ($ch >= "0" && $ch <= "9") { return true; }
    return false;
}

// Split a whitespace-delimited chunk into alphanumeric runs and single
// punctuation marks. "don't" -> ["don", "'", "t"], and the apostrophe carries
// real signal ("youre" vs "your e") so it is kept rather than dropped.
def _chunks($word) {
    $out = [];
    $n = len($word);
    $i = 0;
    while ($i < $n) {
        $ch = substr($word, $i, 1);
        if (_isAlnum($ch)) {
            $j = $i;
            while ($j < $n && _isAlnum(substr($word, $j, 1))) { $j = $j + 1; }
            append($out, substr($word, $i, $j - $i));
            $i = $j;
        } else {
            append($out, $ch);
            $i = $i + 1;
        }
    }
    return $out;
}

def tokenize($text) {
    $out = [];
    each ($w in split($text, " ")) {
        if ($w == "") { continue; }
        each ($c in _chunks($w)) {
            extend($out, piecesOf($c));
        }
    }
    return $out;
}

// ── encode ──────────────────────────────────────────────────────────────────
// [CLS] occupies position 0. Mean pooling does not need it, but it gives the
// model one position whose content never varies, which is a free bias term and
// keeps the positional embedding anchored.
def encodeTokens($tokens, $maxLen) {
    $ids = [$CLS];
    $i = 0;
    while ($i < len($tokens) && len($ids) < $maxLen) {
        append($ids, id($tokens[$i]));
        $i = $i + 1;
    }
    $n = len($ids);
    $mask = [];
    $k = 0;
    while ($k < $maxLen) {
        if ($k < $n) { append($mask, 1); } else { append($mask, 0); append($ids, $PAD); }
        $k = $k + 1;
    }
    return { "ids": $ids, "mask": $mask, "n": $n };
}

def encode($text, $maxLen) {
    return encodeTokens(tokenize($text), $maxLen);
}

// ── long text ───────────────────────────────────────────────────────────────
// A post longer than the window is covered by overlapping windows; the caller
// takes the per-label maximum. Truncating instead would make "a paragraph of
// pleasantries followed by a threat" score as pleasantries, which is exactly
// how a moderation classifier gets evaded.
def windows($text, $maxLen, $stride) {
    $tokens = tokenize($text);
    $body = $maxLen - 1;              // one position goes to [CLS]
    $out = [];
    if (len($tokens) <= $body) {
        append($out, encodeTokens($tokens, $maxLen));
        return $out;
    }
    $start = 0;
    while ($start < len($tokens)) {
        $slice = [];
        $i = $start;
        $stop = $start + $body;
        if ($stop > len($tokens)) { $stop = len($tokens); }
        while ($i < $stop) { append($slice, $tokens[$i]); $i = $i + 1; }
        append($out, encodeTokens($slice, $maxLen));
        if ($stop >= len($tokens)) { break; }
        $start = $start + $stride;
    }
    return $out;
}
