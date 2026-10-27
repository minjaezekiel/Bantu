// ============================================================================
// lugha/normalise.b — text normalisation, the stage before the model
//
// Folds the whole family of obfuscations into one canonical ASCII form, so the
// classifier never has to spend capacity learning spelling. See
// docs/lugha-architecture.md §2 for why this is not the model's job.
//
// Exposes:
//   codepoints($text)  -> list of Unicode code points
//   fold($text)        -> confusables folded, zero-width stripped, lowercased
//   normalise($text)   -> fold + leetspeak + separator strip + run collapse
// ============================================================================

// ── the confusable table ────────────────────────────────────────────────────
// Built once, at include time. Keys are code points as strings (Bantu dict keys
// are strings); a value of "" means "drop this code point entirely".
$_FOLD = {};

def _fold1($cp, $ch) { $_FOLD[str($cp)] = $ch; }

def _foldRange($from, $to, $ch) {
    $i = $from;
    while ($i <= $to) {
        $_FOLD[str($i)] = $ch;
        $i = $i + 1;
    }
}

// Map a contiguous run of code points onto a contiguous run of ASCII, which is
// how every alphabet-shaped Unicode block is laid out: fullwidth Latin,
// mathematical bold/italic/script/monospace, circled letters.
def _foldSeq($from, $count, $asciiStart) {
    $i = 0;
    while ($i < $count) {
        $_FOLD[str($from + $i)] = chr($asciiStart + $i);
        $i = $i + 1;
    }
}

def _buildFoldTable() {
    // Latin-1 accented forms.
    _foldRange(192, 197, "a");  _fold1(198, "ae"); _fold1(199, "c");
    _foldRange(200, 203, "e");  _foldRange(204, 207, "i"); _fold1(208, "d");
    _fold1(209, "n");           _foldRange(210, 214, "o"); _fold1(216, "o");
    _foldRange(217, 220, "u");  _fold1(221, "y");  _fold1(223, "ss");
    _foldRange(224, 229, "a");  _fold1(230, "ae"); _fold1(231, "c");
    _foldRange(232, 235, "e");  _foldRange(236, 239, "i"); _fold1(240, "d");
    _fold1(241, "n");           _foldRange(242, 246, "o"); _fold1(248, "o");
    _foldRange(249, 252, "u");  _fold1(253, "y");  _fold1(255, "y");

    // Latin Extended-A / B — the ones that actually turn up in evasion.
    _fold1(263, "c");  _fold1(269, "c");  _fold1(271, "d");  _fold1(275, "e");
    _fold1(279, "e");  _fold1(287, "g");  _fold1(305, "i");  _fold1(322, "l");
    _fold1(324, "n");  _fold1(328, "n");  _fold1(339, "oe"); _fold1(345, "r");
    _fold1(347, "s");  _fold1(351, "s");  _fold1(353, "s");  _fold1(357, "t");
    _fold1(363, "u");  _fold1(371, "u");  _fold1(378, "z");  _fold1(382, "z");
    _fold1(402, "f");                                      // ƒ
    _fold1(409, "s");  _fold1(437, "z");

    // Cyrillic homoglyphs — lowercase then uppercase.
    _fold1(1072, "a"); _fold1(1074, "b"); _fold1(1077, "e"); _fold1(1082, "k");
    _fold1(1084, "m"); _fold1(1085, "h"); _fold1(1086, "o"); _fold1(1088, "p");
    _fold1(1089, "c"); _fold1(1090, "t"); _fold1(1091, "y"); _fold1(1093, "x");
    _fold1(1077, "e"); _fold1(1110, "i"); _fold1(1109, "s"); _fold1(1169, "g");
    _fold1(1040, "a"); _fold1(1042, "b"); _fold1(1045, "e"); _fold1(1050, "k");
    _fold1(1052, "m"); _fold1(1053, "h"); _fold1(1054, "o"); _fold1(1056, "p");
    _fold1(1057, "c"); _fold1(1058, "t"); _fold1(1059, "y"); _fold1(1061, "x");
    _fold1(1030, "i"); _fold1(1029, "s");

    // Greek homoglyphs.
    _fold1(945, "a");  _fold1(946, "b");  _fold1(949, "e");  _fold1(953, "i");
    _fold1(954, "k");  _fold1(957, "v");  _fold1(959, "o");  _fold1(961, "p");
    _fold1(964, "t");  _fold1(965, "u");  _fold1(967, "x");  _fold1(921, "i");
    _fold1(913, "a");  _fold1(914, "b");  _fold1(917, "e");  _fold1(919, "h");
    _fold1(922, "k");  _fold1(924, "m");  _fold1(925, "n");  _fold1(927, "o");
    _fold1(929, "p");  _fold1(932, "t");  _fold1(935, "x");

    // Fullwidth Latin and digits.
    _foldSeq(65313, 26, 97);    // Ａ-Ｚ -> a-z
    _foldSeq(65345, 26, 97);    // ａ-ｚ -> a-z
    _foldSeq(65296, 10, 48);    // ０-９ -> 0-9

    // Mathematical alphanumerics. Thirteen blocks of 26+26, all laid out
    // uppercase-then-lowercase, which is why one loop covers them all.
    $blocks = [119808, 119860, 119912, 119964, 120016, 120068, 120120,
               120172, 120224, 120276, 120328, 120380, 120432];
    each ($b in $blocks) {
        _foldSeq($b, 26, 97);        // the uppercase half, folded to lowercase
        _foldSeq($b + 26, 26, 97);   // the lowercase half
    }
    _foldSeq(120782, 10, 48);        // mathematical bold digits

    // Circled / parenthesised letters.
    _foldSeq(9398, 26, 97);          // Ⓐ-Ⓩ
    _foldSeq(9424, 26, 97);          // ⓐ-ⓩ

    // Code points to delete outright: zero-width, bidi controls, soft hyphen,
    // combining marks, variation selectors. These exist in evasion precisely
    // because they are invisible.
    _fold1(173, "");
    _foldRange(768, 879, "");        // combining diacriticals
    _foldRange(8203, 8207, "");      // ZWSP, ZWNJ, ZWJ, LRM, RLM
    _foldRange(8234, 8238, "");      // bidi embedding controls
    _foldRange(8288, 8292, "");      // word joiner, invisible operators
    _foldRange(65024, 65039, "");    // variation selectors
    _fold1(65279, "");               // BOM / ZWNBSP

    // Punctuation that a lookalike smuggles in.
    _fold1(8216, "'"); _fold1(8217, "'"); _fold1(8220, "\""); _fold1(8221, "\"");
    _fold1(8211, "-"); _fold1(8212, "-"); _fold1(8230, "...");
    _fold1(160,  " "); _foldRange(8192, 8202, " "); _fold1(8239, " ");
    _fold1(9, " ");    _fold1(10, " ");  _fold1(13, " ");
}

_buildFoldTable();

// ── UTF-8 -> code points ────────────────────────────────────────────────────
// Hand-rolled because the interpreter has no decoder, and because a classifier
// fed mojibake is a classifier that can be evaded with an encoding trick.
// A malformed sequence becomes U+FFFD rather than being skipped: skipping lets
// a byte sequence vanish, and anything that can vanish can hide a word.
def codepoints($text) {
    $b = bytes($text);
    $n = len($b);
    $out = [];
    $i = 0;
    while ($i < $n) {
        $c = $b[$i];
        if ($c < 128) {
            append($out, $c);
            $i = $i + 1;
        } else if ($c >= 240 && $c < 248 && $i + 3 < $n) {
            append($out, (($c - 240) * 262144) + (($b[$i + 1] - 128) * 4096)
                       + (($b[$i + 2] - 128) * 64) + ($b[$i + 3] - 128));
            $i = $i + 4;
        } else if ($c >= 224 && $i + 2 < $n) {
            append($out, (($c - 224) * 4096) + (($b[$i + 1] - 128) * 64)
                       + ($b[$i + 2] - 128));
            $i = $i + 3;
        } else if ($c >= 192 && $i + 1 < $n) {
            append($out, (($c - 192) * 64) + ($b[$i + 1] - 128));
            $i = $i + 2;
        } else {
            append($out, 65533);
            $i = $i + 1;
        }
    }
    return $out;
}

// ── fold: confusables + strip + lowercase ───────────────────────────────────
def fold($text) {
    $cps = codepoints($text);
    $parts = [];
    each ($cp in $cps) {
        $mapped = $_FOLD[str($cp)];
        if ($mapped != null) {
            if ($mapped != "") { append($parts, $mapped); }
        } else if ($cp < 128) {
            append($parts, chr($cp));
        } else {
            // An unmapped non-ASCII code point becomes a space rather than
            // disappearing, so "hate⟨emoji⟩you" cannot become one word.
            append($parts, " ");
        }
    }
    return join($parts, "").lower();
}

// ── leetspeak ───────────────────────────────────────────────────────────────
// Only inside an alphanumeric run that already contains a letter. A bare
// number must survive: "i am 13" is evidence, and turning it into "i am ie"
// destroys exactly the signal a grooming classifier needs.
$_LEET = {
    "@": "a", "4": "a", "8": "b", "(": "c", "3": "e", "6": "g", "9": "g",
    "1": "i", "!": "i", "|": "i", "0": "o", "5": "s", "$": "s", "7": "t",
    "+": "t", "2": "z"
};

// A mask character stands for a letter the writer chose not to type. There is
// no principled way to guess WHICH letter, so it is dropped: "sh*t" -> "sht",
// "f**k" -> "fk". The skeleton that remains is not an English word and is
// still highly diagnostic, and the corpus carries masked forms so the model
// sees them. Guessing a letter would be inventing evidence.
$_MASK = { "*": true, "#": true, "%": true, "&": true, "~": true };

def _isLetter($ch)  { return $ch >= "a" && $ch <= "z"; }
def _isDigit($ch)   { return $ch >= "0" && $ch <= "9"; }

def _isWordish($ch) {
    if (_isLetter($ch) || _isDigit($ch)) { return true; }
    if ($_MASK[$ch] != null) { return true; }
    return $_LEET[$ch] != null;
}

def deleet($s) {
    $n = len($s);
    $out = [];
    $i = 0;
    while ($i < $n) {
        // Take the next run of word-ish characters.
        $j = $i;
        $hasLetter = false;
        while ($j < $n) {
            $ch = substr($s, $j, 1);
            if (!_isWordish($ch)) { break; }
            if (_isLetter($ch)) { $hasLetter = true; }
            $j = $j + 1;
        }
        if ($j == $i) {
            append($out, substr($s, $i, 1));
            $i = $i + 1;
        } else {
            $k = $i;
            while ($k < $j) {
                $ch = substr($s, $k, 1);
                if ($hasLetter && $_MASK[$ch] != null) {
                    // dropped — see $_MASK above
                } else if ($hasLetter && $_LEET[$ch] != null) {
                    append($out, $_LEET[$ch]);
                } else {
                    append($out, $ch);
                }
                $k = $k + 1;
            }
            $i = $j;
        }
    }
    return join($out, "");
}

// ── separator strip inside a word ───────────────────────────────────────────
// "f.u.c.k", "f-u-c-k" and "f u c k" all mean "fuck". The rule is the same in
// every case — three or more single letters joined by the same separator — but
// it has to be applied at two different scales, which is why there are two
// functions rather than one clever one.
//
// Per word for punctuation: splitting the WHOLE string on "." makes the last
// piece of "f.u.c.k you" into "k you", which is not a single letter, so the
// run ends one character early and leaves "fuc.k you" behind. Words first.
$_SEPS = [".", "-", "_", ",", "'", "/", ":"];

// Join runs of >= 3 single-letter pieces; leave everything else alone.
def _joinSingles($parts, $sep) {
    $n = len($parts);
    $out = [];
    $i = 0;
    while ($i < $n) {
        $j = $i;
        while ($j < $n && len($parts[$j]) == 1 && _isLetter($parts[$j])) {
            $j = $j + 1;
        }
        if ($j - $i >= 3) {
            $run = [];
            $k = $i;
            while ($k < $j) { append($run, $parts[$k]); $k = $k + 1; }
            append($out, join($run, ""));
            $i = $j;
        } else {
            append($out, $parts[$i]);
            $i = $i + 1;
        }
    }
    return join($out, $sep);
}

def _stripSeparatedWord($w) {
    each ($sep in $_SEPS) {
        if (len($w) < 5) { return $w; }
        $parts = split($w, $sep);
        if (len($parts) >= 3) { $w = _joinSingles($parts, $sep); }
    }
    return $w;
}

def _stripSeparated($s) {
    // Space-separated single letters first, at string scale.
    $s = _joinSingles(split($s, " "), " ");
    $words = split($s, " ");
    $out = [];
    each ($w in $words) { append($out, _stripSeparatedWord($w)); }
    return join($out, " ");
}

// ── run collapse ────────────────────────────────────────────────────────────
// Three or more identical letters collapse to TWO, not one: "too", "see" and
// "all" are words, and "fuuuuck" and "fuuck" tokenize the same way.
def _collapseRuns($s) {
    $n = len($s);
    $out = [];
    $i = 0;
    while ($i < $n) {
        $ch = substr($s, $i, 1);
        $j = $i;
        while ($j < $n && substr($s, $j, 1) == $ch) { $j = $j + 1; }
        $runLen = $j - $i;
        if ($runLen > 2) { $runLen = 2; }
        $k = 0;
        while ($k < $runLen) { append($out, $ch); $k = $k + 1; }
        $i = $j;
    }
    return join($out, "");
}

// ── whitespace ──────────────────────────────────────────────────────────────
def _squeezeSpaces($s) {
    $parts = split($s, " ");
    $kept = [];
    each ($p in $parts) {
        if ($p != "") { append($kept, $p); }
    }
    return join($kept, " ");
}

// ── the whole pipeline ──────────────────────────────────────────────────────
def normalise($text) {
    if ($text == null) { return ""; }
    $s = fold($text);
    $s = _stripSeparated($s);
    $s = deleet($s);
    $s = _collapseRuns($s);
    return trim(_squeezeSpaces($s));
}
