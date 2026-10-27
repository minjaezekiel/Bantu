// ════════════════════════════════════════════════════════════════════════
//  lugha_tokenizer_test.b — normalisation and WordPiece.
//
//  The normaliser is the stage an evader attacks first, so it is tested
//  against the actual evasion families rather than against pretty input:
//  Unicode confusables, zero-width insertions, leetspeak, masked letters,
//  stretched letters, and separator-spelled words. Each has an expected
//  canonical form, and the test states it rather than asserting "something
//  changed".
//
//  The hard requirement in the other direction gets equal space: a number
//  must survive. "i am 13" is the single most load-bearing fact a grooming
//  classifier can read, and a leetspeak rule that rewrites every digit turns
//  it into "i am ie".
//
//  Run:  bantu run tests/lugha_tokenizer_test.b
// ════════════════════════════════════════════════════════════════════════

include "../lugha/normalise.b" as nz;
include "../lugha/tokenizer.b" as tk;

$R = {"pass": 0, "fail": 0};

def eq($got, $want, $name) {
    if ($got == $want) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name);
        print("          got:  [" + str($got) + "]");
        print("          want: [" + str($want) + "]");
    }
}
def ok($cond, $name) {
    if ($cond) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $name); }
}

print("-- UTF-8 decoding --");
eq(len(nz.codepoints("abc")), 3, "three ASCII bytes are three code points");
eq(len(nz.codepoints("café")), 4, "a 2-byte sequence is one code point");
eq(len(nz.codepoints("日本")), 2, "a 3-byte sequence is one code point");
eq(nz.codepoints("A")[0], 65, "the code point value is right");
eq(nz.codepoints("é")[0], 233, "and for a Latin-1 supplement character");
// A truncated sequence must not vanish: anything that can vanish can hide a word.
ok(len(nz.codepoints(frombytes([195]))) == 1, "a truncated sequence becomes one replacement");

print("-- confusable folding --");
eq(nz.normalise("ƒuck"), "fuck", "the florin sign folds to f");
eq(nz.normalise("ѕех"), "sex", "Cyrillic lookalikes fold to Latin");
eq(nz.normalise("ВITCH"), "bitch", "a Cyrillic capital in a Latin word");
eq(nz.normalise("héllo café"), "hello cafe", "accents are stripped");
eq(nz.normalise("ＨＡＴＥ ｙｏｕ"), "hate you", "fullwidth forms fold and lowercase");
eq(nz.normalise("ⓗⓐⓣⓔ"), "hate", "circled letters fold");
eq(nz.normalise("𝐡𝐚𝐭𝐞"), "hate", "mathematical bold folds");

print("-- invisible characters --");
eq(nz.normalise("f" + frombytes([226, 128, 139]) + "uck"), "fuck",
   "a zero-width space is deleted, not turned into a gap");
eq(nz.normalise("fu" + frombytes([239, 187, 191]) + "ck"), "fuck",
   "a BOM in the middle of a word is deleted");
eq(nz.normalise("fu" + frombytes([194, 173]) + "ck"), "fuck",
   "a soft hyphen is deleted");

print("-- leetspeak, and the digits that must survive --");
eq(nz.normalise("@ss h0le"), "ass hole", "@ and 0 inside words are letters");
eq(nz.normalise("b!tch"), "bitch", "! is an i inside a word");
eq(nz.normalise("$hit"), "shit", "$ is an s");
eq(nz.normalise("i am 13 years old"), "i am 13 years old",
   "A BARE NUMBER IS NOT LEETSPEAK — this is the one that matters");
eq(nz.normalise("im 15 and i play chess"), "im 15 and i play chess",
   "an age next to words still survives");
eq(nz.normalise("meet me at 7"), "meet me at 7", "a lone digit survives");
eq(nz.normalise("year 9 is fine"), "year 9 is fine", "a digit as its own word survives");

print("-- masked letters --");
eq(nz.normalise("sh*t"), "sht", "a mask is dropped, not guessed");
eq(nz.normalise("f**k"), "fk", "two masks are dropped");
eq(nz.normalise("b#tch"), "btch", "# is a mask too");

print("-- stretched letters --");
eq(nz.normalise("fuuuuuck"), "fuuck", "a long run collapses to two");
eq(nz.normalise("fuuck"), "fuuck", "a run of two is already canonical");
eq(nz.normalise("too many spaces"), "too many spaces",
   "a genuine double letter survives — this is why runs collapse to TWO");
eq(nz.normalise("alll goood"), "all good", "three collapses to two, which is the word");

print("-- separator-spelled words --");
eq(nz.normalise("f.u.c.k you"), "fuck you", "dot-separated, mid-sentence");
eq(nz.normalise("F-U-C-K"), "fuck", "hyphen-separated");
eq(nz.normalise("f u c k off"), "fuck off", "space-separated");
eq(nz.normalise("u.s.a is a country"), "usa is a country",
   "an acronym normalises to its letters, which is the same token");
eq(nz.normalise("a-b and c-d"), "a-b and c-d",
   "two letters are NOT a run — ordinary hyphenation survives");
eq(nz.normalise("well-meaning person"), "well-meaning person",
   "a hyphenated word survives");

print("-- whitespace and empties --");
eq(nz.normalise("  lots   of    space  "), "lots of space", "runs of space collapse");
eq(nz.normalise(""), "", "empty stays empty");
eq(nz.normalise(null), "", "null normalises to empty, not to \"null\"");
eq(nz.normalise("\n\t  \n"), "", "whitespace only becomes empty");

print("-- the tokenizer --");
$size = tk.load("lugha/vocab.txt");
ok($size > 200, "the vocabulary loaded (" + str($size) + " tokens)");
eq(tk.id("[PAD]"), 0, "[PAD] is id 0");
eq(tk.id("[UNK]"), 1, "[UNK] is id 1");
eq(tk.id("[CLS]"), 2, "[CLS] is id 2");
eq(tk.tokenAt(0), "[PAD]", "ids map back to tokens");
eq(tk.id("a token that is definitely not in the vocabulary"), 1,
   "an unknown token is [UNK]");

print("-- round-tripping through pieces --");
// Whatever the vocabulary happens to contain, stripping "##" and joining the
// pieces must reproduce the word. If it does not, the tokenizer is dropping
// or duplicating characters, which no accuracy number would explain.
def rejoin($word) {
    $parts = [];
    each ($p in tk.piecesOf($word)) {
        if (substr($p, 0, 2) == "##") { append($parts, substr($p, 2, len($p))); }
        else { append($parts, $p); }
    }
    return join($parts, "");
}
$roundTripFails = 0;
each ($w in ["hello", "chess", "grooming", "zzzqqq", "football", "13",
             "photography", "a", "xy", "secret", "parents"]) {
    $pieces = tk.piecesOf($w);
    if (len($pieces) == 1 && $pieces[0] == "[UNK]") { continue; }
    if (rejoin($w) != $w) {
        $roundTripFails = $roundTripFails + 1;
        print("          " + $w + " -> " + join($pieces, " ") + " -> " + rejoin($w));
    }
}
ok($roundTripFails == 0, "every word rejoins to itself from its pieces");

print("-- punctuation is its own token --");
$t = tk.tokenize("dont, really");
ok(len($t) >= 3, "a comma is split off rather than glued to a word");
$t2 = tk.tokenize("i'll go");
$hasApostrophe = false;
each ($p in $t2) { if ($p == "'") { $hasApostrophe = true; } }
ok($hasApostrophe, "an apostrophe survives as a token");

print("-- encode --");
$e = tk.encode("hello world", 16);
eq(len($e["ids"]), 16, "ids are padded to the window");
eq(len($e["mask"]), 16, "the mask is the same length");
eq($e["ids"][0], 2, "position 0 is [CLS]");
ok($e["n"] >= 3, "the real length counts CLS plus the tokens");
eq($e["mask"][$e["n"] - 1], 1, "the last real position is masked in");
eq($e["mask"][$e["n"]], 0, "the first pad position is masked out");
eq($e["ids"][15], 0, "padding is [PAD]");

$long = "";
$i = 0;
while ($i < 200) { $long = $long + "word "; $i = $i + 1; }
$eLong = tk.encode(nz.normalise($long), 64);
eq($eLong["n"], 64, "a long text fills the window exactly");
eq(len($eLong["ids"]), 64, "and does not overflow it");

print("-- windows cover long text --");
$w = tk.windows(nz.normalise($long), 64, 48);
ok(len($w) > 1, "a long text becomes several windows (" + str(len($w)) + ")");
$allRight = true;
each ($win in $w) {
    if (len($win["ids"]) != 64) { $allRight = false; }
    if ($win["n"] < 1 || $win["n"] > 64) { $allRight = false; }
}
ok($allRight, "every window is exactly one window long");
$short = tk.windows(nz.normalise("hello"), 64, 48);
eq(len($short), 1, "a short text is one window");

print("");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); } else { print("  RESULT: FAILED"); }
