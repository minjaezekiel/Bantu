// ════════════════════════════════════════════════════════════════════════
//  one_taylor_recsys_test.b — similarity, ranking, and the order of
//  operations that keeps the age wall intact.
//
//  The property that matters here is not "are the scores good". It is that
//  GATES RUN BEFORE SCORING. A recommender that scores everything and filters
//  afterwards is one forgotten branch away from showing a child to an adult,
//  and the bug is invisible in any test that only checks the top result.
//
//  So: the arithmetic is tested for the properties a similarity measure must
//  have (bounded, symmetric, 1 on identity, 0 on disjoint), the ranking is
//  tested for stability and for actually being descending, and combine() is
//  tested across the numba crossover so both paths agree.
//
//  Run:  bantu run tests/one_taylor_recsys_test.b
// ════════════════════════════════════════════════════════════════════════

include "../samples/one-taylor/backend/recsys/vectors.b" as vec;

$R = {"pass": 0, "fail": 0};
def ok($cond, $name) {
    if ($cond) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $name); }
}
def close($got, $want, $tol, $name) {
    $d = $got - $want;
    if ($d < 0) { $d = 0 - $d; }
    if ($d <= $tol) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name + "  got " + str($got) + " want " + str($want));
    }
}

def tags($list, $kind) {
    $rows = [];
    each ($t in $list) { append($rows, { "kind": $kind, "tag": $t }); }
    return $rows;
}

print("-- tagSet --");
$rows = [{"kind": "interest", "tag": "chess"},
         {"kind": "interest", "tag": "baking"},
         {"kind": "hobby", "tag": "guitar"},
         {"kind": "dream_partner", "tag": "kind"}];
ok(len(keys(vec.tagSet($rows, "interest"))) == 2, "tagSet filters by kind");
ok(len(keys(vec.tagSet($rows, null))) == 4, "a null kind takes everything");
ok(vec.tagSet($rows, "interest")["chess"] == true, "and indexes by tag");
ok(len(keys(vec.tagSet([], null))) == 0, "an empty list gives an empty set");
// A duplicate tag must not inflate the overlap: the uniqueness index on
// user_tags makes this impossible through the API, but jaccard must not rely
// on a database constraint for its arithmetic to be right.
$dupes = [{"kind": "interest", "tag": "chess"}, {"kind": "interest", "tag": "chess"}];
ok(len(keys(vec.tagSet($dupes, "interest"))) == 1, "duplicates collapse");

print("-- jaccard: the properties a similarity measure must have --");
$a = vec.tagSet(tags(["chess", "baking", "hiking"], "interest"), null);
$b = vec.tagSet(tags(["chess", "baking", "hiking"], "interest"), null);
$c = vec.tagSet(tags(["chess", "guitar"], "interest"), null);
$d = vec.tagSet(tags(["running", "swimming"], "interest"), null);
$empty = {};

close(vec.jaccard($a, $b), 1.0, 0.0000001, "identical sets score 1");
close(vec.jaccard($a, $d), 0.0, 0.0000001, "disjoint sets score 0");
close(vec.jaccard($a, $c), 0.25, 0.0000001, "one shared of four total is 0.25");
close(vec.jaccard($c, $a), 0.25, 0.0000001, "and it is symmetric");
close(vec.jaccard($a, $empty), 0.0, 0.0000001, "an empty profile matches nothing");
close(vec.jaccard($empty, $empty), 0.0, 0.0000001,
      "two empty profiles are NOT a perfect match — 0/0 is 0 here, on purpose");

$bounded = true;
each ($x in [$a, $b, $c, $d, $empty]) {
    each ($y in [$a, $b, $c, $d, $empty]) {
        $j = vec.jaccard($x, $y);
        if ($j < 0.0 || $j > 1.0) { $bounded = false; }
    }
}
ok($bounded, "jaccard stays within [0, 1] for every pair");

print("-- overlap and shared --");
ok(vec.overlapCount($a, $c) == 1, "overlapCount counts the intersection");
ok(vec.overlapCount($c, $a) == 1, "walking the smaller set gives the same answer");
ok(vec.overlapCount($a, $empty) == 0, "nothing overlaps an empty set");
$shared = vec.sharedTags($a, $c);
ok(len($shared) == 1 && $shared[0] == "chess", "sharedTags names the overlap");
ok(len(vec.sharedTags($a, $d)) == 0, "and is empty when there is none");
// Sorted, because the UI prints it and an unstable order makes the same match
// look different on every refresh.
$twoWay = vec.sharedTags($a, $b);
ok($twoWay[0] == "baking" && $twoWay[1] == "chess" && $twoWay[2] == "hiking",
   "sharedTags is sorted, so the explanation is stable between refreshes");

print("-- combine: the weighted sum --");
$comp = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]];
$w = [0.75, 0.25];
$out = vec.combine($comp, $w);
close($out[0], 0.75, 0.0000001, "the first candidate takes the first weight");
close($out[1], 0.25, 0.0000001, "the second takes the second");
close($out[2], 0.5, 0.0000001, "and a tie splits as it should");
ok(len(vec.combine([], [])) == 0, "no components means no scores");
ok(len(vec.combine([[]], [1.0])) == 0, "no candidates means no scores");

print("-- the two paths agree across the numba crossover --");
// vectors.b switches from interpreted lists to native arrays above 10,000
// candidates. The two must produce the same numbers, or the ranking changes
// shape at an arbitrary population size.
$n = 10050;
$c1 = [];
$c2 = [];
$i = 0;
while ($i < $n) {
    append($c1, fmod($i, 7) / 7.0);
    append($c2, fmod($i, 3) / 3.0);
    $i = $i + 1;
}
$big = vec.combine([$c1, $c2], [0.6, 0.4]);
ok(len($big) == $n, "the native path returns one score per candidate");
$maxDiff = 0.0;
$i = 0;
while ($i < $n) {
    $expected = 0.6 * $c1[$i] + 0.4 * $c2[$i];
    $diff = $big[$i] - $expected;
    if ($diff < 0) { $diff = 0 - $diff; }
    if ($diff > $maxDiff) { $maxDiff = $diff; }
    $i = $i + 1;
}
close($maxDiff, 0.0, 0.0000000001, "and agrees with the interpreted path to rounding");

print("-- rankTop --");
$items = [{"name": "a"}, {"name": "b"}, {"name": "c"}, {"name": "d"}];
$scores = [0.1, 0.9, 0.5, 0.7];
$top = vec.rankTop($items, $scores, 2);
ok(len($top) == 2, "rankTop honours the limit");
ok($top[0]["name"] == "b", "the highest score comes first");
ok($top[1]["name"] == "d", "then the next highest");
close($top[0]["score"], 0.9, 0.0000001, "the score travels with the item");

$all = vec.rankTop($items, $scores, null);
ok(len($all) == 4, "a null limit returns everything");
$descending = true;
$i = 1;
while ($i < len($all)) {
    if ($all[$i]["score"] > $all[$i - 1]["score"]) { $descending = false; }
    $i = $i + 1;
}
ok($descending, "the whole list is descending");

$more = vec.rankTop($items, $scores, 99);
ok(len($more) == 4, "a limit past the end is not an error");

// Ties must not reorder between calls: the feed is re-ranked on every request
// and a user watching two equal posts swap places sees a bug, not a tie.
$tied = vec.rankTop([{"name": "x"}, {"name": "y"}, {"name": "z"}],
                    [0.5, 0.5, 0.5], null);
$tied2 = vec.rankTop([{"name": "x"}, {"name": "y"}, {"name": "z"}],
                     [0.5, 0.5, 0.5], null);
ok($tied[0]["name"] == $tied2[0]["name"]
   && $tied[1]["name"] == $tied2[1]["name"]
   && $tied[2]["name"] == $tied2[2]["name"],
   "a three-way tie ranks identically twice — the sort is stable");

print("-- recency decay --");
close(vec.recencyWeight(0, 36), 1.0, 0.0000001, "a brand-new post has full weight");
close(vec.recencyWeight(36, 36), 0.5, 0.0000001, "one half-life halves it");
close(vec.recencyWeight(72, 36), 0.25, 0.0000001, "two half-lives quarter it");
ok(vec.recencyWeight(1000, 36) > 0.0, "and it never reaches zero, so old posts are reachable");
close(vec.recencyWeight(-5, 36), 0.0, 0.0000001,
      "a clock skew into the future weighs 0 rather than exploding");
ok(vec.recencyWeight(null, 36) == 0.0, "a missing timestamp weighs 0");

$monotone = true;
$h = 0;
$prev = 2.0;
while ($h <= 240) {
    $w2 = vec.recencyWeight($h, 36);
    if ($w2 > $prev) { $monotone = false; }
    $prev = $w2;
    $h = $h + 6;
}
ok($monotone, "decay is monotone over ten days");

print("");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); } else { print("  RESULT: FAILED"); }
