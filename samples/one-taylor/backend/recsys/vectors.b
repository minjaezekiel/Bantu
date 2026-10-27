// ============================================================================
// one-taylor/backend/recsys/vectors.b — similarity and ranking arithmetic
//
// A correction to the first draft of this design, recorded because it was
// wrong in an instructive way: the plan said tag overlap would move to numba
// above ten thousand candidates. It should not. Tag overlap is a sparse set
// intersection -- a hash join over a few dozen strings per user -- and the
// plain dict version is already O(total tags). Handing it to numba would mean
// building a dense users x tags matrix, which is both slower and quadratically
// larger. numba is for dense arithmetic on contiguous numbers.
//
// What IS dense is the final score combination: five weighted components over
// every candidate, then a sort. That is where numba earns its place, and only
// once the candidate list is big enough to pay back the 1-3 us per builtin
// call -- numba's own documented crossover, which docs/numba.md puts at about
// ten thousand elements.
//
// Exposes:
//   tagSet($rows, $kind)         -> dict tag -> true
//   jaccard($setA, $setB)        -> 0..1
//   overlapCount($setA, $setB)   -> int
//   sharedTags($setA, $setB)     -> list (for "why you matched")
//   combine($components, $weights) -> list of scores   (numba above CROSSOVER)
//   rankTop($items, $scores, $n) -> the n highest, highest first
// ============================================================================

include "numba" as np;

const $CROSSOVER = 10000;

def tagSet($rows, $kind) {
    $s = {};
    each ($r in $rows) {
        if ($kind == null || $r["kind"] == $kind) { $s[$r["tag"]] = true; }
    }
    return $s;
}

def overlapCount($a, $b) {
    $n = 0;
    // Walk the smaller set: the cost is min(|a|, |b|) lookups, not |a| + |b|.
    $small = $a;
    $large = $b;
    if (len(keys($b)) < len(keys($a))) { $small = $b; $large = $a; }
    each ($k in keys($small)) {
        if ($large[$k] != null) { $n = $n + 1; }
    }
    return $n;
}

def sharedTags($a, $b) {
    $out = [];
    each ($k in keys($a)) {
        if ($b[$k] != null) { append($out, $k); }
    }
    return sort($out);
}

def jaccard($a, $b) {
    $ka = len(keys($a));
    $kb = len(keys($b));
    if ($ka == 0 || $kb == 0) { return 0.0; }
    $inter = overlapCount($a, $b);
    $union = $ka + $kb - $inter;
    if ($union == 0) { return 0.0; }
    return $inter / $union;
}

// $components is a list of equal-length score lists; $weights the same length
// as $components. Returns one combined score per candidate.
def combine($components, $weights) {
    if (len($components) == 0) { return []; }
    $n = len($components[0]);
    if ($n == 0) { return []; }

    if ($n < $CROSSOVER) {
        $out = [];
        $i = 0;
        while ($i < $n) {
            $s = 0.0;
            $c = 0;
            while ($c < len($components)) {
                $s = $s + $weights[$c] * $components[$c][$i];
                $c = $c + 1;
            }
            append($out, $s);
            $i = $i + 1;
        }
        return $out;
    }

    // Dense path. One allocation per component, then native multiply-add.
    $acc = np.zeros([$n], "f64");
    $c = 0;
    while ($c < len($components)) {
        np.add($acc, np.multiply(np.array($components[$c], "f64"), $weights[$c], null), $acc);
        $c = $c + 1;
    }
    return np.to_list($acc);
}

// Highest first. sort() with a KEY calls the key n times and orders natively;
// a comparator would be called O(n log n) times through the interpreter, which
// docs/language-features.md measures at 17x slower on 100k rows.
def rankTop($items, $scores, $n) {
    $paired = [];
    $i = 0;
    while ($i < len($items)) {
        $item = $items[$i];
        $item["score"] = $scores[$i];
        append($paired, $item);
        $i = $i + 1;
    }
    $sorted = sort($paired, {"key": def($r) { return $r["score"]; }, "desc": true});
    if ($n == null || $n >= len($sorted)) { return $sorted; }
    $out = [];
    $k = 0;
    while ($k < $n) { append($out, $sorted[$k]); $k = $k + 1; }
    return $out;
}

// Exponential recency decay: half-life in hours.
def recencyWeight($ageHours, $halfLife) {
    if ($ageHours == null || $ageHours < 0) { return 0.0; }
    return pow(0.5, $ageHours / $halfLife);
}
