// ════════════════════════════════════════════════════════════════════════
//  pillow/morph.b — ImageMorph: binary morphology as a 512-entry look-up
//  table over the 3x3 neighbourhood.
//
//  The operation is defined ENTIRELY by that table, indexed by the nine
//  neighbours as nine bits (top-left is bit 0, the centre is bit 4,
//  bottom-right is bit 8). `LutBuilder` compiles a pattern language into
//  it and `MorphOp.apply` applies it:
//
//      $op = morph.MorphOp(null, "dilation4", null);
//      $r = morph.apply($op, $im);        // [changed count, image]
//
//      $op = morph.MorphOp(null, null, ["4:(... .1. .0.)->0"]);
//
//  A pattern is `<option>:(<nine cells>)-><result>` where a cell is 0, 1 or
//  `.` (either), the option is `4` (add the four rotations), `M` (add the
//  mirror) or `N` (add the negation, with the result flipped), and LATER
//  patterns overwrite earlier ones at the same index.
//
//  Applying it is nine shifted comparisons, a weighted sum and one nd_take
//  of the table -- no pixel loop, the same shape for every pattern anyone
//  writes. The border is left at zero, as _imagingmorph.c leaves it, and
//  the returned count is the number of pixels that CHANGED.
//
//  docs/pillow-architecture.md §7.6.
// ════════════════════════════════════════════════════════════════════════

include "./pxbuf.b" as pxbuf;
include "./image.b" as image;

$LUT_SIZE = 512;

// The two permutations, as Pillow spells them: a rotation by 90 degrees and
// a mirror, each as an index map over the nine cells.
$ROTATION_MATRIX = [6, 3, 0,
                    7, 4, 1,
                    8, 5, 2];
$MIRROR_MATRIX = [2, 1, 0,
                  5, 4, 3,
                  8, 7, 6];

$KNOWN = {
    "corner": ["1:(... ... ...)->0", "4:(00. 01. ...)->1"],
    "dilation4": ["4:(... .0. .1.)->1"],
    "dilation8": ["4:(... .0. .1.)->1", "4:(... .0. ..1)->1"],
    "erosion4": ["4:(... .1. .0.)->0"],
    "erosion8": ["4:(... .1. .0.)->0", "4:(... .1. ..0)->0"],
    "edge": ["1:(... ... ...)->0", "4:(.0. .1. ...)->1", "4:(01. .1. ...)->1"]
};

def known_patterns() {
    $out = [];
    for $k, $v in $KNOWN.items() { push($out, $k); }
    return sort($out);
}

def _permute($pattern, $perm) {
    $out = [];
    $i = 0;
    while ($i < 9) { push($out, $pattern[$perm[$i]]); $i = $i + 1; }
    return join($out, "");
}

// _patternPermute($pattern, $options, $result) -> every [pattern, result]
// the options imply. Pillow rotates the LAST pattern four times (so the
// fourth rotation is the original again), mirrors or negates the whole set
// so far, and keeps them all.
def _patternPermute($pattern, $options, $result) {
    $pats = [[$pattern, $result]];
    if (contains($options, "4")) {
        $i = 0;
        while ($i < 4) {
            $last = $pats[len($pats) - 1];
            push($pats, [_permute($last[0], $ROTATION_MATRIX), $result]);
            $i = $i + 1;
        }
    }
    if (contains($options, "M")) {
        $n = len($pats);
        $i = 0;
        while ($i < $n) {
            push($pats, [_permute($pats[$i][0], $MIRROR_MATRIX), $pats[$i][1]]);
            $i = $i + 1;
        }
    }
    if (contains($options, "N")) {
        $n = len($pats);
        $i = 0;
        while ($i < $n) {
            $p = $pats[$i][0];
            $flip = [];
            $k = 0;
            while ($k < len($p)) {
                if ($p[$k] == "0") { push($flip, "1"); }
                else if ($p[$k] == "1") { push($flip, "0"); }
                else { push($flip, $p[$k]); }
                $k = $k + 1;
            }
            push($pats, [join($flip, ""), 1 - $pats[$i][1]]);
            $i = $i + 1;
        }
    }
    return $pats;
}

// _parse($p) -> [options, pattern, result].
//
// Pillow's regex is (\w):?\s*\((.+?)\)\s*->\s*(\d) and re.search takes the
// LEFTMOST start that satisfies the whole thing -- so in "4M:(...)->1" the
// options are "M" alone, not "4M". That is reproduced rather than
// corrected, because a pattern file written against Pillow depends on it.
def _parse($p) {
    $open = indexOf($p, "(");
    $close = indexOf($p, ")");
    $arrow = indexOf($p, "->");
    if ($open < 1 || $close < $open || $arrow < $close) {
        throw "Syntax error in pattern \"" + str($p) + "\"";
    }
    // the single word character immediately before the ':' or '('
    $i = $open - 1;
    while ($i >= 0 && ($p[$i] == ":" || $p[$i] == " " || $p[$i] == "\t")) { $i = $i - 1; }
    if ($i < 0) { throw "Syntax error in pattern \"" + str($p) + "\""; }
    $options = $p[$i];
    $raw = substr($p, $open + 1, $close - $open - 1);
    $cells = [];
    $k = 0;
    while ($k < len($raw)) {
        $c = $raw[$k];
        if ($c != " " && $c != "\n" && $c != "\t") {
            if ($c != "0" && $c != "1" && $c != ".") {
                throw "Syntax error in pattern \"" + str($p) + "\": '" + $c +
                      "' is not 0, 1 or .";
            }
            push($cells, $c);
        }
        $k = $k + 1;
    }
    if (len($cells) != 9) {
        throw "Syntax error in pattern \"" + str($p) + "\": " + str(len($cells)) +
              " cells, not 9";
    }
    $rest = trim(substr($p, $arrow + 2, len($p) - $arrow - 2));
    if ($rest != "0" && $rest != "1") {
        throw "Syntax error in pattern \"" + str($p) + "\": the result is 0 or 1";
    }
    return [$options, join($cells, ""), num($rest)];
}

// build_lut($patterns) -> the 512-entry table.
//
// The default table is the identity -- entry i is bit 4 of i, the centre
// pixel -- so a pattern set only changes what it names.
def build_lut($patterns) {
    $lut = [];
    $i = 0;
    while ($i < $LUT_SIZE) {
        if (band($i, 16) > 0) { push($lut, 1); } else { push($lut, 0); }
        $i = $i + 1;
    }
    $all = [];
    $k = 0;
    while ($k < len($patterns)) {
        $parsed = _parse($patterns[$k]);
        $perms = _patternPermute($parsed[1], $parsed[0], $parsed[2]);
        $j = 0;
        while ($j < len($perms)) { push($all, $perms[$j]); $j = $j + 1; }
        $k = $k + 1;
    }
    $i = 0;
    while ($i < $LUT_SIZE) {
        // The nine bits of i as a string, bit 0 first, which is the order
        // the pattern's cells are written in.
        $bits = [];
        $b = 0;
        while ($b < 9) {
            if (band($i, pow(2, $b)) > 0) { push($bits, "1"); } else { push($bits, "0"); }
            $b = $b + 1;
        }
        $j = 0;
        while ($j < len($all)) {
            $pat = $all[$j][0];
            $ok = true;
            $c = 0;
            while ($c < 9) {
                if ($pat[$c] != "." && $pat[$c] != $bits[$c]) { $ok = false; break; }
                $c = $c + 1;
            }
            // No break: the LAST matching pattern wins, as in Pillow.
            if ($ok) { $lut[$i] = $all[$j][1]; }
            $j = $j + 1;
        }
        $i = $i + 1;
    }
    return $lut;
}

// LutBuilder($patterns, $op_name) -> the builder, Pillow's class as a dict.
def LutBuilder($patterns, $op_name) {
    if ($op_name != null) {
        $k = $KNOWN[$op_name];
        if ($k == null) { throw "Unknown pattern " + str($op_name) + "!"; }
        return {"patterns": $k, "lut": null};
    }
    if ($patterns == null) { return {"patterns": [], "lut": null}; }
    return {"patterns": $patterns, "lut": null};
}

def add_patterns($lb, $patterns) {
    $i = 0;
    while ($i < len($patterns)) { push($lb.patterns, $patterns[$i]); $i = $i + 1; }
    return $lb;
}
def build($lb) {
    $lb.lut = build_lut($lb.patterns);
    return $lb.lut;
}
def get_lut($lb) { return $lb.lut; }

// MorphOp($lut, $op_name, $patterns) -> the operator.
def MorphOp($lut, $op_name, $patterns) {
    if ($patterns == null && $op_name == null) {
        return {"lut": $lut};
    }
    return {"lut": build_lut(LutBuilder($patterns, $op_name).patterns)};
}

def set_lut($op, $lut) { $op.lut = $lut; return $op; }

def save_lut($op, $path) {
    if ($op.lut == null) { throw "No operator loaded"; }
    $bytes = [];
    $i = 0;
    while ($i < $LUT_SIZE) { push($bytes, $op.lut[$i]); $i = $i + 1; }
    writefile($path, frombytes($bytes));
    return true;
}

def load_lut($op, $path) {
    $data = readfile($path);
    if (len($data) != $LUT_SIZE) { throw "Wrong size operator file!"; }
    $op.lut = bytes($data);
    return $op.lut;
}

// _index($px) -> the 3x3 neighbourhood index of every interior pixel, as
// an (h-2, w-2) array. Nine shifted slices and a weighted sum.
def _index($px) {
    $d = pxbuf.dims($px);
    $h = $d[0];
    $w = $d[1];
    $on = nd_reshape(nd_where(nd_not_equal($px, 0), 1, 0), [$h, $w]);
    $acc = nd_zeros([$h - 2, $w - 2], "i64");
    $b = 0;
    $dy = 0 - 1;
    while ($dy <= 1) {
        $dx = 0 - 1;
        while ($dx <= 1) {
            $tap = nd_slice($on, [[1 + $dy, $h - 1 + $dy], [1 + $dx, $w - 1 + $dx]]);
            $acc = nd_add($acc, nd_multiply($tap, pow(2, $b)));
            $b = $b + 1;
            $dx = $dx + 1;
        }
        $dy = $dy + 1;
    }
    return $acc;
}

// apply($op, $im) -> [the number of pixels that changed, the new image].
def apply($op, $im) {
    if ($op.lut == null) { throw "No operator loaded"; }
    if ($im.mode != "1" && $im.mode != "L") { throw "Image mode must be 1 or L"; }
    $d = pxbuf.dims($im.px);
    $out = pxbuf.make($d[0], $d[1], 1, "i64");
    if ($d[0] < 3 || $d[1] < 3) {
        return [0, image.wrap($im.mode, $out, null, {}, null)];
    }
    $idx = _index($im.px);
    $hit = nd_take(nd($op.lut, "i64"), nd_ravel($idx));
    $on = nd_multiply(nd_reshape($hit, [$d[0] - 2, $d[1] - 2]), 255);
    pxbuf.copy_into(nd_slice($out, [[1, $d[0] - 1], [1, $d[1] - 1], null]),
                    nd_reshape($on, [$d[0] - 2, $d[1] - 2, 1]));
    // The count is how many pixels CHANGED, over the INTERIOR only --
    // _imagingmorph.c's loops never reach the border, so a non-zero border
    // pixel that the all-zero output leaves behind is not a change. Counting
    // the border too inflates every operator by the same amount, which is
    // exactly how the off-by-21 was spotted.
    $inIn = nd_where(nd_not_equal(
        nd_slice($im.px, [[1, $d[0] - 1], [1, $d[1] - 1], null]), 0), 255, 0);
    $outIn = nd_slice($out, [[1, $d[0] - 1], [1, $d[1] - 1], null]);
    $count = pxbuf.scalar(nd_sum(nd_where(nd_not_equal($outIn, $inIn), 1, 0)));
    return [$count, image.wrap($im.mode, $out, null, {}, null)];
}

// match($op, $im) -> the [x, y] of every interior pixel whose neighbourhood
// the table marks, which is what MorphOp.match returns.
def match($op, $im) {
    if ($op.lut == null) { throw "No operator loaded"; }
    if ($im.mode != "1" && $im.mode != "L") { throw "Image mode must be 1 or L"; }
    $d = pxbuf.dims($im.px);
    if ($d[0] < 3 || $d[1] < 3) { return []; }
    $idx = _index($im.px);
    $hit = nd_reshape(nd_take(nd($op.lut, "i64"), nd_ravel($idx)),
                      [$d[0] - 2, $d[1] - 2]);
    // nd_nonzero returns FLAT indices, so the row and column come back by
    // division and modulo over the interior's width.
    $iw = $d[1] - 2;
    $flat = nd_to_list(nd_nonzero($hit));
    $out = [];
    $i = 0;
    while ($i < len($flat)) {
        $k = $flat[$i];
        push($out, [($k % $iw) + 1, floor($k / $iw) + 1]);
        $i = $i + 1;
    }
    return $out;
}

// get_on_pixels($im) -> the [x, y] of every non-zero pixel, border included.
def get_on_pixels($im) {
    if ($im.mode != "1" && $im.mode != "L") { throw "Image mode must be 1 or L"; }
    $d = pxbuf.dims($im.px);
    $flat = nd_to_list(nd_nonzero(nd_reshape($im.px, [$d[0], $d[1]])));
    $out = [];
    $i = 0;
    while ($i < len($flat)) {
        $k = $flat[$i];
        push($out, [$k % $d[1], floor($k / $d[1])]);
        $i = $i + 1;
    }
    return $out;
}
