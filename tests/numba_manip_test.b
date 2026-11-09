// ════════════════════════════════════════════════════════════════════════
//  numba_manip_test.b — phases A6-A9: concat/stack/split, pad,
//  repeat/tile, im2col/col2im.
//
//  Two of these five families are ADJOINTS of others, and that is how the
//  hard part is tested rather than by example:
//
//      split  is concat's adjoint:  concat(split(x)) == x, exactly
//      col2im is im2col's adjoint:  <im2col(x), y> == <x, col2im(y)>
//
//  The second identity is the one that matters. im2col/col2im is where a
//  convolution's gradient lives, and a col2im that overwrites instead of
//  accumulating produces a plausible-looking gradient that is wrong wherever
//  two kernel positions overlap -- which, at stride 1, is almost everywhere.
//  The inner-product identity catches that; an example-based test on a
//  stride-equals-kernel case does not, because then nothing overlaps.
//
//  Every numeric family is also checked against an independent pure-Bantu
//  reference loop, written with nd_get/nd_set only.
//
//  Run:  bantu run tests/numba_manip_test.b
// ════════════════════════════════════════════════════════════════════════

$R = {"pass": 0, "fail": 0};
def ok($cond, $name) {
    if ($cond) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $name); }
}
def same($x, $y, $name) {
    if (nd_shape($x) != nd_shape($y)) {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name + "  shape " + str(nd_shape($x)) + " vs " + str(nd_shape($y)));
        return null;
    }
    ok(nd_array_equal($x, $y), $name);
    return null;
}
def eqL($got, $want, $name) {
    if (str($got) == str($want)) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name);
        print("          got:  " + str($got));
        print("          want: " + str($want));
    }
}
def raises($f, $name) {
    try { $f(); } catch ($e) { ok(true, $name); return null; }
    ok(false, $name + "  (no error raised)");
    return null;
}
def sc($x) { return nd_get(nd_reshape($x, [1]), [0]); }

print("========================================");
print("  A6 — concat / stack / split");
print("========================================");

$a = nd([[1.0, 2.0], [3.0, 4.0]], "f64");
$b = nd([[5.0, 6.0]], "f64");
eqL(nd_to_list(nd_concat([$a, $b], 0)), [[1.0,2.0],[3.0,4.0],[5.0,6.0]], "concat along axis 0");
eqL(nd_to_list(nd_concat([$a, nd([[9.0],[8.0]], "f64")], 1)), [[1.0,2.0,9.0],[3.0,4.0,8.0]],
    "concat along axis 1");
eqL(nd_shape(nd_concat([$a], 0)), [2, 2], "a one-element list concatenates to itself");
eqL(nd_to_list(nd_concat([$a, $a, $a], 0))[4], [1.0, 2.0], "three-way concat lands in order");
// axis defaults to 0, which is what every caller of cat() means.
same(nd_concat([$a, $b], null), nd_concat([$a, $b], 0), "concat's axis defaults to 0");
eqL(nd_shape(nd_concat([$a, $b], 0 - 2)), [3, 2], "a negative axis counts from the end");

print("");
print("== concat: dtype promotion, and a reference loop ==");
ok(nd_dtype(nd_concat([nd([1, 2], "i64"), nd([3.0], "f64")], 0)) == "f64",
   "i64 + f64 promotes to f64");
ok(nd_dtype(nd_concat([nd([true], "bool"), nd([2], "i64")], 0)) == "i64",
   "bool + i64 promotes to i64");
ok(nd_dtype(nd_concat([nd([true], "bool"), nd([false], "bool")], 0)) == "bool",
   "bool + bool stays bool");
// An independent reference: build the joined array with nd_set, one cell at a
// time, so a stride mistake in the kernel has nothing to mirror.
def refConcat0($xs) {
    $rows = 0; $cols = nd_shape($xs[0])[1];
    $i = 0;
    while ($i < len($xs)) { $rows = $rows + nd_shape($xs[$i])[0]; $i = $i + 1; }
    $out = nd_zeros([$rows, $cols], "f64");
    $at = 0; $i = 0;
    while ($i < len($xs)) {
        $r = 0;
        while ($r < nd_shape($xs[$i])[0]) {
            $c = 0;
            while ($c < $cols) {
                nd_set($out, [$at, $c], nd_get($xs[$i], [$r, $c]));
                $c = $c + 1;
            }
            $r = $r + 1; $at = $at + 1;
        }
        $i = $i + 1;
    }
    return $out;
}
nd_seed(424242);
$p = nd_random_normal([3, 4]); $q = nd_random_normal([2, 4]); $r2 = nd_random_normal([5, 4]);
same(nd_concat([$p, $q, $r2], 0), refConcat0([$p, $q, $r2]), "concat matches the reference loop");

print("");
print("== concat: a strided input is read through its strides ==");
$big = nd_reshape(nd_astype(nd_arange(0.0, 24.0, null), "f64"), [6, 4]);
$odd = nd_slice($big, [[1, 6, 2]]);                  // rows 1,3,5
eqL(nd_to_list($odd)[0], [4.0,5.0,6.0,7.0], "the strided view is what we think it is");
eqL(nd_to_list(nd_concat([$odd, nd_zeros([1, 4], "f64")], 0))[0], [4.0,5.0,6.0,7.0],
    "concat of a strided view keeps its values");
// A transposed input: C-order and buffer order disagree completely.
$t = nd_transpose($a, null);
eqL(nd_to_list(nd_concat([$t, $t], 0)), [[1.0,3.0],[2.0,4.0],[1.0,3.0],[2.0,4.0]],
    "concat of a transposed view reads in C order, not buffer order");

print("");
print("== concat: every failure mode raises, by name ==");
raises(def() { return nd_concat([], 0); },                 "an empty list raises rather than guessing a rank");
raises(def() { return nd_concat($a, 0); },                 "a bare array instead of a list raises");
raises(def() { return nd_concat([$a, nd_zeros([2,3],"f64")], 0); },
       "a mismatch on a non-join axis raises");
raises(def() { return nd_concat([$a, nd_zeros([2],"f64")], 0); }, "a rank mismatch raises");
raises(def() { return nd_concat([nd(1.0,"f64"), nd(2.0,"f64")], 0); },
       "concatenating 0-d arrays raises (there is no axis)");
raises(def() { return nd_concat([$a, $b], 5); },           "an out-of-range axis raises");
try { nd_concat([$a, nd_zeros([2, 3], "f64")], 0); } catch ($e) {
    // shapeStr (C++) writes "[2,2]"; Bantu's str() on a list writes "[2, 2]".
    // The message comes from C++, so the assertion must use C++'s spelling.
    ok(contains($e.message, "[2,2]") && contains($e.message, "[2,3]") && contains($e.message, "axis 1"),
       "the mismatch error prints both shapes and names the offending axis");
}

print("");
print("== stack adds a NEW axis ==");
eqL(nd_shape(nd_stack([$a, $a], 0)), [2, 2, 2], "stack(axis=0)");
eqL(nd_shape(nd_stack([$a, $a], 1)), [2, 2, 2], "stack(axis=1)");
eqL(nd_shape(nd_stack([$a, $a], 2)), [2, 2, 2], "stack(axis=2) -- the last legal position");
eqL(nd_to_list(nd_stack([nd([1.0,2.0],"f64"), nd([3.0,4.0],"f64")], 0)), [[1.0,2.0],[3.0,4.0]],
    "stacking rows gives a matrix");
eqL(nd_to_list(nd_stack([nd([1.0,2.0],"f64"), nd([3.0,4.0],"f64")], 1)), [[1.0,3.0],[2.0,4.0]],
    "stacking on axis 1 interleaves");
eqL(nd_to_list(nd_stack([nd(5.0,"f64"), nd(7.0,"f64")], 0)), [5.0, 7.0], "0-d arrays stack into a vector");
eqL(nd_shape(nd_stack([$a, $a], 0 - 1)), [2, 2, 2], "a negative stack axis counts from the end");
raises(def() { return nd_stack([$a, $b], 0); }, "stack refuses a differing axis (that is concat's job)");
raises(def() { return nd_stack([$a, $a], 3); }, "an out-of-range stack axis raises");
try { nd_stack([$a, $b], 0); } catch ($e) {
    ok(contains($e.message, "nd_concat"), "and the error says which function does allow it");
}

print("");
print("== split is concat's adjoint: the round trip is EXACT ==");
$six = nd_reshape(nd_astype(nd_arange(0.0, 24.0, null), "f64"), [6, 4]);
$ps = nd_split($six, 3, 0);
ok(len($ps) == 3, "split into 3 gives 3 parts");
eqL(nd_shape($ps[0]), [2, 4], "each part has the right shape");
same(nd_concat($ps, 0), $six, "concat(split(x, 3)) == x, exactly");
$pu = nd_split($six, [1, 2, 3], 0);
eqL(nd_shape($pu[2]), [3, 4], "an uneven split by explicit sizes");
same(nd_concat($pu, 0), $six, "concat(split(x, [1,2,3])) == x, exactly");
$pa1 = nd_split($six, 2, 1);
eqL(nd_shape($pa1[0]), [6, 2], "split along axis 1");
same(nd_concat($pa1, 1), $six, "and its round trip is exact too");
// The parts are VIEWS, which is the point: splitting a batch must not copy it.
ok(nd_is_view($ps[0]), "the parts are views, not copies");
ok(nd_shares_memory($ps[0], $six), "and they share the original's memory");
nd_set($ps[1], [0, 0], 999.0);
ok(nd_get($six, [2, 0]) == 999.0, "writing through a part writes through to the base");
nd_set($ps[1], [0, 0], 8.0);
raises(def() { return nd_split($six, 4, 0); },      "a count that does not divide exactly raises");
raises(def() { return nd_split($six, [1, 2], 0); }, "sizes that do not sum to the axis raise");
raises(def() { return nd_split($six, 0, 0); },      "splitting into 0 sections raises");
raises(def() { return nd_split(nd(1.0, "f64"), 1, 0); }, "splitting a 0-d array raises");
try { nd_split($six, 4, 0); } catch ($e) {
    ok(contains($e.message, "list of sizes"), "and the error says how to split unevenly");
}

print("");
print("== A6 stress: 1000-way concat, and a length-1 axis ==");
$many = [];
$i = 0;
while ($i < 1000) { push($many, nd_full([1, 3], $i * 1.0)); $i = $i + 1; }
$joined = nd_concat($many, 0);
eqL(nd_shape($joined), [1000, 3], "1000 arrays concatenate");
ok(sc(nd_sum($joined, null, null)) == 3.0 * (999.0 * 1000.0 / 2.0),
   "and every one of them landed exactly once");
same(nd_concat(nd_split($joined, 1000, 0), 0), $joined, "a 1000-way split round-trips exactly");
eqL(nd_shape(nd_concat([nd_zeros([1, 4], "f64"), nd_zeros([1, 4], "f64")], 0)), [2, 4],
    "a length-1 join axis works");
eqL(nd_shape(nd_concat([nd_zeros([0, 4], "f64"), nd_zeros([3, 4], "f64")], 0)), [3, 4],
    "a length-0 input contributes nothing and breaks nothing");

print("");
print("========================================");
print("  A7 — pad");
print("========================================");

eqL(nd_to_list(nd_pad($a, [[1, 0], [0, 1]], 0.0)), [[0.0,0.0,0.0],[1.0,2.0,0.0],[3.0,4.0,0.0]],
    "asymmetric pads, one pair per axis");
eqL(nd_shape(nd_pad($a, 2, 0.0)), [6, 6], "a single number pads both sides of every axis");
eqL(nd_shape(nd_pad($a, [1, 3], 0.0)), [6, 6], "one [before, after] pair applies to every axis");
same(nd_pad($a, 0, 0.0), $a, "a zero pad is the exact identity");
same(nd_pad($a, [[0, 0], [0, 0]], 7.0), $a, "...even with a non-zero fill, since nothing is added");
eqL(nd_to_list(nd_pad(nd([1.0], "f64"), [[2, 2]], 9.0)), [9.0,9.0,1.0,9.0,9.0],
    "a non-zero fill reaches every padded cell");
ok(nd_dtype(nd_pad(nd([1, 2], "i64"), 1, 0.0)) == "i64", "pad preserves the dtype");
eqL(nd_to_list(nd_pad(nd([true], "bool"), [[1, 1]], 0.0)), [false, true, false],
    "a bool array pads with false");

// Reference loop: place the original by hand, inside an array of fill.
def refPad2($x, $bt, $bl, $at, $al, $fill) {
    $h = nd_shape($x)[0]; $w = nd_shape($x)[1];
    $out = nd_full([$h + $bt + $at, $w + $bl + $al], $fill);
    $r = 0;
    while ($r < $h) {
        $c = 0;
        while ($c < $w) {
            nd_set($out, [$r + $bt, $c + $bl], nd_get($x, [$r, $c]));
            $c = $c + 1;
        }
        $r = $r + 1;
    }
    return $out;
}
nd_seed(77);
$pr = nd_random_normal([3, 5]);
same(nd_pad($pr, [[2, 1], [0, 3]], 0.0), refPad2($pr, 2, 0, 1, 3, 0.0),
     "pad matches the reference loop");
same(nd_pad($pr, [[1, 1], [1, 1]], 0 - 2.5), refPad2($pr, 1, 1, 1, 1, 0 - 2.5),
     "...with a negative fill value too");
// A strided source.
same(nd_pad($odd, [[1, 1], [0, 0]], 0.0),
     nd_pad(nd_copy($odd, null), [[1, 1], [0, 0]], 0.0),
     "padding a strided view gives the same answer as padding its copy");

print("");
print("== pad: a pad larger than the array, and the refusals ==");
$tiny = nd([[1.0]], "f64");
eqL(nd_shape(nd_pad($tiny, 50, 0.0)), [101, 101], "a pad 50x the array is fine");
ok(sc(nd_sum(nd_pad($tiny, 50, 0.0), null, null)) == 1.0, "and adds only zeros");
raises(def() { return nd_pad($a, 0 - 1, 0.0); },           "a negative pad raises -- it is not a crop");
raises(def() { return nd_pad($a, [[0 - 1, 0], [0, 0]], 0.0); }, "...including one negative entry among many");
raises(def() { return nd_pad($a, [[1, 1]], 0.0); },        "the wrong number of pairs raises");
raises(def() { return nd_pad($a, [[1], [2]], 0.0); },      "an entry that is not a pair raises");
raises(def() { return nd_pad(nd(1.0, "f64"), 1, 0.0); },   "padding a 0-d array raises");
raises(def() { return nd_pad($a, "two", 0.0); },           "a string pad raises");
try { nd_pad($a, 0 - 1, 0.0); } catch ($e) {
    ok(contains($e.message, "nd_slice"), "and the error names the function that does crop");
}

print("");
print("========================================");
print("  A8 — repeat vs tile");
print("========================================");

// The two are routinely confused, so every assertion here states both.
$v = nd([1.0, 2.0, 3.0], "f64");
eqL(nd_to_list(nd_repeat($v, 2, null)), [1.0,1.0,2.0,2.0,3.0,3.0], "repeat duplicates each ELEMENT");
eqL(nd_to_list(nd_tile($v, 2)),         [1.0,2.0,3.0,1.0,2.0,3.0], "tile duplicates the whole BLOCK");
ok(sc(nd_sum(nd_repeat($v, 3, null), null, null)) == sc(nd_sum(nd_tile($v, 3), null, null)),
   "they agree on the total, and on nothing else");

eqL(nd_to_list(nd_repeat($a, 2, 0)), [[1.0,2.0],[1.0,2.0],[3.0,4.0],[3.0,4.0]], "repeat along axis 0");
eqL(nd_to_list(nd_repeat($a, 2, 1)), [[1.0,1.0,2.0,2.0],[3.0,3.0,4.0,4.0]], "repeat along axis 1");
eqL(nd_to_list(nd_tile($a, [2, 1])), [[1.0,2.0],[3.0,4.0],[1.0,2.0],[3.0,4.0]], "tile down");
eqL(nd_to_list(nd_tile($a, [1, 2])), [[1.0,2.0,1.0,2.0],[3.0,4.0,3.0,4.0]], "tile across");
eqL(nd_shape(nd_tile($a, [2, 3])), [4, 6], "tile in both directions");
// repeat with no axis flattens first, which is NumPy's rule and a trap.
eqL(nd_to_list(nd_repeat($a, 2, null)), [1.0,1.0,2.0,2.0,3.0,3.0,4.0,4.0],
    "repeat with no axis FLATTENS first (NumPy's rule)");

print("");
print("== repeat/tile: the degenerate factors ==");
same(nd_repeat($v, 1, null), $v, "repeat by 1 is the identity");
same(nd_tile($v, 1), $v, "tile by 1 is the identity");
eqL(nd_shape(nd_repeat($v, 0, null)), [0], "repeat by 0 gives an empty array, not an error");
eqL(nd_shape(nd_tile($v, 0)), [0], "tile by 0 gives an empty array");
eqL(nd_shape(nd_repeat($a, 0, 1)), [2, 0], "repeat by 0 along an axis collapses that axis only");
// reps longer than the rank prepends axes, as NumPy does.
eqL(nd_shape(nd_tile($v, [2, 2])), [2, 6], "a reps list longer than the rank prepends an axis");
eqL(nd_shape(nd_tile($a, [2])), [2, 4], "a reps list shorter than the rank applies to the last axes");
ok(nd_dtype(nd_repeat(nd([1, 2], "i64"), 2, null)) == "i64", "repeat preserves the dtype");
ok(nd_dtype(nd_tile(nd([true], "bool"), 2)) == "bool", "tile preserves the dtype");
raises(def() { return nd_repeat($v, 0 - 1, null); }, "a negative repeat raises");
raises(def() { return nd_tile($v, 0 - 1); },         "a negative rep raises");
raises(def() { return nd_repeat($v, 2, 5); },        "an out-of-range repeat axis raises");
raises(def() { return nd_tile($v, "two"); },         "a string reps raises");

print("");
print("== A8: strided sources, and a reference loop ==");
same(nd_repeat($odd, 2, 0), nd_repeat(nd_copy($odd, null), 2, 0),
     "repeat of a strided view matches repeat of its copy");
same(nd_tile($t, [2, 2]), nd_tile(nd_copy($t, null), [2, 2]),
     "tile of a transposed view matches tile of its copy");
def refRepeat0($x, $r) {
    $h = nd_shape($x)[0]; $w = nd_shape($x)[1];
    $out = nd_zeros([$h * $r, $w], "f64");
    $i = 0;
    while ($i < $h) {
        $k = 0;
        while ($k < $r) {
            $c = 0;
            while ($c < $w) { nd_set($out, [$i * $r + $k, $c], nd_get($x, [$i, $c])); $c = $c + 1; }
            $k = $k + 1;
        }
        $i = $i + 1;
    }
    return $out;
}
same(nd_repeat($pr, 3, 0), refRepeat0($pr, 3), "repeat matches the reference loop");

print("");
print("== A8 stress: a large expansion, against the allocation ceiling ==");
$grow = nd_tile(nd_ones([1000], "f64"), 1000);
eqL(nd_shape($grow), [1000000], "a 1000x expansion of 1000 elements allocates");
ok(sc(nd_sum($grow, null, null)) == 1000000.0, "and every cell is filled");
// The ceiling must refuse, not OOM. nd_max_bytes is the hard limit.
raises(def() { return nd_tile(nd_ones([1000000], "f64"), 100000000); },
       "an expansion past the allocation ceiling RAISES rather than OOM-killing");
raises(def() { return nd_repeat(nd_ones([1000000], "f64"), 100000000, null); },
       "...and so does repeat");

print("");
print("========================================");
print("  A9 — im2col / col2im");
print("========================================");

// Hand-computed: a 4x4 image, 2x2 kernel, stride 2 -> four non-overlapping
// patches, which is the one case where the answer can be read off by eye.
$img = nd_reshape(nd_astype(nd_arange(0.0, 16.0, null), "f64"), [1, 1, 4, 4]);
$c22 = nd_im2col($img, 2, 2, 2, 2, 0, 0, null, null);
eqL(nd_shape($c22), [4, 4], "im2col of 4x4 with a 2x2 stride-2 kernel is [4 patches, 4 taps]");
eqL(nd_to_list($c22), [[0.0,1.0,4.0,5.0],[2.0,3.0,6.0,7.0],[8.0,9.0,12.0,13.0],[10.0,11.0,14.0,15.0]],
    "and each row is one patch, read in (c, kh, kw) order");
eqL(nd_to_list(nd_conv_outsize(4, 4, 2, 2, 2, 2, null, null, null, null)), [2, 2],
    "nd_conv_outsize agrees about the output grid");

// 1x1 kernels: im2col is then a pure reshape, so the identity is checkable.
$x11 = nd_random_normal([2, 3, 4, 5]);
$c11 = nd_im2col($x11, 1, 1, 1, 1, 0, 0, null, null);
eqL(nd_shape($c11), [40, 3], "a 1x1 kernel gives [N*H*W, C]");
ok(nd_allclose($c11, nd_reshape(nd_ascontiguous(nd_transpose($x11, [0, 2, 3, 1])), [40, 3]), 1e-15, 1e-15),
   "and it is exactly the NHWC reshape of the input");

print("");
print("== im2col: padding, stride and dilation ==");
eqL(nd_shape(nd_im2col($img, 3, 3, 1, 1, 1, 1, null, null)), [16, 9],
    "3x3 stride 1 pad 1 keeps the grid at 4x4");
// The corner patch of a pad-1 convolution has five zeros in it.
$cpad = nd_im2col($img, 3, 3, 1, 1, 1, 1, null, null);
eqL(nd_to_list($cpad)[0], [0.0,0.0,0.0,0.0,0.0,1.0,0.0,4.0,5.0],
    "the top-left patch is the padded one, zeros and all");
eqL(nd_shape(nd_im2col($img, 2, 2, 1, 1, 0, 0, 2, 2)), [4, 4],
    "a dilated 2x2 kernel spans 3x3 and so fits 2x2 times in 4x4");
$cdil = nd_im2col($img, 2, 2, 1, 1, 0, 0, 2, 2);
eqL(nd_to_list($cdil)[0], [0.0, 2.0, 8.0, 10.0], "and its first patch skips every other pixel");
eqL(nd_shape(nd_im2col(nd_random_normal([3, 2, 7, 9]), 3, 3, 2, 2, 1, 1, null, null)),
    [3 * 4 * 5, 2 * 9], "a non-square input with stride 2 and pad 1");
// Asymmetric everything at once.
eqL(nd_to_list(nd_conv_outsize(7, 9, 3, 2, 2, 3, 1, 0, null, null)), [4, 3],
    "outsize handles differing kernel, stride and pad per axis");
eqL(nd_shape(nd_im2col(nd_random_normal([1, 1, 7, 9]), 3, 2, 2, 3, 1, 0, null, null)),
    [12, 6], "and im2col agrees with it");

print("");
print("== im2col: the refusals ==");
raises(def() { return nd_im2col(nd_random_normal([4, 4]), 2, 2, 1, 1, 0, 0, null, null); },
       "a non-4-d input raises");
raises(def() { return nd_im2col($img, 5, 5, 1, 1, 0, 0, null, null); },
       "a kernel larger than the padded input raises");
raises(def() { return nd_im2col($img, 3, 3, 1, 1, 0, 0, 3, 3); },
       "...including when dilation is what makes it too large");
raises(def() { return nd_im2col($img, 2, 2, 0, 1, 0, 0, null, null); },
       "a stride of 0 raises");
raises(def() { return nd_im2col($img, 2, 2, 1, 1, 0, 0, 0, 1); },
       "a dilation of 0 raises");
raises(def() { return nd_im2col($img, 0, 2, 1, 1, 0, 0, null, null); },
       "a kernel dimension of 0 raises");
raises(def() { return nd_im2col($img, 2, 2, 1, 1, 0 - 1, 0, null, null); },
       "a negative pad raises");
try { nd_im2col($img, 5, 5, 1, 1, 0, 0, null, null); } catch ($e) {
    ok(contains($e.message, "5x5") && contains($e.message, "4x4"),
       "and the error prints the kernel and the padded input it would not fit in");
}

print("");
print("== THE ADJOINT IDENTITY: <im2col(x), y> == <x, col2im(y)> ==");
// This is the assertion the whole family rests on. col2im accumulates, so it
// is im2col's transpose; if it did not, the two inner products would differ
// wherever two kernel positions read the same pixel -- which at stride 1 is
// nearly every pixel. An example-based test at stride == kernel never overlaps
// and so cannot see the difference.
def adj($shape, $kh, $kw, $sh, $sw, $ph, $pw, $dh, $dw, $name) {
    nd_seed(9090);
    $x = nd_random_normal($shape);
    $cols = nd_im2col($x, $kh, $kw, $sh, $sw, $ph, $pw, $dh, $dw);
    $y = nd_random_normal(nd_shape($cols));
    $lhs = sc(nd_sum(nd_multiply($cols, $y, null), null, null));
    $back = nd_col2im($y, $shape, $kh, $kw, $sh, $sw, $ph, $pw, $dh, $dw);
    $rhs = sc(nd_sum(nd_multiply($x, $back, null), null, null));
    $den = 1.0;
    $al = $lhs; if ($al < 0.0) { $al = 0.0 - $al; }
    if ($al > $den) { $den = $al; }
    $d = $lhs - $rhs; if ($d < 0.0) { $d = 0.0 - $d; }
    ok($d / $den < 1e-12, $name + "  (lhs " + str($lhs) + " rhs " + str($rhs) + ")");
    return null;
}
adj([1,1,4,4], 2,2, 2,2, 0,0, null,null, "2x2 stride 2 -- no overlap");
adj([1,1,4,4], 2,2, 1,1, 0,0, null,null, "2x2 stride 1 -- OVERLAPPING, the case that matters");
adj([1,1,5,5], 3,3, 1,1, 1,1, null,null, "3x3 stride 1 pad 1 -- the standard conv layer");
adj([2,3,6,6], 3,3, 1,1, 1,1, null,null, "batched and multi-channel");
adj([2,3,7,9], 3,3, 2,2, 1,1, null,null, "stride 2, non-square, padded");
adj([1,2,6,6], 2,2, 1,1, 0,0, 2,2,       "dilated");
adj([1,1,5,5], 1,1, 1,1, 0,0, null,null, "1x1 kernel");
adj([1,1,7,7], 3,2, 2,3, 1,0, null,null, "everything asymmetric at once");

print("");
print("== col2im: accumulation is visible, and the refusals ==");
// Ones in every column of a stride-1 3x3 pad-1 column matrix: each interior
// pixel is read by 9 kernel positions, so col2im must return 9 there.
$onecols = nd_ones([25, 9], "f64");
$acc = nd_col2im($onecols, [1, 1, 5, 5], 3, 3, 1, 1, 1, 1, null, null);
ok(nd_get($acc, [0, 0, 2, 2]) == 9.0, "an interior pixel is read 9 times and accumulates to 9");
ok(nd_get($acc, [0, 0, 0, 0]) == 4.0, "a corner pixel is read 4 times (the rest fell on padding)");
ok(nd_get($acc, [0, 0, 0, 2]) == 6.0, "an edge pixel is read 6 times");
// 4 corners read 4 times, 12 edge cells 6 times, the 3x3 interior 9 times.
// (The first draft of this line wrote 25*9 - 9*16 = 81, which is not that sum
// and not anything else either.)
ok(sc(nd_sum($acc, null, null)) == 4.0 * 4.0 + 12.0 * 6.0 + 9.0 * 9.0,
   "and the total equals the taps that landed inside the image (169 of 225)");
raises(def() { return nd_col2im($onecols, [1, 1, 5], 3, 3, 1, 1, 1, 1, null, null); },
       "a shape that is not 4 elements raises");
raises(def() { return nd_col2im($onecols, [1, 1, 5, 5], 2, 2, 1, 1, 0, 0, null, null); },
       "columns that do not match the geometry raise");
raises(def() { return nd_col2im(nd_ones([5], "f64"), [1,1,5,5], 3,3, 1,1, 1,1, null,null); },
       "a non-2-d column matrix raises");
try { nd_col2im($onecols, [1,1,5,5], 2,2, 1,1, 0,0, null,null); } catch ($e) {
    ok(contains($e.message, "[25,9]"), "and the error prints the columns it was given");
}

print("");
print("== A9: a convolution really is one matmul ==");
// The point of the whole family. Compute a convolution two ways: through
// im2col + matmul, and with four nested Bantu loops. They must agree.
nd_seed(31337);
$xc = nd_random_normal([2, 2, 5, 5]);
$wc = nd_random_normal([3, 2, 3, 3]);          // [out_c, in_c, kh, kw]
$cols2 = nd_im2col($xc, 3, 3, 1, 1, 1, 1, null, null);          // [2*5*5, 2*3*3]
$wmat = nd_reshape(nd_ascontiguous($wc), [3, 18]);
$outm = nd_matmul($cols2, nd_transpose($wmat, null), null);     // [50, 3]
// Reference: the definition of a convolution, written out.
def refConv($x, $w, $pad) {
    $n = nd_shape($x)[0]; $ci = nd_shape($x)[1];
    $h = nd_shape($x)[2]; $wd = nd_shape($x)[3];
    $co = nd_shape($w)[0]; $kh = nd_shape($w)[2]; $kw = nd_shape($w)[3];
    $oh = $h + 2 * $pad - $kh + 1; $ow = $wd + 2 * $pad - $kw + 1;
    $out = nd_zeros([$n, $oh, $ow, $co], "f64");
    $ni = 0;
    while ($ni < $n) {
        $oy = 0;
        while ($oy < $oh) {
            $ox = 0;
            while ($ox < $ow) {
                $oc = 0;
                while ($oc < $co) {
                    $acc = 0.0;
                    $c = 0;
                    while ($c < $ci) {
                        $i = 0;
                        while ($i < $kh) {
                            $j = 0;
                            while ($j < $kw) {
                                $iy = $oy + $i - $pad; $ix = $ox + $j - $pad;
                                if ($iy >= 0 && $iy < $h && $ix >= 0 && $ix < $wd) {
                                    $acc = $acc + nd_get($x, [$ni, $c, $iy, $ix])
                                                * nd_get($w, [$oc, $c, $i, $j]);
                                }
                                $j = $j + 1;
                            }
                            $i = $i + 1;
                        }
                        $c = $c + 1;
                    }
                    nd_set($out, [$ni, $oy, $ox, $oc], $acc);
                    $oc = $oc + 1;
                }
                $ox = $ox + 1;
            }
            $oy = $oy + 1;
        }
        $ni = $ni + 1;
    }
    return $out;
}
$ref = refConv($xc, $wc, 1);
ok(nd_allclose(nd_reshape($outm, [2, 5, 5, 3]), $ref, 1e-12, 1e-12),
   "im2col + one matmul == the four-loop definition of a convolution");
// The row order is (n, oh, ow), so the result reshapes with no transpose.
ok(nd_shape(nd_reshape($outm, [2, 5, 5, 3])) == [2, 5, 5, 3],
   "and the row order means the result reshapes to NHWC with no transpose");

print("");
print("== A9 stress: a realistic layer, and no leaks ==");
// 64 channels of 56x56 at stride 1 with a 3x3 kernel: one ResNet block's
// worth of im2col. The column matrix is 3136 x 576.
$t0 = clock();
$feat = nd_random_normal([1, 64, 56, 56]);
$colsBig = nd_im2col($feat, 3, 3, 1, 1, 1, 1, null, null);
$t1 = clock();
eqL(nd_shape($colsBig), [3136, 576], "a 64x56x56 feature map gives a 3136 x 576 column matrix");
print("        64x56x56 im2col (1.8M taps): " + str($t1 - $t0) + " ms");
ok($t1 - $t0 < 2000, "and it takes under 2s");
$t2 = clock();
$backBig = nd_col2im($colsBig, [1, 64, 56, 56], 3, 3, 1, 1, 1, 1, null, null);
$t3 = clock();
print("        the matching col2im:          " + str($t3 - $t2) + " ms");
ok(nd_shape($backBig) == [1, 64, 56, 56], "col2im returns the image shape");
// Every interior pixel was read 9 times, so col2im(im2col(x)) is 9x there.
ok(nd_allclose(nd_slice($backBig, [null, null, [1, 55, 1], [1, 55, 1]]),
               nd_multiply(nd_slice($feat, [null, null, [1, 55, 1], [1, 55, 1]]), 9.0, null),
               1e-9, 1e-9),
   "col2im(im2col(x)) is exactly 9x in the interior -- the overlap count");

$before = nd_live_bytes();
$n = 0;
while ($n < 300) {
    nd_col2im(nd_im2col($img, 3, 3, 1, 1, 1, 1, null, null), [1,1,4,4], 3,3, 1,1, 1,1, null,null);
    $n = $n + 1;
}
$after = nd_live_bytes();
print("        live array bytes: " + str($before) + " -> " + str($after));
ok($after - $before < 1000000, "300 im2col/col2im round trips grew live bytes by under 1MB");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
