// ════════════════════════════════════════════════════════════════════════
//  numba_index_add_test.b — nd_index_add, phase A5 of btorch.
//
//  nd_index_add(a, indices, values, axis?) adds slabs into `a` along `axis`,
//  in place, ACCUMULATING where an index repeats. It is the exact adjoint of
//  nd_take, and that is how it is tested: not just "does it add", but
//
//      <take(x, ix), y>  ==  <x, index_add(zeros_like(x), ix, y)>
//
//  for random x, y and ix. An adjoint identity is a much stronger statement
//  than any hand-checked example, because it has to hold for every input and
//  it fails if the geometry is off by one axis, one stride or one index.
//
//  Why this exists at all, when nd_put is right there: nd_put OVERWRITES.
//  Scattering a gradient with it silently drops every repeated index -- the
//  case an embedding table hits the first time a batch repeats a token. The
//  duplicate-index sections below are the whole point of the function.
//
//  Every numeric case is also checked against a pure-Bantu reference loop
//  (`ref1d`, `refAxis`) written independently of the kernel, so the test does
//  not merely assert that the kernel agrees with itself.
//
//  Run:  bantu run tests/numba_index_add_test.b
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
def close($x, $y, $name) { ok(nd_allclose($x, $y, 1e-12, 1e-12), $name); return null; }
// A 0-d reduction is an array, not a number; comparing it to a number is
// always false (btorch-suite/DECISIONS.md, the 0-d trap).
def sc($x) { return nd_get(nd_reshape($x, [1]), [0]); }
def raises($f, $name) {
    try { $f(); } catch ($e) { ok(true, $name); return null; }
    ok(false, $name + "  (no error raised)");
    return null;
}

// ── the reference implementations ────────────────────────────────────────
// Deliberately naive and written in terms of nd_get/nd_set only, so a stride,
// axis or accumulation mistake in the kernel cannot be mirrored here.
def ref1d($n, $ix, $vals) {
    $out = nd_zeros([$n], "f64");
    $k = 0;
    while ($k < len($ix)) {
        $i = $ix[$k];
        if ($i < 0) { $i = $i + $n; }
        nd_set($out, [$i], nd_get($out, [$i]) + $vals[$k]);
        $k = $k + 1;
    }
    return $out;
}
// 2-D only, either axis. Returns a fresh array.
def refAxis($rows, $cols, $ix, $v, $axis) {
    $out = nd_zeros([$rows, $cols], "f64");
    $k = 0;
    while ($k < len($ix)) {
        $i = $ix[$k];
        if ($axis == 0) {
            if ($i < 0) { $i = $i + $rows; }
            $c = 0;
            while ($c < $cols) {
                nd_set($out, [$i, $c], nd_get($out, [$i, $c]) + nd_get($v, [$k, $c]));
                $c = $c + 1;
            }
        }
        if ($axis == 1) {
            if ($i < 0) { $i = $i + $cols; }
            $r = 0;
            while ($r < $rows) {
                nd_set($out, [$r, $i], nd_get($out, [$r, $i]) + nd_get($v, [$r, $k]));
                $r = $r + 1;
            }
        }
        $k = $k + 1;
    }
    return $out;
}

print("== 1-D: the basics, and duplicates accumulate ==");
$a = nd_zeros([5], "f64");
$r = nd_index_add($a, [0, 2, 0, 4, 2, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0);
same($a, ref1d(5, [0,2,0,4,2,2], [1.0,2.0,3.0,4.0,5.0,6.0]), "1-D duplicates match the reference loop");
ok(nd_to_list($a) == [4.0, 0.0, 13.0, 0.0, 4.0], "1-D duplicates: the hand-computed answer");
ok(nd_base_id($r) == nd_base_id($a), "the return value is the destination, not a copy");

// This is the assertion that nd_put cannot satisfy, stated as a contrast so a
// future refactor that routes one through the other fails here.
$p = nd_zeros([5], "f64");
nd_put($p, [2, 2, 2], [1.0, 2.0, 3.0]);
ok(nd_get($p, [2]) == 3.0, "nd_put overwrites: last write wins (3, not 6)");
$q = nd_zeros([5], "f64");
nd_index_add($q, [2, 2, 2], [1.0, 2.0, 3.0], 0);
ok(nd_get($q, [2]) == 6.0, "nd_index_add accumulates: 1+2+3 = 6");

// Accumulating ONTO existing contents, not just into zeros.
$base = nd([10.0, 20.0, 30.0], "f64");
nd_index_add($base, [0, 0, 2], [1.0, 2.0, 3.0], 0);
ok(nd_to_list($base) == [13.0, 20.0, 33.0], "adds to what is already there");

print("");
print("== negative indices, and the default axis ==");
$d = nd_zeros([4], "f64");
nd_index_add($d, [-1, -4, -1], [1.0, 2.0, 3.0], 0);
same($d, ref1d(4, [-1,-4,-1], [1.0,2.0,3.0]), "negative indices match the reference");
ok(nd_to_list($d) == [2.0, 0.0, 0.0, 4.0], "-1 is the last element and -4 the first");
$e = nd_zeros([3], "f64");
nd_index_add($e, [1], [5.0]);                       // axis omitted
ok(nd_to_list($e) == [0.0, 5.0, 0.0], "axis defaults to the last axis, as nd_take's does");
// The default has to AGREE with nd_take's default, or the adjoint is wrong.
$m = nd([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "f64");
ok(nd_shape(nd_take($m, [0, 1])) == [2, 2], "nd_take's default axis is the last one");
$mz = nd_zeros([2, 3], "f64");
nd_index_add($mz, [0, 1], nd_ones([2, 2], "f64"));
ok(nd_to_list($mz) == [[1.0,1.0,0.0],[1.0,1.0,0.0]], "nd_index_add's default axis matches it");

print("");
print("== one value splats over every scattered position ==");
$s = nd_zeros([4], "f64");
nd_index_add($s, [1, 1, 3], 2.5, 0);
ok(nd_to_list($s) == [0.0, 5.0, 0.0, 2.5], "a bare number splats and still accumulates");
$s2 = nd_zeros([3, 2], "f64");
nd_index_add($s2, [0, 0, 2], 1.0, 0);
ok(nd_to_list($s2) == [[2.0,2.0],[0.0,0.0],[1.0,1.0]], "the splat covers the whole slab");
$s3 = nd_zeros([3], "f64");
nd_index_add($s3, [0], nd_zeros([1], "f64"), 0);
ok(nd_to_list($s3) == [0.0, 0.0, 0.0], "a size-1 ARRAY splats too, not only a number");
// A single index with a single value is ambiguous between splat and
// one-per-index. Both readings give the same answer, which is why it is safe.
$s4 = nd_zeros([2], "f64");
nd_index_add($s4, [1], [7.0], 0);
ok(nd_to_list($s4) == [0.0, 7.0], "the one-index/one-value overlap is consistent");

print("");
print("== 2-D, both axes, against the reference loop ==");
nd_seed(20261109);
$v0 = nd_random_normal([4, 3], 0.0, 1.0);
$ix0 = [0, 2, 2, 0];
$t0 = nd_zeros([3, 3], "f64");
nd_index_add($t0, $ix0, $v0, 0);
close($t0, refAxis(3, 3, $ix0, $v0, 0), "axis 0 matches the reference loop");

$v1 = nd_random_normal([3, 4], 0.0, 1.0);
$ix1 = [1, 1, 0, 1];
$t1 = nd_zeros([3, 3], "f64");
nd_index_add($t1, $ix1, $v1, 1);
close($t1, refAxis(3, 3, $ix1, $v1, 1), "axis 1 matches the reference loop");

// The embedding gradient, by hand: a 3x2 table, tokens [0,2,0].
$w = nd_zeros([3, 2], "f64");
nd_index_add($w, [0, 2, 0], nd([[1.0,2.0],[3.0,4.0],[5.0,6.0]], "f64"), 0);
ok(nd_to_list($w) == [[6.0,8.0],[0.0,0.0],[3.0,4.0]], "the embedding gradient, hand-computed");

print("");
print("== 3-D and higher: every axis ==");
$x3 = nd_reshape(nd_arange(0.0, 24.0, 1.0), [2, 3, 4]);
$ax = 0;
while ($ax < 3) {
    $k = 2;
    $sh = nd_shape($x3);
    $vsh = [$sh[0], $sh[1], $sh[2]];
    $vsh[$ax] = $k;
    $vv = nd_ones($vsh, "f64");
    $dst = nd_zeros([$sh[0], $sh[1], $sh[2]], "f64");
    nd_index_add($dst, [0, 0], $vv, $ax);
    // Two unit slabs into index 0 means that slab is exactly 2 everywhere and
    // the rest is 0, so the total is 2 * (slab size).
    $slab = (24 / $sh[$ax]);
    ok(sc(nd_sum($dst, null, null)) == 2.0 * $slab, "3-D axis " + str($ax) + ": total is 2x the slab");
    ok(sc(nd_max($dst, null, null)) == 2.0, "3-D axis " + str($ax) + ": duplicates reached 2, not 1");
    $ax = $ax + 1;
}

print("");
print("== THE ADJOINT IDENTITY: <take(x,ix), y> == <x, index_add(0,ix,y)> ==");
// take and index_add are linear maps that must be each other's transpose. If
// they are, the two inner products below are equal for EVERY x, y, ix. This
// single property pins down the axis, the strides and the accumulation at once.
def adjoint($shape, $ix, $axis, $name) {
    nd_seed(7777);
    $x = nd_random_normal($shape, 0.0, 1.0);
    $taken = nd_take($x, $ix, $axis);
    $y = nd_random_normal(nd_shape($taken), 0.0, 1.0);
    $lhs = sc(nd_sum(nd_multiply($taken, $y, null), null, null));
    $back = nd_zeros_like($x);
    nd_index_add($back, $ix, $y, $axis);
    $rhs = sc(nd_sum(nd_multiply($x, $back, null), null, null));
    $den = 1.0;
    if ($lhs > $den) { $den = $lhs; }
    if (0.0 - $lhs > $den) { $den = 0.0 - $lhs; }
    $diff = $lhs - $rhs;
    if ($diff < 0.0) { $diff = 0.0 - $diff; }
    ok($diff / $den < 1e-12, $name + "  (lhs " + str($lhs) + " rhs " + str($rhs) + ")");
    return null;
}
adjoint([6],        [0, 3, 3, 5, 0],    0, "1-D, duplicates");
adjoint([5, 4],     [0, 2, 2],          0, "2-D axis 0 (embedding)");
adjoint([5, 4],     [1, 3, 1, 1],       1, "2-D axis 1");
adjoint([3, 4, 2],  [0, 2, 1, 2],       1, "3-D middle axis");
adjoint([3, 4, 2],  [1, 1],             2, "3-D last axis");
adjoint([4],        [-1, -1, 0],        0, "negative indices are adjoint too");
adjoint([7, 3],     [0, 1, 2, 3, 4, 5, 6], 0, "a full permutation-free pass");

print("");
print("== strided and transposed destinations ==");
// Every other element of a 1-D array: the kernel must follow the stride, not
// assume element n sits at offset n.
$full = nd_zeros([8], "f64");
$odd = nd_slice($full, [[1, 8, 2]]);                 // [1,3,5,7], stride 2
ok(nd_shape($odd) == [4], "the slice is the shape we think it is");
nd_index_add($odd, [0, 3, 0], [1.0, 2.0, 3.0], 0);
ok(nd_to_list($full) == [0.0,4.0,0.0,0.0,0.0,0.0,0.0,2.0], "a strided view writes through to its base");
ok(nd_to_list($odd) == [4.0, 0.0, 0.0, 2.0], "and the view reads back the same values");

// A transposed destination: axis 0 of the transpose is axis 1 of the base.
$bm = nd_zeros([2, 3], "f64");
$tm = nd_transpose($bm, null);                       // [3,2]
nd_index_add($tm, [0, 0, 2], nd_ones([3, 2], "f64"), 0);
ok(nd_to_list($bm) == [[2.0,0.0,1.0],[2.0,0.0,1.0]], "a transposed view scatters into the right base cells");

// A strided SOURCE.
$src = nd_slice(nd_reshape(nd_arange(0.0, 12.0, 1.0), [6, 2]), [[0, 6, 2]]);  // [3,2] stride 4
$dst2 = nd_zeros([2, 2], "f64");
nd_index_add($dst2, [0, 1, 0], $src, 0);
ok(nd_to_list($dst2) == [[8.0,10.0],[4.0,5.0]], "a strided source is read through its strides");

print("");
print("== aliasing: source and destination sharing one buffer ==");
// x.index_add(ix, x.take(ix)) is the natural caller, and nd_take already
// copies. The dangerous form is a VIEW of the destination as the source, where
// a naive loop would read values it has already written.
$al = nd([1.0, 2.0, 3.0, 4.0], "f64");
$alv = nd_slice($al, [[0, 2, 1]]);                   // a view of [1,2]
ok(nd_shares_memory($al, $alv), "the source really does share the destination's buffer");
nd_index_add($al, [0, 1], $alv, 0);
// Snapshot semantics: both reads see the ORIGINAL [1,2], so the answer is
// [1+1, 2+2, 3, 4]. A read-after-write loop would give [2, 4, ...] only by
// luck of ordering here, so the stronger case is the overlapping one below.
ok(nd_to_list($al) == [2.0, 4.0, 3.0, 4.0], "an overlapping source is snapshotted, not read as it changes");

$al2 = nd([1.0, 2.0, 3.0], "f64");
$sh2 = nd_slice($al2, [[1, 3, 1]]);                  // [2,3], overlaps indices 1 and 2
nd_index_add($al2, [0, 1], $sh2, 0);
ok(nd_to_list($al2) == [3.0, 5.0, 3.0], "overlapping, offset source: 1+2, 2+3, 3 unchanged");

print("");
print("== dtypes ==");
$i = nd_zeros([4], "i64");
nd_index_add($i, [0, 0, 3], [2.0, 3.0, 4.0], 0);
ok(nd_dtype($i) == "i64", "an i64 destination stays i64");
ok(nd_to_list($i) == [5, 0, 0, 4], "i64 accumulates exactly for whole numbers");
$b = nd_zeros([3], "bool");
nd_index_add($b, [1, 1], [1.0, 1.0], 0);
ok(nd_to_list($b) == [false, true, false], "a bool destination saturates rather than wrapping");
// f64 values into an i64 destination round at every step, by the same policy
// nd_put uses. Asserted so the behaviour is pinned, not discovered.
$ir = nd_zeros([1], "i64");
nd_index_add($ir, [0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], 0);
ok(nd_get($ir, [0]) == 4, "f64 into i64 rounds each step (0.5 -> 1 four times), as nd_put does");
// Mixed: i64 source into an f64 destination.
$fm = nd_zeros([2], "f64");
nd_index_add($fm, [0, 0], nd([3, 4], "i64"), 0);
ok(nd_get($fm, [0]) == 7.0, "an i64 source adds into an f64 destination");

print("");
print("== degenerate shapes ==");
$z = nd_zeros([3], "f64");
nd_index_add($z, [], [], 0);
ok(nd_to_list($z) == [0.0, 0.0, 0.0], "no indices is a no-op, not an error");
$one = nd_zeros([1, 4], "f64");
nd_index_add($one, [0, 0], nd_ones([2, 4], "f64"), 0);
ok(nd_to_list($one) == [[2.0,2.0,2.0,2.0]], "a length-1 axis takes every index");
$e1 = nd_zeros([0, 3], "f64");
nd_index_add($e1, [], nd_zeros([0, 3], "f64"), 0);
ok(nd_shape($e1) == [0, 3], "an empty array with no indices survives untouched");
$col = nd_zeros([4, 1], "f64");
nd_index_add($col, [0, 0], nd_ones([2, 1], "f64"), 0);
ok(nd_to_list($col) == [[2.0],[0.0],[0.0],[0.0]], "a width-1 slab");

print("");
print("== every failure mode raises, by name ==");
$t = nd_zeros([5], "f64");
raises(def() { return nd_index_add($t, [5], [1.0], 0); },  "an index one past the end raises");
raises(def() { return nd_index_add($t, [-6], [1.0], 0); }, "an index one before the start raises");
raises(def() { return nd_index_add($t, [0.5], [1.0], 0); },"a fractional index raises");
raises(def() { return nd_index_add($t, [0, 1], nd_zeros([3], "f64"), 0); }, "a mismatched value count raises");
raises(def() { return nd_index_add($t, [0], [1.0], 7); },  "an out-of-range axis raises");
raises(def() { return nd_index_add($t, [0]); },            "too few arguments raises");
raises(def() { return nd_index_add($t, [0], nd_zeros([2, 2], "f64"), 0); }, "a values array of the wrong rank raises");
// An empty destination axis has no valid index at all.
$ez = nd_zeros([0], "f64");
raises(def() { return nd_index_add($ez, [0], [1.0], 0); }, "any index into an empty axis raises");
// A broadcast view repeats one element, so accumulating into it would add
// many times to the same memory. That must be refused, not silently wrong.
$ro = nd_broadcast_to(nd_zeros([1], "f64"), [4]);
raises(def() { return nd_index_add($ro, [0], [1.0], 0); }, "a read-only broadcast destination raises");
// A list destination would accumulate into a temporary array and vanish.
raises(def() { return nd_index_add([0.0, 0.0], [0], [1.0], 0); }, "a plain list destination raises rather than silently vanishing");
// 0-d has no axis.
raises(def() { return nd_index_add(nd(1.0, "f64"), [0], [1.0], 0); }, "a 0-d destination raises");

print("");
print("== the error messages say what is wrong ==");
try { nd_index_add($t, [9], [1.0], 0); } catch ($err) {
    ok(contains($err.message, "nd_index_add") && contains($err.message, "9") && contains($err.message, "5"),
       "the range error names the function, the index and the extent");
}
try { nd_index_add($t, [0, 1], nd_zeros([3], "f64"), 0); } catch ($err) {
    ok(contains($err.message, "[3]") && contains($err.message, "[2]"),
       "the shape error prints both the shape given and the shape needed");
}

print("");
print("== stress: scale, repetition, and the all-duplicate worst case ==");
// clock() is MILLISECONDS (evaluator.hpp:8771). The first draft of this
// section divided it by 1000 as if it were microseconds, which printed every
// timing 1000x too small and made both budget assertions vacuous.
//
// And a wall-clock budget is the weak form of the claim anyway: it measures
// the machine. The real claim is the COST SHAPE. nd_put materialises the
// destination's whole flat index list on every call, so its cost grows with
// the ARRAY even when the index count does not -- a 3-index write into a 10M
// array costs 10M. nd_index_add must not. So: the same 1M indices, the same
// valid range, scattered into two destinations 20x apart in size. The bigger
// one is allowed to cost more (random writes over 80MB miss cache in a way
// writes over 4MB do not) but it must not cost 20x.
$N = 1000000;
$six = nd_astype(nd_random_int([$N], 0, 500000), "i64");
$svals = nd_full([$N], 1.0);

$small = nd_zeros([500000], "f64");
$t0 = clock();
nd_index_add($small, $six, $svals, 0);
$tsmall = clock() - $t0;

$big = nd_zeros([10000000], "f64");          // 20x the array, identical indices
$t1 = clock();
nd_index_add($big, $six, $svals, 0);
$tbig = clock() - $t1;

ok(sc(nd_sum($small, null, null)) == 1000000.0, "1M scatters into 500k: every value landed exactly once");
ok(sc(nd_sum($big, null, null)) == 1000000.0,   "1M scatters into 10M: every value landed exactly once");
print("        1M scatters: " + str($tsmall) + " ms into 500k, " + str($tbig) + " ms into 10M");
ok($tbig < 10 * $tsmall + 50,
   "a 20x larger destination costs well under 20x -- the cost follows the indices, not the array");

// All-duplicate: a million adds to ONE cell. Worst case for any ordering trick.
$hot = nd_zeros([10000], "f64");
$zix = nd_zeros([1000000], "i64");
nd_index_add($hot, $zix, nd_full([1000000], 1.0), 0);
ok(nd_get($hot, [0]) == 1000000.0, "1M adds to a single cell accumulate to exactly 1e6");
ok(sc(nd_sum($hot, null, null)) == 1000000.0, "and nothing landed anywhere else");

// The embedding shape at a realistic size, through the tight f64 slab path.
$emb = nd_zeros([50000, 128], "f64");
$tok = nd_astype(nd_random_int([20000], 0, 50000), "i64");
$gr = nd_full([20000, 128], 0.5);
$t2 = clock();
nd_index_add($emb, $tok, $gr, 0);
$temb = clock() - $t2;
print("        20k x 128 embedding gradient: " + str($temb) + " ms");
ok(nd_allclose(nd_sum($emb, null, null), nd(20000.0 * 128.0 * 0.5, "f64"), 1e-6, 1e-6),
   "the embedding gradient sums to rows x cols x value");
ok($temb < 400, "2.56M element-adds through the tight slab path stay under 400 ms");

// Repetition: 20,000 calls must not leak. nd_live_bytes is the allocator's
// own accounting, so it is a direct statement about arrays, not RSS.
$acc = nd_zeros([1000], "f64");
$before = nd_live_bytes();
$n = 0;
while ($n < 20000) {
    nd_index_add($acc, [1, 1, 2], [1.0, 1.0, 1.0], 0);
    $n = $n + 1;
}
$after = nd_live_bytes();
ok(nd_get($acc, [1]) == 40000.0, "20,000 repeated calls accumulated 2 per call into cell 1");
ok($after == $before, "20,000 calls leaked no array bytes (" + str($before) + " -> " + str($after) + ")");

// A long strided run, so the generic odometer gets the same scale treatment as
// the tight path.
$sb = nd_zeros([2000000], "f64");
$sv = nd_slice($sb, [[0, 2000000, 2]]);              // 1M elements, stride 2
nd_index_add($sv, nd_astype(nd_random_int([200000], 0, 1000000), "i64"), 1.0, 0);
ok(sc(nd_sum($sb, null, null)) == 200000.0, "200k scatters through a stride-2 view all landed");
$oddsum = sc(nd_sum(nd_slice($sb, [[1, 2000000, 2]]), null, null));
ok($oddsum == 0.0, "and not one of them touched an odd offset");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
