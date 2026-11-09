// ════════════════════════════════════════════════════════════════════════
//  btorch_conv_test.b — phase 2's join/split/gather and phase 5's
//  convolution, pooling and normalization.
//
//  The design claim being tested: there is NO convolution-specific gradient
//  anywhere in btorch. conv2d is im2col composed with reshape, matmul, add
//  and permute; pooling is im2col composed with mean or max. im2col's
//  backward is col2im, its verified adjoint. Everything else was gradchecked
//  in phase 2.
//
//  So if that claim holds, gradcheck on conv2d cannot fail for a reason
//  specific to convolution -- and the forward can be checked independently
//  against the four-loop definition. Both are asserted below.
//
//  Run:  bantu run tests/btorch_conv_test.b
// ════════════════════════════════════════════════════════════════════════

include "../btorch/btorch.b" as bt;
include "../btorch/nn.b" as nn;
include "../btorch/optim.b" as optim;

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
def eqL($got, $want, $name) {
    if (str($got) == str($want)) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name);
        print("          got:  " + str($got));
        print("          want: " + str($want));
    }
}
def raises($fn, $name) {
    try { $fn(); $R.fail = $R.fail + 1; print("  FAIL  " + $name + " (no error raised)"); }
    catch ($e) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    return null;
}
def gc($f, $inputs, $name) {
    $r = bt.gradcheck($f, $inputs, null, null);
    if ($r.ok) { $R.pass = $R.pass + 1; print("  ok    gradcheck " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  gradcheck " + $name + "  worst rel err " + str($r.worst)
              + " at " + str($r.where));
    }
    return null;
}
def sc($x) { return nd_get(nd_reshape($x, [1]), [0]); }
// A C++ runtime error arrives as an OBJECT with .message; a Bantu `throw "..."`
// arrives as the STRING itself. btorch raises both kinds, so asserting on a
// message has to handle both -- `$e.message` on a string is null, and the
// assertion then fails for a reason that has nothing to do with the error.
def errText($e) {
    if (type($e) == "string") { return $e; }
    return str($e.message);
}
// An exact integer floor. nd_astype(x, "i64") ROUNDS (llround), so it is not
// one -- which is how the reference convolution below first computed an output
// height of 4 where the real answer is 3.
def fdiv($a, $b) { return nd_get(nd_floor_divide(nd_full([1], $a * 1.0), $b * 1.0, null), [0]); }

bt.manual_seed(20261109);

print("========================================");
print("  cat / stack / chunk / split");
print("========================================");

$a = bt.tensor([[1.0, 2.0], [3.0, 4.0]], true);
$b = bt.tensor([[5.0, 6.0]], true);
eqL(bt.cat([$a, $b], 0).to_list(), [[1.0,2.0],[3.0,4.0],[5.0,6.0]], "cat along axis 0");
eqL(bt.cat([$a, bt.tensor([[9.0],[8.0]], false)], 1).to_list(), [[1.0,2.0,9.0],[3.0,4.0,8.0]],
    "cat along axis 1");
// Weighted, so each input's gradient is distinguishable rather than all ones.
$wc = bt.tensor([[1.0,2.0],[3.0,4.0],[5.0,6.0]], false);
bt.sum(bt.mul(bt.cat([$a, $b], 0), $wc), null, null).backward(null);
eqL(nd_to_list($a.grad), [[1.0,2.0],[3.0,4.0]], "cat's gradient for input 0 is its own slice");
eqL(nd_to_list($b.grad), [[5.0,6.0]], "and input 1 gets the rest, not a copy of the first");

$s1 = bt.tensor([1.0, 2.0], true);
$s2 = bt.tensor([3.0, 4.0], true);
eqL(bt.stack([$s1, $s2], 0).to_list(), [[1.0,2.0],[3.0,4.0]], "stack on axis 0");
eqL(bt.stack([$s1, $s2], 1).to_list(), [[1.0,3.0],[2.0,4.0]], "stack on axis 1 interleaves");
bt.sum(bt.mul(bt.stack([$s1, $s2], 0), bt.tensor([[1.0,2.0],[3.0,4.0]], false)), null, null).backward(null);
eqL(nd_to_list($s1.grad), [1.0, 2.0], "stack's gradient for input 0 is slab 0");
eqL(nd_to_list($s2.grad), [3.0, 4.0], "and for input 1 is slab 1");

// The same tensor twice: both contributions must land, which is the smallest
// graph that tells a correct engine from one that overwrites.
$twice = bt.tensor([1.0, 2.0], true);
bt.sum(bt.cat([$twice, $twice], 0), null, null).backward(null);
eqL(nd_to_list($twice.grad), [2.0, 2.0], "cat of one tensor with ITSELF accumulates both contributions");

$c4 = bt.tensor([1.0, 2.0, 3.0, 4.0, 5.0], true);
$ch = bt.chunk($c4, 2, 0);
// PyTorch's rule is ceil(extent/chunks) PER CHUNK, so chunk(5, 2) is [3, 2]:
// TWO pieces, not two-and-a-bit. The count can come out below `chunks`.
ok(len($ch) == 2, "chunk(5, 2) gives 2 pieces of ceil(5/2)=3, as PyTorch does");
eqL($ch[0].to_list(), [1.0, 2.0, 3.0], "the first chunk takes the ceiling");
eqL($ch[1].to_list(), [4.0, 5.0], "and the last is the short one");
$ch7 = bt.chunk(bt.tensor(nd_astype(nd_arange(0.0, 7.0, null), "f64"), false), 3, 0);
ok(len($ch7) == 3, "chunk(7, 3) gives 3 pieces");
eqL($ch7[0].to_list(), [0.0, 1.0, 2.0], "...of 3");
eqL($ch7[2].to_list(), [6.0], "...with 1 left over");
$sp = bt.split($c4, [2, 3], 0);
eqL($sp[1].to_list(), [3.0, 4.0, 5.0], "split by explicit sizes");
// Every piece's gradient must find its way back into one input.
$pieces = bt.chunk($c4, 2, 0);
bt.sum(bt.cat($pieces, 0), null, null).backward(null);
eqL(nd_to_list($c4.grad), [1.0,1.0,1.0,1.0,1.0], "cat(chunk(x)) sends exactly 1 back to every element");
raises(def() { return bt.cat([], 0); },          "cat of an empty list raises");
raises(def() { return bt.cat($a, 0); },          "cat of a bare tensor raises");
raises(def() { return bt.chunk($c4, 0, 0); },    "chunk into 0 raises");
raises(def() { return bt.split($c4, [1, 1], 0); },"split sizes that do not sum raise");

print("");
print("== pad ==");
eqL(bt.pad($a, [[1,0],[0,1]], 0.0).to_list(), [[0.0,0.0,0.0],[1.0,2.0,0.0],[3.0,4.0,0.0]],
    "pad with one pair per axis");
eqL(bt.pad($a, 1, 7.0).shape(), [4, 4], "a scalar pad widens every axis");
// The gradient of a padded cell carries nowhere; a real cell keeps its own.
$pg = bt.tensor([[1.0, 2.0]], true);
bt.sum(bt.mul(bt.pad($pg, [[1,1],[1,1]], 0.0),
              bt.tensor(nd_full([3,4], 2.0), false)), null, null).backward(null);
eqL(nd_to_list($pg.grad), [[2.0, 2.0]], "pad's backward crops: only the real cells get gradient");
raises(def() { return bt.pad($a, 0 - 1, 0.0); }, "a negative pad raises at the btorch level too");

print("");
print("== gather: per-element indexing ==");
$g2 = bt.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]], true);
eqL(bt.gather($g2, 1, nd([[2],[0]], "i64")).to_list(), [[3.0],[4.0]], "gather one element per row");
eqL(bt.gather($g2, 1, nd([[0,2],[1,1]], "i64")).to_list(), [[1.0,3.0],[5.0,5.0]],
    "gather several per row, repeats included");
eqL(bt.gather($g2, 0, nd([[0,1,0]], "i64")).to_list(), [[1.0,5.0,3.0]], "gather along dim 0");
// A repeated index must ACCUMULATE, as everywhere else.
bt.gather($g2, 1, nd([[1,1],[0,0]], "i64")).sum(null, null).backward(null);
eqL(nd_to_list($g2.grad), [[0.0,2.0,0.0],[2.0,0.0,0.0]],
    "an index gathered twice gets gradient 2");
// The flattening is what makes validation necessary: 3 is a legal flat index
// into a [2,3] tensor but an illegal column index, so it must be refused.
raises(def() { return bt.gather($g2, 1, nd([[3],[0]], "i64")); },
       "an index past the dim's extent raises, even though it is a legal FLAT index");
raises(def() { return bt.gather($g2, 1, nd([[0 - 1],[0]], "i64")); }, "a negative index raises");
raises(def() { return bt.gather($g2, 1, nd([0, 1], "i64")); },  "indices of the wrong rank raise");
raises(def() { return bt.gather($g2, null, nd([[0]], "i64")); }, "gather without a dim raises");
try { bt.gather($g2, 1, nd([[3],[0]], "i64")); } catch ($e) {
    ok(contains(errText($e), "different row"),
       "and the error explains WHY flattening makes this a silent bug otherwise");
}

print("");
print("== repeat_interleave / tile ==");
$r3 = bt.tensor([1.0, 2.0, 3.0], true);
eqL(bt.repeat_interleave($r3, 2, null).to_list(), [1.0,1.0,2.0,2.0,3.0,3.0], "repeat_interleave");
eqL(bt.tile($r3, [2]).to_list(), [1.0,2.0,3.0,1.0,2.0,3.0], "tile");
bt.repeat_interleave($r3, 3, null).sum(null, null).backward(null);
eqL(nd_to_list($r3.grad), [3.0,3.0,3.0], "repeat_interleave's adjoint sums the 3 copies");
$r3b = bt.tensor([1.0, 2.0, 3.0], true);
bt.tile($r3b, [4]).sum(null, null).backward(null);
eqL(nd_to_list($r3b.grad), [4.0,4.0,4.0], "tile's adjoint sums the 4 blocks");
$r2d = bt.tensor([[1.0,2.0],[3.0,4.0]], true);
bt.tile($r2d, [2, 3]).sum(null, null).backward(null);
eqL(nd_to_list($r2d.grad), [[6.0,6.0],[6.0,6.0]], "a 2x3 tile sends 6 back to every element");
raises(def() { return bt.tile($r2d, [2]); }, "tile with the wrong number of reps raises");

print("");
print("========================================");
print("  conv2d — against the definition");
print("========================================");

// Hand-computed: 4x4 image 0..15, a 2x2 all-ones kernel, stride 1, no pad.
$img = bt.tensor(nd_reshape(nd_astype(nd_arange(0.0, 16.0, null), "f64"), [1,1,4,4]), true);
$ones = bt.tensor(nd_ones([1,1,2,2], "f64"), true);
$o = bt.conv2d($img, $ones, null, 1, 0, null);
eqL($o.shape(), [1,1,3,3], "a 2x2 kernel on 4x4 with stride 1 gives 3x3");
eqL(nd_to_list(nd_reshape($o.data, [9])), [10.0,14.0,18.0,26.0,30.0,34.0,42.0,46.0,50.0],
    "and every window sums correctly (0+1+4+5 = 10, ...)");

// The four-loop definition, as an independent reference.
def refConv($x, $w, $pad, $stride) {
    $n = nd_shape($x)[0]; $ci = nd_shape($x)[1];
    $h = nd_shape($x)[2]; $wd = nd_shape($x)[3];
    $co = nd_shape($w)[0]; $kh = nd_shape($w)[2]; $kw = nd_shape($w)[3];
    $oh = fdiv($h + 2 * $pad - $kh, $stride) + 1;
    $ow = fdiv($wd + 2 * $pad - $kw, $stride) + 1;
    $out = nd_zeros([$n, $co, $oh, $ow], "f64");
    $ni = 0;
    while ($ni < $n) {
        $oc = 0;
        while ($oc < $co) {
            $oy = 0;
            while ($oy < $oh) {
                $ox = 0;
                while ($ox < $ow) {
                    $acc = 0.0;
                    $c = 0;
                    while ($c < $ci) {
                        $i = 0;
                        while ($i < $kh) {
                            $j = 0;
                            while ($j < $kw) {
                                $iy = $oy * $stride + $i - $pad;
                                $ix = $ox * $stride + $j - $pad;
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
                    nd_set($out, [$ni, $oc, $oy, $ox], $acc);
                    $ox = $ox + 1;
                }
                $oy = $oy + 1;
            }
            $oc = $oc + 1;
        }
        $ni = $ni + 1;
    }
    return $out;
}

$xr = nd_random_normal([2, 3, 6, 7]);
$wr = nd_random_normal([4, 3, 3, 3]);
$tx = bt.tensor($xr, true);
$tw = bt.tensor($wr, true);
ok(nd_allclose(bt.conv2d($tx, $tw, null, 1, 0, null).data, refConv($xr, $wr, 0, 1), 1e-12, 1e-12),
   "conv2d == the four-loop definition (stride 1, no pad)");
ok(nd_allclose(bt.conv2d($tx, $tw, null, 1, 1, null).data, refConv($xr, $wr, 1, 1), 1e-12, 1e-12),
   "...with padding 1");
ok(nd_allclose(bt.conv2d($tx, $tw, null, 2, 1, null).data, refConv($xr, $wr, 1, 2), 1e-12, 1e-12),
   "...with stride 2 and padding 1");
// Bias is added per output channel, which the NHWC layout makes a plain
// broadcast -- and getting the axis wrong here is invisible until training.
$bias = bt.tensor([1.0, 10.0, 100.0, 1000.0], true);
$ob = bt.conv2d($tx, $tw, $bias, 1, 0, null);
$onb = bt.conv2d($tx, $tw, null, 1, 0, null);
$diff = nd_subtract($ob.data, $onb.data, null);
ok(nd_allclose(nd_mean(nd_reshape(nd_slice($diff, [null, [1,2,1], null, null]),
                                 [nd_size(nd_slice($diff, [null, [1,2,1], null, null]))]), null, null),
               nd(10.0, "f64"), 1e-12, 1e-12),
   "the bias lands on the CHANNEL axis -- channel 1 gained exactly 10");

eqL(bt.conv2d($tx, $tw, null, 1, 1, null).shape(), [2, 4, 6, 7], "shape with padding 1 is preserved");
raises(def() { return bt.conv2d(bt.tensor(nd_zeros([3,4],"f64"), false), $tw, null, 1, 0, null); },
       "a non-4-d input raises");
raises(def() { return bt.conv2d($tx, bt.tensor(nd_zeros([4,9,3,3],"f64"), false), null, 1, 0, null); },
       "a channel mismatch raises by name");
raises(def() { return bt.conv2d($tx, $tw, null, 1, 0, 5); },
       "a dilation that makes the kernel too large raises");

print("");
print("== conv1d ==");
$x1 = bt.tensor(nd_random_normal([2, 3, 10]), true);
$w1 = bt.tensor(nd_random_normal([5, 3, 3]), true);
eqL(bt.conv1d($x1, $w1, null, 1, 0, null).shape(), [2, 5, 8], "conv1d shrinks the length by k-1");
eqL(bt.conv1d($x1, $w1, null, 1, 1, null).shape(), [2, 5, 10], "with padding 1 the length is kept");
eqL(bt.conv1d($x1, $w1, null, 2, 0, null).shape(), [2, 5, 4], "stride 2 halves it");
// conv1d is conv2d with a height of 1, so it must agree with one.
$x1b = nd_random_normal([1, 2, 6]);
$w1b = nd_random_normal([3, 2, 3]);
$viaConv2 = bt.conv2d(bt.tensor(nd_reshape($x1b, [1,2,1,6]), false),
                      bt.tensor(nd_reshape($w1b, [3,2,1,3]), false), null, 1, 0, null);
ok(nd_allclose(bt.conv1d(bt.tensor($x1b, false), bt.tensor($w1b, false), null, 1, 0, null).data,
               nd_reshape($viaConv2.data, [1, 3, 4]), 1e-13, 1e-13),
   "conv1d agrees with the equivalent conv2d exactly");

print("");
print("========================================");
print("  pooling");
print("========================================");

eqL(bt.max_pool2d($img, 2, null, null, null, null, null).to_list(), [[[[5.0,7.0],[13.0,15.0]]]],
    "max_pool2d 2x2 (stride defaults to the KERNEL, not 1)");
eqL(bt.avg_pool2d($img, 2, null, null, null, null, null).to_list(), [[[[2.5,4.5],[10.5,12.5]]]],
    "avg_pool2d 2x2");
eqL(bt.max_pool2d($img, 2, null, 1, null, null, null).shape(), [1,1,3,3],
    "an explicit stride of 1 overlaps the windows");
// The padding trap: im2col pads with ZERO, which for a maximum over negative
// values would give 0. max_pool2d pads with a large negative sentinel instead.
$neg = bt.tensor(nd_full([1,1,2,2], 0.0 - 5.0), true);
eqL(bt.max_pool2d($neg, 2, null, null, null, null, null).to_list(), [[[[0 - 5.0]]]],
    "max over all-negative values is negative, not 0");
$neg2 = bt.tensor(nd_full([1,1,2,2], 0.0 - 5.0), true);
eqL(bt.max_pool2d($neg2, 2, null, 2, null, 1, null).to_list(),
    [[[[0 - 5.0, 0 - 5.0],[0 - 5.0, 0 - 5.0]]]],
    "...and PADDED max pooling of negatives does not leak a 0 either");
// Pooling's gradient routes only to the winners.
$pm = bt.tensor([[[[1.0,2.0],[3.0,4.0]]]], true);
bt.max_pool2d($pm, 2, null, null, null, null, null).sum(null, null).backward(null);
eqL(nd_to_list($pm.grad), [[[[0.0,0.0],[0.0,1.0]]]], "max_pool2d's gradient reaches only the max");
$pa = bt.tensor([[[[1.0,2.0],[3.0,4.0]]]], true);
bt.avg_pool2d($pa, 2, null, null, null, null, null).sum(null, null).backward(null);
eqL(nd_to_list($pa.grad), [[[[0.25,0.25],[0.25,0.25]]]], "avg_pool2d's gradient splits evenly");
// Overlapping windows: an element in two windows gets both gradients.
$po = bt.tensor([[[[1.0,2.0,3.0]]]], true);
bt.max_pool2d($po, 1, 2, 1, 1, null, null).sum(null, null).backward(null);
eqL(nd_to_list($po.grad), [[[[0.0,1.0,1.0]]]],
    "with overlapping 1x2 windows the middle element wins once and the last once");
raises(def() { return bt.max_pool2d(bt.tensor(nd_zeros([4,4],"f64"), false), 2, null, null, null, null, null); },
       "pooling a non-4-d tensor raises");

print("");
print("========================================");
print("  TIER 3 — gradcheck");
print("========================================");
// Coefficients drawn ONCE, outside every closure (DECISIONS B17).
$k1 = bt.tensor(nd_random_normal([3, 2]), false);
gc(def($in) { return bt.sum(bt.mul(bt.cat([$in[0], $in[1]], 0), $k1), null, null); },
   [bt.tensor(nd_random_normal([2,2]), true), bt.tensor(nd_random_normal([1,2]), true)],
   "cat, both inputs");
$k2 = bt.tensor(nd_random_normal([2, 3]), false);
gc(def($in) { return bt.sum(bt.mul(bt.stack([$in[0], $in[1]], 0), $k2), null, null); },
   [bt.tensor(nd_random_normal([3]), true), bt.tensor(nd_random_normal([3]), true)],
   "stack, both inputs");
gc(def($in) { return bt.sum(bt.exp(bt.cat([$in[0], $in[0]], 0)), null, null); },
   [bt.tensor(nd_random_normal([2]), true)], "cat of a tensor with itself, under exp");
$k3 = bt.tensor(nd_random_normal([4, 5]), false);
gc(def($in) { return bt.sum(bt.mul(bt.pad($in[0], [[1,1],[1,1]], 0.0), $k3), null, null); },
   [bt.tensor(nd_random_normal([2,3]), true)], "pad");
$k4 = bt.tensor(nd_random_normal([2,2]), false);
gc(def($in) { return bt.sum(bt.mul(bt.gather($in[0], 1, nd([[1,1],[0,2]], "i64")), $k4), null, null); },
   [bt.tensor(nd_random_normal([2,3]), true)], "gather with a repeated index");
gc(def($in) { return bt.sum(bt.tanh(bt.repeat_interleave($in[0], 3, null)), null, null); },
   [bt.tensor(nd_random_normal([3]), true)], "repeat_interleave under tanh");
gc(def($in) { return bt.sum(bt.tanh(bt.tile($in[0], [2,2])), null, null); },
   [bt.tensor(nd_random_normal([2,2]), true)], "tile under tanh");
gc(def($in) { return bt.sum(bt.exp(bt.chunk($in[0], 2, 0)[0]), null, null); },
   [bt.tensor(nd_random_normal([5]), true)], "chunk, through one piece");

print("");
print("== gradcheck: im2col, and then convolution itself ==");
gc(def($in) { return bt.sum(bt.exp(bt.im2col($in[0], [2,2,1,1,0,0,1,1])), null, null); },
   [bt.tensor(nd_random_normal([1,1,3,3]), true)], "im2col, stride 1 (overlapping)");
gc(def($in) { return bt.sum(bt.exp(bt.im2col($in[0], [2,2,1,1,1,1,1,1])), null, null); },
   [bt.tensor(nd_random_normal([1,2,3,3]), true)], "im2col, padded and multi-channel");
gc(def($in) { return bt.sum(bt.exp(bt.im2col($in[0], [2,2,1,1,0,0,2,2])), null, null); },
   [bt.tensor(nd_random_normal([1,1,4,4]), true)], "im2col, dilated");

// Both conv inputs at once, which is what training actually needs.
$kc = bt.tensor(nd_random_normal([1,2,2,2]), false);
gc(def($in) { return bt.sum(bt.mul(bt.conv2d($in[0], $in[1], null, 1, 0, null), $kc), null, null); },
   [bt.tensor(nd_random_normal([1,1,3,3]), true), bt.tensor(nd_random_normal([2,1,2,2]), true)],
   "conv2d, w.r.t. BOTH input and weight");
$kc2 = bt.tensor(nd_random_normal([1,2,3,3]), false);
gc(def($in) { return bt.sum(bt.mul(bt.conv2d($in[0], $in[1], $in[2], 1, 1, null), $kc2), null, null); },
   [bt.tensor(nd_random_normal([1,2,3,3]), true), bt.tensor(nd_random_normal([2,2,3,3]), true),
    bt.tensor(nd_random_normal([2]), true)],
   "conv2d with padding and bias, w.r.t. all THREE");
$kc3 = bt.tensor(nd_random_normal([1,2,2,2]), false);
gc(def($in) { return bt.sum(bt.mul(bt.conv2d($in[0], $in[1], null, 2, 1, null), $kc3), null, null); },
   [bt.tensor(nd_random_normal([1,1,4,4]), true), bt.tensor(nd_random_normal([2,1,3,3]), true)],
   "conv2d at stride 2 with padding");
$kc4 = bt.tensor(nd_random_normal([2,3,2,2]), false);
gc(def($in) { return bt.sum(bt.mul(bt.conv2d($in[0], $in[1], null, 1, 0, null), $kc4), null, null); },
   [bt.tensor(nd_random_normal([2,2,3,3]), true), bt.tensor(nd_random_normal([3,2,2,2]), true)],
   "conv2d, batched and multi-channel");
$kc5 = bt.tensor(nd_random_normal([1,2,4]), false);
gc(def($in) { return bt.sum(bt.mul(bt.conv1d($in[0], $in[1], null, 1, 0, null), $kc5), null, null); },
   [bt.tensor(nd_random_normal([1,1,5]), true), bt.tensor(nd_random_normal([2,1,2]), true)],
   "conv1d");

print("");
print("== gradcheck: pooling ==");
$kp = bt.tensor(nd_random_normal([1,1,2,2]), false);
gc(def($in) { return bt.sum(bt.mul(bt.avg_pool2d($in[0], 2, null, null, null, null, null), $kp), null, null); },
   [bt.tensor(nd_random_normal([1,1,4,4]), true)], "avg_pool2d 2x2");
gc(def($in) { return bt.sum(bt.mul(bt.max_pool2d($in[0], 2, null, null, null, null, null), $kp), null, null); },
   [bt.tensor(nd_random_normal([1,1,4,4]), true)], "max_pool2d 2x2");
$kp2 = bt.tensor(nd_random_normal([1,2,2,2]), false);
gc(def($in) { return bt.sum(bt.mul(bt.max_pool2d($in[0], 2, null, 1, null, null, null), $kp2), null, null); },
   [bt.tensor(nd_random_normal([1,2,3,3]), true)], "max_pool2d with OVERLAPPING windows");
gc(def($in) { return bt.sum(bt.mul(bt.avg_pool2d($in[0], 2, null, 1, null, null, null), $kp2), null, null); },
   [bt.tensor(nd_random_normal([1,2,3,3]), true)], "avg_pool2d with overlapping windows");

print("");
print("== gradcheck: a whole CNN, end to end ==");
// conv -> relu -> pool -> flatten -> linear -> cross_entropy, differentiated
// with respect to every parameter at once. If anything in the composition is
// wrong, this is where it shows.
$cw = bt.tensor(nd_random_normal([2,1,3,3], 0.0, 0.5), true);
$cb = bt.tensor(nd_random_normal([2], 0.0, 0.5), true);
$lw = bt.tensor(nd_random_normal([8, 3], 0.0, 0.5), true);
gc(def($in) {
       $h = bt.conv2d($in[0], $in[1], $in[2], 1, 1, null);
       $h = bt.relu($h);
       $h = bt.avg_pool2d($h, 2, null, null, null, null, null);
       $h = bt.flatten($h);
       $logits = bt.matmul($h, $in[3]);
       return nn.cross_entropy($logits, [1], null);
   },
   [bt.tensor(nd_random_normal([1,1,4,4]), true), $cw, $cb, $lw],
   "conv -> relu -> avgpool -> flatten -> linear -> cross_entropy, all 4 inputs");

print("");
print("========================================");
print("  nn layers");
print("========================================");
bt.manual_seed(5);
$conv = nn.Conv2d(3, 8, 3, 1, 1, null);
$inp = bt.tensor(nd_random_normal([2,3,8,8]), false);
eqL($conv.forward($inp).shape(), [2, 8, 8, 8], "Conv2d keeps the spatial size with padding 1");
ok(len($conv.parameters()) == 2, "Conv2d has a weight and a bias");
eqL(nd_shape($conv.param("weight").data), [8,3,3,3], "the weight is [out, in, kh, kw]");
ok(len(nn.Conv2d(3, 8, 3, 1, 1, false).parameters()) == 1, "bias=false gives one parameter");
eqL(nn.Conv2d(3, 4, [3, 5], 1, 0, null).param("weight").shape(), [4,3,3,5],
    "a [kh, kw] pair gives a non-square kernel");
eqL(nn.MaxPool2d(2, null, null).forward($inp).shape(), [2,3,4,4], "MaxPool2d as a layer");
eqL(nn.AvgPool2d(2, null, null).forward($inp).shape(), [2,3,4,4], "AvgPool2d as a layer");
eqL(nn.Conv1d(3, 6, 3, 1, 1, null).forward(bt.tensor(nd_random_normal([2,3,10]), false)).shape(),
    [2,6,10], "Conv1d as a layer");

print("");
print("== BatchNorm1d: the train/eval split ==");
$bn = nn.BatchNorm1d(3, null, null);
$bx = bt.tensor([[1.0,2.0,3.0],[3.0,4.0,5.0],[5.0,6.0,7.0]], false);
$bo = $bn.forward($bx);
ok(nd_allclose(nd_mean($bo.data, 0, null), nd_zeros([3], "f64"), 1e-12, 1e-12),
   "in train mode the output has mean 0 per feature");
ok(nd_allclose(nd_std($bo.data, 0, null, 0), nd_ones([3], "f64"), 1e-6, 1e-6),
   "and standard deviation 1");
// running_mean = (1-momentum)*0 + momentum*batch_mean = 0.1 * [3,4,5]
eqL(nd_to_list($bn.buffer("running_mean").data), [0.3, 0.4, 0.5],
    "the running mean is the momentum-weighted batch mean");
// The batch variance is 8/3; the STORED one is Bessel-corrected to 4.
close(nd_get($bn.buffer("running_var").data, [0]), 0.9 + 0.1 * 4.0, 1e-12,
      "the stored variance is Bessel-corrected (PyTorch stores the unbiased one)");
ok(len($bn.parameters()) == 2, "BatchNorm1d has gamma and beta as parameters");
// The running stats must NOT be parameters -- an optimizer stepping them would
// corrupt the statistics it is meant to be tracking.
$pids = {};
$ps = $bn.parameters();
$i = 0;
while ($i < len($ps)) { $pids[str($ps[$i].id)] = true; $i = $i + 1; }
ok($pids[str($bn.buffer("running_mean").id)] == null,
   "and the running stats are BUFFERS, so no optimizer can step them");
$bn.eval();
$eo = $bn.forward($bx);
ok(!nd_allclose(nd_mean($eo.data, 0, null), nd_zeros([3], "f64"), 1e-6, 1e-6),
   "eval() switches to the running stats, so the output is NOT re-centred");
$rmBefore = nd_copy($bn.buffer("running_mean").data, null);
$bn.forward($bx);
ok(nd_array_equal($bn.buffer("running_mean").data, $rmBefore),
   "and eval() updates nothing -- a forward pass leaves the stats untouched");
$bn.train();
$bn.forward($bx);
ok(!nd_array_equal($bn.buffer("running_mean").data, $rmBefore),
   "train() starts updating them again");
gc(def($in) { return bt.sum(bt.mul(nn.BatchNorm1d(2, null, null).forward($in[0]),
                                   bt.tensor([[1.0,2.0],[3.0,4.0],[5.0,6.0]], false)), null, null); },
   [bt.tensor(nd_random_normal([3,2]), true)], "BatchNorm1d in train mode");

print("");
print("== BatchNorm2d normalizes per channel over (N, H, W) ==");
$bn2 = nn.BatchNorm2d(2, null, null);
$b2x = bt.tensor(nd_random_normal([4,2,3,3]), false);
$b2o = $bn2.forward($b2x);
eqL($b2o.shape(), [4,2,3,3], "BatchNorm2d preserves the shape");
// Channel 0 over all of N, H and W must be centred.
$ch0 = nd_reshape(nd_ascontiguous(nd_slice($b2o.data, [null, [0,1,1], null, null])), [36]);
ok(nd_allclose(nd_mean($ch0, null, null), nd(0.0, "f64"), 1e-10, 1e-10),
   "channel 0 has mean 0 over N, H and W together");
ok(len($bn2.parameters()) == 2, "BatchNorm2d has two parameters");

print("");
print("== GELU and Softmax ==");
// PyTorch's tanh-approximation values.
close(nn.gelu(bt.tensor([1.0], false)).item(), 0.8411919906082768, 1e-12, "gelu(1) matches PyTorch");
close(nn.gelu(bt.tensor([0.0], false)).item(), 0.0, 1e-15, "gelu(0) is exactly 0");
close(nn.gelu(bt.tensor([0 - 1.0], false)).item(), 0 - 0.15880800939172324, 1e-12, "gelu(-1)");
close(nn.gelu(bt.tensor([0 - 10.0], false)).item(), 0.0, 1e-9, "gelu(-10) is ~0, not NaN");
close(nn.gelu(bt.tensor([10.0], false)).item(), 10.0, 1e-9, "gelu(10) is ~10");
gc(def($in) { return bt.sum(nn.gelu($in[0]), null, null); },
   [bt.tensor(nd_random_normal([5]), true)], "gelu");
eqL(nn.GELU().forward(bt.tensor([0.0], false)).to_list(), [0.0], "GELU as a layer");
ok(nd_allclose(nd_sum(nn.Softmax().forward(bt.tensor([[1.0,2.0,3.0]], false)).data, 1, null),
               nd_ones([1], "f64"), 1e-14, 1e-14), "Softmax as a layer sums to 1");

print("");
print("========================================");
print("  TIER 4 — stress, and one real CNN step");
print("========================================");

// A LeNet-shaped network on 28x28 input: the phase-10 MNIST architecture,
// exercised here for shape and gradient flow rather than accuracy.
bt.manual_seed(11);
$cnn = nn.Sequential([
    nn.Conv2d(1, 6, 5, 1, 2, null),   nn.ReLU(), nn.MaxPool2d(2, null, null),
    nn.Conv2d(6, 16, 5, 1, 0, null),  nn.ReLU(), nn.MaxPool2d(2, null, null),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120, null), nn.ReLU(),
    nn.Linear(120, 84, null),         nn.ReLU(),
    nn.Linear(84, 10, null)
]);
$batch = bt.tensor(nd_random_normal([8, 1, 28, 28]), false);
$t0 = clock();
$logits = $cnn.forward($batch);
$t1 = clock();
eqL($logits.shape(), [8, 10], "LeNet on 8x1x28x28 gives [8, 10]");
ok($cnn.num_parameters() == 61706, "and has exactly LeNet-5's 61,706 parameters");
print("        LeNet forward (batch 8): " + str($t1 - $t0) + " ms");

$labels = nd_astype(nd_random_int([8], 0, 10), "i64");
$opt = optim.SGD($cnn.parameters(), 0.01, 0.9, null);
$t2 = clock();
$loss = nn.cross_entropy($logits, $labels, null);
$loss.backward(null);
$opt.step();
$t3 = clock();
print("        LeNet backward + step:   " + str($t3 - $t2) + " ms");
ok($loss.item() > 0.0, "the loss is positive");
// Every parameter must have received a gradient. A shape error anywhere in the
// composition would leave one of them at null.
$missing = 0;
$ps = $cnn.parameters();
$i = 0;
while ($i < len($ps)) {
    if ($ps[$i].grad == null) { $missing = $missing + 1; }
    $i = $i + 1;
}
ok($missing == 0, "every one of LeNet's " + str(len($ps)) + " parameter tensors got a gradient");

// Ten steps on ONE fixed batch must drive the loss down. This proves the
// plumbing, not correctness (DECISIONS B10 -- a loss going down proves
// nothing on its own), which is why every gradient above was gradchecked.
$first = 0.0;
$last = 0.0;
$step = 0;
while ($step < 10) {
    $opt.zero_grad();
    $l = nn.cross_entropy($cnn.forward($batch), $labels, null);
    if ($step == 0) { $first = $l.item(); }
    $last = $l.item();
    $l.backward(null);
    $opt.step();
    $step = $step + 1;
}
print("        10 steps on one batch: loss " + str($first) + " -> " + str($last));
ok($last < $first, "10 steps on one batch reduce the loss (the plumbing works end to end)");

// No leaks across repeated conv forward/backward.
$before = nd_live_bytes();
$small = nn.Conv2d(2, 3, 3, 1, 1, null);
$sx = bt.tensor(nd_random_normal([2,2,6,6]), false);
$n = 0;
while ($n < 200) {
    $small.zero_grad();
    $small.forward($sx).sum(null, null).backward(null);
    $n = $n + 1;
}
$after = nd_live_bytes();
print("        live array bytes over 200 conv rounds: " + str($before) + " -> " + str($after));
ok($after - $before < 2000000, "200 conv forward/backward rounds grew live bytes by under 2MB");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
