// ════════════════════════════════════════════════════════════════════════════
//  btorch.b — tensors and reverse-mode automatic differentiation for Bantu.
//
//  A PyTorch-shaped deep-learning library. The API is pure Bantu; the inner
//  loops are numba's native nd_* kernels.
//
//      include "btorch" as bt;
//
//      $x = bt.tensor([[1.0, 2.0], [3.0, 4.0]], true);
//      $w = bt.randn([2, 3], true);
//      $y = $x.matmul($w).relu().sum(null, null);
//      $y.backward(null);
//      print($w.grad);
//
//  Operators work too, because a class can now overload them:
//      $z = $a * $b + $c;
//
//  Design: docs/btorch-architecture.md.  Decisions: btorch-suite/DECISIONS.md.
//  Requires an interpreter with the native ndarray atoms (has_native("ndarray")).
// ════════════════════════════════════════════════════════════════════════════

$_BT_OK = false;
try { $_BT_OK = has_native("ndarray"); } catch ($e) { $_BT_OK = false; }
def _need() {
    if (!$_BT_OK) {
        throw "btorch: this interpreter lacks the native ndarray atoms "
            + "(need has_native('ndarray')). Rebuild Bantu.";
    }
    return null;
}

// ── module state ────────────────────────────────────────────────────────────
// A dict, not a plain variable: a function body is an assignment boundary, so
// `$_BT_GRAD = false` inside a function would make a local and the flag would
// never change. Mutating a field of a dict reaches the real object.
$_BT = {"grad": true, "nextid": 0};

def _nextId() { $_BT.nextid = $_BT.nextid + 1; return $_BT.nextid; }

// Is grad recording on? Ops consult this before attaching a tape node.
def is_grad_enabled() { return $_BT.grad; }

// Run $fn with gradient recording off, and restore the previous setting
// afterwards -- INCLUDING when $fn throws. An error inside a no_grad block
// must not leave the whole process with autograd silently disabled.
def no_grad($fn) {
    $prev = $_BT.grad;
    $_BT.grad = false;
    $out = null;
    try { $out = $fn(); }
    catch ($e) { $_BT.grad = $prev; throw $e; }
    $_BT.grad = $prev;
    return $out;
}
// Alias: reads better around an evaluation pass.
def inference_mode($fn) { return no_grad($fn); }

// Run $fn with recording ON even inside a no_grad block.
def enable_grad($fn) {
    $prev = $_BT.grad;
    $_BT.grad = true;
    $out = null;
    try { $out = $fn(); }
    catch ($e) { $_BT.grad = $prev; throw $e; }
    $_BT.grad = $prev;
    return $out;
}

// ── helpers ─────────────────────────────────────────────────────────────────

// Is this value a native ndarray handle? type() returns the handle's TAG --
// "ndarray" -- not the word "handle", which is what the first version of this
// checked for. The effect was that every array handed in as a raw array fell
// through to nd(), which refused it, so Linear could not initialize at all.
def _isNd($x) { return type($x) == "ndarray"; }

// Coerce anything a user might pass as an operand into a plain NdArray.
// A Tensor contributes its data; a list or number is built into an f64 array.
def _arr($x) {
    if ($x == null) { throw "btorch: expected a tensor or a number, got null"; }
    if (type($x) == "instance") { return $x.data; }
    if (_isNd($x)) { return $x; }
    return nd([$x], "f64");
}

// Is this value one of our Tensors?
def is_tensor($x) {
    if (type($x) != "instance") { return false; }
    // Duck-typed on the fields the engine needs, so a Tensor from a different
    // include of this module still counts.
    return $x.__bt_tensor == true;
}

// Everything in btorch computes in f64 (DECISIONS B6). An i64 array -- which
// is what nd_arange and nd_sum-of-integers produce -- would silently truncate
// a gradient to zero, so it is converted on the way in rather than later.
def _f64($a) {
    if (nd_dtype($a) == "f64") { return $a; }
    return nd_astype($a, "f64");
}

// ── the gradient of a broadcast (architecture doc section 4.5) ──────────────
// If forward broadcast a [3] bias against a [128,3] activation, the gradient
// arriving is [128,3] and the bias's gradient must be [3] -- SUMMED over the
// broadcast axes. Reshaping or slicing instead is the single most common way a
// hand-written autodiff engine goes quietly wrong: the shapes line up, the
// numbers do not.
def _unbroadcast($g, $shape) {
    if (nd_shape($g) == $shape) { return $g; }
    // 1. axes that broadcasting ADDED on the left: sum them away entirely.
    while (len(nd_shape($g)) > len($shape)) { $g = nd_sum($g, 0, null); }
    // 2. axes that were stretched from 1: sum them but keep the axis, so the
    //    positions of the remaining axes do not shift under us.
    $i = 0;
    while ($i < len($shape)) {
        $gs = nd_shape($g);
        if ($shape[$i] == 1 && $gs[$i] > 1) { $g = nd_sum($g, $i, true); }
        $i = $i + 1;
    }
    return nd_reshape($g, $shape);
}

// ── the tape ────────────────────────────────────────────────────────────────

class Node {
    // $bw is a closure (gradOut) -> list of gradients, one per parent,
    // positionally, with null for a parent that needs none.
    def init($name, $parents, $bw) {
        this.name = $name;
        this.parents = $parents;
        this.bw = $bw;
    }
}

class Tensor {
    def init($data, $requires_grad) {
        this.__bt_tensor = true;
        this.id = _nextId();
        this.data = _f64($data);
        this.grad = null;
        this.requires_grad = $requires_grad == true;
        this.ctx = null;            // the Node that produced this tensor
        this.is_leaf = true;
        this.retains_grad = false;
    }

    // ── introspection ───────────────────────────────────────────────────
    def shape()   { return nd_shape(this.data); }
    def ndim()    { return nd_ndim(this.data); }
    def size()    { return nd_size(this.data); }
    def dtype()   { return nd_dtype(this.data); }
    def to_list() { return nd_to_list(this.data); }
    def numel()   { return nd_size(this.data); }

    // The single value of a one-element tensor, as a plain Bantu number.
    def item() {
        if (nd_size(this.data) != 1) {
            throw "btorch: item() needs a one-element tensor, but this one has "
                + str(nd_size(this.data)) + " elements (shape " + str(nd_shape(this.data)) + ")";
        }
        return nd_get(nd_reshape(this.data, [1]), [0]);
    }

    def show() {
        $s = "tensor(" + str(nd_to_list(this.data));
        if (this.requires_grad) { $s = $s + ", requires_grad=true"; }
        if (this.ctx != null)   { $s = $s + ", from=" + this.ctx.name; }
        return $s + ")";
    }

    // ── graph control ───────────────────────────────────────────────────
    // Detach from the tape: same numbers, no history, no gradient.
    def detach() { return new Tensor(this.data, false); }

    // Ask for .grad to be filled on a NON-leaf tensor (off by default, as in
    // PyTorch, because keeping every intermediate gradient is usually waste).
    def retain_grad() { this.retains_grad = true; return this; }

    def requires_grad_($flag) {
        if (!this.is_leaf) {
            throw "btorch: requires_grad_() can only be set on a leaf tensor; "
                + "this one was produced by " + this.ctx.name + "()";
        }
        this.requires_grad = $flag == true;
        return this;
    }

    def zero_grad() { this.grad = null; return this; }

    // ── backward ────────────────────────────────────────────────────────
    def backward($gradient) { return _backward(this, $gradient, false); }
    def backward_retain($gradient) { return _backward(this, $gradient, true); }

    // ── operators (these work because a class can overload them now) ────
    def __add__($o)      { return add(this, $o); }
    def __radd__($o)     { return add($o, this); }
    def __sub__($o)      { return sub(this, $o); }
    def __rsub__($o)     { return sub($o, this); }
    def __mul__($o)      { return mul(this, $o); }
    def __rmul__($o)     { return mul($o, this); }
    def __truediv__($o)  { return div(this, $o); }
    def __rtruediv__($o) { return div($o, this); }
    def __neg__()        { return neg(this); }

    // ── arithmetic as methods ───────────────────────────────────────────
    def add($o) { return add(this, $o); }
    def sub($o) { return sub(this, $o); }
    def mul($o) { return mul(this, $o); }
    def div($o) { return div(this, $o); }
    def pow($o) { return pow(this, $o); }
    def neg()   { return neg(this); }

    // ── unary maths ─────────────────────────────────────────────────────
    def exp()     { return exp(this); }
    def log()     { return log(this); }
    def sqrt()    { return sqrt(this); }
    def tanh()    { return tanh(this); }
    def sigmoid() { return sigmoid(this); }
    def relu()    { return relu(this); }
    def abs()     { return abs_(this); }

    // ── reductions ──────────────────────────────────────────────────────
    def sum($axis, $keepdims)  { return sum(this, $axis, $keepdims); }
    def mean($axis, $keepdims) { return mean(this, $axis, $keepdims); }
    def max($axis, $keepdims)  { return max(this, $axis, $keepdims); }
    def min($axis, $keepdims)  { return min(this, $axis, $keepdims); }
    def argmax($axis)          { return argmax(this, $axis); }
    def argmin($axis)          { return argmin(this, $axis); }
    def maximum($o)            { return maximum(this, $o); }
    def minimum($o)            { return minimum(this, $o); }
    def clamp_min($v)          { return maximum(this, $v); }
    def clamp_max($v)          { return minimum(this, $v); }

    // ── indexing ────────────────────────────────────────────────────────
    def take($indices, $axis)        { return take(this, $indices, $axis); }
    def index_select($axis, $idx)    { return take(this, $idx, $axis); }
    def index_add($idx, $v, $axis)   { return index_add(this, $idx, $v, $axis); }

    // ── shape ───────────────────────────────────────────────────────────
    def reshape($shape) { return reshape(this, $shape); }
    def view($shape)    { return reshape(this, $shape); }
    def t()             { return transpose(this); }
    def matmul($o)      { return matmul(this, $o); }
    def mm($o)          { return matmul(this, $o); }
    def permute($axes)  { return permute(this, $axes); }
    def pad($p, $v)     { return pad(this, $p, $v); }
    def tile($r)        { return tile(this, $r); }
    def repeat_interleave($r, $ax) { return repeat_interleave(this, $r, $ax); }
    def gather($dim, $i){ return gather(this, $dim, $i); }
    def chunk($n, $ax)  { return chunk(this, $n, $ax); }
    def split($s, $ax)  { return split(this, $s, $ax); }
    def squeeze($axis)  { return squeeze(this, $axis); }
    def unsqueeze($ax)  { return unsqueeze(this, $ax); }
    def movedim($s, $d) { return movedim(this, $s, $d); }
    def flatten()       { return flatten(this); }
}

// ── constructing a result and attaching its tape node ───────────────────────
// Every op funnels through here, so the rules about WHEN a graph is built live
// in exactly one place.
def _mk($data, $parents, $name, $bw) {
    $out = new Tensor($data, false);
    if (!$_BT.grad) { return $out; }
    $need = false;
    $i = 0;
    while ($i < len($parents)) {
        $p = $parents[$i];
        if ($p != null && $p.requires_grad) { $need = true; }
        $i = $i + 1;
    }
    if (!$need) { return $out; }
    $out.requires_grad = true;
    $out.is_leaf = false;
    $out.ctx = new Node($name, $parents, $bw);
    return $out;
}

// Wrap a non-Tensor operand so the graph has a uniform parent list. A constant
// gets requires_grad=false, so it ends the chain without special-casing.
def _asT($x) {
    if (is_tensor($x)) { return $x; }
    return new Tensor(_arr($x), false);
}

// ════════════════════════════════════════════════════════════════════════════
//  backward()
//
//  Two things here are load-bearing:
//
//  1. The topological sort is ITERATIVE, with an explicit stack. The
//     interpreter caps recursion at 1000 calls and says so by name, and a
//     recursive DFS would therefore die on any graph deeper than that -- an
//     RNN unrolled 100 steps over 10 ops reaches it. The error would arrive
//     during training, not during a unit test.
//
//  2. The sweep runs in REVERSE topological order, not plain DFS order. A
//     tensor used twice must receive BOTH contributions before it propagates
//     any of them. `$y = $x.mul($x).add($x)` is the smallest graph that tells
//     a correct engine from one that merely looks correct.
// ════════════════════════════════════════════════════════════════════════════
def _backward($root, $gradient, $retain) {
    _need();
    if (!$root.requires_grad) {
        throw "btorch: backward() on a tensor that does not require grad "
            + "(nothing upstream of it was marked requires_grad)";
    }

    // ── the seed ────────────────────────────────────────────────────────
    $seed = null;
    if ($gradient == null) {
        if (nd_size($root.data) != 1) {
            throw "btorch: backward() on a non-scalar tensor (shape "
                + str(nd_shape($root.data)) + ") needs an explicit gradient of "
                + "the same shape -- call .backward($g), or reduce to a scalar "
                + "first with .sum(null, null) or .mean(null, null)";
        }
        $seed = nd_ones(nd_shape($root.data), "f64");
    } else {
        $seed = _f64(_arr($gradient));
        if (nd_shape($seed) != nd_shape($root.data)) {
            throw "btorch: backward() was given a gradient of shape "
                + str(nd_shape($seed)) + " for a tensor of shape "
                + str(nd_shape($root.data));
        }
    }

    // ── iterative post-order toposort ───────────────────────────────────
    $topo = [];
    $expanded = {};     // id -> true once its parents have been pushed
    $emitted = {};      // id -> true once it is in $topo
    $stack = [$root];
    while (len($stack) > 0) {
        $n = $stack[len($stack) - 1];
        $k = str($n.id);
        if ($expanded[$k] == null) {
            $expanded[$k] = true;
            if ($n.ctx != null) {
                $ps = $n.ctx.parents;
                $i = 0;
                while ($i < len($ps)) {
                    $p = $ps[$i];
                    if ($p != null && $expanded[str($p.id)] == null) {
                        push($stack, $p);
                    }
                    $i = $i + 1;
                }
            }
        } else {
            // Parents are all dealt with: this node's turn.
            // pop() mutates the list in place and returns the ELEMENT, where
            // push() returns the list -- so `$stack = pop($stack)` would
            // assign a Tensor to $stack. The result is discarded on purpose.
            pop($stack);
            if ($emitted[$k] == null) {
                $emitted[$k] = true;
                push($topo, $n);
            }
        }
    }

    // ── the reverse sweep ───────────────────────────────────────────────
    // Intermediate gradients live in a map keyed by tensor id. Only leaves
    // (and anything that asked via retain_grad) get a .grad field, which is
    // PyTorch's behaviour and keeps the memory down.
    $g = {};
    $g[str($root.id)] = $seed;

    $i = len($topo) - 1;
    while ($i >= 0) {
        $n = $topo[$i];
        $k = str($n.id);
        $gn = $g[$k];
        if ($gn != null && $n.ctx != null) {
            $bw = $n.ctx.bw;                 // the closure, into a local first
            $outs = $bw($gn);
            $ps = $n.ctx.parents;
            if (len($outs) != len($ps)) {
                throw "btorch: the backward of " + $n.ctx.name + " returned "
                    + str(len($outs)) + " gradients for " + str(len($ps))
                    + " inputs -- it must return exactly one per input, "
                    + "with null for an input that needs none";
            }
            $j = 0;
            while ($j < len($ps)) {
                $p = $ps[$j];
                if ($p != null && $p.requires_grad) {
                    $pg = $outs[$j];
                    if ($pg != null) {
                        // The shape contract: the gradient for a parent has
                        // exactly that parent's shape. Enforcing it here is
                        // what makes a forgotten _unbroadcast a loud failure
                        // instead of a model that trains to nonsense.
                        $pg = _unbroadcast($pg, nd_shape($p.data));
                        $pk = str($p.id);
                        if ($g[$pk] == null) { $g[$pk] = $pg; }
                        else { $g[$pk] = nd_add($g[$pk], $pg, null); }
                    }
                }
                $j = $j + 1;
            }
        }
        $i = $i - 1;
    }

    // ── land the gradients ──────────────────────────────────────────────
    // Accumulated with +=, as in PyTorch: two backward() calls without a
    // zero_grad() in between sum, which is how gradient accumulation over
    // micro-batches is meant to work.
    $i = 0;
    while ($i < len($topo)) {
        $n = $topo[$i];
        if ($n.requires_grad) {
            if ($n.is_leaf || $n.retains_grad) {
                $gv = $g[str($n.id)];
                if ($gv != null) {
                    if ($n.grad == null) { $n.grad = nd_copy($gv, null); }
                    else { $n.grad = nd_add($n.grad, $gv, null); }
                }
            }
        }
        $i = $i + 1;
    }

    // ── free the graph ──────────────────────────────────────────────────
    // Unless asked to keep it. Dropping ctx releases the backward closures and
    // with them the saved arrays, which is what keeps a training loop's memory
    // flat over thousands of steps.
    if (!$retain) {
        $i = 0;
        while ($i < len($topo)) {
            $n = $topo[$i];
            if (!$n.is_leaf) { $n.ctx = null; }
            $i = $i + 1;
        }
    }
    return null;
}

// ════════════════════════════════════════════════════════════════════════════
//  The ops.
//
//  EVERY backward closure is produced by a `_*Backward(...)` factory whose
//  parameters hold the saved values. This is not a style preference -- Bantu
//  closures capture by REFERENCE, so a closure written inline inside a loop
//  sees whatever its captured variable ends up as, not what it held when the
//  closure was made. Probed and measured: three closures built over a loop
//  variable returned the final value three times. Inline backward closures
//  would therefore produce silently wrong gradients, which is the hardest
//  possible bug to find. A factory's parameters are bound once per call.
//  (DECISIONS B3; tests/btorch_autograd_test.b has the ten-nodes-in-a-loop
//  case that this rule exists to prevent.)
//
//  Factories capture NdArrays, never Tensors: a closure holding a Tensor that
//  holds the Node that holds the closure is a reference cycle, and the
//  interpreter is refcounted with no cycle collector.
// ════════════════════════════════════════════════════════════════════════════

// ── add ─────────────────────────────────────────────────────────────────────
def _addBackward() {
    return def($gr) { return [$gr, $gr]; };
}
def add($a, $b) {
    _need();
    $ta = _asT($a); $tb = _asT($b);
    return _mk(nd_add($ta.data, $tb.data, null), [$ta, $tb], "add", _addBackward());
}

// ── sub ─────────────────────────────────────────────────────────────────────
def _subBackward() {
    return def($gr) { return [$gr, nd_negative($gr, null)]; };
}
def sub($a, $b) {
    _need();
    $ta = _asT($a); $tb = _asT($b);
    return _mk(nd_subtract($ta.data, $tb.data, null), [$ta, $tb], "sub", _subBackward());
}

// ── mul ─────────────────────────────────────────────────────────────────────
def _mulBackward($ad, $bd) {
    return def($gr) {
        return [nd_multiply($gr, $bd, null), nd_multiply($gr, $ad, null)];
    };
}
def mul($a, $b) {
    _need();
    $ta = _asT($a); $tb = _asT($b);
    return _mk(nd_multiply($ta.data, $tb.data, null), [$ta, $tb], "mul",
               _mulBackward($ta.data, $tb.data));
}

// ── div ─────────────────────────────────────────────────────────────────────
// d(a/b)/da = 1/b ;  d(a/b)/db = -a/b^2
def _divBackward($ad, $bd) {
    return def($gr) {
        $ga = nd_divide($gr, $bd, null);
        $gb = nd_negative(nd_divide(nd_multiply($gr, $ad, null),
                                    nd_multiply($bd, $bd, null), null), null);
        return [$ga, $gb];
    };
}
def div($a, $b) {
    _need();
    $ta = _asT($a); $tb = _asT($b);
    return _mk(nd_divide($ta.data, $tb.data, null), [$ta, $tb], "div",
               _divBackward($ta.data, $tb.data));
}

// ── pow ─────────────────────────────────────────────────────────────────────
// d(a^b)/da = b*a^(b-1) ;  d(a^b)/db = a^b * ln(a)
// The exponent's gradient needs ln(a), which is undefined for a <= 0. Where
// the exponent is a constant -- the overwhelmingly common case -- that
// gradient is never asked for, so it is computed lazily rather than producing
// a NaN that would poison an unrelated branch.
def _powBackward($ad, $bd, $outd, $bNeeds) {
    return def($gr) {
        $ga = nd_multiply($gr, nd_multiply($bd,
                  nd_power($ad, nd_subtract($bd, 1.0, null), null), null), null);
        $gb = null;
        if ($bNeeds) {
            $gb = nd_multiply($gr, nd_multiply($outd, nd_log($ad, null), null), null);
        }
        return [$ga, $gb];
    };
}
def pow($a, $b) {
    _need();
    $ta = _asT($a); $tb = _asT($b);
    $out = nd_power($ta.data, $tb.data, null);
    return _mk($out, [$ta, $tb], "pow",
               _powBackward($ta.data, $tb.data, $out, $tb.requires_grad));
}

// ── neg ─────────────────────────────────────────────────────────────────────
def _negBackward() { return def($gr) { return [nd_negative($gr, null)]; }; }
def neg($a) {
    _need();
    $ta = _asT($a);
    return _mk(nd_negative($ta.data, null), [$ta], "neg", _negBackward());
}

// ── exp ─────────────────────────────────────────────────────────────────────
// d(exp a) = exp(a) = out, so the OUTPUT is saved rather than the input: one
// fewer exp() in the backward pass.
def _expBackward($outd) {
    return def($gr) { return [nd_multiply($gr, $outd, null)]; };
}
def exp($a) {
    _need();
    $ta = _asT($a);
    $out = nd_exp($ta.data, null);
    return _mk($out, [$ta], "exp", _expBackward($out));
}

// ── log ─────────────────────────────────────────────────────────────────────
def _logBackward($ad) {
    return def($gr) { return [nd_divide($gr, $ad, null)]; };
}
def log($a) {
    _need();
    $ta = _asT($a);
    return _mk(nd_log($ta.data, null), [$ta], "log", _logBackward($ta.data));
}

// ── sqrt ────────────────────────────────────────────────────────────────────
// d(sqrt a) = 1/(2*sqrt a) = 1/(2*out)
def _sqrtBackward($outd) {
    return def($gr) {
        return [nd_divide($gr, nd_multiply($outd, 2.0, null), null)];
    };
}
def sqrt($a) {
    _need();
    $ta = _asT($a);
    $out = nd_sqrt($ta.data, null);
    return _mk($out, [$ta], "sqrt", _sqrtBackward($out));
}

// ── abs ─────────────────────────────────────────────────────────────────────
// d|a| = sign(a). At exactly 0 the derivative does not exist; sign(0) = 0 is
// the subgradient PyTorch uses.
def _absBackward($ad) {
    return def($gr) { return [nd_multiply($gr, nd_sign($ad, null), null)]; };
}
def abs_($a) {
    _need();
    $ta = _asT($a);
    return _mk(nd_abs($ta.data, null), [$ta], "abs", _absBackward($ta.data));
}

// ── tanh ────────────────────────────────────────────────────────────────────
// d(tanh a) = 1 - tanh(a)^2 = 1 - out^2
def _tanhBackward($outd) {
    return def($gr) {
        return [nd_multiply($gr, nd_subtract(1.0,
                    nd_multiply($outd, $outd, null), null), null)];
    };
}
def tanh($a) {
    _need();
    $ta = _asT($a);
    $out = nd_tanh($ta.data, null);
    return _mk($out, [$ta], "tanh", _tanhBackward($out));
}

// ── sigmoid ─────────────────────────────────────────────────────────────────
// Computed as 0.5*(1+tanh(x/2)) rather than 1/(1+exp(-x)): for x around -800,
// exp(-x) overflows to +inf and the naive form gives 0 with a NaN gradient,
// while tanh saturates cleanly. d(sigmoid) = out*(1-out).
def _sigmoidBackward($outd) {
    return def($gr) {
        return [nd_multiply($gr, nd_multiply($outd,
                    nd_subtract(1.0, $outd, null), null), null)];
    };
}
def sigmoid($a) {
    _need();
    $ta = _asT($a);
    $out = nd_multiply(nd_add(nd_tanh(nd_multiply($ta.data, 0.5, null), null),
                              1.0, null), 0.5, null);
    return _mk($out, [$ta], "sigmoid", _sigmoidBackward($out));
}

// ── relu ────────────────────────────────────────────────────────────────────
// The gradient is gated on the INPUT being > 0, not the output being > 0. They
// agree here, but using the input is the definition and survives a change of
// formulation. At exactly 0 the subgradient is 0, as in PyTorch.
def _reluBackward($ad) {
    return def($gr) {
        $mask = nd_astype(nd_greater($ad, 0.0, null), "f64");
        return [nd_multiply($gr, $mask, null)];
    };
}
def relu($a) {
    _need();
    $ta = _asT($a);
    return _mk(nd_maximum($ta.data, 0.0, null), [$ta], "relu", _reluBackward($ta.data));
}

// ── leaky_relu ──────────────────────────────────────────────────────────────
def _leakyBackward($ad, $slope) {
    return def($gr) {
        $pos = nd_astype(nd_greater($ad, 0.0, null), "f64");
        $neg = nd_multiply(nd_astype(nd_less_equal($ad, 0.0, null), "f64"), $slope, null);
        return [nd_multiply($gr, nd_add($pos, $neg, null), null)];
    };
}
def leaky_relu($a, $slope) {
    _need();
    if ($slope == null) { $slope = 0.01; }
    $ta = _asT($a);
    $out = nd_add(nd_maximum($ta.data, 0.0, null),
                  nd_multiply(nd_minimum($ta.data, 0.0, null), $slope, null), null);
    return _mk($out, [$ta], "leaky_relu", _leakyBackward($ta.data, $slope));
}

// ── sum ─────────────────────────────────────────────────────────────────────
// The gradient of a sum is the incoming gradient broadcast back over every
// element that went into it. With an axis and keepdims=false the summed axis
// has been dropped, so it is put back before broadcasting -- otherwise the
// broadcast would align from the wrong end.
def _sumBackward($shape, $axis) {
    return def($gr) {
        $g = $gr;
        if ($axis != null) {
            $ks = $shape;
            $i = 0;
            $target = [];
            while ($i < len($ks)) {
                if ($i == $axis) { push($target, 1); }
                else { push($target, $ks[$i]); }
                $i = $i + 1;
            }
            $g = nd_reshape($g, $target);
        }
        return [nd_multiply(nd_ones($shape, "f64"), $g, null)];
    };
}
def sum($a, $axis, $keepdims) {
    _need();
    $ta = _asT($a);
    $out = nd_sum($ta.data, $axis, $keepdims);
    return _mk($out, [$ta], "sum", _sumBackward(nd_shape($ta.data), $axis));
}

// ── mean ────────────────────────────────────────────────────────────────────
// Same as sum, divided by the number of elements that were averaged.
def _meanBackward($shape, $axis, $n) {
    return def($gr) {
        $g = $gr;
        if ($axis != null) {
            $i = 0;
            $target = [];
            while ($i < len($shape)) {
                if ($i == $axis) { push($target, 1); }
                else { push($target, $shape[$i]); }
                $i = $i + 1;
            }
            $g = nd_reshape($g, $target);
        }
        return [nd_divide(nd_multiply(nd_ones($shape, "f64"), $g, null), $n, null)];
    };
}
def mean($a, $axis, $keepdims) {
    _need();
    $ta = _asT($a);
    $shape = nd_shape($ta.data);
    $n = nd_size($ta.data);
    if ($axis != null) { $n = $shape[$axis]; }
    $out = nd_mean($ta.data, $axis, $keepdims);
    return _mk($out, [$ta], "mean", _meanBackward($shape, $axis, $n));
}

// ── reshape / view ──────────────────────────────────────────────────────────
def _reshapeBackward($shape) {
    return def($gr) { return [nd_reshape($gr, $shape)]; };
}
def reshape($a, $shape) {
    _need();
    $ta = _asT($a);
    return _mk(nd_reshape($ta.data, $shape), [$ta], "reshape",
               _reshapeBackward(nd_shape($ta.data)));
}
def flatten($a) {
    _need();
    $ta = _asT($a);
    $s = nd_shape($ta.data);
    if (len($s) < 2) { return reshape($ta, [nd_size($ta.data)]); }
    $rest = 1; $i = 1;
    while ($i < len($s)) { $rest = $rest * $s[$i]; $i = $i + 1; }
    return reshape($ta, [$s[0], $rest]);
}

// ── transpose ───────────────────────────────────────────────────────────────
def _transposeBackward() {
    return def($gr) { return [nd_transpose($gr, null)]; };
}
def transpose($a) {
    _need();
    $ta = _asT($a);
    return _mk(nd_transpose($ta.data, null), [$ta], "transpose", _transposeBackward());
}

// ── matmul ──────────────────────────────────────────────────────────────────
// dL/dA = G @ B^T ;  dL/dB = A^T @ G
//
// The results are then passed through _unbroadcast by the engine, which is
// what makes a batched matmul against a 2-D weight work: the [B,n,k] gradient
// for a [n,k] weight gets summed over the batch axis rather than reshaped.
def _matmulBackward($ad, $bd) {
    return def($gr) {
        $ga = nd_matmul($gr, nd_transpose($bd, null), null);
        $gb = nd_matmul(nd_transpose($ad, null), $gr, null);
        return [$ga, $gb];
    };
}
def matmul($a, $b) {
    _need();
    $ta = _asT($a); $tb = _asT($b);
    return _mk(nd_matmul($ta.data, $tb.data, null), [$ta, $tb], "matmul",
               _matmulBackward($ta.data, $tb.data));
}

// ════════════════════════════════════════════════════════════════════════════
//  Indexing (phase 2)
//
//  Every gather here is linear, so its backward is the gather's ADJOINT: a
//  scatter back into a zero array of the input's shape. The scatter has to
//  ACCUMULATE, because an index may repeat -- a batch that mentions the same
//  token twice, a gather that reads one row three times. nd_put would be the
//  obvious tool and would be silently wrong: its last write wins, so every
//  repeat but the final one would vanish from the gradient. That is the whole
//  reason phase A5 added nd_index_add (DECISIONS.md B15), and the adjoint
//  identity that pins the two together is asserted in
//  tests/numba_index_add_test.b.
// ════════════════════════════════════════════════════════════════════════════

// Indices are snapshotted to an i64 array at call time. The backward closure
// runs long after the forward -- possibly after the caller has reused its
// index buffer for the next batch -- and nd_astype copies even when the dtype
// already matches, so the snapshot is the same cost as the validation.
// NOT built on _arr: _arr wraps its argument one list deeper (nd([$x]) so a
// bare number becomes shape [1]), which turns a flat index list [2,0,1] into a
// [1,3] row and every downstream shape check into a confusing error.
def _idx($x) {
    if (is_tensor($x)) { return nd_astype($x.data, "i64"); }
    if (_isNd($x))     { return nd_astype($x, "i64"); }
    if (type($x) == "list") { return nd_astype(nd($x, "i64"), "i64"); }
    if (type($x) == "number") { return nd([$x], "i64"); }
    throw "btorch: indices must be a list, an array or a tensor (got "
        + type($x) + ")";
}

// ── take / index_select ─────────────────────────────────────────────────────
def _takeBackward($shape, $idx, $axis) {
    return def($gr) {
        $g = nd_zeros($shape, "f64");
        nd_index_add($g, $idx, $gr, $axis);
        return [$g];
    };
}
// `axis` is required on purpose. numba's nd_take defaults to the LAST axis
// (NumPy's rule) while every caller in a neural net means axis 0, so either
// default would be a trap for half the callers. Saying it is cheaper than
// debugging it.
def take($a, $indices, $axis) {
    _need();
    if ($axis == null) {
        throw "btorch.take(a, indices, axis): axis is required -- "
            + "nd_take defaults to the last axis and embeddings want axis 0, "
            + "so there is no safe default to pick for you";
    }
    $ta = _asT($a);
    $idx = _idx($indices);
    return _mk(nd_take($ta.data, $idx, $axis), [$ta], "take",
               _takeBackward(nd_shape($ta.data), $idx, $axis));
}
// PyTorch's name and argument order.
def index_select($a, $axis, $indices) { return take($a, $indices, $axis); }

// ── embedding ───────────────────────────────────────────────────────────────
// A row lookup, which is take along axis 0. Nothing more is needed: the
// accumulate in _takeBackward is exactly what makes a repeated token add its
// two gradients together instead of keeping one.
def embedding($weight, $ids) {
    _need();
    return take(_asT($weight), $ids, 0);
}

// ── one_hot ─────────────────────────────────────────────────────────────────
// A constant, so it carries no gradient. Built with one native scatter rather
// than a per-row interpreter loop: this is on the path of every
// classification step.
def one_hot($ids, $classes) {
    _need();
    $idx = _idx($ids);
    $n = nd_size($idx);
    if (len(nd_shape($idx)) != 1) {
        throw "btorch.one_hot: expected a flat list of class indices, got shape "
            + str(nd_shape($idx));
    }
    if ($n > 0) {
        $lo = nd_get(nd_reshape(nd_min($idx, null, null), [1]), [0]);
        $hi = nd_get(nd_reshape(nd_max($idx, null, null), [1]), [0]);
        if ($lo < 0 || $hi >= $classes) {
            // Only now is it worth finding WHICH row is wrong.
            $bad = nd_logical_or(nd_less($idx, 0, null),
                                 nd_greater_equal($idx, $classes, null), null);
            $row = nd_get(nd_reshape(nd_argmax($bad, null, null), [1]), [0]);
            throw "btorch.one_hot: class index " + str(nd_get($idx, [$row]))
                + " at position " + str($row) + " is out of range for "
                + str($classes) + " classes";
        }
    }
    $out = nd_zeros([$n, $classes], "f64");
    if ($n > 0) {
        // Flat positions row*classes + class, computed in one vector pass.
        $rows = nd_multiply(nd_astype(nd_arange(0.0, $n * 1.0, null), "f64"),
                            $classes * 1.0, null);
        $pos = nd_astype(nd_add($rows, nd_astype($idx, "f64"), null), "i64");
        nd_index_add(nd_reshape($out, [$n * $classes]), $pos, 1.0, 0);
    }
    return new Tensor($out, false);
}

// ── index_add (the scatter itself, as an op) ────────────────────────────────
// out = a with `values` added at `indices` along `axis`. Both inputs carry
// gradient: a's is the incoming gradient untouched (it passed straight
// through), and values' is the gradient GATHERED back from where it was
// scattered -- take is index_add's adjoint in this direction too.
def _indexAddBackward($idx, $axis) {
    return def($gr) {
        return [$gr, nd_take($gr, $idx, $axis)];
    };
}
def index_add($a, $indices, $values, $axis) {
    _need();
    if ($axis == null) { throw "btorch.index_add(a, indices, values, axis): axis is required"; }
    $ta = _asT($a); $tv = _asT($values);
    $idx = _idx($indices);
    // nd_copy first: the op is functional at the btorch level even though the
    // kernel is in place, because a tape node whose input was mutated under it
    // cannot produce a correct gradient on a second backward pass.
    $out = nd_copy($ta.data);
    nd_index_add($out, $idx, $tv.data, $axis);
    return _mk($out, [$ta, $tv], "index_add", _indexAddBackward($idx, $axis));
}

// ── max / min ───────────────────────────────────────────────────────────────
// amax/amin semantics: where several elements tie for the extreme, the
// gradient is SPLIT EVENLY between them rather than handed to one of them.
//
// PyTorch's max(dim) returns (values, indices) and routes the whole gradient
// to the argmax; its amax splits. Splitting is the choice here because it
// needs no second return value and no per-element scatter primitive, and
// because it is the symmetric answer -- max(x,x) then has d/dx = 1, not 1 for
// the copy that happens to be first. Ties are a measure-zero set, so gradcheck
// on random input never sees the difference (DECISIONS.md B16).
def _extremeBackward($ad, $axis, $isMax) {
    return def($gr) {
        $m = nd_max($ad, $axis, true);
        if (!$isMax) { $m = nd_min($ad, $axis, true); }
        $mask = nd_astype(nd_equal($ad, $m, null), "f64");
        $cnt = nd_sum($mask, $axis, true);
        $g = $gr;
        if ($axis != null) {
            $s = nd_shape($ad);
            $target = [];
            $i = 0;
            while ($i < len($s)) {
                if ($i == $axis) { push($target, 1); }
                else { push($target, $s[$i]); }
                $i = $i + 1;
            }
            $g = nd_reshape($gr, $target);
        }
        return [nd_divide(nd_multiply($mask, $g, null), $cnt, null)];
    };
}
def max($a, $axis, $keepdims) {
    _need();
    $ta = _asT($a);
    return _mk(nd_max($ta.data, $axis, $keepdims), [$ta], "max",
               _extremeBackward($ta.data, $axis, true));
}
def min($a, $axis, $keepdims) {
    _need();
    $ta = _asT($a);
    return _mk(nd_min($ta.data, $axis, $keepdims), [$ta], "min",
               _extremeBackward($ta.data, $axis, false));
}
// Elementwise against another tensor or a scalar: a different op entirely,
// and the one relu is built from.
def _maximumBackward($ad, $bd, $isMax) {
    return def($gr) {
        $take_a = nd_greater_equal($ad, $bd, null);
        if (!$isMax) { $take_a = nd_less_equal($ad, $bd, null); }
        // A tie gives the gradient to a, as PyTorch's maximum does not quite
        // do (it splits) -- but a one-sided rule keeps the two gradients
        // summing to exactly the incoming one, which is what matters here.
        $am = nd_astype($take_a, "f64");
        return [nd_multiply($gr, $am, null),
                nd_multiply($gr, nd_subtract(1.0, $am, null), null)];
    };
}
def maximum($a, $b) {
    _need();
    $ta = _asT($a); $tb = _asT($b);
    return _mk(nd_maximum($ta.data, $tb.data, null), [$ta, $tb], "maximum",
               _maximumBackward($ta.data, $tb.data, true));
}
def minimum($a, $b) {
    _need();
    $ta = _asT($a); $tb = _asT($b);
    return _mk(nd_minimum($ta.data, $tb.data, null), [$ta, $tb], "minimum",
               _maximumBackward($ta.data, $tb.data, false));
}

// ── argmax / argmin ─────────────────────────────────────────────────────────
// Integer outputs, so they are constants with no tape node at all. Returning
// a non-differentiable Tensor rather than a raw array keeps them usable in
// the same expressions as everything else.
def argmax($a, $axis) {
    _need();
    $ta = _asT($a);
    return new Tensor(nd_astype(nd_argmax($ta.data, $axis, null), "f64"), false);
}
def argmin($a, $axis) {
    _need();
    $ta = _asT($a);
    return new Tensor(nd_astype(nd_argmin($ta.data, $axis, null), "f64"), false);
}

// ── permute / squeeze / unsqueeze / movedim ─────────────────────────────────
// All four are pure re-labellings of the same memory, so each backward is the
// inverse relabelling. squeeze and unsqueeze take the lazy route and reshape
// straight back to the recorded input shape, which is correct for every axis
// argument including null.
def _permuteBackward($axes) {
    return def($gr) {
        $inv = [];
        $i = 0;
        while ($i < len($axes)) { push($inv, 0); $i = $i + 1; }
        $i = 0;
        while ($i < len($axes)) { $inv[$axes[$i]] = $i; $i = $i + 1; }
        return [nd_transpose($gr, $inv)];
    };
}
def permute($a, $axes) {
    _need();
    $ta = _asT($a);
    if ($axes == null) { throw "btorch.permute(a, axes): axes is required"; }
    if (len($axes) != len(nd_shape($ta.data))) {
        throw "btorch.permute: axes has " + str(len($axes)) + " entries for a "
            + str(len(nd_shape($ta.data))) + "-dimensional tensor";
    }
    return _mk(nd_transpose($ta.data, $axes), [$ta], "permute", _permuteBackward($axes));
}
def squeeze($a, $axis) {
    _need();
    $ta = _asT($a);
    return _mk(nd_squeeze($ta.data, $axis), [$ta], "squeeze",
               _reshapeBackward(nd_shape($ta.data)));
}
def unsqueeze($a, $axis) {
    _need();
    $ta = _asT($a);
    if ($axis == null) { throw "btorch.unsqueeze(a, axis): axis is required"; }
    return _mk(nd_expand_dims($ta.data, $axis), [$ta], "unsqueeze",
               _reshapeBackward(nd_shape($ta.data)));
}
def movedim($a, $src, $dst) {
    _need();
    $ta = _asT($a);
    $nd = len(nd_shape($ta.data));
    $s = $src; $d = $dst;
    if ($s < 0) { $s = $s + $nd; }
    if ($d < 0) { $d = $d + $nd; }
    $order = [];
    $i = 0;
    while ($i < $nd) { if ($i != $s) { push($order, $i); } $i = $i + 1; }
    $front = [];
    $i = 0;
    while ($i < $d) { push($front, $order[$i]); $i = $i + 1; }
    push($front, $s);
    $i = $d;
    while ($i < len($order)) { push($front, $order[$i]); $i = $i + 1; }
    return permute($ta, $front);
}

// ════════════════════════════════════════════════════════════════════════════
//  Joining and splitting (phase 2, on A6)
//
//  cat and stack have N parents, and the engine already handles any arity: the
//  backward returns one gradient per parent and _backward checks the count.
//  Each gradient is a slice of the incoming one -- which is exactly nd_split,
//  concat's adjoint, so the round trip is exact by construction rather than by
//  arithmetic that could drift.
// ════════════════════════════════════════════════════════════════════════════

// Collect a list of tensors, raising something readable if it is not one.
def _tensors($xs, $who) {
    if (type($xs) != "list") {
        throw "btorch." + $who + ": expected a LIST of tensors (got " + type($xs) + ")";
    }
    if (len($xs) == 0) {
        throw "btorch." + $who + ": the list is empty -- the result's rank would be "
            + "unknowable, so this is an error rather than an empty tensor";
    }
    $out = [];
    $i = 0;
    while ($i < len($xs)) { push($out, _asT($xs[$i])); $i = $i + 1; }
    return $out;
}

// ── cat ─────────────────────────────────────────────────────────────────────
def _catBackward($sizes, $axis) {
    return def($gr) {
        // nd_split hands back VIEWS of $gr. They must be copied: the engine
        // stores a returned gradient directly and may later nd_add into it, and
        // two parents' gradients aliasing one buffer would corrupt both.
        $parts = nd_split($gr, $sizes, $axis);
        $out = [];
        $i = 0;
        while ($i < len($parts)) { push($out, nd_copy($parts[$i], null)); $i = $i + 1; }
        return $out;
    };
}
def cat($tensors, $axis) {
    _need();
    $ts = _tensors($tensors, "cat");
    $ax = $axis;
    if ($ax == null) { $ax = 0; }
    $datas = [];
    $sizes = [];
    $i = 0;
    while ($i < len($ts)) {
        push($datas, $ts[$i].data);
        push($sizes, nd_shape($ts[$i].data)[$ax]);
        $i = $i + 1;
    }
    return _mk(nd_concat($datas, $ax), $ts, "cat", _catBackward($sizes, $ax));
}

// ── stack ───────────────────────────────────────────────────────────────────
// The gradient for input k is slab k of the incoming gradient, with the stacked
// axis removed. nd_take keeps the axis (length 1), so it is squeezed off.
def _stackBackward($n, $axis) {
    return def($gr) {
        $out = [];
        $k = 0;
        while ($k < $n) {
            push($out, nd_squeeze(nd_take($gr, [$k], $axis), $axis));
            $k = $k + 1;
        }
        return $out;
    };
}
def stack($tensors, $axis) {
    _need();
    $ts = _tensors($tensors, "stack");
    $ax = $axis;
    if ($ax == null) { $ax = 0; }
    $datas = [];
    $i = 0;
    while ($i < len($ts)) { push($datas, $ts[$i].data); $i = $i + 1; }
    return _mk(nd_stack($datas, $ax), $ts, "stack", _stackBackward(len($ts), $ax));
}

// ── chunk / split ───────────────────────────────────────────────────────────
// Both return a LIST of tensors, each a `take` of a contiguous index range --
// built on take rather than on nd_split so each piece gets its own tape node
// and the gradients flow back independently, accumulating into one input.
def _rangeIdx($from, $count) {
    $ix = [];
    $i = 0;
    while ($i < $count) { push($ix, $from + $i); $i = $i + 1; }
    return $ix;
}
def chunk($a, $n, $axis) {
    _need();
    $ta = _asT($a);
    $ax = $axis;
    if ($ax == null) { $ax = 0; }
    if ($n == null || $n <= 0) { throw "btorch.chunk(a, n, axis): n must be positive"; }
    $extent = nd_shape($ta.data)[$ax];
    // PyTorch's rule: ceil(extent/n) per chunk, so the LAST one is the short
    // one and there may be FEWER than n chunks -- chunk(5, 2) gives [3, 2],
    // which is two pieces, not three. Computed as an exact integer ceiling
    // rather than with float arithmetic and a correction.
    $per = _floorDiv($extent + $n - 1, $n);
    if ($per < 1) { $per = 1; }
    $out = [];
    $at = 0;
    while ($at < $extent) {
        // NOT named $take: Bantu keeps variables and functions in ONE
        // namespace, so a local $take shadows the take() function for the rest
        // of the scope and the next line fails with "it holds a number, not a
        // function". Same root cause as the _Linear naming convention in nn.b.
        $count = $per;
        if ($at + $count > $extent) { $count = $extent - $at; }
        push($out, take($ta, _rangeIdx($at, $count), $ax));
        $at = $at + $count;
    }
    return $out;
}
def split($a, $sizes, $axis) {
    _need();
    $ta = _asT($a);
    $ax = $axis;
    if ($ax == null) { $ax = 0; }
    if (type($sizes) != "list") {
        throw "btorch.split(a, sizes, axis): sizes must be a list (use chunk for a count)";
    }
    $extent = nd_shape($ta.data)[$ax];
    $total = 0;
    $i = 0;
    while ($i < len($sizes)) { $total = $total + $sizes[$i]; $i = $i + 1; }
    if ($total != $extent) {
        throw "btorch.split: the sizes sum to " + str($total) + " but axis "
            + str($ax) + " has " + str($extent) + " elements";
    }
    $out = [];
    $at = 0;
    $i = 0;
    while ($i < len($sizes)) {
        push($out, take($ta, _rangeIdx($at, $sizes[$i]), $ax));
        $at = $at + $sizes[$i];
        $i = $i + 1;
    }
    return $out;
}

// ── pad ─────────────────────────────────────────────────────────────────────
// The adjoint of a constant pad is a crop: the gradient of a cell that was
// invented carries nowhere, and the gradient of a real cell is itself. So the
// backward slices the interior back out -- and that is also why nd_pad refuses
// a negative pad, since padding and cropping are not one operation.
def _padBackward($shape, $before) {
    return def($gr) {
        $spec = [];
        $i = 0;
        while ($i < len($shape)) {
            push($spec, [$before[$i], $before[$i] + $shape[$i], 1]);
            $i = $i + 1;
        }
        return [nd_copy(nd_slice($gr, $spec), null)];
    };
}
def pad($a, $pads, $value) {
    _need();
    $ta = _asT($a);
    $shape = nd_shape($ta.data);
    $nd = len($shape);
    // Normalise whatever form `pads` came in as, because the backward needs
    // the per-axis `before` amounts and must not re-derive them differently.
    $before = [];
    $i = 0;
    if (type($pads) == "number") {
        while ($i < $nd) { push($before, $pads); $i = $i + 1; }
    } else {
        if (type($pads) != "list") { throw "btorch.pad: pads must be a number or a list"; }
        if (len($pads) == 2 && type($pads[0]) == "number") {
            while ($i < $nd) { push($before, $pads[0]); $i = $i + 1; }
        } else {
            if (len($pads) != $nd) {
                throw "btorch.pad: pads has " + str(len($pads)) + " entries for a "
                    + str($nd) + "-dimensional tensor";
            }
            while ($i < $nd) { push($before, $pads[$i][0]); $i = $i + 1; }
        }
    }
    $v = $value;
    if ($v == null) { $v = 0.0; }
    return _mk(nd_pad($ta.data, $pads, $v), [$ta], "pad",
               _padBackward($shape, $before));
}

// ── repeat_interleave / tile ────────────────────────────────────────────────
// Both are linear, and both adjoints are a SUM over the copies. Written with
// reshape+sum rather than a hand-rolled scatter: the copies are contiguous in
// a predictable place, so a reshape exposes them as an axis and the sum over
// that axis is the adjoint by inspection.
def _repeatBackward($shape, $r, $axis) {
    return def($gr) {
        if ($axis == null) {
            // Flattened: [n*r] -> [n, r] -> sum over r.
            $n = 1;
            $i = 0;
            while ($i < len($shape)) { $n = $n * $shape[$i]; $i = $i + 1; }
            return [nd_reshape(nd_sum(nd_reshape($gr, [$n, $r]), 1, null), $shape)];
        }
        // Along an axis: split that axis into [extent, r] and sum the r.
        $t = [];
        $i = 0;
        while ($i < len($shape)) {
            if ($i == $axis) { push($t, $shape[$i]); push($t, $r); }
            else { push($t, $shape[$i]); }
            $i = $i + 1;
        }
        return [nd_reshape(nd_sum(nd_reshape($gr, $t), $axis + 1, null), $shape)];
    };
}
def repeat_interleave($a, $repeats, $axis) {
    _need();
    $ta = _asT($a);
    if ($repeats == null) { throw "btorch.repeat_interleave(a, repeats, axis): repeats is required"; }
    return _mk(nd_repeat($ta.data, $repeats, $axis), [$ta], "repeat_interleave",
               _repeatBackward(nd_shape($ta.data), $repeats, $axis));
}
// tile repeats the whole block, so copy k of the input sits at stride
// `shape[d]` along each axis. The adjoint sums the blocks, which a reshape to
// [reps[0], shape[0], reps[1], shape[1], ...] exposes -- but only when the
// input is contiguous in C order, which a fresh nd_tile result is.
def _tileBackward($shape, $reps) {
    return def($gr) {
        $g = $gr;
        $nd = len($shape);
        // Sum the leading repetition off one axis at a time, innermost last,
        // so the shape bookkeeping stays local.
        $d = 0;
        while ($d < $nd) {
            $r = $reps[$d];
            if ($r != 1) {
                $cur = nd_shape($g);
                $t = [];
                $i = 0;
                while ($i < len($cur)) {
                    if ($i == $d) { push($t, $r); push($t, $shape[$d]); }
                    else { push($t, $cur[$i]); }
                    $i = $i + 1;
                }
                $g = nd_sum(nd_reshape(nd_ascontiguous($g), $t), $d, null);
            }
            $d = $d + 1;
        }
        return [nd_reshape(nd_ascontiguous($g), $shape)];
    };
}
def tile($a, $reps) {
    _need();
    $ta = _asT($a);
    $shape = nd_shape($ta.data);
    $rs = $reps;
    if (type($rs) == "number") {
        $rs = [];
        $i = 0;
        while ($i < len($shape) - 1) { push($rs, 1); $i = $i + 1; }
        push($rs, $reps);
    }
    if (len($rs) != len($shape)) {
        throw "btorch.tile: reps has " + str(len($rs)) + " entries for a "
            + str(len($shape)) + "-dimensional tensor -- btorch requires one per "
            + "axis (nd_tile allows fewer, but then the backward's shape "
            + "bookkeeping depends on which end they applied to)";
    }
    return _mk(nd_tile($ta.data, $rs), [$ta], "tile", _tileBackward($shape, $rs));
}

// ── gather ──────────────────────────────────────────────────────────────────
// out[i,j,...] = a[i, ..., idx[i,j,...], ...]: per-ELEMENT indexing, where
// `take` is per-slab. PyTorch's gather(dim, index).
//
// Deliberately NOT a new native kernel. The per-element flat index is
//     flat = idx * stride[dim] + sum over the other axes of coord_d * stride_d
// and every term is one broadcast arange, so the whole thing is a handful of
// vector ops over the already-native take/index_add pair. A `scatter_add`
// kernel would have been a third scatter primitive to test and keep correct.
def _gatherFlat($shape, $dim, $idx) {
    // C-order strides of the SOURCE.
    $stride = [];
    $i = 0;
    while ($i < len($shape)) { push($stride, 1); $i = $i + 1; }
    $acc = 1;
    $d = len($shape) - 1;
    while ($d >= 0) { $stride[$d] = $acc; $acc = $acc * $shape[$d]; $d = $d - 1; }

    $oshape = nd_shape($idx);
    $flat = nd_multiply(nd_astype($idx, "f64"), $stride[$dim] * 1.0, null);
    $d = 0;
    while ($d < len($oshape)) {
        if ($d != $dim) {
            // An arange along axis d, shaped [1,...,n,...,1] so it broadcasts.
            $t = [];
            $i = 0;
            while ($i < len($oshape)) {
                if ($i == $d) { push($t, $oshape[$i]); } else { push($t, 1); }
                $i = $i + 1;
            }
            $co = nd_reshape(nd_astype(nd_arange(0.0, $oshape[$d] * 1.0, null), "f64"), $t);
            $flat = nd_add($flat, nd_multiply($co, $stride[$d] * 1.0, null), null);
        }
        $d = $d + 1;
    }
    return nd_astype(nd_reshape(nd_ascontiguous($flat), [nd_size($idx)]), "i64");
}
def _gatherBackward($shape, $flat, $oshape) {
    return def($gr) {
        $n = 1;
        $i = 0;
        while ($i < len($shape)) { $n = $n * $shape[$i]; $i = $i + 1; }
        $g = nd_zeros([$n], "f64");
        nd_index_add($g, $flat, nd_reshape(nd_ascontiguous($gr), [nd_size($gr)]), 0);
        return [nd_reshape($g, $shape)];
    };
}
def gather($a, $dim, $indices) {
    _need();
    $ta = _asT($a);
    $shape = nd_shape($ta.data);
    if ($dim == null) { throw "btorch.gather(a, dim, indices): dim is required"; }
    $idx = _idx($indices);
    $oshape = nd_shape($idx);
    if (len($oshape) != len($shape)) {
        throw "btorch.gather: indices is " + str($oshape) + " but the tensor is "
            + str($shape) + " -- gather needs indices of the SAME RANK (use take "
            + "for a flat list of slab indices)";
    }
    $i = 0;
    while ($i < len($shape)) {
        if ($i != $dim && $oshape[$i] != $shape[$i]) {
            throw "btorch.gather: gathering along dim " + str($dim) + ", so every "
                + "other axis must agree -- the tensor is " + str($shape)
                + " but indices is " + str($oshape) + " (they differ on axis "
                + str($i) + ")";
        }
        $i = $i + 1;
    }
    // Validated here rather than left to nd_take on the flattened array, where
    // an out-of-range index would silently land in a DIFFERENT row instead of
    // raising -- the flattening is what makes every index nominally in range.
    if (nd_size($idx) > 0) {
        $lo = nd_get(nd_reshape(nd_min($idx, null, null), [1]), [0]);
        $hi = nd_get(nd_reshape(nd_max($idx, null, null), [1]), [0]);
        if ($lo < 0 || $hi >= $shape[$dim]) {
            throw "btorch.gather: an index outside [0, " + str($shape[$dim])
                + ") along dim " + str($dim) + " (saw " + str($lo) + " .. " + str($hi)
                + ") -- after flattening, an out-of-range index would read a "
                + "different row rather than fail, so it is refused here";
        }
    }
    $flat = _gatherFlat($shape, $dim, $idx);
    $src = nd_reshape(nd_ascontiguous($ta.data), [nd_size($ta.data)]);
    $out = nd_reshape(nd_take($src, $flat, 0), $oshape);
    return _mk($out, [$ta], "gather", _gatherBackward($shape, $flat, $oshape));
}

// ════════════════════════════════════════════════════════════════════════════
//  Convolution (phase 5, on A9)
//
//  conv2d is NOT a hand-written gradient. im2col is given one op with one
//  backward -- col2im, which is its verified adjoint (tests/numba_manip_test.b
//  asserts <im2col(x),y> == <x,col2im(y)> over eight geometries) -- and the
//  convolution is then a composition of ops that were each already
//  gradchecked: im2col, reshape, transpose, matmul, add, permute.
//
//  That is the whole reason to pay for a native im2col. A convolution written
//  as one matmul over a gathered matrix has no convolution-specific gradient
//  to derive, so there is no convolution-specific gradient to get wrong.
// ════════════════════════════════════════════════════════════════════════════

// A real floor. The obvious spelling -- nd_astype(x, "i64") -- is NOT one:
// setFromDouble uses std::llround, so it ROUNDS, and _floor(2.5) came back 3.
// nd_floor_divide is exact and says what it means.
def _floorDiv($a, $b) {
    return nd_get(nd_floor_divide(nd_full([1], $a * 1.0), $b * 1.0, null), [0]);
}

def _im2colBackward($shape, $g) {
    return def($gr) {
        return [nd_col2im($gr, $shape, $g[0], $g[1], $g[2], $g[3], $g[4], $g[5], $g[6], $g[7])];
    };
}
// im2col($x, [kh, kw, sh, sw, ph, pw, dh, dw]) -- the geometry as one list, so
// the forward and the backward cannot disagree about it.
def im2col($a, $geom) {
    _need();
    $ta = _asT($a);
    if (type($geom) != "list" || len($geom) != 8) {
        throw "btorch.im2col(a, geom): geom must be [kh, kw, sh, sw, ph, pw, dh, dw]";
    }
    $g = $geom;
    $out = nd_im2col($ta.data, $g[0], $g[1], $g[2], $g[3], $g[4], $g[5], $g[6], $g[7]);
    return _mk($out, [$ta], "im2col", _im2colBackward(nd_shape($ta.data), $g));
}

// stride, padding and dilation may each be a NUMBER (both axes) or a
// [height, width] PAIR, as PyTorch's are. The pair form is not a convenience:
// conv1d is implemented as a conv2d of height 1 and so must say "stride 1 down,
// stride s across" -- without it, conv1d could not be written in terms of
// conv2d at all, which is how the first draft failed.
def _pair($v, $dflt, $who) {
    if ($v == null) { return [$dflt, $dflt]; }
    if (type($v) == "list") {
        if (len($v) != 2) {
            throw "btorch." + $who + ": expected a number or a [height, width] pair, got "
                + str(len($v)) + " entries";
        }
        return [$v[0], $v[1]];
    }
    if (type($v) != "number") {
        throw "btorch." + $who + ": expected a number or a [height, width] pair, got "
            + type($v);
    }
    return [$v, $v];
}
// The geometry as the one list every layer below passes around, so the forward
// and the backward cannot disagree about it.
def _geom($kh, $kw, $stride, $padding, $dilation, $sdflt, $who) {
    $sd = $sdflt;
    if ($sd == null) { $sd = 1; }
    $st = _pair($stride, $sd, $who);
    if ($stride == null && $sdflt == 0) { $st = [$kh, $kw]; }   // pooling: stride = kernel
    $pd = _pair($padding, 0, $who);
    $dl = _pair($dilation, 1, $who);
    return [$kh, $kw, $st[0], $st[1], $pd[0], $pd[1], $dl[0], $dl[1]];
}

// conv2d(x, weight, bias, stride, padding, dilation)
//   x      [N, Cin, H, W]
//   weight [Cout, Cin, kh, kw]
//   bias   [Cout] or null
//   -> [N, Cout, OH, OW]
def conv2d($x, $weight, $bias, $stride, $padding, $dilation) {
    _need();
    $tx = _asT($x); $tw = _asT($weight);
    $xs = nd_shape($tx.data);
    $ws = nd_shape($tw.data);
    if (len($xs) != 4) { throw "btorch.conv2d: x must be [N, Cin, H, W], got " + str($xs); }
    if (len($ws) != 4) { throw "btorch.conv2d: weight must be [Cout, Cin, kh, kw], got " + str($ws); }
    if ($ws[1] != $xs[1]) {
        throw "btorch.conv2d: x has " + str($xs[1]) + " channels but weight expects "
            + str($ws[1]);
    }
    $g = _geom($ws[2], $ws[3], $stride, $padding, $dilation, 1, "conv2d");
    $o = nd_to_list(nd_conv_outsize($xs[2], $xs[3], $g[0], $g[1], $g[2], $g[3],
                                    $g[4], $g[5], $g[6], $g[7]));
    $cols = im2col($tx, $g);                                  // [N*OH*OW, Cin*kh*kw]
    $wmat = reshape($tw, [$ws[0], $ws[1] * $g[0] * $g[1]]);   // [Cout, Cin*kh*kw]
    $out = matmul($cols, transpose($wmat));                   // [N*OH*OW, Cout]
    if ($bias != null) { $out = add($out, _asT($bias)); }
    // The im2col row order is (n, oh, ow), so this reshape needs no transpose.
    $out = reshape($out, [$xs[0], $o[0], $o[1], $ws[0]]);
    return permute($out, [0, 3, 1, 2]);
}

// conv1d(x, weight, bias, stride, padding, dilation)
//   x [N, Cin, L], weight [Cout, Cin, k] -> [N, Cout, OL]
// A 2-d convolution with a height of 1. Written as a reshape rather than as a
// second kernel so there is one geometry implementation, not two.
def conv1d($x, $weight, $bias, $stride, $padding, $dilation) {
    _need();
    $tx = _asT($x); $tw = _asT($weight);
    $xs = nd_shape($tx.data); $ws = nd_shape($tw.data);
    if (len($xs) != 3) { throw "btorch.conv1d: x must be [N, Cin, L], got " + str($xs); }
    if (len($ws) != 3) { throw "btorch.conv1d: weight must be [Cout, Cin, k], got " + str($ws); }
    $x4 = reshape($tx, [$xs[0], $xs[1], 1, $xs[2]]);
    $w4 = reshape($tw, [$ws[0], $ws[1], 1, $ws[2]]);
    $st = $stride;   if ($st == null) { $st = 1; }
    $pd = $padding;  if ($pd == null) { $pd = 0; }
    $dl = $dilation; if ($dl == null) { $dl = 1; }
    $out = conv2d($x4, $w4, $bias, [1, $st], [0, $pd], [1, $dl]);
    $os = nd_shape($out.data);
    return reshape($out, [$os[0], $os[1], $os[3]]);
}

// ── pooling ─────────────────────────────────────────────────────────────────
// Each channel pools independently, so the channels are folded into the batch
// ([N,C,H,W] -> [N*C,1,H,W]) and one im2col then produces [N*C*OH*OW, kh*kw]:
// one window per row, every value of that window in it. The pool is then a
// reduction along axis 1, and the gradient is whatever that reduction's
// gradient already is -- no pooling-specific backward at all.
def avg_pool2d($x, $kh, $kw, $sh, $sw, $ph, $pw) {
    _need();
    $tx = _asT($x);
    $xs = nd_shape($tx.data);
    if (len($xs) != 4) { throw "btorch.avg_pool2d: x must be [N, C, H, W], got " + str($xs); }
    // Pooling's stride defaults to the KERNEL, not to 1 -- the sentinel 0
    // asks _geom for that rule.
    $kk = $kw;
    if ($kk == null) { $kk = $kh; }
    $g = _geom($kh, $kk, $sh, $ph, null, 0, "avg_pool2d");
    $o = nd_to_list(nd_conv_outsize($xs[2], $xs[3], $g[0], $g[1], $g[2], $g[3],
                                    $g[4], $g[5], 1, 1));
    $merged = reshape($tx, [$xs[0] * $xs[1], 1, $xs[2], $xs[3]]);
    // Zero padding is exactly right here: PyTorch's count_include_pad defaults
    // to true, so a padded cell contributes a 0 to the mean and is counted.
    $cols = im2col($merged, $g);
    $m = mean($cols, 1, null);
    return reshape($m, [$xs[0], $xs[1], $o[0], $o[1]]);
}
// ── the fused max-pool node ─────────────────────────────────────────────────
// Pooling used to be `max(im2col(x), 1)`, which is correct and was measured at
// 394 ms per forward+backward on one LeNet layer. The forward was 7 ms of that.
//
// The cost was `max`'s GENERIC backward, which cannot know anything about
// pooling: to find out which element of each window won, it recomputes the
// maximum, compares every element against it, counts the ties and divides --
// five passes over a 150,528 x 4 column matrix, three of them broadcasting
// against a stride-0 axis.
//
// The forward already knew the answer and threw it away. So this op keeps it:
// one `nd_argmax` in the forward, and the backward is a single scatter of the
// incoming gradient straight to the winning positions. 394 ms -> 23 ms.
//
// It also makes pooling match PyTorch EXACTLY, which the old version did not:
// argmax picks the first maximum, so a tie inside a window sends the whole
// gradient to one element rather than splitting it (DECISIONS B28). That
// matters in practice -- a window of post-ReLU zeros is a complete tie.
def _maxPoolBackward($shape, $idx, $g) {
    return def($gr) {
        $R = nd_size($idx);
        $K = $g[0] * $g[1];
        $flat = nd_zeros([$R * $K], "f64");
        nd_index_add($flat, $idx, nd_reshape(nd_ascontiguous($gr), [$R]), 0);
        return [nd_col2im(nd_reshape($flat, [$R, $K]), $shape,
                          $g[0], $g[1], $g[2], $g[3], $g[4], $g[5], $g[6], $g[7])];
    };
}
def _maxPoolCols($a, $geom) {
    _need();
    $ta = _asT($a);
    $cols = nd_im2col($ta.data, $geom[0], $geom[1], $geom[2], $geom[3],
                      $geom[4], $geom[5], $geom[6], $geom[7]);
    $cs = nd_shape($cols);
    $R = $cs[0];
    $K = $cs[1];
    $out = nd_max($cols, 1, null);
    // The argmax is only needed to build a backward, so it is not computed
    // when there will not be one. Inference is the common case at deployment
    // and it should not pay for a gradient nobody asked for.
    if (!$_BT.grad || !$ta.requires_grad) { return new Tensor($out, false); }
    // Flat indices into the column matrix: row * K + winner.
    $am = nd_astype(nd_argmax($cols, 1, null), "i64");
    $rowBase = nd_multiply(nd_astype(nd_arange(0.0, $R * 1.0, null), "i64"), $K, null);
    $idx = nd_astype(nd_add($rowBase, $am, null), "i64");
    return _mk($out, [$ta], "max_pool",
               _maxPoolBackward(nd_shape($ta.data), $idx, $geom));
}

def max_pool2d($x, $kh, $kw, $sh, $sw, $ph, $pw) {
    _need();
    $tx = _asT($x);
    $xs = nd_shape($tx.data);
    if (len($xs) != 4) { throw "btorch.max_pool2d: x must be [N, C, H, W], got " + str($xs); }
    $kk = $kw;
    if ($kk == null) { $kk = $kh; }
    $g = _geom($kh, $kk, $sh, $ph, null, 0, "max_pool2d");
    $o = nd_to_list(nd_conv_outsize($xs[2], $xs[3], $g[0], $g[1], $g[2], $g[3],
                                    $g[4], $g[5], 1, 1));
    $merged = reshape($tx, [$xs[0] * $xs[1], 1, $xs[2], $xs[3]]);
    // im2col pads with ZERO, which is wrong for a maximum: a window of all
    // negative values would pool to 0. So the padding is done first, with a
    // sentinel far below any real activation, and im2col is then told not to
    // pad. A finite sentinel rather than -inf, so that nothing downstream can
    // turn (-inf) - (-inf) into NaN.
    $padded = $merged;
    $gg = $g;
    if ($g[4] != 0 || $g[5] != 0) {
        $padded = pad($merged, [[0, 0], [0, 0], [$g[4], $g[4]], [$g[5], $g[5]]], 0.0 - 1.0e300);
        $gg = [$g[0], $g[1], $g[2], $g[3], 0, 0, $g[6], $g[7]];
    }
    $m = _maxPoolCols($padded, $gg);
    return reshape($m, [$xs[0], $xs[1], $o[0], $o[1]]);
}

// ════════════════════════════════════════════════════════════════════════════
//  Creation
// ════════════════════════════════════════════════════════════════════════════
def tensor($data, $requires_grad) {
    _need();
    if (type($data) == "instance") { return new Tensor($data.data, $requires_grad); }
    if (_isNd($data))             { return new Tensor($data, $requires_grad); }
    return new Tensor(nd($data, "f64"), $requires_grad);
}
def zeros($shape, $requires_grad)  { _need(); return new Tensor(nd_zeros($shape, "f64"), $requires_grad); }
def ones($shape, $requires_grad)   { _need(); return new Tensor(nd_ones($shape, "f64"), $requires_grad); }
def full($shape, $v, $requires_grad) { _need(); return new Tensor(nd_full($shape, $v), $requires_grad); }
def eye($n, $requires_grad)        { _need(); return new Tensor(nd_eye($n, $n, 0), $requires_grad); }
def arange($a, $b, $requires_grad) { _need(); return new Tensor(nd_astype(nd_arange($a, $b, null), "f64"), $requires_grad); }
def linspace($a, $b, $n, $requires_grad) { _need(); return new Tensor(nd_linspace($a, $b, $n), $requires_grad); }
def randn($shape, $requires_grad)  { _need(); return new Tensor(nd_random_normal($shape), $requires_grad); }
def rand($shape, $requires_grad)   { _need(); return new Tensor(nd_random_uniform($shape, 0.0, 1.0), $requires_grad); }
def zeros_like($t, $requires_grad) { _need(); return new Tensor(nd_zeros_like($t.data), $requires_grad); }
def ones_like($t, $requires_grad)  { _need(); return new Tensor(nd_ones_like($t.data), $requires_grad); }
def manual_seed($s) { _need(); nd_seed($s); return null; }

// ════════════════════════════════════════════════════════════════════════════
//  gradcheck — central differences against the analytic gradient.
//
//  This is the gate every op in btorch has to pass (DECISIONS B9). A gradient
//  derived by hand and a gradient estimated numerically are wrong in DIFFERENT
//  ways, so agreement between them is real evidence rather than two copies of
//  the same mistake. The independent reference here is calculus itself, which
//  is why this ships as library API and not just as test scaffolding.
//
//      $f = def($ins) { return $ins[0].mul($ins[1]).sum(null, null); };
//      $r = bt.gradcheck($f, [$a, $b], null, null);
//      print($r.ok);        // true / false
//      print($r.worst);     // the largest relative disagreement seen
//
//  $f takes a LIST of tensors and must return a one-element tensor.
//
//  h = 1e-5 in f64 is the usual sweet spot: big enough that (f(x+h)-f(x-h))
//  does not vanish into rounding, small enough that the O(h^2) truncation of
//  the central difference stays below the tolerance. Note this is comfortable
//  only because btorch computes in f64 (B6); at f32 it would be marginal.
// ════════════════════════════════════════════════════════════════════════════
def gradcheck($f, $inputs, $h, $tol) {
    _need();
    if ($h == null)   { $h = 0.00001; }
    if ($tol == null) { $tol = 0.0000001; }

    // ── analytic, from one backward pass ────────────────────────────────
    $i = 0;
    while ($i < len($inputs)) { $inputs[$i].grad = null; $i = $i + 1; }
    $out = $f($inputs);
    if (nd_size($out.data) != 1) {
        throw "btorch.gradcheck: the function must return a one-element tensor, "
            + "got shape " + str(nd_shape($out.data));
    }
    $out.backward(null);

    $analytic = [];
    $i = 0;
    while ($i < len($inputs)) {
        $gi = $inputs[$i].grad;
        if ($gi == null) { $gi = nd_zeros(nd_shape($inputs[$i].data), "f64"); }
        push($analytic, nd_reshape(nd_copy($gi, null), [nd_size($gi)]));
        $i = $i + 1;
    }

    // ── numerical, element by element ───────────────────────────────────
    $worst = 0.0;
    $worstAt = "";
    $i = 0;
    while ($i < len($inputs)) {
        $t = $inputs[$i];
        $n = nd_size($t.data);
        // A reshape is a VIEW, so writing through it perturbs the tensor's own
        // buffer in place -- no copy, and the perturbation is undone exactly.
        $flat = nd_reshape($t.data, [$n]);
        $ana = $analytic[$i];
        $k = 0;
        while ($k < $n) {
            $orig = nd_get($flat, [$k]);

            nd_set($flat, [$k], $orig + $h);
            $plus = no_grad(def() { return $f($inputs).item(); });

            nd_set($flat, [$k], $orig - $h);
            $minus = no_grad(def() { return $f($inputs).item(); });

            nd_set($flat, [$k], $orig);            // restore, exactly

            $num = ($plus - $minus) / (2.0 * $h);
            $got = nd_get($ana, [$k]);

            // Relative error, with an absolute floor so that a gradient of
            // zero does not divide by zero and does not demand impossible
            // precision either.
            $diff = $num - $got;
            if ($diff < 0) { $diff = 0 - $diff; }
            $scale = 1.0;
            $an = $num; if ($an < 0) { $an = 0 - $an; }
            $ag = $got; if ($ag < 0) { $ag = 0 - $ag; }
            if ($an > $scale) { $scale = $an; }
            if ($ag > $scale) { $scale = $ag; }
            $rel = $diff / $scale;
            if ($rel > $worst) {
                $worst = $rel;
                $worstAt = "input " + str($i) + " element " + str($k)
                         + " (analytic " + str($got) + " vs numerical " + str($num) + ")";
            }
            $k = $k + 1;
        }
        $i = $i + 1;
    }
    return {"ok": $worst <= $tol, "worst": $worst, "where": $worstAt, "tol": $tol};
}

// ── misc ────────────────────────────────────────────────────────────────────
def allclose($a, $b, $tol) {
    _need();
    if ($tol == null) { $tol = 0.00000001; }
    return nd_allclose(_arr($a), _arr($b), $tol, $tol);
}
def version() { return "btorch 0.2.0 (phases 1-4, indexing)"; }
