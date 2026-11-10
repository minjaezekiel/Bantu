// ════════════════════════════════════════════════════════════════════════════
//  nn.b — the module system, layers and losses.
//
//      include "btorch" as bt;
//      include "btorch/nn.b" as nn;
//
//      class MLP extends Module {
//          def init() {
//              super();
//              this.add("fc1", nn.Linear(784, 128, true));
//              this.add("fc2", nn.Linear(128, 10, true));
//          }
//          def forward($x) {
//              return this.get("fc2").forward(bt.relu(this.get("fc1").forward($x)));
//          }
//      }
//
//  `extends Module` works because Bantu's class registry is flat: a class
//  declared in an included module is visible by its bare name. Registration of
//  children is EXPLICIT -- this.add("fc1", ...) -- because Bantu has no
//  __setattr__ hook to intercept assignment the way PyTorch does. Three
//  characters more typing, and the parameter tree is inspectable (DECISIONS B5).
//
//  NAMING. The concrete layers are declared as `_Linear`, `_Sequential` and so
//  on, with a same-named factory function -- `nn.Linear(4, 3, true)` -- as the
//  public API. The underscore is not decoration: a `def Linear()` and a
//  `class Linear` collide, because the function shadows the class name, so
//  `new Linear(...)` inside the factory re-enters the FACTORY and recurses
//  until the interpreter's 1000-call depth guard fires. `Module` keeps its bare
//  name, since users extend it and never call it.
//
//  Design: docs/btorch-architecture.md section 6.
// ════════════════════════════════════════════════════════════════════════════

include "./btorch.b" as bt;

// A module's top-level defs are not shared across includes by bare name, so
// this is nn.b's own copy rather than a reference to btorch.b's. type() of a
// native array returns the handle's TAG, "ndarray", not the word "handle".
def _isNd($x) { return type($x) == "ndarray"; }

// ── Module ──────────────────────────────────────────────────────────────────
class Module {
    def init() {
        this.__bt_module = true;
        this._params = {};      // name -> Tensor
        this._buffers = {};     // name -> Tensor (no grad: running stats, masks)
        this._modules = {};     // name -> Module
        this.training = true;
    }

    // ── registration ────────────────────────────────────────────────────
    def add($name, $child) {
        this._modules[$name] = $child;
        return $child;
    }
    def get($name) {
        $m = this._modules[$name];
        if ($m == null) {
            throw "btorch.nn: this module has no child called '" + str($name) + "'";
        }
        return $m;
    }
    def register($name, $tensor) {
        this._params[$name] = $tensor;
        return $tensor;
    }
    def param($name) {
        $p = this._params[$name];
        if ($p == null) {
            throw "btorch.nn: this module has no parameter called '" + str($name) + "'";
        }
        return $p;
    }
    def register_buffer($name, $tensor) { this._buffers[$name] = $tensor; return $tensor; }
    def buffer($name) { return this._buffers[$name]; }

    // ── the parameter tree ──────────────────────────────────────────────
    // DEDUPLICATED BY IDENTITY. A weight shared between two layers -- tied
    // embeddings are a real pattern -- would otherwise be returned twice and
    // stepped twice by the optimizer, which is a silent 2x learning rate on
    // that tensor alone. Collected iteratively, so a deep model cannot hit the
    // interpreter's 1000-call recursion cap.
    def named_parameters() {
        $out = [];
        $seen = {};
        $stack = [["", this]];
        while (len($stack) > 0) {
            $top = $stack[len($stack) - 1];
            pop($stack);
            $prefix = $top[0];
            $mod = $top[1];
            for $pname, $pval in $mod._params.items() {
                if ($pval != null) {
                    $key = str($pval.id);
                    if ($seen[$key] == null) {
                        $seen[$key] = true;
                        push($out, [$prefix + $pname, $pval]);
                    }
                }
            }
            for $cname, $cmod in $mod._modules.items() {
                if ($cmod != null) { push($stack, [$prefix + $cname + ".", $cmod]); }
            }
        }
        return $out;
    }

    def parameters() {
        $named = this.named_parameters();
        $out = [];
        $i = 0;
        while ($i < len($named)) { push($out, $named[$i][1]); $i = $i + 1; }
        return $out;
    }

    def num_parameters() {
        $ps = this.parameters();
        $n = 0;
        $i = 0;
        while ($i < len($ps)) { $n = $n + $ps[$i].size(); $i = $i + 1; }
        return $n;
    }

    // ── dtype casting (D2c) ─────────────────────────────────────────────
    // Casts every parameter and buffer IN PLACE, mutating each Tensor's .data
    // rather than replacing the Tensor. That is deliberate and load-bearing:
    // named_parameters() deduplicates by tensor identity, so a weight shared
    // between two layers (tied embeddings) must stay the SAME object or it
    // would be cast twice and, worse, stepped twice by the optimizer.
    //
    // Gradients are dropped: a stale f64 gradient against an f32 parameter is
    // a dtype mismatch waiting to happen, and after a cast there is no
    // meaningful gradient to keep.
    //
    // Call this BEFORE constructing the optimizer. An optimizer built first
    // holds the same Tensor objects, so the parameters themselves follow -- but
    // any moment buffers it has already allocated stay at the old dtype.
    def to($dtype) {
        $ps = this.parameters();
        $i = 0;
        while ($i < len($ps)) {
            $ps[$i].data = nd_astype($ps[$i].data, $dtype);
            $ps[$i].grad = null;
            $i = $i + 1;
        }
        $bs = this.buffers();
        $j = 0;
        while ($j < len($bs)) {
            // A buffer may be an integer count (BatchNorm's num_batches_tracked),
            // and casting that to a float would be wrong. Only floats convert.
            $d = nd_dtype($bs[$j].data);
            if ($d == "f32" || $d == "f64") {
                $bs[$j].data = nd_astype($bs[$j].data, $dtype);
            }
            $j = $j + 1;
        }
        return this;
    }
    def float()  { return this.to("f32"); }
    def double() { return this.to("f64"); }

    // The dtype this module's parameters are in, or null if it has none.
    def dtype() {
        $ps = this.parameters();
        if (len($ps) == 0) { return null; }
        return nd_dtype($ps[0].data);
    }

    def zero_grad() {
        $ps = this.parameters();
        $i = 0;
        while ($i < len($ps)) { $ps[$i].grad = null; $i = $i + 1; }
        return this;
    }

    // ── train / eval, recursively ───────────────────────────────────────
    def _setMode($flag) {
        $stack = [this];
        while (len($stack) > 0) {
            $m = $stack[len($stack) - 1];
            pop($stack);
            $m.training = $flag;
            for $cname, $cmod in $m._modules.items() {
                if ($cmod != null) { push($stack, $cmod); }
            }
        }
        return this;
    }
    def train() { return this._setMode(true); }
    def eval()  { return this._setMode(false); }

    // Buffers, by the same iterative walk and the same identity dedup.
    //
    // This existed as `register_buffer` with no way to ENUMERATE what had been
    // registered, which meant state_dict silently omitted every buffer. A
    // BatchNorm's running mean and variance are buffers precisely so no
    // optimizer steps them; a state_dict without them reloads a model whose
    // eval() is wrong, and that model trains and scores fine right up until it
    // does not.
    def named_buffers() {
        $out = [];
        $seen = {};
        $stack = [["", this]];
        while (len($stack) > 0) {
            $top = $stack[len($stack) - 1];
            pop($stack);
            $prefix = $top[0];
            $mod = $top[1];
            for $bname, $bval in $mod._buffers.items() {
                if ($bval != null) {
                    $key = str($bval.id);
                    if ($seen[$key] == null) {
                        $seen[$key] = true;
                        push($out, [$prefix + $bname, $bval]);
                    }
                }
            }
            for $cname, $cmod in $mod._modules.items() {
                if ($cmod != null) { push($stack, [$prefix + $cname + ".", $cmod]); }
            }
        }
        return $out;
    }
    def buffers() {
        $named = this.named_buffers();
        $out = [];
        $i = 0;
        while ($i < len($named)) { push($out, $named[$i][1]); $i = $i + 1; }
        return $out;
    }

    // Parameters and buffers together, as name -> ndarray. This is what the
    // binary saver walks; state_dict() below is the nested-list form kept for
    // callers that want something printable.
    def state_arrays() {
        $out = {};
        $named = this.named_parameters();
        $i = 0;
        while ($i < len($named)) { $out[$named[$i][0]] = $named[$i][1].data; $i = $i + 1; }
        $nb = this.named_buffers();
        $i = 0;
        while ($i < len($nb)) { $out[$nb[$i][0]] = $nb[$i][1].data; $i = $i + 1; }
        return $out;
    }

    // Copy arrays INTO the existing tensors, by name. In place, so an
    // optimizer's state -- which keys on tensor id -- keeps pointing at the
    // tensors the model actually uses.
    def load_state_arrays($map) {
        if (type($map) != "dict") {
            throw "btorch.nn.load_state_arrays: expected a dict of name -> array";
        }
        $np = 0;
        $nbuf = 0;
        $all = [[this.named_parameters(), "parameter"], [this.named_buffers(), "buffer"]];
        $g = 0;
        while ($g < 2) {
            $named = $all[$g][0];
            $what = $all[$g][1];
            $i = 0;
            while ($i < len($named)) {
                $k = $named[$i][0];
                $v = $map[$k];
                if ($v == null) {
                    throw "btorch.nn.load_state_arrays: the file has no " + $what
                        + " called '" + $k + "'";
                }
                $target = $named[$i][1];
                if (nd_shape($v) != nd_shape($target.data)) {
                    throw "btorch.nn.load_state_arrays: '" + $k + "' is "
                        + str(nd_shape($v)) + " in the file but the " + $what
                        + " is " + str(nd_shape($target.data));
                }
                nd_add(nd_astype($v, nd_dtype($target.data)), 0.0, $target.data);
                if ($g == 0) { $np = $np + 1; } else { $nbuf = $nbuf + 1; }
                $i = $i + 1;
            }
            $g = $g + 1;
        }
        return {"params": $np, "buffers": $nbuf};
    }

    // ── state_dict, with flat "fc1.weight" keys ─────────────────────────
    def state_dict() {
        $out = {};
        $named = this.named_parameters();
        $i = 0;
        while ($i < len($named)) {
            $out[$named[$i][0]] = nd_to_list($named[$i][1].data);
            $i = $i + 1;
        }
        // Buffers belong in a state_dict, as they do in PyTorch's.
        $nb = this.named_buffers();
        $i = 0;
        while ($i < len($nb)) {
            $out[$nb[$i][0]] = nd_to_list($nb[$i][1].data);
            $i = $i + 1;
        }
        return $out;
    }
    def load_state_dict($sd) {
        $named = this.named_parameters();
        $nb = this.named_buffers();
        $i = 0;
        while ($i < len($nb)) { push($named, $nb[$i]); $i = $i + 1; }
        $i = 0;
        $loaded = 0;
        while ($i < len($named)) {
            $k = $named[$i][0];
            $v = $sd[$k];
            if ($v == null) {
                throw "btorch.nn: load_state_dict is missing a value for '" + $k + "'";
            }
            $incoming = nd($v, "f64");
            $target = $named[$i][1];
            if (nd_shape($incoming) != nd_shape($target.data)) {
                throw "btorch.nn: load_state_dict: '" + $k + "' has shape "
                    + str(nd_shape($incoming)) + " but the parameter is "
                    + str(nd_shape($target.data));
            }
            // Copy INTO the existing buffer, so anything already holding this
            // parameter -- an optimizer's state, most importantly -- keeps
            // pointing at the tensor the model actually uses.
            nd_add($incoming, 0.0, $target.data);
            $loaded = $loaded + 1;
            $i = $i + 1;
        }
        return $loaded;
    }

    // Subclasses override this.
    def forward($x) {
        throw "btorch.nn: this module does not implement forward()";
    }
    // So a Module can be used anywhere a callable layer is expected.
    def call($x) { return this.forward($x); }
}

// ════════════════════════════════════════════════════════════════════════════
//  Initialization
//
//  This is not an afterthought. A wrong initializer does not fail -- it trains
//  slowly, or not at all, and looks like a bad learning rate. So the default
//  matches PyTorch's (Kaiming uniform on fan_in) and the test asserts the
//  VARIANCE of a large initialized weight against the analytic value, which is
//  the only way a unit test can catch this.
// ════════════════════════════════════════════════════════════════════════════

// Kaiming/He uniform: U(-bound, bound) with bound = sqrt(6/((1+a^2)*fan_in)).
// For a = sqrt(5), which is PyTorch's nn.Linear default, this reduces to
// sqrt(1/fan_in) -- the same bound PyTorch uses for both weight and bias.
def kaiming_uniform($shape, $fan_in) {
    $bound = nd_get(nd_sqrt(nd_full([1], 1.0 / $fan_in), null), [0]);
    return nd_random_uniform($shape, 0.0 - $bound, $bound);
}
// Xavier/Glorot uniform: bound = sqrt(6/(fan_in+fan_out)).
def xavier_uniform($shape, $fan_in, $fan_out) {
    $bound = nd_get(nd_sqrt(nd_full([1], 6.0 / ($fan_in + $fan_out)), null), [0]);
    return nd_random_uniform($shape, 0.0 - $bound, $bound);
}

// ── Linear ──────────────────────────────────────────────────────────────────
class _Linear extends Module {
    def init($in_features, $out_features, $bias) {
        super();
        this.in_features = $in_features;
        this.out_features = $out_features;
        this.use_bias = $bias != false;
        this.register("weight", bt.tensor(kaiming_uniform([$in_features, $out_features], $in_features), true));
        if (this.use_bias) {
            this.register("bias", bt.tensor(kaiming_uniform([$out_features], $in_features), true));
        }
    }
    def forward($x) {
        $out = bt.matmul($x, this.param("weight"));
        if (this.use_bias) { $out = bt.add($out, this.param("bias")); }
        return $out;
    }
}
def Linear($in_f, $out_f, $bias) { return new _Linear($in_f, $out_f, $bias); }

// ── activations as layers ───────────────────────────────────────────────────
class _ReLU extends Module {
    def init() { super(); }
    def forward($x) { return bt.relu($x); }
}
class _Tanh extends Module {
    def init() { super(); }
    def forward($x) { return bt.tanh($x); }
}
class _Sigmoid extends Module {
    def init() { super(); }
    def forward($x) { return bt.sigmoid($x); }
}
class _LeakyReLU extends Module {
    def init($slope) { super(); this.slope = $slope; if ($slope == null) { this.slope = 0.01; } }
    def forward($x) { return bt.leaky_relu($x, this.slope); }
}
class _Flatten extends Module {
    def init() { super(); }
    def forward($x) { return bt.flatten($x); }
}
def ReLU()         { return new _ReLU(); }
def Tanh()         { return new _Tanh(); }
def Sigmoid()      { return new _Sigmoid(); }
def LeakyReLU($s)  { return new _LeakyReLU($s); }
def Flatten()      { return new _Flatten(); }

// ── Sequential ──────────────────────────────────────────────────────────────
// Children are registered under "0", "1", ... so state_dict keys read like
// PyTorch's ("0.weight"), and _order keeps the run order explicit rather than
// relying on dict iteration.
class _Sequential extends Module {
    def init($layers) {
        super();
        this._order = [];
        $i = 0;
        while ($i < len($layers)) {
            this.add(str($i), $layers[$i]);
            push(this._order, str($i));
            $i = $i + 1;
        }
    }
    def forward($x) {
        $out = $x;
        $i = 0;
        while ($i < len(this._order)) {
            $layer = this.get(this._order[$i]);
            $out = $layer.forward($out);
            $i = $i + 1;
        }
        return $out;
    }
}
def Sequential($layers) { return new _Sequential($layers); }

// ── Dropout ─────────────────────────────────────────────────────────────────
// Inverted dropout: scale by 1/(1-p) at TRAIN time so that eval is the exact
// identity and needs no rescaling. A layer that forgets this is a model whose
// accuracy silently drops when it is put in eval mode.
class _Dropout extends Module {
    def init($p) {
        super();
        this.p = $p;
        if ($p == null) { this.p = 0.5; }
        if (this.p < 0 || this.p >= 1) {
            throw "btorch.nn.Dropout: p must be in [0, 1), got " + str(this.p);
        }
    }
    def forward($x) {
        if (!this.training || this.p == 0) { return $x; }
        $keep = nd_astype(nd_greater_equal(nd_random_uniform($x.shape(), 0.0, 1.0), this.p, null),
                          nd_dtype($x.data));
        $scaled = nd_divide($keep, 1.0 - this.p, null);
        return bt.mul($x, bt.tensor($scaled, false));
    }
}
def Dropout($p) { return new _Dropout($p); }

// ── LayerNorm ───────────────────────────────────────────────────────────────
// Normalizes over the LAST axis, like PyTorch's LayerNorm(normalized_shape)
// for a 1-D shape. Written out of btorch ops so it differentiates for free.
class _LayerNorm extends Module {
    def init($n, $eps) {
        super();
        this.n = $n;
        this.eps = $eps;
        if ($eps == null) { this.eps = 0.00001; }
        this.register("gamma", bt.ones([$n], true));
        this.register("beta", bt.zeros([$n], true));
    }
    def forward($x) {
        $mu = bt.mean($x, 1, true);
        $centered = bt.sub($x, $mu);
        $var = bt.mean(bt.mul($centered, $centered), 1, true);
        $norm = bt.div($centered, bt.sqrt(bt.add($var, this.eps)));
        return bt.add(bt.mul($norm, this.param("gamma")), this.param("beta"));
    }
}
def LayerNorm($n, $eps) { return new _LayerNorm($n, $eps); }

// ── Embedding ───────────────────────────────────────────────────────────────
// A [num_embeddings, dim] lookup table. The forward is a row gather and the
// backward is a scatter that ACCUMULATES, which is the entire reason phase A5
// added nd_index_add: a batch that mentions the same token twice must add both
// of its gradients to that row. Scattered with nd_put instead, the first
// occurrence would be overwritten and silently lost.
//
// Any index shape is accepted. [B, L] ids give [B, L, dim] out, as PyTorch
// does, by gathering flat and reshaping -- and bt.reshape is differentiable,
// so the gradient finds its way back without a second code path.
//
// padding_idx is handled by masking the OUTPUT rather than by special-casing
// the gradient: multiplying the gathered rows by 0 where the id is the pad id
// makes the forward zero there and makes that row's gradient exactly zero,
// which is the whole observable content of PyTorch's rule.
class _Embedding extends Module {
    def init($num_embeddings, $dim, $padding_idx) {
        super();
        this.num_embeddings = $num_embeddings;
        this.dim = $dim;
        this.padding_idx = $padding_idx;
        $w = nd_random_normal([$num_embeddings, $dim], 0.0, 1.0);
        if ($padding_idx != null) {
            if ($padding_idx < 0 || $padding_idx >= $num_embeddings) {
                throw "btorch.nn.Embedding: padding_idx " + str($padding_idx)
                    + " is out of range for " + str($num_embeddings) + " embeddings";
            }
            // Zero the pad row in place, through a view of it.
            $row = nd_slice($w, [[$padding_idx, $padding_idx + 1, 1]]);
            nd_multiply($row, 0.0, $row);
        }
        this.register("weight", bt.tensor($w, true));
    }
    def forward($ids) {
        $ia = nd_astype(_asIntArray($ids), "i64");
        $sh = nd_shape($ia);
        if (len($sh) == 0) {
            throw "btorch.nn.Embedding: expected at least one index, got a scalar";
        }
        $n = nd_size($ia);
        $flat = nd_reshape(nd_ascontiguous($ia), [$n]);
        $out = bt.embedding(this.param("weight"), $flat);
        if (this.padding_idx != null) {
            $keep = nd_astype(nd_not_equal($flat, this.padding_idx, null),
                              nd_dtype($out.data));
            $out = bt.mul($out, bt.tensor(nd_reshape($keep, [$n, 1]), false));
        }
        if (len($sh) != 1) {
            $target = [];
            $i = 0;
            while ($i < len($sh)) { push($target, $sh[$i]); $i = $i + 1; }
            push($target, this.dim);
            $out = bt.reshape($out, $target);
        }
        return $out;
    }
}
def Embedding($num_embeddings, $dim, $padding_idx) {
    return new _Embedding($num_embeddings, $dim, $padding_idx);
}

// ── Conv2d / Conv1d ─────────────────────────────────────────────────────────
// weight [out_channels, in_channels, kh, kw]; fan_in is in_channels*kh*kw,
// which is what PyTorch's default init uses and what makes a deep stack keep
// its activation scale.
//
// There is no convolution-specific gradient here or in btorch.b: conv2d is a
// composition of im2col (whose backward is its verified adjoint col2im) with
// reshape, matmul, add and permute, all of which were gradchecked in phase 2.
class _Conv2d extends Module {
    def init($in_channels, $out_channels, $kernel, $stride, $padding, $bias) {
        super();
        this.in_channels = $in_channels;
        this.out_channels = $out_channels;
        $k = $kernel;
        if (type($k) == "number") { $k = [$k, $k]; }
        if (type($k) != "list" || len($k) != 2) {
            throw "btorch.nn.Conv2d: kernel must be a number or a [kh, kw] pair";
        }
        this.kh = $k[0];
        this.kw = $k[1];
        this.stride = $stride;
        if (this.stride == null) { this.stride = 1; }
        this.padding = $padding;
        if (this.padding == null) { this.padding = 0; }
        this.use_bias = $bias != false;
        $fan_in = $in_channels * this.kh * this.kw;
        this.register("weight",
            bt.tensor(kaiming_uniform([$out_channels, $in_channels, this.kh, this.kw], $fan_in), true));
        if (this.use_bias) {
            this.register("bias", bt.tensor(kaiming_uniform([$out_channels], $fan_in), true));
        }
    }
    def forward($x) {
        $b = null;
        if (this.use_bias) { $b = this.param("bias"); }
        return bt.conv2d($x, this.param("weight"), $b, this.stride, this.padding, null);
    }
}
def Conv2d($in_c, $out_c, $kernel, $stride, $padding, $bias) {
    return new _Conv2d($in_c, $out_c, $kernel, $stride, $padding, $bias);
}

class _Conv1d extends Module {
    def init($in_channels, $out_channels, $kernel, $stride, $padding, $bias) {
        super();
        this.in_channels = $in_channels;
        this.out_channels = $out_channels;
        this.k = $kernel;
        this.stride = $stride;
        if (this.stride == null) { this.stride = 1; }
        this.padding = $padding;
        if (this.padding == null) { this.padding = 0; }
        this.use_bias = $bias != false;
        $fan_in = $in_channels * $kernel;
        this.register("weight",
            bt.tensor(kaiming_uniform([$out_channels, $in_channels, $kernel], $fan_in), true));
        if (this.use_bias) {
            this.register("bias", bt.tensor(kaiming_uniform([$out_channels], $fan_in), true));
        }
    }
    def forward($x) {
        $b = null;
        if (this.use_bias) { $b = this.param("bias"); }
        return bt.conv1d($x, this.param("weight"), $b, this.stride, this.padding, null);
    }
}
def Conv1d($in_c, $out_c, $kernel, $stride, $padding, $bias) {
    return new _Conv1d($in_c, $out_c, $kernel, $stride, $padding, $bias);
}

// ── pooling as layers ───────────────────────────────────────────────────────
// Stride defaults to the KERNEL, as PyTorch's does -- not to 1, which is the
// convolution default and the easy mistake.
class _MaxPool2d extends Module {
    def init($kernel, $stride, $padding) {
        super();
        $k = $kernel;
        if (type($k) == "number") { $k = [$k, $k]; }
        this.kh = $k[0]; this.kw = $k[1];
        this.stride = $stride;
        this.padding = $padding;
        if (this.padding == null) { this.padding = 0; }
    }
    def forward($x) {
        return bt.max_pool2d($x, this.kh, this.kw, this.stride, null,
                             this.padding, null);
    }
}
class _AvgPool2d extends Module {
    def init($kernel, $stride, $padding) {
        super();
        $k = $kernel;
        if (type($k) == "number") { $k = [$k, $k]; }
        this.kh = $k[0]; this.kw = $k[1];
        this.stride = $stride;
        this.padding = $padding;
        if (this.padding == null) { this.padding = 0; }
    }
    def forward($x) {
        return bt.avg_pool2d($x, this.kh, this.kw, this.stride, null,
                             this.padding, null);
    }
}
def MaxPool2d($k, $stride, $padding) { return new _MaxPool2d($k, $stride, $padding); }
def AvgPool2d($k, $stride, $padding) { return new _AvgPool2d($k, $stride, $padding); }

// ── BatchNorm ───────────────────────────────────────────────────────────────
// The one layer whose behaviour DIFFERS between train and eval, which is why
// `training` exists on Module at all:
//
//   train: normalize by the batch's own mean/var, and update the running stats
//   eval:  normalize by the running stats, and update nothing
//
// Getting that backwards is the classic silent accuracy loss -- the model
// trains fine and then scores badly, because at eval time it is still using
// one batch's statistics. So the test asserts both modes separately and
// asserts that eval() changes the answer.
//
// The running stats are BUFFERS, not parameters: they must be saved and loaded
// with the model but must never be stepped by an optimizer.
class _BatchNorm1d extends Module {
    def init($n, $eps, $momentum) {
        super();
        this.n = $n;
        this.eps = $eps;
        if (this.eps == null) { this.eps = 0.00001; }
        this.momentum = $momentum;
        if (this.momentum == null) { this.momentum = 0.1; }
        this.register("gamma", bt.ones([$n], true));
        this.register("beta", bt.zeros([$n], true));
        this.register_buffer("running_mean", bt.zeros([$n], false));
        this.register_buffer("running_var", bt.ones([$n], false));
    }
    def forward($x) {
        if (this.training) {
            // Biased variance for the normalization (PyTorch uses the biased
            // estimate here and the UNBIASED one for the running stat).
            $mu = bt.mean($x, 0, true);
            $cen = bt.sub($x, $mu);
            $var = bt.mean(bt.mul($cen, $cen), 0, true);
            $norm = bt.div($cen, bt.sqrt(bt.add($var, this.eps)));
            // The running update is bookkeeping, not part of the graph.
            $bn = this;
            bt.no_grad(def() {
                $m = nd_reshape($mu.data, [$bn.n]);
                $v = nd_reshape($var.data, [$bn.n]);
                $cnt = nd_shape($x.data)[0];
                if ($cnt > 1) {
                    // Bessel's correction, as PyTorch applies it to the stat
                    // it stores but not to the one it normalizes with.
                    $v = nd_multiply($v, ($cnt * 1.0) / ($cnt - 1.0), null);
                }
                $rm = $bn.buffer("running_mean");
                $rv = $bn.buffer("running_var");
                $rm.data = nd_add(nd_multiply($rm.data, 1.0 - $bn.momentum, null),
                                  nd_multiply($m, $bn.momentum, null), null);
                $rv.data = nd_add(nd_multiply($rv.data, 1.0 - $bn.momentum, null),
                                  nd_multiply($v, $bn.momentum, null), null);
                return null;
            });
            return bt.add(bt.mul($norm, this.param("gamma")), this.param("beta"));
        }
        $rm = this.buffer("running_mean");
        $rv = this.buffer("running_var");
        $norm = bt.div(bt.sub($x, bt.tensor($rm.data, false)),
                       bt.sqrt(bt.add(bt.tensor($rv.data, false), this.eps)));
        return bt.add(bt.mul($norm, this.param("gamma")), this.param("beta"));
    }
}
def BatchNorm1d($n, $eps, $momentum) { return new _BatchNorm1d($n, $eps, $momentum); }

// BatchNorm2d normalizes per CHANNEL over (N, H, W). Written by folding the
// spatial axes into the batch and reusing BatchNorm1d's arithmetic, so there
// is one implementation of the train/eval split rather than two.
class _BatchNorm2d extends Module {
    def init($channels, $eps, $momentum) {
        super();
        this.channels = $channels;
        this.add("bn", BatchNorm1d($channels, $eps, $momentum));
    }
    def forward($x) {
        $s = $x.shape();
        if (len($s) != 4) {
            throw "btorch.nn.BatchNorm2d: expected [N, C, H, W], got " + str($s);
        }
        // [N,C,H,W] -> [N*H*W, C], normalize, and back.
        $nhwc = bt.reshape(bt.permute($x, [0, 2, 3, 1]), [$s[0] * $s[2] * $s[3], $s[1]]);
        $out = this.get("bn").forward($nhwc);
        return bt.permute(bt.reshape($out, [$s[0], $s[2], $s[3], $s[1]]), [0, 3, 1, 2]);
    }
}
def BatchNorm2d($c, $eps, $momentum) { return new _BatchNorm2d($c, $eps, $momentum); }

// ── GELU ────────────────────────────────────────────────────────────────────
// The tanh approximation, which is what every transformer implementation
// actually ships:  0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))
// Built from btorch ops, so it differentiates for free and gradchecks.
def gelu($x) {
    $t = bt.tensor(_asIntArray($x), false);
    if (bt.is_tensor($x)) { $t = $x; }
    $x3 = bt.mul(bt.mul($t, $t), $t);
    $inner = bt.mul(bt.add($t, bt.mul($x3, 0.044715)), 0.7978845608028654);
    return bt.mul(bt.mul($t, 0.5), bt.add(bt.tanh($inner), 1.0));
}
class _GELU extends Module {
    def init() { super(); }
    def forward($x) { return gelu($x); }
}
def GELU() { return new _GELU(); }

class _Softmax extends Module {
    def init() { super(); }
    def forward($x) { return softmax($x); }
}
def Softmax() { return new _Softmax(); }

// ════════════════════════════════════════════════════════════════════════════
//  Losses
//
//  Each takes `reduction` in {"mean", "sum", "none"}; null means "mean".
// ════════════════════════════════════════════════════════════════════════════
def _reduce($t, $reduction) {
    if ($reduction == null || $reduction == "mean") { return bt.mean($t, null, null); }
    if ($reduction == "sum")  { return bt.sum($t, null, null); }
    if ($reduction == "none") { return $t; }
    throw "btorch.nn: reduction must be 'mean', 'sum' or 'none', got " + str($reduction);
}

def mse_loss($pred, $target, $reduction) {
    $d = bt.sub($pred, $target);
    return _reduce(bt.mul($d, $d), $reduction);
}
def l1_loss($pred, $target, $reduction) {
    return _reduce(bt.abs_(bt.sub($pred, $target)), $reduction);
}
// Huber / smooth L1: quadratic within delta, linear outside. Built from
// btorch ops so it differentiates without a hand-written backward.
def huber_loss($pred, $target, $delta, $reduction) {
    if ($delta == null) { $delta = 1.0; }
    $d = bt.sub($pred, $target);
    $a = bt.abs_($d);
    // where(|d| <= delta, 0.5 d^2, delta*(|d| - 0.5 delta))  -- as a blend, so
    // no non-differentiable select is needed
    $small = nd_astype(nd_less_equal($a.data, $delta, null), nd_dtype($a.data));
    $big = nd_subtract(1.0, $small, null);
    $quad = bt.mul(bt.mul($d, $d), 0.5);
    $lin = bt.mul(bt.sub($a, 0.5 * $delta), $delta);
    $blend = bt.add(bt.mul($quad, bt.tensor($small, false)),
                    bt.mul($lin, bt.tensor($big, false)));
    return _reduce($blend, $reduction);
}

// ── log_softmax, numerically stable ─────────────────────────────────────────
// log_softmax(x) = x - max - log(sum(exp(x - max)))
//
// The max shift is not a refinement. Without it, exp() of a logit around 1000
// overflows to +inf and the loss is NaN for a confident prediction -- a
// library that cannot train. DECISIONS B8.
def log_softmax($x) {
    $mx = bt.tensor(nd_max($x.data, 1, true), false);
    $shifted = bt.sub($x, $mx);
    $lse = bt.log(bt.sum(bt.exp($shifted), 1, true));
    return bt.sub($shifted, $lse);
}
def softmax($x) { return bt.exp(log_softmax($x)); }

// ── cross entropy ───────────────────────────────────────────────────────────
// Takes raw LOGITS and integer class indices, and fuses log_softmax with the
// negative log-likelihood so no probability is ever materialized that could
// underflow to zero and give -inf.
def nll_loss($logp, $targets, $reduction) {
    $n = $logp.shape()[0];
    $k = $logp.shape()[1];
    // one-hot the targets, then select by multiplying: the selection is a
    // constant, so it needs no gradient of its own.
    //
    // This used to run one interpreter iteration per row, on the hot path of
    // every training step -- a 1024-row batch meant 1024 trips through the
    // evaluator before the loss was even formed. bt.one_hot does the same work
    // in one native scatter (phase A5), and does the range check with two
    // reductions instead of a per-row branch.
    $onehot = bt.one_hot($targets, $k, nd_dtype($logp.data));
    if ($onehot.shape()[0] != $n) {
        throw "btorch.nn.nll_loss: " + str($onehot.shape()[0]) + " targets for "
            + str($n) + " rows of predictions";
    }
    $picked = bt.sum(bt.mul($logp, $onehot), 1, null);
    return _reduce(bt.neg($picked), $reduction);
}
def cross_entropy($logits, $targets, $reduction) {
    return nll_loss(log_softmax($logits), $targets, $reduction);
}
def _asIntArray($t) {
    if (type($t) == "instance") { return $t.data; }
    if (_isNd($t)) { return $t; }
    return nd($t, "f64");
}

// ── binary cross entropy with logits ────────────────────────────────────────
// max(x,0) - x*y + log(1 + exp(-|x|)) -- the stable form. The naive
// log(sigmoid(x)) is -inf once sigmoid saturates.
def bce_with_logits($logits, $targets, $reduction) {
    $x = $logits;
    $y = $targets;
    if (type($y) != "instance") { $y = bt.tensor($y, false); }
    $relu = bt.relu($x);
    $xy = bt.mul($x, $y);
    $softplus = bt.log(bt.add(bt.exp(bt.neg(bt.abs_($x))), 1.0));
    return _reduce(bt.add(bt.sub($relu, $xy), $softplus), $reduction);
}
def bce_loss($probs, $targets, $reduction) {
    $y = $targets;
    if (type($y) != "instance") { $y = bt.tensor($y, false); }
    $eps = 0.0000000000001;
    $p = bt.add($probs, $eps);
    $q = bt.add(bt.sub(1.0, $probs), $eps);
    $loss = bt.neg(bt.add(bt.mul($y, bt.log($p)),
                          bt.mul(bt.sub(1.0, $y), bt.log($q))));
    return _reduce($loss, $reduction);
}

// ── loss objects, for the `$crit.forward(pred, target)` style ──────────────
class _MSELoss extends Module {
    def init($reduction) { super(); this.reduction = $reduction; }
    def forward2($p, $t) { return mse_loss($p, $t, this.reduction); }
}
class _CrossEntropyLoss extends Module {
    def init($reduction) { super(); this.reduction = $reduction; }
    def forward2($p, $t) { return cross_entropy($p, $t, this.reduction); }
}
class _L1Loss extends Module {
    def init($reduction) { super(); this.reduction = $reduction; }
    def forward2($p, $t) { return l1_loss($p, $t, this.reduction); }
}
def MSELoss($r)          { return new _MSELoss($r); }
def CrossEntropyLoss($r) { return new _CrossEntropyLoss($r); }
def L1Loss($r)           { return new _L1Loss($r); }

// ── gradient clipping ───────────────────────────────────────────────────────
// Scales every gradient by max_norm/total_norm when the global L2 norm of all
// gradients exceeds max_norm. The norm is over ALL parameters jointly, which is
// what "clip_grad_norm_" means -- clipping each tensor separately is a
// different operation and a common mistake.
def clip_grad_norm($params, $max_norm) {
    $total = 0.0;
    $i = 0;
    while ($i < len($params)) {
        $g = $params[$i].grad;
        // A full reduction returns a 0-d ARRAY, not a number (NumPy's
        // convention), so the single value has to be pulled out -- adding the
        // handle straight onto $total would make $total an array and the
        // nd_full() below would refuse it.
        if ($g != null) {
            $sq = nd_sum(nd_multiply($g, $g, null), null, null);
            $total = $total + nd_get(nd_reshape($sq, [1]), [0]);
        }
        $i = $i + 1;
    }
    $norm = nd_get(nd_sqrt(nd_full([1], $total), null), [0]);
    if ($norm <= $max_norm || $norm == 0) { return $norm; }
    $scale = $max_norm / $norm;
    $i = 0;
    while ($i < len($params)) {
        $g = $params[$i].grad;
        if ($g != null) { nd_multiply($g, $scale, $g); }
        $i = $i + 1;
    }
    return $norm;
}
