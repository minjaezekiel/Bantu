// ════════════════════════════════════════════════════════════════════════════
//  optim.b — optimizers and learning-rate schedules.
//
//      include "btorch/optim.b" as optim;
//
//      $opt = optim.Adam($model.parameters(), 0.001, null, null, null);
//      ...
//      $opt.zero_grad();
//      $loss.backward(null);
//      $opt.step();
//
//  Every update runs under bt.no_grad: a parameter update is not part of the
//  computation being differentiated, and recording it would both build a tape
//  nobody reads and make the next backward() walk into the previous step.
//
//  The tests assert the CLOSED FORM of each update, not that a loss went down.
//  Gradient descent on a wrong gradient still often reduces a loss, so "it
//  trains" is not evidence (DECISIONS B10). Adam's bias correction in
//  particular is the part everyone gets wrong, so it is asserted directly at
//  t = 1, 2 and 3.
// ════════════════════════════════════════════════════════════════════════════

include "./btorch.b" as bt;

// Every optimizer shares this. Parameters are held by reference, so the
// optimizer updates the very tensors the model uses.
class Optimizer {
    def init($params, $lr) {
        this.__bt_optim = true;
        if ($params == null || len($params) == 0) {
            throw "btorch.optim: the optimizer was given no parameters "
                + "(did you call $model.parameters()?)";
        }
        if ($lr == null) {
            throw "btorch.optim: a learning rate is required";
        }
        if ($lr <= 0) {
            throw "btorch.optim: the learning rate must be positive, got " + str($lr);
        }
        this.params = $params;
        this.lr = $lr;
        this.t = 0;              // step count, for bias correction
        this.state = {};         // param id -> per-parameter state
    }

    def zero_grad() {
        $i = 0;
        while ($i < len(this.params)) { this.params[$i].grad = null; $i = $i + 1; }
        return this;
    }

    // Per-parameter state, created on first use and keyed by tensor identity.
    def _slot($p, $name) {
        $key = str($p.id) + "/" + $name;
        $v = this.state[$key];
        if ($v == null) {
            $v = nd_zeros(nd_shape($p.data), "f64");
            this.state[$key] = $v;
        }
        return $v;
    }

    def set_lr($lr) { this.lr = $lr; return this; }
    def get_lr() { return this.lr; }
}

// ── SGD, with momentum, Nesterov and weight decay ──────────────────────────
//
//   g      = grad + wd * p
//   buf    = momentum * buf + g
//   update = g + momentum * buf     (Nesterov)   or   buf   (classical)
//   p     -= lr * update
class _SGD extends Optimizer {
    def init($params, $lr, $momentum, $weight_decay, $nesterov) {
        super($params, $lr);
        this.momentum = $momentum; if ($momentum == null) { this.momentum = 0.0; }
        this.weight_decay = $weight_decay; if ($weight_decay == null) { this.weight_decay = 0.0; }
        this.nesterov = $nesterov == true;
        if (this.nesterov && this.momentum <= 0) {
            throw "btorch.optim.SGD: Nesterov momentum needs momentum > 0";
        }
    }
    def step() {
        $self = this;
        bt.no_grad(def() {
            $i = 0;
            while ($i < len($self.params)) {
                $p = $self.params[$i];
                if ($p.grad != null) {
                    $g = nd_copy($p.grad, null);
                    if ($self.weight_decay != 0) {
                        nd_add($g, nd_multiply($p.data, $self.weight_decay, null), $g);
                    }
                    if ($self.momentum != 0) {
                        $buf = $self._slot($p, "momentum");
                        // buf = momentum*buf + g
                        nd_multiply($buf, $self.momentum, $buf);
                        nd_add($buf, $g, $buf);
                        if ($self.nesterov) { nd_add($g, nd_multiply($buf, $self.momentum, null), $g); }
                        else { $g = nd_copy($buf, null); }
                    }
                    nd_subtract($p.data, nd_multiply($g, $self.lr, null), $p.data);
                }
                $i = $i + 1;
            }
            return null;
        });
        this.t = this.t + 1;
        return this;
    }
}
def SGD($params, $lr, $momentum, $weight_decay, $nesterov) {
    return new _SGD($params, $lr, $momentum, $weight_decay, $nesterov);
}

// ── Adam ───────────────────────────────────────────────────────────────────
//
//   m = b1*m + (1-b1)*g
//   v = b2*v + (1-b2)*g^2
//   mhat = m / (1 - b1^t)        <- the bias correction
//   vhat = v / (1 - b2^t)
//   p   -= lr * mhat / (sqrt(vhat) + eps)
//
// The bias correction is what makes the FIRST step a full-size step rather
// than a tiny one: at t=1, mhat = g and vhat = g^2, so the update is exactly
// -lr * sign(g) up to eps. That identity is asserted in the tests, because an
// Adam missing its correction still trains -- just wrongly and slowly.
class _Adam extends Optimizer {
    def init($params, $lr, $beta1, $beta2, $eps) {
        super($params, $lr);
        this.beta1 = $beta1; if ($beta1 == null) { this.beta1 = 0.9; }
        this.beta2 = $beta2; if ($beta2 == null) { this.beta2 = 0.999; }
        this.eps = $eps; if ($eps == null) { this.eps = 0.00000001; }
        this.weight_decay = 0.0;
        this.decoupled = false;        // AdamW sets this
    }
    def step() {
        this.t = this.t + 1;
        $self = this;
        bt.no_grad(def() {
            $t = $self.t;
            // pow() on plain numbers: these are scalars, not arrays
            $bc1 = 1.0 - nd_get(nd_power(nd_full([1], $self.beta1), $t, null), [0]);
            $bc2 = 1.0 - nd_get(nd_power(nd_full([1], $self.beta2), $t, null), [0]);
            $i = 0;
            while ($i < len($self.params)) {
                $p = $self.params[$i];
                if ($p.grad != null) {
                    $g = nd_copy($p.grad, null);
                    // Coupled (classic Adam) weight decay folds into the
                    // gradient; decoupled (AdamW) is applied to the parameter
                    // directly and is NOT adapted by v. They are genuinely
                    // different algorithms, not a reparameterisation.
                    if ($self.weight_decay != 0 && !$self.decoupled) {
                        nd_add($g, nd_multiply($p.data, $self.weight_decay, null), $g);
                    }
                    $m = $self._slot($p, "m");
                    $v = $self._slot($p, "v");
                    // m = b1*m + (1-b1)*g
                    nd_multiply($m, $self.beta1, $m);
                    nd_add($m, nd_multiply($g, 1.0 - $self.beta1, null), $m);
                    // v = b2*v + (1-b2)*g*g
                    nd_multiply($v, $self.beta2, $v);
                    nd_add($v, nd_multiply(nd_multiply($g, $g, null), 1.0 - $self.beta2, null), $v);
                    $mhat = nd_divide($m, $bc1, null);
                    $vhat = nd_divide($v, $bc2, null);
                    $upd = nd_divide($mhat, nd_add(nd_sqrt($vhat, null), $self.eps, null), null);
                    if ($self.weight_decay != 0 && $self.decoupled) {
                        nd_add($upd, nd_multiply($p.data, $self.weight_decay, null), $upd);
                    }
                    nd_subtract($p.data, nd_multiply($upd, $self.lr, null), $p.data);
                }
                $i = $i + 1;
            }
            return null;
        });
        return this;
    }
}
def Adam($params, $lr, $beta1, $beta2, $eps) {
    return new _Adam($params, $lr, $beta1, $beta2, $eps);
}

// ── AdamW: Adam with DECOUPLED weight decay ────────────────────────────────
def AdamW($params, $lr, $weight_decay, $beta1, $beta2) {
    $o = new _Adam($params, $lr, $beta1, $beta2, null);
    $o.weight_decay = $weight_decay;
    if ($weight_decay == null) { $o.weight_decay = 0.01; }
    $o.decoupled = true;
    return $o;
}

// ── RMSprop ────────────────────────────────────────────────────────────────
//   v = alpha*v + (1-alpha)*g^2 ;  p -= lr*g/(sqrt(v)+eps)
class _RMSprop extends Optimizer {
    def init($params, $lr, $alpha, $eps) {
        super($params, $lr);
        this.alpha = $alpha; if ($alpha == null) { this.alpha = 0.99; }
        this.eps = $eps; if ($eps == null) { this.eps = 0.00000001; }
    }
    def step() {
        $self = this;
        bt.no_grad(def() {
            $i = 0;
            while ($i < len($self.params)) {
                $p = $self.params[$i];
                if ($p.grad != null) {
                    $g = $p.grad;
                    $v = $self._slot($p, "v");
                    nd_multiply($v, $self.alpha, $v);
                    nd_add($v, nd_multiply(nd_multiply($g, $g, null), 1.0 - $self.alpha, null), $v);
                    $upd = nd_divide($g, nd_add(nd_sqrt($v, null), $self.eps, null), null);
                    nd_subtract($p.data, nd_multiply($upd, $self.lr, null), $p.data);
                }
                $i = $i + 1;
            }
            return null;
        });
        this.t = this.t + 1;
        return this;
    }
}
def RMSprop($params, $lr, $alpha, $eps) { return new _RMSprop($params, $lr, $alpha, $eps); }

// ── Adagrad ────────────────────────────────────────────────────────────────
//   s += g^2 ;  p -= lr*g/(sqrt(s)+eps)
class _Adagrad extends Optimizer {
    def init($params, $lr, $eps) {
        super($params, $lr);
        this.eps = $eps; if ($eps == null) { this.eps = 0.0000000001; }
    }
    def step() {
        $self = this;
        bt.no_grad(def() {
            $i = 0;
            while ($i < len($self.params)) {
                $p = $self.params[$i];
                if ($p.grad != null) {
                    $g = $p.grad;
                    $s = $self._slot($p, "sum");
                    nd_add($s, nd_multiply($g, $g, null), $s);
                    $upd = nd_divide($g, nd_add(nd_sqrt($s, null), $self.eps, null), null);
                    nd_subtract($p.data, nd_multiply($upd, $self.lr, null), $p.data);
                }
                $i = $i + 1;
            }
            return null;
        });
        this.t = this.t + 1;
        return this;
    }
}
def Adagrad($params, $lr, $eps) { return new _Adagrad($params, $lr, $eps); }

// ════════════════════════════════════════════════════════════════════════════
//  Learning-rate schedules
//
//  Each holds the optimizer and rewrites its lr on step(). base_lr is captured
//  at construction, so a schedule is a pure function of the epoch count rather
//  than a repeated multiplication that drifts.
// ════════════════════════════════════════════════════════════════════════════
class _StepLR {
    def init($opt, $step_size, $gamma) {
        this.opt = $opt;
        this.step_size = $step_size;
        this.gamma = $gamma; if ($gamma == null) { this.gamma = 0.1; }
        this.base_lr = $opt.lr;
        this.epoch = 0;
    }
    def step() {
        this.epoch = this.epoch + 1;
        $k = (this.epoch - (this.epoch % this.step_size)) / this.step_size;
        this.opt.lr = this.base_lr * nd_get(nd_power(nd_full([1], this.gamma), $k, null), [0]);
        return this.opt.lr;
    }
}
class _ExponentialLR {
    def init($opt, $gamma) {
        this.opt = $opt;
        this.gamma = $gamma;
        this.base_lr = $opt.lr;
        this.epoch = 0;
    }
    def step() {
        this.epoch = this.epoch + 1;
        this.opt.lr = this.base_lr * nd_get(nd_power(nd_full([1], this.gamma), this.epoch, null), [0]);
        return this.opt.lr;
    }
}
// Cosine annealing: lr goes from base_lr to eta_min over T_max epochs.
class _CosineAnnealingLR {
    def init($opt, $T_max, $eta_min) {
        this.opt = $opt;
        this.T_max = $T_max;
        this.eta_min = $eta_min; if ($eta_min == null) { this.eta_min = 0.0; }
        this.base_lr = $opt.lr;
        this.epoch = 0;
    }
    def step() {
        this.epoch = this.epoch + 1;
        $frac = this.epoch / this.T_max;
        if ($frac > 1.0) { $frac = 1.0; }
        $cos = nd_get(nd_cos(nd_full([1], 3.141592653589793 * $frac), null), [0]);
        this.opt.lr = this.eta_min + (this.base_lr - this.eta_min) * (1.0 + $cos) / 2.0;
        return this.opt.lr;
    }
}
def StepLR($o, $s, $g)              { return new _StepLR($o, $s, $g); }
def ExponentialLR($o, $g)           { return new _ExponentialLR($o, $g); }
def CosineAnnealingLR($o, $t, $e)   { return new _CosineAnnealingLR($o, $t, $e); }
