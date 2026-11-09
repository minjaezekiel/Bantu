# btorch — architecture

A PyTorch-shaped deep-learning library for Bantu: tensors with automatic
differentiation, an `nn.Module` system, optimizers, data loading and
serialization. The public API is **pure Bantu**. The *atoms* it stands on are
native, exactly as `arctic` stands on `col_*` and `numba` stands on `nd_*`.

> **The one-sentence design.** btorch is a reverse-mode autodiff tape plus a
> module system written in Bantu on top of numba's `NdArray`; the only native
> code it adds is the handful of kernels that a tape *cannot* express at
> acceptable cost — scatter-add, concat/pad/repeat, and `im2col`.

Companion documents: [`../btorch-suite/ROADMAP.md`](../btorch-suite/ROADMAP.md)
(phases and their gates), [`../btorch-suite/DECISIONS.md`](../btorch-suite/DECISIONS.md)
(numbered decisions, `B1`…), [`../btorch-suite/CHANGELOG.md`](../btorch-suite/CHANGELOG.md)
(what each phase actually measured).

---

## 1. What this has to be

"Same features as PyTorch" is the brief. PyTorch is enormous, so the brief is
read as **the training loop and everything it touches**, to a standard where a
real model trains to a real accuracy:

```bantu
include "btorch" as bt;
include "btorch/nn.b" as nn;

class MLP extends nn.Module {
    def init() {
        super();
        this.add("fc1", nn.Linear(784, 128));
        this.add("fc2", nn.Linear(128, 10));
    }
    def forward($x) {
        return this.get("fc2").forward(this.get("fc1").forward($x).relu());
    }
}

$model = new MLP();
$opt   = bt.optim.Adam($model.parameters(), 0.001);
$loss  = nn.CrossEntropyLoss();

for $xb, $yb in $loader.batches() {
    $opt.zero_grad();
    $out = $model.forward($xb);
    $l   = $loss.forward($out, $yb);
    $l.backward();
    $opt.step();
}
```

Everything in that snippet is in scope. The deferred list in §10 is the honest
boundary: no GPU, no JIT, no distributed training, and f64 before f32.

### 1.1 What "entirely in Bantu" means here

It means the same thing it means for `arctic` and `numba`, and the project has
been consistent about it: **the library is Bantu; the atoms are native.** A
tree-walking interpreter costs ~1.8 µs per function call (measured, §3), so a
per-element loop in Bantu is four orders of magnitude off the pace. The split
is therefore not a compromise, it is the architecture:

| layer | language | why |
|---|---|---|
| `Tensor`, autograd tape, `nn.*`, `optim.*`, data, serialization | **Bantu** | one call per *tensor op*, not per element — the interpreter cost amortizes over thousands of elements |
| `nd_*` (171 kernels, already shipped) | native | the inner loops |
| `nd_index_add`, `nd_concat`, `nd_pad`, `nd_repeat`, `nd_im2col` (new, §4) | native | a tape cannot express these at acceptable cost |

The test for whether a piece belongs in Bantu is arithmetic: an op on a
128×256 tensor touches 32,768 elements. At ~4 µs of Bantu bookkeeping per op
(measured: 3.85 µs for `new` + a method call) that is **0.12 ns of overhead per
element** against a kernel floor of ~1.3 ns. Bookkeeping is ~9% — acceptable.
At size 1 it is 100%, which is why btorch is a *tensor* library and tells users
so.

---

## 2. The foundation, measured

Everything below was measured on this machine (Apple silicon, `./bantu`
v1.3.2) rather than assumed. These are the numbers the design is built on.

| what | measured | note |
|---|---|---|
| `nd_matmul` 128×784×256 | **6.8 ms, 7.56 GFLOP/s** | the forward pass of an MLP layer, batch 128 |
| `nd_add(arr, arr)` 1M | **1.76 ms, 14.3 GB/s** | tier 0, the good path |
| `nd_maximum(arr, arr)` 1M | **1.54 ms, 16.3 GB/s** | tier 0 |
| `nd_exp` 1M | 6.45 ms | libm-bound, ~6.4 ns/element |
| `nd_tanh` 1M | 8.25 ms | libm-bound |
| Bantu function call | **1.78 µs** | the autograd overhead unit |
| `new` + method call | **3.85 µs** | the cost of one tape node |
| fixed cost of an `nd_*` call | **~5 µs** | at size 1; sets the minimum useful tensor size |

numba supplies 171 `nd_*` kernels: creation, 55 ufuncs with broadcasting,
reductions with `axis`/`keepdims` (pairwise-accurate to 2.33e-16), scans,
sorting, fancy and boolean indexing, zero-copy views
(`reshape`/`transpose`/`slice`/`broadcast_to`/`flip`/`moveaxis`), and a
complete linalg set (`matmul`/`solve`/`inv`/`det`/`qr`/`svd`/`eigh`/`lstsq`).

**That is most of a tensor library already.** The autograd engine is new work;
the numerics mostly are not.

### 2.1 Language facts the design must respect

Probed, not assumed:

- **Closures capture by reference, not by value** — `for` loop bodies that
  build closures over the loop variable all see its *final* value (measured:
  three closures over `$k` returned `102,102,102`, not `100,101,102`). This is
  Python's semantics, not a defect. **Consequence: every backward closure is
  built by a factory function that takes its saved values as parameters.** This
  is decision **B3** and it is not optional — it is the difference between
  correct and silently-wrong gradients.
- Closures **do** survive being stored in an object field and called later, and
  they capture `NdArray` handles correctly. The tape design depends on this and
  it works.
- Classes support `init`, `this`, methods, `new`, single inheritance via
  `extends`, and **virtual dispatch** (a parent method calling `this.name()`
  reaches the child's override — verified).
- `super($args)` calls the parent constructor. `super.method()` does **not**
  work (§3, defect D4).
- `nd_*` operator dispatch works on handles: `$a + $b`, `$a * 2`, `$a[0]`,
  `$a[1][2] = 9`, and writes through views land in the base buffer.

---

## 3. The measured walls

Four defects and five missing primitives were found while probing the
foundation for this design. Each is recorded here because each one shaped the
plan, and three of them are **silent wrong answers or crashes that affect code
already shipped** — not btorch-specific wishes.

### D1 — the tier-1 splat kernel is dead code *(4.5× on every scalar op)*

`ndarray_kernels.hpp:206-215` defines `tier1` and `tier1r`, with a comment
explaining why they earn their keep ("the scalar stays in a register instead of
being re-read through a stride-0 walk"). **Neither is ever called.** A
tree-wide search finds exactly two mentions: the two definitions. `tier0` has
ten call sites.

So `applyBinary` goes tier 0 → tier 2, and any operand whose shape differs —
i.e. **every scalar** — falls to the general strided odometer with a
`loadD`/`storeD` per element:

| call | measured | path |
|---|---|---|
| `nd_add($a, $b)` | 1.76 ms / 14.3 GB/s | tier 0 |
| `nd_add($a, 1.0)` | **7.86 ms / 3.2 GB/s** | tier 2 |
| `nd_maximum($a, $b)` | 1.54 ms / 16.3 GB/s | tier 0 |
| `nd_maximum($a, 0.0)` | **7.38 ms / 3.4 GB/s** | tier 2 |
| `nd_multiply($a, 2.0)` | **7.52 ms / 3.3 GB/s** | tier 2 |

A scalar operand does *less* memory traffic and runs **4.5× slower**. This is
not an exotic path: it is `relu(x) = maximum(x, 0)`, bias add, `grad * lr`,
`p - lr*g`, the `1/n` in every mean, and every loss normalization — the
arithmetic a neural net is mostly made of. numba's ROADMAP marks "tier 1 splat"
done with a ✅; the kernel was written and never wired in.

Fixing it is ~20 lines in `applyBinary`. It is **A1**, the first thing btorch
does, and it speeds up numba and arctic too.

### D2 — `$a + $b` on class instances silently returns 0

There is no dunder dispatch for `CLASS_INSTANCE` in `evalBinaryOp`, and no
error either. Instances fall through to
`case PLUS: return Value(left.numberVal + right.numberVal)`, and `numberVal` is
0 for an instance — so `$a + $b` is **0**, and `$a + $b` then `.x` reads as
`null`.

This is the *third* instance of a defect this file has already fixed twice: the
comments at `evaluator.hpp:3818` and `:3840` document `$handle + 1` silently
giving 1 and `[1] + [2,3]` silently giving 0, both now fixed. Instances are the
same bug, still open, and it affects **every class already shipped** —
`arctic.Series`, `DataFrame`, `pillow.Image`.

For btorch it is also the difference between

```bantu
$y = $x.matmul($w).add($b);          // works today
$y = $x @ $w + $b;                   // what a PyTorch user writes
```

Fix: dispatch `__add__`/`__sub__`/`__mul__`/`__truediv__`/`__pow__`/`__neg__`/
`__matmul__`/`__eq__`/`__lt__`… to a method of that name when the operand is an
instance, and **raise a named error** when there is no such method instead of
returning 0. That is **A2**.

### D3 — a three-level `super()` chain segfaults the interpreter

```bantu
class Base       { def init($n) { this.n = $n; } }
class Child      extends Base  { def init($n) { super($n); this.extra = 42; } }
class GrandChild extends Child { def init($n) { super($n); this.g = 1; } }
$g = new GrandChild(3);                       // → SIGSEGV, exit 139, no output
```

Two levels work. Three crash, with **no diagnostic whatsoever** — no error, no
message, no partial output, just exit 139.

Root cause, from the code: the `super()` handler (`evaluator.hpp:~6190`) reads
`currentClassName_` to find the parent, then runs the parent constructor's body
via `runStatements(fn->body)` — **without setting `currentClassName_` to the
parent**. Every assignment to that member is at lines 6157, 6176, 6281 and
6288; none is in the `super()` path. So `super($n)` inside `Child.init` still
resolves `currentClassName_ == "GrandChild"`, finds parent `Child`, and calls
`Child.init` again — unbounded recursion until the stack dies.

`nn.Module` hierarchies are naturally three deep (`Module → _Norm → BatchNorm1d`),
and a user subclassing any btorch layer makes a third level without thinking
about it. Fix: save, set and restore `currentClassName_` around the parent
constructor call, which also makes the recursion finite and therefore
diagnosable. That is **A3**.

### D4 — `super.method()` is parsed but unbound

`parser.hpp:729` turns `super.` into `VariableNode("super")`, which evaluates
to `Undefined variable: super`. Only `super(...)` (the constructor) is wired.
Low priority — btorch's own hierarchy can be written without it — but it is a
keyword that lexes, parses, and then fails at runtime, so it is recorded and
fixed with A3 while that code is open.

### D5…D9 — primitives a tape cannot express cheaply

| missing | needed by | why not composed in Bantu |
|---|---|---|
| **scatter-add** (`nd_index_add`) ✅ **built, phase A5** | backward of `take`/`gather`/indexing, and **every** embedding layer | `nd_put` *overwrites*; duplicate indices must **accumulate**. Composing it needs a Bantu loop per index |
| **`nd_concat` / `nd_split`** ✅ **built, A6** | `cat`, `stack`, `chunk`, RNN gates, backward of both | composable from `nd_zeros` + slice-assign, but that is one interpreter round-trip per slice |
| **`nd_pad`** ✅ **built, A7** | `conv2d(padding=…)`, and the backward of cropping | same |
| **`nd_repeat` / `nd_tile`** ✅ **built, A8** | `expand`, broadcasting backward, `repeat_interleave` | same |
| **`nd_im2col` / `nd_col2im`** ✅ **built, A9** | `conv1d`/`conv2d` forward **and** backward | the whole point is to turn convolution into one `matmul`; doing the gather in Bantu costs more than the convolution |

`nd_put` also materializes the destination's entire flat index list on **every
call** (`ndarray_sort_reg.hpp:415-417`), so it is O(n) in the *array* rather
than O(k) in the *indices* — a scatter of 10 values into a 1M tensor walks 1M
offsets. `nd_index_add` will not repeat that.

These land in **A5-A9**, each with a Bantu reference implementation in the test
so the native kernel is checked against something independent.

**A5 is done, and it set the pattern for the rest.** `nd_index_add` was built
as the exact *adjoint* of `nd_take` — same axis convention, same default — and
tested with the adjoint identity `⟨take(x,ix), y⟩ == ⟨x, index_add(0,ix,y)⟩`
rather than only with examples. That identity pins the axis, the strides and
the accumulation at once, and no kernel that merely agrees with itself can
satisfy it. `nd_col2im` is `nd_im2col`'s adjoint in exactly the same sense and
should be tested the same way. Measured: the cost is **flat** across a 20×
change in destination size (24 ms and 21 ms for the same 1M indices), which is
the property `nd_put` lacks.

---

## 4. The core: `Tensor` and the tape

### 4.1 `Tensor`

```
class Tensor {
    data            // NdArray handle — the values
    grad            // NdArray handle or null — accumulated gradient
    requires_grad   // bool
    ctx             // Node or null — the op that produced this tensor
    is_leaf         // bool — created by the user, not by an op
}
```

`data` is a plain numba array, so every one of the 171 kernels is reachable and
zero-copy views stay zero-copy. `grad` is allocated lazily — a tensor that
never receives a gradient never pays for one.

### 4.2 The tape is the graph

btorch is **define-by-run**, like PyTorch: there is no graph object. Each op,
when at least one input requires grad and grad mode is on, attaches a `Node` to
its output:

```
class Node {
    name            // "add", "matmul" — for error messages and debugging
    parents         // list of Tensor: the inputs
    backward        // closure: (NdArray gradOut) -> list of NdArray|null
}
```

`backward` returns one gradient per parent, positionally, with `null` for a
parent that needs none (an integer index tensor, say). The *shape contract* is
absolute: **the gradient returned for parent `i` has exactly parent `i`'s
shape.** Enforcing that at the accumulation point (§4.5) is what makes
broadcasting bugs findable instead of mysterious.

### 4.3 B3 — backward closures are built by factories

Because closures capture by reference (§2.1), this is **wrong**:

```bantu
// WRONG: $saved is the enclosing scope's variable, whatever it ends up as
def mul($a, $b) {
    $saved = $b.data;
    $out.ctx = new Node("mul", [$a, $b], def($g) { return [nd_multiply($g, $saved)]; });
}
```

and this is right:

```bantu
// RIGHT: $bd is this invocation's own parameter, bound once
def _mulBackward($ad, $bd) {
    return def($g) { return [nd_multiply($g, $bd), nd_multiply($g, $ad)]; };
}
```

Every backward in btorch is produced by a `_*Backward(...)` factory. It is a
mechanical rule, it is enforced by review, and the autograd test suite includes
a case that builds ten nodes in a loop and checks all ten gradients differ —
the exact shape of the bug this rule prevents.

**Factories capture `NdArray` data, never `Tensor`s.** A closure holding a
`Tensor` that holds the `Node` that holds the closure is a reference cycle; a
closure holding raw arrays is a DAG that frees itself. This also keeps the
graph's memory to the tensors actually needed for the backward pass, which is
what `requires_grad=false` on an input is *for*.

### 4.4 `backward()`

```
1. toposort  — DFS from the root over ctx.parents, post-order, visited-marked
2. seed      — root.grad = ones(root.shape), or the caller's gradient;
               a non-scalar root without an explicit gradient is an ERROR,
               named, exactly as PyTorch does
3. sweep     — in reverse topological order, for each node:
                 g = node.output_grad
                 for (parent, pgrad) in zip(node.parents, node.backward(g)):
                     accumulate(parent, pgrad)
4. free      — drop ctx on non-leaf tensors unless retain_graph
```

Reverse topological order (not naive DFS) is required: a tensor used twice must
receive **both** contributions before it propagates further. A diamond
(`$y = $x.mul($x).add($x)`) is the smallest test that distinguishes a correct
engine from one that looks correct, and it is in the suite.

### 4.5 The broadcast-gradient rule

This is where hand-written autodiff engines usually break. If forward
broadcast a `[3]` bias against a `[128,3]` activation, the gradient arriving is
`[128,3]` and the bias's gradient must be `[3]` — **summed over the broadcast
axes**, not reshaped, not sliced:

```
_unbroadcast(g, shape):
    while ndim(g) > len(shape):            # leading axes forward added
        g = nd_sum(g, 0)
    for each axis i where shape[i] == 1 and g.shape[i] > 1:
        g = nd_sum(g, i, keepdims=true)    # stretched axes
    return nd_reshape(g, shape)
```

Every binary op's backward ends by passing each gradient through
`_unbroadcast` against its parent's shape. `accumulate()` then asserts the
shape matches before adding, so a missing `_unbroadcast` fails loudly in the
test that introduced it rather than silently training a wrong model.

Gradient accumulation is `+=`, not `=`: `grad` starts null, the first
contribution is stored (a copy, so the caller's array is never aliased into the
tensor), and later ones are added in place.

### 4.6 Grad mode

A module-level flag, `no_grad` taking a closure:

```bantu
bt.no_grad(def() { $acc = evaluate($model, $test); });
```

Inside, ops attach no `Node`, so inference allocates no tape at all. The flag
is saved and restored around the call, including on the exception path — an
error inside a `no_grad` block must not leave grad mode off for the rest of the
process. Optimizer steps run under it too.

### 4.7 In-place ops and the version counter

PyTorch guards in-place mutation of a tensor the tape needs with a version
counter. btorch's phase 1 takes the simpler, safer position: **the ops that
build the graph are pure**, and the in-place family (`add_`, `mul_`,
`zero_`, and optimizer updates) is only legal on leaf tensors under `no_grad`,
which is checked and raises by name otherwise. That removes the entire class of
"your gradient is wrong because something overwrote a saved tensor" bug at the
cost of a little memory, and a version counter can be added later without an
API change.

---

## 5. The op table

Each entry is forward (a numba call) and backward (the factory). This is the
complete Phase 2 contract; the ✅ column is filled in as it lands.

| op | forward | backward w.r.t. inputs |
|---|---|---|
| `add` | `nd_add` | `g`, `g` (both `_unbroadcast`) |
| `sub` | `nd_subtract` | `g`, `-g` |
| `mul` | `nd_multiply` | `g*b`, `g*a` |
| `div` | `nd_divide` | `g/b`, `-g*a/b²` |
| `pow` | `nd_power` | `g*b*a^(b-1)`, `g*out*ln a` |
| `neg` | `nd_negative` | `-g` |
| `matmul` | `nd_matmul` | `g @ bᵀ`, `aᵀ @ g` (+ batch unbroadcast) |
| `sum` | `nd_sum` | `g` broadcast back to input shape |
| `mean` | `nd_mean` | `g / n`, broadcast back |
| `max`/`min` | `nd_max` | a tie **splits** `g` evenly — amax semantics, decided in DECISIONS B16, not the argmax routing originally sketched here |
| `exp` | `nd_exp` | `g * out` |
| `log` | `nd_log` | `g / a` |
| `sqrt` | `nd_sqrt` | `g / (2*out)` |
| `tanh` | `nd_tanh` | `g * (1 - out²)` |
| `sigmoid` | composed | `g * out * (1-out)` |
| `relu` | `nd_maximum(a,0)` | `g * (a > 0)` |
| `reshape`/`view` | `nd_reshape` | `g` reshaped to the input's shape |
| `transpose`/`T` | `nd_transpose` | `g` transposed back |
| `slice`/`index` | `nd_slice` | zeros of input shape, `g` scattered in |
| `gather`/`take` | `nd_take` | **`nd_index_add`** (A5 ✅) |
| `cat`/`stack` | **`nd_concat`** (A6 ✅) | `g` split along the axis |
| `softmax` | composed, max-shifted | `(g - (g·out).sum) * out` |
| `log_softmax` | composed | `g - exp(out) * g.sum` |

**Numerical stability is part of the contract, not a refinement.** `softmax`
subtracts the row max before exponentiating; `log_softmax` uses the
log-sum-exp form; `CrossEntropyLoss` fuses `log_softmax` with the negative
log-likelihood so it never materializes a probability that can underflow to
zero and produce `-inf`. The test for each is an input like `[1000, 1001]` that
a naive implementation turns into `NaN`.

### 5.1 How gradients are verified

Every gradient is checked **twice**, and the second check is the one that
matters:

1. **Analytic** — against a value worked out by hand, in the test.
2. **Numerical** — central-difference gradient checking,
   `(f(x+h) - f(x-h)) / 2h` with `h = 1e-5` in f64, asserted to agree within
   `1e-7` relative. This is `bt.gradcheck($f, $inputs)`, it ships as part of
   the library, and **every op in the table above has a gradcheck case**.

A hand-derived gradient and a numerical gradient are wrong in different ways,
so agreement between them is real evidence. This is the differential method
that `pillow` used against Pillow and `numba` used against NumPy — here the
independent reference is calculus itself.

---

## 6. `nn.Module`

```
class Module {
    _params     // dict: name -> Tensor
    _buffers    // dict: name -> Tensor (no grad; running stats, masks)
    _modules    // dict: name -> Module
    training    // bool
}
```

- `parameters()` / `named_parameters()` — recursive, **deduplicated by identity**
  so a weight shared between two layers is returned once and stepped once.
  (Tied embeddings are a real pattern; stepping a shared weight twice is a
  silent 2× learning rate.)
- `zero_grad()`, `train()`, `eval()` — recursive.
- `state_dict()` / `load_state_dict()` — flat `"fc1.weight"` keys.
- `forward($x)` — the subclass overrides this. Virtual dispatch through a
  parent method is verified to work (§2.1), so `Sequential` can call
  `child.forward()` polymorphically.

Registration is explicit — `this.add("fc1", nn.Linear(784,128))` — rather than
Python's attribute interception, because Bantu has no `__setattr__` hook. It is
three characters more typing and it makes the parameter tree inspectable.

**Layers:** `Linear`, `Sequential`, `ReLU`/`Sigmoid`/`Tanh`/`GELU`/`LeakyReLU`,
`Dropout`, `Flatten`, `LayerNorm`, `BatchNorm1d`, `Embedding`, `Conv1d`,
`Conv2d`, `MaxPool2d`, `AvgPool2d`, `RNN`, `LSTM`, `GRU`,
`MultiheadAttention`, `TransformerEncoderLayer`.

**Initialization matters and is not an afterthought.** `Linear` defaults to
Kaiming-uniform on `fan_in` (PyTorch's default), and the test asserts the
*variance* of a 1000×1000 initialized weight against the analytic
`2/fan_in` — a wrong initializer trains slowly rather than failing, which makes
it exactly the kind of bug a unit test has to catch because a human will not.

**Losses:** `MSELoss`, `L1Loss`, `CrossEntropyLoss`, `NLLLoss`, `BCELoss`,
`BCEWithLogitsLoss`, `HuberLoss`, each with `reduction` ∈ `mean|sum|none`.

**Optimizers:** `SGD` (momentum, Nesterov, weight decay), `Adam`, `AdamW`,
`RMSprop`, `Adagrad`; schedulers `StepLR`, `ExponentialLR`, `CosineAnnealingLR`,
`ReduceLROnPlateau`; plus `clip_grad_norm_`.

Adam's test is not "the loss went down" but the closed form: for a constant
gradient, the first step is exactly `-lr * m̂/(√v̂+ε)` with the bias correction
at `t=1`, which is `-lr` up to `ε`. Bias correction is the part everyone gets
wrong, so it is asserted directly at `t=1,2,3`.

---

## 7. Phases

Detail and gates in [`../btorch-suite/ROADMAP.md`](../btorch-suite/ROADMAP.md).

| phase | content |
|---|---|
| **0** | these documents |
| **A** | adoption plumbing: D1 splat, D2 instance operators, D3 `super()` segfault, D4 `super.method`, D5-D9 the five native primitives |
| **1** | `Tensor`, the tape, `backward()`, `_unbroadcast`, `no_grad`, `gradcheck` |
| **2** | the op table of §5, every one gradchecked |
| **3** | `nn.Module`, `Linear`, `Sequential`, activations, `Dropout`, `Flatten`, norms |
| **4** | losses, optimizers, schedulers, grad clipping |
| **5** | `Conv1d`/`Conv2d`, pooling, via `im2col` |
| **6** | `Dataset`, `DataLoader`, batching, shuffling, transforms |
| **7** | `save`/`load`, `state_dict`, inference mode |
| **8** | `Embedding`, `RNN`/`LSTM`/`GRU`, attention, transformer block |
| **9** | performance: threading, f32, fused kernels — each on a measured wall |
| **10** | the package, docs, samples, and **a model trained end to end to a stated accuracy** |

Phase 10's gate is the only one that proves the whole thing: an MLP on MNIST
reaching **≥ 97% test accuracy**, and a CNN reaching **≥ 98.5%**, from a script
in `samples/btorch/` that CI runs. Every unit test can pass while the library
is still unable to train anything; that gate cannot.

---

## 7.5 Data loading and persistence (phases 6 and 7)

### 7.5.1 The data pipeline

Three types, each doing one thing:

| type | contract |
|---|---|
| `Dataset` | `size()` and `get(i)` → a list of tensors. Everything else composes with it |
| `TensorDataset(tensors)` | N tensors sharing a first dimension; `get(i)` is row `i` of each |
| `Subset(dataset, indices)` | a view by index list — what `random_split` returns |
| `DataLoader(dataset, batch_size, shuffle, drop_last, seed)` | an epoch of batches |

**Bantu has no iterator protocol, so `DataLoader` is explicit rather than
magic**: `num_batches()`, `batch(i)`, and `reshuffle()` to start the next epoch.
A `for` loop over batches is then an ordinary `while` over an index, which is
also the form the rest of this library already uses. `epoch()` returns the whole
list for callers that prefer it.

**Shuffling is `nd_argsort(nd_random_uniform([n]))`.** That is a genuine uniform
permutation, it costs one native sort rather than `n` interpreter iterations of
Fisher-Yates, and it draws from numba's seeded stream — so `nd_seed(k)` makes an
epoch's batch order reproducible. Two properties get asserted, and they are
different properties:

1. same seed → **identical** batch order;
2. `reshuffle()` → a **different** order (the stream advances; it does not
   restart).

The third property is the one that actually protects training: with
`drop_last=false`, the indices of one epoch's batches must be a **permutation of
`0..n-1`** — every sample exactly once, none twice. A loader that silently drops
or repeats samples still trains, and still looks fine.

### 7.5.2 The `.btsd` state file

A `state_dict` is tensors, so a text format is out: `nd_to_list` on a 61,706-
parameter model builds 61,706 Bantu numbers, and reparsing them is lossy at the
17th digit unless every one is printed in full. The format is therefore a small
text header with a binary payload:

```
offset 0   "BTSD"                       4 bytes, magic
offset 4   version                      u32 LE  (currently 1)
offset 8   headerLen                    u32 LE
offset 12  header                       JSON, headerLen bytes
offset 12+headerLen  payload            every entry's raw bytes, in order
```

The header carries, per entry, `name`, `dtype`, `shape` and `nbytes`, plus the
payload's total length and its **SHA-256**. `nd_tobytes`/`nd_frombytes` do the
element encoding, little-endian, which makes a file portable between the
platforms Bantu builds for.

**`i64` had to be added to `nd_tobytes` for this** (§4, A-extra): numba *has* an
i64 dtype and offered nothing wider than `i32` to write it with, so an i64 array
could only be serialized lossily. The same fix made the integer encoder read
`int64` directly instead of through a `double`, which was silently lossless only
below 2^53 — invisible while no 8-byte integer element existed.

**Buffers are saved, not just parameters.** A `BatchNorm` layer's running mean
and variance are buffers precisely so no optimizer steps them; a `state_dict`
that omitted them would reload a model whose `eval()` is wrong, and would train
and score fine right up until it did not. So `named_buffers()` exists and
`state_dict` includes it, with buffers reported separately in the header.

**Every load failure is a different failure, and says so.** `json.parse` in this
interpreter returns garbage (`{: null}`) or `null` for malformed input rather
than raising, so the loader cannot lean on it and validates every field itself:

| condition | message |
|---|---|
| file missing/unreadable | names the path |
| shorter than 12 bytes | "not a btorch state file" |
| bad magic | "not a btorch state file (magic …)" |
| version above the reader's | "written by a newer btorch (v…)" — never silently best-effort |
| header truncated | names the bytes expected vs present |
| header not an object, or `entries` not a list | "corrupt header" |
| an entry missing a field, or of the wrong type | names the entry and the field |
| payload length ≠ `payload_bytes` | **truncated**, and by how many bytes |
| SHA-256 mismatch | **corrupt**, with both digests |

The last two are deliberately separate. A truncated file and a flipped byte are
different accidents with different causes, and collapsing them into "bad file"
throws away the only diagnostic the reader has.

---

## 8. The test tiers

Every phase is gated by six tiers, all green before it advances. The first five
are the house standard from `numba` and `pillow`; tier 3 is btorch's own.

| tier | proves | form |
|---|---|---|
| 1. feature | every function does what it claims, including its failure modes | `tests/btorch_*_test.b`, ending `RESULT: ALL GREEN` |
| 2. differential | the answers are *right* — PyTorch reference values embedded in the test | hand-computed small cases; closed forms for optimizers |
| 3. **gradcheck** | **every gradient agrees with central differences to 1e-7** | `bt.gradcheck`, one case per op in §5 |
| 4. stress | it survives size, repetition and abuse | deep graphs, long training runs, RSS over 10k steps, adversarial shapes |
| 5. regression | nothing else broke | the whole existing suite on the same build |
| 6. sanitizers | no latent memory defect | ASan + UBSan over the full btorch suite |

Two rules inherited from numba, both learned the hard way, both of which apply
with full force here:

- **Never assert on a stringified number.** `str()` gives six significant
  digits and leaks scientific notation. Use `nd_allclose` or the `close()`
  helper with an explicit tolerance.
- **Never use the C-style `for` above 100,000 iterations.** `evaluator.hpp`
  caps it at `safety < 100000` and exits **silently** — a training loop written
  that way would stop early and report a plausible, wrong accuracy. Use
  `while`.

A third, specific to this library:

- **A loss that goes down proves nothing.** Gradient descent on a wrong
  gradient often still reduces a loss. Correctness comes from tier 3; the
  training runs in tier 4 and phase 10 are there to catch what unit tests
  structurally cannot, not to substitute for them.

---

## 9. Performance: targets and the honest ceiling

btorch inherits numba's measured ceiling and its per-platform discipline (N22:
a flat GB/s target is unreachable by construction on a high-latency vCPU, so
targets are stated per platform or as a ratio).

The ceiling is **one core, f64, no fused kernels**. For the MLP above
(784→128→10, batch 128) the forward matmuls are ~26 MFLOP; at the measured
7.56 GFLOP/s that is ~3.5 ms, and with backward roughly 3× that — **~10 ms per
step**, so ~470 steps/epoch on MNIST is ~5 s/epoch. That is slow next to
PyTorch-on-GPU and entirely usable for learning, research on small models, and
embedded inference. The design states this rather than implying otherwise.

| target | gate |
|---|---|
| A1 (splat) | scalar ops within **1.3×** of the array-array path, i.e. ≥ 10 GB/s on this machine, up from 3.2 |
| autograd overhead | tape bookkeeping **< 15%** of wall time for a 128×256 MLP step |
| `backward()` | within **3×** the cost of `forward()` for an MLP |
| memory | 10,000 training steps with **no RSS growth** after the first (proves the graph is freed) |
| conv2d | `im2col` + `matmul` within **2×** of the matmul's own FLOP rate |
| phase 10 | MNIST MLP ≥ 97%, CNN ≥ 98.5% |

### 9.1 f32 — deferred deliberately, and why

PyTorch's default dtype is f32; btorch's is f64. f32 would halve memory traffic
and roughly double every bandwidth-bound op. It is **not** in phase A because
`DType` is `{F64, I64, BOOL}` (`ndarray_native.hpp:61`) and adding a variant
touches the dispatch of all 171 kernels — a numba-wide change with its own
accuracy gates, not a btorch phase. It is phase 9, on measurement, and
`DECISIONS.md` **B7** records the reasoning. f64 is meanwhile *more* accurate
than PyTorch's default, which makes gradchecking at 1e-7 comfortable rather
than marginal — a real benefit while correctness is being established.

---

## 10. Deferred, each needing its own justification

Nothing here is scheduled; each lands only on a concrete measured need.

- **GPU / CUDA / Metal.** Out of scope. The `device` argument is accepted and
  must be `"cpu"`, so the API does not have to change later.
- **JIT / `torch.compile` / graph fusion.** The tape is an interpreter; fusing
  `a*b+c` into one kernel pass is the single biggest remaining CPU win and is
  phase 9 territory at best.
- **Distributed / multi-process training.** Note that `sua` runs handlers on
  detached threads and numba's limits were made thread-local for exactly that
  reason (N18); any parallel training must respect it.
- **Complex dtypes, FFT, sparse tensors.**
- **`vmap`, forward-mode AD, double backward.** The tape stores closures over
  arrays, not over tensors, so second derivatives would need those closures to
  be differentiable too — a deliberate design limit, recorded as **B4**.
- **ONNX export.**

---

## 11. Sequence layers — phase 8 (ecosystem stage 5)

Paired with `btext` ([`btext-architecture.md`](btext-architecture.md)) because
neither is useful alone. Tracked as phase 8 in
[`../btorch-suite/ROADMAP.md`](../btorch-suite/ROADMAP.md).

**Every primitive this needs already exists.** Verified, not assumed:
`cat`, `chunk`, `permute`, `transpose`, `matmul`, `reshape`, `split` in
`btorch.b`; `nn.log_softmax` / `nn.softmax`, numerically stable, in `nn.b`;
`nd_where` for masking; `clip_grad_norm` in `optim.b`. **No new native kernel,
and no new gradient.** That is the B18 principle applied again — compose, do
not add a special case.

### 11.1 The fused gate matmul

An LSTM cell computes four gates from the same two inputs. The naive form is
eight matmuls per timestep; the right form is two:

```
gates = x @ W_ih^T + h @ W_hh^T + b          // [N, 4H] in one matmul each
i, f, g, o = chunk(gates, 4, 1)              // views, so the split is free
```

`chunk` returns views over the existing storage (A6: `nd_split` returns views),
so the split costs nothing and the backward is a `cat` of four gradients. Four
times fewer tape nodes and four times fewer kernel launches per timestep.

### 11.2 The tape cost, computed before building

A Bantu call costs **1.78 µs** (B1), and a recurrence is a Bantu loop over
timesteps — so this is the one place in btorch where per-*op* overhead could
dominate. The arithmetic, for one LSTM layer:

| quantity | value |
|---|---|
| ops per timestep (fused) | ~14 |
| calls at `T = 128` | ~1,800 |
| interpreter overhead | ~3.2 ms per layer per forward |
| tape nodes at `T = 128` | ~1,800 |

Acceptable, and well inside the 1,000-call depth guard's concern (B14) because
the loop is iterative, not recursive. But it sets the honest limit: **`T` in
the hundreds, not the tens of thousands.** A 4,096-token context would spend
~100 ms per layer in interpreter overhead alone, before any arithmetic. Recorded
as a limit of the design, not something to be discovered later.

### 11.3 Causal masks compose; no `nd_tril` needed

`nd_tril`/`nd_triu` do not exist, and should not be added. A causal mask is a
comparison of two broadcast index grids:

```bantu
$rows = nd_reshape(nd_arange(0.0, $T * 1.0, null), [$T, 1]);
$cols = nd_reshape(nd_arange(0.0, $T * 1.0, null), [1, $T]);
$mask = nd_greater_equal($rows, $cols, null);        // [T, T], lower triangular
```

Three existing calls instead of a new kernel, and it generalises to any
banded or windowed mask for free. `masked_fill` is `nd_where` with a scalar.

### 11.4 Scope

| item | phase |
|---|---|
| `RNNCell`, `GRUCell`, `LSTMCell` | 8.1 |
| `RNN`, `GRU`, `LSTM` sequence wrappers — `batch_first`, `num_layers`, `bidirectional` | 8.2 |
| `MultiheadAttention` — heads via `reshape`+`permute`, scaled dot product, stable softmax | 8.3 |
| `TransformerEncoderLayer`, `TransformerEncoder` | 8.4 |
| positional encodings: sinusoidal, learned, rotary | 8.5 |
| KV cache for autoregressive inference (`no_grad` only) | 8.6 |
| packed sequences / padding masks | 8.7 |

### 11.5 The gates

Tier 3 (gradcheck) is where this phase lives or dies, with one wrinkle:

**A gradcheck over a `T`-step recurrence is numerically fragile.** Central
differences on a deep composition accumulate error, and an exploding recurrence
makes the finite-difference estimate meaningless. So gradchecks run at
**`T ≤ 4` with small, well-conditioned weights**, which is enough to exercise
every path through the cell — the recurrence is the same closure at every step,
so a correct gradient at `T = 4` is correct at `T = 128`.

Separately, a **stress tier** runs `T = 512` and asserts the gradient norm is
finite and that `clip_grad_norm` bounds it, which is the property that actually
matters at depth.

And the phase gate: **an LSTM trained on a character-level language-modelling
task reaches a stated perplexity**, measured from a reloaded checkpoint. B10
applies — a falling loss proves nothing, so the gradcheck is the real test and
the training run is the integration test.

---

## 12. int8 quantization — `quant.b` (ecosystem stage 6)

Lives inside btorch ([E5]) because it must know every layer type in order to
swap it; a separate package would break every time `nn.b` adds a layer.

**Why it matters here more than elsewhere.** This stack has no GPU, so int8 is
*the* inference performance story: 4× the arithmetic throughput and 4× less
memory, pairing exactly with the single-static-binary deployment wedge
(`ecosystem-architecture.md` §7.6). A 61,706-parameter LeNet is not the target;
an imported ResNet (`bonnx`, stage 4) running at the edge is.

### 12.1 The `q8_` family, and a real `I8` dtype

Unlike `bkit`'s binned matrix and `btext`'s merge tables — both opaque handles
— **quantized weights must be a real `NdArray`**. Users inspect them, save them
to `.btsd`, and load them back. So stage 6 adds `DType::I8` properly.

This is the third dtype the templating pass ([E16]) pays for, and the reason
that pass is phase D1 rather than something done inline with f32.

| kernel | purpose |
|---|---|
| `q8_quantize` | `f32/f64 → i8` given scale and zero-point, saturating |
| `q8_dequantize` | `i8 → f32/f64` |
| `q8_gemm` | int8 × int8 → **i32 accumulate** → requantize to i8 |
| `q8_gemm_f` | int8 × int8 → i32 → dequantize to f32 (for the last layer) |

The i32 accumulator is not optional: `127 × 127 × K` overflows `int16` at
`K = 5`, and silently wrapping is a wrong answer that looks like noise.

### 12.2 Scheme: per-channel symmetric weights, per-tensor asymmetric activations

| tensor | scheme | why |
|---|---|---|
| weights | **per-output-channel, symmetric** (`zero_point = 0`) | per-channel recovers most of the accuracy lost to quantization, because channel ranges differ by orders of magnitude. Symmetric removes the zero-point cross-term from the GEMM inner loop entirely |
| activations | **per-tensor, asymmetric** | activations after ReLU are one-sided, so asymmetric uses the full 255 levels instead of 127. Per-tensor because per-channel activation scales would have to be tracked at runtime |

The asymmetry is deliberate and standard: it puts the cheap scheme where the
hot loop is (weights, symmetric → two-term GEMM) and the accurate scheme where
the dynamic range is unpredictable (activations).

### 12.3 Phases

| item | phase |
|---|---|
| `q8_` kernels + `DType::I8` | Q1 |
| observers — `MinMax`, `MovingAverageMinMax`, `Histogram` (percentile clipping) | Q2 |
| **dynamic** quantization — weights offline, activations per-batch; no calibration data needed | Q3 |
| **static** PTQ — a calibration pass over representative data, then layer swap | Q4 |
| `QuantizedLinear`, `QuantizedConv2d`, and `Module.quantize($scheme)` | Q5 |
| accuracy and throughput gates | Q6 |

Dynamic before static, deliberately: it needs no calibration dataset, so it is
usable the day it lands and it validates the kernels before the calibration
machinery is written.

### 12.4 The gates

| gate | target |
|---|---|
| **accuracy** | MNIST CNN int8 within **0.5%** of the f32 result (f32 baseline established at D2; f64 reference is 98.98%) |
| **throughput** | `q8_gemm` measurably faster than `gemmContig` on the same shapes — recorded whatever it is |
| **round-trip** | `dequantize(quantize(x))` within the quantization step of `x`, for every scale, including saturation at both ends |
| **no silent overflow** | a fuzz test driving accumulators toward `int32` limits, asserting saturation rather than wrap |
| **container** | a quantized model saves and loads from `.btsd` and predicts bit-identically ([C3]) |

### 12.5 The honest uncertainty

**The 4× throughput figure assumes SIMD integer dot-product instructions**
(AVX-512 VNNI on x86, SDOT on ARM). Without them, int8 GEMM on a generic
scalar path may be no faster than f32 — the win would be only in memory
bandwidth and cache residency, perhaps 1.5–2×.

Whether the compiler auto-vectorises the i32 accumulation well enough is the
**first measurement of Q1**, before the rest of the phase is scoped. If it does
not, the options are intrinsics behind a feature check, or accepting the
memory-bandwidth win alone and saying so. What is not acceptable is publishing
a 4× claim that the measurement does not support — phase 9's rule: measure,
because two of three suspects were wrong.

---

## 13. Deferred items that now have homes

Updating §10, where these were listed as deferred with no plan:

| item | now |
|---|---|
| f32 | ecosystem stage 1, phase D2 ([`numba-dtype-architecture.md`](numba-dtype-architecture.md)) |
| FFT, sparse | ecosystem stage 6 — FFT in `numba` (D5), sparse in `bsci` ([E4], [`bsci-architecture.md`](bsci-architecture.md)) |
| ONNX export | `bonnx` phase O8, unscheduled; import is the asymmetric win ([`bonnx-architecture.md`](bonnx-architecture.md) §6) |
| graph fusion / `torch.compile` | still deferred; stage 1's threading and packed GEMM are the cheaper wins first |
| distributed training | still deferred; stage 1's threading is the honest ceiling |
| `vmap`, forward-mode AD, double backward | still deferred, by design (B4) |
| `device` strings | still `"cpu"`; the wedge is explicitly CPU + static binary |
