# btorch

A PyTorch-shaped deep-learning library for Bantu. Tensors with reverse-mode
automatic differentiation, `nn.Module`, optimizers, data loading, and
checkpointing — the public API is pure Bantu; the inner loops are numba's
native `nd_*` kernels.

```bantu
include "btorch/btorch.b" as bt;

$x = bt.tensor([2.0, 3.0], true);
$y = bt.sum(bt.mul($x, $x), null, null);
$y.backward(null);
print(nd_to_list($x.grad));        // [4, 6]
```

- [Install](#install)
- [Tensors and autograd](#tensors-and-autograd)
- [Operations](#operations)
- [nn — layers and losses](#nn--layers-and-losses)
- [optim — optimizers and schedules](#optim--optimizers-and-schedules)
- [data — datasets and batching](#data--datasets-and-batching)
- [serialize — saving and loading](#serialize--saving-and-loading)
- [datasets — MNIST](#datasets--mnist)
- [Verifying a gradient](#verifying-a-gradient)
- [Things that will bite you](#things-that-will-bite-you)

---

## Install

btorch is five Bantu files plus a manifest. Copy `btorch/` next to your code,
or install it from the local registry:

```sh
bantu add btorch
```

It needs a `bantu` binary with numba's `nd_*` builtins, including
`nd_index_add`, `nd_concat` and `nd_im2col`. If you see

```
Undefined variable: nd_im2col
```

your interpreter is older than the library. Run `bantu doctor` — it compares
the binary you ran with the one on your `PATH` and names the stale one.

| file | what it holds |
|---|---|
| `btorch/btorch.b` | `Tensor`, the tape, `backward()`, every operation, `gradcheck` |
| `btorch/nn.b` | `Module`, layers, initializers, losses |
| `btorch/optim.b` | SGD, Adam, AdamW, RMSprop, Adagrad, LR schedules |
| `btorch/data.b` | `Dataset`, `TensorDataset`, `Subset`, `DataLoader`, `random_split` |
| `btorch/serialize.b` | `save`, `load`, `save_model`, `load_model` |
| `btorch/datasets.b` | the IDX reader and MNIST |

Everything computes in **f64**. There is no f32 yet.

---

## Tensors and autograd

`bt.tensor(data, requires_grad)` takes a number, a nested list, or a numba
array.

```bantu
$a = bt.tensor([[1.0, 2.0], [3.0, 4.0]], true);
print($a.shape());        // [2, 2]
print($a.dtype());        // f64
print($a.size());         // 4
print(bt.tensor([7.0], false).item());   // 7 -- item() needs ONE element
```

Asking a multi-element tensor for its `item()` raises, naming the shape:

```bantu
// EXPECT-RAISES
print(bt.tensor([[1.0, 2.0]], false).item());
```

Operations on a tensor that requires a gradient build a tape. Calling
`backward()` walks it and leaves each leaf's gradient in `.grad`, as a numba
array:

```bantu
$w = bt.tensor([2.0], true);
$loss = bt.mul($w, $w);                   // w^2
$loss.backward(null);
print(nd_to_list($w.grad));               // [4]   (dw^2/dw = 2w)
```

`backward()` **accumulates**, as PyTorch's does. Two calls without a
`zero_grad()` in between sum — which is how gradient accumulation over
micro-batches is meant to work, and a surprise if you did not want it.

```bantu
$w.zero_grad();                           // or $opt.zero_grad()
```

Operators work on tensors directly:

```bantu
$c = ($a + $b) * 2.0 - $a / $b;
$d = 0.0 - $c;                            // unary minus: write 0.0 - x
```

### Turning the tape off

```bantu
$out = bt.no_grad(def() { return $model.forward($x); });
$out = bt.inference_mode(def() { return $model.forward($x); });   // the same thing
```

`no_grad` is about the **tape**; `Module.eval()` is about **layer behaviour**
(dropout off, batch-norm frozen). They are orthogonal, and an evaluation
usually wants both:

```bantu
$model.eval();
$acc = bt.inference_mode(def() { return measure($model, $testx, $testy); });
$model.train();
```

### Non-leaf gradients

Only leaves get a `.grad` by default. Ask for an intermediate one explicitly:

```bantu
$v = bt.tensor([1.0, 2.0], true);
$h = bt.mul($v, 3.0);
$h.retain_grad();
bt.sum($h, null, null).backward(null);
print(nd_to_list($h.grad));        // [1, 1] -- the gradient AT h, not at v
```

`detach()` gives the same numbers with no history.

---

## Operations

Every one of these is differentiable and every gradient is checked against
central differences in `tests/btorch_autograd_test.b`,
`tests/btorch_indexing_test.b` and `tests/btorch_conv_test.b`.

**Arithmetic** — `add sub mul div pow neg`, plus the operator forms. Broadcasting
follows NumPy, and a broadcast gradient is summed back over the stretched axes.

**Unary** — `exp log sqrt abs_ tanh sigmoid relu leaky_relu`.

**Reductions** — `sum mean max min`, each taking `(axis, keepdims)`. `axis` of
`null` reduces everything.

```bantu
bt.sum($x, null, null)       // a scalar
bt.sum($x, 1, true)          // along axis 1, axis kept as length 1
bt.max($x, 1, null)          // the maximum along axis 1
```

> `max`/`min` **split** the gradient between tied elements rather than giving it
> all to one. `max([4,4])` therefore has gradient `[0.5, 0.5]`. PyTorch's
> `max(dim)` routes to the argmax instead; `amax` splits, as this does.

**Elementwise extremes** — `maximum minimum` take two operands (a tensor or a
number) and give a tie to the left one.

**Shape** — `reshape view flatten transpose t permute squeeze unsqueeze movedim`.

**Indexing** — `take index_select embedding one_hot gather index_add`.

```bantu
bt.take($w, [0, 2, 0], 0)         // rows 0, 2 and 0 again
bt.index_select($w, 1, [2, 0])    // PyTorch's argument order
bt.embedding($table, $tokenIds)   // a row lookup
bt.gather($logits, 1, $idx)       // per-ELEMENT indexing
```

> `bt.take` **requires** an axis. numba's `nd_take` defaults to the last axis
> and every neural-net caller means axis 0, so there is no safe default.

A repeated index **accumulates** in the backward pass. A token appearing twice
in a batch adds both of its gradients to that embedding row — which is the
whole reason this is not built on `nd_put`.

**Joining** — `cat stack chunk split pad tile repeat_interleave`.

```bantu
bt.cat([$a, $b], 0)
bt.stack([$a, $b], 0)             // a NEW axis
bt.chunk($x, 3, 0)                // ceil(n/3) per piece, so possibly < 3 pieces
bt.pad($x, [[1, 1], [2, 0]], 0.0) // [before, after] per axis
```

**Linear algebra** — `matmul` (`mm`), which broadcasts over leading batch axes.

**Convolution** — `conv1d conv2d max_pool2d avg_pool2d`, and `im2col` underneath.

```bantu
$img = bt.tensor(nd_random_normal([8, 3, 32, 32]), false);   // [N, Cin, H, W]
$k = bt.tensor(nd_random_normal([16, 3, 3, 3]), true);       // [Cout, Cin, kh, kw]
$cb = bt.tensor(nd_zeros([16], "f64"), true);                // [Cout], or null

$out = bt.conv2d($img, $k, $cb, 1, 1, null);     // stride, padding, dilation
print($out.shape());                              // [8, 16, 32, 32]

$pooled = bt.max_pool2d($out, 2, null, null, null, null, null);
print($pooled.shape());                           // [8, 16, 16, 16]
```

`stride`, `padding` and `dilation` each take a number or a `[height, width]`
pair. Pooling's stride defaults to the **kernel**; convolution's defaults to 1.

---

## nn — layers and losses

```bantu
include "btorch/nn.b" as nn;

$net = nn.Sequential([
    nn.Linear(784, 128, null), nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10, null)
]);
$logits = $net.forward(bt.tensor(nd_random_normal([8, 784]), false));
```

**Layers** — `Linear Conv1d Conv2d MaxPool2d AvgPool2d Dropout LayerNorm
BatchNorm1d BatchNorm2d Embedding Flatten Sequential`, and the activations
`ReLU Tanh Sigmoid LeakyReLU GELU Softmax`.

**Losses** — `mse_loss l1_loss huber_loss cross_entropy nll_loss
bce_with_logits bce_loss`, plus `MSELoss L1Loss CrossEntropyLoss` as objects.
Each takes `reduction` in `{"mean", "sum", "none"}`; `null` means `"mean"`.

```bantu
$loss = nn.cross_entropy($logits, $labels, null);   // labels are class INDICES
```

`cross_entropy` fuses `log_softmax` with the negative log-likelihood, so
logits of 1000 give a large finite loss rather than `NaN`.

### Writing your own layer

Subclass `Module`, call `super()`, and `register` each parameter:

```bantu
class _Scale extends nn.Module {
    def init($n) {
        super();
        this.register("gain", bt.ones([$n], true));
    }
    def forward($x) { return bt.mul($x, this.param("gain")); }
}
def Scale($n) { return new _Scale($n); }
```

> The factory must **not** share its name with the class. `def Scale()`
> containing `new Scale(...)` re-enters the factory and recurses until the
> interpreter's depth guard fires — Bantu keeps functions and variables in one
> namespace. Hence the `_Scale` convention throughout `nn.b`.

Children registered with `add(name, child)` are found automatically:

```bantu
class _Block extends nn.Module {
    def init($n) {
        super();
        this.add("fc", nn.Linear($n, $n, null));
        this.add("norm", nn.LayerNorm($n, null));
    }
    def forward($x) { return this.get("norm").forward(this.get("fc").forward($x)); }
}
```

`parameters()` and `named_parameters()` walk the whole tree, **deduplicated by
identity** — a weight shared between two layers is returned once, so an
optimizer cannot step it twice.

`buffers()` and `named_buffers()` do the same for non-parameter state such as
batch-norm running statistics. Both go into `state_dict()`.

---

## optim — optimizers and schedules

```bantu
include "btorch/optim.b" as optim;

$opt = optim.Adam($net.parameters(), 0.001, null, null, null);
$opt.zero_grad();
$loss.backward(null);
$opt.step();
```

`SGD(params, lr, momentum, weight_decay)` · `Adam(params, lr, beta1, beta2, eps)`
· `AdamW` · `RMSprop` · `Adagrad`.

Schedules wrap an optimizer and are stepped once per **epoch**:

```bantu
$sched = optim.StepLR($opt, 10, 0.5);          // halve every 10 epochs
$sched = optim.ExponentialLR($opt, 0.95);
$sched = optim.CosineAnnealingLR($opt, 50, 0.0);
$sched.step();
```

Gradient clipping, over all parameters **jointly**:

```bantu
nn.clip_grad_norm($net.parameters(), 1.0);
```

---

## data — datasets and batching

```bantu
include "btorch/data.b" as btdata;

// NOT named $ds: a variable shares its namespace with an `include ... as ds`
// alias, so `$ds = ...` would make `ds.mnist(...)` stop working.
$dset = btdata.TensorDataset([$x, $y]);         // tensors sharing a first axis
$loader = btdata.DataLoader($dset, 16, true, false, 42);
//                           dataset, batch, shuffle, drop_last, seed
```

Bantu has no iterator protocol, so a pass is an index loop:

```bantu
$b = 0;
while ($b < $loader.num_batches()) {
    $batch = $loader.batch($b);                 // a list of tensors
    $opt.zero_grad();
    nn.cross_entropy($net.forward($batch[0]), $batch[1].data, null).backward(null);
    $opt.step();
    $b = $b + 1;
}
$loader.reshuffle();                            // start the next epoch
```

`reshuffle()` draws a **new** permutation — it does not restart the stream, so
epoch 2 differs from epoch 1 while the whole run stays reproducible from its
seed.

`random_split` takes **exact counts**, never fractions:

```bantu
// The counts must sum to the dataset exactly -- here 64 rows.
$parts = btdata.random_split($dset, [50, 14], 42);
$trainLoader = btdata.DataLoader($parts[0], 16, true, false, null);
print($parts[0].size());                          // 50
print($parts[1].size());                          // 14
```

A custom dataset needs `size()` and `get(i)`:

```bantu
class _Mine extends btdata.Dataset {
    def init($n) { super(); this.n = $n; }
    def size() { return this.n; }
    def get($i) { return [bt.tensor([$i * 1.0], false), bt.tensor([0.0], false)]; }
}
```

`get(i)` returns a **row** per tensor; `DataLoader` stacks rows into batches.

---

## serialize — saving and loading

```bantu
include "btorch/serialize.b" as ser;

ser.save($net, "model.btsd");
ser.load_model($net, "model.btsd");
```

`save` takes a `Module` (parameters **and** buffers) or a dict of tensors.
`load_model` copies into the existing tensors **in place**, so an optimizer
already holding them keeps working.

`load` returns the raw contents without touching a model:

```bantu
ser.save($net, "model.btsd");
$f = ser.load("model.btsd");
print($f.order);                          // names, in file order
print(nd_shape($f.entries["0.weight"]));
print($f.kinds["0.weight"]);              // "param" (a buffer says "buffer")
```

The format is a small JSON header over a raw little-endian payload with a
SHA-256. Round trips are **bit-identical**. A truncated file and a corrupted
one raise different errors, and a file written by a newer btorch is refused
rather than read best-effort.

---

## datasets — MNIST

```sh
bash samples/btorch/get_mnist.sh         # 54 MB into ~/.bantu/datasets/mnist
```

```bantu
include "btorch/datasets.b" as ds;

$m = ds.mnist(null, "standard");
// train_x [60000, 784]   train_y [60000]
// test_x  [10000, 784]   test_y  [10000]
```

`normalize` is `"standard"` (MNIST's own mean/std — what makes an MLP converge
in a handful of epochs), `"unit"` (x/255), or `"raw"`.

```bantu
$flat = bt.tensor(nd_random_normal([8, 784]), false);
$images = ds.as_images($flat);            // [N, 784] -> [N, 1, 28, 28]
print($images.shape());                   // [8, 1, 28, 28]

$acc = ds.accuracy($logits, $labels);     // argmax over the class axis
```

`read_idx(path)` reads any IDX file, big-endian header and all.

---

## Verifying a gradient

If you write a custom backward, check it numerically. `gradcheck` compares the
analytic gradient with central differences:

```bantu
$r = bt.gradcheck(def($in) { return bt.sum(bt.mul($in[0], $in[0]), null, null); },
                  [bt.tensor([1.0, 2.0, 3.0], true)], null, null);
print($r.ok);                 // true
print($r.worst);              // the worst relative disagreement
```

The function must return a **one-element** tensor, and it must be a **pure
function of its inputs**. Drawing a random coefficient inside it means
`f(x+h)` and `f(x-h)` use different coefficients, and the result is nonsense
that looks exactly like a broken gradient.

```bantu
$coef = bt.tensor(nd_random_normal([3]), false);     // ONCE, outside
$r = bt.gradcheck(def($in) { return bt.sum(bt.mul($in[0], $coef), null, null); },
                  [bt.tensor(nd_random_normal([3]), true)], null, null);
```

---

## Things that will bite you

**A loss going down proves nothing.** Gradient descent on a *wrong* gradient
still often reduces a loss. If you add an operation, gradcheck it.

**A full reduction returns a 0-d array, not a number.** `nd_sum($a, null, null) == 0`
compares a handle with a number and is always false. Use `.item()` on a tensor,
or `nd_get(nd_reshape($r, [1]), [0])` on an array.

**Closures capture by reference.** Three closures built over a loop variable all
see its final value. Anything needing per-iteration capture must be built by a
factory whose parameters bind once per call — which is why every backward in
`btorch.b` comes from a `_*Backward(...)` function.

**`pop(list)` returns the element; `push(list, v)` returns the list.** So
`$stack = pop($stack)` assigns an element to `$stack`. Use `pop($x);` bare.

**Recursion is capped at 1000 calls.** Anything walking a user-sized structure
must be iterative. `backward`, `named_parameters` and `state_dict` all are.

**The C-style `for` silently stops above 100,000 iterations.** Use `while`.

**Unary minus on a literal:** write `0.0 - $x`, not `-$x`, in expressions where
the parser would read the minus as subtraction.

**A variable can shadow a module alias.** `include "btorch/datasets.b" as ds;`
followed by `$ds = something;` makes `ds.mnist(...)` fail with "it holds null,
not a function" — Bantu keeps variables, functions and aliases in one
namespace. Name the variable something else.

**Everything is f64.** A 60,000 × 784 dataset is 376 MB. There is no f32 yet.

---

## Measured

On an Apple-silicon laptop, single-threaded:

| | |
|---|---|
| MNIST MLP (784-256-128-10), 8 epochs | **98.16%** test accuracy, ~20 s/epoch |
| MNIST CNN (LeNet-5, 61,706 params), 4 epochs | **98.98%** test accuracy, ~107 s/epoch |
| 60,000-index shuffle | 8 ms |
| one epoch of 469 batches (data loading only) | 84 ms |
| LeNet-5 checkpoint | 494 KB, saved in 7 ms, loaded in 8 ms |
| `im2col` on 64×56×56 (a ResNet block) | 18 ms |
| embedding gradient, 20k × 128 | 7 ms |
| one LeNet training step, batch 128 | 240 ms |
| `MaxPool2d` forward / forward+backward | 4.6 ms / 29 ms |

`btorch-suite/` holds the roadmap, the decision log and the measurement
history. `docs/btorch-architecture.md` is the design.
