# btorch, A to Z

A tutorial. It starts with a single number and ends with a convolutional
network that reads handwriting at 99% accuracy. Every code block here is
executed by CI, so if you can read it, it runs.

The companion [`btorch.md`](btorch.md) is the reference — look things up there.
This is the one to read front to back.

**Contents**

1. [Hello, gradient](#1-hello-gradient)
2. [Tensors](#2-tensors)
3. [How autograd actually works](#3-how-autograd-actually-works)
4. [Your first model, by hand](#4-your-first-model-by-hand)
5. [The same model with `nn`](#5-the-same-model-with-nn)
6. [Losses](#6-losses)
7. [Optimizers and learning rates](#7-optimizers-and-learning-rates)
8. [A real training loop](#8-a-real-training-loop)
9. [Loading data](#9-loading-data)
10. [Train, validate, test](#10-train-validate-test)
11. [Saving and resuming](#11-saving-and-resuming)
12. [Classifying handwriting: the MLP](#12-classifying-handwriting-the-mlp)
13. [Convolutional networks](#13-convolutional-networks)
14. [Embeddings and sequences](#14-embeddings-and-sequences)
15. [When it goes wrong](#15-when-it-goes-wrong)
16. [Writing your own layer](#16-writing-your-own-layer)
17. [What to read next](#17-what-to-read-next)

---

## 1. Hello, gradient

Install btorch by copying `btorch/` next to your code. Then:

```bantu
include "btorch/btorch.b" as bt;

$x = bt.tensor([3.0], true);        // true = "I want the gradient of this"
$y = bt.mul($x, $x);                // y = x²
$y.backward(null);                  // work out dy/dx

print(nd_to_list($x.grad));         // [6]
```

`x²` differentiated is `2x`, and at `x = 3` that is `6`. btorch did not know
the formula — it recorded what you did and walked it backwards.

That is the whole library in one idea. Everything else is scale.

If instead you see

```
Undefined variable: nd_index_add
```

your `bantu` binary is older than btorch. Run `bantu doctor`; it compares the
binary you ran with the one on your `PATH` and tells you which is stale.

---

## 2. Tensors

A tensor is an n-dimensional array of f64 numbers, optionally tracking
gradients.

```bantu
include "btorch/btorch.b" as bt;

$scalar = bt.tensor([42.0], false);
$vector = bt.tensor([1.0, 2.0, 3.0], false);
$matrix = bt.tensor([[1.0, 2.0], [3.0, 4.0]], false);

print($matrix.shape());             // [2, 2]
print($vector.size());              // 3
print($scalar.item());              // 42
print($matrix.to_list());           // [[1, 2], [3, 4]]
```

`item()` is for one-element tensors only — it raises otherwise, rather than
quietly giving you the first element.

Ready-made constructors:

```bantu
print(bt.zeros([2, 3], false).shape());        // [2, 3]
print(bt.ones([2], false).to_list());          // [1, 1]
print(bt.eye(3, false).shape());               // [3, 3]
print(bt.arange(0.0, 5.0, false).to_list());   // [0, 1, 2, 3, 4]
print(bt.linspace(0.0, 1.0, 5, false).to_list());
bt.manual_seed(42);
print(bt.randn([2, 2], false).shape());        // normal(0, 1)
print(bt.rand([2, 2], false).shape());         // uniform(0, 1)
```

Arithmetic works with operators or with named functions, and broadcasts the
way NumPy does:

```bantu
$a = bt.tensor([[1.0, 2.0], [3.0, 4.0]], false);
$b = bt.tensor([10.0, 20.0], false);           // one row, stretched over two

print(($a + $b).to_list());                    // [[11, 22], [13, 24]]
print(($a * 2.0).to_list());
print(bt.matmul($a, $a).to_list());            // or $a.mm($a)
print($a.t().to_list());                       // transpose
print($a.reshape([4]).to_list());
print(bt.sum($a, null, null).item());          // 10
print(bt.sum($a, 1, null).to_list());          // [3, 7] -- along axis 1
print(bt.mean($a, 0, null).to_list());         // [2, 3] -- along axis 0
```

Write `0.0 - $x` for negation. A bare `-$x` reads as subtraction in most
positions.

---

## 3. How autograd actually works

btorch records a **tape**. Every operation on a tensor that requires a gradient
appends a node remembering how to undo itself. `backward()` walks that tape in
reverse, multiplying as it goes — the chain rule, mechanised.

Two consequences you will meet.

**The tape is built as you run, not ahead of time.** An `if` is just an `if`;
whichever branch ran is what gets differentiated.

```bantu
$x = bt.tensor([2.0], true);
$y = $x;
$i = 0;
while ($i < 3) { $y = bt.mul($y, $x); $i = $i + 1; }   // y = x⁴
$y.backward(null);
print(nd_to_list($x.grad));          // [32]  = 4·2³
```

**Gradients add up until you clear them.** This is deliberate — it is how you
accumulate over several small batches — and surprising the first time.

```bantu
$w = bt.tensor([1.0], true);
bt.mul($w, 5.0).backward(null);
print(nd_to_list($w.grad));          // [5]
bt.mul($w, 5.0).backward(null);
print(nd_to_list($w.grad));          // [10]  -- summed, not replaced
$w.zero_grad();
print($w.grad);                      // null
```

Only the tensors you created (the "leaves") keep a `.grad`. Ask for an
intermediate one explicitly:

```bantu
$x = bt.tensor([1.0, 2.0], true);
$h = bt.mul($x, 3.0);
$h.retain_grad();
bt.sum($h, null, null).backward(null);
print(nd_to_list($h.grad));          // [1, 1]
print(nd_to_list($x.grad));          // [3, 3]
```

When you are only predicting, turn the tape off. It saves memory and time:

```bantu
$model = bt.tensor([2.0], true);
$out = bt.no_grad(def() { return bt.mul($model, $model); });
print($out.requires_grad);           // false
```

`bt.inference_mode` is the same thing with a name that reads better at a call
site.

---

## 4. Your first model, by hand

Before using `nn`, do it the hard way once. The task: recover `y = 3x + 2`
from noisy samples.

```bantu
include "btorch/btorch.b" as bt;

bt.manual_seed(1);
$xs = nd_random_uniform([100, 1], 0.0 - 1.0, 1.0);
$ys = nd_add(nd_multiply($xs, 3.0, null), 2.0, null);

$w = bt.tensor([[0.0]], true);        // start from nothing
$b = bt.tensor([0.0], true);
$tx = bt.tensor($xs, false);
$ty = bt.tensor($ys, false);

$step = 0;
while ($step < 300) {
    $pred = bt.add(bt.matmul($tx, $w), $b);
    $diff = bt.sub($pred, $ty);
    $loss = bt.mean(bt.mul($diff, $diff), null, null);    // mean squared error

    $w.zero_grad();
    $b.zero_grad();
    $loss.backward(null);

    // Gradient descent, by hand. Under no_grad, because updating the weights
    // is not part of the model -- taping it would grow the graph every step.
    bt.no_grad(def() {
        nd_subtract($w.data, nd_multiply($w.grad, 0.5, null), $w.data);
        nd_subtract($b.data, nd_multiply($b.grad, 0.5, null), $b.data);
        return null;
    });
    $step = $step + 1;
}

print("w = " + str(nd_get(nd_reshape($w.data, [1]), [0])));    // ~3
print("b = " + str(nd_get(nd_reshape($b.data, [1]), [0])));    // ~2
```

Four things happened there, and they are the four things every training loop
does: **predict**, **measure**, **blame** (`backward`), **adjust**.

---

## 5. The same model with `nn`

`nn.Module` keeps the parameters for you.

```bantu
include "btorch/btorch.b" as bt;
include "btorch/nn.b" as nn;

bt.manual_seed(1);
$model = nn.Linear(1, 1, null);                 // in, out, bias (null = yes)
print(len($model.parameters()));                 // 2 -- weight and bias
print($model.param("weight").shape());           // [1, 1]
```

Layers compose with `Sequential`:

```bantu
$net = nn.Sequential([
    nn.Linear(4, 16, null), nn.ReLU(),
    nn.Linear(16, 16, null), nn.ReLU(),
    nn.Linear(16, 3, null)
]);
print($net.num_parameters());                    // 371
print($net.forward(bt.tensor(nd_random_normal([5, 4]), false)).shape());   // [5, 3]
```

`parameters()` walks the whole tree and **deduplicates by identity**, so a
weight shared between two layers is returned once — an optimizer cannot
accidentally step it twice.

The layers you get: `Linear`, `Conv1d`, `Conv2d`, `MaxPool2d`, `AvgPool2d`,
`Dropout`, `LayerNorm`, `BatchNorm1d`, `BatchNorm2d`, `Embedding`, `Flatten`,
`Sequential`, and the activations `ReLU`, `Tanh`, `Sigmoid`, `LeakyReLU`,
`GELU`, `Softmax`.

Two layers behave differently while training and while predicting — `Dropout`
and the `BatchNorm`s. Switch modes explicitly:

```bantu
$net.eval();     print($net.training);     // false
$net.train();    print($net.training);     // true
```

Forgetting `eval()` is the classic silent accuracy loss: the model trains fine
and scores badly, because dropout is still firing.

---

## 6. Losses

A loss turns a prediction and a target into one number to minimise.

```bantu
include "btorch/btorch.b" as bt;
include "btorch/nn.b" as nn;

$pred = bt.tensor([[1.0], [2.0], [3.0]], true);
$want = bt.tensor([[1.5], [2.0], [2.0]], false);

print(nn.mse_loss($pred, $want, null).item());        // mean squared error
print(nn.l1_loss($pred, $want, null).item());         // mean absolute error
print(nn.huber_loss($pred, $want, 1.0, null).item()); // squared near 0, linear far out
```

For classification, pass **raw scores** (logits) and **class indices**:

```bantu
$logits = bt.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], true);
$labels = nd([0, 1], "i64");                      // row 0 is class 0, row 1 is class 1
print(nn.cross_entropy($logits, $labels, null).item());
```

Do **not** apply softmax yourself first. `cross_entropy` fuses the softmax with
the log, which is what keeps it stable:

```bantu
$huge = bt.tensor([[1000.0, 1001.0]], false);
print(nn.cross_entropy($huge, nd([1], "i64"), null).item());   // finite, not NaN
print(nn.softmax($huge).to_list());                            // [[0.269, 0.731]]
```

A naive `log(exp(1000) / sum)` is `NaN`. This is not.

For two-class problems with a single output, use `bce_with_logits` (again on
raw scores):

```bantu
print(nn.bce_with_logits(bt.tensor([0 - 500.0], false),
                         bt.tensor([1.0], false), null).item());   // 500, not inf
```

`reduction` is the last argument everywhere: `null` or `"mean"` averages,
`"sum"` adds, `"none"` returns one loss per row.

```bantu
$p = bt.tensor([[1.0], [2.0]], false);
$t = bt.tensor([[0.0], [0.0]], false);
print(nn.mse_loss($p, $t, "none").to_list());     // [[1], [4]]
print(nn.mse_loss($p, $t, "sum").item());         // 5
```

---

## 7. Optimizers and learning rates

An optimizer owns the update rule, so you stop writing it by hand.

```bantu
include "btorch/btorch.b" as bt;
include "btorch/nn.b" as nn;
include "btorch/optim.b" as optim;

$net = nn.Linear(4, 2, null);

$opt = optim.SGD($net.parameters(), 0.1, 0.9, null);      // lr, momentum, weight_decay
$opt = optim.Adam($net.parameters(), 0.001, null, null, null);
$opt = optim.AdamW($net.parameters(), 0.001, 0.01, null, null);   // lr, weight_decay
$opt = optim.RMSprop($net.parameters(), 0.01, null, null);
$opt = optim.Adagrad($net.parameters(), 0.01, null);
```

Which to pick: **Adam at `1e-3`** if you have no reason to think otherwise, and
**SGD with momentum 0.9** when you are willing to tune for the last fraction of
a percent.

The three-line ritual, in this order:

```bantu
$mdl = nn.Linear(4, 2, null);
$sgd = optim.Adam($mdl.parameters(), 0.001, null, null, null);
$x = bt.tensor(nd_random_normal([8, 4]), false);
$y = bt.tensor(nd_random_normal([8, 2]), false);

$sgd.zero_grad();                                  // 1. clear last step's blame
nn.mse_loss($mdl.forward($x), $y, null).backward(null);    // 2. compute this step's
$sgd.step();                                       // 3. move the weights
```

Miss `zero_grad()` and gradients accumulate across steps — the model will
appear to train with a wildly wrong learning rate.

Learning-rate schedules wrap the optimizer and are stepped once per **epoch**,
not per batch:

```bantu
$mdl = nn.Linear(4, 2, null);
$sgd = optim.SGD($mdl.parameters(), 0.1, null, null, null);
$sched = optim.StepLR($sgd, 10, 0.5);              // halve every 10 epochs
print($sgd.lr);                                    // 0.1
$e = 0;
while ($e < 10) { $sched.step(); $e = $e + 1; }
print($sgd.lr);                                    // 0.05
```

Also `ExponentialLR($opt, gamma)` and `CosineAnnealingLR($opt, total, min_lr)`.

If gradients ever blow up, clip them over all parameters **jointly**:

```bantu
nn.clip_grad_norm($net.parameters(), 1.0);
```

---

## 8. A real training loop

Everything so far, assembled. The task is two interleaved spirals — a problem
no straight line can separate, so success means the non-linearity and its
gradient both work.

```bantu
include "btorch/btorch.b" as bt;
include "btorch/nn.b" as nn;
include "btorch/optim.b" as optim;

bt.manual_seed(3);
$N = 200;
$xs = nd_zeros([$N, 2], "f64");
$ys = nd_zeros([$N], "f64");
$i = 0;
while ($i < $N) {
    $cls = $i % 2;
    $t = ($i * 1.0 / $N) * 4.0;
    $ang = $t * 2.5 + $cls * 3.14159265358979;
    $r = 0.3 + $t * 0.25;
    nd_set($xs, [$i, 0], $r * nd_get(nd_cos(nd_full([1], $ang), null), [0]));
    nd_set($xs, [$i, 1], $r * nd_get(nd_sin(nd_full([1], $ang), null), [0]));
    nd_set($ys, [$i], $cls * 1.0);
    $i = $i + 1;
}

$net = nn.Sequential([
    nn.Linear(2, 32, null), nn.Tanh(),
    nn.Linear(32, 32, null), nn.Tanh(),
    nn.Linear(32, 2, null)
]);
$opt = optim.Adam($net.parameters(), 0.02, null, null, null);
$tx = bt.tensor($xs, false);

$epoch = 0;
while ($epoch < 150) {
    $net.train();
    $opt.zero_grad();
    $loss = nn.cross_entropy($net.forward($tx), nd_astype($ys, "i64"), null);
    $loss.backward(null);
    $opt.step();
    if (($epoch + 1) % 50 == 0) {
        print("  epoch " + str($epoch + 1) + "   loss " + str($loss.item()));
    }
    $epoch = $epoch + 1;
}

$net.eval();
$acc = bt.inference_mode(def() {
    $pred = nd_argmax($net.forward($tx).data, 1, null);
    $hit = nd_sum(nd_astype(nd_equal(nd_astype($pred, "i64"),
                                     nd_astype($ys, "i64"), null), "f64"), null, null);
    return nd_get(nd_reshape($hit, [1]), [0]) / ($N * 1.0);
});
print("  accuracy " + str($acc * 100.0) + "%");
```

> **A falling loss proves nothing on its own.** Gradient descent on a *wrong*
> gradient still often reduces a loss. If you write a custom operation, verify
> it with `gradcheck` (§15) — do not take the training curve as evidence.

---

## 9. Loading data

Doing full-batch gradient descent as above stops working once the data does not
fit in memory, or once you want the noise that small batches give you.

```bantu
include "btorch/btorch.b" as bt;
include "btorch/data.b" as btdata;

$xs = nd_random_normal([100, 4]);
$ys = nd_astype(nd_random_int([100], 0, 3), "f64");

$dset = btdata.TensorDataset([$xs, $ys]);      // tensors sharing a first axis
print($dset.size());                            // 100
```

Do not name that variable `$ds` if you also wrote `include ... as ds` — Bantu
keeps variables, functions and module aliases in one namespace, and the second
would silently destroy the first.

A `DataLoader` cuts it into batches:

```bantu
$dset = btdata.TensorDataset([nd_random_normal([100, 4]),
                              nd_astype(nd_random_int([100], 0, 3), "f64")]);
$loader = btdata.DataLoader($dset, 16, true, false, 42);
//                          dataset, batch_size, shuffle, drop_last, seed
print($loader.num_batches());                   // 7  (6 full + 1 of 4)
print($loader.batch(0)[0].shape());             // [16, 4]
print($loader.batch(6)[0].shape());             // [4, 4]  -- the remainder
```

Bantu has no `for x in y`, so a pass over an epoch is an index loop:

```bantu
$dl = btdata.DataLoader(btdata.TensorDataset([nd_random_normal([100, 4]),
          nd_astype(nd_random_int([100], 0, 3), "f64")]), 16, true, false, 42);
$b = 0;
while ($b < $dl.num_batches()) {
    $batch = $dl.batch($b);
    $inputs = $batch[0];                        // a tensor
    $targets = $batch[1].data;                  // the raw array, for cross_entropy
    $b = $b + 1;
}
$dl.reshuffle();                                // next epoch, new order
print("saw " + str($b) + " batches");
```

`reshuffle()` draws a **new** permutation rather than restarting the random
stream — so epoch 2 differs from epoch 1, while the whole run stays
reproducible from its seed. Both of those matter, and they are different
things.

With `drop_last = false`, one epoch's batches cover **every sample exactly
once**. You can check that yourself:

```bantu
$seen = nd_zeros([100], "f64");
$b = 0;
while ($b < $loader.num_batches()) {
    nd_index_add($seen, $loader.batch_indices($b), 1.0, 0);
    $b = $b + 1;
}
print(nd_array_equal($seen, nd_ones([100], "f64")));     // true
```

---

## 10. Train, validate, test

Never measure on what you trained on. `random_split` takes **exact counts**,
because rounding a fraction is how a test set silently ends up with one sample
in it.

```bantu
include "btorch/btorch.b" as bt;
include "btorch/data.b" as btdata;

$dset = btdata.TensorDataset([nd_random_normal([100, 4]),
                              nd_astype(nd_random_int([100], 0, 3), "f64")]);
$parts = btdata.random_split($dset, [70, 15, 15], 42);
print($parts[0].size());                        // 70  train
print($parts[1].size());                        // 15  validation
print($parts[2].size());                        // 15  test

$trainLoader = btdata.DataLoader($parts[0], 16, true, false, 42);
$valLoader   = btdata.DataLoader($parts[1], 16, false, false, null);
```

The parts are **disjoint** and cover everything — no leakage. Shuffle the
training loader; do not bother shuffling the others.

Use the validation set to decide things (when to stop, which learning rate) and
the test set exactly once, at the end, to report a number.

---

## 11. Saving and resuming

```bantu
include "btorch/btorch.b" as bt;
include "btorch/nn.b" as nn;
include "btorch/serialize.b" as ser;

bt.manual_seed(2);
$net = nn.Sequential([nn.Linear(4, 8, null), nn.BatchNorm1d(8, null, null),
                      nn.ReLU(), nn.Linear(8, 2, null)]);
$net.train();
$net.forward(bt.tensor(nd_random_normal([6, 4]), false));    // move the BN stats

$info = ser.save($net, "model.btsd");
print($info.entries);        // 8 -- six parameters AND two BatchNorm buffers
```

That count matters. A `BatchNorm` layer's running mean and variance are
**buffers**: state the model needs but no optimizer should ever touch. A
checkpoint that saved only parameters would reload a model whose `eval()` is
wrong, and it would train fine and score badly with nothing pointing at the
loader. btorch saves both.

Loading copies **into** the existing tensors, so an optimizer already holding
them keeps working:

```bantu
bt.manual_seed(999);
$fresh = nn.Sequential([nn.Linear(4, 8, null), nn.BatchNorm1d(8, null, null),
                        nn.ReLU(), nn.Linear(8, 2, null)]);
$c = ser.load_model($fresh, "model.btsd");
print(str($c.params) + " parameters, " + str($c.buffers) + " buffers");
```

Round trips are **bit-identical**, not approximate. To look inside a file
without touching a model:

```bantu
$f = ser.load("model.btsd");
print($f.order);                        // names, in file order
print($f.kinds["1.running_mean"]);      // "buffer"
```

A truncated file and a corrupted one raise **different** errors, and a file
written by a newer btorch is refused rather than read half-way.

For a long run, write a checkpoint every epoch and record where you are, so a
power cut costs one epoch rather than the whole run.
`samples/btorch/mnist_cnn.b` does exactly this — and it does it because the
first attempt at that run was lost to a shutdown.

---

## 12. Classifying handwriting: the MLP

Fetch the data once:

```sh
bash samples/btorch/get_mnist.sh      # 54 MB, SHA-256 verified
```

```bantu
include "btorch/btorch.b" as bt;
include "btorch/datasets.b" as ds;

$m = ds.mnist(null, "standard");
print(nd_shape($m.train_x));        // [60000, 784]
print(nd_shape($m.test_y));         // [10000]
```

`"standard"` subtracts MNIST's own mean and divides by its standard deviation.
It is not cosmetic — it is the difference between converging in four epochs and
in forty.

The whole model is three lines, and the training loop is the one from §8 with a
`DataLoader` in it. The full script is
[`samples/btorch/mnist_mlp.b`](../samples/btorch/mnist_mlp.b):

```bantu
$net = nn.Sequential([
    nn.Linear(784, 256, null), nn.ReLU(),
    nn.Linear(256, 128, null), nn.ReLU(),
    nn.Linear(128, 10, null)
]);
print($net.num_parameters());        // 235146
```

Run it and you get **98.16%** after 8 epochs, about 20 seconds an epoch:

```
epoch 1/8   loss 0.272   test acc 95.4%
epoch 4/8   loss 0.052   test acc 97.51%
epoch 8/8   loss 0.021   test acc 98.16%
```

Measuring accuracy is worth doing properly — in `eval()` mode so dropout is
off, and under `inference_mode` so you do not build 10,000 rows of tape for
nothing:

```bantu
include "btorch/nn.b" as nn;
$net = nn.Sequential([nn.Linear(4, 3, null)]);
$x = bt.tensor(nd_random_normal([20, 4]), false);
$y = nd_astype(nd_random_int([20], 0, 3), "f64");

$net.eval();
$acc = bt.inference_mode(def() { return ds.accuracy($net.forward($x), $y); });
$net.train();
print("accuracy " + str($acc));
```

---

## 13. Convolutional networks

An MLP throws away the fact that neighbouring pixels are related. A convolution
does not.

```bantu
include "btorch/btorch.b" as bt;
include "btorch/nn.b" as nn;

bt.manual_seed(4);
$images = bt.tensor(nd_random_normal([8, 1, 28, 28]), false);    // [N, C, H, W]

$conv = nn.Conv2d(1, 6, 5, 1, 2, null);   // in_ch, out_ch, kernel, stride, padding
print($conv.forward($images).shape());     // [8, 6, 28, 28] -- padding 2 keeps the size
print($conv.param("weight").shape());      // [6, 1, 5, 5]

$pool = nn.MaxPool2d(2, null, null);
print($pool.forward($conv.forward($images)).shape());    // [8, 6, 14, 14]
```

Pooling's stride defaults to the **kernel** (so 2×2 pooling halves the image);
convolution's stride defaults to 1. `stride`, `padding` and `dilation` each
take a number or a `[height, width]` pair.

Images are `[batch, channels, height, width]`. MNIST arrives flat, so reshape:

```bantu
include "btorch/datasets.b" as ds;
$flat = bt.tensor(nd_random_normal([8, 784]), false);
print(ds.as_images($flat).shape());        // [8, 1, 28, 28]
```

LeNet-5, the classic, is the architecture in
[`samples/btorch/mnist_cnn.b`](../samples/btorch/mnist_cnn.b):

```bantu
$cnn = nn.Sequential([
    nn.Conv2d(1, 6, 5, 1, 2, null),   nn.ReLU(), nn.MaxPool2d(2, null, null),
    nn.Conv2d(6, 16, 5, 1, 0, null),  nn.ReLU(), nn.MaxPool2d(2, null, null),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120, null), nn.ReLU(),
    nn.Linear(120, 84, null),         nn.ReLU(),
    nn.Linear(84, 10, null)
]);
print($cnn.num_parameters());              // 61706
print($cnn.forward(bt.tensor(nd_random_normal([2, 1, 28, 28]), false)).shape());
```

That reaches **98.98%** in 4 epochs, about 107 seconds an epoch. Working out
`16 * 5 * 5` for the first `Linear` is the fiddly part: 28 → pool → 14 → conv
with no padding → 10 → pool → 5, with 16 channels. If you get it wrong, the
error names both shapes.

Convolution is still the slowest part of btorch, though less so than it was:
profiling and fixing three things took a LeNet training step from 620 ms to
240 ms at batch 128. What remains is a matrix-multiply blocking problem for very
skinny shapes, which needs a packed micro-kernel. `btorch-suite/CHANGELOG.md`
has the measurements and `DECISIONS.md` B29 has the reasoning.

---

## 14. Embeddings and sequences

Words are not numbers. An `Embedding` is a lookup table mapping each token id
to a learned vector.

```bantu
include "btorch/btorch.b" as bt;
include "btorch/nn.b" as nn;

bt.manual_seed(5);
$emb = nn.Embedding(1000, 16, null);       // vocabulary, dimensions, padding_idx
$tokens = nd([[4, 19, 7], [2, 2, 0]], "i64");       // 2 sentences, 3 tokens each
print($emb.forward($tokens).shape());      // [2, 3, 16]
```

Any index shape works; you get that shape plus the embedding dimension.

A repeated token **accumulates** both of its gradients into that row — which is
the whole reason the backward pass uses a scatter that adds rather than one
that overwrites:

```bantu
$e = nn.Embedding(3, 2, null);
$e.zero_grad();
$e.forward(nd([0, 0, 1], "i64")).sum(null, null).backward(null);
print(nd_to_list(nd_take($e.param("weight").grad, [0], 0)));   // [[2, 2]] -- seen twice
print(nd_to_list(nd_take($e.param("weight").grad, [2], 0)));   // [[0, 0]] -- never seen
```

`padding_idx` pins a row at zero and gives it no gradient, for padded
positions:

```bantu
$pad = nn.Embedding(5, 3, 0);
print(nd_to_list(nd_take($pad.param("weight").data, [0], 0)));   // [[0, 0, 0]]
```

For building sequence models, the pieces are here — `cat`, `stack`, `chunk`,
`gather`, `permute`, `LayerNorm`, `Softmax` — but `RNN`, `LSTM` and
`MultiheadAttention` are not yet written. You would assemble them from the
above.

---

## 15. When it goes wrong

### Check the gradient, not the loss

If you write a custom operation, verify it numerically. `gradcheck` nudges
every input a little and compares the change it sees with the gradient you
claim:

```bantu
include "btorch/btorch.b" as bt;

$r = bt.gradcheck(def($in) { return bt.sum(bt.mul($in[0], $in[0]), null, null); },
                  [bt.tensor([1.0, 2.0, 3.0], true)], null, null);
print($r.ok);                 // true
print($r.worst);              // the worst relative disagreement
```

The function must return **one element**, and it must be a **pure function of
its inputs**. Draw randomness inside it and the two nudged evaluations use
different random numbers — the result is garbage that looks exactly like a
broken gradient (a relative error near 1.0 and a numerical gradient in the
thousands). Draw it once, outside:

```bantu
$coef = bt.tensor(nd_random_normal([3]), false);          // ONCE
$r = bt.gradcheck(def($in) { return bt.sum(bt.mul($in[0], $coef), null, null); },
                  [bt.tensor(nd_random_normal([3]), true)], null, null);
print($r.ok);
```

### Errors you will actually hit

**`backward() on a non-scalar tensor`** — `backward()` needs one number to
start from. Reduce first:

```bantu
$x = bt.tensor([1.0, 2.0], true);
bt.sum(bt.mul($x, $x), null, null).backward(null);         // right
print(nd_to_list($x.grad));
```

**`backward() on a tensor that does not require grad`** — nothing upstream was
marked. Pass `true` when you create the parameter.

**`item() needs a one-element tensor`** — you have more than one. Use
`.to_list()`, or reduce.

**A comparison that is always false.** A full reduction gives a 0-d *array*,
not a number:

```bantu
$a = nd([1.0, 2.0], "f64");
print(nd_sum($a, null, null) == 3.0);                      // false -- a handle vs a number
print(nd_get(nd_reshape(nd_sum($a, null, null), [1]), [0]) == 3.0);    // true
print(bt.sum(bt.tensor($a, false), null, null).item() == 3.0);         // true
```

**`Undefined variable: nd_something`** — your `bantu` binary is older than
btorch. `bantu doctor`.

**`it holds a number, not a function`** — you named a variable the same as a
function. Bantu has one namespace for variables, functions and module aliases,
so `$take = 3` breaks `take(...)` and `$ds = x` breaks `ds.mnist(...)`.

**A model that trains and scores badly** — you forgot `eval()`, or your
checkpoint lost its buffers. Both are covered above.

### Things that bite quietly

- **Closures capture by reference.** Three closures built over a loop variable
  all see its final value. Build per-iteration closures with a factory
  function whose parameters bind once per call.
- **`pop(list)` returns the element; `push(list, v)` returns the list.** So
  `$s = pop($s)` assigns an element to `$s`. Write `pop($s);` on its own.
- **Recursion stops at 1000 calls.** Anything walking a user-sized structure
  must loop, not recurse.
- **The C-style `for` silently stops above 100,000 iterations.** Use `while`.
- **Everything is f64.** MNIST as f64 is 376 MB. There is no f32 yet.

---

## 16. Writing your own layer

Subclass `Module`, call `super()`, register parameters:

```bantu
include "btorch/btorch.b" as bt;
include "btorch/nn.b" as nn;

class _Affine extends nn.Module {
    def init($n) {
        super();
        this.register("scale", bt.ones([$n], true));
        this.register("shift", bt.zeros([$n], true));
    }
    def forward($x) {
        return bt.add(bt.mul($x, this.param("scale")), this.param("shift"));
    }
}
def Affine($n) { return new _Affine($n); }

$layer = Affine(3);
print(len($layer.parameters()));                 // 2
print($layer.forward(bt.tensor([[1.0, 2.0, 3.0]], false)).to_list());
```

**The factory must not share the class's name.** `def Affine()` containing
`new Affine(...)` re-enters the factory and recurses until the depth guard
fires — one namespace again. Hence the leading underscore.

Because the layer is built from btorch operations, its gradient comes for free.
Check it anyway:

```bantu
// Affine, repeated so this block stands alone.
class _Aff2 extends nn.Module {
    def init($n) {
        super();
        this.register("scale", bt.ones([$n], true));
        this.register("shift", bt.zeros([$n], true));
    }
    def forward($x) {
        return bt.add(bt.mul($x, this.param("scale")), this.param("shift"));
    }
}
$k = bt.tensor(nd_random_normal([1, 3]), false);
$r = bt.gradcheck(def($in) {
        $l = new _Aff2(3);
        nd_add($in[0].data, 0.0, $l.param("scale").data);
        return bt.sum(bt.mul($l.forward($k), $in[0]), null, null);
     },
     [bt.tensor(nd_random_normal([3]), true)], null, null);
print($r.ok);
```

Nest layers with `add` and `get`:

```bantu
class _Block extends nn.Module {
    def init($n) {
        super();
        this.add("norm", nn.LayerNorm($n, null));
        this.add("fc", nn.Linear($n, $n, null));
    }
    def forward($x) {
        return bt.add($x, this.get("fc").forward(this.get("norm").forward($x)));
    }
}
$blk = new _Block(8);
print(len($blk.parameters()));          // 4 -- found through the tree
print($blk.forward(bt.tensor(nd_random_normal([2, 8]), false)).shape());
```

For state that is **not** a parameter — a running average, a mask — use
`register_buffer`. It is saved with the model and no optimizer will touch it:

```bantu
class _Counter extends nn.Module {
    def init() {
        super();
        this.register_buffer("seen", bt.zeros([1], false));
    }
    def forward($x) {
        $c = this.buffer("seen");
        nd_add($c.data, 1.0, $c.data);
        return $x;
    }
}
$cnt = new _Counter();
$cnt.forward(bt.tensor([1.0], false));
$cnt.forward(bt.tensor([1.0], false));
print(nd_to_list($cnt.buffer("seen").data));     // [2]
print(len($cnt.parameters()));                    // 0 -- a buffer is not a parameter
```

---

## 17. What to read next

- [`btorch.md`](btorch.md) — the reference. Every function, grouped.
- [`samples/btorch/`](../samples/btorch/) — `01_autograd`,
  `02_linear_regression`, `03_classifier`, `04_save_load`, and the two MNIST
  gates. All runnable.
- [`btorch-architecture.md`](btorch-architecture.md) — why it is built the way
  it is: the Bantu/native split, the tape, and why convolution has no
  hand-written gradient.
- [`btorch-suite/DECISIONS.md`](../btorch-suite/DECISIONS.md) — 27 decisions
  with their reasoning, including the ones that are deliberately *not* what
  PyTorch does.
- [`btorch-suite/CHANGELOG.md`](../btorch-suite/CHANGELOG.md) — every
  measurement, and every defect the tests caught.

Run the test suite to confirm your install:

```sh
bantu run tests/btorch_autograd_test.b     # 121 assertions, 33 gradchecks
bantu run tests/btorch_conv_test.b         # 119 assertions, 24 gradchecks
bash tests/btorch_samples_test.sh          # every sample and every doc example
```
