# The Bantu AI/ML ecosystem — architecture and plan

How Bantu gets a competitive machine-learning stack, what goes where, and in
what order. This is the umbrella document: each stage below links to its own
design doc, and each new package gets the usual
`<pkg>-suite/{ROADMAP,DECISIONS,CHANGELOG}.md` triplet once it starts.

Written after `btorch` 1.0.0 shipped (MNIST: 98.16% MLP, 98.98% LeNet-5). The
question that prompted it was "should all of this be one package?" — the answer
is no, and the reasoning is [E2](#e2).

---

## 1. Three tiers, not two

The framing that resolves most of the packaging questions. Every item on the
roadmap splits across three places, and **the highest-leverage work is not in a
package at all**:

| tier | what lives there | versioned with | example |
|---|---|---|---|
| **1. The binary** | native kernel families | the language | `nd_matmul`, `tree_hist`, `tok_bpe` |
| **2. Packages** | pure Bantu over those atoms | themselves, via the registry | `numba`, `btorch`, `bkit` |
| **3. Neither** | patterns, samples, CLI subcommands, docs | nothing | the serving pattern |

f32, threading, a packed GEMM, FFT, int8 matmul and histogram split-finding are
**all tier 1**. They ship inside `bantu`. A package cannot add a dtype, cannot
start a thread pool, and cannot make a kernel faster — so planning stage 1 as a
package would have been a category error.

Tier 3 is the one that gets forgotten. Not everything useful is a library.

---

## 2. The dependency graph

Strictly a DAG. An arrow means "depends on". Nothing points left.

```
                         ┌──────────────┐
                         │ bantu binary │   nd_ col_ bp_ tree_ tok_ sp_ q8_
                         └──────┬───────┘
                                │
                         ┌──────▼───────┐
                         │    numba     │   arrays, dtypes, FFT, RNG authority
                         └──┬────┬───┬──┘
              ┌─────────────┘    │   └──────────────┬─────────────┐
              │                  │                  │             │
        ┌─────▼─────┐      ┌─────▼─────┐      ┌─────▼────┐  ┌─────▼────┐
        │  arctic   │      │  btorch   │      │  bsci    │  │  btext   │
        │ dataframes│      │ autograd  │      │  scipy   │  │tokenizers│
        │ + parquet │      │ + nn      │      │          │  │          │
        └─────┬─────┘      └──┬─────┬──┘      └────┬─────┘  └────┬─────┘
              │               │     │               │             │
              │         ┌─────▼──┐  │               │             │
              │         │ bonnx  │  │               │             │
              │         │ import │  │               │             │
              │         └────────┘  │               │             │
              │                     │               │             │
              └──────────┬──────────┴───────────────┘             │
                         │                                        │
                   ┌─────▼──────┐                                 │
                   │    bkit    │◄────────────────────────────────┘
                   │ classical  │   (btext vectorizers implement
                   │     ML     │    bkit's transformer protocol)
                   └─────┬──────┘
                         │
                   ┌─────▼──────┐        ┌──────────┐
                   │   bstats   │        │ bvision  │──► pillow, btorch
                   │ statsmodels│        │transforms│
                   └────────────┘        └──────────┘
```

Two edges deserve comment:

- **`bsci` does not depend on `btorch`** ([E18](#e18)). `bsci.optimize` takes a
  plain objective plus an optional gradient over numba arrays. A btorch user
  bridges it with a five-line documented pattern. Pointing the arrow the other
  way would make every scientific-computing user install an autograd engine.
- **`btext` → `bkit` is a protocol edge, not a code edge in the obvious
  direction.** `btext`'s vectorizers implement `bkit`'s transformer protocol so
  they drop into a `Pipeline`. That means `btext` depends on `bkit` for the
  protocol definition only — which is why the protocol is defined in a single
  small file that is cheap to depend on ([E12](#e12)).

---

## 3. Package map

**Four new packages, two deferred.**

| package | analogue | contains | native family | depends on | stage |
|---|---|---|---|---|---|
| **`bkit`** | scikit-learn **+** xgboost/lightgbm | preprocessing, model selection, metrics, linear models, pipelines, trees, **GBDT**, k-means/PCA, naive Bayes, kNN, SVM | `tree_` | numba, arctic | 3 |
| **`bonnx`** | onnxruntime (import side) | ONNX model import → `nn.Module`; export later | — | numba, btorch | 4 |
| **`btext`** | tokenizers + sklearn.feature_extraction.text | BPE/WordPiece/Unigram, normalizers, vocab, tf-idf, n-grams | `tok_` | numba, bkit | 5 |
| **`bsci`** | scipy | optimize, stats, signal, sparse, interpolate, special | `sp_` | numba | 6 |
| `bvision` *(deferred)* | torchvision | augmentation, transforms, CIFAR/ImageFolder readers | — | pillow, btorch | — |
| `bstats` *(deferred)* | statsmodels | GLM, ARIMA/ETS, survival, inference | — | bsci, arctic | — |

**Existing packages that grow instead:**

| item | home | why not its own package |
|---|---|---|
| dtype templating, f32, threading, packed GEMM, FFT, `nd_memmap`, int8 dtype | **binary + `numba`** | it is the array layer; a package cannot add a dtype |
| RNN/GRU/LSTM, MultiheadAttention, TransformerEncoder | **`btorch`** (`rnn.b`, `attention.b`) | they are `nn.Module`s over primitives that already exist; torch ships them in core |
| ConvTranspose2d, grouped/depthwise conv, AdaptiveAvgPool2d | **`btorch`** (`nn.b`) | same |
| int8 quantization | **`btorch`** (`quant.b`) | it must know every layer type to swap it ([E5](#e5)) |
| Parquet, Arrow, out-of-core scans | **`arctic`** | Parquet is a dataframe format; arctic already has optional Arrow |

---

## 4. Native family allocation

Adding a native prefix means adding a row to `kNativeFamilies` in
`bantu-src/compiler/src/types.hpp:499`, so that an unbound name produces the
stale-binary diagnosis rather than a bare `Undefined variable` (B21). Families
are a deliberate, enumerable commitment — not an accident.

| family | owner | status | added in |
|---|---|---|---|
| `nd_` | numba | shipped | — |
| `col_` | arctic | shipped | — |
| `bp_` | bplot / raster | shipped | — |
| `aead_` `sodium_` `pwhash_` `b64url_` `ct_` | crypto | shipped | — |
| `webpush_` | Web Push | shipped | — |
| `gc_` | the collector | shipped | — |
| **`tree_`** | bkit | planned | stage 3 |
| **`tok_`** | btext | planned | stage 5 |
| **`q8_`** | btorch.quant | planned | stage 6 |
| **`sp_`** | bsci.sparse | planned | stage 6 |

**No `pb_` family.** ONNX parsing needs a varint/length-delimited scanner, and
the first instinct is to make it native. The decision is to **measure first**
([E9](#e9)): write the reader in Bantu over `nd_frombytes`, profile it on a
100 MB model, and add `pb_` only if it is the bottleneck. The project's own
norm is that two of three performance suspects were wrong when last measured.

---

## 5. The five contracts

These must be settled once. Each is cheap now and expensive after three
packages have each invented their own.

### C1 — One estimator protocol

`bkit` defines it; `btext` vectorizers and (later) `bvision` transforms conform.

```
fit($X, $y)              -> this            // $y null for unsupervised
predict($X)              -> array
predict_proba($X)        -> array           // classifiers only
transform($X)            -> array           // transformers only
fit_transform($X, $y)    -> array
score($X, $y)            -> number
get_params()             -> dict            // for grid search
set_params($dict)        -> this
clone()                  -> a fresh unfitted copy with the same params
```

`clone()` is not optional: cross-validation must refit from scratch, and
reusing a fitted estimator across folds leaks the test set into training. That
is a silent correctness bug that still produces a plausible score — the same
class of failure as B10 ("a falling loss is not evidence").

### C2 — One dataset protocol

`btorch.data.Dataset` already defines `len()` / `get(i)`. `bvision` and `btext`
datasets **extend** it. They do not reinvent it, and `DataLoader` works on all
of them without a special case.

### C3 — One model container

`.btsd` already exists: a JSON header over a binary payload with a SHA-256
(B23). `bkit` saves GBDTs and scalers into **the same container** with a
different `kind` field in the header — it does not invent `.bkit`.

```json
{ "kind": "btorch.module" | "bkit.estimator" | "btext.tokenizer",
  "format_version": 1, "...": "per-kind payload description" }
```

One loader, one integrity story, and one `bantu inspect <file>` that works on
any model any package produced. Twelve distinct load-failure messages already
exist; they should not be written thrice.

### C4 — One RNG

`nd_seed` (native, thread-local) is already the single authority, and
`bt.manual_seed` delegates to it. `bkit`'s KFold shuffle, `btext`'s sampling
and `bvision`'s augmentation **must** delegate too, never carry their own
state. Reproducibility is the ecosystem's differentiator; it dies quietly if
each package seeds separately.

*Consequence for stage 1:* `nd_seed` is documented as per-thread. A thread pool
([E15](#e15)) must therefore either seed deterministically per partition or
keep all RNG on the calling thread. Resolved in
[`numba-dtype-architecture.md`](numba-dtype-architecture.md).

### C5 — One metrics home

`ds.accuracy` currently lives in `btorch/datasets.b`. Once `bkit.metrics`
exists, that is the wrong house: move it, leave a delegating alias in btorch so
no existing script breaks, and note the deprecation in the CHANGELOG.

---

## 6. Decisions

Numbered so code comments, tests and CHANGELOGs can cite them. A decision is
recorded when a different choice was genuinely available.

### E1 — Kernels in the binary, libraries in packages, patterns in neither
See §1. The corollary that matters: a package's `package.json` must be able to
declare a **language** floor, not just package versions ([E17](#e17)).

### E2 — Not one mega-package
A single `bai` fails for a concrete reason, not an aesthetic one: **btorch is
at 1.0.0 behind MNIST accuracy gates.** Put a tokenizer in the same package and
every tokenizer change must clear a 107-second CNN epoch to merge — blast
radius becomes release cadence. And `bonnx` will churn for a year as operators
are added while a GBDT can reach 1.0 in weeks; those cannot share a version
number honestly.

Twelve packages fails too: each costs a manifest, a suite triplet, a docs page,
a tutorial, a test tier and a registry publish. btorch just paid that bill.

**The rule:** *if A is always installed with B, they are one package.*

**Rejected:** grouping by technique (all the "linear" things together, all the
"tree" things together). Users do not install by technique.

### E3 — GBDT lives inside `bkit`
In Python, `xgboost` is separate from `scikit-learn` because they are separate
*organizations* — not for architectural reasons. Metrics, preprocessing and
cross-validation are exactly what wraps a GBDT in practice, so splitting forces
co-installation anyway. [E2](#e2)'s rule applies: merge.

### E4 — FFT lives in `numba`, not `bsci`
NumPy has `numpy.fft`, and that is correct: a DFT is an array transform. So
`nd_fft`/`nd_rfft`/`nd_ifft` are kernels exposed through numba, and
`bsci.signal` builds windows, spectrograms, filters, `convolve` and `resample`
on top.

**Rejected:** FFT in bsci. It would mean a numba user cannot transform an array
without a scipy-class dependency, which inverts the layering.

### E5 — Quantization lives inside `btorch`
It must know every layer type in order to swap it. A separate package breaks
every time `nn.b` adds a layer — a dependency edge pointing at a moving
internal API. torch ships `torch.quantization` in core for the same reason.

### E6 — Parquet/Arrow in `arctic`; memory-mapping in `numba`
Parquet is a dataframe format and arctic already has optional Arrow. `nd_memmap`
belongs in numba because NumPy has `memmap` and the thing being mapped is an
array. Neither is a new package.

### E7 — Serving is a sample, not a package
It is `btorch` + `sua` + a documented pattern. Write
`samples/btorch/serve.b`. If that sample gets copied three times in real use,
*then* it is a package. Not before.

**Rejected:** a `bserve` package now. Batching, model registry, warm-up and
health checks are perhaps 300 lines, but designing them without a single user
is how you get an API nobody wants and cannot remove.

### E8 — Experiment tracking is deferred entirely
A run directory, a metrics JSONL and a config snapshot cover 80% of it, and
`.btsd` plus `bplot` cover the rest. Revisit only if stages 3–6 produce a real
need.

### E9 — No general protobuf package; a minimal reader inside `bonnx`
ONNX has a fixed schema. A general protobuf library is a year of work for a
feature nobody asked for. Write the wire-format reader in Bantu inside bonnx,
profile it on a 100 MB model, and promote it to a `pb_` native family **only if
measured to be the bottleneck** — not on suspicion.

### E10 — One model container: `.btsd` with a `kind` header
See [C3](#c3-one-model-container).

### E11 — `nd_seed` is the only RNG
See [C4](#c4-one-rng).

### E12 — The estimator protocol is defined by `bkit`, in one small file
`bkit/base.b` holds the protocol and nothing else, so `btext` can depend on it
without pulling in GBDT, SVMs and cross-validation. A protocol defined inside a
large module becomes a reason not to conform.

### E13 — The dataset protocol is defined by `btorch.data`
See [C2](#c2-one-dataset-protocol). It already exists and already works; a
second one would be strictly worse.

### E14 — Metrics move to `bkit.metrics`, with an alias left behind
See [C5](#c5-one-metrics-home). Non-destructive: the language is in production,
so nothing that works today may stop working.

### E15 — Threading uses a deterministic fixed partition, not work stealing
**The single most consequential decision here.** A work-stealing pool changes
floating-point reduction order run to run, which makes results irreproducible —
killing the determinism-and-audit story that is Bantu's best differentiator
against PyTorch.

So: a thread pool with a **fixed partition computed from shape and thread
count**, giving bit-identical results regardless of how many threads run, plus
`nd_set_threads(n)`. Slightly slower than work stealing. Keeps the property
that makes the stack worth choosing in a regulated setting.

**Rejected:** (a) work stealing, for the reason above; (b) language-level
threads, which touch the GC and the evaluator and are a far larger change for
the same arithmetic benefit.

### E16 — The dtype templating pass precedes f32
Every kernel today has an f64 fast path and a generic path. Adding f32 and
int8 by hand turns that into a 4×2 matrix of hand-written branches. Template
each kernel once over `T` **first**, then add dtypes. More boring; the
difference between a week and a quarter at stage 6.

Grounded: the surface is **76 `DType::F64` branches** and ~44 `double*` sites
across `ndarray_*`, and `NdArray::ptr<T>()` is **already templated** with
`offset`/`strides` counted in **elements, not bytes** — so f32 requires no
stride arithmetic changes at all. The job is bounded.

### E17 — `package.json` gains a `bantu` engine field
No manifest has one today (verified across the tree). The moment `btorch` 1.1
needs `nd_fft`, it depends on a **binary** version, not a package version, and
the manifest cannot say so. The failure mode is the stale-binary bug already
fixed once (B21), arriving through the registry and reported as
`Undefined variable: nd_fft`.

### E20 — stage 1 is resequenced: value-ordered, not completion-ordered
**The original order inside stage 1 was wrong, and re-deriving it from
dependencies rather than from tidiness is what showed it.**

It said D2b (generic ufunc functors) → D3 (threading) → D4 (packed GEMM). But
**D2b's entire benefit was gated behind D4.** f32 was unusable for training
until `nd_matmul` preserved it (N31), so making f32 element-wise arithmetic
happen at f32 width delivered nothing to anybody: there were no f32 models to
run it on. Putting the gated item first is the one ordering that cannot pay.

So D4's **f32-GEMM half was pulled forward** and done first, together with D2c
(the btorch plumbing). That turns f32 from a shipped-but-unusable dtype into a
working one — the worst state to leave anything in is "built, tested, and
reachable by nobody".

Revised order, by value delivered per unit of risk:

| | item | why here |
|---|---|---|
| ✅ | **f32 GEMM + D2c** | completes a shipped feature; halves model and optimizer memory |
| next | **D3 deterministic threading** | the largest multiplier that exists (core count), and the one stage 3 also wants |
| then | **stage 2 + stage 3 (`bkit`)** | where the paying work is |
| after | **D4 packed GEMM** (skinny shapes) | helps convolution only |
| after | **D2b** generic functors | helps f32 element-wise only |

### E21 — stage 1 and stage 3 are interleaved, not swapped
The open trade was: *GBDTs are nearly insensitive to f32 vs f64, so should
`bkit` jump ahead of the array layer?* Neither "yes" nor "no" is the best
answer.

**`bkit` does not need f32 at all** — its hot structure is a byte-per-cell
binned matrix ([K1]) and its hot loop is integer bin indexing with float
accumulation. True.

**But `bkit` badly wants D3.** `tree_hist` is O(samples x features) per node and
partitions cleanly over samples: each thread builds a partial histogram and the
partials are summed in fixed index order, so it is race-free *and*
deterministic by construction — exactly the shape D3's fixed-partition design
([E15]) serves. Built before D3, `bkit` either ships single-threaded or gets
revisited; built after, it gets the cores nearly free.

So the right cut is not between stages, it is **through** stage 1:

- take the stage-1 parts that either complete a shipped feature (f32 GEMM) or
  multiply `bkit` (D3 threading), **then** do stage 2 and `bkit`;
- defer the stage-1 parts that serve narrower constituencies — D4's packed-GEMM
  half (convolution) and D2b (f32 element-wise) — until after.

`bkit` therefore moves **earlier** than the original plan, without `bkit`
paying for a single-threaded histogram or the array layer being left
half-finished.

**Requirement this places on D3:** the thread pool must be a *reusable
primitive*, not something welded into the `nd_*` kernels, because `tree_` and
later `tok_` need it too. If D3 ships a pool that only `nd_*` can call, this
interleave stops working and `bkit` is back to single-threaded.

---

### E18 — `bsci.optimize` does not depend on `btorch`
See §2. The bridge is a documented pattern, not a dependency edge.

### E19 — f64 remains the default dtype, forever
f32 is opt-in. Every existing test must pass unchanged after stage 1, bit for
bit. A default-dtype change would silently alter every published result and
every gradcheck tolerance in the tree.

---

## 7. The stages

Order fixed by the user. Each stage's gates follow btorch's six tiers: feature,
differential, gradcheck (where gradients exist), stress, regression,
sanitizers. **Every stage re-runs `benchmarks/run.sh`** so an interpreter-wide
regression cannot hide.

| stage | what | where | design doc |
|---|---|---|---|
| **1** | dtype templating → f32 → **f32 GEMM** → deterministic threading → (packed GEMM, generic functors deferred — E20) | binary + `numba` | [`numba-dtype-architecture.md`](numba-dtype-architecture.md) |
| **2** | the engine field + dependency check | binary (package manager) | §7.2 below |
| **3** | `bkit` with GBDT | new package | [`bkit-architecture.md`](bkit-architecture.md) |
| **4** | `bonnx` import | new package | [`bonnx-architecture.md`](bonnx-architecture.md) |
| **5** | `btext` + btorch phase 8 | new package + `btorch` | [`btext-architecture.md`](btext-architecture.md) |
| **6** | int8 quant + `bsci` | `btorch` + new package | [`bsci-architecture.md`](bsci-architecture.md) |

### 7.1 Stage 1 — the array layer (multiplies everything below)

Phases tracked in [`../numba-suite/ROADMAP.md`](../numba-suite/ROADMAP.md) as
D1–D4. Full design in
[`numba-dtype-architecture.md`](numba-dtype-architecture.md).

| phase | item | gate |
|---|---|---|
| **D1** | dtype templating pass — `load<T>`/`store<T>`, `itemsize`, `promote`, a dispatch macro replacing the hand-written `if (cdt == F64)` pairs | **every existing test bit-identical**; no new dtype yet |
| **D2** | `f32` dtype end to end — construction, ufuncs, reductions, linalg, sort, manip, I/O, `nd_astype` | f32 within 1e-6 of the f64 answer; f64 paths bit-identical; memory halves |
| **D3** | deterministic threading — fixed partition, `nd_set_threads(n)` | **bit-identical across 1/2/4/8/16 threads**; ASan+TSan clean |
| **D4** | packed GEMM micro-kernel for skinny shapes (`K=25, N=6`) | the three conv matmuls faster; `numba_fastpath_test.b` still exact |

Why first: it multiplies every stage below. f32 halves memory and doubles SIMD
width; threading multiplies by core count; the packed GEMM closes the
convolution gap that phase 9 measured and left open.

The counter-argument, stated honestly: **GBDTs on tabular data are nearly
insensitive to f32 vs f64 and are the commercially valuable item.** If
`bkit` landing sooner matters more than every later package being faster, stages
1 and 3 swap. The order below reflects the choice made.

### 7.2 Stage 2 — the engine field (small; unblocks safe publishing)

| phase | item | gate |
|---|---|---|
| **L1** | `"bantu": ">=1.4.0"` parsed from `package.json` into `Manifest` | malformed ranges rejected by name |
| **L2** | `bantu install` refuses, with the required version and the current build id, before writing any file | a too-old binary produces one clear message, not a runtime `Undefined variable` |
| **L3** | backfill every existing manifest with the floor it actually needs | `bantu doctor` reports each package's floor against the running build |
| **L4** | a native-family → minimum-version table, so `nd_fft` maps to a stated floor | adding a kernel without updating the table fails a test |

Deliberately before stage 3: `bkit` ships a new native family (`tree_`), and it
is the first package that would hit this trap in the wild.

### 7.3 Stage 3 — `bkit` (where the paying work is)

Phases K0–K12 in [`../bkit-suite/ROADMAP.md`](../bkit-suite/ROADMAP.md);
design in [`bkit-architecture.md`](bkit-architecture.md).

The headline is **K7–K8**: histogram-based trees and a gradient-boosting
implementation of xgboost/lightgbm class. Blunt truth behind the ordering —
most production tabular ML is GBDT, not deep learning, and btorch cannot do it.

### 7.4 Stage 4 — `bonnx` import (relevance without a training-speed race)

Phases O0–O8 in [`../bonnx-suite/ROADMAP.md`](../bonnx-suite/ROADMAP.md).

Strategically the highest-leverage single package: train on a GPU in PyTorch,
deploy with one static Bantu binary at the edge. It buys access to the world's
existing models without competing on training throughput, which is a race this
stack cannot win and should not enter.

### 7.5 Stage 5 — `btext` + btorch phase 8 (opens NLP)

`btext` phases T0–T7 in
[`../btext-suite/ROADMAP.md`](../btext-suite/ROADMAP.md); btorch's sequence
layers as phase 8 in
[`../btorch-suite/ROADMAP.md`](../btorch-suite/ROADMAP.md). Paired because
neither is useful alone: a tokenizer with nothing to feed, or an attention
layer with no way to turn text into ids.

### 7.6 Stage 6 — int8 quantization + `bsci` (edge inference, then science)

Quantization as `btorch/quant.b` plus the `q8_` family; `bsci` phases S0–S6 in
[`../bsci-suite/ROADMAP.md`](../bsci-suite/ROADMAP.md).

int8 is the performance story for a stack with no GPU: 4× the arithmetic
throughput and 4× less memory, pairing exactly with the static-binary
deployment wedge.

---

## 8. Not scheduled

Honest list of things identified as valuable and deliberately left out of the
six stages, so they are not mistaken for oversights:

| item | why not now |
|---|---|
| `nd_memmap` / out-of-core arrays | needed only once a dataset exceeds RAM; no such dataset in the tree yet |
| `bvision` | `bkit` + `pillow` cover the near-term cases; torchvision's value is its model zoo, which `bonnx` supplies instead |
| `bstats` | depends on `bsci.optimize` (stage 6), so it cannot start earlier |
| ONNX **export** | import is the asymmetric win; export matters only once models are *trained* here at scale |
| distributed training | no multi-machine story, and stage 1's threading is the honest ceiling |
| GPU | a different project; the wedge is explicitly CPU + static binary |
| A4 `super.method()` | still deferred from btorch phase A; nothing in stages 1–6 needs it |
| f16 / bf16 | revisit after f32 and int8 prove the templating holds |

---

## 9. Open questions

Recorded rather than guessed, with the measurement that would settle each.

1. **Does the protobuf reader need to be native?** Settled by profiling a
   100 MB ONNX model in stage 4 ([E9](#e9)).
2. **How much does the packed GEMM actually buy at `K=25, N=6`?** The inner
   loop is too short to vectorise; the win may be in the packing, not the
   kernel. Measure before committing to D4's scope.
3. **Can threading stay bit-identical for `nd_sum`?** Pairwise summation's tree
   shape depends on partition size. A fixed partition fixes the shape, but the
   result will differ from the single-threaded answer unless the partition is
   also used at one thread. Stage 1 must decide whether "identical across
   thread counts" includes `n=1` — see
   [`numba-dtype-architecture.md`](numba-dtype-architecture.md) §5.
4. **Does `bkit` need `arctic` at all, or just numba arrays?** A GBDT wants
   column-major access and missing-value handling, which is arctic's job. But
   the dependency is heavy for a user who already has arrays. Possibly two
   entry points.
5. **Is `tree_` one family or does `bkit` also need `hist_`?** Depends on
   whether binning is reusable outside trees.

---

## Where to resume

**Stage 1: D1 and D2a are done. D2b is next, then D3, D4.**

| item | state |
|---|---|
| **D1** dtype templating pass | ✅ done — 18 suites green, `numba_fastpath_test.b` exact, paired A/B shows no regression |
| **D2a** f32 in numba | ✅ done — `tests/numba_f32_test.b` 78/78; stores in 4 bytes, **computes in double** (N29) |
| **D4 f32 GEMM** (half of D4, pulled forward — E20) | ✅ done — `nd_matmul`/`nd_dot` preserve f32 with an **f64 accumulator**; measured 5.93e-08 at K=1024 |
| **D2c** btorch f32 plumbing | ✅ done — `set_default_dtype`, optional dtype on 9 constructors, `Module.to/float/double/dtype`, f32 `.btsd`; `tests/btorch_f32_test.b` 36/36. **MNIST MLP in f32: 97.99% against f64's 98.16%, on exactly half the memory (210.8 MB vs 421.0 MB)** |
| **D3** deterministic threading | **next** — the largest multiplier, and `bkit` wants it too (E21). Must ship the pool as a *reusable primitive* |
| **stage 2 + stage 3 (`bkit`)** | after D3, moved earlier than originally planned (E21) |
| **D4** packed GEMM (skinny shapes) | deferred — helps convolution only |
| **D2b** generic ufunc functors | deferred — helps f32 element-wise only; its benefit was gated behind D4 (E20) |

Three things f32 falsified, each of which would have been silent, and each now
pinned by a named assertion rather than by luck:

- **N28** — a bare Bantu number is an f64 array, so `f32 + 0.5` returned f64.
  One scalar multiply undid the dtype. Fixed with weak scalars.
- **N30** — three kernels encoded "not f64 therefore i64", so `nd_cumsum` on
  f32 returned **i64** and truncated 1116.8 to 1117.
- **N31** — linear algebra still returns f64, which is why D2c is *blocked*
  rather than merely pending.

Read in this order:
1. this file (you are here)
2. [`numba-dtype-architecture.md`](numba-dtype-architecture.md) — §2.4 records what implementation changed about the plan
3. [`../numba-suite/ROADMAP.md`](../numba-suite/ROADMAP.md) — D1–D5 tables and the measured results
4. [`../numba-suite/DECISIONS.md`](../numba-suite/DECISIONS.md) — N23–N31
