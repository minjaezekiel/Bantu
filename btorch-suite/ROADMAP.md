# btorch — Roadmap

A PyTorch-shaped deep-learning library for Bantu. The public API is pure Bantu;
the *atoms* are numba's native `nd_*` kernels plus five new ones. Design and
rationale: [`../docs/btorch-architecture.md`](../docs/btorch-architecture.md)
and [`DECISIONS.md`](DECISIONS.md).

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done (feature **and**
gradcheck **and** stress green)

**Every phase is gated by six tiers, all green before it advances:**

| tier | proves | form |
|---|---|---|
| 1. feature | every function does what it claims, including its failure modes | `tests/btorch_*_test.b`, ending `RESULT: ALL GREEN` |
| 2. differential | the answers are *right* — PyTorch reference values embedded in the test | hand-computed small cases; closed forms for optimizers |
| 3. **gradcheck** | every gradient agrees with central differences to 1e-7 | `bt.gradcheck`, one case per op |
| 4. stress | survives size, repetition, abuse | deep graphs, 10k-step runs, RSS, adversarial shapes |
| 5. regression | nothing else broke | the whole existing suite on the same build |
| 6. sanitizers | no latent memory defect | ASan + UBSan over the full btorch suite |

Results recorded in [`CHANGELOG.md`](CHANGELOG.md). Every phase re-runs
`benchmarks/run.sh` so an interpreter-wide regression cannot hide.

**Three rules that apply to every phase.**
1. Never assert on a stringified number — `str()` gives six significant digits
   and leaks scientific notation. Use `nd_allclose` or `close()` with a tolerance.
2. Never use the C-style `for` above 100,000 iterations — `evaluator.hpp` caps
   it at `safety < 100000` and exits **silently**. Use `while`.
3. **A loss that goes down proves nothing.** Gradient descent on a wrong
   gradient still often reduces a loss. Correctness comes from tier 3.

---

## Phase 0 — Tracking docs ✅
- [x] `docs/btorch-architecture.md`
- [x] `btorch-suite/ROADMAP.md`, `DECISIONS.md`, `CHANGELOG.md`

## Phase A — Adoption plumbing — **PARTIAL** (A1-A4 ✅, A5-A9 open)

Cross-cutting interpreter work, not btorch code. A1–A3 are **defects already
shipped**, found while probing the foundation for this design; each one affects
every existing package, so each is fixed for `arctic`, `numba`, `pillow` and
`bplot` at the same time.

| Item | How | Feature test | Stress test | Status |
|---|---|---|---|---|
| **A1 ✅ [found] tier-1 splat is dead code** — `nd_add($a,1.0)` runs **4.5× slower** than `nd_add($a,$b)` at 3.2 vs 14.3 GB/s. `tier1`/`tier1r` are defined at `ndarray_kernels.hpp:206-215` and a tree-wide search finds **zero call sites**; `applyBinary` goes tier 0 → tier 2, so every scalar operand walks the strided odometer. This is `relu`, bias add, `grad*lr`, `p-lr*g` — most of a neural net | wire tier 1 into `applyBinary` for a size-1 operand against a contiguous other side, both orders, f64/i64/bool, compare and logic variants | `tests/numba_ufunc_test.b` extended: scalar op results **identical** to the array-array equivalent for every ufunc, both operand orders, every dtype pair | scalar-vs-array equivalence fuzzed over shapes/strides/dtypes; a 0-d array, a `[1]` array and a bare number all agree; **MEASURED: 3.2 → 19.4 GB/s, a 5.9x speedup**, now FASTER than array-array (13.2) because it reads one array instead of two. `tests/numba_splat_test.b` **66/66**, tier-0 path unchanged (best-of-5 paired: 1.56/1.62/1.56 vs 1.70/1.58/1.56 — indistinguishable) | [x] |
| **A2 ✅ [found] `$a + $b` on instances silently returns 0** — no `CLASS_INSTANCE` arm in `evalBinaryOp`, so instances read `numberVal` (always 0) and no error is raised. Affects `arctic.Series`, `DataFrame`, `pillow.Image` today. Third occurrence of a defect this file already fixed for handles (`:3818`) and lists (`:3840`) | dispatch `__add__ __sub__ __mul__ __truediv__ __floordiv__ __mod__ __pow__ __neg__ __matmul__ __eq__ __ne__ __lt__ __le__ __gt__ __ge__` to a method of that name; **raise a named error** when absent, never return 0 | `tests/lang_operator_overload_test.b`: every operator, both operand orders, instance⊕number, instance⊕instance, missing-dunder raises by name | mixed instance/handle/list/string/null in every position; `==`/`!=` on instances **unchanged** (identity) unless `__eq__` exists; `$obj + "str"` still concatenates. `tests/lang_operator_overload_test.b` **64/64** | [x] |
| **A3 ✅ [found] three-level `super()` segfaults** — `Base→Child→GrandChild` with chained `super()` exits **139 with no output at all**. The handler reads `currentClassName_` to resolve the parent but never sets it while running the parent ctor (assignments only at `evaluator.hpp:6157,6176,6281,6288`), so `super()` in `Child.init` re-resolves `GrandChild` → calls `Child.init` → unbounded recursion | save/set/restore `currentClassName_` around the parent constructor call | `tests/lang_inherit_test.b`: 2-, 3- and 4-level chains construct correctly; fields from every level present; virtual dispatch reaches the deepest override | 8-level chains; each constructor proven to run **exactly once** via a trace list; restore on the **exception** path; 2,000 instantiations. `tests/lang_inherit_test.b` **40/40** | [x] |
| **A4 ⏸ [found] `super.method()` lexes, parses, then fails** — `parser.hpp:729` emits `VariableNode("super")` → `Undefined variable: super` | bind `super.m()` to the parent's method with the child's `this` | `tests/lang_inherit_test.b` | an override calling `super.m()` that itself calls `super.m()` two levels up | [ ] **deferred** — btorch's own hierarchy does not need it; `super($args)` for the constructor is what `nn.Module` uses and that works |
| **A5 ✅ `nd_index_add(a, indices, values, axis?)`** — scatter **accumulate**. `nd_put` overwrites, and duplicate indices *must* add: this is the backward of `take`/`gather`/indexing and of every `Embedding` | typed accumulate loop along an axis, bounds-checked, O(indices) not O(array); the exact **adjoint of `nd_take`**, default axis and all (DECISIONS B15) | `tests/numba_index_add_test.b` **72/72** — the **adjoint identity** `⟨take(x,ix),y⟩ == ⟨x,index_add(0,ix,y)⟩` over 7 shape/axis/sign cases, two independent pure-Bantu reference loops, and a contrast pair pinning `nd_put`→3 vs `nd_index_add`→6 | **MEASURED: 1M indices cost 24 ms into a 500k array and 21 ms into a 10M array — flat across a 20× size change**, which is the property `nd_put` lacks; 2.56M-element embedding gradient in 7 ms (366 M adds/s); 1M adds to one cell exact; 20,000 calls leak 0 bytes; 11 failure modes raise by name | [x] |
| **A6 ✅ `nd_concat` / `nd_stack` / `nd_split`** | one shared `blockCopy` for all of A6-A8, with an `accumulate` flag; `nd_split` returns **views**, so splitting a batch does not copy it | `tests/numba_manip_test.b` **142/142** — `concat(split(x)) == x` exactly, for a count, for explicit sizes and on both axes; dtype promotion; a reference loop; strided and transposed inputs | 1000-way concat and its 1000-way split round-trip; length-0 and length-1 join axes; 7 named failure modes, incl. the error printing **both** shapes and the offending axis | [x] |
| **A7 ✅ `nd_pad`** | constant-value edge pad, N-d; `pads` as a number, one pair, or one pair per axis | zero pad is the **exact** identity; asymmetric pads; a reference loop; fill reaches every padded cell; dtype preserved | a pad 50× the array; **negative pad raises** and the error names `nd_slice` as the thing that does crop | [x] |
| **A8 ✅ `nd_repeat` / `nd_tile`** | element-wise and block-wise, both axis-aware | the two semantics asserted **side by side** in every case, since they are routinely confused; `repeat` with no axis flattens (NumPy's rule, and a trap) | factors 0 and 1; reps longer/shorter than the rank; a 1000× expansion; **past the allocation ceiling raises** rather than OOM | [x] |
| **A9 ✅ `nd_im2col` / `nd_col2im` / `nd_conv_outsize`** | `[N,C,H,W] → [N·OH·OW, C·KH·KW]`, row order `(n,oh,ow)` so the matmul result reshapes to NHWC with **no transpose**; `col2im` **accumulates** | **the adjoint identity** `⟨im2col(x),y⟩ == ⟨x,col2im(y)⟩` over 8 geometries incl. the overlapping, padded, dilated and asymmetric ones; and **im2col + one matmul == a four-loop reference convolution** | 64×56×56 (a ResNet block) in **18 ms**, col2im in 9 ms; `col2im(im2col(x))` is exactly 9× in the interior — the overlap count; 1×1 kernels; 7 named refusals | [x] |

All four were built the way A5 was: as the **adjoint** of an existing kernel
where one exists, tested by the adjoint identity rather than only by example.
That is what catches an accumulate written as an overwrite — a `col2im` that
overwrites produces a plausible gradient that is wrong wherever two kernel
positions overlap, which at stride 1 is almost everywhere, and an
example-based test at `stride == kernel` never overlaps and cannot see it.

A1 was first: everything else rode on it, and it was the cheapest large win
in the project.

### A-extra — four more defects, found while doing the above ✅

None of these were in the plan. Each is a **pre-existing defect in shipped
code**, each was found by probing or by a test written for something else, and
each is fixed and regression-tested. Full analysis in [`CHANGELOG.md`](CHANGELOG.md).

| Item | Symptom | Fix | Test | Status |
|---|---|---|---|---|
| **AX1 constructor inherited from only ONE level up** | `class Q0{init}` → `Q1 extends Q0` (no ctor) → `Q2 extends Q1` (no ctor): `new Q2(12).n` was **null**, silently. `instantiateClass` read `parentClass->methods.find(...)`, the immediate parent's own table, so a constructor two levels up was never found | walk the chain for the class that **defines** the constructor, and run it under **that** class's name (using the immediate parent's name would reintroduce A3's recursion by another route) | `tests/lang_inherit_test.b` — 4-level no-ctor chains, and an inherited ctor that itself calls `super()` | [x] |
| **AX2 `class A extends A` → SIGSEGV (use-after-free class graph)** | `class A{}; class A extends A{}` then `new A(1)` exited **139 with no output** on the shipped build. `evalClassDecl` registered the new name **before** resolving `extends`, so the parent lookup found the class being declared — `parentClass == classDef`, a self-cycle — and `getMethod`/`isInstanceOf` recursed forever | resolve `extends`/`implements` **before** binding the name; reject a chain that leads back to itself by name; depth-bound (`kMaxInheritWalk = 256`) every walk over these raw pointers so any residual cycle is an error, not a crash | `tests/lang_inherit_test.b`; the two cycle cases verified by hand to exit 0 | [x] |
| **AX3 every `ClassDefinition` leaked, owned by nothing** | a bare `new ClassDefinition(...)` with no `delete` anywhere, and redefining a class overwrote its registry slot | an arena (`classArena_`) owns them for the interpreter's life. **Deliberately not a `unique_ptr` in the registry**: `parentClass`/`implementsClasses` are raw back-pointers, so freeing a redefined class would turn a bounded leak into a use-after-free. Recorded as **B12** | ASan-clean across the full suite | [x] |
| **AX4 `$x = push($x, v)` was O(n²)** | **20,000 appends: 12,781 ms**, against 22 ms for the bare `push($x, v);`. The earlier fix only covered the discarded form — but the assigned form is the one the README teaches, so the quadratic path was the **documented** one | recognise the shape at parse time (`AssignNode::selfMutatingCall`): same variable, a mutator that returns the list, result unused → skip the copy and the store. Shadow-safe (a user-defined `push` still wins), and `pop`/`remove` excluded since they return the element | `tests/lang_list_selfmutate_test.b` **39/39** | [x] |

**AX4 measured:** 20,000 appends **12,781 ms → 29 ms (441x)**, and now linear
(5,000: 6 ms; 20,000: 24 ms). This is why btorch's tape is not quadratic.

## Phase 1 — `Tensor`, the tape, `backward()` ✅

`tests/btorch_autograd_test.b` **121/121** (33 of them gradchecks);
`tests/btorch_stress.sh` **10/10**; ASan+UBSan clean.

| Item | How | Feature test | Stress test | Status |
|---|---|---|---|---|
| `Tensor` ✅ | `data`/`grad`/`requires_grad`/`ctx`/`is_leaf`; constructors `tensor/zeros/ones/randn/rand/arange/eye/full/from_list`; `shape/ndim/size/dtype/item/to_list`; `print()` repr | `tests/btorch_tensor_test.b` | 200k tensors with flat RSS; 0-d and empty tensors | [ ] |
| `Node` + the tape ✅ | `name`/`parents`/`backward`; attached only when an input requires grad **and** grad mode is on | ″ | a **2,000-node chain** backwards in the feature suite, 5,000 in the stress suite — a recursive toposort could not, since the interpreter caps recursion at 1000 calls | [x] |
| **B3 factory-built closures** ✅ | every backward from a `_*Backward(...)` factory; closures capture `NdArray`, never `Tensor` | **ten nodes built in a loop, all ten gradients distinct (1..10)** — the exact shape of the by-reference capture bug | — | [x] |
| `backward()` ✅ | DFS toposort, reverse sweep, `+=` accumulation, graph freed unless `retain_graph` | diamond `$y=$x.mul($x).add($x)` → `dy/dx = 2x+1` ✅; `x^3+x^2 → 3x^2+2x` ✅; a non-scalar root without a gradient **raises by name** ✅ | **10,000 forward+backward+update steps grew RSS by 424 KB**; a 2,000-node chain backwards; a 5,000-node graph proven freed by comparing rounds 2-21 (+1,980 KB) against 22-41 (+976 KB) — tailing off, so drift not a leak | [x] |
| `_unbroadcast` ✅ | sum leading axes, then stretched axes with `keepdims` | a `[3]` bias against `[4,3]` gets a `[3]` gradient that is the **sum** over the 4 rows; a stretched size-1 axis keeps its shape | 4 broadcast gradchecks | [x] |
| `no_grad` ✅ | flag saved/restored **including on the exception path** | no tape allocated inside; **an error inside does not leave grad mode off** (asserted) | nested blocks | [x] |
| `gradcheck` ✅ | central differences, `h=1e-5`, rel tol 1e-7 | **it catches a deliberately wrong gradient** and quantifies it at 0.5 relative against a 1e-7 tolerance — the test of the test | — | [x] |
| In-place guard ✅ | `add_`/`mul_`/`zero_` legal only on leaves under `no_grad` | violation **raises by name** | — | [x] |

## Phase 2 — The op table ✅ **COMPLETE**
All of `docs/btorch-architecture.md` §5. In `btorch/btorch.b`, with the softmax
family in `btorch/nn.b`.
- [x] arithmetic `add/sub/mul/div/pow/neg`, each gradchecked
- [x] unary `exp/log/sqrt/abs/tanh/sigmoid/relu/leaky_relu`, each gradchecked
- [x] `matmul`, gradchecked; `sigmoid` built on `tanh` so `sigmoid(-800)` is 0 and not `NaN`
- [x] reductions `sum`/`mean` with `axis`/`keepdims`, gradchecked on both axes
- [x] shape ops `reshape`/`view`/`flatten`/`transpose`/`t`, gradchecked
- [x] the stable softmax family — `log_softmax`/`softmax`/`cross_entropy`/`bce_with_logits`,
      each asserted on logits of **1000** (which make a naive version `NaN`) and each gradchecked
- [x] **indexing** — `take`/`index_select`/`embedding`/`one_hot`/`index_add`, on A5.
      `axis` is **required** on `bt.take` by decision (B15). Duplicate indices
      accumulate, which is the whole point and is what the tests lead with
- [x] **`max`/`min`** — **amax semantics**: a tie **splits** the gradient evenly
      rather than routing it to the argmax (B16, with the reasoning). Plus
      elementwise `maximum`/`minimum` (tie to the left operand) and the
      non-differentiable `argmax`/`argmin`
- [x] **`permute`/`squeeze`/`unsqueeze`/`movedim`**, each gradchecked under a
      non-linearity, and `permute`'s backward checked against a
      **non-self-inverse** permutation — applying the same permutation twice
      instead of its inverse passes every symmetric test
- [x] **`cat`/`stack`/`chunk`/`split`/`pad`/`tile`/`repeat_interleave`**, on A6-A8.
      `cat` and `stack` have **N parents** and the engine already handled any
      arity; each gradient is a slice of the incoming one, i.e. `nd_split` —
      concat's adjoint — so the round trip is exact by construction
- [x] **`gather`** (per-element indexing) — written in **pure Bantu** on A5
      rather than as a `scatter_add` kernel (B20). The flat index is
      `idx*stride[dim] + Σ coord_d*stride_d` and every term is one broadcast
      `arange`, so it is a handful of vector ops over the already-native
      `take`/`index_add` pair, and there is no third scatter primitive to keep
      correct

`tests/btorch_indexing_test.b` **116/116**, 19 gradchecks. `nn.Embedding` lives
in `nn.b` and is tested there too: `[B,L]` ids give `[B,L,dim]`, the batched
gather is proven element-for-element identical to the flat one reshaped, and
`padding_idx` receives exactly no gradient.

`nll_loss` was rewritten onto the native scatter in the same pass — it had been
running one interpreter iteration per row of every batch, on the hot path of
every training step.

## Phase 3 — `nn.Module` and layers ✅ (`BatchNorm1d` and `GELU` outstanding)

`btorch/nn.b`. `tests/btorch_nn_test.b` **111/111** (shared with phase 4); ASan clean.
- [x] `Module`: `parameters()` **deduplicated by identity** — proven two ways, a module registered
      under two names and one tensor registered under two names, each contributing once;
      `named_parameters` with dotted paths, `zero_grad`, `train`/`eval` (both recursive and
      **iterative**, so a deep model cannot hit the 1000-call recursion cap), `state_dict`
      round-trip with named errors for a missing key and a wrong shape
- [x] `Linear` — Kaiming-uniform; **variance asserted against the analytic `1/(3·fan_in)` within 8%**,
      mean ~0, and the bound both respected and approached
- [x] `Sequential` (children keyed `"0"`,`"1"`, so state_dict keys read like PyTorch's)
- [x] `ReLU`/`Sigmoid`/`Tanh`/`LeakyReLU`/`Flatten`
- [x] `Dropout` — inverted, so **eval is the exact identity** (asserted), mean preserved, `p=1` refused
- [x] `LayerNorm` — each row asserted to mean 0 / variance 1 independently; gradchecked
- [ ] `BatchNorm1d` (running stats as buffers; train≠eval asserted)
- [ ] `GELU`, `Softmax` as a layer object
- [x] losses: `mse`/`l1`/`huber`/`cross_entropy`/`nll`/`bce`/`bce_with_logits`,
      `reduction` ∈ `mean|sum|none`, every one gradchecked, out-of-range target classes raising by name

**Naming convention, worth knowing before adding a layer:** a `def Linear()`
factory and a `class Linear` collide — the function shadows the class, so
`new Linear(...)` inside the factory re-enters the *factory* and recurses until
the depth guard fires. Concrete layers are therefore declared `_Linear`,
`_Sequential`, … with a same-named factory as the public API. `Module` and
`Optimizer` keep bare names because users extend them and never call them.

## Phase 4 — Optimizers and schedulers ✅ (`ReduceLROnPlateau` outstanding)

`btorch/optim.b`. Covered by `tests/btorch_nn_test.b` **111/111**.
- [x] losses — see phase 3
- [x] `SGD` (momentum, Nesterov, weight decay), `Adam`, `AdamW`, `RMSprop`, `Adagrad`,
      each asserted against its **closed form**, not against a falling loss (B10)
- [x] **Adam's bias correction asserted at `t=1,2,3`**: at `t=1` the step is exactly `-lr`
      *regardless of gradient size* — verified with gradients 5.0 and 0.001, a 5000x range,
      both giving `-lr`. That identity is the signature of a correct correction
- [x] AdamW's **decoupled** decay proven to differ from Adam's coupled decay (different algorithms)
- [x] every update runs under `no_grad`: asserted to leave **no tape node** on the parameter
- [x] `clip_grad_norm` — over **all** parameters jointly (`sqrt(3²+4²)=5`), not per-tensor
- [x] `StepLR`, `ExponentialLR`, `CosineAnnealingLR` (reaches `eta_min` exactly at `T_max`)
- [ ] `ReduceLROnPlateau`

## Phase 5 — Convolution and pooling ✅
- [x] `conv2d`, `conv1d` on A9's `im2col` + `matmul`; `max_pool2d`, `avg_pool2d`
- [x] `nn.Conv2d`, `nn.Conv1d`, `nn.MaxPool2d`, `nn.AvgPool2d`
- [x] differential against a **four-loop reference convolution** at three
      geometries (stride 1 no pad, pad 1, stride 2 + pad 1), and the hand-read
      case (a 2×2 all-ones kernel on 0..15)
- [x] gradchecked w.r.t. **input, weight and bias at once**, at stride 2, padded,
      batched, multi-channel, dilated — and the whole CNN end to end
      (`conv → relu → avgpool → flatten → linear → cross_entropy`, all four inputs)

**There is no convolution-specific gradient in btorch** (B18). `im2col` is one
op with one backward — `col2im`, its adjoint, verified by the inner-product
identity in A9 — and the convolution is then a composition of ops that were
each already gradchecked in phase 2: `im2col`, `reshape`, `transpose`,
`matmul`, `add`, `permute`. Pooling is `im2col` composed with `mean` or `max`.
That is the entire reason to pay for a native `im2col`: a convolution with no
convolution-specific gradient has no convolution-specific gradient to get wrong.

`tests/btorch_conv_test.b` **119/119**, 24 gradchecks. LeNet-5 builds with
exactly **61,706** parameters and runs forward in 15 ms / backward+step in
22 ms at batch 8.

Still open here: the FLOP-rate gate (phase 9's business), `ConvTranspose2d`,
`AdaptiveAvgPool2d`, and `groups`/depthwise convolution.

## Phase 6 — Data ✅
- [x] `Dataset` (the `size()`/`get(i)` contract), `TensorDataset`, `Subset`,
      `random_split`, `DataLoader`
- [x] shuffling uses numba's own seeded stream, so a run is **reproducible**;
      same seed → same batch order, asserted — **and** `reshuffle()` gives a
      *different* order, which is the separate fact that makes shuffling worth
      anything

`btorch/data.b`. Three things decided here (B22):

- **`DataLoader` is explicit, not magic** — `num_batches()`, `batch(i)`,
  `reshuffle()`, plus `epoch()` for callers who want the list. Bantu has no
  iterator protocol, and an index loop is the form the rest of btorch already
  uses; it also makes batch boundaries testable one at a time.
- **Shuffling is `nd_argsort(nd_random_uniform([n]))`** — a genuine uniform
  permutation for one native sort, where Fisher-Yates in Bantu would be *n*
  interpreter iterations (60,000 per epoch for MNIST).
- **`random_split` takes exact counts, not fractions.** 0.8/0.2 of 1000 is
  unambiguous; of 999 it is not, and silently rounding is how a test set ends
  up with one sample in it.

The assertion that matters: **one epoch's batch indices are a permutation of
`0..n-1`** — every sample exactly once, none twice — checked over 9 dataset
sizes × 6 batch sizes × shuffle on/off, and again at 60,000 rows. A loader that
quietly drops or repeats samples still trains and still looks fine.

Measured: 60,000 indices shuffled in **8 ms**; a full epoch of 469 batches of
128 in **84 ms**; 50 epochs grow live array bytes by 8 KB.

## Phase 7 — Persistence ✅
- [x] `bt.save`/`bt.load` of a `state_dict` — the `.btsd` format, with a magic,
      a version, per-entry dtype/shape and a payload SHA-256
- [x] round-trip **exact** (bit-identical, not merely close)
- [x] a **truncated** file and a **corrupt** file raise differently, by name
- [x] `inference_mode` / `Module.eval()` interaction asserted

`btorch/serialize.b`; format and reasoning in
`docs/btorch-architecture.md` §7.5.2, decisions B23-B24.

Why binary: `nd_to_list` on a 61,706-parameter model builds 61,706 Bantu
numbers, and `str()` of a double gives six significant digits — so a text round
trip is both slow and **lossy**. The test asserts a 17-digit double, `1e-300`,
`1e300` and `-0.0` all survive exactly, and that an i64 **above 2^53** does too.

Two things this phase forced:

- **`i64` added to `nd_tobytes`/`nd_frombytes`** (A-extra). numba *has* an i64
  dtype and offered nothing wider than `i32` to write it with. The same fix
  made the integer encoder read `int64` directly instead of through a `double`
  — silently lossless only below 2^53, invisible while no 8-byte integer
  element existed.
- **`named_buffers()` on `Module`**, and buffers in `state_dict`. `register_buffer`
  existed with no way to *enumerate* what had been registered, so `state_dict`
  silently omitted every buffer — a saved BatchNorm model reloaded with default
  running statistics, which trains and scores fine right up until it does not.

`tests/btorch_data_test.b` **135/135** covers both phases. It builds a
truncated file and a corrupt one *by hand* from a good one — one by cutting
bytes off, one by flipping a byte and leaving the length alone, which only the
digest catches. A trained LeNet-5 saved and reloaded into a fresh model
predicts **identically to 1e-15**; its file is 494,564 bytes for a 493,648-byte
payload, so the header costs under 1 KB.

## Phase 8 — Sequence and attention
- [ ] `Embedding` (on A5's `nd_index_add`), `RNN`, `LSTM`, `GRU`
- [ ] `MultiheadAttention`, `TransformerEncoderLayer`, positional encoding
- [ ] gradchecked through 10 unrolled timesteps

## Phase 9 — Performance, each item on a measured wall
- [ ] profile a real training step and publish where the time goes
- [ ] `f32` dtype (**B7**) — a numba-wide change with its own accuracy gates
- [ ] threading for matmul/reductions, respecting `sua`'s detached handler threads (N18)
- [ ] fused kernels for the hot composites (`a*b+c`, the Adam update)

## Phase 10 — Package, docs, and the gate that matters
- [x] `btorch/{btorch.b, nn.b, optim.b, data.b, serialize.b, datasets.b, package.json}`
- [x] `docs/btorch.md` — 475 lines, written for users rather than maintainers
- [x] `samples/btorch/` — four teaching samples plus the two gates, and
      `get_mnist.sh`, which verifies the dataset by SHA-256 so a bad mirror
      fails at download rather than three hours into a run
- [x] **every documented example executed by CI** — `tests/btorch_samples_test.sh`
      extracts all 30 ```` ```bantu ```` blocks from `docs/btorch.md` and runs
      each one. It found **10 that did not work**, which is exactly numba's B5
      correction playing out again: the claim is worth nothing unless something
      executes it. One of the ten was a real trap worth documenting — a
      variable named `$ds` shadows an `include ... as ds` alias, because Bantu
      keeps variables, functions and aliases in one namespace.
- [x] **MNIST MLP: 98.16%** test accuracy (gate ≥97%) — `samples/btorch/mnist_mlp.b`
- [x] **MNIST CNN: 98.98%** test accuracy (gate ≥98.5%) — `samples/btorch/mnist_cnn.b`, LeNet-5

The MLP run, 784-256-128-10 with Adam(1e-3) at batch 128, 8 epochs:

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| loss | 0.272 | 0.106 | 0.072 | 0.052 | 0.042 | 0.033 | 0.026 | 0.021 |
| test | 95.40 | 96.77 | 97.07 | 97.51 | 97.81 | 98.15 | 97.87 | **98.16** |

≈20 s/epoch, 215 s total including data loading. Train accuracy 99.6%, so the
model is in the normal mildly-overfit regime rather than having found a
shortcut.

The CNN run, LeNet-5 with Adam(1e-3) and StepLR(3, ×0.4) at batch 128:

| epoch | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| loss | 0.3099 | 0.0715 | 0.0482 | 0.0307 |
| test | 97.24 | 98.31 | 98.47 | **98.98** |

≈107 s/epoch (was 308 s before the phase-9 work: **2.9×**), 456 s in total.

An earlier, *resumed* run of the same script reached 99.00%. The 0.02%
difference is the documented resume caveat, not noise and not the phase-9 work:
that run did epoch 1, stopped, and resumed — which rebuilds Adam's moments from
zero. Epoch 1's loss is **identical** across both runs (0.309945), and epoch 1
is the only epoch the two computed the same way.

Both gates are measured on the **best checkpoint, reloaded
from disk** — so the number reported is one that survived a `save` and a
`load`, not one that only ever existed in memory.

**The CNN script is resumable**, and that is not a convenience: the first
attempt at this gate was lost when the machine was switched off part-way
through. It keeps a sidecar recording epochs completed and best accuracy,
reloads the checkpoint, and **replays the learning-rate schedule** so a resumed
run sits on the same curve a continuous one would have. Adam's moments are
deliberately *not* checkpointed — they rebuild from zero, which costs a few
batches of slightly-wrong step sizes, and the file says so rather than
implying a resume is exact.

Phase 10's gate is the only one that proves the whole library. Every unit test
can pass while btorch is still unable to train anything; that gate cannot.

### Phase 9, started: the measured 8× gap, closed to ~2.6×

Phase 10 recorded that convolution ran at roughly an eighth of the MLP's FLOP
rate and named three suspects. **Profiling contradicted two of the three** —
the full reasoning and the bit-identity argument are `DECISIONS.md` B28-B29.

| | before | after |
|---|---|---|
| `nd_max` over `[150528, 4]` | **102 ms** | **1.2 ms** |
| `nd_matmul [100352,25] × [25,6]` | 20.6 ms | 10 ms |
| `nd_matmul [25,100352]ᵀ × [100352,6]` | 45.8 ms | 10 ms |
| `MaxPool2d` forward | 117 ms | **4.6 ms** |
| `MaxPool2d` forward+backward | ~279 ms | **29 ms** |
| `Conv2d(1→6)` forward+backward | 133 ms | 74 ms |
| **one LeNet step (batch 128)** | **620 ms** | **240 ms** |
| MNIST MLP, 8 epochs | 215 s | **158 s** |
| MNIST CNN, per epoch | 308 s | **107 s** |

Three changes, all in numba except the third:

1. **`innerSweep` gained the contiguous fast path.** It is called once per
   output and `nditer` heap-allocates three vectors per call, so one pooling
   layer meant ~450,000 allocations to do 600,000 comparisons. `nd_sum` had its
   own fast path, which is why sums looked fine while sixteen other reductions
   did not. One chokepoint, all of them fixed.
2. **`nd_matmul` runs on the buffers** when both operands are f64 and
   contiguous, instead of copying through `toMat`'s zero-filled `std::vector`
   and a per-element dtype switch. Plus `gemmTransA` for a transposed left
   operand — `Aᵀ · G`, every weight gradient.
3. **`max_pool2d` is one fused node** that keeps its argmax, so the backward is
   a single scatter rather than five passes through `max`'s generic gradient.
   It also now matches PyTorch on ties, which the composed version did not.

Verified by `tests/numba_fastpath_test.b` (49 assertions, **exact** equality
against the generic paths, not `allclose`) and by the MLP gate returning
98.16% with every epoch's loss matching the previous run to six figures.

**Still open in phase 9:** `im2col`/`col2im` are ~20 ms of the remaining 74,
close to the memory bound for streaming 40 MB. The rest is a genuine GEMM
blocking problem for extreme aspect ratios (`K=25, N=6` leaves an inner loop
too short to vectorise), which wants a packed micro-kernel — real GEMM
engineering, with its own gate. And `f32` (B6) is still deferred.

---

## WHERE TO RESUME

Everything needed to pick this up cold.

### Status at a glance

| phase | state |
|---|---|
| 0 docs | ✅ |
| A plumbing | ✅ A1-A3, AX1-AX4, A5-A9; A4 (`super.method()`) deferred |
| 1 Tensor + tape | ✅ 121/121 |
| 2 op table | ✅ 116/116 + 19 gradchecks |
| 3 nn.Module | ✅ 111/111 |
| 4 optimizers | ✅ — `ReduceLROnPlateau` outstanding |
| 5 conv/pool | ✅ 119/119 + 24 gradchecks |
| 6 data | ✅ 135/135 (shared with 7) |
| 7 save/load | ✅ |
| 8 RNN/attention | ⬜ every primitive exists; the layers are not written |
| 9 performance | ⚠️ **started** — LeNet step 620→240 ms, `nd_max` 85× faster, every reduction and every matmul improved; GEMM micro-kernel and f32 (B6) still open |
| 10 package + docs + MNIST | ✅ **MLP 98.16%** (gate 97%), **CNN 98.98%** (gate 98.5%); 81/81 documented examples run in CI |

### The files

| file | what |
|---|---|
| `btorch/btorch.b` | Tensor, Node, tape, `backward`, `_unbroadcast`, `no_grad`, every op + its `_*Backward` factory, creation, `gradcheck` |
| `btorch/nn.b` | `Module`, layers, initializers, losses, softmax family, `clip_grad_norm` |
| `btorch/optim.b` | `Optimizer` base, SGD/Adam/AdamW/RMSprop/Adagrad, schedulers |
| `btorch/package.json` | manifest, `main: btorch.b` |
| `tests/btorch_autograd_test.b` | phases 1-2, 121 assertions, 33 gradchecks |
| `tests/btorch_nn_test.b` | phases 3-4, 111 assertions |
| `tests/btorch_indexing_test.b` | phase 2 indexing/extremes/shape, 116 assertions, 19 gradchecks |
| `tests/numba_index_add_test.b` | the A5 kernel, 72 assertions, incl. the adjoint identity |
| `tests/numba_manip_test.b` | A6-A9 kernels, 142 assertions, incl. 8 adjoint identities |
| `tests/btorch_conv_test.b` | phases 2+5: join/gather/conv/pool/norm, 119 assertions, 24 gradchecks |
| `btorch/data.b` | `Dataset`, `TensorDataset`, `Subset`, `random_split`, `DataLoader` |
| `btorch/serialize.b` | the `.btsd` format: `save`, `load`, `save_model`, `load_model` |
| `tests/btorch_data_test.b` | phases 6+7, 135 assertions |
| `tests/numba_fastpath_test.b` | the optimized paths vs the generic ones, exact equality, 49 assertions |
| `btorch/datasets.b` | the IDX reader, MNIST, `as_images`, `accuracy` |
| `docs/btorch.md` | the user-facing documentation |
| `samples/btorch/` | 4 samples + both gates + `get_mnist.sh` |
| `tests/btorch_samples_test.sh` | runs every sample AND every doc example |
| `tests/btorch_stress.sh` | memory and abuse, 10 gates, RSS sampled externally |

### How to run everything

```sh
B=./bantu-src/compiler/build/bantu          # NOT ./bantu, which is the old binary
$B --quiet run tests/btorch_autograd_test.b         # expect 121/121 ALL GREEN
$B --quiet run tests/btorch_nn_test.b               # expect 111/111 ALL GREEN
$B --quiet run tests/btorch_indexing_test.b         # expect 116/116 ALL GREEN
$B --quiet run tests/numba_index_add_test.b         # expect  72/72  ALL GREEN
$B --quiet run tests/numba_manip_test.b             # expect 142/142 ALL GREEN
$B --quiet run tests/btorch_conv_test.b             # expect 119/119 ALL GREEN
$B --quiet run tests/btorch_data_test.b             # expect 135/135 ALL GREEN
$B --quiet run tests/numba_fastpath_test.b          # expect  49/49  ALL GREEN
BANTU=$B bash tests/btorch_samples_test.sh          # expect 34/34 ALL GREEN
bash samples/btorch/get_mnist.sh                    # 54 MB, SHA-256 verified
BTORCH_RUN_GATES=1 BANTU=$B bash tests/btorch_samples_test.sh   # + the gates (~50 min)
BANTU=$B bash tests/btorch_stress.sh                # expect 10/10 ALL GREEN
# the phase-A interpreter fixes:
for t in numba_splat_test lang_operator_overload_test lang_inherit_test lang_list_selfmutate_test; do
  $B --quiet run tests/$t.b | tail -3
done
cd bantu-src/compiler && ./build-mac.sh             # rebuild after any C++ change
cd bantu-src/compiler && ./build-mac.sh --install   # ...and install it over the one on PATH
bantu doctor                                        # says whether PATH is stale (step 6b)
bantu --version                                     # prints the build id and the binary's path
```

Only `ndarray_native.cpp` changes for a kernel-only edit, so an incremental
recompile of that one translation unit plus a relink is ~80 s against
`build-mac.sh`'s ~3 min clean rebuild:

```sh
g++ -std=c++17 -O3 -ftree-vectorize -I src -I/opt/homebrew/include -DBANTU_FFI \
    -c src/ndarray_native.cpp -o build/ndarray_native.o && \
g++ -std=c++17 -O2 build/*.o -o build/bantu \
    -L/opt/homebrew/lib -lsqlite3 -lcurl -lffi -ldl
```

### The next three things to do, in order

**Phases 0 through 10 are done.** What is left is improvement, not completion.

1. **Phase 9 (performance).** The conv path runs at ~⅛ the MLP's FLOP rate;
   the three measured suspects are listed under phase 10 above. The
   cheapest-looking one is `max_pool2d`'s backward, which recomputes four full
   passes over the column matrix through `max`'s generic gradient — a pooling
   backward that remembered its argmax would skip all four. Then `f32` (B6),
   which is a new dtype through every numba kernel, not a btorch change.
2. **Phase 8 (RNN/attention).** `RNN`, `LSTM`, `MultiheadAttention`. Every
   primitive they need exists; `cat`, `gather` and `Embedding` were the
   blockers and all three are in.
3. **Publish it.** `bantu publish btorch` puts it in the local registry
   alongside `numba` and `arctic`, which is what makes `bantu add btorch`
   work. Nothing in the library blocks this; it has not been done because
   nothing has been committed to git yet either.

Smaller and unblocked: `ReduceLROnPlateau`, `ConvTranspose2d`,
`AdaptiveAvgPool2d`, `groups`/depthwise convolution, A4 (`super.method()`).

### Traps that cost time already — read before writing Bantu here

1. **Closures capture by REFERENCE.** Three closures over a loop variable all
   see its final value. Every backward is built by a `_*Backward(...)` factory
   (B3). Inline closures in a loop give silently wrong gradients.
2. **`pop()` returns the ELEMENT; `push()` returns the list.** `$stack = pop($stack)`
   assigns a Tensor to `$stack`. This broke the toposort once. Use `pop($x);` bare.
3. **Prefer `push($x, v);` bare** over `$x = push($x, v)`. Both are linear now
   (AX4), but the bare form is also clearer about intent.
4. **A full reduction returns a 0-d ARRAY, not a number** (NumPy's convention).
   `nd_min($a, null, null) == 0` compares a handle to a number and is always
   false. Use `nd_get(nd_reshape($r, [1]), [0])`, or `.item()` on a Tensor.
   This produced four false failures in one sitting.
5. **`type()` of an array returns `"ndarray"`**, the handle's tag — not `"handle"`.
6. **A `def Foo()` factory shadows a `class Foo`** → infinite recursion. Hence
   the `_Linear` convention.
7. **Module-level `def`s are not shared across includes by bare name.** `nn.b`
   keeps its own `_isNd` rather than reaching into `btorch.b`'s.
8. **A top-level `def` is global for the rest of the file**, so a test that
   defines `def push(...)` to check shadowing must do so LAST or it poisons
   every later section.
9. **Recursion is capped at 1000 calls**, and the guard is calibrated to an
   8 MB stack. Any tree walk over a user-sized structure must be iterative with
   an explicit stack — `backward`, `named_parameters` and `_setMode` all are.
10. **`for` (C-style) silently stops above 100,000 iterations.** Use `while`.
11. **Never assert on a stringified number** — `str()` gives six significant
    digits and leaks scientific notation.
12. **A gradcheck closure must be a PURE function of its inputs.** `gradcheck`
    evaluates `f` at `x+h` and `x-h` separately, so a coefficient tensor drawn
    with `nd_random_*` *inside* the closure makes the difference quotient
    measure the random draw. Signature: numerical gradient in the tens of
    thousands, relative error almost exactly 1.0. Draw constants once, outside,
    and capture them (B17). This produced five false failures in one sitting.
13. **A list of whole numbers infers `i64`, not `f64`.** `nd([1.0, 2.0])` is
    i64 because the values are integral. This is why an f64-only bug in the A5
    kernel passed every list-based smoke test and crashed on the first real
    array — and why `btorch` runs everything through `_f64()` on the way in.
14. **`clock()` is MILLISECONDS** (`evaluator.hpp:8771`), not microseconds.
    Dividing by 1000 "to get ms" reports every timing 1000× small and turns a
    budget assertion into a no-op.
15. **`nd_astype(x, "i64")` ROUNDS, it does not truncate** — `setFromDouble`
    uses `std::llround`, so `astype(2.5)` is 3. It is not a floor. This cost
    two separate bugs in one sitting: `chunk`'s piece size, and a *test's*
    reference convolution computing an output height of 4 where the answer was
    3. Use `nd_floor_divide` when you mean a floor.
16. **Variables and functions share ONE namespace**, so a local `$take` shadows
    the `take()` function for the rest of the scope and the next call fails
    with "it holds a number, not a function". Same root cause as the `_Linear`
    convention, but for locals — and the error message is good, which is the
    only reason it cost minutes rather than an hour.
17. **A Bantu `throw "string"` arrives in `catch ($e)` as the STRING**, not as
    an object. `$e.message` is null for it, while a C++ runtime error does give
    `$e.message`. btorch raises both kinds, so a test asserting on message text
    needs a helper that handles both (`errText` in `btorch_conv_test.b`).
18. **`parser.hpp` and `evaluator.hpp` are HEADER-ONLY** — `parser.cpp` is a
    22-byte stub. A change to either must recompile **every** translation unit
    that includes it, `main.cpp` included: the lint pass lives in `main.cpp`,
    so recompiling only `evaluator.cpp` leaves the old grammar in the linter
    and the change appears to have no effect at all.
19. **`extends` now accepts a QUALIFIED name** (`extends data.Dataset`), and
    only the last segment is kept. It used to be a parse error pointing at the
    `.` — misleading, since the class registry is flat and the qualifier
    carries no information. Fixed in `parseClass`.
20. **`json.parse` does not raise on malformed input** — it returns garbage
    (`{: null}`) or `null`. Anything reading a file must validate every field
    itself rather than leaning on a parse failure.
21. **`batch_of` and friends return TENSORS, not arrays** — `.to_list()`, not
    `nd_to_list()`. The error ("expected an array (got <instance>)") is clear,
    but the two spellings are easy to mix within one assertion.
22. **A variable shadows a module ALIAS too.** `include "x.b" as ds;` followed
    by `$ds = something;` makes `ds.foo()` fail with "it holds null, not a
    function". One namespace covers variables, functions *and* aliases. Found
    by the doc-example runner, in the runner's own preamble.
23. **`bantu run <file> <arg>` treats the extra argument as another FILE.**
    There is no `argv` for a script to read; use `env("NAME")`.
24. **The builtin is `env(name)`, not `getenv`,** and it returns `""` for an
    unset variable, never null.
25. **Bantu's stdout is block-buffered when redirected to a file.** A long run
    writing to a log shows nothing until it exits or fills the buffer, and a
    crash loses everything buffered. Use a pty (`script -q /dev/null`) when you
    need progress, and do not conclude a run produced no output.
26. **`timeout`/`gtimeout` do not exist on this machine**; a test harness that
    wraps a command in `timeout` silently reports every suite as failing.

### Known-failing suites that are NOT btorch's doing

Latest full sweep: **74 of 77 `tests/*.b` green.** The three others:

- `const_bad.b` — a **deliberate negative fixture** (it reassigns a `const` and
  assigns a string to a `number`). It is *supposed* to fail.
- `pillow_geometry_test` and `pillow_resample_test` — differentials against
  real Pillow, from the in-progress pillow P8 work. Pre-existing.

Run the sweep in **two batches**: running all 75 suites in one shell loop gets
OOM-killed (exit 137).

`lang_depth_test` fails **only under ASan**, because ASan's instrumented frames
are several times larger so the native stack goes before the 1000-call guard —
the normal build fires the guard at exactly 1000 and passes 21/21. Recorded as
`DECISIONS.md` B14.

Shell suites: `btorch_stress.sh` 10/10, `numba_stress.sh` 26/0, `gc_stress.sh` 19/0,
`btorch_samples_test.sh` 81/81.

Under ASan, **timing** assertions fail because the sanitizer runs 4-8× slower.
Set `BANTU_SKIP_PERF=1` for sanitizer runs: `numba_fastpath_test` honours it and
reports those three assertions as skipped. `numba_linalg_test`'s 4 GFLOP/s gate
**now honours it too** (fixed during ecosystem stage 1) — it used to fail on
every sanitizer run and be explained away each time. Verified both ways: skipped
under ASan (68/68 green), and still firing on a normal build at 6.60 GFLOP/s.

---

## Phase 8 — Sequence layers (ecosystem stage 5)

Design: [`../docs/btorch-architecture.md`](../docs/btorch-architecture.md) §11.
Paired with `btext` ([`../btext-suite/ROADMAP.md`](../btext-suite/ROADMAP.md))
because neither is useful alone: a tokenizer with nothing to feed, or an
attention layer with no way to turn text into ids.

**Every primitive this needs already exists — verified, not assumed.**
`cat`, `chunk`, `permute`, `transpose`, `matmul`, `reshape`, `split` in
`btorch.b`; `nn.log_softmax`/`nn.softmax`, numerically stable, in `nn.b`;
`nd_where` for masking; `clip_grad_norm` in `optim.b`. **No new native kernel
and no new gradient** — B18 applied again: compose, do not add a special case.

| Item | How | Feature test | Stress test | Status |
|---|---|---|---|---|
| **8.1** `RNNCell`, `GRUCell`, `LSTMCell` | **one fused matmul for all gates**, then `chunk` — `chunk` returns views (A6), so the split is free and the backward is a `cat` | gradcheck at `T=1`; gate values against hand-computed references | — | [ ] |
| **8.2** `RNN`/`GRU`/`LSTM` wrappers | `batch_first`, `num_layers`, `bidirectional`, `dropout` between layers | output and final state shapes; bidirectional equals two runs concatenated | `T=512`: gradient norm **finite**, and `clip_grad_norm` bounds it | [ ] |
| **8.3** `MultiheadAttention` | heads via `reshape`+`permute`; scaled dot product; `nn.log_softmax` for stability | gradcheck; equals a single-head reference at `num_heads=1`; masking verified | attention over `T=1024` within the allocation ceiling | [ ] |
| **8.4** `TransformerEncoderLayer`, `TransformerEncoder` | pre-norm, residuals, feed-forward | gradcheck through a 2-layer stack | 12-layer stack | [ ] |
| **8.5** positional encodings | sinusoidal, learned, rotary | sinusoidal matches a closed form; rotary preserves norms | — | [ ] |
| **8.6** KV cache | `no_grad` only, append-in-place | cached decode equals full recompute **exactly** | 1000-step decode: memory flat | [ ] |
| **8.7** packed sequences / padding masks | — | a padded batch equals per-sequence runs | ragged batches | [ ] |

### The gradcheck wrinkle, decided in advance

**A gradcheck over a `T`-step recurrence is numerically fragile.** Central
differences on a deep composition accumulate error, and an exploding recurrence
makes the finite-difference estimate meaningless.

So gradchecks run at **`T ≤ 4` with small, well-conditioned weights** — enough
to exercise every path, because the recurrence is *the same closure at every
step*, so a correct gradient at `T=4` is correct at `T=128`. The stress tier
then runs `T=512` and asserts the gradient norm is finite and that
`clip_grad_norm` bounds it, which is the property that actually matters at
depth.

### The interpreter cost, computed before building

A Bantu call costs **1.78 µs** (B1), and a recurrence is a Bantu loop over
timesteps — the one place in btorch where per-*op* overhead could dominate:

| quantity | value |
|---|---|
| ops per timestep, fused | ~14 |
| calls at `T=128` | ~1,800 |
| interpreter overhead | **~3.2 ms per layer per forward** |

Acceptable, and inside B14's concern because the loop is iterative not
recursive. It sets the honest limit: **`T` in the hundreds, not the tens of
thousands.** A 4,096-token context would spend ~100 ms per layer in overhead
before any arithmetic. Recorded now rather than discovered later.

### Phase gate
An LSTM trained on character-level language modelling reaches a stated
perplexity, **measured from a reloaded checkpoint**. B10 applies: the gradcheck
is the real test, the training run is the integration test.

---

## Phase Q — int8 quantization (ecosystem stage 6)

Design: [`../docs/btorch-architecture.md`](../docs/btorch-architecture.md) §12.
Lives in btorch ([E5]) because it must know every layer type to swap it; a
separate package would break every time `nn.b` adds a layer.

**Why it matters here more than elsewhere:** this stack has no GPU, so int8 is
*the* inference performance story — 4× arithmetic throughput and 4× less
memory, pairing exactly with the single-static-binary deployment wedge. The
target is not a 61,706-parameter LeNet; it is an imported ResNet (`bonnx`,
stage 4) running at the edge.

**Prerequisite:** ecosystem stage 1 item D1 (the templating pass). Quantized
weights must be a **real `NdArray`** — users inspect them, save them to
`.btsd` and load them back — so this phase adds `DType::I8` properly. It is the
third dtype D1 pays for.

| Item | How | Feature test | Stress test | Status |
|---|---|---|---|---|
| **Q1** `q8_` family + `DType::I8` | `q8_quantize` (saturating), `q8_dequantize`, `q8_gemm` (**i32 accumulate** → requantize), `q8_gemm_f` (→ f32) | round-trip within one quantization step, for every scale, including saturation at both ends | **fuzz accumulators toward `int32` limits: saturation, never wrap** | [ ] |
| **Q2** observers | `MinMax`, `MovingAverageMinMax`, `Histogram` (percentile clipping) | observed range matches a reference over the same data | 1M-batch calibration, memory flat | [ ] |
| **Q3** **dynamic** quantization | weights offline, activations per batch; **no calibration data needed** | MNIST CNN dynamic-int8 accuracy recorded | — | [ ] |
| **Q4** **static** PTQ | a calibration pass, then layer swap | matches a hand-quantized reference | — | [ ] |
| **Q5** `QuantizedLinear`, `QuantizedConv2d`, `Module.quantize($scheme)` | per-output-channel **symmetric** weights; per-tensor **asymmetric** activations | a quantized module's output within tolerance of the dequantized reference | every layer type in `nn.b` covered, or refused by name | [ ] |
| **Q6** gates | — | **MNIST CNN int8 within 0.5% of the f32 result**; a quantized model saves to `.btsd` and predicts bit-identically ([C3]) | — | [ ] |

Dynamic before static, deliberately: it needs no calibration dataset, so it is
usable the day it lands and it validates the kernels before the calibration
machinery exists.

The i32 accumulator is not optional — `127 × 127 × K` overflows `int16` at
`K = 5`, and silently wrapping is a wrong answer that looks like noise.

### The honest uncertainty, stated before the work starts

**The 4× throughput figure assumes SIMD integer dot-product instructions**
(AVX-512 VNNI on x86, SDOT on ARM). Without them, int8 GEMM on a generic
scalar path may be no faster than f32, and the win would be memory bandwidth
and cache residency alone — perhaps 1.5–2×.

Whether the compiler auto-vectorises the i32 accumulation well enough is the
**first measurement of Q1**, before the rest of the phase is scoped. If it does
not, the options are intrinsics behind a feature check, or accepting the
bandwidth win and saying so. What is not acceptable is publishing a 4× claim
the measurement does not support — phase 9's rule, where two of three suspects
were wrong.

---

## WHERE TO RESUME (updated)

Phases 0–10 are **done**. Next work, in ecosystem order
([`../docs/ecosystem-architecture.md`](../docs/ecosystem-architecture.md)):

1. **not btorch** — ecosystem stage 1 (numba items D1–D4), then stage 2, 3, 4
2. **btorch phase 8** — sequence layers (ecosystem stage 5)
3. **btorch phase Q** — int8 quantization (ecosystem stage 6)

Still open from earlier phases, unblocked and small:
`ReduceLROnPlateau`, `ConvTranspose2d`, `AdaptiveAvgPool2d`, `groups`/depthwise
conv (**forced by `bonnx`'s MobileNetV2 gate in stage 4**), A4
(`super.method()` — and `bkit` K3 confirms it is still not needed),
`bantu publish btorch` to the local registry.
