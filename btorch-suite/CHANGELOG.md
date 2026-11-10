# btorch — Changelog

What each phase actually built, measured and found. Numbers are from this
machine (Apple silicon, `./bantu` v1.3.2) unless stated.

---

## Phase 0 — Tracking docs ✅

- `docs/btorch-architecture.md` — the design: the Bantu/native split, the tape,
  the broadcast-gradient rule, the op table, the module system, the six test
  tiers, per-platform performance targets, and the deferred list.
- `btorch-suite/ROADMAP.md` — eleven phases, each with its gates.
- `btorch-suite/DECISIONS.md` — B1-B11.

### Measured foundation

Taken before designing anything, so the design rests on numbers rather than
expectations:

| what | measured |
|---|---|
| `nd_matmul` 128×784×256 | 6.8 ms, **7.56 GFLOP/s** |
| `nd_add(arr,arr)` 1M | 1.76 ms, **14.3 GB/s** |
| `nd_maximum(arr,arr)` 1M | 1.54 ms, **16.3 GB/s** |
| `nd_exp` 1M | 6.45 ms (libm-bound) |
| `nd_tanh` 1M | 8.25 ms (libm-bound) |
| Bantu function call | **1.78 µs** |
| `new` + method call | **3.85 µs** (the cost of one tape node) |
| fixed cost of an `nd_*` call | **~5 µs** at size 1 |

### Language facts probed, not assumed

- **Closures capture by reference.** Three closures over a loop variable
  returned `102,102,102`, not `100,101,102`. Python's semantics; reshaped the
  whole autograd design into factory-built closures (**B3**).
- Closures survive storage in an object field and capture `NdArray` handles
  correctly — the tape depends on this and it works.
- Single inheritance via `extends` works, **including virtual dispatch**: a
  parent method calling `this.name()` reaches the child's override.
- `nd_*` operator dispatch works on handles, and writes through a view land in
  the base buffer.

### Four defects found while probing

Each affects code already shipped. Full analysis in
`docs/btorch-architecture.md` §3; fixes scheduled as phase A.

**D1 — the tier-1 splat kernel is dead code (4.5× on every scalar op).**
`tier1`/`tier1r` are defined at `ndarray_kernels.hpp:206-215` with a comment
explaining why they pay off. A tree-wide search finds **zero call sites** —
only the two definitions, against ten for `tier0`. `applyBinary` therefore goes
tier 0 → tier 2, and any operand whose shape differs (i.e. **every scalar**)
walks the general strided odometer with a `loadD`/`storeD` per element:

| call | measured | path |
|---|---|---|
| `nd_add($a,$b)` | 1.76 ms / 14.3 GB/s | tier 0 |
| `nd_add($a,1.0)` | **7.86 ms / 3.2 GB/s** | tier 2 |
| `nd_maximum($a,$b)` | 1.54 ms / 16.3 GB/s | tier 0 |
| `nd_maximum($a,0.0)` | **7.38 ms / 3.4 GB/s** | tier 2 |
| `nd_multiply($a,2.0)` | **7.52 ms / 3.3 GB/s** | tier 2 |

A scalar operand does *less* memory traffic and runs 4.5× slower. It is not an
exotic path: `relu(x) = maximum(x,0)`, bias add, `grad*lr`, `p-lr*g`, the `1/n`
in every mean. `out=` does not help (218 vs 226 µs at 32k), confirming it is
the walk and not allocation. numba's ROADMAP marks "tier 1 splat" ✅ — the
kernel was written, never wired in.

**D2 — `$a + $b` on class instances silently returns 0.** No `CLASS_INSTANCE`
arm in `evalBinaryOp`, and no error: instances fall through to
`left.numberVal + right.numberVal`, and `numberVal` is 0 for an instance. The
third occurrence of a defect this file already fixed twice, for handles
(`evaluator.hpp:3818`) and lists (`:3840`). Affects `arctic.Series`,
`DataFrame` and `pillow.Image` today.

**D3 — a three-level `super()` chain segfaults, silently.**
`Base→Child→GrandChild` with chained `super()` exits **139 with no output at
all** — no error, no message, no partial output. Two levels are fine. Root
cause from the code: the handler reads `currentClassName_` to resolve the
parent, then runs the parent constructor via `runStatements(fn->body)` without
setting it — every assignment to that member is at `evaluator.hpp:6157, 6176,
6281, 6288`, none in the `super()` path. So `super()` inside `Child.init` still
resolves `GrandChild`, finds parent `Child`, and calls `Child.init` again,
unboundedly. `nn.Module` hierarchies are naturally three deep.

**D4 — `super.method()` lexes, parses, then fails.** `parser.hpp:729` emits
`VariableNode("super")` → `Undefined variable: super`. Only `super(...)` is
wired.

### Five missing primitives

`nd_index_add` (scatter-**add**: `nd_put` overwrites, and duplicate indices
must accumulate — the backward of `take`/`gather`/indexing and of every
`Embedding`), `nd_concat`/`nd_split`, `nd_pad`, `nd_repeat`/`nd_tile`, and
`nd_im2col`/`nd_col2im`. Each is composable in Bantu but only at one
interpreter round-trip per slice, which for `im2col` costs more than the
convolution it is meant to accelerate.

Noted while reading: `nd_put` materializes the destination's entire flat index
list on **every call** (`ndarray_sort_reg.hpp:415-417`), making it O(n) in the
array rather than O(k) in the indices — a scatter of 10 values into a 1M tensor
walks 1M offsets. `nd_index_add` will not repeat that.

---

## Phase A — Adoption plumbing (A1-A3, plus four more defects found along the way)

All measured on this machine (Apple silicon, `bantu-src/compiler/build/bantu`).
Every number below is from a run, not an estimate.

### A1 — the tier-1 splat kernel, wired in

`tier1`/`tier1r` were defined at `ndarray_kernels.hpp:206-215` and called from
nowhere. Wired into `applyBinary` for a size-1 operand against a contiguous
other side, both operand orders, f64 and i64, compare and logic variants.

| 1M f64 | before | after | change |
|---|---|---|---|
| `nd_add($a, 1.0)` | 7.86 ms / 3.2 GB/s | **1.30 ms / 19.4 GB/s** | **5.9×** |
| `nd_multiply($a, 2.0)` | 7.52 ms / 3.3 GB/s | **1.28 ms / 19.7 GB/s** | **5.7×** |
| `nd_maximum($a, 0.0)` (relu) | 7.38 ms / 3.4 GB/s | **1.32 ms / 19.1 GB/s** | **5.6×** |
| `nd_add($a, $onearray)` | 7.46 ms / 3.4 GB/s | **1.14 ms / 22.1 GB/s** | **6.5×** |

The scalar path is now **faster than array-array** (19.4 vs 13.2 GB/s), which is
what it should always have been: it reads one array instead of two. The gate was
"≥ 10 GB/s, within 1.3× of array-array"; it beats array-array outright.

**The tier-0 path was proven undisturbed with a control.** A single-shot reading
suggested array-array had regressed 1.72 → 1.90 ms, which would have been
outside the ±1.97% noise floor this project already documents. Best-of-5, paired,
three times: new 1.56/1.62/1.56 against old 1.70/1.58/1.56 — indistinguishable.
The apparent regression was measurement noise, and saying so required the control
rather than the reading.

`tests/numba_splat_test.b` **66/66**. Every case computes the result twice — once
with a scalar (the new path) and once with a full array of that scalar (tier 0,
untouched) — and asserts they are **exactly** equal, not merely close: they are
the same IEEE operation on the same bits. Covers both operand orders, i64
exactness, promotion, 0-d and `[1]` operands, `out=` and its gates, in-place
aliasing, ±inf/NaN/denormal edge values, and 1M-scale agreement. The comparison
is NaN-aware (`equal OR both-NaN`), because an exact comparison alone would call
two identical NaN results different.

Critically it also asserts tier 1 does **not** fire where it would be wrong:
strided, transposed and negative-stride views all still take tier 2 and agree
with their contiguous copies. Tier 1 reads a raw pointer, so firing on a
non-contiguous array would read the wrong elements.

### A2 — operator dispatch for class instances

`__add__ __sub__ __mul__ __truediv__ __mod__ __neg__`, the reflected
`__radd__ __rsub__ __rmul__ __rtruediv__ __rmod__`, and comparisons
`__lt__ __le__ __gt__ __ge__ __eq__ __ne__`. A comparison's reflection is the
*swapped* operator, as in Python: `0 < $t` asks `$t` the same question as
`$t > 0`, so there is no `__rlt__`.

A missing dunder now **raises**, naming the class and the method to add, instead
of returning 0. Two behaviours were deliberately preserved and are asserted:
`$obj + "str"` still concatenates (so printing keeps working for a class with no
`__add__`), and `==`/`!=` still compare identity unless the class defines
`__eq__`/`__ne__`.

`tests/lang_operator_overload_test.b` **64/64**, including that handles, lists,
strings, `&&`/`||` and plain numbers are all unchanged.

### A3 — the three-level `super()` segfault

`currentClassName_` is now saved, set to the parent's name, and restored around
the parent constructor — including on the **exception** path, so a constructor
that throws cannot leave the interpreter resolving `super` against the wrong
class for the rest of the process.

`tests/lang_inherit_test.b` **40/40**: 2-, 3-, 4- and 8-level chains; each
constructor proven to run **exactly once** via a trace list (running twice, which
the old recursion would have done, and running zero times both show up);
virtual dispatch reaching the deepest override from a method defined on the
base; a sibling hierarchy built afterwards still resolving correctly, which is
the test that the name was really restored; and 2,000 instantiations of a
4-level class.

### AX1 — a constructor was inherited from only ONE level up

Found by the 3-level no-constructor case in the suite above, and **pre-existing**
(verified identical on `./bantu`):

```bantu
class Q0 { def init($n) { this.n = $n; } }
class Q1 extends Q0 { def extra() { return 42; } }   // no ctor
class Q2 extends Q1 { }                              // no ctor
(new Q2(12)).n                                       // -> null, silently
```

`instantiateClass` read `parentClass->methods.find("__constructor__")` — the
immediate parent's **own** table — so a constructor two or more levels up was
never found. Nothing ran and every field stayed null, with no error. `Q1` worked
and `Q2` did not, which is the signature of a one-level lookup.

Fixed by walking the chain for the class that *defines* the constructor and
running it under **that** class's name. Using the immediate parent's name would
have reintroduced A3's recursion by a different route — `Q0`'s constructor
running under the name `"Q1"` would find parent `Q0` and call itself. The suite
covers an inherited constructor that itself calls `super()`, two no-ctor levels
deep.

### AX2 — `class A extends A` was a use-after-free, then a hang

```bantu
class A { def init($n) { this.n = $n; } }
class A extends A { }
new A(1);            // shipped build: SIGSEGV, exit 139, no output
```

`evalClassDecl` bound the class's own name into `classRegistry_` **before**
resolving `extends`, so the parent lookup found the class being declared:
`parentClass == classDef`, a self-referential cycle. `getMethod` and
`isInstanceOf` both walk that chain with no termination condition but a null
parent, so the next inherited lookup recursed until the stack died.

Three fixes, smallest first:
1. Resolve `extends`/`implements` **before** binding the name. The cycle is then
   unreachable by construction, and a redefinition cleanly extends the *previous*
   definition of that name — which is the only coherent reading of it. A first
   definition naming itself still finds nothing and warns, exactly as before.
2. Reject, by name, a definition whose ancestry reaches back to itself. Checked
   once per class declaration, never on a method call.
3. Depth-bound every walk over these raw pointers (`kMaxInheritWalk = 256` in
   `class.hpp`, applied to `getMethod`, `isInstanceOf` and the new constructor
   walk), so a cycle arriving by any future route is a diagnosable error rather
   than a stack overflow or an infinite loop.

Worth recording that the first version of fix (1) alone turned the **SIGSEGV
into a hang** — the new bounded constructor walk looped forever on the cycle
instead of recursing off the stack. Both are unacceptable; that is why (2) and
(3) are there too.

### AX3 — every `ClassDefinition` leaked

A bare `new ClassDefinition(...)` with no `delete` anywhere. Now arena-owned for
the interpreter's life. Deliberately **not** freed on redefinition, because the
raw `parentClass` back-pointers held by subclasses would dangle — see
**B12** for why a bounded leak was the safer of the two bugs, and why `shared_ptr`
is recorded as future work rather than done here.

### AX4 — `$x = push($x, v)` was quadratic

| 20,000 appends | before | after |
|---|---|---|
| `$l = push($l, $i);` | **12,781 ms** | **29 ms** (441×) |
| `push($l, $i);` | 22 ms | 20 ms |

Scaling before: 632 / 2,400 / 12,781 ms for 5k / 10k / 20k — quadratic. After:
7 / 23 / 29 ms — linear, and matching the bare form.

The earlier fix for this defect covered only the discarded-result form, but
`$x = push($x, v)` is the form the README and docs teach, so **the quadratic path
was the documented one**. Recognised at parse time so it costs nothing on the
hot path; shadow-safe; `pop`/`remove` excluded. Rationale and the exclusions in
**B13**. `tests/lang_list_selfmutate_test.b` **39/39**.

This is also why btorch's tape is not quadratic: `backward` accumulates its
topological order into a list.

### Regression

63 of 67 `tests/*_test.b` suites ALL GREEN on the new binary. The four others
were verified **identical on the pre-change binary**, so none is caused by this
work: `pillow_geometry_test` and `pillow_resample_test` (differential against
real Pillow, from the in-progress pillow P8 work), `one_taylor_privacy_test`
(needs `BANTU_SODIUM=1`), and `lang_test` (a flaky detection in the harness — it
reports 80/0 ALL GREEN when run directly).

A harness note that cost real time: `timeout`/`gtimeout` do not exist on this
machine, so a loop wrapping each suite in `timeout 300` reported **every** suite
as failing, including ones that had just passed by hand.

### Memory safety — ASan + UBSan

Built separately in the scratchpad (`-fsanitize=address,undefined`,
`-fno-sanitize-recover=undefined`); the production `build-mac.sh` was not
touched. Run over every `tests/*_test.b` suite.

**Clean.** No use-after-free, no buffer overflow, no undefined behaviour — in
particular across the new tier-1 kernel (which reads raw pointers), the instance
operator dispatch, the class arena, and the parse-time assignment change. The
six new suites all report the same assertion counts under the sanitizer as under
the normal build:

| suite | result under ASan+UBSan |
|---|---|
| `btorch_autograd_test` | 121/121, clean |
| `btorch_nn_test` | 111/111, clean |
| `numba_splat_test` | 66/66, clean |
| `lang_operator_overload_test` | 64/64, clean |
| `lang_inherit_test` | 40/40, clean |
| `lang_list_selfmutate_test` | 39/39, clean |

One suite fails **only** under the sanitizer: `lang_depth_test`, which exists to
drive recursion into the 1000-call guard. ASan's instrumented frames are several
times larger, so the native stack goes before the guard is reached — ASan
detecting that is ASan working as designed. The normal build fires the guard at
exactly depth 1000 and passes 21/21. It did, however, show the margin is not
large; recorded as **B14** with the Windows 1 MB stack caveat.

---

## Phase 1 — Tensor and the tape ✅

`btorch/btorch.b`. `tests/btorch_autograd_test.b` **121/121**, of which **33 are
gradchecks**; `tests/btorch_stress.sh` **10/10**; ASan+UBSan clean.

Highlights of what the suite actually pins:

- **The diamond.** `$y = $x.mul($x).add($x)` → `dy/dx = 2x+1`, and
  `x³ + x²` → `3x² + 2x`. A tensor used twice must receive both contributions
  before it propagates either; this is the smallest graph that separates a
  correct engine from one that merely looks correct.
- **B3, the by-reference capture trap.** Ten nodes built in a loop, all ten
  gradients asserted distinct (1..10). Had the backward closures been written
  inline instead of produced by factories, every node would have reported the
  last node's gradient — silently.
- **Broadcasting.** A `[3]` bias against a `[4,3]` activation gets a `[3]`
  gradient that is the **sum** over the four rows, not a reshape or a slice. A
  stretched size-1 axis keeps its shape and sums its stretched extent.
- **A 2,000-node chain backwards** in the feature suite and 5,000 in the stress
  suite. The toposort is iterative with an explicit stack because the
  interpreter caps recursion at 1000 calls — a recursive DFS would have failed
  during training, not during a unit test.
- **Stability.** `sigmoid` is built as `0.5(1+tanh(x/2))`, so `sigmoid(-800)` is
  0 rather than the `NaN` the `1/(1+exp(-x))` form gives once `exp` overflows.
- **gradcheck checks itself.** A deliberately doubled gradient is **rejected**,
  and the disagreement quantified at 0.5 relative against a 1e-7 tolerance. A
  checker that cannot fail checks nothing.

### Memory

| measured | result | gate |
|---|---|---|
| 10,000 forward+backward+update steps | **+424 KB** | < 4 MB |
| 20,000 tensors created and dropped | **0 KB** | < 4 MB |
| 2,000 unretained graphs | +980 KB | < 4 MB |
| 20,000 deliberately-raising calls | +56 KB | < 4 MB |
| step time, 8→4 layer at batch 16 | 0.75 ms | — |

The 10,000-step figure is the one that matters: it proves `backward()` really
drops the tape. Each step allocates a Tensor, a Node and a closure over saved
arrays per op; at ~6 ops a step, a tape that was never freed would leave 60,000
live objects and be unmistakable.

**The deep-graph gate took three attempts to state honestly.** One 5,000-node
graph peaks at ~26 MB, which is legitimate: 5,000 Nodes, ~15,000 Tensors and
5,000 saved-array closures are all genuinely alive at once. Peak RSS is
therefore not evidence of anything. Two measurements could not separate a slow
leak from an allocator high-water mark either. Three can: rounds 2-21 added
1,980 KB and rounds 22-41 added 976 KB — **halving, so it tails off**. A
per-round leak would have added ~20 MB every round and never tapered.

---

## Phases 3 and 4 — `nn.Module`, layers, losses, optimizers ✅

`btorch/nn.b` and `btorch/optim.b`. `tests/btorch_nn_test.b` **111/111**;
ASan clean.

The assertions chosen are the ones a reviewer cannot make by eye:

- **`Linear`'s initializer against the analytic variance.** A Kaiming-uniform
  bound of `sqrt(1/fan_in)` implies `Var = 1/(3·fan_in)`; asserted on a 100×400
  weight within 8%, with mean ~0 and the bound both respected and approached. A
  wrong initializer does not fail — it trains slowly and looks like a bad
  learning rate, so only a test like this catches it.
- **`parameters()` deduplicates by identity**, proven two ways: one module
  registered under two names, and one tensor registered under two names, each
  contributing once. Returned twice, a tied weight is stepped twice — a silent
  2× learning rate on that tensor alone.
- **Adam's bias correction at `t=1,2,3`.** At `t=1` the step is exactly `-lr`
  *regardless of gradient magnitude*; verified with gradients of 5.0 and 0.001,
  a 5000× range, both giving `-lr`. That invariance is the signature of a correct
  correction, and a broken Adam still reduces a loss (B10), so nothing here
  asserts that a loss went down.
- **Numerical stability.** `softmax`/`log_softmax`/`cross_entropy` asserted on
  logits of **1000**, which make a naive implementation `NaN`; a confident wrong
  prediction gives a large **finite** loss, not `-inf`; `bce_with_logits(-500, 1)`
  is exactly 500.
- **Dropout's eval path is the exact identity**, not merely close — inverted
  dropout scales at train time so eval needs no rescaling. A layer that gets
  this wrong silently loses accuracy the moment the model is put in eval mode.
- **`clip_grad_norm` over all parameters jointly** (`sqrt(3²+4²) = 5`), not
  per-tensor, which is a different operation and a common mistake.
- **A 3-level module hierarchy** — exactly what A3's segfault made impossible.

Nine gradchecks run through whole layers and losses, including
`cross_entropy` through a 2-layer `Sequential` differentiated with respect to
**all four** of its parameters at once.

### Three defects in btorch's own code, found by its tests

1. `$stack = pop($stack)` in the toposort — `pop` returns the **element**, not
   the list, so the stack became a Tensor and no gradient reached the leaves.
2. `type()` of a native array returns `"ndarray"` (the handle's tag), not
   `"handle"`. Every array handed in raw fell through to `nd()`, which refused
   it, so `Linear` could not initialize at all.
3. A `def Linear()` factory shadows a `class Linear`, so `new Linear(...)` inside
   the factory re-entered the **factory** and recursed to the depth guard. Hence
   the `_Linear` naming convention, which is documented in `nn.b`'s header
   because anyone adding a layer will otherwise hit it.

---

## Phase A5 — `nd_index_add`, the scatter that accumulates

`bantu-src/compiler/src/ndarray_sort_reg.hpp`, beside `nd_put`.

`nd_index_add(a, indices, values, axis?)` adds slabs into `a` along `axis`, in
place, accumulating where an index repeats. Rationale and the three decisions
that came with it: `DECISIONS.md` B15.

### Measured

| | |
|---|---|
| 1M indices into a **500k** array | **24 ms** |
| the same 1M indices into a **10M** array (20× the destination) | **21 ms** |
| 20,000 × 128 embedding gradient (2.56M element-adds, tight f64 slab path) | **7 ms** (366 M adds/s) |
| 1M adds to a **single** cell | exactly `1e6`, no drift |
| 20,000 repeated calls | **0 bytes** of live-array growth |

The first two rows are the point. The cost is flat across a 20× change in
destination size, which is the property `nd_put` does not have — it
materialises the destination's whole flat index list per call.

### One defect, in the new code, found by its own test

`vstr[0]` was read before checking that `outer` was non-empty. A 1-D
destination leaves `outer` empty, so this subscripted an empty `std::vector`.

What makes it worth recording is **why it hid**. The guard existed on the line
above (`outer.size() == 1 && astr[0] == 1`) and short-circuited correctly;
the line below tested the dtype first (`V.dtype == DType::F64 && vstr[0] == 1`)
and so only reached the bad subscript when the values were f64. A Bantu list of
whole numbers infers **i64**, which short-circuited before the subscript — so
every 1-D scatter written the obvious way passed, and only the ones whose
values came from a real f64 array crashed. The smoke test used lists. The real
suite used arrays.

Fixed by hoisting `slab1 = (outer.size() == 1)` and requiring it on both lines.

### Tests — `tests/numba_index_add_test.b`, 72/72

- **The adjoint identity**, over seven shape/axis/sign combinations:
  `⟨take(x,ix), y⟩ == ⟨x, index_add(0,ix,y)⟩`. This is the assertion that
  matters; it pins axis, strides and accumulation at once and cannot be
  satisfied by a kernel that merely agrees with itself.
- Two independent **pure-Bantu reference loops** (`ref1d`, `refAxis`) written
  in terms of `nd_get`/`nd_set` only, so a stride or axis mistake in the kernel
  has nothing to mirror.
- A **contrast pair** asserting that `nd_put` gives 3 and `nd_index_add` gives
  6 for the same three writes to one cell, so a later refactor that routes one
  through the other fails here rather than in a model.
- Strided, transposed and overlapping-buffer destinations; an overlapping
  source is **snapshotted**, not read as it changes.
- i64/bool destinations, the f64→i64 per-step rounding policy pinned
  explicitly, empty indices, length-1 axes, width-1 slabs.
- Eleven failure modes, each raising by name — including a **read-only
  broadcast** destination and a plain **list** destination, which would
  otherwise accumulate into a temporary and vanish.

---

## Phase 2 — indexing, extremes, shape ops

`btorch/btorch.b`. Everything A5 unblocked:

`take` / `index_select` / `embedding` / `one_hot` / `index_add`,
`max` / `min` / `maximum` / `minimum` / `argmax` / `argmin`,
`permute` / `squeeze` / `unsqueeze` / `movedim`, plus the matching `Tensor`
methods. `nn.Embedding` in `btorch/nn.b`.

### Decisions recorded rather than guessed
`DECISIONS.md` B15 (`bt.take` requires `axis` — no default is right for both
numba's callers and a neural net's), B16 (`max` splits the gradient on ties).

### `nll_loss` rewritten onto the native scatter
It built its one-hot with **one interpreter iteration per row of every batch**,
on the hot path of every training step — a 1024-row batch meant 1024 trips
through the evaluator before the loss was even formed. It is now one
`nd_index_add` over flat positions computed in a single vector pass, with the
range check done by two reductions instead of a per-row branch. The expensive
path (naming *which* row is out of range) now runs only when there is an error.

Verified unchanged numerically: `cross_entropy` still equals
`-(logp[0,t₀] + logp[1,t₁])/2` to 1e-14, uniform logits over 4 classes still
give exactly `log(4)`, and both out-of-range cases still raise.

### Tests — `tests/btorch_indexing_test.b`, 116/116 (19 gradchecks)

The duplicate-index cases are the main case here, not an edge case, because a
scatter that overwrites instead of accumulating is **only** visible on repeats:

- a row taken twice gets gradient **2, not 1** — the single assertion that
  fails if the backward scatter overwrites;
- two different weights on one row **add** (1+2 and 10+20), so the two
  contributions are distinguishable rather than both 1;
- indices are **snapshotted**: mutating the caller's index buffer between the
  forward and the backward cannot redirect the gradient;
- in `nn.Embedding`, token 1 appearing three times in a `[2,2]` batch gets
  gradient 3, and a token never mentioned gets none;
- `padding_idx` receives **exactly** no gradient;
- 100,000 hits on one row accumulate to exactly `1e5` with the other row at
  exactly zero.

Ties, which gradcheck cannot see (measure zero), are pinned by hand: a two-way
tie splits `0.5/0.5`, a four-way tie gives `0.25` four times.

`permute`'s backward is checked against a **non-self-inverse** permutation with
a hand-computed expected array, because applying the same permutation twice
instead of its inverse passes every symmetric test.

Stress: a 32k × 64 table with a 4096-token batch, forward **and** backward in
**37 ms**, with every row's gradient verified independently against
`nd_bincount`; 2000 forward/backward rounds growing live array bytes by
**3.2 KB**; 200 chained `permute`s returning the original shape and a gradient
of exactly ones; and 50 SGD steps through an `Embedding` leaving every row it
never looked up **bit-identical**.

### Five false failures, all one test bug
Every gradcheck whose closure drew its coefficient tensor *inside* the body
failed with a relative error of almost exactly 1.0 and a numerical gradient in
the tens of thousands. `gradcheck` evaluates `f` at `x+h` and `x-h`
separately, so a fresh random draw each call means the difference quotient
measures the draw. Recorded as `DECISIONS.md` B17, because the failure looks
exactly like a broken gradient.

### Verification

| tier | result |
|---|---|
| feature + differential + gradcheck | `btorch_indexing_test` **116/116**, `btorch_autograd_test` **121/121**, `btorch_nn_test` **111/111**, `numba_index_add_test` **72/72** |
| stress | `btorch_stress.sh` **10/10**, `numba_stress.sh` **26/0**, `gc_stress.sh` **19/0** |
| regression | **70/73** `tests/*.b` green; the three others are `const_bad.b` (a deliberate negative fixture) and the two in-progress pillow differentials |
| sanitizers | ASan + UBSan clean over all five affected suites, identical PASS counts |

---

## Phases A6-A9 — the rest of the primitives a tape cannot express

New file: `bantu-src/compiler/src/ndarray_manip_reg.hpp`. Nine builtins:
`nd_concat`, `nd_stack`, `nd_split`, `nd_pad`, `nd_repeat`, `nd_tile`,
`nd_im2col`, `nd_col2im`, `nd_conv_outsize`. Plus `checkedAdd` in
`ndarray_native.hpp`, which `nd_pad` needs — a pad of `SIZE_MAX` would wrap
the padded extent round to something small and allocate far too little for the
writes that follow.

All four families were built the way A5 was: as the **adjoint** of an existing
kernel where one exists, and tested by the adjoint identity rather than only by
example.

| | |
|---|---|
| 64×56×56 `im2col` (a ResNet block, 1.8M taps) | **18 ms** |
| the matching `col2im` | **9 ms** |
| 1000-way `concat` + its 1000-way `split` | exact round trip |
| 300 `im2col`/`col2im` round trips | **0 bytes** of live-array growth |

### Tests — `tests/numba_manip_test.b`, 142/142

**Eight adjoint identities** for `im2col`/`col2im`:
`⟨im2col(x),y⟩ == ⟨x,col2im(y)⟩` at stride 2 (no overlap), stride 1
(overlapping — the case that matters), 3×3 pad 1 (the standard conv layer),
batched and multi-channel, stride 2 non-square padded, dilated, 1×1, and
everything asymmetric at once.

This is the assertion the family rests on. A `col2im` that overwrote instead of
accumulating would produce a plausible-looking gradient that is wrong wherever
two kernel positions read the same pixel — at stride 1, nearly every pixel —
and an example-based test at `stride == kernel` never overlaps, so it cannot
see the difference.

Also: `concat(split(x)) == x` exactly, for a count, for explicit sizes and on
both axes; **im2col + one matmul == a four-loop reference convolution**;
`col2im(im2col(x))` is exactly 9× in the interior, which is the overlap count;
four independent pure-Bantu reference loops; strided, transposed and
overlapping inputs throughout; and 30 named failure modes.

### Three test bugs, worth recording for the pattern

- Two assertions compared Bantu's `str()` spelling of a shape (`[2, 2]`, with a
  space) against C++'s `shapeStr` (`[2,2]`, without). The message came from
  C++.
- One expected total was simply wrong arithmetic: the taps a 3×3 pad-1 kernel
  lands inside a 5×5 image is `4·4 + 12·6 + 9·9 = 169`, not `25·9 − 9·16`.

---

## Phase 2 — completed

`cat`, `stack`, `chunk`, `split`, `pad`, `tile`, `repeat_interleave`, `gather`,
and the matching `Tensor` methods.

`cat` and `stack` have **N parents**, and the engine already handled any arity —
`_backward` checks the returned count against the parent count. Each gradient is
a slice of the incoming one, i.e. `nd_split`, concat's adjoint, so the round
trip is exact by construction rather than by arithmetic that could drift. The
split parts are **copied** before being handed back: the engine stores a
returned gradient directly and may later `nd_add` into it, and two parents'
gradients aliasing one buffer would corrupt both.

`gather` is pure Bantu on A5 rather than a third native scatter (B20), with an
explicit range check *before* flattening — after flattening every index is
nominally in range, so an out-of-range column index would read a different row
instead of raising.

---

## Phase 5 — convolution and pooling

`conv2d`, `conv1d`, `max_pool2d`, `avg_pool2d` in `btorch/btorch.b`;
`nn.Conv2d`, `nn.Conv1d`, `nn.MaxPool2d`, `nn.AvgPool2d`, `nn.BatchNorm1d`,
`nn.BatchNorm2d`, `nn.GELU`, `nn.Softmax` in `btorch/nn.b`.

**There is no convolution-specific gradient** (B18). `im2col` is one op with one
backward — `col2im`, its verified adjoint — and the convolution is a composition
of ops gradchecked in phase 2. Pooling is `im2col` composed with `mean` or
`max`. That is the entire reason to pay for a native `im2col`.

### Tests — `tests/btorch_conv_test.b`, 119/119, 24 gradchecks

- `conv2d` against a **four-loop reference convolution** at three geometries,
  plus the hand-read case (a 2×2 all-ones kernel over 0..15 giving
  10, 14, 18, 26, …).
- Gradchecked w.r.t. **input, weight and bias at once**; at stride 2; padded;
  batched and multi-channel; dilated; and **the whole CNN end to end** —
  `conv → relu → avgpool → flatten → linear → cross_entropy`, all four inputs.
- The bias lands on the **channel** axis, asserted by giving channel 1 a bias of
  exactly 10 and measuring that channel's mean shift. Getting that axis wrong is
  invisible until training.
- `conv1d` agrees with the equivalent `conv2d` to 1e-13.
- Pooling: max over all-negative values is negative, **not 0** (B19); padded max
  pooling of negatives does not leak a 0 either; overlapping windows give an
  element both of its gradients.
- `BatchNorm1d`'s train/eval split asserted in both directions — train centres
  the batch, `eval()` switches to the running stats so the output is *not*
  re-centred, `eval()` updates nothing, `train()` starts again. The running
  stats are proven to be **buffers, not parameters**, by checking their ids
  against `parameters()`: an optimizer stepping them would corrupt the
  statistics it is meant to track.
- `GELU` matches PyTorch's tanh approximation to 1e-12 at ±1, and is ~0 at −10
  rather than `NaN`.
- **LeNet-5 builds with exactly 61,706 parameters**, forward 15 ms /
  backward+step 22 ms at batch 8, every one of its parameter tensors receives a
  gradient, and 10 steps on one batch take the loss 2.274 → 2.051.

### Two real defects found by the suite

1. **`_floor` was `nd_astype(x, "i64")`, which ROUNDS.** `setFromDouble` uses
   `std::llround`, so `_floor(2.5)` returned 3. It broke `chunk`'s piece size,
   and the same mistake in the *test's* reference convolution computed an output
   height of 4 where the answer was 3. Now `nd_floor_divide`, and `chunk` uses
   an exact integer ceiling instead of float arithmetic with a correction.
2. **`conv1d` could not be written in terms of `conv2d`** — it passes
   `[1, stride]` pairs, and `_geom` only accepted scalars. stride, padding and
   dilation now each accept a number or a `[height, width]` pair, as PyTorch's
   do.

### Verification

| tier | result |
|---|---|
| feature + differential + gradcheck | `numba_manip_test` **142/142**, `btorch_conv_test` **119/119**, and the four earlier suites unchanged at 121/111/116/72 |
| stress | `btorch_stress.sh` 10/10, `numba_stress.sh` 26/0, `gc_stress.sh` 19/0 |
| regression | **72/75** `tests/*.b` green; the three others are `const_bad.b` (a deliberate negative fixture) and the two in-progress pillow differentials |
| sanitizers | ASan + UBSan clean over all four new/changed suites, identical PASS counts |

---

## The `/usr/local/bin/bantu` lag — root-caused and fixed

Three binaries, built twelve days apart, **all reporting `v1.3.2`**. Full
reasoning in `DECISIONS.md` B21. Four changes:

- **`BANTU_BUILD_ID`**, a 12-character digest of every compiler source, printed
  by `--version` with the build date and the binary's own path; `--build-id`
  prints it alone so one binary can interrogate another.
- **`bantu doctor` step 6b** resolves `bantu` on `PATH`, asks that file for its
  id, and says plainly when it differs. A binary predating the flag is reported
  as `pre-build-id`.
- **The unbound-name error diagnoses itself**: a name in a native family
  (`nd_`, `col_`, `bp_`, `aead_`, …) now says that an unbound one usually means
  a stale binary, and points at `bantu doctor`. Keyed on *prefixes*, so new
  kernels are covered the day they are written.
- **`build-mac.sh --install`** (and `--install-to`), plus a staleness warning on
  every build whether or not it installs. The scripts used to end at
  `build/bantu` and say nothing — the mechanism by which the install fell
  behind. Mirrored into `build.sh` and `build-win.sh`.

**One defect found doing it:** `getSelfBinaryPath()` tried `/proc/self/exe` then
`/proc/curproc/file`, neither of which exists on Darwin, and returned the
literal string `"bantu"`. Everything built on it was quietly useless on the
development platform. Now `_NSGetExecutablePath` under `__APPLE__`.

`/usr/local/bin/bantu` is now build `3e1c69184daf` and runs the btorch suites
directly.

---

## Phase 6 — Data

`btorch/data.b`: `Dataset`, `TensorDataset`, `Subset`, `random_split`,
`DataLoader`. Decisions in `DECISIONS.md` B22.

| | |
|---|---|
| shuffling 60,000 indices | **8 ms** |
| one epoch, 469 batches of 128 over 60,000 × 32 | **84 ms** |
| 50 epochs of batching | 8 KB of live-array growth |

The assertion that matters is not any of the timings: **one epoch's batch
indices are a permutation of `0..n-1`**, checked over 9 dataset sizes × 6 batch
sizes × shuffle on/off, and again at 60,000 rows via a native scatter into a
count array. A loader that quietly drops or repeats samples still trains and
still looks fine.

Reproducibility is asserted as **two** separate facts, because a loader that
restarted the random stream instead of advancing it would satisfy only the
first: the same seed gives the identical batch order, *and* `reshuffle()` gives
a different one. Two reshuffles from one seed also land in the same place, which
is what makes a resumed run reproducible.

`TensorDataset.batch_of` is one `nd_take` per tensor, and a `Subset` composes
the index maps rather than going row by row — so a `DataLoader` over a `Subset`
of a `TensorDataset` still costs two native gathers per batch rather than
`batch_size` interpreter rounds. The generic row-by-row path still exists for a
custom `Dataset` that only knows how to produce one row, and the suite asserts
the two paths agree.

---

## Phase 7 — Persistence

`btorch/serialize.b` and the `.btsd` format (`docs/btorch-architecture.md`
§7.5.2, `DECISIONS.md` B23-B24): `"BTSD"` + version + header length + a JSON
header + a raw little-endian payload.

| | |
|---|---|
| LeNet-5 (61,706 params) saved | **494,564 bytes in 7 ms** |
| ...loaded | **8 ms** |
| header overhead | **916 bytes** over a 493,648-byte payload |

Round trips are **bit-identical**, not close: a 17-digit double, `1e-300`,
`1e300`, `-0.0`, and an i64 **above 2^53** all survive exactly, as do length-0
arrays, 4-d all-ones shapes, and strided views (written in logical order, so a
view comes back as it looked).

A trained LeNet-5 saved and reloaded into a freshly initialised model predicts
**identically to 1e-15**.

### Two gaps this phase exposed, both fixed rather than worked around

1. **`nd_tobytes` could not write an i64.** numba *has* an i64 dtype and offered
   nothing wider than `i32`, so an i64 array could only be serialized lossily.
   `i64`/`u64` element types added — and the integer encoder now reads `int64`
   **directly** instead of through a `double`, which was silently lossless only
   below 2^53 and invisible while no 8-byte integer element existed.
2. **`register_buffer` had no enumerator**, so `state_dict` silently omitted
   every buffer. A saved `BatchNorm` model reloaded with running mean 0 and
   variance 1: `train()` unaffected, `eval()` wrong, nothing pointing at the
   loader. `named_buffers()` added (same iterative walk and identity dedup as
   `named_parameters()`), buffers now in `state_dict`/`load_state_dict`, and
   the binary format records each entry's `kind`.

`load_state_arrays` copies **into** the existing tensors, and the suite asserts
the parameter tensor **ids** are unchanged across a load — an optimizer's state
keys on those ids, so a load that replaced the tensors would orphan every
momentum buffer. There is a test that takes three SGD steps, saves, loads, and
steps again.

### Every load failure, by name

Twelve cases, each breaking exactly one thing in a file built from a good one:
missing, empty, too short, bad magic, a version from the future, a header
length past the end, a header that is not JSON (**with the length kept right**,
or the truncation check fires first — which is how the first draft of that case
failed), a payload cut short (**TRUNCATED**, and by how many bytes), a single
flipped byte at the same length (**CORRUPT**, caught only by the SHA-256),
extra trailing bytes, an entry claiming bytes past the end, and leftover
payload no entry accounts for.

### One language fix it needed

**`extends data.Dataset` did not parse** — `extends` took a bare identifier
only, and the error pointed at the `.` with "Expected '{' to start class body".
That is the spelling every user writing a custom `Dataset` will type, and the
class registry is flat, so the qualifier carries no information. `parseClass`
now accepts a qualified name and keeps the last segment; `implements` too.

Finding it cost time for an unrelated reason worth recording: `parser.hpp` is
**header-only** (`parser.cpp` is a 22-byte stub), and the lint pass is compiled
into `main.cpp`. Recompiling `evaluator.cpp` alone left the old grammar in the
linter, so the fix appeared to do nothing.

### Verification

| tier | result |
|---|---|
| feature + differential | `btorch_data_test` **135/135** |
| regression | **74/77** `tests/*.b` green (the three others are `const_bad.b`, a deliberate negative fixture, and the two in-progress pillow differentials); `btorch_stress.sh` 10/10, `numba_stress.sh` 26/0, `gc_stress.sh` 19/0, `lang_module_stress.sh` 7/0 |
| sanitizers | ASan + UBSan clean over `btorch_data_test`, `numba_manip_test`, `btorch_conv_test`, `btorch_nn_test`, `lang_inherit_test`, identical PASS counts |

---

## Phase 10 — Package, docs, and the gate that proves the library

### The MLP gate: 98.16% (gate ≥97%)

`samples/btorch/mnist_mlp.b` — 784-256-128-10, 235,146 parameters,
Adam(lr=1e-3), batch 128, 8 epochs.

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| mean loss | 0.272 | 0.106 | 0.072 | 0.052 | 0.042 | 0.033 | 0.026 | 0.021 |
| test acc % | 95.40 | 96.77 | 97.07 | 97.51 | 97.81 | 98.15 | 97.87 | **98.16** |

≈20 s/epoch, 215 s total including the 4 s data load. Final train accuracy
99.6%, i.e. the normal mildly-overfit regime rather than something that found a
shortcut.

### The CNN gate: 99.00% (gate ≥98.5%)

`samples/btorch/mnist_cnn.b` — LeNet-5, 61,706 parameters, Adam(lr=1e-3) with
StepLR(every 3 epochs, ×0.4), batch 128, 4 epochs.

| epoch | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| mean loss | 0.310 | 0.0730 | 0.0464 | 0.0275 |
| test acc % | 97.24 | 98.44 | 98.64 | **99.00** |

≈5.1 min/epoch (308 s, 469 batches, ~620 ms/batch). Re-measured after the
phase-9 work below: **107 s/epoch** and **98.98%** over four continuous epochs.
The 0.02% is the resume caveat, not the optimization — the 99.00% run did epoch
1, stopped and resumed, which rebuilds Adam's moments from zero, while epoch
1's loss is identical across both runs (0.309945) and is the only epoch the two
computed the same way. The reported figure is the
best checkpoint **reloaded from disk**, and it matched the in-memory number
exactly.

**The script is resumable, because the first attempt was lost.** That run was
killed when the machine was switched off part-way through, and because Bantu's
stdout is block-buffered when redirected to a file, the log was empty — so from
the outside it looked like a 2½-hour stall. It was not: a subsequent probe of 50
training batches showed live array bytes constant at **2,778,432** and a steady
~620 ms/batch, no drift and no leak.

The lesson taken was not "watch it more carefully" but "make it survivable". The
run now keeps a sidecar with epochs completed and best accuracy, reloads the
checkpoint, and replays the learning-rate schedule so a resumed run is on the
curve a continuous one would have been. Adam's moments are **not** checkpointed;
they rebuild from zero, which costs a few batches of slightly-wrong step sizes,
and the file states that rather than implying a resume is exact. Saving
optimizer state is a phase-9 nicety.

### No interpreter regression, established by paired A/B

`benchmarks/results.md` records a `fib(28)` baseline of 614 ms/iter; this
machine measures 1420 ms/iter, which looks like a 2.3× regression — and is not
one. That baseline is from a v1.2.2 build of unknown provenance, and every
other benchmark in the same file came out **faster**, which is not what a
real slowdown looks like.

The decisive test is the paired control, as it was for A1: the same workload on
the pre-change binary (`./bantu`, dated before any of this work) and the new one,
back to back, best of three.

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| old binary | 485 ms | 473 ms | 471 ms |
| new binary | 481 ms | 477 ms | 477 ms |

Indistinguishable. A first measurement taken while the CNN was training showed
`fib` and the arithmetic loop 3× slow while `list push` and `dict set` came out
*faster* — the signature of CPU contention, not regression. Those numbers were
discarded rather than recorded.

### The data

`btorch/datasets.b`: `read_idx` (IDX, big-endian header), `mnist`, `as_images`,
`accuracy`. `samples/btorch/get_mnist.sh` fetches 54 MB into
`~/.bantu/datasets/mnist` and **verifies each file by SHA-256** (B25).

Verified on load, before any training: the label counts came back
`[5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]` and the test
counts `[980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]` — the
canonical MNIST distributions — the first ten labels are `5 0 4 1 9 2 1 3 1 4`,
and an ASCII render of image 0 is legibly a **5**, which is what its label
says. After `"standard"` normalisation the pixel mean is −1.3e-4 and the
standard deviation 1.00003.

Loaded in 2.2 s, 419 MB live. The read is **chunked** (B26): the obvious
read-then-convert would peak at 752 MB for MNIST, because u8 input becomes an
i64 array before the f64 copy.

### Documentation, and the ten examples that did not work

`docs/btorch.md`, 475 lines, written for users. Four teaching samples in
`samples/btorch/` (`01_autograd`, `02_linear_regression` — fits y=3x+2 and
asserts both coefficients within 0.05, `03_classifier` — two interleaving
spirals, ≥95% held-out, `04_save_load`).

`tests/btorch_samples_test.sh` runs every sample **and extracts and runs all 30
```` ```bantu ```` blocks from the documentation**. This is numba's B5
correction taken seriously, and it earned its keep immediately: **10 of the 30
blocks did not run.** None was a typo in the prose — they were snippets relying
on context the paragraphs gave a human and nothing gave an interpreter. Fixing
them improved the documentation: the convolution example now prints the shapes
it produces rather than describing them in a comment.

One of the ten was a genuine trap, found in the runner's own preamble: **a
variable shadows a module alias.** `include "btorch/datasets.b" as ds;`
followed by `$ds = something;` makes `ds.as_images(...)` fail with "it holds
null, not a function", because Bantu keeps variables, functions *and* aliases
in one namespace. Now documented in `docs/btorch.md` and the roadmap's trap
list, and the doc's own data example is named `$dset` so it does not teach the
footgun.

Result: **34/34** (30 doc blocks + 4 samples). The two gates are skipped unless
`BTORCH_RUN_GATES=1`, because they take about an hour between them.

### `btorch/package.json`

Version 1.0.0, `numba` as a dependency, all six `.b` files listed, plus
pointers to the docs, the samples and the seven test entries.

### Measured and deliberately left for phase 9

The convolution path runs at roughly **an eighth of the MLP's FLOP rate** per
MAC — the MLP does ~90M MACs per batch in 45 ms (≈2 GFLOP/s), LeNet-5 does
~160M in ~680 ms (≈0.24 GFLOP/s). Three suspects, measured but not
investigated, because phase 10's gate is accuracy and phase 9's is throughput:

1. `im2col`'s matmuls are tall-and-skinny (`[100352, 25] × [25, 6]` for the
   first layer) — memory-bound, no cache reuse.
2. `max_pool2d`'s backward goes through `max`'s generic gradient, which
   recomputes `nd_max`, `nd_equal`, `nd_sum` and `nd_divide` over the whole
   column matrix: four extra full passes per pooling layer per step. A pooling
   backward that remembered its argmax would skip all four.
3. `conv2d`'s `reshape`/`permute` chain materialises copies a fused kernel
   would not.

This is the honest statement of where btorch stands: correct, complete through
phase 10, and roughly 8× off its own achievable throughput on convolution.

---

## Phase 9 (started) — the 8× convolution gap, measured and mostly closed

Phase 10 left an honest statement: convolution ran at roughly an eighth of the
MLP's FLOP rate, with three named suspects. Reasoning from shapes turned out to
be wrong about two of them. Reasoning from a profiler was not.

| | before | after | factor |
|---|---|---|---|
| `nd_max` over a `[150528, 4]` array | **102 ms** | **1.2 ms** | **85×** |
| `nd_matmul [100352,25] × [25,6]` | 20.6 ms | 10 ms | 2.1× |
| `nd_matmul [25,100352]ᵀ × [100352,6]` | 45.8 ms | 10 ms | 4.6× |
| `MaxPool2d` forward | 117 ms | 4.6 ms | 25× |
| `MaxPool2d` forward+backward | ~279 ms | 29 ms | ~10× |
| `Conv2d(1→6)` forward+backward | 133 ms | 74 ms | 1.8× |
| **one LeNet training step, batch 128** | **620 ms** | **240 ms** | **2.6×** |
| MNIST MLP gate, 8 epochs | 215 s | 158 s | 1.36× |
| MNIST CNN gate, per epoch | 308 s | **107 s** | **2.9×** |

Rationale in `DECISIONS.md` B28 and B29. In short:

1. **The reduction chokepoint.** `innerSweep` runs once per output and `nditer`
   heap-allocated three `std::vector`s per call — ~450,000 allocations to
   perform 600,000 comparisons in one pooling layer. `nd_sum` had a private
   fast path, which is precisely why sums looked healthy while `max`, `min`,
   `prod`, `any`, `all`, `argmax`, `argmin`, `ptp`, `var`, `std`, `median` and
   the `nan*` family did not. One fix at the chokepoint repaired all of them.
2. **`nd_matmul`'s plumbing, not its kernel.** `toMat` allocated a vector,
   zero-filled it, and copied element by element through a dtype switch —
   ~45 MB of traffic and 2.5M switch dispatches to perform 15 MFLOP. Now the
   kernel reads the buffers directly when both operands are f64 and
   contiguous, plus `gemmTransA` for the transposed left operand that every
   weight gradient has.
3. **`max_pool2d` keeps its argmax.** The backward was five passes through
   `max`'s generic gradient recomputing what the forward already knew.

### What makes this safe rather than merely fast

Floating-point addition is not associative. A matmul that changed its
summation order would silently change every result in the library — the kind
of regression no test asserting `allclose` would ever catch.

In the blocked kernel, for a fixed `(i, j)`, `k` ascends monotonically across
the whole computation. So summing `k` in order in both new kernels is
**bit-for-bit** identical, and the `aik == 0.0` skip was kept so that a zero
times an infinity behaves exactly as before.

`tests/numba_fastpath_test.b`, 49 assertions, requires **exact** equality
between each fast path and the generic path it replaces — forcing the generic
path with an i64 operand, a strided view or a broadcast. It covers sixteen
reductions, NaN and ±∞ propagation, both sides of the `BS = 64` blocking
boundary, vectors, a **square** transpose (where misreading the orientation
would be silent) and a **double** transpose (contiguous again, and must not be
mistaken for one). Three hundred rounds through all three paths leak nothing.

The confirmation that mattered: re-running the MNIST MLP gate gave **98.16%**
again, with every epoch's mean loss matching the previous run to six
significant figures — 0.271603, 0.105855, 0.0716019, 0.0522984, 0.0424616,
0.0331691, 0.0262224, 0.0212818. Identical arithmetic, 1.36× faster.

### A fix that also removed a deviation from PyTorch

B16 had recorded that pooling inherited `max`'s tie-splitting rule. `nd_argmax`
picks the first maximum, so the fused pooling node sends a tie's whole gradient
to one element — which is what PyTorch's `max_pool2d` does. The case is not
exotic: a pooling window of post-ReLU zeros is a complete tie, and it happens
constantly. Faster *and* closer to the reference.

### Left for the rest of phase 9

`im2col`/`col2im` account for ~20 ms of the remaining 74, close to the memory
bound for streaming 40 MB, so little is available without fusing the gather
into the matmul. The rest is a real GEMM blocking problem for extreme aspect
ratios — `K = 25, N = 6` leaves an inner loop too short to vectorise — which
wants a packed micro-kernel, with its own gate. `f32` (B6) remains deferred.

### Deprecation — `datasets.accuracy` moves to `bkit.metrics`

- **[docs]** `ds.accuracy($logits, $labels)` is **deprecated** in favour of
  `bkit.metrics.accuracy_score($y_true, $y_pred)`, which is the ecosystem's one metrics home
  ([C5]/[E14]). It still works; nothing that runs today stops running.
- **[design]** **It is not delegated**, and the reason is [E2]'s own argument applied to this
  package. btorch is **1.0.0 and gated on MNIST accuracy**; `bkit` is `0.x` and changes every
  phase. Delegating would make the stable package depend on the churning one, so a `bkit` release
  could break a `btorch` install — *"blast radius becomes release cadence"*, which is why these are
  separate packages at all. A conditional dependency was the third option and is worse: two code
  paths for one metric.
- **[test]** The two are not even the same function — this one takes **logits** and argmaxes them,
  `accuracy_score` takes labels on both sides — so what stops them drifting is a **differential
  test** rather than a shared implementation. `tests/bkit_metrics_test.b` asserts they return
  identical numbers on a hand case and on 200 random rows of 10 classes, and that btorch's still
  refuses a label-count mismatch.
