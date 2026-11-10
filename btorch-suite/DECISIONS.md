# btorch — Decisions

Numbered so they can be cited from code comments, tests and the CHANGELOG.
A decision is recorded here when a different choice was genuinely available.

---

### B1 — The library is Bantu; the atoms are native
Same split as `arctic` on `col_*` and `numba` on `nd_*`. Not a compromise but
arithmetic: a Bantu function call costs **1.78 µs** (measured), so per-element
Bantu is four orders of magnitude off. One call per *tensor op* over a 32,768
element tensor is ~0.12 ns/element of overhead against a ~1.3 ns kernel floor —
about 9%. btorch adds exactly five native kernels (A5-A9), each because a tape
cannot express it at acceptable cost.

**Rejected:** writing the kernels in Bantu for purity. It would be ~1000×
slower and the result could not train anything, which fails the actual brief.

---

### B2 — Define-by-run tape, not a static graph
PyTorch's model, and the right one here: no graph object, each op attaches a
`Node` to its output. Control flow in Bantu (`if`, `while`, recursion) just
works, because the graph *is* the execution history.

**Rejected:** a static graph with a compile step. It would allow fusion (the
biggest remaining CPU win) but needs a whole IR and scheduler, and it makes
debugging a Bantu program indirect. Fusion is revisited in phase 9 as a
peephole over the tape instead.

---

### B3 — Every backward closure is built by a factory function
**Load-bearing, and discovered by probe rather than assumed.** Bantu closures
capture **by reference**: three closures built over a loop variable `$k` all
returned the *final* value (`102,102,102`, not `100,101,102`). This is Python's
semantics, not a defect — but a backward closure written inline in a loop would
capture whatever its saved tensor variable ends up as, i.e. **silently wrong
gradients**, the hardest possible bug to find.

So: every backward comes from a `_*Backward($saved...)` factory whose
parameters are bound once per invocation. A test builds ten nodes in a loop and
asserts all ten gradients differ — the exact shape of this bug.

Corollary: **factories capture `NdArray` data, never `Tensor`s.** A closure
holding a `Tensor` holding the `Node` holding the closure is a reference cycle;
raw arrays keep the graph a DAG that frees itself.

---

### B4 — No double backward, deliberately
Backward closures operate on `NdArray`s with raw `nd_*` calls, so the backward
pass is **not** itself traced. That rules out `grad`-of-`grad`, and with it
`vmap` and forward-mode AD. Making it work would mean writing every backward in
terms of `Tensor` ops — doubling the tape's cost on the common path to serve a
rare one. Recorded as a limit, not an oversight.

---

### B5 — Explicit child registration, `this.add("fc1", …)`
Bantu has no `__setattr__` hook, so attribute interception (how PyTorch
discovers submodules) is not available. The alternative was scanning instance
fields reflectively; explicit registration is three characters more typing,
makes the parameter tree inspectable, and cannot silently miss a layer.

---

### B6 — f64 now, f32 in phase 9
PyTorch defaults to f32; btorch defaults to f64 because numba's `DType` is
`{F64, I64, BOOL}` (`ndarray_native.hpp:61`) and adding a variant touches the
dispatch of all 171 kernels — a numba-wide change with its own accuracy gates,
not a btorch phase. Meanwhile f64 is *more* accurate than PyTorch's default,
which makes gradchecking at 1e-7 comfortable rather than marginal — a real
benefit while correctness is being established. Cost: ~2× the memory traffic on
every bandwidth-bound op, which is the measured majority of them.

---

### B7 — Pure graph ops; in-place only on leaves under `no_grad`
PyTorch guards in-place mutation with a version counter. Phase 1 instead makes
the graph-building ops pure and restricts `add_`/`mul_`/`zero_` and optimizer
updates to leaf tensors under `no_grad`, raising by name otherwise. That
removes the entire class of "the gradient is wrong because something overwrote
a saved tensor" bug for the price of some memory. A version counter can be
added later without an API change.

---

### B8 — Numerical stability is part of the op contract
`softmax` subtracts the row max; `log_softmax` uses log-sum-exp;
`CrossEntropyLoss` fuses `log_softmax` with NLL so no probability is ever
materialized that can underflow to zero and give `-inf`. Each is tested with an
input like `[1000, 1001]` that turns a naive implementation into `NaN`. These
are not refinements to add later: a loss that returns `NaN` on a confident
prediction is a library that cannot train.

---

### B9 — Gradients are verified twice, and the second one is the real test
Analytic (hand-worked values in the test) **and** numerical (central
differences, `h=1e-5`, rel tol 1e-7). A hand-derived gradient and a numerical
gradient are wrong in *different* ways, so agreement is evidence. This is the
differential method `pillow` used against Pillow and `numba` against NumPy;
here the independent reference is calculus itself. `gradcheck` ships as library
API, and its own test asserts **it catches a deliberately wrong gradient** —
a checker that never fails checks nothing.

---

### B10 — A falling loss is not evidence
Gradient descent on a *wrong* gradient frequently still reduces a loss, so
"it trains" cannot be a correctness gate. Correctness is tier 3 (gradcheck).
The training runs in tier 4 and phase 10 exist to catch what unit tests
structurally cannot — wrong initialization, a mis-wired module tree, an
optimizer that steps a shared weight twice — not to substitute for them.

---

### B11 — Fix A1-A4 before writing any btorch code
A1 (dead splat kernel, 4.5× on scalar ops), A2 (instance `+` silently 0),
A3 (3-level `super()` segfault) and A4 (`super.method()` unbound) are all
**pre-existing defects in shipped code**, found while probing the foundation.
Building on top of them would mean btorch's own performance numbers and class
hierarchy were shaped by bugs. Fixing them first also fixes `arctic`, `numba`,
`pillow` and `bplot`, and A1 alone is the cheapest large speedup available
anywhere in the project.

---

### B12 — `ClassDefinition`s are arena-owned, not freed on redefinition
`evalClassDecl` allocated with a bare `new` and never deleted, so every class
declaration leaked its definition. The leak is bounded by the number of `class`
statements in a program — it does not grow with instances or requests — but
nothing owned the objects at all.

They are now owned by `classArena_`, a `vector<unique_ptr<ClassDefinition>>`
that lives as long as the interpreter. **Deliberately not a `unique_ptr` in the
registry.** `ClassDefinition::parentClass` and `::implementsClasses` are raw
back-pointers held by *other* definitions, and redefining a class overwrites its
registry slot. Freeing the old definition at that moment would leave every
subclass pointing into freed memory — turning a bounded, harmless leak into a
use-after-free. The arena keeps the lifetime those raw pointers already assume
and releases everything at shutdown, so the process is clean.

Making the pointers `shared_ptr` is the deeper fix. It is recorded as future
work rather than done, because it touches every inheritance site and the
language is in production.

---

### B13 — `$x = push($x, v)` is recognised at parse time, not run time
The list mutators already edit the caller's list in place through
`resolveLValue`. When the assignment target *is* that list, the only work the
assignment adds is copying the whole list out as a return value and storing it
back over itself — which made a loop of appends quadratic. Measured: 20,000
appends took **12,781 ms**, against 22 ms for the bare `push($x, v);`. A 581×
gap between two forms the documentation presents as equivalent, and the slow one
is the form the README teaches.

Recognised in `parseExpressionStatement` (`AssignNode::selfMutatingCall`) rather
than in the evaluator, so it costs **nothing** on the hot path for every other
assignment. Deliberately narrow: it fires only when the callee is a bare name,
that name is one of `append`/`push`/`insert`/`extend`, and the first argument is
the same variable as the target.

Two exclusions matter. `pop` and `remove` return the removed **element**, not the
list, so `$x = pop($x)` is a legitimate rebinding and must not take this path.
And a **user-defined** function of the same name wins over the builtin, so the
evaluator re-checks `env_->has(name)` before skipping the store — the parser
cannot see shadowing. Both are asserted in
`tests/lang_list_selfmutate_test.b`.

Result: **12,781 ms → 29 ms (441×)**, and linear.

---

### B14 — the 1000-call depth guard is calibrated to an 8 MB stack
Not changed, but recorded, because it is a portability assumption rather than a
fact. The guard fires at exactly depth 1000 on this machine and raises
catchably, which is correct. Under ASan — whose instrumented frames are several
times larger — the native stack is exhausted *before* 1000 interpreter frames,
even with `ulimit -s 65536`.

ASan detecting that is ASan working, not a defect. But it demonstrates the
margin is not large, and `tests/lang_depth_test.b`'s own header notes Windows
defaults to a **1 MB** thread stack against 8 MB here. A 1000-frame limit may
therefore not be survivable on Windows, or on a `sua` handler thread with a
small stack.

Left alone on purpose: lowering it is a behaviour change for programs that
currently recurse near the limit, and the language is in production. The
long-term fix is a platform-conditional limit (`platform_compat.hpp` already
exists for exactly this kind of split) or a probe of remaining stack, decided
with a Windows machine actually to hand — which this work did not have.

---

### B15 — `nd_index_add` is a separate primitive from `nd_put`, and it is `nd_take`'s adjoint
`nd_put` was already there and already scatters. It was not reused, and the
reason is one line of semantics: **`nd_put` overwrites, so its last write
wins.** Scattering a gradient with it drops every repeated index. An embedding
table hits that on the first batch that mentions a token twice, and the failure
is silent — the model simply learns a slightly wrong thing, forever.

So A5 adds `nd_index_add(a, indices, values, axis?)`, which accumulates.

Three things were decided along with it:

- **It is the exact adjoint of `nd_take`,** which is also how it is tested:
  `⟨take(x,ix), y⟩ == ⟨x, index_add(0,ix,y)⟩` for random `x`, `y`, `ix` over
  seven shape/axis/sign combinations. An adjoint identity pins the axis, the
  strides and the accumulation simultaneously, and it cannot be satisfied by a
  kernel that merely agrees with itself.
- **`axis` defaults exactly as `nd_take`'s does** (the last axis, via
  `axisArg`). An adjoint that disagreed with its forward about the default
  would be a trap, not an API. `nd_put`'s flat-index convention is deliberately
  *not* followed here, because `nd_put` indexes the flattened array and this
  indexes an axis.
- **At the btorch level `axis` is required, with no default at all.** `bt.take`
  refuses to guess: numba's default is the last axis and every neural-net
  caller means axis 0, so either choice is wrong for half the callers.

Cost shape, which is the other half of why it is not built on `nd_put`:
`nd_put` materialises the destination's whole flat index list on every call
(`ndarray_sort_reg.hpp:415-417`), so a 3-index write into a 10M array costs
10M. `nd_index_add` is stride arithmetic only. Measured: the same 1M indices
scattered into a 500k array and into a 10M array cost **24 ms and 21 ms** —
flat across a 20× size change.

### B16 — `max`/`min` split the gradient on ties (amax semantics), not route it to one
PyTorch has both: `max(dim)` returns `(values, indices)` and hands the whole
gradient to the argmax; `amax` splits it evenly among the tied elements.
btorch implements the **splitting** rule and calls it `max`.

Why: splitting needs no second return value and no per-element scatter
primitive (which would be a different kernel again — `scatter_add` with an
index the shape of the source, not of a slab). It is also the symmetric answer.
Under the routing rule, `max(x, x)` has derivative 1 with respect to whichever
copy happens to be first and 0 for the other, which is an arbitrary choice
dressed up as a derivative.

Ties are a measure-zero set, so gradcheck on random input never distinguishes
the two rules — which means the tie behaviour has to be pinned by
hand-computed cases instead, and `tests/btorch_indexing_test.b` does that: a
two-way tie gives `0.5/0.5`, a four-way tie `0.25` four times, and both sum to
exactly the incoming gradient.

`maximum`/`minimum` (elementwise, two operands) use the opposite rule and give
a tie wholly to the left operand. That is deliberate: there the thing that
matters is that the two returned gradients sum to exactly the incoming one,
which a one-sided rule guarantees and a split also would, but one-sided is
cheaper and matches `relu`'s kink at 0.

### B17 — a gradcheck closure must be a pure function of its inputs
Recorded because it cost five false failures in one sitting, every one of
which looked like a broken gradient.

`gradcheck` calls `f` once for the analytic pass and then twice per element for
the central difference. Any randomness *inside* `f` — a coefficient tensor
drawn with `nd_random_normal` in the closure body — means `f(x+h)` and
`f(x-h)` are computed against **different** coefficients. The difference
quotient then measures the random draw, not the derivative, and comes back in
the tens of thousands with a relative error of almost exactly 1.0.

The fix is to draw every constant **once, outside the closure**, and capture
it. The relative error of ~1.0 is the signature: it means the numerical
estimate is enormous compared with the analytic one, which a genuinely wrong
gradient almost never produces (a wrong gradient is usually wrong by a factor,
or in a few elements, not by five orders of magnitude everywhere).

---

### B18 — there is no convolution-specific gradient
`conv2d` is not a hand-derived backward. `im2col` is given **one** op with one
backward — `col2im`, which is its adjoint and is verified by the inner-product
identity `⟨im2col(x),y⟩ == ⟨x,col2im(y)⟩` over eight geometries — and the
convolution is then a composition of ops that were each already gradchecked in
phase 2:

```
conv2d(x, w, b) = permute(reshape(matmul(im2col(x), transpose(reshape(w))) + b))
```

This is the whole reason a native `im2col` is worth paying for. The usual way
to implement convolution is to write `conv2d_backward_input` and
`conv2d_backward_weight` by hand, which means two more derivations to get
wrong, each only wrong in the overlap regions and so invisible on a toy test.
A convolution with no convolution-specific gradient has none to get wrong, and
the gradcheck that covers it is the *composition's* gradcheck, which is also
the one a user's model actually exercises.

Pooling follows the same rule: `[N,C,H,W]` folds to `[N·C,1,H,W]`, one `im2col`
makes `[N·C·OH·OW, kh·kw]` with one window per row, and the pool is then a
reduction along axis 1 — so the pooling gradient *is* `mean`'s or `max`'s
gradient, already tested.

`conv1d` is `conv2d` at height 1, for the same reason: one geometry
implementation rather than two. That is what forced stride/padding/dilation to
accept a `[height, width]` pair as well as a number — conv1d has to say
"stride 1 down, stride s across", and without the pair form it could not be
written in terms of conv2d at all.

### B19 — `max_pool2d` pads with a finite sentinel, not with zero or −∞
`nd_im2col` pads with **zero**, which is correct for `avg_pool2d` — PyTorch's
`count_include_pad` defaults to true, so a padded cell contributes a 0 to the
mean and is counted — and **wrong** for a maximum: a window of entirely
negative activations would pool to 0.

So `max_pool2d` pads first, with `bt.pad`, and then tells `im2col` not to pad.
The fill is `-1e300` rather than `-inf` so that nothing downstream can turn
`(-inf) - (-inf)` into `NaN`. `bt.pad`'s backward crops, so the padded cells
receive exactly no gradient, which is the behaviour wanted anyway.

Note that `max_pool2d` inherits B16: a tie inside a pooling window splits the
gradient rather than routing it to one element. Ties are common here — a window
of post-ReLU zeros ties completely — and splitting is a valid subgradient, but
it is not what PyTorch does, so it is asserted explicitly rather than left to
be discovered.

### B20 — `gather` is written in Bantu, not as a `scatter_add` kernel
`gather(dim, index)` is per-**element** indexing, where A5's `nd_index_add` is
per-**slab**. The obvious move is a third native scatter, `scatter_add`, with
an index the shape of the source.

It was not taken. The flat index into a C-contiguous source is

```
flat = idx·stride[dim] + Σ_{d≠dim} coord_d·stride_d
```

and every term on the right is one broadcast `arange`, so the whole thing is a
handful of vector ops over the `take`/`index_add` pair that is already native
and already tested. A third scatter primitive would be a third thing to keep
correct, and the arithmetic above is not where the risk lives.

One consequence had to be handled explicitly: after flattening, **every index
is nominally in range**, so an out-of-range column index would silently read a
different row instead of raising. `bt.gather` therefore validates against
`shape[dim]` itself, with two reductions, before it flattens anything — and the
error says why.

### B21 — a binary's identity is part of its interface
Three `bantu` binaries sat on this machine, built twelve days apart, and all
three reported `v1.3.2`. A stale install was therefore invisible, and its only
symptom was `Undefined variable: nd_<something>` from a library newer than the
interpreter running it — which reads as a typo.

Four things were changed, and the first three matter more than the install:

1. **`BANTU_BUILD_ID`** — a 12-character digest of every compiler source,
   passed by the build scripts and printed by `--version` along with the build
   date and the binary's own path. It changes exactly when behaviour can. A
   binary built before this exists reports `unknown`, which is itself the
   useful answer. `--build-id` prints it alone so one binary can ask another.
2. **`bantu doctor` step 6b** — resolves `bantu` on `PATH`, asks that file for
   its build id, and says plainly when it differs from the running one. A
   binary that does not understand `--build-id` is reported as `pre-build-id`,
   i.e. definitely the older of the two.
3. **The unbound-name error diagnoses itself.** A name matching a native family
   prefix (`nd_`, `col_`, `bp_`, `aead_`, …) now says so and points at
   `bantu doctor`. The list is of *prefixes*, so a new kernel in an existing
   family is covered the day it is written, with nothing to keep in sync.
4. **`build-mac.sh --install`**, plus a staleness warning printed by *every*
   build whether or not it installs. The scripts used to end at `build/bantu`
   and say nothing, which is the mechanism by which the installed binary fell
   twelve days behind.

A defect was found doing this: `getSelfBinaryPath()` tried `/proc/self/exe`
(Linux) and then `/proc/curproc/file` (FreeBSD), neither of which exists on
Darwin, and returned the literal string `"bantu"`. Everything built on it was
quietly useless on the platform this is developed on — `bantu doctor` printed
`Running binary: bantu`. Now uses `_NSGetExecutablePath` under `__APPLE__`.

---

### B22 — `DataLoader` is explicit; shuffling is `argsort(uniform)`; splits take counts
Three choices in `btorch/data.b`, none of them the obvious one.

**`DataLoader` exposes `num_batches()` / `batch(i)` / `reshuffle()`, not an
iterator.** Bantu has no iterator protocol, so any "for batch in loader" shape
would have to be invented — a closure taking a callback, or a stateful
`next()`/`done()` pair. An index loop needs neither, is the form the rest of
btorch already uses, and makes the batch boundaries testable one at a time.
`epoch()` returns the whole list for callers who prefer it.

**Shuffling is `nd_argsort(nd_random_uniform([n]))`.** That is a genuine uniform
permutation for one native sort. Fisher-Yates would be *n* interpreter
iterations — 60,000 per epoch for MNIST, which is the same order as the training
step itself. It also draws from numba's seeded stream, so `nd_seed(k)` makes an
epoch reproducible for free.

Two properties follow and they are **different** properties, so both are
asserted: the same seed gives the identical order, *and* `reshuffle()` gives a
different one. A loader that restarted the stream instead of advancing it would
pass the first test and make shuffling worthless.

**`random_split` takes exact counts, never fractions.** 0.8/0.2 of 1000 is
unambiguous; of 999 it is not. Rounding silently is how a test set ends up with
one sample in it, and the error message says so rather than just reporting a
sum mismatch.

The assertion that actually protects training is none of the above: it is that
**one epoch's batch indices are a permutation of `0..n-1`**. Every sample
exactly once, none twice. A loader that quietly drops or repeats samples still
trains and still looks fine, so it is checked over 9 sizes × 6 batch sizes ×
shuffle on/off, and again at 60,000 rows via a native scatter.

### B23 — `.btsd` is a text header over a binary payload, and the digest is not optional
A `state_dict` is tensors, so a text format is out: `nd_to_list` on 61,706
parameters builds 61,706 Bantu numbers, and `str()` of a double gives six
significant digits — a text round trip is both slow and **lossy**. The format is
a JSON header (names, dtypes, shapes, byte counts) over raw little-endian
`nd_tobytes` output. 494,564 bytes for a 493,648-byte payload: the header costs
under 1 KB.

**A truncated file and a corrupt file are separate failures, reported
separately.** The length check catches the first; only a payload SHA-256 catches
the second — a file with a flipped byte is exactly as long as a good one.
Collapsing them into "bad file" throws away the only diagnostic the reader has,
so the tests build both by hand from a good file and assert the two different
messages.

**A version from the future is refused, not read best-effort.** A field this
reader does not know about is a field it would silently drop, and a checkpoint
that loads 90% of itself is worse than one that refuses.

**`json.parse` in this interpreter returns garbage (`{: null}`) or `null` for
malformed input rather than raising**, so the loader cannot lean on it. Every
field is type-checked, every entry's byte range is bounds-checked against the
payload, and leftover payload bytes that no entry accounts for are an error.

### B24 — buffers are part of the state, and `register_buffer` had no enumerator
`Module` had `register_buffer` and `buffer(name)` and **no way to list what had
been registered**. `state_dict` therefore walked `named_parameters()` only and
silently omitted every buffer.

The cost of that is specific: a `BatchNorm` layer's running mean and variance
are buffers *precisely* so no optimizer steps them. A saved model reloaded
without them comes back with running mean 0 and variance 1 — so `train()` is
unaffected and `eval()` is wrong. The model trains fine, scores badly, and
nothing points at the loader.

So `named_buffers()` exists, mirroring `named_parameters()` exactly: the same
iterative walk (the 1000-call recursion cap applies here too) and the same
deduplication by tensor identity. `state_dict` includes buffers,
`load_state_dict` restores them, and the binary format records each entry's
`kind` so a reader can tell them apart.

`load_state_arrays` copies **into** the existing tensors rather than replacing
them, and the test asserts the parameter tensor *ids* are unchanged across a
load — because an optimizer's state keys on those ids, and a load that replaced
the tensors would silently orphan every momentum buffer.

---

### B25 — the MNIST dataset is fetched, verified and never committed
54 MB of binary does not belong in a repository, so `samples/btorch/get_mnist.sh`
fetches it into `~/.bantu/datasets/mnist` (overridable with `BANTU_MNIST_DIR`)
and **verifies every file by SHA-256** against the canonical digests.

The digests are not ceremony. The original `yann.lecun.com` URLs now return 403
and every mirror is someone's convenience copy; a mirror serving something
subtly different — a different byte order, a re-encoded subset — would show up
as an accuracy figure a few points low, three hours into a training run, with
nothing to point at. Failing at download costs seconds.

The loader checks the data as well: `read_idx` validates the IDX magic (and
says "is it still gzipped?" when the first bytes are `1f 8b`), the element-type
code, the dimension count, and that the file's length is *exactly* what its
declared shape needs. The MNIST wrapper then checks that the image and label
counts agree.

### B26 — `read_idx` loads in chunks, because f64 is the only float
60,000 × 784 as f64 is 376 MB — which is what B6 (f64 only) costs for a dataset
this size, and the obvious read-then-convert costs it *twice at once*: once as
the i64 array `nd_frombytes` produces from u8 input, once as the f64 copy. Peak
752 MB against a 2 GB allocation ceiling, for MNIST, which is a small dataset.

So the file is read once as bytes and converted two million elements at a time
into a preallocated f64 array through a writable slice view. Peak is the array
plus one chunk. Normalisation is then three in-place ufunc passes with `out=`
rather than three more 376 MB copies.

### B27 — the documented examples are extracted and run, and ten of them failed
numba's B5 recorded that "documented examples run in CI" had been claimed in
that project and was false: two of nine failed the first time a runner was
actually pointed at them. So `tests/btorch_samples_test.sh` extracts all 30
```` ```bantu ```` blocks from `docs/btorch.md` and runs every one.

**Ten failed.** Not one was a typo in the prose — they were snippets that
needed context the surrounding paragraphs supplied to a human reader and
nothing supplied to an interpreter. The fix was to make the examples
self-sufficient, which made them better documentation: the convolution example
now prints the shapes it produces instead of describing them in a comment, and
the `retain_grad` example creates a tensor that actually requires a gradient.

Two mechanics were needed, and both say something:

- Doc blocks are **snippets**, so the runner prepends a preamble defining the
  names the prose introduces. A block carrying its own `include "btorch/nn.b"`
  has that path rewritten to this tree, which also checks it names a real file.
- Blocks are run **from a temp directory**, so an example writing
  `"model.btsd"` does not drop a file in the repository. That required
  resolving `$BANTU` to an absolute path first — and not doing so made all 30
  blocks fail at once, which is at least a loud failure.

A block that demonstrates an **error** is marked `// EXPECT-RAISES` in the doc
and is required to fail. Without that, the honest way to document a failure
mode is to not document it.

---

### B28 — `max_pool2d` keeps its argmax, and so matches PyTorch exactly
Pooling was `max(im2col(x), 1)`: correct, composed from already-gradchecked
ops, and measured at **394 ms** per forward+backward on one LeNet layer. Seven
of those milliseconds were the forward.

The cost was `max`'s **generic** backward, which cannot know anything about
pooling. To find which element of each window won, it recomputes the maximum,
compares every element against it, counts the ties and divides — five passes
over a 150,528 × 4 column matrix, three of them broadcasting against a
stride-0 axis. The forward pass already knew the answer and threw it away.

So `max_pool2d` is now one fused node: the forward keeps `nd_argmax`, and the
backward is a single `nd_index_add` of the incoming gradient straight to the
winning positions. **394 ms → 29 ms.**

The argmax is computed only when a backward will exist (`$_BT.grad &&
requires_grad`), so inference does not pay for a gradient nobody asked for:
forward alone is **4.6 ms**, down from 117 ms.

This also *fixes* a deviation rather than introducing one. B16 noted that
pooling inherited `max`'s tie-splitting rule; `nd_argmax` picks the first
maximum, so a tie now sends the whole gradient to one element — which is what
PyTorch's `max_pool2d` does. That case is not exotic: a pooling window of
post-ReLU zeros is a complete tie, and it happens constantly.

### B29 — three fast paths, each proved bit-identical to the path it replaces
Phase 10 recorded that convolution ran at about an eighth of the MLP's FLOP
rate, and named three suspects. **Profiling contradicted two of the three.**
What the measurements actually said, per LeNet layer at batch 128:

| | before | after |
|---|---|---|
| `nd_max` over a `[150528, 4]` array | **102 ms** | **1.2 ms** |
| `nd_matmul [100352,25] × [25,6]` | 20.6 ms | 10 ms |
| `nd_matmul [25,100352]ᵀ × [100352,6]` | 45.8 ms | 10 ms |
| `MaxPool2d` forward | 117 ms | 4.6 ms |
| `MaxPool2d` forward+backward | ~279 ms | 29 ms |
| `Conv2d(1→6)` forward+backward | 133 ms | 74 ms |
| **one LeNet training step** | **620 ms** | **240 ms** |

**1. The reduction chokepoint.** `innerSweep` is called once per output, and
`nditer` heap-allocates three `std::vector`s per call — so reducing the last
axis of a `[150528, 4]` array meant 150,528 calls and ~450,000 allocations to
perform 600,000 comparisons. `nd_sum` had a fast path of its own, which is
exactly why sums looked fine while `max`, `min`, `prod`, `any`, `all`,
`argmax`, `argmin`, `ptp`, `var`, `std`, `median` and the `nan*` family did
not. Fixing the one chokepoint fixed all of them. **85× on `nd_max`.**

**2. `nd_matmul`'s plumbing, not its kernel.** The blocked i-k-j kernel is
fine. `toMat` allocated a `std::vector`, **zero-filled it**, then copied
element by element through `getAsDouble`'s dtype switch; `fromMat` copied back.
For a convolution's first layer that is ~45 MB of avoidable traffic and 2.5M
switch dispatches to perform 15 MFLOP. When both operands are already f64 and
C-contiguous — which is every array a ufunc or `im2col` just produced — the
kernel now runs on the buffers and writes into the destination.

**3. A transposed left operand.** `Aᵀ · G` is every weight gradient, and it is
the one case where materialising the transpose copies the *larger* operand: a
20 MB copy to produce a 150-element result. `gemmTransA` reads both inputs
along their contiguous axis and accumulates into the small output, which stays
in L1. A small strided operand (`Bᵀ` in a matmul backward is the weight matrix,
a few hundred doubles) is normalised with one copy instead, which is cheaper
than routing around it.

**The non-negotiable part.** Floating-point addition is not associative, so a
matmul that changed its summation order would silently change every result in
the library. In the blocked kernel, for a fixed `(i, j)`, `k` still ascends
monotonically across the whole computation — so summing `k` in order in
`gemmContig` and `gemmTransA` is **bit-for-bit** the same answer, and the
`aik == 0.0` skip is kept so that a zero times an infinity behaves as before.

`tests/numba_fastpath_test.b` (49 assertions) requires exact equality, not
`allclose`, between each fast path and the generic one it replaces — forcing
the generic path with an i64 operand, a strided view or a broadcast. It covers
sixteen reductions, NaN and ±∞, both sides of the `BS = 64` blocking boundary,
vectors, a **square** transpose (the one case where misreading the orientation
would be silent) and a **double** transpose (which is contiguous again and must
not be mistaken for one).

The proof that landed it: re-running the MNIST MLP gate afterwards gave
**98.16%** again with every epoch's mean loss matching the previous run to six
significant figures — 0.271603, 0.105855, 0.0716019, and so on. Identical
arithmetic, 1.36× faster; the CNN step 2.6× faster.

Timing assertions skip under `BANTU_SKIP_PERF=1`, because ASan runs 4–8× slower
and a budget that had to accommodate that would prove nothing. `numba_linalg_test`'s
pre-existing 4 GFLOP/s gate has the same property.

### What is still slow, and why it was left
Convolution is now ~74 ms per layer forward+backward, of which `im2col` and
`col2im` are ~20 ms — close to the memory bound for streaming 40 MB, so there
is little left there without fusing the gather into the matmul. The remaining
matmul time is a genuine blocking problem for extreme aspect ratios
(`K = 25, N = 6` leaves an inner loop too short to vectorise), which wants a
packed micro-kernel. That is real GEMM engineering and belongs in phase 9 with
its own gate, not bolted on here.

---

### B30 — a tensor may be f32 or f64, and nothing else
`_floatDtype()` is the single gate, and it refuses `i64` and `bool` **by name**
rather than letting them through: a gradient has to be able to hold a fraction,
and a model whose parameters are integers trains to integers while every test
that checks shapes still passes.

f64 remains the default forever (ecosystem E19). f32 is opt-in two ways:
`bt.set_default_dtype("f32")` for a whole program, or an **optional trailing
dtype** on each of the nine constructors. Bantu passes `null` for a missing
argument, so `bt.zeros([3], false)` still means exactly what it always did —
which is why extending nine signatures broke nothing.

One asymmetry, deliberate: with no dtype given, an **array** operand passes
through at whatever float dtype it already carries, so
`bt.tensor(f32_array, false)` stays f32. A **list** has no dtype of its own and
is built at the default. Widening a caller's f32 array back to the default
would make `bt.tensor` the place f32 quietly died.

---

### B31 — every derived buffer takes its dtype from the tensor it will meet
**The defect class that made f32 look like it worked when it did not.**

Thirteen places inside btorch built a mask, a zero buffer, a ones buffer, the
backward seed, `one_hot`'s output or an optimizer state slot at a hard-coded
`"f64"`. Every one was correct while f64 was the only float dtype. Every one
silently widened an f32 computation back to f64 the moment there was one:

| site | what it did |
|---|---|
| `_reluBackward`, `_leakyBackward`, `_extremeBackward` | an f64 mask times an f32 gradient = f64 gradient |
| `_sumBackward`, `_meanBackward` | f64 `nd_ones` times an f32 gradient |
| `_takeBackward`, gather, `_maxPoolBackward` | f64 scatter destination |
| the `backward()` seed | an f64 seed widened the chain at step one |
| `one_hot` | f64 selection mask widened every `cross_entropy` |
| `Optimizer._slot` | f64 momentum behind f32 parameters |
| `Dropout`, `Embedding` padding, `huber_loss` | f64 masks in forward |

The rule now: **a derived buffer takes `nd_dtype()` of the array it is about to
be combined with** — the incoming gradient for a backward closure, the input
tensor for a forward mask, the parameter for an optimizer slot.

The optimizer slot is the one with the largest consequence. Adam keeps two
slots per parameter, so they outweigh the parameters themselves; f64 slots
behind f32 weights would have cut the memory saving from a half to a third and
made the whole exercise marginal. Measured after the fix: **210.8 MB against
421.0 MB, which is half to within 0.1%** — the remainder being the label arrays,
which are i64 and correctly do not shrink.

How this was caught is worth recording, because the first version of the test
missed it: `tests/btorch_f32_test.b` originally asserted that f32 *values*
agreed with f64 to 1e-6, and they did — a widened f64 result holds an f32 value
perfectly. Only asserting the **dtype** of each result exposed it. A dtype bug
is invisible to a value assertion.

---

### B32 — `Module.to()` mutates each tensor in place
It assigns `$p.data`, it does not replace the `Tensor` object.

That is load-bearing twice over. `named_parameters()` deduplicates by tensor
identity, so a weight shared between two layers — tied embeddings are a real
pattern — must stay the *same object*, or it would be cast twice and, far
worse, **stepped twice by the optimizer**: a silent 2x learning rate on that
tensor alone (the hazard B5's dedup exists to prevent). And an optimizer built
before the cast holds those same objects, so the parameters follow it.

Gradients are dropped on a cast: a stale f64 gradient against an f32 parameter
is a dtype mismatch waiting to happen, and after a cast there is no meaningful
gradient to preserve.

What does **not** follow: optimizer state slots already allocated. So `to()` is
documented as "call it before building the optimizer", and the MNIST sample
does exactly that. An integer buffer — `BatchNorm`'s `num_batches_tracked` — is
left alone, because casting a count to a float would be wrong.
