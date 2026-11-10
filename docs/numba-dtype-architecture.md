# Stage 1 — dtypes, f32, deterministic threading, packed GEMM

The array layer. Phases **D1–D4**, tracked in
[`../numba-suite/ROADMAP.md`](../numba-suite/ROADMAP.md). Context and ordering:
[`ecosystem-architecture.md`](ecosystem-architecture.md).

This stage is first because it multiplies every stage after it: f32 halves
memory and doubles SIMD width, threading multiplies by core count, and the
packed GEMM closes the convolution gap that btorch phase 9 measured and left
open. None of it is a package — it all ships in the `bantu` binary plus
`numba`'s thin Bantu surface.

---

## 1. What is actually there now

Measured, not assumed:

| fact | value | where |
|---|---|---|
| `DType` variants | `{ F64, I64, BOOL }` | `ndarray_native.hpp:61` |
| `itemsize(d)` | `d == BOOL ? 1 : 8` | `ndarray_native.hpp:63` |
| `NdArray::ptr<T>()` | **already templated** | `ndarray_native.hpp:344` |
| `offset`, `strides` | **in elements, signed — not bytes** | `ndarray_native.hpp:332-334` |
| `DType::F64` branches | **76**, across 9 files | `ndarray_*` |
| `double*` / 8-byte sites | **~44**, concentrated in `linalg` (11), `ufunc` (6), `native.cpp` (5) | `ndarray_*` |
| compute-dtype decision | one function, `computeDType(Kind, a, b)` | `ndarray_ufunc.hpp:22` |
| generic element access | `loadD`/`loadI` → `getAsDouble` | `ndarray_kernels.hpp:121-129`, `native.hpp:519` |
| accurate summation | `Pairwise`, a binary-counter tree | `ndarray_reduce.hpp:21-43` |

Three of these make the job far smaller than it looks:

- **`ptr<T>()` is already a template.** Nothing about pointer access needs
  inventing.
- **Strides are in elements.** A 4-byte dtype changes *no* stride arithmetic
  anywhere. This is the single biggest reason f32 is tractable.
- **`computeDType` is one function.** Promotion rules live in one place, so
  adding F32 to the lattice is a local edit, not a sweep.

And one that makes it harder than it looks:

- **`itemsize` returning 8 for everything non-bool is load-bearing in ~44
  places.** Every one must be audited, because a wrong itemsize is not a crash
  — it is silent memory corruption that reads half of one float and half of
  the next.

---

## 2. D1 — the dtype templating pass

**No new dtype in this phase.** The deliverable is that every existing test
passes **bit-identically**, with the hand-written dtype branches replaced by
one dispatch mechanism. This is pure refactor, and it is separated from D2 on
purpose: if f32 and the refactor land together and a test breaks, there is no
way to know which caused it.

### 2.1 The shape of the change

Today, a typical fast path reads (`ndarray_ufunc.hpp:190`):

```cpp
if (cdt == DType::F64) {
    const double* ap = a->ptr<double>();
    const double* bp = b->ptr<double>();
    if (rdt == DType::BOOL) tier0cmp(ap, bp, out->ptr<uint8_t>(), n, ...);
    else                    tier0(ap, bp, out->ptr<double>(), n, fd);
} else {
    /* the i64 copy of the same code */
}
```

After D1:

```cpp
NB_DISPATCH_REAL(cdt, T, {
    const T* ap = a->ptr<T>();
    const T* bp = b->ptr<T>();
    if (rdt == DType::BOOL) tier0cmp(ap, bp, out->ptr<uint8_t>(), n, ...);
    else                    tier0(ap, bp, out->ptr<T>(), n, fd);
});
```

Each kernel body is written **once** over `T`. Adding F32 in D2, and `q8_`'s
int8 in stage 6, then touches the macro's dtype list rather than 76 branch
sites.

### 2.2 Concrete work items

| # | item | note |
|---|---|---|
| 1 | `itemsize(DType)` becomes a real table | `F64→8, I64→8, BOOL→1`, ready for `F32→4, I8→1` |
| 2 | **audit all ~44 `double*`/8-byte sites** | each one either becomes `T`-generic or is annotated with why f64 is correct there (e.g. an accumulator that must stay wide) |
| 3 | `loadD`/`storeD`/`loadI`/`storeI` → `load<T>`/`store<T>` | keep the old names as `load<double>` aliases so nothing outside `ndarray_*` breaks |
| 4 | `NB_DISPATCH_REAL` / `NB_DISPATCH_ALL` macros | one place that enumerates dtypes |
| 5 | `promote()` left alone | F32 rules are D2's problem, not D1's |
| 6 | a `dtypeTraits<T>` struct | `name`, `DType` tag, `widest` accumulator type, limits |

Item 6 earns its keep immediately: **the accumulator type must be separable
from the storage type.** An f32 sum accumulated in f32 loses precision badly
(a 10M-element f32 sum in f32 has error ~1e-3 relative). `dtypeTraits<float>::acc
= double` keeps f32 arrays with f64-quality reductions, which is what NumPy
does and what makes f32 usable for training at all.

### 2.4 What implementation changed about the plan

Two refinements, found while doing the work rather than while planning it.

**A macro was the wrong tool for the ufunc layer.** The plan sketched one
`NB_DISPATCH_REAL(cdt, T, {...})` replacing each `if (cdt == F64) {...} else
{...}` pair. That turned out to make the code *longer*, because the three arms
are not identical modulo `T`: the float arm calls the `fd` functor, the integer
arm calls `fi`, and the BOOL arm calls `fi` on **widened** ints (`x ? 1 : 0`)
and has no arithmetic form at all. Folding them under one macro would have
needed an `if constexpr` ladder inside the macro body — more code, and harder
to read than what it replaced.

What actually works is extracting each arm into a **type-templated helper**:

```cpp
template <class T, class Op>
inline void binTier0(const NdArray& a, const NdArray& b, NdArray& out,
                     size_t n, bool boolOut, Op op) { ... }
```

and leaving a one-line dispatch at the call site:

```cpp
if      (cdt == DType::F64) binTier0<double >(*a, *b, *out, n, boolOut0, fd);
else if (cdt == DType::I64) binTier0<int64_t>(*a, *b, *out, n, boolOut0, fi);
else { /* BOOL: widened functor, cmp form only */ }
```

D2 then adds `else if (cdt == DType::F32) binTier0<float>(..., fd);` — one line
per site, which was the whole goal. Six near-duplicate blocks across
`applyBinary`'s three tiers and `applyUnary`'s two became five helpers.

So the deliverable is sharper than "replace branches with a macro": **every
dtype-specific arm becomes a call to a type-templated helper.** The two macros
still exist and are still the right tool where arms *are* pure storage moves
(copy, cast, byte I/O, `blockCopy`) — hence the split between them:

| macro | dispatches on | used by |
|---|---|---|
| `NB_DISPATCH_COMPUTE` | the **arithmetic** width (`double` \| `int64_t`) | ufuncs, reductions |
| `NB_DISPATCH_STORAGE` | the **stored** width (`double` \| `int64_t` \| `uint8_t`) | copy, cast, byte I/O |

That split is itself a finding. The old code conflated the two: BOOL compute
went down the integer arm and arithmetic happened in `int64_t`, which is
correct (counting bools in `uint8_t` wraps at 255) but was nowhere stated.
`dtypeTraits<uint8_t>::acc = int64_t` now says it.

**f32 correctness is nearly free; only speed needs arms.** The second finding
changes D2's shape. Every generic path in the library funnels through
`getAsDouble` / `setFromDouble` (`ndarray_native.hpp:519`), and every reduction
funnels through a `double`-returning callback. So adding an `F32` case to those
few accessors makes **every kernel in the library work correctly on f32 at
once** — slowly, through the strided path, but correctly.

The fast paths then become opt-in performance, one templated arm at a time.
That makes D2 tractable in a way the original phase description did not convey:

1. `DType::F32` through the enum, `itemsize`, `dtypeName`, `dtypeFromName`, `promote`
2. `F32` cases in `getAsDouble`, `setFromDouble`, `loadI`, `storeI`, `getAsValue`
   — **this is the point at which f32 works everywhere**
3. allocation and constructors (`nd_zeros(..., "f32")`), `nd_astype`, `contiguousCopy`
4. fast-path arms: ufunc tiers (one line each), `extremeFast<float>`, matmul
5. `nd_frombytes`/`nd_tobytes` (the `f32le`/`f32be` *codes* already exist and
   widen to f64; D2 makes f32 a storage dtype rather than only a wire format)

Correctness first through the generic path, then speed through arms — rather
than dtype-by-kernel, which would leave the library half-f32 for a long time.

### 2.3 D1 gate

- Every test in the tree passes, and **`numba_fastpath_test.b`'s exact-equality
  assertions still hold** — it already asserts bit-identity, not `allclose`,
  which makes it the ideal canary for this refactor.
- `benchmarks/run.sh` paired A/B against the pre-refactor binary:
  no regression beyond noise. The templating must not cost anything; if the
  compiler fails to inline through the macro, that shows up here.
- ASan + UBSan clean across the full suite.

**Risk:** a `T`-generic body that compiles for `int64_t` but is wrong for it
(e.g. `std::floor` on an integer, or a division that should be integer
division). Mitigation: `NB_DISPATCH_REAL` excludes integer types where the
operation is float-only — `Kind::REAL` and `Kind::PREDICATE` already compute
in f64 by design (`ndarray_ufunc.hpp:24-29`), and that must survive the
refactor unchanged.

---

## 3. D2 — the `f32` dtype

| # | item |
|---|---|
| 1 | `DType::F32` added to the enum, `itemsize`, `dtypeName`, `parseElem` |
| 2 | `promote()` lattice: `F32+F32→F32`, `F32+F64→F64`, `F32+I64→F64`, `F32+BOOL→F32` |
| 3 | `nd_astype(x, "f32")` both directions, with overflow to `±inf` documented |
| 4 | construction: `nd_zeros/ones/full/arange/linspace/random_*(..., "f32")` |
| 5 | every ufunc, reduction, scan, sort, search, manip and linalg kernel over F32 |
| 6 | reductions accumulate in `dtypeTraits<float>::acc` = `double` |
| 7 | `nd_frombytes`/`nd_tobytes` gain `f32`/`f32be` codes |
| 8 | `.btsd` serialisation of f32 arrays (a header dtype string, already generic) |
| 9 | btorch: `bt.tensor(..., "f32")` and a `Module.to("f32")` cast |

### 3.1 The promotion choice

`F32 + F64 → F64` (widening), matching NumPy and PyTorch's type-promotion
rules. **Rejected:** `F32 + F64 → F32`, which would make a model silently lose
precision the moment a user wrote a plain f64 literal into an f32 graph —
exactly the trap that makes mixed precision painful elsewhere.

`F32 + I64 → F64`, not `F32`: an i64 can hold values an f32 cannot represent,
so narrowing to f32 would lose data silently. f64 can hold the f32 range and
i64 to 2^53, which is the existing documented integer ceiling.

### 3.2 D2 gate

- **f64 paths bit-identical.** `f64` remains the default forever ([E19]) — a
  default change would silently alter every published result in the tree.
- f32 results within `1e-6` relative of the f64 answer for every op.
- `nd_live_bytes` for an f32 array is **exactly half** its f64 equivalent.
- A new `tests/numba_f32_test.b`: every op in both dtypes side by side, the
  promotion lattice asserted exhaustively, `inf`/`NaN`/subnormal round-trips,
  and `astype` overflow landing on `±inf` rather than undefined behaviour.
- btorch's MLP gate re-run in f32: accuracy within 0.3% of the f64 98.16%, in
  measurably less memory. **This is the real test** — an f32 implementation
  that passes unit tests but cannot train is a failure.

---

## 4. D3 — deterministic threading

The consequential phase. Decision [E15]: a **fixed partition**, never work
stealing.

### 4.1 Why work stealing is disqualified

Floating-point addition is not associative. A work-stealing pool assigns
different element ranges to different threads on every run, so the reduction
order changes, so the last bits of every sum change run to run. That makes
results irreproducible — and reproducibility is the property that makes this
stack worth choosing over PyTorch in a regulated setting. Trading it for a few
percent of scheduling efficiency is the wrong trade.

### 4.2 The `Pairwise` problem, stated precisely

This is the finding that shapes the whole phase.

`Pairwise` (`ndarray_reduce.hpp:21`) is a binary-counter tree: after `n`
elements, set bit `k` of `n` holds the sum of exactly `2^k` values. **Its tree
shape is a function of `n` alone.** So if 1M elements are split as 2×500k, each
thread builds a tree over 500k and the two partials are then added — a
different summation order, and therefore a different answer in the last bits,
than one tree over 1M.

There is no partitioning of a `Pairwise` reduction that reproduces the
single-threaded answer. Three options:

| option | bit-identical across thread counts | equals today's answer | verdict |
|---|---|---|---|
| (a) partition by thread count | **no** | no | disqualified |
| (b) fixed chunk size, combine by chunk index, used at **all** thread counts including 1 | **yes** | **no** | **chosen** |
| (c) keep reductions single-threaded | yes | yes | fallback for small `n` |

**Chosen: (b), with (c) below a threshold.** A reduction is split into chunks
of a fixed `kReduceChunk` elements regardless of how many threads exist; each
chunk is summed with `Pairwise`; the chunk totals are combined with a second
`Pairwise` **in ascending chunk index**. One thread and sixteen threads then
produce the same bits, because the combine order is fixed by index, not by
completion order.

The cost, stated plainly: **`nd_sum` and `nd_mean` change in their last bits,
once.** That is a deliberate, documented, one-time re-baseline, and it must be
called out in `numba-suite/CHANGELOG.md` and in the release notes. Any test
asserting an exact f64 sum has to be re-baselined with it.

Accuracy is not harmed — it is still a tree, so error still grows as `log n`.
The existing gate (10M values summed to 2e-16 relative) must be re-verified and
is expected to hold or improve.

Below `kReduceChunk`, a reduction stays on exactly today's path (option c), so
small arrays are untouched and the common case keeps its current bits.

### 4.3 What gets threaded

| kernel class | threaded | partition |
|---|---|---|
| element-wise ufuncs | yes | fixed row/block ranges over the output |
| reductions | yes, above `kReduceChunk` | fixed chunks, combined by index (§4.2) |
| `nd_matmul` | yes | fixed `i`-block ranges; each output block is written by one thread, so **no cross-thread accumulation at all** and the result is trivially identical |
| `im2col`/`col2im` | im2col yes; **col2im no** | col2im *accumulates* into overlapping destinations — a data race, and ordering-dependent. Left single-threaded until a per-thread-accumulator design is measured |
| sort, cumulative scans | no | sequential by nature; not worth the complexity now |
| RNG (`nd_random_*`) | **no** | `nd_seed` is thread-local ([C4]); parallel generation would need per-partition seeding, a separate decision |

`nd_matmul` is the happy case and the most valuable one: because each thread
owns distinct output rows, the blocked `i-k-j` kernel's per-`(i,j)` summation
order is untouched. The bit-identity argument that btorch's `numba_fastpath_test.b`
already asserts carries over unchanged.

`col2im` being excluded is deliberate and costs real speed in convolution
backward. It is listed as an open item rather than hidden.

### 4.4 The API

```
nd_set_threads($n)    // 0 = auto (hardware_concurrency), 1 = fully sequential
nd_get_threads()
```

Default: **1.** Threading is opt-in for the first release. A library whose
numeric results change when it is installed on a machine with more cores is a
support nightmare; opt-in means the user chooses the moment their results
change, and `nd_set_threads(1)` is always an exact reproduction path.

A pool is created once, lazily, and lives for the interpreter's life. No pool
is created at `n == 1`.

### 4.4a Concurrency: the pool meets sua's threading model

The design question the plan did not cover, and the one N18 guarantees will
bite. sua accepts a connection and runs its Bantu handler on a **detached
`std::thread`** (`server.hpp:863`), so every numba global is touched
concurrently. That is why the allocation counters are `std::atomic` and the PRNG
is `thread_local`. A thread pool is the first *large* piece of process-global
mutable state numba would own, and two sua requests can enter `nd_matmul` at
the same instant.

Three ways to handle that:

| option | consequence |
|---|---|
| a global pool behind a mutex | the second request **blocks** until the first finishes its parallel region — one slow matmul stalls an unrelated HTTP request |
| a pool per thread | N threads x M workers; a 50-connection server would spawn hundreds of workers |
| **a global pool, and a caller that finds it busy runs sequentially** | **chosen** |

The third option is only available *because* of the fixed partition ([E15]),
and that is worth stating plainly: **the result of a parallel region does not
depend on how many threads executed it.** So falling back to sequential
execution when the pool is already in use is not a behaviour change at all — it
is the same answer, computed on one thread. A work-stealing design could not do
this; it would have to choose between blocking and non-determinism.

So:

- one pool, created lazily on the first parallel region with `threads > 1`, and
  living for the interpreter's life;
- a single `std::atomic_flag` claims it. Claimed → you parallelise. Already
  claimed → you run the identical loop sequentially and return;
- a `thread_local` "already inside a parallel region" flag makes nested
  parallel calls run sequentially too, so a threaded kernel calling another
  threaded kernel cannot deadlock or oversubscribe.

**`nd_set_threads(n)` is process-global and atomic, not `thread_local`.** This
departs from N18's PRNG precedent on purpose: N18 made the PRNG thread-local
because a shared stream means one request *silently reshapes another's
results*. A shared thread count cannot do that — every thread count produces
identical output — so the objection does not apply, and a global is far more
useful (set it once at startup rather than in every handler). The reasoning is
recorded as N32 so the inconsistency with N18 is deliberate and visible rather
than looking like an oversight.

### 4.4b What the pool must be, for stage 3's sake

[E21] makes `bkit` depend on this phase: `tree_hist` is O(samples x features)
per node and partitions cleanly over samples, so it wants the same machinery.
Therefore the pool is a **general primitive over chunk indices**:

```cpp
// Runs body(0..nChunks-1). Chunk i always computes the same thing regardless of
// which worker runs it or how many workers there are.
void parallelFor(size_t nChunks, const std::function<void(size_t)>& body);
```

and *not* something welded into the `nd_*` kernels. A pool only `nd_*` can call
would strand `tree_` and `tok_` back on one core, which is the whole reason D3
precedes `bkit`.

The one invariant the caller owes: **chunk `i` must write only to storage no
other chunk writes to**, or produce a per-chunk partial that the caller combines
in ascending index order. That is what makes the fixed partition deterministic,
and it is why `nd_col2im` is excluded (§4.3) — it accumulates into overlapping
destinations and satisfies neither form.

### 4.4c No per-call allocation

Invariant 5 of §7, and the phase-9 lesson: `nditer` heap-allocating three
vectors per call cost ~450,000 allocations in one pooling layer and made
`nd_max` 85x slower than it needed to be. A pool that allocates a task vector
per parallel region reintroduces exactly that defect one level up, on a hotter
path.

So the task descriptor is a fixed struct reused across regions, worker threads
are created once, and `parallelFor` on the sequential path (`threads == 1`,
pool busy, or `nChunks == 1`) does **no allocation and takes no lock** — it is a
plain loop, which is what the overwhelming majority of calls will do given the
default of 1.

### 4.4d The completion counter is a lifetime invariant

`parallelFor` wraps the caller's body in a `std::function` that lives on **the
caller's stack**; the pool holds a pointer to it. So "has every worker
finished?" is not bookkeeping, it is the difference between a correct program
and a use-after-free. The invariant is one sentence: *`active_` counts exactly
the workers a region asked to take chunks, and each of those reports back
exactly once.*

Implementation found it off by one. A worker started with `seen = 0`, so a
worker created *during* a region immediately consumed the generation of the
region that had already finished and decremented `active_` for the region being
set up — one phantom decrement, so the caller stopped waiting while a real
worker was still inside the body. ThreadSanitizer on the **integrated
interpreter** reported it as a worker's write to a matmul output racing a later
main-thread read of that buffer; the determinism suite was 57/57 green
throughout, because the answers were right. N33 records the fix and the two
lessons: a sanitizer gate on the component is not a gate on the integrated
build, and a determinism suite cannot find a lifetime bug.

### 4.4e D3b: which element-wise ops get threads

Measured, not assumed. An element-wise loop either saturates the memory bus or
saturates the ALU, and only the second has anything left for a second core:
`exp` goes 1.74x at 16k elements and 4.83x at 1M, while `add` is 0.83x at 1M
and 0.88x at 16M -- *slower*, because one core already saturates the bus and
the region handshake is pure cost. `sqrt` and `divide` win only inside a
131k-262k band whose edges are this machine's cache sizes.

So the transcendentals are threaded and nothing else is. The cost is declared
at the registration site as a trailing `Cost::HEAVY`, defaulting to CHEAP, so
a new ufunc is sequential until measured -- the failure mode of forgetting the
tag is "no speed-up", never "slower" and never "wrong". The threshold is 16384
elements in fixed 4096-element chunks (32 KB, L1-resident); fixed, because a
boundary that moved with the thread count is what §4.4's contract forbids. The
numbers and the reasoning are N34.

Two things this does not do. It does not speed up the MNIST MLP much -- relu
activations, a 1280-element softmax, and an epoch dominated by matmul and cheap
optimizer arithmetic. And it does not touch tier 2 (strided and broadcast
inputs), which walks a shared index and would need its own partitioning; that
is where a bias-add lives, and it is left for later measurement rather than
guessed at.

The error path came with it (N35): a body that throws is caught per chunk, the
**lowest-indexed** exception is rethrown on the caller's thread, the rest are
abandoned, and the pool is released first. No threaded kernel throws today --
everything that can is CHEAP -- so the property is asserted in C++, in
`bantu-src/compiler/tests/parallel_test.cpp`, rather than left as a comment for
bkit to discover.

### 4.4f D3c: the reduction fix was the loop order, not the core count

Measuring before building, as 4.4e now requires, changed what D3c is.
`nd_sum($m, 0, ...)` on a 2048×2048 f64 array took **66 ms while axis 1 took
2.4 ms** — a 27× asymmetry for the same 32 MB and the same adds, because the
loops were nested the wrong way: one output at a time, each walking a column
with stride 2048. No number of threads fixes that; it divides it.

Inverting the loops is bit-identical (same per-output element order, so the same
`Pairwise` tree, the same running product, the same first NaN) and gives 13.3×
on `sum`, 16.1× on `max` and 17.1× on `min`. The two tricks that take it from
4.7× to 12.1× — the shared carry depth and the eight-rows-at-a-time injection
at level 3 — are in N37, with the measurements.

Finding that exposed a second one: with the strided path blocked, `nd_max` over
a **contiguous** axis was slower than over a strided one, because its NaN early
`return` is a loop-carried control dependency and nothing vectorised. Rewritten
as eight independent accumulators plus a branchless `x != x` flag, with a
rescan for the first NaN only when the flag is set: 1.83×, semantics untouched.

**And then the threading is not worth doing.** The fused sum runs at 12.46 GB/s
against the contiguous path's 12.4 — both at the bus, and 4.4e already measured
what threading a bandwidth-bound loop does (`add`: 0.83×). What is left worth
threading is the compute-bound reductions: `nd_var`/`nd_std`, where Welford's
per-element division has the pass running at 0.92 GB/s.

A 27× loop-order defect was sitting underneath a plan to buy 6× with threads.
Measure the single-threaded kernel before deciding it needs cores.

### 4.4g The partition that would have worked anyway



The plan (N26) was to accept a one-time change to `nd_sum`'s last bits, on the
grounds that `Pairwise`'s tree shape depends on the element count and so no
partition could reproduce it. That was wrong, and the premise is what makes it
wrong: *because* the shape is a function of the count, after a multiple of
`2^m` adds level `m` holds the sum of the last `2^m` values associated as a
perfect subtree, with every level below it empty.

So:

* a chunk of exactly `2^m` blocks, summed independently from an empty counter,
  **is** the double the global tree would hold at level `m`;
* injecting the chunk sums back at level `m` in ascending order — one
  `addLevel(m, v)` mirroring `add` with the counter shifted — rebuilds the
  global state exactly;
* the remainder, always under one chunk, is added sequentially and cannot carry
  into level `m`.

`bantu-src/compiler/tests/pairwise_partition_test.cpp` checks 1813 size/chunk
combinations and all 1813 are bit-identical. N36 records it. Axis reductions
need none of this: one accumulator per output element is disjoint-write
parallelism, and `nd_argmin`/`nd_argmax` keep lowest-index ties by combining in
ascending order.

The constraint this accepts is alignment: chunks are `128 * 2^m` elements and
only whole chunks are parallel. That is not a compromise on the answer, which
is the point.

### 4.5 D3 gate

- **`tests/numba_threading_test.b`: every threaded kernel produces bit-identical
  output at 1, 2, 4, 8 and 16 threads.** Asserted with exact equality, not
  `allclose` — the same method that proved the three phase-9 fast paths safe.
- TSan clean **on the integrated interpreter**, not only on a standalone pool
  harness (new tier, in addition to ASan + UBSan). The distinction is not
  pedantry: the standalone harness passed while the pool had a lifetime bug,
  and the integrated build found it in under two minutes (N33).
- `nd_live_bytes` returns to its pre-call value after a threaded op: no
  per-call pool allocation, no leak across 20,000 calls.
- The sequential path measured against the previous build, paired and
  alternating: wrapping a tier-0 loop in a chunk lambda must cost nothing, and
  the `__restrict` promise must survive the capture (it is re-asserted on
  locals inside each chunk body, which is what keeps the vectorisation).
- `bantu-src/compiler/tests/parallel_test.cpp` green, including under TSan: the
  properties no Bantu program can reach -- a throwing body, which exception
  wins, the pool surviving an error.
- Scaling measured and recorded honestly: a kernel that does not scale is
  reported as not scaling, not quietly left in.
- The full suite green at `nd_set_threads(0)` as well as the default 1.

---

## 5. D4 — packed GEMM for skinny shapes

Phase 9 left this measured and open: the three convolution matmuls take 33 ms
of the remaining 74 ms, and `im2col`/`col2im` another ~20 ms near the memory
bound. The blocker is shape — LeNet's second convolution is
`M=? K=25 N=6`, and a `K=25, N=6` inner loop is too short to vectorise or to
amortise loop overhead.

| # | item |
|---|---|
| 1 | pack `A` into a register-blocked panel (`MR×KC`) once, reuse across `N` |
| 2 | pack `B` into `KC×NR` panels |
| 3 | a `4×4` (or `8×4`, measured) register micro-kernel with explicit accumulators |
| 4 | shape-driven selection: packed path only where it measures faster |
| 5 | **the summation order per `(i,j)` must remain monotonic in `k`** |

Item 5 is the correctness constraint carried over from phase 9's B29: the
existing fast paths were proven safe precisely because `k` ascends monotonically
for each output element in the blocked loop, making the result bit-identical to
the generic path. A packed micro-kernel with multiple `k` accumulators
**breaks that** — `(k0+k4+k8…) + (k1+k5+k9…)` is a different order.

So D4 has a fork that must be resolved by measurement before it is built:

- **(a)** keep monotonic `k` (single accumulator chain per `(i,j)`), accept less
  instruction-level parallelism, keep bit-identity and keep
  `numba_fastpath_test.b` passing unchanged;
- **(b)** use multiple `k` accumulators for speed, and accept that matmul's last
  bits change — a second one-time re-baseline on top of D3's.

Preference is **(a)**, on the same reasoning as [E15]. (b) is only worth it if
the measured gap is large, and *that measurement is the first task of D4* —
see open question 2 in the ecosystem doc. The honest possibility is that at
`K=25, N=6` the win lives in the packing and the loop structure, not in extra
accumulators, in which case there is no fork at all.

### 5.1 D4 gate

- `tests/numba_fastpath_test.b` unchanged and passing — exact equality against
  the generic path, with i64 operands, strided views, broadcasts, NaN/±∞, both
  sides of the block size, vectors, square transposes and double transposes.
- The three LeNet convolution matmuls measured before and after; the number
  recorded whatever it is.
- `benchmarks/run.sh` paired A/B: no regression in `nd_matmul` at the large
  square shapes the current kernel is already good at (7.8 GFLOP/s). A fast
  path for skinny shapes that slows down the shapes that already work is a
  net loss.

---

## 6. Order within the stage, and why

```
D1 templating  ──► D2 f32  ──► D3 threading  ──► D4 packed GEMM
   (no dtype)       (opt-in)      (opt-in, def 1)     (shape-gated)
```

D1 before D2 is [E16]: hand-adding f32 to 76 branch sites, then int8 to the
resulting 152, is the path that turns a week into a quarter.

D2 before D3 because f32 halves the bytes each thread moves, and these
kernels are memory-bound — threading an f64 kernel measures the memory bus, not
the cores. Doing f32 first means D3's scaling numbers are the real ones.

D3 before D4 because the packed GEMM's benefit depends on how much of the
matmul is already parallel; measuring D4 against a single-threaded baseline
would answer the wrong question.

Each phase is independently shippable and independently revertible. That is the
point of splitting them: D1 can land and sit for a week proving itself before
any dtype is added.

---

## 7. Invariants for the whole stage

Carried from btorch's DECISIONS and extended. Any of these breaking is a
stop-the-line event, not a tradeoff:

1. **f64 is the default dtype, forever** ([E19]).
2. **f64 results are bit-identical before and after D1, D2 and D4.** The one
   sanctioned exception is D3's reduction re-baseline (§4.2), which is
   documented, deliberate and announced.
3. **Threaded results are bit-identical across thread counts** ([E15]).
4. **Nothing that works today stops working.** The language is in production.
5. **No per-call heap allocation in a kernel.** This is what phase 9 found —
   `nditer` heap-allocating three vectors per call cost ~450,000 allocations in
   one pooling layer, and fixing it made `nd_max` 85× faster. A threading
   implementation that allocates a task vector per call reintroduces exactly
   that defect at a larger scale.
6. **Every new dtype is tested against the f64 answer, not only against
   itself.** A dtype that is self-consistently wrong passes any test written
   only in terms of itself.

---

## 8. Open items

| # | item | settled by |
|---|---|---|
| 1 | Does the templating cost inlining? | D1's paired A/B benchmark |
| 2 | Does `col2im` admit a race-free parallel form? | a per-thread-accumulator prototype, measured against the memory bound |
| 3 | Is the D4 fork real at `K=25, N=6`? | measuring packing-only against accumulators-plus-packing, first task of D4 |
| 4 | Parallel RNG | deferred; needs a per-partition seeding decision that does not break [C4] |
| 5 | f16 / bf16 | after f32 and int8 prove the templating holds |
| 6 | `nd_memmap` | unscheduled; see ecosystem doc §8 |
