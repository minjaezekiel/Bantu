# numba — Design Decisions

Rationale for the choices behind Numerical Bantu, so they aren't re-litigated. Each entry:
**decision · why · alternatives rejected · implication.**

Full reasoning with measurements lives in [`docs/numba-architecture.md`](../docs/numba-architecture.md).

---

### N1 — Native atoms + pure-Bantu library (arctic's A1, reapplied)
**Decision:** Add native *primitives* (an `NdArray` type, `nd_*` kernels); write `numba` (the API,
defaults, composed helpers) in pure Bantu on top.
**Why:** the interpreter costs ~0.38 µs per element (1M-iteration loop, measured on an i7-9750H:
383 ms pure Bantu, down from 2,119 ms before the dispatch fix of
[`docs/interpreter-performance.md`](../docs/interpreter-performance.md) -- which does not change
this conclusion, since NumPy-class is ~10 ns per element, forty times faster still) and a `Value` is ~190 bytes carrying a string, a vector, a `std::function` and
three `shared_ptr`s at once. Any element loop written in Bantu is a toy. This is the split arctic
already proved in production.
**Rejected:** a pure-Bantu ndarray (unusable); a full native library with no Bantu layer (not
"written in Bantu", larger trusted surface, and no optional arguments — see N8).
**Implication:** the native surface is small and general; `numba.b` is portable Bantu.

### N2 — A new `NdArray` type, not an extended `arctic::Column`
**Decision:** New header `ndarray_native.hpp`, namespace `numba`, handle tag `"ndarray"`.
**Why:** `Column` stores data in `std::vector<double>`, and a vector owns its allocation — it cannot
portably alias memory it does not own, so **zero-copy views are structurally impossible there**.
`Column`'s null mask would also have to become strided (a transposed view needs a transposed mask),
and all 83 `col_*` kernels assume `n` is the whole extent, so adding shape means auditing every one
on a production interpreter. The two libraries also have divergent dtype needs (arctic needs UTF8
and never wants F32; numba the reverse).
**Rejected:** adding `shape`/`strides` to `Column`.
**Implication:** two independent native layers that never include each other; the bridge lives in
`evaluator.hpp`, the one place that legitimately sees both.

### N3 — Buffer ownership split from array metadata
**Decision:** `struct Buffer { void* data; size_t nbytes; bool owned; shared_ptr<void> keepalive; }`
held by `shared_ptr`; `NdArray` carries `buf`, `offset`, `shape`, `strides`, `writable`.
**Why:** this is what makes `reshape`/`transpose`/`slice`/`flip` zero-copy — the defining property of
an ndarray library. The `shared_ptr` keeps the base alive behind any view, which is the same RAII
argument that made A3 choose `NATIVE_HANDLE` in the first place. `keepalive` additionally lets a
buffer *borrow* an arctic column's storage (N9).
**Implication:** a view outliving its base is safe by construction, not by discipline.

### N4 — Strides in elements, signed; contiguity computed, never cached
**Decision:** `strides` is `vector<ptrdiff_t>` counted in elements (NumPy uses bytes);
`isCContig()`/`isFContig()` are computed on demand.
**Why (elements):** with one itemsize per array the two are equivalent, and elements removes an
entire class of "forgot to multiply by 8" bugs. Signed strides are what make `nd_flip` a view.
**Why (computed):** a cached flag must be recomputed by every op touching shape or strides; one
missed update leaves a stale `true`, which sends a kernel down the raw-pointer fast path over
strided data — **silent memory corruption with no crash**. Computing it is ≤ 20 ns against a kernel
that then touches millions of elements.
**Rejected:** byte strides; a cached flag.

### N5 — dtypes f64 / i64 / bool; no validity mask; no complex in v1
**Decision:** three dtypes. BOOL is one byte per element, not a bitmask. NaN is the null for f64.
**Why (byte bools):** a bitmask cannot carry a stride, so boolean views and transposes would be
impossible. **Why (no mask):** no NumPy analogue; every view would need a parallel strided mask,
doubling the inner loop. Nulls stay arctic's concern and the bridge refuses a column that has any.
**Why (no complex):** doubles every ufunc, Bantu's `Value` has no complex scalar, and its only
compelling consumers (FFT, general `eig`) are already out of scope.
**Implication:** `DType` is forward-compatible — add `C128`, have `itemsize` return 16, and only
opted-in kernels need new arms.

### N6 — `nd_` naming
**Decision:** all atoms are `nd_`-prefixed `verb_noun`; `nd(...)` constructs.
**Why:** pairs with the `"ndarray"` tag and `nd_ndim`, greppable, no collisions with existing
builtins. Mirrors A5's reasoning for `col_`.
**Rejected:** `arr_` (A5 rejected it for arctic as less intuitive; the argument is weaker here since
an array *is* the concept, but `nd_` still wins on tag symmetry).

### N7 — Operator dispatch on native handles, not class dunder metamethods
**Decision:** `evalBinaryOp`, `evalUnaryOp`, `evalIndexAccess`, `evalIndexAssign`, `evalDotAccess`
and `Value::toString` each gain one arm that dispatches when the operand is a `NATIVE_HANDLE`.
**Why:** before this, `$a * $b` on a handle read `numberVal` and silently produced **0**, `$a[i]`
produced **null**, and `$a[i] = v` threw. There was exactly one handle tag (`"column"`) and
`arctic.b` never puts a column in an arithmetic expression, so **every path being filled in was
dead** — this is a bug fix, not a behaviour change. The hot-path cost is one predicted branch
(~0.3–1 ns) against ~1 µs per interpreted iteration.
**Rejected:** full `CLASS_INSTANCE` dunder metamethods — a permanent language commitment deserving
its own coherent design (`__add__`/`__radd__`/`__getitem__`/… with reflected forms), and unnecessary
here because the handle *is* the object, with no class wrapper needing it. If ever added: **dunder
names only** — dispatching `+` to a method named `add` would silently hijack `+` for any class with
one (a `Set`, a `Counter`, a `Cart`).
**Implication:** gated on a real measurement — `benchmarks/run.sh` within ±2%, both numbers in the
CHANGELOG.

### N8 — The pure-Bantu façade is the public API; `nd_*` is plumbing
**Decision:** users call `np.zeros([3,4])`, not `nd_zeros([3,4])`.
**Why:** not cosmetic. **The façade can have optional arguments and the builtins cannot.** Bantu
binds missing arguments to null, so `def arange($start, $stop, $step)` called as `np.arange(0, 10)`
works, while the raw builtin would need `nd_arange(0, 10, null)`. It also lets the composed helpers
(`polyfit`, `interp`, `gradient`, `cov`, `meshgrid`) live in Bantu, which is N1's whole point.
**Implication:** every builtin ships with its wrapper, its doc line and its runnable example in the
same phase. A builtin with none of those is not finished.

### N9 — Zero-copy in from arctic, copy out
**Decision:** `Column → NdArray` borrows the column's buffer via `Buffer::keepalive` and is
`writable = false`. `NdArray → Column` copies. `nd_from_frame` builds F-contiguous.
**Why (borrow in):** free, and arctic documents columns as immutable so read-only honours the
contract. **Why (copy out):** `Column` stores a `std::vector`, which cannot adopt a foreign pointer.
**Why (F-contiguous):** each column becomes one `memcpy` rather than a strided scatter, and every
kernel already handles F order via `isFContig()`.
**Preconditions, reported by name:** dtype F64/I64/BOOL, `null_count == 0`, `logical == NONE`.

### N10 — arctic gets its own transcendental kernels; it does not delegate
**Decision:** write the ~15 missing `col_sqrt/exp/log/…` natively in `dataframe_native.hpp`.
**Why:** each is ~10 lines on the existing `unaryOp` shape. Delegating to numba would make
`series.sqrt()` require a numba-capable build and would lose null semantics, since NaN is not null.
The bridge exists for what arctic genuinely cannot do — matmul, `solve`, `svd`, 2-D reductions.

### N11 — Kernels in their own TU, compiled at `-O3`
**Decision:** `ndarray_native.cpp` is the only TU including the implementation header, and each build
script overrides its flags to `-O3 -ftree-vectorize`.
**Why:** the production Linux binary is built in `ubuntu:22.04` with `-O2`, and **GCC 11 does not
enable `-ftree-loop-vectorize` at `-O2`** (GCC 12 does). `build-mac.sh` also uses `-O2` but that is
Apple Clang, which *does*. So a kernel benchmarked on macOS gets NEON while the same kernel in the
shipped Linux binary gets a scalar loop — **any Mac-measured number is not the number users get**,
and no benchmark gate is meaningful until this is closed. It also keeps `evaluator.hpp` (already
8,883 lines) from growing by two thousand.
**Rejected:** `#pragma GCC optimize("O3")` — GCC documents the function-level attribute as unsuitable
for production and Clang ignores the pragma entirely, so it would fix nothing on macOS while risking
Linux. `-march=native` — portability to older CPUs is a documented project constraint.
**Implication:** CI compiles that TU with `-fopt-info-vec-optimized` and greps for the tier-0 loops,
exactly as it already greps for `event loop: epoll`. A silent regression to scalar is otherwise
invisible.

### N12 — No new runtime dependency: no OpenMP, no BLAS in v1
**Decision:** everything hand-written; single-threaded in v1.
**Why (no OpenMP):** links libgomp, against the house rule that produced a from-scratch P-256 rather
than linking OpenSSL and rejected libuv for sua; needs `brew install libomp` for every macOS
contributor; and the headline op is DRAM-bandwidth-bound (0.042 flop/byte), so the payoff is
~1.5–2.5×, not 4×.
**Why (no BLAS):** the existing optional flags (`BANTU_SODIUM`, `BANTU_ARROW`) gate **capability** —
things that genuinely cannot be written in-tree, which `has_native()` can answer yes/no about. A
`BANTU_BLAS` flag would gate **speed**, creating a two-tier contract where the same program runs 6×
slower on the binary users download. A blocked matmul is 60 lines and gets within 5–10× of OpenBLAS.
**Deferred, not closed:** if a concrete wall appears, gate it as `has_native("blas")`, swap only
`gemm`/`gesv`/`gesdd` behind identical names so only speed changes, and prototype on macOS where
Accelerate ships with the OS.

### N13 — Easy algorithms for `eigh` and `svd`; no general `eig`
**Decision:** cyclic Jacobi for `eigh`; one-sided Jacobi for `svd`. No nonsymmetric `eig`.
**Why:** Jacobi is ~100–120 lines, unconditionally convergent, needs no shift strategy, and
one-sided Jacobi has *higher* relative accuracy on small singular values than Golub–Kahan. It is
3–5× slower than LAPACK and impractical much beyond n ≈ 500–800, which is documented rather than
hidden. General `eig` needs balancing + Hessenberg + Francis double-shift QR + complex arithmetic we
do not have; **a fragile `eig` is worse than no `eig`**, and `eigh` covers covariance, PCA and graph
Laplacians.

### N14 — Numerical accuracy is a gate, not an aspiration
**Decision:** pairwise summation for `sum`/`mean`, Welford for `var`/`std` (matching A8).
**Why:** a naive f64 accumulator over 10M values loses ~6 significant digits. Users will diff against
NumPy, which does pairwise. The phase-3 gate — sum of 10M copies of `0.1`, relative error < 1e-12 —
**fails a naive implementation by design.**

### N15 — Security: unvalidated input must never reach an allocation or an index
**Decision:** checked multiplication for every shape product; a configurable allocation ceiling
(default ~2 GB); `writable = false` on `broadcast_to` results; bounds checks on every index;
`out=` overlap detected by buffer identity plus element extent, then copied.
**Why:** `nd_zeros([2^22,2^22,2^22])` overflows `size_t` to a small number, allocates a tiny buffer,
and every later kernel writes past it — a heap overflow from one line of user script. A stride-0
broadcast view silently corrupts on write. And numba may run inside a sua request handler, where an
unbounded allocation kills a server.
**Precedent:** the `sua.udp` review, where two of four defects killed the process because one
unvalidated argument sized an allocation. Only adversarial tests find these.

### N16 — Adoption is a requirement, gated like any other
**Decision:** the module resolver learns `./bantu_modules/<name>/`; a repeat `include` binds the
cached module instead of returning early; `print($handle)` renders the array.
**Why:** `bantu add numba` installed to `./bantu_modules/<name>/` and the resolver had **no rule for
that directory**, so `include "numba" as np` did not work — the first thing every new user hits. And
the include cycle guard returned *before* binding the alias, so two modules both including one
package left the second with an **unbound alias** and only a stderr line, which any real application
hits. Both were latent for every existing package (`arctic`, `hash`, `crypto`, `uuid`, `random`,
`orm`), so fixing them fixes all of them.
**Implication:** these land in Phase A, before any numba code, because everything else rides on them.

### N17 — The allocation ceiling bounds total live bytes, and script cannot raise it
**Decision:** two numbers, not one. A **hard** ceiling read once at startup from
`BANTU_ND_MAX_BYTES`, and a **soft** ceiling that `nd_max_bytes()` may only move *downward* from it.
Admission is tested against `live + requested`, not the request alone, with live bytes tracked in an
atomic counter incremented in the `Buffer` constructor and decremented in its destructor.
**Why:** the Phase 1 ceiling was per-allocation, which does not bound a loop — twelve 200 MB arrays
were measured held simultaneously against a 2 GiB per-call limit, with no error. A loop is how a
request handler actually exhausts a server, so the per-call limit missed the case it existed for.
And a limit the untrusted side can raise for itself (`nd_max_bytes(1e18)`) is a guard against
mistakes, not against hostile input.
**Precedent:** CPython's answer to CVE-2020-10735 has exactly this shape — `PYTHONINTMAXSTRDIGITS`
plus `-X int_max_str_digits` owned by the operator, and `sys.set_int_max_str_digits()` for the
program. The env var is the half that a script cannot reach.
**Implication:** the error message must distinguish "you can raise this" from "the operator capped
this", or it advises a fix that cannot work. A failed allocation must return its accounted bytes
before unwinding — accounted-but-never-freed is worse than no limit, because the process slowly
refuses everything and nothing says why. Gated by a stress assertion that live bytes return to
baseline **exactly**, which RSS is far too noisy to show.

### N18 — Process-global mutable state is a bug, because sua handlers are threads
**Decision:** the allocation counters are `std::atomic`; the PRNG is `thread_local`.
**Why:** sua accepts a connection and runs its Bantu handler on a detached `std::thread`
(`server.hpp:863`), so every numba global is touched concurrently. A plain `size_t` limit is a data
race, which is undefined behaviour rather than a stale read. Worse, a shared PRNG means `nd_seed()`
in one request silently reshapes every other in-flight request's random arrays — which is the exact
objection that made numba carry its own stream instead of using the global `random()`, one level up.
**Implication:** each thread seeds its own stream, and `nd_seed` documents that. numba is the first
thing in the tree to put *mutable* process-global state behind a builtin, so it is the first to
expose that sua's threading model and the interpreter's globals were never reconciled; anything else
adding a global will hit the same wall.

### N19 — The view extent invariant lives in `makeView`, not in its callers
**Decision:** `makeView` validates that every element the shape and strides can address lies inside
the buffer, and `nd_slice`'s user-supplied arithmetic uses a checked **signed** multiply.
**Why:** `makeView` validated nothing — it trusted caller-supplied shape, strides and offset. That
survived Phase 1 only because seven view constructors each derived strides from a valid parent, i.e.
the invariant held by construction rather than by checking, and phases 2–5 add many more. It was
already wrong: `nd_slice` never type-checked `start`/`stop`/`step`, so `[[1,2], null, null]` was read
as index 0, and a step of -2^62 on a stride-8 axis wrapped `stride * step` to **0** — manufacturing a
stride-0 axis on an array still marked writable, which is precisely the state `broadcast_to` refuses
to produce. Signed overflow is undefined behaviour, so unlike the unsigned case it cannot be detected
after the fact.
**Precedent:** NumPy's `PyArray_CheckStrides`, and NumPy's own `as_strided` documentation — get
strides wrong and "array elements can point to invalid memory and can corrupt results or crash your
program."
**Implication:** `nd_as_strided` is **deliberately never shipped**. Not exposing it is what keeps
every stride in the system numba-derived, which is what makes this invariant maintainable at all.
The same extent walk backs `nd_shares_memory`, which Phase 2's `out=` needs for aliasing.

### N20 — The `memset` in `Buffer` is load-bearing; do not switch to `calloc`
**Decision:** allocation touches every page at the point of allocation. `nd_empty` is the single
exception and gets its DoS bound from N17's accounting instead.
**Why:** under Linux's default overcommit a large `posix_memalign` or `calloc` succeeds without
committing a page — `calloc` serves large blocks from kernel zero pages — so the memory is charged
only when a kernel touches it. The ceiling's bookkeeping would say everything is fine and the OOM
killer would arrive later, mid-kernel, with no exception for a Bantu `try/catch` and a dead worker
inside a sua handler. The `memset` converts a deferred uncatchable kill into an error raised on the
line that caused it. It costs ~40 ms per 10M f64, which is the price of deterministic failure.
**Implication:** this is the obvious performance "fix" (NumPy splits `npy_alloc_cache` from
`npy_alloc_cache_zero` exactly this way), so it is recorded at the `memset` itself,
in `docs/numba-security.md` §2 and in `docs/numba-acceleration.md` §4.

### N21 — No GPU backend before Phase 7, and never for ufuncs
**Decision:** no CUDA/Metal/ROCm/SYCL backend in phases 1–6. Revisit only for Phase 5 linear algebra,
only on a measured wall, and then as an explicit residency model (`nd_to_device`/`nd_to_host`) behind
a capability flag — never as a transparent accelerator.
**Why:** `c = a + b` has an arithmetic intensity of 0.042 flop/byte and is DRAM-bandwidth-bound. A
discrete GPU must first move the data across PCIe at ~25 GB/s against host DRAM at 50–200 GB/s, so
the transfer alone costs more than doing the whole operation on the CPU. GPUs pay only when data
stays *resident* across many kernels, which is a different programming model — it is why CuPy has a
separate array type rather than making NumPy faster, and why CuPy tells users not to bother below
~10k elements. On top of that, a GPU dependency is the `BANTU_BLAS` objection several times over:
it would gate *speed*, not capability, so the same program would run several times slower on the
binary users download.
**Implication:** the one accommodation taken now is **page alignment for large buffers**, which costs
one constant and serves non-temporal stores, huge pages, and a possible future zero-copy `MTLBuffer`
(`newBufferWithBytesNoCopy` requires page-aligned memory) all at once. `Buffer` already separates
ownership from array metadata, which is the seam a device allocation would use; it does **not** get a
speculative `device` enum today. Full analysis in `docs/numba-acceleration.md`.

### N22 — One core cannot saturate DRAM, so the Phase 2 bandwidth gate is per-platform
**Decision:** the roadmap's "10M `nd_add` ≥ 10 GB/s effective" becomes a per-platform target, and a
`std::thread` parallel tier-0 is prototyped in Phase 2 rather than assumed unnecessary.
**Why:** `docs/numba-architecture.md` §3 concluded that threading was "not needed for the headline
op" because it is bandwidth- rather than ALU-bound. The premise is right; the conclusion does not
follow. A single core sustains only ~10 concurrent L1 misses, capping one thread near
`10 × 64 B / latency` — about **8.1 GB/s** at a 79 ns memory latency. Bandwidth-bound means you need
enough cores to fill the memory pipeline, then no more. Single-threaded passes the gate comfortably
on Apple silicon and plausibly cannot reach it on a high-latency shared cloud vCPU, which is the
machine the Linux binary actually runs on.
**Why not OpenMP:** it links `libgomp` — a new runtime dependency needing `brew install libomp` for
every contributor. `std::thread` is already used in the tree (`server.hpp`), so a small fixed-size
pool adds nothing. It must be threshold-gated (thread wake-up is microseconds) and must not multiply
against sua's per-connection threads, the same interaction that forced N18.

---

### N23 — The dtype templating pass precedes f32, not the other way round
Every kernel today has an f64 fast path and a generic path. Adding f32 by hand
makes that a 3×2 matrix of hand-written branches; adding int8 in ecosystem
stage 6 makes it 4×2. Template each kernel once over `T` **first** (item D1),
then add dtypes.

More boring, and it is the difference between a week and a quarter by the time
`q8_` lands. Grounded in a count rather than an impression: the surface is
**76 `DType::F64` branches** and ~44 `double*` sites, and `NdArray::ptr<T>()`
is already templated with strides in elements (N4), so the job is bounded.

Ecosystem-wide this is [E16].

---

### N24 — The accumulator type is separate from the storage type
`dtypeTraits<T>::acc`, introduced in D1 and load-bearing from D2 onward.

An f32 sum accumulated in f32 is unusable: a 10M-element f32 reduction has
~1e-3 relative error, against the ~2e-16 this library already guarantees for
f64. So `dtypeTraits<float>::acc = double` — f32 **arrays** with f64-quality
**reductions**, which is what NumPy does and what makes f32 viable for training
at all.

Without this split, f32 would halve memory and quietly destroy the accuracy
property that N-era pairwise summation was built to provide.

---

### N25 — Threading uses a fixed partition, never work stealing; default 1
Floating-point addition is not associative, so a work-stealing pool assigns
different element ranges on every run, changing the last bits of every
reduction run to run. That makes results irreproducible — and reproducibility
is the property that makes this stack worth choosing over the alternatives in a
regulated setting. A few percent of scheduling efficiency is not worth it.

So: a fixed partition computed from shape, giving **bit-identical results at
any thread count**, plus `nd_set_threads(n)`.

**Default 1, opt-in.** A library whose numeric results change when it is
installed on a machine with more cores is a support nightmare; opt-in means the
user chooses the moment their results change, and `nd_set_threads(1)` is always
an exact reproduction path. No pool is created at `n == 1`.

Ecosystem-wide this is [E15]. **Rejected:** language-level threads — they touch
the GC and the evaluator, a far larger change for the same arithmetic benefit.

---

### N26 — ~~`nd_sum`'s last bits change once~~ — WRONG, and nothing has to change
**Superseded by N36.** Kept in full because the mistake is instructive: the
plan was prepared to make a user-visible change to a shipped function, and the
justification sounded airtight.

> ~~`Pairwise` is a binary-counter tree whose shape is a function of the element
> count alone (`ndarray_reduce.hpp:21`), so **no partitioning of a threaded
> reduction reproduces the single-threaded answer.** The chosen scheme — fixed
> chunks combined by ascending index, used at *all* thread counts including one
> — is self-consistent and reproducible, but differs from today's answer in the
> last bits.~~

The premise is true and the conclusion does not follow. The tree's shape being
a function of the element count is exactly what makes an *aligned* partition
reproduce it: see N36. A plan that is ready to change a shipped answer should
be made to prove the change is necessary, not merely plausible — the proof is
80 lines and it came out the other way.

---

### N27 — FFT belongs here, not in `bsci`
NumPy has `numpy.fft`, and that is the correct layering: a DFT is an array
transform. So `nd_fft`/`nd_ifft`/`nd_rfft`/`nd_irfft`/`nd_fftfreq` are kernels
in this package (item D5), and `bsci.signal` builds windows, spectrograms,
filters, `convolve` and `resample` on top.

**Rejected:** FFT in `bsci`. A numba user would be unable to transform an array
without a scipy-class dependency, which inverts the layering.

Three size regimes, because **every length must work**: radix-4 for powers of
two, mixed-radix for small prime factors, and **Bluestein** for arbitrary
lengths. Skipping Bluestein is the tempting shortcut and it is a silent bug —
`nd_fft` on 1000 points would either error or zero-pad to 1024, and a silently
zero-padded spectrum is wrong in a way that looks entirely plausible.

Ecosystem-wide this is [E4].

---

### N28 — a bare Bantu number is a WEAK operand against f32
**Found by implementation, and f32 is useless without it.**

Bantu numbers are doubles, so a literal becomes an f64 array and the promotion
lattice dutifully widens: `f32_array + 0.5` returned **f64**. One scalar
multiply — `grad * lr`, `x * 0.5`, the `1/n` in every mean — would promote an
entire f32 model back to f64 and undo the whole point of the dtype. Measured
before the fix:

```
f32 + 0.5  -> f64        f32 * 2  -> f64
```

So a bare number (or bool) is **weak**: it does not drive the result dtype
against an f32 array. NumPy reached the same conclusion in NEP 50 and PyTorch
has always worked this way.

Deliberately narrow — `weakDType` fires **only** where a scalar's f64 meets an
f32 array:

- `i64_array + 1.5` still gives **f64**. The scalar is not being weakened
  against an f32 operand, so the pre-f32 lattice is untouched and nothing that
  worked before changes by a bit.
- a real f64 **array** against an f32 array still widens to f64. Only a bare
  number is weak; an array is an explicit statement of precision.

**Rejected:** general value-based casting (inspecting whether `1.0` "looks like"
an integer). Bantu cannot distinguish `1` from `1.0` — both are doubles — so
any value-based rule would be guessing, and guessing about precision is how you
get a library whose dtypes depend on the data.

---

### N29 — f32 stores in 4 bytes but currently COMPUTES in double
The honest limit of what landed, recorded so the next person does not have to
measure it to find out.

f32 was meant to buy two things: half the memory and twice the SIMD width. **It
delivers the first, not yet the second.** The ufunc functors are declared
`[](double x, double y)` in `ndarray_ufunc_reg.hpp`, so an f32 operand is
promoted to double, the arithmetic happens in double, and the result is narrowed
back to float on store.

That is *more* accurate than true f32 arithmetic and perfectly correct — but it
means no f32 vector-width gain on element-wise work.

Getting it needs the ~55 registered functors to become generic (`[](auto x,
auto y)`), which is mechanical but touches every ufunc, so it is split out as
**D2b** rather than smuggled into a change that is already large. For f64 the
generic form deduces `double` and is bit-identical, so D2b carries no risk to
existing paths — it was separated for reviewability, not for safety.

Memory is the win that is real today, and for training it is the one that
matters more: it is what lets a model and its optimizer state fit.

---

### N30 — "not f64" never again means "therefore integer"
Three separate kernels had independently written the rule

```cpp
const DType rdt = (x->dtype == DType::F64) ? DType::F64 : DType::I64;
```

which was correct while the only float dtype was f64, and became **silently
destructive** the moment f32 existed: `nd_cumsum` on an f32 array returned
**i64**, truncating every partial sum to a whole number. Measured: a cumsum
whose true values ran to 1116.8 returned 1117, a relative error of 1.0 against
the 5.9e-8 that f32 storage actually permits.

`nd_sum`, `nd_prod`, `nd_cumsum`, `nd_cumprod`, `nd_diff` and `nd_unique` all
carried a copy of it. The replacement is two named rules, written once in
`ndarray_reduce.hpp`:

| rule | for | behaviour |
|---|---|---|
| `reduceDType(in)` | sum, prod, cumsum, cumprod, diff, unique | a float keeps its own width; everything integer-ish becomes i64 |
| `floatResultDType(in)` | mean, var, std, median, quantile, nan* | always a float; f32 stays f32 rather than widening |

The second rule matters as much as the first and is less obvious: widening
`mean(f32)` to f64 is not *wrong*, but it promotes a whole f32 pipeline back to
f64 one call in — the same failure as N28, arriving through reductions instead
of scalars.

A `grep` for the old pattern across `ndarray_*` now returns nothing, and
`tests/numba_f32_test.b` asserts the result dtype of **every** reduction and
scan by name so a fourth copy cannot reappear quietly.

---

### N31 — linear algebra stays f64 for now, and that is why btorch f32 is not reachable yet
`nd_matmul`, `nd_dot`, `nd_solve`, `nd_inv`, `nd_det` and every decomposition
compute on `double*` buffers and **return f64 whatever dtype they were given.**
Values are correct — a 4x4 f32 solve agrees with the f64 answer to 1.37e-07,
which is f32 input resolution — but the dtype is not preserved.

For the decompositions that is a defensible permanent choice: LU, QR, Cholesky,
Jacobi eigh and SVD are precision-sensitive and are not in a training hot path,
so computing and returning f64 is the right default.

**For `nd_matmul` it is a real limitation**, because matmul is the operation a
neural network is made of. Until it preserves f32, a `Linear` layer turns an
f32 model back into f64 at the first layer — the same failure as N28 and N30,
arriving through linear algebra. That is the concrete reason item **D2c**
(btorch f32 plumbing) is recorded as not yet reachable rather than merely not
yet done.

The fix is not "return f32": it is **f32 in, f64 accumulator, f32 out**, which
is what every real BLAS does for mixed precision and what N24 already requires
elsewhere. Writing it needs the blocked kernels templated over the input type
with the accumulator held separately — and those kernels are being rewritten in
**D4** (the packed GEMM) anyway, so doing it twice would be waste.

**Rejected:** widening both inputs to f64, multiplying, and storing the result
as f32. It would preserve the dtype today and keep full accuracy, but it
allocates an f64 copy of both operands — doubling peak memory on the largest
arrays in the program, which is precisely the cost f32 was adopted to avoid.
Correct dtype bought with the memory f32 exists to save is not a trade worth
making.

Pinned, not left to chance: `tests/numba_f32_test.b` asserts all four current
return dtypes by name, with a comment saying these are the assertions to change
when D4 lands.

---

### N32 — threading is OPT-IN, and the thread count is a PROCESS-WIDE setting

`nd_set_threads(n)` with a default of **1**. Nothing in numba uses more than one
thread until a program asks for it.

Two reasons the default is not "as many as the hardware reports":

* **sua.** A server runs each connection's handler on its own detached thread
  (N18). If every handler silently claimed every core, eight concurrent
  requests would oversubscribe the machine by 8x and each would be *slower*
  than single-threaded. The process-wide knob lets the operator decide once,
  at start-up, with knowledge the library does not have.
* **Reproducibility is a promise, so it must not be a surprise.** It is kept at
  any thread count (N25), but a user who never asked for threads should not
  have to take that on trust.

The count is stored in one atomic, read once per region, and is *not*
thread-local. A thread-local count was considered and rejected: a Bantu program
that set threads at start-up and then served requests on handler threads would
find the setting silently gone, because those threads never ran the line that
set it. One process, one answer to "how wide is this machine allowed to go".

`nd_set_threads(0)` means "as many as the hardware reports", and
`nd_hardware_threads()` reports that number without adopting it.

### N33 — the pool's completion counter is a memory-safety invariant, not bookkeeping

`parallelFor(nChunks, body)` wraps the caller's body in a `std::function` that
lives on **the caller's stack**, and the pool holds only a pointer to it. So
`run()` returning while any worker is still inside the body is not an untidiness
— it is a use-after-free, plus a second region claiming chunk indices while a
straggler is still executing the previous one's body.

Everything therefore rests on one invariant: **`active_` counts exactly the
workers a region asked to take chunks, and each of those reports back exactly
once.** Off by one in one direction is a hang; off by one in the other is the
use-after-free.

It was off by one. A worker thread started with `seen = 0`, so a worker created
during region *G* found `generation_ != seen` true the instant it started,
consumed the generation of the region that had **already finished**
(`body_ == nullptr`, so it ran nothing), and decremented `active_` for the
region that was at that moment being set up. One phantom decrement: the counter
reached zero while a real worker was still inside the body, and `run()`
returned.

Nothing in the results showed it. The determinism suite was 57/57 green, the
1024³ matmul was bit-identical at 1, 2, 4, 6, 8 and 12 threads, and the MNIST
losses matched byte for byte. The window is a few microseconds wide and only
opens on the *first* region after a worker is created, so the straggler almost
always finished before the caller's frame was reused.

**ThreadSanitizer on the integrated interpreter found it in under two minutes**,
as a write by a worker to a matmul output racing a later main-thread read of the
same buffer. The standalone pool stress test — 12 concurrent callers, 60
regions, every thread count from 1 to 16 — had not, because it warmed the pool
before measuring and therefore never created a worker inside a region under
test.

The fix is to hand each new worker the generation current **at its birth**, so
its first wake-up is for the next region — the one whose `active_` counts it.
Two further rules fell out of writing the invariant down:

* only the workers that will *take* chunks are counted
  (`active_ = min(limit_, workers_.size())`), and only those report back. A
  region asking for 2 threads from a pool that once ran 16 wide no longer wakes
  fifteen threads to have fourteen of them immediately decrement a counter.
* the sequential fallbacks — pool busy, nested region, one chunk, one thread —
  never touch the counter at all.

Two things to keep from this. **A sanitizer gate on the integrated build is not
the same test as a sanitizer gate on the component**: the component test was
clean and wrong. And **a determinism suite cannot find a lifetime bug** — it
compares answers, and the answers were right. They are different tiers for
different defects, which is why §7's six tiers are six and not five.

---

### N34 — only the compute-bound element-wise ops get threads

An element-wise loop either saturates the memory bus or saturates the ALU, and
only the second kind has anything left over for a second core. Measured on this
machine (10 cores, f64, 1 thread -> 6 threads, alternating order, 2026-11-10):

| op | n=16k | n=131k | n=262k | n=1M | n=4M | n=16M |
|---|---|---|---|---|---|---|
| `exp` | 1.74x | 3.12x | 3.51x | 4.83x | | |
| `log` | 2.12x | 3.15x | | 5.51x | | |
| `tanh` | 1.25x | 3.11x | 2.90x | 4.53x | | |
| `power` | 2.81x | 3.90x | 4.55x | 5.06x | | |
| `sqrt` | 0.64x | 1.74x | 1.94x | 1.29x | | |
| `divide` | 0.57x | 1.22x | 1.50x | 1.00x | | |
| `abs` | | 0.92x | 1.10x | 1.08x | | |
| `add` | | | | 0.83x | 0.91x | 0.88x |

So the transcendentals are threaded and **nothing else is**. Three readings
behind that:

* **`add` is SLOWER on six threads at every size large enough to matter.** One
  core already saturates the bus; the region handshake is pure cost. An op that
  got *slower* when a user turned threading on would be worse than an op that
  stayed the same speed, because it would make `nd_set_threads` a thing to be
  afraid of.
* **`sqrt` and `divide` win, but only inside a band.** 131k-262k elements is
  where the arrays no longer fit in one core's L2 but the bus is not yet the
  limit — and the band's edges are *this machine's cache sizes*. A rule that
  holds on one machine is not a rule.
* **The threshold is 16384 elements**, which is where a threaded region first
  pays for itself (1.74x on `exp`), with a fixed 4096-element chunk — 32 KB of
  f64, so a chunk is L1-resident. Fixed, because a boundary that moved with the
  thread count is exactly what N25 forbids.

Cost is declared at the registration site, next to the functor, as a trailing
`Cost::HEAVY`; the default is `Cost::CHEAP`, so **a new ufunc is sequential
until somebody measures it**. That ordering is deliberate: the failure mode of
forgetting the tag is "no speed-up", not "slower than before" and not "wrong".

What this does NOT do is speed up the MNIST MLP much. Its activations are
`relu`, its softmax `exp` is 1280 elements, and its real cost is matmul (D3a)
and the optimizer's cheap arithmetic. The ops this increment makes 3-5x faster
are the ones a *scientific* workload is made of — `nd_exp` over a grid, `log`
in a likelihood, `power` in a kernel — plus `tanh`/`sigmoid` networks, where
the activation is above the threshold.

### N35 — a parallel body that throws

`nd_left_shift` raises on a shift count outside 0..63, element-wise, so the
throw comes from inside the loop. An exception escaping a worker thread is
`std::terminate`: a crash, where the sequential path raises a catchable Bantu
error. So the pool catches per chunk, keeps the exception from the **lowest
chunk index** and rethrows it on the caller's thread after releasing the pool.

Three details, each the answer to a way of getting it wrong:

* **By index, not by arrival.** The sequential loop fails on the first
  offending *element*, which is in the lowest offending chunk, so this is the
  error the single-threaded run would have raised — and the message a user sees
  does not depend on which thread got there first.
* **Remaining chunks are abandoned** (an atomic flag checked before each claim),
  matching the sequential path, which also stops at the first failure.
* **The pool is released before the rethrow.** A failure that left the pool
  claimed would make every later region in the process run single-threaded —
  a performance bug invisible to any test of the answers.

Today no threaded kernel throws: everything that can is in the CHEAP class and
runs sequentially. The property is still implemented and still tested, because
the next threaded kernel to be written (bkit's histogram builder, ecosystem
stage 3) will be the first to need it, and `bantu-src/compiler/tests/parallel_test.cpp`
asserts it directly in C++ — the error path, the lowest-index rule, nesting,
twelve concurrent callers, and the pool still working afterwards. Properties
that no Bantu program can reach yet get a C++ test rather than a comment.


---

### N36 — a threaded reduction is bit-identical to the sequential one, and N26 was wrong

`Pairwise` is a binary counter: after a multiple of `2^m` adds, level `m` holds
the sum of the last `2^m` values **associated as a perfect subtree**, and every
level below `m` is empty, because reaching level `m` takes `2^m` adds.

Three consequences, which together are the whole design:

1. A chunk of exactly `2^m` blocks, summed independently with the same
   algorithm from an empty counter, produces *exactly* the double the global
   tree would hold at level `m` for that range. Same association, same rounding,
   same bits.
2. Injecting those chunk values back at level `m` in ascending chunk order
   rebuilds the global tree's state exactly — one `addLevel(m, v)` that mirrors
   `add` with the counter shifted by `m`.
3. The remainder — always fewer than one chunk, including the sub-128 element
   tail — is then added sequentially, and its carries cannot reach level `m`.

So the answer does not change. Not "changes only in the last bits", not
"changes once, announced": **it is the same double**, at every thread count,
for every size. `bantu-src/compiler/tests/pairwise_partition_test.cpp` checks
1813 size/chunk combinations — chunks from 1024 to 65536 elements, sizes landing
on, just before and just after chunk boundaries, with tails of 0, 1, 127, 128,
129 and 1000 elements — and all 1813 agree bit for bit.

Axis reductions are easier still and need none of this: each output element has
its own accumulator, so threading over OUTPUT elements is disjoint-write
parallelism with no combine step at all. `nd_min`/`nd_max` are exact, and
`nd_argmin`/`nd_argmax` keep lowest-index tie-breaking by combining chunk
results in ascending order.

What this costs is the alignment constraint: the chunk must be `128 * 2^m`
elements (the 128 is the existing block loop's stride), and only *whole* chunks
go to the pool. At `m = 9` that is 65536 elements per chunk, which is the right
order of magnitude for a region anyway (N34 found 16384 is where a region
starts paying for itself even with transcendental work in it).

The lesson is not about summation. It is that **"this is impossible, so we will
change the user's answer" deserves a proof, not an argument** — and the proof
is cheap compared to the apology.