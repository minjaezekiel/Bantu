# numba — Design Decisions

Rationale for the choices behind Numerical Bantu, so they aren't re-litigated. Each entry:
**decision · why · alternatives rejected · implication.**

Full reasoning with measurements lives in [`docs/numba-architecture.md`](../docs/numba-architecture.md).

---

### N1 — Native atoms + pure-Bantu library (arctic's A1, reapplied)
**Decision:** Add native *primitives* (an `NdArray` type, `nd_*` kernels); write `numba` (the API,
defaults, composed helpers) in pure Bantu on top.
**Why:** the interpreter costs ~1 µs per element (1M-iteration loop: 196 ms built-in harness,
1,169 ms pure Bantu) and a `Value` is ~190 bytes carrying a string, a vector, a `std::function` and
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
