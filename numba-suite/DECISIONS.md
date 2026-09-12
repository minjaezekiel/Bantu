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
