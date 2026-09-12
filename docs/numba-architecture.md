# numba — architecture

**Numerical Bantu: NumPy-class array computing, as a native `NdArray` layer with a pure-Bantu
library on top.**

This is the reference design. It is written before the code and the code follows it. Where a
decision was contested, the rejected alternative and the reason are recorded here rather than lost.

Companion documents: [`numba-suite/ROADMAP.md`](../numba-suite/ROADMAP.md) (phases and gates),
[`numba-suite/DECISIONS.md`](../numba-suite/DECISIONS.md) (the numbered decision record),
[`docs/numba.md`](numba.md) (the user-facing reference).

---

## 1. Why this shape, and not another

### 1.1 A pure-Bantu inner loop cannot do numerics

Bantu is a tree-walking interpreter. `Evaluator::evalNode` is a chain of 48 `dynamic_cast`s, there
is no bytecode and no JIT, and `docs/v1.3.0-status.md` records the bytecode VM as deliberately
deferred. The measured cost, from `benchmarks/results.md`:

| workload | Bantu | Node 20 | CPython 3.11 |
|---|---|---|---|
| 1M-iteration arithmetic loop | 196 ms (built-in harness) / 1,169 ms (pure Bantu) | 71 ms | 103 ms |
| list push ×100k | 142 ms | 4.1 ms | 18 ms |

That is roughly **1 µs per element**. A `Value` compounds it: the struct carries a `std::string`, a
`std::vector<Value>`, a `std::function` and three `shared_ptr`s *simultaneously* — about 190 bytes —
so a Bantu list of a million doubles is a million 190-byte objects, pointer-chased, and copying one
list copies all of it.

**Any numeric library whose element loop runs in Bantu is a toy.** This is not a criticism of the
interpreter; it is the same reason NumPy is not written in Python.

### 1.2 The architecture already exists in this repo

`arctic` solved precisely this problem and its solution is recorded as decision A1: *add small,
general native primitives; write the library in pure Bantu on top.* Concretely,
`dataframe_native.hpp` defines a `Column` over real contiguous `std::vector<double>` /
`std::vector<int64_t>` buffers, hands it to Bantu as a refcounted `NATIVE_HANDLE`, and exposes 83
vectorized `col_*` kernels. `arctic/arctic.b` — 1,330 lines of pure Bantu — is a thin façade over
them. It performs: `col_sum` over 5M rows in 70 ms, `col_mul` in 111 ms, `col_filter` in 201 ms, and
500k columns created and dropped leaves RSS flat.

numba is that pattern applied to n-dimensional arrays. The feature-probe (`has_native`), the
`NATIVE_HANDLE` RAII lifetime, the `verb_noun` builtin naming, the suite-docs discipline and the
phase-gate rule (A7) are all reused unchanged.

### 1.3 What was missing

1. **No n-dimensional anything.** `Column` is strictly 1-D. No shape, ndim, stride, reshape,
   transpose, matmul or broadcasting existed anywhere in the tree.
2. **No operator dispatch.** `evalBinaryOp` switched on the token and read `numberVal` directly, so
   `$a * $b` on a handle silently yielded **0**, `$a[i]` yielded **null**, and `$a[i] = v` threw.
3. **No transcendental kernels**, and barely any scalar maths: no `exp`, `atan2`, `asin`, `acos`,
   `log10`, `PI` or `E`; `max`/`min` took exactly two arguments.
4. **Nothing printed usefully.** A `NATIVE_HANDLE` stringified to nothing.

---

## 2. The array

### 2.1 A new type, not an extended `Column`

**Decision: a new `NdArray` in `ndarray_native.hpp`, namespace `numba`, handle tag `"ndarray"`.**

Three reasons, in order of weight:

1. **`Column` structurally cannot support views.** It stores data in `std::vector<double> f64`, and a
   `std::vector` owns its allocation — there is no portable way to make one alias memory it does not
   own. Zero-copy `reshape`/`transpose`/`slice`, the single property that makes NumPy NumPy,
   requires separating *buffer ownership* from *array metadata*. That is a different struct, not a
   field added to this one.
2. **The null mask has no NumPy analogue and would have to become strided.** `Column::valid` is a
   `vector<uint8_t>` parallel to the data; a transposed view of a masked array needs a transposed
   *mask* view too, doubling every stride computation and every kernel's inner loop.
3. **All 83 `col_*` kernels assume `n` is the whole extent.** Adding shape would mean auditing every
   one for ndim > 1 correctness, on a production interpreter with a green suite — exactly the risk
   decision B2 avoided when it chose logical overlays over new `DType` values.

Secondarily, the two libraries have divergent dtype needs — arctic needs `UTF8` and numba never
will; numba may want `F32` and arctic never will — and coupling them means a Parquet change can
break a matmul.

### 2.2 The struct

```cpp
namespace numba {

static const char* NDARRAY_TAG = "ndarray";

// v1 dtype set, deliberately three:
//   F64  — the workhorse; Bantu numbers are double, so this is the lossless dtype
//   I64  — exact integers: indices, counts, argsort/argmin results, histograms
//   BOOL — one BYTE per element, not a bitmask: a bitmask cannot carry a stride,
//          so boolean views and transposes would be impossible
enum class DType : uint8_t { F64, I64, BOOL };

// Split out from NdArray so that reshape/transpose/slice/flip produce a NEW NdArray
// that SHARES this Buffer — that is the zero-copy view. The shared_ptr keeps the
// base alive as long as any view of it lives.
struct Buffer {
    void*  data   = nullptr;
    size_t nbytes = 0;
    bool   owned  = true;
    std::shared_ptr<void> keepalive;   // keeps a foreign owner alive when we borrow

    explicit Buffer(size_t bytes);                              // 64-byte aligned
    Buffer(void* p, size_t bytes, std::shared_ptr<void> owner);  // borrowed
    ~Buffer();
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
using BufferPtr = std::shared_ptr<Buffer>;

struct NdArray {
    BufferPtr              buf;              // shared with every view of it
    DType                  dtype  = DType::F64;
    size_t                 offset = 0;       // in ELEMENTS from buf->data
    std::vector<size_t>    shape;            // ndim == shape.size(); {} is a 0-d scalar
    std::vector<ptrdiff_t> strides;          // in ELEMENTS, SIGNED (negative = flipped)
    bool                   writable = true;  // false for broadcast_to and borrowed buffers

    size_t ndim() const;
    size_t size() const;
    bool isCContig() const;                  // COMPUTED, never cached — see 2.3
    bool isFContig() const;
};
using ArrayPtr = std::shared_ptr<NdArray>;

}
```

64-byte buffer alignment lets the contiguous kernels vectorize without a scalar peel prologue, and
guarantees two arrays never share a cache line.

### 2.3 Three choices worth defending

**Contiguity is computed, not cached.** A cached flag must be recomputed by every operation that
touches `shape` or `strides`; forgetting one leaves a stale `true`, which sends a kernel down the
raw-pointer fast path over strided data — **silent memory corruption with no crash**. Computing it
is `ndim` integer comparisons, under 20 ns, against a kernel that then touches millions of elements.
This is not a place to trade correctness for nanoseconds.

**Strides are in elements, not bytes.** NumPy uses bytes. With one itemsize per array the two are
equivalent, and elements removes an entire class of "forgot to multiply by 8" bugs while making
`ptr<T>()[i * stride]` read naturally. This stays correct if `F32` is ever added.

**There is no validity mask.** NumPy's answer for f64 is NaN and for integers is "no nulls", and
`nd_isnan` is the null test. Nulls remain arctic's concern; the bridge (§6) refuses a column that
still has any.

**Complex is not in v1.** It doubles every ufunc, Bantu's `Value` has no complex scalar so every
materialization would need a list-or-dict convention, and its only compelling consumers — FFT and
general `eig` — are already out of scope. The struct is forward-compatible: add `C128` to `DType`,
have `itemsize` return 16, and only the kernels that opt in need new arms.

### 2.4 File layout, and why it is split

`ndarray_native.hpp` is the implementation header and is included by **exactly one** translation
unit, `ndarray_native.cpp`. `evaluator.hpp` sees only a small `ndarray_api.hpp` carrying
`registerBuiltins(DefineFn)` and the dispatch hooks.

This buys three things. `evaluator.hpp` is already 8,883 lines and gains about fifteen rather than
two thousand. Per-TU compile memory stays bounded, a stated concern in `build.sh`. And — the reason
that actually forced it — **it lets the kernel TU be compiled at `-O3` while everything else stays
at `-O2`** (§4.4).

Registration is a callback so the header never needs `makeNative` or `ErrorHandler`, and so every
builtin gets the same treatment uniformly:

```cpp
numba::registerBuiltins([this](const char* name, NativeFn fn) {
    env_->define(name, makeNative(
        [name, fn = std::move(fn)](std::vector<Value> a) -> Value {
            try { return fn(std::move(a)); }
            catch (const std::exception& e) {
                ErrorHandler::throwError(std::string(name) + ": " + e.what(),
                                         0, 0, ErrorHandler::RUNTIME_ERROR);
            }
            return Value();
        }));
});
```

Every bad argument therefore becomes a **catchable Bantu error, never a process kill**. The `sua.udp`
review is the cautionary precedent: two of its four defects killed the process outright because one
unvalidated argument reached an allocation.

---

## 3. The kernel surface

About 150 builtins, `nd_` prefixed, matching arctic's `col_` convention and decision A5's
"self-documenting `verb_noun`, greppable, no builtin collisions".

| group | contents |
|---|---|
| **creation** | `nd`, `nd_zeros/ones/full/empty(_like)`, `nd_arange`, `nd_linspace`, `nd_eye/identity/diag`, `nd_random_uniform/normal/int/permutation`, `nd_seed`, `nd_copy`, `nd_ascontiguous`, `nd_astype` |
| **introspection** | `nd_shape/ndim/size/dtype/strides/itemsize/nbytes/is_view/is_contiguous/writable/base_id` |
| **shape (views)** | `nd_reshape`, `nd_transpose/T`, `nd_ravel`, `nd_swapaxes/moveaxis`, `nd_expand_dims/squeeze`, `nd_slice`, `nd_broadcast_to`, `nd_flip`, `nd_split` |
| **shape (copies)** | `nd_flatten`, `nd_concatenate/stack/vstack/hstack`, `nd_repeat/tile/pad`, `nd_triu/tril` |
| **ufuncs** | arithmetic; `sqrt cbrt exp expm1 log log1p log2 log10 sin cos tan asin acos atan atan2 hypot sinh cosh tanh asinh acosh atanh floor ceil trunc round rint sign reciprocal square isnan isinf isfinite clip where minimum maximum copysign`; comparisons and boolean logic; `nd_isclose/allclose/array_equal` |
| **reductions** | `sum prod mean std var min max ptp argmin argmax any all count_nonzero median quantile`, the `nan*` variants, `cumsum cumprod cummax cummin diff` — all with `axis` and `keepdims` |
| **indexing / search** | `nd_get/set/index/take/put/mask/compress/nonzero`, `nd_sort/argsort/partition/argpartition/searchsorted/unique/unique_counts/bincount/histogram` |
| **linear algebra** | `nd_dot/matmul/inner/outer`, `nd_solve/inv/det/slogdet/lu/solve_triangular/cholesky/cho_solve/qr/lstsq/pinv/norm/trace/matrix_rank/cond`, `nd_svd`, `nd_eigh` |

Every binary ufunc takes an optional trailing `out` array. Every reduction takes `axis` (null, a
number, or a list) and `keepdims`.

**Promotion rule, stated once and applied everywhere:** `bool → i64 → f64`. `i64 ⊕ i64` stays exact
`i64` for `+ - * // %`, mirroring arctic's integer path; `/`, `pow` and every transcendental always
produce `f64`; comparisons always produce `bool`.

**numba owns its own PRNG** — xoshiro256++, seeded explicitly — rather than the global `random()`
builtin. Reproducibility is non-negotiable for numerics, and a shared stream means an unrelated
`random()` call elsewhere in the program silently changes your matrix.

**`sum` and `mean` use pairwise summation; `std` and `var` use Welford.** A naive f64 accumulator
over 10M values loses roughly six significant digits. Users will diff against NumPy, which does
pairwise, and the phase-3 gate (`sum` of 10M copies of `0.1`, relative error < 1e-12) fails a naive
implementation by design.

### 3.1 Linear algebra: what is hand-written, and where it stops

Hand-written, each 40–150 lines of textbook numerics: blocked matmul (64×64 tiles, `ikj` inner order
so B streams at unit stride), blocked copy-transpose, LU with partial pivoting (which yields `det`,
`solve`, `inv`, `slogdet`, `matrix_rank`), forward/back triangular substitution, Cholesky and
`cho_solve`, Householder QR and `lstsq`, norms, trace and condition number.

Two deliberately "easy algorithm" choices:

- **`eigh` by cyclic Jacobi** — about 100 lines, unconditionally convergent, orthogonality to machine
  precision. Slower than tridiagonal QR, and bulletproof.
- **SVD by one-sided Jacobi** — about 120 lines against roughly 500 for Golub–Kahan implicit-QR, no
  convergence tuning, and *higher* relative accuracy on small singular values. It costs several
  sweeps, so it is perhaps 3–5× slower than LAPACK's `dgesdd` and impractical much beyond n ≈ 500–800.
  That limit is acceptable and is documented rather than hidden.

**Not shipping, and `docs/numba.md` says so plainly:** general nonsymmetric `eig` (balancing +
Hessenberg reduction + Francis double-shift QR + complex eigenvalue extraction is a research-grade
implementation needing complex arithmetic we do not have — a fragile `eig` is worse than none, and
`eigh` already covers covariance matrices, PCA and graph Laplacians), plus `expm`, `schur`, FFT and
sparse.

### 3.2 Why not BLAS

The two existing optional build flags gate **capability**: without libsodium you cannot do Argon2id
at all, without Arrow you cannot read Parquet at all. Both are things that genuinely cannot be
written in-tree, and `has_native()` can answer yes or no about them meaningfully.

A `BANTU_BLAS` flag would gate **speed**. That creates a two-tier contract in which the same program
takes 0.3 s on the maintainer's build and 2 s on the binary users download, with nothing useful to
feature-detect. Add the Fortran ABI, the LP64/ILP64 integer-width split, and the fact that Ubuntu's
default `-llapack` is unoptimized reference netlib that is frequently *slower* than a decent blocked
kernel.

The honest counter-argument is that anyone doing 4000×4000 linear algebra will find nothing
hand-written acceptable. So this is **deferred, not closed**. If it is ever taken up: gate it as
`has_native("blas")`, swap only `gemm`/`gesv`/`gesdd` behind identical builtin names so behaviour is
byte-identical and only speed changes, and prototype on macOS, where Accelerate.framework ships with
the OS and the install burden is zero.

---

## 4. Broadcasting, the kernel loop, and the compiler

### 4.1 Shape broadcasting

NumPy semantics: right-align the shapes; each pair must be equal or one of them 1. A stretched axis
is given **stride 0**, so stepping along it re-reads the same element — that zero stride *is*
broadcasting, and no data is ever copied.

Error messages name the offending axis and both extents. This is the most common error in array
code, and a message that says only "shape mismatch" is a permanent support burden:

```
nd_add: shapes [3,4] and [5,4] cannot be broadcast together (axis 0: 3 vs 5)
```

### 4.2 Three tiers

Checked in order:

- **Tier 0** — same shape, all C-contiguous. One flat `__restrict` pointer loop. This is over 90% of
  real calls and the only loop the compiler must vectorize.
- **Tier 1** — one operand is a scalar or 0-d. Contiguous loop with a splat.
- **Tier 2** — general strided. **Coalesce dimensions first**, then an N-d odometer with four
  specialized inner loops and the branch hoisted outside the hot loop.

Dimension coalescing is the part that gets skipped and it matters more than SIMD: merge adjacent
axes when `stride[i] == stride[i+1] * shape[i+1]` *in every operand simultaneously*, and drop
extent-1 axes. It collapses a `(1000,1000)` operation that reached tier 2 back into a single
1e6-iteration flat loop. NumPy's `nditer` does exactly this; without it an n-d element-wise op runs
an odometer step per row and loses 2–3×.

### 4.3 What performance to expect, honestly

For `c = a + b` on 10M f64: each array is 80 MB, the loop moves 2 reads + 1 write ≈ 240 MB, plus
read-for-ownership on the write, so roughly 320 MB of DRAM traffic for 10M flops. That is **0.042
flops per byte** — about two orders of magnitude below any modern machine's balance point. **This
loop is DRAM-bandwidth-bound, not ALU-bound.**

| machine | streaming bandwidth | expected |
|---|---|---|
| Apple M-series, single core | 60–100 GB/s | 4–7 ms |
| modern x86 desktop, single core | 20–30 GB/s | 11–16 ms |
| shared cloud vCPU | 5–10 GB/s | 30–60 ms |

NumPy on the same desktop takes ~8–15 ms, because it waits on the same memory. Add 1–3 ms of
first-touch page faults on the freshly allocated 80 MB output.

**SIMD, OpenMP and threading are therefore not needed for the headline operation**, and are
deferred. What SIMD does buy, and where it will eventually be spent: L2-resident arrays (2–4×),
transcendentals (3–5×), matmul (5–15×).

### 4.4 One finding that invalidates naive benchmarking

The production Linux binary is built inside `ubuntu:22.04` with `-O2 -mtune=generic`. Ubuntu 22.04
ships **GCC 11, which does not enable `-ftree-loop-vectorize` at `-O2`** — GCC 12 does.
`build-mac.sh` also uses `-O2`, but that is Apple Clang, which *does* vectorize at `-O2`.

So a kernel written and benchmarked on macOS gets NEON, and the identical kernel in the binary
Linux users download gets a scalar loop. **Any number measured on the Mac is not the number users
get**, and no benchmark gate is meaningful until this is closed.

The fix is to compile `ndarray_native.cpp` alone at `-O3 -ftree-vectorize`, via a per-file flag
override in each build script. That is what the separate-TU layout in §2.4 exists to enable.

Rejected along the way:

- **`#pragma GCC optimize("O3")` / `__attribute__((optimize))`** — GCC documents the function-level
  attribute as unsuitable for production (it silently disables some IPA analysis and is a known
  miscompile source), and Clang ignores the pragma entirely, so it would fix nothing on macOS while
  risking Linux.
- **`-march=native`** — `CMakeLists.txt` documents why not: portability to Render's free tier and
  older CPUs. Baseline x86-64 `-O3` gives SSE2 (2 doubles per vector); arm64 gives NEON (2 doubles).
  Given §4.3 that is enough. Runtime dispatch via `__builtin_cpu_supports` is the right answer if
  matmul or transcendentals ever prove ALU-bound.
- **OpenMP** — links libgomp, a new runtime dependency, against the house rule that produced a
  from-scratch P-256 rather than linking OpenSSL and rejected libuv for sua. It also needs
  `brew install libomp` for every macOS contributor, and buys ~1.5–2.5× on a bandwidth-bound loop.

**Verify, do not assume:** CI compiles that TU with `-fopt-info-vec-optimized` and greps the log for
the tier-0 loops, exactly as `ci.yml` already greps for `event loop: epoll` to prove Linux got the
right backend. Without that check, a silent regression to scalar code is invisible.

### 4.5 Where we will be visibly behind

`nd_exp` over 10M elements via scalar `std::exp` is roughly 60–120 ms against NumPy's 25–40 ms,
because NumPy ships a vectorized exp. The plan is to ship libm first — obviously correct — and later
add Cephes-style reduce-and-polynomial `exp`/`log`/`sin`/`cos`, which auto-vectorize because they
are branch-free. That lands only behind a differential gate asserting ≤ 1 ULP against libm across a
dense sample of the domain including denormals, ±inf, NaN and the overflow boundaries.

---

## 5. The Bantu-facing design

### 5.1 The handle is the object

arctic wraps its column handle in a Bantu `Series` class. numba does not. The raw `NATIVE_HANDLE`
*is* the user-facing array, and five small additive arms in the evaluator make it behave like one:

| site | before | after |
|---|---|---|
| `evalBinaryOp` | read `numberVal` → silently 0 | dispatch `+ - * / % == != < <= > >=` |
| `evalUnaryOp` | `-$a` → `-0` | dispatch unary `-` |
| `evalIndexAccess` | fell through → null | dispatch to `nd_index` |
| `evalIndexAssign` | threw "Cannot index-assign to this type" | dispatch to `nd_set` |
| `evalDotAccess` | fell through → null | bound native closures for method names |
| `Value::toString` | printed nothing useful | a NumPy-style repr |

The dot-access arm has direct precedent in the same file: dict pseudo-methods (`$d.keys()`,
`$d.items()`) and number pseudo-methods (`.floor()`, `.ceil()`, `.round()`) already work exactly
this way.

**Why this beats a Bantu class wrapper.** The class route would need full `CLASS_INSTANCE` dunder
metamethods — a permanent language commitment — purely to get `+`. The handle-is-the-object design
gets `+`, `[]`, `.method()` and printing from five narrow additive changes with **zero new language
semantics**.

### 5.2 The dispatch, and its cost

```cpp
Value evalBinaryOp(BinaryOpNode* n) {
    Value left  = evalNode(n->left);
    Value right = evalNode(n->right);

    // Fast reject. The overwhelmingly common case is two plain numbers; this is ONE
    // predictable branch on a byte already in L1. When it fires we fall THROUGH to
    // the existing switch, so string +, ==/!= on any type, and &&/|| are untouched.
    if (__builtin_expect(left.type != NUMBER || right.type != NUMBER, 0)) {
        if (left.isNativeHandle() || right.isNativeHandle()) {
            Value out;
            if (numba::dispatchBinary((int)n->op, left, right, out)) return out;
        }
    }
    switch (n->op) { /* unchanged */ }
}
```

The added work in the common path is one integer compare and one well-predicted branch — roughly
0.3–1 ns, against a measured ~1 µs per interpreted loop iteration. That is 0.03–0.1%, below the
benchmark noise floor. For scale: constructing the returned `Value` is already far more expensive,
because it default-constructs a `std::string`, a `std::vector<Value>`, a `std::function` and three
`shared_ptr`s on every arithmetic result.

That reasoning is an estimate, so **the phase gate is a measurement**: `benchmarks/run.sh`, five
iterations before and after, required within **±2%**, with both numbers recorded in the CHANGELOG.

**Backward compatibility.** Before this change, `$handle + 1` read `handle.numberVal`, which is
always 0, and silently produced `1`. There was exactly one handle tag in existence (`"column"`) and
`arctic.b` never places a column in an arithmetic expression — it calls `col_add`. Every path being
filled in was dead. The same holds for `$a[i]` returning null and `$a[i] = v` throwing.

**Rejected: class dunder metamethods.** They are a language feature and deserve to be designed as
one coherent thing — `__add__`, `__radd__`, `__sub__`, `__mul__`, `__eq__`, `__lt__`, `__getitem__`,
`__setitem__`, `__len__`, `__str__` and the reflected forms — with their own tests, docs and
benchmark gate. Bolting half of it onto a numeric library is how a language ends up with `__add__`
and no `__radd__` forever. And if they are ever added: **dunder names only**. Dispatching `+` to a
method named `add` would silently hijack `+` for any class that happens to have one — a `Set`, a
`Counter`, a `ShoppingCart`.

### 5.3 A consequence worth documenting loudly

`$m[1][2] = 9.5` works **without any `resolveLValue` support**. For a handle, the "copy"
`evalIndexAssign` makes is a `shared_ptr` to the same view over the same buffer, so the write lands
in the base array. Handles have reference semantics; Bantu lists do not. This is the opposite of
list behaviour and will surprise people in both directions, so it is stated prominently in
`docs/numba.md` rather than left to be discovered.

### 5.4 The façade is the public API

The `nd_*` builtins are the atoms. `numba/numba.b` is what users are expected to call.

This is not cosmetic. **The façade can have optional arguments and the builtins cannot.** Bantu binds
missing arguments to null, so a Bantu-level `def arange($start, $stop, $step)` called as
`np.arange(0, 10)` works, while the raw builtin needs `nd_arange(0, 10, null)`. Every builtin gets a
wrapper supplying sane defaults; `numba.b` additionally holds what is genuinely *composed* from the
atoms — `polyfit`/`polyval`, `interp`, `gradient`, `cov`/`corrcoef`, `meshgrid`, moving averages —
plus `help()` and `info()`.

```bantu
include "numba" as np;

$a = np.arange(0, 12);              // no trailing null
$m = np.reshape($a, [3, 4]);        // zero-copy view
print($m);                          // [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
                                    // shape=[3,4] dtype=i64
print($m.sum(0));                   // [12, 15, 18, 21]

$x = np.linspace(0, 1, 1000000);
$y = ($x * $x + $x.sin()).sqrt();
print($y.mean());
```

Chaining (`$x.mul($x).add($x.sin()).sqrt()`) works from phase 1 via the dot arm and remains the
permanent fallback.

### 5.5 Where the abstraction costs something

Every builtin call goes through `evalCall`, which builds a `std::vector<Value>` and copies each
~190-byte `Value` into it — roughly 1–3 µs per call. That is invisible beside a kernel touching 10M
elements (≈10 ms) and *dominant* for a 100-element array (kernel ≈ 50 ns).

**So numba is for arrays of about 10,000 elements and up.** Below that, a plain Bantu list is simpler
and not meaningfully slower. `docs/numba.md` says this outright.

The other real cost is temporaries: `$a.add($b).mul(2).sqrt()` on 10M f64 peaks at three live 80 MB
buffers. Mitigated by the `out=` parameter on every binary ufunc and an explicit `nd_release($a)`,
both documented in a "working with large arrays" section.

---

## 6. Interop with arctic

**`Column → NdArray` is genuinely zero-copy and read-only.** A `Buffer` borrows the column's vector
storage and holds the `ColumnPtr` alive through `keepalive`. Preconditions are checked and reported
by name: dtype F64/I64/BOOL (not UTF8), `null_count == 0` (numba has no validity mask, so nulls must
be filled first), and `logical == NONE`. The result is `writable = false`, because arctic documents
columns as immutable and that should be honoured rather than quietly broken.

**`NdArray → Column` copies.** `Column` stores a `std::vector` and there is no portable way to make
one adopt a foreign pointer. One `memcpy`; 80 MB is 10–20 ms. NaN → null is opt-in and off by
default, because doing it silently loses information.

**`nd_from_frame(cols)`** is the most-wanted bridge — a frame to linear algebra or ML. It is built
**F-contiguous**, so each column is a single `memcpy` rather than a strided scatter, and every kernel
already handles that via `isFContig()`.

The glue lives in `evaluator.hpp`'s builtin block, the one place that legitimately sees both
namespaces, so `dataframe_native.hpp` and `ndarray_native.hpp` never include each other.

**arctic gets its own transcendental kernels; it does not delegate.** The ~15 missing `col_*`
transcendentals are about ten lines each on the existing `unaryOp` shape with null-mask propagation.
Delegating them to numba would make `series.sqrt()` require a numba-capable build and would lose null
semantics, since NaN is not null. The bridge exists for what arctic genuinely cannot do — matmul,
`solve`, `svd`, reductions over a 2-D block.

---

## 7. Security model

A numeric library's attack surface is unvalidated input reaching an allocation or an index. Four
hard requirements, each gated by tests from phase 1:

- **Shape-product overflow.** `nd_zeros([2^22, 2^22, 2^22])` overflows `size_t` to a small number,
  allocates a tiny buffer, and every subsequent kernel writes past it — a heap overflow reachable
  from one line of user script. Every shape product uses checked multiplication.
- **Allocation denial of service.** `nd_zeros([1e15])` must raise rather than OOM-kill the process.
  This matters most when numba runs inside a sua request handler, where the process is a server. A
  configurable ceiling, default ~2 GB.
- **Stride-0 writes.** A `broadcast_to` view has stride 0 on stretched axes, so writing through it
  silently corrupts — every write lands on the same element. Broadcast results carry
  `writable = false`, enforced in `nd_set`, `nd_put` and every `out=` path.
- **Bounds and aliasing.** Every index is checked against its own axis extent. `out=` aliasing is
  detected by buffer identity plus element-extent overlap and then copied, which is NumPy's
  behaviour, rather than producing garbage.

These are the same class of defect as the `sua.udp` `maxBytes` crashes — one unvalidated argument
sizing an allocation — and only adversarial tests find them.

---

## 8. Testing

Nothing is done until it is tested **and stress tested** on a real build. Five tiers, all green
before any phase advances, with numbers recorded in `numba-suite/CHANGELOG.md` (decision A7).

| tier | proves | form |
|---|---|---|
| feature | every builtin does what it claims, including its failure modes | `tests/numba_*_test.b`, ending `RESULT: ALL GREEN` |
| differential | the answers are *right*, not merely stable | NumPy reference values embedded in the tests; libm for transcendentals; pure-Bantu loops for small cases |
| stress | it survives size, repetition and abuse | scale, RSS, adversarial input, aliasing, numerical stress, concurrency |
| regression | nothing else broke | the whole existing suite on the same build |
| sanitizers | no latent memory defect | ASan + UBSan over the full numba suite |

Stress specifically means: 10M-element arrays for every ufunc and reduction with wall-clock gates;
200k arrays created and dropped with RSS measured before and after; every builtin fed negative sizes,
zero, null, wrong types, wrong arity, NaN, ±inf, empty and 0-d arrays, 2^53, and overflowing shape
products, where the pass condition is a catchable error with a useful message and never a crash;
`out=` overlapping its inputs; writes through transposed, flipped and stride-0 views; a view
outliving its base; sum of 10M copies of `0.1`; singular and ill-conditioned matrices; and numba
called from inside a sua handler under load.

**One testing rule is absolute: never assert on a stringified number.** `str()` gives six significant
digits and leaks scientific notation (`str(0.000012345678)` → `"1.23457e-05"`). Numeric assertions
use `nd_allclose`.

**One coding rule is absolute: use `while`, never the C-style `for`, above 100,000 iterations.**
`evaluator.hpp` caps it at `safety < 100000` and exits **silently**, so a benchmark looping 200k
times reports a wrong-but-plausible number and passes. One test asserts the cap explicitly so it is
documented rather than rediscovered.

---

## 9. Adoption

A library nobody can pick up in five minutes has failed regardless of its benchmarks. These are
requirements, not polish.

1. **The façade is the API**, the `nd_*` builtins are plumbing (§5.4).
2. **`print($a)` prints the array** — nested brackets, aligned columns, edge-truncation past ~1000
   elements, and a `shape=[3,4] dtype=f64` summary. This is the single feature that makes array
   programming feel approachable, and it fixes arctic's columns at the same time.
3. **Three lines to a first result**, and they open the documentation.
4. **`include "numba" as np;` resolves after `bantu add numba`.** It did not: the installer writes to
   `./bantu_modules/<name>/` and the module resolver had no rule for that directory. Fixed, which
   fixes `arctic`, `hash`, `crypto`, `uuid`, `random` and `orm` too.
5. **Including a package twice does not silently break.** The include cycle guard returned *before*
   binding the alias, so two modules both including one package left the second with an unbound
   alias and a stderr line. Fixed.
6. **Names people already know** — `zeros ones arange linspace reshape sum mean std argsort dot
   matmul`. Anyone who has used NumPy should be able to guess and be right.
7. **Errors teach** — what was expected, what arrived, and the fix.
8. **Discoverability inside the language** — `np.help()`, `np.help("solve")`, `np.info($a)`.
9. **Progressive disclosure** — `np.array([1,2,3])` works knowing nothing about dtypes, strides,
   views, broadcasting or `out=`.
10. **A gallery, not just a reference** — `samples/numba/`, with every example executed by CI so none
    of them can rot.

---

## 10. Rejected alternatives

| rejected | why |
|---|---|
| Extend `arctic::Column` with shape and strides | `std::vector`-owned storage makes zero-copy views structurally impossible; the null mask would have to become strided; all 83 `col_*` kernels would need a 1-D-correctness audit on a production interpreter |
| Wrap the array in a Bantu class | Would require full `CLASS_INSTANCE` dunder metamethods — a permanent language commitment — just to get `+` |
| Strides in bytes | With one itemsize per array, elements is equivalent and removes a class of "forgot ×8" bugs |
| A cached contiguity flag | A stale `true` sends a kernel down the raw-pointer fast path over strided data: silent corruption, no crash |
| A validity mask on `NdArray` | No NumPy analogue; needs a parallel strided mask per view; NaN already serves |
| Complex dtype in v1 | Doubles every ufunc; no complex `Value`; its consumers are already out of scope |
| OpenMP | New runtime dependency; `brew install libomp` for every contributor; ~1.5–2.5× on a bandwidth-bound op |
| `-march=native` | Portability to older CPUs is a documented project constraint |
| `#pragma GCC optimize("O3")` | Documented as unsuitable for production by GCC; ignored entirely by Clang |
| `BANTU_BLAS` in v1 | Existing flags gate capability; this would gate speed, creating a two-tier contract nobody can reason about. Deferred, not closed |
| General nonsymmetric `eig` | Research-grade, and needs complex arithmetic we do not have. A fragile `eig` is worse than none |
| Golub–Kahan implicit-QR SVD | ~500 lines with real convergence tuning; one-sided Jacobi is ~120, unconditionally convergent, and more accurate on small singular values |
| Reusing `add`/`sub` as operator metamethods | Would silently hijack `+` for any class with an `add` method |
| Registering 150 builtins inline in `evaluator.hpp` | That file is already 8,883 lines; the callback design keeps the table in its own TU and is what makes the `-O3` compile possible |
