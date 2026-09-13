# numba — Changelog

Every entry is prefixed so the log can be grepped: **[design]**, **[feature]**, **[bug fix]**,
**[perf]**, **[test]**, **[docs]**.

Phases do not advance until all five gate tiers are green (see
[`ROADMAP.md`](ROADMAP.md)). Measured numbers are recorded here, with the machine they were taken
on, because a number without a machine is not a measurement.

---

## Phase 0 — Tracking docs

- **[design]** `docs/numba-architecture.md` — the reference design: why a native `NdArray` layer with
  a pure-Bantu library on top is the only shape that works (the interpreter costs ~1 µs per element
  and a `Value` is ~190 bytes); the struct with buffer ownership split from array metadata so
  views are zero-copy; the ~150-builtin kernel surface; broadcasting and the three-tier kernel loop;
  the operator-dispatch design with its hot-path analysis; the security model; the arctic bridge;
  the testing regime; adoption requirements; and every rejected alternative with its reason.
- **[design]** `numba-suite/DECISIONS.md` — N1–N16.
- **[design]** `numba-suite/ROADMAP.md` — phases 0, A, 1–7, each with its feature test, its stress
  test and its numeric gate.

### Findings recorded during design, before any code

- **The shipped Linux binary does not vectorize.** It is built in `ubuntu:22.04` with `-O2`, and GCC
  11 does not enable `-ftree-loop-vectorize` at `-O2` (GCC 12 does). `build-mac.sh` also uses `-O2`,
  but that is Apple Clang, which does. So a kernel benchmarked on macOS gets NEON while the same
  kernel in the binary users download gets a scalar loop — **any Mac-measured number is not the
  number users get.** Closed by compiling `ndarray_native.cpp` alone at `-O3 -ftree-vectorize`
  (N11), with a CI grep of `-fopt-info-vec-optimized` so a silent regression to scalar is visible.
- **`include "numba" as np` would not have worked.** `bantu add` installs to `./bantu_modules/<name>/`
  and `module_resolver.hpp` has no rule for that directory. Latent for every existing package.
  Fixed in Phase A (N16).
- **A second `include` of the same file binds nothing.** The cycle guard returns *before* defining
  the alias, so two modules both including one package leave the second with an unbound alias and
  only a line on stderr. Any real application hits this. Also latent for every existing package.
  Fixed in Phase A (N16).
- **`print($handle)` renders nothing useful** for arctic columns today, and would have done the same
  for arrays. Fixed in Phase A (N16).
- **The C-style `for` loop silently stops at 100,000 iterations** (`evaluator.hpp`,
  `safety < 100000`) with no error. A benchmark looping 200k times reports a wrong-but-plausible
  number and passes. Not numba's to fix, but every numba test and benchmark uses `while`, and one
  test asserts the cap explicitly so it is documented rather than rediscovered.

---

## Phase A — Adoption plumbing ✅

All five gate tiers green: 24 `.b` suites (including the `const_bad` negative), 7 package-local
suites, 9 server suites — 40 in total, on macOS arm64.

### Resolution and binding

- **[feature]** `include "numba" as np;` now resolves inside `bantu_modules/`. `bantu add <pkg>`
  installs to `./bantu_modules/<pkg>/`, but `module_resolver.hpp` had no rule for that directory, so
  an installed package could only be reached by spelling out its full path. Bare names now search
  `bantu_modules/<name>/` next to the importing file and under the cwd, honouring `package.json`'s
  `"main"`, then falling back to `<name>.b`, `index.b`, `main.b`. Explicit relative paths are
  untouched, and the manifest scanner distinguishes a `"main"` key from a value that reads the same.
- **[bug fix]** A repeat `include` of the same file **bound nothing**. The cycle guard returned
  before the alias was ever defined, so two modules both doing `include "arctic" as arctic;` left
  the second with an undefined `arctic` and one line on stderr. Modules are now cached by canonical
  path and every later include binds that same namespace object — module-singleton semantics, as in
  Node. A genuine cycle (still mid-execution, nothing exported yet) is reported by name and skipped.
- **[bug fix]** `sua.include()` had the same defect in a different shape: a second call returned
  `{"_cached": true, "_path": ...}` instead of the module. It now shares the same cache and returns
  the module.

### print()

- **[feature]** A `NATIVE_HANDLE` stringified to `"<column>"`, so `print($col)` told you the type
  and nothing about the data — the only way to look at a column was to materialize it to a Bantu
  list first. `Value::toString` now consults a registry of renderers keyed by handle tag, and arctic
  registers one for columns. Summarization follows NumPy's rule (every element up to 1000, then
  three from each end). 100 reprs of a 200,000-element column take **0 ms** — rendering is O(1) in
  length. numba registers its own renderer in Phase 1.

### Two pre-existing quadratic defects, found by measurement

Neither was introduced by this work: the **shipped release binary** reproduces both.

- **[perf]** `push` deep-copied the whole list on every call. It returned the mutated list, and a
  Bantu list is a `std::vector<Value>` with value semantics where each `Value` is ~190 bytes
  carrying a string, a vector, a `std::function` and three `shared_ptr`s. An O(1) append was O(n),
  and a loop of appends was **O(n²)**:

  | pushes | before | after |
  |---|---|---|
  | 2,500 | 122 ms | — |
  | 5,000 | 420 ms | — |
  | 10,000 | 1,853 ms | — |
  | 20,000 | **9,616 ms** | **70 ms** |
  | 100,000 | ~4 minutes | **271 ms** |

  For comparison the identical `append($l, x)` took 69 ms at 20,000 — a 137× gap. The release binary
  measured 9,314 ms for the same 20,000 pushes, and could not finish `bantu bench`'s own
  "list push 100k" benchmark in ten minutes; it now runs at **250 ms/iter**.

  The fix records on the AST whether a call's result is discarded (`parseExpressionStatement` is the
  one place in the grammar where that is true), so `push($l, x);` as a statement skips the copy
  while `$x = push($x, v)` — a documented, tested idiom — still gets its list. `$l.push(x)` returns
  the new length, as in JavaScript.

- **[perf]** `len($var)` copied its argument. Passing a list to any function copies it, and `len` is
  the one builtin routinely called on the very container being built:

  ```bantu
  while (...) { $out[len($out)] = $v; }    // 20,000 items: 7,027 ms -> 58 ms
  ```

  That idiom appears throughout `hash.b` and `crypto.b`, so their pure-Bantu paths were quadratic in
  input length. `len($var)` now reads the length from the real storage. Every answer is identical —
  list, string, dict (`0`), non-container (`0`), literal argument — and a user-defined `len()` still
  shadows the builtin.

### Hot-path check

The operator and call paths are what Phase 4 will touch, so they were measured before and after
against the shipped release on the same machine. No regression:

| | release | this build |
|---|---|---|
| 1M-iteration arithmetic `while` loop | 2,336 ms | 2,186 ms |
| `fib(24)` recursive | 1,906 ms | 1,941 ms |
| 50k dict set | 171 ms | 178 ms |

### Tests added

`tests/lang_module_test.b` (13), `tests/lang_repr_test.b` (17), `tests/lang_list_test.b` (23), plus
fixtures under `tests/fixtures/` and two fixture packages under `tests/bantu_modules/`. All three
are picked up automatically by CI's `tests/*.b` glob, which is shallow and so does not reach the
fixture directories.

---

## Phase 1 — `NdArray` core, buffers, zero-copy views ✅

All five gate tiers green: 42 suites on macOS arm64 (24 `.b` including the `const_bad` negative, 7
package-local, 10 shell), with `tests/numba_array_test.b` at **108/108** and
`tests/numba_stress.sh` at **9/9**.

### What landed

- **[feature]** `ndarray_native.hpp` (the implementation), `ndarray_native.cpp` (the single TU that
  includes it), `ndarray_api.hpp` (all `evaluator.hpp` sees). Handle tag `"ndarray"`;
  `has_native("ndarray")` reports it.
- **[feature]** 35 `nd_*` builtins: creation (`nd`, `nd_zeros/ones/full/empty` and their `_like`
  forms, `nd_arange`, `nd_linspace`, `nd_eye`, `nd_identity`, `nd_seed`,
  `nd_random_uniform/normal/int`), introspection (`nd_shape/ndim/size/dtype/strides/itemsize/
  nbytes/writable/is_contiguous/is_view/base_id`), shape operations (`nd_reshape`,
  `nd_transpose`, `nd_T`, `nd_ravel`, `nd_flatten`, `nd_swapaxes`, `nd_moveaxis`,
  `nd_expand_dims`, `nd_squeeze`, `nd_slice`, `nd_broadcast_to`, `nd_flip`), element access
  (`nd_get`, `nd_set`, `nd_to_list`), casts (`nd_copy`, `nd_astype`, `nd_ascontiguous`) and the
  allocation ceiling (`nd_max_bytes`).
- **[feature]** `print($a)` renders arrays through the Phase A repr registry, with NumPy's
  summarization rule and a `shape=[3,4] dtype=i64` tail.
- **[feature]** Reproducible randomness on numba's own xoshiro256++ stream, seeded explicitly —
  deliberately *not* the global `random()`, because a shared stream means an unrelated `random()`
  call elsewhere in a program silently changes your matrix.

### Zero-copy views, proven rather than asserted

`reshape`, `transpose`, `T`, `ravel`, `slice`, `flip` and `broadcast_to` all return a new `NdArray`
sharing the base's `Buffer`. The tests prove it two ways: `nd_base_id()` compares the actual buffer
address, and a write through one handle is read back through another. A view keeps its base alive
through the `shared_ptr`, which is checked by returning a slice from a function whose base has gone
out of scope.

- 10M-element allocation: **47 ms**. `reshape` + `transpose` of it: **0 ms**. Strided slice: **0 ms**.
- `nd_is_view` had to be corrected during the work: it first asked only "do I cover the whole
  buffer?", which a transposed view does — so it called a transpose an independent array. It now
  asks whether the buffer is shared, which is the question a user actually has.

### Security, gated from this phase

- **Shape-product overflow.** `nd_zeros([2^22, 2^22, 2^22])` wraps `size_t` to a small number under
  plain multiplication, allocating a tiny buffer that every later kernel writes past — a heap
  overflow reachable from one line of script. Every product uses checked multiplication and raises.
- **Allocation ceiling.** Default 2 GiB, adjustable with `nd_max_bytes(n)`. `nd_zeros([1e15])`
  raises instead of OOM-killing the process, which matters most when numba runs inside a sua handler.
- **Stride-0 writes.** `broadcast_to` returns `writable = false`; `nd_set` through it raises with an
  explanation rather than silently hitting one element repeatedly.
- **Bounds.** Every index is checked against its own axis, and the message names the axis, its
  extent and the offending value.
- Every builtin is wrapped so a bad argument raises a **catchable Bantu error naming the builtin**.
  19 adversarial cases are asserted individually, each also checking the message mentions the thing
  that was wrong.

### Stress

| | result |
|---|---|
| 200,000 arrays created and dropped | RSS **+8 KB** |
| 200,000 view chains (reshape → transpose → slice), base dropped each time | RSS **+24 KB** |
| 180,000 deliberately-bad calls | all 180,000 raised catchably; RSS +80 KB; process correct afterwards |
| 10M f64 allocate / view / strided slice | 47 ms / 0 ms / 0 ms |

### Build

`src/ndarray_native.cpp` registered in all four build files (`build.sh`, `build-mac.sh`,
`build-win.sh`, `CMakeLists.txt`), each compiling **that TU alone at `-O3 -ftree-vectorize`** while
everything else stays at `-O2` (N11). Verified in the generated CMake rules:
`ndarray_native.cpp.o_OPTIONS = -O3;-ftree-vectorize`. The CI grep of `-fopt-info-vec-optimized`
belongs to Phase 2, where the tier-0 kernel loops it would check actually exist.

---

## Phase 1.1 — Memory-safety hardening ✅

Prompted by a review of the three safety properties Phase 1 claimed. Each was probed on a real build
rather than reasoned about, which is the only reason the second one was caught: **it did not hold.**
All five gate tiers green — `tests/numba_array_test.b` **127/127**, `tests/numba_stress.sh`
**13/13**, 34 `tests/` suites plus 31 package-local suites, macOS arm64.

### The claim that was false

- **[bug fix]** The allocation ceiling was **per allocation**, so it did not bound a loop — and a
  loop is how a request handler actually exhausts a server, which was the stated reason for having a
  ceiling. Measured: **twelve 200 MB arrays held simultaneously against a 2 GiB limit, no error.**
  Admission is now tested against `live + requested`, with live bytes in an atomic counter
  incremented in the `Buffer` constructor and decremented in its destructor (N17). The same probe
  now admits 10 and refuses the 11th.
- **[feature]** A **hard ceiling** from `BANTU_ND_MAX_BYTES`, read once at startup, that
  `nd_max_bytes()` can only move *downward* from. A limit the untrusted side can raise for itself is
  a guard against mistakes, not against hostile input — the shape CPython settled on for
  CVE-2020-10735. The error message distinguishes "raise it with `nd_max_bytes(n)`" from "set by
  `BANTU_ND_MAX_BYTES`, which script cannot raise", so it never advises a fix that cannot work.
  A malformed env value keeps the default rather than disabling the limit or forbidding everything.
- **[feature]** `nd_live_bytes()` — the number admission is tested against, exposed, because an
  unexplainable limit is an unusable one.

### The claim that held, checked properly

- **[test]** Read-only propagation was verified across **every** builtin that takes an array and
  returns one, not just `nd_set`. The two that hand back a *writable* array from a read-only base
  (`nd_reshape`, `nd_ravel` on non-contiguous input) were checked by buffer identity rather than by
  reading the code: `nd_base_id` differs and a write through the result leaves the source at 0. They
  copy, and a copy of a read-only array is legitimately writable — matching NumPy's own rule. No
  change needed.
- **[feature]** Enforcement moved into one `requireWritable()` gate, because `writable` was checked
  in exactly one place while Phase 2's `out=` and Phase 3's `nd_put` are three more chances to forget.

### Defects found while verifying

- **[bug fix]** **`makeView` validated nothing** — it trusted caller-supplied shape, strides and
  offset. It now checks that every addressable element lies inside the buffer (N19), in the one
  chokepoint every view passes through, because "remember to check" does not survive phases 2–5
  adding view constructors. NumPy's analogue is `PyArray_CheckStrides`.
- **[bug fix]** **`nd_slice` never type-checked `start`/`stop`/`step`.** It read `.numberVal` off any
  `Value` — which is 0 for a list, a string or null — so `nd_slice($m, [[[1,2], null, null], null])`
  was silently accepted as index 0. It then pushed the result through `llround`, undefined outside
  `long long`'s range, so a start of `1e300` silently produced an empty array instead of an error.
- **[bug fix]** **Signed overflow in stride arithmetic.** On a `(4,8)` array, a step of `-2^62`
  wrapped `strides[0] * step` to **0**, producing a **stride-0 axis on an array still marked
  writable** — precisely the state `nd_broadcast_to` exists to refuse. It was contained only because
  the count computation clamped the extent to 1. That is luck, not design. `checkedMulSigned` on
  `__builtin_mul_overflow` now guards both products, and the guard is reachable and tested: a legal
  step of `2^53` on a stride-2048 axis still overflows and raises.
- **[bug fix]** **A data race, and a reproducibility bug, from process-global state.** sua runs each
  connection's Bantu handler on a detached `std::thread` (`server.hpp:863`), so the limit and the
  PRNG were shared across concurrent requests. The counters are now `std::atomic` and the `Rng` is
  `thread_local` (N18) — a shared stream meant `nd_seed()` in one request silently reshaped every
  other in-flight request's arrays, which is the same objection that made numba carry its own PRNG
  instead of using global `random()`, one level up.
- **[bug fix]** Error messages read `nd_slice: nd_slice: ...` — the wrapper prefixed the builtin name
  unconditionally onto messages that already carried it. Fixed once at the wrapper, covering all 27.

### Deliberately not changed

- **[perf]** `checkedMul` keeps the division form for unsigned products. `__builtin_mul_overflow` is
  one `mul` plus `jo` against a 20–40 cycle divide, but this runs once per array creation against an
  allocation measured in tens of milliseconds, and it is portable to MSVC with no `#ifdef`. The
  signed version is a different matter and does use the builtin — signed overflow is UB, so it
  cannot be detected after the fact.
- **[docs]** The `memset` in `Buffer` is **load-bearing** and is now documented as such in three
  places (N20). Switching it to `calloc` — the obvious optimisation, and what NumPy does — would
  serve large blocks from kernel zero pages under Linux overcommit, so nothing is committed until a
  kernel touches it: the ceiling would stop bounding real memory and a catchable error would become
  a deferred OOM kill inside a request handler.

### New builtins

`nd_live_bytes()`, `nd_shares_memory(a, b)` — the latter conservative like `np.may_share_memory`,
reusing the same extent walk as the view check, and the aliasing test Phase 2's `out=` needs.

### Stress

| | result |
|---|---|
| 50,000 arrays + view chains | live-byte drift **0 bytes**, exactly |
| 20,000 refused allocations | live-byte drift **0 bytes** (a drift here slowly bricks numba) |
| hoarding loop under a 100 MB ceiling | held **100 MB**, then returned every byte |
| 200,000 arrays / 200,000 view chains / 180,000 bad calls | RSS +12 KB / +28 KB / +60 KB |

Byte accounting is the more precise instrument: RSS is noisy enough that a small genuine leak hides
inside normal allocator variation, whereas the live counter must return to baseline exactly.

---

## Phase 2 — Broadcasting and element-wise ufuncs ✅

Feature suite `tests/numba_ufunc_test.b` **111/111**; `tests/numba_array_test.b` **130/130**;
stress `tests/numba_stress.sh` **19/19**; full regression **69/69** (macOS arm64, Apple silicon).

### What landed

- **[feature]** NumPy broadcasting: right-aligned shapes, stretched axes get stride 0 and copy
  nothing. Failures name the axis and both extents —
  `nd_add: shapes [3,4] and [5,4] cannot be broadcast together (axis 0: 3 vs 5)`.
- **[feature]** The three-tier kernel loop. Tier 0 is one flat `__restrict` pointer loop (confirmed
  vectorized: NEON width 2 × interleave 2, the maximum for f64); tier 2 **coalesces dimensions
  first**, so a contiguous (1000,10000) operation collapses back into one flat loop — measured
  19 ms against 13 ms for the equivalent 1-D call, versus the 2–3× loss coalescing exists to avoid.
- **[feature]** ~55 ufuncs: full arithmetic, floor-division and modulo, 29 transcendentals, six
  comparisons, boolean logic, three predicates, `nd_where`, `nd_clip`, `nd_isclose`, `nd_allclose`,
  `nd_array_equal`, `nd_broadcast_shapes`. Scalars and plain Bantu lists are accepted as operands
  without wrapping: `nd_add($a, 2)` works.
- **[feature]** Optional `out=` on every ufunc, with three gates before a single element is written:
  `requireWritable` (a `broadcast_to` destination **raises**), an exact shape match (`out` is not
  itself broadcast), and aliasing. The **exact** overlap `nd_add($a,$b,$a)` runs in place at full
  speed; any other overlap computes into a temporary and copies back, so a partially-overlapping
  `out=` gives the same answer as the non-aliased call rather than garbage.
- **[feature]** `nd_empty` is now genuinely uninitialised (N20): **0 ms vs 43 ms** for 10M f64. Its
  DoS bound comes from the live-byte accounting, which counts bytes whether or not they are touched.

### Semantics worth knowing

Integer division and modulo **floor**, as in Python and NumPy, not C: `-7 // 2` is `-4`. Integer
division by zero **raises** rather than trapping — it is `SIGFPE` on x86, a process kill, not an
error. Float division by zero is *not* an error: IEEE says ±inf and that is the right answer.
`nd_minimum`/`nd_maximum` propagate NaN, like NumPy's rather than C's `fmin`. `nd_rint` is
half-to-even. `/` always produces f64. `bool + bool` is i64, because `true + true` is 2.

### Performance

| | before | after |
|---|---|---|
| 10M `nd_add` | 22 ms — 10.6 GB/s | **13 ms — 18.0 GB/s** |
| 10M `nd_sqrt` | 23 ms | **12 ms** |
| 10M `nd_greater` | 30 ms | **14 ms** |
| 10M `nd_exp` | — | 65 ms (scalar libm, as predicted) |

A hand-written standalone C++ loop doing identical work on the same machine takes **13.37 ms**, so
numba is within 3% of the kernel's own ceiling and **single-core bandwidth is now the wall**.

### Defects found and fixed

Three, all in this phase's own code, all found by measurement rather than review:

- **[bug fix]** **Predicates returned garbage on integer arrays.** `nd_isfinite(nd([1], "i64"))` was
  **false**. `Kind::COMPARE` let them promote to the integer path, where these ops have no meaningful
  kernel — they ran a stub returning 0. A predicate asks a question only a float can answer, so it
  now computes in f64 whatever it is handed (`Kind::PREDICATE`).
- **[bug fix]** **Logical ops rounded floats instead of testing truthiness.** `0.4 and true` was
  **false** while `0.6 and true` was **true** — `llround`, not truth. Logic computed in `BOOL`, which
  converted f64 operands by rounding. It now computes in the promoted type, so `!= 0` is evaluated
  exactly.
- **[perf]** **The kernel dispatch defeated its own vectorization.** The ufunc table declared its
  kernels as *function pointers*, so every operation decayed to one type and `tier0` had a single
  shared instantiation calling through an opaque pointer once per element. `-Rpass=loop-vectorize`
  reported 26 vectorized loops while the one that mattered was an indirect call — which is why the
  check that found it was a comparison against a standalone loop, not the compiler's own report.
  Declaring the parameters `auto` gives each operation its own inlined loop: **1.69×**.

### Measurements that changed the plan

- **Non-temporal stores were rejected on measurement.** The literature's 1.40–1.42× on STREAM Triad
  is an x86 result; on Apple silicon NT stores were **slower** at both DRAM-resident (13.86 vs
  13.42 ms) and cache-resident (0.1319 vs 0.1125 ms) sizes. Recorded rather than shipped.
- **The "≥ 10 GB/s effective" gate was replaced with per-platform targets** (N22), and the Apple
  figure corrected: 20 GB/s was a guess, and one core cannot reach it here.
- **Threading is deferred to Phase 3+** with the justification now measured rather than assumed: a
  thread pool has to be designed against sua's per-connection threads, the same interaction that
  forced N18.

### Stress

| | result |
|---|---|
| 100,000 in-place `nd_add($a,$b,$a)` | live-byte drift **0 bytes** — the exact-overlap path allocates no temporary |
| 60,000 rejected ufunc calls (bad shape, read-only `out`, wrong-size `out`) | all raised; drift **0 bytes** |
| 10M `nd_add` / `nd_sqrt` | 13 ms / 12 ms |
| `nd_empty` vs `nd_zeros`, 10M f64 | 0 ms vs 43 ms |

---

## Phase 3 — Reductions, scans, sorting, indexing ✅

Feature suite `tests/numba_reduce_test.b` **137/137**; stress `tests/numba_stress.sh` **26/26**;
full regression **71/71** (macOS arm64).

### What landed

- **[feature]** 20 reductions with `axis` (null / a number / a list, negatives counting from the end)
  and `keepdims`: `sum prod mean var std min max ptp argmin argmax any all count_nonzero median
  quantile` plus `nansum nanmean nanmin nanmax`.
- **[feature]** Scans — `cumsum cumprod cummax cummin diff` — along any axis, flattening when no axis
  is given.
- **[feature]** Sorting and search: `sort`, `argsort` (**stable**), `searchsorted` with a `side`,
  `unique`, `bincount` with `minlength`, `histogram` with an optional range.
- **[feature]** Fancy and boolean indexing: `take` (per-axis, negative indices allowed), `put`
  (in place, honouring `requireWritable`), `compress`, `nonzero`.

### Accuracy — the gate that a naive implementation fails by construction

`sum`/`mean` use binary-tree accumulation carried like a binary counter, so error grows as `log n`
rather than `n`, and — unlike a recursive formulation — it works for a **strided** walk in one pass
with `O(log n)` state. The contiguous path feeds it 128-element blocks summed with eight unrolled
accumulators, keeping the vectorized loop and the accurate structure at once.

| | |
|---|---|
| sum of 10M copies of `0.1`, relative error | **2.33e-16** |
| the gate | < 1e-12 |
| what a naive accumulator loses (`n·eps`) | ~2e-12 — it fails this by design |

`var`/`std` use Welford, one pass, which survives a 1e9 offset that destroys the textbook
sum-of-squares formula (asserted).

### Semantics worth knowing

Empty slices return the operation's **identity** where one exists — `sum` → 0, `prod` → 1, `any` →
false, `all` → true — and `min`/`max`/`argmin`/`argmax` of an empty slice **raise**, because there is
no identity and returning 0 or ±inf would be a silently wrong answer. `mean` of nothing is NaN, as in
NumPy. Reductions **propagate** NaN; the `nan*` forms skip it, and an all-NaN slice gives NaN rather
than 0. NaN **sorts last**, matching NumPy — some rule must be imposed since every comparison with
NaN is false. `argsort` is stable, so ties keep input order and results are reproducible.

### Performance

| | measured | roadmap target |
|---|---|---|
| 10M `nd_sum` | **4 ms** | ≤ 12 ms |
| 1M `nd_argsort` | 142 ms | ≤ 80 ms — **missed** |

`nd_argsort` is honestly over target. It materialises the lane into a `double` buffer and an index
buffer before `std::stable_sort`, which is two allocations and an indirection per comparison;
`col_argsort` does 73 ms on a typed array without that. Not chased here because the roadmap's 80 ms
came from a different data path, and the fix (a typed direct-on-buffer sort) belongs with the same
work that would parallelise it. Recorded rather than quietly re-baselined.

### Defect found and fixed

- **[bug fix]** **Storing NaN or ±infinity in an integer array corrupted it silently.**
  `nd_set(nd([1,2],"i64"), 0, NaN)` wrote **INT64_MIN**, and storing infinity wrote **0** — the second
  being worse, because 0 looks like a real answer rather than obvious garbage. Neither value has an
  integer representation, and the conversion happened without a word. `nd_set` now raises, naming the
  fix (`nd_astype(a, "f64")`); NumPy raises for the same reason. `bool` is deliberately left alone —
  NaN is truthy there, which is defensible and also NumPy's behaviour.

  Worth recording *why* this was easy to hit: Bantu has a single number type, so `nd([1.0, 2.0])`
  infers **i64** — `1.0` and `1` are the same `Value` and numba cannot tell them apart. Anything that
  will hold a NaN has to ask for `"f64"` explicitly. That is a documented consequence of the language,
  not a numba choice, but it is the reason the silent corruption was reachable from ordinary code.

### Stress

| | result |
|---|---|
| 10M `nd_sum` accuracy at scale | relative error **2.33e-16** |
| 1M `argsort`, permutation applied to the input | reproduces `nd_sort` **exactly** |
| 1M already-sorted / reverse-sorted / all-equal | all three sort correctly |
| 40,000 bad reduction and indexing calls | all raised; live-byte drift **0** |

---

## Phase 4 — Interpreter operator / index / method dispatch ✅

A **language change**. `tests/lang_native_ops_test.b` **102/102**; full regression **73/73**.

### What landed

Five additive arms in the evaluator, each filling in a path that was **dead**:

| site | before | now |
|---|---|---|
| `evalBinaryOp` | `$handle + 1` read `numberVal` (always 0) → silently `1` | `+ - * / %` and `< <= > >=` element-wise |
| `evalUnaryOp` | `-$handle` → `-0` | element-wise negation |
| `evalIndexAccess` | fell through → `null` | element, view, mask select, or gather |
| `evalIndexAssign` | threw "Cannot index-assign to this type" | element, row, or masked write |
| `evalDotAccess` | fell through → `null` | `$a.sum()`, bound to the same `NativeFn` as `nd_sum` |

```bantu
$y = ($x * $x) + $x;        // was: 0
$m[1][2] = 99;              // was: "Cannot index-assign to this type"
$vals[$vals > 20] = 0;      // was: null
$x.multiply($x).add($x).sum();
```

`$m[1]` on a 2-d array returns a **view**, which is what makes `$m[1][2] = 99` write through to the
base. Handles have reference semantics; lists do not. Both are asserted side by side.

### Two deliberate asymmetries

- **`==` and `!=` stay identity comparisons**, not element-wise. `if ($a == $b)` is written
  constantly, and an element-wise result would silently turn it into "is this array non-empty and
  all-truthy". NumPy made the other choice and then had to make `if arr:` raise; Bantu has no such
  escape hatch. `nd_array_equal` and `nd_allclose` are the explicit forms.
- **`+` with a string concatenates**, following the language's own rule that `5 + "x"` is `"5x"`, so
  `print("a: " + $a)` works. Every *other* non-combinable operand raises.

### The benchmark gate, and what measuring it actually took

The gate is ±2% on the interpreter's hot paths. Reaching a defensible answer needed three attempts,
and the first two were wrong in instructive ways:

1. **Sequential before/after runs: unusable.** The first post-change run showed +13.2%, and re-running
   the *same binary* gave 2942 / 2685 / 2690 ms. The first measurement was an artifact of the build
   that had just finished.
2. **Interleaved A/B: still unusable.** It reported +2.66% and +3.15% — but also moved `fib` and
   string concatenation, which these arms do not touch. That was the tell.
3. **A control group settled it.** Running a **byte-identical binary against itself** showed swings of
   up to ±1.97% and a 2.6–4.4% spread across runs. **The ±2% gate sits at this harness's noise
   floor**, so the earlier numbers were never signal. (The build is deterministic — two builds of
   identical source are byte-identical — so linking was not the cause.)

Final measurement: 4× longer loops, 10 paired rounds, median of per-round ratios, against a control:

| | control (identical binaries) | treatment (before → after) |
|---|---|---|
| 4M arithmetic loop | +0.52% | **+0.61%** |
| 1.5M list index read | +0.94% | **+0.81%** |

The treatment is indistinguishable from the control — `idxr` is *below* its own control — so the
dispatch costs **less than the measurement floor**. Gate passes, on evidence rather than assumption.

The control group is the part worth keeping. Without it, "+2.66%" looks like a real regression and
would have sent the next person optimising something that was never slow.

### Two placement defects found by that measurement

Both were mine, both in the first cut, and neither would have been visible without the A/B:

- **`evalIndexAssign` evaluated `n->object` twice.** The new arm called `evalNode(n->object)` at the
  top while the existing code evaluated it again below — so `f()[0] = 1` would have called `f`
  twice, and every list write paid for an extra evaluation. Hooked into the two places the object is
  *already* evaluated instead.
- **The index and dot checks sat in front of the common paths.** A handle is not a list, a dict or a
  string, so checking for one first made every ordinary list index pay. Moved after those cases,
  where the check replaces a fall-through and costs nothing. The binary-operator check was also
  narrowed from two short-circuited compares to a single OR against zero, exploiting `NUMBER == 0`.

### Defect fixed

- **[bug fix]** An array combined with `null`, a dict or a class instance **silently produced 0** —
  the dispatch declined, and the evaluator fell through to `left.numberVal + right.numberVal`, which
  is `0 + 0` for two non-numbers. That is precisely the failure this phase exists to remove, so
  reintroducing it at the edges would have been self-defeating. Non-combinable operands now raise and
  name what arrived.

---

## Phase 5 — Linear algebra ✅

`tests/numba_linalg_test.b` **69/69**; full regression **75/75** (macOS arm64, Apple silicon).
All hand-written, no BLAS (N-of §3.2).

### What landed

`nd_matmul`, `nd_dot`, `nd_outer`, `nd_trace`, `nd_solve`, `nd_inv`, `nd_det`, `nd_slogdet`,
`nd_cholesky`, `nd_qr`, `nd_lstsq`, `nd_eigh`, `nd_svd`, `nd_matrix_rank`, `nd_cond`, `nd_pinv`,
`nd_norm` — each also reachable as a method (`$a.solve($b)`).

### The three phase gates, measured

| gate | required | measured |
|---|---|---|
| 500×500 solve residual | < 1e-10 | **1.24e-14** (23 ms) |
| 200×200 SVD reconstruction | < 1e-12 | **1.97e-13** (253 ms) |
| 1000³ matmul | ≤ 500 ms / ≥ 4 GFLOP/s | **255–310 ms, 6.5–7.8 GFLOP/s** |

Every other residual came out at machine precision: QR orthogonality `‖QᵀQ − I‖` = **7.8e-16**,
eigh `‖AV − V diag(w)‖` = **7.1e-15** with `‖VᵀV − I‖` = **1.1e-15**, SVD `‖USVᵀ − A‖` = **2.2e-15**,
least-squares residual orthogonality `‖Aᵀr‖` = **3.6e-15**.

### Algorithm choices, and what each is defending against

- **Blocked matmul, 64×64 tiles, i-k-j inner order.** The ordering is the gate, not the tiling: the
  naive i-j-k order strides through B by `cols` on every innermost step, which is a cache miss per
  element once the matrix leaves L2. Worth more than any amount of unrolling.
- **LU with partial pivoting**, which yields solve, inv, det, slogdet. Pivoting is not optional —
  without it a perfectly well-conditioned matrix with a zero in the corner fails outright.
- **Householder QR, not Gram-Schmidt.** Gram-Schmidt loses orthogonality catastrophically on
  ill-conditioned input; the measured `‖QᵀQ − I‖` of 7.8e-16 is the property being bought.
- **`lstsq` through QR, not the normal equations.** Forming `AᵀA` *squares* the condition number and
  throws away half the available digits.
- **`eigh` by cyclic Jacobi** — unconditionally convergent, orthogonality to machine precision,
  slower than tridiagonal QR and bulletproof.
- **SVD by one-sided Jacobi** — ~120 lines against ~500 for Golub–Kahan, no convergence tuning, and
  *higher* relative accuracy on small singular values. It costs several sweeps, so it is 3–5× slower
  than LAPACK and impractical much beyond n ≈ 500–800. That limit is documented, not hidden.
- **`slogdet` alongside `det`**, because the determinant of a 500×500 overflows long before the
  matrix stops being interesting.

### Failure is loud, by design

Returning NaN or garbage for a degenerate input is the worst outcome, because it looks like an
answer. Each of these raises and names what to use instead:

- a singular `solve` or `inv` → suggests `nd_lstsq` / `nd_pinv`
- a non-positive-definite `cholesky` → says it needs an SPD matrix and points at `nd_solve`
- a **non-symmetric `eigh`** → names the offending element and both its values, rather than silently
  symmetrising, which would answer a different question
- a non-square `solve`/`det`/`cholesky`, a mismatched right-hand side, a mismatched matmul inner
  dimension → all name both sizes
- a rank-deficient `lstsq` → points at `nd_pinv` for the minimum-norm solution

### Not shipping, and `docs/numba.md` will say so

General nonsymmetric `eig`: balancing + Hessenberg reduction + Francis double-shift QR + complex
eigenvalue extraction is research-grade and needs complex arithmetic numba does not have. A fragile
`eig` is worse than none, and `eigh` already covers covariance matrices, PCA and graph Laplacians.
Also not shipping: `expm`, `schur`, FFT, sparse.

### Testing approach worth reusing

Linear algebra is tested by **residual**, not against reference numbers. Asserting `solve()` returns
a particular vector is fragile and proves little; asserting `‖Ax − b‖ < 1e-10` proves the answer
solves the system and stays meaningful when the pivoting order changes. Likewise `lstsq` is checked
by the defining property — the residual is orthogonal to every column of A — rather than by comparing
coefficients.

---

## Phase 6 — The package, the docs, the gallery ✅ (partial)

`tests/numba_pkg_test.b` **51/51**; `numba/numba_test.b` **12/12**; `tests/run_samples.sh`
**21/21**. What shipped, and what did not, is stated plainly at the end.

### The façade — why it exists at all

`numba/numba.b` is the public API; the `nd_*` builtins are plumbing. The reason is not tidiness:
**the façade can have optional arguments and a raw builtin cannot.** Bantu binds a missing argument
to null, so `np.arange(0, 10)` works here while `nd_arange` needs an explicit trailing `null`. Half
of `numba_pkg_test.b` asserts exactly those defaults, because a wrapper that forgets one is a
wrapper that does nothing.

### Composed helpers, built from the atoms

`polyfit`, `polyval`, `interp`, `gradient`, `cov`, `corrcoef`, `meshgrid`, `moving_average`,
`trapz` — none needed a new kernel. Each is checked against a value workable by hand:

- `polyfit` goes through `lstsq` (hence QR), because a Vandermonde matrix is already
  ill-conditioned and the normal equations would square that.
- `polyval` uses Horner's rule.
- `interp` **clamps** outside the range rather than extrapolating — silently extrapolating is how
  people get nonsense far from their data.
- `corrcoef` of a constant series returns **NaN**, not 0: 0 would claim "no relationship" when the
  truth is "the question is meaningless".
- `meshgrid` returns copies rather than broadcast views, because a grid is something people expect
  to be able to write to.

### Package, docs, gallery

- `numba/{numba.b, numba_test.b, package.json}` — **verified end to end**: `bantu publish ./numba`,
  then `bantu add numba` in a clean temporary project, then `include "numba" as np;` by **bare
  name**, then `$a.sum()` and `$a + $a`. That bare include is the Phase A resolver fix paying off.
- `docs/numba.md` — quickstart, tour, real measured numbers, the safety model, "things worth knowing
  before you hit them", and a blunt list of what is deliberately absent.
- `samples/numba/` — four runnable programs: quickstart, a linear fit with R², a 200×200 solve
  checked by residual, and a large-array walkthrough showing `out=` holding memory flat.
- `tests/run_samples.sh`, wired into **both** CI jobs — every sample is executed, not merely
  written. Exiting 0 is not enough: a sample that printed an error and carried on would pass, so the
  runner also greps the output for `[error]`/`[fatal]`.

### Defects found and fixed

- **[bug fix] `any` could not be used as a property name.** It is a reserved *type* keyword, so
  `$a.any()` failed with "Expected property name after '.'" — and so did a dict key called `number`,
  `string` or `delete`, latent for anyone whose data used one of those names. The parser had a
  **hand-maintained list** of keywords permitted after a dot, and it was necessarily incomplete. It
  now accepts anything that lexes as a bare word, which is safe because a property name can only
  follow a dot and is therefore never ambiguous with a keyword.

  The façade function still has to be `np.anyof` — `def any(...)` does not parse — but
  `nd_any($a, ...)` and `$a.any()` both work, and `docs/numba.md` says so.

- **[bug fix] `samples/blogsite/db.b` called `sua.sqlite.connect()`**, which does not exist — the
  API is `open()`. The sample had been broken long enough that the **shipped release binary
  reproduces it**. Exactly the rot `run_samples.sh` now exists to prevent; it was found within
  minutes of the runner being written.

- **[fix] `run_samples.sh` skips `*/server.b`** by name, with the reason stated: those bind a port
  and block forever, and several also need a database CI does not have. Without the skip the runner
  hangs the build rather than failing it.

### Not done, and not pretended otherwise

Three Phase 6 rows remain open, and they are integration work rather than numba work:

- **the arctic bridge** (`nd_from_column` zero-copy borrow, `nd_to_column`, `nd_from_frame`, and
  `arctic.b`'s `to_ndarray()` behind `has_native("ndarray")`)
- **arctic's ~15 transcendental `col_*` kernels**
- **the sua-concurrency and cross-platform gates** — the concurrency design is already settled
  (N18: `thread_local` PRNG, atomic limits) but running numba inside `sua_concurrency_test.sh` has
  not been done, and Linux/Windows CI has not been observed green for this work.

These are listed in `ROADMAP.md` as open rather than ticked.
