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
