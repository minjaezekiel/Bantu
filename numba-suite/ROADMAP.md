# numba — Roadmap

Native `NdArray` foundations plus the pure-Bantu `numba` library that sits on them. The *atoms* (an
`NdArray` type with zero-copy views, ~150 `nd_*` kernels, operator dispatch) are native; everything
above them — the API, defaults, composed helpers — is Bantu. Design and rationale live in
[`docs/numba-architecture.md`](../docs/numba-architecture.md) and
[`DECISIONS.md`](DECISIONS.md).

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done (feature **and** stress tests green)

**Every phase is gated by five tiers, all green before it advances:**

| tier | proves | form |
|---|---|---|
| 1. feature | every builtin does what it claims, including its failure modes | `tests/numba_*_test.b`, ending `RESULT: ALL GREEN` |
| 2. differential | the answers are *right*, not merely stable | NumPy reference values embedded in the test; libm for transcendentals; pure-Bantu loops for small cases |
| 3. **stress** | it survives size, repetition and abuse | scale · RSS · adversarial input · aliasing · numerical · concurrency |
| 4. regression | nothing else broke | the whole existing suite (lang, scope, crypto, uuid, random, orm, arctic, sua, webpush) on the same build |
| 5. sanitizers | no latent memory defect | ASan + UBSan over the full numba suite |

Results recorded in [`CHANGELOG.md`](CHANGELOG.md). Every phase also re-runs `benchmarks/run.sh` so
an interpreter-wide regression cannot hide.

**Two rules that apply to every phase.** Never assert on a stringified number — `str()` gives six
significant digits and leaks scientific notation; use `nd_allclose`. Never use the C-style `for`
above 100,000 iterations — `evaluator.hpp` caps it at `safety < 100000` and exits **silently**, so a
benchmark looping 200k times reports a wrong-but-plausible number and passes.

---

## Phase 0 — Tracking docs
- [x] `docs/numba-architecture.md`
- [x] `numba-suite/ROADMAP.md`, `DECISIONS.md`, `CHANGELOG.md`

## Phase A — Adoption plumbing ✅ (first: everything else rides on it)

These are cross-cutting interpreter fixes, not numba code. Each was latent for every existing
package, so each fixes `arctic`, `hash`, `crypto`, `uuid`, `random` and `orm` at the same time.

| Item | How | Feature test | Stress test | Status |
|---|---|---|---|---|
| `include "numba" as np` resolves after `bantu add` | `module_resolver.hpp`: `bantu_modules/<name>/` added to the candidate list, honouring `package.json`'s `main`, then `<name>.b`/`index.b`/`main.b` | ✅ `tests/lang_module_test.b` 13/13 | ✅ manifest `main` in a subdirectory, no-manifest fallback, a `"main"`-lookalike value, explicit relative paths unaffected, missing module | [x] |
| A repeat `include` binds the alias | modules cached by canonical path; every later include binds that same namespace object (module-singleton, as in Node). `sua.include()` shares the cache | ✅ `tests/lang_module_test.b` | ✅ diamond dependency, both aliases proven to be the same object, genuine cycle terminates and is reported by name | [x] |
| `print($handle)` renders the value | `types.hpp`: a repr registry keyed by handle tag; arctic registers one for columns (numba's lands in Phase 1) | ✅ `tests/lang_repr_test.b` 17/17 | ✅ empty, single, nulls, utf8 quoting, datetime overlay, 1000/1001 summarization boundary, **100 reprs of a 200k column in 0 ms** | [x] |
| **[found] `push` was O(n²)** | it returned the mutated list, deep-copying every `Value`. `parseExpressionStatement` now marks a call whose result is discarded, so the copy is skipped; `$x = push($x, v)` still returns the list | ✅ `tests/lang_list_test.b` 23/23 | ✅ 20,000 pushes **9,616 ms → 70 ms**; 100,000 **~4 min → 271 ms**; linear scaling asserted | [x] |
| **[found] `len($var)` copied its argument** | reads the length from the real storage; a user-defined `len()` still shadows the builtin | ✅ `tests/lang_list_test.b` | ✅ `$out[len($out)] = v` ×20,000 **7,027 ms → 58 ms**; every `len` answer unchanged (list/string/dict/non-container/literal) | [x] |

Both quadratic defects were **pre-existing** — the shipped release binary reproduces them, and could
not finish `bantu bench`'s own "list push 100k" in ten minutes. Hot paths measured before and after
against that release on the same machine: 1M arithmetic loop 2,336 → 2,186 ms, `fib(24)`
1,906 → 1,941 ms, 50k dict set 171 → 178 ms. Full detail in [`CHANGELOG.md`](CHANGELOG.md).

## Phase 1 — `NdArray` core, buffers, zero-copy views ✅

Feature suite `tests/numba_array_test.b` **108/108**; stress suite `tests/numba_stress.sh` **9/9**;
full regression green across 42 suites.

| Item | How | Feature test | Stress test | Status |
|---|---|---|---|---|
| `NdArray`, `Buffer`, handle tag `"ndarray"`, `has_native("ndarray")` | `ndarray_native.{hpp,cpp}` + `ndarray_api.hpp`; registered in all four build files, that TU alone at `-O3 -ftree-vectorize` (verified in the generated CMake rules) | ✅ | — | [x] |
| Creation, introspection, astype/copy/ascontiguous, `print()` repr | 35 `nd_*` builtins; own xoshiro256++ stream so results are reproducible and independent of `random()` | ✅ | ✅ `nd_zeros([10000000])` in **47 ms** (gate 300 ms) | [x] |
| Shape ops as **views** | `nd_reshape/transpose/T/ravel/flatten/swapaxes/moveaxis/expand_dims/squeeze/slice/broadcast_to/flip` | ✅ | ✅ **zero-copy proved two ways** — `nd_base_id` compares the real buffer address, and a write through one handle is read through another. reshape+transpose of 10M: **0 ms**; strided slice: **0 ms** | [x] |
| **Security: shape-product overflow + allocation ceiling** | checked multiplication for every product; ceiling default 2 GiB, `nd_max_bytes(n)`; `broadcast_to` is `writable=false`; every index bounds-checked with the axis and extent named | ✅ 19 adversarial cases, each also asserting the message names what was wrong | ✅ **180,000 bad calls all raised catchably**, RSS +80 KB, process correct afterwards | [x] |
| Lifetime | `shared_ptr` RAII; a view keeps its base alive | ✅ a slice returned from a function whose base went out of scope still reads correctly | ✅ 200k arrays **RSS +8 KB**; 200k view chains **RSS +24 KB** | [x] |
| `np.help()`, `np.info($a)` | deferred to Phase 6 with the rest of the `numba.b` façade | — | — | [~] |

One correction made during the work: `nd_is_view` first asked only "do I cover the whole buffer?",
which a transposed view does — so it reported a transpose as an independent array. It now asks
whether the buffer is shared, which is the question a user actually has.

## Phase 2 — Broadcasting and element-wise ufuncs

| Item | How | Feature test | Stress test | Status |
|---|---|---|---|---|
| Broadcasting | right-aligned shape rule; stretched axes get **stride 0**; errors name the axis and both extents | `tests/numba_ufunc_test.b` | ~30 shape pairs incl. every failure case with its expected message | [ ] |
| The 3-tier loop | tier 0 flat `__restrict`; tier 1 splat; tier 2 **coalesce dims** then an N-d odometer with the branch hoisted out of the inner loop | ″ | 4-D strided ops; `out=` views forcing tier 2 | [ ] |
| ~45 ufuncs + comparisons + boolean logic + `out=` | promotion `bool → i64 → f64`; i64 exact for `+ - * // %` | ″ | differential vs pure Bantu on 100k, max rel err < 1e-15; denormals, ±inf, NaN, overflow boundaries | [ ] |
| **Aliasing and writability** | overlap by buffer identity + element extent → copy (NumPy behaviour) | ″ | `nd_add($a,$a,$a)`; `out=` on a partially overlapping view; **writes through a `broadcast_to` result raise** | [ ] |
| Performance | — | — | **10M `nd_add` ≥ 10 GB/s effective**, reported as GB/s (portable) alongside ms; machine recorded | [ ] |
| Vectorization actually happened | `-O3` per-file override + CI grep of `-fopt-info-vec-optimized` | — | tier-0 loops present in the log on **both** Linux GCC and macOS Clang | [ ] |

## Phase 3 — Reductions with `axis`, scans, sorting, indexing

| Item | How | Feature test | Stress test | Status |
|---|---|---|---|---|
| Reductions with `axis`/`keepdims` | `axis` accepts null / number / list | `tests/numba_reduce_test.b` | every axis and axis-pair of a (7,5,3) array vs embedded NumPy reference values | [ ] |
| **Accuracy** | pairwise summation for `sum`/`mean`; Welford for `var`/`std` | ″ | **sum of 10M copies of 0.1 → rel err < 1e-12** (a naive accumulator fails this by design) | [ ] |
| Scans, `diff`, `nan*` variants | `cumsum/cumprod/cummax/cummin` | ″ | all-NaN slices; empty axes | [ ] |
| Sorting and search | `sort/argsort/partition/argpartition/searchsorted/unique/unique_counts/bincount/histogram` | ″ | 10M sort; already-sorted and reverse-sorted; all-equal; NaN ordering | [ ] |
| Fancy and boolean indexing | `take/put/mask/compress/nonzero/where` | ″ | out-of-range indices **raise**; empty masks; a mask of the wrong length | [ ] |
| Performance | — | — | 10M `nd_sum` ≤ 12 ms; 1M `nd_argsort` ≤ 80 ms (`col_argsort` does 73 ms) | [ ] |

## Phase 4 — Interpreter operator / index / dot dispatch  *(a language change)*

| Item | How | Feature test | Stress test | Status |
|---|---|---|---|---|
| `$a + $b`, `2 * $a`, `-$a`, `$a > 0.5` | one predicted branch at the top of `evalBinaryOp`/`evalUnaryOp`, falling through to the existing switch | `tests/lang_native_ops_test.b` | mixed handle/number/string/list/dict/null operands in every position | [ ] |
| `$a[$mask]`, `$m[1]` is a view, `$m[1][2] = 9` writes through | arms in `evalIndexAccess` / `evalIndexAssign` | ″ | chained subscripts; negative indices; out-of-range **raises** | [ ] |
| Nothing else changed | — | ″ | strings still concat; `==` on lists/dicts unchanged; `&&`/`\|\|` unchanged; **full existing regression green** | [ ] |
| **The benchmark gate** | `benchmarks/run.sh`, 5 iterations, before and after | — | 1M arithmetic loop within **±2%**; both numbers in the CHANGELOG | [ ] |

## Phase 5 — Linear algebra

| Item | How | Feature test | Stress test | Status |
|---|---|---|---|---|
| `dot/matmul/inner/outer`, transpose | blocked matmul, 64×64 tiles, `ikj` order; blocked copy-transpose | `tests/numba_linalg_test.b` | 1000×1000×1000 matmul **≤ 500 ms (≥ 4 GFLOP/s)**; batched matmul; non-conformable shapes raise | [ ] |
| LU family | partial pivoting → `solve`, `inv`, `det`, `slogdet`, `matrix_rank`; triangular substitution | ″ | 500×500 solve residual **< 1e-10**; singular and near-singular matrices **raise clearly, never return garbage** | [ ] |
| Cholesky, QR, `lstsq`, `pinv`, `norm`, `trace`, `cond` | Householder QR | ″ | non-SPD input to `cholesky` raises; rank-deficient `lstsq` | [ ] |
| `svd` (one-sided Jacobi), `eigh` (cyclic Jacobi) | N13 | ″ | SVD reconstruction 200×200 **< 1e-12**; `eigh` orthogonality ‖VᵀV−I‖ and ‖AV−VΛ‖ **< 1e-12**; ill-conditioned inputs | [ ] |

## Phase 6 — The package, the arctic bridge, the docs

| Item | How | Feature test | Stress test | Status |
|---|---|---|---|---|
| Composed helpers | `polyfit/polyval`, `interp`, `gradient`, `cov/corrcoef`, `meshgrid`, moving average — pure Bantu | `tests/numba_pkg_test.b` | degenerate fits; single-point interp | [ ] |
| arctic transcendentals | ~15 `col_sqrt/exp/log/…` natively, on the existing `unaryOp` shape with null propagation | `tests/arctic_transcendental_test.b` | nulls propagate; domain errors → NaN not crash | [ ] |
| The bridge | `nd_from_column` (zero-copy borrow), `nd_to_column` (copy), `nd_from_frame` (F-contiguous); `arctic.b` `to_ndarray()` behind `has_native("ndarray")` | `tests/numba_arctic_bridge_test.b` | round-trip exact; borrow **provably zero-copy via `nd_base_id`**; a column with nulls is refused by name; the borrowed array outliving the column | [ ] |
| Package, docs, gallery | `numba/{numba.b, numba_test.b, package.json}`, `docs/numba.md`, `samples/numba/`, `tests/run_samples.sh` | — | **every documented example and every sample executed by CI**; `bantu add numba` then `include "numba" as np` works from a clean project | [ ] |
| Concurrency | — | — | numba called inside a sua handler under `sua_concurrency_test.sh`; a big allocation inside a handler **raises rather than OOM-killing the worker** | [ ] |
| Cross-platform | — | — | CI green on Linux, macOS **and** Windows | [ ] |

## Phase 7 — Deferred, each needing its own justification

Nothing here is scheduled. Each item lands only on a concrete measured wall, with its own gate.

- AVX2/FMA **runtime** dispatch via `__builtin_cpu_supports` (portable binary preserved), only where
  profiling shows ALU-bound — realistically matmul and transcendentals.
- Vectorizable polynomial `exp`/`log`/`sin`/`cos`, gated on **≤ 1 ULP vs libm** across a dense
  domain sample including denormals, ±inf, NaN and overflow boundaries.
- A buffer free-list, to remove first-touch page-fault cost in chained expressions.
- Threading via `nd_set_threads(k)`, default 1, only above ~1M elements, and only for matmul,
  reductions and sorts. Note sua forks workers and the evaluator is single-threaded, so a pool
  inside the interpreter needs care.
- `F32` dtype; complex; FFT.
- `BANTU_BLAS` — revisit only on a concrete user wall, macOS/Accelerate first (N12).
