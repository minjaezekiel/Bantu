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

Feature suite `tests/numba_array_test.b` **127/127**; stress suite `tests/numba_stress.sh` **13/13**;
full regression green (34 `tests/` suites, 31 package-local, plus the `const_bad` linter negative).

**Phase 1.1 hardening ✅** — a review of the three claimed safety properties probed each on a real
build instead of reasoning about it, and found the allocation ceiling did **not** hold: it was
per-allocation, so twelve 200 MB arrays sat live against a 2 GiB limit with no error. Fixed with
live-byte accounting plus a `BANTU_ND_MAX_BYTES` hard ceiling script cannot raise (N17). Also found
and fixed: `makeView` validated nothing (N19), `nd_slice` never type-checked its arguments and
wrapped `stride * step` to 0 via signed overflow — producing a stride-0 axis on a *writable* array
(N19) — and the limit and PRNG were process-global while sua runs handlers on detached threads (N18).
Read-only propagation was re-verified across every view- and copy-producing builtin and **held**.
See `docs/numba-security.md`.

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

## Phase 2 — Broadcasting and element-wise ufuncs ✅

Feature suite `tests/numba_ufunc_test.b` **111/111**; stress **19/19**; full regression **69/69**.
10M `nd_add` at **13 ms / 18.0 GB/s**, within 3% of a hand-written standalone C++ loop on the same
machine — single-core bandwidth is the wall. Three defects found by measurement and fixed: predicates
returned garbage on i64 arrays, logical ops rounded floats instead of testing truthiness, and the
kernel dispatch declared its ops as function pointers, defeating inlining and vectorization (1.69×
recovered). Non-temporal stores were **rejected on measurement** — the literature's 1.40× is an x86
result and they are slower here. See `numba-suite/CHANGELOG.md` and `docs/numba-acceleration.md`.


| Item | How | Feature test | Stress test | Status |
|---|---|---|---|---|
| Broadcasting ✅ | right-aligned shape rule; stretched axes get **stride 0**; errors name the axis and both extents | `tests/numba_ufunc_test.b` | ~30 shape pairs incl. every failure case with its expected message | [ ] |
| The 3-tier loop ✅ | tier 0 flat `__restrict`; tier 1 splat; tier 2 **coalesce dims** then an N-d odometer with the branch hoisted out of the inner loop | ″ | 4-D strided ops; `out=` views forcing tier 2 | [ ] |
| ~55 ufuncs + comparisons + boolean logic + `out=` ✅ | promotion `bool → i64 → f64`; i64 exact for `+ - * // %` | ″ | differential vs pure Bantu on 100k, max rel err < 1e-15; denormals, ±inf, NaN, overflow boundaries | [ ] |
| **Aliasing and writability** ✅ | overlap by buffer identity + element extent → copy (NumPy behaviour) | ″ | `nd_add($a,$a,$a)`; `out=` on a partially overlapping view; **writes through a `broadcast_to` result raise** | [ ] |
| Performance ✅ | — | — | **MEASURED: 13 ms / 18.0 GB/s on Apple M-series, vs 13.37 ms for a standalone C++ loop.** PER-PLATFORM target, not a flat 10 GB/s** — see N22: one core sustains ~10 concurrent L1 misses, capping a single thread near 8.1 GB/s at a 79 ns memory latency, so the flat figure was unreachable by construction on a high-latency cloud vCPU. Apple silicon ≥ 15 GB/s (20 was a guess; one core measured at 17.5 GB/s for the bare loop); x86 desktop ≥ 10 GB/s; shared vCPU ≥ 5 GB/s single-threaded, or ≥ 10 GB/s with the `std::thread` tier-0 if it lands | [ ] |
| Memory-system work (N22) — **NT stores measured and rejected**; threading deferred to Phase 3+ with the ceiling now confirmed reached; page-alignment and huge pages still open | page-align large buffers; `madvise(MADV_HUGEPAGE)`; a fixed-size `std::thread` parallel-for designed against sua's per-connection threads | — | NT stores measured against the scalar path on cache-resident **and** DRAM-resident sizes (they *lose* when the destination fits in cache); the thread pool must not multiply against sua's per-connection threads | [ ] |
| `out=` honours the safety gates ✅ | every destination through `requireWritable()` + `nd_shares_memory()` | `broadcast_to` as `out=` **raises**; an `out=` overlapping an input is copied or refused, never garbage | aliasing fuzzed across offset/stride/flip combinations | [ ] |
| `nd_empty` genuinely uninitialised (N20) ✅ | skip the `memset` for it alone; live-byte accounting still bounds it | contents unspecified but shape/dtype/strides correct | allocation of 10M `nd_empty` is **not** charged 40 ms of page-touching | [ ] |
| Vectorization actually happened ✅ | `-O3` per-file override + CI grep of `-fopt-info-vec-optimized` | — | tier-0 loops present in the log on **both** Linux GCC and macOS Clang | [ ] |

## Phase 3 — Reductions with `axis`, scans, sorting, indexing ✅

`tests/numba_reduce_test.b` **137/137**; stress **26/26**; regression **71/71**. The accuracy gate
came out at **2.33e-16** relative for 10M copies of `0.1`, against the 1e-12 threshold a naive
accumulator fails by construction. 10M `nd_sum` at **4 ms** (target 12). One target missed honestly:
1M `nd_argsort` at **142 ms** against 80 — see `numba-suite/CHANGELOG.md` for why it is recorded
rather than re-baselined. One defect found and fixed: storing NaN or infinity in an i64 array wrote
INT64_MIN and 0 respectively, silently.


| Item | How | Feature test | Stress test | Status |
|---|---|---|---|---|
| Reductions with `axis`/`keepdims` ✅ | `axis` accepts null / number / list | `tests/numba_reduce_test.b` | every axis and axis-pair of a (7,5,3) array vs embedded NumPy reference values | [ ] |
| **Accuracy** ✅ **2.33e-16** | pairwise summation for `sum`/`mean`; Welford for `var`/`std` | ″ | **sum of 10M copies of 0.1 → rel err < 1e-12** (a naive accumulator fails this by design) | [ ] |
| Scans, `diff`, `nan*` variants ✅ | `cumsum/cumprod/cummax/cummin` | ″ | all-NaN slices; empty axes | [ ] |
| Sorting and search ✅ | `sort/argsort/partition/argpartition/searchsorted/unique/unique_counts/bincount/histogram` | ″ | 10M sort; already-sorted and reverse-sorted; all-equal; NaN ordering | [ ] |
| Fancy and boolean indexing ✅ | `take/put/mask/compress/nonzero/where` | ″ | out-of-range indices **raise**; empty masks; a mask of the wrong length | [ ] |
| Performance — `nd_sum` ✅, `nd_argsort` ⚠ | — | — | 10M `nd_sum` **4 ms** (target 12) ✅; 1M `nd_argsort` **142 ms** against a target of 80 — **missed**, recorded not re-baselined: it materialises a double buffer plus an index buffer before `std::stable_sort`, where `col_argsort`'s 73 ms is a typed direct-on-buffer sort. The fix belongs with the work that would parallelise it | [ ] |

## Phase 4 — Interpreter operator / index / dot dispatch ✅  *(a language change)*

`tests/lang_native_ops_test.b` **102/102**; regression **73/73**. The ±2% benchmark gate **passes**,
but establishing that took a control group: a **byte-identical binary measured against itself**
swings up to ±1.97%, so the gate sits at this harness's noise floor and the naive before/after
readings (+13.2%, then +2.66%) were never signal. Final paired-ratio measurement against that
control: arithmetic +0.61% vs a +0.52% control, list index +0.81% vs a +0.94% control — i.e.
indistinguishable from zero. Two placement defects and one silent-zero defect found and fixed along
the way; see `numba-suite/CHANGELOG.md`.


| Item | How | Feature test | Stress test | Status |
|---|---|---|---|---|
| `$a + $b`, `2 * $a`, `-$a`, `$a > 0.5` ✅ | one predicted branch at the top of `evalBinaryOp`/`evalUnaryOp`, falling through to the existing switch | `tests/lang_native_ops_test.b` | mixed handle/number/string/list/dict/null operands in every position | [ ] |
| `$a[$mask]`, `$m[1]` is a view, `$m[1][2] = 9` writes through ✅ | arms in `evalIndexAccess` / `evalIndexAssign` | ″ | chained subscripts; negative indices; out-of-range **raises** | [ ] |
| Nothing else changed ✅ | — | ″ | strings still concat; `==` on lists/dicts unchanged; `&&`/`\|\|` unchanged; **full existing regression green** | [ ] |
| **The benchmark gate** ✅ | `benchmarks/hotpath.b` + a paired-ratio harness **with a control group**: the same byte-identical binary measured against itself, to establish the noise floor before claiming anything about the treatment | — | **PASSED.** Control +0.52%/+0.94%, treatment +0.61%/+0.81% — indistinguishable. The floor is ±1.97%, so the ±2% gate can only be answered with a control; without one, +2.66% reads as a regression that does not exist | [ ] |

## Phase 5 — Linear algebra ✅

`tests/numba_linalg_test.b` **69/69**; regression **75/75**. All three gates pass with room:
500×500 solve residual **1.24e-14** (gate 1e-10), 200×200 SVD reconstruction **1.97e-13**
(gate 1e-12), 1000³ matmul **255 ms / 7.8 GFLOP/s** (gate 500 ms / 4 GFLOP/s). QR orthogonality
7.8e-16, eigh orthogonality 1.1e-15. No defects found — the residual-based test design caught
nothing because there was nothing to catch.


| Item | How | Feature test | Stress test | Status |
|---|---|---|---|---|
| `dot/matmul/inner/outer`, transpose | blocked matmul, 64×64 tiles, `ikj` order; blocked copy-transpose | `tests/numba_linalg_test.b` | 1000×1000×1000 matmul **≤ 500 ms (≥ 4 GFLOP/s)**; batched matmul; non-conformable shapes raise | [ ] |
| LU family | partial pivoting → `solve`, `inv`, `det`, `slogdet`, `matrix_rank`; triangular substitution | ″ | 500×500 solve residual **< 1e-10**; singular and near-singular matrices **raise clearly, never return garbage** | [ ] |
| Cholesky, QR, `lstsq`, `pinv`, `norm`, `trace`, `cond` | Householder QR | ″ | non-SPD input to `cholesky` raises; rank-deficient `lstsq` | [ ] |
| `svd` (one-sided Jacobi), `eigh` (cyclic Jacobi) | N13 | ″ | SVD reconstruction 200×200 **< 1e-12**; `eigh` orthogonality ‖VᵀV−I‖ and ‖AV−VΛ‖ **< 1e-12**; ill-conditioned inputs | [ ] |

## Phase 6 — The package, the arctic bridge, the docs — PARTIAL

Done: the façade and its composed helpers (`tests/numba_pkg_test.b` **51/51**), the package
(`bantu publish` → `bantu add` → bare `include "numba" as np` **verified end to end in a clean
project**), `docs/numba.md`, `samples/numba/` and `tests/run_samples.sh` wired into both CI jobs.
Two defects fixed: `any` and other keywords could not be used as property names (latent for any dict
with a key called `number`/`string`/`delete`), and `samples/blogsite/db.b` called a `sua.sqlite`
method that does not exist — broken on the shipped release, found within minutes of writing the
sample runner.

**Still open: the arctic bridge, arctic's transcendental `col_*` kernels, the sua-concurrency gate
and cross-platform CI observation.**


| Item | How | Feature test | Stress test | Status |
|---|---|---|---|---|
| Composed helpers ✅ | `polyfit/polyval`, `interp`, `gradient`, `cov/corrcoef`, `meshgrid`, moving average — pure Bantu | `tests/numba_pkg_test.b` | degenerate fits; single-point interp | [ ] |
| arctic transcendentals | ~15 `col_sqrt/exp/log/…` natively, on the existing `unaryOp` shape with null propagation | `tests/arctic_transcendental_test.b` | nulls propagate; domain errors → NaN not crash | [ ] |
| The bridge | `nd_from_column` (zero-copy borrow), `nd_to_column` (copy), `nd_from_frame` (F-contiguous); `arctic.b` `to_ndarray()` behind `has_native("ndarray")` | `tests/numba_arctic_bridge_test.b` | round-trip exact; borrow **provably zero-copy via `nd_base_id`**; a column with nulls is refused by name; the borrowed array outliving the column | [ ] |
| Package, docs, gallery ✅ | `numba/{numba.b, numba_test.b, package.json}`, `docs/numba.md`, `samples/numba/`, `tests/run_samples.sh` | — | **every documented example and every sample executed by CI**; `bantu add numba` then `include "numba" as np` works from a clean project | [ ] |
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
