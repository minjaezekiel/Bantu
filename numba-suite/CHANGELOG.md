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
