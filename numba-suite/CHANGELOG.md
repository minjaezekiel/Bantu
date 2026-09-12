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
