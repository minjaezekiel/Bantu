# bsci — Roadmap

Scientific computing for Bantu: scipy's shape over numba's arrays. Paired with
btorch int8 quantization — edge inference first, then science.

Design: [`../docs/bsci-architecture.md`](../docs/bsci-architecture.md).
Decisions: [`DECISIONS.md`](DECISIONS.md). Ecosystem context:
[`../docs/ecosystem-architecture.md`](../docs/ecosystem-architecture.md)
(stage 6).

**No dependency on `btorch`** ([E18], [S1]). `bsci.optimize` takes a plain
objective and an optional gradient over numba arrays; the btorch bridge is a
five-line documented pattern, not a package edge.

| tier | proves | form |
|---|---|---|
| 1. feature | every function does what it claims, including failure modes | `tests/bsci_*_test.b` |
| 2. **differential** | agreement with SciPy to 1e-10 (1e-6 for iterative methods) | committed fixtures, generated offline |
| 3. **identity** | mathematical invariants, not just examples | below |
| 4. stress | 10M-point FFT, 10⁶×10⁶ sparse, 1000-parameter L-BFGS | memory flat across repetition |
| 5. regression | nothing else broke | the whole suite |
| 6. sanitizers | ASan + UBSan | — |

### Tier 3 — the identities

Stronger than example-based tests, and the method that caught btorch's
`col2im` accumulate bug:

| identity | catches |
|---|---|
| `ifft(fft(x)) == x` to 1e-12 | any transform or permutation error |
| **Parseval**: `Σ\|x\|² == (1/N)Σ\|X\|²` | **normalisation convention errors — the most common FFT bug** |
| `rfft` agrees with `fft` on the first `N/2+1` bins | the real-input fast path |
| `convolve(a,b)` via FFT == the direct O(n²) sum | the whole signal chain at once |
| known minima of Rosenbrock, Himmelblau, Beale, Booth | every `optimize` method, independently of SciPy |
| CDF/PPF round-trip to 1e-10 | every distribution, both directions |
| `dense(coo_to_csr(x)) == dense_accumulate(x)` | duplicate-coordinate summation ([S3]) |

There are three FFT normalisation conventions in circulation (`1/N` forward,
`1/N` inverse, or `1/√N` on both); **every one produces a spectrum that looks
right**, and only Parseval or a round-trip distinguishes them.

---

## Phase S0 — Tracking docs
- [x] `docs/bsci-architecture.md`
- [x] `bsci-suite/ROADMAP.md`, `DECISIONS.md`

## Phase S1 — FFT kernels — **in `numba`, not here** ([E4])

Tracked as item **D5** in
[`../numba-suite/ROADMAP.md`](../numba-suite/ROADMAP.md). NumPy has
`numpy.fft`, and that is the correct layering: a DFT is an array transform.

| size | algorithm | Status |
|---|---|---|
| power of two | iterative radix-4 with a radix-2 final stage, bit-reversal permutation | [ ] |
| small prime factors (2,3,5,7) | mixed-radix Cooley–Tukey | [ ] |
| large prime / arbitrary | **Bluestein** (chirp-z) via a padded power-of-two convolution | [ ] |
| real input | `nd_rfft`/`nd_irfft` — ~2× faster, half the output | [ ] |

Bluestein matters more than it looks: without it, `nd_fft` on a 1000-point
signal either errors or **silently zero-pads to 1024**, and a silently
zero-padded spectrum is wrong in a way that looks plausible. Every length must
work.

f32 support comes free from stage 1's templating ([E16]).

## Phase S2 — `signal.b`
Windows (Hann, Hamming, Blackman, Kaiser), `spectrogram`, `stft`, FIR design
and `lfilter`, `convolve`, `correlate`, `resample`, `find_peaks`,
`hilbert`.

## Phase S3 — `optimize.b`

| method | for | Status |
|---|---|---|
| **BFGS** | smooth unconstrained with a gradient; the default | [ ] |
| **L-BFGS** | the same with many parameters | [ ] |
| **Nelder–Mead** | no gradient, non-smooth, few parameters | [ ] |
| **Levenberg–Marquardt** (`least_squares`, `curve_fit`) | nonlinear curve fitting — the most-requested scientific task | [ ] |
| **Brent** (`minimize_scalar`) | one dimension | [ ] |
| `root` (Newton, Brent, bisect) | — | [ ] |

Finite-difference gradients when none is supplied, step
`sqrt(eps) * max(|x|, 1)` per component — **and a warning when the objective's
scale makes finite differences unreliable**, because a silently bad gradient
produces a confidently wrong minimum. [B10] again: convergence is not evidence
of correctness.

## Phase S4 — `stats.b`
Distributions (normal, t, chi², F, binomial, Poisson, gamma, beta, uniform,
lognormal) with pdf/cdf/ppf/rvs — `rvs` seeded via `nd_seed` only ([C4]).
Tests: t-test (1-sample, 2-sample, paired), chi², KS, Mann–Whitney, ANOVA,
Shapiro–Wilk. Descriptive: skew, kurtosis, correlation (Pearson, Spearman,
Kendall).

Which distributions are actually needed is surveyed against `bstats`'
requirements before this list is fixed.

## Phase S5 — `sparse.b` and the `sp_` family

| kernel | purpose | Status |
|---|---|---|
| `sp_matmul` | CSR × dense and CSR × CSR | [ ] |
| `sp_matvec` | the inner loop of every iterative solver | [ ] |
| `sp_solve` | conjugate gradient first, direct LU later ([S4]) | [ ] |
| `sp_convert` | COO ↔ CSR ↔ CSC, with sorting and **duplicate summation** | [ ] |
| **register `sp_` in `kNativeFamilies`** | `types.hpp:499` ([B21]) | [ ] |

`sp_convert` summing duplicates rather than overwriting them is the same
distinction as `nd_index_add` versus `nd_put` ([B15]), and the same trap: a
converter that overwrites produces a plausible matrix that is wrong wherever
two entries share a coordinate. Tested by the identity in tier 3.

Primary consumer: `btext`'s vectorizers, which migrate from dense to CSR here
([T5]).

## Phase S6 — `interpolate.b`, `special.b`
Linear, cubic spline, PCHIP, `interp1d`/`interp2d`; `gamma`, `lgamma`, `erf`,
`erfc`, `beta`, `digamma`, Bessel. Probably pure Bantu — a few dozen lines of
rational approximation each.

---

## Where to resume
**S0 done; S1 next**, and S1 is work in `numba` (item D5), not in this package.
