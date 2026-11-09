# Stage 6 — `bsci`: scientific computing

Phases **S0–S6** in [`../bsci-suite/ROADMAP.md`](../bsci-suite/ROADMAP.md).
Paired with btorch int8 quantization (designed in
[`btorch-architecture.md`](btorch-architecture.md) §10): edge inference first,
then science. Context:
[`ecosystem-architecture.md`](ecosystem-architecture.md).

---

## 1. Module layout

| file | contains |
|---|---|
| `optimize.b` | `minimize` (BFGS, L-BFGS, Nelder-Mead), `least_squares`, `root`, `curve_fit`, `minimize_scalar` |
| `signal.b` | windows, spectrogram, FIR/IIR filters, `convolve`, `correlate`, `resample`, `find_peaks` |
| `sparse.b` | CSR/CSC matrices over the `sp_` family |
| `stats.b` | distributions, hypothesis tests, descriptive statistics |
| `interpolate.b` | linear, cubic spline, PCHIP, `interp1d`, `interp2d` |
| `special.b` | `gamma`, `lgamma`, `erf`, `erfc`, `beta`, `digamma`, Bessel |

---

## 2. No dependency on `btorch` ([E18])

`bsci.optimize` takes a plain objective function and an **optional** gradient
function, both over numba arrays:

```bantu
$res = sci.minimize(def($x) { return f($x); },        // objective
                    $x0,
                    def($x) { return grad_f($x); },   // optional; null = finite differences
                    { "method": "bfgs", "tol": 1e-8 });
```

Pointing the arrow the other way — `bsci` depending on `btorch` so that
objectives could be autograd graphs — would make every scientific-computing
user install an autograd engine they do not need.

### 2.1 The btorch bridge, as a documented pattern

Five lines, in the tutorial, not in a package:

```bantu
def bt_objective($params_shape) {
    return def($x) {
        $t = bt.tensor(nd_reshape($x, $params_shape), true);
        $loss = my_model_loss($t);
        $loss.backward(null);
        return [$loss.item(), nd_reshape($t.grad, [nd_size($t.grad)])];
    };
}
```

A user with a differentiable objective gets L-BFGS over it without a package
edge. This is tier 3 of the ecosystem's three tiers ([E1]): not everything
useful is a library.

---

## 3. FFT lives in `numba`, not here ([E4])

NumPy has `numpy.fft`, and that is the correct layering: a DFT is an array
transform. So **phase S1 adds kernels to numba** (`nd_fft`, `nd_ifft`,
`nd_rfft`, `nd_irfft`, `nd_fftfreq`), tracked as **D5** in
[`../numba-suite/ROADMAP.md`](../numba-suite/ROADMAP.md), and `bsci.signal`
builds on them.

| size | algorithm |
|---|---|
| power of two | iterative radix-4 with a radix-2 final stage, bit-reversal permutation |
| small prime factors (2,3,5,7) | mixed-radix Cooley–Tukey |
| large prime / arbitrary | Bluestein (chirp-z), via a padded power-of-two convolution |

Bluestein matters more than it looks: without it, `nd_fft` on a 1000-point
signal either errors or silently zero-pads to 1024, and a silently zero-padded
spectrum is wrong in a way that looks plausible. The three-tier structure means
**every length works**, which is the only acceptable contract.

`nd_rfft` for real input is roughly 2× faster and halves the output — the
common case in signal processing, so it is not an afterthought.

**f32 support comes free** from stage 1's templating ([E16]), which is one of
the reasons that ordering was chosen.

---

## 4. Sparse matrices and the `sp_` family

| kernel | why native |
|---|---|
| `sp_matmul` | CSR × dense and CSR × CSR; irregular indexed access, the entire point of sparse |
| `sp_matvec` | the inner loop of every iterative solver |
| `sp_solve` | sparse LU / conjugate gradient |
| `sp_convert` | COO ↔ CSR ↔ CSC, with sorting and duplicate summation |

`sp_convert` summing duplicates rather than overwriting them is the same
distinction as `nd_index_add` versus `nd_put` (btorch B15), and the same trap:
a converter that overwrites produces a plausible matrix that is wrong wherever
two entries share a coordinate. It is tested by the same method — an adjoint-
style identity, `dense(coo_to_csr(x)) == dense_accumulate(x)`.

Primary consumer: `btext`'s vectorizers, which currently return dense arrays
with a documented size warning and migrate to CSR once this lands (see
`btext-architecture.md` §5).

---

## 5. `optimize` — the methods and why these

| method | for |
|---|---|
| **BFGS** | smooth unconstrained, gradient available; the default |
| **L-BFGS** | the same with many parameters, where BFGS's dense Hessian approximation does not fit |
| **Nelder–Mead** | no gradient, non-smooth, few parameters; the honest fallback |
| **Levenberg–Marquardt** (`least_squares`) | nonlinear curve fitting, the single most-requested scientific task |
| **Brent** (`minimize_scalar`) | one dimension, where the general machinery is waste |

Finite-difference gradients when none is supplied, with the step chosen as
`sqrt(eps) * max(|x|, 1)` per component — and a **warning when the objective's
scale makes finite differences unreliable**, because a silently bad gradient
produces a confidently wrong minimum. btorch's B10 again: convergence is not
evidence of correctness.

**Rejected:** constrained optimisation (SLSQP, trust-constr) in stage 6. It is
a large amount of work, the use cases in this project are unconstrained, and a
half-implemented constraint handler is worse than none.

---

## 6. Gates

Differential against SciPy throughout, by the method already proven three times
here: generate reference values offline, commit the fixture, never depend on
Python at test time.

| tier | proves | form |
|---|---|---|
| 1. feature | every function does what it claims, including failure modes | `tests/bsci_*_test.b` |
| 2. **differential** | agreement with SciPy to 1e-10 (1e-6 for iterative methods) | committed fixtures |
| 3. **identity** | mathematical invariants, not just examples | §6.1 |
| 4. stress | 10M-point FFT, 10⁶×10⁶ sparse, 1000-parameter L-BFGS | memory flat across repetition |
| 5. regression | nothing else broke | the whole suite |
| 6. sanitizers | ASan + UBSan | full bsci suite |

### 6.1 Tier 3 — the identities

Stronger than example-based tests, and the method that caught btorch's
`col2im` accumulate bug:

| identity | catches |
|---|---|
| `ifft(fft(x)) == x` to 1e-12 | any transform or permutation error |
| **Parseval**: `Σ\|x\|² == (1/N)Σ\|X\|²` | normalisation convention errors, the most common FFT bug |
| `rfft` agrees with `fft` on the first `N/2+1` bins | the real-input fast path |
| `convolve(a,b)` via FFT == the direct O(n²) sum | the whole signal chain at once |
| known minima of Rosenbrock, Himmelblau, Beale, Booth | every `optimize` method, independently of SciPy |
| CDF/PPF round-trip to 1e-10 | every distribution, both directions |
| `sp_matmul` against a dense reference | sparse indexing |

The FFT normalisation identity earns a special mention: there are three
conventions in circulation (`1/N` on forward, on inverse, or `1/√N` on both),
every one of them produces a spectrum that *looks* right, and only Parseval or
a round-trip distinguishes them.

---

## 7. What is out

| item | why |
|---|---|
| constrained optimisation | §5 |
| ODE/PDE solvers (`integrate.solve_ivp`) | a package-sized domain; split out if ever wanted |
| spatial (KD-trees, Delaunay) | `bkit.neighbors` covers the ML use; geometry is its own thing |
| `scipy.ndimage` | `pillow` territory |
| global optimisation (differential evolution, basin hopping) | add on demand |
| wavelets | no demand |

---

## 8. Open items

| # | item | settled by |
|---|---|---|
| 1 | does FFT want stage 1's threading? | partitions cleanly over the batch axis, not within a transform; measure |
| 2 | sparse solver: direct LU or iterative CG first? | CG is far less code and covers symmetric positive-definite, which is most of it |
| 3 | which distributions are actually needed | survey `bstats`' requirements before S4's list is fixed |
| 4 | is `special` big enough to need its own `sp_`-style kernels? | `erf`/`lgamma` are a few dozen lines of rational approximation each; probably pure Bantu |
