# bsci — Decisions

Ecosystem decisions are `E*`/`C*` in
[`../docs/ecosystem-architecture.md`](../docs/ecosystem-architecture.md);
btorch's are `B*`; numba's are `N*`.

---

### S1 — No dependency on `btorch`
Cites [E18]. `bsci.optimize` takes a plain objective function and an optional
gradient function, both over numba arrays.

Pointing the arrow the other way — so objectives could be autograd graphs —
would make **every scientific-computing user install an autograd engine they do
not need.** A user with a differentiable objective gets L-BFGS over it through
a five-line documented bridge instead:

```bantu
def bt_objective($shape) {
    return def($x) {
        $t = bt.tensor(nd_reshape($x, $shape), true);
        $loss = my_model_loss($t);
        $loss.backward(null);
        return [$loss.item(), nd_reshape($t.grad, [nd_size($t.grad)])];
    };
}
```

This is tier 3 of the ecosystem's three tiers ([E1]): not everything useful is
a library.

---

### S2 — FFT lives in `numba`, not in this package
Cites [E4]. NumPy has `numpy.fft`, and that is correct — a DFT is an array
transform, so the kernels belong with the arrays and `bsci.signal` builds
windows, spectrograms, filters and `resample` on top.

**Rejected:** FFT in `bsci`. It would mean a numba user cannot transform an
array without a scipy-class dependency, which inverts the layering.

Consequence: phase S1 is work in
[`../numba-suite/ROADMAP.md`](../numba-suite/ROADMAP.md) (item D5), and this
package's roadmap records it rather than owning it.

---

### S3 — Three FFT size regimes, because every length must work
Power-of-two radix-4, mixed-radix for small prime factors, and **Bluestein for
arbitrary lengths.**

Bluestein is the one that is tempting to skip, and skipping it is the bug:
without it, `nd_fft` on a 1000-point signal either errors or silently zero-pads
to 1024. **A silently zero-padded spectrum is wrong in a way that looks
plausible** — the amplitudes are close, the peaks are near the right bins, and
nothing warns. The three-tier structure makes "every length works" the
contract.

---

### S4 — Conjugate gradient before direct sparse LU
CG is far less code and covers symmetric positive-definite systems, which is
most of what comes up (normal equations, graph Laplacians, finite-difference
operators). A direct sparse LU with fill-reducing ordering is a substantial
project on its own.

Stated as an ordering, not an exclusion.

---

### S5 — No constrained optimisation in this stage
SLSQP and trust-constr are a large amount of work, the use cases in this
project are unconstrained, and **a half-implemented constraint handler is worse
than none** — it silently returns points that violate the constraints.

---

### S6 — `sp_convert` sums duplicates; it does not overwrite them
The same distinction as `nd_index_add` versus `nd_put` ([B15]), and the same
trap: a COO→CSR converter that overwrites produces a plausible matrix that is
wrong wherever two entries share a coordinate — which, in an assembled
finite-element or co-occurrence matrix, is almost everywhere.

Tested by an identity (`dense(coo_to_csr(x)) == dense_accumulate(x)`) rather
than by example, because an example with no duplicate coordinates cannot see
the bug. This is exactly how btorch's `col2im` accumulate error was caught.
