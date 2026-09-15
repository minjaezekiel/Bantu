# numba — Numerical Bantu

NumPy-class arrays for Bantu. N-dimensional and strided, with zero-copy views so `reshape`,
`transpose` and `slice` cost nothing at any size; NumPy broadcasting; ~55 element-wise functions;
reductions over any axis; sorting, searching and indexing; and linear algebra written from scratch.

- **Source:** [`numba/numba.b`](../numba/numba.b) (the API) over `bantu-src/compiler/src/ndarray_*.hpp` (the kernels)
- **Tests:** `tests/numba_array_test.b`, `numba_ufunc_test.b`, `numba_reduce_test.b`, `numba_linalg_test.b`, `numba_pkg_test.b`, `lang_native_ops_test.b`, `numba_stress.sh`
- **Design:** [`docs/numba-architecture.md`](numba-architecture.md), [`numba-security.md`](numba-security.md), [`numba-acceleration.md`](numba-acceleration.md)

---

## Three lines to a result

```bantu
include "numba" as np;

$a = np.array([[1, 2], [3, 4]], null);
print($a.sum());
```

```
10  (shape=[], dtype=i64)
```

---

## Why it exists

Bantu's interpreter is tree-walking: a `Value` is a ~190-byte struct and an interpreted loop costs
roughly a microsecond per element. A million-element loop written in Bantu takes about 2.6 seconds.
The same arithmetic through numba takes **13 milliseconds** — because the loop is native and the
data is a flat `double` buffer rather than a million 190-byte objects.

That is the whole bargain, and it is exactly NumPy's: the kernels are native, the library on top is
not. `numba/numba.b` is ordinary Bantu you can read.

**numba is for arrays of about 10,000 elements and up.** Below that, a builtin call costs 1–3 µs of
interpreter overhead and a plain Bantu list is simpler and no slower. The docs say so rather than
letting you find out.

---

## A tour

### Creating

```bantu
$z = np.zeros([3, 4], null);          // f64 by default
$o = np.ones([5], "i64");
$r = np.arange(0, 10, null);          // 0..9, step defaults to 1
$l = np.linspace(0, 1, 11, null);     // 11 points including both ends
$i = np.identity(3);
$q = np.random([1000], null, null);   // uniform in [0, 1)
$n = np.randn([1000], null, null);    // standard normal
```

`np.seed(42)` makes the random functions reproducible. numba carries **its own generator**, so an
unrelated `random()` call elsewhere in your program cannot change your matrix. It is **per thread**,
so inside a sua handler each request seeds its own stream.

### Operators work

```bantu
$x = np.linspace(0, 1, 100, null);
$y = ($x * $x) + $x;
$big = $y[$y > 0.5];                  // boolean mask
$m = np.reshape(np.arange(0, 6, null), [2, 3]);
$m[1][2] = 99;                        // writes through — $m[1] is a view
```

Methods chain, and work identically on any build:

```bantu
$x.multiply($x).add($x).sqrt().mean();
```

### Reductions

```bantu
np.sum($m, null, null);        // everything
np.sum($m, 0, null);           // down the columns
np.sum($m, [0, 1], null);      // several axes at once
np.mean($m, 1, true);          // keepdims — the result broadcasts back
```

`keepdims` is what makes centring one line:

```bantu
$centred = np.subtract($m, np.mean($m, 1, true));
```

### Linear algebra

```bantu
$A = np.randn([500, 500], null, null);
$b = np.ones([500], null);
$x = np.solve($A, $b);
print(np.get(np.max(np.abs(np.subtract(np.matmul($A, $x), $b, null), null), null, null), []));
```

---

## How it works, with real numbers

Measured on Apple silicon, one core. Every number here is produced by a test in `tests/`.

### Element-wise operations are memory-bound, not compute-bound

`$c = $a + $b` on 10M f64 reads 160 MB and writes 80 MB to do 10M additions — an arithmetic
intensity of **0.042 flops per byte**. The ALUs are idle, waiting on DRAM.

| operation, 10M f64 | time |
|---|---|
| `nd_add` | **13 ms** (18.0 GB/s effective) |
| `nd_multiply` | 13 ms |
| `nd_sqrt` | 12 ms |
| `nd_greater` | 14 ms |
| `nd_exp` | 65 ms |
| `nd_sum` | **4 ms** |

A hand-written standalone C++ loop doing the same addition takes 13.37 ms on the same machine, so
numba is within 3% of the kernel's own ceiling and single-core memory bandwidth is the limit. Going
faster needs more cores, not better code — see [`numba-acceleration.md`](numba-acceleration.md),
which also explains why a GPU would make this **slower**.

`nd_exp` is the honest outlier: it calls scalar `libm` where NumPy ships a vectorized `exp`. A
vectorizable polynomial version is deferred behind a ≤ 1 ULP accuracy gate.

### Views are free

| | |
|---|---|
| allocate 10M f64 | 40 ms |
| `reshape` + `transpose` of it | **0 ms** |
| strided slice of it | **0 ms** |

`reshape`, `transpose`, `T`, `ravel`, `slice`, `flip` and `broadcast_to` return a new array sharing
the original's buffer. `np.shares_memory($a, $b)` tells you whether two arrays overlap;
`np.is_view($a)` tells you whether one is a view.

### Summation is accurate, not just fast

A naive accumulator loses about `n · eps`. Summing 10M copies of `0.1` that is roughly **2e-12**
relative — and people diff numba against NumPy, which does pairwise. numba accumulates in a binary
tree, so the error grows as `log n`:

| | relative error |
|---|---|
| numba `sum` of 10M × `0.1` | **2.33e-16** |
| a naive loop | ~2e-12 |

`var` and `std` use Welford, which survives a 1e9 offset that destroys the textbook
sum-of-squares formula.

### Linear algebra

| | measured |
|---|---|
| 1000×1000 matmul | **255 ms — 7.8 GFLOP/s** |
| 500×500 solve | 23 ms, residual **1.2e-14** |
| 200×200 SVD | 253 ms, reconstruction **2.0e-13** |
| QR orthogonality ‖QᵀQ − I‖ | **7.8e-16** |
| eigh orthogonality ‖VᵀV − I‖ | **1.1e-15** |

All hand-written, no BLAS. An optional BLAS flag would gate *speed* rather than *capability*,
producing a two-tier contract where the same program runs several times slower on the binary users
download.

---

## Working with large arrays

A chained expression holds several full-size temporaries at once: `$a.add($b).multiply(2).sqrt()` on
10M f64 peaks at three live 80 MB buffers. Every binary function takes an optional **destination**,
which is what lets a loop run in constant memory:

```bantu
$a = np.random([10000000], null, null);
$b = np.random([10000000], null, null);
$out = np.empty([10000000], null);
$i = 0;
while ($i < 100) {
    np.add($a, $b, $out);             // no allocation at all
    $i = $i + 1;
}
```

`np.add($a, $b, $a)` — accumulating in place — is detected as safe and runs at full speed with no
temporary. A destination that only *partially* overlaps an input is detected too, and computed
through a temporary so the answer is correct rather than garbage.

`np.live_bytes()` reports how much numba is holding. `np.max_bytes(n)` lowers the ceiling for the
current program.

---

## Safety

numba can run inside a sua request handler, where the process is a server. Four things are enforced
from the kernel outward:

- **Shape products are checked for overflow.** `np.zeros([2^22, 2^22, 2^22])` would wrap `size_t` to
  a small number and allocate a tiny buffer that every later kernel writes past. It raises.
- **Total live memory is bounded**, not just one allocation — a loop is how a handler actually
  exhausts a server. The operator sets a hard ceiling with `BANTU_ND_MAX_BYTES`; a script can lower
  its own limit with `np.max_bytes(n)` but **never raise it past** the operator's.
- **Broadcast views are read-only.** Their stretched axes have stride 0, so a write would hit one
  element repeatedly.
- **Every view is bounds-checked at construction**, and every bad argument raises a catchable Bantu
  error naming the builtin — never a crash.

Full detail in [`numba-security.md`](numba-security.md).

---

## Things worth knowing before you hit them

**Bantu has one number type, so `np.array([1.0, 2.0])` gives an `i64` array.** `1.0` and `1` are the
same `Value` and numba cannot tell them apart. Ask for the dtype when it matters:

```bantu
$f = np.array([1.0, 2.0], "f64");
```

Storing NaN or infinity in an `i64` array raises, rather than silently writing `INT64_MIN` or `0`.

**`np.anyof`, not `np.any`.** `any` is a reserved type keyword in Bantu, so `def any(...)` does not
parse. The builtin `nd_any` and the method `$a.any()` both work normally — only the façade function
needed a different name.

**`==` on arrays is an identity comparison, not element-wise.** `if ($a == $b)` keeps meaning what it
says. Use `np.array_equal($a, $b)` or `np.allclose($a, $b, ...)` for the element-wise question.
NumPy made the other choice and then had to make `if arr:` raise; Bantu has no such escape hatch.

**Arrays have reference semantics; lists do not.** `$m[1]` is a view, so `$m[1][2] = 9` changes `$m`.
Use `np.copy($a)` when you want an independent array.

**`i64` values materialise into Bantu as `double`**, so integers above 2^53 lose exactness on the way
out. The arithmetic inside numba stays exact.

**The C-style `for` loop silently stops at 100,000 iterations.** Use `while` for anything longer.
This is a language-wide issue, not a numba one, but numba is where people hit it.

---

## Deliberately not included

- **Eigenvalues of a general non-symmetric matrix (`eig`).** Doing it properly needs balancing,
  Hessenberg reduction, Francis double-shift QR and complex arithmetic the language does not have.
  A fragile `eig` is worse than none, and `eigh` already covers covariance matrices, PCA and graph
  Laplacians.
- **`expm`, `schur`, FFT, sparse matrices.**
- **float32 and complex dtypes.** Three dtypes — `f64`, `i64`, `bool` — cover what people write.
- **A GPU backend.** An element-wise add is memory-bound, and moving the data across PCIe costs more
  than doing the work on the CPU. See [`numba-acceleration.md`](numba-acceleration.md) §2.
- **BLAS.** See above.

## Caveats, plainly

- **SVD is impractical much beyond n ≈ 500–800.** One-sided Jacobi is ~120 lines against ~500 for
  Golub–Kahan and is *more* accurate on small singular values, but it costs several sweeps and is
  3–5× slower than LAPACK.
- **Transcendentals are 2–3× behind NumPy**, which ships vectorized versions where numba calls libm.
- **Everything is single-threaded.** One core cannot saturate DRAM, so a parallel tier would help the
  bandwidth-bound operations — it is deferred until it can be designed against sua's own threads.
- **`argsort` is slower than it should be** (142 ms for 1M against a 80 ms target): it materialises a
  value buffer and an index buffer before sorting. Recorded rather than hidden.
