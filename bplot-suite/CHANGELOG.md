# bplot — Changelog

Every entry is prefixed so the log can be grepped: **[design]**, **[feature]**, **[bug fix]**,
**[perf]**, **[test]**, **[docs]**.

Phases do not advance until all five gate tiers are green (see [`ROADMAP.md`](ROADMAP.md)). Measured
numbers are recorded here with the machine they were taken on, because a number without a machine is
not a measurement.

> Unless another machine is named: **Intel Core i7-9750H @ 2.60 GHz, macOS 15.7.9, Apple clang 17**,
> `bantu` built by `build-mac.sh`.

---

## Phase B0 — Tracking docs and the language gaps

- **[design]** `docs/bplot-architecture.md` — the reference design: why the object model, scales,
  ticking, layout and the SVG backend are pure Bantu while only rasterisation is native; the
  measured string-building problem and the two fixes for it; the Figure/Axes/Artist model and the
  three Bantu constraints that shape it; coordinates and the single y-flip; scales; the
  `MaxNLocator` tick algorithm with the two details that are not optional; number formatting; the
  SVG backend and **the security section that is the real teeth of this design**; text metrics; NaN
  and degenerate-input handling; the raster backend; interop; and every rejected alternative with
  its reason.
- **[design]** `bplot-suite/DECISIONS.md` — BP1–BP15.
- **[design]** `bplot-suite/ROADMAP.md` — phases B0–B6, each with its feature test, its stress test
  and its gate.

### Findings recorded during design, before any code

The design work was supposed to confirm that bplot needed **zero** interpreter changes. Measurement
disproved that, and found three defects on the way. All four are general language problems, not
bplot features, so all four are fixed here rather than worked around.

- **Building a string is O(n²).** `$s = $s + $part` in a loop: **20,000 appends 1,116 ms, 40,000
  appends 6,752 ms** — 6.05× the time for 2× the work, because `evalBinaryOp` builds a fresh
  `std::string` holding a copy of the left operand every time. Forty thousand appends to a growing
  1.16 MB buffer is ~23 GB of `memcpy`. A 100,000-element figure would have taken upwards of forty
  seconds, and the B1 stress gate would have failed on the first realistic chart.
- **`max(1, 2, 9)` answers 2.** `max` and `min` read `args[0]` and `args[1]` and ignore everything
  after. Not a crash — a **silently wrong answer**, which is the failure mode worth caring about,
  and it has been in every shipped build.
- **`str(1e21)` answers `"-9223372036854775808"`.** `Value::toString` casts any integral double to
  `long long`; converting a floating-point value outside the destination integer range is undefined
  behaviour (ISO C++ [conv.fpint]) and saturates to `INT64_MIN` on x86-64 and ARM64. Reachable from
  `pow(10, 21)`, and reachable from any numba reduction whose total exceeds 2⁶³.
- **`$a[$i]` is O(n) in the length of `$a`.** `evalNode` returns a `Value` by value, and a `Value`
  holding a list owns its elements inline — reading one element out of a 20,000-element list
  deep-copied all 20,000, each a ~190-byte struct carrying a `std::string`, a `std::vector`, a
  `std::function` and three `shared_ptr`s. **10,000 reads 1,919 ms, 20,000 reads 8,093 ms** — 4.2×
  for 2× the work, 405 µs to read one element. Every loop over a list in every Bantu program was
  quadratic. This is the same class as the `push` defect found in numba's Phase A, in the same type,
  for the same reason.

### Solved defects

- **[bug fix] `max`/`min` ignored every argument after the second.** Both are now variadic, and a
  single list argument is reduced over it. `max(1, 2, 9)` is 9; `max([3, 17, 5])` is 17. The
  two-argument form is unchanged. NaN propagates (NumPy's `max`, not its `nanmax`) because a
  primitive must not silently discard a value it was handed; bad arguments raise a catchable error
  naming the element and its type, where before a string argument was quietly read as 0.
  *Side benefit:* `max($list)` over 100,000 elements is **13 ms** natively, against a Bantu loop
  that does the same work in ~100 ms — which is how bplot takes the range of a large series.

- **[bug fix] `str()` past 2⁶³ was undefined behaviour.** The integral fast path is now range-guarded
  at 9.2e18, below 2⁶³, and larger magnitudes fall through to the stream. `str(1e21)` is `"1e+21"`,
  `str(1e308)` is `"1e+308"`, and `num(str($x))` round-trips at every magnitude. `inf`, `-inf` and
  `nan` are unaffected — they never took the integral path.

- **[feature] `join(list [, sep])`** — the missing inverse of `split()`, and the fix for the O(n²)
  problem. It reserves the exact total length for the all-strings case and appends once. Non-string
  elements stringify as `print` would; a missing separator means `""`; a non-list first argument or
  a non-string separator raises.

  | | `$s = $s + part` | `push` + `join` |
  |---|---|---|
  | 20,000 parts | 1,116 ms | **147 ms** collect + **2 ms** join |
  | 40,000 parts | 6,752 ms | **147 ms** collect + **7 ms** join |
  | 100,000 parts (2.9 MB) | ~42 s extrapolated | **327 ms** collect + **38 ms** join |

  **42× faster at 40,000 parts**, and linear rather than quadratic.

- **[bug fix] `$a[$i]` copied the whole list to read one element.** `evalIndexAccess` now borrows a
  pointer to the live container instead of evaluating it into a temporary, for the cases where that
  is provably safe: the index expression must be a literal or a variable, so if the borrow fails
  half-way the fallback path can re-evaluate it without any observable effect. Index chains
  (`$m[1][2]`) are borrowed recursively.

  | | before | after | |
  |---|---|---|---|
  | 10,000 reads | 1,919 ms | **26 ms** | 74× |
  | 20,000 reads | 8,093 ms | **52 ms** | **156×** |
  | 100,000 reads | ~3.4 min extrapolated | **282 ms** | |

  Linear, where it was quadratic. Deliberately **not** the existing `resolveLValue`, which the
  assignment paths use: its dict branch is `(*objectVal)[key]`, which *creates* the entry on lookup —
  correct for `$d["new"] = 1` and wrong for a read, where it would quietly insert a null every time
  you looked up a key that was not there. The borrowing path uses `find()` and reports failure, and
  a test asserts that a missing key is not inserted. List *semantics* are untouched: `$b = $a;` is
  still a copy.

- **[feature] The scalar maths surface.** `PI TAU E INF NAN` as constants; `exp expm1 log1p log2
  log10 cbrt asin acos atan atan2 hypot sinh cosh tanh asinh acosh atanh trunc sign fmod copysign
  degrees radians clamp isnan isinf isfinite` as functions. Domain and range behaviour is IEEE 754's
  — `acos(2)` is NaN rather than an error, `log(0)` is `-inf`, NaN propagates — and a **non-number
  argument raises**, naming the argument and its type. The older maths builtins (`sqrt`, `sin`,
  `log`, …) read `args[0].numberVal` blind, so `sqrt("hello")` answers 0; they are shipped behaviour
  and are left alone, and the difference is documented.

  `isnan`/`isinf`/`isfinite` close a real gap: these values were always *producible* (`log(0)`,
  `sqrt(-1)`, `pow(10,400)`) and there was no way to *test* for one — which for a plotting library
  means emitting a NaN coordinate, which browsers render as nothing at all.

- **[patch] `numba/numba.b` re-exports the language constants** rather than defining its own. `$INF`
  and `$NAN` were built by dividing one-element arrays, so importing numba allocated two arrays
  before doing anything.

- **[patch] `tests/arctic_lazy_test.b` renamed `$exp` to `$explain`.** Variables and functions share
  one namespace with the `$` stripped, so `$exp = …` replaced the new `exp()` builtin for the rest of
  that program. Harmless there — it never called `exp()` — and a landmine for the next edit.

### Tests

- **[test]** `tests/lang_math_test.b` — **174 assertions, ALL GREEN.** Each function is checked
  against an *identity* it must satisfy (`exp(log(x)) == x`, `sin² + cos² == 1`, `cosh² − sinh² == 1`,
  `asin(sin(x)) == x`, `degrees(radians(x)) == x`) rather than against constants copied from
  somewhere, because copied constants only prove the copying worked. Plus the exact values where
  libm implementations differ, the four quadrants of `atan2`, `hypot(1e200, 1e200)` staying finite
  where the naive form overflows, every domain edge (`asin(2)`, `acosh(0.5)`, `atanh(1)`, `log(0)`,
  `exp(±1000)`), NaN propagation through everything, and the failure modes — a string argument, an
  empty list, an inverted `clamp` range — each of which must *raise*, not answer.

  Numbers are asserted with a relative tolerance, never through `str()`, which gives six significant
  digits and would hide a 1e-7 relative error. The exception is `str()`'s own tests, where the string
  is the contract.

  The three defects each have their own block, including linearity assertions for `join` and for
  list indexing that fail if either becomes quadratic again.

- **[test]** Full regression: **33 `.b` suites + 12 `.sh` suites + the `const_bad` negative fixture,
  all green**, on the same build. These changes touch `Value::toString`, `evalIndexAccess` and two
  shipped builtins, so the whole tree is the gate, not just the new test.

- **[test] ASan + UBSan, clean.** A `-fsanitize=address,undefined -fno-sanitize-recover=undefined`
  build over the language, numba and arctic suites: **no sanitizer report anywhere**. This tier
  earns its place here more than usual, because the index fix holds a raw `Value*` across
  evaluation. A hand-written adversarial script exercises the shapes that could break it — an index
  expression with a side effect, one that reassigns the base, one that pushes 5,000 elements onto
  the base *while it is being indexed* (reallocating the very buffer being borrowed), three-deep
  chains, a dict of lists of dicts, a closure over a local list, a list inside a class instance, and
  a self-referential list. All correct, all clean.

  `numba_linalg_test.b` reports one failure under the sanitizer build — the "1000³ matmul reaches
  4 GFLOP/s" gate — because ASan instrumentation at `-O1` is several times slower than the `-O2`
  binary. It is a throughput gate, not a correctness one; all 68 correctness assertions pass.

### Findings from the verification itself

- **The borrow had to be taken *after* the index expression, not before.** `$c[grow()]`, where
  `grow()` pushes 5,000 elements onto `$c`, reallocates the list's buffer while the index is being
  computed. Borrowing first would have left a pointer into the freed buffer — a use-after-free
  reachable from ordinary script. Evaluating the index first costs nothing and cannot be wrong.
  (`unordered_map` keeps element addresses stable across a rehash, so the scope-chain half of the
  borrow was safe either way; the list buffer is not a `unordered_map` and was not.)

- **[perf] The hot-path benchmark has a ~2.7% position bias, which is larger than the ±2% gate.**
  Running the baseline binary first and the new one second showed the 1M arithmetic loop **2.7%
  slower** in all three rounds — consistent enough to look real. Reversing the order flipped it: the
  new binary measured *faster* in two of three rounds. Best-of across both orderings: **2,593 ms new
  against 2,594 ms baseline**, a difference of 0.04% on a loop neither change touches.

  This confirms, a second time and by a different route, that the ±2% benchmark gate is below this
  harness's noise floor unless the A/B ordering is controlled. **Any future hot-path gate must run
  both orderings**; a single ordering will manufacture a 2–3% result in whichever direction the
  binaries happened to be run.

  Order-controlled, and the only figure that survives it:

  | hot path | baseline | after | |
  |---|---|---|---|
  | 1M arithmetic while loop | 2,594 ms | 2,593 ms | unchanged |
  | 1M comparison loop | 2,459 ms | 2,470 ms | +0.4%, within noise |
  | 500k unary negation | 1,174 ms | 1,197 ms | +2.0%, within noise |
  | `fib(24)` recursive | 1,915 ms | 1,904 ms | −0.6%, within noise |
  | **200k list index read** | **635 ms** | **584 ms** | **8.0% faster** |
  | 100k list index write | 211 ms | 212 ms | unchanged |
  | 50k dict set | 179 ms | 176 ms | unchanged |
  | 100k string concat | 181 ms | 185 ms | +2.2%, within noise |

  (Best-of-three per ordering, then best across both orderings — the floor, not the mean, since a
  mean measures the machine's background as much as the interpreter. The rows marked "within noise"
  are inside the 2.7% position bias this harness demonstrates on a loop neither change touches.)

  The list-index gain holds in **both** orderings (635→595 and 641→584), which is what distinguishes
  it from the arithmetic figure that did not. Note this benchmark indexes a *short* list, so it
  measures only the removed double scope-chain walk and the removed cast; the 74–156× figures above
  are what the same fix does to a list long enough for the copy to dominate.

### Verification still outstanding

- **Cross-platform.** macOS only, as for every numba phase. CI runs Linux and Windows jobs and they
  pick up `tests/*.b` automatically, but no Linux or Windows machine and no Docker is available here,
  so **CI has not been observed**. Stated rather than assumed.
