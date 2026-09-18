# numba — the memory-safety model

How the three safety properties claimed in Phase 1 are actually enforced, where they are weaker than
claimed, and what the best available answer is for each. Every finding below was reproduced on a real
build (macOS arm64, `bantu-src/compiler/build/bantu`, branch `numba-foundations`); nothing here is
reasoned-about-but-untested.

The three properties, restated as the threats they exist to stop:

| # | threat | mechanism today |
|---|---|---|
| 1 | a shape whose product wraps `size_t`, allocating a small buffer that kernels then write past | checked multiplication at every product |
| 2 | one allocation exhausting a process that is also a server | a byte ceiling, default 2 GiB |
| 3 | a write through a stride-0 axis hitting one element repeatedly | broadcast results carry `writable = false` |

**Summary of what the measurements found.** Property 1 holds and property 3 holds — property 3 more
completely than had been claimed, having been checked across every view- and copy-producing builtin.
Property 2 **did not hold as described**: the ceiling was per-allocation, and total live memory was
unbounded. Two further defects turned up that were not on the list at all: the limit and the PRNG were
process-global mutable state reached from sua's per-connection threads, and `nd_slice` never
type-checked its arguments and performed signed multiplication that is undefined on overflow.

**Status: all of the above are fixed** (decisions N17–N20, `numba-suite/CHANGELOG.md` Phase 1.1).
Each section below keeps the measurement that motivated the change, then records what was done. The
acceleration counterpart — GPU, threads, huge pages, and why the `memset` in §2(d) must survive
optimisation — is [`docs/numba-acceleration.md`](numba-acceleration.md).

---

## 1. Shape-product overflow

### What is there now

`checkedMul` ([ndarray_native.hpp:157](../bantu-src/compiler/src/ndarray_native.hpp#L157)) uses the
division form, `a > SIZE_MAX / b`, and every product routes through it — `shapeProduct` for the
element count, and again for `count × itemsize` to get bytes. `shapeFrom` additionally rejects
non-finite, negative, non-integral and `> 2^53` extents, and caps rank at 32.

### Measured

All three raise a catchable error:

```
nd_zeros([2^22, 2^22, 2^22])   raised     (element product wraps)
nd_zeros([2^32, 2^32])          raised     (element product wraps exactly to 0)
nd_zeros([2^61])                raised     (element product fine; ×8 for BYTES wraps)
```

The third is the one worth keeping a test on: the element count `2^61` is a perfectly representable
`size_t`, and only the conversion to bytes overflows. A check placed on the shape product alone would
pass it.

The rank cap of 32 is not arbitrary — **CVE-2021-33430** in NumPy was a buffer overflow in
`PyArray_NewFromDescr_int` reached by "specifying arrays of large dimensions (over 32)". NumPy 1.x
used `NPY_MAXDIMS = 32`; NumPy 2 raised it to 64. Staying at 32 is the conservative choice and costs
nothing real.

### Best available answer, and why it is mostly *not* a change to `checkedMul`

The obvious upgrade is `__builtin_mul_overflow`, which compiles to one `mul` plus `jo` on x86-64 and
`umulh`+`cmp` on arm64, against a 20–40 cycle division. C23 standardises the same thing as `ckd_mul`
in `<stdckdint.h>`; MSVC has neither and needs `_umul128` or the division fallback.

**That upgrade is close to pointless here, and saying so is more useful than making it.** These
products are computed once per array creation, against a `nd_zeros([10000000])` that takes 47 ms.
Daniel Lemire's write-up on the same question notes that optimising compilers frequently convert the
division form into comparisons on their own. Switching would buy nanoseconds on a millisecond
operation. It is worth doing only as part of a helper that also standardises the **signed** case,
which is where the actual defect is (§5).

**The upgrade that matters is moving the invariant from the callers into one chokepoint.**
`makeView` ([ndarray_native.hpp:414](../bantu-src/compiler/src/ndarray_native.hpp#L414)) validates
*nothing*. It takes a shape, a stride vector and an offset from the caller and trusts all three.
Today that is survivable because there are only seven view constructors and each one derives its
strides from an already-valid parent — the invariant is maintained by construction rather than by
checking. Phases 2 through 5 add many more, and `nd_slice` already gets it wrong (§5).

NumPy's analogue is `PyArray_CheckStrides`, and its own documentation for `as_strided` is the
argument for having one:

> it manipulates the internal data structure of ndarray and, if done incorrectly, the array elements
> can point to invalid memory and can corrupt results or crash your program.

The check is the addressable extent of the view, in elements:

```cpp
// Every view must be contained by its buffer. O(ndim), once per view construction,
// against kernels that then touch millions of elements.
inline void checkExtent(const NdArray& v, const char* what) {
    ptrdiff_t lo = (ptrdiff_t)v.offset, hi = (ptrdiff_t)v.offset;
    for (size_t d = 0; d < v.shape.size(); d++) {
        if (v.shape[d] == 0) return;                     // empty addresses nothing
        ptrdiff_t span;                                  // (extent-1) * stride, CHECKED
        if (__builtin_mul_overflow((ptrdiff_t)(v.shape[d] - 1), v.strides[d], &span))
            throw std::runtime_error(std::string(what) + ": stride arithmetic overflows on axis "
                                     + std::to_string(d));
        if (span < 0) lo += span; else hi += span;
    }
    const size_t item = itemsize(v.dtype);
    if (lo < 0 || (size_t)(hi + 1) * item > v.buf->nbytes)
        throw std::runtime_error(std::string(what) + ": view extends outside its buffer");
}
```

Called at the end of `makeView`, this makes every present and future view safe by construction, and
it catches the `nd_slice` overflow in §5 as a side effect rather than needing its own fix.

**Done (N19).** `checkExtent` runs inside `makeView`, so no view constructor can skip it.
`checkedMul` keeps its division form for unsigned products; stride arithmetic uses `checkedMulSigned`
on `__builtin_mul_overflow`, because signed overflow is undefined behaviour and so — unlike the
unsigned case — cannot be detected by inspecting the result afterwards.

---

## 2. The allocation ceiling — the one that did not hold

### What is there now

`Buffer`'s constructor rejects `bytes > maxBytesRef()`, default 2 GiB, adjustable from script with
`nd_max_bytes(n)`.

### Measured

`nd_zeros([1e15])` raises, as claimed. But the ceiling is **per allocation**, and nothing tracks the
total:

```
default ceiling                                    2147483648   (2048 MB)
held 12 live arrays = 2400 MB                      no error raised
```

Twelve 200 MB arrays, each far under the ceiling, held live simultaneously. There is no bound at all
on what a loop can accumulate. The stated purpose — "numba may run inside a sua handler where an
unbounded allocation kills a server" — is not met by a per-allocation limit, because the way a
handler kills a server is a loop, not a single call.

### Four distinct problems, in order of severity

**(a) Per-allocation instead of total-live.** The fix is an accounting allocator: one process-wide
`std::atomic<size_t>` incremented in the `Buffer` constructor and decremented in the destructor,
with the admission test on `live + bytes`. Two atomic operations against a 47 ms allocation is not a
measurable cost. This is the model GraalVM's sandbox uses for embedded languages — limit the
context, not the individual call.

The destructor decrement must be exception-safe: increment only *after* the allocation succeeds, or
a failed `posix_memalign` leaks accounted bytes that are never returned and the process slowly
refuses to allocate anything. That failure mode is worse than no limit, because it is invisible.

**(b) A script can raise its own ceiling.** `nd_max_bytes(1e18)` is callable from any Bantu code, so
the limit guards against *mistakes*, not against hostile or untrusted script. As shipped it is a
seatbelt, not a sandbox, and the docs should not imply otherwise.

CPython's answer to exactly this class of problem — CVE-2020-10735, quadratic `int(str)` conversion —
is the precedent worth copying, and it has three surfaces, not one: the environment variable
`PYTHONINTMAXSTRDIGITS`, the command-line `-X int_max_str_digits`, and the runtime
`sys.set_int_max_str_digits()`, with `0` meaning disabled. The operator sets the first two; script
code can only reach the third.

Adopt the shape, not just the setter:

- `BANTU_ND_MAX_BYTES` read once at startup, establishing a **hard ceiling**.
- `nd_max_bytes(n)` may only ever *lower* the effective limit, never raise it past the hard ceiling.
  A ratchet is what turns an advisory number into a control.
- `0` disables, matching CPython, for the single-tenant batch case where the limit is just in the way.

CPython also went out of its way to name the setter in the error message
([gh-96875](https://github.com/python/cpython/pull/96875)). numba's message already does this
(`"raise it with nd_max_bytes(n) if you meant it"`) and it should keep doing so — but once the
ratchet exists the message must distinguish "you can raise this" from "the operator has capped this",
or it will advise a fix that cannot work.

**(c) It is global mutable state, and sua reaches it from threads.** `maxBytesRef()` and `rng()` are
both function-local statics, i.e. process-global. sua accepts a connection and runs the Bantu handler
in a detached thread:

```cpp
// server.hpp:863
std::thread(&SuaServer::handleConnection, this, clientSock, clientAddr).detach();
```

with `Value result = handler(reqObj)` inside it. So two concurrent requests share one limit and one
PRNG:

- concurrent unsynchronised writes to a plain `size_t` are a data race, formally UB;
- `nd_max_bytes` in one request silently changes the limit for every other in flight;
- **`nd_seed` in one request shifts the stream for every other** — which destroys the reproducibility
  that was the entire stated reason for numba owning its own PRNG rather than using the global
  `random()` ("a shared stream means an unrelated `random()` call elsewhere in a program silently
  changes your matrix", `numba-suite/CHANGELOG.md`). That argument applies verbatim to sharing
  numba's own stream across request threads, which is what happens today.

The minimum fix is `std::atomic<size_t>` for the limit and `thread_local` for the `Rng`. `thread_local`
is the better answer than a mutex here: it is faster, it removes the race, and it gives each request
its own reproducible stream, which is the semantics a user actually wants. It does mean a seed must
be set per thread, which needs documenting.

Worth noting this is not numba's defect alone — it is the first thing in the tree to put *mutable*
process-global state behind a builtin, so it is the first to expose that sua's threading model and
the interpreter's globals were never reconciled. Any other package that adds a global will hit it.

**(d) The ceiling only bounds RSS because of an accident.** On Linux with default overcommit,
`posix_memalign` of 2 GiB succeeds without committing a page; the OOM killer arrives later, at first
touch, killing a process that never saw an error. What makes numba's ceiling bind real memory is that
`Buffer`'s constructor does `memset(data, 0, bytes)`, touching every page immediately.

That is load-bearing and currently undocumented, which makes it fragile — the obvious performance fix
is to switch large allocations to `calloc`, which serves them from mmap'd zero pages and makes
`nd_zeros([10000000])` close to free instead of 47 ms. NumPy does exactly this, splitting
`npy_alloc_cache` from `npy_alloc_cache_zero`. Taking that optimisation would silently convert a
hard, immediate, catchable failure into a deferred OOM kill.

**Done (N20).** The `memset` is kept and the reason is recorded at the `memset` itself, here, and in
`docs/numba-acceleration.md` §4, so nobody removes it as dead work. The 47 ms is buying deterministic failure. Separately, make `nd_empty` genuinely
uninitialised — it is currently memset like everything else, which makes the name a lie — and let the
accounting counter from (a) provide its DoS bound instead. That matters for Phase 2, where every
binary ufunc allocates a result buffer it is about to overwrite completely.

**Done (N17, N18, N20).** Admission is tested against `live + requested` via an atomic counter
(the same probe now admits 10 arrays and refuses the 11th); `BANTU_ND_MAX_BYTES` sets a hard ceiling
that `nd_max_bytes()` can only lower, and the error message says which of the two applies; the
counters are atomic and the `Rng` is `thread_local`; `nd_live_bytes()` exposes the counter. A failed
allocation returns its accounted bytes before unwinding, gated by a stress assertion that 20,000
refused allocations leave the counter at **exactly** its baseline.

### Where this deliberately differs from NumPy

NumPy imposes **no allocation ceiling whatsoever** and offers a pluggable allocator (NEP 49,
`PyDataMem_Handler`) so an embedder can impose one. That is the right default for NumPy, which
assumes a trusted single-tenant process. Bantu arrays can run inside a request handler on a shared
server, so numba is deliberately stricter, and the ratchet in (b) is the part NumPy has no analogue
for.

---

## 3. Stride-0 broadcast views — sound, and more thoroughly than claimed

### Measured

`broadcast_to` returns `writable = false` and `nd_set` through it raises. The question that actually
matters is whether read-only-ness *survives every other operation*, since an escape hatch anywhere
makes the flag decorative. Checked across every builtin that takes an array and returns one:

| from a `writable=false` base | result | correct? |
|---|---|---|
| `nd_T`, `nd_flip`, `nd_expand_dims`, `nd_slice`, `nd_swapaxes` | read-only | yes — views, inherit via `makeView` |
| `nd_reshape`, `nd_ravel` (non-contiguous input) | **writable** | yes — verified a *different* `base_id`; they copied |
| `nd_copy`, `nd_ascontiguous`, `nd_astype` | writable | yes — independent buffers |

The two writable results were the suspicious ones, so they were checked by buffer identity rather
than by reading the code: `nd_base_id` differs from the source's, and writing 7 through the reshaped
array left the original at 0. They are copies, and a copy of a read-only array is legitimately
writable. This matches NumPy's own rule — a view inherits WRITEABLE from its base, and a view of a
locked array may not be made writeable.

So property 3 holds. No change was needed to keep it holding today; the items below are about not
losing it in Phase 2.

**Done.** `requireWritable()` is now the single gate (N19), and `nd_shares_memory()` ships, reusing
`checkExtent`'s extent walk. `nd_as_strided` is recorded as deliberately never-shipped.

### Keeping it

**Centralise the check.** `writable` is currently tested in exactly one place, `nd_set`. Phase 2 adds
`out=` to every binary ufunc and Phase 3 adds `nd_put` and boolean-mask assignment — each a new write
path, each able to forget. A `requireWritable(arr, what)` called at the top of every mutating builtin
costs nothing and cannot be forgotten if the kernel entry point takes its output through it.

**Never ship `as_strided`.** NumPy's is the single most dangerous function in the library, documented
as such, and it exists because NumPy must support arbitrary externally-produced strides. numba has no
such obligation. Not exposing it is precisely what makes the `checkExtent` invariant in §1
maintainable — every stride in the system is derived by numba from a valid parent. This should be a
recorded decision, not an accident of not having got to it yet.

**Do not add a `setflags`-style writable override.** NumPy permits setting WRITEABLE back to True when
the array owns its memory; there is no use case here that `nd_copy` does not serve more safely.

**The remaining exposure is self-overlap that is not broadcast**, which arrives with `out=` in Phase 2:
`nd_add($a, $b, $a_shifted_view)` where the output overlaps an input produces garbage, not an error.
NumPy's history here is the cautionary tale — `broadcast_arrays` returned *writable* internally
overlapping views for years and needed a multi-release deprecation
([gh-12609](https://github.com/numpy/numpy/pull/12609),
[gh-13974](https://github.com/numpy/numpy/issues/13974)) to make them read-only. Starting read-only
was the right call; the lesson is that loosening it later is nearly impossible.

The mechanism NumPy settled on is `np.shares_memory` / `may_share_memory`. numba can get the same
thing almost free: the `lo`/`hi` extent computation from `checkExtent` is exactly what a
`nd_shares_memory(a, b)` needs — same buffer, and overlapping `[lo, hi]` ranges. One function, two
uses, and it gives Phase 2 a principled aliasing test rather than a pointer-equality guess.

---

## 4. What this changes about Phase 2

Phase 2 is where kernels start writing through strides in bulk, which is what turns any of the above
from a wrong answer into a memory-safety bug. Three items should land **before** the ufunc kernels,
not after:

1. ~~`checkExtent` inside `makeView`, plus `checkedMulSigned` for stride arithmetic~~ — **done** (N19).
2. ~~The accounting allocator and the env-var ratchet~~ — **done** (N17).
3. ~~`requireWritable` as the single write-path gate, and `nd_shares_memory` for `out=` aliasing~~ —
   **done** (N19).
4. ~~`std::atomic` for the limit, `thread_local` for the `Rng`~~ — **done** (N18).

What Phase 2 still owes on this front: `out=` must call `requireWritable` and `nd_shares_memory` on
every destination, and `nd_empty` must become genuinely uninitialised while still counting against
the live-byte accounting (N20).

An ASan + UBSan build over the numba suite is the tier-5 gate that would have caught §5 without a
hand-written probe, and it is worth standing up before the kernel work rather than after.

---

## 5. Found while verifying: `nd_slice` does not validate its arguments

Not one of the three properties, but it is in the same class and it undermines §1, so it belongs here.

`nd_slice` ([ndarray_native.cpp](../bantu-src/compiler/src/ndarray_native.cpp)) reads `.numberVal`
off each of `start`, `stop` and `step` without ever calling `isNumber()`, passes the result through
`std::llround` — undefined for values outside `long long` — and then computes `strides[d] * step` as
plain signed multiplication, where overflow is undefined behaviour.

Measured on a `(4, 8)` array whose axis-0 stride is 8:

```
nd_slice(m, [[null, null, -4611686018427387904], null])
    -> shape=[1,8]  strides=[0,1]        8 * -2^62 wrapped to 0
nd_slice(m, [[pow(10,300), null, null], null])
    -> shape=[0,8]                       silently empty, no error
nd_slice(m, [[[1,2], null, null], null])
    -> shape=[4,8]                       a LIST accepted as an index; .numberVal read as 0
```

The first is the pointed one: signed overflow produced **stride 0 on an array still marked
writable** — the exact configuration `broadcast_to` goes out of its way to forbid. It is contained
here only because the count computation clamped the extent to 1, so nothing can step along that axis.
That is luck, not design, and it is the clearest possible argument for §1's recommendation: the
safety of a view should not depend on clamping arithmetic that is itself undefined.

**Done (N19).** `nd_slice` rejects non-numbers and bounds every argument to `±2^53` before
`llround`, and both user-influenced products use `checkedMulSigned`. The guard is reachable and
tested rather than merely present: a step of `2^53` — a legal argument — still overflows the stride
product on a 2048-wide axis and raises. `checkExtent` in `makeView` backstops all of it.

---

## Sources

- [NumPy — memory management / NEP 49](https://numpy.org/doc/stable/reference/c-api/data_memory.html)
- [NumPy — `as_strided` warnings](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html)
- [NumPy — `ndarray.flags`, WRITEABLE rules](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html)
- [NumPy gh-13974 — deprecating writable `broadcast_arrays`](https://github.com/numpy/numpy/issues/13974)
- [CVE-2021-33430 — `PyArray_NewFromDescr_int`, dimensions over 32](https://github.com/advisories/GHSA-6p56-wp2h-9hxr)
- [CVE-2020-10735 / CPython `set_int_max_str_digits`](https://python-security.readthedocs.io/vuln/large-int-str-dos.html)
  and the [fix commit](https://github.com/python/cpython/commit/511ca9452033ef95bc7d7fc404b8161068226002)
- [GCC — integer overflow builtins](https://gcc.gnu.org/onlinedocs/gcc/Integer-Overflow-Builtins.html)
- [Lemire — checking multiplication overflow](https://lemire.me/blog/2026/05/06/checking-multiplication-overflow/)
- [Linux — overcommit accounting](https://www.kernel.org/doc/html/v5.1/vm/overcommit-accounting.html)
- [GraalVM — sandbox resource limits](https://www.graalvm.org/22.0/reference-manual/embed-languages/sandbox-resource-limits/)
