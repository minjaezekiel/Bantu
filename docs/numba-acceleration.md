# numba — making it fast: GPU, threads and RAM

Where numba's remaining performance actually is, which of the obvious accelerators are worth their
cost, and what the current design must not preclude. Written before the Phase 2 kernels, because two
of the findings change Phase 2's design and one changes a Phase 2 *gate*.

**The conclusion up front, because it is the opposite of the intuitive answer:** a GPU would make
numba's headline operation **slower**, not faster, and the speed that is actually available — roughly
2× on the ordinary `$c = $a + $b` case — is in memory-system work that needs no new dependency at all.
GPU earns its place only in Phase 5 linear algebra, and only as an optional backend.

---

## 1. What numba is actually limited by

`c = a + b` on 10M f64 reads 160 MB and writes 80 MB to perform 10M additions. That is an
**arithmetic intensity of 0.042 flop/byte**, against hardware that can do tens of flops per byte of
bandwidth. On the roofline it sits far to the left of every machine balance point: the ALUs are idle,
waiting on DRAM. Nothing that adds compute throughput — SIMD width, more cores' worth of FLOPs, a
GPU's thousands of lanes — addresses the actual constraint.

That single number drives every conclusion below. Operations divide cleanly:

| class | intensity | limited by | accelerator that helps |
|---|---|---|---|
| ufuncs (`add`, `mul`, comparisons) | ~0.04 flop/byte | DRAM bandwidth | memory-system work (§3) |
| transcendentals (`exp`, `sin`) | ~5–20 flop/byte | ALU / libm | SIMD polynomials (Phase 7) |
| reductions (`sum`, `mean`) | ~0.12 flop/byte | DRAM bandwidth | memory-system work |
| **matmul, solve, SVD** | **O(n) flop/byte** | **ALU** | **threads, blocking, GPU** |

Only the last row is compute-bound, and it is the only row where a GPU has anything to offer.

---

## 2. GPU: the honest analysis

### Why it loses on the common case

To add two arrays on a discrete GPU you must move them there. PCIe 4.0 ×16 carries ~25 GB/s in
practice; host DRAM delivers 50–100 GB/s on a desktop and ~200 GB/s on Apple silicon. So the transfer
alone costs **more than doing the whole operation on the CPU** — before the kernel launches, before
anything comes back. A single `$a + $b` on the GPU is strictly slower, and no amount of kernel
tuning changes it, because the loss happens on the bus.

This is not a marginal call. CuPy — the most mature NumPy-on-GPU implementation there is — tells
users plainly not to use it below ~10k elements, and reports break-even against NumPy somewhere
between 1000×1000 and 5000×5000 *for matmul*, which is the friendliest possible case.

GPUs win when data becomes **resident**: uploaded once, then hit by many kernels before coming back.
That is a different programming model, not a transparent speedup. It is why CuPy has a distinct
`cupy.ndarray` type and an explicit `asnumpy()`, rather than making `numpy.ndarray` faster. Any
honest numba GPU story requires the same — a separate device-array type and explicit transfers — and
that is a large surface for a benefit confined to Phase 5.

### The dependency cost, against the house rule

Bantu links no runtime dependency it could write itself: OpenSSL was declined in favour of a
from-scratch P-256, libuv was declined for sua, and BLAS was declined for numba precisely because a
`BANTU_BLAS` flag would gate *speed* rather than *capability*, producing a two-tier contract where
the same program runs several times slower on the binary users actually download.

A GPU backend is that objection several times over. CUDA means a toolkit and a vendor driver; Metal
means an Objective-C/Swift framework and macOS only; ROCm means AMD only; SYCL and OpenCL mean an
ICD loader and a runtime that may be absent. All of them mean the shipped binary either carries the
dependency or silently has no GPU — and the gap between the two is a multiple, not a few percent.

**Recommendation: no GPU backend in phases 1–6.** Revisit at Phase 7 only if Phase 5 linear algebra
proves a measured wall on real workloads, and then as `BANTU_METAL` / `BANTU_CUDA` gating a
`nd_to_device()` / `nd_to_host()` residency model — never as a transparent accelerator for ufuncs,
which cannot be made to pay.

### What Apple silicon changes, and the one thing worth doing now

Unified memory removes the transfer entirely: CPU and GPU address the same physical pages, and
`newBufferWithBytesNoCopy` wraps an existing host pointer as a `MTLBuffer` with no copy at all. That
turns the argument above from "the bus kills it" into an ordinary question of which processor is
better at the kernel — and for bandwidth-bound work on an SoC sharing one memory controller, the
answer is often "neither, they are both waiting on the same DRAM".

But it comes with a concrete constraint worth recording now, because it is nearly free to satisfy and
impossible to retrofit: **`newBufferWithBytesNoCopy` requires page-aligned memory**, and ordinary
`malloc`/`posix_memalign` returns arbitrary or 64-byte boundaries.

numba currently aligns to 64 bytes ([ndarray_native.hpp](../bantu-src/compiler/src/ndarray_native.hpp)).
Raising that to page alignment for allocations above a threshold costs one `posix_memalign` argument
and serves **three** independent purposes: zero-copy Metal later, huge-page promotion now (§3.2), and
non-temporal stores now (§3.1). That is the whole of "account for GPU in the design" — one alignment
constant, taken for reasons that pay off even if no GPU backend is ever written.

**Nothing else in the current design precludes a GPU.** `Buffer` already separates ownership from
the array (`owned`, plus `keepalive` for borrowed memory), which is exactly the seam a device
allocation would slot into. It does not need a `device` enum today; adding one now would be
speculative generality for a feature deliberately deferred.

---

## 3. Where the speed actually is

### 3.1 Non-temporal stores — the single largest win, and free

When a kernel writes `c[i]`, the CPU first *reads* the cache line containing `c[i]` from DRAM so it
can own it for modification — a **read-for-ownership**. For `c = a + b` that is a third stream of
traffic nobody asked for: two arrays read, one written, one read pointlessly.

Non-temporal (streaming) stores write straight to memory and skip the RFO. Data-volume arithmetic
predicts a 1.33× speedup on a triad from dropping four streams to three; the **measured** figure on
STREAM Triad is **1.40–1.42×**, better than predicted because memory-subsystem efficiency itself
improves with fewer concurrent streams.

That is a ~40% improvement on numba's single most common operation, needing no dependency and no new
runtime — `_mm256_stream_pd` on x86-64, `stnp` on AArch64, or `__builtin_nontemporal_store` which
both GCC and Clang provide portably.

The caveats are real and must be encoded, not assumed away:
- **Only for destinations that do not fit in cache.** For a small array the RFO is free (the line is
  already resident) and bypassing the cache throws away a value the next operation wants. Needs a
  size threshold, tuned, not guessed.
- **Only for contiguous, aligned, whole-line writes.** A partial line forces a read-modify-write and
  is slower than an ordinary store. This means tier-0 of the kernel loop only.
- **Requires a fence** (`_mm_sfence`) before the result is read by other code.

**Recommendation: Phase 2, tier-0 loop only, above a measured threshold, with the scalar path kept
for everything else.**

### 3.2 Huge pages

A 10M-element f64 array is 80 MB = 20,000 pages of 4 KB. Walking it touches far more TLB entries than
any TLB holds, so a large fraction of accesses take a page-table walk. With 2 MB pages the same array
is 40 entries, and coverage rises from roughly 6 MB to ~3 GB.

Reported effects range from 10–50× fewer TLB misses to a 2.42× speedup on pointer-chasing workloads;
streaming access patterns like numba's benefit less, because hardware prefetchers hide much of the
walk, but the effect is real and costs one `madvise(MADV_HUGEPAGE)` call on Linux above a size
threshold. macOS has `VM_FLAGS_SUPERPAGE_SIZE_2MB`; Windows requires a privilege and should be
skipped.

**Recommendation: Phase 2, behind the same size threshold and the same page alignment as §3.1.**
Measure it — do not assume it. THP has a real history of *hurting* latency-sensitive workloads via
allocation stalls, which is why several databases advise disabling it system-wide; numba's usage
(large, long-lived, sequentially-scanned buffers) is the pattern it suits best, but that is a
hypothesis until benchmarked.

### 3.3 Threads — and a Phase 2 gate that needs revising

This is the finding that changes a stated commitment. `docs/numba-architecture.md` §3 concluded that
"SIMD, OpenMP and threading are therefore not needed for the headline op", reasoning that the
operation is bandwidth-bound rather than ALU-bound. The premise is right and the conclusion does not
follow:

> A single core of modern multi- and many-core systems cannot saturate main memory bandwidth.

A core sustains only about ten concurrent L1 misses, which caps one thread at roughly
`10 lines × 64 B / latency`. At a 79 ns memory latency that is **~8.1 GB/s from a single core** on a
machine whose DRAM will deliver many times that. Bandwidth-bound does not mean single-threaded is
enough; it means you need enough cores to *fill the memory pipeline*, and then no more.

**This puts a Phase 2 gate at risk.** The roadmap requires *10M `nd_add` ≥ 10 GB/s effective*.
Single-threaded on Apple silicon that passes comfortably (low latency, deep memory-level parallelism).
Single-threaded on a shared cloud vCPU with high memory latency — the machine the Linux binary
actually runs on — 8.1 GB/s is the plausible ceiling and the gate fails on hardware, not on code
quality. Either the gate becomes per-platform, or Phase 2 ships a parallel tier-0.

OpenMP remains correctly rejected: it links `libgomp`, a new runtime dependency, and needs
`brew install libomp` for every contributor. But the alternative is not "stay single-threaded" — it
is a small `std::thread` parallel-for over the flat index range. `std::thread` is already used in the
tree ([server.hpp:863](../bantu-src/compiler/src/server.hpp#L863)), so this adds **no dependency
whatsoever**.

Two hard requirements if it lands:
- **A threshold, not blanket parallelism.** Thread wake-up is microseconds; below a few hundred
  thousand elements a parallel-for is pure loss.
- **A fixed-size pool created once**, never a thread per call, and it must not fight sua's
  per-connection threads — numba running inside 64 concurrent request handlers must not spawn
  64 × N workers. This is exactly the interaction that made `thread_local` the right choice for the
  PRNG, and it needs the same care.

**Recommendation: prototype in Phase 2, gate on measurement, and fix the roadmap's bandwidth target
to be per-platform in the meantime.**

### 3.4 A free-list for buffers

Every `nd_zeros` is a fresh `posix_memalign` plus a full `memset`; every drop returns the pages to
the OS, so the next allocation takes the page faults again. NumPy solves this with a small size-keyed
cache (`npy_alloc_cache` / `npy_free_cache`) precisely to avoid this thrashing, and it matters most
for the temporaries a chained expression creates — `$a.add($b).mul(2).sqrt()` allocates and frees
three 80 MB buffers.

This interacts directly with the accounting allocator added for the ceiling: **cached bytes are not
live bytes**, and conflating them would make the ceiling reject allocations that the cache could
have served. The free list must decrement the live counter on release and re-increment on reuse,
with the cache itself capped separately.

**Recommendation: Phase 2 or 3, once `out=` exists** — `out=` removes the temporaries that make the
free list valuable, so do that first and re-measure.

### 3.5 Out-of-core arrays

`Buffer`'s borrowing constructor (`Buffer(void* p, size_t bytes, shared_ptr<void> owner)`) already
supports memory it does not own, which is the whole structural requirement for an `nd_memmap` over a
file — NumPy's `np.memmap` model, where the OS page cache does the paging.

Worth recording that it is *possible* and cheap, and equally that it is a trap for random access:
disk is 100–1000× slower than RAM, so it pays only for sequential or chunked single passes. Also
worth noting that mmap reserves virtual address space for the whole array up front.

**Recommendation: not scheduled. The seam exists; build it when someone has the file.**

---

## 4. The `calloc` trap — do not take this optimisation

`Buffer`'s constructor ends with `memset(data, 0, bytes)`, and the obvious optimisation is `calloc`,
which for large blocks is served from the kernel's pre-zeroed pages and makes `nd_zeros([10000000])`
close to free instead of ~40 ms. NumPy does exactly this, splitting `npy_alloc_cache` from
`npy_alloc_cache_zero`.

**Taking it would silently break the allocation ceiling.** Under Linux's default overcommit, a large
`posix_memalign` or `calloc` succeeds without committing a single page; the memory is charged to the
process only when touched. So with `calloc`, `nd_zeros` returns successfully, the ceiling's
bookkeeping says everything is fine, and the process is killed by the OOM killer later — during a
kernel, with no exception, nothing for a Bantu `try/catch` to catch, and inside a sua handler that
means the worker dies mid-request.

The `memset` is what touches every page **at the point of allocation**, converting a deferred,
uncatchable kill into a `std::bad_alloc`-shaped error raised where the user can see which line
caused it. It is buying deterministic failure, and ~40 ms per 10M f64 is the price.

This is now recorded in three places so it cannot be removed as dead work: a comment at the `memset`
itself, [`docs/numba-security.md`](numba-security.md) §2, and here.

**The one safe exception is `nd_empty`**, whose contract is explicitly "uninitialised" and which today
memsets anyway — making the name a lie and the function pointlessly as slow as `nd_zeros`. Phase 2's
ufunc results are exactly this case: a destination buffer about to be overwritten in full. The DoS
bound for those comes from the **live-byte accounting**, which is independent of whether pages are
touched, so skipping the memset there loses nothing that matters. It does mean an `nd_empty` buffer
can exceed the ceiling's intent in *committed* terms, which is acceptable precisely because it is
about to be written.

---

## 5. What this changes

| item | phase | why |
|---|---|---|
| Page-align large allocations | 2 | serves NT stores, huge pages, and any future zero-copy Metal — one constant |
| Non-temporal stores in tier 0 | 2 | measured 1.40–1.42× on the most common operation, no dependency |
| `std::thread` parallel tier 0 | 2 | one core cannot saturate DRAM; the current gate may be unreachable without it |
| Revise the "≥ 10 GB/s effective" gate to be per-platform | 2 | it encodes an assumption about single-core bandwidth that does not hold everywhere |
| `madvise(MADV_HUGEPAGE)` above a threshold | 2 | measure; plausible, not proven, for streaming patterns |
| True uninitialised `nd_empty` | 2 | `out=` and every ufunc result; accounting still bounds it |
| Buffer free list | 2–3 | after `out=`, which removes most of the temporaries |
| GPU backend | 7, conditional | only for Phase 5 linalg, only on a measured wall, never for ufuncs |
| `nd_memmap` | unscheduled | the seam exists; no demand yet |

---

## Sources

- [Roofline model](https://modal.com/gpu-glossary/perf/roofline-model)
- [CuPy vs NumPy — break-even and transfer overhead](https://carpentries-incubator.github.io/gpu-speedups/01_CuPy_and_Numba_on_the_GPU/index.html)
- [Apple Silicon unified memory, `newBufferWithBytesNoCopy`](https://lilting.ch/en/articles/wasm-metal-zero-copy-gpu-inference-apple-silicon)
- [Georg Hager — a case for the non-temporal store](https://blogs.fau.de/hager/archives/2103)
- [Intel Haswell ECM analysis — STREAM Triad NT-store speedups](https://arxiv.org/pdf/1511.03639)
- [Transparent Hugepage Support — Linux kernel](https://docs.kernel.org/admin-guide/mm/transhuge.html)
- [THP and TLB pressure, measured](https://www.abhik.ai/concepts/systems/transparent-huge-pages)
- [ECM model — a single core cannot saturate memory bandwidth](https://arxiv.org/pdf/1509.03118)
- [NumPy — memory management, `npy_alloc_cache`, NEP 49](https://numpy.org/doc/stable/reference/c-api/data_memory.html)
- [Linux — overcommit accounting](https://www.kernel.org/doc/html/v5.1/vm/overcommit-accounting.html)
- [mmap vs Zarr/HDF5 — out-of-core tradeoffs](https://pythonspeed.com/articles/mmap-vs-zarr-hdf5/)
