#pragma once
/**
 * numba — broadcasting and the element-wise kernel loop.
 *
 * Included by ndarray_native.cpp only, which every build script compiles at
 * -O3 -ftree-vectorize while the rest of the interpreter stays at -O2 (N11).
 * That override exists for the loops in THIS file: GCC 11 does not enable
 * -ftree-loop-vectorize at -O2 and the production Linux binary is built in
 * ubuntu:22.04, so without it the tier-0 loops would be scalar in the binary
 * users download while vectorizing on the Mac they were benchmarked on.
 *
 * Design: docs/numba-architecture.md §4. The short version:
 *
 *   `c = a + b` on 10M f64 moves ~320 MB for 10M flops -- 0.042 flops/byte,
 *   two orders of magnitude below any machine's balance point. These loops are
 *   DRAM-bandwidth-bound, not ALU-bound, so the work that pays is reducing
 *   memory traffic and keeping the compiler able to vectorize: one flat
 *   __restrict loop for the common case, and dimension coalescing so the
 *   strided case collapses back into one.
 */

#include "ndarray_native.hpp"
#include "parallel.hpp"

namespace numba {

// ── broadcasting ─────────────────────────────────────────────────────────────
// NumPy semantics: right-align the shapes; each pair must be equal or one of
// them 1. A stretched axis gets stride 0 later, so stepping along it re-reads
// the same element -- that zero stride IS broadcasting, and nothing is copied.
inline std::vector<size_t> broadcastShapes(const std::vector<size_t>& a,
                                           const std::vector<size_t>& b,
                                           const char* what) {
    const size_t n = std::max(a.size(), b.size());
    std::vector<size_t> out(n);
    for (size_t i = 0; i < n; i++) {
        // Walk from the right; a missing axis on either side behaves as 1.
        const size_t ad = (i < n - a.size()) ? 1 : a[i - (n - a.size())];
        const size_t bd = (i < n - b.size()) ? 1 : b[i - (n - b.size())];
        if (ad == bd)      out[i] = ad;
        else if (ad == 1)  out[i] = bd;
        else if (bd == 1)  out[i] = ad;
        else {
            // The most common error in array code. A message that says only
            // "shape mismatch" is a permanent support burden, so name the axis
            // and both extents.
            throw std::runtime_error(std::string(what) + ": shapes " + shapeStr(a) + " and " +
                shapeStr(b) + " cannot be broadcast together (axis " + std::to_string(i) +
                ": " + std::to_string(ad) + " vs " + std::to_string(bd) + ")");
        }
    }
    return out;
}

// Strides for `a` re-expressed against `to`, with 0 on every stretched or
// padded axis. Not a view: just the walk order an operand contributes.
inline std::vector<ptrdiff_t> stridesFor(const NdArray& a, const std::vector<size_t>& to) {
    const size_t n = to.size(), pad = n - a.ndim();
    std::vector<ptrdiff_t> st(n, 0);
    for (size_t i = 0; i < a.ndim(); i++) {
        st[pad + i] = (a.shape[i] == 1 && to[pad + i] != 1) ? 0 : a.strides[i];
    }
    return st;
}

// ── dimension coalescing ─────────────────────────────────────────────────────
// The part that gets skipped, and it matters more than SIMD. Merge adjacent
// axes when they are contiguous with each other IN EVERY OPERAND at once, and
// drop extent-1 axes. A (1000,1000) op that reached tier 2 collapses back into
// one 1e6-iteration flat loop; without it an n-d op pays an odometer step per
// row and loses 2-3x. NumPy's nditer does exactly this.
//
// `strides` is one stride vector per operand, all aligned to `shape`.
inline void coalesce(std::vector<size_t>& shape,
                     std::vector<std::vector<ptrdiff_t>>& strides) {
    const size_t nops = strides.size();
    if (shape.empty()) return;

    // Drop extent-1 axes first: they contribute one iteration and no movement,
    // and removing them exposes merges that were previously non-adjacent.
    {
        std::vector<size_t> s2;
        std::vector<std::vector<ptrdiff_t>> t2(nops);
        for (size_t d = 0; d < shape.size(); d++) {
            if (shape[d] == 1) continue;
            s2.push_back(shape[d]);
            for (size_t k = 0; k < nops; k++) t2[k].push_back(strides[k][d]);
        }
        shape = std::move(s2);
        strides = std::move(t2);
    }
    if (shape.size() < 2) return;

    std::vector<size_t> s2;
    std::vector<std::vector<ptrdiff_t>> t2(nops);
    s2.push_back(shape[0]);
    for (size_t k = 0; k < nops; k++) t2[k].push_back(strides[k][0]);

    for (size_t d = 1; d < shape.size(); d++) {
        const size_t last = s2.size() - 1;
        bool mergeable = true;
        for (size_t k = 0; k < nops && mergeable; k++) {
            // Axis d-1 is contiguous with axis d for this operand exactly when
            // one step along d-1 equals a full sweep of d.
            if (t2[k][last] != strides[k][d] * (ptrdiff_t)shape[d]) mergeable = false;
        }
        if (mergeable) {
            s2[last] *= shape[d];
            for (size_t k = 0; k < nops; k++) t2[k][last] = strides[k][d];
        } else {
            s2.push_back(shape[d]);
            for (size_t k = 0; k < nops; k++) t2[k].push_back(strides[k][d]);
        }
    }
    shape = std::move(s2);
    strides = std::move(t2);
}

// ── typed element access ─────────────────────────────────────────────────────
// Reading an i64 through double would round above 2^53, so the integer path is
// genuinely separate rather than a convenience (docs §4.6).
inline int64_t loadI(const NdArray& a, size_t off) {
    switch (a.dtype) {
        case DType::F64:  return (int64_t)std::llround(static_cast<const double*>(a.buf->data)[off]);
        case DType::F32:  return (int64_t)std::llround((double)static_cast<const float*>(a.buf->data)[off]);
        case DType::I64:  return static_cast<const int64_t*>(a.buf->data)[off];
        case DType::BOOL: return static_cast<const uint8_t*>(a.buf->data)[off] ? 1 : 0;
    }
    return 0;
}
inline double loadD(const NdArray& a, size_t off) { return getAsDouble(a, off); }

inline void storeD(NdArray& a, size_t off, double v) { setFromDouble(a, off, v); }
inline void storeI(NdArray& a, size_t off, int64_t v) {
    switch (a.dtype) {
        case DType::F64:  static_cast<double*>(a.buf->data)[off]  = (double)v; break;
        case DType::F32:  static_cast<float*>(a.buf->data)[off]   = narrowToFloat((double)v); break;
        case DType::I64:  static_cast<int64_t*>(a.buf->data)[off] = v;         break;
        case DType::BOOL: static_cast<uint8_t*>(a.buf->data)[off] = v ? 1 : 0; break;
    }
}

// ── the same two accessors, reached by compute type instead of by name (D1) ──
// `load<T>` reads one element of ANY dtype and converts it to T; `store<T>`
// writes a T into any dtype. They are the generic-path counterparts of the
// NB_DISPATCH_COMPUTE macro: a kernel body written over `T` calls load<T>/
// store<T> and works for every dtype without a second code path.
//
// These are thin wrappers over loadD/loadI/storeD/storeI rather than new logic,
// so the generic path's conversion rules -- llround on the way into i64, `!= 0`
// on the way into bool -- are unchanged and remain in exactly one place.
template <class T> inline T    load(const NdArray& a, size_t off);
template <class T> inline void store(NdArray& a, size_t off, T v);

template <> inline double  load<double>(const NdArray& a, size_t off)  { return loadD(a, off); }
template <> inline float   load<float>(const NdArray& a, size_t off)   { return narrowToFloat(loadD(a, off)); }
template <> inline int64_t load<int64_t>(const NdArray& a, size_t off) { return loadI(a, off); }
template <> inline uint8_t load<uint8_t>(const NdArray& a, size_t off) { return loadI(a, off) ? 1 : 0; }

template <> inline void store<double>(NdArray& a, size_t off, double v)   { storeD(a, off, v); }
template <> inline void store<float>(NdArray& a, size_t off, float v)    { storeD(a, off, (double)v); }
template <> inline void store<int64_t>(NdArray& a, size_t off, int64_t v) { storeI(a, off, v); }
template <> inline void store<uint8_t>(NdArray& a, size_t off, uint8_t v) { storeI(a, off, v ? 1 : 0); }

// Raw typed access for the CONTIGUOUS fast paths, where the dtype is already
// known to match and the strided accessors above would be pure overhead.
template <class T> inline const T* rawPtr(const NdArray& a) { return a.ptr<T>(); }
template <class T> inline       T* rawPtr(NdArray& a)       { return a.ptr<T>(); }

// Is this array stored as T, and flat enough for a raw pointer walk? The
// precondition every `rawPtr` use needs, in one place.
template <class T> inline bool isRaw(const NdArray& a) {
    return a.dtype == dtypeTraits<T>::tag && a.isCContig();
}

// ── the odometer ─────────────────────────────────────────────────────────────
// Tier 2. `body` receives one buffer offset per operand. The counters live
// outside the innermost loop so the hot axis is a plain stride walk.
template <class Body>
inline void nditer(const std::vector<size_t>& shape,
                   const std::vector<std::vector<ptrdiff_t>>& strides,
                   const std::vector<size_t>& offsets,
                   Body&& body) {
    const size_t nd = shape.size(), nops = offsets.size();
    if (nd == 0) { body(offsets.data()); return; }

    size_t total = 1;
    for (size_t d = 0; d < nd; d++) total *= shape[d];
    if (total == 0) return;

    std::vector<size_t>    idx(nd, 0);
    std::vector<ptrdiff_t> cur(nops);
    for (size_t k = 0; k < nops; k++) cur[k] = (ptrdiff_t)offsets[k];

    const size_t inner = shape[nd - 1];
    std::vector<size_t> pos(nops);
    while (true) {
        // Innermost axis: a straight stride walk, which is what the compiler
        // can turn into a vector loop when the strides happen to be 1.
        for (size_t k = 0; k < nops; k++) pos[k] = (size_t)cur[k];
        for (size_t i = 0; i < inner; i++) {
            body(pos.data());
            for (size_t k = 0; k < nops; k++) pos[k] = (size_t)((ptrdiff_t)pos[k] + strides[k][nd - 1]);
        }
        if (nd == 1) return;
        // Carry.
        size_t d = nd - 1;
        while (d-- > 0) {
            idx[d]++;
            for (size_t k = 0; k < nops; k++) cur[k] += strides[k][d];
            if (idx[d] < shape[d]) break;
            for (size_t k = 0; k < nops; k++) cur[k] -= strides[k][d] * (ptrdiff_t)shape[d];
            idx[d] = 0;
            if (d == 0) return;
        }
    }
}

// ── tier 0: the loop that must vectorize ─────────────────────────────────────
// __restrict is what lets the compiler assume the output does not alias the
// inputs. The aliasing gate in applyBinary is what makes that promise true --
// if an overlapping `out=` reached here, this would be silent corruption rather
// than a slow path, which is why the gate is not optional.
#if defined(__GNUC__) || defined(__clang__)
    #define NB_RESTRICT __restrict__
#else
    #define NB_RESTRICT
#endif

// ── threading the element-wise loops (D3b) ───────────────────────────────────
// Element i of a tier-0/1 loop reads only input element i and writes only
// output element i, so chunking by index satisfies the determinism contract
// without any care at all: the boundaries come from n alone, the writes are
// disjoint, and there is no combine step to order. Unlike the reductions
// (D3c), nothing about the ANSWER changes here -- which is why this is the
// increment that needed measurement rather than argument.
//
// The chunk size is fixed, not derived from the thread count, because a
// boundary that moved with the thread count is exactly what the contract
// forbids. kElemMin is the point below which a region costs more than it
// saves; both constants are measured, see docs/numba-dtype-architecture.md.
// WHICH ops get threads is a measured property of the op, not of the array.
// An element-wise loop either saturates the memory bus or saturates the ALU,
// and only the second kind has anything left for a second core:
//
//   1 thread -> 6 threads, f64, this machine (10 cores, measured 2026-11-10)
//     exp    n=16384  1.78x   n=131072  3.30x   n=262144  3.51x   n=1M  2.98x
//     tanh                    n=131072  2.90x                     n=1M  5.00x
//     power                                     n=262144  4.55x
//     sqrt   n=16384  0.64x   n=131072  1.74x   n=262144  1.94x   n=1M  1.29x
//     divide n=16384  0.57x   n=131072  1.22x   n=262144  1.50x   n=1M  1.00x
//     abs                     n=131072  0.92x   n=262144  1.10x   n=1M  1.08x
//     add    n=1M 0.83x   n=4M 0.91x   n=16M 0.88x
//
// So the transcendentals are threaded and nothing else is. `add` is SLOWER on
// six threads at every size large enough to matter -- one core already
// saturates the bus, and the region handshake is pure cost. sqrt and divide
// win, but only inside a band (131k-262k) whose edges are this machine's cache
// sizes; a rule that only holds on one machine is not a rule, and an op that
// got slower when a user turned threading on would be worse than an op that
// stayed the same speed.
//
// Cost is declared at the registration site, next to the functor, because that
// is the only place the answer is knowable. The default is CHEAP: a new ufunc
// is not threaded until someone measures it.
enum class Cost { CHEAP, HEAVY };

// Fixed, because a boundary that moved with the thread count is exactly what
// the determinism contract forbids. 4096 doubles is 32 KB, so a chunk is
// L1-resident; the minimum is where a region first pays for itself (1.78x).
inline constexpr size_t kElemChunk = 4096;
inline constexpr size_t kElemMin   = 16384;

template <class Body>
inline void elemwiseFor(size_t n, Cost cost, Body&& body) {   // body(lo, hi)
    if (cost == Cost::CHEAP || n < kElemMin ||
        getThreads() <= 1 || inRegionFlag()) {
        body(size_t(0), n);                             // the overwhelming case
        return;
    }
    const size_t chunks = chunkCount(n, kElemChunk);
    if (chunks <= 1) { body(size_t(0), n); return; }
    parallelFor(chunks, [&](size_t c) {
        const size_t lo = c * kElemChunk;
        body(lo, std::min(n, lo + kElemChunk));
    });
}

// Each loop re-asserts NB_RESTRICT on locals inside the chunk body. Passing the
// pointers through a lambda capture would otherwise drop the no-alias promise
// the signature makes, and with it the vectorisation that is the whole point of
// tier 0 -- a 30% loss on the SEQUENTIAL path, which is most calls.
template <class T, class Op>
inline void tier0(const T* NB_RESTRICT a, const T* NB_RESTRICT b,
                  T* NB_RESTRICT o, size_t n, Cost cost, Op op) {
    elemwiseFor(n, cost, [=](size_t lo, size_t hi) {
        const T* NB_RESTRICT ap = a;
        const T* NB_RESTRICT bp = b;
        T* NB_RESTRICT        op2 = o;
        for (size_t i = lo; i < hi; i++) op2[i] = op(ap[i], bp[i]);
    });
}

template <class T, class R, class Op>
inline void tier0cmp(const T* NB_RESTRICT a, const T* NB_RESTRICT b,
                     R* NB_RESTRICT o, size_t n, Cost cost, Op op) {
    elemwiseFor(n, cost, [=](size_t lo, size_t hi) {
        const T* NB_RESTRICT ap = a;
        const T* NB_RESTRICT bp = b;
        R* NB_RESTRICT        op2 = o;
        for (size_t i = lo; i < hi; i++) op2[i] = op(ap[i], bp[i]) ? 1 : 0;
    });
}

// Tier 1: one side is a single value, splatted. Worth its own loop because the
// scalar stays in a register instead of being re-read through a stride-0 walk.
template <class T, class Op>
inline void tier1(const T* NB_RESTRICT a, T s, T* NB_RESTRICT o, size_t n,
                  Cost cost, Op op) {
    elemwiseFor(n, cost, [=](size_t lo, size_t hi) {
        const T* NB_RESTRICT ap = a;
        T* NB_RESTRICT        op2 = o;
        for (size_t i = lo; i < hi; i++) op2[i] = op(ap[i], s);
    });
}
template <class T, class Op>
inline void tier1r(T s, const T* NB_RESTRICT b, T* NB_RESTRICT o, size_t n,
                   Cost cost, Op op) {
    elemwiseFor(n, cost, [=](size_t lo, size_t hi) {
        const T* NB_RESTRICT bp = b;
        T* NB_RESTRICT        op2 = o;
        for (size_t i = lo; i < hi; i++) op2[i] = op(s, bp[i]);
    });
}

template <class T, class Op>
inline void tier0u(const T* NB_RESTRICT a, T* NB_RESTRICT o, size_t n,
                   Cost cost, Op op) {
    elemwiseFor(n, cost, [=](size_t lo, size_t hi) {
        const T* NB_RESTRICT ap = a;
        T* NB_RESTRICT        op2 = o;
        for (size_t i = lo; i < hi; i++) op2[i] = op(ap[i]);
    });
}

}   // namespace numba
