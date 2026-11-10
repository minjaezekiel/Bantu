#pragma once
/**
 * numba — reductions, scans, sorting and indexing.
 *
 * Included by ndarray_native.cpp only. docs/numba-architecture.md §4.8-4.11.
 *
 * The load-bearing part of this file is the accumulator, not the plumbing.
 * A naive summation loop loses about n*eps of relative accuracy, which for
 * 10M values is ~2e-12 -- and people diff numba against NumPy, which does
 * pairwise. Accuracy here is a correctness property, not polish.
 */

#include "ndarray_ufunc.hpp"

#include <functional>
#include <numeric>
#include <type_traits>

namespace numba {

// ── accurate summation ───────────────────────────────────────────────────────
// Binary-tree (pairwise) accumulation, carried like a binary counter: after n
// elements, each set bit k of n holds the sum of a block of exactly 2^k values,
// so error grows as log n rather than n. Unlike the usual recursive
// formulation this works for a STRIDED walk in one pass with O(log n) state,
// which is what a general axis reduction needs.
struct Pairwise {
    static const int MAXLEVEL = 64;          // 2^64 elements; count is a size_t
    double partial[MAXLEVEL];
    size_t count = 0;

    void add(double v) {
        size_t c = count++;
        int k = 0;
        while (c & 1) { v += partial[k]; c >>= 1; k++; }
        partial[k] = v;
    }
    double total() const {
        double s = 0.0;
        size_t c = count;
        for (int k = 0; c; k++, c >>= 1) if (c & 1) s += partial[k];
        return s;
    }
};

// The contiguous fast path: 128-element blocks with eight independent
// accumulators (so the adds pipeline and the loop vectorizes), each block sum
// handed to the tree above. Keeps the vectorized loop AND the accurate
// structure, rather than trading one for the other.
template <class T>
inline double sumContigT(const T* NB_RESTRICT p, size_t n) {
    Pairwise acc;
    size_t i = 0;
    for (; i + 128 <= n; i += 128) {
        double s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;
        for (size_t j = 0; j < 128; j += 8) {
            s0 += (double)p[i+j+0]; s1 += (double)p[i+j+1];
            s2 += (double)p[i+j+2]; s3 += (double)p[i+j+3];
            s4 += (double)p[i+j+4]; s5 += (double)p[i+j+5];
            s6 += (double)p[i+j+6]; s7 += (double)p[i+j+7];
        }
        acc.add(((s0+s1)+(s2+s3)) + ((s4+s5)+(s6+s7)));
    }
    for (; i < n; i++) acc.add((double)p[i]);
    return acc.total();
}

// The f64 entry point, unchanged in behaviour: with T = double every cast above
// is a no-op, so this instantiation is the same code -- and the same summation
// ORDER -- as before D2. The accumulators stay `double` for every T, which is
// N24's storage/accumulator split: an f32 array sums through f64 and so keeps
// this library's ~2e-16 accuracy guarantee instead of f32's ~1e-3.
inline double sumContig(const double* NB_RESTRICT p, size_t n) {
    return sumContigT<double>(p, n);
}

// ── what dtype a reduction produces (D2) ─────────────────────────────────────
// Two rules, written once each, because getting either wrong for a new dtype is
// SILENT. An f32 sum landing in i64 truncates every value; an f32 mean landing
// in f64 promotes a whole f32 pipeline back to f64 one call in and quietly
// undoes the reason for using f32 at all.

// sum / prod: a float input keeps its own width; everything integer-ish becomes
// i64, because the point of carrying an integer path is that counts and indices
// stay exact.
inline DType reduceDType(DType in) {
    if (in == DType::F64 || in == DType::F32) return in;
    return DType::I64;
}

// mean / var / std / median / quantile: always a float, and an f32 input keeps
// f32 rather than widening. Matches NumPy.
inline DType floatResultDType(DType in) {
    return (in == DType::F32) ? DType::F32 : DType::F64;
}

// ── the axis plan (docs §4.8) ────────────────────────────────────────────────
// Splits the input's axes into the ones that survive (outer) and the ones being
// reduced (inner).
struct ReducePlan {
    std::vector<size_t>    outerShape;
    std::vector<ptrdiff_t> outerStrides;     // into the input
    std::vector<size_t>    innerShape;
    std::vector<ptrdiff_t> innerStrides;     // into the input
    std::vector<size_t>    resultShape;      // after keepdims is applied
    size_t innerCount = 1;
};

// axis: null -> every axis; a number; or a list of numbers. Negative counts
// from the end.
inline std::vector<size_t> axesFrom(const Value& v, size_t ndim, const char* what) {
    std::vector<size_t> ax;
    if (v.isNull()) { for (size_t d = 0; d < ndim; d++) ax.push_back(d); return ax; }
    auto one = [&](const Value& e) {
        if (!e.isNumber()) {
            throw std::runtime_error(std::string(what) + ": axis must be a number (got " +
                                     e.toString() + ")");
        }
        const double d = e.numberVal;
        if (std::isnan(d) || std::isinf(d) || d != std::floor(d)) {
            throw std::runtime_error(std::string(what) + ": axis must be a whole number (got " +
                                     e.toString() + ")");
        }
        ptrdiff_t a = (ptrdiff_t)d;
        if (a < 0) a += (ptrdiff_t)ndim;
        if (a < 0 || (size_t)a >= ndim) {
            throw std::runtime_error(std::string(what) + ": axis " + e.toString() +
                " is out of range for a " + std::to_string(ndim) + "-dimensional array");
        }
        ax.push_back((size_t)a);
    };
    if (v.isList()) for (const Value& e : v.listVal) one(e);
    else one(v);

    std::sort(ax.begin(), ax.end());
    for (size_t i = 1; i < ax.size(); i++) {
        if (ax[i] == ax[i - 1]) {
            throw std::runtime_error(std::string(what) + ": axis " + std::to_string(ax[i]) +
                                     " is given more than once");
        }
    }
    return ax;
}

inline ReducePlan planReduce(const NdArray& a, const std::vector<size_t>& axes,
                             bool keepdims) {
    ReducePlan p;
    std::vector<bool> reduced(a.ndim(), false);
    for (size_t d : axes) reduced[d] = true;

    for (size_t d = 0; d < a.ndim(); d++) {
        if (reduced[d]) {
            p.innerShape.push_back(a.shape[d]);
            p.innerStrides.push_back(a.strides[d]);
            if (keepdims) p.resultShape.push_back(1);
        } else {
            p.outerShape.push_back(a.shape[d]);
            p.outerStrides.push_back(a.strides[d]);
            p.resultShape.push_back(a.shape[d]);
        }
    }
    // Coalescing the inner axes is what lets the contiguous fast path apply at
    // all: reducing the last axis of a C-contiguous array, or reducing
    // everything, collapses to one stride-1 run.
    {
        std::vector<std::vector<ptrdiff_t>> st = { p.innerStrides };
        coalesce(p.innerShape, st);
        p.innerStrides = st.empty() ? std::vector<ptrdiff_t>() : st[0];
    }
    {
        std::vector<std::vector<ptrdiff_t>> st = { p.outerStrides };
        coalesce(p.outerShape, st);
        p.outerStrides = st.empty() ? std::vector<ptrdiff_t>() : st[0];
    }
    p.innerCount = 1;
    for (size_t d : p.innerShape) p.innerCount *= d;
    return p;
}

// True when the inner sweep is a single stride-1 run, so the raw-pointer path
// applies. Computed, never assumed -- the same argument as isCContig (N4).
inline bool innerIsContig(const ReducePlan& p) {
    return p.innerShape.size() == 1 && p.innerStrides.size() == 1 && p.innerStrides[0] == 1;
}

// Walk one inner slice starting at `base`, calling body(offset) per element.
//
// The contiguous case gets a straight run, and this is load-bearing rather
// than tidy. A reduction calls innerSweep ONCE PER OUTPUT, and `nditer` heap-
// allocates three std::vectors on every call -- so reducing the last axis of a
// [150528, 4] array meant 150,528 calls and ~450,000 allocations to do 600,000
// comparisons. That is where the time went: `nd_max` over that array took
// **102 ms**, which is ~680 ns per output for four comparisons.
//
// `nd_sum` already had a fast path of its own (sumContig), which is why sums
// looked fine while max, min, prod, any, all, argmax, argmin, ptp, var and std
// -- every reduction that funnels through here -- did not. Fixing the
// chokepoint fixes all of them at once, and the visit order is unchanged: a
// stride-1 run walked in order is the same sequence nditer produced.
// ── the contiguous extreme fast path, over the stored type (D1) ──────────────
// nd_max/nd_min's hot path was written against `const double*` directly. It is
// now a factory over T, so D2 adds f32 with ONE line at the call site:
//
//     if (A.dtype == DType::F32) ... extremeFast<float>(A, P, wantMax) ...
//
// The callback still returns `double` whatever T is -- that is N24's
// storage/accumulator split: an f32 array reduces through an f64 result, so f32
// does not inherit f32-precision reductions.
//
// NaN propagates by early return rather than by comparison, because every
// comparison against NaN is false and a plain `x > best` loop silently skips
// NaNs instead of returning one (which is amax semantics, btorch B16).
// The early return on NaN is correct and it was also the reason this loop ran
// at 5.5 GB/s while the summation loop next to it ran at 12.4: a `return` from
// inside the loop is a loop-carried control dependency, so nothing vectorizes
// and nothing pipelines. nd_max over a contiguous axis was SLOWER than nd_max
// over a strided one once the strided path was blocked, which is the kind of
// inversion that says the fast path is not fast.
//
// Split in two instead, with the semantics untouched:
//
//   * eight independent extremum accumulators over blocks of eight, so the
//     comparisons pipeline and the loop vectorises. A NaN can never enter an
//     accumulator -- every comparison against NaN is false -- so this is the
//     extremum of the non-NaN values, which is the answer whenever there is no
//     NaN. min/max are exact and order-independent, so splitting the
//     accumulators cannot change the result by even a bit.
//   * one branchless NaN flag, OR-ed across the same pass (`x != x`).
//
// If the flag is set, a scalar rescan finds the FIRST NaN and returns it --
// that is numba's documented first-NaN-wins rule, and it keeps the NaN's exact
// payload. The rescan costs a second pass, but only on arrays that contain a
// NaN at all, where correctness is the only thing that matters.
template <class T>
inline double extremeContig(const T* NB_RESTRICT q, size_t n, bool wantMax) {
    if (n == 0) return 0.0;                      // callers gate on this (nd_max)
    bool sawNan = false;
    T b[8];
    for (int k = 0; k < 8; k++) b[k] = q[0];
    size_t i = 0;
    if (n >= 8) {
        for (; i + 8 <= n; i += 8) {
            for (int k = 0; k < 8; k++) {
                const T x = q[i + k];
                sawNan |= (x != x);
                if (wantMax) { if (x > b[k]) b[k] = x; }
                else         { if (x < b[k]) b[k] = x; }
            }
        }
    }
    T best = b[0];
    for (int k = 1; k < 8; k++) {
        if (wantMax) { if (b[k] > best) best = b[k]; }
        else         { if (b[k] < best) best = b[k]; }
    }
    for (; i < n; i++) {
        const T x = q[i];
        sawNan |= (x != x);
        if (wantMax) { if (x > best) best = x; }
        else         { if (x < best) best = x; }
    }
    if (sawNan) {
        for (size_t j = 0; j < n; j++) if (q[j] != q[j]) return (double)q[j];
    }
    return (double)best;
}

template <class T>
inline auto extremeFast(const NdArray& A, const ReducePlan& p, bool wantMax) {
    static_assert(std::is_floating_point<T>::value,
                  "extremeFast is the float fast path; integers have no NaN to propagate");
    const T* const  data = static_cast<const T*>(A.buf->data);
    const size_t    n    = p.innerCount;
    return [data, n, wantMax](size_t base) -> double {
        return extremeContig<T>(data + base, n, wantMax);
    };
}

// ── reducing a STRIDED axis: the same answer, an order of magnitude faster ───
//
// `nd_sum($m, 0, null, null)` on a 2048x2048 array took 66 ms while axis 1 took
// 2.4 ms -- a 27x asymmetry for the same 32 MB of data and the same number of
// adds. The reason is the loop order: one output at a time, each walking a
// column with stride 2048, so every single load is its own cache line and the
// array is re-streamed once per output.
//
// The fix is to invert the loops -- stream the input in memory order, holding
// one accumulator per output in a tile -- and it is BIT-IDENTICAL, because each
// output still sees its elements in the same order, so `Pairwise` builds the
// same tree with the same roundings. Two further things make it fast rather
// than merely better:
//
//  * THE CARRY DEPTH IS SHARED. Every accumulator in a tile has been fed the
//    same number of elements, so the binary counter's carry depth at step i is
//    the same for all of them -- the number of trailing 1 bits of i. Hoisted
//    out of the per-output loop, it stops being an unpredictable branch per
//    element and the inner loop becomes a contiguous vector add.
//  * EIGHT ROWS AT A TIME. Pairwise's association for eight adds from an
//    aligned count is exactly ((r7+r6)+(r5+r4)) + ((r3+r2)+(r1+r0)), so eight
//    rows can be combined in registers and injected at level 3 instead of
//    touching the partials array eight times. Without this the level-0 writes
//    alone double the memory traffic.
//
// Measured on a 2048x2048 f64 array, axis 0: 32.7 -> 2.69 ms, **12.1x**, at
// 12.46 GB/s against the contiguous path's ~13 GB/s -- so the asymmetry is
// gone, not reduced. Zero bit differences, including for 2047 rows (a ragged
// group of eight) and 13 rows (no full group at all).
//
// The prototype that established both the speed and the bit-identity is kept
// as bantu-src/compiler/tests/pairwise_partition_test.cpp's sibling reasoning;
// the assertions live in tests/numba_reduce_axis_test.b.

inline constexpr size_t kOuterTile = 1024;      // 1024 outputs x ~12 levels = 96 KB

// Scratch for the partials, allocated once per thread and reused. A reduction
// must not heap-allocate per call (invariant 5): nditer doing exactly that cost
// ~450,000 allocations in one pooling layer.
inline std::vector<double>& reduceScratch() {
    static thread_local std::vector<double> v;
    return v;
}

// One run: W consecutive outputs (outer stride 1), each summing n elements
// spaced S apart, starting at `base`. Writes W doubles to `out`.
template <class T>
inline void sumOuterFused(const T* data, size_t base, size_t W, size_t S, size_t n,
                          double* out) {
    int levels = 1;
    while ((size_t(1) << levels) <= n) levels++;
    levels += 1;                                 // slack, so level `levels-1` is spare
    const size_t tile = (W < kOuterTile) ? W : kOuterTile;
    std::vector<double>& sc = reduceScratch();
    if (sc.size() < (size_t)levels * tile) sc.resize((size_t)levels * tile);
    double* const part = sc.data();

    for (size_t j0 = 0; j0 < W; j0 += tile) {
        const size_t w = (j0 + tile <= W) ? tile : W - j0;
        const T* const col = data + base + j0;
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            const T* r0 = col + (i + 0) * S;  const T* r1 = col + (i + 1) * S;
            const T* r2 = col + (i + 2) * S;  const T* r3 = col + (i + 3) * S;
            const T* r4 = col + (i + 4) * S;  const T* r5 = col + (i + 5) * S;
            const T* r6 = col + (i + 6) * S;  const T* r7 = col + (i + 7) * S;
            size_t c = i >> 3;                   // the level-3 count before this group
            int k = 3;
            while (c & 1) { c >>= 1; k++; }
            double* dst = part + (size_t)k * tile;
            if (k == 3) {
                for (size_t t = 0; t < w; t++) {
                    dst[t] = (((double)r7[t] + (double)r6[t]) +
                              ((double)r5[t] + (double)r4[t])) +
                             (((double)r3[t] + (double)r2[t]) +
                              ((double)r1[t] + (double)r0[t]));
                }
            } else {
                for (size_t t = 0; t < w; t++) {
                    double v = (((double)r7[t] + (double)r6[t]) +
                                ((double)r5[t] + (double)r4[t])) +
                               (((double)r3[t] + (double)r2[t]) +
                                ((double)r1[t] + (double)r0[t]));
                    for (int lev = 3; lev < k; lev++) v += part[(size_t)lev * tile + t];
                    dst[t] = v;
                }
            }
        }
        // The tail: fewer than eight elements left, added one at a time with
        // the carry depth still hoisted. Their counts continue from i, so the
        // tree they land in is the one the sequential walk would have built.
        for (; i < n; i++) {
            const T* row = col + i * S;
            size_t c = i;
            int k = 0;
            while (c & 1) { c >>= 1; k++; }
            double* dst = part + (size_t)k * tile;
            if (k == 0) {
                for (size_t t = 0; t < w; t++) dst[t] = (double)row[t];
            } else {
                for (size_t t = 0; t < w; t++) {
                    double v = (double)row[t];
                    for (int lev = 0; lev < k; lev++) v += part[(size_t)lev * tile + t];
                    dst[t] = v;
                }
            }
        }
        // Pairwise::total(): the set bits of n, low to high, in that order.
        for (size_t t = 0; t < w; t++) {
            double acc = 0.0;
            size_t c = n;
            for (int k = 0; c; k++, c >>= 1) if (c & 1) acc += part[(size_t)k * tile + t];
            out[j0 + t] = acc;
        }
    }
}

// ── the same loop inversion for the SIMPLE accumulators ──────────────────────
// min, max, prod, any and all keep one double per output and update it per
// element, so inverting the loops needs none of the sum path's level trick --
// just a tile of accumulators and the input streamed in memory order. Each
// output still sees its elements in the same order, so an order-dependent
// accumulator (prod's running product, min/max's first-NaN-wins rule) produces
// exactly the same double it does today.
//
// Measured, 2048x2048 f64 axis 0: nd_max 1350 -> 103 ms, nd_min 1572 -> 104 ms.
inline std::vector<double>& reduceScratch2() {
    static thread_local std::vector<double> v;
    return v;
}

// upd(acc, x) folds one element in. `initFromFirst` makes the first element of
// each output the starting value (min/max), otherwise `initVal` is
// (prod: 1, any: 0, all: 1).
template <class T, class Upd>
inline void reduceOuterSimple(const T* data, size_t base, size_t W, size_t S, size_t n,
                              double* out, bool initFromFirst, double initVal, Upd upd) {
    const size_t tile = (W < kOuterTile) ? W : kOuterTile;
    std::vector<double>& sc = reduceScratch2();
    if (sc.size() < tile) sc.resize(tile);
    double* const acc = sc.data();

    for (size_t j0 = 0; j0 < W; j0 += tile) {
        const size_t w = (j0 + tile <= W) ? tile : W - j0;
        const T* const col = data + base + j0;
        size_t i = 0;
        if (initFromFirst) {
            for (size_t t = 0; t < w; t++) acc[t] = (double)col[t];
            i = 1;
        } else {
            for (size_t t = 0; t < w; t++) acc[t] = initVal;
        }
        for (; i < n; i++) {
            const T* row = col + i * S;
            for (size_t t = 0; t < w; t++) upd(acc[t], (double)row[t]);
        }
        for (size_t t = 0; t < w; t++) out[j0 + t] = acc[t];
    }
}

template <class Upd>
inline ArrayPtr runReduceOuterSimple(const NdArray& a, const ReducePlan& p, DType rdt,
                                     const char* what, bool initFromFirst, double initVal,
                                     Upd upd) {
    ArrayPtr out = makeArray(p.resultShape, rdt, what);
    if (out->size() == 0) return out;

    const size_t last = p.outerShape.size() - 1;
    const size_t W    = p.outerShape[last];
    const size_t S    = (size_t)p.innerStrides[0];
    const size_t n    = p.innerCount;

    std::vector<double> tmp(W);
    size_t k = 0;
    auto run = [&](size_t base) {
        switch (a.dtype) {
            case DType::F64:
                reduceOuterSimple<double>(static_cast<const double*>(a.buf->data),
                                          base, W, S, n, tmp.data(), initFromFirst, initVal, upd);
                break;
            case DType::F32:
                reduceOuterSimple<float>(static_cast<const float*>(a.buf->data),
                                         base, W, S, n, tmp.data(), initFromFirst, initVal, upd);
                break;
            case DType::I64:
                reduceOuterSimple<int64_t>(static_cast<const int64_t*>(a.buf->data),
                                           base, W, S, n, tmp.data(), initFromFirst, initVal, upd);
                break;
            default:
                reduceOuterSimple<uint8_t>(static_cast<const uint8_t*>(a.buf->data),
                                           base, W, S, n, tmp.data(), initFromFirst, initVal, upd);
                break;
        }
        for (size_t t = 0; t < W; t++) setFromDouble(*out, k + t, tmp[t]);
        k += W;
    };

    if (p.outerShape.size() == 1) {
        run(a.offset);
    } else {
        std::vector<size_t>    oshape(p.outerShape.begin(), p.outerShape.end() - 1);
        std::vector<ptrdiff_t> ostride(p.outerStrides.begin(), p.outerStrides.end() - 1);
        nditer(oshape, { ostride }, { a.offset }, [&](const size_t* o) { run(o[0]); });
    }
    return out;
}

// The simple accumulators work for every dtype, so this is the sum path's
// predicate without its float-only restriction.
inline bool outerSimpleApplies(const NdArray& a, const ReducePlan& p) {
    if (innerIsContig(p)) return false;
    if (p.innerShape.size() != 1 || p.innerStrides.size() != 1) return false;
    if (p.innerStrides[0] <= 0) return false;
    if (p.outerShape.empty()) return false;
    const size_t last = p.outerShape.size() - 1;
    if (p.outerStrides[last] != 1) return false;
    if (p.outerShape[last] < 8 || p.innerCount < 2) return false;
    if (a.buf == nullptr || a.buf->data == nullptr) return false;
    return true;
}

// Is this reduction one the fused path can take? Computed, never assumed --
// the same discipline as isCContig (N4).
//
//   * the reduced sweep is a single coalesced axis (so there is one stride S),
//     and it is NOT contiguous -- if it were, the existing block loop is
//     already at bandwidth;
//   * the innermost REMAINING axis has stride 1, which is what makes the
//     per-output accumulators consecutive and the loads contiguous;
//   * there is enough of both to pay for the tiling.
inline bool outerFusedApplies(const NdArray& a, const ReducePlan& p) {
    if (a.dtype != DType::F64 && a.dtype != DType::F32) return false;
    if (innerIsContig(p)) return false;
    if (p.innerShape.size() != 1 || p.innerStrides.size() != 1) return false;
    if (p.innerStrides[0] <= 0) return false;         // reversed views keep the old path
    if (p.outerShape.empty()) return false;
    const size_t last = p.outerShape.size() - 1;
    if (p.outerStrides[last] != 1) return false;
    if (p.outerShape[last] < 8 || p.innerCount < 4) return false;
    return true;
}

// Drive it: one fused run per position of the outer axes except the innermost,
// whose W outputs each run writes consecutively. The result is freshly
// allocated and C-contiguous, so the odometer order is the output order --
// the same assumption runReduce already documents.
inline ArrayPtr runReduceSumFused(const NdArray& a, const ReducePlan& p, DType rdt,
                                  const char* what) {
    ArrayPtr out = makeArray(p.resultShape, rdt, what);
    if (out->size() == 0) return out;

    const size_t last = p.outerShape.size() - 1;
    const size_t W    = p.outerShape[last];
    const size_t S    = (size_t)p.innerStrides[0];
    const size_t n    = p.innerCount;

    // f64 results are written straight into the destination; an f32 result is
    // narrowed afterwards, because the accumulator is f64 whatever the storage
    // is (N24).
    std::vector<double> tmp;
    double* dst = nullptr;
    if (rdt == DType::F64) dst = static_cast<double*>(out->buf->data) + out->offset;
    else { tmp.resize(W); }

    size_t k = 0;
    auto run = [&](size_t base) {
        double* target = (rdt == DType::F64) ? (dst + k) : tmp.data();
        if (a.dtype == DType::F64)
            sumOuterFused<double>(static_cast<const double*>(a.buf->data), base, W, S, n, target);
        else
            sumOuterFused<float>(static_cast<const float*>(a.buf->data), base, W, S, n, target);
        if (rdt != DType::F64) {
            for (size_t t = 0; t < W; t++) setFromDouble(*out, k + t, tmp[t]);
        }
        k += W;
    };

    if (p.outerShape.size() == 1) {
        run(a.offset);
    } else {
        std::vector<size_t>    oshape(p.outerShape.begin(), p.outerShape.end() - 1);
        std::vector<ptrdiff_t> ostride(p.outerStrides.begin(), p.outerStrides.end() - 1);
        nditer(oshape, { ostride }, { a.offset }, [&](const size_t* o) { run(o[0]); });
    }
    return out;
}

template <class Body>
inline void innerSweep(const ReducePlan& p, size_t base, Body&& body) {
    if (innerIsContig(p)) {
        const size_t n = p.innerCount;
        for (size_t i = 0; i < n; i++) body(base + i);
        return;
    }
    nditer(p.innerShape, { p.innerStrides }, { base },
           [&](const size_t* o) { body(o[0]); });
}

// ── the generic reduction driver ─────────────────────────────────────────────
// `reduceSlice(base, count)` produces one output value from one inner slice.
template <class Fn>
inline ArrayPtr runReduce(const NdArray& a, const ReducePlan& p, DType rdt,
                          const char* what, Fn reduceSlice) {
    ArrayPtr out = makeArray(p.resultShape, rdt, what);
    if (out->size() == 0) return out;

    // The result is freshly allocated and C-contiguous, so its elements are
    // visited in the same order the outer odometer produces them.
    size_t k = 0;
    if (p.outerShape.empty()) {
        setFromDouble(*out, 0, reduceSlice(a.offset));
    } else {
        nditer(p.outerShape, { p.outerStrides }, { a.offset },
               [&](const size_t* o) { setFromDouble(*out, k++, reduceSlice(o[0])); });
    }
    return out;
}

}   // namespace numba
