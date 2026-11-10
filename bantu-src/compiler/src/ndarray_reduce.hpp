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
template <class T>
inline auto extremeFast(const NdArray& A, const ReducePlan& p, bool wantMax) {
    static_assert(std::is_floating_point<T>::value,
                  "extremeFast is the float fast path; integers have no NaN to propagate");
    const T* const  data = static_cast<const T*>(A.buf->data);
    const size_t    n    = p.innerCount;
    return [data, n, wantMax](size_t base) -> double {
        const T* q = data + base;
        T best = q[0];
        if (std::isnan(best)) return (double)best;
        if (wantMax) {
            for (size_t i = 1; i < n; i++) {
                const T x = q[i];
                if (std::isnan(x)) return (double)x;
                if (x > best) best = x;
            }
        } else {
            for (size_t i = 1; i < n; i++) {
                const T x = q[i];
                if (std::isnan(x)) return (double)x;
                if (x < best) best = x;
            }
        }
        return (double)best;
    };
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
