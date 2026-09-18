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
inline double sumContig(const double* NB_RESTRICT p, size_t n) {
    Pairwise acc;
    size_t i = 0;
    for (; i + 128 <= n; i += 128) {
        double s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;
        for (size_t j = 0; j < 128; j += 8) {
            s0 += p[i+j+0]; s1 += p[i+j+1]; s2 += p[i+j+2]; s3 += p[i+j+3];
            s4 += p[i+j+4]; s5 += p[i+j+5]; s6 += p[i+j+6]; s7 += p[i+j+7];
        }
        acc.add(((s0+s1)+(s2+s3)) + ((s4+s5)+(s6+s7)));
    }
    for (; i < n; i++) acc.add(p[i]);
    return acc.total();
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

// Walk one inner slice starting at `base`, calling body(offset) per element.
template <class Body>
inline void innerSweep(const ReducePlan& p, size_t base, Body&& body) {
    nditer(p.innerShape, { p.innerStrides }, { base },
           [&](const size_t* o) { body(o[0]); });
}

// True when the inner sweep is a single stride-1 run, so the raw-pointer path
// applies. Computed, never assumed -- the same argument as isCContig (N4).
inline bool innerIsContig(const ReducePlan& p) {
    return p.innerShape.size() == 1 && p.innerStrides.size() == 1 && p.innerStrides[0] == 1;
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
