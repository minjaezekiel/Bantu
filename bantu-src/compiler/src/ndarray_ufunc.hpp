#pragma once
/**
 * numba — the ufunc layer: promotion, the out= gates, and tier selection.
 *
 * Included by ndarray_native.cpp only. docs/numba-architecture.md §4.6-4.7.
 */

#include "ndarray_kernels.hpp"

namespace numba {

// What an operation computes in, independent of what its operands are.
enum class Kind : uint8_t {
    ARITH,      // + - * // %  -- promote, and i64 stays EXACT
    REAL,       // / pow and every transcendental -- always f64
    COMPARE,    // == != < <= > >=  -- promote to compare, emit bool
    LOGIC,      // and/or/xor/not -- truthiness in, bool out
    PREDICATE   // isnan/isinf/isfinite -- ALWAYS f64 in, bool out
};

inline DType computeDType(Kind k, DType a, DType b) {
    switch (k) {
        case Kind::REAL:      return DType::F64;
        // A predicate asks a question only a float can answer, so it computes
        // in f64 whatever it was handed. Letting it promote to i64 sent it down
        // the integer path, where these ops have no meaningful kernel at all:
        // nd_isfinite(nd([1], "i64")) returned FALSE.
        case Kind::PREDICATE: return DType::F64;
        // Truthiness is `!= 0`, which has to be evaluated in the operand's own
        // type. Computing logic in BOOL meant f64 operands were converted by
        // ROUNDING, so 0.4 was false and 0.6 was true -- llround, not truth.
        case Kind::LOGIC:     return promote(a, b);
        case Kind::COMPARE: {
            const DType p = promote(a, b);
            return p == DType::BOOL ? DType::I64 : p;   // compare bools as 0/1
        }
        case Kind::ARITH: {
            const DType p = promote(a, b);
            // bool + bool is i64, not bool: `true + true` is 2, and silently
            // saturating it to `true` would be a wrong answer, not a choice.
            return p == DType::BOOL ? DType::I64 : p;
        }
    }
    return DType::F64;
}

inline DType resultDType(Kind k, DType compute) {
    return (k == Kind::COMPARE || k == Kind::LOGIC || k == Kind::PREDICATE)
           ? DType::BOOL : compute;
}

// ── the out= gates (docs §4.7) ───────────────────────────────────────────────
// Three checks, in order, before a single element is written.
struct OutPlan {
    ArrayPtr dest;        // where the kernel actually writes
    ArrayPtr finalOut;    // null unless dest is a temporary to be copied back
};

// Is `out` element-wise identical to `in` -- same buffer, offset, strides and
// shape? Then `nd_add($a, $b, $a)` reads and writes the same index and is safe
// in place. Every OTHER overlap takes a temporary. Being conservative in the
// other direction would make the single idiom out= exists for the one idiom
// that allocates.
inline bool exactlyAliases(const NdArray& out, const NdArray& in) {
    return out.buf.get() == in.buf.get() && out.offset == in.offset &&
           out.dtype == in.dtype && out.shape == in.shape && out.strides == in.strides;
}

inline OutPlan planOut(const Value* outArg,
                       const std::vector<size_t>& shape, DType rdt,
                       const std::vector<const NdArray*>& inputs,
                       const char* what) {
    OutPlan p;
    if (!outArg || outArg->isNull()) {
        p.dest = makeArray(shape, rdt, what);
        return p;
    }
    ArrayPtr out = asArray(*outArg);

    // (1) A broadcast_to result has stride 0 on its stretched axes, so writing
    // through it would hit one element many times. This is the gate that makes
    // the read-only flag worth carrying (N19).
    requireWritable(*out, what);

    // (2) out is NOT itself broadcast: a destination smaller than the result
    // would silently discard elements and a larger one would leave stale ones.
    if (out->shape != shape) {
        throw std::runtime_error(std::string(what) + ": out has shape " + shapeStr(out->shape) +
            " but the result is " + shapeStr(shape) + " (out is not broadcast to fit)");
    }

    // (3) Aliasing. Overlap that is not exact means an element could be read
    // after it has already been overwritten -- a wrong answer, not a crash,
    // which is the worse failure. Compute into a temporary and copy back.
    bool needsTemp = false;
    for (const NdArray* in : inputs) {
        if (!in) continue;
        if (sharesMemory(*out, *in) && !exactlyAliases(*out, *in)) { needsTemp = true; break; }
    }
    if (needsTemp) {
        p.dest     = makeArray(shape, rdt, what);
        p.finalOut = out;
    } else {
        p.dest = out;
    }
    return p;
}

// Copy `src` into `dst` element-wise, converting dtype. Used to land a
// temporary back into the caller's out=, and by astype/copy.
inline void copyInto(const NdArray& src, NdArray& dst, const char* what) {
    std::vector<size_t> shape = dst.shape;
    std::vector<std::vector<ptrdiff_t>> st = { stridesFor(src, shape), dst.strides };
    coalesce(shape, st);
    const std::vector<size_t> offs = { src.offset, dst.offset };
    const bool intPath = (src.dtype != DType::F64 && dst.dtype != DType::F64);
    const NdArray& s = src; NdArray& d = dst;
    nditer(shape, st, offs, [&](const size_t* o) {
        if (intPath) storeI(d, o[1], loadI(s, o[0]));
        else         storeD(d, o[1], loadD(s, o[0]));
    });
    (void)what;
}

// ── binary application ───────────────────────────────────────────────────────
// FD/FI are the f64 and i64 kernels; only one runs, chosen by the compute dtype.
template <class FD, class FI>
inline Value applyBinary(const char* what, Kind kind,
                         const Value& av, const Value& bv, const Value* outArg,
                         FD fd, FI fi) {
    ArrayPtr a = asArray(av), b = asArray(bv);
    const std::vector<size_t> shape = broadcastShapes(a->shape, b->shape, what);
    const DType cdt = computeDType(kind, a->dtype, b->dtype);
    const DType rdt = resultDType(kind, cdt);

    OutPlan plan = planOut(outArg, shape, rdt, {a.get(), b.get()}, what);
    ArrayPtr out = plan.dest;

    const size_t n = out->size();
    if (n == 0) { if (plan.finalOut) copyInto(*out, *plan.finalOut, what);
                  return wrap(plan.finalOut ? plan.finalOut : out); }

    // ── tier 0 ──────────────────────────────────────────────────────────────
    // Same shape, already the compute dtype, all C-contiguous. Over 90% of real
    // calls, and the only loop the compiler must vectorize.
    const bool flat = a->shape == shape && b->shape == shape &&
                      a->isCContig() && b->isCContig() && out->isCContig() &&
                      a->dtype == cdt && b->dtype == cdt && out->dtype == rdt;
    if (flat) {
        if (cdt == DType::F64) {
            const double* ap = a->ptr<double>();
            const double* bp = b->ptr<double>();
            if (rdt == DType::BOOL)
                tier0cmp(ap, bp, out->ptr<uint8_t>(), n,
                         [&](double x, double y) { return fd(x, y) != 0.0; });
            else tier0(ap, bp, out->ptr<double>(), n, fd);
        } else if (cdt == DType::I64) {
            const int64_t* ap = a->ptr<int64_t>();
            const int64_t* bp = b->ptr<int64_t>();
            if (rdt == DType::BOOL)
                tier0cmp(ap, bp, out->ptr<uint8_t>(), n,
                         [&](int64_t x, int64_t y) { return fi(x, y) != 0; });
            else tier0(ap, bp, out->ptr<int64_t>(), n, fi);
        } else {   // BOOL compute: logic ops only
            const uint8_t* ap = a->ptr<uint8_t>();
            const uint8_t* bp = b->ptr<uint8_t>();
            tier0cmp(ap, bp, out->ptr<uint8_t>(), n,
                     [&](uint8_t x, uint8_t y) { return fi(x ? 1 : 0, y ? 1 : 0) != 0; });
        }
        if (plan.finalOut) copyInto(*out, *plan.finalOut, what);
        return wrap(plan.finalOut ? plan.finalOut : out);
    }

    // ── tier 2 ──────────────────────────────────────────────────────────────
    // General strided, after coalescing collapses whatever it can back into a
    // flat walk.
    std::vector<size_t> sh = shape;
    std::vector<std::vector<ptrdiff_t>> st = {
        stridesFor(*a, shape), stridesFor(*b, shape), out->strides
    };
    coalesce(sh, st);
    const std::vector<size_t> offs = { a->offset, b->offset, out->offset };

    const NdArray& A = *a; const NdArray& B = *b; NdArray& O = *out;
    const bool boolOut = (rdt == DType::BOOL);
    if (cdt == DType::I64 || cdt == DType::BOOL) {
        nditer(sh, st, offs, [&](const size_t* o) {
            const int64_t r = fi(loadI(A, o[0]), loadI(B, o[1]));
            if (boolOut) storeI(O, o[2], r != 0 ? 1 : 0); else storeI(O, o[2], r);
        });
    } else {
        nditer(sh, st, offs, [&](const size_t* o) {
            const double r = fd(loadD(A, o[0]), loadD(B, o[1]));
            if (boolOut) storeI(O, o[2], r != 0.0 ? 1 : 0); else storeD(O, o[2], r);
        });
    }
    if (plan.finalOut) copyInto(*out, *plan.finalOut, what);
    return wrap(plan.finalOut ? plan.finalOut : out);
}

// ── unary application ────────────────────────────────────────────────────────
template <class FD, class FI>
inline Value applyUnary(const char* what, Kind kind,
                        const Value& av, const Value* outArg, FD fd, FI fi) {
    ArrayPtr a = asArray(av);
    const std::vector<size_t> shape = a->shape;
    const DType cdt = computeDType(kind, a->dtype, a->dtype);
    const DType rdt = resultDType(kind, cdt);

    OutPlan plan = planOut(outArg, shape, rdt, {a.get()}, what);
    ArrayPtr out = plan.dest;
    const size_t n = out->size();
    if (n == 0) { if (plan.finalOut) copyInto(*out, *plan.finalOut, what);
                  return wrap(plan.finalOut ? plan.finalOut : out); }

    const bool flat = a->isCContig() && out->isCContig() &&
                      a->dtype == cdt && out->dtype == rdt;
    if (flat) {
        if (cdt == DType::F64) {
            const double* ap = a->ptr<double>();
            if (rdt == DType::BOOL)
                tier0cmp(ap, ap, out->ptr<uint8_t>(), n,
                         [&](double x, double) { return fd(x) != 0.0; });
            else tier0u(ap, out->ptr<double>(), n, fd);
        } else if (cdt == DType::I64) {
            const int64_t* ap = a->ptr<int64_t>();
            if (rdt == DType::BOOL)
                tier0cmp(ap, ap, out->ptr<uint8_t>(), n,
                         [&](int64_t x, int64_t) { return fi(x) != 0; });
            else tier0u(ap, out->ptr<int64_t>(), n, fi);
        } else {
            const uint8_t* ap = a->ptr<uint8_t>();
            tier0cmp(ap, ap, out->ptr<uint8_t>(), n,
                     [&](uint8_t x, uint8_t) { return fi(x ? 1 : 0) != 0; });
        }
        if (plan.finalOut) copyInto(*out, *plan.finalOut, what);
        return wrap(plan.finalOut ? plan.finalOut : out);
    }

    std::vector<size_t> sh = shape;
    std::vector<std::vector<ptrdiff_t>> st = { stridesFor(*a, shape), out->strides };
    coalesce(sh, st);
    const std::vector<size_t> offs = { a->offset, out->offset };
    const NdArray& A = *a; NdArray& O = *out;
    const bool boolOut = (rdt == DType::BOOL);
    if (cdt == DType::I64 || cdt == DType::BOOL) {
        nditer(sh, st, offs, [&](const size_t* o) {
            const int64_t r = fi(loadI(A, o[0]));
            storeI(O, o[1], boolOut ? (r != 0 ? 1 : 0) : r);
        });
    } else {
        nditer(sh, st, offs, [&](const size_t* o) {
            const double r = fd(loadD(A, o[0]));
            if (boolOut) storeI(O, o[1], r != 0.0 ? 1 : 0); else storeD(O, o[1], r);
        });
    }
    if (plan.finalOut) copyInto(*out, *plan.finalOut, what);
    return wrap(plan.finalOut ? plan.finalOut : out);
}

// ── integer division and modulo ──────────────────────────────────────────────
// Integer division by zero is SIGFPE on x86 -- a process kill, not an error, so
// it must be caught before the instruction executes. LLONG_MIN / -1 overflows
// and traps the same way. Float division by zero is NOT an error: IEEE says
// +-inf and that is the right answer.
inline int64_t idiv(int64_t a, int64_t b) {
    if (b == 0) throw std::runtime_error("integer division by zero");
    if (b == -1 && a == INT64_MIN) throw std::runtime_error("integer overflow in division");
    // Floor division, matching Python and NumPy rather than C's truncation:
    // -7 // 2 is -4, not -3.
    int64_t q = a / b, r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) q--;
    return q;
}
inline int64_t imod(int64_t a, int64_t b) {
    if (b == 0) throw std::runtime_error("integer modulo by zero");
    if (b == -1 && a == INT64_MIN) throw std::runtime_error("integer overflow in modulo");
    int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}
inline double fdiv(double a, double b) { return std::floor(a / b); }
inline double fmod_py(double a, double b) {
    const double r = std::fmod(a, b);
    return (r != 0.0 && ((r < 0) != (b < 0))) ? r + b : r;
}

}   // namespace numba
