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
    PREDICATE,  // isnan/isinf/isfinite -- ALWAYS f64 in, bool out
    BITWISE     // & | ^ ~ << >>  -- integers only, a float operand is an error
};

inline DType computeDType(Kind k, DType a, DType b) {
    switch (k) {
        case Kind::REAL:      return (promote(a, b) == DType::F32) ? DType::F32
                                                                   : DType::F64;
        // A predicate asks a question only a float can answer, so it computes
        // in f64 whatever it was handed. Letting it promote to i64 sent it down
        // the integer path, where these ops have no meaningful kernel at all:
        // nd_isfinite(nd([1], "i64")) returned FALSE.
        case Kind::PREDICATE: return (promote(a, b) == DType::F32) ? DType::F32
                                                                   : DType::F64;
        // Truthiness is `!= 0`, which has to be evaluated in the operand's own
        // type. Computing logic in BOOL meant f64 operands were converted by
        // ROUNDING, so 0.4 was false and 0.6 was true -- llround, not truth.
        case Kind::LOGIC:     return promote(a, b);
        case Kind::COMPARE: {
            const DType p = promote(a, b);
            return p == DType::BOOL ? DType::I64 : p;   // compare bools as 0/1
        }
        // Bitwise asks about the bits of an integer. An f64 operand has no
        // answer -- rounding it to i64 to produce one would make a bug look
        // like a feature -- so applyBinary/applyUnary reject it by name.
        case Kind::BITWISE: {
            const DType p = promote(a, b);
            return p == DType::BOOL ? DType::I64 : p;
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

// ── a bare Bantu number is a WEAK operand (D2) ───────────────────────────────
// Without this, `x * 0.5` on an f32 array returns f64, and ONE scalar multiply
// undoes the entire point of f32. Bantu numbers are doubles, so a literal
// becomes an f64 array and the promotion lattice dutifully widens the result.
// NumPy solves this with NEP 50's weak scalars and PyTorch with the same idea.
//
// Deliberately narrow: this fires ONLY where a scalar's f64 meets an f32 array.
// Every pre-f32 combination is untouched and bit-identical -- notably
// `i64_array + 1.5`, which still gives f64, because the scalar is not being
// weakened against an f32 operand. A real f64 ARRAY against an f32 array also
// still widens to f64, since only a bare number is weak.
inline DType weakDType(const Value& v, const NdArray& self, DType other) {
    if (self.dtype == DType::F64 && other == DType::F32 &&
        (v.isNumber() || v.isBool())) {
        return DType::F32;
    }
    return self.dtype;
}

// The same rule, applied where the RAW arguments are still in hand.
//
// Needed because the nd_* registration wrapper calls toArrayValue() on both
// operands BEFORE applyBinary, so by then a bare number has already become an
// f64 array and is indistinguishable from a real one -- weakDType above can
// never fire for it. The operator path (ndarray_dispatch.hpp) does pass raw
// Values, which is why `$f32 + 0.5` worked while `nd_add($f32, 0.5)` did not.
//
// Narrowing a size-1 array is a copy of one element, so this costs nothing on
// the path it fires on.
inline void weakenScalars(const Value& rawA, Value& av,
                          const Value& rawB, Value& bv) {
    ArrayPtr A = asArray(av), B = asArray(bv);
    const DType wa = weakDType(rawA, *A, B->dtype);
    const DType wb = weakDType(rawB, *B, A->dtype);
    if (wa != A->dtype) av = wrap(contiguousCopy(A, wa));
    if (wb != B->dtype) bv = wrap(contiguousCopy(B, wb));
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
// ── bitwise operands (docs/numba-architecture.md §4.6.1) ─────────────────────
// A Bantu number is a double, so `nd_bitwise_and($a, 3)` hands a BITWISE op an
// f64 operand for an integer the caller wrote as an integer -- and so does any
// array that came out of nd_divide. Converting a value that already IS an
// integer is the only reading of such a call. Converting 1.5 would be a guess,
// so that is an error naming the value rather than a silent rounding.
inline ArrayPtr asIntegerOperand(const ArrayPtr& a, const char* what) {
    // Any FLOAT dtype goes through the integrality check, not just f64 (D2).
    // Accepting an integral-valued f64 array while refusing the identical f32
    // one would make f32 a dtype with surprising holes; widening f32 to f64
    // here is exact, so the check below is unchanged.
    if (a->dtype != DType::F64 && a->dtype != DType::F32) return a;
    ArrayPtr c = contiguousCopy(a, DType::F64);
    const double* p = c->ptr<double>();
    const size_t n = c->size();
    for (size_t i = 0; i < n; i++) {
        const double v = p[i];
        if (!(v >= -9223372036854775808.0) || !(v <= 9223372036854775807.0) ||
            v != std::floor(v)) {
            throw std::runtime_error(std::string(what) +
                ": bitwise operations need integers, and " + std::to_string(v) +
                " is not one");
        }
    }
    return contiguousCopy(c, DType::I64);
}

// --- one arm of each tier, over the compute type (D1) ------------------------
// The float and integer arms of all three binary tiers and both unary tiers
// were structurally identical modulo T and which functor they carried -- six
// near-duplicate blocks. Each is now written ONCE over T, so:
//
//   * a new dtype (f32 in D2, i8 in ecosystem stage 6) is ONE line per call
//     site instead of a copied block, and
//   * the tiers cannot drift apart, which is how tier 1 came to be dead code
//     in the first place (see the tier-1 comment inside applyBinary).
//
// BOOL compute keeps its own arm at each site rather than joining these: its
// functor is called on WIDENED ints (`x ? 1 : 0`) and it has no arithmetic
// form at all, so folding it in would need a narrower helper for every caller.

template <class T, class Op>
inline void binTier0(const NdArray& a, const NdArray& b, NdArray& out,
                     size_t n, bool boolOut, Op op) {
    const T* ap = a.ptr<T>();
    const T* bp = b.ptr<T>();
    if (boolOut) tier0cmp(ap, bp, out.ptr<uint8_t>(), n,
                          [&](T x, T y) { return op(x, y) != T(0); });
    else         tier0(ap, bp, out.ptr<T>(), n, op);
}

template <class T, class Op>
inline void binTier1(const NdArray& arr, const NdArray& scl, NdArray& out,
                     size_t n, bool boolOut, bool scalarRight, Op op) {
    const T* p = arr.ptr<T>();
    const T  s = load<T>(scl, scl.offset);
    if (boolOut) {
        uint8_t* o = out.ptr<uint8_t>();
        if (scalarRight) for (size_t i = 0; i < n; i++) o[i] = op(p[i], s) != T(0) ? 1 : 0;
        else             for (size_t i = 0; i < n; i++) o[i] = op(s, p[i]) != T(0) ? 1 : 0;
    } else {
        T* o = out.ptr<T>();
        if (scalarRight) tier1(p, s, o, n, op);
        else             tier1r(s, p, o, n, op);
    }
}

template <class T, class Op>
inline void binTier2(const NdArray& A, const NdArray& B, NdArray& O,
                     const std::vector<size_t>& sh,
                     const std::vector<std::vector<ptrdiff_t>>& st,
                     const std::vector<size_t>& offs, bool boolOut, Op op) {
    nditer(sh, st, offs, [&](const size_t* o) {
        const T r = op(load<T>(A, o[0]), load<T>(B, o[1]));
        // The bool result goes through storeI deliberately: the destination is
        // BOOL and the value is already reduced to 0/1, so store<T> would send
        // a double back through setFromDouble's `!= 0.0` for nothing.
        if (boolOut) storeI(O, o[2], r != T(0) ? 1 : 0);
        else         store<T>(O, o[2], r);
    });
}

template <class T, class Op>
inline void unTier0(const NdArray& a, NdArray& out, size_t n, bool boolOut, Op op) {
    const T* ap = a.ptr<T>();
    if (boolOut) tier0cmp(ap, ap, out.ptr<uint8_t>(), n,
                          [&](T x, T) { return op(x) != T(0); });
    else         tier0u(ap, out.ptr<T>(), n, op);
}

template <class T, class Op>
inline void unTier2(const NdArray& A, NdArray& O,
                    const std::vector<size_t>& sh,
                    const std::vector<std::vector<ptrdiff_t>>& st,
                    const std::vector<size_t>& offs, bool boolOut, Op op) {
    nditer(sh, st, offs, [&](const size_t* o) {
        const T r = op(load<T>(A, o[0]));
        if (boolOut) storeI(O, o[1], r != T(0) ? 1 : 0);
        else         store<T>(O, o[1], r);
    });
}

template <class FD, class FI>

inline Value applyBinary(const char* what, Kind kind,
                         const Value& av, const Value& bv, const Value* outArg,
                         FD fd, FI fi) {
    ArrayPtr a = asArray(av), b = asArray(bv);
    if (kind == Kind::BITWISE) {
        a = asIntegerOperand(a, what);
        b = asIntegerOperand(b, what);
    }
    const std::vector<size_t> shape = broadcastShapes(a->shape, b->shape, what);
    const DType cdt = computeDType(kind, weakDType(av, *a, b->dtype),
                                         weakDType(bv, *b, a->dtype));
    if (kind == Kind::BITWISE && cdt != DType::I64) {
        throw std::runtime_error(std::string(what) +
            ": bitwise operations need integers, got " + dtypeName(cdt));
    }
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
        const bool boolOut0 = (rdt == DType::BOOL);
        if      (cdt == DType::F64) binTier0<double >(*a, *b, *out, n, boolOut0, fd);
        else if (cdt == DType::F32) binTier0<float  >(*a, *b, *out, n, boolOut0, fd);
        else if (cdt == DType::I64) binTier0<int64_t>(*a, *b, *out, n, boolOut0, fi);
        else {   // BOOL compute: logic ops only, functor called on widened ints
            const uint8_t* ap = a->ptr<uint8_t>();
            const uint8_t* bp = b->ptr<uint8_t>();
            tier0cmp(ap, bp, out->ptr<uint8_t>(), n,
                     [&](uint8_t x, uint8_t y) { return fi(x ? 1 : 0, y ? 1 : 0) != 0; });
        }
        if (plan.finalOut) copyInto(*out, *plan.finalOut, what);
        return wrap(plan.finalOut ? plan.finalOut : out);
    }

    // ── tier 1 ──────────────────────────────────────────────────────────────
    // One side is a single value, splatted over the other. The kernels for this
    // (tier1/tier1r, ndarray_kernels.hpp) were written with a comment
    // explaining why they earn their keep -- and then never called from
    // anywhere. A tree-wide search found exactly two mentions, the two
    // definitions, against ten call sites for tier0.
    //
    // So every scalar operand fell through to tier 2's strided odometer, which
    // re-reads the scalar through a stride-0 walk and does a loadD/storeD per
    // element instead of a vectorizable loop. Measured on 1M f64 before this
    // block existed:
    //
    //     nd_add($a, $b)       1.76 ms   14.3 GB/s   tier 0
    //     nd_add($a, 1.0)      7.86 ms    3.2 GB/s   tier 2   <- 4.5x slower
    //     nd_maximum($a, $b)   1.54 ms   16.3 GB/s   tier 0
    //     nd_maximum($a, 0.0)  7.38 ms    3.4 GB/s   tier 2   <- 4.8x slower
    //
    // A scalar operand does LESS memory traffic and ran 4.5x slower. And it is
    // not an exotic path: `relu(x)` is maximum(x, 0), a bias add broadcasts,
    // and `grad * lr`, `p - lr*g` and the 1/n in every mean are all scalars.
    // An `out=` destination did not help (218 vs 226 us at 32k), which is what
    // confirmed the cost was the walk and not the allocation.
    //
    // The scalar side only has to be size 1 -- its single element is converted
    // to the compute dtype here, which is free for one value, so a `[1]` i64
    // array against an f64 array takes this path too rather than falling back.
    {
        const bool outFlat = out->isCContig() && out->dtype == rdt;
        // Which side is the array, which the single value?
        const bool aFull = a->shape == shape && a->isCContig() && a->dtype == cdt;
        const bool bFull = b->shape == shape && b->isCContig() && b->dtype == cdt;
        const bool aOne  = a->size() == 1;
        const bool bOne  = b->size() == 1;
        // n > 1 so that a 1-element result (where both sides are size 1) keeps
        // going to tier 2 -- there is nothing to vectorize and the odometer is
        // already correct for it.
        if (outFlat && n > 1 &&
            (cdt == DType::F64 || cdt == DType::F32 || cdt == DType::I64) &&
            ((aFull && bOne) || (bFull && aOne))) {
            const bool scalarRight = (aFull && bOne);
            const NdArray& arr = scalarRight ? *a : *b;
            const NdArray& scl = scalarRight ? *b : *a;
            const bool boolOut1 = (rdt == DType::BOOL);
            if      (cdt == DType::F64)
                 binTier1<double >(arr, scl, *out, n, boolOut1, scalarRight, fd);
            else if (cdt == DType::F32)
                 binTier1<float  >(arr, scl, *out, n, boolOut1, scalarRight, fd);
            else binTier1<int64_t>(arr, scl, *out, n, boolOut1, scalarRight, fi);
            if (plan.finalOut) copyInto(*out, *plan.finalOut, what);
            return wrap(plan.finalOut ? plan.finalOut : out);
        }
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
    // One call covers I64 and BOOL compute together: load<int64_t>/store<int64_t>
    // ARE loadI/storeI, which already widen a BOOL operand to 0/1 and narrow
    // back on the way out -- exactly what the old two-arm form did by hand.
    if (cdt == DType::I64 || cdt == DType::BOOL)
         binTier2<int64_t>(A, B, O, sh, st, offs, boolOut, fi);
    else if (cdt == DType::F32)
         binTier2<float  >(A, B, O, sh, st, offs, boolOut, fd);
    else binTier2<double >(A, B, O, sh, st, offs, boolOut, fd);
    if (plan.finalOut) copyInto(*out, *plan.finalOut, what);
    return wrap(plan.finalOut ? plan.finalOut : out);
}

// ── unary application ────────────────────────────────────────────────────────
template <class FD, class FI>
inline Value applyUnary(const char* what, Kind kind,
                        const Value& av, const Value* outArg, FD fd, FI fi) {
    ArrayPtr a = asArray(av);
    if (kind == Kind::BITWISE) a = asIntegerOperand(a, what);
    const std::vector<size_t> shape = a->shape;
    const DType cdt = computeDType(kind, a->dtype, a->dtype);
    if (kind == Kind::BITWISE && cdt != DType::I64) {
        throw std::runtime_error(std::string(what) +
            ": bitwise operations need integers, got " + dtypeName(cdt));
    }
    const DType rdt = resultDType(kind, cdt);

    OutPlan plan = planOut(outArg, shape, rdt, {a.get()}, what);
    ArrayPtr out = plan.dest;
    const size_t n = out->size();
    if (n == 0) { if (plan.finalOut) copyInto(*out, *plan.finalOut, what);
                  return wrap(plan.finalOut ? plan.finalOut : out); }

    const bool flat = a->isCContig() && out->isCContig() &&
                      a->dtype == cdt && out->dtype == rdt;
    if (flat) {
        const bool boolOutU = (rdt == DType::BOOL);
        if      (cdt == DType::F64) unTier0<double >(*a, *out, n, boolOutU, fd);
        else if (cdt == DType::F32) unTier0<float  >(*a, *out, n, boolOutU, fd);
        else if (cdt == DType::I64) unTier0<int64_t>(*a, *out, n, boolOutU, fi);
        else {   // BOOL compute: functor called on widened ints
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
    if (cdt == DType::I64 || cdt == DType::BOOL)
         unTier2<int64_t>(A, O, sh, st, offs, boolOut, fi);
    else if (cdt == DType::F32)
         unTier2<float  >(A, O, sh, st, offs, boolOut, fd);
    else unTier2<double >(A, O, sh, st, offs, boolOut, fd);
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
