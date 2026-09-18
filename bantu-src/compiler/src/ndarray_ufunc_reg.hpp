#pragma once
/**
 * numba — the ufunc builtin table.
 *
 * Included by ndarray_native.cpp only, after ndarray_api.hpp. Split out so the
 * ~55 registrations do not bury the creation/shape builtins.
 *
 * Every binary ufunc takes an optional trailing `out`, because
 * `$a.add($b).mul(2)` on 10M f64 holds three live 80 MB buffers and out= is
 * what lets a loop run in constant memory (docs/numba-architecture.md §4.7).
 */

#include "ndarray_ufunc.hpp"
#include "ndarray_api.hpp"

namespace numba {

// A scalar, a list or an array all become an array, so `nd_add($a, 2)` works
// without the caller wrapping the 2. A 0-d array broadcasts against anything.
Value toArrayValue(const Value& v, const char* what);

inline const Value* outOf(const std::vector<Value>& a, size_t i) {
    return (a.size() > i && !a[i].isNull()) ? &a[i] : nullptr;
}

inline void registerUfuncs(const DefineFn& define) {

    // ── binary: arithmetic ───────────────────────────────────────────────────
    // `auto` here is load-bearing, not style. Declaring these as function
    // POINTERS made every op decay to one type, so tier0 had a single shared
    // instantiation calling through an opaque pointer once per element -- which
    // cannot inline and cannot vectorize. Measured at 11 GB/s against 16 GB/s
    // for the identical loop with the operation's own type. Generic lambdas
    // keep each op distinct, so each gets its own inlined, vectorized loop.
    auto bin = [&define](const char* name, Kind k, auto fd, auto fi) {
        std::string n = name;
        define(name, [n, k, fd, fi](std::vector<Value> a) -> Value {
            if (a.size() < 2) {
                throw std::runtime_error(n + "(a, b, out?) needs two operands");
            }
            const Value av = toArrayValue(a[0], n.c_str());
            const Value bv = toArrayValue(a[1], n.c_str());
            return applyBinary(n.c_str(), k, av, bv, outOf(a, 2), fd, fi);
        });
    };

    bin("nd_add",      Kind::ARITH, [](double x, double y) { return x + y; },
                                    [](int64_t x, int64_t y) { return (int64_t)((uint64_t)x + (uint64_t)y); });
    bin("nd_subtract", Kind::ARITH, [](double x, double y) { return x - y; },
                                    [](int64_t x, int64_t y) { return (int64_t)((uint64_t)x - (uint64_t)y); });
    bin("nd_multiply", Kind::ARITH, [](double x, double y) { return x * y; },
                                    [](int64_t x, int64_t y) { return (int64_t)((uint64_t)x * (uint64_t)y); });
    // True division always yields f64 -- as in NumPy and Python 3, unlike C.
    bin("nd_divide",   Kind::REAL,  [](double x, double y) { return x / y; },
                                    [](int64_t, int64_t) -> int64_t { return 0; });
    bin("nd_floor_divide", Kind::ARITH, fdiv, idiv);
    bin("nd_mod",      Kind::ARITH, fmod_py, imod);
    bin("nd_power",    Kind::REAL,  [](double x, double y) { return std::pow(x, y); },
                                    [](int64_t, int64_t) -> int64_t { return 0; });
    bin("nd_atan2",    Kind::REAL,  [](double x, double y) { return std::atan2(x, y); },
                                    [](int64_t, int64_t) -> int64_t { return 0; });
    bin("nd_hypot",    Kind::REAL,  [](double x, double y) { return std::hypot(x, y); },
                                    [](int64_t, int64_t) -> int64_t { return 0; });
    bin("nd_copysign", Kind::REAL,  [](double x, double y) { return std::copysign(x, y); },
                                    [](int64_t, int64_t) -> int64_t { return 0; });
    // NaN propagates, matching NumPy's minimum/maximum (not fmin/fmax).
    bin("nd_minimum",  Kind::ARITH,
        [](double x, double y) { return (std::isnan(x) || std::isnan(y)) ? NAN : (x < y ? x : y); },
        [](int64_t x, int64_t y) { return x < y ? x : y; });
    bin("nd_maximum",  Kind::ARITH,
        [](double x, double y) { return (std::isnan(x) || std::isnan(y)) ? NAN : (x > y ? x : y); },
        [](int64_t x, int64_t y) { return x > y ? x : y; });

    // ── binary: comparisons and logic, all producing bool ────────────────────
    bin("nd_equal",         Kind::COMPARE, [](double x, double y) { return (double)(x == y); },
                                           [](int64_t x, int64_t y) { return (int64_t)(x == y); });
    bin("nd_not_equal",     Kind::COMPARE, [](double x, double y) { return (double)(x != y); },
                                           [](int64_t x, int64_t y) { return (int64_t)(x != y); });
    bin("nd_less",          Kind::COMPARE, [](double x, double y) { return (double)(x <  y); },
                                           [](int64_t x, int64_t y) { return (int64_t)(x <  y); });
    bin("nd_less_equal",    Kind::COMPARE, [](double x, double y) { return (double)(x <= y); },
                                           [](int64_t x, int64_t y) { return (int64_t)(x <= y); });
    bin("nd_greater",       Kind::COMPARE, [](double x, double y) { return (double)(x >  y); },
                                           [](int64_t x, int64_t y) { return (int64_t)(x >  y); });
    bin("nd_greater_equal", Kind::COMPARE, [](double x, double y) { return (double)(x >= y); },
                                           [](int64_t x, int64_t y) { return (int64_t)(x >= y); });
    bin("nd_logical_and", Kind::LOGIC, [](double x, double y) { return (double)(x != 0 && y != 0); },
                                       [](int64_t x, int64_t y) { return (int64_t)(x != 0 && y != 0); });
    bin("nd_logical_or",  Kind::LOGIC, [](double x, double y) { return (double)(x != 0 || y != 0); },
                                       [](int64_t x, int64_t y) { return (int64_t)(x != 0 || y != 0); });
    bin("nd_logical_xor", Kind::LOGIC, [](double x, double y) { return (double)((x != 0) != (y != 0)); },
                                       [](int64_t x, int64_t y) { return (int64_t)((x != 0) != (y != 0)); });

    // ── unary ────────────────────────────────────────────────────────────────
    auto un = [&define](const char* name, Kind k, auto fd, auto fi) {
        std::string n = name;
        define(name, [n, k, fd, fi](std::vector<Value> a) -> Value {
            if (a.empty()) throw std::runtime_error(n + "(a, out?) needs an operand");
            const Value av = toArrayValue(a[0], n.c_str());
            return applyUnary(n.c_str(), k, av, outOf(a, 1), fd, fi);
        });
    };

    un("nd_negative",   Kind::ARITH, [](double x) { return -x; },
                                     [](int64_t x) { return (int64_t)(0 - (uint64_t)x); });
    un("nd_abs",        Kind::ARITH, [](double x) { return std::fabs(x); },
                                     [](int64_t x) { return x < 0 ? (int64_t)(0 - (uint64_t)x) : x; });
    un("nd_sign",       Kind::ARITH,
       [](double x) { return std::isnan(x) ? NAN : (double)((x > 0) - (x < 0)); },
       [](int64_t x) { return (int64_t)((x > 0) - (x < 0)); });
    un("nd_square",     Kind::ARITH, [](double x) { return x * x; },
                                     [](int64_t x) { return (int64_t)((uint64_t)x * (uint64_t)x); });

    auto real1 = [&un](const char* name, auto fd) {
        un(name, Kind::REAL, fd, [](int64_t) -> int64_t { return 0; });
    };
    real1("nd_sqrt",   [](double x) { return std::sqrt(x); });
    real1("nd_cbrt",   [](double x) { return std::cbrt(x); });
    real1("nd_exp",    [](double x) { return std::exp(x); });
    real1("nd_expm1",  [](double x) { return std::expm1(x); });
    real1("nd_log",    [](double x) { return std::log(x); });
    real1("nd_log1p",  [](double x) { return std::log1p(x); });
    real1("nd_log2",   [](double x) { return std::log2(x); });
    real1("nd_log10",  [](double x) { return std::log10(x); });
    real1("nd_sin",    [](double x) { return std::sin(x); });
    real1("nd_cos",    [](double x) { return std::cos(x); });
    real1("nd_tan",    [](double x) { return std::tan(x); });
    real1("nd_asin",   [](double x) { return std::asin(x); });
    real1("nd_acos",   [](double x) { return std::acos(x); });
    real1("nd_atan",   [](double x) { return std::atan(x); });
    real1("nd_sinh",   [](double x) { return std::sinh(x); });
    real1("nd_cosh",   [](double x) { return std::cosh(x); });
    real1("nd_tanh",   [](double x) { return std::tanh(x); });
    real1("nd_asinh",  [](double x) { return std::asinh(x); });
    real1("nd_acosh",  [](double x) { return std::acosh(x); });
    real1("nd_atanh",  [](double x) { return std::atanh(x); });
    real1("nd_floor",  [](double x) { return std::floor(x); });
    real1("nd_ceil",   [](double x) { return std::ceil(x); });
    real1("nd_trunc",  [](double x) { return std::trunc(x); });
    // Half-to-even, matching NumPy's rint and IEEE: round(0.5) is 0, not 1.
    real1("nd_rint",   [](double x) { return std::nearbyint(x); });
    real1("nd_round",  [](double x) { return std::nearbyint(x); });
    real1("nd_reciprocal", [](double x) { return 1.0 / x; });
    real1("nd_degrees", [](double x) { return x * (180.0 / 3.14159265358979323846); });
    real1("nd_radians", [](double x) { return x * (3.14159265358979323846 / 180.0); });

    // Predicates: f64 compute, bool result.
    auto pred = [&define](const char* name, auto fd) {
        std::string n = name;
        define(name, [n, fd](std::vector<Value> a) -> Value {
            if (a.empty()) throw std::runtime_error(n + "(a, out?) needs an operand");
            const Value av = toArrayValue(a[0], n.c_str());
            return applyUnary(n.c_str(), Kind::PREDICATE, av, outOf(a, 1), fd,
                              [](int64_t) -> int64_t { return 0; });
        });
    };
    pred("nd_isnan",    [](double x) { return (double)(std::isnan(x) ? 1 : 0); });
    pred("nd_isinf",    [](double x) { return (double)(std::isinf(x) ? 1 : 0); });
    pred("nd_isfinite", [](double x) { return (double)(std::isfinite(x) ? 1 : 0); });

    define("nd_logical_not", [](std::vector<Value> a) -> Value {
        if (a.empty()) throw std::runtime_error("nd_logical_not(a, out?) needs an operand");
        const Value av = toArrayValue(a[0], "nd_logical_not");
        return applyUnary("nd_logical_not", Kind::LOGIC, av, outOf(a, 1),
                          [](double x) { return (double)(x == 0.0); },
                          [](int64_t x) { return (int64_t)(x == 0); });
    });
}

// ── three-operand and whole-array predicates ────────────────────────────────
// These do not fit the binary/unary shapes: where() broadcasts THREE operands
// and clip() has two optional bounds.
inline void registerUfuncExtras(const DefineFn& define) {

    // nd_where(cond, x, y) -- element-wise select, all three broadcast together.
    define("nd_where", [](std::vector<Value> a) -> Value {
        if (a.size() < 3) throw std::runtime_error("nd_where(cond, x, y) needs three operands");
        ArrayPtr c = asArray(toArrayValue(a[0], "nd_where"));
        ArrayPtr x = asArray(toArrayValue(a[1], "nd_where"));
        ArrayPtr y = asArray(toArrayValue(a[2], "nd_where"));
        std::vector<size_t> shape = broadcastShapes(c->shape, x->shape, "nd_where");
        shape = broadcastShapes(shape, y->shape, "nd_where");
        const DType rdt = promote(x->dtype, y->dtype);
        auto out = makeArray(shape, rdt, "nd_where");
        if (out->size() == 0) return wrap(out);

        std::vector<size_t> sh = shape;
        std::vector<std::vector<ptrdiff_t>> st = {
            stridesFor(*c, shape), stridesFor(*x, shape), stridesFor(*y, shape), out->strides
        };
        coalesce(sh, st);
        const std::vector<size_t> offs = { c->offset, x->offset, y->offset, out->offset };
        const NdArray& C = *c; const NdArray& X = *x; const NdArray& Y = *y; NdArray& O = *out;
        if (rdt == DType::F64) {
            nditer(sh, st, offs, [&](const size_t* o) {
                storeD(O, o[3], loadD(C, o[0]) != 0.0 ? loadD(X, o[1]) : loadD(Y, o[2]));
            });
        } else {
            nditer(sh, st, offs, [&](const size_t* o) {
                storeI(O, o[3], loadD(C, o[0]) != 0.0 ? loadI(X, o[1]) : loadI(Y, o[2]));
            });
        }
        return wrap(out);
    });

    // nd_clip(a, lo, hi) -- either bound may be null, meaning "unbounded that side".
    define("nd_clip", [](std::vector<Value> a) -> Value {
        if (a.size() < 3) throw std::runtime_error("nd_clip(a, lo, hi, out?) needs an array and two bounds");
        if (a[1].isNull() && a[2].isNull()) {
            throw std::runtime_error("nd_clip: at least one of lo and hi must be given");
        }
        ArrayPtr x = asArray(toArrayValue(a[0], "nd_clip"));
        const bool hasLo = !a[1].isNull(), hasHi = !a[2].isNull();
        ArrayPtr lo = hasLo ? asArray(toArrayValue(a[1], "nd_clip")) : nullptr;
        ArrayPtr hi = hasHi ? asArray(toArrayValue(a[2], "nd_clip")) : nullptr;
        std::vector<size_t> shape = x->shape;
        if (lo) shape = broadcastShapes(shape, lo->shape, "nd_clip");
        if (hi) shape = broadcastShapes(shape, hi->shape, "nd_clip");
        DType rdt = x->dtype;
        if (lo) rdt = promote(rdt, lo->dtype);
        if (hi) rdt = promote(rdt, hi->dtype);
        if (rdt == DType::BOOL) rdt = DType::I64;

        const Value* outArg = outOf(a, 3);
        std::vector<const NdArray*> ins = { x.get(), lo.get(), hi.get() };
        OutPlan plan = planOut(outArg, shape, rdt, ins, "nd_clip");
        ArrayPtr out = plan.dest;
        if (out->size() != 0) {
            // A missing bound walks a 1-element dummy with stride 0, so the
            // odometer stays one shape rather than four specialisations.
            ArrayPtr dummy = makeArray({1}, DType::F64, "nd_clip");
            std::vector<size_t> sh = shape;
            std::vector<std::vector<ptrdiff_t>> st = {
                stridesFor(*x, shape),
                stridesFor(lo ? *lo : *dummy, shape),
                stridesFor(hi ? *hi : *dummy, shape),
                out->strides
            };
            coalesce(sh, st);
            const std::vector<size_t> offs = { x->offset, lo ? lo->offset : dummy->offset,
                                               hi ? hi->offset : dummy->offset, out->offset };
            const NdArray& X = *x; NdArray& O = *out;
            const NdArray& L = lo ? *lo : *dummy;
            const NdArray& H = hi ? *hi : *dummy;
            if (rdt == DType::F64) {
                nditer(sh, st, offs, [&](const size_t* o) {
                    double v = loadD(X, o[0]);
                    if (hasLo) { const double l = loadD(L, o[1]); if (v < l) v = l; }
                    if (hasHi) { const double h = loadD(H, o[2]); if (v > h) v = h; }
                    storeD(O, o[3], v);
                });
            } else {
                nditer(sh, st, offs, [&](const size_t* o) {
                    int64_t v = loadI(X, o[0]);
                    if (hasLo) { const int64_t l = loadI(L, o[1]); if (v < l) v = l; }
                    if (hasHi) { const int64_t h = loadI(H, o[2]); if (v > h) v = h; }
                    storeI(O, o[3], v);
                });
            }
        }
        if (plan.finalOut) copyInto(*out, *plan.finalOut, "nd_clip");
        return wrap(plan.finalOut ? plan.finalOut : out);
    });

    // Tolerance comparison, NumPy's formula exactly: |a-b| <= atol + rtol*|b|.
    // It is deliberately asymmetric in b -- matching NumPy matters more than
    // elegance here, because people diff numba against NumPy.
    auto closeTest = [](double x, double y, double rtol, double atol, bool equalNan) {
        if (std::isnan(x) || std::isnan(y)) return equalNan && std::isnan(x) && std::isnan(y);
        if (std::isinf(x) || std::isinf(y)) return x == y;
        return std::fabs(x - y) <= atol + rtol * std::fabs(y);
    };

    define("nd_isclose", [closeTest](std::vector<Value> a) -> Value {
        if (a.size() < 2) throw std::runtime_error("nd_isclose(a, b, rtol?, atol?, equal_nan?) needs two operands");
        ArrayPtr x = asArray(toArrayValue(a[0], "nd_isclose"));
        ArrayPtr y = asArray(toArrayValue(a[1], "nd_isclose"));
        const double rtol = (a.size() > 2 && !a[2].isNull()) ? a[2].numberVal : 1e-5;
        const double atol = (a.size() > 3 && !a[3].isNull()) ? a[3].numberVal : 1e-8;
        const bool eqNan  = (a.size() > 4 && !a[4].isNull()) ? a[4].isTruthy() : false;
        const std::vector<size_t> shape = broadcastShapes(x->shape, y->shape, "nd_isclose");
        auto out = makeArray(shape, DType::BOOL, "nd_isclose");
        if (out->size() == 0) return wrap(out);
        std::vector<size_t> sh = shape;
        std::vector<std::vector<ptrdiff_t>> st = {
            stridesFor(*x, shape), stridesFor(*y, shape), out->strides };
        coalesce(sh, st);
        const std::vector<size_t> offs = { x->offset, y->offset, out->offset };
        const NdArray& X = *x; const NdArray& Y = *y; NdArray& O = *out;
        nditer(sh, st, offs, [&](const size_t* o) {
            storeI(O, o[2], closeTest(loadD(X, o[0]), loadD(Y, o[1]), rtol, atol, eqNan) ? 1 : 0);
        });
        return wrap(out);
    });

    define("nd_allclose", [closeTest](std::vector<Value> a) -> Value {
        if (a.size() < 2) throw std::runtime_error("nd_allclose(a, b, rtol?, atol?, equal_nan?) needs two operands");
        ArrayPtr x = asArray(toArrayValue(a[0], "nd_allclose"));
        ArrayPtr y = asArray(toArrayValue(a[1], "nd_allclose"));
        const double rtol = (a.size() > 2 && !a[2].isNull()) ? a[2].numberVal : 1e-5;
        const double atol = (a.size() > 3 && !a[3].isNull()) ? a[3].numberVal : 1e-8;
        const bool eqNan  = (a.size() > 4 && !a[4].isNull()) ? a[4].isTruthy() : false;
        const std::vector<size_t> shape = broadcastShapes(x->shape, y->shape, "nd_allclose");
        std::vector<size_t> sh = shape;
        std::vector<std::vector<ptrdiff_t>> st = { stridesFor(*x, shape), stridesFor(*y, shape) };
        coalesce(sh, st);
        const std::vector<size_t> offs = { x->offset, y->offset };
        const NdArray& X = *x; const NdArray& Y = *y;
        bool all = true;
        nditer(sh, st, offs, [&](const size_t* o) {
            if (all && !closeTest(loadD(X, o[0]), loadD(Y, o[1]), rtol, atol, eqNan)) all = false;
        });
        return Value(all);
    });

    // Shape-sensitive equality: unlike allclose this does NOT broadcast, because
    // "are these the same array" and "do these agree where they overlap" are
    // different questions and conflating them hides bugs.
    define("nd_array_equal", [](std::vector<Value> a) -> Value {
        if (a.size() < 2) throw std::runtime_error("nd_array_equal(a, b) needs two operands");
        ArrayPtr x = asArray(toArrayValue(a[0], "nd_array_equal"));
        ArrayPtr y = asArray(toArrayValue(a[1], "nd_array_equal"));
        if (x->shape != y->shape) return Value(false);
        std::vector<size_t> sh = x->shape;
        std::vector<std::vector<ptrdiff_t>> st = { x->strides, y->strides };
        coalesce(sh, st);
        const std::vector<size_t> offs = { x->offset, y->offset };
        const NdArray& X = *x; const NdArray& Y = *y;
        bool same = true;
        nditer(sh, st, offs, [&](const size_t* o) {
            if (same && loadD(X, o[0]) != loadD(Y, o[1])) same = false;
        });
        return Value(same);
    });

    // The broadcast result shape, without allocating anything. Lets a Bantu
    // caller check compatibility before committing to a large allocation.
    define("nd_broadcast_shapes", [](std::vector<Value> a) -> Value {
        if (a.size() < 2) throw std::runtime_error("nd_broadcast_shapes(a, b) needs two operands");
        ArrayPtr x = asArray(toArrayValue(a[0], "nd_broadcast_shapes"));
        ArrayPtr y = asArray(toArrayValue(a[1], "nd_broadcast_shapes"));
        const std::vector<size_t> s = broadcastShapes(x->shape, y->shape, "nd_broadcast_shapes");
        std::vector<Value> out;
        for (size_t d : s) out.push_back(Value((double)d));
        return Value(out);
    });
}

} // namespace numba
