#pragma once
/**
 * numba — operator, index and method dispatch.
 *
 * Included by ndarray_native.cpp only, after ndarray_api.hpp.
 * docs/numba-architecture.md §5.
 *
 * The handle IS the object (N-of §5.1): arctic wraps its column handle in a
 * Bantu `Series` class, numba does not. Four small additive arms in the
 * evaluator make the raw handle behave like an array, which is what lets
 * `$a + $b` and `$m[1][2] = 9` mean what they look like without a wrapper
 * class, without operator overloading in the language, and without touching
 * any path that works today.
 */

#include "ndarray_reduce.hpp"
#include "ndarray_api.hpp"

#include <unordered_set>

namespace numba {

Value toArrayValue(const Value& v, const char* what);

// An operand is dispatchable when it is an array, or when it is something an
// array can be combined with: a number, a bool, or a list.
inline bool isOperand(const Value& v) {
    return isArray(v) || v.isNumber() || v.isBool() || v.isList();
}

// ── binary operators ─────────────────────────────────────────────────────────
inline bool dispatchBinaryImpl(Op op, const Value& l, const Value& r, Value& out) {
    // At least one side must be an array. `$col + 1` on an arctic column falls
    // through untouched, because a column is not an ndarray handle.
    if (!isArray(l) && !isArray(r)) return false;

    // `==`/`!=` stay identity comparisons -- see the Eq/Ne case below for why --
    // so they decline here rather than reaching the operand check.
    if (op == Op::Eq || op == Op::Ne) return false;

    // `+` with a string follows the language's own rule, where everything
    // concatenates: `5 + "x"` is "5x", so `$a + "x"` is the array's repr plus
    // "x". Declining lets the existing string arm handle it.
    if (op == Op::Add && (l.isString() || r.isString())) return false;

    // Anything else that is not combinable must RAISE. Declining here would
    // fall through to `left.numberVal + right.numberVal`, which is 0 + 0 for a
    // handle and a dict -- silently producing 0. That is exactly the failure
    // mode this dispatch exists to remove, so it must not be reintroduced at
    // the edges.
    if (!isOperand(l) || !isOperand(r)) {
        const Value& bad = isOperand(l) ? r : l;
        throw std::runtime_error("an array cannot be combined with " +
            std::string(bad.isNull() ? "null" :
                        bad.isObject() ? "a dict" :
                        bad.isString() ? "a string" :
                        bad.isList() ? "a list" :
                        bad.isNativeHandle() ? ("<" + bad.handleTag() + ">") :
                        "that value") +
            " -- use a number, a list of numbers, or another array");
    }

    const char* what = "operator";
    const Value a = toArrayValue(l, what);
    const Value b = toArrayValue(r, what);

    switch (op) {
        case Op::Add: out = applyBinary("+", Kind::ARITH, a, b, nullptr,
            [](double x, double y) { return x + y; },
            [](int64_t x, int64_t y) { return (int64_t)((uint64_t)x + (uint64_t)y); }); return true;
        case Op::Sub: out = applyBinary("-", Kind::ARITH, a, b, nullptr,
            [](double x, double y) { return x - y; },
            [](int64_t x, int64_t y) { return (int64_t)((uint64_t)x - (uint64_t)y); }); return true;
        case Op::Mul: out = applyBinary("*", Kind::ARITH, a, b, nullptr,
            [](double x, double y) { return x * y; },
            [](int64_t x, int64_t y) { return (int64_t)((uint64_t)x * (uint64_t)y); }); return true;
        // `/` yields f64 even for two integer arrays, as in NumPy and Python 3.
        case Op::Div: out = applyBinary("/", Kind::REAL, a, b, nullptr,
            [](double x, double y) { return x / y; },
            [](int64_t, int64_t) -> int64_t { return 0; }); return true;
        case Op::Mod: out = applyBinary("%", Kind::ARITH, a, b, nullptr,
            fmod_py, imod); return true;
        case Op::Lt: out = applyBinary("<", Kind::COMPARE, a, b, nullptr,
            [](double x, double y) { return (double)(x <  y); },
            [](int64_t x, int64_t y) { return (int64_t)(x <  y); }); return true;
        case Op::Le: out = applyBinary("<=", Kind::COMPARE, a, b, nullptr,
            [](double x, double y) { return (double)(x <= y); },
            [](int64_t x, int64_t y) { return (int64_t)(x <= y); }); return true;
        case Op::Gt: out = applyBinary(">", Kind::COMPARE, a, b, nullptr,
            [](double x, double y) { return (double)(x >  y); },
            [](int64_t x, int64_t y) { return (int64_t)(x >  y); }); return true;
        case Op::Ge: out = applyBinary(">=", Kind::COMPARE, a, b, nullptr,
            [](double x, double y) { return (double)(x >= y); },
            [](int64_t x, int64_t y) { return (int64_t)(x >= y); }); return true;

        // `==` and `!=` are the two that deliberately do NOT become element-wise.
        // `if ($a == $b)` is written constantly, and an element-wise result would
        // make it mean "is this array non-empty and all-truthy", silently. NumPy
        // chose element-wise and then had to make `if arr:` raise, which is a
        // permanent papercut; Bantu has no such escape hatch, so identity is the
        // safer answer. nd_array_equal and nd_allclose are the explicit forms.
        case Op::Eq:
        case Op::Ne:
            return false;
    }
    return false;
}

inline bool dispatchNegateImpl(const Value& v, Value& out) {
    if (!isArray(v)) return false;
    out = applyUnary("-", Kind::ARITH, v, nullptr,
        [](double x) { return -x; },
        [](int64_t x) { return (int64_t)(0 - (uint64_t)x); });
    return true;
}

// ── indexing ─────────────────────────────────────────────────────────────────
// `$m[1]` on a 2-d array gives a VIEW of row 1, not a copy -- so `$m[1][2] = 9`
// writes through to the base. That is the opposite of list behaviour and is
// documented loudly (§5.3): handles have reference semantics, lists do not.
inline bool dispatchIndexImpl(const Value& obj, const Value& idx, Value& out) {
    if (!isArray(obj)) return false;
    ArrayPtr a = asArray(obj);

    // A boolean array selects; an integer array gathers. Both are the "fancy
    // indexing" people expect from `$a[$mask]`.
    if (isArray(idx)) {
        ArrayPtr ix = asArray(idx);
        if (ix->dtype == DType::BOOL) {
            if (ix->size() != a->size()) {
                throw std::runtime_error("index: the mask has " + std::to_string(ix->size()) +
                    " elements but the array has " + std::to_string(a->size()));
            }
            std::vector<double> keep;
            std::vector<size_t> mo; mo.reserve(ix->size());
            const NdArray& M = *ix; const NdArray& A = *a;
            forEachIndex(M, [&](size_t o, size_t) { mo.push_back(o); });
            size_t k = 0;
            forEachIndex(A, [&](size_t o, size_t) {
                if (getAsDouble(M, mo[k++]) != 0.0) keep.push_back(getAsDouble(A, o));
            });
            ArrayPtr res = makeArray({ keep.size() }, a->dtype, "index");
            for (size_t i = 0; i < keep.size(); i++) setFromDouble(*res, i, keep[i]);
            out = wrap(res);
            return true;
        }
        // Integer array: gather along axis 0.
        if (a->ndim() == 0) throw std::runtime_error("index: cannot index a 0-d array");
        const ptrdiff_t extent = (ptrdiff_t)a->shape[0];
        std::vector<ptrdiff_t> want;
        const NdArray& IX = *ix;
        forEachIndex(IX, [&](size_t o, size_t) {
            const double d = getAsDouble(IX, o);
            ptrdiff_t i = (ptrdiff_t)std::llround(d);
            if (i < 0) i += extent;
            if (i < 0 || i >= extent) {
                throw std::runtime_error("index " + std::to_string((ptrdiff_t)d) +
                    " is out of range for axis 0 with " + std::to_string(extent) + " elements");
            }
            want.push_back(i);
        });
        std::vector<size_t> shape = a->shape;
        shape[0] = want.size();
        ArrayPtr res = makeArray(shape, a->dtype, "index");
        const size_t rowLen = res->size() / std::max<size_t>(1, want.size());
        for (size_t r = 0; r < want.size(); r++) {
            std::vector<size_t> rs(a->shape.begin() + 1, a->shape.end());
            std::vector<ptrdiff_t> rst(a->strides.begin() + 1, a->strides.end());
            ArrayPtr row = makeView(a, rs, rst,
                (size_t)((ptrdiff_t)a->offset + want[r] * a->strides[0]), "index");
            size_t k = r * rowLen;
            const NdArray& RW = *row;
            forEachIndex(RW, [&](size_t o, size_t) { setFromDouble(*res, k++, getAsDouble(RW, o)); });
        }
        out = wrap(res);
        return true;
    }

    if (!idx.isNumber()) return false;
    if (a->ndim() == 0) throw std::runtime_error("index: cannot index a 0-d array");
    const size_t i = normIndex(idx.numberVal, a->shape[0], 0);

    // A 1-d array yields the element itself; anything higher yields a view of
    // the sub-array, which is what makes $m[1][2] work.
    if (a->ndim() == 1) {
        out = getAsValue(*a, (size_t)((ptrdiff_t)a->offset + (ptrdiff_t)i * a->strides[0]));
        return true;
    }
    std::vector<size_t> rs(a->shape.begin() + 1, a->shape.end());
    std::vector<ptrdiff_t> rst(a->strides.begin() + 1, a->strides.end());
    out = wrap(makeView(a, rs, rst,
        (size_t)((ptrdiff_t)a->offset + (ptrdiff_t)i * a->strides[0]), "index"));
    return true;
}

inline bool dispatchIndexAssignImpl(const Value& obj, const Value& idx, const Value& val) {
    if (!isArray(obj)) return false;
    ArrayPtr a = asArray(obj);
    requireWritable(*a, "index assignment");

    // $a[$mask] = v — write v wherever the mask is true.
    if (isArray(idx) && asArray(idx)->dtype == DType::BOOL) {
        ArrayPtr ix = asArray(idx);
        if (ix->size() != a->size()) {
            throw std::runtime_error("index assignment: the mask has " +
                std::to_string(ix->size()) + " elements but the array has " +
                std::to_string(a->size()));
        }
        if (!val.isNumber() && !val.isBool()) {
            throw std::runtime_error("index assignment: a masked write needs a single number");
        }
        const double v = val.isBool() ? (val.boolVal ? 1.0 : 0.0) : val.numberVal;
        if (a->dtype == DType::I64 && (std::isnan(v) || std::isinf(v))) {
            throw std::runtime_error("index assignment: cannot store " + val.toString() +
                " in an i64 array -- it has no integer representation");
        }
        std::vector<size_t> mo; mo.reserve(ix->size());
        const NdArray& M = *ix;
        forEachIndex(M, [&](size_t o, size_t) { mo.push_back(o); });
        size_t k = 0;
        NdArray& A = *a;
        forEachIndex(A, [&](size_t o, size_t) {
            if (getAsDouble(M, mo[k++]) != 0.0) setFromDouble(A, o, v);
        });
        return true;
    }

    if (!idx.isNumber()) return false;
    if (a->ndim() == 0) throw std::runtime_error("index assignment: cannot index a 0-d array");
    const size_t i = normIndex(idx.numberVal, a->shape[0], 0);
    const size_t off = (size_t)((ptrdiff_t)a->offset + (ptrdiff_t)i * a->strides[0]);

    if (a->ndim() == 1) {
        if (!val.isNumber() && !val.isBool()) {
            throw std::runtime_error("index assignment: expected a number (got " +
                                     val.toString() + ")");
        }
        const double v = val.isBool() ? (val.boolVal ? 1.0 : 0.0) : val.numberVal;
        if (a->dtype == DType::I64 && (std::isnan(v) || std::isinf(v))) {
            throw std::runtime_error("index assignment: cannot store " + val.toString() +
                " in an i64 array -- it has no integer representation");
        }
        setFromDouble(*a, off, v);
        return true;
    }

    // Assigning to a whole row: broadcast the value across it.
    std::vector<size_t> rs(a->shape.begin() + 1, a->shape.end());
    std::vector<ptrdiff_t> rst(a->strides.begin() + 1, a->strides.end());
    ArrayPtr row = makeView(a, rs, rst, off, "index assignment");
    const Value rv = toArrayValue(val, "index assignment");
    ArrayPtr src = asArray(rv);
    // The row is the destination, so the source must broadcast INTO it.
    const std::vector<size_t> bshape = broadcastShapes(row->shape, src->shape, "index assignment");
    if (bshape != row->shape) {
        throw std::runtime_error("index assignment: cannot write shape " +
            shapeStr(src->shape) + " into a row of shape " + shapeStr(row->shape));
    }
    std::vector<size_t> sh = row->shape;
    std::vector<std::vector<ptrdiff_t>> st = { stridesFor(*src, sh), row->strides };
    coalesce(sh, st);
    const std::vector<size_t> offs = { src->offset, row->offset };
    const NdArray& S = *src; NdArray& D = *row;
    nditer(sh, st, offs, [&](const size_t* o) { setFromDouble(D, o[1], getAsDouble(S, o[0])); });
    return true;
}

// ── methods ──────────────────────────────────────────────────────────────────
// `$a.sum()` is the chaining form. It works on any build, with or without
// operator dispatch, which is why every method maps onto a builtin that already
// exists rather than duplicating its logic: one implementation, two spellings.
//
// The table is populated during registerBuiltins with the very same NativeFn
// objects handed to `define`, so a method and its builtin cannot drift apart.
inline std::unordered_map<std::string, NativeFn>& methodTable() {
    static std::unordered_map<std::string, NativeFn> t;
    return t;
}

// Called once per builtin at registration: remember `nd_sum` under "sum".
inline void noteMethod(const char* builtinName, const NativeFn& fn) {
    static const std::unordered_set<std::string> kMethods = {
        "sum", "prod", "mean", "var", "std", "min", "max", "ptp",
        "argmin", "argmax", "any", "all", "count_nonzero", "median", "quantile",
        "nansum", "nanmean", "nanmin", "nanmax",
        "shape", "ndim", "size", "dtype", "strides", "itemsize", "nbytes",
        "is_contiguous", "is_view", "writable", "base_id",
        "reshape", "transpose", "T", "ravel", "flatten", "swapaxes", "moveaxis",
        "expand_dims", "squeeze", "slice", "broadcast_to", "flip",
        "copy", "astype", "ascontiguous", "to_list", "get", "set",
        "add", "subtract", "multiply", "divide", "floor_divide", "mod", "power",
        "abs", "negative", "sign", "square", "sqrt", "cbrt", "exp", "expm1",
        "log", "log1p", "log2", "log10", "sin", "cos", "tan", "asin", "acos",
        "atan", "atan2", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
        "hypot", "copysign", "floor", "ceil", "trunc", "round", "rint",
        "reciprocal", "degrees", "radians", "minimum", "maximum",
        "isnan", "isinf", "isfinite", "clip",
        "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
        "logical_and", "logical_or", "logical_xor", "logical_not",
        "cumsum", "cumprod", "cummax", "cummin", "diff",
        "sort", "argsort", "searchsorted", "unique", "bincount", "histogram",
        "take", "put", "compress", "nonzero",
        "shares_memory", "array_equal", "allclose", "isclose",
        // linear algebra
        "matmul", "dot", "outer", "trace", "solve", "inv", "det", "slogdet",
        "cholesky", "qr", "lstsq", "eigh", "svd", "matrix_rank", "cond",
        "pinv", "norm"
    };
    std::string n = builtinName;
    if (n.rfind("nd_", 0) != 0) return;
    const std::string shortName = n.substr(3);
    if (kMethods.count(shortName)) methodTable()[shortName] = fn;
}

inline bool dispatchMethodImpl(const Value& obj, const std::string& name, Value& out) {
    if (!isArray(obj)) return false;
    auto it = methodTable().find(name);
    if (it == methodTable().end()) return false;
    // Bind the receiver as the first argument. Copying the handle Value is two
    // shared_ptr increments, not a data copy -- the array itself is untouched.
    NativeFn fn = it->second;
    Value self = obj;
    out = Value(NativeFn([fn, self](std::vector<Value> args) -> Value {
        std::vector<Value> full;
        full.reserve(args.size() + 1);
        full.push_back(self);
        for (auto& a : args) full.push_back(std::move(a));
        return fn(std::move(full));
    }));
    return true;
}

}   // namespace numba
