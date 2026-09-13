/**
 * numba — builtin registration and the Bantu-facing surface.
 *
 * The ONLY translation unit that includes ndarray_native.hpp, so it is the one
 * the build scripts compile at -O3 while everything else stays at -O2 (N11).
 *
 * Phase 1 scope: the array itself, buffers, zero-copy views, creation,
 * introspection, shape operations and element access. Broadcasting and the
 * ufuncs arrive in Phase 2; see numba-suite/ROADMAP.md.
 */

#include "ndarray_native.hpp"
#include "ndarray_api.hpp"

namespace numba {

// ═════════════════════════════════════════════════════════════════════════════
// Argument helpers
//
// Every one of these raises a std::exception on bad input, which registerBuiltins
// turns into a catchable Bantu error naming the builtin. Nothing here may kill
// the process: the sua.udp review is the precedent, where two of four defects
// were fatal because one unvalidated argument reached an allocation.
// ═════════════════════════════════════════════════════════════════════════════

static void needArgs(const std::vector<Value>& a, size_t n, const char* usage) {
    if (a.size() < n) throw std::runtime_error(std::string("usage: ") + usage);
}

// One non-negative, integral extent.
static size_t extentFrom(const Value& v, const char* what) {
    if (!v.isNumber()) {
        throw std::runtime_error(std::string(what) + ": a dimension must be a number (got " +
                                 v.toString() + ")");
    }
    const double d = v.numberVal;
    if (std::isnan(d) || std::isinf(d)) {
        throw std::runtime_error(std::string(what) + ": a dimension must be finite (got " +
                                 v.toString() + ")");
    }
    if (d < 0) {
        throw std::runtime_error(std::string(what) + ": a dimension cannot be negative (got " +
                                 v.toString() + ")");
    }
    if (d != std::floor(d)) {
        throw std::runtime_error(std::string(what) + ": a dimension must be a whole number (got " +
                                 v.toString() + ")");
    }
    if (d > 9007199254740992.0) {   // 2^53: beyond this a double is not exact
        throw std::runtime_error(std::string(what) + ": dimension " + v.toString() +
                                 " is too large to be represented exactly");
    }
    return (size_t)d;
}

// A shape: either a single number (1-D) or a list of numbers. An empty list is
// a 0-d scalar, which is a legitimate shape, not an error.
static std::vector<size_t> shapeFrom(const Value& v, const char* what) {
    std::vector<size_t> shape;
    if (v.isNumber()) { shape.push_back(extentFrom(v, what)); return shape; }
    if (!v.isList()) {
        throw std::runtime_error(std::string(what) +
            ": shape must be a number or a list of numbers (got " + v.toString() + ")");
    }
    if (v.listVal.size() > 32) {
        throw std::runtime_error(std::string(what) + ": " + std::to_string(v.listVal.size()) +
                                 " dimensions is more than the 32 supported");
    }
    for (const Value& e : v.listVal) shape.push_back(extentFrom(e, what));
    // Validated here so every later size() can multiply without checking.
    shapeProduct(shape, what);
    return shape;
}

static DType dtypeArg(const std::vector<Value>& a, size_t i, DType dflt) {
    if (a.size() <= i || a[i].isNull()) return dflt;
    if (!a[i].isString()) {
        throw std::runtime_error("dtype must be a string: \"f64\", \"i64\" or \"bool\" (got " +
                                 a[i].toString() + ")");
    }
    return dtypeFromName(a[i].stringVal);
}

static double numArg(const std::vector<Value>& a, size_t i, double dflt, const char* what) {
    if (a.size() <= i || a[i].isNull()) return dflt;
    if (!a[i].isNumber()) {
        throw std::runtime_error(std::string(what) + ": expected a number (got " +
                                 a[i].toString() + ")");
    }
    return a[i].numberVal;
}

// ═════════════════════════════════════════════════════════════════════════════
// Nested Bantu list -> array
// ═════════════════════════════════════════════════════════════════════════════

// Shape is taken from the first element at each level, then every sibling is
// checked against it -- a ragged list is an error, with the offending depth and
// both lengths named.
static void inferShape(const Value& v, std::vector<size_t>& shape, size_t depth) {
    if (!v.isList()) return;
    if (depth > 32) throw std::runtime_error("nd: nesting deeper than 32 levels");
    shape.push_back(v.listVal.size());
    if (!v.listVal.empty()) inferShape(v.listVal[0], shape, depth + 1);
}

static void checkRectangular(const Value& v, const std::vector<size_t>& shape, size_t depth) {
    if (depth >= shape.size()) {
        if (v.isList()) {
            throw std::runtime_error("nd: the nesting is ragged -- a list appears at depth " +
                                     std::to_string(depth) + " where a scalar was expected");
        }
        return;
    }
    if (!v.isList()) {
        throw std::runtime_error("nd: the nesting is ragged -- a scalar appears at depth " +
                                 std::to_string(depth) + " where a list of " +
                                 std::to_string(shape[depth]) + " was expected");
    }
    if (v.listVal.size() != shape[depth]) {
        throw std::runtime_error("nd: the nesting is ragged -- depth " + std::to_string(depth) +
                                 " has a list of " + std::to_string(v.listVal.size()) +
                                 " where " + std::to_string(shape[depth]) + " was expected");
    }
    for (const Value& e : v.listVal) checkRectangular(e, shape, depth + 1);
}

// bool if every element is a bool; i64 if every element is a whole number;
// f64 otherwise. Matches what NumPy infers for the same literals.
static void inferDType(const Value& v, bool& allBool, bool& allInt, bool& any) {
    if (v.isList()) { for (const Value& e : v.listVal) inferDType(e, allBool, allInt, any); return; }
    any = true;
    if (v.isBool()) return;
    allBool = false;
    if (!v.isNumber()) {
        throw std::runtime_error("nd: elements must be numbers or booleans (got " +
                                 v.toString() + ")");
    }
    const double d = v.numberVal;
    if (std::isnan(d) || std::isinf(d) || d != std::floor(d)) allInt = false;
}

static void fillNested(const Value& v, NdArray& out, size_t& k) {
    if (v.isList()) { for (const Value& e : v.listVal) fillNested(e, out, k); return; }
    if (v.isBool()) setFromDouble(out, k++, v.boolVal ? 1.0 : 0.0);
    else            setFromDouble(out, k++, v.numberVal);
}

// ═════════════════════════════════════════════════════════════════════════════
// repr
// ═════════════════════════════════════════════════════════════════════════════

static void reprScalar(std::ostringstream& o, const NdArray& a, size_t flat) {
    Value v = getAsValue(a, flat);
    o << v.toString();
}

static void reprRecurse(std::ostringstream& o, const NdArray& a,
                        size_t depth, ptrdiff_t off, bool summarize) {
    const size_t kEdge = 3;
    if (depth == a.ndim()) { reprScalar(o, a, (size_t)off); return; }

    const size_t n = a.shape[depth];
    const ptrdiff_t st = a.strides[depth];
    o << "[";
    if (!summarize || n <= 2 * kEdge + 1) {
        for (size_t i = 0; i < n; i++) {
            if (i) o << ", ";
            reprRecurse(o, a, depth + 1, off + (ptrdiff_t)i * st, summarize);
        }
    } else {
        for (size_t i = 0; i < kEdge; i++) {
            if (i) o << ", ";
            reprRecurse(o, a, depth + 1, off + (ptrdiff_t)i * st, summarize);
        }
        o << ", ...";
        for (size_t i = n - kEdge; i < n; i++) {
            o << ", ";
            reprRecurse(o, a, depth + 1, off + (ptrdiff_t)i * st, summarize);
        }
    }
    o << "]";
}

std::string reprArray(const NdArray& a) {
    std::ostringstream o;
    if (!a.buf) return "<ndarray>";
    // NumPy's threshold: print everything up to 1000 elements, summarize above.
    const bool summarize = a.size() > 1000;
    if (a.ndim() == 0) {
        reprScalar(o, a, a.offset);
    } else {
        reprRecurse(o, a, 0, (ptrdiff_t)a.offset, summarize);
    }
    o << "  (shape=" << shapeStr(a.shape) << ", dtype=" << dtypeName(a.dtype) << ")";
    return o.str();
}

// ═════════════════════════════════════════════════════════════════════════════
// Registration
// ═════════════════════════════════════════════════════════════════════════════

void registerBuiltins(const DefineFn& define) {
    // print($a) renders the array rather than "<ndarray>".
    registerHandleRepr(NDARRAY_TAG, &reprArrayHandle);

    // ── the allocation ceiling ───────────────────────────────────────────────
    define("nd_max_bytes", [](std::vector<Value> a) -> Value {
        if (!a.empty() && !a[0].isNull()) {
            const double d = a[0].numberVal;
            if (!a[0].isNumber() || d < 1024 || std::isnan(d) || std::isinf(d)) {
                throw std::runtime_error("nd_max_bytes: expected a byte count of at least 1024");
            }
            maxBytesRef() = (size_t)d;
        }
        return Value((double)maxBytesRef());
    });

    // ── creation ─────────────────────────────────────────────────────────────

    // nd(nestedList, dtype?) — shape inferred from the nesting depth.
    define("nd", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd(list, dtype?)");
        if (a[0].isNumber() || a[0].isBool()) {          // a 0-d scalar array
            DType dt = dtypeArg(a, 1, a[0].isBool() ? DType::BOOL : DType::F64);
            auto out = makeArray({}, dt, "nd");
            setFromDouble(*out, 0, a[0].isBool() ? (a[0].boolVal ? 1.0 : 0.0) : a[0].numberVal);
            return wrap(out);
        }
        if (!a[0].isList()) {
            throw std::runtime_error("nd: expected a list or a number (got " + a[0].toString() + ")");
        }
        std::vector<size_t> shape;
        inferShape(a[0], shape, 0);
        checkRectangular(a[0], shape, 0);

        bool allBool = true, allInt = true, any = false;
        inferDType(a[0], allBool, allInt, any);
        DType inferred = DType::F64;
        if (any) inferred = allBool ? DType::BOOL : (allInt ? DType::I64 : DType::F64);
        DType dt = dtypeArg(a, 1, inferred);

        auto out = makeArray(shape, dt, "nd");
        size_t k = 0;
        fillNested(a[0], *out, k);
        return wrap(out);
    });

    define("nd_zeros", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_zeros(shape, dtype?)");
        return wrap(makeArray(shapeFrom(a[0], "nd_zeros"), dtypeArg(a, 1, DType::F64), "nd_zeros"));
    });

    // Buffers arrive zero-filled, so "empty" is zeros. Kept as a separate name
    // because it documents intent, and because a later free-list would make it
    // genuinely uninitialized.
    define("nd_empty", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_empty(shape, dtype?)");
        return wrap(makeArray(shapeFrom(a[0], "nd_empty"), dtypeArg(a, 1, DType::F64), "nd_empty"));
    });

    auto filled = [](const std::vector<Value>& a, double v, const char* what) -> Value {
        auto out = makeArray(shapeFrom(a[0], what), dtypeArg(a, 1, DType::F64), what);
        const size_t n = out->size();
        for (size_t i = 0; i < n; i++) setFromDouble(*out, i, v);
        return wrap(out);
    };
    define("nd_ones", [filled](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_ones(shape, dtype?)");
        return filled(a, 1.0, "nd_ones");
    });
    define("nd_full", [](std::vector<Value> a) -> Value {
        needArgs(a, 2, "nd_full(shape, value, dtype?)");
        if (!a[1].isNumber() && !a[1].isBool()) {
            throw std::runtime_error("nd_full: the fill value must be a number or a boolean (got " +
                                     a[1].toString() + ")");
        }
        const double v = a[1].isBool() ? (a[1].boolVal ? 1.0 : 0.0) : a[1].numberVal;
        DType dflt = a[1].isBool() ? DType::BOOL : DType::F64;
        auto out = makeArray(shapeFrom(a[0], "nd_full"), dtypeArg(a, 2, dflt), "nd_full");
        const size_t n = out->size();
        for (size_t i = 0; i < n; i++) setFromDouble(*out, i, v);
        return wrap(out);
    });

    auto likeOf = [](const std::vector<Value>& a, double v, const char* what) -> Value {
        auto src = asArray(a[0]);
        auto out = makeArray(src->shape, dtypeArg(a, 1, src->dtype), what);
        const size_t n = out->size();
        for (size_t i = 0; i < n; i++) setFromDouble(*out, i, v);
        return wrap(out);
    };
    define("nd_zeros_like", [likeOf](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_zeros_like(array, dtype?)");
        return likeOf(a, 0.0, "nd_zeros_like");
    });
    define("nd_ones_like", [likeOf](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_ones_like(array, dtype?)");
        return likeOf(a, 1.0, "nd_ones_like");
    });
    define("nd_full_like", [](std::vector<Value> a) -> Value {
        needArgs(a, 2, "nd_full_like(array, value, dtype?)");
        auto src = asArray(a[0]);
        if (!a[1].isNumber() && !a[1].isBool()) {
            throw std::runtime_error("nd_full_like: the fill value must be a number or a boolean");
        }
        const double v = a[1].isBool() ? (a[1].boolVal ? 1.0 : 0.0) : a[1].numberVal;
        auto out = makeArray(src->shape, dtypeArg(a, 2, src->dtype), "nd_full_like");
        const size_t n = out->size();
        for (size_t i = 0; i < n; i++) setFromDouble(*out, i, v);
        return wrap(out);
    });

    // nd_arange(stop) | nd_arange(start, stop) | nd_arange(start, stop, step)
    // Arity dispatch, because Bantu builtins cannot have default parameters --
    // that is what the numba.b facade exists to paper over for everything else.
    define("nd_arange", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_arange(stop) | nd_arange(start, stop, step?)");
        double start = 0, stop = 0, step = 1;
        if (a.size() == 1 || a[1].isNull()) { stop = numArg(a, 0, 0, "nd_arange"); }
        else {
            start = numArg(a, 0, 0, "nd_arange");
            stop  = numArg(a, 1, 0, "nd_arange");
            step  = numArg(a, 2, 1, "nd_arange");
        }
        if (step == 0) throw std::runtime_error("nd_arange: step cannot be zero");
        if (std::isnan(start) || std::isnan(stop) || std::isnan(step) ||
            std::isinf(start) || std::isinf(stop) || std::isinf(step)) {
            throw std::runtime_error("nd_arange: start, stop and step must be finite");
        }
        double raw = std::ceil((stop - start) / step);
        if (raw < 0) raw = 0;
        if (raw > 9007199254740992.0) {
            throw std::runtime_error("nd_arange: that range would produce more elements than "
                                     "can be counted exactly");
        }
        const size_t n = (size_t)raw;
        const bool integral = (start == std::floor(start) && step == std::floor(step));
        auto out = makeArray({n}, dtypeArg(a, 3, integral ? DType::I64 : DType::F64), "nd_arange");
        for (size_t i = 0; i < n; i++) setFromDouble(*out, i, start + (double)i * step);
        return wrap(out);
    });

    // nd_linspace(start, stop, num?, endpoint?) — num defaults to 50, as NumPy.
    define("nd_linspace", [](std::vector<Value> a) -> Value {
        needArgs(a, 2, "nd_linspace(start, stop, num?, endpoint?)");
        const double start = numArg(a, 0, 0, "nd_linspace");
        const double stop  = numArg(a, 1, 0, "nd_linspace");
        const double numD  = numArg(a, 2, 50, "nd_linspace");
        if (numD < 0 || numD != std::floor(numD) || std::isnan(numD) || std::isinf(numD)) {
            throw std::runtime_error("nd_linspace: num must be a whole number of at least 0 (got " +
                                     std::to_string(numD) + ")");
        }
        const bool endpoint = (a.size() <= 3 || a[3].isNull()) ? true : a[3].isTruthy();
        const size_t n = (size_t)numD;
        auto out = makeArray({n}, DType::F64, "nd_linspace");
        if (n == 1) { setFromDouble(*out, 0, start); return wrap(out); }
        const double div = endpoint ? (double)(n - 1) : (double)n;
        for (size_t i = 0; i < n; i++) {
            setFromDouble(*out, i, start + (stop - start) * ((double)i / div));
        }
        // Land exactly on `stop` rather than within one ulp of it.
        if (endpoint && n > 0) setFromDouble(*out, n - 1, stop);
        return wrap(out);
    });

    // nd_eye(n, m?, k?) — ones on the k-th diagonal.
    define("nd_eye", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_eye(n, m?, k?)");
        const size_t n = extentFrom(a[0], "nd_eye");
        const size_t m = (a.size() > 1 && !a[1].isNull()) ? extentFrom(a[1], "nd_eye") : n;
        const ptrdiff_t k = (ptrdiff_t)std::llround(numArg(a, 2, 0, "nd_eye"));
        auto out = makeArray({n, m}, dtypeArg(a, 3, DType::F64), "nd_eye");
        for (size_t i = 0; i < n; i++) {
            const ptrdiff_t j = (ptrdiff_t)i + k;
            if (j >= 0 && (size_t)j < m) setFromDouble(*out, i * m + (size_t)j, 1.0);
        }
        return wrap(out);
    });
    define("nd_identity", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_identity(n, dtype?)");
        const size_t n = extentFrom(a[0], "nd_identity");
        auto out = makeArray({n, n}, dtypeArg(a, 1, DType::F64), "nd_identity");
        for (size_t i = 0; i < n; i++) setFromDouble(*out, i * n + i, 1.0);
        return wrap(out);
    });

    // ── random ───────────────────────────────────────────────────────────────
    define("nd_seed", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_seed(n)");
        if (!a[0].isNumber()) throw std::runtime_error("nd_seed: expected a number");
        rng().seed((uint64_t)(int64_t)std::llround(a[0].numberVal));
        return Value();
    });
    define("nd_random_uniform", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_random_uniform(shape, lo?, hi?)");
        const double lo = numArg(a, 1, 0.0, "nd_random_uniform");
        const double hi = numArg(a, 2, 1.0, "nd_random_uniform");
        auto out = makeArray(shapeFrom(a[0], "nd_random_uniform"), DType::F64, "nd_random_uniform");
        const size_t n = out->size();
        for (size_t i = 0; i < n; i++) setFromDouble(*out, i, lo + (hi - lo) * rng().uniform());
        return wrap(out);
    });
    define("nd_random_normal", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_random_normal(shape, mu?, sigma?)");
        const double mu    = numArg(a, 1, 0.0, "nd_random_normal");
        const double sigma = numArg(a, 2, 1.0, "nd_random_normal");
        auto out = makeArray(shapeFrom(a[0], "nd_random_normal"), DType::F64, "nd_random_normal");
        const size_t n = out->size();
        // Box-Muller: two normals per pair of uniforms. Adequate here; a
        // Ziggurat is a Phase 7 concern and only if profiling asks for it.
        for (size_t i = 0; i < n; i += 2) {
            double u1 = rng().uniform();
            const double u2 = rng().uniform();
            if (u1 < 1e-300) u1 = 1e-300;                 // log(0) guard
            const double r = std::sqrt(-2.0 * std::log(u1));
            const double t = 6.283185307179586 * u2;
            setFromDouble(*out, i, mu + sigma * r * std::cos(t));
            if (i + 1 < n) setFromDouble(*out, i + 1, mu + sigma * r * std::sin(t));
        }
        return wrap(out);
    });
    define("nd_random_int", [](std::vector<Value> a) -> Value {
        needArgs(a, 3, "nd_random_int(shape, lo, hi)");
        const double lo = numArg(a, 1, 0, "nd_random_int");
        const double hi = numArg(a, 2, 0, "nd_random_int");
        if (hi <= lo) throw std::runtime_error("nd_random_int: hi must be greater than lo");
        const uint64_t span = (uint64_t)(hi - lo);          // half-open [lo, hi)
        auto out = makeArray(shapeFrom(a[0], "nd_random_int"), DType::I64, "nd_random_int");
        const size_t n = out->size();
        for (size_t i = 0; i < n; i++) {
            setFromDouble(*out, i, lo + (double)(rng().next() % span));
        }
        return wrap(out);
    });

    // ── introspection ────────────────────────────────────────────────────────
    define("nd_shape", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_shape(array)");
        auto x = asArray(a[0]);
        std::vector<Value> out;
        for (size_t d : x->shape) out.push_back(Value((double)d));
        return Value(std::move(out));
    });
    define("nd_ndim", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_ndim(array)"); return Value((double)asArray(a[0])->ndim()); });
    define("nd_size", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_size(array)"); return Value((double)asArray(a[0])->size()); });
    define("nd_dtype", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_dtype(array)");
        return Value(std::string(dtypeName(asArray(a[0])->dtype))); });
    define("nd_itemsize", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_itemsize(array)");
        return Value((double)itemsize(asArray(a[0])->dtype)); });
    define("nd_nbytes", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_nbytes(array)");
        auto x = asArray(a[0]);
        return Value((double)(x->size() * itemsize(x->dtype))); });
    define("nd_strides", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_strides(array)");
        auto x = asArray(a[0]);
        std::vector<Value> out;
        for (ptrdiff_t s : x->strides) out.push_back(Value((double)s));
        return Value(std::move(out));
    });
    define("nd_writable", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_writable(array)"); return Value(asArray(a[0])->writable); });
    define("nd_is_contiguous", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_is_contiguous(array, order?)");
        auto x = asArray(a[0]);
        const std::string order = (a.size() > 1 && a[1].isString()) ? a[1].stringVal : "C";
        if (order == "F" || order == "f") return Value(x->isFContig());
        return Value(x->isCContig());
    });
    // "Does this array share memory with another one?" -- the question a user
    // actually has, because it decides whether writing through this handle is
    // visible elsewhere. Covering the whole buffer is NOT enough to answer it:
    // a transposed view covers exactly the same bytes as its base, so
    // ownsAll() alone would call it an independent array. The buffer's
    // reference count is what makes sharing observable.
    define("nd_is_view", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_is_view(array)");
        auto x = asArray(a[0]);
        return Value(x->buf.use_count() > 1 || !x->ownsAll());
    });
    // The stable identity of the underlying buffer. This is how a test PROVES
    // two arrays share memory rather than merely comparing equal.
    define("nd_base_id", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_base_id(array)");
        auto x = asArray(a[0]);
        return Value((double)(uintptr_t)x->buf->data);
    });

    // ── element access ───────────────────────────────────────────────────────
    auto indexList = [](const Value& v, const NdArray& x, const char* what)
                     -> std::vector<size_t> {
        std::vector<size_t> idx;
        if (v.isNumber()) idx.push_back(0);       // placeholder, filled below
        std::vector<Value> raw;
        if (v.isList())        raw = v.listVal;
        else if (v.isNumber()) raw = { v };
        else throw std::runtime_error(std::string(what) +
                 ": the index must be a number or a list of numbers (got " + v.toString() + ")");
        if (raw.size() != x.ndim()) {
            throw std::runtime_error(std::string(what) + ": got " + std::to_string(raw.size()) +
                " index" + (raw.size() == 1 ? "" : "es") + " for a " +
                std::to_string(x.ndim()) + "-dimensional array of shape " + shapeStr(x.shape));
        }
        idx.clear();
        for (size_t d = 0; d < raw.size(); d++) {
            if (!raw[d].isNumber()) {
                throw std::runtime_error(std::string(what) + ": index " + std::to_string(d) +
                                         " is not a number (got " + raw[d].toString() + ")");
            }
            idx.push_back(normIndex(raw[d].numberVal, x.shape[d], d));
        }
        return idx;
    };

    define("nd_get", [indexList](std::vector<Value> a) -> Value {
        needArgs(a, 2, "nd_get(array, index)");
        auto x = asArray(a[0]);
        if (x->ndim() == 0) return getAsValue(*x, x->offset);
        return getAsValue(*x, flatOffset(*x, indexList(a[1], *x, "nd_get")));
    });

    define("nd_set", [indexList](std::vector<Value> a) -> Value {
        needArgs(a, 3, "nd_set(array, index, value)");
        auto x = asArray(a[0]);
        if (!x->writable) {
            throw std::runtime_error("nd_set: this array is read-only. A broadcast view has a "
                                     "stride of 0 on its stretched axes, so writing through it "
                                     "would silently hit the same element many times -- copy it "
                                     "first with nd_copy()");
        }
        if (!a[2].isNumber() && !a[2].isBool()) {
            throw std::runtime_error("nd_set: the value must be a number or a boolean (got " +
                                     a[2].toString() + ")");
        }
        const double v = a[2].isBool() ? (a[2].boolVal ? 1.0 : 0.0) : a[2].numberVal;
        const size_t flat = (x->ndim() == 0) ? x->offset
                                             : flatOffset(*x, indexList(a[1], *x, "nd_set"));
        setFromDouble(*x, flat, v);
        return a[0];
    });

    define("nd_to_list", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_to_list(array)");
        auto x = asArray(a[0]);
        // Recursive so the nesting mirrors what nd() accepts, making a
        // round-trip through Bantu lossless in shape as well as value.
        std::function<Value(size_t, ptrdiff_t)> build = [&](size_t depth, ptrdiff_t off) -> Value {
            if (depth == x->ndim()) return getAsValue(*x, (size_t)off);
            std::vector<Value> out;
            out.reserve(x->shape[depth]);
            for (size_t i = 0; i < x->shape[depth]; i++) {
                out.push_back(build(depth + 1, off + (ptrdiff_t)i * x->strides[depth]));
            }
            return Value(std::move(out));
        };
        return build(0, (ptrdiff_t)x->offset);
    });

    // ── copies and casts ─────────────────────────────────────────────────────
    define("nd_copy", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_copy(array)");
        auto x = asArray(a[0]);
        return wrap(contiguousCopy(x, x->dtype));
    });
    define("nd_astype", [](std::vector<Value> a) -> Value {
        needArgs(a, 2, "nd_astype(array, dtype)");
        auto x = asArray(a[0]);
        return wrap(contiguousCopy(x, dtypeFromName(a[1].toString())));
    });
    // Already contiguous: hand the same array back rather than copying.
    define("nd_ascontiguous", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_ascontiguous(array)");
        auto x = asArray(a[0]);
        if (x->isCContig()) return a[0];
        return wrap(contiguousCopy(x, x->dtype));
    });

    // ── shape operations ─────────────────────────────────────────────────────

    // A view when the strides permit it, a copy otherwise. nd_is_view() tells
    // you which happened, and on a large array that is the difference between
    // zero bytes and tens of megabytes -- so it is worth being able to ask.
    define("nd_reshape", [](std::vector<Value> a) -> Value {
        needArgs(a, 2, "nd_reshape(array, shape)");
        auto x = asArray(a[0]);

        // One -1 may be inferred, as in NumPy.
        std::vector<ptrdiff_t> want;
        if (a[1].isNumber()) want.push_back((ptrdiff_t)std::llround(a[1].numberVal));
        else if (a[1].isList()) {
            for (const Value& e : a[1].listVal) {
                if (!e.isNumber()) throw std::runtime_error("nd_reshape: shape entries must be numbers");
                want.push_back((ptrdiff_t)std::llround(e.numberVal));
            }
        } else throw std::runtime_error("nd_reshape: shape must be a number or a list of numbers");

        ptrdiff_t inferAt = -1;
        size_t known = 1;
        for (size_t d = 0; d < want.size(); d++) {
            if (want[d] == -1) {
                if (inferAt >= 0) throw std::runtime_error("nd_reshape: only one dimension may be -1");
                inferAt = (ptrdiff_t)d;
            } else if (want[d] < 0) {
                throw std::runtime_error("nd_reshape: a dimension cannot be negative (got " +
                                         std::to_string(want[d]) + ")");
            } else {
                known = checkedMul(known, (size_t)want[d], "nd_reshape");
            }
        }
        const size_t total = x->size();
        std::vector<size_t> shape;
        if (inferAt >= 0) {
            if (known == 0 || total % known != 0) {
                throw std::runtime_error("nd_reshape: cannot reshape " + std::to_string(total) +
                                         " elements into " + shapeStr(x->shape) + " -> the "
                                         "requested shape with -1 does not divide evenly");
            }
            for (size_t d = 0; d < want.size(); d++) {
                shape.push_back(d == (size_t)inferAt ? total / known : (size_t)want[d]);
            }
        } else {
            for (ptrdiff_t d : want) shape.push_back((size_t)d);
            if (shapeProduct(shape, "nd_reshape") != total) {
                throw std::runtime_error("nd_reshape: cannot reshape an array of shape " +
                                         shapeStr(x->shape) + " (" + std::to_string(total) +
                                         " elements) into " + shapeStr(shape));
            }
        }
        if (x->isCContig()) return wrap(makeView(x, shape, cStrides(shape), x->offset));
        // Non-contiguous: materialize, then the reshape is trivially a view of
        // the fresh buffer.
        auto c = contiguousCopy(x, x->dtype);
        c->shape   = shape;
        c->strides = cStrides(shape);
        return wrap(c);
    });

    define("nd_transpose", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_transpose(array, axes?)");
        auto x = asArray(a[0]);
        const size_t nd = x->ndim();
        std::vector<size_t> perm;
        if (a.size() > 1 && !a[1].isNull()) {
            if (!a[1].isList()) throw std::runtime_error("nd_transpose: axes must be a list");
            if (a[1].listVal.size() != nd) {
                throw std::runtime_error("nd_transpose: axes has " +
                    std::to_string(a[1].listVal.size()) + " entries for a " +
                    std::to_string(nd) + "-dimensional array");
            }
            std::vector<bool> seen(nd, false);
            for (const Value& e : a[1].listVal) {
                const size_t ax = normAxis(e.numberVal, nd);
                if (seen[ax]) throw std::runtime_error("nd_transpose: axis " +
                                                       std::to_string(ax) + " is repeated");
                seen[ax] = true;
                perm.push_back(ax);
            }
        } else {
            for (ptrdiff_t d = (ptrdiff_t)nd - 1; d >= 0; d--) perm.push_back((size_t)d);
        }
        std::vector<size_t>    shape(nd);
        std::vector<ptrdiff_t> st(nd);
        for (size_t d = 0; d < nd; d++) { shape[d] = x->shape[perm[d]]; st[d] = x->strides[perm[d]]; }
        return wrap(makeView(x, shape, st, x->offset));
    });
    define("nd_T", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_T(array)");
        auto x = asArray(a[0]);
        const size_t nd = x->ndim();
        std::vector<size_t>    shape(nd);
        std::vector<ptrdiff_t> st(nd);
        for (size_t d = 0; d < nd; d++) { shape[d] = x->shape[nd - 1 - d]; st[d] = x->strides[nd - 1 - d]; }
        return wrap(makeView(x, shape, st, x->offset));
    });

    define("nd_ravel", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_ravel(array)");
        auto x = asArray(a[0]);
        if (x->isCContig()) {
            return wrap(makeView(x, {x->size()}, {1}, x->offset));
        }
        auto c = contiguousCopy(x, x->dtype);
        c->shape = {c->size()}; c->strides = {1};
        return wrap(c);
    });
    define("nd_flatten", [](std::vector<Value> a) -> Value {   // always a copy
        needArgs(a, 1, "nd_flatten(array)");
        auto x = asArray(a[0]);
        auto c = contiguousCopy(x, x->dtype);
        c->shape = {c->size()}; c->strides = {1};
        return wrap(c);
    });

    define("nd_swapaxes", [](std::vector<Value> a) -> Value {
        needArgs(a, 3, "nd_swapaxes(array, i, j)");
        auto x = asArray(a[0]);
        const size_t i = normAxis(a[1].numberVal, x->ndim());
        const size_t j = normAxis(a[2].numberVal, x->ndim());
        auto shape = x->shape; auto st = x->strides;
        std::swap(shape[i], shape[j]); std::swap(st[i], st[j]);
        return wrap(makeView(x, shape, st, x->offset));
    });

    define("nd_moveaxis", [](std::vector<Value> a) -> Value {
        needArgs(a, 3, "nd_moveaxis(array, src, dst)");
        auto x = asArray(a[0]);
        const size_t nd = x->ndim();
        const size_t src = normAxis(a[1].numberVal, nd);
        const size_t dst = normAxis(a[2].numberVal, nd);
        std::vector<size_t> order;
        for (size_t d = 0; d < nd; d++) if (d != src) order.push_back(d);
        order.insert(order.begin() + (ptrdiff_t)dst, src);
        std::vector<size_t>    shape(nd);
        std::vector<ptrdiff_t> st(nd);
        for (size_t d = 0; d < nd; d++) { shape[d] = x->shape[order[d]]; st[d] = x->strides[order[d]]; }
        return wrap(makeView(x, shape, st, x->offset));
    });

    define("nd_expand_dims", [](std::vector<Value> a) -> Value {
        needArgs(a, 2, "nd_expand_dims(array, axis)");
        auto x = asArray(a[0]);
        const size_t nd = x->ndim();
        ptrdiff_t ax = (ptrdiff_t)std::llround(a[1].numberVal);
        if (ax < 0) ax += (ptrdiff_t)nd + 1;
        if (ax < 0 || (size_t)ax > nd) {
            throw std::runtime_error("nd_expand_dims: axis " + a[1].toString() +
                " is out of range for a " + std::to_string(nd) + "-dimensional array");
        }
        auto shape = x->shape; auto st = x->strides;
        // Stride of the new length-1 axis is arbitrary; use the extent it would
        // have had so that a later reshape sees plausible strides.
        const ptrdiff_t newStride = ((size_t)ax < nd) ? st[(size_t)ax] * (ptrdiff_t)shape[(size_t)ax] : 1;
        shape.insert(shape.begin() + ax, 1);
        st.insert(st.begin() + ax, newStride);
        return wrap(makeView(x, shape, st, x->offset));
    });

    define("nd_squeeze", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_squeeze(array, axis?)");
        auto x = asArray(a[0]);
        std::vector<size_t>    shape;
        std::vector<ptrdiff_t> st;
        if (a.size() > 1 && !a[1].isNull()) {
            const size_t ax = normAxis(a[1].numberVal, x->ndim());
            if (x->shape[ax] != 1) {
                throw std::runtime_error("nd_squeeze: axis " + std::to_string(ax) + " has " +
                    std::to_string(x->shape[ax]) + " elements, so it cannot be squeezed");
            }
            for (size_t d = 0; d < x->ndim(); d++) {
                if (d == ax) continue;
                shape.push_back(x->shape[d]); st.push_back(x->strides[d]);
            }
        } else {
            for (size_t d = 0; d < x->ndim(); d++) {
                if (x->shape[d] == 1) continue;
                shape.push_back(x->shape[d]); st.push_back(x->strides[d]);
            }
        }
        return wrap(makeView(x, shape, st, x->offset));
    });

    // nd_flip(array, axis?) — a view with a NEGATIVE stride, which is exactly
    // why strides are signed.
    define("nd_flip", [](std::vector<Value> a) -> Value {
        needArgs(a, 1, "nd_flip(array, axis?)");
        auto x = asArray(a[0]);
        auto shape = x->shape; auto st = x->strides;
        ptrdiff_t off = (ptrdiff_t)x->offset;
        auto flipOne = [&](size_t d) {
            if (shape[d] == 0) return;
            off += st[d] * (ptrdiff_t)(shape[d] - 1);
            st[d] = -st[d];
        };
        if (a.size() > 1 && !a[1].isNull()) flipOne(normAxis(a[1].numberVal, x->ndim()));
        else for (size_t d = 0; d < x->ndim(); d++) flipOne(d);
        return wrap(makeView(x, shape, st, (size_t)off));
    });

    // nd_broadcast_to(array, shape) — a READ-ONLY view. A stretched axis gets a
    // stride of 0, so stepping along it re-reads the same element; that zero
    // stride IS broadcasting, and no data is copied. Writing through it would
    // hit one element repeatedly, so the result is not writable (N15).
    define("nd_broadcast_to", [](std::vector<Value> a) -> Value {
        needArgs(a, 2, "nd_broadcast_to(array, shape)");
        auto x = asArray(a[0]);
        const std::vector<size_t> to = shapeFrom(a[1], "nd_broadcast_to");
        if (to.size() < x->ndim()) {
            throw std::runtime_error("nd_broadcast_to: cannot broadcast shape " +
                shapeStr(x->shape) + " to the smaller-rank " + shapeStr(to));
        }
        const size_t pad = to.size() - x->ndim();
        std::vector<ptrdiff_t> st(to.size(), 0);
        for (size_t i = 0; i < x->ndim(); i++) {
            const size_t from = x->shape[i];
            const size_t want = to[pad + i];
            if (from == want)      st[pad + i] = x->strides[i];
            else if (from == 1)    st[pad + i] = 0;          // stretched
            else {
                throw std::runtime_error("nd_broadcast_to: cannot broadcast shape " +
                    shapeStr(x->shape) + " to " + shapeStr(to) + " (axis " +
                    std::to_string(pad + i) + ": " + std::to_string(from) + " vs " +
                    std::to_string(want) + ")");
            }
        }
        auto v = makeView(x, to, st, x->offset);
        v->writable = false;
        return wrap(v);
    });

    // nd_slice(array, specs) — one [start, stop, step] (or null for the whole
    // axis) per dimension. Always a view.
    define("nd_slice", [](std::vector<Value> a) -> Value {
        needArgs(a, 2, "nd_slice(array, [[start,stop,step], ...])");
        auto x = asArray(a[0]);
        if (!a[1].isList()) throw std::runtime_error("nd_slice: expected a list of specs");
        const std::vector<Value>& specs = a[1].listVal;
        if (specs.size() > x->ndim()) {
            throw std::runtime_error("nd_slice: " + std::to_string(specs.size()) +
                " specs for a " + std::to_string(x->ndim()) + "-dimensional array");
        }
        std::vector<size_t>    shape = x->shape;
        std::vector<ptrdiff_t> st    = x->strides;
        ptrdiff_t off = (ptrdiff_t)x->offset;

        for (size_t d = 0; d < specs.size(); d++) {
            if (specs[d].isNull()) continue;                 // the whole axis
            if (!specs[d].isList()) {
                throw std::runtime_error("nd_slice: spec " + std::to_string(d) +
                    " must be null or a list [start, stop, step]");
            }
            const std::vector<Value>& s = specs[d].listVal;
            const ptrdiff_t extent = (ptrdiff_t)x->shape[d];
            ptrdiff_t step = (s.size() > 2 && !s[2].isNull()) ? (ptrdiff_t)std::llround(s[2].numberVal) : 1;
            if (step == 0) throw std::runtime_error("nd_slice: step cannot be zero on axis " +
                                                    std::to_string(d));
            ptrdiff_t start, stop;
            if (step > 0) {
                start = (s.size() > 0 && !s[0].isNull()) ? (ptrdiff_t)std::llround(s[0].numberVal) : 0;
                stop  = (s.size() > 1 && !s[1].isNull()) ? (ptrdiff_t)std::llround(s[1].numberVal) : extent;
            } else {
                start = (s.size() > 0 && !s[0].isNull()) ? (ptrdiff_t)std::llround(s[0].numberVal) : extent - 1;
                stop  = (s.size() > 1 && !s[1].isNull()) ? (ptrdiff_t)std::llround(s[1].numberVal) : -1 - extent;
            }
            if (start < 0) start += extent;
            if (stop < 0 && !(step < 0 && stop == -1 - extent)) stop += extent;
            if (step < 0 && stop == -1 - extent) stop = -1;          // "all the way down"
            start = std::max<ptrdiff_t>(0, std::min(start, step > 0 ? extent : extent - 1));

            ptrdiff_t count;
            if (step > 0) {
                stop  = std::min(stop, extent);
                count = (stop > start) ? (stop - start + step - 1) / step : 0;
            } else {
                stop  = std::max<ptrdiff_t>(stop, -1);
                count = (start > stop) ? (start - stop + (-step) - 1) / (-step) : 0;
            }
            off += start * st[d];
            shape[d] = (size_t)count;
            st[d]    = st[d] * step;
        }
        return wrap(makeView(x, shape, st, (size_t)off));
    });
}

} // namespace numba
