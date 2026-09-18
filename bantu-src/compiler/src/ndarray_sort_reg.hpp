#pragma once
/**
 * numba — scans, sorting, search and indexing.
 *
 * Included by ndarray_native.cpp only, after ndarray_api.hpp.
 * docs/numba-architecture.md §4.10-4.11.
 */

#include "ndarray_reduce.hpp"
#include "ndarray_api.hpp"

namespace numba {

Value toArrayValue(const Value& v, const char* what);

// One axis argument, defaulting to the LAST -- which is what sort, cumsum and
// take all mean by "no axis given", and matches NumPy.
inline size_t axisArg(const std::vector<Value>& a, size_t i, size_t ndim, const char* what) {
    if (ndim == 0) {
        throw std::runtime_error(std::string(what) + ": a 0-d array has no axes");
    }
    if (a.size() <= i || a[i].isNull()) return ndim - 1;
    if (!a[i].isNumber()) {
        throw std::runtime_error(std::string(what) + ": axis must be a number (got " +
                                 a[i].toString() + ")");
    }
    return normAxis(a[i].numberVal, ndim);
}

// Walk every 1-D lane along `axis`: body(baseOffset, extent, stride).
template <class Body>
inline void forEachLane(const NdArray& a, size_t axis, Body&& body) {
    std::vector<size_t>    outer;
    std::vector<ptrdiff_t> ostr;
    for (size_t d = 0; d < a.ndim(); d++) {
        if (d == axis) continue;
        outer.push_back(a.shape[d]);
        ostr.push_back(a.strides[d]);
    }
    const size_t    n  = a.shape[axis];
    const ptrdiff_t st = a.strides[axis];
    if (outer.empty()) { body(a.offset, n, st); return; }
    nditer(outer, { ostr }, { a.offset }, [&](const size_t* o) { body(o[0], n, st); });
}

// NaN sorts to the END (docs §4.10). Not principled -- every comparison with
// NaN is false, so some rule must be imposed -- but it is NumPy's, and users
// already know it. Without an explicit rule std::sort's behaviour with NaN is
// unspecified and can corrupt, not merely misorder.
inline bool lessNaNLast(double x, double y) {
    if (std::isnan(x)) return false;
    if (std::isnan(y)) return true;
    return x < y;
}

inline void registerSorting(const DefineFn& define) {

    // ── scans ────────────────────────────────────────────────────────────────
    auto scan = [](const char* what, int mode) {   // 0 sum, 1 prod, 2 max, 3 min
        return [what, mode](std::vector<Value> args) -> Value {
            if (args.empty()) throw std::runtime_error(std::string(what) + "(a, axis?) needs an array");
            ArrayPtr a = asArray(toArrayValue(args[0], what));
            // A flat scan over everything is the natural reading when no axis is
            // given and the array is n-d, which is also what NumPy does.
            ArrayPtr src = a;
            bool flattened = false;
            if ((args.size() < 2 || args[1].isNull()) && a->ndim() != 1) {
                src = contiguousCopy(a, a->dtype);
                std::vector<size_t> flat = { src->size() };
                src = makeView(src, flat, cStrides(flat), 0, what);
                flattened = true;
            }
            const size_t axis = flattened ? 0 : axisArg(args, 1, src->ndim(), what);
            const DType rdt = (mode <= 1 && src->dtype != DType::F64) ? DType::I64
                            : (src->dtype == DType::BOOL ? DType::I64 : src->dtype);
            ArrayPtr out = makeArray(src->shape, rdt, what);
            if (out->size() == 0) return wrap(out);

            // The output is C-contiguous and freshly made, so its lanes line up
            // with the input's in the same odometer order.
            std::vector<size_t> lanePos;
            forEachLane(*out, axis, [&](size_t b, size_t, ptrdiff_t) { lanePos.push_back(b); });
            size_t li = 0;
            const NdArray& S = *src; NdArray& O = *out;
            const ptrdiff_t ost = O.strides[axis];
            forEachLane(S, axis, [&](size_t base, size_t n, ptrdiff_t st) {
                const size_t ob = lanePos[li++];
                double acc = (mode == 1) ? 1.0 : 0.0;
                for (size_t i = 0; i < n; i++) {
                    const double x = getAsDouble(S, (size_t)((ptrdiff_t)base + (ptrdiff_t)i * st));
                    if (i == 0)      acc = x;
                    else if (mode == 0) acc += x;
                    else if (mode == 1) acc *= x;
                    else if (mode == 2) acc = (std::isnan(x) || x > acc) ? x : acc;
                    else                acc = (std::isnan(x) || x < acc) ? x : acc;
                    setFromDouble(O, (size_t)((ptrdiff_t)ob + (ptrdiff_t)i * ost), acc);
                }
            });
            return wrap(out);
        };
    };
    define("nd_cumsum",  scan("nd_cumsum", 0));
    define("nd_cumprod", scan("nd_cumprod", 1));
    define("nd_cummax",  scan("nd_cummax", 2));
    define("nd_cummin",  scan("nd_cummin", 3));

    // nd_diff(a, axis?) — first differences along an axis.
    define("nd_diff", [](std::vector<Value> args) -> Value {
        if (args.empty()) throw std::runtime_error("nd_diff(a, axis?) needs an array");
        ArrayPtr a = asArray(toArrayValue(args[0], "nd_diff"));
        const size_t axis = axisArg(args, 1, a->ndim(), "nd_diff");
        std::vector<size_t> shape = a->shape;
        if (shape[axis] == 0) {
            throw std::runtime_error("nd_diff: axis " + std::to_string(axis) + " is empty");
        }
        shape[axis] -= 1;
        const DType rdt = (a->dtype == DType::F64) ? DType::F64 : DType::I64;
        ArrayPtr out = makeArray(shape, rdt, "nd_diff");
        if (out->size() == 0) return wrap(out);
        std::vector<size_t> lanePos;
        forEachLane(*out, axis, [&](size_t b, size_t, ptrdiff_t) { lanePos.push_back(b); });
        size_t li = 0;
        const NdArray& A = *a; NdArray& O = *out;
        const ptrdiff_t ost = O.strides[axis];
        forEachLane(A, axis, [&](size_t base, size_t n, ptrdiff_t st) {
            const size_t ob = lanePos[li++];
            for (size_t i = 0; i + 1 < n; i++) {
                const double x0 = getAsDouble(A, (size_t)((ptrdiff_t)base + (ptrdiff_t)i * st));
                const double x1 = getAsDouble(A, (size_t)((ptrdiff_t)base + (ptrdiff_t)(i+1) * st));
                setFromDouble(O, (size_t)((ptrdiff_t)ob + (ptrdiff_t)i * ost), x1 - x0);
            }
        });
        return wrap(out);
    });

    // ── sort / argsort ───────────────────────────────────────────────────────
    define("nd_sort", [](std::vector<Value> args) -> Value {
        if (args.empty()) throw std::runtime_error("nd_sort(a, axis?) needs an array");
        ArrayPtr a = asArray(toArrayValue(args[0], "nd_sort"));
        if (a->ndim() == 0) return wrap(contiguousCopy(a, a->dtype));
        const size_t axis = axisArg(args, 1, a->ndim(), "nd_sort");
        ArrayPtr out = contiguousCopy(a, a->dtype);
        NdArray& O = *out;
        std::vector<double> buf;
        forEachLane(O, axis, [&](size_t base, size_t n, ptrdiff_t st) {
            buf.clear(); buf.reserve(n);
            for (size_t i = 0; i < n; i++)
                buf.push_back(getAsDouble(O, (size_t)((ptrdiff_t)base + (ptrdiff_t)i * st)));
            std::sort(buf.begin(), buf.end(), lessNaNLast);
            for (size_t i = 0; i < n; i++)
                setFromDouble(O, (size_t)((ptrdiff_t)base + (ptrdiff_t)i * st), buf[i]);
        });
        return wrap(out);
    });

    define("nd_argsort", [](std::vector<Value> args) -> Value {
        if (args.empty()) throw std::runtime_error("nd_argsort(a, axis?) needs an array");
        ArrayPtr a = asArray(toArrayValue(args[0], "nd_argsort"));
        if (a->ndim() == 0) return wrap(makeArray({}, DType::I64, "nd_argsort"));
        const size_t axis = axisArg(args, 1, a->ndim(), "nd_argsort");
        ArrayPtr out = makeArray(a->shape, DType::I64, "nd_argsort");
        if (out->size() == 0) return wrap(out);
        std::vector<size_t> lanePos;
        forEachLane(*out, axis, [&](size_t b, size_t, ptrdiff_t) { lanePos.push_back(b); });
        size_t li = 0;
        const NdArray& A = *a; NdArray& O = *out;
        const ptrdiff_t ost = O.strides[axis];
        std::vector<double> buf;
        std::vector<int64_t> idx;
        forEachLane(A, axis, [&](size_t base, size_t n, ptrdiff_t st) {
            const size_t ob = lanePos[li++];
            buf.clear(); idx.clear(); buf.reserve(n); idx.reserve(n);
            for (size_t i = 0; i < n; i++) {
                buf.push_back(getAsDouble(A, (size_t)((ptrdiff_t)base + (ptrdiff_t)i * st)));
                idx.push_back((int64_t)i);
            }
            // STABLE: equal elements keep their input order, so the result is
            // reproducible run to run rather than depending on sort internals.
            std::stable_sort(idx.begin(), idx.end(), [&](int64_t p, int64_t q) {
                return lessNaNLast(buf[(size_t)p], buf[(size_t)q]);
            });
            for (size_t i = 0; i < n; i++)
                setFromDouble(O, (size_t)((ptrdiff_t)ob + (ptrdiff_t)i * ost), (double)idx[i]);
        });
        return wrap(out);
    });

    // ── searchsorted / unique / bincount / histogram ─────────────────────────
    define("nd_searchsorted", [](std::vector<Value> args) -> Value {
        if (args.size() < 2) {
            throw std::runtime_error("nd_searchsorted(sorted, values, side?) needs two arrays");
        }
        ArrayPtr s = asArray(toArrayValue(args[0], "nd_searchsorted"));
        ArrayPtr v = asArray(toArrayValue(args[1], "nd_searchsorted"));
        if (s->ndim() != 1) {
            throw std::runtime_error("nd_searchsorted: the sorted array must be 1-dimensional (got " +
                                     shapeStr(s->shape) + ")");
        }
        const bool right = (args.size() > 2 && !args[2].isNull() && args[2].isString() &&
                            args[2].stringVal == "right");
        std::vector<double> hay; hay.reserve(s->size());
        for (size_t i = 0; i < s->size(); i++)
            hay.push_back(getAsDouble(*s, (size_t)((ptrdiff_t)s->offset + (ptrdiff_t)i * s->strides[0])));
        ArrayPtr out = makeArray(v->shape, DType::I64, "nd_searchsorted");
        size_t k = 0;
        const NdArray& V = *v; NdArray& O = *out;
        forEachIndex(V, [&](size_t o, size_t) {
            const double x = getAsDouble(V, o);
            const auto it = right ? std::upper_bound(hay.begin(), hay.end(), x)
                                  : std::lower_bound(hay.begin(), hay.end(), x);
            setFromDouble(O, k++, (double)(it - hay.begin()));
        });
        return wrap(out);
    });

    define("nd_unique", [](std::vector<Value> args) -> Value {
        if (args.empty()) throw std::runtime_error("nd_unique(a) needs an array");
        ArrayPtr a = asArray(toArrayValue(args[0], "nd_unique"));
        std::vector<double> v; v.reserve(a->size());
        const NdArray& A = *a;
        forEachIndex(A, [&](size_t o, size_t) { v.push_back(getAsDouble(A, o)); });
        std::sort(v.begin(), v.end(), lessNaNLast);
        // NaN is never equal to itself, so std::unique would keep every copy.
        // NumPy collapses them to one, which is the useful answer.
        std::vector<double> u;
        for (double x : v) {
            if (u.empty()) { u.push_back(x); continue; }
            const double last = u.back();
            if (std::isnan(x) && std::isnan(last)) continue;
            if (x != last) u.push_back(x);
        }
        ArrayPtr out = makeArray({ u.size() }, a->dtype == DType::F64 ? DType::F64 : DType::I64,
                                 "nd_unique");
        for (size_t i = 0; i < u.size(); i++) setFromDouble(*out, i, u[i]);
        return wrap(out);
    });

    define("nd_bincount", [](std::vector<Value> args) -> Value {
        if (args.empty()) throw std::runtime_error("nd_bincount(a, minlength?) needs an array");
        ArrayPtr a = asArray(toArrayValue(args[0], "nd_bincount"));
        size_t minlen = 0;
        if (args.size() > 1 && !args[1].isNull()) {
            if (!args[1].isNumber() || args[1].numberVal < 0) {
                throw std::runtime_error("nd_bincount: minlength must be a non-negative number");
            }
            minlen = (size_t)args[1].numberVal;
        }
        std::vector<int64_t> vals; vals.reserve(a->size());
        const NdArray& A = *a;
        int64_t hi = -1;
        bool bad = false; double badVal = 0;
        forEachIndex(A, [&](size_t o, size_t) {
            const double x = getAsDouble(A, o);
            if (std::isnan(x) || std::isinf(x) || x < 0 || x != std::floor(x)) {
                if (!bad) { bad = true; badVal = x; }
                return;
            }
            const int64_t i = (int64_t)x;
            vals.push_back(i);
            if (i > hi) hi = i;
        });
        if (bad) {
            throw std::runtime_error("nd_bincount: every value must be a non-negative whole number "
                                     "(got " + std::to_string(badVal) + ")");
        }
        const size_t n = std::max(minlen, (size_t)(hi + 1));
        ArrayPtr out = makeArray({ n }, DType::I64, "nd_bincount");
        for (int64_t i : vals) {
            const size_t b = (size_t)i;
            setFromDouble(*out, b, getAsDouble(*out, b) + 1);
        }
        return wrap(out);
    });

    define("nd_histogram", [](std::vector<Value> args) -> Value {
        if (args.empty()) throw std::runtime_error("nd_histogram(a, bins?, lo?, hi?) needs an array");
        ArrayPtr a = asArray(toArrayValue(args[0], "nd_histogram"));
        size_t bins = 10;
        if (args.size() > 1 && !args[1].isNull()) {
            if (!args[1].isNumber() || args[1].numberVal < 1) {
                throw std::runtime_error("nd_histogram: bins must be at least 1");
            }
            bins = (size_t)args[1].numberVal;
        }
        double lo = 0, hi = 0; bool haveRange = false;
        if (args.size() > 3 && !args[2].isNull() && !args[3].isNull()) {
            lo = args[2].numberVal; hi = args[3].numberVal; haveRange = true;
            if (!(lo < hi)) {
                throw std::runtime_error("nd_histogram: lo must be less than hi");
            }
        }
        const NdArray& A = *a;
        if (!haveRange) {
            bool first = true;
            forEachIndex(A, [&](size_t o, size_t) {
                const double x = getAsDouble(A, o);
                if (std::isnan(x)) return;
                if (first) { lo = hi = x; first = false; return; }
                if (x < lo) lo = x;
                if (x > hi) hi = x;
            });
            if (first) { lo = 0; hi = 1; }
            if (lo == hi) { lo -= 0.5; hi += 0.5; }   // a degenerate range is not an error
        }
        ArrayPtr out = makeArray({ bins }, DType::I64, "nd_histogram");
        const double width = (hi - lo) / (double)bins;
        forEachIndex(A, [&](size_t o, size_t) {
            const double x = getAsDouble(A, o);
            if (std::isnan(x) || x < lo || x > hi) return;
            size_t b = (size_t)std::floor((x - lo) / width);
            if (b >= bins) b = bins - 1;              // the top edge is inclusive
            setFromDouble(*out, b, getAsDouble(*out, b) + 1);
        });
        return wrap(out);
    });

    // ── fancy and boolean indexing ───────────────────────────────────────────
    define("nd_take", [](std::vector<Value> args) -> Value {
        if (args.size() < 2) throw std::runtime_error("nd_take(a, indices, axis?) needs two arrays");
        ArrayPtr a = asArray(toArrayValue(args[0], "nd_take"));
        ArrayPtr ix = asArray(toArrayValue(args[1], "nd_take"));
        if (a->ndim() == 0) throw std::runtime_error("nd_take: cannot take from a 0-d array");
        const size_t axis = axisArg(args, 2, a->ndim(), "nd_take");
        const ptrdiff_t extent = (ptrdiff_t)a->shape[axis];

        std::vector<ptrdiff_t> idx; idx.reserve(ix->size());
        const NdArray& IX = *ix;
        forEachIndex(IX, [&](size_t o, size_t) {
            const double d = getAsDouble(IX, o);
            if (std::isnan(d) || std::isinf(d) || d != std::floor(d)) {
                throw std::runtime_error("nd_take: indices must be whole numbers (got " +
                                         std::to_string(d) + ")");
            }
            ptrdiff_t i = (ptrdiff_t)d;
            if (i < 0) i += extent;                 // -1 is the last element
            if (i < 0 || i >= extent) {
                throw std::runtime_error("nd_take: index " + std::to_string((ptrdiff_t)d) +
                    " is out of range for axis " + std::to_string(axis) + " with " +
                    std::to_string(extent) + " elements");
            }
            idx.push_back(i);
        });

        std::vector<size_t> shape = a->shape;
        shape[axis] = idx.size();
        ArrayPtr out = makeArray(shape, a->dtype, "nd_take");
        if (out->size() == 0) return wrap(out);

        const NdArray& A = *a; NdArray& O = *out;
        std::vector<size_t> lanePos;
        forEachLane(O, axis, [&](size_t b, size_t, ptrdiff_t) { lanePos.push_back(b); });
        size_t li = 0;
        const ptrdiff_t ost = O.strides[axis];
        forEachLane(A, axis, [&](size_t base, size_t, ptrdiff_t st) {
            const size_t ob = lanePos[li++];
            for (size_t i = 0; i < idx.size(); i++) {
                const double x = getAsDouble(A, (size_t)((ptrdiff_t)base + idx[i] * st));
                setFromDouble(O, (size_t)((ptrdiff_t)ob + (ptrdiff_t)i * ost), x);
            }
        });
        return wrap(out);
    });

    // nd_compress(mask, a) / nd_mask — select where the mask is true, flattened.
    define("nd_compress", [](std::vector<Value> args) -> Value {
        if (args.size() < 2) throw std::runtime_error("nd_compress(mask, a) needs a mask and an array");
        ArrayPtr m = asArray(toArrayValue(args[0], "nd_compress"));
        ArrayPtr a = asArray(toArrayValue(args[1], "nd_compress"));
        if (m->size() != a->size()) {
            throw std::runtime_error("nd_compress: the mask has " + std::to_string(m->size()) +
                " elements but the array has " + std::to_string(a->size()) +
                " -- a mask of the wrong length is an error, not a truncation");
        }
        std::vector<double> keep;
        const NdArray& M = *m; const NdArray& A = *a;
        std::vector<size_t> mo; mo.reserve(m->size());
        forEachIndex(M, [&](size_t o, size_t) { mo.push_back(o); });
        size_t k = 0;
        forEachIndex(A, [&](size_t o, size_t) {
            if (getAsDouble(M, mo[k++]) != 0.0) keep.push_back(getAsDouble(A, o));
        });
        ArrayPtr out = makeArray({ keep.size() }, a->dtype, "nd_compress");
        for (size_t i = 0; i < keep.size(); i++) setFromDouble(*out, i, keep[i]);
        return wrap(out);
    });

    // nd_nonzero(a) — the flat C-order indices of the non-zero elements.
    define("nd_nonzero", [](std::vector<Value> args) -> Value {
        if (args.empty()) throw std::runtime_error("nd_nonzero(a) needs an array");
        ArrayPtr a = asArray(toArrayValue(args[0], "nd_nonzero"));
        std::vector<double> hits;
        const NdArray& A = *a;
        forEachIndex(A, [&](size_t o, size_t linear) {
            if (getAsDouble(A, o) != 0.0) hits.push_back((double)linear);
        });
        ArrayPtr out = makeArray({ hits.size() }, DType::I64, "nd_nonzero");
        for (size_t i = 0; i < hits.size(); i++) setFromDouble(*out, i, hits[i]);
        return wrap(out);
    });

    // nd_put(a, indices, values) — scatter, in place, along the flattened array.
    define("nd_put", [](std::vector<Value> args) -> Value {
        if (args.size() < 3) throw std::runtime_error("nd_put(a, indices, values) needs three arguments");
        ArrayPtr a = asArray(args[0]);
        requireWritable(*a, "nd_put");
        ArrayPtr ix = asArray(toArrayValue(args[1], "nd_put"));
        ArrayPtr vs = asArray(toArrayValue(args[2], "nd_put"));
        if (vs->size() != 1 && vs->size() != ix->size()) {
            throw std::runtime_error("nd_put: got " + std::to_string(vs->size()) +
                " values for " + std::to_string(ix->size()) +
                " indices (give one value, or one per index)");
        }
        // The flat walk order has to be materialised because the target may be
        // a strided view, where the nth element is not at offset n.
        std::vector<size_t> flat; flat.reserve(a->size());
        const NdArray& A = *a;
        forEachIndex(A, [&](size_t o, size_t) { flat.push_back(o); });

        const ptrdiff_t total = (ptrdiff_t)flat.size();
        std::vector<double> vals; vals.reserve(vs->size());
        const NdArray& VS = *vs;
        forEachIndex(VS, [&](size_t o, size_t) { vals.push_back(getAsDouble(VS, o)); });

        size_t k = 0;
        const NdArray& IX = *ix;
        NdArray& AW = *a;
        forEachIndex(IX, [&](size_t o, size_t) {
            const double d = getAsDouble(IX, o);
            if (std::isnan(d) || std::isinf(d) || d != std::floor(d)) {
                throw std::runtime_error("nd_put: indices must be whole numbers (got " +
                                         std::to_string(d) + ")");
            }
            ptrdiff_t i = (ptrdiff_t)d;
            if (i < 0) i += total;
            if (i < 0 || i >= total) {
                throw std::runtime_error("nd_put: index " + std::to_string((ptrdiff_t)d) +
                    " is out of range for an array of " + std::to_string(total) + " elements");
            }
            setFromDouble(AW, flat[(size_t)i], vals.size() == 1 ? vals[0] : vals[k]);
            k++;
        });
        return args[0];
    });
}

}   // namespace numba
