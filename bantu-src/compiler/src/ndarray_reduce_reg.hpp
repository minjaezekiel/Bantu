#pragma once
/**
 * numba — the reduction, scan, sort and indexing builtin table.
 *
 * Included by ndarray_native.cpp only, after ndarray_api.hpp.
 * docs/numba-architecture.md §4.8-4.11.
 */

#include "ndarray_reduce.hpp"
#include "ndarray_api.hpp"

namespace numba {

Value toArrayValue(const Value& v, const char* what);

// Shared argument shape for every reduction: (array, axis?, keepdims?).
struct RedArgs {
    ArrayPtr    a;
    ReducePlan  plan;
    bool        keepdims = false;
};

inline RedArgs redArgs(const std::vector<Value>& args, const char* what) {
    if (args.empty()) {
        throw std::runtime_error(std::string(what) + "(a, axis?, keepdims?) needs an array");
    }
    RedArgs r;
    r.a = asArray(toArrayValue(args[0], what));
    const Value axisV = (args.size() > 1) ? args[1] : Value();
    r.keepdims = (args.size() > 2 && !args[2].isNull()) ? args[2].isTruthy() : false;
    const std::vector<size_t> axes = axesFrom(axisV, r.a->ndim(), what);
    r.plan = planReduce(*r.a, axes, r.keepdims);
    return r;
}

// An empty slice has no identity for min/max/arg*, and returning 0 or +-inf
// would be a silently wrong answer rather than an error.
inline void requireNonEmptySlice(const ReducePlan& p, const char* what) {
    if (p.innerCount == 0) {
        throw std::runtime_error(std::string(what) +
            ": cannot reduce an empty slice -- there is no identity value for this operation");
    }
}

inline void registerReductions(const DefineFn& define) {

    // ── sum / mean, on the pairwise accumulator (docs §4.9) ──────────────────
    auto sumSlice = [](const NdArray& a, const ReducePlan& p, size_t base) -> double {
        if (a.dtype == DType::F64 && innerIsContig(p)) {
            return sumContig(static_cast<const double*>(a.buf->data) + base, p.innerCount);
        }
        Pairwise acc;
        innerSweep(p, base, [&](size_t o) { acc.add(getAsDouble(a, o)); });
        return acc.total();
    };

    define("nd_sum", [sumSlice](std::vector<Value> args) -> Value {
        RedArgs r = redArgs(args, "nd_sum");
        // i64 in, i64 out: the whole point of carrying an integer path is that
        // counts and indices stay exact.
        const DType rdt = (r.a->dtype == DType::F64) ? DType::F64 : DType::I64;
        const NdArray& A = *r.a; const ReducePlan& P = r.plan;
        return wrap(runReduce(A, P, rdt, "nd_sum",
            [&](size_t base) { return sumSlice(A, P, base); }));
    });

    define("nd_mean", [sumSlice](std::vector<Value> args) -> Value {
        RedArgs r = redArgs(args, "nd_mean");
        const NdArray& A = *r.a; const ReducePlan& P = r.plan;
        const double n = (double)P.innerCount;
        // mean of an empty slice is NaN, as in NumPy -- 0/0, not an error.
        return wrap(runReduce(A, P, DType::F64, "nd_mean",
            [&](size_t base) { return sumSlice(A, P, base) / n; }));
    });

    define("nd_prod", [](std::vector<Value> args) -> Value {
        RedArgs r = redArgs(args, "nd_prod");
        const DType rdt = (r.a->dtype == DType::F64) ? DType::F64 : DType::I64;
        const NdArray& A = *r.a; const ReducePlan& P = r.plan;
        return wrap(runReduce(A, P, rdt, "nd_prod", [&](size_t base) {
            double acc = 1.0;   // the identity, so an empty slice gives 1
            innerSweep(P, base, [&](size_t o) { acc *= getAsDouble(A, o); });
            return acc;
        }));
    });

    // ── var / std, Welford one-pass (docs §4.9) ──────────────────────────────
    auto welford = [](const NdArray& a, const ReducePlan& p, size_t base, double ddof) -> double {
        double mean = 0.0, m2 = 0.0;
        size_t n = 0;
        innerSweep(p, base, [&](size_t o) {
            const double x = getAsDouble(a, o);
            n++;
            const double d = x - mean;
            mean += d / (double)n;
            m2   += d * (x - mean);
        });
        const double denom = (double)n - ddof;
        if (denom <= 0.0) return std::numeric_limits<double>::quiet_NaN();
        return m2 / denom;
    };

    define("nd_var", [welford](std::vector<Value> args) -> Value {
        RedArgs r = redArgs(args, "nd_var");
        const double ddof = (args.size() > 3 && !args[3].isNull()) ? args[3].numberVal : 0.0;
        const NdArray& A = *r.a; const ReducePlan& P = r.plan;
        return wrap(runReduce(A, P, DType::F64, "nd_var",
            [&](size_t base) { return welford(A, P, base, ddof); }));
    });
    define("nd_std", [welford](std::vector<Value> args) -> Value {
        RedArgs r = redArgs(args, "nd_std");
        const double ddof = (args.size() > 3 && !args[3].isNull()) ? args[3].numberVal : 0.0;
        const NdArray& A = *r.a; const ReducePlan& P = r.plan;
        return wrap(runReduce(A, P, DType::F64, "nd_std",
            [&](size_t base) { return std::sqrt(welford(A, P, base, ddof)); }));
    });

    // ── min / max / ptp, and the arg* forms ──────────────────────────────────
    // NaN propagates: if any element is NaN the answer is NaN, matching NumPy's
    // min/max rather than its nanmin/nanmax.
    auto extreme = [](const char* what, bool wantMax) {
        return [what, wantMax](std::vector<Value> args) -> Value {
            RedArgs r = redArgs(args, what);
            requireNonEmptySlice(r.plan, what);
            const DType rdt = r.a->dtype == DType::BOOL ? DType::I64 : r.a->dtype;
            const NdArray& A = *r.a; const ReducePlan& P = r.plan;
            return wrap(runReduce(A, P, rdt, what, [&](size_t base) {
                bool first = true; double best = 0.0;
                innerSweep(P, base, [&](size_t o) {
                    const double x = getAsDouble(A, o);
                    if (first) { best = x; first = false; return; }
                    if (std::isnan(best)) return;
                    if (std::isnan(x)) { best = x; return; }
                    if (wantMax ? (x > best) : (x < best)) best = x;
                });
                return best;
            }));
        };
    };
    define("nd_min", extreme("nd_min", false));
    define("nd_max", extreme("nd_max", true));

    define("nd_ptp", [](std::vector<Value> args) -> Value {
        RedArgs r = redArgs(args, "nd_ptp");
        requireNonEmptySlice(r.plan, "nd_ptp");
        const DType rdt = r.a->dtype == DType::BOOL ? DType::I64 : r.a->dtype;
        const NdArray& A = *r.a; const ReducePlan& P = r.plan;
        return wrap(runReduce(A, P, rdt, "nd_ptp", [&](size_t base) {
            bool first = true; double lo = 0.0, hi = 0.0;
            innerSweep(P, base, [&](size_t o) {
                const double x = getAsDouble(A, o);
                if (first) { lo = hi = x; first = false; return; }
                if (x < lo) lo = x;
                if (x > hi) hi = x;
            });
            return hi - lo;
        }));
    });

    // arg* returns the index WITHIN the reduced sweep, in C order.
    auto argExtreme = [](const char* what, bool wantMax) {
        return [what, wantMax](std::vector<Value> args) -> Value {
            RedArgs r = redArgs(args, what);
            requireNonEmptySlice(r.plan, what);
            const NdArray& A = *r.a; const ReducePlan& P = r.plan;
            return wrap(runReduce(A, P, DType::I64, what, [&](size_t base) {
                size_t idx = 0, bestIdx = 0, i = 0;
                bool first = true; double best = 0.0;
                innerSweep(P, base, [&](size_t o) {
                    const double x = getAsDouble(A, o);
                    i = idx++;
                    if (first) { best = x; bestIdx = i; first = false; return; }
                    if (std::isnan(best)) return;              // NaN wins and keeps its index
                    if (std::isnan(x)) { best = x; bestIdx = i; return; }
                    if (wantMax ? (x > best) : (x < best)) { best = x; bestIdx = i; }
                });
                return (double)bestIdx;
            }));
        };
    };
    define("nd_argmin", argExtreme("nd_argmin", false));
    define("nd_argmax", argExtreme("nd_argmax", true));

    // ── any / all / count_nonzero ────────────────────────────────────────────
    define("nd_any", [](std::vector<Value> args) -> Value {
        RedArgs r = redArgs(args, "nd_any");
        const NdArray& A = *r.a; const ReducePlan& P = r.plan;
        return wrap(runReduce(A, P, DType::BOOL, "nd_any", [&](size_t base) {
            bool found = false;                       // identity: empty -> false
            innerSweep(P, base, [&](size_t o) { if (getAsDouble(A, o) != 0.0) found = true; });
            return found ? 1.0 : 0.0;
        }));
    });
    define("nd_all", [](std::vector<Value> args) -> Value {
        RedArgs r = redArgs(args, "nd_all");
        const NdArray& A = *r.a; const ReducePlan& P = r.plan;
        return wrap(runReduce(A, P, DType::BOOL, "nd_all", [&](size_t base) {
            bool all = true;                          // identity: empty -> true
            innerSweep(P, base, [&](size_t o) { if (getAsDouble(A, o) == 0.0) all = false; });
            return all ? 1.0 : 0.0;
        }));
    });
    define("nd_count_nonzero", [](std::vector<Value> args) -> Value {
        RedArgs r = redArgs(args, "nd_count_nonzero");
        const NdArray& A = *r.a; const ReducePlan& P = r.plan;
        return wrap(runReduce(A, P, DType::I64, "nd_count_nonzero", [&](size_t base) {
            double n = 0;
            innerSweep(P, base, [&](size_t o) { if (getAsDouble(A, o) != 0.0) n += 1; });
            return n;
        }));
    });

    // ── nan-skipping variants ────────────────────────────────────────────────
    define("nd_nansum", [](std::vector<Value> args) -> Value {
        RedArgs r = redArgs(args, "nd_nansum");
        const NdArray& A = *r.a; const ReducePlan& P = r.plan;
        return wrap(runReduce(A, P, DType::F64, "nd_nansum", [&](size_t base) {
            Pairwise acc;
            innerSweep(P, base, [&](size_t o) {
                const double x = getAsDouble(A, o);
                if (!std::isnan(x)) acc.add(x);
            });
            return acc.total();
        }));
    });
    define("nd_nanmean", [](std::vector<Value> args) -> Value {
        RedArgs r = redArgs(args, "nd_nanmean");
        const NdArray& A = *r.a; const ReducePlan& P = r.plan;
        return wrap(runReduce(A, P, DType::F64, "nd_nanmean", [&](size_t base) {
            Pairwise acc; size_t n = 0;
            innerSweep(P, base, [&](size_t o) {
                const double x = getAsDouble(A, o);
                if (!std::isnan(x)) { acc.add(x); n++; }
            });
            // An all-NaN slice gives NaN, as in NumPy -- not 0, which would be
            // a real value that happens to be wrong.
            return n ? acc.total() / (double)n : std::numeric_limits<double>::quiet_NaN();
        }));
    });
    auto nanExtreme = [](const char* what, bool wantMax) {
        return [what, wantMax](std::vector<Value> args) -> Value {
            RedArgs r = redArgs(args, what);
            const NdArray& A = *r.a; const ReducePlan& P = r.plan;
            return wrap(runReduce(A, P, DType::F64, what, [&](size_t base) {
                bool any = false; double best = 0.0;
                innerSweep(P, base, [&](size_t o) {
                    const double x = getAsDouble(A, o);
                    if (std::isnan(x)) return;
                    if (!any) { best = x; any = true; return; }
                    if (wantMax ? (x > best) : (x < best)) best = x;
                });
                return any ? best : std::numeric_limits<double>::quiet_NaN();
            }));
        };
    };
    define("nd_nanmin", nanExtreme("nd_nanmin", false));
    define("nd_nanmax", nanExtreme("nd_nanmax", true));

    // ── median and quantile ──────────────────────────────────────────────────
    // Linear interpolation between order statistics, which is NumPy's default
    // method and the one every other library's "quantile" is compared against.
    auto quantileOf = [](std::vector<double>& v, double q) -> double {
        if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
        std::sort(v.begin(), v.end());
        // NaN sorts last under `<`? No -- comparisons with NaN are false, so
        // std::sort's order is unspecified with NaN present. Detect instead.
        for (double x : v) if (std::isnan(x)) return std::numeric_limits<double>::quiet_NaN();
        const double pos = q * (double)(v.size() - 1);
        const size_t lo = (size_t)std::floor(pos);
        const size_t hi = (size_t)std::ceil(pos);
        if (lo == hi) return v[lo];
        return v[lo] + (v[hi] - v[lo]) * (pos - (double)lo);
    };

    define("nd_median", [quantileOf](std::vector<Value> args) -> Value {
        RedArgs r = redArgs(args, "nd_median");
        const NdArray& A = *r.a; const ReducePlan& P = r.plan;
        return wrap(runReduce(A, P, DType::F64, "nd_median", [&](size_t base) {
            std::vector<double> v; v.reserve(P.innerCount);
            innerSweep(P, base, [&](size_t o) { v.push_back(getAsDouble(A, o)); });
            std::vector<double> tmp = v;
            return quantileOf(tmp, 0.5);
        }));
    });

    define("nd_quantile", [quantileOf](std::vector<Value> args) -> Value {
        if (args.size() < 2) {
            throw std::runtime_error("nd_quantile(a, q, axis?, keepdims?) needs an array and a q");
        }
        if (!args[1].isNumber()) {
            throw std::runtime_error("nd_quantile: q must be a number (got " + args[1].toString() + ")");
        }
        const double q = args[1].numberVal;
        if (std::isnan(q) || q < 0.0 || q > 1.0) {
            throw std::runtime_error("nd_quantile: q must be between 0 and 1 (got " +
                                     args[1].toString() + ")");
        }
        // Shift the remaining arguments so redArgs sees (a, axis?, keepdims?).
        std::vector<Value> shifted;
        shifted.push_back(args[0]);
        for (size_t i = 2; i < args.size(); i++) shifted.push_back(args[i]);
        RedArgs r = redArgs(shifted, "nd_quantile");
        const NdArray& A = *r.a; const ReducePlan& P = r.plan;
        return wrap(runReduce(A, P, DType::F64, "nd_quantile", [&](size_t base) {
            std::vector<double> v; v.reserve(P.innerCount);
            innerSweep(P, base, [&](size_t o) { v.push_back(getAsDouble(A, o)); });
            return quantileOf(v, q);
        }));
    });
}

}   // namespace numba
