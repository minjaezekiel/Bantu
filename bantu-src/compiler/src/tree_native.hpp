// ═════════════════════════════════════════════════════════════════════════════
//  tree_native.hpp — the `tree_` native family: binning, histograms, split
//  finding, partitioning and traversal.
//
//  These five kernels are not an optimisation. A Bantu function call costs
//  1.78 us (btorch B1); `tree_hist` on a 1M x 100 dataset touches 10^8 cells
//  per tree level, four orders of magnitude outside what the evaluator can do
//  per element. Without them there is no gradient booster.
//
//  Design: docs/bkit-architecture.md section 8; bkit DECISIONS K1, K42-K46.
//
//  ── why this is a HEADER included by ndarray_native.cpp, not its own TU ────
//  Every kernel here reads an NdArray's raw typed storage or returns a fresh
//  array. The designated wall between native families (ndarray_api.hpp) carries
//  1-D vector borrow/export and a column-major matrix builder -- deliberately
//  narrow, so arctic and numba need not include each other. Crossing it here
//  would mean exporting X as a flat std::vector<double>: for 1M x 100 that is
//  **800 MB of copy** to feed kernels whose entire purpose is to replace those
//  doubles with one byte per cell. The copy would cost more than the algorithm.
//
//  So the tree kernels live in numba's translation unit -- which also gets them
//  the -O3 + vectorize flags that `ndarray_native.cpp` alone is built with, and
//  which the histogram loop is the reason for -- while keeping their own file,
//  their own prefix and their own registration function. The family is separate;
//  the compilation unit is shared, for a measured reason.
//
//  ── the binned matrix is an opaque handle (K1) ────────────────────────────
//  One byte per cell. numba's DType has no u8, and I64 bins would cost 8 bytes
//  a cell -- 800 MB for 1M x 100, defeating the point of binning. So the binned
//  matrix is an opaque handle owning a uint8_t buffer, like the existing col_*
//  and bp_* handles. Users never index or slice it; five kernels need
//  get(row, feature) and nothing else.
//
//  Because it is opaque, it is also untestable unless something can look
//  inside -- so the accessors (`tree_bin_shape`, `tree_bin_edges`,
//  `tree_bin_nbins`, `tree_bin_at`) exist for the tests, not for callers. An
//  opaque structure with no window is a structure whose contents are asserted
//  by hope.
// ═════════════════════════════════════════════════════════════════════════════

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace numba {
namespace tree {

inline const char* BINS_TAG = "treebins";

// ── the binned matrix ───────────────────────────────────────────────────────
// `bins` is row-major [rows, cols]. Per feature f, value bins are 0..k_f-1 and
// NaN -- when the column had any -- takes bin index k_f. `nbins[f]` is the
// total INCLUDING the NaN bin, so a histogram allocated with max(nbins) is
// always wide enough.
struct Bins {
    size_t rows = 0, cols = 0;
    std::vector<uint8_t> bins;
    std::vector<std::vector<double>> edges;  // per feature: k_f - 1 upper edges
    std::vector<int>  nbins;                 // per feature: total bins used
    std::vector<char> hasNaN;                // per feature: was a NaN bin given
    size_t maxBins = 0;                      // max over features, for hist width
};
using BinsPtr = std::shared_ptr<Bins>;

inline bool isBins(const Value& v) {
    return v.isNativeHandle() && v.handleTag() == BINS_TAG && v.handle;
}
inline BinsPtr asBins(const Value& v, const char* who) {
    if (!isBins(v)) {
        throw std::runtime_error(std::string(who) + ": expected a binned matrix "
            "from tree_bin (got " +
            (v.isNativeHandle() ? "<" + v.handleTag() + ">" : v.toString()) + ")");
    }
    return std::static_pointer_cast<Bins>(v.handle);
}
inline Value wrapBins(BinsPtr b) {
    return Value(std::static_pointer_cast<void>(b), BINS_TAG);
}

// print($binned) shows the shape, not "<treebins>".
inline std::string reprBins(const std::shared_ptr<void>& h) {
    if (!h) return "<treebins: null>";
    const Bins& b = *std::static_pointer_cast<Bins>(h);
    std::string s = "<treebins " + std::to_string(b.rows) + "x" +
                    std::to_string(b.cols) + ", bins=[";
    for (size_t f = 0; f < b.nbins.size(); f++) {
        if (f) s += ",";
        if (f >= 8) { s += "..."; break; }
        s += std::to_string(b.nbins[f]);
    }
    return s + "], " + std::to_string(b.bins.size()) + " bytes>";
}
inline long long lenBins(const std::shared_ptr<void>& h) {
    if (!h) return 0;
    return (long long)std::static_pointer_cast<Bins>(h)->rows;
}

// ── helpers ─────────────────────────────────────────────────────────────────

inline void needTreeArgs(const std::vector<Value>& a, size_t n, const char* usage) {
    if (a.size() < n) {
        throw std::runtime_error(std::string("usage: ") + usage);
    }
}

inline double treeNum(const std::vector<Value>& a, size_t i, double dflt,
                      const char* who, const char* what) {
    if (a.size() <= i || a[i].isNull()) return dflt;
    if (!a[i].isNumber()) {
        throw std::runtime_error(std::string(who) + ": " + what +
            " must be a number, got " + a[i].toString());
    }
    const double d = a[i].numberVal;
    if (std::isnan(d)) {
        throw std::runtime_error(std::string(who) + ": " + what + " is NaN");
    }
    return d;
}

// A 2-D f64 view's values, read through strides so a transposed or sliced X
// works. Copied once because every kernel below makes several passes.
inline void readMatrix(const NdArray& A, std::vector<double>& out,
                       size_t& rows, size_t& cols, const char* who) {
    if (A.shape.size() != 2) {
        throw std::runtime_error(std::string(who) +
            ": X must be 2-d [n_samples, n_features], got " + shapeStr(A.shape));
    }
    rows = A.shape[0];
    cols = A.shape[1];
    if (rows == 0 || cols == 0) {
        throw std::runtime_error(std::string(who) + ": X is empty (" +
            shapeStr(A.shape) + ")");
    }
    if (cols > 100000) {
        throw std::runtime_error(std::string(who) + ": " + std::to_string(cols) +
            " features is past any plausible model");
    }
    out.resize(rows * cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            out[i * cols + j] = getAsDouble(A, flatOffset(A, {i, j}));
        }
    }
}

// A 1-D f64 vector of a required length.
inline void readVector(const Value& v, size_t want, std::vector<double>& out,
                       const char* who, const char* what) {
    ArrayPtr a = asArray(v);
    if (a->shape.size() != 1) {
        throw std::runtime_error(std::string(who) + ": " + what +
            " must be 1-d, got " + shapeStr(a->shape));
    }
    if (a->shape[0] != want) {
        throw std::runtime_error(std::string(who) + ": " + what + " has " +
            std::to_string(a->shape[0]) + " elements but X has " +
            std::to_string(want) + " rows");
    }
    out.resize(want);
    for (size_t i = 0; i < want; i++) out[i] = getAsDouble(*a, flatOffset(*a, {i}));
}

// The sample index set for a node. null means "every row", which is the root
// and is also the common case in a test.
inline void readIndex(const Value& v, size_t rows, std::vector<size_t>& out,
                      const char* who) {
    if (v.isNull()) {
        out.resize(rows);
        for (size_t i = 0; i < rows; i++) out[i] = i;
        return;
    }
    ArrayPtr a = asArray(v);
    if (a->shape.size() != 1) {
        throw std::runtime_error(std::string(who) +
            ": the index array must be 1-d, got " + shapeStr(a->shape));
    }
    const size_t n = a->shape[0];
    out.resize(n);
    for (size_t k = 0; k < n; k++) {
        const double d = getAsDouble(*a, flatOffset(*a, {k}));
        if (std::isnan(d) || d < 0 || d != std::floor(d) || (size_t)d >= rows) {
            throw std::runtime_error(std::string(who) + ": index " +
                std::to_string(d) + " is not a row of a " + std::to_string(rows) +
                "-row matrix");
        }
        out[k] = (size_t)d;
    }
}

inline Value makeI64Vector(const std::vector<size_t>& v) {
    ArrayPtr out = makeArray({v.size()}, DType::I64, "tree_partition");
    int64_t* p = out->ptr<int64_t>();
    for (size_t i = 0; i < v.size(); i++) p[i] = (int64_t)v[i];
    return wrap(out);
}

inline Value makeF64Vector(const std::vector<double>& v) {
    ArrayPtr out = makeArray({v.size()}, DType::F64, "tree");
    double* p = out->ptr<double>();
    for (size_t i = 0; i < v.size(); i++) p[i] = v[i];
    return wrap(out);
}

// ── tree_bin ────────────────────────────────────────────────────────────────
// Quantile binning. The edges are taken from the SORTED DISTINCT values, which
// is what makes a low-cardinality column behave: an integer column with 7
// distinct values gets 7 bins, not 256 mostly-empty ones, and a constant column
// gets exactly 1. Binning on evenly spaced quantiles of the raw sorted array
// instead would place several edges between the same pair of values and produce
// empty bins that every later scan still has to walk.
inline Value kernel_bin(std::vector<Value> a) {
    needTreeArgs(a, 1, "tree_bin(X, max_bins?)");
    ArrayPtr X = asArray(a[0]);
    size_t rows = 0, cols = 0;
    std::vector<double> v;
    readMatrix(*X, v, rows, cols, "tree_bin");

    double mbD = treeNum(a, 1, 256.0, "tree_bin", "max_bins");
    if (mbD != std::floor(mbD) || mbD < 2 || mbD > 256) {
        throw std::runtime_error("tree_bin: max_bins must be a whole number in "
            "2..256 (one byte per cell is the point), got " +
            std::to_string(mbD));
    }
    const int maxBins = (int)mbD;

    BinsPtr B = std::make_shared<Bins>();
    B->rows = rows;
    B->cols = cols;
    B->bins.assign(rows * cols, 0);
    B->edges.resize(cols);
    B->nbins.assign(cols, 1);
    B->hasNaN.assign(cols, 0);

    std::vector<double> col;
    col.reserve(rows);
    for (size_t f = 0; f < cols; f++) {
        col.clear();
        bool anyNaN = false;
        for (size_t i = 0; i < rows; i++) {
            const double x = v[i * cols + f];
            if (std::isnan(x)) { anyNaN = true; continue; }
            if (std::isinf(x)) {
                throw std::runtime_error("tree_bin: feature " + std::to_string(f) +
                    " contains an infinity. A bin edge cannot separate it from "
                    "itself -- replace it or mark it missing (NaN), which gets a "
                    "bin of its own.");
            }
            col.push_back(x);
        }
        B->hasNaN[f] = anyNaN ? 1 : 0;

        if (col.empty()) {
            // Every value missing. One bin, the NaN bin, and say so rather than
            // silently producing a feature that can never split.
            B->nbins[f] = 1;
            for (size_t i = 0; i < rows; i++) B->bins[i * cols + f] = 0;
            continue;
        }
        std::sort(col.begin(), col.end());
        col.erase(std::unique(col.begin(), col.end()), col.end());

        // Edges are upper bounds: bin b holds values <= edges[b], and the last
        // bin is everything above the final edge. k distinct values want k-1
        // edges; more bins than that would be empty.
        const int wantValueBins = (int)std::min<size_t>(col.size(), (size_t)(maxBins - (anyNaN ? 1 : 0)));
        const int k = std::max(1, wantValueBins);
        std::vector<double>& e = B->edges[f];
        e.clear();
        if (k > 1) {
            e.reserve((size_t)k - 1);
            for (int b = 1; b < k; b++) {
                // The cut sits BETWEEN two distinct values, at the lower one,
                // so `x <= edge` is a test no float rounding can straddle.
                const size_t at = (size_t)((double)b * (double)col.size() / (double)k);
                const size_t idx = std::min(at == 0 ? 0 : at - 1, col.size() - 1);
                const double cut = col[idx];
                if (e.empty() || cut > e.back()) e.push_back(cut);
            }
        }
        const int valueBins = (int)e.size() + 1;
        B->nbins[f] = valueBins + (anyNaN ? 1 : 0);
        const uint8_t nanBin = (uint8_t)valueBins;

        for (size_t i = 0; i < rows; i++) {
            const double x = v[i * cols + f];
            if (std::isnan(x)) { B->bins[i * cols + f] = nanBin; continue; }
            // LOWER_bound, not upper_bound. The contract is "bin b holds values
            // <= edges[b]", i.e. bin(x) = min{b : edges[b] >= x} -- which is
            // lower_bound. upper_bound finds the first edge strictly GREATER
            // than x, so a value sitting exactly on its own edge was pushed one
            // bin right: with 8 distinct values the bins came out
            // [1,2,3,4,5,6,7,7] instead of [0..7], collapsing the top two and
            // leaving bin 0 empty. The split's *gain* was still right (the same
            // partition was reachable one bin over), so only a test that
            // checked the BIN INDEX against an independently computed argmax
            // caught it -- and then the threshold lookup edges[bin] was off by
            // one row at predict time.
            const size_t b = (size_t)(std::lower_bound(e.begin(), e.end(), x) - e.begin());
            B->bins[i * cols + f] = (uint8_t)b;
        }
    }
    size_t mx = 1;
    for (size_t f = 0; f < cols; f++) mx = std::max(mx, (size_t)B->nbins[f]);
    B->maxBins = mx;
    return wrapBins(B);
}

// ── accessors, for the tests ────────────────────────────────────────────────
inline Value kernel_bin_shape(std::vector<Value> a) {
    needTreeArgs(a, 1, "tree_bin_shape(binned)");
    BinsPtr B = asBins(a[0], "tree_bin_shape");
    ArrayPtr out = makeArray({2}, DType::I64, "tree_bin_shape");
    out->ptr<int64_t>()[0] = (int64_t)B->rows;
    out->ptr<int64_t>()[1] = (int64_t)B->cols;
    return wrap(out);
}
inline Value kernel_bin_nbins(std::vector<Value> a) {
    needTreeArgs(a, 1, "tree_bin_nbins(binned)");
    BinsPtr B = asBins(a[0], "tree_bin_nbins");
    ArrayPtr out = makeArray({B->cols}, DType::I64, "tree_bin_nbins");
    for (size_t f = 0; f < B->cols; f++) out->ptr<int64_t>()[f] = B->nbins[f];
    return wrap(out);
}
inline Value kernel_bin_edges(std::vector<Value> a) {
    needTreeArgs(a, 2, "tree_bin_edges(binned, feature)");
    BinsPtr B = asBins(a[0], "tree_bin_edges");
    const double fD = treeNum(a, 1, 0.0, "tree_bin_edges", "feature");
    if (fD != std::floor(fD) || fD < 0 || (size_t)fD >= B->cols) {
        throw std::runtime_error("tree_bin_edges: feature " + std::to_string(fD) +
            " is out of range for " + std::to_string(B->cols) + " features");
    }
    return makeF64Vector(B->edges[(size_t)fD]);
}
inline Value kernel_bin_at(std::vector<Value> a) {
    needTreeArgs(a, 3, "tree_bin_at(binned, row, feature)");
    BinsPtr B = asBins(a[0], "tree_bin_at");
    const double rD = treeNum(a, 1, 0.0, "tree_bin_at", "row");
    const double fD = treeNum(a, 2, 0.0, "tree_bin_at", "feature");
    if (rD != std::floor(rD) || rD < 0 || (size_t)rD >= B->rows ||
        fD != std::floor(fD) || fD < 0 || (size_t)fD >= B->cols) {
        throw std::runtime_error("tree_bin_at: (" + std::to_string(rD) + "," +
            std::to_string(fD) + ") is outside a " + std::to_string(B->rows) +
            "x" + std::to_string(B->cols) + " matrix");
    }
    return Value((double)B->bins[(size_t)rD * B->cols + (size_t)fD]);
}
inline Value kernel_bin_bytes(std::vector<Value> a) {
    needTreeArgs(a, 1, "tree_bin_bytes(binned)");
    return Value((double)asBins(a[0], "tree_bin_bytes")->bins.size());
}

// ── tree_hist ───────────────────────────────────────────────────────────────
// The hot loop. Output is [cols, maxBins, 3] holding (sum g, sum h, count).
// One byte load and three adds per (sample, feature).
inline Value kernel_hist(std::vector<Value> a) {
    needTreeArgs(a, 3, "tree_hist(binned, grad, hess, idx?)");
    BinsPtr B = asBins(a[0], "tree_hist");
    std::vector<double> g, h;
    readVector(a[1], B->rows, g, "tree_hist", "grad");
    readVector(a[2], B->rows, h, "tree_hist", "hess");
    std::vector<size_t> idx;
    readIndex(a.size() > 3 ? a[3] : Value(), B->rows, idx, "tree_hist");

    const size_t cols = B->cols, nb = B->maxBins;
    ArrayPtr out = makeArray({cols, nb, 3}, DType::F64, "tree_hist");
    double* H = out->ptr<double>();
    const uint8_t* bp = B->bins.data();

    for (size_t k = 0; k < idx.size(); k++) {
        const size_t i = idx[k];
        const double gi = g[i], hi = h[i];
        const uint8_t* row = bp + i * cols;
        for (size_t f = 0; f < cols; f++) {
            double* cell = H + ((f * nb) + (size_t)row[f]) * 3;
            cell[0] += gi;
            cell[1] += hi;
            cell[2] += 1.0;
        }
    }
    return wrap(out);
}

// ── tree_best_split ─────────────────────────────────────────────────────────
// A running prefix over bins per feature. Returns
//   [feature, bin, gain, G_L, H_L, n_L, G_R, H_R, n_R, nan_left]
// with feature = -1 when nothing passed the constraints.
//
// `bin` is the HIGHEST VALUE BIN in the left child, and -1 means "no value bins
// on the left" -- which is only reachable with nan_left, i.e. the split that
// separates missing from present.
//
// ── the NaN direction is LEARNED, not fixed ───────────────────────────────
// The NaN bin is the highest bin index per feature, so a plain prefix scan
// would always leave it in the right child -- a fixed policy masquerading as an
// algorithm, and one that silently disagreed with tree_partition's own default.
// Instead each feature is scanned TWICE, once with the NaN bin's (G,H,n) added
// to the left accumulator and once without, and the winner carries its
// direction back in `nan_left`. This is XGBoost's sparsity-aware behaviour and
// it matters: imputing to the mean first is both lossy and a different model.
//
// The binned matrix is a parameter because the histogram alone cannot say which
// bin is the NaN bin -- that is per feature, and it lives with the bins.
//
// ── the hessian guard is this kernel's real test ──────────────────────────
// A saturated logistic gives h -> 0, so with lambda = 0 the term G^2/(H+lambda)
// divides by zero. A NaN gain would LOSE every `>` comparison, which sounds
// safe; an INFINITE gain wins every one of them, and would pick a split on a
// child with no weight in it. So a child whose H + lambda is not strictly
// positive is skipped before the division, not after it.
inline Value kernel_best_split(std::vector<Value> a) {
    needTreeArgs(a, 2, "tree_best_split(hist, binned, lambda?, gamma?, "
                       "min_child_weight?, min_data?, cuts?)");
    ArrayPtr Hst = asArray(a[0]);
    BinsPtr  B   = asBins(a[1], "tree_best_split");
    if (Hst->shape.size() != 3 || Hst->shape[2] != 3) {
        throw std::runtime_error("tree_best_split: hist must be "
            "[features, bins, 3] from tree_hist, got " + shapeStr(Hst->shape));
    }
    const size_t cols = Hst->shape[0], nb = Hst->shape[1];
    if (cols != B->cols) {
        throw std::runtime_error("tree_best_split: the histogram has " +
            std::to_string(cols) + " features and the binned matrix has " +
            std::to_string(B->cols));
    }
    if (nb < B->maxBins) {
        throw std::runtime_error("tree_best_split: the histogram is " +
            std::to_string(nb) + " bins wide but the binned matrix uses up to " +
            std::to_string(B->maxBins));
    }
    const double lam   = treeNum(a, 2, 1.0,  "tree_best_split", "lambda");
    const double gam   = treeNum(a, 3, 0.0,  "tree_best_split", "gamma");
    const double minCW = treeNum(a, 4, 1e-3, "tree_best_split", "min_child_weight");
    const double minN  = treeNum(a, 5, 1.0,  "tree_best_split", "min_data");
    if (lam < 0.0) {
        throw std::runtime_error("tree_best_split: lambda must be >= 0, got " +
            std::to_string(lam));
    }

    // Optional: one chosen bin per feature. When present, feature f considers
    // ONLY the cut at cuts[f] instead of scanning every bin -- which is all
    // ExtraTrees' "random split thresholds" amounts to (§8.9). It is an
    // argument rather than a second kernel on purpose: the gain formula, the
    // NaN double pass, the guards, the tie-break and the 10-slot return are
    // then inherited rather than reimplemented, and a randomly chosen cut needs
    // those guards MORE than a scanned one does, because it is far likelier to
    // leave a child empty or weightless.
    //
    // The draw itself stays in Bantu, so the forest is reproducible from
    // nd_seed alone and -- the part that matters for the gate -- this kernel
    // stays fully DETERMINISTIC: a test hands it a known vector and asserts the
    // exact split it must return.
    std::vector<long long> cuts;
    bool haveCuts = false;
    if (a.size() > 6 && !a[6].isNull()) {
        std::vector<double> cv;
        readVector(a[6], cols, cv, "tree_best_split", "cuts");
        cuts.resize(cols);
        for (size_t f = 0; f < cols; f++) {
            const double v = cv[f];
            const long long b = (long long)v;
            // No clamping. An index out of range is a caller error, and
            // rounding it into range would turn an off-by-one in the drawing
            // expression into a working forest with a biased cut distribution --
            // K48's failure mode exactly, a wrong answer that still looks right.
            const int nbf   = B->nbins[f];
            const int vbins = (B->hasNaN[f] != 0) ? nbf - 1 : nbf;
            if ((double)b != v) {
                throw std::runtime_error("tree_best_split: cuts[" +
                    std::to_string(f) + "] = " + std::to_string(v) +
                    " is not a whole number");
            }
            if (b < -1 || b + 1 >= (long long)vbins) {
                throw std::runtime_error("tree_best_split: cuts[" +
                    std::to_string(f) + "] = " + std::to_string(b) +
                    " is not a cut on a feature with " + std::to_string(vbins) +
                    " value bins (valid: -1.." + std::to_string(vbins - 2) + ")");
            }
            cuts[f] = b;
        }
        haveCuts = true;
    }

    std::vector<double> flat(cols * nb * 3);
    for (size_t f = 0; f < cols; f++)
        for (size_t b = 0; b < nb; b++)
            for (size_t c = 0; c < 3; c++)
                flat[(f * nb + b) * 3 + c] =
                    getAsDouble(*Hst, flatOffset(*Hst, {f, b, c}));

    double bestGain = -std::numeric_limits<double>::infinity();
    long long bestF = -1, bestB = -1;
    bool bestNanLeft = false;
    double bGL = 0, bHL = 0, bNL = 0, bGR = 0, bHR = 0, bNR = 0;

    for (size_t f = 0; f < cols; f++) {
        const int nbf       = B->nbins[f];
        const bool anyNaN   = B->hasNaN[f] != 0;
        const int valueBins = anyNaN ? nbf - 1 : nbf;
        const int nanIdx    = anyNaN ? nbf - 1 : -1;

        double G = 0, Hs = 0, N = 0;
        for (int b = 0; b < nbf; b++) {
            G  += flat[(f * nb + (size_t)b) * 3 + 0];
            Hs += flat[(f * nb + (size_t)b) * 3 + 1];
            N  += flat[(f * nb + (size_t)b) * 3 + 2];
        }
        if (!(Hs + lam > 0.0)) continue;        // also catches a NaN total
        const double parent = (G * G) / (Hs + lam);

        double Gn = 0, Hn = 0, Nn = 0;
        if (nanIdx >= 0) {
            Gn = flat[(f * nb + (size_t)nanIdx) * 3 + 0];
            Hn = flat[(f * nb + (size_t)nanIdx) * 3 + 1];
            Nn = flat[(f * nb + (size_t)nanIdx) * 3 + 2];
        }

        // pass 0: NaN to the right (or no NaN at all). pass 1: NaN to the left,
        // which also admits bin = -1, the missing-vs-present split.
        const int passes = anyNaN ? 2 : 1;
        for (int pass = 0; pass < passes; pass++) {
            const bool nanLeft = (pass == 1);
            double GL = nanLeft ? Gn : 0.0;
            double HL = nanLeft ? Hn : 0.0;
            double NL = nanLeft ? Nn : 0.0;
            const int firstCut = nanLeft ? -1 : 0;
            for (int b = firstCut; b + 1 < valueBins; b++) {
                if (b >= 0) {
                    GL += flat[(f * nb + (size_t)b) * 3 + 0];
                    HL += flat[(f * nb + (size_t)b) * 3 + 1];
                    NL += flat[(f * nb + (size_t)b) * 3 + 2];
                }
                // The prefix must still be accumulated past the chosen cut's
                // predecessors, so this skips the EVALUATION and not the loop:
                // `continue` after the accumulation above, never before it.
                if (haveCuts && b != (int)cuts[f]) continue;
                const double GR = G - GL, HR = Hs - HL, NR = N - NL;
                if (NL < minN || NR < minN) continue;
                if (HL < minCW || HR < minCW) continue;
                if (!(HL + lam > 0.0) || !(HR + lam > 0.0)) continue;
                const double gain = 0.5 * ((GL * GL) / (HL + lam)
                                         + (GR * GR) / (HR + lam)
                                         - parent) - gam;
                // > not >=, so the LOWEST (feature, bin) wins a tie and the
                // answer does not depend on iteration order. The NaN-right pass
                // runs first, so it also wins a tie against NaN-left.
                if (gain > bestGain) {
                    bestGain = gain; bestF = (long long)f; bestB = (long long)b;
                    bestNanLeft = nanLeft;
                    bGL = GL; bHL = HL; bNL = NL; bGR = GR; bHR = HR; bNR = NR;
                }
            }
        }
    }
    std::vector<double> r(10, 0.0);
    r[0] = (double)bestF;
    r[1] = (double)bestB;
    r[2] = (bestF < 0) ? 0.0 : bestGain;
    r[3] = bGL; r[4] = bHL; r[5] = bNL;
    r[6] = bGR; r[7] = bHR; r[8] = bNR;
    r[9] = bestNanLeft ? 1.0 : 0.0;
    return makeF64Vector(r);
}

// ── tree_partition ──────────────────────────────────────────────────────────
// Stable: both children keep the parent's relative order, so a run is
// reproducible and a later histogram-subtraction bug cannot hide behind a
// reshuffle. `nan_left` decides where the dedicated NaN bin goes.
inline Value kernel_partition(std::vector<Value> a) {
    needTreeArgs(a, 3, "tree_partition(binned, idx, feature, bin, nan_left?)");
    BinsPtr B = asBins(a[0], "tree_partition");
    std::vector<size_t> idx;
    readIndex(a[1], B->rows, idx, "tree_partition");
    const double fD = treeNum(a, 2, 0.0, "tree_partition", "feature");
    const double bD = treeNum(a, 3, 0.0, "tree_partition", "bin");
    if (fD != std::floor(fD) || fD < 0 || (size_t)fD >= B->cols) {
        throw std::runtime_error("tree_partition: feature " + std::to_string(fD) +
            " is out of range for " + std::to_string(B->cols) + " features");
    }
    // -1 is legitimate: it is the "no value bins on the left" cut that
    // tree_best_split returns for a missing-vs-present split, reachable only
    // together with nan_left.
    if (bD != std::floor(bD) || bD < -1) {
        throw std::runtime_error("tree_partition: bin must be a whole number "
            ">= -1, got " + std::to_string(bD));
    }
    const size_t f = (size_t)fD;
    const int cut = (int)bD;
    // NOT defaulted to true. The scan in tree_best_split decides which way the
    // NaN bin goes and returns it; a default here would silently disagree with
    // the gain that was used to choose the split, which is the kind of
    // mismatch that produces a tree whose own splits do not reproduce their
    // own gains. false matches the scan's first pass, so a caller that omits
    // it gets the NaN-right behaviour the prefix scan assumes.
    bool nanLeft = false;
    if (a.size() > 4 && !a[4].isNull()) nanLeft = a[4].isTruthy();
    const int nanBin = B->hasNaN[f] ? B->nbins[f] - 1 : -1;

    std::vector<size_t> L, Rt;
    L.reserve(idx.size());
    Rt.reserve(idx.size());
    for (size_t k = 0; k < idx.size(); k++) {
        const size_t i = idx[k];
        const int b = (int)B->bins[i * B->cols + f];
        bool left;
        if (b == nanBin) left = nanLeft;
        else             left = (b <= cut);
        if (left) L.push_back(i); else Rt.push_back(i);
    }
    Value out(std::vector<Value>{ makeI64Vector(L), makeI64Vector(Rt) });
    return out;
}

// ── tree_predict ────────────────────────────────────────────────────────────
// Traversal over a flat node array, SEVEN doubles per node:
//
//   [0] is_leaf   [1] feature   [2] threshold (a raw X value, not a bin)
//   [3] left      [4] right     [5] value     [6] nan_left
//
// Thresholds are raw values rather than bin indices on purpose: prediction then
// needs nothing but X and the node array -- no binned handle, no edge table, so
// a saved model is one array. It also means predict is exact on values the
// training set never contained, which a bin lookup would have to extrapolate.
inline Value kernel_predict(std::vector<Value> a) {
    needTreeArgs(a, 2, "tree_predict(nodes, X)");
    ArrayPtr Nd = asArray(a[0]);
    ArrayPtr X  = asArray(a[1]);
    if (Nd->shape.size() != 2 || Nd->shape[1] != 7) {
        throw std::runtime_error("tree_predict: nodes must be [n_nodes, 7] "
            "(is_leaf, feature, threshold, left, right, value, nan_left), got " +
            shapeStr(Nd->shape));
    }
    size_t rows = 0, cols = 0;
    std::vector<double> v;
    readMatrix(*X, v, rows, cols, "tree_predict");

    const size_t nn = Nd->shape[0];
    if (nn == 0) throw std::runtime_error("tree_predict: the tree has no nodes");
    std::vector<double> nodes(nn * 7);
    for (size_t i = 0; i < nn; i++)
        for (size_t j = 0; j < 7; j++)
            nodes[i * 7 + j] = getAsDouble(*Nd, flatOffset(*Nd, {i, j}));

    // Validate every node ONCE rather than per sample: a bad child index would
    // otherwise be a per-row bounds check in the hot loop, or a segfault.
    for (size_t i = 0; i < nn; i++) {
        if (nodes[i * 7 + 0] != 0.0) continue;
        const double fe = nodes[i * 7 + 1];
        const double l  = nodes[i * 7 + 3];
        const double r  = nodes[i * 7 + 4];
        if (fe != std::floor(fe) || fe < 0 || (size_t)fe >= cols) {
            throw std::runtime_error("tree_predict: node " + std::to_string(i) +
                " splits on feature " + std::to_string(fe) + ", but X has " +
                std::to_string(cols) + " features");
        }
        if (l != std::floor(l) || r != std::floor(r) ||
            l < 0 || r < 0 || (size_t)l >= nn || (size_t)r >= nn) {
            throw std::runtime_error("tree_predict: node " + std::to_string(i) +
                " has children (" + std::to_string(l) + "," + std::to_string(r) +
                ") outside a " + std::to_string(nn) + "-node tree");
        }
        if ((size_t)l == i || (size_t)r == i) {
            throw std::runtime_error("tree_predict: node " + std::to_string(i) +
                " is its own child, which would never terminate");
        }
    }

    ArrayPtr out = makeArray({rows}, DType::F64, "tree_predict");
    double* y = out->ptr<double>();
    for (size_t i = 0; i < rows; i++) {
        size_t at = 0;
        // A malformed tree could still cycle without a self-loop, so the walk
        // is bounded by the node count: a cycle raises instead of hanging.
        for (size_t step = 0; step <= nn; step++) {
            if (nodes[at * 7 + 0] != 0.0) { y[i] = nodes[at * 7 + 5]; break; }
            const double x = v[i * cols + (size_t)nodes[at * 7 + 1]];
            const bool goLeft = std::isnan(x) ? (nodes[at * 7 + 6] != 0.0)
                                              : (x <= nodes[at * 7 + 2]);
            at = (size_t)nodes[at * 7 + (goLeft ? 3 : 4)];
            if (step == nn) {
                throw std::runtime_error("tree_predict: the node array contains "
                    "a cycle -- traversal did not reach a leaf in " +
                    std::to_string(nn) + " steps");
            }
        }
    }
    return wrap(out);
}

// ── registration ────────────────────────────────────────────────────────────
inline void registerTreeBuiltins(const DefineFn& define) {
    registerHandleRepr(BINS_TAG, &reprBins);
    registerHandleLen(BINS_TAG, &lenBins);
    define("tree_bin",        kernel_bin);
    define("tree_bin_shape",  kernel_bin_shape);
    define("tree_bin_nbins",  kernel_bin_nbins);
    define("tree_bin_edges",  kernel_bin_edges);
    define("tree_bin_at",     kernel_bin_at);
    define("tree_bin_bytes",  kernel_bin_bytes);
    define("tree_hist",       kernel_hist);
    define("tree_best_split", kernel_best_split);
    define("tree_partition",  kernel_partition);
    define("tree_predict",    kernel_predict);
}

} // namespace tree
} // namespace numba
