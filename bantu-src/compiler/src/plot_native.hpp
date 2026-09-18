#pragma once
// ════════════════════════════════════════════════════════════════════════════
//  plot_native.hpp — bplot's native kernels (decision BP33).
//
//  bplot is pure Bantu, and it stays pure Bantu for everything that scales with
//  the size of the CHART. These kernels exist for the one thing that scales
//  with the size of the DATA: turning a long series into the handful of points
//  a canvas can actually show. Measured before they existed, one
//  1,000,000-point line took ~20 s and 3.1 GB of resident memory, almost all of
//  it spent materialising and copying Bantu lists (docs/bplot-architecture.md
//  §13).
//
//  THE RULE THAT MAKES THEM SAFE TO SUBSTITUTE: they are handed PIXEL
//  coordinates, not data, and they do no floating-point arithmetic that could
//  round differently from the interpreter. The scale transform and the affine
//  map `x * a + b` happen before, in Bantu, as separate numba passes -- a C++
//  compiler is allowed to fuse a multiply and an add into one FMA instruction,
//  which rounds differently in the last bit, and a different last bit is a
//  different two-decimal string on a rounding boundary. Two separate kernel
//  calls cannot be fused. What is left here is comparisons, std::round (which
//  is what Bantu's round() calls), and in _fmt2 below operations that are
//  exact on the values they see.
//
//  Both kernels therefore reproduce the pure-Bantu path byte for byte, and
//  tests/bplot_data_test.b holds them to it by rendering the same data both
//  ways.
//
//  This header sees ndarrays only through numba's exportVector(), the same
//  wall the arctic bridge respects, so it cannot reach numba's internals.
// ════════════════════════════════════════════════════════════════════════════

#include "types.hpp"
#include "ndarray_api.hpp"

#include <cmath>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace bplot_native {

using DefineFn = std::function<void(const char* name, NativeFn fn)>;

inline void readVector(const Value& v, std::vector<double>& out, const char* fn, const char* what) {
    int dt = 0;
    if (!numba::exportVector(v, out, dt)) {
        throw std::runtime_error(std::string(fn) + ": " + what + " must be a 1-dimensional ndarray");
    }
}

// ── BP3, verbatim ───────────────────────────────────────────────────────────
// bplot.b's _colPoints: first, then min, max and last where they add
// information. `x` is the pixel COLUMN (an integer), not the original x.
inline void colPoints(std::vector<Value>& out, double x, double first, double lo, double hi, double last) {
    out.emplace_back(x); out.emplace_back(first);
    if (lo != first && lo != last)             { out.emplace_back(x); out.emplace_back(lo); }
    if (hi != first && hi != last && hi != lo) { out.emplace_back(x); out.emplace_back(hi); }
    if (last != first)                         { out.emplace_back(x); out.emplace_back(last); }
}

// bplot.b's _simplify applied to one run of finite points [begin, end).
// Below 1,000 points, or with simplification off, the run is returned whole.
inline Value simplifyRun(const std::vector<double>& px, const std::vector<double>& py,
                         const std::vector<size_t>& idx, bool on) {
    std::vector<Value> out;
    const size_t n = idx.size();
    if (!on || n < 1000) {
        out.reserve(n * 2);
        for (size_t k = 0; k < n; k++) { out.emplace_back(px[idx[k]]); out.emplace_back(py[idx[k]]); }
        return Value(std::move(out));
    }
    double cx = 0, first = 0, last = 0, lo = 0, hi = 0;
    bool have = false;
    for (size_t k = 0; k < n; k++) {
        const double y = py[idx[k]];
        const double col = std::round(px[idx[k]]);
        if (!have) {
            cx = col; first = y; last = y; lo = y; hi = y; have = true;
        } else {
            if (col != cx) {
                colPoints(out, cx, first, lo, hi, last);
                cx = col; first = y; lo = y; hi = y;
            }
            if (y < lo) lo = y;
            if (y > hi) hi = y;
            last = y;
        }
    }
    if (have) colPoints(out, cx, first, lo, hi, last);
    return Value(std::move(out));
}

// ── _fmt(v, 2), verbatim ────────────────────────────────────────────────────
// Every operation here is either a single multiply or division (nothing to
// fuse) or exact on the integral doubles it sees: `ip * 100` and
// `units - ip * 100` stay far below 2^53, so a fused multiply-subtract gives
// the same result as two separate operations.
inline void appendIntegral(std::string& s, double v) {
    // Value::toString for an integral double below 9.2e18.
    s += std::to_string((long long)v);
}

inline void fmt2(std::string& s, double x) {
    if (std::isnan(x)) { s += "NaN"; return; }
    if (std::isinf(x)) { s += (x > 0 ? "inf" : "-inf"); return; }
    const bool neg = x < 0;
    const double a = std::fabs(x);
    if (a >= 1000000000000000.0) { s += Value(x).toString(); return; }
    const double scale = 100.0;
    const double units = std::round(a * scale);
    const double ip = std::floor(units / scale);
    const double fp = units - ip * scale;
    std::string body;
    appendIntegral(body, ip);
    body += '.';
    std::string fs;
    appendIntegral(fs, fp);
    while (fs.size() < 2) fs.insert(fs.begin(), '0');
    body += fs;
    if (neg && units > 0) s += '-';
    s += body;
}

inline void registerBuiltins(const DefineFn& define) {

    // bp_line_runs(px, py, finite, simplify) -> [run, run, ...]
    //
    // `finite` marks the points whose DATA was finite -- computed in Bantu on
    // the projected data, not re-derived here from the pixels, because a
    // finite value near 1e308 can overflow to an infinite pixel and the Bantu
    // path splits on the data. Each run is a flat [x0, y0, x1, y1, ...]; a
    // two-element run is a lone point, drawn as a dot.
    define("bp_line_runs", [](std::vector<Value> a) -> Value {
        if (a.size() < 3) throw std::runtime_error("bp_line_runs(px, py, finite, simplify) needs three arrays");
        std::vector<double> px, py, fin;
        readVector(a[0], px,  "bp_line_runs", "px");
        readVector(a[1], py,  "bp_line_runs", "py");
        readVector(a[2], fin, "bp_line_runs", "finite");
        if (px.size() != py.size() || px.size() != fin.size()) {
            throw std::runtime_error("bp_line_runs: px, py and finite must be the same length, got " +
                std::to_string(px.size()) + ", " + std::to_string(py.size()) + " and " +
                std::to_string(fin.size()));
        }
        const bool on = a.size() > 3 && a[3].isTruthy();
        std::vector<Value> runs;
        std::vector<size_t> run;
        auto flush = [&]() {
            if (run.size() >= 2) runs.push_back(simplifyRun(px, py, run, on));
            else if (run.size() == 1) runs.push_back(Value(std::vector<Value>{ Value(px[run[0]]), Value(py[run[0]]) }));
            run.clear();
        };
        for (size_t i = 0; i < px.size(); i++) {
            if (fin[i] != 0.0) run.push_back(i);
            else flush();
        }
        flush();
        return Value(std::move(runs));
    });

    // bp_scatter_path(px, py, finite, r) -> the `d` of one <path> holding a
    // circle for every finite point, or "" when there are none.
    //
    // One element for the whole series instead of one <circle> each: at a
    // million points that is the difference between a document a browser can
    // lay out and one it cannot. Each circle is two half-arcs, the smallest
    // exact circle a path can express. Used at 1,000 points and above, BP3's
    // own threshold, so no existing smaller document changes.
    define("bp_scatter_path", [](std::vector<Value> a) -> Value {
        if (a.size() < 4) throw std::runtime_error("bp_scatter_path(px, py, finite, r) needs four arguments");
        std::vector<double> px, py, fin;
        readVector(a[0], px,  "bp_scatter_path", "px");
        readVector(a[1], py,  "bp_scatter_path", "py");
        readVector(a[2], fin, "bp_scatter_path", "finite");
        if (px.size() != py.size() || px.size() != fin.size()) {
            throw std::runtime_error("bp_scatter_path: px, py and finite must be the same length");
        }
        if (!a[3].isNumber()) throw std::runtime_error("bp_scatter_path: r must be a number");
        const double r = a[3].numberVal;
        // Built once, outside the loop, exactly as bplot.b builds them.
        std::string rs;  fmt2(rs, r);
        std::string d2;  fmt2(d2, r + r);
        std::string md2; fmt2(md2, -(r + r));
        std::string arc  = "a" + rs + "," + rs + " 0 1,0 ";
        std::string d;
        d.reserve(px.size() * 44);
        for (size_t i = 0; i < px.size(); i++) {
            if (fin[i] == 0.0) continue;
            d += 'M'; fmt2(d, px[i] - r); d += ','; fmt2(d, py[i]);
            d += arc; d += d2;  d += ",0";
            d += arc; d += md2; d += ",0";
        }
        return Value(d);
    });

    // bp_grid_paths(rows, vmin, vmax, xpx, ypx) -> {colour index: path d}
    //
    // bplot.b's _drawGridCells loop, verbatim: quantise each cell to one of 256
    // colours (or -1 for no data), merge horizontal runs of one colour, and add
    // each run as an `M x y h w v h h -w Z` subpath to its colour's path. The
    // cell EDGES arrive in pixels, transformed in Bantu, so nothing here is a
    // multiply followed by an add that a compiler could fuse. A 256x256 grid of
    // real data is ~30,000 runs: 7 s interpreted, milliseconds here.
    define("bp_grid_paths", [](std::vector<Value> a) -> Value {
        if (a.size() < 5 || !a[0].isList() || !a[1].isNumber() || !a[2].isNumber() ||
            !a[3].isList() || !a[4].isList()) {
            throw std::runtime_error("bp_grid_paths(rows, vmin, vmax, xpx, ypx): rows and edges are lists, vmin and vmax numbers");
        }
        const std::vector<Value>& rows = a[0].listVal;
        const double vmin = a[1].numberVal, vmax = a[2].numberVal;
        auto edges = [](const Value& v, const char* what) {
            std::vector<double> e;
            e.reserve(v.listVal.size());
            for (const Value& x : v.listVal) {
                if (!x.isNumber()) throw std::runtime_error(std::string("bp_grid_paths: every ") + what + " edge must be a number");
                e.push_back(x.numberVal);
            }
            return e;
        };
        const std::vector<double> xe = edges(a[3], "x"), ye = edges(a[4], "y");
        const size_t h = rows.size();
        if (ye.size() != h + 1) throw std::runtime_error("bp_grid_paths: needs one more y edge than rows");
        // _cellIndex: -1 for no data, 128 for a flat range, else round(clamp(t) * 255).
        auto cellIndex = [&](const Value& v) -> int {
            if (!v.isNumber() || !std::isfinite(v.numberVal)) return -1;
            if (vmax <= vmin) return 128;
            double t = (v.numberVal - vmin) / (vmax - vmin);
            t = t < 0 ? 0 : (t > 1 ? 1 : t);
            return (int)std::round(t * 255);
        };
        std::map<int, std::string> buckets;
        for (size_t r = 0; r < h; r++) {
            if (!rows[r].isList()) throw std::runtime_error("bp_grid_paths: every row must be a list");
            const std::vector<Value>& row = rows[r].listVal;
            const size_t w = row.size();
            if (xe.size() != w + 1) throw std::runtime_error("bp_grid_paths: needs one more x edge than columns");
            const double yA = ye[r], yB = ye[r + 1];
            size_t c = 0;
            while (c < w) {
                const int idx = cellIndex(row[c]);
                size_t c2 = c + 1;
                while (c2 < w && cellIndex(row[c2]) == idx) c2++;
                // _cellRect, with its hairline overlap.
                const double xA = xe[c], xB = xe[c2];
                const double x = std::min(xA, xB), y = std::min(yA, yB);
                const double rw = std::fabs(xB - xA) + 0.5, rh = std::fabs(yB - yA) + 0.5;
                std::string& d = buckets[idx];
                d += 'M'; fmt2(d, x); d += ' '; fmt2(d, y);
                d += 'h'; fmt2(d, rw); d += 'v'; fmt2(d, rh);
                d += 'h'; fmt2(d, 0.0 - rw); d += 'Z';
                c = c2;
            }
        }
        ObjectMap out;
        for (auto& kv : buckets) out[std::to_string(kv.first)] = Value(std::move(kv.second));
        return Value(std::move(out));
    });

    // bp_escape(s) -> s with & < > " ' escaped and control bytes other than
    // tab, LF and CR removed.
    //
    // Exactly bplot.b's _esc (decision BP7), which stays the definition and
    // the fallback. _esc walks every character in interpreted Bantu to find
    // control bytes, which is fine for a label and minutes for a path holding
    // a million circles. There is still no way to emit an attribute WITHOUT
    // escaping it; this only makes the mandatory step cheap.
    define("bp_escape", [](std::vector<Value> a) -> Value {
        const std::string s = a.empty() ? std::string() : (a[0].isString() ? a[0].stringVal : a[0].toString());
        std::string o;
        o.reserve(s.size() + 16);
        for (unsigned char c : s) {
            switch (c) {
                case '&':  o += "&amp;";  break;
                case '<':  o += "&lt;";   break;
                case '>':  o += "&gt;";   break;
                case '"':  o += "&quot;"; break;
                case '\'': o += "&#39;";  break;
                default:
                    if (c < 32 && c != 9 && c != 10 && c != 13) break;
                    o += (char)c;
            }
        }
        return Value(o);
    });
}

} // namespace bplot_native
