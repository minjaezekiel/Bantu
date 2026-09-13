#pragma once
/**
 * numba — the linear-algebra builtin table.
 *
 * Included by ndarray_native.cpp only, after ndarray_api.hpp.
 */

#include "ndarray_linalg.hpp"
#include "ndarray_api.hpp"

namespace numba {

Value toArrayValue(const Value& v, const char* what);

// A right-hand side is either a vector (n) or a matrix (n, k). Both are handled
// as a matrix so there is one solve path rather than two.
inline Mat rhsToMat(const NdArray& b, size_t n, const char* what, bool& wasVector) {
    if (b.ndim() == 1) {
        if (b.shape[0] != n) {
            throw std::runtime_error(std::string(what) + ": the matrix is " + std::to_string(n) +
                "x" + std::to_string(n) + " but b has " + std::to_string(b.shape[0]) + " elements");
        }
        wasVector = true;
        Mat m(n, 1);
        size_t k = 0;
        forEachIndex(b, [&](size_t o, size_t) { m.a[k++] = getAsDouble(b, o); });
        return m;
    }
    require2D(b, what);
    if (b.shape[0] != n) {
        throw std::runtime_error(std::string(what) + ": the matrix is " + std::to_string(n) + "x" +
            std::to_string(n) + " but b has " + std::to_string(b.shape[0]) + " rows");
    }
    wasVector = false;
    return toMat(b);
}

inline void registerLinalg(const DefineFn& define) {

    // ── matmul / dot ─────────────────────────────────────────────────────────
    // nd_matmul handles (m,k)x(k,n), and promotes a 1-d operand to a row or
    // column as NumPy does, dropping that axis from the result.
    define("nd_matmul", [](std::vector<Value> args) -> Value {
        if (args.size() < 2) throw std::runtime_error("nd_matmul(a, b) needs two matrices");
        ArrayPtr A = asArray(toArrayValue(args[0], "nd_matmul"));
        ArrayPtr B = asArray(toArrayValue(args[1], "nd_matmul"));
        if (A->ndim() == 0 || B->ndim() == 0) {
            throw std::runtime_error("nd_matmul: a 0-d array is not a matrix -- use nd_multiply");
        }
        const bool aVec = (A->ndim() == 1), bVec = (B->ndim() == 1);
        if (A->ndim() > 2 || B->ndim() > 2) {
            throw std::runtime_error("nd_matmul: only 1- and 2-dimensional operands are supported "
                                     "(got " + shapeStr(A->shape) + " and " + shapeStr(B->shape) + ")");
        }
        Mat ma, mb;
        if (aVec) { ma = Mat(1, A->shape[0]); size_t k = 0;
                    forEachIndex(*A, [&](size_t o, size_t) { ma.a[k++] = getAsDouble(*A, o); }); }
        else ma = toMat(*A);
        if (bVec) { mb = Mat(B->shape[0], 1); size_t k = 0;
                    forEachIndex(*B, [&](size_t o, size_t) { mb.a[k++] = getAsDouble(*B, o); }); }
        else mb = toMat(*B);

        Mat c = matmul(ma, mb, "nd_matmul");
        if (aVec && bVec) {                       // an inner product: a 0-d scalar
            ArrayPtr out = makeArray({}, DType::F64, "nd_matmul");
            setFromDouble(*out, 0, c(0, 0));
            return wrap(out);
        }
        if (aVec || bVec) {                       // one axis collapses
            std::vector<double> v(c.a.begin(), c.a.end());
            return wrap(fromVec(v, "nd_matmul"));
        }
        return wrap(fromMat(c, "nd_matmul"));
    });
    define("nd_dot", [](std::vector<Value> args) -> Value {
        if (args.size() < 2) throw std::runtime_error("nd_dot(a, b) needs two arrays");
        ArrayPtr A = asArray(toArrayValue(args[0], "nd_dot"));
        ArrayPtr B = asArray(toArrayValue(args[1], "nd_dot"));
        // For two vectors this is the inner product, computed with the pairwise
        // accumulator so it is as accurate as nd_sum.
        if (A->ndim() == 1 && B->ndim() == 1) {
            if (A->shape[0] != B->shape[0]) {
                throw std::runtime_error("nd_dot: vectors of length " +
                    std::to_string(A->shape[0]) + " and " + std::to_string(B->shape[0]) +
                    " cannot be combined");
            }
            std::vector<size_t> ao, bo;
            forEachIndex(*A, [&](size_t o, size_t) { ao.push_back(o); });
            forEachIndex(*B, [&](size_t o, size_t) { bo.push_back(o); });
            Pairwise acc;
            for (size_t i = 0; i < ao.size(); i++)
                acc.add(getAsDouble(*A, ao[i]) * getAsDouble(*B, bo[i]));
            ArrayPtr out = makeArray({}, DType::F64, "nd_dot");
            setFromDouble(*out, 0, acc.total());
            return wrap(out);
        }
        throw std::runtime_error("nd_dot: for matrices use nd_matmul");
    });
    define("nd_outer", [](std::vector<Value> args) -> Value {
        if (args.size() < 2) throw std::runtime_error("nd_outer(a, b) needs two vectors");
        ArrayPtr A = asArray(toArrayValue(args[0], "nd_outer"));
        ArrayPtr B = asArray(toArrayValue(args[1], "nd_outer"));
        std::vector<double> av, bv;
        forEachIndex(*A, [&](size_t o, size_t) { av.push_back(getAsDouble(*A, o)); });
        forEachIndex(*B, [&](size_t o, size_t) { bv.push_back(getAsDouble(*B, o)); });
        Mat c(av.size(), bv.size());
        for (size_t i = 0; i < av.size(); i++)
            for (size_t j = 0; j < bv.size(); j++) c(i, j) = av[i] * bv[j];
        return wrap(fromMat(c, "nd_outer"));
    });

    define("nd_trace", [](std::vector<Value> args) -> Value {
        if (args.empty()) throw std::runtime_error("nd_trace(a) needs a matrix");
        ArrayPtr A = asArray(toArrayValue(args[0], "nd_trace"));
        require2D(*A, "nd_trace");
        Pairwise acc;
        const size_t n = std::min(A->shape[0], A->shape[1]);
        for (size_t i = 0; i < n; i++)
            acc.add(getAsDouble(*A, flatOffset(*A, { i, i })));
        ArrayPtr out = makeArray({}, DType::F64, "nd_trace");
        setFromDouble(*out, 0, acc.total());
        return wrap(out);
    });

    // ── LU-based: solve, inv, det, slogdet ───────────────────────────────────
    define("nd_solve", [](std::vector<Value> args) -> Value {
        if (args.size() < 2) throw std::runtime_error("nd_solve(a, b) needs a matrix and a right-hand side");
        ArrayPtr A = asArray(toArrayValue(args[0], "nd_solve"));
        ArrayPtr B = asArray(toArrayValue(args[1], "nd_solve"));
        requireSquare(*A, "nd_solve");
        const size_t n = A->shape[0];
        bool wasVec = false;
        Mat rhs = rhsToMat(*B, n, "nd_solve", wasVec);
        LU f = luFactor(toMat(*A));
        if (f.singular) {
            throw std::runtime_error("nd_solve: the matrix is singular to machine precision -- "
                                     "it has no unique solution. Try nd_lstsq for a "
                                     "least-squares answer, or nd_pinv");
        }
        Mat X(n, rhs.cols);
        std::vector<double> col(n);
        for (size_t c = 0; c < rhs.cols; c++) {
            for (size_t i = 0; i < n; i++) col[i] = rhs(i, c);
            luSolveVec(f, col);
            for (size_t i = 0; i < n; i++) X(i, c) = col[i];
        }
        if (wasVec) return wrap(fromVec(std::vector<double>(X.a.begin(), X.a.end()), "nd_solve"));
        return wrap(fromMat(X, "nd_solve"));
    });

    define("nd_inv", [](std::vector<Value> args) -> Value {
        if (args.empty()) throw std::runtime_error("nd_inv(a) needs a square matrix");
        ArrayPtr A = asArray(toArrayValue(args[0], "nd_inv"));
        requireSquare(*A, "nd_inv");
        const size_t n = A->shape[0];
        LU f = luFactor(toMat(*A));
        if (f.singular) {
            throw std::runtime_error("nd_inv: the matrix is singular to machine precision and has "
                                     "no inverse. Try nd_pinv for the pseudo-inverse");
        }
        Mat X(n, n);
        std::vector<double> e(n);
        for (size_t c = 0; c < n; c++) {
            std::fill(e.begin(), e.end(), 0.0);
            e[c] = 1.0;
            luSolveVec(f, e);
            for (size_t i = 0; i < n; i++) X(i, c) = e[i];
        }
        return wrap(fromMat(X, "nd_inv"));
    });

    define("nd_det", [](std::vector<Value> args) -> Value {
        if (args.empty()) throw std::runtime_error("nd_det(a) needs a square matrix");
        ArrayPtr A = asArray(toArrayValue(args[0], "nd_det"));
        requireSquare(*A, "nd_det");
        const size_t n = A->shape[0];
        LU f = luFactor(toMat(*A));
        double d = (double)f.sign;
        for (size_t i = 0; i < n; i++) d *= f.U(i, i);
        ArrayPtr out = makeArray({}, DType::F64, "nd_det");
        setFromDouble(*out, 0, d);
        return wrap(out);
    });

    // The determinant of a 500x500 overflows long before the matrix is
    // interesting, so slogdet returns (sign, log|det|) -- the form anyone doing
    // likelihoods actually needs.
    define("nd_slogdet", [](std::vector<Value> args) -> Value {
        if (args.empty()) throw std::runtime_error("nd_slogdet(a) needs a square matrix");
        ArrayPtr A = asArray(toArrayValue(args[0], "nd_slogdet"));
        requireSquare(*A, "nd_slogdet");
        const size_t n = A->shape[0];
        LU f = luFactor(toMat(*A));
        double sign = (double)f.sign, logAbs = 0.0;
        for (size_t i = 0; i < n; i++) {
            const double u = f.U(i, i);
            if (u == 0.0) { sign = 0.0; logAbs = -std::numeric_limits<double>::infinity(); break; }
            if (u < 0.0) sign = -sign;
            logAbs += std::log(std::fabs(u));
        }
        return wrap(fromVec({ sign, logAbs }, "nd_slogdet"));
    });

    // ── Cholesky ─────────────────────────────────────────────────────────────
    define("nd_cholesky", [](std::vector<Value> args) -> Value {
        if (args.empty()) throw std::runtime_error("nd_cholesky(a) needs a square matrix");
        ArrayPtr A = asArray(toArrayValue(args[0], "nd_cholesky"));
        requireSquare(*A, "nd_cholesky");
        return wrap(fromMat(cholesky(toMat(*A), "nd_cholesky"), "nd_cholesky"));
    });

    // ── QR and least squares ─────────────────────────────────────────────────
    define("nd_qr", [](std::vector<Value> args) -> Value {
        if (args.empty()) throw std::runtime_error("nd_qr(a) needs a matrix");
        ArrayPtr A = asArray(toArrayValue(args[0], "nd_qr"));
        require2D(*A, "nd_qr");
        const size_t m = A->shape[0], n = A->shape[1];
        QR f = qrFactor(toMat(*A));
        const size_t k = std::min(m, n);
        // Reduced form: Q is m-by-k, R is k-by-n. Q is built one column at a
        // time by applying the stored reflectors to a unit vector -- never as
        // an m-by-m matrix, which for a tall input would be enormous.
        Mat Q(m, k), R(k, n);
        std::vector<double> e(m);
        for (size_t j = 0; j < k; j++) {
            std::fill(e.begin(), e.end(), 0.0);
            e[j] = 1.0;
            applyQ(f, e);
            for (size_t i = 0; i < m; i++) Q(i, j) = e[i];
        }
        for (size_t i = 0; i < k; i++)
            for (size_t j = 0; j < n; j++) R(i, j) = (j >= i) ? f.R(i, j) : 0.0;
        std::vector<Value> pair = { wrap(fromMat(Q, "nd_qr")), wrap(fromMat(R, "nd_qr")) };
        return Value(pair);
    });

    define("nd_lstsq", [](std::vector<Value> args) -> Value {
        if (args.size() < 2) throw std::runtime_error("nd_lstsq(a, b) needs a matrix and a vector");
        ArrayPtr A = asArray(toArrayValue(args[0], "nd_lstsq"));
        ArrayPtr B = asArray(toArrayValue(args[1], "nd_lstsq"));
        require2D(*A, "nd_lstsq");
        const size_t m = A->shape[0], n = A->shape[1];
        if (B->ndim() != 1 || B->shape[0] != m) {
            throw std::runtime_error("nd_lstsq: b must be a vector with " + std::to_string(m) +
                " elements, one per row of a (got " + shapeStr(B->shape) + ")");
        }
        // Solved through QR rather than the normal equations: forming At*A
        // SQUARES the condition number, which throws away half the available
        // digits on anything even mildly ill-conditioned.
        QR f = qrFactor(toMat(*A));
        std::vector<double> qtb(m);
        size_t k = 0;
        forEachIndex(*B, [&](size_t o, size_t) { qtb[k++] = getAsDouble(*B, o); });
        // Q^T b by applying the reflectors, not by multiplying an m-by-m Q.
        applyQt(f, qtb);
        const size_t r = std::min(m, n);
        std::vector<double> x(n, 0.0);
        for (size_t i = r; i-- > 0; ) {
            double s = qtb[i];
            for (size_t j = i + 1; j < n; j++) s -= f.R(i, j) * x[j];
            if (f.R(i, i) == 0.0) {
                throw std::runtime_error("nd_lstsq: the matrix is rank-deficient -- "
                                         "use nd_pinv for the minimum-norm solution");
            }
            x[i] = s / f.R(i, i);
        }
        return wrap(fromVec(x, "nd_lstsq"));
    });

    // ── eigh, SVD, and what they give you ────────────────────────────────────
    define("nd_eigh", [](std::vector<Value> args) -> Value {
        if (args.empty()) throw std::runtime_error("nd_eigh(a) needs a symmetric matrix");
        ArrayPtr A = asArray(toArrayValue(args[0], "nd_eigh"));
        requireSquare(*A, "nd_eigh");
        Eigh e = eighJacobi(toMat(*A), "nd_eigh");
        std::vector<Value> pair = { wrap(fromVec(e.w, "nd_eigh")), wrap(fromMat(e.V, "nd_eigh")) };
        return Value(pair);
    });

    define("nd_svd", [](std::vector<Value> args) -> Value {
        if (args.empty()) throw std::runtime_error("nd_svd(a) needs a matrix");
        ArrayPtr A = asArray(toArrayValue(args[0], "nd_svd"));
        require2D(*A, "nd_svd");
        SVD s = svdJacobi(toMat(*A));
        std::vector<Value> three = {
            wrap(fromMat(s.U, "nd_svd")),
            wrap(fromVec(s.s, "nd_svd")),
            wrap(fromMat(s.V, "nd_svd"))
        };
        return Value(three);
    });

    define("nd_matrix_rank", [](std::vector<Value> args) -> Value {
        if (args.empty()) throw std::runtime_error("nd_matrix_rank(a) needs a matrix");
        ArrayPtr A = asArray(toArrayValue(args[0], "nd_matrix_rank"));
        require2D(*A, "nd_matrix_rank");
        SVD s = svdJacobi(toMat(*A));
        const double biggest = s.s.empty() ? 0.0 : s.s[0];
        // NumPy's default: max(m,n) * eps * largest singular value.
        const double tol = (double)std::max(A->shape[0], A->shape[1]) * 2.220446049250313e-16 * biggest;
        double r = 0;
        for (double v : s.s) if (v > tol) r += 1;
        ArrayPtr out = makeArray({}, DType::I64, "nd_matrix_rank");
        setFromDouble(*out, 0, r);
        return wrap(out);
    });

    define("nd_cond", [](std::vector<Value> args) -> Value {
        if (args.empty()) throw std::runtime_error("nd_cond(a) needs a matrix");
        ArrayPtr A = asArray(toArrayValue(args[0], "nd_cond"));
        require2D(*A, "nd_cond");
        SVD s = svdJacobi(toMat(*A));
        ArrayPtr out = makeArray({}, DType::F64, "nd_cond");
        if (s.s.empty()) { setFromDouble(*out, 0, 0.0); return wrap(out); }
        const double lo = s.s.back();
        setFromDouble(*out, 0, lo == 0.0 ? std::numeric_limits<double>::infinity()
                                         : s.s.front() / lo);
        return wrap(out);
    });

    define("nd_pinv", [](std::vector<Value> args) -> Value {
        if (args.empty()) throw std::runtime_error("nd_pinv(a) needs a matrix");
        ArrayPtr A = asArray(toArrayValue(args[0], "nd_pinv"));
        require2D(*A, "nd_pinv");
        const size_t m = A->shape[0], n = A->shape[1];
        SVD s = svdJacobi(toMat(*A));
        const double biggest = s.s.empty() ? 0.0 : s.s[0];
        const double tol = (double)std::max(m, n) * 2.220446049250313e-16 * biggest;
        const size_t k = s.s.size();
        Mat P(n, m);
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < m; j++) {
                double acc = 0.0;
                for (size_t t = 0; t < k; t++) {
                    if (s.s[t] <= tol) continue;
                    acc += s.V(i, t) * s.U(j, t) / s.s[t];
                }
                P(i, j) = acc;
            }
        }
        return wrap(fromMat(P, "nd_pinv"));
    });

    // ── norms ────────────────────────────────────────────────────────────────
    // nd_norm(a, ord?) -- "fro"/null Frobenius, 1, 2, "inf".
    define("nd_norm", [](std::vector<Value> args) -> Value {
        if (args.empty()) throw std::runtime_error("nd_norm(a, ord?) needs an array");
        ArrayPtr A = asArray(toArrayValue(args[0], "nd_norm"));
        std::string ord = "fro";
        if (args.size() > 1 && !args[1].isNull()) {
            ord = args[1].isString() ? args[1].stringVal
                                     : std::to_string((long long)args[1].numberVal);
        }
        ArrayPtr out = makeArray({}, DType::F64, "nd_norm");

        if (A->ndim() <= 1 || ord == "fro" || ord == "2") {
            if (A->ndim() == 2 && ord == "2") {
                // The spectral norm is the largest singular value, not the
                // Frobenius norm -- they differ, and conflating them is a
                // classic quiet error.
                SVD s = svdJacobi(toMat(*A));
                setFromDouble(*out, 0, s.s.empty() ? 0.0 : s.s[0]);
                return wrap(out);
            }
            Pairwise acc;
            const NdArray& X = *A;
            forEachIndex(X, [&](size_t o, size_t) {
                const double v = getAsDouble(X, o);
                acc.add(v * v);
            });
            setFromDouble(*out, 0, std::sqrt(acc.total()));
            return wrap(out);
        }
        require2D(*A, "nd_norm");
        const size_t m = A->shape[0], n = A->shape[1];
        if (ord == "1") {                       // max absolute column sum
            double best = 0.0;
            for (size_t j = 0; j < n; j++) {
                double s = 0.0;
                for (size_t i = 0; i < m; i++) s += std::fabs(getAsDouble(*A, flatOffset(*A, {i, j})));
                if (s > best) best = s;
            }
            setFromDouble(*out, 0, best);
            return wrap(out);
        }
        if (ord == "inf") {                     // max absolute row sum
            double best = 0.0;
            for (size_t i = 0; i < m; i++) {
                double s = 0.0;
                for (size_t j = 0; j < n; j++) s += std::fabs(getAsDouble(*A, flatOffset(*A, {i, j})));
                if (s > best) best = s;
            }
            setFromDouble(*out, 0, best);
            return wrap(out);
        }
        throw std::runtime_error("nd_norm: unknown order '" + ord +
                                 "' (use \"fro\", 1, 2 or \"inf\")");
    });
}

}   // namespace numba
