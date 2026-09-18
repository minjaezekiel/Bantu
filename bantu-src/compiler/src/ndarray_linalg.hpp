#pragma once
/**
 * numba — linear algebra.
 *
 * Included by ndarray_native.cpp only. docs/numba-architecture.md §3.1.
 *
 * All hand-written, no BLAS (N-of §3.2): an optional BLAS flag would gate
 * *speed* rather than *capability*, producing a two-tier contract where the
 * same program runs several times slower on the binary users actually
 * download. Every routine here is textbook numerics in 40-150 lines.
 *
 * Everything works on a private, C-contiguous f64 copy of its input. That is
 * deliberate: the factorisations destroy their working matrix, and taking a
 * copy once is cheaper and far less error-prone than making every kernel
 * stride-aware.
 */

#include "ndarray_reduce.hpp"

namespace numba {

// ── a plain dense matrix, row-major, for the factorisations to chew on ───────
// Working storage for the factorisations. It goes through the SAME admission
// test as an ndarray (N17), which it did not originally: std::vector allocates
// straight from the heap, so a linalg routine could ask the OS for hundreds of
// gigabytes and be SIGKILLed -- precisely the failure the ceiling exists to
// prevent, reached through the one path that skipped it. A 200,000x3 least
// squares did exactly that.
inline void admitMatrix(size_t r, size_t c, const char* what) {
    const size_t bytes = checkedMul(checkedMul(r, c, what), sizeof(double), what);
    const size_t soft = maxBytesRef().load(std::memory_order_relaxed);
    const size_t live = liveBytesRef().load(std::memory_order_relaxed);
    if (bytes > soft || live > soft - bytes) {
        std::ostringstream o;
        o << what << ": this needs a " << r << "x" << c << " working matrix (" << bytes
          << " bytes), which is over numba's " << soft << "-byte limit";
        throw std::runtime_error(o.str());
    }
}

struct Mat {
    size_t rows = 0, cols = 0;
    std::vector<double> a;
    Mat() = default;
    Mat(size_t r, size_t c) : rows(r), cols(c) {
        admitMatrix(r, c, "matrix");
        a.assign(checkedMul(r, c, "matrix"), 0.0);
    }
    double&       operator()(size_t i, size_t j)       { return a[i * cols + j]; }
    const double& operator()(size_t i, size_t j) const { return a[i * cols + j]; }
};

inline void require2D(const NdArray& x, const char* what) {
    if (x.ndim() != 2) {
        throw std::runtime_error(std::string(what) + ": expected a 2-dimensional matrix (got " +
                                 shapeStr(x.shape) + ")");
    }
}
inline void requireSquare(const NdArray& x, const char* what) {
    require2D(x, what);
    if (x.shape[0] != x.shape[1]) {
        throw std::runtime_error(std::string(what) + ": expected a square matrix (got " +
                                 shapeStr(x.shape) + ")");
    }
}

inline Mat toMat(const NdArray& x) {
    Mat m(x.shape[0], x.shape[1]);
    size_t k = 0;
    forEachIndex(x, [&](size_t o, size_t) { m.a[k++] = getAsDouble(x, o); });
    return m;
}

inline ArrayPtr fromMat(const Mat& m, const char* what) {
    ArrayPtr out = makeArray({ m.rows, m.cols }, DType::F64, what);
    std::memcpy(out->buf->data, m.a.data(), m.a.size() * sizeof(double));
    return out;
}

inline ArrayPtr fromVec(const std::vector<double>& v, const char* what) {
    ArrayPtr out = makeArray({ v.size() }, DType::F64, what);
    if (!v.empty()) std::memcpy(out->buf->data, v.data(), v.size() * sizeof(double));
    return out;
}

// ── matmul ───────────────────────────────────────────────────────────────────
// Blocked, with the inner order i-k-j so B streams at unit stride and C is
// accumulated in registers. The naive i-j-k order strides through B by `cols`
// on every innermost step, which is a cache miss per element once the matrix
// leaves L2 -- that ordering alone is worth more than any amount of unrolling.
inline Mat matmul(const Mat& A, const Mat& B, const char* what) {
    if (A.cols != B.rows) {
        throw std::runtime_error(std::string(what) + ": cannot multiply a " +
            std::to_string(A.rows) + "x" + std::to_string(A.cols) + " by a " +
            std::to_string(B.rows) + "x" + std::to_string(B.cols) +
            " (the inner dimensions must match: " + std::to_string(A.cols) + " vs " +
            std::to_string(B.rows) + ")");
    }
    Mat C(A.rows, B.cols);
    const size_t M = A.rows, K = A.cols, N = B.cols;
    const size_t BS = 64;                       // 64x64 f64 tiles: 32 KB, L1-resident
    for (size_t ii = 0; ii < M; ii += BS) {
        const size_t iMax = std::min(ii + BS, M);
        for (size_t kk = 0; kk < K; kk += BS) {
            const size_t kMax = std::min(kk + BS, K);
            for (size_t jj = 0; jj < N; jj += BS) {
                const size_t jMax = std::min(jj + BS, N);
                for (size_t i = ii; i < iMax; i++) {
                    for (size_t k = kk; k < kMax; k++) {
                        const double aik = A.a[i * K + k];
                        if (aik == 0.0) continue;
                        const double* NB_RESTRICT bp = &B.a[k * N];
                        double* NB_RESTRICT cp = &C.a[i * N];
                        for (size_t j = jj; j < jMax; j++) cp[j] += aik * bp[j];
                    }
                }
            }
        }
    }
    return C;
}

// ── LU with partial pivoting ─────────────────────────────────────────────────
// The workhorse: solve, inv, det, slogdet and matrix_rank all come from here.
struct LU {
    Mat U;                      // factors in place
    std::vector<size_t> piv;    // row i was swapped with piv[i]
    int sign = 1;               // for the determinant
    bool singular = false;
    double smallestPivot = 0.0;
};

inline LU luFactor(Mat A) {
    const size_t n = A.rows;
    LU r;
    r.piv.resize(n);
    double smallest = std::numeric_limits<double>::infinity();
    for (size_t k = 0; k < n; k++) {
        // Partial pivoting is not optional: without it a perfectly well
        // conditioned matrix with a zero in the corner fails outright, and a
        // small one loses most of its digits.
        size_t p = k;
        double best = std::fabs(A(k, k));
        for (size_t i = k + 1; i < n; i++) {
            const double v = std::fabs(A(i, k));
            if (v > best) { best = v; p = i; }
        }
        r.piv[k] = p;
        if (p != k) {
            for (size_t j = 0; j < n; j++) std::swap(A(k, j), A(p, j));
            r.sign = -r.sign;
        }
        const double pivot = A(k, k);
        if (std::fabs(pivot) < smallest) smallest = std::fabs(pivot);
        if (pivot == 0.0) { r.singular = true; continue; }
        for (size_t i = k + 1; i < n; i++) {
            const double f = A(i, k) / pivot;
            A(i, k) = f;
            if (f == 0.0) continue;
            for (size_t j = k + 1; j < n; j++) A(i, j) -= f * A(k, j);
        }
    }
    r.smallestPivot = (n == 0) ? 0.0 : smallest;
    r.U = std::move(A);
    return r;
}

// Solve in place for one right-hand side, using an existing factorisation.
inline void luSolveVec(const LU& f, std::vector<double>& b) {
    const size_t n = f.U.rows;
    for (size_t k = 0; k < n; k++) if (f.piv[k] != k) std::swap(b[k], b[f.piv[k]]);
    for (size_t i = 1; i < n; i++) {
        double s = b[i];
        for (size_t j = 0; j < i; j++) s -= f.U(i, j) * b[j];
        b[i] = s;
    }
    for (size_t i = n; i-- > 0; ) {
        double s = b[i];
        for (size_t j = i + 1; j < n; j++) s -= f.U(i, j) * b[j];
        b[i] = s / f.U(i, i);
    }
}

// ── Cholesky ─────────────────────────────────────────────────────────────────
// Lower-triangular L with A = L Lt. Fails loudly on a non-positive-definite
// matrix rather than returning NaN, because "is this matrix SPD" is exactly
// what a caller is usually asking.
inline Mat cholesky(const Mat& A, const char* what) {
    const size_t n = A.rows;
    Mat L(n, n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j <= i; j++) {
            double s = A(i, j);
            for (size_t k = 0; k < j; k++) s -= L(i, k) * L(j, k);
            if (i == j) {
                if (s <= 0.0) {
                    throw std::runtime_error(std::string(what) +
                        ": the matrix is not positive definite (it failed at row " +
                        std::to_string(i) + "). Cholesky needs a symmetric positive-definite "
                        "matrix; use nd_solve for a general one");
                }
                L(i, j) = std::sqrt(s);
            } else {
                L(i, j) = s / L(j, j);
            }
        }
    }
    return L;
}

// ── Householder QR ───────────────────────────────────────────────────────────
// Reduced QR: Q is m-by-k and R is k-by-k with k = min(m,n). Householder
// reflections rather than Gram-Schmidt, which loses orthogonality catastrophically
// on anything ill-conditioned -- the classic demonstration is that modified
// Gram-Schmidt on a Hilbert matrix produces a "Q" whose columns are visibly not
// orthogonal, while Householder stays at machine precision.
// The reflectors are STORED, not accumulated into an explicit Q.
//
// Accumulating Q needs an m-by-m matrix, and least squares is overwhelmingly
// used on TALL data -- 200,000 rows by 3 columns is an ordinary regression, and
// an explicit Q for it is 200,000^2 doubles, or 320 GB. That is not a slow path,
// it is an instant SIGKILL. Keeping the reflectors is O(m*k) instead, and any
// product with Q or Q-transpose is applied one reflector at a time. It is also
// what LAPACK does, for the same reason.
struct QR {
    Mat R;                              // m-by-n, upper triangular in its top k rows
    std::vector<std::vector<double>> v; // reflector k, entries k..m-1
    std::vector<double> vn;             // its squared norm
    size_t m = 0, n = 0;
};

inline QR qrFactor(Mat A) {
    const size_t m = A.rows, n = A.cols;
    QR r;
    r.m = m; r.n = n;
    const size_t steps = std::min(m ? m - 1 : 0, n);
    r.v.resize(steps);
    r.vn.assign(steps, 0.0);

    std::vector<double> v(m);
    for (size_t k = 0; k < steps; k++) {
        double norm = 0.0;
        for (size_t i = k; i < m; i++) norm += A(i, k) * A(i, k);
        norm = std::sqrt(norm);
        if (norm == 0.0) continue;
        // Sign chosen to move AWAY from cancellation: alpha = -sign(a_kk)*norm.
        const double alpha = (A(k, k) > 0.0) ? -norm : norm;
        for (size_t i = k; i < m; i++) v[i] = A(i, k);
        v[k] -= alpha;
        double vsq = 0.0;
        for (size_t i = k; i < m; i++) vsq += v[i] * v[i];
        if (vsq == 0.0) continue;

        for (size_t j = k; j < n; j++) {
            double s = 0.0;
            for (size_t i = k; i < m; i++) s += v[i] * A(i, j);
            s = 2.0 * s / vsq;
            for (size_t i = k; i < m; i++) A(i, j) -= s * v[i];
        }
        r.v[k].assign(v.begin() + (ptrdiff_t)k, v.begin() + (ptrdiff_t)m);
        r.vn[k] = vsq;
    }
    r.R = std::move(A);
    return r;
}

// b <- Q^T b, applying the reflectors in order. O(m*k), no m-by-m anything.
inline void applyQt(const QR& f, std::vector<double>& b) {
    for (size_t k = 0; k < f.v.size(); k++) {
        if (f.vn[k] == 0.0) continue;
        const std::vector<double>& v = f.v[k];
        double s = 0.0;
        for (size_t i = 0; i < v.size(); i++) s += v[i] * b[k + i];
        s = 2.0 * s / f.vn[k];
        for (size_t i = 0; i < v.size(); i++) b[k + i] -= s * v[i];
    }
}

// b <- Q b, applying the reflectors in REVERSE. Used to build the reduced Q
// one column at a time.
inline void applyQ(const QR& f, std::vector<double>& b) {
    for (size_t kk = f.v.size(); kk-- > 0; ) {
        if (f.vn[kk] == 0.0) continue;
        const std::vector<double>& v = f.v[kk];
        double s = 0.0;
        for (size_t i = 0; i < v.size(); i++) s += v[i] * b[kk + i];
        s = 2.0 * s / f.vn[kk];
        for (size_t i = 0; i < v.size(); i++) b[kk + i] -= s * v[i];
    }
}

// ── symmetric eigenproblem: cyclic Jacobi ────────────────────────────────────
// About 100 lines, unconditionally convergent, and orthogonality to machine
// precision. Slower than tridiagonal QR and bulletproof, which is the right
// trade for a library that cannot afford a subtly wrong answer.
struct Eigh {
    std::vector<double> w;   // eigenvalues, ascending
    Mat V;                   // eigenvectors in columns
};

inline Eigh eighJacobi(Mat A, const char* what) {
    const size_t n = A.rows;
    Mat V(n, n);
    for (size_t i = 0; i < n; i++) V(i, i) = 1.0;

    // A caller passing a non-symmetric matrix is asking the wrong question, and
    // silently symmetrising would answer a different one.
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            const double d = std::fabs(A(i, j) - A(j, i));
            const double s = std::fabs(A(i, j)) + std::fabs(A(j, i)) + 1.0;
            if (d > 1e-12 * s) {
                throw std::runtime_error(std::string(what) +
                    ": the matrix is not symmetric (element [" + std::to_string(i) + "," +
                    std::to_string(j) + "] is " + std::to_string(A(i, j)) + " but [" +
                    std::to_string(j) + "," + std::to_string(i) + "] is " +
                    std::to_string(A(j, i)) + ")");
            }
        }
    }

    for (int sweep = 0; sweep < 60; sweep++) {
        double off = 0.0;
        for (size_t i = 0; i < n; i++)
            for (size_t j = i + 1; j < n; j++) off += A(i, j) * A(i, j);
        if (off <= 1e-30) break;

        for (size_t p = 0; p < n; p++) {
            for (size_t q = p + 1; q < n; q++) {
                const double apq = A(p, q);
                if (std::fabs(apq) < 1e-300) continue;
                const double theta = (A(q, q) - A(p, p)) / (2.0 * apq);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                                 (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                for (size_t k = 0; k < n; k++) {
                    const double akp = A(k, p), akq = A(k, q);
                    A(k, p) = c * akp - s * akq;
                    A(k, q) = s * akp + c * akq;
                }
                for (size_t k = 0; k < n; k++) {
                    const double apk = A(p, k), aqk = A(q, k);
                    A(p, k) = c * apk - s * aqk;
                    A(q, k) = s * apk + c * aqk;
                }
                for (size_t k = 0; k < n; k++) {
                    const double vkp = V(k, p), vkq = V(k, q);
                    V(k, p) = c * vkp - s * vkq;
                    V(k, q) = s * vkp + c * vkq;
                }
            }
        }
    }

    Eigh r;
    r.w.resize(n);
    for (size_t i = 0; i < n; i++) r.w[i] = A(i, i);
    // Ascending, with the eigenvectors carried along -- NumPy's eigh ordering.
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; i++) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return r.w[a] < r.w[b]; });
    Eigh out;
    out.w.resize(n);
    out.V = Mat(n, n);
    for (size_t j = 0; j < n; j++) {
        out.w[j] = r.w[idx[j]];
        for (size_t i = 0; i < n; i++) out.V(i, j) = V(i, idx[j]);
    }
    return out;
}

// ── SVD: one-sided Jacobi ────────────────────────────────────────────────────
// ~120 lines against roughly 500 for Golub-Kahan implicit-QR, no convergence
// tuning, and HIGHER relative accuracy on small singular values. It costs
// several sweeps, so it is perhaps 3-5x slower than LAPACK's dgesdd and
// impractical much beyond n ~ 500-800. That limit is documented, not hidden.
struct SVD {
    Mat U;                   // m-by-k
    std::vector<double> s;   // k singular values, descending
    Mat V;                   // n-by-k
};

inline SVD svdJacobi(Mat A) {
    const size_t m = A.rows, n = A.cols;
    Mat V(n, n);
    for (size_t i = 0; i < n; i++) V(i, i) = 1.0;

    for (int sweep = 0; sweep < 60; sweep++) {
        double maxOff = 0.0;
        for (size_t p = 0; p < n; p++) {
            for (size_t q = p + 1; q < n; q++) {
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (size_t i = 0; i < m; i++) {
                    const double ap = A(i, p), aq = A(i, q);
                    alpha += ap * ap; beta += aq * aq; gamma += ap * aq;
                }
                if (alpha * beta > 0.0) {
                    const double off = std::fabs(gamma) / std::sqrt(alpha * beta);
                    if (off > maxOff) maxOff = off;
                }
                if (std::fabs(gamma) < 1e-300) continue;
                const double zeta = (beta - alpha) / (2.0 * gamma);
                const double t = (zeta >= 0.0 ? 1.0 : -1.0) /
                                 (std::fabs(zeta) + std::sqrt(1.0 + zeta * zeta));
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = c * t;
                for (size_t i = 0; i < m; i++) {
                    const double ap = A(i, p), aq = A(i, q);
                    A(i, p) = c * ap - s * aq;
                    A(i, q) = s * ap + c * aq;
                }
                for (size_t i = 0; i < n; i++) {
                    const double vp = V(i, p), vq = V(i, q);
                    V(i, p) = c * vp - s * vq;
                    V(i, q) = s * vp + c * vq;
                }
            }
        }
        if (maxOff < 1e-15) break;
    }

    const size_t k = std::min(m, n);
    std::vector<double> sv(n);
    for (size_t j = 0; j < n; j++) {
        double nrm = 0.0;
        for (size_t i = 0; i < m; i++) nrm += A(i, j) * A(i, j);
        sv[j] = std::sqrt(nrm);
    }
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; i++) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return sv[a] > sv[b]; });

    SVD r;
    r.U = Mat(m, k);
    r.V = Mat(n, k);
    r.s.resize(k);
    for (size_t j = 0; j < k; j++) {
        const size_t c = idx[j];
        r.s[j] = sv[c];
        if (sv[c] > 0.0) {
            for (size_t i = 0; i < m; i++) r.U(i, j) = A(i, c) / sv[c];
        }
        for (size_t i = 0; i < n; i++) r.V(i, j) = V(i, c);
    }
    return r;
}

}   // namespace numba
