#pragma once
/**
 * numba — Numerical Bantu: the native n-dimensional array.
 *
 * Design and rationale: docs/numba-architecture.md; decisions N1-N16 in
 * numba-suite/DECISIONS.md. The short version:
 *
 *   The interpreter costs ~1 us per element and a Value is ~190 bytes carrying
 *   a string, a vector, a std::function and three shared_ptrs at once, so any
 *   element loop written in Bantu is a toy. The kernels are native; the library
 *   on top is Bantu. That is what NumPy is, and it is the split `arctic`
 *   already proved here (decision A1).
 *
 *   This is a NEW type rather than an extended arctic::Column because Column
 *   stores its data in std::vector, which owns its allocation and cannot alias
 *   memory it does not own -- so zero-copy views, the property that makes an
 *   ndarray library worth having, are structurally impossible there (N2).
 *
 * This header is the implementation. It is included by exactly ONE translation
 * unit, ndarray_native.cpp, which each build script compiles at -O3 while the
 * rest of the interpreter stays at -O2. That is not a micro-optimization: the
 * production Linux binary is built in ubuntu:22.04, and GCC 11 does not enable
 * -ftree-loop-vectorize at -O2 (GCC 12 does), while the macOS build's Apple
 * Clang does -- so without the override a kernel benchmarked on a Mac would get
 * NEON and the same kernel in the shipped Linux binary would get a scalar loop
 * (N11). evaluator.hpp sees only ndarray_api.hpp.
 */

#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
    #include <malloc.h>
#endif

namespace numba {

// The tag stored in a NATIVE_HANDLE Value for arrays.
static const char* NDARRAY_TAG = "ndarray";

// ── dtypes ───────────────────────────────────────────────────────────────────
// Three, deliberately (N5):
//   F64  — the workhorse; Bantu numbers are double, so this is the lossless one
//   I64  — exact integers: indices, counts, argsort/argmin results, histograms
//   BOOL — one BYTE per element, NOT a bitmask: a bitmask cannot carry a
//          stride, so boolean views and transposes would be impossible
// No validity mask: NaN is the null for f64, and nulls stay arctic's concern.
enum class DType : uint8_t { F64, I64, BOOL };

inline size_t itemsize(DType d) { return d == DType::BOOL ? 1 : 8; }

inline const char* dtypeName(DType d) {
    switch (d) {
        case DType::F64:  return "f64";
        case DType::I64:  return "i64";
        case DType::BOOL: return "bool";
    }
    return "?";
}

inline DType dtypeFromName(const std::string& s) {
    if (s == "f64" || s == "float" || s == "double") return DType::F64;
    if (s == "i64" || s == "int"   || s == "long")   return DType::I64;
    if (s == "bool")                                 return DType::BOOL;
    throw std::runtime_error("unknown dtype '" + s + "' (use \"f64\", \"i64\" or \"bool\")");
}

// Promotion, applied everywhere: bool -> i64 -> f64.
inline DType promote(DType a, DType b) {
    if (a == DType::F64 || b == DType::F64) return DType::F64;
    if (a == DType::I64 || b == DType::I64) return DType::I64;
    return DType::BOOL;
}

// ── allocation limits ────────────────────────────────────────────────────────
// A single bad argument must not be able to take the process down. This matters
// most when numba runs inside a sua request handler, where the process is a
// server: nd_zeros([1e15]) has to raise, not OOM-kill the worker (N15). The
// ceiling is generous by default and adjustable from Bantu via nd_max_bytes().
inline size_t& maxBytesRef() {
    static size_t cap = (size_t)2 * 1024 * 1024 * 1024;   // 2 GiB
    return cap;
}

// Multiply with overflow detection. shape [2^22, 2^22, 2^22] silently wraps to
// a small number under plain multiplication, which would allocate a tiny buffer
// that every later kernel writes past -- a heap overflow reachable from one
// line of user script.
inline size_t checkedMul(size_t a, size_t b, const char* what) {
    if (a == 0 || b == 0) return 0;
    if (a > SIZE_MAX / b) {
        throw std::runtime_error(std::string(what) + ": size overflows (the shape is too large "
                                 "to describe, let alone allocate)");
    }
    return a * b;
}

inline size_t shapeProduct(const std::vector<size_t>& shape, const char* what) {
    size_t n = 1;
    for (size_t d : shape) n = checkedMul(n, d, what);
    return n;
}

inline std::string shapeStr(const std::vector<size_t>& s) {
    std::ostringstream o;
    o << "[";
    for (size_t i = 0; i < s.size(); i++) { if (i) o << ","; o << s[i]; }
    o << "]";
    return o.str();
}

// ── the refcounted data buffer ───────────────────────────────────────────────
// Split out from NdArray so that reshape/transpose/slice/flip can produce a NEW
// NdArray that SHARES this Buffer -- that is a zero-copy view. The shared_ptr
// keeps the base alive as long as any view of it lives, which is the same RAII
// argument that made decision A3 choose NATIVE_HANDLE in the first place.
inline void* alignedAlloc64(size_t bytes) {
#ifdef _WIN32
    return _aligned_malloc(bytes ? bytes : 64, 64);
#else
    void* p = nullptr;
    // posix_memalign rather than std::aligned_alloc: the latter requires the
    // size to be a multiple of the alignment, which ours rarely is.
    if (posix_memalign(&p, 64, bytes ? bytes : 64) != 0) return nullptr;
    return p;
#endif
}
inline void alignedFree64(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}

struct Buffer {
    void*  data   = nullptr;
    size_t nbytes = 0;
    bool   owned  = true;
    // Keeps a foreign owner alive when we BORROW its memory -- how an arctic
    // column's storage is wrapped zero-copy. Null for buffers we allocated.
    std::shared_ptr<void> keepalive;

    explicit Buffer(size_t bytes) : nbytes(bytes) {
        if (bytes > maxBytesRef()) {
            std::ostringstream o;
            o << "allocation of " << bytes << " bytes exceeds the limit of "
              << maxBytesRef() << " (raise it with nd_max_bytes(n) if you meant it)";
            throw std::runtime_error(o.str());
        }
        // 64-byte alignment lets the contiguous kernels vectorize without a
        // scalar peel prologue, and keeps two arrays off the same cache line.
        data = alignedAlloc64(bytes);
        if (!data) throw std::runtime_error("out of memory allocating " +
                                            std::to_string(bytes) + " bytes");
        std::memset(data, 0, bytes ? bytes : 64);
    }
    Buffer(void* p, size_t bytes, std::shared_ptr<void> owner)
        : data(p), nbytes(bytes), owned(false), keepalive(std::move(owner)) {}

    ~Buffer() { if (owned && data) alignedFree64(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
using BufferPtr = std::shared_ptr<Buffer>;

// ── the array ────────────────────────────────────────────────────────────────
struct NdArray {
    BufferPtr              buf;
    DType                  dtype  = DType::F64;
    size_t                 offset = 0;       // in ELEMENTS from buf->data
    std::vector<size_t>    shape;            // {} is a 0-d scalar
    std::vector<ptrdiff_t> strides;          // in ELEMENTS, SIGNED (negative = flipped)
    bool                   writable = true;  // false for broadcast_to / borrowed

    size_t ndim() const { return shape.size(); }
    size_t size() const {
        size_t n = 1;
        for (size_t d : shape) n *= d;       // already validated at construction
        return n;
    }

    template <class T> T*       ptr()       { return static_cast<T*>(buf->data) + offset; }
    template <class T> const T* ptr() const { return static_cast<const T*>(buf->data) + offset; }

    // Computed, NOT cached (N4). A cached flag has to be recomputed by every
    // operation that touches shape or strides, and one missed update leaves a
    // stale `true`, which sends a kernel down the raw-pointer fast path over
    // strided data -- silent memory corruption with no crash. Computing it is
    // ndim integer comparisons against a kernel that then touches millions of
    // elements, so this is not a place to trade correctness for nanoseconds.
    bool isCContig() const {
        ptrdiff_t expect = 1;
        for (ptrdiff_t d = (ptrdiff_t)shape.size() - 1; d >= 0; d--) {
            if (shape[d] == 0) return true;                          // empty: trivially so
            if (shape[d] != 1 && strides[d] != expect) return false;  // relaxed strides
            expect *= (ptrdiff_t)shape[d];
        }
        return true;
    }
    bool isFContig() const {
        ptrdiff_t expect = 1;
        for (size_t d = 0; d < shape.size(); d++) {
            if (shape[d] == 0) return true;
            if (shape[d] != 1 && strides[d] != expect) return false;
            expect *= (ptrdiff_t)shape[d];
        }
        return true;
    }
    // True when this array is the sole, complete owner of its buffer -- i.e.
    // it is a base array rather than a view of one.
    bool ownsAll() const {
        return buf && buf->owned && offset == 0 &&
               size() * itemsize(dtype) == buf->nbytes;
    }
};
using ArrayPtr = std::shared_ptr<NdArray>;

// C-contiguous strides for a shape.
inline std::vector<ptrdiff_t> cStrides(const std::vector<size_t>& shape) {
    std::vector<ptrdiff_t> s(shape.size());
    ptrdiff_t acc = 1;
    for (ptrdiff_t d = (ptrdiff_t)shape.size() - 1; d >= 0; d--) {
        s[d] = acc;
        acc *= (ptrdiff_t)shape[d];
    }
    return s;
}

// A fresh, zero-filled, C-contiguous array.
inline ArrayPtr makeArray(const std::vector<size_t>& shape, DType dt, const char* what) {
    auto a = std::make_shared<NdArray>();
    a->dtype   = dt;
    a->shape   = shape;
    a->strides = cStrides(shape);
    const size_t n     = shapeProduct(shape, what);
    const size_t bytes = checkedMul(n, itemsize(dt), what);
    a->buf = std::make_shared<Buffer>(bytes);
    return a;
}

// A view sharing `base`'s buffer. Views inherit writability: a read-only base
// can never yield a writable view.
inline ArrayPtr makeView(const ArrayPtr& base,
                         std::vector<size_t> shape,
                         std::vector<ptrdiff_t> strides,
                         size_t offset) {
    auto v = std::make_shared<NdArray>();
    v->buf      = base->buf;
    v->dtype    = base->dtype;
    v->offset   = offset;
    v->shape    = std::move(shape);
    v->strides  = std::move(strides);
    v->writable = base->writable;
    return v;
}

// ── Value glue ───────────────────────────────────────────────────────────────
inline Value wrap(ArrayPtr a) {
    return Value(std::static_pointer_cast<void>(a), NDARRAY_TAG);
}
inline bool isArray(const Value& v) {
    return v.isNativeHandle() && v.handleTag() == NDARRAY_TAG && v.handle;
}
inline ArrayPtr asArray(const Value& v) {
    if (!isArray(v)) {
        throw std::runtime_error("expected an array (got " +
            (v.isNativeHandle() ? "<" + v.handleTag() + ">" : v.toString()) + ")");
    }
    return std::static_pointer_cast<NdArray>(v.handle);
}

// ── element read/write through strides ───────────────────────────────────────
inline size_t flatOffset(const NdArray& a, const std::vector<size_t>& idx) {
    ptrdiff_t off = (ptrdiff_t)a.offset;
    for (size_t d = 0; d < idx.size(); d++) off += (ptrdiff_t)idx[d] * a.strides[d];
    return (size_t)off;
}

inline double getAsDouble(const NdArray& a, size_t flat) {
    switch (a.dtype) {
        case DType::F64:  return static_cast<const double*>(a.buf->data)[flat];
        case DType::I64:  return (double)static_cast<const int64_t*>(a.buf->data)[flat];
        case DType::BOOL: return static_cast<const uint8_t*>(a.buf->data)[flat] ? 1.0 : 0.0;
    }
    return 0.0;
}

inline Value getAsValue(const NdArray& a, size_t flat) {
    switch (a.dtype) {
        case DType::F64:  return Value(static_cast<const double*>(a.buf->data)[flat]);
        // i64 materializes to a Bantu number, which is a double -- the same
        // 2^53 caveat arctic documents for its own i64 columns (decision A4).
        case DType::I64:  return Value((double)static_cast<const int64_t*>(a.buf->data)[flat]);
        case DType::BOOL: return Value((bool)(static_cast<const uint8_t*>(a.buf->data)[flat] != 0));
    }
    return Value();
}

inline void setFromDouble(NdArray& a, size_t flat, double v) {
    switch (a.dtype) {
        case DType::F64:  static_cast<double*>(a.buf->data)[flat]  = v; break;
        case DType::I64:  static_cast<int64_t*>(a.buf->data)[flat] = (int64_t)std::llround(v); break;
        case DType::BOOL: static_cast<uint8_t*>(a.buf->data)[flat] = (v != 0.0) ? 1 : 0; break;
    }
}

// ── iteration over every element of a possibly-strided array ─────────────────
// Calls body(flatOffsetIntoBuffer, linearIndexInCOrder). Used by the generic
// paths; the contiguous fast path skips it entirely.
template <class F>
inline void forEachIndex(const NdArray& a, F&& body) {
    const size_t nd = a.ndim();
    const size_t n  = a.size();
    if (n == 0) return;
    if (nd == 0) { body(a.offset, (size_t)0); return; }

    std::vector<size_t> counter(nd, 0);
    ptrdiff_t off = (ptrdiff_t)a.offset;
    for (size_t k = 0; k < n; k++) {
        body((size_t)off, k);
        // Odometer, least-significant axis last.
        for (ptrdiff_t d = (ptrdiff_t)nd - 1; d >= 0; d--) {
            if (++counter[d] < a.shape[d]) { off += a.strides[d]; break; }
            counter[d] = 0;
            off -= a.strides[d] * (ptrdiff_t)(a.shape[d] - 1);
        }
    }
}

// ── conversions ──────────────────────────────────────────────────────────────
inline ArrayPtr contiguousCopy(const ArrayPtr& a, DType to) {
    auto out = makeArray(a->shape, to, "copy");
    size_t k = 0;
    forEachIndex(*a, [&](size_t off, size_t) {
        setFromDouble(*out, k++, getAsDouble(*a, off));
    });
    return out;
}

// ── normalising user-supplied indices ────────────────────────────────────────
// Python's convention: a negative index counts from the end. Out of range is an
// error naming the axis, its extent and the offending value -- "index out of
// range" alone is a support burden.
inline size_t normIndex(double raw, size_t extent, size_t axis) {
    ptrdiff_t i = (ptrdiff_t)std::llround(raw);
    const ptrdiff_t orig = i;
    if (i < 0) i += (ptrdiff_t)extent;
    if (i < 0 || (size_t)i >= extent) {
        std::ostringstream o;
        o << "index " << orig << " is out of range for axis " << axis
          << " with " << extent << (extent == 1 ? " element" : " elements");
        throw std::runtime_error(o.str());
    }
    return (size_t)i;
}

inline size_t normAxis(double raw, size_t ndim) {
    ptrdiff_t ax = (ptrdiff_t)std::llround(raw);
    const ptrdiff_t orig = ax;
    if (ax < 0) ax += (ptrdiff_t)ndim;
    if (ax < 0 || (size_t)ax >= ndim) {
        std::ostringstream o;
        o << "axis " << orig << " is out of range for a " << ndim
          << "-dimensional array";
        throw std::runtime_error(o.str());
    }
    return (size_t)ax;
}

// ── PRNG ─────────────────────────────────────────────────────────────────────
// numba's own stream, not the global random() builtin. Reproducibility is not
// optional for numerics, and sharing a stream means an unrelated random() call
// elsewhere in the program silently changes your matrix (N-of the creation
// notes in the architecture doc).
struct Rng {
    uint64_t s[4];
    void seed(uint64_t x) {
        // splitmix64, the standard seeding routine for xoshiro.
        for (int i = 0; i < 4; i++) {
            x += 0x9E3779B97F4A7C15ULL;
            uint64_t z = x;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            s[i] = z ^ (z >> 31);
        }
    }
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    uint64_t next() {                                  // xoshiro256++
        const uint64_t result = rotl(s[0] + s[3], 23) + s[0];
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t;    s[3] = rotl(s[3], 45);
        return result;
    }
    // [0,1) with 53 bits of entropy, the standard construction.
    double uniform() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }
};

inline Rng& rng() {
    static Rng r = [] { Rng x; x.seed(0x2545F4914F6CDD1DULL); return x; }();
    return r;
}

// ── repr ─────────────────────────────────────────────────────────────────────
// Summarization follows NumPy's rule: every element up to 1000, then three from
// each end. Matching a convention people already know beats inventing one, and
// it keeps print() usable on a ten-million-element array.
std::string reprArray(const NdArray& a);
inline std::string reprArrayHandle(const std::shared_ptr<void>& h) {
    if (!h) return "<ndarray>";
    return reprArray(*std::static_pointer_cast<NdArray>(h));
}

} // namespace numba
