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
#include <atomic>
#include <cerrno>
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
// server: nd_zeros([1e15]) has to raise, not OOM-kill the worker (N15).
//
// TWO numbers, not one, because they answer different questions (N17):
//
//   the HARD ceiling  — set once at startup from BANTU_ND_MAX_BYTES, by whoever
//                       runs the process. Script code can never raise past it.
//   the SOFT ceiling  — what nd_max_bytes() moves, and only ever DOWNWARD from
//                       the hard one. A limit a script can raise for itself is a
//                       guard against mistakes, not against hostile input; the
//                       ratchet is what makes it an actual control. This is the
//                       shape CPython settled on for the same class of problem
//                       (CVE-2020-10735): an env var the operator owns plus a
//                       runtime setter that cannot exceed it.
//
// Both are atomic because sua runs each connection's Bantu handler on its own
// detached std::thread (server.hpp), so these are touched concurrently. A plain
// size_t here is a data race, which is undefined behaviour and not merely a
// stale read.
inline std::atomic<size_t>& hardMaxBytesRef() {
    static std::atomic<size_t> cap{0};   // 0 until initLimits() runs
    return cap;
}
inline std::atomic<size_t>& maxBytesRef() {
    static std::atomic<size_t> cap{(size_t)2 * 1024 * 1024 * 1024};   // 2 GiB
    return cap;
}

// Bytes currently held by live Buffers. The ceiling above is PER ALLOCATION,
// which does not bound a loop: twelve 200 MB arrays held at once were measured
// sailing past a 2 GiB per-call limit without an error, and a loop is how a
// request handler actually exhausts a server. So admission is tested against
// live + requested, and the counter is decremented in ~Buffer.
inline std::atomic<size_t>& liveBytesRef() {
    static std::atomic<size_t> live{0};
    return live;
}

// Read BANTU_ND_MAX_BYTES once. Called from registerBuiltins, so it happens
// before any script line runs and therefore before any thread exists.
inline void initLimits() {
    size_t hard = (size_t)8 * 1024 * 1024 * 1024;   // 8 GiB unless told otherwise
    if (const char* env = std::getenv("BANTU_ND_MAX_BYTES")) {
        errno = 0;
        char* end = nullptr;
        const unsigned long long v = std::strtoull(env, &end, 10);
        // Anything unparseable leaves the default in place rather than becoming
        // zero -- a typo in a deployment env var must not silently disable the
        // limit, nor silently forbid every allocation. 0 means "no operator cap",
        // which lifts the ratchet but does NOT by itself raise the soft ceiling:
        // the program still has to ask for more with nd_max_bytes().
        if (end != env && errno == 0) hard = (v == 0) ? SIZE_MAX : (size_t)v;
    }
    hardMaxBytesRef().store(hard, std::memory_order_relaxed);
    if (maxBytesRef().load(std::memory_order_relaxed) > hard) {
        maxBytesRef().store(hard, std::memory_order_relaxed);
    }
}

// Multiply with overflow detection. shape [2^22, 2^22, 2^22] silently wraps to
// a small number under plain multiplication, which would allocate a tiny buffer
// that every later kernel writes past -- a heap overflow reachable from one
// line of user script.
//
// The division form is kept deliberately. __builtin_mul_overflow is one `mul`
// plus `jo` against a 20-40 cycle divide, but this runs ONCE per array creation
// against an allocation measured at 47 ms for 10M elements, and compilers
// commonly lower this form to comparisons anyway. It is portable to MSVC with
// no #ifdef, which the signed version below cannot be.
inline size_t checkedMul(size_t a, size_t b, const char* what) {
    if (a == 0 || b == 0) return 0;
    if (a > SIZE_MAX / b) {
        throw std::runtime_error(std::string(what) + ": size overflows (the shape is too large "
                                 "to describe, let alone allocate)");
    }
    return a * b;
}

// The SIGNED counterpart, for stride arithmetic. This one is not a style
// preference: signed overflow is undefined behaviour, so it cannot be detected
// after the fact by inspecting the result the way the unsigned form can. It was
// reachable -- nd_slice with a step near -2^62 on a stride-8 axis wrapped
// `stride * step` to 0, manufacturing a stride-0 axis on an array still marked
// writable, which is exactly the state broadcast_to refuses to produce.
inline bool mulOverflows(ptrdiff_t a, ptrdiff_t b, ptrdiff_t* out) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_mul_overflow(a, b, out);
#else
    // Checked BEFORE multiplying (CERT INT32-C): multiplying first and dividing
    // back is itself the signed overflow this exists to prevent.
    constexpr ptrdiff_t MX = PTRDIFF_MAX, MN = PTRDIFF_MIN;
    if (a > 0) {
        if (b > 0 ? a > MX / b : b < MN / a) return true;
    } else if (a < 0) {
        if (b > 0 ? a < MN / b : b < MX / a) return true;
    }
    *out = a * b;
    return false;
#endif
}
inline ptrdiff_t checkedMulSigned(ptrdiff_t a, ptrdiff_t b, const char* what) {
    ptrdiff_t r = 0;
    if (mulOverflows(a, b, &r)) {
        throw std::runtime_error(std::string(what) + ": stride arithmetic overflows");
    }
    return r;
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

    // `zero` is false only for nd_empty, whose contract is explicitly
    // "uninitialised" -- see the memset comment below for why every other
    // allocation touches its pages (N20).
    explicit Buffer(size_t bytes, bool zero = true) : nbytes(bytes) {
        const size_t soft = maxBytesRef().load(std::memory_order_relaxed);
        if (bytes > soft) {
            const size_t hard = hardMaxBytesRef().load(std::memory_order_relaxed);
            std::ostringstream o;
            o << "allocation of " << bytes << " bytes exceeds the limit of " << soft;
            // Only advise nd_max_bytes() when it could actually help. Telling a
            // user to raise a ceiling the operator has capped sends them round a
            // loop that cannot terminate.
            if (soft < hard) o << " (raise it with nd_max_bytes(n) if you meant it)";
            else o << " set by BANTU_ND_MAX_BYTES, which script cannot raise";
            throw std::runtime_error(o.str());
        }
        // The per-allocation check above does not bound a LOOP, so admission is
        // also tested against everything currently live. compare_exchange rather
        // than fetch_add: two threads must not both observe room and both take
        // it (sua runs handlers on concurrent threads).
        size_t live = liveBytesRef().load(std::memory_order_relaxed);
        for (;;) {
            if (live > soft - bytes) {   // soft >= bytes, checked above: cannot wrap
                std::ostringstream o;
                o << "allocating " << bytes << " bytes would put numba over its "
                  << soft << "-byte limit (" << live << " already held by live arrays)";
                throw std::runtime_error(o.str());
            }
            if (liveBytesRef().compare_exchange_weak(live, live + bytes,
                                                     std::memory_order_relaxed)) break;
        }

        // 64-byte alignment lets the contiguous kernels vectorize without a
        // scalar peel prologue, and keeps two arrays off the same cache line.
        data = alignedAlloc64(bytes);
        if (!data) {
            // Give the bytes back before unwinding. An accounted-but-never-freed
            // allocation is worse than no limit at all, because the process
            // slowly refuses to allocate anything and nothing says why.
            liveBytesRef().fetch_sub(bytes, std::memory_order_relaxed);
            throw std::runtime_error("out of memory allocating " +
                                     std::to_string(bytes) + " bytes");
        }

        // LOAD-BEARING, not just zero-initialization. Do not "optimize" this
        // into calloc (docs/numba-acceleration.md §4): Linux overcommit means a
        // large posix_memalign succeeds without committing a page, and calloc
        // would serve it from mmap'd zero pages, so nothing is charged to this
        // process until a kernel touches it -- at which point the OOM killer
        // arrives instead of a catchable error, and the ceiling above stops
        // bounding anything real. Touching every page here is what converts a
        // deferred kill into an exception a Bantu try/catch can handle. It costs
        // ~40 ms per 10M f64, and that is the price of deterministic failure.
        //
        // nd_empty is the one exception, and it is safe because the DoS bound
        // comes from the live-byte accounting above, which counts the bytes
        // whether or not they are touched.
        if (zero) std::memset(data, 0, bytes ? bytes : 64);
    }
    Buffer(void* p, size_t bytes, std::shared_ptr<void> owner)
        : data(p), nbytes(bytes), owned(false), keepalive(std::move(owner)) {}

    ~Buffer() {
        if (owned && data) {
            alignedFree64(data);
            liveBytesRef().fetch_sub(nbytes, std::memory_order_relaxed);
        }
    }
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

// A fresh, C-contiguous array -- zero-filled unless `zero` is false, which only
// nd_empty passes.
inline ArrayPtr makeArray(const std::vector<size_t>& shape, DType dt, const char* what,
                          bool zero = true) {
    auto a = std::make_shared<NdArray>();
    a->dtype   = dt;
    a->shape   = shape;
    a->strides = cStrides(shape);
    const size_t n     = shapeProduct(shape, what);
    const size_t bytes = checkedMul(n, itemsize(dt), what);
    a->buf = std::make_shared<Buffer>(bytes, zero);
    return a;
}

// Every element this array can address must lie inside its buffer.
//
// This is the one invariant that makes the whole view system safe, and it lives
// here -- in the single chokepoint every view passes through -- rather than in
// each of the callers, because there is no version of "remember to check" that
// survives phases 2 through 5 adding view constructors. NumPy's analogue is
// PyArray_CheckStrides, and NumPy's own as_strided documentation is the argument
// for having one: get strides wrong and "array elements can point to invalid
// memory and can corrupt results or crash your program".
//
// Cost is ndim integer operations once per view construction, against kernels
// that then touch millions of elements.
inline void checkExtent(const NdArray& v, const char* what) {
    if (!v.buf) throw std::runtime_error(std::string(what) + ": view has no buffer");
    // Walk to the lowest and highest element offsets the shape can reach. A
    // negative stride (a flipped view) runs downward from the offset, so the two
    // ends must be tracked separately rather than assuming offset is the base.
    ptrdiff_t lo = (ptrdiff_t)v.offset, hi = (ptrdiff_t)v.offset;
    for (size_t d = 0; d < v.shape.size(); d++) {
        if (v.shape[d] == 0) return;              // empty: addresses nothing at all
        const ptrdiff_t span =
            checkedMulSigned((ptrdiff_t)(v.shape[d] - 1), v.strides[d], what);
        if (span < 0) lo += span; else hi += span;
    }
    const size_t item  = itemsize(v.dtype);
    const size_t limit = v.buf->nbytes / item;    // capacity in ELEMENTS
    if (lo < 0 || (size_t)hi >= limit) {
        std::ostringstream o;
        o << what << ": this view would reach elements " << lo << ".." << hi
          << " of a buffer holding " << limit;
        throw std::runtime_error(o.str());
    }
}

// A view sharing `base`'s buffer. Views inherit writability: a read-only base
// can never yield a writable view.
inline ArrayPtr makeView(const ArrayPtr& base,
                         std::vector<size_t> shape,
                         std::vector<ptrdiff_t> strides,
                         size_t offset,
                         const char* what = "view") {
    if (shape.size() != strides.size()) {
        throw std::runtime_error(std::string(what) + ": " + std::to_string(shape.size()) +
            " dimensions but " + std::to_string(strides.size()) + " strides");
    }
    auto v = std::make_shared<NdArray>();
    v->buf      = base->buf;
    v->dtype    = base->dtype;
    v->offset   = offset;
    v->shape    = std::move(shape);
    v->strides  = std::move(strides);
    v->writable = base->writable;
    checkExtent(*v, what);
    return v;
}

// Do two arrays address any of the same memory? Same question np.shares_memory
// answers, and the reason checkExtent's lo/hi walk is factored the way it is:
// Phase 2's `out=` needs exactly this to decide whether a destination overlaps
// an input, which otherwise produces garbage rather than an error.
inline bool extentRange(const NdArray& v, ptrdiff_t& lo, ptrdiff_t& hi) {
    lo = hi = (ptrdiff_t)v.offset;
    for (size_t d = 0; d < v.shape.size(); d++) {
        if (v.shape[d] == 0) return false;                    // addresses nothing
        ptrdiff_t span = 0;
        if (mulOverflows((ptrdiff_t)(v.shape[d] - 1), v.strides[d], &span)) return false;
        if (span < 0) lo += span; else hi += span;
    }
    return true;
}
inline bool sharesMemory(const NdArray& a, const NdArray& b) {
    if (!a.buf || !b.buf || a.buf.get() != b.buf.get()) return false;
    ptrdiff_t alo, ahi, blo, bhi;
    if (!extentRange(a, alo, ahi) || !extentRange(b, blo, bhi)) return false;
    // Conservative, like np.may_share_memory: overlapping bounding ranges count
    // as sharing even when the two stride patterns would never collide. A false
    // positive costs a defensive copy; a false negative costs a wrong answer.
    return alo <= bhi && blo <= ahi;
}

// The single gate on every write path. `writable` was enforced in exactly one
// place (nd_set) when phases 2 and 3 were about to add `out=`, nd_put and
// boolean-mask assignment -- three more chances to forget.
inline void requireWritable(const NdArray& a, const char* what) {
    if (!a.writable) {
        throw std::runtime_error(std::string(what) +
            ": this array is read-only. Broadcast views repeat one element along a "
            "stretched axis, so writing to them would hit the same memory many times "
            "-- use nd_copy() to get a writable array with the same values");
    }
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

// thread_local, not a process-wide singleton (N18). sua runs each connection's
// Bantu handler on its own detached std::thread (server.hpp), so a shared stream
// means nd_seed() in one request silently reshapes every other request's random
// arrays -- which is the exact objection that made numba carry its own PRNG
// instead of using the global random() in the first place, just one level up. It
// also removes a data race on the state, which is UB rather than merely untidy.
// The cost is that each thread must seed for itself; nd_seed says so.
inline Rng& rng() {
    static thread_local Rng r = [] { Rng x; x.seed(0x2545F4914F6CDD1DULL); return x; }();
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
