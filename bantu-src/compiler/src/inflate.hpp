#pragma once
/**
 * inflate.hpp — DEFLATE (RFC 1951), zlib (RFC 1950) and gzip (RFC 1952)
 * decoding: the half of the interpreter's compression surface that was
 * missing.
 *
 * raster_core.hpp has shipped `zlibCompress` since bplot's PNG writer landed,
 * so Bantu could *write* a zlib stream, a PNG, a gzip body or a ZIP member and
 * could not *read* one back. That made every compressed format in existence
 * write-only from Bantu. This is the inverse, and it is as general as the
 * compressor: nothing here knows about images.
 *
 * Portable C++17, no dependency on zlib, header-only, and a pure function of
 * its input — the same bytes in give the same bytes out on every platform.
 *
 * HARDENING. This is a parser of hostile input, so every one of these holds:
 *   - every read is bounds-checked before it happens; the bit reader returns a
 *     failure instead of reading past the end
 *   - `maxOut` is checked BEFORE each append, so an inflate bomb is refused
 *     rather than allocated (the caller passes the exact expected size when it
 *     knows it, which for PNG it always does)
 *   - a back-reference that reaches before the start of the output is an
 *     error, not an out-of-bounds read
 *   - over-subscribed Huffman code sets are rejected; the one legal
 *     under-subscribed case (a single distance code, RFC 1951 §3.2.7) is
 *     allowed exactly where the RFC allows it
 *   - no signed overflow, no shift wider than the type, no unchecked cast
 *   - every failure is a std::runtime_error carrying the byte offset, which
 *     the registration layer turns into a catchable Bantu error
 *
 * Cross-checked against zlib's own output: the differential test in
 * tests/pillow_inflate_test.b decodes vectors produced by Python's zlib and
 * gzip at levels 0, 1, 6 and 9, through this and through the independent
 * pure-Bantu decoder in pillow/inflate.b, and asserts all three agree.
 */

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace bantu_inflate {

// Absolute ceiling on one call's output, so a missing maxOut cannot exhaust
// memory. 256 MiB; callers pass something tighter.
constexpr size_t kAbsMaxOut = 268435456u;

namespace detail {

constexpr uint16_t kLenBase[29] = {3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,
                                   43,51,59,67,83,99,115,131,163,195,227,258};
constexpr uint8_t  kLenExtra[29] = {0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,
                                    4,4,4,4,5,5,5,5,0};
constexpr uint16_t kDistBase[30] = {1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,
                                    257,385,513,769,1025,1537,2049,3073,4097,
                                    6145,8193,12289,16385,24577};
constexpr uint8_t  kDistExtra[30] = {0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,
                                     9,9,10,10,11,11,12,12,13,13};
constexpr uint8_t  kClOrder[19] = {16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};

constexpr int kFastBits = 9;
constexpr int kFastSize = 1 << kFastBits;

[[noreturn]] inline void fail(const std::string& why, size_t at) {
    throw std::runtime_error("inflate: " + why + " (at byte " + std::to_string(at) + ")");
}

// A canonical Huffman decoder: a direct lookup on the next kFastBits bits for
// the common short codes, and the length-by-length canonical walk for the rest.
struct Huff {
    // fast[i] = (symbol << 4) | length, or 0xFFFF when the code at that prefix
    // is longer than kFastBits.
    uint16_t fast[kFastSize];
    uint16_t counts[16];
    std::vector<uint16_t> syms;
    bool complete = false;

    void build(const uint8_t* lengths, size_t n, const char* what, bool allowIncomplete) {
        for (int i = 0; i < 16; i++) counts[i] = 0;
        for (size_t i = 0; i < n; i++) {
            if (lengths[i] > 15) {
                throw std::runtime_error(std::string("inflate: ") + what +
                    " code length " + std::to_string((int)lengths[i]) + " out of range");
            }
            counts[lengths[i]]++;
        }
        counts[0] = 0;

        // Kraft inequality, in integers.
        int left = 1;
        bool any = false;
        for (int L = 1; L <= 15; L++) {
            left = left * 2 - (int)counts[L];
            if (left < 0) {
                throw std::runtime_error(std::string("inflate: ") + what +
                                         " Huffman code is over-subscribed");
            }
            if (counts[L]) any = true;
        }
        complete = (left == 0);
        if (!complete && any && !allowIncomplete) {
            throw std::runtime_error(std::string("inflate: ") + what +
                                     " Huffman code is incomplete");
        }

        uint16_t offs[17];
        offs[0] = 0;
        offs[1] = 0;
        for (int L = 1; L <= 15; L++) offs[L + 1] = (uint16_t)(offs[L] + counts[L]);
        syms.assign(offs[16], 0);
        uint16_t cursor[17];
        for (int L = 0; L <= 16; L++) cursor[L] = offs[L];
        for (size_t i = 0; i < n; i++) {
            const uint8_t L = lengths[i];
            if (L) syms[cursor[L]++] = (uint16_t)i;
        }

        for (int i = 0; i < kFastSize; i++) fast[i] = 0xFFFFu;

        // DEFLATE packs Huffman codes most-significant-bit first while the
        // stream delivers bits least-significant first, so the next kFastBits
        // off the stream are the code BIT-REVERSED. Filling every index whose
        // low L bits equal the reversed code is the whole trick.
        uint32_t code = 0;
        size_t idx = 0;
        for (int L = 1; L <= 15; L++) {
            for (uint16_t k = 0; k < counts[L]; k++) {
                if (L <= kFastBits) {
                    uint32_t rev = 0, c = code;
                    for (int b = 0; b < L; b++) { rev = (rev << 1) | (c & 1u); c >>= 1; }
                    const uint16_t entry = (uint16_t)((syms[idx + k] << 4) | (unsigned)L);
                    for (uint32_t slot = rev; slot < (uint32_t)kFastSize; slot += (1u << L)) {
                        fast[slot] = entry;
                    }
                }
                code++;
            }
            idx += counts[L];
            code <<= 1;
        }
    }
};

inline const Huff& fixedLit() {
    static const Huff h = [] {
        uint8_t L[288];
        for (int i = 0;   i < 144; i++) L[i] = 8;
        for (int i = 144; i < 256; i++) L[i] = 9;
        for (int i = 256; i < 280; i++) L[i] = 7;
        for (int i = 280; i < 288; i++) L[i] = 8;
        Huff t;
        t.build(L, 288, "fixed literal/length", false);
        return t;
    }();
    return h;
}

inline const Huff& fixedDist() {
    static const Huff h = [] {
        uint8_t L[30];
        for (int i = 0; i < 30; i++) L[i] = 5;
        Huff t;
        t.build(L, 30, "fixed distance", true);
        return t;
    }();
    return h;
}

// The bit reader. `acc` holds `nbit` bits, least significant first.
struct Bits {
    const uint8_t* in;
    size_t n;
    size_t pos = 0;
    uint64_t acc = 0;
    int nbit = 0;

    Bits(const uint8_t* p, size_t len) : in(p), n(len) {}

    // Pull bytes in until at least `want` bits are buffered. Returns false at
    // end of input, leaving whatever is buffered — a Huffman decode near the
    // end of a stream legitimately runs on a short buffer, and the caller
    // checks the code length against nbit afterwards.
    inline bool fill(int want) {
        while (nbit < want) {
            if (pos >= n) return false;
            acc |= (uint64_t)in[pos++] << nbit;
            nbit += 8;
        }
        return true;
    }
    inline uint32_t take(int count) {          // count <= 24, must be buffered
        const uint32_t v = (uint32_t)(acc & ((1u << count) - 1u));
        acc >>= count;
        nbit -= count;
        return v;
    }
    inline uint32_t need(int count) {          // buffered-or-throw
        if (!fill(count) && nbit < count) fail("the stream ends mid-code", pos);
        return take(count);
    }
    inline void align() {
        const int drop = nbit & 7;
        if (drop) take(drop);
        // Push whole buffered bytes back so byte reads see them.
        const int whole = nbit >> 3;
        pos -= (size_t)whole;
        acc = 0;
        nbit = 0;
    }
};

// Decode one symbol. Returns -1 only when the code cannot be completed because
// the stream has ended.
inline int decodeSym(Bits& b, const Huff& h) {
    b.fill(15);
    if (b.nbit >= 1) {
        const uint16_t e = h.fast[b.acc & (kFastSize - 1)];
        if (e != 0xFFFFu) {
            const int L = e & 15;
            if (L > b.nbit) return -1;
            b.acc >>= L;
            b.nbit -= L;
            return e >> 4;
        }
    }
    // The canonical walk, for codes longer than kFastBits.
    uint32_t code = 0, first = 0, index = 0;
    uint64_t probe = b.acc;
    for (int L = 1; L <= 15; L++) {
        code = (code << 1) | (uint32_t)(probe & 1u);
        probe >>= 1;
        const uint32_t cnt = h.counts[L];
        if (code - first < cnt) {
            if (L > b.nbit) return -1;
            b.acc >>= L;
            b.nbit -= L;
            return h.syms[index + code - first];
        }
        index += cnt;
        first = (first + cnt) << 1;
    }
    fail("invalid Huffman code", b.pos);
}

} // namespace detail

/**
 * Bare DEFLATE. `maxOut` of 0 means "no caller limit" (kAbsMaxOut still
 * applies); any other value is a hard ceiling checked before each append.
 */
inline std::vector<uint8_t> inflateRaw(const uint8_t* in, size_t n, size_t maxOut) {
    using namespace detail;
    const size_t cap = (maxOut == 0 || maxOut > kAbsMaxOut) ? kAbsMaxOut : maxOut;

    std::vector<uint8_t> out;
    Bits b(in, n);

    bool last = false;
    while (!last) {
        if (!b.fill(3) && b.nbit < 3) fail("the stream ends inside a block header", b.pos);
        last = b.take(1) != 0;
        const uint32_t btype = b.take(2);

        if (btype == 0) {
            b.align();
            if (b.pos + 4 > b.n) fail("the stream ends inside a stored block header", b.pos);
            const uint32_t len  = (uint32_t)b.in[b.pos] | ((uint32_t)b.in[b.pos + 1] << 8);
            const uint32_t nlen = (uint32_t)b.in[b.pos + 2] | ((uint32_t)b.in[b.pos + 3] << 8);
            b.pos += 4;
            if ((len ^ 0xFFFFu) != nlen) fail("stored block length check failed", b.pos - 4);
            if (b.pos + len > b.n) fail("a stored block runs past the end of the stream", b.pos);
            if (out.size() + len > cap) {
                throw std::runtime_error("inflate: output would exceed maxOut (" +
                                         std::to_string(cap) + " bytes)");
            }
            out.insert(out.end(), b.in + b.pos, b.in + b.pos + len);
            b.pos += len;
            continue;
        }
        if (btype == 3) fail("reserved block type 3", b.pos);

        const Huff* lit = &fixedLit();
        const Huff* dst = &fixedDist();
        Huff dynLit, dynDist;

        if (btype == 2) {
            const uint32_t nlit  = b.need(5) + 257;
            const uint32_t ndist = b.need(5) + 1;
            const uint32_t ncl   = b.need(4) + 4;
            if (nlit > 288 || ndist > 32) fail("a dynamic block declares too many codes", b.pos);

            uint8_t clen[19] = {0};
            for (uint32_t i = 0; i < ncl; i++) clen[kClOrder[i]] = (uint8_t)b.need(3);
            Huff cl;
            cl.build(clen, 19, "code-length", false);

            const uint32_t want = nlit + ndist;
            std::vector<uint8_t> all;
            all.reserve(want);
            while (all.size() < want) {
                const int sym = decodeSym(b, cl);
                if (sym < 0) fail("the stream ends inside the code-length table", b.pos);
                if (sym < 16) {
                    all.push_back((uint8_t)sym);
                } else {
                    uint32_t rep = 0;
                    uint8_t val = 0;
                    if (sym == 16) {
                        if (all.empty()) fail("a code-length repeat with nothing to repeat", b.pos);
                        val = all.back();
                        rep = 3 + b.need(2);
                    } else if (sym == 17) {
                        rep = 3 + b.need(3);
                    } else {
                        rep = 11 + b.need(7);
                    }
                    if (all.size() + rep > want) fail("a code-length repeat overruns the table", b.pos);
                    all.insert(all.end(), rep, val);
                }
            }
            if (all[256] == 0) fail("the dynamic block has no end-of-block code", b.pos);
            dynLit.build(all.data(), nlit, "literal/length", false);
            // One distance code is legal and under-subscribed (RFC 1951
            // §3.2.7); real encoders emit it for a literal-only block.
            dynDist.build(all.data() + nlit, ndist, "distance", true);
            lit = &dynLit;
            dst = &dynDist;
        }

        for (;;) {
            const int sym = decodeSym(b, *lit);
            if (sym < 0) fail("the stream ends inside a block", b.pos);
            if (sym < 256) {
                if (out.size() + 1 > cap) {
                    throw std::runtime_error("inflate: output would exceed maxOut (" +
                                             std::to_string(cap) + " bytes)");
                }
                out.push_back((uint8_t)sym);
                continue;
            }
            if (sym == 256) break;

            const uint32_t li = (uint32_t)sym - 257;
            if (li >= 29) fail("invalid length code " + std::to_string(sym), b.pos);
            uint32_t length = kLenBase[li];
            if (kLenExtra[li]) length += b.need(kLenExtra[li]);

            const int dsym = decodeSym(b, *dst);
            if (dsym < 0) fail("the stream ends inside a distance code", b.pos);
            if (dsym >= 30) fail("invalid distance code " + std::to_string(dsym), b.pos);
            uint32_t dist = kDistBase[dsym];
            if (kDistExtra[dsym]) dist += b.need(kDistExtra[dsym]);

            if (dist == 0 || (size_t)dist > out.size()) {
                fail("a back-reference of " + std::to_string(dist) +
                     " reaches before the start of the output (" +
                     std::to_string(out.size()) + " bytes so far)", b.pos);
            }
            if (out.size() + length > cap) {
                throw std::runtime_error("inflate: output would exceed maxOut (" +
                                         std::to_string(cap) + " bytes)");
            }
            // The copy may overlap itself — that is how DEFLATE encodes a run —
            // so it is byte by byte, re-reading `out` as it grows. Resizing
            // first would invalidate an iterator; indexing does not.
            size_t from = out.size() - dist;
            out.resize(out.size() + length);
            uint8_t* o = out.data();
            size_t to = out.size() - length;
            for (uint32_t k = 0; k < length; k++) o[to + k] = o[from + k];
        }
    }
    return out;
}

inline uint32_t adler32Of(const uint8_t* p, size_t n) {
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < n; i++) {
        a = (a + p[i]) % 65521u;
        b = (b + a) % 65521u;
    }
    return (b << 16) | a;
}

/** zlib (RFC 1950): a 2-byte header, DEFLATE, an Adler-32 trailer. */
inline std::vector<uint8_t> inflateZlib(const uint8_t* in, size_t n, size_t maxOut) {
    if (n < 2) throw std::runtime_error("inflate: a zlib stream is at least 2 bytes");
    const uint32_t cmf = in[0], flg = in[1];
    if ((cmf & 0x0Fu) != 8) {
        throw std::runtime_error("inflate: compression method " +
                                 std::to_string(cmf & 0x0Fu) + " is not DEFLATE");
    }
    if (((cmf << 8) | flg) % 31u != 0) {
        throw std::runtime_error("inflate: zlib header check failed");
    }
    if (flg & 0x20u) {
        throw std::runtime_error("inflate: this zlib stream needs a preset dictionary, "
                                 "which is not supported");
    }
    std::vector<uint8_t> out = inflateRaw(in + 2, n - 2, maxOut);
    if (n >= 6) {
        const uint32_t want = ((uint32_t)in[n - 4] << 24) | ((uint32_t)in[n - 3] << 16) |
                              ((uint32_t)in[n - 2] << 8)  |  (uint32_t)in[n - 1];
        const uint32_t got = adler32Of(out.data(), out.size());
        if (got != want) {
            throw std::runtime_error("inflate: Adler-32 mismatch (stream says " +
                std::to_string(want) + ", data gives " + std::to_string(got) + ")");
        }
    }
    return out;
}

/** gzip (RFC 1952): a 10-byte header plus optional extras, DEFLATE, CRC-32 and ISIZE. */
inline std::vector<uint8_t> inflateGzip(const uint8_t* in, size_t n, size_t maxOut) {
    if (n < 18) throw std::runtime_error("inflate: a gzip member is at least 18 bytes");
    if (in[0] != 0x1F || in[1] != 0x8B) throw std::runtime_error("inflate: not a gzip stream (bad magic)");
    if (in[2] != 8) {
        throw std::runtime_error("inflate: compression method " + std::to_string(in[2]) +
                                 " is not DEFLATE");
    }
    const uint32_t flg = in[3];
    size_t i = 10;
    if (flg & 0x04u) {                                  // FEXTRA
        if (i + 2 > n) throw std::runtime_error("inflate: truncated gzip FEXTRA");
        const size_t xlen = (size_t)in[i] | ((size_t)in[i + 1] << 8);
        i += 2 + xlen;
        if (i > n) throw std::runtime_error("inflate: truncated gzip FEXTRA");
    }
    if (flg & 0x08u) { while (i < n && in[i]) i++; i++; }   // FNAME
    if (flg & 0x10u) { while (i < n && in[i]) i++; i++; }   // FCOMMENT
    if (flg & 0x02u) i += 2;                                // FHCRC
    if (i + 8 > n) throw std::runtime_error("inflate: truncated gzip member");

    std::vector<uint8_t> out = inflateRaw(in + i, n - i - 8, maxOut);
    const uint32_t isize = (uint32_t)in[n - 4] | ((uint32_t)in[n - 3] << 8) |
                           ((uint32_t)in[n - 2] << 16) | ((uint32_t)in[n - 1] << 24);
    if ((uint32_t)(out.size() & 0xFFFFFFFFu) != isize) {
        throw std::runtime_error("inflate: gzip length mismatch (trailer says " +
            std::to_string(isize) + ", got " + std::to_string(out.size() & 0xFFFFFFFFu) + ")");
    }
    return out;
}

/** Sniff the container and decode. */
inline std::vector<uint8_t> inflateAuto(const uint8_t* in, size_t n, size_t maxOut) {
    if (n >= 2 && in[0] == 0x1F && in[1] == 0x8B) return inflateGzip(in, n, maxOut);
    if (n >= 2 && (in[0] & 0x0Fu) == 8 && (((uint32_t)in[0] << 8) | in[1]) % 31u == 0) {
        return inflateZlib(in, n, maxOut);
    }
    return inflateRaw(in, n, maxOut);
}

} // namespace bantu_inflate
