#pragma once
// ════════════════════════════════════════════════════════════════════════════
//  raster_core.hpp — bplot's raster canvas and PNG encoder, in portable C++.
//
//  Knows nothing about Bantu: no Value, no builtins. That is what lets it be
//  checked on its own against decoders we did not write (Python's zlib and
//  Pillow), and what raster_native.cpp wraps.
//
//  THE CONTRACT is byte-identical output on Linux, macOS and Windows
//  (docs/bplot-raster-architecture.md §2). So, in this file:
//
//    * Every step that decides a pixel or a compressed byte is INTEGER
//      arithmetic. The one floating-point operation is quantising a
//      coordinate, centi(): a single multiply and std::round, the same two
//      steps bplot's _px performs for SVG -- which is why the PNG draws the
//      SVG's numbers to a hundredth of a pixel.
//    * No libm transcendental function, no system zlib, no hash-seeded or
//      unordered container, no time, no locale.
//    * Every tie in a choice -- a filter, a Huffman code length, a block type,
//      a match -- is broken by a fixed rule, so the output is a function of
//      the input alone.
// ════════════════════════════════════════════════════════════════════════════

#include "raster_font.hpp"
#include "raster_tables.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace bplot_raster {

// ── Limits (§10) ────────────────────────────────────────────────────────────
constexpr uint32_t kMaxSide    = 32767;
constexpr uint64_t kMaxPixels  = (uint64_t)1 << 28;    // ~805 MB of RGB at the cap
constexpr uint32_t kMaxDpi     = 2400;
constexpr double   kMaxCoord   = 1e8;                   // user units; beyond any canvas

// ── Checksums ───────────────────────────────────────────────────────────────
inline const std::array<uint32_t, 256>& crcTable() {
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[i] = c;
        }
        return t;
    }();
    return table;
}

inline uint32_t crc32Update(uint32_t c, const uint8_t* p, size_t n) {
    const auto& t = crcTable();
    for (size_t i = 0; i < n; i++) c = t[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    return c;
}

inline uint32_t crc32(const uint8_t* p, size_t n) {
    return crc32Update(0xFFFFFFFFu, p, n) ^ 0xFFFFFFFFu;
}

inline uint32_t adler32(const uint8_t* p, size_t n) {
    uint32_t a = 1, b = 0;
    while (n > 0) {
        // 5552 is the longest run for which b cannot overflow 32 bits.
        size_t k = n < 5552 ? n : 5552;
        n -= k;
        while (k--) { a += *p++; b += a; }
        a %= 65521u;
        b %= 65521u;
    }
    return (b << 16) | a;
}

// ── Bit output, least significant bit first, as deflate requires ────────────
class BitWriter {
public:
    explicit BitWriter(std::vector<uint8_t>& out) : out_(out) {}

    void bits(uint32_t v, int n) {
        if (n <= 0) return;
        if (n < 32) v &= (1u << n) - 1u;
        acc_ |= (uint64_t)v << count_;
        count_ += n;
        while (count_ >= 8) {
            out_.push_back((uint8_t)(acc_ & 0xFFu));
            acc_ >>= 8;
            count_ -= 8;
        }
    }

    // A Huffman code is defined most-significant-bit first; deflate packs bits
    // least-significant first, so the code goes in reversed.
    void code(uint32_t c, int len) {
        uint32_t r = 0;
        for (int i = 0; i < len; i++) { r = (r << 1) | (c & 1u); c >>= 1; }
        bits(r, len);
    }

    void alignToByte() {
        if (count_ > 0) {
            out_.push_back((uint8_t)(acc_ & 0xFFu));
            acc_ = 0;
            count_ = 0;
        }
    }

    // Only after alignToByte().
    void bytes(const uint8_t* p, size_t n) { out_.insert(out_.end(), p, p + n); }

    int pendingBits() const { return count_; }

private:
    std::vector<uint8_t>& out_;
    uint64_t acc_ = 0;
    int count_ = 0;
};

// ── Huffman code lengths, limited, by package-merge ─────────────────────────
//
// Optimal lengths subject to no code exceeding maxLen. Symbols are ordered by
// (frequency, symbol) with a STABLE sort, and a leaf wins every weight tie in
// the merge, so the lengths are a pure function of the frequencies.
inline std::vector<uint8_t> limitedLengths(const std::vector<uint32_t>& freq, int maxLen) {
    const size_t n = freq.size();
    std::vector<uint8_t> len(n, 0);
    std::vector<uint32_t> syms;
    for (uint32_t s = 0; s < (uint32_t)n; s++) if (freq[s]) syms.push_back(s);
    if (syms.empty()) return len;
    if (syms.size() == 1) { len[syms[0]] = 1; return len; }
    std::stable_sort(syms.begin(), syms.end(),
                     [&](uint32_t a, uint32_t b) { return freq[a] < freq[b]; });

    struct Node { uint64_t w; int32_t left; int32_t right; int32_t sym; };
    std::vector<Node> nodes;
    nodes.reserve(syms.size() * ((size_t)maxLen + 1) * 2 + 8);
    std::vector<int32_t> leaves;
    leaves.reserve(syms.size());
    for (uint32_t s : syms) {
        leaves.push_back((int32_t)nodes.size());
        nodes.push_back(Node{ freq[s], -1, -1, (int32_t)s });
    }

    std::vector<int32_t> list = leaves;
    for (int level = maxLen - 1; level >= 1; level--) {
        std::vector<int32_t> packages;
        packages.reserve(list.size() / 2);
        for (size_t i = 0; i + 1 < list.size(); i += 2) {
            const uint64_t w = nodes[(size_t)list[i]].w + nodes[(size_t)list[i + 1]].w;
            packages.push_back((int32_t)nodes.size());
            nodes.push_back(Node{ w, list[i], list[i + 1], -1 });
        }
        std::vector<int32_t> merged;
        merged.reserve(leaves.size() + packages.size());
        size_t a = 0, b = 0;
        while (a < leaves.size() || b < packages.size()) {
            const bool takeLeaf = b >= packages.size() ||
                (a < leaves.size() && nodes[(size_t)leaves[a]].w <= nodes[(size_t)packages[b]].w);
            merged.push_back(takeLeaf ? leaves[a++] : packages[b++]);
        }
        list.swap(merged);
    }

    // The first 2n-2 items; each appearance of a leaf lengthens its code by one.
    const size_t take = 2 * syms.size() - 2;
    std::vector<int32_t> stack;
    for (size_t i = 0; i < take && i < list.size(); i++) stack.push_back(list[i]);
    while (!stack.empty()) {
        const int32_t id = stack.back();
        stack.pop_back();
        const Node& nd = nodes[(size_t)id];
        if (nd.sym >= 0) len[(size_t)nd.sym]++;
        else { stack.push_back(nd.left); stack.push_back(nd.right); }
    }
    return len;
}

// RFC 1951 §3.2.2: canonical codes from lengths.
inline std::vector<uint32_t> canonicalCodes(const std::vector<uint8_t>& len) {
    uint32_t count[16] = {0};
    for (uint8_t l : len) if (l) count[l]++;
    uint32_t next[16] = {0};
    uint32_t code = 0;
    for (int bits = 1; bits <= 15; bits++) {
        code = (code + count[bits - 1]) << 1;
        next[bits] = code;
    }
    std::vector<uint32_t> codes(len.size(), 0);
    for (size_t s = 0; s < len.size(); s++) if (len[s]) codes[s] = next[len[s]]++;
    return codes;
}

// ── Deflate tables ──────────────────────────────────────────────────────────
static const uint16_t kLenBase[29]  = {3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258};
static const uint8_t  kLenExtra[29] = {0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};
static const uint16_t kDistBase[30] = {1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577};
static const uint8_t  kDistExtra[30]= {0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};
static const uint8_t  kClOrder[19]  = {16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};

inline uint32_t lengthIndex(uint32_t len) {
    static const std::array<uint8_t, 259> table = [] {
        std::array<uint8_t, 259> t{};
        for (uint32_t l = 3; l <= 258; l++) {
            uint8_t idx = 0;
            for (int i = 28; i >= 0; i--) if (l >= kLenBase[i]) { idx = (uint8_t)i; break; }
            t[l] = idx;
        }
        return t;
    }();
    return table[len];
}

inline uint32_t distanceIndex(uint32_t dist) {
    for (int i = 29; i >= 0; i--) if (dist >= kDistBase[i]) return (uint32_t)i;
    return 0;
}

// A token: a literal byte (< 256), or a match packed as
// 0x80000000 | (length - 3) << 16 | (distance - 1).
constexpr uint32_t kMatchBit = 0x80000000u;

// ── LZ77 over a 32 KiB window ───────────────────────────────────────────────
//
// Hash chains of at most kMaxChain candidates, one step of lazy matching below
// kLazyLimit, and a match of the maximum 258 bytes ends the search at once --
// which is what keeps a flat background, the commonest thing in a chart, from
// walking a full chain at every byte.
inline void lz77(const uint8_t* in, size_t n, std::vector<uint32_t>& out) {
    const uint32_t kWindow = 32768, kMask = kWindow - 1, kNil = 0xFFFFFFFFu;
    const int kMaxChain = 128;
    const uint32_t kLazyLimit = 32;
    if (n > 0xFFFFFFF0u) throw std::runtime_error("deflate: input larger than 4 GB");
    std::vector<uint32_t> head(32768, kNil), prev(kWindow, kNil);

    auto hashAt = [&](size_t i) -> uint32_t {
        return (((uint32_t)in[i] << 10) ^ ((uint32_t)in[i + 1] << 5) ^ (uint32_t)in[i + 2]) & 0x7FFFu;
    };
    auto insert = [&](size_t i) {
        if (i + 2 < n) {
            const uint32_t h = hashAt(i);
            prev[i & kMask] = head[h];
            head[h] = (uint32_t)i;
        }
    };
    auto longest = [&](size_t i, uint32_t& dist) -> uint32_t {
        dist = 0;
        if (i + 2 >= n) return 0;
        const size_t maxLen = std::min<size_t>(258, n - i);
        uint32_t best = 0;
        uint32_t cand = head[hashAt(i)];
        int chain = kMaxChain;
        while (cand != kNil && chain-- > 0) {
            if (cand >= i) break;
            const size_t d = i - cand;
            if (d > kWindow) break;
            if (in[cand + best] == in[i + best]) {
                size_t l = 0;
                while (l < maxLen && in[cand + l] == in[i + l]) l++;
                if (l > best) {
                    best = (uint32_t)l;
                    dist = (uint32_t)d;
                    if (l >= maxLen) break;
                }
            }
            const uint32_t nxt = prev[cand & kMask];
            // An entry overwritten by a newer position after the window wrapped.
            if (nxt != kNil && nxt >= cand) break;
            cand = nxt;
        }
        return best >= 3 ? best : 0;
    };

    size_t i = 0;
    while (i < n) {
        uint32_t dist = 0;
        const uint32_t len = longest(i, dist);
        if (len == 0) {
            out.push_back(in[i]);
            insert(i);
            i += 1;
            continue;
        }
        if (len < kLazyLimit && i + 1 < n) {
            insert(i);
            uint32_t d2 = 0;
            const uint32_t l2 = longest(i + 1, d2);
            if (l2 > len) {
                out.push_back(in[i]);
                i += 1;
                continue;
            }
            out.push_back(kMatchBit | ((len - 3) << 16) | (dist - 1));
            for (uint32_t k = 1; k < len; k++) insert(i + k);
            i += len;
            continue;
        }
        out.push_back(kMatchBit | ((len - 3) << 16) | (dist - 1));
        for (uint32_t k = 0; k < len; k++) insert(i + k);
        i += len;
    }
}

// ── One deflate block ───────────────────────────────────────────────────────
struct BlockCodes {
    std::vector<uint8_t> litLen, distLen;
};

inline uint64_t tokenBits(const std::vector<uint32_t>& toks, size_t t0, size_t t1,
                          const std::vector<uint8_t>& litLen, const std::vector<uint8_t>& distLen) {
    uint64_t bits = litLen[256];
    for (size_t t = t0; t < t1; t++) {
        const uint32_t tk = toks[t];
        if (!(tk & kMatchBit)) { bits += litLen[tk]; continue; }
        const uint32_t len = ((tk >> 16) & 0x1FFu) + 3, dist = (tk & 0xFFFFu) + 1;
        const uint32_t li = lengthIndex(len), di = distanceIndex(dist);
        bits += (uint64_t)litLen[257 + li] + kLenExtra[li] + distLen[di] + kDistExtra[di];
    }
    return bits;
}

// Make a code set complete when it has fewer than two symbols: a decoder needs
// a complete prefix code, and an unused symbol of length 1 costs nothing.
inline void completeCode(std::vector<uint8_t>& lens) {
    size_t used = 0;
    for (uint8_t l : lens) if (l) used++;
    if (used >= 2) return;
    if (used == 0) { lens[0] = 1; lens[1] = 1; return; }
    for (size_t s = 0; s < lens.size(); s++) {
        if (!lens[s]) { lens[s] = 1; return; }
    }
}

struct DynamicHeader {
    uint32_t hlit = 257, hdist = 1, hclen = 4;
    std::vector<uint8_t> clLen;               // 19 entries
    std::vector<uint32_t> clCode;
    std::vector<std::pair<uint8_t, uint8_t>> rle;   // (symbol, extra value)
    uint64_t bits = 0;
};

inline DynamicHeader buildHeader(const std::vector<uint8_t>& litLen, const std::vector<uint8_t>& distLen) {
    DynamicHeader h;
    h.hlit = 286;
    while (h.hlit > 257 && litLen[h.hlit - 1] == 0) h.hlit--;
    h.hdist = 30;
    while (h.hdist > 1 && distLen[h.hdist - 1] == 0) h.hdist--;

    std::vector<uint8_t> all(litLen.begin(), litLen.begin() + h.hlit);
    all.insert(all.end(), distLen.begin(), distLen.begin() + h.hdist);

    size_t i = 0;
    while (i < all.size()) {
        const uint8_t v = all[i];
        size_t run = 1;
        while (i + run < all.size() && all[i + run] == v) run++;
        size_t r = run;
        if (v == 0) {
            while (r >= 11) { const size_t k = std::min<size_t>(138, r); h.rle.push_back({18, (uint8_t)(k - 11)}); r -= k; }
            if (r >= 3) { h.rle.push_back({17, (uint8_t)(r - 3)}); r = 0; }
            while (r > 0) { h.rle.push_back({0, 0}); r--; }
        } else {
            h.rle.push_back({v, 0});
            r -= 1;
            while (r >= 3) { const size_t k = std::min<size_t>(6, r); h.rle.push_back({16, (uint8_t)(k - 3)}); r -= k; }
            while (r > 0) { h.rle.push_back({v, 0}); r--; }
        }
        i += run;
    }

    std::vector<uint32_t> clFreq(19, 0);
    for (const auto& p : h.rle) clFreq[p.first]++;
    h.clLen = limitedLengths(clFreq, 7);
    completeCode(h.clLen);
    h.clCode = canonicalCodes(h.clLen);
    h.hclen = 19;
    while (h.hclen > 4 && h.clLen[kClOrder[h.hclen - 1]] == 0) h.hclen--;

    h.bits = 5 + 5 + 4 + 3ull * h.hclen;
    for (const auto& p : h.rle) {
        h.bits += h.clLen[p.first];
        if (p.first == 16) h.bits += 2;
        else if (p.first == 17) h.bits += 3;
        else if (p.first == 18) h.bits += 7;
    }
    return h;
}

inline void writeTokens(BitWriter& bw, const std::vector<uint32_t>& toks, size_t t0, size_t t1,
                        const std::vector<uint8_t>& litLen, const std::vector<uint32_t>& litCode,
                        const std::vector<uint8_t>& distLen, const std::vector<uint32_t>& distCode) {
    for (size_t t = t0; t < t1; t++) {
        const uint32_t tk = toks[t];
        if (!(tk & kMatchBit)) { bw.code(litCode[tk], litLen[tk]); continue; }
        const uint32_t len = ((tk >> 16) & 0x1FFu) + 3, dist = (tk & 0xFFFFu) + 1;
        const uint32_t li = lengthIndex(len), di = distanceIndex(dist);
        bw.code(litCode[257 + li], litLen[257 + li]);
        bw.bits(len - kLenBase[li], kLenExtra[li]);
        bw.code(distCode[di], distLen[di]);
        bw.bits(dist - kDistBase[di], kDistExtra[di]);
    }
    bw.code(litCode[256], litLen[256]);
}

inline const BlockCodes& fixedCodes() {
    static const BlockCodes codes = [] {
        BlockCodes c;
        c.litLen.assign(288, 0);
        for (int s = 0; s < 144; s++) c.litLen[s] = 8;
        for (int s = 144; s < 256; s++) c.litLen[s] = 9;
        for (int s = 256; s < 280; s++) c.litLen[s] = 7;
        for (int s = 280; s < 288; s++) c.litLen[s] = 8;
        c.distLen.assign(30, 5);
        return c;
    }();
    return codes;
}

// Emits the cheapest of a stored, fixed-Huffman and dynamic-Huffman block for
// tokens [t0, t1), which cover input bytes [r0, r1). A tie goes to the simpler
// block, in that order.
inline void emitBlock(BitWriter& bw, const std::vector<uint32_t>& toks, size_t t0, size_t t1,
                      const uint8_t* in, size_t r0, size_t r1, bool last) {
    std::vector<uint32_t> litFreq(286, 0), distFreq(30, 0);
    litFreq[256] = 1;
    for (size_t t = t0; t < t1; t++) {
        const uint32_t tk = toks[t];
        if (!(tk & kMatchBit)) { litFreq[tk]++; continue; }
        const uint32_t len = ((tk >> 16) & 0x1FFu) + 3, dist = (tk & 0xFFFFu) + 1;
        litFreq[257 + lengthIndex(len)]++;
        distFreq[distanceIndex(dist)]++;
    }

    std::vector<uint8_t> dynLit = limitedLengths(litFreq, 15);
    std::vector<uint8_t> dynDist = limitedLengths(distFreq, 15);
    completeCode(dynLit);
    completeCode(dynDist);
    const DynamicHeader hdr = buildHeader(dynLit, dynDist);

    const BlockCodes& fx = fixedCodes();
    const uint64_t fixedBits = 3 + tokenBits(toks, t0, t1, fx.litLen, fx.distLen);
    const uint64_t dynBits = 3 + hdr.bits + tokenBits(toks, t0, t1, dynLit, dynDist);
    const size_t rawLen = r1 - r0;
    uint64_t storedBits = UINT64_MAX;
    if (rawLen <= 65535) {
        const int pad = (8 - ((bw.pendingBits() + 3) % 8)) % 8;
        storedBits = 3 + (uint64_t)pad + 32 + 8ull * rawLen;
    }

    if (storedBits < fixedBits && storedBits < dynBits) {
        bw.bits(last ? 1u : 0u, 1);
        bw.bits(0u, 2);
        bw.alignToByte();
        const uint8_t hdrBytes[4] = { (uint8_t)(rawLen & 0xFF), (uint8_t)(rawLen >> 8),
                                      (uint8_t)(~rawLen & 0xFF), (uint8_t)((~rawLen >> 8) & 0xFF) };
        bw.bytes(hdrBytes, 4);
        bw.bytes(in + r0, rawLen);
        return;
    }
    if (fixedBits <= dynBits) {
        bw.bits(last ? 1u : 0u, 1);
        bw.bits(1u, 2);
        writeTokens(bw, toks, t0, t1, fx.litLen, canonicalCodes(fx.litLen), fx.distLen, canonicalCodes(fx.distLen));
        return;
    }
    bw.bits(last ? 1u : 0u, 1);
    bw.bits(2u, 2);
    bw.bits(hdr.hlit - 257, 5);
    bw.bits(hdr.hdist - 1, 5);
    bw.bits(hdr.hclen - 4, 4);
    for (uint32_t k = 0; k < hdr.hclen; k++) bw.bits(hdr.clLen[kClOrder[k]], 3);
    for (const auto& p : hdr.rle) {
        bw.code(hdr.clCode[p.first], hdr.clLen[p.first]);
        if (p.first == 16) bw.bits(p.second, 2);
        else if (p.first == 17) bw.bits(p.second, 3);
        else if (p.first == 18) bw.bits(p.second, 7);
    }
    writeTokens(bw, toks, t0, t1, dynLit, canonicalCodes(dynLit), dynDist, canonicalCodes(dynDist));
}

// ── zlib (RFC 1950) around deflate (RFC 1951) ───────────────────────────────
inline std::vector<uint8_t> zlibCompress(const uint8_t* in, size_t n) {
    std::vector<uint8_t> out;
    out.reserve(n / 4 + 64);
    out.push_back(0x78);   // deflate, 32 KiB window
    out.push_back(0x9C);   // default level; (0x78 * 256 + 0x9C) is a multiple of 31
    std::vector<uint32_t> toks;
    toks.reserve(n / 8 + 16);
    lz77(in, n, toks);

    BitWriter bw(out);
    const size_t kBlockTokens = 50000;
    if (toks.empty()) {
        emitBlock(bw, toks, 0, 0, in, 0, 0, true);
    } else {
        size_t t = 0, r = 0;
        while (t < toks.size()) {
            const size_t t1 = std::min(toks.size(), t + kBlockTokens);
            size_t r1 = r;
            for (size_t k = t; k < t1; k++) {
                r1 += (toks[k] & kMatchBit) ? (((toks[k] >> 16) & 0x1FFu) + 3) : 1;
            }
            emitBlock(bw, toks, t, t1, in, r, r1, t1 == toks.size());
            t = t1;
            r = r1;
        }
    }
    bw.alignToByte();
    const uint32_t ad = adler32(in, n);
    out.push_back((uint8_t)(ad >> 24));
    out.push_back((uint8_t)(ad >> 16));
    out.push_back((uint8_t)(ad >> 8));
    out.push_back((uint8_t)ad);
    return out;
}

// ── PNG ─────────────────────────────────────────────────────────────────────
inline void putBE32(std::vector<uint8_t>& o, uint32_t v) {
    o.push_back((uint8_t)(v >> 24)); o.push_back((uint8_t)(v >> 16));
    o.push_back((uint8_t)(v >> 8));  o.push_back((uint8_t)v);
}

inline void putChunk(std::vector<uint8_t>& o, const char* tag, const uint8_t* data, size_t n) {
    putBE32(o, (uint32_t)n);
    const size_t start = o.size();
    o.insert(o.end(), tag, tag + 4);
    if (n) o.insert(o.end(), data, data + n);
    putBE32(o, crc32(o.data() + start, n + 4));
}

inline uint8_t paeth(int a, int b, int c) {
    const int p = a + b - c;
    const int pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
    if (pa <= pb && pa <= pc) return (uint8_t)a;
    if (pb <= pc) return (uint8_t)b;
    return (uint8_t)c;
}

// 8-bit RGB, one IDAT stream. Each row takes the filter with the smallest sum
// of absolute residuals -- the PNG specification's recommended heuristic -- and
// a tie goes to the lowest filter number. No tIME chunk: a timestamp would make
// every render of the same figure a different file.
inline std::vector<uint8_t> encodePng(const uint8_t* rgb, uint32_t w, uint32_t h, uint32_t dpi) {
    const size_t stride = (size_t)w * 3;
    std::vector<uint8_t> raw;
    raw.reserve((stride + 1) * h);
    std::vector<uint8_t> cand[5];
    for (auto& c : cand) c.resize(stride);
    const std::vector<uint8_t> zero(stride, 0);

    for (uint32_t y = 0; y < h; y++) {
        const uint8_t* cur = rgb + (size_t)y * stride;
        const uint8_t* up = y ? rgb + (size_t)(y - 1) * stride : zero.data();
        uint64_t cost[5] = {0, 0, 0, 0, 0};
        for (size_t x = 0; x < stride; x++) {
            const int a = x >= 3 ? cur[x - 3] : 0;
            const int b = up[x];
            const int c = x >= 3 ? up[x - 3] : 0;
            const int v = cur[x];
            const uint8_t f[5] = {
                (uint8_t)v,
                (uint8_t)(v - a),
                (uint8_t)(v - b),
                (uint8_t)(v - ((a + b) >> 1)),
                (uint8_t)(v - paeth(a, b, c))
            };
            for (int k = 0; k < 5; k++) {
                cand[k][x] = f[k];
                cost[k] += f[k] < 128 ? f[k] : 256u - f[k];
            }
        }
        int best = 0;
        for (int k = 1; k < 5; k++) if (cost[k] < cost[best]) best = k;
        raw.push_back((uint8_t)best);
        raw.insert(raw.end(), cand[best].begin(), cand[best].end());
    }

    const std::vector<uint8_t> z = zlibCompress(raw.data(), raw.size());

    std::vector<uint8_t> png;
    png.reserve(z.size() + 128);
    static const uint8_t kSig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    png.insert(png.end(), kSig, kSig + 8);

    std::vector<uint8_t> ihdr;
    putBE32(ihdr, w);
    putBE32(ihdr, h);
    ihdr.push_back(8);   // bit depth
    ihdr.push_back(2);   // colour type: RGB
    ihdr.push_back(0);   // compression
    ihdr.push_back(0);   // filter method
    ihdr.push_back(0);   // no interlace
    putChunk(png, "IHDR", ihdr.data(), ihdr.size());

    // pHYs: pixels per metre, dpi / 0.0254 rounded half up -- in integers.
    std::vector<uint8_t> phys;
    const uint32_t ppm = (uint32_t)(((uint64_t)dpi * 10000u + 127u) / 254u);
    putBE32(phys, ppm);
    putBE32(phys, ppm);
    phys.push_back(1);
    putChunk(png, "pHYs", phys.data(), phys.size());

    const size_t kIdat = 1u << 20;
    for (size_t off = 0; off < z.size(); off += kIdat) {
        putChunk(png, "IDAT", z.data() + off, std::min(kIdat, z.size() - off));
    }
    putChunk(png, "IEND", nullptr, 0);
    return png;
}

// ── Coordinates (§3) ────────────────────────────────────────────────────────

// Hundredths of a user unit, with exactly the arithmetic bplot's _px uses for
// SVG: sign * round(|v| * 100). One multiply and a round -- nothing a compiler
// can fuse, and nothing a libm can disagree about.
inline int64_t centi(double v) {
    if (!std::isfinite(v)) throw std::runtime_error("a coordinate is not a finite number");
    const bool neg = v < 0;
    double a = std::fabs(v);
    if (a > kMaxCoord) a = kMaxCoord;
    const int64_t u = (int64_t)std::round(a * 100.0);
    return neg ? -u : u;
}

// Q24.8 device units: centi * dpi * 256 / 9600, rounded half away from zero.
inline int64_t toQ8(int64_t c, uint32_t dpi) {
    const int64_t num = c * (int64_t)dpi * 256;
    const int64_t den = 9600;
    return num >= 0 ? (num + den / 2) / den : -((-num + den / 2) / den);
}

// A user length as whole device pixels, rounded half up.
inline uint64_t devicePixels(int64_t c, uint32_t dpi) {
    return ((uint64_t)c * dpi + 4800u) / 9600u;
}

// ── The canvas ──────────────────────────────────────────────────────────────
struct Rgb { uint8_t r, g, b; };

struct Canvas {
    uint32_t w = 0, h = 0, dpi = 96;
    std::vector<uint8_t> rgb;
    // The clip, in whole device pixels, half-open.
    int64_t clipX0 = 0, clipY0 = 0, clipX1 = 0, clipY1 = 0;
};

inline Canvas makeCanvas(double userW, double userH, uint32_t dpi, Rgb bg) {
    if (dpi < 1 || dpi > kMaxDpi) {
        throw std::runtime_error("dpi must be a whole number from 1 to " + std::to_string(kMaxDpi));
    }
    if (!std::isfinite(userW) || !std::isfinite(userH) || userW <= 0 || userH <= 0) {
        throw std::runtime_error("width and height must be positive numbers");
    }
    const uint64_t w = devicePixels(centi(userW), dpi), h = devicePixels(centi(userH), dpi);
    if (w < 1 || h < 1) throw std::runtime_error("the canvas would be smaller than one pixel");
    if (w > kMaxSide || h > kMaxSide) {
        throw std::runtime_error("the canvas would be " + std::to_string(w) + "x" + std::to_string(h) +
            " pixels -- each side is limited to " + std::to_string(kMaxSide));
    }
    if (w * h > kMaxPixels) {
        throw std::runtime_error("the canvas would be " + std::to_string(w * h) +
            " pixels -- the limit is " + std::to_string(kMaxPixels));
    }
    Canvas c;
    c.w = (uint32_t)w;
    c.h = (uint32_t)h;
    c.dpi = dpi;
    c.rgb.resize((size_t)(w * h * 3));
    for (size_t i = 0; i < c.rgb.size(); i += 3) { c.rgb[i] = bg.r; c.rgb[i + 1] = bg.g; c.rgb[i + 2] = bg.b; }
    c.clipX1 = c.w;
    c.clipY1 = c.h;
    return c;
}

// Source-over in 8-bit sRGB, rounded exactly: (s*a + d*(255-a) + 127) / 255.
inline void blend(uint8_t* px, Rgb c, uint32_t a) {
    if (a == 0) return;
    if (a >= 255) { px[0] = c.r; px[1] = c.g; px[2] = c.b; return; }
    const uint32_t ia = 255 - a;
    px[0] = (uint8_t)((c.r * a + px[0] * ia + 127) / 255);
    px[1] = (uint8_t)((c.g * a + px[1] * ia + 127) / 255);
    px[2] = (uint8_t)((c.b * a + px[2] * ia + 127) / 255);
}

// Opacity 0..1 as an 8-bit alpha: one multiply and a round.
inline uint32_t alpha8(double opacity) {
    if (!std::isfinite(opacity)) throw std::runtime_error("opacity must be a number from 0 to 1");
    if (opacity <= 0) return 0;
    if (opacity >= 1) return 255;
    return (uint32_t)std::lround(opacity * 255.0);
}

// An axis-aligned rectangle in Q24.8, with exact fractional coverage at its
// edges: each pixel's alpha is the covered area, in 1/65536 of a pixel, times
// the fill's alpha. A zero or negative width or height draws nothing.
inline void fillRectQ8(Canvas& cv, int64_t x0, int64_t y0, int64_t x1, int64_t y1, Rgb col, uint32_t a8) {
    if (a8 == 0 || x1 <= x0 || y1 <= y0) return;
    x0 = std::max<int64_t>(x0, cv.clipX0 * 256);
    y0 = std::max<int64_t>(y0, cv.clipY0 * 256);
    x1 = std::min<int64_t>(x1, cv.clipX1 * 256);
    y1 = std::min<int64_t>(y1, cv.clipY1 * 256);
    if (x1 <= x0 || y1 <= y0) return;
    const int64_t px0 = x0 >> 8, px1 = (x1 + 255) >> 8;
    const int64_t py0 = y0 >> 8, py1 = (y1 + 255) >> 8;
    for (int64_t py = py0; py < py1; py++) {
        const int64_t top = std::max<int64_t>(y0, py * 256), bot = std::min<int64_t>(y1, py * 256 + 256);
        const uint32_t cy = (uint32_t)(bot - top);                      // 1..256
        uint8_t* row = cv.rgb.data() + (size_t)py * cv.w * 3;
        for (int64_t px = px0; px < px1; px++) {
            const int64_t lft = std::max<int64_t>(x0, px * 256), rgt = std::min<int64_t>(x1, px * 256 + 256);
            const uint32_t cx = (uint32_t)(rgt - lft);                  // 1..256
            const uint32_t a = (uint32_t)(((uint64_t)cx * cy * a8 + 32768u) >> 16);
            blend(row + (size_t)px * 3, col, a);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  The rasteriser (B6c)
//
//  ANTI-ALIASING: sixteen sub-scanlines per pixel row, with EXACT coverage
//  along x. Vertical resolution is 1/16 of a pixel and horizontal is 1/256, so
//  a near-vertical edge -- the commonest in a chart, being a bar or an axis --
//  is as smooth as an 8-bit channel can show. Exact area accumulation would be
//  smoother on near-horizontal edges and far harder to keep integer-exact; this
//  is the trade stb_truetype's first rasteriser made.
//
//  ONE MASK PER SHAPE. A stroked polyline is segments, joins and caps, which
//  overlap. They are rasterised into a single coverage mask and composited
//  once, so a translucent line does not darken at every joint.
//
//  Checked against analytic areas: an axis-aligned rectangle and a triangle
//  come out exact, a clipped shape exact, and a circle matches the inscribed
//  polygon it is (1960.43 against 1960.34 for r = 25 at 64 steps) rather than
//  the ideal circle -- a chord error of 0.03 px, inside the 1/16 px budget.
// ════════════════════════════════════════════════════════════════════════════

// The unit-circle tables, generated by scripts/gen_circle_tables.py. Circles
// and arcs step around these rather than calling cos() or sin(), which differ
// in the last bit between platforms' libm and would break byte identity.
inline const int32_t* cosTable(int n) {
    switch (n) {
        case 16:  return kCos16;
        case 32:  return kCos32;
        case 64:  return kCos64;
        case 128: return kCos128;
        default:  return kCos256;
    }
}
inline const int32_t* sinTable(int n) {
    switch (n) {
        case 16:  return kSin16;
        case 32:  return kSin32;
        case 64:  return kSin64;
        case 128: return kSin128;
        default:  return kSin256;
    }
}

constexpr int kSubScanlines = 16;                 // vertical AA steps per pixel
constexpr int64_t kCoordLimit = (int64_t)1 << 22; // Q8: +/- 16,384 pixels

struct PointQ8 { int64_t x, y; };
using ContourQ8 = std::vector<PointQ8>;

// ── Integer square root of a 64-bit value (exact floor) ─────────────────────
inline uint64_t isqrt64(uint64_t n) {
    if (n == 0) return 0;
    uint64_t x = n, y = (x + 1) / 2;
    // Newton's method on integers; terminates, and the result is floor(sqrt(n)).
    while (y < x) { x = y; y = (x + n / x) / 2; }
    return x;
}

// Rounded division that is symmetric about zero, so a mirrored shape is the
// mirror of the shape.
inline int64_t divRound(int64_t num, int64_t den) {
    if (den < 0) { num = -num; den = -den; }
    return num >= 0 ? (num + den / 2) / den : -((-num + den / 2) / den);
}

// ── Clipping, so no coordinate can overflow the crossing arithmetic ─────────
//
// Sutherland-Hodgman against a rectangle, in integers. Clamping coordinates
// instead would change an edge's slope and bend the shape at the canvas edge.

// Where segment a-b crosses the line axis == limit. The exact rational when its
// product fits in 64 bits -- every coordinate a real figure produces. A point
// can be up to ~2^40 away in Q8 (kMaxCoord at the top dpi), where that product
// would overflow, so beyond 2^31 it BISECTS instead: halving only adds and
// shifts, cannot overflow, and lands within one Q8 unit of the true crossing.
// (No __int128: the Windows build is MSVC.)
inline PointQ8 crossAt(PointQ8 a, PointQ8 b, int axis, int64_t limit) {
    const int64_t av = axis == 0 ? a.x : a.y, bv = axis == 0 ? b.x : b.y;
    const int64_t t = bv - av;
    if (t == 0) return a;
    const int64_t ao = axis == 0 ? a.y : a.x, bo = axis == 0 ? b.y : b.x;
    const int64_t span = bo - ao, reach = limit - av;
    const int64_t kFit = (int64_t)1 << 31;
    if (span > -kFit && span < kFit && reach > -kFit && reach < kFit) {
        const int64_t o = ao + divRound(span * reach, t);
        return axis == 0 ? PointQ8{limit, o} : PointQ8{o, limit};
    }
    // Keep lo on a's side of the line and hi on b's until they meet.
    PointQ8 lo = a, hi = b;
    const bool aBelow = av < limit;
    for (int i = 0; i < 64; i++) {
        const PointQ8 mid{ lo.x + (hi.x - lo.x) / 2, lo.y + (hi.y - lo.y) / 2 };
        const int64_t mv = axis == 0 ? mid.x : mid.y;
        if (mv == limit || (mid.x == lo.x && mid.y == lo.y)) { lo = mid; break; }
        if ((mv < limit) == aBelow) lo = mid; else hi = mid;
    }
    const int64_t o = axis == 0 ? lo.y : lo.x;
    return axis == 0 ? PointQ8{limit, o} : PointQ8{o, limit};
}

inline ContourQ8 clipEdge(const ContourQ8& in, int axis, int64_t limit, bool keepGreater) {
    ContourQ8 out;
    if (in.empty()) return out;
    auto inside = [&](const PointQ8& p) {
        const int64_t v = axis == 0 ? p.x : p.y;
        return keepGreater ? v >= limit : v <= limit;
    };
    for (size_t i = 0; i < in.size(); i++) {
        const PointQ8& a = in[i];
        const PointQ8& b = in[(i + 1) % in.size()];
        const bool ina = inside(a), inb = inside(b);
        if (ina) out.push_back(a);
        if (ina != inb) out.push_back(crossAt(a, b, axis, limit));
    }
    return out;
}

// An OPEN polyline clipped to a box: the runs of it that lie inside. Strokes go
// through this before anything squares a length, so every coordinate a stroke
// sees is inside the box -- which is what keeps dx*dx, and a cap's radius times
// the circle table, inside 64 bits.
inline std::vector<std::vector<PointQ8>> clipPolyline(const std::vector<PointQ8>& pts,
                                                      int64_t x0, int64_t y0, int64_t x1, int64_t y1) {
    std::vector<std::vector<PointQ8>> runs;
    std::vector<PointQ8> cur;
    for (size_t i = 0; i + 1 < pts.size(); i++) {
        PointQ8 a = pts[i], b = pts[i + 1];
        bool keep = true;
        const int64_t lim[4] = { x0, x1, y0, y1 };
        for (int e = 0; e < 4 && keep; e++) {
            const int axis = e / 2;
            const bool greater = (e % 2) == 0;
            auto in = [&](const PointQ8& p) {
                const int64_t v = axis == 0 ? p.x : p.y;
                return greater ? v >= lim[e] : v <= lim[e];
            };
            const bool ia = in(a), ib = in(b);
            if (!ia && !ib) keep = false;
            else if (!ia) a = crossAt(a, b, axis, lim[e]);
            else if (!ib) b = crossAt(a, b, axis, lim[e]);
        }
        if (!keep) continue;
        if (cur.empty() || cur.back().x != a.x || cur.back().y != a.y) {
            if (cur.size() >= 2) runs.push_back(cur);
            cur.assign(1, a);
        }
        cur.push_back(b);
    }
    if (cur.size() >= 2) runs.push_back(cur);
    return runs;
}

inline ContourQ8 clipToBox(const ContourQ8& c, int64_t x0, int64_t y0, int64_t x1, int64_t y1) {
    ContourQ8 r = clipEdge(c, 0, x0, true);
    r = clipEdge(r, 0, x1, false);
    r = clipEdge(r, 1, y0, true);
    r = clipEdge(r, 1, y1, false);
    return r;
}

// ── The coverage mask ───────────────────────────────────────────────────────
struct Mask {
    int64_t x0 = 0, y0 = 0, w = 0, h = 0;      // whole pixels
    std::vector<uint16_t> cov;                  // 0 .. kSubScanlines * 256
};

struct EdgeQ8 { int64_t x0, y0, x1, y1; int dir; };

// Rasterise contours with the nonzero winding rule into a coverage mask
// bounded by the clip rectangle.
inline Mask rasterise(const std::vector<ContourQ8>& contours,
                      int64_t clipX0, int64_t clipY0, int64_t clipX1, int64_t clipY1) {
    Mask m;
    const int64_t bx0 = clipX0 * 256, by0 = clipY0 * 256;
    const int64_t bx1 = clipX1 * 256, by1 = clipY1 * 256;

    std::vector<EdgeQ8> edges;
    int64_t minX = bx1, maxX = bx0, minY = by1, maxY = by0;
    for (const ContourQ8& raw : contours) {
        if (raw.size() < 3) continue;
        const ContourQ8 c = clipToBox(raw, bx0, by0, bx1, by1);
        if (c.size() < 3) continue;
        for (size_t i = 0; i < c.size(); i++) {
            PointQ8 a = c[i], b = c[(i + 1) % c.size()];
            if (a.y == b.y) continue;                  // horizontal edges add nothing
            int dir = 1;
            if (a.y > b.y) { std::swap(a, b); dir = -1; }
            edges.push_back(EdgeQ8{a.x, a.y, b.x, b.y, dir});
            minX = std::min({minX, a.x, b.x});
            maxX = std::max({maxX, a.x, b.x});
            minY = std::min(minY, a.y);
            maxY = std::max(maxY, b.y);
        }
    }
    if (edges.empty()) return m;

    m.x0 = std::max<int64_t>(clipX0, minX >> 8);
    m.y0 = std::max<int64_t>(clipY0, minY >> 8);
    const int64_t x1 = std::min<int64_t>(clipX1, (maxX + 255) >> 8);
    const int64_t y1 = std::min<int64_t>(clipY1, (maxY + 255) >> 8);
    if (x1 <= m.x0 || y1 <= m.y0) { m.w = m.h = 0; return m; }
    m.w = x1 - m.x0;
    m.h = y1 - m.y0;
    m.cov.assign((size_t)(m.w * m.h), 0);

    // Crossings of one sub-scanline, as (x, direction).
    std::vector<std::pair<int64_t, int>> xs;
    for (int64_t py = m.y0; py < y1; py++) {
        uint16_t* row = m.cov.data() + (size_t)(py - m.y0) * m.w;
        for (int k = 0; k < kSubScanlines; k++) {
            // The centre of this sub-scanline, in Q8.
            const int64_t sy = py * 256 + (2 * k + 1) * 256 / (2 * kSubScanlines);
            xs.clear();
            for (const EdgeQ8& e : edges) {
                if (sy < e.y0 || sy >= e.y1) continue;
                const int64_t x = e.x0 + divRound((e.x1 - e.x0) * (sy - e.y0), e.y1 - e.y0);
                xs.push_back({x, e.dir});
            }
            if (xs.size() < 2) continue;
            std::sort(xs.begin(), xs.end());        // by x, then direction: deterministic
            int winding = 0;
            int64_t spanStart = 0;
            for (const auto& c : xs) {
                const int before = winding;
                winding += c.second;
                if (before == 0 && winding != 0) spanStart = c.first;
                else if (before != 0 && winding == 0) {
                    // Add exact horizontal coverage for [spanStart, c.first).
                    int64_t a = std::max(spanStart, m.x0 * 256);
                    int64_t b = std::min(c.first, x1 * 256);
                    if (b <= a) continue;
                    int64_t pa = a >> 8, pb = (b - 1) >> 8;
                    if (pa == pb) {
                        row[pa - m.x0] += (uint16_t)(b - a);
                    } else {
                        row[pa - m.x0] += (uint16_t)(256 - (a & 255));
                        for (int64_t px = pa + 1; px < pb; px++) row[px - m.x0] += 256;
                        row[pb - m.x0] += (uint16_t)(b - (pb << 8));
                    }
                }
            }
        }
    }
    return m;
}

// Composite one mask, once, so overlapping pieces of the same shape do not
// blend with each other.
inline void compositeMask(Canvas& cv, const Mask& m, Rgb col, uint32_t a8) {
    if (a8 == 0 || m.w <= 0 || m.h <= 0) return;
    const uint32_t full = (uint32_t)kSubScanlines * 256;
    for (int64_t y = 0; y < m.h; y++) {
        const uint16_t* row = m.cov.data() + (size_t)y * m.w;
        uint8_t* dst = cv.rgb.data() + (size_t)((m.y0 + y) * cv.w + m.x0) * 3;
        for (int64_t x = 0; x < m.w; x++) {
            const uint32_t c = row[x];
            if (!c) { dst += 3; continue; }
            const uint32_t a = (uint32_t)(((uint64_t)std::min<uint32_t>(c, full) * a8 + full / 2) / full);
            blend(dst, col, a);
            dst += 3;
        }
    }
}

// ── Circles and arcs, from the embedded tables ──────────────────────────────
inline int circleSteps(int64_t radiusQ8) {
    // Chord error under 1/16 pixel: r * (1 - cos(pi/N)) < 1/16.
    const int64_t r = radiusQ8 >> 8;
    if (r <= 2) return 16;
    if (r <= 8) return 32;
    if (r <= 32) return 64;
    if (r <= 128) return 128;
    return 256;
}

inline ContourQ8 circleContour(int64_t cx, int64_t cy, int64_t r) {
    const int n = circleSteps(r);
    const int32_t* ct = cosTable(n);
    const int32_t* st = sinTable(n);
    ContourQ8 out;
    out.reserve((size_t)n);
    for (int k = 0; k < n; k++) {
        out.push_back(PointQ8{ cx + ((int64_t)ct[k] * r >> 30), cy + ((int64_t)st[k] * r >> 30) });
    }
    return out;
}

// Twice the signed area. Coordinates are shifted down by four bits first, so
// the products cannot overflow for any contour the clip box can produce.
inline int64_t signedArea2(const ContourQ8& c) {
    int64_t s = 0;
    for (size_t i = 0; i < c.size(); i++) {
        const PointQ8& a = c[i];
        const PointQ8& b = c[(i + 1) % c.size()];
        s += (a.x >> 4) * (b.y >> 4) - (b.x >> 4) * (a.y >> 4);
    }
    return s;
}

// Every piece of a stroke must wind the SAME way.
//
// Found by comparing a stroked diagonal against its analytic area: the caps
// contributed exactly nothing. The segment quadrilateral and the cap circles
// wound in opposite directions, so under the nonzero rule their overlap summed
// to zero and left a notch at each end -- and a polyline that changes direction
// would have grown the same hole at every join, because the quadrilateral's
// orientation follows the segment's direction.
inline void addOriented(std::vector<ContourQ8>& out, ContourQ8 c) {
    if (signedArea2(c) < 0) std::reverse(c.begin(), c.end());
    out.push_back(std::move(c));
}

// ── Strokes ─────────────────────────────────────────────────────────────────
//
// Each segment becomes a quadrilateral; each interior vertex and each round cap
// becomes a circle. They are filled together, nonzero, as one mask.
inline void strokeContours(const std::vector<PointQ8>& pts, int64_t widthQ8,
                           bool roundCap, std::vector<ContourQ8>& out) {
    const int64_t hw = std::max<int64_t>(widthQ8 / 2, 1);
    for (size_t i = 0; i + 1 < pts.size(); i++) {
        const int64_t dx = pts[i + 1].x - pts[i].x, dy = pts[i + 1].y - pts[i].y;
        const uint64_t len = isqrt64((uint64_t)(dx * dx + dy * dy));
        if (len == 0) continue;
        const int64_t nx = divRound(-dy * hw, (int64_t)len);
        const int64_t ny = divRound(dx * hw, (int64_t)len);
        addOriented(out, ContourQ8{
            PointQ8{pts[i].x + nx,     pts[i].y + ny},
            PointQ8{pts[i + 1].x + nx, pts[i + 1].y + ny},
            PointQ8{pts[i + 1].x - nx, pts[i + 1].y - ny},
            PointQ8{pts[i].x - nx,     pts[i].y - ny}});
    }
    // Round joins at the interior vertices, and round caps at the ends.
    const size_t first = roundCap ? 0 : 1;
    const size_t last = roundCap ? pts.size() : pts.size() - 1;
    for (size_t i = first; i < last; i++) addOriented(out, circleContour(pts[i].x, pts[i].y, hw));
}


// ── Arcs, from the tables ───────────────────────────────────────────────────
//
// SVG gives an arc by its endpoints, so the centre is found with the integer
// square root. The sweep then steps between two TABLE indices: the sector each
// endpoint falls in is found by cross products, and every vertex is the table
// direction scaled by the radius. Rotating a vector repeatedly would drift by a
// bit per step; indexing the table cannot drift at all, and neither needs atan2.
inline int sectorOf(int64_t vx, int64_t vy, int n, const int32_t* ct, const int32_t* st) {
    for (int k = 0; k < n; k++) {
        const int j = (k + 1) % n;
        const int64_t c1 = (int64_t)ct[k] * vy - (int64_t)st[k] * vx;
        const int64_t c2 = (int64_t)ct[j] * vy - (int64_t)st[j] * vx;
        if (c1 >= 0 && c2 < 0) return k;
    }
    return 0;
}

inline void appendArc(ContourQ8& cur, int64_t x0, int64_t y0, int64_t x1, int64_t y1,
                      int64_t r, bool large, bool sweep) {
    const int64_t dx = x1 - x0, dy = y1 - y0;
    // ponytail: an arc spanning more than ~2 million pixels draws as its chord.
    // Squaring anything larger would overflow, and no canvas is a hundredth of that.
    const int64_t kArcLimit = (int64_t)1 << 29;
    if (dx <= -kArcLimit || dx >= kArcLimit || dy <= -kArcLimit || dy >= kArcLimit || r >= kArcLimit) {
        cur.push_back(PointQ8{x1, y1});
        return;
    }
    const uint64_t chord2 = (uint64_t)(dx * dx + dy * dy);
    if (chord2 == 0 || r <= 0) { cur.push_back(PointQ8{x1, y1}); return; }
    const int64_t chord = (int64_t)isqrt64(chord2);
    if (2 * r < chord) r = chord / 2 + 1;          // SVG scales an impossible radius up
    const int64_t half = chord / 2;
    const int64_t hh = (int64_t)isqrt64((uint64_t)(r * r - half * half));
    const int64_t mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
    const int64_t px = divRound(-dy * hh, chord), py = divRound(dx * hh, chord);
    const int sign = (large != sweep) ? 1 : -1;
    const int64_t cx = mx + sign * px, cy = my + sign * py;

    const int n = circleSteps(r);
    const int32_t* ct = cosTable(n);
    const int32_t* st = sinTable(n);
    const int a0 = sectorOf(x0 - cx, y0 - cy, n, ct, st);
    const int a1 = sectorOf(x1 - cx, y1 - cy, n, ct, st);
    const int dir = sweep ? 1 : -1;
    const int steps = ((a1 - a0) * dir % n + n) % n;
    for (int i = 1; i <= steps; i++) {
        const int k = (((a0 + i * dir) % n) + n) % n;
        cur.push_back(PointQ8{ cx + ((int64_t)ct[k] * r >> 30), cy + ((int64_t)st[k] * r >> 30) });
    }
    cur.push_back(PointQ8{x1, y1});
}

// ── Path data: the subset bplot emits ───────────────────────────────────────
//
// M m L l H h V v A a Z z, and nothing else -- an unknown command raises by
// name rather than being skipped, because a silently ignored command is a
// silently wrong picture. Numbers go through the same centi() quantisation as
// every other coordinate, so a path draws where the SVG says it does.
class PathParser {
public:
    PathParser(const std::string& d, uint32_t dpi) : s_(d), dpi_(dpi) {}

    // forStroke keeps open subpaths of two or more points, and closes a Z'd
    // one by returning to its start, so the result can be stroked as polylines.
    std::vector<ContourQ8> parse(size_t maxPoints, bool forStroke = false) {
        std::vector<ContourQ8> out;
        ContourQ8 cur;
        int64_t cx = 0, cy = 0, sx = 0, sy = 0;
        size_t total = 0;
        char cmd = 0;
        auto push = [&](int64_t x, int64_t y) {
            if (++total > maxPoints) {
                throw std::runtime_error("path: more than " + std::to_string(maxPoints) + " points");
            }
            cur.push_back(PointQ8{x, y});
        };
        const size_t keep = forStroke ? 2 : 3;
        auto flush = [&]() { if (cur.size() >= keep) out.push_back(cur); cur.clear(); };

        while (!atEnd()) {
            const char c = s_[i_];
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) { cmd = c; i_++; }
            else if (cmd == 0) throw std::runtime_error("path: expected a command letter");
            else if (cmd == 'M') cmd = 'L';          // an implicit lineto follows a moveto
            else if (cmd == 'm') cmd = 'l';
            switch (cmd) {
                case 'M': case 'm': {
                    flush();
                    int64_t x = coord(), y = coord();
                    if (cmd == 'm') { x += cx; y += cy; }
                    cx = sx = x; cy = sy = y;
                    push(cx, cy);
                    break;
                }
                case 'L': case 'l': {
                    int64_t x = coord(), y = coord();
                    if (cmd == 'l') { x += cx; y += cy; }
                    cx = x; cy = y;
                    push(cx, cy);
                    break;
                }
                case 'H': case 'h': {
                    int64_t x = coord();
                    if (cmd == 'h') x += cx;
                    cx = x;
                    push(cx, cy);
                    break;
                }
                case 'V': case 'v': {
                    int64_t y = coord();
                    if (cmd == 'v') y += cy;
                    cy = y;
                    push(cx, cy);
                    break;
                }
                case 'A': case 'a': {
                    const int64_t rx = coord(), ry = coord();
                    (void)number();                       // x-axis rotation: bplot emits 0
                    const bool large = number() != 0;
                    const bool sweep = number() != 0;
                    int64_t x = coord(), y = coord();
                    if (cmd == 'a') { x += cx; y += cy; }
                    const size_t before = cur.size();
                    appendArc(cur, cx, cy, x, y, std::max(rx, ry), large, sweep);
                    total += cur.size() - before;
                    if (total > maxPoints) throw std::runtime_error("path: an arc exceeded the point limit");
                    cx = x; cy = y;
                    break;
                }
                case 'Z': case 'z': {
                    if (forStroke && !cur.empty()) push(sx, sy);
                    flush();
                    cx = sx; cy = sy;
                    break;
                }
                default:
                    throw std::runtime_error(std::string("path: unsupported command '") + cmd +
                        "' -- this backend draws M, L, H, V, A and Z, in either case");
            }
        }
        flush();
        return out;
    }

private:
    bool atEnd() { skip(); return i_ >= s_.size(); }
    void skip() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == ',' || s_[i_] == '\t' ||
                                  s_[i_] == '\n' || s_[i_] == '\r')) i_++;
    }
    double number() {
        skip();
        const char* start = s_.c_str() + i_;
        char* end = nullptr;
        const double v = std::strtod(start, &end);
        if (end == start) {
            throw std::runtime_error("path: expected a number at offset " + std::to_string(i_));
        }
        i_ += (size_t)(end - start);
        return v;
    }
    int64_t coord() { return toQ8(centi(number()), dpi_); }

    const std::string& s_;
    size_t i_ = 0;
    uint32_t dpi_ = 96;
};

// ── Dashes ──────────────────────────────────────────────────────────────────
// Walks the polyline by integer arc length, splitting it into the runs that are
// drawn. The pattern repeats, and a run that ends mid-segment ends exactly
// where the length says, not at the nearest vertex.
inline std::vector<std::vector<PointQ8>> applyDash(const std::vector<PointQ8>& pts,
                                                   const std::vector<int64_t>& pattern) {
    std::vector<std::vector<PointQ8>> runs;
    if (pattern.empty() || pts.size() < 2) {
        if (pts.size() >= 2) runs.push_back(pts);
        return runs;
    }
    size_t pi = 0;
    int64_t remain = pattern[0];
    bool on = true;
    std::vector<PointQ8> cur{ pts[0] };
    for (size_t i = 0; i + 1 < pts.size(); i++) {
        PointQ8 a = pts[i];
        const PointQ8 b = pts[i + 1];
        int64_t segLen = (int64_t)isqrt64((uint64_t)((b.x - a.x) * (b.x - a.x) +
                                                     (b.y - a.y) * (b.y - a.y)));
        const PointQ8 dir{ b.x - a.x, b.y - a.y };
        int64_t walked = 0;
        while (segLen - walked > 0) {
            const int64_t left = segLen - walked;
            if (remain >= left) {
                remain -= left;
                walked = segLen;
                if (on) cur.push_back(b);
            } else {
                walked += remain;
                const int64_t nx = a.x + divRound(dir.x * walked, segLen);
                const int64_t ny = a.y + divRound(dir.y * walked, segLen);
                if (on) {
                    cur.push_back(PointQ8{nx, ny});
                    if (cur.size() >= 2) runs.push_back(cur);
                    cur.clear();
                } else {
                    cur.clear();
                    cur.push_back(PointQ8{nx, ny});
                }
                on = !on;
                pi = (pi + 1) % pattern.size();
                remain = pattern[pi];
            }
        }
    }
    if (on && cur.size() >= 2) runs.push_back(cur);
    return runs;
}

// ════════════════════════════════════════════════════════════════════════════
//  Text (B6d)
//
//  DejaVu Sans from raster_font.hpp, drawn through the same rasteriser as every
//  other shape: no hinting, no system font, so the pixels are a function of the
//  string alone. Positions are summed in FONT UNITS and scaled once per point,
//  so a long label does not drift by a rounding per glyph.
// ════════════════════════════════════════════════════════════════════════════

constexpr size_t kMaxTextChars = 2000;           // keeps the pen times the size inside 64 bits
constexpr int64_t kMaxFontQ8 = (int64_t)1 << 20; // 4096 px
constexpr int kQuadSteps = 8;    // ponytail: fixed curve subdivision; make it size-aware if
                                 // glyphs over ~200 px ever show facets

// UTF-8 to code points. Anything malformed -- a stray continuation byte, a
// truncated sequence, an overlong form, a surrogate -- becomes U+FFFD, one per
// bad byte, rather than an error: a label is never worth failing a figure over.
inline std::vector<uint32_t> decodeUtf8(const std::string& s) {
    std::vector<uint32_t> out;
    size_t i = 0;
    while (i < s.size()) {
        const uint8_t b = (uint8_t)s[i];
        int n = 0;
        uint32_t cp = 0, min = 0;
        if (b < 0x80)                { out.push_back(b); i++; continue; }
        else if ((b & 0xE0) == 0xC0) { n = 1; cp = b & 0x1F; min = 0x80; }
        else if ((b & 0xF0) == 0xE0) { n = 2; cp = b & 0x0F; min = 0x800; }
        else if ((b & 0xF8) == 0xF0) { n = 3; cp = b & 0x07; min = 0x10000; }
        else                         { out.push_back(0xFFFD); i++; continue; }
        bool ok = i + (size_t)n < s.size();
        for (int k = 1; ok && k <= n; k++) {
            const uint8_t c = (uint8_t)s[i + k];
            if ((c & 0xC0) != 0x80) ok = false; else cp = (cp << 6) | (c & 0x3F);
        }
        if (!ok || cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
            out.push_back(0xFFFD); i++; continue;
        }
        out.push_back(cp);
        i += (size_t)n + 1;
    }
    return out;
}

// The glyph for a code point, or U+FFFD's. ponytail: linear scan of 210
// entries; a sorted table if strings ever get long enough to matter.
inline const int32_t* glyphFor(uint32_t cp) {
    const int32_t* fallback = nullptr;
    for (const auto& g : kGlyphIndex) {
        if ((uint32_t)g[0] == cp) return g;
        if (g[0] == 0xFFFD) fallback = g;
    }
    return fallback;
}

// Width in font units: the sum of the advances.
inline int64_t textAdvance(const std::vector<uint32_t>& cps) {
    int64_t u = 0;
    for (uint32_t cp : cps) u += glyphFor(cp)[3];
    return u;
}

// Outlines for a string whose baseline starts at (x, y) in Q8, `anchor` 0/1/2
// for start/middle/end, rotated clockwise (SVG's sense, y down) by
// rotStep * 360/256 degrees about (x, y).
//
// ponytail: rotation snaps to the circle table's 256 steps (1.4 degrees), which
// is invisible on a tick label and needs no trigonometry. Integer CORDIC is the
// upgrade if an exact angle is ever needed.
inline std::vector<ContourQ8> textContours(const std::vector<uint32_t>& cps, int64_t x, int64_t y,
                                           int64_t sizeQ8, int anchor, int rotStep) {
    std::vector<ContourQ8> out;
    const int64_t upm = kFontUnitsPerEm;
    const int64_t total = textAdvance(cps);
    int64_t pen = anchor == 1 ? -total / 2 : anchor == 2 ? -total : 0;   // font units
    const int k = ((rotStep % 256) + 256) % 256;
    const int64_t c = kCos256[k], s = kSin256[k];

    // A point in font units times `den`, relative to the pen, into device Q8.
    auto place = [&](int64_t fx, int64_t fy, int64_t den) -> PointQ8 {
        const int64_t dx = divRound((fx + pen * den) * sizeQ8, upm * den);
        const int64_t dy = divRound(-fy * sizeQ8, upm * den);       // font y is up
        if (k == 0) return PointQ8{x + dx, y + dy};
        return PointQ8{x + divRound(dx * c - dy * s, (int64_t)1 << 30),
                       y + divRound(dx * s + dy * c, (int64_t)1 << 30)};
    };

    for (uint32_t cp : cps) {
        const int32_t* g = glyphFor(cp);
        ContourQ8 cur;
        int64_t px = 0, py = 0;
        for (int32_t i = g[1]; i < g[1] + g[2]; i++) {
            const int16_t* seg = kGlyphSegments[i];
            if (seg[0] == 0) {
                if (cur.size() >= 3) out.push_back(std::move(cur));
                cur.clear();
                px = seg[1]; py = seg[2];
                cur.push_back(place(px, py, 1));
            } else if (seg[0] == 1) {
                px = seg[1]; py = seg[2];
                cur.push_back(place(px, py, 1));
            } else {
                // B(t) = (N-i)^2 P0 + 2i(N-i) C + i^2 P1, all over N^2: exact integers.
                const int64_t N = kQuadSteps, cx = seg[1], cy = seg[2], ex = seg[3], ey = seg[4];
                for (int64_t t = 1; t <= N; t++) {
                    const int64_t a = (N - t) * (N - t), b = 2 * t * (N - t), d = t * t;
                    cur.push_back(place(a * px + b * cx + d * ex, a * py + b * cy + d * ey, N * N));
                }
                px = ex; py = ey;
            }
        }
        if (cur.size() >= 3) out.push_back(std::move(cur));
        pen += g[3];
    }
    return out;
}

} // namespace bplot_raster
