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

} // namespace bplot_raster
