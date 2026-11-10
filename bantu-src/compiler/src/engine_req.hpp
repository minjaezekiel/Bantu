#pragma once
// ═══════════════════════════════════════════════════════════════════════════
//  The engine requirement: which binary a package needs (ecosystem stage 2).
//
//  A Bantu package does not only depend on other packages. It depends on the
//  INTERPRETER, because its libraries are written on native builtins — numba on
//  nd_*, pillow on raster_*, arctic on col_*. Nothing in a manifest could say
//  so, and the failure mode was the worst kind: the package installs happily,
//  and then a program using it dies on `Undefined variable: nd_zeros` with no
//  hint that the binary is the problem.
//
//  Two things live here:
//
//    1. VERSION REQUIREMENTS -- parsing `"bantu": ">=1.4.0"` and deciding
//       whether the running build satisfies it. Malformed ranges are rejected
//       BY NAME rather than ignored: a range nobody can parse must not quietly
//       become "any version", because that is the same silence it replaced.
//
//    2. THE NATIVE FAMILY FLOOR TABLE -- which release first provided each
//       family of builtins. This is what lets `bantu engine <dir>` compute the
//       floor a package needs from the calls its sources actually make, so the
//       number in the manifest is derived rather than remembered.
//
//  The table's floors are evidence, not recollection: the released versions are
//  1.1.0 through 1.3.2, `git ls-tree` at 1.3.2 has none of the native sources,
//  and every family below therefore first ships in 1.4.0. When that stops being
//  true -- when a family lands in 1.5.0 -- the new entry says 1.5.0 and the
//  rest stay put.
// ═══════════════════════════════════════════════════════════════════════════

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

namespace bantu_engine {

// ── semantic versions ───────────────────────────────────────────────────────
struct SemVer {
    int  major = 0, minor = 0, patch = 0;
    bool ok    = false;
};

inline SemVer parseSemVer(const std::string& in) {
    SemVer v;
    std::string s = in;
    // Trim, and allow a leading v: "v1.4.0" is what people type.
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.erase(s.begin());
    while (!s.empty() && (s.back()  == ' ' || s.back()  == '\t')) s.pop_back();
    if (!s.empty() && (s[0] == 'v' || s[0] == 'V')) s.erase(s.begin());
    if (s.empty()) return v;

    int part[3] = { 0, 0, 0 };
    int which = 0;
    bool anyDigit = false;
    for (size_t i = 0; i < s.size(); i++) {
        const char c = s[i];
        if (c >= '0' && c <= '9') {
            if (part[which] > 100000) return v;            // absurd, refuse
            part[which] = part[which] * 10 + (c - '0');
            anyDigit = true;
        } else if (c == '.') {
            if (!anyDigit) return v;                       // "1..0", ".1"
            if (++which > 2) return v;                     // "1.2.3.4"
            anyDigit = false;
        } else {
            return v;                                      // letters, -, +, spaces
        }
    }
    if (!anyDigit) return v;                               // "1.2."
    // "1" and "1.4" are accepted and mean 1.0.0 and 1.4.0: a floor stated to
    // less precision is still a floor.
    v.major = part[0]; v.minor = part[1]; v.patch = part[2];
    v.ok = true;
    return v;
}

inline std::string semVerString(const SemVer& v) {
    return std::to_string(v.major) + "." + std::to_string(v.minor) + "." +
           std::to_string(v.patch);
}

// -1, 0, 1
inline int compareSemVer(const SemVer& a, const SemVer& b) {
    if (a.major != b.major) return a.major < b.major ? -1 : 1;
    if (a.minor != b.minor) return a.minor < b.minor ? -1 : 1;
    if (a.patch != b.patch) return a.patch < b.patch ? -1 : 1;
    return 0;
}

// ── requirements ────────────────────────────────────────────────────────────
struct VersionReq {
    enum Kind { ANY, GTE, GT, CARET, TILDE } kind = ANY;
    SemVer      v;
    bool        ok = false;        // false => `error` says why, and NOTHING is assumed
    std::string error;
    std::string raw;
};

// A range nobody can parse is an ERROR, never "any". Everything refused is
// named in the message, because the author needs to know which token lost.
inline VersionReq parseVersionReq(const std::string& in) {
    VersionReq r;
    r.raw = in;
    std::string s = in;
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.erase(s.begin());
    while (!s.empty() && (s.back()  == ' ' || s.back()  == '\t')) s.pop_back();

    if (s.empty() || s == "*" || s == "x" || s == "latest" || s == "any") {
        r.kind = VersionReq::ANY;
        r.ok   = true;
        return r;
    }
    if (s.find("||") != std::string::npos) {
        r.error = "alternatives (\"||\") are not supported: state one floor";
        return r;
    }
    if (s.find(' ') != std::string::npos || s.find('\t') != std::string::npos) {
        r.error = "compound ranges are not supported: state one floor, not \"" + s + "\"";
        return r;
    }
    if (s[0] == '<') {
        r.error = "an upper bound (\"" + s + "\") is not supported -- nobody can "
                  "know at publish time which future binary breaks a package, and a "
                  "guess about the future makes every later release look broken";
        return r;
    }
    if (s.find('-') != std::string::npos || s.find('+') != std::string::npos) {
        r.error = "prerelease and build metadata are not supported in an engine "
                  "requirement: \"" + s + "\"";
        return r;
    }

    std::string num = s;
    if (s.rfind(">=", 0) == 0)      { r.kind = VersionReq::GTE;   num = s.substr(2); }
    else if (s.rfind("=>", 0) == 0) { r.kind = VersionReq::GTE;   num = s.substr(2); }
    else if (s[0] == '>')           { r.kind = VersionReq::GT;    num = s.substr(1); }
    else if (s[0] == '^')           { r.kind = VersionReq::CARET; num = s.substr(1); }
    else if (s[0] == '~')           { r.kind = VersionReq::TILDE; num = s.substr(1); }
    else if (s[0] == '=')           { r.kind = VersionReq::GTE;   num = s.substr(1); }
    else {
        // A bare version is a FLOOR, not an exact pin. An engine requirement of
        // exactly one build is almost never what an author means, and a package
        // pinned to one binary could never be installed on the next release.
        r.kind = VersionReq::GTE;
    }

    const SemVer v = parseSemVer(num);
    if (!v.ok) {
        r.error = "\"" + s + "\" is not a version requirement bantu understands "
                  "(try \">=1.4.0\", \"^1.4.0\", \"~1.4.0\" or \"*\")";
        return r;
    }
    r.v  = v;
    r.ok = true;
    return r;
}

inline bool satisfies(const VersionReq& req, const SemVer& running) {
    if (!req.ok) return false;                 // a requirement we cannot read is not met
    switch (req.kind) {
        case VersionReq::ANY:   return true;
        case VersionReq::GTE:   return compareSemVer(running, req.v) >= 0;
        case VersionReq::GT:    return compareSemVer(running, req.v) >  0;
        case VersionReq::CARET: return running.major == req.v.major &&
                                       compareSemVer(running, req.v) >= 0;
        case VersionReq::TILDE: return running.major == req.v.major &&
                                       running.minor == req.v.minor &&
                                       compareSemVer(running, req.v) >= 0;
    }
    return false;
}

inline std::string describeReq(const VersionReq& req) {
    if (!req.ok) return "unreadable (" + req.error + ")";
    switch (req.kind) {
        case VersionReq::ANY:   return "any version";
        case VersionReq::GTE:   return ">= " + semVerString(req.v);
        case VersionReq::GT:    return "> "  + semVerString(req.v);
        case VersionReq::CARET: return ">= " + semVerString(req.v) + " and < " +
                                       std::to_string(req.v.major + 1) + ".0.0";
        case VersionReq::TILDE: return ">= " + semVerString(req.v) + " and < " +
                                       std::to_string(req.v.major) + "." +
                                       std::to_string(req.v.minor + 1) + ".0";
    }
    return "?";
}

// ── the native family floor table ───────────────────────────────────────────
// prefix -> the first RELEASED version that provides that family.
struct Family {
    const char* prefix;
    const char* floor;
    const char* provides;      // what a reader needs to know it is
    bool        exact;         // true: this NAME, not everything starting with it
};

// A prefix family claims every builtin that starts with it, which is what
// makes `nd_fft` inherit `nd_`'s floor the day it is written. An exact entry
// claims one name only -- so a future `sorted` does not silently inherit
// `sort`'s floor, it fails the coverage test and gets a decision.

// Ordered so that a longer prefix is tried before a shorter one it contains,
// because familyFor returns the first match.
//
// Every floor here is 1.4.0, and that is not laziness. The released versions
// are 1.1.0 (2026-06-17) through 1.3.2 (2026-09-06), and 1.3.2 registered
// FORTY-NINE builtins -- abs, ceil, len, open, read, str, sua and the like
// (tests/builtins_v1.3.2.txt has the list, taken from git). Not one nd_*,
// col_* or bp_* among them; `sort`, `PI` and `mkdir` did not exist either.
// 319 of today's 368 builtins arrived after that release, so 1.4.0 is the
// first version that can honestly be named as providing any of them.
//
// A family added in a later release gets that release's number here, and these
// stay put. tests/engine_floor_table_test.sh fails if a builtin exists that is
// neither in the 1.3.2 baseline nor matched below.
inline const std::vector<Family>& families() {
    static const std::vector<Family> f = {
        // ── prefix families ────────────────────────────────────────────────
        { "nd_",        "1.4.0", "numba's n-dimensional arrays", false },
        { "col_",       "1.4.0", "arctic's columnar primitives", false },
        { "bp_",        "1.4.0", "bplot's raster backend", false },
        { "webpush_",   "1.4.0", "sua's Web Push primitives", false },
        { "read_",      "1.4.0", "arctic's readers (csv, feather, parquet, sqlite)", false },
        { "write_",     "1.4.0", "arctic's writers (csv, feather, parquet)", false },
        { "gc_",        "1.4.0", "the collector's controls", false },
        { "aead_",      "1.4.0", "the AEAD primitives", false },
        { "b64url_",    "1.4.0", "the base64url primitives", false },
        { "native_",    "1.4.0", "the native hash helpers", false },
        { "ct_",        "1.4.0", "the constant-time comparison", false },
        { "file_",      "1.4.0", "the file predicates", false },
        { "has_",       "1.4.0", "the capability probes", false },
        { "pwhash",     "1.4.0", "the password-hash primitives", false },
        { "sodium_",    "1.4.0", "the libsodium probe", false },

        // ── exact names, which belong to no prefix family ─────────────────
        // The 32-bit integer operations exist for hashing and checksums; the
        // byte/hex/base64 pairs for the crypto suite; the math constants, the
        // file helpers and sort/join/reverse/ord for the standard library.
        { "add32",      "1.4.0", "32-bit integer arithmetic", true },
        { "mul32",      "1.4.0", "32-bit integer arithmetic", true },
        { "band",       "1.4.0", "32-bit bitwise operations", true },
        { "bor",        "1.4.0", "32-bit bitwise operations", true },
        { "bxor",       "1.4.0", "32-bit bitwise operations", true },
        { "bnot",       "1.4.0", "32-bit bitwise operations", true },
        { "shl",        "1.4.0", "32-bit bitwise operations", true },
        { "shr",        "1.4.0", "32-bit bitwise operations", true },
        { "rotl",       "1.4.0", "32-bit rotations", true },
        { "rotr",       "1.4.0", "32-bit rotations", true },
        { "bytes",      "1.4.0", "the byte-buffer type", true },
        { "frombytes",  "1.4.0", "the byte-buffer type", true },
        { "tohex",      "1.4.0", "hex encoding", true },
        { "fromhex",    "1.4.0", "hex encoding", true },
        { "b64encode",  "1.4.0", "base64 encoding", true },
        { "b64decode",  "1.4.0", "base64 encoding", true },
        { "randbytes",  "1.4.0", "the CSPRNG", true },
        { "inflate",    "1.4.0", "DEFLATE decompression", true },
        { "gunzip",     "1.4.0", "gzip decompression", true },
        { "nd",         "1.4.0", "numba's array constructor", true },
        { "col",        "1.4.0", "arctic's column constructor", true },
        { "clamp",      "1.4.0", "the clamp helper", true },
        { "sort",       "1.4.0", "list sorting", true },
        { "join",       "1.4.0", "list joining", true },
        { "reverse",    "1.4.0", "list reversal", true },
        { "ord",        "1.4.0", "character codes", true },
        { "eprint",     "1.4.0", "stderr output", true },
        { "exists",     "1.4.0", "the path predicates", true },
        { "isfile",     "1.4.0", "the path predicates", true },
        { "isdir",      "1.4.0", "the path predicates", true },
        { "filesize",   "1.4.0", "the path predicates", true },
        { "listdir",    "1.4.0", "directory listing", true },
        { "mkdir",      "1.4.0", "directory creation", true },
        { "mkdirs",     "1.4.0", "directory creation", true },
        { "unlink",     "1.4.0", "file removal", true },
        { "PI",         "1.4.0", "the math constants", true },
        { "TAU",        "1.4.0", "the math constants", true },
        { "E",          "1.4.0", "the math constants", true },
        { "INF",        "1.4.0", "the math constants", true },
        { "NAN",        "1.4.0", "the math constants", true },

        // The scalar math surface. v1.3.2 had abs, ceil, cos, floor, log, max,
        // min, pow, round, sin, sqrt and tan; everything below arrived after
        // it. These were FOUND by `bantu engine --check-table` and not by
        // reading the sources: they are registered through a helper the
        // source grep did not know about, which is the whole argument for
        // asking the running binary what it provides instead of grepping for
        // registration calls.
        { "exp",        "1.4.0", "the scalar math functions", true },
        { "expm1",      "1.4.0", "the scalar math functions", true },
        { "log2",       "1.4.0", "the scalar math functions", true },
        { "log10",      "1.4.0", "the scalar math functions", true },
        { "log1p",      "1.4.0", "the scalar math functions", true },
        { "cbrt",       "1.4.0", "the scalar math functions", true },
        { "asin",       "1.4.0", "the scalar math functions", true },
        { "acos",       "1.4.0", "the scalar math functions", true },
        { "atan",       "1.4.0", "the scalar math functions", true },
        { "atan2",      "1.4.0", "the scalar math functions", true },
        { "sinh",       "1.4.0", "the scalar math functions", true },
        { "cosh",       "1.4.0", "the scalar math functions", true },
        { "tanh",       "1.4.0", "the scalar math functions", true },
        { "asinh",      "1.4.0", "the scalar math functions", true },
        { "acosh",      "1.4.0", "the scalar math functions", true },
        { "atanh",      "1.4.0", "the scalar math functions", true },
        { "hypot",      "1.4.0", "the scalar math functions", true },
        { "copysign",   "1.4.0", "the scalar math functions", true },
        { "fmod",       "1.4.0", "the scalar math functions", true },
        { "trunc",      "1.4.0", "the scalar math functions", true },
        { "sign",       "1.4.0", "the scalar math functions", true },
        { "degrees",    "1.4.0", "the scalar math functions", true },
        { "radians",    "1.4.0", "the scalar math functions", true },
        { "isnan",      "1.4.0", "the float predicates", true },
        { "isinf",      "1.4.0", "the float predicates", true },
        { "isfinite",   "1.4.0", "the float predicates", true },
    };
    return f;
}

// Which family does this builtin belong to? nullptr if none -- which for the
// table-coverage test means "a family nobody has given a floor".
inline const Family* familyFor(const std::string& builtin) {
    for (const Family& f : families()) {
        const std::string p = f.prefix;
        if (f.exact) {
            if (builtin == p) return &f;
        } else if (builtin.size() >= p.size() && builtin.compare(0, p.size(), p) == 0) {
            return &f;
        }
    }
    return nullptr;
}

// The floor a piece of Bantu source implies: the highest floor among the native
// builtins it calls. Names are matched as whole identifiers, so a variable
// called `$nd_thing` or the word "nd_sum" inside a comment or string cannot
// raise a package's floor.
struct SourceFloor {
    SemVer      floor;                      // 0.0.0 when nothing native is used
    std::string reason;                     // the builtin that set it
    std::vector<std::string> builtins;      // every native builtin seen, sorted
};

inline bool identChar(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_';
}

// Comments and string literals are skipped, because a floor must come from
// what the code CALLS. A comment reading `// nd_fft( is next` or a message
// containing "nd_max(" would otherwise raise a package's declared floor to a
// version it does not need -- conservative, and still wrong, and reported to
// the author as a fact about their code.
inline void scanSourceFloor(const std::string& text, SourceFloor& out) {
    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        const char c = text[i];
        // line comment
        if (c == '/' && i + 1 < n && text[i + 1] == '/') {
            while (i < n && text[i] != '\n') i++;
            continue;
        }
        // block comment
        if (c == '/' && i + 1 < n && text[i + 1] == '*') {
            i += 2;
            while (i + 1 < n && !(text[i] == '*' && text[i + 1] == '/')) i++;
            i = (i + 1 < n) ? i + 2 : n;
            continue;
        }
        // string literal, single or double quoted, with escapes
        if (c == '"' || c == '\'') {
            const char quote = c;
            i++;
            while (i < n) {
                if (text[i] == '\\') { i += 2; continue; }
                if (text[i] == quote) { i++; break; }
                i++;
            }
            continue;
        }
        if (!identChar(c)) { i++; continue; }
        const size_t start = i;
        while (i < n && identChar(text[i])) i++;
        // A name preceded by $ or . is a variable or a member, not a builtin call.
        if (start > 0 && (text[start - 1] == '$' || text[start - 1] == '.')) continue;
        const std::string word = text.substr(start, i - start);
        const Family* f = familyFor(word);
        if (!f) continue;
        // It must be CALLED: the next non-space character is '('.
        size_t j = i;
        while (j < n && (text[j] == ' ' || text[j] == '\t')) j++;
        if (j >= n || text[j] != '(') continue;

        if (std::find(out.builtins.begin(), out.builtins.end(), word) == out.builtins.end()) {
            out.builtins.push_back(word);
        }
        const SemVer fv = parseSemVer(f->floor);
        if (!out.floor.ok || compareSemVer(fv, out.floor) > 0) {
            out.floor  = fv;
            out.reason = word;
        }
    }
    std::sort(out.builtins.begin(), out.builtins.end());
}

}  // namespace bantu_engine
