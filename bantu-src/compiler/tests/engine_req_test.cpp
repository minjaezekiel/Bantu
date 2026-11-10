// ════════════════════════════════════════════════════════════════════════
//  engine_req_test.cpp — the engine requirement parser (ecosystem stage 2).
//
//  Two properties carry the whole feature:
//
//    * a requirement that CAN be read is applied exactly;
//    * a requirement that CANNOT be read is an ERROR, never "any version".
//
//  The second is the one worth a test. A range parser that silently treats
//  what it does not understand as "no constraint" restores the exact silence
//  the field was added to end: the package installs, and then dies inside a
//  library on `Undefined variable: nd_zeros`.
//
//  Build and run:
//    g++ -std=c++17 -O2 -I src tests/engine_req_test.cpp -o /tmp/engtest && /tmp/engtest
// ════════════════════════════════════════════════════════════════════════

#include <cstdio>
#include <string>

#include "engine_req.hpp"

static int failures = 0;
static void ok(bool cond, const std::string& name) {
    std::printf(cond ? "  ok    %s\n" : "  FAIL  %s\n", name.c_str());
    if (!cond) failures++;
}

using namespace bantu_engine;

static void expectSatisfied(const char* range, const char* running, bool want) {
    const VersionReq r = parseVersionReq(range);
    const SemVer v = parseSemVer(running);
    const bool got = r.ok && satisfies(r, v);
    ok(got == want, std::string("\"") + range + "\" vs " + running + " -> " +
                    (want ? "allowed" : "refused"));
}

static void expectRejected(const char* range, const char* becauseContains) {
    const VersionReq r = parseVersionReq(range);
    bool good = !r.ok && r.error.find(becauseContains) != std::string::npos;
    ok(good, std::string("\"") + range + "\" is rejected, naming \"" +
             becauseContains + "\"" + (r.ok ? "  [it was ACCEPTED]" : ""));
}

int main() {
    std::printf("\n  engine requirements\n  ═════════════════════\n\n");

    // ── versions ────────────────────────────────────────────────────────────
    ok(parseSemVer("1.4.0").ok && semVerString(parseSemVer("1.4.0")) == "1.4.0",
       "1.4.0 parses");
    ok(semVerString(parseSemVer("v2.0.1")) == "2.0.1", "a leading v is allowed");
    ok(semVerString(parseSemVer("1.4")) == "1.4.0", "1.4 means 1.4.0");
    ok(semVerString(parseSemVer("2")) == "2.0.0", "2 means 2.0.0");
    ok(semVerString(parseSemVer(" 1.4.0 ")) == "1.4.0", "surrounding space is trimmed");
    ok(!parseSemVer("").ok,        "the empty string is not a version");
    ok(!parseSemVer("1.4.0-rc1").ok, "a prerelease suffix is not a version here");
    ok(!parseSemVer("1..0").ok,    "1..0 is not a version");
    ok(!parseSemVer(".1").ok,      ".1 is not a version");
    ok(!parseSemVer("1.2.").ok,    "1.2. is not a version");
    ok(!parseSemVer("1.2.3.4").ok, "1.2.3.4 is not a version");
    ok(!parseSemVer("latest").ok,  "\"latest\" is not a version");
    ok(!parseSemVer("1.x").ok,     "1.x is not a version");

    // ordering, including the trap that string comparison gets wrong
    ok(compareSemVer(parseSemVer("1.4.0"),  parseSemVer("1.4.0")) == 0, "equal versions");
    ok(compareSemVer(parseSemVer("1.10.0"), parseSemVer("1.9.0"))  > 0,
       "1.10.0 > 1.9.0 (a string compare says otherwise)");
    ok(compareSemVer(parseSemVer("2.0.0"),  parseSemVer("1.99.99")) > 0, "major wins");
    ok(compareSemVer(parseSemVer("1.4.1"),  parseSemVer("1.4.0"))  > 0, "patch counts");

    // ── what is accepted ────────────────────────────────────────────────────
    std::printf("\n");
    expectSatisfied(">=1.4.0", "1.4.0", true);
    expectSatisfied(">=1.4.0", "1.4.1", true);
    expectSatisfied(">=1.4.0", "2.0.0", true);
    expectSatisfied(">=1.4.0", "1.3.2", false);
    expectSatisfied(">=1.10.0", "1.9.0", false);
    expectSatisfied(">1.4.0", "1.4.0", false);
    expectSatisfied(">1.4.0", "1.4.1", true);
    expectSatisfied("^1.4.0", "1.9.0", true);
    expectSatisfied("^1.4.0", "2.0.0", false);     // caret pins the major
    expectSatisfied("^1.4.0", "1.3.0", false);
    expectSatisfied("~1.4.0", "1.4.9", true);
    expectSatisfied("~1.4.0", "1.5.0", false);     // tilde pins the minor
    expectSatisfied("*", "1.0.0", true);
    expectSatisfied("", "1.0.0", true);
    // A bare version is a FLOOR, not a pin: an engine requirement of exactly
    // one build could never be satisfied by the next release.
    expectSatisfied("1.4.0", "1.4.0", true);
    expectSatisfied("1.4.0", "1.9.0", true);
    expectSatisfied("1.4.0", "1.3.0", false);
    expectSatisfied("=1.4.0", "1.5.0", true);
    expectSatisfied(">=1.4", "1.4.0", true);

    // ── what is refused, and it is refused BY NAME ──────────────────────────
    std::printf("\n");
    expectRejected(">=1.2 <2.0", "compound");
    expectRejected("1.2 || 1.3", "alternatives");
    expectRejected("<2.0.0", "upper bound");
    expectRejected("<=1.9.0", "upper bound");
    expectRejected(">=1.4.0-rc1", "prerelease");
    expectRejected(">=1.4.0+build7", "prerelease");
    expectRejected(">=banana", "not a version requirement");
    expectRejected("~>1.4.0", "not a version requirement");
    expectRejected(">=", "not a version requirement");
    // and an unreadable requirement is NOT satisfied by anything
    {
        const VersionReq r = parseVersionReq(">=1.2 <2.0");
        ok(!satisfies(r, parseSemVer("9.9.9")),
           "an unreadable requirement is not satisfied even by a huge version");
    }

    // ── the floor table ─────────────────────────────────────────────────────
    std::printf("\n");
    ok(familyFor("nd_sum") != nullptr,        "nd_sum has a family");
    ok(familyFor("nd_fft") != nullptr,        "a builtin that does not exist YET inherits nd_*");
    ok(familyFor("col_mean") != nullptr,      "col_mean has a family");
    ok(familyFor("bp_png") != nullptr,        "bp_png has a family");
    ok(familyFor("sort") != nullptr,          "sort has a family");
    ok(familyFor("sorted") == nullptr,
       "sorted does NOT inherit sort's floor (exact entries are exact)");
    ok(familyFor("len") == nullptr,           "len is in the released baseline, no floor");
    ok(familyFor("print") == nullptr,         "print has no floor");
    ok(std::string(familyFor("nd_sum")->floor) == "1.4.0", "nd_* floors at 1.4.0");

    // ── the source scan ─────────────────────────────────────────────────────
    std::printf("\n");
    {
        SourceFloor sf;
        scanSourceFloor("def f($a) { return nd_sum($a, null, null, null); }", sf);
        ok(sf.floor.ok && semVerString(sf.floor) == "1.4.0", "a call raises the floor");
        ok(sf.reason == "nd_sum", "and names the builtin that did it");
    }
    {
        SourceFloor sf;
        scanSourceFloor("def f() { return 1 + 2; }", sf);
        ok(!sf.floor.ok, "pure Bantu implies no floor at all");
    }
    {
        // The three ways a name can LOOK like a call and not be one.
        SourceFloor sf;
        scanSourceFloor("$nd_sum = 1; // nd_mean( is a comment\n"
                        "$s = \"nd_max(\"; $obj.nd_min(1);", sf);
        ok(!sf.floor.ok,
           "a variable, a member call and a string do not raise the floor");
    }
    {
        SourceFloor sf;
        scanSourceFloor("nd_sum(x); nd_sum(y); bp_png(z);", sf);
        ok(sf.builtins.size() == 2, "each builtin is listed once");
    }
    {
        SourceFloor sf;
        scanSourceFloor("nd_sum ($a);", sf);
        ok(sf.floor.ok, "space before the paren is still a call");
    }

    std::printf("\n  %s\n\n", failures == 0 ? "RESULT: ALL GREEN"
                                            : "RESULT: FAILURES PRESENT");
    return failures == 0 ? 0 : 1;
}
