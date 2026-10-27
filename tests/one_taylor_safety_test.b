// ════════════════════════════════════════════════════════════════════════
//  one_taylor_safety_test.b — the age wall, exhaustively.
//
//  This is not a sample of interesting cases. It is EVERY ordered pair of
//  ages from 13 to 80 -- 4,624 of them -- checked in both directions against
//  a rule stated independently of the implementation. A sampled test over an
//  age boundary tells you about the samples; the boundary is where the bugs
//  are, and there are 68 of them here, one per age.
//
//  The independent rule, written out here rather than imported, so that a
//  mistake in safety/age.b cannot also be the thing the test believes:
//
//    canSee(a, b)     a and b are on the same side of 18, AND if they are
//                     under 18, their ages differ by at most 2.
//    canMatch(a, b)   canSee both ways, AND each is inside the other's
//                     declared range (floor 18 for adults; a minor's range
//                     is the +/-2 rule).
//
//  Also checked: the exact morning of an 18th birthday, leap-day birthdays,
//  the signup floor, and that no range a user can ask for ever lets an adult
//  below 18 or a minor above 17 into view.
//
//  Run:  bantu run tests/one_taylor_safety_test.b
// ════════════════════════════════════════════════════════════════════════

include "../samples/one-taylor/backend/safety/age.b" as age;

$R = {"pass": 0, "fail": 0, "shown": 0};

def ok($cond, $name) {
    if ($cond) { $R.pass = $R.pass + 1; }
    else {
        $R.fail = $R.fail + 1;
        if ($R.shown < 25) { print("  FAIL  " + $name); $R.shown = $R.shown + 1; }
    }
}

def okLoud($cond, $name) {
    if ($cond) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $name); }
}

// A fixed "today", so this test means the same thing on every day it runs.
const $TODAY = "2026-07-01";
age.setToday($TODAY);

// Build a user whose age on $TODAY is exactly $years.
def userAged($years, $seekMin, $seekMax) {
    $birthYear = 2026 - $years;
    return { "dob": str($birthYear) + "-01-01",
             "seek_min_age": $seekMin, "seek_max_age": $seekMax };
}

print("-- the date arithmetic itself --");
okLoud(age.ageOn("2008-07-01", "2026-07-01") == 18, "18 exactly on the birthday");
okLoud(age.ageOn("2008-07-02", "2026-07-01") == 17, "still 17 the day before");
okLoud(age.ageOn("2008-06-30", "2026-07-01") == 18, "18 the day after");
okLoud(age.ageOn("2008-08-01", "2026-07-01") == 17, "17 when the month is later");
okLoud(age.ageOn("2008-12-31", "2026-01-01") == 17, "17 across a year boundary");
okLoud(age.ageOn("2008-02-29", "2026-02-28") == 17, "a leap-day birthday is not yet 18 on Feb 28");
okLoud(age.ageOn("2008-02-29", "2026-03-01") == 18, "and is 18 by Mar 1");
okLoud(age.parseDob("2008-13-01") == null, "month 13 is rejected");
okLoud(age.parseDob("2008-02-32") == null, "day 32 is rejected");
okLoud(age.parseDob("not-a-date") == null, "junk is rejected");
okLoud(age.parseDob("1823-01-01") == null, "a year before 1900 is rejected");
okLoud(age.ageOn("2030-01-01", "2026-07-01") == -1, "a future birth date is -1, not a negative age");

print("-- the band boundary, one age at a time --");
$a = 1;
$bandErrors = 0;
while ($a <= 100) {
    $u = userAged($a, 18, 99);
    $want = "adult";
    if ($a < 18) { $want = "minor"; }
    if (age.bandFor($u["dob"]) != $want) { $bandErrors = $bandErrors + 1; }
    $a = $a + 1;
}
okLoud($bandErrors == 0, "bandFor is correct for every age from 1 to 100");

okLoud(!age.isSignupAgeOk(userAged(12, 13, 17)["dob"]), "a 12-year-old cannot sign up");
okLoud(age.isSignupAgeOk(userAged(13, 13, 17)["dob"]), "a 13-year-old can");
okLoud(age.isSignupAgeOk(userAged(17, 13, 17)["dob"]), "a 17-year-old can");
okLoud(age.isSignupAgeOk(userAged(18, 18, 28)["dob"]), "an 18-year-old can");
okLoud(!age.isSignupAgeOk(userAged(130, 18, 99)["dob"]), "an impossible age cannot");

// ── the matrix ──────────────────────────────────────────────────────────────
print("-- canSee over every ordered pair of ages 13..80 --");

// The rule, restated independently of safety/age.b.
def expectedCanSee($va, $ta) {
    $vMinor = $va < 18;
    $tMinor = $ta < 18;
    if ($vMinor != $tMinor) { return false; }
    if ($vMinor) {
        $d = $va - $ta;
        if ($d < 0) { $d = 0 - $d; }
        return $d <= 2;
    }
    return true;
}

$mismatches = 0;
$crossings = 0;
$pairs = 0;
$va = 13;
while ($va <= 80) {
    $ta = 13;
    while ($ta <= 80) {
        // Widest legal range on both sides, so canSee is tested on its own
        // terms and not through a narrow preference.
        $v = userAged($va, 18, 120);
        $t = userAged($ta, 18, 120);
        $got = age.canSee($v, $t);
        $want = expectedCanSee($va, $ta);
        if ($got != $want) { $mismatches = $mismatches + 1; }
        // Independently: a crossing is a crossing, whatever the rule says.
        if ($got && (($va < 18) != ($ta < 18))) { $crossings = $crossings + 1; }
        $pairs = $pairs + 1;
        $ta = $ta + 1;
    }
    $va = $va + 1;
}
okLoud($pairs == 4624, "the matrix covered 68 x 68 = 4,624 ordered pairs");
okLoud($mismatches == 0, "canSee matches the independent rule on every pair");
okLoud($crossings == 0, "NOT ONE pair crosses the 18 boundary in either direction");

print("-- canSee is symmetric, which the follow path depends on --");
$asym = 0;
$va = 13;
while ($va <= 80) {
    $ta = 13;
    while ($ta <= 80) {
        $v = userAged($va, 18, 120);
        $t = userAged($ta, 18, 120);
        if (age.canSee($v, $t) != age.canSee($t, $v)) { $asym = $asym + 1; }
        $ta = $ta + 1;
    }
    $va = $va + 1;
}
okLoud($asym == 0, "canSee(a,b) == canSee(b,a) for all 4,624 pairs");

print("-- canMatch: the wall holds, and the range applies on top of it --");
def expectedCanMatch($va, $ta, $vMin, $vMax, $tMin, $tMax) {
    if (!expectedCanSee($va, $ta)) { return false; }
    if ($va < 18) { return true; }          // minors: the +/-2 rule is all of it
    // Adult ranges, with the floor applied as clampRange would.
    $vLo = $vMin; if ($vLo < 18) { $vLo = 18; }
    $tLo = $tMin; if ($tLo < 18) { $tLo = 18; }
    if ($ta < $vLo || $ta > $vMax) { return false; }
    if ($va < $tLo || $va > $tMax) { return false; }
    return true;
}

$matchMismatch = 0;
$matchCrossings = 0;
$va = 13;
while ($va <= 80) {
    $ta = 13;
    while ($ta <= 80) {
        // Deliberately asymmetric, deliberately illegal on the low side: both
        // viewers ask to see from 14, which clampRange must refuse.
        $v = userAged($va, 14, $va + 8);
        $t = userAged($ta, 14, $ta + 8);
        $got = age.canMatch($v, $t);
        $want = expectedCanMatch($va, $ta, 14, $va + 8, 14, $ta + 8);
        if ($got != $want) { $matchMismatch = $matchMismatch + 1; }
        if ($got && (($va < 18) != ($ta < 18))) {
            $matchCrossings = $matchCrossings + 1;
        }
        if ($got && $va >= 18 && $ta < 18) { $matchCrossings = $matchCrossings + 1; }
        $ta = $ta + 1;
    }
    $va = $va + 1;
}
okLoud($matchMismatch == 0, "canMatch matches the independent rule on every pair");
okLoud($matchCrossings == 0, "no adult is ever matched with a minor, whatever range they ask for");

print("-- a requested range can never breach the band --");
// Every age, asking for every absurd range, must still come back legal.
$rangeErrors = 0;
$a = 13;
while ($a <= 80) {
    each ($lo in [0, 1, 10, 12, 13, 17, 18, 19, 40]) {
        each ($hi in [0, 5, 17, 18, 25, 99, 150, 1000]) {
            $r = age.clampRange($a, $lo, $hi);
            if ($a < 18) {
                // A minor's range is not a preference: it must come back as
                // the +/-2 band whatever was asked for.
                if ($r["min"] < 13 || $r["max"] > 17) { $rangeErrors = $rangeErrors + 1; }
                if ($r["min"] > $r["max"]) { $rangeErrors = $rangeErrors + 1; }
            } else {
                if ($r["min"] < 18) { $rangeErrors = $rangeErrors + 1; }
                if ($r["max"] > 120) { $rangeErrors = $rangeErrors + 1; }
                if ($r["max"] < $r["min"]) { $rangeErrors = $rangeErrors + 1; }
            }
        }
    }
    $a = $a + 1;
}
okLoud($rangeErrors == 0, "clampRange is legal for 68 ages x 72 absurd requests");

print("-- the SQL pre-filter never admits the other band --");
$filterErrors = 0;
$a = 13;
while ($a <= 80) {
    $u = userAged($a, 18, 120);
    $vp = age.visibilityPairs($u);
    $wantBand = "adult";
    if ($a < 18) { $wantBand = "minor"; }
    // pairs[1] is ["author_band", band]; pairs[2]/[3] are the age bounds.
    if ($vp[1][1] != $wantBand) { $filterErrors = $filterErrors + 1; }
    if ($a < 18) {
        if ($vp[3][1] > 17) { $filterErrors = $filterErrors + 1; }
        if ($vp[2][1] < 13) { $filterErrors = $filterErrors + 1; }
    } else {
        if ($vp[2][1] < 18) { $filterErrors = $filterErrors + 1; }
    }
    if ($vp[0][1] != "live") { $filterErrors = $filterErrors + 1; }
    $a = $a + 1;
}
okLoud($filterErrors == 0, "visibilityPairs carries the right band and bounds for every age");

print("-- the 18th birthday, to the day --");
// A user born 2008-07-01 is 17 on 2026-06-30 and 18 on 2026-07-01.
age.setToday("2026-06-30");
$almost = { "dob": "2008-07-01", "seek_min_age": 18, "seek_max_age": 99 };
$teen = { "dob": "2010-07-01", "seek_min_age": 13, "seek_max_age": 17 };
$adult = { "dob": "1996-07-01", "seek_min_age": 18, "seek_max_age": 99 };
okLoud(age.bandFor($almost["dob"]) == "minor", "the day before: still a minor");
okLoud(age.canSee($almost, $teen), "the day before: can see a 16-year-old");
okLoud(!age.canSee($almost, $adult), "the day before: cannot see a 30-year-old");

age.setToday("2026-07-01");
okLoud(age.bandFor($almost["dob"]) == "adult", "on the morning: an adult");
okLoud(!age.canSee($almost, $teen), "on the morning: the 16-year-old is gone");
okLoud(age.canSee($almost, $adult), "on the morning: the 30-year-old appears");

// And the reverse direction, which is the one that actually protects anybody.
okLoud(!age.canSee($teen, $almost), "and the 16-year-old can no longer see them either");

age.clearToday();
print("");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); } else { print("  RESULT: FAILED"); }
