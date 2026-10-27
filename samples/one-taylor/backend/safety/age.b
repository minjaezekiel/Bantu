// ============================================================================
// one-taylor/backend/safety/age.b — the age wall
//
// THE invariant of this application:
//
//   No account on one side of the 18 boundary can observe, reach, or be
//   reached by an account on the other side -- in the feed, in search, in
//   recommendations, in a follow, in a like, in a report, or by typing a URL.
//
// It is implemented here, once, and composed into every read path. A rule
// written out eleven times is a rule with eleven chances to be written wrong,
// and the eleventh is the one that ships.
//
//   band       minor 13-17 | adult 18+          (under 13 cannot sign up)
//   minor      same band AND |ageA - ageB| <= 2
//   adult      same band AND target inside the viewer's range, floor 18
//
// canSee is DIRECTIONAL -- a 40-year-old with a narrow range does not see a
// 25-year-old who would see them. canInteract is symmetric by construction,
// because a one-sided follow is still contact.
//
// Exposes:
//   today() / parseDob($s) / ageOn($dob, $date) / ageNow($dob)
//   bandForAge($age) / bandFor($dob)
//   isSignupAgeOk($dob)
//   defaultRange($age) -> {min, max}
//   clampRange($age, $min, $max)
//   canSee($viewer, $target) / canInteract($a, $b)
//   visibilityPairs($viewer) -> ORM filter pairs for `posts`
//   userVisibilityPairs($viewer) -> ORM filter pairs for `users`
// ============================================================================

include "../config.b" as cfg;

const $MINOR = "minor";
const $ADULT = "adult";

// ── dates ───────────────────────────────────────────────────────────────────
// Dates are "YYYY-MM-DD" strings. Comparing them as strings is correct for
// ordering because the format is fixed-width and big-endian, which is the one
// property that makes ISO 8601 worth using.
// The calendar, in Bantu, with no query behind it.
//
// This used to ask SQLite ("SELECT date('now')") because the interpreter has
// epoch milliseconds and no calendar. That worked, but it put a database round
// trip inside the single function every read path calls -- and it meant the age
// wall could not be reasoned about without a database connection, which is a
// strange dependency for a pure arithmetic rule.
//
// Howard Hinnant's days-from-civil / civil-from-days algorithms do it in a
// dozen lines of integer arithmetic, exactly, for every proleptic Gregorian
// date. They are the same two functions C++20's <chrono> is built on.
//
// clock() is epoch milliseconds UTC, and SQLite's CURRENT_TIMESTAMP is UTC
// too, so the two agree about what day it is.
// Memoised per calendar day, because today() is on the hottest path in the
// application and it is not a cheap function.
//
// One feed request gates 180 candidate posts, and each one calls canSee (which
// recomputes the viewer's and the author's bands) and publicUser (which
// derives the author's age) -- so today() was being called several hundred
// times per request, each time running civil-from-days and rebuilding a
// string, for an answer that changes once every 86,400 seconds.
//
// The cache key is the epoch DAY, not a timestamp: it is correct across
// midnight by construction rather than by a timeout, and a stale band is the
// one failure this module exists to prevent.
$CLOCK = { "today": null, "day": null, "date": null, "civil": null };

def setToday($d) {
    $CLOCK["today"] = $d;
    $CLOCK["day"] = null;          // drop the memo; the override wins
}
def clearToday() {
    $CLOCK["today"] = null;
    $CLOCK["day"] = null;
}

// Days since 1970-01-01 for a civil date. Exact: no floating-point division
// survives into the result, because every quotient here is of a non-negative
// integer by a positive one.
def daysFromCivil($y, $m, $d) {
    $yy = $y;
    if ($m <= 2) { $yy = $yy - 1; }
    $era = floor($yy / 400);
    if ($yy < 0) { $era = floor(($yy - 399) / 400); }
    $yoe = $yy - $era * 400;                       // 0..399
    $mp = fmod($m + 9, 12);                        // Mar=0 .. Feb=11
    $doy = floor((153 * $mp + 2) / 5) + $d - 1;    // 0..365
    $doe = $yoe * 365 + floor($yoe / 4) - floor($yoe / 100) + $doy;
    return $era * 146097 + $doe - 719468;
}

def civilFromDays($z) {
    $zz = $z + 719468;
    $era = floor($zz / 146097);
    if ($zz < 0) { $era = floor(($zz - 146096) / 146097); }
    $doe = $zz - $era * 146097;                                        // 0..146096
    $yoe = floor(($doe - floor($doe / 1460) + floor($doe / 36524)
                  - floor($doe / 146096)) / 365);                      // 0..399
    $y = $yoe + $era * 400;
    $doy = $doe - (365 * $yoe + floor($yoe / 4) - floor($yoe / 100));  // 0..365
    $mp = floor((5 * $doy + 2) / 153);                                 // 0..11
    $d = $doy - floor((153 * $mp + 2) / 5) + 1;                        // 1..31
    $m = $mp + 3;
    if ($mp >= 10) { $m = $mp - 9; }
    if ($m <= 2) { $y = $y + 1; }
    return { "y": $y, "m": $m, "d": $d };
}

def _pad2($n) {
    if ($n < 10) { return "0" + str(floor($n)); }
    return str(floor($n));
}

def _pad4($n) {
    $s = str(floor($n));
    while (len($s) < 4) { $s = "0" + $s; }
    return $s;
}

def formatDate($c) {
    return _pad4($c["y"]) + "-" + _pad2($c["m"]) + "-" + _pad2($c["d"]);
}

// Epoch seconds. clock() is epoch milliseconds.
def nowSeconds() { return floor(clock() / 1000); }

def today() {
    if ($CLOCK["today"] != null) { return $CLOCK["today"]; }
    $day = floor(nowSeconds() / 86400);
    if ($CLOCK["day"] != $day) {
        $c = civilFromDays($day);
        $CLOCK["day"] = $day;
        $CLOCK["civil"] = $c;
        $CLOCK["date"] = formatDate($c);
    }
    return $CLOCK["date"];
}

// today(), already parsed. Saves formatting a string and parsing it straight
// back on every age computation.
def todayCivil() {
    if ($CLOCK["today"] != null) { return parseDob($CLOCK["today"]); }
    today();
    return $CLOCK["civil"];
}

// "YYYY-MM-DD HH:MM:SS" (what SQLite's CURRENT_TIMESTAMP writes, in UTC) to
// epoch seconds. Used for recency ranking and for account age, both of which
// previously cost a query each.
def epochOf($sqlDatetime) {
    if ($sqlDatetime == null || len($sqlDatetime) < 10) { return null; }
    $y = num(substr($sqlDatetime, 0, 4), null);
    $m = num(substr($sqlDatetime, 5, 2), null);
    $d = num(substr($sqlDatetime, 8, 2), null);
    if ($y == null || $m == null || $d == null) { return null; }
    $secs = daysFromCivil(floor($y), floor($m), floor($d)) * 86400;
    if (len($sqlDatetime) >= 19) {
        $hh = num(substr($sqlDatetime, 11, 2), null);
        $mm = num(substr($sqlDatetime, 14, 2), null);
        $ss = num(substr($sqlDatetime, 17, 2), null);
        if ($hh != null && $mm != null && $ss != null) {
            $secs = $secs + floor($hh) * 3600 + floor($mm) * 60 + floor($ss);
        }
    }
    return $secs;
}

def hoursSince($sqlDatetime) {
    $e = epochOf($sqlDatetime);
    if ($e == null) { return 0; }
    $h = (nowSeconds() - $e) / 3600.0;
    if ($h < 0) { return 0; }
    return $h;
}

def secondsSince($sqlDatetime) {
    $e = epochOf($sqlDatetime);
    if ($e == null) { return 999999999; }
    $s = nowSeconds() - $e;
    if ($s < 0) { return 0; }
    return $s;
}

def parseDob($s) {
    if ($s == null || len($s) < 10) { return null; }
    $y = num(substr($s, 0, 4), null);
    $m = num(substr($s, 5, 2), null);
    $d = num(substr($s, 8, 2), null);
    if ($y == null || $m == null || $d == null) { return null; }
    if ($m < 1 || $m > 12 || $d < 1 || $d > 31) { return null; }
    if ($y < 1900 || $y > 2100) { return null; }
    return { "y": floor($y), "m": floor($m), "d": floor($d) };
}

// Completed years as of $onDate. "Has the birthday happened yet this year?" is
// the whole calculation, and getting it wrong by a day puts a 17-year-old in
// the adult band for up to twelve months.
def ageOn($dob, $onDate) {
    $b = parseDob($dob);
    $n = parseDob($onDate);
    if ($b == null || $n == null) { return -1; }
    $age = $n["y"] - $b["y"];
    if ($n["m"] < $b["m"]) { $age = $age - 1; }
    else if ($n["m"] == $b["m"] && $n["d"] < $b["d"]) { $age = $age - 1; }
    if ($age < 0) { return -1; }
    return $age;
}

// Against the memoised civil date rather than a freshly formatted string.
def ageNow($dob) {
    $b = parseDob($dob);
    if ($b == null) { return -1; }
    $n = todayCivil();
    if ($n == null) { return -1; }
    $age = $n["y"] - $b["y"];
    if ($n["m"] < $b["m"]) { $age = $age - 1; }
    else if ($n["m"] == $b["m"] && $n["d"] < $b["d"]) { $age = $age - 1; }
    if ($age < 0) { return -1; }
    return $age;
}

// ── bands ───────────────────────────────────────────────────────────────────
def bandForAge($age) {
    if ($age >= cfg.CFG["adultAge"]) { return $ADULT; }
    return $MINOR;
}

def bandFor($dob) { return bandForAge(ageNow($dob)); }

def isSignupAgeOk($dob) {
    $age = ageNow($dob);
    if ($age < cfg.CFG["minSignupAge"]) { return false; }
    if ($age > 120) { return false; }
    return true;
}

// ── dating ranges ───────────────────────────────────────────────────────────
// A minor's range is not a preference. It is +/- 2 years, it is not settable,
// and it is clamped to the band on both ends.
def defaultRange($age) {
    if (bandForAge($age) == $MINOR) {
        $lo = $age - cfg.CFG["minorSpread"];
        $hi = $age + cfg.CFG["minorSpread"];
        if ($lo < cfg.CFG["minSignupAge"]) { $lo = cfg.CFG["minSignupAge"]; }
        if ($hi > cfg.CFG["adultAge"] - 1) { $hi = cfg.CFG["adultAge"] - 1; }
        return { "min": $lo, "max": $hi };
    }
    $spread = cfg.CFG["adultDefaultSpread"];
    $lo = $age - $spread;
    if ($lo < cfg.CFG["adultAge"]) { $lo = cfg.CFG["adultAge"]; }
    return { "min": $lo, "max": $age + $spread };
}

// An adult may widen or narrow their range; the floor of 18 is not negotiable
// and is applied here rather than trusted from the request.
def clampRange($age, $min, $max) {
    if (bandForAge($age) == $MINOR) { return defaultRange($age); }
    $lo = floor($min);
    $hi = floor($max);
    if ($lo < cfg.CFG["adultAge"]) { $lo = cfg.CFG["adultAge"]; }
    if ($hi > 120) { $hi = 120; }
    if ($hi < $lo) { $hi = $lo; }
    return { "min": $lo, "max": $hi };
}

// ── the gate ────────────────────────────────────────────────────────────────
//
// TWO questions, deliberately separate, because conflating them was a real
// error in the first draft of this design:
//
//   canSee    may the viewer see this account's CONTENT?
//             Band, and for a minor also within +/- 2 years. An adult sees
//             every adult. This is the SAFETY rule.
//
//   canMatch  should this account be OFFERED to the viewer as a match?
//             Band, plus the viewer's dating range, plus the other party's
//             range -- both directions, because a one-sided match is not one.
//             This is the PREFERENCE rule.
//
// Gating the feed on the dating range looks safer and is not: a 19-year-old
// whose range is 25-29 would stop seeing their own posts, and their friends'.
// A preference about who to date is not a statement about whose life stories
// you may read, and overloading it produces a feed nobody can explain.
//
// What the two share is the wall. Neither crosses 18, ever.
def _age($u) {
    if ($u == null) { return -1; }
    if ($u["age"] != null) { return floor($u["age"]); }
    return ageNow($u["dob"]);
}

def _band($u) {
    if ($u == null) { return null; }
    // Recomputed from the date of birth, never read from the stored column.
    // The column is a cache; the date of birth is the fact. A stale band is
    // precisely the failure this whole module exists to prevent.
    return bandFor($u["dob"]);
}

def canSee($viewer, $target) {
    if ($viewer == null || $target == null) { return false; }
    $vb = _band($viewer);
    $tb = _band($target);
    if ($vb == null || $tb == null) { return false; }
    if ($vb != $tb) { return false; }                  // the wall

    $va = _age($viewer);
    $ta = _age($target);
    if ($va < 0 || $ta < 0) { return false; }

    if ($vb == $MINOR) {
        $d = $va - $ta;
        if ($d < 0) { $d = 0 - $d; }
        return $d <= cfg.CFG["minorSpread"];
    }
    return true;                                       // adult sees adult
}

def _inRange($viewer, $target) {
    $va = _age($viewer);
    $ta = _age($target);
    if (_band($viewer) == $MINOR) { return true; }     // the +/-2 rule is canSee's
    $lo = $viewer["seek_min_age"];
    $hi = $viewer["seek_max_age"];
    if ($lo == null || $hi == null) {
        $r = defaultRange($va);
        $lo = $r["min"];
        $hi = $r["max"];
    }
    $r = clampRange($va, $lo, $hi);
    return $ta >= $r["min"] && $ta <= $r["max"];
}

// Mutual by construction: a match the other person's own range would not have
// produced is a match they did not ask for.
def canMatch($a, $b) {
    if (!canSee($a, $b) || !canSee($b, $a)) { return false; }
    return _inRange($a, $b) && _inRange($b, $a);
}

// Contact -- follow, like, report. Content visibility in both directions.
// Symmetric for minors by the +/-2 rule and trivially true among adults.
def canInteract($a, $b) {
    return canSee($a, $b) && canSee($b, $a);
}

// ── ORM filter pairs ────────────────────────────────────────────────────────
// The single source of the band condition in SQL. A route composes these into
// its queryset; no route writes the condition by hand.
def _contentBounds($viewer) {
    $va = _age($viewer);
    $vb = _band($viewer);
    if ($vb == $MINOR) {
        $lo = $va - cfg.CFG["minorSpread"];
        $hi = $va + cfg.CFG["minorSpread"];
        if ($lo < cfg.CFG["minSignupAge"]) { $lo = cfg.CFG["minSignupAge"]; }
        if ($hi > cfg.CFG["adultAge"] - 1) { $hi = cfg.CFG["adultAge"] - 1; }
        return { "band": $MINOR, "min": $lo, "max": $hi };
    }
    return { "band": $ADULT, "min": cfg.CFG["adultAge"], "max": 120 };
}

def _matchBounds($viewer) {
    $vb = _band($viewer);
    if ($vb == $MINOR) { return _contentBounds($viewer); }
    $va = _age($viewer);
    $lo = $viewer["seek_min_age"];
    $hi = $viewer["seek_max_age"];
    if ($lo == null || $hi == null) {
        $r = defaultRange($va);
        $lo = $r["min"];
        $hi = $r["max"];
    }
    $r = clampRange($va, $lo, $hi);
    return { "band": $ADULT, "min": $r["min"], "max": $r["max"] };
}

// For `posts`: the author's band and their age AT THE TIME OF POSTING.
def visibilityPairs($viewer) {
    $b = _contentBounds($viewer);
    return [["status", "live"],
            ["author_band", $b["band"]],
            ["author_age_at_post__gte", $b["min"]],
            ["author_age_at_post__lte", $b["max"]]];
}

// For `users`, content visibility. The band is a column, but age is derived
// from dob, so an age bound becomes a date-of-birth bound -- "older than X"
// means "born before a date". Computed here so no caller has to get the
// inequality the right way round.
def userVisibilityPairs($viewer) {
    return _userPairs(_contentBounds($viewer));
}

// For `users`, match candidates: the same, narrowed to the dating range.
def matchPairs($viewer) {
    return _userPairs(_matchBounds($viewer));
}

def _userPairs($b) {
    $t = parseDob(today());
    // age >= min  <=>  dob <= today - min years
    $maxDob = _shiftYears($t, 0 - $b["min"]);
    // age <= max  <=>  dob >= today - (max+1) years. Whole-year granularity,
    // so this NARROWS and canSee/canMatch decides -- the exact check is never
    // the SQL's job.
    $minDob = _shiftYears($t, 0 - ($b["max"] + 1));
    return [["band", $b["band"]],
            ["status", "active"],
            ["dob__lte", $maxDob],
            ["dob__gte", $minDob]];
}

def _shiftYears($d, $years) {
    return _pad4($d["y"] + $years) + "-" + _pad2($d["m"]) + "-" + _pad2($d["d"]);
}

def contentBounds($viewer) { return _contentBounds($viewer); }
def matchBounds($viewer) { return _matchBounds($viewer); }
def MINOR_BAND() { return $MINOR; }
def ADULT_BAND() { return $ADULT; }
