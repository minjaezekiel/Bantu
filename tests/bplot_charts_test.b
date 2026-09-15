// ════════════════════════════════════════════════════════════════════════
//  bplot_charts_test.b — the B2 chart types, scales, ticks and date axes.
//
//  WHAT THIS IS GATING, and why each part earns its place:
//
//    1. THE SCALES. A log axis handed a zero is the classic silent-garbage
//       case: log10(0) is -inf, which reaches the document as a NaN
//       coordinate, which every browser renders as NOTHING AT ALL with no
//       error in any log. Every path into a log scale is checked to raise
//       and to name the offending value.
//
//    2. THE STATISTICS. hist, boxplot and violin are not drawing routines --
//       they are a binning rule, five order statistics and a density
//       estimate. Each has a definition that already exists in this tree
//       (numba's nd_histogram and nd_quantile) or in the tool everyone
//       compares against, so each is asserted against REFERENCE VALUES
//       generated from numpy and matplotlib and pasted in as literals. A
//       test that only checks bplot against itself proves nothing.
//
//    3. THE TICK LOCATORS. Log, symlog and date ticks are compared against
//       matplotlib's own output across the ranges where its answer is right,
//       AND against bplot's deliberately different answer on the two ranges
//       where matplotlib produces an axis with no labelled tick at all. The
//       divergence is asserted so it stays a decision rather than decaying
//       into a bug.
//
//  Numbers are never asserted through str(): six significant digits would
//  hide a real error. Comparisons go through approx() on the value itself.
//
//  Reference values in this file were generated once, at authoring time,
//  from numpy 2.5.2 and matplotlib 3.11.2 and written in as literals.
//  NOTHING HERE DEPENDS ON PYTHON AT RUN TIME.
//
//  Run:  bantu run tests/bplot_charts_test.b
// ════════════════════════════════════════════════════════════════════════

include "./bplot/bplot.b" as plt;

$R = {"pass": 0, "fail": 0};

def ok($cond, $name) {
    if ($cond) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $name); }
}

def eq($got, $want, $name) {
    if ($got == $want) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name);
        print("          got:  " + str($got));
        print("          want: " + str($want));
    }
}

// Relative tolerance, so the same helper works at 1e-9 and at 1e9.
def approx($got, $want, $name) {
    $tol = 0.000000001 * max(abs($want), 1);
    if (isnan($got) && isnan($want)) { $R.pass = $R.pass + 1; print("  ok    " + $name); return null; }
    if (abs($got - $want) <= $tol) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name);
        print("          got:  " + str($got));
        print("          want: " + str($want));
    }
}

def raises($f, $name) {
    $caught = false;
    try { $f(); } catch ($e) { $caught = true; }
    ok($caught, $name);
}

// Raises AND the message names the given substring — "errors teach" is
// testable, so it is tested.
def raisesWith($f, $needle, $name) {
    $caught = false;
    $msg = "";
    try { $f(); } catch ($e) {
        $caught = true;
        if (type($e) == "string") { $msg = $e; } else { $msg = str($e["message"]); }
    }
    if ($caught && contains($msg, $needle)) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name);
        if (!$caught) { print("          did not raise"); }
        else { print("          message lacked \"" + $needle + "\": " + $msg); }
    }
}

def has($hay, $needle, $name) { ok(contains($hay, $needle), $name); }
def hasnt($hay, $needle, $name) { ok(!contains($hay, $needle), $name); }

// Ticks joined through the axis's OWN label formatter, so the assertion is
// about what a reader sees rather than about an internal representation.
def joinDecades($vals) {
    $o = [];
    $i = 0;
    while ($i < len($vals)) { push($o, plt._fmtDecade($vals[$i])); $i = $i + 1; }
    return join($o, ",");
}
def logTicks($lo, $hi) { return joinDecades(plt._ticksLog($lo, $hi)["major"]); }
def dateTickStr($lo, $hi) {
    $d = plt._dateTicks($lo, $hi);
    $o = [];
    $i = 0;
    while ($i < len($d["ticks"])) { push($o, plt._fmtDate($d["ticks"][$i], "full")); $i = $i + 1; }
    return join($o, "|");
}

def newAxes() { $f = plt.figure(640, 480); return $f.addAxes(); }

print("========================================");
print("  bplot — charts, scales, ticks (B2)");
print("========================================");

// ════════════════════════════════════════════════════════════════════════
//  1. Scales
// ════════════════════════════════════════════════════════════════════════
print("");
print("-- log: forward, inverse, and the raise that stops NaN reaching SVG --");
approx(plt._fwd(1, "log", 1), 0, "log10(1) = 0");
approx(plt._fwd(100, "log", 1), 2, "log10(100) = 2");
approx(plt._fwd(0.001, "log", 1), -3, "log10(0.001) = -3");
approx(plt._inv(plt._fwd(37.5, "log", 1), "log", 1), 37.5, "log round-trips");
raisesWith(def() { plt._fwd(0, "log", 1); }, "0", "log of zero raises, naming the value");
raisesWith(def() { plt._fwd(-5, "log", 1); }, "-5", "log of a negative raises, naming the value");
raisesWith(def() { plt._fwd(0, "log", 1); }, "symlog", "the log error suggests symlog");

print("");
print("-- symlog: matplotlib's SymmetricalLogTransform, to nine decimals --");
// Reference: matplotlib.scale.SymmetricalLogTransform(10, linthresh, 1.0)
approx(plt._fwd(0, "symlog", 1), 0, "symlog lt=1: 0");
approx(plt._fwd(0.5, "symlog", 1), 0.555555556, "symlog lt=1: 0.5");
approx(plt._fwd(1, "symlog", 1), 1.111111111, "symlog lt=1: 1 (the join)");
approx(plt._fwd(2, "symlog", 1), 1.412141107, "symlog lt=1: 2");
approx(plt._fwd(10, "symlog", 1), 2.111111111, "symlog lt=1: 10");
approx(plt._fwd(100, "symlog", 1), 3.111111111, "symlog lt=1: 100");
approx(plt._fwd(1000, "symlog", 1), 4.111111111, "symlog lt=1: 1000");
approx(plt._fwd(-1, "symlog", 1), -1.111111111, "symlog lt=1: -1");
approx(plt._fwd(-10, "symlog", 1), -2.111111111, "symlog lt=1: -10");
approx(plt._fwd(-100, "symlog", 1), -3.111111111, "symlog lt=1: -100");
approx(plt._fwd(0.5, "symlog", 2), 0.555555556, "symlog lt=2: 0.5");
approx(plt._fwd(2, "symlog", 2), 2.222222222, "symlog lt=2: 2 (the join)");
approx(plt._fwd(10, "symlog", 2), 3.620162231, "symlog lt=2: 10");
approx(plt._fwd(100, "symlog", 2), 5.620162231, "symlog lt=2: 100");
approx(plt._fwd(0.5, "symlog", 0.1), 0.181008112, "symlog lt=0.1: 0.5");
approx(plt._fwd(1, "symlog", 0.1), 0.211111111, "symlog lt=0.1: 1");
approx(plt._fwd(-100, "symlog", 0.1), -0.411111111, "symlog lt=0.1: -100");
approx(plt._inv(plt._fwd(-73.2, "symlog", 1), "symlog", 1), -73.2, "symlog round-trips, negative");
approx(plt._inv(plt._fwd(0.25, "symlog", 1), "symlog", 1), 0.25, "symlog round-trips, inside the band");
ok(plt._fwd(0, "symlog", 1) == 0, "symlog maps zero to zero — the whole point of it");

print("");
print("-- symlog is continuous at the join, which is what makes it one axis --");
approx(plt._fwd(1.0000000001, "symlog", 1), plt._fwd(1, "symlog", 1), "no step at +linthresh");
approx(plt._fwd(-1.0000000001, "symlog", 1), plt._fwd(-1, "symlog", 1), "no step at -linthresh");

print("");
print("-- linthresh must be positive and finite, or it divides by zero --");
raisesWith(def() { $a = newAxes(); $a.setScale("y", "symlog", 0); }, "0", "linthresh 0 raises");
raisesWith(def() { $a = newAxes(); $a.setScale("y", "symlog", -1); }, "-1", "negative linthresh raises");
raises(def() { $a = newAxes(); $a.setScale("y", "symlog", INF); }, "infinite linthresh raises");
raisesWith(def() { $a = newAxes(); $a.setScale("y", "loog", null); }, "loog", "an unknown scale name raises, quoting it");

print("");
print("-- linear is untouched: _fwd is the identity, so B1 pays nothing --");
approx(plt._fwd(123.456, "linear", 1), 123.456, "linear forward is identity");
approx(plt._inv(123.456, "linear", 1), 123.456, "linear inverse is identity");

// ════════════════════════════════════════════════════════════════════════
//  2. Log and symlog tick locators
// ════════════════════════════════════════════════════════════════════════
print("");
print("-- log ticks agree with matplotlib where matplotlib is right --");
// Reference: LogLocator through a drawn axis, filtered to the view interval.
eq(logTicks(1, 1000), "1,10,100,1000", "1..1000");
eq(logTicks(1, 100), "1,10,100", "1..100");
eq(logTicks(0.001, 1), "0.001,0.01,0.1,1", "0.001..1");
eq(logTicks(1, 10), "1,10", "1..10");
eq(logTicks(1, 1000000), "1,10,100,1000,10000,100000,1e6", "1..1e6");
eq(logTicks(0.5, 50), "1,10", "0.5..50");
eq(logTicks(0.000000001, 0.000001), "1e-9,1e-8,1e-7,1e-6", "1e-9..1e-6");
eq(logTicks(5, 5000), "10,100,1000", "5..5000");

print("");
print("-- and DELIBERATELY differ where matplotlib leaves the axis blank --");
// matplotlib returns NO major ticks for 2..9 and exactly one for 1..3.
// An axis with no labelled tick is not a style difference (decision BP17).
eq(logTicks(2, 9), "2,3,4,5,6,7,8,9", "2..9 promotes the minors (matplotlib: none)");
eq(logTicks(1, 3), "1,2,3", "1..3 promotes the minors (matplotlib: just 1)");
$t = plt._ticksLog(2, 9);
eq(len($t["minor"]), 0, "promoting clears the minor set, so nothing is drawn twice");
$t = plt._ticksLog(1, 1000);
eq(len($t["minor"]), 24, "1..1000 has 24 minor ticks (2..9 of three decades)");
eq($t["minor"][0], 2, "the first minor is 2");

print("");
print("-- symlog ticks: zero, then decades from linthresh outward --");
def symTicks($lo, $hi, $lt) { return joinDecades(plt._ticksSymlog($lo, $hi, $lt)["major"]); }
eq(symTicks(-100, 100, 1), "-100,-10,-1,0,1,10,100", "-100..100, lt 1");
eq(symTicks(-1000, 1000, 1), "-1000,-100,-10,-1,0,1,10,100,1000", "-1000..1000, lt 1");
eq(symTicks(-10, 10, 1), "-10,-1,0,1,10", "-10..10, lt 1");
eq(symTicks(0, 100, 1), "0,1,10,100", "0..100, lt 1");
eq(symTicks(-100, 100, 10), "-100,-10,0,10,100", "lt 10 drops the +/-1 ticks");
eq(symTicks(-5, 5, 1), "-1,0,1", "-5..5 has no decade at 5");

print("");
print("-- decade labels: plain while one exists, then 1e6 --");
eq(plt._fmtDecade(1), "1", "1");
eq(plt._fmtDecade(100), "100", "100");
eq(plt._fmtDecade(100000), "100000", "1e5 still reads plainly");
eq(plt._fmtDecade(1000000), "1e6", "1e6 switches");
eq(plt._fmtDecade(0.001), "0.001", "0.001");
eq(plt._fmtDecade(0.0001), "0.0001", "1e-4 is the last plain one");
eq(plt._fmtDecade(0.00001), "1e-5", "1e-5 switches");
eq(plt._fmtDecade(0), "0", "zero, for symlog's middle tick");
eq(plt._fmtDecade(-10), "-10", "negatives, for symlog's left half");

// ════════════════════════════════════════════════════════════════════════
//  3. Dates
// ════════════════════════════════════════════════════════════════════════
print("");
print("-- the calendar: Hinnant's civil_from_days, exact and table-free --");
eq(join(plt._civilFromDays(0), "-"), "1970-1-1", "day 0 is the epoch");
eq(join(plt._civilFromDays(19723), "-"), "2024-1-1", "day 19723 is 2024-01-01");
eq(join(plt._civilFromDays(-1), "-"), "1969-12-31", "day -1 is before the epoch");
eq(join(plt._civilFromDays(11016), "-"), "2000-2-29", "2000 was a leap year (divisible by 400)");
eq(plt._daysFromCivil(1970, 1, 1), 0, "inverse: epoch");
eq(plt._daysFromCivil(2024, 1, 1), 19723, "inverse: 2024");
eq(plt._daysFromCivil(1900, 3, 1), plt._daysFromCivil(1900, 2, 28) + 1,
   "1900 was NOT a leap year (divisible by 100, not 400)");
eq(plt._daysFromCivil(2000, 3, 1), plt._daysFromCivil(2000, 2, 28) + 2, "2000 WAS a leap year");

// Round-trip every day across a four-year window spanning a leap day.
$bad = 0;
$d = 19000;
while ($d < 20461) {
    $c = plt._civilFromDays($d);
    if (plt._daysFromCivil($c[0], $c[1], $c[2]) != $d) { $bad = $bad + 1; }
    $d = $d + 1;
}
eq($bad, 0, "1,461 consecutive days round-trip exactly (four years, one leap day)");

print("");
print("-- date ticks match matplotlib's AutoDateLocator exactly --");
$E = 1704067200000;    // 2024-01-01T00:00:00Z
eq(dateTickStr($E, $E + 10000),
   "2024-01-01 00:00:00|2024-01-01 00:00:01|2024-01-01 00:00:02|2024-01-01 00:00:03|" +
   "2024-01-01 00:00:04|2024-01-01 00:00:05|2024-01-01 00:00:06|2024-01-01 00:00:07|" +
   "2024-01-01 00:00:08|2024-01-01 00:00:09|2024-01-01 00:00:10",
   "ten seconds -> 1-second ticks");
eq(dateTickStr($E, $E + 300000),
   "2024-01-01 00:00:00|2024-01-01 00:01:00|2024-01-01 00:02:00|2024-01-01 00:03:00|" +
   "2024-01-01 00:04:00|2024-01-01 00:05:00",
   "five minutes -> 1-minute ticks");
eq(dateTickStr($E, $E + 21600000),
   "2024-01-01 00:00:00|2024-01-01 01:00:00|2024-01-01 02:00:00|2024-01-01 03:00:00|" +
   "2024-01-01 04:00:00|2024-01-01 05:00:00|2024-01-01 06:00:00",
   "six hours -> 1-hour ticks");
eq(dateTickStr($E, $E + 604800000),
   "2024-01-01 00:00:00|2024-01-02 00:00:00|2024-01-03 00:00:00|2024-01-04 00:00:00|" +
   "2024-01-05 00:00:00|2024-01-06 00:00:00|2024-01-07 00:00:00|2024-01-08 00:00:00",
   "one week -> daily ticks");
// The case a look-alike implementation gets wrong: matplotlib switches to
// SEMI-MONTHLY here, not to months and not to 14-day intervals.
eq(dateTickStr($E, 1711929600000),
   "2024-01-01 00:00:00|2024-01-15 00:00:00|2024-02-01 00:00:00|2024-02-15 00:00:00|" +
   "2024-03-01 00:00:00|2024-03-15 00:00:00|2024-04-01 00:00:00",
   "three months -> the 1st and the 15th");
eq(dateTickStr($E, 1735689600000),
   "2024-01-01 00:00:00|2024-03-01 00:00:00|2024-05-01 00:00:00|2024-07-01 00:00:00|" +
   "2024-09-01 00:00:00|2024-11-01 00:00:00|2025-01-01 00:00:00",
   "one year -> every second month");
eq(dateTickStr(1577836800000, 1735689600000),
   "2020-01-01 00:00:00|2021-01-01 00:00:00|2022-01-01 00:00:00|2023-01-01 00:00:00|" +
   "2024-01-01 00:00:00|2025-01-01 00:00:00",
   "five years -> yearly ticks");
// The other one: a FOUR-year interval on multiples of four, not five.
eq(dateTickStr(946684800000, 1735689600000),
   "2000-01-01 00:00:00|2004-01-01 00:00:00|2008-01-01 00:00:00|2012-01-01 00:00:00|" +
   "2016-01-01 00:00:00|2020-01-01 00:00:00|2024-01-01 00:00:00",
   "25 years -> every fourth year, anchored on multiples of four");

print("");
print("-- date labels carry only what the step needs --");
eq(plt._fmtDate($E, "second"), "00:00:00", "a seconds axis drops the date");
// These two read `"00:00", "a minutes axis drops the seconds"` until B4. A bare
// time is ambiguous on any axis that crosses midnight, so minute and hour labels
// now carry the day exactly as matplotlib's defaults do: '%d %H:%M' and
// '%m-%d %H' (date.autoformatter.minute / .hour, checked against 3.11.2).
eq(plt._fmtDate($E, "minute"), "01 00:00", "a minutes axis carries the day, as matplotlib's does");
eq(plt._fmtDate($E + 13 * 3600000, "hour"), "01-01 13", "an hours axis carries month and day, as matplotlib's does");
eq(plt._fmtDate($E, "day"), "2024-01-01", "a daily axis drops the time");
eq(plt._fmtDate($E, "month"), "2024-01", "a monthly axis drops the day");
eq(plt._fmtDate($E, "year"), "2024", "a yearly axis is just the year");
eq(plt._fmtDate($E + 3723000, "full"), "2024-01-01 01:02:03", "full form, zero-padded");

// ════════════════════════════════════════════════════════════════════════
//  4. Statistics
// ════════════════════════════════════════════════════════════════════════
print("");
print("-- quantiles are numpy's linear method, which is nd_quantile's --");
$d1 = sort([1, 2, 3, 4, 5, 6, 7, 8, 9, 100]);
approx(plt._quantile($d1, 0.25), 3.25, "q1 of the ten-value set");
approx(plt._quantile($d1, 0.5), 5.5, "median");
approx(plt._quantile($d1, 0.75), 7.75, "q3");
$d2 = sort([2, 3, 4, 5, 6]);
approx(plt._quantile($d2, 0.25), 3, "q1, odd count");
approx(plt._quantile($d2, 0.5), 4, "median, odd count");
approx(plt._quantile($d2, 0.75), 5, "q3, odd count");
approx(plt._quantile([7], 0.5), 7, "a single value is its own median");
approx(plt._quantile($d1, 0), 1, "q=0 is the minimum");
approx(plt._quantile($d1, 1), 100, "q=1 is the maximum");

print("");
print("-- boxplot statistics match matplotlib's cbook.boxplot_stats --");
$s = plt._boxStats($d1);
approx($s["q1"], 3.25, "q1");
approx($s["med"], 5.5, "median");
approx($s["q3"], 7.75, "q3");
approx($s["wlo"], 1, "lower whisker lands on an OBSERVED value");
approx($s["whi"], 9, "upper whisker stops at 9, not at the 1.5*IQR fence");
eq(len($s["out"]), 1, "one outlier");
approx($s["out"][0], 100, "and it is the 100");
$s2 = plt._boxStats($d2);
eq(len($s2["out"]), 0, "a tight set has no outliers");
approx($s2["wlo"], 2, "whiskers reach the extremes when nothing is an outlier");
approx($s2["whi"], 6, "upper likewise");
$s3 = plt._boxStats([7]);
approx($s3["med"], 7, "a boxplot of ONE point has a median");
approx($s3["wlo"], 7, "and zero-length whiskers");
eq(len($s3["out"]), 0, "and no outliers");

print("");
print("-- histogram binning matches nd_histogram / numpy exactly --");
eq(join(plt._histCounts([1,2,3,4,5,6,7,8,9,10], 5, 1, 10), ","), "2,2,2,2,2", "ten values, five bins");
eq(join(plt._histCounts([0,0.5,1], 2, 0, 1), ","), "1,2", "the TOP EDGE IS INCLUSIVE");
eq(join(plt._histCounts([1,2,3,4,5], 3, 2, 4), ","), "1,1,1", "out-of-range values are dropped, not clamped");
eq(join(plt._histCounts([1,NAN,2,NAN], 2, 1, 2), ","), "1,1", "NaN is dropped");
eq(join(plt._histCounts([1,1,1], 1, 0.5, 1.5), ","), "3", "a single bin holds everything");

print("");
print("-- the inclusive top edge, on its own, because it is the one that bites --");
// Without it the largest value silently vanishes and the chart lies about
// its range.
$c = plt._histCounts([0, 1, 2, 3, 4, 5], 5, 0, 5);
eq(join($c, ","), "1,1,1,1,2", "the maximum lands in the last bin, not nowhere");
$tot = 0;
$i = 0;
while ($i < len($c)) { $tot = $tot + $c[$i]; $i = $i + 1; }
eq($tot, 6, "every value is counted exactly once");

// ════════════════════════════════════════════════════════════════════════
//  5. The chart types, as drawn documents
// ════════════════════════════════════════════════════════════════════════
print("");
print("-- hist --");
$ax = newAxes();
$ax.hist([1,2,2,3,3,3,4,4,5], {"bins": 5, "label": "counts"});
$a = $ax.artists[0];
eq(len($a["counts"]), 5, "five bins");
eq(join($a["counts"], ","), "1,2,3,2,1", "counts");
approx($a["edges"][0], 1, "first edge is the minimum");
approx($a["edges"][5], 5, "last edge is the maximum");
approx($a["by"][1], 3, "the value axis reaches the tallest bar");
approx($a["by"][0], 0, "and starts at zero");

$ax = newAxes();
$ax.hist([1,2,2,3,3,3,4,4,5], {"bins": 5, "density": true});
$a = $ax.artists[0];
// Reference: numpy histogram(..., density=True) over the same data.
approx($a["heights"][0], 0.138888889, "density bin 0");
approx($a["heights"][1], 0.277777778, "density bin 1");
approx($a["heights"][2], 0.416666667, "density bin 2");
$area = 0;
$w = $a["edges"][1] - $a["edges"][0];
$i = 0;
while ($i < 5) { $area = $area + $a["heights"][$i] * $w; $i = $i + 1; }
approx($area, 1, "density normalises AREA to 1, not height");

$fg = plt.figure(400, 300); $fg.addAxes().hist([1,2,3,4,5], {"bins": 3});
$svg = $fg.to_svg();
has($svg, "<rect", "hist emits rectangles");

print("");
print("-- boxplot --");
$ax = newAxes();
$ax.boxplot([[1,2,3,4,5,100],[2,3,4,5,6]], {"labels": ["alpha", "beta"]});
$a = $ax.artists[0];
eq(len($a["stats"]), 2, "two groups");
eq($ax.xtickOverride["labels"][0], "alpha", "the group label becomes a tick label");
eq($ax.xtickOverride["labels"][1], "beta", "and the second");
approx($ax.xtickOverride["pos"][0], 1, "groups sit at 1..n");
approx($a["bx"][0], 0.5, "with half a slot of margin either side");
approx($a["bx"][1], 2.5, "likewise on the right");
// One sequence, not a list of sequences — told apart by the first element.
$ax = newAxes();
$ax.boxplot([1,2,3,4,5], null);
eq(len($ax.artists[0]["stats"]), 1, "a bare sequence is ONE group, not five");
raisesWith(def() { $a = newAxes(); $a.boxplot([[1,2],[3,4]], {"labels": ["only one"]}); },
           "2 groups but 1 labels", "a label count mismatch raises, naming both");
raises(def() { $a = newAxes(); $a.boxplot([], null); }, "an empty boxplot raises");
raisesWith(def() { $a = newAxes(); $a.boxplot([NAN, NAN], null); },
           "no finite values", "a boxplot of nothing but NaN raises");

print("");
print("-- violin --");
$ax = newAxes();
$ax.violin([[1,2,3,4,5,4,3,2,1,2,3,4]], null);
$a = $ax.artists[0];
eq(len($a["shapes"]), 1, "one violin");
ok(len($a["shapes"][0]["x"]) > 8, "the density is evaluated on a grid");
ok($a["shapes"][0]["y"][0] >= 0, "a density is never negative");
// A single repeated value has no spread and therefore no bandwidth.
$ax = newAxes();
$ax.violin([[5,5,5,5]], null);
approx($ax.artists[0]["shapes"][0]["flat"], 5, "no spread degenerates to a line, not a divide by zero");
$fg = plt.figure(400, 300); $fg.addAxes().violin([[1,2,3,4,5,4,3,2]], null);
$svg = $fg.to_svg();
has($svg, "<polygon", "violin emits a polygon");

print("");
print("-- errorbar --");
$ax = newAxes();
$ax.errorbar([1,2,3], [10,20,30], {"yerr": 5});
$a = $ax.artists[0];
approx($a["yerr"][0][0], 5, "a scalar error is symmetric and constant");
approx($a["by"][0], 5, "the value axis widens to the bottom of the bars");
approx($a["by"][1], 35, "and to the top");
$ax = newAxes();
$ax.errorbar([1,2], [10,20], {"yerr": [[1,2],[3,4]]});
$a = $ax.artists[0];
approx($a["yerr"][0][1], 2, "two sequences give an asymmetric error, lower");
approx($a["yerr"][1][1], 4, "and upper");
approx($a["by"][0], 9, "lower bound uses the lower error");
approx($a["by"][1], 24, "upper bound uses the upper error");
// Drawn, a negative error produces an INVERTED bar that reads as a smaller
// error than the real one, so it is always a bug.
raisesWith(def() { $a = newAxes(); $a.errorbar([1,2], [1,2], {"yerr": -1}); },
           "-1", "a negative scalar error raises, naming it");
raisesWith(def() { $a = newAxes(); $a.errorbar([1,2], [1,2], {"yerr": [1,-2]}); },
           "position 1", "a negative error in a sequence raises, naming the position");
raisesWith(def() { $a = newAxes(); $a.errorbar([1,2], [1,2], {"yerr": [1,2,3]}); },
           "3 entries for 2 points", "a length mismatch raises, naming both");

print("");
print("-- fill_between --");
$ax = newAxes();
$ax.fill_between([1,2,3], [2,3,4], [1,1,1], null);
$a = $ax.artists[0];
approx($a["by"][0], 1, "the band's lower edge counts toward the limits");
approx($a["by"][1], 4, "and the upper");
$ax = newAxes();
$ax.fill_between([1,2,3], [2,3,4], null, null);
approx($ax.artists[0]["y2"][0], 0, "y2 defaults to zero");
$ax = newAxes();
$ax.fill_between([1,2,3], [2,3,4], 1.5, null);
approx($ax.artists[0]["y2"][2], 1.5, "a number for y2 is broadcast");
$fg = plt.figure(400, 300); $fg.addAxes().fill_between([1,2,3], [2,3,4], null, null);
$svg = $fg.to_svg();
has($svg, "<polygon", "fill_between emits a polygon");
has($svg, "fill-opacity", "and it is translucent, so a line behind it stays visible");
raisesWith(def() { $a = newAxes(); $a.fill_between([1,2,3], [1,2], null, null); },
           "3 points and y1 has 2", "a length mismatch raises, naming both");

print("");
print("-- step --");
$ax = newAxes();
$ax.step([1,2,3], [1,3,2], {"where": "post"});
eq($ax.artists[0]["where"], "post", "where is kept");
raisesWith(def() { $a = newAxes(); $a.step([1,2], [1,2], {"where": "middle"}); },
           "middle", "an unknown `where` raises, quoting it");
$fig = plt.figure(400, 300);
$fig.addAxes().step([0,1,2], [0,1,0], {"where": "pre"});
has($fig.to_svg(), "<polyline", "step emits a polyline");

print("");
print("-- stem --");
$ax = newAxes();
$ax.stem([1,2,3], [1,-2,3], null);
$a = $ax.artists[0];
approx($a["by"][0], -2, "negative values reach the limits");
approx($a["bottom"], 0, "the baseline defaults to zero");
$ax = newAxes();
$ax.stem([1,2], [5,6], {"bottom": 4});
approx($ax.artists[0]["by"][0], 4, "a baseline above the data still bounds the axis");

print("");
print("-- pie --");
$ax = newAxes();
$ax.pie([30,20,50], {"labels": ["a","b","c"], "percent": true});
eq($ax.frameOn, false, "a pie turns the frame off");
approx($ax.artists[0]["total"], 100, "the total is the sum");
eq(len($ax.artists[0]["colors"]), 3, "one colour per slice");
$fg = plt.figure(400, 400); $fg.addAxes().pie([1,1,1], null);
$svg = $fg.to_svg();
has($svg, "<path", "pie emits arc paths");
// The frame is the only unfilled, stroked rect; the other two rects in any
// document are the background and the clip path.
hasnt($svg, "fill=\"none\" stroke=\"#888888\"", "and no axes frame");
// A single slice is a full circle; an arc from a point back to itself draws
// nothing at all.
$fg = plt.figure(400, 400); $fg.addAxes().pie([5], null);
$svg = $fg.to_svg();
has($svg, "<circle", "a single 100% slice is a circle, not an empty arc");
raisesWith(def() { $a = newAxes(); $a.pie([1,-2,3], null); },
           "-2", "a negative slice raises, naming the value");
raisesWith(def() { $a = newAxes(); $a.pie([1,-2,3], null); },
           "bar()", "and points at the chart type that does allow negatives");
raisesWith(def() { $a = newAxes(); $a.pie([0,0], null); }, "sum to zero", "an all-zero pie raises");
raisesWith(def() { $a = newAxes(); $a.pie([1,NAN], null); }, "finite", "a NaN slice raises");

print("");
print("-- text and annotations --");
$fig = plt.figure(400, 300);
$ax = $fig.addAxes();
$ax.plot([1,2,3], [1,2,3], null);
$ax.addText(2, 2, "here", {"rotate": 45});
$svg = $fig.to_svg();
has($svg, ">here<", "text is drawn");
has($svg, "rotate(45", "and rotated");
$fig = plt.figure(400, 300);
$ax = $fig.addAxes();
$ax.plot([1,2,3], [1,2,3], null);
$ax.annotate("peak", 1.5, 2.5, {"to": [2, 2]});
$svg = $fig.to_svg();
has($svg, ">peak<", "an annotation draws its text");
has($svg, "<polygon", "and an arrowhead");
has($svg, "<line", "and a shaft");
raisesWith(def() { $a = newAxes(); $a.annotate("x", 1, 1, {"to": [1]}); },
           "to", "a malformed arrow target raises");

// ════════════════════════════════════════════════════════════════════════
//  6. Scales through the full drawing path
// ════════════════════════════════════════════════════════════════════════
print("");
print("-- a log axis end to end --");
$fig = plt.figure(640, 480);
$ax = $fig.addAxes();
$ax.setScale("y", "log", null);
$ax.plot([1,2,3,4], [1,10,100,1000], null);
$ax.setGrid(true);
$svg = $fig.to_svg();
hasnt($svg, "NaN", "no NaN reaches the document");
hasnt($svg, "inf", "and no infinity");
has($svg, ">1000<", "the top decade is labelled");
has($svg, ">1<", "and the bottom one");

print("");
print("-- and the raise that a log axis exists to produce --");
raisesWith(def() {
    $f = plt.figure(400, 300);
    $a = $f.addAxes();
    $a.setScale("y", "log", null);
    $a.plot([1,2,3], [1,0,3], null);
    $f.to_svg();
}, "0", "a zero in the data raises when the log axis is drawn");
raisesWith(def() {
    $f = plt.figure(400, 300);
    $a = $f.addAxes();
    $a.setScale("y", "log", null);
    $a.plot([1,2,3], [1,-4,3], null);
    $f.to_svg();
}, "-4", "a negative in the data raises, naming it");
raisesWith(def() {
    $f = plt.figure(400, 300);
    $a = $f.addAxes();
    $a.setScale("x", "log", null);
    $a.setXLim(0, 100);
    $a.plot([1,2,3], [1,2,3], null);
    $f.to_svg();
}, "log", "an explicit xlim starting at zero raises too");

print("");
print("-- symlog draws data that crosses zero, which is the whole point --");
$fig = plt.figure(640, 480);
$ax = $fig.addAxes();
$ax.setScale("y", "symlog", null);
$ax.plot([1,2,3,4,5], [-1000, -10, 0, 10, 1000], null);
$svg = $fig.to_svg();
hasnt($svg, "NaN", "zero and negatives survive symlog");
has($svg, ">0<", "zero is a labelled tick");

print("");
print("-- a date axis end to end --");
$fig = plt.figure(720, 400);
$ax = $fig.addAxes();
$ax.setDateAxis(true);
$ax.plot([1704067200000, 1704153600000, 1704240000000, 1704326400000,
          1704412800000, 1704499200000, 1704585600000, 1704672000000],
         [3, 1, 4, 1, 5, 9, 2, 6], null);
$svg = $fig.to_svg();
has($svg, "2024-01-01", "dates are labelled as dates, not as milliseconds");
hasnt($svg, "1704067200000", "and never as raw epoch numbers");

print("");
print("-- a bar chart on a log axis draws from the view floor, not from zero --");
$fig = plt.figure(400, 300);
$ax = $fig.addAxes();
$ax.setScale("y", "log", null);
$ax.bar([1,2,3], [10,100,1000], null);
$svg = $fig.to_svg();
hasnt($svg, "NaN", "the zero baseline does not become log10(0)");

// ════════════════════════════════════════════════════════════════════════
//  7. Escaping still has no opt-out, through every new path
// ════════════════════════════════════════════════════════════════════════
print("");
print("-- hostile text through the B2 surfaces --");
$evil = "</text><script>alert(1)</script>";
$fig = plt.figure(500, 400);
$ax = $fig.addAxes();
$ax.boxplot([[1,2,3],[4,5,6]], {"labels": [$evil, "ok"]});
$ax.addText(1, 2, $evil, null);
$ax.annotate($evil, 1.5, 3, {"to": [2, 4]});
$ax.setTitle($evil);
$svg = $fig.to_svg();
hasnt($svg, "<script", "no script element survives");
hasnt($svg, "</text><text", "the text element cannot be closed early");
has($svg, "&lt;script&gt;", "it is escaped instead");
$fig = plt.figure(500, 400);
$ax = $fig.addAxes();
$ax.pie([1,1], {"labels": [$evil, "ok"]});
hasnt($fig.to_svg(), "<script", "pie slice labels are escaped too");

print("");
print("========================================");
print("  passed: " + str($R.pass) + "   failed: " + str($R.fail));
if ($R.fail == 0) { print("RESULT: ALL GREEN"); }
else { print("RESULT: FAILURES"); }
print("========================================");
