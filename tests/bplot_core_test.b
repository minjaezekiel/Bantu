// ════════════════════════════════════════════════════════════════════════
//  bplot_core_test.b — the B1 core: figures, axes, scales, ticks, SVG.
//
//  WHAT THIS IS GATING
//  bplot emits SVG, which is text, so the output is directly assertable:
//  that it is well-formed, that it contains the elements it should, that a
//  hostile label did not survive unescaped, and that degenerate data renders
//  or raises rather than producing a document a browser silently ignores.
//
//  THE THREE THINGS MOST WORTH TESTING, and why:
//
//    1. ESCAPING. SVG is not an image format the way PNG is -- it is XML
//       that browsers execute, and sua already serves image/svg+xml. A chart
//       title taken from a request parameter is a stored XSS unless every
//       text node and attribute is escaped. There is no opt-out and no raw
//       markup hatch, and that is asserted here rather than assumed.
//
//    2. NaN AND INFINITY. Untreated, a NaN coordinate emits
//       points="NaN,12 ..." which every browser renders as NOTHING AT ALL,
//       with no error in any log. That is the worst failure mode available
//       to a plotting library, so NaN must break the line into segments and
//       must never reach the document.
//
//    3. TICKS. "Pick about six round numbers spanning this range" is the one
//       thing that separates a chart that looks designed from one that looks
//       generated. These are checked against matplotlib's own MaxNLocator
//       choices across twenty ranges including negatives, tiny and huge.
//
//  Numbers are never asserted through str() -- six significant digits would
//  hide a real error. The exception is _fmt's own tests, where the string IS
//  the contract, and the SVG coordinates, which _fmt produces.
//
//  Run:  bantu run tests/bplot_core_test.b
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

def raises($f, $name) {
    $caught = false;
    try { $f(); } catch ($e) { $caught = true; }
    ok($caught, $name);
}

def has($hay, $needle, $name) { ok(contains($hay, $needle), $name); }
def hasnt($hay, $needle, $name) { ok(!contains($hay, $needle), $name); }

// Ticks as a comma-joined string, so a whole tick set is one assertion.
def ticksOf($lo, $hi, $d) {
    $t = plt._ticks($lo, $hi, 6);
    $out = [];
    $i = 0;
    while ($i < len($t)) { push($out, plt._fmt($t[$i], $d)); $i = $i + 1; }
    return join($out, ",");
}

print("========================================");
print("  bplot — core");
print("========================================");

// ── Number formatting ───────────────────────────────────────────────────
print("");
print("-- _fmt: str() gives six significant digits and leaks 1.23e-05 --");
eq(plt._fmt(0, 2), "0.00", "zero");
eq(plt._fmt(1, 0), "1", "an integer with no decimals");
eq(plt._fmt(2.5, 1), "2.5", "one decimal");
eq(plt._fmt(2.345, 2), "2.35", "rounds up at the half");
eq(plt._fmt(2.344, 2), "2.34", "and down below it");
eq(plt._fmt(1 / 3, 4), "0.3333", "a third, where str() would give 0.333333");
eq(plt._fmt(-0.5, 2), "-0.50", "a negative");
eq(plt._fmt(-0.004, 2), "0.00", "a negative that rounds to zero does not print as -0.00");
eq(plt._fmt(0.000012345678, 8), "0.00001235", "small, where str() leaks scientific notation");
eq(plt._fmt(1000000, 0), "1000000", "a million, in full");
eq(plt._fmt(0.999, 2), "1.00", "carrying into the integer part");
eq(plt._fmt(9.999, 2), "10.00", "and carrying a digit");
eq(plt._fmt(-9.999, 2), "-10.00", "negative, carrying");
eq(plt._fmt(NAN, 2), "NaN", "NaN is named, not printed as a coordinate");
eq(plt._fmt(INF, 2), "inf", "infinity");
eq(plt._fmt(0 - INF, 2), "-inf", "negative infinity");

// ── Ticks ───────────────────────────────────────────────────────────────
print("");
print("-- ticks: matplotlib's MaxNLocator, with 2.5 among the candidates --");
eq(ticksOf(0, 1, 2), "0.00,0.20,0.40,0.60,0.80,1.00", "0 to 1");
eq(ticksOf(0, 10, 0), "0,2,4,6,8,10", "0 to 10");
eq(ticksOf(0, 100, 0), "0,20,40,60,80,100", "0 to 100");
eq(ticksOf(0, 7, 0), "0,2,4,6", "0 to 7");
eq(ticksOf(-1, 1, 1), "-1.0,-0.5,0.0,0.5,1.0", "across zero");
eq(ticksOf(0, 3, 1), "0.0,0.5,1.0,1.5,2.0,2.5,3.0", "0 to 3 picks a half step");
eq(ticksOf(2.3, 7.9, 0), "3,4,5,6,7", "a range that starts and ends off-step");
eq(ticksOf(-50, 50, 0), "-40,-20,0,20,40", "negatives");
eq(ticksOf(0, 1000000, 0), "0,200000,400000,600000,800000,1000000", "a million");
eq(ticksOf(0, 0.001, 4), "0.0000,0.0002,0.0004,0.0006,0.0008,0.0010", "a thousandth");
eq(ticksOf(-0.5, 0.5, 1), "-0.4,-0.2,0.0,0.2,0.4", "a small range across zero");
eq(ticksOf(100, 101, 1), "100.0,100.2,100.4,100.6,100.8,101.0", "a narrow band far from zero");
ok(len(plt._ticks(0, 1e9, 6)) >= 4, "a billion still produces ticks");
ok(len(plt._ticks(0, 1e-9, 6)) >= 4, "and so does a billionth");

print("");
print("-- the top tick is not lost to floating-point drift --");
// Accumulating the step instead of computing k*step drops the final tick on
// exactly this range, and the top tick is the one a reader looks for.
$t01 = plt._ticks(0, 1, 6);
eq(len($t01), 6, "0 to 1 by 0.2 keeps all six");
eq(plt._fmt($t01[5], 1), "1.0", "including the last one");
$t03 = plt._ticks(0, 0.3, 6);
eq(plt._fmt($t03[len($t03) - 1], 2), "0.30", "0 to 0.3 keeps its top tick too");

print("");
print("-- a degenerate range does not hang or divide by zero --");
ok(len(plt._ticks(5, 5, 6)) >= 1, "lo == hi");
ok(len(plt._ticks(5, 4, 6)) >= 1, "hi below lo");
ok(len(plt._ticks(0, INF, 6)) >= 1, "an infinite bound");
ok(len(plt._ticks(NAN, 1, 6)) >= 1, "a NaN bound");

// ── Escaping: the security gate ─────────────────────────────────────────
print("");
print("-- escaping is mandatory and has no opt-out --");
eq(plt._esc("a<b"), "a&lt;b", "less-than");
eq(plt._esc("a>b"), "a&gt;b", "greater-than");
eq(plt._esc("a&b"), "a&amp;b", "ampersand");
eq(plt._esc("a\"b"), "a&quot;b", "double quote");
eq(plt._esc("a'b"), "a&#39;b", "single quote");
// & must be replaced FIRST or the others would be double-escaped.
eq(plt._esc("&lt;"), "&amp;lt;", "an already-escaped entity is escaped again, not left alone");
eq(plt._esc("<&>"), "&lt;&amp;&gt;", "all three together, in the right order");
eq(plt._esc(""), "", "empty");
eq(plt._esc("plain"), "plain", "nothing to do");
// Bytes below 0x20 are not representable in XML 1.0 at all -- one of them
// makes the whole document fail to parse, so they are dropped.
eq(plt._esc("a" + chr(7) + "b"), "ab", "a control byte is dropped, not escaped");
eq(plt._esc("a" + chr(9) + "b"), "a" + chr(9) + "b", "but tab is legal and survives");
eq(plt._esc("a" + chr(10) + "b"), "a" + chr(10) + "b", "and so is newline");

print("");
print("-- a hostile title cannot reach the document --");
$evil = "</text><script>alert(1)</script>";
$f1 = plt.figure(400, 300);
$a1 = $f1.addAxes();
$a1.plot([1, 2], [1, 2], null);
$a1.setTitle($evil);
$a1.setXLabel($evil);
$a1.setYLabel($evil);
$s1 = $f1.to_svg();
hasnt($s1, "<script", "no <script survives in the output");
hasnt($s1, "</text><text", "the text element was not closed early");
has($s1, "&lt;/text&gt;&lt;script&gt;", "the title appears, escaped");
ok(contains($s1, "<svg") && contains($s1, "</svg>"), "and the document is still a document");
// A label goes through the legend too, which is a different emission path.
$f1b = plt.figure(400, 300);
$a1b = $f1b.addAxes();
$a1b.plot([1, 2], [1, 2], {"label": $evil});
$a1b.setLegend(true);
hasnt($f1b.to_svg(), "<script", "a hostile legend label is escaped as well");

print("");
print("-- colours are validated, not interpolated into an attribute --");
eq(plt._color("#1f77b4", null), "#1f77b4", "a six-digit hex");
eq(plt._color("#abc", null), "#abc", "a three-digit hex");
eq(plt._color("red", null), "#d62728", "a named colour");
eq(plt._color([255, 0, 128], null), "#ff0080", "an [r, g, b] list");
eq(plt._color(null, "#123456"), "#123456", "null takes the fallback");
raises(def() { return plt._color("\" onload=\"alert(1)", null); },
       "an injection attempt through a colour raises");
raises(def() { return plt._color("#12g456", null); }, "a non-hex digit raises");
raises(def() { return plt._color("#12345", null); }, "a wrong-length hex raises");
raises(def() { return plt._color("chartreuse", null); }, "an unknown name raises, listing the options");
raises(def() { return plt._color([1, 2], null); }, "a short colour list raises");

// ── The document ────────────────────────────────────────────────────────
print("");
print("-- a chart contains what it should --");
$f2 = plt.figure(640, 480);
$ax2 = $f2.addAxes();
$ax2.plot([1, 2, 3, 4], [2, 4, 9, 3], {"label": "alpha"});
$ax2.plot([1, 2, 3, 4], [5, 1, 6, 2], {"label": "beta", "color": "red"});
$ax2.setTitle("Chart");
$ax2.setXLabel("time");
$ax2.setYLabel("value");
$ax2.setGrid(true);
$ax2.setLegend(true);
$s2 = $f2.to_svg();
has($s2, "<svg xmlns=\"http://www.w3.org/2000/svg\"", "an svg root with a namespace");
has($s2, "width=\"640.00\" height=\"480.00\"", "the size it was asked for");
has($s2, "viewBox=\"0 0 640.00 480.00\"", "and a matching viewBox");
has($s2, "<polyline", "the line is a polyline, not one element per point");
has($s2, "stroke=\"#1f77b4\"", "the first series takes the first cycle colour");
has($s2, "stroke=\"#d62728\"", "the second takes the colour it was given");
has($s2, "<clipPath", "the data is clipped to the axes box");
has($s2, "clip-path=\"url(#", "and the group references that clip");
has($s2, ">Chart</text>", "the title");
has($s2, ">time</text>", "the x label");
has($s2, ">value</text>", "the y label");
has($s2, "rotate(-90", "the y label is rotated");
has($s2, ">alpha</text>", "the legend names the first series");
has($s2, ">beta</text>", "and the second");
has($s2, "#dddddd", "the grid is drawn");
ok(len($s2) > 1000, "and the whole thing is a real document");

print("");
print("-- the document is balanced --");
def countOf($s, $needle) {
    $n = 0;
    $parts = split($s, $needle);
    return len($parts) - 1;
}
eq(countOf($s2, "<svg"), 1, "exactly one <svg");
eq(countOf($s2, "</svg>"), 1, "and one closing tag");
eq(countOf($s2, "<g"), countOf($s2, "</g>"), "every group is closed");
eq(countOf($s2, "<text"), countOf($s2, "</text>"), "every text element is closed");
eq(countOf($s2, "<clipPath"), countOf($s2, "</clipPath>"), "and every clip path");
hasnt($s2, "NaN", "no NaN reached a coordinate");
hasnt($s2, "inf", "and no infinity");
hasnt($s2, "undefined", "and nothing undefined");

// ── Transforms ──────────────────────────────────────────────────────────
print("");
print("-- data maps to pixels, and y is flipped exactly once --");
$f3 = plt.figure(200, 100);
$ax3 = $f3.addAxes();
$lim = [0, 10];
// The axes box is inset by the gutters; the mapping is linear across it.
$pxLo = $ax3.px(0, $lim);
$pxHi = $ax3.px(10, $lim);
$pxMid = $ax3.px(5, $lim);
ok($pxHi > $pxLo, "x increases to the right");
ok(abs($pxMid - ($pxLo + $pxHi) / 2) < 0.001, "and the midpoint is in the middle");
$pyLo = $ax3.py(0, $lim);
$pyHi = $ax3.py(10, $lim);
ok($pyHi < $pyLo, "y increases UPWARD, which is the flip -- SVG's origin is top-left");
$pyMid = $ax3.py(5, $lim);
ok(abs($pyMid - ($pyLo + $pyHi) / 2) < 0.001, "and y is linear too");

// ── Degenerate data ─────────────────────────────────────────────────────
print("");
print("-- degenerate data renders or raises; it never emits broken SVG --");
def svgOf($xs, $ys) {
    $f = plt.figure(300, 200);
    $a = $f.addAxes();
    $a.plot($xs, $ys, null);
    return $f.to_svg();
}
$sEmpty = svgOf([], []);
has($sEmpty, "</svg>", "an empty series still draws the axes");
hasnt($sEmpty, "NaN", "with no NaN in it");
$sOne = svgOf([5], [5]);
has($sOne, "</svg>", "a single point");
hasnt($sOne, "NaN", "with no NaN");
$sFlat = svgOf([1, 2, 3], [7, 7, 7]);
has($sFlat, "</svg>", "all y values equal");
hasnt($sFlat, "NaN", "with no NaN -- the range is expanded rather than divided by zero");
$sZero = svgOf([0, 0, 0], [0, 0, 0]);
has($sZero, "</svg>", "everything at the origin");
hasnt($sZero, "NaN", "still no NaN");

print("");
print("-- NaN breaks the line rather than silently emptying the chart --");
$sNaN = svgOf([1, 2, 3, 4, 5], [1, 2, NAN, 4, 5]);
hasnt($sNaN, "NaN", "no NaN reaches the document");
eq(countOf($sNaN, "<polyline"), 2, "the line is split into two runs at the gap");
$sInf = svgOf([1, 2, 3], [1, INF, 3]);
hasnt($sInf, "inf", "an infinity does not reach the document either");
has($sInf, "</svg>", "and the chart still renders");
$sAllNaN = svgOf([1, 2, 3], [NAN, NAN, NAN]);
has($sAllNaN, "</svg>", "all-NaN data draws the axes and nothing else");
hasnt($sAllNaN, "NaN", "with no NaN in the output");
$sLead = svgOf([1, 2, 3, 4], [NAN, 2, 3, 4]);
has($sLead, "<polyline", "a leading NaN still leaves a drawable run");

// ── Errors teach ────────────────────────────────────────────────────────
print("");
print("-- bad input raises, naming what was wrong --");
raises(def() { $f = plt.figure(300, 200); $a = $f.addAxes(); $a.plot([1, 2, 3], [1, 2], null); return 0; },
       "mismatched lengths raise");
raises(def() { $f = plt.figure(300, 200); $a = $f.addAxes(); $a.plot(null, [1], null); return 0; },
       "a null sequence raises");
raises(def() { $f = plt.figure(300, 200); $a = $f.addAxes(); $a.plot("nope", [1], null); return 0; },
       "a string where a sequence belongs raises");
raises(def() { return plt.figure(0, 100); }, "a zero-width figure raises");
raises(def() { return plt.figure(-5, 100); }, "a negative size raises");
raises(def() { return plt.figure("wide", 100); }, "a non-numeric size raises");
raises(def() { $f = plt.figure(300, 200); $a = $f.addAxes(); $a.setXLim(5, 1); return 0; },
       "an inverted xlim raises");
raises(def() { $f = plt.figure(300, 200); $a = $f.addAxes(); $a.setYLim(1, NAN); return 0; },
       "a NaN limit raises");
raises(def() { $f = plt.figure(300, 200); return $f.savefig("chart.jpg"); },
       "savefig to an unknown format raises (.svg and .png are the two)");
raises(def() { $f = plt.figure(300, 200); return $f.savefig(""); },
       "an empty path raises");

// A mismatched-length message must name BOTH lengths, or it does not help.
$msg = "";
try {
    $f = plt.figure(300, 200);
    $a = $f.addAxes();
    $a.plot([1, 2, 3], [1, 2], null);
} catch ($e) { $msg = str($e); }
ok(contains($msg, "3") && contains($msg, "2"), "and the message names both lengths");

// ── Explicit limits ─────────────────────────────────────────────────────
print("");
print("-- explicit limits are honoured --");
$f4 = plt.figure(300, 200);
$ax4 = $f4.addAxes();
$ax4.plot([1, 2, 3], [1, 2, 3], null);
$ax4.setXLim(0, 10);
$ax4.setYLim(0, 100);
$lx = $ax4.viewLimits("x");
eq($lx[0], 0, "xlim lower bound is used exactly, with no margin added");
eq($lx[1], 10, "and the upper bound");
$ly = $ax4.viewLimits("y");
eq($ly[0], 0, "ylim lower bound");
eq($ly[1], 100, "and upper");

print("");
print("-- automatic limits pad the data so points are not on the frame --");
$f5 = plt.figure(300, 200);
$ax5 = $f5.addAxes();
$ax5.plot([0, 10], [0, 10], null);
$l5 = $ax5.viewLimits("x");
ok($l5[0] < 0, "the lower bound sits below the smallest point");
ok($l5[1] > 10, "and the upper above the largest");

// ── Other chart types ───────────────────────────────────────────────────
print("");
print("-- scatter, bar and barh --");
$f6 = plt.figure(320, 240);
$ax6 = $f6.addAxes();
$ax6.scatter([1, 2, 3], [3, 1, 2], {"size": 4});
$s6 = $f6.to_svg();
eq(countOf($s6, "<circle"), 3, "scatter draws one circle per point");
hasnt($s6, "NaN", "with clean coordinates");

$f7 = plt.figure(320, 240);
$ax7 = $f7.addAxes();
$ax7.bar([1, 2, 3], [4, 7, 2], null);
$s7 = $f7.to_svg();
// The background and the frame are rects too, hence >= rather than ==.
ok(countOf($s7, "<rect") >= 5, "bar draws a rect per bar");
$ly7 = $ax7.viewLimits("y");
ok($ly7[0] <= 0, "a bar chart's value axis includes zero, or the bars would float");

$f8 = plt.figure(320, 240);
$ax8 = $f8.addAxes();
$ax8.barh([1, 2, 3], [4, 7, 2], null);
$s8 = $f8.to_svg();
ok(countOf($s8, "<rect") >= 5, "barh draws its bars");
$lx8 = $ax8.viewLimits("x");
ok($lx8[0] <= 0, "and its value axis is the x one");

$f9 = plt.figure(320, 240);
$ax9 = $f9.addAxes();
$ax9.bar([1, 2], [0 - 3, 5], null);
$ly9 = $ax9.viewLimits("y");
ok($ly9[0] < 0 && $ly9[1] > 0, "negative bars extend the axis the other way");

print("");
print("-- a single bar still has a width --");
$f10 = plt.figure(320, 240);
$ax10 = $f10.addAxes();
$ax10.bar([1], [5], null);
$s10 = $f10.to_svg();
ok(countOf($s10, "<rect") >= 3, "one bar renders");
hasnt($s10, "NaN", "with no NaN from a zero gap between neighbours");

// ── The stateful API ────────────────────────────────────────────────────
print("");
print("-- the stateful API is the quickstart, and shares one figure --");
plt.clf();
plt.plot([1, 2, 3], [1, 4, 9], {"label": "squares"});
plt.title("Stateful");
plt.grid(true);
plt.legend(true);
$sState = plt.to_svg();
has($sState, ">Stateful</text>", "plt.title reached the figure plt.plot created");
has($sState, ">squares</text>", "and so did the legend label");
has($sState, "<polyline", "and the data");
// Two plt.plot calls must land on the SAME axes, not two figures.
plt.clf();
plt.plot([1, 2], [1, 2], null);
plt.plot([1, 2], [2, 1], null);
eq(countOf(plt.to_svg(), "<polyline"), 2, "two plt.plot calls share one axes");
plt.clf();
eq(countOf(plt.to_svg(), "<polyline"), 0, "clf() starts a fresh figure");

// ── Saving ──────────────────────────────────────────────────────────────
print("");
print("-- savefig writes the file, and it round-trips --");
plt.clf();
plt.plot([1, 2, 3], [4, 5, 6], null);
plt.title("Saved");
$path = "/tmp/bplot_core_test.svg";
eq(plt.savefig($path), $path, "savefig returns the path it wrote");
ok(file_exists($path), "and the file is there");
$back = readfile($path);
has($back, "<svg", "what was written is an svg");
has($back, ">Saved</text>", "with the title in it");
has($back, "</svg>", "and it is closed");

print("");
print("-- path simplification bounds the output by the canvas, not the data --");
// A 100k-point line cannot show more than ~canvas-width distinct columns, so
// emitting 100k coordinates carries no information that ~4x the width does.
$bigX = [];
$bigY = [];
$i = 0;
while ($i < 20000) {
    push($bigX, $i);
    push($bigY, sin($i / 200) * 100);
    $i = $i + 1;
}
$f11 = plt.figure(600, 400);
$ax11 = $f11.addAxes();
$ax11.plot($bigX, $bigY, null);
$s11 = $f11.to_svg();
eq(countOf($s11, "<polyline"), 1, "20,000 points are one polyline");
ok(len($s11) < 200000, "and the document stays small -- simplification fired");
hasnt($s11, "NaN", "with clean coordinates throughout");
// Switching it off must produce a bigger document, or it was never doing
// anything in the first place.
$f12 = plt.figure(600, 400);
$ax12 = $f12.addAxes();
$ax12.simplify = false;
$ax12.plot($bigX, $bigY, null);
ok(len($f12.to_svg()) > len($s11), "turning it off produces a larger document");

print("");
print("-- help() and version() --");
ok(plt.version() != null, "version() answers");
plt.help();
ok(true, "help() prints without raising");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
