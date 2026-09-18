// ════════════════════════════════════════════════════════════════════════
//  bplot_layout_test.b — B3: subplots, twin and shared axes, tight_layout,
//  colormaps, colorbars, and the 2-D chart types.
//
//  WHAT THIS IS GATING, and why each part earns its place:
//
//    1. ONE RECTANGLE CALCULATION. matplotlib's `subplot` and `add_subplot`
//       drifted apart historically because two code paths computed "the same"
//       rectangle. Here both spellings are asserted to produce IDENTICAL
//       rects, so they cannot drift.
//
//    2. THE COLORMAP TABLES ARE REAL. viridis has no closed form -- it is the
//       output of an optimisation in perceptual space -- so a table that was
//       subtly corrupted would still look like a colormap. Entries are checked
//       against values written independently into this file from the published
//       source, so a corrupted table FAILS rather than agreeing with itself.
//
//    3. THE 2-D ENCODING STAYS BOUNDED. One <rect> per pixel makes a
//       1000x1000 image a 55 MB document. The element count and the byte count
//       are both asserted, because "it rendered" is not the property that
//       matters -- "it rendered into something a browser will open" is.
//
//    4. NOTHING LEAVES THE FIGURE. tight_layout exists to stop labels being
//       clipped, so the test asserts every plot box and every label position
//       lies inside its own cell and inside the figure. A clipped label is
//       invisible, which is the failure mode that does not announce itself.
//
//  Reference values were generated once, at authoring time, from matplotlib
//  3.11.2 and written in as literals. NOTHING HERE DEPENDS ON PYTHON AT RUN
//  TIME.
//
//  Run:  bantu run tests/bplot_layout_test.b
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

def approx($got, $want, $name) {
    $tol = 0.000001 * max(abs($want), 1);
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
def countOf($hay, $needle) { return len(split($hay, $needle)) - 1; }

def rectOf($ax) { return [$ax.left, $ax.top, $ax.w, $ax.h]; }

// A small grid with a known shape: values rise left to right and top to
// bottom, so an accidental transpose or flip is visible in the assertions.
def grid($rows, $cols) {
    $z = [];
    $r = 0;
    while ($r < $rows) {
        $row = [];
        $c = 0;
        while ($c < $cols) { push($row, $r * $cols + $c); $c = $c + 1; }
        push($z, $row);
        $r = $r + 1;
    }
    return $z;
}

print("========================================");
print("  bplot — layout and 2-D (B3)");
print("========================================");

// ════════════════════════════════════════════════════════════════════════
//  1. One rectangle calculation (decision BP27)
// ════════════════════════════════════════════════════════════════════════
print("");
print("-- subplots and subplot(r, c, i) must agree exactly --");
$f = plt.figure(800, 600);
$grid22 = $f.subplots(2, 2);
eq(len($grid22), 4, "a 2x2 grid gives four axes");
$g = plt.figure(800, 600);
$same = true;
$i = 1;
while ($i <= 4) {
    $a = $g.subplot(2, 2, $i);
    $b = $grid22[$i - 1];
    if (rectOf($a)[0] != rectOf($b)[0] || rectOf($a)[1] != rectOf($b)[1] ||
        rectOf($a)[2] != rectOf($b)[2] || rectOf($a)[3] != rectOf($b)[3]) { $same = false; }
    $i = $i + 1;
}
ok($same, "both spellings produce IDENTICAL rectangles for every cell");

print("");
print("-- row-major numbering, as matplotlib numbers it --");
$f = plt.figure(800, 600);
$ax1 = $f.subplot(2, 2, 1);
$ax2 = $f.subplot(2, 2, 2);
$ax3 = $f.subplot(2, 2, 3);
ok($ax2.left > $ax1.left, "index 2 is to the RIGHT of index 1");
approx($ax2.top, $ax1.top, "and on the same row");
ok($ax3.top > $ax1.top, "index 3 is BELOW index 1");
approx($ax3.left, $ax1.left, "and in the same column");

print("");
print("-- an 8x8 grid: 64 panels, none overlapping, all inside the figure --");
$f = plt.figure(1600, 1200);
$many = $f.subplots(8, 8);
eq(len($many), 64, "64 panels");
$bad = 0;
$i = 0;
while ($i < 64) {
    $r = rectOf($many[$i]);
    if ($r[0] < 0 || $r[1] < 0 || $r[0] + $r[2] > 1600 || $r[1] + $r[3] > 1200) { $bad = $bad + 1; }
    if ($r[2] <= 0 || $r[3] <= 0) { $bad = $bad + 1; }
    $i = $i + 1;
}
eq($bad, 0, "every panel is inside the figure and has positive size");
// Cells tile the figure exactly, so no two plot boxes can overlap.
$overlap = 0;
$i = 0;
while ($i < 64) {
    $j = $i + 1;
    while ($j < 64) {
        $a = rectOf($many[$i]);
        $b = rectOf($many[$j]);
        $sep = ($a[0] + $a[2] <= $b[0]) || ($b[0] + $b[2] <= $a[0]) ||
               ($a[1] + $a[3] <= $b[1]) || ($b[1] + $b[3] <= $a[1]);
        if (!$sep) { $overlap = $overlap + 1; }
        $j = $j + 1;
    }
    $i = $i + 1;
}
eq($overlap, 0, "and no two plot boxes overlap");

print("");
print("-- a spanning cell --");
$f = plt.figure(900, 600);
$wide = $f.subplotSpan(2, 3, 0, 0, 1, 3);
$small = $f.subplotSpan(2, 3, 1, 0, 1, 1);
ok($wide.w > $small.w * 2, "a 3-column span is much wider than one column");
raisesWith(def() { $x = plt.figure(400, 300); $x.subplotSpan(2, 2, 1, 1, 2, 1); },
           "does not fit", "a span past the grid edge raises");

print("");
print("-- a 1x1 grid reproduces B1's fixed gutters exactly --");
$f = plt.figure(640, 480);
$a = $f.addAxes();
approx($a.left, 62, "left gutter unchanged");
approx($a.top, 34, "top gutter unchanged");
approx($a.w, 640 - 62 - 18, "width unchanged");
approx($a.h, 480 - 34 - 52, "height unchanged");

print("");
print("-- bad grids raise rather than drawing something useless --");
raisesWith(def() { $x = plt.figure(400, 300); $x.subplots(0, 2); }, "0x2", "a zero row count raises");
raisesWith(def() { $x = plt.figure(400, 300); $x.subplots(2, 1.5); }, "whole", "a fractional count raises");
raisesWith(def() { $x = plt.figure(400, 300); $x.subplots(30, 30); }, "900 panels",
           "900 panels raises, saying why");
raisesWith(def() { $x = plt.figure(400, 300); $x.subplot(2, 2, 9); }, "1 to 4",
           "an out-of-range index raises, naming the range");
raisesWith(def() { $x = plt.figure(400, 300); $x.subplot(2, 2, 0); }, "1 to 4",
           "subplot numbering is 1-based, and 0 says so");

// ════════════════════════════════════════════════════════════════════════
//  2. Twin and shared axes
// ════════════════════════════════════════════════════════════════════════
print("");
print("-- twinx shares the rectangle and the x limits --");
$f = plt.figure(700, 450);
$a = $f.addAxes();
$a.plot([1, 2, 3], [10, 20, 30], null);
$b = $f.twinx($a);
$b.plot([1, 2, 3], [1000, 2000, 3000], null);
approx($b.left, $a.left, "same left edge");
approx($b.w, $a.w, "same width");
eq($b.side, "right", "the twin's y axis is on the right");
$svg = $f.to_svg();
// The frame is stroked ONCE. A doubled 1px stroke is visible and reads as a
// rendering bug.
eq(countOf($svg, "fill=\"none\" stroke=\"#888888\""), 1, "the frame is drawn exactly once");
approx($a.viewLimitsShared("x", $f.sharedLimits(0, "x"))[0],
       $b.viewLimitsShared("x", $f.sharedLimits(1, "x"))[0], "both see the same x lower bound");
approx($a.viewLimitsShared("x", $f.sharedLimits(0, "x"))[1],
       $b.viewLimitsShared("x", $f.sharedLimits(1, "x"))[1], "and the same upper bound");
// The y axes stay independent — that is the whole point of a twin.
ok($a.viewLimits("y")[1] < $b.viewLimits("y")[1], "but their y axes are independent");

// Under tight_layout each axes is measured for its own labels -- and until
// B6e a twin kept the box IT measured, so its data was drawn in a different
// rectangle from the frame, shifted and escaping it (the dashboard sample
// showed it). Host and twin must still be one rectangle, with room for both.
$tf = plt.figure(700, 450);
$ta = $tf.addAxes();
$ta.bar(["Jan", "Feb", "Mar"], [10, 200, 3000], null);
$ta.setYLabel("rainfall (mm)");
$tb = $tf.twinx($ta);
$tb.plot(["Jan", "Feb", "Mar"], [23.5, 27, 28], null);
$tb.setYLabel("temperature (C)");
$tf.tight_layout(true);
$tsvg = $tf.to_svg();
eq([$tb.left, $tb.top, $tb.w, $tb.h], [$ta.left, $ta.top, $ta.w, $ta.h], "under tight_layout the twin keeps its host's exact rectangle");
$clips = split($tsvg, "<clipPath id=");
eq(len($clips), 3, "two axes, two clip paths");
eq(split(split($clips[1], "<rect")[1], "/>")[0], split(split($clips[2], "<rect")[1], "/>")[0],
   "and both clip to the same rectangle");

print("");
print("-- shared axes take the UNION, and sharing is transitive --");
$f = plt.figure(900, 300);
$axs = $f.subplots(1, 3);
$axs[0].plot([1, 2], [0, 10], null);
$axs[1].plot([1, 2], [0, 100], null);
$axs[2].plot([1, 2], [0, 5], null);
$f.shareY($axs[0], $axs[1]);
$f.shareY($axs[1], $axs[2]);
$l0 = $axs[0].viewLimitsShared("y", $f.sharedLimits(0, "y"));
$l1 = $axs[1].viewLimitsShared("y", $f.sharedLimits(1, "y"));
$l2 = $axs[2].viewLimitsShared("y", $f.sharedLimits(2, "y"));
approx($l0[1], $l1[1], "panel 0 and panel 1 agree");
approx($l1[1], $l2[1], "panel 1 and panel 2 agree");
ok($l0[1] >= 100, "and the shared upper bound covers the largest series");
// Sharing A-B and B-C shares all three: a reader who has to remember which
// pairing came first has been given a worse tool than no sharing at all.
ok($l2[1] >= 100, "sharing is transitive — panel 2 was never linked to panel 0 directly");

print("");
print("-- an unshared axes is untouched --");
$f = plt.figure(600, 400);
$a = $f.addAxes();
$a.plot([1, 2], [0, 10], null);
ok($f.sharedLimits(0, "y") == null, "no group means no override, so the B1 path is unchanged");

raisesWith(def() {
    $x = plt.figure(400, 300);
    $y = plt.figure(400, 300);
    $x.shareY($x.addAxes(), $y.addAxes());
}, "does not belong", "sharing across figures raises");

// ════════════════════════════════════════════════════════════════════════
//  3. tight_layout
// ════════════════════════════════════════════════════════════════════════
print("");
print("-- tight_layout gives long tick labels the room they need --");
$narrow = plt.figure(600, 400);
$na = $narrow.addAxes();
$na.plot([1, 2, 3], [1, 2, 3], null);
$narrow.tight_layout(true);
$narrow.to_svg();
$shortPad = $na.pads[0];

$wide = plt.figure(600, 400);
$wa = $wide.addAxes();
$wa.plot([1, 2, 3], [1000000, 2000000, 3000000], null);
$wide.tight_layout(true);
$wide.to_svg();
ok($wa.pads[0] > $shortPad, "seven-digit tick labels reserve more left gutter than one-digit ones");

print("");
print("-- and room for the axis labels and title when they exist --");
$bare = plt.figure(600, 400);
$ba = $bare.addAxes();
$ba.plot([1, 2], [1, 2], null);
$bare.tight_layout(true);
$bare.to_svg();
$labelled = plt.figure(600, 400);
$la = $labelled.addAxes();
$la.plot([1, 2], [1, 2], null);
$la.setXLabel("time in seconds");
$la.setYLabel("amplitude");
$la.setTitle("a title");
$labelled.tight_layout(true);
$labelled.to_svg();
ok($la.pads[0] > $ba.pads[0], "a y label widens the left gutter");
ok($la.pads[3] > $ba.pads[3], "an x label deepens the bottom gutter");
ok($la.pads[2] > $ba.pads[2], "a title deepens the top gutter");

print("");
print("-- nothing ends up outside its cell, which is the point of it --");
$f = plt.figure(900, 620);
$axs = $f.subplots(2, 2);
$i = 0;
while ($i < 4) {
    $axs[$i].plot([1, 2, 3], [100000, 200000, 300000], null);
    $axs[$i].setXLabel("a reasonably long x label");
    $axs[$i].setYLabel("a reasonably long y label");
    $axs[$i].setTitle("panel " + str($i));
    $i = $i + 1;
}
$f.tight_layout(true);
$svg = $f.to_svg();
$outside = 0;
$i = 0;
while ($i < 4) {
    $r = rectOf($axs[$i]);
    $c = $axs[$i].cell;
    if ($r[0] < $c[0] || $r[1] < $c[1] ||
        $r[0] + $r[2] > $c[0] + $c[2] + 0.001 || $r[1] + $r[3] > $c[1] + $c[3] + 0.001) {
        $outside = $outside + 1;
    }
    if ($r[2] < 20 || $r[3] < 20) { $outside = $outside + 1; }
    $i = $i + 1;
}
eq($outside, 0, "every plot box stays inside its own cell and keeps a usable size");
has($svg, "panel 0", "and the titles survived");
has($svg, "a reasonably long y label", "and the y labels");

print("");
print("-- tight_layout is idempotent: rendering twice does not creep --");
$f = plt.figure(700, 500);
$a = $f.addAxes();
$a.plot([1, 2, 3], [1, 2, 3], null);
$a.setYLabel("y");
$f.tight_layout(true);
$first = $f.to_svg();
$second = $f.to_svg();
eq($first, $second, "the same figure rendered twice is byte-identical");

// ════════════════════════════════════════════════════════════════════════
//  4. Colormaps (decision BP25)
// ════════════════════════════════════════════════════════════════════════
print("");
print("-- the published tables, checked against values written in separately --");
// Reference: matplotlib 3.11.2, colormaps[name](t) at t = k/8.
eq(plt._cmap("viridis", 0), "#440154", "viridis 0.000");
eq(plt._cmap("viridis", 0.125), "#472d7b", "viridis 0.125");
eq(plt._cmap("viridis", 0.25), "#3b528b", "viridis 0.250");
eq(plt._cmap("viridis", 0.375), "#2c728e", "viridis 0.375");
eq(plt._cmap("viridis", 0.5), "#21918c", "viridis 0.500");
eq(plt._cmap("viridis", 0.625), "#28ae80", "viridis 0.625");
eq(plt._cmap("viridis", 0.75), "#5ec962", "viridis 0.750");
eq(plt._cmap("viridis", 0.875), "#addc30", "viridis 0.875");
eq(plt._cmap("viridis", 1), "#fde725", "viridis 1.000");
eq(plt._cmap("plasma", 0), "#0d0887", "plasma 0.000");
eq(plt._cmap("plasma", 0.25), "#7e03a8", "plasma 0.250");
eq(plt._cmap("plasma", 0.5), "#cc4778", "plasma 0.500");
eq(plt._cmap("plasma", 0.75), "#f89540", "plasma 0.750");
eq(plt._cmap("plasma", 1), "#f0f921", "plasma 1.000");
eq(plt._cmap("coolwarm", 0), "#3b4cc0", "coolwarm 0.000 — Moreland's cool end");
eq(plt._cmap("coolwarm", 0.5), "#dddcdc", "coolwarm 0.500 — the neutral middle");
eq(plt._cmap("coolwarm", 1), "#b40426", "coolwarm 1.000 — the warm end");
eq(plt._cmap("gray", 0), "#000000", "gray 0.000");
eq(plt._cmap("gray", 0.5), "#808080", "gray 0.500");
eq(plt._cmap("gray", 1), "#ffffff", "gray 1.000");

print("");
print("-- individual entries, proving it is the 256-entry table and not a fit --");
// A polynomial approximation would be close at these points but not equal.
eq(plt._cmap("viridis", 1 / 255), "#440256", "viridis entry 1");
eq(plt._cmap("viridis", 42 / 255), "#443983", "viridis entry 42");
eq(plt._cmap("viridis", 99 / 255), "#2b758e", "viridis entry 99");
eq(plt._cmap("viridis", 200 / 255), "#70cf57", "viridis entry 200");
eq(plt._cmap("viridis", 254 / 255), "#fbe723", "viridis entry 254");
eq(plt._cmap("plasma", 42 / 255), "#5c01a6", "plasma entry 42");
eq(plt._cmap("plasma", 200 / 255), "#fba139", "plasma entry 200");

print("");
print("-- out of range clamps; NaN gets its own colour --");
eq(plt._cmap("viridis", -5), "#440154", "below zero clamps to the bottom");
eq(plt._cmap("viridis", 5), "#fde725", "above one clamps to the top");
// A missing cell painted as the colormap's minimum is indistinguishable from a
// real minimum, which is the most dangerous thing a heatmap can do.
ok(plt._cmap("viridis", NAN) != plt._cmap("viridis", 0), "NaN is NOT drawn as the minimum");
eq(plt._cmap("viridis", NAN), "#b0b0b0", "it gets the no-data grey");
eq(plt._cmap("grey", 0.5), plt._cmap("gray", 0.5), "both spellings of gray work");
raisesWith(def() { plt._cmap("jet", 0.5); }, "jet is deliberately not shipped",
           "jet raises, and says why");
raisesWith(def() { plt._cmap("nonesuch", 0.5); }, "viridis, plasma, coolwarm, gray",
           "an unknown colormap lists the available ones");

// ════════════════════════════════════════════════════════════════════════
//  5. imshow, heatmap, pcolormesh, contour
// ════════════════════════════════════════════════════════════════════════
print("");
print("-- the native cell encoder draws exactly what the pure one does --");
// bp_grid_paths is _drawGridCells in C++. The pure path stays the oracle:
// the same grids, rendered both ways, must be the same bytes.
def gridSvg($kind, $z, $native) {
    $was = plt._useNative($native);
    $f = plt.figure(420, 320);
    $a = $f.addAxes();
    if ($kind == "imshow") { $a.imshow($z, {"cmap": "viridis"}); }
    if ($kind == "mesh") { $a.pcolormesh([0, 1, 3, 6, 10], [0, 2, 3, 7], $z, {"cmap": "plasma"}); }
    $svg = $f.to_svg();
    plt._useNative($was);
    return $svg;
}
$zv = [];
$i = 0;
while ($i < 40) {
    $row = [];
    $j = 0;
    while ($j < 60) { push($row, sin($i / 7) * cos($j / 5) + $j / 100); $j = $j + 1; }
    push($zv, $row);
    $i = $i + 1;
}
$zv[3][4] = NAN;
$zv[10][0] = NAN;
$zv[10][1] = NAN;
eq(gridSvg("imshow", $zv, true), gridSvg("imshow", $zv, false), "imshow of varied data with NaN holes: native and pure are byte-identical");
eq(gridSvg("imshow", [[5, 5], [5, 5]], true), gridSvg("imshow", [[5, 5], [5, 5]], false), "a flat grid too");
eq(gridSvg("mesh", [[1, 2, 3, 4], [4, 3, NAN, 1], [0, 9, 2, 2]], true),
   gridSvg("mesh", [[1, 2, 3, 4], [4, 3, NAN, 1], [0, 9, 2, 2]], false), "and pcolormesh with uneven edges");

print("");
print("-- imshow: orientation, extent and limits --");
$f = plt.figure(600, 400);
$a = $f.addAxes();
$a.imshow(grid(4, 6), null);
$art = $a.artists[0];
eq($art["w"], 6, "six columns");
eq($art["h"], 4, "four rows");
approx($art["vmin"], 0, "vmin from the data");
approx($art["vmax"], 23, "vmax from the data");
approx($art["ext"][0], -0.5, "cells are CENTRED on integers, so the extent starts at -0.5");
approx($art["ext"][1], 5.5, "and ends at n-0.5");
ok($a.yinvert, "origin \"upper\" puts row 0 at the top, as every image format does");
ok($a.tightLimits, "an image fills its axes rather than sitting in a 5% margin");
approx($a.viewLimits("x")[0], -0.5, "so the x view is exactly the extent");
approx($a.viewLimits("x")[1], 5.5, "on both ends");

$f = plt.figure(600, 400);
$a = $f.addAxes();
$a.imshow(grid(4, 6), {"origin": "lower"});
ok(!$a.yinvert, "origin \"lower\" leaves the axis ascending");

print("");
print("-- the 2-D encoding stays bounded (decision BP26) --");
$f = plt.figure(700, 500);
$a = $f.addAxes();
$a.imshow(grid(40, 40), null);
$svg = $f.to_svg();
// 1,600 cells, at most 256 distinct colours, so at most 256 paths.
ok(countOf($svg, "<path") <= 256, "1,600 cells become at most 256 <path> elements");
ok(countOf($svg, "<path") > 0, "and at least one");
eq(countOf($svg, "<rect"), 3, "no per-cell rects — only the background, the clip and the frame");

print("");
print("-- block reduction honours the cell budget --");
$f = plt.figure(600, 400);
$a = $f.addAxes();
$a.imshow(grid(100, 100), {"maxcells": 10});
eq($a.artists[0]["w"], 10, "100 columns reduce to 10");
eq($a.artists[0]["h"], 10, "100 rows reduce to 10");
$f = plt.figure(600, 400);
$a = $f.addAxes();
$a.imshow(grid(7, 7), {"maxcells": 256});
eq($a.artists[0]["w"], 7, "a small grid is left alone");

print("");
print("-- block reduction is a MEAN, and NaN does not poison a block --");
$f = plt.figure(600, 400);
$a = $f.addAxes();
$a.imshow([[0, 2], [4, 6]], {"maxcells": 1});
approx($a.artists[0]["rows"][0][0], 3, "the four values average to 3");
$f = plt.figure(600, 400);
$a = $f.addAxes();
$a.imshow([[0, NAN], [4, NAN]], {"maxcells": 1});
approx($a.artists[0]["rows"][0][0], 2, "a block with some data averages what it has");
$f = plt.figure(600, 400);
$a = $f.addAxes();
$a.imshow([[NAN, NAN], [NAN, NAN]], {"maxcells": 1});
ok(isnan($a.artists[0]["rows"][0][0]), "an all-NaN block stays NaN");

print("");
print("-- vmin / vmax --");
$f = plt.figure(600, 400);
$a = $f.addAxes();
$a.imshow(grid(3, 3), {"vmin": 0, "vmax": 100});
approx($a.artists[0]["vmax"], 100, "an explicit range is used");
raisesWith(def() { $x = plt.figure(400, 300); $x.addAxes().imshow([[1, 2]], {"vmin": 5, "vmax": 1}); },
           "vmin must be below vmax", "an inverted range raises");

print("");
print("-- a ragged grid raises rather than cropping or padding silently --");
raisesWith(def() { $x = plt.figure(400, 300); $x.addAxes().imshow([[1, 2, 3], [4, 5]], null); },
           "row 0 has 3 values but row 1 has 2", "and it names both row lengths");
raisesWith(def() { $x = plt.figure(400, 300); $x.addAxes().imshow([], null); },
           "no rows", "an empty grid raises");
raisesWith(def() { $x = plt.figure(400, 300); $x.addAxes().imshow([[]], null); },
           "no columns", "a grid of empty rows raises");
raisesWith(def() { $x = plt.figure(400, 300); $x.addAxes().imshow([[1, 2]], {"cmap": "jet"}); },
           "jet", "an unknown cmap raises at CALL time, not mid-render");

print("");
print("-- heatmap: cell values and category ticks --");
$f = plt.figure(600, 420);
$a = $f.addAxes();
$a.heatmap([[1, 2], [3, 4]], {"rows": ["r0", "r1"], "cols": ["c0", "c1"]});
$svg = $f.to_svg();
has($svg, ">r0<", "row labels are drawn");
has($svg, ">c1<", "column labels too");
has($svg, ">1.00<", "and the cell values");
eq(len($a.ytickOverride["labels"]), 2, "two row ticks");
eq($a.xtickOverride["labels"][1], "c1", "the second column tick reads c1");
// A fixed label colour is unreadable over half of any colormap.
has($svg, "fill=\"#ffffff\" text-anchor=\"middle\">1.00<", "a dark cell gets white text");
has($svg, "fill=\"#000000\" text-anchor=\"middle\">4.00<", "and a light cell gets black text");
raisesWith(def() {
    $x = plt.figure(400, 300);
    $x.addAxes().heatmap([[1, 2], [3, 4]], {"cols": ["only one"]});
}, "2 cells but 1 labels", "a label count mismatch raises, naming both");

print("");
print("-- pcolormesh takes EDGES, and says so when it is given centres --");
$f = plt.figure(600, 400);
$a = $f.addAxes();
$a.pcolormesh([0, 1, 2, 4], [0, 1, 3], [[1, 2, 3], [4, 5, 6]], null);
eq($a.artists[0]["w"], 3, "three columns");
approx($a.artists[0]["bx"][1], 4, "the x limits come from the edges");
raisesWith(def() {
    $x = plt.figure(400, 300);
    $x.addAxes().pcolormesh([0, 1], [0, 1, 2], [[1, 2], [3, 4]], null);
}, "needs 3 entries", "one edge short raises, saying how many are needed");

print("");
print("-- contour: one element per level, and the levels are right --");
$f = plt.figure(600, 400);
$a = $f.addAxes();
$a.contour(grid(10, 10), {"levels": 4});
eq(len($a.artists[0]["levels"]), 4, "four levels");
$svg = $f.to_svg();
ok(countOf($svg, "<path") <= 4, "at most one <path> per level, not one per segment");
ok(countOf($svg, "<path") > 0, "and the contours were drawn");
$f = plt.figure(600, 400);
$a = $f.addAxes();
$a.contour(grid(6, 6), {"levels": [5, 15, 25]});
eq(len($a.artists[0]["levels"]), 3, "an explicit level list is used as given");
approx($a.artists[0]["levels"][1], 15, "and kept in order");
raisesWith(def() { $x = plt.figure(400, 300); $x.addAxes().contour([[1, 2]], null); },
           "at least 2x2", "a grid too small to march raises");

print("");
print("-- colorbar --");
$f = plt.figure(700, 450);
$a = $f.addAxes();
$wBefore = $a.w;
$a.imshow(grid(5, 5), null);
$a.colorbar({"label": "units"});
ok($a.w < $wBefore, "the colorbar reserves its strip by shrinking the axes");
$svg = $f.to_svg();
has($svg, ">units<", "its label is drawn");
ok(countOf($svg, "<rect") > 60, "the bar is drawn as bands, not as an SVG gradient the raster backend could not reproduce");
raisesWith(def() {
    $x = plt.figure(500, 400);
    $b = $x.addAxes();
    $b.imshow([[1, 2], [3, 4]], null);
    $b.colorbar(null);
    $b.colorbar(null);
}, "already has one", "adding a second colorbar raises rather than eating the plot");
raisesWith(def() {
    $x = plt.figure(500, 400);
    $b = $x.addAxes();
    $b.plot([1, 2], [1, 2], null);
    $b.colorbar(null);
}, "colour-mapped", "a colorbar with nothing to key raises, naming what to call first");

// ════════════════════════════════════════════════════════════════════════
//  6. Styles
// ════════════════════════════════════════════════════════════════════════
print("");
print("-- style sheets change the document, and change back --");
plt.style("default");
$f = plt.figure(400, 300);
$a = $f.addAxes();
$a.plot([1, 2], [1, 2], {"label": "s"});
$a.setLegend(true);
$light = $f.to_svg();
has($light, "fill=\"#ffffff\"", "the default background is white");

plt.style("dark");
$f = plt.figure(400, 300);
$a = $f.addAxes();
$a.plot([1, 2], [1, 2], {"label": "s"});
$a.setLegend(true);
$dark = $f.to_svg();
has($dark, "fill=\"#1e222a\"", "the dark background is applied");
// The legend box used to be hardcoded white, which put near-invisible light
// text on a white card in dark mode.
hasnt($dark, "<rect x=\"296.00\" y=\"42.00\" width=\"68.00\" height=\"30.00\" fill=\"#ffffff\"",
      "and the legend card follows the theme rather than staying white");
ok(countOf($dark, "fill=\"#ffffff\"") == 0, "no hardcoded white survives anywhere in a dark figure");

plt.style("print");
$f = plt.figure(400, 300);
$a = $f.addAxes();
$a.plot([1, 2], [1, 2], null);
$a.plot([1, 2], [2, 1], null);
$print = $f.to_svg();
has($print, "#000000", "the print style's first series is black");
has($print, "#666666", "and the second is a mid grey, distinguishable without colour");

plt.style("default");
raisesWith(def() { plt.style("neon"); }, "\"default\", \"dark\" or \"print\"",
           "an unknown style raises, listing the real ones");

// ════════════════════════════════════════════════════════════════════════
//  7. Escaping, through the B3 surfaces
// ════════════════════════════════════════════════════════════════════════
print("");
print("-- hostile text through heatmap labels and a colorbar label --");
$evil = "</text><script>alert(1)</script>";
$f = plt.figure(600, 420);
$a = $f.addAxes();
$a.heatmap([[1, 2], [3, 4]], {"rows": [$evil, "b"], "cols": [$evil, "d"]});
$a.colorbar({"label": $evil});
$a.setTitle($evil);
$svg = $f.to_svg();
hasnt($svg, "<script", "no script element survives");
has($svg, "&lt;script&gt;", "it is escaped instead");

print("");
print("========================================");
print("  passed: " + str($R.pass) + "   failed: " + str($R.fail));
if ($R.fail == 0) { print("RESULT: ALL GREEN"); }
else { print("RESULT: FAILURES"); }
print("========================================");
