// ════════════════════════════════════════════════════════════════════════
//  bplot_png_corpus_test.b — the byte-identity gate (B6f).
//
//  The raster backend's contract is that the same figure is the same FILE on
//  Linux, macOS and Windows (docs/bplot-raster-architecture.md §2). This
//  renders a fixed corpus and compares each PNG's SHA-256 with a literal
//  committed below. CI runs it in all three jobs, so the three platforms are
//  compared through these literals.
//
//  IF THIS FAILS
//    * on ONE platform only: a byte-identity defect -- something platform-
//      dependent reached a pixel or a compressed byte. Find it; do not
//      re-record the hash.
//    * on EVERY platform, after a change meant to alter output: re-record
//      with   BPLOT_CORPUS_PRINT=1 bantu run tests/bplot_png_corpus_test.b
//      and say in the commit why the pictures changed.
//
//  Run:  bantu run tests/bplot_png_corpus_test.b
// ════════════════════════════════════════════════════════════════════════

include "./bplot/bplot.b" as plt;

$R = {"pass": 0, "fail": 0};
$PRINT = env("BPLOT_CORPUS_PRINT") == "1";

// The recorded hashes. Recorded on macOS; matched on Linux and Windows by CI.
$WANT = {
    "canvas-shapes":     "9d7d15632e2cd4cca3393fa74bf8fe8eb64a3e74626d0ce3778ef6f2c33b70c0",
    "canvas-strokes":    "709d550c913191538379a6830f8dc8f65a0de8fbca02f782d552be334b093999",
    "canvas-text":       "9e12c7ede230da432d5e7f9a40f9cee6dccee004f16fbb420cb7c3ff07bd20f0",
    "canvas-far":        "777bb133daedbc341f8b2e7f8fa7b024ec50059d475743a4ee417771c8c8c825",
    "figure-line-96":    "d7b95a24c4f08c1d2244dd7d809550418db5bae29dff100379ae9e96e5bad03e",
    "figure-line-150":   "e980a651ac82f9fbd64d0a00dcd2b05e04eb43cf7d589d67bcd24f218b6aad4d",
    "figure-charts-96":  "22d1f841ee7a70d6b284a6486d13ba25f6f2ef3248fe9f6a569c19a1ef6a2126",
    "figure-grid-120":   "f4df9ea3524a76eea136b223924cc1c854d6ef65d941f944c1a12d929c637f52"
};

def check($name, $png) {
    $got = native_sha256($png);
    if ($PRINT) { print("    \"" + $name + "\": \"" + $got + "\","); return null; }
    if ($got == $WANT[$name]) { $R["pass"] = $R["pass"] + 1; print("  ok    " + $name); }
    else {
        $R["fail"] = $R["fail"] + 1;
        print("  FAIL  " + $name + " -- sha256 " + $got + ", recorded " + str($WANT[$name]));
    }
    return null;
}

if (!has_native("raster")) {
    print("  --    this interpreter has no raster backend; skipping");
    print("RESULT: ALL GREEN");
} else {

// ── The canvas, primitive by primitive ────────────────────────────────────
$c = bp_canvas_new(160, 120, 150, "#f7f7f7");
bp_fill_rect($c, 3.25, 4.5, 70.3, 40.7, "#1f77b4", null);
bp_fill_rect($c, 50, 30, 60, 50, "#d62728", 0.55);
bp_canvas_clip($c, 10, 10, 120, 90);
bp_fill_polygon($c, [20, 100, 80, 15, 140, 110, 60, 60], "#2ca02c", 0.7);
bp_canvas_clip($c, null, null, null, null);
bp_fill_path($c, "M 120 60 L 120 20 A 40 40 0 0 1 150 60 Z", "#9467bd", null);
bp_fill_path($c, "M 30 90 a 12 12 0 1 0 24 0 a 12 12 0 1 0 -24 0 Z", "#ff7f0e", 0.8);
check("canvas-shapes", bp_png($c));

$c = bp_canvas_new(160, 120, 96, "#ffffff");
bp_stroke_polyline($c, [10, 10, 60, 100, 110, 15, 150, 90], "#1f77b4", 3.5, null, null, true);
bp_stroke_polyline($c, [10, 60, 150, 60], "#d62728", 1.25, 0.6, "6,3", false);
bp_stroke_polyline($c, [20, 110, 140, 105], "#000000", 0.75, null, "1.5,2.5", true);
bp_stroke_path($c, "M 5 5 h 150 v 110 h -150 Z M 40 40 L 120 80", "#2ca02c", 1, null);
check("canvas-strokes", bp_png($c));

$c = bp_canvas_new(220, 140, 192, "#ffffff");
bp_text($c, 8, 24, "Temperature (°C) ± 0.5", 14, "#000000", null, null, null);
bp_text($c, 110, 50, "centred — 12×3 ≤ ∞", 11, "#1f77b4", "middle", null, null);
bp_text($c, 212, 72, "Ångström çà ÿ € ™", 11, "#d62728", "end", null, 0.7);
bp_text($c, 16, 134, "rotated -90", 10, "#000000", null, 0 - 90, null);
bp_text($c, 60, 130, "rotated 30°", 10, "#9467bd", null, 30, null);
bp_text($c, 150, 130, "bad " + chr(255) + " byte", 10, "#000000", null, 0 - 45, null);
check("canvas-text", bp_png($c));

$c = bp_canvas_new(100, 100, 300, "#ffffff");
bp_stroke_polyline($c, [0, 0, 100000000, 50000000], "#ff0000", 2, null, "4,2", false);
bp_fill_polygon($c, [0 - 100000000, 90, 100000000, 95, 0, 100000000], "#00aa00", 0.5);
bp_fill_path($c, "M -99999999 10 L 99999999 10 L 99999999 20 Z", "#0000ff", null);
check("canvas-far", bp_png($c));

// ── Whole figures, through bplot ──────────────────────────────────────────
$f = plt.figure(480, 300);
$a = $f.addAxes();
$a.plot([1, 2, 3, 4, 5, 6], [2.5, 4, 9, 3, 7.25, 6], {"label": "measured"});
$a.plot([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], {"label": "expected", "color": "orange"});
$a.setTitle("Measured against expected");
$a.setXLabel("trial");
$a.setYLabel("score");
$a.setGrid(true);
$a.setLegend(true);
$f.tight_layout(true);
check("figure-line-96", $f.to_png(96));
check("figure-line-150", $f.to_png(150));

$months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"];
$f = plt.figure(900, 600);
$b = $f.subplot(2, 2, 1);
$b.bar($months, [66, 61, 118, 290, 197, 33], {"color": "#1f77b4"});
$t = $f.twinx($b);
$t.plot($months, [28, 28, 28, 27, 26, 24], {"color": "#d62728", "width": 2});
$t.setYLabel("temperature (C)");
$s = $f.subplot(2, 2, 2);
$s.scatter([1, 2, 3, 4, 5, 6, 7], [3, 1, 4, 1, 5, 9, 2], {"size": 4});
$s.errorbar([1, 2, 3], [2, 3, 2.5], {"yerr": [0.5, 0.25, 0.75]});
$h = $f.subplot(2, 2, 3);
$h.hist([1, 2, 2, 3, 3, 3, 4, 4, 5, 7, 8, 8, 9], {"bins": 5});
$p = $f.subplot(2, 2, 4);
$p.pie([48, 27, 15, 10], {"labels": ["direct", "search", "social", "other"], "percent": true});
$f.tight_layout(true);
check("figure-charts-96", $f.to_png(96));

$z = [];
$i = 0;
while ($i < 12) {
    $row = [];
    $j = 0;
    while ($j < 16) { push($row, ($i * 7 + $j * 3) % 11 - $j / 4); $j = $j + 1; }
    push($z, $row);
    $i = $i + 1;
}
$z[4][5] = NAN;
$f = plt.figure(520, 360);
$g = $f.addAxes();
$g.imshow($z, {"cmap": "viridis"});
$g.colorbar({"label": "value"});
$g.setTitle("a grid, with a hole");
$f.tight_layout(true);
check("figure-grid-120", $f.to_png(120));

if (!$PRINT) {
    print("");
    print("Passed: " + str($R["pass"]) + "   Failed: " + str($R["fail"]));
    if ($R["fail"] == 0) { print("RESULT: ALL GREEN"); } else { print("RESULT: FAILURES"); }
}
}
