// ════════════════════════════════════════════════════════════════════════
//  bplot_raster_test.b — the canvas and the PNG encoder (B6b).
//
//  Every assertion here names an exact value, because the phase's gate is
//  byte-identical output on three platforms and "about grey" cannot be
//  compared. A pixel half covered by black over white is #7f7f7f: coverage
//  128/256, alpha 128, and (0*128 + 255*127 + 127) / 255 = 127.
//
//  What this file cannot check is the encoder's agreement with the rest of
//  the world -- that is tests/bplot_png_test.sh, where Python's zlib and
//  Pillow decode what Bantu wrote.
//
//  Design: docs/bplot-raster-architecture.md
// ════════════════════════════════════════════════════════════════════════

$R = {"pass": 0, "fail": 0};
def ok($cond, $what) {
    if ($cond) { $R["pass"] = $R["pass"] + 1; }
    else { $R["fail"] = $R["fail"] + 1; print("  FAIL: " + $what); }
}
def eq($got, $want, $what) {
    if ($got == $want) { $R["pass"] = $R["pass"] + 1; }
    else { $R["fail"] = $R["fail"] + 1; print("  FAIL: " + $what + " -- got " + str($got) + ", wanted " + str($want)); }
}
def raises($fn, $fragment, $what) {
    $msg = null;
    try { $fn(); } catch ($e) { $msg = str($e); }
    if ($msg == null) { ok(false, $what + " (did not raise)"); return null; }
    if (!contains($msg, $fragment)) { print("    message was: " + $msg); }
    ok(contains($msg, $fragment), $what);
    return null;
}

if (!has_native("raster")) {
    print("  --    this interpreter has no raster backend; skipping");
    print("RESULT: ALL GREEN");
} else {

print("── a canvas, and what its pixels are ─────────────────────────────");

$c = bp_canvas_new(10, 10, null, null);
$info = bp_canvas_info($c);
eq($info["width"], 10, "a 10x10 canvas is 10 device pixels wide");
eq($info["height"], 10, "and 10 tall");
eq($info["dpi"], 96, "at 96 dpi, the default: one device pixel per user unit");
eq(bp_canvas_pixel($c, 0, 0), "#ffffff", "the default background is white");

$c2 = bp_canvas_new(4, 4, null, "#1f77b4");
eq(bp_canvas_pixel($c2, 2, 2), "#1f77b4", "a background colour is the colour it was given");

bp_fill_rect($c, 0, 0, 10, 10, "#000000", null);
eq(bp_canvas_pixel($c, 5, 5), "#000000", "an opaque fill covers the canvas");

print("── coverage and blending are exact ───────────────────────────────");

$f = bp_canvas_new(4, 4, null, "#ffffff");
// Half of the second column: coverage 128/256, so alpha 128.
bp_fill_rect($f, 0, 0, 1.5, 4, "#000000", null);
eq(bp_canvas_pixel($f, 0, 0), "#000000", "a fully covered pixel is the fill colour");
eq(bp_canvas_pixel($f, 1, 0), "#7f7f7f", "a half-covered pixel is exactly #7f7f7f");
eq(bp_canvas_pixel($f, 2, 0), "#ffffff", "and the pixel beyond it is untouched");

$g = bp_canvas_new(4, 4, null, "#ffffff");
bp_fill_rect($g, 0, 0, 4, 4, "#000000", 0.5);
eq(bp_canvas_pixel($g, 1, 1), "#7f7f7f", "half opacity over white is the same #7f7f7f");

$q = bp_canvas_new(4, 4, null, "#ffffff");
bp_fill_rect($q, 0, 0, 0.25, 4, "#000000", null);
eq(bp_canvas_pixel($q, 0, 0), "#bfbfbf", "a quarter-covered pixel is #bfbfbf");

$z = bp_canvas_new(4, 4, null, "#ffffff");
bp_fill_rect($z, 0, 0, 4, 4, "#000000", 0);
eq(bp_canvas_pixel($z, 1, 1), "#ffffff", "zero opacity draws nothing");
bp_fill_rect($z, 0, 0, 4, 4, "none", null);
eq(bp_canvas_pixel($z, 1, 1), "#ffffff", "and neither does \"none\"");

print("── degenerate rectangles draw nothing, and raise nothing ─────────");

$d = bp_canvas_new(6, 6, null, "#ffffff");
bp_fill_rect($d, 0, 0, 0, 6, "#000000", null);
bp_fill_rect($d, 0, 0, 6, 0, "#000000", null);
bp_fill_rect($d, 3, 3, 0 - 2, 0 - 2, "#000000", null);
bp_fill_rect($d, 100, 100, 5, 5, "#000000", null);
bp_fill_rect($d, 0 - 50, 0 - 50, 5, 5, "#000000", null);
eq(bp_canvas_pixel($d, 3, 3), "#ffffff", "zero, negative and off-canvas rectangles change nothing");
// From (-2, -2) to (0.5, 0.5): pixel (0, 0) is half covered in x AND half in
// y, so a quarter overall -- alpha 64, not 128.
bp_fill_rect($d, 0 - 2, 0 - 2, 2.5, 2.5, "#000000", null);
eq(bp_canvas_pixel($d, 0, 0), "#bfbfbf", "a rectangle mostly off the canvas draws exactly the corner that is on it");

print("── the clip rectangle ────────────────────────────────────────────");

$k = bp_canvas_new(8, 8, null, "#ffffff");
bp_canvas_clip($k, 2, 2, 4, 4);
bp_fill_rect($k, 0, 0, 8, 8, "#d62728", null);
eq(bp_canvas_pixel($k, 0, 0), "#ffffff", "outside the clip is untouched");
eq(bp_canvas_pixel($k, 3, 3), "#d62728", "inside it is filled");
eq(bp_canvas_pixel($k, 6, 6), "#ffffff", "and the far edge of the clip is exclusive");
bp_canvas_clip($k, null, null, null, null);
bp_fill_rect($k, 0, 0, 8, 8, "#2ca02c", null);
eq(bp_canvas_pixel($k, 0, 0), "#2ca02c", "resetting the clip restores the whole canvas");

print("── dpi scales the pixels, not the coordinates ────────────────────");

$hi = bp_canvas_new(10, 5, 300, "#ffffff");
$hinfo = bp_canvas_info($hi);
eq($hinfo["width"], 31, "10 units at 300 dpi is 31 pixels (10 * 300 / 96, rounded)");
eq($hinfo["height"], 16, "and 5 units is 16");
bp_fill_rect($hi, 0, 0, 5, 5, "#000000", null);
eq(bp_canvas_pixel($hi, 14, 8), "#000000", "a rectangle in user units lands at the scaled pixels");
eq(bp_canvas_pixel($hi, 20, 8), "#ffffff", "and stops where the user units say");

print("── the pieces of the encoder, against known answers ──────────────");

eq(bp_crc32("123456789"), 3421780262, "CRC-32 of \"123456789\" is 0xCBF43926");
eq(bp_crc32(""), 0, "CRC-32 of nothing is 0");
eq(bp_adler32("Wikipedia"), 300286872, "Adler-32 of \"Wikipedia\" is 0x11E60398");
eq(bp_adler32(""), 1, "Adler-32 of nothing is 1");

$z1 = bp_zlib("");
ok(len($z1) >= 6, "an empty input still makes a valid zlib stream");
eq(ord($z1[0]), 120, "which starts with the zlib header byte 0x78");
eq(ord($z1[1]), 156, "and 0x9c");
$flat = "";
$i = 0;
while ($i < 2000) { $flat = $flat + "aaaaaaaaaa"; $i = $i + 1; }
$z2 = bp_zlib($flat);
ok(len($z2) < len($flat) / 100, "20,000 identical bytes compress by more than 100x");

print("── the PNG itself ────────────────────────────────────────────────");

$p = bp_canvas_new(20, 10, null, "#1f77b4");
$png = bp_png($p);
eq(ord($png[0]), 137, "the signature begins 0x89");
eq(substr($png, 1, 3), "PNG", "then PNG");
eq(ord($png[2 + 2]), 13, "then CR");
eq(ord($png[5]), 10, "LF");
eq(ord($png[6]), 26, "0x1A");
eq(ord($png[7]), 10, "and LF -- the bytes a text-mode write would have mangled");
ok(contains($png, "IHDR"), "there is an IHDR chunk");
ok(contains($png, "pHYs"), "a pHYs chunk, so the dpi survives into the file");
ok(contains($png, "IDAT"), "an IDAT chunk");
ok(contains($png, "IEND"), "and an IEND chunk");

// Rendering the same canvas twice must give the same bytes: no timestamp, no
// hash order, nothing that varies. This is the property the three-platform
// gate rests on, checked here within one process.
ok(bp_png($p) == $png, "rendering the same canvas twice is byte-identical");

$raw = bp_canvas_raw($p);
eq(len($raw), 20 * 10 * 3, "the raw pixels are three bytes each");
eq(ord($raw[0]), 31, "and are the background colour: 0x1f");
eq(ord($raw[1]), 119, "0x77");
eq(ord($raw[2]), 180, "0xb4");

// Through the binary file modes of B6a, which is what they were fixed for.
$path = "/tmp/bantu_raster_test_" + str(clock()) + ".png";
$n = bp_png_save($p, $path);
eq($n, len($png), "bp_png_save writes the same bytes bp_png returns");
ok(readfile($path, "rb") == $png, "and the file holds them, byte for byte");

print("── polygons ──────────────────────────────────────────────────────");

// A 20x10 rectangle as a polygon: every interior pixel is the fill, and the
// area is exact because the edges land on pixel boundaries.
$pg = bp_canvas_new(40, 20, null, "#ffffff");
bp_fill_polygon($pg, [5, 5, 25, 5, 25, 15, 5, 15], "#1f77b4", null);
eq(bp_canvas_pixel($pg, 15, 10), "#1f77b4", "a polygon fills its inside");
eq(bp_canvas_pixel($pg, 4, 10), "#ffffff", "and nothing outside it");
eq(bp_canvas_pixel($pg, 25, 10), "#ffffff", "its far edge is exclusive, like a rectangle's");

// A triangle, whose diagonal must be anti-aliased rather than stepped.
$tri = bp_canvas_new(20, 20, null, "#ffffff");
bp_fill_polygon($tri, [0, 0, 20, 0, 0, 20], "#000000", null);
eq(bp_canvas_pixel($tri, 2, 2), "#000000", "inside the triangle");
eq(bp_canvas_pixel($tri, 17, 17), "#ffffff", "outside it");
// Probe a pixel the hypotenuse actually crosses. The edge is the line
// x + y = 20, so pixel (10, 10) touches it only at a corner and is correctly
// empty; (9, 10) is the one the edge cuts in half.
eq(bp_canvas_pixel($tri, 10, 10), "#ffffff", "a pixel the edge only touches at a corner stays empty");
$edgePix = bp_canvas_pixel($tri, 9, 10);
ok($edgePix != "#000000" && $edgePix != "#ffffff",
   "and a pixel the edge cuts is a partial value, not a staircase step");

// Winding: a polygon drawn the other way round fills the same pixels.
$cw = bp_canvas_new(20, 20, null, "#ffffff");
bp_fill_polygon($cw, [2, 2, 12, 2, 12, 12, 2, 12], "#000000", null);
$ccw = bp_canvas_new(20, 20, null, "#ffffff");
bp_fill_polygon($ccw, [2, 12, 12, 12, 12, 2, 2, 2], "#000000", null);
ok(bp_canvas_raw($cw) == bp_canvas_raw($ccw), "the direction a polygon is written in does not change it");

bp_fill_polygon($pg, [1, 1, 2, 2], "#000000", null);
eq(bp_canvas_pixel($pg, 1, 1), "#ffffff", "two points are not a polygon, and draw nothing");

print("── strokes ───────────────────────────────────────────────────────");

$ln = bp_canvas_new(40, 20, null, "#ffffff");
bp_stroke_polyline($ln, [5, 10, 35, 10], "#000000", 4, null, null, null);
eq(bp_canvas_pixel($ln, 20, 10), "#000000", "a stroke covers its centre");
eq(bp_canvas_pixel($ln, 20, 7), "#ffffff", "and stops at its half-width");
eq(bp_canvas_pixel($ln, 20, 9), "#000000", "which is 2 px above the centre line");

// The defect the draft's area check caught: a join between segments that wind
// opposite ways used to cancel to nothing, leaving a hole.
$zz = bp_canvas_new(60, 40, null, "#ffffff");
bp_stroke_polyline($zz, [10, 10, 50, 10, 10, 30], "#000000", 8, null, null, null);
eq(bp_canvas_pixel($zz, 50, 10), "#000000", "the join at a reversal is filled, not holed");
eq(bp_canvas_pixel($zz, 30, 10), "#000000", "and so is the segment either side of it");

// Round caps stick out past the end; butt caps do not.
$rc = bp_canvas_new(30, 20, null, "#ffffff");
bp_stroke_polyline($rc, [10, 10, 20, 10], "#000000", 8, null, null, true);
$bc = bp_canvas_new(30, 20, null, "#ffffff");
bp_stroke_polyline($bc, [10, 10, 20, 10], "#000000", 8, null, null, false);
eq(bp_canvas_pixel($rc, 7, 10), "#000000", "a round cap reaches beyond the end point");
eq(bp_canvas_pixel($bc, 7, 10), "#ffffff", "a butt cap does not");

// A translucent stroke must not darken where its pieces overlap.
$tr = bp_canvas_new(40, 40, null, "#ffffff");
bp_stroke_polyline($tr, [5, 20, 20, 20, 35, 20], "#000000", 10, 0.5, null, true);
eq(bp_canvas_pixel($tr, 20, 20), bp_canvas_pixel($tr, 10, 20),
   "a translucent stroke is one mask: the join is no darker than the line");

bp_stroke_polyline($tr, [5, 5, 35, 5], "#000000", 0, null, null, null);
eq(bp_canvas_pixel($tr, 20, 5), "#ffffff", "a zero-width stroke draws nothing");

print("── dashes ────────────────────────────────────────────────────────");

$ds = bp_canvas_new(40, 10, null, "#ffffff");
bp_stroke_polyline($ds, [0, 5, 40, 5], "#000000", 4, null, "6,6", false);
eq(bp_canvas_pixel($ds, 2, 5), "#000000", "a dash starts drawn");
eq(bp_canvas_pixel($ds, 8, 5), "#ffffff", "then leaves a gap");
eq(bp_canvas_pixel($ds, 14, 5), "#000000", "then draws again");
raises(def() { bp_stroke_polyline($ds, [0, 5, 9, 5], "#000000", 2, null, "bad", null); },
       "dash must be numbers", "a dash that is not numbers raises");
raises(def() { bp_stroke_polyline($ds, [0, 5, 9, 5], "#000000", 2, null, "0,4", null); },
       "positive", "a zero dash length raises");

print("── paths ─────────────────────────────────────────────────────────");

// The three shapes bplot actually emits: a cell (relative h/v), a pie slice
// (absolute arc) and a scatter circle (two relative half-arcs).
$pa = bp_canvas_new(40, 40, null, "#ffffff");
bp_fill_path($pa, "M5 5h20v10h-20Z", "#2ca02c", null);
eq(bp_canvas_pixel($pa, 15, 10), "#2ca02c", "a cell path, drawn with relative h and v");
eq(bp_canvas_pixel($pa, 26, 10), "#ffffff", "and it ends where it says");

$pie = bp_canvas_new(60, 60, null, "#ffffff");
bp_fill_path($pie, "M 30 30 L 55 30 A 25 25 0 0 1 30 55 Z", "#d62728", null);
eq(bp_canvas_pixel($pie, 45, 40), "#d62728", "a quarter-circle slice is filled inside");
eq(bp_canvas_pixel($pie, 12, 12), "#ffffff", "and empty in the opposite quarter");
eq(bp_canvas_pixel($pie, 45, 20), "#ffffff", "and empty above it");

$sc = bp_canvas_new(40, 40, null, "#ffffff");
bp_fill_path($sc, "M14.00,20.00a6.00,6.00 0 1,0 12.00,0a6.00,6.00 0 1,0 -12.00,0", "#9467bd", null);
eq(bp_canvas_pixel($sc, 20, 20), "#9467bd", "a scatter circle, as two relative half-arcs");
eq(bp_canvas_pixel($sc, 20, 13), "#ffffff", "with nothing beyond its radius");
eq(bp_canvas_pixel($sc, 20, 15), "#9467bd", "and everything inside it");

$multi = bp_canvas_new(40, 20, null, "#ffffff");
bp_fill_path($multi, "M2 2h8v8h-8ZM20 2h8v8h-8Z", "#000000", null);
eq(bp_canvas_pixel($multi, 5, 5), "#000000", "a path with two subpaths fills the first");
eq(bp_canvas_pixel($multi, 23, 5), "#000000", "and the second");
eq(bp_canvas_pixel($multi, 15, 5), "#ffffff", "and not the gap between them");

raises(def() { bp_fill_path($pa, "M0 0 Q 5 5 10 0 Z", "#000000", null); },
       "unsupported command 'Q'", "an unsupported path command raises, naming it");
raises(def() { bp_fill_path($pa, "M0 0 L", "#000000", null); },
       "expected a number", "a truncated path raises");
raises(def() { bp_fill_path($pa, "10 10 20 20", "#000000", null); },
       "expected a command letter", "a path with no command raises");
raises(def() { bp_fill_polygon($pa, [1, 2, 3], "#000000", null); },
       "pairs", "an odd number of coordinates raises");
raises(def() { bp_fill_polygon($pa, [1, "two", 3, 4], "#000000", null); },
       "must be a number", "a non-numeric coordinate raises");

print("── hostile and impossible arguments ──────────────────────────────");

raises(def() { bp_canvas_new(0, 10, null, null); }, "positive", "a zero width raises");
raises(def() { bp_canvas_new(10, 0 - 5, null, null); }, "positive", "a negative height raises");
raises(def() { bp_canvas_new(NAN, 10, null, null); }, "finite", "a NaN size raises");
raises(def() { bp_canvas_new(100000, 10, null, null); }, "32767", "too wide raises, naming the limit");
raises(def() { bp_canvas_new(20000, 20000, null, null); }, "pixels", "too many pixels raises, naming the limit");
raises(def() { bp_canvas_new(10, 10, 0, null); }, "dpi", "dpi 0 raises");
raises(def() { bp_canvas_new(10, 10, 5000, null); }, "dpi", "dpi above the cap raises");
raises(def() { bp_canvas_new(10, 10, 96.5, null); }, "whole number", "a fractional dpi raises");
raises(def() { bp_canvas_new(10, 10, null, "blue"); }, "#rrggbb", "a colour name raises -- the backend takes hex");
raises(def() { bp_canvas_new(10, 10, null, "#12345"); }, "#rrggbb", "a short hex colour raises");
raises(def() { bp_canvas_new(10, 10, null, "#gggggg"); }, "#rrggbb", "a non-hex colour raises");
raises(def() { bp_canvas_new(10, 10, null, "none"); }, "background", "a \"none\" background raises, saying why");
raises(def() { bp_fill_rect("not a canvas", 0, 0, 1, 1, "#000000", null); }, "expected a canvas", "a non-canvas raises");
raises(def() { bp_fill_rect($p, INF, 0, 1, 1, "#000000", null); }, "finite", "an infinite coordinate raises");
raises(def() { bp_canvas_pixel($p, 100, 0); }, "outside a canvas", "reading a pixel outside the canvas raises, naming the size");
raises(def() { bp_canvas_pixel($p, 0 - 1, 0); }, "outside a canvas", "and a negative one too");
raises(def() { bp_png_save($p, ""); }, "file path", "saving with no path raises");
raises(def() { bp_zlib(7); }, "string of bytes", "compressing a number raises");

// A canvas at the very edge of the caps still works.
$edge = bp_canvas_new(32767, 8, null, "#ffffff");
eq(bp_canvas_info($edge)["width"], 32767, "a canvas exactly at the side limit is allowed");

print("");
print("Passed: " + str($R["pass"]) + "   Failed: " + str($R["fail"]));
if ($R["fail"] == 0) { print("RESULT: ALL GREEN"); } else { print("RESULT: FAILURES"); }
}
