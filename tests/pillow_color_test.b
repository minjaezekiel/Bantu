// ════════════════════════════════════════════════════════════════════════
//  tests/pillow_color_test.b — pillow Phase P2: ImageColor.
//
//  Every assertion that matters here is DIFFERENTIAL: it compares against
//  tests/fixtures/pillow/reference/, which was produced by running Python's
//  Pillow (tools/gen_pillow_reference.py), not by reading its docs. That is
//  the only way "the same results as Pillow" means anything.
//
//  Run:  bantu run tests/pillow_color_test.b
// ════════════════════════════════════════════════════════════════════════

include "./pillow/color.b" as color;

$R = {"p": 0, "f": 0};
def eq($got, $want, $name) {
    if (str($got) == str($want)) { $R.p = $R.p + 1; }
    else {
        $R.f = $R.f + 1;
        print("  FAIL  " + $name);
        print("          got:  " + str($got));
        print("          want: " + str($want));
    }
}
def throws($fn, $name) {
    $caught = false;
    try { $fn(); } catch ($e) { $caught = true; }
    if ($caught) { $R.p = $R.p + 1; }
    else { $R.f = $R.f + 1; print("  FAIL  (expected an error) " + $name); }
}

$REF = "tests/fixtures/pillow/reference/";

// _rows($file) -> the non-comment lines.
def _rows($file) {
    $all = split(readfile($REF + $file), "\n");
    $out = [];
    $i = 0;
    while ($i < len($all)) {
        $L = trim($all[$i]);
        if (len($L) > 0 && $L[0] != "#") { push($out, $all[$i]); }
        $i = $i + 1;
    }
    return $out;
}

// ─── all 148 named colours, against Pillow's colormap ───────────────────
print("── the 148 named colours ──");
$rows = _rows("colors.txt");
$bad = 0;
$i = 0;
while ($i < len($rows)) {
    $f = split(trim($rows[$i]), " ");
    $want = [num($f[1]), num($f[2]), num($f[3])];
    $got = color.getrgb($f[0]);
    if (str($got) != str($want)) {
        $bad = $bad + 1;
        if ($bad < 6) { print("  FAIL  " + $f[0] + " got " + str($got) + " want " + str($want)); }
    }
    $i = $i + 1;
}
eq(len($rows), 148, "the reference has 148 names");
eq($bad, 0, "all 148 named colours match Pillow");
eq(len(color.names()), 148, "colormap() holds 148 names");

// the name table is case-insensitive, as Pillow lowercases the spec
eq(color.getrgb("RED"), [255, 0, 0], "an upper-case name works");
eq(color.getrgb("DarkSlateGray"), color.getrgb("darkslategray"), "mixed case equals lower case");
eq(color.getrgb("rebeccapurple"), [102, 51, 153], "rebeccapurple, the CSS4 addition");

// ─── every colour spec form, against Pillow ─────────────────────────────
print("── getrgb spec forms ──");
$rows = _rows("color_specs.txt");
$bad = 0;
$checked = 0;
$i = 0;
while ($i < len($rows)) {
    $f = split($rows[$i], "\t");
    $spec = $f[0];
    $kind = $f[1];
    $got = null;
    $threw = false;
    try { $got = color.getrgb($spec); } catch ($e) { $threw = true; }
    if ($kind == "ERROR") {
        if (!$threw) {
            $bad = $bad + 1;
            print("  FAIL  '" + $spec + "' should be an error, got " + str($got));
        }
    } else {
        $parts = split($f[2], ",");
        $want = [];
        $k = 0;
        while ($k < len($parts)) { push($want, num($parts[$k])); $k = $k + 1; }
        if ($threw) {
            $bad = $bad + 1;
            print("  FAIL  '" + $spec + "' threw, want " + str($want));
        } else {
            if (str($got) != str($want)) {
                $bad = $bad + 1;
                print("  FAIL  '" + $spec + "' got " + str($got) + " want " + str($want));
            }
        }
    }
    $checked = $checked + 1;
    $i = $i + 1;
}
eq($bad, 0, "all " + str($checked) + " spec forms match Pillow");

// ─── getcolor across every mode, against Pillow ─────────────────────────
print("── getcolor(spec, mode) over every mode ──");
$rows = _rows("getcolor.txt");
$bad = 0;
$checked = 0;
$i = 0;
while ($i < len($rows)) {
    $f = split($rows[$i], "\t");
    $m = $f[0];
    $spec = $f[1];
    $kind = $f[2];
    $got = null;
    $threw = false;
    try { $got = color.getcolor($spec, $m); } catch ($e) { $threw = true; }
    if ($kind == "ERROR") {
        if (!$threw) { $bad = $bad + 1; print("  FAIL  getcolor('" + $spec + "','" + $m + "') should error"); }
    } else {
        $parts = split($f[3], ",");
        $want = [];
        $k = 0;
        while ($k < len($parts)) { push($want, num($parts[$k])); $k = $k + 1; }
        // Pillow returns a bare number for a single-band result.
        if (len($want) == 1) { $wantV = $want[0]; } else { $wantV = $want; }
        if ($threw) { $bad = $bad + 1; print("  FAIL  getcolor('" + $spec + "','" + $m + "') threw, want " + str($wantV)); }
        else {
            if (str($got) != str($wantV)) {
                $bad = $bad + 1;
                if ($bad < 8) {
                    print("  FAIL  getcolor('" + $spec + "','" + $m + "') got " + str($got) + " want " + str($wantV));
                }
            }
        }
    }
    $checked = $checked + 1;
    $i = $i + 1;
}
eq($bad, 0, "all " + str($checked) + " getcolor results match Pillow");

// ─── the quirks, called out so a future change cannot quietly drop them ─
print("── Pillow's quirks, asserted on purpose ──");

// getcolor switches on BASEMODE, so CMYK gets three values, not four.
eq(color.getcolor("red", "CMYK"), [255, 0, 0], "getcolor to CMYK returns 3 values");
eq(color.getcolor("red", "P"), [255, 0, 0], "getcolor to P returns the RGB triple");
// mode "1" is not thresholded
eq(color.getcolor("red", "1"), 76, "getcolor to '1' returns the grey level, not 0/255");
eq(color.getcolor("lime", "1"), 150, "getcolor('lime','1') is 150");
// premultiplied targets multiply by alpha
eq(color.getcolor("rgba(255,255,255,128)", "RGBa"), [128, 128, 128, 128], "RGBa premultiplies");
eq(color.getcolor("rgba(255,255,255,128)", "RGBA"), [255, 255, 255, 128], "RGBA does not");
// the spec is not trimmed
throws(def() { return color.getrgb("  red  "); }, "a padded name is an error, as in Pillow");
// transparent is not a colour
throws(def() { return color.getrgb("transparent"); }, "'transparent' is not a colour name");
// no 9- or 12-digit hex
throws(def() { return color.getrgb("#112233445566"); }, "#rrrrggggbbbb is not accepted");
throws(def() { return color.getrgb("#112233445"); }, "a 9-digit hex is not accepted");
// integers only in rgb()
throws(def() { return color.getrgb("rgb(1.5,0,0)"); }, "rgb() takes integers only");
throws(def() { return color.getrgb("rgb(-1,0,0)"); }, "rgb() rejects a negative");
// over-long specs
$long = []; $i = 0;
while ($i < 120) { push($long, "a"); $i = $i + 1; }
throws(def() { return color.getrgb(join($long, "")); }, "a spec over 100 characters is refused");

// ─── failure modes ──────────────────────────────────────────────────────
print("── failure modes ──");
throws(def() { return color.getrgb("not a colour"); }, "nonsense");
throws(def() { return color.getrgb(""); }, "an empty spec");
throws(def() { return color.getrgb("#"); }, "a bare #");
throws(def() { return color.getrgb("#gg0000"); }, "non-hex digits");
throws(def() { return color.getrgb("rgb(1,2)"); }, "too few rgb arguments");
throws(def() { return color.getrgb("rgb(1,2,3,4)"); }, "too many rgb arguments");
throws(def() { return color.getrgb("rgba(1,2,3)"); }, "too few rgba arguments");
throws(def() { return color.getrgb("hsl(0,100,50)"); }, "hsl without percent signs");
throws(def() { return color.getrgb(42); }, "a number is not a colour string");
throws(def() { return color.getrgb(null); }, "null is not a colour string");

// ─── colorsys, exercised directly ───────────────────────────────────────
print("── the colorsys ports ──");
eq(color.rgb_to_hsv(1.0, 0.0, 0.0), [0.0, 1.0, 1.0], "rgb_to_hsv red");
eq(color.rgb_to_hsv(0.0, 0.0, 0.0), [0.0, 0.0, 0.0], "rgb_to_hsv black");
eq(color.rgb_to_hsv(0.5, 0.5, 0.5), [0.0, 0.0, 0.5], "rgb_to_hsv grey");
eq(color.hsv_to_rgb(0.0, 1.0, 1.0), [1.0, 0.0, 0.0], "hsv_to_rgb red");
eq(color.hls_to_rgb(0.0, 0.5, 1.0), [1.0, 0.0, 0.0], "hls_to_rgb red");
eq(color.hls_to_rgb(0.0, 1.0, 0.0), [1.0, 1.0, 1.0], "hls_to_rgb white");
// hsv -> rgb -> hsv round trips over the hue circle
$bad = 0; $h = 0;
while ($h < 360) {
    $f = color.hsv_to_rgb($h / 360.0, 1.0, 1.0);
    $back = color.rgb_to_hsv($f[0], $f[1], $f[2]);
    if (floor($back[0] * 360 + 0.5) % 360 != $h) { $bad = $bad + 1; }
    $h = $h + 1;
}
eq($bad, 0, "hsv round-trips at all 360 hues");

print("");
print("checks: " + str($R.p + $R.f) + "  passed: " + str($R.p) + "  failed: " + str($R.f));
if ($R.f == 0) { print("RESULT: ALL GREEN"); }
else { print("RESULT: FAILED"); }
