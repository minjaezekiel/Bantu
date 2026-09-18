// ════════════════════════════════════════════════════════════════════════
//  gen_font_tables.b — the embedded font bplot's raster backend draws text
//  with.
//
//      FONT=/path/to/DejaVuSans.ttf bantu run scripts/gen_font_tables.b
//
//  writes bantu-src/compiler/src/raster_font.hpp (set OUT=<path> to write
//  elsewhere). Authoring-time tooling: it runs once, by hand, and its OUTPUT
//  is checked in. Nothing in the build or at run time runs it. The font ships
//  with matplotlib (mpl-data/fonts/ttf/) and with most Linux distributions.
//
//  WHY AN EMBEDDED TABLE AND NOT A FONT FILE
//  bplot's PNG output must be byte-identical on Linux, macOS and Windows
//  (docs/bplot-raster-architecture.md §6). System fonts differ by machine, and
//  hinting differs by renderer, so neither can appear anywhere near a pixel. A
//  font FILE parsed at run time would be portable but is a parser the
//  interpreter does not want to own. So the outlines are converted ONCE, here,
//  and embedded as integers. At run time the rasteriser walks line and
//  quadratic segments in font units, scales them by a rational, and fills them
//  through the same coverage code as everything else. No hinting, no shaping,
//  no kerning -- which is also why the result is the same on every machine.
//
//  THE READER
//  A TrueType reader for exactly what this needs: the table directory, head,
//  maxp, hhea, hmtx, loca, cmap (format 4) and glyf, simple glyphs and
//  composites (accented letters are a base glyph plus an accent, placed by an
//  x/y offset). Anything outside that -- a scaled or point-matched component,
//  cubic outlines, a font without a format-4 cmap -- raises rather than
//  guessing. Contours are walked the way fontTools' pen protocol walks them,
//  which is how the checked-in header was first produced, so the output
//  is the same file byte for byte.
//
//  WHAT IS INCLUDED
//  ASCII, Latin-1, and the symbols charts actually use, plus U+FFFD for
//  anything else. 210 glyphs and 4,105 segments: about 41 KB of integer data.
//
//  LICENCE
//  DejaVu Sans is (c) Bitstream (Bitstream Vera terms) with Arev additions
//  (c) Tavmjong Bah, both of which permit redistribution provided the notices
//  travel with the data. The licence text beside the font file is copied into
//  the generated header, so the notice is in the source that carries the
//  glyphs. The designs are not modified, so the renaming clause does not apply.
// ════════════════════════════════════════════════════════════════════════

$CODEPOINTS = [];
$cp = 32;
while ($cp < 127) { push($CODEPOINTS, $cp); $cp = $cp + 1; }      // ASCII
$cp = 160;
while ($cp < 256) { push($CODEPOINTS, $cp); $cp = $cp + 1; }      // Latin-1 supplement
each ($cp in [
    8211, 8212,                     // en dash, em dash
    8216, 8217, 8220, 8221,         // quotes
    8226, 8230,                     // bullet, ellipsis
    8722,                           // minus sign
    8364, 8482,                     // euro, trade mark
    8776, 8800, 8804, 8805, 8734,   // approx, not equal, <=, >=, infinity
    956,                            // micro
    9728,                           // sun, for a weather chart
    65533                           // the replacement character
]) { push($CODEPOINTS, $cp); }

// ── Reading big-endian integers out of the font ─────────────────────────────
// $F is the whole file as a list of bytes; every read is by absolute offset.

$FONT = env("FONT");
if ($FONT == null || $FONT == "") {
    throw "set FONT to the path of DejaVuSans.ttf, e.g. FONT=/path/DejaVuSans.ttf bantu run scripts/gen_font_tables.b";
}
$F = bytes(readfile($FONT, "rb"));

def u8($i) { return $F[$i]; }
def u16($i) { return $F[$i] * 256 + $F[$i + 1]; }
def i16($i) { $v = u16($i); if ($v >= 32768) { return $v - 65536; } return $v; }
def i8($i) { $v = $F[$i]; if ($v >= 128) { return $v - 256; } return $v; }
def u32($i) { return u16($i) * 65536 + u16($i + 2); }
def has($flags, $bit) { return band($flags, $bit) != 0; }

// The table directory: tag -> offset.
$TABLES = {};
$n = u16(4);
$i = 0;
while ($i < $n) {
    $rec = 12 + 16 * $i;
    $TABLES[frombytes([$F[$rec], $F[$rec + 1], $F[$rec + 2], $F[$rec + 3]])] = u32($rec + 8);
    $i = $i + 1;
}
def table($tag) {
    if ($TABLES[$tag] == null) { throw "the font has no '" + $tag + "' table"; }
    return $TABLES[$tag];
}

$UPM = u16(table("head") + 18);
$LONG_LOCA = i16(table("head") + 50) == 1;
$NUM_GLYPHS = u16(table("maxp") + 4);
$NUM_HMETRICS = u16(table("hhea") + 34);
$HMTX = table("hmtx");
$LOCA = table("loca");
$GLYF = table("glyf");

// {advance, left side bearing} of a glyph. Glyphs past numberOfHMetrics share
// the last advance and carry only a bearing.
def hmetric($g) {
    if ($g < $NUM_HMETRICS) { return [u16($HMTX + 4 * $g), i16($HMTX + 4 * $g + 2)]; }
    return [u16($HMTX + 4 * ($NUM_HMETRICS - 1)),
            i16($HMTX + 4 * $NUM_HMETRICS + 2 * ($g - $NUM_HMETRICS))];
}

// [offset, length] of a glyph's data in the file; length 0 is an empty glyph.
def glyphSpan($g) {
    if ($LONG_LOCA) { $a = u32($LOCA + 4 * $g); $b = u32($LOCA + 4 * $g + 4); }
    else { $a = u16($LOCA + 2 * $g) * 2; $b = u16($LOCA + 2 * $g + 2) * 2; }
    return [$GLYF + $a, $b - $a];
}

// The cmap's (3,1) format-4 subtable: codepoint -> glyph, 0 when absent.
def findCmap() {
    $c = table("cmap");
    $n = u16($c + 2);
    $i = 0;
    while ($i < $n) {
        $r = $c + 4 + 8 * $i;
        if (u16($r) == 3 && u16($r + 2) == 1) {
            $sub = $c + u32($r + 4);
            if (u16($sub) != 4) { throw "the (3,1) cmap is format " + str(u16($sub)) + ", not 4"; }
            return $sub;
        }
        $i = $i + 1;
    }
    throw "the font has no Windows Unicode BMP (3,1) cmap";
}
$CMAP = findCmap();
def glyphOf($cp) {
    $segs = u16($CMAP + 6) / 2;
    $ends = $CMAP + 14;
    $starts = $ends + 2 * $segs + 2;
    $deltas = $starts + 2 * $segs;
    $ranges = $deltas + 2 * $segs;
    $i = 0;
    while ($i < $segs) {
        if (u16($ends + 2 * $i) >= $cp) {
            $start = u16($starts + 2 * $i);
            if ($start > $cp) { return 0; }
            $delta = u16($deltas + 2 * $i);
            $ro = u16($ranges + 2 * $i);
            if ($ro == 0) { return ($cp + $delta) % 65536; }
            $g = u16($ranges + 2 * $i + $ro + 2 * ($cp - $start));
            if ($g == 0) { return 0; }
            return ($g + $delta) % 65536;
        }
        $i = $i + 1;
    }
    return 0;
}

// ── Outlines to segments ────────────────────────────────────────────────────
// Each glyph becomes a run of {kind, x1, y1, x2, y2} in one shared list:
//   0 = move, 1 = line, 2 = quadratic (one control point, one end point)

$SEGS = [];

def mid($a, $b) { return [floor(($a[0] + $b[0]) / 2), floor(($a[1] + $b[1]) / 2)]; }
def point($kind, $p) { return [$kind, $p[0], $p[1]]; }
def quad($c, $e) { return [2, $c[0], $c[1], $e[0], $e[1]]; }

// One closed contour: $pts is [[x, y], ...], $on whether each is on the curve.
// Returns the number of segments added.
def contour($pts, $on) {
    $n = len($pts);
    $first = 0 - 1;
    $i = 0;
    while ($i < $n) { if ($on[$i]) { $first = $i; break; } $i = $i + 1; }
    if ($first < 0) {
        // No on-curve point at all: every point is a control, and the curve
        // passes through the midpoints between them, cyclically.
        push($SEGS, point(0, mid($pts[$n - 1], $pts[0])));
        $i = 0;
        while ($i < $n) {
            push($SEGS, quad($pts[$i], mid($pts[$i], $pts[($i + 1) % $n])));
            $i = $i + 1;
        }
        return $n + 1;
    }
    // Start at the first on-curve point and walk back round to it. The line
    // that closes the contour is implied, so it is not emitted.
    $count = 1;
    push($SEGS, point(0, $pts[$first]));
    $offs = [];
    $k = 1;
    while ($k <= $n) {
        $j = ($first + $k) % $n;
        if (!$on[$j]) { push($offs, $pts[$j]); }
        else {
            $m = len($offs);
            if ($m == 0) {
                if ($k < $n) { push($SEGS, point(1, $pts[$j])); $count = $count + 1; }
            } else {
                // A run of off-curve points: implied on-curve points between
                // consecutive controls, then the real one to finish.
                $q = 0;
                while ($q < $m) {
                    $end = $pts[$j];
                    if ($q + 1 < $m) { $end = mid($offs[$q], $offs[$q + 1]); }
                    push($SEGS, quad($offs[$q], $end));
                    $count = $count + 1;
                    $q = $q + 1;
                }
                $offs = [];
            }
        }
        $k = $k + 1;
    }
    return $count;
}

// A glyph's outline, shifted by (dx, dy). Composites recurse into their
// components. Returns the number of segments added.
def outline($g, $dx, $dy, $top) {
    $span = glyphSpan($g);
    if ($span[1] == 0) { return 0; }
    $p = $span[0];
    $nc = i16($p);
    $count = 0;
    if ($nc < 0) {
        $p = $p + 10;
        $more = true;
        while ($more) {
            $flags = u16($p);
            $child = u16($p + 2);
            $p = $p + 4;
            if (has($flags, 1)) { $ax = i16($p); $ay = i16($p + 2); $p = $p + 4; }
            else { $ax = i8($p); $ay = i8($p + 1); $p = $p + 2; }
            if (!has($flags, 2)) { throw "glyph " + str($g) + ": a point-matched component is not supported"; }
            if (has($flags, 8) || has($flags, 64) || has($flags, 128)) {
                throw "glyph " + str($g) + ": a scaled component is not supported";
            }
            $count = $count + outline($child, $dx + $ax, $dy + $ay, false);
            $more = has($flags, 32);
        }
        return $count;
    }
    // A simple glyph. At the top level it is shifted so its left edge sits
    // at its left side bearing, as the pen protocol does.
    if ($top) { $dx = $dx + hmetric($g)[1] - i16($p + 2); }
    $ends = [];
    $i = 0;
    while ($i < $nc) { push($ends, u16($p + 10 + 2 * $i)); $i = $i + 1; }
    $npts = $ends[$nc - 1] + 1;
    $p = $p + 10 + 2 * $nc;
    $p = $p + 2 + u16($p);                      // skip the instructions
    $flags = [];
    while (len($flags) < $npts) {
        $f = u8($p);
        $p = $p + 1;
        if (has($f, 128)) { throw "glyph " + str($g) + ": cubic outlines are not supported"; }
        push($flags, $f);
        if (has($f, 8)) {
            $r = u8($p);
            $p = $p + 1;
            while ($r > 0) { push($flags, $f); $r = $r - 1; }
        }
    }
    $xs = [];
    $x = $dx;
    each ($f in $flags) {
        if (has($f, 2)) {
            if (has($f, 16)) { $x = $x + u8($p); } else { $x = $x - u8($p); }
            $p = $p + 1;
        } else {
            if (!has($f, 16)) { $x = $x + i16($p); $p = $p + 2; }
        }
        push($xs, $x);
    }
    $pts = [];
    $y = $dy;
    $i = 0;
    each ($f in $flags) {
        if (has($f, 4)) {
            if (has($f, 32)) { $y = $y + u8($p); } else { $y = $y - u8($p); }
            $p = $p + 1;
        } else {
            if (!has($f, 32)) { $y = $y + i16($p); $p = $p + 2; }
        }
        push($pts, [$xs[$i], $y]);
        $i = $i + 1;
    }
    $start = 0;
    each ($end in $ends) {
        $cpts = [];
        $con = [];
        $i = $start;
        while ($i <= $end) { push($cpts, $pts[$i]); push($con, has($flags[$i], 1)); $i = $i + 1; }
        $count = $count + contour($cpts, $con);
        $start = $end + 1;
    }
    return $count;
}

// ── The header ──────────────────────────────────────────────────────────────

// U+0020: a codepoint as it is conventionally written, in hex.
def uplus($n) {
    $s = "";
    while (len($s) < 4 || $n > 0) { $s = substr("0123456789ABCDEF", $n % 16, 1) + $s; $n = floor($n / 16); }
    return "U+" + $s;
}

$missing = [];
each ($cp in $CODEPOINTS) { if (glyphOf($cp) == 0) { push($missing, uplus($cp)); } }
if (len($missing) > 0) { throw "the font has no glyph for: " + join($missing, ", "); }

$entries = [];
each ($cp in $CODEPOINTS) {
    $g = glyphOf($cp);
    $start = len($SEGS);
    $ops = outline($g, 0, 0, true);
    push($entries, "    {" + str($cp) + ", " + str($start) + ", " + str($ops) + ", " + str(hmetric($g)[0]) + "},");
}

// The licence, from beside the font, so the notice travels with the data.
$parts = split($FONT, "/");
$name = $parts[len($parts) - 1];
$dir = ".";
if (len($parts) > 1) {
    $dir = $parts[0];
    $i = 1;
    while ($i < len($parts) - 1) { $dir = $dir + "/" + $parts[$i]; $i = $i + 1; }
}
$licence = null;
each ($l in ["LICENSE_DEJAVU", "LICENSE", "LICENCE"]) {
    if ($licence == null && file_exists($dir + "/" + $l)) { $licence = readfile($dir + "/" + $l, "rb"); }
}
if ($licence == null) { throw "no licence file found beside " + $FONT + ": the notice must ship with the glyphs"; }

// "// " + line, without trailing whitespace.
def commentLine($line) {
    $s = "// " + $line;
    $n = len($s);
    while ($n > 0 && (substr($s, $n - 1, 1) == " " || substr($s, $n - 1, 1) == "\t" || substr($s, $n - 1, 1) == "\r")) {
        $n = $n - 1;
    }
    return substr($s, 0, $n);
}

$out = [
    "#pragma once",
    "// Generated by scripts/gen_font_tables.b -- do not edit by hand.",
    "// Source: " + $name + ", " + str(len($entries)) + " glyphs, " + str(len($SEGS)) + " segments.",
    "//",
    "// The glyph outlines below are DejaVu Sans, redistributed under the terms",
    "// reproduced here in full. The designs are unmodified.",
    "//"
];
$lines = split(replace($licence, "\r\n", "\n"), "\n");
$last = len($lines);
while ($last > 0 && $lines[$last - 1] == "") { $last = $last - 1; }   // trailing blank lines
$i = 0;
while ($i < $last) { push($out, commentLine($lines[$i])); $i = $i + 1; }
push($out, "");
push($out, "#include <cstdint>");
push($out, "");
push($out, "namespace bplot_raster {");
push($out, "");
push($out, "static const int32_t kFontUnitsPerEm = " + str($UPM) + ";");
push($out, "");
push($out, "// {kind, x1, y1, x2, y2} -- kind 0 move, 1 line, 2 quadratic (control, end).");
push($out, "static const int16_t kGlyphSegments[" + str(len($SEGS)) + "][5] = {");
each ($s in $SEGS) {
    // A move or a line has one point; its unused second point is written as 0, 0.
    $row = str($s[0]) + ", " + str($s[1]) + ", " + str($s[2]) + ", 0, 0";
    if (len($s) == 5) { $row = join([str($s[0]), str($s[1]), str($s[2]), str($s[3]), str($s[4])], ", "); }
    push($out, "    {" + $row + "},");
}
push($out, "};");
push($out, "");
push($out, "// {codepoint, first segment, segment count, advance width}");
push($out, "static const int32_t kGlyphIndex[" + str(len($entries)) + "][4] = {");
each ($e in $entries) { push($out, $e); }
push($out, "};");
push($out, "");
push($out, "} // namespace bplot_raster");

$path = env("OUT");
if ($path == null || $path == "") { $path = "bantu-src/compiler/src/raster_font.hpp"; }
writefile($path, join($out, "\n") + "\n", "wb");
print("wrote " + $path + ": " + str(len($entries)) + " glyphs, " + str(len($SEGS)) + " segments");
