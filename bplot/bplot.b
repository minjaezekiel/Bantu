// ════════════════════════════════════════════════════════════════════════
//  bplot — data visualisation for Bantu, in the shape of matplotlib.
//
//  100% pure Bantu. It emits SVG, which is text, so producing it is string
//  work and the interpreter is fast enough: the cost of a chart is
//  proportional to the number of DRAWN ELEMENTS (a few hundred), not to the
//  number of data points. A 100,000-point line is one <polyline>.
//
//  Three lines to a chart:
//
//      include "bplot" as plt;
//      plt.plot([1, 2, 3], [2, 4, 9]);
//      plt.savefig("chart.svg");
//
//  The layering is matplotlib's -- a stateful plt.* API over Figure / Axes /
//  Artist over a swappable backend -- because that is what let matplotlib
//  change output targets for twenty years without breaking its API. The
//  objects are there when you need them and appear nowhere in the quickstart.
//
//  SECURITY, and please read this one: SVG is not an image format the way PNG
//  is. It is XML that browsers EXECUTE -- it can carry <script>, on* handlers
//  and <foreignObject> -- and sua already serves image/svg+xml. So a chart
//  title taken from a request parameter is an XSS vector. Every text node and
//  every attribute value here is escaped at the point of emission, with no
//  way to opt out and no raw-SVG hatch. That makes injection THROUGH bplot
//  impossible; it cannot make SVG stop being executable, so if you serve
//  charts built from untrusted data, serve them from a separate origin or
//  under a restrictive Content-Security-Policy. See docs/bplot.md.
//
//  Source:  bplot/bplot.b
//  Docs:    docs/bplot.md   ·   Design: docs/bplot-architecture.md
//  Tests:   tests/bplot_core_test.b, tests/bplot_stress.sh
// ════════════════════════════════════════════════════════════════════════

// ── Palette ──────────────────────────────────────────────────────────────
// matplotlib's tab10: the most-recognised categorical palette in data
// visualisation, and chosen for distinguishability rather than for looks.
$_CYCLE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"];

$_FG        = "#333333";
$_GRIDCOLOR = "#dddddd";
$_AXISCOLOR = "#888888";
$_BG        = "#ffffff";
$_FONT      = "sans-serif";

// Helvetica advance widths, ASCII 32..126, in 1/1000 em. Helvetica and Arial
// share metrics and are what sans-serif resolves to on most systems, so this
// is accurate where it matters and approximate elsewhere. Used for legend
// boxes and axis gutters; layout pads outward, so an error produces
// whitespace rather than a collision. A data table, not a font parser --
// the same reasoning that keeps a from-scratch P-256 in this tree instead
// of a dependency on OpenSSL.
$_ADVANCE = [278,278,355,556,556,889,667,191,333,333,389,584,278,333,278,278,
             556,556,556,556,556,556,556,556,556,556,
             278,278,584,584,584,556,1015,
             667,667,722,722,667,611,778,722,278,500,667,556,833,722,778,667,
             778,722,667,611,722,667,944,667,667,611,
             278,278,278,469,556,333,
             556,556,500,556,556,278,556,556,222,222,500,222,833,556,556,556,
             556,333,500,278,556,500,722,500,500,500,
             334,260,334,584];

// ════════════════════════════════════════════════════════════════════════
//  Internal helpers
// ════════════════════════════════════════════════════════════════════════

// Escape for XML. `&` MUST go first or it would double-escape the others.
// Control bytes below 0x20 other than tab/LF/CR are not representable in
// XML 1.0 at all -- one of them makes the whole document fail to parse,
// turning a chart into a blank page with nothing in any log -- so they are
// dropped rather than escaped.
def _esc($s) {
    $t = str($s);
    $t = replace($t, "&", "&amp;");
    $t = replace($t, "<", "&lt;");
    $t = replace($t, ">", "&gt;");
    $t = replace($t, "\"", "&quot;");
    $t = replace($t, "'", "&#39;");
    // Only walk the string if it actually contains something to strip.
    $clean = [];
    $i = 0;
    $dirty = false;
    while ($i < len($t)) {
        $c = ord($t[$i]);
        if ($c < 32 && $c != 9 && $c != 10 && $c != 13) { $dirty = true; }
        else { push($clean, $t[$i]); }
        $i = $i + 1;
    }
    if ($dirty) { return join($clean, ""); }
    return $t;
}

// Fixed-decimal formatting. str() cannot be used for a tick label: it gives
// six significant digits (1/3 -> "0.333333") and leaks scientific notation
// (0.000012345678 -> "1.23457e-05").
def _fmt($x, $d) {
    if (isnan($x))  { return "NaN"; }
    if (isinf($x))  { if ($x > 0) { return "inf"; } return "-inf"; }
    if ($d == null) { $d = 0; }
    $neg = $x < 0;
    $a = abs($x);
    // Past 2^53 the scaling below loses the digits it is trying to place.
    if ($a >= 1000000000000000) { return str($x); }
    $scale = pow(10, $d);
    $units = round($a * $scale);
    $ip = floor($units / $scale);
    $fp = $units - $ip * $scale;
    $out = str($ip);
    if ($d > 0) {
        $fs = str($fp);
        while (len($fs) < $d) { $fs = "0" + $fs; }
        $out = $out + "." + $fs;
    }
    if ($neg && $units > 0) { $out = "-" + $out; }
    return $out;
}

// Coordinates for the SVG itself. Two decimals: a user unit is a pixel, a
// hundredth of one is below any display's resolution, and it roughly halves
// the file size against the six digits str() would give.
def _px($v) { return _fmt($v, 2); }

// How many decimals does a tick step need?  step 0.25 -> 2, step 20 -> 0.
def _decimalsFor($step) {
    if ($step <= 0) { return 0; }
    $d = 0 - floor(log10($step));
    if ($d < 0) { $d = 0; }
    if ($d > 8) { $d = 8; }
    return $d;
}

// Width of a string at a given font size, in pixels.
def _textWidth($s, $size) {
    $t = str($s);
    $u = 0;
    $i = 0;
    while ($i < len($t)) {
        $c = ord($t[$i]);
        if ($c >= 32 && $c <= 126) { $u = $u + $_ADVANCE[$c - 32]; }
        else { $u = $u + 556; }          // non-ASCII: the width of `n`
        $i = $i + 1;
    }
    return $u * $size / 1000;
}

// A colour is a named colour, a #rgb / #rrggbb string, or an [r,g,b] list.
// Colours reach ATTRIBUTES, and an unvalidated attribute value is the second
// favourite SVG injection point after a text node -- so this checks the shape
// and refuses anything else rather than interpolating what it was handed.
$_NAMED = {"black": "#000000", "white": "#ffffff", "red": "#d62728",
           "green": "#2ca02c", "blue": "#1f77b4", "orange": "#ff7f0e",
           "purple": "#9467bd", "grey": "#7f7f7f", "gray": "#7f7f7f",
           "brown": "#8c564b", "pink": "#e377c2", "cyan": "#17becf",
           "olive": "#bcbd22", "none": "none"};

def _hexdigit($c) {
    return ($c >= 48 && $c <= 57) || ($c >= 97 && $c <= 102) || ($c >= 65 && $c <= 70);
}

def _color($c, $fallback) {
    if ($c == null) { return $fallback; }
    if (type($c) == "list") {
        if (len($c) < 3) { throw "bplot: a colour list needs [r, g, b], got " + str(len($c)) + " entries"; }
        $out = "#";
        $i = 0;
        while ($i < 3) {
            $v = round(clamp($c[$i], 0, 255));
            $h = "0123456789abcdef";
            $out = $out + $h[floor($v / 16)] + $h[$v - floor($v / 16) * 16];
            $i = $i + 1;
        }
        return $out;
    }
    $s = str($c);
    if (len($s) > 0 && $s[0] == "#") {
        if (len($s) != 4 && len($s) != 7) {
            throw "bplot: '" + $s + "' is not a colour -- use #rgb or #rrggbb";
        }
        $i = 1;
        while ($i < len($s)) {
            if (!_hexdigit(ord($s[$i]))) { throw "bplot: '" + $s + "' is not a hex colour"; }
            $i = $i + 1;
        }
        return $s;
    }
    $named = $_NAMED[$s];
    if ($named != null) { return $named; }
    throw "bplot: unknown colour '" + $s + "' -- use #rrggbb, [r,g,b], or one of " + join(keys($_NAMED), ", ");
}

// ── Reading data ─────────────────────────────────────────────────────────
// A sequence may be a Bantu list, a numba ndarray, or an arctic column --
// converted once, here, at the API boundary, so nothing downstream has to
// care which it was. (Series/DataFrame land in B4.)
def _seq($v, $what) {
    if ($v == null) { throw "bplot: " + $what + " is null"; }
    $t = type($v);
    if ($t == "list") { return $v; }
    if ($t == "ndarray") { return nd_to_list($v); }
    if ($t == "column")  { return col_to_list($v); }
    throw "bplot: " + $what + " must be a list" +
          " (or an ndarray or a column), got " + $t;
}

// Numeric limits over a sequence, skipping NaN and infinities.
//
// The fast path is one native call: min()/max() over a list walk it in C++,
// which is how the limits of a 100,000-point series cost ~10 ms instead of a
// 100,000-iteration Bantu loop. But the language's min/max PROPAGATE NaN by
// design -- a primitive should not silently discard a value it was handed --
// so a NaN or an infinity in the data sends us to the filtering loop.
def _limits($xs) {
    if (len($xs) == 0) { return null; }
    $lo = min($xs);
    $hi = max($xs);
    if (isfinite($lo) && isfinite($hi)) { return [$lo, $hi]; }
    $flo = INF;
    $fhi = 0 - INF;
    $any = false;
    $i = 0;
    while ($i < len($xs)) {
        $v = $xs[$i];
        if (isfinite($v)) {
            if (!$any) { $flo = $v; $fhi = $v; $any = true; }
            if ($v < $flo) { $flo = $v; }
            if ($v > $fhi) { $fhi = $v; }
        }
        $i = $i + 1;
    }
    if (!$any) { return null; }
    return [$flo, $fhi];
}

// A range with no extent cannot be drawn. Expand it the way matplotlib does:
// 5% of the magnitude, or +/- 0.5 at zero.
def _expand($lo, $hi) {
    if ($lo == null) { return [0, 1]; }
    if ($hi > $lo) { return [$lo, $hi]; }
    $m = abs($lo) * 0.05;
    if ($m == 0) { $m = 0.5; }
    return [$lo - $m, $lo + $m];
}

// ── Ticking: MaxNLocator ─────────────────────────────────────────────────
// "Pick about n round numbers spanning this range" is the single thing that
// separates a chart that looks designed from one that looks generated.
//
// 2.5 belongs in the candidate list: without it, 0-1 at six ticks gives steps
// of 0.2 where 0.25 (five ticks) is the better choice, and matplotlib
// includes it.
def _ticks($lo, $hi, $n) {
    if ($n == null || $n < 2) { $n = 6; }
    if (!isfinite($lo) || !isfinite($hi) || $hi <= $lo) { return [$lo]; }
    $raw = ($hi - $lo) / $n;
    if ($raw <= 0) { return [$lo]; }
    $mag = pow(10, floor(log10($raw)));
    $steps = [1, 2, 2.5, 5, 10];
    $step = 10 * $mag;
    $i = 0;
    while ($i < len($steps)) {
        if ($steps[$i] * $mag >= $raw) { $step = $steps[$i] * $mag; break; }
        $i = $i + 1;
    }
    // The epsilon MUST be relative. Absolute, floating-point accumulation
    // drops the FINAL tick on ranges like 0-1 by 0.1 -- and the top tick is
    // exactly the one a reader looks for.
    $eps = 0.0000000001 * max(abs($lo), abs($hi), 1);
    $k = ceil(($lo - $eps) / $step);
    $out = [];
    $guard = 0;
    while ($guard < 1000) {
        // Computed from k rather than accumulated, so the error does not grow
        // along the axis.
        $t = $k * $step;
        if ($t > $hi + $eps) { break; }
        if ($t >= $lo - $eps) { push($out, $t); }
        $k = $k + 1;
        $guard = $guard + 1;
    }
    if (len($out) == 0) { push($out, $lo); }
    return $out;
}

def _tickStep($ticks) {
    if (len($ticks) < 2) { return 1; }
    return abs($ticks[1] - $ticks[0]);
}

// ════════════════════════════════════════════════════════════════════════
//  The SVG backend
//
//  A backend is an object with a fixed method set. Nothing above it knows
//  which one it is, which is what will let the raster backend (B6) drop in
//  without touching a single artist.
//
//  Every method escapes what it emits. There is deliberately no way to write
//  raw markup: a library that offers one has delegated its security to every
//  caller, and the first caller to pass user text through it reopens the hole.
// ════════════════════════════════════════════════════════════════════════
class BPlotSvg {
    def init($w, $h) {
        $this.w = $w;
        $this.h = $h;
        // Parts in a list, joined once. Building this with `$s = $s + part`
        // is the classic quadratic trap; join() is the idiom regardless of
        // whether the interpreter has the in-place-append optimisation.
        $this.parts = [];
        $this.clipId = 0;
    }

    def open() {
        push($this.parts, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" +
             _px($this.w) + "\" height=\"" + _px($this.h) + "\" viewBox=\"0 0 " +
             _px($this.w) + " " + _px($this.h) + "\">");
        return $this;
    }

    def close() { push($this.parts, "</svg>"); return $this; }

    def rect($x, $y, $w, $h, $fill, $stroke, $sw) {
        $s = "<rect x=\"" + _px($x) + "\" y=\"" + _px($y) + "\" width=\"" +
             _px($w) + "\" height=\"" + _px($h) + "\" fill=\"" + _esc($fill) + "\"";
        if ($stroke != null) {
            $s = $s + " stroke=\"" + _esc($stroke) + "\" stroke-width=\"" + _px($sw) + "\"";
        }
        push($this.parts, $s + "/>");
        return $this;
    }

    def line($x1, $y1, $x2, $y2, $stroke, $sw) {
        push($this.parts, "<line x1=\"" + _px($x1) + "\" y1=\"" + _px($y1) +
             "\" x2=\"" + _px($x2) + "\" y2=\"" + _px($y2) + "\" stroke=\"" +
             _esc($stroke) + "\" stroke-width=\"" + _px($sw) + "\"/>");
        return $this;
    }

    def circle($cx, $cy, $r, $fill) {
        push($this.parts, "<circle cx=\"" + _px($cx) + "\" cy=\"" + _px($cy) +
             "\" r=\"" + _px($r) + "\" fill=\"" + _esc($fill) + "\"/>");
        return $this;
    }

    // pts is a flat list [x0,y0,x1,y1,...]; NaN coordinates are impossible
    // here because the caller splits runs at them (see _drawLine).
    def polyline($pts, $stroke, $sw, $dash) {
        if (len($pts) < 4) { return $this; }
        $coords = [];
        $i = 0;
        while ($i + 1 < len($pts)) {
            push($coords, _px($pts[$i]) + "," + _px($pts[$i + 1]));
            $i = $i + 2;
        }
        $s = "<polyline points=\"" + join($coords, " ") + "\" fill=\"none\" stroke=\"" +
             _esc($stroke) + "\" stroke-width=\"" + _px($sw) + "\"";
        if ($dash != null) { $s = $s + " stroke-dasharray=\"" + _esc($dash) + "\""; }
        push($this.parts, $s + " stroke-linecap=\"round\" stroke-linejoin=\"round\"/>");
        return $this;
    }

    // anchor: "start" | "middle" | "end"
    def text($x, $y, $s, $size, $fill, $anchor, $rotate) {
        $t = "<text x=\"" + _px($x) + "\" y=\"" + _px($y) + "\" font-family=\"" +
             $_FONT + "\" font-size=\"" + _px($size) + "\" fill=\"" + _esc($fill) +
             "\" text-anchor=\"" + _esc($anchor) + "\"";
        if ($rotate != null && $rotate != 0) {
            $t = $t + " transform=\"rotate(" + _px($rotate) + " " + _px($x) + " " + _px($y) + ")\"";
        }
        push($this.parts, $t + ">" + _esc($s) + "</text>");
        return $this;
    }

    def groupOpen($clip) {
        if ($clip == null) { push($this.parts, "<g>"); }
        else { push($this.parts, "<g clip-path=\"url(#" + _esc($clip) + ")\">"); }
        return $this;
    }
    def groupClose() { push($this.parts, "</g>"); return $this; }

    def clip($x, $y, $w, $h) {
        $this.clipId = $this.clipId + 1;
        $id = "bpclip" + str($this.clipId);
        push($this.parts, "<defs><clipPath id=\"" + $id + "\"><rect x=\"" + _px($x) +
             "\" y=\"" + _px($y) + "\" width=\"" + _px($w) + "\" height=\"" +
             _px($h) + "\"/></clipPath></defs>");
        return $id;
    }

    def render() { return join($this.parts, "\n"); }
}

// ════════════════════════════════════════════════════════════════════════
//  Axes — the plotting box
// ════════════════════════════════════════════════════════════════════════
class BPlotAxes {
    def init($fig, $left, $top, $w, $h) {
        $this.fig = $fig;
        // Position in FIGURE PIXELS, top-left origin (SVG's own).
        $this.left = $left;
        $this.top = $top;
        $this.w = $w;
        $this.h = $h;
        $this.artists = [];
        $this.xlabel = "";
        $this.ylabel = "";
        $this.title = "";
        $this.xlimSet = null;
        $this.ylimSet = null;
        $this.gridOn = false;
        $this.legendOn = false;
        $this.colorIdx = 0;
        $this.simplify = true;
    }

    def nextColor() {
        $c = $_CYCLE[$this.colorIdx - floor($this.colorIdx / len($_CYCLE)) * len($_CYCLE)];
        $this.colorIdx = $this.colorIdx + 1;
        return $c;
    }

    // ── Artists ──────────────────────────────────────────────────────────
    def plot($x, $y, $opts) {
        $xs = _seq($x, "x");
        $ys = _seq($y, "y");
        if (len($xs) != len($ys)) {
            throw "bplot.plot: x has " + str(len($xs)) + " points and y has " +
                  str(len($ys)) + " -- they must match";
        }
        if ($opts == null) { $opts = {}; }
        push($this.artists, {
            "kind":  "line",
            "x":     $xs,
            "y":     $ys,
            "color": _color($opts["color"], $this.nextColor()),
            "width": _optNum($opts["width"], 1.8),
            "label": $opts["label"],
            "dash":  $opts["dash"]
        });
        return $this;
    }

    def scatter($x, $y, $opts) {
        $xs = _seq($x, "x");
        $ys = _seq($y, "y");
        if (len($xs) != len($ys)) {
            throw "bplot.scatter: x has " + str(len($xs)) + " points and y has " +
                  str(len($ys)) + " -- they must match";
        }
        if ($opts == null) { $opts = {}; }
        push($this.artists, {
            "kind":  "scatter",
            "x":     $xs,
            "y":     $ys,
            "color": _color($opts["color"], $this.nextColor()),
            "size":  _optNum($opts["size"], 3),
            "label": $opts["label"]
        });
        return $this;
    }

    def bar($x, $y, $opts) {
        return $this.addBar($x, $y, $opts, false);
    }
    def barh($x, $y, $opts) {
        return $this.addBar($x, $y, $opts, true);
    }
    def addBar($x, $y, $opts, $horiz) {
        $xs = _seq($x, "x");
        $ys = _seq($y, "y");
        if (len($xs) != len($ys)) {
            throw "bplot.bar: x has " + str(len($xs)) + " bars and y has " +
                  str(len($ys)) + " heights -- they must match";
        }
        if ($opts == null) { $opts = {}; }
        push($this.artists, {
            "kind":  "bar",
            "x":     $xs,
            "y":     $ys,
            "color": _color($opts["color"], $this.nextColor()),
            "width": _optNum($opts["width"], 0.8),
            "label": $opts["label"],
            "horiz": $horiz
        });
        return $this;
    }

    // ── Decoration ───────────────────────────────────────────────────────
    def setXLabel($s) { $this.xlabel = str($s); return $this; }
    def setYLabel($s) { $this.ylabel = str($s); return $this; }
    def setTitle($s)  { $this.title  = str($s); return $this; }
    def setGrid($on)  { if ($on == null) { $on = true; } $this.gridOn = $on; return $this; }
    def setLegend($on){ if ($on == null) { $on = true; } $this.legendOn = $on; return $this; }
    def setXLim($lo, $hi) { $this.xlimSet = _checkLim($lo, $hi, "xlim"); return $this; }
    def setYLim($lo, $hi) { $this.ylimSet = _checkLim($lo, $hi, "ylim"); return $this; }

    // ── Limits ───────────────────────────────────────────────────────────
    def dataLimits($which) {
        $lo = null;
        $hi = null;
        $i = 0;
        while ($i < len($this.artists)) {
            $a = $this.artists[$i];
            $vals = $a[$which];
            // A bar's baseline is zero, so the value axis must include it.
            $lim = _limits($vals);
            if ($lim != null) {
                if ($lo == null || $lim[0] < $lo) { $lo = $lim[0]; }
                if ($hi == null || $lim[1] > $hi) { $hi = $lim[1]; }
            }
            if ($a["kind"] == "bar") {
                $isValueAxis = ($which == "y" && !$a["horiz"]) || ($which == "x" && $a["horiz"]);
                if ($isValueAxis) {
                    if ($lo == null || $lo > 0) { $lo = 0; }
                    if ($hi == null || $hi < 0) { $hi = 0; }
                }
            }
            $i = $i + 1;
        }
        return [$lo, $hi];
    }

    def viewLimits($which) {
        $set = $this.xlimSet;
        if ($which == "y") { $set = $this.ylimSet; }
        if ($set != null) { return $set; }
        $d = $this.dataLimits($which);
        $e = _expand($d[0], $d[1]);
        // matplotlib's default 5% margin, so points never sit on the frame.
        $pad = ($e[1] - $e[0]) * 0.05;
        return [$e[0] - $pad, $e[1] + $pad];
    }

    // ── Transform: data -> pixels ────────────────────────────────────────
    // The y flip lives here and nowhere else. An artist that forgets it is
    // the single most common bug in hand-rolled plotting code, so there is
    // exactly one place it can be forgotten.
    // Both axes reduce to `pixel = value * a + b`. The coefficients are
    // computed once per axis and the per-point loops multiply-add inline --
    // 100,000 points meant 200,000 method calls through px()/py(), which was
    // 3.2 s of the 3.5 s a big chart took.
    //
    // THE Y FLIP IS THE `0 - $a` BELOW, and it is the only one in the library.
    // SVG's origin is top-left and a chart's is bottom-left; an artist that
    // applies the flip itself, or forgets to, is the most common bug in
    // hand-rolled plotting code. px() and py() are defined in terms of these
    // same coefficients, so there is one formula, not two that must agree.
    def coefX($lim) {
        $a = $this.w / ($lim[1] - $lim[0]);
        return [$a, $this.left - $lim[0] * $a];
    }
    def coefY($lim) {
        $a = $this.h / ($lim[1] - $lim[0]);
        return [0 - $a, $this.top + $this.h + $lim[0] * $a];
    }
    def px($v, $lim) { $c = $this.coefX($lim); return $v * $c[0] + $c[1]; }
    def py($v, $lim) { $c = $this.coefY($lim); return $v * $c[0] + $c[1]; }
}

// A helper outside the class: `new` is only reachable inside this module, and
// options dicts want a numeric default without a null creeping through.
def _optNum($v, $dflt) {
    if ($v == null) { return $dflt; }
    if (type($v) != "number") { throw "bplot: expected a number, got " + type($v); }
    return $v;
}

def _checkLim($lo, $hi, $what) {
    if ($lo == null || $hi == null) { return null; }
    if (type($lo) != "number" || type($hi) != "number") {
        throw "bplot." + $what + ": both bounds must be numbers";
    }
    if (!isfinite($lo) || !isfinite($hi)) {
        throw "bplot." + $what + ": bounds must be finite, got " + str($lo) + " and " + str($hi);
    }
    if ($hi <= $lo) {
        throw "bplot." + $what + ": upper bound " + str($hi) + " must be above lower bound " + str($lo);
    }
    return [$lo, $hi];
}

// ════════════════════════════════════════════════════════════════════════
//  Figure — the page
// ════════════════════════════════════════════════════════════════════════
class BPlotFigure {
    def init($w, $h) {
        if ($w == null) { $w = 640; }
        if ($h == null) { $h = 480; }
        if (type($w) != "number" || type($h) != "number") {
            throw "bplot.figure: width and height must be numbers";
        }
        if ($w <= 0 || $h <= 0 || !isfinite($w) || !isfinite($h)) {
            throw "bplot.figure: size must be positive and finite, got " +
                  str($w) + "x" + str($h);
        }
        $this.w = $w;
        $this.h = $h;
        $this.axesList = [];
        $this.cur = 0 - 1;
    }

    def addAxes() {
        // Gutters: room for tick labels, axis labels and a title. B3's
        // tight_layout will compute these from the real text metrics; B1
        // reserves a sensible fixed margin.
        $l = 62;
        $r = 18;
        $t = 34;
        $b = 52;
        $ax = new BPlotAxes($this, $l, $t, $this.w - $l - $r, $this.h - $t - $b);
        push($this.axesList, $ax);
        $this.cur = len($this.axesList) - 1;
        return $ax;
    }

    def gca() {
        if (len($this.axesList) == 0) { return $this.addAxes(); }
        return $this.axesList[$this.cur];
    }

    def to_svg() {
        $bk = new BPlotSvg($this.w, $this.h);
        $bk.open();
        $bk.rect(0, 0, $this.w, $this.h, $_BG, null, null);
        $i = 0;
        while ($i < len($this.axesList)) {
            _drawAxes($bk, $this.axesList[$i]);
            $i = $i + 1;
        }
        $bk.close();
        return $bk.render();
    }

    def savefig($path) {
        if ($path == null || type($path) != "string" || len($path) == 0) {
            throw "bplot.savefig: needs a file path";
        }
        if (!_endsWith($path, ".svg")) {
            throw "bplot.savefig: only .svg is supported in this release, got '" + $path +
                  "' -- PNG output lands with the raster backend";
        }
        writefile($path, $this.to_svg());
        return $path;
    }
}

def _endsWith($s, $suffix) {
    if (len($s) < len($suffix)) { return false; }
    return substr($s, len($s) - len($suffix), len($suffix)) == $suffix;
}

// ════════════════════════════════════════════════════════════════════════
//  Rendering
// ════════════════════════════════════════════════════════════════════════

def _drawAxes($bk, $ax) {
    $xlim = $ax.viewLimits("x");
    $ylim = $ax.viewLimits("y");
    $xt = _ticks($xlim[0], $xlim[1], 6);
    $yt = _ticks($ylim[0], $ylim[1], 6);
    $xd = _decimalsFor(_tickStep($xt));
    $yd = _decimalsFor(_tickStep($yt));

    $x0 = $ax.left;
    $y0 = $ax.top;
    $x1 = $ax.left + $ax.w;
    $y1 = $ax.top + $ax.h;

    // Grid, behind everything.
    if ($ax.gridOn) {
        $i = 0;
        while ($i < len($xt)) {
            $px = $ax.px($xt[$i], $xlim);
            $bk.line($px, $y0, $px, $y1, $_GRIDCOLOR, 1);
            $i = $i + 1;
        }
        $i = 0;
        while ($i < len($yt)) {
            $py = $ax.py($yt[$i], $ylim);
            $bk.line($x0, $py, $x1, $py, $_GRIDCOLOR, 1);
            $i = $i + 1;
        }
    }

    // Data, clipped to the box so a point outside the view cannot escape it.
    $clipId = $bk.clip($x0, $y0, $ax.w, $ax.h);
    $bk.groupOpen($clipId);
    $i = 0;
    while ($i < len($ax.artists)) {
        _drawArtist($bk, $ax, $ax.artists[$i], $xlim, $ylim);
        $i = $i + 1;
    }
    $bk.groupClose();

    // Frame.
    $bk.rect($x0, $y0, $ax.w, $ax.h, "none", $_AXISCOLOR, 1);

    // Ticks and their labels.
    $i = 0;
    while ($i < len($xt)) {
        $px = $ax.px($xt[$i], $xlim);
        $bk.line($px, $y1, $px, $y1 + 4, $_AXISCOLOR, 1);
        $bk.text($px, $y1 + 17, _fmt($xt[$i], $xd), 11, $_FG, "middle", null);
        $i = $i + 1;
    }
    $i = 0;
    while ($i < len($yt)) {
        $py = $ax.py($yt[$i], $ylim);
        $bk.line($x0 - 4, $py, $x0, $py, $_AXISCOLOR, 1);
        $bk.text($x0 - 8, $py + 4, _fmt($yt[$i], $yd), 11, $_FG, "end", null);
        $i = $i + 1;
    }

    // Labels and title.
    if (len($ax.xlabel) > 0) {
        $bk.text(($x0 + $x1) / 2, $y1 + 40, $ax.xlabel, 12, $_FG, "middle", null);
    }
    if (len($ax.ylabel) > 0) {
        $bk.text($x0 - 44, ($y0 + $y1) / 2, $ax.ylabel, 12, $_FG, "middle", -90);
    }
    if (len($ax.title) > 0) {
        $bk.text(($x0 + $x1) / 2, $y0 - 12, $ax.title, 14, $_FG, "middle", null);
    }

    if ($ax.legendOn) { _drawLegend($bk, $ax, $x1, $y0); }
}

def _drawArtist($bk, $ax, $a, $xlim, $ylim) {
    $k = $a["kind"];
    if ($k == "line")    { _drawLine($bk, $ax, $a, $xlim, $ylim); return null; }
    if ($k == "scatter") { _drawScatter($bk, $ax, $a, $xlim, $ylim); return null; }
    if ($k == "bar")     { _drawBar($bk, $ax, $a, $xlim, $ylim); return null; }
    return null;
}

// NaN breaks the line, as it does in matplotlib. Untreated, a NaN coordinate
// emits points="NaN,12 ..." which every browser renders as NOTHING AT ALL,
// with no error anywhere -- the worst failure mode available. Splitting draws
// the valid segments and makes the gap visible as a gap.
def _drawLine($bk, $ax, $a, $xlim, $ylim) {
    $xs = $a["x"];
    $ys = $a["y"];
    $cx = $ax.coefX($xlim);
    $cy = $ax.coefY($ylim);
    $xa = $cx[0]; $xb = $cx[1];
    $ya = $cy[0]; $yb = $cy[1];
    $run = [];
    $i = 0;
    $n = len($xs);
    while ($i < $n) {
        $xv = $xs[$i];
        $yv = $ys[$i];
        if (isfinite($xv) && isfinite($yv)) {
            push($run, $xv * $xa + $xb);
            push($run, $yv * $ya + $yb);
        } else {
            if (len($run) >= 4) { $bk.polyline(_simplify($run, $ax.simplify), $a["color"], $a["width"], $a["dash"]); }
            else { if (len($run) == 2) { $bk.circle($run[0], $run[1], $a["width"], $a["color"]); } }
            $run = [];
        }
        $i = $i + 1;
    }
    if (len($run) >= 4) { $bk.polyline(_simplify($run, $ax.simplify), $a["color"], $a["width"], $a["dash"]); }
    else { if (len($run) == 2) { $bk.circle($run[0], $run[1], $a["width"], $a["color"]); } }
    return null;
}

// Path simplification (decision BP3). An 800-pixel-wide canvas cannot show
// more than ~800 distinct columns, so 100,000 vertices carry no information
// that 1,600 do not. For each pixel column keep the first, last, minimum-y
// and maximum-y points: that preserves every spike, which naive decimation
// does not, and a dropped spike is a chart that lies.
def _simplify($pts, $on) {
    $n = floor(len($pts) / 2);
    if (!$on || $n < 1000) { return $pts; }
    $out = [];
    $ci = 0;
    $cx = 0;
    $first = 0; $last = 0; $lo = 0; $hi = 0;
    $have = false;
    $i = 0;
    while ($i < $n) {
        $x = $pts[$i * 2];
        $y = $pts[$i * 2 + 1];
        $col = round($x);
        if (!$have) {
            $cx = $col; $first = $y; $last = $y; $lo = $y; $hi = $y; $have = true;
        } else {
            if ($col != $cx) {
                extend($out, _colPoints($cx, $first, $lo, $hi, $last));
                $cx = $col; $first = $y; $lo = $y; $hi = $y;
            }
            if ($y < $lo) { $lo = $y; }
            if ($y > $hi) { $hi = $y; }
            $last = $y;
        }
        $i = $i + 1;
    }
    if ($have) { extend($out, _colPoints($cx, $first, $lo, $hi, $last)); }
    return $out;
}

// Returns the column's points rather than appending to a list it was handed.
// A Bantu list passed to a function is a COPY -- value semantics, unlike a
// dict or a class instance -- so pushing into a parameter mutates the copy
// and the caller sees nothing. extend() at the call site mutates the real one.
def _colPoints($x, $first, $lo, $hi, $last) {
    $p = [$x, $first];
    if ($lo != $first && $lo != $last) { push($p, $x); push($p, $lo); }
    if ($hi != $first && $hi != $last && $hi != $lo) { push($p, $x); push($p, $hi); }
    if ($last != $first) { push($p, $x); push($p, $last); }
    return $p;
}

def _drawScatter($bk, $ax, $a, $xlim, $ylim) {
    $xs = $a["x"];
    $ys = $a["y"];
    $r = $a["size"];
    $c = $a["color"];
    $cx = $ax.coefX($xlim);
    $cy = $ax.coefY($ylim);
    $xa = $cx[0]; $xb = $cx[1];
    $ya = $cy[0]; $yb = $cy[1];
    $i = 0;
    $n = len($xs);
    while ($i < $n) {
        $xv = $xs[$i];
        $yv = $ys[$i];
        if (isfinite($xv) && isfinite($yv)) {
            $bk.circle($xv * $xa + $xb, $yv * $ya + $yb, $r, $c);
        }
        $i = $i + 1;
    }
    return null;
}

def _drawBar($bk, $ax, $a, $xlim, $ylim) {
    $xs = $a["x"];
    $ys = $a["y"];
    $horiz = $a["horiz"];
    $c = $a["color"];
    $n = len($xs);
    // Bar thickness: a fraction of the gap between neighbours, so bars do not
    // touch and a single bar still has a sensible width.
    $gap = 1;
    if ($n > 1) { $gap = abs($xs[1] - $xs[0]); }
    if ($gap <= 0 || !isfinite($gap)) { $gap = 1; }
    $i = 0;
    while ($i < $n) {
        $xv = $xs[$i];
        $yv = $ys[$i];
        if (isfinite($xv) && isfinite($yv)) {
            if ($horiz) {
                $yA = $ax.py($xv - $gap * $a["width"] / 2, $ylim);
                $yB = $ax.py($xv + $gap * $a["width"] / 2, $ylim);
                $xZ = $ax.px(0, $xlim);
                $xV = $ax.px($yv, $xlim);
                $bk.rect(min($xZ, $xV), min($yA, $yB), abs($xV - $xZ), abs($yB - $yA), $c, null, null);
            } else {
                $xA = $ax.px($xv - $gap * $a["width"] / 2, $xlim);
                $xB = $ax.px($xv + $gap * $a["width"] / 2, $xlim);
                $yZ = $ax.py(0, $ylim);
                $yV = $ax.py($yv, $ylim);
                $bk.rect(min($xA, $xB), min($yZ, $yV), abs($xB - $xA), abs($yV - $yZ), $c, null, null);
            }
        }
        $i = $i + 1;
    }
    return null;
}

def _drawLegend($bk, $ax, $x1, $y0) {
    $entries = [];
    $i = 0;
    while ($i < len($ax.artists)) {
        $a = $ax.artists[$i];
        if ($a["label"] != null) { push($entries, [str($a["label"]), $a["color"]]); }
        $i = $i + 1;
    }
    if (len($entries) == 0) { return null; }
    $fs = 11;
    $pad = 7;
    $rowH = 16;
    $swatch = 16;
    $wMax = 0;
    $i = 0;
    while ($i < len($entries)) {
        $w = _textWidth($entries[$i][0], $fs);
        if ($w > $wMax) { $wMax = $w; }
        $i = $i + 1;
    }
    $boxW = $pad * 2 + $swatch + 6 + $wMax;
    $boxH = $pad * 2 + $rowH * len($entries);
    $bx = $x1 - $boxW - 8;
    $by = $y0 + 8;
    $bk.rect($bx, $by, $boxW, $boxH, "#ffffff", $_AXISCOLOR, 1);
    $i = 0;
    while ($i < len($entries)) {
        $ly = $by + $pad + $rowH * $i + $rowH / 2;
        $bk.line($bx + $pad, $ly, $bx + $pad + $swatch, $ly, $entries[$i][1], 3);
        $bk.text($bx + $pad + $swatch + 6, $ly + 4, $entries[$i][0], $fs, $_FG, "start", null);
        $i = $i + 1;
    }
    return null;
}

// ════════════════════════════════════════════════════════════════════════
//  Factories, and the stateful API
//
//  `new alias.Class()` does not parse across a module boundary, so these
//  factory functions are the only way to reach a figure from outside.
// ════════════════════════════════════════════════════════════════════════

def figure($w, $h) { return new BPlotFigure($w, $h); }

// The current figure. Module state, which means it is PER PROCESS and shared
// across sua's request handlers -- each of which runs on its own thread. Two
// concurrent requests plotting into the implicit figure would interleave into
// one chart. Servers should use the object API: $fig = plt.figure(); ...
//
// Held in a DICT rather than a plain variable, and that is not a style
// choice. `$_CUR = figure()` inside a function assigns to a function-LOCAL:
// Environment::assign stops at the nearest function boundary, so the module
// binding would never change and every call would build a new figure. A dict
// is reference-semantic, so writing through a field mutates the one object
// every caller can see. This is the same rule that makes `$s = $s + "x"`
// inside a function leave the global alone.
$_STATE = {"fig": null};

def gcf() {
    if ($_STATE["fig"] == null) { $_STATE["fig"] = figure(null, null); }
    return $_STATE["fig"];
}
def gca() { return gcf().gca(); }
def clf() { $_STATE["fig"] = null; return null; }

def plot($x, $y, $opts)    { return gca().plot($x, $y, $opts); }
def scatter($x, $y, $opts) { return gca().scatter($x, $y, $opts); }
def bar($x, $y, $opts)     { return gca().bar($x, $y, $opts); }
def barh($x, $y, $opts)    { return gca().barh($x, $y, $opts); }

def xlabel($s)      { return gca().setXLabel($s); }
def ylabel($s)      { return gca().setYLabel($s); }
def title($s)       { return gca().setTitle($s); }
def grid($on)       { return gca().setGrid($on); }
def legend($on)     { return gca().setLegend($on); }
def xlim($lo, $hi)  { return gca().setXLim($lo, $hi); }
def ylim($lo, $hi)  { return gca().setYLim($lo, $hi); }

def to_svg()        { return gcf().to_svg(); }
def savefig($path)  { $p = gcf().savefig($path); clf(); return $p; }

// There is no exec, no system and no shell builtin in Bantu, so there is
// nothing to hand a file to. show() writes it and tells you where it is.
def show($path) {
    if ($path == null) { $path = "bplot.svg"; }
    $p = savefig($path);
    print("bplot: wrote " + $p);
    return $p;
}

// ── Discoverability ──────────────────────────────────────────────────────
def help() {
    print("bplot — data visualisation for Bantu (SVG)");
    print("");
    print("  Three lines to a chart:");
    print("    include \"bplot\" as plt;");
    print("    plt.plot([1, 2, 3], [2, 4, 9]);");
    print("    plt.savefig(\"chart.svg\");");
    print("");
    print("  Charts        plot(x, y, opts)    scatter(x, y, opts)");
    print("                bar(x, y, opts)     barh(x, y, opts)");
    print("  Decoration    title(s)   xlabel(s)   ylabel(s)");
    print("                grid(on)   legend(on)  xlim(lo, hi)   ylim(lo, hi)");
    print("  Output        savefig(path)   to_svg()   show(path)   clf()");
    print("  Objects       figure(w, h) -> $fig;  $fig.addAxes() -> $ax");
    print("                every plt.* call above is $ax.<the same thing>");
    print("");
    print("  opts is a dict: {\"color\": \"#1f77b4\", \"width\": 2, \"label\": \"series\"}");
    print("    color   a named colour, \"#rgb\", \"#rrggbb\", or [r, g, b]");
    print("    label   makes the series appear in legend()");
    print("");
    print("  x and y accept a Bantu list, a numba ndarray, or an arctic column.");
    print("");
    print("  Serving charts built from untrusted data? SVG is an executable");
    print("  document format. bplot escapes everything it emits, but serve it");
    print("  from a separate origin or under a strict CSP. See docs/bplot.md.");
    return null;
}

def version() { return "1.0.0"; }
