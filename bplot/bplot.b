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

// ── Palette and theme ────────────────────────────────────────────────────
// The default cycle is matplotlib's tab10: the most-recognised categorical
// palette in data visualisation, chosen for distinguishability rather than for
// looks.

// The active style. A DICT, not seven separate variables, and that is not a
// preference: `$_STYLE["fg"] = "#eee"` inside style() would create a function-LOCAL and
// change nothing, because Environment::assign stops at the nearest function
// boundary. A dict is reference-semantic, so writing through a field mutates
// the one object every caller sees -- the same reason the current figure lives
// in $_STATE.
//
// Minor gridlines are lighter than major ones on purpose: a log decade carries
// eight of them, and at the major weight they read as data rather than as
// background.
$_STYLE = {
    "fg":     "#333333",
    "grid":   "#dddddd",
    "minor":  "#f0f0f0",
    "axis":   "#888888",
    "bg":     "#ffffff",
    "font":   "sans-serif",
    "cycle":  ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
};

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
// ── Native acceleration (decisions BP32, BP33) ───────────────────────────
// bplot is pure Bantu. When the interpreter also has numba and bplot's own
// kernels, numeric data is kept as an ndarray until it becomes pixels, which
// is what took one 1,000,000-point line from ~20 s and 3.1 GB to well under a
// second. The pure path is kept -- for an interpreter without the kernels, and
// as the differential oracle: _useNative(false) forces it, and the tests
// require the same data to render byte-identically both ways.
$_NAT = {"on": false};
def _nativeAvailable() {
    $ok = false;
    try { $ok = has_native("bplot") && has_native("ndarray"); } catch ($e) { $ok = false; }
    return $ok;
}
$_NAT["on"] = _nativeAvailable();

// Returns the previous setting. Turning it on only succeeds where the kernels exist.
def _useNative($on) {
    $was = $_NAT["on"];
    if ($on == true) { $_NAT["on"] = _nativeAvailable(); } else { $_NAT["on"] = false; }
    return $was;
}

def _esc($s) {
    // bp_escape is exactly this function, in C++ (plot_native.hpp). Escaping
    // stays mandatory either way; native only makes it linear in real time --
    // the loop below walks every character in interpreted Bantu, which is fine
    // for a label and minutes for a path holding a million circles.
    if ($_NAT["on"]) { return bp_escape($s); }
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
// Helvetica's advances: what the SVG backend measures with, since an SVG's
// font is whatever the viewer's browser picks for sans-serif. Layout asks the
// BACKEND (textWidth), never a global, so a PNG and an SVG rendered at once on
// two sua threads cannot change each other's layout.
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

// ── Reading data (decisions BP31, BP32, BP34) ────────────────────────────
//
// A sequence may be a Bantu list, a numba ndarray, an arctic column or an
// arctic Series; a table may be an arctic DataFrame. bplot includes neither
// library -- it recognises them by SHAPE, so a plotting library does not
// depend on a dataframe library, and any future type that wraps a column
// works unchanged.

def _kindOf($v) {
    $t = type($v);
    if ($t != "instance") { return $t; }
    // A missing field reads as null, so these probes are safe on any object.
    if (type($v.col) == "column") { return "series"; }
    if (type($v.names) == "list" && type($v.cols) == "dict") { return "frame"; }
    return "instance";
}

// " ('price')" for a Series, so a length mismatch names both sides.
def _nameOf($v) {
    if (_kindOf($v) == "series") { return " ('" + str($v.name) + "')"; }
    return "";
}

def _inList($list, $v) {
    $i = 0;
    while ($i < len($list)) { if ($list[$i] == $v) { return true; } $i = $i + 1; }
    return false;
}

// What a column holds, as far as a chart is concerned.
def _colKind($c) {
    $dt = col_dtype($c);
    if ($dt == "datetime" || $dt == "date") { return $dt; }
    if ($dt == "f64" || $dt == "i64" || $dt == "bool") { return "number"; }
    return "text";
}

// A numeric, datetime or date column as a null-free f64 column (BP34):
// a null is a missing value and becomes NaN -- a gap, not an invented zero.
// Datetimes are epoch milliseconds already; a date is days, scaled to ms.
def _colF64($c) {
    $k = _colKind($c);
    $n = $c;
    if ($k == "datetime") { $n = col_cast($c, "i64"); }
    if ($k == "date") { $n = col_mul(col_cast($c, "f64"), 86400000); }
    $f = col_cast($n, "f64");
    if (col_null_count($f) > 0) { $f = col_fill_null($f, NAN); }
    return $f;
}

// Numeric columns, in frame order: what a table plots by default. Booleans
// and datetimes are left out of the DEFAULT -- nobody asked for them -- but a
// caller may still name one explicitly.
def _numericNames($names, $cols, $except) {
    $out = [];
    $i = 0;
    while ($i < len($names)) {
        $nm = $names[$i];
        $dt = col_dtype($cols[$nm]);
        if ($nm != $except && ($dt == "f64" || $dt == "i64")) { push($out, $nm); }
        $i = $i + 1;
    }
    return $out;
}

// A list's nulls, as the column path treats them: NaN among numbers, the
// label "null" among text. Only walks the list when a null is actually there.
def _listNulls($xs) {
    if (!_hasNull($xs)) { return $xs; }
    $text = false;
    $i = 0;
    while ($i < len($xs)) { if (type($xs[$i]) == "string") { $text = true; break; } $i = $i + 1; }
    $out = [];
    $i = 0;
    while ($i < len($xs)) {
        $e = $xs[$i];
        if ($e == null) { if ($text) { push($out, "null"); } else { push($out, NAN); } }
        else { push($out, $e); }
        $i = $i + 1;
    }
    return $out;
}

// contains() learned list membership in this release; an older interpreter
// answers false for every list, so fall back to a walk there.
def _hasNull($xs) {
    if ($_NAT["on"]) { return contains($xs, null); }
    $i = 0;
    while ($i < len($xs)) { if ($xs[$i] == null) { return true; } $i = $i + 1; }
    return false;
}

def _frameMisuse($what) {
    return "bplot: " + $what + " is a DataFrame -- pass one column, $df.get(\"name\"), " +
           "or draw the whole table with plot_frame($df)";
}

// Any sequence as a Bantu LIST. Used by the chart kinds whose inputs are small
// by nature (bars, steps, stems, pies, grids).
def _seq($v, $what) {
    if ($v == null) { throw "bplot: " + $what + " is null"; }
    $k = _kindOf($v);
    if ($k == "series") { $v = $v.col; $k = "column"; }
    if ($k == "list") { return _listNulls($v); }
    if ($k == "ndarray") { return nd_to_list($v); }
    if ($k == "column") {
        if (_colKind($v) == "text") {
            $raw = col_to_list($v);
            if (!_hasNull($raw)) { return $raw; }
            $out = [];
            $i = 0;
            while ($i < len($raw)) {
                if ($raw[$i] == null) { push($out, "null"); } else { push($out, $raw[$i]); }
                $i = $i + 1;
            }
            return $out;
        }
        return col_to_list(_colF64($v));
    }
    if ($k == "frame") { throw _frameMisuse($what); }
    throw "bplot: " + $what + " must be a list, an ndarray, or an arctic column or Series, got " + $k;
}

// Any sequence in the NATIVE representation (BP32): {"a": 1-D f64 ndarray,
// "date": is-a-time-column}, or null when the data is text and belongs on a
// categorical axis. Only called when _NAT is on.
def _arr($v, $what) {
    if ($v == null) { throw "bplot: " + $what + " is null"; }
    $k = _kindOf($v);
    if ($k == "series") { $v = $v.col; $k = "column"; }
    if ($k == "ndarray") {
        if (len(nd_shape($v)) != 1) {
            throw "bplot: " + $what + " must be 1-dimensional, got an array of shape " + str(nd_shape($v));
        }
        $a = $v;
        if (nd_dtype($v) != "f64") { $a = nd_astype($v, "f64"); }
        return {"a": $a, "date": false};
    }
    if ($k == "column") {
        $ck = _colKind($v);
        if ($ck == "text") { return null; }
        // Zero-copy: the ndarray borrows the filled column's own storage.
        return {"a": nd_from_column(_colF64($v)), "date": $ck == "datetime" || $ck == "date"};
    }
    if ($k == "list") {
        $xs = _listNulls($v);
        $a = null;
        try { $a = nd($xs, "f64"); } catch ($e) { $a = null; }
        // Text, or a mix: the categorical path owns that, and its message.
        if ($a == null) { return null; }
        if (len(nd_shape($a)) != 1) {
            throw "bplot: " + $what + " must be a flat sequence of numbers, got nested lists";
        }
        return {"a": $a, "date": false};
    }
    if ($k == "frame") { throw _frameMisuse($what); }
    throw "bplot: " + $what + " must be a list, an ndarray, or an arctic column or Series, got " + $k;
}

// A sequence of numbers in whichever representation the current mode uses.
def _nums($v, $what) {
    if ($_NAT["on"]) {
        $A = _arr($v, $what);
        if ($A != null) { return $A["a"]; }
    }
    return _seq($v, $what);
}

def _isTimeData($v) {
    $k = _kindOf($v);
    if ($k == "series") { $v = $v.col; $k = "column"; }
    if ($k != "column") { return false; }
    $ck = _colKind($v);
    return $ck == "datetime" || $ck == "date";
}

// Finite limits of an ndarray, as three native passes and no Bantu loop.
def _limitsArr($a) {
    if (nd_size($a) == 0) { return null; }
    $f = nd_compress(nd_isfinite($a), $a);
    if (nd_size($f) == 0) { return null; }
    return [nd_to_list(nd_min($f, null)), nd_to_list(nd_max($f, null))];
}

// Numeric limits over a sequence, skipping NaN and infinities.
//
// The fast path is one native call: min()/max() over a list walk it in C++,
// which is how the limits of a 100,000-point series cost ~10 ms instead of a
// 100,000-iteration Bantu loop. But the language's min/max PROPAGATE NaN by
// design -- a primitive should not silently discard a value it was handed --
// so a NaN or an infinity in the data sends us to the filtering loop.
def _limits($xs) {
    if (type($xs) == "ndarray") { return _limitsArr($xs); }
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
//  Scales (decision BP16)
//
//  A scale is the one NON-AFFINE step in the transform chain. It is applied
//  as a PRE-PASS over the whole sequence, never per point inside a drawing
//  loop: at ~0.38 us per interpreted operation a 100,000-point line would
//  otherwise pay for 200,000 extra calls, and -- worse -- the branch deciding
//  whether to call would sit in the loop, so LINEAR axes would pay for a
//  feature they do not use. Linear is most axes and it stays exactly as fast
//  as it was in B1.
//
//  It also means the non-positive check lives in exactly one function. Four
//  artists each doing their own check is four chances to forget, and the one
//  that forgets emits NaN coordinates -- which browsers render as nothing.
// ════════════════════════════════════════════════════════════════════════

// matplotlib's symlog constant: linscale / (1 - 1/base), with linscale = 1
// and base = 10. Checked against matplotlib.scale.SymmetricalLogTransform.
$_LINADJ = 1.1111111111111112;

def _checkLinthresh($lt, $which) {
    if ($lt == null) { return 1; }
    if (type($lt) != "number" || !isfinite($lt) || $lt <= 0) {
        throw "bplot: " + $which + " symlog linthresh must be a positive finite number, got " + str($lt);
    }
    return $lt;
}

// Forward: data value -> transformed value. Linear returns its argument, so
// the caller can skip the whole pre-pass.
def _fwd($v, $scale, $lt) {
    if ($scale == "linear") { return $v; }
    if ($scale == "log") {
        if ($v <= 0) {
            throw "bplot: a log scale cannot show " + str($v) +
                  " -- log10 of a non-positive value is not a number, and it would " +
                  "reach the document as a NaN coordinate, which browsers render as " +
                  "nothing at all. Use scale \"symlog\" to plot data that crosses zero.";
        }
        return log10($v);
    }
    // symlog: linear inside the band, logarithmic outside, continuous at the
    // join (both branches give lt * _LINADJ there).
    $a = abs($v);
    if ($a <= $lt) { return $v * $_LINADJ; }
    $s = 1;
    if ($v < 0) { $s = 0 - 1; }
    return $s * $lt * ($_LINADJ + log10($a / $lt));
}

// Inverse, needed to turn expanded view limits back into data space so the
// tick locators can work in the units a reader thinks in.
def _inv($u, $scale, $lt) {
    if ($scale == "linear") { return $u; }
    if ($scale == "log") { return pow(10, $u); }
    $band = $lt * $_LINADJ;
    if (abs($u) <= $band) { return $u / $_LINADJ; }
    $s = 1;
    if ($u < 0) { $s = 0 - 1; }
    return $s * $lt * pow(10, abs($u) / $lt - $_LINADJ);
}

// The pre-pass. Returns a NEW list of transformed values; non-finite entries
// pass through untouched so the NaN-splits-the-line rule (BP9) still works.
def _project($vals, $scale, $lt) {
    $out = [];
    $i = 0;
    $n = len($vals);
    while ($i < $n) {
        $v = $vals[$i];
        if (isfinite($v)) { push($out, _fwd($v, $scale, $lt)); }
        else { push($out, $v); }
        $i = $i + 1;
    }
    return $out;
}

// ── Log ticks (decision BP17) ────────────────────────────────────────────
// Majors are the decades in view; minors are 2..9 x each decade. Matches
// matplotlib exactly on every range where matplotlib produces a usable axis.
//
// WHERE IT DELIBERATELY DIFFERS: asked for 2..9, matplotlib returns NO major
// ticks at all -- an axis with no labelled tick anywhere on it. 1..3 gets
// exactly one. So when fewer than two decades fall in view, bplot promotes
// the 2..9 minors to labelled ticks instead. An unlabelled axis is not a
// style difference; it is a chart nobody can read.
def _decadesIn($lo, $hi) {
    $out = [];
    if ($lo <= 0) { return $out; }
    $k = ceil(log10($lo) - 0.000000001);
    $guard = 0;
    while ($guard < 400) {
        $t = pow(10, $k);
        if ($t > $hi * 1.000000001) { break; }
        if ($t >= $lo * 0.999999999) { push($out, $t); }
        $k = $k + 1;
        $guard = $guard + 1;
    }
    return $out;
}

def _subdecadesIn($lo, $hi) {
    $out = [];
    if ($lo <= 0) { return $out; }
    $k = floor(log10($lo));
    $guard = 0;
    while ($guard < 400) {
        $mag = pow(10, $k);
        if ($mag > $hi) { break; }
        $m = 2;
        while ($m <= 9) {
            $t = $m * $mag;
            if ($t >= $lo * 0.999999999 && $t <= $hi * 1.000000001) { push($out, $t); }
            $m = $m + 1;
        }
        $k = $k + 1;
        $guard = $guard + 1;
    }
    return $out;
}

// Returns {"major": [...], "minor": [...]}.
def _ticksLog($lo, $hi) {
    $maj = _decadesIn($lo, $hi);
    $min = _subdecadesIn($lo, $hi);
    if (len($maj) < 2) {
        // Promote. Keep any decade that is in view so 1..3 reads 1,2,3.
        $all = [];
        $i = 0;
        while ($i < len($maj)) { push($all, $maj[$i]); $i = $i + 1; }
        $i = 0;
        while ($i < len($min)) { push($all, $min[$i]); $i = $i + 1; }
        if (len($all) == 0) { return {"major": [$lo], "minor": []}; }
        return {"major": sort($all), "minor": []};
    }
    return {"major": $maj, "minor": $min};
}

// ── symlog ticks ─────────────────────────────────────────────────────────
// Zero, plus +/-10^k for every decade at or above linthresh that is in view.
def _ticksSymlog($lo, $hi, $lt) {
    $maj = [];
    if ($lo <= 0 && $hi >= 0) { push($maj, 0); }
    $k0 = floor(log10($lt) + 0.000000001);
    $k = $k0;
    $guard = 0;
    while ($guard < 400) {
        $t = pow(10, $k);
        if ($t > $hi && 0 - $t < $lo) { break; }
        if ($t >= $lo && $t <= $hi) { push($maj, $t); }
        if (0 - $t >= $lo && 0 - $t <= $hi) { push($maj, 0 - $t); }
        $k = $k + 1;
        $guard = $guard + 1;
    }
    if (len($maj) == 0) { return {"major": [$lo, $hi], "minor": []}; }
    return {"major": sort($maj), "minor": []};
}

// ── Decade labels (architecture section 7.1) ─────────────────────────────
// A plain decimal while one exists, then 1e6 / 1e-7. Not a Unicode
// superscript: those glyphs are missing from many of the fonts sans-serif
// resolves to and render as boxes, and a <tspan baseline-shift> would have to
// be reimplemented by the raster backend for the sake of a label.
def _fmtDecade($v) {
    if ($v == 0) { return "0"; }
    $neg = $v < 0;
    $a = abs($v);
    $k = round(log10($a));
    // Only a true power of ten gets the decade treatment.
    if (abs(pow(10, $k) - $a) > $a * 0.000001) { return _fmt($v, _decimalsFor($a)); }
    $s = "";
    if ($k >= 0 && $k <= 5) { $s = _fmt(pow(10, $k), 0); }
    else { if ($k < 0 && $k >= -4) { $s = _fmt($a, 0 - $k); }
           else { $s = "1e" + _fmt($k, 0); } }
    if ($neg) { return "-" + $s; }
    return $s;
}

// ════════════════════════════════════════════════════════════════════════
//  Colormaps (decision BP25)
//
//  These are the PUBLISHED 256-entry tables, not approximations. There is no
//  closed form for viridis -- it is the output of an optimisation in CAM02-UCS
//  perceptual space -- so a polynomial fit called "viridis" would be a
//  different colormap wearing the name, and perceptual uniformity, the entire
//  reason to use it, is exactly what a fit loses.
//
//  Each map is 256 entries of "rrggbb", 1,536 hex characters, sliced six at a
//  time. A hex string rather than 768 numbers because it is a twelfth of the
//  source size, it cannot be half-edited into a valid-but-wrong table, and
//  parsing costs four ord() calls.
//
//  PROVENANCE. Generated once, at authoring time, from the published tables
//  and pasted in as literals. NOTHING HERE DEPENDS ON ANYTHING AT RUN TIME.
//    viridis, plasma  -- Nathaniel Smith and Stefan van der Walt, released
//                        into the PUBLIC DOMAIN (CC0). https://bids.github.io/colormap/
//    coolwarm         -- Kenneth Moreland's diverging map, as matplotlib
//                        samples it. https://www.kennethmoreland.com/color-maps/
//    gray             -- linear, and exact by definition.
//
//  jet is deliberately absent (BP13): it is not perceptually uniform, it
//  invents a band of structure that is not in the data, and it is unreadable
//  in greyscale and under the common colour-vision deficiencies. Shipping it
//  would mean people using it.
// ════════════════════════════════════════════════════════════════════════

$_CM_VIRIDIS =
    "44015444025645045745055946075a46085c460a5d460b5e470d60470e61471063471164471365481467481668481769" +
    "48186a481a6c481b6d481c6e481d6f481f70482071482173482374482475482576482677482878482979472a7a472c7a" +
    "472d7b472e7c472f7d46307e46327e46337f463480453581453781453882443983443a83443b84433d84433e85423f85" +
    "4240864241864142874144874045884046883f47883f48893e49893e4a893e4c8a3d4d8a3d4e8a3c4f8a3c508b3b518b" +
    "3b528b3a538b3a548c39558c39568c38588c38598c375a8c375b8d365c8d365d8d355e8d355f8d34608d34618d33628d" +
    "33638d32648e32658e31668e31678e31688e30698e306a8e2f6b8e2f6c8e2e6d8e2e6e8e2e6f8e2d708e2d718e2c718e" +
    "2c728e2c738e2b748e2b758e2a768e2a778e2a788e29798e297a8e297b8e287c8e287d8e277e8e277f8e27808e26818e" +
    "26828e26828e25838e25848e25858e24868e24878e23888e23898e238a8d228b8d228c8d228d8d218e8d218f8d21908d" +
    "21918c20928c20928c20938c1f948c1f958b1f968b1f978b1f988b1f998a1f9a8a1e9b8a1e9c891e9d891f9e891f9f88" +
    "1fa0881fa1881fa1871fa28720a38620a48621a58521a68522a78522a88423a98324aa8325ab8225ac8226ad8127ad81" +
    "28ae8029af7f2ab07f2cb17e2db27d2eb37c2fb47c31b57b32b67a34b67935b77937b87838b9773aba763bbb753dbc74" +
    "3fbc7340bd7242be7144bf7046c06f48c16e4ac16d4cc26c4ec36b50c46a52c56954c56856c66758c7655ac8645cc863" +
    "5ec96260ca6063cb5f65cb5e67cc5c69cd5b6ccd5a6ece5870cf5773d05675d05477d1537ad1517cd2507fd34e81d34d" +
    "84d44b86d54989d5488bd6468ed64590d74393d74195d84098d83e9bd93c9dd93ba0da39a2da37a5db36a8db34aadc32" +
    "addc30b0dd2fb2dd2db5de2bb8de29bade28bddf26c0df25c2df23c5e021c8e020cae11fcde11dd0e11cd2e21bd5e21a" +
    "d8e219dae319dde318dfe318e2e418e5e419e7e419eae51aece51befe51cf1e51df4e61ef6e620f8e621fbe723fde725";

$_CM_PLASMA =
    "0d088710078813078916078a19068c1b068d1d068e20068f2206902406912605912805922a05932c05942e05952f0596" +
    "31059733059735049837049938049a3a049a3c049b3e049c3f049c41049d43039e44039e46039f48039f4903a04b03a1" +
    "4c02a14e02a25002a25102a35302a35502a45601a45801a45901a55b01a55c01a65e01a66001a66100a76300a76400a7" +
    "6600a76700a86900a86a00a86c00a86e00a86f00a87100a87201a87401a87501a87701a87801a87a02a87b02a87d03a8" +
    "7e03a88004a88104a78305a78405a78606a68707a68808a68a09a58b0aa58d0ba58e0ca48f0da4910ea3920fa39410a2" +
    "9511a19613a19814a099159f9a169f9c179e9d189d9e199da01a9ca11b9ba21d9aa31e9aa51f99a62098a72197a82296" +
    "aa2395ab2494ac2694ad2793ae2892b02991b12a90b22b8fb32c8eb42e8db52f8cb6308bb7318ab83289ba3388bb3488" +
    "bc3587bd3786be3885bf3984c03a83c13b82c23c81c33d80c43e7fc5407ec6417dc7427cc8437bc9447aca457acb4679" +
    "cc4778cc4977cd4a76ce4b75cf4c74d04d73d14e72d24f71d35171d45270d5536fd5546ed6556dd7566cd8576bd9586a" +
    "da5a6ada5b69db5c68dc5d67dd5e66de5f65de6164df6263e06363e16462e26561e26660e3685fe4695ee56a5de56b5d" +
    "e66c5ce76e5be76f5ae87059e97158e97257ea7457eb7556eb7655ec7754ed7953ed7a52ee7b51ef7c51ef7e50f07f4f" +
    "f0804ef1814df1834cf2844bf3854bf3874af48849f48948f58b47f58c46f68d45f68f44f79044f79143f79342f89441" +
    "f89540f9973ff9983ef99a3efa9b3dfa9c3cfa9e3bfb9f3afba139fba238fca338fca537fca636fca835fca934fdab33" +
    "fdac33fdae32fdaf31fdb130fdb22ffdb42ffdb52efeb72dfeb82cfeba2cfebb2bfebd2afebe2afec029fdc229fdc328" +
    "fdc527fdc627fdc827fdca26fdcb26fccd25fcce25fcd025fcd225fbd324fbd524fbd724fad824fada24f9dc24f9dd25" +
    "f8df25f8e125f7e225f7e425f6e626f6e826f5e926f5eb27f4ed27f3ee27f3f027f2f227f1f426f1f525f0f724f0f921";

$_CM_COOLWARM =
    "3b4cc03c4ec23d50c33e51c53f53c64055c84257c94358cb445acc455cce465ecf485fd14961d24a63d34b64d54c66d6" +
    "4e68d84f69d9506bda516ddb536edd5470de5572df5673e05875e15977e35a78e45b7ae55d7ce65e7de75f7fe86180e9" +
    "6282ea6384eb6485ec6687ed6788ee688aef6a8bef6b8df06c8ff16e90f26f92f37093f37295f47396f57597f67699f6" +
    "779af7799cf87a9df87b9ff97da0f97ea1fa80a3fa81a4fb82a6fb84a7fc85a8fc86a9fc88abfd89acfd8badfd8caffe" +
    "8db0fe8fb1fe90b2fe92b4fe93b5fe94b6ff96b7ff97b8ff98b9ff9abbff9bbcff9dbdff9ebeff9fbfffa1c0ffa2c1ff" +
    "a3c2fea5c3fea6c4fea7c5fea9c6fdaac7fdabc8fdadc9fdaec9fcafcafcb1cbfcb2ccfbb3cdfbb5cdfab6cefab7cff9" +
    "b9d0f9bad0f8bbd1f8bcd2f7bed2f6bfd3f6c0d4f5c1d4f4c3d5f4c4d5f3c5d6f2c6d6f1c7d7f0c9d7f0cad8efcbd8ee" +
    "ccd9edcdd9eccedaebcfdaead1dae9d2dbe8d3dbe7d4dbe6d5dbe5d6dce4d7dce3d8dce2d9dce1dadce0dbdcdedcdddd" +
    "dddcdcdedcdbdfdbd9e0dbd8e1dad6e2dad5e3d9d3e4d9d2e5d8d1e6d7cfe7d7cee8d6cce9d5cbead5c9ead4c8ebd3c6" +
    "ecd3c5edd2c3edd1c2eed0c0efcfbfefcebdf0cdbbf1cdbaf1ccb8f2cbb7f2cab5f2c9b4f3c8b2f3c7b1f4c6aff4c5ad" +
    "f5c4acf5c2aaf5c1a9f5c0a7f6bfa6f6bea4f6bda2f7bca1f7ba9ff7b99ef7b89cf7b79bf7b599f7b497f7b396f7b194" +
    "f7b093f7af91f7ad90f7ac8ef7aa8cf7a98bf7a889f7a688f6a586f6a385f6a283f5a081f59f80f59d7ef59c7df49a7b" +
    "f4987af39778f39577f39475f29274f29072f18f71f18d6ff08b6ef08a6cef886bee8669ee8468ed8366ec8165ec7f63" +
    "eb7d62ea7b60e97a5fe9785de8765ce7745be67259e57058e46e56e36c55e36b54e26952e16751e0654fdf634ede614d" +
    "dd5f4bdc5d4ada5a49d95847d85646d75445d65244d55042d44e41d24b40d1493fd0473dcf453ccd423bcc403acb3e38" +
    "ca3b37c83836c73635c53334c43032c32e31c12b30c0282fbe242ebd1f2dbb1b2cba162bb8122ab70d28b50927b40426";

$_CM_GRAY =
    "0000000101010202020303030404040505050606060707070808080909090a0a0a0b0b0b0c0c0c0d0d0d0e0e0e0f0f0f" +
    "1010101111111212121313131414141515151616161717171818181919191a1a1a1b1b1b1c1c1c1d1d1d1e1e1e1f1f1f" +
    "2020202121212222222323232424242525252626262727272828282929292a2a2a2b2b2b2c2c2c2d2d2d2e2e2e2f2f2f" +
    "3030303131313232323333333434343535353636363737373838383939393a3a3a3b3b3b3c3c3c3d3d3d3e3e3e3f3f3f" +
    "4040404141414242424343434444444545454646464747474848484949494a4a4a4b4b4b4c4c4c4d4d4d4e4e4e4f4f4f" +
    "5050505151515252525353535454545555555656565757575858585959595a5a5a5b5b5b5c5c5c5d5d5d5e5e5e5f5f5f" +
    "6060606161616262626363636464646565656666666767676868686969696a6a6a6b6b6b6c6c6c6d6d6d6e6e6e6f6f6f" +
    "7070707171717272727373737474747575757676767777777878787979797a7a7a7b7b7b7c7c7c7d7d7d7e7e7e7f7f7f" +
    "8080808181818282828383838484848585858686868787878888888989898a8a8a8b8b8b8c8c8c8d8d8d8e8e8e8f8f8f" +
    "9090909191919292929393939494949595959696969797979898989999999a9a9a9b9b9b9c9c9c9d9d9d9e9e9e9f9f9f" +
    "a0a0a0a1a1a1a2a2a2a3a3a3a4a4a4a5a5a5a6a6a6a7a7a7a8a8a8a9a9a9aaaaaaabababacacacadadadaeaeaeafafaf" +
    "b0b0b0b1b1b1b2b2b2b3b3b3b4b4b4b5b5b5b6b6b6b7b7b7b8b8b8b9b9b9babababbbbbbbcbcbcbdbdbdbebebebfbfbf" +
    "c0c0c0c1c1c1c2c2c2c3c3c3c4c4c4c5c5c5c6c6c6c7c7c7c8c8c8c9c9c9cacacacbcbcbcccccccdcdcdcecececfcfcf" +
    "d0d0d0d1d1d1d2d2d2d3d3d3d4d4d4d5d5d5d6d6d6d7d7d7d8d8d8d9d9d9dadadadbdbdbdcdcdcdddddddedededfdfdf" +
    "e0e0e0e1e1e1e2e2e2e3e3e3e4e4e4e5e5e5e6e6e6e7e7e7e8e8e8e9e9e9eaeaeaebebebecececedededeeeeeeefefef" +
    "f0f0f0f1f1f1f2f2f2f3f3f3f4f4f4f5f5f5f6f6f6f7f7f7f8f8f8f9f9f9fafafafbfbfbfcfcfcfdfdfdfefefeffffff";

def _cmapTable($name) {
    if ($name == null) { $name = "viridis"; }
    if ($name == "viridis")  { return $_CM_VIRIDIS; }
    if ($name == "plasma")   { return $_CM_PLASMA; }
    if ($name == "coolwarm") { return $_CM_COOLWARM; }
    if ($name == "gray" || $name == "grey") { return $_CM_GRAY; }
    throw "bplot: unknown colormap \"" + str($name) +
          "\" -- available: viridis, plasma, coolwarm, gray. " +
          "jet is deliberately not shipped: it is not perceptually uniform and " +
          "invents structure that is not in the data.";
}

// A colour from a map at position t in [0, 1]. Out-of-range clamps, as
// matplotlib's `over`/`under` do by default.
//
// NaN gets its own colour rather than a clamp. A missing cell painted as the
// colormap's minimum is indistinguishable from a real minimum, which is the
// most dangerous thing a heatmap can do: it reports data where there is none.
$_CM_BAD = "#b0b0b0";

def _cmap($name, $t) {
    if (isnan($t)) { return $_CM_BAD; }
    $tab = _cmapTable($name);
    // floor(t * 256) clamped to 255 -- matplotlib's own quantisation, not the
    // round(t * 255) that looks equivalent and is not. The two agree at 0,
    // 0.5 and 1 and disagree at 0.625, where round gives entry 159 and
    // matplotlib gives 160 -- a visibly different green. Checked entry by
    // entry against the published tables.
    $i = floor(clamp($t, 0, 1) * 256);
    if ($i > 255) { $i = 255; }
    return "#" + substr($tab, $i * 6, 6);
}

// ════════════════════════════════════════════════════════════════════════
//  Date axes (decision BP22)
//
//  x values are EPOCH MILLISECONDS, UTC -- which is exactly how arctic stores
//  a DATETIME column (dataframe_native.hpp:77), so an arctic column plots
//  without conversion. Everything here is UTC: local time needs a timezone
//  database, and that is not a dependency this tree will take for an axis
//  label.
//
//  Why not just use _ticks() on the milliseconds? Because MaxNLocator would
//  propose a step of 2,500,000,000 ms -- a tick every 28.9 days, landing
//  mid-afternoon on drifting dates. And months and years are not fixed-length,
//  so no amount of millisecond arithmetic can produce "the first of each
//  month". Ticks are ANCHORED to calendar boundaries, never accumulated.
// ════════════════════════════════════════════════════════════════════════

// Howard Hinnant's civil_from_days: exact over the proleptic Gregorian
// calendar, no lookup table, no leap-year special cases beyond shifting the
// year so it starts in March (which puts the leap day last).
def _civilFromDays($z) {
    $z = $z + 719468;
    $era = floor($z / 146097);
    if ($z < 0) { $era = floor(($z - 146096) / 146097); }
    $doe = $z - $era * 146097;                                   // [0, 146096]
    $yoe = floor(($doe - floor($doe / 1460) + floor($doe / 36524) - floor($doe / 146096)) / 365);
    $y = $yoe + $era * 400;
    $doy = $doe - (365 * $yoe + floor($yoe / 4) - floor($yoe / 100));
    $mp = floor((5 * $doy + 2) / 153);
    $d = $doy - floor((153 * $mp + 2) / 5) + 1;
    $m = $mp + 3;
    if ($mp >= 10) { $m = $mp - 9; }
    if ($m <= 2) { $y = $y + 1; }
    return [$y, $m, $d];
}

def _daysFromCivil($y, $m, $d) {
    if ($m <= 2) { $y = $y - 1; }
    $era = floor($y / 400);
    $yoe = $y - $era * 400;
    $mp = $m - 3;
    if ($m <= 2) { $mp = $m + 9; }
    $doy = floor((153 * $mp + 2) / 5) + $d - 1;
    $doe = $yoe * 365 + floor($yoe / 4) - floor($yoe / 100) + $doy;
    return $era * 146097 + $doe - 719468;
}

$_MSDAY = 86400000;

// Split epoch ms into [y, mo, d, h, mi, s, ms].
def _partsOfMs($ms) {
    $days = floor($ms / $_MSDAY);
    $rem = $ms - $days * $_MSDAY;
    $c = _civilFromDays($days);
    $h = floor($rem / 3600000);
    $rem = $rem - $h * 3600000;
    $mi = floor($rem / 60000);
    $rem = $rem - $mi * 60000;
    $s = floor($rem / 1000);
    return [$c[0], $c[1], $c[2], $h, $mi, $s, $rem - $s * 1000];
}

def _msOfCivil($y, $mo, $d, $h, $mi, $s) {
    return _daysFromCivil($y, $mo, $d) * $_MSDAY + $h * 3600000 + $mi * 60000 + $s * 1000;
}

def _pad2($n) {
    if ($n < 10) { return "0" + _fmt($n, 0); }
    return _fmt($n, 0);
}

// Label format follows the step, which is what keeps a date axis readable:
// a seconds axis does not need the year, and a years axis does not need 00:00.
def _fmtDate($ms, $unit) {
    $p = _partsOfMs($ms);
    $ymd = _fmt($p[0], 0) + "-" + _pad2($p[1]) + "-" + _pad2($p[2]);
    $hms = _pad2($p[3]) + ":" + _pad2($p[4]) + ":" + _pad2($p[5]);
    if ($unit == "second") { return $hms; }
    // Minute and hour labels carry the DAY, as matplotlib's defaults do
    // (date.autoformatter.minute '%d %H:%M', .hour '%m-%d %H'). They used to
    // be a bare "HH:MM", so any axis crossing midnight read "12:00 00:00 12:00"
    // with no way to tell which day a tick belonged to -- found in B4, when a
    // four-day datetime column produced exactly that.
    if ($unit == "minute") { return _pad2($p[2]) + " " + _pad2($p[3]) + ":" + _pad2($p[4]); }
    if ($unit == "hour")   { return _pad2($p[1]) + "-" + _pad2($p[2]) + " " + _pad2($p[3]); }
    if ($unit == "day")    { return $ymd; }
    if ($unit == "month")  { return _fmt($p[0], 0) + "-" + _pad2($p[1]); }
    if ($unit == "year")   { return _fmt($p[0], 0); }
    return $ymd + " " + $hms;
}

// The ladder, and the rule for choosing from it, are matplotlib's
// AutoDateLocator reproduced rather than approximated -- the algorithm turned
// out to be four lines, and reproducing it is strictly better than a
// look-alike that disagrees on ranges nobody thought to check:
//
//   walk the units coarsest first; take the FIRST whose span is at least
//   minticks (5) units; within it take the SMALLEST interval satisfying
//   span <= interval * (maxticks - 1).
//
// Verified against matplotlib on eight ranges from ten seconds to 25 years.
$_DATEUNITS  = ["year", "month", "day", "hour", "minute", "second"];
$_DATEMAX    = [11, 12, 11, 12, 11, 11];
$_DATEIVALS  = [[1, 2, 4, 5, 10, 20, 40, 50, 100, 200, 400, 500, 1000, 2000],
                [1, 2, 3, 4, 6],
                [1, 2, 3, 7, 14, 21],
                [1, 2, 3, 4, 6, 12],
                [1, 5, 10, 15, 30],
                [1, 5, 10, 15, 30]];

// Days of the month a day-interval lands on. matplotlib anchors day ticks to
// the CALENDAR rather than stepping a fixed number of milliseconds, so a
// fortnightly axis reads 1st and 15th of each month instead of drifting
// through them -- and 7 gives the 1st, 8th, 15th and 22nd.
def _dayAnchors($step) {
    if ($step == 14) { return [1, 15]; }
    if ($step == 21) { return [1, 22]; }
    if ($step == 7)  { return [1, 8, 15, 22]; }
    $out = [];
    $d = 1;
    while ($d <= 31) { push($out, $d); $d = $d + $step; }
    return $out;
}

def _daysInMonth($y, $m) {
    $n = 31;
    if ($m == 4 || $m == 6 || $m == 9 || $m == 11) { $n = 30; }
    if ($m == 2) {
        $n = 28;
        if (($y - floor($y / 4) * 4) == 0 && (($y - floor($y / 100) * 100) != 0 ||
            ($y - floor($y / 400) * 400) == 0)) { $n = 29; }
    }
    return $n;
}

def _dateTicksFor($lo, $hi, $unit, $step) {
    $out = [];
    if ($unit == "second" || $unit == "minute" || $unit == "hour") {
        $ms = 1000;
        if ($unit == "minute") { $ms = 60000; }
        if ($unit == "hour")   { $ms = 3600000; }
        $ms = $ms * $step;
        // Anchor to midnight UTC, not to the epoch: a 6-hour step must land on
        // 00/06/12/18, which epoch-relative arithmetic gives only by accident
        // of the epoch being midnight. Say it explicitly.
        $day0 = floor($lo / $_MSDAY) * $_MSDAY;
        $k = ceil(($lo - $day0) / $ms);
        $guard = 0;
        while ($guard < 5000) {
            $t = $day0 + $k * $ms;
            if ($t > $hi) { break; }
            if ($t >= $lo) { push($out, $t); }
            $k = $k + 1;
            $guard = $guard + 1;
        }
        return $out;
    }
    // Calendar steps: walk months and years, which no millisecond count can do.
    $p = _partsOfMs($lo);
    $y = $p[0];
    $mo = 1;
    if ($unit == "year") {
        // Years land on multiples of the interval, so a 4-year axis reads
        // 2000, 2004, 2008 rather than 2001, 2005, 2009.
        $y = ceil($y / $step) * $step;
    } else {
        $y = $p[0];
        $mo = 1;
    }
    $anchors = [1];
    if ($unit == "day") { $anchors = _dayAnchors($step); }
    $guard = 0;
    while ($guard < 5000) {
        $t0 = _msOfCivil($y, $mo, 1, 0, 0, 0);
        if ($t0 > $hi) { break; }
        if ($unit == "year") {
            if ($t0 >= $lo) { push($out, $t0); }
            $y = $y + $step;
        } else {
            if ($unit == "month") {
                // Months land on multiples of the interval counted from
                // January, so a quarterly axis reads Jan/Apr/Jul/Oct.
                if (($mo - 1) - floor(($mo - 1) / $step) * $step == 0 && $t0 >= $lo) { push($out, $t0); }
                $mo = $mo + 1;
            } else {
                $dim = _daysInMonth($y, $mo);
                $ai = 0;
                while ($ai < len($anchors)) {
                    $d = $anchors[$ai];
                    if ($d <= $dim) {
                        $t = _msOfCivil($y, $mo, $d, 0, 0, 0);
                        if ($t >= $lo && $t <= $hi) { push($out, $t); }
                    }
                    $ai = $ai + 1;
                }
                $mo = $mo + 1;
            }
            if ($mo > 12) { $mo = 1; $y = $y + 1; }
        }
        $guard = $guard + 1;
    }
    return $out;
}

// How many whole units of each kind does this range span?
def _dateSpans($lo, $hi) {
    $a = _partsOfMs($lo);
    $b = _partsOfMs($hi);
    $months = ($b[0] - $a[0]) * 12 + ($b[1] - $a[1]);
    // A partial final month does not count as a whole one.
    if ($b[2] < $a[2]) { $months = $months - 1; }
    $ms = $hi - $lo;
    return [floor($months / 12), $months, floor($ms / $_MSDAY),
            floor($ms / 3600000), floor($ms / 60000), floor($ms / 1000)];
}

def _dateTicks($lo, $hi) {
    if (!isfinite($lo) || !isfinite($hi) || $hi <= $lo) {
        return {"ticks": [$lo], "unit": "day"};
    }
    $spans = _dateSpans($lo, $hi);
    $u = 0;
    while ($u < 6) {
        $span = $spans[$u];
        if ($span >= 5) {
            $ivals = $_DATEIVALS[$u];
            $maxt = $_DATEMAX[$u];
            $pick = $ivals[len($ivals) - 1];
            $k = 0;
            while ($k < len($ivals)) {
                if ($span <= $ivals[$k] * ($maxt - 1)) { $pick = $ivals[$k]; break; }
                $k = $k + 1;
            }
            $t = _dateTicksFor($lo, $hi, $_DATEUNITS[$u], $pick);
            if (len($t) > 0) { return {"ticks": $t, "unit": $_DATEUNITS[$u]}; }
        }
        $u = $u + 1;
    }
    // Under five seconds there is no calendar unit left; fall back to the
    // ordinary nice-number locator over the milliseconds themselves.
    return {"ticks": _ticks($lo, $hi, 6), "unit": "full"};
}

// ════════════════════════════════════════════════════════════════════════
//  Statistics the chart types are made of (decisions BP19-BP21)
// ════════════════════════════════════════════════════════════════════════

// Finite values only, ascending. sort() is a native builtin as of this phase
// -- it did not exist before, which is why every quantile in every Bantu
// program used to be an interpreted sort.
def _finiteSorted($xs, $what) {
    if (type($xs) == "ndarray") {
        $fa = nd_compress(nd_isfinite($xs), $xs);
        if (nd_size($fa) == 0) {
            throw "bplot." + $what + ": no finite values to summarise";
        }
        return nd_sort($fa, null);
    }
    $f = [];
    $i = 0;
    $n = len($xs);
    while ($i < $n) {
        if (isfinite($xs[$i])) { push($f, $xs[$i]); }
        $i = $i + 1;
    }
    if (len($f) == 0) {
        throw "bplot." + $what + ": no finite values to summarise";
    }
    return sort($f);
}

// NumPy's default quantile (method "linear", Hyndman-Fan type 7) -- which is
// precisely what numba's nd_quantile computes. There are nine defensible
// definitions; the one that matters is the one the rest of the tree uses,
// because a median that moves depending on whether the data arrived as a list
// or an ndarray is worse than any of the nine. Input must already be sorted.
def _quantile($v, $q) {
    $n = len($v);
    if ($n == 0) { return NAN; }
    if ($n == 1) { return $v[0]; }
    $pos = $q * ($n - 1);
    $lo = floor($pos);
    $hi = ceil($pos);
    if ($lo == $hi) { return $v[$lo]; }
    return $v[$lo] + ($v[$hi] - $v[$lo]) * ($pos - $lo);
}

// Binning that matches numba's nd_histogram decision for decision (BP19):
// floor((x-lo)/width); the TOP EDGE IS INCLUSIVE; out-of-range dropped, not
// clamped; NaN dropped; a degenerate range widened by +/-0.5.
//
// The inclusive top edge is the one that bites. Without it the largest value
// in the data silently vanishes, and a histogram missing its maximum lies
// about its range.
def _histCounts($xs, $bins, $lo, $hi) {
    // nd_histogram IS the rule below, decision for decision (BP19).
    if (type($xs) == "ndarray") { return nd_to_list(nd_histogram($xs, $bins, $lo, $hi)); }
    $counts = [];
    $i = 0;
    while ($i < $bins) { push($counts, 0); $i = $i + 1; }
    $width = ($hi - $lo) / $bins;
    $i = 0;
    $n = len($xs);
    while ($i < $n) {
        $x = $xs[$i];
        if (isfinite($x) && $x >= $lo && $x <= $hi) {
            $b = floor(($x - $lo) / $width);
            if ($b >= $bins) { $b = $bins - 1; }
            if ($b < 0) { $b = 0; }
            $counts[$b] = $counts[$b] + 1;
        }
        $i = $i + 1;
    }
    return $counts;
}

def _histRange($xs, $lo, $hi) {
    if ($lo != null && $hi != null) {
        if (!($lo < $hi)) { throw "bplot.hist: range lo must be below hi, got " + str($lo) + " and " + str($hi); }
        return [$lo, $hi];
    }
    $lim = _limits($xs);
    if ($lim == null) { return [0, 1]; }
    if ($lim[0] == $lim[1]) { return [$lim[0] - 0.5, $lim[1] + 0.5]; }
    return $lim;
}

// Binned kernel density estimate (BP21). Binning first makes the cost
// O(n + bins*grid) instead of O(n*grid): the textbook form is 12.8 million
// interpreted operations for 100k points on a 128-point grid, which is most
// of a minute for one violin. The error is bounded by the bin width, far
// below the bandwidth the kernel is already smoothing with. R's density()
// makes the same approximation, via an FFT.
def _kde($sorted, $grid) {
    $n = len($sorted);
    $lo = $sorted[0];
    $hi = $sorted[$n - 1];
    if ($hi <= $lo) { return null; }               // no spread: caller draws a line
    // Scott's rule, matching scipy.stats.gaussian_kde's default factor.
    $mean = 0;
    $ss = 0;
    if (type($sorted) == "ndarray") {
        // The last element of a cumulative sum IS the left-to-right sum the
        // loop below computes -- not nd_sum, which is pairwise and rounds
        // differently. Same order, same bits, same violin both ways.
        $cs = nd_cumsum($sorted, null);
        $mean = $cs[$n - 1] / $n;
        $dev = nd_subtract($sorted, $mean, null);
        $sq = nd_cumsum(nd_multiply($dev, $dev, null), null);
        $ss = $sq[$n - 1];
    } else {
        $i = 0;
        while ($i < $n) { $mean = $mean + $sorted[$i]; $i = $i + 1; }
        $mean = $mean / $n;
        $i = 0;
        while ($i < $n) { $d = $sorted[$i] - $mean; $ss = $ss + $d * $d; $i = $i + 1; }
    }
    $sd = sqrt($ss / $n);
    if ($sd <= 0) { return null; }
    $bw = 1.06 * $sd * pow($n, 0 - 0.2);
    if ($bw <= 0) { return null; }

    $nb = 512;
    $bw2 = 2 * $bw * $bw;
    $bins = _histCounts($sorted, $nb, $lo, $hi);
    $bwidth = ($hi - $lo) / $nb;

    // Pad the grid by three bandwidths so the tails close instead of being
    // chopped flat at the data extremes.
    $glo = $lo - 3 * $bw;
    $ghi = $hi + 3 * $bw;
    $xs = [];
    $ys = [];
    $g = 0;
    while ($g < $grid) {
        $x = $glo + ($ghi - $glo) * $g / ($grid - 1);
        $acc = 0;
        $b = 0;
        while ($b < $nb) {
            $c = $bins[$b];
            if ($c > 0) {
                $d = $x - ($lo + ($b + 0.5) * $bwidth);
                $acc = $acc + $c * exp(0 - $d * $d / $bw2);
            }
            $b = $b + 1;
        }
        push($xs, $x);
        push($ys, $acc / ($n * $bw * sqrt(2 * PI)));
        $g = $g + 1;
    }
    return {"x": $xs, "y": $ys};
}

// Tukey's five-number summary plus outliers, as matplotlib draws it. The
// whiskers land on the most extreme OBSERVED value inside 1.5*IQR -- not on
// the fence itself, which is the frequent error and draws a whisker into
// empty space where no observation exists.
def _boxStats($sorted) {
    $q1 = _quantile($sorted, 0.25);
    $q2 = _quantile($sorted, 0.5);
    $q3 = _quantile($sorted, 0.75);
    $iqr = $q3 - $q1;
    $floLim = $q1 - 1.5 * $iqr;
    $fhiLim = $q3 + 1.5 * $iqr;
    $wlo = $q1;
    $whi = $q3;
    $out = [];
    if (type($sorted) == "ndarray") {
        // The loop below, as masks: inside the fences sets the whiskers,
        // outside is an outlier, kept in sorted order.
        $inside = nd_logical_and(nd_greater_equal($sorted, $floLim, null),
                                 nd_less_equal($sorted, $fhiLim, null), null);
        $in = nd_compress($inside, $sorted);
        if (nd_size($in) > 0) {
            $mn = nd_to_list(nd_min($in, null));
            $mx = nd_to_list(nd_max($in, null));
            if ($mn < $wlo) { $wlo = $mn; }
            if ($mx > $whi) { $whi = $mx; }
        }
        $out = nd_to_list(nd_compress(nd_logical_not($inside, null), $sorted));
        return {"q1": $q1, "med": $q2, "q3": $q3, "wlo": $wlo, "whi": $whi, "out": $out};
    }
    $i = 0;
    $n = len($sorted);
    while ($i < $n) {
        $v = $sorted[$i];
        if ($v < $floLim || $v > $fhiLim) { push($out, $v); }
        else {
            if ($v < $wlo) { $wlo = $v; }
            if ($v > $whi) { $whi = $v; }
        }
        $i = $i + 1;
    }
    return {"q1": $q1, "med": $q2, "q3": $q3, "wlo": $wlo, "whi": $whi, "out": $out};
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

    // A closed filled shape: violins, filled bands, arrowheads, pie slices
    // that happen to be polygons. pts is flat [x0,y0,x1,y1,...].
    def polygon($pts, $fill, $stroke, $sw, $opacity) {
        if (len($pts) < 6) { return $this; }
        $coords = [];
        $i = 0;
        while ($i + 1 < len($pts)) {
            push($coords, _px($pts[$i]) + "," + _px($pts[$i + 1]));
            $i = $i + 2;
        }
        $s = "<polygon points=\"" + join($coords, " ") + "\" fill=\"" + _esc($fill) + "\"";
        if ($opacity != null) { $s = $s + " fill-opacity=\"" + _px($opacity) + "\""; }
        if ($stroke != null) {
            $s = $s + " stroke=\"" + _esc($stroke) + "\" stroke-width=\"" + _px($sw) + "\"";
        }
        push($this.parts, $s + "/>");
        return $this;
    }

    // An arbitrary path. `d` is the ONE parameter in this backend that could
    // in principle carry markup, so it is not a free-text field: every `d`
    // reaching here is assembled by bplot's own arc/rect helpers out of
    // numbers that have already been through _px, and it is escaped like any
    // other attribute value. There is still no route from user text to a `d`.
    def path($d, $fill, $stroke, $sw) {
        $s = "<path d=\"" + _esc($d) + "\" fill=\"" + _esc($fill) + "\"";
        if ($stroke != null) {
            $s = $s + " stroke=\"" + _esc($stroke) + "\" stroke-width=\"" + _px($sw) + "\"";
        }
        push($this.parts, $s + "/>");
        return $this;
    }

    // anchor: "start" | "middle" | "end"
    def text($x, $y, $s, $size, $fill, $anchor, $rotate) {
        $t = "<text x=\"" + _px($x) + "\" y=\"" + _px($y) + "\" font-family=\"" +
             $_STYLE["font"] + "\" font-size=\"" + _px($size) + "\" fill=\"" + _esc($fill) +
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
    def textWidth($s, $size) { return _textWidth($s, $size); }
}

// ════════════════════════════════════════════════════════════════════════
//  The raster backend (B6e)
//
//  The same method set as BPlotSvg, over the native canvas: every call is one
//  bp_* builtin with the SVG backend's own arguments, so the PNG draws the
//  SVG's numbers. There is no markup here, so nothing to escape -- the
//  natives validate colours and numbers themselves.
// ════════════════════════════════════════════════════════════════════════
def _hex6($c) {
    if ($c == null) { return "none"; }
    if (len($c) == 4 && $c[0] == "#") { return "#" + $c[1] + $c[1] + $c[2] + $c[2] + $c[3] + $c[3]; }
    return $c;
}

class BPlotRaster {
    def init($w, $h, $dpi) {
        $this.w = $w;
        $this.h = $h;
        $this.cv = bp_canvas_new($w, $h, $dpi, "#ffffff");
        $this.clipRect = null;
    }
    def open() { return $this; }
    def close() { return $this; }

    def rect($x, $y, $w, $h, $fill, $stroke, $sw) {
        bp_fill_rect($this.cv, $x, $y, $w, $h, _hex6($fill), null);
        if ($stroke != null) {
            bp_stroke_polyline($this.cv, [$x, $y, $x + $w, $y, $x + $w, $y + $h, $x, $y + $h, $x, $y],
                               _hex6($stroke), $sw, null, null, true);
        }
        return $this;
    }
    // SVG's <line> has butt caps.
    def line($x1, $y1, $x2, $y2, $stroke, $sw) {
        bp_stroke_polyline($this.cv, [$x1, $y1, $x2, $y2], _hex6($stroke), $sw, null, null, false);
        return $this;
    }
    // Formatted with _px, as the SVG backend formats it: str() keeps only six
    // significant digits, which would move a marker at x = 12345.678.
    def circle($cx, $cy, $r, $fill) {
        bp_fill_path($this.cv, "M " + _px($cx - $r) + " " + _px($cy) + " a " + _px($r) + " " + _px($r) +
                     " 0 1 0 " + _px(2 * $r) + " 0 a " + _px($r) + " " + _px($r) + " 0 1 0 " +
                     _px(0 - 2 * $r) + " 0 Z", _hex6($fill), null);
        return $this;
    }
    def polyline($pts, $stroke, $sw, $dash) {
        if (len($pts) < 4) { return $this; }
        bp_stroke_polyline($this.cv, $pts, _hex6($stroke), $sw, null, $dash, true);
        return $this;
    }
    def polygon($pts, $fill, $stroke, $sw, $opacity) {
        if (len($pts) < 6) { return $this; }
        bp_fill_polygon($this.cv, $pts, _hex6($fill), $opacity);
        if ($stroke != null) {
            $ring = $pts;
            push($ring, $pts[0]);
            push($ring, $pts[1]);
            bp_stroke_polyline($this.cv, $ring, _hex6($stroke), $sw, null, null, true);
        }
        return $this;
    }
    def path($d, $fill, $stroke, $sw) {
        bp_fill_path($this.cv, $d, _hex6($fill), null);
        if ($stroke != null) { bp_stroke_path($this.cv, $d, _hex6($stroke), $sw, null); }
        return $this;
    }
    def text($x, $y, $s, $size, $fill, $anchor, $rotate) {
        bp_text($this.cv, $x, $y, str($s), $size, _hex6($fill), $anchor, $rotate, null);
        return $this;
    }
    // One clip per axes: clip() records it, groupOpen() applies it.
    def clip($x, $y, $w, $h) { $this.clipRect = [$x, $y, $w, $h]; return "raster"; }
    def groupOpen($clip) {
        if ($clip != null && $this.clipRect != null) {
            $r = $this.clipRect;
            bp_canvas_clip($this.cv, $r[0], $r[1], $r[2], $r[3]);
        }
        return $this;
    }
    def groupClose() { bp_canvas_clip($this.cv, null, null, null, null); return $this; }
    def render() { return bp_png($this.cv); }
    // The font this backend draws, DejaVu Sans, which is wider than Helvetica.
    def textWidth($s, $size) { return bp_text_width(str($s), $size); }
}

// ════════════════════════════════════════════════════════════════════════
//  Axes — the plotting box
// ════════════════════════════════════════════════════════════════════════
class BPlotAxes {
    def init($figW, $figH, $left, $top, $w, $h) {
        // NO BACK-REFERENCE TO THE FIGURE, and that is deliberate.
        //
        // A Figure holds its Axes; an Axes holding its Figure would close a
        // reference CYCLE, and Bantu's object lifetime is refcounted, which
        // frees cycles never. Measured while building this phase: a figure
        // with a back-reference costs ~45 KB that is never returned, so a sua
        // handler drawing a chart per request grows without bound.
        //
        // Everything that needs the figure is therefore a FIGURE method
        // (twinx, shareX, setCurrent) rather than an axes method.
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
        // B2: scales, date axes, free-standing text, and the frame switch a
        // pie needs (a pie inside a box with ticks and a grid is nobody's
        // intent -- plt.axis("off") is matplotlib's spelling).
        $this.xscale = "linear";
        $this.yscale = "linear";
        $this.xlt = 1;
        $this.ylt = 1;
        $this.xdate = false;
        $this.frameOn = true;
        $this.texts = [];
        // Categorical tick labels, set by boxplot/violin/heatmap rather than
        // by the locator: their axis is positions, not a measurement.
        $this.xtickOverride = null;
        $this.ytickOverride = null;
        // Category names, in first-seen order, when an axis is categorical.
        $this.xcats = null;
        $this.ycats = null;
        // B3: an inverted y axis, which imshow needs so that row 0 is the TOP
        // row -- the convention every image format and every matrix printout
        // uses. Kept separate from setYLim, which still refuses an inverted
        // pair: "my bounds came out backwards" is a bug, and "I want this axis
        // to descend" is an intention, and they should not share a spelling.
        $this.yinvert = false;
        // Space reserved on the right for a colorbar.
        $this.cbar = null;
        // An image fills its axes: the 5% data margin that keeps points off
        // the frame is exactly wrong for a grid of cells, which would sit in a
        // band of background and look like a rendering mistake.
        $this.tightLimits = false;
        // The figure cell this axes owns, and the pads that inset the plot box
        // within it. One rectangle calculation for every entry point (BP27).
        $this.cell = [0, 0, $figW, $figH];
        $this.pads = [62, 18, 34, 52];        // left, right, top, bottom
        // "left" draws the y axis on the left and owns the frame; "right" is a
        // twin, which draws only its own y ticks -- the frame is drawn once,
        // because a doubled 1px stroke is visible and reads as a bug.
        $this.side = "left";
        // A twin's host, as an INDEX into the figure's axes list: a reference
        // would be a cycle (see the sharing notes on BPlotFigure).
        $this.hostIdx = null;
    }

    def applyPads() {
        $c = $this.cell;
        $p = $this.pads;
        $this.left = $c[0] + $p[0];
        $this.top = $c[1] + $p[2];
        $this.w = $c[2] - $p[0] - $p[1];
        $this.h = $c[3] - $p[2] - $p[3];
        if ($this.w < 10) { $this.w = 10; }
        if ($this.h < 10) { $this.h = 10; }
        // A colorbar reserved its strip by shrinking the box; re-applying the
        // pads would hand that strip back and draw the plot over the bar.
        if ($this.cbar != null) { $this.w = $this.w - $this.cbar["width"] - $this.cbar["gap"]; }
        return $this;
    }

    // ── Twin and shared axes ─────────────────────────────────────────────



    // ── Categorical axes ─────────────────────────────────────────────────
    //
    // bar(["Jan", "Feb", ...], values) is the first thing anyone types, and
    // without this it died in min() with "element 1 must be a number" — an
    // error about the wrong thing entirely.
    //
    // Strings become positions 0..n-1 and the labels become the ticks. The
    // mapping is kept ON THE AXES and extended, not rebuilt, so two series
    // sharing categories line up: if the second series were numbered from
    // scratch its bars would sit under the wrong labels, which is a chart
    // that lies rather than one that errors.
    def categorise($vals, $which) {
        if (len($vals) == 0) { return $vals; }
        $strs = 0;
        $i = 0;
        while ($i < len($vals)) {
            if (type($vals[$i]) == "string") { $strs = $strs + 1; }
            $i = $i + 1;
        }
        if ($strs == 0) { return $vals; }
        if ($strs != len($vals)) {
            throw "bplot: the " + $which + " values mix text and numbers -- an axis is either " +
                  "categorical or numeric, and guessing which would silently misplace points";
        }
        $cats = $this.xcats;
        if ($which == "y") { $cats = $this.ycats; }
        if ($cats == null) { $cats = []; }
        $out = [];
        $i = 0;
        while ($i < len($vals)) {
            $at = 0 - 1;
            $k = 0;
            while ($k < len($cats)) {
                if ($cats[$k] == $vals[$i]) { $at = $k; break; }
                $k = $k + 1;
            }
            if ($at < 0) { push($cats, $vals[$i]); $at = len($cats) - 1; }
            push($out, $at);
            $i = $i + 1;
        }
        $over = _gridTicks($cats, len($cats), "categories");
        if ($which == "y") { $this.ycats = $cats; $this.ytickOverride = $over; }
        else { $this.xcats = $cats; $this.xtickOverride = $over; }
        return $out;
    }

    // A datetime or date column on x makes this a date axis (BP34), so a time
    // series reads as one with no separate xdate() call.
    def dateIfTime($x) {
        if (_isTimeData($x)) { $this.xdate = true; }
        return null;
    }

    // plot() and scatter() over native data (BP32). The artist holds the
    // ndarrays themselves -- a shared_ptr each -- rather than lists, so no
    // later call copies the data again.
    def addNativeXY($kind, $X, $Y, $x, $y, $opts, $fn) {
        $nx = nd_size($X["a"]);
        $ny = nd_size($Y["a"]);
        if ($nx != $ny) {
            throw "bplot." + $fn + ": x" + _nameOf($x) + " has " + str($nx) + " points and y" +
                  _nameOf($y) + " has " + str($ny) + " -- they must match";
        }
        if ($opts == null) { $opts = {}; }
        if ($X["date"]) { $this.xdate = true; }
        $art = {
            "kind":   $kind,
            "native": true,
            "x":      $X["a"],
            "y":      $Y["a"],
            "bx":     _limitsArr($X["a"]),
            "by":     _limitsArr($Y["a"]),
            "color":  _color($opts["color"], $this.nextColor()),
            "label":  $opts["label"]
        };
        if ($kind == "line") {
            $art["width"] = _optNum($opts["width"], 1.8);
            $art["dash"] = $opts["dash"];
        } else {
            $art["size"] = _optNum($opts["size"], 3);
        }
        push($this.artists, $art);
        return $this;
    }

    // plot_frame(df, {kind, x, y, title, bins}) — draw a table (BP35).
    //
    //   kind  "line" (default), "bar", "barh", "scatter", "hist", "box", "step"
    //   x     the column on the x axis; default is the row number
    //   y     a column name or a list of names; default is every numeric column
    //
    // More than one series turns the legend on, and groups bars side by side.
    // A column named explicitly that is not numeric raises, naming it: silently
    // skipping a column the caller asked for by name is the worst answer. Only
    // the DEFAULT selection skips text columns, because nobody asked for those.
    def plot_frame($df, $opts) {
        if ($opts == null) { $opts = {}; }
        $k = _kindOf($df);
        if ($k == "series") {
            // A Series is a one-column table named after itself.
            $one = {};
            $one[str($df.name)] = $df.col;
            return $this.plotTable([str($df.name)], $one, $opts);
        }
        if ($k != "frame") {
            throw "bplot.plot_frame: expected an arctic DataFrame or Series, got " + $k;
        }
        return $this.plotTable($df.names, $df.cols, $opts);
    }

    def plotTable($names, $cols, $opts) {
        $kinds = ["line", "bar", "barh", "scatter", "hist", "box", "step"];
        $kind = $opts["kind"];
        if ($kind == null) { $kind = "line"; }
        if (!_inList($kinds, $kind)) {
            throw "bplot.plot_frame: unknown kind \"" + str($kind) + "\" -- use " + join($kinds, ", ");
        }
        $xname = $opts["x"];
        if ($xname != null && !_inList($names, $xname)) { throw _noColumn($xname, $names); }

        $ynames = [];
        $yopt = $opts["y"];
        if ($yopt == null) {
            $ynames = _numericNames($names, $cols, $xname);
            if (len($ynames) == 0) {
                throw "bplot.plot_frame: there is no numeric column to plot -- the columns are " +
                      _describeColumns($names, $cols);
            }
        } else {
            if (type($yopt) == "string") { $yopt = [$yopt]; }
            if (type($yopt) != "list") {
                throw "bplot.plot_frame: y must be a column name or a list of names, got " + type($yopt);
            }
            $i = 0;
            while ($i < len($yopt)) {
                $nm = $yopt[$i];
                if (!_inList($names, $nm)) { throw _noColumn($nm, $names); }
                if (_colKind($cols[$nm]) == "text") {
                    throw "bplot.plot_frame: column '" + str($nm) + "' is " + col_dtype($cols[$nm]) +
                          ", not numeric -- a " + $kind + " chart needs numbers. The columns are " +
                          _describeColumns($names, $cols);
                }
                push($ynames, $nm);
                $i = $i + 1;
            }
        }
        $nrows = 0;
        if (len($names) > 0) { $nrows = col_len($cols[$names[0]]); }
        $multi = len($ynames) > 1;

        if ($kind == "hist") {
            each ($nm in $ynames) { $this.hist($cols[$nm], {"label": $nm, "bins": $opts["bins"]}); }
            if (!$multi) { $this.setXLabel($ynames[0]); }
        }
        if ($kind == "box") {
            $data = [];
            each ($nm in $ynames) { push($data, $cols[$nm]); }
            $this.boxplot($data, {"labels": $ynames});
        }
        if ($kind == "scatter") {
            if ($xname == null || len($ynames) != 1) {
                throw "bplot.plot_frame: a scatter needs one x column and one y column -- " +
                      "{\"kind\": \"scatter\", \"x\": \"a\", \"y\": \"b\"}";
            }
            $this.scatter($cols[$xname], $cols[$ynames[0]], {"label": $ynames[0]});
            $this.setXLabel($xname);
            $this.setYLabel($ynames[0]);
        }
        if ($kind == "line" || $kind == "step") {
            $xs = null;
            if ($xname == null) { $xs = _rowNumbers($nrows); } else { $xs = $cols[$xname]; }
            each ($nm in $ynames) {
                if ($kind == "line") { $this.plot($xs, $cols[$nm], {"label": $nm}); }
                else { $this.step($xs, $cols[$nm], {"label": $nm}); }
            }
            if ($xname != null) { $this.setXLabel($xname); }
            if (!$multi) { $this.setYLabel($ynames[0]); }
        }
        if ($kind == "bar" || $kind == "barh") {
            // Grouped: m series share each slot, each 0.8/m wide, centred on
            // the category position -- the positions categorise() would give.
            $labels = [];
            if ($xname == null) {
                $i = 0;
                while ($i < $nrows) { push($labels, _fmt($i, 0)); $i = $i + 1; }
            } else {
                $raw = _seq($cols[$xname], "x");
                if (_colKind($cols[$xname]) == "datetime" || _colKind($cols[$xname]) == "date") {
                    $raw = col_to_list($cols[$xname]);
                }
                $i = 0;
                while ($i < len($raw)) { push($labels, _labelText($raw[$i])); $i = $i + 1; }
            }
            $m = len($ynames);
            $w = 0.8 / $m;
            $j = 0;
            while ($j < $m) {
                $pos = [];
                $i = 0;
                while ($i < $nrows) { push($pos, $i + ($j - ($m - 1) / 2) * $w); $i = $i + 1; }
                $o = {"label": $ynames[$j], "width": $w};
                if ($kind == "bar") { $this.bar($pos, $cols[$ynames[$j]], $o); }
                else { $this.barh($pos, $cols[$ynames[$j]], $o); }
                $j = $j + 1;
            }
            $over = _gridTicks($labels, len($labels), "plot_frame");
            if ($kind == "bar") {
                $this.xcats = $labels; $this.xtickOverride = $over;
                if ($xname != null) { $this.setXLabel($xname); }
                if (!$multi) { $this.setYLabel($ynames[0]); }
            } else {
                $this.ycats = $labels; $this.ytickOverride = $over;
                if ($xname != null) { $this.setYLabel($xname); }
                if (!$multi) { $this.setXLabel($ynames[0]); }
            }
        }
        if ($multi && $kind != "box") { $this.setLegend(true); }
        if ($opts["title"] != null) { $this.setTitle($opts["title"]); }
        return $this;
    }

    def nextColor() {
        $c = $_STYLE["cycle"][$this.colorIdx - floor($this.colorIdx / len($_STYLE["cycle"])) * len($_STYLE["cycle"])];
        $this.colorIdx = $this.colorIdx + 1;
        return $c;
    }

    // ── Artists ──────────────────────────────────────────────────────────
    //
    // Every artist carries its own data bounds as "bx" and "by", computed
    // once when it is created. The alternative -- rediscovering the limits of
    // every series on every render -- re-walks the data for each of the
    // several renders a figure typically gets, and it forces dataLimits() to
    // know the shape of every artist kind, which is exactly the coupling that
    // makes adding a chart type a change in three places.
    def plot($x, $y, $opts) {
        if ($_NAT["on"]) {
            $X = _arr($x, "x");
            $Y = _arr($y, "y");
            if ($X != null && $Y != null) { return $this.addNativeXY("line", $X, $Y, $x, $y, $opts, "plot"); }
        }
        $this.dateIfTime($x);
        $xs = $this.categorise(_seq($x, "x"), "x");
        $ys = $this.categorise(_seq($y, "y"), "y");
        if (len($xs) != len($ys)) {
            throw "bplot.plot: x" + _nameOf($x) + " has " + str(len($xs)) + " points and y" +
                  _nameOf($y) + " has " + str(len($ys)) + " -- they must match";
        }
        if ($opts == null) { $opts = {}; }
        push($this.artists, {
            "kind":  "line",
            "x":     $xs,
            "y":     $ys,
            "bx":    _limits($xs),
            "by":    _limits($ys),
            "color": _color($opts["color"], $this.nextColor()),
            "width": _optNum($opts["width"], 1.8),
            "label": $opts["label"],
            "dash":  $opts["dash"]
        });
        return $this;
    }

    def scatter($x, $y, $opts) {
        if ($_NAT["on"]) {
            $X = _arr($x, "x");
            $Y = _arr($y, "y");
            if ($X != null && $Y != null) { return $this.addNativeXY("scatter", $X, $Y, $x, $y, $opts, "scatter"); }
        }
        $this.dateIfTime($x);
        $xs = $this.categorise(_seq($x, "x"), "x");
        $ys = $this.categorise(_seq($y, "y"), "y");
        if (len($xs) != len($ys)) {
            throw "bplot.scatter: x" + _nameOf($x) + " has " + str(len($xs)) + " points and y" +
                  _nameOf($y) + " has " + str(len($ys)) + " -- they must match";
        }
        if ($opts == null) { $opts = {}; }
        push($this.artists, {
            "kind":  "scatter",
            "x":     $xs,
            "y":     $ys,
            "bx":    _limits($xs),
            "by":    _limits($ys),
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
        // For barh the categories sit on y: the bars run along x.
        $catAxis = "x";
        $valAxis = "y";
        if ($horiz) { $catAxis = "y"; $valAxis = "x"; }
        $xs = $this.categorise(_seq($x, "x"), $catAxis);
        $ys = $this.categorise(_seq($y, "y"), $valAxis);
        if (len($xs) != len($ys)) {
            throw "bplot.bar: x has " + str(len($xs)) + " bars and y has " +
                  str(len($ys)) + " heights -- they must match";
        }
        if ($opts == null) { $opts = {}; }
        $bx = _limits($xs);
        $by = _limits($ys);
        // A bar is drawn from a baseline, so the value axis must include it.
        if ($horiz) { $bx = _withZero($bx); } else { $by = _withZero($by); }
        push($this.artists, {
            "kind":  "bar",
            "x":     $xs,
            "y":     $ys,
            "bx":    $bx,
            "by":    $by,
            "color": _color($opts["color"], $this.nextColor()),
            "width": _optNum($opts["width"], 0.8),
            "label": $opts["label"],
            "horiz": $horiz
        });
        return $this;
    }

    // ── B2 chart types ───────────────────────────────────────────────────

    // hist(values, {bins, range:[lo,hi], density, color, label, horiz})
    // Binning matches numba's nd_histogram decision for decision (BP19), so
    // the same data binned in Bantu and through numba gives the same chart.
    def hist($v, $opts) {
        $vs = _nums($v, "values");
        if ($opts == null) { $opts = {}; }
        $bins = _optNum($opts["bins"], 10);
        if ($bins < 1 || $bins != floor($bins)) {
            throw "bplot.hist: bins must be a whole number of at least 1, got " + str($bins);
        }
        $rlo = null;
        $rhi = null;
        $r = $opts["range"];
        if ($r != null) {
            if (type($r) != "list" || len($r) != 2) {
                throw "bplot.hist: range must be [lo, hi]";
            }
            $rlo = $r[0];
            $rhi = $r[1];
        }
        $rng = _histRange($vs, $rlo, $rhi);
        $counts = _histCounts($vs, $bins, $rng[0], $rng[1]);
        $width = ($rng[1] - $rng[0]) / $bins;
        $edges = [];
        $i = 0;
        while ($i <= $bins) { push($edges, $rng[0] + $i * $width); $i = $i + 1; }
        $tot = 0;
        $i = 0;
        while ($i < $bins) { $tot = $tot + $counts[$i]; $i = $i + 1; }
        $heights = [];
        $i = 0;
        while ($i < $bins) {
            // density normalises so the AREAS sum to 1 (NumPy's definition),
            // not the heights -- which differ the moment bin widths differ.
            if ($opts["density"] == true && $tot > 0) { push($heights, $counts[$i] / ($tot * $width)); }
            else { push($heights, $counts[$i]); }
            $i = $i + 1;
        }
        $hmax = 0;
        $i = 0;
        while ($i < $bins) { if ($heights[$i] > $hmax) { $hmax = $heights[$i]; } $i = $i + 1; }
        $horiz = $opts["horiz"] == true;
        $bx = [$rng[0], $rng[1]];
        $by = [0, $hmax];
        if ($horiz) { $bx = [0, $hmax]; $by = [$rng[0], $rng[1]]; }
        push($this.artists, {
            "kind":    "hist",
            "edges":   $edges,
            "heights": $heights,
            "counts":  $counts,
            "bx":      $bx,
            "by":      $by,
            "color":   _color($opts["color"], $this.nextColor()),
            "label":   $opts["label"],
            "horiz":   $horiz
        });
        return $this;
    }

    // boxplot(data, {labels, color, width}) — data is one sequence or a list
    // of sequences. Tukey's box with whiskers at the most extreme OBSERVED
    // value inside 1.5*IQR (BP20).
    def boxplot($data, $opts) {
        if ($opts == null) { $opts = {}; }
        $groups = _groupsOf($data, "boxplot");
        $stats = [];
        $lo = null;
        $hi = null;
        $i = 0;
        while ($i < len($groups)) {
            $s = _boxStats(_finiteSorted($groups[$i], "boxplot"));
            push($stats, $s);
            $blo = $s["wlo"];
            $bhi = $s["whi"];
            $j = 0;
            while ($j < len($s["out"])) {
                if ($s["out"][$j] < $blo) { $blo = $s["out"][$j]; }
                if ($s["out"][$j] > $bhi) { $bhi = $s["out"][$j]; }
                $j = $j + 1;
            }
            if ($lo == null || $blo < $lo) { $lo = $blo; }
            if ($hi == null || $bhi > $hi) { $hi = $bhi; }
            $i = $i + 1;
        }
        $this.xtickOverride = _categoryTicks(_labelsFor($data, $opts["labels"]), len($groups), "boxplot");
        push($this.artists, {
            "kind":  "box",
            "stats": $stats,
            "bx":    [0.5, len($groups) + 0.5],
            "by":    [$lo, $hi],
            "color": _color($opts["color"], $this.nextColor()),
            "width": _optNum($opts["width"], 0.5),
            "label": $opts["label"]
        });
        return $this;
    }

    // violin(data, {labels, color, width, points}) — a binned KDE (BP21), so
    // the cost does not scale with the number of points.
    def violin($data, $opts) {
        if ($opts == null) { $opts = {}; }
        $groups = _groupsOf($data, "violin");
        $grid = _optNum($opts["points"], 128);
        if ($grid < 8) { $grid = 8; }
        $shapes = [];
        $lo = null;
        $hi = null;
        $i = 0;
        while ($i < len($groups)) {
            $s = _finiteSorted($groups[$i], "violin");
            $d = _kde($s, $grid);
            if ($d == null) {
                // No spread at all: one value repeated. There is no density to
                // estimate, so draw the value rather than divide by zero.
                push($shapes, {"flat": $s[0]});
                if ($lo == null || $s[0] < $lo) { $lo = $s[0]; }
                if ($hi == null || $s[0] > $hi) { $hi = $s[0]; }
            } else {
                push($shapes, $d);
                $l = _limits($d["x"]);
                if ($lo == null || $l[0] < $lo) { $lo = $l[0]; }
                if ($hi == null || $l[1] > $hi) { $hi = $l[1]; }
            }
            $i = $i + 1;
        }
        $this.xtickOverride = _categoryTicks(_labelsFor($data, $opts["labels"]), len($groups), "violin");
        push($this.artists, {
            "kind":   "violin",
            "shapes": $shapes,
            "bx":     [0.5, len($groups) + 0.5],
            "by":     [$lo, $hi],
            "color":  _color($opts["color"], $this.nextColor()),
            "width":  _optNum($opts["width"], 0.7),
            "label":  $opts["label"]
        });
        return $this;
    }

    // errorbar(x, y, {yerr, xerr, color, width, size, label, cap})
    // yerr/xerr take a number, one sequence (symmetric) or two sequences
    // [lower, upper] (asymmetric).
    def errorbar($x, $y, $opts) {
        $this.dateIfTime($x);
        $xs = $this.categorise(_seq($x, "x"), "x");
        $ys = $this.categorise(_seq($y, "y"), "y");
        if (len($xs) != len($ys)) {
            throw "bplot.errorbar: x has " + str(len($xs)) + " points and y has " +
                  str(len($ys)) + " -- they must match";
        }
        if ($opts == null) { $opts = {}; }
        $ye = _errPair($opts["yerr"], len($ys), "yerr");
        $xe = _errPair($opts["xerr"], len($xs), "xerr");
        push($this.artists, {
            "kind":  "errorbar",
            "x":     $xs,
            "y":     $ys,
            "xerr":  $xe,
            "yerr":  $ye,
            "bx":    _spread($xs, $xe),
            "by":    _spread($ys, $ye),
            "color": _color($opts["color"], $this.nextColor()),
            "width": _optNum($opts["width"], 1.4),
            "size":  _optNum($opts["size"], 3),
            "cap":   _optNum($opts["cap"], 4),
            "line":  $opts["line"] == true,
            "label": $opts["label"]
        });
        return $this;
    }

    // fill_between(x, y1, y2, {color, opacity, label})
    // y2 defaults to zero. NaN in either edge breaks the band, for the same
    // reason NaN breaks a line (BP9).
    def fill_between($x, $y1, $y2, $opts) {
        $this.dateIfTime($x);
        $xs = $this.categorise(_seq($x, "x"), "x");
        $a = _seq($y1, "y1");
        if (len($xs) != len($a)) {
            throw "bplot.fill_between: x has " + str(len($xs)) + " points and y1 has " +
                  str(len($a)) + " -- they must match";
        }
        $b = [];
        if ($y2 == null) {
            $i = 0;
            while ($i < len($xs)) { push($b, 0); $i = $i + 1; }
        } else {
            if (type($y2) == "number") {
                $i = 0;
                while ($i < len($xs)) { push($b, $y2); $i = $i + 1; }
            } else {
                $b = _seq($y2, "y2");
                if (len($b) != len($xs)) {
                    throw "bplot.fill_between: x has " + str(len($xs)) + " points and y2 has " +
                          str(len($b)) + " -- they must match";
                }
            }
        }
        if ($opts == null) { $opts = {}; }
        $bya = _limits($a);
        $byb = _limits($b);
        $by = $bya;
        if ($bya == null) { $by = $byb; }
        else { if ($byb != null) { $by = [min($bya[0], $byb[0]), max($bya[1], $byb[1])]; } }
        push($this.artists, {
            "kind":    "band",
            "x":       $xs,
            "y1":      $a,
            "y2":      $b,
            "bx":      _limits($xs),
            "by":      $by,
            "color":   _color($opts["color"], $this.nextColor()),
            "opacity": _optNum($opts["opacity"], 0.35),
            "edge":    $opts["edge"] == true,
            "label":   $opts["label"]
        });
        return $this;
    }

    // step(x, y, {where: "pre"|"post"|"mid", ...}) — matplotlib's names,
    // because a step chart drawn with the wrong convention is off by one
    // sample and still looks plausible.
    def step($x, $y, $opts) {
        $this.dateIfTime($x);
        $xs = $this.categorise(_seq($x, "x"), "x");
        $ys = $this.categorise(_seq($y, "y"), "y");
        if (len($xs) != len($ys)) {
            throw "bplot.step: x has " + str(len($xs)) + " points and y has " +
                  str(len($ys)) + " -- they must match";
        }
        if ($opts == null) { $opts = {}; }
        $where = $opts["where"];
        if ($where == null) { $where = "pre"; }
        if ($where != "pre" && $where != "post" && $where != "mid") {
            throw "bplot.step: where must be \"pre\", \"post\" or \"mid\", got \"" + str($where) + "\"";
        }
        push($this.artists, {
            "kind":  "step",
            "x":     $xs,
            "y":     $ys,
            "bx":    _limits($xs),
            "by":    _limits($ys),
            "where": $where,
            "color": _color($opts["color"], $this.nextColor()),
            "width": _optNum($opts["width"], 1.8),
            "label": $opts["label"],
            "dash":  $opts["dash"]
        });
        return $this;
    }

    // stem(x, y, {bottom, color, size, label})
    def stem($x, $y, $opts) {
        $this.dateIfTime($x);
        $xs = $this.categorise(_seq($x, "x"), "x");
        $ys = $this.categorise(_seq($y, "y"), "y");
        if (len($xs) != len($ys)) {
            throw "bplot.stem: x has " + str(len($xs)) + " points and y has " +
                  str(len($ys)) + " -- they must match";
        }
        if ($opts == null) { $opts = {}; }
        $bottom = _optNum($opts["bottom"], 0);
        push($this.artists, {
            "kind":   "stem",
            "x":      $xs,
            "y":      $ys,
            "bx":     _limits($xs),
            "by":     _withValue(_limits($ys), $bottom),
            "bottom": $bottom,
            "color":  _color($opts["color"], $this.nextColor()),
            "width":  _optNum($opts["width"], 1.2),
            "size":   _optNum($opts["size"], 3),
            "label":  $opts["label"]
        });
        return $this;
    }

    // pie(values, {labels, colors, startangle, explode, percent, radius})
    // Angles come from each value's share of the total; the axes turns its
    // frame off, because a pie in a box with ticks and a grid is nobody's
    // intent.
    def pie($v, $opts) {
        $vs = _seq($v, "values");
        if ($opts == null) { $opts = {}; }
        $tot = 0;
        $i = 0;
        while ($i < len($vs)) {
            $x = $vs[$i];
            if (!isfinite($x)) {
                throw "bplot.pie: every value must be finite, got " + str($x) + " at position " + str($i);
            }
            if ($x < 0) {
                throw "bplot.pie: a slice cannot be negative, got " + str($x) + " at position " + str($i) +
                      " -- a pie shows parts of a whole, so use bar() for data that can go below zero";
            }
            $tot = $tot + $x;
            $i = $i + 1;
        }
        if ($tot <= 0) { throw "bplot.pie: the values sum to zero, so there is nothing to divide"; }
        $cols = [];
        $given = $opts["colors"];
        $i = 0;
        while ($i < len($vs)) {
            if ($given != null && $i < len($given)) { push($cols, _color($given[$i], null)); }
            else { push($cols, $this.nextColor()); }
            $i = $i + 1;
        }
        $this.frameOn = false;
        push($this.artists, {
            "kind":    "pie",
            "values":  $vs,
            "total":   $tot,
            "colors":  $cols,
            "labels":  $opts["labels"],
            "start":   _optNum($opts["startangle"], 90),
            "radius":  _optNum($opts["radius"], 1),
            "percent": $opts["percent"] == true,
            "bx":      [0 - 1.35, 1.35],
            "by":      [0 - 1.2, 1.2]
        });
        return $this;
    }

    // ── B3: 2-D ──────────────────────────────────────────────────────────

    // imshow(z, {cmap, vmin, vmax, extent, origin, maxcells, label})
    //
    // z is a list of rows, or a 2-D numba ndarray. Rows go top to bottom by
    // default (origin "upper"), which is what every image format and every
    // printed matrix does.
    def imshow($z, $opts) {
        if ($opts == null) { $opts = {}; }
        $rows = _grid2d($z, "imshow");
        $maxc = _optNum($opts["maxcells"], 256);
        if ($maxc < 1) { throw "bplot.imshow: maxcells must be at least 1, got " + str($maxc); }
        $rows = _blockReduce($rows, $maxc);
        $h = len($rows);
        $w = len($rows[0]);
        $rng = _gridRange($rows, $opts["vmin"], $opts["vmax"], "imshow");
        $cm = $opts["cmap"];
        if ($cm == null) { $cm = "viridis"; }
        _cmapTable($cm);                 // validate the name now, not mid-render
        $origin = $opts["origin"];
        if ($origin == null) { $origin = "upper"; }
        if ($origin != "upper" && $origin != "lower") {
            throw "bplot.imshow: origin must be \"upper\" or \"lower\", got \"" + str($origin) + "\"";
        }
        // matplotlib's convention: cell (r, c) is CENTRED on (c, r), so the
        // extent runs from -0.5 to n-0.5 and integer ticks land on cells.
        $ext = $opts["extent"];
        if ($ext == null) { $ext = [0 - 0.5, $w - 0.5, 0 - 0.5, $h - 0.5]; }
        else {
            if (type($ext) != "list" || len($ext) != 4) {
                throw "bplot.imshow: extent must be [x0, x1, y0, y1]";
            }
        }
        if ($origin == "upper") { $this.yinvert = true; }
        $this.tightLimits = true;
        push($this.artists, {
            "kind":   "image",
            "rows":   $rows,
            "w":      $w,
            "h":      $h,
            "ext":    $ext,
            "vmin":   $rng[0],
            "vmax":   $rng[1],
            "cmap":   $cm,
            "labels": $opts["cells"] == true,
            "bx":     [$ext[0], $ext[1]],
            "by":     [$ext[2], $ext[3]],
            "label":  null,
            "color":  null
        });
        return $this;
    }

    // heatmap(z, {rows, cols, ...}) — imshow with the cell values written in
    // and the axes labelled by category. The combination people actually want
    // when they say "heatmap", and the one SVG is genuinely good at.
    def heatmap($z, $opts) {
        // A frame is drawn as its numeric columns, labelled with their names,
        // unless the caller labelled them.
        if (_kindOf($z) == "frame") {
            $o = {};
            if ($opts != null) { each ($k in keys($opts)) { $o[$k] = $opts[$k]; } }
            if ($o["cols"] == null) { $o["cols"] = _numericNames($z.names, $z.cols, null); }
            $opts = $o;
        }
        if ($opts == null) { $opts = {}; }
        $o = {};
        each ($k in keys($opts)) { $o[$k] = $opts[$k]; }
        if ($o["cells"] == null) { $o["cells"] = true; }
        $this.imshow($z, $o);
        $a = $this.artists[len($this.artists) - 1];
        $this.xtickOverride = _gridTicks($opts["cols"], $a["w"], "heatmap cols");
        $this.ytickOverride = _gridTicks($opts["rows"], $a["h"], "heatmap rows");
        $this.gridOn = false;
        return $this;
    }

    // pcolormesh(x, y, z) — like imshow but with explicit cell EDGES, so the
    // cells need not be uniform. x has one more entry than z's width.
    def pcolormesh($x, $y, $z, $opts) {
        if ($opts == null) { $opts = {}; }
        $rows = _grid2d($z, "pcolormesh");
        $xe = _seq($x, "x");
        $ye = _seq($y, "y");
        $h = len($rows);
        $w = len($rows[0]);
        if (len($xe) != $w + 1) {
            throw "bplot.pcolormesh: x holds the cell EDGES, so it needs " + str($w + 1) +
                  " entries for " + str($w) + " columns, got " + str(len($xe));
        }
        if (len($ye) != $h + 1) {
            throw "bplot.pcolormesh: y holds the cell EDGES, so it needs " + str($h + 1) +
                  " entries for " + str($h) + " rows, got " + str(len($ye));
        }
        $rng = _gridRange($rows, $opts["vmin"], $opts["vmax"], "pcolormesh");
        $cm = $opts["cmap"];
        if ($cm == null) { $cm = "viridis"; }
        _cmapTable($cm);
        $this.tightLimits = true;
        push($this.artists, {
            "kind":  "mesh",
            "rows":  $rows,
            "xe":    $xe,
            "ye":    $ye,
            "w":     $w,
            "h":     $h,
            "vmin":  $rng[0],
            "vmax":  $rng[1],
            "cmap":  $cm,
            "bx":    _limits($xe),
            "by":    _limits($ye),
            "label": null,
            "color": null
        });
        return $this;
    }

    // contour(z, {levels, extent, color, cmap}) — marching squares.
    def contour($z, $opts) {
        if ($opts == null) { $opts = {}; }
        $rows = _grid2d($z, "contour");
        $rows = _blockReduce($rows, _optNum($opts["maxcells"], 256));
        $h = len($rows);
        $w = len($rows[0]);
        if ($h < 2 || $w < 2) {
            throw "bplot.contour: needs a grid of at least 2x2, got " + str($h) + "x" + str($w);
        }
        $rng = _gridRange($rows, $opts["vmin"], $opts["vmax"], "contour");
        $levels = $opts["levels"];
        if ($levels == null) { $levels = 8; }
        if (type($levels) == "number") {
            $n = $levels;
            if ($n < 1) { throw "bplot.contour: levels must be at least 1, got " + str($n); }
            $levels = [];
            $i = 1;
            while ($i <= $n) { push($levels, $rng[0] + ($rng[1] - $rng[0]) * $i / ($n + 1)); $i = $i + 1; }
        } else {
            if (type($levels) != "list") { throw "bplot.contour: levels must be a number or a list"; }
        }
        $ext = $opts["extent"];
        if ($ext == null) { $ext = [0, $w - 1, 0, $h - 1]; }
        $cm = $opts["cmap"];
        if ($cm != null) { _cmapTable($cm); }
        push($this.artists, {
            "kind":   "contour",
            "rows":   $rows,
            "w":      $w,
            "h":      $h,
            "ext":    $ext,
            "levels": $levels,
            "vmin":   $rng[0],
            "vmax":   $rng[1],
            "cmap":   $cm,
            "color":  _color($opts["color"], $this.nextColor()),
            "width":  _optNum($opts["width"], 1.2),
            "bx":     [$ext[0], $ext[1]],
            "by":     [$ext[2], $ext[3]],
            "label":  $opts["label"]
        });
        return $this;
    }

    // colorbar() — the key for the last colour-mapped artist. It reserves its
    // own strip by SHRINKING the axes once, which is why adding it twice is
    // refused rather than silently eating the plot.
    def colorbar($opts) {
        if ($this.cbar != null) { throw "bplot.colorbar: this axes already has one"; }
        if ($opts == null) { $opts = {}; }
        $src = null;
        $i = len($this.artists) - 1;
        while ($i >= 0) {
            $k = $this.artists[$i]["kind"];
            if ($k == "image" || $k == "mesh" || $k == "contour") { $src = $this.artists[$i]; break; }
            $i = $i - 1;
        }
        if ($src == null) {
            throw "bplot.colorbar: nothing on these axes is colour-mapped -- " +
                  "call imshow, heatmap, pcolormesh or contour first";
        }
        $width = _optNum($opts["width"], 16);
        $gapv = _optNum($opts["gap"], 46);
        $take = $width + $gapv;
        if ($take >= $this.w - 40) {
            throw "bplot.colorbar: the axes is too narrow to give up " + str($take) + " pixels";
        }
        $this.w = $this.w - $take;
        $cm = $src["cmap"];
        if ($cm == null) { $cm = "viridis"; }
        $this.cbar = {"cmap": $cm, "vmin": $src["vmin"], "vmax": $src["vmax"],
                      "width": $width, "gap": $gapv, "label": $opts["label"]};
        return $this;
    }

    // ── Text and annotations ─────────────────────────────────────────────
    def addText($x, $y, $s, $opts) {
        if ($opts == null) { $opts = {}; }
        push($this.texts, {
            "x":      $x,
            "y":      $y,
            "s":      str($s),
            "size":   _optNum($opts["size"], 12),
            "color":  _color($opts["color"], $_STYLE["fg"]),
            "anchor": _anchorOf($opts["anchor"]),
            "rotate": _optNum($opts["rotate"], 0),
            "arrow":  null
        });
        return $this;
    }

    // annotate(text, x, y, {to: [x, y], ...}) — the text sits at (x, y); when
    // `to` is given an arrow is drawn from the text to that point.
    def annotate($s, $x, $y, $opts) {
        if ($opts == null) { $opts = {}; }
        $to = $opts["to"];
        if ($to != null) {
            if (type($to) != "list" || len($to) != 2) {
                throw "bplot.annotate: `to` must be [x, y]";
            }
        }
        push($this.texts, {
            "x":      $x,
            "y":      $y,
            "s":      str($s),
            "size":   _optNum($opts["size"], 12),
            "color":  _color($opts["color"], $_STYLE["fg"]),
            "anchor": _anchorOf($opts["anchor"]),
            "rotate": _optNum($opts["rotate"], 0),
            "arrow":  $to
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

    // ── Scales ───────────────────────────────────────────────────────────
    def setScale($which, $scale, $linthresh) {
        if ($scale != "linear" && $scale != "log" && $scale != "symlog") {
            throw "bplot: scale must be \"linear\", \"log\" or \"symlog\", got \"" + str($scale) + "\"";
        }
        $lt = _checkLinthresh($linthresh, $which);
        if ($which == "x") { $this.xscale = $scale; $this.xlt = $lt; }
        else { $this.yscale = $scale; $this.ylt = $lt; }
        return $this;
    }
    def setDateAxis($on) {
        if ($on == null) { $on = true; }
        $this.xdate = $on;
        return $this;
    }
    def setFrame($on) {
        if ($on == null) { $on = true; }
        $this.frameOn = $on;
        return $this;
    }
    def invertY($on) {
        if ($on == null) { $on = true; }
        $this.yinvert = $on;
        return $this;
    }

    def scaleOf($which) { if ($which == "y") { return $this.yscale; } return $this.xscale; }
    def ltOf($which)    { if ($which == "y") { return $this.ylt; }    return $this.xlt; }

    // ── Limits ───────────────────────────────────────────────────────────
    // Each artist published its own bounds when it was created, so this is a
    // union over a handful of pairs rather than a re-walk of every series.
    def dataLimits($which) {
        $key = "bx";
        if ($which == "y") { $key = "by"; }
        $lo = null;
        $hi = null;
        $i = 0;
        while ($i < len($this.artists)) {
            $lim = $this.artists[$i][$key];
            if ($lim != null && $lim[0] != null && $lim[1] != null) {
                if ($lo == null || $lim[0] < $lo) { $lo = $lim[0]; }
                if ($hi == null || $lim[1] > $hi) { $hi = $lim[1]; }
            }
            $i = $i + 1;
        }
        return [$lo, $hi];
    }

    // View limits in DATA space. The 5% margin is applied in TRANSFORMED
    // space and mapped back, because on a log axis a linear 5% of the value
    // range is most of the chart at the bottom and invisible at the top.
    def viewLimits($which) { return $this.viewLimitsShared($which, null); }

    // `share` is the union of the share group's data limits, or null when this
    // axes belongs to no group — in which case this is exactly the B1 path.
    def viewLimitsShared($which, $share) {
        $lim = $this.viewLimitsAsc($which, $share);
        if ($which == "y" && $this.yinvert) { return [$lim[1], $lim[0]]; }
        return $lim;
    }

    def viewLimitsAsc($which, $share) {
        $set = $this.xlimSet;
        if ($which == "y") { $set = $this.ylimSet; }
        $scale = $this.scaleOf($which);
        $lt = $this.ltOf($which);
        if ($set != null) {
            if ($scale == "log" && $set[0] <= 0) {
                throw "bplot: a log " + $which + " axis cannot start at " + str($set[0]) +
                      " -- the lower bound must be above zero";
            }
            return $set;
        }
        $d = $this.dataLimits($which);
        if ($share != null) { $d = $share; }
        if ($scale == "log") {
            // The bounds are CLAMPED here rather than raised on, and the
            // distinction matters. A bar, a histogram and a stem all publish a
            // baseline of zero as part of their bounds -- that zero is
            // decoration, not data, so raising on it would make bar() and
            // hist() unusable on a log axis for no good reason.
            //
            // Real non-positive DATA still raises: it does so in _project,
            // on the drawing path, where the offending value is in hand and
            // can be named. A line with a zero in it must raise, because a
            // silently missing segment is invisible; a bar of height zero is
            // simply not drawn, because an absent bar already reads as zero.
            if ($d[0] == null || $d[1] == null || $d[1] <= 0) { return [1, 10]; }
            if ($d[0] <= 0) { $d = [$d[1] / 1000, $d[1]]; }
            if ($d[1] == $d[0]) { return [$d[0] / 3, $d[1] * 3]; }
        }
        $e = _expand($d[0], $d[1]);
        $tlo = _fwd($e[0], $scale, $lt);
        $thi = _fwd($e[1], $scale, $lt);
        $pad = ($thi - $tlo) * 0.05;
        if ($this.tightLimits) { $pad = 0; }
        return [_inv($tlo - $pad, $scale, $lt), _inv($thi + $pad, $scale, $lt)];
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

// A bar or a stem is drawn from a baseline, so the value axis has to reach it
// even when no data point does.
def _withValue($lim, $v) {
    if ($lim == null) { return [$v, $v]; }
    return [min($lim[0], $v), max($lim[1], $v)];
}
def _withZero($lim) { return _withValue($lim, 0); }

def _anchorOf($a) {
    if ($a == null) { return "middle"; }
    if ($a != "start" && $a != "middle" && $a != "end") {
        throw "bplot: anchor must be \"start\", \"middle\" or \"end\", got \"" + str($a) + "\"";
    }
    return $a;
}

// boxplot/violin take either one sequence or a list of sequences. Telling
// them apart by looking at the first element is the only way -- and getting it
// wrong silently turns fifty samples into fifty one-point boxes.
def _noColumn($nm, $names) {
    return "bplot.plot_frame: there is no column '" + str($nm) + "' -- the columns are " + join($names, ", ");
}

def _describeColumns($names, $cols) {
    $parts = [];
    $i = 0;
    while ($i < len($names)) {
        push($parts, $names[$i] + " (" + col_dtype($cols[$names[$i]]) + ")");
        $i = $i + 1;
    }
    return join($parts, ", ");
}

// 0..n-1 in the current representation: the x of a table with no x column.
def _rowNumbers($n) {
    if ($_NAT["on"]) { return nd_arange(0, $n, 1); }
    $out = [];
    $i = 0;
    while ($i < $n) { push($out, $i); $i = $i + 1; }
    return $out;
}

// A category label: text as itself, a whole number without a decimal point.
def _labelText($v) {
    if (type($v) == "number" && isfinite($v) && $v == floor($v)) { return _fmt($v, 0); }
    return str($v);
}

// Group labels default to a frame's numeric column names.
def _labelsFor($data, $labels) {
    if ($labels == null && _kindOf($data) == "frame") {
        return _numericNames($data.names, $data.cols, null);
    }
    return $labels;
}

def _groupsOf($data, $what) {
    if ($data == null) { throw "bplot." + $what + ": data is null"; }
    $k = _kindOf($data);
    if ($k == "ndarray" || $k == "column" || $k == "series") { return [_nums($data, "data")]; }
    if ($k == "frame") {
        $out = [];
        each ($nm in _numericNames($data.names, $data.cols, null)) {
            push($out, _nums($data.cols[$nm], "column '" + $nm + "'"));
        }
        if (len($out) == 0) { throw "bplot." + $what + ": the frame has no numeric columns"; }
        return $out;
    }
    if ($k != "list") {
        throw "bplot." + $what + ": data must be a list of numbers, or a list of such lists, got " + $k;
    }
    if (len($data) == 0) { throw "bplot." + $what + ": data is empty"; }
    $k0 = _kindOf($data[0]);
    if ($k0 == "list" || $k0 == "ndarray" || $k0 == "column" || $k0 == "series") {
        $out = [];
        $i = 0;
        while ($i < len($data)) { push($out, _nums($data[$i], "data")); $i = $i + 1; }
        return $out;
    }
    return [_nums($data, "data")];
}

// Positions 1..n with the caller's labels, or the position number.
def _categoryTicks($labels, $n, $what) {
    $pos = [];
    $lab = [];
    if ($labels != null) {
        if (type($labels) != "list") { throw "bplot." + $what + ": labels must be a list"; }
        if (len($labels) != $n) {
            throw "bplot." + $what + ": " + str($n) + " groups but " + str(len($labels)) + " labels";
        }
    }
    $i = 0;
    while ($i < $n) {
        push($pos, $i + 1);
        if ($labels != null) { push($lab, str($labels[$i])); }
        else { push($lab, _fmt($i + 1, 0)); }
        $i = $i + 1;
    }
    return {"pos": $pos, "labels": $lab};
}

// An error specification becomes [lower[], upper[]]. A number is symmetric
// and constant; one sequence is symmetric and per-point; two sequences are
// asymmetric. A NEGATIVE error is always a bug -- drawn, it produces an
// inverted bar that reads as a SMALLER error than the real one -- so it
// raises rather than being taken as its absolute value.
def _errPair($e, $n, $what) {
    if ($e == null) { return null; }
    if (type($e) == "number") {
        if ($e < 0 || !isfinite($e)) {
            throw "bplot.errorbar: " + $what + " must be finite and not negative, got " + str($e);
        }
        $a = [];
        $i = 0;
        while ($i < $n) { push($a, $e); $i = $i + 1; }
        return [$a, $a];
    }
    $lst = _seq($e, $what);
    if (len($lst) == 2 && type($lst[0]) == "list") {
        $lo = _checkErr(_seq($lst[0], $what), $n, $what);
        $hi = _checkErr(_seq($lst[1], $what), $n, $what);
        return [$lo, $hi];
    }
    $v = _checkErr($lst, $n, $what);
    return [$v, $v];
}

def _checkErr($v, $n, $what) {
    if (len($v) != $n) {
        throw "bplot.errorbar: " + $what + " has " + str(len($v)) + " entries for " + str($n) + " points";
    }
    $i = 0;
    while ($i < $n) {
        if ($v[$i] < 0) {
            throw "bplot.errorbar: " + $what + " cannot be negative, got " + str($v[$i]) +
                  " at position " + str($i);
        }
        $i = $i + 1;
    }
    return $v;
}

// Bounds of a series widened by its error bars.
def _spread($vals, $err) {
    $lim = _limits($vals);
    if ($err == null || $lim == null) { return $lim; }
    $lo = $lim[0];
    $hi = $lim[1];
    $i = 0;
    $n = len($vals);
    while ($i < $n) {
        if (isfinite($vals[$i])) {
            $a = $vals[$i] - $err[0][$i];
            $b = $vals[$i] + $err[1][$i];
            if ($a < $lo) { $lo = $a; }
            if ($b > $hi) { $hi = $b; }
        }
        $i = $i + 1;
    }
    return [$lo, $hi];
}

// ── 2-D input ────────────────────────────────────────────────────────────

// A grid is a list of equal-length rows, or a 2-D numba ndarray. A RAGGED
// grid raises naming both row lengths: taking the shortest row silently
// crops the data, and taking the longest pads it with values nobody supplied.
def _grid2d($z, $what) {
    if ($z == null) { throw "bplot." + $what + ": data is null"; }
    $t = type($z);
    if (_kindOf($z) == "frame") {
        // The numeric columns, one row per frame row.
        $names = _numericNames($z.names, $z.cols, null);
        if (len($names) == 0) { throw "bplot." + $what + ": the frame has no numeric columns"; }
        $filled = [];
        each ($nm in $names) { push($filled, _colF64($z.cols[$nm])); }
        $z = nd_to_list(nd_from_frame($filled));
        $t = "list";
    }
    if ($t == "ndarray") {
        $shape = nd_shape($z);
        if (len($shape) != 2) {
            throw "bplot." + $what + ": needs a 2-D array, got one with " +
                  str(len($shape)) + " dimensions";
        }
        $z = nd_to_list($z);
        $t = "list";
    }
    if ($t != "list") {
        throw "bplot." + $what + ": needs a list of rows or a 2-D ndarray, got " + $t;
    }
    if (len($z) == 0) { throw "bplot." + $what + ": the grid has no rows"; }
    $rows = [];
    $w = 0;
    $i = 0;
    while ($i < len($z)) {
        $r = _seq($z[$i], "row " + str($i));
        if ($i == 0) { $w = len($r); }
        else {
            if (len($r) != $w) {
                throw "bplot." + $what + ": row 0 has " + str($w) + " values but row " +
                      str($i) + " has " + str(len($r)) + " -- the grid must be rectangular";
            }
        }
        push($rows, $r);
        $i = $i + 1;
    }
    if ($w == 0) { throw "bplot." + $what + ": the grid has no columns"; }
    return $rows;
}

// Block-mean down to at most maxc cells per axis (decision BP26). The axes
// box is around 500x380 CSS pixels, so cells beyond this were never visible;
// averaging is a more honest reduction than the nearest-neighbour sampling a
// browser would otherwise apply to the same data.
def _blockReduce($rows, $maxc) {
    $h = len($rows);
    $w = len($rows[0]);
    if ($h <= $maxc && $w <= $maxc) { return $rows; }
    $fy = ceil($h / $maxc);
    $fx = ceil($w / $maxc);
    if ($fy < 1) { $fy = 1; }
    if ($fx < 1) { $fx = 1; }
    $out = [];
    $r0 = 0;
    while ($r0 < $h) {
        $row = [];
        $c0 = 0;
        while ($c0 < $w) {
            $sum = 0;
            $cnt = 0;
            $r = $r0;
            while ($r < $r0 + $fy && $r < $h) {
                $src = $rows[$r];
                $c = $c0;
                while ($c < $c0 + $fx && $c < $w) {
                    $v = $src[$c];
                    // A block of all-NaN stays NaN; a block with some data
                    // averages what it has. Letting one NaN poison the block
                    // would punch holes in an otherwise fine image.
                    if (isfinite($v)) { $sum = $sum + $v; $cnt = $cnt + 1; }
                    $c = $c + 1;
                }
                $r = $r + 1;
            }
            if ($cnt > 0) { push($row, $sum / $cnt); } else { push($row, NAN); }
            $c0 = $c0 + $fx;
        }
        push($out, $row);
        $r0 = $r0 + $fy;
    }
    return $out;
}

def _gridRange($rows, $vmin, $vmax, $what) {
    if ($vmin != null && $vmax != null) {
        if (!($vmin < $vmax)) {
            throw "bplot." + $what + ": vmin must be below vmax, got " + str($vmin) + " and " + str($vmax);
        }
        return [$vmin, $vmax];
    }
    $lo = null;
    $hi = null;
    $i = 0;
    while ($i < len($rows)) {
        $l = _limits($rows[$i]);
        if ($l != null) {
            if ($lo == null || $l[0] < $lo) { $lo = $l[0]; }
            if ($hi == null || $l[1] > $hi) { $hi = $l[1]; }
        }
        $i = $i + 1;
    }
    if ($lo == null) { return [0, 1]; }         // an all-NaN grid still draws
    if ($vmin != null) { $lo = $vmin; }
    if ($vmax != null) { $hi = $vmax; }
    if (!($lo < $hi)) { return [$lo - 0.5, $lo + 0.5]; }
    return [$lo, $hi];
}

// Category ticks at cell centres 0..n-1.
def _gridTicks($labels, $n, $what) {
    if ($labels == null) { return null; }
    if (type($labels) != "list") { throw "bplot." + $what + ": labels must be a list"; }
    if (len($labels) != $n) {
        throw "bplot." + $what + ": " + str($n) + " cells but " + str(len($labels)) + " labels";
    }
    $pos = [];
    $lab = [];
    $i = 0;
    while ($i < $n) { push($pos, $i); push($lab, str($labels[$i])); $i = $i + 1; }
    return {"pos": $pos, "labels": $lab};
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
        $this.tight = false;
        // Share groups, as lists of axes INDICES (see shareX).
        $this.xgroups = [];
        $this.ygroups = [];
    }

    // ── Layout (decision BP27) ───────────────────────────────────────────
    //
    // ONE rectangle calculation, used by everything. An axes owns a CELL of
    // the figure's grid, and its plot box is that cell inset by four pads.
    // A 1x1 grid with the default pads reproduces B1's fixed gutters exactly,
    // which is why B1's output did not move when this landed.
    //
    // matplotlib's `subplot` and `add_subplot` drifted apart historically
    // because two code paths computed "the same" rectangle. Here there is one,
    // and the test asserts both spellings produce identical rects.
    def cellRect($row, $col, $rows, $cols, $rowspan, $colspan) {
        $cw = $this.w / $cols;
        $ch = $this.h / $rows;
        return [$col * $cw, $row * $ch, $cw * $colspan, $ch * $rowspan];
    }

    def addAxesIn($cell) {
        $ax = new BPlotAxes($this.w, $this.h, 0, 0, 1, 1);
        $ax.cell = $cell;
        $ax.applyPads();
        push($this.axesList, $ax);
        $this.cur = len($this.axesList) - 1;
        return $ax;
    }

    def addAxes() { return $this.addAxesIn([0, 0, $this.w, $this.h]); }

    // subplots(rows, cols) -> a flat list in row-major order, which is the
    // order subplot(r, c, i) numbers them in.
    def subplots($rows, $cols) {
        if ($rows == null) { $rows = 1; }
        if ($cols == null) { $cols = 1; }
        _checkGrid($rows, $cols);
        $out = [];
        $r = 0;
        while ($r < $rows) {
            $c = 0;
            while ($c < $cols) {
                push($out, $this.addAxesIn($this.cellRect($r, $c, $rows, $cols, 1, 1)));
                $c = $c + 1;
            }
            $r = $r + 1;
        }
        $this.cur = 0;
        return $out;
    }

    // matplotlib's 1-based, row-major spelling. Produces the same rectangle
    // as subplots() does for the same cell — asserted, not assumed.
    def subplot($rows, $cols, $index) {
        _checkGrid($rows, $cols);
        if (type($index) != "number" || $index < 1 || $index > $rows * $cols || $index != floor($index)) {
            throw "bplot.subplot: index must be a whole number from 1 to " + str($rows * $cols) +
                  " (row-major), got " + str($index);
        }
        $r = floor(($index - 1) / $cols);
        $c = ($index - 1) - $r * $cols;
        return $this.addAxesIn($this.cellRect($r, $c, $rows, $cols, 1, 1));
    }

    // A cell spanning several rows or columns, for a dashboard where one panel
    // is the headline and the rest are small.
    def subplotSpan($rows, $cols, $row, $col, $rowspan, $colspan) {
        _checkGrid($rows, $cols);
        if ($rowspan == null) { $rowspan = 1; }
        if ($colspan == null) { $colspan = 1; }
        if ($row < 0 || $col < 0 || $row + $rowspan > $rows || $col + $colspan > $cols) {
            throw "bplot.subplotSpan: cell (" + str($row) + "," + str($col) + ") spanning " +
                  str($rowspan) + "x" + str($colspan) + " does not fit in a " +
                  str($rows) + "x" + str($cols) + " grid";
        }
        return $this.addAxesIn($this.cellRect($row, $col, $rows, $cols, $rowspan, $colspan));
    }

    def gca() {
        if (len($this.axesList) == 0) { return $this.addAxes(); }
        return $this.axesList[$this.cur];
    }

    def setCurrent($ax) { $this.cur = $this.indexOf($ax); return $ax; }

    def indexOf($ax) {
        $i = 0;
        while ($i < len($this.axesList)) {
            if ($this.axesList[$i] == $ax) { return $i; }
            $i = $i + 1;
        }
        throw "bplot: that axes does not belong to this figure";
    }

    // ── Sharing and twins ────────────────────────────────────────────────
    //
    // Share groups live on the FIGURE, as lists of INDICES. Two reasons, and
    // the first is not style:
    //
    //   1. Axes holding each other would be a reference cycle, and cycles are
    //      never freed (see BPlotAxes.init). Indices point at nothing.
    //   2. Sharing is symmetric. A parent pointer makes the first axes
    //      special, which breaks the moment it is the one removed.
    // NOTE the assignment back. A list passed to a function is a COPY in Bantu,
    // so _joinGroup cannot mutate the caller's list; it returns the new one and
    // the field is rewritten. Writing through the field is what makes the
    // change visible, because the instance is reference-semantic.
    def shareX($a, $b) { $this.xgroups = _joinGroup($this.xgroups, $this.indexOf($a), $this.indexOf($b)); return $this; }
    def shareY($a, $b) { $this.ygroups = _joinGroup($this.ygroups, $this.indexOf($a), $this.indexOf($b)); return $this; }

    // A second y axis over the same rectangle, sharing x — the classic
    // temperature-and-rainfall chart. Its ticks go on the right, and it does
    // NOT redraw the frame: a doubled 1px stroke is visible and reads as a
    // rendering bug.
    def twinx($ax) {
        $tw = $this.addAxesIn($ax.cell);
        $tw.pads = $ax.pads;
        $tw.applyPads();
        $tw.side = "right";
        $tw.hostIdx = $this.indexOf($ax);
        $this.shareX($ax, $tw);
        return $tw;
    }

    def twiny($ax) {
        $tw = $this.addAxesIn($ax.cell);
        $tw.pads = $ax.pads;
        $tw.applyPads();
        $tw.side = "top";
        $tw.hostIdx = $this.indexOf($ax);
        $this.shareY($ax, $tw);
        return $tw;
    }

    // The limits an axes must use once its share group is taken into account:
    // the union of every member's data limits. Returns null when the axes is
    // in no group, so an unshared axes takes exactly the B1 path.
    def sharedLimits($idx, $which) {
        $groups = $this.xgroups;
        if ($which == "y") { $groups = $this.ygroups; }
        $g = _groupWith($groups, $idx);
        if ($g == null) { return null; }
        $lo = null;
        $hi = null;
        $i = 0;
        while ($i < len($g)) {
            $d = $this.axesList[$g[$i]].dataLimits($which);
            if ($d[0] != null && $d[1] != null) {
                if ($lo == null || $d[0] < $lo) { $lo = $d[0]; }
                if ($hi == null || $d[1] > $hi) { $hi = $d[1]; }
            }
            $i = $i + 1;
        }
        if ($lo == null) { return null; }
        return [$lo, $hi];
    }

    // tight_layout: recompute every axes' pads from the text that will
    // actually be drawn. It runs at RENDER time (see to_svg), not here, because
    // labels are normally set after the axes exists -- running it now would
    // measure an empty axes, which is matplotlib's trap and the reason its
    // answer is "call it last".
    def tight_layout($on) {
        if ($on == null) { $on = true; }
        $this.tight = $on;
        return $this;
    }

    def to_svg() { return $this.draw(new BPlotSvg($this.w, $this.h)); }

    // The PNG, as bytes. dpi defaults to 96: one pixel per unit, the SVG's size.
    def to_png($dpi) {
        if (!has_native("raster")) {
            throw "bplot.to_png: this interpreter has no raster backend -- save as .svg instead";
        }
        if ($dpi == null) { $dpi = 96; }
        return $this.draw(new BPlotRaster($this.w, $this.h, $dpi));
    }

    def draw($bk) {
        $bk.open();
        $bk.rect(0, 0, $this.w, $this.h, $_STYLE["bg"], null, null);
        // tight_layout runs HERE, not when it was called, because the labels
        // it measures are normally set after the axes exists (decision BP24).
        if ($this.tight) {
            $i = 0;
            while ($i < len($this.axesList)) {
                _tightenAxes($bk, $this.axesList[$i], $this.sharedLimits($i, "x"), $this.sharedLimits($i, "y"));
                $i = $i + 1;
            }
            // A twin and its host are ONE rectangle. Each was measured for its
            // own labels, so both take the larger pad on every side.
            $i = 0;
            while ($i < len($this.axesList)) {
                $tw = $this.axesList[$i];
                if ($tw.hostIdx != null) {
                    $host = $this.axesList[$tw.hostIdx];
                    $p = [max($host.pads[0], $tw.pads[0]), max($host.pads[1], $tw.pads[1]),
                          max($host.pads[2], $tw.pads[2]), max($host.pads[3], $tw.pads[3])];
                    $host.pads = $p;
                    $host.applyPads();
                    $tw.pads = $p;
                    $tw.applyPads();
                }
                $i = $i + 1;
            }
        }
        $i = 0;
        while ($i < len($this.axesList)) {
            _drawAxes($bk, $this.axesList[$i],
                      $this.sharedLimits($i, "x"), $this.sharedLimits($i, "y"));
            $i = $i + 1;
        }
        $bk.close();
        return $bk.render();
    }

    // savefig("chart.svg") or savefig("chart.png", {"dpi": 150}).
    def savefig($path, $opts) {
        if ($path == null || type($path) != "string" || len($path) == 0) {
            throw "bplot.savefig: needs a file path";
        }
        if (_endsWith($path, ".png")) {
            $dpi = null;
            if ($opts != null) { $dpi = $opts["dpi"]; }
            writefile($path, $this.to_png($dpi), "wb");
            return $path;
        }
        if (!_endsWith($path, ".svg")) {
            throw "bplot.savefig: save as .svg or .png, got '" + $path + "'";
        }
        writefile($path, $this.to_svg());
        return $path;
    }
}

// Union two axes into one share group, merging groups that already exist.
// Sharing is transitive: sharing A with B and B with C shares all three, and
// a reader who has to remember which pairing came first has been given a
// worse tool than no sharing at all.
def _joinGroup($groups, $i, $j) {
    $gi = 0 - 1;
    $gj = 0 - 1;
    $k = 0;
    while ($k < len($groups)) {
        if (_hasIdx($groups[$k], $i)) { $gi = $k; }
        if (_hasIdx($groups[$k], $j)) { $gj = $k; }
        $k = $k + 1;
    }
    if ($gi < 0 && $gj < 0) { push($groups, [$i, $j]); return $groups; }
    if ($gi >= 0 && $gj < 0) { push($groups[$gi], $j); return $groups; }
    if ($gj >= 0 && $gi < 0) { push($groups[$gj], $i); return $groups; }
    if ($gi == $gj) { return $groups; }
    // Two existing groups meet: merge the second into the first and leave the
    // second empty rather than removing it, so the surviving indices stay
    // valid.
    extend($groups[$gi], $groups[$gj]);
    $groups[$gj] = [];
    return $groups;
}

def _hasIdx($g, $i) {
    $k = 0;
    while ($k < len($g)) { if ($g[$k] == $i) { return true; } $k = $k + 1; }
    return false;
}

def _groupWith($groups, $i) {
    $k = 0;
    while ($k < len($groups)) {
        if (_hasIdx($groups[$k], $i)) { return $groups[$k]; }
        $k = $k + 1;
    }
    return null;
}

def _checkGrid($rows, $cols) {
    if (type($rows) != "number" || type($cols) != "number" ||
        $rows < 1 || $cols < 1 || $rows != floor($rows) || $cols != floor($cols)) {
        throw "bplot: a subplot grid needs whole row and column counts of at least 1, got " +
              str($rows) + "x" + str($cols);
    }
    if ($rows * $cols > 400) {
        throw "bplot: a " + str($rows) + "x" + str($cols) + " grid is " + str($rows * $cols) +
              " panels -- past a few dozen each one is a handful of pixels. Draw several figures.";
    }
    return true;
}

def _endsWith($s, $suffix) {
    if (len($s) < len($suffix)) { return false; }
    return substr($s, len($s) - len($suffix), len($suffix)) == $suffix;
}

// ════════════════════════════════════════════════════════════════════════
//  Rendering
// ════════════════════════════════════════════════════════════════════════

// The transform, built once per axes and handed to every artist.
//
// It holds the view limits in DATA space (which is what ticks are chosen in)
// and the affine coefficients in TRANSFORMED space (which is what pixels are
// computed from). Artists never see a scale name unless they are projecting a
// bulk sequence; they multiply-add.
//
// THE Y FLIP IS STILL IN EXACTLY ONE PLACE: BPlotAxes.coefY, which this calls.
// It is not re-derived here, because two formulas that must agree are two
// formulas that will eventually disagree.
def _mkT($ax, $xshare, $yshare) {
    $xlim = $ax.viewLimitsShared("x", $xshare);
    $ylim = $ax.viewLimitsShared("y", $yshare);
    $txlo = _fwd($xlim[0], $ax.xscale, $ax.xlt);
    $txhi = _fwd($xlim[1], $ax.xscale, $ax.xlt);
    $tylo = _fwd($ylim[0], $ax.yscale, $ax.ylt);
    $tyhi = _fwd($ylim[1], $ax.yscale, $ax.ylt);
    // Only EQUAL bounds are degenerate. A descending pair is an inverted axis,
    // which is deliberate (imshow), and the coefficients handle it by changing
    // sign — so this must not "helpfully" reorder them.
    if ($txhi == $txlo) { $txhi = $txlo + 1; }
    if ($tyhi == $tylo) { $tyhi = $tylo + 1; }
    $cx = $ax.coefX([$txlo, $txhi]);
    $cy = $ax.coefY([$tylo, $tyhi]);
    return {"xlim": $xlim, "ylim": $ylim,
            "xs": $ax.xscale, "ys": $ax.yscale, "xlt": $ax.xlt, "ylt": $ax.ylt,
            "xa": $cx[0], "xb": $cx[1], "ya": $cy[0], "yb": $cy[1]};
}

// One value -> one pixel. For ticks, bars, baselines: a handful of calls.
// Bulk sequences go through _pxs/_pys instead, which project once.
def _TX($T, $v) { return _fwd($v, $T["xs"], $T["xlt"]) * $T["xa"] + $T["xb"]; }
def _TY($T, $v) { return _fwd($v, $T["ys"], $T["ylt"]) * $T["ya"] + $T["yb"]; }

// Bulk, for artists that need a materialised pixel list anyway (errorbar,
// stem, violin — each of which emits several elements per point, so one more
// pass over the data is lost in the noise).
//
// THE HOT ARTISTS DO NOT USE THESE. A line or a scatter fuses the projection
// and the affine step into the single loop B1 had, because measuring showed
// the extra list cost 100,000 points 1,480 ms against 1,136 — a 30% tax on
// LINEAR axes for a feature only log and symlog use. See _drawLine.
def _pxs($T, $vals) {
    $v = $vals;
    if ($T["xs"] != "linear") { $v = _project($vals, $T["xs"], $T["xlt"]); }
    $a = $T["xa"];
    $b = $T["xb"];
    $out = [];
    $i = 0;
    $n = len($v);
    while ($i < $n) { push($out, $v[$i] * $a + $b); $i = $i + 1; }
    return $out;
}
def _pys($T, $vals) {
    $v = $vals;
    if ($T["ys"] != "linear") { $v = _project($vals, $T["ys"], $T["ylt"]); }
    $a = $T["ya"];
    $b = $T["yb"];
    $out = [];
    $i = 0;
    $n = len($v);
    while ($i < $n) { push($out, $v[$i] * $a + $b); $i = $i + 1; }
    return $out;
}

// Which ticks does this axis get, and what do they read?
// Returns {"major": [...], "labels": [...], "minor": [...]}.
def _axisTicks($ax, $which, $lim0) {
    $scale = $ax.scaleOf($which);
    // Locators all work over an ascending range; an inverted axis is a
    // property of the TRANSFORM, not of where the round numbers are.
    $lim = $lim0;
    if ($lim[1] < $lim[0]) { $lim = [$lim0[1], $lim0[0]]; }
    $over = $ax.xtickOverride;
    if ($which == "y") { $over = $ax.ytickOverride; }
    if ($over != null && $scale == "linear" && !($which == "x" && $ax.xdate)) {
        return {"major": $over["pos"], "labels": $over["labels"], "minor": []};
    }
    if ($which == "x" && $ax.xdate) {
        $d = _dateTicks($lim[0], $lim[1]);
        $lab = [];
        $i = 0;
        while ($i < len($d["ticks"])) {
            push($lab, _fmtDate($d["ticks"][$i], $d["unit"]));
            $i = $i + 1;
        }
        return {"major": $d["ticks"], "labels": $lab, "minor": []};
    }
    if ($scale == "log" || $scale == "symlog") {
        $t = {};
        if ($scale == "log") { $t = _ticksLog($lim[0], $lim[1]); }
        else { $t = _ticksSymlog($lim[0], $lim[1], $ax.ltOf($which)); }
        $lab = [];
        $i = 0;
        while ($i < len($t["major"])) { push($lab, _fmtDecade($t["major"][$i])); $i = $i + 1; }
        return {"major": $t["major"], "labels": $lab, "minor": $t["minor"]};
    }
    $t = _ticks($lim[0], $lim[1], 6);
    $d = _decimalsFor(_tickStep($t));
    $lab = [];
    $i = 0;
    while ($i < len($t)) { push($lab, _fmt($t[$i], $d)); $i = $i + 1; }
    return {"major": $t, "labels": $lab, "minor": []};
}

// tight_layout (decision BP24): compute each pad from the text that will
// actually be drawn, using the embedded Helvetica metrics.
//
// It always pads OUTWARD. The viewer picks the font, so the measurement is an
// estimate; an overestimate is whitespace, an underestimate is a collision.
// That asymmetry is why an approximate metrics table is good enough to be
// useful, and it is deliberate rather than sloppy.
def _tightenAxes($bk, $ax, $xshare, $yshare) {
    // Measure against the CURRENT box, then re-inset the cell. Measuring is
    // not perfectly self-consistent -- moving the box can change the ticks,
    // which changes the widest label -- so this runs twice, which in practice
    // converges. A third pass has never changed the answer in testing.
    $pass = 0;
    while ($pass < 2) {
        $xlim = $ax.viewLimitsShared("x", $xshare);
        $ylim = $ax.viewLimitsShared("y", $yshare);
        $xt = _axisTicks($ax, "x", $xlim);
        $yt = _axisTicks($ax, "y", $ylim);

        $ywidest = 0;
        $i = 0;
        while ($i < len($yt["labels"])) {
            $w = $bk.textWidth($yt["labels"][$i], 11);
            if ($w > $ywidest) { $ywidest = $w; }
            $i = $i + 1;
        }
        // tick length + gap + the widest label + a little air
        $left = 4 + 6 + $ywidest + 8;
        // A rotated label sits 14px inside the cell edge and is ~12px tall, so
        // it needs 26px of its own before the tick labels start.
        if (len($ax.ylabel) > 0) { $left = $left + 22; }

        $bottom = 4 + 17 + 6;
        if (len($ax.xlabel) > 0) { $bottom = $bottom + 20; }

        $top = 10;
        if (len($ax.title) > 0) { $top = 14 + 12; }

        // The last x tick label is centred on the frame's right edge, so half
        // of it hangs outside. Reserve that half, or it is clipped by the
        // figure edge -- which looks like a broken label, not a tight layout.
        // matplotlib's tight_layout keeps a pad of about 1.08 font sizes on
        // every side; without it the frame sits flush against the figure edge
        // and, in a grid, the right-hand column looks cropped.
        $right = 14;
        if (len($xt["labels"]) > 0) {
            $half = $bk.textWidth($xt["labels"][len($xt["labels"]) - 1], 11) / 2;
            if ($half + 8 > $right) { $right = $half + 8; }
        }
        if ($ax.side == "right") {
            // A twin puts its labels on the other side; swap what is reserved.
            $swap = $left;
            $left = $right;
            $right = $swap;
        }
        $ax.pads = [$left, $right, $top, $bottom];
        $ax.applyPads();
        $pass = $pass + 1;
    }
    return null;
}

def _drawAxes($bk, $ax, $xshare, $yshare) {
    $T = _mkT($ax, $xshare, $yshare);
    $xlim = $T["xlim"];
    $ylim = $T["ylim"];
    $xt = _axisTicks($ax, "x", $xlim);
    $yt = _axisTicks($ax, "y", $ylim);

    $x0 = $ax.left;
    $y0 = $ax.top;
    $x1 = $ax.left + $ax.w;
    $y1 = $ax.top + $ax.h;

    // Grid, behind everything. Minor gridlines are drawn lighter, because on a
    // log axis there are eight of them per decade and at full weight they read
    // as the data.
    if ($ax.gridOn) {
        $i = 0;
        while ($i < len($xt["minor"])) {
            $px = _TX($T, $xt["minor"][$i]);
            $bk.line($px, $y0, $px, $y1, $_STYLE["minor"], 1);
            $i = $i + 1;
        }
        $i = 0;
        while ($i < len($yt["minor"])) {
            $py = _TY($T, $yt["minor"][$i]);
            $bk.line($x0, $py, $x1, $py, $_STYLE["minor"], 1);
            $i = $i + 1;
        }
        $i = 0;
        while ($i < len($xt["major"])) {
            $px = _TX($T, $xt["major"][$i]);
            $bk.line($px, $y0, $px, $y1, $_STYLE["grid"], 1);
            $i = $i + 1;
        }
        $i = 0;
        while ($i < len($yt["major"])) {
            $py = _TY($T, $yt["major"][$i]);
            $bk.line($x0, $py, $x1, $py, $_STYLE["grid"], 1);
            $i = $i + 1;
        }
    }

    // Data, clipped to the box so a point outside the view cannot escape it.
    $clipId = $bk.clip($x0, $y0, $ax.w, $ax.h);
    $bk.groupOpen($clipId);
    $i = 0;
    while ($i < len($ax.artists)) {
        _drawArtist($bk, $ax, $ax.artists[$i], $T);
        $i = $i + 1;
    }
    $i = 0;
    while ($i < len($ax.texts)) {
        _drawText($bk, $ax.texts[$i], $T);
        $i = $i + 1;
    }
    $bk.groupClose();

    if ($ax.frameOn) {
        // A twin shares the rectangle with the axes it was made from, so it
        // draws ONLY its own axis: the frame is stroked once, because a
        // doubled 1px stroke is visible and reads as a rendering bug.
        $primary = $ax.side == "left";
        if ($primary) { $bk.rect($x0, $y0, $ax.w, $ax.h, "none", $_STYLE["axis"], 1); }

        // X ticks: the bottom edge normally, the top edge for a twiny.
        if ($ax.side != "right") {
            $ty = $y1;
            $dir = 1;
            $lab = $y1 + 17;
            if ($ax.side == "top") { $ty = $y0; $dir = 0 - 1; $lab = $y0 - 8; }
            $i = 0;
            while ($i < len($xt["minor"])) {
                $px = _TX($T, $xt["minor"][$i]);
                $bk.line($px, $ty, $px, $ty + 2 * $dir, $_STYLE["axis"], 1);
                $i = $i + 1;
            }
            $i = 0;
            while ($i < len($xt["major"])) {
                $px = _TX($T, $xt["major"][$i]);
                $bk.line($px, $ty, $px, $ty + 4 * $dir, $_STYLE["axis"], 1);
                $bk.text($px, $lab, $xt["labels"][$i], 11, $_STYLE["fg"], "middle", null);
                $i = $i + 1;
            }
            if (len($ax.xlabel) > 0) {
                // Positioned against the CELL, not by a fixed offset from the
                // frame. With computed pads the frame moves, and a constant
                // offset put the label outside the figure, where it was
                // silently clipped -- which is the failure tight_layout
                // exists to prevent.
                $ly = $ax.cell[1] + $ax.cell[3] - 8;
                if ($ax.side == "top") { $ly = $ax.cell[1] + 14; }
                $bk.text(($x0 + $x1) / 2, $ly, $ax.xlabel, 12, $_STYLE["fg"], "middle", null);
            }
        }

        // Y ticks: the left edge normally, the right edge for a twinx.
        if ($ax.side != "top") {
            $tx = $x0;
            $dir = 0 - 1;
            $anchor = "end";
            $lx = $x0 - 8;
            if ($ax.side == "right") { $tx = $x1; $dir = 1; $anchor = "start"; $lx = $x1 + 8; }
            $i = 0;
            while ($i < len($yt["minor"])) {
                $py = _TY($T, $yt["minor"][$i]);
                $bk.line($tx + 2 * $dir, $py, $tx, $py, $_STYLE["axis"], 1);
                $i = $i + 1;
            }
            $i = 0;
            while ($i < len($yt["major"])) {
                $py = _TY($T, $yt["major"][$i]);
                $bk.line($tx + 4 * $dir, $py, $tx, $py, $_STYLE["axis"], 1);
                $bk.text($lx, $py + 4, $yt["labels"][$i], 11, $_STYLE["fg"], $anchor, null);
                $i = $i + 1;
            }
            if (len($ax.ylabel) > 0) {
                $lxx = $ax.cell[0] + 14;
                $rot = -90;
                if ($ax.side == "right") { $lxx = $ax.cell[0] + $ax.cell[2] - 14; $rot = 90; }
                $bk.text($lxx, ($y0 + $y1) / 2, $ax.ylabel, 12, $_STYLE["fg"], "middle", $rot);
            }
        }
    }

    if (len($ax.title) > 0) {
        $ty = $y0 - 12;
        if ($ty < $ax.cell[1] + 14) { $ty = $ax.cell[1] + 14; }
        $bk.text(($x0 + $x1) / 2, $ty, $ax.title, 14, $_STYLE["fg"], "middle", null);
    }

    if ($ax.cbar != null) { _drawColorbar($bk, $ax); }
    if ($ax.legendOn) { _drawLegend($bk, $ax, $x1, $y0); }
}

def _drawArtist($bk, $ax, $a, $T) {
    $k = $a["kind"];
    if ($k == "line")     { _drawLine($bk, $ax, $a, $T); return null; }
    if ($k == "scatter")  { _drawScatter($bk, $ax, $a, $T); return null; }
    if ($k == "bar")      { _drawBar($bk, $ax, $a, $T); return null; }
    if ($k == "hist")     { _drawHist($bk, $ax, $a, $T); return null; }
    if ($k == "box")      { _drawBox($bk, $ax, $a, $T); return null; }
    if ($k == "violin")   { _drawViolin($bk, $ax, $a, $T); return null; }
    if ($k == "errorbar") { _drawErrorbar($bk, $ax, $a, $T); return null; }
    if ($k == "band")     { _drawBand($bk, $ax, $a, $T); return null; }
    if ($k == "step")     { _drawStep($bk, $ax, $a, $T); return null; }
    if ($k == "stem")     { _drawStem($bk, $ax, $a, $T); return null; }
    if ($k == "pie")      { _drawPie($bk, $ax, $a, $T); return null; }
    if ($k == "image")    { _drawImage($bk, $ax, $a, $T); return null; }
    if ($k == "mesh")     { _drawMesh($bk, $ax, $a, $T); return null; }
    if ($k == "contour")  { _drawContour($bk, $ax, $a, $T); return null; }
    return null;
}

// ── The 2-D encoder (decision BP26) ──────────────────────────────────────
//
// One <rect> per cell makes a 1000x1000 image a 55 MB document. A colormapped
// image has at most 256 distinct colours BY CONSTRUCTION, so every cell of one
// colour goes into a single <path> as `M x y h w v h h -w Z` subpaths, with
// horizontally adjacent equal-coloured cells merged into one run first.
//
// The result is <= 257 elements regardless of grid size, and about 14 bytes
// per cell instead of 55.
def _cellRect($x0, $y0, $x1, $y1) {
    $x = min($x0, $x1);
    $y = min($y0, $y1);
    $w = abs($x1 - $x0);
    $h = abs($y1 - $y0);
    // A hairline overlap stops the background showing through as a pale grid
    // between cells -- browsers antialias adjacent edges independently.
    $w = $w + 0.5;
    $h = $h + 0.5;
    return "M" + _px($x) + " " + _px($y) + "h" + _px($w) + "v" + _px($h) +
           "h" + _px(0 - $w) + "Z";
}

// Quantised colour index, or -1 for "no data".
def _cellIndex($v, $vmin, $vmax) {
    if (!isfinite($v)) { return 0 - 1; }
    if ($vmax <= $vmin) { return 128; }
    return round(clamp(($v - $vmin) / ($vmax - $vmin), 0, 1) * 255);
}

// Emit the accumulated per-colour paths. Ordered by index so the output is
// deterministic -- a document that differs run to run cannot be diffed, and a
// golden-file test would be useless.
def _flushCells($bk, $buckets, $cmap) {
    $ks = sort(keys($buckets));
    $i = 0;
    while ($i < len($ks)) {
        $k = $ks[$i];
        $idx = num($k);
        $col = $_CM_BAD;
        if ($idx >= 0) { $col = _cmap($cmap, $idx / 255); }
        $d = $buckets[$k];
        if (type($d) == "list") { $d = join($d, ""); }
        $bk.path($d, $col, null, null);
        $i = $i + 1;
    }
    return null;
}

def _drawGridCells($bk, $ax, $a, $T, $xedges, $yedges) {
    $rows = $a["rows"];
    $vmin = $a["vmin"];
    $vmax = $a["vmax"];
    if ($_NAT["on"]) {
        // bp_grid_paths is the loop below in C++ (plot_native.hpp). The edges
        // go in as pixels, transformed here, exactly as the loop transforms them.
        $xpx = [];
        $i = 0;
        while ($i < len($xedges)) { push($xpx, _TX($T, $xedges[$i])); $i = $i + 1; }
        $ypx = [];
        $i = 0;
        while ($i < len($yedges)) { push($ypx, _TY($T, $yedges[$i])); $i = $i + 1; }
        _flushCells($bk, bp_grid_paths($rows, $vmin, $vmax, $xpx, $ypx), $a["cmap"]);
        return null;
    }
    $buckets = {};
    $r = 0;
    while ($r < $a["h"]) {
        $row = $rows[$r];
        $yA = _TY($T, $yedges[$r]);
        $yB = _TY($T, $yedges[$r + 1]);
        $c = 0;
        while ($c < $a["w"]) {
            // Merge the run of identical colours starting here.
            $idx = _cellIndex($row[$c], $vmin, $vmax);
            $c2 = $c + 1;
            while ($c2 < $a["w"] && _cellIndex($row[$c2], $vmin, $vmax) == $idx) { $c2 = $c2 + 1; }
            $xA = _TX($T, $xedges[$c]);
            $xB = _TX($T, $xedges[$c2]);
            $key = _fmt($idx, 0);
            if ($buckets[$key] == null) { $buckets[$key] = []; }
            push($buckets[$key], _cellRect($xA, $yA, $xB, $yB));
            $c = $c2;
        }
        $r = $r + 1;
    }
    _flushCells($bk, $buckets, $a["cmap"]);
    return null;
}

def _edgesFrom($lo, $hi, $n) {
    $e = [];
    $i = 0;
    while ($i <= $n) { push($e, $lo + ($hi - $lo) * $i / $n); $i = $i + 1; }
    return $e;
}

def _drawImage($bk, $ax, $a, $T) {
    $ext = $a["ext"];
    $xe = _edgesFrom($ext[0], $ext[1], $a["w"]);
    $ye = _edgesFrom($ext[2], $ext[3], $a["h"]);
    _drawGridCells($bk, $ax, $a, $T, $xe, $ye);
    if ($a["labels"]) { _drawCellLabels($bk, $a, $T, $xe, $ye); }
    return null;
}

def _drawMesh($bk, $ax, $a, $T) {
    _drawGridCells($bk, $ax, $a, $T, $a["xe"], $a["ye"]);
    return null;
}

// The value written into each cell of a heatmap. Black or white, chosen by
// the cell's own luminance -- a fixed colour is unreadable over half of any
// colormap, and this is the one place where a wrong choice makes the number
// disappear entirely.
def _drawCellLabels($bk, $a, $T, $xe, $ye) {
    $rows = $a["rows"];
    $d = _decimalsFor(($a["vmax"] - $a["vmin"]) / 100);
    if ($d > 3) { $d = 3; }
    $r = 0;
    while ($r < $a["h"]) {
        $yM = (_TY($T, $ye[$r]) + _TY($T, $ye[$r + 1])) / 2;
        $c = 0;
        while ($c < $a["w"]) {
            $v = $rows[$r][$c];
            if (isfinite($v)) {
                $xM = (_TX($T, $xe[$c]) + _TX($T, $xe[$c + 1])) / 2;
                $idx = _cellIndex($v, $a["vmin"], $a["vmax"]);
                $bk.text($xM, $yM + 4, _fmt($v, $d), 10, _readableOn($a["cmap"], $idx), "middle", null);
            }
            $c = $c + 1;
        }
        $r = $r + 1;
    }
    return null;
}

def _readableOn($cmap, $idx) {
    if ($idx < 0) { return $_STYLE["fg"]; }
    return _contrastOn(_cmap($cmap, $idx / 255));
}

// Black or white, whichever is legible on the given fill. A FIXED colour is
// unreadable over half of any palette -- white text vanishes on the "print"
// style's light greys, black text vanishes on a dark slice -- and this is the
// one place where the wrong choice makes the number disappear entirely.
def _contrastOn($hex) {
    if ($hex == null || len($hex) != 7) { return $_STYLE["fg"]; }
    // Rec. 601 luma, the standard weighting for perceived brightness.
    $r = _hexPair($hex, 1);
    $g = _hexPair($hex, 3);
    $b = _hexPair($hex, 5);
    if (0.299 * $r + 0.587 * $g + 0.114 * $b > 140) { return "#000000"; }
    return "#ffffff";
}

def _hexPair($s, $at) {
    return _hexVal(ord($s[$at])) * 16 + _hexVal(ord($s[$at + 1]));
}
def _hexVal($c) {
    if ($c >= 48 && $c <= 57) { return $c - 48; }
    if ($c >= 97 && $c <= 102) { return $c - 87; }
    return $c - 55;
}

// ── Contour: marching squares ────────────────────────────────────────────
//
// Each cell's four corners are above or below the level, giving 16 cases.
// Segment ends are placed by LINEAR INTERPOLATION along the edge rather than
// at its midpoint: midpoints give a visibly faceted contour that looks like a
// rendering artefact, and the interpolation is one subtraction and one divide.
//
// Every segment for a level goes into ONE <path>, so a 256x256 grid at eight
// levels is eight elements, not thousands of <line>s.
def _mix($v0, $v1, $lev, $p0, $p1) {
    $d = $v1 - $v0;
    if ($d == 0) { return $p0; }
    return $p0 + ($p1 - $p0) * ($lev - $v0) / $d;
}

def _drawContour($bk, $ax, $a, $T) {
    $rows = $a["rows"];
    $ext = $a["ext"];
    $w = $a["w"];
    $h = $a["h"];
    $li = 0;
    while ($li < len($a["levels"])) {
        $lev = $a["levels"][$li];
        $segs = [];
        $r = 0;
        while ($r < $h - 1) {
            $c = 0;
            while ($c < $w - 1) {
                // Corner values, clockwise from top-left.
                $v00 = $rows[$r][$c];
                $v01 = $rows[$r][$c + 1];
                $v11 = $rows[$r + 1][$c + 1];
                $v10 = $rows[$r + 1][$c];
                if (isfinite($v00) && isfinite($v01) && isfinite($v11) && isfinite($v10)) {
                    // Corner positions in data space.
                    $xa = $ext[0] + ($ext[1] - $ext[0]) * $c / ($w - 1);
                    $xb = $ext[0] + ($ext[1] - $ext[0]) * ($c + 1) / ($w - 1);
                    $ya = $ext[2] + ($ext[3] - $ext[2]) * $r / ($h - 1);
                    $yb = $ext[2] + ($ext[3] - $ext[2]) * ($r + 1) / ($h - 1);
                    // Crossing points on each of the four edges, when present.
                    $pts = [];
                    if (($v00 < $lev) != ($v01 < $lev)) { push($pts, _mix($v00, $v01, $lev, $xa, $xb)); push($pts, $ya); }
                    if (($v01 < $lev) != ($v11 < $lev)) { push($pts, $xb); push($pts, _mix($v01, $v11, $lev, $ya, $yb)); }
                    if (($v10 < $lev) != ($v11 < $lev)) { push($pts, _mix($v10, $v11, $lev, $xa, $xb)); push($pts, $yb); }
                    if (($v00 < $lev) != ($v10 < $lev)) { push($pts, $xa); push($pts, _mix($v00, $v10, $lev, $ya, $yb)); }
                    // Two crossings is one segment. Four is the ambiguous
                    // saddle: both pairings are defensible, and joining them
                    // in edge order is the conventional resolution.
                    $k = 0;
                    while ($k + 3 < len($pts)) {
                        push($segs, "M" + _px(_TX($T, $pts[$k])) + " " + _px(_TY($T, $pts[$k + 1])) +
                                    "L" + _px(_TX($T, $pts[$k + 2])) + " " + _px(_TY($T, $pts[$k + 3])));
                        $k = $k + 4;
                    }
                }
                $c = $c + 1;
            }
            $r = $r + 1;
        }
        if (len($segs) > 0) {
            $col = $a["color"];
            if ($a["cmap"] != null && $a["vmax"] > $a["vmin"]) {
                $col = _cmap($a["cmap"], ($lev - $a["vmin"]) / ($a["vmax"] - $a["vmin"]));
            }
            $bk.path(join($segs, ""), "none", $col, $a["width"]);
        }
        $li = $li + 1;
    }
    return null;
}

// ── Colorbar ─────────────────────────────────────────────────────────────
// Drawn as a stack of thin rectangles rather than an SVG <linearGradient>,
// because a gradient is an SVG-only construct: the raster backend of B6 would
// have to reimplement it, and the backend's job is to stay small.
def _drawColorbar($bk, $ax) {
    $cb = $ax.cbar;
    $x = $ax.left + $ax.w + $cb["gap"] - 24;
    $y0 = $ax.top;
    $hgt = $ax.h;
    $n = 64;
    $i = 0;
    while ($i < $n) {
        $t = ($n - 1 - $i) / ($n - 1);
        // Half a pixel of overlap, so no seam shows between the bands.
        $bk.rect($x, $y0 + $hgt * $i / $n, $cb["width"], $hgt / $n + 0.5,
                 _cmap($cb["cmap"], $t), null, null);
        $i = $i + 1;
    }
    $bk.rect($x, $y0, $cb["width"], $hgt, "none", $_STYLE["axis"], 1);
    $ticks = _ticks($cb["vmin"], $cb["vmax"], 5);
    $d = _decimalsFor(_tickStep($ticks));
    $span = $cb["vmax"] - $cb["vmin"];
    $i = 0;
    while ($i < len($ticks)) {
        if ($span > 0) {
            $ty = $y0 + $hgt * (1 - ($ticks[$i] - $cb["vmin"]) / $span);
            $bk.line($x + $cb["width"], $ty, $x + $cb["width"] + 4, $ty, $_STYLE["axis"], 1);
            $bk.text($x + $cb["width"] + 7, $ty + 4, _fmt($ticks[$i], $d), 10, $_STYLE["fg"], "start", null);
        }
        $i = $i + 1;
    }
    if ($cb["label"] != null) {
        $bk.text($x + $cb["width"] + 34, $y0 + $hgt / 2, str($cb["label"]), 11, $_STYLE["fg"], "middle", -90);
    }
    return null;
}

// NaN breaks the line, as it does in matplotlib. Untreated, a NaN coordinate
// emits points="NaN,12 ..." which every browser renders as NOTHING AT ALL,
// with no error anywhere -- the worst failure mode available. Splitting draws
// the valid segments and makes the gap visible as a gap.
// The scale pre-pass over an ndarray, as numba passes that reproduce _fwd bit
// for bit: the same libm calls in the same order. A non-positive value on a log
// axis raises through _fwd itself, so the message is the same one, naming the
// first offending value in data order.
def _projectArr($a, $scale, $lt) {
    if ($scale == "linear") { return $a; }
    if ($scale == "log") {
        $bad = nd_logical_and(nd_isfinite($a), nd_less_equal($a, 0, null), null);
        if (nd_to_list(nd_any($bad, null))) { _fwd($a[nd_to_list(nd_argmax($bad, null))], "log", $lt); }
        return nd_log10($a, null);
    }
    $abs = nd_abs($a, null);
    $inside = nd_multiply($a, $_LINADJ, null);
    $sgnlt = nd_multiply(nd_sign($a, null), $lt, null);
    $outside = nd_multiply($sgnlt, nd_add(nd_log10(nd_divide($abs, $lt, null), null), $_LINADJ, null), null);
    return nd_where(nd_less_equal($abs, $lt, null), $inside, $outside);
}

// Pixels as separate numba passes -- multiply, then add -- so no compiler can
// fuse them into a multiply-add that rounds differently from the interpreter's
// `x * a + b` (BP33). `finite` is decided on the DATA, as the pure path does.
def _pixelsArr($a, $T) {
    $xs = _projectArr($a["x"], $T["xs"], $T["xlt"]);
    $ys = _projectArr($a["y"], $T["ys"], $T["ylt"]);
    return {
        "fin": nd_logical_and(nd_isfinite($xs), nd_isfinite($ys), null),
        "px":  nd_add(nd_multiply($xs, $T["xa"], null), $T["xb"], null),
        "py":  nd_add(nd_multiply($ys, $T["ya"], null), $T["yb"], null)
    };
}

def _drawLineNative($bk, $ax, $a, $T) {
    $P = _pixelsArr($a, $T);
    $runs = bp_line_runs($P["px"], $P["py"], $P["fin"], $ax.simplify);
    each ($run in $runs) {
        if (len($run) >= 4) { $bk.polyline($run, $a["color"], $a["width"], $a["dash"]); }
        else { $bk.circle($run[0], $run[1], $a["width"], $a["color"]); }
    }
    return null;
}

def _drawLine($bk, $ax, $a, $T) {
    if ($a["native"] == true) { return _drawLineNative($bk, $ax, $a, $T); }
    // The pre-pass happens here and only when it is needed; the loop below is
    // B1's, unchanged, so a linear axis runs exactly the code it used to.
    $xs = $a["x"];
    $ys = $a["y"];
    if ($T["xs"] != "linear") { $xs = _project($xs, $T["xs"], $T["xlt"]); }
    if ($T["ys"] != "linear") { $ys = _project($ys, $T["ys"], $T["ylt"]); }
    $xa = $T["xa"]; $xb = $T["xb"];
    $ya = $T["ya"]; $yb = $T["yb"];
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

// At SCATTER_BATCH finite points and above -- BP3's own threshold -- a series
// is one <path> of circles instead of one <circle> each. Each circle is two
// half-arcs, the smallest exact circle a path can express. Below it nothing
// changes, so no existing smaller document moves.
$_SCATTER_BATCH = 1000;

def _scatterArc($r) {
    return "a" + _px($r) + "," + _px($r) + " 0 1,0 ";
}

def _drawScatterNative($bk, $ax, $a, $T) {
    $P = _pixelsArr($a, $T);
    $nf = nd_to_list(nd_sum(nd_astype($P["fin"], "f64"), null));
    if ($nf >= $_SCATTER_BATCH) {
        $bk.path(bp_scatter_path($P["px"], $P["py"], $P["fin"], $a["size"]), $a["color"], null, null);
        return null;
    }
    $cx = nd_to_list(nd_compress($P["fin"], $P["px"]));
    $cy = nd_to_list(nd_compress($P["fin"], $P["py"]));
    $i = 0;
    while ($i < len($cx)) { $bk.circle($cx[$i], $cy[$i], $a["size"], $a["color"]); $i = $i + 1; }
    return null;
}

def _drawScatter($bk, $ax, $a, $T) {
    if ($a["native"] == true) { return _drawScatterNative($bk, $ax, $a, $T); }
    $xs = $a["x"];
    $ys = $a["y"];
    if ($T["xs"] != "linear") { $xs = _project($xs, $T["xs"], $T["xlt"]); }
    if ($T["ys"] != "linear") { $ys = _project($ys, $T["ys"], $T["ylt"]); }
    $xa = $T["xa"]; $xb = $T["xb"];
    $ya = $T["ya"]; $yb = $T["yb"];
    $r = $a["size"];
    $c = $a["color"];
    $n = len($xs);
    $nf = 0;
    $i = 0;
    while ($i < $n) { if (isfinite($xs[$i]) && isfinite($ys[$i])) { $nf = $nf + 1; } $i = $i + 1; }
    if ($nf >= $_SCATTER_BATCH) {
        // The same string bp_scatter_path builds, from the same arithmetic.
        $arc = _scatterArc($r);
        $d2 = _px($r + $r);
        $md2 = _px(0 - ($r + $r));
        $parts = [];
        $i = 0;
        while ($i < $n) {
            $xv = $xs[$i];
            $yv = $ys[$i];
            if (isfinite($xv) && isfinite($yv)) {
                $pxv = $xv * $xa + $xb;
                $pyv = $yv * $ya + $yb;
                push($parts, "M" + _px($pxv - $r) + "," + _px($pyv) + $arc + $d2 + ",0" + $arc + $md2 + ",0");
            }
            $i = $i + 1;
        }
        $bk.path(join($parts, ""), $c, null, null);
        return null;
    }
    $i = 0;
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

// The baseline a bar or an area is drawn from. Zero is right on a linear
// axis; on a LOG axis zero does not exist, so the baseline is the bottom of
// the view -- which is what a log bar chart means anyway.
def _baseline($T, $which) {
    if ($which == "y") {
        if ($T["ys"] == "log") { return $T["ylim"][0]; }
        return 0;
    }
    if ($T["xs"] == "log") { return $T["xlim"][0]; }
    return 0;
}

// Can this value be drawn on that axis at all?
//
// A bar, a histogram bar or a stem of height zero on a LOG axis is not an
// error and not a silent loss: the bar is absent, and an absent bar reads as
// zero, which is exactly what it is. This is deliberately different from a
// line, where a missing segment is invisible and misleading, and where
// _project raises instead.
def _drawableOn($T, $which, $v) {
    if (!isfinite($v)) { return false; }
    if ($which == "y") { if ($T["ys"] == "log" && $v <= 0) { return false; } }
    else { if ($T["xs"] == "log" && $v <= 0) { return false; } }
    return true;
}

def _drawBar($bk, $ax, $a, $T) {
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
    $half = $gap * $a["width"] / 2;
    $i = 0;
    while ($i < $n) {
        $xv = $xs[$i];
        $yv = $ys[$i];
        $valueOk = true;
        if ($horiz) { $valueOk = _drawableOn($T, "x", $yv) && _drawableOn($T, "y", $xv); }
        else        { $valueOk = _drawableOn($T, "y", $yv) && _drawableOn($T, "x", $xv); }
        if ($valueOk) {
            if ($horiz) {
                $yA = _TY($T, $xv - $half);
                $yB = _TY($T, $xv + $half);
                $xZ = _TX($T, _baseline($T, "x"));
                $xV = _TX($T, $yv);
                $bk.rect(min($xZ, $xV), min($yA, $yB), abs($xV - $xZ), abs($yB - $yA), $c, null, null);
            } else {
                $xA = _TX($T, $xv - $half);
                $xB = _TX($T, $xv + $half);
                $yZ = _TY($T, _baseline($T, "y"));
                $yV = _TY($T, $yv);
                $bk.rect(min($xA, $xB), min($yZ, $yV), abs($xB - $xA), abs($yV - $yZ), $c, null, null);
            }
        }
        $i = $i + 1;
    }
    return null;
}

// ── B2 artists ───────────────────────────────────────────────────────────

// Adjacent rectangles from the bin edges. Unlike bar(), the width comes from
// the edges themselves, so unequal bins draw correctly.
def _drawHist($bk, $ax, $a, $T) {
    $e = $a["edges"];
    $h = $a["heights"];
    $c = $a["color"];
    $n = len($h);
    $i = 0;
    while ($i < $n) {
        if ($a["horiz"]) {
            if (_drawableOn($T, "x", $h[$i]) && _drawableOn($T, "y", $e[$i]) &&
                _drawableOn($T, "y", $e[$i + 1])) {
                $yA = _TY($T, $e[$i]);
                $yB = _TY($T, $e[$i + 1]);
                $xZ = _TX($T, _baseline($T, "x"));
                $xV = _TX($T, $h[$i]);
                if (abs($xV - $xZ) > 0) {
                    $bk.rect(min($xZ, $xV), min($yA, $yB), abs($xV - $xZ), abs($yB - $yA), $c, $_STYLE["bg"], 0.5);
                }
            }
        } else {
            if (_drawableOn($T, "y", $h[$i]) && _drawableOn($T, "x", $e[$i]) &&
                _drawableOn($T, "x", $e[$i + 1])) {
                $xA = _TX($T, $e[$i]);
                $xB = _TX($T, $e[$i + 1]);
                $yZ = _TY($T, _baseline($T, "y"));
                $yV = _TY($T, $h[$i]);
                if (abs($yV - $yZ) > 0) {
                    $bk.rect(min($xA, $xB), min($yZ, $yV), abs($xB - $xA), abs($yV - $yZ), $c, $_STYLE["bg"], 0.5);
                }
            }
        }
        $i = $i + 1;
    }
    return null;
}

def _drawBox($bk, $ax, $a, $T) {
    $c = $a["color"];
    $i = 0;
    while ($i < len($a["stats"])) {
        $s = $a["stats"][$i];
        $pos = $i + 1;
        $xA = _TX($T, $pos - $a["width"] / 2);
        $xB = _TX($T, $pos + $a["width"] / 2);
        $xM = _TX($T, $pos);
        $yQ1 = _TY($T, $s["q1"]);
        $yQ3 = _TY($T, $s["q3"]);
        $yMed = _TY($T, $s["med"]);
        $yLo = _TY($T, $s["wlo"]);
        $yHi = _TY($T, $s["whi"]);
        // Whiskers, then caps, then the box over them.
        $bk.line($xM, $yLo, $xM, min($yQ1, $yQ3), $_STYLE["fg"], 1);
        $bk.line($xM, $yHi, $xM, max($yQ1, $yQ3), $_STYLE["fg"], 1);
        $capA = _TX($T, $pos - $a["width"] / 4);
        $capB = _TX($T, $pos + $a["width"] / 4);
        $bk.line($capA, $yLo, $capB, $yLo, $_STYLE["fg"], 1);
        $bk.line($capA, $yHi, $capB, $yHi, $_STYLE["fg"], 1);
        $bk.rect(min($xA, $xB), min($yQ1, $yQ3), abs($xB - $xA), abs($yQ3 - $yQ1), $c, $_STYLE["fg"], 1);
        $bk.line(min($xA, $xB), $yMed, max($xA, $xB), $yMed, $_STYLE["fg"], 1.6);
        $j = 0;
        while ($j < len($s["out"])) {
            $bk.circle($xM, _TY($T, $s["out"][$j]), 2, $_STYLE["fg"]);
            $j = $j + 1;
        }
        $i = $i + 1;
    }
    return null;
}

def _drawViolin($bk, $ax, $a, $T) {
    $c = $a["color"];
    $i = 0;
    while ($i < len($a["shapes"])) {
        $sh = $a["shapes"][$i];
        $pos = $i + 1;
        if ($sh["flat"] != null) {
            // One repeated value: a line at that value, not a division by a
            // zero bandwidth.
            $y = _TY($T, $sh["flat"]);
            $bk.line(_TX($T, $pos - $a["width"] / 2), $y, _TX($T, $pos + $a["width"] / 2), $y, $c, 2);
            $i = $i + 1;
            continue;
        }
        $ys = $sh["y"];
        $dmax = max($ys);
        if ($dmax <= 0) { $i = $i + 1; continue; }
        $scale = $a["width"] / 2 / $dmax;
        $py = _pys($T, $sh["x"]);
        // Up the right side, back down the left, giving one closed polygon.
        $pts = [];
        $j = 0;
        while ($j < len($ys)) {
            push($pts, _TX($T, $pos + $ys[$j] * $scale));
            push($pts, $py[$j]);
            $j = $j + 1;
        }
        $j = len($ys) - 1;
        while ($j >= 0) {
            push($pts, _TX($T, $pos - $ys[$j] * $scale));
            push($pts, $py[$j]);
            $j = $j - 1;
        }
        $bk.polygon($pts, $c, $_STYLE["fg"], 0.8, 0.75);
        $i = $i + 1;
    }
    return null;
}

def _drawErrorbar($bk, $ax, $a, $T) {
    $px = _pxs($T, $a["x"]);
    $py = _pys($T, $a["y"]);
    $c = $a["color"];
    $w = $a["width"];
    $cap = $a["cap"];
    $xs = $a["x"];
    $ys = $a["y"];
    if ($a["line"]) {
        $run = [];
        $i = 0;
        while ($i < len($px)) {
            if (isfinite($px[$i]) && isfinite($py[$i])) { push($run, $px[$i]); push($run, $py[$i]); }
            $i = $i + 1;
        }
        if (len($run) >= 4) { $bk.polyline($run, $c, $w, null); }
    }
    $i = 0;
    $n = len($px);
    while ($i < $n) {
        if (isfinite($px[$i]) && isfinite($py[$i])) {
            if ($a["yerr"] != null) {
                $lo = _TY($T, $ys[$i] - $a["yerr"][0][$i]);
                $hi = _TY($T, $ys[$i] + $a["yerr"][1][$i]);
                $bk.line($px[$i], $lo, $px[$i], $hi, $c, $w);
                $bk.line($px[$i] - $cap, $lo, $px[$i] + $cap, $lo, $c, $w);
                $bk.line($px[$i] - $cap, $hi, $px[$i] + $cap, $hi, $c, $w);
            }
            if ($a["xerr"] != null) {
                $lo = _TX($T, $xs[$i] - $a["xerr"][0][$i]);
                $hi = _TX($T, $xs[$i] + $a["xerr"][1][$i]);
                $bk.line($lo, $py[$i], $hi, $py[$i], $c, $w);
                $bk.line($lo, $py[$i] - $cap, $lo, $py[$i] + $cap, $c, $w);
                $bk.line($hi, $py[$i] - $cap, $hi, $py[$i] + $cap, $c, $w);
            }
            if ($a["size"] > 0) { $bk.circle($px[$i], $py[$i], $a["size"], $c); }
        }
        $i = $i + 1;
    }
    return null;
}

// NaN breaks the band into separate polygons, for the same reason it breaks a
// line: a polygon with a NaN vertex renders as nothing at all.
def _drawBand($bk, $ax, $a, $T) {
    $px = _pxs($T, $a["x"]);
    $p1 = _pys($T, $a["y1"]);
    $p2 = _pys($T, $a["y2"]);
    $top = [];
    $bot = [];
    $i = 0;
    $n = len($px);
    while ($i <= $n) {
        $ok = false;
        if ($i < $n) { $ok = isfinite($px[$i]) && isfinite($p1[$i]) && isfinite($p2[$i]); }
        if ($ok) {
            push($top, $px[$i]); push($top, $p1[$i]);
            push($bot, $px[$i]); push($bot, $p2[$i]);
        } else {
            if (len($top) >= 4) {
                $pts = [];
                extend($pts, $top);
                $j = len($bot) - 2;
                while ($j >= 0) { push($pts, $bot[$j]); push($pts, $bot[$j + 1]); $j = $j - 2; }
                $edge = null;
                if ($a["edge"]) { $edge = $a["color"]; }
                $bk.polygon($pts, $a["color"], $edge, 1, $a["opacity"]);
            }
            $top = [];
            $bot = [];
        }
        $i = $i + 1;
    }
    return null;
}

// "pre": the value changes at the sample, so the step rises BEFORE the point.
// "post": it holds until the next sample. "mid": it changes halfway between.
// Drawn with the wrong convention a step chart is off by one sample and still
// looks plausible, which is why the names match matplotlib's exactly.
def _drawStep($bk, $ax, $a, $T) {
    $px = _pxs($T, $a["x"]);
    $py = _pys($T, $a["y"]);
    $where = $a["where"];
    $run = [];
    $i = 0;
    $n = len($px);
    while ($i < $n) {
        if (!isfinite($px[$i]) || !isfinite($py[$i])) {
            if (len($run) >= 4) { $bk.polyline($run, $a["color"], $a["width"], $a["dash"]); }
            $run = [];
            $i = $i + 1;
            continue;
        }
        if (len($run) == 0) { push($run, $px[$i]); push($run, $py[$i]); }
        else {
            $lx = $run[len($run) - 2];
            $ly = $run[len($run) - 1];
            if ($where == "pre") {
                push($run, $lx); push($run, $py[$i]);
                push($run, $px[$i]); push($run, $py[$i]);
            } else {
                if ($where == "post") {
                    push($run, $px[$i]); push($run, $ly);
                    push($run, $px[$i]); push($run, $py[$i]);
                } else {
                    $mx = ($lx + $px[$i]) / 2;
                    push($run, $mx); push($run, $ly);
                    push($run, $mx); push($run, $py[$i]);
                    push($run, $px[$i]); push($run, $py[$i]);
                }
            }
        }
        $i = $i + 1;
    }
    if (len($run) >= 4) { $bk.polyline($run, $a["color"], $a["width"], $a["dash"]); }
    return null;
}

def _drawStem($bk, $ax, $a, $T) {
    $px = _pxs($T, $a["x"]);
    $py = _pys($T, $a["y"]);
    $base = $a["bottom"];
    if (!_drawableOn($T, "y", $base)) { $base = _baseline($T, "y"); }
    $yb = _TY($T, $base);
    $c = $a["color"];
    // The baseline is drawn, so the sign of each value is readable.
    $bk.line($ax.left, $yb, $ax.left + $ax.w, $yb, $_STYLE["axis"], 1);
    $i = 0;
    $n = len($px);
    while ($i < $n) {
        if (isfinite($px[$i]) && isfinite($py[$i])) {
            $bk.line($px[$i], $yb, $px[$i], $py[$i], $c, $a["width"]);
            if ($a["size"] > 0) { $bk.circle($px[$i], $py[$i], $a["size"], $c); }
        }
        $i = $i + 1;
    }
    return null;
}

// A pie slice is an SVG arc: move to the centre, line to the start of the arc,
// sweep, close. `large` must be set past 180 degrees or the renderer takes the
// short way round and draws the complement of the slice.
// Angles grow counter-clockwise ON SCREEN (y = cy - r sin a), as matplotlib's
// pie does. In SVG's y-down space that is the NEGATIVE-angle direction, so the
// sweep flag is 0. It was 1 until B6e, and every slice was drawn as the mirror
// arc about its chord -- in the SVG as much as in the PNG.
def _arcPath($cx, $cy, $r, $a0, $a1) {
    $x0 = $cx + $r * cos($a0);
    $y0 = $cy - $r * sin($a0);
    $x1 = $cx + $r * cos($a1);
    $y1 = $cy - $r * sin($a1);
    $large = 0;
    if (abs($a1 - $a0) > PI) { $large = 1; }
    return "M " + _px($cx) + " " + _px($cy) +
           " L " + _px($x0) + " " + _px($y0) +
           " A " + _px($r) + " " + _px($r) + " 0 " + _fmt($large, 0) + " 0 " +
           _px($x1) + " " + _px($y1) + " Z";
}

def _drawPie($bk, $ax, $a, $T) {
    $vs = $a["values"];
    $tot = $a["total"];
    // A pie must be round, so the radius is the SMALLER of the two pixel
    // extents. Scaling x and y independently gives an ellipse, which
    // misrepresents every slice.
    $cx = _TX($T, 0);
    $cy = _TY($T, 0);
    $rx = abs(_TX($T, 1) - $cx);
    $ry = abs(_TY($T, 1) - $cy);
    $r = min($rx, $ry) * $a["radius"];
    $ang = radians($a["start"]);
    $i = 0;
    while ($i < len($vs)) {
        $frac = $vs[$i] / $tot;
        $next = $ang + $frac * 2 * PI;
        if ($frac >= 0.999999999) {
            // One slice holding everything: an arc from a point back to the
            // same point draws nothing, so it is a circle.
            $bk.circle($cx, $cy, $r, $a["colors"][$i]);
        } else {
            if ($frac > 0) { $bk.path(_arcPath($cx, $cy, $r, $ang, $next), $a["colors"][$i], $_STYLE["bg"], 1); }
        }
        if ($frac > 0.02) {
            $mid = ($ang + $next) / 2;
            $lr = $r * 0.68;
            $lx = $cx + $lr * cos($mid);
            $ly = $cy - $lr * sin($mid);
            $txt = "";
            if ($a["labels"] != null && $i < len($a["labels"])) { $txt = str($a["labels"][$i]); }
            if ($a["percent"]) {
                $p = _fmt($frac * 100, 1) + "%";
                if (len($txt) > 0) { $txt = $txt + " " + $p; } else { $txt = $p; }
            }
            if (len($txt) > 0) { $bk.text($lx, $ly + 4, $txt, 11, _contrastOn($a["colors"][$i]), "middle", null); }
        }
        $ang = $next;
        $i = $i + 1;
    }
    return null;
}

// Free-standing text, and the arrow an annotation may carry. The arrowhead is
// a triangle built from the direction of travel -- there is no marker element
// to define, which keeps the backend's method set at eleven.
def _drawText($bk, $t, $T) {
    $x = _TX($T, $t["x"]);
    $y = _TY($T, $t["y"]);
    if (!isfinite($x) || !isfinite($y)) { return null; }
    if ($t["arrow"] != null) {
        $tx = _TX($T, $t["arrow"][0]);
        $ty = _TY($T, $t["arrow"][1]);
        if (isfinite($tx) && isfinite($ty)) {
            $dx = $tx - $x;
            $dy = $ty - $y;
            $L = sqrt($dx * $dx + $dy * $dy);
            if ($L > 1) {
                $ux = $dx / $L;
                $uy = $dy / $L;
                // Stop the shaft short of the head so the two do not overlap
                // into a blunt wedge.
                $hl = 9;
                $bx = $tx - $ux * $hl;
                $by = $ty - $uy * $hl;
                $bk.line($x + $ux * 4, $y + $uy * 4, $bx, $by, $t["color"], 1);
                $bk.polygon([$tx, $ty,
                             $bx - $uy * 3.5, $by + $ux * 3.5,
                             $bx + $uy * 3.5, $by - $ux * 3.5],
                            $t["color"], null, null, null);
            }
        }
    }
    $rot = $t["rotate"];
    if ($rot == 0) { $rot = null; }
    $bk.text($x, $y, $t["s"], $t["size"], $t["color"], $t["anchor"], $rot);
    return null;
}

def _drawLegend($bk, $ax, $x1, $y0) {
    $entries = [];
    $i = 0;
    while ($i < len($ax.artists)) {
        $a = $ax.artists[$i];
        // A pie has one colour per slice and no single colour of its own, so
        // it is not a legend entry; its labels are drawn on the slices.
        if ($a["label"] != null && $a["color"] != null) {
            // Area-like artists get a filled swatch, line-like a stroke: a
            // filled band represented by a thin line in the key is unreadable.
            $k = $a["kind"];
            $solid = $k == "bar" || $k == "hist" || $k == "band" || $k == "box" || $k == "violin";
            push($entries, [str($a["label"]), $a["color"], $solid]);
        }
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
        $w = $bk.textWidth($entries[$i][0], $fs);
        if ($w > $wMax) { $wMax = $w; }
        $i = $i + 1;
    }
    $boxW = $pad * 2 + $swatch + 6 + $wMax;
    $boxH = $pad * 2 + $rowH * len($entries);
    $bx = $x1 - $boxW - 8;
    $by = $y0 + 8;
    $bk.rect($bx, $by, $boxW, $boxH, $_STYLE["bg"], $_STYLE["axis"], 1);
    $i = 0;
    while ($i < len($entries)) {
        $ly = $by + $pad + $rowH * $i + $rowH / 2;
        if ($entries[$i][2]) {
            $bk.rect($bx + $pad, $ly - 5, $swatch, 10, $entries[$i][1], null, null);
        } else {
            $bk.line($bx + $pad, $ly, $bx + $pad + $swatch, $ly, $entries[$i][1], 3);
        }
        $bk.text($bx + $pad + $swatch + 6, $ly + 4, $entries[$i][0], $fs, $_STYLE["fg"], "start", null);
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

def hist($v, $opts)                 { return gca().hist($v, $opts); }
def boxplot($data, $opts)           { return gca().boxplot($data, $opts); }
def violin($data, $opts)            { return gca().violin($data, $opts); }
def errorbar($x, $y, $opts)         { return gca().errorbar($x, $y, $opts); }
def fill_between($x, $y1, $y2, $o)  { return gca().fill_between($x, $y1, $y2, $o); }
def step($x, $y, $opts)             { return gca().step($x, $y, $opts); }
def stem($x, $y, $opts)             { return gca().stem($x, $y, $opts); }
def pie($v, $opts)                  { return gca().pie($v, $opts); }
def plot_frame($df, $opts)          { return gca().plot_frame($df, $opts); }

def text($x, $y, $s, $opts)         { return gca().addText($x, $y, $s, $opts); }
def annotate($s, $x, $y, $opts)     { return gca().annotate($s, $x, $y, $opts); }

// Scales. linthresh applies to "symlog" only and defaults to 1.
def xscale($scale, $linthresh) { return gca().setScale("x", $scale, $linthresh); }
def yscale($scale, $linthresh) { return gca().setScale("y", $scale, $linthresh); }
// x values are epoch milliseconds, UTC — which is how arctic stores a
// datetime column, so a column plots without conversion.
def xdate($on)  { return gca().setDateAxis($on); }
def axis($on)   { if ($on == "off") { return gca().setFrame(false); } return gca().setFrame(true); }

def xlabel($s)      { return gca().setXLabel($s); }
def ylabel($s)      { return gca().setYLabel($s); }
def title($s)       { return gca().setTitle($s); }
def grid($on)       { return gca().setGrid($on); }
def legend($on)     { return gca().setLegend($on); }
def xlim($lo, $hi)  { return gca().setXLim($lo, $hi); }
def ylim($lo, $hi)  { return gca().setYLim($lo, $hi); }

// ── B3: layout and 2-D ───────────────────────────────────────────────────
def subplots($rows, $cols) { clf(); return gcf().subplots($rows, $cols); }
def subplot($rows, $cols, $index) {
    $ax = gcf().subplot($rows, $cols, $index);
    gcf().setCurrent($ax);
    return $ax;
}
def sca($ax)          { return gcf().setCurrent($ax); }
def twinx($ax)        { if ($ax == null) { $ax = gca(); } return gcf().twinx($ax); }
def twiny($ax)        { if ($ax == null) { $ax = gca(); } return gcf().twiny($ax); }
def sharex($a, $b)    { return gcf().shareX($a, $b); }
def sharey($a, $b)    { return gcf().shareY($a, $b); }
def tight_layout($on) { return gcf().tight_layout($on); }

def imshow($z, $opts)              { return gca().imshow($z, $opts); }
def heatmap($z, $opts)             { return gca().heatmap($z, $opts); }
def pcolormesh($x, $y, $z, $opts)  { return gca().pcolormesh($x, $y, $z, $opts); }
def contour($z, $opts)             { return gca().contour($z, $opts); }
def colorbar($opts)                { return gca().colorbar($opts); }
def invert_yaxis($on)              { return gca().invertY($on); }

// ── Style sheets ─────────────────────────────────────────────────────────
//
// A style is PROCESS-WIDE, like the current figure, and carries the same
// caveat (BP15): a sua handler must not set one, because handlers run on
// their own threads and would race. Set it once at start-up, or not at all.
//
// Three, each with a reason to exist rather than a mood:
//   default  tab10 on white -- the most-recognised categorical palette.
//   dark     for a dark dashboard or slide; the cycle is lightened, because
//            tab10 on near-black loses the blue and the purple entirely.
//   print    greyscale-safe, ordered by LIGHTNESS rather than hue, so the
//            series stay distinguishable in a photocopy and under the common
//            colour-vision deficiencies. This is the same argument that makes
//            viridis the default colormap and keeps jet out of the tree.
def style($name) {
    if ($name == null) { $name = "default"; }
    if ($name == "default") {
        $_STYLE["fg"] = "#333333"; $_STYLE["grid"] = "#dddddd";
        $_STYLE["minor"] = "#f0f0f0"; $_STYLE["axis"] = "#888888";
        $_STYLE["bg"] = "#ffffff"; $_STYLE["font"] = "sans-serif";
        $_STYLE["cycle"] = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"];
        return $name;
    }
    if ($name == "dark") {
        $_STYLE["fg"] = "#e6e6e6"; $_STYLE["grid"] = "#3a3f4b";
        $_STYLE["minor"] = "#2c313a"; $_STYLE["axis"] = "#8a94a6";
        $_STYLE["bg"] = "#1e222a"; $_STYLE["font"] = "sans-serif";
        $_STYLE["cycle"] = ["#61afef", "#e5c07b", "#98c379", "#e06c75", "#c678dd",
                            "#d19a66", "#f5a0c8", "#abb2bf", "#dcdcaa", "#56b6c2"];
        return $name;
    }
    if ($name == "print") {
        $_STYLE["fg"] = "#000000"; $_STYLE["grid"] = "#d9d9d9";
        $_STYLE["minor"] = "#efefef"; $_STYLE["axis"] = "#000000";
        $_STYLE["bg"] = "#ffffff"; $_STYLE["font"] = "sans-serif";
        $_STYLE["cycle"] = ["#000000", "#666666", "#b3b3b3", "#333333", "#8c8c8c",
                            "#4d4d4d", "#999999", "#1a1a1a", "#cccccc", "#737373"];
        return $name;
    }
    throw "bplot.style: unknown style \"" + str($name) + "\" -- use \"default\", \"dark\" or \"print\"";
}

def to_svg()        { return gcf().to_svg(); }
def to_png($dpi)    { return gcf().to_png($dpi); }
def savefig($path, $opts) { $p = gcf().savefig($path, $opts); clf(); return $p; }

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
    print("bplot — data visualisation for Bantu (SVG and PNG)");
    print("");
    print("  Three lines to a chart:");
    print("    include \"bplot\" as plt;");
    print("    plt.plot([1, 2, 3], [2, 4, 9]);");
    print("    plt.savefig(\"chart.svg\");");
    print("");
    print("  Charts        plot(x, y, opts)    scatter(x, y, opts)");
    print("                bar(x, y, opts)     barh(x, y, opts)");
    print("                hist(values, opts)  boxplot(data, opts)");
    print("                violin(data, opts)  errorbar(x, y, opts)");
    print("                fill_between(x, y1, y2, opts)");
    print("                step(x, y, opts)    stem(x, y, opts)   pie(values, opts)");
    print("  Tables        plot_frame(df, {kind, x, y, title})  kind: line bar barh");
    print("                scatter hist box step;  with arctic, $df.plot(opts)");
    print("  Scales        xscale(s, linthresh)   yscale(s, linthresh)");
    print("                s is \"linear\", \"log\" or \"symlog\"");
    print("                xdate(on)   x values are epoch ms, UTC");
    print("  Text          text(x, y, s, opts)    annotate(s, x, y, {to: [x, y]})");
    print("  2-D           imshow(z, opts)     heatmap(z, opts)");
    print("                pcolormesh(x, y, z, opts)   contour(z, opts)");
    print("                colorbar(opts)      cmaps: viridis plasma coolwarm gray");
    print("  Layout        subplots(rows, cols) -> a list of axes");
    print("                subplot(rows, cols, index)   twinx(ax)   twiny(ax)");
    print("                sharex(a, b)   sharey(a, b)   tight_layout(on)");
    print("  Style         style(\"default\" | \"dark\" | \"print\")");
    print("  Decoration    title(s)   xlabel(s)   ylabel(s)   axis(\"off\")");
    print("                grid(on)   legend(on)  xlim(lo, hi)   ylim(lo, hi)");
    print("  Output        savefig(path)   savefig(\"x.png\", {\"dpi\": 150})");
    print("                to_svg()   to_png(dpi)   show(path)   clf()");
    print("  Objects       figure(w, h) -> $fig;  $fig.addAxes() -> $ax");
    print("                every plt.* call above is $ax.<the same thing>");
    print("");
    print("  opts is a dict: {\"color\": \"#1f77b4\", \"width\": 2, \"label\": \"series\"}");
    print("    color   a named colour, \"#rgb\", \"#rrggbb\", or [r, g, b]");
    print("    label   makes the series appear in legend()");
    print("");
    print("  x and y accept a Bantu list, a numba ndarray, or an arctic column or");
    print("  Series. A null is a gap; a datetime column makes a date axis.");
    print("");
    print("  Serving charts built from untrusted data? SVG is an executable");
    print("  document format. bplot escapes everything it emits, but serve it");
    print("  from a separate origin or under a strict CSP. See docs/bplot.md.");
    return null;
}

def version() { return "1.1.0"; }
