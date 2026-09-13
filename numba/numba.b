// ════════════════════════════════════════════════════════════════════════
//  numba — Numerical Bantu.
//
//  NumPy-class arrays for Bantu: n-dimensional, strided, zero-copy views,
//  broadcasting, ~55 element-wise functions, reductions over any axis,
//  sorting, indexing and linear algebra.
//
//      include "numba" as np;
//
//      $a = np.array([[1, 2], [3, 4]]);
//      print($a.sum());
//
//  THIS FILE IS THE PUBLIC API. The nd_* builtins underneath are the
//  plumbing, and they stay documented and callable for anyone who wants
//  them -- but they cannot have optional arguments, and this can. Bantu
//  binds a missing argument to null, so `np.arange(0, 10)` works here while
//  the raw builtin needs `nd_arange(0, 10, null)`. That is the whole reason
//  this layer exists, and why every builtin gets a wrapper even when the
//  wrapper looks like it does nothing.
//
//  Source:  numba/numba.b
//  Docs:    docs/numba.md
//  Tests:   tests/numba_*_test.b
// ════════════════════════════════════════════════════════════════════════

// ── constants ────────────────────────────────────────────────────────────
// Re-exported from the language so np.PI and PI are the same number rather
// than two copies that could drift. INF and NAN used to be built here by
// dividing a one-element array, which meant importing numba allocated two
// arrays before it did anything; they are language constants now.
$PI  = PI;
$E   = E;
$TAU = TAU;
$INF = INF;
$NAN = NAN;

def available() { return has_native("ndarray"); }

// ── creation ─────────────────────────────────────────────────────────────
def array($data, $dtype)        { return nd($data, $dtype); }
def zeros($shape, $dtype)       { return nd_zeros($shape, $dtype); }
def ones($shape, $dtype)        { return nd_ones($shape, $dtype); }
def full($shape, $value, $dt)   { return nd_full($shape, $value, $dt); }
def empty($shape, $dtype)       { return nd_empty($shape, $dtype); }
def zeros_like($a, $dtype)      { return nd_zeros_like($a, $dtype); }
def ones_like($a, $dtype)       { return nd_ones_like($a, $dtype); }
def full_like($a, $v, $dtype)   { return nd_full_like($a, $v, $dtype); }
def eye($n, $m)                 { return nd_eye($n, $m); }
def identity($n)                { return nd_identity($n); }

// step defaults to 1, which is the reason this wrapper exists.
def arange($start, $stop, $step) { return nd_arange($start, $stop, $step); }
def linspace($start, $stop, $num, $endpoint) {
    $n = $num;
    if ($n == null) { $n = 50; }
    return nd_linspace($start, $stop, $n, $endpoint);
}

// ── randomness ───────────────────────────────────────────────────────────
// numba carries its own stream, so an unrelated random() elsewhere in the
// program cannot change your matrix. Note it is PER THREAD: inside a sua
// handler each request seeds its own.
def seed($n)                          { return nd_seed($n); }
def random($shape, $lo, $hi)          {
    $a = $lo; $b = $hi;
    if ($a == null) { $a = 0; }
    if ($b == null) { $b = 1; }
    return nd_random_uniform($shape, $a, $b);
}
def randn($shape, $mu, $sigma) {
    $m = $mu; $s = $sigma;
    if ($m == null) { $m = 0; }
    if ($s == null) { $s = 1; }
    return nd_random_normal($shape, $m, $s, null);
}
def randint($shape, $lo, $hi)         { return nd_random_int($shape, $lo, $hi); }

// ── introspection ────────────────────────────────────────────────────────
def shape($a)          { return nd_shape($a); }
def ndim($a)           { return nd_ndim($a); }
def size($a)           { return nd_size($a); }
def dtype($a)          { return nd_dtype($a); }
def strides($a)        { return nd_strides($a); }
def itemsize($a)       { return nd_itemsize($a); }
def nbytes($a)         { return nd_nbytes($a); }
def is_view($a)        { return nd_is_view($a); }
def is_contiguous($a)  { return nd_is_contiguous($a); }
def writable($a)       { return nd_writable($a); }
def shares_memory($a, $b) { return nd_shares_memory($a, $b); }
def live_bytes()       { return nd_live_bytes(); }
def max_bytes($n)      { return nd_max_bytes($n); }

// ── shape operations ─────────────────────────────────────────────────────
def reshape($a, $shape)   { return nd_reshape($a, $shape); }
def transpose($a, $axes)  { return nd_transpose($a, $axes); }
def T($a)                 { return nd_T($a); }
def ravel($a)             { return nd_ravel($a); }
def flatten($a)           { return nd_flatten($a); }
def swapaxes($a, $i, $j)  { return nd_swapaxes($a, $i, $j); }
def moveaxis($a, $i, $j)  { return nd_moveaxis($a, $i, $j); }
def expand_dims($a, $ax)  { return nd_expand_dims($a, $ax); }
def squeeze($a, $ax)      { return nd_squeeze($a, $ax); }
def flip($a, $ax)         { return nd_flip($a, $ax); }
def broadcast_to($a, $sh) { return nd_broadcast_to($a, $sh); }
def broadcast_shapes($a, $b) { return nd_broadcast_shapes($a, $b); }
def slice($a, $specs)     { return nd_slice($a, $specs); }
def copy($a)              { return nd_copy($a); }
def astype($a, $dtype)    { return nd_astype($a, $dtype); }
def ascontiguous($a)      { return nd_ascontiguous($a); }
def to_list($a)           { return nd_to_list($a); }
def get($a, $idx)         { return nd_get($a, $idx); }
def set($a, $idx, $v)     { return nd_set($a, $idx, $v); }

// ── element-wise ─────────────────────────────────────────────────────────
def add($a, $b, $out)      { return nd_add($a, $b, $out); }
def subtract($a, $b, $out) { return nd_subtract($a, $b, $out); }
def multiply($a, $b, $out) { return nd_multiply($a, $b, $out); }
def divide($a, $b, $out)   { return nd_divide($a, $b, $out); }
def floor_divide($a, $b, $out) { return nd_floor_divide($a, $b, $out); }
def mod($a, $b, $out)      { return nd_mod($a, $b, $out); }
def power($a, $b, $out)    { return nd_power($a, $b, $out); }
def negative($a, $out)     { return nd_negative($a, $out); }
def abs($a, $out)          { return nd_abs($a, $out); }
def sign($a, $out)         { return nd_sign($a, $out); }
def square($a, $out)       { return nd_square($a, $out); }
def sqrt($a, $out)         { return nd_sqrt($a, $out); }
def cbrt($a, $out)         { return nd_cbrt($a, $out); }
def exp($a, $out)          { return nd_exp($a, $out); }
def expm1($a, $out)        { return nd_expm1($a, $out); }
def log($a, $out)          { return nd_log($a, $out); }
def log1p($a, $out)        { return nd_log1p($a, $out); }
def log2($a, $out)         { return nd_log2($a, $out); }
def log10($a, $out)        { return nd_log10($a, $out); }
def sin($a, $out)          { return nd_sin($a, $out); }
def cos($a, $out)          { return nd_cos($a, $out); }
def tan($a, $out)          { return nd_tan($a, $out); }
def asin($a, $out)         { return nd_asin($a, $out); }
def acos($a, $out)         { return nd_acos($a, $out); }
def atan($a, $out)         { return nd_atan($a, $out); }
def atan2($a, $b, $out)    { return nd_atan2($a, $b, $out); }
def sinh($a, $out)         { return nd_sinh($a, $out); }
def cosh($a, $out)         { return nd_cosh($a, $out); }
def tanh($a, $out)         { return nd_tanh($a, $out); }
def asinh($a, $out)        { return nd_asinh($a, $out); }
def acosh($a, $out)        { return nd_acosh($a, $out); }
def atanh($a, $out)        { return nd_atanh($a, $out); }
def hypot($a, $b, $out)    { return nd_hypot($a, $b, $out); }
def copysign($a, $b, $out) { return nd_copysign($a, $b, $out); }
def floor($a, $out)        { return nd_floor($a, $out); }
def ceil($a, $out)         { return nd_ceil($a, $out); }
def trunc($a, $out)        { return nd_trunc($a, $out); }
def round($a, $out)        { return nd_round($a, $out); }
def rint($a, $out)         { return nd_rint($a, $out); }
def reciprocal($a, $out)   { return nd_reciprocal($a, $out); }
def degrees($a, $out)      { return nd_degrees($a, $out); }
def radians($a, $out)      { return nd_radians($a, $out); }
def minimum($a, $b, $out)  { return nd_minimum($a, $b, $out); }
def maximum($a, $b, $out)  { return nd_maximum($a, $b, $out); }
def isnan($a, $out)        { return nd_isnan($a, $out); }
def isinf($a, $out)        { return nd_isinf($a, $out); }
def isfinite($a, $out)     { return nd_isfinite($a, $out); }
def equal($a, $b, $out)         { return nd_equal($a, $b, $out); }
def not_equal($a, $b, $out)     { return nd_not_equal($a, $b, $out); }
def less($a, $b, $out)          { return nd_less($a, $b, $out); }
def less_equal($a, $b, $out)    { return nd_less_equal($a, $b, $out); }
def greater($a, $b, $out)       { return nd_greater($a, $b, $out); }
def greater_equal($a, $b, $out) { return nd_greater_equal($a, $b, $out); }
def logical_and($a, $b, $out)   { return nd_logical_and($a, $b, $out); }
def logical_or($a, $b, $out)    { return nd_logical_or($a, $b, $out); }
def logical_xor($a, $b, $out)   { return nd_logical_xor($a, $b, $out); }
def logical_not($a, $out)       { return nd_logical_not($a, $out); }
def where($cond, $x, $y)   { return nd_where($cond, $x, $y); }
def clip($a, $lo, $hi, $out) { return nd_clip($a, $lo, $hi, $out); }

def isclose($a, $b, $rtol, $atol, $equal_nan) {
    return nd_isclose($a, $b, $rtol, $atol, $equal_nan);
}
def allclose($a, $b, $rtol, $atol, $equal_nan) {
    return nd_allclose($a, $b, $rtol, $atol, $equal_nan);
}
def array_equal($a, $b) { return nd_array_equal($a, $b); }

// ── reductions ───────────────────────────────────────────────────────────
def sum($a, $axis, $keepdims)   { return nd_sum($a, $axis, $keepdims); }
def prod($a, $axis, $keepdims)  { return nd_prod($a, $axis, $keepdims); }
def mean($a, $axis, $keepdims)  { return nd_mean($a, $axis, $keepdims); }
def var($a, $axis, $keepdims, $ddof)  { return nd_var($a, $axis, $keepdims, $ddof); }
def std($a, $axis, $keepdims, $ddof)  { return nd_std($a, $axis, $keepdims, $ddof); }
def min($a, $axis, $keepdims)   { return nd_min($a, $axis, $keepdims); }
def max($a, $axis, $keepdims)   { return nd_max($a, $axis, $keepdims); }
def ptp($a, $axis, $keepdims)   { return nd_ptp($a, $axis, $keepdims); }
def argmin($a, $axis, $keepdims) { return nd_argmin($a, $axis, $keepdims); }
def argmax($a, $axis, $keepdims) { return nd_argmax($a, $axis, $keepdims); }
// NOT `any`: that is a reserved type keyword in Bantu, so `def any(...)` does
// not parse. `$a.any()` and `nd_any($a, ...)` both work -- only the façade
// function needs the different name.
def anyof($a, $axis, $keepdims) { return nd_any($a, $axis, $keepdims); }
def all($a, $axis, $keepdims)   { return nd_all($a, $axis, $keepdims); }
def count_nonzero($a, $axis, $keepdims) { return nd_count_nonzero($a, $axis, $keepdims); }
def median($a, $axis, $keepdims) { return nd_median($a, $axis, $keepdims); }
def quantile($a, $q, $axis, $keepdims) { return nd_quantile($a, $q, $axis, $keepdims); }
def nansum($a, $axis, $keepdims)  { return nd_nansum($a, $axis, $keepdims); }
def nanmean($a, $axis, $keepdims) { return nd_nanmean($a, $axis, $keepdims); }
def nanmin($a, $axis, $keepdims)  { return nd_nanmin($a, $axis, $keepdims); }
def nanmax($a, $axis, $keepdims)  { return nd_nanmax($a, $axis, $keepdims); }

// ── scans, sorting, indexing ─────────────────────────────────────────────
def cumsum($a, $axis)   { return nd_cumsum($a, $axis); }
def cumprod($a, $axis)  { return nd_cumprod($a, $axis); }
def cummax($a, $axis)   { return nd_cummax($a, $axis); }
def cummin($a, $axis)   { return nd_cummin($a, $axis); }
def diff($a, $axis)     { return nd_diff($a, $axis); }
def sort($a, $axis)     { return nd_sort($a, $axis); }
def argsort($a, $axis)  { return nd_argsort($a, $axis); }
def searchsorted($sorted, $values, $side) { return nd_searchsorted($sorted, $values, $side); }
def unique($a)          { return nd_unique($a); }
def bincount($a, $minlength) { return nd_bincount($a, $minlength); }
def histogram($a, $bins, $lo, $hi) { return nd_histogram($a, $bins, $lo, $hi); }
def take($a, $indices, $axis) { return nd_take($a, $indices, $axis); }
def put($a, $indices, $values) { return nd_put($a, $indices, $values); }
def compress($mask, $a) { return nd_compress($mask, $a); }
def nonzero($a)         { return nd_nonzero($a); }

// ── linear algebra ───────────────────────────────────────────────────────
def matmul($a, $b)   { return nd_matmul($a, $b); }
def dot($a, $b)      { return nd_dot($a, $b); }
def outer($a, $b)    { return nd_outer($a, $b); }
def trace($a)        { return nd_trace($a); }
def solve($a, $b)    { return nd_solve($a, $b); }
def inv($a)          { return nd_inv($a); }
def det($a)          { return nd_det($a); }
def slogdet($a)      { return nd_slogdet($a); }
def cholesky($a)     { return nd_cholesky($a); }
def qr($a)           { return nd_qr($a); }
def lstsq($a, $b)    { return nd_lstsq($a, $b); }
def eigh($a)         { return nd_eigh($a); }
def svd($a)          { return nd_svd($a); }
def matrix_rank($a)  { return nd_matrix_rank($a); }
def cond($a)         { return nd_cond($a); }
def pinv($a)         { return nd_pinv($a); }
def norm($a, $ord)   { return nd_norm($a, $ord); }

// ════════════════════════════════════════════════════════════════════════
//  COMPOSED HELPERS
//
//  These are the reason the façade is Bantu rather than a table of aliases:
//  each is built out of the atoms above, so none of them needs a new kernel
//  and all of them are readable by anyone who wants to check the maths.
// ════════════════════════════════════════════════════════════════════════

// Least-squares polynomial fit of the given degree, highest power first --
// the same coefficient order NumPy's polyfit uses, so polyval agrees.
def polyfit($x, $y, $deg) {
    if ($deg == null) { $deg = 1; }
    $n = nd_size($x);
    // The Vandermonde matrix: column j holds x^(deg-j).
    $V = nd_zeros([$n, $deg + 1], "f64");
    $j = 0;
    while ($j <= $deg) {
        $col = nd_power($x, ($deg - $j) * 1.0, null);
        $i = 0;
        while ($i < $n) {
            nd_set($V, [$i, $j], nd_get($col, [$i]));
            $i = $i + 1;
        }
        $j = $j + 1;
    }
    // Through lstsq, which goes via QR -- the normal equations would square
    // the condition number, and a Vandermonde matrix is already ill-conditioned.
    return nd_lstsq($V, nd_astype($y, "f64"));
}

// Evaluate a polynomial by Horner's rule: fewer operations than powers, and
// numerically better behaved.
def polyval($coef, $x) {
    $k = nd_size($coef);
    $acc = nd_full_like(nd_astype($x, "f64"), 0.0, "f64");
    $i = 0;
    while ($i < $k) {
        $acc = nd_add(nd_multiply($acc, $x, null), nd_get($coef, [$i]), null);
        $i = $i + 1;
    }
    return $acc;
}

// Linear interpolation of $y over $x, evaluated at $xnew. $x must be
// ascending. Values outside the range clamp to the ends, as NumPy's interp
// does -- extrapolating silently is how people get nonsense far from the data.
def interp($xnew, $x, $y) {
    $idx = nd_searchsorted($x, $xnew, null);
    $n   = nd_size($x);
    $out = nd_zeros([nd_size($xnew)], "f64");
    $i = 0;
    while ($i < nd_size($xnew)) {
        $j = nd_get($idx, [$i]);
        if ($j <= 0) {
            nd_set($out, [$i], nd_get($y, [0]));
        } else {
            if ($j >= $n) {
                nd_set($out, [$i], nd_get($y, [$n - 1]));
            } else {
                $x0 = nd_get($x, [$j - 1]); $x1 = nd_get($x, [$j]);
                $y0 = nd_get($y, [$j - 1]); $y1 = nd_get($y, [$j]);
                $t = 0;
                if ($x1 != $x0) { $t = (nd_get($xnew, [$i]) - $x0) / ($x1 - $x0); }
                nd_set($out, [$i], $y0 + ($y1 - $y0) * $t);
            }
        }
        $i = $i + 1;
    }
    return $out;
}

// Central differences in the interior, one-sided at the ends -- second-order
// accurate inside, which a plain diff() is not.
def gradient($y, $h) {
    $step = $h;
    if ($step == null) { $step = 1.0; }
    $n = nd_size($y);
    $g = nd_zeros([$n], "f64");
    if ($n < 2) { return $g; }
    nd_set($g, [0], (nd_get($y, [1]) - nd_get($y, [0])) / $step);
    nd_set($g, [$n - 1], (nd_get($y, [$n - 1]) - nd_get($y, [$n - 2])) / $step);
    $i = 1;
    while ($i < $n - 1) {
        nd_set($g, [$i], (nd_get($y, [$i + 1]) - nd_get($y, [$i - 1])) / (2.0 * $step));
        $i = $i + 1;
    }
    return $g;
}

// Covariance of two equal-length vectors, with the sample (ddof=1) estimator.
def cov($x, $y) {
    $n = nd_size($x);
    if ($n < 2) { return 0.0; }
    $mx = nd_get(nd_mean($x, null, null), []);
    $my = nd_get(nd_mean($y, null, null), []);
    $dx = nd_subtract(nd_astype($x, "f64"), $mx, null);
    $dy = nd_subtract(nd_astype($y, "f64"), $my, null);
    return nd_get(nd_sum(nd_multiply($dx, $dy, null), null, null), []) / ($n - 1.0);
}

// Pearson correlation. Returns NaN when either side is constant, because the
// coefficient is genuinely undefined there rather than 0.
def corrcoef($x, $y) {
    $sx = nd_get(nd_std($x, null, null, 1), []);
    $sy = nd_get(nd_std($y, null, null, 1), []);
    if ($sx == 0 || $sy == 0) { return $NAN; }
    return cov($x, $y) / ($sx * $sy);
}

// A coordinate grid from two 1-d vectors: X varies along the columns and Y
// along the rows, matching NumPy's default "xy" indexing.
def meshgrid($x, $y) {
    $nx = nd_size($x);
    $ny = nd_size($y);
    $X = nd_broadcast_to(nd_reshape($x, [1, $nx]), [$ny, $nx]);
    $Y = nd_broadcast_to(nd_reshape($y, [$ny, 1]), [$ny, $nx]);
    // Copied, because a broadcast view is read-only and a grid is something
    // people expect to be able to write to.
    return [nd_copy($X), nd_copy($Y)];
}

// Trailing moving average over a window of $w. The result is shorter than
// the input by $w-1, which is the honest length -- padding it would invent
// values.
def moving_average($a, $w) {
    $n = nd_size($a);
    if ($w == null || $w < 1) { $w = 1; }
    if ($w > $n) {
        throw "np.moving_average: the window (" + str($w) + ") is longer than the data (" + str($n) + ")";
    }
    $c = nd_cumsum(nd_astype($a, "f64"), null);
    $out = nd_zeros([$n - $w + 1], "f64");
    $i = 0;
    while ($i < $n - $w + 1) {
        $hi = nd_get($c, [$i + $w - 1]);
        $lo = 0.0;
        if ($i > 0) { $lo = nd_get($c, [$i - 1]); }
        nd_set($out, [$i], ($hi - $lo) / ($w * 1.0));
        $i = $i + 1;
    }
    return $out;
}

// Trapezoidal integration.
def trapz($y, $dx) {
    $h = $dx;
    if ($h == null) { $h = 1.0; }
    $n = nd_size($y);
    if ($n < 2) { return 0.0; }
    $total = nd_get(nd_sum($y, null, null), []);
    return ($total - 0.5 * (nd_get($y, [0]) + nd_get($y, [$n - 1]))) * $h;
}

// ════════════════════════════════════════════════════════════════════════
//  DISCOVERABILITY
// ════════════════════════════════════════════════════════════════════════

def help($topic) {
    if ($topic != null) {
        $t = str($topic);
        if ($t == "solve") {
            print("np.solve(a, b) — solve a x = b for x.");
            print("  a must be square and non-singular; b is a vector or a matrix.");
            print("  Raises on a singular matrix — use np.lstsq or np.pinv instead.");
            print("");
            print("  $A = np.array([[4.0, 1.0], [1.0, 3.0]]);");
            print("  $x = np.solve($A, np.array([1.0, 2.0]));");
            return null;
        }
        if ($t == "sum" || $t == "mean") {
            print("np.sum(a, axis, keepdims) / np.mean(a, axis, keepdims)");
            print("  axis: null for everything, a number, or a list of numbers.");
            print("  keepdims: true leaves the reduced axis as length 1, so the");
            print("  result broadcasts back against the input.");
            print("");
            print("  $centred = np.subtract($m, np.mean($m, 1, true));");
            return null;
        }
        if ($t == "reshape") {
            print("np.reshape(a, shape) — a zero-copy VIEW when the strides allow.");
            print("  One axis may be -1, meaning 'work it out'.");
            print("  Writing through the view changes the original.");
            return null;
        }
        print("np.help(\"" + $t + "\"): no entry. np.help() lists the groups.");
        return null;
    }
    print("numba — Numerical Bantu.  np.help(\"solve\") for one entry.");
    print("");
    print("  create      array zeros ones full empty eye identity arange linspace");
    print("              zeros_like ones_like full_like  random randn randint seed");
    print("  inspect     shape ndim size dtype strides nbytes is_view is_contiguous");
    print("  shape ops   reshape transpose T ravel flatten swapaxes moveaxis");
    print("              expand_dims squeeze flip broadcast_to slice copy astype");
    print("  maths       add subtract multiply divide power mod abs sqrt exp log");
    print("              sin cos tan asin acos atan atan2 hypot floor ceil round");
    print("              minimum maximum clip where sign square reciprocal");
    print("  compare     equal not_equal less greater less_equal greater_equal");
    print("              logical_and logical_or logical_not isclose allclose");
    print("  reduce      sum prod mean var std min max ptp argmin argmax anyof all");
    print("              median quantile count_nonzero  nansum nanmean nanmin nanmax");
    print("  scan/sort   cumsum cumprod cummax cummin diff sort argsort unique");
    print("              searchsorted bincount histogram");
    print("  index       take put compress nonzero");
    print("  linalg      matmul dot outer trace solve inv det slogdet cholesky qr");
    print("              lstsq eigh svd pinv norm matrix_rank cond");
    print("  composed    polyfit polyval interp gradient cov corrcoef meshgrid");
    print("              moving_average trapz");
    print("");
    print("  Operators work directly: $a + $b, 2 * $a, -$a, $a > 0.5, $m[1][2] = 9");
    print("  np.anyof(), not np.any() -- `any` is a reserved word in Bantu.");
    print("  Methods chain:           $x.multiply($x).add($x).sum()");
    return null;
}

// Everything you want to know about an array in one line each.
def info($a) {
    print("shape      " + str(nd_shape($a)));
    print("dtype      " + nd_dtype($a));
    print("size       " + str(nd_size($a)) + " elements, " + str(nd_nbytes($a)) + " bytes");
    print("strides    " + str(nd_strides($a)) + "  (in elements)");
    print("view       " + str(nd_is_view($a)) + "   contiguous " + str(nd_is_contiguous($a))
          + "   writable " + str(nd_writable($a)));
    if (nd_size($a) > 0) {
        print("min/mean/max  " + str(nd_get(nd_min($a, null, null), [])) + " / "
              + str(nd_get(nd_mean($a, null, null), [])) + " / "
              + str(nd_get(nd_max($a, null, null), [])));
    }
    return null;
}
