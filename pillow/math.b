// ════════════════════════════════════════════════════════════════════════
//  pillow/math.b — ImageMath: arithmetic over whole images, and a small
//  expression evaluator for the grammar Pillow's examples use.
//
//  Pillow has two entry points. `lambda_eval(f, **kw)` calls a CALLABLE
//  with a dict of operands, and maps onto Bantu directly -- pass a Bantu
//  function, receive a dict, call the operators as functions, because
//  Bantu has no operator overloading. `unsafe_eval(expr, **kw)` compiles a
//  PYTHON expression, and cannot be mapped: there is no Python here.
//
//  So `eval($expr, $env)` ships a parser for exactly this grammar:
//
//      comparison  <  >  ==  !=  <=  >=
//      bitwise     |  ^  &  <<  >>  ~
//      arithmetic  +  -  *  /  //  %  **   (Python's precedence, ** right)
//      atoms       numbers, names, ( ), and the named functions
//                  abs min max convert float int equal notequal
//
//  An expression outside that grammar is a PARSE ERROR naming the offending
//  token, not a silent misread. docs/pillow.md states the grammar rather
//  than implying that arbitrary Bantu is evaluated.
//
//  ─── THREE OPERATOR SEMANTICS THAT ARE NOT THE HOST LANGUAGE'S ───────
//
//  From _imagingmath.c, and all three are deliberate:
//    * DIVISION BY ZERO YIELDS ZERO, for both int and float. Not an error,
//      not an infinity.
//    * integer division and % are C's: truncating, with % taking the sign
//      of the DIVIDEND (so -7 % 3 is -1, not Python's 2).
//    * pow in mode I is (int)(pow(x, y) + 0.5), clamped to int32 --
//      rounded, not truncated, and saturating rather than wrapping.
//
//  Operands are mode I (int32) or F (float32); 1 and L are converted up to
//  I, and two operands of different modes are both promoted to F. The
//  result is cropped to the smaller of the two in each axis.
//
//  docs/pillow-architecture.md §7.5.
// ════════════════════════════════════════════════════════════════════════

include "./mode.b" as mode;
include "./pxbuf.b" as pxbuf;
include "./image.b" as image;

$INT_MIN = 0 - 2147483648;
$INT_MAX = 2147483647;

// ─── Operands ───────────────────────────────────────────────────────────

// _operand($x, $selfMode, $size) -> {"m": "I"|"F", "px": (h,w,1)}.
//
// __fixup: an image in mode 1 or L is converted to I, I and F pass
// through, anything else is "unsupported mode". A CONSTANT becomes an
// image of the operand's own mode -- I when the other side is 1/L/I, F
// otherwise -- which is why 1/2 over an I image is integer division.
def _fixup($x, $selfMode, $size) {
    if (type($x) == "dict" && $x.m != null && $x.px != null) { return $x; }
    if (type($x) == "instance") {
        if ($x.mode == "1" || $x.mode == "L") {
            $c = $x.convert("I", null, null);
            return {"m": "I", "px": $c.px};
        }
        if ($x.mode == "I") { return {"m": "I", "px": $x.px}; }
        if ($x.mode == "F") { return {"m": "F", "px": $x.px}; }
        throw "unsupported mode: " + $x.mode;
    }
    if (type($x) == "number") {
        $m = "F";
        if ($selfMode == "1" || $selfMode == "L" || $selfMode == "I") { $m = "I"; }
        if ($m == "I") { $v = floor($x); } else { $v = $x; }
        return {"m": $m, "px": nd_full([$size[1], $size[0], 1], $v,
                                       mode.dtype($m)), "const": true};
    }
    throw "ImageMath: an operand must be an image or a number, got " + type($x);
}

def _toF($o) {
    if ($o.m == "F") { return $o; }
    return {"m": "F", "px": pxbuf.to_f32(nd_astype($o.px, "f64"))};
}
def _toI($o) {
    if ($o.m == "I") { return $o; }
    // Pillow's F -> I conversion truncates toward zero.
    return {"m": "I", "px": nd_astype(
        nd_multiply(nd_sign($o.px), nd_floor(nd_abs($o.px))), "i64")};
}

// _align($a, $b) -> the two operands in the same mode, cropped to the same
// size, with a plain number turned into a constant image of the OTHER
// side's mode -- which is what makes `a / 2` integer division over an I
// image and float division over an F one.
def _align($a, $b) {
    if (type($a) == "number" && type($b) == "number") {
        throw "ImageMath: an operation needs at least one image";
    }
    $ctx = $a;
    if (type($a) == "number") { $ctx = $b; }
    $cs = nd_shape($ctx.px);
    $size = [$cs[1], $cs[0]];
    $x = _fixup($a, $ctx.m, $size);
    $y = _fixup($b, $ctx.m, $size);
    if ($x.m != $y.m) { $x = _toF($x); $y = _toF($y); }
    $sa = nd_shape($x.px);
    $sb = nd_shape($y.px);
    $h = min($sa[0], $sb[0]);
    $w = min($sa[1], $sb[1]);
    if ($sa[0] != $h || $sa[1] != $w) { $x = {"m": $x.m, "px": pxbuf.crop($x.px, 0, 0, $w, $h)}; }
    if ($sb[0] != $h || $sb[1] != $w) { $y = {"m": $y.m, "px": pxbuf.crop($y.px, 0, 0, $w, $h)}; }
    return [$x, $y];
}

def _wrapRes($m, $px) {
    if ($m == "F") { return {"m": "F", "px": pxbuf.to_f32($px)}; }
    return {"m": "I", "px": nd_astype($px, "i64")};
}

// ─── The operators ──────────────────────────────────────────────────────

def add($a, $b) { if (type($a) == "number" && type($b) == "number") { return $a + $b; } $p = _align($a, $b); return _wrapRes($p[0].m, nd_add($p[0].px, $p[1].px)); }
def sub($a, $b) { if (type($a) == "number" && type($b) == "number") { return $a - $b; } $p = _align($a, $b); return _wrapRes($p[0].m, nd_subtract($p[0].px, $p[1].px)); }
def mul($a, $b) { if (type($a) == "number" && type($b) == "number") { return $a * $b; } $p = _align($a, $b); return _wrapRes($p[0].m, nd_multiply($p[0].px, $p[1].px)); }

// div: zero where the divisor is zero, and TRUNCATING for mode I.
def div($a, $b) {
    if (type($a) == "number" && type($b) == "number") {
        if ($b == 0) { return 0; }
        return $a / $b;
    }
    $p = _align($a, $b);
    $x = $p[0].px;
    $y = $p[1].px;
    $safe = nd_where(nd_equal($y, 0), 1, $y);
    if ($p[0].m == "I") {
        $q = nd_divide($x, $safe);
        $t = nd_multiply(nd_sign($q), nd_floor(nd_abs($q)));
    } else {
        $t = nd_divide($x, $safe);
    }
    return _wrapRes($p[0].m, nd_where(nd_equal($y, 0), 0, $t));
}

// mod: C's %, which takes the sign of the DIVIDEND, and zero on a zero
// divisor. nd_mod follows the divisor's sign, so the sign is reapplied.
def mod($a, $b) {
    if (type($a) == "number" && type($b) == "number") {
        if ($b == 0) { return 0; }
        $r = abs($a) % abs($b);
        if ($a < 0) { return 0 - $r; }
        return $r;
    }
    $p = _align($a, $b);
    $x = $p[0].px;
    $y = $p[1].px;
    $safe = nd_where(nd_equal($y, 0), 1, $y);
    $r = nd_multiply(nd_sign($x), nd_mod(nd_abs($x), nd_abs($safe)));
    return _wrapRes($p[0].m, nd_where(nd_equal($y, 0), 0, $r));
}

// pow: in mode I, (int)(pow(x, y) + 0.5) saturated to int32.
def pow_($a, $b) {
    if (type($a) == "number" && type($b) == "number") { return pow($a, $b); }
    $p = _align($a, $b);
    $v = nd_power($p[0].px, $p[1].px);
    if ($p[0].m == "I") {
        $r = nd_clip(nd_floor(nd_add($v, 0.5)), $INT_MIN, $INT_MAX);
        return _wrapRes("I", $r);
    }
    return _wrapRes("F", $v);
}

def neg($a) {
    if (type($a) == "number") { return 0 - $a; }
    $o = _fixup($a, null, null);
    return _wrapRes($o.m, nd_negative($o.px));
}
def abs_($a) {
    if (type($a) == "number") { return abs($a); }
    $o = _fixup($a, null, null);
    return _wrapRes($o.m, nd_abs($o.px));
}

def _needI($o, $who) {
    if ($o.m != "I") { throw "bad operand type for '" + $who + "'"; }
    return $o;
}
def bit_and($a, $b) {
    if (type($a) == "number" && type($b) == "number") { return band($a, $b); }
    $p = _align($a, $b);
    _needI($p[0], "and");
    return _wrapRes("I", nd_bitwise_and($p[0].px, $p[1].px));
}
def bit_or($a, $b) {
    if (type($a) == "number" && type($b) == "number") { return bor($a, $b); }
    $p = _align($a, $b);
    _needI($p[0], "or");
    return _wrapRes("I", nd_bitwise_or($p[0].px, $p[1].px));
}
def bit_xor($a, $b) {
    if (type($a) == "number" && type($b) == "number") { return bxor($a, $b); }
    $p = _align($a, $b);
    _needI($p[0], "xor");
    return _wrapRes("I", nd_bitwise_xor($p[0].px, $p[1].px));
}
def bit_not($a) {
    if (type($a) == "number") { return 0 - $a - 1; }
    $o = _needI(_fixup($a, null, null), "invert");
    return _wrapRes("I", nd_bitwise_not($o.px));
}
def lshift($a, $b) {
    if (type($a) == "number" && type($b) == "number") { return $a * pow(2, $b); }
    $p = _align($a, $b);
    _needI($p[0], "lshift");
    return _wrapRes("I", nd_left_shift($p[0].px, $p[1].px));
}
def rshift($a, $b) {
    if (type($a) == "number" && type($b) == "number") { return floor($a / pow(2, $b)); }
    $p = _align($a, $b);
    _needI($p[0], "rshift");
    return _wrapRes("I", nd_right_shift($p[0].px, $p[1].px));
}

def _cmp($a, $b, $op, $forceI) {
    if (type($a) == "number" && type($b) == "number") {
        if ($op == "eq") { if ($a == $b) { return 1; } return 0; }
        if ($op == "ne") { if ($a != $b) { return 1; } return 0; }
        if ($op == "lt") { if ($a < $b) { return 1; } return 0; }
        if ($op == "le") { if ($a <= $b) { return 1; } return 0; }
        if ($op == "gt") { if ($a > $b) { return 1; } return 0; }
        if ($a >= $b) { return 1; }
        return 0;
    }
    $p = _align($a, $b);
    $x = $p[0].px;
    $y = $p[1].px;
    if ($op == "eq") { $r = nd_equal($x, $y); }
    else if ($op == "ne") { $r = nd_not_equal($x, $y); }
    else if ($op == "lt") { $r = nd_less($x, $y); }
    else if ($op == "le") { $r = nd_less_equal($x, $y); }
    else if ($op == "gt") { $r = nd_greater($x, $y); }
    else { $r = nd_greater_equal($x, $y); }
    $m = $p[0].m;
    if ($forceI) { $m = "I"; }
    return _wrapRes($m, nd_where($r, 1, 0));
}

def eq($a, $b) { return _cmp($a, $b, "eq", false); }
def ne($a, $b) { return _cmp($a, $b, "ne", false); }
def lt($a, $b) { return _cmp($a, $b, "lt", false); }
def le($a, $b) { return _cmp($a, $b, "le", false); }
def gt($a, $b) { return _cmp($a, $b, "gt", false); }
def ge($a, $b) { return _cmp($a, $b, "ge", false); }

// equal / notequal are the named functions, and they force mode I.
def equal($a, $b)    { return _cmp($a, $b, "eq", true); }
def notequal($a, $b) { return _cmp($a, $b, "ne", true); }

def min_($a, $b) {
    if (type($a) == "number" && type($b) == "number") { return min($a, $b); }
    $p = _align($a, $b);
    return _wrapRes($p[0].m, nd_minimum($p[0].px, $p[1].px));
}
def max_($a, $b) {
    if (type($a) == "number" && type($b) == "number") { return max($a, $b); }
    $p = _align($a, $b);
    return _wrapRes($p[0].m, nd_maximum($p[0].px, $p[1].px));
}

def to_int($a) {
    if (type($a) == "number") { return floor($a); }
    return _toI(_fixup($a, null, null));
}
def to_float($a) {
    if (type($a) == "number") { return $a; }
    return _toF(_fixup($a, null, null));
}

// convert($operand, $mode) -> the operand through Image.convert, which is
// how ImageMath's convert() reaches modes outside I and F.
def convert($a, $m) {
    $o = _fixup($a, null, null);
    $im = image.wrap($o.m, $o.px, null, {}, null);
    $c = $im.convert($m, null, null);
    if ($m == "I" || $m == "F") { return {"m": $m, "px": $c.px}; }
    return {"m": $m, "px": $c.px, "image": $c};
}

// to_image($operand) -> the Image an expression's result stands for.
def to_image($o) {
    if (type($o) == "number") { return $o; }
    if (type($o) == "instance") { return $o; }
    if ($o.image != null) { return $o.image; }
    return image.wrap($o.m, $o.px, null, {}, null);
}

// ─── lambda_eval: Pillow's callable form ────────────────────────────────
//
// The callable receives a dict of operands, keyed by name, and calls the
// operators above. Pillow writes `lambda args: args["a"] + args["b"]`;
// here that is `def($a) { return mt.add($a.a, $a.b); }`.
def lambda_eval($fn, $env) {
    $args = {};
    if ($env != null) {
        for $k, $v in $env.items() {
            if (type($v) == "instance") { $args[$k] = _fixup($v, null, null); }
            else { $args[$k] = $v; }
        }
    }
    return to_image($fn($args));
}

// ─── eval: the expression evaluator ─────────────────────────────────────

// Longest first: "<<" and "<=" must be tried before "<", or `a << 1`
// tokenises as two comparisons and the error message blames the wrong thing.
$_PUNCT = ["**", "<<", ">>", "<=", ">=", "==", "!=", "//",
           "+", "-", "*", "/", "%", "&", "|", "^", "~", "(", ")", ",",
           "<", ">"];

def _isDigit($c) { return $c >= "0" && $c <= "9"; }
def _isNameStart($c) {
    return ($c >= "a" && $c <= "z") || ($c >= "A" && $c <= "Z") || $c == "_";
}
def _isNameChar($c) { return _isNameStart($c) || _isDigit($c); }

// _tokenise($s) -> a list of {"t": "num"|"name"|"op", "v": ...}
def _tokenise($s) {
    $out = [];
    $i = 0;
    $n = len($s);
    while ($i < $n) {
        $c = $s[$i];
        if ($c == " " || $c == "\t" || $c == "\n" || $c == "\r") { $i = $i + 1; continue; }
        if (_isDigit($c) || ($c == "." && $i + 1 < $n && _isDigit($s[$i + 1]))) {
            $j = $i;
            $seen = false;
            while ($j < $n && (_isDigit($s[$j]) || $s[$j] == ".")) {
                if ($s[$j] == ".") {
                    if ($seen) { break; }
                    $seen = true;
                }
                $j = $j + 1;
            }
            push($out, {"t": "num", "v": num(substr($s, $i, $j - $i))});
            $i = $j;
            continue;
        }
        if (_isNameStart($c)) {
            $j = $i;
            while ($j < $n && _isNameChar($s[$j])) { $j = $j + 1; }
            push($out, {"t": "name", "v": substr($s, $i, $j - $i)});
            $i = $j;
            continue;
        }
        $hit = null;
        $k = 0;
        while ($k < len($_PUNCT)) {
            $p = $_PUNCT[$k];
            if ($i + len($p) <= $n && substr($s, $i, len($p)) == $p) { $hit = $p; break; }
            $k = $k + 1;
        }
        if ($hit == null) {
            throw "ImageMath.eval: unexpected character '" + $c + "' at " + str($i) +
                  ". The grammar is + - * / // % ** & | ^ << >> ~, the comparisons, " +
                  "parentheses, numbers, names and abs min max convert float int " +
                  "equal notequal.";
        }
        push($out, {"t": "op", "v": $hit});
        $i = $i + len($hit);
    }
    return $out;
}

// The parser is precedence climbing over a cursor in a dict, because a
// Bantu function cannot return two values and a list argument is copied.
def _peek($st) {
    if ($st.i >= len($st.toks)) { return null; }
    return $st.toks[$st.i];
}
def _next($st) {
    $t = _peek($st);
    if ($t != null) { $st.i = $st.i + 1; }
    return $t;
}
def _expectOp($st, $v) {
    $t = _next($st);
    if ($t == null || $t.t != "op" || $t.v != $v) {
        throw "ImageMath.eval: expected '" + $v + "'";
    }
    return true;
}

def _isOp($t, $v) { return $t != null && $t.t == "op" && $t.v == $v; }

def _parseExpr($st)  { return _parseCmp($st); }

def _parseCmp($st) {
    $left = _parseBitOr($st);
    $t = _peek($st);
    while ($t != null && $t.t == "op" &&
           ($t.v == "<" || $t.v == ">" || $t.v == "<=" || $t.v == ">=" ||
            $t.v == "==" || $t.v == "!=")) {
        _next($st);
        $right = _parseBitOr($st);
        if ($t.v == "<") { $left = lt($left, $right); }
        else if ($t.v == ">") { $left = gt($left, $right); }
        else if ($t.v == "<=") { $left = le($left, $right); }
        else if ($t.v == ">=") { $left = ge($left, $right); }
        else if ($t.v == "==") { $left = eq($left, $right); }
        else { $left = ne($left, $right); }
        $t = _peek($st);
    }
    return $left;
}

def _parseBitOr($st) {
    $left = _parseBitXor($st);
    while (_isOp(_peek($st), "|")) {
        _next($st);
        $left = bit_or($left, _parseBitXor($st));
    }
    return $left;
}
def _parseBitXor($st) {
    $left = _parseBitAnd($st);
    while (_isOp(_peek($st), "^")) {
        _next($st);
        $left = bit_xor($left, _parseBitAnd($st));
    }
    return $left;
}
def _parseBitAnd($st) {
    $left = _parseShift($st);
    while (_isOp(_peek($st), "&")) {
        _next($st);
        $left = bit_and($left, _parseShift($st));
    }
    return $left;
}
def _parseShift($st) {
    $left = _parseArith($st);
    $t = _peek($st);
    while (_isOp($t, "<<") || _isOp($t, ">>")) {
        _next($st);
        $right = _parseArith($st);
        if ($t.v == "<<") { $left = lshift($left, $right); }
        else { $left = rshift($left, $right); }
        $t = _peek($st);
    }
    return $left;
}
def _parseArith($st) {
    $left = _parseTerm($st);
    $t = _peek($st);
    while (_isOp($t, "+") || _isOp($t, "-")) {
        _next($st);
        $right = _parseTerm($st);
        if ($t.v == "+") { $left = add($left, $right); }
        else { $left = sub($left, $right); }
        $t = _peek($st);
    }
    return $left;
}
def _parseTerm($st) {
    $left = _parseUnary($st);
    $t = _peek($st);
    while (_isOp($t, "*") || _isOp($t, "/") || _isOp($t, "%") || _isOp($t, "//")) {
        _next($st);
        $right = _parseUnary($st);
        if ($t.v == "*") { $left = mul($left, $right); }
        else if ($t.v == "%") { $left = mod($left, $right); }
        else { $left = div($left, $right); }
        $t = _peek($st);
    }
    return $left;
}
def _parseUnary($st) {
    $t = _peek($st);
    if (_isOp($t, "-")) { _next($st); return neg(_parseUnary($st)); }
    if (_isOp($t, "+")) { _next($st); return _parseUnary($st); }
    if (_isOp($t, "~")) { _next($st); return bit_not(_parseUnary($st)); }
    return _parsePower($st);
}
def _parsePower($st) {
    $base = _parseAtom($st);
    if (_isOp(_peek($st), "**")) {
        _next($st);
        // ** is right-associative and binds tighter than unary minus on its
        // left, which _parseUnary already arranges.
        return pow_($base, _parseUnary($st));
    }
    return $base;
}

def _parseAtom($st) {
    $t = _next($st);
    if ($t == null) { throw "ImageMath.eval: the expression ends early"; }
    if ($t.t == "num") { return $t.v; }
    if (_isOp($t, "(")) {
        $v = _parseExpr($st);
        _expectOp($st, ")");
        return $v;
    }
    if ($t.t == "name") {
        if (_isOp(_peek($st), "(")) {
            _next($st);
            $args = [];
            if (!_isOp(_peek($st), ")")) {
                push($args, _parseExpr($st));
                while (_isOp(_peek($st), ",")) {
                    _next($st);
                    push($args, _parseExpr($st));
                }
            }
            _expectOp($st, ")");
            return _call($st, $t.v, $args);
        }
        $v = $st.env[$t.v];
        if ($v == null) { throw "'" + $t.v + "' not allowed"; }
        return $v;
    }
    throw "ImageMath.eval: unexpected '" + str($t.v) + "'";
}

def _call($st, $name, $args) {
    if ($name == "abs") { return abs_($args[0]); }
    if ($name == "int") { return to_int($args[0]); }
    if ($name == "float") { return to_float($args[0]); }
    if ($name == "min") { return min_($args[0], $args[1]); }
    if ($name == "max") { return max_($args[0], $args[1]); }
    if ($name == "equal") { return equal($args[0], $args[1]); }
    if ($name == "notequal") { return notequal($args[0], $args[1]); }
    if ($name == "convert") {
        $m = $args[1];
        if (type($m) != "string") {
            throw "ImageMath.eval: convert's second argument is a mode name";
        }
        return convert($args[0], $m);
    }
    throw "'" + $name + "' not allowed";
}

// eval($expr, $env) -> the image (or number) the expression evaluates to.
//
// $env maps names to images or numbers. A name the expression uses and the
// environment does not define is an error, as Pillow's scan() makes it.
def eval_($expr, $env) {
    if (type($expr) != "string") { throw "ImageMath.eval: the expression is a string"; }
    $e = {};
    if ($env != null) {
        for $k, $v in $env.items() {
            if (type($v) == "instance") { $e[$k] = _fixup($v, null, null); }
            else { $e[$k] = $v; }
        }
    }
    // A mode name in a convert() call arrives as a bare name, so the mode
    // names are in scope as strings.
    $modes = ["1", "L", "I", "F", "RGB", "RGBA", "CMYK", "YCbCr", "LAB", "HSV", "P"];
    $i = 0;
    while ($i < len($modes)) {
        if ($e[$modes[$i]] == null) { $e[$modes[$i]] = $modes[$i]; }
        $i = $i + 1;
    }
    $st = {"toks": _tokenise($expr), "i": 0, "env": $e};
    if (len($st.toks) == 0) { throw "ImageMath.eval: the expression is empty"; }
    $v = _parseExpr($st);
    if ($st.i != len($st.toks)) {
        $t = $st.toks[$st.i];
        throw "ImageMath.eval: unexpected '" + str($t.v) + "' after a complete expression";
    }
    return to_image($v);
}

// unsafe_eval is Pillow's name for the same thing, kept so that ported
// code reads the same. There is no Python being compiled here, so it is
// not unsafe in Pillow's sense -- the grammar is the sandbox.
def unsafe_eval($expr, $env) { return eval_($expr, $env); }
