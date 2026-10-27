// ============================================================================
// lugha/encoder.b — transformer encoder + multi-label head
//
// Pre-norm blocks, 4-head self-attention, GELU feed-forward, mean pooling,
// sigmoid head. One sequence at a time: a text is sliced to its real token
// count, so there is NO padding inside the model and therefore no attention
// mask anywhere. A mask is the thing people get wrong (zeroing after the
// softmax instead of before it leaves the weights summing to less than one),
// and not having one is better than having a correct one.
//
// Exposes:
//   config($m)                      -> {L, H, D, F, T, V, C, dh}
//   forward($m, $ids, $n, $cache)   -> {"logits", "scores", ...activations}
//   layernorm / gelu / softmaxRows / sigmoid   (shared with train.b)
//   headCols($W, $h, $dh) / headRows($W, $h, $dh)
//   PARAM_NAMES($L)                 -> ordered tensor names for .btw
// ============================================================================

include "numba" as np;

const $EPS = 0.00001;
const $GELU_C = 0.7978845608028654;      // sqrt(2/pi)

// ── config, read from the model's own meta tensor ───────────────────────────
def config($m) {
    $meta = np.to_list($m["meta"]);
    $D = floor($meta[2]);
    $H = floor($meta[1]);
    return { "L": floor($meta[0]), "H": $H, "D": $D, "F": floor($meta[3]),
             "T": floor($meta[4]), "V": floor($meta[5]), "C": floor($meta[6]),
             "dh": floor($D / $H) };
}

def PARAM_NAMES($L) {
    $names = ["meta", "tok_embed", "pos_embed"];
    $i = 0;
    while ($i < $L) {
        $p = "blk" + str($i) + ".";
        extend($names, [$p + "ln1.g", $p + "ln1.b",
                        $p + "wq", $p + "wk", $p + "wv", $p + "wo",
                        $p + "ln2.g", $p + "ln2.b",
                        $p + "ff1", $p + "ff1b", $p + "ff2", $p + "ff2b"]);
        $i = $i + 1;
    }
    extend($names, ["ln_f.g", "ln_f.b", "head.w", "head.b"]);
    return $names;
}

// ── head views ──────────────────────────────────────────────────────────────
// Zero-copy strided views: slicing costs nothing at any size, which is what
// makes the per-head loop below as cheap as a concatenated one.
def headCols($W, $h, $dh) {
    return np.slice($W, [null, [$h * $dh, ($h + 1) * $dh, null]]);
}

def headRows($W, $h, $dh) {
    return np.slice($W, [[$h * $dh, ($h + 1) * $dh, null], null]);
}

// ── primitives ──────────────────────────────────────────────────────────────

// LayerNorm over the last axis. numba's var is Welford, so this survives a
// large activation mean, which the textbook E[x^2]-E[x]^2 does not.
def layernorm($X, $g, $b) {
    $mu   = np.mean($X, 1, true);
    $var  = np.var($X, 1, true, 0);
    $rstd = np.reciprocal(np.sqrt(np.add($var, $EPS, null), null), null);
    $xhat = np.multiply(np.subtract($X, $mu, null), $rstd, null);
    $Y    = np.add(np.multiply($xhat, $g, null), $b, null);
    return { "Y": $Y, "xhat": $xhat, "rstd": $rstd };
}

// GELU, tanh approximation: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))
def gelu($X) {
    $x3 = np.multiply(np.multiply($X, $X, null), $X, null);
    $inner = np.multiply(np.add($X, np.multiply($x3, 0.044715, null), null), $GELU_C, null);
    $t = np.tanh($inner, null);
    return { "Y": np.multiply(np.multiply($X, 0.5, null), np.add($t, 1.0, null), null),
             "t": $t, "inner": $inner };
}

// Row-wise softmax, max-subtracted. Without the subtraction a logit of 800
// overflows to inf and the row becomes NaN.
def softmaxRows($S) {
    $m = np.max($S, 1, true);
    $E = np.exp(np.subtract($S, $m, null), null);
    $s = np.sum($E, 1, true);
    return np.divide($E, $s, null);
}

def sigmoid($X) {
    return np.reciprocal(np.add(np.exp(np.negative($X, null), null), 1.0, null), null);
}

// ── forward ─────────────────────────────────────────────────────────────────
// $ids is a Bantu list of token ids; $n is how many of them are real. When
// $cache is true every intermediate the backward pass needs is returned, which
// is the only reason train.b does not have its own copy of this function.
def forward($m, $ids, $n, $cache) {
    $c = config($m);
    $dh = $c["dh"];
    $scale = 1.0 / sqrt($dh);

    // Only the real tokens. No padding enters the model.
    $real = [];
    $i = 0;
    while ($i < $n) { append($real, $ids[$i]); $i = $i + 1; }

    $tok = np.take($m["tok_embed"], $real, 0);                       // [n, D]
    $pos = np.slice($m["pos_embed"], [[0, $n, null], null]);         // [n, D]
    $X = np.add($tok, $pos, null);

    $blocks = [];
    $l = 0;
    while ($l < $c["L"]) {
        $p = "blk" + str($l) + ".";
        $bc = {};

        // ── attention, pre-norm ──────────────────────────────────────────
        $ln1 = layernorm($X, $m[$p + "ln1.g"], $m[$p + "ln1.b"]);
        $A1 = $ln1["Y"];

        // Multi-head attention without concatenation. Because the output
        // projection is block-structured over the head axis, concat-then-
        // project equals the sum of per-head projections -- algebraically
        // identical, and numba has no concatenate. See
        // docs/lugha-architecture.md 3.1.
        $attn = np.zeros([$n, $c["D"]], "f64");
        $heads = [];
        $h = 0;
        while ($h < $c["H"]) {
            $Q = np.matmul($A1, headCols($m[$p + "wq"], $h, $dh));
            $K = np.matmul($A1, headCols($m[$p + "wk"], $h, $dh));
            $V = np.matmul($A1, headCols($m[$p + "wv"], $h, $dh));
            $S = np.multiply(np.matmul($Q, np.transpose($K, null)), $scale, null);
            $P = softmaxRows($S);
            $ctx = np.matmul($P, $V);
            np.add($attn, np.matmul($ctx, headRows($m[$p + "wo"], $h, $dh)), $attn);
            if ($cache) {
                append($heads, { "Q": $Q, "K": $K, "V": $V, "P": $P, "ctx": $ctx });
            }
            $h = $h + 1;
        }
        $X = np.add($X, $attn, null);

        // ── feed-forward, pre-norm ───────────────────────────────────────
        $ln2 = layernorm($X, $m[$p + "ln2.g"], $m[$p + "ln2.b"]);
        $A2 = $ln2["Y"];
        $Hpre = np.add(np.matmul($A2, $m[$p + "ff1"]), $m[$p + "ff1b"], null);
        $G = gelu($Hpre);
        $ff = np.add(np.matmul($G["Y"], $m[$p + "ff2"]), $m[$p + "ff2b"], null);
        $X = np.add($X, $ff, null);

        if ($cache) {
            $bc["ln1"] = $ln1;
            $bc["A1"] = $A1;
            $bc["heads"] = $heads;
            $bc["ln2"] = $ln2;
            $bc["A2"] = $A2;
            $bc["Hpre"] = $Hpre;
            $bc["G"] = $G;
            $bc["Xmid"] = np.subtract($X, $ff, null);   // after attention, before ff
            append($blocks, $bc);
        }
        $l = $l + 1;
    }

    $lnf = layernorm($X, $m["ln_f.g"], $m["ln_f.b"]);
    $Z = $lnf["Y"];                                     // [n, D]
    $pooled = np.reshape(np.mean($Z, 0, false), [1, $c["D"]]);
    $logits = np.add(np.matmul($pooled, $m["head.w"]), $m["head.b"], null);   // [1, C]
    $scores = sigmoid($logits);

    $out = { "logits": $logits, "scores": $scores, "n": $n };
    if ($cache) {
        $out["ids"] = $real;
        $out["blocks"] = $blocks;
        $out["lnf"] = $lnf;
        $out["Xfinal"] = $X;
        $out["pooled"] = $pooled;
        $out["cfg"] = $c;
    }
    return $out;
}

// Convenience: the 7 scores as a plain Bantu list.
def scoreList($m, $ids, $n) {
    return np.to_list(np.reshape(forward($m, $ids, $n, false)["scores"], [-1]));
}
