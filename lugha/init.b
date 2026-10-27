// ============================================================================
// lugha/init.b — model construction
//
// Shared by train.b (fresh weights) and the tests (tiny deterministic models).
// Separated from train.b so a test can build a 2-token, 4-dimension model and
// check the maths by hand without dragging in the optimiser.
//
// Exposes:
//   build($V, $D, $F, $L, $H, $T, $C, $scale) -> model dict
// ============================================================================

include "numba" as np;
include "./encoder.b" as enc;

// Weight init. A transformer at this depth is not fussy: scaled-normal for
// every matrix, zeros for every bias, ones/zeros for LayerNorm. The output
// projections are scaled by 1/sqrt(2L) -- the GPT-2 residual scaling -- so the
// residual stream does not grow with depth.
def build($V, $D, $F, $L, $H, $T, $C, $seed) {
    np.seed($seed);
    $m = {};
    $m["meta"] = np.array([$L, $H, $D, $F, $T, $V, $C], "f64");

    $m["tok_embed"] = np.randn([$V, $D], 0.0, 0.02);
    $m["pos_embed"] = np.randn([$T, $D], 0.0, 0.02);

    $resScale = 0.02 / sqrt(2 * $L);
    $i = 0;
    while ($i < $L) {
        $p = "blk" + str($i) + ".";
        $m[$p + "ln1.g"] = np.ones([$D], "f64");
        $m[$p + "ln1.b"] = np.zeros([$D], "f64");
        $m[$p + "wq"]    = np.randn([$D, $D], 0.0, 0.02);
        $m[$p + "wk"]    = np.randn([$D, $D], 0.0, 0.02);
        $m[$p + "wv"]    = np.randn([$D, $D], 0.0, 0.02);
        $m[$p + "wo"]    = np.randn([$D, $D], 0.0, $resScale);
        $m[$p + "ln2.g"] = np.ones([$D], "f64");
        $m[$p + "ln2.b"] = np.zeros([$D], "f64");
        $m[$p + "ff1"]   = np.randn([$D, $F], 0.0, 0.02);
        $m[$p + "ff1b"]  = np.zeros([$F], "f64");
        $m[$p + "ff2"]   = np.randn([$F, $D], 0.0, $resScale);
        $m[$p + "ff2b"]  = np.zeros([$D], "f64");
        $i = $i + 1;
    }
    $m["ln_f.g"] = np.ones([$D], "f64");
    $m["ln_f.b"] = np.zeros([$D], "f64");
    $m["head.w"] = np.randn([$D, $C], 0.0, 0.02);
    // A multi-label head starts pessimistic on purpose: sigmoid(-2) is about
    // 0.12, so an untrained model abstains rather than accusing everyone.
    $m["head.b"] = np.full([$C], -2.0, "f64");
    return $m;
}

def paramCount($m, $L) {
    $total = 0;
    each ($name in enc.PARAM_NAMES($L)) {
        if ($name == "meta") { continue; }
        $n = 1;
        each ($d in np.shape($m[$name])) { $n = $n * $d; }
        $total = $total + $n;
    }
    return $total;
}
