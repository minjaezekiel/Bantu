// ════════════════════════════════════════════════════════════════════════
//  pillow/jpegdct.b — the array side of JPEG: dequantise, inverse DCT,
//  upsample, colour transform. Everything after the entropy scan.
//
//  The reference is libjpeg-turbo, not Pillow: Pillow hands the file to
//  libjpeg and asks for scanlines, so matching Pillow here means matching
//  jidctint.c, jdsample.c and jdcolor.c bit for bit. All four stages are
//  fixed-point integer arithmetic with no freedom in them, which is why
//  bit-exactness is a reasonable target rather than an optimistic one.
//
//  ─── THE SHAPE THAT MAKES IT POSSIBLE ────────────────────────────────
//
//  jpeg_idct_islow is written as a loop over the 8 columns of one block,
//  then its 8 rows. Transcribed literally that is a loop over coefficients:
//  rule PW2 forbids it and it would cost minutes.
//
//  So the ARITHMETIC is transcribed exactly and only what each operation
//  applies to changes. The whole component's coefficients are one
//  (N, 8, 8) array -- N blocks, 8 rows, 8 columns, natural (de-zigzagged)
//  order, which is libjpeg's JBLOCK layout. Then
//
//      pass 1 reads row r of EVERY block as nd_slice(a, [null, [r,r+1], null])
//      -- an (N, 1, 8) view, free -- so it computes all 8 columns of all N
//      blocks at once, and the kernel's 12 multiplies and 32 adds become 12
//      nd_multiply and 32 nd_add calls.
//
//      pass 2 does the same with column slices.
//
//  The cost is ~70 array operations PER COMPONENT, whatever the image size.
//
//  Two consequences of the same arrangement:
//
//  * The C's two "all AC terms are zero" short-circuits are not reproduced
//    and need not be. They are exact algebraic identities -- a DC-only
//    column gives DESCALE(dc << 13, 11) == dc << 2, which is the shortcut's
//    dcval -- and a vectorised kernel cannot branch per column anyway.
//
//  * The C accumulates pass 1 into `int workspace[64]` and would OVERFLOW
//    AND WRAP on a crafted coefficient block. Our store is i64, so it
//    neither wraps nor saturates. On valid input the two agree exactly; on
//    hostile input ours stays arithmetic. Decision PW11.
//
//  DESCALE(x, n) is an arithmetic right shift with rounding, and an
//  arithmetic right shift IS a floor division -- including for negatives,
//  which is what nd_floor_divide does and what the C does.
//
//  docs/pillow-architecture.md §6.4.1 through §6.4.4.
// ════════════════════════════════════════════════════════════════════════

include "./pxbuf.b" as pxbuf;
include "./jpeg_tables.b" as jt;

// jidctint.c's constants. CONST_BITS 13, PASS1_BITS 2, so the pass-1
// descale is by 11 and the pass-2 descale by 13 + 2 + 3 = 18.
$C_0_298631336 = 2446;
$C_0_390180644 = 3196;
$C_0_541196100 = 4433;
$C_0_765366865 = 6270;
$C_0_899976223 = 7373;
$C_1_175875602 = 9633;
$C_1_501321110 = 12299;
$C_1_847759065 = 15137;
$C_1_961570560 = 16069;
$C_2_053119869 = 16819;
$C_2_562915447 = 20995;
$C_3_072711026 = 25172;

// Module state lives in a DICT, never in a bare module variable: a function
// assigning to `$_RANGE` would bind a function-LOCAL of that name and leave
// the module variable null, so the table would be rebuilt on every call and
// any OTHER function reading it would see null. Same arrangement as
// codecs/registry.b, and for the same reason.
$_C = {"range": null, "crr": null, "cbb": null, "crg": null, "cbg": null};

// _range_table() -> the range-limit table as an i64 array, built once.
def _range_table() {
    if ($_C.range == null) { $_C.range = nd(jt.IDCT_RANGE, "i64"); }
    return $_C.range;
}

// ─── DESCALE ────────────────────────────────────────────────────────────

// descale($a, $n) -> (a + 2^(n-1)) >> n, arithmetic. One add, one divide.
def descale($a, $n) {
    $half = pow(2, $n - 1);
    return nd_floor_divide(nd_add($a, $half), pow(2, $n));
}

// ─── Dequantise ─────────────────────────────────────────────────────────

// dequantise($coef, $qtab) -> coef * the quantisation table.
//
// $coef is (N, 64) in natural order, $qtab a 64-entry list in natural
// order. One nd_multiply against a (1, 64) broadcast: no loop at all.
def dequantise($coef, $qtab) {
    $s = nd_shape($coef);
    if (len($s) != 2 || $s[1] != 64) {
        throw "jpegdct.dequantise: coefficients must be (N, 64), got " + str($s);
    }
    if (len($qtab) != 64) {
        throw "jpegdct.dequantise: a quantisation table is 64 entries, got " +
              str(len($qtab));
    }
    return nd_multiply($coef, nd_reshape(nd($qtab, "i64"), [1, 64]));
}

// ─── The 8x8 inverse DCT ────────────────────────────────────────────────

// _row($a, $r) -> row r of every block, an (N, 1, 8) view. Free.
def _row($a, $r) { return nd_slice($a, [null, [$r, $r + 1], null]); }

// _col($a, $c) -> column c of every block, an (N, 8, 1) view. Free.
def _col($a, $c) { return nd_slice($a, [null, null, [$c, $c + 1]]); }

// idct8($coef) -> (N, 8, 8) samples, range-limited to 0..255.
//
// $coef is (N, 64), already dequantised, in natural order. This is
// jpeg_idct_islow, statement for statement, with every variable an array.
def idct8($coef) {
    $s = nd_shape($coef);
    $n = $s[0];
    $a = nd_reshape($coef, [$n, 8, 8]);
    $ws = nd_zeros([$n, 8, 8], "i64");

    // ── Pass 1: the columns. Reads rows of the input. ──
    //
    // Even part. The rotator is sqrt(2) * c(-6).
    $z2 = _row($a, 2);
    $z3 = _row($a, 6);
    $z1 = nd_multiply(nd_add($z2, $z3), $C_0_541196100);
    $tmp2 = nd_add($z1, nd_multiply($z3, 0 - $C_1_847759065));
    $tmp3 = nd_add($z1, nd_multiply($z2, $C_0_765366865));

    $z2 = _row($a, 0);
    $z3 = _row($a, 4);
    $tmp0 = nd_multiply(nd_add($z2, $z3), 8192);        // LEFT_SHIFT(.., 13)
    $tmp1 = nd_multiply(nd_subtract($z2, $z3), 8192);

    $tmp10 = nd_add($tmp0, $tmp3);
    $tmp13 = nd_subtract($tmp0, $tmp3);
    $tmp11 = nd_add($tmp1, $tmp2);
    $tmp12 = nd_subtract($tmp1, $tmp2);

    // Odd part per figure 8. i0..i3 are y7, y5, y3, y1.
    $t0 = _row($a, 7);
    $t1 = _row($a, 5);
    $t2 = _row($a, 3);
    $t3 = _row($a, 1);

    $z1 = nd_add($t0, $t3);
    $z2 = nd_add($t1, $t2);
    $z3 = nd_add($t0, $t2);
    $z4 = nd_add($t1, $t3);
    $z5 = nd_multiply(nd_add($z3, $z4), $C_1_175875602);

    $t0 = nd_multiply($t0, $C_0_298631336);
    $t1 = nd_multiply($t1, $C_2_053119869);
    $t2 = nd_multiply($t2, $C_3_072711026);
    $t3 = nd_multiply($t3, $C_1_501321110);
    $z1 = nd_multiply($z1, 0 - $C_0_899976223);
    $z2 = nd_multiply($z2, 0 - $C_2_562915447);
    $z3 = nd_multiply($z3, 0 - $C_1_961570560);
    $z4 = nd_multiply($z4, 0 - $C_0_390180644);

    $z3 = nd_add($z3, $z5);
    $z4 = nd_add($z4, $z5);

    $t0 = nd_add($t0, nd_add($z1, $z3));
    $t1 = nd_add($t1, nd_add($z2, $z4));
    $t2 = nd_add($t2, nd_add($z2, $z3));
    $t3 = nd_add($t3, nd_add($z1, $z4));

    // Output stage, descaled by CONST_BITS - PASS1_BITS = 11.
    nd_add(descale(nd_add($tmp10, $t3), 11), 0, _row($ws, 0));
    nd_add(descale(nd_subtract($tmp10, $t3), 11), 0, _row($ws, 7));
    nd_add(descale(nd_add($tmp11, $t2), 11), 0, _row($ws, 1));
    nd_add(descale(nd_subtract($tmp11, $t2), 11), 0, _row($ws, 6));
    nd_add(descale(nd_add($tmp12, $t1), 11), 0, _row($ws, 2));
    nd_add(descale(nd_subtract($tmp12, $t1), 11), 0, _row($ws, 5));
    nd_add(descale(nd_add($tmp13, $t0), 11), 0, _row($ws, 3));
    nd_add(descale(nd_subtract($tmp13, $t0), 11), 0, _row($ws, 4));

    // ── Pass 2: the rows. Reads columns of the workspace. ──
    $out = nd_zeros([$n, 8, 8], "i64");

    $z2 = _col($ws, 2);
    $z3 = _col($ws, 6);
    $z1 = nd_multiply(nd_add($z2, $z3), $C_0_541196100);
    $tmp2 = nd_add($z1, nd_multiply($z3, 0 - $C_1_847759065));
    $tmp3 = nd_add($z1, nd_multiply($z2, $C_0_765366865));

    $w0 = _col($ws, 0);
    $w4 = _col($ws, 4);
    $tmp0 = nd_multiply(nd_add($w0, $w4), 8192);
    $tmp1 = nd_multiply(nd_subtract($w0, $w4), 8192);

    $tmp10 = nd_add($tmp0, $tmp3);
    $tmp13 = nd_subtract($tmp0, $tmp3);
    $tmp11 = nd_add($tmp1, $tmp2);
    $tmp12 = nd_subtract($tmp1, $tmp2);

    $t0 = _col($ws, 7);
    $t1 = _col($ws, 5);
    $t2 = _col($ws, 3);
    $t3 = _col($ws, 1);

    $z1 = nd_add($t0, $t3);
    $z2 = nd_add($t1, $t2);
    $z3 = nd_add($t0, $t2);
    $z4 = nd_add($t1, $t3);
    $z5 = nd_multiply(nd_add($z3, $z4), $C_1_175875602);

    $t0 = nd_multiply($t0, $C_0_298631336);
    $t1 = nd_multiply($t1, $C_2_053119869);
    $t2 = nd_multiply($t2, $C_3_072711026);
    $t3 = nd_multiply($t3, $C_1_501321110);
    $z1 = nd_multiply($z1, 0 - $C_0_899976223);
    $z2 = nd_multiply($z2, 0 - $C_2_562915447);
    $z3 = nd_multiply($z3, 0 - $C_1_961570560);
    $z4 = nd_multiply($z4, 0 - $C_0_390180644);

    $z3 = nd_add($z3, $z5);
    $z4 = nd_add($z4, $z5);

    $t0 = nd_add($t0, nd_add($z1, $z3));
    $t1 = nd_add($t1, nd_add($z2, $z4));
    $t2 = nd_add($t2, nd_add($z2, $z3));
    $t3 = nd_add($t3, nd_add($z1, $z4));

    // Descaled by CONST_BITS + PASS1_BITS + 3 = 18, then range-limited.
    nd_add(descale(nd_add($tmp10, $t3), 18), 0, _col($out, 0));
    nd_add(descale(nd_subtract($tmp10, $t3), 18), 0, _col($out, 7));
    nd_add(descale(nd_add($tmp11, $t2), 18), 0, _col($out, 1));
    nd_add(descale(nd_subtract($tmp11, $t2), 18), 0, _col($out, 6));
    nd_add(descale(nd_add($tmp12, $t1), 18), 0, _col($out, 2));
    nd_add(descale(nd_subtract($tmp12, $t1), 18), 0, _col($out, 5));
    nd_add(descale(nd_add($tmp13, $t0), 18), 0, _col($out, 3));
    nd_add(descale(nd_subtract($tmp13, $t0), 18), 0, _col($out, 4));

    return range_limit($out);
}

// range_limit($v) -> IDCT_RANGE[v & 1023], elementwise.
//
// NOT a clamp. For v in [-512, 511] it equals clamp(v + 128, 0, 255), but
// outside that window libjpeg's mask WRAPS onto deliberate table values, and
// a clamp would differ there. One gather for the whole component.
def range_limit($v) {
    $s = nd_shape($v);
    $idx = nd_bitwise_and(nd_ravel($v), 1023);
    return nd_reshape(nd_take(_range_table(), $idx), $s);
}

// ─── The scaled inverse DCTs, for draft() ───────────────────────────────
//
// libjpeg picks a SMALLER IDCT rather than resampling: scale_denom 2, 4 and
// 8 select the 4x4, 2x2 and 1x1 kernels, which take the low-frequency
// corner of the 8x8 block and transform that. These are separate kernels in
// jidctint.c with their own constants, not the 8x8 one with inputs zeroed.

// idct4($coef) -> (N, 4, 4). jidctred.c's jpeg_idct_4x4: cK is
// sqrt(2) * cos(K*pi/16), and row/column 4 of the input is SKIPPED -- the
// 4-point kernel has no term for it, which is the part a reader expects to
// be a bug and is not.
//
// Its constants are NOT the 8x8 kernel's. The scaled kernels are separate
// derivations in the C with their own FIX values and their own shifts
// (CONST_BITS+1 on the DC term, 12 out of pass 1, 19 out of pass 2), and
// borrowing the 8x8 ones produces an image that is recognisable and wrong.
$C_0_211164243 = 1730;
$C_0_509795579 = 4176;
$C_0_601344887 = 4926;
$C_1_061594337 = 8697;
$C_1_451774981 = 11893;
$C_2_172734803 = 17799;
$C_0_720959822 = 5906;
$C_0_850430095 = 6967;
$C_1_272758580 = 10426;
$C_3_624509785 = 29692;

def idct4($coef) {
    $n = nd_shape($coef)[0];
    $a = nd_reshape($coef, [$n, 8, 8]);
    $ws = nd_zeros([$n, 4, 8], "i64");

    // ── Pass 1: columns, reading rows 0, 1, 2, 3, 5, 6, 7 ──
    $tmp0 = nd_multiply(_row($a, 0), 16384);             // << CONST_BITS + 1
    $tmp2 = nd_add(nd_multiply(_row($a, 2), $C_1_847759065),
                   nd_multiply(_row($a, 6), 0 - $C_0_765366865));
    $tmp10 = nd_add($tmp0, $tmp2);
    $tmp12 = nd_subtract($tmp0, $tmp2);

    $z1 = _row($a, 7);
    $z2 = _row($a, 5);
    $z3 = _row($a, 3);
    $z4 = _row($a, 1);
    $odd0 = nd_add(nd_add(nd_multiply($z1, 0 - $C_0_211164243),
                          nd_multiply($z2, $C_1_451774981)),
                   nd_add(nd_multiply($z3, 0 - $C_2_172734803),
                          nd_multiply($z4, $C_1_061594337)));
    $odd2 = nd_add(nd_add(nd_multiply($z1, 0 - $C_0_509795579),
                          nd_multiply($z2, 0 - $C_0_601344887)),
                   nd_add(nd_multiply($z3, $C_0_899976223),
                          nd_multiply($z4, $C_2_562915447)));

    nd_add(descale(nd_add($tmp10, $odd2), 12), 0, nd_slice($ws, [null, [0, 1], null]));
    nd_add(descale(nd_subtract($tmp10, $odd2), 12), 0, nd_slice($ws, [null, [3, 4], null]));
    nd_add(descale(nd_add($tmp12, $odd0), 12), 0, nd_slice($ws, [null, [1, 2], null]));
    nd_add(descale(nd_subtract($tmp12, $odd0), 12), 0, nd_slice($ws, [null, [2, 3], null]));

    // ── Pass 2: rows. Column 4 is skipped again. ──
    $out = nd_zeros([$n, 4, 4], "i64");
    $c0 = nd_slice($ws, [null, null, [0, 1]]);
    $c1 = nd_slice($ws, [null, null, [1, 2]]);
    $c2 = nd_slice($ws, [null, null, [2, 3]]);
    $c3 = nd_slice($ws, [null, null, [3, 4]]);
    $c5 = nd_slice($ws, [null, null, [5, 6]]);
    $c6 = nd_slice($ws, [null, null, [6, 7]]);
    $c7 = nd_slice($ws, [null, null, [7, 8]]);

    $tmp0 = nd_multiply($c0, 16384);
    $tmp2 = nd_add(nd_multiply($c2, $C_1_847759065),
                   nd_multiply($c6, 0 - $C_0_765366865));
    $tmp10 = nd_add($tmp0, $tmp2);
    $tmp12 = nd_subtract($tmp0, $tmp2);

    $odd0 = nd_add(nd_add(nd_multiply($c7, 0 - $C_0_211164243),
                          nd_multiply($c5, $C_1_451774981)),
                   nd_add(nd_multiply($c3, 0 - $C_2_172734803),
                          nd_multiply($c1, $C_1_061594337)));
    $odd2 = nd_add(nd_add(nd_multiply($c7, 0 - $C_0_509795579),
                          nd_multiply($c5, 0 - $C_0_601344887)),
                   nd_add(nd_multiply($c3, $C_0_899976223),
                          nd_multiply($c1, $C_2_562915447)));

    nd_add(descale(nd_add($tmp10, $odd2), 19), 0, nd_slice($out, [null, null, [0, 1]]));
    nd_add(descale(nd_subtract($tmp10, $odd2), 19), 0, nd_slice($out, [null, null, [3, 4]]));
    nd_add(descale(nd_add($tmp12, $odd0), 19), 0, nd_slice($out, [null, null, [1, 2]]));
    nd_add(descale(nd_subtract($tmp12, $odd0), 19), 0, nd_slice($out, [null, null, [2, 3]]));

    return range_limit($out);
}

// idct2($coef) -> (N, 2, 2). jidctred.c's jpeg_idct_2x2: coefficients 0, 1,
// 3, 5 and 7 of each dimension, with 2, 4 and 6 skipped. The DC term is
// shifted by CONST_BITS+2 and the passes descale by 13 and 20.
def idct2($coef) {
    $n = nd_shape($coef)[0];
    $a = nd_reshape($coef, [$n, 8, 8]);
    $ws = nd_zeros([$n, 2, 8], "i64");

    $tmp10 = nd_multiply(_row($a, 0), 32768);            // << CONST_BITS + 2
    $tmp0 = nd_add(nd_add(nd_multiply(_row($a, 7), 0 - $C_0_720959822),
                          nd_multiply(_row($a, 5), $C_0_850430095)),
                   nd_add(nd_multiply(_row($a, 3), 0 - $C_1_272758580),
                          nd_multiply(_row($a, 1), $C_3_624509785)));

    nd_add(descale(nd_add($tmp10, $tmp0), 13), 0, nd_slice($ws, [null, [0, 1], null]));
    nd_add(descale(nd_subtract($tmp10, $tmp0), 13), 0, nd_slice($ws, [null, [1, 2], null]));

    $out = nd_zeros([$n, 2, 2], "i64");
    $c0 = nd_slice($ws, [null, null, [0, 1]]);
    $c1 = nd_slice($ws, [null, null, [1, 2]]);
    $c3 = nd_slice($ws, [null, null, [3, 4]]);
    $c5 = nd_slice($ws, [null, null, [5, 6]]);
    $c7 = nd_slice($ws, [null, null, [7, 8]]);

    $tmp10b = nd_multiply($c0, 32768);
    $tmp0b = nd_add(nd_add(nd_multiply($c7, 0 - $C_0_720959822),
                           nd_multiply($c5, $C_0_850430095)),
                    nd_add(nd_multiply($c3, 0 - $C_1_272758580),
                           nd_multiply($c1, $C_3_624509785)));

    nd_add(descale(nd_add($tmp10b, $tmp0b), 20), 0, nd_slice($out, [null, null, [0, 1]]));
    nd_add(descale(nd_subtract($tmp10b, $tmp0b), 20), 0, nd_slice($out, [null, null, [1, 2]]));

    return range_limit($out);
}

// idct1($coef) -> (N, 1, 1). jpeg_idct_1x1: the DC coefficient alone,
// descaled by 3. This is why a 1/8-scale decode is cheap -- after the scan.
def idct1($coef) {
    $n = nd_shape($coef)[0];
    $dc = nd_slice($coef, [null, [0, 1]]);
    return range_limit(nd_reshape(descale($dc, 3), [$n, 1, 1]));
}

// idct($coef, $size) -> the kernel for an output block of `size` samples.
def idct($coef, $size) {
    if ($size == 8) { return idct8($coef); }
    if ($size == 4) { return idct4($coef); }
    if ($size == 2) { return idct2($coef); }
    if ($size == 1) { return idct1($coef); }
    throw "jpegdct.idct: no kernel for a " + str($size) + "x" + str($size) +
          " block (libjpeg scales to 1, 2, 4 or 8 for scale_denom 8, 4, 2, 1)";
}

// ─── Blocks to a plane ──────────────────────────────────────────────────

// plane($blocks, $bh, $bw, $size) -> (bh*size, bw*size) samples.
//
// $blocks is (bh*bw, size, size) in block-raster order. The plane needs the
// axes in (block row, y, block col, x) order, which is one swapaxes.
def plane($blocks, $bh, $bw, $size) {
    $g = nd_reshape($blocks, [$bh, $bw, $size, $size]);
    $t = nd_ascontiguous(nd_swapaxes($g, 1, 2));
    return nd_reshape($t, [$bh * $size, $bw * $size]);
}

// ─── Upsampling ─────────────────────────────────────────────────────────
//
// Every fancy upsampler reads one sample of context on each side, and
// jdmainct.c manufactures that context by DUPLICATING the first or last real
// row; h2v1_fancy_upsample's first and last column cases are the same
// duplication done inline. So each one is written here against an
// EDGE-CLAMPED neighbour array, which makes the C's special cases fall out
// of the general one rather than needing their own code.

// _shift_cols($a, $d) -> a with every column taken from (c + d), clamped at
// both edges. One gather along axis 1.
def _shift_cols($a, $d) {
    $w = nd_shape($a)[1];
    $idx = [];
    $c = 0;
    while ($c < $w) {
        $k = $c + $d;
        if ($k < 0) { $k = 0; }
        if ($k > $w - 1) { $k = $w - 1; }
        push($idx, $k);
        $c = $c + 1;
    }
    return nd_take($a, nd($idx, "i64"), 1);
}

// _shift_rows($a, $d) -> the same along axis 0.
def _shift_rows($a, $d) {
    $h = nd_shape($a)[0];
    $idx = [];
    $r = 0;
    while ($r < $h) {
        $k = $r + $d;
        if ($k < 0) { $k = 0; }
        if ($k > $h - 1) { $k = $h - 1; }
        push($idx, $k);
        $r = $r + 1;
    }
    return nd_take($a, nd($idx, "i64"), 0);
}

// _interleave_cols($even, $odd) -> a row twice as wide, even columns from
// $even and odd columns from $odd. Two strided writes, no loop.
def _interleave_cols($even, $odd) {
    $s = nd_shape($even);
    $out = nd_zeros([$s[0], $s[1] * 2], "i64");
    nd_add($even, 0, nd_slice($out, [null, [0, $s[1] * 2, 2]]));
    nd_add($odd, 0, nd_slice($out, [null, [1, $s[1] * 2, 2]]));
    return $out;
}

// _interleave_rows($even, $odd) -> twice as tall, the same way.
def _interleave_rows($even, $odd) {
    $s = nd_shape($even);
    $out = nd_zeros([$s[0] * 2, $s[1]], "i64");
    nd_add($even, 0, nd_slice($out, [[0, $s[0] * 2, 2], null]));
    nd_add($odd, 0, nd_slice($out, [[1, $s[0] * 2, 2], null]));
    return $out;
}

// h2v1_fancy($a) -> twice as wide, jdsample.c's triangle filter.
//
//     out[2c]   = (3*in[c] + in[c-1] + 1) >> 2
//     out[2c+1] = (3*in[c] + in[c+1] + 2) >> 2
//
// with in[-1] == in[0] and in[w] == in[w-1]. The C's first column writes
// in[0] unweighted, which is this formula with in[-1] == in[0]:
// (3*v + v + 1) >> 2 == v. The last column is the same identity.
//
// The alternating +1 / +2 bias is deliberate: rounding 0.5 the same way
// every time would bias the whole image upward.
def h2v1_fancy($a) {
    $three = nd_multiply($a, 3);
    $even = nd_floor_divide(nd_add(nd_add($three, _shift_cols($a, 0 - 1)), 1), 4);
    $odd  = nd_floor_divide(nd_add(nd_add($three, _shift_cols($a, 1)), 2), 4);
    return _interleave_cols($even, $odd);
}

// h1v2_fancy($a) -> twice as tall. Same filter down the other axis, with
// bias 1 for the upper output row and 2 for the lower.
def h1v2_fancy($a) {
    $three = nd_multiply($a, 3);
    $up   = nd_floor_divide(nd_add(nd_add($three, _shift_rows($a, 0 - 1)), 1), 4);
    $down = nd_floor_divide(nd_add(nd_add($three, _shift_rows($a, 1)), 2), 4);
    return _interleave_rows($up, $down);
}

// h2v2_fancy($a) -> twice as wide AND twice as tall, 9/16-3/16-3/16-1/16.
//
// The C computes a column sum first -- 3 * the nearer row + the further row
// -- and then applies the horizontal triangle to the sums:
//
//     out[2c]   = (3*cs[c] + cs[c-1] + 8) >> 4
//     out[2c+1] = (3*cs[c] + cs[c+1] + 7) >> 4
//
// and its first-column case (cs*4 + 8) >> 4 is this with cs[-1] == cs[c],
// which is why edge clamping is all that case needs.
def h2v2_fancy($a) {
    $up   = nd_add(nd_multiply($a, 3), _shift_rows($a, 0 - 1));
    $down = nd_add(nd_multiply($a, 3), _shift_rows($a, 1));
    $rows = [];
    $k = 0;
    while ($k < 2) {
        $cs = $up;
        if ($k == 1) { $cs = $down; }
        $three = nd_multiply($cs, 3);
        $even = nd_floor_divide(nd_add(nd_add($three, _shift_cols($cs, 0 - 1)), 8), 16);
        $odd  = nd_floor_divide(nd_add(nd_add($three, _shift_cols($cs, 1)), 7), 16);
        push($rows, _interleave_cols($even, $odd));
        $k = $k + 1;
    }
    return _interleave_rows($rows[0], $rows[1]);
}

// box($a, $hx, $vx) -> replication by whole factors, jdsample.c's
// int_upsample / h2v1_upsample / h2v2_upsample. Two gathers.
def box($a, $hx, $vx) {
    $out = $a;
    if ($hx > 1) {
        $w = nd_shape($out)[1];
        $idx = [];
        $c = 0;
        while ($c < $w * $hx) { push($idx, floor($c / $hx)); $c = $c + 1; }
        $out = nd_take($out, nd($idx, "i64"), 1);
    }
    if ($vx > 1) {
        $h = nd_shape($out)[0];
        $idx = [];
        $r = 0;
        while ($r < $h * $vx) { push($idx, floor($r / $vx)); $r = $r + 1; }
        $out = nd_take($out, nd($idx, "i64"), 0);
    }
    return $out;
}

// upsample($a, $hx, $vx, $fancy) -> the plane expanded hx by vx.
//
// The selection is jdsample.c's _jinit_upsampler, which is the reason this
// takes a `fancy` flag rather than deciding for itself: `do_fancy` is
// `do_fancy_upsampling && min_DCT_scaled_size > 1`, and a component narrower
// than three samples uses the box filter even when fancy is on.
def upsample($a, $hx, $vx, $fancy) {
    if ($hx == 1 && $vx == 1) { return $a; }
    $w = nd_shape($a)[1];
    if ($fancy && $w > 2) {
        if ($hx == 2 && $vx == 1) { return h2v1_fancy($a); }
        if ($hx == 1 && $vx == 2) { return h1v2_fancy($a); }
        if ($hx == 2 && $vx == 2) { return h2v2_fancy($a); }
    }
    if ($fancy && $hx == 1 && $vx == 2) { return h1v2_fancy($a); }
    return box($a, $hx, $vx);
}

// ─── Colour transforms ──────────────────────────────────────────────────

def _tables() {
    if ($_C.crr == null) {
        $_C.crr = nd(jt.CR_R, "i64");
        $_C.cbb = nd(jt.CB_B, "i64");
        $_C.crg = nd(jt.CR_G, "i64");
        $_C.cbg = nd(jt.CB_G, "i64");
    }
    return true;
}

def _gather($tab, $a) {
    return nd_reshape(nd_take($tab, nd_ravel($a)), nd_shape($a));
}

// ycc_to_rgb($y, $cb, $cr) -> [r, g, b], jdcolor.c's ycc_rgb_convert.
//
//     r = limit[y + Cr_r[cr]]
//     g = limit[y + ((Cb_g[cb] + Cr_g[cr]) >> 16)]
//     b = limit[y + Cb_b[cb]]
//
// where `limit` here is the SIMPLE range-limit table, not the post-IDCT one:
// over the reachable domain [-179, 434] it is a plain clamp to 0..255, and
// that equivalence is asserted by the test rather than assumed.
def ycc_to_rgb($y, $cb, $cr) {
    _tables();
    $r = nd_clip(nd_add($y, _gather($_C.crr, $cr)), 0, 255);
    $g = nd_clip(nd_add($y, nd_floor_divide(
             nd_add(_gather($_C.cbg, $cb), _gather($_C.crg, $cr)), 65536)), 0, 255);
    $b = nd_clip(nd_add($y, _gather($_C.cbb, $cb)), 0, 255);
    return [$r, $g, $b];
}

// ycck_to_cmyk($y, $cb, $cr, $k) -> [c, m, y, k], jdcolor.c's
// ycck_cmyk_convert: the YCbCr-to-RGB transform, then 255 - each, with K
// passed through untouched.
def ycck_to_cmyk($y, $cb, $cr, $k) {
    _tables();
    $c = nd_clip(nd_subtract(255, nd_add($y, _gather($_C.crr, $cr))), 0, 255);
    $m = nd_clip(nd_subtract(255, nd_add($y, nd_floor_divide(
             nd_add(_gather($_C.cbg, $cb), _gather($_C.crg, $cr)), 65536))), 0, 255);
    $yy = nd_clip(nd_subtract(255, nd_add($y, _gather($_C.cbb, $cb))), 0, 255);
    return [$c, $m, $yy, $k];
}

// ════════════════════════════════════════════════════════════════════════
//  The encoder's side of the same arithmetic
// ════════════════════════════════════════════════════════════════════════
//
// jfdctint.c, jccolor.c and jcsample.c, transcribed the same way as their
// decoding twins: the arithmetic is the C's, statement for statement, and
// only what each statement applies to changes.

$_FC = {"ry": null, "gy": null, "by": null, "rcb": null, "gcb": null,
        "bcb": null, "rcr": null, "gcr": null, "bcr": null};

def _ftables() {
    if ($_FC.ry == null) {
        $_FC.ry = nd(jt.RGB_RY, "i64");
        $_FC.gy = nd(jt.RGB_GY, "i64");
        $_FC.by = nd(jt.RGB_BY, "i64");
        $_FC.rcb = nd(jt.RGB_RCB, "i64");
        $_FC.gcb = nd(jt.RGB_GCB, "i64");
        $_FC.bcb = nd(jt.RGB_BCB, "i64");
        $_FC.rcr = nd(jt.RGB_RCR, "i64");
        $_FC.gcr = nd(jt.RGB_GCR, "i64");
        $_FC.bcr = nd(jt.RGB_BCR, "i64");
    }
    return true;
}

// rgb_to_ycc($r, $g, $b) -> [y, cb, cr], jccolor.c's rgb_ycc_convert.
//
//   y  = (R_Y[r]  + G_Y[g]  + B_Y[b])  >> 16
//   cb = (R_CB[r] + G_CB[g] + B_CB[b]) >> 16
//   cr = (R_CR[r] + G_CR[g] + B_CR[b]) >> 16
//
// with the rounding half and the 128 centre folded into the B_Y, B_CB and
// R_CR tables, exactly as rgb_ycc_start folds them.
def rgb_to_ycc($r, $g, $b) {
    _ftables();
    $y = nd_floor_divide(nd_add(nd_add(_gather($_FC.ry, $r), _gather($_FC.gy, $g)),
                                _gather($_FC.by, $b)), 65536);
    $cb = nd_floor_divide(nd_add(nd_add(_gather($_FC.rcb, $r), _gather($_FC.gcb, $g)),
                                 _gather($_FC.bcb, $b)), 65536);
    $cr = nd_floor_divide(nd_add(nd_add(_gather($_FC.rcr, $r), _gather($_FC.gcr, $g)),
                                 _gather($_FC.bcr, $b)), 65536);
    return [$y, $cb, $cr];
}

// rgb_to_y($r, $g, $b) -> the luminance alone, for a greyscale encode.
// jccolor.c's rgb_gray_convert, which is the same first row.
def rgb_to_y($r, $g, $b) {
    _ftables();
    return nd_floor_divide(nd_add(nd_add(_gather($_FC.ry, $r), _gather($_FC.gy, $g)),
                                  _gather($_FC.by, $b)), 65536);
}

// pad_edge($a, $h, $w) -> the plane grown to (h, w) by DUPLICATING the last
// row and the last column, which is jcsample.c's expand_right_edge and
// jcmainct.c's bottom-row duplication. Padding with zeros instead would put
// a hard edge in the block the DCT then has to spend bits on.
def pad_edge($a, $h, $w) {
    $s = nd_shape($a);
    $out = $a;
    if ($w > $s[1]) {
        $idx = [];
        $c = 0;
        while ($c < $w) {
            $k = $c;
            if ($k > $s[1] - 1) { $k = $s[1] - 1; }
            push($idx, $k);
            $c = $c + 1;
        }
        $out = nd_take($out, nd($idx, "i64"), 1);
    }
    if ($h > $s[0]) {
        $idx = [];
        $r = 0;
        while ($r < $h) {
            $k = $r;
            if ($k > $s[0] - 1) { $k = $s[0] - 1; }
            push($idx, $k);
            $r = $r + 1;
        }
        $out = nd_take($out, nd($idx, "i64"), 0);
    }
    return $out;
}

// _evens($a, $axis) / _odds($a, $axis) -> every other sample along an axis.
def _evens($a, $axis) {
    $s = nd_shape($a);
    if ($axis == 1) { return nd_slice($a, [null, [0, $s[1], 2]]); }
    return nd_slice($a, [[0, $s[0], 2], null]);
}
def _odds($a, $axis) {
    $s = nd_shape($a);
    if ($axis == 1) { return nd_slice($a, [null, [1, $s[1], 2]]); }
    return nd_slice($a, [[1, $s[0], 2], null]);
}

// downsample($a, $hx, $vx) -> the plane reduced by hx horizontally and vx
// vertically, jcsample.c.
//
// The two common ratios have their own kernels in the C, and their ROUNDING
// BIAS ALTERNATES ACROSS COLUMNS -- 0,1,0,1 for h2v1 and 1,2,1,2 for h2v2 --
// which is a detail that costs half a level of brightness if it is dropped.
// Every other ratio goes through int_downsample's plain average.
def downsample($a, $hx, $vx) {
    if ($hx == 1 && $vx == 1) { return $a; }
    if ($hx == 2 && $vx == 1) {
        $sum = nd_add(_evens($a, 1), _odds($a, 1));
        $w = nd_shape($sum)[1];
        $bias = nd_reshape(nd_mod(nd_arange(0, $w, 1), 2), [1, $w]);
        return nd_floor_divide(nd_add($sum, $bias), 2);
    }
    if ($hx == 2 && $vx == 2) {
        $top = nd_add(_evens(_evens($a, 0), 1), _odds(_evens($a, 0), 1));
        $bot = nd_add(_evens(_odds($a, 0), 1), _odds(_odds($a, 0), 1));
        $sum = nd_add($top, $bot);
        $w = nd_shape($sum)[1];
        $bias = nd_reshape(nd_add(nd_mod(nd_arange(0, $w, 1), 2), 1), [1, $w]);
        return nd_floor_divide(nd_add($sum, $bias), 4);
    }
    // int_downsample: (sum + numpix/2) / numpix, no alternating bias.
    $s = nd_shape($a);
    $oh = floor($s[0] / $vx);
    $ow = floor($s[1] / $hx);
    $acc = nd_zeros([$oh, $ow], "i64");
    $r = 0;
    while ($r < $vx) {
        $c = 0;
        while ($c < $hx) {
            $part = nd_slice($a, [[$r, $r + $oh * $vx, $vx], [$c, $c + $ow * $hx, $hx]]);
            $acc = nd_add($acc, $part);
            $c = $c + 1;
        }
        $r = $r + 1;
    }
    $numpix = $hx * $vx;
    return nd_floor_divide(nd_add($acc, floor($numpix / 2)), $numpix);
}

// blocks_of($plane, $bh, $bw) -> (bh*bw, 8, 8), the inverse of plane().
def blocks_of($plane, $bh, $bw) {
    $g = nd_reshape($plane, [$bh, 8, $bw, 8]);
    $t = nd_ascontiguous(nd_swapaxes($g, 1, 2));
    return nd_reshape($t, [$bh * $bw, 8, 8]);
}

// ─── The forward DCT ────────────────────────────────────────────────────

// fdct8($blocks) -> (N, 64) coefficients in natural order.
//
// jpeg_fdct_islow. $blocks is (N, 8, 8) SAMPLES, from which 128 is
// subtracted here (jcdctmgr.c's CENTERJSAMPLE shift), so the caller passes
// 0..255 and gets coefficients.
//
// Pass 1 processes rows and so reads a COLUMN slice -- (N, 8, 1), every row
// of every block at once -- and pass 2 processes columns and reads a ROW
// slice. Same arrangement as idct8, mirrored.
def fdct8($blocks) {
    $n = nd_shape($blocks)[0];
    $a = nd_subtract($blocks, 128);
    $ws = nd_zeros([$n, 8, 8], "i64");

    // ── Pass 1: rows ──
    $t0 = nd_add(_col($a, 0), _col($a, 7));
    $t7 = nd_subtract(_col($a, 0), _col($a, 7));
    $t1 = nd_add(_col($a, 1), _col($a, 6));
    $t6 = nd_subtract(_col($a, 1), _col($a, 6));
    $t2 = nd_add(_col($a, 2), _col($a, 5));
    $t5 = nd_subtract(_col($a, 2), _col($a, 5));
    $t3 = nd_add(_col($a, 3), _col($a, 4));
    $t4 = nd_subtract(_col($a, 3), _col($a, 4));

    $t10 = nd_add($t0, $t3);
    $t13 = nd_subtract($t0, $t3);
    $t11 = nd_add($t1, $t2);
    $t12 = nd_subtract($t1, $t2);

    nd_add(nd_multiply(nd_add($t10, $t11), 4), 0, _col($ws, 0));
    nd_add(nd_multiply(nd_subtract($t10, $t11), 4), 0, _col($ws, 4));
    $z1 = nd_multiply(nd_add($t12, $t13), $C_0_541196100);
    nd_add(descale(nd_add($z1, nd_multiply($t13, $C_0_765366865)), 11), 0, _col($ws, 2));
    nd_add(descale(nd_add($z1, nd_multiply($t12, 0 - $C_1_847759065)), 11), 0, _col($ws, 6));

    $z1 = nd_add($t4, $t7);
    $z2 = nd_add($t5, $t6);
    $z3 = nd_add($t4, $t6);
    $z4 = nd_add($t5, $t7);
    $z5 = nd_multiply(nd_add($z3, $z4), $C_1_175875602);
    $t4 = nd_multiply($t4, $C_0_298631336);
    $t5 = nd_multiply($t5, $C_2_053119869);
    $t6 = nd_multiply($t6, $C_3_072711026);
    $t7 = nd_multiply($t7, $C_1_501321110);
    $z1 = nd_multiply($z1, 0 - $C_0_899976223);
    $z2 = nd_multiply($z2, 0 - $C_2_562915447);
    $z3 = nd_multiply($z3, 0 - $C_1_961570560);
    $z4 = nd_multiply($z4, 0 - $C_0_390180644);
    $z3 = nd_add($z3, $z5);
    $z4 = nd_add($z4, $z5);
    nd_add(descale(nd_add(nd_add($t4, $z1), $z3), 11), 0, _col($ws, 7));
    nd_add(descale(nd_add(nd_add($t5, $z2), $z4), 11), 0, _col($ws, 5));
    nd_add(descale(nd_add(nd_add($t6, $z2), $z3), 11), 0, _col($ws, 3));
    nd_add(descale(nd_add(nd_add($t7, $z1), $z4), 11), 0, _col($ws, 1));

    // ── Pass 2: columns ──
    $out = nd_zeros([$n, 8, 8], "i64");
    $t0 = nd_add(_row($ws, 0), _row($ws, 7));
    $t7 = nd_subtract(_row($ws, 0), _row($ws, 7));
    $t1 = nd_add(_row($ws, 1), _row($ws, 6));
    $t6 = nd_subtract(_row($ws, 1), _row($ws, 6));
    $t2 = nd_add(_row($ws, 2), _row($ws, 5));
    $t5 = nd_subtract(_row($ws, 2), _row($ws, 5));
    $t3 = nd_add(_row($ws, 3), _row($ws, 4));
    $t4 = nd_subtract(_row($ws, 3), _row($ws, 4));

    $t10 = nd_add($t0, $t3);
    $t13 = nd_subtract($t0, $t3);
    $t11 = nd_add($t1, $t2);
    $t12 = nd_subtract($t1, $t2);

    nd_add(descale(nd_add($t10, $t11), 2), 0, _row($out, 0));
    nd_add(descale(nd_subtract($t10, $t11), 2), 0, _row($out, 4));
    $z1 = nd_multiply(nd_add($t12, $t13), $C_0_541196100);
    nd_add(descale(nd_add($z1, nd_multiply($t13, $C_0_765366865)), 15), 0, _row($out, 2));
    nd_add(descale(nd_add($z1, nd_multiply($t12, 0 - $C_1_847759065)), 15), 0, _row($out, 6));

    $z1 = nd_add($t4, $t7);
    $z2 = nd_add($t5, $t6);
    $z3 = nd_add($t4, $t6);
    $z4 = nd_add($t5, $t7);
    $z5 = nd_multiply(nd_add($z3, $z4), $C_1_175875602);
    $t4 = nd_multiply($t4, $C_0_298631336);
    $t5 = nd_multiply($t5, $C_2_053119869);
    $t6 = nd_multiply($t6, $C_3_072711026);
    $t7 = nd_multiply($t7, $C_1_501321110);
    $z1 = nd_multiply($z1, 0 - $C_0_899976223);
    $z2 = nd_multiply($z2, 0 - $C_2_562915447);
    $z3 = nd_multiply($z3, 0 - $C_1_961570560);
    $z4 = nd_multiply($z4, 0 - $C_0_390180644);
    $z3 = nd_add($z3, $z5);
    $z4 = nd_add($z4, $z5);
    nd_add(descale(nd_add(nd_add($t4, $z1), $z3), 15), 0, _row($out, 7));
    nd_add(descale(nd_add(nd_add($t5, $z2), $z4), 15), 0, _row($out, 5));
    nd_add(descale(nd_add(nd_add($t6, $z2), $z3), 15), 0, _row($out, 3));
    nd_add(descale(nd_add(nd_add($t7, $z1), $z4), 15), 0, _row($out, 1));

    return nd_reshape($out, [$n, 64]);
}

// quantise($coef, $qtab) -> (N, 64) quantised coefficients.
//
// jcdctmgr.c's forward_DCT for JDCT_ISLOW: the divisor is the table value
// times 8 (the DCT's own scale factor), and the division ROUNDS TO NEAREST
// BY MAGNITUDE -- the C negates, adds half the divisor, divides and negates
// back, which is not what a floor division does for a negative numerator.
def quantise($coef, $qtab) {
    if (len($qtab) != 64) {
        throw "jpegdct.quantise: a quantisation table is 64 entries, got " +
              str(len($qtab));
    }
    $q = nd_multiply(nd_reshape(nd($qtab, "i64"), [1, 64]), 8);
    $half = nd_floor_divide($q, 2);
    $mag = nd_floor_divide(nd_add(nd_abs($coef), $half), $q);
    return nd_multiply($mag, nd_sign($coef));
}
