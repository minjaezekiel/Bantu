// ════════════════════════════════════════════════════════════════════════
//  pillow/quantize.b — Image.quantize: an ADAPTIVE palette, which is what
//  convert("P") needs for an RGBA source and what GIF will need in P8.
//
//  Pillow offers four methods. Three are implemented here and the fourth is
//  declined by name:
//
//      0  MEDIANCUT     Quant.c's median cut, over the box with the most
//                       PIXELS
//      1  MAXCOVERAGE   the same, over the box with the largest VOLUME
//      2  FASTOCTREE    an octree, which is the only method Pillow allows
//                       for RGBA because it can carry alpha
//      3  LIBIMAGEQUANT an external C library (PW10's reasoning: a
//                       half-implementation of someone else's quantiser
//                       would be worse than declining it)
//
//  ─── WHERE THIS MATCHES PILLOW AND WHERE IT DOES NOT ─────────────────
//
//  MATCHES, from Quant.c, and each one is a choice a reimplementation gets
//  wrong by default:
//    * the split axis is the channel with the largest range WEIGHTED BY
//      LUMINANCE -- (rh-rl)*77, (gh-gl)*150, (bh-bl)*29 -- with ties to the
//      lower-indexed channel, not the largest raw range.
//    * the split point is the COUNT MEDIAN along that axis, not the
//      midpoint of the colour range, and every colour sharing the median's
//      axis value stays on the left, so equal values are never separated.
//    * a box of volume 1 is discarded rather than split, so the palette can
//      come back SMALLER than asked for and is not padded.
//    * a palette entry is the PIXEL-WEIGHTED mean of its box, rounded half
//      up -- (int)(0.5 + sum/count) per channel.
//    * pixels are mapped to the exact nearest entry in squared RGB with
//      alpha ignored, which is what Quant.c's pruned search computes.
//
//  DOES NOT MATCH, stated rather than hidden: Quant.c hashes colours into a
//  65,536-entry table and MERGES colours whose hashes collide, coarsening
//  the whole colour space with a right shift when the table overflows. This
//  works from the exact distinct colours instead. On an image with few
//  enough colours to avoid a collision the two agree; on a photograph the
//  palette can differ, and the test asserts the honest claim -- that ours is
//  no worse than Pillow's against the source, measured -- rather than
//  byte equality it cannot have.
//
//  FASTOCTREE is a straight octree: distinct colours inserted to depth 8,
//  then the deepest, least-populous nodes merged into their parents until
//  the leaf count fits. Pillow's own octree makes different merge choices,
//  so this one is also compared by error rather than by bytes.
// ════════════════════════════════════════════════════════════════════════

include "./mode.b" as mode;
include "./pxbuf.b" as pxbuf;

$MEDIANCUT = 0;
$MAXCOVERAGE = 1;
$FASTOCTREE = 2;
$LIBIMAGEQUANT = 3;

// A budget, in the shape of PW6 and PW13: the box lists hold one Bantu
// Value per distinct colour per channel, so a colour count this large is
// refused by name rather than left to swap.
$CFG = {"MAX_DISTINCT": 400000};
def max_distinct()       { return $CFG.MAX_DISTINCT; }
def set_max_distinct($n) { $CFG.MAX_DISTINCT = $n; return $n; }

// _bands($px, $m) -> [r, g, b] as (h*w) flat arrays, and the alpha band or
// null. L is treated as grey RGB, which is what quantizing an L image means.
def _bands($px, $m) {
    $d = pxbuf.dims($px);
    $n = $d[0] * $d[1];
    if ($m == "L" || $m == "1" || $m == "P") {
        $g = nd_reshape(nd_clip(pxbuf.band($px, 0), 0, 255), [$n]);
        return [$g, $g, $g, null];
    }
    if ($m == "RGB" || $m == "RGBX" || $m == "YCbCr") {
        return [nd_reshape(pxbuf.materialise(pxbuf.band($px, 0), "i64"), [$n]),
                nd_reshape(pxbuf.materialise(pxbuf.band($px, 1), "i64"), [$n]),
                nd_reshape(pxbuf.materialise(pxbuf.band($px, 2), "i64"), [$n]),
                null];
    }
    if ($m == "RGBA" || $m == "RGBa") {
        return [nd_reshape(pxbuf.materialise(pxbuf.band($px, 0), "i64"), [$n]),
                nd_reshape(pxbuf.materialise(pxbuf.band($px, 1), "i64"), [$n]),
                nd_reshape(pxbuf.materialise(pxbuf.band($px, 2), "i64"), [$n]),
                nd_reshape(pxbuf.materialise(pxbuf.band($px, 3), "i64"), [$n])];
    }
    throw "Image.quantize: cannot quantize mode " + str($m) +
          "; Pillow quantizes L, RGB and RGBA";
}

// _histogram($r, $g, $b) -> the distinct colours and their pixel counts.
//
// One pack, one nd_unique, one nd_searchsorted: no loop over pixels.
def _histogram($r, $g, $b) {
    $packed = nd_add(nd_add(nd_multiply($r, 65536), nd_multiply($g, 256)), $b);
    $uniq = nd_unique($packed);
    $nu = nd_shape($uniq)[0];
    if ($nu > $CFG.MAX_DISTINCT) {
        throw "Image.quantize: this image has " + str($nu) +
              " distinct colours, over the budget of " + str($CFG.MAX_DISTINCT) +
              ". Median cut holds one value per distinct colour per channel; " +
              "reduce the image first, or raise the budget with " +
              "quantize.set_max_distinct().";
    }
    $sorted = nd_sort($packed);
    $left = nd_searchsorted($sorted, $uniq);
    $ll = nd_to_list($left);
    $total = nd_shape($packed)[0];
    $counts = [];
    $i = 0;
    while ($i < $nu) {
        if ($i + 1 < $nu) { push($counts, $ll[$i + 1] - $ll[$i]); }
        else { push($counts, $total - $ll[$i]); }
        $i = $i + 1;
    }
    $ul = nd_to_list($uniq);
    $ur = [];
    $ug = [];
    $ub = [];
    $i = 0;
    while ($i < $nu) {
        $v = $ul[$i];
        $rr = floor($v / 65536);
        $gg = floor(($v - $rr * 65536) / 256);
        push($ur, $rr);
        push($ug, $gg);
        push($ub, $v - $rr * 65536 - $gg * 256);
        $i = $i + 1;
    }
    // Both forms of the four columns: the lists for the octree's dict keys
    // and the arrays for median cut's splits.
    return {"n": $nu, "r": $ur, "g": $ug, "b": $ub, "count": $counts,
            "ar": nd($ur, "i64"), "ag": nd($ug, "i64"), "ab": nd($ub, "i64"),
            "acount": nd($counts, "i64"),
            "packed": $ul, "total": $total};
}

// ─── Median cut, with every split an array operation ────────────────────
//
// THE SHAPE THAT MATTERS. A box is three index ARRAYS -- its colours in
// order of red, of green and of blue -- and a split is:
//
//     the count median   nd_cumsum over the gathered counts, then
//                        nd_searchsorted for the crossing point
//     the sides          one nd_greater over the box's axis values
//     the partition      nd_compress of each of the three orders by that
//                        mask, which keeps them sorted for free
//     the new bounds     nd_min and nd_max over the gathered values
//
// Fifteen array calls per split, whatever the box holds. Written with Bantu
// lists and per-colour loops instead -- the obvious transcription of
// Quant.c's linked lists -- the same image took 48 seconds, then 8.7 after
// the dict-field reads were localised (PW14), and 0.2 after this. The rule
// is PW2's: no loop proportional to the DATA, and 19,000 distinct colours
// is data.

def _gatherVals($h, $ch, $idx) { return nd_take($h[$ch], $idx); }

// _boundsOf($h, $idx) -> [[rlo, glo, blo], [rhi, ghi, bhi]].
def _boundsOf($h, $idx) {
    $lo = [];
    $hi = [];
    $chs = ["ar", "ag", "ab"];
    $k = 0;
    while ($k < 3) {
        $v = nd_take($h[$chs[$k]], $idx);
        push($lo, pxbuf.scalar(nd_min($v)));
        push($hi, pxbuf.scalar(nd_max($v)));
        $k = $k + 1;
    }
    return [$lo, $hi];
}

def _volume($lo, $hi) {
    return ($hi[0] - $lo[0] + 1) * ($hi[1] - $lo[1] + 1) * ($hi[2] - $lo[2] + 1);
}

def _mkBox($h, $ir, $ig, $ib, $count) {
    $b = _boundsOf($h, $ir);
    return {"ir": $ir, "ig": $ig, "ib": $ib, "count": $count,
            "lo": $b[0], "hi": $b[1], "vol": _volume($b[0], $b[1])};
}

// _split($h, $box) -> [left, right], or null if the box cannot be split.
def _split($h, $box) {
    $lo = $box.lo;
    $hi = $box.hi;
    // The axis: the largest range WEIGHTED BY LUMINANCE, ties to the lower
    // channel. 77, 150 and 29 are Quant.c's own weights.
    $f0 = ($hi[0] - $lo[0]) * 77;
    $f1 = ($hi[1] - $lo[1]) * 150;
    $f2 = ($hi[2] - $lo[2]) * 29;
    $axis = 0;
    $best = $f0;
    if ($f1 > $best) { $axis = 1; $best = $f1; }
    if ($f2 > $best) { $axis = 2; $best = $f2; }
    $chs = ["ar", "ag", "ab"];
    $orders = ["ir", "ig", "ib"];
    $order = $box[$orders[$axis]];
    $n = nd_shape($order)[0];
    if ($n < 2) { return null; }

    $vals = nd_take($h[$chs[$axis]], $order);
    $counts = nd_take($h.acount, $order);
    // The COUNT MEDIAN: the first position where the running total exceeds
    // half the box, which searchsorted finds without a loop.
    $cum = nd_cumsum($counts, 0);
    $at = pxbuf.scalar(nd_searchsorted($cum, nd([floor($box.count / 2) + 1], "i64")));
    if (type($at) == "list") { $at = $at[0]; }
    if ($at >= $n) { $at = $n - 1; }
    $cut = nd_get($vals, [$at]);

    // Every colour sharing the median's axis value stays on the LEFT.
    $side = nd_greater($h[$chs[$axis]], $cut);
    $anyRight = pxbuf.scalar(nd_sum(nd_where(nd_take($side, $order), 1, 0))) > 0;
    if (!$anyRight) {
        // Quant.c's fallback: move the colours equal to the TAIL value to
        // the right, so neither child is empty.
        $tail = nd_get($vals, [$n - 1]);
        if ($tail == nd_get($vals, [0])) { return null; }
        $side = nd_greater_equal($h[$chs[$axis]], $tail);
        $anyRight = pxbuf.scalar(nd_sum(nd_where(nd_take($side, $order), 1, 0))) > 0;
        if (!$anyRight) { return null; }
    }

    // Partition all three orders by the same mask, which keeps each of them
    // sorted -- nd_compress is a stable selection.
    $l = [];
    $r = [];
    $k = 0;
    while ($k < 3) {
        $o = $box[$orders[$k]];
        $m = nd_take($side, $o);
        // nd_compress takes the MASK first, like NumPy's compress.
        push($r, nd_compress($m, $o));
        push($l, nd_compress(nd_logical_not($m), $o));
        $k = $k + 1;
    }
    if (nd_shape($l[0])[0] == 0 || nd_shape($r[0])[0] == 0) { return null; }
    $lc = pxbuf.scalar(nd_sum(nd_take($h.acount, $l[0])));
    return [_mkBox($h, $l[0], $l[1], $l[2], $lc),
            _mkBox($h, $r[0], $r[1], $r[2], $box.count - $lc)];
}

// _medianCut($h, $colors, $byVolume) -> the boxes.
def _medianCut($h, $colors, $byVolume) {
    $all = nd_arange(0, $h.n, 1);
    $root = _mkBox($h, nd_argsort($h.ar), nd_argsort($h.ag), nd_argsort($h.ab),
                   $h.total);
    $boxes = [$root];

    while (len($boxes) < $colors) {
        // The next box: the largest by pixel count (MEDIANCUT) or by volume
        // weighted by coverage (MAXCOVERAGE), skipping any of volume 1.
        $pick = 0 - 1;
        $bestKey = 0 - 1;
        $i = 0;
        while ($i < len($boxes)) {
            $b = $boxes[$i];
            if ($b.vol > 1) {
                $key = $b.count;
                if ($byVolume) { $key = $b.vol * $b.count; }
                if ($key > $bestKey) { $bestKey = $key; $pick = $i; }
            }
            $i = $i + 1;
        }
        if ($pick < 0) { break; }
        $kids = _split($h, $boxes[$pick]);
        if ($kids == null) {
            $boxes[$pick].vol = 1;          // unsplittable: stop picking it
            continue;
        }
        $next = [];
        $i = 0;
        while ($i < len($boxes)) {
            if ($i != $pick) { push($next, $boxes[$i]); }
            $i = $i + 1;
        }
        push($next, $kids[0]);
        push($next, $kids[1]);
        $boxes = $next;
    }
    return $boxes;
}

// _paletteOf($h, $boxes) -> [the palette as a list of [r, g, b], the
// per-colour box assignment as an array].
//
// Each entry is the PIXEL-WEIGHTED mean of its box, rounded half up:
// (int)(0.5 + sum/count) per channel, which is what
// compute_palette_from_median_cut computes by walking the pixels.
def _paletteOf($h, $boxes) {
    $pal = [];
    $assign = nd_zeros([$h.n], "i64");
    $i = 0;
    while ($i < len($boxes)) {
        $idx = $boxes[$i].ir;
        $w = nd_take($h.acount, $idx);
        $n = pxbuf.scalar(nd_sum($w));
        if ($n > 0) {
            $sr = pxbuf.scalar(nd_sum(nd_multiply(nd_take($h.ar, $idx), $w)));
            $sg = pxbuf.scalar(nd_sum(nd_multiply(nd_take($h.ag, $idx), $w)));
            $sb = pxbuf.scalar(nd_sum(nd_multiply(nd_take($h.ab, $idx), $w)));
            push($pal, [floor(0.5 + $sr / $n), floor(0.5 + $sg / $n),
                        floor(0.5 + $sb / $n)]);
        } else {
            push($pal, [0, 0, 0]);
        }
        nd_put($assign, $idx, nd_full([nd_shape($idx)[0]], $i, "i64"));
        $i = $i + 1;
    }
    return [$pal, $assign];
}

// _mapNearest($h, $pal) -> the palette index of every distinct colour, by
// exact squared RGB distance with ties to the lowest index.
//
// One pass per palette entry over the DISTINCT COLOURS, never over the
// pixels: Quant.c's pruned search from the pixel's own box computes the
// same exact nearest neighbour, and the distinct-colour table is how the
// pixels then get their index in two gathers.
def _mapNearest($h, $pal) {
    $best = nd_full([$h.n], 0, "i64");
    $bestD = null;
    $i = 0;
    while ($i < len($pal)) {
        $dr = nd_subtract($h.ar, $pal[$i][0]);
        $dg = nd_subtract($h.ag, $pal[$i][1]);
        $db = nd_subtract($h.ab, $pal[$i][2]);
        $d = nd_add(nd_add(nd_multiply($dr, $dr), nd_multiply($dg, $dg)),
                    nd_multiply($db, $db));
        if ($i == 0) { $bestD = $d; }
        else {
            $closer = nd_less($d, $bestD);
            $best = nd_where($closer, $i, $best);
            $bestD = nd_where($closer, $d, $bestD);
        }
        $i = $i + 1;
    }
    return $best;
}

// _octree($h, $colors) -> a palette by octree reduction.
//
// Distinct colours are leaves at depth 8. Then, deepest level first, the
// SIBLING GROUPS with the fewest pixels are merged into their parent one
// group at a time until the leaf count fits -- merging a whole level at
// once (the obvious version) overshoots badly and costs several times the
// error, which is measurable and was measured.
//
// Pillow's own fast octree picks its merges differently, so this is
// compared by error against the source rather than by bytes.
def _octree($h, $colors) {
    // A leaf is keyed by its colour prefix at the current depth. The four
    // colour lists are localised first, for PW14's reason.
    $hr = $h.r;
    $hg = $h.g;
    $hb = $h.b;
    $hc = $h.count;
    $leaves = {};
    $i = 0;
    while ($i < $h.n) {
        $w = $hc[$i];
        $key = str($hr[$i]) + "," + str($hg[$i]) + "," + str($hb[$i]);
        $e = $leaves[$key];
        if ($e == null) {
            $leaves[$key] = {"r": $hr[$i] * $w, "g": $hg[$i] * $w,
                             "b": $hb[$i] * $w, "n": $w};
        } else {
            $e.r = $e.r + $hr[$i] * $w;
            $e.g = $e.g + $hg[$i] * $w;
            $e.b = $e.b + $hb[$i] * $w;
            $e.n = $e.n + $w;
            $leaves[$key] = $e;
        }
        $i = $i + 1;
    }

    $depth = 8;
    while ($depth > 0 && len(keys($leaves)) > $colors) {
        $shift = pow(2, 9 - $depth);           // the parent's quantisation step
        // Group the leaves by parent.
        $groups = {};
        for $k, $v in $leaves.items() {
            $f = split($k, ",");
            $pk = str(floor(num($f[0]) / $shift) * $shift) + "," +
                  str(floor(num($f[1]) / $shift) * $shift) + "," +
                  str(floor(num($f[2]) / $shift) * $shift);
            $g = $groups[$pk];
            if ($g == null) { $groups[$pk] = {"keys": [$k], "n": $v.n}; }
            else {
                push($g.keys, $k);
                $g.n = $g.n + $v.n;
                $groups[$pk] = $g;
            }
        }
        // The merge order: fewest pixels first, and only groups that
        // actually reduce the leaf count.
        $order = [];
        for $k, $v in $groups.items() {
            if (len($v.keys) > 1) { push($order, [$v.n, $k]); }
        }
        if (len($order) == 0) { $depth = $depth - 1; continue; }
        // An insertion sort over at most a few thousand groups.
        $sorted = [];
        $i = 0;
        while ($i < len($order)) {
            $at = len($sorted);
            $j = 0;
            while ($j < len($sorted)) {
                if ($sorted[$j][0] > $order[$i][0]) { $at = $j; break; }
                $j = $j + 1;
            }
            $new = [];
            $k = 0;
            while ($k < $at) { push($new, $sorted[$k]); $k = $k + 1; }
            push($new, $order[$i]);
            while ($k < len($sorted)) { push($new, $sorted[$k]); $k = $k + 1; }
            $sorted = $new;
            $i = $i + 1;
        }

        $i = 0;
        while ($i < len($sorted) && len(keys($leaves)) > $colors) {
            $pk = $sorted[$i][1];
            $g = $groups[$pk];
            $sr = 0;
            $sg = 0;
            $sb = 0;
            $sn = 0;
            $k = 0;
            while ($k < len($g.keys)) {
                $v = $leaves[$g.keys[$k]];
                if ($v != null) {
                    $sr = $sr + $v.r;
                    $sg = $sg + $v.g;
                    $sb = $sb + $v.b;
                    $sn = $sn + $v.n;
                }
                $k = $k + 1;
            }
            $out = {};
            for $lk, $lv in $leaves.items() {
                $keep = true;
                $k = 0;
                while ($k < len($g.keys)) {
                    if ($lk == $g.keys[$k]) { $keep = false; break; }
                    $k = $k + 1;
                }
                if ($keep) { $out[$lk] = $lv; }
            }
            $out[$pk] = {"r": $sr, "g": $sg, "b": $sb, "n": $sn};
            $leaves = $out;
            $i = $i + 1;
        }
        $depth = $depth - 1;
    }

    $pal = [];
    for $k, $v in $leaves.items() {
        if (len($pal) < $colors && $v.n > 0) {
            push($pal, [floor(0.5 + $v.r / $v.n), floor(0.5 + $v.g / $v.n),
                        floor(0.5 + $v.b / $v.n)]);
        }
    }
    return $pal;
}

// quantize($px, $m, $colors, $method, $kmeans) ->
//     {"px": the index buffer, "palette": the (n, 3) palette array}
def quantize($px, $m, $colors, $method, $kmeans) {
    $n = $colors;
    if ($n == null) { $n = 256; }
    if ($n < 1 || $n > 256) {
        throw "Image.quantize: colors must be 1..256, got " + str($n);
    }
    $meth = $method;
    if ($meth == null) {
        $meth = $MEDIANCUT;
        if ($m == "RGBA" || $m == "RGBa") { $meth = $FASTOCTREE; }
    }
    if ($meth == $LIBIMAGEQUANT) {
        throw "Image.quantize: method 3 is libimagequant, an external C " +
              "library pillow declines to half-implement (decision PW10). " +
              "Use 0 (median cut), 1 (max coverage) or 2 (fast octree).";
    }
    if (($m == "RGBA" || $m == "RGBa") && $meth != $FASTOCTREE) {
        throw "Fast Octree (method == 2) and libimagequant (method == 3) are " +
              "the only valid methods for quantizing RGBA images";
    }
    if ($meth != $MEDIANCUT && $meth != $MAXCOVERAGE && $meth != $FASTOCTREE) {
        throw "Image.quantize: unknown method " + str($meth);
    }

    $d = pxbuf.dims($px);
    $bands = _bands($px, $m);
    $h = _histogram($bands[0], $bands[1], $bands[2]);

    if ($meth == $FASTOCTREE) { $pal = _octree($h, $n); }
    else {
        $boxes = _medianCut($h, $n, $meth == $MAXCOVERAGE);
        $pr = _paletteOf($h, $boxes);
        $pal = $pr[0];
    }
    if (len($pal) == 0) { $pal = [[0, 0, 0]]; }

    // Every pixel to its nearest entry, through the distinct-colour table:
    // searchsorted puts each pixel on its colour, and one gather does the
    // rest. No loop over pixels anywhere.
    $assign = _mapNearest($h, $pal);
    $packed = nd_add(nd_add(nd_multiply($bands[0], 65536),
                            nd_multiply($bands[1], 256)), $bands[2]);
    $pos = nd_searchsorted(nd($h.packed, "i64"), $packed);
    $idx = nd_take($assign, $pos);
    $out = nd_reshape($idx, [$d[0], $d[1], 1]);

    $flat = [];
    $i = 0;
    while ($i < len($pal)) {
        push($flat, $pal[$i][0]);
        push($flat, $pal[$i][1]);
        push($flat, $pal[$i][2]);
        $i = $i + 1;
    }
    return {"px": pxbuf.materialise($out, "i64"),
            "palette": nd_reshape(nd($flat, "i64"), [len($pal), 3]),
            "method": $meth, "colors": len($pal)};
}
