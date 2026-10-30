// ════════════════════════════════════════════════════════════════════════
//  pillow/codecs/jpeg.b — JPEG, read and write.
//
//  The second format that matters most, and the only one in this package
//  whose reference is NOT Pillow's own C: Pillow hands a JPEG to
//  libjpeg-turbo and asks for scanlines, so matching Pillow here means
//  matching jdhuff.c, jdphuff.c, jidctint.c, jdsample.c and jdcolor.c.
//  Everything after the entropy scan lives in ../jpegdct.b, which is
//  written against those files bit for bit; this module is the part that
//  cannot be vectorised -- the marker layer and the Huffman scan.
//
//  READS: baseline (SOF0/SOF1) and progressive (SOF2) DCT, 8-bit, 1-, 3-
//  and 4-component frames, every subsampling (4:4:4, 4:2:2, 4:2:0, 4:1:1
//  and the odd ratios), restart markers, JFIF, Adobe APP14, EXIF, ICC
//  profiles split across APP2 segments, and draft-mode scaled decoding at
//  1/2, 1/4 and 1/8.
//
//  WRITES: baseline and progressive, quality 1-100 with Annex K's tables
//  scaled libjpeg's way, `optimize` (per-image Huffman tables), the three
//  subsamplings, caller-supplied qtables, and restart intervals.
//
//  ─── WHY THE SCAN IS THE SLOW PART, AND WHAT THAT COSTS ──────────────
//
//  A JPEG's entropy-coded scan is a bit-by-bit Huffman walk over every
//  non-zero coefficient. It is sequential by construction: the amplitude
//  bits of one coefficient sit between the Huffman codes of its
//  neighbours, so there is no way to find the next block without decoding
//  the current one. This is the documented exception to rule PW2 -- a
//  codec may loop over bytes of the COMPRESSED stream; it may not loop
//  over pixels of the decoded image -- and the measured cost is published
//  in docs/pillow.md beside `open` rather than left to be discovered.
//
//  Three things keep it as fast as an interpreter allows, all of them
//  copied from inflate.b, which had the same problem:
//
//    * the bit accumulator lives in LOCALS inside one function, because a
//      Bantu call is ~0.3 us and a per-symbol bits() call would pay it on
//      every symbol. bitio.b is the readable reference; this is its
//      inlined twin.
//    * Huffman decoding is a 256-entry direct lookup on the next 8 bits,
//      which is libjpeg's own HUFF_LOOKAHEAD trick. Codes longer than 8
//      bits fall back to the canonical walk.
//    * powers of two come from a table, never from pow(), which is a call.
//
//  ─── WHERE THE COEFFICIENTS LIVE ─────────────────────────────────────
//
//  A BASELINE scan is decoded once, in order, and never read back, so the
//  decoder emits two flat lists -- the flat indices and the values of the
//  NON-ZERO coefficients -- and nd_put scatters them into a zeroed array
//  in chunks. Memory is proportional to the non-zero count, which for a
//  photograph is 5-15% of the coefficients.
//
//  A PROGRESSIVE scan cannot: successive-approximation refinement reads
//  each coefficient back to decide whether to consume a correction bit, so
//  the store has to be randomly readable DURING decoding, and nd_get at
//  ~2.5 us would cost more than the scan. Progressive decoding therefore
//  holds each component's coefficients in a flat Bantu list, which costs a
//  Value per coefficient -- about 100 bytes. That is why there is a
//  coefficient budget, checked from the frame header before the first scan
//  is read (decision PW13): a 4000x3000 progressive JPEG would need ~1.8 GB
//  of list and is refused by name rather than left to swap. Baseline files
//  of any size are unaffected.
//
//  ─── AND WHERE OURS IS DELIBERATELY NOT libjpeg ──────────────────────
//
//  libjpeg accumulates the IDCT's first pass into `int workspace[64]`,
//  which would overflow and wrap on a crafted coefficient block. Our store
//  is i64, so it neither wraps nor saturates: on valid input the two agree
//  exactly, on hostile input ours stays arithmetic. Decision PW11.
//
//  Codecs deal in buffers, never Images -- see codecs/registry.b.
//
//  docs/pillow-architecture.md §6.3, §6.4.
// ════════════════════════════════════════════════════════════════════════

include "../pxbuf.b" as pxbuf;
include "../jpegdct.b" as dct;
include "../jpeg_tables.b" as jt;

// 2^k for k in 0..32, so the scan never calls pow().
$_P2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
        32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304,
        8388608, 16777216, 33554432, 67108864, 134217728, 268435456,
        536870912, 1073741824, 2147483648, 4294967296];

$MAX_PIXELS = 89478485;

// Decision PW13: the progressive coefficient budget. A progressive scan
// needs every coefficient randomly readable, which in Bantu means a list,
// which means ~100 bytes each. 6 M coefficients is ~600 MB of list and is
// already generous; past that the answer is to refuse by name.
$CFG = {"MAX_PROGRESSIVE_COEFFS": 6000000};
def max_progressive_coeffs()       { return $CFG.MAX_PROGRESSIVE_COEFFS; }
def set_max_progressive_coeffs($n) { $CFG.MAX_PROGRESSIVE_COEFFS = $n; return $n; }

// The inverse of jt.ZIGZAG: _INV[natural] = the zigzag position holding it.
// The scan stores coefficients in ZIGZAG order -- that is the order they
// arrive in and the order Ss/Se name -- and one nd_take along axis 1
// de-zigzags the whole component at the end. Per-coefficient de-zigzagging
// in the scan would be a list read per coefficient for no gain.
$_INV = null;
def _inv() {
    if ($_INV == null) {
        $t = nd_to_list(nd_zeros([64], "i64"));
        $k = 0;
        while ($k < 64) { $t[jt.ZIGZAG[$k]] = $k; $k = $k + 1; }
        $_INV = nd($t, "i64");
    }
    return $_INV;
}

def sniff($head) {
    if (len($head) < 3) { return false; }
    return ord($head[0]) == 255 && ord($head[1]) == 216 && ord($head[2]) == 255;
}

// ─── Huffman tables ─────────────────────────────────────────────────────
//
// _buildHuff($bits, $vals) -> the decoding table, built the way jdhuff.c
// builds it: the canonical code assignment of T.81 Annex C, plus an
// 8-bit lookahead table for the codes that fit in one peek.
//
//   fsym[b] / flen[b]  for the 256 possible next bytes: the symbol and its
//                      code length, or flen 0 if the code is longer than 8
//   maxcode[l]         the largest length-l code, or -1 if there are none
//   valoff[l]          vals index of the first length-l symbol, minus the
//                      smallest length-l code, so a symbol is one add
def _buildHuff($bits, $vals, $what) {
    $n = 0;
    $l = 1;
    while ($l <= 16) {
        if ($bits[$l] < 0) { throw "JPEG: negative code count in " + $what; }
        $n = $n + $bits[$l];
        $l = $l + 1;
    }
    if ($n > 256 || $n == 0) {
        throw "JPEG: " + $what + " declares " + str($n) + " codes (1..256 expected)";
    }
    if (len($vals) < $n) {
        throw "JPEG: " + $what + " is truncated (" + str(len($vals)) +
              " symbols for " + str($n) + " codes)";
    }

    $fsym = nd_to_list(nd_zeros([256], "i64"));
    $flen = nd_to_list(nd_zeros([256], "i64"));
    $maxcode = nd_to_list(nd_zeros([18], "i64"));
    $valoff = nd_to_list(nd_zeros([18], "i64"));

    $code = 0;
    $k = 0;
    $l = 1;
    while ($l <= 16) {
        $cnt = $bits[$l];
        $valoff[$l] = $k - $code;
        if ($cnt > 0) {
            // The fast table: every byte whose top l bits are this code.
            if ($l <= 8) {
                $i = 0;
                while ($i < $cnt) {
                    $base = ($code + $i) * $_P2[8 - $l];
                    $j = 0;
                    while ($j < $_P2[8 - $l]) {
                        $fsym[$base + $j] = $vals[$k + $i];
                        $flen[$base + $j] = $l;
                        $j = $j + 1;
                    }
                    $i = $i + 1;
                }
            }
            $code = $code + $cnt;
            $k = $k + $cnt;
            $maxcode[$l] = $code - 1;
        } else {
            $maxcode[$l] = 0 - 1;
        }
        if ($code > $_P2[$l]) {
            throw "JPEG: " + $what + " is over-subscribed at code length " + str($l);
        }
        $code = $code * 2;
        $l = $l + 1;
    }
    $maxcode[17] = 1048576;          // a sentinel the walk can never pass
    return {"fsym": $fsym, "flen": $flen, "maxcode": $maxcode,
            "valoff": $valoff, "vals": $vals, "n": $n};
}

// ─── Byte helpers ───────────────────────────────────────────────────────

def _u16($b, $i) { return $b[$i] * 256 + $b[$i + 1]; }

// ─── Two interpreter facts the scan is shaped around ────────────────────
//
// Measured on this build, and they decide the shape of every function
// below far more than the format does:
//
//   * a LIST IS COPIED when it is passed to a Bantu function or read out
//     of a dict field. `$state.table[$i]` is therefore O(len(table)) PER
//     READ -- 500 us on a 20,000-entry list -- and a helper taking the
//     file's byte list copies the file on every call.
//   * a list read, a string index and an `ord` are each ~0.5 us, and
//     builtins take their arguments without copying.
//
// So: the scan is one function; every table it reads in the inner loop is
// a FLAT LOCAL list with per-component base offsets computed outside the
// loop; and no helper is called per byte, per coefficient or per block.
// Ugly, and the alternative is a decoder 100x slower.

// _tabsFor($sc, $fr) -> the scan's Huffman tables concatenated into flat
// lists, so the inner loop reads `$FSYM[$tbase + $look]` instead of
// fetching a 256-entry list out of a dict.
//
// Table t = component index within the scan * 2, + 1 for its AC table.
// FSYM/FLEN/VALS are 256 entries per table, MAXC/VOFF 18.
def _tabsFor($sc, $fr) {
    $fsym = [];
    $flen = [];
    $maxc = [];
    $voff = [];
    $vals = [];
    $i = 0;
    $ncs = len($sc.comps);
    while ($i < $ncs) {
        $sci = $sc.comps[$i];
        $two = [$sci.dc, $sci.ac];
        $w = 0;
        while ($w < 2) {
            $t = $two[$w];
            $k = 0;
            if ($t == null) {
                // A scan that codes no DC (Ss > 0) needs no DC table, and
                // a DC-only scan needs no AC table. Pad so the offsets of
                // the tables that DO exist stay where they are.
                while ($k < 256) { push($fsym, 0); push($flen, 0); push($vals, 0); $k = $k + 1; }
                $k = 0;
                while ($k < 18) { push($maxc, 0 - 1); push($voff, 0); $k = $k + 1; }
            } else {
                $tf = $t.fsym;
                $tl = $t.flen;
                $tv = $t.vals;
                $nv = len($tv);
                while ($k < 256) {
                    push($fsym, $tf[$k]);
                    push($flen, $tl[$k]);
                    if ($k < $nv) { push($vals, $tv[$k]); } else { push($vals, 0); }
                    $k = $k + 1;
                }
                $tm = $t.maxcode;
                $to = $t.valoff;
                $k = 0;
                while ($k < 18) { push($maxc, $tm[$k]); push($voff, $to[$k]); $k = $k + 1; }
            }
            $w = $w + 1;
        }
        $i = $i + 1;
    }
    return {"fsym": $fsym, "flen": $flen, "maxc": $maxc, "voff": $voff, "vals": $vals};
}

// _geomFor($sc, $fr) -> the scan's per-component geometry as flat lists.
def _geomFor($sc, $fr) {
    $wb = [];
    $h = [];
    $v = [];
    $off = [];
    $i = 0;
    while ($i < len($sc.comps)) {
        $c = $fr.comps[$sc.comps[$i].ci];
        push($wb, $c.wb);
        push($h, $c.h);
        push($v, $c.v);
        push($off, $c.off);
        $i = $i + 1;
    }
    return {"wb": $wb, "h": $h, "v": $v, "off": $off};
}

// _nextMarker($b, $pos, $srclen) -> the offset of the next 0xFF that
// begins a real marker: stuffed 0xFF00, fill bytes and restart markers are
// part of the entropy stream and are skipped.
//
// $b is a LOCAL byte list in the caller, so this copies it -- which is why
// it is only ever called once per scan, at the end.
def _nextMarker($b, $pos, $srclen) {
    $k = $pos;
    while ($k + 1 < $srclen) {
        if ($b[$k] == 255) {
            $m = $b[$k + 1];
            if ($m != 0 && $m != 255 && ($m < 208 || $m > 215)) { return $k; }
        }
        $k = $k + 1;
    }
    return $srclen;
}

// ════════════════════════════════════════════════════════════════════════
//  The baseline scan
// ════════════════════════════════════════════════════════════════════════
//
// $acc holds $n bits, MSB-first, as a number below 2^n -- 24 bits at most,
// which keeps every intermediate exact in a float64. The non-zero
// coefficients go into two flat lists and are scattered into the frame's
// one coefficient array every $_CHUNK of them, so memory stays
// proportional to the chunk and not to the image (architecture §6.4.5).
//
// Returns [the offset of the marker that ends the scan].
$_CHUNK = 65536;

def _scanBaseline($b, $pos, $fr, $sc) {
    $srclen = len($b);
    $P2 = $_P2;
    $acc = 0;
    $n = 0;
    $ncs = len($sc.comps);
    $ri = $fr.restart;
    $coef = $fr.coef;

    $T = _tabsFor($sc, $fr);
    $FSYM = $T.fsym;
    $FLEN = $T.flen;
    $MAXC = $T.maxc;
    $VOFF = $T.voff;
    $VALS = $T.vals;
    $G = _geomFor($sc, $fr);
    $CWB = $G.wb;
    $CH = $G.h;
    $CV = $G.v;
    $COFF = $G.off;

    $preds = nd_to_list(nd_zeros([$ncs], "i64"));
    $idx = [];
    $val = [];
    $nidx = 0;

    if ($ncs == 1) {
        $c0 = $fr.comps[$sc.comps[0].ci];
        $mcus = $c0.bw * $c0.bh;
        $perRow = $c0.bw;
        if ($perRow == 0) { $perRow = 1; }
    } else {
        $mcus = $fr.mcusPerRow * $fr.mcuRows;
        $perRow = $fr.mcusPerRow;
    }
    if ($mcus == 0) { return _nextMarker($b, $pos, $srclen); }

    $todo = $mcus + 1;
    if ($ri > 0) { $todo = $ri; }

    $mcu = 0;
    while ($mcu < $mcus) {
        if ($todo == 0) {
            // ── a restart marker is due: byte-align and consume it ──
            $acc = 0;
            $n = 0;
            $k = $pos;
            while ($k + 1 < $srclen) {
                if ($b[$k] == 255) {
                    $m = $b[$k + 1];
                    if ($m >= 208 && $m <= 215) { break; }
                    if ($m != 0 && $m != 255) { break; }
                }
                $k = $k + 1;
            }
            if ($k + 1 < $srclen && $b[$k] == 255 && $b[$k + 1] >= 208 && $b[$k + 1] <= 215) {
                $pos = $k + 2;
            } else {
                // No restart where one was due. libjpeg warns and carries
                // on with the predictors reset, which is what a damaged or
                // truncated file needs; the refill pads with zero bits.
                $pos = $k;
            }
            $i = 0;
            while ($i < $ncs) { $preds[$i] = 0; $i = $i + 1; }
            $todo = $ri;
        }
        $todo = $todo - 1;

        $ci = 0;
        while ($ci < $ncs) {
            $dcb = $ci * 512;
            $acb = $dcb + 256;
            $dcw = $ci * 36;
            $acw = $dcw + 18;
            $wb = $CWB[$ci];
            $cbase = $COFF[$ci];

            if ($ncs == 1) {
                $nbv = 1;
                $nbh = 1;
                $brow = floor($mcu / $perRow);
                $bcol = $mcu % $perRow;
            } else {
                $nbv = $CV[$ci];
                $nbh = $CH[$ci];
                $brow = floor($mcu / $perRow) * $nbv;
                $bcol = ($mcu % $perRow) * $nbh;
            }

            $bv = 0;
            while ($bv < $nbv) {
                $rowbase = $cbase + ($brow + $bv) * $wb * 64;
                $bh = 0;
                while ($bh < $nbh) {
                    $base = $rowbase + ($bcol + $bh) * 64;

                    // ── DC: one Huffman symbol, then that many magnitude bits ──
                    while ($n < 16) {
                        $cb = 0;
                        if ($pos < $srclen) {
                            $cb = $b[$pos];
                            if ($cb == 255) {
                                $c2 = 0;
                                if ($pos + 1 < $srclen) { $c2 = $b[$pos + 1]; }
                                if ($c2 == 0) { $pos = $pos + 2; } else { $cb = 0; }
                            } else { $pos = $pos + 1; }
                        }
                        $acc = $acc * 256 + $cb;
                        $n = $n + 8;
                    }
                    $look = floor($acc / $P2[$n - 8]) % 256;
                    $cl = $FLEN[$dcb + $look];
                    if ($cl > 0) {
                        $s = $FSYM[$dcb + $look];
                        $n = $n - $cl;
                        $acc = $acc % $P2[$n];
                    } else {
                        $c16 = floor($acc / $P2[$n - 16]);
                        $cl = 9;
                        $code = 0;
                        while ($cl <= 16) {
                            $code = floor($c16 / $P2[16 - $cl]);
                            if ($code <= $MAXC[$dcw + $cl]) { break; }
                            $cl = $cl + 1;
                        }
                        if ($cl > 16) {
                            throw "JPEG: a DC Huffman code longer than 16 bits at byte " + str($pos);
                        }
                        $s = $VALS[$dcb + $code + $VOFF[$dcw + $cl]];
                        $n = $n - $cl;
                        $acc = $acc % $P2[$n];
                    }
                    if ($s > 0) {
                        if ($s > 16) { throw "JPEG: a DC magnitude of " + str($s) + " bits"; }
                        while ($n < $s) {
                            $cb = 0;
                            if ($pos < $srclen) {
                                $cb = $b[$pos];
                                if ($cb == 255) {
                                    $c2 = 0;
                                    if ($pos + 1 < $srclen) { $c2 = $b[$pos + 1]; }
                                    if ($c2 == 0) { $pos = $pos + 2; } else { $cb = 0; }
                                } else { $pos = $pos + 1; }
                            }
                            $acc = $acc * 256 + $cb;
                            $n = $n + 8;
                        }
                        $v = floor($acc / $P2[$n - $s]);
                        $n = $n - $s;
                        $acc = $acc % $P2[$n];
                        if ($v < $P2[$s - 1]) { $v = $v - $P2[$s] + 1; }
                        $preds[$ci] = $preds[$ci] + $v;
                    }
                    if ($preds[$ci] != 0) {
                        push($idx, $base);
                        push($val, $preds[$ci]);
                        $nidx = $nidx + 1;
                    }

                    // ── AC: run/size pairs until coefficient 63 or an EOB ──
                    $k = 1;
                    while ($k < 64) {
                        while ($n < 16) {
                            $cb = 0;
                            if ($pos < $srclen) {
                                $cb = $b[$pos];
                                if ($cb == 255) {
                                    $c2 = 0;
                                    if ($pos + 1 < $srclen) { $c2 = $b[$pos + 1]; }
                                    if ($c2 == 0) { $pos = $pos + 2; } else { $cb = 0; }
                                } else { $pos = $pos + 1; }
                            }
                            $acc = $acc * 256 + $cb;
                            $n = $n + 8;
                        }
                        $look = floor($acc / $P2[$n - 8]) % 256;
                        $cl = $FLEN[$acb + $look];
                        if ($cl > 0) {
                            $rs = $FSYM[$acb + $look];
                            $n = $n - $cl;
                            $acc = $acc % $P2[$n];
                        } else {
                            $c16 = floor($acc / $P2[$n - 16]);
                            $cl = 9;
                            $code = 0;
                            while ($cl <= 16) {
                                $code = floor($c16 / $P2[16 - $cl]);
                                if ($code <= $MAXC[$acw + $cl]) { break; }
                                $cl = $cl + 1;
                            }
                            if ($cl > 16) {
                                throw "JPEG: an AC Huffman code longer than 16 bits at byte " + str($pos);
                            }
                            $rs = $VALS[$acb + $code + $VOFF[$acw + $cl]];
                            $n = $n - $cl;
                            $acc = $acc % $P2[$n];
                        }
                        $s = $rs % 16;
                        $r = ($rs - $s) / 16;
                        if ($s == 0) {
                            if ($r != 15) { break; }
                            $k = $k + 16;
                        } else {
                            $k = $k + $r;
                            if ($k > 63) { break; }
                            while ($n < $s) {
                                $cb = 0;
                                if ($pos < $srclen) {
                                    $cb = $b[$pos];
                                    if ($cb == 255) {
                                        $c2 = 0;
                                        if ($pos + 1 < $srclen) { $c2 = $b[$pos + 1]; }
                                        if ($c2 == 0) { $pos = $pos + 2; } else { $cb = 0; }
                                    } else { $pos = $pos + 1; }
                                }
                                $acc = $acc * 256 + $cb;
                                $n = $n + 8;
                            }
                            $v = floor($acc / $P2[$n - $s]);
                            $n = $n - $s;
                            $acc = $acc % $P2[$n];
                            if ($v < $P2[$s - 1]) { $v = $v - $P2[$s] + 1; }
                            push($idx, $base + $k);
                            push($val, $v);
                            $nidx = $nidx + 1;
                            $k = $k + 1;
                        }
                    }
                    if ($nidx >= $_CHUNK) {
                        nd_put($coef, nd($idx, "i64"), nd($val, "i64"));
                        $idx = [];
                        $val = [];
                        $nidx = 0;
                    }
                    $bh = $bh + 1;
                }
                $bv = $bv + 1;
            }
            $ci = $ci + 1;
        }
        $mcu = $mcu + 1;
    }

    if ($nidx > 0) { nd_put($coef, nd($idx, "i64"), nd($val, "i64")); }
    return _nextMarker($b, $pos, $srclen);
}

// ════════════════════════════════════════════════════════════════════════
//  The progressive scans
// ════════════════════════════════════════════════════════════════════════
//
// jdphuff.c, four entry points: DC first, DC refine, AC first, AC refine.
// Successive approximation reads a coefficient back to decide whether to
// consume a correction bit, so the store is a flat Bantu LIST -- copied in
// once at the top of the scan and back once at the bottom, because a dict
// field read copies it on every access (see the note above _tabsFor).
//
// Coefficients are held in ZIGZAG order, which is the order Ss and Se name
// them, so libjpeg's natural_order[k] indirection is not needed here; the
// whole component is de-zigzagged by one nd_take at the end.
def _scanProgressive($b, $pos, $fr, $sc) {
    $srclen = len($b);
    $P2 = $_P2;
    $acc = 0;
    $n = 0;
    $ncs = len($sc.comps);
    $ri = $fr.restart;
    $ss = $sc.ss;
    $se = $sc.se;
    $ah = $sc.ah;
    $al = $sc.al;

    $T = _tabsFor($sc, $fr);
    $FSYM = $T.fsym;
    $FLEN = $T.flen;
    $MAXC = $T.maxc;
    $VOFF = $T.voff;
    $VALS = $T.vals;
    $G = _geomFor($sc, $fr);
    $CWB = $G.wb;
    $CH = $G.h;
    $CV = $G.v;
    $COFF = $G.off;

    $st = $fr.plist;                 // one copy in
    $preds = nd_to_list(nd_zeros([$ncs], "i64"));
    $eobrun = 0;

    if ($ss == 0) {
        // ── a DC scan: interleaved over every component in the scan ──
        if ($ncs == 1) {
            $c0 = $fr.comps[$sc.comps[0].ci];
            $mcus = $c0.bw * $c0.bh;
            $perRow = $c0.bw;
            if ($perRow == 0) { $perRow = 1; }
        } else {
            $mcus = $fr.mcusPerRow * $fr.mcuRows;
            $perRow = $fr.mcusPerRow;
        }
        $todo = $mcus + 1;
        if ($ri > 0) { $todo = $ri; }
        $p1 = $P2[$al];

        $mcu = 0;
        while ($mcu < $mcus) {
            if ($todo == 0) {
                $acc = 0;
                $n = 0;
                $k = $pos;
                while ($k + 1 < $srclen) {
                    if ($b[$k] == 255) {
                        $m = $b[$k + 1];
                        if ($m >= 208 && $m <= 215) { break; }
                        if ($m != 0 && $m != 255) { break; }
                    }
                    $k = $k + 1;
                }
                if ($k + 1 < $srclen && $b[$k] == 255 && $b[$k + 1] >= 208 && $b[$k + 1] <= 215) {
                    $pos = $k + 2;
                } else { $pos = $k; }
                $i = 0;
                while ($i < $ncs) { $preds[$i] = 0; $i = $i + 1; }
                $eobrun = 0;
                $todo = $ri;
            }
            $todo = $todo - 1;

            $ci = 0;
            while ($ci < $ncs) {
                $dcb = $ci * 512;
                $dcw = $ci * 36;
                $wb = $CWB[$ci];
                $cbase = $COFF[$ci];
                if ($ncs == 1) {
                    $nbv = 1;
                    $nbh = 1;
                    $brow = floor($mcu / $perRow);
                    $bcol = $mcu % $perRow;
                } else {
                    $nbv = $CV[$ci];
                    $nbh = $CH[$ci];
                    $brow = floor($mcu / $perRow) * $nbv;
                    $bcol = ($mcu % $perRow) * $nbh;
                }
                $bv = 0;
                while ($bv < $nbv) {
                    $rowbase = $cbase + ($brow + $bv) * $wb * 64;
                    $bh = 0;
                    while ($bh < $nbh) {
                        $base = $rowbase + ($bcol + $bh) * 64;
                        if ($ah == 0) {
                            // ── DC first: a difference, shifted left by Al ──
                            while ($n < 16) {
                                $cb = 0;
                                if ($pos < $srclen) {
                                    $cb = $b[$pos];
                                    if ($cb == 255) {
                                        $c2 = 0;
                                        if ($pos + 1 < $srclen) { $c2 = $b[$pos + 1]; }
                                        if ($c2 == 0) { $pos = $pos + 2; } else { $cb = 0; }
                                    } else { $pos = $pos + 1; }
                                }
                                $acc = $acc * 256 + $cb;
                                $n = $n + 8;
                            }
                            $look = floor($acc / $P2[$n - 8]) % 256;
                            $cl = $FLEN[$dcb + $look];
                            if ($cl > 0) {
                                $s = $FSYM[$dcb + $look];
                                $n = $n - $cl;
                                $acc = $acc % $P2[$n];
                            } else {
                                $c16 = floor($acc / $P2[$n - 16]);
                                $cl = 9;
                                $code = 0;
                                while ($cl <= 16) {
                                    $code = floor($c16 / $P2[16 - $cl]);
                                    if ($code <= $MAXC[$dcw + $cl]) { break; }
                                    $cl = $cl + 1;
                                }
                                if ($cl > 16) {
                                    throw "JPEG: a DC Huffman code longer than 16 bits at byte " + str($pos);
                                }
                                $s = $VALS[$dcb + $code + $VOFF[$dcw + $cl]];
                                $n = $n - $cl;
                                $acc = $acc % $P2[$n];
                            }
                            if ($s > 0) {
                                if ($s > 16) { throw "JPEG: a DC magnitude of " + str($s) + " bits"; }
                                while ($n < $s) {
                                    $cb = 0;
                                    if ($pos < $srclen) {
                                        $cb = $b[$pos];
                                        if ($cb == 255) {
                                            $c2 = 0;
                                            if ($pos + 1 < $srclen) { $c2 = $b[$pos + 1]; }
                                            if ($c2 == 0) { $pos = $pos + 2; } else { $cb = 0; }
                                        } else { $pos = $pos + 1; }
                                    }
                                    $acc = $acc * 256 + $cb;
                                    $n = $n + 8;
                                }
                                $v = floor($acc / $P2[$n - $s]);
                                $n = $n - $s;
                                $acc = $acc % $P2[$n];
                                if ($v < $P2[$s - 1]) { $v = $v - $P2[$s] + 1; }
                                $preds[$ci] = $preds[$ci] + $v;
                            }
                            $st[$base] = $preds[$ci] * $p1;
                        } else {
                            // ── DC refine: one correction bit ──
                            if ($n < 1) {
                                $cb = 0;
                                if ($pos < $srclen) {
                                    $cb = $b[$pos];
                                    if ($cb == 255) {
                                        $c2 = 0;
                                        if ($pos + 1 < $srclen) { $c2 = $b[$pos + 1]; }
                                        if ($c2 == 0) { $pos = $pos + 2; } else { $cb = 0; }
                                    } else { $pos = $pos + 1; }
                                }
                                $acc = $acc * 256 + $cb;
                                $n = $n + 8;
                            }
                            $n = $n - 1;
                            $bit = floor($acc / $P2[$n]);
                            $acc = $acc % $P2[$n];
                            if ($bit == 1) {
                                $cur = $st[$base];
                                if ($cur < 0) {
                                    $u = $cur + 4294967296;
                                    $u = bor($u, $p1);
                                    $st[$base] = $u - 4294967296;
                                } else {
                                    $st[$base] = bor($cur, $p1);
                                }
                            }
                        }
                        $bh = $bh + 1;
                    }
                    $bv = $bv + 1;
                }
                $ci = $ci + 1;
            }
            $mcu = $mcu + 1;
        }
    } else {
        // ── an AC scan: always exactly one component (T.81 G.1.1.1.2) ──
        if ($ncs != 1) {
            throw "JPEG: an AC progressive scan names " + str($ncs) +
                  " components; the spec allows exactly one";
        }
        $c0 = $fr.comps[$sc.comps[0].ci];
        $wb = $c0.wb;
        $cbase = $c0.off;
        $bw = $c0.bw;
        $blocks = $bw * $c0.bh;
        $acb = 256;
        $acw = 18;
        $p1 = $P2[$al];
        $m1 = 0 - $P2[$al];
        $todo = $blocks + 1;
        if ($ri > 0) { $todo = $ri; }

        $blk = 0;
        while ($blk < $blocks) {
            if ($todo == 0) {
                $acc = 0;
                $n = 0;
                $k = $pos;
                while ($k + 1 < $srclen) {
                    if ($b[$k] == 255) {
                        $m = $b[$k + 1];
                        if ($m >= 208 && $m <= 215) { break; }
                        if ($m != 0 && $m != 255) { break; }
                    }
                    $k = $k + 1;
                }
                if ($k + 1 < $srclen && $b[$k] == 255 && $b[$k + 1] >= 208 && $b[$k + 1] <= 215) {
                    $pos = $k + 2;
                } else { $pos = $k; }
                $eobrun = 0;
                $todo = $ri;
            }
            $todo = $todo - 1;

            $base = $cbase + (floor($blk / $bw) * $wb + ($blk % $bw)) * 64;

            if ($ah == 0) {
                // ── AC first: new coefficients, shifted left by Al ──
                if ($eobrun > 0) {
                    $eobrun = $eobrun - 1;
                } else {
                    $k = $ss;
                    while ($k <= $se) {
                        while ($n < 16) {
                            $cb = 0;
                            if ($pos < $srclen) {
                                $cb = $b[$pos];
                                if ($cb == 255) {
                                    $c2 = 0;
                                    if ($pos + 1 < $srclen) { $c2 = $b[$pos + 1]; }
                                    if ($c2 == 0) { $pos = $pos + 2; } else { $cb = 0; }
                                } else { $pos = $pos + 1; }
                            }
                            $acc = $acc * 256 + $cb;
                            $n = $n + 8;
                        }
                        $look = floor($acc / $P2[$n - 8]) % 256;
                        $cl = $FLEN[$acb + $look];
                        if ($cl > 0) {
                            $rs = $FSYM[$acb + $look];
                            $n = $n - $cl;
                            $acc = $acc % $P2[$n];
                        } else {
                            $c16 = floor($acc / $P2[$n - 16]);
                            $cl = 9;
                            $code = 0;
                            while ($cl <= 16) {
                                $code = floor($c16 / $P2[16 - $cl]);
                                if ($code <= $MAXC[$acw + $cl]) { break; }
                                $cl = $cl + 1;
                            }
                            if ($cl > 16) {
                                throw "JPEG: an AC Huffman code longer than 16 bits at byte " + str($pos);
                            }
                            $rs = $VALS[$acb + $code + $VOFF[$acw + $cl]];
                            $n = $n - $cl;
                            $acc = $acc % $P2[$n];
                        }
                        $s = $rs % 16;
                        $r = ($rs - $s) / 16;
                        if ($s > 0) {
                            $k = $k + $r;
                            if ($k > $se) { break; }
                            while ($n < $s) {
                                $cb = 0;
                                if ($pos < $srclen) {
                                    $cb = $b[$pos];
                                    if ($cb == 255) {
                                        $c2 = 0;
                                        if ($pos + 1 < $srclen) { $c2 = $b[$pos + 1]; }
                                        if ($c2 == 0) { $pos = $pos + 2; } else { $cb = 0; }
                                    } else { $pos = $pos + 1; }
                                }
                                $acc = $acc * 256 + $cb;
                                $n = $n + 8;
                            }
                            $v = floor($acc / $P2[$n - $s]);
                            $n = $n - $s;
                            $acc = $acc % $P2[$n];
                            if ($v < $P2[$s - 1]) { $v = $v - $P2[$s] + 1; }
                            $st[$base + $k] = $v * $p1;
                        } else {
                            if ($r != 15) {
                                $eobrun = $P2[$r];
                                if ($r > 0) {
                                    while ($n < $r) {
                                        $cb = 0;
                                        if ($pos < $srclen) {
                                            $cb = $b[$pos];
                                            if ($cb == 255) {
                                                $c2 = 0;
                                                if ($pos + 1 < $srclen) { $c2 = $b[$pos + 1]; }
                                                if ($c2 == 0) { $pos = $pos + 2; } else { $cb = 0; }
                                            } else { $pos = $pos + 1; }
                                        }
                                        $acc = $acc * 256 + $cb;
                                        $n = $n + 8;
                                    }
                                    $v = floor($acc / $P2[$n - $r]);
                                    $n = $n - $r;
                                    $acc = $acc % $P2[$n];
                                    $eobrun = $eobrun + $v;
                                }
                                $eobrun = $eobrun - 1;
                                break;
                            }
                            $k = $k + 15;
                        }
                        $k = $k + 1;
                    }
                }
            } else {
                // ── AC refine: correction bits, and newly non-zero ones ──
                $k = $ss;
                if ($eobrun == 0) {
                    while ($k <= $se) {
                        while ($n < 16) {
                            $cb = 0;
                            if ($pos < $srclen) {
                                $cb = $b[$pos];
                                if ($cb == 255) {
                                    $c2 = 0;
                                    if ($pos + 1 < $srclen) { $c2 = $b[$pos + 1]; }
                                    if ($c2 == 0) { $pos = $pos + 2; } else { $cb = 0; }
                                } else { $pos = $pos + 1; }
                            }
                            $acc = $acc * 256 + $cb;
                            $n = $n + 8;
                        }
                        $look = floor($acc / $P2[$n - 8]) % 256;
                        $cl = $FLEN[$acb + $look];
                        if ($cl > 0) {
                            $rs = $FSYM[$acb + $look];
                            $n = $n - $cl;
                            $acc = $acc % $P2[$n];
                        } else {
                            $c16 = floor($acc / $P2[$n - 16]);
                            $cl = 9;
                            $code = 0;
                            while ($cl <= 16) {
                                $code = floor($c16 / $P2[16 - $cl]);
                                if ($code <= $MAXC[$acw + $cl]) { break; }
                                $cl = $cl + 1;
                            }
                            if ($cl > 16) {
                                throw "JPEG: an AC Huffman code longer than 16 bits at byte " + str($pos);
                            }
                            $rs = $VALS[$acb + $code + $VOFF[$acw + $cl]];
                            $n = $n - $cl;
                            $acc = $acc % $P2[$n];
                        }
                        $s = $rs % 16;
                        $r = ($rs - $s) / 16;
                        $newval = 0;
                        if ($s > 0) {
                            // The size of a newly non-zero coefficient is
                            // always 1 in a refinement scan; the bit says
                            // which sign.
                            if ($n < 1) {
                                $cb = 0;
                                if ($pos < $srclen) {
                                    $cb = $b[$pos];
                                    if ($cb == 255) {
                                        $c2 = 0;
                                        if ($pos + 1 < $srclen) { $c2 = $b[$pos + 1]; }
                                        if ($c2 == 0) { $pos = $pos + 2; } else { $cb = 0; }
                                    } else { $pos = $pos + 1; }
                                }
                                $acc = $acc * 256 + $cb;
                                $n = $n + 8;
                            }
                            $n = $n - 1;
                            $bit = floor($acc / $P2[$n]);
                            $acc = $acc % $P2[$n];
                            if ($bit == 1) { $newval = $p1; } else { $newval = $m1; }
                        } else {
                            if ($r != 15) {
                                $eobrun = $P2[$r];
                                if ($r > 0) {
                                    while ($n < $r) {
                                        $cb = 0;
                                        if ($pos < $srclen) {
                                            $cb = $b[$pos];
                                            if ($cb == 255) {
                                                $c2 = 0;
                                                if ($pos + 1 < $srclen) { $c2 = $b[$pos + 1]; }
                                                if ($c2 == 0) { $pos = $pos + 2; } else { $cb = 0; }
                                            } else { $pos = $pos + 1; }
                                        }
                                        $acc = $acc * 256 + $cb;
                                        $n = $n + 8;
                                    }
                                    $v = floor($acc / $P2[$n - $r]);
                                    $n = $n - $r;
                                    $acc = $acc % $P2[$n];
                                    $eobrun = $eobrun + $v;
                                }
                                break;
                            }
                        }
                        // Advance over the coefficients that are already
                        // non-zero, appending a correction bit to each, and
                        // over r still-zero ones.
                        while ($k <= $se) {
                            $cur = $st[$base + $k];
                            if ($cur != 0) {
                                if ($n < 1) {
                                    $cb = 0;
                                    if ($pos < $srclen) {
                                        $cb = $b[$pos];
                                        if ($cb == 255) {
                                            $c2 = 0;
                                            if ($pos + 1 < $srclen) { $c2 = $b[$pos + 1]; }
                                            if ($c2 == 0) { $pos = $pos + 2; } else { $cb = 0; }
                                        } else { $pos = $pos + 1; }
                                    }
                                    $acc = $acc * 256 + $cb;
                                    $n = $n + 8;
                                }
                                $n = $n - 1;
                                $bit = floor($acc / $P2[$n]);
                                $acc = $acc % $P2[$n];
                                if ($bit == 1) {
                                    if (band($cur, $p1) == 0) {
                                        if ($cur >= 0) { $st[$base + $k] = $cur + $p1; }
                                        else { $st[$base + $k] = $cur + $m1; }
                                    }
                                }
                            } else {
                                $r = $r - 1;
                                if ($r < 0) { break; }
                            }
                            $k = $k + 1;
                        }
                        if ($newval != 0 && $k <= $se) { $st[$base + $k] = $newval; }
                        $k = $k + 1;
                    }
                }
                if ($eobrun > 0) {
                    // Past the end of the band: a correction bit for every
                    // remaining non-zero coefficient, and nothing else.
                    while ($k <= $se) {
                        $cur = $st[$base + $k];
                        if ($cur != 0) {
                            if ($n < 1) {
                                $cb = 0;
                                if ($pos < $srclen) {
                                    $cb = $b[$pos];
                                    if ($cb == 255) {
                                        $c2 = 0;
                                        if ($pos + 1 < $srclen) { $c2 = $b[$pos + 1]; }
                                        if ($c2 == 0) { $pos = $pos + 2; } else { $cb = 0; }
                                    } else { $pos = $pos + 1; }
                                }
                                $acc = $acc * 256 + $cb;
                                $n = $n + 8;
                            }
                            $n = $n - 1;
                            $bit = floor($acc / $P2[$n]);
                            $acc = $acc % $P2[$n];
                            if ($bit == 1) {
                                if (band($cur, $p1) == 0) {
                                    if ($cur >= 0) { $st[$base + $k] = $cur + $p1; }
                                    else { $st[$base + $k] = $cur + $m1; }
                                }
                            }
                        }
                        $k = $k + 1;
                    }
                    $eobrun = $eobrun - 1;
                }
            }
            $blk = $blk + 1;
        }
    }

    $fr.plist = $st;                 // and one copy out
    return _nextMarker($b, $pos, $srclen);
}

// ════════════════════════════════════════════════════════════════════════
//  The marker layer
// ════════════════════════════════════════════════════════════════════════

def _ceilDiv($a, $b) { return floor(($a + $b - 1) / $b); }

// _colourSpace($fr) -> "GRAY", "YCbCr", "RGB", "CMYK" or "YCCK".
//
// jdinput.c's default_decompress_parms, which is the guess libjpeg makes
// and therefore the guess Pillow's pixels come from: JFIF wins, then
// Adobe's transform byte, then the component IDs, then YCbCr.
def _colourSpace($fr) {
    $nc = len($fr.comps);
    if ($nc == 1) { return "GRAY"; }
    if ($nc == 3) {
        if ($fr.jfif) { return "YCbCr"; }
        if ($fr.adobe != null) {
            if ($fr.adobeTransform == 0) { return "RGB"; }
            return "YCbCr";
        }
        $i0 = $fr.comps[0].id;
        $i1 = $fr.comps[1].id;
        $i2 = $fr.comps[2].id;
        if ($i0 == 82 && $i1 == 71 && $i2 == 66) { return "RGB"; }
        return "YCbCr";
    }
    if ($nc == 4) {
        if ($fr.adobe != null && $fr.adobeTransform == 0) { return "CMYK"; }
        if ($fr.adobe != null) { return "YCCK"; }
        return "CMYK";
    }
    throw "JPEG: cannot handle a " + str($nc) + "-component frame";
}

// _assemble($fr, $denom, $want) -> {"mode", "size", "px"}.
//
// Everything after the scan, and all of it array work: de-zigzag,
// dequantise, inverse DCT, blocks-to-plane, upsample, colour transform.
// The IDCT size per component is libjpeg's own choice
// (jpeg_calc_output_dimensions): a subsampled component gets a LARGER
// IDCT so the upsampler has less to do, which is the detail that decides
// the whole shape of draft mode -- architecture §6.4.4.
def _assemble($fr, $denom, $want) {
    $minDCT = floor(8 / $denom);
    $outW = _ceilDiv($fr.w * $minDCT, 8);
    $outH = _ceilDiv($fr.h * $minDCT, 8);
    $fancy = $minDCT > 1;

    $planes = [];
    $ci = 0;
    while ($ci < len($fr.comps)) {
        $c = $fr.comps[$ci];

        // The IDCT size libjpeg would pick for this component.
        $ssize = $minDCT;
        while ($ssize < 8 &&
               $c.h * $ssize * 2 <= $fr.maxH * $minDCT &&
               $c.v * $ssize * 2 <= $fr.maxV * $minDCT) {
            $ssize = $ssize * 2;
        }

        $nb = $c.wb * $c.hb;
        $flat = nd_slice($fr.coef, [[$c.off, $c.off + $nb * 64]]);
        $zz = nd_reshape($flat, [$nb, 64]);
        $nat = nd_take($zz, _inv(), 1);
        if ($c.qt == null) {
            throw "JPEG: component " + str($c.id) + " names quantisation table " +
                  str($c.tq) + ", which the file never defines";
        }
        $deq = dct.dequantise($nat, $c.qt);
        $blocks = dct.idct($deq, $ssize);
        $pl = dct.plane($blocks, $c.hb, $c.wb, $ssize);

        // The component's true sample size at this scale, then libjpeg's
        // upsampling ratio: h_in_group vs max_h_samp_factor.
        $cw = _ceilDiv($fr.w * $c.h * $ssize, $fr.maxH * 8);
        $ch = _ceilDiv($fr.h * $c.v * $ssize, $fr.maxV * 8);
        $pl = nd_slice($pl, [[0, $ch], [0, $cw]]);

        $hin = $c.h * $ssize / $minDCT;
        $vin = $c.v * $ssize / $minDCT;
        if ($hin != floor($hin) || $vin != floor($vin) ||
            $fr.maxH % $hin != 0 || $fr.maxV % $vin != 0) {
            throw "JPEG: sampling factors " + str($c.h) + "x" + str($c.v) +
                  " in a " + str($fr.maxH) + "x" + str($fr.maxV) +
                  " frame are not an integer upsampling ratio";
        }
        $hx = $fr.maxH / $hin;
        $vx = $fr.maxV / $vin;
        if ($hx > 1 || $vx > 1) { $pl = dct.upsample($pl, $hx, $vx, $fancy); }

        $ph = nd_shape($pl)[0];
        $pw = nd_shape($pl)[1];
        if ($ph < $outH || $pw < $outW) {
            throw "JPEG: component " + str($c.id) + " yields " + str($pw) + "x" +
                  str($ph) + " samples for a " + str($outW) + "x" + str($outH) + " image";
        }
        push($planes, nd_slice($pl, [[0, $outH], [0, $outW]]));
        $ci = $ci + 1;
    }

    $cs = _colourSpace($fr);
    $i = 0;
    $parts = [];
    // draft(mode, size) may ask a 3-component YCbCr file for L or YCbCr
    // instead of RGB, which in libjpeg is out_color_space and NOT a
    // conversion of the RGB result: L is the Y plane itself.
    if ($want != null && $cs == "YCbCr") {
        if ($want == "L") {
            return {"mode": "L", "size": [$outW, $outH],
                    "px": pxbuf.stack([nd_reshape(pxbuf.materialise2d($planes[0]),
                                                  [$outH, $outW, 1])], "i64")};
        }
        if ($want == "YCbCr") {
            while ($i < 3) {
                push($parts, nd_reshape(pxbuf.materialise2d($planes[$i]), [$outH, $outW, 1]));
                $i = $i + 1;
            }
            return {"mode": "YCbCr", "size": [$outW, $outH], "px": pxbuf.stack($parts, "i64")};
        }
        throw "JPEG: draft cannot ask a YCbCr file for mode " + str($want);
    }
    if ($cs == "GRAY") {
        $mode = "L";
        push($parts, nd_reshape(pxbuf.materialise2d($planes[0]), [$outH, $outW, 1]));
    } else if ($cs == "YCbCr") {
        $mode = "RGB";
        $rgb = dct.ycc_to_rgb($planes[0], $planes[1], $planes[2]);
        while ($i < 3) {
            push($parts, nd_reshape(pxbuf.materialise2d($rgb[$i]), [$outH, $outW, 1]));
            $i = $i + 1;
        }
    } else if ($cs == "RGB") {
        $mode = "RGB";
        while ($i < 3) {
            push($parts, nd_reshape(pxbuf.materialise2d($planes[$i]), [$outH, $outW, 1]));
            $i = $i + 1;
        }
    } else {
        // Four components. Pillow's rawmode for a CMYK JPEG is "CMYK;I",
        // unconditionally -- it assumes Adobe's inverted convention -- so
        // every value libjpeg produces is subtracted from 255 here.
        $mode = "CMYK";
        if ($cs == "YCCK") {
            $cmyk = dct.ycck_to_cmyk($planes[0], $planes[1], $planes[2], $planes[3]);
        } else {
            $cmyk = $planes;
        }
        while ($i < 4) {
            push($parts, nd_reshape(pxbuf.materialise2d(
                nd_subtract(255, nd_clip($cmyk[$i], 0, 255))), [$outH, $outW, 1]));
            $i = $i + 1;
        }
    }
    return {"mode": $mode, "size": [$outW, $outH], "px": pxbuf.stack($parts, "i64")};
}

// decode($data) -> the buffer dict registry.b expects.
def decode($data) { return decode_scaled($data, 1, null); }

// decode_scaled($data, $denom) -> the same, decoded at 1/denom through a
// scaled IDCT. $denom is 1, 2, 4 or 8, which is exactly what Pillow's
// draft() can ask libjpeg for.
//
// NOTE, because the obvious reading of draft() is wrong: scaling happens in
// the IDCT, and the entropy-coded scan is parsed in FULL at every scale.
// libjpeg does the same amount of Huffman work for a 1/8 decode as for a
// full one, because the amplitude bits of every coefficient are interleaved
// into the bitstream. What draft saves is the IDCT, the upsampling, the
// colour transform and the output buffer -- architecture §6.4.6.
def decode_scaled($data, $denom, $want) {
    if ($denom != 1 && $denom != 2 && $denom != 4 && $denom != 8) {
        throw "JPEG: a scale denominator of " + str($denom) +
              "; libjpeg scales by 1, 1/2, 1/4 or 1/8";
    }
    if (!sniff($data)) { throw "JPEG: not a JPEG file (no SOI marker)"; }
    $b = bytes($data);
    $srclen = len($b);

    $qt = [null, null, null, null];
    $hdc = [null, null, null, null];
    $hac = [null, null, null, null];
    $fr = null;
    $restart = 0;
    $info = {};
    $icc = [];
    $iccCount = 0;
    $progressive = false;
    $jfif = false;
    $adobe = null;
    $adobeTransform = 0 - 1;
    $scans = 0;

    $pos = 2;
    $done = false;
    while (!$done) {
        // ── find the next marker ──
        while ($pos < $srclen && $b[$pos] != 255) { $pos = $pos + 1; }
        while ($pos < $srclen && $b[$pos] == 255) { $pos = $pos + 1; }
        if ($pos >= $srclen) { break; }
        $mk = $b[$pos];
        $pos = $pos + 1;

        if ($mk == 217) { break; }                             // EOI
        if ($mk == 216 || $mk == 1 || ($mk >= 208 && $mk <= 215)) { continue; }

        if ($pos + 2 > $srclen) { throw "JPEG: a marker segment runs off the end of the file"; }
        $seglen = $b[$pos] * 256 + $b[$pos + 1];
        if ($seglen < 2) { throw "JPEG: a marker segment of length " + str($seglen); }
        $sp = $pos + 2;
        $send = $pos + $seglen;
        if ($send > $srclen) {
            throw "JPEG: a " + str($seglen) + "-byte segment at byte " + str($pos) +
                  " runs past the end of a " + str($srclen) + "-byte file";
        }

        if ($mk == 219) {
            // ── DQT: one or more quantisation tables ──
            $q = $sp;
            while ($q < $send) {
                $pq = floor($b[$q] / 16);
                $tq = $b[$q] % 16;
                if ($tq > 3) { throw "JPEG: quantisation table id " + str($tq); }
                $q = $q + 1;
                $need = 64;
                if ($pq == 1) { $need = 128; }
                if ($q + $need > $send) { throw "JPEG: a truncated quantisation table"; }
                $t = nd_to_list(nd_zeros([64], "i64"));
                $k = 0;
                while ($k < 64) {
                    if ($pq == 1) { $v = $b[$q + $k * 2] * 256 + $b[$q + $k * 2 + 1]; }
                    else { $v = $b[$q + $k]; }
                    // The file stores the table in zigzag order; every
                    // consumer here wants it in natural order.
                    $t[jt.ZIGZAG[$k]] = $v;
                    $k = $k + 1;
                }
                $qt[$tq] = $t;
                $q = $q + $need;
            }

        } else if ($mk == 196) {
            // ── DHT: one or more Huffman tables ──
            $q = $sp;
            while ($q < $send) {
                $tc = floor($b[$q] / 16);
                $th = $b[$q] % 16;
                if ($th > 3 || $tc > 1) {
                    throw "JPEG: Huffman table class " + str($tc) + " id " + str($th);
                }
                $q = $q + 1;
                if ($q + 16 > $send) { throw "JPEG: a truncated Huffman table"; }
                $bits = [0];
                $tot = 0;
                $k = 0;
                while ($k < 16) {
                    push($bits, $b[$q + $k]);
                    $tot = $tot + $b[$q + $k];
                    $k = $k + 1;
                }
                $q = $q + 16;
                if ($q + $tot > $send) {
                    throw "JPEG: a Huffman table declaring " + str($tot) +
                          " symbols has only " + str($send - $q) + " bytes left";
                }
                $vals = [];
                $k = 0;
                while ($k < $tot) { push($vals, $b[$q + $k]); $k = $k + 1; }
                $q = $q + $tot;
                $what = "the DC table " + str($th);
                if ($tc == 1) { $what = "the AC table " + str($th); }
                $tab = _buildHuff($bits, $vals, $what);
                if ($tc == 0) { $hdc[$th] = $tab; } else { $hac[$th] = $tab; }
            }

        } else if ($mk == 221) {
            // ── DRI ──
            if ($sp + 2 > $send) { throw "JPEG: a truncated DRI segment"; }
            $restart = $b[$sp] * 256 + $b[$sp + 1];

        } else if ($mk == 192 || $mk == 193 || $mk == 194) {
            // ── SOF0 / SOF1 baseline, SOF2 progressive ──
            if ($fr != null) { throw "JPEG: a second SOF marker; one frame per file"; }
            $progressive = $mk == 194;
            if ($sp + 6 > $send) { throw "JPEG: a truncated SOF segment"; }
            $prec = $b[$sp];
            if ($prec != 8) { throw "JPEG: cannot handle " + str($prec) + "-bit layers"; }
            $h = $b[$sp + 1] * 256 + $b[$sp + 2];
            $w = $b[$sp + 3] * 256 + $b[$sp + 4];
            $nc = $b[$sp + 5];
            if ($w == 0 || $h == 0) { throw "JPEG: a " + str($w) + "x" + str($h) + " frame"; }
            if ($nc < 1 || $nc > 4) {
                throw "JPEG: cannot handle " + str($nc) + "-layer images";
            }
            if ($w * $h > $MAX_PIXELS) {
                throw "JPEG: " + str($w) + "x" + str($h) + " is " + str($w * $h) +
                      " pixels, over MAX_IMAGE_PIXELS (" + str($MAX_PIXELS) +
                      "); this could be a decompression bomb";
            }
            if ($sp + 6 + $nc * 3 > $send) { throw "JPEG: a truncated SOF component list"; }
            $comps = [];
            $maxH = 1;
            $maxV = 1;
            $i = 0;
            while ($i < $nc) {
                $o = $sp + 6 + $i * 3;
                $hv = $b[$o + 1];
                $ch = floor($hv / 16);
                $cv = $hv % 16;
                if ($ch < 1 || $ch > 4 || $cv < 1 || $cv > 4) {
                    throw "JPEG: sampling factors " + str($ch) + "x" + str($cv) +
                          " for component " + str($i) + " (1..4 each)";
                }
                if ($ch > $maxH) { $maxH = $ch; }
                if ($cv > $maxV) { $maxV = $cv; }
                push($comps, {"id": $b[$o], "h": $ch, "v": $cv, "tq": $b[$o + 2],
                              "qt": null, "wb": 0, "hb": 0, "bw": 0, "bh": 0, "off": 0});
                $i = $i + 1;
            }
            $mcusPerRow = _ceilDiv($w, 8 * $maxH);
            $mcuRows = _ceilDiv($h, 8 * $maxV);
            $total = 0;
            $i = 0;
            while ($i < $nc) {
                $c = $comps[$i];
                $c.wb = $mcusPerRow * $c.h;
                $c.hb = $mcuRows * $c.v;
                $c.bw = _ceilDiv(_ceilDiv($w * $c.h, $maxH), 8);
                $c.bh = _ceilDiv(_ceilDiv($h * $c.v, $maxV), 8);
                $c.off = $total;
                $total = $total + $c.wb * $c.hb * 64;
                $i = $i + 1;
            }
            $fr = {"w": $w, "h": $h, "comps": $comps, "maxH": $maxH, "maxV": $maxV,
                   "mcusPerRow": $mcusPerRow, "mcuRows": $mcuRows,
                   "restart": $restart, "progressive": $progressive,
                   "jfif": $jfif, "adobe": $adobe, "adobeTransform": $adobeTransform,
                   "total": $total, "coef": null, "plist": null};
            if ($progressive) {
                // Decision PW13: the budget is checked HERE, from the frame
                // header, before a byte of scan is read or a list allocated.
                if ($total > $CFG.MAX_PROGRESSIVE_COEFFS) {
                    throw "JPEG: this progressive image needs " + str($total) +
                          " coefficients, over the budget of " +
                          str($CFG.MAX_PROGRESSIVE_COEFFS) +
                          ". A progressive scan must hold every coefficient in a " +
                          "randomly-readable store (~100 bytes each in Bantu), so " +
                          "decoding it would need about " + str(floor($total / 10000000)) +
                          " GB. Re-save the file as baseline, or raise the budget " +
                          "with jpeg.set_max_progressive_coeffs() if the memory is there.";
                }
                $fr.plist = nd_to_list(nd_zeros([$total], "i64"));
            } else {
                $fr.coef = nd_zeros([$total], "i64");
            }

        } else if ($mk == 195 || ($mk >= 197 && $mk <= 203) ||
                   ($mk >= 205 && $mk <= 207)) {
            // SOF3, SOF5-7 (hierarchical/differential), SOF9-11 and SOF13-15
            // (arithmetic coding). libjpeg itself refuses these.
            throw "JPEG: SOF" + str($mk - 192) + " is lossless, hierarchical or " +
                  "arithmetic-coded, which libjpeg does not decode either";

        } else if ($mk == 218) {
            // ── SOS: the scan header, then the entropy-coded data ──
            if ($fr == null) { throw "JPEG: a scan before any frame header"; }
            if ($sp + 1 > $send) { throw "JPEG: a truncated SOS segment"; }
            $ns = $b[$sp];
            if ($ns < 1 || $ns > 4) { throw "JPEG: a scan over " + str($ns) + " components"; }
            if ($sp + 1 + $ns * 2 + 3 > $send) { throw "JPEG: a truncated SOS component list"; }
            $scomps = [];
            $i = 0;
            while ($i < $ns) {
                $cid = $b[$sp + 1 + $i * 2];
                $tt = $b[$sp + 2 + $i * 2];
                $ci = 0 - 1;
                $k = 0;
                while ($k < len($fr.comps)) {
                    if ($fr.comps[$k].id == $cid) { $ci = $k; break; }
                    $k = $k + 1;
                }
                if ($ci < 0) {
                    throw "JPEG: the scan names component " + str($cid) +
                          ", which the frame does not contain";
                }
                $dcId = floor($tt / 16);
                $acId = $tt % 16;
                // Latch the quantisation table in force for this component,
                // the way libjpeg latches it at the start of the first scan
                // the component appears in.
                $c = $fr.comps[$ci];
                if ($c.qt == null) {
                    if ($c.tq > 3) { throw "JPEG: quantisation table id " + str($c.tq); }
                    $c.qt = $qt[$c.tq];
                }
                push($scomps, {"ci": $ci, "dc": $hdc[$dcId], "ac": $hac[$acId]});
                $i = $i + 1;
            }
            $o = $sp + 1 + $ns * 2;
            $ss = $b[$o];
            $se = $b[$o + 1];
            $a = $b[$o + 2];
            $ah = floor($a / 16);
            $al = $a % 16;
            if ($se > 63 || $ss > 63 || $ss > $se) {
                throw "JPEG: a scan over coefficients " + str($ss) + ".." + str($se);
            }
            $fr.restart = $restart;
            $sc = {"comps": $scomps, "ss": $ss, "se": $se, "ah": $ah, "al": $al};
            if ($progressive) {
                if ($ss == 0 && $ah == 0) {
                    $i = 0;
                    while ($i < $ns) {
                        if ($scomps[$i].dc == null) {
                            throw "JPEG: a DC scan names Huffman table " +
                                  "that the file never defines";
                        }
                        $i = $i + 1;
                    }
                }
                $pos = _scanProgressive($b, $send, $fr, $sc);
            } else {
                if ($ss != 0 || $se != 63) {
                    throw "JPEG: a sequential scan must cover coefficients 0..63, " +
                          "not " + str($ss) + ".." + str($se);
                }
                $i = 0;
                while ($i < $ns) {
                    if ($scomps[$i].dc == null || $scomps[$i].ac == null) {
                        throw "JPEG: the scan names a Huffman table the file never defines";
                    }
                    $i = $i + 1;
                }
                $pos = _scanBaseline($b, $send, $fr, $sc);
            }
            $scans = $scans + 1;
            continue;

        } else if ($mk == 224) {
            // ── APP0: JFIF ──
            if ($send - $sp >= 5 && $b[$sp] == 74 && $b[$sp + 1] == 70 &&
                $b[$sp + 2] == 73 && $b[$sp + 3] == 70 && $b[$sp + 4] == 0) {
                $jfif = true;
                if ($send - $sp >= 7) {
                    $ver = $b[$sp + 5] * 256 + $b[$sp + 6];
                    $info.jfif = $ver;
                    $info.jfif_version = [floor($ver / 256), $ver % 256];
                }
                if ($send - $sp >= 12) {
                    $unit = $b[$sp + 7];
                    $dx = $b[$sp + 8] * 256 + $b[$sp + 9];
                    $dy = $b[$sp + 10] * 256 + $b[$sp + 11];
                    if ($unit == 1) { $info.dpi = [$dx, $dy]; }
                    else if ($unit == 2) { $info.dpi = [$dx * 2.54, $dy * 2.54]; }
                    $info.jfif_unit = $unit;
                    $info.jfif_density = [$dx, $dy];
                }
            }

        } else if ($mk == 225) {
            // ── APP1: EXIF or XMP ──
            if ($send - $sp >= 6 && $b[$sp] == 69 && $b[$sp + 1] == 120 &&
                $b[$sp + 2] == 105 && $b[$sp + 3] == 102 &&
                $b[$sp + 4] == 0 && $b[$sp + 5] == 0) {
                $seg = substr($data, $sp, $send - $sp);
                if ($info.exif == null) { $info.exif = $seg; }
                else { $info.exif = $info.exif + substr($seg, 6, len($seg) - 6); }
            } else if ($send - $sp > 29 && substr($data, $sp, 29) ==
                       "http://ns.adobe.com/xap/1.0/\x00") {
                $info.xmp = substr($data, $sp + 29, $send - $sp - 29);
            }

        } else if ($mk == 226) {
            // ── APP2: an ICC profile, possibly split over many segments ──
            if ($send - $sp >= 14 && substr($data, $sp, 12) == "ICC_PROFILE\x00") {
                push($icc, {"seq": $b[$sp + 12], "of": $b[$sp + 13],
                            "data": substr($data, $sp + 14, $send - $sp - 14)});
                $iccCount = $iccCount + 1;
            }

        } else if ($mk == 238) {
            // ── APP14: Adobe ──
            if ($send - $sp >= 7 && substr($data, $sp, 5) == "Adobe") {
                $adobe = $b[$sp + 5] * 256 + $b[$sp + 6];
                $info.adobe = $adobe;
                if ($send - $sp >= 12) {
                    $adobeTransform = $b[$sp + 11];
                    $info.adobe_transform = $adobeTransform;
                }
                if ($fr != null) {
                    $fr.adobe = $adobe;
                    $fr.adobeTransform = $adobeTransform;
                }
            }

        } else if ($mk == 254) {
            // ── COM ──
            $info.comment = substr($data, $sp, $send - $sp);
        }

        $pos = $send;
    }

    if ($fr == null) { throw "JPEG: no frame header (SOF) in the file"; }
    if ($scans == 0) { throw "JPEG: no scan (SOS) in the file"; }

    if ($progressive) {
        $fr.coef = nd($fr.plist, "i64");
        $fr.plist = null;
        $info.progressive = 1;
        $info.progression = 1;
    }
    $fr.jfif = $jfif;
    if ($fr.adobe == null) {
        $fr.adobe = $adobe;
        $fr.adobeTransform = $adobeTransform;
    }

    // The ICC profile, reassembled in sequence order the way Pillow does:
    // the segments carry their own numbering and need not arrive in order.
    if ($iccCount > 0) {
        $ok = true;
        $parts = [];
        $i = 0;
        while ($i < $iccCount) { push($parts, null); $i = $i + 1; }
        $i = 0;
        while ($i < $iccCount) {
            $e = $icc[$i];
            if ($e.of != $iccCount || $e.seq < 1 || $e.seq > $iccCount ||
                $parts[$e.seq - 1] != null) { $ok = false; }
            else { $parts[$e.seq - 1] = $e.data; }
            $i = $i + 1;
        }
        if ($ok) { $info.icc_profile = join($parts, ""); }
    }

    $out = _assemble($fr, $denom, $want);
    $info.jpeg_qtables = _qtablesOf($fr);
    $info.jpeg_subsampling = _samplingOf($fr);
    // Retained so that draft() can re-decode at a smaller scale, the way
    // Pillow's lazy open lets libjpeg do it -- an Image here is never lazy,
    // so the bytes are the only way back. See image.b's draft().
    $info.jpeg_src = $data;
    return {"mode": $out.mode, "size": $out.size, "px": $out.px,
            "palette": null, "info": $info};
}

// _qtablesOf($fr) -> the quantisation tables actually used, in natural
// order, as a list indexed by table id. Pillow exposes the same thing as
// im.quantization and accepts it back as the `qtables` save option.
def _qtablesOf($fr) {
    $out = {};
    $i = 0;
    while ($i < len($fr.comps)) {
        $c = $fr.comps[$i];
        if ($c.qt != null) { $out[str($c.tq)] = $c.qt; }
        $i = $i + 1;
    }
    return $out;
}

// _samplingOf($fr) -> 0 for 4:4:4, 1 for 4:2:2, 2 for 4:2:0, else -1.
// Pillow's JpegImagePlugin.get_sampling, which is what `subsampling="keep"`
// reads.
def _samplingOf($fr) {
    if (len($fr.comps) != 3) { return 0 - 1; }
    $c = $fr.comps;
    if ($c[1].h != 1 || $c[1].v != 1 || $c[2].h != 1 || $c[2].v != 1) { return 0 - 1; }
    if ($c[0].h == 1 && $c[0].v == 1) { return 0; }
    if ($c[0].h == 2 && $c[0].v == 1) { return 1; }
    if ($c[0].h == 2 && $c[0].v == 2) { return 2; }
    return 0 - 1;
}

// ════════════════════════════════════════════════════════════════════════
//  The encoder
// ════════════════════════════════════════════════════════════════════════
//
// WHAT IS AND IS NOT CLAIMED HERE. The decoder above is bit-exact with
// libjpeg because every stage of a decode is spec- or table-determined. An
// ENCODER has freedom: which quantisation tables, which Huffman tables,
// which scan script. So this encoder matches libjpeg where libjpeg's choice
// is published -- Annex K's tables scaled by jpeg_quality_scaling, the
// islow forward DCT, jcsample's downsamplers with their alternating
// rounding bias, jccolor's RGB->YCbCr tables -- and the test asserts a PSNR
// FLOOR against Pillow's output rather than byte equality.
//
// Two deliberate simplifications, both visible only in file size:
//
//   * `progressive` writes a SPECTRAL-SELECTION script -- one DC scan, then
//     one AC scan per component -- rather than libjpeg's successive-
//     approximation script. The file is a valid progressive JPEG, decodes
//     to the same pixels, and is a few per cent larger. Successive
//     approximation is the upgrade path if that matters.
//   * end-of-band runs are emitted one at a time rather than accumulated.
//     The standard Annex K tables contain no EOBn symbol but EOB itself, so
//     a run longer than one cannot be spelled with them at all; with
//     `optimize` it could, at a few per cent.

// _qualityScale($q) -> jpeg_quality_scaling: the percentage the Annex K
// tables are multiplied by.
def _qualityScale($q) {
    $v = floor($q);
    if ($v <= 0) { $v = 1; }
    if ($v > 100) { $v = 100; }
    if ($v < 50) { return floor(5000 / $v); }
    return 200 - $v * 2;
}

// _scaleTable($base, $scale) -> jpeg_add_quant_table with force_baseline,
// which is what Pillow asks for: every value in 1..255.
def _scaleTable($base, $scale) {
    $out = [];
    $i = 0;
    while ($i < 64) {
        $t = floor(($base[$i] * $scale + 50) / 100);
        if ($t <= 0) { $t = 1; }
        if ($t > 255) { $t = 255; }
        push($out, $t);
        $i = $i + 1;
    }
    return $out;
}

// _ehufOf($bits, $vals) -> {"co": [...], "si": [...]} indexed by symbol:
// the canonical code and its length, which is jchuff.c's jpeg_make_c_derived_tbl.
def _ehufOf($bits, $vals) {
    $co = nd_to_list(nd_zeros([256], "i64"));
    $si = nd_to_list(nd_zeros([256], "i64"));
    $code = 0;
    $k = 0;
    $l = 1;
    while ($l <= 16) {
        $i = 0;
        while ($i < $bits[$l]) {
            $sym = $vals[$k];
            if ($sym < 0 || $sym > 255) { throw "JPEG: a Huffman symbol of " + str($sym); }
            $co[$sym] = $code;
            $si[$sym] = $l;
            $code = $code + 1;
            $k = $k + 1;
            $i = $i + 1;
        }
        $code = $code * 2;
        $l = $l + 1;
    }
    return {"co": $co, "si": $si};
}

// _genOptimal($freq) -> [bits, vals], jchuff.c's jpeg_gen_optimal_table.
//
// T.81 Annex K.2's algorithm: build the Huffman tree by repeatedly merging
// the two least frequent symbols, read off the code LENGTHS, then limit any
// length over 16 by the standard pair-shuffling fix-up. One codeword of all
// ones is reserved (freq[256] = 1) because a JPEG decoder must never see it.
def _genOptimal($freq) {
    $f = [];
    $i = 0;
    while ($i < 257) {
        if ($i < len($freq)) { push($f, $freq[$i]); } else { push($f, 0); }
        $i = $i + 1;
    }
    $f[256] = 1;                     // reserve the all-ones codeword

    $codesize = nd_to_list(nd_zeros([257], "i64"));
    $others = [];
    $i = 0;
    while ($i < 257) { push($others, 0 - 1); $i = $i + 1; }

    $done = false;
    while (!$done) {
        // The two smallest non-zero frequencies; ties go to the HIGHER
        // index, which is what the C's `>=` does and what makes its output
        // reproducible.
        $c1 = 0 - 1;
        $v1 = 0;
        $i = 0;
        while ($i <= 256) {
            if ($f[$i] > 0 && ($c1 < 0 || $f[$i] <= $v1)) { $c1 = $i; $v1 = $f[$i]; }
            $i = $i + 1;
        }
        $c2 = 0 - 1;
        $v2 = 0;
        $i = 0;
        while ($i <= 256) {
            if ($i != $c1 && $f[$i] > 0 && ($c2 < 0 || $f[$i] <= $v2)) {
                $c2 = $i;
                $v2 = $f[$i];
            }
            $i = $i + 1;
        }
        if ($c2 < 0) { $done = true; }
        else {
            $f[$c1] = $f[$c1] + $f[$c2];
            $f[$c2] = 0;
            $codesize[$c1] = $codesize[$c1] + 1;
            while ($others[$c1] >= 0) {
                $c1 = $others[$c1];
                $codesize[$c1] = $codesize[$c1] + 1;
            }
            $others[$c1] = $c2;
            $codesize[$c2] = $codesize[$c2] + 1;
            while ($others[$c2] >= 0) {
                $c2 = $others[$c2];
                $codesize[$c2] = $codesize[$c2] + 1;
            }
        }
    }

    // How many codes of each length.
    $bits = nd_to_list(nd_zeros([33], "i64"));
    $i = 0;
    while ($i <= 256) {
        if ($codesize[$i] > 0) {
            if ($codesize[$i] > 32) { throw "JPEG: a Huffman code length over 32 bits"; }
            $bits[$codesize[$i]] = $bits[$codesize[$i]] + 1;
        }
        $i = $i + 1;
    }

    // Limit the lengths to 16 bits, Annex K's way: take a leaf from the
    // over-long level and re-hang it one level up, which costs one bit on
    // one symbol and never changes the code's prefix-freeness.
    $i = 32;
    while ($i > 16) {
        while ($bits[$i] > 0) {
            $j = $i - 2;
            while ($bits[$j] == 0) { $j = $j - 1; }
            $bits[$i] = $bits[$i] - 2;
            $bits[$i - 1] = $bits[$i - 1] + 1;
            $bits[$j + 1] = $bits[$j + 1] + 2;
            $bits[$j] = $bits[$j] - 1;
        }
        $i = $i - 1;
    }
    // Remove the reserved all-ones codeword, which is the last one at the
    // longest length in use.
    while ($bits[$i] == 0) { $i = $i - 1; }
    $bits[$i] = $bits[$i] - 1;

    $outBits = [0];
    $i = 1;
    while ($i <= 16) { push($outBits, $bits[$i]); $i = $i + 1; }

    // The symbols, in order of increasing code length and then of symbol.
    $vals = [];
    $l = 1;
    while ($l <= 32) {
        $i = 0;
        while ($i <= 255) {
            if ($codesize[$i] == $l) { push($vals, $i); }
            $i = $i + 1;
        }
        $l = $l + 1;
    }
    return [$outBits, $vals];
}

// ─── Blocks in scan order ───────────────────────────────────────────────
//
// _blockOrder($fr, $scomps) -> [the flat block index of every block the
// scan codes, in the order it codes them].
//
// Every component's quantised coefficients live in ONE array of
// (totalBlocks, 64), component ci starting at $fr.comps[ci].blk. A scan
// wants them in MCU order, so one gather by this permutation puts them in
// exactly the order the entropy coder walks -- after which the coder reads
// a contiguous slice at a time and never indexes an array per coefficient.
def _blockOrder($fr, $scomps) {
    $out = [];
    $ncs = len($scomps);
    if ($ncs == 1) {
        $c = $fr.comps[$scomps[0]];
        $r = 0;
        while ($r < $c.bh) {
            $k = 0;
            while ($k < $c.bw) {
                push($out, $c.blk + $r * $c.wb + $k);
                $k = $k + 1;
            }
            $r = $r + 1;
        }
        return $out;
    }
    $my = 0;
    while ($my < $fr.mcuRows) {
        $mx = 0;
        while ($mx < $fr.mcusPerRow) {
            $i = 0;
            while ($i < $ncs) {
                $c = $fr.comps[$scomps[$i]];
                $bv = 0;
                while ($bv < $c.v) {
                    $bh = 0;
                    while ($bh < $c.h) {
                        push($out, $c.blk + ($my * $c.v + $bv) * $c.wb + $mx * $c.h + $bh);
                        $bh = $bh + 1;
                    }
                    $bv = $bv + 1;
                }
                $i = $i + 1;
            }
            $mx = $mx + 1;
        }
        $my = $my + 1;
    }
    return $out;
}

// _tableOfBlock($fr, $scomps) -> which scan component each block of one MCU
// belongs to, so the hot loop picks its Huffman tables with one list read.
def _tableOfBlock($fr, $scomps) {
    $out = [];
    $ncs = len($scomps);
    if ($ncs == 1) { return [0]; }
    $i = 0;
    while ($i < $ncs) {
        $c = $fr.comps[$scomps[$i]];
        $k = 0;
        while ($k < $c.h * $c.v) { push($out, $i); $k = $k + 1; }
        $i = $i + 1;
    }
    return $out;
}

// _encSeqScan($fr, $scomps, $dcOnly, $count, $freq) ->
//     {"bytes": <the entropy-coded scan>, "freq": <symbol counts>}
//
// jchuff.c's encode_one_block, over the MCU order of _blockOrder. With
// $dcOnly it is jcphuff.c's encode_mcu_DC_first at Al = 0, which is the
// same code with the AC loop skipped.
//
// $count true runs the SAME loop without emitting, to gather the symbol
// frequencies `optimize` needs. One copy of the logic, two passes over it.
def _encSeqScan($fr, $scomps, $dcOnly, $count, $freq) {
    $P2 = $_P2;
    $ncs = len($scomps);
    $order = _blockOrder($fr, $scomps);
    $nblk = len($order);
    $tob = _tableOfBlock($fr, $scomps);
    $perMcu = len($tob);

    // The scan's coefficients, in scan order, as one array.
    $seq = nd_take($fr.coefs, nd($order, "i64"), 0);
    $flat = nd_reshape($seq, [$nblk * 64]);

    // Huffman codes, flattened: table t of scan component i is at i * 512
    // for DC and i * 512 + 256 for AC.
    $CO = [];
    $SI = [];
    $i = 0;
    while ($i < $ncs) {
        $c = $fr.comps[$scomps[$i]];
        $two = [$c.dcEh, $c.acEh];
        $w = 0;
        while ($w < 2) {
            $t = $two[$w];
            $k = 0;
            if ($t == null) {
                while ($k < 256) { push($CO, 0); push($SI, 0); $k = $k + 1; }
            } else {
                $tc = $t.co;
                $ts = $t.si;
                while ($k < 256) { push($CO, $tc[$k]); push($SI, $ts[$k]); $k = $k + 1; }
            }
            $w = $w + 1;
        }
        $i = $i + 1;
    }
    // Which frequency slot each scan component's DC and AC tables use.
    $FOFF = [];
    $i = 0;
    while ($i < $ncs) {
        $c = $fr.comps[$scomps[$i]];
        push($FOFF, $c.dcNo * 256);
        push($FOFF, 1024 + $c.acNo * 256);
        $i = $i + 1;
    }
    $F = $freq;
    if ($F == null) { $F = nd_to_list(nd_zeros([2048], "i64")); }

    $ob = [];
    $acc = 0;
    $n = 0;
    $preds = nd_to_list(nd_zeros([$ncs], "i64"));
    $ri = $fr.restart;
    $mcusLeft = $ri;
    $rst = 0;

    $CHUNK = 1024;
    $blk = 0;
    $chunkLo = 0;
    $chunk = [];
    while ($blk < $nblk) {
        if ($blk >= $chunkLo + $CHUNK || $blk == 0) {
            $chunkLo = $blk;
            $hi = $chunkLo + $CHUNK;
            if ($hi > $nblk) { $hi = $nblk; }
            $chunk = nd_to_list(nd_slice($flat, [[$chunkLo * 64, $hi * 64]]));
        }
        // A restart interval boundary, counted in MCUs.
        if ($ri > 0 && $mcusLeft == 0 && $blk % $perMcu == 0) {
            if (!$count) {
                // Pad to a byte with 1-bits, which is what jchuff does, then
                // the marker.
                if ($n > 0) {
                    $pad = 8 - $n;
                    $acc = $acc * $P2[$pad] + ($P2[$pad] - 1);
                    $n = 8;
                    $by = $acc % 256;
                    push($ob, $by);
                    if ($by == 255) { push($ob, 0); }
                    $acc = 0;
                    $n = 0;
                }
                push($ob, 255);
                push($ob, 208 + $rst % 8);
            }
            $rst = $rst + 1;
            $i = 0;
            while ($i < $ncs) { $preds[$i] = 0; $i = $i + 1; }
            $mcusLeft = $ri;
        }
        if ($blk % $perMcu == $perMcu - 1) { $mcusLeft = $mcusLeft - 1; }

        $ci = $tob[$blk % $perMcu];
        $co = $ci * 512;
        $ao = $co + 256;
        $fdc = $FOFF[$ci * 2];
        $fac = $FOFF[$ci * 2 + 1];
        $base = ($blk - $chunkLo) * 64;

        // ── the DC difference ──
        $dc = $chunk[$base];
        $diff = $dc - $preds[$ci];
        $preds[$ci] = $dc;
        $t2 = $diff;
        $t = $diff;
        if ($t < 0) { $t = 0 - $t; $t2 = $t2 - 1; }
        $nb = 0;
        while ($t > 0) { $nb = $nb + 1; $t = floor($t / 2); }
        if ($nb > 15) { throw "JPEG: a DC difference of " + str($diff) + " needs " + str($nb) + " bits"; }
        if ($count) {
            $F[$fdc + $nb] = $F[$fdc + $nb] + 1;
        } else {
            $sz = $SI[$co + $nb];
            if ($sz == 0) { throw "JPEG: no Huffman code for DC category " + str($nb); }
            $acc = $acc * $P2[$sz] + $CO[$co + $nb];
            $n = $n + $sz;
            while ($n >= 8) {
                $n = $n - 8;
                $by = floor($acc / $P2[$n]);
                $acc = $acc % $P2[$n];
                push($ob, $by);
                if ($by == 255) { push($ob, 0); }
            }
            if ($nb > 0) {
                $v = $t2 % $P2[$nb];
                if ($v < 0) { $v = $v + $P2[$nb]; }
                $acc = $acc * $P2[$nb] + $v;
                $n = $n + $nb;
                while ($n >= 8) {
                    $n = $n - 8;
                    $by = floor($acc / $P2[$n]);
                    $acc = $acc % $P2[$n];
                    push($ob, $by);
                    if ($by == 255) { push($ob, 0); }
                }
            }
        }

        if (!$dcOnly) {
            // ── the AC coefficients, in zigzag order ──
            $r = 0;
            $k = 1;
            while ($k < 64) {
                $v = $chunk[$base + $k];
                if ($v == 0) {
                    $r = $r + 1;
                } else {
                    while ($r > 15) {
                        if ($count) { $F[$fac + 240] = $F[$fac + 240] + 1; }
                        else {
                            $sz = $SI[$ao + 240];
                            if ($sz == 0) { throw "JPEG: no Huffman code for ZRL"; }
                            $acc = $acc * $P2[$sz] + $CO[$ao + 240];
                            $n = $n + $sz;
                            while ($n >= 8) {
                                $n = $n - 8;
                                $by = floor($acc / $P2[$n]);
                                $acc = $acc % $P2[$n];
                                push($ob, $by);
                                if ($by == 255) { push($ob, 0); }
                            }
                        }
                        $r = $r - 16;
                    }
                    $t2 = $v;
                    $t = $v;
                    if ($t < 0) { $t = 0 - $t; $t2 = $t2 - 1; }
                    $nb = 1;
                    $t = floor($t / 2);
                    while ($t > 0) { $nb = $nb + 1; $t = floor($t / 2); }
                    if ($nb > 10) {
                        throw "JPEG: an AC coefficient of " + str($v) + " needs " +
                              str($nb) + " bits, over the 10 a baseline JPEG allows";
                    }
                    $sym = $r * 16 + $nb;
                    if ($count) { $F[$fac + $sym] = $F[$fac + $sym] + 1; }
                    else {
                        $sz = $SI[$ao + $sym];
                        if ($sz == 0) {
                            throw "JPEG: no Huffman code for run/size " + str($sym);
                        }
                        $acc = $acc * $P2[$sz] + $CO[$ao + $sym];
                        $n = $n + $sz;
                        while ($n >= 8) {
                            $n = $n - 8;
                            $by = floor($acc / $P2[$n]);
                            $acc = $acc % $P2[$n];
                            push($ob, $by);
                            if ($by == 255) { push($ob, 0); }
                        }
                        $vv = $t2 % $P2[$nb];
                        if ($vv < 0) { $vv = $vv + $P2[$nb]; }
                        $acc = $acc * $P2[$nb] + $vv;
                        $n = $n + $nb;
                        while ($n >= 8) {
                            $n = $n - 8;
                            $by = floor($acc / $P2[$n]);
                            $acc = $acc % $P2[$n];
                            push($ob, $by);
                            if ($by == 255) { push($ob, 0); }
                        }
                    }
                    $r = 0;
                }
                $k = $k + 1;
            }
            if ($r > 0) {
                if ($count) { $F[$fac] = $F[$fac] + 1; }
                else {
                    $sz = $SI[$ao];
                    if ($sz == 0) { throw "JPEG: no Huffman code for EOB"; }
                    $acc = $acc * $P2[$sz] + $CO[$ao];
                    $n = $n + $sz;
                    while ($n >= 8) {
                        $n = $n - 8;
                        $by = floor($acc / $P2[$n]);
                        $acc = $acc % $P2[$n];
                        push($ob, $by);
                        if ($by == 255) { push($ob, 0); }
                    }
                }
            }
        }
        $blk = $blk + 1;
    }

    if (!$count && $n > 0) {
        // jchuff's finish_pass: pad the last byte with 1-bits.
        $pad = 8 - $n;
        $acc = $acc * $P2[$pad] + ($P2[$pad] - 1);
        $by = $acc % 256;
        push($ob, $by);
        if ($by == 255) { push($ob, 0); }
    }
    return {"bytes": frombytes($ob), "freq": $F};
}

// _encAcScan($fr, $ci, $ss, $se, $count, $freq) -> the AC scan of ONE
// component, jcphuff.c's encode_mcu_AC_first at Al = 0.
//
// End-of-band runs are emitted one block at a time (see the note at the top
// of the encoder), so this is encode_one_block's AC half with its own EOB.
def _encAcScan($fr, $ci, $ss, $se, $count, $freq) {
    $P2 = $_P2;
    $c = $fr.comps[$ci];
    $order = _blockOrder($fr, [$ci]);
    $nblk = len($order);
    $seq = nd_take($fr.coefs, nd($order, "i64"), 0);
    $flat = nd_reshape($seq, [$nblk * 64]);

    $CO = $c.acEh.co;
    $SI = $c.acEh.si;
    $fac = 1024 + $c.acNo * 256;
    $F = $freq;
    if ($F == null) { $F = nd_to_list(nd_zeros([2048], "i64")); }

    $ob = [];
    $acc = 0;
    $n = 0;
    $ri = $fr.restart;
    $left = $ri;
    $rst = 0;

    $CHUNK = 1024;
    $blk = 0;
    $chunkLo = 0;
    $chunk = [];
    while ($blk < $nblk) {
        if ($blk >= $chunkLo + $CHUNK || $blk == 0) {
            $chunkLo = $blk;
            $hi = $chunkLo + $CHUNK;
            if ($hi > $nblk) { $hi = $nblk; }
            $chunk = nd_to_list(nd_slice($flat, [[$chunkLo * 64, $hi * 64]]));
        }
        if ($ri > 0 && $left == 0) {
            if (!$count) {
                if ($n > 0) {
                    $pad = 8 - $n;
                    $acc = $acc * $P2[$pad] + ($P2[$pad] - 1);
                    $by = $acc % 256;
                    push($ob, $by);
                    if ($by == 255) { push($ob, 0); }
                    $acc = 0;
                    $n = 0;
                }
                push($ob, 255);
                push($ob, 208 + $rst % 8);
            }
            $rst = $rst + 1;
            $left = $ri;
        }
        $left = $left - 1;
        $base = ($blk - $chunkLo) * 64;

        $r = 0;
        $k = $ss;
        while ($k <= $se) {
            $v = $chunk[$base + $k];
            if ($v == 0) {
                $r = $r + 1;
            } else {
                while ($r > 15) {
                    if ($count) { $F[$fac + 240] = $F[$fac + 240] + 1; }
                    else {
                        $sz = $SI[240];
                        if ($sz == 0) { throw "JPEG: no Huffman code for ZRL"; }
                        $acc = $acc * $P2[$sz] + $CO[240];
                        $n = $n + $sz;
                        while ($n >= 8) {
                            $n = $n - 8;
                            $by = floor($acc / $P2[$n]);
                            $acc = $acc % $P2[$n];
                            push($ob, $by);
                            if ($by == 255) { push($ob, 0); }
                        }
                    }
                    $r = $r - 16;
                }
                $t2 = $v;
                $t = $v;
                if ($t < 0) { $t = 0 - $t; $t2 = $t2 - 1; }
                $nb = 1;
                $t = floor($t / 2);
                while ($t > 0) { $nb = $nb + 1; $t = floor($t / 2); }
                $sym = $r * 16 + $nb;
                if ($count) { $F[$fac + $sym] = $F[$fac + $sym] + 1; }
                else {
                    $sz = $SI[$sym];
                    if ($sz == 0) { throw "JPEG: no Huffman code for run/size " + str($sym); }
                    $acc = $acc * $P2[$sz] + $CO[$sym];
                    $n = $n + $sz;
                    while ($n >= 8) {
                        $n = $n - 8;
                        $by = floor($acc / $P2[$n]);
                        $acc = $acc % $P2[$n];
                        push($ob, $by);
                        if ($by == 255) { push($ob, 0); }
                    }
                    $vv = $t2 % $P2[$nb];
                    if ($vv < 0) { $vv = $vv + $P2[$nb]; }
                    $acc = $acc * $P2[$nb] + $vv;
                    $n = $n + $nb;
                    while ($n >= 8) {
                        $n = $n - 8;
                        $by = floor($acc / $P2[$n]);
                        $acc = $acc % $P2[$n];
                        push($ob, $by);
                        if ($by == 255) { push($ob, 0); }
                    }
                }
                $r = 0;
            }
            $k = $k + 1;
        }
        if ($r > 0) {
            if ($count) { $F[$fac] = $F[$fac] + 1; }
            else {
                $sz = $SI[0];
                if ($sz == 0) { throw "JPEG: no Huffman code for EOB"; }
                $acc = $acc * $P2[$sz] + $CO[0];
                $n = $n + $sz;
                while ($n >= 8) {
                    $n = $n - 8;
                    $by = floor($acc / $P2[$n]);
                    $acc = $acc % $P2[$n];
                    push($ob, $by);
                    if ($by == 255) { push($ob, 0); }
                }
            }
        }
        $blk = $blk + 1;
    }
    if (!$count && $n > 0) {
        $pad = 8 - $n;
        $acc = $acc * $P2[$pad] + ($P2[$pad] - 1);
        $by = $acc % 256;
        push($ob, $by);
        if ($by == 255) { push($ob, 0); }
    }
    return {"bytes": frombytes($ob), "freq": $F};
}

// ─── Marker writing ─────────────────────────────────────────────────────

def _m($code) { return "\xff" + chr($code); }
def _b16($v) { return chr(floor($v / 256) % 256) + chr($v % 256); }

def _seg($code, $payload) {
    return _m($code) + _b16(len($payload) + 2) + $payload;
}

def _dqtSeg($id, $tab) {
    $out = [chr($id)];
    $k = 0;
    while ($k < 64) {
        // The table goes out in ZIGZAG order; it is held in natural order.
        push($out, chr($tab[jt.ZIGZAG[$k]] % 256));
        $k = $k + 1;
    }
    return _seg(219, join($out, ""));
}

def _dhtSeg($cls, $id, $bits, $vals) {
    $out = [chr($cls * 16 + $id)];
    $k = 1;
    while ($k <= 16) { push($out, chr($bits[$k])); $k = $k + 1; }
    $k = 0;
    while ($k < len($vals)) { push($out, chr($vals[$k])); $k = $k + 1; }
    return _seg(196, join($out, ""));
}

// _dummyBlocks($coefs, $c) -> the MCU padding filled libjpeg's way.
//
// A component's real blocks are bw x bh; an interleaved scan needs wb x hb,
// which is the MCU-aligned size. jccoefct.c's compress_output does NOT
// encode image content in the difference: it zeroes those blocks entirely
// and copies the DC from the PREVIOUS block in MCU order, so the DC
// difference is zero and there is nothing to spell but an end-of-block.
//
// Encoding the padded samples there instead costs real bits -- measured at
// 5 % of a 23x17 file -- and is the only thing that stopped this encoder
// being byte-identical to libjpeg on an image whose size is not a multiple
// of the MCU. The loop is over the PADDING, which is a perimeter.
def _dummyBlocks($coefs, $c) {
    if ($c.bw >= $c.wb && $c.bh >= $c.hb) { return false; }
    $mcuW = $c.h;
    $r = 0;
    while ($r < $c.hb) {
        $rowDummy = $r >= $c.bh;
        $cc = 0;
        while ($cc < $c.wb) {
            if ($rowDummy || $cc >= $c.bw) {
                $blk = $c.blk + $r * $c.wb + $cc;
                $v = nd_slice($coefs, [[$blk, $blk + 1], null]);
                nd_multiply($v, 0, $v);
                $prev = 0 - 1;
                if ($cc % $mcuW != 0) { $prev = $blk - 1; }
                else if ($r > 0) {
                    // The last block of the previous block row WITHIN THE
                    // SAME MCU, which is what MCU_buffer[blkn-1] is there.
                    $mcuCol = floor($cc / $mcuW);
                    $prev = $c.blk + ($r - 1) * $c.wb + $mcuCol * $mcuW + $mcuW - 1;
                }
                if ($prev >= 0) { nd_set($coefs, [$blk, 0], nd_get($coefs, [$prev, 0])); }
            }
            $cc = $cc + 1;
        }
        $r = $r + 1;
    }
    return true;
}

// ─── encode ─────────────────────────────────────────────────────────────

// encode($spec) -> the JPEG file bytes.
//
// Options ride on the spec dict, which is how every codec here takes them:
//
//   quality      1..100, "keep", or one of Pillow's preset names
//   optimize     true to build Huffman tables from this image's statistics
//   progressive  true to write a progressive (spectral-selection) file
//   subsampling  0 (4:4:4), 1 (4:2:2), 2 (4:2:0) or "keep"
//   qtables      a list of 64-entry tables in NATURAL order, as
//                info["jpeg_qtables"] hands them back
//   restart_marker_blocks / restart_marker_rows
//
// and info["dpi"], info["exif"], info["icc_profile"] and info["comment"]
// are written as JFIF density, APP1, APP2 and COM.
def encode($spec) {
    $m = $spec.mode;
    $px = $spec.px;
    $d = pxbuf.dims($px);
    $h = $d[0];
    $w = $d[1];
    if ($w == 0 || $h == 0) { throw "JPEG: cannot write a zero-sized image"; }
    if ($w > 65535 || $h > 65535) {
        throw "JPEG: " + str($w) + "x" + str($h) +
              " is over the 65535 a JPEG frame header can spell";
    }
    $info = $spec.info;
    if ($info == null) { $info = {}; }

    // ── the options ──
    $quality = 75;
    $subsampling = $spec.subsampling;
    $qtables = $spec.qtables;
    if ($spec.quality != null) { $quality = $spec.quality; }
    if (type($quality) == "string") {
        if ($quality == "keep") {
            if ($info.jpeg_qtables == null) {
                throw "JPEG: quality=\"keep\" needs the original tables in " +
                      "info[\"jpeg_qtables\"], which only a decoded JPEG has";
            }
            $keep = $info.jpeg_qtables;
            $qtables = [];
            $i = 0;
            while ($i < 4) {
                $t = $keep[str($i)];
                if ($t != null) { push($qtables, $t); }
                $i = $i + 1;
            }
            if ($subsampling == null && $info.jpeg_subsampling != null) {
                $subsampling = $info.jpeg_subsampling;
            }
            $quality = 75;
        } else {
            $pre = jt.PRESETS[$quality];
            if ($pre == null) {
                throw "JPEG: '" + $quality + "' is not a quality preset. " +
                      "Pillow's are web_low web_medium web_high web_very_high " +
                      "web_maximum low medium high maximum";
            }
            if ($subsampling == null) { $subsampling = $pre.subsampling; }
            if ($qtables == null) { $qtables = $pre.quantization; }
            $quality = 75;
        }
    }
    if ($subsampling == "keep") {
        if ($info.jpeg_subsampling == null) {
            throw "JPEG: subsampling=\"keep\" needs a decoded JPEG's info";
        }
        $subsampling = $info.jpeg_subsampling;
    }
    $optimize = $spec.optimize == true;
    $progressive = $spec.progressive == true;

    // ── mode -> colour space and planes ──
    $nc = 3;
    $cs = "YCbCr";
    if ($m == "L" || $m == "1") { $nc = 1; $cs = "GRAY"; }
    else if ($m == "RGB" || $m == "RGBX") { $nc = 3; $cs = "YCbCr"; }
    else if ($m == "YCbCr") { $nc = 3; $cs = "RAW3"; }
    else if ($m == "CMYK") { $nc = 4; $cs = "CMYK"; }
    else {
        throw "JPEG: cannot write mode " + str($m) + " as JPEG. Pillow writes " +
              "1, L, RGB, RGBX, CMYK and YCbCr; convert first.";
    }

    $planes = [];
    if ($cs == "GRAY") {
        push($planes, nd_reshape(nd_clip(pxbuf.band($px, 0), 0, 255), [$h, $w]));
    } else if ($cs == "YCbCr") {
        $r = nd_reshape(nd_clip(pxbuf.band($px, 0), 0, 255), [$h, $w]);
        $g = nd_reshape(nd_clip(pxbuf.band($px, 1), 0, 255), [$h, $w]);
        $bl = nd_reshape(nd_clip(pxbuf.band($px, 2), 0, 255), [$h, $w]);
        $ycc = dct.rgb_to_ycc($r, $g, $bl);
        push($planes, $ycc[0]);
        push($planes, $ycc[1]);
        push($planes, $ycc[2]);
    } else if ($cs == "RAW3") {
        $i = 0;
        while ($i < 3) {
            push($planes, nd_reshape(nd_clip(pxbuf.band($px, $i), 0, 255), [$h, $w]));
            $i = $i + 1;
        }
    } else {
        // CMYK;I: Pillow writes a CMYK JPEG inverted, which is what its
        // reader then undoes.
        $i = 0;
        while ($i < 4) {
            push($planes, nd_subtract(255,
                nd_reshape(nd_clip(pxbuf.band($px, $i), 0, 255), [$h, $w])));
            $i = $i + 1;
        }
    }

    // ── sampling factors, quantisation and Huffman table assignment ──
    // jcparam.c's jpeg_set_colorspace, which is what libjpeg would pick.
    $ss = $subsampling;
    if ($ss == null) { $ss = 2; }                 // libjpeg's own default: 4:2:0
    if ($nc != 3) { $ss = 0; }
    if ($ss != 0 && $ss != 1 && $ss != 2) {
        throw "JPEG: subsampling " + str($ss) + "; 0 (4:4:4), 1 (4:2:2) and " +
              "2 (4:2:0) are what Pillow writes";
    }
    $ids = [1, 2, 3, 4];
    $qno = [0, 1, 1, 1];
    $hno = [0, 1, 1, 1];
    if ($cs == "GRAY") { $ids = [1]; $qno = [0]; $hno = [0]; }
    else if ($cs == "CMYK") {
        $ids = [67, 77, 89, 75];                  // 'C' 'M' 'Y' 'K'
        $qno = [0, 0, 0, 0];
        $hno = [0, 0, 0, 0];
    }

    $comps = [];
    $maxH = 1;
    $maxV = 1;
    $i = 0;
    while ($i < $nc) {
        $ch = 1;
        $cv = 1;
        if ($i == 0 && $nc == 3) {
            if ($ss == 1) { $ch = 2; }
            else if ($ss == 2) { $ch = 2; $cv = 2; }
        }
        if ($ch > $maxH) { $maxH = $ch; }
        if ($cv > $maxV) { $maxV = $cv; }
        push($comps, {"id": $ids[$i], "h": $ch, "v": $cv, "qno": $qno[$i],
                      "dcNo": $hno[$i], "acNo": $hno[$i], "wb": 0, "hb": 0,
                      "bw": 0, "bh": 0, "blk": 0, "dcEh": null, "acEh": null});
        $i = $i + 1;
    }

    // Quantisation tables, in natural order.
    $qt = [null, null, null, null];
    if ($qtables != null) {
        if (type($qtables) != "list" || len($qtables) == 0) {
            throw "JPEG: qtables must be a list of 64-entry tables";
        }
        $i = 0;
        while ($i < len($qtables) && $i < 4) {
            if (len($qtables[$i]) != 64) {
                throw "JPEG: quantisation table " + str($i) + " has " +
                      str(len($qtables[$i])) + " entries, not 64";
            }
            $t = [];
            $k = 0;
            while ($k < 64) {
                $v = floor($qtables[$i][$k]);
                if ($v < 1 || $v > 255) {
                    throw "JPEG: a quantisation value of " + str($v) +
                          "; a baseline JPEG allows 1..255";
                }
                push($t, $v);
                $k = $k + 1;
            }
            $qt[$i] = $t;
            $i = $i + 1;
        }
        // A component whose table was not supplied falls back to the last one.
        $last = $i - 1;
        $k = 0;
        while ($k < $nc) {
            if ($qt[$comps[$k].qno] == null) { $comps[$k].qno = $last; }
            $k = $k + 1;
        }
    } else {
        if (type($quality) != "number") { throw "JPEG: quality must be a number"; }
        $scale = _qualityScale($quality);
        $qt[0] = _scaleTable(jt.STD_LUMA_QUANT, $scale);
        $qt[1] = _scaleTable(jt.STD_CHROMA_QUANT, $scale);
    }

    // ── the coefficients ──
    $mcusPerRow = _ceilDiv($w, 8 * $maxH);
    $mcuRows = _ceilDiv($h, 8 * $maxV);
    $padW = $mcusPerRow * 8 * $maxH;
    $padH = $mcuRows * 8 * $maxV;
    $total = 0;
    $i = 0;
    while ($i < $nc) {
        $c = $comps[$i];
        $c.wb = $mcusPerRow * $c.h;
        $c.hb = $mcuRows * $c.v;
        $c.bw = _ceilDiv(_ceilDiv($w * $c.h, $maxH), 8);
        $c.bh = _ceilDiv(_ceilDiv($h * $c.v, $maxV), 8);
        $c.blk = $total;
        $total = $total + $c.wb * $c.hb;
        $i = $i + 1;
    }
    $coefs = nd_zeros([$total, 64], "i64");
    $zig = nd(jt.ZIGZAG, "i64");
    $i = 0;
    while ($i < $nc) {
        $c = $comps[$i];
        $pl = dct.pad_edge($planes[$i], $padH, $padW);
        $ds = dct.downsample($pl, $maxH / $c.h, $maxV / $c.v);
        $blocks = dct.blocks_of(pxbuf.materialise2d($ds), $c.hb, $c.wb);
        $q = dct.quantise(dct.fdct8($blocks), $qt[$c.qno]);
        // Held in ZIGZAG order from here on, which is the order both the
        // entropy coder and the DQT segment want.
        nd_add(nd_take($q, $zig, 1), 0,
               nd_slice($coefs, [[$c.blk, $c.blk + $c.wb * $c.hb], null]));
        _dummyBlocks($coefs, $c);
        $i = $i + 1;
    }

    // ── restart interval ──
    $restart = 0;
    if ($spec.restart_marker_blocks != null) { $restart = floor($spec.restart_marker_blocks); }
    if ($spec.restart_marker_rows != null) {
        $restart = floor($spec.restart_marker_rows) * $mcusPerRow;
    }
    if ($restart < 0 || $restart > 65535) {
        throw "JPEG: a restart interval of " + str($restart);
    }

    $fr = {"comps": $comps, "mcusPerRow": $mcusPerRow, "mcuRows": $mcuRows,
           "restart": $restart, "coefs": $coefs};

    // ── the scan script ──
    $allc = [];
    $i = 0;
    while ($i < $nc) { push($allc, $i); $i = $i + 1; }
    $scans = [];
    if ($progressive) {
        push($scans, {"comps": $allc, "dcOnly": true, "ss": 0, "se": 0});
        $i = 0;
        while ($i < $nc) {
            push($scans, {"comps": [$i], "dcOnly": false, "ss": 1, "se": 63});
            $i = $i + 1;
        }
    } else {
        push($scans, {"comps": $allc, "dcOnly": false, "ss": 0, "se": 63});
    }

    // ── the file ──
    $out = [];
    push($out, _m(216));                                   // SOI
    if ($cs == "GRAY" || $cs == "YCbCr" || $cs == "RAW3") {
        // JFIF APP0, which libjpeg writes for grayscale and YCbCr.
        $unit = 0;
        $dx = 1;
        $dy = 1;
        if ($info.dpi != null) {
            $unit = 1;
            $dx = floor($info.dpi[0] + 0.5);
            $dy = floor($info.dpi[1] + 0.5);
            if ($dx < 1) { $dx = 1; }
            if ($dy < 1) { $dy = 1; }
        }
        push($out, _seg(224, "JFIF\x00" + chr(1) + chr(1) + chr($unit) +
                             _b16($dx) + _b16($dy) + chr(0) + chr(0)));
    }
    if ($cs == "CMYK") {
        // APP14, which is how a reader knows a 4-component file is CMYK and
        // that its values are inverted.
        push($out, _seg(238, "Adobe" + _b16(100) + _b16(0) + _b16(0) + chr(0)));
    }
    if ($info.exif != null) {
        $ex = $info.exif;
        if (len($ex) >= 6 && substr($ex, 0, 6) == "Exif\x00\x00") {
            push($out, _seg(225, $ex));
        } else {
            push($out, _seg(225, "Exif\x00\x00" + $ex));
        }
    }
    if ($info.icc_profile != null) {
        // An ICC profile can exceed a marker's 65533 bytes, so it goes out
        // in numbered APP2 segments the way the ICC specification says.
        $prof = $info.icc_profile;
        $room = 65533 - 2 - 14;
        $nseg = _ceilDiv(len($prof), $room);
        if ($nseg < 1) { $nseg = 1; }
        if ($nseg > 255) { throw "JPEG: an ICC profile needing " + str($nseg) + " segments"; }
        $k = 0;
        while ($k < $nseg) {
            $lo = $k * $room;
            $cut = len($prof) - $lo;
            if ($cut > $room) { $cut = $room; }
            push($out, _seg(226, "ICC_PROFILE\x00" + chr($k + 1) + chr($nseg) +
                                 substr($prof, $lo, $cut)));
            $k = $k + 1;
        }
    }
    if ($info.comment != null) { push($out, _seg(254, $info.comment)); }

    // DQT, one segment per table actually used.
    $used = {};
    $i = 0;
    while ($i < $nc) { $used[str($comps[$i].qno)] = true; $i = $i + 1; }
    $i = 0;
    while ($i < 4) {
        if ($used[str($i)] != null) {
            if ($qt[$i] == null) { throw "JPEG: quantisation table " + str($i) + " is missing"; }
            push($out, _dqtSeg($i, $qt[$i]));
        }
        $i = $i + 1;
    }

    // SOF0 (baseline) or SOF2 (progressive).
    $sof = chr(8) + _b16($h) + _b16($w) + chr($nc);
    $i = 0;
    while ($i < $nc) {
        $c = $comps[$i];
        $sof = $sof + chr($c.id) + chr($c.h * 16 + $c.v) + chr($c.qno);
        $i = $i + 1;
    }
    $sofCode = 192;
    if ($progressive) { $sofCode = 194; }
    push($out, _seg($sofCode, $sof));

    // The standard Annex K tables, unless `optimize` builds better ones.
    $stdDc = [{"bits": jt.STD_DC_LUMA_BITS, "vals": jt.STD_DC_LUMA_VALS},
              {"bits": jt.STD_DC_CHROMA_BITS, "vals": jt.STD_DC_CHROMA_VALS}];
    $stdAc = [{"bits": jt.STD_AC_LUMA_BITS, "vals": jt.STD_AC_LUMA_VALS},
              {"bits": jt.STD_AC_CHROMA_BITS, "vals": jt.STD_AC_CHROMA_VALS}];

    $si = 0;
    while ($si < len($scans)) {
        $sc = $scans[$si];
        $si = $si + 1;
        $scomps = $sc.comps;

        // Which tables this scan needs, and their definitions.
        $needDc = {};
        $needAc = {};
        $i = 0;
        while ($i < len($scomps)) {
            $c = $comps[$scomps[$i]];
            if ($sc.ss == 0) { $needDc[str($c.dcNo)] = true; }
            if (!$sc.dcOnly) { $needAc[str($c.acNo)] = true; }
            $i = $i + 1;
        }

        $dcTab = [null, null, null, null];
        $acTab = [null, null, null, null];
        if ($optimize) {
            // The counting pass: the same scan loop with emission off.
            $i = 0;
            while ($i < len($scomps)) {
                $c = $comps[$scomps[$i]];
                $c.dcEh = null;
                $c.acEh = null;
                $i = $i + 1;
            }
            if ($sc.dcOnly || $sc.ss == 0) {
                $cnt = _encSeqScan($fr, $scomps, $sc.dcOnly, true, null);
            } else {
                $cnt = _encAcScan($fr, $scomps[0], $sc.ss, $sc.se, true, null);
            }
            $F = $cnt.freq;
            for $k, $v in $needDc.items() {
                $no = num($k);
                $sub = [];
                $j = 0;
                while ($j < 256) { push($sub, $F[$no * 256 + $j]); $j = $j + 1; }
                $g = _genOptimal($sub);
                $dcTab[$no] = {"bits": $g[0], "vals": $g[1]};
            }
            for $k, $v in $needAc.items() {
                $no = num($k);
                $sub = [];
                $j = 0;
                while ($j < 256) { push($sub, $F[1024 + $no * 256 + $j]); $j = $j + 1; }
                $g = _genOptimal($sub);
                $acTab[$no] = {"bits": $g[0], "vals": $g[1]};
            }
        } else {
            for $k, $v in $needDc.items() {
                $no = num($k);
                $dcTab[$no] = $stdDc[$no % 2];
            }
            for $k, $v in $needAc.items() {
                $no = num($k);
                $acTab[$no] = $stdAc[$no % 2];
            }
        }

        // DHT for every table this scan uses, then the scan header.
        $i = 0;
        while ($i < 4) {
            if ($dcTab[$i] != null) {
                push($out, _dhtSeg(0, $i, $dcTab[$i].bits, $dcTab[$i].vals));
            }
            if ($acTab[$i] != null) {
                push($out, _dhtSeg(1, $i, $acTab[$i].bits, $acTab[$i].vals));
            }
            $i = $i + 1;
        }
        $i = 0;
        while ($i < len($scomps)) {
            $c = $comps[$scomps[$i]];
            if ($dcTab[$c.dcNo] != null) {
                $c.dcEh = _ehufOf($dcTab[$c.dcNo].bits, $dcTab[$c.dcNo].vals);
            }
            if ($acTab[$c.acNo] != null) {
                $c.acEh = _ehufOf($acTab[$c.acNo].bits, $acTab[$c.acNo].vals);
            }
            $i = $i + 1;
        }

        // DRI goes after the Huffman tables and before SOS, and only when
        // the interval changes -- jcmarker.c's write_scan_header, which is
        // why it is here and not beside the frame header.
        if ($restart > 0 && $si == 1) { push($out, _seg(221, _b16($restart))); }

        $sos = chr(len($scomps));
        $i = 0;
        while ($i < len($scomps)) {
            $c = $comps[$scomps[$i]];
            $dn = $c.dcNo;
            if ($sc.ss != 0) { $dn = 0; }
            $an = $c.acNo;
            if ($sc.dcOnly) { $an = 0; }
            $sos = $sos + chr($c.id) + chr($dn * 16 + $an);
            $i = $i + 1;
        }
        $sos = $sos + chr($sc.ss) + chr($sc.se) + chr(0);
        push($out, _seg(218, $sos));

        if ($sc.ss == 0) {
            $r = _encSeqScan($fr, $scomps, $sc.dcOnly, false, null);
        } else {
            $r = _encAcScan($fr, $scomps[0], $sc.ss, $sc.se, false, null);
        }
        push($out, $r.bytes);
    }

    push($out, _m(217));                                   // EOI
    return join($out, "");
}
