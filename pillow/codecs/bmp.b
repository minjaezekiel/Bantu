// ════════════════════════════════════════════════════════════════════════
//  pillow/codecs/bmp.b — Windows BMP / DIB, read and write.
//
//  READS: 1, 4, 8, 16, 24 and 32 bits per pixel; BITMAPCOREHEADER (12),
//  BITMAPINFOHEADER (40) and the V2/V3/V4/V5 extensions (52/56/108/124);
//  uncompressed, RLE4, RLE8 and BITFIELDS; bottom-up and top-down.
//
//  WRITES: 1, L, P and RGB at their natural depths.
//
//  ─── THE THREE THINGS A BMP READER GETS WRONG ────────────────────────
//
//  1. ROWS ARE BOTTOM-UP. A positive height in the DIB header means the
//     first row in the file is the BOTTOM row of the image. A negative
//     height means top-down. Reading a BMP upside down is the classic bug,
//     and it looks plausible enough on a symmetric test image to survive.
//  2. ROWS ARE PADDED TO FOUR BYTES. An 11-pixel 24-bit row is 33 bytes of
//     pixels and 36 bytes in the file. Using the pixel width as the stride
//     shears the image diagonally.
//  3. PALETTE ENTRIES ARE BGRA, AND PIXELS ARE BGR. Everything in a BMP is
//     byte-reversed relative to the name of the format it is stored in.
//
//  Pillow's BMP writer DROPS ALPHA -- saving an RGBA image and reading it
//  back gives RGB -- so this does the same rather than inventing a 32-bit
//  variant Pillow would not read as RGBA anyway.
//
//  Codecs deal in buffers, never Images -- see codecs/registry.b.
// ════════════════════════════════════════════════════════════════════════

include "../pxbuf.b" as pxbuf;

$MAX_PIXELS = 89478485;

def _u16($s, $i) { return ord($s[$i]) + ord($s[$i + 1]) * 256; }
def _u32($s, $i) {
    return ord($s[$i]) + ord($s[$i + 1]) * 256 +
           ord($s[$i + 2]) * 65536 + ord($s[$i + 3]) * 16777216;
}
def _i32($s, $i) {
    $v = _u32($s, $i);
    if ($v > 2147483647) { return $v - 4294967296; }
    return $v;
}
def _p16($v) { return chr($v % 256) + chr(floor($v / 256) % 256); }
def _p32($v) {
    $x = $v;
    if ($x < 0) { $x = $x + 4294967296; }
    return chr($x % 256) + chr(floor($x / 256) % 256) +
           chr(floor($x / 65536) % 256) + chr(floor($x / 16777216) % 256);
}

def sniff($head) {
    if (len($head) < 2) { return false; }
    // "BM" is the only variant in practice; BA/CI/CP/IC/PT are OS/2 archives
    // that no current writer emits and that Pillow does not read either.
    return substr($head, 0, 2) == "BM";
}

// ─── decode ─────────────────────────────────────────────────────────────
def decode($data) {
    $n = len($data);
    if ($n < 14) { throw "BMP: the file header is 14 bytes, got " + str($n); }
    if (!sniff($data)) { throw "BMP: not a BMP file (expected 'BM')"; }
    $dataOffset = _u32($data, 10);
    if ($n < 18) { throw "BMP: the file ends inside the DIB header"; }
    $hdrSize = _u32($data, 14);

    if ($hdrSize == 12) {
        // BITMAPCOREHEADER: 16-bit dimensions, no compression field.
        if ($n < 26) { throw "BMP: the file ends inside a 12-byte DIB header"; }
        $w = _u16($data, 18);
        $h = _u16($data, 20);
        $planes = _u16($data, 22);
        $bits = _u16($data, 24);
        $comp = 0;
        $palCount = 0;
        $palEntry = 3;              // a core header's palette is RGB triples
        $topDown = false;
    } else {
        if ($hdrSize < 40) {
            throw "BMP: a DIB header of " + str($hdrSize) + " bytes is not a known variant";
        }
        if ($n < 14 + $hdrSize) { throw "BMP: the file ends inside the DIB header"; }
        $w = _i32($data, 18);
        $rawH = _i32($data, 22);
        $planes = _u16($data, 26);
        $bits = _u16($data, 28);
        $comp = _u32($data, 30);
        $palCount = _u32($data, 46);
        $palEntry = 4;
        $topDown = ($rawH < 0);
        $h = $rawH;
        if ($topDown) { $h = 0 - $rawH; }
    }

    if ($w <= 0 || $h <= 0) {
        throw "BMP: bad dimensions " + str($w) + "x" + str($h);
    }
    if ($planes != 1) { throw "BMP: " + str($planes) + " colour planes; only 1 is legal"; }
    // The pixel budget, from the header, before anything is allocated.
    if ($w > $MAX_PIXELS / $h) {
        throw "BMP: " + str($w) + "x" + str($h) + " is over the " + str($MAX_PIXELS) + "-pixel limit";
    }

    if ($bits != 1 && $bits != 4 && $bits != 8 && $bits != 16 && $bits != 24 && $bits != 32) {
        throw "BMP: " + str($bits) + " bits per pixel is not supported (1, 4, 8, 16, 24, 32)";
    }
    if ($comp != 0 && $comp != 1 && $comp != 2 && $comp != 3) {
        throw "BMP: compression method " + str($comp) + " is not supported";
    }
    if ($comp == 1 && $bits != 8) { throw "BMP: RLE8 needs 8 bits per pixel"; }
    if ($comp == 2 && $bits != 4) { throw "BMP: RLE4 needs 4 bits per pixel"; }

    // ── the palette, for 1/4/8 bits ──
    $palette = null;
    $info = {};
    if ($bits <= 8) {
        $entries = $palCount;
        if ($entries == 0) { $entries = pow(2, $bits); }
        if ($entries > 256) { $entries = 256; }
        $palStart = 14 + $hdrSize;
        $need = $entries * $palEntry;
        if ($palStart + $need > $n) {
            throw "BMP: the palette needs " + str($need) + " bytes but the file ends";
        }
        $palette = nd_zeros([256, 3], "i64");
        $raw = nd_frombytes(substr($data, $palStart, $need), "u8", [$entries, $palEntry]);
        // BGRA on the wire, RGB in the palette.
        nd_add(nd_slice($raw, [null, [2, 3]]), 0, nd_slice($palette, [[0, $entries], [0, 1]]));
        nd_add(nd_slice($raw, [null, [1, 2]]), 0, nd_slice($palette, [[0, $entries], [1, 2]]));
        nd_add(nd_slice($raw, [null, [0, 1]]), 0, nd_slice($palette, [[0, $entries], [2, 3]]));
    }

    // ── locate the pixels ──
    $off = $dataOffset;
    if ($off == 0) {
        $off = 14 + $hdrSize;
        if ($bits <= 8) {
            $entries = $palCount;
            if ($entries == 0) { $entries = pow(2, $bits); }
            $off = $off + $entries * $palEntry;
        }
    }
    if ($off > $n) { throw "BMP: the pixel offset " + str($off) + " is past the end of the file"; }

    if ($comp == 1 || $comp == 2) {
        $idx = _decodeRLE($data, $off, $w, $h, $bits);
        $px = $idx;
        $mode = "P";
        if ($topDown) { $px = pxbuf.materialise(nd_flip($px, 0), "i64"); }
        return {"mode": $mode, "size": [$w, $h], "px": $px, "palette": $palette, "info": $info};
    }

    // ── uncompressed: rows padded to a multiple of 4 bytes ──
    $rowBits = $w * $bits;
    $rowBytes = floor(($rowBits + 31) / 32) * 4;
    $need = $rowBytes * $h;
    if ($off + $need > $n) {
        throw "BMP: the pixel data needs " + str($need) + " bytes from offset " +
              str($off) + " but only " + str($n - $off) + " remain";
    }
    $rows = nd_frombytes(substr($data, $off, $need), "u8", [$h, $rowBytes]);

    if ($bits == 24) {
        $b = nd_slice($rows, [null, [0, $w * 3, 3]]);
        $g = nd_slice($rows, [null, [1, $w * 3, 3]]);
        $r = nd_slice($rows, [null, [2, $w * 3, 3]]);
        $px = pxbuf.stack([_c($r, $h, $w), _c($g, $h, $w), _c($b, $h, $w)], "i64");
        $mode = "RGB";
    } else {
    if ($bits == 32) {
        $b = nd_slice($rows, [null, [0, $w * 4, 4]]);
        $g = nd_slice($rows, [null, [1, $w * 4, 4]]);
        $r = nd_slice($rows, [null, [2, $w * 4, 4]]);
        // A plain 32-bit BMP has an ignored fourth byte; only BITFIELDS with
        // an alpha mask makes it meaningful, and Pillow reads the common case
        // as RGB. Matched, so a round trip does not invent transparency.
        $px = pxbuf.stack([_c($r, $h, $w), _c($g, $h, $w), _c($b, $h, $w)], "i64");
        $mode = "RGB";
    } else {
    if ($bits == 16) {
        // 5-5-5 by default; BITFIELDS can say 5-6-5, and that is the only
        // masking case worth carrying because it is what cameras emit.
        $lo = nd_slice($rows, [null, [0, $w * 2, 2]]);
        $hi = nd_slice($rows, [null, [1, $w * 2, 2]]);
        $v = nd_add(_c($lo, $h, $w), nd_multiply(_c($hi, $h, $w), 256));
        $g565 = false;
        if ($comp == 3 && 14 + $hdrSize + 12 <= $n) {
            $gmask = _u32($data, 14 + $hdrSize + 4);
            if ($gmask == 2016) { $g565 = true; }        // 0x07E0
        }
        if ($g565) {
            $r = nd_mod(nd_floor_divide($v, 2048), 32);
            $g = nd_mod(nd_floor_divide($v, 32), 64);
            $b = nd_mod($v, 32);
            $px = pxbuf.stack([nd_multiply($r, 8), nd_multiply($g, 4), nd_multiply($b, 8)], "i64");
        } else {
            $r = nd_mod(nd_floor_divide($v, 1024), 32);
            $g = nd_mod(nd_floor_divide($v, 32), 32);
            $b = nd_mod($v, 32);
            $px = pxbuf.stack([nd_multiply($r, 8), nd_multiply($g, 8), nd_multiply($b, 8)], "i64");
        }
        $mode = "RGB";
    } else {
        // 1, 4 or 8 bits: palette indices, packed most significant first.
        $px = _unpack($rows, $w, $h, $bits);
        $mode = "P";
        // Pillow does not report a palette image as P when the palette says
        // the image is really grey: a 1-bit black-and-white palette gives
        // mode "1", and a full 256-entry grey ramp gives mode "L". Returning
        // P for those is not wrong about the PIXELS -- for a grey ramp the
        // index equals the grey level, so the bytes are identical -- but it
        // is wrong about the mode, and every later operation would then carry
        // a palette it does not need.
        if ($bits == 1 && _isBlackWhite($palette)) {
            $px = nd_multiply($px, 255);
            $mode = "1";
            $palette = null;
        } else {
            if ($bits == 8 && _isGreyRamp($palette)) {
                $mode = "L";
                $palette = null;
            }
        }
    } } }

    // Positive height means the FIRST row in the file is the BOTTOM row.
    if (!$topDown) { $px = pxbuf.materialise(nd_flip($px, 0), "i64"); }
    return {"mode": $mode, "size": [$w, $h], "px": $px, "palette": $palette, "info": $info};
}

def _c($a, $h, $w) { return nd_reshape(pxbuf.materialise2d($a), [$h, $w, 1]); }

// _isGreyRamp($pal) -> is every entry (i, i, i)?
//
// Checked with array operations, not a 256-step loop: the ramp is compared
// against nd_arange broadcast across the three channels, so it is three
// comparisons whatever the palette holds.
def _isGreyRamp($pal) {
    if ($pal == null) { return false; }
    $ramp = nd_broadcast_to(nd_reshape(nd_arange(0, 256, 1), [256, 1]), [256, 3]);
    return pxbuf.scalar(nd_all(nd_equal($pal, $ramp)));
}

def _isBlackWhite($pal) {
    if ($pal == null) { return false; }
    if (nd_get($pal, [0, 0]) != 0 || nd_get($pal, [0, 1]) != 0 || nd_get($pal, [0, 2]) != 0) { return false; }
    return nd_get($pal, [1, 0]) == 255 && nd_get($pal, [1, 1]) == 255 && nd_get($pal, [1, 2]) == 255;
}

// _unpack -> sub-byte indices, most significant first. The loop is over BITS.
def _unpack($rows, $w, $h, $bits) {
    $per = floor(8 / $bits);
    $cols = floor(($w + $per - 1) / $per);
    $wide = $cols * $per;
    $out = nd_zeros([$h, $wide], "i64");
    $k = 0;
    while ($k < $per) {
        $shift = pow(2, $bits * ($per - 1 - $k));
        $v = nd_mod(nd_floor_divide(nd_slice($rows, [null, [0, $cols]]), $shift), pow(2, $bits));
        nd_add($v, 0, nd_slice($out, [null, [$k, $wide, $per]]));
        $k = $k + 1;
    }
    return nd_reshape(pxbuf.materialise2d(nd_slice($out, [null, [0, $w]])), [$h, $w, 1]);
}

// _decodeRLE -> RLE8 (comp 1) and RLE4 (comp 2).
//
// Sequential by nature: a run's length decides where the next one starts.
// The loop is over the COMPRESSED bytes, which §6 of the architecture
// permits; it never loops over pixels of the output except to fill a run.
def _decodeRLE($data, $off, $w, $h, $bits) {
    $out = nd_zeros([$h, $w, 1], "i64");
    $row = [];
    $i = 0;
    while ($i < $w) { push($row, 0); $i = $i + 1; }
    $n = len($data);
    $p = $off;
    $x = 0;
    $y = 0;
    $done = false;
    while (!$done) {
        if ($p + 1 >= $n) { $done = true; }
        else {
            $a = ord($data[$p]);
            $b = ord($data[$p + 1]);
            $p = $p + 2;
            if ($a > 0) {
                // an encoded run of `a` pixels
                $k = 0;
                while ($k < $a) {
                    if ($x < $w) {
                        if ($bits == 8) { $row[$x] = $b; }
                        else {
                            if ($k % 2 == 0) { $row[$x] = floor($b / 16); }
                            else { $row[$x] = $b % 16; }
                        }
                        $x = $x + 1;
                    }
                    $k = $k + 1;
                }
            } else {
                if ($b == 0) {
                    // end of line
                    _flushRow($out, $row, $y, $w, $h);
                    $y = $y + 1;
                    $x = 0;
                    $i = 0;
                    while ($i < $w) { $row[$i] = 0; $i = $i + 1; }
                    if ($y >= $h) { $done = true; }
                } else {
                    if ($b == 1) {
                        // end of bitmap
                        _flushRow($out, $row, $y, $w, $h);
                        $done = true;
                    } else {
                        if ($b == 2) {
                            // a delta: skip dx, dy
                            if ($p + 1 >= $n) { $done = true; }
                            else {
                                $dx = ord($data[$p]);
                                $dy = ord($data[$p + 1]);
                                $p = $p + 2;
                                _flushRow($out, $row, $y, $w, $h);
                                $x = $x + $dx;
                                $y = $y + $dy;
                                if ($y >= $h) { $done = true; }
                            }
                        } else {
                            // an absolute run of `b` pixels
                            if ($bits == 8) { $take = $b; } else { $take = floor(($b + 1) / 2); }
                            if ($p + $take > $n) { $done = true; }
                            else {
                                $k = 0;
                                while ($k < $b) {
                                    if ($x < $w) {
                                        if ($bits == 8) { $row[$x] = ord($data[$p + $k]); }
                                        else {
                                            $byte = ord($data[$p + floor($k / 2)]);
                                            if ($k % 2 == 0) { $row[$x] = floor($byte / 16); }
                                            else { $row[$x] = $byte % 16; }
                                        }
                                        $x = $x + 1;
                                    }
                                    $k = $k + 1;
                                }
                                // runs are padded to a 16-bit boundary
                                $p = $p + $take;
                                if ($take % 2 == 1) { $p = $p + 1; }
                            }
                        }
                    }
                }
            }
        }
    }
    return $out;
}

def _flushRow($out, $row, $y, $w, $h) {
    if ($y < 0 || $y >= $h) { return false; }
    nd_add(nd_reshape(nd($row, "i64"), [1, $w, 1]), 0,
           nd_slice($out, [[$y, $y + 1], null, null]));
    return true;
}

// ─── encode ─────────────────────────────────────────────────────────────
def encode($spec) {
    $m = $spec.mode;
    $px = $spec.px;
    $d = pxbuf.dims($px);
    $h = $d[0];
    $w = $d[1];
    if ($w == 0 || $h == 0) { throw "BMP: cannot write a zero-sized image"; }

    if ($m == "1")   { $bits = 1;  $pal = _bwPalette(); $idx = nd_floor_divide(nd_clip($px, 0, 255), 255); }
    else { if ($m == "L") { $bits = 8; $pal = _greyPalette(); $idx = nd_clip($px, 0, 255); }
    else { if ($m == "P") {
             $bits = 8;
             if ($spec.palette == null) { throw "BMP: a P-mode image needs a palette to write"; }
             $pal = $spec.palette;
             $idx = nd_clip($px, 0, 255);
           }
    else { if ($m == "RGB" || $m == "RGBA") { $bits = 24; $pal = null; $idx = null; }
    else {
        throw "BMP: cannot write mode " + str($m) +
              ". Convert to 1, L, P or RGB first (Pillow has the same restriction, " +
              "and its BMP writer drops alpha).";
    } } } }

    // rows bottom-up, padded to 4 bytes
    if ($bits == 24) {
        $rowBytes = floor(($w * 3 + 3) / 4) * 4;
        $buf = nd_zeros([$h, $rowBytes], "i64");
        $src = nd_flip(pxbuf.bands_from($px, 0, 3), 0);
        $b = nd_reshape(pxbuf.materialise2d(nd_reshape(nd_slice($src, [null, null, [2, 3]]), [$h, $w])), [$h, $w]);
        $g = nd_reshape(pxbuf.materialise2d(nd_reshape(nd_slice($src, [null, null, [1, 2]]), [$h, $w])), [$h, $w]);
        $r = nd_reshape(pxbuf.materialise2d(nd_reshape(nd_slice($src, [null, null, [0, 1]]), [$h, $w])), [$h, $w]);
        nd_add(nd_clip($b, 0, 255), 0, nd_slice($buf, [null, [0, $w * 3, 3]]));
        nd_add(nd_clip($g, 0, 255), 0, nd_slice($buf, [null, [1, $w * 3, 3]]));
        nd_add(nd_clip($r, 0, 255), 0, nd_slice($buf, [null, [2, $w * 3, 3]]));
        $pixels = nd_tobytes($buf, "u8");
        $palBytes = "";
        $palEntries = 0;
    } else {
        if ($bits == 1) {
            $cols = floor(($w + 7) / 8);
            $rowBytes = floor(($cols + 3) / 4) * 4;
            $padded = nd_zeros([$h, $cols * 8], "i64");
            nd_add(nd_reshape(nd_flip($idx, 0), [$h, $w]), 0, nd_slice($padded, [null, [0, $w]]));
            $acc = nd_zeros([$h, $cols], "i64");
            $k = 0;
            while ($k < 8) {
                $acc = nd_add($acc, nd_multiply(nd_slice($padded, [null, [$k, $cols * 8, 8]]), pow(2, 7 - $k)));
                $k = $k + 1;
            }
            $buf = nd_zeros([$h, $rowBytes], "i64");
            nd_add($acc, 0, nd_slice($buf, [null, [0, $cols]]));
            $pixels = nd_tobytes($buf, "u8");
        } else {
            $rowBytes = floor(($w + 3) / 4) * 4;
            $buf = nd_zeros([$h, $rowBytes], "i64");
            nd_add(nd_reshape(nd_flip($idx, 0), [$h, $w]), 0, nd_slice($buf, [null, [0, $w]]));
            $pixels = nd_tobytes($buf, "u8");
        }
        $palEntries = pow(2, $bits);
        if ($palEntries > 256) { $palEntries = 256; }
        // BGRA on the wire
        $pb = nd_zeros([$palEntries, 4], "i64");
        nd_add(nd_slice($pal, [[0, $palEntries], [2, 3]]), 0, nd_slice($pb, [null, [0, 1]]));
        nd_add(nd_slice($pal, [[0, $palEntries], [1, 2]]), 0, nd_slice($pb, [null, [1, 2]]));
        nd_add(nd_slice($pal, [[0, $palEntries], [0, 1]]), 0, nd_slice($pb, [null, [2, 3]]));
        $palBytes = nd_tobytes($pb, "u8");
    }

    $offset = 14 + 40 + len($palBytes);
    $total = $offset + len($pixels);
    $fileHdr = "BM" + _p32($total) + _p32(0) + _p32($offset);
    $dib = _p32(40) + _p32($w) + _p32($h) + _p16(1) + _p16($bits) +
           _p32(0) + _p32(len($pixels)) + _p32(0) + _p32(0) +
           _p32($palEntries) + _p32(0);
    return $fileHdr + $dib + $palBytes + $pixels;
}

def _bwPalette() {
    $p = nd_zeros([256, 3], "i64");
    nd_add(nd_zeros([1, 3], "i64"), 255, nd_slice($p, [[1, 2], null]));
    return $p;
}
def _greyPalette() {
    $p = nd_zeros([256, 3], "i64");
    $ramp = nd_reshape(nd_arange(0, 256, 1), [256, 1]);
    nd_add(nd_broadcast_to($ramp, [256, 3]), 0, $p);
    return $p;
}
