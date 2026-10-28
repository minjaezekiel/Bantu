// ════════════════════════════════════════════════════════════════════════
//  pillow/codecs/tga.b — Truevision TGA, read and write.
//
//  READS: image types 1 (colour-mapped), 2 (truecolour) and 3 (greyscale),
//  and their run-length-encoded forms 9, 10 and 11; 8, 15, 16, 24 and 32
//  bits per pixel; both vertical origins; an ID field of any length.
//
//  WRITES: 1, L, P, RGB and RGBA, uncompressed or RLE.
//
//  ─── WHAT MAKES TGA AWKWARD ──────────────────────────────────────────
//
//  * THERE IS NO MAGIC NUMBER. A TGA begins with its ID-field length, so
//    any byte is a legal first byte. sniff() therefore validates the whole
//    18-byte header for self-consistency instead of comparing a signature,
//    and the registry still falls back to the file extension. A format with
//    no magic is exactly why registry.identify sniffs first and only then
//    trusts the name.
//  * THE ORIGIN IS USUALLY AT THE BOTTOM. Bit 5 of the image descriptor is
//    set when the first row in the file is the TOP row; unset -- the common
//    case -- means bottom-up, like BMP.
//  * PIXELS ARE BGR(A), and a 15- or 16-bit pixel is 5-5-5 with the top bit
//    either ignored or an alpha flag.
//  * AN RLE PACKET MAY CROSS A ROW BOUNDARY. Decoding row by row and
//    resetting the run at each row's end is the classic bug, and it only
//    shows on images whose runs happen to span the edge.
//
//  Codecs deal in buffers, never Images -- see codecs/registry.b.
// ════════════════════════════════════════════════════════════════════════

include "../pxbuf.b" as pxbuf;

$MAX_PIXELS = 89478485;

def _u16($s, $i) { return ord($s[$i]) + ord($s[$i + 1]) * 256; }
def _p16($v) { return chr($v % 256) + chr(floor($v / 256) % 256); }

// sniff($head) -> is this a self-consistent TGA header?
//
// With no magic number the only test available is whether every field is
// legal and they agree with each other. That is weaker than a signature and
// it is stated as such: a file that passes may still not be a TGA, which is
// why the registry also matches on the extension.
def sniff($head) {
    if (len($head) < 18) { return false; }
    $cmapType = ord($head[1]);
    $imgType = ord($head[2]);
    $cmapDepth = ord($head[7]);
    $w = _u16($head, 12);
    $h = _u16($head, 14);
    $depth = ord($head[16]);
    $desc = ord($head[17]);
    if ($cmapType != 0 && $cmapType != 1) { return false; }
    if ($imgType != 1 && $imgType != 2 && $imgType != 3 &&
        $imgType != 9 && $imgType != 10 && $imgType != 11) { return false; }
    if ($w == 0 || $h == 0) { return false; }
    // Depth 1 is not in the TGA specification, but Pillow WRITES mode "1"
    // that way -- a greyscale image at one bit per pixel -- so a reader that
    // rejects it cannot read Pillow's own output.
    if ($depth != 1 && $depth != 8 && $depth != 15 && $depth != 16 &&
        $depth != 24 && $depth != 32) { return false; }
    if ($depth == 1 && $imgType != 3 && $imgType != 11) { return false; }
    // A colour-mapped image needs a map; a truecolour one must not claim a
    // map depth it will not use.
    if (($imgType == 1 || $imgType == 9) && $cmapType != 1) { return false; }
    if ($cmapType == 1 && $cmapDepth != 15 && $cmapDepth != 16 &&
        $cmapDepth != 24 && $cmapDepth != 32) { return false; }
    // bits 6-7 of the descriptor are reserved and interleaving is obsolete
    if (floor($desc / 64) % 4 != 0) { return false; }
    return true;
}

// ─── decode ─────────────────────────────────────────────────────────────
def decode($data) {
    $n = len($data);
    if ($n < 18) { throw "TGA: the header is 18 bytes, got " + str($n); }
    $idLen = ord($data[0]);
    $cmapType = ord($data[1]);
    $imgType = ord($data[2]);
    $cmapFirst = _u16($data, 3);
    $cmapLen = _u16($data, 5);
    $cmapDepth = ord($data[7]);
    $w = _u16($data, 12);
    $h = _u16($data, 14);
    $depth = ord($data[16]);
    $desc = ord($data[17]);
    $topDown = (floor($desc / 32) % 2 == 1);

    if ($w == 0 || $h == 0) { throw "TGA: bad dimensions " + str($w) + "x" + str($h); }
    if ($w > $MAX_PIXELS / $h) {
        throw "TGA: " + str($w) + "x" + str($h) + " is over the " + str($MAX_PIXELS) + "-pixel limit";
    }
    if ($depth != 1 && $depth != 8 && $depth != 15 && $depth != 16 &&
        $depth != 24 && $depth != 32) {
        throw "TGA: " + str($depth) + " bits per pixel is not supported (1, 8, 15, 16, 24, 32)";
    }
    $rle = ($imgType >= 9);
    $base = $imgType % 8;            // 1 mapped, 2 truecolour, 3 grey
    if ($base != 1 && $base != 2 && $base != 3) {
        throw "TGA: image type " + str($imgType) + " is not supported";
    }
    // Checked AFTER $base exists, not before it.
    if ($depth == 1 && $base != 3) {
        throw "TGA: one bit per pixel is only meaningful for a greyscale image";
    }

    $pos = 18 + $idLen;
    if ($pos > $n) { throw "TGA: the ID field runs past the end of the file"; }

    // ── the colour map ──
    $palette = null;
    if ($cmapType == 1) {
        $entryBytes = floor(($cmapDepth + 7) / 8);
        $need = $cmapLen * $entryBytes;
        if ($pos + $need > $n) {
            throw "TGA: the colour map needs " + str($need) + " bytes but the file ends";
        }
        $palette = nd_zeros([256, 3], "i64");
        $count = $cmapLen;
        if ($cmapFirst + $count > 256) { $count = 256 - $cmapFirst; }
        if ($count > 0) {
            $raw = nd_frombytes(substr($data, $pos, $count * $entryBytes), "u8",
                                [$count, $entryBytes]);
            if ($entryBytes >= 3) {
                // BGR(A) on the wire
                nd_add(nd_slice($raw, [null, [2, 3]]), 0,
                       nd_slice($palette, [[$cmapFirst, $cmapFirst + $count], [0, 1]]));
                nd_add(nd_slice($raw, [null, [1, 2]]), 0,
                       nd_slice($palette, [[$cmapFirst, $cmapFirst + $count], [1, 2]]));
                nd_add(nd_slice($raw, [null, [0, 1]]), 0,
                       nd_slice($palette, [[$cmapFirst, $cmapFirst + $count], [2, 3]]));
            } else {
                // a 15/16-bit map: 5-5-5 packed little-endian
                $v = nd_add(nd_slice($raw, [null, [0, 1]]),
                            nd_multiply(nd_slice($raw, [null, [1, 2]]), 256));
                $r = nd_multiply(nd_mod(nd_floor_divide($v, 1024), 32), 8);
                $g = nd_multiply(nd_mod(nd_floor_divide($v, 32), 32), 8);
                $b = nd_multiply(nd_mod($v, 32), 8);
                nd_add($r, 0, nd_slice($palette, [[$cmapFirst, $cmapFirst + $count], [0, 1]]));
                nd_add($g, 0, nd_slice($palette, [[$cmapFirst, $cmapFirst + $count], [1, 2]]));
                nd_add($b, 0, nd_slice($palette, [[$cmapFirst, $cmapFirst + $count], [2, 3]]));
            }
        }
        $pos = $pos + $need;
    }

    // ── the 1-bit case: rows packed MSB-first, padded to a whole byte ──
    //
    // Not an RLE path, because Pillow writes mode "1" uncompressed, and a
    // bit-packed RLE stream is not a thing any writer emits.
    if ($depth == 1) {
        $rowBytes = floor(($w + 7) / 8);
        $need = $rowBytes * $h;
        if ($pos + $need > $n) {
            throw "TGA: a 1-bit image needs " + str($need) + " bytes but only " +
                  str($n - $pos) + " remain";
        }
        $raw = nd_frombytes(substr($data, $pos, $need), "u8", [$h, $rowBytes]);
        $wide = $rowBytes * 8;
        $un = nd_zeros([$h, $wide], "i64");
        $bit = 0;
        while ($bit < 8) {
            $v = nd_mod(nd_floor_divide($raw, pow(2, 7 - $bit)), 2);
            nd_add(nd_multiply($v, 255), 0, nd_slice($un, [null, [$bit, $wide, 8]]));
            $bit = $bit + 1;
        }
        $px = nd_reshape(pxbuf.materialise2d(nd_slice($un, [null, [0, $w]])), [$h, $w, 1]);
        if (!$topDown) { $px = pxbuf.materialise(nd_flip($px, 0), "i64"); }
        return {"mode": "1", "size": [$w, $h], "px": $px, "palette": null, "info": {}};
    }

    // ── the pixels ──
    $bpp = floor(($depth + 7) / 8);
    $count = $w * $h;
    if ($rle) { $flat = _decodeRLE($data, $pos, $count, $bpp); }
    else {
        $need = $count * $bpp;
        if ($pos + $need > $n) {
            throw "TGA: the pixel data needs " + str($need) + " bytes but only " +
                  str($n - $pos) + " remain";
        }
        $flat = nd_frombytes(substr($data, $pos, $need), "u8", [$count, $bpp]);
    }

    // ── assemble ──
    if ($base == 1) {
        $px = nd_reshape(nd_slice($flat, [null, [0, 1]]), [$h, $w, 1]);
        $mode = "P";
    } else {
    if ($base == 3) {
        if ($depth == 8) {
            $px = nd_reshape(pxbuf.materialise2d(nd_slice($flat, [null, [0, 1]])), [$h, $w, 1]);
            $mode = "L";
        } else {
            throw "TGA: a greyscale image at " + str($depth) + " bits is not supported";
        }
    } else {
        if ($depth == 24) {
            $b = _col($flat, 0, $h, $w);
            $g = _col($flat, 1, $h, $w);
            $r = _col($flat, 2, $h, $w);
            $px = pxbuf.stack([$r, $g, $b], "i64");
            $mode = "RGB";
        } else {
        if ($depth == 32) {
            $b = _col($flat, 0, $h, $w);
            $g = _col($flat, 1, $h, $w);
            $r = _col($flat, 2, $h, $w);
            $a = _col($flat, 3, $h, $w);
            $px = pxbuf.stack([$r, $g, $b, $a], "i64");
            $mode = "RGBA";
        } else {
            // 15 or 16 bits: 5-5-5 little-endian, the top bit ignored
            $v = nd_add(_col($flat, 0, $h, $w), nd_multiply(_col($flat, 1, $h, $w), 256));
            $r = nd_multiply(nd_mod(nd_floor_divide($v, 1024), 32), 8);
            $g = nd_multiply(nd_mod(nd_floor_divide($v, 32), 32), 8);
            $b = nd_multiply(nd_mod($v, 32), 8);
            $px = pxbuf.stack([$r, $g, $b], "i64");
            $mode = "RGB";
        } }
    } }

    // Bit 5 of the descriptor set means top-down; unset -- the common case --
    // means the first row in the file is the BOTTOM row.
    if (!$topDown) { $px = pxbuf.materialise(nd_flip($px, 0), "i64"); }
    return {"mode": $mode, "size": [$w, $h], "px": $px, "palette": $palette, "info": {}};
}

def _col($flat, $i, $h, $w) {
    return nd_reshape(pxbuf.materialise2d(nd_slice($flat, [null, [$i, $i + 1]])), [$h, $w, 1]);
}

// _decodeRLE -> TGA's run-length packets.
//
// A packet's high bit says run (1) or literal (0) and its low 7 bits are the
// count minus one. Packets DO cross row boundaries, so this decodes the whole
// image as one stream rather than row by row -- resetting at each row is the
// classic TGA bug and it only shows when a run happens to span the edge.
def _decodeRLE($data, $pos, $count, $bpp) {
    $out = nd_zeros([$count, $bpp], "i64");
    $n = len($data);
    $p = $pos;
    $i = 0;
    while ($i < $count) {
        if ($p >= $n) { throw "TGA: the RLE stream ends after " + str($i) + " of " + str($count) + " pixels"; }
        $hdr = ord($data[$p]);
        $p = $p + 1;
        $run = floor($hdr / 128) % 2;
        $len = ($hdr % 128) + 1;
        if ($i + $len > $count) { $len = $count - $i; }
        if ($run == 1) {
            if ($p + $bpp > $n) { throw "TGA: an RLE run header is truncated at byte " + str($p); }
            $k = 0;
            while ($k < $bpp) {
                $v = ord($data[$p + $k]);
                $dst = nd_slice($out, [[$i, $i + $len], [$k, $k + 1]]);
                nd_multiply($dst, 0, $dst);
                nd_add($dst, $v, $dst);
                $k = $k + 1;
            }
            $p = $p + $bpp;
        } else {
            $need = $len * $bpp;
            if ($p + $need > $n) { throw "TGA: an RLE literal run is truncated at byte " + str($p); }
            $chunk = nd_frombytes(substr($data, $p, $need), "u8", [$len, $bpp]);
            nd_add($chunk, 0, nd_slice($out, [[$i, $i + $len], null]));
            $p = $p + $need;
        }
        $i = $i + $len;
    }
    return $out;
}

// ─── encode ─────────────────────────────────────────────────────────────
//
// $spec may carry "tga_rle" (default false) and "tga_topdown" (default
// false, i.e. bottom-up, which is what every TGA writer emits).
def encode($spec) {
    $m = $spec.mode;
    $px = $spec.px;
    $d = pxbuf.dims($px);
    $h = $d[0];
    $w = $d[1];
    if ($w == 0 || $h == 0) { throw "TGA: cannot write a zero-sized image"; }
    if ($w > 65535 || $h > 65535) {
        throw "TGA: dimensions are 16-bit, so " + str($w) + "x" + str($h) + " does not fit";
    }
    $rle = false;
    if ($spec.tga_rle != null) { $rle = $spec.tga_rle; }
    $topDown = false;
    if ($spec.tga_topdown != null) { $topDown = $spec.tga_topdown; }

    $src = $px;
    if (!$topDown) { $src = nd_flip($px, 0); }

    if ($m == "1") {
        // Pillow writes mode "1" as a GREYSCALE TGA at one bit per pixel,
        // rows packed MSB-first and padded to a byte. Writing it at 8 bits
        // would be larger and would read back as L.
        $rowBytes = floor(($w + 7) / 8);
        $padded = nd_zeros([$h, $rowBytes * 8], "i64");
        $bits1 = nd_floor_divide(nd_clip($src, 0, 255), 255);
        nd_add(nd_reshape($bits1, [$h, $w]), 0, nd_slice($padded, [null, [0, $w]]));
        $acc = nd_zeros([$h, $rowBytes], "i64");
        $k = 0;
        while ($k < 8) {
            $acc = nd_add($acc, nd_multiply(
                nd_slice($padded, [null, [$k, $rowBytes * 8, 8]]), pow(2, 7 - $k)));
            $k = $k + 1;
        }
        $desc1 = 0;
        if ($topDown) { $desc1 = 32; }
        $hdr1 = chr(0) + chr(0) + chr(3) + _p16(0) + _p16(0) + chr(0) +
                _p16(0) + _p16(0) + _p16($w) + _p16($h) + chr(1) + chr($desc1);
        return $hdr1 + nd_tobytes($acc, "u8");
    }
    if (false) {
        $imgType = 3;  $depth = 8;  $cmapType = 0;
        $flat = null;
        $palBytes = "";
    } else {
    if ($m == "L") {
        $imgType = 3;  $depth = 8;  $cmapType = 0;
        $flat = nd_reshape(pxbuf.materialise(nd_clip($src, 0, 255), "i64"), [$h * $w, 1]);
        $palBytes = "";
    } else {
    if ($m == "P") {
        if ($spec.palette == null) { throw "TGA: a P-mode image needs a palette to write"; }
        $imgType = 1;  $depth = 8;  $cmapType = 1;
        $flat = nd_reshape(pxbuf.materialise(nd_clip($src, 0, 255), "i64"), [$h * $w, 1]);
        // BGR on the wire
        $pb = nd_zeros([256, 3], "i64");
        nd_add(nd_slice($spec.palette, [null, [2, 3]]), 0, nd_slice($pb, [null, [0, 1]]));
        nd_add(nd_slice($spec.palette, [null, [1, 2]]), 0, nd_slice($pb, [null, [1, 2]]));
        nd_add(nd_slice($spec.palette, [null, [0, 1]]), 0, nd_slice($pb, [null, [2, 3]]));
        $palBytes = nd_tobytes($pb, "u8");
    } else {
    if ($m == "RGB") {
        $imgType = 2;  $depth = 24; $cmapType = 0;
        $flat = _bgr($src, $h, $w, 3);
        $palBytes = "";
    } else {
    if ($m == "RGBA") {
        $imgType = 2;  $depth = 32; $cmapType = 0;
        $flat = _bgr($src, $h, $w, 4);
        $palBytes = "";
    } else {
        throw "TGA: cannot write mode " + str($m) +
              ". Convert to 1, L, P, RGB or RGBA first.";
    } } } } }

    $bpp = floor($depth / 8);
    if ($rle) {
        $imgType = $imgType + 8;
        $pixels = _encodeRLE($flat, $h * $w, $bpp);
    } else {
        $pixels = nd_tobytes($flat, "u8");
    }

    $desc = 0;
    if ($topDown) { $desc = 32; }
    if ($depth == 32) { $desc = $desc + 8; }       // 8 alpha bits
    $cmapLen = 0;
    $cmapDepth = 0;
    if ($cmapType == 1) { $cmapLen = 256; $cmapDepth = 24; }

    $hdr = chr(0) + chr($cmapType) + chr($imgType) +
           _p16(0) + _p16($cmapLen) + chr($cmapDepth) +
           _p16(0) + _p16(0) + _p16($w) + _p16($h) +
           chr($depth) + chr($desc);
    return $hdr + $palBytes + $pixels;
}

def _bgr($src, $h, $w, $nb) {
    $flat = nd_zeros([$h * $w, $nb], "i64");
    $f = nd_reshape(pxbuf.materialise($src, "i64"), [$h * $w, pxbuf.nbands($src)]);
    nd_add(nd_clip(nd_slice($f, [null, [2, 3]]), 0, 255), 0, nd_slice($flat, [null, [0, 1]]));
    nd_add(nd_clip(nd_slice($f, [null, [1, 2]]), 0, 255), 0, nd_slice($flat, [null, [1, 2]]));
    nd_add(nd_clip(nd_slice($f, [null, [0, 1]]), 0, 255), 0, nd_slice($flat, [null, [2, 3]]));
    if ($nb == 4) {
        nd_add(nd_clip(nd_slice($f, [null, [3, 4]]), 0, 255), 0, nd_slice($flat, [null, [3, 4]]));
    }
    return $flat;
}

// _encodeRLE -> runs of up to 128 identical pixels, literals otherwise.
def _encodeRLE($flat, $count, $bpp) {
    $vals = nd_to_list(nd_ravel($flat));
    $parts = [];
    $i = 0;
    while ($i < $count) {
        // how many identical pixels follow?
        $run = 1;
        while ($run < 128 && $i + $run < $count && _same($vals, $i, $i + $run, $bpp)) {
            $run = $run + 1;
        }
        if ($run >= 2) {
            push($parts, chr(128 + $run - 1));
            $k = 0;
            while ($k < $bpp) { push($parts, chr($vals[$i * $bpp + $k])); $k = $k + 1; }
            $i = $i + $run;
        } else {
            // a literal packet, ending as soon as a run of 2 appears
            $lit = 1;
            while ($lit < 128 && $i + $lit < $count &&
                   !_same($vals, $i + $lit, $i + $lit - 1, $bpp)) {
                $lit = $lit + 1;
            }
            push($parts, chr($lit - 1));
            $k = 0;
            while ($k < $lit * $bpp) { push($parts, chr($vals[$i * $bpp + $k])); $k = $k + 1; }
            $i = $i + $lit;
        }
    }
    return join($parts, "");
}

def _same($vals, $a, $b, $bpp) {
    $k = 0;
    while ($k < $bpp) {
        if ($vals[$a * $bpp + $k] != $vals[$b * $bpp + $k]) { return false; }
        $k = $k + 1;
    }
    return true;
}
