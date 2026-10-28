// ════════════════════════════════════════════════════════════════════════
//  pillow/codecs/png.b — PNG, read and write.
//
//  The format pillow has to support before any other, and the one phase P1's
//  inflate/deflate atoms existed for: a PNG's pixels are a zlib stream, so
//  without a decompressor the format was unreadable from Bantu at all.
//
//  READS: all five colour types (0 grey, 2 truecolour, 3 indexed, 4 grey+
//  alpha, 6 truecolour+alpha), bit depths 1, 2, 4, 8 and 16, interlaced
//  (Adam7) and not, the five scanline filters, PLTE, tRNS, gAMA, pHYs, sRGB,
//  bKGD, tEXt, zTXt and iTXt.
//
//  WRITES: 1, L, LA, P, PA, RGB, RGBA and I;16, with a chosen filter and
//  compression level.
//
//  ─── THE MODE TABLE, from Pillow's PngImagePlugin._MODES ─────────────
//
//      bits  type           mode     note
//        1     0  grey      "1"
//        2     0  grey      "L"      unpacked and SCALED to 0..255
//        4     0  grey      "L"      unpacked and scaled
//        8     0  grey      "L"
//       16     0  grey      "I;16"
//        8     2  rgb       "RGB"
//       16     2  rgb       "RGB"    the low byte of each sample is dropped
//        1     3  indexed   "P"
//        2     3  indexed   "P"
//        4     3  indexed   "P"
//        8     3  indexed   "P"
//        8     4  grey+a    "LA"
//       16     4  grey+a    "RGBA"   Pillow has no 16-bit LA, so it widens
//        8     6  rgb+a     "RGBA"
//       16     6  rgb+a     "RGBA"
//
//  The two surprises are both Pillow's, and both are matched: 16-bit
//  truecolour comes back as 8-bit RGB (the low byte of every sample is
//  thrown away), and 16-bit grey+alpha comes back as RGBA rather than LA.
//
//  ─── UNFILTERING, AND WHY IT IS THE SLOW PART ────────────────────────
//
//  Each scanline carries a filter byte, and the five filters are not equally
//  tractable:
//
//      0 None     nothing to do
//      1 Sub      row[i] += row[i - bpp]       a strided prefix sum
//      2 Up       row[i] += prev[i]            one array add
//      3 Average  row[i] += (left + up) >> 1   sequential: left is needed
//      4 Paeth    a data-dependent predictor   sequential
//
//  None, Sub and Up are whole-row array operations -- Sub is `nd_cumsum`
//  along the strided channel lane, modulo 256, which is the trick that makes
//  it vectorisable at all. Average and Paeth genuinely cannot be: each byte
//  needs the byte to its left, which needs the one to ITS left. They are the
//  documented exception to rule PW2, like the dither, and they are the
//  reason a photographic PNG is seconds rather than milliseconds. The whole-
//  image fast path (every row filter 0) skips unfiltering entirely.
//
//  Codecs deal in buffers, never Images -- see codecs/registry.b.
// ════════════════════════════════════════════════════════════════════════

include "../pxbuf.b" as pxbuf;
include "../inflate.b" as binflate;
include "../deflate.b" as bdeflate;

$SIGNATURE = "\x89PNG\r\n\x1a\n";
$MAX_PIXELS = 89478485;

// Pillow's LOAD_TRUNCATED_IMAGES: when on, a bad CRC is tolerated so a
// partially-written file still yields what it has. Off by default, because
// silently decoding corrupt data is how a bad file comes to look like a good
// one. Set it deliberately with png.allow_truncated(true).
$LOAD_TRUNCATED = {"on": false};
def allow_truncated($yes) { $LOAD_TRUNCATED.on = $yes; return $yes; }
def truncated_allowed() { return $LOAD_TRUNCATED.on; }

// Does the interpreter have the native codecs? pillow works either way; the
// Bantu ones are 400x slower and exactly as correct.
$_NAT = {"inflate": false, "crc": false, "zlib": false};
try { $_NAT.inflate = has_native("inflate"); } catch ($e) { $_NAT.inflate = false; }
try { $_NAT.crc = has_native("raster"); } catch ($e) { $_NAT.crc = false; }
$_NAT.zlib = $_NAT.crc;

def _inflate($data, $maxOut) {
    if ($_NAT.inflate) { return inflate($data, $maxOut); }
    return binflate.inflate_str($data, $maxOut);
}
def _crc32($data) {
    if ($_NAT.crc) { return bp_crc32($data); }
    return bdeflate.crc32($data);
}
def _compress($data, $level) {
    if ($_NAT.zlib && ($level == null || $level >= 6)) { return bp_zlib($data); }
    return bdeflate.zlib($data, $level);
}

// ─── Byte helpers ───────────────────────────────────────────────────────

def _u32($s, $i) {
    return ord($s[$i]) * 16777216 + ord($s[$i + 1]) * 65536 +
           ord($s[$i + 2]) * 256 + ord($s[$i + 3]);
}

def _p32($v) {
    $x = $v;
    if ($x < 0) { $x = $x + 4294967296; }
    return chr(floor($x / 16777216) % 256) + chr(floor($x / 65536) % 256) +
           chr(floor($x / 256) % 256) + chr($x % 256);
}

def sniff($head) {
    if (len($head) < 8) { return false; }
    return substr($head, 0, 8) == $SIGNATURE;
}

// ─── The mode table ─────────────────────────────────────────────────────
//
// _modeFor($bits, $colour) -> {"mode", "channels", "scale"} or null.
// `channels` is how many samples each pixel has IN THE FILE; `scale` is the
// multiplier that turns a sub-byte grey level into 0..255, which is how
// Pillow expands bit depths 2 and 4.
def _modeFor($bits, $colour) {
    if ($colour == 0) {
        if ($bits == 1)  { return {"mode": "1",    "channels": 1, "scale": 255}; }
        if ($bits == 2)  { return {"mode": "L",    "channels": 1, "scale": 85}; }
        if ($bits == 4)  { return {"mode": "L",    "channels": 1, "scale": 17}; }
        if ($bits == 8)  { return {"mode": "L",    "channels": 1, "scale": 1}; }
        if ($bits == 16) { return {"mode": "I;16", "channels": 1, "scale": 1}; }
        return null;
    }
    if ($colour == 2) {
        if ($bits == 8)  { return {"mode": "RGB", "channels": 3, "scale": 1}; }
        if ($bits == 16) { return {"mode": "RGB", "channels": 3, "scale": 1}; }
        return null;
    }
    if ($colour == 3) {
        if ($bits == 1 || $bits == 2 || $bits == 4 || $bits == 8) {
            return {"mode": "P", "channels": 1, "scale": 1};
        }
        return null;
    }
    if ($colour == 4) {
        if ($bits == 8)  { return {"mode": "LA",   "channels": 2, "scale": 1}; }
        // Pillow has no 16-bit LA, so it widens to RGBA.
        if ($bits == 16) { return {"mode": "RGBA", "channels": 2, "scale": 1}; }
        return null;
    }
    if ($colour == 6) {
        if ($bits == 8)  { return {"mode": "RGBA", "channels": 4, "scale": 1}; }
        if ($bits == 16) { return {"mode": "RGBA", "channels": 4, "scale": 1}; }
        return null;
    }
    return null;
}

// ─── Chunk walking ──────────────────────────────────────────────────────
//
// Every read is bounds-checked BEFORE it happens and every failure names the
// chunk and the offset, because this is a parser of hostile input.
def _chunks($data) {
    $n = len($data);
    if ($n < 8 || !sniff($data)) { throw "PNG: not a PNG file (bad signature)"; }
    $out = [];
    $pos = 8;
    $seenIEND = false;
    while ($pos < $n) {
        if ($pos + 8 > $n) {
            throw "PNG: a chunk header runs past the end of the file at byte " + str($pos);
        }
        $len = _u32($data, $pos);
        $type = substr($data, $pos + 4, 4);
        // A length field is 31 bits by the spec, and a hostile file will
        // claim 4 GB to make the reader allocate it.
        if ($len > 2147483647) {
            throw "PNG: chunk '" + $type + "' at byte " + str($pos) +
                  " declares an illegal length of " + str($len);
        }
        if ($pos + 12 + $len > $n) {
            throw "PNG: chunk '" + $type + "' at byte " + str($pos) + " declares " +
                  str($len) + " bytes but only " + str($n - $pos - 12) + " remain";
        }
        $body = "";
        if ($len > 0) { $body = substr($data, $pos + 8, $len); }

        // EVERY CHUNK'S CRC IS VERIFIED. The CRC is the only thing standing
        // between a corrupted file and a decoder that trusts its header, and
        // PngSuite's xhdn0g08.png is exactly that case: a valid-looking IHDR
        // with a wrong checksum, which decodes to a plausible but wrong
        // image if the CRC is skipped. Pillow rejects it; so does this.
        //
        // The CRC covers the TYPE as well as the data, which is easy to get
        // wrong and would make every check fail.
        if (!$LOAD_TRUNCATED.on) {
            $want = _u32($data, $pos + 8 + $len);
            $got = _crc32($type + $body);
            if ($got != $want) {
                throw "PNG: chunk '" + $type + "' at byte " + str($pos) +
                      " has a bad CRC (file says " + str($want) + ", data gives " +
                      str($got) + "). The file is corrupt.";
            }
        }
        push($out, {"type": $type, "data": $body, "at": $pos});
        if ($type == "IEND") { $seenIEND = true; $pos = $n; }
        else { $pos = $pos + 12 + $len; }
    }
    if (!$seenIEND) { throw "PNG: the file ends without an IEND chunk"; }
    return $out;
}

// ─── Unfiltering ────────────────────────────────────────────────────────
//
// _unfilter($raw, $h, $rowBytes, $bpp) -> an (h, rowBytes) i64 array.
//
// `bpp` is the byte distance to the pixel on the left -- ceil(bits*channels/8),
// at least 1 -- which is what Sub, Average and Paeth step by.
def _unfilter($raw, $h, $rowBytes, $bpp) {
    $stride = $rowBytes + 1;
    $need = $stride * $h;
    if (len($raw) < $need) {
        throw "PNG: the decompressed image is " + str(len($raw)) + " bytes, " +
              str($need) + " are needed for " + str($h) + " rows of " + str($rowBytes);
    }
    if ($h == 0 || $rowBytes == 0) { return nd_zeros([max($h, 0), max($rowBytes, 0)], "i64"); }

    $all = nd_frombytes(substr($raw, 0, $need), "u8", [$h, $stride]);
    $filters = nd_to_list(nd_reshape(nd_slice($all, [null, [0, 1]]), [$h]));

    // FAST PATH: every row unfiltered. Common for small images and for what
    // this codec itself writes at filter 0, and it skips the whole per-byte
    // loop below.
    $allNone = true;
    $i = 0;
    while ($i < $h) {
        if ($filters[$i] != 0) { $allNone = false; $i = $h; }
        else { $i = $i + 1; }
    }
    if ($allNone) {
        return pxbuf.materialise2d(nd_slice($all, [null, [1, $stride]]));
    }

    $out = nd_zeros([$h, $rowBytes], "i64");
    $prev = [];
    $i = 0;
    while ($i < $rowBytes) { push($prev, 0); $i = $i + 1; }

    $y = 0;
    while ($y < $h) {
        $f = $filters[$y];
        $row = nd_to_list(nd_reshape(nd_slice($all, [[$y, $y + 1], [1, $stride]]), [$rowBytes]));

        if ($f == 0) {
            // nothing to do
        } else {
            if ($f == 2) {
                // Up: one whole-row add, no left dependency.
                $i = 0;
                while ($i < $rowBytes) { $row[$i] = ($row[$i] + $prev[$i]) % 256; $i = $i + 1; }
            } else {
                if ($f == 1) {
                    // Sub: a prefix sum along each channel lane.
                    $i = $bpp;
                    while ($i < $rowBytes) { $row[$i] = ($row[$i] + $row[$i - $bpp]) % 256; $i = $i + 1; }
                } else {
                    if ($f == 3) {
                        // Average: (left + up) >> 1, so left is needed first.
                        $i = 0;
                        while ($i < $bpp) { $row[$i] = ($row[$i] + floor($prev[$i] / 2)) % 256; $i = $i + 1; }
                        while ($i < $rowBytes) {
                            $row[$i] = ($row[$i] + floor(($row[$i - $bpp] + $prev[$i]) / 2)) % 256;
                            $i = $i + 1;
                        }
                    } else {
                        if ($f == 4) {
                            // Paeth: the predictor nearest to left + up - upleft.
                            $i = 0;
                            while ($i < $bpp) { $row[$i] = ($row[$i] + $prev[$i]) % 256; $i = $i + 1; }
                            while ($i < $rowBytes) {
                                $a = $row[$i - $bpp];
                                $b = $prev[$i];
                                $c = $prev[$i - $bpp];
                                $pp = $a + $b - $c;
                                $pa = $pp - $a;  if ($pa < 0) { $pa = 0 - $pa; }
                                $pb = $pp - $b;  if ($pb < 0) { $pb = 0 - $pb; }
                                $pc = $pp - $c;  if ($pc < 0) { $pc = 0 - $pc; }
                                if ($pa <= $pb && $pa <= $pc) { $pr = $a; }
                                else { if ($pb <= $pc) { $pr = $b; } else { $pr = $c; } }
                                $row[$i] = ($row[$i] + $pr) % 256;
                                $i = $i + 1;
                            }
                        } else {
                            throw "PNG: row " + str($y) + " uses filter type " +
                                  str($f) + ", which is not one of 0..4";
                        }
                    }
                }
            }
        }
        nd_add(nd_reshape(nd($row, "i64"), [1, $rowBytes]), 0,
               nd_slice($out, [[$y, $y + 1], null]));
        $prev = $row;
        $y = $y + 1;
    }
    return $out;
}

// ─── Sample extraction ──────────────────────────────────────────────────
//
// _samples($rows, $w, $h, $bits, $channels, $scale) -> an (h, w, channels)
// i64 array of sample values.
//
// Sub-byte depths are unpacked with the eight bit planes trick: the loop is
// over BITS, never over pixels.
def _samples($rows, $w, $h, $bits, $channels, $scale) {
    if ($h == 0 || $w == 0) { return nd_zeros([$h, $w, $channels], "i64"); }

    if ($bits == 8) {
        $sub = nd_slice($rows, [null, [0, $w * $channels]]);
        return nd_reshape(pxbuf.materialise2d($sub), [$h, $w, $channels]);
    }
    if ($bits == 16) {
        // Big-endian pairs: high * 256 + low.
        $hi = nd_slice($rows, [null, [0, $w * $channels * 2, 2]]);
        $lo = nd_slice($rows, [null, [1, $w * $channels * 2, 2]]);
        $v = nd_add(nd_multiply($hi, 256), $lo);
        return nd_reshape(pxbuf.materialise2d($v), [$h, $w, $channels]);
    }
    // 1, 2 or 4 bits, one channel, packed most significant first.
    $per = floor(8 / $bits);
    $mask = pow(2, $bits) - 1;
    $cols = floor(($w + $per - 1) / $per);
    $wide = $cols * $per;
    $out = nd_zeros([$h, $wide], "i64");
    $k = 0;
    while ($k < $per) {
        $shift = pow(2, $bits * ($per - 1 - $k));
        $v = nd_mod(nd_floor_divide(nd_slice($rows, [null, [0, $cols]]), $shift), $mask + 1);
        if ($scale != 1) { $v = nd_multiply($v, $scale); }
        nd_add($v, 0, nd_slice($out, [null, [$k, $wide, $per]]));
        $k = $k + 1;
    }
    $trim = nd_slice($out, [null, [0, $w]]);
    return nd_reshape(pxbuf.materialise2d($trim), [$h, $w, 1]);
}

// ─── Adam7 ──────────────────────────────────────────────────────────────
//
// Seven passes, each a subsampled grid. The pass geometry is a table rather
// than arithmetic because the table is what the spec gives and what every
// other implementation checks against.
$_A7_XSTART = [0, 4, 0, 2, 0, 1, 0];
$_A7_YSTART = [0, 0, 4, 0, 2, 0, 1];
$_A7_XSTEP  = [8, 8, 4, 4, 2, 2, 1];
$_A7_YSTEP  = [8, 8, 8, 4, 4, 2, 2];

def _passSize($w, $h, $p) {
    $pw = floor(($w - $_A7_XSTART[$p] + $_A7_XSTEP[$p] - 1) / $_A7_XSTEP[$p]);
    $ph = floor(($h - $_A7_YSTART[$p] + $_A7_YSTEP[$p] - 1) / $_A7_YSTEP[$p]);
    if ($pw < 0) { $pw = 0; }
    if ($ph < 0) { $ph = 0; }
    return [$pw, $ph];
}

def _rowBytes($w, $bits, $channels) {
    return floor(($w * $bits * $channels + 7) / 8);
}

// ─── decode ─────────────────────────────────────────────────────────────
def decode($data) {
    $chunks = _chunks($data);
    if (len($chunks) == 0 || $chunks[0].type != "IHDR") {
        throw "PNG: the first chunk must be IHDR";
    }
    $ihdr = $chunks[0].data;
    if (len($ihdr) != 13) {
        throw "PNG: IHDR must be 13 bytes, got " + str(len($ihdr));
    }
    $w = _u32($ihdr, 0);
    $h = _u32($ihdr, 4);
    $bits = ord($ihdr[8]);
    $colour = ord($ihdr[9]);
    $comp = ord($ihdr[10]);
    $filt = ord($ihdr[11]);
    $inter = ord($ihdr[12]);

    if ($w == 0 || $h == 0) { throw "PNG: a zero dimension (" + str($w) + "x" + str($h) + ")"; }
    if ($comp != 0) { throw "PNG: compression method " + str($comp) + " is not DEFLATE"; }
    if ($filt != 0) { throw "PNG: filter method " + str($filt) + " is not the standard set"; }
    if ($inter != 0 && $inter != 1) { throw "PNG: interlace method " + str($inter) + " is unknown"; }

    // THE PIXEL BUDGET IS CHECKED HERE, from the header, before anything is
    // decompressed or allocated. A 13-byte IHDR can claim 60,000 x 60,000.
    if ($w > $MAX_PIXELS / $h) {
        throw "PNG: " + str($w) + "x" + str($h) + " is over the " +
              str($MAX_PIXELS) + "-pixel limit";
    }

    $mi = _modeFor($bits, $colour);
    if ($mi == null) {
        throw "PNG: bit depth " + str($bits) + " with colour type " + str($colour) +
              " is not a legal PNG combination";
    }

    // ── gather the chunks we care about ──
    $idat = [];
    $plte = null;
    $trns = null;
    $info = {};
    $i = 1;
    while ($i < len($chunks)) {
        $c = $chunks[$i];
        $t = $c.type;
        if ($t == "IDAT") { push($idat, $c.data); }
        else {
        if ($t == "PLTE") {
            if (len($c.data) % 3 != 0) { throw "PNG: PLTE is not a multiple of 3 bytes"; }
            if (len($c.data) > 768) { throw "PNG: PLTE holds more than 256 entries"; }
            $plte = $c.data;
        }
        else { if ($t == "tRNS") { $trns = $c.data; }
        else { if ($t == "gAMA" && len($c.data) == 4) { $info.gamma = _u32($c.data, 0) / 100000.0; }
        else { if ($t == "sRGB" && len($c.data) == 1) { $info.srgb = ord($c.data[0]); }
        else { if ($t == "pHYs" && len($c.data) == 9) {
                 $unit = ord($c.data[8]);
                 $px = _u32($c.data, 0);
                 $py = _u32($c.data, 4);
                 // unit 1 is metres; Pillow reports dpi, 0.0254 m per inch.
                 if ($unit == 1) { $info.dpi = [$px * 0.0254, $py * 0.0254]; }
                 $info.aspect = [$px, $py];
               }
        else { if ($t == "tEXt") { _readText($info, $c.data, false); }
        else { if ($t == "zTXt") { _readText($info, $c.data, true); }
        else { if ($t == "iTXt") { _readITXt($info, $c.data); }
        else { if ($t == "bKGD") { $info.background = bytes($c.data); }
        } } } } } } } }
        }
        $i = $i + 1;
    }
    if (len($idat) == 0) { throw "PNG: the file has no IDAT chunk"; }
    if ($colour == 3 && $plte == null) {
        throw "PNG: colour type 3 needs a PLTE chunk";
    }

    // ── decompress, with the exact expected size as the bomb ceiling ──
    $channels = $mi.channels;
    if ($inter == 0) {
        $rowBytes = _rowBytes($w, $bits, $channels);
        $expect = ($rowBytes + 1) * $h;
    } else {
        $expect = 0;
        $p = 0;
        while ($p < 7) {
            $ps = _passSize($w, $h, $p);
            if ($ps[0] > 0 && $ps[1] > 0) {
                $expect = $expect + (_rowBytes($ps[0], $bits, $channels) + 1) * $ps[1];
            }
            $p = $p + 1;
        }
    }
    $raw = _inflate(join($idat, ""), $expect);
    if (len($raw) < $expect) {
        throw "PNG: the image data decompresses to " + str(len($raw)) +
              " bytes, " + str($expect) + " are needed";
    }

    $bpp = floor(($bits * $channels + 7) / 8);
    if ($bpp < 1) { $bpp = 1; }

    if ($inter == 0) {
        $rows = _unfilter($raw, $h, _rowBytes($w, $bits, $channels), $bpp);
        $samp = _samples($rows, $w, $h, $bits, $channels, $mi.scale);
    } else {
        $samp = _deinterlace($raw, $w, $h, $bits, $channels, $mi.scale, $bpp);
    }

    // ── assemble the buffer in the target mode ──
    $r = _toMode($samp, $mi.mode, $bits, $colour, $plte, $trns, $w, $h, $info);
    return {"mode": $r.mode, "size": [$w, $h], "px": $r.px,
            "palette": $r.palette, "info": $info};
}

// _deinterlace -> the seven Adam7 passes, each unfiltered on its own and
// scattered into the full image with a strided slice assignment.
def _deinterlace($raw, $w, $h, $bits, $channels, $scale, $bpp) {
    $out = nd_zeros([$h, $w, $channels], "i64");
    $pos = 0;
    $p = 0;
    while ($p < 7) {
        $ps = _passSize($w, $h, $p);
        $pw = $ps[0];
        $ph = $ps[1];
        if ($pw > 0 && $ph > 0) {
            $rb = _rowBytes($pw, $bits, $channels);
            $need = ($rb + 1) * $ph;
            if ($pos + $need > len($raw)) {
                throw "PNG: interlace pass " + str($p + 1) + " needs " + str($need) +
                      " bytes but only " + str(len($raw) - $pos) + " remain";
            }
            $rows = _unfilter(substr($raw, $pos, $need), $ph, $rb, $bpp);
            $samp = _samples($rows, $pw, $ph, $bits, $channels, $scale);
            // scatter: every `step` pixel starting at `start`
            $dst = nd_slice($out, [[$_A7_YSTART[$p], $h, $_A7_YSTEP[$p]],
                                   [$_A7_XSTART[$p], $w, $_A7_XSTEP[$p]], null]);
            nd_add($samp, 0, $dst);
            $pos = $pos + $need;
        }
        $p = $p + 1;
    }
    return $out;
}

// _toMode -> turn the raw samples into the final mode's buffer, applying the
// palette, tRNS and the 16-bit narrowing Pillow does.
def _toMode($samp, $mode, $bits, $colour, $plte, $trns, $w, $h, $info) {
    $palette = null;

    if ($colour == 3) {
        // Indexed: the samples ARE the indices, and the palette travels with
        // them. tRNS here is a per-entry alpha table, which Pillow keeps in
        // info rather than applying.
        $palette = nd_zeros([256, 3], "i64");
        $n = floor(len($plte) / 3);
        $flat = nd_frombytes($plte, "u8", [$n, 3]);
        nd_add($flat, 0, nd_slice($palette, [[0, $n], null]));
        if ($trns != null) { $info.transparency = bytes($trns); }
        return {"mode": "P", "px": $samp, "palette": $palette};
    }

    if ($colour == 0) {
        if ($bits == 16) {
            if ($trns != null && len($trns) >= 2) {
                $info.transparency = ord($trns[0]) * 256 + ord($trns[1]);
            }
            return {"mode": "I;16", "px": $samp, "palette": null};
        }
        if ($trns != null && len($trns) >= 2) {
            // A 1/2/4-bit tRNS value is a raw level, so it is scaled the same
            // way the samples were.
            $tv = ord($trns[0]) * 256 + ord($trns[1]);
            if ($bits == 1) { $info.transparency = $tv * 255; }
            else { if ($bits == 2) { $info.transparency = $tv * 85; }
            else { if ($bits == 4) { $info.transparency = $tv * 17; }
            else { $info.transparency = $tv; } } }
        }
        if ($bits == 1) { return {"mode": "1", "px": $samp, "palette": null}; }
        return {"mode": "L", "px": $samp, "palette": null};
    }

    if ($colour == 2) {
        if ($bits == 16) {
            // Pillow's rawmode is RGB;16B, which keeps only the HIGH byte of
            // each sample -- a 16-bit truecolour PNG comes back as 8-bit RGB.
            if ($trns != null && len($trns) >= 6) {
                $info.transparency = [ord($trns[0]) * 256 + ord($trns[1]),
                                      ord($trns[2]) * 256 + ord($trns[3]),
                                      ord($trns[4]) * 256 + ord($trns[5])];
            }
            return {"mode": "RGB", "px": nd_floor_divide($samp, 256), "palette": null};
        }
        if ($trns != null && len($trns) >= 6) {
            $info.transparency = [ord($trns[1]), ord($trns[3]), ord($trns[5])];
        }
        return {"mode": "RGB", "px": $samp, "palette": null};
    }

    if ($colour == 4) {
        if ($bits == 16) {
            // Pillow widens 16-bit grey+alpha to RGBA, keeping the high byte.
            $g = nd_floor_divide(pxbuf.band($samp, 0), 256);
            $a = nd_floor_divide(pxbuf.band($samp, 1), 256);
            return {"mode": "RGBA",
                    "px": pxbuf.stack([$g, $g, $g, $a], "i64"), "palette": null};
        }
        return {"mode": "LA", "px": $samp, "palette": null};
    }

    // colour type 6
    if ($bits == 16) {
        return {"mode": "RGBA", "px": nd_floor_divide($samp, 256), "palette": null};
    }
    return {"mode": "RGBA", "px": $samp, "palette": null};
}

// ─── Text chunks ────────────────────────────────────────────────────────
//
// A keyword, a NUL, then the text. zTXt adds a compression byte and the text
// is a zlib stream; iTXt adds a language tag and a translated keyword.
//
// MAX_TEXT_CHUNK bounds the decompressed size: a zTXt chunk is the easiest
// decompression bomb in the format, because nothing about it looks like image
// data and a naive reader inflates it without a ceiling.
$MAX_TEXT_CHUNK = 1048576;

def _split0($s, $from) {
    $i = $from;
    while ($i < len($s)) {
        if (ord($s[$i]) == 0) { return $i; }
        $i = $i + 1;
    }
    return -1;
}

def _readText($info, $body, $compressed) {
    $z = _split0($body, 0);
    if ($z < 0) { return null; }
    $key = substr($body, 0, $z);
    if ($info.text == null) { $info.text = {}; }
    if (!$compressed) {
        $info.text[$key] = substr($body, $z + 1, len($body) - $z - 1);
        return null;
    }
    if ($z + 2 > len($body)) { return null; }
    $method = ord($body[$z + 1]);
    if ($method != 0) { return null; }
    $payload = substr($body, $z + 2, len($body) - $z - 2);
    $txt = null;
    try { $txt = _inflate($payload, $MAX_TEXT_CHUNK); } catch ($e) { $txt = null; }
    if ($txt != null) { $info.text[$key] = $txt; }
    return null;
}

def _readITXt($info, $body) {
    $z = _split0($body, 0);
    if ($z < 0 || $z + 2 > len($body)) { return null; }
    $key = substr($body, 0, $z);
    $flag = ord($body[$z + 1]);
    $method = ord($body[$z + 2]);
    $lang = _split0($body, $z + 3);
    if ($lang < 0) { return null; }
    $trans = _split0($body, $lang + 1);
    if ($trans < 0) { return null; }
    $payload = substr($body, $trans + 1, len($body) - $trans - 1);
    if ($info.text == null) { $info.text = {}; }
    if ($flag == 0) { $info.text[$key] = $payload; return null; }
    if ($method != 0) { return null; }
    $txt = null;
    try { $txt = _inflate($payload, $MAX_TEXT_CHUNK); } catch ($e) { $txt = null; }
    if ($txt != null) { $info.text[$key] = $txt; }
    return null;
}

// ─── encode ─────────────────────────────────────────────────────────────
//
// $spec is {"mode", "size", "px", "palette", "info"} plus optional
// "png_filter" (0..4, default 0) and "png_level" (0..9).
//
// Filter 0 is the default because this codec's own fast read path requires
// it and because the gain from adaptive filtering is small on the images a
// Bantu program typically writes. Pass png_filter to choose another.
def encode($spec) {
    $m = $spec.mode;
    $px = $spec.px;
    $d = pxbuf.dims($px);
    $h = $d[0];
    $w = $d[1];
    if ($w == 0 || $h == 0) { throw "PNG: cannot write a zero-sized image"; }

    $filter = 0;
    if ($spec.png_filter != null) { $filter = $spec.png_filter; }
    if ($filter < 0 || $filter > 4) { throw "PNG: filter must be 0..4, got " + str($filter); }
    $level = null;
    if ($spec.png_level != null) { $level = $spec.png_level; }

    // mode -> (bits, colour type, the samples to write)
    if ($m == "1")    { $bits = 1;  $ct = 0; $ch = 1; $samp = nd_floor_divide(nd_clip($px, 0, 255), 255); }
    else { if ($m == "L")    { $bits = 8;  $ct = 0; $ch = 1; $samp = nd_clip($px, 0, 255); }
    else { if ($m == "LA")   { $bits = 8;  $ct = 4; $ch = 2; $samp = nd_clip($px, 0, 255); }
    else { if ($m == "RGB")  { $bits = 8;  $ct = 2; $ch = 3; $samp = nd_clip($px, 0, 255); }
    else { if ($m == "RGBA") { $bits = 8;  $ct = 6; $ch = 4; $samp = nd_clip($px, 0, 255); }
    else { if ($m == "P")    { $bits = 8;  $ct = 3; $ch = 1; $samp = nd_clip($px, 0, 255); }
    else { if ($m == "PA")   { $bits = 8;  $ct = 3; $ch = 1; $samp = nd_clip(pxbuf.band($px, 0), 0, 255); }
    else { if ($m == "I;16" || $m == "I;16L" || $m == "I;16B" || $m == "I;16N") {
             $bits = 16; $ct = 0; $ch = 1; $samp = nd_clip($px, 0, 65535); }
    else {
        throw "PNG: cannot write mode " + str($m) +
              ". Convert to 1, L, LA, P, RGB, RGBA or I;16 first.";
    } } } } } } } }

    // ── the filtered scanlines ──
    $rowBytes = _rowBytes($w, $bits, $ch);
    $parts = [];
    if ($bits == 1) {
        // pack 8 pixels per byte, most significant first
        $cols = floor(($w + 7) / 8);
        $padded = nd_zeros([$h, $cols * 8], "i64");
        nd_add(nd_reshape($samp, [$h, $w]), 0, nd_slice($padded, [null, [0, $w]]));
        $acc = nd_zeros([$h, $cols], "i64");
        $k = 0;
        while ($k < 8) {
            $plane = nd_slice($padded, [null, [$k, $cols * 8, 8]]);
            $acc = nd_add($acc, nd_multiply($plane, pow(2, 7 - $k)));
            $k = $k + 1;
        }
        $body = nd_tobytes($acc, "u8");
    } else {
        if ($bits == 16) { $body = nd_tobytes(nd_reshape($samp, [$h, $w * $ch]), "u16be"); }
        else { $body = nd_tobytes(nd_reshape($samp, [$h, $w * $ch]), "u8"); }
    }

    if ($filter == 0) {
        // The common case: a NUL filter byte in front of each row.
        $y = 0;
        while ($y < $h) {
            push($parts, "\x00");
            push($parts, substr($body, $y * $rowBytes, $rowBytes));
            $y = $y + 1;
        }
    } else {
        $bpp = floor(($bits * $ch + 7) / 8);
        if ($bpp < 1) { $bpp = 1; }
        $prev = [];
        $i = 0;
        while ($i < $rowBytes) { push($prev, 0); $i = $i + 1; }
        $y = 0;
        while ($y < $h) {
            $row = bytes(substr($body, $y * $rowBytes, $rowBytes));
            $enc = _filterRow($row, $prev, $filter, $bpp, $rowBytes);
            push($parts, chr($filter));
            push($parts, frombytes($enc));
            $prev = $row;
            $y = $y + 1;
        }
    }

    $z = _compress(join($parts, ""), $level);

    // ── the chunks ──
    $out = [];
    push($out, $SIGNATURE);
    $ihdr = _p32($w) + _p32($h) + chr($bits) + chr($ct) + "\x00\x00\x00";
    push($out, _chunk("IHDR", $ihdr));
    if ($ct == 3) {
        $pal = $spec.palette;
        if ($pal == null) { throw "PNG: a P-mode image needs a palette to write"; }
        push($out, _chunk("PLTE", nd_tobytes(nd_slice($pal, [[0, 256], null]), "u8")));
    }
    push($out, _chunk("IDAT", $z));
    push($out, _chunk("IEND", ""));
    return join($out, "");
}

def _chunk($type, $body) {
    return _p32(len($body)) + $type + $body + _p32(_crc32($type + $body));
}

// _filterRow -> encode one row with the chosen filter.
def _filterRow($row, $prev, $f, $bpp, $n) {
    $out = [];
    $i = 0;
    if ($f == 1) {
        while ($i < $n) {
            $left = 0;
            if ($i >= $bpp) { $left = $row[$i - $bpp]; }
            push($out, ($row[$i] - $left + 256) % 256);
            $i = $i + 1;
        }
        return $out;
    }
    if ($f == 2) {
        while ($i < $n) { push($out, ($row[$i] - $prev[$i] + 256) % 256); $i = $i + 1; }
        return $out;
    }
    if ($f == 3) {
        while ($i < $n) {
            $left = 0;
            if ($i >= $bpp) { $left = $row[$i - $bpp]; }
            push($out, ($row[$i] - floor(($left + $prev[$i]) / 2) + 256) % 256);
            $i = $i + 1;
        }
        return $out;
    }
    if ($f == 4) {
        while ($i < $n) {
            $a = 0;
            $c = 0;
            if ($i >= $bpp) { $a = $row[$i - $bpp]; $c = $prev[$i - $bpp]; }
            $b = $prev[$i];
            $pp = $a + $b - $c;
            $pa = $pp - $a;  if ($pa < 0) { $pa = 0 - $pa; }
            $pb = $pp - $b;  if ($pb < 0) { $pb = 0 - $pb; }
            $pc = $pp - $c;  if ($pc < 0) { $pc = 0 - $pc; }
            if ($pa <= $pb && $pa <= $pc) { $pr = $a; }
            else { if ($pb <= $pc) { $pr = $b; } else { $pr = $c; } }
            push($out, ($row[$i] - $pr + 256) % 256);
            $i = $i + 1;
        }
        return $out;
    }
    while ($i < $n) { push($out, $row[$i]); $i = $i + 1; }
    return $out;
}
