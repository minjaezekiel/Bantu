// ════════════════════════════════════════════════════════════════════════
//  pillow/codecs/ppm.b — Netpbm: PBM, PGM and PPM (P1-P6).
//
//  The first codec, and deliberately so: it has no compression, so it
//  exercises the whole path from file bytes to pixel buffer and back without
//  depending on inflate, and it makes the core testable end to end on day
//  one. Pillow's PpmImagePlugin covers the same seven formats.
//
//      P1  plain (ASCII) bitmap   -> mode "1"
//      P2  plain greymap          -> mode "L" or "I" (maxval > 255)
//      P3  plain pixmap           -> mode "RGB"
//      P4  raw bitmap             -> mode "1"
//      P5  raw greymap            -> mode "L" / "I;16B"
//      P6  raw pixmap             -> mode "RGB"
//
//  Codecs deal in buffers, never Images (see registry.b's header).
// ════════════════════════════════════════════════════════════════════════

include "../pxbuf.b" as pxbuf;

// ─── Header parsing ─────────────────────────────────────────────────────
//
// A Netpbm header is whitespace-separated tokens with '#' comments to
// end-of-line, and the token after the last header field is followed by
// EXACTLY ONE whitespace byte before raw data begins. Getting that single
// byte wrong shifts every pixel, so it is handled explicitly rather than by
// skipping whitespace greedily.

def _isSpace($c) {
    return $c == 32 || $c == 9 || $c == 10 || $c == 13 || $c == 11 || $c == 12;
}

// _token($s, $pos) -> {"tok": string, "next": index} skipping whitespace and
// comments. Returns null at end of input.
def _token($s, $pos) {
    $n = len($s);
    $i = $pos;
    while ($i < $n) {
        $c = ord($s[$i]);
        if ($c == 35) {                             // '#': to end of line
            while ($i < $n && ord($s[$i]) != 10) { $i = $i + 1; }
        } else {
            if (!_isSpace($c)) { break; }
            $i = $i + 1;
        }
    }
    if ($i >= $n) { return null; }
    $start = $i;
    while ($i < $n) {
        $c = ord($s[$i]);
        if (_isSpace($c) || $c == 35) { break; }
        $i = $i + 1;
    }
    return {"tok": substr($s, $start, $i - $start), "next": $i};
}

// _allDigits($s) -> is the token a bare run of 0-9?
//
// num("x") returns 0, not null, so a numeric conversion cannot be used to
// VALIDATE a token: "x", "" and "12abc" would all read as pixel values. The
// digits are checked first and converted second.
def _allDigits($s) {
    $n = len($s);
    if ($n == 0) { return false; }
    $i = 0;
    while ($i < $n) {
        $c = ord($s[$i]);
        if ($c < 48 || $c > 57) { return false; }
        $i = $i + 1;
    }
    return true;
}

def _intToken($s, $pos, $what) {
    $t = _token($s, $pos);
    if ($t == null) { throw "PPM: the header ends before " + $what; }
    if (!_allDigits($t.tok)) {
        throw "PPM: " + $what + " is not a non-negative whole number ('" + $t.tok + "')";
    }
    return {"v": num($t.tok), "next": $t.next};
}

$_MAX_PIXELS = 89478485;        // Pillow's MAX_IMAGE_PIXELS

def sniff($head) {
    if (len($head) < 2) { return false; }
    if ($head[0] != "P") { return false; }
    $c = ord($head[1]);
    if ($c < 49 || $c > 54) { return false; }       // '1'..'6'
    // A real Netpbm file has whitespace after the magic. Requiring it stops
    // "Pa..." and "P7" (which is PAM, a different format) matching.
    if (len($head) < 3) { return true; }
    return _isSpace(ord($head[2]));
}

// ─── decode ─────────────────────────────────────────────────────────────
def decode($data) {
    if (!sniff($data)) { throw "PPM: not a Netpbm file (bad magic)"; }
    $kind = ord($data[1]) - 48;                     // 1..6

    $pos = 2;
    $r = _intToken($data, $pos, "the width");
    $w = $r.v;
    $r = _intToken($data, $r.next, "the height");
    $h = $r.v;
    $pos = $r.next;

    $maxval = 1;
    if ($kind != 1 && $kind != 4) {
        $r = _intToken($data, $pos, "the maximum value");
        $maxval = $r.v;
        $pos = $r.next;
        if ($maxval < 1 || $maxval > 65535) {
            throw "PPM: maximum value " + str($maxval) + " is outside 1..65535";
        }
    }

    // The pixel budget is checked BEFORE anything is allocated, which is the
    // whole point of checking it (decision PW6).
    if ($w * $h > $_MAX_PIXELS) {
        throw "PPM: " + str($w) + "x" + str($h) + " is " + str($w * $h) +
              " pixels, over the " + str($_MAX_PIXELS) + "-pixel limit";
    }

    $isRaw = ($kind >= 4);
    $bands = 1;
    if ($kind == 3 || $kind == 6) { $bands = 3; }

    if ($kind == 1 || $kind == 4) { $m = "1"; }
    else {
        if ($bands == 3) { $m = "RGB"; }
        else {
            if ($maxval > 255) { $m = "I;16B"; } else { $m = "L"; }
        }
    }

    if ($w == 0 || $h == 0) {
        return {"mode": $m, "size": [$w, $h],
                "px": pxbuf.make($h, $w, $bands, "i64"),
                "palette": null, "info": {}};
    }

    if ($isRaw) {
        // Exactly one whitespace byte separates the header from the data.
        if ($pos >= len($data)) { throw "PPM: the file ends before the pixel data"; }
        if (!_isSpace(ord($data[$pos]))) {
            throw "PPM: expected one whitespace byte before the pixel data";
        }
        $pos = $pos + 1;
        $px = _decodeRaw($data, $pos, $kind, $w, $h, $maxval);
    } else {
        $px = _decodePlain($data, $pos, $kind, $w, $h, $maxval);
    }

    return {"mode": $m, "size": [$w, $h], "px": $px, "palette": null,
            "info": {"maxval": $maxval}};
}

def _decodeRaw($data, $pos, $kind, $w, $h, $maxval) {
    if ($kind == 4) {
        // P4: one bit per pixel, rows padded to a byte boundary. Unpacking is
        // vectorised: read the row bytes as an array once, then eight shifted
        // comparisons give the eight bit planes. The loop is over BITS (8),
        // not over pixels.
        $rowBytes = floor(($w + 7) / 8);
        $need = $rowBytes * $h;
        if ($pos + $need > len($data)) {
            throw "PPM: P4 needs " + str($need) + " bytes of pixel data, found " +
                  str(len($data) - $pos);
        }
        $raw = nd_frombytes(substr($data, $pos, $need), "u8", [$h, $rowBytes]);
        $out = nd_zeros([$h, $rowBytes * 8, 1], "i64");
        $bit = 0;
        while ($bit < 8) {
            // bit 7 is the leftmost pixel
            $shifted = nd_floor_divide($raw, pow(2, 7 - $bit));
            $onoff = nd_mod($shifted, 2);
            // In PBM, 1 means BLACK, so mode "1" stores 255 where the bit is 0.
            $val = nd_multiply(nd_subtract(1, $onoff), 255);
            $dest = nd_slice($out, [null, [$bit, $rowBytes * 8, 8], null]);
            nd_add(nd_reshape($val, [$h, $rowBytes, 1]), 0, $dest);
            $bit = $bit + 1;
        }
        if ($rowBytes * 8 == $w) { return $out; }
        return pxbuf.materialise(nd_slice($out, [null, [0, $w], null]), "i64");
    }

    $bands = 1;
    if ($kind == 6) { $bands = 3; }
    $wide = ($maxval > 255);
    $elem = "u8";
    $bpv = 1;
    if ($wide) { $elem = "u16be"; $bpv = 2; }       // Netpbm is big-endian
    $need = $w * $h * $bands * $bpv;
    if ($pos + $need > len($data)) {
        throw "PPM: needs " + str($need) + " bytes of pixel data, found " +
              str(len($data) - $pos);
    }
    return nd_frombytes(substr($data, $pos, $need), $elem, [$h, $w, $bands]);
}

def _decodePlain($data, $pos, $kind, $w, $h, $maxval) {
    $bands = 1;
    if ($kind == 3) { $bands = 3; }
    $want = $w * $h * $bands;
    $vals = [];
    $p = $pos;
    while (len($vals) < $want) {
        $t = _token($data, $p);
        if ($t == null) {
            throw "PPM: the file holds " + str(len($vals)) + " of " + str($want) +
                  " expected values";
        }
        if (!_allDigits($t.tok)) {
            throw "PPM: '" + $t.tok + "' is not a pixel value";
        }
        push($vals, num($t.tok));
        $p = $t.next;
    }
    $flat = nd($vals, "i64");
    $px = nd_reshape($flat, [$h, $w, $bands]);
    if ($kind == 1) {
        // P1: 1 means black, as in P4.
        return nd_multiply(nd_subtract(1, nd_clip($px, 0, 1)), 255);
    }
    return nd_clip($px, 0, $maxval);
}

// ─── encode ─────────────────────────────────────────────────────────────
//
// $spec is {"mode", "size", "px", ...} plus an optional "ppm_plain" flag.
// Writes the raw form (P4/P5/P6) by default, which is what Pillow writes.
def encode($spec) {
    $m = $spec.mode;
    $px = $spec.px;
    $d = pxbuf.dims($px);
    $h = $d[0];
    $w = $d[1];

    $plain = false;
    if ($spec.ppm_plain != null) { $plain = $spec.ppm_plain; }

    if ($m == "1") {
        if ($plain) {
            $hdr = "P1\n" + str($w) + " " + str($h) + "\n";
            // 1 means black, so invert what mode "1" stores.
            $bits = nd_subtract(1, nd_floor_divide(nd_clip($px, 0, 255), 255));
            return $hdr + _plainBody($bits, $w, $h, 1);
        }
        $hdr = "P4\n" + str($w) + " " + str($h) + "\n";
        $rowBytes = floor(($w + 7) / 8);
        $padded = nd_zeros([$h, $rowBytes * 8, 1], "i64");
        // 1 means black: a stored 0 becomes a set bit.
        $bits = nd_subtract(1, nd_floor_divide(nd_clip($px, 0, 255), 255));
        nd_add($bits, 0, nd_slice($padded, [null, [0, $w], null]));
        $acc = nd_zeros([$h, $rowBytes], "i64");
        $bit = 0;
        while ($bit < 8) {
            $plane = nd_reshape(nd_slice($padded, [null, [$bit, $rowBytes * 8, 8], null]),
                                [$h, $rowBytes]);
            $acc = nd_add($acc, nd_multiply($plane, pow(2, 7 - $bit)));
            $bit = $bit + 1;
        }
        return $hdr + nd_tobytes($acc, "u8");
    }

    if ($m == "L") {
        $hdr = "P5\n" + str($w) + " " + str($h) + "\n255\n";
        if ($plain) { return "P2\n" + str($w) + " " + str($h) + "\n255\n" + _plainBody($px, $w, $h, 1); }
        return $hdr + nd_tobytes(nd_clip($px, 0, 255), "u8");
    }

    if ($m == "I;16" || $m == "I;16L" || $m == "I;16B" || $m == "I;16N" || $m == "I") {
        $hdr = "P5\n" + str($w) + " " + str($h) + "\n65535\n";
        if ($plain) { return "P2\n" + str($w) + " " + str($h) + "\n65535\n" + _plainBody($px, $w, $h, 1); }
        return $hdr + nd_tobytes(nd_clip($px, 0, 65535), "u16be");
    }

    if ($m == "RGB") {
        if ($plain) { return "P3\n" + str($w) + " " + str($h) + "\n255\n" + _plainBody($px, $w, $h, 3); }
        $hdr = "P6\n" + str($w) + " " + str($h) + "\n255\n";
        return $hdr + nd_tobytes(nd_clip($px, 0, 255), "u8");
    }

    throw "PPM: cannot write mode " + str($m) +
          ". Convert to 1, L, I;16 or RGB first (Pillow has the same restriction)";
}

// _plainBody -> the ASCII pixel values, 70 columns or less per line as the
// Netpbm spec requires. Built with push + join, never string +=.
def _plainBody($px, $w, $h, $bands) {
    $vals = nd_to_list(nd_ravel($px));
    $parts = [];
    $line = [];
    $lineLen = 0;
    $i = 0;
    while ($i < len($vals)) {
        $t = str($vals[$i]);
        if ($lineLen + len($t) + 1 > 70) {
            push($parts, join($line, " "));
            $line = [];
            $lineLen = 0;
        }
        push($line, $t);
        $lineLen = $lineLen + len($t) + 1;
        $i = $i + 1;
    }
    if (len($line) > 0) { push($parts, join($line, " ")); }
    return join($parts, "\n") + "\n";
}
