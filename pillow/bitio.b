// ════════════════════════════════════════════════════════════════════════
//  pillow/bitio.b — one bit reader and one bit writer, in Bantu.
//
//  Every format pillow reads is a bitstream, and the two bit orders in use
//  disagree in exactly the way that produces a silent, format-specific bug:
//
//      "lsb"  DEFLATE / PNG / zlib / GIF-LZW   — bits fill from bit 0 upward
//      "msb"  JPEG / TIFF-CCITT / BMP-1bit     — bits fill from bit 7 downward
//
//  Writing that twice is how the two come to disagree, so it is written once,
//  parameterised by order, and every codec uses it.
//
//  State is a dict, not a class, because `new alias.Class()` does not parse
//  across a module boundary in Bantu — so the constructor is a function.
//
//      include "./bitio.b" as bitio;
//      $r = bitio.reader($byteList, "lsb");
//      $v = bitio.bits($r, 5);
//
//  NOTE ON SPEED: a Bantu function call is ~1 us, so a codec that pulls one
//  symbol at a time through bits() pays that per symbol. inflate.b therefore
//  inlines its own accumulator in locals inside one function (see the comment
//  there). This module is for the codecs whose symbol rate is low enough that
//  clarity wins — and it is the reference the inlined versions are tested
//  against, which is why both exist.
// ════════════════════════════════════════════════════════════════════════

// ─── Reader ─────────────────────────────────────────────────────────────
// `src` is a byte-list (list of integers 0..255). Use bytes($s) to make one.
//
//   pos   next byte to consume
//   acc   bits held but not yet returned
//   n     how many bits acc holds
def reader($src, $order) {
    if ($order != "lsb" && $order != "msb") {
        throw "bitio.reader: order must be \"lsb\" or \"msb\" (got " + str($order) + ")";
    }
    return {"src": $src, "len": len($src), "pos": 0, "acc": 0, "n": 0,
            "order": $order, "eof": false};
}

// How many whole bytes remain unconsumed (not counting buffered bits).
def remaining($r) { return $r.len - $r.pos; }

// Byte offset of the next unread byte — for error messages that name a place.
def pos($r) { return $r.pos; }

// Bits buffered but not yet returned.
def buffered($r) { return $r.n; }

// Did a read run past the end? Reads past the end return zero bits and set
// this, so a truncated file is a detectable condition rather than a crash.
def at_eof($r) { return $r.eof; }

// bits($r, $count) -> the next `count` bits as a number, or null past the end.
//
// At most 24 bits per call: the accumulator is a float64, exact to 2^53, and
// 24 + 8 = 32 keeps every intermediate well inside that.
def bits($r, $count) {
    if ($count < 0 || $count > 24) {
        throw "bitio.bits: count must be 0..24 (got " + str($count) + ")";
    }
    if ($count == 0) { return 0; }
    while ($r.n < $count) {
        if ($r.pos >= $r.len) { $r.eof = true; return null; }
        $b = $r.src[$r.pos];
        $r.pos = $r.pos + 1;
        if ($r.order == "lsb") {
            // DEFLATE: the new byte becomes the HIGH bits of the accumulator.
            $r.acc = $r.acc + $b * pow(2, $r.n);
        } else {
            // JPEG: the new byte becomes the LOW bits.
            $r.acc = $r.acc * 256 + $b;
        }
        $r.n = $r.n + 8;
    }
    $scale = pow(2, $count);
    if ($r.order == "lsb") {
        $v = $r.acc % $scale;                 // low `count` bits, in order
        $r.acc = ($r.acc - $v) / $scale;
        $r.n = $r.n - $count;
        return $v;
    }
    // msb: the top `count` bits of the n held.
    $drop = pow(2, $r.n - $count);
    $v = floor($r.acc / $drop);
    $r.acc = $r.acc - $v * $drop;
    $r.n = $r.n - $count;
    return $v;
}

// peek($r, $count) -> the next `count` bits WITHOUT consuming them.
// Returns null only if the stream cannot supply that many even with padding;
// a short tail peeks as if zero-padded, which is what a Huffman decoder needs
// at the very end of a scan.
def peek($r, $count) {
    if ($count < 0 || $count > 24) {
        throw "bitio.peek: count must be 0..24 (got " + str($count) + ")";
    }
    if ($count == 0) { return 0; }
    while ($r.n < $count) {
        if ($r.pos >= $r.len) { break; }
        $b = $r.src[$r.pos];
        $r.pos = $r.pos + 1;
        if ($r.order == "lsb") { $r.acc = $r.acc + $b * pow(2, $r.n); }
        else { $r.acc = $r.acc * 256 + $b; }
        $r.n = $r.n + 8;
    }
    if ($r.n == 0) { $r.eof = true; return null; }
    if ($r.order == "lsb") { return $r.acc % pow(2, $count); }
    // msb with a short tail: pad on the right with zeros.
    if ($r.n < $count) { return $r.acc * pow(2, $count - $r.n); }
    return floor($r.acc / pow(2, $r.n - $count));
}

// skip($r, $count) -> drop `count` bits. Any count, not just 24.
def skip($r, $count) {
    $left = $count;
    while ($left > 24) {
        if (bits($r, 24) == null) { return false; }
        $left = $left - 24;
    }
    if ($left > 0) { if (bits($r, $left) == null) { return false; } }
    return true;
}

// align($r) -> discard buffered bits up to the next byte boundary.
// DEFLATE's stored blocks and JPEG's markers both need this.
def align($r) {
    $drop = $r.n % 8;
    if ($drop > 0) { bits($r, $drop); }
    // Whole buffered bytes are pushed back so byte reads see them.
    $whole = ($r.n - ($r.n % 8)) / 8;
    if ($whole > 0) { $r.pos = $r.pos - $whole; }
    $r.acc = 0;
    $r.n = 0;
    return true;
}

// byte($r) -> the next whole byte, assuming the reader is byte-aligned.
def byte($r) {
    if ($r.n > 0) { align($r); }
    if ($r.pos >= $r.len) { $r.eof = true; return null; }
    $b = $r.src[$r.pos];
    $r.pos = $r.pos + 1;
    return $b;
}

// ─── Writer ─────────────────────────────────────────────────────────────
// Bits accumulate into whole bytes, which are pushed onto a list; `finish`
// pads the last partial byte with zeros and returns a byte string. Output is
// built by push + frombytes, never by string concatenation, which is O(n^2).
def writer($order) {
    if ($order != "lsb" && $order != "msb") {
        throw "bitio.writer: order must be \"lsb\" or \"msb\" (got " + str($order) + ")";
    }
    return {"out": [], "acc": 0, "n": 0, "order": $order};
}

// put($w, $value, $count) -> append the low `count` bits of value.
def put($w, $value, $count) {
    if ($count < 0 || $count > 24) {
        throw "bitio.put: count must be 0..24 (got " + str($count) + ")";
    }
    if ($count == 0) { return true; }
    $v = $value % pow(2, $count);
    if ($v < 0) { $v = $v + pow(2, $count); }
    if ($w.order == "lsb") {
        $w.acc = $w.acc + $v * pow(2, $w.n);
    } else {
        $w.acc = $w.acc * pow(2, $count) + $v;
    }
    $w.n = $w.n + $count;
    while ($w.n >= 8) {
        if ($w.order == "lsb") {
            $b = $w.acc % 256;
            $w.acc = ($w.acc - $b) / 256;
        } else {
            $shift = pow(2, $w.n - 8);
            $b = floor($w.acc / $shift);
            $w.acc = $w.acc - $b * $shift;
        }
        push($w.out, $b);
        $w.n = $w.n - 8;
    }
    return true;
}

// put_byte($w, $b) -> a whole byte, byte-aligning first.
def put_byte($w, $b) {
    if ($w.n > 0) { flush_align($w); }
    push($w.out, $b % 256);
    return true;
}

// flush_align($w) -> pad the partial byte with zeros so the next write starts
// on a byte boundary. Returns how many padding bits were added.
def flush_align($w) {
    if ($w.n == 0) { return 0; }
    $pad = 8 - $w.n;
    put($w, 0, $pad);
    return $pad;
}

// bitcount($w) -> how many bits have been written in total.
def bitcount($w) { return len($w.out) * 8 + $w.n; }

// finish($w) -> the bytes written, as a byte string (binary-safe).
def finish($w) {
    flush_align($w);
    return frombytes($w.out);
}

// finish_bytes($w) -> the bytes written, as a byte-list.
def finish_bytes($w) {
    flush_align($w);
    return $w.out;
}
