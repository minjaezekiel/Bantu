// ════════════════════════════════════════════════════════════════════════
//  pillow/deflate.b — DEFLATE (RFC 1951), zlib (RFC 1950) and gzip
//  (RFC 1952) ENCODING, written entirely in Bantu.
//
//  The interpreter ships bp_zlib, a native compressor, and pillow uses it
//  when it is there. This exists so that WRITING a PNG, like reading one,
//  works on a build with no native help at all -- the package is complete in
//  Bantu alone, which is the point of the whole exercise.
//
//  WHAT IT EMITS, and why that is enough:
//
//    * stored blocks        -- no compression, always valid
//    * fixed-Huffman blocks -- RFC 1951 §3.2.6's built-in code, with LZ77
//                              matching over a 32 KB window
//
//  It does NOT build dynamic Huffman tables. A dynamic block needs a
//  length-limited Huffman construction (package-merge or Katajainen) over two
//  alphabets plus the code-length alphabet, and it buys maybe 15% over fixed
//  Huffman on image data. Every decoder in the world reads fixed-Huffman
//  blocks -- they are the same format -- so the output is a correct, ordinary
//  zlib stream that is slightly larger than zlib's. For a fallback path that
//  is the right trade; bp_zlib is the one that optimises.
//
//  Tested by round-tripping through BOTH decoders -- the native inflate and
//  pillow/inflate.b -- and by Python's zlib reading what this writes.
//
//      include "./deflate.b" as deflate;
//      $z = deflate.zlib($bytes, 6);        // a zlib stream
//      $d = deflate.raw($bytes, 6);         // bare DEFLATE
// ════════════════════════════════════════════════════════════════════════

$_P2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
        32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304,
        8388608, 16777216, 33554432, 67108864, 134217728, 268435456,
        536870912, 1073741824];

$_LEN_BASE  = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43,
               51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
$_LEN_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3,
               3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
$_DIST_BASE  = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257,
                385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289,
                16385, 24577];
$_DIST_EXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
                9, 9, 10, 10, 11, 11, 12, 12, 13, 13];

$_WINDOW = 32768;
$_MIN_MATCH = 3;
$_MAX_MATCH = 258;

// ─── The fixed-Huffman code, as (code, bits) pairs ──────────────────────
//
// RFC 1951 §3.2.6. Literal/length lengths are 8, 9, 7, 8 by range, and the
// canonical codes that follow are built once at load. DEFLATE writes a
// Huffman code most-significant-bit FIRST into an LSB-first bit stream, so
// each code is stored here already bit-reversed and the writer just appends
// its low `bits` bits.
def _revBits($v, $n) {
    $r = 0;
    $x = $v;
    $i = 0;
    while ($i < $n) {
        $r = $r * 2 + ($x % 2);
        $x = floor($x / 2);
        $i = $i + 1;
    }
    return $r;
}

def _buildFixed() {
    // Canonical codes from the fixed lengths.
    $lens = [];
    $i = 0;
    while ($i < 144) { push($lens, 8); $i = $i + 1; }
    while ($i < 256) { push($lens, 9); $i = $i + 1; }
    while ($i < 280) { push($lens, 7); $i = $i + 1; }
    while ($i < 288) { push($lens, 8); $i = $i + 1; }

    $codes = [];
    $i = 0;
    while ($i < 288) { push($codes, 0); $i = $i + 1; }

    $code = 0;
    $L = 1;
    while ($L <= 9) {
        $s = 0;
        while ($s < 288) {
            if ($lens[$s] == $L) { $codes[$s] = _revBits($code, $L); $code = $code + 1; }
            $s = $s + 1;
        }
        $code = $code * 2;
        $L = $L + 1;
    }
    return {"codes": $codes, "lens": $lens};
}
$_FIXED = _buildFixed();

// Distance codes are all 5 bits in the fixed code.
def _buildFixedDist() {
    $codes = [];
    $d = 0;
    while ($d < 30) { push($codes, _revBits($d, 5)); $d = $d + 1; }
    return $codes;
}
$_FIXED_DIST = _buildFixedDist();

// Which length code covers a match length, and which distance code a distance.
def _lenCode($length) {
    $i = 28;
    while ($i > 0) {
        if ($_LEN_BASE[$i] <= $length) { return $i; }
        $i = $i - 1;
    }
    return 0;
}
def _distCode($dist) {
    $i = 29;
    while ($i > 0) {
        if ($_DIST_BASE[$i] <= $dist) { return $i; }
        $i = $i - 1;
    }
    return 0;
}

// ─── raw($src, $level) — bare DEFLATE ───────────────────────────────────
//
// level 0      stored blocks only (fastest, no compression)
// level 1..9   fixed-Huffman with LZ77; the level sets how hard the matcher
//              looks (chain length), exactly as zlib's does
//
// Returns a byte string.
def raw($src, $level) {
    $in = $src;
    if (type($src) == "string") { $in = bytes($src); }
    if (type($in) != "list") { throw "deflate.raw: expected bytes or a byte-list"; }
    $n = len($in);
    $lvl = $level;
    if ($lvl == null) { $lvl = 6; }
    if ($lvl < 0 || $lvl > 9) { throw "deflate.raw: level must be 0..9 (got " + str($lvl) + ")"; }

    // ── level 0: stored blocks of at most 65,535 bytes ──
    if ($lvl == 0) {
        $out = [];
        if ($n == 0) {
            // One final, empty stored block.
            push($out, 1); push($out, 0); push($out, 0); push($out, 255); push($out, 255);
            return frombytes($out);
        }
        $pos = 0;
        while ($pos < $n) {
            $chunk = $n - $pos;
            if ($chunk > 65535) { $chunk = 65535; }
            $final = 0;
            if ($pos + $chunk >= $n) { $final = 1; }
            push($out, $final);
            push($out, $chunk % 256);
            push($out, floor($chunk / 256));
            push($out, (65535 - $chunk) % 256);
            push($out, floor((65535 - $chunk) / 256));
            $k = 0;
            while ($k < $chunk) { push($out, $in[$pos + $k]); $k = $k + 1; }
            $pos = $pos + $chunk;
        }
        return frombytes($out);
    }

    // ── levels 1..9: one fixed-Huffman block, LZ77 matched ──
    //
    // The matcher is a hash chain over 3-byte prefixes, which is what zlib
    // uses. `chainMax` is the level: a longer chain finds longer matches and
    // costs more time.
    $chainMax = $lvl * 8;
    $lazy = ($lvl >= 4);

    // head[h] = most recent position with hash h; prev[p] = the one before.
    $HSIZE = 32768;
    $head = [];
    $i = 0;
    while ($i < $HSIZE) { push($head, -1); $i = $i + 1; }
    $prev = [];
    $i = 0;
    while ($i < $n) { push($prev, -1); $i = $i + 1; }

    // The bit writer, inlined: a Bantu function call is ~1 us and this is the
    // hot path. `acc`/`nb` hold bits LSB-first; whole bytes go onto `out`.
    $out = [];
    $acc = 0;
    $nb = 0;

    // BFINAL = 1, BTYPE = 01 (fixed Huffman) -> 3 bits, value 0b011 = 3 LSB-first
    $acc = 3;
    $nb = 3;

    $codes = $_FIXED.codes;
    $clens = $_FIXED.lens;
    $dcodes = $_FIXED_DIST;
    $P2 = $_P2;

    $pos = 0;
    while ($pos < $n) {
        // ── find the longest match at `pos` ──
        $bestLen = 0;
        $bestDist = 0;
        if ($pos + $_MIN_MATCH <= $n) {
            $h = ($in[$pos] * 7129 + $in[$pos + 1] * 83 + $in[$pos + 2]) % $HSIZE;
            $cand = $head[$h];
            $chain = 0;
            while ($cand >= 0 && $chain < $chainMax) {
                $dist = $pos - $cand;
                if ($dist > $_WINDOW) { $cand = -1; }
                else {
                    // how far do they agree?
                    $L = 0;
                    $limit = $n - $pos;
                    if ($limit > $_MAX_MATCH) { $limit = $_MAX_MATCH; }
                    while ($L < $limit && $in[$cand + $L] == $in[$pos + $L]) { $L = $L + 1; }
                    if ($L > $bestLen) { $bestLen = $L; $bestDist = $dist; }
                    if ($bestLen >= $_MAX_MATCH) { $cand = -1; }
                    else { $cand = $prev[$cand]; }
                }
                $chain = $chain + 1;
            }
            // Insert `pos` into the chain for later positions.
            $prev[$pos] = $head[$h];
            $head[$h] = $pos;
        }

        // Lazy matching: if the next position has a strictly longer match,
        // emit this byte as a literal and take that one instead. This is
        // zlib's own heuristic and it is worth a few percent.
        if ($lazy && $bestLen >= $_MIN_MATCH && $bestLen < $_MAX_MATCH && $pos + $_MIN_MATCH + 1 <= $n) {
            $h2 = ($in[$pos + 1] * 7129 + $in[$pos + 2] * 83 + $in[$pos + 3 - 1]) % $HSIZE;
            if ($pos + 3 < $n) {
                $h2 = ($in[$pos + 1] * 7129 + $in[$pos + 2] * 83 + $in[$pos + 3]) % $HSIZE;
                $c2 = $head[$h2];
                $chain = 0;
                $next = 0;
                while ($c2 >= 0 && $chain < $chainMax) {
                    $d2 = ($pos + 1) - $c2;
                    if ($d2 > $_WINDOW || $d2 <= 0) { $c2 = -1; }
                    else {
                        $L = 0;
                        $limit = $n - ($pos + 1);
                        if ($limit > $_MAX_MATCH) { $limit = $_MAX_MATCH; }
                        while ($L < $limit && $in[$c2 + $L] == $in[$pos + 1 + $L]) { $L = $L + 1; }
                        if ($L > $next) { $next = $L; }
                        $c2 = $prev[$c2];
                    }
                    $chain = $chain + 1;
                }
                if ($next > $bestLen) { $bestLen = 0; }
            }
        }

        if ($bestLen >= $_MIN_MATCH) {
            // ── emit a length/distance pair ──
            $lc = _lenCode($bestLen);
            $sym = 257 + $lc;
            $acc = $acc + $codes[$sym] * $P2[$nb];
            $nb = $nb + $clens[$sym];
            while ($nb >= 8) { push($out, $acc % 256); $acc = floor($acc / 256); $nb = $nb - 8; }
            $xb = $_LEN_EXTRA[$lc];
            if ($xb > 0) {
                $acc = $acc + ($bestLen - $_LEN_BASE[$lc]) * $P2[$nb];
                $nb = $nb + $xb;
                while ($nb >= 8) { push($out, $acc % 256); $acc = floor($acc / 256); $nb = $nb - 8; }
            }
            $dc = _distCode($bestDist);
            $acc = $acc + $dcodes[$dc] * $P2[$nb];
            $nb = $nb + 5;
            while ($nb >= 8) { push($out, $acc % 256); $acc = floor($acc / 256); $nb = $nb - 8; }
            $xb = $_DIST_EXTRA[$dc];
            if ($xb > 0) {
                $acc = $acc + ($bestDist - $_DIST_BASE[$dc]) * $P2[$nb];
                $nb = $nb + $xb;
                while ($nb >= 8) { push($out, $acc % 256); $acc = floor($acc / 256); $nb = $nb - 8; }
            }
            // Insert the covered positions so later matches can reach them.
            $k = 1;
            while ($k < $bestLen) {
                $p = $pos + $k;
                if ($p + $_MIN_MATCH <= $n) {
                    $hh = ($in[$p] * 7129 + $in[$p + 1] * 83 + $in[$p + 2]) % $HSIZE;
                    $prev[$p] = $head[$hh];
                    $head[$hh] = $p;
                }
                $k = $k + 1;
            }
            $pos = $pos + $bestLen;

        } else {
            // ── emit a literal ──
            $sym = $in[$pos];
            $acc = $acc + $codes[$sym] * $P2[$nb];
            $nb = $nb + $clens[$sym];
            while ($nb >= 8) { push($out, $acc % 256); $acc = floor($acc / 256); $nb = $nb - 8; }
            $pos = $pos + 1;
        }
    }

    // end of block: symbol 256
    $acc = $acc + $codes[256] * $P2[$nb];
    $nb = $nb + $clens[256];
    while ($nb >= 8) { push($out, $acc % 256); $acc = floor($acc / 256); $nb = $nb - 8; }
    if ($nb > 0) { push($out, $acc % 256); }

    return frombytes($out);
}

// ─── Containers ─────────────────────────────────────────────────────────

def adler32($src) {
    $in = $src;
    if (type($src) == "string") { $in = bytes($src); }
    $a = 1;
    $b = 0;
    $i = 0;
    $n = len($in);
    while ($i < $n) {
        $a = ($a + $in[$i]) % 65521;
        $b = ($b + $a) % 65521;
        $i = $i + 1;
    }
    return $b * 65536 + $a;
}

// zlib($src, $level) — RFC 1950. CMF/FLG chosen so the check byte is valid.
def zlib($src, $level) {
    $body = raw($src, $level);
    $ad = adler32($src);
    // CMF: DEFLATE (8) with a 32 KB window (7 << 4) = 0x78.
    $cmf = 120;
    // FLG: the low 5 bits make (CMF<<8 | FLG) divisible by 31; bits 6-7 are
    // the compression-level hint, which no decoder acts on.
    $flg = 31 - (($cmf * 256) % 31);
    if ($flg == 31) { $flg = 0; }
    $out = [$cmf, $flg];
    $bb = bytes($body);
    $i = 0;
    while ($i < len($bb)) { push($out, $bb[$i]); $i = $i + 1; }
    push($out, floor($ad / 16777216) % 256);
    push($out, floor($ad / 65536) % 256);
    push($out, floor($ad / 256) % 256);
    push($out, $ad % 256);
    return frombytes($out);
}

// gzip($src, $level) — RFC 1952. CRC-32 comes from the interpreter's
// bp_crc32 when it is there, and from a Bantu table otherwise.
$_CRC = {"table": null};
def _crcTable() {
    if ($_CRC.table != null) { return $_CRC.table; }
    $t = [];
    $i = 0;
    while ($i < 256) {
        $c = $i;
        $k = 0;
        while ($k < 8) {
            if ($c % 2 == 1) { $c = bxor(3988292384, floor($c / 2)); }
            else { $c = floor($c / 2); }
            $k = $k + 1;
        }
        push($t, $c);
        $i = $i + 1;
    }
    $_CRC.table = $t;
    return $t;
}
def crc32($src) {
    $in = $src;
    if (type($src) == "string") { $in = bytes($src); }
    $t = _crcTable();
    $c = 4294967295;
    $i = 0;
    $n = len($in);
    while ($i < $n) {
        $c = bxor($t[bxor($c % 256, $in[$i]) % 256], floor($c / 256));
        $i = $i + 1;
    }
    return bxor($c, 4294967295);
}

def gzip($src, $level) {
    $body = raw($src, $level);
    $in = $src;
    if (type($src) == "string") { $in = bytes($src); }
    $out = [31, 139, 8, 0, 0, 0, 0, 0, 0, 255];
    $bb = bytes($body);
    $i = 0;
    while ($i < len($bb)) { push($out, $bb[$i]); $i = $i + 1; }
    $cr = crc32($in);
    push($out, $cr % 256);
    push($out, floor($cr / 256) % 256);
    push($out, floor($cr / 65536) % 256);
    push($out, floor($cr / 16777216) % 256);
    $n = len($in) % 4294967296;
    push($out, $n % 256);
    push($out, floor($n / 256) % 256);
    push($out, floor($n / 65536) % 256);
    push($out, floor($n / 16777216) % 256);
    return frombytes($out);
}
