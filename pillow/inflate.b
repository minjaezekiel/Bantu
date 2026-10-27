// ════════════════════════════════════════════════════════════════════════
//  pillow/inflate.b — DEFLATE (RFC 1951), zlib (RFC 1950) and gzip
//  (RFC 1952) decoding, written entirely in Bantu.
//
//  WHY THIS EXISTS when the interpreter also ships a native `inflate`:
//
//    1. It differential-tests the atom. A decompressor is a parser of
//       hostile input; two independent implementations agreeing on the same
//       corpus is the strongest cheap evidence either one is right.
//       tests/pillow_inflate_test.b decodes both ways and compares bytes.
//    2. It is the fallback. pillow probes has_native("inflate") and routes
//       accordingly, so the package is complete in Bantu alone on any build.
//
//  SPEED, measured rather than estimated — see tests/pillow_bench.b:
//  this is ~3 us per output byte, so a 256x256 RGBA PNG (262 KB raw) is
//  under a second and a 1080p RGBA PNG (8.3 MB) is half a minute. The native
//  atom is the default for exactly that reason. Do not "fix" this by making
//  it less readable; its job is to be obviously correct.
//
//  HOW IT IS FAST ENOUGH TO BE USABLE AT ALL:
//
//    - The whole decoder is ONE function with the bit accumulator in locals.
//      A Bantu function call is ~1 us, which a per-symbol bits() call would
//      pay on every symbol. bitio.b is the readable reference; this is the
//      inlined twin, and the test asserts they agree.
//    - Huffman decoding is a 512-entry direct lookup on the next 9 bits, so
//      the common literal costs one table index instead of a nine-iteration
//      canonical walk. Codes longer than 9 bits fall back to the walk.
//    - Powers of two come from a table, not from pow(), which is a call.
//
//      include "./inflate.b" as inflate;
//      $raw = inflate.inflate($zlibBytes);          // auto-detects container
//      $raw = inflate.raw($deflateBytes);           // bare DEFLATE
// ════════════════════════════════════════════════════════════════════════

// ─── Tables (RFC 1951 §3.2.5 and §3.2.7) ────────────────────────────────

// 2^k for k in 0..30, so the hot loop never calls pow().
$_P2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
        32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304,
        8388608, 16777216, 33554432, 67108864, 134217728, 268435456,
        536870912, 1073741824];

// Length codes 257..285: first byte count, then extra bits.
$_LEN_BASE  = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43,
               51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
$_LEN_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3,
               3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];

// Distance codes 0..29.
$_DIST_BASE  = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257,
                385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289,
                16385, 24577];
$_DIST_EXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
                9, 9, 10, 10, 11, 11, 12, 12, 13, 13];

// The order the code-length code lengths arrive in (RFC 1951 §3.2.7).
$_CL_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

// How many bits the fast Huffman table is indexed by.
$_FAST_BITS = 9;
$_FAST_SIZE = 512;

// Hard ceiling on one inflate call's output, so a bomb cannot exhaust memory
// before maxOut is consulted. 256 MB of bytes; callers pass a tighter maxOut.
$_ABS_MAX_OUT = 268435456;


// ─── Huffman table construction ─────────────────────────────────────────
//
// `lengths` is a list of code lengths, one per symbol, 0 meaning "unused".
// Returns a dict holding:
//
//   fast     a $_FAST_SIZE table; entry = symbol * 16 + codelength, or -1
//            when the code at that prefix is longer than $_FAST_BITS
//   counts   how many codes of each length 1..15
//   syms     symbols sorted by (length, symbol) — the canonical order
//   maxlen   the longest code present
//
// A canonical Huffman code is fully determined by the lengths, so this is the
// whole decoder's state. Over- and under-subscribed sets are rejected: an
// over-subscribed set is a corrupt stream, and an under-subscribed one is
// only legal in the single special case of one distance code (§3.2.7), which
// real encoders emit, so it is allowed exactly there.
def _build($lengths, $what, $allowIncomplete) {
    $counts = [];
    $i = 0;
    while ($i <= 15) { push($counts, 0); $i = $i + 1; }

    $n = len($lengths);
    $i = 0;
    while ($i < $n) {
        $L = $lengths[$i];
        if ($L < 0 || $L > 15) {
            throw "inflate: " + $what + " code length " + str($L) + " out of range at symbol " + str($i);
        }
        $counts[$L] = $counts[$L] + 1;
        $i = $i + 1;
    }
    $counts[0] = 0;

    // Kraft check: sum of 2^-len must be <= 1, and == 1 unless incomplete is
    // allowed. Done in integers by the usual doubling.
    $left = 1;
    $L = 1;
    while ($L <= 15) {
        $left = $left * 2 - $counts[$L];
        if ($left < 0) { throw "inflate: " + $what + " Huffman code is over-subscribed"; }
        $L = $L + 1;
    }
    $complete = ($left == 0);
    if (!$complete && !$allowIncomplete) {
        // An empty set (every length 0) is incomplete too; a block that then
        // uses the code is what fails, so an all-zero set is tolerated here
        // and caught at use.
        $any = false;
        $L = 1;
        while ($L <= 15) { if ($counts[$L] > 0) { $any = true; } $L = $L + 1; }
        if ($any) { throw "inflate: " + $what + " Huffman code is incomplete"; }
    }

    // offsets[L] = index in `syms` where length-L symbols start.
    $offs = [0, 0];
    $L = 1;
    while ($L <= 15) { push($offs, $offs[$L] + $counts[$L]); $L = $L + 1; }

    $total = $offs[16];
    $syms = [];
    $i = 0;
    while ($i < $total) { push($syms, 0); $i = $i + 1; }
    $cursor = [];
    $L = 0;
    while ($L <= 16) { push($cursor, $offs[$L]); $L = $L + 1; }
    $i = 0;
    while ($i < $n) {
        $L = $lengths[$i];
        if ($L != 0) { $syms[$cursor[$L]] = $i; $cursor[$L] = $cursor[$L] + 1; }
        $i = $i + 1;
    }

    $maxlen = 0;
    $L = 1;
    while ($L <= 15) { if ($counts[$L] > 0) { $maxlen = $L; } $L = $L + 1; }

    // The fast table. DEFLATE packs Huffman codes most-significant-bit first
    // while the stream delivers bits least-significant first, so the next 9
    // bits off the stream are the code BIT-REVERSED. Filling every index whose
    // low L bits equal the reversed code is therefore the whole trick.
    $fast = [];
    $i = 0;
    while ($i < $_FAST_SIZE) { push($fast, -1); $i = $i + 1; }

    $code = 0;
    $idx = 0;
    $L = 1;
    while ($L <= 15) {
        $cnt = $counts[$L];
        $k = 0;
        while ($k < $cnt) {
            if ($L <= $_FAST_BITS) {
                // Reverse the L-bit code.
                $rev = 0;
                $c = $code;
                $b = 0;
                while ($b < $L) {
                    $rev = $rev * 2 + ($c % 2);
                    $c = floor($c / 2);
                    $b = $b + 1;
                }
                $step = $_P2[$L];
                $slot = $rev;
                $entry = $syms[$idx + $k] * 16 + $L;
                while ($slot < $_FAST_SIZE) {
                    $fast[$slot] = $entry;
                    $slot = $slot + $step;
                }
            }
            $code = $code + 1;
            $k = $k + 1;
        }
        $idx = $idx + $cnt;
        $code = $code * 2;
        $L = $L + 1;
    }

    return {"fast": $fast, "counts": $counts, "syms": $syms,
            "maxlen": $maxlen, "complete": $complete};
}


// ─── The fixed Huffman tables (RFC 1951 §3.2.6), built once at load ─────
def _fixedLit() {
    $L = [];
    $i = 0;
    while ($i < 144) { push($L, 8); $i = $i + 1; }
    while ($i < 256) { push($L, 9); $i = $i + 1; }
    while ($i < 280) { push($L, 7); $i = $i + 1; }
    while ($i < 288) { push($L, 8); $i = $i + 1; }
    return _build($L, "fixed literal/length", false);
}
def _fixedDist() {
    $L = [];
    $i = 0;
    while ($i < 30) { push($L, 5); $i = $i + 1; }
    return _build($L, "fixed distance", true);
}
$_FIXED_LIT  = _fixedLit();
$_FIXED_DIST = _fixedDist();


// ─── raw($src, $maxOut) — bare DEFLATE ──────────────────────────────────
//
// `src` is a byte-list or a byte string. Returns a byte-list.
// `maxOut` (optional) refuses a stream that would expand past it — the
// defence against an inflate bomb. Pass the exact expected size when it is
// known, which for PNG it always is.
//
// This is one long function on purpose: the bit accumulator, the output and
// both Huffman tables stay in locals, so the hot loop pays no call overhead.
def raw($src, $maxOut) {
    $in = $src;
    if (type($src) == "string") { $in = bytes($src); }
    if (type($in) != "list") { throw "inflate.raw: expected bytes or a byte-list"; }

    $cap = $_ABS_MAX_OUT;
    if ($maxOut != null) {
        if ($maxOut < 0) { throw "inflate.raw: maxOut must not be negative"; }
        if ($maxOut < $cap) { $cap = $maxOut; }
    }

    $srclen = len($in);
    $pos = 0;          // next byte to consume
    $acc = 0;          // bits held, LSB-first
    $nbit = 0;         // how many bits acc holds
    $out = [];
    $outn = 0;
    $P2 = $_P2;

    $done = false;
    while (!$done) {
        // ── block header: BFINAL(1), BTYPE(2) ──
        while ($nbit < 3) {
            if ($pos >= $srclen) { throw "inflate: stream ends inside a block header at byte " + str($pos); }
            $acc = $acc + $in[$pos] * $P2[$nbit];
            $pos = $pos + 1;
            $nbit = $nbit + 8;
        }
        $final = $acc % 2;
        $acc = floor($acc / 2);
        $btype = $acc % 4;
        $acc = floor($acc / 4);
        $nbit = $nbit - 3;
        if ($final == 1) { $done = true; }

        if ($btype == 0) {
            // ── stored: align, LEN, NLEN, then raw bytes ──
            $drop = $nbit % 8;
            if ($drop > 0) { $acc = floor($acc / $P2[$drop]); $nbit = $nbit - $drop; }
            $whole = ($nbit - ($nbit % 8)) / 8;
            if ($whole > 0) { $pos = $pos - $whole; $acc = 0; $nbit = 0; }
            if ($pos + 4 > $srclen) { throw "inflate: stream ends inside a stored block header at byte " + str($pos); }
            $blen  = $in[$pos] + $in[$pos + 1] * 256;
            $nlen  = $in[$pos + 2] + $in[$pos + 3] * 256;
            $pos = $pos + 4;
            if ($blen + $nlen != 65535) {
                throw "inflate: stored block length check failed at byte " + str($pos - 4) +
                      " (LEN " + str($blen) + ", NLEN " + str($nlen) + ")";
            }
            if ($pos + $blen > $srclen) {
                throw "inflate: stored block of " + str($blen) + " bytes runs past the end of the stream at byte " + str($pos);
            }
            if ($outn + $blen > $cap) {
                throw "inflate: output would exceed maxOut (" + str($cap) + " bytes)";
            }
            $k = 0;
            while ($k < $blen) { push($out, $in[$pos + $k]); $k = $k + 1; }
            $pos = $pos + $blen;
            $outn = $outn + $blen;

        } else {
            if ($btype == 3) { throw "inflate: reserved block type 3 at byte " + str($pos); }

            $lit = $_FIXED_LIT;
            $dst = $_FIXED_DIST;

            if ($btype == 2) {
                // ── dynamic: read the two code tables ──
                while ($nbit < 14) {
                    if ($pos >= $srclen) { throw "inflate: stream ends inside a dynamic block header at byte " + str($pos); }
                    $acc = $acc + $in[$pos] * $P2[$nbit];
                    $pos = $pos + 1;
                    $nbit = $nbit + 8;
                }
                $hlit  = $acc % 32;  $acc = floor($acc / 32);
                $hdist = $acc % 32;  $acc = floor($acc / 32);
                $hclen = $acc % 16;  $acc = floor($acc / 16);
                $nbit = $nbit - 14;
                $nlit  = $hlit + 257;
                $ndist = $hdist + 1;
                $ncl   = $hclen + 4;

                // The 3-bit code-length code lengths, in their permuted order.
                $clen = [];
                $i = 0;
                while ($i <= 18) { push($clen, 0); $i = $i + 1; }
                $i = 0;
                while ($i < $ncl) {
                    while ($nbit < 3) {
                        if ($pos >= $srclen) { throw "inflate: stream ends inside the code-length table at byte " + str($pos); }
                        $acc = $acc + $in[$pos] * $P2[$nbit];
                        $pos = $pos + 1;
                        $nbit = $nbit + 8;
                    }
                    $clen[$_CL_ORDER[$i]] = $acc % 8;
                    $acc = floor($acc / 8);
                    $nbit = $nbit - 3;
                    $i = $i + 1;
                }
                $clTab = _build($clen, "code-length", false);
                $clFast = $clTab.fast;
                $clCounts = $clTab.counts;
                $clSyms = $clTab.syms;

                // Then nlit + ndist code lengths, run-length coded by
                // symbols 16 (repeat previous 3-6), 17 (zeros 3-10) and
                // 18 (zeros 11-138).
                $want = $nlit + $ndist;
                $all = [];
                $prev = 0;
                while (len($all) < $want) {
                    // decode one code-length symbol
                    while ($nbit < 15) {
                        if ($pos >= $srclen) { break; }
                        $acc = $acc + $in[$pos] * $P2[$nbit];
                        $pos = $pos + 1;
                        $nbit = $nbit + 8;
                    }
                    $sym = -1;
                    $used = 0;
                    if ($nbit >= 1) {
                        $e = $clFast[$acc % 512];
                        if ($e >= 0) {
                            $used = $e % 16;
                            $sym = ($e - $used) / 16;
                        }
                    }
                    if ($sym < 0) {
                        // The canonical walk, for codes longer than 9 bits.
                        // `code` is shifted ONCE per iteration -- at the top,
                        // folded into the bit append. Shifting again at the
                        // bottom (as an earlier draft did) doubles the code on
                        // every step and the walk then finds nothing. It was
                        // invisible for 35 corpus vectors because the 9-bit
                        // fast table answers almost every symbol; the first
                        // stream with a >9-bit DISTANCE code found it.
                        // zlib's puff.c is the reference.
                        $code = 0; $first = 0; $index = 0; $L = 1; $probe = $acc;
                        while ($L <= 15) {
                            $code = $code * 2 + ($probe % 2);
                            $probe = floor($probe / 2);
                            $cnt = $clCounts[$L];
                            if ($code - $first < $cnt) { $sym = $clSyms[$index + $code - $first]; $used = $L; $L = 16; }
                            else { $index = $index + $cnt; $first = ($first + $cnt) * 2; $L = $L + 1; }
                        }
                        if ($sym < 0) { throw "inflate: invalid code-length code at byte " + str($pos); }
                    }
                    if ($used > $nbit) { throw "inflate: stream ends inside the code-length table at byte " + str($pos); }
                    $acc = floor($acc / $P2[$used]);
                    $nbit = $nbit - $used;

                    if ($sym < 16) {
                        push($all, $sym);
                        $prev = $sym;
                    } else {
                        if ($sym == 16) {
                            if (len($all) == 0) { throw "inflate: code-length repeat with nothing to repeat"; }
                            $extra = 2; $base = 3; $val = $prev;
                        } else {
                            if ($sym == 17) { $extra = 3; $base = 3; $val = 0; }
                            else { $extra = 7; $base = 11; $val = 0; }
                        }
                        while ($nbit < $extra) {
                            if ($pos >= $srclen) { throw "inflate: stream ends inside a code-length repeat at byte " + str($pos); }
                            $acc = $acc + $in[$pos] * $P2[$nbit];
                            $pos = $pos + 1;
                            $nbit = $nbit + 8;
                        }
                        $rep = $base + ($acc % $P2[$extra]);
                        $acc = floor($acc / $P2[$extra]);
                        $nbit = $nbit - $extra;
                        if (len($all) + $rep > $want) { throw "inflate: code-length repeat overruns the table"; }
                        $k = 0;
                        while ($k < $rep) { push($all, $val); $k = $k + 1; }
                        if ($sym != 16) { $prev = 0; }
                    }
                }

                $litLens = [];
                $i = 0;
                while ($i < $nlit) { push($litLens, $all[$i]); $i = $i + 1; }
                $dstLens = [];
                while ($i < $want) { push($dstLens, $all[$i]); $i = $i + 1; }
                if ($litLens[256] == 0) { throw "inflate: the dynamic block has no end-of-block code"; }
                $lit = _build($litLens, "literal/length", false);
                // One distance code is legal and under-subscribed (RFC 1951
                // §3.2.7); real encoders emit it for a literal-only block.
                $dst = _build($dstLens, "distance", true);
            }

            // ── the symbol loop ──
            $litFast   = $lit.fast;
            $litCounts = $lit.counts;
            $litSyms   = $lit.syms;
            $dstFast   = $dst.fast;
            $dstCounts = $dst.counts;
            $dstSyms   = $dst.syms;

            $inBlock = true;
            while ($inBlock) {
                // refill to 15 bits so the fast table and the walk both have
                // enough; short at the very end of the stream is tolerated and
                // caught by the `used > nbit` check.
                while ($nbit < 15) {
                    if ($pos >= $srclen) { break; }
                    $acc = $acc + $in[$pos] * $P2[$nbit];
                    $pos = $pos + 1;
                    $nbit = $nbit + 8;
                }

                $sym = -1;
                $used = 0;
                if ($nbit >= 1) {
                    $e = $litFast[$acc % 512];
                    if ($e >= 0) { $used = $e % 16; $sym = ($e - $used) / 16; }
                }
                if ($sym < 0) {
                    $code = 0; $first = 0; $index = 0; $L = 1; $probe = $acc;
                    while ($L <= 15) {
                        $code = $code * 2 + ($probe % 2);
                        $probe = floor($probe / 2);
                        $cnt = $litCounts[$L];
                        if ($code - $first < $cnt) { $sym = $litSyms[$index + $code - $first]; $used = $L; $L = 16; }
                        else { $index = $index + $cnt; $first = ($first + $cnt) * 2; $L = $L + 1; }
                    }
                    if ($sym < 0) { throw "inflate: invalid literal/length code at byte " + str($pos); }
                }
                if ($used > $nbit) { throw "inflate: stream ends inside a block at byte " + str($pos); }
                $acc = floor($acc / $P2[$used]);
                $nbit = $nbit - $used;

                if ($sym < 256) {
                    if ($outn >= $cap) { throw "inflate: output would exceed maxOut (" + str($cap) + " bytes)"; }
                    push($out, $sym);
                    $outn = $outn + 1;

                } else {
                    if ($sym == 256) { $inBlock = false; }
                    else {
                        $li = $sym - 257;
                        if ($li >= 29) { throw "inflate: invalid length code " + str($sym) + " at byte " + str($pos); }
                        $xb = $_LEN_EXTRA[$li];
                        $length = $_LEN_BASE[$li];
                        if ($xb > 0) {
                            while ($nbit < $xb) {
                                if ($pos >= $srclen) { throw "inflate: stream ends inside a length code at byte " + str($pos); }
                                $acc = $acc + $in[$pos] * $P2[$nbit];
                                $pos = $pos + 1;
                                $nbit = $nbit + 8;
                            }
                            $length = $length + ($acc % $P2[$xb]);
                            $acc = floor($acc / $P2[$xb]);
                            $nbit = $nbit - $xb;
                        }

                        // the distance symbol
                        while ($nbit < 15) {
                            if ($pos >= $srclen) { break; }
                            $acc = $acc + $in[$pos] * $P2[$nbit];
                            $pos = $pos + 1;
                            $nbit = $nbit + 8;
                        }
                        $dsym = -1;
                        $used = 0;
                        if ($nbit >= 1) {
                            $e = $dstFast[$acc % 512];
                            if ($e >= 0) { $used = $e % 16; $dsym = ($e - $used) / 16; }
                        }
                        if ($dsym < 0) {
                            $code = 0; $first = 0; $index = 0; $L = 1; $probe = $acc;
                            while ($L <= 15) {
                                $code = $code * 2 + ($probe % 2);
                                $probe = floor($probe / 2);
                                $cnt = $dstCounts[$L];
                                if ($code - $first < $cnt) { $dsym = $dstSyms[$index + $code - $first]; $used = $L; $L = 16; }
                                else { $index = $index + $cnt; $first = ($first + $cnt) * 2; $L = $L + 1; }
                            }
                            if ($dsym < 0) { throw "inflate: invalid distance code at byte " + str($pos); }
                        }
                        if ($used > $nbit) { throw "inflate: stream ends inside a distance code at byte " + str($pos); }
                        $acc = floor($acc / $P2[$used]);
                        $nbit = $nbit - $used;

                        if ($dsym >= 30) { throw "inflate: invalid distance code " + str($dsym) + " at byte " + str($pos); }
                        $xb = $_DIST_EXTRA[$dsym];
                        $distv = $_DIST_BASE[$dsym];
                        if ($xb > 0) {
                            while ($nbit < $xb) {
                                if ($pos >= $srclen) { throw "inflate: stream ends inside a distance code at byte " + str($pos); }
                                $acc = $acc + $in[$pos] * $P2[$nbit];
                                $pos = $pos + 1;
                                $nbit = $nbit + 8;
                            }
                            $distv = $distv + ($acc % $P2[$xb]);
                            $acc = floor($acc / $P2[$xb]);
                            $nbit = $nbit - $xb;
                        }
                        if ($distv > $outn) {
                            throw "inflate: back-reference of " + str($distv) +
                                  " reaches before the start of the output (only " + str($outn) + " bytes so far)";
                        }
                        if ($outn + $length > $cap) {
                            throw "inflate: output would exceed maxOut (" + str($cap) + " bytes)";
                        }
                        // The copy may overlap itself — that is how DEFLATE
                        // encodes a run — so it is byte by byte, reading from
                        // the output as it grows.
                        $from = $outn - $distv;
                        $k = 0;
                        while ($k < $length) {
                            push($out, $out[$from + $k]);
                            $k = $k + 1;
                        }
                        $outn = $outn + $length;
                    }
                }
            }
        }
    }
    return $out;
}


// ─── Container formats ──────────────────────────────────────────────────

// zlib($src, $maxOut) — RFC 1950: a 2-byte header, DEFLATE, an Adler-32.
// The checksum is verified; a mismatch is an error, because a PNG whose
// zlib checksum fails is a corrupt PNG.
def zlib($src, $maxOut) {
    $in = $src;
    if (type($src) == "string") { $in = bytes($src); }
    if (len($in) < 2) { throw "inflate.zlib: a zlib stream is at least 2 bytes"; }
    $cmf = $in[0];
    $flg = $in[1];
    if ($cmf % 16 != 8) {
        throw "inflate.zlib: compression method " + str($cmf % 16) + " is not DEFLATE";
    }
    if (($cmf * 256 + $flg) % 31 != 0) {
        throw "inflate.zlib: header check failed (CMF " + str($cmf) + ", FLG " + str($flg) + ")";
    }
    $start = 2;
    if (floor($flg / 32) % 2 == 1) {
        // FDICT: a 4-byte preset dictionary id follows. We have no dictionary
        // to apply, so this is refused rather than silently mis-decoded.
        throw "inflate.zlib: this stream needs a preset dictionary, which is not supported";
    }
    $body = [];
    $i = $start;
    $n = len($in);
    while ($i < $n) { push($body, $in[$i]); $i = $i + 1; }
    $out = raw($body, $maxOut);
    // Adler-32 over the output, if the trailer is present.
    if (len($body) >= 4) {
        $a = 1; $b = 0;
        $i = 0; $m = len($out);
        while ($i < $m) {
            $a = ($a + $out[$i]) % 65521;
            $b = ($b + $a) % 65521;
            $i = $i + 1;
        }
        $want = $b * 65536 + $a;
        $tn = len($in);
        $got = $in[$tn - 4] * 16777216 + $in[$tn - 3] * 65536 + $in[$tn - 2] * 256 + $in[$tn - 1];
        if ($got != $want) {
            throw "inflate.zlib: Adler-32 mismatch (stream says " + str($got) + ", data gives " + str($want) + ")";
        }
    }
    return $out;
}

// gunzip($src, $maxOut) — RFC 1952: a 10-byte header plus optional extras,
// DEFLATE, then CRC-32 and the length modulo 2^32.
def gunzip($src, $maxOut) {
    $in = $src;
    if (type($src) == "string") { $in = bytes($src); }
    $n = len($in);
    if ($n < 18) { throw "inflate.gunzip: a gzip member is at least 18 bytes"; }
    if ($in[0] != 31 || $in[1] != 139) { throw "inflate.gunzip: not a gzip stream (bad magic)"; }
    if ($in[2] != 8) { throw "inflate.gunzip: compression method " + str($in[2]) + " is not DEFLATE"; }
    $flg = $in[3];
    $i = 10;
    if (floor($flg / 4) % 2 == 1) {                    // FEXTRA
        if ($i + 2 > $n) { throw "inflate.gunzip: truncated FEXTRA"; }
        $xlen = $in[$i] + $in[$i + 1] * 256;
        $i = $i + 2 + $xlen;
    }
    if (floor($flg / 8) % 2 == 1) {                    // FNAME
        while ($i < $n && $in[$i] != 0) { $i = $i + 1; }
        $i = $i + 1;
    }
    if (floor($flg / 16) % 2 == 1) {                   // FCOMMENT
        while ($i < $n && $in[$i] != 0) { $i = $i + 1; }
        $i = $i + 1;
    }
    if (floor($flg / 2) % 2 == 1) { $i = $i + 2; }     // FHCRC
    if ($i >= $n - 8) { throw "inflate.gunzip: truncated gzip member"; }

    $body = [];
    $end = $n - 8;
    $k = $i;
    while ($k < $end) { push($body, $in[$k]); $k = $k + 1; }
    $out = raw($body, $maxOut);

    $isize = $in[$n - 4] + $in[$n - 3] * 256 + $in[$n - 2] * 65536 + $in[$n - 1] * 16777216;
    if (len($out) % 4294967296 != $isize) {
        throw "inflate.gunzip: length mismatch (trailer says " + str($isize) +
              ", got " + str(len($out) % 4294967296) + ")";
    }
    return $out;
}

// inflate($src, $maxOut) — sniff the container and decode.
// gzip starts 1f 8b; a zlib header has CMF&15 == 8 and (CMF<<8|FLG) % 31 == 0;
// anything else is treated as bare DEFLATE.
def inflate($src, $maxOut) {
    $in = $src;
    if (type($src) == "string") { $in = bytes($src); }
    $n = len($in);
    if ($n >= 2 && $in[0] == 31 && $in[1] == 139) { return gunzip($in, $maxOut); }
    if ($n >= 2 && $in[0] % 16 == 8 && ($in[0] * 256 + $in[1]) % 31 == 0) {
        return zlib($in, $maxOut);
    }
    return raw($in, $maxOut);
}

// The string forms, for callers holding or wanting a byte string.
def inflate_str($src, $maxOut) { return frombytes(inflate($src, $maxOut)); }
def zlib_str($src, $maxOut)    { return frombytes(zlib($src, $maxOut)); }
def raw_str($src, $maxOut)     { return frombytes(raw($src, $maxOut)); }
