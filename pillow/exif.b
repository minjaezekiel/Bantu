// ════════════════════════════════════════════════════════════════════════
//  pillow/exif.b — EXIF, read and write, and ImageOps.exif_transpose.
//
//  EXIF is a TIFF file stored inside a JPEG APP1 segment, so this module is
//  a TIFF tag reader with a different entry point. That is why it is
//  written in P5 beside the JPEG codec rather than in P8 beside TIFF, and
//  why codecs/tiff.b will include it instead of reimplementing it.
//
//  Both byte orders, all twelve TIFF field types plus BigTIFF's three, the
//  four sub-IFDs Pillow exposes (Exif, GPSInfo, Interop, and MakerNote as
//  raw bytes), the thumbnail IFD, and ExifTags.TAGS / GPSTAGS / IFD / Base
//  / GPS / Interop generated from Pillow's own dictionaries.
//
//  ─── TWO PROPERTIES THAT ARE LOAD-BEARING ────────────────────────────
//
//  * AN IFD CHAIN CAN POINT AT ITSELF. A next-IFD pointer that loops, or a
//    sub-IFD that points back at its parent, is an infinite loop in a naive
//    reader. Every offset is checked against the block length AND against a
//    visited set before it is followed.
//  * A TAG'S COUNT TIMES ITS TYPE SIZE CAN OVERSHOOT THE BLOCK. Checked
//    BEFORE the read, not after: a LONG tag claiming 2^31 entries is a
//    header, not an allocation.
//
//  ─── VALUES ──────────────────────────────────────────────────────────
//
//  Pillow's rules, so that a tag read here equals the tag Pillow reads:
//    * count 1            -> a scalar
//    * ASCII              -> a string, one trailing NUL stripped
//    * UNDEFINED / BYTE   -> a byte string
//    * RATIONAL/SRATIONAL -> [numerator, denominator]  (Pillow's
//                            IFDRational, which prints as a fraction)
//    * anything else, count > 1 -> a list
//
//      include "./pillow/exif.b" as exif;
//      $ex = exif.load($im.info.exif);          // from a JPEG's APP1
//      print(exif.tag_name(271));               // "Make"
//      print($ex.ifd0["271"]);
//      $ex2 = exif.empty();
//      exif.put($ex2, "ifd0", 274, 6);          // Orientation = 6
//      $bytes = exif.tobytes($ex2);             // an APP1 payload
//
//  docs/pillow-architecture.md §6.5.
// ════════════════════════════════════════════════════════════════════════

include "./exif_tags.b" as tags;

// TIFF field types: byte width, and whether the value is signed.
//   1 BYTE  2 ASCII  3 SHORT  4 LONG  5 RATIONAL  6 SBYTE  7 UNDEFINED
//   8 SSHORT  9 SLONG  10 SRATIONAL  11 FLOAT  12 DOUBLE  13 IFD
//   16 LONG8  17 SLONG8  18 IFD8      (BigTIFF, which Pillow also reads)
$_SIZE = {"1": 1, "2": 1, "3": 2, "4": 4, "5": 8, "6": 1, "7": 1, "8": 2,
          "9": 4, "10": 8, "11": 4, "12": 8, "13": 4, "16": 8, "17": 8, "18": 8};

$MAX_ENTRIES = 4096;          // per IFD; a real one has tens
$MAX_IFDS = 64;               // across the whole block, chains included

def tag_name($t)     { $n = tags.TAGS[str($t)]; if ($n == null) { return "Tag " + str($t); } return $n; }
def gps_tag_name($t)  { $n = tags.GPSTAGS[str($t)]; if ($n == null) { return "GPSTag " + str($t); } return $n; }
def TAGS()    { return tags.TAGS; }
def GPSTAGS() { return tags.GPSTAGS; }
def IFD()     { return tags.IFD; }
def BASE()    { return tags.BASE; }
def GPS()     { return tags.GPS; }
def INTEROP() { return tags.INTEROP; }

// ─── Reading ────────────────────────────────────────────────────────────

def _u16($b, $i, $le) {
    if ($le) { return $b[$i] + $b[$i + 1] * 256; }
    return $b[$i] * 256 + $b[$i + 1];
}
def _u32($b, $i, $le) {
    if ($le) {
        return $b[$i] + $b[$i + 1] * 256 + $b[$i + 2] * 65536 + $b[$i + 3] * 16777216;
    }
    return $b[$i] * 16777216 + $b[$i + 1] * 65536 + $b[$i + 2] * 256 + $b[$i + 3];
}
def _s32($v) { if ($v >= 2147483648) { return $v - 4294967296; } return $v; }
def _s16($v) { if ($v >= 32768) { return $v - 65536; } return $v; }

// _f32($bits) -> the IEEE-754 single that `bits` spells.
def _f32($bits) {
    $sign = 1;
    if ($bits >= 2147483648) { $sign = 0 - 1; $bits = $bits - 2147483648; }
    $e = floor($bits / 8388608);
    $m = $bits % 8388608;
    if ($e == 0) {
        if ($m == 0) { return $sign * 0; }
        return $sign * $m * pow(2, 0 - 149);
    }
    if ($e == 255) { if ($m == 0) { return $sign * INF; } return NAN; }
    return $sign * (1 + $m / 8388608) * pow(2, $e - 127);
}

// _f64($hi, $lo) -> the IEEE-754 double that two 32-bit halves spell.
def _f64($hi, $lo) {
    $sign = 1;
    if ($hi >= 2147483648) { $sign = 0 - 1; $hi = $hi - 2147483648; }
    $e = floor($hi / 1048576);
    $m = ($hi % 1048576) * 4294967296 + $lo;
    if ($e == 0) {
        if ($m == 0) { return $sign * 0; }
        return $sign * $m * pow(2, 0 - 1074);
    }
    if ($e == 2047) { if ($m == 0) { return $sign * INF; } return NAN; }
    return $sign * (1 + $m / 4503599627370496) * pow(2, $e - 1023);
}

// load($data) -> {"endian", "ifd0", "exif", "gps", "interop", "ifd1",
//                 "makernote", "thumbnail"}
//
// $data is either a JPEG APP1 payload beginning "Exif\0\0" or a bare TIFF
// block. Every IFD is a dict keyed by the tag number as a STRING, because
// that is how a Bantu dict is keyed.
def load($data) {
    if (type($data) != "string") { throw "exif.load: expected the APP1 payload as a string"; }
    $off = 0;
    if (len($data) >= 6 && substr($data, 0, 6) == "Exif\x00\x00") { $off = 6; }
    $raw = substr($data, $off, len($data) - $off);
    if (len($raw) < 8) { throw "exif: a TIFF block is at least 8 bytes, got " + str(len($raw)); }
    $b = bytes($raw);
    $le = false;
    if ($b[0] == 73 && $b[1] == 73) { $le = true; }
    else if ($b[0] == 77 && $b[1] == 77) { $le = false; }
    else { throw "exif: byte order is neither II nor MM"; }
    $magic = _u16($b, 2, $le);
    if ($magic != 42 && $magic != 43) {
        throw "exif: TIFF magic is " + str($magic) + ", not 42 (or 43 for BigTIFF)";
    }
    $first = _u32($b, 4, $le);

    $st = {"b": $b, "len": len($raw), "le": $le, "seen": {}, "ifds": 0};
    $out = {"endian": "MM", "ifd0": {}, "exif": {}, "gps": {}, "interop": {},
            "ifd1": {}, "makernote": null, "thumbnail": null, "raw": $raw};
    if ($le) { $out.endian = "II"; }

    $r0 = _readIfd($st, $first);
    $out.ifd0 = $r0.tags;

    // The sub-IFDs Pillow exposes, each reached through a pointer tag.
    $pe = $out.ifd0[str(tags.IFD.Exif)];
    if ($pe != null) { $out.exif = _readIfd($st, _asOffset($pe)).tags; }
    $pg = $out.ifd0[str(tags.IFD.GPSInfo)];
    if ($pg != null) { $out.gps = _readIfd($st, _asOffset($pg)).tags; }
    $pi = $out.exif[str(tags.IFD.Interop)];
    if ($pi != null) { $out.interop = _readIfd($st, _asOffset($pi)).tags; }
    $pm = $out.exif[str(tags.IFD.MakerNote)];
    if ($pm != null) { $out.makernote = $pm; }

    // IFD1 is the thumbnail directory, reached by the next-IFD pointer.
    if ($r0.next > 0) {
        $r1 = _readIfd($st, $r0.next);
        $out.ifd1 = $r1.tags;
        $o = $out.ifd1["513"];
        $n = $out.ifd1["514"];
        if ($o != null && $n != null) {
            $oo = _asOffset($o);
            $nn = _asOffset($n);
            if ($oo >= 0 && $nn > 0 && $oo + $nn <= $st.len) {
                $out.thumbnail = substr($raw, $oo, $nn);
            }
        }
    }
    return $out;
}

def _asOffset($v) {
    if (type($v) == "list") { if (len($v) == 0) { return 0 - 1; } return floor($v[0]); }
    if (type($v) != "number") { return 0 - 1; }
    return floor($v);
}

// _readIfd($st, $at) -> {"tags": {...}, "next": offset}
//
// The loop guard and the bounds checks of the header comment live here.
def _readIfd($st, $at) {
    if ($at < 0 || $at + 2 > $st.len) {
        throw "exif: an IFD at offset " + str($at) + " is outside a " +
              str($st.len) + "-byte block";
    }
    if ($st.seen[str($at)] != null) {
        throw "exif: the IFD chain points back at offset " + str($at) +
              "; a self-referential directory is not followed";
    }
    $st.seen[str($at)] = true;
    $st.ifds = $st.ifds + 1;
    if ($st.ifds > $MAX_IFDS) {
        throw "exif: more than " + str($MAX_IFDS) + " directories in one block";
    }
    $b = $st.b;
    $le = $st.le;
    $n = _u16($b, $at, $le);
    if ($n > $MAX_ENTRIES) {
        throw "exif: an IFD declaring " + str($n) + " entries (limit " +
              str($MAX_ENTRIES) + ")";
    }
    if ($at + 2 + $n * 12 + 4 > $st.len) {
        throw "exif: an IFD of " + str($n) + " entries at offset " + str($at) +
              " runs past the end of a " + str($st.len) + "-byte block";
    }
    $tagsOut = {};
    $i = 0;
    while ($i < $n) {
        $e = $at + 2 + $i * 12;
        $tag = _u16($b, $e, $le);
        $typ = _u16($b, $e + 2, $le);
        $cnt = _u32($b, $e + 4, $le);
        $i = $i + 1;
        $sz = $_SIZE[str($typ)];
        if ($sz == null) { continue; }               // an unknown type is skipped, as Pillow does
        // The overflow check, before any read: count * size, in doubles,
        // which are exact past 2^53 and so cannot wrap here.
        if ($cnt * $sz > $st.len) { continue; }
        $vp = $e + 8;
        if ($cnt * $sz > 4) {
            $vp = _u32($b, $e + 8, $le);
            if ($vp < 0 || $vp + $cnt * $sz > $st.len) { continue; }
        }
        $tagsOut[str($tag)] = _readValue($b, $vp, $typ, $cnt, $le, $st);
    }
    $next = _u32($b, $at + 2 + $n * 12, $le);
    return {"tags": $tagsOut, "next": $next};
}

def _readValue($b, $vp, $typ, $cnt, $le, $st) {
    if ($typ == 2) {
        // ASCII: one trailing NUL stripped, as Pillow strips it.
        $s = [];
        $k = 0;
        while ($k < $cnt) { push($s, chr($b[$vp + $k])); $k = $k + 1; }
        $str = join($s, "");
        if (len($str) > 0 && $str[len($str) - 1] == "\x00") {
            $str = substr($str, 0, len($str) - 1);
        }
        return $str;
    }
    if ($typ == 7 || $typ == 1 || $typ == 6) {
        // UNDEFINED and BYTE come back as bytes, which is Pillow's rule for
        // 7 and for 1 when the count is over one.
        if ($cnt == 1 && $typ != 7) { return $b[$vp]; }
        $s = [];
        $k = 0;
        while ($k < $cnt) { push($s, chr($b[$vp + $k])); $k = $k + 1; }
        return join($s, "");
    }
    $vals = [];
    $k = 0;
    while ($k < $cnt) {
        if ($typ == 3) { push($vals, _u16($b, $vp + $k * 2, $le)); }
        else if ($typ == 8) { push($vals, _s16(_u16($b, $vp + $k * 2, $le))); }
        else if ($typ == 4 || $typ == 13) { push($vals, _u32($b, $vp + $k * 4, $le)); }
        else if ($typ == 9) { push($vals, _s32(_u32($b, $vp + $k * 4, $le))); }
        else if ($typ == 5) {
            push($vals, [_u32($b, $vp + $k * 8, $le), _u32($b, $vp + $k * 8 + 4, $le)]);
        }
        else if ($typ == 10) {
            push($vals, [_s32(_u32($b, $vp + $k * 8, $le)),
                         _s32(_u32($b, $vp + $k * 8 + 4, $le))]);
        }
        else if ($typ == 11) { push($vals, _f32(_u32($b, $vp + $k * 4, $le))); }
        else if ($typ == 12) {
            $h = _u32($b, $vp + $k * 8, $le);
            $l = _u32($b, $vp + $k * 8 + 4, $le);
            if ($le) { push($vals, _f64($l, $h)); } else { push($vals, _f64($h, $l)); }
        }
        else if ($typ == 16 || $typ == 18) {
            $h = _u32($b, $vp + $k * 8, $le);
            $l = _u32($b, $vp + $k * 8 + 4, $le);
            if ($le) { push($vals, $h + $l * 4294967296); }
            else { push($vals, $h * 4294967296 + $l); }
        }
        else if ($typ == 17) {
            $h = _u32($b, $vp + $k * 8, $le);
            $l = _u32($b, $vp + $k * 8 + 4, $le);
            if ($le) { $u = $h + $l * 4294967296; } else { $u = $h * 4294967296 + $l; }
            if ($u >= 9223372036854775808) { $u = $u - 18446744073709551616; }
            push($vals, $u);
        }
        $k = $k + 1;
    }
    if ($cnt == 1) { return $vals[0]; }
    return $vals;
}

// ─── Writing ────────────────────────────────────────────────────────────

// empty() -> a blank EXIF structure, ready for put().
def empty() {
    return {"endian": "MM", "ifd0": {}, "exif": {}, "gps": {}, "interop": {},
            "ifd1": {}, "makernote": null, "thumbnail": null, "raw": null};
}

// put($ex, $which, $tag, $value) -> set a tag. $which is "ifd0", "exif",
// "gps", "interop" or "ifd1".
def put($ex, $which, $tag, $value) {
    $d = $ex[$which];
    if ($d == null) { throw "exif.put: no IFD called '" + str($which) + "'"; }
    $d[str($tag)] = $value;
    $ex[$which] = $d;
    return $ex;
}

// get($ex, $which, $tag) -> the tag's value, or null.
def get($ex, $which, $tag) {
    $d = $ex[$which];
    if ($d == null) { return null; }
    return $d[str($tag)];
}

// remove($ex, $which, $tag) -> the tag deleted, if it was there.
def remove($ex, $which, $tag) {
    $d = $ex[$which];
    if ($d == null) { return false; }
    if ($d[str($tag)] == null) { return false; }
    $out = {};
    for $k, $v in $d.items() { if ($k != str($tag)) { $out[$k] = $v; } }
    $ex[$which] = $out;
    return true;
}

// _typeOf($tag, $value) -> the TIFF type to write it as.
//
// The type is a property of the TAG in the EXIF spec, not of the value, so
// the few tags whose type is not guessable from the value are tabulated.
// Everything else follows the value: a string is ASCII, a [n, d] pair is
// RATIONAL, a whole number that fits is SHORT, anything larger is LONG.
$_TYPE_BY_TAG = {
    "33434": 5, "33437": 5, "37377": 10, "37378": 5, "37379": 10,
    "37380": 10, "37381": 5, "37386": 5, "41483": 5, "41486": 5,
    "41487": 5, "41988": 5, "41989": 3, "41990": 3,
    "282": 5, "283": 5, "531": 3, "296": 3, "274": 3
};

def _typeOf($tag, $v) {
    $forced = $_TYPE_BY_TAG[str($tag)];
    if (type($v) == "string") { return 2; }
    if (type($v) == "list") {
        if (len($v) == 2 && type($v[0]) == "number" && type($v[1]) == "number" &&
            $v[0] == floor($v[0]) && $v[1] == floor($v[1]) && $v[1] != 0 &&
            ($forced == 5 || $forced == 10)) {
            return $forced;
        }
        if (len($v) == 0) { return 4; }
        $i = 0;
        $allShort = true;
        $anyFloat = false;
        $anyNeg = false;
        while ($i < len($v)) {
            $e = $v[$i];
            if (type($e) == "list") { return 5; }
            if (type($e) != "number") { throw "exif: cannot write " + type($e) + " in a tag"; }
            if ($e != floor($e)) { $anyFloat = true; }
            if ($e < 0) { $anyNeg = true; }
            if ($e < 0 || $e > 65535) { $allShort = false; }
            $i = $i + 1;
        }
        if ($anyFloat) { return 12; }
        if ($anyNeg) { return 9; }
        if ($allShort) { return 3; }
        return 4;
    }
    if (type($v) == "number") {
        if ($forced == 5 || $forced == 10) { return $forced; }
        if ($v != floor($v)) { return 12; }
        if ($v < 0) { return 9; }
        if ($forced == 3 && $v <= 65535) { return 3; }
        if ($v <= 65535) { return 3; }
        if ($v <= 4294967295) { return 4; }
        return 16;
    }
    throw "exif: cannot write a " + type($v) + " in a tag";
}

def _p16($v, $le) {
    $x = floor($v) % 65536;
    if ($x < 0) { $x = $x + 65536; }
    if ($le) { return chr($x % 256) + chr(floor($x / 256)); }
    return chr(floor($x / 256)) + chr($x % 256);
}
def _p32($v, $le) {
    $x = floor($v) % 4294967296;
    if ($x < 0) { $x = $x + 4294967296; }
    $b0 = $x % 256;
    $b1 = floor($x / 256) % 256;
    $b2 = floor($x / 65536) % 256;
    $b3 = floor($x / 16777216) % 256;
    if ($le) { return chr($b0) + chr($b1) + chr($b2) + chr($b3); }
    return chr($b3) + chr($b2) + chr($b1) + chr($b0);
}

// _packF64($v, $le) -> eight bytes of IEEE-754 double.
def _packF64($v, $le) {
    $a = nd([$v], "f64");
    $s = nd_tobytes($a, "f64le");
    if ($le) { return $s; }
    $out = [];
    $k = 7;
    while ($k >= 0) { push($out, $s[$k]); $k = $k - 1; }
    return join($out, "");
}

// _packValue($tag, $v, $le) -> [type, count, bytes]
def _packValue($tag, $v, $le) {
    $typ = _typeOf($tag, $v);
    if ($typ == 2) {
        return [2, len($v) + 1, $v + "\x00"];
    }
    $list = $v;
    if (type($v) != "list") { $list = [$v]; }
    else if ($typ == 5 || $typ == 10) {
        if (len($v) == 2 && type($v[0]) == "number" && type($v[1]) == "number") {
            $list = [$v];
        }
    }
    $parts = [];
    $i = 0;
    while ($i < len($list)) {
        $e = $list[$i];
        if ($typ == 3 || $typ == 8) { push($parts, _p16($e, $le)); }
        else if ($typ == 4 || $typ == 9 || $typ == 13) { push($parts, _p32($e, $le)); }
        else if ($typ == 5 || $typ == 10) {
            if (type($e) == "list") { push($parts, _p32($e[0], $le) + _p32($e[1], $le)); }
            else { push($parts, _p32($e, $le) + _p32(1, $le)); }
        }
        else if ($typ == 12) { push($parts, _packF64($e, $le)); }
        else if ($typ == 16) {
            $lo = $e % 4294967296;
            $hi = floor($e / 4294967296);
            if ($le) { push($parts, _p32($lo, $le) + _p32($hi, $le)); }
            else { push($parts, _p32($hi, $le) + _p32($lo, $le)); }
        }
        else { throw "exif: no writer for TIFF type " + str($typ); }
        $i = $i + 1;
    }
    return [$typ, len($list), join($parts, "")];
}

// tobytes($ex) -> the APP1 payload, "Exif\0\0" and all, ready to be written
// into a JPEG or handed to Pillow's Image.getexif().
//
// The structure is built in one pass over three IFDs: IFD0 first, then the
// Exif, GPS and Interop sub-IFDs after it, with every value over four bytes
// appended to a data area that follows all of them. Entries are sorted by
// tag number, which the spec requires and which Pillow's reader relies on.
def tobytes($ex) {
    $le = $ex.endian == "II";

    // The sub-IFDs that will exist, so IFD0 can carry pointers to them.
    $subs = [];
    if (len(keys($ex.exif)) > 0 || $ex.makernote != null) {
        push($subs, {"name": "exif", "ptr": tags.IFD.Exif});
    }
    if (len(keys($ex.gps)) > 0) { push($subs, {"name": "gps", "ptr": tags.IFD.GPSInfo}); }

    // Interop lives inside the Exif IFD, so it only exists if that does.
    $wantInterop = len(keys($ex.interop)) > 0 && len($subs) > 0;

    // Two passes: the first lays out sizes to learn every offset, the
    // second writes with those offsets. A single pass cannot, because
    // IFD0's pointer tags name offsets that are only known once the
    // sub-IFDs are sized.
    $ifd0 = _entryList($ex.ifd0, $le);
    $exifEntries = _entryList($ex.exif, $le);
    if ($ex.makernote != null) {
        push($exifEntries, {"tag": tags.IFD.MakerNote, "typ": 7,
                            "cnt": len($ex.makernote), "data": $ex.makernote});
        $exifEntries = _sortEntries($exifEntries);
    }
    $gpsEntries = _entryList($ex.gps, $le);
    $interopEntries = _entryList($ex.interop, $le);

    // Reserve the pointer entries in IFD0 (and Interop's in Exif).
    $i = 0;
    while ($i < len($subs)) {
        push($ifd0, {"tag": $subs[$i].ptr, "typ": 4, "cnt": 1, "data": _p32(0, $le)});
        $i = $i + 1;
    }
    $ifd0 = _sortEntries($ifd0);
    if ($wantInterop) {
        push($exifEntries, {"tag": tags.IFD.Interop, "typ": 4, "cnt": 1, "data": _p32(0, $le)});
        $exifEntries = _sortEntries($exifEntries);
    }

    // Layout. Every IFD is 2 + 12n + 4 bytes; its oversized values follow.
    $pos = 8;
    $lay = {};
    $order = ["ifd0"];
    $dirs = {"ifd0": $ifd0};
    if ($wantInterop) { $dirs.interop = $interopEntries; }
    $i = 0;
    while ($i < len($subs)) {
        $nm = $subs[$i].name;
        if ($nm == "exif") { $dirs.exif = $exifEntries; } else { $dirs.gps = $gpsEntries; }
        push($order, $nm);
        $i = $i + 1;
    }
    if ($wantInterop) { push($order, "interop"); }

    $i = 0;
    while ($i < len($order)) {
        $nm = $order[$i];
        $ents = $dirs[$nm];
        $lay[$nm] = $pos;
        $pos = $pos + 2 + len($ents) * 12 + 4;
        $k = 0;
        while ($k < len($ents)) {
            $d = $ents[$k].data;
            if (len($d) > 4) { $pos = $pos + len($d) + len($d) % 2; }
            $k = $k + 1;
        }
        $i = $i + 1;
    }

    // Fill the pointer entries now that the offsets are known.
    $i = 0;
    while ($i < len($ifd0)) {
        $t = $ifd0[$i].tag;
        if ($t == tags.IFD.Exif && $lay.exif != null) { $ifd0[$i].data = _p32($lay.exif, $le); }
        if ($t == tags.IFD.GPSInfo && $lay.gps != null) { $ifd0[$i].data = _p32($lay.gps, $le); }
        $i = $i + 1;
    }
    $i = 0;
    while ($i < len($exifEntries)) {
        if ($exifEntries[$i].tag == tags.IFD.Interop && $lay.interop != null) {
            $exifEntries[$i].data = _p32($lay.interop, $le);
        }
        $i = $i + 1;
    }
    $dirs.ifd0 = $ifd0;
    if ($lay.exif != null) { $dirs.exif = $exifEntries; }

    // Write.
    $out = [];
    if ($le) { push($out, "II"); } else { push($out, "MM"); }
    push($out, _p16(42, $le));
    push($out, _p32($lay.ifd0, $le));
    $i = 0;
    while ($i < len($order)) {
        $nm = $order[$i];
        $ents = $dirs[$nm];
        $i = $i + 1;
        // The data area for this IFD starts after its entry table.
        $dp = $lay[$nm] + 2 + len($ents) * 12 + 4;
        push($out, _p16(len($ents), $le));
        $tail = [];
        $k = 0;
        while ($k < len($ents)) {
            $e = $ents[$k];
            $k = $k + 1;
            push($out, _p16($e.tag, $le));
            push($out, _p16($e.typ, $le));
            push($out, _p32($e.cnt, $le));
            if (len($e.data) > 4) {
                push($out, _p32($dp, $le));
                push($tail, $e.data);
                $dp = $dp + len($e.data);
                if (len($e.data) % 2 == 1) { push($tail, "\x00"); $dp = $dp + 1; }
            } else {
                $pad = $e.data;
                while (len($pad) < 4) { $pad = $pad + "\x00"; }
                push($out, $pad);
            }
        }
        push($out, _p32(0, $le));          // no next IFD
        push($out, join($tail, ""));
    }
    return "Exif\x00\x00" + join($out, "");
}

def _entryList($d, $le) {
    $out = [];
    if ($d == null) { return $out; }
    for $k, $v in $d.items() {
        $tag = num($k);
        if ($tag == tags.IFD.Exif || $tag == tags.IFD.GPSInfo ||
            $tag == tags.IFD.Interop) { continue; }
        $p = _packValue($tag, $v, $le);
        push($out, {"tag": $tag, "typ": $p[0], "cnt": $p[1], "data": $p[2]});
    }
    return _sortEntries($out);
}

// _sortEntries($list) -> sorted by tag number. An insertion sort: an IFD
// has tens of entries, so the simple one is the right one.
def _sortEntries($list) {
    $out = [];
    $i = 0;
    while ($i < len($list)) {
        $e = $list[$i];
        $j = 0;
        $at = len($out);
        while ($j < len($out)) {
            if ($out[$j].tag > $e.tag) { $at = $j; break; }
            $j = $j + 1;
        }
        $new = [];
        $k = 0;
        while ($k < $at) { push($new, $out[$k]); $k = $k + 1; }
        push($new, $e);
        while ($k < len($out)) { push($new, $out[$k]); $k = $k + 1; }
        $out = $new;
        $i = $i + 1;
    }
    return $out;
}

// ─── ImageOps.exif_transpose ────────────────────────────────────────────

// The EXIF Orientation tag, and the transpose each value calls for. These
// are Pillow's own mappings (ImageOps.exif_transpose), and the transpose
// ids are pillow's own constants.
$ORIENTATION = 274;
$_METHOD = {"2": 0, "3": 3, "4": 1, "5": 5, "6": 4, "7": 6, "8": 2};

// transpose_for($orientation) -> the transpose id, or null for 1/unknown.
def transpose_for($o) {
    if ($o == null) { return null; }
    return $_METHOD[str(floor($o))];
}

// orientation_of($info) -> the Orientation tag from an image's info dict,
// or null. Reads info["exif"], which is where every codec puts it.
def orientation_of($info) {
    if ($info == null || $info.exif == null) { return null; }
    $ex = null;
    try { $ex = load($info.exif); } catch ($e) { return null; }
    return get($ex, "ifd0", $ORIENTATION);
}

// exif_transpose($im) -> a new image with the orientation applied and the
// tag removed, or a copy if there is nothing to do. ImageOps.exif_transpose.
def exif_transpose($im) {
    $o = orientation_of($im.info);
    $m = transpose_for($o);
    if ($m == null) { return $im.copy(); }
    $out = $im.transpose($m);
    // Pillow drops the tag from the result, and from the thumbnail IFD too,
    // so that applying the transform twice does not rotate twice.
    if ($out.info.exif != null) {
        $ok = true;
        $ex = null;
        try { $ex = load($out.info.exif); } catch ($e) { $ok = false; }
        if ($ok) {
            remove($ex, "ifd0", $ORIENTATION);
            remove($ex, "ifd1", $ORIENTATION);
            $out.info.exif = tobytes($ex);
        }
    }
    return $out;
}
