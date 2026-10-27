// ============================================================================
// lugha/weights.b — the .btw tensor container
//
// Format (little-endian, which is every platform Bantu builds for):
//
//   0   4          magic "BTW1"
//   4   4  u32     format version (1)
//   8   4  u32     tensor count N
//   12  4  u32     flags (0)
//   16  ...        N records:
//          2  u16  name length L
//          L       name, ASCII
//          1  u8   dtype (0 = f64)
//          1  u8   ndim
//          4*ndim  u32 shape
//   then the payloads, in record order, 8 bytes per f64.
//
// The header is parsed from a Bantu string (substr is byte-indexed and a
// string carries arbitrary bytes intact); the payloads never become Bantu
// values at all — np.fromfile reads them straight into array buffers at the
// recorded byte offsets. That is the whole reason nd_fromfile exists: a
// 1.5M-element Bantu list would cost ~285 MB to hold an 11 MB model.
//
// Exposes:
//   save($path, $names, $tensors)  -> bytes written
//   load($path)                    -> dict of name -> array
//   describe($path)                -> list of {name, shape, bytes}
// ============================================================================

include "numba" as np;

const $MAGIC = "BTW1";
const $HEADER_FIXED = 16;

// ── little-endian readers over a byte-carrying string ───────────────────────
def _u8($s, $i)  { return bytes(substr($s, $i, 1))[0]; }

def _u16($s, $i) {
    $b = bytes(substr($s, $i, 2));
    return $b[0] + $b[1] * 256;
}

def _u32($s, $i) {
    $b = bytes(substr($s, $i, 4));
    return $b[0] + $b[1] * 256 + $b[2] * 65536 + $b[3] * 16777216;
}

// ── little-endian writers ───────────────────────────────────────────────────
// These RETURN the bytes rather than appending into a list passed in, because
// a Bantu list is a value: a list handed to a function is copied, and anything
// the function appends is appended to the copy. The caller extends its own
// list, which does mutate.
def _u16bytes($v) {
    return [fmod($v, 256), fmod(floor($v / 256), 256)];
}

def _u32bytes($v) {
    return [fmod($v, 256),
            fmod(floor($v / 256), 256),
            fmod(floor($v / 65536), 256),
            fmod(floor($v / 16777216), 256)];
}

// ── save ────────────────────────────────────────────────────────────────────
// $names is an ordered list; $tensors maps name -> array. The order is the
// file's order, and a reader does not need it — every record carries its own
// shape — but a stable order makes two checkpoints diffable.
def save($path, $names, $tensors) {
    $h = [];
    extend($h, bytes($MAGIC));
    extend($h, _u32bytes(1));
    extend($h, _u32bytes(len($names)));
    extend($h, _u32bytes(0));

    each ($name in $names) {
        $a = $tensors[$name];
        if ($a == null) { throw "lugha.weights.save: no tensor named '" + $name + "'"; }
        $nameBytes = bytes($name);
        extend($h, _u16bytes(len($nameBytes)));
        extend($h, $nameBytes);
        append($h, 0);                       // dtype f64
        $shape = np.shape($a);
        append($h, len($shape));
        each ($d in $shape) { extend($h, _u32bytes($d)); }
    }

    writefile($path, frombytes($h), "wb");
    $total = len($h);
    each ($name in $names) {
        $total = $total + np.tofile(np.astype($tensors[$name], "f64"), $path, true);
    }
    return $total;
}

// ── header walk, shared by load and describe ────────────────────────────────
def _records($path) {
    // The header is a few hundred bytes; the file may be megabytes. readfile
    // hands back a string rather than a list, so this is one string copy and
    // not one Value per byte.
    $s = readfile($path, "rb");
    if (len($s) < $HEADER_FIXED) { throw "lugha.weights: '" + $path + "' is too short to be a .btw file"; }
    if (substr($s, 0, 4) != $MAGIC) {
        throw "lugha.weights: '" + $path + "' is not a .btw file (magic was '"
              + substr($s, 0, 4) + "')";
    }
    $version = _u32($s, 4);
    if ($version != 1) {
        throw "lugha.weights: '" + $path + "' is format version " + str($version)
              + ", this build reads version 1";
    }
    $count = _u32($s, 8);
    $recs = [];
    $p = $HEADER_FIXED;
    $i = 0;
    while ($i < $count) {
        $nameLen = _u16($s, $p);
        $p = $p + 2;
        $name = substr($s, $p, $nameLen);
        $p = $p + $nameLen;
        $dtype = _u8($s, $p);
        $p = $p + 1;
        if ($dtype != 0) {
            throw "lugha.weights: tensor '" + $name + "' has dtype " + str($dtype)
                  + "; only 0 (f64) is defined";
        }
        $ndim = _u8($s, $p);
        $p = $p + 1;
        $shape = [];
        $elems = 1;
        $d = 0;
        while ($d < $ndim) {
            $dim = _u32($s, $p);
            $p = $p + 4;
            append($shape, $dim);
            $elems = $elems * $dim;
            $d = $d + 1;
        }
        append($recs, { "name": $name, "shape": $shape, "elems": $elems });
        $i = $i + 1;
    }
    // Payload offsets follow the whole header, in record order.
    $off = $p;
    $out = [];
    each ($r in $recs) {
        append($out, { "name": $r["name"], "shape": $r["shape"],
                       "offset": $off, "bytes": $r["elems"] * 8 });
        $off = $off + $r["elems"] * 8;
    }
    if ($off != len($s)) {
        throw "lugha.weights: '" + $path + "' says it holds " + str($off)
              + " bytes but the file is " + str(len($s));
    }
    return $out;
}

def describe($path) { return _records($path); }

def load($path) {
    $out = {};
    each ($r in _records($path)) {
        $out[$r["name"]] = np.fromfile($path, "f64", $r["shape"], $r["offset"]);
    }
    return $out;
}
