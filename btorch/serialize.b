// ════════════════════════════════════════════════════════════════════════════
//  btorch/serialize.b — saving and loading state (phase 7)
//
//  Format and rationale: docs/btorch-architecture.md §7.5.2.
//
//      offset 0   "BTSD"         4 bytes, magic
//      offset 4   version        u32 LE  (currently 1)
//      offset 8   headerLen      u32 LE
//      offset 12  header         JSON, headerLen bytes
//      offset 12+headerLen       payload: every entry's raw bytes, in order
//
//  Why binary and not JSON all the way down: nd_to_list on a 61,706-parameter
//  model builds 61,706 Bantu numbers, and str() of a double gives six
//  significant digits -- so a text round trip is both slow and LOSSY unless
//  every value is printed in full. nd_tobytes/nd_frombytes are exact.
//
//  Two things this file is careful about:
//
//  1. json.parse in this interpreter returns garbage ({: null}) or null for
//     malformed input rather than raising. The loader therefore validates
//     every field itself and never assumes a parse failed loudly.
//
//  2. A truncated file and a corrupted one are DIFFERENT accidents, with
//     different causes, and are reported differently. The length check and
//     the SHA-256 check are separate on purpose; collapsing them into "bad
//     file" throws away the only diagnostic the reader has.
// ════════════════════════════════════════════════════════════════════════════

include "btorch.b" as bt;

$_BTSD = {"magic": "BTSD", "version": 1};

def _need() {
    if (nd_tobytes == null) {
        throw "btorch.serialize needs numba's nd_* builtins -- the bantu binary "
            + "is older than this library. Run 'bantu doctor'.";
    }
    return null;
}

// ── u32 little-endian, through the array codec ──────────────────────────────
// Not hand-rolled byte arithmetic: nd_tobytes already defines the encoding,
// and two implementations of "little-endian u32" is one more than can stay in
// agreement.
def _u32($v) { return nd_tobytes(nd([$v], "i64"), "u32"); }
def _readU32($s, $at) {
    return nd_get(nd_frombytes(substr($s, $at, 4), "u32", [1]), [0]);
}

// The element type that writes a dtype without loss. i64 is here because A-extra
// added it to nd_tobytes for exactly this -- numba HAS an i64 dtype and used to
// offer nothing wider than i32 to serialize it with.
def _elemFor($dtype) {
    if ($dtype == "f64")  { return "f64"; }
    // f32 is written as f32, not widened to f64 (D2c). Widening would round-trip
    // correctly but double every checkpoint an f32 model writes, which is half
    // the reason to train in f32 at all.
    if ($dtype == "f32")  { return "f32"; }
    if ($dtype == "i64")  { return "i64"; }
    if ($dtype == "bool") { return "u8"; }
    throw "btorch.serialize: no lossless encoding for dtype '" + str($dtype) + "'";
}

// ── collecting what to save ────────────────────────────────────────────────
// A Module contributes its parameters AND its buffers. Buffers are the part
// that is easy to forget and expensive to forget: a BatchNorm's running mean
// and variance are buffers precisely so no optimizer steps them, and a file
// that omitted them would reload a model whose eval() is wrong -- which trains
// and scores fine right up until it does not.
def _entriesOf($obj) {
    $out = [];
    if (type($obj) == "instance" && $obj.__bt_module == true) {
        $named = $obj.named_parameters();
        $i = 0;
        while ($i < len($named)) {
            push($out, [$named[$i][0], $named[$i][1].data, "param"]);
            $i = $i + 1;
        }
        $nb = $obj.named_buffers();
        $i = 0;
        while ($i < len($nb)) {
            push($out, [$nb[$i][0], $nb[$i][1].data, "buffer"]);
            $i = $i + 1;
        }
        return $out;
    }
    if (type($obj) == "dict") {
        for $k, $v in $obj.items() {
            $arr = $v;
            if (bt.is_tensor($v)) { $arr = $v.data; }
            if (type($arr) == "list") { $arr = nd($arr, "f64"); }
            if (type($arr) != "ndarray") {
                throw "btorch.serialize: '" + str($k) + "' is a " + type($v)
                    + " -- save() takes a Module, or a dict of tensors/arrays";
            }
            push($out, [str($k), $arr, "param"]);
        }
        return $out;
    }
    throw "btorch.serialize.save(obj, path): obj must be a Module or a dict of "
        + "tensors (got " + type($obj) + ")";
}

// ── save ───────────────────────────────────────────────────────────────────
def save($obj, $path) {
    _need();
    if (type($path) != "string") { throw "btorch.serialize.save(obj, path): path must be a string"; }
    $entries = _entriesOf($obj);

    $meta = [];
    $payload = "";
    $i = 0;
    while ($i < len($entries)) {
        $name = $entries[$i][0];
        $arr = $entries[$i][1];
        $kind = $entries[$i][2];
        $dt = nd_dtype($arr);
        $bytes = nd_tobytes($arr, _elemFor($dt));
        push($meta, {
            "name": $name,
            "kind": $kind,
            "dtype": $dt,
            "shape": nd_shape($arr),
            "nbytes": len($bytes)
        });
        $payload = $payload + $bytes;
        $i = $i + 1;
    }

    $header = json.stringify({
        "version": $_BTSD.version,
        "entries": $meta,
        "payload_bytes": len($payload),
        "payload_sha256": native_sha256($payload)
    });

    $blob = $_BTSD.magic + _u32($_BTSD.version) + _u32(len($header)) + $header + $payload;
    writefile($path, $blob, "wb");
    // writefile raises on a failed write, so reaching here means the bytes
    // went out. Report what was written, which is what a caller logs.
    return {"entries": len($meta), "bytes": len($blob), "path": $path};
}

// ── load ───────────────────────────────────────────────────────────────────
// Returns {"entries": {name -> ndarray}, "kinds": {name -> "param"|"buffer"},
//          "version": n, "order": [names in file order]}.
def load($path) {
    _need();
    if (type($path) != "string") { throw "btorch.serialize.load(path): path must be a string"; }
    if (!file_exists($path)) {
        throw "btorch.serialize.load: no such file '" + $path + "'";
    }
    $blob = readfile($path, "rb");
    $n = len($blob);
    if ($n < 12) {
        throw "btorch.serialize.load: '" + $path + "' is " + str($n) + " bytes, "
            + "too short to be a btorch state file (the header alone is 12)";
    }
    if (substr($blob, 0, 4) != $_BTSD.magic) {
        throw "btorch.serialize.load: '" + $path + "' is not a btorch state file "
            + "(expected magic 'BTSD', found '" + substr($blob, 0, 4) + "')";
    }
    $ver = _readU32($blob, 4);
    if ($ver > $_BTSD.version) {
        throw "btorch.serialize.load: '" + $path + "' was written by a newer "
            + "btorch (file format v" + str($ver) + ", this reader understands v"
            + str($_BTSD.version) + ") -- refusing rather than reading it "
            + "best-effort, because a field this reader does not know about is "
            + "a field it would silently drop";
    }
    $hlen = _readU32($blob, 8);
    if (12 + $hlen > $n) {
        throw "btorch.serialize.load: '" + $path + "' is TRUNCATED -- its header "
            + "claims " + str($hlen) + " bytes but only " + str($n - 12)
            + " remain after the magic";
    }
    $header = json.parse(substr($blob, 12, $hlen));
    // json.parse returns garbage or null for malformed input instead of
    // raising, so every field below is checked rather than trusted.
    if (type($header) != "dict") {
        throw "btorch.serialize.load: '" + $path + "' has a corrupt header "
            + "(it did not parse as a JSON object)";
    }
    $meta = $header.entries;
    if (type($meta) != "list") {
        throw "btorch.serialize.load: '" + $path + "' has a corrupt header "
            + "(no 'entries' list)";
    }
    $claimed = $header.payload_bytes;
    if (type($claimed) != "number") {
        throw "btorch.serialize.load: '" + $path + "' has a corrupt header "
            + "(no 'payload_bytes')";
    }
    $payload = substr($blob, 12 + $hlen, $n - 12 - $hlen);
    if (len($payload) != $claimed) {
        $missing = $claimed - len($payload);
        if ($missing > 0) {
            throw "btorch.serialize.load: '" + $path + "' is TRUNCATED -- "
                + str($missing) + " of " + str($claimed) + " payload bytes are missing";
        }
        throw "btorch.serialize.load: '" + $path + "' has " + str(0 - $missing)
            + " bytes MORE than its header claims (" + str($claimed) + ")";
    }
    $want = $header.payload_sha256;
    if (type($want) == "string") {
        $got = native_sha256($payload);
        if ($got != $want) {
            // The length matched and the bytes did not: this is corruption,
            // not truncation, and saying so is the whole point of the digest.
            throw "btorch.serialize.load: '" + $path + "' is CORRUPT -- the "
                + "payload is the right length but its SHA-256 is " + $got
                + ", not the " + $want + " recorded when it was saved";
        }
    }

    $out = {};
    $kinds = {};
    $order = [];
    $at = 0;
    $i = 0;
    while ($i < len($meta)) {
        $e = $meta[$i];
        if (type($e) != "dict") {
            throw "btorch.serialize.load: entry " + str($i) + " of '" + $path
                + "' is not an object";
        }
        $name = $e.name;
        if (type($name) != "string") {
            throw "btorch.serialize.load: entry " + str($i) + " of '" + $path
                + "' has no 'name'";
        }
        $dt = $e.dtype;
        $shape = $e.shape;
        $nb = $e.nbytes;
        if (type($dt) != "string" || type($shape) != "list" || type($nb) != "number") {
            throw "btorch.serialize.load: entry '" + $name + "' of '" + $path
                + "' is missing dtype, shape or nbytes";
        }
        if ($at + $nb > len($payload)) {
            throw "btorch.serialize.load: entry '" + $name + "' of '" + $path
                + "' claims " + str($nb) + " bytes at offset " + str($at)
                + ", past the end of a " + str(len($payload)) + "-byte payload";
        }
        $raw = substr($payload, $at, $nb);
        $arr = nd_frombytes($raw, _elemFor($dt), $shape);
        if (nd_dtype($arr) != $dt) { $arr = nd_astype($arr, $dt); }
        $out[$name] = $arr;
        $kinds[$name] = $e.kind;
        push($order, $name);
        $at = $at + $nb;
        $i = $i + 1;
    }
    if ($at != len($payload)) {
        throw "btorch.serialize.load: '" + $path + "' has " + str(len($payload) - $at)
            + " payload bytes that no entry accounts for -- the header and the "
            + "payload disagree";
    }
    return {"entries": $out, "kinds": $kinds, "version": $ver, "order": $order};
}

// ── the convenience pair ───────────────────────────────────────────────────
def save_model($module, $path) { return save($module, $path); }

// Loads parameters AND buffers into an existing module, in place, so an
// optimizer already holding those tensors keeps pointing at the right ones.
// Returns a count of each, so a caller can assert it got what it expected.
def load_model($module, $path) {
    _need();
    if (type($module) != "instance" || $module.__bt_module != true) {
        throw "btorch.serialize.load_model(module, path): the first argument is not a Module";
    }
    $file = load($path);
    return $module.load_state_arrays($file.entries);
}
