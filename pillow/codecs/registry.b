// ════════════════════════════════════════════════════════════════════════
//  pillow/codecs/registry.b — format sniffing and the open/save dispatch.
//
//  Pillow's plugin registry, with Pillow's names: register_open,
//  register_save, register_mime, register_extension.
//
//  THE LAYERING THAT MATTERS: a codec here deals in PIXEL BUFFERS, never in
//  Image objects. decode() returns a plain dict
//
//      {"mode": "RGB", "size": [w, h], "px": <buffer>,
//       "palette": <256x3 buffer or null>, "info": {...}}
//
//  and encode() takes one. So image.b can include this, and the codecs can
//  be included from here, without any module depending on image.b. Without
//  that rule there is an include cycle between every codec and the Image
//  object, and Bantu's cycle guard would silently skip one of them.
//
//  A codec registers itself with:
//      registry.register("PNG", ["png"], "image/png", $sniffFn, $openFn, $saveFn);
//  where $sniffFn($head) -> bool and the head is the first 32 bytes.
// ════════════════════════════════════════════════════════════════════════

// State lives in a dict: a function assigning to a bare $var would bind a
// function-local, not the module variable.
$_REG = {"ids": [], "by": {}, "ext": {}, "mime": {}};

// register($id, $extensions, $mime, $sniff, $open, $save)
//
// $sniff and $save may be null -- a read-only format registers no saver, and
// a format with no magic bytes (PPM's is weak, XBM has none) registers no
// sniffer and is found by extension.
def register($id, $exts, $mime, $sniff, $open, $save) {
    if (type($id) != "string" || len($id) == 0) {
        throw "registry.register: a format id is required";
    }
    if ($_REG.by[$id] == null) { push($_REG.ids, $id); }
    $_REG.by[$id] = {"id": $id, "exts": $exts, "mime": $mime,
                     "sniff": $sniff, "open": $open, "save": $save};
    $i = 0;
    while ($i < len($exts)) {
        $_REG.ext[$exts[$i]] = $id;
        $i = $i + 1;
    }
    if ($mime != null) { $_REG.mime[$id] = $mime; }
    return true;
}

// formats() -> every registered format id, in registration order.
def formats() {
    $out = [];
    $i = 0;
    while ($i < len($_REG.ids)) { push($out, $_REG.ids[$i]); $i = $i + 1; }
    return $out;
}

// can_read($id) / can_write($id)
def can_read($id)  { $e = $_REG.by[$id]; return $e != null && $e.open != null; }
def can_write($id) { $e = $_REG.by[$id]; return $e != null && $e.save != null; }

// extensions() -> {".png": "PNG", ...}, Pillow's Image.registered_extensions().
def extensions() {
    $out = {};
    for $k, $v in $_REG.ext.items() { $out["." + $k] = $v; }
    return $out;
}

// mime_for($id) -> the MIME type, or null.
def mime_for($id) { return $_REG.mime[$id]; }

// _extOf($path) -> the lowercased extension with no dot, or "".
def _extOf($path) {
    $i = len($path) - 1;
    while ($i >= 0) {
        if ($path[$i] == ".") {
            $raw = substr($path, $i + 1, len($path) - $i - 1);
            $out = [];
            $k = 0;
            while ($k < len($raw)) {
                $c = ord($raw[$k]);
                if ($c >= 65 && $c <= 90) { push($out, chr($c + 32)); }
                else { push($out, $raw[$k]); }
                $k = $k + 1;
            }
            return join($out, "");
        }
        if ($path[$i] == "/" || $path[$i] == "\\") { return ""; }
        $i = $i - 1;
    }
    return "";
}

// identify($data, $path) -> the format id, or null.
//
// Magic bytes first, extension second. That order is deliberate: a PNG named
// .jpg is a PNG, and trusting the name over the content is how a decoder gets
// fed the wrong format.
def identify($data, $path) {
    $head = $data;
    if (len($data) > 32) { $head = substr($data, 0, 32); }
    $i = 0;
    while ($i < len($_REG.ids)) {
        $e = $_REG.by[$_REG.ids[$i]];
        if ($e.sniff != null) {
            $fn = $e.sniff;
            $hit = false;
            try { $hit = $fn($head); } catch ($err) { $hit = false; }
            if ($hit) { return $e.id; }
        }
        $i = $i + 1;
    }
    if ($path != null) {
        $id = $_REG.ext[_extOf($path)];
        if ($id != null) { return $id; }
    }
    return null;
}

// decode($data, $path) -> the buffer dict, or a clear error.
//
// UnidentifiedImageError is Pillow's own name for this failure, and it is
// used here so a caller catching it reads the same way.
def decode($data, $path) {
    $id = identify($data, $path);
    if ($id == null) {
        throw "UnidentifiedImageError: cannot identify the image format" +
              _whereFrom($path);
    }
    $e = $_REG.by[$id];
    if ($e.open == null) {
        throw $id + ": this format is registered for writing only" + _whereFrom($path);
    }
    $fn = $e.open;
    $r = $fn($data);
    $r.format = $id;
    return $r;
}

// encode($spec, $id) -> the file bytes.
def encode($spec, $id) {
    $e = $_REG.by[$id];
    if ($e == null) { throw "unknown format '" + str($id) + "'"; }
    if ($e.save == null) { throw $id + ": this format is read-only in pillow"; }
    $fn = $e.save;
    return $fn($spec);
}

// format_for_path($path) -> the id implied by the extension, or null.
def format_for_path($path) { return $_REG.ext[_extOf($path)]; }

def _whereFrom($path) {
    if ($path == null) { return ""; }
    return " (" + $path + ")";
}
