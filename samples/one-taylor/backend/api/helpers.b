// ============================================================================
// one-taylor/backend/api/helpers.b — shared request/response plumbing
//
// sua.server.use() registers middleware that never runs (the dispatch loop
// has no middleware step), so there is no framework hook to hang auth or
// validation on. Shared logic is a function each handler calls, and these are
// those functions.
//
// Exposes:
//   fail($res, $code, $msg) / okJson($res, $obj)
//   bodyOf($req)            -> dict, or null
//   strField($body, $name, $max)
//   intField($body, $name, $dflt)
//   listField($body, $name) -> list of trimmed strings
//   pageLimit($req, $dflt, $cap)
// ============================================================================

def fail($res, $code, $msg) {
    $res.status($code).json({ "ok": false, "error": $msg });
    return null;
}

def okJson($res, $obj) {
    $obj["ok"] = true;
    $res.json($obj);
    return null;
}

// $req.body is parsed JSON when the content type contains "json", the raw
// string otherwise, and null when there is no body. Anything that is not an
// object is not a request this API understands.
def bodyOf($req) {
    $b = $req.body;
    if ($b == null) { return null; }
    if (type($b) != "dict") { return null; }
    return $b;
}

def strField($body, $name, $max) {
    if ($body == null) { return ""; }
    $v = $body[$name];
    if ($v == null) { return ""; }
    if (type($v) != "string") { $v = str($v); }
    $v = trim($v);
    if (len($v) > $max) { $v = substr($v, 0, $max); }
    return $v;
}

def intField($body, $name, $dflt) {
    if ($body == null) { return $dflt; }
    $v = $body[$name];
    if ($v == null) { return $dflt; }
    if (type($v) == "number") { return floor($v); }
    $n = num(str($v), null);
    if ($n == null) { return $dflt; }
    return floor($n);
}

def boolField($body, $name, $dflt) {
    if ($body == null) { return $dflt; }
    $v = $body[$name];
    if ($v == null) { return $dflt; }
    if (type($v) == "bool") { return $v; }
    return str($v) == "true";
}

// Accepts a JSON list or a comma-separated string, because a form and a fetch
// disagree about which they send and both are reasonable.
def listField($body, $name) {
    $out = [];
    if ($body == null) { return $out; }
    $v = $body[$name];
    if ($v == null) { return $out; }
    if (type($v) == "list") {
        each ($x in $v) {
            $t = trim(str($x));
            if ($t != "") { append($out, $t); }
        }
        return $out;
    }
    each ($part in split(str($v), ",")) {
        $t = trim($part);
        if ($t != "") { append($out, $t); }
    }
    return $out;
}

def pageLimit($req, $dflt, $cap) {
    $n = num($req.query["limit"], null);
    if ($n == null) { return $dflt; }
    $n = floor($n);
    if ($n < 1) { return $dflt; }
    if ($n > $cap) { return $cap; }
    return $n;
}
