// ============================================================================
// one-taylor/backend/api/media_routes.b — storing and serving images
//
// sua has no multipart parser, so an image arrives base64-encoded inside the
// JSON body. The client resizes to at most 4 MB first and
// sua.server.limits({"max_body_bytes": ...}) is raised to match the ~1.34x
// base64 inflation.
//
// uploads/ is NOT a sua.server.static root, and cannot be: a static root is
// ungated by construction, and the entire point of this application is that
// media is gated. GET /api/media/:id re-runs session -> band -> label opt-in
// -> report status before a single byte goes out.
//
// The MIME type is taken from the FILE, not from the client. A caller that
// says "image/jpeg" and sends HTML is trying to get a stored XSS served from
// our origin; the magic bytes decide, and anything unrecognised is refused.
//
// Exposes: store($postId, $items, $position) -> {"saved", "errors"}
//          register($sua)
// ============================================================================

include "crypto" as crypto;
include "hash" as hashlib;
include "orm" as orm;
include "../config.b" as cfg;
include "../models.b" as models;
include "../safety/age.b" as age;
include "../safety/labels.b" as labels;
include "../auth/sessions.b" as sessions;
include "./helpers.b" as h;

const $MAX_IMAGE_BYTES = 4194304;      // 4 MB after decoding
const $MAX_PER_POST = 4;

// Magic bytes, checked against the decoded data. The list is short on purpose:
// these four are what a browser will both produce and render.
//
// Only the first twelve bytes are turned into a list. bytes() on a whole image
// would build one ~190-byte Value per byte -- about 760 MB for a 4 MB photo --
// which is the same trap the native b64decode exists to avoid.
def _sniff($raw) {
    if (len($raw) < 12) { return null; }
    $b = bytes(substr($raw, 0, 12));
    if ($b[0] == 255 && $b[1] == 216 && $b[2] == 255) { return "image/jpeg"; }
    if ($b[0] == 137 && $b[1] == 80 && $b[2] == 78 && $b[3] == 71) { return "image/png"; }
    if ($b[0] == 71 && $b[1] == 73 && $b[2] == 70 && $b[3] == 56) { return "image/gif"; }
    // RIFF....WEBP
    if ($b[0] == 82 && $b[1] == 73 && $b[2] == 70 && $b[3] == 70
        && $b[8] == 87 && $b[9] == 69 && $b[10] == 66 && $b[11] == 80) {
        return "image/webp";
    }
    return null;
}

def _extFor($mime) {
    if ($mime == "image/jpeg") { return ".jpg"; }
    if ($mime == "image/png")  { return ".png"; }
    if ($mime == "image/gif")  { return ".gif"; }
    return ".webp";
}

// A data: URL and a bare base64 payload are both accepted, because both are
// what a browser hands you depending on how the file was read.
def _stripDataUrl($s) {
    if (substr($s, 0, 5) != "data:") { return $s; }
    $comma = indexOf($s, ",");
    if ($comma < 0) { return $s; }
    return substr($s, $comma + 1, len($s));
}

def store($postId, $items, $startPosition) {
    $saved = [];
    $errors = [];
    $pos = $startPosition;
    $i = 0;
    while ($i < len($items)) {
        if (len($saved) >= $MAX_PER_POST) {
            append($errors, "at most " + str($MAX_PER_POST) + " images per post");
            break;
        }
        $item = $items[$i];
        $i = $i + 1;
        $raw = null;
        if (type($item) == "string") { $raw = $item; }
        else if (type($item) == "dict") { $raw = $item["data"]; }
        if ($raw == null || type($raw) != "string" || $raw == "") {
            append($errors, "an image was empty");
            continue;
        }
        // b64decode is native and works string-to-string: the result is a
        // Bantu string of raw bytes, which writefile(..., "wb") stores
        // verbatim. The pure-Bantu codec in crypto.b returns a LIST, which is
        // right for a 32-byte key and ruinous for an image -- 1,299 ms and
        // four million Values for 100 KB, against 1 ms here.
        $decoded = b64decode(_stripDataUrl($raw));
        if ($decoded == null) {
            append($errors, "an image was not valid base64");
            continue;
        }
        if (len($decoded) == 0) {
            append($errors, "an image decoded to nothing");
            continue;
        }
        if (len($decoded) > $MAX_IMAGE_BYTES) {
            append($errors, "an image is larger than 4 MB");
            continue;
        }
        $mime = _sniff($decoded);
        if ($mime == null) {
            append($errors, "that file is not a JPEG, PNG, GIF or WebP");
            continue;
        }

        $digest = hashlib.sha256($decoded);
        $dir = cfg.CFG["uploadDir"] + "/" + substr($digest, 0, 2);
        $path = $dir + "/" + $digest + _extFor($mime);

        // The sharded directory has to be created. writefile() does not create
        // parents, and until mkdirs() existed in the language this line could
        // not be written at all -- the upload failed with "cannot write" for a
        // path whose parent was missing, and the only visible symptom was a
        // post that arrived without its picture.
        if (!mkdirs($dir)) {
            append($errors, "could not create the upload directory " + $dir);
            continue;
        }
        try {
            // Content-addressed, so the same image uploaded twice is stored
            // once and a retry after a failed request is idempotent.
            writefile($path, $decoded, "wb");
        } catch ($e) {
            // The underlying reason, not a shrug. "could not store an image"
            // is what made the missing-directory bug hard to find.
            $why = str($e);
            if (type($e) == "dict" && $e.message != null) { $why = str($e.message); }
            append($errors, "could not store an image: " + $why);
            continue;
        }
        if (!isfile($path)) {
            append($errors, "the image did not land on disk at " + $path);
            continue;
        }

        $ins = orm.insert(models.M["post_media"], {
            "post_id": $postId,
            "path": $path,
            "mime": $mime,
            "bytes": len($decoded),
            "sha256": $digest,
            "position": $pos
        });
        append($saved, { "id": $ins.id, "mime": $mime, "bytes": len($decoded) });
        $pos = $pos + 1;
    }
    return { "saved": $saved, "errors": $errors };
}

// ── serving ─────────────────────────────────────────────────────────────────
def getMediaHandler($req, $res) {
    $viewer = sessions.requireSession($req, $res);
    if ($viewer == null) { return null; }

    $id = num($req.params["id"], null);
    if ($id == null) { return h.fail($res, 404, "no such image"); }

    $m = orm.first(orm.filter(orm.query(models.M["post_media"]), [["id", floor($id)]]));
    if ($m == null) { return h.fail($res, 404, "no such image"); }

    $post = orm.first(orm.filter(orm.query(models.M["posts"]), [["id", $m["post_id"]]]));
    if ($post == null) { return h.fail($res, 404, "no such image"); }
    if ($post["status"] != "live") { return h.fail($res, 404, "no such image"); }

    // The whole gate, again, on the raw bytes. Every one of these checks is
    // also made when the post is listed; repeating them here is the point --
    // the feed is not the only door, and a guessed id is a door.
    $author = orm.first(orm.filter(orm.query(models.M["users"]), [["id", $post["author_id"]]]));
    if ($author == null) { return h.fail($res, 404, "no such image"); }
    if (!age.canSee($viewer, $author)) { return h.fail($res, 404, "no such image"); }

    if (orm.exists(orm.filter(orm.query(models.M["blocks"]),
            [["user_id", $viewer["id"]], ["blocked_id", $author["id"]]]))) {
        return h.fail($res, 404, "no such image");
    }
    if (orm.exists(orm.filter(orm.query(models.M["blocks"]),
            [["user_id", $author["id"]], ["blocked_id", $viewer["id"]]]))) {
        return h.fail($res, 404, "no such image");
    }
    if (orm.exists(orm.filter(orm.query(models.M["hidden_posts"]),
            [["user_id", $viewer["id"]], ["post_id", $post["id"]]]))) {
        return h.fail($res, 404, "no such image");
    }

    $cats = [];
    each ($l in orm.all(orm.filter(orm.query(models.M["post_labels"]),
                                   [["post_id", $post["id"]]]))) {
        append($cats, $l["category"]);
    }
    $decision = labels.decide($cats, $viewer);
    if ($decision["mode"] == "deny") { return h.fail($res, 404, "no such image"); }
    if ($decision["mode"] == "blur") {
        // Not a silent substitution: the client is told why, and shows its own
        // blurred placeholder. Serving the real bytes with a "please blur this"
        // header would put the decision in the client, where it is a devtools
        // panel away from being reversed.
        $res.status(403).json({ "ok": false, "error": "content warning",
                                "blurredFor": $decision["categories"] });
        return null;
    }

    $data = null;
    try {
        $data = readfile($m["path"], "rb");
    } catch ($e) {
        return h.fail($res, 404, "that image is no longer stored");
    }
    // frombytes round-trips arbitrary bytes, NULs included, and the response
    // is length-explicit, so a JPEG survives this intact.
    $res.set("Cache-Control", "private, max-age=300");
    $res.set("X-Content-Type-Options", "nosniff");
    $res.type($m["mime"]).send($data);
    return null;
}

def register($sua) {
    $sua.server.get("/api/media/:id", getMediaHandler);
}
