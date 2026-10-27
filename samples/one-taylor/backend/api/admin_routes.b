// ============================================================================
// one-taylor/backend/api/admin_routes.b — the review queue
//
//   GET  /api/admin/moderation/queue
//   POST /api/admin/moderation/:postId/release
//   POST /api/admin/moderation/:postId/remove
//   GET  /api/admin/moderation/audit?subjectId=
//
// The queue is the part of the design that admits the classifier is not an
// oracle. A blocked post, a borderline one and an uncounted report all land
// here for a person to look at.
//
// Authorisation is by a shared operator token in ONE_TAYLOR_ADMIN_TOKEN,
// compared in constant time. Not a role column on users: a moderator account
// that can also post and match is an account whose compromise is much more
// expensive, and this app has no org chart to model.
//
// Exposes: register($sua)
// ============================================================================

include "crypto" as crypto;
include "hash" as hashlib;
include "orm" as orm;
include "../models.b" as models;
include "../safety/moderation.b" as moderation;
include "../auth/privacy.b" as privacy;
include "./helpers.b" as h;

def _adminOk($req) {
    $want = env("ONE_TAYLOR_ADMIN_TOKEN");
    if ($want == null || len($want) < 16) { return false; }
    $got = $req.headers["x-admin-token"];
    if ($got == null) { return false; }
    // Constant-time: comparing with == leaks the length of the shared prefix
    // through timing, and a moderation endpoint is worth guessing at.
    return ct_equal(bytes(hashlib.sha256_bytes($want)),
                    bytes(hashlib.sha256_bytes($got)));
}

def queueHandler($req, $res) {
    if (!_adminOk($req)) { return h.fail($res, 404, "not found"); }

    $blocked = orm.all(orm.limit(orm.orderBy(
        orm.filter(orm.query(models.M["posts"]), [["status", "blocked"]]), ["-id"]), 100));
    $review = orm.all(orm.limit(orm.orderBy(
        orm.filter(orm.query(models.M["posts"]), [["status", "review"]]), ["-id"]), 100));

    // Posts carrying reports that did not count toward the takedown threshold:
    // a weak report from a throwaway account, or a reason that needs a human.
    // A brigade is itself worth a human's attention, which is why those reports
    // are recorded rather than dropped.
    //
    // Grouped in Bantu rather than with SQL's GROUP BY, because the ORM has no
    // aggregate-with-grouping and reaching past it for a report dashboard is
    // not worth a second query dialect in this file. The row count is bounded
    // by the limit below.
    $flagged = orm.all(orm.limit(orm.orderBy(
        orm.whereRaw(orm.query(models.M["reports"]), "counted = 0 OR weak = 1"),
        ["-id"]), 400));
    $byPost = {};
    each ($rep in $flagged) {
        $k = str($rep["post_id"]);
        if ($byPost[$k] == null) { $byPost[$k] = 0; }
        $byPost[$k] = $byPost[$k] + 1;
    }
    $uncounted = [];
    each ($pid in keys($byPost)) {
        if (len($uncounted) >= 50) { break; }
        $post = orm.first(orm.filter(orm.query(models.M["posts"]),
                                     [["id", floor(num($pid))]]));
        if ($post == null) { continue; }
        append($uncounted, { "id": $post["id"], "body": $post["body"],
                             "author_band": $post["author_band"],
                             "status": $post["status"], "n": $byPost[$pid] });
    }
    $uncounted = sort($uncounted, {"key": def($r) { return $r["n"]; }, "desc": true});

    def _shape($rows) {
        $out = [];
        each ($p in $rows) {
            $author = orm.first(orm.filter(orm.query(models.M["users"]),
                                           [["id", $p["author_id"]]]));
            $ev = orm.all(orm.limit(orm.orderBy(
                orm.filter(orm.query(models.M["moderation_events"]),
                           [["subject_type", "post"], ["subject_id", $p["id"]]]),
                ["-id"]), 3));
            append($out, {
                "id": $p["id"],
                "body": $p["body"],
                "authorBand": $p["author_band"],
                "authorAge": $p["author_age_at_post"],
                "status": $p["status"],
                "createdAt": $p["created_at"],
                "author": privacy.publicUser($author, null),
                "events": $ev
            });
        }
        return $out;
    }

    return h.okJson($res, {
        "blocked": _shape($blocked),
        "review": _shape($review),
        "brigaded": $uncounted,
        "counts": {
            "blocked": len($blocked),
            "review": len($review),
            "brigaded": len($uncounted)
        }
    });
}

def _decide($req, $res, $newStatus, $decision) {
    if (!_adminOk($req)) { return h.fail($res, 404, "not found"); }
    $id = num($req.params["postId"], null);
    if ($id == null) { return h.fail($res, 404, "no such post"); }
    $post = orm.first(orm.filter(orm.query(models.M["posts"]), [["id", floor($id)]]));
    if ($post == null) { return h.fail($res, 404, "no such post"); }
    $note = h.strField(h.bodyOf($req), "note", 500);

    orm.modify(orm.filter(orm.query(models.M["posts"]), [["id", $post["id"]]]),
               {"status": $newStatus});
    moderation.record("post", $post["id"], $decision, "moderator", $note, {});
    return h.okJson($res, { "postId": $post["id"], "status": $newStatus });
}

def releaseHandler($req, $res) {
    return _decide($req, $res, "live", "released_by_moderator");
}

def removeHandler($req, $res) {
    return _decide($req, $res, "removed", "removed_by_moderator");
}

def auditHandler($req, $res) {
    if (!_adminOk($req)) { return h.fail($res, 404, "not found"); }
    $pairs = [];
    $sid = num($req.query["subjectId"], null);
    if ($sid != null) { append($pairs, ["subject_id", floor($sid)]); }
    $stype = $req.query["subjectType"];
    if ($stype != null && $stype != "") { append($pairs, ["subject_type", $stype]); }
    $qs = orm.query(models.M["moderation_events"]);
    if (len($pairs) > 0) { $qs = orm.filter($qs, $pairs); }
    return h.okJson($res, {
        "events": orm.all(orm.limit(orm.orderBy($qs, ["-id"]), 200))
    });
}

def register($sua) {
    $sua.server.get("/api/admin/moderation/queue", queueHandler);
    $sua.server.post("/api/admin/moderation/:postId/release", releaseHandler);
    $sua.server.post("/api/admin/moderation/:postId/remove", removeHandler);
    $sua.server.get("/api/admin/moderation/audit", auditHandler);
}
