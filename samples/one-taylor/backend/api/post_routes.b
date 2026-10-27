// ============================================================================
// one-taylor/backend/api/post_routes.b — posts
//
//   POST   /api/posts            {body, tags?, media?, labels?}
//   GET    /api/posts/:id
//   DELETE /api/posts/:id
//   POST   /api/posts/:id/like
//   POST   /api/posts/:id/unlike
//
// Moderation runs HERE, at write time, once. Not on read: sua.sqlite and the
// handler share one thread, so a 15 ms model call on the feed path would be
// 15 ms added to every other client waiting on that worker.
//
// A blocked post is still written -- with status "blocked" -- rather than
// discarded. An author who is told "no" and has their words deleted cannot
// appeal, and a moderator cannot review a decision whose subject no longer
// exists.
//
// Exposes: register($sua)
// ============================================================================

include "orm" as orm;
include "../config.b" as cfg;
include "../models.b" as models;
include "../safety/age.b" as age;
include "../safety/labels.b" as labels;
include "../safety/moderation.b" as moderation;
include "../auth/privacy.b" as privacy;
include "../auth/sessions.b" as sessions;
include "../recsys/posts.b" as feedposts;
include "./helpers.b" as h;
include "./media_routes.b" as media;

const $MAX_BODY = 5000;
const $MAX_TAGS = 10;

def _normTag($t) {
    $s = trim($t).lower();
    $out = [];
    $i = 0;
    while ($i < len($s) && len($out) < 30) {
        $c = substr($s, $i, 1);
        if (($c >= "a" && $c <= "z") || ($c >= "0" && $c <= "9")) { append($out, $c); }
        else if ($c == " " || $c == "-" || $c == "_") {
            if (len($out) > 0 && $out[len($out) - 1] != "-") { append($out, "-"); }
        }
        $i = $i + 1;
    }
    $j = join($out, "");
    while (len($j) > 0 && substr($j, len($j) - 1, 1) == "-") {
        $j = substr($j, 0, len($j) - 1);
    }
    return $j;
}

def createHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $body = h.bodyOf($req);
    if ($body == null) { return h.fail($res, 400, "send a JSON body"); }

    $text = privacy.sanitiseText($body["body"], $MAX_BODY);
    $mediaItems = $body["media"];
    if ($mediaItems == null) { $mediaItems = []; }
    if (type($mediaItems) != "list") { $mediaItems = []; }
    if ($text == "" && len($mediaItems) == 0) {
        return h.fail($res, 400, "write something or add a picture");
    }

    $band = age.bandFor($u["dob"]);
    $years = age.ageNow($u["dob"]);

    // ── moderation, once, before anything is visible ───────────────────────
    $verdict = moderation.assess($text, $band);
    $v = $verdict["verdict"];
    // "hold" and "block" both keep the post out of every other feed; the
    // difference is what the author is told and how reversible it is. "watch"
    // publishes and queues.
    $status = "live";
    if ($v == "block") { $status = "blocked"; }
    else if ($v == "hold") { $status = "review"; }

    $ins = orm.insert(models.M["posts"], {
        "author_id": $u["id"],
        "body": $text,
        "author_band": $band,
        "author_age_at_post": $years,
        "status": $status
    });
    if (!$ins.ok) { return h.fail($res, 500, "could not save that post"); }
    $postId = $ins.id;

    moderation.record("post", $postId, $verdict["verdict"], "system",
                      join($verdict["reasons"], ","), $verdict["scores"]);

    // Content labels: the author's own declarations, plus the classifier's.
    $declared = h.listField($body, "labels");
    $labelSeen = {};
    each ($c in $declared) {
        if (labels.isCategory($c) && $labelSeen[$c] == null) {
            $labelSeen[$c] = true;
            orm.insert(models.M["post_labels"], {
                "post_id": $postId, "category": $c,
                "source": "author", "confidence": 1.0
            });
        }
    }
    each ($l in $verdict["labels"]) {
        if ($labelSeen[$l["category"]] == null) {
            $labelSeen[$l["category"]] = true;
            orm.insert(models.M["post_labels"], {
                "post_id": $postId, "category": $l["category"],
                "source": $l["source"], "confidence": $l["confidence"]
            });
        }
    }

    // A minor cannot label a post explicit, and a minor's post cannot carry a
    // label: there is no audience for it on this side of the wall.
    if ($band == age.MINOR_BAND() && len(keys($labelSeen)) > 0) {
        orm.remove(orm.filter(orm.query(models.M["post_labels"]),
                   [["post_id", $postId]]), false);
        $labelSeen = {};
    }

    $tags = h.listField($body, "tags");
    $tagCount = 0;
    $tagSeen = {};
    each ($t in $tags) {
        if ($tagCount >= $MAX_TAGS) { break; }
        $n = _normTag($t);
        if ($n == "" || $tagSeen[$n] != null) { continue; }
        $tagSeen[$n] = true;
        orm.insert(models.M["post_tags"], {"post_id": $postId, "tag": $n});
        $tagCount = $tagCount + 1;
    }

    $mediaResult = { "saved": [], "errors": [] };
    if (len($mediaItems) > 0 && $status != "blocked") {
        $mediaResult = media.store($postId, $mediaItems, 0);
    }

    if ($status == "blocked") {
        $res.status(422);
        return h.okJson($res, {
            "postId": $postId,
            "status": "blocked",
            "message": "This post was not published.",
            "reasons": $verdict["reasons"],
            "appealable": true
        });
    }

    $post = orm.first(orm.filter(orm.query(models.M["posts"]), [["id", $postId]]));
    $res.status(201);

    // The verdict and a sentence explaining it go back to the author, always.
    //
    // Without them the response was a plain 201 and the client had nothing to
    // show, so a HELD post looked exactly like a published one that had
    // vanished — the backend was moderating correctly and the person who wrote
    // it was told nothing. "It does not moderate" is what that looks like from
    // the outside, and it is the worst kind of bug: the safety machinery works
    // and nobody can tell.
    $message = null;
    if ($v == "hold") {
        $message = "Held for review — only you can see this until a moderator "
                 + "has looked at it.";
    } else if ($v == "watch") {
        $message = "Posted. A moderator will read it as well.";
    } else if ($v == "label") {
        $message = "Posted with a content warning.";
    }
    return h.okJson($res, {
        "post": feedposts.hydrate($post, $u),
        "status": $status,
        "verdict": $v,
        "message": $message,
        "reasons": $verdict["reasons"],
        "mediaErrors": $mediaResult["errors"],
        "queuedForReview": $v == "hold" || $v == "watch"
    });
}

def getHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $id = num($req.params["id"], null);
    if ($id == null) { return h.fail($res, 404, "no such post"); }

    $post = orm.first(orm.filter(orm.query(models.M["posts"]), [["id", floor($id)]]));
    if ($post == null) { return h.fail($res, 404, "no such post"); }

    // An author may always read their own post, whatever its status -- that is
    // how they see a block and appeal it.
    $mine = $post["author_id"] == $u["id"];
    if (!$mine) {
        if ($post["status"] != "live") { return h.fail($res, 404, "no such post"); }
        $author = orm.first(orm.filter(orm.query(models.M["users"]),
                                       [["id", $post["author_id"]]]));
        if (!age.canSee($u, $author)) { return h.fail($res, 404, "no such post"); }
        if (orm.exists(orm.filter(orm.query(models.M["hidden_posts"]),
                [["user_id", $u["id"]], ["post_id", $post["id"]]]))) {
            return h.fail($res, 404, "no such post");
        }
        if (orm.exists(orm.filter(orm.query(models.M["blocks"]),
                [["user_id", $u["id"]], ["blocked_id", $author["id"]]]))) {
            return h.fail($res, 404, "no such post");
        }
        if (orm.exists(orm.filter(orm.query(models.M["blocks"]),
                [["user_id", $author["id"]], ["blocked_id", $u["id"]]]))) {
            return h.fail($res, 404, "no such post");
        }
    }

    $hydrated = feedposts.hydrate($post, $u);
    if ($hydrated == null) { return h.fail($res, 404, "no such post"); }
    $hydrated["mine"] = $mine;
    if ($mine) { $hydrated["status"] = $post["status"]; }
    return h.okJson($res, { "post": $hydrated });
}

def deleteHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $id = num($req.params["id"], null);
    if ($id == null) { return h.fail($res, 404, "no such post"); }
    $post = orm.first(orm.filter(orm.query(models.M["posts"]), [["id", floor($id)]]));
    if ($post == null) { return h.fail($res, 404, "no such post"); }
    if ($post["author_id"] != $u["id"]) {
        // 404, not 403: "that exists but is not yours" is information.
        return h.fail($res, 404, "no such post");
    }
    orm.modify(orm.filter(orm.query(models.M["posts"]), [["id", $post["id"]]]),
               {"status": "deleted"});
    moderation.record("post", $post["id"], "deleted_by_author", "user:" + str($u["id"]),
                      "", {});
    return h.okJson($res, { "deleted": $post["id"] });
}

def _likeable($u, $postId) {
    $post = orm.first(orm.filter(orm.query(models.M["posts"]), [["id", $postId]]));
    if ($post == null || $post["status"] != "live") { return null; }
    $author = orm.first(orm.filter(orm.query(models.M["users"]),
                                   [["id", $post["author_id"]]]));
    // A like is contact: it names you to the author. canInteract, not canSee.
    if (!age.canInteract($u, $author)) { return null; }
    return $post;
}

def likeHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $id = num($req.params["id"], null);
    if ($id == null) { return h.fail($res, 404, "no such post"); }
    $post = _likeable($u, floor($id));
    if ($post == null) { return h.fail($res, 404, "no such post"); }

    if (orm.exists(orm.filter(orm.query(models.M["likes"]),
            [["post_id", $post["id"]], ["user_id", $u["id"]]]))) {
        return h.okJson($res, { "liked": true, "likeCount": $post["like_count"] });
    }
    orm.insert(models.M["likes"], {"post_id": $post["id"], "user_id": $u["id"]});
    // Recounted rather than incremented: a counter that drifts is worse than
    // a count that costs one indexed query.
    $n = orm.count(orm.filter(orm.query(models.M["likes"]), [["post_id", $post["id"]]]));
    orm.modify(orm.filter(orm.query(models.M["posts"]), [["id", $post["id"]]]),
               {"like_count": $n});
    return h.okJson($res, { "liked": true, "likeCount": $n });
}

def unlikeHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $id = num($req.params["id"], null);
    if ($id == null) { return h.fail($res, 404, "no such post"); }
    orm.remove(orm.filter(orm.query(models.M["likes"]),
               [["post_id", floor($id)], ["user_id", $u["id"]]]), false);
    $n = orm.count(orm.filter(orm.query(models.M["likes"]), [["post_id", floor($id)]]));
    orm.modify(orm.filter(orm.query(models.M["posts"]), [["id", floor($id)]]),
               {"like_count": $n});
    return h.okJson($res, { "liked": false, "likeCount": $n });
}

def register($sua) {
    $sua.server.post("/api/posts", createHandler);
    $sua.server.get("/api/posts/:id", getHandler);
    $sua.server.delete("/api/posts/:id", deleteHandler);
    $sua.server.post("/api/posts/:id/like", likeHandler);
    $sua.server.post("/api/posts/:id/unlike", unlikeHandler);
}
