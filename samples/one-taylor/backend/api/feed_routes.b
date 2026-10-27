// ============================================================================
// one-taylor/backend/api/feed_routes.b — feed, discover, health
//
//   GET /api/feed?limit=&before=
//   GET /api/discover?limit=
//   GET /api/health
//
// /api/health is the one route with no session, and it deliberately reports
// whether the classifier loaded. A moderation system running degraded must say
// so out loud; a green health check that hides it is worse than a red one.
//
// Exposes: register($sua)
// ============================================================================

include "orm" as orm;
include "../config.b" as cfg;
include "../models.b" as models;
include "../safety/age.b" as age;
include "../safety/labels.b" as labels;
include "../safety/moderation.b" as moderation;
include "../auth/sessions.b" as sessions;
include "../recsys/posts.b" as feedposts;
include "../recsys/people.b" as people;
include "./helpers.b" as h;

def feedHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $limit = h.pageLimit($req, 20, 50);
    $before = num($req.query["before"], null);
    if ($before != null) { $before = floor($before); }

    $posts = feedposts.feed($u, $limit, $before);
    $nextBefore = null;
    if (len($posts) > 0) {
        // The lowest id in the page, so the next request continues from there.
        // Cursor-by-id, not offset: an offset page shifts under you every time
        // somebody posts.
        $lowest = $posts[0]["id"];
        each ($p in $posts) { if ($p["id"] < $lowest) { $lowest = $p["id"]; } }
        $nextBefore = $lowest;
    }
    $b = age.contentBounds($u);
    return h.okJson($res, {
        "posts": $posts,
        "nextBefore": $nextBefore,
        "band": $b["band"],
        "ageWindow": { "min": $b["min"], "max": $b["max"] }
    });
}

def discoverHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $limit = h.pageLimit($req, 20, 50);
    $matches = people.candidates($u, $limit);
    $b = age.matchBounds($u);
    return h.okJson($res, {
        "matches": $matches,
        "band": $b["band"],
        "range": { "min": $b["min"], "max": $b["max"] },
        "rangeLocked": age.bandFor($u["dob"]) == age.MINOR_BAND()
    });
}

def healthHandler($req, $res) {
    $res.json({
        "ok": true,
        "service": "one-taylor",
        "version": "1.0.0",
        "moderation": {
            "classifier": moderation.ready(),
            "reason": moderation.reason(),
            // Visible on purpose: a moderation system running with half its
            // labels advisory is running degraded, and a health check that
            // hides that is worse than a red one.
            "blocking": moderation.blockingLabels(),
            "advisory": moderation.advisoryLabels()
        },
        "reportThreshold": cfg.CFG["reportThreshold"],
        "explicitCategories": labels.CATEGORIES,
        "users": orm.count(orm.query(models.M["users"])),
        "posts": orm.count(orm.filter(orm.query(models.M["posts"]), [["status", "live"]]))
    });
    return null;
}

def register($sua) {
    $sua.server.get("/api/feed", feedHandler);
    $sua.server.get("/api/discover", discoverHandler);
    $sua.server.get("/api/health", healthHandler);
}
