// ============================================================================
// one-taylor/backend/api/social_routes.b — following and blocking
//
//   POST /api/users/:username/follow
//   POST /api/users/:username/unfollow
//   POST /api/users/:username/block
//   POST /api/users/:username/unblock
//   GET  /api/users/:username/followers
//   GET  /api/users/:username/following
//
// A follow is contact, so it needs canInteract (content visibility in BOTH
// directions), not canSee. Blocking needs neither: you may always block, and
// unfollowing is not gated either -- a gate on leaving is a trap.
//
// Exposes: register($sua)
// ============================================================================

include "orm" as orm;
include "../models.b" as models;
include "../safety/age.b" as age;
include "../auth/privacy.b" as privacy;
include "../auth/sessions.b" as sessions;
include "./helpers.b" as h;

def _target($req) {
    $name = $req.params["username"];
    if ($name == null || $name == "") { return null; }
    return orm.first(orm.filter(orm.query(models.M["users"]),
                                [["username_lower", $name.lower()]]));
}

def followHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $t = _target($req);
    if ($t == null || $t["status"] != "active") {
        return h.fail($res, 404, "no such account");
    }
    if ($t["id"] == $u["id"]) {
        return h.fail($res, 400, "you cannot follow yourself");
    }
    // Pull the target's range so canInteract has both sides.
    $tp = orm.first(orm.filter(orm.query(models.M["profiles"]), [["user_id", $t["id"]]]));
    if ($tp != null) {
        $t["seek_min_age"] = $tp["seek_min_age"];
        $t["seek_max_age"] = $tp["seek_max_age"];
    }
    if (!age.canInteract($u, $t)) { return h.fail($res, 404, "no such account"); }

    if (orm.exists(orm.filter(orm.query(models.M["blocks"]),
            [["user_id", $t["id"]], ["blocked_id", $u["id"]]]))) {
        return h.fail($res, 404, "no such account");
    }
    if (orm.exists(orm.filter(orm.query(models.M["blocks"]),
            [["user_id", $u["id"]], ["blocked_id", $t["id"]]]))) {
        return h.fail($res, 400, "unblock them first");
    }

    if (!orm.exists(orm.filter(orm.query(models.M["follows"]),
            [["follower_id", $u["id"]], ["followee_id", $t["id"]]]))) {
        orm.insert(models.M["follows"],
                   {"follower_id": $u["id"], "followee_id": $t["id"]});
    }
    return h.okJson($res, {
        "following": true,
        "followers": orm.count(orm.filter(orm.query(models.M["follows"]),
                                          [["followee_id", $t["id"]]]))
    });
}

def unfollowHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $t = _target($req);
    if ($t == null) { return h.fail($res, 404, "no such account"); }
    orm.remove(orm.filter(orm.query(models.M["follows"]),
               [["follower_id", $u["id"]], ["followee_id", $t["id"]]]), false);
    return h.okJson($res, {
        "following": false,
        "followers": orm.count(orm.filter(orm.query(models.M["follows"]),
                                          [["followee_id", $t["id"]]]))
    });
}

def blockHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $t = _target($req);
    if ($t == null) { return h.fail($res, 404, "no such account"); }
    if ($t["id"] == $u["id"]) { return h.fail($res, 400, "you cannot block yourself"); }

    if (!orm.exists(orm.filter(orm.query(models.M["blocks"]),
            [["user_id", $u["id"]], ["blocked_id", $t["id"]]]))) {
        orm.insert(models.M["blocks"], {"user_id": $u["id"], "blocked_id": $t["id"]});
    }
    // A block severs the graph in both directions. Leaving the other half of
    // the follow in place means the person you blocked keeps seeing you in
    // their feed, which is the opposite of what the button appeared to do.
    orm.remove(orm.filter(orm.query(models.M["follows"]),
               [["follower_id", $u["id"]], ["followee_id", $t["id"]]]), false);
    orm.remove(orm.filter(orm.query(models.M["follows"]),
               [["follower_id", $t["id"]], ["followee_id", $u["id"]]]), false);
    return h.okJson($res, { "blocked": true });
}

def unblockHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $t = _target($req);
    if ($t == null) { return h.fail($res, 404, "no such account"); }
    orm.remove(orm.filter(orm.query(models.M["blocks"]),
               [["user_id", $u["id"]], ["blocked_id", $t["id"]]]), false);
    return h.okJson($res, { "blocked": false });
}

// The follower and following lists are read paths, so they are gated like any
// other: an adult's follower list must not leak a minor, and vice versa, even
// though the edge could only exist from before a band transition.
def _listPeople($req, $res, $column, $pick) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $t = _target($req);
    if ($t == null || $t["status"] != "active") {
        return h.fail($res, 404, "no such account");
    }
    if ($t["id"] != $u["id"] && !age.canSee($u, $t)) {
        return h.fail($res, 404, "no such account");
    }
    $rows = orm.all(orm.limit(orm.filter(orm.query(models.M["follows"]),
                                         [[$column, $t["id"]]]), 500));
    $out = [];
    each ($f in $rows) {
        $otherId = $f[$pick];
        $other = orm.first(orm.filter(orm.query(models.M["users"]), [["id", $otherId]]));
        if ($other == null || $other["status"] != "active") { continue; }
        if ($other["id"] != $u["id"] && !age.canSee($u, $other)) { continue; }
        $p = orm.first(orm.filter(orm.query(models.M["profiles"]),
                                  [["user_id", $other["id"]]]));
        append($out, privacy.publicUserWithProfile($other, $p, $u));
    }
    return h.okJson($res, { "people": $out, "count": len($out) });
}

def followersHandler($req, $res) {
    return _listPeople($req, $res, "followee_id", "follower_id");
}

def followingHandler($req, $res) {
    return _listPeople($req, $res, "follower_id", "followee_id");
}

def register($sua) {
    $sua.server.post("/api/users/:username/follow", followHandler);
    $sua.server.post("/api/users/:username/unfollow", unfollowHandler);
    $sua.server.post("/api/users/:username/block", blockHandler);
    $sua.server.post("/api/users/:username/unblock", unblockHandler);
    $sua.server.get("/api/users/:username/followers", followersHandler);
    $sua.server.get("/api/users/:username/following", followingHandler);
}
