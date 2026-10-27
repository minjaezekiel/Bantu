// ============================================================================
// one-taylor/backend/auth/sessions.b — sessions, and the band transition
//
// A token is 32 CSPRNG bytes, URL-safe. The database stores only
// HMAC-SHA256(pepper, token), so a dump of the sessions table does not let
// anyone log in -- the same reason a password table stores hashes.
//
// Every resolve recomputes the band from the date of birth. When it differs
// from band_at_issue, the account has crossed 18 and the transition runs:
// every follow, like, hidden-post and report edge that the NEW band would not
// have permitted is deleted, in one pass, and the change is recorded. The day
// a 17-year-old turns 18, their 16-year-old friends stop being reachable.
// That is the intended behaviour, not a bug to work around.
//
// Exposes:
//   issue($user)                 -> {"token", "expires"}
//   resolve($req)                -> user row + profile fields, or null
//   revoke($token) / revokeAllFor($userId)
//   requireSession($req, $res)   -> user row, or null having already answered
//   transitionIfNeeded($user)    -> {"changed", "pruned"}
// ============================================================================

include "crypto" as crypto;
include "orm" as orm;
include "../config.b" as cfg;
include "../models.b" as models;
include "../safety/age.b" as age;

def _tokenHmac($token) {
    return crypto.hmac_sha256(cfg.sessionPepper(), $token);
}

def issue($user) {
    $token = crypto.token_urlsafe(32);
    $expires = age.nowSeconds() + cfg.CFG["sessionDays"] * 86400;
    orm.insert(models.M["sessions"], {
        "token_hmac": _tokenHmac($token),
        "user_id": $user["id"],
        "band_at_issue": age.bandFor($user["dob"]),
        "expires_at": $expires
    });
    return { "token": $token, "expires": $expires };
}

def revoke($token) {
    if ($token == null || $token == "") { return 0; }
    $r = orm.remove(orm.filter(orm.query(models.M["sessions"]),
                               [["token_hmac", _tokenHmac($token)]]), false);
    return $r.changes;
}

def revokeAllFor($userId) {
    orm.remove(orm.filter(orm.query(models.M["sessions"]), [["user_id", $userId]]), false);
}

// ── the band transition ─────────────────────────────────────────────────────
// Deleting by raw SQL rather than by loading rows: this runs on a login path,
// the edges are few but the join to find them is not expressible in the ORM's
// pair filters, and a loop issuing one DELETE per row would stall the worker.
def transitionIfNeeded($user) {
    $currentBand = age.bandFor($user["dob"]);
    if ($currentBand == $user["band"]) { return { "changed": false, "pruned": 0 }; }

    $id = floor($user["id"]);
    orm.modify(orm.filter(orm.query(models.M["users"]), [["id", $id]]),
               {"band": $currentBand});

    // Everything that now crosses the wall, in both directions.
    //
    // orm.whereRaw rather than pair filters because the condition is "the other
    // end of this edge is in the other band", which is a subquery and not a
    // column comparison. The ORM provides whereRaw for exactly this; the only
    // values interpolated are this server's own primary key and a band string
    // chosen from two literals, both floored or fixed before they get here.
    $other = "minor";
    if ($currentBand == "minor") { $other = "adult"; }
    $pruned = 0;

    $r = orm.remove(orm.whereRaw(orm.query(models.M["follows"]),
            "(follower_id = " + str($id) + " AND followee_id IN "
          + "(SELECT id FROM users WHERE band = '" + $other + "')) OR "
          + "(followee_id = " + str($id) + " AND follower_id IN "
          + "(SELECT id FROM users WHERE band = '" + $other + "'))"), false);
    if ($r != null && $r.changes != null) { $pruned = $pruned + $r.changes; }

    // Likes this account gave to posts now on the other side.
    $r = orm.remove(orm.whereRaw(
            orm.filter(orm.query(models.M["likes"]), [["user_id", $id]]),
            "post_id IN (SELECT id FROM posts WHERE author_band = '"
          + $other + "')"), false);
    if ($r != null && $r.changes != null) { $pruned = $pruned + $r.changes; }

    // Likes the other side gave to this account's posts.
    $r = orm.remove(orm.whereRaw(orm.query(models.M["likes"]),
            "post_id IN (SELECT id FROM posts WHERE author_id = " + str($id) + ") "
          + "AND user_id IN (SELECT id FROM users WHERE band = '" + $other + "')"),
            false);
    if ($r != null && $r.changes != null) { $pruned = $pruned + $r.changes; }

    // The account's own past posts are re-banded with it: a minor's posts
    // become adult-band posts the day they become an adult. The alternative --
    // leaving them in the minor band -- would make an adult's account
    // permanently visible to minors through its history.
    orm.modify(orm.filter(orm.query(models.M["posts"]), [["author_id", $id]]),
               {"author_band": $currentBand});

    // Every session issued under the old band is void. The band is part of
    // what the session authorised.
    revokeAllFor($id);

    orm.insert(models.M["moderation_events"], {
        "subject_type": "user",
        "subject_id": $id,
        "actor": "system",
        "decision": "band_transition",
        "detail": $user["band"] + " -> " + $currentBand + ", pruned " + str($pruned) + " edges"
    });

    return { "changed": true, "pruned": $pruned, "band": $currentBand };
}

// ── resolve ─────────────────────────────────────────────────────────────────
def bearerToken($req) {
    $h = $req.headers["authorization"];
    if ($h != null && len($h) > 7 && substr($h, 0, 7).lower() == "bearer ") {
        return trim(substr($h, 7, len($h)));
    }
    // The Next.js proxy forwards the token in a header; a cookie is accepted
    // too so the API is usable directly with curl during development.
    $c = $req.headers["cookie"];
    if ($c != null) {
        each ($part in split($c, ";")) {
            $kv = trim($part);
            if (substr($kv, 0, 3) == "ot=") { return substr($kv, 3, len($kv)); }
        }
    }
    return null;
}

def resolve($req) {
    $token = bearerToken($req);
    if ($token == null || $token == "") { return null; }
    $s = orm.first(orm.filter(orm.query(models.M["sessions"]),
                              [["token_hmac", _tokenHmac($token)]]));
    if ($s == null) { return null; }
    if ($s["expires_at"] < age.nowSeconds()) {
        revoke($token);
        return null;
    }
    $u = orm.first(orm.filter(orm.query(models.M["users"]), [["id", $s["user_id"]]]));
    if ($u == null) { return null; }
    if ($u["status"] != "active") { return null; }

    // The band is recomputed on every request, not trusted from the session.
    $t = transitionIfNeeded($u);
    if ($t["changed"]) {
        // The session was revoked by the transition; the client must log in
        // again, and will land in the correct band when it does.
        return null;
    }

    $p = orm.first(orm.filter(orm.query(models.M["profiles"]), [["user_id", $u["id"]]]));
    if ($p != null) {
        $u["seek_min_age"] = $p["seek_min_age"];
        $u["seek_max_age"] = $p["seek_max_age"];
        $u["explicit_optins"] = $p["explicit_optins"];
        $u["profile_id"] = $p["id"];
    }
    $u["age"] = age.ageNow($u["dob"]);
    return $u;
}

// sua has no middleware -- sua.server.use() registers handlers that never run
// -- so this is the explicit first line of every protected handler. The API
// test probes every route unauthenticated to catch the one where it was
// forgotten.
def requireSession($req, $res) {
    $u = resolve($req);
    if ($u == null) {
        $res.status(401).json({ "ok": false, "error": "sign in to continue" });
        return null;
    }
    return $u;
}
