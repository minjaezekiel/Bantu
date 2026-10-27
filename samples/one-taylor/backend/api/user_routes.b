// ============================================================================
// one-taylor/backend/api/user_routes.b — profiles, interests, search
//
//   GET  /api/users/:username
//   PUT  /api/settings/profile    {displayName, bio, theme}
//   PUT  /api/settings/tags       {interests, hobbies, dreamPartner}
//   PUT  /api/settings/range      {minAge, maxAge, genders}
//   PUT  /api/settings/explicit   {categories}
//   GET  /api/search/users?q=&band=
//
// Every read composes age.userVisibilityPairs() and then re-checks with
// age.canSee(). The SQL narrows; canSee decides. A date-of-birth bound is
// whole-year and therefore approximate, and "approximately" is not a property
// the age wall may have.
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

const $MAX_TAGS_PER_KIND = 12;
const $TAG_KINDS = ["interest", "hobby", "dream_partner"];

def _normTag($t) {
    $s = trim($t).lower();
    if (len($s) > 30) { $s = substr($s, 0, 30); }
    $out = [];
    $i = 0;
    while ($i < len($s)) {
        $c = substr($s, $i, 1);
        if (($c >= "a" && $c <= "z") || ($c >= "0" && $c <= "9") || $c == " ") {
            append($out, $c);
        }
        $i = $i + 1;
    }
    return trim(join($out, ""));
}

def getUserHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $name = $req.params["username"];
    if ($name == null || $name == "") { return h.fail($res, 404, "no such account"); }

    $target = orm.first(orm.filter(orm.query(models.M["users"]),
                                   [["username_lower", $name.lower()]]));
    if ($target == null) { return h.fail($res, 404, "no such account"); }
    if ($target["status"] != "active" && $target["id"] != $u["id"]) {
        return h.fail($res, 404, "no such account");
    }
    // 404 and not 403, in both directions. A minor must not be able to confirm
    // that an adult's username exists, and vice versa: "that account exists
    // but you may not see it" is exactly the fact the wall is meant to hide.
    if ($target["id"] != $u["id"] && !age.canSee($u, $target)) {
        return h.fail($res, 404, "no such account");
    }
    if (orm.exists(orm.filter(orm.query(models.M["blocks"]),
            [["user_id", $target["id"]], ["blocked_id", $u["id"]]]))) {
        return h.fail($res, 404, "no such account");
    }

    $profile = orm.first(orm.filter(orm.query(models.M["profiles"]),
                                    [["user_id", $target["id"]]]));
    $tags = orm.all(orm.filter(orm.query(models.M["user_tags"]),
                               [["user_id", $target["id"]]]));
    $interests = [];
    $hobbies = [];
    $wants = [];
    each ($t in $tags) {
        if ($t["kind"] == "interest") { append($interests, $t["tag"]); }
        else if ($t["kind"] == "hobby") { append($hobbies, $t["tag"]); }
        else if ($t["kind"] == "dream_partner") { append($wants, $t["tag"]); }
    }

    return h.okJson($res, {
        "user": privacy.publicUserWithProfile($target, $profile, $u),
        "interests": sort($interests),
        "hobbies": sort($hobbies),
        "dreamPartner": sort($wants),
        "posts": feedposts.byAuthor($u, $target["id"], 30),
        "followers": orm.count(orm.filter(orm.query(models.M["follows"]),
                                          [["followee_id", $target["id"]]])),
        "following": orm.count(orm.filter(orm.query(models.M["follows"]),
                                          [["follower_id", $target["id"]]])),
        "youFollow": orm.exists(orm.filter(orm.query(models.M["follows"]),
                        [["follower_id", $u["id"]], ["followee_id", $target["id"]]])),
        // Both directions, because messaging and calls need a MUTUAL follow —
        // the frontend hides the Message button until then rather than showing
        // one that fails.
        "followsYou": orm.exists(orm.filter(orm.query(models.M["follows"]),
                        [["follower_id", $target["id"]], ["followee_id", $u["id"]]])),
        "canMatch": age.canMatch($u, $target),
        "isYou": $target["id"] == $u["id"]
    });
}

def updateProfileHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $body = h.bodyOf($req);
    if ($body == null) { return h.fail($res, 400, "send a JSON body"); }

    $bio = privacy.sanitiseBio($body["bio"]);
    $displayName = privacy.sanitiseText($body["displayName"], 40);
    $theme = h.strField($body, "theme", 10);
    if ($theme != "light" && $theme != "dark" && $theme != "system") {
        $theme = "system";
    }

    // A bio is published text, so it goes through the same moderation as a
    // post. A profile is where grooming actually starts -- "15, mature for my
    // age, dm me" is a bio, not a post -- so exempting it would leave the
    // front door open.
    $band = age.bandFor($u["dob"]);
    $verdict = moderation.assess($bio + " " + $displayName, $band);
    $v = $verdict["verdict"];
    // Only a lexicon hit refuses a bio outright. A "hold" or a "watch" saves
    // it and queues it: a bio is how somebody introduces themselves, and
    // refusing one on a two-thirds-confident guess tells a teenager that
    // describing their own hobbies is against the rules.
    if ($v == "block") {
        moderation.record("profile", $u["id"], "block", "system",
                          join($verdict["reasons"], ","), $verdict["scores"]);
        return h.fail($res, 422, "that bio cannot be saved: "
                      + join($verdict["reasons"], ", "));
    }
    if ($v == "hold" || $v == "watch") {
        moderation.record("profile", $u["id"], $v, "system",
                          join($verdict["reasons"], ","), $verdict["scores"]);
    }

    if ($displayName == "") { $displayName = $u["username"]; }
    orm.modify(orm.filter(orm.query(models.M["profiles"]), [["user_id", $u["id"]]]),
               {"bio": $bio, "display_name": $displayName, "theme": $theme,
                "updated_at": orm.raw("CURRENT_TIMESTAMP")});
    $p = orm.first(orm.filter(orm.query(models.M["profiles"]), [["user_id", $u["id"]]]));
    return h.okJson($res, { "user": privacy.publicUserWithProfile($u, $p, $u) });
}

def updateTagsHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $body = h.bodyOf($req);
    if ($body == null) { return h.fail($res, 400, "send a JSON body"); }

    $groups = {
        "interest": h.listField($body, "interests"),
        "hobby": h.listField($body, "hobbies"),
        "dream_partner": h.listField($body, "dreamPartner")
    };

    // Replace wholesale per kind, so the client sends what it wants to end up
    // with rather than a diff it has to compute correctly.
    $saved = {};
    each ($kind in $TAG_KINDS) {
        orm.remove(orm.filter(orm.query(models.M["user_tags"]),
                   [["user_id", $u["id"]], ["kind", $kind]]), false);
        $seen = {};
        $kept = [];
        each ($raw in $groups[$kind]) {
            if (len($kept) >= $MAX_TAGS_PER_KIND) { break; }
            $t = _normTag($raw);
            if ($t == "" || $seen[$t] != null) { continue; }
            $seen[$t] = true;
            append($kept, $t);
            orm.insert(models.M["user_tags"],
                       {"user_id": $u["id"], "kind": $kind, "tag": $t});
        }
        $saved[$kind] = sort($kept);
    }
    return h.okJson($res, {
        "interests": $saved["interest"],
        "hobbies": $saved["hobby"],
        "dreamPartner": $saved["dream_partner"]
    });
}

def updateRangeHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $body = h.bodyOf($req);
    if ($body == null) { return h.fail($res, 400, "send a JSON body"); }

    $years = age.ageNow($u["dob"]);
    $wantMin = h.intField($body, "minAge", $years - cfg.CFG["adultDefaultSpread"]);
    $wantMax = h.intField($body, "maxAge", $years + cfg.CFG["adultDefaultSpread"]);

    // clampRange is where the floor of 18 lives, and for a minor it ignores
    // the request entirely and returns the +/- 2 band. A minor's range is not
    // a preference, so there is nothing here to validate -- only to overrule.
    $range = age.clampRange($years, $wantMin, $wantMax);

    $genders = h.listField($body, "genders");
    $serialised = "any";
    if (len($genders) > 0) {
        $ok = [];
        each ($g in $genders) {
            $t = trim($g).lower();
            each ($valid in ["woman", "man", "non-binary", "other"]) {
                if ($t == $valid) { append($ok, $t); }
            }
        }
        if (len($ok) > 0) { $serialised = join($ok, ","); }
    }

    orm.modify(orm.filter(orm.query(models.M["profiles"]), [["user_id", $u["id"]]]),
               {"seek_min_age": $range["min"], "seek_max_age": $range["max"],
                "seek_genders": $serialised});
    return h.okJson($res, {
        "range": $range,
        "genders": $serialised,
        "locked": age.bandFor($u["dob"]) == age.MINOR_BAND(),
        "requested": { "minAge": $wantMin, "maxAge": $wantMax }
    });
}

def updateExplicitHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    if (age.bandFor($u["dob"]) == age.MINOR_BAND()) {
        // Not "forbidden" with a list of what they cannot have. There is no
        // such setting for this account, and the answer says so.
        return h.fail($res, 404, "no such setting");
    }
    $body = h.bodyOf($req);
    if ($body == null) { return h.fail($res, 400, "send a JSON body"); }
    $cats = h.listField($body, "categories");
    $serialised = labels.serialiseOptins($cats);
    orm.modify(orm.filter(orm.query(models.M["profiles"]), [["user_id", $u["id"]]]),
               {"explicit_optins": $serialised});
    return h.okJson($res, {
        "optins": labels.parseOptins($serialised),
        "available": labels.CATEGORIES
    });
}

def searchHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $q = $req.query["q"];
    if ($q == null) { $q = ""; }
    $q = trim($q).lower();
    $limit = h.pageLimit($req, 25, 50);

    $pairs = [];
    extend($pairs, age.userVisibilityPairs($u));
    if ($q != "") { append($pairs, ["username_lower__contains", $q]); }

    // An optional narrower age window WITHIN the band. The band itself is not
    // a parameter: a client cannot ask to search the other side of the wall,
    // because the band comes from the session and never from the query.
    $minAge = num($req.query["minAge"], null);
    $maxAge = num($req.query["maxAge"], null);

    $rows = orm.all(orm.limit(orm.orderBy(
        orm.filter(orm.query(models.M["users"]), $pairs), ["username_lower"]), 200));

    $blockedBy = {};
    each ($b in orm.all(orm.filter(orm.query(models.M["blocks"]),
                                   [["blocked_id", $u["id"]]]))) {
        $blockedBy[str($b["user_id"])] = true;
    }

    $out = [];
    each ($r in $rows) {
        if (len($out) >= $limit) { break; }
        if ($blockedBy[str($r["id"])] != null) { continue; }
        if ($r["id"] != $u["id"] && !age.canSee($u, $r)) { continue; }
        $a = age.ageNow($r["dob"]);
        if ($minAge != null && $a < $minAge) { continue; }
        if ($maxAge != null && $a > $maxAge) { continue; }
        $p = orm.first(orm.filter(orm.query(models.M["profiles"]),
                                  [["user_id", $r["id"]]]));
        append($out, privacy.publicUserWithProfile($r, $p, $u));
    }

    $b = age.contentBounds($u);
    return h.okJson($res, {
        "query": $q,
        "results": $out,
        "band": $b["band"],
        "ageWindow": { "min": $b["min"], "max": $b["max"] }
    });
}

def register($sua) {
    $sua.server.get("/api/users/:username", getUserHandler);
    $sua.server.put("/api/settings/profile", updateProfileHandler);
    $sua.server.put("/api/settings/tags", updateTagsHandler);
    $sua.server.put("/api/settings/range", updateRangeHandler);
    $sua.server.put("/api/settings/explicit", updateExplicitHandler);
    $sua.server.get("/api/search/users", searchHandler);
}
