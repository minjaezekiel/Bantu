// ============================================================================
// one-taylor/backend/recsys/posts.b — the feed
//
// Gates first, scoring second, same as people.b and for the same reason.
//
// The gate is age.visibilityPairs() -- the single source of the band
// condition -- plus: removed posts, posts the viewer hid, posts by accounts
// either party blocked, and (for a minor) any post carrying a content label.
//
//   rank = 0.40 tagAffinity + 0.25 follows + 0.20 engagement + 0.15 recency
//
// Candidates come from three places so the feed is neither an echo chamber nor
// a stranger's firehose: accounts the viewer follows, accounts the matcher
// scores highly, and what is popular inside the viewer's own band.
//
// Exposes:
//   feed($viewer, $limit, $beforeId) -> list of hydrated, gated posts
//   hydrate($post, $viewer)          -> the API shape, with label decision
// ============================================================================

include "orm" as orm;
include "../models.b" as models;
include "../safety/age.b" as age;
include "../safety/labels.b" as labels;
include "../auth/privacy.b" as privacy;
include "./vectors.b" as vec;

const $W_AFFINITY   = 0.40;
const $W_FOLLOWS    = 0.25;
const $W_ENGAGEMENT = 0.20;
const $W_RECENCY    = 0.15;

const $HALF_LIFE_HOURS = 36;

// ── hydration ───────────────────────────────────────────────────────────────
// Every post that leaves this server goes through here, so the label decision
// and the author projection cannot be forgotten by a route.
def hydrate($post, $viewer) {
    $author = orm.first(orm.filter(orm.query(models.M["users"]),
                                   [["id", $post["author_id"]]]));
    $profile = null;
    if ($author != null) {
        $profile = orm.first(orm.filter(orm.query(models.M["profiles"]),
                                        [["user_id", $author["id"]]]));
    }
    $labelRows = orm.all(orm.filter(orm.query(models.M["post_labels"]),
                                    [["post_id", $post["id"]]]));
    $cats = [];
    each ($l in $labelRows) { append($cats, $l["category"]); }

    $decision = labels.decide($cats, $viewer);
    if ($decision["mode"] == "deny") { return null; }

    $mediaRows = orm.all(orm.orderBy(orm.filter(orm.query(models.M["post_media"]),
                                                [["post_id", $post["id"]]]), ["position"]));
    $media = [];
    each ($mr in $mediaRows) {
        // The id, never the path. A path is a filesystem fact and a gate the
        // client could route around; /api/media/:id re-runs the whole check.
        append($media, { "id": $mr["id"], "mime": $mr["mime"] });
    }

    $tagRows = orm.all(orm.filter(orm.query(models.M["post_tags"]),
                                  [["post_id", $post["id"]]]));
    $tags = [];
    each ($tr in $tagRows) { append($tags, $tr["tag"]); }

    return {
        "id": $post["id"],
        "body": $post["body"],
        "createdAt": $post["created_at"],
        "likeCount": $post["like_count"],
        "author": privacy.publicUserWithProfile($author, $profile, $viewer),
        "media": $media,
        "tags": $tags,
        "labels": $cats,
        "display": $decision["mode"],
        "blurredFor": $decision["categories"]
    };
}

// ── the feed ────────────────────────────────────────────────────────────────
//
// Entirely through orm.b, and BATCHED. The batching is not an optimisation.
//
// The first version called hydrate() per candidate post. hydrate() issues five
// querysets of its own -- author, profile, labels, media, tags -- and the
// ranking needed a sixth for the post's age. Over a 1,200-post table that is
// about 7,000 round trips for one feed request, and it measured 677 ms. Since
// sua.sqlite never suspends, those 677 ms are added to every other client
// waiting on the same worker: one person opening their feed slows everybody
// down.
//
// Now: one queryset for the candidates, then five more with `field__in` for
// everything the ranking and the rendering need -- six in total, whatever the
// page size -- and the post's age comes from age.hoursSince(), which is
// arithmetic rather than a query.
//
// 180 candidates for a 20-post page is nine times the oversampling the ranking
// needs, and it bounds the work a single request can ask of a worker. Older
// posts stay reachable through the `before` cursor.
const $CANDIDATE_CAP = 180;

def _groupBy($rows, $key) {
    $out = {};
    each ($r in $rows) {
        $k = str($r[$key]);
        if ($out[$k] == null) { $out[$k] = []; }
        $list = $out[$k];
        append($list, $r);
        $out[$k] = $list;
    }
    return $out;
}

def _ids($rows, $field) {
    $seen = {};
    $out = [];
    each ($r in $rows) {
        $k = str($r[$field]);
        if ($seen[$k] == null) { $seen[$k] = true; append($out, $r[$field]); }
    }
    return $out;
}

def feed($viewer, $limit, $beforeId) {
    // ── the gate, in SQL ───────────────────────────────────────────────────
    // age.visibilityPairs is the single source of the band condition.
    $qs = orm.filter(orm.query(models.M["posts"]), age.visibilityPairs($viewer));
    if ($beforeId != null && $beforeId > 0) {
        $qs = orm.filter($qs, [["id__lt", floor($beforeId)]]);
    }
    // Blocks and per-viewer hides, excluded here rather than after hydration:
    // otherwise they are fetched, assembled and thrown away. orm.whereRaw is
    // the ORM's own escape hatch for a predicate its pair filters cannot
    // express -- the id is this server's own primary key, floored, so nothing
    // but a number reaches the SQL.
    $me = floor($viewer["id"]);
    $qs = orm.whereRaw($qs,
        "author_id NOT IN (SELECT blocked_id FROM blocks WHERE user_id = " + str($me) + ")");
    $qs = orm.whereRaw($qs,
        "author_id NOT IN (SELECT user_id FROM blocks WHERE blocked_id = " + str($me) + ")");
    $qs = orm.whereRaw($qs,
        "id NOT IN (SELECT post_id FROM hidden_posts WHERE user_id = " + str($me) + ")");

    $rows = orm.all(orm.limit(orm.orderBy($qs, ["-id"]), $CANDIDATE_CAP));
    if (len($rows) == 0) { return []; }

    // ── everything else, five querysets, whatever the page size ───────────
    $postIds = _ids($rows, "id");
    $authorIds = _ids($rows, "author_id");

    $authors = {};
    each ($a in orm.all(orm.filter(orm.query(models.M["users"]),
                                   [["id__in", $authorIds]]))) {
        $authors[str($a["id"])] = $a;
    }
    $profiles = {};
    each ($pr in orm.all(orm.filter(orm.query(models.M["profiles"]),
                                    [["user_id__in", $authorIds]]))) {
        $profiles[str($pr["user_id"])] = $pr;
    }
    $labelsByPost = _groupBy(orm.all(orm.filter(orm.query(models.M["post_labels"]),
                                     [["post_id__in", $postIds]])), "post_id");
    $mediaByPost = _groupBy(orm.all(orm.orderBy(
                                orm.filter(orm.query(models.M["post_media"]),
                                           [["post_id__in", $postIds]]),
                                ["position"])), "post_id");
    $tagsByPost = _groupBy(orm.all(orm.filter(orm.query(models.M["post_tags"]),
                                   [["post_id__in", $postIds]])), "post_id");

    $following = {};
    each ($f in orm.all(orm.filter(orm.query(models.M["follows"]),
                                   [["follower_id", $viewer["id"]]]))) {
        $following[str($f["followee_id"])] = true;
    }
    $myInterests = {};
    each ($mt in orm.all(orm.filter(orm.query(models.M["user_tags"]),
                                    [["user_id", $viewer["id"]]]))) {
        $myInterests[$mt["tag"]] = true;
    }

    // ── gate again exactly, then score ─────────────────────────────────────
    $items = [];
    $cAff = [];
    $cFol = [];
    $cEng = [];
    $cRec = [];
    each ($r in $rows) {
        $author = $authors[str($r["author_id"])];
        if ($author == null) { continue; }
        // The SQL bound on author_age_at_post is a pre-filter. canSee is the
        // decision, and it runs on every row, every time.
        if (!age.canSee($viewer, $author)) { continue; }

        $cats = [];
        $lr = $labelsByPost[str($r["id"])];
        if ($lr != null) { each ($l in $lr) { append($cats, $l["category"]); } }
        $decision = labels.decide($cats, $viewer);
        if ($decision["mode"] == "deny") { continue; }

        $media = [];
        $mr = $mediaByPost[str($r["id"])];
        if ($mr != null) {
            // The id, never the path. A path is a filesystem fact and a gate
            // the client could route around; /api/media/:id re-runs the check.
            each ($m in $mr) { append($media, { "id": $m["id"], "mime": $m["mime"] }); }
        }
        $tags = [];
        $postTags = {};
        $tr = $tagsByPost[str($r["id"])];
        if ($tr != null) {
            each ($tg in $tr) { append($tags, $tg["tag"]); $postTags[$tg["tag"]] = true; }
        }

        append($cAff, vec.jaccard($myInterests, $postTags));
        $isFollowed = 0.0;
        if ($following[str($r["author_id"])] != null) { $isFollowed = 1.0; }
        append($cFol, $isFollowed);
        // Engagement, compressed: 0 to 5 likes should matter more than 100 to 105.
        append($cEng, clamp(log1p($r["like_count"]) / 5.0, 0.0, 1.0));
        append($cRec, vec.recencyWeight(age.hoursSince($r["created_at"]),
                                        $HALF_LIFE_HOURS));

        append($items, {
            "id": $r["id"],
            "body": $r["body"],
            "createdAt": $r["created_at"],
            "likeCount": $r["like_count"],
            "author": privacy.publicUserWithProfile($author,
                          $profiles[str($r["author_id"])], $viewer),
            "media": $media,
            "tags": $tags,
            "labels": $cats,
            "display": $decision["mode"],
            "blurredFor": $decision["categories"]
        });
    }

    if (len($items) == 0) { return []; }
    $scores = vec.combine([$cAff, $cFol, $cEng, $cRec],
                          [$W_AFFINITY, $W_FOLLOWS, $W_ENGAGEMENT, $W_RECENCY]);
    return vec.rankTop($items, $scores, $limit);
}

// A profile page's posts. Still gated: a profile page is a read path like any
// other, and "I typed the username" is not an authorisation.
def byAuthor($viewer, $authorId, $limit) {
    $mine = $viewer["id"] == $authorId;
    if ($mine) {
        // On your OWN profile you see your held and blocked posts too, each
        // carrying its status. A post that was quietly withheld and appears
        // nowhere is indistinguishable from one the app lost, and it leaves
        // nothing to appeal; showing it, badged, is what makes the decision
        // reviewable by the person it was made about.
        $qs = orm.limit(orm.orderBy(orm.whereRaw(
                orm.filter(orm.query(models.M["posts"]), [["author_id", $authorId]]),
                "status <> 'deleted'"), ["-id"]), $limit);
        $out = [];
        each ($p in orm.all($qs)) {
            $h = hydrate($p, $viewer);
            if ($h == null) { continue; }
            $h["mine"] = true;
            $h["status"] = $p["status"];
            append($out, $h);
        }
        return $out;
    }

    $pairs = [];
    extend($pairs, age.visibilityPairs($viewer));
    append($pairs, ["author_id", $authorId]);
    $qs = orm.limit(orm.orderBy(orm.filter(orm.query(models.M["posts"]), $pairs),
                                ["-id"]), $limit);
    $out = [];
    each ($p in orm.all($qs)) {
        $h = hydrate($p, $viewer);
        if ($h != null) { append($out, $h); }
    }
    return $out;
}
