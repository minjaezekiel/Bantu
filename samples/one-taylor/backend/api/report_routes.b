// ============================================================================
// one-taylor/backend/api/report_routes.b — reporting and hiding
//
//   POST /api/posts/:id/report   {reason}
//   POST /api/posts/:id/hide
//   POST /api/posts/:id/unhide
//
// One report hides the post from the reporter IMMEDIATELY. Whatever the
// community eventually decides, the person who objected stops seeing it now.
//
// TWO distinct reporters on an explicit/policy reason takes the post down.
//
// Brigading guard: two accounts are easy to make, so a report is RECORDED but
// not COUNTED toward the threshold when it comes from a throwaway. The
// threshold stays at two, as specified; the uncounted reports still appear in
// the review queue, because a brigade is itself worth seeing.
//
// The definition of "throwaway" matters, and the first version of it was
// wrong: "account under an hour old" alone would mean that on the day this
// app launches, every account is under an hour old and NO report counts for
// anything. The safety feature would be off exactly when the community is
// smallest and most fragile. The API test caught it.
//
// Nor is "new and empty, therefore ignored" right on its own -- the second
// version of this rule. A lurker who signs up, reads for ten minutes and
// reports something genuinely vile has posted nothing and followed nobody,
// and lurkers are a large share of real reporters. Dropping their report
// entirely silences the people most likely to use the button.
//
// So: a brand-new, empty account's report is WEAK, and at most ONE weak
// report counts toward a post's threshold.
//
//   attacker with two sockpuppets  -> 1 counted, no takedown. Closed.
//   one lurker + one real user     -> 2 counted, takedown. Works.
//   two genuine new lurkers        -> 1 counted; the post goes to the review
//                                     queue for a person instead. Visible,
//                                     not silent.
//
// Exposes: register($sua)
// ============================================================================

include "orm" as orm;
include "../config.b" as cfg;
include "../models.b" as models;
include "../safety/age.b" as age;
include "../safety/moderation.b" as moderation;
include "../auth/sessions.b" as sessions;
include "./helpers.b" as h;

$REASONS = ["explicit", "harassment", "violence", "self_harm", "spam",
            "underage", "impersonation", "other"];

// The reasons that count toward an automatic takedown. "spam" and
// "impersonation" need a human: two people calling something spam is a
// disagreement about taste, not a policy violation, and impersonation needs
// evidence nobody can supply by pressing a button.
$POLICY_REASONS = ["explicit", "harassment", "violence", "self_harm", "underage"];

const $NEW_ACCOUNT_SECONDS = 3600;

def _validReason($r) {
    each ($x in $REASONS) { if ($x == $r) { return true; } }
    return false;
}

def _isPolicy($r) {
    each ($x in $POLICY_REASONS) { if ($x == $r) { return true; } }
    return false;
}

// How old is this account, in seconds? Read from the row and computed in
// Bantu (age.secondsSince) rather than with strftime arithmetic in SQL: it is
// one fewer round trip on a write path, and the calendar it uses is the same
// one the age wall uses, which is the property that matters.
def _accountAgeSeconds($userId) {
    $u = orm.first(orm.values(orm.filter(orm.query(models.M["users"]),
                              [["id", $userId]]), ["created_at"]));
    if ($u == null || $u["created_at"] == null) { return 999999; }
    return age.secondsSince($u["created_at"]);
}

// New and empty: never posted, never followed anybody. Either one on its own
// is ordinary; together, on a brand-new account, it is a button-presser.
def _isThrowaway($userId) {
    if (_accountAgeSeconds($userId) >= $NEW_ACCOUNT_SECONDS) { return false; }
    if (orm.exists(orm.filter(orm.query(models.M["posts"]), [["author_id", $userId]]))) {
        return false;
    }
    if (orm.exists(orm.filter(orm.query(models.M["follows"]),
                              [["follower_id", $userId]]))) {
        return false;
    }
    return true;
}

def reportHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $id = num($req.params["id"], null);
    if ($id == null) { return h.fail($res, 404, "no such post"); }
    $body = h.bodyOf($req);
    $reason = h.strField($body, "reason", 30).lower();
    if (!_validReason($reason)) {
        return h.fail($res, 400, "pick a reason: " + join($REASONS, ", "));
    }

    $post = orm.first(orm.filter(orm.query(models.M["posts"]), [["id", floor($id)]]));
    if ($post == null) { return h.fail($res, 404, "no such post"); }
    $author = orm.first(orm.filter(orm.query(models.M["users"]),
                                   [["id", $post["author_id"]]]));
    if ($author == null) { return h.fail($res, 404, "no such post"); }
    // You can only report what you can see. Otherwise reporting is a probe for
    // the existence of posts on the other side of the wall.
    if (!age.canSee($u, $author)) { return h.fail($res, 404, "no such post"); }
    if ($author["id"] == $u["id"]) {
        return h.fail($res, 400, "delete your own post instead");
    }

    // Hide it from this reporter straight away, whatever happens next.
    if (!orm.exists(orm.filter(orm.query(models.M["hidden_posts"]),
            [["user_id", $u["id"]], ["post_id", $post["id"]]]))) {
        orm.insert(models.M["hidden_posts"],
                   {"user_id": $u["id"], "post_id": $post["id"], "reason": $reason});
    }

    // Does this report count, and if so does it count fully? Recorded either
    // way -- an uncounted report is still evidence, and still reaches the
    // review queue.
    $counted = 1;
    $weak = 0;
    $why = "";
    if (_isThrowaway($u["id"])) {
        $weak = 1;
        $why = "reporter is a new account with no posts and no follows";
    }
    if (orm.exists(orm.filter(orm.query(models.M["blocks"]),
            [["user_id", $author["id"]], ["blocked_id", $u["id"]]]))) {
        $counted = 0;
        $why = "author had blocked the reporter";
    }
    if (!_isPolicy($reason)) {
        $counted = 0;
        $why = "reason needs a human";
    }

    // The unique index on (post_id, reporter_id) makes a second report from
    // the same account a no-op rather than a second vote.
    $already = orm.first(orm.filter(orm.query(models.M["reports"]),
                   [["post_id", $post["id"]], ["reporter_id", $u["id"]]]));
    if ($already == null) {
        orm.insert(models.M["reports"], {
            "post_id": $post["id"], "reporter_id": $u["id"],
            "reason": $reason, "counted": $counted, "weak": $weak
        });
        if (($counted == 0 || $weak == 1) && $why != "") {
            $kind = "report_weak";
            if ($counted == 0) { $kind = "report_uncounted"; }
            moderation.record("post", $post["id"], $kind,
                              "user:" + str($u["id"]), $why, {});
        }
    }

    // Strong reports count individually; every weak one together counts once.
    $strong = orm.count(orm.filter(orm.query(models.M["reports"]),
                   [["post_id", $post["id"]], ["counted", 1], ["weak", 0]]));
    $weakN = orm.count(orm.filter(orm.query(models.M["reports"]),
                   [["post_id", $post["id"]], ["counted", 1], ["weak", 1]]));
    $countedTotal = $strong;
    if ($weakN > 0) { $countedTotal = $countedTotal + 1; }
    $allTotal = orm.count(orm.filter(orm.query(models.M["reports"]),
                       [["post_id", $post["id"]]]));
    orm.modify(orm.filter(orm.query(models.M["posts"]), [["id", $post["id"]]]),
               {"report_count": $allTotal});

    $removed = false;
    if ($countedTotal >= cfg.CFG["reportThreshold"] && $post["status"] == "live") {
        orm.modify(orm.filter(orm.query(models.M["posts"]), [["id", $post["id"]]]),
                   {"status": "removed"});
        moderation.record("post", $post["id"], "removed_by_reports", "system",
                          str($strong) + " strong + " + str($weakN)
                          + " weak reports = " + str($countedTotal)
                          + " counted (threshold "
                          + str(cfg.CFG["reportThreshold"]) + ")", {});
        $removed = true;
    }

    return h.okJson($res, {
        "reported": true,
        "hiddenFromYou": true,
        // The reporter is told the post is gone, and nothing more. The count,
        // the threshold and whose reports counted are moderation internals;
        // publishing them turns the threshold into a game with a visible score.
        "removed": $removed
    });
}

def hideHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $id = num($req.params["id"], null);
    if ($id == null) { return h.fail($res, 404, "no such post"); }
    if (!orm.exists(orm.filter(orm.query(models.M["posts"]), [["id", floor($id)]]))) {
        return h.fail($res, 404, "no such post");
    }
    if (!orm.exists(orm.filter(orm.query(models.M["hidden_posts"]),
            [["user_id", $u["id"]], ["post_id", floor($id)]]))) {
        orm.insert(models.M["hidden_posts"],
                   {"user_id": $u["id"], "post_id": floor($id), "reason": "hidden"});
    }
    return h.okJson($res, { "hidden": true });
}

def unhideHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $id = num($req.params["id"], null);
    if ($id == null) { return h.fail($res, 404, "no such post"); }
    orm.remove(orm.filter(orm.query(models.M["hidden_posts"]),
               [["user_id", $u["id"]], ["post_id", floor($id)]]), false);
    return h.okJson($res, { "hidden": false });
}

def register($sua) {
    $sua.server.post("/api/posts/:id/report", reportHandler);
    $sua.server.post("/api/posts/:id/hide", hideHandler);
    $sua.server.post("/api/posts/:id/unhide", unhideHandler);
}
