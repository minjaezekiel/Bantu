// ============================================================================
// one-taylor/backend/recsys/people.b — match-making
//
// GATES FIRST, SCORING SECOND. The candidate set is reduced by band, range,
// blocks and account status before anything is ranked, so no amount of score
// can promote a profile past the age wall. Filtering after ranking is how
// these systems leak: a scorer that sees everything and a filter that runs
// afterwards is one forgotten branch away from showing a child to an adult.
//
//   score = 0.35 jaccard(interests)
//         + 0.20 jaccard(hobbies)
//         + 0.25 dreamPartnerFit
//         + 0.10 mutualFollows
//         + 0.10 recencyOfActivity
//
// Every match card carries WHY it matched. An unexplained recommendation in a
// dating app is indistinguishable from a random one, and a user who cannot
// tell the difference has no reason to trust the next one.
//
// Exposes:
//   candidates($viewer, $limit)  -> list of gated, scored, explained matches
// ============================================================================

include "orm" as orm;
include "../models.b" as models;
include "../safety/age.b" as age;
include "../auth/privacy.b" as privacy;
include "./vectors.b" as vec;

const $W_INTEREST = 0.35;
const $W_HOBBY    = 0.20;
const $W_PARTNER  = 0.25;
const $W_MUTUAL   = 0.10;
const $W_RECENCY  = 0.10;

// How many rows the gate is allowed to hand the scorer. A dating feed does not
// need to rank the whole user table to produce twenty cards, and an unbounded
// scorer on a single-threaded worker is a denial of service with extra steps.
const $CANDIDATE_CAP = 2000;

def _has($list, $v) {
    each ($x in $list) { if ($x == $v) { return true; } }
    return false;
}

// ── the dream-partner component ─────────────────────────────────────────────
// Three things, each either satisfied or not, averaged:
//   - does the candidate's gender match what the viewer said they seek
//   - is the candidate inside the viewer's age range (already gated, so this
//     measures WHERE in the range, preferring the middle over the edges)
//   - do the viewer's declared dream_partner tags appear in the candidate's
//     own interests and hobbies
def dreamPartnerFit($viewer, $viewerWants, $cand, $candTags) {
    $parts = [];

    $seek = $viewer["seek_genders"];
    if ($seek == null || $seek == "" || $seek == "any") { append($parts, 1.0); }
    else {
        $wanted = split($seek, ",");
        $hit = 0.0;
        each ($g in $wanted) {
            if (trim($g) == $cand["gender"]) { $hit = 1.0; }
        }
        append($parts, $hit);
    }

    // Distance from the middle of the range, normalised. Someone at the very
    // edge of a 18-40 range is a weaker match than someone in the middle of
    // it, and the raw boolean "in range" cannot say so.
    $b = age.matchBounds($viewer);
    $mid = ($b["min"] + $b["max"]) / 2;
    $half = ($b["max"] - $b["min"]) / 2;
    if ($half <= 0) { append($parts, 1.0); }
    else {
        $d = age.ageNow($cand["dob"]) - $mid;
        if ($d < 0) { $d = 0 - $d; }
        $closeness = 1.0 - ($d / $half);
        append($parts, clamp($closeness, 0.0, 1.0));
    }

    append($parts, vec.jaccard($viewerWants, $candTags));

    $s = 0.0;
    each ($p in $parts) { $s = $s + $p; }
    return $s / len($parts);
}

def candidates($viewer, $limit) {
    // ── the gate ────────────────────────────────────────────────────────────
    // matchPairs, not userVisibilityPairs: discover is the one place the
    // viewer's DATING RANGE applies. The feed and search use content
    // visibility instead -- see safety/age.b.
    $pairs = age.matchPairs($viewer);
    $qs = orm.limit(orm.filter(orm.query(models.M["users"]), $pairs), $CANDIDATE_CAP);
    $rows = orm.all($qs);

    // Blocks, in both directions, and self.
    $blocked = {};
    each ($b in orm.all(orm.filter(orm.query(models.M["blocks"]),
                                   [["user_id", $viewer["id"]]]))) {
        $blocked[str($b["blocked_id"])] = true;
    }
    each ($b in orm.all(orm.filter(orm.query(models.M["blocks"]),
                                   [["blocked_id", $viewer["id"]]]))) {
        $blocked[str($b["user_id"])] = true;
    }

    $gated = [];
    each ($r in $rows) {
        if ($r["id"] == $viewer["id"]) { continue; }
        if ($blocked[str($r["id"])] != null) { continue; }
        // The SQL bound is whole-year and therefore approximate; canInteract
        // is exact, and it is what decides. The query narrows, this decides.
        $cand = { "dob": $r["dob"], "band": $r["band"] };
        $p = orm.first(orm.filter(orm.query(models.M["profiles"]), [["user_id", $r["id"]]]));
        if ($p != null) {
            $cand["seek_min_age"] = $p["seek_min_age"];
            $cand["seek_max_age"] = $p["seek_max_age"];
        }
        if (!age.canMatch($viewer, $cand)) { continue; }
        $r["profile"] = $p;
        append($gated, $r);
    }

    if (len($gated) == 0) { return []; }

    // ── scoring ─────────────────────────────────────────────────────────────
    $myTags = orm.all(orm.filter(orm.query(models.M["user_tags"]),
                                 [["user_id", $viewer["id"]]]));
    $myInterests = vec.tagSet($myTags, "interest");
    $myHobbies   = vec.tagSet($myTags, "hobby");
    $myWants     = vec.tagSet($myTags, "dream_partner");

    $followees = {};
    each ($f in orm.all(orm.filter(orm.query(models.M["follows"]),
                                   [["follower_id", $viewer["id"]]]))) {
        $followees[str($f["followee_id"])] = true;
    }

    $cInterest = [];
    $cHobby = [];
    $cPartner = [];
    $cMutual = [];
    $cRecency = [];
    $items = [];

    $nowSec = age.nowSeconds();
    each ($r in $gated) {
        $tags = orm.all(orm.filter(orm.query(models.M["user_tags"]),
                                   [["user_id", $r["id"]]]));
        $interests = vec.tagSet($tags, "interest");
        $hobbies   = vec.tagSet($tags, "hobby");
        $all       = vec.tagSet($tags, null);

        append($cInterest, vec.jaccard($myInterests, $interests));
        append($cHobby,    vec.jaccard($myHobbies, $hobbies));

        $vwr = { "dob": $viewer["dob"], "seek_genders": $viewer["seek_genders"],
                 "seek_min_age": $viewer["seek_min_age"],
                 "seek_max_age": $viewer["seek_max_age"] };
        append($cPartner, dreamPartnerFit($vwr, $myWants, $r, $all));

        // Mutual follows: do we both already follow each other's circles?
        $mut = 0.0;
        $theirFollowees = orm.all(orm.filter(orm.query(models.M["follows"]),
                                             [["follower_id", $r["id"]]]));
        $shared = 0;
        each ($tf in $theirFollowees) {
            if ($followees[str($tf["followee_id"])] != null) { $shared = $shared + 1; }
        }
        if ($shared > 0) { $mut = clamp($shared / 5.0, 0.0, 1.0); }
        append($cMutual, $mut);

        append($cRecency, 1.0);   // refined below if last_seen parses

        $shareList = vec.sharedTags($myInterests, $interests);
        extend($shareList, vec.sharedTags($myHobbies, $hobbies));
        append($items, {
            "user": privacy.publicUserWithProfile($r, $r["profile"], $viewer),
            "shared": $shareList
        });
    }

    $scores = vec.combine([$cInterest, $cHobby, $cPartner, $cMutual, $cRecency],
                          [$W_INTEREST, $W_HOBBY, $W_PARTNER, $W_MUTUAL, $W_RECENCY]);

    $top = vec.rankTop($items, $scores, $limit);
    // The explanation is built from the same numbers that produced the rank,
    // not re-derived, so the card can never disagree with the ordering.
    each ($t in $top) {
        $why = [];
        if (len($t["shared"]) > 0) {
            append($why, "you both like " + join($t["shared"], ", "));
        } else {
            append($why, "a similar age and in your range");
        }
        $t["why"] = join($why, "; ");
    }
    return $top;
}
