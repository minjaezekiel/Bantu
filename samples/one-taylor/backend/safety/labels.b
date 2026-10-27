// ============================================================================
// one-taylor/backend/safety/labels.b — explicit content categories and opt-ins
//
// A label is a fact about a post. An opt-in is a fact about a viewer. This
// module owns both vocabularies and the one function that decides how a
// labelled post is delivered to a given viewer.
//
// Three outcomes, and the difference between them matters:
//
//   "show"   no label the viewer has not accepted
//   "blur"   labelled, adult viewer, not opted in -> delivered blurred, with
//            the category named, revealed on a tap
//   "deny"   minor viewer and ANY label -> not delivered at all, 404, whether
//            the request came from a feed or from a guessed post id
//
// "deny" is not "filter from the feed". A post reachable by id is a post that
// is reachable, and the feed is not the only door.
//
// Exposes:
//   CATEGORIES / isCategory($c)
//   parseOptins($s) / serialiseOptins($list)
//   decide($labels, $viewer)  -> {"mode", "categories"}
//   fromModelScores($scores, $flags) -> list of {category, confidence}
// ============================================================================

include "./age.b" as age;

$CATEGORIES = ["suggestive", "nudity", "sexual", "fetish",
               "violence", "gore", "substances"];

def isCategory($c) {
    each ($k in $CATEGORIES) { if ($k == $c) { return true; } }
    return false;
}

// Stored as a comma-joined string in profiles.explicit_optins. A joined string
// is the wrong shape for searching, which is why user TAGS get their own
// table -- but opt-ins are only ever read for one user at a time, so a column
// is right here and a join would be ceremony.
def parseOptins($s) {
    $out = [];
    if ($s == null || $s == "") { return $out; }
    each ($part in split($s, ",")) {
        $t = trim($part);
        if ($t != "" && isCategory($t)) { append($out, $t); }
    }
    return $out;
}

def serialiseOptins($list) {
    $kept = [];
    if ($list == null) { return ""; }
    each ($c in $list) {
        if (isCategory($c)) { append($kept, $c); }
    }
    return join($kept, ",");
}

def _has($list, $v) {
    each ($x in $list) { if ($x == $v) { return true; } }
    return false;
}

// $labels is a list of category strings on the post.
def decide($labels, $viewer) {
    if ($labels == null || len($labels) == 0) {
        return { "mode": "show", "categories": [] };
    }
    if ($viewer == null) {
        return { "mode": "deny", "categories": $labels };
    }
    if (age.bandFor($viewer["dob"]) == age.MINOR_BAND()) {
        // No opt-in exists for a minor. There is no setting to find, no
        // control to enable, and no code path that consults one.
        return { "mode": "deny", "categories": $labels };
    }
    $optins = parseOptins($viewer["explicit_optins"]);
    $unaccepted = [];
    each ($c in $labels) {
        if (!_has($optins, $c)) { append($unaccepted, $c); }
    }
    if (len($unaccepted) == 0) { return { "mode": "show", "categories": $labels }; }
    return { "mode": "blur", "categories": $unaccepted };
}

// The classifier speaks in harm labels (sexual, sexual_minor, ...); the
// content-warning vocabulary is different and coarser. This is the mapping,
// and it is deliberately conservative: the model's "sexual" becomes
// "suggestive" rather than "nudity", because a text classifier cannot see an
// image and should not claim to.
def fromModelScores($scores, $flags) {
    $out = [];
    if ($flags["sexual"]) {
        append($out, { "category": "suggestive", "confidence": $scores["sexual"] });
    }
    if ($flags["threat"]) {
        append($out, { "category": "violence", "confidence": $scores["threat"] });
    }
    return $out;
}
