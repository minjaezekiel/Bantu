// ============================================================================
// one-taylor/backend/realtime/chat.b — conversations and messages
//
// WHO MAY MESSAGE WHOM, and why it is stricter than who may see whom:
//
//   1. canInteract — the age wall, in both directions. Same as everywhere.
//   2. neither has blocked the other.
//   3. THEY FOLLOW EACH OTHER. A mutual follow is this app's "match": both
//      people chose it, separately, before either could send a word.
//
// Rule 3 is the one that does the work. A feed is a thing you read; a direct
// message is a thing that arrives, and an app where any stranger in your band
// can open a thread with a child is an app with a grooming channel built in —
// whatever the classifier does afterwards. Requiring both sides to opt in
// first costs an adult one tap and removes the channel entirely.
//
// Every message is moderated with the same decision table as a post
// (safety/moderation.b). This is where grooming actually happens, so a message
// gets the strict treatment and not a lighter one.
//
// Exposes:
//   pairKey($a, $b)                    -> {low, high}
//   mayMessage($a, $b)                 -> {"ok", "reason"}
//   findOrOpen($viewer, $other)        -> conversation row, or null
//   listFor($viewer, $limit)           -> conversation summaries
//   history($viewer, $conv, $limit, $beforeId)
//   send($sender, $conv, $text)        -> {"ok", "message", "verdict", ...}
//   markRead($viewer, $conv)
//   participants($conv)                -> {low, high}
//   otherId($conv, $meId)
// ============================================================================

include "orm" as orm;
include "../models.b" as models;
include "../safety/age.b" as age;
include "../safety/moderation.b" as moderation;
include "../auth/privacy.b" as privacy;

const $MAX_MESSAGE = 2000;

def pairKey($a, $b) {
    $x = floor($a);
    $y = floor($b);
    if ($x <= $y) { return { "low": $x, "high": $y }; }
    return { "low": $y, "high": $x };
}

def _blockedEitherWay($aId, $bId) {
    if (orm.exists(orm.filter(orm.query(models.M["blocks"]),
            [["user_id", $aId], ["blocked_id", $bId]]))) { return true; }
    return orm.exists(orm.filter(orm.query(models.M["blocks"]),
            [["user_id", $bId], ["blocked_id", $aId]]));
}

def _follows($aId, $bId) {
    return orm.exists(orm.filter(orm.query(models.M["follows"]),
            [["follower_id", $aId], ["followee_id", $bId]]));
}

def mutualFollow($aId, $bId) {
    return _follows($aId, $bId) && _follows($bId, $aId);
}

// $a and $b are user rows carrying dob (and, for an adult, the seek range).
def mayMessage($a, $b) {
    if ($a == null || $b == null) { return { "ok": false, "reason": "no such account" }; }
    if ($a["id"] == $b["id"]) { return { "ok": false, "reason": "that is you" }; }
    if (!age.canInteract($a, $b)) {
        // The same answer the rest of the API gives across the wall: not
        // "you may not", which would confirm the account exists.
        return { "ok": false, "reason": "no such account" };
    }
    if ($b["status"] != null && $b["status"] != "active") {
        return { "ok": false, "reason": "no such account" };
    }
    if (_blockedEitherWay($a["id"], $b["id"])) {
        return { "ok": false, "reason": "no such account" };
    }
    if (!mutualFollow($a["id"], $b["id"])) {
        return { "ok": false,
                 "reason": "you both need to follow each other before you can talk" };
    }
    return { "ok": true, "reason": null };
}

def findOrOpen($viewer, $other) {
    $gate = mayMessage($viewer, $other);
    if (!$gate["ok"]) { return null; }
    $k = pairKey($viewer["id"], $other["id"]);
    $conv = orm.first(orm.filter(orm.query(models.M["conversations"]),
                      [["low_id", $k["low"]], ["high_id", $k["high"]]]));
    if ($conv != null) { return $conv; }
    // The band is recorded on the conversation so a transition at 18 can find
    // and close the threads that are no longer allowed, without re-deriving
    // both participants' ages.
    $ins = orm.insert(models.M["conversations"], {
        "low_id": $k["low"], "high_id": $k["high"],
        "band": age.bandFor($viewer["dob"]),
        "last_preview": ""
    });
    if (!$ins.ok) { return null; }
    return orm.first(orm.filter(orm.query(models.M["conversations"]),
                     [["id", $ins.id]]));
}

def participants($conv) {
    return { "low": $conv["low_id"], "high": $conv["high_id"] };
}

def otherId($conv, $meId) {
    if ($conv["low_id"] == $meId) { return $conv["high_id"]; }
    return $conv["low_id"];
}

def isParticipant($conv, $meId) {
    return $conv["low_id"] == $meId || $conv["high_id"] == $meId;
}

// Re-checked on every read and every write, not just when the thread opens.
// A block, an unfollow or an 18th birthday must close a thread that was
// legitimate yesterday.
def stillAllowed($viewer, $conv) {
    if ($conv == null || !isParticipant($conv, $viewer["id"])) { return false; }
    $other = _userRow(otherId($conv, $viewer["id"]));
    if ($other == null) { return false; }
    return mayMessage($viewer, $other)["ok"];
}

def _userRow($id) {
    $u = orm.first(orm.filter(orm.query(models.M["users"]), [["id", $id]]));
    if ($u == null) { return null; }
    $p = orm.first(orm.filter(orm.query(models.M["profiles"]), [["user_id", $id]]));
    if ($p != null) {
        $u["seek_min_age"] = $p["seek_min_age"];
        $u["seek_max_age"] = $p["seek_max_age"];
        $u["profile"] = $p;
    }
    return $u;
}

def userRow($id) { return _userRow($id); }

// ── listing ─────────────────────────────────────────────────────────────────
def listFor($viewer, $limit) {
    $me = floor($viewer["id"]);
    $rows = orm.all(orm.limit(orm.orderBy(
        orm.whereRaw(orm.query(models.M["conversations"]),
                     "low_id = " + str($me) + " OR high_id = " + str($me)),
        ["-last_message_at", "-id"]), $limit));

    $out = [];
    each ($c in $rows) {
        $other = _userRow(otherId($c, $me));
        if ($other == null) { continue; }
        // A thread the wall no longer permits simply is not listed. It is not
        // deleted either — if the block is lifted it comes back.
        if (!mayMessage($viewer, $other)["ok"]) { continue; }
        $unread = orm.count(orm.whereRaw(
            orm.filter(orm.query(models.M["messages"]),
                       [["conversation_id", $c["id"]], ["status", "live"]]),
            "sender_id <> " + str($me) + " AND read_at IS NULL"));
        append($out, {
            "id": $c["id"],
            "with": privacy.publicUserWithProfile($other, $other["profile"], $viewer),
            "lastMessageAt": $c["last_message_at"],
            "preview": $c["last_preview"],
            "unread": $unread
        });
    }
    return $out;
}

def _shape($m, $meId) {
    return {
        "id": $m["id"],
        "body": $m["body"],
        "mine": $m["sender_id"] == $meId,
        "kind": $m["kind"],
        "status": $m["status"],
        "createdAt": $m["created_at"],
        "readAt": $m["read_at"]
    };
}

def history($viewer, $conv, $limit, $beforeId) {
    $me = floor($viewer["id"]);
    $qs = orm.filter(orm.query(models.M["messages"]),
                     [["conversation_id", $conv["id"]]]);
    // A held or blocked message is visible to its sender only: they need to
    // see what happened to their own words, and the recipient was never meant
    // to receive them.
    $qs = orm.whereRaw($qs, "(status = 'live' OR sender_id = " + str($me) + ")");
    if ($beforeId != null && $beforeId > 0) {
        $qs = orm.filter($qs, [["id__lt", floor($beforeId)]]);
    }
    $rows = orm.all(orm.limit(orm.orderBy($qs, ["-id"]), $limit));
    // Oldest first for rendering; the query is newest-first so the LIMIT takes
    // the most recent page.
    $out = [];
    $i = len($rows) - 1;
    while ($i >= 0) {
        append($out, _shape($rows[$i], $me));
        $i = $i - 1;
    }
    return $out;
}

def markRead($viewer, $conv) {
    $me = floor($viewer["id"]);
    $r = orm.modify(orm.whereRaw(
            orm.filter(orm.query(models.M["messages"]),
                       [["conversation_id", $conv["id"]]]),
            "sender_id <> " + str($me) + " AND read_at IS NULL"),
            {"read_at": orm.raw("CURRENT_TIMESTAMP")});
    if ($r == null || $r.changes == null) { return 0; }
    return $r.changes;
}

// ── sending ─────────────────────────────────────────────────────────────────
def send($sender, $conv, $text) {
    $body = privacy.sanitiseText($text, $MAX_MESSAGE);
    if ($body == "") { return { "ok": false, "error": "write something first" }; }
    if (!stillAllowed($sender, $conv)) {
        return { "ok": false, "error": "you cannot message this account" };
    }

    $band = age.bandFor($sender["dob"]);
    $verdict = moderation.assess($body, $band);
    $v = $verdict["verdict"];

    // A message is not a post, and the difference changes who may stop one.
    //
    // For a post, "hold" is cheap: the author waits, nobody else was expecting
    // it. For a direct message, not delivering it IS the irreversible action —
    // one side of a conversation simply stops hearing from the other, and
    // neither of them is told why in a way they can act on. At the classifier's
    // measured precision that would silently break about one innocent exchange
    // in three, and the test caught exactly that: a 15-year-old's "scored twice
    // at practice today, best session all term" was held and never arrived.
    //
    // So on this path only a LEXICON hit stops a message. A model flag --
    // reliable or advisory -- delivers it and sends it to the review queue,
    // where a person reads the thread. A reviewer reading a grooming thread is
    // worth more than a blocked line that tells the groomer which words to
    // avoid next time.
    $status = "live";
    if ($v == "block") { $status = "blocked"; }
    $queued = $v != "allow";

    $ins = orm.insert(models.M["messages"], {
        "conversation_id": $conv["id"],
        "sender_id": $sender["id"],
        "body": $body,
        "status": $status,
        "kind": "text"
    });
    if (!$ins.ok) { return { "ok": false, "error": "could not send that" }; }

    moderation.record("message", $ins.id, $v, "system",
                      join($verdict["reasons"], ","), $verdict["scores"]);

    if ($status == "live") {
        // The preview is truncated here rather than at render time so a long
        // message does not make the conversation list enormous.
        $preview = $body;
        if (len($preview) > 80) { $preview = substr($preview, 0, 80); }
        orm.modify(orm.filter(orm.query(models.M["conversations"]),
                              [["id", $conv["id"]]]),
                   {"last_message_at": orm.raw("CURRENT_TIMESTAMP"),
                    "last_preview": $preview});
    }

    $row = orm.first(orm.filter(orm.query(models.M["messages"]), [["id", $ins.id]]));
    $message = _shape($row, floor($sender["id"]));
    $note = null;
    if ($v == "block") {
        $note = "This was not sent. It matched: " + join($verdict["reasons"], ", ")
              + ". A moderator can review that.";
    } else if ($queued) {
        // Said plainly rather than hidden. Someone whose message is being read
        // by a moderator should know, and someone being groomed should see that
        // the app is paying attention.
        $note = "Sent. A moderator will read this conversation as well.";
    }
    return { "ok": true, "message": $message, "verdict": $v,
             "delivered": $status == "live", "queued": $queued, "note": $note };
}

// A call that happened (or was missed) leaves a line in the thread, so the
// history reads the way people remember it.
def recordCall($conv, $senderId, $text) {
    $ins = orm.insert(models.M["messages"], {
        "conversation_id": $conv["id"],
        "sender_id": $senderId,
        "body": $text,
        "status": "live",
        "kind": "call"
    });
    orm.modify(orm.filter(orm.query(models.M["conversations"]), [["id", $conv["id"]]]),
               {"last_message_at": orm.raw("CURRENT_TIMESTAMP"),
                "last_preview": $text});
    if (!$ins.ok) { return null; }
    return $ins.id;
}
