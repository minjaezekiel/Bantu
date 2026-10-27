// ============================================================================
// one-taylor/backend/api/realtime_routes.b — the HTTP half of messaging
//
//   POST /api/realtime/ticket                  a one-shot WebSocket ticket
//   GET  /api/realtime/config                  WS url + ICE servers
//   GET  /api/conversations                    the thread list
//   POST /api/conversations/:username          open (or find) a thread
//   GET  /api/conversations/:username/messages history, paginated
//   POST /api/conversations/:username/messages send without a WebSocket
//   POST /api/conversations/:username/read     mark read
//
// The WebSocket carries live traffic; these carry history, the first paint,
// and a working fallback for a browser whose socket is blocked. Both go
// through the same gate and the same moderation, because two doors with one
// lock is the usual way the second one ends up unlocked.
//
// Exposes: register($sua)
// ============================================================================

include "orm" as orm;
include "../config.b" as cfg;
include "../models.b" as models;
include "../safety/age.b" as age;
include "../auth/sessions.b" as sessions;
include "../auth/privacy.b" as privacy;
include "../realtime/tickets.b" as tickets;
include "../realtime/chat.b" as chat;
include "./helpers.b" as h;

def _other($req, $res, $u) {
    $name = $req.params["username"];
    if ($name == null || $name == "") {
        h.fail($res, 404, "no such account");
        return null;
    }
    $row = orm.first(orm.filter(orm.query(models.M["users"]),
                     [["username_lower", $name.lower()]]));
    if ($row != null) { $row = chat.userRow($row["id"]); }
    $gate = chat.mayMessage($u, $row);
    if (!$gate["ok"]) {
        // 404 for a wall or a block, 403 for "you both need to follow each
        // other" — the first must not confirm the account exists, the second
        // is an action the user can actually take.
        if ($gate["reason"] == "no such account") {
            h.fail($res, 404, $gate["reason"]);
        } else {
            h.fail($res, 403, $gate["reason"]);
        }
        return null;
    }
    return $row;
}

def ticketHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $t = tickets.issue($u);
    return h.okJson($res, { "ticket": $t["ticket"], "ttl": $t["ttl"] });
}

def configHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    // ICE servers come from the environment so an operator can point at their
    // own STUN/TURN -- `bantu relay` ships one. The default is a public STUN,
    // which is enough to connect two peers that are not both behind symmetric
    // NAT; a TURN server is what covers the rest, and there is no honest way
    // to provide one by default.
    $stun = env("ONE_TAYLOR_STUN");
    if ($stun == null || $stun == "") { $stun = "stun:stun.l.google.com:19302"; }
    $ice = [{ "urls": [$stun] }];
    $turn = env("ONE_TAYLOR_TURN");
    if ($turn != null && $turn != "") {
        $entry = { "urls": [$turn] };
        $tu = env("ONE_TAYLOR_TURN_USER");
        $tp = env("ONE_TAYLOR_TURN_PASS");
        if ($tu != null && $tu != "") { $entry["username"] = $tu; }
        if ($tp != null && $tp != "") { $entry["credential"] = $tp; }
        append($ice, $entry);
    }
    $wsUrl = env("ONE_TAYLOR_WS_URL");
    if ($wsUrl == null || $wsUrl == "") {
        $wsUrl = "ws://127.0.0.1:" + str(cfg.CFG["port"]) + "/";
    }
    return h.okJson($res, {
        "wsUrl": $wsUrl,
        "iceServers": $ice,
        "hasTurn": $turn != null && $turn != "",
        "me": privacy.publicUser($u, $u)
    });
}

def listHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    return h.okJson($res, {
        "conversations": chat.listFor($u, h.pageLimit($req, 40, 100))
    });
}

def openHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $other = _other($req, $res, $u);
    if ($other == null) { return null; }
    $conv = chat.findOrOpen($u, $other);
    if ($conv == null) { return h.fail($res, 403, "could not open that conversation"); }
    return h.okJson($res, {
        "conversationId": $conv["id"],
        "with": privacy.publicUserWithProfile($other, $other["profile"], $u)
    });
}

def historyHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $other = _other($req, $res, $u);
    if ($other == null) { return null; }
    $conv = chat.findOrOpen($u, $other);
    if ($conv == null) { return h.fail($res, 403, "could not open that conversation"); }
    $before = num($req.query["before"], null);
    $messages = chat.history($u, $conv, h.pageLimit($req, 40, 100), $before);
    $next = null;
    if (len($messages) > 0) { $next = $messages[0]["id"]; }
    return h.okJson($res, {
        "conversationId": $conv["id"],
        "with": privacy.publicUserWithProfile($other, $other["profile"], $u),
        "messages": $messages,
        "nextBefore": $next
    });
}

def sendHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $other = _other($req, $res, $u);
    if ($other == null) { return null; }
    $conv = chat.findOrOpen($u, $other);
    if ($conv == null) { return h.fail($res, 403, "could not open that conversation"); }
    $body = h.bodyOf($req);
    $r = chat.send($u, $conv, h.strField($body, "body", 2000));
    if (!$r["ok"]) { return h.fail($res, 400, $r["error"]); }
    $res.status(201);
    return h.okJson($res, { "message": $r["message"], "verdict": $r["verdict"],
                            "delivered": $r["delivered"], "note": $r["note"] });
}

def readHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $other = _other($req, $res, $u);
    if ($other == null) { return null; }
    $conv = chat.findOrOpen($u, $other);
    if ($conv == null) { return h.fail($res, 403, "could not open that conversation"); }
    return h.okJson($res, { "marked": chat.markRead($u, $conv) });
}

def register($sua) {
    $sua.server.post("/api/realtime/ticket", ticketHandler);
    $sua.server.get("/api/realtime/config", configHandler);
    $sua.server.get("/api/conversations", listHandler);
    $sua.server.post("/api/conversations/:username", openHandler);
    $sua.server.get("/api/conversations/:username/messages", historyHandler);
    $sua.server.post("/api/conversations/:username/messages", sendHandler);
    $sua.server.post("/api/conversations/:username/read", readHandler);
}
