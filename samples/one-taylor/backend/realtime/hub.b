// ============================================================================
// one-taylor/backend/realtime/hub.b — one WebSocket, three jobs
//
// Presence, chat delivery and WebRTC call signalling all ride the same
// `sua.ws` socket, multiplexed by a frame `type`. One socket per tab rather
// than three is not tidiness: each extra socket costs a slot against
// max_connections, another TLS handshake, and another thing to reconnect.
//
// THE SERVER IS NOT A WEBRTC PEER. It holds no RTCPeerConnection, terminates
// no media and never sees a frame of audio or video. It relays offers,
// answers and ICE candidates between two browsers and gets out of the way —
// which is also why a call costs the server almost nothing.
//
// What it DOES do is check, on every single relayed frame, that the two
// parties are still allowed to talk to each other. A signalling relay that
// trusts the `to` field in a frame is a relay that will connect a 40-year-old
// to a 14-year-old the moment somebody edits a number in devtools.
//
// Frames in:
//   {type:"hello", ticket}                      authenticate this socket
//   {type:"ping"}                               keepalive
//   {type:"message", to, body}                  send a direct message
//   {type:"read", to}                           mark a thread read
//   {type:"typing", to, on}                     typing indicator
//   {type:"call-offer", to, sdp, video}         start a call
//   {type:"call-answer", to, sdp}               accept
//   {type:"call-ice", to, candidate}            trickle ICE
//   {type:"call-reject", to} / {type:"call-end", to}
//
// Frames out:
//   welcome · presence · message · message-sent · read · typing
//   call-ringing · call-answer · call-ice · call-reject · call-end · error
//
// Exposes: install() — registers the sua.ws handlers
// ============================================================================

include "orm" as orm;
include "../models.b" as models;
include "../safety/age.b" as age;
include "../auth/privacy.b" as privacy;
include "./tickets.b" as tickets;
include "./chat.b" as chat;

// Socket state. A dict, so the handlers below can mutate it in place: a list
// or a scalar assigned to a global name inside a function would write a local
// and the change would silently vanish — which in a disconnect handler means
// a peer who never goes offline.
$S = {
    "bySocket": {},      // socket id -> {userId, username}
    "sockets": {},       // user id (as string) -> dict of socket id -> true
    "calls": {}          // user id -> the user id they are in a call with
};

def _send($socketId, $obj) {
    sua.ws.send($socketId, json.stringify($obj));
}

def _error($socketId, $message) {
    _send($socketId, { "type": "error", "message": $message });
}

def _socketsOf($userId) {
    $m = $S["sockets"][str($userId)];
    if ($m == null) { return []; }
    return keys($m);
}

def isOnline($userId) { return len(_socketsOf($userId)) > 0; }

// Every tab the user has open gets the frame. A message that arrives in one
// tab and not another is the kind of bug people report as "it lost my
// messages".
def _toUser($userId, $obj) {
    $frame = json.stringify($obj);
    $n = 0;
    each ($sid in _socketsOf($userId)) {
        sua.ws.send($sid, $frame);
        $n = $n + 1;
    }
    return $n;
}

// Every socket this user has open EXCEPT the one that acted.
//
// A call rings in all of someone's tabs, which is what you want -- they may
// not be looking at the one that happens to hold the thread. But once one tab
// answers or declines, the others have to stop ringing, or the user is left
// dismissing a call that is already connected somewhere else.
def _toOtherSockets($userId, $exceptSocket, $obj) {
    $frame = json.stringify($obj);
    each ($sid in _socketsOf($userId)) {
        if ($sid != $exceptSocket) { sua.ws.send($sid, $frame); }
    }
    return null;
}

def _register($socketId, $userId, $username) {
    $S["bySocket"][$socketId] = { "userId": $userId, "username": $username };
    $key = str($userId);
    if ($S["sockets"][$key] == null) { $S["sockets"][$key] = {}; }
    $set = $S["sockets"][$key];
    $set[$socketId] = true;
    $S["sockets"][$key] = $set;
}

// There is no key-removal builtin in Bantu, so a dict is "rebuilt without the
// key and reassigned to its field" — which does propagate, because that is an
// index-set on a container rather than a rebinding of a global.
def _forget($socketId) {
    $who = $S["bySocket"][$socketId];
    if ($who == null) { return null; }
    $nextBySocket = {};
    each ($k in keys($S["bySocket"])) {
        if ($k != $socketId) { $nextBySocket[$k] = $S["bySocket"][$k]; }
    }
    $S["bySocket"] = $nextBySocket;

    $key = str($who["userId"]);
    $set = $S["sockets"][$key];
    if ($set != null) {
        $next = {};
        each ($k in keys($set)) {
            if ($k != $socketId) { $next[$k] = true; }
        }
        $S["sockets"][$key] = $next;
    }
    return $who;
}

def _clearCall($userId) {
    $next = {};
    each ($k in keys($S["calls"])) {
        if ($k != str($userId)) { $next[$k] = $S["calls"][$k]; }
    }
    $S["calls"] = $next;
}

def _inCall($userId) { return $S["calls"][str($userId)]; }

// ── who is this socket, and may it talk to that user? ───────────────────────
def _me($socketId) {
    $who = $S["bySocket"][$socketId];
    if ($who == null) { return null; }
    return chat.userRow($who["userId"]);
}

// Resolves `to` and re-checks the gate. Called on EVERY frame that names
// another user, including each ICE candidate: the check is cheap and the cost
// of skipping it once is a connected call across the age wall.
def _peerFor($socketId, $m) {
    $me = _me($socketId);
    if ($me == null) {
        _error($socketId, "say hello first");
        return null;
    }
    $toName = $m["to"];
    if ($toName == null || $toName == "") {
        _error($socketId, "that frame needs a 'to'");
        return null;
    }
    $other = orm.first(orm.filter(orm.query(models.M["users"]),
                       [["username_lower", str($toName).lower()]]));
    if ($other != null) { $other = chat.userRow($other["id"]); }
    $gate = chat.mayMessage($me, $other);
    if (!$gate["ok"]) {
        _error($socketId, $gate["reason"]);
        return null;
    }
    $conv = chat.findOrOpen($me, $other);
    if ($conv == null) {
        _error($socketId, "could not open that conversation");
        return null;
    }
    return { "me": $me, "other": $other, "conv": $conv };
}

// ── handlers ────────────────────────────────────────────────────────────────
def _onHello($socketId, $m) {
    $userId = tickets.redeem($m["ticket"]);
    if ($userId == null) {
        _error($socketId, "that ticket is not valid — reload the page");
        return null;
    }
    $u = chat.userRow($userId);
    if ($u == null || $u["status"] != "active") {
        _error($socketId, "that account cannot connect");
        return null;
    }
    // A band transition revokes sessions, and it must revoke sockets too:
    // recomputed here rather than trusted from the ticket.
    if (age.bandFor($u["dob"]) != $u["band"]) {
        _error($socketId, "your account has changed age band — sign in again");
        return null;
    }

    _register($socketId, $u["id"], $u["username"]);
    _send($socketId, {
        "type": "welcome",
        "self": privacy.publicUser($u, $u),
        "conversations": chat.listFor($u, 50)
    });

    // Presence goes only to the people this user can actually talk to. A
    // global presence broadcast would tell every account in the band that this
    // person is online, which is both a privacy leak and O(n) frames per
    // connect.
    each ($c in chat.listFor($u, 200)) {
        _toUser($c["with"]["id"],
                { "type": "presence", "username": $u["username"], "online": true });
    }
    return null;
}

def _onMessage($socketId, $m) {
    $p = _peerFor($socketId, $m);
    if ($p == null) { return null; }
    $r = chat.send($p["me"], $p["conv"], $m["body"]);
    if (!$r["ok"]) {
        _error($socketId, $r["error"]);
        return null;
    }
    // The sender always gets their own message back, whatever the verdict —
    // including a blocked one, with the note. Someone whose message silently
    // fails to appear assumes the app is broken and sends it again.
    _send($socketId, { "type": "message-sent", "with": $p["other"]["username"],
                       "message": $r["message"], "verdict": $r["verdict"],
                       "note": $r["note"] });
    if ($r["delivered"]) {
        $copy = {};
        for $k, $v in $r["message"].items() { $copy[$k] = $v; }
        $copy["mine"] = false;
        _toUser($p["other"]["id"],
                { "type": "message", "with": $p["me"]["username"],
                  "from": privacy.publicUser($p["me"], null), "message": $copy });
    }
    return null;
}

def _onRead($socketId, $m) {
    $p = _peerFor($socketId, $m);
    if ($p == null) { return null; }
    $n = chat.markRead($p["me"], $p["conv"]);
    if ($n > 0) {
        _toUser($p["other"]["id"],
                { "type": "read", "with": $p["me"]["username"], "count": $n });
    }
    return null;
}

def _onTyping($socketId, $m) {
    $p = _peerFor($socketId, $m);
    if ($p == null) { return null; }
    _toUser($p["other"]["id"], { "type": "typing", "with": $p["me"]["username"],
                                 "on": $m["on"] == true });
    return null;
}

// ── calls ───────────────────────────────────────────────────────────────────
// The server relays SDP and ICE. It does not parse the SDP, and it does not
// need to: a peer that lies about its own media description only breaks its
// own call. What the server guards is WHO is connected to WHOM.
def _onCallOffer($socketId, $m) {
    $p = _peerFor($socketId, $m);
    if ($p == null) { return null; }
    if (!isOnline($p["other"]["id"])) {
        _send($socketId, { "type": "call-reject", "with": $p["other"]["username"],
                           "reason": "offline" });
        chat.recordCall($p["conv"], $p["me"]["id"], "Called — no answer");
        return null;
    }
    $busyWith = _inCall($p["other"]["id"]);
    if ($busyWith != null) {
        _send($socketId, { "type": "call-reject", "with": $p["other"]["username"],
                           "reason": "busy" });
        return null;
    }
    $S["calls"][str($p["me"]["id"])] = $p["other"]["id"];
    $S["calls"][str($p["other"]["id"])] = $p["me"]["id"];

    $video = $m["video"] == true;
    _toUser($p["other"]["id"], {
        "type": "call-ringing",
        "with": $p["me"]["username"],
        "from": privacy.publicUser($p["me"], null),
        "video": $video,
        "sdp": $m["sdp"]
    });
    return null;
}

def _onCallAnswer($socketId, $m) {
    $p = _peerFor($socketId, $m);
    if ($p == null) { return null; }
    // The answer must belong to a call this server set up. Without this check
    // an answer frame is a way to inject an SDP into somebody else's call.
    if (_inCall($p["me"]["id"]) != $p["other"]["id"]) {
        _error($socketId, "there is no call to answer");
        return null;
    }
    _toUser($p["other"]["id"], { "type": "call-answer",
                                 "with": $p["me"]["username"], "sdp": $m["sdp"] });
    // This tab took the call; the user's other tabs stop ringing.
    _toOtherSockets($p["me"]["id"], $socketId,
                    { "type": "call-end", "with": $p["other"]["username"] });
    return null;
}

def _onCallIce($socketId, $m) {
    $p = _peerFor($socketId, $m);
    if ($p == null) { return null; }
    if (_inCall($p["me"]["id"]) != $p["other"]["id"]) { return null; }
    _toUser($p["other"]["id"], { "type": "call-ice",
                                 "with": $p["me"]["username"],
                                 "candidate": $m["candidate"] });
    return null;
}

def _endCall($p, $socketId, $kind, $note) {
    _clearCall($p["me"]["id"]);
    _clearCall($p["other"]["id"]);
    _toUser($p["other"]["id"], { "type": $kind, "with": $p["me"]["username"] });
    // And this user's other tabs, so a ringing one closes.
    _toOtherSockets($p["me"]["id"], $socketId,
                    { "type": "call-end", "with": $p["other"]["username"] });
    if ($note != null) { chat.recordCall($p["conv"], $p["me"]["id"], $note); }
    return null;
}

def _onCallReject($socketId, $m) {
    $p = _peerFor($socketId, $m);
    if ($p == null) { return null; }
    return _endCall($p, $socketId, "call-reject", "Call declined");
}

def _onCallEnd($socketId, $m) {
    $p = _peerFor($socketId, $m);
    if ($p == null) { return null; }
    return _endCall($p, $socketId, "call-end", "Call ended");
}

// ── wiring ──────────────────────────────────────────────────────────────────
def install() {
    sua.ws.on("connect", def($client) {
        // Registered by sua, but anonymous until it says hello with a ticket.
        // Nothing can be sent to or from it before that.
        return null;
    });

    sua.ws.on("disconnect", def($client) {
        $who = _forget($client.id);
        if ($who == null) { return null; }
        // Only when the LAST socket for this user closes: someone with three
        // tabs open has not gone offline because they shut one.
        if (!isOnline($who["userId"])) {
            $peer = _inCall($who["userId"]);
            if ($peer != null) {
                _clearCall($who["userId"]);
                _clearCall($peer);
                _toUser($peer, { "type": "call-end", "with": $who["username"] });
            }
            $u = chat.userRow($who["userId"]);
            if ($u != null) {
                each ($c in chat.listFor($u, 200)) {
                    _toUser($c["with"]["id"], { "type": "presence",
                                                "username": $who["username"],
                                                "online": false });
                }
            }
        }
        return null;
    });

    sua.ws.on("message", def($msg) {
        $m = $msg.json;
        if ($m == null) {
            _error($msg.client, "frames must be JSON objects");
            return null;
        }
        $type = $m["type"];
        if ($type == "hello")        { return _onHello($msg.client, $m); }
        if ($type == "ping")         { _send($msg.client, {"type": "pong"}); return null; }
        if ($type == "message")      { return _onMessage($msg.client, $m); }
        if ($type == "read")         { return _onRead($msg.client, $m); }
        if ($type == "typing")       { return _onTyping($msg.client, $m); }
        if ($type == "call-offer")   { return _onCallOffer($msg.client, $m); }
        if ($type == "call-answer")  { return _onCallAnswer($msg.client, $m); }
        if ($type == "call-ice")     { return _onCallIce($msg.client, $m); }
        if ($type == "call-reject")  { return _onCallReject($msg.client, $m); }
        if ($type == "call-end")     { return _onCallEnd($msg.client, $m); }
        _error($msg.client, "unknown frame type: " + str($type));
        return null;
    });
}
