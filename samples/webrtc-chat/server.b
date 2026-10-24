// ============================================================================
//  webrtc-chat/server.b — signalling relay for browser-to-browser calls
//
//  The server is NOT a WebRTC peer. It holds no RTCPeerConnection, terminates
//  no media and never sees a frame of audio or video: it relays SDP and ICE
//  between two browsers and then gets out of the way. `sua.webrtc.*` is
//  deliberately unused here -- that is a server-side peer API, and in a
//  browser-to-browser call the peers are the browsers.
//
//  Everything moves over one WebSocket, because signalling has to PUSH: the
//  callee learns a call is incoming because the server sends it a frame
//  unprompted, which no POST endpoint can do. Presence comes free with it.
//
//  Protocol and rationale: docs/webrtc-chat-design.md
//
//  Run:  bantu run server.b          then open http://localhost:8080
// ============================================================================

// ─── Peer registry ──────────────────────────────────────────────────────────
// clientId -> display name, held as a FIELD of a global container rather than
// as a global of its own.
//
// [found] The obvious `$peers = {}` plus `$peers = $next;` inside a function
// silently does nothing. Rebinding a global name from inside a function body
// creates a local that is thrown away at the end of the call, so a peer that
// disconnected was never removed: it stayed in every roster, "online" never
// went back down, and a second browser saw a ghost of the first one. Writing
// a FIELD of a reference-typed container -- $S["peers"] = ... -- is an
// index-set on the container, which does propagate. Verified both ways
// before relying on it.
$S = { "peers": {} };

def peerName($id) { return $S["peers"][$id]; }

// Bantu has no key-removal builtin (`delete` is a reserved word -- it is
// sua.server.delete), so a key is dropped by rebuilding without it and
// replacing the field. The registry is one entry per connected browser.
def dropPeer($id) {
    $next = {};
    each ($k in keys($S["peers"])) {
        if ($k != $id) { $next[$k] = $S["peers"][$k]; }
    }
    $S["peers"] = $next;
}

def peerCount() { return len(keys($S["peers"])); }

// A display name is rendered in every other browser, so it is bounded and
// stripped of control characters here rather than trusted. The clients render
// with textContent too; this is the half an attacker cannot skip.

// Drop C0 controls and DEL. Compared by ord(), not by string: `$ch >= " "`
// used to answer true for EVERY character, because two strings were compared
// as numbers and both read as 0. That is fixed in the interpreter now, but
// ord() says exactly what is meant and cannot be read two ways.
def stripControl($s) {
    $out = "";
    $i = 0;
    while ($i < len($s)) {
        $b = ord(substr($s, $i, 1));
        if ($b >= 32 && $b != 127) { $out = $out + substr($s, $i, 1); }
        $i = $i + 1;
    }
    return $out;
}

// Cut to at most $maxChars CHARACTERS without splitting a UTF-8 sequence.
//
// len() and substr() count bytes, so a plain substr($s, 0, 24) can cut an
// emoji or an accented letter in half. The result is invalid UTF-8, it goes
// straight into a WebSocket text frame, and a browser that reads one is
// required to fail the connection -- so one badly-placed byte would drop
// every peer in the room, not just the one with the long name. A byte in
// 0x80..0xBF continues the character before it, so it travels with that
// character instead of counting as one of its own.
def truncChars($s, $maxChars) {
    $out = "";
    $chars = 0;
    $i = 0;
    while ($i < len($s)) {
        $b = ord(substr($s, $i, 1));
        if ($b < 128 || $b >= 192) {
            if ($chars >= $maxChars) { return $out; }
            $chars = $chars + 1;
        }
        $out = $out + substr($s, $i, 1);
        $i = $i + 1;
    }
    return $out;
}

def cleanName($raw) {
    if ($raw == null) { return "guest"; }
    $s = trim(stripControl(str($raw)));
    if (len($s) == 0) { return "guest"; }
    return truncChars($s, 24);
}

def rosterList() {
    $list = [];
    each ($k in keys($S["peers"])) {
        push($list, { "id": $k, "name": $S["peers"][$k] });
    }
    return $list;
}

def sendTo($id, $obj) {
    sua.ws.send($id, json.stringify($obj));
}

def broadcastRoster() {
    $frame = json.stringify({ "type": "roster", "peers": rosterList() });
    each ($k in keys($S["peers"])) {
        sua.ws.send($k, $frame);
    }
}

// ─── Signalling ─────────────────────────────────────────────────────────────

sua.ws.on("connect", def($client) {
    // Registered, but unnamed and not yet announced: a peer appears in other
    // people's rosters only once it has said hello, so the list never shows a
    // nameless entry for a socket that is still opening.
    print("[ws] open " + $client.id);
});

sua.ws.on("disconnect", def($client) {
    dropPeer($client.id);
    broadcastRoster();
    print("[ws] close " + $client.id + " (" + str(peerCount()) + " online)");
});

sua.ws.on("message", def($msg) {
    $m = $msg.json;
    if ($m == null) {
        sendTo($msg.client, { "type": "error", "message": "frames must be JSON objects" });
        return null;
    }
    $type = $m["type"];
    $from = $msg.client;

    // hello -> welcome. The id is the socket's, assigned by sua.
    if ($type == "hello") {
        $S["peers"][$from] = cleanName($m["name"]);
        sendTo($from, {
            "type": "welcome",
            "self": { "id": $from, "name": peerName($from) },
            "peers": rosterList()
        });
        broadcastRoster();
        print("[ws] hello " + peerName($from) + " as " + $from);
        return null;
    }

    // Heartbeat. sua reaps a WebSocket after idle_timeout_ms (5 minutes by
    // default), and browser JavaScript cannot send a protocol-level PING --
    // the WebSocket API exposes no way to. Without an application-level
    // keepalive, a chat connection that nobody types into is guaranteed to be
    // dropped and then reconnected, which is exactly the online/offline churn
    // it looks like. Answered before the hello check so it costs nothing.
    if ($type == "ping") {
        sendTo($from, { "type": "pong" });
        return null;
    }

    // Everything past this point requires a registered peer, so a frame that
    // arrives before hello is reported rather than acted on.
    if (peerName($from) == null) {
        sendTo($from, { "type": "error", "message": "send hello before anything else" });
        return null;
    }

    // Room-wide or direct chat. Text is relayed verbatim; the receiving
    // browsers render it as text, never as markup.
    if ($type == "chat") {
        $text = $m["text"];
        if ($text == null || len(str($text)) == 0) {
            sendTo($from, { "type": "error", "message": "chat needs text" });
            return null;
        }
        // `from` is stamped from the socket the frame arrived on, never taken
        // from the frame. Otherwise any peer could post as any other.
        // No server timestamp: Bantu has no now()/time builtin (calling one
        // raised "Undefined variable: now" inside the handler, which sua
        // reported and then dropped the frame -- so chat silently did not
        // work at all). The receiving browser stamps its own clock, which is
        // the time a reader actually wants and needs no clock agreement.
        $out = {
            "type": "chat",
            "from": $from,
            "fromName": peerName($from),
            "text": str($text)
        };
        $to = $m["to"];
        if ($to != null && len(str($to)) > 0) {
            $out["direct"] = true;
            if (peerName(str($to)) == null) {
                sendTo($from, { "type": "error", "message": "no such peer: " + str($to) });
                return null;
            }
            sendTo(str($to), $out);
            sendTo($from, $out);          // the sender sees their own message once
            return null;
        }
        $frame = json.stringify($out);
        each ($k in keys($S["peers"])) {
            sua.ws.send($k, $frame);
        }
        return null;
    }

    // offer / answer / ice / decline / bye are relayed to one named peer and
    // are otherwise opaque to the server: it does not model a call at all.
    // Call state lives in the two browsers, which are the only parties that
    // know it.
    if ($type == "offer" || $type == "answer" || $type == "ice"
        || $type == "decline" || $type == "bye") {
        $to = $m["to"];
        if ($to == null) {
            sendTo($from, { "type": "error", "message": $type + " needs a 'to'" });
            return null;
        }
        $to = str($to);
        if (peerName($to) == null) {
            // Silence here is indistinguishable from a network fault and is
            // the most confusing thing to debug in a signalling server.
            sendTo($from, { "type": "error", "message": "peer is not connected: " + $to });
            return null;
        }
        $m["from"] = $from;
        $m["fromName"] = peerName($from);
        sendTo($to, $m);
        return null;
    }

    sendTo($from, { "type": "error", "message": "unknown frame type: " + str($type) });
});

// ─── HTTP ───────────────────────────────────────────────────────────────────

sua.server.get("/health", def($req, $res) {
    $res.json({
        "ok": true,
        "service": "webrtc-chat",
        "online": peerCount(),
        "peers": rosterList()
    });
});

// Two static roots so the sample runs from its own directory AND from the
// repo root -- roots are searched in order, and a path that only resolves
// from one working directory is how samples rot.
sua.server.static("./public");
sua.server.static("./samples/webrtc-chat/public");

$port = 8080;
if (env("PORT") != null && env("PORT") != "") { $port = num(env("PORT")); }

print("");
print("  Bantu WebRTC chat — signalling relay");
print("  ───────────────────────────────────────────────");
print("  On this machine : http://localhost:" + str($port));
print("");
print("  From a phone, plain HTTP will NOT give you a microphone:");
print("  getUserMedia needs a secure context, and a LAN IP is not one.");
print("  Run the TLS front end in a second terminal, then use its URL:");
print("      python3 tls-dev-proxy.py --target " + str($port));
print("");

sua.server.listen($port);
