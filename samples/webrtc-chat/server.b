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

// Presence is announced as a DELTA -- one small frame per event -- not as the
// whole roster.
//
// [found] This used to broadcast the full roster to every peer on every join
// and every leave, which is O(n^2) BYTES per event and O(n^3) over a room
// filling up. Measured: at 400 peers the server held 254 MB, almost all of it
// queued roster frames (635 KB per connection); at 800 peers each roster frame
// is ~27 KB and 800 joins queue ~21 MB per connection, far past the 4 MB
// per-connection backpressure cap, so sua started closing sockets to defend
// itself -- 519 of 800 peers were dropped and p99 hello->welcome hit 1.9 s.
// From the browser that is exactly "it connects, then everyone goes offline".
//
// A delta is ~50 bytes regardless of room size, so a join costs 50n instead of
// 34n^2: at 800 peers, 40 KB instead of 21 MB. The full roster is still sent
// ONCE, in the welcome, which is the only time a client has nothing to patch.
def broadcastExcept($skipId, $obj) {
    $frame = json.stringify($obj);
    each ($k in keys($S["peers"])) {
        if ($k != $skipId) { sua.ws.send($k, $frame); }
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
    // Announced only if it was ever announced: a socket that closed before
    // saying hello was never in anyone's roster, so a "leave" for it would
    // have the other clients hunting for a peer they never had.
    $wasKnown = peerName($client.id) != null;
    dropPeer($client.id);
    if ($wasKnown) {
        broadcastExcept($client.id, { "type": "leave", "id": $client.id });
    }
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
        // A second hello on the same socket is a rename, not a new peer, so
        // the others get a "join" for a peer they already have -- which their
        // roster code treats as an update. Saying hello twice cannot inflate
        // the room: the key is the socket id.
        $S["peers"][$from] = cleanName($m["name"]);
        sendTo($from, {
            "type": "welcome",
            "self": { "id": $from, "name": peerName($from) },
            "peers": rosterList()
        });
        broadcastExcept($from, {
            "type": "join",
            "peer": { "id": $from, "name": peerName($from) }
        });
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

// ─── Limits ─────────────────────────────────────────────────────────────────
// sua's defaults allow 10,000 connections and no per-address cap, which is the
// right default for a server behind a load balancer and the wrong one for this
// sample. Measured on one laptop with presence deltas in place, a peer costs
// about 60 KB (3,200 peers held 203 MB), so the room is capped where the box is still comfortable rather
// than where it starts swapping.
//
// max_connections_per_ip exists because one machine opened 300 sockets to this
// server in a test and nothing stopped it. Behind a proxy every connection
// shares the proxy's address, so a per-IP cap would throttle the whole site --
// here, where browsers connect directly, it is the difference between one
// laptop filling the room and not. The TLS front end is one such proxy, so the
// cap has to be loose enough for every device that goes through it.
$maxPeers = 1000;
if (env("MAX_PEERS") != null && env("MAX_PEERS") != "") { $maxPeers = num(env("MAX_PEERS")); }

// 256, not a tighter number, because the TLS front end is a proxy: every
// device that reaches this server through it shares the proxy's address, and a
// browser opens several connections per page on top of its WebSocket. A cap
// that only fits six phones looks like the server being down. It is still well
// under max_connections, so one address cannot take the room.
$perIp = 256;
if (env("MAX_PER_IP") != null && env("MAX_PER_IP") != "") { $perIp = num(env("MAX_PER_IP")); }

$limits = {
    "max_connections": $maxPeers,
    "max_connections_per_ip": $perIp,
    // A signalling frame is an SDP at worst: ~8 KB. 1 MB is already generous
    // and makes "fill the server's memory one frame at a time" uninteresting.
    "max_ws_frame_bytes": 262144,
    "max_ws_message_bytes": 262144
};

// Host allowlist. Left OPEN by default and warned about below, because this
// sample is reached at localhost, at a LAN address, and through the TLS front
// end on a different port -- a list that does not include the address you
// happen to be testing from looks exactly like the server being down. Set it
// the moment this is reachable from a network you do not control:
//     ALLOWED_HOSTS=localhost:8080,192.168.1.7:8443 bantu run server.b
$allowed = env("ALLOWED_HOSTS");
if ($allowed != null && $allowed != "") {
    $limits["allowed_hosts"] = split($allowed, ",");
}
sua.server.limits($limits);

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
if ($allowed == null || $allowed == "") {
    print("  Host check OFF: any Host header is accepted. Fine on a trusted");
    print("  LAN; set ALLOWED_HOSTS=host:port,... before exposing this.");
    print("");
}
print("  From a phone, plain HTTP will NOT give you a microphone:");
print("  getUserMedia needs a secure context, and a LAN IP is not one.");
print("  Run the TLS front end in a second terminal, then use its URL:");
print("      python3 tls-dev-proxy.py --target " + str($port));
print("");

sua.server.listen($port);
