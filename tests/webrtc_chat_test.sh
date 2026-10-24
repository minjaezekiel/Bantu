#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  webrtc_chat_test.sh — samples/webrtc-chat's signalling server, driven by
#  two real WebSocket clients.
#
#  The sample used to be a façade: server.b had four parse errors and had
#  never executed, the browser never built an RTCPeerConnection, and
#  "messaging" printed the sender's own text into the sender's own log. So
#  this asserts the server's whole contract rather than that it starts.
#
#  What this CAN cover is everything the server is responsible for: the
#  roster, chat fan-out, direct messages, the relay of offer/answer/ice/
#  decline/bye, that `from` cannot be forged, that an unknown recipient is
#  reported rather than dropped, and that a peer who leaves actually leaves.
#
#  What it CANNOT cover is the browser half -- getUserMedia, ICE, and
#  whether audio is audible. samples/webrtc-chat/README.md carries the
#  manual two-browser and phone scripts for that, with the expected
#  observation at each step.
#
#  Run:  BANTU=./bantu-src/compiler/build/bantu bash tests/webrtc_chat_test.sh
# ════════════════════════════════════════════════════════════════════════
set -u
BANTU="${BANTU:-bantu}"
case "$BANTU" in
    /*) ;;
    */*) BANTU="$(cd "$(dirname "$BANTU")" && pwd)/$(basename "$BANTU")" ;;
esac
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-39971}"
SAMPLE="$ROOT/samples/webrtc-chat"
TMP="$(mktemp -d)"
cleanup() { [ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null; rm -rf "$TMP"; }
trap cleanup EXIT

[ -f "$SAMPLE/server.b" ] || { echo "  FAIL  no $SAMPLE/server.b"; exit 1; }

# A server already on this port would make every assertion below run against
# somebody else's process -- which is how a stale server from an earlier run
# silently tested an OLD binary and reported a failure that had already been
# fixed. Refuse instead of guessing.
if curl -fsS -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null; then
    echo "  FAIL  something is already listening on port $PORT — stop it, or set PORT=..."
    exit 1
fi

# Run from the sample's own directory, which is what a reader does.
( cd "$SAMPLE" && PORT="$PORT" "$BANTU" run server.b ) > "$TMP/server.log" 2>&1 &
SRV=$!
for _ in $(seq 1 60); do
    curl -fsS -o /dev/null "http://127.0.0.1:$PORT/health" 2>/dev/null && break
    sleep 0.2
done
if ! curl -fsS -o /dev/null "http://127.0.0.1:$PORT/health" 2>/dev/null; then
    echo "  FAIL  server did not start"
    sed 's/^/          /' "$TMP/server.log"
    exit 1
fi

PY="$(command -v python3 || command -v python || true)"
[ -n "$PY" ] || { echo "  skip  no python3 — cannot drive a WebSocket client"; exit 0; }

PORT="$PORT" SAMPLE="$SAMPLE" "$PY" - <<'PYEOF'
import base64, hashlib, json, os, socket, struct, sys, time, urllib.request

PORT = int(os.environ["PORT"])
SAMPLE = os.environ["SAMPLE"]
ORIGIN = "http://127.0.0.1:%d" % PORT

PASS = FAIL = 0
def ok(n):  globals().__setitem__("PASS", PASS + 1); print("  ok    %s" % n)
def bad(n, detail=""):
    globals().__setitem__("FAIL", FAIL + 1)
    print("  FAIL  %s" % n)
    if detail: print("          %s" % detail)
def check(name, got, want):
    if got == want: ok(name)
    else: bad(name, "got %r want %r" % (got, want))
def truthy(name, cond, detail=""):
    if cond: ok(name)
    else: bad(name, detail)


class WS:
    """Minimal RFC 6455 client. Validates Sec-WebSocket-Accept, because a
    server that gets it wrong is one no browser can talk to."""
    # RFC 6455 1.3, cross-checked against the `websockets` library's constant.
    GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

    def __init__(self, path="/ws", timeout=6):
        self.sock = socket.create_connection(("127.0.0.1", PORT), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((
            "GET %s HTTP/1.1\r\nHost: 127.0.0.1:%d\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            "Sec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n"
            "Origin: %s\r\n\r\n" % (path, PORT, key, ORIGIN)).encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise RuntimeError("closed during handshake")
            buf += chunk
        head, _, rest = buf.partition(b"\r\n\r\n")
        self.status = head.split(b"\r\n")[0].decode()
        if "101" not in self.status:
            raise RuntimeError("upgrade refused: %s" % self.status)
        want = base64.b64encode(hashlib.sha1((key + self.GUID).encode()).digest()).decode()
        if want.encode() not in head:
            raise RuntimeError("Sec-WebSocket-Accept is not SHA1(key+GUID)")
        self.buf = rest

    def send(self, obj):
        data = json.dumps(obj).encode()
        mask = os.urandom(4)
        n = len(data)
        if n < 126:    hdr = struct.pack("!BB", 0x81, 0x80 | n)
        elif n < 65536: hdr = struct.pack("!BBH", 0x81, 0x80 | 126, n)
        else:          hdr = struct.pack("!BBQ", 0x81, 0x80 | 127, n)
        self.sock.sendall(hdr + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def _fill(self, n):
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise RuntimeError("closed")
            self.buf += chunk

    def recv(self, timeout=6):
        end = time.time() + timeout
        while True:
            self.sock.settimeout(max(0.05, end - time.time()))
            self._fill(2)
            op = self.buf[0] & 0x0F
            ln = self.buf[1] & 0x7F
            off = 2
            if ln == 126:
                self._fill(4); ln = struct.unpack("!H", self.buf[2:4])[0]; off = 4
            elif ln == 127:
                self._fill(10); ln = struct.unpack("!Q", self.buf[2:10])[0]; off = 10
            self._fill(off + ln)
            payload = self.buf[off:off + ln]
            self.buf = self.buf[off + ln:]
            if op == 0x1:
                return json.loads(payload.decode())
            if op == 0x8:
                raise RuntimeError("server closed")

    def expect(self, kind, timeout=6):
        """The next frame of a kind, skipping others: roster broadcasts
        interleave with everything and are not what a given assertion is
        about."""
        end = time.time() + timeout
        seen = []
        while time.time() < end:
            f = self.recv(timeout=max(0.1, end - time.time()))
            seen.append(f.get("type"))
            if f.get("type") == kind:
                return f
        raise AssertionError("never saw %s; saw %s" % (kind, seen))

    def close(self):
        try: self.sock.sendall(struct.pack("!BB", 0x88, 0x80) + os.urandom(4))
        except OSError: pass
        try: self.sock.close()
        except OSError: pass


def health():
    with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=5) as r:
        return json.load(r)


print("-- the page is served, from the sample's own directory --")
for name, needle in (("index.html", b"Bantu"), ("app.js", b"RTCPeerConnection"), ("styles.css", b"--accent")):
    url = "http://127.0.0.1:%d/%s" % (PORT, "" if name == "index.html" else name)
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            body = r.read()
        truthy("%s is served" % name, r.status == 200 and needle in body,
               "status=%s len=%d" % (r.status, len(body)))
    except Exception as exc:
        bad("%s is served" % name, str(exc))

print("\n-- health --")
h = health()
check("health reports the service", h.get("service"), "webrtc-chat")
check("nobody online to begin with", h.get("online"), 0)

print("\n-- hello / welcome / roster --")
a = WS(); a.send({"type": "hello", "name": "alice"})
wa = a.expect("welcome"); A = wa["self"]["id"]
check("welcome names the caller", wa["self"]["name"], "alice")
truthy("welcome assigns an id", bool(A), repr(A))

b = WS(); b.send({"type": "hello", "name": "bob"})
wb = b.expect("welcome"); B = wb["self"]["id"]
check("a joiner sees who is already here",
      sorted(p["name"] for p in wb["peers"]), ["alice", "bob"])
def roster_with(ws, n, timeout=6):
    """The next roster frame listing n peers.

    A roster is broadcast on every presence change, so alice's socket already
    holds the one from her OWN hello (just herself) before the one that
    followed bob's. Waiting for the size wanted is the difference between
    asserting on presence and asserting on arrival order."""
    end = time.time() + timeout
    last = None
    while time.time() < end:
        last = ws.expect("roster", timeout=max(0.2, end - time.time()))
        if len(last["peers"]) == n:
            return last
    raise AssertionError("no roster of %d peers; last was %r" % (n, last))

ra = roster_with(a, 2)
check("everyone is told when someone joins",
      sorted(p["name"] for p in ra["peers"]), ["alice", "bob"])

print("\n-- chat --")
a.send({"type": "chat", "text": "hello room"})
f = b.expect("chat")
check("a room message reaches the other peer", f["text"], "hello room")
check("and carries the sender's name", f["fromName"], "alice")
truthy("and is not marked direct", not f.get("direct"))
# A room message is fanned out to everyone including its sender, so alice has
# her own copy queued. Consumed here so the next assertion reads the direct
# message's echo rather than this one.
own = a.expect("chat")
check("the sender also receives their own room message", own["text"], "hello room")

a.send({"type": "chat", "text": "psst", "to": B})
f = b.expect("chat")
check("a direct message reaches its recipient", f["text"], "psst")
truthy("and is marked direct", bool(f.get("direct")))
f = a.expect("chat")
check("the sender sees their own direct message", f["text"], "psst")

# `from` is stamped from the socket the frame arrived on. Without that, any
# peer could post as any other -- in a chat, and into someone else's call.
a.send({"type": "chat", "text": "I am bob", "from": B, "fromName": "bob"})
f = b.expect("chat")
check("a forged 'from' is overwritten with the real sender", f["from"], A)
check("and so is the forged name", f["fromName"], "alice")

print("\n-- call signalling is relayed verbatim --")
a.send({"type": "offer", "to": B, "media": "video",
        "sdp": {"type": "offer", "sdp": "v=0\r\no=- TEST"}})
o = b.expect("offer")
check("the offer arrives", o["from"], A)
check("the media kind survives", o["media"], "video")
check("the SDP survives byte for byte", o["sdp"]["sdp"], "v=0\r\no=- TEST")

# Trickle ICE: the caller sends candidates before the callee has answered.
a.send({"type": "ice", "to": B, "candidate": {"candidate": "candidate:1 1 udp 2 10.0.0.1 1 typ host"}})
i = b.expect("ice")
check("an ICE candidate is relayed before any answer",
      i["candidate"]["candidate"], "candidate:1 1 udp 2 10.0.0.1 1 typ host")

b.send({"type": "answer", "to": A, "sdp": {"type": "answer", "sdp": "v=0\r\no=- ANS"}})
an = a.expect("answer")
check("the answer comes back", an["sdp"]["sdp"], "v=0\r\no=- ANS")

b.send({"type": "decline", "to": A, "reason": "busy"})
check("a decline carries its reason", a.expect("decline")["reason"], "busy")

b.send({"type": "bye", "to": A})
check("bye is relayed", a.expect("bye")["from"], B)

print("\n-- refusals are reported, never silent --")
# Silence here is indistinguishable from a network fault, and is the single
# most confusing thing to debug in a signalling server.
a.send({"type": "offer", "to": "ws-nonexistent", "sdp": {}})
truthy("an unknown recipient is reported",
       "not connected" in a.expect("error")["message"])
a.send({"type": "offer", "sdp": {}})
truthy("a missing 'to' is reported", "'to'" in a.expect("error")["message"])
a.send({"type": "chat"})
truthy("chat with no text is reported", "text" in a.expect("error")["message"])
a.send({"type": "nonsense"})
truthy("an unknown frame type is reported",
       "unknown frame type" in a.expect("error")["message"])

c = WS(); c.send({"type": "chat", "text": "too early"})
truthy("a frame before hello is reported", "hello" in c.expect("error")["message"])
c.close()

print("\n-- names are bounded and stripped --")
d = WS(); d.send({"type": "hello", "name": "x" * 200})
check("an over-long name is cut to 24", len(d.expect("welcome")["self"]["name"]), 24)
e = WS(); e.send({"type": "hello", "name": None})
check("a null name becomes a placeholder", e.expect("welcome")["self"]["name"], "guest")
g = WS(); g.send({"type": "hello", "name": "ok\u0007\u0001name"})
check("control characters are stripped", g.expect("welcome")["self"]["name"], "okname")
for x in (d, e, g):
    x.close()

print("\n-- leaving --")
# [found] sua used to leave a closed WebSocket in its table whenever frames
# the client never read were still queued: on("disconnect") did not fire, and
# the peer stayed in every roster until the 5-minute idle timeout. With two
# browsers that is a roster full of ghosts of yourself.
time.sleep(0.6)
b.close()
deadline = time.time() + 6
names = None
while time.time() < deadline:
    names = sorted(p["name"] for p in health()["peers"])
    if "bob" not in names:
        break
    time.sleep(0.2)
check("a peer that disconnects is removed", names, ["alice"])

a.close()
deadline = time.time() + 6
while time.time() < deadline:
    if health()["online"] == 0:
        break
    time.sleep(0.2)
check("the last peer out leaves nobody behind", health()["online"], 0)

print("")
print("  %d passed, %d failed" % (PASS, FAIL))
sys.exit(1 if FAIL else 0)
PYEOF
RC=$?
if [ "$RC" -ne 0 ]; then
    echo "  ── server log ──"
    sed 's/^/          /' "$TMP/server.log" | tail -25
fi
exit "$RC"
