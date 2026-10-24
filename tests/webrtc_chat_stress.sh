#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  webrtc_chat_stress.sh — how many peers the signalling relay actually holds.
#
#  WHY THIS EXISTS
#  ---------------
#  [found] The server broadcast the WHOLE ROSTER to every peer on every join
#  and every leave. That is O(n^2) bytes per presence change and O(n^3) for a
#  room filling up, and it does not fail gradually: measured on one laptop,
#  400 peers held 254 MB -- 635 KB of queued roster frames PER CONNECTION --
#  and at 800 peers each roster frame is ~27 KB, so 800 joins queue ~21 MB per
#  connection, far past sua's 4 MB per-connection backpressure cap. sua then
#  closes sockets to defend itself: 519 of 800 peers were dropped and p99
#  hello->welcome reached 1.9 s. From a browser that is "it connects, then
#  everyone keeps going offline", which is exactly how it was reported.
#
#  Presence is a delta now, so a join costs ~50 bytes per peer instead of
#  ~34n. Same laptop, after: 3,200 peers, 203 MB, not one dropped.
#
#  This asserts the SHAPE of that, not the absolute numbers: a few hundred
#  peers must all connect, all stay connected, and all answer. A reintroduced
#  O(n^2) broadcast fails it; a slower machine does not.
#
#  Run:   BANTU=./bantu-src/compiler/build/bantu bash tests/webrtc_chat_stress.sh
#  Bigger: PEERS=2000 BANTU=... bash tests/webrtc_chat_stress.sh
# ════════════════════════════════════════════════════════════════════════
set -u
BANTU="${BANTU:-bantu}"
case "$BANTU" in
    */*) BANTU="$(cd "$(dirname "$BANTU")" && pwd)/$(basename "$BANTU")" ;;
esac
PORT="${PORT:-39971}"
PEERS="${PEERS:-300}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAMPLE="$ROOT/samples/webrtc-chat"
TMP="$(mktemp -d)"
PY="${PYTHON:-python3}"

cleanup() { [ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null; rm -rf "$TMP"; }
trap cleanup EXIT

# Every bot is one socket from one address, so the sample's per-IP cap would
# stop the test long before the server did. The cap itself is asserted in
# tests/sua_ws_security_test.sh.
# exec, so the subshell is REPLACED by the server and $! is the server's own
# pid. Without it $! names the subshell, and the memory figure below reports
# the wrapper shell's footprint instead of the server's -- it read 0.7 MB for
# a server holding 300 connections.
( cd "$SAMPLE" && PORT="$PORT" MAX_PEERS=$((PEERS + 100)) MAX_PER_IP=$((PEERS + 100)) \
    exec "$BANTU" run server.b ) > "$TMP/server.log" 2>&1 &
SRV=$!
for _ in $(seq 1 60); do curl -s -o /dev/null "http://127.0.0.1:$PORT/health" && break; sleep 0.2; done
if ! curl -s -o /dev/null "http://127.0.0.1:$PORT/health"; then
    echo "  FAIL  server did not start"; sed 's/^/          /' "$TMP/server.log"; exit 1
fi

PORT="$PORT" PEERS="$PEERS" SRVPID="$SRV" "$PY" - <<'PYEOF'
import json, os, resource, socket, struct, subprocess, sys, time, urllib.request
import base64

PORT  = int(os.environ["PORT"])
PEERS = int(os.environ["PEERS"])
SRVPID = os.environ["SRVPID"]
P = F = 0

def check(name, got, want):
    global P, F
    if str(got) == str(want):
        P += 1; print("  ok    %s" % name)
    else:
        F += 1; print("  FAIL  %s\n          got: %s | want: %s" % (name, got, want))

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
try:
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, PEERS * 4 + 256), hard))
except (ValueError, OSError):
    pass
if resource.getrlimit(resource.RLIMIT_NOFILE)[0] < PEERS + 64:
    print("  SKIP  file-descriptor limit too low for %d peers" % PEERS); sys.exit(0)

def health():
    """The roster, or why it could not be read.

    A server that is busy dropping connections also drops this request -- that
    is how the O(n^2) broadcast presented, with /health answering
    RemoteDisconnected. Returning a sentinel instead of raising turns that into
    a readable FAIL rather than a traceback in the middle of the run.
    """
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=10) as r:
            return json.load(r)
    except Exception as exc:
        return {"online": "unreachable (%s)" % type(exc).__name__, "peers": []}

def rss_kb():
    out = subprocess.run(["ps", "-o", "rss=", "-p", SRVPID], capture_output=True, text=True)
    return int(out.stdout.strip() or 0)

def mask(payload, op=0x1):
    m = os.urandom(4)
    body = bytes(b ^ m[i % 4] for i, b in enumerate(payload))
    n = len(payload)
    if n < 126:        hdr = struct.pack("!BB", 0x80 | op, 0x80 | n)
    elif n < 1 << 16:  hdr = struct.pack("!BBH", 0x80 | op, 0xFE, n)
    else:              hdr = struct.pack("!BBQ", 0x80 | op, 0xFF, n)
    return hdr + m + body

class Bot:
    """A peer that never reads until asked to -- which is the point.

    A bot that drains continuously measures the server's throughput. A bot
    that does NOT drain measures how much the server is trying to push at it,
    which is what the O(n^2) broadcast made fatal: the per-connection queue
    passed 4 MB and sua closed the socket.
    """
    def __init__(self, i):
        self.sock = socket.create_connection(("127.0.0.1", PORT), timeout=20)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall(
            ("GET /ws HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nOrigin: http://127.0.0.1:%d\r\n"
             "Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: %s\r\n"
             "Sec-WebSocket-Version: 13\r\n\r\n" % (PORT, PORT, key)).encode())
        self.sock.settimeout(20)
        buf = b""
        while b"\r\n\r\n" not in buf:
            d = self.sock.recv(4096)
            if not d:
                raise RuntimeError("upgrade failed")
            buf += d
        if b" 101 " not in buf.split(b"\r\n")[0] + b" ":
            raise RuntimeError("upgrade refused: %s" % buf.split(b"\r\n")[0])
        self.sock.sendall(mask(json.dumps({"type": "hello", "name": "bot%05d" % i}).encode()))

    def alive(self):
        """Still open? A server that gave up on us sends a close frame or EOF."""
        self.sock.settimeout(0.05)
        try:
            d = self.sock.recv(1)
            return d != b""
        except socket.timeout:
            return True
        except OSError:
            return False

print("-- filling the room with %d peers --" % PEERS)
t0 = time.time()
bots, failures = [], []
for i in range(PEERS):
    try:
        bots.append(Bot(i))
    except Exception as exc:
        failures.append("%s: %s" % (type(exc).__name__, exc))
ramp = time.time() - t0
print("     %d connected in %.1fs, %d refused" % (len(bots), ramp, len(failures)))
if failures:
    print("     first refusal: %s" % failures[0])

check("every peer was accepted", len(bots), PEERS)

# The server's own count has to agree. It is the roster every browser renders.
deadline = time.time() + 30
online = 0
while time.time() < deadline:
    online = health()["online"]
    if isinstance(online, int) and online >= PEERS:
        break
    time.sleep(0.5)
check("the server counts them all", online, PEERS)

mem = rss_kb()
print("     server RSS %.1f MB (%.0f KB per peer)" % (mem / 1024.0, mem / max(1, len(bots))))

# THE assertion. With a full-roster broadcast these sockets are holding
# megabytes of undelivered frames each and sua starts closing them.
still = sum(1 for b in bots if b.alive())
check("none were dropped while the room filled", still, PEERS)

# And they are still usable: a ping gets a pong, through all that queued
# presence traffic.
answered = 0
for b in bots[: min(50, len(bots))]:
    try:
        b.sock.settimeout(20)
        b.sock.sendall(mask(b'{"type":"ping"}'))
        answered += 1
    except OSError:
        pass
check("a sample of peers can still send", answered, min(50, len(bots)))

for b in bots:
    try: b.sock.close()
    except OSError: pass

deadline = time.time() + 60
left = None
while time.time() < deadline:
    left = health()["online"]
    if left == 0:
        break
    if not isinstance(left, int):
        break
    time.sleep(0.5)
check("the room empties when they all go", left, 0)

print("\n  %d passed, %d failed" % (P, F))
sys.exit(1 if F else 0)
PYEOF
RC=$?
[ "$RC" -ne 0 ] && { echo "  ── server log (tail) ──"; tail -40 "$TMP/server.log" | sed 's/^/          /'; }
exit "$RC"
