#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  webrtc_tls_proxy_test.sh — samples/webrtc-chat/tls-dev-proxy.py
#
#  [found] The proxy ran the TLS handshake ON THE ACCEPT THREAD, on a socket
#  with no timeout. One client that opened a TCP connection and sent nothing
#  therefore stopped it accepting ANY further connection for as long as that
#  socket stayed open. Connections already running were unaffected, because
#  each has its own pump threads -- so the symptom was "my phone and laptop
#  are fine, but a third device cannot load the page at all", which is how it
#  was reported. Browsers open exactly such sockets: Safari and Chrome both
#  preconnect speculatively and may never send bytes on the spare.
#
#  Measured before the fix: a request made while one silent socket was open
#  timed out after 6,002 ms, every time, and recovered the instant that socket
#  closed. After: 3-5 ms.
#
#  Run:  BANTU=./bantu-src/compiler/build/bantu bash tests/webrtc_tls_proxy_test.sh
# ════════════════════════════════════════════════════════════════════════
set -u
BANTU="${BANTU:-bantu}"
# Resolved before anything cds into the sample: a relative BANTU would stop
# naming the binary the moment the working directory changed.
case "$BANTU" in
    */*) BANTU="$(cd "$(dirname "$BANTU")" && pwd)/$(basename "$BANTU")" ;;
esac
PORT="${PORT:-39961}"
TLSPORT="${TLSPORT:-39962}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAMPLE="$ROOT/samples/webrtc-chat"
TMP="$(mktemp -d)"
PY="${PYTHON:-python3}"

cleanup() {
    [ -n "${SRV:-}" ]   && kill "$SRV" 2>/dev/null
    [ -n "${PROXY:-}" ] && kill "$PROXY" 2>/dev/null
    rm -rf "$TMP"
}
trap cleanup EXIT

command -v openssl >/dev/null 2>&1 || { echo "  SKIP  openssl not on PATH"; exit 0; }
"$PY" -c "import ssl" 2>/dev/null     || { echo "  SKIP  python3 has no ssl module"; exit 0; }

( cd "$SAMPLE" && PORT="$PORT" "$BANTU" run server.b ) > "$TMP/server.log" 2>&1 &
SRV=$!
for _ in $(seq 1 60); do curl -s -o /dev/null "http://127.0.0.1:$PORT/health" && break; sleep 0.2; done
if ! curl -s -o /dev/null "http://127.0.0.1:$PORT/health"; then
    echo "  FAIL  chat server did not start"; sed 's/^/          /' "$TMP/server.log"; exit 1
fi

( cd "$SAMPLE" && "$PY" -u tls-dev-proxy.py --target "$PORT" --listen "$TLSPORT" ) > "$TMP/proxy.log" 2>&1 &
PROXY=$!

TLSPORT="$TLSPORT" "$PY" - <<'PYEOF'
import os, socket, ssl, sys, time

PORT = int(os.environ["TLSPORT"])
P = F = 0

def check(name, got, want):
    global P, F
    if str(got) == str(want):
        P += 1; print("  ok    %s" % name)
    else:
        F += 1; print("  FAIL  %s\n          got: %s | want: %s" % (name, got, want))

def ctx():
    c = ssl.create_default_context()
    c.check_hostname = False
    c.verify_mode = ssl.CERT_NONE
    return c

def fetch(timeout=8.0):
    """A real request through the proxy. Returns the status line, or why not."""
    try:
        raw = socket.create_connection(("127.0.0.1", PORT), timeout=timeout)
        s = ctx().wrap_socket(raw)
        s.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        line = s.recv(64).splitlines()[0].decode(errors="replace")
        s.close()
        return line
    except Exception as exc:
        return "%s: %s" % (type(exc).__name__, exc)

# The proxy generates a certificate on first run, which takes a moment.
for _ in range(80):
    if fetch(3.0).startswith("HTTP/1.1 200"):
        break
    time.sleep(0.25)

check("the proxy forwards HTTPS to the chat server", fetch(), "HTTP/1.1 200 OK")

# THE regression. One socket, connected and silent.
silent = socket.create_connection(("127.0.0.1", PORT), timeout=5)
time.sleep(0.3)
check("a silent socket does not block anyone else", fetch(), "HTTP/1.1 200 OK")

more = [socket.create_connection(("127.0.0.1", PORT), timeout=5) for _ in range(5)]
time.sleep(0.3)
check("six silent sockets do not block anyone either", fetch(), "HTTP/1.1 200 OK")

for s in [silent] + more:
    s.close()
time.sleep(0.3)
check("and it still serves after they close", fetch(), "HTTP/1.1 200 OK")

# A socket that sends garbage instead of a ClientHello must be its own problem
# too -- that is a port scanner, or a browser that tried plain HTTP.
junk = socket.create_connection(("127.0.0.1", PORT), timeout=5)
junk.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
time.sleep(0.3)
check("a plain-HTTP request to the TLS port blocks nobody", fetch(), "HTTP/1.1 200 OK")
junk.close()

print("\n  %d passed, %d failed" % (P, F))
sys.exit(1 if F else 0)
PYEOF
RC=$?
if [ "$RC" -ne 0 ]; then
    echo "  ── proxy log ──"; sed 's/^/          /' "$TMP/proxy.log"
fi
exit "$RC"
