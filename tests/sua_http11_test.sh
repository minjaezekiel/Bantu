#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  sua_http11_test.sh — persistent connections, chunked bodies, cacheable
#  files, and the protocol details around them.
#
#  Every assertion here is a behaviour that was wrong before
#  docs/sua-http11-design.md: the server closed after every response, dropped
#  chunked bodies without a word, re-read static files on every request, and
#  answered HEAD with a body.
#
#  Run:  BANTU=./bantu-src/compiler/build/bantu bash tests/sua_http11_test.sh
# ════════════════════════════════════════════════════════════════════════
set -u
BANTU="${BANTU:-bantu}"
case "$BANTU" in /*) ;; */*) BANTU="$(cd "$(dirname "$BANTU")" && pwd)/$(basename "$BANTU")" ;; esac
cd "$(dirname "$0")/.."
PORT="${PORT:-39231}"
TMP="$(mktemp -d)"
SRV_PID=""
cleanup() { [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null; rm -rf "$TMP"; }
trap cleanup EXIT

mkdir -p "$TMP/pub"
printf 'hello static' > "$TMP/pub/s.txt"
printf 'spaced name' > "$TMP/pub/a b.txt"
python3 -c "open('$TMP/pub/big.bin','wb').write(b'x'*300000)"

cat > "$TMP/srv.b" <<EOF
sua.server.get("/fast", def(\$req, \$res) { \$res.json({"ok": true}); });
sua.server.post("/echo", def(\$req, \$res) { \$res.json({"body": str(\$req.body)}); });
sua.server.get("/user/:id", def(\$req, \$res) { \$res.json({"id": \$req.params.id}); });
sua.server.get("/whoami", def(\$req, \$res) { \$res.json({"m": \$req.method}); });
sua.server.head("/only-head", def(\$req, \$res) { \$res.set("X-From", "head-route"); \$res.send(""); });
sua.server.static("$TMP/pub");
sua.server.listen($PORT);
EOF

"$BANTU" -q run "$TMP/srv.b" > "$TMP/srv.log" 2>&1 &
SRV_PID=$!
for _ in $(seq 1 50); do
    curl -s -o /dev/null "http://127.0.0.1:$PORT/fast" && break
    sleep 0.2
done

PASS=0
FAIL=0
ok()  { PASS=$((PASS+1)); echo "  ok    $1"; }
bad() { FAIL=$((FAIL+1)); echo "  FAIL  $1"; }
check() { if [ "$2" = "$3" ]; then ok "$1"; else bad "$1 (expected '$3', got '$2')"; fi; }

# ── the protocol behaviours, driven from one python client ──────────────
OUT="$TMP/probe.out"
PORT="$PORT" python3 - > "$OUT" 2>&1 <<'PY'
import os, socket
P = int(os.environ["PORT"])

def session(reqs, wait=3.0):
    """Send each request on ONE connection, reading a full response between."""
    s = socket.create_connection(("127.0.0.1", P)); s.settimeout(wait)
    out = []
    for r in reqs:
        try:
            s.sendall(r)
            d = b""
            while b"\r\n\r\n" not in d:
                c = s.recv(65536)
                if not c: raise ConnectionError("closed")
                d += c
            head, rest = d.split(b"\r\n\r\n", 1)
            cl = 0
            for ln in head.split(b"\r\n"):
                if ln.lower().startswith(b"content-length:"):
                    cl = int(ln.split(b":", 1)[1])
            # A HEAD reply states the length it WOULD have sent and sends
            # nothing; waiting for that body is the client's mistake.
            if r.startswith(b"HEAD"): cl = 0
            while len(rest) < cl:
                rest += s.recv(65536)
            out.append((head.decode("latin1"), rest))
        except Exception as e:
            out.append(("ERR " + type(e).__name__, b"")); break
    s.close(); return out

def burst(req, expect, wait=3.0):
    """Write everything at once; count responses (pipelining)."""
    s = socket.create_connection(("127.0.0.1", P)); s.settimeout(wait)
    s.sendall(req); d = b""
    try:
        while d.count(b"HTTP/1.1") < expect:
            c = s.recv(65536)
            if not c: break
            d += c
    except Exception:
        pass
    s.close(); return d

def status(h): return h.split("\r\n")[0] if h else ""
def hdr(h, name):
    for ln in h.split("\r\n"):
        if ln.lower().startswith(name.lower() + ":"):
            return ln.split(":", 1)[1].strip()
    return ""

KA = b"GET /fast HTTP/1.1\r\nHost: x\r\n\r\n"

r = session([KA, KA, KA])
print("reuse3", sum(1 for h, _ in r if h.startswith("HTTP/1.1 200")))
print("connhdr", hdr(r[0][0], "Connection"))

r = session([b"GET /fast HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n", KA])
print("closehdr", hdr(r[0][0], "Connection"), len(r))
print("closed_after", "yes" if r[1][0].startswith("ERR") else "no")

r = session([b"GET /fast HTTP/1.0\r\n\r\n", KA])
print("http10_closed", "yes" if r[1][0].startswith("ERR") else "no")

print("pipelined", burst(KA * 4, 4).count(b"HTTP/1.1"))

r = session([b"POST /echo HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n"
             b"5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n", KA])
print("chunked_body", r[0][1].decode("latin1"))
print("chunked_keeps_alive", "yes" if r[1][0].startswith("HTTP/1.1 200") else "no")

r = session([b"POST /echo HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n"
             b"5;ext=1\r\nhello\r\n0\r\nX-Trailer: v\r\n\r\n"])
print("chunk_ext_trailer", r[0][1].decode("latin1"))

r = session([b"POST /echo HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n"
             b"Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n"])
print("cl_te", status(r[0][0]))

r = session([b"HEAD /fast HTTP/1.1\r\nHost: x\r\n\r\n", KA])
print("head_body", len(r[0][1]), "cl", hdr(r[0][0], "Content-Length"))
print("head_keeps_alive", "yes" if r[1][0].startswith("HTTP/1.1 200") else "no")

r = session([b"HEAD /only-head HTTP/1.1\r\nHost: x\r\n\r\n"])
print("head_route", status(r[0][0]), hdr(r[0][0], "X-From"))

r = session([b"GET /whoami HTTP/1.1\r\nHost: x\r\n\r\n"])
print("method_get", r[0][1].decode("latin1"))

r = session([b"DELETE /fast HTTP/1.1\r\nHost: x\r\n\r\n"])
print("405", status(r[0][0]), "|", hdr(r[0][0], "Allow"))

r = session([b"GET /nope HTTP/1.1\r\nHost: x\r\n\r\n"])
print("404", status(r[0][0]))

r = session([b"GET /user/a%20b HTTP/1.1\r\nHost: x\r\n\r\n"])
print("param", r[0][1].decode("latin1"))

r = session([b"GET http://elsewhere/user/7 HTTP/1.1\r\nHost: x\r\n\r\n"])
print("absform", r[0][1].decode("latin1"))

# A half-sent request must not be answered, and the connection stays usable.
s = socket.create_connection(("127.0.0.1", P)); s.settimeout(1.0)
s.sendall(b"GET /fast HTTP/1.1\r\nHost: x\r\n")   # no terminator
try:
    got = s.recv(4096)
except Exception:
    got = b""
s.sendall(b"\r\n")
d = b""
try:
    while b"\r\n\r\n" not in d: d += s.recv(65536)
except Exception:
    pass
print("partial_then_complete", "yes" if (not got and d.startswith(b"HTTP/1.1 200")) else "no")
s.close()
PY

get() { grep -m1 "^$1 " "$OUT" | cut -d' ' -f2-; }

check "three requests share one connection"        "$(get reuse3)" "3"
check "the response says Connection: keep-alive"   "$(get connhdr)" "keep-alive"
check "Connection: close is echoed back"           "$(get closehdr)" "close 2"
check "and the connection really closes"           "$(get closed_after)" "yes"
check "HTTP/1.0 closes unless it asks otherwise"   "$(get http10_closed)" "yes"
check "four pipelined requests get four replies"   "$(get pipelined)" "4"
check "a chunked body reaches the handler"         "$(get chunked_body)" '{"body":"hello world"}'
check "and the connection survives it"             "$(get chunked_keeps_alive)" "yes"
check "chunk extensions and trailers are skipped"  "$(get chunk_ext_trailer)" '{"body":"hello"}'
check "Content-Length + chunked is refused"        "$(get cl_te)" "HTTP/1.1 400 Bad Request"
check "HEAD sends no body, keeps Content-Length"   "$(get head_body)" "0 cl 11"
check "HEAD leaves the connection usable"          "$(get head_keeps_alive)" "yes"
check "an explicit HEAD route still serves HEAD"   "$(get head_route)" "HTTP/1.1 200 OK head-route"
check "a GET still reports its own method"         "$(get method_get)" '{"m":"GET"}'
check "a known path, wrong method, is 405 + Allow" "$(get 405)" "HTTP/1.1 405 Method Not Allowed | GET, HEAD"
check "an unknown path is still 404"               "$(get 404)" "HTTP/1.1 404 Not Found"
check "path parameters are percent-decoded"        "$(get param)" '{"id":"a b"}'
check "absolute-form targets route by path"        "$(get absform)" '{"id":"7"}'
check "a partial request waits for its terminator" "$(get partial_then_complete)" "yes"

# ── static files ────────────────────────────────────────────────────────
BASE="http://127.0.0.1:$PORT"
ETAG="$(curl -s -D - -o /dev/null "$BASE/s.txt" | grep -i '^etag:' | tr -d '\r' | cut -d' ' -f2)"
[ -n "$ETAG" ] && ok "a static file carries an ETag" || bad "no ETag on a static file"
check "If-None-Match answers 304" \
    "$(curl -s -o /dev/null -w '%{http_code}/%{size_download}' -H "If-None-Match: $ETAG" "$BASE/s.txt")" "304/0"
check "If-Modified-Since answers 304" \
    "$(curl -s -o /dev/null -w '%{http_code}/%{size_download}' -H 'If-Modified-Since: Thu, 01 Jan 2099 00:00:00 GMT' "$BASE/s.txt")" "304/0"
check "a stale If-Modified-Since still sends the file" \
    "$(curl -s -o /dev/null -w '%{http_code}/%{size_download}' -H 'If-Modified-Since: Mon, 01 Jan 1990 00:00:00 GMT' "$BASE/s.txt")" "200/12"
check "a byte range answers 206" \
    "$(curl -s -o /dev/null -w '%{http_code}/%{size_download}' -H 'Range: bytes=0-99' "$BASE/big.bin")" "206/100"
check "a suffix range answers 206" \
    "$(curl -s -w '/%{http_code}' -H 'Range: bytes=-5' "$BASE/s.txt")" "tatic/206"
check "an unsatisfiable range answers 416" \
    "$(curl -s -o /dev/null -w '%{http_code}' -H 'Range: bytes=999999-' "$BASE/big.bin")" "416"
check "a percent-encoded file name resolves" \
    "$(curl -s "$BASE/a%20b.txt")" "spaced name"
check "encoded traversal is refused" \
    "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/%2e%2e%2f%2e%2e%2fetc%2fpasswd")" "404"
check "HEAD on a static file sends no body" \
    "$(curl -s -I -o /dev/null -w '%{http_code}/%{size_download}' "$BASE/s.txt")" "200/0"

# An edited file must not be served from the cache.
BEFORE="$(curl -s "$BASE/s.txt")"
sleep 1.1                              # mtime has one-second resolution
printf 'edited now!!' > "$TMP/pub/s.txt"
AFTER="$(curl -s "$BASE/s.txt")"
check "an edited file is re-read, not cached" "$BEFORE/$AFTER" "hello static/edited now!!"

# ── keep-alive is actually used end to end ──────────────────────────────
if command -v ab >/dev/null 2>&1; then
    KAC="$(ab -q -k -c 10 -n 600 "$BASE/fast" 2>/dev/null | grep 'Keep-Alive requests:' | tr -d ' ' | cut -d: -f2)"
    if [ "${KAC:-0}" -ge 500 ]; then ok "ab reports $KAC/600 requests on reused connections"
    else bad "ab reported only ${KAC:-0}/600 keep-alive requests"; fi
else
    echo "  --    ab not installed; end-to-end keep-alive count skipped"
fi

# The server must still be alive after all of that.
check "the server survived the suite" "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/fast")" "200"

echo ""
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || { echo "--- server log ---"; tail -20 "$TMP/srv.log"; exit 1; }
