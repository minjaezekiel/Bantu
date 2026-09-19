#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  sua_udp_server_test.sh — the callback UDP server and multicast.
#
#  Covers what docs/sua-udp-server-design.md adds to sua.udp: the two-line
#  server form of bind(), $packet.reply, serving UDP and HTTP from one loop,
#  and joining a multicast group.
#
#  Run:  BANTU=./bantu-src/compiler/build/bantu bash tests/sua_udp_server_test.sh
# ════════════════════════════════════════════════════════════════════════
set -u
BANTU="${BANTU:-bantu}"
case "$BANTU" in /*) ;; */*) BANTU="$(cd "$(dirname "$BANTU")" && pwd)/$(basename "$BANTU")" ;; esac
cd "$(dirname "$0")/.."
UDP_PORT="${UDP_PORT:-39261}"
MIX_HTTP="${MIX_HTTP:-39262}"
MIX_UDP="${MIX_UDP:-39263}"
MC_PORT="${MC_PORT:-39264}"
TMP="$(mktemp -d)"
PIDS=""
cleanup() { for p in $PIDS; do kill "$p" 2>/dev/null; done; rm -rf "$TMP"; }
trap cleanup EXIT

PASS=0
FAIL=0
ok()  { PASS=$((PASS+1)); echo "  ok    $1"; }
bad() { FAIL=$((FAIL+1)); echo "  FAIL  $1"; }
check() { if [ "$2" = "$3" ]; then ok "$1"; else bad "$1 (expected '$3', got '$2')"; fi; }

# ── a UDP-only server: bind(port, handler) + listen() ───────────────────
cat > "$TMP/udp.b" <<EOF
sua.udp.bind($UDP_PORT, def(\$packet) {
    if (len(\$packet.data) == 3) {
        \$boom = \$nopeNotDefined;            // a handler that raises
    }
    \$packet.reply(\$packet.data);
});
sua.udp.listen();
EOF
"$BANTU" -q run "$TMP/udp.b" > "$TMP/udp.log" 2>&1 &
PIDS="$PIDS $!"
sleep 2

UDP_PORT="$UDP_PORT" python3 - > "$TMP/udp.out" 2>&1 <<'PY'
import os, socket
P = int(os.environ["UDP_PORT"])
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(3)

s.sendto(b"ping", ("127.0.0.1", P))
data, frm = s.recvfrom(4096)
print("echo", data.decode())
print("from_port", "yes" if frm[1] == P else "no")

for i in range(20):
    s.sendto(b"burst%02d" % i, ("127.0.0.1", P))
got = set()
try:
    for _ in range(20):
        got.add(s.recvfrom(4096)[0])
except Exception:
    pass
print("burst", len(got))

# A datagram whose handler raises must cost only that datagram.
s.sendto(b"bad", ("127.0.0.1", P))
try:
    s.recvfrom(4096)
    print("raised_replied", "yes")
except Exception:
    print("raised_replied", "no")
s.sendto(b"after", ("127.0.0.1", P))
try:
    print("survives", s.recvfrom(4096)[0].decode())
except Exception:
    print("survives", "DEAD")

# Nothing is truncated below the datagram maximum.
big = bytes(1400)
s.sendto(big, ("127.0.0.1", P))
print("big", len(s.recvfrom(65536)[0]))
PY

uget() { grep -m1 "^$1 " "$TMP/udp.out" | cut -d' ' -f2-; }
check "bind(port, handler) answers a datagram"   "$(uget echo)" "ping"
check "the reply comes from the bound port"      "$(uget from_port)" "yes"
check "20 datagrams in a burst are all answered" "$(uget burst)" "20"
check "a handler that raises sends no reply"     "$(uget raised_replied)" "no"
check "and the server serves the next datagram"  "$(uget survives)" "after"
check "a 1400-byte datagram is not truncated"    "$(uget big)" "1400"
if grep -qi "handler error" "$TMP/udp.log"; then ok "the handler error is reported on stderr"
else bad "the handler error was not logged"; fi

# ── HTTP and UDP served by one loop ─────────────────────────────────────
cat > "$TMP/mix.b" <<EOF
sua.udp.bind($MIX_UDP, def(\$packet) { \$packet.reply([79, 75]); });   // "OK"
sua.server.get("/ping", def(\$req, \$res) { \$res.json({"ok": true}); });
sua.server.listen($MIX_HTTP);
EOF
"$BANTU" -q run "$TMP/mix.b" > "$TMP/mix.log" 2>&1 &
PIDS="$PIDS $!"
for _ in $(seq 1 50); do curl -s -o /dev/null "http://127.0.0.1:$MIX_HTTP/ping" && break; sleep 0.2; done

check "the HTTP route answers" \
    "$(curl -s "http://127.0.0.1:$MIX_HTTP/ping")" '{"ok":true}'
check "the UDP socket answers on the same loop" \
    "$(MIX_UDP=$MIX_UDP python3 -c '
import os, socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(3)
s.sendto(b"x", ("127.0.0.1", int(os.environ["MIX_UDP"])))
try: print(s.recvfrom(64)[0].decode())
except Exception: print("TIMEOUT")')" "OK"
check "HTTP still answers after the datagram" \
    "$(curl -s "http://127.0.0.1:$MIX_HTTP/ping")" '{"ok":true}'

# ── multicast ───────────────────────────────────────────────────────────
cat > "$TMP/mc.b" <<EOF
\$s = sua.udp.socket({});
sua.udp.bind(\$s, "0.0.0.0:$MC_PORT");
// The loopback interface is named explicitly: with "any" the kernel picks a
// route, and on macOS that is not the one a local sender uses.
print("joined " + str(sua.udp.join_group(\$s, "239.255.42.99", "127.0.0.1")));
\$pkt = sua.udp.recvfrom(\$s, {"timeoutMs": 6000});
if (\$pkt.timeout) { print("RESULT timeout"); } else { print("RESULT " + tohex(\$pkt.data)); }
print("left " + str(sua.udp.leave_group(\$s, "239.255.42.99")));
sua.udp.close(\$s);
EOF
"$BANTU" -q run "$TMP/mc.b" > "$TMP/mc.log" 2>&1 &
MC_PID=$!
PIDS="$PIDS $MC_PID"
sleep 2
MC_PORT="$MC_PORT" python3 - <<'PY'
import os, socket, time
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
try:
    s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton("127.0.0.1"))
except Exception:
    pass
for _ in range(4):
    s.sendto(b"\xbe\xef", ("239.255.42.99", int(os.environ["MC_PORT"])))
    time.sleep(0.4)
PY
wait "$MC_PID" 2>/dev/null
if grep -q "joined true" "$TMP/mc.log"; then ok "join_group succeeds on a bound socket"
else bad "join_group failed: $(tail -2 "$TMP/mc.log")"; fi
if grep -q "left true" "$TMP/mc.log"; then ok "leave_group succeeds"
else bad "leave_group failed: $(tail -2 "$TMP/mc.log")"; fi
if grep -q "RESULT beef" "$TMP/mc.log"; then
    ok "a datagram sent to the group is received"
elif grep -q "RESULT timeout" "$TMP/mc.log"; then
    echo "  --    no multicast delivery in this environment; membership still verified"
else
    bad "the multicast receiver did not run: $(tail -3 "$TMP/mc.log")"
fi

# ── the errors that teach ───────────────────────────────────────────────
run_err() { "$BANTU" -q run "$1" 2>&1 | tr '\n' ' '; }

cat > "$TMP/e1.b" <<'EOF'
$s = sua.udp.socket({});
sua.udp.join_group($s, "239.1.2.3");
EOF
case "$(run_err "$TMP/e1.b")" in
    *"bind the socket first"*) ok "joining an unbound socket names the fix" ;;
    *) bad "unbound join_group error: $(run_err "$TMP/e1.b")" ;;
esac

cat > "$TMP/e2.b" <<'EOF'
$s = sua.udp.socket({"family": "ipv6"});
sua.udp.bind($s, "[::]:0");
sua.udp.join_group($s, "239.1.2.3");
EOF
case "$(run_err "$TMP/e2.b")" in
    *"IPv4 group but the socket is IPv6"*) ok "a family mismatch says which is which" ;;
    *) bad "family mismatch error: $(run_err "$TMP/e2.b")" ;;
esac

cat > "$TMP/e3.b" <<'EOF'
sua.udp.listen();
EOF
case "$(run_err "$TMP/e3.b")" in
    *"no UDP server registered"*) ok "listen() with nothing bound says so" ;;
    *) bad "empty listen() error: $(run_err "$TMP/e3.b")" ;;
esac

cat > "$TMP/e4.b" <<'EOF'
$h = sua.udp.bind(0, def($packet) { $packet.reply($packet.data); });
print("handle " + str($h.bound) + " " + $h.family);
print("closed " + str(sua.udp.close($h)));
EOF
check "the server form returns a usable handle" \
    "$("$BANTU" -q run "$TMP/e4.b" 2>&1 | tr '\n' ' ')" "handle true ipv4 closed true "

echo ""
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
