#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  bplot_sua_test.sh — the B5 serving gate: samples/bplot/server.b serves a
#  real chart, safely, to concurrent clients.
#
#  What it holds the sample to:
#    * the chart arrives as image/svg+xml with a CSP header, and parses;
#    * a hostile title from the query string is escaped, not executed;
#    * 40 concurrent requests, each titled differently, each get THEIR OWN
#      chart -- the object API is what makes that true, and a handler that
#      used plt.* instead would fail this;
#    * the server is still serving afterwards.
#
#  Run:  BANTU=./bantu-src/compiler/build/bantu bash tests/bplot_sua_test.sh
# ════════════════════════════════════════════════════════════════════════
set -u
BANTU="${BANTU:-bantu}"
case "$BANTU" in /*) ;; */*) BANTU="$(cd "$(dirname "$BANTU")" && pwd)/$(basename "$BANTU")" ;; esac
ROOT="${BPLOT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
SERVER="${BPLOT_SERVER:-$ROOT/samples/bplot/server.b}"
PORT="${PORT:-39931}"
TMP="$(mktemp -d)"
PASS=0
FAIL=0
cleanup() { [ -n "${PID:-}" ] && kill "$PID" 2>/dev/null; wait "${PID:-}" 2>/dev/null; rm -rf "$TMP"; }
trap cleanup EXIT
ok()    { PASS=$((PASS+1)); echo "  ok    $1"; }
bad()   { FAIL=$((FAIL+1)); echo "  FAIL  $1"; }
check() { if [ "$1" = "1" ]; then ok "$2"; else bad "$2"; fi; }
PY="$(command -v python3 || command -v python || true)"
wellformed() {
    [ -n "$PY" ] || return 0
    "$PY" -c 'import sys, xml.dom.minidom; xml.dom.minidom.parse(sys.argv[1])' "$1" >/dev/null 2>&1
}

# From the repository root, so `include "bplot"` resolves to ./bplot/ exactly
# as it resolves to ./bantu_modules/bplot/ in an installed project.
( cd "$ROOT" && PORT="$PORT" exec "$BANTU" -q run "$SERVER" ) > "$TMP/server.log" 2>&1 &
PID=$!

i=0
while [ $i -lt 150 ]; do
    curl -s -o /dev/null "http://127.0.0.1:$PORT/" 2>/dev/null && break
    sleep 0.1
    i=$((i+1))
done
if [ $i -ge 150 ]; then
    bad "the server never came up"
    cat "$TMP/server.log"
    exit 1
fi

echo "-- the chart --"
CODE="$(curl -s -D "$TMP/h.txt" -o "$TMP/chart.svg" -w '%{http_code}' "http://127.0.0.1:$PORT/chart.svg")"
check "$([ "$CODE" = "200" ] && echo 1 || echo 0)" "GET /chart.svg answers 200"
check "$(grep -ciE '^content-type: image/svg\+xml' "$TMP/h.txt" | awk '{print ($1 > 0) ? 1 : 0}')" "as image/svg+xml"
check "$(grep -ciE "^content-security-policy: default-src 'none'" "$TMP/h.txt" | awk '{print ($1 > 0) ? 1 : 0}')" "with a Content-Security-Policy that runs nothing"
check "$(grep -ciE '^x-content-type-options: nosniff' "$TMP/h.txt" | awk '{print ($1 > 0) ? 1 : 0}')" "and nosniff"
check "$(wellformed "$TMP/chart.svg" && echo 1 || echo 0)" "the body parses as XML"
check "$(grep -c '>Monthly rainfall, Dar es Salaam</text>' "$TMP/chart.svg")" "and is the chart, with its default title"
check "$(grep -c '>Apr</text>' "$TMP/chart.svg")" "with its categories"
PAGE="$(curl -s "http://127.0.0.1:$PORT/")"
check "$(printf '%s' "$PAGE" | grep -c 'img src="/chart.svg"')" "the page embeds the chart"

echo "-- a hostile title from the query string --"
curl -s -o "$TMP/evil.svg" "http://127.0.0.1:$PORT/chart.svg?title=%3C%2Ftext%3E%3Cscript%3Ealert(1)%3C%2Fscript%3E"
check "$([ -s "$TMP/evil.svg" ] && ! grep -q '<script' "$TMP/evil.svg" && echo 1 || echo 0)" "no <script survives into the document"
check "$(grep -c '&lt;/text&gt;&lt;script&gt;' "$TMP/evil.svg")" "the title is there, escaped"
check "$(wellformed "$TMP/evil.svg" && echo 1 || echo 0)" "and the document still parses"

echo "-- 40 concurrent requests, each with its own title --"
PIDS=""
for n in $(seq 1 40); do
    curl -s --max-time 60 -o "$TMP/c$n.svg" "http://127.0.0.1:$PORT/chart.svg?title=request-$n-end" &
    PIDS="$PIDS $!"
done
for p in $PIDS; do wait "$p" 2>/dev/null; done
OWN=0; CROSS=0; MISSING=0
for n in $(seq 1 40); do
    if [ ! -s "$TMP/c$n.svg" ]; then MISSING=$((MISSING+1)); continue; fi
    titles="$(grep -o 'request-[0-9]*-end' "$TMP/c$n.svg" | sort -u)"
    if [ "$titles" = "request-$n-end" ]; then OWN=$((OWN+1)); else CROSS=$((CROSS+1)); fi
done
echo "        $OWN own chart, $CROSS crossed, $MISSING missing"
check "$([ "$OWN" = "40" ] && echo 1 || echo 0)" "every request got exactly its own chart"
check "$([ "$CROSS" = "0" ] && echo 1 || echo 0)" "no chart carried another request's title"
check "$([ "$MISSING" = "0" ] && echo 1 || echo 0)" "and none was dropped"
check "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/chart.svg" | grep -c '^200$')" "the server is still serving afterwards"

echo ""
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
