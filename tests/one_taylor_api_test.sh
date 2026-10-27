#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  one_taylor_api_test.sh — end to end, against a running backend.
#
#  The assertion this file exists for is the last section: a minor's feed,
#  search, discover, profile view and media fetch must never intersect an
#  adult's, in EITHER direction, and a guessed post id must not be a way
#  around any of it. Everything before that is the setup needed to say so.
#
#  It starts its own server on a scratch database and stops it at the end.
#
#  Run:  bash tests/one_taylor_api_test.sh
# ════════════════════════════════════════════════════════════════════════
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BANTU="$HERE/bantu"
BACKEND="$HERE/samples/one-taylor/backend"
PORT="${ONE_TAYLOR_TEST_PORT:-4177}"
BASE="http://127.0.0.1:$PORT"
WORK="$(mktemp -d)"
ADMIN="test-admin-token-0123456789abcdef"

PASS=0
FAIL=0

ok()   { PASS=$((PASS+1)); echo "  ok    $1"; }
bad()  { FAIL=$((FAIL+1)); echo "  FAIL  $1"; }

# $1 description, $2 expected substring, $3 actual
expect() {
  if [[ "$3" == *"$2"* ]]; then ok "$1"; else
    bad "$1"
    echo "          wanted substring: $2"
    echo "          got: ${3:0:400}"
  fi
}
reject() {
  if [[ "$3" != *"$2"* ]]; then ok "$1"; else
    bad "$1"
    echo "          must NOT contain: $2"
    echo "          got: ${3:0:400}"
  fi
}

cleanup() {
  if [[ -n "${SRV_PID:-}" ]]; then kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; fi
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "════════════════════════════════════════════════════════════════"
echo "  One Taylor — API end to end"
echo "════════════════════════════════════════════════════════════════"

# ── start the server on a scratch database ──────────────────────────────
cd "$BACKEND" || exit 1
ONE_TAYLOR_SECRET="$(openssl rand -hex 32)" \
ONE_TAYLOR_ADMIN_TOKEN="$ADMIN" \
ONE_TAYLOR_DB="$WORK/test.db" \
PORT="$PORT" \
"$BANTU" run server.b > "$WORK/server.log" 2>&1 &
SRV_PID=$!

for _ in $(seq 1 60); do
  if curl -fsS --max-time 2 "$BASE/api/health" >/dev/null 2>&1; then break; fi
  sleep 0.5
done
HEALTH="$(curl -fsS --max-time 5 "$BASE/api/health" 2>&1)"
expect "the server answers /api/health" '"ok":true' "$HEALTH"
if [[ "$HEALTH" == *'"classifier":true'* ]]; then
  ok "the moderation classifier loaded"
else
  echo "  note  classifier not loaded — lexicon-only moderation for this run"
  echo "        $(echo "$HEALTH" | sed 's/.*"reason":"\([^"]*\)".*/\1/' | cut -c1-120)"
fi

api() { # method path token json
  local m="$1" p="$2" t="$3" body="${4:-}"
  if [[ -n "$body" ]]; then
    curl -s --max-time 20 -X "$m" "$BASE$p" \
      -H 'content-type: application/json' \
      ${t:+-H "authorization: Bearer $t"} -d "$body"
  else
    curl -s --max-time 20 -X "$m" "$BASE$p" ${t:+-H "authorization: Bearer $t"}
  fi
}
tokenOf() { echo "$1" | sed -n 's/.*"token":"\([^"]*\)".*/\1/p'; }
# Built in a variable rather than inline: inside "$( ... )" bash applies brace
# expansion to {"a":1,"b":2}, so the braces vanish and everything after the
# first comma becomes separate words.
jsonBody() { printf '%s' '{"body":"'"$1"'"}'; }
# grep -o, not sed: `.*"id":` is GREEDY, so it matched the LAST "id" in the
# response — the author's, or a media row's — rather than the post's. Every
# earlier assertion that used this passed anyway, because they only needed an
# id that existed, which is how the bug survived.
idOf()    { echo "$1" | grep -o '"id":[0-9]*' | head -1 | cut -d: -f2; }

echo
echo "── accounts ──"

# Two adults in range of each other, one adult far outside, two minors.
R="$(api POST /api/auth/signup "" '{"username":"amina","email":"amina@example.com","password":"correct-horse-7","dob":"2000-04-11","gender":"woman"}')"
AMINA="$(tokenOf "$R")"
expect "an adult can sign up" '"band":"adult"' "$R"
reject "signup does not echo the email" "amina@example.com" "$R"
reject "signup does not echo the date of birth" "2000-04-11" "$R"
reject "signup does not echo a password hash" 'argon2' "$R"

R="$(api POST /api/auth/signup "" '{"username":"bakari","email":"bakari@example.com","password":"correct-horse-7","dob":"1998-02-02","gender":"man"}')"
BAKARI="$(tokenOf "$R")"
expect "a second adult can sign up" '"band":"adult"' "$R"

R="$(api POST /api/auth/signup "" '{"username":"mzee","email":"mzee@example.com","password":"correct-horse-7","dob":"1966-02-02","gender":"man"}')"
MZEE="$(tokenOf "$R")"
expect "an older adult can sign up" '"band":"adult"' "$R"

R="$(api POST /api/auth/signup "" '{"username":"juma","email":"juma@example.com","password":"correct-horse-7","dob":"2011-04-11","gender":"man"}')"
JUMA="$(tokenOf "$R")"
expect "a 15-year-old can sign up" '"band":"minor"' "$R"
expect "a minor's range is locked to +/-2" '"min":13,"max":17' "$R"

R="$(api POST /api/auth/signup "" '{"username":"neema","email":"neema@example.com","password":"correct-horse-7","dob":"2010-06-01","gender":"woman"}')"
NEEMA="$(tokenOf "$R")"
expect "a 16-year-old can sign up" '"band":"minor"' "$R"

echo
echo "── signup refusals ──"
R="$(api POST /api/auth/signup "" '{"username":"tiny","email":"tiny@example.com","password":"correct-horse-7","dob":"2018-01-01","gender":"man"}')"
expect "an 8-year-old is refused" "at least 13" "$R"
R="$(api POST /api/auth/signup "" '{"username":"amina","email":"other@example.com","password":"correct-horse-7","dob":"2000-01-01","gender":"woman"}')"
expect "a duplicate username is refused" "taken" "$R"
R="$(api POST /api/auth/signup "" '{"username":"amina2","email":"amina@example.com","password":"correct-horse-7","dob":"2000-01-01","gender":"woman"}')"
expect "a duplicate email is refused via the blind index" "cannot be used" "$R"
R="$(api POST /api/auth/signup "" '{"username":"weak","email":"weak@example.com","password":"qwertyuiop","dob":"2000-01-01","gender":"man"}')"
expect "a common password is refused" "too common" "$R"
R="$(api POST /api/auth/signup "" '{"username":"short","email":"short@example.com","password":"abc","dob":"2000-01-01","gender":"man"}')"
expect "a short password is refused" "at least 10" "$R"
R="$(api POST /api/auth/signup "" '{"username":"nodate","email":"nodate@example.com","password":"correct-horse-7","dob":"not-a-date","gender":"man"}')"
expect "a malformed date of birth is refused" "YYYY-MM-DD" "$R"

echo
echo "── login ──"
R="$(api POST /api/auth/login "" '{"username":"amina","password":"correct-horse-7"}')"
expect "login succeeds with the right password" '"ok":true' "$R"
R="$(api POST /api/auth/login "" '{"username":"amina","password":"wrong-password-xx"}')"
expect "login fails with the wrong password" "do not match" "$R"
R="$(api POST /api/auth/login "" '{"username":"ghost","password":"correct-horse-7"}')"
expect "an unknown user gets the same answer as a wrong password" "do not match" "$R"

echo
echo "── authentication is required ──"
for route in /api/feed /api/discover /api/auth/me /api/search/users; do
  R="$(api GET "$route" "")"
  expect "GET $route needs a session" '"ok":false' "$R"
done
R="$(api POST /api/posts "" '{"body":"hello"}')"
expect "POST /api/posts needs a session" '"ok":false' "$R"
R="$(api GET /api/media/1 "")"
expect "GET /api/media/:id needs a session" '"ok":false' "$R"
R="$(api GET /api/admin/moderation/queue "")"
expect "the admin queue is invisible without a token" "not found" "$R"

echo
echo "── profiles and interests ──"
R="$(api PUT /api/settings/tags "$AMINA" '{"interests":["chess","photography","hiking"],"hobbies":["baking"],"dreamPartner":["chess","kind"]}')"
expect "an adult can set interests" '"chess"' "$R"
R="$(api PUT /api/settings/tags "$BAKARI" '{"interests":["chess","photography"],"hobbies":["baking"],"dreamPartner":["chess"]}')"
expect "a second adult can set interests" '"chess"' "$R"
R="$(api PUT /api/settings/tags "$JUMA" '{"interests":["football","coding"],"hobbies":["guitar"]}')"
expect "a minor can set interests" '"coding"' "$R"
R="$(api PUT /api/settings/tags "$NEEMA" '{"interests":["football","coding"],"hobbies":["guitar"]}')"
expect "a second minor can set interests" '"coding"' "$R"

R="$(api PUT /api/settings/profile "$AMINA" '{"displayName":"Amina","bio":"Chess, film photography, long walks.","theme":"dark"}')"
expect "an adult can save a bio" '"ok":true' "$R"
R="$(api PUT /api/settings/profile "$JUMA" '{"displayName":"Juma","bio":"I play football and write small games.","theme":"light"}')"
expect "a minor can save an innocent bio" '"ok":true' "$R"

echo
echo "── the dating range ──"
R="$(api PUT /api/settings/range "$MZEE" '{"minAge":14,"maxAge":70}')"
expect "an adult's range floor is forced to 18" '"min":18' "$R"
reject "the requested floor of 14 is not honoured" '"min":14,' "$R"
R="$(api PUT /api/settings/range "$JUMA" '{"minAge":18,"maxAge":40}')"
expect "a minor's range request is overruled" '"min":13,"max":17' "$R"
expect "and the response says it is locked" '"locked":true' "$R"

echo
echo "── explicit categories ──"
R="$(api PUT /api/settings/explicit "$AMINA" '{"categories":["suggestive"]}')"
expect "an adult can opt into a category" '"suggestive"' "$R"
R="$(api PUT /api/settings/explicit "$JUMA" '{"categories":["suggestive"]}')"
expect "a minor has no such setting at all" "no such setting" "$R"

echo
echo "── posting ──"
R="$(api POST /api/posts "$AMINA" '{"body":"Made pilau for the first time and it actually worked.","tags":["food","chess"]}')"
POST_AMINA="$(idOf "$R")"
expect "an adult can post" '"ok":true' "$R"
R="$(api POST /api/posts "$BAKARI" '{"body":"Lost three games of chess in a row. Humbling.","tags":["chess"]}')"
POST_BAKARI="$(idOf "$R")"
expect "a second adult can post" '"ok":true' "$R"
R="$(api POST /api/posts "$JUMA" '{"body":"Scored twice at practice today, best session all term.","tags":["football"]}')"
POST_JUMA="$(idOf "$R")"
expect "a minor can post" '"ok":true' "$R"
R="$(api POST /api/posts "$NEEMA" '{"body":"Learning guitar chords, my fingers hurt.","tags":["guitar"]}')"
POST_NEEMA="$(idOf "$R")"
expect "a second minor can post" '"ok":true' "$R"
R="$(api POST /api/posts "$AMINA" '{"body":""}')"
expect "an empty post is refused" "write something" "$R"

echo
echo "── moderation at write time ──"
R="$(api POST /api/posts "$JUMA" '{"body":"dont tell your parents, this is our secret, add me somewhere else"}')"
expect "a grooming pattern from a minor account is blocked" '"status":"blocked"' "$R"
expect "and the author is told it can be appealed" '"appealable":true' "$R"
R="$(api POST /api/posts "$JUMA" '{"body":"ill kill it at the audition tomorrow, wish me luck"}')"
expect "an idiom is NOT blocked (ill kill it at the audition)" '"ok":true' "$R"
R="$(api POST /api/posts "$AMINA" '{"body":"i know where you live and i will find you"}')"
expect "a threat from an adult account is blocked too" '"status":"blocked"' "$R"
R="$(api POST /api/posts "$AMINA" '{"body":"this weather is absolute shit and the wifi is worse"}')"
expect "undirected profanity from an adult is allowed" '"ok":true' "$R"

echo
echo "── posting a picture ──"
# A real PNG, built here so the test needs no image library, uploaded base64
# inside the JSON body (sua has no multipart parser).
#
# Two bugs this guards. The upload directory is sharded by hash and
# writefile() does not create parents, so every upload failed with "could not
# store an image" and the post arrived without its picture. And the decoder
# was the pure-Bantu one, which returns a list of bytes: 1,299 ms and four
# million Values for 100 KB, so a phone photo was not slow but impossible.
python3 - "$WORK/img.json" <<'PYIMG'
import base64, json, struct, sys, zlib
w = h = 160
rows = []
for y in range(h):
    row = bytearray([0])
    for x in range(w):
        row += bytes([(x * 255) // w, (y * 255) // h, 160])
    rows.append(bytes(row))
def chunk(tag, data):
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xffffffff)
png = (b"\x89PNG\r\n\x1a\n"
       + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
       + chunk(b"IDAT", zlib.compress(b"".join(rows), 6))
       + chunk(b"IEND", b""))
open(sys.argv[1], "w").write(json.dumps({
    "body": "a picture of a gradient",
    "media": [{"data": "data:image/png;base64," + base64.b64encode(png).decode()}]}))
open(sys.argv[1] + ".png", "wb").write(png)
PYIMG

R="$(curl -s --max-time 60 -X POST "$BASE/api/posts"       -H 'content-type: application/json'       -H "authorization: Bearer $AMINA" --data-binary @"$WORK/img.json")"
expect "a post with a picture is accepted" '"status":"live"' "$R"
expect "with no media errors" '"mediaErrors":[]' "$R"
expect "and the picture is attached" '"mime":"image/png"' "$R"
MEDIA_ID="$(echo "$R" | grep -o '"media":\[{"id":[0-9]*' | grep -o '[0-9]*$')"

if [[ -n "$MEDIA_ID" ]]; then
  CODE="$(curl -s -o "$WORK/got.png" -w '%{http_code}' --max-time 30           "$BASE/api/media/$MEDIA_ID" -H "authorization: Bearer $AMINA")"
  expect "the image serves to its author" "200" "$CODE"
  if cmp -s "$WORK/img.json.png" "$WORK/got.png"; then
    ok "and the bytes are byte-identical to what was uploaded"
  else
    bad "the served bytes differ from the upload"
  fi
  CODE="$(curl -s -o /dev/null -w '%{http_code}' --max-time 30           "$BASE/api/media/$MEDIA_ID" -H "authorization: Bearer $MZEE")"
  expect "and to another adult" "200" "$CODE"
  CODE="$(curl -s -o /dev/null -w '%{http_code}' --max-time 30           "$BASE/api/media/$MEDIA_ID" -H "authorization: Bearer $JUMA")"
  expect "but a minor gets 404, not the bytes" "404" "$CODE"
  CODE="$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 "$BASE/api/media/$MEDIA_ID")"
  expect "and an anonymous caller gets 401" "401" "$CODE"
  R="$(api GET /api/feed "$MZEE")"
  expect "the picture appears in another adult's feed" "a picture of a gradient" "$R"
else
  bad "could not read the media id from the response"
fi

R="$(api POST /api/posts "$AMINA" '{"body":"a bad attachment","media":[{"data":"data:image/png;base64,!!!!not base64!!!!"}]}')"
expect "a corrupt attachment does not stop the post" '"status":"live"' "$R"
expect "and the reason is reported, not swallowed" "not valid base64" "$R"

echo
echo "── the author is TOLD what happened to their post ──"
# The bug this guards: the backend held a flagged post correctly and returned
# a plain 201 with no verdict and no message, so the client had nothing to
# show. The post vanished silently and the author concluded that moderation
# was not running at all. A safety decision nobody is told about is not a
# safety decision, it is a lost post.
R="$(api POST /api/posts "$AMINA" "$(jsonBody 'I wanna kill you')")"
expect "a threat is held rather than published" '"status":"review"' "$R"
expect "the response names the verdict" '"verdict":"hold"' "$R"
expect "and explains it in a sentence" "Held for review" "$R"
expect "and says a moderator will look" '"queuedForReview":true' "$R"
HELD_ID="$(idOf "$R")"

R="$(api POST /api/posts "$AMINA" "$(jsonBody 'ill kill it at the audition tomorrow')")"
expect "while the idiom is published" '"status":"live"' "$R"
expect "with an allow verdict" '"verdict":"allow"' "$R"
reject "and no scary message" "Held for review" "$R"

R="$(api GET /api/users/amina "$AMINA")"
expect "the author sees their own held post on their profile" '"status":"review"' "$R"
R="$(api GET /api/users/amina "$MZEE")"
reject "but nobody else does" '"status":"review"' "$R"
reject "and its text is not served to them either" "I wanna kill you" "$R"

R="$(api GET /api/feed "$MZEE")"
reject "a held post is absent from everyone else's feed" "I wanna kill you" "$R"
R="$(api GET "/api/posts/$HELD_ID" "$MZEE")"
expect "and 404 by id for them" "no such post" "$R"
R="$(api GET "/api/posts/$HELD_ID" "$AMINA")"
expect "while its author can still open it" '"ok":true' "$R"

echo
echo "── the feed ──"
F_AMINA="$(api GET /api/feed "$AMINA")"
expect "an adult sees an adult feed" '"band":"adult"' "$F_AMINA"
expect "the adult feed contains another adult's post" "Lost three games of chess" "$F_AMINA"
F_JUMA="$(api GET /api/feed "$JUMA")"
expect "a minor sees a minor feed" '"band":"minor"' "$F_JUMA"
expect "the minor feed contains another minor's post" "Learning guitar chords" "$F_JUMA"

echo
echo "════ THE AGE WALL ════"
reject "a minor's feed contains NO adult post (pilau)" "Made pilau" "$F_JUMA"
reject "a minor's feed contains NO adult post (chess)" "Lost three games of chess" "$F_JUMA"
reject "an adult's feed contains NO minor post (football)" "Scored twice at practice" "$F_AMINA"
reject "an adult's feed contains NO minor post (guitar)" "Learning guitar chords" "$F_AMINA"

R="$(api GET "/api/posts/$POST_AMINA" "$JUMA")"
expect "a minor fetching an adult's post by id gets 404" "no such post" "$R"
R="$(api GET "/api/posts/$POST_JUMA" "$AMINA")"
expect "an adult fetching a minor's post by id gets 404" "no such post" "$R"
R="$(api GET "/api/posts/$POST_NEEMA" "$JUMA")"
expect "a minor CAN fetch another in-band minor's post" '"ok":true' "$R"

R="$(api GET /api/users/amina "$JUMA")"
expect "a minor looking up an adult's profile gets 404" "no such account" "$R"
R="$(api GET /api/users/juma "$AMINA")"
expect "an adult looking up a minor's profile gets 404" "no such account" "$R"
R="$(api GET /api/users/neema "$JUMA")"
expect "a minor CAN see another in-band minor's profile" '"username":"neema"' "$R"

R="$(api GET /api/search/users?q=a "$JUMA")"
reject "a minor's search returns no adult (amina)" '"username":"amina"' "$R"
reject "a minor's search returns no adult (bakari)" '"username":"bakari"' "$R"
R="$(api GET /api/search/users?q=a "$AMINA")"
reject "an adult's search returns no minor (juma)" '"username":"juma"' "$R"
reject "an adult's search returns no minor (neema)" '"username":"neema"' "$R"
expect "an adult's search does return another adult" '"username":"bakari"' "$R"

R="$(api POST /api/users/amina/follow "$JUMA")"
expect "a minor cannot follow an adult" "no such account" "$R"
R="$(api POST /api/users/juma/follow "$AMINA")"
expect "an adult cannot follow a minor" "no such account" "$R"
R="$(api POST /api/users/neema/follow "$JUMA")"
expect "a minor CAN follow an in-band minor" '"following":true' "$R"
R="$(api POST /api/users/bakari/follow "$AMINA")"
expect "an adult CAN follow another adult" '"following":true' "$R"

R="$(api POST "/api/posts/$POST_AMINA/like" "$JUMA")"
expect "a minor cannot like an adult's post" "no such post" "$R"
R="$(api POST "/api/posts/$POST_AMINA/report" "$JUMA" '{"reason":"explicit"}')"
expect "a minor cannot even report across the wall" "no such post" "$R"

echo
echo "── discover (the dating range, not the content wall) ──"
R="$(api GET /api/discover "$AMINA")"
expect "discover works for an adult" '"band":"adult"' "$R"
expect "a shared interest is matched and explained" "you both like" "$R"
reject "discover never crosses the wall (juma)" '"username":"juma"' "$R"
R="$(api GET /api/discover "$JUMA")"
expect "discover works for a minor" '"band":"minor"' "$R"
expect "a minor's discover range is locked" '"rangeLocked":true' "$R"
reject "a minor's discover never returns an adult" '"username":"amina"' "$R"
R="$(api GET /api/discover "$MZEE")"
reject "an out-of-range adult is not offered as a match" '"username":"amina"' "$R"

echo
echo "── reports: one hides, two remove ──"
R="$(api POST "/api/posts/$POST_BAKARI/report" "$AMINA" '{"reason":"harassment"}')"
expect "a report is accepted" '"reported":true' "$R"
expect "and hides the post from the reporter at once" '"hiddenFromYou":true' "$R"
R="$(api GET /api/feed "$AMINA")"
reject "the reported post is gone from the reporter's feed" "Lost three games of chess" "$R"
R="$(api GET /api/feed "$MZEE")"
expect "but is still visible to everyone else" "Lost three games of chess" "$R"

R="$(api POST "/api/posts/$POST_BAKARI/report" "$AMINA" '{"reason":"harassment"}')"
reject "a second report from the SAME account does not remove it" '"removed":true' "$R"
R="$(api POST "/api/posts/$POST_BAKARI/report" "$MZEE" '{"reason":"harassment"}')"
expect "a second DISTINCT reporter removes it" '"removed":true' "$R"
R="$(api GET /api/feed "$MZEE")"
reject "the removed post is gone for everyone" "Lost three games of chess" "$R"
R="$(api GET "/api/posts/$POST_BAKARI" "$MZEE")"
expect "and is 404 by id" "no such post" "$R"

echo
echo "── the moderation queue ──"
R="$(curl -s --max-time 20 "$BASE/api/admin/moderation/queue" -H "x-admin-token: $ADMIN")"
expect "the queue opens with the operator token" '"ok":true' "$R"
expect "the blocked grooming post is in it" "our secret" "$R"
R="$(curl -s --max-time 20 "$BASE/api/admin/moderation/queue" -H "x-admin-token: wrong-token-entirely")"
expect "a wrong operator token gets 404, not 403" "not found" "$R"

echo
echo "── messaging over HTTP (the fallback when a socket is blocked) ──"
# A separate code path from the WebSocket, with the same gate and the same
# moderation. tests/one_taylor_realtime_test.py covers the socket.
R="$(api POST /api/conversations/bakari "$AMINA" '{}')"
expect "a thread needs a MUTUAL follow, not just one way" "follow each other" "$R"
R="$(api POST /api/users/amina/follow "$BAKARI" '{}')"
expect "bakari follows back" '"following":true' "$R"
R="$(api POST /api/conversations/bakari "$AMINA" '{}')"
expect "now the thread opens" '"conversationId"' "$R"
R="$(api POST /api/conversations/bakari/messages "$AMINA" "$(jsonBody 'are you free for a game on saturday')")"
expect "a message sends" '"delivered":true' "$R"
R="$(api GET /api/conversations/bakari/messages "$AMINA")"
expect "and is in the history" "game on saturday" "$R"
R="$(api GET /api/conversations "$BAKARI")"
expect "the recipient sees the thread" '"unread":1' "$R"
R="$(api POST /api/conversations/bakari/read "$AMINA" '{}')"
expect "marking read works" '"ok":true' "$R"

R="$(api POST /api/conversations/juma "$AMINA" '{}')"
expect "an adult cannot open a thread with a minor" "no such account" "$R"
R="$(api POST /api/conversations/amina "$JUMA" '{}')"
expect "and a minor cannot with an adult" "no such account" "$R"
R="$(api GET /api/conversations/juma/messages "$AMINA")"
expect "nor read one across the wall" "no such account" "$R"

R="$(api POST /api/realtime/ticket "$AMINA" '{}')"
expect "a WebSocket ticket is issued to a session" '"ticket"' "$R"
R="$(api POST /api/realtime/ticket "" '{}')"
expect "and refused without one" '"ok":false' "$R"
R="$(api GET /api/realtime/config "$AMINA")"
expect "the realtime config names a WebSocket url" '"wsUrl"' "$R"
expect "and carries ICE servers for calls" '"iceServers"' "$R"

echo
echo "── logout ──"
R="$(api POST /api/auth/logout "$NEEMA" '{}')"
expect "logout succeeds" '"loggedOut":true' "$R"
R="$(api GET /api/auth/me "$NEEMA")"
expect "the revoked token no longer works" '"ok":false' "$R"

echo
echo "════════════════════════════════════════════════════════════════"
echo "  PASS: $PASS   FAIL: $FAIL"
if [[ "$FAIL" -eq 0 ]]; then
  echo "  RESULT: ALL GREEN"
  exit 0
else
  echo "  RESULT: FAILED"
  echo "  server log: $WORK/server.log (removed on exit; rerun with -x to keep)"
  tail -20 "$WORK/server.log"
  exit 1
fi
