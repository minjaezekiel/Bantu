#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  one_taylor_stress.sh — does it hold up, and does it still tell the truth
#  while it is under load?
#
#  Four things, in order of how much they would hurt in production:
#
#   1. Memory. sua workers are long-lived; a leak per request is a worker
#      killed at 3am. Resident memory is measured at the start and after
#      several thousand requests, and the fail condition is growth, not an
#      absolute number.
#
#   2. The age wall under concurrency. The wall is enforced per request from
#      session state; if any of that state were shared or cached across
#      requests, a parallel mix of minor and adult traffic is what would
#      expose it. Every response is checked, not just the last.
#
#   3. A feed with thousands of posts. The candidate cap and the indexes are
#      supposed to keep this flat; without them it is O(posts) per request.
#
#   4. A report storm. Fifty accounts reporting one post must take it down
#      once, not fifty times, and must not corrupt the count.
#
#  Run:  bash tests/one_taylor_stress.sh
# ════════════════════════════════════════════════════════════════════════
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BANTU="$HERE/bantu"
BACKEND="$HERE/samples/one-taylor/backend"
PORT="${ONE_TAYLOR_STRESS_PORT:-4188}"
BASE="http://127.0.0.1:$PORT"
WORK="$(mktemp -d)"

POSTS="${STRESS_POSTS:-1200}"
REQUESTS="${STRESS_REQUESTS:-1500}"
CONCURRENCY="${STRESS_CONCURRENCY:-24}"
REPORTERS="${STRESS_REPORTERS:-40}"

PASS=0
FAIL=0
ok()  { PASS=$((PASS+1)); echo "  ok    $1"; }
bad() { FAIL=$((FAIL+1)); echo "  FAIL  $1"; }

# The server log is the only evidence when a worker dies under load, so it is
# copied out before $WORK is removed. ONE_TAYLOR_STRESS_KEEP names where.
KEEP="${ONE_TAYLOR_STRESS_KEEP:-}"
cleanup() {
  if [[ -n "$KEEP" && -f "$WORK/server.log" ]]; then
    cp "$WORK/server.log" "$KEEP" 2>/dev/null && echo "  note  server log kept at $KEEP"
  fi
  if [[ -n "${SRV_PID:-}" ]]; then kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; fi
  rm -rf "$WORK"
}
trap cleanup EXIT

# Is the server process still there? Asked explicitly at each stage, because
# "the next assertion failed" is a terrible way to learn that it died.
alive() { kill -0 "$SRV_PID" 2>/dev/null; }
requireAlive() {
  if alive; then ok "the server is still running after $1"; else
    bad "THE SERVER DIED during $1"
    tail -30 "$WORK/server.log"
    return 1
  fi
}

echo "════════════════════════════════════════════════════════════════"
echo "  One Taylor — stress"
echo "  $POSTS posts · $REQUESTS requests at concurrency $CONCURRENCY · $REPORTERS reporters"
echo "════════════════════════════════════════════════════════════════"

cd "$BACKEND" || exit 1
ONE_TAYLOR_SECRET="$(openssl rand -hex 32)" \
ONE_TAYLOR_DB="$WORK/stress.db" \
PORT="$PORT" \
"$BANTU" -q run server.b > "$WORK/server.log" 2>&1 &
SRV_PID=$!

for _ in $(seq 1 60); do
  curl -fsS --max-time 2 "$BASE/api/health" >/dev/null 2>&1 && break
  sleep 0.5
done
if ! curl -fsS --max-time 2 "$BASE/api/health" >/dev/null 2>&1; then
  echo "  FAIL  the server never came up"; tail -20 "$WORK/server.log"; exit 1
fi

rss_kb() { ps -o rss= -p "$SRV_PID" 2>/dev/null | tr -d ' '; }

api() {
  local m="$1" p="$2" t="$3" body="${4:-}"
  if [[ -n "$body" ]]; then
    curl -s --max-time 30 -X "$m" "$BASE$p" -H 'content-type: application/json' \
      ${t:+-H "authorization: Bearer $t"} -d "$body"
  else
    curl -s --max-time 30 -X "$m" "$BASE$p" ${t:+-H "authorization: Bearer $t"}
  fi
}
tokenOf() { echo "$1" | sed -n 's/.*"token":"\([^"]*\)".*/\1/p'; }

# The body is built in a variable, with single-quoted literal JSON spliced
# around the values, and NOT written as "{\"k\":\"$v\"}" inline.
#
# That inline form is a trap worth naming: inside "$( ... )" bash applies brace
# expansion to {"a":1,"b":2}, so the braces vanish and everything after the
# first comma becomes separate words. curl then receives a fragment, the server
# answers "send a JSON body", and nothing in the error mentions quoting.
jsonSignup() { # username dob
  printf '%s' '{"username":"'"$1"'","email":"'"$1"'@example.com","password":"correct-horse-7","dob":"'"$2"'","gender":"other"}'
}

signup() { # username dob -> token
  local body
  body="$(jsonSignup "$1" "$2")"
  tokenOf "$(api POST /api/auth/signup "" "$body")"
}

jsonPost() { # body-text
  printf '%s' '{"body":"'"$1"'"}'
}

echo
echo "── seeding ──"
ADULT="$(signup stressadult 1996-05-05)"
ADULT2="$(signup stressadult2 1994-05-05)"
MINOR="$(signup stressminor 2011-05-05)"
MINOR2="$(signup stressminor2 2010-05-05)"
if [[ -z "$ADULT" || -z "$MINOR" ]]; then
  echo "  FAIL  could not create the test accounts"
  echo "        raw: $(api POST /api/auth/signup "" "$(jsonSignup probe 1996-05-05)")"
  tail -20 "$WORK/server.log"
  exit 1
fi
ok "four accounts created"

START_RSS="$(rss_kb)"
echo "  note  resident memory at start: ${START_RSS} KB"

# Posts are written directly to SQLite rather than through the API: the point
# of this section is to measure the READ path against a large table, and
# paying for 1,200 moderation passes first would measure the write path.
python3 - "$WORK/stress.db" "$POSTS" <<'PY'
import sqlite3, sys
db, n = sys.argv[1], int(sys.argv[2])
con = sqlite3.connect(db)
cur = con.cursor()
cur.execute("SELECT id, band, dob FROM users ORDER BY id")
users = cur.fetchall()
rows = []
for i in range(n):
    uid, band, dob = users[i % len(users)]
    age = 30 if band == "adult" else 15
    rows.append((uid, f"stress post number {i} about chess and baking", band, age, "live"))
cur.executemany(
    "INSERT INTO posts (author_id, body, author_band, author_age_at_post, status) "
    "VALUES (?, ?, ?, ?, ?)", rows)
con.commit()
print(f"  note  {n} posts inserted directly")
con.close()
PY

echo
echo "── 1. the feed against a large table ──"
t0=$(python3 -c 'import time;print(int(time.time()*1000))')
FEED="$(api GET "/api/feed?limit=20" "$ADULT")"
t1=$(python3 -c 'import time;print(int(time.time()*1000))')
echo "  note  one adult feed request over $POSTS posts: $((t1-t0)) ms"
if [[ "$FEED" == *'"band":"adult"'* ]]; then ok "the feed still answers"; else bad "the feed broke under size"; fi
if [[ "$FEED" == *"stress post number"* ]]; then ok "and returns posts"; else bad "the feed returned nothing"; fi

echo
echo "── 2. the age wall under concurrency ──"
# Minor and adult traffic interleaved, every response checked. A shared or
# cached gate would show up here and nowhere else.
BREACH_FILE="$WORK/breaches"
: > "$BREACH_FILE"

check_worker() { # token forbidden-marker label
  local tok="$1" forbidden="$2" label="$3" i
  for ((i=0; i<REQUESTS/CONCURRENCY; i++)); do
    local body
    body="$(curl -s --max-time 30 "$BASE/api/feed?limit=20" -H "authorization: Bearer $tok")"
    if [[ "$body" == *"\"author_band\":\"$forbidden\""* || "$body" == *"\"band\":\"$forbidden\""* ]]; then
      echo "$label saw $forbidden" >> "$BREACH_FILE"
    fi
    # The decisive check: a minor must never see an adult's age, and vice versa.
    if [[ "$label" == "minor" && "$body" == *'"age":30'* ]]; then
      echo "minor saw an adult age" >> "$BREACH_FILE"
    fi
    if [[ "$label" == "adult" && "$body" == *'"age":15'* ]]; then
      echo "adult saw a minor age" >> "$BREACH_FILE"
    fi
  done
}

# The worker PIDs are collected and waited on BY NAME. A bare `wait` would also
# wait for the server, which was started with & in this same shell and never
# exits — so the script hung until the 600-second harness timeout, and the only
# reason an earlier run ever got past this point was that the server had died.
t0=$(python3 -c 'import time;print(int(time.time()*1000))')
WORKER_PIDS=()
for ((w=0; w<CONCURRENCY; w++)); do
  if (( w % 2 == 0 )); then
    check_worker "$ADULT" "minor" "adult" &
  else
    check_worker "$MINOR" "adult" "minor" &
  fi
  WORKER_PIDS+=("$!")
done
wait "${WORKER_PIDS[@]}"
t1=$(python3 -c 'import time;print(int(time.time()*1000))')
TOTAL=$(( (REQUESTS/CONCURRENCY) * CONCURRENCY ))
echo "  note  $TOTAL feed requests in $((t1-t0)) ms ($(( TOTAL * 1000 / (t1-t0+1) )) req/s)"

requireAlive "the concurrency run" || true

BREACHES="$(wc -l < "$BREACH_FILE" | tr -d ' ')"
if [[ "$BREACHES" == "0" ]]; then
  ok "ZERO age-wall breaches across $TOTAL concurrent mixed requests"
else
  bad "$BREACHES age-wall breaches under load"
  head -5 "$BREACH_FILE"
fi

echo
echo "── 3. memory ──"
END_RSS="$(rss_kb)"
echo "  note  resident memory after $TOTAL requests: ${END_RSS} KB (was ${START_RSS} KB)"
if [[ -n "$START_RSS" && -n "$END_RSS" ]]; then
  GROWTH=$(( END_RSS - START_RSS ))
  PER_REQ=$(( GROWTH * 1024 / (TOTAL > 0 ? TOTAL : 1) ))
  echo "  note  growth ${GROWTH} KB = about ${PER_REQ} bytes per request"
  # A genuine per-request leak shows up as megabytes over this many requests.
  # Some growth is expected: SQLite's page cache and sua's buffers both warm up.
  if (( GROWTH < 40000 )); then
    ok "memory growth is bounded (< 40 MB over $TOTAL requests)"
  else
    bad "memory grew by ${GROWTH} KB — that is a leak, not a warm cache"
  fi
else
  echo "  note  could not read resident memory on this platform; skipped"
fi

echo
echo "── 4. a report storm ──"
# One post, many reporters. The unique index makes a repeat a no-op; the
# threshold must fire once and the count must stay sane.
TARGET="$(api POST /api/posts "$ADULT2" "$(jsonPost 'a post that is about to be reported a great many times')")"
TARGET_ID="$(echo "$TARGET" | sed -n 's/.*"id":\([0-9]*\).*/\1/p' | head -1)"
if [[ -z "$TARGET_ID" ]]; then bad "could not create the target post"; else
  ok "target post #$TARGET_ID created"

  REMOVED=0
  for ((r=0; r<REPORTERS; r++)); do
    RT="$(signup "reporter$r" 1995-01-01)"
    [[ -z "$RT" ]] && continue
    # Give each reporter a post, so none is a throwaway and every report counts.
    api POST /api/posts "$RT" "$(jsonPost "reporter $r is a real user with a post")" >/dev/null
    RES="$(api POST "/api/posts/$TARGET_ID/report" "$RT" '{"reason":"harassment"}')"
    [[ "$RES" == *'"removed":true'* ]] && REMOVED=$((REMOVED+1))
  done
  echo "  note  $REPORTERS reporters, 'removed:true' returned $REMOVED time(s)"
  if (( REMOVED == 1 )); then
    ok "the takedown fired exactly once, not once per report"
  else
    bad "the takedown fired $REMOVED times"
  fi

  GONE="$(api GET "/api/posts/$TARGET_ID" "$ADULT")"
  if [[ "$GONE" == *"no such post"* ]]; then ok "and the post is gone"; else bad "the post survived the storm"; fi

  # Repeat reports from an account that already reported must stay no-ops.
  RT0="$(api POST /api/auth/login "" '{"username":"reporter0","password":"correct-horse-7"}')"
  RT0="$(tokenOf "$RT0")"
  for ((k=0; k<5; k++)); do
    api POST "/api/posts/$TARGET_ID/report" "$RT0" '{"reason":"harassment"}' >/dev/null
  done
  COUNT="$(python3 - "$WORK/stress.db" "$TARGET_ID" <<'PY'
import sqlite3, sys
con = sqlite3.connect(sys.argv[1])
n = con.execute("SELECT COUNT(*) FROM reports WHERE post_id = ? AND reporter_id = "
                "(SELECT id FROM users WHERE username = 'reporter0')",
                (int(sys.argv[2]),)).fetchone()[0]
print(n)
PY
)"
  if [[ "$COUNT" == "1" ]]; then
    ok "six reports from one account are still one row"
  else
    bad "one account produced $COUNT report rows"
  fi
fi

echo
echo "── 5. the server is still healthy ──"
H="$(api GET /api/health "")"
if [[ "$H" == *'"ok":true'* ]]; then ok "health is still green after everything"; else bad "the server is unwell"; fi

echo
echo "════════════════════════════════════════════════════════════════"
echo "  PASS: $PASS   FAIL: $FAIL"
if [[ "$FAIL" -eq 0 ]]; then echo "  RESULT: ALL GREEN"; exit 0; fi
echo "  RESULT: FAILED"; tail -20 "$WORK/server.log"; exit 1
