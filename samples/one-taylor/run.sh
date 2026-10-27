#!/usr/bin/env bash
# ============================================================================
#  run.sh — start One Taylor for local testing
#
#  Starts the Bantu backend on :4100 and the Next.js frontend on :3000.
#
#  The secrets live in .secrets.env, created on first run and git-ignored.
#  They are kept rather than regenerated because every key is derived from
#  ONE_TAYLOR_SECRET: change it and existing sessions die, stored email
#  addresses stop decrypting, and the blind indexes that enforce "one account
#  per email" no longer match.
#
#    bash run.sh            backend + frontend
#    bash run.sh backend    just the backend
#    bash run.sh frontend   just the frontend
# ============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BANTU="$ROOT/bantu"
WHAT="${1:-both}"

if [ ! -f "$HERE/.secrets.env" ]; then
  printf 'ONE_TAYLOR_SECRET=%s\nONE_TAYLOR_ADMIN_TOKEN=%s\n' \
    "$(openssl rand -hex 32)" "$(openssl rand -hex 24)" > "$HERE/.secrets.env"
  chmod 600 "$HERE/.secrets.env"
  echo "[run] created .secrets.env"
fi
set -a; . "$HERE/.secrets.env"; set +a

export PORT="${PORT:-4100}"
export ONE_TAYLOR_API="${ONE_TAYLOR_API:-http://127.0.0.1:$PORT}"

if [ "$WHAT" = "backend" ] || [ "$WHAT" = "both" ]; then
  if [ ! -f "$HERE/backend/model/toxicity.btw" ]; then
    echo "[run] no classifier at backend/model/toxicity.btw — the lexicon stage"
    echo "      will still run. Train one with:  bantu run lugha/train.b"
  fi
  ( cd "$HERE/backend" && exec "$BANTU" run server.b ) &
  echo "[run] backend  -> http://localhost:$PORT/api/health"
fi

if [ "$WHAT" = "frontend" ] || [ "$WHAT" = "both" ]; then
  [ -d "$HERE/web/node_modules" ] || ( cd "$HERE/web" && npm install )
  ( cd "$HERE/web" && exec npx next dev ) &
  echo "[run] frontend -> http://localhost:3000"
fi

wait
