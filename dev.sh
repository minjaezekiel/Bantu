#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
#  ChatBantu — Local Development Launcher
#  ────────────────────────────────────────────────────────────────────
#  One-command setup that:
#    1. Checks prerequisites (bantu binary + libs)
#    2. Sets sensible PORT + DB path defaults
#    3. (Optional) Rebuilds the Bantu binary from source
#    4. (Optional) Resets the SQLite database to a clean state
#    5. Starts the ChatBantu backend in the foreground
#    6. (Optional) Opens the browser to the running site
#
#  Usage:
#    ./dev.sh                  Run the app (default)
#    ./dev.sh --build          Rebuild Bantu from source first
#    ./dev.sh --reset-db       Wipe chatbantu.db before starting
#    ./dev.sh --port 9000      Use a custom port
#    ./dev.sh --no-browser     Don't auto-open the browser
#    ./dev.sh --docker         Run inside Docker (uses Dockerfile.dev)
#    ./dev.sh --help           Show this help
# ════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Resolve repo dir (where this script lives) ───────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ─────────────────────────────────────────────────────────
PORT="${PORT:-8080}"
DB_PATH="${DB_PATH:-./chatbantu.db}"
OPEN_BROWSER=1
REBUILD=0
RESET_DB=0
USE_DOCKER=0

# ── Parse args ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)         REBUILD=1; shift ;;
    --reset-db)      RESET_DB=1; shift ;;
    --port)          PORT="$2"; shift 2 ;;
    --no-browser)    OPEN_BROWSER=0; shift ;;
    --docker)        USE_DOCKER=1; shift ;;
    --help|-h)
      cat <<'EOF'
ChatBantu — Local Development Launcher

Usage:
  ./dev.sh                  Run the app (default)
  ./dev.sh --build          Rebuild Bantu from source first
  ./dev.sh --reset-db       Wipe chatbantu.db before starting
  ./dev.sh --port 9000      Use a custom port
  ./dev.sh --no-browser     Don't auto-open the browser
  ./dev.sh --docker         Run inside Docker (uses Dockerfile.dev)
  ./dev.sh --help           Show this help

Steps performed (native mode):
  1. Check bantu binary + shared libraries
  2. (Optional) Rebuild bantu from source
  3. (Optional) Reset SQLite DB
  4. Start the ChatBantu backend in the foreground
  5. (Optional) Open browser to http://localhost:$PORT

Environment variables:
  PORT      Port to listen on (default: 8080)
  DB_PATH   SQLite path (default: ./chatbantu.db)
EOF
      exit 0 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

export PORT
export DB_PATH

# ─────────────────────────────────────────────────────────────────────
#  Docker mode — hand off to docker compose
# ─────────────────────────────────────────────────────────────────────
if [[ "$USE_DOCKER" -eq 1 ]]; then
  echo "▶ Starting ChatBantu in Docker (port $PORT)…"
  if ! command -v docker >/dev/null 2>&1; then
    echo "✗ docker not found. Install Docker: https://docs.docker.com/get-docker/" >&2
    exit 1
  fi
  PORT="$PORT" docker compose -f docker-compose.dev.yml up --build
  exit 0
fi

# ─────────────────────────────────────────────────────────────────────
#  Native mode
# ─────────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ChatBantu — Local Dev                                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# ── 1. Check for bantu binary ────────────────────────────────────────
BANTU_BIN="$SCRIPT_DIR/bantu"
if [[ ! -x "$BANTU_BIN" ]]; then
  echo "✗ Bantu binary not found at: $BANTU_BIN"
  echo
  echo "You can rebuild it from source:"
  echo "  ./dev.sh --build"
  echo
  echo "Or run in Docker (no local build needed):"
  echo "  ./dev.sh --docker"
  exit 1
fi

# ── 2. Check shared libraries ────────────────────────────────────────
echo "▶ Checking shared libraries…"
MISSING_LIBS=()
ldd "$BANTU_BIN" 2>/dev/null | grep -E "not found" && {
  echo "✗ Missing shared libraries detected. Install them:"
  echo
  echo "  Debian/Ubuntu:  sudo apt-get install -y libsqlite3-0 libcurl4 ca-certificates"
  echo "  Fedora:         sudo dnf install -y libsqlite3x libcurl ca-certificates"
  echo "  Alpine:         apk add sqlite-libs curl ca-certificates"
  echo "  macOS:          brew install sqlite curl ca-certificates"
  echo
  echo "Or run in Docker instead:  ./dev.sh --docker"
  exit 1
} || echo "  ✓ All shared libraries available"

# ── 3. Optional: rebuild from source ─────────────────────────────────
if [[ "$REBUILD" -eq 1 ]]; then
  echo
  echo "▶ Rebuilding Bantu from source…"
  if [[ ! -d "$SCRIPT_DIR/bantu-src/compiler" ]]; then
    echo "✗ bantu-src/compiler/ not found — cannot rebuild."
    exit 1
  fi
  cd "$SCRIPT_DIR/bantu-src/compiler"
  if [[ ! -x ./build.sh ]]; then chmod +x ./build.sh; fi
  ./build.sh
  cp -f ./build/bantu "$SCRIPT_DIR/bantu"
  chmod +x "$SCRIPT_DIR/bantu"
  cd "$SCRIPT_DIR"
  echo "  ✓ New bantu binary installed: $(./bantu --version 2>&1 | head -1)"
fi

# ── 4. Optional: reset DB ────────────────────────────────────────────
if [[ "$RESET_DB" -eq 1 ]]; then
  echo
  echo "▶ Resetting database…"
  rm -f "$SCRIPT_DIR/chatbantu.db" "$SCRIPT_DIR/chatbantu.db-wal" "$SCRIPT_DIR/chatbantu.db-shm"
  echo "  ✓ Deleted chatbantu.db"
fi

# ── 5. Show runtime info ─────────────────────────────────────────────
echo
echo "──────────────────────────────────────────────────────────────────"
echo "  Binary:   $BANTU_BIN"
echo "  Version:  $("$BANTU_BIN" --version 2>&1 | head -1)"
echo "  App:      $SCRIPT_DIR/server.b"
echo "  Port:     $PORT"
echo "  DB:       $DB_PATH"
echo "  URL:      http://localhost:$PORT"
echo "  Demo user: silivestir / bantu123"
echo "──────────────────────────────────────────────────────────────────"
echo

# ── 6. Open browser (in background, after a short delay) ─────────────
if [[ "$OPEN_BROWSER" -eq 1 ]]; then
  (sleep 1.5 && {
    URL="http://localhost:$PORT"
    if command -v xdg-open >/dev/null 2>&1; then
      xdg-open "$URL" >/dev/null 2>&1 || true
    elif command -v open >/dev/null 2>&1; then
      open "$URL" >/dev/null 2>&1 || true
    fi
  }) &
fi

# ── 7. Launch! (foreground — Ctrl-C to stop) ─────────────────────────
exec "$BANTU_BIN" run "$SCRIPT_DIR/server.b"
