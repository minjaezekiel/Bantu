# ════════════════════════════════════════════════════════════════════
#  ChatBantu — Makefile for common dev tasks
#
#  Usage:
#    make run          # start the app on PORT (default 8080)
#    make docker       # start in Docker
#    make build        # rebuild the bantu binary from source
#    make reset-db     # delete chatbantu.db (will reseed on next start)
#    make logs         # tail server logs (when running in background)
#    make stop         # stop background server
#    make test         # curl the health endpoint + demo login
#    make clean        # remove all build + db artifacts
#    make help         # show all targets
# ════════════════════════════════════════════════════════════════════

PORT     ?= 8080
DB_PATH  ?= ./chatbantu.db
BANTU    := ./bantu
LOG_FILE := /tmp/chatbantu.log
PID_FILE := /tmp/chatbantu.pid

.PHONY: help run bg stop logs docker build reset-db test clean

help: ## Show this help
	@echo "ChatBantu — common dev tasks"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

run: ## Start ChatBantu in the foreground (Ctrl-C to stop)
	PORT=$(PORT) DB_PATH=$(DB_PATH) $(BANTU) run server.b

bg: ## Start ChatBantu in the background
	@echo "Starting on port $(PORT)…  (logs: tail -f $(LOG_FILE))"
	@PORT=$(PORT) DB_PATH=$(DB_PATH) nohup $(BANTU) run server.b > $(LOG_FILE) 2>&1 & \
	  echo $$! > $(PID_FILE)
	@sleep 1 && curl -fsS "http://localhost:$(PORT)/api/health" && echo "" \
	  && echo "✓ Running. PID=$$(cat $(PID_FILE))"

stop: ## Stop background server
	@if [ -f $(PID_FILE) ]; then \
	  kill $$(cat $(PID_FILE)) 2>/dev/null || true; \
	  rm -f $(PID_FILE); \
	  echo "✓ Stopped"; \
	else echo "(no PID file — nothing to stop)"; fi

logs: ## Tail server logs
	@tail -f $(LOG_FILE)

docker: ## Start in Docker (no local libs needed)
	PORT=$(PORT) docker compose -f docker-compose.dev.yml up --build

build: ## Rebuild the bantu binary from source
	cd bantu-src/compiler && ./build.sh
	cp -f bantu-src/compiler/build/bantu ./bantu
	chmod +x ./bantu
	@echo "✓ New binary: $$($(BANTU) --version)"

reset-db: ## Wipe chatbantu.db (reseeded on next start)
	rm -f chatbantu.db chatbantu.db-wal chatbantu.db-shm
	@echo "✓ Database reset"

test: ## Smoke test: health + login as demo user
	@echo "→ Health check…"
	@curl -fsS "http://localhost:$(PORT)/api/health" | head -c 200; echo ""
	@echo "→ Login as silivestir / bantu123…"
	@curl -fsS -X POST "http://localhost:$(PORT)/api/auth/login" \
	  -H "Content-Type: application/json" \
	  -d '{"username":"silivestir","password":"bantu123"}' \
	  | head -c 300; echo ""

clean: ## Remove all build artifacts + DB
	rm -f chatbantu.db chatbantu.db-wal chatbantu.db-shm $(LOG_FILE) $(PID_FILE)
	rm -rf bantu-src/compiler/build
	@echo "✓ Cleaned"
