# 💬 ChatBantu

A full-featured **social network** built entirely with the **Bantu programming language** and its **Sua** web framework — no Node.js, no Python, no other runtime in the backend. The only thing running on the server is the Bantu interpreter.

## What is this?

ChatBantu is a real-world test of whether Bantu can serve a production-style web app on its own. It can. The backend is a single `server.b` file interpreted by the native Bantu binary; the frontend is plain HTML, CSS, and JavaScript.

## Features

- **User accounts** — register, login, token-based auth
- **Social feed** — create posts, like, comment, share
- **People directory** — discover users, follow / unfollow
- **Real-time 1-to-1 chat** — polls the server every 1.5 s for new messages
- **Live presence** — heartbeat every 30 s, "online" badge on users
- **Real-time notifications** — likes, comments, follows, messages, calls
- **WebRTC video calls** — full signaling server (offer / answer / ICE) running on Bantu
- **Persistent SQLite** — data survives restarts (Render disk volume)

## Stack

| Layer        | Technology                                  |
|--------------|---------------------------------------------|
| Language     | [Bantu](https://github.com/AsseySilivestir/bantu-lang) v1.1.0 |
| Web framework| Sua (built into Bantu)                      |
| Database     | SQLite 3 (file-based)                       |
| Frontend     | Vanilla HTML / CSS / JS (no framework)      |
| Real-time    | HTTP polling (chat, presence, notifications)|
| Video calls  | WebRTC (browser) + Bantu signaling server   |
| NAT traversal| Google public STUN (`stun:stun.l.google.com`)|
| Deployment   | Docker → Render                             |

## Repo layout

```
ChatBantu/
├── server.b              ← entire backend (one Bantu file)
├── bantu                 ← native Linux x86_64 Bantu binary (rebuilt for Ubuntu 22.04)
├── bantu-src/            ← interpreter source (rebuildable via build.sh)
│   └── compiler/
│       ├── build.sh           ← one-command rebuild
│       ├── CMakeLists.txt
│       ├── src/               ← evaluator, lexer, parser, server, …
│       └── stubs/             ← GLIBCXX compatibility shim
├── dev.sh                ← one-command local dev launcher (Linux/Mac)
├── Makefile              ← common dev tasks (run, build, reset-db, test, …)
├── Dockerfile            ← production Dockerfile (multi-stage, builds Bantu)
├── Dockerfile.dev        ← local dev Dockerfile (uses prebuilt binary)
├── docker-compose.dev.yml← one-command local Docker dev
├── render.yaml           ← Render blueprint (free tier + 1 GB disk)
├── .env.example          ← environment variable template
├── .vscode/launch.json   ← VS Code F5 to start the server
├── QUICKSTART.md         ← 60-second quick start guide
├── .dockerignore
├── .gitignore
└── public/
    ├── index.html         ← login / register
    ├── feed.html          ← social feed (posts, likes, comments)
    ├── chat.html          ← real-time messaging
    ├── call.html          ← WebRTC video call
    ├── people.html        ← user directory
    ├── notifications.html ← notification feed
    ├── css/
    │   └── styles.css     ← full design system
    └── js/
        └── api.js         ← frontend API client + helpers
```

## API reference

All endpoints are JSON. Pass `Authorization: Bearer <token>` for authed routes.

| Method | Path                              | Purpose                              |
|--------|-----------------------------------|--------------------------------------|
| GET    | `/api/health`                     | Service health                       |
| POST   | `/api/auth/register`              | Create account                       |
| POST   | `/api/auth/login`                 | Sign in (returns token)              |
| GET    | `/api/auth/me`                    | Current user                         |
| GET    | `/api/users`                      | List users                           |
| POST   | `/api/users/:id/follow`           | Follow                               |
| DELETE | `/api/users/:id/follow`           | Unfollow                             |
| GET    | `/api/posts`                      | Feed (latest 100)                    |
| POST   | `/api/posts`                      | Create post                          |
| POST   | `/api/posts/:id/like`             | Toggle like                          |
| GET    | `/api/posts/:id/comments`         | List comments                        |
| POST   | `/api/posts/:id/comments`         | Add comment                          |
| GET    | `/api/messages/:id?since=N`       | Poll new messages                    |
| POST   | `/api/messages/:id`               | Send message                         |
| GET    | `/api/conversations`              | Conversation list                    |
| GET    | `/api/unread`                     | Unread counts                        |
| POST   | `/api/presence`                   | Heartbeat                            |
| GET    | `/api/presence`                   | Who's online                         |
| GET    | `/api/notifications`              | List + mark read                     |
| POST   | `/api/call/offer/:id`             | WebRTC: send SDP offer               |
| GET    | `/api/call/offer/:id`             | WebRTC: poll for incoming offer      |
| POST   | `/api/call/answer/:id`            | WebRTC: send SDP answer              |
| GET    | `/api/call/answer/:id`            | WebRTC: poll for answer              |
| POST   | `/api/call/ice/:id`               | WebRTC: send ICE candidate           |
| GET    | `/api/call/ice/:id`               | WebRTC: poll ICE candidates          |
| POST   | `/api/call/hangup/:id`            | WebRTC: end call                     |

## Run locally

ChatBantu ships with a one-command local dev launcher. See **`QUICKSTART.md`** for the full guide — short version below.

### Option A — Native (Linux x86_64, fastest)

```bash
git clone https://github.com/AsseySilivestir/ChatBantu.git
cd ChatBantu
chmod +x dev.sh
./dev.sh
```

The script auto-detects missing libraries and tells you exactly what to install:
```bash
sudo apt-get install -y libsqlite3-0 libcurl4 ca-certificates
```

Common flags:
```bash
./dev.sh --port 9000        # use a different port
./dev.sh --reset-db         # wipe chatbantu.db and reseed
./dev.sh --build            # rebuild the bantu binary from source first
./dev.sh --no-browser       # don't auto-open browser
./dev.sh --help             # show all options
```

### Option B — Docker (cross-platform, no system deps)

```bash
./dev.sh --docker
# or directly:
docker compose -f docker-compose.dev.yml up --build
```

### Option C — Makefile shortcuts

```bash
make run          # foreground server (Ctrl-C to stop)
make bg           # background server (logs: tail -f /tmp/chatbantu.log)
make stop         # stop background server
make test         # curl health + demo login
make build        # rebuild bantu from source
make reset-db     # wipe DB
make help         # show all targets
```

Open `http://localhost:8080` and sign in with the demo account:
- **username:** `silivestir`
- **password:** `bantu123`

## Deploy to Render

The repo includes a `render.yaml` blueprint. In Render:

1. **New → Blueprint** → connect this GitHub repo.
2. Render will detect `render.yaml` and create the `chatbantu` web service.
3. The Dockerfile installs `libsqlite3-0`, `libcurl-gnutls4`, and `ca-certificates`,
   copies the Bantu binary and the app, then runs `bantu run server.b`.
4. Render injects `$PORT` (read by `env("PORT")` in `server.b`).
5. SQLite is persisted at `/data/chatbantu.db` via a 1 GB disk.

Alternatively, deploy manually:

1. **New → Web Service → Docker** → connect this repo.
2. Set the Docker build context to the repo root.
3. Render auto-detects the `Dockerfile` and the `PORT` env var.

## How real-time works

Bantu's HTTP server is **mult-threaded and synchronous** 

- **Chat** — client polls `/api/messages/:id?since=<lastId>` every 1.5 s
- **Presence** — client POSTs `/api/presence` every 30 s; "online" = heartbeat within last 60 s
- **Notifications** — polled indirectly via the sidebar badge every 15 s
- **Video calls** — WebRTC peer connection is established in the browser; the Bantu backend only relays SDP offers/answers and ICE candidates (also polled)

This is enough for a small social network at modest scale. For higher throughput,
the Bantu interpreter would need a threaded accept loop — a small C++ change
in `evaluator.hpp` (the `bantuHandleHttpRequest` call site).

## Binary compatibility (Render / Ubuntu 22.04)

Render's free tier runs **Ubuntu 22.04 (jammy)** with **glibc 2.35** and
**libstdc++ from GCC 11** (max `GLIBCXX_3.4.30`). The Bantu interpreter is
developed on Debian 13 (glibc 2.41, GCC 14), so a naïve build will fail
on Render with:

```
bantu: /lib/x86_64-linux-gnu/libm.so.6: version `GLIBC_2.38' not found (required by bantu)
bantu: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by bantu)
```

The fix lives in `bantu-src/compiler/`:

1. **`src/evaluator.hpp`** — replaced `std::strtol` / `std::stoull` / `std::atoi`
   (which redirect to `__isoc23_strtol@GLIBC_2.38` under `_GNU_SOURCE`) with
   manual parsers. Replaced `std::fmod` (which pulls `fmod@GLIBC_2.38`) with
   a manual `a - floor(a/b) * b` implementation.
2. **`src/main.cpp`** — same treatment for `std::atoi`.
3. **`stubs/ios_base_library_initv.c`** — provides a no-op
   `_ZSt21ios_base_library_initv` so the binary doesn't require
   `GLIBCXX_3.4.32` (only present on GCC 14+ runtimes).
4. **`CMakeLists.txt` / `build.sh`** — dropped `-march=native`, use
   `-mtune=generic` for CPU portability across Render's instances.

After the fix the binary requires at most `GLIBC_2.34` and `GLIBCXX_3.4.9`,
both well within Ubuntu 22.04's runtime. Rebuild with:

```bash
cd bantu-src/compiler
./build.sh   # → produces ./build/bantu
```

## Why Bantu?

Bantu is a programming language designed for  developers who need speed, readability,fast deployment.
This project proves that Bantu + Sua + SQLite can serve a real, multi-user, real-time
app in production — not just demo scripts. Every line of backend code is `.b`, and the
only process running on the server is the `bantu` binary.

## License

MIT
