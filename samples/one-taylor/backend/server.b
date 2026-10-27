// ============================================================================
// One Taylor — backend entry point
// ----------------------------------------------------------------------------
//   Backend   Bantu 1.3.2 + sua (HTTP) + orm (SQLite) + lugha (transformer)
//   Frontend  Next.js 15 + TypeScript, in ../web
//   Design    docs/one-taylor-architecture.md
//
//   ONE_TAYLOR_SECRET=$(openssl rand -hex 32) bantu run server.b
//
// Startup order matters and is not arbitrary:
//   1. config  — refuses to continue without a secret or without libsodium
//   2. db      — open, define models, migrate, create indexes
//   3. model   — load the classifier; a failure here degrades, not dies
//   4. routes  — register every handler
//   5. listen
// ============================================================================

include "./config.b" as cfg;
include "./models.b" as models;
include "./safety/age.b" as age;
include "./safety/moderation.b" as moderation;
include "./api/auth_routes.b" as authRoutes;
include "./api/post_routes.b" as postRoutes;
include "./api/user_routes.b" as userRoutes;
include "./api/social_routes.b" as socialRoutes;
include "./api/report_routes.b" as reportRoutes;
include "./api/feed_routes.b" as feedRoutes;
include "./api/media_routes.b" as mediaRoutes;
include "./api/admin_routes.b" as adminRoutes;
include "./api/realtime_routes.b" as realtimeRoutes;
include "./realtime/hub.b" as hub;

def _inList($list, $v) {
    each ($x in $list) { if ($x == $v) { return true; } }
    return false;
}

print("");
print("  ┌──────────────────────────────────────────────┐");
print("  │  One Taylor                                  │");
print("  │  age-safe social + dating, backend in Bantu  │");
print("  └──────────────────────────────────────────────┘");
print("");

// ── 1. configuration ────────────────────────────────────────────────────────
cfg.load();
print("  [config] secret loaded, libsodium present (argon2id + AEAD)");

// ── 2. database ─────────────────────────────────────────────────────────────
models.openDb(cfg.CFG["dbPath"]);
models.define();
$plan = models.migrate();
$pg = models.pragmas();
$journal = "?";
if ($pg != null && len($pg) > 0) { $journal = str($pg[0]["journal"]); }
print("  [db]     " + cfg.CFG["dbPath"] + " ready, " + str(len($plan))
      + " migration statements applied, journal=" + $journal);
print("  [db]     today is " + age.today());

// ── 3. the classifier ───────────────────────────────────────────────────────
// A failure to load the model is not a failure to start. The deterministic
// lexicon still runs, and it is the stage with perfect recall on what it
// lists. Running degraded and SAYING SO beats refusing to serve -- but it has
// to be said, so it appears here and in /api/health.
$mod = moderation.init(cfg.CFG["modelPath"], cfg.CFG["vocabPath"], "./safety/lexicons");
if ($mod["ready"]) {
    print("  [model]  lugha loaded: " + str($mod["model"]["layers"])
          + " layers, d_model " + str($mod["model"]["d_model"])
          + ", vocab " + str($mod["model"]["vocab"]));
    // Which labels the model is allowed to act on, printed at startup rather
    // than buried: an operator should not have to read the code to find out
    // that grooming is advisory on this build.
    print("  [model]  may block on: " + join($mod["model"]["blocking"], ", "));
    $adv = [];
    each ($l in moderation.allLabels()) {
        if (!_inList($mod["model"]["blocking"], $l)) { append($adv, $l); }
    }
    if (len($adv) > 0) {
        print("  [model]  advisory only: " + join($adv, ", ")
              + "   (these reach the review queue, they never block a post)");
    }
} else {
    print("  [model]  *** CLASSIFIER UNAVAILABLE ***");
    print("           " + $mod["error"]);
    print("           The lexicon stage still runs. Train the model with:");
    print("             bantu run lugha/train.b");
}
$lexTotal = 0;
for $k, $v in $mod["lexicon"].items() { $lexTotal = $lexTotal + $v; }
print("  [model]  lexicon: " + str($lexTotal) + " terms across "
      + str(len(keys($mod["lexicon"]))) + " categories");

// ── 4. routes ───────────────────────────────────────────────────────────────
authRoutes.register(sua);
postRoutes.register(sua);
userRoutes.register(sua);
socialRoutes.register(sua);
reportRoutes.register(sua);
feedRoutes.register(sua);
mediaRoutes.register(sua);
adminRoutes.register(sua);
realtimeRoutes.register(sua);
print("  [routes] " + str(len(sua.server.routes())) + " routes registered");

// ── 4b. the WebSocket: presence, chat, and call signalling ──────────────────
hub.install();
print("  [ws]     presence, direct messages and WebRTC signalling installed");

// ── 5. listen ───────────────────────────────────────────────────────────────
// A base64 image needs about 1.34x its own size in the body, so the cap is
// raised from sua's 8 MB default to fit the client's 4 MB-per-image limit.
sua.server.limits({ "max_body_bytes": cfg.CFG["maxBodyBytes"] });

// The browser talks to Next.js on :3000 and to this server directly for the
// WebSocket, so the upgrade is cross-origin and has to be allowed explicitly.
//
// It is an ALLOWLIST and not a wildcard on purpose. sua checks Origin against
// Host, and both are sent by the client: without a list, any page on the
// internet could open a WebSocket to a user's localhost server and, if it ever
// got a ticket, read their messages. ONE_TAYLOR_ORIGINS overrides it in
// production.
$origins = ["http://localhost:3000", "http://127.0.0.1:3000"];
$envOrigins = env("ONE_TAYLOR_ORIGINS");
if ($envOrigins != null && $envOrigins != "") {
    $origins = [];
    each ($o in split($envOrigins, ",")) {
        $t2 = trim($o);
        if ($t2 != "") { append($origins, $t2); }
    }
}
sua.server.limits({
    "ws_check_origin": true,
    "ws_allowed_origins": $origins,
    // A signalling frame is an SDP at worst, a few kilobytes. 256 KB is
    // already generous and makes "exhaust memory one frame at a time"
    // uninteresting.
    "max_ws_frame_bytes": 262144,
    "max_ws_message_bytes": 262144
});
print("  [ws]     origins: " + join($origins, ", "));

// Workers are OPT-IN, and the default of one is a deliberate choice rather
// than an oversight.
//
// Each sua worker is a separate process with its own event loop and its own
// SQLite connection, so n workers is close to n times the throughput on a
// read-heavy workload like this one. What it costs: SQLite then has n writers,
// which only works because models.b sets WAL and a busy timeout -- without
// those a second concurrent write fails outright. Multi-worker is also
// POSIX-only, and every global becomes per-worker, which is safe here because
// all shared state is in the database.
//
// Set ONE_TAYLOR_WORKERS=0 for one worker per core, or a number for exactly
// that many.
$workers = env("ONE_TAYLOR_WORKERS");
if ($workers != null && $workers != "") {
    $n = floor(num($workers));
    sua.server.workers($n);
    if ($n == 0) { print("  [server] one worker per core"); }
    else { print("  [server] " + str($n) + " workers"); }
}

// uploads/ is deliberately NOT a static root -- see api/media_routes.b. The
// only static directory is the API's own landing page.
sua.server.static("./public");

print("");
print("  API      http://localhost:" + str(cfg.CFG["port"]) + "/api/health");
print("  WS       ws://localhost:" + str(cfg.CFG["port"]) + "/   (messages + calls)");
print("  Frontend cd ../web && npm run dev   ->  http://localhost:3000");
print("");
print("  No TLS: sua.server has no certificate option. Put a terminator in");
print("  front for anything beyond localhost -- secure cookies AND voice or");
print("  video calls need a secure context, because getUserMedia is only");
print("  granted in one. http://localhost is secure; a LAN address is not,");
print("  so a phone pointed at this machine can message but cannot call.");
print("");
print("  Calls are peer to peer. This server relays the offer, the answer and");
print("  the ICE candidates and never sees a frame of media. Two peers behind");
print("  symmetric NAT need a TURN relay: run `bantu relay` and set");
print("  ONE_TAYLOR_TURN=turn://host:port.");
print("");
sua.server.listen(cfg.CFG["port"]);
