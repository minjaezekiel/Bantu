// ============================================================================
// one-taylor/backend/models.b — schema
//
// ORM models over SQLite (orm/orm.b). Every column the visibility filter
// touches carries an index, because sua.sqlite never suspends: a slow query
// stalls the whole worker, not just its own request.
//
// Two columns on `posts` are denormalised on purpose -- author_band and
// author_age_at_post. The feed filters on them on every read and must not
// join `users` to do it; and more importantly a post should be filtered by WHO
// WROTE IT, WHEN. If a 17-year-old's posts became visible to adults the moment
// they turned 18, the age wall would leak backwards through history.
//
// Exposes:
//   openDb($path) -> conn
//   M              -> dict of models
//   migrate()      -> applied SQL
// ============================================================================

include "orm" as orm;

$M = {};
$DB = { "conn": null };

// Not "connect": that is a Bantu reserved word and `def connect(...)` does
// not parse. The ORM sidesteps the same set of words -- open, insert, modify,
// remove -- for the same reason.
def openDb($path) {
    $conn = orm.open({"driver": "sqlite", "path": $path});
    $DB["conn"] = $conn;

    // Two pragmas sua does not set for you, and the second one is the
    // difference between multi-worker mode working and failing.
    //
    // WAL lets readers run while a write is in flight. In the default rollback
    // journal a writer blocks every reader, so one person posting stalls
    // everyone else's feed -- on a single-threaded event loop that is the
    // whole server.
    //
    // busy_timeout is what makes more than one process possible at all. sua
    // holds one connection per worker process, and with no timeout a second
    // writer gets SQLITE_BUSY immediately rather than waiting the few
    // milliseconds the first one needs. Without this, sua.server.workers(n)
    // for n > 1 produces intermittent write failures under any load -- which
    // is worse than being slow, because it is a lost post rather than a late
    // one.
    sua.sqlite.exec("PRAGMA journal_mode=WAL");
    sua.sqlite.exec("PRAGMA busy_timeout=5000");
    sua.sqlite.exec("PRAGMA foreign_keys=ON");
    // NORMAL rather than FULL: with WAL this still survives a process crash,
    // and only loses the last transactions on an OS crash or power cut. For a
    // social feed that is the right trade; for a ledger it would not be.
    sua.sqlite.exec("PRAGMA synchronous=NORMAL");
    return $conn;
}

def pragmas() {
    return sua.sqlite.query(
        "SELECT (SELECT * FROM pragma_journal_mode) AS journal, "
      + "(SELECT * FROM pragma_busy_timeout) AS busy_timeout");
}

def conn() { return $DB["conn"]; }

def define() {
    $c = $DB["conn"];

    $M["users"] = orm.model($c, "users", [
        orm.pk("id"),
        orm.field("username",      "text", {"nullable": false, "unique": true}),
        orm.field("username_lower","text", {"nullable": false, "unique": true}),
        orm.field("password_hash", "text", {"nullable": false}),
        // Encrypted at rest. The *_bidx columns are HMAC blind indexes: an
        // encrypted column cannot be searched, so uniqueness checks and
        // "log in by email" run against the index and the plaintext never
        // appears in a query.
        orm.field("email_enc",     "text", {"nullable": false}),
        orm.field("email_bidx",    "text", {"nullable": false, "unique": true}),
        orm.field("phone_enc",     "text", {}),
        orm.field("phone_bidx",    "text", {}),
        orm.field("dob",           "text", {"nullable": false}),   // YYYY-MM-DD
        orm.field("gender",        "text", {"nullable": false}),
        orm.field("band",          "text", {"nullable": false}),   // minor | adult
        orm.field("status",        "text", {"defaultVal": "active"}),
        orm.field("created_at",    "datetime", {"defaultVal": "now"}),
        orm.field("last_seen",     "datetime", {"defaultVal": "now"})
    ]);

    $M["profiles"] = orm.model($c, "profiles", [
        orm.pk("id"),
        orm.fk("user_id", "users.id"),
        orm.field("display_name",  "text", {}),
        orm.field("bio",           "text", {"defaultVal": ""}),
        orm.field("avatar_media",  "text", {}),
        orm.field("seek_min_age",  "int",  {"nullable": false}),
        orm.field("seek_max_age",  "int",  {"nullable": false}),
        orm.field("seek_genders",  "text", {"defaultVal": "any"}),
        orm.field("explicit_optins", "text", {"defaultVal": ""}),
        orm.field("theme",         "text", {"defaultVal": "system"}),
        orm.field("updated_at",    "datetime", {"defaultVal": "now"})
    ]);

    // One row per tag, rather than a comma-joined string, so interest overlap
    // is a join and not a substring search.
    $M["user_tags"] = orm.model($c, "user_tags", [
        orm.pk("id"),
        orm.fk("user_id", "users.id"),
        orm.field("kind", "text", {"nullable": false}),   // interest|hobby|dream_partner
        orm.field("tag",  "text", {"nullable": false})
    ]);

    $M["posts"] = orm.model($c, "posts", [
        orm.pk("id"),
        orm.fk("author_id", "users.id"),
        orm.field("body",       "text", {"defaultVal": ""}),
        orm.field("author_band","text", {"nullable": false}),
        orm.field("author_age_at_post", "int", {"nullable": false}),
        orm.field("status",     "text", {"defaultVal": "live"}),  // live|removed|blocked|review
        orm.field("like_count", "int",  {"defaultVal": 0}),
        orm.field("report_count","int", {"defaultVal": 0}),
        orm.field("created_at", "datetime", {"defaultVal": "now"})
    ]);

    $M["post_media"] = orm.model($c, "post_media", [
        orm.pk("id"),
        orm.fk("post_id", "posts.id"),
        orm.field("path",   "text", {"nullable": false}),
        orm.field("mime",   "text", {"nullable": false}),
        orm.field("bytes",  "int",  {"defaultVal": 0}),
        orm.field("sha256", "text", {"nullable": false}),
        orm.field("position","int", {"defaultVal": 0})
    ]);

    $M["post_labels"] = orm.model($c, "post_labels", [
        orm.pk("id"),
        orm.fk("post_id", "posts.id"),
        orm.field("category",   "text", {"nullable": false}),
        orm.field("source",     "text", {"nullable": false}),   // author | model
        orm.field("confidence", "real", {"defaultVal": 0})
    ]);

    $M["post_tags"] = orm.model($c, "post_tags", [
        orm.pk("id"),
        orm.fk("post_id", "posts.id"),
        orm.field("tag", "text", {"nullable": false})
    ]);

    $M["follows"] = orm.model($c, "follows", [
        orm.pk("id"),
        orm.fk("follower_id", "users.id"),
        orm.fk("followee_id", "users.id"),
        orm.field("created_at", "datetime", {"defaultVal": "now"})
    ]);

    $M["likes"] = orm.model($c, "likes", [
        orm.pk("id"),
        orm.fk("post_id", "posts.id"),
        orm.fk("user_id", "users.id"),
        orm.field("created_at", "datetime", {"defaultVal": "now"})
    ]);

    $M["blocks"] = orm.model($c, "blocks", [
        orm.pk("id"),
        orm.fk("user_id", "users.id"),
        orm.fk("blocked_id", "users.id"),
        orm.field("created_at", "datetime", {"defaultVal": "now"})
    ]);

    // A single report hides the post from the reporter straight away, whatever
    // the community eventually decides.
    $M["hidden_posts"] = orm.model($c, "hidden_posts", [
        orm.pk("id"),
        orm.fk("user_id", "users.id"),
        orm.fk("post_id", "posts.id"),
        orm.field("reason", "text", {"defaultVal": ""})
    ]);

    $M["reports"] = orm.model($c, "reports", [
        orm.pk("id"),
        orm.fk("post_id", "posts.id"),
        orm.fk("reporter_id", "users.id"),
        orm.field("reason",  "text", {"nullable": false}),
        orm.field("counted", "int",  {"defaultVal": 1}),
        // A report that counts, but only once per post however many
        // throwaway accounts file one. See api/report_routes.b.
        orm.field("weak",    "int",  {"defaultVal": 0}),
        orm.field("created_at", "datetime", {"defaultVal": "now"})
    ]);

    // Append-only. A disputed takedown has to be reconstructable, and a
    // threshold change has to be evaluable against what already happened.
    $M["moderation_events"] = orm.model($c, "moderation_events", [
        orm.pk("id"),
        orm.field("subject_type", "text", {"nullable": false}),  // post|profile|user
        orm.field("subject_id",   "int",  {"defaultVal": 0}),
        orm.field("actor",        "text", {"defaultVal": "system"}),
        orm.field("decision",     "text", {"nullable": false}),
        orm.field("detail",       "text", {"defaultVal": ""}),
        orm.field("scores",       "text", {"defaultVal": ""}),
        orm.field("created_at",   "datetime", {"defaultVal": "now"})
    ]);

    // ── messaging ───────────────────────────────────────────────────────
    // A conversation stores its two participants as (low_id, high_id) rather
        // than (from, to), so a pair has exactly one row whichever of them
    // opens it. Without the ordering you get two conversations between the
    // same two people and neither sees the other's messages.
    $M["conversations"] = orm.model($c, "conversations", [
        orm.pk("id"),
        orm.fk("low_id", "users.id"),
        orm.fk("high_id", "users.id"),
        orm.field("band",           "text", {"nullable": false}),
        orm.field("last_message_at","datetime", {}),
        orm.field("last_preview",   "text", {"defaultVal": ""}),
        orm.field("created_at",     "datetime", {"defaultVal": "now"})
    ]);

    $M["messages"] = orm.model($c, "messages", [
        orm.pk("id"),
        orm.fk("conversation_id", "conversations.id"),
        orm.fk("sender_id", "users.id"),
        orm.field("body",       "text", {"defaultVal": ""}),
        // live | held | blocked — the same vocabulary as posts, for the same
        // reason: a refused message is kept so the decision can be reviewed.
        orm.field("status",     "text", {"defaultVal": "live"}),
        orm.field("kind",       "text", {"defaultVal": "text"}),  // text | call
        orm.field("read_at",    "datetime", {}),
        orm.field("created_at", "datetime", {"defaultVal": "now"})
    ]);

    // A one-shot, short-lived ticket that lets a browser open the WebSocket.
    //
    // The session token lives in an httpOnly cookie on the Next.js origin, so
    // the browser will not send it to the Bantu origin — and putting a session
    // token in a WebSocket URL would write it into every proxy log on the
    // path. The page asks its own server for a ticket, the ticket is good for
    // sixty seconds and one use, and only its HMAC is stored here.
    $M["ws_tickets"] = orm.model($c, "ws_tickets", [
        orm.pk("id"),
        orm.field("ticket_hmac", "text", {"nullable": false, "unique": true}),
        orm.fk("user_id", "users.id"),
        orm.field("expires_at", "int",  {"nullable": false}),
        orm.field("used",       "int",  {"defaultVal": 0}),
        orm.field("created_at", "datetime", {"defaultVal": "now"})
    ]);

    $M["sessions"] = orm.model($c, "sessions", [
        orm.pk("id"),
        orm.field("token_hmac", "text", {"nullable": false, "unique": true}),
        orm.fk("user_id", "users.id"),
        orm.field("band_at_issue", "text", {"nullable": false}),
        orm.field("expires_at", "int", {"nullable": false}),
        orm.field("created_at", "datetime", {"defaultVal": "now"})
    ]);

    return $M;
}

// Composite uniqueness and the read-path indexes.
//
// THE ONLY RAW SQL IN THIS APPLICATION, along with the pragmas above, and both
// for the same reason: orm.b models tables and rows, and an index is neither.
// Every query, insert, update and delete in the backend goes through the ORM.
//
// These are not optional extras. The unique index on (post_id, reporter_id) is
// what makes a repeat report a no-op rather than a second vote toward a
// takedown, and the composite index on posts(status, author_band,
// author_age_at_post, created_at) is what the age wall's feed query scans. The
// ORM declares uniqueness per column only, so leaving them out would mean
// enforcing them in Bantu -- which is a race between two requests, not a
// constraint.
$INDEXES = [
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_reports_post_reporter ON reports(post_id, reporter_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_follows_pair ON follows(follower_id, followee_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_likes_pair ON likes(post_id, user_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_blocks_pair ON blocks(user_id, blocked_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_hidden_pair ON hidden_posts(user_id, post_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_user_tag ON user_tags(user_id, kind, tag)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_profile_user ON profiles(user_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_label_pair ON post_labels(post_id, category)",
    "CREATE INDEX IF NOT EXISTS ix_posts_feed ON posts(status, author_band, author_age_at_post, created_at)",
    "CREATE INDEX IF NOT EXISTS ix_posts_author ON posts(author_id, created_at)",
    "CREATE INDEX IF NOT EXISTS ix_users_band ON users(band, status)",
    "CREATE INDEX IF NOT EXISTS ix_tags_tag ON user_tags(tag, kind)",
    "CREATE INDEX IF NOT EXISTS ix_media_post ON post_media(post_id, position)",
    "CREATE INDEX IF NOT EXISTS ix_labels_post ON post_labels(post_id)",
    "CREATE INDEX IF NOT EXISTS ix_follows_follower ON follows(follower_id)",
    "CREATE INDEX IF NOT EXISTS ix_follows_followee ON follows(followee_id)",
    "CREATE INDEX IF NOT EXISTS ix_sessions_token ON sessions(token_hmac)",
    "CREATE INDEX IF NOT EXISTS ix_modevents_subject ON moderation_events(subject_type, subject_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_conv_pair ON conversations(low_id, high_id)",
    "CREATE INDEX IF NOT EXISTS ix_conv_low ON conversations(low_id, last_message_at)",
    "CREATE INDEX IF NOT EXISTS ix_conv_high ON conversations(high_id, last_message_at)",
    "CREATE INDEX IF NOT EXISTS ix_msg_conv ON messages(conversation_id, id)",
    "CREATE INDEX IF NOT EXISTS ix_tickets_hmac ON ws_tickets(ticket_hmac)"
];

def migrate() {
    $models = [];
    for $k, $v in $M.items() { append($models, $v); }
    $plan = orm.migrate($models, {"dryRun": false});
    each ($sql in $INDEXES) { sua.sqlite.exec($sql); }
    return $plan;
}
