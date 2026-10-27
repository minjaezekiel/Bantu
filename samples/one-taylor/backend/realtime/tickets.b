// ============================================================================
// one-taylor/backend/realtime/tickets.b — how a browser authenticates a socket
//
// The session token lives in an httpOnly cookie on the Next.js origin. The
// WebSocket connects to the Bantu origin, so the browser will not send that
// cookie — and putting a session token in a WebSocket URL is worse than
// useless: a URL is written to every proxy and access log on the path, and
// WebSocket URLs are not redacted by anything.
//
// So the page asks its own server (which does hold the cookie) for a TICKET:
// 32 random bytes, good for sixty seconds and exactly one use, stored here
// only as an HMAC. The socket sends it in its first frame.
//
// Exposes:
//   issue($user)   -> {"ticket", "expires"}
//   redeem($t)     -> user id, or null
//   sweep()        -> how many expired rows were removed
// ============================================================================

include "crypto" as crypto;
include "orm" as orm;
include "../config.b" as cfg;
include "../models.b" as models;
include "../safety/age.b" as age;

const $TTL_SECONDS = 60;

def _hmac($ticket) {
    // Keyed from the root secret under its own HKDF label, so a ticket and a
    // session token are not interchangeable even though both are HMACs.
    return crypto.hmac_sha256(cfg.wsTicketPepper(), $ticket);
}

def issue($user) {
    sweep();
    $ticket = crypto.token_urlsafe(32);
    $expires = age.nowSeconds() + $TTL_SECONDS;
    orm.insert(models.M["ws_tickets"], {
        "ticket_hmac": _hmac($ticket),
        "user_id": $user["id"],
        "expires_at": $expires,
        "used": 0
    });
    return { "ticket": $ticket, "expires": $expires, "ttl": $TTL_SECONDS };
}

// Single use: the row is consumed before the caller gets the id back, so a
// replayed ticket finds nothing even if it arrives within the sixty seconds.
def redeem($ticket) {
    if ($ticket == null || $ticket == "") { return null; }
    $row = orm.first(orm.filter(orm.query(models.M["ws_tickets"]),
                     [["ticket_hmac", _hmac($ticket)]]));
    if ($row == null) { return null; }
    orm.remove(orm.filter(orm.query(models.M["ws_tickets"]),
               [["id", $row["id"]]]), false);
    if ($row["used"] != 0) { return null; }
    if ($row["expires_at"] < age.nowSeconds()) { return null; }
    return $row["user_id"];
}

def sweep() {
    $r = orm.remove(orm.filter(orm.query(models.M["ws_tickets"]),
                    [["expires_at__lt", age.nowSeconds()]]), false);
    if ($r == null || $r.changes == null) { return 0; }
    return $r.changes;
}
