// ============================================================================
// one-taylor/backend/auth/privacy.b — what leaves the server, and what never does
//
// Email addresses and phone numbers are encrypted at rest with
// XChaCha20-Poly1305 under a key derived from ONE_TAYLOR_SECRET. An encrypted
// column cannot be searched, so each has a BLIND INDEX next to it:
// HMAC-SHA256(pepper, lowercased value), unique. Uniqueness checks and
// "log in by email" run against the index and the plaintext never appears in
// a query, a log or an error message.
//
// publicUser() is the ONLY way a user reaches a response anywhere in this API.
// email_enc, email_bidx, phone_enc, phone_bidx, dob, password_hash and status
// are not filtered out of it -- they are never put in. A route cannot leak a
// field it never holds, which is a stronger guarantee than a route that
// remembers to delete it.
//
// Exposes:
//   encryptField($plain) / decryptField($token)
//   blindIndex($plain)
//   publicUser($row, $viewer) / publicUserWithProfile($row, $profile, $viewer)
//   sanitiseBio($s) / sanitiseText($s)
// ============================================================================

include "crypto" as crypto;
include "../config.b" as cfg;
include "../safety/age.b" as age;

def encryptField($plain) {
    if ($plain == null || $plain == "") { return ""; }
    return crypto.encrypt(cfg.fieldKey(), $plain, "one-taylor/field");
}

def decryptField($token) {
    if ($token == null || $token == "") { return null; }
    $pt = crypto.decrypt(cfg.fieldKey(), $token, "one-taylor/field");
    // null means the ciphertext was tampered with or the key is wrong. It is a
    // rejection, never an empty string.
    if ($pt == null) { return null; }
    return frombytes($pt);
}

def blindIndex($plain) {
    if ($plain == null || $plain == "") { return ""; }
    return crypto.hmac_sha256(cfg.blindPepper(), trim($plain).lower());
}

// ── the projection ──────────────────────────────────────────────────────────
// $viewer may be null (an unauthenticated context); nothing here depends on
// it except whether the "self" fields are included.
def publicUser($row, $viewer) {
    if ($row == null) { return null; }
    $out = {
        "id": $row["id"],
        "username": $row["username"],
        "band": $row["band"],
        "age": age.ageNow($row["dob"]),
        "gender": $row["gender"],
        "joined": $row["created_at"]
    };
    if ($viewer != null && $viewer["id"] == $row["id"]) {
        // Even to yourself: the age, never the date of birth. A birth date is
        // identifying in a way an age is not, and this response is one
        // screenshot away from being public.
        $out["self"] = true;
    }
    return $out;
}

def publicUserWithProfile($row, $profile, $viewer) {
    $u = publicUser($row, $viewer);
    if ($u == null) { return null; }
    if ($profile != null) {
        $u["displayName"] = $profile["display_name"];
        $u["bio"] = $profile["bio"];
        $u["avatar"] = $profile["avatar_media"];
        $u["seekMinAge"] = $profile["seek_min_age"];
        $u["seekMaxAge"] = $profile["seek_max_age"];
        $u["seekGenders"] = $profile["seek_genders"];
        if ($viewer != null && $viewer["id"] == $row["id"]) {
            $u["theme"] = $profile["theme"];
            $u["explicitOptins"] = $profile["explicit_optins"];
        }
    }
    return $u;
}

// ── input hygiene ───────────────────────────────────────────────────────────
// The frontend renders text as text, never as HTML, so this is not an XSS
// defence -- it is a storage one. Control characters and unbounded length are
// how a post breaks every consumer downstream of it, including the moderation
// classifier's tokenizer.
def sanitiseText($s, $maxLen) {
    if ($s == null) { return ""; }
    $b = bytes($s);
    $out = [];
    $i = 0;
    while ($i < len($b)) {
        $c = $b[$i];
        // Keep newline and tab; drop every other C0 control and DEL.
        if ($c == 10 || $c == 9 || $c >= 32) {
            if ($c != 127) { append($out, $c); }
        }
        $i = $i + 1;
    }
    $t = trim(frombytes($out));
    if (len($t) > $maxLen) { $t = substr($t, 0, $maxLen); }
    return $t;
}

def sanitiseBio($s) { return sanitiseText($s, 600); }
