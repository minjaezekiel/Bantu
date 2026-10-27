// ============================================================================
// one-taylor/backend/config.b — configuration and derived keys
//
// Every secret in the app descends from ONE_TAYLOR_SECRET by HKDF with a
// purpose-specific info string, so a key used for one thing cannot be
// substituted for another, and rotating the root rotates everything.
//
// The server REFUSES TO START without that variable. A default development
// key is the single most common way an application ships with a known secret,
// and "it was only for dev" is not a property the deployed binary knows about.
//
// Exposes:
//   CFG         — the resolved configuration
//   sessionPepper() / blindPepper() / fieldKey() / wsTicketPepper()
//   requireHealthy()
// ============================================================================

include "crypto" as crypto;

$CFG = {
    "port": 4100,
    "dbPath": "one-taylor.db",
    "uploadDir": "./uploads",
    "modelPath": "./model/toxicity.btw",
    "vocabPath": "./model/vocab.txt",
    "maxBodyBytes": 12582912,          // 12 MB: a 4 MB image is ~5.5 MB base64
    "sessionDays": 30,
    "reportThreshold": 2,
    "minSignupAge": 13,
    "adultAge": 18,
    "minorSpread": 2,                  // minors see +/- this many years
    "adultDefaultSpread": 10,
    "moderationReady": false
};

$KEYS = { "root": null };

def _envOr($name, $dflt) {
    $v = env($name);
    if ($v == null || $v == "") { return $dflt; }
    return $v;
}

def load() {
    $secret = env("ONE_TAYLOR_SECRET");
    if ($secret == null || len($secret) < 32) {
        print("");
        print("  ONE_TAYLOR_SECRET is missing or shorter than 32 characters.");
        print("");
        print("  Every key this app uses is derived from it: session tokens,");
        print("  the blind indexes that make encrypted emails searchable, and");
        print("  the key that encrypts emails and phone numbers at rest.");
        print("  There is no development default, on purpose.");
        print("");
        print("    export ONE_TAYLOR_SECRET=$(openssl rand -hex 32)");
        print("");
        throw "config: ONE_TAYLOR_SECRET not set";
    }
    if (!crypto.encryption_available()) {
        print("");
        print("  This interpreter was built without libsodium, so there is no");
        print("  argon2id to hash passwords with and no AEAD to encrypt email");
        print("  addresses with. Both are load-bearing here.");
        print("");
        print("    BANTU_SODIUM=1 bash bantu-src/compiler/build-mac.sh");
        print("");
        throw "config: libsodium not available (crypto.encryption_available() is false)";
    }

    $KEYS["root"] = $secret;
    $CFG["port"] = floor(num(_envOr("PORT", "4100")));
    $CFG["dbPath"] = _envOr("ONE_TAYLOR_DB", $CFG["dbPath"]);
    $CFG["modelPath"] = _envOr("ONE_TAYLOR_MODEL", $CFG["modelPath"]);
    $CFG["vocabPath"] = _envOr("ONE_TAYLOR_VOCAB", $CFG["vocabPath"]);
    return $CFG;
}

// Separate info strings: the same root, used for three different jobs, must
// not produce the same bytes. RFC 5869 calls this context binding.
def sessionPepper() {
    return crypto.hkdf_sha256($KEYS["root"], "one-taylor", "session token v1", 32);
}

def blindPepper() {
    return crypto.hkdf_sha256($KEYS["root"], "one-taylor", "blind index v1", 32);
}

def fieldKey() {
    return crypto.hkdf_sha256($KEYS["root"], "one-taylor", "field encryption v1", 32);
}

// A separate label again: a WebSocket ticket and a session token must not be
// interchangeable, and they would be if both were HMAC'd under the same key.
def wsTicketPepper() {
    return crypto.hkdf_sha256($KEYS["root"], "one-taylor", "ws ticket v1", 32);
}
