// ============================================================================
// one-taylor/backend/auth/passwords.b — argon2id
//
// crypto.hash_password is libsodium's argon2id: memory-hard, salted, and
// self-describing, so the parameters travel inside the stored string and a
// later increase does not invalidate existing hashes. Verification is
// constant-time.
//
// There is no fallback. PBKDF2 written in interpreted Bantu is either too slow
// to log in with at a secure iteration count, or too weak to store at a fast
// one, and shipping the fast one behind a comment is how a password database
// ends up crackable. config.b refuses to start the server without libsodium.
//
// Exposes:
//   check($password)  -> {"ok", "error"}
//   hash($password)   -> encoded argon2id string
//   verify($encoded, $password) -> bool
// ============================================================================

include "crypto" as crypto;

const $MIN_LEN = 10;
const $MAX_LEN = 200;

// The ten thousand most common passwords are not available offline, so this
// checks the shapes that list is made of: too short, all one character class,
// or the handful everybody tries first.
$COMMON = ["password", "password1", "12345678", "123456789", "qwertyuiop",
           "letmein", "iloveyou", "welcome1", "admin123", "onetaylor",
           "football", "princess", "qwerty123", "1q2w3e4r", "passw0rd"];

def check($password) {
    if ($password == null || type($password) != "string") {
        return { "ok": false, "error": "a password is required" };
    }
    if (len($password) < $MIN_LEN) {
        return { "ok": false,
                 "error": "use at least " + str($MIN_LEN) + " characters" };
    }
    if (len($password) > $MAX_LEN) {
        return { "ok": false, "error": "that password is too long" };
    }
    $low = $password.lower();
    each ($c in $COMMON) {
        if ($low == $c) {
            return { "ok": false, "error": "that password is too common" };
        }
    }
    // Variety, not a rule about symbols: "correct horse battery staple" should
    // pass and "Passw0rd!" should not get credit for its punctuation.
    $classes = 0;
    $hasLower = false;
    $hasUpper = false;
    $hasDigit = false;
    $hasOther = false;
    $i = 0;
    while ($i < len($password)) {
        $ch = substr($password, $i, 1);
        if ($ch >= "a" && $ch <= "z") { $hasLower = true; }
        else if ($ch >= "A" && $ch <= "Z") { $hasUpper = true; }
        else if ($ch >= "0" && $ch <= "9") { $hasDigit = true; }
        else { $hasOther = true; }
        $i = $i + 1;
    }
    if ($hasLower) { $classes = $classes + 1; }
    if ($hasUpper) { $classes = $classes + 1; }
    if ($hasDigit) { $classes = $classes + 1; }
    if ($hasOther) { $classes = $classes + 1; }
    if ($classes < 2 && len($password) < 16) {
        return { "ok": false,
                 "error": "mix in another kind of character, or use 16+ characters" };
    }
    return { "ok": true, "error": null };
}

def hash($password) { return crypto.hash_password($password); }

def verify($encoded, $password) {
    if ($encoded == null || $encoded == "" || $password == null) { return false; }
    return crypto.verify_password($encoded, $password);
}
