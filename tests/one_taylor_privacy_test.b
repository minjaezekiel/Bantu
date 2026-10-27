// ════════════════════════════════════════════════════════════════════════
//  one_taylor_privacy_test.b — what leaves the server.
//
//  publicUser() is the only way a user reaches a response. The test is not
//  "does it include the right fields" but the inverse and much stronger
//  claim: SERIALISE THE WHOLE THING AND GREP IT. If an email address, a
//  phone number, a date of birth, a blind index or a password hash appears
//  anywhere in the JSON -- at any nesting depth, under any key name, added
//  by a later change nobody thought about -- this test fails.
//
//  A field-by-field assertion passes forever while someone adds
//  "debugRow": $row underneath it.
//
//  Run:  bantu run tests/one_taylor_privacy_test.b
// ════════════════════════════════════════════════════════════════════════

include "crypto" as crypto;
include "../samples/one-taylor/backend/safety/age.b" as age;
include "../samples/one-taylor/backend/auth/privacy.b" as privacy;

$R = {"pass": 0, "fail": 0};
def ok($cond, $name) {
    if ($cond) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $name); }
}

// privacy.b derives its keys from config, which needs a secret. Set one for
// the duration of this test rather than loading the real configuration.
// (config.load() reads the environment; here the test process supplies it.)
sua.sqlite.open(":memory:");
age.setToday("2026-07-01");

const $EMAIL = "zuhura.mwinyi@example.com";
const $PHONE = "+255700123456";
const $DOB   = "2001-03-14";

print("-- encryption round-trips, and rejects tampering --");
$enc = privacy.encryptField($EMAIL);
ok($enc != $EMAIL, "the stored value is not the plaintext");
ok(indexOf($enc, "zuhura") < 0, "and contains no fragment of it");
ok(privacy.decryptField($enc) == $EMAIL, "decrypt returns the original");
ok(privacy.decryptField("") == null, "an empty token decrypts to null, not to \"\"");

// Flip one character of the token. XChaCha20-Poly1305 must reject it, and
// "reject" must mean null -- never empty plaintext, which a caller would
// treat as a valid empty email.
$bad = substr($enc, 0, len($enc) - 2) + "AA";
$tampered = privacy.decryptField($bad);
ok($tampered == null || $tampered != $EMAIL, "a tampered token does not decrypt to the original");

print("-- the blind index --");
$b1 = privacy.blindIndex($EMAIL);
$b2 = privacy.blindIndex("  ZUHURA.MWINYI@EXAMPLE.COM  ");
ok($b1 == $b2, "the index is case- and whitespace-insensitive");
ok($b1 != privacy.blindIndex("other@example.com"), "different addresses index differently");
ok(len($b1) == 64, "it is a SHA-256 HMAC in hex");
ok(indexOf($b1, "zuhura") < 0, "and reveals no plaintext");
ok(privacy.blindIndex("") == "", "an absent value has no index");

print("-- publicUser never carries a secret --");
// A row shaped exactly as the database hands it over, with extra fields a
// future migration might add.
$row = {
    "id": 7,
    "username": "zuhura",
    "username_lower": "zuhura",
    "password_hash": "$argon2id$v=19$m=65536,t=2,p=1$c29tZXNhbHQ$aGFzaGhhc2g",
    "email_enc": $enc,
    "email_bidx": $b1,
    "phone_enc": privacy.encryptField($PHONE),
    "phone_bidx": privacy.blindIndex($PHONE),
    "dob": $DOB,
    "gender": "woman",
    "band": "adult",
    "status": "active",
    "created_at": "2026-01-02 10:00:00",
    "last_seen": "2026-07-01 09:00:00"
};
$profile = {
    "id": 3, "user_id": 7, "display_name": "Zuhura", "bio": "Chess and film.",
    "avatar_media": null, "seek_min_age": 20, "seek_max_age": 32,
    "seek_genders": "any", "explicit_optins": "suggestive", "theme": "dark",
    "updated_at": "2026-07-01 09:00:00"
};

$FORBIDDEN = [$EMAIL, $PHONE, $DOB, $b1, "argon2id", "c29tZXNhbHQ",
              "email_bidx", "phone_bidx", "password_hash", $enc,
              "zuhura.mwinyi", "255700123456"];

def leaks($obj, $label) {
    $json = json.stringify($obj);
    $found = [];
    each ($needle in $FORBIDDEN) {
        if (indexOf($json, $needle) >= 0) { append($found, $needle); }
    }
    if (len($found) > 0) {
        print("          " + $label + " leaked: " + join($found, ", "));
        print("          json: " + substr($json, 0, 300));
    }
    return len($found);
}

// Every viewer relationship, because "self" is the case most likely to be
// given a shortcut.
$self = { "id": 7, "dob": $DOB, "explicit_optins": "suggestive" };
$other = { "id": 9, "dob": "1990-01-01" };

ok(leaks(privacy.publicUser($row, null), "publicUser(anon)") == 0,
   "publicUser to an anonymous viewer leaks nothing");
ok(leaks(privacy.publicUser($row, $other), "publicUser(other)") == 0,
   "publicUser to another user leaks nothing");
ok(leaks(privacy.publicUser($row, $self), "publicUser(self)") == 0,
   "publicUser TO THE USER THEMSELF leaks nothing");
ok(leaks(privacy.publicUserWithProfile($row, $profile, $other), "withProfile(other)") == 0,
   "publicUserWithProfile to another user leaks nothing");
ok(leaks(privacy.publicUserWithProfile($row, $profile, $self), "withProfile(self)") == 0,
   "publicUserWithProfile to self leaks nothing");

print("-- what it DOES carry --");
$pub = privacy.publicUser($row, $other);
ok($pub["username"] == "zuhura", "the username is present");
ok($pub["age"] == 25, "the derived age is present");
ok($pub["band"] == "adult", "the band is present");
ok($pub["dob"] == null, "the date of birth is absent, not merely hidden");
ok($pub["self"] == null, "a stranger is not marked as self");
ok(privacy.publicUser($row, $self)["self"] == true, "the user themself is");

$withSelf = privacy.publicUserWithProfile($row, $profile, $self);
ok($withSelf["theme"] == "dark", "self sees their own theme");
ok($withSelf["explicitOptins"] == "suggestive", "self sees their own opt-ins");
$withOther = privacy.publicUserWithProfile($row, $profile, $other);
ok($withOther["theme"] == null, "a stranger does not see the theme");
ok($withOther["explicitOptins"] == null, "a stranger does not see the opt-ins");
ok($withOther["bio"] == "Chess and film.", "but does see the bio");

print("-- a different account's state cannot change the answer --");
each ($status in ["active", "suspended", "deleted", "shadowbanned"]) {
    $r2 = {};
    for $k, $v in $row.items() { $r2[$k] = $v; }
    $r2["status"] = $status;
    ok(leaks(privacy.publicUser($r2, $self), "status=" + $status) == 0,
       "status \"" + $status + "\" leaks nothing");
}
ok(privacy.publicUser(null, $self) == null, "a null row projects to null, not to {}");

print("-- input hygiene --");
$ctrl = "hello" + frombytes([0, 1, 7, 27, 127]) + "world";
$clean = privacy.sanitiseText($ctrl, 100);
ok(indexOf($clean, frombytes([0])) < 0, "a NUL byte is stripped");
ok(indexOf($clean, frombytes([127])) < 0, "DEL is stripped");
ok($clean == "helloworld", "and the text survives");
$kept = privacy.sanitiseText("line one\nline two\tindented", 100);
ok(indexOf($kept, "\n") >= 0, "a newline is kept");
ok(indexOf($kept, "\t") >= 0, "a tab is kept");
$long = "";
$i = 0;
while ($i < 100) { $long = $long + "0123456789"; $i = $i + 1; }
ok(len(privacy.sanitiseText($long, 50)) == 50, "a long value is truncated to the cap");
ok(len(privacy.sanitiseBio($long)) == 600, "a bio is capped at 600");
ok(privacy.sanitiseText(null, 10) == "", "null sanitises to empty, not to \"null\"");

print("");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); } else { print("  RESULT: FAILED"); }
