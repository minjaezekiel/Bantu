// ============================================================================
// one-taylor/backend/api/auth_routes.b — signup, login, logout, me
//
//   POST /api/auth/signup   {username, email, password, dob, gender, phone?}
//   POST /api/auth/login    {username, password}
//   POST /api/auth/logout
//   GET  /api/auth/me
//
// Signup asks for a BIRTH DATE, not an age. An age integer is wrong within a
// year and wrong forever after: a 17-year-old who typed "17" stays 17 in the
// database while turning 18, 19, 20 -- on the minor side of the wall, with
// minors, indefinitely. A date of birth is a fact; an age is a fact's expiry
// date. See docs/one-taylor-architecture.md 1.2.
//
// Exposes: register($sua)
// ============================================================================

include "orm" as orm;
include "../config.b" as cfg;
include "../models.b" as models;
include "../safety/age.b" as age;
include "../auth/privacy.b" as privacy;
include "../auth/passwords.b" as passwords;
include "../auth/sessions.b" as sessions;
include "./helpers.b" as h;

const $GENDERS = ["woman", "man", "non-binary", "other", "prefer-not-to-say"];

def _validUsername($u) {
    if (len($u) < 3 || len($u) > 20) { return "pick 3 to 20 characters"; }
    $i = 0;
    while ($i < len($u)) {
        $c = substr($u, $i, 1);
        $okChar = ($c >= "a" && $c <= "z") || ($c >= "A" && $c <= "Z")
               || ($c >= "0" && $c <= "9") || $c == "_" || $c == ".";
        if (!$okChar) { return "letters, numbers, dot and underscore only"; }
        $i = $i + 1;
    }
    $low = $u.lower();
    each ($r in ["admin", "root", "support", "onetaylor", "moderator", "help",
                 "system", "api", "me", "null", "undefined"]) {
        if ($low == $r) { return "that username is reserved"; }
    }
    return null;
}

def _validEmail($e) {
    $at = indexOf($e, "@");
    if ($at < 1) { return "that does not look like an email address"; }
    $rest = substr($e, $at + 1, len($e));
    if (indexOf($rest, ".") < 1) { return "that does not look like an email address"; }
    if (len($e) > 254) { return "that email address is too long"; }
    if (indexOf($e, " ") >= 0) { return "an email address cannot contain a space"; }
    return null;
}

def _validGender($g) {
    each ($x in $GENDERS) { if ($x == $g) { return true; } }
    return false;
}

def signupHandler($req, $res) {
    $body = h.bodyOf($req);
    if ($body == null) { return h.fail($res, 400, "send a JSON body"); }

    $username = h.strField($body, "username", 40);
    $email    = h.strField($body, "email", 300);
    $password = $body["password"];
    $dob      = h.strField($body, "dob", 10);
    $gender   = h.strField($body, "gender", 40);
    $phone    = h.strField($body, "phone", 40);

    $e = _validUsername($username);
    if ($e != null) { return h.fail($res, 400, $e); }
    $e = _validEmail($email);
    if ($e != null) { return h.fail($res, 400, $e); }
    if (!_validGender($gender)) {
        return h.fail($res, 400, "choose a gender from the list");
    }
    if (age.parseDob($dob) == null) {
        return h.fail($res, 400, "give your date of birth as YYYY-MM-DD");
    }
    $years = age.ageNow($dob);
    if ($years < 0) {
        return h.fail($res, 400, "that date of birth is in the future");
    }
    if (!age.isSignupAgeOk($dob)) {
        if ($years < cfg.CFG["minSignupAge"]) {
            return h.fail($res, 403, "you need to be at least "
                          + str(cfg.CFG["minSignupAge"]) + " to use One Taylor");
        }
        return h.fail($res, 400, "check that date of birth");
    }
    $pw = passwords.check($password);
    if (!$pw["ok"]) { return h.fail($res, 400, $pw["error"]); }

    // Uniqueness. The email check runs against the blind index, so the
    // plaintext never reaches a query.
    $lower = $username.lower();
    if (orm.exists(orm.filter(orm.query(models.M["users"]),
                              [["username_lower", $lower]]))) {
        return h.fail($res, 409, "that username is taken");
    }
    $bidx = privacy.blindIndex($email);
    if (orm.exists(orm.filter(orm.query(models.M["users"]), [["email_bidx", $bidx]]))) {
        // Deliberately the same shape of answer as a taken username. Saying
        // "that email already has an account" is an account-existence oracle
        // for anybody with a list of addresses -- but saying nothing at all
        // leaves a real user stuck, so it points at sign-in instead.
        return h.fail($res, 409, "that email cannot be used to sign up — try signing in");
    }

    $band = age.bandFor($dob);
    $ins = orm.insert(models.M["users"], {
        "username": $username,
        "username_lower": $lower,
        "password_hash": passwords.hash($password),
        "email_enc": privacy.encryptField($email),
        "email_bidx": $bidx,
        "phone_enc": privacy.encryptField($phone),
        "phone_bidx": privacy.blindIndex($phone),
        "dob": $dob,
        "gender": $gender,
        "band": $band
    });
    if (!$ins.ok) { return h.fail($res, 500, "could not create that account"); }
    $userId = $ins.id;

    $range = age.defaultRange($years);
    orm.insert(models.M["profiles"], {
        "user_id": $userId,
        "display_name": $username,
        "bio": "",
        "seek_min_age": $range["min"],
        "seek_max_age": $range["max"],
        "seek_genders": "any",
        "explicit_optins": "",
        "theme": "system"
    });

    $user = orm.first(orm.filter(orm.query(models.M["users"]), [["id", $userId]]));
    $s = sessions.issue($user);
    $res.status(201);
    return h.okJson($res, {
        "token": $s["token"],
        "expires": $s["expires"],
        "user": privacy.publicUser($user, $user),
        "band": $band,
        "range": $range
    });
}

def loginHandler($req, $res) {
    $body = h.bodyOf($req);
    if ($body == null) { return h.fail($res, 400, "send a JSON body"); }
    $username = h.strField($body, "username", 40);
    $password = $body["password"];
    if ($username == "" || $password == null) {
        return h.fail($res, 400, "username and password are both required");
    }

    $user = orm.first(orm.filter(orm.query(models.M["users"]),
                                 [["username_lower", $username.lower()]]));
    // One answer for "no such user" and "wrong password", and the hash is
    // verified either way so the two take the same time. A faster "no such
    // user" is a username oracle.
    $hash = "$argon2id$v=19$m=65536,t=2,p=1$AAAAAAAAAAAAAAAAAAAAAA$"
          + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
    if ($user != null) { $hash = $user["password_hash"]; }
    $good = passwords.verify($hash, $password);
    if ($user == null || !$good) {
        return h.fail($res, 401, "that username and password do not match");
    }
    if ($user["status"] != "active") {
        return h.fail($res, 403, "that account is not active");
    }

    // Crossing 18 is handled here as well as on resolve, so the very first
    // request after a birthday is already in the right band.
    sessions.transitionIfNeeded($user);
    $user = orm.first(orm.filter(orm.query(models.M["users"]), [["id", $user["id"]]]));

    orm.modify(orm.filter(orm.query(models.M["users"]), [["id", $user["id"]]]),
               {"last_seen": orm.raw("CURRENT_TIMESTAMP")});

    $s = sessions.issue($user);
    return h.okJson($res, {
        "token": $s["token"],
        "expires": $s["expires"],
        "user": privacy.publicUser($user, $user),
        "band": $user["band"]
    });
}

def logoutHandler($req, $res) {
    $token = sessions.bearerToken($req);
    sessions.revoke($token);
    return h.okJson($res, { "loggedOut": true });
}

def meHandler($req, $res) {
    $u = sessions.requireSession($req, $res);
    if ($u == null) { return null; }
    $p = orm.first(orm.filter(orm.query(models.M["profiles"]), [["user_id", $u["id"]]]));
    $tags = orm.all(orm.filter(orm.query(models.M["user_tags"]), [["user_id", $u["id"]]]));
    $interests = [];
    $hobbies = [];
    $wants = [];
    each ($t in $tags) {
        if ($t["kind"] == "interest") { append($interests, $t["tag"]); }
        else if ($t["kind"] == "hobby") { append($hobbies, $t["tag"]); }
        else if ($t["kind"] == "dream_partner") { append($wants, $t["tag"]); }
    }
    return h.okJson($res, {
        "user": privacy.publicUserWithProfile($u, $p, $u),
        "interests": sort($interests),
        "hobbies": sort($hobbies),
        "dreamPartner": sort($wants),
        "band": age.bandFor($u["dob"])
    });
}

def register($sua) {
    $sua.server.post("/api/auth/signup", signupHandler);
    $sua.server.post("/api/auth/login",  loginHandler);
    $sua.server.post("/api/auth/logout", logoutHandler);
    $sua.server.get("/api/auth/me",      meHandler);
}
