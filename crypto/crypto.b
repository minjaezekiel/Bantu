// ════════════════════════════════════════════════════════════════════════
//  crypto.b — Bantu cryptography module (secure random, HKDF, encoding)
//
//  Built on the native OS CSPRNG (randbytes) and the pure-Bantu hash module
//  (HMAC-SHA256). Everything security-sensitive draws from the CSPRNG — NEVER
//  from the `random` module (which is a predictable PRNG).
//
//  Include it namespaced:
//      include "./crypto.b" as crypto;
//      $token = crypto.token_urlsafe(32);          // a URL-safe secret
//      $mac   = crypto.hmac_sha256("k", "msg");    // hex MAC
//      $key   = crypto.hkdf_sha256($ikm, $salt, "context", 32);
//
//  ⚠ SCOPE: this module intentionally does NOT include symmetric encryption
//  (AES / ChaCha20) or password hashing (PBKDF2 / argon2). Those require a
//  vetted native implementation for both security and performance and are
//  out of scope for the pure-Bantu "Core" suite — do not hand-roll them.
// ════════════════════════════════════════════════════════════════════════

include "../hash/hash.b" as hash;

// ─── local helpers ──────────────────────────────────────────────────────
def _bytesOf($x) {
    if (type($x) == "string") { return bytes($x); }
    return $x;
}
def _cat($a, $b) {
    $r = $a;
    extend($r, $b);
    return $r;
}

$_B64    = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
$_B64URL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";


// ─── Secure randomness (OS CSPRNG) ──────────────────────────────────────

// random_bytes($n) → list of n cryptographically-secure bytes.
def random_bytes($n) { return randbytes($n); }

// token_hex($n) → 2n-char hex string of n secure bytes (like Python secrets.token_hex).
def token_hex($n) { return tohex(randbytes($n)); }

// token_urlsafe($n) → URL-safe base64 (no padding) of n secure bytes.
def token_urlsafe($n) { return _b64enc(randbytes($n), $_B64URL, false); }

// random_int($max) → an unbiased integer in [0, max) using rejection sampling.
def random_int($max) {
    if ($max <= 1) { return 0; }
    $limit = floor(4294967296 / $max) * $max;
    $v = 4294967296;
    while ($v >= $limit) {
        $rb = randbytes(4);
        $v = $rb[0] * 16777216 + $rb[1] * 65536 + $rb[2] * 256 + $rb[3];
    }
    return $v % $max;
}


// ─── HMAC + verification ────────────────────────────────────────────────

def hmac_sha256($key, $msg)       { return hash.hmac_sha256($key, $msg); }
def hmac_sha256_bytes($key, $msg) { return hash.hmac_sha256_bytes($key, $msg); }

// verify_hmac($key, $msg, $tag) → bool, constant-time. $tag may be a hex string
// or a byte-list. Always use this (not ==) to check a MAC.
def verify_hmac($key, $msg, $tag) {
    $expected = hash.hmac_sha256_bytes($key, $msg);
    $actual = $tag;
    if (type($tag) == "string") { $actual = fromhex($tag); }
    return ct_equal($expected, $actual);
}


// ─── HKDF-SHA256 (RFC 5869) ─────────────────────────────────────────────

def hkdf_extract($salt, $ikm) {
    return hash.hmac_sha256_bytes($salt, $ikm);
}

def hkdf_expand($prk, $info, $length) {
    $infob = _bytesOf($info);
    $okm = [];
    $t = [];
    $counter = 1;
    while (len($okm) < $length) {
        $block = _cat(_cat($t, $infob), [$counter]);
        $t = hash.hmac_sha256_bytes($prk, $block);
        extend($okm, $t);
        $counter = $counter + 1;
    }
    // truncate to the requested length
    $out = [];
    $i = 0;
    while ($i < $length) {
        $out[len($out)] = $okm[$i];
        $i = $i + 1;
    }
    return $out;
}

// hkdf_sha256($ikm, $salt, $info, $length) → derived key as a byte-list.
def hkdf_sha256($ikm, $salt, $info, $length) {
    $prk = hkdf_extract(_bytesOf($salt), _bytesOf($ikm));
    return hkdf_expand($prk, $info, $length);
}
def hkdf_sha256_hex($ikm, $salt, $info, $length) {
    return tohex(hkdf_sha256($ikm, $salt, $info, $length));
}


// ─── Encoding: hex + base64 + base64url (RFC 4648) ──────────────────────

def hex_encode($x) { return tohex(_bytesOf($x)); }
def hex_decode($s) { return fromhex($s); }

// _b64enc($input, $alpha, $pad) → base64 with the given 64-char alphabet.
def _b64enc($input, $alpha, $pad) {
    $b = _bytesOf($input);
    $out = "";
    $n = len($b);
    $i = 0;
    while ($i + 3 <= $n) {
        $v = $b[$i] * 65536 + $b[$i + 1] * 256 + $b[$i + 2];
        $out = $out + substr($alpha, band(shr($v, 18), 63), 1);
        $out = $out + substr($alpha, band(shr($v, 12), 63), 1);
        $out = $out + substr($alpha, band(shr($v, 6), 63), 1);
        $out = $out + substr($alpha, band($v, 63), 1);
        $i = $i + 3;
    }
    $rem = $n - $i;
    if ($rem == 1) {
        $v = $b[$i] * 65536;
        $out = $out + substr($alpha, band(shr($v, 18), 63), 1);
        $out = $out + substr($alpha, band(shr($v, 12), 63), 1);
        if ($pad) { $out = $out + "=="; }
    }
    if ($rem == 2) {
        $v = $b[$i] * 65536 + $b[$i + 1] * 256;
        $out = $out + substr($alpha, band(shr($v, 18), 63), 1);
        $out = $out + substr($alpha, band(shr($v, 12), 63), 1);
        $out = $out + substr($alpha, band(shr($v, 6), 63), 1);
        if ($pad) { $out = $out + "="; }
    }
    return $out;
}

// _b64dec($str, $alpha) → decode base64/base64url into a byte-list. Ignores
// padding and any non-alphabet characters (whitespace, newlines).
def _b64dec($str, $alpha) {
    $out = [];
    $acc = 0;
    $bits = 0;
    $L = len($str);
    $i = 0;
    while ($i < $L) {
        $v = indexOf($alpha, substr($str, $i, 1));
        if ($v >= 0) {
            $acc = $acc * 64 + $v;
            $bits = $bits + 6;
            if ($bits >= 8) {
                $rembits = $bits - 8;
                $div = pow(2, $rembits);
                $byte = floor($acc / $div);
                $out[len($out)] = $byte % 256;
                $acc = $acc - $byte * $div;
                $bits = $rembits;
            }
        }
        $i = $i + 1;
    }
    return $out;
}

def base64_encode($x)    { return _b64enc($x, $_B64, true); }
def base64_decode($s)    { return _b64dec($s, $_B64); }
def base64url_encode($x) { return _b64enc($x, $_B64URL, false); }
def base64url_decode($s) { return _b64dec($s, $_B64URL); }
