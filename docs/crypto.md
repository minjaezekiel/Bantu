# Bantu `crypto`

Practical cryptography built on the OS CSPRNG (`randbytes`) and the pure-Bantu
[`hash`](hash.md) module: secure random values & tokens, HMAC-SHA256 with constant-time
verification, HKDF-SHA256 (RFC 5869), and base64/base64url/hex encoding.

- **Source:** [`crypto/crypto.b`](../crypto/crypto.b) · **Tests:** [`crypto/crypto_test.b`](../crypto/crypto_test.b)

```bantu
include "./crypto.b" as crypto;

$token = crypto.token_urlsafe(32);              // a URL-safe secret string
$mac   = crypto.hmac_sha256("key", "message");  // hex MAC
if (crypto.verify_hmac("key", "message", $mac)) { /* authentic */ }

$key = crypto.hkdf_sha256($ikm, $salt, "app v1 encryption key", 32);   // 32-byte derived key
```

```sh
bantu run crypto/crypto_test.b   # RFC 4648 + RFC 5869 vectors → ALL GREEN
```

---

## ⚠ Scope & security

- **Randomness comes from the OS CSPRNG only.** Never use the `random` module for keys, salts,
  nonces, tokens, or anything secret — it is a predictable PRNG.
- **This module has no symmetric encryption (AES/ChaCha) and no password hashing (PBKDF2/argon2).**
  Those require a vetted native implementation for security and performance; hand-rolling them in an
  interpreted language would be both slower and less safe. They belong in a future native layer.
- **Verify MACs with `verify_hmac`** (constant-time), never with `==` on the hex string.

---

## API

### Secure random
| Function | Returns |
|---|---|
| `crypto.random_bytes(n)` | list of `n` CSPRNG bytes |
| `crypto.token_hex(n)` | `2n`-char hex string of `n` secure bytes |
| `crypto.token_urlsafe(n)` | URL-safe base64 (no padding) of `n` secure bytes |
| `crypto.random_int(max)` | unbiased integer in `[0, max)` (rejection sampling) |

### Integrity
| Function | Returns |
|---|---|
| `crypto.hmac_sha256(key, msg)` / `_bytes` | hex / byte-list MAC |
| `crypto.verify_hmac(key, msg, tag)` | bool, constant-time (`tag` = hex string or bytes) |

### Key derivation
| Function | Returns |
|---|---|
| `crypto.hkdf_sha256(ikm, salt, info, length)` / `_hex` | derived key (bytes / hex), RFC 5869 |

### Encoding
| Function | Returns |
|---|---|
| `crypto.base64_encode/decode(x)` | standard base64 (with padding) |
| `crypto.base64url_encode/decode(x)` | URL-safe base64 (no padding) |
| `crypto.hex_encode/decode(x)` | hex |

All functions accept a **string or a byte-list**; decoders return a **byte-list** (use `frombytes`
to get a string back).
