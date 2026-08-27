# Bantu `hash`

Cryptographic and non-cryptographic hashing, written **in pure Bantu** on the interpreter's native
32-bit primitives (`band/bxor/rotr/add32/…`, `bytes/tohex`). Every digest is **bit-exact to the
official RFC/NIST test vectors** and cross-checked against `shasum`/`md5`/`openssl`.

- **Source:** [`hash/hash.b`](../hash/hash.b) · **Tests:** [`hash/hash_test.b`](../hash/hash_test.b)

```bantu
include "./hash.b" as hash;

print(hash.sha256("abc"));   // ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
print(hash.md5("abc"));      // 900150983cd24fb0d6963f7d28e17f72
print(hash.hmac_sha256("key", "message"));   // hex MAC
```

```sh
bantu run hash/hash_test.b   # 25 vectors → RESULT: ALL GREEN
```

---

## ⚠ Security notice

**MD5 and SHA-1 are cryptographically broken** (practical collisions exist). This module ships them
for **checksums, legacy interop, and RFC-4122 UUID namespaces only**. Do **not** use them for
integrity against an adversary, digital signatures, passwords, or deduplicating untrusted data.
Use **SHA-256** and **HMAC-SHA256** for anything security-relevant.

The non-cryptographic hashes (`djb2`, `fnv1a`) are for hash tables / checksums and are **not**
collision-resistant.

---

## API

Every function accepts a **string or a byte-list** (a list of integers 0–255). The default form
returns a lowercase **hex string**; the `*_bytes` form returns a **byte-list**.

| Function | Returns | Notes |
|---|---|---|
| `hash.md5(x)` / `md5_bytes(x)` | hex / bytes | RFC 1321 · **non-secure** |
| `hash.sha1(x)` / `sha1_bytes(x)` | hex / bytes | RFC 3174 · **non-secure** |
| `hash.sha224(x)` / `sha224_bytes(x)` | hex / bytes | FIPS 180 |
| `hash.sha256(x)` / `sha256_bytes(x)` | hex / bytes | FIPS 180 |
| `hash.hmac_sha256(key, msg)` / `hmac_sha256_bytes(...)` | hex / bytes | RFC 2104 / 4231 |
| `hash.djb2(x)` | 32-bit number | non-crypto |
| `hash.fnv1a(x)` | 32-bit number | non-crypto |
| `hash.hash_file(path, algo)` | hex | `algo` ∈ `"md5"`,`"sha1"`,`"sha224"`,`"sha256"` |

Each `*_hex(x)` alias is also provided (identical to the bare form).

### Verifying a MAC safely
Compare MACs with the constant-time primitive to avoid timing side-channels:

```bantu
$expected = hash.hmac_sha256_bytes($key, $msg);
$actual   = fromhex($tagFromClient);
if (ct_equal($expected, $actual)) { /* authentic */ }
```

---

## How it works
The 64-bit SHA-512 family and AES/password-KDFs are intentionally **not** here (Bantu's numbers are
float64, so SHA-512's 64-bit words aren't exact; AES/KDFs belong in a vetted native layer). The
32-bit families (MD5/SHA-1/SHA-224/SHA-256) use only 32-bit words, which are exact in float64 once
the native `add32`/`mul32`/bitwise primitives provide defined wraparound.

**Performance.** These are pure-Bantu digests on a tree-walking interpreter — throughput is roughly
**~3 KB/s**. Ideal for the common cases (passwords, API keys, tokens, short messages, UUID
namespaces — all well under a kilobyte) and correct at any size, but **not suited to hashing large
files** (a 1 MB file takes minutes); shell out to a native tool for bulk file hashing. Correctness is
validated by differential fuzzing against Python's `hashlib` (480 digests over random inputs, zero
mismatches) and cross-checked against `openssl`/`shasum`/`md5`.
