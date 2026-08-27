# Crypto Suite — Design Decisions

Rationale for the choices behind the `hash`/`crypto`/`uuid` modules, so they aren't re-litigated.
Each entry: **decision · why · alternatives rejected · security implication.**

---

### D1 — Engine: native primitives + pure-Bantu algorithms
**Decision:** Add general-purpose native primitives (bitwise, byte access, CSPRNG) to the
interpreter; implement the hash/UUID **algorithms in Bantu** on top.
**Why:** Bantu structurally cannot express crypto today (no bitwise ops, no `ord()`, ASCII-only
`chr()`, no bytes, float64-only). Adding *primitives* (not a C++ crypto library) keeps the
algorithms auditable and genuinely "written in Bantu," while making them correct and fast enough.
**Rejected:** (a) pure-Bantu bit-emulation — correct but unusably slow and error-prone → the user
explicitly rejected "educational"; (b) FFI to OpenSSL — output-buffer marshalling is impractical
(Bantu has no byte type to read digests back), Windows FFI is stubbed; (c) a full C++ crypto layer —
not "in Bantu," and larger attack surface to trust.
**Security:** Correctness is proven by bit-exact official test vectors; the trusted computing base is
small (a handful of arithmetic primitives).

### D2 — Bytes = Bantu list of integers 0–255
**Decision:** Represent binary data as a `list` of numbers 0–255 rather than adding a `bytes` type.
**Why:** Minimal interpreter surface; reuses `len`/indexing/list builtins; avoids a large type
addition for Core.
**Rejected:** a first-class `bytes` type (more invasive, deferred); using `string` as bytes (breaks
on values ≥128 and UTF-8, and there's no `ord`).
**Security:** Avoids silent corruption of binary input/keys/digests that a string-as-bytes model
would cause.

### D3 — CSPRNG only; never the `random` module for crypto
**Decision:** All key/salt/nonce/UUIDv4 material comes from native `randbytes(n)` (OS entropy:
`arc4random_buf` / `getrandom` / `/dev/urandom`).
**Why:** The `random` module is a Mersenne-Twister/LCG — predictable; using it for secrets is the
vulnerability, not the fix.
**Rejected:** seeding the LCG from `clock()` for "randomness" (predictable); reading `/dev/urandom`
from pure Bantu (blocked — no `ord()` to turn bytes ≥128 into numbers).
**Security:** Unpredictable secrets; this is the single most important control in the suite.

### D4 — MD5 and SHA-1 shipped, but documented as NON-secure
**Decision:** Provide `md5`/`sha1` for checksums, legacy interop, and RFC-4122 UUID v3/v5 — with
prominent "broken for security; do not use for integrity/passwords/signatures" notes.
**Why:** They're still needed (ETags, file checksums, UUID namespaces) and users expect them.
**Rejected:** omitting them (breaks UUID v3/v5 and common interop); shipping them silently (a
security expert must label collision-broken primitives).
**Security:** Explicit non-security labeling prevents misuse; steer users to SHA-256/HMAC.

### D5 — u32 only for Core; SHA-512 deferred
**Decision:** Ship the 32-bit families (MD5, SHA-1, SHA-224/256, HMAC-SHA256, HKDF-SHA256). Defer
SHA-384/512 and BLAKE2b.
**Why:** SHA-256 words are 32-bit → exact in float64 with the `add32/mul32` primitives. SHA-512 uses
**64-bit** words, which float64 cannot represent exactly → needs a real int64 path (a bigger,
separate change).
**Rejected:** faking 64-bit with hi/lo 32-bit pairs now (extra complexity/risk for Core).
**Security:** SHA-256 + HMAC-SHA256 cover the vast majority of modern needs; no security lost.

### D6 — AES and password-KDFs (PBKDF2/scrypt/argon2) out of scope for Core
**Decision:** No symmetric encryption or password hashing in Core; documented as a future **vetted
native** layer.
**Why:** AEAD (GHASH/GF-math, nonce discipline) and high-iteration KDFs (600k+ iterations) are
complex, side-channel-sensitive, and impractically slow in a tree-walking interpreter.
Reimplementing them in Bantu would be *less* secure and unusably slow.
**Rejected:** hand-rolled AES-GCM / PBKDF2 (the classic "don't roll your own crypto" trap).
**Security:** Avoids shipping weak/slow primitives users would trust for encryption and passwords.

### D7 — Constant-time comparison for all secret/tag checks
**Decision:** Native `ct_equal`; `verify_hmac` and any tag/digest comparison use it.
**Why:** Byte-by-byte `==` with early exit leaks timing; interpreter-level timing is unreliable, so
the primitive is native.
**Security:** Closes a standard timing side-channel on MAC verification.

### D9 — Fixed a real language bug: function-local variable scoping
**Decision:** Changed plain-variable assignment (`$x = v`) so it resolves only up to the enclosing
**function boundary**, defining a local if not found there — instead of walking the whole scope chain
and mutating a caller's/global's binding. Implemented with a `functionScope` flag on `Environment`
(set on function-call envs, module envs, and the global root) and a new `Environment::assign()`.
**Why:** During stress testing, calling a function that internally reuses a caller's variable name
(e.g. a loop counter `$i`) clobbered the caller's variable → infinite loops. This is what looked
like an "O(N²)"/perf cliff in D8; it was actually a **scoping defect**. Every hash/HMAC uses `$i`,
so `sha256()` called from a `while ($i < n)` loop hung. Reads still fall through all scopes, so
functions can read globals/module state.
**Rejected:** working around it by renaming variables everywhere (fragile; the language stays
broken); leaving it and documenting (a crypto lib you can't call in a loop isn't production-grade).
**Compatibility:** Verified no regressions — lang/classes 28/28, ORM+Sua 61/61, random 33/33,
orm-demo, and a live Sua HTTP server all pass. Existing code was already written defensively around
this (the ORM uses shared dicts for cross-scope state, which is unaffected — field/index mutation
targets the object, not a variable binding). Also fixes module top-level `$vars` leaking into global
(they now stay in the module namespace). Guarded by `tests/scope_test.b`. This supersedes D8: no
mitigation needed — hash cores are fast per-call and now safe to call in loops.

### D8 — O(N²) call-overhead bug: mitigate now, fix as stretch (SUPERSEDED by D9)
**Decision:** Keep hash hot-loops to **native builtins + arithmetic + array indexing only** (no
`if`/user-call inside loops; use constant tables and branchless `Ch/Maj`). Investigate the
underlying evaluator scope-accumulation separately.
**Why:** A single loop-heavy call is fast; the bug only bites *repeated* calls to functions with
`if`/calls inside loops. Native-only hot loops sidestep it, so a single `sha256()` stays fast and
HMAC/HKDF (few calls) remain cheap.
**Security:** No security impact; a real fix (stretch) also benefits the rest of the language.
