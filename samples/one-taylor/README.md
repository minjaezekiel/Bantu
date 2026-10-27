# One Taylor

A social feed that doubles as a dating app, whose defining property is that
**minors and adults never meet**.

- **Backend** — Bantu, with `sua` for HTTP, `orm` over SQLite, `crypto` for
  argon2id and AEAD, and [`lugha`](../../lugha) — a transformer written in
  Bantu over `numba`'s kernels — for moderation.
- **Frontend** — Next.js 15 and TypeScript, with hand-written CSS.
- **Design** — [`docs/one-taylor-architecture.md`](../../docs/one-taylor-architecture.md)
  and [`docs/lugha-architecture.md`](../../docs/lugha-architecture.md).

---

## The one rule

> No account on one side of the 18 boundary can observe, reach, or be reached
> by an account on the other side — in the feed, in search, in
> recommendations, in a follow, in a like, in a report, or by typing a URL.

| band | ages | sees | may be matched with |
|---|---|---|---|
| `minor` | 13–17 | the same band, within **±2 years** | the same, and the range is not a setting |
| `adult` | 18+ | every adult | adults inside their dating range, floor 18 |

Content visibility and the dating range are **separate questions**, and
conflating them was a real error in the first draft of this design: gating the
feed on a dating range means a 19-year-old looking for 25–29 stops seeing their
own posts. A preference about who to date is not a statement about whose life
stories you may read.

The whole rule lives in [`backend/safety/age.b`](backend/safety/age.b) and is
composed into every read path. `tests/one_taylor_safety_test.b` checks it
against **every ordered pair of ages from 13 to 80** — 4,624 pairs, both
directions — plus the exact morning of an 18th birthday.

## Running it

You need an interpreter built with libsodium. argon2id and AEAD are
load-bearing here and there is no fallback:

```sh
BANTU_SODIUM=1 bash bantu-src/compiler/build-mac.sh      # build.sh on Linux
cp bantu-src/compiler/build/bantu ./bantu
echo 'print(sodium_available());' | ./bantu               # -> true
```

Train the classifier once (about twelve minutes), or skip it and run with
lexicon-only moderation:

```sh
bantu run lugha/corpus/generate.b       # the labelled corpus
bantu run lugha/corpus/build_vocab.b    # the WordPiece vocabulary
bantu run lugha/train.b                 # -> backend/model/toxicity.btw
cp lugha/vocab.txt samples/one-taylor/backend/model/vocab.txt
```

Then, in three shells:

```sh
# 1. the backend
cd samples/one-taylor/backend
ONE_TAYLOR_SECRET=$(openssl rand -hex 32) \
ONE_TAYLOR_ADMIN_TOKEN=$(openssl rand -hex 24) \
bantu run server.b                      # http://localhost:4100 (HTTP + WS)

# 2. the frontend
cd samples/one-taylor/web
npm install
ONE_TAYLOR_API=http://127.0.0.1:4100 npm run dev   # http://localhost:3000

# 3. the tests, from the repo root
bantu run tests/one_taylor_safety_test.b
bantu run tests/one_taylor_privacy_test.b
bantu run tests/one_taylor_recsys_test.b
bash  tests/one_taylor_api_test.sh      # starts its own server
python3 tests/one_taylor_realtime_test.py   # the WebSocket: messages + calls
bash  tests/one_taylor_stress.sh
```

Optional environment:

| variable | what it does |
|---|---|
| `ONE_TAYLOR_ADMIN_TOKEN` | opens the review queue at `/moderation` |
| `ONE_TAYLOR_ORIGINS` | WebSocket origin allowlist (default: localhost:3000) |
| `ONE_TAYLOR_WS_URL` | what the browser should dial for the socket |
| `ONE_TAYLOR_STUN` / `ONE_TAYLOR_TURN` | ICE servers for calls; `bantu relay` provides a TURN |
| `ONE_TAYLOR_WORKERS` | `0` for one worker per core (needs the WAL pragmas `models.b` sets) |

`ONE_TAYLOR_SECRET` is required and has no development default: every key the
app uses — session tokens, the blind indexes that make encrypted emails
searchable, the key that encrypts emails and phone numbers — is derived from it
by HKDF with a purpose-specific label. A shipped default secret is the most
common way an application ends up with a known one.

## What it does

**Accounts.** Username and password, argon2id. Signup asks for a **birth date**
rather than an age, because an age integer is wrong within a year and wrong
forever after — a 17-year-old who typed "17" would stay 17 while turning 18, 19,
20, on the minor side of the wall. Email and phone are encrypted at rest with
XChaCha20-Poly1305 and searched through HMAC blind indexes; neither is ever
returned by any endpoint, including to the account's owner.

**Posts.** Text and up to four pictures. `sua` has no multipart parser, so
images arrive base64 inside the JSON body and the browser resizes them first.
Uploads are content-addressed by SHA-256 and are **not** served from a static
directory — `/api/media/:id` re-runs the whole gate before a byte goes out,
because a static root is ungated by construction.

**Moderation.** One pass at write time, never on read. Normalise → lexicon →
transformer → decision. Authority is split by the quality of the evidence:

| evidence | action |
|---|---|
| a lexicon term matched exactly | **block** — never visible, appealable |
| a model flag on a label with measured precision | **hold** — only the author sees it, pending review |
| an advisory model flag, or a borderline score | **watch** — published, and queued for a human |

The model is not allowed to delete a post on its own. Its measured precision is
0.44–0.92 depending on the label, so a blocking policy would delete roughly one
innocent post in three — and it did: the API test caught a 15-year-old's
"Scored twice at practice today" being blocked for grooming before the policy
was split this way.

**Reports.** One report hides a post from the reporter immediately. Two distinct
reporters on a policy reason take it down. A report from a brand-new account
that has never posted or followed anybody is *weak*, and at most one weak report
counts toward a post's threshold — so two sockpuppets cannot remove anything,
while one lurker plus one real user still can.

**Explicit content, for adults.** Seven categories, declared by the poster or
assigned by the classifier. Adults opt in per category; anything they have not
opted into arrives blurred with the category named. A minor is never served a
labelled post — not filtered from the feed and reachable by id, *never served*.

**Matching.** Interest and hobby overlap, declared preferences, mutual follows
and recency, weighted. Gates run **before** scoring, so no score can promote
anybody past the wall. Every match card says *why* it matched.

**Messaging, voice and video.** Direct messages, typing indicators, read
receipts, presence, and peer-to-peer voice and video calls — all over one
`sua.ws` WebSocket, with WebRTC signalling relayed by the backend.

A conversation requires a **mutual follow**: both people chose it before either
could send a word. A feed is something you read; a message is something that
arrives, and an app where any stranger in your band can open a thread with a
child has a grooming channel built in whatever the classifier does afterwards.

Messages are moderated like posts, but only an exact lexicon match stops one.
For a message, non-delivery *is* the irreversible action, and at the
classifier's measured precision a blocking policy would silently break about
one innocent exchange in three — the realtime test caught a 15-year-old's
"scored twice at practice today" being held and never arriving. A model flag
delivers the message and sends the thread to a human instead, and the thread
says so.

Calls are peer to peer: the server relays the offer, the answer and the ICE
candidates and never sees media. It re-checks the age wall on **every** relayed
frame, including each ICE candidate. Calls need a secure context for
`getUserMedia`, so they work on `http://localhost` but not on a LAN address
without TLS; two peers behind symmetric NAT need `bantu relay` and
`ONE_TAYLOR_TURN`.

## Honest limits

- **No TLS in `sua.server`.** Put a terminator in front for anything beyond
  localhost; `samples/webrtc-chat/tls-dev-proxy.py` is the dev stand-in.
  Without HTTPS, `Secure` cookies and camera access do not work off localhost.
- **Age is self-declared.** Nothing here verifies it. The bands, the labels and
  the audit trail are what software can do; age *verification* is identity
  infrastructure and policy, and pretending otherwise would be the most
  dangerous sentence in this file.
- **The classifier is a baseline**, trained on a corpus generated in this
  repository. Its held-out numbers are in
  [`lugha/corpus/METRICS.md`](../../lugha/corpus/METRICS.md), it has a human
  review queue behind it by design, and `tools/convert_hf.py` exists so better
  weights are a file swap rather than a rewrite.
- Hosting user-uploaded explicit media carries legal duties — age verification,
  reporting obligations, jurisdiction — that code does not discharge.
