# One Taylor — architecture

A social feed that doubles as a dating app, whose defining property is that **minors and adults
never meet**. Backend in Bantu (`sua` + `orm` + SQLite), frontend in Next.js/TypeScript, text
moderation by [`lugha`](lugha-architecture.md).

- **App:** `samples/one-taylor/` — `backend/` (Bantu), `web/` (Next.js)
- **Tests:** `tests/one_taylor_*`

---

## 1. The one invariant

Everything else in this document is in service of a single rule:

> No account on one side of the 18 boundary can observe, reach, or be reached by an account on the
> other side — in the feed, in search, in recommendations, in a follow, in a like, in a report, or
> by typing a URL.

That rule is implemented **once**, in `backend/safety/age.b`, and composed into every read path. It
is not re-derived per route, because a rule written out eleven times is a rule with eleven chances
to be written wrong, and the eleventh is the one that ships.

### 1.1 The bands

| band | ages | sees | may be matched with |
|---|---|---|---|
| `minor` | 13–17 | the same band, within **±2 years** of own age | the same; the range is not a setting |
| `adult` | 18+ | every adult | adults inside their dating range (default `[age−10, age+10]`, **hard floor 18**) |

Signup below 13 is refused outright.

```bantu
band($dob)                     // "minor" | "adult"
canSee($viewer, $target)       // may the viewer see this account's CONTENT?
canMatch($a, $b)               // should this account be OFFERED as a match?
canInteract($a, $b)            // follow / like / report
visibilityPairs($viewer)       // ORM pairs for `posts` — the ONE source of the feed's SQL
userVisibilityPairs($viewer)   // ORM pairs for `users` — content visibility
matchPairs($viewer)            // ORM pairs for `users` — narrowed to the dating range
```

**`canSee` and `canMatch` are deliberately separate, and conflating them was a
real error in the first draft of this design.** Gating the feed on the dating
range looks safer and is not: a 19-year-old whose range is 25–29 would stop
seeing their own posts, and their friends'. A preference about who to date is
not a statement about whose life stories you may read, and overloading one into
the other produces a feed nobody can explain.

What the two share is the wall. Neither crosses 18, ever. `canSee` is symmetric
— the ±2 rule is symmetric for minors and trivially true among adults — which
is what lets `canInteract` be a simple conjunction, since a one-sided follow is
still contact. `canMatch` additionally requires that each party is inside the
other's range, because a match the other person's own range would not have
produced is a match they did not ask for.

Only **Discover** uses `matchPairs`. The feed, search and profile pages use
content visibility.

### 1.2 Date of birth, not age

The signup form asks for a birth date, not "your age". Three reasons, and the first is the one that
matters:

- An `age` integer is **wrong within a year** and wrong forever after. A 17-year-old who typed
  `17` stays 17 in the database while turning 18, 19, 20 — on the minor side of the wall, with
  minors, indefinitely. A `dob` is a fact; an `age` is a fact's expiry date.
- It is auditable. "Was this account correctly banded on 4 March?" is answerable from a `dob` and
  not from an `age`.
- It makes the 18th-birthday transition a computation rather than a migration.

The UI still speaks in plain terms and shows the age it derived, so nothing is hidden from the user.

### 1.3 Crossing 18

A band change is not a cosmetic relabel: every edge the account holds was created under the *old*
band's rules, and some of those edges now cross the wall. On session resolve, `sessions.b`
recomputes the band from `dob`; if it differs from `band_at_issue`, it runs a transition that
prunes, in one transaction, every follow, like, hidden-post row and match-cache entry that the new
band would not have allowed — then records the transition in `moderation_events`.

So the day a 17-year-old turns 18, their 16-year-old friends are no longer reachable, and that is
the intended behaviour rather than a bug to be worked around.

## 2. Data model

SQLite via `orm/orm.b` (`orm.model` auto-creates; `orm.migrate` diffs and alters). Every column the
visibility filter touches is indexed, because `sua.sqlite` never suspends — a slow query stalls its
whole worker, not just its own request.

| table | columns of note |
|---|---|
| `users` | `username`, `password_hash` (argon2id), `email_enc`, `email_bidx`, `phone_enc`, `phone_bidx`, `dob`, `gender`, `band`, `status` |
| `profiles` | `user_id`, `bio`, `seek_min_age`, `seek_max_age`, `seek_genders`, `explicit_optins` |
| `user_tags` | `user_id`, `tag`, `kind ∈ interest \| hobby \| dream_partner` |
| `posts` | `author_id`, `body`, `author_band`, `author_age_at_post`, `status`, `created_at` |
| `post_media` | `post_id`, `path`, `mime`, `sha256` |
| `post_labels` | `post_id`, `category`, `source ∈ author \| model`, `confidence` |
| `post_tags`, `follows`, `likes`, `blocks`, `hidden_posts` | the social graph |
| `reports` | `post_id`, `reporter_id`, `reason` — unique on `(post_id, reporter_id)` |
| `moderation_events` | append-only audit: subject, scores, decision, threshold, actor |
| `sessions` | `token_hmac`, `user_id`, `expires_at`, `band_at_issue` |

**`posts.author_band` and `author_age_at_post` are denormalised on purpose.** The feed query filters
on the author's band and age millions of times and must not join `users` to do it; and a post should
be filtered by *who wrote it, when* — if a 17-year-old's post becomes visible to adults the moment
they turn 18, the wall leaks backwards through history.

## 3. Privacy of email and phone

Both are **encrypted at rest** with XChaCha20-Poly1305 (`crypto.encrypt`), under a key derived by
`crypto.hkdf_sha256` from `ONE_TAYLOR_SECRET` with a purpose-specific `info` string. The server
refuses to start without that env var rather than falling back to a default key.

Encrypted columns cannot be searched, so each has a **blind index**: `hmac_sha256(pepper, lower(x))`,
stored alongside and marked unique. Uniqueness checks and "log in by email" work against the index;
the plaintext is never in a query.

One function, `privacy.publicUser($row, $viewer)`, is the **only** way a user object reaches a
response anywhere in the API. `email_enc`, `email_bidx`, `phone_enc`, `phone_bidx`, `dob`,
`password_hash` and `status` are structurally absent from what it builds — not filtered out, never
added. A route cannot leak them by forgetting, because a route never holds them.

`tests/one_taylor_privacy_test.b` asserts that property against every account state.

## 4. Moderation

One decision function, called on post create and on profile save, never on read:

```
normalise → lexicon → lugha (7 scores) → decision table → moderation_events row
```

### 4.1 Who is allowed to block

This is the part of the design that changed most during the build, and the
reason is worth recording.

The classifier's measured precision at its operating points is **0.44 to 0.92**
depending on the label (`lugha/corpus/METRICS.md`). On the labels that matter
most for a minor it is about two thirds. The first version let any flag block a
post, and the consequence was not hypothetical: the API test caught a
15-year-old's *"Scored twice at practice today, best session all term"* being
blocked for grooming. One innocent post in three deleted is not a tuning
problem — it is a model with no discriminating power being given authority.

So authority follows the quality of the evidence:

| evidence | action | what the author sees |
|---|---|---|
| a **lexicon** term matched exactly, on a word boundary | **block** | not published; appealable; the text is kept |
| a model flag on a label whose measured precision ≥ 0.6 | **hold** | saved, visible only to them, pending review |
| an advisory model flag, or a score within 0.15 of a threshold | **watch** | published, and queued for a human |
| nothing | **allow** / **label** | published, with a content warning if one applies |

A hold is reversible in one click and the author is told it is pending. A block
is not something to do on two-thirds confidence.

Which labels may block is not a constant in the source — it is the
`reliability` tensor stored in the model file, computed by `lugha/calibrate.b`
from held-out precision, and reported by `/api/health` and at startup so the
policy in force is observable rather than inferred.

### 4.2 Band differences

| author | outcome |
|---|---|
| `minor` | the full table above, every label |
| `adult` | only `sexual_minor`, `grooming` and `threat` can hold or block; everything else becomes a content **label** |

Adults keep the freedom the brief asked for: what the classifier does for an
adult is label, so the opt-in gate has something to work with. Minors get the
strict table, which is the entire reason the model exists.

### 4.3 Where it runs

**Never on a read.** `sua.sqlite` and the handler share a thread, so a 10 ms
model call on the feed path is 10 ms added to every other client's latency on
that worker. Scores are computed once, at write, and stored.

A profile **bio** goes through the same pass as a post. A profile is where
grooming actually starts — *"15, mature for my age, dm me"* is a bio, not a
post — so exempting it would leave the front door open. But only a lexicon hit
refuses a bio: refusing one on a two-thirds-confident guess tells a teenager
that describing their own hobbies is against the rules.

Every call appends a `moderation_events` row carrying the scores that produced
the decision, so a disputed takedown can be reconstructed and a threshold change
can be evaluated against history.

## 5. Reports

`POST /api/posts/:id/report` is idempotent per `(post, reporter)` — the unique index makes a second
report from the same account a no-op rather than a second vote.

- **1 report** hides the post from that reporter immediately (`hidden_posts`). Whatever the
  community decides, the person who objected stops seeing it now.
- **≥ 2 distinct reporters** on an explicit/policy reason flips `posts.status` to `removed`, with
  an audit row naming the reporters' count and the threshold.

**Brigading guard**, in its third form — the first two were both wrong, and
instructively:

1. *"an account under an hour old does not count"* — on the day the app
   launches, every account is under an hour old and no report counts for
   anything. The safety feature would be off exactly when the community is
   smallest and most fragile. The API test caught this.
2. *"a new account with no posts and no follows does not count"* — a lurker who
   signs up, reads for ten minutes and reports something genuinely vile has
   posted nothing and followed nobody. Dropping their report silences the people
   most likely to use the button.
3. **A new, empty account's report is *weak*, and at most one weak report counts
   toward a post's threshold.**

| | outcome |
|---|---|
| attacker with two sockpuppets | 1 counted → no takedown. Closed. |
| one lurker + one real user | 2 counted → takedown. Works. |
| two genuine new lurkers | 1 counted → the post goes to the review queue instead |

A report whose reason needs a human (`spam`, `impersonation`, `other`), or that
comes from an account the author has blocked, is recorded and never counted.
Uncounted reports appear in the queue under "possible brigades", because a
brigade is itself worth seeing.

## 6. Explicit content, for adults

Categories: `suggestive`, `nudity`, `sexual`, `fetish`, `violence`, `gore`, `substances`.

A label arrives from the poster (self-declared) or from the classifier, recorded with its `source`
and `confidence` in `post_labels`. Adults set `profiles.explicit_optins`; a labelled post the
viewer has not opted into is delivered **blurred, with the category named and a tap to reveal** —
so consent is per-category and per-post, and nobody is surprised by an image.

A `minor` is never served a labelled post. Not filtered from the feed and reachable by id — *never
served*: `GET /api/posts/:id` and `GET /api/media/:id` both re-check the band, and both answer 404.

## 7. Media

`sua` has no multipart parser, so images arrive base64 inside the JSON body (the client resizes to
≤ 4 MB first, and `sua.server.limits` is raised to match). The handler decodes with
`crypto.base64_decode`, verifies the magic bytes against the claimed MIME type, and writes with
`writefile($p, $bytes, "wb")` under `uploads/<sha256 prefix>/`.

`uploads/` is **not** a `sua.server.static` root. It cannot be, because a static root is ungated by
construction and the whole point of this app is that media is gated. `media_routes.b` re-runs
session → band → label-optin → report-status, then streams the bytes:

```bantu
$res.type($mime).send(frombytes(readfile($path, "rb")));
```

which is byte-exact — `frombytes` round-trips arbitrary bytes, NULs included (verified:
`[0,255,65,10,200,0,66]` → `00ff410ac80042`, 7 bytes).

## 8. Recommendations

**Gates first, scoring second.** The candidate set is filtered by band, range, blocks, report status
and label opt-ins *before* anything is ranked, so no amount of score can promote a post past the
wall. Filtering after ranking is how these systems leak.

People (`recsys/people.b`):

```
score =  0.35 · jaccard(interests)
       + 0.20 · jaccard(hobbies)
       + 0.25 · dreamPartnerFit(gender, age, declared preferences)
       + 0.10 · mutualFollows
       + 0.10 · recencyOfActivity
```

Posts (`recsys/posts.b`): candidates from follows, from high-scoring people matches, and from
in-band popularity; ranked by tag affinity (the viewer's interest vector · the post's tag vector),
engagement, and a recency decay of `0.5 ^ (ageHours / 36)`.

`recsys/vectors.b` does the arithmetic in plain Bantu lists for small candidate
sets and switches to numba above ~10,000 — numba's own documented crossover,
below which a builtin call's 1–3 µs of interpreter overhead costs more than the
loop it replaces. `tests/one_taylor_recsys_test.b` checks that both paths agree
to rounding across that boundary, so the ranking does not change shape at an
arbitrary population size.

**A correction, recorded because it was wrong in an instructive way.** The plan
said tag overlap would move to numba above ten thousand candidates. It should
not. Tag overlap is a sparse set intersection — a hash join over a few dozen
strings per user — and the dict version is already O(total tags). Handing it to
numba would mean building a dense users × tags matrix, which is both slower and
quadratically larger. numba is for dense arithmetic on contiguous numbers; the
weighted score combination is that, and the set intersection is not.

### 8.1 The feed is batched, and that is not an optimisation

The first version called a per-post hydration function, which issues five
queries of its own — author, profile, labels, media, tags — plus a sixth for the
post's age in hours. Over a 1,200-post table that is about **7,000 round trips
for one feed request, measured at 677 ms**. Since `sua.sqlite` never suspends,
those 677 ms are added to every other client waiting on the same worker: one
person opening their feed slows everybody down.

The feed now fetches its candidates in **one** query that computes the age in
hours in SQL and excludes blocked authors and hidden posts there too, then loads
everything the ranking and the rendering need in **four** more queries using
`IN (…)` — five in total, whatever the page size. The per-post function remains
for single posts, where N is 1.

Each match card carries **why** it matched (the shared interests), because an unexplained
recommendation in a dating app is indistinguishable from a random one.

## 9. Messaging, voice and video

### 9.1 Who may message whom, and why it is stricter than who may see whom

Three conditions, all of them:

1. `canInteract` — the age wall, both directions. Same as everywhere else.
2. Neither has blocked the other.
3. **They follow each other.** A mutual follow is this app's *match*: both
   people chose it, separately, before either could send a word.

Condition 3 does the work. A feed is something you read; a direct message is
something that *arrives*. An app where any stranger in your band can open a
thread with a child has a grooming channel built into it, whatever the
classifier does afterwards — and a classifier is a filter on a channel that
should not exist. Requiring both sides to opt in first costs an adult one tap
and removes the channel.

The gate is re-checked on every read and every write, not just when the thread
opens: a block, an unfollow or an 18th birthday has to close a thread that was
legitimate yesterday. A thread the wall no longer permits is simply not listed;
it is not deleted either, so lifting a block brings it back.

### 9.2 Moderation on a message, and who may stop one

Every message goes through the same decision function as a post. The *action*
differs, and the difference is the point:

| | a post | a direct message |
|---|---|---|
| lexicon hit | **block** | **block** — not delivered |
| reliable model flag | **hold** (author-only, pending review) | **deliver**, and queue for review |
| advisory flag / borderline | **watch** (published, queued) | **deliver**, and queue for review |

For a post, holding is cheap: the author waits and nobody was expecting it. For
a message, **not delivering it is the irreversible action** — one side of a
conversation stops hearing from the other and neither can act on why. At the
classifier's measured precision that silently breaks about one innocent
exchange in three, and the realtime test caught exactly that: a 15-year-old's
*"scored twice at practice today, best session all term"* was held and never
arrived.

So on this path only an exact lexicon match stops a message. A model flag
delivers it and sends the thread to a human. A reviewer reading a grooming
thread is worth more than a blocked line that teaches the groomer which words
to avoid next time — and the person being groomed is told, in the thread, that
a moderator is reading.

### 9.3 One socket, three jobs

Presence, chat and call signalling share one `sua.ws` socket, multiplexed by a
frame `type`. Three sockets would cost three slots against `max_connections`,
three handshakes and three things to reconnect.

Authentication is a **ticket**, not the session token. The token lives in an
httpOnly cookie on the Next.js origin; the socket connects to the Bantu origin,
so the browser would not send it — and a token in a WebSocket URL is written to
every proxy log on the path, unredacted. The page asks its own server for a
ticket worth sixty seconds and one use, stored here only as an HMAC under its
own HKDF label, and spends it in the first frame. A socket that has not said
hello can do nothing at all.

The WebSocket upgrade is checked against an **origin allowlist**
(`ws_allowed_origins`), not a wildcard. sua compares `Origin` to `Host` and the
client sends both, so without a list any page on the internet could open a
socket to a user's localhost server.

### 9.4 Calls

**The server is not a WebRTC peer.** It holds no `RTCPeerConnection`,
terminates no media and never sees a frame of audio or video. It relays the
offer, the answer and the ICE candidates between two browsers and gets out of
the way, which is also why a call costs it almost nothing.

What it does do is re-check the gate on **every relayed frame**, including each
ICE candidate. A signalling relay that trusts the `to` field in a frame is a
relay that connects a 40-year-old to a 14-year-old the moment somebody edits a
number in devtools. The `call-answer` path additionally requires that the
server itself set up that call, or an answer frame becomes a way to inject an
SDP into somebody else's.

On the browser side, three things that are easy to get wrong:

- **Capture happens before the peer connection is built.** If the camera is
  refused, nothing has been allocated and the refusal can be explained.
- **ICE candidates that arrive before the remote description are queued.**
  `addIceCandidate` throws without one, and the answerer always receives
  candidates before it has finished answering.
- **Every track is stopped explicitly on hang-up.** Closing the peer connection
  does *not* release the camera, and a camera light that stays on after a call
  is the most alarming bug this feature can have.

A video call falls back to voice when a camera is absent or refused, because a
voice call is a much better outcome than no call.

`getUserMedia` is granted only in a **secure context**. `http://localhost` is
one; a LAN address is not — so a phone pointed at this machine can message but
cannot call until there is TLS in front. Two peers both behind symmetric NAT
need a TURN relay: `bantu relay` provides one, and `ONE_TAYLOR_TURN` points at
it. Without TURN the UI says so when a connection fails, rather than spinning.

## 9. Sessions

Password hashing is argon2id (`crypto.hash_password`) — memory-hard, salted, self-describing, with
constant-time verification. This requires an interpreter built with libsodium
(`BANTU_SODIUM=1 bash bantu-src/compiler/build-mac.sh`); `config.b` checks
`crypto.encryption_available()` at startup and refuses to run without it, because the fallback —
PBKDF2 in interpreted Bantu — is either too slow to log in with or too weak to store.

A session token is `crypto.token_urlsafe(32)` from the OS CSPRNG. The database stores only
`hmac_sha256(pepper, token)`, so a dump of the `sessions` table does not let anyone log in.

`sua` has no cookie or session helper and can emit only one `Set-Cookie` per response, so the
browser never talks to the Bantu server directly about sessions: the Next.js route handlers in
`web/app/api/*` proxy every call and own a single `httpOnly`, `SameSite=Lax` cookie. The token is
never visible to client JavaScript.

## 10. Frontend

Next.js 15 App Router, TypeScript, and **hand-written CSS with design tokens** — no component
library, because "custom and modern" is a design brief and a component library is someone else's
answer to it. `web/styles/tokens.css` declares the palette on `:root`, redeclares it under
`@media (prefers-color-scheme: dark)` guarded by `:root:not([data-theme="light"])`, and again under
`:root[data-theme="dark"]`, so the OS preference and the manual toggle compose instead of fighting.

Screens: signup (birth date, gender, username — stating plainly what the age band will mean),
login, feed, post detail, profile, discover, search, settings, moderation queue. For a minor
account the explicit-category settings are **absent**, not disabled: a disabled control is an
advertisement.

## 11. Known limits

- **No TLS in `sua.server`.** Production needs a terminator in front of it;
  `samples/webrtc-chat/tls-dev-proxy.py` is the dev stand-in. Without HTTPS, `Secure` cookies and
  any camera access do not work off localhost.
- **`sua.server.use()` middleware never runs**, so the auth-and-gate call is the explicit first
  line of every handler. `tests/one_taylor_api_test.sh` probes every route unauthenticated to catch
  the one where it was forgotten.
- **Age is self-declared.** Nothing here verifies it. The bands, the labels and the audit trail are
  what software can do; age *verification* is identity infrastructure and policy, and pretending
  otherwise would be the most dangerous sentence in this document.
- **The classifier is a baseline.** See [lugha-architecture.md § 6](lugha-architecture.md).
- Hosting user-uploaded explicit media carries legal duties — age verification, reporting
  obligations, jurisdiction — that code does not discharge.
