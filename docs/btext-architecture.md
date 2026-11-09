# Stage 5 — `btext`: tokenizers and text features

Phases **T0–T7** in [`../btext-suite/ROADMAP.md`](../btext-suite/ROADMAP.md).
Paired with btorch's sequence layers (phase 8, designed in
[`btorch-architecture.md`](btorch-architecture.md) §9) because neither is
useful alone: a tokenizer with nothing to feed, or an attention layer with no
way to turn text into ids. Context:
[`ecosystem-architecture.md`](ecosystem-architecture.md).

---

## 1. The pipeline

HuggingFace's four-stage shape, because it is the right decomposition and
because matching it means published tokenizer configs can be read directly.

```
text ──► normalize ──► pre-tokenize ──► model ──► post-process ──► ids
         NFC, lower,    whitespace,     BPE /     specials,
         strip accents  byte-level,     WordPiece  padding,
                        punctuation     Unigram    truncation
```

Each stage is independently configurable and independently testable. The model
stage is the only one with a hot loop.

---

## 2. Byte-level BPE is the default, and that is a deliberate dodge

**The decision that removes the most work.** Byte-level BPE (GPT-2's scheme)
operates on the 256 raw bytes of UTF-8, not on Unicode code points or
graphemes. The consequences:

| property | why it matters |
|---|---|
| **no Unicode tables at all** | no case-folding data, no normalisation tables, no grapheme-cluster rules, no ICU dependency. A Unicode database is tens of thousands of lines of generated tables that must track a yearly spec |
| **no out-of-vocabulary token, ever** | every byte sequence is representable, so no `[UNK]` and no silent information loss |
| **`decode(encode(x)) == x` exactly** | a byte-exact round-trip, which makes the strongest possible test available (see §6) |
| **every language works on day one** | including scripts nobody tested |

Bantu already has `ord`, `chr`, `bytes` and `utf8` builtins, so the byte layer
needs no language work.

**Rejected:** code-point-level BPE with Unicode normalisation as the default. It
is what WordPiece wants, it is more "correct" linguistically, and it would mean
shipping and maintaining Unicode tables before a single token was produced.
NFC/NFD and accent stripping are still implemented as *optional* normalisers in
T1 for users who need WordPiece compatibility — but they are not on the
critical path, and the default tokenizer works without them.

---

## 3. The `tok_` native family

Four kernels. Everything else — vocab management, training-loop bookkeeping,
configuration, special tokens — is Bantu, called once per batch rather than
once per character.

| kernel | signature | why native |
|---|---|---|
| `tok_bpe_encode` | `(bytes, merges_handle) -> ids` | **the hot loop**: repeated lowest-rank-pair merging over a word; O(len × merges) with a priority structure |
| `tok_bpe_train` | `(corpus, vocab_size, min_freq) -> merges` | pair-frequency counting over the whole corpus, repeated `vocab_size` times |
| `tok_trie_match` | `(bytes, trie_handle) -> spans` | WordPiece/Unigram longest-match; a trie walk per position |
| `tok_vocab` | `(strings) -> handle` | an owned string→id map, so lookups are not Bantu dict calls |

A Bantu function call costs **1.78 µs** (btorch B1). A 1 GB corpus is ~10⁹
bytes; per-byte Bantu is six orders of magnitude out. These four are not an
optimisation.

`merges_handle`, `trie_handle` and `tok_vocab`'s result are **opaque native
handles**, the same pattern `bkit` uses for its binned matrix and for the same
reason: they are internal structures nobody indexes, and expressing them as
`NdArray`s would need string dtypes that numba does not have and should not
grow.

---

## 4. Training must be deterministic, and that is not free

BPE training repeatedly selects the most frequent adjacent pair. **Ties are
common** — especially early, when many pairs occur exactly once — and the
obvious implementation breaks them by whatever order a hash map happens to
iterate in. That makes training irreproducible: the same corpus and the same
`vocab_size` produce a different vocabulary on a different run or a different
platform.

That is unacceptable under [C4]. The tie-break is therefore specified:

> **Among pairs of equal frequency, choose the one whose first occurrence in
> the corpus is earliest. Among those, choose the lexicographically smaller
> pair.**

Deterministic, platform-independent, and testable: a test trains the same
corpus twice and asserts the merge lists are **identical**, then trains it with
the corpus split into chunks in a different order and asserts the same result.

This is the text-domain instance of the same principle as [E15] — the cheap
implementation is nondeterministic, and determinism is the property worth
paying for.

---

## 5. Classical text features conform to `bkit`'s protocol

`CountVectorizer`, `TfidfVectorizer` and `HashingVectorizer` implement [C1] —
`fit`/`transform`/`fit_transform`/`get_params`/`clone` — so they drop into a
`bkit.Pipeline` beside a `StandardScaler` and a GBDT without an adapter.

This is why `bkit/base.b` holds the protocol and nothing else ([E12]): `btext`
depends on that one small file, not on gradient boosting and cross-validation.

Output is sparse by nature. Until `bsci.sparse` lands in stage 6, vectorizer
output is a dense `f32` array with a documented size warning, and a
`max_features` default low enough that the dense form is reasonable. Once
`sp_` exists, these return CSR and the dense path becomes opt-in. Recorded so
the eventual change is a planned migration rather than a surprise.

---

## 6. Gates

| tier | proves | form |
|---|---|---|
| 1. feature | every component does what it claims, including failure modes | `tests/btext_*_test.b` |
| 2. **round-trip** | `decode(encode(x)) == x` **byte-exactly**, for every input | fuzzed over random bytes, all scripts, emoji, lone surrogates, 0-length, 1 MB |
| 3. **differential** | ids match HuggingFace `tokenizers` exactly | committed fixtures from a real GPT-2 vocab, generated offline |
| 4. **determinism** | same corpus → identical merges, twice, and under reordering | §4 |
| 5. stress | 1 GB corpus, 100k vocab, pathological inputs | a single 10 MB "word"; all-whitespace; one repeated byte |
| 6. sanitizers | no latent memory defect | ASan + UBSan |

Tier 2 is the strongest test in the package and it exists only because of the
byte-level choice (§2). An exact round-trip over fuzzed random bytes catches
every class of encoder bug — a dropped byte, a mis-ordered merge, a boundary
error — in one assertion. Code-point-level tokenization cannot offer it,
because normalisation is lossy by design.

Tier 3 uses the method already proven twice in this project (pillow's
differential references, btorch's embedded PyTorch values): generate reference
output with the reference implementation **offline**, commit the fixture, and
never make the test depend on having Python installed.

### 6.1 The accuracy gate

**A real GPT-2 vocabulary (50,257 merges) loaded from its published
`vocab.json`/`merges.txt`, encoding a fixed 10,000-sentence corpus to ids
identical to HuggingFace `tokenizers` on every sentence.** Not approximately —
identically. A tokenizer that is off by one token on one sentence in ten
thousand will silently degrade every model built on it.

---

## 7. What is out

| item | why |
|---|---|
| SentencePiece model files | a different protobuf schema; cheap once `bonnx`'s reader exists, so deferred until after stage 4 |
| Unicode normalisation as a default | §2; available as an opt-in normaliser |
| language detection, stemming, lemmatisation | linguistic tooling, not tokenization; no clear demand |
| pretrained embeddings (word2vec/GloVe loaders) | a loader is twenty lines; add on request, not speculatively |
| regex pre-tokenizers | needs Bantu's regex support surveyed first; whitespace + punctuation + byte-level covers the default |

---

## 8. Open items

| # | item | settled by |
|---|---|---|
| 1 | does `tok_bpe_train` need the stage-1 thread pool? | pair counting partitions cleanly over the corpus and is race-free per chunk; measure on 1 GB |
| 2 | Unigram (SentencePiece) algorithm — worth it? | only if a target model needs it; BPE + WordPiece cover most |
| 3 | dense vs sparse vectorizer output before stage 6 | §5; `max_features` default chosen by measurement |
| 4 | does Bantu's regex support cover a `Split` pre-tokenizer? | survey before T1 |
