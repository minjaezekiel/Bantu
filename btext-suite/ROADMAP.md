# btext — Roadmap

Tokenizers and text features for Bantu. Paired with btorch's sequence layers
(phase 8) because neither is useful alone.

Design: [`../docs/btext-architecture.md`](../docs/btext-architecture.md).
Decisions: [`DECISIONS.md`](DECISIONS.md). Ecosystem context:
[`../docs/ecosystem-architecture.md`](../docs/ecosystem-architecture.md)
(stage 5).

**Prerequisites:** `bkit` phase K1 (`base.b`, the estimator protocol) for the
vectorizers in T6.

| tier | proves | form |
|---|---|---|
| 1. feature | every component does what it claims, including failure modes | `tests/btext_*_test.b` |
| 2. **round-trip** | `decode(encode(x)) == x` **byte-exactly** | fuzzed over random bytes, all scripts, emoji, lone surrogates, 0-length, 1 MB |
| 3. **differential** | ids match HuggingFace `tokenizers` exactly | committed fixtures from a real GPT-2 vocab |
| 4. **determinism** | same corpus → identical merges, twice, and under chunk reordering | [T3] |
| 5. stress | 1 GB corpus, 100k vocab, a single 10 MB "word", all-whitespace | — |
| 6. sanitizers | ASan + UBSan | — |

Tier 2 is the strongest test in the package and exists **only** because of the
byte-level choice ([T1]). An exact round-trip over fuzzed random bytes catches
every class of encoder bug — a dropped byte, a mis-ordered merge, a boundary
error — in one assertion.

---

## Phase T0 — Tracking docs
- [x] `docs/btext-architecture.md`
- [x] `btext-suite/ROADMAP.md`, `DECISIONS.md`

## Phase T1 — Normalizers and pre-tokenizers
Byte-level (default), whitespace, punctuation; and **optional** NFC/NFD,
lowercase, accent-strip for WordPiece compatibility. Bantu's `ord`/`chr`/
`bytes`/`utf8` builtins cover the byte layer, so no language work is needed.

Open: whether Bantu's regex support covers a `Split` pre-tokenizer — surveyed
before this phase is scoped.

## Phase T2 — The `tok_` native family

| kernel | purpose | Status |
|---|---|---|
| `tok_vocab` | an owned string→id map, so lookups are not Bantu dict calls | [ ] |
| `tok_bpe_encode` | **the hot loop**: repeated lowest-rank-pair merging with a priority structure | [ ] |
| `tok_bpe_train` | pair-frequency counting over the corpus, `vocab_size` times | [ ] |
| `tok_trie_match` | WordPiece/Unigram longest match | [ ] |
| **register `tok_` in `kNativeFamilies`** | `types.hpp:499` ([B21]) | [ ] |

A Bantu call costs **1.78 µs** ([B1]); a 1 GB corpus is ~10⁹ bytes. Per-byte
Bantu is six orders of magnitude out. These are not an optimisation.

Handles are **opaque**, like `bkit`'s binned matrix ([K1]) and for the same
reason: internal structures nobody indexes, needing string dtypes numba does
not have and should not grow.

## Phase T3 — BPE train, encode, decode
With the **specified deterministic tie-break** ([T3] in DECISIONS):

> Among pairs of equal frequency, choose the one whose first occurrence in the
> corpus is earliest; among those, the lexicographically smaller pair.

Tested by training the same corpus twice and asserting the merge lists are
identical, then training with the corpus chunked in a different order and
asserting the same result.

## Phase T4 — WordPiece and Unigram
Over `tok_trie_match`. Unigram is conditional on a target model needing it.

## Phase T5 — Vocab, special tokens, padding, truncation, attention masks
Loading published `vocab.json` / `merges.txt`. Batch encoding returning ids,
attention mask and token-type ids as numba arrays ready for `bt.tensor`.

## Phase T6 — Classical text features
`CountVectorizer`, `TfidfVectorizer`, `HashingVectorizer`, **implementing
`bkit`'s [C1] protocol** so they drop into a `bkit.Pipeline` with no adapter.

Dense `f32` output with a documented size warning and a conservative
`max_features` default until `bsci.sparse` lands in stage 6; then CSR, with
dense becoming opt-in. Recorded so that change is a planned migration, not a
surprise.

## Phase T7 — The package, docs, samples, gate

> **A real GPT-2 vocabulary (50,257 merges) loaded from its published
> `vocab.json`/`merges.txt`, encoding a fixed 10,000-sentence corpus to ids
> identical to HuggingFace `tokenizers` on every sentence.**

Not approximately — identically. A tokenizer off by one token on one sentence
in ten thousand silently degrades every model built on it.

---

## Where to resume
**T0 done; T1 next.** Blocked on `bkit` K1 for T6 only; T1–T5 can start as soon
as stage 5 begins.
