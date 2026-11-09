# btext — Decisions

Ecosystem decisions are `E*`/`C*` in
[`../docs/ecosystem-architecture.md`](../docs/ecosystem-architecture.md);
btorch's are `B*`; bkit's are `K*`.

---

### T1 — Byte-level BPE is the default, and that is a deliberate dodge
**The decision that removes the most work in this package.** Byte-level BPE
(GPT-2's scheme) operates on the 256 raw bytes of UTF-8, not on code points or
graphemes. The consequences:

- **No Unicode tables at all.** No case-folding data, no normalisation tables,
  no grapheme-cluster rules, no ICU. A Unicode database is tens of thousands of
  lines of generated tables that must track a yearly spec.
- **No out-of-vocabulary token, ever.** Every byte sequence is representable,
  so no `[UNK]` and no silent information loss.
- **`decode(encode(x)) == x` byte-exactly**, which makes the strongest possible
  test available (tier 2).
- **Every language works on day one**, including scripts nobody tested.

**Rejected:** code-point-level BPE with Unicode normalisation as the default. It
is what WordPiece wants and it is more correct linguistically — and it would
mean shipping and maintaining Unicode tables before a single token was
produced. NFC/NFD and accent-stripping are still implemented as *optional*
normalisers in T1, but they are not on the critical path.

---

### T2 — The `tok_` handles are opaque, not `NdArray`s
Same decision as `bkit`'s [K1], same reason: merge tables, tries and vocab maps
are internal structures nobody indexes, and expressing them as arrays would
need string dtypes numba does not have and should not grow.

---

### T3 — BPE training's tie-break is specified, because the obvious one is nondeterministic
BPE training repeatedly selects the most frequent adjacent pair. **Ties are
common** — especially early, when many pairs occur exactly once — and the
obvious implementation breaks them by whatever order a hash map happens to
iterate in. The same corpus and the same `vocab_size` then produce a different
vocabulary on a different run or a different platform.

Unacceptable under [C4]. So:

> **Among pairs of equal frequency, choose the one whose first occurrence in
> the corpus is earliest. Among those, choose the lexicographically smaller
> pair.**

Deterministic, platform-independent, and testable — which is the whole point.
This is the text-domain instance of the same principle as [E15]: the cheap
implementation is nondeterministic, and determinism is worth paying for.

---

### T4 — Vectorizers implement bkit's protocol rather than defining their own
`CountVectorizer` and friends satisfy [C1], so they compose with
`bkit.Pipeline` beside a `StandardScaler` and a GBDT without an adapter. This
is why `bkit/base.b` holds the protocol and nothing else ([E12], [K7]) — this
package depends on that one small file, not on gradient boosting and
cross-validation.

---

### T5 — Dense vectorizer output now, CSR after stage 6, and said so in advance
Vectorizer output is sparse by nature, but `bsci.sparse` does not exist until
ecosystem stage 6. Until then the output is a dense `f32` array with a
documented size warning and a conservative `max_features` default.

Recording it here means the eventual switch to CSR is a **planned migration
with a stated trigger**, not an API break that surprises users. The alternative
— waiting for stage 6 — would block the whole package on an unrelated one.
