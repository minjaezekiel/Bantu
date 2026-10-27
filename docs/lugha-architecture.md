# `lugha` — architecture

`lugha` ("language") is Bantu's NLP package: a text normaliser, a WordPiece tokenizer, and a
transformer encoder with a multi-label classification head — all written in Bantu over
[`numba`](numba.md)'s native kernels. Its first caller is One Taylor's moderation pipeline, which
has to answer *"is this text abusive?"* about every post written by an under-18 account.

- **Source:** `lugha/` — `normalise.b`, `tokenizer.b`, `encoder.b`, `weights.b`, `train.b`
- **Tests:** `tests/lugha_tokenizer_test.b`, `lugha_encoder_test.b`, `lugha_train_test.b`
- **Caller:** `samples/one-taylor/backend/safety/moderation.b`

---

## 1. Why a transformer, and why in Bantu

A keyword list cannot read context. *"I'll kill you"* and *"I'll kill it at the audition"* share a
verb and differ entirely; *"she's sick"* is an insult or a diagnosis depending on three words
either side. Negation, sarcasm and target ("you are" vs "I am") are exactly what self-attention is
for, and exactly what a lexicon gets wrong — in both directions, so you pay in missed abuse *and*
in false accusations against children.

The alternative to writing it here was to call someone's API. That fails on the specific thing this
app exists for: the text being screened is written by minors, and shipping it to a third party to be
logged is a worse privacy outcome than the abuse. So the model runs in-process, on the same machine
as the database, and the text never leaves it.

Bantu can carry this because numba's kernels are native. The maths below is a few hundred
`matmul`/`exp`/`sum` calls per text — the interpreter orchestrates, it does not compute.

## 2. The pipeline

```
raw text
  → normalise.b    fold confusables, undo leetspeak, collapse runs, strip zero-width
  → tokenizer.b    WordPiece, greedy longest-match, 64-token window
  → encoder.b      embed → 2 × transformer block → mean-pool → sigmoid head
  → 7 scores in [0,1]
```

**Normalisation is deliberately *not* the model's job.** A model asked to learn that `f.u.c.k`,
`fuuuuck`, `f**k`, `ƒuck` and `fцck` are one word spends its capacity on spelling instead of
meaning, and it can only ever learn the obfuscations present in the training data. A normaliser
closes the whole family at once, in code you can read and test:

| transform | example |
|---|---|
| Unicode confusable folding | `ƒuck` → `fuck`, `сunt` (Cyrillic с) → `cunt` |
| zero-width / combining strip | `f‌u‌c‌k` → `fuck` |
| leetspeak | `@`→`a`, `1`/`!`→`i`, `3`→`e`, `0`→`o`, `$`→`s`, `7`→`t` |
| run collapse | `fuuuuuck` → `fuuck` (two, not one — `too` must survive) |
| separator strip inside words | `f.u.c.k`, `f-u-c-k`, `f u c k` → `fuck` |

The separator strip is the one that needs care: it must not join ordinary words, so it only fires
when single characters are separated by the *same* separator three or more times in a row.

## 3. The model

| | |
|---|---|
| vocab | 8,192 WordPiece subwords, built from the corpus |
| context | 64 tokens (longer text is scored in overlapping windows, max taken) |
| layers | 2 |
| heads | 4 (`d_head` = 32) |
| `d_model` | 128 |
| `d_ff` | 512 |
| norm | pre-norm LayerNorm |
| activation | GELU (tanh approximation) |
| pooling | mean over unmasked positions |
| head | linear 128 → 7, sigmoid (multi-label, not softmax) |
| parameters | ≈ 1.46 M, of which 1.05 M is the embedding table |

**Multi-label, not multi-class.** A single message is routinely both a slur and a threat, and the
decision table downstream needs each category independently. Softmax would force them to compete.

The seven labels: `harassment`, `sexual`, `sexual_minor`, `grooming`, `threat`, `self_harm`,
`slur`.

### 3.1 Attention without `concatenate`

numba has no `concatenate`, and multi-head attention is conventionally written as
*concat the heads, then project*. It does not need to be. Because the output projection `W_o` is
block-structured with respect to the head axis, concat-then-project is a sum of per-head projections:

```
          ┌ head 0 ┐          ┌ W_o[0:32,  :] ┐
concat =  │ head 1 │   W_o =  │ W_o[32:64, :] │     concat · W_o  =  Σ  head_h · W_o[32h:32h+32, :]
          │ head 2 │          │ W_o[64:96, :] │                      h
          └ head 3 ┘          └ W_o[96:128,:] ┘
```

so the implementation is:

```bantu
$out = np.zeros([$T, $D], "f64");
$h = 0;
while ($h < $H) {
    $Qh = np.matmul($X, headSlice($Wq, $h));        // [T, dh]  — slices are free views
    $Kh = np.matmul($X, headSlice($Wk, $h));
    $Vh = np.matmul($X, headSlice($Wv, $h));
    $A  = softmaxRows(scaleAndMask(np.matmul($Qh, np.transpose($Kh, null)), $mask));
    np.add($out, np.matmul(np.matmul($A, $Vh), headRows($Wo, $h)), $out);
    $h = $h + 1;
}
```

`headSlice` and `headRows` are `np.slice` views — zero copy, zero allocation. The result is
bit-equivalent to the concat formulation and needs only `matmul`, `add` and a softmax.

### 3.2 Softmax and LayerNorm numerics

Softmax subtracts the row max before exponentiating (`np.max(axis, keepdims)` → `subtract` → `exp`
→ sum → divide). Without it, a logit of 800 overflows to `inf` and the row becomes `NaN`; with it,
the largest exponent is always `exp(0) = 1`.

Masked positions get `−1e9` *before* the max, not `0` after the softmax: zeroing afterwards leaves
the unmasked weights summing to less than one, which silently scales the output down in proportion
to how much padding the text happened to need.

LayerNorm uses numba's `var`, which is Welford — the textbook `E[x²] − E[x]²` loses all precision
once activations carry a large mean, and activations do.

## 4. The `.btw` weight format

A flat, self-describing container. Little-endian, which is every platform Bantu builds for.

```
offset  size            field
0       4               magic         "BTW1"
4       4  uint32       format version (1)
8       4  uint32       tensor count N
12      4  uint32       flags (0)
16      —               N tensor records, back to back
```

Each tensor record:

```
0       2  uint16       name length L
2       L               name, ASCII
2+L     1  uint8        dtype: 0 = f64
3+L     1  uint8        ndim
4+L     4*ndim uint32   shape
4+L+4n  8*prod(shape)   payload, f64 little-endian
```

The loader (`weights.b`) reads the **header only** through `readfile(path, "rb")` — a few hundred
bytes as a byte-list, which the interpreter handles fine — and then reads each payload with
`np.fromfile(path, "f64", $shape, $byteOffset)`, straight into the array with no Bantu list in
between. That is why `nd_fromfile` was added to numba (see [numba.md](numba.md) § Raw binary file
I/O): the whole 1.46 M-parameter model loads in about **12 ms**.

Decoding the payload in Bantu instead would mean a dozen float operations per value to reassemble
an IEEE-754 double from eight bytes, and a 1.46 M-element Bantu list costs roughly 190 bytes per
element — 280 MB to hold an 11 MB model. Not slow: impossible.

`tools/convert_hf.py` writes this format from any PyTorch `state_dict`, so a stronger pretrained
classifier is a file swap and not a code change. The tensor names are the contract:

```
tok_embed [V, D]   pos_embed [T, D]
blk{i}.ln1.g [D]   blk{i}.ln1.b [D]
blk{i}.wq [D, D]   blk{i}.wk [D, D]   blk{i}.wv [D, D]   blk{i}.wo [D, D]
blk{i}.ln2.g [D]   blk{i}.ln2.b [D]
blk{i}.ff1 [D, F]  blk{i}.ff1b [F]    blk{i}.ff2 [F, D]  blk{i}.ff2b [D]
ln_f.g [D]         ln_f.b [D]
head.w [D, 7]      head.b [7]
```

## 5. Training

`train.b` does forward and a hand-written backward pass — numba has every operation the backward
needs, since a transformer's gradients are the same matmuls transposed. Adam (β₁ 0.9, β₂ 0.999,
ε 1e-8), linear warmup over 200 steps then cosine decay, gradient-norm clipping at 1.0, batch 16,
dropout omitted (the model is small and the corpus is templated, so the regularisation that matters
is the corpus, not the noise).

Cost per step is roughly 3 GFLOP against numba's measured 7.8 GFLOP/s matmul rate, so about
0.4 s/step and a 2,000-step run in the region of fifteen minutes. That is a one-off; inference is
10–20 ms.

Gradient correctness is not assumed. `tests/lugha_train_test.b` checks every parameter group's
analytic gradient against a central finite difference, `(f(θ+h) − f(θ−h)) / 2h` with h = 1e-5, and
requires agreement to 1e-6 relative. A transformer with one sign error in the backward pass still
trains — to something worse — and nothing but this test notices.

## 6. What this model is, honestly

The weights shipped in this repo are trained on a corpus assembled here: hand-authored labelled
lines plus a templated expansion crossing lexicons with contexts, and a deliberate body of **hard
negatives** (*"this game is sick"*, *"I'll kill it at the audition"*, *"my mum is a nurse"*,
*"we killed that presentation"*) so the model learns context rather than vocabulary. Held-out
metrics are reported in `lugha/corpus/README.md` and regenerated by `train.b` on every run.

What it is: a defensible baseline that beats keyword matching on negation, target and obfuscation,
running locally with no third party in the loop.

What it is not: a production safety system. It is one stage of a pipeline whose other stages are a
deterministic lexicon, a human review queue, and user reporting — and the thresholds are set so
that a borderline score routes a post *to a human*, not to an automatic verdict. Any claim stronger
than that would need a corpus orders of magnitude larger and an evaluation against real reported
content, which this repo does not have and should not pretend to.

## 7. Alternatives rejected

| Option | Why not |
|---|---|
| Lexicon only | Cannot read negation or target; fails in both directions, and the false-positive direction means accusing children |
| Third-party moderation API | The text is written by minors; shipping it off-box to be logged is the worse privacy outcome, and it makes the app unusable offline |
| Bag-of-words + logistic regression | Cheap and genuinely decent on slurs, useless on *"I'll kill it at the audition"* — no word order at all |
| Train a larger model (6 layers, d=512) | ~20× the training cost for a corpus that cannot support the capacity; it would memorise the templates |
| Char-level model, no tokenizer | Needs far more context length for the same text and learns the obfuscations the normaliser already closes |
| IEEE-754 decoding in pure Bantu | 280 MB of `Value`s for an 11 MB model; `nd_fromfile` exists for this reason |
| `concatenate` the attention heads | numba has no `concatenate`; the per-head sum is algebraically identical and needs nothing new |
