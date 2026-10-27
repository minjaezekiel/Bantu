# `lugha` — NLP for Bantu

Text normalisation, a WordPiece tokenizer, and a transformer encoder with a
multi-label classification head — written in Bantu over [`numba`](numba.md)'s
native kernels. Its first caller is One Taylor's moderation pipeline.

- **Source:** [`lugha/`](../lugha) · **Design:** [`lugha-architecture.md`](lugha-architecture.md)
- **Tests:** `tests/lugha_tokenizer_test.b` (61 assertions),
  `tests/lugha_train_test.b` (37, every gradient against finite differences)
- **Metrics:** [`lugha/corpus/METRICS.md`](../lugha/corpus/METRICS.md)

```bantu
include "lugha" as lugha;

lugha.init("model/toxicity.btw", "model/vocab.txt");

$r = lugha.score("you are such a pathetic idiot");
print(str($r["flags"]["harassment"]));        // true
print(str($r["scores"]["harassment"]));       // 0.87
print(str($r["blocking"]));                   // labels precise enough to act on
```

---

## What it is for

A keyword list cannot read context. *"I'll kill you"* and *"I'll kill it at the
audition"* share a verb and differ entirely; *"she's sick"* is an insult or a
diagnosis depending on three words either side. Negation, target and sarcasm are
what self-attention is for, and what a lexicon gets wrong in **both**
directions — so you pay in missed abuse *and* in false accusations.

The alternative to writing it here was calling someone's API. That fails on the
specific thing it is for: the text being screened is written by minors, and
shipping it off-box to be logged is a worse privacy outcome than the abuse.

## API

| call | returns |
|---|---|
| `init($modelPath, $vocabPath)` | `{vocab, labels, layers, d_model, blocking}` |
| `ready()` | bool |
| `labels()` | the seven label names |
| `thresholds()` | dict label → operating point |
| `reliable()` | dict label → may this label *block*, or only advise? |
| `normalise($text)` | the canonical form the model sees |
| `tokens($text)` | the WordPiece pieces, for debugging a score |
| `score($text)` | see below |
| `margin($result)` | how close the nearest label came to its threshold |

`score()` returns:

```
{
  "scores":     {label: 0..1},        every label
  "flags":      {label: bool},        at or over that label's threshold
  "blocking":   {label: score},       flagged AND precise enough to act on
  "advisory":   {label: score},       flagged but only precise enough to advise
  "flagged":    bool,
  "top":        the label furthest OVER its own threshold,
  "normalised": what the model actually read,
  "windows":    how many 64-token windows the text needed
}
```

The seven labels: `harassment`, `sexual`, `sexual_minor`, `grooming`, `threat`,
`self_harm`, `slur`. Multi-label, not multi-class — a message is routinely both
a slur and a threat, and softmax would make them compete.

## Three things worth knowing before you rely on it

**`blocking` and `advisory` are not the same evidence.** Each label's operating
point is calibrated on held-out data, and a label whose precision at that point
is under 0.6 is **advisory**: it is reported, but a caller must not take an
irreversible action on it. One Taylor's rule is that advisory flags route a post
to a human and never delete one, and that rule exists because the earlier
version — where any flag could block — blocked a 15-year-old's "Scored twice at
practice today" for grooming.

**Normalisation is a separate stage, and it runs first.** `f.u.c.k`, `ƒuck`,
`FUUUCK`, `f**k` and `fцck` all become `fuck` before the model sees anything, so
the model never spends capacity on spelling. Run `lugha.normalise()` on anything
you store alongside a score, or your training and inference will disagree.

**A bare number survives normalisation.** Leetspeak folding applies only inside
a run that already contains a letter, so `i am 13` stays `i am 13`. That is the
single most load-bearing fact a grooming classifier can read, and a naive
digit-to-letter rule turns it into `i am ie`.

## Long text

A text longer than the 64-token window is scored in overlapping windows and the
per-label **maximum** is taken. Truncating would let a paragraph of
pleasantries followed by one threat score as pleasantries, which is how these
classifiers get evaded in practice.

## The model file

`.btw` is a flat, self-describing tensor container — magic, version, tensor
count, then a record per tensor with its name, dtype and shape, then the
payloads. The header is parsed from a Bantu string; the payloads go straight
into arrays through `np.fromfile` at their recorded byte offsets, which is why
[`nd_fromfile`](numba.md#raw-binary-file-io) was added to numba. The whole
1.1M-parameter model loads in about 12 ms.

`tools/convert_hf.py` writes the same format from a PyTorch `state_dict`, so a
stronger pretrained classifier is a file swap and not a code change. The tensor
names are the contract — see
[`lugha-architecture.md`](lugha-architecture.md#4-the-btw-weight-format).

## Training and calibration

```sh
bantu run lugha/corpus/generate.b      # the labelled corpus, deterministic
bantu run lugha/corpus/build_vocab.b   # BPE merges -> vocab.txt
bantu run lugha/train.b                # forward + hand-written backward, Adam
bantu run lugha/calibrate.b            # re-derive thresholds WITHOUT retraining
```

`calibrate.b` is separate on purpose: "what are the right thresholds" is a
policy question that changes whenever the floors change or a reviewer says the
queue is too noisy, and re-answering it used to cost twelve minutes of gradient
descent that could not change the answer.

## Cost

| | measured |
|---|---|
| `score()` on a short text | **9–10 ms** |
| `score()` at the full 64-token window | 18 ms |
| loading the model | 12 ms |
| one training step, batch 16 | ~320 ms |
| a 2,200-step training run | ~12 minutes |

Inference belongs at **write** time. 10 ms on a read path is 10 ms added to
every other client on that `sua` worker, because `sua.sqlite` and the handler
share a thread.

## What this model is not

The weights in this repository are trained on a corpus generated in
`lugha/corpus/`: hand-authored lines, a templated expansion, and a deliberate
body of hard negatives. The split is over **generators** — a fifth of the
templates and a fifth of each slot's values are held out — because an earlier
random row split reported a macro-F1 of 1.000, which measured memorisation of
89 sentence patterns and nothing else.

Honest numbers, from `lugha/corpus/METRICS.md`: **0.61 macro-F1** on held-out
templates, **0.50** on 71 hand-written sentences that follow no template.

That is a defensible baseline that beats keyword matching on negation, target
and obfuscation. It is not a production safety system, and it is one stage of a
pipeline whose other stages are a deterministic lexicon, user reports, and a
human review queue.
