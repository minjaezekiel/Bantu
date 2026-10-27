# `lugha/corpus` — the moderation training corpus

Everything here is generated, deterministically, from two files you can read:

| file | what it is |
|---|---|
| `slots.b` | the word lists the templates draw from |
| `templates.b` | 150 labelled sentence patterns |
| `hard_seed.tsv` | 71 hand-written sentences that follow no pattern |
| `generate.b` | expands, augments, normalises, splits |
| `build_vocab.b` | induces the WordPiece vocabulary by BPE merges |
| `METRICS.md` | held-out numbers, rewritten by every training run |

```sh
bantu run lugha/corpus/generate.b      # -> moderation_{train,eval,hard}.tsv
bantu run lugha/corpus/build_vocab.b   # -> lugha/vocab.txt
```

Each row is `<7 label bits>\t<text>`, and the label order is fixed everywhere in
`lugha`: `harassment`, `sexual`, `sexual_minor`, `grooming`, `threat`,
`self_harm`, `slur`.

The text is stored **already normalised**, which is the point: the model trains
on exactly the surface form `lugha.score()` will hand it at inference. A
mismatch there is a silent accuracy loss that no test would catch.

## The split is over generators, not rows

The first version shuffled all the generated rows and took 15% as held-out. It
reported a **macro-F1 of 1.000**, which is how the mistake was found: train and
eval came from the same templates, so an eval row was a training row with
different words in the slots. The model had learned 150 sentence patterns, and
the evaluation asked it to recognise those 150 sentence patterns.

So the split is over the **generators**:

- about a fifth of the **templates** are held out, stratified by primary label
  so every label has both training and evaluation phrasings;
- about a fifth of each **slot's** values are held out too.

A held-out sentence is therefore a phrasing the model has never seen, filled
with vocabulary it has never seen in that role. `moderation_hard.tsv` goes
further: hand-written sentences with no template behind them at all. It is never
used for training **or** for calibrating a threshold — a number measured on data
that shaped the model is not a measurement.

The vocabulary is built over all three splits. That is not label leakage — a
subword inventory carries no labels — and it matches how tokenizers are actually
built, on a large unlabelled corpus chosen before any split exists. What it does
mean is that held-out words are tokenizable rather than falling back to
characters, so the reported numbers do not include "the tokenizer had never seen
this word" as a source of error.

## Hard negatives are half the work

A classifier that flags *"I'll kill it at the audition"* is worse than useless:
it trains people to ignore it, and on this app it would be refusing a
15-year-old's post about a school play. So the corpus deliberately contains the
dangerous words used innocently —

> *"that exam killed me"* · *"this heat is killing me"* · *"we killed that
> presentation"* · *"that goal was sick"* · *"dont tell mum i ate the last
> mandazi"* · *"im 15 and i love chess"* · *"sex education was mostly diagrams
> and silence"* · *"this weather is shit"* · *"i coach the under 15s"*

— and roughly a third of all rows carry no label at all.

## Augmentation, and why it runs before normalisation

A quarter of rows are obfuscated before being normalised: a masked letter, a
stretched letter, a dot-separated word, a leetspeak substitution. What survives
into the corpus is only the **residue** the normaliser cannot undo — `sht` from
`sh*t`, `fuuck` from `fuuuuck`. That residue is what the model has to learn; the
rest the normaliser already handles, and training on it would teach nothing.

## On the slur list

`slots.b` contains widely-recognised profane insults. It does **not** contain an
exhaustive list of ethnic, religious or sexual-orientation slurs, and
`safety/lexicons/` in the app ships nearly empty for the same reason:
publishing such a list is itself a harm, and a worse one than the gap it leaves.
Those terms are exactly the cases the deterministic lexicon handles with perfect
recall once an operator adds the ones their policy requires, and exactly the
cases users report.

## What these numbers are worth

See `METRICS.md` for the current run. Honestly: around **0.6 macro-F1** on
held-out templates and **0.5** on the hand-written set. That is a defensible
baseline which beats keyword matching on negation, target and obfuscation, and
it is the reason the app's moderation policy does not let the model block a post
on its own — see `docs/one-taylor-architecture.md` section 4.1.

It is not a production safety system. A corpus orders of magnitude larger, and
an evaluation against real reported content, is what that would take.
