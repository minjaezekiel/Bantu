# Operator lexicons

One term per line, lowercase, `#` for a comment. A file here **adds to** the
defaults compiled into `safety/lexicon.b`; it does not replace them.

    slur.txt  sexual.txt  sexual_minor.txt  grooming.txt  threat.txt  self_harm.txt

Terms are matched against **normalised** text — lugha's normaliser has already
folded confusables, undone leetspeak, collapsed character runs and stripped
zero-width characters — so one entry `fuck` covers `f.u.c.k`, `ƒuck`, `FUUUCK`
and `f**k`. Write the plain form; do not try to enumerate spellings.

Matching is on **word boundaries**, so `fag` does not fire inside `fagott` and
`cp` does not fire inside `cpu`.

## Why these files ship nearly empty

This repository does not contain an exhaustive list of ethnic, religious or
sexual-orientation slurs. Publishing one is itself a harm, and a worse one than
the gap it leaves: those terms are exactly the cases the transformer and the
user report system handle well, because they are unambiguous in context and
people report them.

Add the terms your jurisdiction, community and moderation policy require. The
files are `.gitignore`d by pattern so an operator's list does not end up in a
public diff.
