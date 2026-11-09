# bonnx — Decisions

Ecosystem-wide decisions are `E*` in
[`../docs/ecosystem-architecture.md`](../docs/ecosystem-architecture.md);
btorch's are `B*`.

---

### O1 — No general protobuf package; a minimal reader here, native only if measured
Cites [E9]. ONNX has a fixed schema, and a general protobuf library is a year
of work for a feature nobody asked for.

The reader is written in Bantu first and profiled on a 100 MB model. A `pb_`
native family is added **only if the measurement says it is the bottleneck.**

The reason to resist the instinct: btorch's phase 9 found that **two of three
performance suspects were wrong**, and the real cost was somewhere nobody had
guessed. The structural argument here is that a 100 MB model is a few thousand
protobuf *messages* wrapping a few hundred large tensor blobs, and the blobs go
through `nd_frombytes` in one call each — so the per-field Bantu work may be
trivial. Or it may not.

---

### O2 — Interpret the graph; do not rebuild an `nn.Module`
The central design decision, and the non-obvious one.

ONNX graphs are **not layer trees.** Exporters fuse, split and reorder, so a
`Conv` followed by `BatchNormalization` may arrive as two nodes or as one
folded `Conv`. A reconstruction that only handles graphs resembling btorch
layers fails on most real exports — and fails *by not recognising a pattern*,
which is very hard to report usefully.

Interpreting needs only that each operator maps to btorch tensor operations.
Since those are differentiable, **an interpreted graph is already trainable**,
which removes most of reconstruction's advantage.

`to_module` survives as an opt-in (O7) that is *allowed* to fail. The
interpreter is not.

---

### O3 — f32 is a prerequisite, not a nice-to-have
ONNX models are almost universally f32. Loading into f64 works — every f32 is
exactly representable — but it doubles resident memory on the one thing that is
already large, and makes every intermediate differ in the last bits from the
exporting framework. Differential testing to 1e-5 against `onnxruntime` is only
meaningful in the dtype the reference computed in.

f64 loading stays available, and is useful for exactly one thing: telling
whether a disagreement is a precision artefact or a real bug.

---

### O4 — An unsupported operator fails by name, at load time
ONNX has roughly 190 operators. Covering all of them is not a goal, and
pretending to would be worse than not trying.

So the refusal is a **tier of its own** (tier 3): an unsupported operator
raises at load time with the node name, the opset version, **and how many nodes
of that type the graph contains** — because "this model needs `Resize`, 14
times" is actionable and "unsupported operator" is not.

Never a silent wrong answer. Never a late failure mid-inference.

---

### O5 — Differential fixtures are committed; Python is not a test dependency
The method already proven twice in this project (pillow's differential
references, btorch's embedded PyTorch values): a dev-only script in `tools/`
exports models, runs them under `onnxruntime`, and writes inputs and outputs to
a SHA-256-verified `.btsd` fixture. The fixture is committed; the script is not
a build dependency.

Per-**operator** fixtures as well as whole-model ones, because a whole-model
match is reassuring but a whole-model mismatch is nearly undiagnosable.
