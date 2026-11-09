# Stage 4 — `bonnx`: ONNX model import

Phases **O0–O8** in [`../bonnx-suite/ROADMAP.md`](../bonnx-suite/ROADMAP.md).
Context: [`ecosystem-architecture.md`](ecosystem-architecture.md).

**The strategic case.** Train on a GPU in PyTorch or TensorFlow; deploy with
one static Bantu binary at the edge. This package buys access to the world's
existing models without competing on training throughput — a race this stack
cannot win and should not enter. It is the highest-leverage single package on
the roadmap for that reason, not because importing a file format is
interesting.

---

## 1. Hard dependency on stage 1's f32

**ONNX models are almost universally f32.** Loading them into f64 works — it
widens exactly, since every f32 is representable as an f64 — but it doubles
memory on the one thing that is already large (a 100 MB model becomes 200 MB
resident) and makes every intermediate result differ in the last bits from what
the exporting framework computed.

With f32 (D2), weights load byte-for-byte and activations match the reference
framework to f32 precision, which is what differential testing against
`onnxruntime` requires. This is the concrete reason stage 1 precedes stage 4
rather than the other way round.

f64 loading stays available as a fallback and is useful for one thing:
diagnosing whether a disagreement with the reference is a precision artefact or
a real bug.

---

## 2. Reading the file

### 2.1 A minimal protobuf reader, in Bantu

ONNX is protobuf. Decision [E9]: **no general protobuf package.** ONNX has a
fixed schema, and a general protobuf library is a year of work for a feature
nobody asked for.

The wire format needs exactly four things:

| piece | what |
|---|---|
| varint | base-128, little-endian groups, continuation bit |
| tag | `(field_number << 3) \| wire_type` |
| wire types | `0` varint, `1` 64-bit, `2` length-delimited, `5` 32-bit |
| skip | unknown fields must be skipped by wire type, never by guess |

Skipping unknown fields correctly is the part that matters for longevity: ONNX
gains fields with every opset, and a reader that errors on an unrecognised
field number breaks on next year's exports. A reader that skips them keeps
working.

Weight payloads are read with `nd_frombytes` directly out of the
length-delimited region — no element-by-element decode, which would be
1.78 µs per weight.

### 2.2 Whether the reader must be native

**Open, and settled by measurement, not suspicion** ([E9]). Write it in Bantu
first, profile it on a 100 MB model, and promote a varint/field scanner to a
`pb_` native family **only if it is the bottleneck.**

The reason to resist the instinct: btorch's phase 9 found that **two of three
performance suspects were wrong**, and the real cost was somewhere nobody had
guessed. The structural argument here is that a 100 MB model is a few thousand
protobuf *messages* wrapping a few hundred large tensor blobs — the blobs go
through `nd_frombytes` in one call each, so the per-field Bantu work may be
trivial. Or it may not. Measure.

### 2.3 The schema subset

Only what is needed, hand-written rather than generated:

```
ModelProto      -> opset_import, graph
GraphProto      -> node[], initializer[], input[], output[], value_info[]
NodeProto       -> op_type, name, input[], output[], attribute[]
AttributeProto  -> name, type, i/f/s/ints/floats/t
TensorProto     -> dims[], data_type, raw_data | float_data | int64_data
```

---

## 3. Execution: interpret the graph, do not rebuild a Module

The central design decision, and the non-obvious one.

| approach | verdict |
|---|---|
| **(a) interpret the graph** — hold nodes and initializers, execute in topological order over btorch tensors | **chosen as the primary path** |
| (b) reconstruct an `nn.Module` tree — map `Conv` → `nn.Conv2d`, etc. | opt-in, pattern-matched, secondary |

**(a) is strictly more general.** ONNX graphs are not layer trees; exporters
fuse, split and reorder, so a `Conv` followed by `BatchNormalization` may
arrive as either two nodes or one folded `Conv`. A reconstruction that only
handles graphs resembling btorch layers fails on most real exports, and fails
*by not recognising a pattern* — which is hard to report usefully.

Interpreting the graph needs only that each operator maps to btorch tensor
operations. Since those operations are differentiable, **an interpreted graph is
already trainable**: gradients flow through it without any extra work. That
removes most of (b)'s advantage.

**(b) survives as `bonnx.to_module($model)`**, which pattern-matches recognised
sequences and returns a real `nn.Sequential` where it can, `null` where it
cannot. Useful for fine-tuning with standard optimisers and for inspecting an
imported model with btorch's own tools. It is allowed to fail; the interpreter
is not.

### 3.1 Shape inference

Needed before execution, because `Reshape(-1, ...)` and `Gemm`'s transpose
flags cannot be resolved without it, and because a shape error should be
reported at load time against a node name rather than mid-inference as an
array error. Implemented per operator alongside the operator itself — each
entry in the operator table supplies both `infer` and `run`.

---

## 4. Operator coverage, and refusing honestly

ONNX has roughly 190 operators. Covering all of them is not a goal and
pretending to would be worse than not trying.

| tier | operators | phase |
|---|---|---|
| **MLP** | `Gemm` `MatMul` `Add` `Mul` `Relu` `Sigmoid` `Tanh` `Softmax` `LogSoftmax` `Flatten` `Reshape` `Transpose` `Identity` `Constant` | O3 |
| **CNN** | `Conv` `MaxPool` `AveragePool` `GlobalAveragePool` `BatchNormalization` `Pad` `Concat` `Dropout` `Clip` `LeakyRelu` `Gelu` `Erf` | O4 |
| **shape/reduce** | `Shape` `Slice` `Gather` `Squeeze` `Unsqueeze` `Split` `ReduceMean` `ReduceSum` `ReduceMax` `ArgMax` `Cast` `Where` `Expand` | O5 |
| **sequence** | `LSTM` `GRU` `RNN` `Attention` `LayerNormalization` | after stage 5 |

**An unsupported operator must fail by name, at load time, with the node name
and the opset version.** Never a silent wrong answer, never a late failure. The
error also lists how many nodes of that type the graph contains, because "this
model needs `Resize`, 14 times" is actionable and "unsupported operator" is not.

Every btorch operation these map onto already exists — `Conv` becomes
`bt.conv2d`, which is im2col composed with matmul (B18), so there is no new
gradient anywhere in this package.

---

## 5. Differential testing against `onnxruntime`

The only way to know an import is right. The method is pillow's, already proven
in this project: **generate reference values with the reference implementation
offline, embed them in the test, and never make the test depend on having the
reference installed.**

| # | step |
|---|---|
| 1 | a Python script (dev-only, in `tools/`) exports small models from PyTorch to ONNX |
| 2 | it runs them under `onnxruntime` on fixed seeded inputs |
| 3 | it writes inputs and outputs to a `.btsd` fixture, SHA-256 verified |
| 4 | `tests/bonnx_differential_test.b` loads the fixture, runs the model under `bonnx`, and asserts agreement to **1e-5** in f32 |
| 5 | the fixture is committed; the Python script is not a build dependency |

Per-operator fixtures as well as whole-model ones: a whole-model match is
reassuring but a mismatch in it is nearly undiagnosable, while a per-operator
fixture names the culprit.

### 5.1 The accuracy gate

The gate is not "it loads". It is:

**A ResNet-18 exported from PyTorch, imported by `bonnx`, classifies a fixed
set of images identically to `onnxruntime` — same argmax on every image, logits
agreeing to 1e-4.** Plus a MobileNetV2 for depthwise convolution, which is
also what forces grouped convolution into btorch.

A model that loads and produces plausible-looking but wrong logits is the
failure mode this package must not have, and only a reference comparison
catches it.

---

## 6. What is out

| item | why |
|---|---|
| **ONNX export** | import is the asymmetric win; export matters once models are trained here at scale. Deferred to O8, unscheduled |
| training-mode graphs (`TrainingInfoProto`) | effectively unused in practice |
| quantized ONNX (`QLinearConv`) | revisit after stage 6's `q8_` lands — then it is cheap |
| control flow (`If`, `Loop`, `Scan`) | rare outside sequence models; needs subgraph execution |
| custom operators / domains | out of scope |
| opset < 7 | pre-2017; not worth the branches |

---

## 7. Open items

| # | item | settled by |
|---|---|---|
| 1 | does the protobuf reader need `pb_`? | profiling a 100 MB model, O1 |
| 2 | how much of a real export does the O3+O4 set cover? | survey ten exported torchvision models before committing to O5's list |
| 3 | does `to_module` pattern matching pay for itself? | if fine-tuning an imported model is a real request; otherwise drop it |
| 4 | external-data models (weights in a sidecar file) | common above 2 GB; needs a path-resolution and sandboxing decision |
