# bonnx — Roadmap

ONNX model import for Bantu. Train on a GPU in PyTorch; deploy with one static
Bantu binary at the edge.

Design: [`../docs/bonnx-architecture.md`](../docs/bonnx-architecture.md).
Decisions: [`DECISIONS.md`](DECISIONS.md). Ecosystem context:
[`../docs/ecosystem-architecture.md`](../docs/ecosystem-architecture.md)
(stage 4).

**Prerequisites:** ecosystem stage 1 item **D2** (`f32`). ONNX models are
almost universally f32; loading them into f64 doubles resident memory and makes
every intermediate differ in the last bits from what the exporting framework
computed, which is exactly what differential testing needs to match.

**Gates:** the usual six tiers, with tier 2 being **differential against
`onnxruntime`** — the only way to know an import is right.

| tier | proves | form |
|---|---|---|
| 1. feature | each operator and the loader do what they claim | `tests/bonnx_*_test.b` |
| 2. **differential** | agreement with `onnxruntime` to 1e-5 in f32 | committed `.btsd` fixtures, generated offline |
| 3. **refusal** | an unsupported operator fails **by name**, at load time | every unsupported op asserted to raise |
| 4. stress | a 100 MB model; 1,000-node graphs; adversarial protobuf | malformed input raises, never crashes |
| 5. regression | nothing else broke | the whole suite |
| 6. sanitizers | ASan + UBSan over the full bonnx suite | — |

---

## Phase O0 — Tracking docs
- [x] `docs/bonnx-architecture.md`
- [x] `bonnx-suite/ROADMAP.md`, `DECISIONS.md`

## Phase O1 — The protobuf wire reader

| Item | How | Test | Status |
|---|---|---|---|
| varint, tag, wire types 0/1/2/5 | pure Bantu over `nd_frombytes` | round-trip against hand-encoded bytes; 10-byte varints; truncation raises | [ ] |
| **skip unknown fields by wire type** | never by guess | a message with injected unknown fields of every wire type parses identically | [ ] |
| weight payloads | read via `nd_frombytes` out of the length-delimited region, **one call per tensor** | a 10M-parameter tensor loads in one call | [ ] |
| **profile on a 100 MB model** | decides whether a `pb_` native family is needed ([O1] in DECISIONS) | the measurement recorded whatever it says | [ ] |

Skipping unknown fields correctly is what gives this reader a lifespan: ONNX
gains fields with every opset, and a reader that errors on an unrecognised
field number breaks on next year's exports.

## Phase O2 — The schema subset
`ModelProto`, `GraphProto`, `NodeProto`, `AttributeProto`, `TensorProto` —
hand-written, not generated. Initializer binding, input/output naming,
`value_info` shapes.

## Phase O3 — Operators: the MLP set
`Gemm` `MatMul` `Add` `Mul` `Relu` `Sigmoid` `Tanh` `Softmax` `LogSoftmax`
`Flatten` `Reshape` `Transpose` `Identity` `Constant`.

Each table entry supplies **both** `infer` (shape) and `run`. Per-operator
differential fixtures, because a whole-model mismatch is nearly undiagnosable
while a per-operator one names the culprit.

## Phase O4 — Operators: the CNN set
`Conv` `MaxPool` `AveragePool` `GlobalAveragePool` `BatchNormalization` `Pad`
`Concat` `Dropout` `Clip` `LeakyRelu` `Gelu` `Erf`.

`Conv` maps to `bt.conv2d`, which is im2col composed with matmul ([B18]) — so
**no new gradient anywhere in this package.** Depthwise/grouped convolution is
forced into btorch here by the MobileNetV2 gate.

## Phase O5 — Operators: shape and reduction
`Shape` `Slice` `Gather` `Squeeze` `Unsqueeze` `Split` `ReduceMean` `ReduceSum`
`ReduceMax` `ArgMax` `Cast` `Where` `Expand`.

Scope fixed by **surveying ten exported torchvision models first**, rather than
by guessing which operators matter.

## Phase O6 — The graph executor
Topological order, shape inference at load, initializer binding, intermediate
lifetime (free a tensor once its last consumer has run — otherwise a 50-layer
network holds every activation).

Because btorch's operations are differentiable, **an interpreted graph is
already trainable**; gradients flow with no extra work.

## Phase O7 — `to_module` (opt-in, allowed to fail)
Pattern-matches recognised sequences into a real `nn.Sequential`, returning
`null` where it cannot. For fine-tuning and for inspecting an imported model
with btorch's own tools. The interpreter (O6) is the path that must always
work.

## Phase O8 — Export — **unscheduled**
Import is the asymmetric win. Export matters once models are trained here at
scale.

---

## The accuracy gate

Not "it loads":

> **A ResNet-18 exported from PyTorch, imported by `bonnx`, classifies a fixed
> image set identically to `onnxruntime` — same argmax on every image, logits
> agreeing to 1e-4.** Plus MobileNetV2, for depthwise convolution.

A model that loads and produces plausible-looking but wrong logits is the
failure mode this package must not have, and only a reference comparison
catches it.

---

## Where to resume
**O0 done; O1 next.** Blocked on stage 1 item D2 (`f32`).
