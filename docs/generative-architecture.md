# Generative models in Bantu — the design, and what the measurements say

**Objective.** Generative-adversarial and related models that produce
**images**, **voices** and **video**, with the underlying packages designed to
support them rather than retrofitted later.

This document exists because the obvious plan — "add GAN layers to btorch" — is
the wrong plan, and a day of measurement says so. The package layer is already
within 5% of its kernels. Every hour a generative model will spend is in four
specific places, three of them below the packages entirely. Designing the
package surface without fixing those would produce a GAN API that is correct,
tested, and unusable.

It is the companion to [ecosystem-architecture.md](ecosystem-architecture.md),
which already places FFT in numba ([E4]), and RNN/attention/ConvTranspose2d in
btorch. This document does not relitigate those. It adds the audio layer, the
generative layer, and the performance work those two depend on — and it puts a
measured number on each.

---

## 1. What already works

Measured on this machine (12 hardware threads, f64, **1 thread** unless a row
says otherwise — 1 is numba's default, which §3 is about).

### 1.1 Audio, end to end, with no new kernels

| step | result |
|---|---|
| decode a 16 kHz mono WAV | ~60 lines of pure Bantu: walk the RIFF chunks, `nd_frombytes(…, "i16")`, scale by 1/32768 |
| verified against | a WAV written by Python's `wave` module — peak reads back `0.610352`, exactly the `20000/32768` written |
| STFT | a matmul against a real DFT basis `[nbins, n_fft]` |
| STFT verified | 1000 / 2000 / 4000 Hz tones peak in bins 25 / 50 / 100 — exact, not approximate |
| log-mel, `n_fft=400`, `hop=160`, 80 mels | 1 s → 17 ms · 10 s → 102 ms · 60 s → 613 ms = **98× realtime** |
| mel filterbank construction | 140 ms, once per process |

**There is no FFT kernel in the ecosystem, and the audio frontend does not need
one.** An STFT as a dense matmul is `O(n_fft²)` per frame against the FFT's
`O(n_fft log n_fft)`, but at `n_fft = 400` the constant decides it: the matmul
runs in native GEMM, a radix-2 FFT written in Bantu would run in the evaluator.
98× realtime is enough to feed any trainer this stack can run.

So `nd_rfft` is **not** on the critical path for voice. It stays an [E4]
commitment for `bsci.signal`, for long windows and for phase-accurate
reconstruction — not a blocker.

### 1.2 A speech model trains

A Conv1d keyword spotter, 134k parameters, `[80 mels, 101 frames]` input,
35 classes, batch 32:

| | |
|---|---|
| forward + backward + Adam step | 292 ms |
| per clip | 9.1 ms |
| **one epoch of Speech Commands** (85k clips) | **12.9 min** |

Twenty epochs is an afternoon. Keyword spotting, speaker identification, voice
activity detection, language and emotion classification are all reachable now.

### 1.3 An image GAN trains

A DCGAN at 64×64, batch 16. Upsampling is `repeat_interleave`, which btorch
already differentiates — and which modern generators prefer over transposed
convolution anyway, because it cannot produce checkerboard artifacts.

| model | parameters | per step | 50k steps |
|---|---|---|---|
| base 128 | G 361,731 + D 99,489 | 1,875 ms | 26.0 h |
| base 256 | G 916,995 + D 392,513 | 4,627 ms | 64.3 h |
| base 256, **8 threads** | same | 3,196 ms | **44.4 h** |

Reachable, in days. That is the honest baseline the rest of this document tries
to improve, and §3 says by how much.

---

## 2. The package layer is not the problem

This was the surprise, and it redirected the whole plan.

| measurement | result |
|---|---|
| native `nd_im2col` + `nd_matmul`, `[16,64,32,32]` k=3 | 270.1 ms (4.47 GFLOP/s) |
| the same conv through `bt.conv2d` (autograd tensors) | 284.8 ms — **1.054× the kernel** |
| forward + backward | 757.3 ms = 2.80× the kernel, which is near-optimal: backward is ~2× forward's work |
| `bt.relu` with grad vs without, 2.1M elements | 4.0 ms vs 4.6 ms |
| native call dispatch | 5.87 µs |

**btorch's autograd costs about 5%.** A tree-walking interpreter was the prime
suspect and it is not guilty: at ML tensor sizes one native call per op amortises
the 5.87 µs dispatch into nothing, and btorch makes one call per op.

The consequence for this project is specific and worth stating plainly:

> **Writing generative layers in pure Bantu costs nothing measurable. Making
> them fast is entirely a kernel problem.**

That is the opposite of the usual conclusion about interpreted ML, and it is why
§4 designs the package surface freely while §3 does the optimisation work
underneath it.

---

## 3. Where the time actually goes

Four places. Ordered by measured payoff, which is not the order intuition gives.

### G1 — Threads default to 1, and the ML stack never asks

| 1024³ f64 matmul | GFLOP/s |
|---|---|
| 1 thread | 5.94 |
| 2 | 11.97 |
| 4 | 21.91 |
| 8 | **31.58** |
| 12 (all) | 30.83 |
| 8, f32 | 31.58 — **no dtype gain** |

A **5.3× speedup, available with one call**, that nothing in btorch or bkit
makes. The knob (`nd_set_threads`) was literally unreachable from package code
until it was wrapped in bkit phase K1; before that no Bantu program could ask
for it at all.

The interpreter default stays 1 — see [G7](#g7), which corrects the reason the
existing documentation gives for it. **The ML packages must opt in on the user's
behalf**, because for them the default is simply 1/5 of the machine.

Note also that f32 buys *nothing* in GEMM (31.58 either way), which says the
GEMM is memory- or blocking-bound rather than SIMD-width-bound, and predicts
[G3](#g3).

### G2 — `im2col` memory traffic dominates convolution

One 32→32 conv at 64×64, batch 16, on 8 threads: **194 ms for 1.21 GFLOP =
6.2 GFLOP/s**, while plain GEMM on the same threads does 31.6.

The cause is the `im2col` matrix: `[N·OH·OW, Cin·k·k]` = `[65536, 288]` =
18.9M elements = **151 MB written and read back** to perform 1.21 GFLOP. The
arithmetic is not the cost; manufacturing its operand is.

This is why 8 threads bought the GAN only **1.45×** instead of 5.3×: a convnet
is not a GEMM benchmark.

**Design:** a direct blocked convolution kernel that never materialises the
column matrix, keeping a tile of it in cache. This is the single largest win
available to image and audio generation, because every layer of both a generator
and a discriminator is a convolution.

**Rejected:** threading `im2col`. It makes a memory-bound copy wider, not
cheaper, and leaves the 151 MB intact.

### G3 — Elementwise ops run at a fraction of memory bandwidth

2,097,152 f64 elements (`[16,32,64,64]`), 8 threads:

| op | time | effective |
|---|---|---|
| `nd_add`, `nd_multiply`, `nd_maximum` | 4.25 ms each | **7.9 GB/s** |
| `nd_transpose` (a view) | 0 ms | — |
| `bt.repeat_interleave` ×2 (the upsample) | 25.1 ms | — |

7.9 GB/s, on a machine whose memory system does an order of magnitude more.
Identical timings for three different ops is itself the diagnosis: they are all
bound by the same loop, not by their arithmetic.

A generator is *mostly* these — every activation, every BatchNorm, every
upsample, every residual add.

**Threading is not the answer here, and was already known not to be.** The
changelog records `nd_add` at **0.83×** on six threads at a million elements —
slower — which is why only the ops with arithmetic to share were threaded in the
first place. Measured again here: 4.25 ms at 1 thread, 4.8 ms at 8.

That also means 7.9 GB/s is *not* a bandwidth ceiling, because more threads
would move it if it were. Something else binds the loop. The leading suspect is
the **output allocation**: every call mints a fresh 16.8 MB buffer, so the cost
includes first-touch page faults on memory that is written once and thrown away.
The existing tier-1 comment reports `out=` not helping — but it measured that at
32k elements, where the buffer fits in cache and there are no faults to pay. At
2.1M that test needs redoing before anything is designed on top of it.

**Design, in the order the evidence supports it:** measure `out=` and a reused
destination at 2.1M first; then fuse the scalar and broadcast cases so a chain
like `(x - mu) / sigma * gamma + beta` does not round-trip four full buffers
through memory; then the `f32` path, which halves the traffic for real. [G1]
showed f32 is worthless for GEMM; for a loop that is actually traffic-bound it
is the whole point. **Not** threading, which has already been measured and
refuted twice.

### G4 — A broadcast operand cost 3.4× the same-shape one

This section started out claiming something else, and the measurement refuted
it. The original text is worth stating because the correction is the finding:

> ~~`BatchNorm2d` costs 112 ms against a ~16 ms memory bound because it
> `permute`s the whole tensor twice per forward pass. This one is a
> package-level fix with no kernel work at all.~~

The permutes are real — 13.8 ms each to materialise `[16,32,64,64]` in a
transposed stride order, four per training step. So BatchNorm2d was rewritten to
reduce over axes `(0,2,3)` and broadcast `[1,C,1,1]` statistics back, with no
permute anywhere. Measured result: **112 ms → 109 ms.** A 2% win on a predicted
7×.

Which meant the diagnosis was wrong, so the next step was to time the pieces:

| op on `[16,32,64,64]` f64, 8 threads | time |
|---|---|
| `nd_add(x, 1.0)` — scalar operand | 4.8 ms |
| `nd_multiply(x, y)` — same shape | 5.3 ms |
| **`nd_subtract(x, [1,C,1,1])` — broadcast** | **18.0 ms** |
| **`nd_multiply(x, [1,C,1,1])` — broadcast** | **16.0 ms** |
| **`[65536,32] - [1,32]` — trailing broadcast** | **17.8 ms** |
| `nd_mean(x, [0,2,3])` | 7.7 ms |

**A broadcast operand does strictly less memory traffic and took 3.4× as long.**
And it was not a stride-pattern accident: broadcasting on the innermost axis and
on the outermost cost the same, so it was the walk itself.

That accounts for BatchNorm2d almost exactly — four broadcast ops at ~18 ms
each, in both the old implementation and the new one. The permutes were a 29 ms
side-show in front of a 72 ms main event.

**Cause.** `applyBinary` in `ndarray_ufunc.hpp` has tier 0 (same shape,
contiguous, vectorized) and tier 1 (one operand a single value, vectorized).
Anything else fell to tier 2, a per-element odometer that calls a lambda and a
dtype-dispatched `load`/`store` for every element. A broadcast *vector* matched
neither fast tier.

This is the tier-1 story one rank up, and the code says so: tier 1 exists
because scalar operands were found to be 4.5× slower than same-shape ones for
*less* traffic. The same reasoning was never extended to a broadcast vector.

**Fix — tier 1b.** After coalescing, the innermost run of a broadcast operation
is always one of three shapes, and each is already a solved problem:

| | innermost strides | per-run kernel |
|---|---|---|
| mode 0 | both operands contiguous (`[N,C]` op `[1,C]`) | `tier0` |
| mode 1 | right operand stride 0 — one value over the run | `tier1` |
| mode 2 | left operand stride 0 | `tier1r` |

So tier 1b changes only the *walk*: it hands each innermost run to the identical
vectorized kernel tier 0 and tier 1 already use, on the same values in the same
order. Results are bit-for-bit what tier 2 produced, which [E15]/N25 require.

| | before | after |
|---|---|---|
| `x - [1,C,1,1]`, 2.1M f64 | 18.0 ms | **4.9 ms** |

**3.7× — and now faster than the same-shape case** (4.9 against 5.9 ms), which
is the right answer: it reads 16.8 MB plus 32 elements instead of 33.6 MB.

**Why this is the item that mattered most.** Broadcasting against a per-channel
vector is not a corner case — it is every normalisation layer, every bias add,
every conditioning and scaling operation in a generator or discriminator. It was
sitting under all of them.

**On the rewritten `BatchNorm2d`:** it is kept, now that broadcast is fast, for
the 29 ms of pure copying it removes — but the honest record is that it was
written to fix something it did not fix, and it earned its place only after the
real cause was found. It keeps the child `BatchNorm1d` registered so that every
serialized parameter name (`bn.gamma`, `bn.running_mean`, …) is unchanged and
checkpoints written before it still load.

**The gate was broken, and the fix is why we know.** Tier 1b's tests passed —
73 green — and then the kernel was deliberately broken five ways, as every
phase of this project requires. **Four of the five mutations passed**, including
one that swapped the operands of `a - b`. The fault was not in the kernel: it
was that `same()`, the helper whose only job is bit-for-bit comparison, wrote

```bantu
ok(nd_all(nd_logical_or($eq, $bothnan, null), null, null), $name);
```

and `nd_all` returns a **0-d ndarray handle**, which is always truthy. Every
`same()` assertion in `numba_fastpath_test.b` and `numba_splat_test.b` — the two
files that exist to prove fast paths agree bit for bit — had been unable to
fail, along with three more sites in the f32 tests.

Repairing it exposed three assertions that had been wrong for as long as they
had existed: they called `nd_random_normal(...)` **twice**, once on each side,
so they compared products of *different* random matrices. With the comparison
real, M1 is caught by exactly the two assertions written for it. Recorded as
numba [N41].

This is the second time in this document that the thing being measured was
fine and the instrument was not. It is worth stating as a working rule: **a new
fast path is not verified until a deliberately broken version of it fails the
suite**, and that check tests the suite as much as the kernel.

**A latent bug found on the way.** `bt.mean($x, [0,2,3], true)` — the reduction
BatchNorm2d, GroupNorm and InstanceNorm2d all need — computed the right forward
value and then raised inside `nd_reshape` on the backward pass. `nd_mean` had
always accepted a list of axes; btorch's `_meanBackward`/`_sumBackward` assumed a
scalar, comparing an axis index against a list (never true) and indexing
`$shape[$axis]` with one. Multi-axis reductions were unusable with autograd.
Fixed with one axis-normalisation helper shared by `sum` and `mean`, which also
made negative axes correct; gradchecked across list, scalar, negative and null
axes.

---

## 4. The package plan

With §2 established, the surface can be designed on correctness and ergonomics
alone. Nothing below needs to be native.

### G5 — A new package, `sauti`, for audio I/O and features

`sauti` (Swahili, *sound/voice*) is to audio what `pillow` is to images: it owns
decoding, encoding and the feature frontend. It depends on numba alone.

| module | contains |
|---|---|
| `sauti/wav.b` | RIFF read/write, 8/16/24/32-bit PCM and float, mono and multichannel |
| `sauti/feature.b` | framing, windows, STFT (matmul basis now, `nd_rfft` when it lands), mel filterbanks, log-mel, MFCC, deltas |
| `sauti/transform.b` | resample, trim, normalise, pre-emphasis, SpecAugment |
| `sauti/griffinlim.b` | spectrogram → waveform, so a generative model is *audible* without a neural vocoder |

**Why not inside `bsci.signal`** ([E4] put spectrograms there): a spectrogram is
a signal-processing operation, but *decoding a WAV file* is not, and a voice
model needs both. Forcing audio I/O into a scipy-class package means a user who
wants to read a file imports an optimiser. `sauti` owns the format; `bsci.signal`
keeps the general filter/convolve/resample theory, and may call into `sauti` or
duplicate twenty lines rather than invert the layering.

**Why not inside `pillow`:** it is an imaging package. Sharing nothing but the
word "codec" is not a reason to share a package.

**Griffin–Lim earns its place as a correctness tool, not a feature.** Without it
the only way to check an audio generator is to look at a spectrogram, which is
exactly how people ship models that score well and sound wrong.

### G6 — Generative pieces go into `btorch`, split by what they are

[E2] (not one mega-package) and the existing btorch layout both apply.

| file | contains | why there |
|---|---|---|
| `btorch/nn.b` (grow) | `ConvTranspose2d`, `Upsample`, grouped/depthwise conv, `InstanceNorm2d`, `GroupNorm`, `PixelShuffle`, `AdaptiveAvgPool2d` | they are `nn.Module`s over existing primitives; torch ships them in core |
| `btorch/gan.b` (new) | the four adversarial objectives — non-saturating BCE, hinge, least-squares, WGAN — plus `gradient_penalty`, R1, spectral norm, EMA of generator weights | a loss is not a layer, and a GAN's correctness lives almost entirely here |
| `btorch/rnn.b` (new) | `RNN`, `GRU`, `LSTM` | already an [ecosystem] commitment; needed for audio sequence models |
| `btorch/attention.b` (new) | `MultiheadAttention`, `TransformerEncoderLayer` | same. `lugha/encoder.b` has a hand-written 4-head attention with its own backward — that is the proof it works, and the reason to write it once properly |

**`btorch/gan.b` exists as its own file because of what breaks in GANs.** The
failure modes are mode collapse, a discriminator that wins, and a generator that
diverges — all of which are properties of the *objective and its gradients*, not
of any layer. Putting the four objectives, the penalties and the EMA in one
tested file is what makes those failures diagnosable instead of mysterious.

**Spectral normalisation needs a power iteration carried across steps**, which
means state on the module and a `train()`/`eval()` distinction that already
exists for Dropout and BatchNorm. It is listed here so that requirement is
designed in, not discovered.

### G7 — The thread default stays 1, but not for the documented reason

`DECISIONS N25` and the comment at `ndarray_native.cpp:236` justify the default
of 1 like this:

> A library whose numeric results change when it is installed on a machine with
> more cores is a support nightmare.

**They do not change.** Checked byte-for-byte via `nd_tobytes` at 1, 3, 7, 8 and
12 threads, on deliberately non-round shapes — a 1,000,003-element `sum` and
`mean`, and a 997×1013×601 `matmul`:

```
threads=1   sum true  mean true  matmul true
threads=3   sum true  mean true  matmul true
threads=7   sum true  mean true  matmul true
threads=8   sum true  mean true  matmul true
threads=12  sum true  mean true  matmul true
```

Which is exactly what N25's own fixed partition promises two paragraphs earlier
— "bit-identical results at any thread count". The decision contradicts its own
design: the stated *reason* for the default is refuted by the stated *property*
of the mechanism.

The default should still be 1, for the reason that survives measurement: a
`sua` web server embedding this interpreter must not silently take 12 cores for
one request. That is an oversubscription argument, not a reproducibility one,
and it says something different about who should opt in.

**So:** correct the rationale in both places, and have the ML packages opt in
explicitly, with `nd_set_threads(1)` documented as the reproduction path it
genuinely is — not as protection against a divergence that does not occur.

---

## 5. Video — the honest answer

**Not on this compute story, and no amount of the above changes that.**

Video generation means a 3-D generator over tens of frames. The 64×64 image GAN
in §1.3 takes 3.2 s per step for 1.3M parameters. A 16-frame 64×64 model is at
least 16× the activations and in practice 30–50× the compute of its image
equivalent, at a parameter count several times larger. Applying every win in
§3 — call it 10× in aggregate, which is optimistic — still leaves a single
training run in the months.

The gap is not kernel quality. It is that this is a CPU, and video generation is
where the arithmetic stops fitting on one.

**What this document commits to instead:**

- Images at 64×64 now, 128×128 after [G2](#g2) and [G3](#g3).
- Audio generation — a GAN vocoder on mel input, or a small diffusion model —
  which is *smaller* than the image case and well inside reach.
- Nothing is designed in a way that forecloses video. Conv3d is a geometry
  change to the same im2col/GEMM path, and `[N,C,T,H,W]` is a shape, not an
  architecture. If a GPU backend ever happens, video becomes a scheduling
  question rather than a redesign.

**Rejected: designing a video API now.** An untested API for a workload that
cannot run is a liability — it would be written against guesses, and the first
real run would break it. The measurement to revisit this is a 128×128 image GAN
completing in under 6 hours. Until that holds, video is out of scope and said so.

---

## 6. Order of work

Ordered by measured payoff per unit of risk, not by what is most interesting.

| # | item | basis | kind | status |
|---|---|---|---|---|
| 1 | **tier 1b: broadcast operands** | [G4]: 18.0 → **4.9 ms**, 3.7× | kernel | **done** |
| 2 | **multi-axis `sum`/`mean` backward** | [G4]: it raised; GroupNorm/InstanceNorm need it | package | **done** |
| 3 | **`BatchNorm2d` without the permutes** | [G4]: 29 ms of copying removed | package | **done** |
| 4 | ML packages opt into threads; correct N25's rationale | [G1]: 5.3× on GEMM | package + docs | **done** — `bt.use_all_cores()`, and N25's reason was wrong ([G7]) |
| 5 | direct blocked convolution | [G2]: 6.2 against 31.6 GFLOP/s, 151 MB of avoidable traffic | kernel | |
| 6 | threaded + fused elementwise loop, f32 path | [G3]: 7.9 GB/s against an order of magnitude more | kernel | |
| 7 | `sauti` — wav, feature, transform, griffinlim | [G5]; the frontend already runs at 98× realtime | package | |
| 8 | `btorch/nn.b` growth, then `btorch/gan.b` | [G6] | package | |
| 9 | `btorch/rnn.b`, `btorch/attention.b` | [G6] | package | |
| 10 | `nd_rfft` / `nd_irfft` | [E4]; *not* a blocker — see §1.1 | kernel | |

Items 5 and 6 are where the remaining order of magnitude is, and both are
bounded, testable kernel work rather than architecture.

**Every item ships with its measurement.** The numbers in this document were
produced by benchmarks in the session that wrote it, several of which refuted
what the plan assumed — that the interpreter was the bottleneck (§2), that f32
would help GEMM ([G1]), that threading would help a convnet ([G2]), and that
thread count perturbs results ([G7]). The next eight items get the same
treatment, and this document gets corrected when they do — [G4] is the first
entry that was, and it was corrected by measurement within hours of being
written.
