#!/usr/bin/env python3
"""
convert_hf.py — write a lugha `.btw` model file from a PyTorch state dict.

The point of this script is that the weights in this repository are a baseline,
not a product. lugha's inference is a fixed architecture reading a documented
file format, so a stronger classifier is a file swap rather than a rewrite — and
this is the file that performs the swap.

    python3 tools/convert_hf.py --help

It deliberately does NOT download anything. Point it at a local checkpoint you
have already obtained and are licensed to use:

    python3 tools/convert_hf.py \
        --state-dict ./my-toxicity-model/pytorch_model.bin \
        --vocab      ./my-toxicity-model/vocab.txt \
        --out        samples/one-taylor/backend/model/toxicity.btw \
        --layers 2 --heads 4 --d-model 128 --d-ff 512 --max-len 64 \
        --map ./tools/name_map.example.json

Afterwards, recalibrate the thresholds on YOUR held-out data — the reliability
tensor decides which labels are allowed to block a post, and carrying over
another model's operating points would be meaningless:

    cp ./my-toxicity-model/vocab.txt lugha/vocab.txt
    bantu run lugha/calibrate.b

Format (little-endian; see docs/lugha-architecture.md section 4):

    0   4            magic "BTW1"
    4   4   u32      format version (1)
    8   4   u32      tensor count N
    12  4   u32      flags (0)
    16  ...          N records: u16 name length, name, u8 dtype(0=f64),
                     u8 ndim, u32 * ndim shape
    then             the payloads, in record order, 8 bytes per f64

Requires numpy. torch only if the checkpoint is a .bin/.pt rather than a
safetensors file or an .npz.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("convert_hf.py needs numpy: pip install numpy")

MAGIC = b"BTW1"
LABELS = ["harassment", "sexual", "sexual_minor", "grooming",
          "threat", "self_harm", "slur"]


def param_names(layers: int) -> list[str]:
    """The tensor names lugha/encoder.b reads. This list IS the contract."""
    names = ["meta", "tok_embed", "pos_embed"]
    for i in range(layers):
        p = f"blk{i}."
        names += [p + "ln1.g", p + "ln1.b",
                  p + "wq", p + "wk", p + "wv", p + "wo",
                  p + "ln2.g", p + "ln2.b",
                  p + "ff1", p + "ff1b", p + "ff2", p + "ff2b"]
    names += ["ln_f.g", "ln_f.b", "head.w", "head.b"]
    return names


def expected_shapes(layers: int, d_model: int, d_ff: int, max_len: int,
                    vocab: int, n_labels: int) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "meta": (7,),
        "tok_embed": (vocab, d_model),
        "pos_embed": (max_len, d_model),
        "ln_f.g": (d_model,), "ln_f.b": (d_model,),
        "head.w": (d_model, n_labels), "head.b": (n_labels,),
    }
    for i in range(layers):
        p = f"blk{i}."
        shapes.update({
            p + "ln1.g": (d_model,), p + "ln1.b": (d_model,),
            p + "wq": (d_model, d_model), p + "wk": (d_model, d_model),
            p + "wv": (d_model, d_model), p + "wo": (d_model, d_model),
            p + "ln2.g": (d_model,), p + "ln2.b": (d_model,),
            p + "ff1": (d_model, d_ff), p + "ff1b": (d_ff,),
            p + "ff2": (d_ff, d_model), p + "ff2b": (d_model,),
        })
    return shapes


def load_state_dict(path: Path) -> dict[str, "np.ndarray"]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path) as z:
            return {k: np.asarray(z[k]) for k in z.files}
    if suffix == ".safetensors":
        try:
            from safetensors.numpy import load_file
        except ImportError:
            sys.exit("a .safetensors checkpoint needs: pip install safetensors")
        return dict(load_file(str(path)))
    try:
        import torch
    except ImportError:
        sys.exit(f"reading {path.name} needs: pip install torch")
    obj = torch.load(str(path), map_location="cpu", weights_only=True)
    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()
    return {k: v.detach().to(torch.float64).numpy() for k, v in obj.items()}


def write_btw(out: Path, names: list[str],
              tensors: dict[str, "np.ndarray"]) -> int:
    header = bytearray(MAGIC)
    header += struct.pack("<III", 1, len(names), 0)
    for name in names:
        raw = name.encode("ascii")
        arr = tensors[name]
        header += struct.pack("<H", len(raw)) + raw
        header += struct.pack("<BB", 0, arr.ndim)
        for dim in arr.shape:
            header += struct.pack("<I", int(dim))

    with out.open("wb") as fh:
        fh.write(header)
        for name in names:
            # C order and float64: nd_fromfile reads a flat, contiguous,
            # native-endian buffer and nothing else.
            fh.write(np.ascontiguousarray(tensors[name], dtype="<f8").tobytes())
    return out.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Write a lugha .btw model file from a PyTorch state dict.")
    ap.add_argument("--state-dict", required=True, type=Path,
                    help=".bin / .pt / .safetensors / .npz checkpoint")
    ap.add_argument("--vocab", required=True, type=Path,
                    help="vocab.txt, one token per line; '##' marks a continuation")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--map", type=Path,
                    help="JSON mapping lugha tensor names -> checkpoint keys")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--d-ff", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--labels", type=int, default=len(LABELS))
    ap.add_argument("--print-names", action="store_true",
                    help="print the tensor names and shapes lugha needs, then exit")
    args = ap.parse_args()

    vocab_size = sum(
        1 for line in args.vocab.read_text(encoding="utf-8").splitlines() if line
    ) if args.vocab.exists() else 0

    shapes = expected_shapes(args.layers, args.d_model, args.d_ff,
                             args.max_len, vocab_size, args.labels)
    names = param_names(args.layers)

    if args.print_names:
        for n in names:
            print(f"{n:20s} {shapes[n]}")
        return 0

    if vocab_size == 0:
        sys.exit(f"{args.vocab} is missing or empty")

    state = load_state_dict(args.state_dict)
    mapping = json.loads(args.map.read_text()) if args.map else {}

    tensors: dict[str, np.ndarray] = {}
    missing: list[str] = []
    wrong: list[str] = []

    for name in names:
        if name == "meta":
            tensors["meta"] = np.array(
                [args.layers, args.heads, args.d_model, args.d_ff,
                 args.max_len, vocab_size, args.labels], dtype="f8")
            continue
        key = mapping.get(name, name)
        if key not in state:
            missing.append(f"{name} (looked for '{key}')")
            continue
        arr = np.asarray(state[key], dtype="f8")
        want = shapes[name]
        if arr.shape != want:
            # A linear layer stored transposed is the single most common
            # difference between frameworks, so it is tried rather than
            # reported as an error.
            if arr.T.shape == want:
                arr = np.ascontiguousarray(arr.T)
            else:
                wrong.append(f"{name}: got {arr.shape}, need {want}")
                continue
        tensors[name] = arr

    if missing or wrong:
        print("Cannot write the model file.\n", file=sys.stderr)
        if missing:
            print("Missing tensors:", file=sys.stderr)
            for m in missing:
                print(f"  {m}", file=sys.stderr)
        if wrong:
            print("Wrong shapes:", file=sys.stderr)
            for w in wrong:
                print(f"  {w}", file=sys.stderr)
        print("\nCheckpoint keys available:", file=sys.stderr)
        for k in sorted(state)[:40]:
            print(f"  {k} {tuple(np.shape(state[k]))}", file=sys.stderr)
        print("\nWrite a --map JSON to connect them, or run --print-names to "
              "see what lugha expects.", file=sys.stderr)
        return 1

    size = write_btw(args.out, names, tensors)
    total = sum(int(np.prod(t.shape)) for n, t in tensors.items() if n != "meta")
    print(f"wrote {args.out} — {size} bytes, {total} parameters")
    print()
    print("The thresholds and the reliability tensor are NOT in this file yet.")
    print("lugha refuses to load a model without them, on purpose: carrying over")
    print("another model's operating points would decide which labels may block")
    print("a post based on measurements of a different model. Run:")
    print()
    print(f"    cp {args.vocab} lugha/vocab.txt")
    print("    bantu run lugha/calibrate.b")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
