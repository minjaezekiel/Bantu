# PngSuite — the canonical PNG conformance corpus

176 files by Willem van Schaik (<willem@schaik.com>), from
<http://www.schaik.com/pngsuite/>, the suite every PNG decoder is checked
against. Released by its author for exactly this purpose: "Permission to use,
copy, modify and distribute these images for any purpose and without fee is
hereby granted."

Each `.png` is paired with a `.exp` holding the pixel bytes Python's Pillow
decodes it to, and `MANIFEST` records, per file, whether Pillow reads it and
with what mode and size. `tests/pillow_codec_test.b` asserts that
`pillow` agrees with Pillow on every one.

The files are deliberately exhaustive and deliberately hostile:

| prefix | what it exercises |
|---|---|
| `basn` / `basi` | every legal colour type and bit depth, non-interlaced and interlaced |
| `bgai` / `bggn` | background chunks, with and without interlacing |
| `cs*` / `cd*` | odd bit depths and colour-map sizes |
| `f00n` / `f01n` … | each of the five scanline filters, alone and mixed |
| `g03n` / `g04n` … | gamma values |
| `oi1n` / `oi2n` … | IDAT split across one, two, four and many chunks |
| `s01n` … `s40n` | every width from 1 to 40 pixels, where row padding bites |
| `tbbn` / `tp0n` … | transparency, including palettes with tRNS |
| `z00n` … `z09n` | every zlib compression level |
| **`x*`** | **intentionally corrupt**: bad signature, bad IHDR CRC, bad IDAT CRC, illegal colour type, zero dimensions. A decoder must REFUSE these, and the test asserts it does |

The `x*` files are the valuable ones. A decoder that reads them is not
lenient, it is wrong: a bad CRC means the data cannot be trusted, and
accepting it silently turns a corrupt file into a plausible image.
