# bplot's raster backend — PNG, byte for byte, on every platform

**Status:** design, written before any code (phase B6).
**Scope:** turning a bplot figure into a PNG natively; the binary-file fix the PNG needs; and the
constraints that make **byte-identical output on Linux, macOS and Windows** a property of the design
rather than a hope.
**Related:** [`docs/bplot-architecture.md`](bplot-architecture.md) §12, which this expands;
[`bplot-suite/ROADMAP.md`](../bplot-suite/ROADMAP.md) Phase B6.

---

## 1. Why a raster backend, and why it is native

SVG is the right format for most charts, and it has two real limits:

- **Size at scale.** A million-point scatter is a 61 MB document (measured in B4). A PNG of the same
  chart is a few hundred kilobytes, because its size is set by pixels, not marks.
- **It executes.** B1's escaping makes injection through bplot impossible, but a chart served from a
  site's own origin is still active content (architecture §8.1). A PNG is inert. For a chart built
  from user-supplied data it is the answer that needs no further argument.

Rasterising is proportional to pixels, not elements. A 900×500 figure is 450,000 pixels, roughly a
dozen operations each for coverage and compositing; at ~0.38 µs per interpreted operation that is
**about two seconds per chart in Bantu** before any text or compression. The rasteriser, the text
renderer and the encoder are therefore native. Everything that decides *what* to draw stays in
`bplot.b`, unchanged: the raster backend is a second implementation of the same thirteen-method
backend the SVG writer already is.

## 2. The contract, and what it rules out

**The same figure produces the same bytes on every platform.** That is B6's gate, and it cannot pass
by accident: a single differing bit anywhere in the pipeline changes the deflate stream and the file.

Byte identity rules out, specifically:

| hazard | where it would bite | how the design removes it |
|---|---|---|
| **fused multiply-add (FMA)** | `x * a + b` may be contracted into one instruction on arm64 (Apple clang contracts within an expression; GCC contracts across statements), rounding differently from two separate operations | pixel-deciding arithmetic is **integer**; the only floating-point step is one quantisation per coordinate (§3) |
| **libm transcendental functions** | `sin`, `cos`, `atan2`, `pow` are not correctly rounded and differ in the last bit between glibc, Apple's libm and the MSVC CRT | **no libm call touches geometry**: circles and arcs use integer tables generated once and embedded (§5); text rotation uses integer CORDIC (§6) |
| **the system zlib** | Linux distributions ship different zlib versions, and zlib does not promise identical compressed output across versions | **a deflate implementation of our own**, whose output is a pure function of its input (§7) |
| **system fonts** | different fonts, versions and hinting on every machine | **one embedded font**, rendered by our own code with no hinting (§6) |
| **text-mode file writes** | Windows turns every `\n` byte of a PNG into `\r\n` | **binary-safe file I/O** (§8) |
| **unordered iteration, hash seeds, time, locale** | anything that varies run to run | none of these reach the output: tables are arrays, the PNG carries no timestamp, number parsing is the "C" locale |

## 3. Coordinates: the PNG draws the SVG's numbers

Every coordinate reaching the raster backend is first **quantised to hundredths of a pixel, with
exactly the arithmetic `_px` uses for the SVG** — `round(|v| × 100)` — and from then on held as an
integer.

Two consequences, both deliberate:

- **The PNG is the SVG's geometry to a hundredth of a pixel.** They cannot disagree about where
  something is, because they start from the same integer.
- **The residual cross-platform risk is exactly the SVG's, and no larger.** Coordinates computed in
  Bantu with `cos`, `log10` or `exp` — a pie slice, a log axis, a violin — can differ in the last bit
  between platforms' libm. That only changes the output if the value lands within one unit in the
  last place of a boundary between two hundredths: roughly one chance in 10¹³ per coordinate. The SVG
  backend has carried the same exposure since B1. Removing it entirely would mean shipping a
  correctly-rounded libm for Bantu's maths builtins — worth doing for reproducible numerics generally,
  and out of proportion as a condition of PNG output. It is recorded here, not hidden.

The device transform is `pixel = centipixel × dpi / 9600`, evaluated as a **rational in 64-bit
integers** and carried at **8 bits of subpixel precision (Q24.8)**. `dpi` 96 is one device pixel per
SVG user unit, so a figure's pixel size is `width × dpi / 96` — and the roadmap's stress case,
"4000×3000 at 300 dpi", is exactly a 1280×960 figure.

## 4. The rasteriser

**Signed-area coverage accumulation**, the technique of `font-rs` and `stb_truetype`'s newer path,
in integers:

1. Every shape is flattened to line segments in Q24.8.
2. Each segment adds its signed area to an accumulation buffer, cell by cell, with exact integer
   arithmetic at 8-bit subpixel resolution.
3. A running sum across each row gives each pixel's coverage.
4. **Nonzero** winding, the SVG default and the only rule bplot relies on; even-odd is available for
   completeness.

**One mask per shape, composited once.** A stroked polyline is its segments, its joins and its caps
— pieces that overlap. Rasterising them into a single coverage mask and compositing that mask once is
what stops a semi-transparent line from darkening at every joint, which is the classic artefact of
drawing strokes as independent pieces.

**Clipping** is an integer rectangle intersected before rasterising, which is all bplot uses (one clip
per axes).

**Compositing** is source-over in 8-bit sRGB, with exact integer rounding:
`out = (src × a + dst × (255 − a) + 127) / 255`. No linear-light blending: browsers composite SVG in
sRGB, and the PNG should look like the SVG, not like a physically-better image nobody asked for.

## 5. Strokes, circles and arcs — without a single trigonometric call

- **Segment normals** use an **integer square root** of a 64-bit sum of squares. The result is exact
  by definition, where a double `sqrt` is correctly rounded too but would force the offsets back into
  floating point.
- **Round joins and caps** (bplot's polylines ask for them) and **circles** are polygons whose vertices
  come from **embedded unit-circle tables**: cos and sin at `k × 2π / N` for N ∈ {16, 32, 64, 128,
  256}, as Q30 integers. The tables are generated once, by a script checked into the repository, and
  are data, not computation. The polygon's resolution is chosen from the radius so the chord error
  stays under a sixteenth of a pixel.
- **Butt caps** for `line`, which is SVG's default for that element.
- **Dashes** (`"6,4"`) walk the path by integer arc length.
- **Elliptical-arc path commands** (`A`/`a`) — bplot emits only circular ones, with no rotation — find
  the centre with the integer square root, then step around the embedded table from the start vector
  until a cross-product sign test says the end vector has been passed. No `atan2`.
- **Zero-width strokes draw nothing**, as in SVG. A degenerate polygon (fewer than three distinct
  points, or zero area) draws nothing and raises nothing.

## 6. Text

**The font is DejaVu Sans, embedded.** 211 glyphs — ASCII, Latin-1, and the symbols charts actually
use (`° µ × − – — … ± ≤ ≥ ≈ ≠ ∞ € ™ ☀`), plus `U+FFFD` for anything else — as 3,493 outline points,
about **13 KB** of integer data. A generator script, run at authoring time with fontTools, writes the
table into a header; nothing parses a font at run time, which is the font-file parsing the roadmap
defers. DejaVu's licence (Bitstream Vera and Arev terms) permits redistribution with its notices, and
the header carries them in full.

**No hinting, no system font, no FreeType.** Glyph outlines are quadratic curves, flattened by fixed
integer subdivision and drawn through the same rasteriser as everything else. Hinting is what makes
text rendering differ between platforms, and it is the first thing a byte-identical renderer gives up.

**Rotation**: multiples of 90° are exact integer transforms. Any other angle uses a rotation computed
by **integer CORDIC** from the degree value — deterministic on every platform, where `cos(θ)` from
libm is not.

**Layout measures the font that will be drawn.** DejaVu Sans is wider than the Helvetica table the SVG
path measures with — **13.5% on a typical label**, and 636 against 556 units for every digit. A PNG
laid out with Helvetica's widths would push labels past the gutters `tight_layout` computed. So text
measurement follows the backend: a PNG render measures with DejaVu's advances, and **the SVG path is
unchanged**, so no existing document moves. The same figure's SVG and PNG may therefore place a
gutter a few pixels apart. Both are correct for the font each draws with — and the SVG's font is
whatever the viewer's browser picks for `sans-serif`, which no metrics table can know.

## 7. The PNG encoder

- **Colour type 2 (RGB), 8 bits**: every bplot background is opaque, so an alpha channel would only
  add a quarter to the raw size.
- **Filtering**: each row picks the filter (None, Sub, Up, Average, Paeth) with the smallest sum of
  absolute residuals — the heuristic the PNG specification recommends — with **ties broken by the
  lowest filter number**, so the choice is a pure function of the pixels.
- **Deflate, written here**: LZ77 over a 32 KiB window with hash chains of a fixed maximum length and
  one-step lazy matching, then **dynamic Huffman codes** whose lengths are limited to 15 bits by
  package-merge, with every tie broken by symbol value. Stored blocks — what the PWA icon writer uses —
  would make a 900×500 chart about 1.4 MB. Fixed Huffman codes alone would be simpler, but measurably
  larger on exactly the flat-colour images charts are.
- **CRC-32 and Adler-32** as in the existing icon writer.
- **A `pHYs` chunk** records the dpi (`round(dpi / 0.0254)` pixels per metre), so a 300 dpi PNG prints
  at its intended size.
- **No `tIME` chunk and no text chunks.** A timestamp would make every render different.

An encoder we wrote must be checked by one we did not: every test PNG is decoded by **Pillow** and by
Python's **zlib**, and the decoded pixels must equal the canvas the encoder was handed.

## 8. Binary-safe file I/O

`open()` accepts `"r"`, `"w"` and `"a"` and sends **anything else to read mode** — so
`open($path, "wb")` silently opens the file for reading and every write fails. And nothing in the file
builtins ever sets `std::ios::binary`, so on Windows a PNG written through `writefile()` has every
`\n` byte turned into `\r\n`.

- `open()` accepts `"rb"`, `"wb"` and `"ab"`, which set `std::ios::binary`, and **an unknown mode
  raises**, naming the modes that exist.
- `readfile($path, $mode)`, `writefile($path, $data, $mode)` and `appendfile($path, $data, $mode)`
  take an optional mode of the same kind. **The default stays text**, so no existing program changes
  behaviour on any platform.
- `savefig("chart.png")` does not route megabytes through a Bantu string at all: the canvas writes its
  own file, in binary, natively. `$fig.to_png()` returns the bytes for a caller that wants them — a sua
  handler, for which `$res.send()` was verified binary-safe (NUL, 0xFF, CR, LF and 0x1A all arrive
  intact, with a correct `Content-Length`).

## 9. The Bantu-facing surface

```bantu
plt.plot([1, 2, 3], [2, 4, 9]);
plt.savefig("chart.png");                     // extension chooses the backend
plt.savefig("print.png", {"dpi": 300});       // 3.125x the pixels, and a pHYs chunk saying so

$png = $fig.to_png({"dpi": 144});             // bytes, for serving
```

Underneath, `BPlotRaster` in `bplot.b` implements the backend method set by calling `bp_canvas_*`
builtins on a native canvas handle — refcounted like every other handle, so a canvas is freed when the
last reference drops. The builtins stay documented and callable, and bplot is what uses them.

## 10. Security

Every `bp_canvas_*` builtin is reachable from any Bantu program, so each one validates its input as
if it came from a stranger — because inside a sua handler, it may have:

- **Canvas size**: width and height at most 32,767 and at most 2²⁸ pixels (about 1 GB of RGB), with
  **checked multiplication**. `bp_canvas_new(100000, 100000)` raises; it does not allocate.
- **Coordinates**: non-finite values raise. Finite values far outside the canvas are clipped in
  integer space before rasterising, and Q24.8 covers ±8 million pixels, so no coordinate bplot can send
  overflows.
- **Path data**: the parser accepts exactly `M m L l H h V v A a Z z`, rejects anything else by name,
  has no recursion, and caps the flattened point count.
- **Text**: length is capped, and invalid UTF-8 renders as `U+FFFD` rather than being trusted.
- **The sanitiser tier**: ASan and UBSan over the raster suite, plus a fuzz of the path parser and the
  rasteriser with degenerate polygons, zero-width strokes, out-of-canvas and huge coordinates, and
  hostile path strings.

## 11. Rejected alternatives

- **libpng, zlib, stb_image_write.** A new dependency against the house rule that produced a
  from-scratch P-256 — and, decisively for this gate, **zlib's compressed output is not promised
  identical across versions**, and Linux distributions ship different ones. A byte-identical PNG gate
  built on the system zlib would fail on the first distribution upgrade.
- **Cairo, Skia, FreeType.** Large dependencies, floating-point rasterisers, and hinting — each of
  which defeats byte identity.
- **A floating-point rasteriser.** Simpler to write; not reproducible across compilers (§2).
- **Rasterising in Bantu.** About two seconds per chart before text and compression (§1).
- **Stored deflate.** Correct and simple; a 900×500 chart would be about 1.4 MB.
- **A stroke (Hershey) font.** Public domain and tiny, and it looks like a plotter from 1970.
- **Switching the SVG path to DejaVu metrics too**, so both backends lay out identically. It would move
  every existing document's gutters to match a font most browsers will not use. Rejected for B6; it can
  be revisited as a decision of its own.

## 12. How B6 is built, and what each step must pass

| step | lands | gate |
|---|---|---|
| **B6a** | binary-safe `open`/`readfile`/`writefile`/`appendfile`; unknown modes raise | every byte 0–255, NUL runs and CR/LF round-trip in binary mode; text mode unchanged; `"x"` raises |
| **B6b** | the canvas handle, rectangle fills, the PNG encoder with filtering and deflate | Pillow and zlib decode every PNG to the exact canvas bytes; deflate beats stored blocks by ≥ 10× on chart-like images; size caps raise |
| **B6c** | the rasteriser: polygons, strokes with joins, caps and dashes, circles, the path parser, clipping | coverage of known shapes within ±1/255 of analytic area; semi-transparent strokes do not darken at joints; degenerate input draws nothing and raises nothing |
| **B6d** | text: the font-table generator, glyph rendering, rotation, measurement | every embedded glyph renders; anchors and rotation land where the metrics say; invalid UTF-8 is `U+FFFD` |
| **B6e** | `BPlotRaster`, `savefig(".png", {dpi})`, `to_png()`, PNG from sua, docs | every chart kind renders to a PNG that decodes; `samples/bplot/` gains PNG output; the sua sample can serve PNG |
| **B6f** | the byte-identity corpus in all three CI jobs; stress; sanitisers; records | **SHA-256 of a fixed corpus of PNGs, committed as literals, matched on Linux, macOS and Windows**; 4000×3000 at 300 dpi within time and memory gates; ASan + UBSan clean over the suite and the fuzz |

## 13. The six questions

**Scalable?** Cost is proportional to pixels, not to data: B4 already reduces a million points to the
canvas before any backend sees them. A 12-megapixel render is bounded by its buffers — about 36 MB
of RGB plus one row of coverage — and the encoder is linear in its input.

**Maintainable, long-term?** No new dependency. Three self-contained parts — rasteriser, font, encoder
— each testable against an external oracle (analytic areas, fontTools' outlines, Pillow and zlib), and
every table generated by a script in the repository rather than pasted by hand.

**Easy, and the Bantu way?** One changed word: `savefig("chart.png")`. The backend interface bplot
already has is the seam; no artist changes.

**Documentable and testable?** Unusually so: the gate is a hash. A platform that produces a different
file fails a literal comparison, with nothing to interpret.

**Efficient?** Integer rasterisation is not slower than floating point on this workload, and
deflate on flat-colour chart images compresses well. The numbers are measured in B6b–B6f, not promised
here.

**Secure?** A PNG is the format that removes SVG's executable-content risk for user-influenced charts,
and the native surface it adds is size-capped, validated and sanitiser-tested (§10).

## 14. The tooling, and the Python in this repository

Two generators live in `scripts/`, and they are the only Python this work adds:

| script | produces | why Python today |
|---|---|---|
| `gen_circle_tables.py` | `raster_tables.hpp` — cos and sin at fixed angles, as Q30 integers | nothing more than `cos`, `sin` and a file write; **a direct translation to Bantu** |
| `gen_font_tables.py` | `raster_font.hpp` — the embedded glyph outlines | reading a TrueType file needs a parser Bantu does not have yet |

Both are **authoring-time tools**: they are run once, by hand, and their *output* is
checked in. No build step runs them, and nothing at run time touches Python — the interpreter has no
dependency on it, and neither does a user's program.

**They are meant to become Bantu.** The circle tables need only what the language already has. The
font tables need a Bantu reader for a TrueType file's `glyf`, `loca`, `cmap` and `hmtx` tables —
which became possible only in B6a, when binary file reads started working. Until then the generated
headers are the contract, and they are readable and reproducible either way.

**One piece of Python is deliberately staying.** `tests/bplot_png_test.sh` decodes our PNGs with
Pillow and our deflate streams with Python's `zlib`, and `tests/run_doc_examples.sh` uses Python only
to check XML. The whole value of those gates is that the decoder is one **we did not write**.
Reimplementing them in Bantu would turn an independent check into a mirror of the thing it checks.

---

## 15. What this does not promise

- **That it looks identical to the SVG.** Same geometry to a hundredth of a pixel; different font
  rendering, since a browser rasterises the SVG with its own fonts and anti-aliasing.
- **Protection from libm differences upstream of quantisation** (§3) — the same exposure the SVG has
  always had.
- **Colour management.** Plain sRGB with no embedded profile, as browsers assume for SVG.
- **Kerning and complex scripts.** Advance widths only; right-to-left and shaped scripts render as
  individual glyphs or as `U+FFFD`.
- **Observed byte identity before CI runs.** Linux and Windows cannot be checked on this machine — the
  Docker daemon is installed but not running — so the three-platform gate is proven by CI, not by the
  author.
