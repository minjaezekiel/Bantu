# P7 — what is known before the code is written

Phase P7 is `draw.b` and `font.b`. This file is the research that precedes them, written down
because it was read out of Pillow's C and out of this repository rather than reasoned from the
documentation — the same discipline §6.4 used for libjpeg. Nothing here is implemented yet.

Status: **P0–P6 are done, tested and pushed.** P7 starts here.

---

## 1. Where P7 starts from: prior art already in this repository

`scripts/gen_font_tables.b` is **a TrueType reader written in Bantu** (385 lines): the table
directory, `head`, `maxp`, `hhea`, `hmtx`, `loca`, `cmap` format 4 and `glyf`, with simple and
composite glyphs, contours walked the way fontTools' pen protocol walks them. It exists to generate
bplot's embedded font tables at authoring time.

`font.b` is that reader **promoted to run at load time over an arbitrary font file**, which is what
the architecture (§8.2) already says. What it has to gain: `cmap` formats 0, 6 and 12, `kern`/`GPOS`
pair kerning, `fvar` for `font_variant`, and a glyph cache keyed by (font, size, codepoint).

The native `bp_*` raster builtins (`bp_fill_path`, `bp_stroke_path`, `bp_text`, `bp_line_runs`) are
bplot's **anti-aliased** backend. They are NOT the right engine for `ImageDraw`: Pillow's ImageDraw
does not anti-alias, so matching it means matching its own integer scanline rules, below.

## 2. What `ImageDraw` must match, from `Draw.c`

These are the rules the implementation is held to. Each is a place a reimplementation goes wrong by
default.

**`point` and `hline`.** `point` writes only if `0 <= x < xsize` and `0 <= y < ysize`. `hline` is
**inclusive** of both ends, returns early for an out-of-range `y`, clamps `x0` up to 0 and `x1` down
to `xsize - 1`, and **never swaps** its ends — so `x0 > x1` draws nothing.

**`line`.** Bresenham, and **the endpoint is never drawn**: a line emits exactly `max(|dx|, |dy|)`
pixels, and a zero-length line emits none. `dy == 0` is an `hline` of `[x0, x1-1]` (or `[x1+1, x0]`
reversed). `dx > dy` strictly selects the x-major loop, so **ties go to the y-major** one, and the
error test is `e >= 0`, so a tie steps the minor axis. Error updates: x-major `e = 2dy - dx`, then
`if e >= 0 { minor step; e -= 2dx } e += 2dy`.

**`rectangle`.** Both ends inclusive. **Only `y` is normalised**, so a reversed `x` draws nothing.
Filled ignores `width` and clamps `y1` to `ysize` (not `ysize - 1`). Unfilled defaults `width` 0 to
**1** and expands **inward**: for `i` in `0..width-1`, two `hline`s at `y0+i` and `y1-i`, then two
vertical `line`s at `x1-i` and `x0+i` from `y0+width` to `y1-width+1`.

**`polygon`, filled.** Edges carry `xmin/xmax`, `ymin/ymax`, the original `x0/y0`, and
`dx = (x1-x0)/(y1-y0)` as a float; a run of collinear horizontal edges is merged into the previous
edge. Horizontal edges (`ymin == ymax`) are drawn as `hline`s in a pre-pass (non-RGBA) and are not
in the scan table. For each `y` from `ymin` to `ymax` **inclusive**, each active edge contributes
`(y - y0) * dx + x0` **in float32**; a crossing at an edge's `ymax` (and `y < ymax`) is counted
**twice**; a crossing at an end of an edge with `dx != 0` runs a corner-join fix-up against
*earlier* edges whose rounded crossing matches. The crossings are sorted and drawn in pairs as
`hline(ROUND_UP(xs[p]), y, ROUND_DOWN(xs[p+1]))`, where `ROUND_UP` is half-away-from-zero and
`ROUND_DOWN` half-toward-zero; an odd leftover is ignored. `width` is ignored when filling, and the
rule is even-odd. The RGBA path differs: it carries an `x_pos` cursor so spans cannot overlap and
blend twice.

**`polygon`, outline.** `width == 1` draws each edge with `line` (so each vertex is drawn once, as a
start point); otherwise each edge becomes a **wide line**.

**Wide line.** `s = (width-1)/2`, `L = hypot(dx, dy)`, `rmax = ROUND_UP(s)/L`,
`rmin = ROUND_DOWN(s)/L`, then the four offsets are `ROUND_DOWN(r * d)` and the quad
`(x0-dxmin, y0+dymax), (x1-dxmin, y1+dymax), (x1+dxmax, y1-dymin), (x0+dxmax, y0-dymin)` goes
through the polygon filler. A zero-length wide line is a single `point`.

**The ellipse family.** An integer quarter-tracer on a grid `X in [-a, a]`, `Y in [-b, b]` with step
2 and the parity of `a` and `b`, mapped to pixels by `x0 + (X+a)/2`. The tracer minimises
`|a²y² + b²x² - a²b²|` over three candidates, and **the third candidate does not refresh the running
minimum** — reproduce that as written. `ellipse_init` runs two tracers, an outer and an inner at
`a - 2(w-1)`, and emits rows LIFO. Filling sets `width = a + b`, which makes the inner ellipse
negative and so solid; **an outline of width 0 draws nothing** (unlike rectangle, which promotes 0
to 1). Ring thickness is inward from the bounding box.

`arc`, `chord` and `pieslice` are the same ring intersected with a tree of half-planes
`A·x + B·y + C >= 0` (eps 1e-9), merged with AND/OR by sorted events. Angles are degrees normalised
to `0 <= al < 360`, `al <= ar <= al + 360`; `al == ar` draws nothing and a full sweep drops the tree.
`arc` builds in swapped coordinates when `a < b`. `pieslice`'s outline is two radius bands, a centre
disk of diameter `width`, and the ring-wedge.

## 3. Text cannot be pixel-exact, and the gate has to say so

Every primitive above is Pillow's own C, so pixel equality is achievable and is the gate. **Text is
not.** `ImageFont.truetype` hands the outline to **FreeType**, whose rasteriser anti-aliases and
hints; an independent rasteriser will not reproduce its coverage values. `ImageFont.load_default()`
is also not a fixed target: since Pillow 10.1 it returns an embedded *FreeType* font when a size is
given and the old embedded bitmap font otherwise.

So P7's gate for text is a **measured** one, in the shape PW17 uses for `quantize`: the inked
region's intersection-over-union against Pillow's, the fraction of pixels within a stated tolerance,
and the advance widths and bounding boxes (`getlength`, `getbbox`, `textbbox`) compared exactly,
since those come from `hmtx` and the layout rather than from the rasteriser. The deviation belongs
in `docs/pillow.md` beside the call, not in a footnote.

## 4. The order to build it in

1. `draw.b`'s primitives in the order above (`point`, `hline`, `line`, `rectangle`, `polygon`,
   wide line), each against a Pillow-generated corpus at every width, before any curve.
2. The ellipse family, which is the largest single piece and needs the quarter-tracer first.
3. `floodfill`, `bitmap`, `shape`/`Outline`.
4. `font.b`: promote the reader, add the cmap formats and kerning, then the glyph cache.
5. Text layout — `anchor`, `spacing`, `align`, `direction`, `stroke_width`, `stroke_fill`,
   `embedded_color` — which is arithmetic over the metrics and can be exact even where the pixels
   are not.
