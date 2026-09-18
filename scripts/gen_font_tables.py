#!/usr/bin/env python3
"""Generate the embedded font bplot's raster backend draws text with.

    pip install fonttools
    python3 scripts/gen_font_tables.py /path/to/DejaVuSans.ttf \\
        > bantu-src/compiler/src/raster_font.hpp

NOT PART OF THE BUILD, AND TO BE CONVERTED TO BANTU
---------------------------------------------------
This is authoring-time tooling: it runs once, by hand, and its OUTPUT is
checked in. No build step and nothing at run time touches Python. It is
written in Python only because reading a TrueType file needs a parser Bantu does not have yet, and it is meant to be rewritten in
Bantu -- it needs a Bantu reader for the glyf, loca, cmap and hmtx tables, which became possible once binary file reads landed in B6a.

WHY AN EMBEDDED TABLE AND NOT A FONT FILE
-----------------------------------------
bplot's PNG output must be byte-identical on Linux, macOS and Windows
(docs/bplot-raster-architecture.md §6). System fonts differ by machine, and
hinting differs by renderer, so neither can appear anywhere near a pixel. A
font FILE parsed at run time would be portable but is a parser this tree does
not want to own and the roadmap defers.

So the outlines are converted ONCE, here, and embedded as integers. At run time
the rasteriser walks line and quadratic segments in font units, scales them by a
rational, and fills them through the same coverage code as everything else. No
hinting, no shaping, no kerning -- which is also why the result is the same on
every machine.

WHAT IS INCLUDED
ASCII, Latin-1, and the symbols charts actually use, plus U+FFFD for anything
else. 210 glyphs and 4,105 segments: about 41 KB of integer data.

LICENCE
DejaVu Sans is (c) Bitstream (Bitstream Vera terms) with Arev additions
(c) Tavmjong Bah, both of which permit redistribution provided the notices
travel with the data. This script copies the licence text from beside the font
file into the generated header, so the notice is in the source that carries the
glyphs. The designs are not modified, so the renaming clause does not apply.
"""

import os
import sys

CODEPOINTS = (
    list(range(0x20, 0x7F)) +        # ASCII
    list(range(0xA0, 0x100)) +       # Latin-1 supplement
    [
        0x2013, 0x2014,              # en dash, em dash
        0x2018, 0x2019, 0x201C, 0x201D,  # quotes
        0x2022, 0x2026,              # bullet, ellipsis
        0x2212,                      # minus sign
        0x20AC, 0x2122,              # euro, trade mark
        0x2248, 0x2260, 0x2264, 0x2265, 0x221E,  # approx, not equal, <=, >=, infinity
        0x03BC,                      # micro
        0x2600,                      # sun, for a weather chart
        0xFFFD,                      # the replacement character
    ]
)


def licence_text(ttf_path):
    """The licence from beside the font, so the notice travels with the data."""
    folder = os.path.dirname(os.path.abspath(ttf_path))
    for name in ("LICENSE_DEJAVU", "LICENSE", "LICENCE"):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read().rstrip("\n").splitlines()
    raise SystemExit(f"no licence file found beside {ttf_path}: the notice must ship with the glyphs")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    ttf = sys.argv[1]

    from fontTools.ttLib import TTFont
    from fontTools.pens.recordingPen import DecomposingRecordingPen

    font = TTFont(ttf)
    upm = font["head"].unitsPerEm
    cmap = font.getBestCmap()
    glyphs = font.getGlyphSet()
    hmtx = font["hmtx"]

    missing = [hex(c) for c in CODEPOINTS if c not in cmap]
    if missing:
        raise SystemExit(f"{os.path.basename(ttf)} has no glyph for: {', '.join(missing)}")

    # Each glyph becomes a run of commands in one shared point array:
    #   0 = move, 1 = line, 2 = quadratic (one control point, one end point)
    points, entries = [], []
    for cp in CODEPOINTS:
        name = cmap[cp]
        # Decomposing: accented letters (o-dieresis, c-cedilla, ...) are
        # COMPOSITES in TrueType, a base glyph plus an accent. A plain recording
        # pen hands back addComponent, which this loop would drop -- a blank.
        pen = DecomposingRecordingPen(glyphs)
        glyphs[name].draw(pen)
        start = len(points)
        ops = 0
        for op, args in pen.value:
            if op == "moveTo":
                points.append((0,) + tuple(round(v) for v in args[0]))
                ops += 1
            elif op == "lineTo":
                points.append((1,) + tuple(round(v) for v in args[0]))
                ops += 1
            elif op == "qCurveTo":
                # TrueType curves are quadratic; fontTools may hand back an
                # implied-on-curve run, so expand it into single segments.
                pts = [tuple(round(v) for v in p) for p in args if p is not None]
                if args[-1] is None:
                    # A contour with no on-curve point at all: every point is a
                    # control, and the on-curve points are the midpoints between
                    # them, cyclically. It needs its own move, to the first one.
                    mid = lambda a, b: ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
                    start_pt = mid(pts[-1], pts[0])
                    points.append((0,) + start_pt)
                    ops += 1
                    for i, ctrl in enumerate(pts):
                        end = mid(ctrl, pts[(i + 1) % len(pts)])
                        points.append((2, ctrl[0], ctrl[1], end[0], end[1]))
                        ops += 1
                    continue
                for i in range(len(pts) - 1):
                    ctrl = pts[i]
                    nxt = pts[i + 1]
                    end = nxt if i + 2 == len(pts) else (
                        (ctrl[0] + nxt[0]) // 2, (ctrl[1] + nxt[1]) // 2)
                    points.append((2, ctrl[0], ctrl[1], end[0], end[1]))
                    ops += 1
            elif op in ("closePath", "endPath"):
                continue
            elif op == "curveTo":
                raise SystemExit("cubic outlines are not supported; this expects a TrueType font")
            else:
                raise SystemExit(f"unhandled pen operation {op!r} in {name}")
        entries.append((cp, start, ops, hmtx[name][0]))

    out = sys.stdout
    out.write("#pragma once\n")
    out.write("// Generated by scripts/gen_font_tables.py -- do not edit by hand.\n")
    out.write(f"// Source: {os.path.basename(ttf)}, {len(entries)} glyphs, {len(points)} segments.\n")
    out.write("//\n")
    out.write("// The glyph outlines below are DejaVu Sans, redistributed under the terms\n")
    out.write("// reproduced here in full. The designs are unmodified.\n//\n")
    for line in licence_text(ttf):
        out.write(("// " + line).rstrip() + "\n")
    out.write("\n#include <cstdint>\n\nnamespace bplot_raster {\n\n")
    out.write(f"static const int32_t kFontUnitsPerEm = {upm};\n\n")
    out.write("// {kind, x1, y1, x2, y2} -- kind 0 move, 1 line, 2 quadratic (control, end).\n")
    out.write(f"static const int16_t kGlyphSegments[{len(points)}][5] = {{\n")
    for p in points:
        padded = tuple(p) + (0,) * (5 - len(p))
        out.write("    {" + ", ".join(str(v) for v in padded) + "},\n")
    out.write("};\n\n")
    out.write("// {codepoint, first segment, segment count, advance width}\n")
    out.write(f"static const int32_t kGlyphIndex[{len(entries)}][4] = {{\n")
    for cp, start, ops, adv in entries:
        out.write(f"    {{{cp}, {start}, {ops}, {adv}}},\n")
    out.write("};\n\n} // namespace bplot_raster\n")


if __name__ == "__main__":
    main()
