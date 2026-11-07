// ════════════════════════════════════════════════════════════════════════
//  pillow/font.b — ImageFont: a bitmap font with no file, and a TrueType
//  reader written in Bantu.
//
//      include "./pillow.b" as pillow;
//      $f = pillow.load_default(null);                  // no file, no deps
//      $f = pillow.truetype("DejaVuSans.ttf", 24, null, null, null);
//      pillow.font_getlength($f, "Hello", null);
//      pillow.font_getbbox($f, "Hello", null, null, 0);
//
//  TWO KINDS OF FONT, AND TWO DIFFERENT GATES. That distinction is the
//  whole shape of this file.
//
//  A **PILfont** is a glyph sheet plus a 256-entry metric table, and
//  rendering it is byte copying: no scaling, no hinting, no rasteriser.
//  Pillow's own renderer (`_font_getmask` in `_imaging.c`) is forty lines,
//  and this reproduces them, so text in the default font is **byte-identical
//  to Pillow's**. The font this ships is the one Pillow embeds in
//  `ImageFont.load_default_imagefont()` — a 798x20 "1"-mode sheet of
//  courB08 — carried here as base64 so that text works with no file and no
//  dependency, which is what the architecture (§8.2) promises.
//
//  A **TrueType font** cannot be byte-identical, and the reason is
//  structural rather than fixable. `ImageFont.truetype` hands the outline to
//  **FreeType**, whose rasteriser anti-aliases and, worse for a
//  reimplementation, **grid-fits**: the TrueType interpreter runs the font's
//  own hinting instructions and moves points onto the pixel grid. So:
//
//    * **advances are exact.** Pillow's default layout engine is libraqm,
//      which asks HarfBuzz for advances, and HarfBuzz loads glyphs
//      *unhinted* — so an advance is `FT_MulFix(hmtx, x_scale)` in 26.6 and
//      nothing else. Measured against Pillow 12.3.0 with DejaVu Sans at 8,
//      12, 24 and 72 px: identical, to the last 1/64 of a pixel.
//    * **bounding boxes are not.** `FT_Glyph_Get_CBox` is taken from the
//      *hinted* outline, so at 24 px Pillow's "Hello, world!" is 21 pixels
//      tall where the unhinted control box is 22: vertical hinting pulls
//      `l` and `d` down onto the cap-height grid. The deviation is bounded
//      and measured rather than hidden — see the gate in
//      tests/pillow_font_test.b and the numbers in docs/pillow.md.
//    * **pixels are not**, for the same reason plus the rasteriser's own
//      coverage arithmetic. The gate is the one PW17 uses for `quantize`:
//      the inked region's intersection-over-union against Pillow's, and the
//      fraction of pixels within a stated tolerance.
//
//  WHAT THE READER COVERS. `head maxp hhea hmtx loca glyf cmap kern GPOS
//  fvar OS/2 name`: `cmap` formats 0, 4, 6 and 12; simple and composite
//  glyphs, including scaled and point-matched components; pair kerning from
//  GPOS lookup type 2 (both formats) with the legacy `kern` table as the
//  fallback. Anything outside that raises, naming the table and the byte
//  offset, rather than guessing — rule 5 of the roadmap's five.
//
//  THE PRIOR ART. `scripts/gen_font_tables.b` has been a TrueType reader in
//  Bantu since bplot: the table directory, `head`, `maxp`, `hhea`, `hmtx`,
//  `loca`, `cmap` format 4 and `glyf` with composites, walked the way
//  fontTools' pen protocol walks them. This is that reader promoted to run
//  at load time over an arbitrary file, with the formats it did not need
//  added and with the restrictions it could afford to raise on removed.
//
//  ─── BANTU, AND WHY THE BYTE READS ARE INLINE ──────────────────────────
//
//  A Bantu list is copied when it is handed to a function and when it is
//  read out of a dict (PW14). A font is several hundred kilobytes, so the
//  `u8($i)` helper that `gen_font_tables.b` can afford — its font is a
//  module-level variable, read and never passed — is not available to a
//  module that holds several fonts at once. Assigning a bare `$var` inside
//  a function binds a function-local, so there is no module-level window
//  either. So the font's bytes live in a numba array (passed by reference,
//  like every other buffer in this package) and each parse step slices out
//  the window it needs with ONE `nd_to_list` and then reads it inline:
//  `$B[$i] * 256 + $B[$i + 1]`, not `_u16($i)`. It is less pretty and it is
//  the difference between a font that loads and a font that does not.
//
//  ─── LICENCE OF THE EMBEDDED FONT ──────────────────────────────────────
//
//  The default glyph sheet and metric table are the ones distributed with
//  Python's Pillow in `src/PIL/ImageFont.py`, which carries the
//  Historical Permission Notice and Disclaimer:
//
//      Copyright © 1997-2011 by Secret Labs AB
//      Copyright © 1995-2011 by Fredrik Lundh and contributors
//      Pillow is the friendly PIL fork. Copyright © 2010 by Jeffrey A. Clark
//      and contributors.
//
//      By obtaining, using, and/or copying this software and/or its
//      associated documentation, you agree that you have read, understood,
//      and will comply with the following terms and conditions:
//
//      Permission to use, copy, modify and distribute this software and its
//      documentation for any purpose and without fee is hereby granted,
//      provided that the above copyright notice appears in all copies, and
//      that both that copyright notice and this permission notice appear in
//      supporting documentation, and that the name of Secret Labs AB or the
//      author not be used in advertising or publicity pertaining to
//      distribution of the software without specific, written prior
//      permission.
//
//  The data is unmodified. The notice travels with it, which is the whole
//  of what the licence asks.
// ════════════════════════════════════════════════════════════════════════

include "./pxbuf.b" as pxbuf;
include "./codecs/png.b" as png;
include "./codecs/ppm.b" as ppm;


// ════════════════════════════════════════════════════════════════════════
//  FreeType's fixed-point arithmetic, which the advances have to match
// ════════════════════════════════════════════════════════════════════════
//
//  A size of `s` pixels gives `x_scale = FT_DivFix(s * 64, units_per_EM)` in
//  16.16, and a font-unit value `v` scales to `FT_MulFix(v, x_scale)` in
//  26.6. Both round half away from zero. Getting either rounding wrong is
//  one 64th of a pixel per glyph, which accumulates visibly across a line
//  and is exactly what the differential tier measures.

// FT_DivFix(a, b) = round(a * 65536 / b), for b > 0.
def _divfix($a, $b) {
    if ($b == 0) { return 0; }
    $n = $a * 65536 + floor($b / 2);
    return floor($n / $b);
}

// FT_MulFix(a, b) = round(a * b / 65536), rounding away from zero.
def _mulfix($a, $b) {
    $v = $a * $b;
    if ($v >= 0) { return floor(($v + 32768) / 65536); }
    return 0 - floor((0 - $v + 32768) / 65536);
}

// HarfBuzz's em_scale, which is NOT FT_MulFix and had to be measured apart
// from it: floor((v * size_in_26.6 + upem / 2) / upem), so a tie rounds
// toward POSITIVE infinity rather than away from zero.
//
// It matters because the two halves of an advance come from two different
// libraries. Pillow's default layout engine is libraqm: the glyph advance is
// FreeType's FT_MulFix(hmtx, x_scale), and the GPOS pair adjustment is
// HarfBuzz's em_scale of the design value. Eight of DejaVu's seventy kern
// pairs land exactly on a half and the two rules disagree there -- "LT" at
// 8 px is -70.5 twenty-sixths of a pixel, which FT_MulFix makes -71 and
// HarfBuzz makes -70. One 64th of a pixel, in eight places, found by
// comparing every pair at every size.
def _emScale($v, $size64, $upem) {
    if ($upem == 0) { return 0; }
    return floor(($v * $size64 + floor($upem / 2)) / $upem);
}

// PIXEL(x) from _imagingft.c: ((x + 32) & -64) >> 6 — round a 26.6 value to
// whole pixels, half away from zero.
def _pixel($x) {
    if ($x >= 0) { return floor(($x + 32) / 64); }
    return 0 - floor((0 - $x + 32) / 64);
}

// FT_FLOOR / FT_CEIL on a 26.6 value, giving whole pixels. These are what
// FT_Glyph_Get_CBox(FT_GLYPH_BBOX_PIXELS) does.
def _ftFloor($x) { return floor($x / 64); }
def _ftCeil($x)  { return 0 - floor((0 - $x) / 64); }


// ════════════════════════════════════════════════════════════════════════
//  The embedded default font
// ════════════════════════════════════════════════════════════════════════
//  A PILfont file is the eight bytes "PILfont\n", a semicolon-separated
//  header line, lines of free-form info, the line "DATA\n", and then exactly
//  256 x 20 bytes of metrics: ten big-endian signed shorts per character,
//  `dx dy dx0 dy0 dx1 dy1 sx0 sy0 sx1 sy1`. The first two are the pen
//  advance; `d*` is the destination rectangle relative to the pen and the
//  baseline; `s*` is the source rectangle in the glyph sheet.

$_PILFONT_METRICS_B64 = join([
    "UElMZm9udAo7Ozs7OzsxMDsKREFUQQoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "AAAAAAYAAAAA//8AAQAAAAAAAAABAAEABgAAAAH/+gADAAAAAQAAAAMABgAGAAAAAf/6AAT//QADAAAABgADAAYA",
    "AAAA//kABQABAAYAAAALAAgABgAAAAD/+AAFAAEACwAAABAACQAGAAAAAP/5AAUAAAAQAAAAFQAHAAYAAP////oA",
    "BQAAABUAAAAbAAYABgAAAAH/+QAE//wAGwAAAB4AAwAGAAAAAf/5AAQAAQAeAAAAIQAIAAYAAAAB//kABAABACEA",
    "AAAkAAgABgAAAAD/+QAE//0AJAAAACgABAAGAAAAAP/6AAX//wAoAAAALQAFAAYAAAAB//8ABAACAC0AAAAwAAMA",
    "BgAAAAD//AAF//0AMAAAADUAAQAGAAAAAf//AAMAAAA1AAAANwABAAYAAAAB//kABQABADcAAAA7AAgABgAAAAD/",
    "+QAFAAAAOwAAAEAABwAGAAAAAP/5AAYAAABAAAAARgAHAAYAAAAA//kABQAAAEYAAABLAAcABgAAAAD/+QAFAAAA",
    "SwAAAFAABwAGAAAAAP/5AAYAAABQAAAAVgAHAAYAAAAA//kABQAAAFYAAABbAAcABgAAAAD/+QAFAAAAWwAAAGAA",
    "BwAGAAAAAP/5AAUAAABgAAAAZQAHAAYAAAAA//kABQAAAGUAAABqAAcABgAAAAD/+QAFAAAAagAAAG8ABwAGAAAA",
    "Af/8AAMAAABvAAAAcQAEAAYAAAAA//wAAwACAHEAAAB0AAYABgAAAAD/+gAE//8AdAAAAHgABQAGAAAAAP/7AAT/",
    "/gB4AAAAfAADAAYAAAAB//oABf//AHwAAACAAAUABgAAAAD/+gAFAAAAgAAAAIUABgAGAAAAAP/5AAYAAQCFAAAA",
    "iwAIAAYAAP////oABgAAAIsAAACSAAYABgAA////+gAFAAAAkgAAAJgABgAGAAAAAP/6AAUAAACYAAAAnQAGAAYA",
    "AP////oABQAAAJ0AAACjAAYABgAA////+gAFAAAAowAAAKkABgAGAAD////6AAUAAACpAAAArwAGAAYAAAAA//oA",
    "BQAAAK8AAAC0AAYABgAA////+gAGAAAAtAAAALsABgAGAAAAAP/6AAQAAAC7AAAAvwAGAAYAAP////oABQAAAL8A",
    "AADFAAYABgAA////+gAGAAAAxQAAAMwABgAGAAD////6AAUAAADMAAAA0gAGAAYAAP////oABQAAANIAAADYAAYA",
    "BgAA////+gAGAAAA2AAAAN8ABgAGAAAAAP/6AAUAAADfAAAA5AAGAAYAAP////oABQAAAOQAAADqAAYABgAAAAD/",
    "+gAFAAEA6gAAAO8ABwAGAAD////6AAYAAADvAAAA9gAGAAYAAAAA//oABQAAAPYAAAD7AAYABgAA////+gAFAAAA",
    "+wAAAQEABgAGAAD////6AAYAAAEBAAABCAAGAAYAAP////oABgAAAQgAAAEPAAYABgAA////+gAGAAABDwAAARYA",
    "BgAGAAAAAP/6AAYAAAEWAAABHAAGAAYAAP////oABgAAARwAAAEjAAYABgAAAAD/+gAFAAABIwAAASgABgAGAAAA",
    "Af/5AAQAAQEoAAABKwAIAAYAAAAA//kABAABASsAAAEvAAgABgAAAAH/+QAEAAEBLwAAATIACAAGAAAAAP/5AAX/",
    "/AEyAAABNwADAAYAAAAAAAEABgACATcAAAE9AAEABgAAAAH/+QAE//wBPQAAAUAAAwAGAAAAAP/7AAYAAAFAAAAB",
    "RgAFAAYAAP////kABQAAAUYAAAFMAAcABgAAAAD/+wAFAAABTAAAAVEABQAGAAAAAP/5AAYAAAFRAAABVwAHAAYA",
    "AAAA//sABQAAAVcAAAFcAAUABgAAAAD/+QAFAAABXAAAAWEABwAGAAAAAP/7AAYAAgFhAAABZwAHAAYAAP////kA",
    "BQAAAWcAAAFtAAcABgAAAAD/+QAGAAABbQAAAXMABwAGAAAAAP/5AAQAAgFzAAABdwAJAAYAAP////kABgAAAXcA",
    "AAF+AAcABgAAAAD/+QAGAAABfgAAAYQABwAGAAD////7AAUAAAGEAAABigAFAAYAAP////sABQAAAYoAAAGQAAUA",
    "BgAAAAD/+wAFAAABkAAAAZUABQAGAAD////7AAUAAgGVAAABmwAHAAYAAAAA//sABgACAZsAAAGhAAcABgAAAAD/",
    "+wAGAAABoQAAAacABQAGAAAAAP/7AAYAAAGnAAABrQAFAAYAAAAA//kABgAAAa0AAAGzAAcABgAA////+wAGAAAB",
    "swAAAboABQAGAAD////7AAUAAAG6AAABwAAFAAYAAP////sABgAAAcAAAAHHAAUABgAAAAD/+wAGAAABxwAAAc0A",
    "BQAGAAD////7AAYAAgHNAAAB1AAHAAYAAAAA//sABQAAAdQAAAHZAAUABgAAAAH/+QAFAAEB2QAAAd0ACAAGAAAA",
    "Av/6AAMAAQHdAAAB3gAHAAYAAAAA//kABAABAd4AAAHiAAgABgAAAAD/+wAF//0B4gAAAecAAgAAAAAAAAAAAAAA",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "AAAAAAAAAAAAAAYAAAAB//sAAwACAecAAAHpAAcABgAAAAD/+QAFAAEB6QAAAe4ACAAGAAAAAP/5AAYAAAHuAAAB",
    "9AAHAAYAAAAA//oABf//AfQAAAH5AAUABgAAAAD/+QAGAAAB+QAAAf8ABwAGAAAAAv/5AAMAAgH/AAACAAAJAAYA",
    "AAAA//kABQABAgAAAAIFAAgABgAAAAH/+gAE//sCBQAAAggAAQAGAAAAAP/5AAYAAAIIAAACDgAHAAYAAAAB//kA",
    "Bf/+Ag4AAAISAAUABgAA////+wAGAAACEgAAAhkABQAGAAAAAP/7AAX//gIZAAACHgADAAYAAAAA//wABf/9Ah4A",
    "AAIjAAEABgAAAAD/+QAHAAACIwAAAioABwAGAAAAAP/6AAT/+wIqAAACLgABAAYAAAAA//kABP/8Ai4AAAIyAAMA",
    "BgAAAAD/+gAFAAACMgAAAjcABgAGAAAAAf/5AAT//QI3AAACOgAEAAYAAAAB//kABP/9AjoAAAI9AAQABgAAAAL/",
    "+QAE//sCPQAAAj8AAgAGAAD////7AAYAAgI/AAACRgAHAAYAAAAA//kABgABAkYAAAJMAAgABgAAAAH//AAD//0C",
    "TAAAAk4AAQAGAAAAAf//AAQAAgJOAAACUQADAAYAAAAB//kABP/9AlEAAAJUAAQABgAAAAH/+QAF//4CVAAAAlgA",
    "BQAGAAD////7AAYAAAJYAAACXwAFAAYAAP////kABgAAAl8AAAJmAAcABgAA////+QAGAAACZgAAAm0ABwAGAAD/",
    "///5AAYAAAJtAAACdAAHAAYAAAAA//sABQACAnQAAAJ5AAcABgAA////9wAGAAACeQAAAoAACQAGAAD////3AAYA",
    "AAKAAAAChwAJAAYAAP////cABgAAAocAAAKOAAkABgAA////9wAGAAACjgAAApUACQAGAAD////4AAYAAAKVAAAC",
    "nAAIAAYAAP////cABgAAApwAAAKjAAkABgAA////+gAGAAACowAAAqoABgAGAAAAAP/6AAUAAgKqAAACrwAIAAYA",
    "AP////cABQAAAq8AAAK1AAkABgAA////9wAFAAACtQAAArsACQAGAAD////3AAUAAAK7AAACwQAJAAYAAP////gA",
    "BQAAAsEAAALHAAgABgAAAAD/9wAEAAACxwAAAssACQAGAAAAAP/3AAQAAALLAAACzwAJAAYAAAAA//cABAAAAs8A",
    "AALTAAkABgAAAAD/+AAEAAAC0wAAAtcACAAGAAD////6AAUAAALXAAAC3QAGAAYAAP////cABgAAAt0AAALkAAkA",
    "BgAAAAD/9wAFAAAC5AAAAukACQAGAAAAAP/3AAUAAALpAAAC7gAJAAYAAAAA//cABQAAAu4AAALzAAkABgAAAAD/",
    "9wAFAAAC8wAAAvgACQAGAAAAAP/4AAUAAAL4AAAC/QAIAAYAAAAA//oABf//Av0AAAMCAAUABgAA////+gAGAAAD",
    "AgAAAwkABgAGAAD////3AAYAAAMJAAADEAAJAAYAAP////cABgAAAxAAAAMXAAkABgAA////9wAGAAADFwAAAx4A",
    "CQAGAAD////4AAYAAAAAAAoABwASAAYAAP////cABgAAAAcACgAOABMABgAA////+gAFAAAADgAKABQAEAAGAAD/",
    "///6AAYAAAAUAAoAGwAQAAYAAAAA//gABgAAABsACgAhABIABgAAAAD/+AAGAAAAIQAKACcAEgAGAAAAAP/4AAYA",
    "AAAnAAoALQASAAYAAAAA//gABgAAAC0ACgAzABIABgAAAAD/+QAGAAAAMwAKADkAEQAGAAAAAP/3AAYAAAA5AAoA",
    "PwATAAYAAP////sABQAAAD8ACgBFAA8ABgAAAAD/+wAFAAIARQAKAEoAEQAGAAAAAP/4AAUAAABKAAoATwASAAYA",
    "AAAA//gABQAAAE8ACgBUABIABgAAAAD/+AAFAAAAVAAKAFkAEgAGAAAAAP/5AAUAAABZAAoAXgARAAYAAAAA//gA",
    "BgAAAF4ACgBkABIABgAAAAD/+AAGAAAAZAAKAGoAEgAGAAAAAP/4AAYAAABqAAoAcAASAAYAAAAA//kABgAAAHAA",
    "CgB2ABEABgAAAAD/+AAFAAAAdgAKAHsAEgAGAAD////4AAYAAAB7AAoAggASAAYAAAAA//gABQAAAIIACgCHABIA",
    "BgAAAAD/+AAFAAAAhwAKAIwAEgAGAAAAAP/4AAUAAACMAAoAkQASAAYAAAAA//gABQAAAJEACgCWABIABgAAAAD/",
    "+QAFAAAAlgAKAJsAEQAGAAAAAP/6AAX//wCbAAoAoAAPAAYAAAAA//oABQABAKAACgClABEABgAA////+AAGAAAA",
    "pQAKAKwAEgAGAAD////4AAYAAACsAAoAswASAAYAAP////gABgAAALMACgC6ABIABgAA////+QAGAAAAugAKAMEA",
    "EQAGAAD////4AAYAAgDBAAoAyAAUAAYAAP////kABQACAMgACgDOABMABgAA////+QAGAAIAzgAKANUAEw=="
], "");

$_PILFONT_SHEET_B64 = join([
    "iVBORw0KGgoAAAANSUhEUgAAAx4AAAAUAQAAAAArMtZoAAAEwElEQVR4nABlAJr/AHVE4czCI/4uMc4b7vuds/xz",
    "jz5/3/7u/n9vMe7vnfH/9++vPn/xyf5zhxzjt8GHw8+2d83u8x27199/nxuQ6Od9M43/5z2I+9n9ZtmDBwMQECDR",
    "Qw/eQIQohJXxpBCNVE6QCCAAAAD//wBlAJr/AgALyj1t/wINwq0gLeNZUworuN1cjTPIzrTX6ofHWeo3v336qPzf",
    "EwRmBnHTtf95/fglZK5N0PDgfRTslpGBvz7LFc4FIUXBWQGjQ5MGCx34EDFPwXiY4YbYxavpnhHFrk14CDAAAAD/",
    "/wBlAJr/AgKqRooH2gAgPeggvUAABu2WfgPoAwzRAABAAAAAAACQgLz/3Uv4Gv+gX7BJgDeeGP6AAAD1NMDzKHD7",
    "ANWr3loYbxsAD791NAADfcoIDyP44K/jv4Y63/Z+t98Ovt+ub4T48LAAAAD//wBlAJr/AuplMlADJAAAAGuAphWp",
    "qhMxin0A/fRvAYBABPgBwBUgABBQ/sYAyv9g0bCHgOLoGAAAAAAAREAAwI7nr0ArYpow7aX8//9LaP/9SjdavWA8",
    "ePHeBIKB//81/83ndznOaXx379wAAAD//wBlAJr/AqDxW+D3AABAAbUh/QMnbQag/gAYAYDAAACgtgD/gOqAAAB5",
    "IA/8AAAk+n9w0AAA8AAAmFRJuPo27ciC0cD5oeW4E7KA/wD3ECMAn2tty8PgwH8AfAxFzC0JzeAMtratAsC/ffwA",
    "AAD//wBlAJr/BGKAyCAA4AAAAvgeYTAwHd1kmQF5chkGABoMIHcL5xVpTfQbUqzlAAAErwAQBgAAEOClA5D9il08",
    "AEh/tUzdCBsXkbgACED+woQg8Si9VeqYlODCn7lmF6NhnAEYgAAA/NMIAAAAAAD//2JgjLZgVGBg5Pv/Tvpc8hwG",
    "BjYGJADjHDrAwPzAjv/H/Wf3PzCwtzcwHmBgYGcwbZz8wHaCAQMDOwMDQ8MCBgYOC3W7mp+f0w+wHOYxO3OG+e37",
    "6hsMZjk3AAAAAP//YmCMY2A4wMAIN5e5gQETPD6AZisDAwMDgzSDAAPjByiHcQMDAwMDg1nOze1lByRu5/47c485",
    "9311AYNZzg0AAAAA//9iYGDBYihOIIMuwIjGL39/fwffA8b//xv/P2BPtzzHwCBjUQAAAAD//yLFBrIBAAAA//9i",
    "1HhcwdhizX7u8NZNzyLbvT97bfrMf/QHI8evOwcSqGUJAAAA//9iYBB81iSwpEE170Qrg5MIYydHqwdDQRMrAwcV",
    "rQAAAAD//2J4x7j9AAMDn8Q/BgYLBoaiAwwMjPdvMDBYM1TvoJodAAAAAP//Yqo/83+dxePWlxl3npsel9lvLfPc",
    "qlE9725C+acfVLMEAAAA//9i+s9gwCoaaGMRevta/58PTEWzr21hufPjA8N+qlnBwAAAAAD//2JiWLci5v1+HmFX",
    "DqcnULE/MxgYGBj+f6CaJQAAAAD//2Ji2FrkY3iYpYC5qDeGgeEMAwPDvwQBBoYvcTwOVLMEAAAA//9isDBgkP//",
    "/0EOg9z35v//Gc/eeW7BwPj5+QGZhANUswMAAAD//2JgqGBgYGBgqEMXlvhMPUsAAAAA//8iYDd1AAAAAP//AwDR",
    "w7IkEbzhVQAAAABJRU5ErkJggg=="
], "");


// ════════════════════════════════════════════════════════════════════════
//  PILfont
// ════════════════════════════════════════════════════════════════════════

// _pilfont($metrics, $sheetSpec, $name) -> a font dict.
//
// `_font_new` in _imaging.c, including its clamping: a source rectangle that
// would reach outside the sheet is pulled back in and the DESTINATION
// rectangle is pulled in by the same amount, so a malformed table cannot
// make the mask bigger than the sheet. That is a denial-of-service guard
// upstream and a correctness one here, because the destination rectangle is
// what sets the mask's height.
def _pilfont($metrics, $sheetSpec, $name) {
    $b = bytes($metrics);
    if (len($b) != 5120) {
        throw "ImageFont: a PILfont metric table is 256 x 20 bytes, got " +
              str(len($b));
    }
    return _pilfontFromTable($b, $sheetSpec, $name);
}

// _pilfontData($file, $sheetSpec, $name) -> a font from a whole PILfont
// file: the magic, the header line, the info lines, "DATA", and then the
// table. Pillow's _load_pilfont_data, with its own order of reads.
def _pilfontData($file, $sheetSpec, $name) {
    $b = bytes($file);
    if (len($b) < 13) {
        throw "ImageFont: " + str($name) + " is too short to be a PILfont";
    }
    $magic = [];
    $i = 0;
    while ($i < 8) { push($magic, chr($b[$i])); $i = $i + 1; }
    if (join($magic, "") != "PILfont\n") {
        throw "ImageFont: " + str($name) + " is not a PILfont file";
    }
    // the header line, then info lines up to and including "DATA"
    $i = 8;
    $lines = 0;
    while ($i < len($b)) {
        $start = $i;
        while ($i < len($b) && $b[$i] != 10) { $i = $i + 1; }
        $i = $i + 1;
        $txt = [];
        $k = $start;
        while ($k < $i - 1) { push($txt, chr($b[$k])); $k = $k + 1; }
        $lines = $lines + 1;
        if ($lines > 1 && join($txt, "") == "DATA") { break; }
        if ($lines > 1000) {
            throw "ImageFont: no DATA line in " + str($name);
        }
    }
    if ($i + 5120 > len($b)) {
        throw "ImageFont: " + str($name) + " has " + str(len($b) - $i) +
              " bytes of metrics, not 5120";
    }
    $table = [];
    $k = 0;
    while ($k < 5120) { push($table, $b[$i + $k]); $k = $k + 1; }
    return _pilfontFromTable($table, $sheetSpec, $name);
}

def _pilfontFromTable($b, $sheetSpec, $name) {
    $sw = $sheetSpec.size[0];
    $sh = $sheetSpec.size[1];
    $g = [];
    $y0 = 0;
    $y1 = 0;
    $i = 0;
    while ($i < 256) {
        $o = $i * 20;
        $row = [];
        $k = 0;
        while ($k < 10) {
            $v = $b[$o + $k * 2] * 256 + $b[$o + $k * 2 + 1];
            if ($v >= 32768) { $v = $v - 65536; }
            push($row, $v);
            $k = $k + 1;
        }
        // row is dx dy dx0 dy0 dx1 dy1 sx0 sy0 sx1 sy1
        if ($row[6] < 0) { $row[2] = $row[2] - $row[6]; $row[6] = 0; }
        if ($row[7] < 0) { $row[3] = $row[3] - $row[7]; $row[7] = 0; }
        if ($row[8] > $sw) { $row[4] = $row[4] - ($row[8] - $sw); $row[8] = $sw; }
        if ($row[9] > $sh) { $row[5] = $row[5] - ($row[9] - $sh); $row[9] = $sh; }
        if ($row[3] < $y0) { $y0 = $row[3]; }
        if ($row[5] > $y1) { $y1 = $row[5]; }
        push($g, $row);
        $i = $i + 1;
    }
    // One flat numba array, so the table crosses a call boundary by
    // reference and a caller lifts it back with a single nd_to_list.
    $flat = [];
    each ($row in $g) { each ($v in $row) { push($flat, $v); } }
    return {"kind": "pil", "glyphs": nd($flat, "i64"),
            "sheet": $sheetSpec.px, "sheetMode": $sheetSpec.mode,
            "sheetW": $sw, "sheetH": $sh,
            "baseline": 0 - $y0, "ysize": $y1 - $y0,
            "path": $name, "size": null, "index": 0};
}

// load_default($size) -> the embedded bitmap font.
//
// PILLOW'S load_default RETURNS SOMETHING ELSE when FreeType is available:
// a FreeType font over an embedded copy of Aileron Regular, which is a CFF
// (PostScript-outline) OpenType font. This reader is a `glyf` reader, so
// Aileron is not readable here, and a CFF interpreter is a second font
// format's worth of work for one embedded file. What is returned instead is
// the font Pillow itself falls back to without FreeType, and which it still
// exposes as `ImageFont.load_default_imagefont()` — so this call matches
// THAT one byte for byte. `$size` is accepted and ignored, because a bitmap
// font has one size; pass a TrueType face to `truetype()` for a scalable one.
def load_default($size) {
    return _pilfontData(b64decode($_PILFONT_METRICS_B64),
                        png.decode(b64decode($_PILFONT_SHEET_B64)),
                        "<default>");
}

def load_default_imagefont() { return load_default(null); }

// load($path) -> a PILfont from a .pil metric file beside its glyph sheet.
// Pillow looks for the sheet as <root>.png, <root>.gif or <root>.pbm, in
// that order, and takes the first that opens in mode "1" or "L".
def load($path) {
    $raw = readfile($path, "rb");
    $root = $path;
    $dot = 0 - 1;
    $k = len($path) - 1;
    while ($k >= 0) {
        if ($path[$k] == ".") { $dot = $k; break; }
        if ($path[$k] == "/") { break; }
        $k = $k - 1;
    }
    if ($dot > 0) { $root = substr($path, 0, $dot); }
    $sheet = null;
    $tried = [];
    each ($ext in [".png", ".pbm"]) {
        if ($sheet != null) { continue; }
        push($tried, $root + $ext);
        try {
            $d = readfile($root + $ext, "rb");
            if ($ext == ".png") { $sheet = png.decode($d); }
            else { $sheet = _openPbm($d); }
        } catch ($e) { $sheet = null; }
        if ($sheet != null && $sheet.mode != "1" && $sheet.mode != "L") { $sheet = null; }
    }
    if ($sheet == null) {
        throw "ImageFont.load: cannot find a glyph sheet for " + str($path) +
              " (tried " + join($tried, ", ") + "). A .gif sheet needs the GIF " +
              "codec, which arrives in phase P8.";
    }
    return _pilfontData($raw, $sheet, $path);
}

// load_path($path) -> Pillow searches sys.path; Bantu has no import path, so
// this tries the name as given and then "fonts/" beside it, and says what it
// tried when it fails.
def load_path($path) {
    $tried = [$path, "fonts/" + $path];
    each ($p in $tried) {
        try { return load($p); } catch ($e) { }
    }
    throw "ImageFont.load_path: cannot find " + str($path) +
          " (tried " + join($tried, ", ") + ")";
}

// A PBM sheet, for the .pil fonts that ship with one.
def _openPbm($data) { return ppm.decode($data); }


// ════════════════════════════════════════════════════════════════════════
//  TrueType
// ════════════════════════════════════════════════════════════════════════

// truetype($path, $size, $index, $encoding, $layout_engine) -> a font dict.
//
// `$index` selects a face in a TrueType collection; `$encoding` picks the
// cmap platform/encoding pair the way Pillow's does ("unic", "symb",
// "ADOB"...); `$layout_engine` is accepted and recorded, because this
// package has one layout engine and saying so is better than ignoring the
// argument.
def truetype($path, $size, $index, $encoding, $layout_engine) {
    $sz = $size;
    if ($sz == null) { $sz = 10; }
    if ($sz <= 0) { throw "ImageFont.truetype: size must be positive, got " + str($sz); }
    $idx = $index;
    if ($idx == null) { $idx = 0; }

    $raw = readfile($path, "rb");
    $data = nd_frombytes($raw, "u8", null);
    $n = pxbuf.scalar(nd_size($data));
    if ($n < 12) { throw "ImageFont.truetype: " + str($path) + " is " + str($n) + " bytes, too short for a font"; }

    $base = 0;
    $B = nd_to_list(nd_slice($data, [[0, 12]]));
    $tag = $B[0] * 16777216 + $B[1] * 65536 + $B[2] * 256 + $B[3];
    if ($tag == 1953784678) {                            // 'ttcf'
        $H = nd_to_list(nd_slice($data, [[0, 16 + 4 * ($idx + 1)]]));
        $nfaces = $H[8] * 16777216 + $H[9] * 65536 + $H[10] * 256 + $H[11];
        if ($idx >= $nfaces) {
            throw "ImageFont.truetype: face " + str($idx) + " of a collection with " +
                  str($nfaces) + " faces";
        }
        $o = 12 + 4 * $idx;
        $base = $H[$o] * 16777216 + $H[$o + 1] * 65536 + $H[$o + 2] * 256 + $H[$o + 3];
        $B = nd_to_list(nd_slice($data, [[$base, $base + 12]]));
        $tag = $B[0] * 16777216 + $B[1] * 65536 + $B[2] * 256 + $B[3];
    } else if ($idx != 0) {
        throw "ImageFont.truetype: " + str($path) + " is not a collection, so face " +
              str($idx) + " does not exist";
    }
    if ($tag != 65536 && $tag != 1953658213) {           // 0x00010000, 'true'
        if ($tag == 1330926671) {                        // 'OTTO'
            throw "ImageFont.truetype: " + str($path) + " is a CFF OpenType font " +
                  "(PostScript outlines). This reader reads 'glyf' outlines; a CFF " +
                  "interpreter is not in this build.";
        }
        if ($tag == 2001684018 || $tag == 2001684019) {  // 'wOFF', 'wOF2'
            throw "ImageFont.truetype: " + str($path) + " is a WOFF font; unpack it " +
                  "to a .ttf first.";
        }
        throw "ImageFont.truetype: " + str($path) + " has sfnt version " + str($tag) +
              ", which is not a TrueType outline font";
    }

    $numTables = $B[4] * 256 + $B[5];
    if ($numTables == 0 || $base + 12 + 16 * $numTables > $n) {
        throw "ImageFont.truetype: the table directory of " + str($path) +
              " claims " + str($numTables) + " tables, which runs past the file";
    }
    $T = nd_to_list(nd_slice($data, [[$base + 12, $base + 12 + 16 * $numTables]]));
    $tables = {};
    $i = 0;
    while ($i < $numTables) {
        $r = $i * 16;
        $name = join([chr($T[$r]), chr($T[$r + 1]), chr($T[$r + 2]), chr($T[$r + 3])], "");
        $off = $T[$r + 8] * 16777216 + $T[$r + 9] * 65536 + $T[$r + 10] * 256 + $T[$r + 11];
        $nbytes = $T[$r + 12] * 16777216 + $T[$r + 13] * 65536 + $T[$r + 14] * 256 + $T[$r + 15];
        if ($off > $n) {
            throw "ImageFont.truetype: table '" + $name + "' starts at byte " +
                  str($off) + ", past the end of a " + str($n) + "-byte font";
        }
        if ($off + $nbytes > $n) { $nbytes = $n - $off; }
        $tables[$name] = [$off, $nbytes];
        $i = $i + 1;
    }
    each ($need in ["head", "maxp", "hhea", "hmtx", "loca", "glyf", "cmap"]) {
        if ($tables[$need] == null) {
            throw "ImageFont.truetype: " + str($path) + " has no '" + $need + "' table";
        }
    }

    $f = {"kind": "tt", "data": $data, "tables": $tables, "path": $path,
          "size": $sz, "index": $idx, "encoding": $encoding,
          "layout_engine": $layout_engine,
          "outlines": {}, "masks": {}, "advCache": {}};

    // ─── head: units per EM, and the loca format ───
    $H = nd_to_list(nd_slice($data, [[$tables.head[0], $tables.head[0] + 54]]));
    $f.upem = $H[18] * 256 + $H[19];
    if ($f.upem < 16 || $f.upem > 16384) {
        throw "ImageFont.truetype: unitsPerEm is " + str($f.upem) +
              ", which is outside the legal 16..16384";
    }
    $ilf = $H[50] * 256 + $H[51];
    if ($ilf >= 32768) { $ilf = $ilf - 65536; }
    $f.longLoca = $ilf == 1;

    // ─── maxp, hhea ───
    $M = nd_to_list(nd_slice($data, [[$tables.maxp[0], $tables.maxp[0] + 6]]));
    $f.numGlyphs = $M[4] * 256 + $M[5];
    $A = nd_to_list(nd_slice($data, [[$tables.hhea[0], $tables.hhea[0] + 36]]));
    $f.ascender_fu = _s16($A[4] * 256 + $A[5]);
    $f.descender_fu = _s16($A[6] * 256 + $A[7]);
    $f.lineGap_fu = _s16($A[8] * 256 + $A[9]);
    $f.numHMetrics = $A[34] * 256 + $A[35];
    if ($f.numHMetrics < 1) { throw "ImageFont.truetype: numberOfHMetrics is 0"; }

    // ─── the scale, and the metrics FreeType would report ───
    $f.size64 = floor($sz * 64);
    $f.xscale = _divfix(floor($sz * 64), $f.upem);
    $f.yscale = $f.xscale;
    $f.ascender = _pixel(_mulfix($f.ascender_fu, $f.yscale) + 63 - (_mulfix($f.ascender_fu, $f.yscale) % 64 + 64) % 64 + ((_mulfix($f.ascender_fu, $f.yscale) % 64 + 64) % 64));
    // FreeType grid-fits the face metrics: ascender up, descender down.
    $f.ascender = _ftCeil(_mulfix($f.ascender_fu, $f.yscale)) * 64;
    $f.descender = _ftFloor(_mulfix($f.descender_fu, $f.yscale)) * 64;
    $f.height = _pixel(_mulfix($f.ascender_fu - $f.descender_fu + $f.lineGap_fu, $f.yscale));

    // ─── hmtx: every advance and bearing, once, as arrays ───
    $hm = $tables.hmtx;
    $want = 4 * $f.numHMetrics + 2 * ($f.numGlyphs - $f.numHMetrics);
    if ($want > $hm[1]) { $want = $hm[1]; }
    $HM = nd_to_list(nd_slice($data, [[$hm[0], $hm[0] + $want]]));
    $advs = [];
    $lsbs = [];
    $lastAdv = 0;
    $i = 0;
    while ($i < $f.numGlyphs) {
        if ($i < $f.numHMetrics) {
            $o = 4 * $i;
            if ($o + 3 < len($HM)) {
                $lastAdv = $HM[$o] * 256 + $HM[$o + 1];
                push($lsbs, _s16($HM[$o + 2] * 256 + $HM[$o + 3]));
            } else { push($lsbs, 0); }
        } else {
            $o = 4 * $f.numHMetrics + 2 * ($i - $f.numHMetrics);
            if ($o + 1 < len($HM)) { push($lsbs, _s16($HM[$o] * 256 + $HM[$o + 1])); }
            else { push($lsbs, 0); }
        }
        push($advs, $lastAdv);
        $i = $i + 1;
    }
    $f.adv = nd($advs, "i64");
    $f.lsb = nd($lsbs, "i64");

    // ─── loca ───
    $lo = $tables.loca;
    $cnt = $f.numGlyphs + 1;
    $step = 2;
    if ($f.longLoca) { $step = 4; }
    $need = $cnt * $step;
    if ($need > $lo[1]) { $need = $lo[1]; }
    $L = nd_to_list(nd_slice($data, [[$lo[0], $lo[0] + $need]]));
    $offs = [];
    $i = 0;
    while ($i < $cnt) {
        $o = $i * $step;
        if ($o + $step > len($L)) { push($offs, 0); }
        else if ($f.longLoca) {
            push($offs, $L[$o] * 16777216 + $L[$o + 1] * 65536 + $L[$o + 2] * 256 + $L[$o + 3]);
        } else {
            push($offs, ($L[$o] * 256 + $L[$o + 1]) * 2);
        }
        $i = $i + 1;
    }
    $f.loca = nd($offs, "i64");

    // ─── cmap, kern, GPOS, fvar, name ───
    $f.cmap = _readCmap($f, $data, $tables.cmap, $encoding);
    $f.kern = _readKern($f, $data, $tables.kern);
    $f.gpos = _readGpos($f, $data, $tables.GPOS);
    return $f;
}


// ─── Text is UTF-8 bytes, so a codepoint has to be decoded ──────────────
//
//  A Bantu string is a BYTE string: len("Angstrom" with an umlaut) is 10,
//  not 8, and `ord($s[0])` is 195 — the first byte of the UTF-8 pair — not
//  196. Every font lookup is by codepoint, so every string that reaches one
//  goes through here first.
//
//  An invalid sequence falls back to LATIN-1, byte for byte, rather than
//  raising. That is not laziness: Pillow accepts `bytes` as well as `str`
//  and uses the bytes directly, and a Bantu string cannot say which it is.
//  Valid UTF-8 is decoded as UTF-8; anything else is read as the bytes it
//  is, which is exactly what Pillow does with a bytes argument.
def _codepoints($text) {
    $n = len($text);
    $out = [];
    $i = 0;
    while ($i < $n) {
        $b = ord($text[$i]);
        $cp = 0 - 1;
        $extra = 0;
        if ($b < 128) { $cp = $b; }
        else if ($b >= 194 && $b < 224) { $cp = $b - 192; $extra = 1; }
        else if ($b >= 224 && $b < 240) { $cp = $b - 224; $extra = 2; }
        else if ($b >= 240 && $b < 245) { $cp = $b - 240; $extra = 3; }
        if ($cp < 0 || $i + $extra >= $n) {
            // not a UTF-8 lead byte, or a truncated sequence: Latin-1.
            push($out, $b);
            $i = $i + 1;
            continue;
        }
        $ok = true;
        $k = 1;
        while ($k <= $extra) {
            $c = ord($text[$i + $k]);
            if ($c < 128 || $c >= 192) { $ok = false; break; }
            $cp = $cp * 64 + ($c - 128);
            $k = $k + 1;
        }
        if (!$ok) {
            push($out, $b);
            $i = $i + 1;
            continue;
        }
        push($out, $cp);
        $i = $i + 1 + $extra;
    }
    return $out;
}

// One codepoint, as UTF-8 bytes in a string. chr() emits one BYTE, so
// anything past U+007F has to be assembled.
def _utf8enc($cp) {
    if ($cp < 128) { return chr($cp); }
    if ($cp < 2048) {
        return join([chr(192 + floor($cp / 64)), chr(128 + $cp % 64)], "");
    }
    if ($cp < 65536) {
        return join([chr(224 + floor($cp / 4096)),
                     chr(128 + floor($cp / 64) % 64), chr(128 + $cp % 64)], "");
    }
    return join([chr(240 + floor($cp / 262144)),
                 chr(128 + floor($cp / 4096) % 64),
                 chr(128 + floor($cp / 64) % 64), chr(128 + $cp % 64)], "");
}

def _s16($v) { if ($v >= 32768) { return $v - 65536; } return $v; }
def _s8($v)  { if ($v >= 128) { return $v - 256; } return $v; }


// ─── cmap ───────────────────────────────────────────────────────────────
//
//  Formats 0, 4, 6 and 12 — the four that matter. The result is one numba
//  array of 65,536 glyph ids for the BMP, so a lookup is a single gather and
//  a string's glyph ids cost one array call rather than a binary search per
//  character. Format 12's astral planes stay as segment arrays, because a
//  1,114,112-entry table to hold a handful of glyphs is not a trade worth
//  making.
def _readCmap($f, $data, $cm, $encoding) {
    $off = $cm[0];
    $H = nd_to_list(nd_slice($data, [[$off, $off + 4]]));
    $nsub = $H[2] * 256 + $H[3];
    if ($nsub == 0) { throw "ImageFont.truetype: the cmap has no subtables"; }
    $R = nd_to_list(nd_slice($data, [[$off + 4, $off + 4 + 8 * $nsub]]));
    // Pillow's encoding names map onto platform/encoding pairs; the default
    // is Unicode, preferring (3,10) then (3,1) then (0,*).
    $wantPairs = [[3, 10], [3, 1], [0, 6], [0, 4], [0, 3], [0, 2], [0, 1], [0, 0]];
    if ($encoding == "symb") { $wantPairs = [[3, 0]]; }
    else if ($encoding == "ADBC" || $encoding == "ADOB") { $wantPairs = [[1, 0]]; }
    else if ($encoding == "armn") { $wantPairs = [[1, 0]]; }
    else if ($encoding != null && $encoding != "" && $encoding != "unic") {
        throw "ImageFont.truetype: encoding \"" + str($encoding) +
              "\" is not one this build knows (unic, symb, ADOB)";
    }
    $chosen = 0 - 1;
    each ($pair in $wantPairs) {
        if ($chosen >= 0) { continue; }
        $i = 0;
        while ($i < $nsub) {
            $r = $i * 8;
            $pid = $R[$r] * 256 + $R[$r + 1];
            $eid = $R[$r + 2] * 256 + $R[$r + 3];
            if ($pid == $pair[0] && $eid == $pair[1]) {
                $chosen = $off + $R[$r + 4] * 16777216 + $R[$r + 5] * 65536 +
                          $R[$r + 6] * 256 + $R[$r + 7];
                break;
            }
            $i = $i + 1;
        }
    }
    if ($chosen < 0) {
        // Nothing recognised: take the first subtable rather than refuse, as
        // FreeType does, but say which one in the font dict.
        $chosen = $off + $R[4] * 16777216 + $R[5] * 65536 + $R[6] * 256 + $R[7];
    }
    $F = nd_to_list(nd_slice($data, [[$chosen, $chosen + 4]]));
    $fmt = $F[0] * 256 + $F[1];
    $bmp = nd_zeros([65536], "i64");
    $extra = [];
    if ($fmt == 0) {
        $S = nd_to_list(nd_slice($data, [[$chosen + 6, $chosen + 262]]));
        $i = 0;
        while ($i < 256) { nd_set($bmp, [$i], $S[$i]); $i = $i + 1; }
    } else if ($fmt == 4) {
        $nbytes = $F[2] * 256 + $F[3];
        if ($nbytes < 14) { throw "ImageFont.truetype: a format-4 cmap of " + str($nbytes) + " bytes"; }
        $S = nd_to_list(nd_slice($data, [[$chosen, $chosen + $nbytes]]));
        $segX2 = $S[6] * 256 + $S[7];
        $segs = floor($segX2 / 2);
        $ends = 14;
        $starts = $ends + $segX2 + 2;
        $deltas = $starts + $segX2;
        $ranges = $deltas + $segX2;
        $i = 0;
        while ($i < $segs) {
            $end = $S[$ends + 2 * $i] * 256 + $S[$ends + 2 * $i + 1];
            $start = $S[$starts + 2 * $i] * 256 + $S[$starts + 2 * $i + 1];
            $delta = $S[$deltas + 2 * $i] * 256 + $S[$deltas + 2 * $i + 1];
            $ro = $S[$ranges + 2 * $i] * 256 + $S[$ranges + 2 * $i + 1];
            if ($start > $end || $end > 65535) { $i = $i + 1; continue; }
            if ($ro == 0) {
                // gid = (cp + delta) mod 65536, over the whole run: one
                // arange, one add, one modulo, one slice write.
                $run = nd_arange($start, $end + 1, 1);
                $gids = nd_mod(nd_add($run, $delta), 65536);
                $dst = nd_slice($bmp, [[$start, $end + 1]]);
                nd_add($gids, 0, $dst);
            } else {
                $cp = $start;
                while ($cp <= $end) {
                    $o = $ranges + 2 * $i + $ro + 2 * ($cp - $start);
                    if ($o + 1 < len($S)) {
                        $g = $S[$o] * 256 + $S[$o + 1];
                        if ($g != 0) { nd_set($bmp, [$cp], ($g + $delta) % 65536); }
                    }
                    $cp = $cp + 1;
                }
            }
            $i = $i + 1;
        }
    } else if ($fmt == 6) {
        $S = nd_to_list(nd_slice($data, [[$chosen, $chosen + 10]]));
        $first = $S[6] * 256 + $S[7];
        $count = $S[8] * 256 + $S[9];
        if ($count > 0) {
            $G = nd_slice($data, [[$chosen + 10, $chosen + 10 + 2 * $count]]);
            $hi = nd_slice($G, [[0, 2 * $count, 2]]);
            $lo = nd_slice($G, [[1, 2 * $count, 2]]);
            $gids = nd_add(nd_multiply($hi, 256), $lo);
            $last = $first + $count;
            if ($last > 65536) { $last = 65536; }
            $dst = nd_slice($bmp, [[$first, $last]]);
            nd_add(nd_slice($gids, [[0, $last - $first]]), 0, $dst);
        }
    } else if ($fmt == 12) {
        $S = nd_to_list(nd_slice($data, [[$chosen, $chosen + 16]]));
        $ngroups = $S[12] * 16777216 + $S[13] * 65536 + $S[14] * 256 + $S[15];
        if ($ngroups > 200000) {
            throw "ImageFont.truetype: a format-12 cmap with " + str($ngroups) + " groups";
        }
        $G = nd_to_list(nd_slice($data, [[$chosen + 16, $chosen + 16 + 12 * $ngroups]]));
        $i = 0;
        while ($i < $ngroups) {
            $o = $i * 12;
            $s = $G[$o] * 16777216 + $G[$o + 1] * 65536 + $G[$o + 2] * 256 + $G[$o + 3];
            $e = $G[$o + 4] * 16777216 + $G[$o + 5] * 65536 + $G[$o + 6] * 256 + $G[$o + 7];
            $g = $G[$o + 8] * 16777216 + $G[$o + 9] * 65536 + $G[$o + 10] * 256 + $G[$o + 11];
            if ($e >= $s) {
                if ($s <= 65535) {
                    $hiEnd = $e;
                    if ($hiEnd > 65535) { $hiEnd = 65535; }
                    $run = nd_arange(0, $hiEnd - $s + 1, 1);
                    $dst = nd_slice($bmp, [[$s, $hiEnd + 1]]);
                    nd_add(nd_add($run, $g), 0, $dst);
                }
                if ($e > 65535) { push($extra, [$s, $e, $g]); }
            }
            $i = $i + 1;
        }
    } else {
        throw "ImageFont.truetype: cmap format " + str($fmt) +
              " is not one this build reads (0, 4, 6 and 12 are)";
    }
    return {"format": $fmt, "bmp": $bmp, "extra": $extra};
}

// gid($f, $cp) -> the glyph id for one codepoint, 0 when the font has none.
def gid($f, $cp) {
    if ($f.kind != "tt") { throw "ImageFont: gid() needs a TrueType font"; }
    if ($cp < 0) { return 0; }
    if ($cp < 65536) { return pxbuf.scalar(nd_get($f.cmap.bmp, [$cp])); }
    each ($g in $f.cmap.extra) {
        if ($cp >= $g[0] && $cp <= $g[1]) { return $g[2] + ($cp - $g[0]); }
    }
    return 0;
}

// _gids($f, $text) -> the glyph ids of a string, as a Bantu list. One array
// gather for the whole string.
def _gids($f, $text) {
    $cps = _codepoints($text);
    if (len($cps) == 0) { return []; }
    $bmpOnly = true;
    each ($c in $cps) { if ($c >= 65536) { $bmpOnly = false; } }
    if ($bmpOnly) { return nd_to_list(nd_take($f.cmap.bmp, nd($cps, "i64"))); }
    $out = [];
    each ($c in $cps) { push($out, gid($f, $c)); }
    return $out;
}


// ─── kerning ────────────────────────────────────────────────────────────
//
//  Two sources, and GPOS wins because that is what HarfBuzz uses and
//  HarfBuzz is what Pillow's default layout engine asks. The legacy `kern`
//  table is the fallback for fonts that have no GPOS 'kern' feature.
//
//  Both are read into one flat dict keyed "left:right" -> value in font
//  units. A dict of a few thousand entries read by key is a hash lookup, and
//  reading a NUMBER out of a dict does not copy anything.

def _readKern($f, $data, $kt) {
    if ($kt == null) { return {}; }
    $out = {};
    $off = $kt[0];
    $H = nd_to_list(nd_slice($data, [[$off, $off + 4]]));
    $ntab = $H[2] * 256 + $H[3];
    if ($H[0] * 256 + $H[1] != 0) { return {}; }     // only version 0 is portable
    $p = $off + 4;
    $t = 0;
    while ($t < $ntab && $t < 32) {
        $S = nd_to_list(nd_slice($data, [[$p, $p + 14]]));
        $nbytes = $S[2] * 256 + $S[3];
        $fmt = $S[4];
        $coverage = $S[5];
        if ($nbytes < 14) { break; }
        if ($fmt == 0 && $coverage % 2 == 1) {         // horizontal, not minimum
            $npairs = $S[6] * 256 + $S[7];
            $need = 6 * $npairs;
            if ($need > $nbytes - 14) { $need = $nbytes - 14; }
            $P = nd_to_list(nd_slice($data, [[$p + 14, $p + 14 + $need]]));
            $i = 0;
            while ($i * 6 + 5 < len($P)) {
                $o = $i * 6;
                $l = $P[$o] * 256 + $P[$o + 1];
                $r = $P[$o + 2] * 256 + $P[$o + 3];
                $v = _s16($P[$o + 4] * 256 + $P[$o + 5]);
                if ($v != 0) { $out[str($l) + ":" + str($r)] = $v; }
                $i = $i + 1;
            }
        }
        $p = $p + $nbytes;
        $t = $t + 1;
    }
    return $out;
}

// GPOS lookup type 2, pair adjustment, formats 1 and 2, restricted to the
// 'kern' feature of the default language system — which is what HarfBuzz
// applies by default and so what Pillow's advances carry.
def _readGpos($f, $data, $gt) {
    if ($gt == null) { return {}; }
    $off = $gt[0];
    $H = nd_to_list(nd_slice($data, [[$off, $off + 10]]));
    $scriptOff = $off + $H[4] * 256 + $H[5];
    $featureOff = $off + $H[6] * 256 + $H[7];
    $lookupOff = $off + $H[8] * 256 + $H[9];

    // Every 'kern' feature's lookup indices.
    $FL = nd_to_list(nd_slice($data, [[$featureOff, $featureOff + 2]]));
    $nfeat = $FL[0] * 256 + $FL[1];
    if ($nfeat == 0 || $nfeat > 4096) { return {}; }
    $FR = nd_to_list(nd_slice($data, [[$featureOff + 2, $featureOff + 2 + 6 * $nfeat]]));
    $lookups = [];
    $i = 0;
    while ($i < $nfeat) {
        $o = $i * 6;
        $tag = join([chr($FR[$o]), chr($FR[$o + 1]), chr($FR[$o + 2]), chr($FR[$o + 3])], "");
        if ($tag == "kern") {
            $fo = $featureOff + $FR[$o + 4] * 256 + $FR[$o + 5];
            $FT = nd_to_list(nd_slice($data, [[$fo, $fo + 4]]));
            $nl = $FT[2] * 256 + $FT[3];
            if ($nl > 0 && $nl < 4096) {
                $LI = nd_to_list(nd_slice($data, [[$fo + 4, $fo + 4 + 2 * $nl]]));
                $k = 0;
                while ($k < $nl) { push($lookups, $LI[2 * $k] * 256 + $LI[2 * $k + 1]); $k = $k + 1; }
            }
        }
        $i = $i + 1;
    }
    if (len($lookups) == 0) { return {}; }

    $LL = nd_to_list(nd_slice($data, [[$lookupOff, $lookupOff + 2]]));
    $nlook = $LL[0] * 256 + $LL[1];
    if ($nlook == 0 || $nlook > 4096) { return {}; }
    $LO = nd_to_list(nd_slice($data, [[$lookupOff + 2, $lookupOff + 2 + 2 * $nlook]]));
    $out = {};
    each ($li in $lookups) {
        if ($li >= $nlook) { continue; }
        $lo = $lookupOff + $LO[2 * $li] * 256 + $LO[2 * $li + 1];
        $LH = nd_to_list(nd_slice($data, [[$lo, $lo + 6]]));
        $ltype = $LH[0] * 256 + $LH[1];
        $nsub = $LH[4] * 256 + $LH[5];
        if ($ltype != 2 || $nsub == 0 || $nsub > 4096) { continue; }
        $SL = nd_to_list(nd_slice($data, [[$lo + 6, $lo + 6 + 2 * $nsub]]));
        $s = 0;
        while ($s < $nsub) {
            _gposPair($data, $lo + $SL[2 * $s] * 256 + $SL[2 * $s + 1], $out);
            $s = $s + 1;
        }
    }
    return $out;
}

// One PairPos subtable, written into $out (a dict, so it is by reference).
def _gposPair($data, $off, $out) {
    $H = nd_to_list(nd_slice($data, [[$off, $off + 10]]));
    $fmt = $H[0] * 256 + $H[1];
    $covOff = $off + $H[2] * 256 + $H[3];
    $vf1 = $H[4] * 256 + $H[5];
    $vf2 = $H[6] * 256 + $H[7];
    // Only an X advance on the first glyph is kerning; anything else is
    // positioning this layout engine does not do, and silently taking its
    // first field would be worse than ignoring it.
    if ($vf1 != 4) { return null; }
    $sz1 = _vrSize($vf1);
    $sz2 = _vrSize($vf2);
    $cov = _readCoverage($data, $covOff);
    if ($fmt == 1) {
        $n = $H[8] * 256 + $H[9];
        if ($n == 0 || $n > 65535) { return null; }
        $PS = nd_to_list(nd_slice($data, [[$off + 10, $off + 10 + 2 * $n]]));
        $i = 0;
        while ($i < $n && $i < len($cov)) {
            $so = $off + $PS[2 * $i] * 256 + $PS[2 * $i + 1];
            $SH = nd_to_list(nd_slice($data, [[$so, $so + 2]]));
            $cnt = $SH[0] * 256 + $SH[1];
            if ($cnt > 0 && $cnt < 65536) {
                $rec = 2 + $sz1 + $sz2;
                $R = nd_to_list(nd_slice($data, [[$so + 2, $so + 2 + $rec * $cnt]]));
                $k = 0;
                while ($k < $cnt) {
                    $o = $k * $rec;
                    $second = $R[$o] * 256 + $R[$o + 1];
                    $v = _s16($R[$o + 2] * 256 + $R[$o + 3]);
                    if ($v != 0) { $out[str($cov[$i]) + ":" + str($second)] = $v; }
                    $k = $k + 1;
                }
            }
            $i = $i + 1;
        }
        return null;
    }
    if ($fmt == 2) {
        $H2 = nd_to_list(nd_slice($data, [[$off, $off + 16]]));
        $cd1 = $off + $H2[8] * 256 + $H2[9];
        $cd2 = $off + $H2[10] * 256 + $H2[11];
        $n1 = $H2[12] * 256 + $H2[13];
        $n2 = $H2[14] * 256 + $H2[15];
        if ($n1 == 0 || $n2 == 0 || $n1 * $n2 > 1000000) { return null; }
        $rec = $sz1 + $sz2;
        $R = nd_to_list(nd_slice($data, [[$off + 16, $off + 16 + $rec * $n1 * $n2]]));
        $g1 = _readClassDef($data, $cd1);
        $g2 = _readClassDef($data, $cd2);
        // class 0 of ClassDef1 is "every covered glyph not otherwise
        // classified", so the coverage list supplies it.
        $byClass1 = {};
        for $g, $c in $g1.items() { }
        $i = 0;
        while ($i < len($cov)) {
            $g = $cov[$i];
            $c1 = $g1.map[str($g)];
            if ($c1 == null) { $c1 = 0; }
            $j = 0;
            while ($j < len($g2.glyphs)) {
                $og = $g2.glyphs[$j];
                $c2 = $g2.map[str($og)];
                if ($c2 == null) { $c2 = 0; }
                if ($c1 < $n1 && $c2 < $n2) {
                    $o = ($c1 * $n2 + $c2) * $rec;
                    if ($o + 1 < len($R)) {
                        $v = _s16($R[$o] * 256 + $R[$o + 1]);
                        if ($v != 0) { $out[str($g) + ":" + str($og)] = $v; }
                    }
                }
                $j = $j + 1;
            }
            $i = $i + 1;
        }
        return null;
    }
    return null;
}

// The size in bytes of a ValueRecord with this ValueFormat: one 16-bit field
// per set bit.
def _vrSize($vf) {
    $n = 0;
    $b = 0;
    while ($b < 16) {
        $bit = pow(2, $b);
        if (floor($vf / $bit) % 2 == 1) { $n = $n + 2; }
        $b = $b + 1;
    }
    return $n;
}

// A Coverage table -> the list of glyph ids it covers, in coverage order.
def _readCoverage($data, $off) {
    $H = nd_to_list(nd_slice($data, [[$off, $off + 4]]));
    $fmt = $H[0] * 256 + $H[1];
    $n = $H[2] * 256 + $H[3];
    $out = [];
    if ($fmt == 1) {
        if ($n == 0 || $n > 65536) { return $out; }
        $G = nd_to_list(nd_slice($data, [[$off + 4, $off + 4 + 2 * $n]]));
        $i = 0;
        while ($i < $n) { push($out, $G[2 * $i] * 256 + $G[2 * $i + 1]); $i = $i + 1; }
        return $out;
    }
    if ($fmt == 2) {
        if ($n == 0 || $n > 65536) { return $out; }
        $G = nd_to_list(nd_slice($data, [[$off + 4, $off + 4 + 6 * $n]]));
        $i = 0;
        while ($i < $n) {
            $o = $i * 6;
            $s = $G[$o] * 256 + $G[$o + 1];
            $e = $G[$o + 2] * 256 + $G[$o + 3];
            $g = $s;
            while ($g <= $e && $g <= 65535) { push($out, $g); $g = $g + 1; }
            $i = $i + 1;
        }
        return $out;
    }
    return $out;
}

// A ClassDef table -> {"map": glyph -> class, "glyphs": [the listed glyphs]}.
def _readClassDef($data, $off) {
    $H = nd_to_list(nd_slice($data, [[$off, $off + 6]]));
    $fmt = $H[0] * 256 + $H[1];
    $map = {};
    $glyphs = [];
    if ($fmt == 1) {
        $start = $H[2] * 256 + $H[3];
        $n = $H[4] * 256 + $H[5];
        if ($n == 0 || $n > 65536) { return {"map": $map, "glyphs": $glyphs}; }
        $G = nd_to_list(nd_slice($data, [[$off + 6, $off + 6 + 2 * $n]]));
        $i = 0;
        while ($i < $n) {
            $c = $G[2 * $i] * 256 + $G[2 * $i + 1];
            $map[str($start + $i)] = $c;
            push($glyphs, $start + $i);
            $i = $i + 1;
        }
        return {"map": $map, "glyphs": $glyphs};
    }
    if ($fmt == 2) {
        $n = $H[2] * 256 + $H[3];
        if ($n == 0 || $n > 65536) { return {"map": $map, "glyphs": $glyphs}; }
        $G = nd_to_list(nd_slice($data, [[$off + 4, $off + 4 + 6 * $n]]));
        $i = 0;
        while ($i < $n) {
            $o = $i * 6;
            $s = $G[$o] * 256 + $G[$o + 1];
            $e = $G[$o + 2] * 256 + $G[$o + 3];
            $c = $G[$o + 4] * 256 + $G[$o + 5];
            $g = $s;
            while ($g <= $e && $g <= 65535) {
                $map[str($g)] = $c;
                push($glyphs, $g);
                $g = $g + 1;
            }
            $i = $i + 1;
        }
        return {"map": $map, "glyphs": $glyphs};
    }
    return {"map": $map, "glyphs": $glyphs};
}


// ════════════════════════════════════════════════════════════════════════
//  Glyph outlines
// ════════════════════════════════════════════════════════════════════════
//
//  A glyph is a list of CONTOURS in font units; each contour is a flat list
//  [x, y, on, x, y, on, ...] because a list of three-element lists costs a
//  copy per point every time it crosses a boundary. `on` is 1 for an
//  on-curve point and 0 for a quadratic control point.
//
//  Composite glyphs are decomposed here rather than at raster time, so the
//  rasteriser never has to know they exist. ARGS_ARE_XY_VALUES, the 2x2
//  transform in all four of its forms, and point matching are all handled;
//  the recursion depth is capped, because a composite that refers to itself
//  is a real file that exists.

def _glyphSpan($f, $gid) {
    if ($gid < 0 || $gid >= $f.numGlyphs) { return [0, 0]; }
    $a = pxbuf.scalar(nd_get($f.loca, [$gid]));
    $b = pxbuf.scalar(nd_get($f.loca, [$gid + 1]));
    if ($b <= $a) { return [0, 0]; }
    $off = $f.tables.glyf[0] + $a;
    $nbytes = $b - $a;
    if ($nbytes > $f.tables.glyf[1]) { $nbytes = $f.tables.glyf[1]; }
    return [$off, $nbytes];
}

// -> a list of flat contours, in font units, already transformed.
def _glyphContours($f, $gid, $depth) {
    if ($depth > 5) {
        throw "ImageFont: glyph " + str($gid) + " nests composites more than five deep";
    }
    $sp = _glyphSpan($f, $gid);
    if ($sp[1] == 0) { return []; }
    $G = nd_to_list(nd_slice($f.data, [[$sp[0], $sp[0] + $sp[1]]]));
    if (len($G) < 10) { return []; }
    $nc = _s16($G[0] * 256 + $G[1]);

    if ($nc < 0) {
        // ── a composite ──
        // Collect every component BEFORE recursing: the recursion re-slices
        // the font and this window would otherwise be stale.
        $comps = [];
        $p = 10;
        $more = true;
        while ($more && $p + 4 <= len($G)) {
            $flags = $G[$p] * 256 + $G[$p + 1];
            $child = $G[$p + 2] * 256 + $G[$p + 3];
            $p = $p + 4;
            $words = floor($flags / 1) % 2 == 1;                 // ARG_1_AND_2_ARE_WORDS
            $xy = floor($flags / 2) % 2 == 1;                    // ARGS_ARE_XY_VALUES
            if ($words) {
                if ($p + 4 > len($G)) { break; }
                $a1 = _s16($G[$p] * 256 + $G[$p + 1]);
                $a2 = _s16($G[$p + 2] * 256 + $G[$p + 3]);
                $p = $p + 4;
            } else {
                if ($p + 2 > len($G)) { break; }
                $a1 = $G[$p];
                $a2 = $G[$p + 1];
                if ($xy) { $a1 = _s8($a1); $a2 = _s8($a2); }
                $p = $p + 2;
            }
            $m = [1, 0, 0, 1];
            if (floor($flags / 8) % 2 == 1) {                    // WE_HAVE_A_SCALE
                if ($p + 2 > len($G)) { break; }
                $sc = _s16($G[$p] * 256 + $G[$p + 1]) / 16384.0;
                $p = $p + 2;
                $m = [$sc, 0, 0, $sc];
            } else if (floor($flags / 64) % 2 == 1) {            // X_AND_Y_SCALE
                if ($p + 4 > len($G)) { break; }
                $m = [_s16($G[$p] * 256 + $G[$p + 1]) / 16384.0, 0, 0,
                      _s16($G[$p + 2] * 256 + $G[$p + 3]) / 16384.0];
                $p = $p + 4;
            } else if (floor($flags / 128) % 2 == 1) {           // TWO_BY_TWO
                if ($p + 8 > len($G)) { break; }
                $m = [_s16($G[$p] * 256 + $G[$p + 1]) / 16384.0,
                      _s16($G[$p + 2] * 256 + $G[$p + 3]) / 16384.0,
                      _s16($G[$p + 4] * 256 + $G[$p + 5]) / 16384.0,
                      _s16($G[$p + 6] * 256 + $G[$p + 7]) / 16384.0];
                $p = $p + 8;
            }
            push($comps, [$child, $a1, $a2, $xy, $m[0], $m[1], $m[2], $m[3]]);
            $more = floor($flags / 32) % 2 == 1;                 // MORE_COMPONENTS
        }
        $out = [];
        $ci = 0;
        while ($ci < len($comps)) {
            $c = $comps[$ci];
            $sub = _glyphContours($f, $c[0], $depth + 1);
            $dx = $c[1];
            $dy = $c[2];
            if (!$c[3]) {
                // POINT MATCHING: the offset is "put the child's point a2 on
                // top of the parent's point a1". Resolving it needs both
                // point lists, so the parent's are gathered first.
                $parentPts = _flatPoints($out);
                $childPts = _flatPoints($sub);
                if ($c[1] * 2 + 1 < len($parentPts) && $c[2] * 2 + 1 < len($childPts)) {
                    $dx = $parentPts[$c[1] * 2] - $childPts[$c[2] * 2];
                    $dy = $parentPts[$c[1] * 2 + 1] - $childPts[$c[2] * 2 + 1];
                } else { $dx = 0; $dy = 0; }
            }
            $k = 0;
            while ($k < len($sub)) {
                $ct = $sub[$k];
                $nct = [];
                $j = 0;
                while ($j + 2 < len($ct) + 1) {
                    $x = $ct[$j];
                    $y = $ct[$j + 1];
                    push($nct, $c[4] * $x + $c[6] * $y + $dx);
                    push($nct, $c[5] * $x + $c[7] * $y + $dy);
                    push($nct, $ct[$j + 2]);
                    $j = $j + 3;
                }
                push($out, $nct);
                $k = $k + 1;
            }
            $ci = $ci + 1;
        }
        return $out;
    }

    // ── a simple glyph ──
    if ($nc == 0) { return []; }
    if (10 + 2 * $nc > len($G)) { return []; }
    $ends = [];
    $i = 0;
    while ($i < $nc) {
        push($ends, $G[10 + 2 * $i] * 256 + $G[10 + 2 * $i + 1]);
        $i = $i + 1;
    }
    $npts = $ends[$nc - 1] + 1;
    if ($npts <= 0 || $npts > 20000) { return []; }
    $p = 10 + 2 * $nc;
    if ($p + 2 > len($G)) { return []; }
    $p = $p + 2 + ($G[$p] * 256 + $G[$p + 1]);          // skip the instructions
    $flags = [];
    while (len($flags) < $npts && $p < len($G)) {
        $fl = $G[$p];
        $p = $p + 1;
        push($flags, $fl);
        if (floor($fl / 8) % 2 == 1) {                   // REPEAT
            if ($p >= len($G)) { break; }
            $r = $G[$p];
            $p = $p + 1;
            while ($r > 0 && len($flags) < $npts) { push($flags, $fl); $r = $r - 1; }
        }
    }
    while (len($flags) < $npts) { push($flags, 0); }
    $xs = [];
    $x = 0;
    $i = 0;
    while ($i < $npts) {
        $fl = $flags[$i];
        if (floor($fl / 2) % 2 == 1) {                   // X_SHORT
            if ($p < len($G)) {
                if (floor($fl / 16) % 2 == 1) { $x = $x + $G[$p]; } else { $x = $x - $G[$p]; }
                $p = $p + 1;
            }
        } else if (floor($fl / 16) % 2 == 0) {
            if ($p + 2 <= len($G)) {
                $x = $x + _s16($G[$p] * 256 + $G[$p + 1]);
                $p = $p + 2;
            }
        }
        push($xs, $x);
        $i = $i + 1;
    }
    $ys = [];
    $y = 0;
    $i = 0;
    while ($i < $npts) {
        $fl = $flags[$i];
        if (floor($fl / 4) % 2 == 1) {                   // Y_SHORT
            if ($p < len($G)) {
                if (floor($fl / 32) % 2 == 1) { $y = $y + $G[$p]; } else { $y = $y - $G[$p]; }
                $p = $p + 1;
            }
        } else if (floor($fl / 32) % 2 == 0) {
            if ($p + 2 <= len($G)) {
                $y = $y + _s16($G[$p] * 256 + $G[$p + 1]);
                $p = $p + 2;
            }
        }
        push($ys, $y);
        $i = $i + 1;
    }
    $out = [];
    $start = 0;
    each ($end in $ends) {
        if ($end < $start || $end >= $npts) { $start = $end + 1; continue; }
        $ct = [];
        $i = $start;
        while ($i <= $end) {
            push($ct, $xs[$i]);
            push($ct, $ys[$i]);
            push($ct, $flags[$i] % 2);
            $i = $i + 1;
        }
        push($out, $ct);
        $start = $end + 1;
    }
    return $out;
}

// Every point of a contour list, flat, for point matching.
def _flatPoints($contours) {
    $out = [];
    $i = 0;
    while ($i < len($contours)) {
        $ct = $contours[$i];
        $j = 0;
        while ($j + 1 < len($ct)) { push($out, $ct[$j]); push($out, $ct[$j + 1]); $j = $j + 3; }
        $i = $i + 1;
    }
    return $out;
}


// ════════════════════════════════════════════════════════════════════════
//  The glyph rasteriser
// ════════════════════════════════════════════════════════════════════════
//
//  NON-ZERO WINDING, because that is TrueType's fill rule and not the
//  even-odd rule draw.b's polygon filler uses. A counter — the hole in an
//  `o` — is wound the opposite way to its outer contour, so the two rules
//  agree on a well-formed glyph and disagree the moment two contours of the
//  same direction overlap, which plenty of real fonts do.
//
//  COVERAGE, not a hit test: sixteen sub-scanlines per pixel row, and within
//  each the span's fractional ends contribute their fraction. That is
//  1/16-of-a-row vertical quantisation and exact horizontal coverage, which
//  is a different approximation from FreeType's exact-area rasteriser and so
//  a different anti-aliasing. It cannot be otherwise — see this file's
//  header — and the difference is measured rather than asserted away.
//
//  Per glyph the cost is proportional to the glyph's AREA, which §8.2 of the
//  architecture allows and rule PW2 does not forbid: a glyph is small, and
//  the result is cached by (font, size, glyph) so a repeated character is a
//  dictionary hit.

// SIXTEEN, and measured rather than chosen: 32 was tried and moved the
// inked-region agreement with Pillow by 0.0003, because the residual is
// FreeType's GRID FITTING -- which moves a stem onto the pixel grid, a whole
// pixel at a time -- and not this rasteriser's vertical quantisation. Thirty-
// two sub-rows cost twice as much for nothing.
$_SS = 16;                                       // sub-scanlines per pixel row

// _glyphMask($f, $gid) -> {"px": an (h, w, 1) i64 coverage buffer 0..255,
//                          "w", "h", "bx", "ty", "adv26"}
// bx is the left edge in pixels and ty the TOP edge, both relative to the
// pen on the baseline, y up.
def _glyphMask($f, $gid) {
    $key = str($gid);
    if ($f.masks[$key] != null) { return $f.masks[$key]; }
    $contours = _glyphContours($f, $gid, 0);

    // Scale to 26.6 and flatten the quadratics into polylines.
    $edges = [];                                  // flat x0,y0,x1,y1 in pixels
    $xmin26 = 0;
    $ymin26 = 0;
    $xmax26 = 0;
    $ymax26 = 0;
    $have = false;
    $ci = 0;
    while ($ci < len($contours)) {
        $ct = $contours[$ci];
        $n = floor(len($ct) / 3);
        if ($n < 2) { $ci = $ci + 1; continue; }
        // scaled points
        $px = [];
        $py = [];
        $on = [];
        $i = 0;
        while ($i < $n) {
            $sx = _mulfix($ct[$i * 3], $f.xscale);
            $sy = _mulfix($ct[$i * 3 + 1], $f.yscale);
            push($px, $sx);
            push($py, $sy);
            push($on, $ct[$i * 3 + 2]);
            if (!$have) {
                $xmin26 = $sx; $xmax26 = $sx; $ymin26 = $sy; $ymax26 = $sy;
                $have = true;
            } else {
                if ($sx < $xmin26) { $xmin26 = $sx; }
                if ($sx > $xmax26) { $xmax26 = $sx; }
                if ($sy < $ymin26) { $ymin26 = $sy; }
                if ($sy > $ymax26) { $ymax26 = $sy; }
            }
            $i = $i + 1;
        }
        push($edges, _contourEdges($px, $py, $on, $n));
        $ci = $ci + 1;
    }

    $bx = _ftFloor($xmin26);
    $ty = _ftCeil($ymax26);
    $w = _ftCeil($xmax26) - $bx;
    $h = $ty - _ftFloor($ymin26);
    if (!$have || $w <= 0 || $h <= 0) {
        $m = {"px": null, "w": 0, "h": 0, "bx": 0, "ty": 0};
        $f.masks[$key] = $m;
        return $f.masks[$key];
    }
    if ($w > 4096 || $h > 4096) {
        throw "ImageFont: glyph " + str($gid) + " rasterises to " + str($w) + "x" +
              str($h) + " pixels, which is past this build's 4096 limit";
    }

    // One flat edge list in PIXEL coordinates (y up), plus a per-row bucket
    // so a sub-scanline only looks at the edges that cross it.
    $E = [];
    $k = 0;
    while ($k < len($edges)) {
        $seg = $edges[$k];
        $j = 0;
        while ($j + 3 < len($seg)) {
            push($E, $seg[$j] / 64.0);
            push($E, $seg[$j + 1] / 64.0);
            push($E, $seg[$j + 2] / 64.0);
            push($E, $seg[$j + 3] / 64.0);
            $j = $j + 4;
        }
        $k = $k + 1;
    }
    $ne = floor(len($E) / 4);
    // An ACTIVE EDGE TABLE, as one sorted flat list of (row, edge) keys
    // rather than a list of per-row lists: `push($buckets[$r], $i)` pushes
    // into a COPY of the element and is lost (PW14). Rows are then walked in
    // order and the list is consumed in a single pass.
    if ($ne > 900000) {
        throw "ImageFont: glyph " + str($gid) + " flattens to " + str($ne) + " segments";
    }
    $pairs = [];
    $i = 0;
    while ($i < $ne) {
        $y0 = $E[$i * 4 + 1];
        $y1 = $E[$i * 4 + 3];
        if ($y0 != $y1) {
            $lo = $y0;
            $hi = $y1;
            if ($lo > $hi) { $lo = $y1; $hi = $y0; }
            // row r covers y in (ty - r - 1, ty - r]
            $ra = floor($ty - $hi);
            $rb = ceil($ty - $lo);
            if ($ra < 0) { $ra = 0; }
            if ($rb > $h) { $rb = $h; }
            $r = $ra;
            while ($r < $rb) { push($pairs, $r * 1000000 + $i); $r = $r + 1; }
        }
        $i = $i + 1;
    }
    $pairs = sort($pairs);
    $np = len($pairs);
    $pi = 0;

    $rows = [];
    $wsub = 1.0 / $_SS;
    $r = 0;
    while ($r < $h) {
        $acc = [];
        $c = 0;
        while ($c < $w) { push($acc, 0.0); $c = $c + 1; }
        $bk = [];
        while ($pi < $np && floor($pairs[$pi] / 1000000) == $r) {
            push($bk, $pairs[$pi] % 1000000);
            $pi = $pi + 1;
        }
        $nb = len($bk);
        if ($nb > 0) {
            $sub = 0;
            while ($sub < $_SS) {
                $sy = $ty - $r - ($sub + 0.5) * $wsub;
                $xs = [];
                $dirs = [];
                $q = 0;
                while ($q < $nb) {
                    $e = $bk[$q] * 4;
                    $y0 = $E[$e + 1];
                    $y1 = $E[$e + 3];
                    $lo = $y0;
                    $hi = $y1;
                    $d = 1;
                    if ($lo > $hi) { $lo = $y1; $hi = $y0; $d = 0 - 1; }
                    if ($sy >= $lo && $sy < $hi) {
                        $x0 = $E[$e];
                        $x1 = $E[$e + 2];
                        push($xs, $x0 + ($sy - $y0) * ($x1 - $x0) / ($y1 - $y0));
                        push($dirs, $d);
                    }
                    $q = $q + 1;
                }
                $m = len($xs);
                if ($m > 1) {
                    // sort the crossings, carrying their winding direction
                    $cross = _sortWith($xs, $dirs);
                    $wind = 0;
                    $spanStart = 0.0;
                    $q = 0;
                    while ($q < $m) {
                        $prev = $wind;
                        $wind = $wind + $cross[$q * 2 + 1];
                        if ($prev == 0 && $wind != 0) { $spanStart = $cross[$q * 2]; }
                        else if ($prev != 0 && $wind == 0) {
                            // _addSpan, inlined: $acc is a list, so a helper
                            // would accumulate into a copy (PW14). Whole
                            // pixels take the full sub-row weight and the two
                            // fractional ends take their fraction of it.
                            $xa = $spanStart - $bx;
                            $xb = $cross[$q * 2] - $bx;
                            if ($xa < 0) { $xa = 0; }
                            if ($xb > $w) { $xb = $w; }
                            if ($xb > $xa) {
                                $ia = floor($xa);
                                $ib = floor($xb);
                                if ($ib >= $w) { $ib = $w - 1; }
                                if ($ia == $ib) {
                                    $acc[$ia] = $acc[$ia] + ($xb - $xa) * $wsub;
                                } else {
                                    $acc[$ia] = $acc[$ia] + ($ia + 1 - $xa) * $wsub;
                                    $kk = $ia + 1;
                                    while ($kk < $ib) {
                                        $acc[$kk] = $acc[$kk] + $wsub;
                                        $kk = $kk + 1;
                                    }
                                    $acc[$ib] = $acc[$ib] + ($xb - $ib) * $wsub;
                                }
                            }
                        }
                        $q = $q + 1;
                    }
                }
                $sub = $sub + 1;
            }
        }
        $c = 0;
        while ($c < $w) {
            $v = floor($acc[$c] * 255.0 + 0.5);
            if ($v < 0) { $v = 0; }
            if ($v > 255) { $v = 255; }
            push($rows, $v);
            $c = $c + 1;
        }
        $r = $r + 1;
    }
    $m = {"px": nd_reshape(nd($rows, "i64"), [$h, $w, 1]),
          "w": $w, "h": $h, "bx": $bx, "ty": $ty};
    $f.masks[$key] = $m;
    return $f.masks[$key];
}

// One contour's line segments, in 26.6, with its quadratics flattened and
// the closing segment implied. Returns a flat x0,y0,x1,y1 list.
def _contourEdges($px, $py, $on, $n) {
    // Find the first on-curve point; a contour of nothing but controls has
    // its curve passing through the midpoints, cyclically.
    $first = 0 - 1;
    $i = 0;
    while ($i < $n) { if ($on[$i] == 1) { $first = $i; break; } $i = $i + 1; }
    $sx = 0;
    $sy = 0;
    if ($first < 0) {
        $sx = floor(($px[$n - 1] + $px[0]) / 2);
        $sy = floor(($py[$n - 1] + $py[0]) / 2);
    } else {
        $sx = $px[$first];
        $sy = $py[$first];
    }
    $out = [];
    $cx = $sx;
    $cy = $sy;
    $pendX = null;
    $pendY = null;
    $k = 1;
    $limit = $n;
    if ($first < 0) { $limit = $n + 1; $first = $n - 1; }
    while ($k <= $limit) {
        $j = ($first + $k) % $n;
        $x = $px[$j];
        $y = $py[$j];
        if ($on[$j] == 1) {
            if ($pendX == null) {
                push($out, $cx); push($out, $cy); push($out, $x); push($out, $y);
                $cx = $x; $cy = $y;
            } else {
                each ($v in _quad($cx, $cy, $pendX, $pendY, $x, $y)) { push($out, $v); }
                $cx = $x; $cy = $y;
                $pendX = null;
            }
        } else {
            if ($pendX != null) {
                // two controls in a row: an implied on-curve point between
                $mx = floor(($pendX + $x) / 2);
                $my = floor(($pendY + $y) / 2);
                each ($v in _quad($cx, $cy, $pendX, $pendY, $mx, $my)) { push($out, $v); }
                $cx = $mx; $cy = $my;
            }
            $pendX = $x;
            $pendY = $y;
        }
        $k = $k + 1;
    }
    if ($pendX != null) {
        each ($v in _quad($cx, $cy, $pendX, $pendY, $sx, $sy)) { push($out, $v); }
    } else if ($cx != $sx || $cy != $sy) {
        push($out, $cx); push($out, $cy); push($out, $sx); push($out, $sy);
    }
    return $out;
}

// A quadratic, flattened by its own flatness: the deviation of the control
// point from the chord decides the step count, so a nearly straight curve
// costs two segments and a tight one costs up to thirty-two.
// RETURNS its segments rather than pushing into the caller's list, because
// a list handed to a function in Bantu is a copy and the pushes would be
// thrown away (PW14). Every piece of mutable state in this rasteriser is
// either returned or a local for that reason.
def _quad($x0, $y0, $cx, $cy, $x1, $y1) {
    $dx = $x0 - 2 * $cx + $x1;
    $dy = $y0 - 2 * $cy + $y1;
    $dev = sqrt($dx * $dx + $dy * $dy) / 64.0;
    $steps = 2 + floor(sqrt($dev * 3));
    if ($steps > 32) { $steps = 32; }
    $out = [];
    $px = $x0;
    $py = $y0;
    $i = 1;
    while ($i <= $steps) {
        $t = $i / $steps;
        $u = 1.0 - $t;
        $qx = $u * $u * $x0 + 2 * $u * $t * $cx + $t * $t * $x1;
        $qy = $u * $u * $y0 + 2 * $u * $t * $cy + $t * $t * $y1;
        push($out, $px); push($out, $py); push($out, $qx); push($out, $qy);
        $px = $qx;
        $py = $qy;
        $i = $i + 1;
    }
    return $out;
}

// Sort the crossings by x, keeping each one's winding direction beside it.
// Returns a flat [x, dir, x, dir, ...]. An insertion sort, because a
// sub-scanline crosses a handful of edges and sort() cannot carry a payload.
def _sortWith($xs, $dirs) {
    $n = len($xs);
    $out = [];
    $i = 0;
    while ($i < $n) { push($out, $xs[$i]); push($out, $dirs[$i]); $i = $i + 1; }
    $i = 1;
    while ($i < $n) {
        $x = $out[$i * 2];
        $d = $out[$i * 2 + 1];
        $j = $i - 1;
        while ($j >= 0 && $out[$j * 2] > $x) {
            $out[($j + 1) * 2] = $out[$j * 2];
            $out[($j + 1) * 2 + 1] = $out[$j * 2 + 1];
            $j = $j - 1;
        }
        $out[($j + 1) * 2] = $x;
        $out[($j + 1) * 2 + 1] = $d;
        $i = $i + 1;
    }
    return $out;
}

// ════════════════════════════════════════════════════════════════════════
//  Layout
// ════════════════════════════════════════════════════════════════════════
//
//  One pass over the string, producing a flat
//  [gid, x_advance26, x_offset26, y_offset26] per character.
//
//  KERNING. Pillow's default layout engine is libraqm, which asks HarfBuzz,
//  which applies the GPOS 'kern' feature's pair adjustments. So GPOS wins and
//  the legacy `kern` table is the fallback, and the adjustment lands on the
//  FIRST glyph of the pair — HarfBuzz's XAdvance applies to glyph one, not
//  glyph two. Scaled with FT_MulFix, which is what makes the result exact:
//  DejaVu's A/V pair is -131 font units, and at 24 px FT_MulFix(-131, 49152)
//  is -98 twenty-sixths-of-a-pixel, which is precisely the 1.53 px by which
//  Pillow's "AV" is narrower than the sum of its two advances.
//
//  WHAT IT DOES NOT DO is the rest of OpenType shaping: no GSUB, so no
//  ligature substitution, no marks, no reordering, no bidi. DejaVu's "fi"
//  ligature is the visible consequence and it is one 64th of a pixel wide:
//  the deviation is measured in tests/pillow_font_test.b rather than
//  described. `direction`, `features` and `language` therefore raise instead
//  of being accepted and ignored, which is what Pillow does without libraqm.
def _layout($f, $text, $direction, $features, $language) {
    if ($direction != null && $direction != "" && $direction != "ltr") {
        throw "ImageFont: direction \"" + str($direction) + "\" needs a shaping " +
              "engine; this build lays out left to right only";
    }
    if ($features != null) {
        throw "ImageFont: OpenType feature selection needs a shaping engine; " +
              "this build applies the 'kern' feature and nothing else";
    }
    if ($language != null && $language != "") {
        throw "ImageFont: language-specific shaping needs a shaping engine";
    }
    $gids = _gids($f, $text);
    $n = len($gids);
    $out = [];
    if ($n == 0) { return $out; }
    $advs = nd_to_list(nd_take($f.adv, nd($gids, "i64")));
    $i = 0;
    while ($i < $n) {
        push($out, $gids[$i]);
        push($out, _mulfix($advs[$i], $f.xscale));
        push($out, 0);
        push($out, 0);
        $i = $i + 1;
    }
    $i = 1;
    while ($i < $n) {
        $key = str($gids[$i - 1]) + ":" + str($gids[$i]);
        $v = $f.gpos[$key];
        if ($v == null) { $v = $f.kern[$key]; }
        if ($v != null && $v != 0) {
            $out[($i - 1) * 4 + 1] = $out[($i - 1) * 4 + 1] +
                                     _emScale($v, $f.size64, $f.upem);
        }
        $i = $i + 1;
    }
    return $out;
}

// _boxAndAnchor -> [width, height, x_offset, y_offset, xmin, ymax, position26]
//
// `bounding_box_and_anchors` from _imagingft.c, with our own unhinted glyph
// control boxes in place of FreeType's hinted ones. Every other step is
// Pillow's: the pen line from 0 to the advanced position is part of the box,
// which is why (0, 0) is always inside it.
def _boxAndAnchor($f, $lay, $anchor) {
    $n = floor(len($lay) / 4);
    $a = $anchor;
    if ($a == null || $a == "") { $a = "la"; }
    if (len($a) != 2) { throw "ImageFont: bad anchor specified: " + str($a); }
    $position = 0;
    $xmin = 0;
    $xmax = 0;
    $ymin = 0;
    $ymax = 0;
    $i = 0;
    while ($i < $n) {
        $px = _pixel($position + $lay[$i * 4 + 2]);
        $py = _pixel($lay[$i * 4 + 3]);
        $position = $position + $lay[$i * 4 + 1];
        $adv = _pixel($position);
        if ($adv > $xmax) { $xmax = $adv; }
        $m = _glyphMask($f, $lay[$i * 4]);
        if ($m.w > 0) {
            $bxmax = $m.bx + $m.w + $px;
            $bxmin = $m.bx + $px;
            $bymax = $m.ty + $py;
            $bymin = $m.ty - $m.h + $py;
            if ($bxmax > $xmax) { $xmax = $bxmax; }
            if ($bxmin < $xmin) { $xmin = $bxmin; }
            if ($bymax > $ymax) { $ymax = $bymax; }
            if ($bymin < $ymin) { $ymin = $bymin; }
        }
        $i = $i + 1;
    }
    $xa = 0;
    $ya = 0;
    if ($n > 0) {
        $h0 = $a[0];
        if ($h0 == "l") { $xa = 0; }
        else if ($h0 == "m") { $xa = _pixel(floor($position / 2)); }
        else if ($h0 == "r") { $xa = _pixel($position); }
        else { throw "ImageFont: bad anchor specified: " + str($a); }
        $v0 = $a[1];
        if ($v0 == "a") { $ya = _pixel($f.ascender); }
        else if ($v0 == "t") { $ya = $ymax; }
        else if ($v0 == "m") { $ya = _pixel(floor(($f.ascender + $f.descender) / 2)); }
        else if ($v0 == "s") { $ya = 0; }
        else if ($v0 == "b") { $ya = $ymin; }
        else if ($v0 == "d") { $ya = _pixel($f.descender); }
        else { throw "ImageFont: bad anchor specified: " + str($a); }
    }
    return [$xmax - $xmin, $ymax - $ymin, 0 - $xa + $xmin,
            0 - (0 - $ya + $ymax), $xmin, $ymax, $position];
}


// ════════════════════════════════════════════════════════════════════════
//  The public surface
// ════════════════════════════════════════════════════════════════════════

// _pilGlyphs($f) -> the 256 x 10 metric table as one flat local list. ONE
// array call; indexing it afterwards is free, where `$f.glyphs[$i]` would
// copy 2,560 numbers per character (PW14).
def _pilGlyphs($f) { return nd_to_list(nd_ravel($f.glyphs)); }

// A PILfont addresses its glyphs by Latin-1 byte, as Pillow's
// _font_text_asBytes does.
def _latin1($text, $who) {
    $out = [];
    each ($c in _codepoints($text)) {
        if ($c > 255) {
            throw $who + ": a bitmap font addresses its glyphs by Latin-1 byte, " +
                  "and U+" + str($c) + " has none. Use truetype() for other scripts.";
        }
        push($out, $c);
    }
    return $out;
}

def _pilWidth($f, $text) {
    $G = _pilGlyphs($f);
    $w = 0;
    each ($c in _latin1($text, "ImageFont")) { $w = $w + $G[$c * 10]; }
    if ($w < 0) { return 0; }
    return $w;
}

// getlength($f, $text, $mode, $direction, $features, $language) -> pixels.
//
// A TrueType length is in 1/64ths, as Pillow's is: it returns the 26.6 sum
// divided by 64, so it is a float and `text` placed at that offset abuts
// exactly. A PILfont length is a whole number of pixels.
def getlength($f, $text, $mode, $direction, $features, $language) {
    if ($f.kind == "pil") { return _pilWidth($f, $text); }
    $lay = _layout($f, $text, $direction, $features, $language);
    $n = floor(len($lay) / 4);
    $sum = 0;
    $i = 0;
    while ($i < $n) { $sum = $sum + $lay[$i * 4 + 1]; $i = $i + 1; }
    return $sum / 64.0;
}

// getbbox($f, $text, $mode, $direction, $features, $language, $anchor,
//         $stroke_width) -> [left, top, right, bottom] around (0, 0).
def getbbox($f, $text, $mode, $direction, $features, $language, $anchor, $stroke_width) {
    $sw = $stroke_width;
    if ($sw == null) { $sw = 0; }
    if ($f.kind == "pil") {
        // The anchor and the stroke are IGNORED, not refused: Pillow's
        // ImageFont.getbbox takes `*args` and drops them, so a bitmap font's
        // box is always its own width by its own height and an anchor has no
        // effect on a single line of it.
        return [0, 0, _pilWidth($f, $text), $f.ysize];
    }
    $lay = _layout($f, $text, $direction, $features, $language);
    $b = _boxAndAnchor($f, $lay, $anchor);
    $l = $b[2] - $sw;
    $t = $b[3] - $sw;
    return [$l, $t, $l + $b[0] + 2 * $sw, $t + $b[1] + 2 * $sw];
}

// getmetrics($f) -> [ascent, descent], the grid-fitted face metrics
// FreeType reports. The descent is POSITIVE, as Pillow returns it negated.
def getmetrics($f) {
    if ($f.kind == "pil") { return [$f.ysize - $f.baseline, $f.baseline]; }
    return [_pixel($f.ascender), 0 - _pixel($f.descender)];
}

// font_variant($f, $size, $index, $encoding, $layout_engine) -> a copy with
// the given settings overridden, exactly as Pillow's does. It is NOT the
// OpenType variation API; see set_variation_by_name.
def font_variant($f, $size, $index, $encoding, $layout_engine) {
    if ($f.kind == "pil") {
        if ($size != null) {
            throw "ImageFont.font_variant: a bitmap font has one size";
        }
        return $f;
    }
    $sz = $size;
    if ($sz == null) { $sz = $f.size; }
    $ix = $index;
    if ($ix == null) { $ix = $f.index; }
    $en = $encoding;
    if ($en == null) { $en = $f.encoding; }
    $le = $layout_engine;
    if ($le == null) { $le = $f.layout_engine; }
    return truetype($f.path, $sz, $ix, $en, $le);
}

// The OpenType variation API. `fvar` is read, so a caller can see what the
// font offers; APPLYING an instance needs `gvar` point-delta interpolation,
// which is a second outline format's worth of work and is not in this build.
// Declined by name rather than silently returning the default instance.
def get_variation_axes($f) {
    if ($f.kind != "tt" || $f.tables.fvar == null) {
        throw "ImageFont: this font has no variation axes";
    }
    $off = $f.tables.fvar[0];
    $H = nd_to_list(nd_slice($f.data, [[$off, $off + 16]]));
    $axisOff = $off + $H[4] * 256 + $H[5];
    $axisCount = $H[8] * 256 + $H[9];
    $axisSize = $H[6] * 256 + $H[7];
    $out = [];
    $i = 0;
    while ($i < $axisCount && $i < 64) {
        $A = nd_to_list(nd_slice($f.data, [[$axisOff + $i * $axisSize,
                                            $axisOff + $i * $axisSize + 20]]));
        $tag = join([chr($A[0]), chr($A[1]), chr($A[2]), chr($A[3])], "");
        push($out, {"tag": $tag,
                    "minimum": _fixed($A, 4), "default": _fixed($A, 8),
                    "maximum": _fixed($A, 12)});
        $i = $i + 1;
    }
    return $out;
}

def _fixed($A, $o) {
    $v = $A[$o] * 16777216 + $A[$o + 1] * 65536 + $A[$o + 2] * 256 + $A[$o + 3];
    if ($v >= 2147483648) { $v = $v - 4294967296; }
    return $v / 65536.0;
}

def set_variation_by_name($f, $name) {
    throw "ImageFont.set_variation_by_name: applying a variation instance needs " +
          "'gvar' point-delta interpolation, which is not in this build. " +
          "get_variation_axes() reports what the font offers.";
}
def set_variation_by_axes($f, $axes) { return set_variation_by_name($f, null); }

// getmask($f, $text, $mode, ...) -> a BUFFER, not an Image: Pillow's getmask
// returns an ImagingCore for the same reason, and ImageDraw.bitmap takes it.
// {"mode", "size", "width", "height", "px"}.
def getmask($f, $text, $mode, $direction, $features, $language, $stroke_width) {
    return getmask2($f, $text, $mode, $direction, $features, $language,
                    null, $stroke_width)[0];
}

// getmask2(...) -> [buffer, [x_offset, y_offset]].
def getmask2($f, $text, $mode, $direction, $features, $language, $anchor, $stroke_width) {
    $m = $mode;
    if ($m == null || $m == "") { $m = "L"; }
    if ($m == "RGBA") {
        throw "ImageFont: embedded_color needs a colour font (COLR, CBDT or sbix), " +
              "which this build does not read";
    }
    if ($m != "1" && $m != "L") {
        throw "ImageFont: a glyph mask is mode \"1\" or \"L\", not " + str($m);
    }
    $sw = $stroke_width;
    if ($sw == null) { $sw = 0; }

    if ($f.kind == "pil") {
        // stroke_width is IGNORED, not refused, because Pillow ignores it:
        // ImageFont.getmask takes `*args` and drops them, so a bitmap font's
        // stroke pass draws the plain glyphs in the stroke colour and the
        // normal pass then covers them. A bitmap font has no outline to
        // offset, so there is nothing better to do and matching is worth
        // more than objecting.
        return [_pilMask($f, $text, $m), [0, 0]];
    }

    $lay = _layout($f, $text, $direction, $features, $language);
    $b = _boxAndAnchor($f, $lay, $anchor);
    $w = $b[0] + 2 * $sw;
    $h = $b[1] + 2 * $sw;
    if ($w <= 0 || $h <= 0) {
        return [{"mode": $m, "size": [0, 0], "width": 0, "height": 0,
                 "px": pxbuf.make(0, 0, 1, "i64")}, [$b[2], $b[3]]];
    }
    if ($w > 65536 || $h > 65536) {
        throw "ImageFont: a " + str($w) + "x" + str($h) + " text mask is past this " +
              "build's limit";
    }
    // Coverage accumulates with MAX, not addition: two glyphs that overlap
    // must not add up to a seam brighter than either, which is what
    // FreeType's own compositing avoids by drawing into one bitmap.
    $acc = pxbuf.make($h, $w, 1, "i64");
    $xmin = $b[4];
    $ymax = $b[5];
    $n = floor(len($lay) / 4);
    $position = 0;
    $i = 0;
    while ($i < $n) {
        $px = _pixel($position + $lay[$i * 4 + 2]);
        $py = _pixel($lay[$i * 4 + 3]);
        $position = $position + $lay[$i * 4 + 1];
        $gm = _glyphMask($f, $lay[$i * 4]);
        if ($gm.w > 0) {
            // the glyph's top-left in mask coordinates
            $dx = $gm.bx + $px - $xmin + $sw;
            $dy = $ymax - ($gm.ty + $py) + $sw;
            _blitMax($acc, $gm.px, $dx, $dy, $gm.w, $gm.h, $w, $h);
        }
        $i = $i + 1;
    }
    if ($sw > 0) { $acc = _dilate($acc, $sw, $w, $h); }
    if ($m == "1") { $acc = nd_multiply(nd_greater_equal($acc, 128), 255); }
    return [{"mode": $m, "size": [$w, $h], "width": $w, "height": $h, "px": $acc},
            [$b[2], $b[3]]];
}

// The maximum of a glyph's coverage and what is already there, clipped to
// the mask. One array call per glyph, not one per pixel.
def _blitMax($acc, $src, $dx, $dy, $gw, $gh, $w, $h) {
    $sx0 = 0;
    $sy0 = 0;
    $x0 = $dx;
    $y0 = $dy;
    if ($x0 < 0) { $sx0 = 0 - $x0; $x0 = 0; }
    if ($y0 < 0) { $sy0 = 0 - $y0; $y0 = 0; }
    $cw = $gw - $sx0;
    $ch = $gh - $sy0;
    if ($x0 + $cw > $w) { $cw = $w - $x0; }
    if ($y0 + $ch > $h) { $ch = $h - $y0; }
    if ($cw <= 0 || $ch <= 0) { return null; }
    $dst = pxbuf.crop($acc, $x0, $y0, $x0 + $cw, $y0 + $ch);
    $s = pxbuf.crop($src, $sx0, $sy0, $sx0 + $cw, $sy0 + $ch);
    nd_add(nd_maximum($dst, $s), 0, $dst);
    return null;
}

// stroke_width, as a morphological dilation by a disc.
//
// NOT PILLOW'S ARITHMETIC, and deliberately. Pillow hands the outline to
// FT_Stroker, which offsets the contours and re-rasterises; a stroker is a
// geometry library of its own. A round stroke of radius r IS the Minkowski
// sum of the glyph with a disc of radius r, so dilating the coverage by that
// disc is the same shape, computed on the raster instead of the outline. The
// difference is at the sub-pixel edge, and it is measured in
// tests/pillow_font_test.b rather than claimed away.
//
// The loop is over the disc's OFFSETS -- at most (2r+1)^2 of them, and r is
// a stroke width -- with each offset one whole-array maximum. Never pixels.
def _dilate($acc, $r, $w, $h) {
    $out = pxbuf.make($h, $w, 1, "i64");
    $dy = 0 - $r;
    while ($dy <= $r) {
        $dx = 0 - $r;
        while ($dx <= $r) {
            if ($dx * $dx + $dy * $dy <= $r * $r) {
                $sx0 = 0;
                $sy0 = 0;
                $x0 = $dx;
                $y0 = $dy;
                if ($x0 < 0) { $sx0 = 0 - $x0; $x0 = 0; }
                if ($y0 < 0) { $sy0 = 0 - $y0; $y0 = 0; }
                $cw = $w - $sx0;
                $ch = $h - $sy0;
                if ($x0 + $cw > $w) { $cw = $w - $x0; }
                if ($y0 + $ch > $h) { $ch = $h - $y0; }
                if ($cw > 0 && $ch > 0) {
                    $d = pxbuf.crop($out, $x0, $y0, $x0 + $cw, $y0 + $ch);
                    $s = pxbuf.crop($acc, $sx0, $sy0, $sx0 + $cw, $sy0 + $ch);
                    nd_add(nd_maximum($d, $s), 0, $d);
                }
            }
            $dx = $dx + 1;
        }
        $dy = $dy + 1;
    }
    return $out;
}

// A PILfont's mask: Pillow's _font_getmask, which is pure byte copying.
// `b` is the baseline and it MOVES, by each glyph's dy, which is how a
// PILfont can carry a sloped or stepped baseline.
def _pilMask($f, $text, $mode) {
    $G = _pilGlyphs($f);
    $codes = _latin1($text, "ImageFont");
    $xsize = 0;
    each ($c in $codes) { $xsize = $xsize + $G[$c * 10]; }
    if ($xsize < 0) { $xsize = 0; }
    $ysize = $f.ysize;
    $out = pxbuf.make($ysize, $xsize, 1, "i64");
    $x = 0;
    $b = $f.baseline;
    $i = 0;
    while ($i < len($codes)) {
        $o = $codes[$i] * 10;
        $sx0 = $G[$o + 6];
        $sy0 = $G[$o + 7];
        $sx1 = $G[$o + 8];
        $sy1 = $G[$o + 9];
        $dx0 = $G[$o + 2] + $x;
        $dy0 = $G[$o + 3] + $b;
        if ($sx1 > $sx0 && $sy1 > $sy0) {
            _pasteClipped($out, $f.sheet, $dx0, $dy0, $sx0, $sy0,
                          $sx1 - $sx0, $sy1 - $sy0, $xsize, $ysize);
        }
        $x = $x + $G[$o];
        $b = $b + $G[$o + 1];
        $i = $i + 1;
    }
    // THE MASK'S MODE IS THE SHEET'S, not the one asked for: Pillow's
    // ImageFont.getmask takes a mode and its renderer ignores it, so a
    // bitmap font always hands back the sheet's own mode. The values are the
    // same either way -- the sheet is "1", holding 0 and 255 -- so a caller
    // that wanted "L" is not short-changed; it just gets a stencil.
    return {"mode": $f.sheetMode, "size": [$xsize, $ysize], "width": $xsize,
            "height": $ysize, "px": $out};
}

def _pasteClipped($dst, $src, $dx, $dy, $sx, $sy, $cw, $ch, $w, $h) {
    if ($dx < 0) { $sx = $sx - $dx; $cw = $cw + $dx; $dx = 0; }
    if ($dy < 0) { $sy = $sy - $dy; $ch = $ch + $dy; $dy = 0; }
    if ($dx + $cw > $w) { $cw = $w - $dx; }
    if ($dy + $ch > $h) { $ch = $h - $dy; }
    if ($cw <= 0 || $ch <= 0) { return null; }
    $d = pxbuf.crop($dst, $dx, $dy, $dx + $cw, $dy + $ch);
    $s = pxbuf.crop($src, $sx, $sy, $sx + $cw, $sy + $ch);
    nd_add($s, 0, $d);
    return null;
}

// getname($f) -> [family, style] from the `name` table, as Pillow's
// font.family / font.style report them.
def getname($f) {
    if ($f.kind == "pil") { return ["<default>", null]; }
    if ($f.tables.name == null) { return [null, null]; }
    $off = $f.tables.name[0];
    $H = nd_to_list(nd_slice($f.data, [[$off, $off + 6]]));
    $count = $H[2] * 256 + $H[3];
    $strOff = $off + $H[4] * 256 + $H[5];
    if ($count == 0 || $count > 4096) { return [null, null]; }
    $R = nd_to_list(nd_slice($f.data, [[$off + 6, $off + 6 + 12 * $count]]));
    $family = null;
    $style = null;
    $i = 0;
    while ($i < $count) {
        $o = $i * 12;
        $pid = $R[$o] * 256 + $R[$o + 1];
        $nid = $R[$o + 6] * 256 + $R[$o + 7];
        $nbytes = $R[$o + 8] * 256 + $R[$o + 9];
        $so = $R[$o + 10] * 256 + $R[$o + 11];
        if (($nid == 1 || $nid == 2) && $nbytes > 0 && $nbytes < 512) {
            $S = nd_to_list(nd_slice($f.data, [[$strOff + $so, $strOff + $so + $nbytes]]));
            $txt = [];
            if ($pid == 3 || $pid == 0) {
                $k = 0;
                while ($k + 1 < len($S)) {
                    $cp = $S[$k] * 256 + $S[$k + 1];
                    if ($cp > 0 && $cp < 1114112) { push($txt, _utf8enc($cp)); }
                    $k = $k + 2;
                }
            } else {
                $k = 0;
                while ($k < len($S)) { push($txt, chr($S[$k])); $k = $k + 1; }
            }
            $v = join($txt, "");
            if ($nid == 1 && $family == null) { $family = $v; }
            if ($nid == 2 && $style == null) { $style = $v; }
        }
        $i = $i + 1;
    }
    return [$family, $style];
}
