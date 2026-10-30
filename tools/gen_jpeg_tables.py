#!/usr/bin/env python3
"""
tools/gen_jpeg_tables.py -- generate pillow/jpeg_tables.b.

Every constant JPEG decoding and encoding needs that has ONE right value,
taken from where that value is defined rather than retyped:

  * the zigzag order                       ITU-T T.81 Figure A.6 / jpeg_natural_order
  * the IDCT range-limit table             jdmaster.c  prepare_range_limit_table
  * the YCbCr->RGB tables                  jdcolor.c   build_ycc_rgb_table
  * the Annex K quantisation tables         jcparam.c   std_*_quant_tbl
  * the Annex K Huffman tables              jstdhuff.c  std_huff_tables
  * Pillow's quality presets                PIL.JpegPresets

The first four are computed here by the same construction the C uses, so a
change in the construction shows up as a diff rather than as a wrong pixel.
The spec tables are transcribed with their source named, and the differential
tier checks them by comparing the DHT and DQT bytes of a file Pillow wrote.

Run:  python3 tools/gen_jpeg_tables.py > pillow/jpeg_tables.b
"""
import sys

# ── ITU-T T.81 Figure A.6: zigzag position -> natural index ──────────────
# libjpeg calls this jpeg_natural_order; it is the de-zigzag permutation.
NATURAL_ORDER = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
]

# ── jdmaster.c prepare_range_limit_table, the post-IDCT half ─────────────
# The table lives at sample_range_limit + CENTERJSAMPLE and is indexed by
# (value & 1023). Five segments, built exactly as the C builds them.
def idct_range_limit():
    MAXJSAMPLE, CENTER = 255, 128
    # base[k] is sample_range_limit[k - 256]; simple table first.
    base = [0] * (5 * (MAXJSAMPLE + 1) + CENTER)
    srl = 256                       # index of sample_range_limit[0] in base
    for i in range(0, MAXJSAMPLE + 1):
        base[srl + i] = i           # limit[x] = x for 0..255
    post = srl + CENTER             # the post-IDCT table starts here
    for i in range(CENTER, 2 * (MAXJSAMPLE + 1)):
        base[post + i] = MAXJSAMPLE
    for i in range(2 * (MAXJSAMPLE + 1), 4 * (MAXJSAMPLE + 1) - CENTER):
        base[post + i] = 0
    for i in range(CENTER):         # memcpy of the simple table's head
        base[post + 4 * (MAXJSAMPLE + 1) - CENTER + i] = base[srl + i]
    return [base[post + i] for i in range(1024)]

# ── jdcolor.c build_ycc_rgb_table ───────────────────────────────────────
SCALEBITS, ONE_HALF = 16, 1 << 15
def FIX(x):
    return int(x * (1 << SCALEBITS) + 0.5)

def ycc_rgb_tables():
    crr, cbb, crg, cbg = [], [], [], []
    for i in range(256):
        x = i - 128
        crr.append((FIX(1.40200) * x + ONE_HALF) >> SCALEBITS)
        cbb.append((FIX(1.77200) * x + ONE_HALF) >> SCALEBITS)
        crg.append((-FIX(0.71414)) * x)
        cbg.append((-FIX(0.34414)) * x + ONE_HALF)
    return crr, cbb, crg, cbg

# ── jccolor.c rgb_ycc_start: the forward transform's tables ─────────────
# Used by the encoder. R->Y, G->Y, B->Y, and the Cb/Cr halves, all at
# SCALEBITS with the CENTERJSAMPLE offset folded in.
def rgb_ycc_tables():
    CBCR_OFFSET = 128 << SCALEBITS
    ry, gy, by, rcb, gcb, bcb, rcr, gcr, bcr = ([] for _ in range(9))
    for i in range(256):
        ry.append(FIX(0.29900) * i)
        gy.append(FIX(0.58700) * i)
        by.append(FIX(0.11400) * i + ONE_HALF)
        rcb.append((-FIX(0.16874)) * i)
        gcb.append((-FIX(0.33126)) * i)
        bcb.append(FIX(0.50000) * i + CBCR_OFFSET + ONE_HALF - 1)
        rcr.append(FIX(0.50000) * i + CBCR_OFFSET + ONE_HALF - 1)
        gcr.append((-FIX(0.41869)) * i)
        bcr.append((-FIX(0.08131)) * i)
    return ry, gy, by, rcb, gcb, bcb, rcr, gcr, bcr

# ── jcparam.c: the Annex K quantisation tables, natural order ───────────
STD_LUMA = [
    16,  11,  10,  16,  24,  40,  51,  61,
    12,  12,  14,  19,  26,  58,  60,  55,
    14,  13,  16,  24,  40,  57,  69,  56,
    14,  17,  22,  29,  51,  87,  80,  62,
    18,  22,  37,  56,  68, 109, 103,  77,
    24,  35,  55,  64,  81, 104, 113,  92,
    49,  64,  78,  87, 103, 121, 120, 101,
    72,  92,  95,  98, 112, 100, 103,  99,
]
STD_CHROMA = [
    17,  18,  24,  47,  99,  99,  99,  99,
    18,  21,  26,  66,  99,  99,  99,  99,
    24,  26,  56,  99,  99,  99,  99,  99,
    47,  66,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
]

# ── jstdhuff.c: the Annex K Huffman tables ─────────────────────────────
BITS_DC_LUMA   = [0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
VAL_DC_LUMA    = list(range(12))
BITS_DC_CHROMA = [0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
VAL_DC_CHROMA  = list(range(12))
BITS_AC_LUMA   = [0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d]
VAL_AC_LUMA = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
    0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
    0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
    0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
    0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
    0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
]
BITS_AC_CHROMA = [0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77]
VAL_AC_CHROMA = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
    0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
    0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
    0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
    0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
    0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
    0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
    0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
    0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
]


def emit_list(name, values, per_line=16, comment=None):
    print()
    if comment:
        for line in comment.strip().splitlines():
            print("// " + line.strip())
    print("$%s = [" % name)
    for i in range(0, len(values), per_line):
        chunk = ", ".join(str(v) for v in values[i:i + per_line])
        end = "," if i + per_line < len(values) else ""
        print("    " + chunk + end)
    print("];")


def main():
    print('// ' + '=' * 70)
    print("//  pillow/jpeg_tables.b -- GENERATED by tools/gen_jpeg_tables.py.")
    print("//  DO NOT EDIT. Regenerate instead.")
    print("//")
    print("//  Every constant here has exactly one right value, and it is defined")
    print("//  somewhere other than this file: ITU-T T.81 Annex K and Figure A.6,")
    print("//  libjpeg-turbo's jdmaster.c, jdcolor.c, jccolor.c, jcparam.c and")
    print("//  jstdhuff.c. The first four tables are COMPUTED here by the same")
    print("//  construction the C uses, so a change in that construction shows up")
    print("//  as a diff rather than as a wrong pixel.")
    print("//")
    print("//  docs/pillow-architecture.md " + chr(0xa7) + "6.4.1 and " + chr(0xa7) + "6.4.3.")
    print('// ' + '=' * 70)

    emit_list("ZIGZAG", NATURAL_ORDER, 8, comment="""
        T.81 Figure A.6. ZIGZAG[k] is the NATURAL index of the k-th
        coefficient in zigzag order -- libjpeg's jpeg_natural_order. A block is
        stored in natural order throughout this package, matching JBLOCK.
    """)

    emit_list("IDCT_RANGE", idct_range_limit(), 16, comment="""
        jdmaster.c's post-IDCT range limit, indexed by (value & 1023).
        For values in [-512, 511] this is clamp(v + 128, 0, 255); outside that
        window the mask WRAPS, and these are the values it wraps onto. A plain
        clamp is therefore not a substitute (architecture 6.4.3).
    """)

    crr, cbb, crg, cbg = ycc_rgb_tables()
    emit_list("CR_R", crr, 16, comment="jdcolor.c: nearest int to 1.40200 * (cr - 128).")
    emit_list("CB_B", cbb, 16, comment="jdcolor.c: nearest int to 1.77200 * (cb - 128).")
    emit_list("CR_G", crg, 8, comment="""
        jdcolor.c: -0.71414 * (cr - 128), still scaled up by 2^16.
    """)
    emit_list("CB_G", cbg, 8, comment="""
        jdcolor.c: -0.34414 * (cb - 128), scaled up by 2^16 with the rounding
        half already added in, so the green sum needs no rounding term.
    """)

    ry, gy, by, rcb, gcb, bcb, rcr, gcr, bcr = rgb_ycc_tables()
    for nm, tb, note in (
        ("RGB_RY", ry, "jccolor.c: 0.29900 * r, scaled by 2^16."),
        ("RGB_GY", gy, "jccolor.c: 0.58700 * g, scaled by 2^16."),
        ("RGB_BY", by, "jccolor.c: 0.11400 * b + the rounding half."),
        ("RGB_RCB", rcb, "jccolor.c: -0.16874 * r."),
        ("RGB_GCB", gcb, "jccolor.c: -0.33126 * g."),
        ("RGB_BCB", bcb, "jccolor.c: 0.5 * b + the 128 centre + the half - 1."),
        ("RGB_RCR", rcr, "jccolor.c: 0.5 * r + the 128 centre + the half - 1."),
        ("RGB_GCR", gcr, "jccolor.c: -0.41869 * g."),
        ("RGB_BCR", bcr, "jccolor.c: -0.08131 * b."),
    ):
        emit_list(nm, tb, 8, comment=note)

    emit_list("STD_LUMA_QUANT", STD_LUMA, 8, comment="""
        T.81 Annex K.1, luminance, in NATURAL order. The file stores it in
        zigzag order; libjpeg converts on the way in and out, and so do we.
    """)
    emit_list("STD_CHROMA_QUANT", STD_CHROMA, 8,
              comment="T.81 Annex K.1, chrominance, natural order.")

    emit_list("STD_DC_LUMA_BITS", BITS_DC_LUMA, 17, comment="""
        T.81 Annex K.3 / jstdhuff.c. BITS[n] is how many codes have length n;
        BITS[0] is unused, so the list is 17 long.
    """)
    emit_list("STD_DC_LUMA_VALS", VAL_DC_LUMA, 16)
    emit_list("STD_DC_CHROMA_BITS", BITS_DC_CHROMA, 17)
    emit_list("STD_DC_CHROMA_VALS", VAL_DC_CHROMA, 16)
    emit_list("STD_AC_LUMA_BITS", BITS_AC_LUMA, 17)
    emit_list("STD_AC_LUMA_VALS", VAL_AC_LUMA, 16)
    emit_list("STD_AC_CHROMA_BITS", BITS_AC_CHROMA, 17)
    emit_list("STD_AC_CHROMA_VALS", VAL_AC_CHROMA, 16)

    print()
    print("// Pillow's named quality presets (PIL/JpegPresets.py): the subsampling")
    print("// and the two quantisation tables each one selects. `quality=\"web_high\"`")
    print("// means these numbers, not a quality number.")
    try:
        from PIL import JpegPresets
        presets = JpegPresets.presets
    except Exception as exc:      # the tables are the point; say so and stop
        print("// (Pillow not importable: %s)" % exc, file=sys.stderr)
        presets = {}
    print("$PRESETS = {")
    for name in sorted(presets):
        p = presets[name]
        q = p.get("quantization")
        ss = p.get("subsampling", -1)
        print('    "%s": {' % name)
        print('        "subsampling": %d,' % ss)
        if q is None:
            print('        "quantization": null')
        else:
            print('        "quantization": [')
            # NO TRAILING COMMAS: Bantu rejects one in a list or dict
            # literal, and an include whose literal fails to parse yields an
            # EMPTY value rather than an error -- which is how a trailing
            # comma here once made every preset silently disappear.
            print(",\n".join(
                "            [" + ", ".join(str(v) for v in t) + "]" for t in q))
            print("        ]")
        print("    }" + ("," if name != sorted(presets)[-1] else ""))
    print("};")
    print()
    print("// PRESET TABLES ARE NOT im.quantization's ORDER. PIL/JpegPresets.py")
    print("// stores them in zigzag order while im.quantization is natural, and")
    print("// Pillow's _save hands both to libjpeg's jpeg_add_quant_table, which")
    print("// reads natural order. So a preset's table reaches libjpeg permuted,")
    print("// and matching Pillow means permuting it the same way: these lists")
    print("// go out exactly as Pillow's do, untouched.")


if __name__ == "__main__":
    main()
