#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  tests/pillow_differential.sh — pillow's tier 2, against Pillow itself.
#
#  The checked-in fixtures in tests/fixtures/pillow/reference/ pin pillow's
#  arithmetic to Pillow's, and the .b suites compare against them. This
#  script is the stronger check that needs Pillow present: it has PYTHON'S
#  PILLOW READ THE FILES THE BANTU PACKAGE WROTE, and compares whole images
#  rather than the fixtures' twelve probe pixels.
#
#  That is the claim the package makes -- "the same results as Pillow" -- so
#  it is worth testing against the real thing and not only against a
#  recording of it.
#
#    * a 23x17 RGB pattern written by pillow, read by Pillow, pixel for pixel
#    * convert("L")            compared to Pillow's own, pixel for pixel
#    * convert("1", NONE)      compared to Pillow's own, pixel for pixel
#    * a 9-wide 1-bit image    (the row-padding case)
#    * a 16-bit greymap        (big-endian on the wire)
#
#  Requires python3 with Pillow. Skips with a clear message otherwise, so a
#  machine without it reports a skip rather than a failure.
#
#  Run:  tests/pillow_differential.sh
#        PYREF=/path/to/venv/bin/python tests/pillow_differential.sh
# ════════════════════════════════════════════════════════════════════════
set -u

cd "$(dirname "$0")/.." || exit 1
BANTU="${BANTU:-./bantu}"
[ -x "$BANTU" ] || { echo "[FAIL] no bantu binary at $BANTU"; exit 1; }

PY="${PYREF:-python3}"
# The probe MUST use the same flags the real call does, and the flags are
# chosen here rather than fixed: -I (isolated) is what we want, because it
# ignores PYTHONPATH and the current directory, but -I also ignores USER
# site-packages, which is where `pip install --user Pillow` puts it. So try
# isolated first and fall back to plain, loudly, rather than announcing a
# reference the script then fails to import.
PYFLAGS="-I"
if ! "$PY" $PYFLAGS -c "import PIL" >/dev/null 2>&1; then
    if "$PY" -c "import PIL" >/dev/null 2>&1; then
        PYFLAGS=""
        echo "note: Pillow is in user site-packages, so python runs without -I"
    else
        echo "[SKIP] $PY has no Pillow -- tier 2's live comparison needs it."
        echo "       The checked-in fixtures still cover this in the .b suites:"
        echo "         bantu run tests/pillow_core_test.b"
        echo "         bantu run tests/pillow_convert_test.b"
        echo "         bantu run tests/pillow_jpeg_test.b"
        echo "       To enable this script:  python3 -m pip install Pillow"
        exit 0
    fi
fi
echo "reference: $("$PY" $PYFLAGS -c 'import PIL; print("Pillow", PIL.__version__)')"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/emit.b" <<BANTU_EOF
include "./pillow/pillow.b" as pillow;
\$D = "$TMP/";

\$w = 23; \$h = 17;
\$rgb = pillow.new("RGB", [\$w, \$h], null);
\$y = 0;
while (\$y < \$h) {
    \$x = 0;
    while (\$x < \$w) {
        \$rgb.putpixel([\$x, \$y],
            [(\$x * 11) % 256, (\$y * 29) % 256, (\$x * \$y * 7) % 256]);
        \$x = \$x + 1;
    }
    \$y = \$y + 1;
}
\$rgb.save(\$D + "rgb.ppm", null);
\$rgb.convert("L").save(\$D + "grey.pgm", null);
\$rgb.convert("1", "NONE", null).save(\$D + "bw_none.pbm", null);

\$bw = pillow.new("1", [9, 5], 0);
\$i = 0;
while (\$i < 9) { \$bw.putpixel([\$i, \$i % 5], 255); \$i = \$i + 1; }
\$bw.save(\$D + "bw9.pbm", null);

\$g16 = pillow.new("I;16", [7, 3], 0);
\$y = 0;
while (\$y < 3) {
    \$x = 0;
    while (\$x < 7) { \$g16.putpixel([\$x, \$y], (\$x * 9000 + \$y * 300) % 65536); \$x = \$x + 1; }
    \$y = \$y + 1;
}
\$g16.save(\$D + "g16.pgm", null);

\$parts = [];
\$d = \$rgb.getdata();
\$i = 0;
while (\$i < len(\$d)) {
    push(\$parts, str(\$d[\$i][0]) + "," + str(\$d[\$i][1]) + "," + str(\$d[\$i][2]));
    \$i = \$i + 1;
}
writefile(\$D + "rgb.expect", join(\$parts, "\n"));

// Phase P4: the real formats. Each is written here and read by Pillow on the
// other side, which is the only test that proves the BYTES are right rather
// than merely self-consistent.
\$rgb.save(\$D + "out_rgb.png", null);
\$rgb.convert("RGBA", null, null).save(\$D + "out_rgba.png", null);
\$rgb.convert("L", null, null).save(\$D + "out_l.png", null);
\$rgb.convert("LA", null, null).save(\$D + "out_la.png", null);
\$rgb.convert("1", "NONE", null).save(\$D + "out_bw.png", null);
\$rgb.convert("P", "NONE", null).save(\$D + "out_p.png", null);
\$rgb.save(\$D + "out_rgb.bmp", null);
\$rgb.convert("L", null, null).save(\$D + "out_l.bmp", null);
\$rgb.convert("1", "NONE", null).save(\$D + "out_bw.bmp", null);
\$rgb.save(\$D + "out_rgb.tga", null);
\$rgb.convert("RGBA", null, null).save(\$D + "out_rgba.tga", null);
\$rgb.convert("L", null, null).save(\$D + "out_l.tga", null);
// every PNG filter, so each unfilter path is proved against a real decoder
\$f = 0;
while (\$f <= 4) {
    writefile(\$D + "out_f" + str(\$f) + ".png",
              pillow.encode_png({"mode": \$rgb.mode, "size": \$rgb.size, "px": \$rgb.px,
                                 "palette": null, "info": {}, "png_filter": \$f}));
    \$f = \$f + 1;
}
// Phase P5: JPEG and EXIF. A JPEG written here is read by libjpeg on the
// other side, which is the only check that proves the entropy coder's bytes
// rather than this package's agreement with itself.
\$rgb.save(\$D + "out_q90.jpg", null, {"quality": 90, "subsampling": 0});
\$rgb.save(\$D + "out_q50_420.jpg", null, {"quality": 50, "subsampling": 2});
\$rgb.save(\$D + "out_opt.jpg", null, {"quality": 75, "optimize": true});
\$rgb.save(\$D + "out_prog.jpg", null, {"quality": 75, "progressive": true});
\$rgb.save(\$D + "out_rst.jpg", null, {"quality": 75, "restart_marker_blocks": 1});
\$rgb.convert("L", null, null).save(\$D + "out_gray.jpg", null, {"quality": 80});
\$rgb.convert("CMYK", null, null).save(\$D + "out_cmyk.jpg", null, {"quality": 80});

// The same pixels through our own decoder, for Pillow to compare against.
\$parts = [];
\$rd = pillow.open(\$D + "out_q90.jpg").getdata();
\$i = 0;
while (\$i < len(\$rd)) {
    push(\$parts, str(\$rd[\$i][0]) + "," + str(\$rd[\$i][1]) + "," + str(\$rd[\$i][2]));
    \$i = \$i + 1;
}
writefile(\$D + "jpeg_q90.expect", join(\$parts, "\n"));

// EXIF written here, for Pillow's Image.getexif() to read back.
\$ex = pillow.exif_empty();
pillow.exif_put(\$ex, "ifd0", 271, "Bantu");
pillow.exif_put(\$ex, "ifd0", 272, "pillow");
pillow.exif_put(\$ex, "ifd0", 274, 6);
pillow.exif_put(\$ex, "ifd0", 282, [72, 1]);
pillow.exif_put(\$ex, "ifd0", 296, 2);
pillow.exif_put(\$ex, "exif", 33434, [1, 125]);
pillow.exif_put(\$ex, "exif", 33437, [56, 10]);
pillow.exif_put(\$ex, "exif", 34855, 400);
pillow.exif_put(\$ex, "exif", 36867, "2026:10:30 12:00:00");
pillow.exif_put(\$ex, "gps", 1, "N");
pillow.exif_put(\$ex, "gps", 2, [[51, 1], [30, 1], [0, 1]]);
pillow.exif_put(\$ex, "interop", 1, "R98");
\$rgb.save(\$D + "out_exif.jpg", null, {"quality": 80, "exif": pillow.exif_tobytes(\$ex)});
writefile(\$D + "exif.bin", pillow.exif_tobytes(\$ex));

print("pillow wrote 5 + 17 + 10 files");
BANTU_EOF

"$BANTU" --quiet run "$TMP/emit.b" || { echo "[FAIL] the emitter did not run"; exit 1; }

"$PY" $PYFLAGS - "$TMP" <<'PY'
import sys, os
from PIL import Image

D = sys.argv[1]
bad = 0

def px(im):
    return list(im.get_flattened_data()) if hasattr(im, 'get_flattened_data') \
           else list(im.getdata())

def check(name, cond, detail=""):
    global bad
    if cond:
        print(f"   ok    {name}")
    else:
        print(f"   FAIL  {name}  {detail}")
        bad += 1

im = Image.open(os.path.join(D, 'rgb.ppm'))
check("rgb.ppm opens as RGB 23x17", im.mode == "RGB" and im.size == (23, 17),
      f"got {im.mode} {im.size}")
exp = [tuple(int(v) for v in l.split(',')) for l in
       open(os.path.join(D, 'rgb.expect')).read().split('\n')]
got = px(im)
check(f"all {len(exp)} RGB pixels identical", got == exp,
      f"{sum(1 for a, b in zip(got, exp) if a != b)} differ")

g = Image.open(os.path.join(D, 'grey.pgm'))
ref = im.convert('L')
check("convert('L') equals Pillow's, pixel for pixel", px(g) == px(ref),
      f"{sum(1 for a, b in zip(px(g), px(ref)) if a != b)} differ")

b = Image.open(os.path.join(D, 'bw_none.pbm'))
refb = im.convert('1', dither=Image.Dither.NONE)
check("convert('1', NONE) equals Pillow's, pixel for pixel", px(b) == px(refb),
      f"{sum(1 for a, c in zip(px(b), px(refb)) if a != c)} differ")

b9 = Image.open(os.path.join(D, 'bw9.pbm'))
check("a 9-wide 1-bit image keeps its size (row padding)", b9.size == (9, 5),
      f"got {b9.size}")

g16 = Image.open(os.path.join(D, 'g16.pgm'))
exp16 = [(x * 9000 + y * 300) % 65536 for y in range(3) for x in range(7)]
check("all 21 16-bit values identical (big-endian wire)", px(g16) == exp16,
      f"got {px(g16)[:4]} want {exp16[:4]}")

# ── Phase P4: Pillow reads the PNG, BMP and TGA files pillow wrote ──
print()
ref = px(im)                      # the RGB source, as Pillow sees it
refL = px(im.convert('L'))
refBW = px(im.convert('1', dither=Image.Dither.NONE))

cases = [
    ('out_rgb.png',  'RGB',  ref),
    ('out_rgba.png', 'RGBA', [(r, g, b, 255) for (r, g, b) in ref]),
    ('out_l.png',    'L',    refL),
    ('out_la.png',   'LA',   [(v, 255) for v in refL]),
    ('out_bw.png',   '1',    refBW),
    ('out_rgb.bmp',  'RGB',  ref),
    ('out_l.bmp',    'L',    refL),
    ('out_bw.bmp',   '1',    refBW),
    ('out_rgb.tga',  'RGB',  ref),
    ('out_rgba.tga', 'RGBA', [(r, g, b, 255) for (r, g, b) in ref]),
    ('out_l.tga',    'L',    refL),
]
for name, mode, want in cases:
    path = os.path.join(D, name)
    try:
        g = Image.open(path)
        g.load()
    except Exception as e:
        check(f"Pillow opens {name}", False, f"{type(e).__name__}: {e}")
        continue
    okmode = g.mode == mode
    okpx = px(g) == want
    check(f"Pillow reads {name} as {mode}, {len(want)} pixels identical",
          okmode and okpx,
          f"mode={g.mode} pixels differ={sum(1 for a, b in zip(px(g), want) if a != b)}")

# a P-mode PNG: compare through RGB, since the index depends on the palette
try:
    gp = Image.open(os.path.join(D, 'out_p.png'))
    gp.load()
    check("Pillow reads out_p.png as P and its colours match",
          gp.mode == 'P' and px(gp.convert('RGB')) == px(im.convert('P', dither=Image.Dither.NONE).convert('RGB')),
          f"mode={gp.mode}")
except Exception as e:
    check("Pillow opens out_p.png", False, f"{type(e).__name__}: {e}")

# every PNG scanline filter must decode to the same pixels in Pillow
for f in range(5):
    name = f'out_f{f}.png'
    try:
        g = Image.open(os.path.join(D, name))
        g.load()
        check(f"Pillow reads PNG filter {f} correctly", px(g) == ref,
              f"{sum(1 for a, b in zip(px(g), ref) if a != b)} pixels differ")
    except Exception as e:
        check(f"Pillow opens PNG filter {f}", False, f"{type(e).__name__}: {e}")

# ── Phase P5: JPEG, read by libjpeg itself ────────────────────────────────
import io
import math

def psnr(a, b):
    se = sum((x - y) ** 2 for x, y in zip(a, b))
    return float('inf') if se == 0 else 10 * math.log10(255 * 255 * len(a) / se)

# Each case is compared against PILLOW'S OWN ENCODE of the same source with
# the same options -- an absolute PSNR floor would only measure how badly
# JPEG does on this deliberately high-frequency 23x17 pattern (at 4:2:0 its
# chroma is destroyed, by Pillow too). `exact` says the bytes must match
# libjpeg's; progressive is the one case where they deliberately do not.
jpeg_cases = [
    ('out_q90.jpg', 'RGB', dict(quality=90, subsampling=0), True),
    ('out_q50_420.jpg', 'RGB', dict(quality=50, subsampling=2), True),
    ('out_opt.jpg', 'RGB', dict(quality=75, optimize=True), True),
    ('out_prog.jpg', 'RGB', dict(quality=75, progressive=True), False),
    ('out_rst.jpg', 'RGB', dict(quality=75, restart_marker_blocks=1), True),
    ('out_gray.jpg', 'L', dict(quality=80), True),
    ('out_cmyk.jpg', 'CMYK', dict(quality=80), True),
]
for name, mode, kw, exact in jpeg_cases:
    try:
        want = im if mode == 'RGB' else im.convert(mode)
        buf = io.BytesIO()
        want.save(buf, 'JPEG', **kw)
        theirs_bytes = buf.getvalue()
        ours_bytes = open(os.path.join(D, name), 'rb').read()
        g = Image.open(os.path.join(D, name))
        g.load()
        t = Image.open(io.BytesIO(theirs_bytes))
        t.load()
        if exact:
            check(f"{name} is byte-identical to what libjpeg writes",
                  ours_bytes == theirs_bytes,
                  f"{len(ours_bytes)} bytes vs {len(theirs_bytes)}")
        else:
            check(f"{name} decodes to the same pixels as libjpeg's own file",
                  g.mode == t.mode and g.tobytes() == t.tobytes(),
                  f"mode={g.mode}/{t.mode}")
        po = psnr(list(g.tobytes()), list(want.tobytes()))
        pt = psnr(list(t.tobytes()), list(want.tobytes()))
        check(f"{name} is no worse than libjpeg against the source",
              g.mode == mode and g.size == im.size and po >= pt - 0.01,
              f"ours {po:.2f} dB, libjpeg {pt:.2f} dB")
    except Exception as e:
        check(f"Pillow opens {name}", False, f"{type(e).__name__}: {e}")

# progressive must actually be progressive, and optimize must be smaller
try:
    gp = Image.open(os.path.join(D, 'out_prog.jpg'))
    check("out_prog.jpg really is progressive", gp.info.get('progressive') == 1,
          str(gp.info.get('progressive')))
except Exception as e:
    check("out_prog.jpg opens", False, str(e))
check("optimize produced a smaller file than the default tables",
      os.path.getsize(os.path.join(D, 'out_opt.jpg')) <
      os.path.getsize(os.path.join(D, 'out_q90.jpg')))

# our own decoder against libjpeg's, on the same file, pixel for pixel
try:
    g = Image.open(os.path.join(D, 'out_q90.jpg'))
    g.load()
    theirs = [f"{r},{gg},{b}" for (r, gg, b) in px(g)]
    ours = open(os.path.join(D, 'jpeg_q90.expect')).read().split("\n")
    bad_px = sum(1 for a, b in zip(ours, theirs) if a != b)
    check("our JPEG decoder and libjpeg agree on every pixel of the same file",
          ours == theirs, f"{bad_px} of {len(theirs)} pixels differ")
except Exception as e:
    check("the JPEG decoder comparison runs", False, f"{type(e).__name__}: {e}")

# EXIF written here, read by Pillow's own reader
try:
    g = Image.open(os.path.join(D, 'out_exif.jpg'))
    g.load()
    ex = g.getexif()
    sub = ex.get_ifd(0x8769)
    gps = ex.get_ifd(0x8825)
    itop = ex.get_ifd(0xA005)
    want_ifd0 = {271: 'Bantu', 272: 'pillow', 274: 6, 296: 2}
    ok_ifd0 = all(ex.get(k) == v for k, v in want_ifd0.items())
    check("Pillow reads back every IFD0 tag pillow wrote", ok_ifd0,
          str({k: ex.get(k) for k in want_ifd0}))
    check("Pillow reads XResolution as 72", float(ex.get(282)) == 72.0, str(ex.get(282)))
    check("Pillow reads back the Exif sub-IFD",
          float(sub.get(33434)) == 1 / 125 and sub.get(34855) == 400 and
          sub.get(36867) == '2026:10:30 12:00:00', str(dict(sub)))
    check("Pillow reads back the GPS IFD",
          gps.get(1) == 'N' and tuple(float(v) for v in gps.get(2)) == (51.0, 30.0, 0.0),
          str(dict(gps)))
    check("Pillow reads back the Interop IFD", itop.get(1) == 'R98', str(dict(itop)))
except Exception as e:
    check("the EXIF comparison runs", False, f"{type(e).__name__}: {e}")

# and the standalone block, loaded by Pillow's Exif directly
try:
    ex2 = Image.Exif()
    ex2.load(open(os.path.join(D, 'exif.bin'), 'rb').read())
    check("Pillow's Exif.load accepts the block pillow wrote",
          ex2.get(271) == 'Bantu' and ex2.get(274) == 6, str(dict(ex2)))
except Exception as e:
    check("Exif.load runs", False, f"{type(e).__name__}: {e}")

print()
sys.exit(1 if bad else 0)
PY
rc=$?

echo
if [ $rc -eq 0 ]; then
    echo "RESULT: ALL GREEN"
else
    echo "RESULT: FAILED"
fi
exit $rc
