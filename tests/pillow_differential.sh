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
# The probe MUST use the same -I (isolated) flag the real call does: without
# it, python3 can find Pillow in user site-packages, -I cannot, and the script
# would announce a reference it then fails to import.
if ! "$PY" -I -c "import PIL" >/dev/null 2>&1; then
    echo "[SKIP] $PY has no Pillow -- tier 2's live comparison needs it."
    echo "       The checked-in fixtures still cover this in the .b suites:"
    echo "         bantu run tests/pillow_core_test.b"
    echo "         bantu run tests/pillow_convert_test.b"
    echo "       To enable this script:  python3 -m pip install Pillow"
    exit 0
fi
echo "reference: $("$PY" -I -c 'import PIL; print("Pillow", PIL.__version__)')"

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
print("pillow wrote 5 + 17 files");
BANTU_EOF

"$BANTU" --quiet run "$TMP/emit.b" || { echo "[FAIL] the emitter did not run"; exit 1; }

"$PY" -I - "$TMP" <<'PY'
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
