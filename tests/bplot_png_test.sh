#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  bplot_png_test.sh — the encoder, checked by decoders we did not write.
#
#  tests/bplot_raster_test.b asserts what each pixel is. This asks the
#  different question: does the rest of the world agree that these bytes are
#  a PNG, and that its pixels are the ones the canvas holds?
#
#    * Python's zlib decompresses every deflate stream bp_zlib produces --
#      empty input, one byte, incompressible random, and a megabyte of one
#      value -- back to exactly the input.
#    * Pillow opens every PNG and its pixels equal bp_canvas_raw(), at
#      several sizes and dpi values.
#    * The same canvas rendered twice, in two separate processes, is
#      byte-identical. That is the property the three-platform gate rests on,
#      and the one an accidental timestamp or hash order would break.
#
#  Run:  BANTU=./bantu-src/compiler/build/bantu bash tests/bplot_png_test.sh
# ════════════════════════════════════════════════════════════════════════
set -u
BANTU="${BANTU:-bantu}"
case "$BANTU" in /*) ;; */*) BANTU="$(cd "$(dirname "$BANTU")" && pwd)/$(basename "$BANTU")" ;; esac
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PASS=0
FAIL=0
ok()    { PASS=$((PASS+1)); echo "  ok    $1"; }
bad()   { FAIL=$((FAIL+1)); echo "  FAIL  $1"; }
check() { if [ "$1" = "1" ]; then ok "$2"; else bad "$2"; fi; }

if ! "$BANTU" -q run /dev/stdin <<'PROBE' 2>/dev/null | grep -q true
print(has_native("raster"));
PROBE
then
    echo "  --    this interpreter has no raster backend; skipping"
    exit 0
fi

PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
    echo "  --    no python; skipping the external-decoder gate"
    exit 0
fi
HAVE_PIL=0
"$PY" -c "import PIL" >/dev/null 2>&1 && HAVE_PIL=1

# ── Bantu writes: deflate streams, PNGs, and the pixels they should hold ──
cat > "$TMP/make.b" <<BEOF
\$dir = "$TMP";

// Deflate streams, with their inputs beside them.
def store(\$name, \$data) {
    writefile(\$dir + "/" + \$name + ".in", \$data, "wb");
    writefile(\$dir + "/" + \$name + ".z", bp_zlib(\$data), "wb");
    return null;
}
store("empty", "");
store("one", "A");
\$flat = "";
\$i = 0;
while (\$i < 100000) { \$flat = \$flat + "aaaaaaaaaa"; \$i = \$i + 1; }
store("flat", \$flat);
\$mixed = "";
\$i = 0;
while (\$i < 256) { \$mixed = \$mixed + chr(\$i); \$i = \$i + 1; }
\$rand = "";
\$i = 0;
nd_seed(11);
\$bytes = nd_to_list(nd_astype(nd_multiply(nd_random_uniform([20000], 0, 255.999), 1, null), "i64"));
while (\$i < len(\$bytes)) { \$rand = \$rand + chr(\$bytes[\$i]); \$i = \$i + 1; }
store("mixed", \$mixed);
store("random", \$rand);

// Canvases, each written as a PNG and as its raw pixels.
def draw(\$name, \$w, \$h, \$dpi) {
    \$c = bp_canvas_new(\$w, \$h, \$dpi, "#ffffff");
    bp_fill_rect(\$c, 0, 0, \$w, \$h / 4, "#1f77b4", null);
    bp_fill_rect(\$c, 2.5, \$h / 3, \$w / 2, \$h / 5, "#d62728", null);
    bp_fill_rect(\$c, \$w / 4, \$h / 2, \$w / 2, \$h / 4, "#2ca02c", 0.5);
    bp_canvas_clip(\$c, 1, 1, \$w - 2, \$h - 2);
    bp_fill_rect(\$c, 0, \$h * 0.8, \$w, \$h, "#9467bd", 0.8);
    bp_canvas_clip(\$c, null, null, null, null);
    bp_png_save(\$c, \$dir + "/" + \$name + ".png");
    writefile(\$dir + "/" + \$name + ".raw", bp_canvas_raw(\$c), "wb");
    \$info = bp_canvas_info(\$c);
    print(\$name + " " + str(\$info["width"]) + " " + str(\$info["height"]) + " " + str(\$info["dpi"]));
    return null;
}
draw("small", 40, 20, null);
draw("chart", 400, 250, null);
draw("print", 200, 120, 300);
draw("tiny",  3, 2, null);
BEOF

if ! "$BANTU" -q run "$TMP/make.b" > "$TMP/make.log" 2>&1; then
    bad "the canvases could not be rendered"
    tail -20 "$TMP/make.log"
    echo "  $PASS passed, $FAIL failed"
    exit 1
fi
ok "every canvas rendered and was written in binary"

# ── Python checks what Bantu wrote ───────────────────────────────────────
"$PY" - "$TMP" "$HAVE_PIL" > "$TMP/py.out" 2>&1 <<'EOF'
import os, sys, zlib
tmp, have_pil = sys.argv[1], sys.argv[2] == "1"
fails = []

for name in ("empty", "one", "flat", "mixed", "random"):
    src = open(os.path.join(tmp, name + ".in"), "rb").read()
    comp = open(os.path.join(tmp, name + ".z"), "rb").read()
    try:
        back = zlib.decompress(comp)
    except Exception as e:
        fails.append(f"{name}: zlib refused the stream ({e})")
        continue
    if back != src:
        fails.append(f"{name}: decompressed to {len(back)} bytes, not {len(src)}")
    else:
        print(f"ZLIB {name} {len(src)} -> {len(comp)}")

if have_pil:
    from PIL import Image
    for line in open(os.path.join(tmp, "make.log")):
        parts = line.split()
        if len(parts) != 4:
            continue
        name, w, h, dpi = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
        png = os.path.join(tmp, name + ".png")
        raw = open(os.path.join(tmp, name + ".raw"), "rb").read()
        im = Image.open(png)
        if im.mode != "RGB":
            fails.append(f"{name}: mode {im.mode}")
        if im.size != (w, h):
            fails.append(f"{name}: Pillow says {im.size}, the canvas says {(w, h)}")
        if im.tobytes() != raw:
            fails.append(f"{name}: decoded pixels differ from the canvas")
        px = im.info.get("dpi")
        if px and abs(px[0] - dpi) > 0.5:
            fails.append(f"{name}: pHYs says {px[0]} dpi, not {dpi}")
        print(f"PNG {name} {im.size} {os.path.getsize(png)}B dpi={px}")
else:
    print("NOPIL")

print("FAILS " + str(len(fails)))
for f in fails:
    print("  " + f)
EOF

sed -n 's/^ZLIB /        deflate /p;s/^PNG /        png     /p' "$TMP/py.out"
NF="$(sed -n 's/^FAILS //p' "$TMP/py.out")"
check "$([ "${NF:-1}" = "0" ] && echo 1 || echo 0)" "Python's zlib and Pillow agree with every file"
[ "${NF:-1}" = "0" ] || sed -n '/^FAILS/,$p' "$TMP/py.out" | tail -n +2
if grep -q NOPIL "$TMP/py.out"; then echo "  --    Pillow not installed; the PNG pixels were not decoded"; fi

# ── Determinism: two processes, one file ─────────────────────────────────
cat > "$TMP/again.b" <<BEOF
\$c = bp_canvas_new(400, 250, null, "#ffffff");
bp_fill_rect(\$c, 0, 0, 400, 62.5, "#1f77b4", null);
bp_fill_rect(\$c, 2.5, 83.333333333333329, 200, 50, "#d62728", null);
bp_fill_rect(\$c, 100, 125, 200, 62.5, "#2ca02c", 0.5);
bp_canvas_clip(\$c, 1, 1, 398, 248);
bp_fill_rect(\$c, 0, 200, 400, 250, "#9467bd", 0.8);
bp_canvas_clip(\$c, null, null, null, null);
bp_png_save(\$c, "$TMP/again.png");
BEOF
"$BANTU" -q run "$TMP/again.b" > /dev/null 2>&1
check "$(cmp -s "$TMP/chart.png" "$TMP/again.png" && echo 1 || echo 0)" \
      "the same canvas in a second process is byte-identical"

SUM1="$("$PY" -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$TMP/chart.png")"
echo "        chart.png sha256 $SUM1"

echo ""
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
