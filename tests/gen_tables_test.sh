#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  gen_tables_test.sh — the generators still produce the checked-in headers.
#
#  scripts/gen_circle_tables.b and scripts/gen_font_tables.b are run by hand,
#  and their output is committed (raster_tables.hpp, raster_font.hpp). A
#  generator nobody runs rots quietly; this reruns both into a temporary
#  directory and requires the output to be the committed file, byte for byte.
#
#  The font generator needs DejaVu Sans and its licence file side by side --
#  as matplotlib ships them. Set FONT=/path/DejaVuSans.ttf, or have matplotlib
#  importable; without either, that check says it was skipped.
#
#  Run:  BANTU=./bantu-src/compiler/build/bantu bash tests/gen_tables_test.sh
# ════════════════════════════════════════════════════════════════════════
set -u
BANTU="${BANTU:-bantu}"
case "$BANTU" in /*) ;; */*) BANTU="$(cd "$(dirname "$BANTU")" && pwd)/$(basename "$BANTU")" ;; esac
cd "$(dirname "$0")/.."
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PASS=0
FAIL=0
ok()  { PASS=$((PASS+1)); echo "  ok    $1"; }
bad() { FAIL=$((FAIL+1)); echo "  FAIL  $1"; }
SRC=bantu-src/compiler/src

if OUT="$TMP/tables.hpp" "$BANTU" -q run scripts/gen_circle_tables.b >"$TMP/c.log" 2>&1 \
   && cmp -s "$TMP/tables.hpp" "$SRC/raster_tables.hpp"; then
    ok "gen_circle_tables.b reproduces raster_tables.hpp byte for byte"
else
    bad "gen_circle_tables.b does not reproduce raster_tables.hpp"
    tail -5 "$TMP/c.log"; diff "$TMP/tables.hpp" "$SRC/raster_tables.hpp" 2>&1 | head -6
fi

FONT="${FONT:-}"
if [ -z "$FONT" ]; then
    PY="$(command -v python3 || true)"
    # Located, not imported: only the font file is wanted, and a matplotlib
    # with a broken dependency still has it on disk.
    [ -n "$PY" ] && FONT="$("$PY" -c 'import os, importlib.util as u; s = u.find_spec("matplotlib"); print(os.path.join(s.submodule_search_locations[0], "mpl-data", "fonts", "ttf", "DejaVuSans.ttf"))' 2>/dev/null || true)"
fi
if [ -n "$FONT" ] && [ -f "$FONT" ]; then
    start=$(date +%s)
    if FONT="$FONT" OUT="$TMP/font.hpp" "$BANTU" -q run scripts/gen_font_tables.b >"$TMP/f.log" 2>&1 \
       && cmp -s "$TMP/font.hpp" "$SRC/raster_font.hpp"; then
        ok "gen_font_tables.b reproduces raster_font.hpp byte for byte ($(( $(date +%s) - start )) s)"
    else
        bad "gen_font_tables.b does not reproduce raster_font.hpp"
        tail -5 "$TMP/f.log"; diff "$TMP/font.hpp" "$SRC/raster_font.hpp" 2>&1 | head -6
    fi
    # A font it cannot read must raise, not write a header.
    printf 'not a font' > "$TMP/bad.ttf"
    if FONT="$TMP/bad.ttf" OUT="$TMP/bad.hpp" "$BANTU" -q run scripts/gen_font_tables.b >/dev/null 2>&1 \
       || [ -f "$TMP/bad.hpp" ]; then
        bad "a file that is not a font is refused"
    else
        ok "a file that is not a font is refused, and no header is written"
    fi
else
    echo "  --    DejaVuSans.ttf not found (set FONT, or install matplotlib); font check skipped"
fi

echo ""
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
