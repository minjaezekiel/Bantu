#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  bplot_package_test.sh — the B5 gate, from a user's side of the screen.
#
#  A library nobody can install has failed, whatever its test suite says.
#  So this does exactly what a new user does, in an empty directory:
#
#      bantu publish ./bplot          (the maintainer's step)
#      bantu add bplot                (the user's step)
#      include "bplot" as plt;        (and it works)
#
#  and then the thing that makes bplot and arctic one library rather than
#  two: $df.plot() in a project where both were installed with `bantu add`,
#  which means arctic's lazy include has to find bplot in bantu_modules/,
#  not in a source checkout.
#
#  HOME is redirected into a temporary directory for the whole run, so the
#  registry `publish` writes to is thrown away afterwards. This test never
#  touches the real ~/.bantu.
#
#  Run:  BANTU=./bantu-src/compiler/build/bantu bash tests/bplot_package_test.sh
# ════════════════════════════════════════════════════════════════════════
set -u
BANTU="${BANTU:-bantu}"
case "$BANTU" in /*) ;; */*) BANTU="$(cd "$(dirname "$BANTU")" && pwd)/$(basename "$BANTU")" ;; esac
ROOT="${BPLOT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PASS=0
FAIL=0
ok()    { PASS=$((PASS+1)); echo "  ok    $1"; }
bad()   { FAIL=$((FAIL+1)); echo "  FAIL  $1"; }
check() { if [ "$1" = "1" ]; then ok "$2"; else bad "$2"; fi; }

PY="$(command -v python3 || command -v python || true)"
wellformed() {
    [ -n "$PY" ] || return 0
    "$PY" -c 'import sys, xml.dom.minidom; xml.dom.minidom.parse(sys.argv[1])' "$1" >/dev/null 2>&1
}

REAL_HOME="${HOME:-}"
export HOME="$TMP/home"
export USERPROFILE="$TMP/home"   # Windows reads this instead
mkdir -p "$HOME"

echo "-- publish, into a throwaway registry --"
( cd "$ROOT" && "$BANTU" publish ./bplot ) > "$TMP/pub_bplot.log" 2>&1
check "$([ $? -eq 0 ] && grep -q 'Published bplot@' "$TMP/pub_bplot.log" && echo 1 || echo 0)" "bantu publish ./bplot"
( cd "$ROOT" && "$BANTU" publish ./arctic ) > "$TMP/pub_arctic.log" 2>&1
check "$([ $? -eq 0 ] && grep -q 'Published arctic@' "$TMP/pub_arctic.log" && echo 1 || echo 0)" "bantu publish ./arctic"
check "$([ -d "$HOME/.bantu/registry/bplot" ] && echo 1 || echo 0)" "the registry is under the redirected HOME"
if [ -n "$REAL_HOME" ] && [ "$REAL_HOME" != "$HOME" ]; then
    check "$([ "$(cd "$ROOT" && pwd)" != "$HOME" ] && echo 1 || echo 0)" "and not the real one"
fi

echo "-- a clean project: bantu add, then include --"
mkdir -p "$TMP/app"
( cd "$TMP/app" && "$BANTU" add bplot && "$BANTU" add arctic ) > "$TMP/add.log" 2>&1
check "$([ $? -eq 0 ] && echo 1 || echo 0)" "bantu add bplot and bantu add arctic succeed in an empty directory"
check "$([ -f "$TMP/app/bantu_modules/bplot/bplot.b" ] && echo 1 || echo 0)" "bplot is installed into bantu_modules/"
check "$(grep -c '"bplot"' "$TMP/app/bantu.json" 2>/dev/null | awk '{print ($1 > 0) ? 1 : 0}')" "and recorded in bantu.json"
check "$(diff -r "$ROOT/bplot" "$TMP/app/bantu_modules/bplot" >/dev/null 2>&1 && echo 1 || echo 0)" \
      "the installed copy is exactly the published source"

cat > "$TMP/app/main.b" <<'BEOF'
include "bplot" as plt;
include "arctic" as ac;

// The three lines from the top of docs/bplot.md.
plt.plot([1, 2, 3], [2, 4, 9]);
print("wrote " + plt.savefig("quickstart.svg"));

// A table, through arctic's lazy include of bplot from bantu_modules/.
$df = ac.dataframe({"month": ["Jan", "Feb", "Mar"], "rain": [66, 61, 118], "temp": [28.1, 28.4, 27.9]}, null);
$df.plot({"kind": "bar", "x": "month", "y": "rain", "title": "Installed"});
print("wrote " + plt.savefig("frame.svg"));

// A Series straight into plot().
plt.plot($df.get("rain"), $df.get("temp"), null);
print("wrote " + plt.savefig("series.svg"));
BEOF
OUT="$(cd "$TMP/app" && "$BANTU" -q run main.b 2>&1)"
check "$(printf '%s' "$OUT" | grep -qiE '\[fatal\]|\[error\]|INCLUDE ERROR' && echo 0 || echo 1)" \
      "include \"bplot\" and include \"arctic\" resolve from bantu_modules and run cleanly"
[ "$(printf '%s' "$OUT" | grep -ciE 'fatal|error')" -gt 0 ] && printf '%s\n' "$OUT" | grep -iE 'fatal|error' | head -3
for f in quickstart frame series; do
    check "$([ -s "$TMP/app/$f.svg" ] && wellformed "$TMP/app/$f.svg" && echo 1 || echo 0)" "$f.svg was written and parses as XML"
done
check "$(grep -c '>Jan</text>' "$TMP/app/frame.svg" 2>/dev/null | awk '{print ($1 > 0) ? 1 : 0}')" \
      "\$df.plot() found bplot through arctic's lazy include and drew the categories"
check "$(grep -c '>Installed</text>' "$TMP/app/frame.svg" 2>/dev/null | awk '{print ($1 > 0) ? 1 : 0}')" \
      "and it drew into the same figure plt.savefig() wrote"

SMOKE="$(cd "$TMP/app" && "$BANTU" -q run bantu_modules/bplot/bplot_test.b 2>&1)"
check "$(printf '%s' "$SMOKE" | grep -c 'RESULT: ALL GREEN')" "the installed package's own smoke test passes"

echo "-- arctic without bplot: the error names the fix --"
mkdir -p "$TMP/nobplot"
( cd "$TMP/nobplot" && "$BANTU" add arctic ) > /dev/null 2>&1
cat > "$TMP/nobplot/main.b" <<'BEOF'
include "arctic" as ac;
$df = ac.dataframe({"v": [1, 2, 3]}, null);
$msg = "";
try { $df.plot(null); } catch ($e) { $msg = str($e); }
print("MSG " + $msg);
// A program that never plots must not have loaded bplot, and must still work.
print("SUM " + str($df.get("v").sum()));
BEOF
NB="$(cd "$TMP/nobplot" && "$BANTU" -q run main.b 2>&1)"
check "$(printf '%s' "$NB" | grep -c 'bantu add bplot')" "\$df.plot() without bplot says to run bantu add bplot"
check "$(printf '%s' "$NB" | grep -c '^SUM 6$')" "and the rest of arctic keeps working"

echo ""
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
