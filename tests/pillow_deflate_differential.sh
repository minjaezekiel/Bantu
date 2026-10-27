#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  tests/pillow_deflate_differential.sh — pillow Phase P1, tier 2.
#
#  The compression work has to be checked against an implementation we did
#  not write, in BOTH directions:
#
#    read:   zlib and gzip produce the corpus in tests/fixtures/pillow/deflate
#            (levels 0, 1, 6, 9), and both pillow decoders must reproduce the
#            original bytes exactly -- tests/pillow_inflate_test.b does that
#            and is run here too.
#    write:  pillow/deflate.b produces streams at every level, and Python's
#            zlib and gzip must read them back to the original bytes.
#
#  Requires python3 (for its zlib). Skips with a clear message if absent, so
#  CI on a machine without Python reports a skip rather than a failure.
#
#  Run:  tests/pillow_deflate_differential.sh
# ════════════════════════════════════════════════════════════════════════
set -u

cd "$(dirname "$0")/.." || exit 1
BANTU="${BANTU:-./bantu}"
[ -x "$BANTU" ] || { echo "[FAIL] no bantu binary at $BANTU"; exit 1; }

if ! command -v python3 >/dev/null 2>&1; then
    echo "[SKIP] python3 not found -- the differential tier needs its zlib"
    exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "── regenerating the read-side corpus and checking both decoders ──"
"$BANTU" --quiet run tests/pillow_inflate_test.b 2>&1 | tail -3
"$BANTU" --quiet run tests/pillow_inflate_test.b 2>&1 | grep -q "RESULT: ALL GREEN" || {
    echo "[FAIL] tests/pillow_inflate_test.b is not green"; exit 1; }

echo
echo "── write side: python zlib reads what pillow/deflate.b writes ──"

cat > "$TMP/emit.b" <<BANTU_EOF
include "./pillow/deflate.b" as deflate;
\$D = "$TMP/";
\$parts = []; \$i = 0;
while (\$i < 200) { push(\$parts, "The quick brown fox jumps over the lazy dog. "); \$i = \$i + 1; }
\$cases = {};
\$cases.english = join(\$parts, "");
\$cases.tiny = "a";
\$cases.empty = "";
\$r = []; \$i = 0; while (\$i < 3000) { push(\$r, 65); \$i = \$i + 1; }
\$cases.runs = frombytes(\$r);
\$a = []; \$i = 0; while (\$i < 256) { push(\$a, \$i); \$i = \$i + 1; }
\$cases.bytes256 = frombytes(\$a);
\$cases.random = frombytes(randbytes(3000));
for \$k, \$v in \$cases.items() {
    writefile(\$D + \$k + ".raw", \$v);
    \$lv = 0;
    while (\$lv <= 9) { writefile(\$D + \$k + "_" + str(\$lv) + ".z", deflate.zlib(\$v, \$lv)); \$lv = \$lv + 1; }
    writefile(\$D + \$k + ".gz", deflate.gzip(\$v, 6));
}
print("emitted " + str(\$cases.size()) + " cases");
BANTU_EOF

"$BANTU" --quiet run "$TMP/emit.b" || { echo "[FAIL] the emitter did not run"; exit 1; }

python3 -I - "$TMP" <<'PY'
import sys, os, zlib, gzip
D = sys.argv[1]
names = sorted(f[:-4] for f in os.listdir(D) if f.endswith('.raw'))
bad = 0; total = 0
for nm in names:
    raw = open(os.path.join(D, nm + '.raw'), 'rb').read()
    for lvl in range(10):
        total += 1
        try:
            got = zlib.decompress(open(os.path.join(D, f'{nm}_{lvl}.z'), 'rb').read())
        except Exception as e:
            print(f'  FAIL zlib {nm} L{lvl}: {e}'); bad += 1; continue
        if got != raw:
            print(f'  FAIL zlib {nm} L{lvl}: {len(got)} bytes, expected {len(raw)}'); bad += 1
    total += 1
    try:
        got = gzip.decompress(open(os.path.join(D, nm + '.gz'), 'rb').read())
        if got != raw:
            print(f'  FAIL gzip {nm}: {len(got)} bytes, expected {len(raw)}'); bad += 1
    except Exception as e:
        print(f'  FAIL gzip {nm}: {e}'); bad += 1
print(f'   python read {total} streams written by pillow/deflate.b: {total - bad} ok, {bad} bad')
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
