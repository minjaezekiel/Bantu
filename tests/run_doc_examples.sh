#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  run_doc_examples.sh — every example in the package docs, executed.
#
#  A documented example that no longer runs still LOOKS right, and the first
#  person to find out is a new reader copying it. numba's roadmap recorded
#  "every documented example executed by CI" as its gate; nothing executed
#  them, and two of the nine were broken when this runner first ran -- a
#  reduction over an axis the array did not have, and a loop over arrays its
#  block never defined.
#
#  HOW A DOC IS RUN: its ```bantu blocks, in reading order, as ONE program --
#  the way a reader follows a tour, where a later block uses what an earlier
#  one built. A block that is illustrative only belongs in a ```text fence.
#
#  Each doc runs in its own empty working directory holding links to the
#  packages, so `include "bplot"` resolves exactly as it does in a project
#  and any file an example writes lands there, never in the repository.
#
#  Run:  BANTU=./bantu-src/compiler/build/bantu bash tests/run_doc_examples.sh
# ════════════════════════════════════════════════════════════════════════
set -u
BANTU="${BANTU:-bantu}"
case "$BANTU" in /*) ;; */*) BANTU="$(cd "$(dirname "$BANTU")" && pwd)/$(basename "$BANTU")" ;; esac
ROOT="${DOC_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
DOCS="${DOCS:-docs/bplot.md docs/numba.md docs/arctic.md}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PASS=0
FAIL=0

for doc in $DOCS; do
    [ -f "$ROOT/$doc" ] || { echo "  FAIL  $doc (missing)"; FAIL=$((FAIL+1)); continue; }
    name="$(basename "$doc" .md)"
    work="$TMP/$name"
    mkdir -p "$work"
    for pkg in bplot numba arctic; do ln -s "$ROOT/$pkg" "$work/$pkg"; done
    awk '/^```bantu[[:space:]]*$/ { inblk = 1; n++; printf "// ── example %d ──\n", n; next }
         inblk && /^```[[:space:]]*$/ { inblk = 0; next }
         inblk { print }
         END { if (inblk) exit 3 }' "$ROOT/$doc" > "$work/doc.b"
    if [ $? -eq 3 ]; then echo "  FAIL  $doc (an unterminated \`\`\`bantu fence)"; FAIL=$((FAIL+1)); continue; fi
    blocks="$(grep -c '^// ── example ' "$work/doc.b")"
    if [ "$blocks" -eq 0 ]; then echo "  --    $doc has no \`\`\`bantu examples"; continue; fi
    out="$(cd "$work" && "$BANTU" -q run ./doc.b 2>&1)"; rc=$?
    # Exiting 0 is not enough: an example that printed an error and carried on
    # would pass, and that is exactly the rot this exists to catch.
    if [ $rc -ne 0 ] || printf '%s' "$out" | grep -qiE '\[fatal\]|\[error\]|INCLUDE ERROR'; then
        echo "  FAIL  $doc ($blocks examples)"
        printf '%s\n' "$out" | grep -iE '\[fatal\]|\[error\]|INCLUDE ERROR' | head -3
        line="$(printf '%s' "$out" | sed -n 's/.*(line \([0-9]*\).*/\1/p' | head -1)"
        if [ -n "$line" ]; then
            awk -v L="$line" 'NR <= L && /^\/\/ ── example/ { ex = $0 } NR == L { print "        in " ex ": " $0 }' "$work/doc.b"
        fi
        FAIL=$((FAIL+1))
    else
        echo "  ok    $doc ($blocks examples, run in reading order)"
        PASS=$((PASS+1))
    fi
done

echo ""
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
