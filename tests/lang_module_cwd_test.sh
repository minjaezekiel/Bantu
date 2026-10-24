#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  lang_module_cwd_test.sh — an include means the same thing from anywhere.
#
#  [found] Module resolution depended on the process's working directory. A
#  bare name was looked for beside the importing file and under the cwd, and
#  nowhere else, so `include "bplot" as plt;` resolved from the repo root and
#  raised "Module not found" from samples/bplot -- the same file, run from its
#  own folder. Every test in the suite ran from the repo root, so nothing in
#  CI could see it; the first person to find out was a user who cd'd into the
#  examples directory, which is the obvious thing to do.
#
#  tests/lang_module_test.b asserts WHAT resolves. This asserts that the
#  answer does not change with the cwd: the same file is run from the repo
#  root, from tests/, from a sample directory, from $HOME and from / -- none
#  of which hold any of the modules it imports -- and every run must be
#  ALL GREEN. Paths given to bantu are absolute, so the only thing varying
#  between runs is the working directory.
#
#  Run:  BANTU=./bantu-src/compiler/build/bantu bash tests/lang_module_cwd_test.sh
# ════════════════════════════════════════════════════════════════════════
set -u
BANTU="${BANTU:-bantu}"
case "$BANTU" in
    /*) ;;
    */*) BANTU="$(cd "$(dirname "$BANTU")" && pwd)/$(basename "$BANTU")" ;;
esac
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEST="$ROOT/tests/lang_module_test.b"
PASS=0
FAIL=0

[ -f "$TEST" ] || { echo "  FAIL  missing $TEST"; exit 1; }

# A directory holding none of the modules the test imports, so a resolution
# that only worked by accident of the cwd has nowhere to succeed from.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

for dir in "$ROOT" "$ROOT/tests" "$ROOT/samples/bplot" "$TMP" "$HOME" "/"; do
    [ -d "$dir" ] || { echo "  skip  $dir (not a directory here)"; continue; }
    out="$(cd "$dir" && "$BANTU" -q run "$TEST" 2>&1)"
    if printf '%s' "$out" | grep -q 'RESULT: ALL GREEN'; then
        PASS=$((PASS+1))
        echo "  ok    all module includes resolve with cwd=$dir"
    else
        FAIL=$((FAIL+1))
        echo "  FAIL  module includes depend on the cwd (cwd=$dir)"
        printf '%s\n' "$out" | grep -iE 'FAIL|INCLUDE ERROR|\[fatal\]' | head -5
    fi
done

# The other half of the contract: a bare name that is genuinely absent must
# still be reported, from every cwd. A walk up the tree that answered "found"
# for anything would make the test above pass for the wrong reason.
cat > "$TMP/absent.b" <<'BEOF'
include "no_such_package_anywhere_xyz" as nope;
$raised = false;
try { $x = nope.anything; } catch ($e) { $raised = true; }
if ($raised) { print("RESULT: ALL GREEN"); }
BEOF
for dir in "$ROOT" "$TMP" "/"; do
    out="$(cd "$dir" && "$BANTU" -q run "$TMP/absent.b" 2>&1)"
    if printf '%s' "$out" | grep -q 'RESULT: ALL GREEN'; then
        PASS=$((PASS+1))
        echo "  ok    an absent bare name is still reported missing (cwd=$dir)"
    else
        FAIL=$((FAIL+1))
        echo "  FAIL  an absent bare name was not reported missing (cwd=$dir)"
    fi
done

echo ""
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
