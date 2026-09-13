#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  run_samples.sh — every runnable sample, executed.
#
#  Documentation rots silently: an example that stopped working still LOOKS
#  right in the docs, and the first person to find out is a new user on
#  their first five minutes with the library. So the gallery is executed,
#  not just written, and a sample that fails fails the build.
#
#  Run:  BANTU=./bantu-src/compiler/build/bantu bash tests/run_samples.sh
# ════════════════════════════════════════════════════════════════════════
set -u
BANTU="${BANTU:-bantu}"
PASS=0
FAIL=0

for f in samples/*/*.b; do
    [ -e "$f" ] || continue
    # server.b is this repo's convention for "binds a port and blocks forever",
    # and several also need a database that is not present in CI. Running them
    # here would hang the build, so they are skipped by name and the reason is
    # stated rather than left for someone to rediscover.
    case "$f" in */server.b) echo "  skip  $f (starts a server; needs a port and a database)"; continue ;; esac
    if out=$("$BANTU" -q run "$f" 2>&1); then
        # Exiting 0 is not enough: a sample that printed an error and carried
        # on would pass, and that is exactly the rot this guards against.
        if printf '%s' "$out" | grep -qiE '\[error\]|\[fatal\]|RESULT: FAILURES'; then
            echo "  FAIL  $f (ran, but reported an error)"
            printf '%s\n' "$out" | grep -iE '\[error\]|\[fatal\]' | head -3
            FAIL=$((FAIL+1))
        else
            echo "  ok    $f"
            PASS=$((PASS+1))
        fi
    else
        echo "  FAIL  $f (non-zero exit)"
        printf '%s\n' "$out" | tail -5
        FAIL=$((FAIL+1))
    fi
done

# The package smoke tests live beside their packages rather than in tests/,
# so CI's tests/*.b glob never reaches them. Run them here.
#
# tests/ is excluded explicitly: a bare */ glob matches it too, which would
# re-run the entire suite a second time -- several minutes of duplicated work
# for no extra coverage.
for f in */[a-z]*_test.b; do
    [ -e "$f" ] || continue
    case "$f" in tests/*) continue ;; esac
    if "$BANTU" -q run "$f" 2>&1 | grep -q "ALL GREEN"; then
        echo "  ok    $f"
        PASS=$((PASS+1))
    else
        echo "  FAIL  $f"
        FAIL=$((FAIL+1))
    fi
done

echo ""
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
