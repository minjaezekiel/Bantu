#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  run_samples.sh — every runnable sample, executed.
#
#  Documentation rots silently: an example that stopped working still LOOKS
#  right in the docs, and the first person to find out is a new user on
#  their first five minutes with the library. So the gallery is executed,
#  not just written, and a sample that fails fails the build.
#
#  [found] Every sample used to be run from the repo root ONLY -- and the repo
#  root is the one working directory in which a cwd-relative include cannot
#  fail. samples/bplot/*.b said `include "./bplot/bplot.b"`, which resolves
#  against the cwd, so CI was green while every one of them died with
#  "Module not found" for anybody who did the obvious thing and cd'd into
#  samples/bplot first. A gate that only tests the cwd that cannot fail is not
#  a gate, so each sample now runs TWICE: once from the root, once from its own
#  directory. Every sample writes its output to an absolute path under /tmp, so
#  the second pass is free of side effects.
#
#  Run:  BANTU=./bantu-src/compiler/build/bantu bash tests/run_samples.sh
# ════════════════════════════════════════════════════════════════════════
set -u
BANTU="${BANTU:-bantu}"
PASS=0
FAIL=0

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# An absolute BANTU, so the second pass can cd without losing the binary.
case "$BANTU" in
    /*) ;;
    */*) BANTU="$(cd "$(dirname "$BANTU")" && pwd)/$(basename "$BANTU")" ;;
esac
# The globs below are relative, so work from the repo root whatever the caller's
# cwd was -- and say so if the gallery is not there. An empty glob would
# otherwise run nothing, report "0 passed, 0 failed" and exit 0: a green build
# that tested nothing, which is the same class of mistake as only ever testing
# the cwd that cannot fail.
cd "$ROOT" || { echo "  FAIL  cannot cd to $ROOT"; exit 1; }
[ -d samples ] || { echo "  FAIL  no samples/ directory under $ROOT"; exit 1; }

# Run one sample with a given working directory and judge the result.
#   $1 = directory to run from   $2 = path to the sample, relative to it
#   $3 = label for the output line
run_sample() {
    local dir="$1" file="$2" label="$3" out
    if out=$(cd "$dir" && "$BANTU" -q run "$file" 2>&1); then
        # Exiting 0 is not enough: a sample that printed an error and carried
        # on would pass, and that is exactly the rot this guards against.
        if printf '%s' "$out" | grep -qiE '\[error\]|\[fatal\]|RESULT: FAILURES'; then
            echo "  FAIL  $label (ran, but reported an error)"
            printf '%s\n' "$out" | grep -iE '\[error\]|\[fatal\]' | head -3
            FAIL=$((FAIL+1))
        else
            echo "  ok    $label"
            PASS=$((PASS+1))
        fi
    else
        echo "  FAIL  $label (non-zero exit)"
        printf '%s\n' "$out" | tail -5
        FAIL=$((FAIL+1))
    fi
}

# server.b is this repo's convention for "binds a port and blocks forever",
# and several also need a database that is not present in CI. Running them
# here would hang the build, so they are skipped by name and the reason is
# stated rather than left for someone to rediscover.
#
# Both passes iterate the glob directly rather than reading from a helper
# through a pipe: a `... | while` body runs in a SUBSHELL, so every PASS/FAIL
# it counted would be thrown away at the end of the pipeline and a failing
# second pass would exit 0.
echo "-- from the repo root --"
for f in samples/*/*.b; do
    [ -e "$f" ] || continue
    case "$f" in */server.b) echo "  skip  $f (starts a server; needs a port and a database)"; continue ;; esac
    run_sample "$ROOT" "$f" "$f"
done

echo ""
echo "-- from each sample's own directory (what a user actually does) --"
for f in samples/*/*.b; do
    [ -e "$f" ] || continue
    case "$f" in */server.b) continue ;; esac
    d="$(dirname "$f")"
    run_sample "$ROOT/$d" "$(basename "$f")" "$f  [cd $d]"
done
echo ""

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
