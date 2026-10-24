#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  lang_module_stress.sh — the bare-name ancestor walk, pushed on.
#
#  A bare `include "pkg"` is searched in the importing file's directory and
#  in every directory above it (see module_resolver.hpp, ancestorDirs). A
#  walk up a tree is the kind of thing that works on the happy path and then
#  hangs, crashes or picks the wrong file on a real filesystem, so the edges
#  are nailed down here rather than left to a user's layout:
#
#    * a package 30 directories above the file that imports it
#    * a file past the 64-level depth cap -- must report missing, not hang
#    * a path that runs through a symlink back to its own ancestor
#    * which of two candidates wins, by distance and by kind
#    * an explicit ./ path, which must NOT gain a walk
#    * 200 resolutions from a deep path, for pathological stat() growth
#
#  Everything is built in a temporary directory, so none of it depends on
#  this checkout's layout.
#
#  Run:  BANTU=./bantu-src/compiler/build/bantu bash tests/lang_module_stress.sh
# ════════════════════════════════════════════════════════════════════════
set -u
BANTU="${BANTU:-bantu}"
case "$BANTU" in
    /*) ;;
    */*) BANTU="$(cd "$(dirname "$BANTU")" && pwd)/$(basename "$BANTU")" ;;
esac

T="$(mktemp -d)"
trap 'chmod -R u+w "$T" 2>/dev/null; rm -rf "$T"' EXIT
PASS=0
FAIL=0
ok()  { PASS=$((PASS+1)); echo "  ok    $1"; }
bad() { FAIL=$((FAIL+1)); echo "  FAIL  $1"; }
chk() { if [ "$1" = "1" ]; then ok "$2"; else bad "$2"; fi; }

# A wall-clock guard for the cases that could hang. `timeout` is GNU
# coreutils and is NOT on a stock macOS, so it is used when present and the
# run goes ahead unguarded when it is not -- rather than reporting a failure
# that is only a missing tool. (That exact mistake cost a debugging detour:
# the symlink case below "failed" with rc=127 from a timeout that did not
# exist, while the resolver had handled it correctly all along.)
GUARD=""
if command -v timeout  >/dev/null 2>&1; then GUARD="timeout 60"
elif command -v gtimeout >/dev/null 2>&1; then GUARD="gtimeout 60"
fi
# Every bantu run below starts from / -- a directory holding none of these
# fixtures -- so nothing can resolve by accident of the working directory.
run_from_root() { (cd / && $GUARD "$BANTU" -q run "$1" 2>&1); }

echo "-- distance --"
mkdir -p "$T/proj/pkgx"
echo '$NAME = "pkgx";' > "$T/proj/pkgx/pkgx.b"
DEEP="$T/proj"
i=0; while [ $i -lt 30 ]; do DEEP="$DEEP/d$i"; i=$((i+1)); done
mkdir -p "$DEEP"
printf 'include "pkgx" as p;\nprint("GOT " + p.NAME);\n' > "$DEEP/deep.b"
out="$(run_from_root "$DEEP/deep.b")"
chk "$(printf '%s' "$out" | grep -c '^GOT pkgx$')" "a package 30 directories above the importing file resolves"

echo "-- the depth cap --"
# The walk stops after 64 levels. Past that the module is simply not found,
# which must be a clean report and a zero exit -- not a hang, and not a crash.
mkdir -p "$T/cap/pkgy"
echo '$NAME = "pkgy";' > "$T/cap/pkgy/pkgy.b"
FAR="$T/cap"
i=0; while [ $i -lt 80 ]; do FAR="$FAR/e$i"; i=$((i+1)); done
mkdir -p "$FAR"
printf 'include "pkgy" as p;\n$r = false;\ntry { $x = p.NAME; } catch ($e) { $r = true; }\nif ($r) { print("RAISED"); }\n' > "$FAR/far.b"
out="$(run_from_root "$FAR/far.b")"; rc=$?
chk "$([ "$rc" -ne 124 ] && [ "$rc" -eq 0 ] && printf '%s' "$out" | grep -q 'RAISED' && echo 1 || echo 0)" \
    "past the 64-level cap it reports missing and exits 0 -- no hang, no crash"

echo "-- a symlink back to an ancestor --"
# realpath() collapses the path before the walk starts, so a cycle through a
# symlink cannot produce an endless climb. Without that, this hangs.
mkdir -p "$T/loop/a/b" "$T/loop/pkgz"
echo '$NAME = "pkgz";' > "$T/loop/pkgz/pkgz.b"
if ln -s "$T/loop" "$T/loop/a/b/up" 2>/dev/null; then
    printf 'include "pkgz" as p;\nprint("GOT " + p.NAME);\n' > "$T/loop/a/b/up/a/b/sym.b"
    out="$(run_from_root "$T/loop/a/b/up/a/b/sym.b")"; rc=$?
    chk "$([ "$rc" -ne 124 ] && printf '%s' "$out" | grep -q '^GOT pkgz$' && echo 1 || echo 0)" \
        "a path through a symlink back to its own ancestor terminates and resolves"
else
    echo "  skip  symlink case (this filesystem refuses symlinks)"
fi

echo "-- which candidate wins --"
mkdir -p "$T/near/bantu_modules/dup" "$T/near/sub/bantu_modules/dup" "$T/near/sub/deeper"
echo '$WHERE = "far";'  > "$T/near/bantu_modules/dup/dup.b"
echo '$WHERE = "near";' > "$T/near/sub/bantu_modules/dup/dup.b"
printf 'include "dup" as d;\nprint("WHERE " + d.WHERE);\n' > "$T/near/sub/deeper/pick.b"
out="$(run_from_root "$T/near/sub/deeper/pick.b")"
chk "$(printf '%s' "$out" | grep -c '^WHERE near$')" "the NEAREST bantu_modules on the way up wins"

# Precedence by kind is unchanged by the walk: an installed package beats a
# folder that merely shares its name, even a closer one. A project's declared
# dependency should not be shadowed by a directory someone happened to create.
mkdir -p "$T/prec/bantu_modules/thing" "$T/prec/sub/thing"
echo '$KIND = "installed";' > "$T/prec/bantu_modules/thing/thing.b"
echo '$KIND = "folder";'    > "$T/prec/sub/thing/thing.b"
printf 'include "thing" as t;\nprint("KIND " + t.KIND);\n' > "$T/prec/sub/use.b"
out="$(run_from_root "$T/prec/sub/use.b")"
chk "$(printf '%s' "$out" | grep -c '^KIND installed$')" "an installed package still beats a NEARER same-named folder"

echo "-- an explicit path gains nothing --"
# The walk is for bare names only. If ./ paths started walking too, a typo'd
# relative path would silently find some other file up the tree.
mkdir -p "$T/expl/pkgq" "$T/expl/sub"
echo '$NAME = "pkgq";' > "$T/expl/pkgq/pkgq.b"
printf 'include "./pkgq/pkgq.b" as p;\n$r = false;\ntry { $x = p.NAME; } catch ($e) { $r = true; }\nif ($r) { print("RAISED"); }\n' > "$T/expl/sub/rel.b"
out="$(run_from_root "$T/expl/sub/rel.b")"
chk "$(printf '%s' "$out" | grep -c 'RAISED')" "an explicit ./ path still means exactly that path, with no walk"

echo "-- cost --"
# Two walks (from the importing file and from the cwd) overlap almost
# completely, and each level adds several candidate paths. The resolver
# de-duplicates before calling stat(), so a deep tree must not turn one
# include into hundreds of filesystem probes.
printf 'include "pkgx" as p;\nprint("GOT " + p.NAME);\n' > "$DEEP/bench.b"
start=$(date +%s)
i=0; while [ $i -lt 200 ]; do (cd / && "$BANTU" -q run "$DEEP/bench.b" >/dev/null 2>&1); i=$((i+1)); done
elapsed=$(( $(date +%s) - start ))
chk "$([ "$elapsed" -lt 60 ] && echo 1 || echo 0)" \
    "200 resolutions from 30 levels down took ${elapsed}s (budget 60s)"

echo ""
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
