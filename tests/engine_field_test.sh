#!/bin/zsh
# ════════════════════════════════════════════════════════════════════════
#  engine_field_test.sh — the "bantu" engine field, end to end.
#
#  What this proves, and why each one needed proving:
#
#    1. Every builtin is either in the last release's baseline or has a stated
#       floor. This is the gate that makes "adding a kernel without updating
#       the floor table" a test failure rather than a user's problem. It asks
#       the RUNNING BINARY what it registers, because grepping the sources for
#       registration calls missed 26 builtins (the scalar math family) that are
#       registered through a helper the grep did not know about.
#
#    2. A package needing a newer binary is refused BEFORE anything is written.
#       The old behaviour was a successful install followed by `Undefined
#       variable: nd_zeros` from inside the library.
#
#    3. A requirement the binary cannot parse is an ERROR, not "any version".
#
#    4. Every package in this repository declares a floor at least as high as
#       its sources imply. A declaration lower than the implication is the
#       dangerous case: it installs on a binary that cannot run it.
#
#  Run:  tests/engine_field_test.sh
# ════════════════════════════════════════════════════════════════════════

ROOT="${0:A:h:h}"
cd "$ROOT" || exit 1
B="$ROOT/bantu-src/compiler/build/bantu"
[[ -x "$B" ]] || B="$(command -v bantu)"
if [[ ! -x "$B" ]]; then
  echo "  [FATAL] no bantu binary found"
  exit 1
fi

pass=0
fail=0
ok()   { pass=$((pass+1)); print "  ok    $1" }
bad()  { fail=$((fail+1)); print "  FAIL  $1" }
check() { if [[ $1 == 0 ]]; then ok "$2"; else bad "$2"; fi }

print ""
print "  bantu engine field (ecosystem stage 2)"
print "  ════════════════════════════════════════"
print ""

# ── 1. the floor table covers every builtin ─────────────────────────────────
out=$("$B" engine --check-table 2>&1)
if print -r -- "$out" | grep -q "Every builtin is either in the released baseline or has a floor"; then
  ok "every builtin has a floor or is in the v1.3.2 baseline"
else
  bad "a builtin has no stated floor:"
  print -r -- "$out" | tail -20
fi

# the running binary agrees with itself about its version
ver=$("$B" engine 2>&1 | awk '/^  version:/ {print $2}')
if [[ -n "$ver" ]]; then ok "bantu engine reports a version ($ver)"; else bad "no version reported"; fi

# ── 2 and 3. install refusals ───────────────────────────────────────────────
TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT
mkdir -p "$TMP/futurepkg" "$TMP/badrange" "$TMP/okpkg" "$TMP/proj"

cat > "$TMP/futurepkg/package.json" <<'JSON'
{ "name": "engtest-future", "version": "1.0.0", "bantu": ">=99.0.0",
  "main": "m.b", "dependencies": {} }
JSON
print 'def f() { return 1; }' > "$TMP/futurepkg/m.b"

cat > "$TMP/badrange/package.json" <<'JSON'
{ "name": "engtest-badrange", "version": "1.0.0", "bantu": ">=1.2 <2.0",
  "main": "m.b", "dependencies": {} }
JSON
print 'def f() { return 1; }' > "$TMP/badrange/m.b"

cat > "$TMP/okpkg/package.json" <<'JSON'
{ "name": "engtest-ok", "version": "1.0.0", "bantu": ">=1.0.0",
  "main": "m.b", "dependencies": {} }
JSON
print 'def f() { return 1; }' > "$TMP/okpkg/m.b"

( cd "$TMP" && "$B" publish ./futurepkg >/dev/null 2>&1
             "$B" publish ./badrange  >/dev/null 2>&1
             "$B" publish ./okpkg     >/dev/null 2>&1 )

cd "$TMP/proj"
out=$("$B" add engtest-future 2>&1)
print -r -- "$out" | grep -q "needs a newer bantu"
check $? "a package requiring >=99.0.0 is refused"
print -r -- "$out" | grep -q ">= 99.0.0"
check $? "and the refusal states the requirement"
print -r -- "$out" | grep -q "running:"
check $? "and the running version"
[[ ! -d "$TMP/proj/bantu_modules/engtest-future" ]]
check $? "and NOTHING was written for it"
[[ ! -d "$TMP/proj/bantu_modules" || -z "$(ls -A "$TMP/proj/bantu_modules" 2>/dev/null)" ]]
check $? "and bantu_modules was not populated at all"

out=$("$B" add engtest-badrange 2>&1)
print -r -- "$out" | grep -q "cannot read"
check $? "an unparseable requirement is refused, not ignored"
print -r -- "$out" | grep -q "compound ranges are not supported"
check $? "and the refusal names the problem"
[[ ! -d "$TMP/proj/bantu_modules/engtest-badrange" ]]
check $? "and nothing was written for it either"

out=$("$B" add engtest-ok 2>&1)
print -r -- "$out" | grep -q "Installed engtest-ok"
check $? "a satisfiable requirement installs normally"
[[ -f "$TMP/proj/bantu_modules/engtest-ok/m.b" ]]
check $? "and its files are there"

cd "$ROOT"

# ── 4. every package here declares what it needs ────────────────────────────
print ""
for p in hash crypto uuid random numba arctic pillow btorch bplot orm lugha; do
  [[ -d "$p" ]] || continue
  out=$("$B" engine "$p" 2>&1)
  if print -r -- "$out" | grep -q "\[OK\]"; then
    declared=$(print -r -- "$out" | awk '/floor declared:/ {print $3}')
    ok "$p declares $declared and its sources agree"
  else
    bad "$p: declared floor is below what its sources need"
    print -r -- "$out" | grep -E "floor|FAIL"
  fi
done

print ""
print "  ========================================"
print "  PASS: $pass   FAIL: $fail"
print "  ========================================"
if [[ $fail -gt 0 ]]; then print "  RESULT: FAILURES PRESENT"; exit 1; fi
print "  RESULT: ALL GREEN"
