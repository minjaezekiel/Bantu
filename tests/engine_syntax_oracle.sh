#!/bin/zsh
# ════════════════════════════════════════════════════════════════════════
#  engine_syntax_oracle.sh — the syntax floor, checked against the real
#  v1.3.2 interpreter instead of against a memory of it.
#
#  A package's engine floor is not only about builtins: the GRAMMAR has grown,
#  and a pure-Bantu package using a construct an older parser rejects is in the
#  same position as one calling a builtin that does not exist -- except it fails
#  at parse time with a message about a brace.
#
#  The honest way to know which constructs those are is to ASK THE OLD BINARY.
#  v1.3.2 is 22 source files and builds in about a minute, so this test checks
#  it out of git, builds it, and compares two verdicts for every source file in
#  every package here:
#
#      oracle    — does the v1.3.2 parser accept this file?
#      detector  — does `bantu engine` see post-baseline syntax in it?
#
#  They must agree on every file. A feature missing from syntaxFeatures() in
#  src/engine_req.hpp shows up as a file the oracle rejects and the detector
#  passes -- which is exactly how `exponent-in-brackets` was found, after
#  else-if and the qualified parent names had been added and 59 of 60 files
#  agreed.
#
#  Of the 60 files in this repository's eleven packages, 12 are rejected by the
#  1.3.2 parser (in arctic, pillow, btorch and lugha).
#
#  Run:  tests/engine_syntax_oracle.sh
#  Skips cleanly (exit 0) when git or a C++ compiler is unavailable, and says so.
# ════════════════════════════════════════════════════════════════════════

ROOT="${0:A:h:h}"
cd "$ROOT" || exit 1
B="$ROOT/bantu-src/compiler/build/bantu"
[[ -x "$B" ]] || B="$(command -v bantu)"
if [[ ! -x "$B" ]]; then print "  [FATAL] no bantu binary"; exit 1; fi

# The release this test uses as the oracle, and the commit that carries it.
BASE_TAG="v1.3.2"
BASE_COMMIT="7f4b2aa"

if ! command -v git >/dev/null || ! git -C "$ROOT" cat-file -e "$BASE_COMMIT^{commit}" 2>/dev/null; then
  print "  [SKIP] $BASE_COMMIT is not reachable from this checkout."
  print "         The oracle cannot be built, so the syntax table is unverified here."
  print "  RESULT: ALL GREEN (skipped)"
  exit 0
fi
if ! command -v g++ >/dev/null && ! command -v clang++ >/dev/null; then
  print "  [SKIP] no C++ compiler, so the $BASE_TAG oracle cannot be built."
  print "  RESULT: ALL GREEN (skipped)"
  exit 0
fi

TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT
print ""
print "  engine syntax floor, against the $BASE_TAG parser"
print "  ═════════════════════════════════════════════════"
print ""
print "  building the oracle from $BASE_COMMIT ..."
git -C "$ROOT" archive "$BASE_COMMIT" bantu-src/compiler/src | tar -x -C "$TMP" || {
  print "  [SKIP] could not extract the $BASE_TAG sources"; print "  RESULT: ALL GREEN (skipped)"; exit 0
}
if [[ -d /opt/homebrew/include ]]; then BP=/opt/homebrew; else BP=/usr/local; fi
CXX=${CXX:-g++}
if ! $CXX -std=c++17 -O1 -I "$TMP/bantu-src/compiler/src" -I"$BP/include" \
      "$TMP"/bantu-src/compiler/src/*.cpp -o "$TMP/bantu-base" \
      -L"$BP/lib" -lsqlite3 -lcurl -ldl 2>"$TMP/build.log"; then
  print "  [SKIP] the $BASE_TAG sources do not build here:"
  sed -n '1,4p' "$TMP/build.log" | sed 's/^/         /'
  print "  RESULT: ALL GREEN (skipped)"
  exit 0
fi
print "  oracle: $("$TMP/bantu-base" --version 2>&1 | head -1)"
print ""

pass=0; fail=0
ok()  { pass=$((pass+1)) }
bad() { fail=$((fail+1)); print "  FAIL  $1" }

# A parse failure, in either binary's wording.
parsefail() { grep -qE "compilation blocked|\[error\] line|SYNTAX ERROR" }

mkdir -p "$TMP/one"
verdicts() {   # $1 = file -> sets $oracle and $det
  setopt LOCAL_OPTIONS NULL_GLOB   # an empty one/ is normal on the first call
  if "$TMP/bantu-base" run "$1" 2>&1 | parsefail; then oracle=reject; else oracle=accept; fi
  rm -f "$TMP/one"/*.b 2>/dev/null
  cp "$1" "$TMP/one/" 2>/dev/null
  local n
  n=$("$B" engine "$TMP/one" 2>&1 | grep -oE "post-baseline syntax:    [0-9]+" | grep -oE "[0-9]+$")
  if [[ "$n" == "0" || -z "$n" ]]; then det=clean; else det=flag; fi
}

# ── 1. every package source file ────────────────────────────────────────────
rejected=0
checked=0
for p in hash crypto uuid random numba arctic pillow btorch bplot orm lugha; do
  [[ -d "$p" ]] || continue
  for f in $p/*.b; do
    [[ -f "$f" ]] || continue
    checked=$((checked+1))
    verdicts "$f"
    [[ $oracle == reject ]] && rejected=$((rejected+1))
    if [[ ( $oracle == reject && $det == flag ) || ( $oracle == accept && $det == clean ) ]]; then
      ok
    else
      bad "$f: the $BASE_TAG parser says $oracle, the detector says $det"
    fi
  done
done
print "  $checked package files checked; the $BASE_TAG parser rejects $rejected of them"

# ── 2. the constructs themselves, including the near misses ─────────────────
snippet() {   # $1 = name, $2 = code
  rm -f "$TMP/snip.b"
  print -r -- "$2" > "$TMP/snip.b"
  verdicts "$TMP/snip.b"
  if [[ ( $oracle == reject && $det == flag ) || ( $oracle == accept && $det == clean ) ]]; then
    ok
  else
    bad "$1: the $BASE_TAG parser says $oracle, the detector says $det"
  fi
}
snippet "else if"                 'if (1 > 2) { print("a"); } else if (2 > 1) { print("b"); }'
snippet "else { if } (old form)"  'if (1 > 2) { print("a"); } else { if (2 > 1) { print("b"); } }'
snippet "qualified extends"       'class B extends mod.A { def init() { return null; } }'
snippet "plain extends"           'class A { def init() { return null; } }
class B extends A { def init() { return null; } }'
snippet "exponent in a call"      '$y = f(1.0e300);'
snippet "exponent in parens"      '$z = (1.0e300);'
snippet "exponent in a list"      '$l = [1.0e300];'
snippet "exponent in a dict"      '$d = {"a": 1.0e300};'
snippet "exponent in a condition" 'if (1.0e300 > 1) { print("x"); }'
snippet "exponent at statement level" '$z = 1.0e300;'
snippet "exponent in a return"    'def g() { return 1.0e300; }'
snippet "exponent with arithmetic" '$z = 1.0e300 + 1.0;'
snippet "a plain float in a call" '$y = f(1.0);'
snippet "an identifier containing e" '$x1e5 = 2; print($x1e5);'

print ""
print "  ========================================"
print "  PASS: $pass   FAIL: $fail"
print "  ========================================"
if [[ $fail -gt 0 ]]; then print "  RESULT: FAILURES PRESENT"; exit 1; fi
print "  RESULT: ALL GREEN"
