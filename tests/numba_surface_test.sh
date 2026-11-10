#!/bin/zsh
# ════════════════════════════════════════════════════════════════════════
#  numba_surface_test.sh — numba.b exposes every kernel it has.
#
#  numba.b's own header states the rule: "every builtin gets a wrapper even
#  when the wrapper looks like it does nothing", because a wrapper can have
#  optional arguments and a builtin cannot. Nothing enforced it, and thirteen
#  kernels had drifted out of reach of the package surface -- among them
#  nd_concat/nd_stack/nd_split/nd_tile/nd_repeat/nd_pad (NumPy's whole
#  joining family), the six bitwise ops, and nd_set_threads, which is the
#  entire D3 threading knob. A user of the `numba` PACKAGE could not turn
#  threading on at all; only someone calling the raw builtin could.
#
#  That is the failure mode this test exists for: the gap is invisible from
#  inside either side. The builtins are registered and tested, the wrappers
#  are present and tested, and no test compared the two SETS.
#
#  It asks the running binary what it registers rather than grepping the C++
#  sources for registration calls, which is the lesson from the engine floor
#  table (ecosystem DECISIONS E24): 26 builtins are registered through a
#  helper that a source grep did not recognise.
#
#  Run:  tests/numba_surface_test.sh
# ════════════════════════════════════════════════════════════════════════

ROOT="${0:A:h:h}"
cd "$ROOT" || exit 1
B="$ROOT/bantu-src/compiler/build/bantu"
[[ -x "$B" ]] || B="$(command -v bantu)"
if [[ ! -x "$B" ]]; then
  print "  [FATAL] no bantu binary found"
  exit 1
fi

pass=0
fail=0
ok()  { pass=$((pass+1)); print "  ok    $1" }
bad() { fail=$((fail+1)); print "  FAIL  $1" }

print ""
print "  numba public surface vs the kernels underneath"
print "  ════════════════════════════════════════"
print ""

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# `engine --builtins` is the binary's own list of what it registers.
"$B" engine --builtins 2>/dev/null | tr -s ' \t,' '\n' \
  | grep -E '^nd_' | sort -u > "$TMP/kernels"

# The wrapper file's references. Two subtleties, both of which bit on the
# first run: case matters, because nd_T is a kernel and a lowercase-only
# pattern drops it; and the match must start at an identifier boundary,
# because `nd_[a-z_]+` finds "nd_dims" inside the word `expand_dims` and then
# reports a kernel that was never called.
grep -ohE '(^|[^A-Za-z_0-9])nd_[A-Za-z_0-9]+' numba/numba.b \
  | sed -E 's/^[^A-Za-z_]*//' | sort -u > "$TMP/wrapped"

n_kernels=$(wc -l < "$TMP/kernels" | tr -d ' ')
if [[ "$n_kernels" -ge 150 ]]; then
  ok "the binary registers $n_kernels nd_* kernels"
else
  bad "only $n_kernels nd_* kernels found -- is this the right binary?"
fi

# ── the deliberate exceptions ───────────────────────────────────────────────
# Each is listed in numba.b's header with a reason. Kept here as data so an
# addition to the list has to be a deliberate edit of this test, not a
# side-effect of adding a kernel.
EXEMPT=(
  nd_im2col nd_col2im nd_conv_outsize      # btorch's convolution lowering
  nd_from_column nd_to_column nd_from_frame # arctic's DataFrame bridge
  nd_base_id                                # a view-identity hook for tests
)
printf '%s\n' $EXEMPT | sort -u > "$TMP/exempt"

# every exemption must still name a REAL kernel, or the list is stale
stale=$(comm -23 "$TMP/exempt" "$TMP/kernels")
if [[ -z "$stale" ]]; then
  ok "all ${#EXEMPT} documented exemptions name kernels that still exist"
else
  bad "an exemption names a kernel that no longer exists:"
  print -r -- "$stale" | sed 's/^/          /'
fi

# ── the gate ────────────────────────────────────────────────────────────────
comm -23 "$TMP/kernels" "$TMP/wrapped" | sort -u > "$TMP/unwrapped"
comm -23 "$TMP/unwrapped" "$TMP/exempt" > "$TMP/missing"

if [[ ! -s "$TMP/missing" ]]; then
  ok "every non-exempt kernel has a numba.b wrapper"
else
  bad "$(wc -l < "$TMP/missing" | tr -d ' ') kernels are unreachable from the package surface:"
  sed 's/^/          /' "$TMP/missing"
fi

# ── and the wrappers are callable, not merely written ────────────────────────
# A wrapper naming a kernel that does not exist parses fine and fails at the
# call, so the names are checked in both directions.
comm -13 "$TMP/kernels" "$TMP/wrapped" > "$TMP/phantom"
if [[ ! -s "$TMP/phantom" ]]; then
  ok "no wrapper names a kernel the binary does not register"
else
  bad "a wrapper calls a kernel that does not exist:"
  sed 's/^/          /' "$TMP/phantom"
fi

print ""
print "========================================"
print "  PASS: $pass   FAIL: $fail"
print "========================================"
if [[ $fail -gt 0 ]]; then print "  RESULT: FAILURES PRESENT"; exit 1; fi
print "  RESULT: ALL GREEN"
