#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
#  build.sh — Build the Bantu interpreter binary (Ubuntu 22.04 compatible)
#
#  Design:
#    - Compiles each .cpp separately (smaller per-TU memory footprint,
#      clearer error messages, survives one-file failures).
#    - Does NOT use `set -e` globally; every step is checked explicitly
#      so that diagnostic commands (objdump/grep) cannot abort the build.
#    - Uses shell "macros" (small helper functions) to eliminate repetition
#      while keeping the flat, linear readability of the original.
#    - Prints clear [PASS]/[FAIL] markers so Render logs are readable.
#
#  Max runtime requirements (verified locally):
#    GLIBC_2.34, GLIBCXX_3.4.9  →  all ≤ Ubuntu 22.04's runtime
# ─────────────────────────────────────────────────────────────────────

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ═══════════════════════════════════════════════════════════════════════
#  Macros — tiny helpers that return 0 on success, 1 on failure
# ═══════════════════════════════════════════════════════════════════════

section()  { echo; echo "── $* ──"; }
hr()       { echo "════════════════════════════════════════════════════════════════"; }
pass()     { echo "[PASS] $*"; }
fail()     { echo "[FAIL] $*"; }

die() {
    fail "$1"
    shift
    [ $# -gt 0 ] && printf '       %s\n' "$@"
    exit 1
}

run_check() {
    local desc="$1"; shift
    if "$@" >/dev/null 2>&1; then
        pass "$desc"
        return 0
    else
        fail "$desc"
        return 1
    fi
}

compile_one() {
    local src="$1" obj="$2"; shift 2
    section "Compiling $src"
    if g++ "${CPP_FLAGS[@]}" -Wall -c "$src" -o "$obj" "$@" 2>&1; then
        pass "$src -> $obj ($(wc -c <"$obj") bytes)"
        return 0
    else
        die "Compilation failed for: $src" "Flags: ${CPP_FLAGS[*]} -Wall -c $src -o $obj $*"
    fi
}

check_header() {
    local hdr="$1"
    if printf '#include <%s>\nint main(){return 0;}\n' "$hdr" \
        | g++ -std=c++17 -x c++ - -o /tmp/hdrtest 2>/dev/null; then
        pass "<$hdr> found"
    else
        rm -f /tmp/hdrtest
        die "<$hdr> not found" \
            "Install libsqlite3-dev and libcurl4-openssl-dev"
    fi
}

# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════

SOURCES=(
    src/lexer.cpp
    src/parser.cpp
    src/ast.cpp
    src/types.cpp
    src/function.cpp
    src/class.cpp
    src/evaluator.cpp
    src/main.cpp
)

CPP_FLAGS=(
    -std=c++17
    -O2
    -mtune=generic
    -fno-plt
    -pthread
    -I src
)

LINK_LIBS=( -lsqlite3 -lcurl -ldl -lpthread )

# ═══════════════════════════════════════════════════════════════════════
#  Build
# ═══════════════════════════════════════════════════════════════════════

hr
echo "  Bantu interpreter build"
echo "  script dir: $SCRIPT_DIR"
echo "  g++ version: $(g++ --version  | head -1)"
echo "  gcc version: $(gcc --version  | head -1)"
hr

section "Checking build tools"
MISSING=()
for tool in g++ gcc ar ld; do
    command -v "$tool" >/dev/null 2>&1 || MISSING+=("$tool")
done
[ ${#MISSING[@]} -eq 0 ] || die "Missing required tools: ${MISSING[*]}"
pass "All required build tools present."

section "Checking headers"
for hdr in sqlite3.h curl/curl.h; do
    check_header "$hdr"
done
rm -f /tmp/hdrtest

section "Cleaning previous build"
rm -rf build; mkdir -p build

section "Compiling stub: stubs/ios_base_library_initv.c"
if gcc -O2 -c stubs/ios_base_library_initv.c -o build/ios_stub.o 2>&1; then
    pass "stub compiled"
else
    die "stub compilation failed"
fi

OBJECTS=()
for src in "${SOURCES[@]}"; do
    obj="build/$(basename "${src%.cpp}").o"
    compile_one "$src" "$obj"
    OBJECTS+=( "$obj" )
done

section "Linking build/bantu"
if g++ "${CPP_FLAGS[@]}" \
        "${OBJECTS[@]}" \
        build/ios_stub.o \
        -o build/bantu \
        "${LINK_LIBS[@]}" 2>&1; then
    pass "Linked build/bantu ($(wc -c <build/bantu) bytes)"
else
    die "Link failed" \
        "Objects: ${OBJECTS[*]}" \
        "Libs:    ${LINK_LIBS[*]}"
fi

section "Verifying binary"
[ -x build/bantu ] || die "build/bantu is not executable"
pass "build/bantu is executable"

section "ldd build/bantu"
ldd build/bantu 2>&1 || echo "(ldd not available or returned non-zero)"

section "Dynamic symbol requirements (informational only)"
if command -v objdump >/dev/null 2>&1; then
    echo "  GLIBC versions referenced:"
    objdump -T build/bantu 2>/dev/null \
        | grep -oE 'GLIBC_[0-9]+\.[0-9]+' \
        | sort -u | sed 's/^/    /' || true
    echo "  GLIBCXX versions referenced:"
    objdump -T build/bantu 2>/dev/null \
        | grep -oE 'GLIBCXX_[0-9]+\.[0-9]+\.[0-9]+' \
        | sort -u | sed 's/^/    /' || true
else
    echo "  (objdump not available — skipping symbol diagnostics)"
fi

echo
hr
echo "  ✓ Build complete: $(pwd)/build/bantu"
hr
