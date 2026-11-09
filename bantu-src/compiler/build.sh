#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
#  build.sh — Build the Bantu interpreter binary (Ubuntu 22.04 compatible)
#
#  Design:
#    - Compiles each .cpp separately (smaller per-TU memory footprint,
#      clearer error messages, survives one-file failures).
#    - Does NOT use `set -e` globally; every step is checked explicitly
#      so that diagnostic commands (objdump/grep) cannot abort the build.
#    - Prints clear [PASS]/[FAIL] markers so Render logs are readable.
#
#  Max runtime requirements (verified locally):
#    GLIBC_2.34, GLIBCXX_3.4.9  →  all ≤ Ubuntu 22.04's runtime
# ─────────────────────────────────────────────────────────────────────

set -u   # treat unset vars as error, but DO NOT use -e/-o pipefail globally

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"


# ─── Arguments ───────────────────────────────────────────────────────────
#   --install           install the built binary over the one on PATH
#   --install-to <path> install it to an explicit path
DO_INSTALL=0
INSTALL_TO=""
while [ $# -gt 0 ]; do
    case "$1" in
        --install)    DO_INSTALL=1 ;;
        --install-to) DO_INSTALL=1; shift; INSTALL_TO="${1:-}" ;;
        -h|--help)
            echo "usage: $0 [--install] [--install-to <path>]"
            exit 0 ;;
        *) echo "unknown option: $1  (try --help)"; exit 2 ;;
    esac
    shift
done

echo "════════════════════════════════════════════════════════════════"
echo "  Bantu interpreter build"
echo "  script dir: $SCRIPT_DIR"
echo "  g++ version: $(g++ --version | head -1)"
echo "  gcc version: $(gcc --version | head -1)"
echo "════════════════════════════════════════════════════════════════"

# ─── Pre-flight: required tools ───────────────────────────────────
MISSING=()
for tool in g++ gcc ar ld; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        MISSING+=("$tool")
    fi
done
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "[FAIL] Missing required tools: ${MISSING[*]}"
    exit 1
fi
echo "[PASS] All required build tools present."

# ─── Pre-flight: required headers ─────────────────────────────────
echo "── Checking headers ──"
for hdr in sqlite3.h curl/curl.h; do
    if printf '#include <%s>\nint main(){return 0;}\n' "$hdr" \
        | g++ -std=c++17 -x c++ - -o /tmp/hdrtest 2>/dev/null; then
        echo "[PASS] <$hdr> found"
    else
        echo "[FAIL] <$hdr> not found — install libsqlite3-dev and libcurl4-openssl-dev"
        rm -f /tmp/hdrtest
        exit 1
    fi
done
rm -f /tmp/hdrtest

# ─── Clean ────────────────────────────────────────────────────────
echo "── Cleaning previous build ──"
rm -rf build
mkdir -p build

# ─── Compile ios_base_library_initv stub ──────────────────────────
echo "── Compiling stub: stubs/ios_base_library_initv.c ──"
if gcc -O2 -c stubs/ios_base_library_initv.c -o build/ios_stub.o 2>&1; then
    echo "[PASS] stub compiled"
else
    echo "[FAIL] stub compilation failed"
    exit 1
fi

# ─── Compile each .cpp separately ─────────────────────────────────
# Order matters: smaller files first to fail fast on simple errors.
SOURCES=(
    src/lexer.cpp
    src/parser.cpp
    src/ast.cpp
    src/types.cpp
    src/function.cpp
    src/class.cpp
    src/ndarray_native.cpp
    src/raster_native.cpp
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
    -DBANTU_FFI          # enable the FFI builtins (loadlib/func)
)


# ═══════════════════════════════════════════════════════════════════════
#  Build identity — so a stale install is visible
#
#  Three bantu binaries once sat on one machine, built twelve days apart,
#  and all three reported "v1.3.2". A stale install was therefore invisible
#  and its only symptom was "Undefined variable: nd_<something>" from a
#  library newer than the interpreter running it.
#
#  BANTU_BUILD_ID is a short digest of the compiler sources, so it changes
#  exactly when the binary's behaviour can. `bantu --version` prints it and
#  `bantu doctor` compares it against the binary your PATH resolves to.
# ═══════════════════════════════════════════════════════════════════════
compute_build_id() {
    # One digest over every compiler source. Sorted so the id does not depend
    # on the shell's glob order. Falls back through three tools because none of
    # shasum/sha256sum/cksum is present everywhere this script runs.
    local hasher=""
    if   command -v shasum    >/dev/null 2>&1; then hasher="shasum -a 256"
    elif command -v sha256sum >/dev/null 2>&1; then hasher="sha256sum"
    elif command -v cksum     >/dev/null 2>&1; then hasher="cksum"
    fi
    if [ -z "$hasher" ]; then echo "nohash"; return; fi
    # shellcheck disable=SC2086
    LC_ALL=C find src -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \) \
        | sort | xargs cat 2>/dev/null | $hasher | cut -c1-12
}
BUILD_ID="$(compute_build_id)"
BUILD_DATE="$(date -u '+%Y-%m-%d %H:%MZ')"
echo "  build id:   $BUILD_ID  ($BUILD_DATE)"
CPP_FLAGS+=( "-DBANTU_BUILD_ID=\"$BUILD_ID\"" "-DBANTU_BUILD_DATE=\"$BUILD_DATE\"" )

# Use -l:libcurl.so.4 (explicit filename) instead of -lcurl.
# Why: on Debian 13 the default `libcurl.so` symlink points to the GnuTLS
# flavor (`libcurl-gnutls.so.4`), which would make our binary depend on
# `libcurl-gnutls.so.4` at runtime — a package that doesn't exist on
# Ubuntu 22.04 (`libcurl4` there only provides `libcurl.so.4`, OpenSSL flavor).
# Forcing the exact soname `libcurl.so.4` makes the binary depend on the
# library that IS universally shipped as `libcurl4` on both Debian and
# Ubuntu, matching what the Dockerfile installs at runtime.
LINK_LIBS=( -lsqlite3 -l:libcurl.so.4 -lffi -ldl -lpthread )

# ── Optional: libsodium AEAD + argon2id, OFF by default ──────────────────
# Enable with:  BANTU_SODIUM=1 bash build.sh
#
# Kept opt-in so the default binary gains NO new runtime dependency, matching
# build-mac.sh. Statically linked where the archive is available so the result
# stays self-contained; the .a lives in a different place on Debian/Ubuntu than
# it does under Homebrew, so both are probed.
if [ "${BANTU_SODIUM:-0}" = "1" ]; then
    SODIUM_HDR=""
    for d in /usr/include /usr/local/include; do
        [ -f "$d/sodium.h" ] && SODIUM_HDR="$d" && break
    done
    if [ -z "$SODIUM_HDR" ]; then
        echo "[FAIL] BANTU_SODIUM=1 but sodium.h not found — install libsodium-dev"
        exit 1
    fi
    echo "  libsodium: $SODIUM_HDR"
    CPP_FLAGS+=( -DBANTU_SODIUM )
    SODIUM_A=""
    for d in /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/lib; do
        [ -f "$d/libsodium.a" ] && SODIUM_A="$d/libsodium.a" && break
    done
    if [ -n "$SODIUM_A" ]; then LINK_LIBS+=( "$SODIUM_A" ); else LINK_LIBS+=( -lsodium ); fi
fi

OBJECTS=()
for src in "${SOURCES[@]}"; do
    obj="build/$(basename "${src%.cpp}").o"
    echo
    echo "── Compiling $src ──"
    # The numba kernels are the one translation unit built at -O3; everything
    # else stays at -O2. This matters most HERE: the production binary is built
    # in ubuntu:22.04, whose GCC 11 does not enable -ftree-loop-vectorize at -O2
    # (GCC 12 does), while the macOS build's Apple Clang does. Without this, a
    # kernel benchmarked on a Mac would get NEON and this binary -- the one
    # users download -- would get a scalar loop for the same source.
    # Unquoted on purpose; these flags contain no spaces.
    kernel_flags=""
    case "$src" in
        */ndarray_native.cpp) kernel_flags="-O3 -ftree-vectorize" ;;
    esac
    # Use -Wall but NOT -Werror — we want warnings, not failures.
    if g++ "${CPP_FLAGS[@]}" $kernel_flags -Wall -c "$src" -o "$obj" 2>&1; then
        echo "[PASS] $src -> $obj ($(wc -c <"$obj") bytes)"
        OBJECTS+=( "$obj" )
    else
        echo "[FAIL] Compilation failed for: $src"
        echo "       Flags: ${CPP_FLAGS[*]}"
        exit 1
    fi
done

# ─── Link ─────────────────────────────────────────────────────────
echo
echo "── Linking build/bantu ──"
if g++ "${CPP_FLAGS[@]}" \
        "${OBJECTS[@]}" \
        build/ios_stub.o \
        -o build/bantu \
        "${LINK_LIBS[@]}" 2>&1; then
    echo "[PASS] Linked build/bantu ($(wc -c <build/bantu) bytes)"
else
    echo "[FAIL] Link failed"
    echo "       Objects: ${OBJECTS[*]}"
    echo "       Libs:    ${LINK_LIBS[*]}"
    exit 1
fi

# ─── Verify the binary is executable ──────────────────────────────
echo
echo "── Verifying binary ──"
if [ ! -x build/bantu ]; then
    echo "[FAIL] build/bantu is not executable"
    exit 1
fi
echo "[PASS] build/bantu is executable"

# ─── Print ldd output (informational only) ────────────────────────
echo
echo "── ldd build/bantu ──"
ldd build/bantu 2>&1 || echo "(ldd not available or returned non-zero)"

# ─── Print GLIBC/GLIBCXX requirements (informational only) ────────
echo
echo "── Dynamic symbol requirements (informational only) ──"
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
echo "════════════════════════════════════════════════════════════════"
echo "  ✓ Build complete: $(pwd)/build/bantu"
echo "════════════════════════════════════════════════════════════════"

BIN_NAME=bantu
hr_or_line() { echo "════════════════════════════════════════════════════════════════"; }

# ═══════════════════════════════════════════════════════════════════════
#  Install, and the staleness warning that should have existed all along
#
#  This script used to end at build/BIN and say nothing more, so every
#  build silently left whatever was on PATH in place. That is the whole
#  mechanism by which the installed binary fell twelve days behind.
# ═══════════════════════════════════════════════════════════════════════
path_bantu() { command -v bantu 2>/dev/null || true; }

ON_PATH="$(path_bantu)"
if [ "${DO_INSTALL:-0}" = "1" ]; then
    TARGET="${INSTALL_TO:-$ON_PATH}"
    if [ -z "$TARGET" ]; then
        TARGET="/usr/local/bin/$BIN_NAME"
        echo "  No bantu on PATH; installing to $TARGET"
    fi
    echo
    hr_or_line
    echo "  Installing build/$BIN_NAME -> $TARGET"
    if install -m 0755 "build/$BIN_NAME" "$TARGET" 2>/dev/null; then
        echo "  [OK] installed"
    elif sudo install -m 0755 "build/$BIN_NAME" "$TARGET"; then
        echo "  [OK] installed (via sudo)"
    else
        echo "  [FAIL] could not install to $TARGET"
        echo "         copy it yourself:  sudo install -m 0755 build/$BIN_NAME $TARGET"
        exit 1
    fi
    echo "  Verify:  bantu --version   # build should read $BUILD_ID"
elif [ -n "$ON_PATH" ]; then
    # Not installing. Say whether what is on PATH matches what was just built,
    # because the difference is otherwise completely silent.
    OTHER_ID="$("$ON_PATH" --build-id 2>/dev/null | head -1 || true)"
    case "$OTHER_ID" in
        "$BUILD_ID") echo "  PATH binary ($ON_PATH) is this build. [ OK ]" ;;
        "")          echo
                     echo "  ⚠ $ON_PATH does not understand --build-id, so it PREDATES"
                     echo "    this build. Code using any newer native builtin will fail"
                     echo "    there with \"Undefined variable: nd_...\"."
                     echo "    Install this build:  $0 --install" ;;
        *)           echo
                     echo "  ⚠ STALE: $ON_PATH is build $OTHER_ID, this is $BUILD_ID."
                     echo "    Install this build:  $0 --install" ;;
    esac
else
    echo "  (no bantu on PATH; run '$0 --install' to install this build)"
fi
