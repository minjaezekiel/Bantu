#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
#  build-mac.sh — Build the Bantu interpreter binary (macOS)
# ─────────────────────────────────────────────────────────────────────

set -u

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

# ═══════════════════════════════════════════════════════════════════════
#  Macros
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

compile_one() {
    local src="$1" obj="$2"; shift 2
    # The numba kernels are the one translation unit built at -O3; everything
    # else stays at -O2. This is not a micro-optimization: the production Linux
    # binary is built in ubuntu:22.04, and GCC 11 does not enable
    # -ftree-loop-vectorize at -O2 (GCC 12 does), while Apple Clang does. Without
    # the override a kernel benchmarked here would get NEON and the same kernel
    # in the shipped Linux binary would get a scalar loop. Kept identical on both
    # platforms so a number measured on one means something on the other.
    # (Unquoted on purpose: an empty array under `set -u` is an error in the
    # bash 3.2 that macOS ships. These flags contain no spaces.)
    local extra=""
    case "$src" in
        */ndarray_native.cpp) extra="-O3 -ftree-vectorize" ;;
    esac
    section "Compiling $src"
    if g++ "${CPP_FLAGS[@]}" $extra -Wall -c "$src" -o "$obj" "$@" 2>&1; then
        pass "$src -> $obj ($(wc -c <"$obj") bytes)"
        return 0
    else
        die "Compilation failed for: $src"
    fi
}

check_header() {
    local hdr="$1"
    if printf '#include <%s>\nint main(){return 0;}\n' "$hdr" \
        | g++ -std=c++17 -x c++ - -o /tmp/hdrtest 2>/dev/null; then
        pass "<$hdr> found"
    else
        rm -f /tmp/hdrtest
        die "<$hdr> not found" "Run: brew install sqlite curl"
    fi
}

# ═══════════════════════════════════════════════════════════════════════
#  Configuration — macOS
# ═══════════════════════════════════════════════════════════════════════

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

# Detect Homebrew prefix
if [ -d /opt/homebrew ]; then
    BREW_PREFIX="/opt/homebrew"
else
    BREW_PREFIX="/usr/local"
fi

CPP_FLAGS=(
    -std=c++17
    -O2
    -I src
    -I"$BREW_PREFIX/include"
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

LINK_LIBS=(
    -L"$BREW_PREFIX/lib"
    -lsqlite3
    -lcurl
    -lffi                # libffi — foreign function interface
    -ldl                 # dlopen/dlsym
)

# ── Optional: libsodium AEAD + argon2id (C3), OFF by default ──────────────
# Enable with:  BANTU_SODIUM=1 bash build-mac.sh
# Kept opt-in so the default binary gains NO new runtime dependency (important
# for existing production deployments). When enabled we STATICALLY link
# libsodium (libsodium.a) so the resulting binary stays self-contained.
if [ "${BANTU_SODIUM:-0}" = "1" ]; then
    SODIUM_PREFIX="$(brew --prefix libsodium 2>/dev/null || true)"
    if [ -z "$SODIUM_PREFIX" ] || [ ! -f "$SODIUM_PREFIX/include/sodium.h" ]; then
        die "BANTU_SODIUM=1 but libsodium not found" \
            "Install it first:  brew install libsodium"
    fi
    echo "  libsodium: $SODIUM_PREFIX (static link)"
    CPP_FLAGS+=( -DBANTU_SODIUM -I"$SODIUM_PREFIX/include" )
    # Prefer the static archive so the binary carries no libsodium .dylib dep.
    if [ -f "$SODIUM_PREFIX/lib/libsodium.a" ]; then
        LINK_LIBS+=( "$SODIUM_PREFIX/lib/libsodium.a" )
    else
        LINK_LIBS+=( -L"$SODIUM_PREFIX/lib" -lsodium )
    fi
fi

# ── Optional: Apache Arrow (Parquet + Feather/IPC), OFF by default ────────
# Enable with:  BANTU_ARROW=1 bash build-mac.sh
# Kept opt-in so the default binary gains NO new runtime dependency (Arrow is a
# heavy library). When enabled we link the shared libarrow/libparquet from brew.
if [ "${BANTU_ARROW:-0}" = "1" ]; then
    # Arrow 14+ headers need C++20 (std::span/popcount). The last -std wins, so
    # appending here upgrades the whole (opt-in) build; the default stays C++17.
    CPP_FLAGS+=( -std=c++20 )
    if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists arrow parquet; then
        echo "  Apache Arrow: $(pkg-config --modversion arrow) (Parquet + Feather)"
        CPP_FLAGS+=( -DBANTU_ARROW $(pkg-config --cflags arrow parquet) )
        LINK_LIBS+=( $(pkg-config --libs arrow parquet) )
    else
        ARROW_PREFIX="$(brew --prefix apache-arrow 2>/dev/null || true)"
        if [ -z "$ARROW_PREFIX" ] || [ ! -f "$ARROW_PREFIX/include/arrow/api.h" ]; then
            die "BANTU_ARROW=1 but Apache Arrow not found" \
                "Install it first:  brew install apache-arrow"
        fi
        echo "  Apache Arrow: $ARROW_PREFIX (Parquet + Feather)"
        CPP_FLAGS+=( -DBANTU_ARROW -I"$ARROW_PREFIX/include" )
        LINK_LIBS+=( -L"$ARROW_PREFIX/lib" -larrow -lparquet )
    fi
fi

# ═══════════════════════════════════════════════════════════════════════
#  Build
# ═══════════════════════════════════════════════════════════════════════

hr
echo "  Bantu interpreter build (macOS)"
echo "  script dir: $SCRIPT_DIR"
echo "  brew prefix: $BREW_PREFIX"
echo "  g++ version: $(g++ --version  | head -1)"
hr

section "Checking build tools"
MISSING=()
for tool in g++ gcc; do
    command -v "$tool" >/dev/null 2>&1 || MISSING+=("$tool")
done
[ ${#MISSING[@]} -eq 0 ] || die "Missing tools: ${MISSING[*]} — Run: brew install gcc"
pass "All required build tools present."

section "Checking headers"
for hdr in sqlite3.h curl/curl.h; do
    check_header "$hdr"
done
rm -f /tmp/hdrtest

section "Cleaning previous build"
rm -rf build; mkdir -p build

OBJECTS=()
for src in "${SOURCES[@]}"; do
    obj="build/$(basename "${src%.cpp}").o"
    compile_one "$src" "$obj"
    OBJECTS+=( "$obj" )
done

section "Linking build/bantu"
if g++ "${CPP_FLAGS[@]}" \
        "${OBJECTS[@]}" \
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

section "otool -L build/bantu"
otool -L build/bantu 2>&1 || true

echo
hr
echo "  ✓ Build complete: $(pwd)/build/bantu"
hr

BIN_NAME=bantu
hr_or_line() { hr; }

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
