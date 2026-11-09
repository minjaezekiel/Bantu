#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
#  get_mnist.sh — fetch MNIST into ~/.bantu/datasets/mnist (or $1).
#
#  The dataset is 54 MB uncompressed and does NOT belong in the repository,
#  so it is fetched on demand and verified by SHA-256. The digests below are
#  the canonical ones; a mirror that serves something else fails here rather
#  than three hours into a training run.
# ─────────────────────────────────────────────────────────────────────────
set -eu

DIR="${1:-${BANTU_MNIST_DIR:-$HOME/.bantu/datasets/mnist}}"
BASE="${MNIST_BASE:-https://ossci-datasets.s3.amazonaws.com/mnist}"

# file:sha256 of the DECOMPRESSED idx file
FILES=(
  "train-images-idx3-ubyte:ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db"
  "train-labels-idx1-ubyte:65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5"
  "t10k-images-idx3-ubyte:0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7"
  "t10k-labels-idx1-ubyte:ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2"
)

sha() {
    if   command -v shasum    >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
    elif command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    else echo "no-hasher"; fi
}

mkdir -p "$DIR"
echo "  MNIST -> $DIR"

for entry in "${FILES[@]}"; do
    name="${entry%%:*}"
    want="${entry##*:}"
    path="$DIR/$name"

    if [ -f "$path" ]; then
        got="$(sha "$path")"
        if [ "$got" = "$want" ] || [ "$got" = "no-hasher" ]; then
            echo "  [ OK ]   $name (already present)"
            continue
        fi
        echo "  [ BAD ]  $name has the wrong digest — refetching"
        rm -f "$path"
    fi

    echo "  [ GET ]  $name"
    curl -fsSL -o "$path.gz" "$BASE/$name.gz"
    gunzip -f "$path.gz"

    got="$(sha "$path")"
    if [ "$got" != "$want" ] && [ "$got" != "no-hasher" ]; then
        echo "  [ FAIL ] $name digest mismatch"
        echo "           expected $want"
        echo "           got      $got"
        exit 1
    fi
    echo "  [ OK ]   $name"
done

echo
echo "  Done. Run the gates with:"
echo "    bantu run samples/btorch/mnist_mlp.b"
echo "    bantu run samples/btorch/mnist_cnn.b"
