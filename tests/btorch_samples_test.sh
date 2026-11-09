#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
#  btorch_samples_test.sh — phase 10's CI gate for examples.
#
#  Two jobs:
#
#    1. Run every sample in samples/btorch/ and require "RESULT: ALL GREEN".
#    2. EXTRACT every ```bantu block from docs/btorch.md AND
#       docs/btorch-tutorial.md, and run every one.
#
#  The second exists because of numba's B5 correction: "documented examples
#  run in CI" was once claimed in that project and was false — two of nine
#  failed the first time a runner was actually pointed at them. A claim like
#  that is worth nothing unless something executes it, so this does.
#
#  Doc blocks are SNIPPETS, not programs: they use names the surrounding
#  prose introduced. The runner therefore prepends a preamble defining those
#  names. A block that demonstrates an ERROR is marked in the doc with
#  `// EXPECT-RAISES` and is required to fail.
#
#  Usage:  BANTU=./bantu-src/compiler/build/bantu bash tests/btorch_samples_test.sh
# ─────────────────────────────────────────────────────────────────────────
set -u

BANTU="${BANTU:-bantu}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Resolve BANTU to an ABSOLUTE path. The doc blocks run from a temp directory
# so that a block writing "model.btsd" does not drop a file in the repository
# -- and a relative ./bantu-src/... would stop resolving the moment we cd.
case "$BANTU" in
    /*) ;;
    */*) BANTU="$ROOT/$BANTU" ;;
    *)  BANTU="$(command -v "$BANTU" || echo "$BANTU")" ;;
esac
[ -x "$BANTU" ] || { echo "  [FAIL] no executable bantu at '$BANTU'"; exit 1; }
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

pass=0; fail=0; skip=0
ok()      { pass=$((pass+1)); echo "  ok      $*"; }
bad()     { fail=$((fail+1)); echo "  FAIL    $*"; }
skipped() { skip=$((skip+1)); echo "  skip    $*"; }

echo "════════════════════════════════════════════════════════════════"
echo "  btorch samples and documented examples"
echo "  binary: $BANTU"
echo "════════════════════════════════════════════════════════════════"
echo
echo "── samples/btorch/*.b ──"

MNIST_DIR="${BANTU_MNIST_DIR:-$HOME/.bantu/datasets/mnist}"
have_mnist=1
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    [ -f "$MNIST_DIR/$f" ] || have_mnist=0
done

for s in samples/btorch/*.b; do
    name="$(basename "$s")"
    case "$name" in
        mnist_*)
            if [ "$have_mnist" = "0" ]; then
                skipped "$name (no MNIST in $MNIST_DIR — run samples/btorch/get_mnist.sh)"
                continue
            fi
            # The gates take minutes. Run them only when asked.
            if [ "${BTORCH_RUN_GATES:-0}" != "1" ]; then
                skipped "$name (set BTORCH_RUN_GATES=1 to run the accuracy gates)"
                continue
            fi
            ;;
    esac
    out="$("$BANTU" --quiet run "$s" 2>&1)"
    if echo "$out" | grep -q "RESULT: ALL GREEN"; then
        ok "$name"
    else
        bad "$name"
        echo "$out" | tail -8 | sed 's/^/            /'
    fi
done

echo
echo "── documented examples ──"

# The preamble. Every name the doc's snippets rely on, defined once.
cat > "$WORK/preamble.b" <<'PRE'
include "ROOT/btorch/btorch.b" as bt;
include "ROOT/btorch/nn.b" as nn;
include "ROOT/btorch/optim.b" as optim;
include "ROOT/btorch/data.b" as btdata;
include "ROOT/btorch/serialize.b" as ser;
include "ROOT/btorch/datasets.b" as ds;
bt.manual_seed(1);
$a = bt.tensor([[1.0, 2.0], [3.0, 4.0]], true);
$b = bt.tensor([[5.0, 6.0], [7.0, 8.0]], true);
$x = bt.tensor(nd_random_normal([8, 784]), false);
$y = bt.tensor(nd_random_normal([8, 1]), false);
$w = bt.tensor(nd_random_normal([4, 3]), true);
$idx = nd_astype(nd_random_int([8, 1], 0, 10), "i64");
$labels = nd_astype(nd_random_int([8], 0, 10), "i64");
$table = bt.tensor(nd_random_normal([10, 4]), true);
$tokenIds = nd_astype(nd_random_int([5], 0, 10), "i64");
$logits = bt.tensor(nd_random_normal([8, 10]), true);
$net = nn.Sequential([nn.Linear(784, 128, null), nn.ReLU(), nn.Linear(128, 10, null)]);
$model = $net;
$opt = optim.Adam($net.parameters(), 0.001, null, null, null);
$loss = nn.cross_entropy($net.forward($x), $labels, null);
// NOT $ds: that would shadow the `as ds` module alias above -- Bantu keeps
// variables, functions and aliases in one namespace, and the symptom is
// "ds.as_images: it holds null, not a function" in an unrelated block.
$dset = btdata.TensorDataset([nd_random_normal([64, 784]), nd_astype(nd_random_int([64], 0, 10), "f64")]);
$loader = btdata.DataLoader($dset, 16, true, false, 1);
$testx = nd_random_normal([8, 784]);
$testy = nd_astype(nd_random_int([8], 0, 10), "f64");
$batch = $loader.batch(0);
$n = 4;
$sched = optim.StepLR($opt, 10, 0.5);
def measure($net, $xx, $yy) { return 1.0; }
PRE
sed -i.bak "s|ROOT|$ROOT|g" "$WORK/preamble.b" && rm -f "$WORK/preamble.b.bak"

# Extract the blocks from every documentation file. awk emits each
# ```bantu ... ``` body to its own file, tagged with which doc it came from so
# a failure says where to look.
for doc in docs/btorch.md docs/btorch-tutorial.md; do
    tag="$(basename "$doc" .md)"
    awk -v dir="$WORK" -v tag="$tag" '
      /^```bantu$/ { n++; inblk=1; f=sprintf("%s/%s__%02d.b", dir, tag, n); next }
      /^```$/      { inblk=0; next }
      inblk        { print > f }
    ' "$doc"
    echo "  $doc: $(ls "$WORK/$tag"__*.b 2>/dev/null | wc -l | tr -d ' ') blocks"
done

for blk in "$WORK"/*__*.b; do
    [ -f "$blk" ] || continue
    name="$(basename "$blk" .b)"
    expect_raises=0
    grep -q "EXPECT-RAISES" "$blk" && expect_raises=1

    # Every block gets the preamble. A block's own `include "btorch/nn.b"` is
    # written for a reader whose btorch sits next to their code, so the path is
    # rewritten to this tree's -- which also checks that it names a real file.
    sed "s|include \"btorch/|include \"$ROOT/btorch/|g" "$blk" > "$WORK/blk.b"
    cat "$WORK/preamble.b" "$WORK/blk.b" > "$WORK/run.b"

    # Run from the work directory, so a block that writes "model.btsd" does
    # not drop a file in the repository.
    out="$(cd "$WORK" && "$BANTU" --quiet run "$WORK/run.b" 2>&1)"
    rc=$?
    failed=0
    echo "$out" | grep -qE "\[FATAL\]|\[error\]|RUNTIME ERROR|REFERENCE ERROR|SYNTAX ERROR" && failed=1
    [ "$rc" -ne 0 ] && failed=1

    if [ "$expect_raises" = "1" ]; then
        if [ "$failed" = "1" ]; then ok "$name (raises, as documented)"
        else bad "$name — documented as raising, but it did not"; fi
    else
        if [ "$failed" = "0" ]; then ok "$name"
        else
            bad "$name"
            echo "$out" | grep -E "\[FATAL\]|\[error\]" | head -3 | sed 's/^/            /'
        fi
    fi
done

echo
echo "════════════════════════════════════════════════════════════════"
echo "  PASS: $pass   FAIL: $fail   SKIP: $skip"
echo "════════════════════════════════════════════════════════════════"
if [ "$fail" -gt 0 ]; then echo "  RESULT: FAILURES PRESENT"; exit 1; fi
echo "  RESULT: ALL GREEN"
