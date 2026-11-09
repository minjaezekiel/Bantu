// ════════════════════════════════════════════════════════════════════════════
//  btorch/datasets.b — the IDX reader, and MNIST (phase 10)
//
//  IDX is the format MNIST ships in. Its header is BIG-ENDIAN, which is the
//  one thing that makes it worth a reader rather than a one-liner:
//
//      byte 0   0
//      byte 1   0
//      byte 2   the element type code (0x08 = unsigned byte)
//      byte 3   the number of dimensions
//      then     one u32 BIG-ENDIAN dimension per axis
//      then     the data, C-order
//
//  nd_frombytes takes a `be` suffix, so the header needs no hand-rolled byte
//  arithmetic -- which is the whole reason it reads cleanly.
//
//  The images are loaded in CHUNKS. 60,000 x 784 as f64 is 376 MB (f64 is
//  btorch's only float -- DECISIONS B6), and the obvious
//  read-whole-file-then-convert costs that twice at once: once as the i64
//  array nd_frombytes produces from u8 input, once as the f64 copy. Chunking
//  caps the extra at one chunk.
// ════════════════════════════════════════════════════════════════════════════

include "btorch.b" as bt;
include "data.b" as btdata;

def _need() {
    if (nd_frombytes == null) {
        throw "btorch.datasets needs numba's nd_* builtins -- the bantu binary "
            + "is older than this library. Run 'bantu doctor'.";
    }
    return null;
}

// The IDX element type codes, and how many bytes each takes.
def _idxElem($code) {
    if ($code == 8)  { return ["u8",    1]; }
    if ($code == 9)  { return ["i8",    1]; }
    if ($code == 11) { return ["i16be", 2]; }
    if ($code == 12) { return ["i32be", 4]; }
    if ($code == 13) { return ["f32be", 4]; }
    if ($code == 14) { return ["f64be", 8]; }
    throw "btorch.datasets: unknown IDX element type code " + str($code)
        + " (expected 8, 9, 11, 12, 13 or 14)";
}

def _u8At($s, $at) { return nd_get(nd_frombytes(substr($s, $at, 1), "u8", [1]), [0]); }
def _u32beAt($s, $at) { return nd_get(nd_frombytes(substr($s, $at, 4), "u32be", [1]), [0]); }

// read_idx(path) -> { "data": <f64 array>, "shape": [...], "count": n }
//
// Always returns f64, because everything downstream of here is f64. The
// conversion happens chunk by chunk rather than all at once.
def read_idx($path) {
    _need();
    if (type($path) != "string") { throw "btorch.datasets.read_idx(path): path must be a string"; }
    if (!file_exists($path)) {
        throw "btorch.datasets.read_idx: no such file '" + $path + "'";
    }
    $raw = readfile($path, "rb");
    $n = len($raw);
    if ($n < 4) {
        throw "btorch.datasets.read_idx: '" + $path + "' is " + str($n)
            + " bytes, too short to hold an IDX header";
    }
    if (_u8At($raw, 0) != 0 || _u8At($raw, 1) != 0) {
        throw "btorch.datasets.read_idx: '" + $path + "' does not start with the "
            + "two zero bytes an IDX header begins with -- is it still gzipped? "
            + "(a .gz file starts 1f 8b)";
    }
    $elem = _idxElem(_u8At($raw, 2));
    $ndim = _u8At($raw, 3);
    if ($ndim < 1 || $ndim > 4) {
        throw "btorch.datasets.read_idx: '" + $path + "' claims " + str($ndim)
            + " dimensions, which is not a shape this reader expects (1 to 4)";
    }
    $hdr = 4 + 4 * $ndim;
    if ($n < $hdr) {
        throw "btorch.datasets.read_idx: '" + $path + "' is TRUNCATED -- its "
            + str($ndim) + "-dimensional header needs " + str($hdr)
            + " bytes but the file is " + str($n);
    }
    $shape = [];
    $total = 1;
    $d = 0;
    while ($d < $ndim) {
        $dim = _u32beAt($raw, 4 + 4 * $d);
        push($shape, $dim);
        $total = $total * $dim;
        $d = $d + 1;
    }
    $want = $hdr + $total * $elem[1];
    if ($n != $want) {
        throw "btorch.datasets.read_idx: '" + $path + "' declares shape "
            + str($shape) + " of " + str($elem[1]) + "-byte elements, which needs "
            + str($want) + " bytes, but the file is " + str($n)
            + " -- it is truncated or not the file it claims to be";
    }

    // Chunked: one chunk's worth of intermediate, not one file's worth.
    $out = nd_zeros([$total], "f64");
    $CHUNK = 2000000;
    $at = 0;
    while ($at < $total) {
        $cnt = $CHUNK;
        if ($at + $cnt > $total) { $cnt = $total - $at; }
        $bytes = substr($raw, $hdr + $at * $elem[1], $cnt * $elem[1]);
        $vals = nd_astype(nd_frombytes($bytes, $elem[0], [$cnt]), "f64");
        nd_add($vals, 0.0, nd_slice($out, [[$at, $at + $cnt, 1]]));
        $at = $at + $cnt;
    }
    return {"data": nd_reshape($out, $shape), "shape": $shape, "count": $shape[0]};
}

// ── MNIST ───────────────────────────────────────────────────────────────────
// The four canonical files, by their canonical names. `dir` defaults to
// ~/.bantu/datasets/mnist, which samples/btorch/get_mnist.sh populates.
$_MNIST_FILES = {
    "train_x": "train-images-idx3-ubyte",
    "train_y": "train-labels-idx1-ubyte",
    "test_x":  "t10k-images-idx3-ubyte",
    "test_y":  "t10k-labels-idx1-ubyte"
};

// The builtin is env(), not getenv(), and it returns "" rather than null for
// an unset variable.
def mnist_dir($dir) {
    if ($dir != null) { return $dir; }
    $e = env("BANTU_MNIST_DIR");
    if ($e != "") { return $e; }
    return env("HOME") + "/.bantu/datasets/mnist";
}

// mnist(dir, normalize) -> { train_x, train_y, test_x, test_y }
//
//   train_x  [60000, 784] f64
//   train_y  [60000]      f64 (class indices 0..9)
//
// normalize: "standard" (default) applies MNIST's usual mean/std, which is
// what makes the MLP converge in a handful of epochs rather than tens;
// "unit" is x/255; "raw" leaves the bytes alone.
def mnist($dir, $normalize) {
    _need();
    $d = mnist_dir($dir);
    $norm = $normalize;
    if ($norm == null) { $norm = "standard"; }

    $missing = [];
    for $k, $f in $_MNIST_FILES.items() {
        if (!file_exists($d + "/" + $f)) { push($missing, $f); }
    }
    if (len($missing) > 0) {
        throw "btorch.datasets.mnist: " + str(len($missing)) + " of 4 MNIST files "
            + "are missing from '" + $d + "' (" + str($missing) + "). "
            + "Fetch them with samples/btorch/get_mnist.sh, or point "
            + "BANTU_MNIST_DIR at a directory that has them.";
    }

    $out = {};
    for $k, $f in $_MNIST_FILES.items() {
        $r = read_idx($d + "/" + $f);
        $a = $r.data;
        $s = nd_shape($a);
        if (len($s) == 3) {
            // [N, 28, 28] -> [N, 784]. Images only.
            $a = nd_reshape($a, [$s[0], $s[1] * $s[2]]);
            if ($norm == "unit") { nd_divide($a, 255.0, $a); }
            if ($norm == "standard") {
                // (x/255 - 0.1307) / 0.3081 -- MNIST's own mean and standard
                // deviation, in place so a 376 MB array is not copied twice.
                nd_divide($a, 255.0, $a);
                nd_subtract($a, 0.1307, $a);
                nd_divide($a, 0.3081, $a);
            }
        }
        $out[$k] = $a;
    }

    if (nd_shape($out.train_x)[0] != nd_size($out.train_y)) {
        throw "btorch.datasets.mnist: " + str(nd_shape($out.train_x)[0])
            + " training images but " + str(nd_size($out.train_y)) + " labels";
    }
    if (nd_shape($out.test_x)[0] != nd_size($out.test_y)) {
        throw "btorch.datasets.mnist: " + str(nd_shape($out.test_x)[0])
            + " test images but " + str(nd_size($out.test_y)) + " labels";
    }
    return $out;
}

// A [N, 784] batch reshaped to [N, 1, 28, 28], which is what Conv2d wants.
def as_images($flat) {
    _need();
    if (bt.is_tensor($flat)) { return bt.reshape($flat, [nd_shape($flat.data)[0], 1, 28, 28]); }
    return nd_reshape($flat, [nd_shape($flat)[0], 1, 28, 28]);
}

// ── accuracy ────────────────────────────────────────────────────────────────
// argmax over the class axis, compared with the labels. Kept here rather than
// written out at every call site, because an off-by-one in the axis gives a
// plausible-looking 10% and nothing else to go on.
def accuracy($logits, $labels) {
    _need();
    $pred = nd_argmax($logits.data, 1, null);
    $lab = nd_astype($labels, "i64");
    if (nd_size($pred) != nd_size($lab)) {
        throw "btorch.datasets.accuracy: " + str(nd_size($pred)) + " predictions for "
            + str(nd_size($lab)) + " labels";
    }
    $hits = nd_sum(nd_astype(nd_equal(nd_astype($pred, "i64"), $lab, null), "f64"), null, null);
    return nd_get(nd_reshape($hits, [1]), [0]) / (nd_size($lab) * 1.0);
}
