// ════════════════════════════════════════════════════════════════════════
//  bkit_compose_test.b — ColumnTransformer, and the Pipeline stress that
//  K1 deferred to here.
//
//  The gate that matters for a composition is EXACTNESS against the
//  hand-written composition it replaces. A ColumnTransformer that is merely
//  plausible -- right shape, reasonable numbers -- is the failure mode worth
//  building against, because a block in the wrong place produces a model that
//  trains fine and predicts from the wrong features.
//
//  So the central assertions here build the same output twice: once through
//  ColumnTransformer, once by slicing the columns and concatenating by hand,
//  and require equality to 1e-12 with the SAME column order.
//
//  Run:  bantu run tests/bkit_compose_test.b
// ════════════════════════════════════════════════════════════════════════

include "../numba/numba.b" as np;
include "../bkit/base.b" as base;
include "../bkit/preprocess.b" as pre;
include "../bkit/linear.b" as lin;
include "../bkit/compose.b" as compose;
include "../arctic/arctic.b" as ar;

$R = {"pass": 0, "fail": 0};
def ok($cond, $name) {
    if ($cond) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else { $R.fail = $R.fail + 1; print("  FAIL  " + $name); }
}
def eq($got, $want, $name) {
    if (str($got) == str($want)) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name);
        print("          got:  " + str($got));
        print("          want: " + str($want));
    }
}
def close($got, $want, $tol, $name) {
    if (abs($got - $want) <= $tol) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    else {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name + "  got " + str($got) + " want " + str($want));
    }
}
// A 0-d reduction is an ndarray HANDLE and every handle is truthy, so it has to
// be extracted before it can be tested (numba N41).
def sc($x) { return nd_get(nd_reshape($x, [1]), [0]); }
def sameArr($a, $b, $tol, $name) {
    if (nd_shape($a) != nd_shape($b)) {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name + "  shape " + str(nd_shape($a)) + " vs "
              + str(nd_shape($b)));
        return null;
    }
    ok(nd_allclose($a, $b, $tol, $tol), $name);
    return null;
}
def raises($fn, $name) {
    try { $fn(); $R.fail = $R.fail + 1; print("  FAIL  " + $name + " (no error raised)"); }
    catch ($e) { $R.pass = $R.pass + 1; print("  ok    " + $name); }
    return null;
}
// A throwing assertion must not abort the remaining ones (K2's lesson).
def safe_ok($fn, $name) {
    try { ok($fn(), $name); }
    catch ($e) {
        $R.fail = $R.fail + 1;
        print("  FAIL  " + $name + "  threw: " + str($e));
    }
    return null;
}

np.seed(20261115);

print("========================================");
print("  bkit/compose.b");
print("========================================");

// A fixed matrix: 3 numeric columns and one small-integer "categorical".
$X = nd([[ 1.0,  10.0, 100.0, 0.0],
         [ 2.0,  20.0, 200.0, 1.0],
         [ 3.0,  30.0, 300.0, 2.0],
         [ 4.0,  40.0, 400.0, 1.0],
         [ 5.0,  50.0, 500.0, 0.0],
         [ 6.0,  60.0, 600.0, 2.0]], "f64");

print("");
print("== tier 1: it composes ==");
$ct = compose.ColumnTransformer([
    ["scale", pre.StandardScaler(null, null), [0, 1]],
    ["bin",   pre.Binarizer(250.0),           [2]]
], null);
$Z = $ct.fit_transform($X, null);
eq(nd_shape($Z), [6, 3], "two specs over three columns give a [6,3] output");
eq($ct.n_features_out(), 3, "n_features_out agrees");
eq($ct.params_.remainder, "drop", "remainder defaults to drop, so column 3 is gone");
eq($ct.output_ranges(), [["scale", 0, 2], ["bin", 2, 3]],
   "output_ranges attributes every output column to the spec that made it");

// EXACTNESS: the same thing by hand.
$byHand = nd_concat([
    pre.StandardScaler(null, null).fit_transform(nd_take($X, nd([0,1], "i64"), 1), null),
    pre.Binarizer(250.0).fit_transform(nd_take($X, nd([2], "i64"), 1), null)
], 1);
sameArr($Z, $byHand, 1.0e-12, "and is EXACTLY the hand-written composition");

print("");
print("== spec order decides column order, not column index ==");
// The specs are given in REVERSE column order on purpose: the output must
// follow the specs, because that is the only order a caller can predict
// without knowing each transformer's width.
$rev = compose.ColumnTransformer([
    ["late",  pre.Binarizer(250.0),           [2]],
    ["early", pre.StandardScaler(null, null),  [0, 1]]
], null);
$Zr = $rev.fit_transform($X, null);
eq($rev.output_ranges(), [["late", 0, 1], ["early", 1, 3]],
   "a spec naming column 2 first puts its block first");
sameArr(nd_take($Zr, nd([0], "i64"), 1), nd_take($Z, nd([2], "i64"), 1), 1.0e-12,
        "the blocks themselves are unchanged, only their order");

print("");
print("== remainder ==");
$pass = compose.ColumnTransformer([
    ["scale", pre.StandardScaler(null, null), [0]]
], "passthrough");
$Zp = $pass.fit_transform($X, null);
eq(nd_shape($Zp), [6, 4], "passthrough keeps the three untouched columns");
eq($pass.output_ranges(), [["scale", 0, 1], ["remainder", 1, 4]],
   "and output_ranges names the remainder block");
// The passthrough block must be X's own columns, in ASCENDING index order --
// not whatever order a dict iteration produced.
sameArr(nd_take($Zp, nd([1,2,3], "i64"), 1), nd_take($X, nd([1,2,3], "i64"), 1),
        1.0e-12, "the passthrough columns are X's own values, in X's own order");
eq($pass.n_features_out(), 4, "n_features_out counts the remainder");

$drop = compose.ColumnTransformer([["s", pre.StandardScaler(null,null), [0]]], "drop");
eq(nd_shape($drop.fit_transform($X, null)), [6, 1], "drop keeps only the named block");

print("");
print("== a column may feed TWO transformers, but not itself twice ==");
$twice = compose.ColumnTransformer([
    ["raw",    pre.Binarizer(3.5),             [0]],
    ["scaled", pre.StandardScaler(null, null), [0]]
], null);
$Zt = $twice.fit_transform($X, null);
eq(nd_shape($Zt), [6, 2], "column 0 legitimately feeds two specs");
// Within one spec it is an error: the block would silently widen.
raises(def() {
    return compose.ColumnTransformer([["d", pre.StandardScaler(null,null), [0, 0]]], null);
}, "the same column twice in ONE spec is refused");
// ...but only at fit, because resolution needs to know how wide X is.
raises(def() {
    $c = compose.ColumnTransformer([["d", pre.StandardScaler(null,null), [1, 1]]], null);
    return $c.fit($X, null);
}, "and refused at fit, where the columns are resolved");

print("");
print("== negative indices count from the right ==");
$neg = compose.ColumnTransformer([["last", pre.Binarizer(1.5), [-1]]], null);
sameArr($neg.fit_transform($X, null),
        pre.Binarizer(1.5).fit_transform(nd_take($X, nd([3], "i64"), 1), null),
        1.0e-12, "-1 addresses the last column");

print("");
print("== a 1-d block is widened, not refused ==");
// OrdinalEncoder returns [n, k]; a transformer that returns a flat column (and
// some do) must still compose. _as2d is the one place that rule lives.
$oe = compose.ColumnTransformer([
    ["ord", pre.OrdinalEncoder(null, null), [3]],
    ["num", pre.MaxAbsScaler(),             [0]]
], null);
$Zo = $oe.fit_transform($X, null);
eq(nd_shape($Zo), [6, 2], "an encoder block composes with a scaler block");

print("");
print("== a one-hot block makes the width unpredictable, which is the point ==");
$oh = compose.ColumnTransformer([
    ["cat", pre.OneHotEncoder(null), [3]],
    ["num", pre.StandardScaler(null, null), [0, 1, 2]]
], null);
$Zh = $oh.fit_transform($X, null);
// column 3 holds {0,1,2} -> three one-hot columns, then three scaled -> 6.
eq(nd_shape($Zh), [6, 6], "three categories widen to three columns");
eq($oh.output_ranges(), [["cat", 0, 3], ["num", 3, 6]],
   "and output_ranges is how you find the numeric block again");
safe_ok(def() { return sc(nd_all(nd_equal(nd_sum(nd_take($Zh, nd([0,1,2],"i64"), 1), 1, null),
                                          nd_ones([6], "f64"), null), null, null)) == true; },
        "every one-hot row sums to exactly 1");

print("");
print("== a nested Pipeline is just another transformer ==");
$nest = compose.ColumnTransformer([
    ["chain", base.Pipeline([["imp", pre.SimpleImputer("mean", null)],
                             ["sc",  pre.StandardScaler(null, null)]]), [0, 1]],
    ["plain", pre.MaxAbsScaler(), [2]]
], null);
$Zn = $nest.fit_transform($X, null);
eq(nd_shape($Zn), [6, 3], "a Pipeline fits inside a spec");
sameArr(nd_take($Zn, nd([0,1], "i64"), 1),
        base.Pipeline([["imp", pre.SimpleImputer("mean", null)],
                       ["sc",  pre.StandardScaler(null, null)]])
            .fit_transform(nd_take($X, nd([0,1], "i64"), 1), null),
        1.0e-12, "and produces exactly what the Pipeline alone would");

print("");
print("== a ColumnTransformer inside a Pipeline, ending in an estimator ==");
$y = nd([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "f64");
$full = base.Pipeline([
    ["prep", compose.ColumnTransformer([
                ["sc", pre.StandardScaler(null, null), [0, 1]],
                ["mx", pre.MinMaxScaler(null, null),   [2]]], null)],
    ["reg",  lin.LinearRegression(null)]
]);
$full.fit($X, $y);
$pred = $full.predict($X);
eq(nd_shape($pred), [6], "the composition predicts");
close(sc(nd_max(nd_abs(nd_subtract($pred, $y, null), null), null, null)), 0.0, 1.0e-8,
      "and fits this exactly-linear target to 1e-8");
eq($full.task(), "regressor", "the pipeline's task is its final step's");

print("");
print("== transform after fit uses the FITTED statistics ==");
// The second matrix has a different mean. If transform refitted, its output
// would be centred -- the classic silent leak.
$X2 = nd_add($X, 100.0, null);
$ct2 = compose.ColumnTransformer([["sc", pre.StandardScaler(null, null), [0, 1]]], null);
$ct2.fit($X, null);
$Zb = $ct2.transform($X2);
ok(abs(sc(nd_mean($Zb, null, null))) > 1.0,
   "transform does NOT recentre on the new data (no refit leak)");
sameArr($Zb,
        pre.StandardScaler(null, null).fit(nd_take($X, nd([0,1],"i64"), 1), null)
            .transform(nd_take($X2, nd([0,1],"i64"), 1)),
        1.0e-12, "and matches the same scaler used by hand");

print("");
print("== clone is deep, so a grid search cannot leak state ==");
$orig = compose.ColumnTransformer([["sc", pre.StandardScaler(null, null), [0, 1]]], null);
$cl   = $orig.clone();
ok(!$cl.fitted(), "a clone starts unfitted");
$cl.fit($X2, null);
ok(!$orig.fitted(), "fitting the clone leaves the original unfitted");
$orig.fit($X, null);
ok(abs(sc(nd_mean($orig.transform($X), null, null))) < 1.0e-12,
   "and the original, fitted on its own data, still centres it");
// The column list is a Bantu list and lists are shared by REFERENCE, so a
// shallow clone would let one composition's spec reach the other's.
$cl2 = $orig.clone();
ok($cl2.get_transformer("sc") != $orig.get_transformer("sc"),
   "the clone's transformer is a different object");

print("");
print("== get_params / set_params address sub-transformers ==");
$gp = compose.ColumnTransformer([
    ["sc",  pre.StandardScaler(null, null), [0]],
    ["bin", pre.Binarizer(5.0),             [1]]
], null);
ok(contains($gp.get_params().keys(), "bin__threshold"),
   "get_params exposes \"<name>__<param>\"");
ok(contains($gp.get_params().keys(), "remainder"), "and remainder itself");
$gp.set_params({"bin__threshold": 25.0});
eq($gp.get_transformer("bin").get_params().threshold, 25.0,
   "set_params reaches into a sub-transformer");
$gp.set_params({"remainder": "passthrough"});
eq($gp.params_.remainder, "passthrough", "and sets remainder");
// An unknown key must RAISE. A grid search that silently ignored one would
// report a best score for a configuration it never tried (K24).
raises(def() { return $gp.set_params({"nope": 1.0}); },
       "an unknown parameter is refused, not ignored");
raises(def() { return $gp.set_params({"missing__threshold": 1.0}); },
       "an unknown transformer NAME is refused");
raises(def() { return $gp.set_params({"remainder": "keep"}); },
       "an invalid remainder is refused");
$gp.set_params({"bin__threshold": 1.0});
ok(!$gp.fitted(), "set_params invalidates the fit");

print("");
print("== construction refuses malformed specs, where the caller wrote them ==");
raises(def() { return compose.ColumnTransformer([["a", pre.MaxAbsScaler()]], null); },
       "a spec that is not [name, estimator, columns] is refused");
raises(def() { return compose.ColumnTransformer([[1.0, pre.MaxAbsScaler(), [0]]], null); },
       "a non-string spec name is refused");
raises(def() { return compose.ColumnTransformer([["a", null, [0]]], null); },
       "a null estimator is refused");
raises(def() { return compose.ColumnTransformer([["dup", pre.MaxAbsScaler(), [0]],
                                                 ["dup", pre.Binarizer(1.0), [1]]], null); },
       "two specs with the same name are refused (names address parameters)");
raises(def() { return compose.ColumnTransformer([["a", pre.MaxAbsScaler(), []]], null); },
       "an empty column list is refused");
raises(def() { return compose.ColumnTransformer([["a", pre.MaxAbsScaler(), [0]]], "maybe"); },
       "an unknown remainder is refused");
raises(def() { return compose.ColumnTransformer([], "drop").fit($X, null); },
       "no transformers plus drop would give zero columns, and says so");
eq(nd_shape(compose.ColumnTransformer([], "passthrough").fit_transform($X, null)), [6, 4],
   "but no transformers plus passthrough is the identity");

print("");
print("== fit refuses what it cannot resolve ==");
raises(def() { return compose.ColumnTransformer([["a", pre.MaxAbsScaler(), [9]]], null)
                             .fit($X, null); },
       "a column index past the end is refused, naming the index");
raises(def() { return compose.ColumnTransformer([["a", pre.MaxAbsScaler(), ["name"]]], null)
                             .fit($X, null); },
       "a column NAME against an array is refused, and says to pass indices");
raises(def() { return compose.ColumnTransformer([["a", pre.MaxAbsScaler(), [1.5]]], null)
                             .fit($X, null); },
       "a fractional column index is refused");
raises(def() { return compose.ColumnTransformer([["a", pre.MaxAbsScaler(), [0]]], null)
                             .fit(nd_zeros([0, 4], "f64"), null); },
       "an empty X is refused rather than fitted to undefined statistics");
raises(def() { return compose.ColumnTransformer([["a", pre.MaxAbsScaler(), [0]]], null)
                             .fit(nd([1.0, 2.0], "f64"), null); },
       "a 1-d X is refused, naming the shape");

print("");
print("== transform refuses a differently-shaped X ==");
$fit1 = compose.ColumnTransformer([["sc", pre.StandardScaler(null,null), [0,1]]], null);
$fit1.fit($X, null);
raises(def() { return $fit1.transform(nd_zeros([3, 7], "f64")); },
       "a different column COUNT at transform time is refused");
raises(def() { return compose.ColumnTransformer([["sc", pre.StandardScaler(null,null), [0]]], null)
                             .transform($X); },
       "transform before fit says to call fit first");
raises(def() { return compose.ColumnTransformer([["sc", pre.StandardScaler(null,null), [0]]], null)
                             .output_ranges(); },
       "output_ranges before fit says the same");

print("");
print("== get_transformer ==");
eq($fit1.get_transformer("sc").name_, "StandardScaler", "get_transformer finds a spec");
raises(def() { return $fit1.get_transformer("nope"); },
       "and names what it does have when it cannot");

print("");
print("== tier 4: the 50-stage Pipeline K1 deferred to here ==");
// Fifty chained transforms. The point is not speed: it is that the chain is
// composed in ORDER and that nothing accumulates per step -- a Pipeline that
// re-fitted, or copied the whole matrix per step, shows up here.
$steps = [];
$i = 0;
while ($i < 50) {
    append($steps, ["s" + str($i), pre.StandardScaler(null, null)]);
    $i = $i + 1;
}
$long = base.Pipeline($steps);
$Xw = np.random([200, 8], -5.0, 5.0);
$t0 = clock();
$Zl = $long.fit_transform($Xw, null);
$ms = clock() - $t0;
eq(nd_shape($Zl), [200, 8], "a 50-stage pipeline preserves the shape");
// Standardising an already-standardised matrix is idempotent, so after fifty
// passes the result must still be exactly standardised -- which is also a
// check that every stage ran rather than being skipped.
ok(abs(sc(nd_mean($Zl, null, null))) < 1.0e-10,
   "and the output is still centred after 50 passes");
close(sc(nd_mean(nd_std($Zl, 0, true, 0), null, null)), 1.0, 1.0e-10,
      "and still unit-variance, so no stage was a no-op");
eq(len($long.get_params().keys()) > 50, true,
   "get_params walks all fifty steps");
$long.set_params({"s49__with_mean": false});
eq($long.get_step("s49").get_params().with_mean, false,
   "and set_params addresses the fiftieth");
print("        50-stage fit_transform on [200,8]: " + str($ms) + " ms");
ok($ms < 3000, "fifty stages stay under 3 s (no per-step quadratic blowup)");

print("");
print("== tier 4: a wide ColumnTransformer ==");
// Twenty specs over twenty columns, plus a passthrough tail.
$wide = [];
$i = 0;
while ($i < 20) {
    append($wide, ["c" + str($i), pre.StandardScaler(null, null), [$i]]);
    $i = $i + 1;
}
$ctw = compose.ColumnTransformer($wide, "passthrough");
$Xb = np.random([5000, 30], -10.0, 10.0);
$t1 = clock();
$Zw = $ctw.fit_transform($Xb, null);
$ms2 = clock() - $t1;
eq(nd_shape($Zw), [5000, 30], "20 specs + 10 passthrough columns give 30 out");
eq($ctw.output_ranges()[19], ["c19", 19, 20], "the twentieth block is where it should be");
eq($ctw.output_ranges()[20], ["remainder", 20, 30], "and the remainder follows it");
sameArr(nd_take($Zw, nd([20,21,22,23,24,25,26,27,28,29], "i64"), 1),
        nd_take($Xb, nd([20,21,22,23,24,25,26,27,28,29], "i64"), 1), 1.0e-12,
        "the passthrough tail is X's own columns, untouched");
print("        20 specs over [5000,30]: " + str($ms2) + " ms");
ok($ms2 < 2000, "and stays under 2 s");

print("");
print("== the composition is reproducible from the seed alone (C4) ==");
np.seed(99);
$a1 = compose.ColumnTransformer([["sc", pre.StandardScaler(null,null), [0,1]]], null)
             .fit_transform($X, null);
np.seed(99);
$a2 = compose.ColumnTransformer([["sc", pre.StandardScaler(null,null), [0,1]]], null)
             .fit_transform($X, null);
ok(nd_tobytes($a1, "f64") == nd_tobytes($a2, "f64"),
   "two fits from one seed are bit-identical");

print("");
print("== an arctic DataFrame: columns by NAME ==");
// This is the only part of bkit that touches arctic (design doc section 8), so
// it gets its own block. compose.b does not include arctic: it duck-types on
// the fields a DataFrame has, because importing a dataframe package into the
// composition layer would make every bkit user pay for one.
$df = ar.dataframe({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    "b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                    "c": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
                    "d": [0.0, 1.0, 2.0, 1.0, 0.0, 2.0]}, null);
eq($df.columns(), ["a", "b", "c", "d"], "the frame has the columns the test expects");

$fct = compose.ColumnTransformer([
    ["num", pre.StandardScaler(null, null), ["a", "b"]],
    ["cut", pre.Binarizer(250.0),           ["c"]]
], null);
$Zf = $fct.fit_transform($df, null);
eq(nd_shape($Zf), [6, 3], "a frame composes by column name");
// It must agree EXACTLY with the array path over the same values.
sameArr($Zf, $Z, 1.0e-12,
        "and is bit-equal to the same composition over the equivalent array");

// Names and indices are the same mechanism after resolution, so mixing them
// inside one spec is allowed and must agree.
$mix = compose.ColumnTransformer([["m", pre.StandardScaler(null,null), ["a", 1]]], null);
sameArr($mix.fit_transform($df, null),
        pre.StandardScaler(null, null).fit_transform(nd_take($X, nd([0,1],"i64"), 1), null),
        1.0e-12, "a spec may mix a name and an index");

$fpass = compose.ColumnTransformer([["num", pre.MaxAbsScaler(), ["b"]]], "passthrough");
eq(nd_shape($fpass.fit_transform($df, null)), [6, 4],
   "remainder works over a frame too");

raises(def() { return compose.ColumnTransformer([["x", pre.MaxAbsScaler(), ["zz"]]], null)
                             .fit($df, null); },
       "an unknown column name is refused, listing the frame's columns");

// The silent-wrong-answer case this exists to prevent: names are resolved to
// POSITIONS at fit time, so a frame whose columns were reordered in between
// would transform different data under the same names.
$fsel = compose.ColumnTransformer([["num", pre.StandardScaler(null,null), ["a", "b"]]], null);
$fsel.fit($df, null);
$swapped = $df.select(["b", "a", "c", "d"]);
raises(def() { return $fsel.transform($swapped); },
       "a REORDERED frame is refused at transform, not silently mistransformed");
safe_ok(def() { return nd_shape($fsel.transform($df)) == [6, 2]; },
        "while the same frame still transforms");

print("");
print("========================================");
print("  PASS: " + str($R.pass) + "   FAIL: " + str($R.fail));
print("========================================");
if ($R.fail > 0) { print("  RESULT: FAILURES PRESENT"); }
if ($R.fail == 0) { print("  RESULT: ALL GREEN"); }
