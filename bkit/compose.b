// ════════════════════════════════════════════════════════════════════════════
//  bkit/compose.b — ColumnTransformer: different preprocessing per column.
//
//      include "bkit/compose.b" as compose;
//      include "bkit/preprocess.b" as pre;
//
//      $ct = compose.ColumnTransformer([
//          ["num", pre.StandardScaler(), [0, 1, 2]],
//          ["cat", pre.OneHotEncoder(null, null), [3]]
//      ], "drop");
//      $Z = $ct.fit_transform($X, null);
//
//  Each spec is `[name, estimator, columns]`. The blocks are concatenated in
//  SPEC ORDER -- not in column order -- because that is the only ordering a
//  caller can predict without knowing each transformer's output width.
//
//  ── why this file exists separately from base.b ────────────────────────────
//  `Pipeline` lives in `base.b` because the protocol test harness needs it and
//  because it depends on nothing but the contract. `ColumnTransformer` is the
//  ONE part of bkit that touches `arctic` (design doc section 9), so it is
//  kept out of the file that `btext` imports for the contract alone. Importing
//  base.b must not drag in a dataframe package.
//
//  ── what the column spec may be ───────────────────────────────────────────
//  Integer indices into X's columns, or -- when X is an arctic DataFrame --
//  column NAMES. Both resolve to integer indices against one materialised
//  matrix before any transformer runs, so there is exactly one code path after
//  resolution and the frame case cannot drift from the array case. A frame is
//  converted ONCE, not once per spec.
//
//  Design: docs/bkit-architecture.md section 7; decisions K37-K41.
// ════════════════════════════════════════════════════════════════════════════

include "../numba/numba.b" as np;
include "./base.b" as base;

// ── helpers ─────────────────────────────────────────────────────────────────

// Is this an arctic DataFrame? `type()` of ANY class instance is "instance"
// (the class registry is flat and type() does not discriminate), and Bantu has
// no `has_method`, so the test is on fields a DataFrame has and nothing else
// here does. Reading a field an instance does not define yields null rather
// than raising, which is what makes this work -- the same mechanism btorch's
// `is_dataset` uses with its `__bt_dataset` marker.
//
// Deliberately NOT a marker field added to arctic: this file is the only thing
// in bkit that knows arctic exists, and asking another package to carry a flag
// for one consumer inverts that. If a second consumer ever appears, a marker
// becomes the right answer.
def _isFrame($X) {
    if ($X == null) { return false; }
    if (type($X) != "instance") { return false; }
    return $X.names != null && $X.cols != null && $X.ncols != null;
}

def _whole($n) {
    return type($n) == "number" && $n == floor($n);
}

// What can be checked about a column list WITHOUT knowing how wide X is.
// This runs at construction, because the init comment's promise -- a malformed
// spec fails once, where the caller wrote it -- has to cover the column list
// too, and an empty list or a literally repeated index needs no X to spot.
//
// It cannot catch every duplicate: on a 4-column X, `[0, -4]` is the same
// column written two ways, and that needs the width. So _resolve checks again
// after resolution, and the two messages say which kind was found.
def _preCheckCols($cols, $who) {
    if ($cols == null) { throw $who + ": the column list is null"; }
    if (type($cols) != "list") {
        throw $who + ": columns must be a list, got " + type($cols);
    }
    if (len($cols) == 0) {
        throw $who + ": the column list is empty. A transformer with no columns "
              + "produces no output and is never what the caller meant -- drop "
              + "the spec instead.";
    }
    $seen = {};
    $i = 0;
    while ($i < len($cols)) {
        $c = $cols[$i];
        if (!_whole($c) && type($c) != "string") {
            throw $who + ": a column must be a whole number or a name, got "
                  + type($c) + " (" + str($c) + ")";
        }
        $key = type($c) + ":" + str($c);
        if (contains($seen.keys(), $key)) {
            throw $who + ": column " + str($c) + " is listed twice in the same "
                  + "spec, which would transform it twice";
        }
        $seen[$key] = true;
        $i = $i + 1;
    }
    return null;
}

// Resolve one spec's column list to integer indices against $ncols.
// $names is X's column names when X was a frame, else null.
def _resolve($cols, $ncols, $names, $who) {
    if ($cols == null) { throw $who + ": the column list is null"; }
    if (type($cols) != "list") {
        throw $who + ": columns must be a list, got " + type($cols);
    }
    if (len($cols) == 0) {
        throw $who + ": the column list is empty. A transformer with no columns "
              + "produces no output and is never what the caller meant -- drop "
              + "the spec instead.";
    }
    $out  = [];
    $seen = {};
    $i = 0;
    while ($i < len($cols)) {
        $c = $cols[$i];
        $idx = null;
        if (_whole($c)) {
            $idx = $c;
            if ($idx < 0) { $idx = $idx + $ncols; }
            if ($idx < 0 || $idx >= $ncols) {
                throw $who + ": column index " + str($c) + " is out of range for "
                      + str($ncols) + " columns";
            }
        }
        if ($idx == null) {
            if (type($c) != "string") {
                throw $who + ": a column must be a whole number or a name, got "
                      + type($c) + " (" + str($c) + ")";
            }
            if ($names == null) {
                throw $who + ": column '" + $c + "' was named, but X is an array "
                      + "with no column names. Pass integer indices, or pass an "
                      + "arctic DataFrame.";
            }
            $j = 0;
            while ($j < len($names)) {
                if ($names[$j] == $c) { $idx = $j; }
                $j = $j + 1;
            }
            if ($idx == null) {
                throw $who + ": there is no column named '" + $c + "'. X has "
                      + str($names) + ".";
            }
        }
        // A column repeated inside ONE spec would be transformed twice and
        // silently widen that block. Across specs it is legal (sklearn allows
        // it, and a column can legitimately feed two transformers), so the
        // check is per spec.
        if (contains($seen.keys(), str($idx))) {
            throw $who + ": column " + str($c) + " is listed twice in the same "
                  + "spec, which would transform it twice";
        }
        $seen[str($idx)] = true;
        append($out, $idx);
        $i = $i + 1;
    }
    return $out;
}

// Take a block of columns out of a [n, p] matrix, in the given order.
def _pick($A, $idx) {
    return np.take($A, np.astype(np.array($idx, "f64"), "i64"), 1);
}

// ── ColumnTransformer ───────────────────────────────────────────────────────
class _ColumnTransformer extends Estimator {
    def init($transformers, $remainder) {
        super();
        this.name_ = "ColumnTransformer";
        this.task_ = "transformer";
        this.specs = $transformers;
        if (this.specs == null) { this.specs = []; }
        if (type(this.specs) != "list") {
            throw "ColumnTransformer: transformers must be a list of "
                  + "[name, estimator, columns], got " + type(this.specs);
        }
        $rem = $remainder;
        if ($rem == null) { $rem = "drop"; }
        if ($rem != "drop" && $rem != "passthrough") {
            throw "ColumnTransformer: remainder must be \"drop\" or "
                  + "\"passthrough\", got " + str($rem);
        }
        // Validate the shape of every spec HERE, at construction, rather than
        // at fit. A grid search clones and refits many times; a malformed spec
        // should fail once, where the caller wrote it.
        $names = {};
        $i = 0;
        while ($i < len(this.specs)) {
            $s = this.specs[$i];
            if (type($s) != "list" || len($s) != 3) {
                throw "ColumnTransformer: spec " + str($i) + " must be "
                      + "[name, estimator, columns]";
            }
            if (type($s[0]) != "string") {
                throw "ColumnTransformer: spec " + str($i) + "'s name must be a "
                      + "string, got " + type($s[0]);
            }
            if (contains($names.keys(), $s[0])) {
                throw "ColumnTransformer: two specs are both named '" + $s[0]
                      + "'. Names address parameters as \"<name>__<param>\", so "
                      + "they have to be unique.";
            }
            $names[$s[0]] = true;
            _preCheckCols($s[2], "ColumnTransformer spec '" + $s[0] + "'");
            if ($s[1] == null || type($s[1]) != "instance") {
                throw "ColumnTransformer: spec '" + $s[0] + "' has no estimator";
            }
            // It has to look like an Estimator. Whether it can actually
            // transform is NOT checked here: there is no has_method in Bantu,
            // and the base class already raises "<name>.transform is not
            // implemented" with the offending name in it. Inventing a worse
            // version of that message at construction time would buy nothing.
            if ($s[1].name_ == null) {
                throw "ColumnTransformer: spec '" + $s[0] + "' holds an object "
                      + "that is not a bkit estimator (no name_). Every column "
                      + "block must be transformed, so each spec needs a "
                      + "transformer -- or a Pipeline whose last step is one.";
            }
            $i = $i + 1;
        }
        // Only `remainder` is a PARAMETER. The transformer count is derived
        // structure, and the contract is that every key get_params() returns
        // must be settable -- protocol_check feeds the first one straight back
        // into set_params(). Exposing a count that set_params cannot accept
        // breaks that; accepting it and doing nothing would be worse, because
        // then `set_params({"n_transformers": 5})` silently succeeds and
        // changes nothing. So it is a method, n_transformers(), not a param.
        this.params_ = {"remainder": $rem};
        this.resolved_  = [];     // per spec: the integer column indices
        this.widths_    = [];     // per spec: output width, learned at fit
        this.rem_idx_   = [];     // passthrough columns, ascending
        this.n_in_      = 0;      // X's column count at fit time
        this.names_in_  = null;   // X's column names at fit time, if a frame
    }

    // X may be a [n, p] array or an arctic DataFrame. A frame is materialised
    // ONCE here; every transformer then sees a slice of the same matrix.
    def _matrix($X, $who) {
        if (_isFrame($X)) {
            return [$X.to_ndarray(null), $X.columns()];
        }
        return [base._asarray($X), null];
    }

    def _shape($A, $who) {
        $s = np.shape($A);
        if (len($s) != 2) {
            throw $who + ": X must be 2-d [n_samples, n_features], got shape "
                  + str($s);
        }
        if ($s[0] == 0) {
            throw $who + ": X has no rows. There is nothing to learn from an "
                  + "empty matrix, and a transformer fitted on one would "
                  + "produce statistics that are not defined.";
        }
        return $s;
    }

    def fit($X, $y) {
        this.fit_transform($X, $y);
        return this;
    }

    // fit_transform is the real implementation: fitting already computes every
    // block, so fitting and then transforming again would do the work twice.
    def fit_transform($X, $y) {
        $m = this._matrix($X, "ColumnTransformer.fit");
        $A = $m[0];
        $s = this._shape($A, "ColumnTransformer.fit");
        this.n_in_     = $s[1];
        this.names_in_ = $m[1];

        if (len(this.specs) == 0 && this.params_.remainder == "drop") {
            throw "ColumnTransformer.fit: no transformers and remainder is "
                  + "\"drop\", so the output would have zero columns. Pass "
                  + "remainder = \"passthrough\" if that is what you meant.";
        }

        this.resolved_ = [];
        this.widths_   = [];
        $used = {};
        $i = 0;
        while ($i < len(this.specs)) {
            $idx = _resolve(this.specs[$i][2], this.n_in_, this.names_in_,
                            "ColumnTransformer spec '" + this.specs[$i][0] + "'");
            append(this.resolved_, $idx);
            $j = 0;
            while ($j < len($idx)) { $used[str($idx[$j])] = true; $j = $j + 1; }
            $i = $i + 1;
        }
        // Ascending, so the passthrough block's order is X's order and not the
        // order a dict happened to hand back.
        this.rem_idx_ = [];
        $c = 0;
        while ($c < this.n_in_) {
            if (!contains($used.keys(), str($c))) { append(this.rem_idx_, $c); }
            $c = $c + 1;
        }

        $blocks = [];
        $i = 0;
        while ($i < len(this.specs)) {
            $B = this.specs[$i][1].fit_transform(_pick($A, this.resolved_[$i]), $y);
            $B = this._as2d($B, this.specs[$i][0]);
            append(this.widths_, np.shape($B)[1]);
            append($blocks, $B);
            $i = $i + 1;
        }
        if (this.params_.remainder == "passthrough" && len(this.rem_idx_) > 0) {
            append($blocks, _pick($A, this.rem_idx_));
        }
        this.fitted_ = true;
        return this._join($blocks, $s[0], "ColumnTransformer.fit_transform");
    }

    def transform($X) {
        this._check_fitted("ColumnTransformer.transform");
        $m = this._matrix($X, "ColumnTransformer.transform");
        $A = $m[0];
        $s = this._shape($A, "ColumnTransformer.transform");
        if ($s[1] != this.n_in_) {
            throw "ColumnTransformer.transform: X has " + str($s[1])
                  + " columns but this was fitted on " + str(this.n_in_)
                  + ". The resolved column indices would address different "
                  + "features.";
        }
        // A frame whose columns were REORDERED between fit and transform would
        // resolve the same names to different indices, which is a silently
        // wrong answer rather than an error. So the names are checked, not just
        // the count.
        if (this.names_in_ != null && $m[1] != null && $m[1] != this.names_in_) {
            throw "ColumnTransformer.transform: this was fitted on columns "
                  + str(this.names_in_) + " and X has " + str($m[1])
                  + ". Column names are resolved to positions at fit time, so a "
                  + "reordered frame would transform the wrong columns.";
        }
        $blocks = [];
        $i = 0;
        while ($i < len(this.specs)) {
            $B = this.specs[$i][1].transform(_pick($A, this.resolved_[$i]));
            $B = this._as2d($B, this.specs[$i][0]);
            if (np.shape($B)[1] != this.widths_[$i]) {
                throw "ColumnTransformer.transform: '" + this.specs[$i][0]
                      + "' produced " + str(np.shape($B)[1]) + " columns at "
                      + "transform time but " + str(this.widths_[$i])
                      + " at fit time. Concatenating it would shift every "
                      + "later block's meaning.";
            }
            append($blocks, $B);
            $i = $i + 1;
        }
        if (this.params_.remainder == "passthrough" && len(this.rem_idx_) > 0) {
            append($blocks, _pick($A, this.rem_idx_));
        }
        return this._join($blocks, $s[0], "ColumnTransformer.transform");
    }

    // A transformer may hand back a 1-d column (LabelEncoder does). Widening
    // it here rather than in each caller keeps the concat rule in one place.
    def _as2d($B, $who) {
        $b = base._asarray($B);
        $sb = np.shape($b);
        if (len($sb) == 1) { return np.reshape($b, [$sb[0], 1]); }
        if (len($sb) != 2) {
            throw "ColumnTransformer: '" + $who + "' returned a rank-"
                  + str(len($sb)) + " array; a column block must be 1-d or 2-d";
        }
        return $b;
    }

    def _join($blocks, $n, $who) {
        if (len($blocks) == 0) {
            throw $who + ": every block was dropped, so the output has no "
                  + "columns";
        }
        $i = 0;
        while ($i < len($blocks)) {
            if (np.shape($blocks[$i])[0] != $n) {
                throw $who + ": block " + str($i) + " has "
                      + str(np.shape($blocks[$i])[0]) + " rows, X has " + str($n)
                      + ". A transformer that changes the row count cannot be "
                      + "composed column-wise.";
            }
            $i = $i + 1;
        }
        if (len($blocks) == 1) { return $blocks[0]; }
        return np.concat($blocks, 1);
    }

    // Where each spec's columns ended up in the output: [name, start, stop).
    // Not decoration -- without it a 40-column one-hot block is unattributable,
    // and "which input produced feature 37" is the first question anyone asks
    // of a fitted composition.
    def output_ranges() {
        this._check_fitted("ColumnTransformer.output_ranges");
        $out = [];
        $at = 0;
        $i = 0;
        while ($i < len(this.specs)) {
            append($out, [this.specs[$i][0], $at, $at + this.widths_[$i]]);
            $at = $at + this.widths_[$i];
            $i = $i + 1;
        }
        if (this.params_.remainder == "passthrough" && len(this.rem_idx_) > 0) {
            append($out, ["remainder", $at, $at + len(this.rem_idx_)]);
        }
        return $out;
    }

    def n_features_out() {
        this._check_fitted("ColumnTransformer.n_features_out");
        $t = 0;
        $i = 0;
        while ($i < len(this.widths_)) { $t = $t + this.widths_[$i]; $i = $i + 1; }
        if (this.params_.remainder == "passthrough") {
            $t = $t + len(this.rem_idx_);
        }
        return $t;
    }

    def get_transformer($name) {
        $i = 0;
        while ($i < len(this.specs)) {
            if (this.specs[$i][0] == $name) { return this.specs[$i][1]; }
            $i = $i + 1;
        }
        throw "ColumnTransformer: no transformer named '" + $name + "'. It has "
              + str(this._names()) + ".";
    }

    def _names() {
        $out = [];
        $i = 0;
        while ($i < len(this.specs)) { append($out, this.specs[$i][0]); $i = $i + 1; }
        return $out;
    }

    // Deep, for the same reason Pipeline's is (K2): a cloned composition must
    // share no estimator with its source, or fitting the clone mutates the
    // original and a grid search scores a configuration it never isolated.
    def clone() {
        $out = [];
        $i = 0;
        while ($i < len(this.specs)) {
            append($out, [this.specs[$i][0], this.specs[$i][1].clone(),
                          _cloneCols(this.specs[$i][2])]);
            $i = $i + 1;
        }
        return ColumnTransformer($out, this.params_.remainder);
    }

    def n_transformers() { return len(this.specs); }

    def get_params() {
        $out = {"remainder": this.params_.remainder};
        $i = 0;
        while ($i < len(this.specs)) {
            $nm = this.specs[$i][0];
            for $k, $v in this.specs[$i][1].get_params().items() {
                $out[$nm + "__" + $k] = $v;
            }
            $i = $i + 1;
        }
        return $out;
    }

    // "<name>__<param>" addresses a sub-transformer's parameter, exactly as
    // Pipeline does, so GridSearchCV can tune inside a composition. An
    // unknown key raises: a search that silently ignored one would report a
    // best score for a configuration it never tried (K24).
    def set_params($d) {
        if ($d == null) { return this; }
        for $k, $v in $d.items() {
            $cut = indexOf($k, "__");
            if ($cut < 0) {
                if ($k != "remainder") {
                    throw "ColumnTransformer.set_params: there is no parameter '"
                          + str($k) + "'. A sub-transformer's parameter is "
                          + "addressed as \"<name>__<param>\"; this one's "
                          + "transformers are " + str(this._names()) + ".";
                }
                if ($v != "drop" && $v != "passthrough") {
                    throw "ColumnTransformer.set_params: remainder must be "
                          + "\"drop\" or \"passthrough\", got " + str($v);
                }
                this.params_.remainder = $v;
            }
            if ($cut >= 0) {
                $nm  = substr($k, 0, $cut);
                $par = substr($k, $cut + 2, len($k) - $cut - 2);
                $one = {};
                $one[$par] = $v;
                this.get_transformer($nm).set_params($one);
            }
        }
        this.fitted_ = false;
        return this;
    }
}

// A column list is a plain Bantu list, and lists are shared by reference, so a
// clone that reused it would let set_params on one composition reach the other.
def _cloneCols($cols) {
    $out = [];
    $i = 0;
    while ($i < len($cols)) { append($out, $cols[$i]); $i = $i + 1; }
    return $out;
}

def ColumnTransformer($transformers, $remainder) {
    return new _ColumnTransformer($transformers, $remainder);
}

def version() { return "bkit.compose 0.6.0 (ColumnTransformer)"; }
