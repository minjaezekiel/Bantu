#pragma once
/**
 * numba — shape manipulation: concat/split, pad, repeat/tile, im2col/col2im.
 *
 * Included by ndarray_native.cpp only, after ndarray_api.hpp.
 * docs/numba-architecture.md §4.12; btorch phases A6-A9.
 *
 * These five families are the primitives a tape cannot express cheaply. Each is
 * composable from nd_zeros plus slice-assignment, and each would then cost one
 * interpreter round-trip per slice -- which for im2col means one per kernel
 * position per output pixel, more than the convolution itself.
 *
 * Two of them are ADJOINTS of others and are tested as such:
 *
 *     split  is concat's adjoint     (concat(split(x)) == x, exactly)
 *     col2im is im2col's adjoint     (<im2col(x),y> == <x,col2im(y)>)
 *
 * An adjoint identity pins the geometry -- axis, strides, accumulation -- all at
 * once, and nothing that merely agrees with itself can satisfy it. Phase A5's
 * nd_index_add established the pattern; see btorch-suite/DECISIONS.md B15.
 */

#include "ndarray_reduce.hpp"
#include "ndarray_api.hpp"

namespace numba {

Value toArrayValue(const Value& v, const char* what);

// ─────────────────────────────────────────────────────────────────────────────
//  Shared: copy one array into a rectangular region of another
// ─────────────────────────────────────────────────────────────────────────────
// `dstBase` is the buffer offset of the region's first element. The region has
// src's shape and dst's strides. Used by concat, pad, repeat and tile, which
// differ only in where the regions go.
//
// `accumulate` makes it `+=` rather than `=`, which is what every adjoint
// needs and what nd_put's overwrite could not give (A5's lesson).
inline void blockCopy(NdArray& dst, size_t dstBase, const NdArray& src,
                      bool accumulate = false) {
    const size_t nd = src.ndim();
    if (nd == 0) {
        const double v = getAsDouble(src, src.offset);
        setFromDouble(dst, dstBase, accumulate ? getAsDouble(dst, dstBase) + v : v);
        return;
    }
    std::vector<ptrdiff_t> dstr(dst.strides.begin(), dst.strides.begin() + (ptrdiff_t)nd);
    // The tight case: both sides contiguous f64 over the last axis. This is the
    // common one for concat along axis 0 of C-contiguous arrays, and for every
    // im2col row.
    const bool tight = dst.dtype == DType::F64 && src.dtype == DType::F64
                    && dstr[nd - 1] == 1 && src.strides[nd - 1] == 1;
    if (tight && nd == 1) {
        double* d = static_cast<double*>(dst.buf->data) + dstBase;
        const double* s = static_cast<const double*>(src.buf->data) + src.offset;
        const size_t n = src.shape[0];
        if (accumulate) { for (size_t i = 0; i < n; i++) d[i] += s[i]; }
        else            { for (size_t i = 0; i < n; i++) d[i]  = s[i]; }
        return;
    }
    nditer(src.shape, { dstr, src.strides }, { dstBase, src.offset },
           [&](const size_t* o) {
        const double v = getAsDouble(src, o[1]);
        setFromDouble(dst, o[0], accumulate ? getAsDouble(dst, o[0]) + v : v);
    });
}

// A list argument that must hold arrays (or things convertible to arrays).
inline std::vector<ArrayPtr> arrayList(const Value& v, const char* what) {
    if (!v.isList()) {
        throw std::runtime_error(std::string(what) +
            ": expected a LIST of arrays (got " + v.toString() + ")");
    }
    if (v.listVal.empty()) {
        throw std::runtime_error(std::string(what) + ": the list is empty -- "
            "there is nothing to work with, which is an error rather than an "
            "empty result, because the result's rank would be unknowable");
    }
    std::vector<ArrayPtr> out;
    out.reserve(v.listVal.size());
    for (const Value& e : v.listVal) out.push_back(asArray(toArrayValue(e, what)));
    return out;
}

// One non-negative whole number from an argument.
inline size_t countArg(const std::vector<Value>& a, size_t i, const char* what,
                       const char* name) {
    if (a.size() <= i || a[i].isNull() || !a[i].isNumber()) {
        throw std::runtime_error(std::string(what) + ": " + name + " must be a number");
    }
    const double d = a[i].numberVal;
    if (std::isnan(d) || std::isinf(d) || d != std::floor(d) || d < 0) {
        throw std::runtime_error(std::string(what) + ": " + name +
            " must be a non-negative whole number (got " + a[i].toString() + ")");
    }
    return (size_t)d;
}

inline void registerManip(const DefineFn& define) {

    // ── A6: nd_concat / nd_split ─────────────────────────────────────────────

    // nd_concat([a, b, ...], axis?) -- join along an existing axis.
    //
    // Every input must agree on rank and on every axis BUT the join axis. The
    // error names the first disagreement and both shapes, because "shapes do
    // not match" with a list of ten arrays is not a usable message.
    define("nd_concat", [](std::vector<Value> args) -> Value {
        if (args.empty()) throw std::runtime_error("nd_concat(arrays, axis?) needs a list of arrays");
        std::vector<ArrayPtr> xs = arrayList(args[0], "nd_concat");
        const size_t nd = xs[0]->ndim();
        if (nd == 0) {
            throw std::runtime_error("nd_concat: cannot concatenate 0-d arrays -- "
                "there is no axis to join along (use nd_stack to make one)");
        }
        size_t axis = 0;
        if (args.size() > 1 && !args[1].isNull()) axis = normAxis(args[1].numberVal, nd);

        DType dt = xs[0]->dtype;
        size_t joined = 0;
        for (size_t k = 0; k < xs.size(); k++) {
            const NdArray& x = *xs[k];
            if (x.ndim() != nd) {
                throw std::runtime_error("nd_concat: array 0 is " + std::to_string(nd) +
                    "-dimensional but array " + std::to_string(k) + " is " +
                    std::to_string(x.ndim()) + "-dimensional");
            }
            for (size_t d = 0; d < nd; d++) {
                if (d == axis) continue;
                if (x.shape[d] != xs[0]->shape[d]) {
                    throw std::runtime_error("nd_concat: joining along axis " +
                        std::to_string(axis) + ", so every other axis must agree -- "
                        "array 0 is " + shapeStr(xs[0]->shape) + " but array " +
                        std::to_string(k) + " is " + shapeStr(x.shape) +
                        " (they differ on axis " + std::to_string(d) + ")");
                }
            }
            dt = promote(dt, x.dtype);
            joined += x.shape[axis];
        }

        std::vector<size_t> shape = xs[0]->shape;
        shape[axis] = joined;
        ArrayPtr out = makeArray(shape, dt, "nd_concat");
        if (out->size() == 0) return wrap(out);

        // Each input lands at one offset along the join axis. Nothing is
        // proportional to the output except the copy itself.
        NdArray& O = *out;
        size_t at = 0;
        for (const ArrayPtr& xp : xs) {
            if (xp->size() != 0) {
                blockCopy(O, (size_t)((ptrdiff_t)O.offset + (ptrdiff_t)at * O.strides[axis]), *xp);
            }
            at += xp->shape[axis];
        }
        return wrap(out);
    });

    // nd_stack([a, b, ...], axis?) -- join along a NEW axis. Every input must
    // have exactly the same shape. Written in terms of the same block copy
    // rather than as reshape+concat, because a reshape of a strided view can
    // copy and the extra copy would be invisible.
    define("nd_stack", [](std::vector<Value> args) -> Value {
        if (args.empty()) throw std::runtime_error("nd_stack(arrays, axis?) needs a list of arrays");
        std::vector<ArrayPtr> xs = arrayList(args[0], "nd_stack");
        const size_t nd = xs[0]->ndim();
        ptrdiff_t axis = 0;
        if (args.size() > 1 && !args[1].isNull()) {
            axis = (ptrdiff_t)std::llround(args[1].numberVal);
            if (axis < 0) axis += (ptrdiff_t)nd + 1;
            if (axis < 0 || (size_t)axis > nd) {
                throw std::runtime_error("nd_stack: axis " + args[1].toString() +
                    " is out of range -- stacking " + std::to_string(nd) +
                    "-dimensional arrays gives " + std::to_string(nd + 1) +
                    " axes, so the new axis may be 0.." + std::to_string(nd));
            }
        }
        DType dt = xs[0]->dtype;
        for (size_t k = 0; k < xs.size(); k++) {
            if (xs[k]->shape != xs[0]->shape) {
                throw std::runtime_error("nd_stack: every array must have the SAME shape "
                    "(nd_concat is the one that allows a differing axis) -- array 0 is " +
                    shapeStr(xs[0]->shape) + " but array " + std::to_string(k) + " is " +
                    shapeStr(xs[k]->shape));
            }
            dt = promote(dt, xs[k]->dtype);
        }
        std::vector<size_t> shape;
        for (size_t d = 0; d < (size_t)axis; d++) shape.push_back(xs[0]->shape[d]);
        shape.push_back(xs.size());
        for (size_t d = (size_t)axis; d < nd; d++) shape.push_back(xs[0]->shape[d]);

        ArrayPtr out = makeArray(shape, dt, "nd_stack");
        if (out->size() == 0) return wrap(out);
        NdArray& O = *out;
        // A slab of the output with the new axis dropped has exactly the
        // inputs' shape, and the strides either side of the new axis.
        std::vector<ptrdiff_t> slab;
        for (size_t d = 0; d < shape.size(); d++) {
            if (d == (size_t)axis) continue;
            slab.push_back(O.strides[d]);
        }
        for (size_t k = 0; k < xs.size(); k++) {
            const size_t base = (size_t)((ptrdiff_t)O.offset + (ptrdiff_t)k * O.strides[axis]);
            const NdArray& S = *xs[k];
            if (S.ndim() == 0) {
                setFromDouble(O, base, getAsDouble(S, S.offset));
                continue;
            }
            nditer(S.shape, { slab, S.strides }, { base, S.offset },
                   [&](const size_t* o) { setFromDouble(O, o[0], getAsDouble(S, o[1])); });
        }
        return wrap(out);
    });

    // nd_split(a, sections, axis?) -- the adjoint of concat, returning a LIST.
    //
    // `sections` is either a count (split into n equal parts, which must divide
    // exactly) or a list of sizes (which must sum to the axis length). NumPy's
    // array_split allows an uneven count; this does not, because a silently
    // uneven split is the kind of thing that shows up as a shape error three
    // functions away.
    define("nd_split", [](std::vector<Value> args) -> Value {
        if (args.size() < 2) throw std::runtime_error("nd_split(a, sections, axis?) needs two arguments");
        ArrayPtr a = asArray(toArrayValue(args[0], "nd_split"));
        const size_t nd = a->ndim();
        if (nd == 0) throw std::runtime_error("nd_split: a 0-d array has no axis to split");
        size_t axis = 0;
        if (args.size() > 2 && !args[2].isNull()) axis = normAxis(args[2].numberVal, nd);
        const size_t extent = a->shape[axis];

        std::vector<size_t> sizes;
        if (args[1].isList()) {
            size_t total = 0;
            for (const Value& v : args[1].listVal) {
                if (!v.isNumber() || v.numberVal < 0 || v.numberVal != std::floor(v.numberVal)) {
                    throw std::runtime_error("nd_split: each section size must be a "
                        "non-negative whole number (got " + v.toString() + ")");
                }
                sizes.push_back((size_t)v.numberVal);
                total += (size_t)v.numberVal;
            }
            if (total != extent) {
                throw std::runtime_error("nd_split: the section sizes sum to " +
                    std::to_string(total) + " but axis " + std::to_string(axis) +
                    " has " + std::to_string(extent) + " elements");
            }
        } else {
            const size_t n = countArg(args, 1, "nd_split", "sections");
            if (n == 0) throw std::runtime_error("nd_split: cannot split into 0 sections");
            if (extent % n != 0) {
                throw std::runtime_error("nd_split: " + std::to_string(extent) +
                    " elements on axis " + std::to_string(axis) + " do not divide into " +
                    std::to_string(n) + " equal parts -- pass a list of sizes to split unevenly");
            }
            sizes.assign(n, extent / n);
        }

        // Each part is a VIEW, not a copy: splitting a batch into minibatches
        // must not duplicate the batch. The caller gets nd_copy if it wants one.
        std::vector<Value> parts;
        parts.reserve(sizes.size());
        ptrdiff_t at = 0;
        for (size_t s : sizes) {
            std::vector<size_t> shape = a->shape;
            shape[axis] = s;
            const size_t off = (size_t)((ptrdiff_t)a->offset + at * a->strides[axis]);
            parts.push_back(wrap(makeView(a, shape, a->strides, off, "nd_split")));
            at += (ptrdiff_t)s;
        }
        return Value(parts);
    });

    // ── A7: nd_pad ───────────────────────────────────────────────────────────

    // nd_pad(a, pads, value?) -- constant-value pad, N-dimensional.
    //
    // `pads` is [[before, after], ...] per axis, or [before, after] applied to
    // every axis, or a single number applied to both sides of every axis.
    // Negative pads RAISE: cropping is a different operation with a different
    // adjoint, and silently accepting one here would make the pad/crop pair
    // non-invertible without anyone noticing.
    define("nd_pad", [](std::vector<Value> args) -> Value {
        if (args.size() < 2) throw std::runtime_error("nd_pad(a, pads, value?) needs two arguments");
        ArrayPtr a = asArray(toArrayValue(args[0], "nd_pad"));
        const size_t nd = a->ndim();
        if (nd == 0) throw std::runtime_error("nd_pad: a 0-d array has no axis to pad");

        std::vector<size_t> before(nd, 0), after(nd, 0);
        auto whole = [](const Value& v, const char* side) -> size_t {
            if (!v.isNumber()) {
                throw std::runtime_error(std::string("nd_pad: the ") + side +
                    " pad must be a number (got " + v.toString() + ")");
            }
            const double d = v.numberVal;
            if (std::isnan(d) || std::isinf(d) || d != std::floor(d)) {
                throw std::runtime_error(std::string("nd_pad: the ") + side +
                    " pad must be a whole number (got " + v.toString() + ")");
            }
            if (d < 0) {
                throw std::runtime_error(std::string("nd_pad: a negative ") + side +
                    " pad (" + v.toString() + ") is not a crop -- padding and cropping "
                    "have different adjoints, so cropping is nd_slice's job");
            }
            return (size_t)d;
        };

        const Value& P = args[1];
        if (P.isNumber()) {
            before.assign(nd, whole(P, "before"));
            after = before;
        } else if (P.isList() && P.listVal.size() == 2 && P.listVal[0].isNumber()) {
            before.assign(nd, whole(P.listVal[0], "before"));
            after.assign(nd, whole(P.listVal[1], "after"));
        } else if (P.isList()) {
            if (P.listVal.size() != nd) {
                throw std::runtime_error("nd_pad: pads has " + std::to_string(P.listVal.size()) +
                    " entries for a " + std::to_string(nd) + "-dimensional array -- give one "
                    "[before, after] pair per axis, or one pair for all of them, or one number");
            }
            for (size_t d = 0; d < nd; d++) {
                const Value& e = P.listVal[d];
                if (!e.isList() || e.listVal.size() != 2) {
                    throw std::runtime_error("nd_pad: entry " + std::to_string(d) +
                        " of pads is not a [before, after] pair");
                }
                before[d] = whole(e.listVal[0], "before");
                after[d]  = whole(e.listVal[1], "after");
            }
        } else {
            throw std::runtime_error("nd_pad: pads must be a number, a [before, after] "
                "pair, or one pair per axis");
        }

        double fill = 0.0;
        if (args.size() > 2 && !args[2].isNull()) {
            if (!args[2].isNumber() && !args[2].isBool()) {
                throw std::runtime_error("nd_pad: the fill value must be a number or a bool");
            }
            fill = args[2].isBool() ? (args[2].boolVal ? 1.0 : 0.0) : args[2].numberVal;
        }

        std::vector<size_t> shape(nd);
        for (size_t d = 0; d < nd; d++) {
            shape[d] = checkedAdd(checkedAdd(before[d], a->shape[d], "nd_pad"),
                                  after[d], "nd_pad");
        }
        // zero=false then fill: filling is one pass either way, and zeroing
        // first would be a second pass over what can be a large array.
        const bool zeroFill = (fill == 0.0);
        ArrayPtr out = makeArray(shape, a->dtype, "nd_pad", zeroFill);
        NdArray& O = *out;
        if (!zeroFill) {
            const size_t n = O.size();
            for (size_t i = 0; i < n; i++) setFromDouble(O, i, fill);
        }
        if (a->size() == 0) return wrap(out);

        ptrdiff_t base = (ptrdiff_t)O.offset;
        for (size_t d = 0; d < nd; d++) base += (ptrdiff_t)before[d] * O.strides[d];
        blockCopy(O, (size_t)base, *a);
        return wrap(out);
    });

    // ── A8: nd_repeat / nd_tile ──────────────────────────────────────────────

    // nd_repeat(a, repeats, axis?) -- repeat each ELEMENT along an axis.
    //   repeat([1,2,3], 2) -> [1,1,2,2,3,3]
    // With axis omitted the array is flattened first, which is NumPy's rule.
    define("nd_repeat", [](std::vector<Value> args) -> Value {
        if (args.size() < 2) throw std::runtime_error("nd_repeat(a, repeats, axis?) needs two arguments");
        ArrayPtr a = asArray(toArrayValue(args[0], "nd_repeat"));
        const size_t r = countArg(args, 1, "nd_repeat", "repeats");

        // No axis: flatten. A strided source has to be materialised first,
        // because the flat order is not the buffer order.
        if (args.size() < 3 || args[2].isNull()) {
            const size_t n = a->size();
            ArrayPtr out = makeArray({ checkedMul(n, r, "nd_repeat") }, a->dtype, "nd_repeat");
            if (r == 0 || n == 0) return wrap(out);
            NdArray& O = *out;
            const NdArray& A = *a;
            size_t at = 0;
            forEachIndex(A, [&](size_t o, size_t) {
                const double v = getAsDouble(A, o);
                for (size_t k = 0; k < r; k++) setFromDouble(O, at++, v);
            });
            return wrap(out);
        }

        const size_t nd = a->ndim();
        const size_t axis = normAxis(args[2].numberVal, nd);
        std::vector<size_t> shape = a->shape;
        shape[axis] = checkedMul(a->shape[axis], r, "nd_repeat");
        ArrayPtr out = makeArray(shape, a->dtype, "nd_repeat");
        if (out->size() == 0) return wrap(out);

        // Each input slab is written r times, consecutively. One stride walk.
        NdArray& O = *out;
        const NdArray& A = *a;
        std::vector<size_t> slabShape = A.shape;
        slabShape[axis] = 1;
        std::vector<ptrdiff_t> ostr = O.strides, astr = A.strides;
        for (size_t i = 0; i < A.shape[axis]; i++) {
            for (size_t k = 0; k < r; k++) {
                const size_t ob = (size_t)((ptrdiff_t)O.offset +
                                           (ptrdiff_t)(i * r + k) * O.strides[axis]);
                const size_t ab = (size_t)((ptrdiff_t)A.offset + (ptrdiff_t)i * A.strides[axis]);
                nditer(slabShape, { ostr, astr }, { ob, ab },
                       [&](const size_t* o) { setFromDouble(O, o[0], getAsDouble(A, o[1])); });
            }
        }
        return wrap(out);
    });

    // nd_tile(a, reps) -- repeat the whole BLOCK.
    //   tile([1,2,3], 2) -> [1,2,3,1,2,3]
    // `reps` is a count (applied to the last axis) or one per axis. A reps list
    // longer than the rank prepends new axes, as NumPy does.
    define("nd_tile", [](std::vector<Value> args) -> Value {
        if (args.size() < 2) throw std::runtime_error("nd_tile(a, reps) needs two arguments");
        ArrayPtr a = asArray(toArrayValue(args[0], "nd_tile"));
        std::vector<size_t> reps;
        if (args[1].isNumber()) {
            reps.assign(a->ndim() == 0 ? 1 : a->ndim(), 1);
            if (reps.empty()) reps.push_back(1);
            reps.back() = countArg(args, 1, "nd_tile", "reps");
        } else if (args[1].isList()) {
            for (size_t i = 0; i < args[1].listVal.size(); i++) {
                const Value& v = args[1].listVal[i];
                if (!v.isNumber() || v.numberVal < 0 || v.numberVal != std::floor(v.numberVal)) {
                    throw std::runtime_error("nd_tile: each rep must be a non-negative "
                        "whole number (got " + v.toString() + ")");
                }
                reps.push_back((size_t)v.numberVal);
            }
        } else {
            throw std::runtime_error("nd_tile: reps must be a number or a list of numbers");
        }

        // Left-pad the shape with 1s if reps is longer, and left-pad reps with
        // 1s if the shape is longer.
        std::vector<size_t> ish = a->shape;
        std::vector<ptrdiff_t> ist = a->strides;
        while (ish.size() < reps.size()) {
            ish.insert(ish.begin(), 1);
            ist.insert(ist.begin(), 0);
        }
        while (reps.size() < ish.size()) reps.insert(reps.begin(), 1);

        const size_t nd = ish.size();
        std::vector<size_t> shape(nd);
        for (size_t d = 0; d < nd; d++) shape[d] = checkedMul(ish[d], reps[d], "nd_tile");
        ArrayPtr out = makeArray(shape, a->dtype, "nd_tile");
        if (out->size() == 0) return wrap(out);

        NdArray& O = *out;
        const NdArray& A = *a;
        // Walk the grid of block positions; copy the whole block at each.
        std::vector<size_t> blockIdx(nd, 0);
        size_t blocks = 1;
        for (size_t d = 0; d < nd; d++) blocks = checkedMul(blocks, reps[d], "nd_tile");
        for (size_t b = 0; b < blocks; b++) {
            ptrdiff_t base = (ptrdiff_t)O.offset;
            for (size_t d = 0; d < nd; d++) {
                base += (ptrdiff_t)(blockIdx[d] * ish[d]) * O.strides[d];
            }
            nditer(ish, { O.strides, ist }, { (size_t)base, A.offset },
                   [&](const size_t* o) { setFromDouble(O, o[0], getAsDouble(A, o[1])); });
            for (ptrdiff_t d = (ptrdiff_t)nd - 1; d >= 0; d--) {
                if (++blockIdx[d] < reps[d]) break;
                blockIdx[d] = 0;
            }
        }
        return wrap(out);
    });

    // ── A9: nd_im2col / nd_col2im ────────────────────────────────────────────
    //
    // The gather that turns a convolution into one matmul, and its adjoint.
    //
    //   x    : [N, C, H, W]
    //   cols : [N * OH * OW, C * KH * KW]
    //
    // With that layout a convolution is exactly
    //     cols @ W.reshape(C*KH*KW, OUT)  ->  [N*OH*OW, OUT]
    // and the row-major order of the first axis is (n, oh, ow), so the result
    // reshapes to [N, OH, OW, OUT] with no transpose. The alternative layout
    // ([C*KH*KW, N*OH*OW]) saves a transpose in the weight gradient and costs
    // one in the forward; the forward runs more often.
    //
    // col2im ACCUMULATES, which is what makes it the adjoint: one input pixel
    // is read by several kernel positions, so its gradient is their sum. This
    // is the same property A5 added nd_index_add for.

    struct ConvGeom {
        size_t N, C, H, W, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW;
    };

    auto convGeom = [](const ArrayPtr& x, const std::vector<Value>& a, size_t first,
                       const char* what) -> ConvGeom {
        auto opt = [&](size_t i, size_t dflt) -> size_t {
            if (a.size() <= i || a[i].isNull()) return dflt;
            if (!a[i].isNumber()) {
                throw std::runtime_error(std::string(what) + ": argument " +
                    std::to_string(i) + " must be a number");
            }
            const double d = a[i].numberVal;
            if (std::isnan(d) || d != std::floor(d) || d < 0) {
                throw std::runtime_error(std::string(what) +
                    ": strides, padding and dilation must be non-negative whole numbers "
                    "(got " + a[i].toString() + ")");
            }
            return (size_t)d;
        };
        ConvGeom g{};
        g.N = x->shape[0]; g.C = x->shape[1]; g.H = x->shape[2]; g.W = x->shape[3];
        g.KH = opt(first + 0, 0); g.KW = opt(first + 1, 0);
        if (g.KH == 0 || g.KW == 0) {
            throw std::runtime_error(std::string(what) +
                ": the kernel height and width are required and must be positive");
        }
        g.SH = opt(first + 2, 1); g.SW = opt(first + 3, 1);
        if (g.SH == 0 || g.SW == 0) {
            throw std::runtime_error(std::string(what) + ": a stride of 0 would never advance");
        }
        g.PH = opt(first + 4, 0); g.PW = opt(first + 5, 0);
        g.DH = opt(first + 6, 1); g.DW = opt(first + 7, 1);
        if (g.DH == 0 || g.DW == 0) {
            throw std::runtime_error(std::string(what) + ": a dilation of 0 is meaningless");
        }
        const size_t effH = (g.KH - 1) * g.DH + 1;
        const size_t effW = (g.KW - 1) * g.DW + 1;
        if (g.H + 2 * g.PH < effH || g.W + 2 * g.PW < effW) {
            throw std::runtime_error(std::string(what) + ": the kernel (" +
                std::to_string(g.KH) + "x" + std::to_string(g.KW) + " dilated to " +
                std::to_string(effH) + "x" + std::to_string(effW) +
                ") does not fit in the padded input (" +
                std::to_string(g.H + 2 * g.PH) + "x" + std::to_string(g.W + 2 * g.PW) + ")");
        }
        g.OH = (g.H + 2 * g.PH - effH) / g.SH + 1;
        g.OW = (g.W + 2 * g.PW - effW) / g.SW + 1;
        return g;
    };

    // nd_im2col(x, kh, kw, sh?, sw?, ph?, pw?, dh?, dw?)
    define("nd_im2col", [convGeom](std::vector<Value> args) -> Value {
        if (args.size() < 3) {
            throw std::runtime_error("nd_im2col(x, kh, kw, sh?, sw?, ph?, pw?, dh?, dw?) "
                "needs at least an array and a kernel size");
        }
        ArrayPtr x = asArray(toArrayValue(args[0], "nd_im2col"));
        if (x->ndim() != 4) {
            throw std::runtime_error("nd_im2col: expected a 4-d [N, C, H, W] array, got " +
                shapeStr(x->shape));
        }
        const ConvGeom g = convGeom(x, args, 1, "nd_im2col");

        const size_t rows = checkedMul(checkedMul(g.N, g.OH, "nd_im2col"), g.OW, "nd_im2col");
        const size_t cols = checkedMul(checkedMul(g.C, g.KH, "nd_im2col"), g.KW, "nd_im2col");
        ArrayPtr out = makeArray({ rows, cols }, DType::F64, "nd_im2col");
        if (out->size() == 0) return wrap(out);

        const NdArray& X = *x;
        NdArray& O = *out;
        double* od = static_cast<double*>(O.buf->data);
        const ptrdiff_t xs0 = X.strides[0], xs1 = X.strides[1],
                        xs2 = X.strides[2], xs3 = X.strides[3];
        // Row-major (n, oh, ow) over rows and (c, kh, kw) over columns, so both
        // halves of the matmul see contiguous memory.
        size_t r = 0;
        for (size_t n = 0; n < g.N; n++) {
            for (size_t oh = 0; oh < g.OH; oh++) {
                for (size_t ow = 0; ow < g.OW; ow++, r++) {
                    double* row = od + r * cols;
                    size_t cix = 0;
                    for (size_t c = 0; c < g.C; c++) {
                        for (size_t kh = 0; kh < g.KH; kh++) {
                            const ptrdiff_t ih = (ptrdiff_t)(oh * g.SH + kh * g.DH) - (ptrdiff_t)g.PH;
                            const bool hOut = (ih < 0 || (size_t)ih >= g.H);
                            for (size_t kw = 0; kw < g.KW; kw++, cix++) {
                                if (hOut) { row[cix] = 0.0; continue; }
                                const ptrdiff_t iw = (ptrdiff_t)(ow * g.SW + kw * g.DW) - (ptrdiff_t)g.PW;
                                if (iw < 0 || (size_t)iw >= g.W) { row[cix] = 0.0; continue; }
                                row[cix] = getAsDouble(X, (size_t)((ptrdiff_t)X.offset +
                                    (ptrdiff_t)n * xs0 + (ptrdiff_t)c * xs1 + ih * xs2 + iw * xs3));
                            }
                        }
                    }
                }
            }
        }
        return wrap(out);
    });

    // nd_col2im(cols, shape, kh, kw, sh?, sw?, ph?, pw?, dh?, dw?)
    //
    // `shape` is the [N, C, H, W] of the ORIGINAL image, which cannot be
    // recovered from the column matrix alone (several H values give the same
    // OH once stride and padding are involved).
    define("nd_col2im", [convGeom](std::vector<Value> args) -> Value {
        if (args.size() < 4) {
            throw std::runtime_error("nd_col2im(cols, shape, kh, kw, sh?, sw?, ph?, pw?, dh?, dw?) "
                "needs the columns, the image shape and a kernel size");
        }
        ArrayPtr cm = asArray(toArrayValue(args[0], "nd_col2im"));
        if (cm->ndim() != 2) {
            throw std::runtime_error("nd_col2im: expected a 2-d column matrix, got " +
                shapeStr(cm->shape));
        }
        if (!args[1].isList() || args[1].listVal.size() != 4) {
            throw std::runtime_error("nd_col2im: shape must be a 4-element list [N, C, H, W] -- "
                "it cannot be inferred from the columns, because several image heights give "
                "the same output height once stride and padding are involved");
        }
        std::vector<size_t> ishape(4);
        for (size_t d = 0; d < 4; d++) {
            const Value& v = args[1].listVal[d];
            if (!v.isNumber() || v.numberVal < 0 || v.numberVal != std::floor(v.numberVal)) {
                throw std::runtime_error("nd_col2im: shape entry " + std::to_string(d) +
                    " must be a non-negative whole number (got " + v.toString() + ")");
            }
            ishape[d] = (size_t)v.numberVal;
        }
        ArrayPtr probe = makeArray({ ishape[0], ishape[1], ishape[2], ishape[3] },
                                   DType::F64, "nd_col2im");
        const ConvGeom g = convGeom(probe, args, 2, "nd_col2im");

        const size_t rows = checkedMul(checkedMul(g.N, g.OH, "nd_col2im"), g.OW, "nd_col2im");
        const size_t cols = checkedMul(checkedMul(g.C, g.KH, "nd_col2im"), g.KW, "nd_col2im");
        if (cm->shape[0] != rows || cm->shape[1] != cols) {
            throw std::runtime_error("nd_col2im: the columns are " + shapeStr(cm->shape) +
                " but a " + shapeStr(ishape) + " image with this kernel and stride gives [" +
                std::to_string(rows) + ", " + std::to_string(cols) + "]");
        }

        ArrayPtr out = probe;            // already zeroed, already the right shape
        if (out->size() == 0 || cm->size() == 0) return wrap(out);
        NdArray& O = *out;
        const NdArray& C2 = *cm;
        double* odata = static_cast<double*>(O.buf->data);
        const ptrdiff_t os0 = O.strides[0], os1 = O.strides[1],
                        os2 = O.strides[2], os3 = O.strides[3];
        size_t r = 0;
        for (size_t n = 0; n < g.N; n++) {
            for (size_t oh = 0; oh < g.OH; oh++) {
                for (size_t ow = 0; ow < g.OW; ow++, r++) {
                    size_t cix = 0;
                    for (size_t c = 0; c < g.C; c++) {
                        for (size_t kh = 0; kh < g.KH; kh++) {
                            const ptrdiff_t ih = (ptrdiff_t)(oh * g.SH + kh * g.DH) - (ptrdiff_t)g.PH;
                            const bool hOut = (ih < 0 || (size_t)ih >= g.H);
                            for (size_t kw = 0; kw < g.KW; kw++, cix++) {
                                if (hOut) continue;
                                const ptrdiff_t iw = (ptrdiff_t)(ow * g.SW + kw * g.DW) - (ptrdiff_t)g.PW;
                                if (iw < 0 || (size_t)iw >= g.W) continue;
                                // += , not = : one input pixel is read by
                                // several kernel positions, so its gradient is
                                // their SUM. An overwrite here is the classic
                                // silently-wrong convolution backward.
                                odata[(size_t)((ptrdiff_t)O.offset + (ptrdiff_t)n * os0 +
                                               (ptrdiff_t)c * os1 + ih * os2 + iw * os3)]
                                    += getAsDouble(C2, (size_t)((ptrdiff_t)C2.offset +
                                                   (ptrdiff_t)r * C2.strides[0] +
                                                   (ptrdiff_t)cix * C2.strides[1]));
                            }
                        }
                    }
                }
            }
        }
        return wrap(out);
    });

    // nd_conv_outsize(h, w, kh, kw, sh?, sw?, ph?, pw?, dh?, dw?) -- the output
    // height and width, as [oh, ow]. Exposed because every caller needs it to
    // reshape the matmul result, and recomputing the floor division in Bantu is
    // exactly the kind of duplicated arithmetic that drifts.
    define("nd_conv_outsize", [convGeom](std::vector<Value> args) -> Value {
        if (args.size() < 4) {
            throw std::runtime_error("nd_conv_outsize(h, w, kh, kw, sh?, sw?, ph?, pw?, dh?, dw?) "
                "needs at least h, w, kh, kw");
        }
        if (!args[0].isNumber() || !args[1].isNumber()) {
            throw std::runtime_error("nd_conv_outsize: h and w must be numbers");
        }
        auto probe = std::make_shared<NdArray>();
        probe->shape = { 1, 1, (size_t)args[0].numberVal, (size_t)args[1].numberVal };
        std::vector<Value> shifted;
        shifted.push_back(Value());                  // placeholder for args[0]
        for (size_t i = 2; i < args.size(); i++) shifted.push_back(args[i]);
        const ConvGeom g = convGeom(probe, shifted, 1, "nd_conv_outsize");
        ArrayPtr out = makeArray({ 2 }, DType::I64, "nd_conv_outsize");
        setFromDouble(*out, 0, (double)g.OH);
        setFromDouble(*out, 1, (double)g.OW);
        return wrap(out);
    });
}

}   // namespace numba
