#pragma once
/**
 * numba — the only header evaluator.hpp sees.
 *
 * The implementation lives in ndarray_native.hpp, included by exactly one
 * translation unit (ndarray_native.cpp) which each build script compiles at
 * -O3. Keeping evaluator.hpp on this side of the wall is what makes that
 * possible, and it keeps ~35 builtin registrations out of a file that is
 * already 8,900 lines (N11).
 *
 * Registration is a callback so this header needs neither makeNative nor
 * ErrorHandler, and so every builtin gets the same error translation applied
 * uniformly in one place: any std::exception from the native layer becomes a
 * catchable Bantu error naming the builtin, never a process kill.
 */

#include "types.hpp"
#include <functional>

namespace numba {

using DefineFn = std::function<void(const char* name, NativeFn fn)>;

// Defines every nd_* builtin through `define`, and teaches print() how to
// render an array. Called once, from Evaluator::registerBuiltins.
void registerBuiltins(const DefineFn& define);

// ── operator dispatch (Phase 4) ──────────────────────────────────────────────
// Four small additive arms in the evaluator make a raw array handle the
// user-facing object, so `$a + $b` and `$m[1][2] = 9` mean what they look like.
//
// Every one of these paths is DEAD today: `$handle + 1` reads numberVal, which
// is always 0 for a handle, and silently yields 1; `$handle[i]` returns null;
// `$handle[i] = v` throws. There is exactly one other handle tag in existence
// ("column") and arctic never puts a column in an arithmetic expression, so
// nothing that works today changes behaviour.
//
// Its own enum rather than BantuTokenType: this header is deliberately the only
// thing evaluator.hpp sees of numba, and it should not drag in the token
// definitions. The evaluator maps its token to one of these.
enum class Op : int {
    Add, Sub, Mul, Div, Mod,
    Eq, Ne, Lt, Le, Gt, Ge
};

// Each returns true when it handled the case; false means "not mine, carry on
// with the existing behaviour". Returning false rather than throwing is what
// keeps the change additive.
bool dispatchBinary(Op op, const Value& l, const Value& r, Value& out);
bool dispatchNegate(const Value& v, Value& out);
bool dispatchIndex(const Value& obj, const Value& idx, Value& out);
bool dispatchIndexAssign(const Value& obj, const Value& idx, const Value& val);

// `$a.sum()`, `$a.shape()` and friends: returns a bound callable. Chaining
// works from here even before the operators land, and on any older build.
bool dispatchMethod(const Value& obj, const std::string& name, Value& out);

} // namespace numba
