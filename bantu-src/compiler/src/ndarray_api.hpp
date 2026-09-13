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

} // namespace numba
