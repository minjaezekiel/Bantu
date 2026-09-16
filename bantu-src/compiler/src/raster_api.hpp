#pragma once
/**
 * bplot's raster backend — the only header evaluator.hpp sees.
 *
 * The implementation lives in raster_core.hpp (portable C++ that knows nothing
 * about Bantu) and raster_native.cpp (the builtins over it), which is the one
 * translation unit that includes either. That split is what lets the core be
 * checked against decoders we did not write — Python's zlib and Pillow — and it
 * keeps another ~300 lines of registration out of a file that is already large.
 *
 * Registration is a callback, exactly as numba's is, so every builtin gets the
 * same error translation in one place: any std::exception from the native layer
 * becomes a catchable Bantu error naming the builtin, never a process kill.
 *
 * Design: docs/bplot-raster-architecture.md
 */

#include "types.hpp"

#include <functional>

namespace bplot_raster {

using DefineFn = std::function<void(const char* name, NativeFn fn)>;

// Defines every bp_canvas_*/bp_png_* builtin through `define`, and teaches
// print() how to render a canvas handle. Called once, from
// Evaluator::registerBuiltins.
void registerBuiltins(const DefineFn& define);

} // namespace bplot_raster
