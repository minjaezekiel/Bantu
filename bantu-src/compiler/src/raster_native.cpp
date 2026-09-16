// ════════════════════════════════════════════════════════════════════════════
//  raster_native.cpp — the Bantu surface of bplot's raster backend.
//
//  The only translation unit that includes raster_core.hpp. Everything here is
//  argument checking, unit conversion and handle lifetime; every decision about
//  a pixel or a compressed byte is in the core, where it can be tested against
//  decoders we did not write.
//
//  Each builtin is reachable from any Bantu program -- including one running
//  inside a sua handler -- so each validates its arguments as if they came from
//  a stranger. A bad one raises a catchable Bantu error; none can allocate
//  beyond the caps in raster_core.hpp.
//
//  Design: docs/bplot-raster-architecture.md
// ════════════════════════════════════════════════════════════════════════════

#include "raster_api.hpp"
#include "raster_core.hpp"

#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace bplot_raster {

static const char* kCanvasTag = "canvas";

// ── Argument helpers ────────────────────────────────────────────────────────

static std::shared_ptr<Canvas> asCanvas(const std::vector<Value>& a, size_t i, const char* fn) {
    if (a.size() <= i || a[i].type != Value::NATIVE_HANDLE || a[i].stringVal != kCanvasTag || !a[i].handle) {
        throw std::runtime_error(std::string(fn) + ": expected a canvas, from bp_canvas_new()");
    }
    return std::static_pointer_cast<Canvas>(a[i].handle);
}

static double asNumber(const std::vector<Value>& a, size_t i, const char* fn, const char* what) {
    if (a.size() <= i || !a[i].isNumber() || !std::isfinite(a[i].numberVal)) {
        throw std::runtime_error(std::string(fn) + ": " + what + " must be a finite number");
    }
    return a[i].numberVal;
}

static bool isAbsent(const std::vector<Value>& a, size_t i) {
    return a.size() <= i || a[i].isNull();
}

// A hex digit pair, rejecting anything that is not one.
static uint8_t hexPair(const std::string& s, size_t at, const char* fn) {
    auto nib = [&](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        throw std::runtime_error(std::string(fn) + ": a colour must be \"#rrggbb\" or \"none\", got \"" + s + "\"");
    };
    return (uint8_t)(nib(s[at]) * 16 + nib(s[at + 1]));
}

// Returns false for "none", which draws nothing -- the SVG spelling bplot uses.
static bool parseColour(const std::vector<Value>& a, size_t i, const char* fn, Rgb& out) {
    if (a.size() <= i || !a[i].isString()) {
        throw std::runtime_error(std::string(fn) + ": a colour must be a string, \"#rrggbb\" or \"none\"");
    }
    const std::string& s = a[i].stringVal;
    if (s == "none") return false;
    if (s.size() != 7 || s[0] != '#') {
        throw std::runtime_error(std::string(fn) + ": a colour must be \"#rrggbb\" or \"none\", got \"" + s + "\"");
    }
    out.r = hexPair(s, 1, fn);
    out.g = hexPair(s, 3, fn);
    out.b = hexPair(s, 5, fn);
    return true;
}

static uint32_t asDpi(const std::vector<Value>& a, size_t i, const char* fn) {
    if (isAbsent(a, i)) return 96;
    const double d = asNumber(a, i, fn, "dpi");
    if (d < 1 || d > (double)kMaxDpi || d != std::floor(d)) {
        throw std::runtime_error(std::string(fn) + ": dpi must be a whole number from 1 to " +
                                 std::to_string(kMaxDpi));
    }
    return (uint32_t)d;
}

static Value bytesValue(const std::vector<uint8_t>& v) {
    return Value(std::string((const char*)v.data(), v.size()));
}

static const std::string& asBytes(const std::vector<Value>& a, size_t i, const char* fn) {
    if (a.size() <= i || !a[i].isString()) {
        throw std::runtime_error(std::string(fn) + ": expected a string of bytes");
    }
    return a[i].stringVal;
}

static std::string reprCanvas(const std::shared_ptr<void>& h) {
    if (!h) return "<canvas>";
    const Canvas& c = *std::static_pointer_cast<Canvas>(h);
    return "<canvas " + std::to_string(c.w) + "x" + std::to_string(c.h) +
           " at " + std::to_string(c.dpi) + " dpi>";
}

// ── Registration ────────────────────────────────────────────────────────────

void registerBuiltins(const DefineFn& define) {
    registerHandleRepr(kCanvasTag, &reprCanvas);

    // bp_canvas_new(width, height [, dpi] [, background])
    // Width and height are USER units -- the same units an SVG figure uses --
    // so the pixel size is width * dpi / 96, and 96 dpi is one pixel per unit.
    define("bp_canvas_new", [](std::vector<Value> a) -> Value {
        const double w = asNumber(a, 0, "bp_canvas_new", "width");
        const double h = asNumber(a, 1, "bp_canvas_new", "height");
        const uint32_t dpi = asDpi(a, 2, "bp_canvas_new");
        Rgb bg{255, 255, 255};
        if (!isAbsent(a, 3) && !parseColour(a, 3, "bp_canvas_new", bg)) {
            throw std::runtime_error("bp_canvas_new: the background cannot be \"none\" -- a PNG has no "
                                     "transparency in this release; give it a colour");
        }
        return Value(std::static_pointer_cast<void>(std::make_shared<Canvas>(makeCanvas(w, h, dpi, bg))),
                     kCanvasTag);
    });

    // bp_canvas_info(canvas) -> {"width", "height", "dpi"} in DEVICE pixels.
    define("bp_canvas_info", [](std::vector<Value> a) -> Value {
        auto c = asCanvas(a, 0, "bp_canvas_info");
        ObjectMap m;
        m["width"] = Value((double)c->w);
        m["height"] = Value((double)c->h);
        m["dpi"] = Value((double)c->dpi);
        return Value(m);
    });

    // bp_canvas_clip(canvas, x, y, w, h) -- or no rectangle at all, to reset.
    // Whole device pixels, because that is all bplot needs (one clip per axes).
    define("bp_canvas_clip", [](std::vector<Value> a) -> Value {
        auto c = asCanvas(a, 0, "bp_canvas_clip");
        if (isAbsent(a, 1)) {
            c->clipX0 = 0; c->clipY0 = 0; c->clipX1 = c->w; c->clipY1 = c->h;
            return Value(true);
        }
        const uint32_t dpi = c->dpi;
        const int64_t x0 = toQ8(centi(asNumber(a, 1, "bp_canvas_clip", "x")), dpi) >> 8;
        const int64_t y0 = toQ8(centi(asNumber(a, 2, "bp_canvas_clip", "y")), dpi) >> 8;
        const int64_t x1 = (toQ8(centi(asNumber(a, 1, "bp_canvas_clip", "x") +
                                      asNumber(a, 3, "bp_canvas_clip", "width")), dpi) + 255) >> 8;
        const int64_t y1 = (toQ8(centi(asNumber(a, 2, "bp_canvas_clip", "y") +
                                      asNumber(a, 4, "bp_canvas_clip", "height")), dpi) + 255) >> 8;
        c->clipX0 = std::max<int64_t>(0, x0);
        c->clipY0 = std::max<int64_t>(0, y0);
        c->clipX1 = std::min<int64_t>(c->w, x1);
        c->clipY1 = std::min<int64_t>(c->h, y1);
        if (c->clipX1 < c->clipX0) c->clipX1 = c->clipX0;
        if (c->clipY1 < c->clipY0) c->clipY1 = c->clipY0;
        return Value(true);
    });

    // bp_fill_rect(canvas, x, y, w, h, colour [, opacity])
    define("bp_fill_rect", [](std::vector<Value> a) -> Value {
        auto c = asCanvas(a, 0, "bp_fill_rect");
        const double x = asNumber(a, 1, "bp_fill_rect", "x");
        const double y = asNumber(a, 2, "bp_fill_rect", "y");
        const double w = asNumber(a, 3, "bp_fill_rect", "width");
        const double h = asNumber(a, 4, "bp_fill_rect", "height");
        Rgb col{0, 0, 0};
        if (!parseColour(a, 5, "bp_fill_rect", col)) return Value(false);
        const uint32_t alpha = isAbsent(a, 6) ? 255u : alpha8(asNumber(a, 6, "bp_fill_rect", "opacity"));
        const uint32_t dpi = c->dpi;
        fillRectQ8(*c, toQ8(centi(x), dpi), toQ8(centi(y), dpi),
                        toQ8(centi(x + w), dpi), toQ8(centi(y + h), dpi), col, alpha);
        return Value(true);
    });

    // bp_canvas_pixel(canvas, x, y) -> "#rrggbb" at a whole device pixel.
    // For tests: it is how an assertion can name a colour at a coordinate.
    define("bp_canvas_pixel", [](std::vector<Value> a) -> Value {
        auto c = asCanvas(a, 0, "bp_canvas_pixel");
        const double xd = asNumber(a, 1, "bp_canvas_pixel", "x");
        const double yd = asNumber(a, 2, "bp_canvas_pixel", "y");
        if (xd < 0 || yd < 0 || xd >= (double)c->w || yd >= (double)c->h) {
            throw std::runtime_error("bp_canvas_pixel: (" + std::to_string((long long)xd) + ", " +
                std::to_string((long long)yd) + ") is outside a canvas of " +
                std::to_string(c->w) + "x" + std::to_string(c->h) + " pixels");
        }
        const uint8_t* p = c->rgb.data() + (((size_t)yd * c->w) + (size_t)xd) * 3;
        static const char* hex = "0123456789abcdef";
        std::string s = "#";
        for (int k = 0; k < 3; k++) { s += hex[p[k] >> 4]; s += hex[p[k] & 15]; }
        return Value(s);
    });

    // bp_canvas_raw(canvas) -> the pixels, three bytes each, row by row.
    define("bp_canvas_raw", [](std::vector<Value> a) -> Value {
        auto c = asCanvas(a, 0, "bp_canvas_raw");
        return bytesValue(c->rgb);
    });

    // bp_png(canvas) -> the PNG, as bytes.
    define("bp_png", [](std::vector<Value> a) -> Value {
        auto c = asCanvas(a, 0, "bp_png");
        return bytesValue(encodePng(c->rgb.data(), c->w, c->h, c->dpi));
    });

    // bp_png_save(canvas, path) -> bytes written.
    // Writes the file itself, in binary, rather than returning megabytes
    // through a Bantu string on the way to writefile().
    define("bp_png_save", [](std::vector<Value> a) -> Value {
        auto c = asCanvas(a, 0, "bp_png_save");
        if (a.size() < 2 || !a[1].isString() || a[1].stringVal.empty()) {
            throw std::runtime_error("bp_png_save: needs a file path");
        }
        const std::vector<uint8_t> png = encodePng(c->rgb.data(), c->w, c->h, c->dpi);
        std::ofstream f(a[1].stringVal, std::ios::out | std::ios::trunc | std::ios::binary);
        if (!f.is_open()) throw std::runtime_error("bp_png_save: cannot write '" + a[1].stringVal + "'");
        f.write((const char*)png.data(), (std::streamsize)png.size());
        f.flush();
        if (!f) throw std::runtime_error("bp_png_save: the write to '" + a[1].stringVal + "' failed");
        return Value((double)png.size());
    });

    // The pieces of the encoder, exposed so tests can check them against
    // known answers and against another implementation.
    define("bp_crc32", [](std::vector<Value> a) -> Value {
        const std::string& s = asBytes(a, 0, "bp_crc32");
        return Value((double)crc32((const uint8_t*)s.data(), s.size()));
    });
    define("bp_adler32", [](std::vector<Value> a) -> Value {
        const std::string& s = asBytes(a, 0, "bp_adler32");
        return Value((double)adler32((const uint8_t*)s.data(), s.size()));
    });
    define("bp_zlib", [](std::vector<Value> a) -> Value {
        const std::string& s = asBytes(a, 0, "bp_zlib");
        return bytesValue(zlibCompress((const uint8_t*)s.data(), s.size()));
    });
}

} // namespace bplot_raster
