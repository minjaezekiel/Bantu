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

// A flat list of numbers -- [x0, y0, x1, y1, ...] -- as device-space points.
// Capped, because the list came from a Bantu program that may be serving a
// request.
static const size_t kMaxPathPoints = 4000000;

static ContourQ8 pointsOf(const std::vector<Value>& a, size_t i, const char* fn, uint32_t dpi) {
    if (a.size() <= i || !a[i].isList()) {
        throw std::runtime_error(std::string(fn) + ": points must be a flat list [x0, y0, x1, y1, ...]");
    }
    const std::vector<Value>& v = a[i].listVal;
    if (v.size() % 2 != 0) {
        throw std::runtime_error(std::string(fn) + ": points must come in pairs, got " +
                                 std::to_string(v.size()) + " numbers");
    }
    if (v.size() / 2 > kMaxPathPoints) {
        throw std::runtime_error(std::string(fn) + ": more than " + std::to_string(kMaxPathPoints) + " points");
    }
    ContourQ8 out;
    out.reserve(v.size() / 2);
    for (size_t k = 0; k + 1 < v.size(); k += 2) {
        if (!v[k].isNumber() || !v[k + 1].isNumber()) {
            throw std::runtime_error(std::string(fn) + ": every point must be a number");
        }
        out.push_back(PointQ8{ toQ8(centi(v[k].numberVal), dpi), toQ8(centi(v[k + 1].numberVal), dpi) });
    }
    return out;
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

    // bp_fill_polygon(canvas, [x0, y0, x1, y1, ...], colour [, opacity])
    // Nonzero winding, as SVG fills by default.
    define("bp_fill_polygon", [](std::vector<Value> a) -> Value {
        auto c = asCanvas(a, 0, "bp_fill_polygon");
        Rgb col{0, 0, 0};
        if (!parseColour(a, 2, "bp_fill_polygon", col)) return Value(false);
        const uint32_t alpha = isAbsent(a, 3) ? 255u : alpha8(asNumber(a, 3, "bp_fill_polygon", "opacity"));
        std::vector<ContourQ8> contours{ pointsOf(a, 1, "bp_fill_polygon", c->dpi) };
        if (contours[0].size() < 3) return Value(false);
        compositeMask(*c, rasterise(contours, c->clipX0, c->clipY0, c->clipX1, c->clipY1), col, alpha);
        return Value(true);
    });

    // bp_stroke_polyline(canvas, points, colour, width [, opacity] [, dash] [, roundCap])
    // The whole stroke -- segments, joins and caps -- is one mask, composited
    // once, so a translucent line does not darken at every joint.
    define("bp_stroke_polyline", [](std::vector<Value> a) -> Value {
        auto c = asCanvas(a, 0, "bp_stroke_polyline");
        Rgb col{0, 0, 0};
        if (!parseColour(a, 2, "bp_stroke_polyline", col)) return Value(false);
        const double w = asNumber(a, 3, "bp_stroke_polyline", "width");
        if (w <= 0) return Value(false);                    // a zero-width stroke draws nothing
        const uint32_t alpha = isAbsent(a, 4) ? 255u : alpha8(asNumber(a, 4, "bp_stroke_polyline", "opacity"));
        const bool roundCap = isAbsent(a, 6) ? true : a[6].isTruthy();
        const ContourQ8 pts = pointsOf(a, 1, "bp_stroke_polyline", c->dpi);
        if (pts.size() < 2) return Value(false);

        std::vector<int64_t> pattern;
        if (!isAbsent(a, 5)) {
            if (!a[5].isString()) throw std::runtime_error("bp_stroke_polyline: a dash must be a string like \"6,4\"");
            const std::string& d = a[5].stringVal;
            size_t i = 0;
            while (i < d.size()) {
                while (i < d.size() && (d[i] == ',' || d[i] == ' ')) i++;
                if (i >= d.size()) break;
                const char* start = d.c_str() + i;
                char* end = nullptr;
                const double v = std::strtod(start, &end);
                if (end == start) throw std::runtime_error("bp_stroke_polyline: a dash must be numbers, like \"6,4\"");
                i += (size_t)(end - start);
                if (v <= 0) throw std::runtime_error("bp_stroke_polyline: a dash length must be positive");
                pattern.push_back(toQ8(centi(v), c->dpi));
            }
            if (pattern.size() == 1) pattern.push_back(pattern[0]);
        }

        // A stroke wider than 65,536 pixels covers any canvas this can make.
        const int64_t widthQ8 = std::min<int64_t>(toQ8(centi(w), c->dpi), (int64_t)1 << 24);
        // Clip to a guard box half a million pixels beyond the clip before
        // anything squares a length (raster_core.hpp, clipPolyline). ponytail: a
        // dash pattern restarts where a line enters the guard box -- only for a
        // line reaching that far off the canvas.
        const int64_t g = ((int64_t)1 << 27) + widthQ8;
        std::vector<ContourQ8> pieces;
        for (const auto& inside : clipPolyline(pts, c->clipX0 * 256 - g, c->clipY0 * 256 - g,
                                                    c->clipX1 * 256 + g, c->clipY1 * 256 + g)) {
            for (const auto& run : applyDash(inside, pattern)) {
                strokeContours(run, widthQ8, roundCap, pieces);
            }
        }
        if (pieces.empty()) return Value(false);
        compositeMask(*c, rasterise(pieces, c->clipX0, c->clipY0, c->clipX1, c->clipY1), col, alpha);
        return Value(true);
    });

    // bp_fill_path(canvas, "M ... Z", colour [, opacity])
    define("bp_fill_path", [](std::vector<Value> a) -> Value {
        auto c = asCanvas(a, 0, "bp_fill_path");
        if (a.size() < 2 || !a[1].isString()) throw std::runtime_error("bp_fill_path: the path must be a string");
        Rgb col{0, 0, 0};
        if (!parseColour(a, 2, "bp_fill_path", col)) return Value(false);
        const uint32_t alpha = isAbsent(a, 3) ? 255u : alpha8(asNumber(a, 3, "bp_fill_path", "opacity"));
        PathParser parser(a[1].stringVal, c->dpi);
        const std::vector<ContourQ8> contours = parser.parse(kMaxPathPoints);
        if (contours.empty()) return Value(false);
        compositeMask(*c, rasterise(contours, c->clipX0, c->clipY0, c->clipX1, c->clipY1), col, alpha);
        return Value(true);
    });

    // bp_text(canvas, x, y, text, size, colour [, anchor] [, rotate] [, opacity])
    // The same arguments as the SVG backend's text(): (x, y) is the start of the
    // BASELINE, anchor is "start" | "middle" | "end", rotate is degrees clockwise
    // about (x, y). Invalid UTF-8 and characters the font lacks draw as U+FFFD.
    define("bp_text", [](std::vector<Value> a) -> Value {
        auto c = asCanvas(a, 0, "bp_text");
        const double x = asNumber(a, 1, "bp_text", "x");
        const double y = asNumber(a, 2, "bp_text", "y");
        if (a.size() < 4 || !a[3].isString()) throw std::runtime_error("bp_text: the text must be a string");
        const double size = asNumber(a, 4, "bp_text", "size");
        Rgb col{0, 0, 0};
        if (!parseColour(a, 5, "bp_text", col)) return Value(false);
        int anchor = 0;
        if (!isAbsent(a, 6)) {
            const std::string an = a[6].isString() ? a[6].stringVal : "";
            if (an == "middle") anchor = 1;
            else if (an == "end") anchor = 2;
            else if (an != "start") throw std::runtime_error("bp_text: anchor must be \"start\", \"middle\" or \"end\"");
        }
        // Degrees to a step of the 256-entry circle table, from the same
        // hundredths every coordinate goes through: integer from here on.
        const int rot = isAbsent(a, 7) ? 0
            : (int)(divRound(centi(asNumber(a, 7, "bp_text", "rotate")) % 36000 * 256, 36000) % 256);
        const uint32_t alpha = isAbsent(a, 8) ? 255u : alpha8(asNumber(a, 8, "bp_text", "opacity"));

        const std::vector<uint32_t> cps = decodeUtf8(a[3].stringVal);
        if (cps.size() > kMaxTextChars) {
            throw std::runtime_error("bp_text: more than " + std::to_string(kMaxTextChars) + " characters");
        }
        const int64_t sizeQ8 = toQ8(centi(size), c->dpi);
        if (sizeQ8 > kMaxFontQ8) throw std::runtime_error("bp_text: a font over 4096 pixels tall");
        if (sizeQ8 <= 0 || cps.empty()) return Value(false);
        const uint32_t dpi = c->dpi;
        const std::vector<ContourQ8> contours =
            textContours(cps, toQ8(centi(x), dpi), toQ8(centi(y), dpi), sizeQ8, anchor, rot);
        if (contours.empty()) return Value(false);            // all spaces
        compositeMask(*c, rasterise(contours, c->clipX0, c->clipY0, c->clipX1, c->clipY1), col, alpha);
        return Value(true);
    });

    // bp_text_width(text, size) -> the advance in USER units, for layout. It
    // measures the font the PNG draws with, which is wider than the SVG's.
    define("bp_text_width", [](std::vector<Value> a) -> Value {
        if (a.empty() || !a[0].isString()) throw std::runtime_error("bp_text_width: the text must be a string");
        const double size = asNumber(a, 1, "bp_text_width", "size");
        return Value((double)textAdvance(decodeUtf8(a[0].stringVal)) * size / kFontUnitsPerEm);
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
