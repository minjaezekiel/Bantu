// ════════════════════════════════════════════════════════════════════════
//  pillow/pillow.b — the umbrella. Include this and you have the package.
//
//      include "./pillow/pillow.b" as pillow;
//
//      $im = pillow.new("RGB", [320, 240], "cornflowerblue");
//      $im = pillow.open("photo.ppm");
//      $grey = $im.convert("L");
//      $grey.save("grey.ppm");
//
//  Pillow's module layout is flat here -- Image, ImageColor and the rest are
//  reachable as pillow.<name> -- because Bantu has one namespace per include
//  and nesting them would mean a caller writing pillow.Image.new, which is
//  one more indirection for nothing. The names themselves are Pillow's.
//
//  WHAT IS IN THIS BUILD: phase P2. The object model, the full 20-mode
//  table, the conversion matrix, ImageColor, and PPM/PGM/PBM read and write.
//  Phases P3-P9 add resampling, PNG, JPEG, the operators, drawing, text and
//  the remaining formats; pillow-suite/ROADMAP.md tracks them item by item.
// ════════════════════════════════════════════════════════════════════════

include "./mode.b" as mode;
include "./pxbuf.b" as pxbuf;
include "./color.b" as color;
include "./convert.b" as convert;
include "./resample.b" as resample;
include "./image.b" as image;
include "./codecs/registry.b" as registry;
include "./codecs/ppm.b" as ppm;
include "./inflate.b" as inflate;
include "./deflate.b" as deflate;
include "./bitio.b" as bitio;

$VERSION = "0.2.0";

// ─── Register the codecs this build ships ───────────────────────────────
//
// Each registers a sniffer, an opener and a saver over BUFFERS, never over
// Image objects -- see codecs/registry.b for why that layering matters.
registry.register("PPM", ["ppm", "pgm", "pbm", "pnm"], "image/x-portable-anymap",
                  def($head) { return ppm.sniff($head); },
                  def($data) { return ppm.decode($data); },
                  def($spec) { return ppm.encode($spec); });

// ─── Image.* ────────────────────────────────────────────────────────────
//
// `new`, `open` and `eval` are module VARIABLES holding a function, because
// the first is a Bantu keyword and the second a builtin, so `def new(...)`
// does not parse. A member access spells them exactly as Pillow does, so a
// caller still writes pillow.new(...) / pillow.open(...) / pillow.eval(...).
$new  = def($m, $size, $colour) { return image.new_image($m, $size, $colour); };
$open = def($path)              { return image.open_image($path); };
$eval = def($im, $fn)           { return image.eval_image($im, $fn); };

def new_image($m, $size, $colour) { return image.new_image($m, $size, $colour); }
def open_image($path)             { return image.open_image($path); }
def eval_image($im, $fn)          { return image.eval_image($im, $fn); }

def open_bytes($data, $hintPath)  { return image.open_bytes($data, $hintPath); }
def frombytes($m, $size, $data)   { return image.frombytes($m, $size, $data); }
def frombuffer($m, $size, $data)  { return image.frombytes($m, $size, $data); }
def fromarray($a, $m)             { return image.fromarray($a, $m); }
def merge($m, $bands)             { return image.merge($m, $bands); }
def blend($a, $b, $alpha)         { return image.blend($a, $b, $alpha); }
def composite($a, $b, $mask)      { return image.composite($a, $b, $mask); }
def alpha_composite($dst, $src)   { return image.alpha_composite($dst, $src); }

// ─── ImageColor ─────────────────────────────────────────────────────────
def getrgb($spec)       { return color.getrgb($spec); }
def getcolor($spec, $m) { return color.getcolor($spec, $m); }
def colormap()          { return color.colormap(); }
def color_names()       { return color.names(); }

// ─── Mode introspection (Pillow's ImageMode) ────────────────────────────
def getmodebands($m)     { return mode.bands($m); }
def getmodebase($m)      { return mode.basemode($m); }
def getmodetype($m)      { return mode.basetype($m); }
def getmodebandnames($m) { return mode.band_names($m); }
def is_valid_mode($m)    { return mode.is_valid($m); }

// ─── The plugin registry (Pillow's Image.register_*) ────────────────────
def register_codec($id, $exts, $mime, $sniff, $openFn, $saveFn) {
    return registry.register($id, $exts, $mime, $sniff, $openFn, $saveFn);
}
def registered_extensions() { return registry.extensions(); }
def formats()               { return registry.formats(); }
def identify($data, $path)  { return registry.identify($data, $path); }

// ─── Configuration ──────────────────────────────────────────────────────
//
// MAX_IMAGE_PIXELS is the decompression-bomb guard, checked from a file's
// HEADER before any pixel memory is allocated. Pillow's default is the same
// number. Raise it only for a source you trust.
def max_image_pixels()       { return image.CFG.MAX_IMAGE_PIXELS; }
def set_max_image_pixels($n) { image.CFG.MAX_IMAGE_PIXELS = $n; return $n; }

// ─── Constants, with Pillow's values ────────────────────────────────────
$NEAREST  = 0;
$LANCZOS  = 1;
$BILINEAR = 2;
$BICUBIC  = 3;
$BOX      = 4;
$HAMMING  = 5;

def filter_name($f) { return resample.filter_name($f); }
def filter_id($n)   { return resample.filter_id($n); }

$FLIP_LEFT_RIGHT = 0;
$FLIP_TOP_BOTTOM = 1;
$ROTATE_90  = 2;
$ROTATE_180 = 3;
$ROTATE_270 = 4;
$TRANSPOSE  = 5;
$TRANSVERSE = 6;

$NONE           = "NONE";
$FLOYDSTEINBERG = "FLOYDSTEINBERG";

$MODES = mode.MODES;

// ─── Compression, exposed because it is useful on its own ───────────────
//
// The package needs these for PNG; a caller gets them because there is no
// reason to hide a working zlib behind an image library.
def zlib_decompress($data, $maxOut) { return inflate.inflate($data, $maxOut); }
def zlib_compress($data, $level)    { return deflate.zlib($data, $level); }
def gzip_decompress($data, $maxOut) { return inflate.gunzip($data, $maxOut); }
def gzip_compress($data, $level)    { return deflate.gzip($data, $level); }
def crc32($data)                    { return deflate.crc32($data); }
def adler32($data)                  { return deflate.adler32($data); }

// ─── help() ─────────────────────────────────────────────────────────────
def help() {
    print("pillow " + $VERSION + " -- imaging for Bantu, in the shape of Python's Pillow");
    print("");
    print("  $im = pillow.new(\"RGB\", [320, 240], \"cornflowerblue\");");
    print("  $im = pillow.open(\"photo.ppm\");");
    print("  $im.convert(\"L\").save(\"grey.ppm\");");
    print("");
    print("  modes   : " + join(mode.ALL_MODES, " "));
    print("  formats : " + join(registry.formats(), " "));
    print("");
    print("  Image      new open frombytes frombuffer fromarray merge blend");
    print("             composite alpha_composite eval");
    print("  instance   mode size width height info palette getbands load close");
    print("             convert copy crop paste split getchannel putalpha");
    print("             getpixel putpixel getdata putdata point histogram");
    print("             getbbox getextrema getcolors tobytes save show");
    print("  ImageColor getrgb getcolor colormap");
    print("");
    print("  This build is phase P2. Resampling, PNG, JPEG, the operators,");
    print("  drawing and text arrive in P3-P8 -- see pillow-suite/ROADMAP.md.");
    return null;
}
