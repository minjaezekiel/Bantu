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
//  WHAT IS IN THIS BUILD: phase P5. The object model, the full 20-mode
//  table, the conversion matrix, ImageColor, ImagePalette, the six
//  resampling filters with resize/rotate/transpose/thumbnail/reduce/
//  transform, ImageTransform, ImagePath, PNG, BMP, TGA and PPM read and
//  write, and -- new in P5 -- JPEG read and write (baseline, progressive,
//  every subsampling, restart markers, draft-mode scaled decoding) with
//  EXIF and ExifTags. Phases P6-P9 add the operation surface, drawing,
//  text and the remaining formats; pillow-suite/ROADMAP.md tracks them
//  item by item.
// ════════════════════════════════════════════════════════════════════════

include "./mode.b" as mode;
include "./pxbuf.b" as pxbuf;
include "./color.b" as color;
include "./convert.b" as convert;
include "./resample.b" as resample;
include "./transform.b" as xform;
include "./path.b" as path;
include "./image.b" as image;
include "./codecs/registry.b" as registry;
include "./codecs/ppm.b" as ppm;
include "./codecs/png.b" as png;
include "./codecs/bmp.b" as bmp;
include "./codecs/jpeg.b" as jpeg;
include "./exif.b" as exif;
include "./codecs/tga.b" as tga;
include "./inflate.b" as inflate;
include "./deflate.b" as deflate;
include "./bitio.b" as bitio;

$VERSION = "0.5.0";

// ─── Register the codecs this build ships ───────────────────────────────
//
// Each registers a sniffer, an opener and a saver over BUFFERS, never over
// Image objects -- see codecs/registry.b for why that layering matters.
registry.register("PNG", ["png"], "image/png",
                  def($head) { return png.sniff($head); },
                  def($data) { return png.decode($data); },
                  def($spec) { return png.encode($spec); });

registry.register("BMP", ["bmp", "dib"], "image/bmp",
                  def($head) { return bmp.sniff($head); },
                  def($data) { return bmp.decode($data); },
                  def($spec) { return bmp.encode($spec); });

registry.register("JPEG", ["jpg", "jpeg", "jpe", "jfif"], "image/jpeg",
                  def($head) { return jpeg.sniff($head); },
                  def($data) { return jpeg.decode($data); },
                  def($spec) { return jpeg.encode($spec); });

// TGA LAST, and with its sniffer registered: a TGA has NO magic number, so
// its sniffer validates the whole 18-byte header for self-consistency rather
// than comparing a signature. That is weaker than a real signature, so it is
// tried after every format that has one, and the extension remains the
// reliable route.
registry.register("TGA", ["tga", "icb", "vda", "vst"], "image/x-tga",
                  def($head) { return tga.sniff($head); },
                  def($data) { return tga.decode($data); },
                  def($spec) { return tga.encode($spec); });

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

// ─── ImageTransform: the four transform classes ─────────────────────────
$AFFINE      = 0;
$EXTENT      = 1;
$PERSPECTIVE = 2;
$QUAD        = 3;
$MESH        = 4;

def AffineTransform($m)      { return xform.AffineTransform($m); }
def ExtentTransform($box)    { return xform.ExtentTransform($box); }
def PerspectiveTransform($m) { return xform.PerspectiveTransform($m); }
def QuadTransform($q)        { return xform.QuadTransform($q); }
def MeshTransform($e)        { return xform.MeshTransform($e); }
def transform_method_name($m) { return xform.method_name($m); }

// ─── ImagePath ──────────────────────────────────────────────────────────
def Path($points)            { return path.Path($points); }
def path_getbbox($p)         { return path.getbbox($p); }
def path_tolist($p, $flat)   { return path.tolist($p, $flat); }
def path_transform($p, $m)   { return path.transform($p, $m); }
def path_compact($p, $d)     { return path.compact($p, $d); }
def path_len($p)             { return path.len_points($p); }

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

// The raw codec entry points, for a caller that needs an option the Image
// layer does not surface -- a specific PNG scanline filter, say, or a TGA
// written run-length encoded.
def encode_png($spec) { return png.encode($spec); }
def encode_bmp($spec) { return bmp.encode($spec); }
def encode_tga($spec) { return tga.encode($spec); }
def encode_jpeg($spec) { return jpeg.encode($spec); }
def decode_jpeg($data, $denom, $want) { return jpeg.decode_scaled($data, $denom, $want); }
def jpeg_max_progressive_coeffs()       { return jpeg.max_progressive_coeffs(); }
def jpeg_set_max_progressive_coeffs($n) { return jpeg.set_max_progressive_coeffs($n); }
def png_allow_truncated($yes) { return png.allow_truncated($yes); }
def adler32($data)                  { return deflate.adler32($data); }

// ─── EXIF and ExifTags ──────────────────────────────────────────────────
//
// EXIF is a TIFF block inside a JPEG APP1 segment, so exif.b is the TIFF
// tag reader; codecs/tiff.b will include it rather than repeat it.
def getexif($im)              { if ($im.info.exif == null) { return exif.empty(); }
                                return exif.load($im.info.exif); }
def exif_load($data)          { return exif.load($data); }
def exif_empty()              { return exif.empty(); }
def exif_tobytes($ex)         { return exif.tobytes($ex); }
def exif_get($ex, $ifd, $tag) { return exif.get($ex, $ifd, $tag); }
def exif_put($ex, $ifd, $tag, $v) { return exif.put($ex, $ifd, $tag, $v); }
def exif_remove($ex, $ifd, $tag)  { return exif.remove($ex, $ifd, $tag); }
def exif_transpose($im)       { return exif.exif_transpose($im); }
def exif_tag_name($t)         { return exif.tag_name($t); }
def exif_gps_tag_name($t)     { return exif.gps_tag_name($t); }
def TAGS()                    { return exif.TAGS(); }
def GPSTAGS()                 { return exif.GPSTAGS(); }
def IFD()                     { return exif.IFD(); }
def ExifBase()                { return exif.BASE(); }
def GPS()                     { return exif.GPS(); }
def Interop()                 { return exif.INTEROP(); }

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
    print("  geometry   resize rotate transpose thumbnail reduce transform draft");
    print("  ExifTags   getexif exif_load exif_put exif_tobytes exif_transpose");
    print("             TAGS GPSTAGS IFD ExifBase GPS Interop");
    print("  ImagePath  Path path_getbbox path_tolist path_transform path_compact");
    print("");
    print("  This build is phase P5. The operation surface, drawing, text and");
    print("  the remaining formats arrive in P6-P8 -- see pillow-suite/ROADMAP.md.");
    return null;
}
