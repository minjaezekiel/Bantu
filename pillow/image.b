// ════════════════════════════════════════════════════════════════════════
//  pillow/image.b — the Image object and the Image.* namespace.
//
//  This is the layer that orchestrates. Everything below it (mode, pxbuf,
//  convert, the codecs) works on pixel BUFFERS and knows nothing about an
//  Image; this file is the only place that knows both.
//
//      include "./pillow.b" as pillow;
//      $im = pillow.new("RGB", [320, 240], "cornflowerblue");
//      $im = pillow.open("photo.ppm").convert("L");
//      $im.save("grey.ppm");
//
//  WHY FACTORY FUNCTIONS. `new alias.Class()` does not parse across a module
//  boundary in Bantu, so PillowImage is never constructed by a caller --
//  new(), open(), frombytes() and the methods return instances made in here.
//  bplot has the same arrangement for the same reason.
//
//  SIZE IS [width, height] because that is what Pillow's users type. The
//  buffer is (height, width, bands) because that is what a row-major array
//  wants. The two orders meet in _wrap() and nowhere else.
// ════════════════════════════════════════════════════════════════════════

include "./mode.b" as mode;
include "./pxbuf.b" as pxbuf;
include "./color.b" as color;
include "./convert.b" as convert;
include "./resample.b" as resample;
include "./transform.b" as xform;
include "./codecs/registry.b" as registry;

// Pillow's guard against a decompression bomb, with Pillow's default.
$CFG = {"MAX_IMAGE_PIXELS": 89478485};

// Resampling filter constants, Pillow's values (used from P3 onward).
$NEAREST = 0;
$LANCZOS = 1;
$BILINEAR = 2;
$BICUBIC = 3;
$BOX = 4;
$HAMMING = 5;

// Transpose constants, Pillow's values.
$FLIP_LEFT_RIGHT = 0;
$FLIP_TOP_BOTTOM = 1;
$ROTATE_90 = 2;
$ROTATE_180 = 3;
$ROTATE_270 = 4;
$TRANSPOSE = 5;
$TRANSVERSE = 6;

// Dither and palette constants.
$DITHER_NONE = "NONE";
$FLOYDSTEINBERG = "FLOYDSTEINBERG";


// ════════════════════════════════════════════════════════════════════════
//  The image
// ════════════════════════════════════════════════════════════════════════
class PillowImage {
    def init($m, $w, $h, $px, $palette, $info, $format) {
        this.mode = $m;
        this.size = [$w, $h];
        this.width = $w;
        this.height = $h;
        this.px = $px;
        this.palette = $palette;
        this.info = $info;
        this.format = $format;
        this.readonly = false;
    }

    // ─── identity ───────────────────────────────────────────────────────
    def getbands() { return mode.band_names(this.mode); }

    // load() exists for Pillow compatibility: an Image here is never lazy
    // beyond open(), so it is a no-op that returns the image.
    def load() { return this; }

    def close() { return null; }

    def __repr() {
        return "<PillowImage mode=" + this.mode + " size=" +
               str(this.width) + "x" + str(this.height) + ">";
    }

    // ─── copying and geometry ───────────────────────────────────────────
    def copy() {
        return _wrap(this.mode, pxbuf.materialise(this.px, mode.dtype(this.mode)),
                     _copyPalette(this.palette), _copyInfo(this.info), this.format);
    }

    // crop($box) -> a new image of box = [left, upper, right, lower].
    //
    // Pillow allows a box OUTSIDE the image and fills the outside with zero,
    // which is relied on by ImageOps.expand and by every tiling script, so it
    // is supported here rather than rejected.
    def crop($box) {
        if ($box == null) { return this.copy(); }
        if (type($box) != "list" || len($box) != 4) {
            throw "Image.crop: box must be [left, upper, right, lower]";
        }
        $l = floor($box[0]);
        $t = floor($box[1]);
        $r = floor($box[2]);
        $b = floor($box[3]);
        if ($r < $l || $b < $t) {
            throw "Image.crop: box " + str($box) + " has a negative width or height";
        }
        $ow = $r - $l;
        $oh = $b - $t;
        $dt = mode.dtype(this.mode);
        $out = pxbuf.make($oh, $ow, mode.bands(this.mode), $dt);
        // The overlap with the source, if any.
        $sl = max($l, 0);
        $st = max($t, 0);
        $sr = min($r, this.width);
        $sb = min($b, this.height);
        if ($sr > $sl && $sb > $st) {
            $src = pxbuf.crop(this.px, $sl, $st, $sr, $sb);
            $dst = pxbuf.crop($out, $sl - $l, $st - $t, $sr - $l, $sb - $t);
            pxbuf.copy_into($dst, $src);
        }
        return _wrap(this.mode, $out, _copyPalette(this.palette),
                     _copyInfo(this.info), null);
    }

    // paste($src, $box, $mask) -> src pasted in place. Returns null, as
    // Pillow's paste mutates and returns None.
    //
    // $src is an image or a colour. $box is [left, upper] or
    // [left, upper, right, lower] or null (meaning [0, 0]).
    def paste($src, $box, $mask) {
        $l = 0;
        $t = 0;
        if ($box != null) {
            if (type($box) != "list" || (len($box) != 2 && len($box) != 4)) {
                throw "Image.paste: box must be [left, upper] or [left, upper, right, lower]";
            }
            $l = floor($box[0]);
            $t = floor($box[1]);
        }

        if (type($src) == "instance") {
            $sw = $src.width;
            $sh = $src.height;
            $srcPx = $src.px;
            if ($src.mode != this.mode) {
                $conv = $src.convert(this.mode, null, null);
                $srcPx = $conv.px;
            }
        } else {
            // a colour: the box must give a size, or the whole image is filled
            if ($box != null && len($box) == 4) {
                $sw = floor($box[2]) - $l;
                $sh = floor($box[3]) - $t;
            } else {
                $sw = this.width;
                $sh = this.height;
                $l = 0;
                $t = 0;
            }
            $srcPx = pxbuf.make($sh, $sw, mode.bands(this.mode), mode.dtype(this.mode));
            pxbuf.fill($srcPx, _colourList($src, this.mode));
        }

        // Clip the paste to the destination; a box partly outside is legal.
        $dl = max($l, 0);
        $dt2 = max($t, 0);
        $dr = min($l + $sw, this.width);
        $db = min($t + $sh, this.height);
        if ($dr <= $dl || $db <= $dt2) { return null; }

        $sub = pxbuf.crop($srcPx, $dl - $l, $dt2 - $t, $dr - $l, $db - $t);
        $dst = pxbuf.crop(this.px, $dl, $dt2, $dr, $db);

        if ($mask == null) {
            pxbuf.copy_into($dst, $sub);
            return null;
        }

        // A mask blends: out = dst + (src - dst) * m / 255, in integers.
        // One pass of array operations, no pixel loop.
        $mk = $mask;
        if (type($mask) == "instance") {
            if ($mask.mode != "L" && $mask.mode != "1" && $mask.mode != "LA" &&
                $mask.mode != "RGBA") {
                $mk = $mask.convert("L", null, null);
            }
            $mpx = $mk.px;
            if (pxbuf.nbands($mpx) > 1) {
                $mpx = pxbuf.band($mpx, pxbuf.nbands($mpx) - 1);
            }
        } else {
            throw "Image.paste: mask must be an image or null";
        }
        $msub = pxbuf.crop($mpx, $dl - $l, $dt2 - $t, $dr - $l, $db - $t);
        $mb = pxbuf.splat($msub, pxbuf.nbands($dst));
        $diff = nd_subtract($sub, $dst);
        $delta = nd_floor_divide(nd_add(nd_multiply($diff, $mb), 127), 255);
        nd_add($dst, $delta, $dst);
        return null;
    }

    // ─── bands ──────────────────────────────────────────────────────────
    def split() {
        $out = [];
        $n = mode.bands(this.mode);
        $single = mode.basetype(this.mode);
        if ($single == "I") { $single = "I"; }
        if ($single == "F") { $single = "F"; }
        if ($single == "L") { $single = "L"; }
        $i = 0;
        while ($i < $n) {
            $b = pxbuf.materialise(pxbuf.band(this.px, $i), mode.dtype(this.mode));
            push($out, _wrap($single, $b, null, {}, null));
            $i = $i + 1;
        }
        return $out;
    }

    def getchannel($which) {
        $names = mode.band_names(this.mode);
        if (type($which) == "string") {
            $i = indexOf($names, $which);
            if ($i < 0) {
                throw "Image.getchannel: '" + $which + "' is not a band of " +
                      this.mode + " (" + join($names, ", ") + ")";
            }
        } else {
            $i = floor($which);
            if ($i < 0 || $i >= len($names)) {
                throw "Image.getchannel: band " + str($i) + " of a " +
                      str(len($names)) + "-band image";
            }
        }
        return this.split()[$i];
    }

    def putalpha($a) {
        $target = this.mode;
        if (!mode.has_alpha($target)) {
            if (mode.is_rgb_like($target)) { $target = "RGBA"; }
            else {
                if (mode.is_grey($target)) { $target = "LA"; }
                else { throw "Image.putalpha: " + this.mode + " cannot carry alpha"; }
            }
            $conv = this.convert($target, null, null);
            this.mode = $conv.mode;
            this.px = $conv.px;
        }
        $ai = mode.alpha_band(this.mode);
        if (type($a) == "instance") {
            $am = $a;
            if ($a.mode != "L") { $am = $a.convert("L", null, null); }
            if ($am.width != this.width || $am.height != this.height) {
                throw "Image.putalpha: the alpha image is " + str($am.width) + "x" +
                      str($am.height) + ", the image is " + str(this.width) + "x" +
                      str(this.height);
            }
            pxbuf.set_band(this.px, $ai, $am.px);
        } else {
            pxbuf.set_band(this.px, $ai, floor($a));
        }
        return null;
    }

    // ─── pixels ─────────────────────────────────────────────────────────
    //
    // getpixel/putpixel are SLOW -- one nd_get is ~2.5 us -- and that is not
    // a defect to be fixed but the reason point(), paste() and the operators
    // exist. Pillow's own getpixel is slow for the same reason. A loop over
    // every pixel through here is what rule PW2 forbids.
    def getpixel($xy) {
        if (type($xy) != "list" || len($xy) != 2) {
            throw "Image.getpixel: expected [x, y]";
        }
        $v = pxbuf.pixel(this.px, floor($xy[0]), floor($xy[1]));
        if (len($v) == 1) { return $v[0]; }
        return $v;
    }

    def putpixel($xy, $value) {
        if (type($xy) != "list" || len($xy) != 2) {
            throw "Image.putpixel: expected [x, y]";
        }
        pxbuf.put_pixel(this.px, floor($xy[0]), floor($xy[1]),
                        _colourList($value, this.mode));
        return null;
    }

    // getdata() -> every pixel, flattened, in row-major order: one value per
    // pixel for a single-band image, a list per pixel otherwise.
    def getdata() {
        $n = mode.bands(this.mode);
        $flat = nd_to_list(nd_ravel(this.px));
        if ($n == 1) { return $flat; }
        $out = [];
        $i = 0;
        while ($i < len($flat)) {
            $p = [];
            $k = 0;
            while ($k < $n) { push($p, $flat[$i + $k]); $k = $k + 1; }
            push($out, $p);
            $i = $i + $n;
        }
        return $out;
    }

    def putdata($data, $scale, $offset) {
        $n = mode.bands(this.mode);
        $want = this.width * this.height;
        if (len($data) > $want) {
            throw "Image.putdata: " + str(len($data)) + " pixels for an image of " + str($want);
        }
        $sc = 1.0;
        $off = 0.0;
        if ($scale != null) { $sc = $scale; }
        if ($offset != null) { $off = $offset; }
        $flat = [];
        $i = 0;
        while ($i < len($data)) {
            $p = $data[$i];
            if (type($p) == "list") {
                $k = 0;
                while ($k < $n) { push($flat, $p[$k] * $sc + $off); $k = $k + 1; }
            } else {
                $k = 0;
                while ($k < $n) { push($flat, $p * $sc + $off); $k = $k + 1; }
            }
            $i = $i + 1;
        }
        // Only the leading pixels are replaced, as in Pillow.
        $count = len($data);
        if ($count == 0) { return null; }
        $strip = nd_reshape(nd($flat, mode.dtype(this.mode)), [$count * $n]);
        $dstFlat = nd_slice(nd_ravel(this.px), [[0, $count * $n]]);
        nd_add($strip, 0, $dstFlat);
        return null;
    }

    // point($lut) -> every value mapped through a lookup table. ONE gather,
    // so a 1080p image costs ~90 ms whatever the table says.
    def point($lut, $m) {
        if (mode.is_float(this.mode)) {
            throw "Image.point: mode F has no lookup table; use ImageMath instead";
        }
        $hi = mode.hi(this.mode);
        $table = $lut;
        if (type($lut) == "function") {
            // a function: evaluate it once per possible value, not per pixel
            $table = [];
            $v = 0;
            while ($v <= $hi && $v < 65536) { push($table, $lut($v)); $v = $v + 1; }
        }
        if (type($table) != "list") { throw "Image.point: expected a list or a function"; }
        $need = $hi + 1;
        if ($need > 65536) { $need = 65536; }
        if (len($table) == $need * mode.bands(this.mode)) {
            // Pillow allows one table per band, concatenated.
            $parts = [];
            $b = 0;
            while ($b < mode.bands(this.mode)) {
                $sub = [];
                $k = 0;
                while ($k < $need) { push($sub, $table[$b * $need + $k]); $k = $k + 1; }
                push($parts, pxbuf.lut(pxbuf.band(this.px, $b), $sub));
                $b = $b + 1;
            }
            $out = pxbuf.stack($parts, "i64");
        } else {
            if (len($table) < $need) {
                throw "Image.point: the table has " + str(len($table)) +
                      " entries, " + str($need) + " are needed for mode " + this.mode;
            }
            $out = pxbuf.lut(this.px, $table);
        }
        $outMode = this.mode;
        if ($m != null) { $outMode = $m; }
        return _wrap($outMode, $out, _copyPalette(this.palette), _copyInfo(this.info), null);
    }

    // ─── statistics ─────────────────────────────────────────────────────
    def getextrema() {
        $n = mode.bands(this.mode);
        if ($n == 1) {
            return [pxbuf.scalar(nd_min(this.px)), pxbuf.scalar(nd_max(this.px))];
        }
        $out = [];
        $i = 0;
        while ($i < $n) {
            $b = pxbuf.band(this.px, $i);
            push($out, [pxbuf.scalar(nd_min($b)), pxbuf.scalar(nd_max($b))]);
            $i = $i + 1;
        }
        return $out;
    }

    // histogram() -> Pillow's flat histogram: 256 bins per band, concatenated.
    def histogram($mask) {
        if (mode.is_float(this.mode)) {
            throw "Image.histogram: mode F has no histogram; use ImageStat instead";
        }
        if ($mask != null) {
            throw "Image.histogram: a mask is not implemented yet (phase P6)";
        }
        $n = mode.bands(this.mode);
        $out = [];
        $i = 0;
        while ($i < $n) {
            $b = nd_clip(pxbuf.band(this.px, $i), 0, 255);
            $h = nd_bincount(nd_ravel($b));
            $hl = nd_to_list($h);
            $k = 0;
            while ($k < 256) {
                if ($k < len($hl)) { push($out, $hl[$k]); } else { push($out, 0); }
                $k = $k + 1;
            }
            $i = $i + 1;
        }
        return $out;
    }

    // getbbox() -> [l, t, r, b] of the non-zero region, or null if all zero.
    //
    // Vectorised: one reduction per axis, then a search. No pixel loop.
    def getbbox() {
        if (this.width == 0 || this.height == 0) { return null; }
        $any = nd_not_equal(this.px, 0);
        // collapse bands, then each axis
        $perPixel = nd_any($any, 2);                      // (h, w)
        $rows = nd_to_list(nd_any($perPixel, 1));         // h booleans
        $cols = nd_to_list(nd_any($perPixel, 0));         // w booleans
        $t = -1;
        $i = 0;
        while ($i < len($rows)) { if ($rows[$i]) { $t = $i; $i = len($rows); } else { $i = $i + 1; } }
        if ($t < 0) { return null; }
        $b = -1;
        $i = len($rows) - 1;
        while ($i >= 0) { if ($rows[$i]) { $b = $i; $i = -1; } else { $i = $i - 1; } }
        $l = -1;
        $i = 0;
        while ($i < len($cols)) { if ($cols[$i]) { $l = $i; $i = len($cols); } else { $i = $i + 1; } }
        $r = -1;
        $i = len($cols) - 1;
        while ($i >= 0) { if ($cols[$i]) { $r = $i; $i = -1; } else { $i = $i - 1; } }
        return [$l, $t, $r + 1, $b + 1];
    }

    // getcolors($maxcolors) -> [[count, colour], ...] or null when there are
    // more than maxcolors distinct colours, which is Pillow's contract.
    def getcolors($maxcolors) {
        $limit = 256;
        if ($maxcolors != null) { $limit = floor($maxcolors); }
        $n = mode.bands(this.mode);
        if ($n == 1) {
            $h = nd_to_list(nd_bincount(nd_ravel(nd_clip(this.px, 0, 255))));
            $out = [];
            $i = 0;
            while ($i < len($h)) {
                if ($h[$i] > 0) {
                    if (len($out) >= $limit) { return null; }
                    push($out, [$h[$i], $i]);
                }
                $i = $i + 1;
            }
            return $out;
        }
        // Multi-band: pack each pixel into one integer and count that. One
        // array pass per band plus one bincount -- still no pixel loop.
        $key = nd_zeros([this.height, this.width, 1], "i64");
        $i = 0;
        while ($i < $n) {
            $key = nd_add(nd_multiply($key, 256), nd_clip(pxbuf.band(this.px, $i), 0, 255));
            $i = $i + 1;
        }
        $flat = nd_ravel($key);
        $uniq = nd_to_list(nd_unique($flat));
        if (len($uniq) > $limit) { return null; }
        $out = [];
        $k = 0;
        while ($k < len($uniq)) {
            $v = $uniq[$k];
            $count = pxbuf.scalar(nd_sum(nd_astype(nd_equal($flat, $v), "i64")));
            $cols = [];
            $b = $n - 1;
            $rem = $v;
            while ($b >= 0) {
                $cols[$b] = $rem % 256;
                $rem = floor($rem / 256);
                $b = $b - 1;
            }
            push($out, [$count, $cols]);
            $k = $k + 1;
        }
        return $out;
    }

    // ─── conversion ─────────────────────────────────────────────────────
    def convert($m, $dither, $matrix) {
        if ($m == null) { return this.copy(); }
        if ($matrix != null) {
            throw "Image.convert: the matrix argument is not implemented yet (phase P6)";
        }
        if ($m == this.mode) { return this.copy(); }
        $r = convert.px(this.px, this.mode, $m, this.palette, $dither);
        $px = $r.px;
        if (nd_is_view($px)) { $px = pxbuf.materialise($px, mode.dtype($m)); }
        return _wrap($m, $px, $r.palette, _copyInfo(this.info), null);
    }

    // ─── geometry: resize ───────────────────────────────────────────────
    //
    // resize($size, $filter, $box, $reducing_gap)
    //
    // Three things Pillow's PYTHON layer does on top of the C, all of which
    // change the answer and none of which belong in resample.b:
    //
    //   1. modes "1" and "P" force NEAREST, whatever filter was asked for.
    //      Interpolating a palette INDEX would blend index 3 with index 200
    //      and produce whatever colour sits at 101; interpolating a bilevel
    //      image would produce values that mode "1" cannot hold.
    //   2. "LA" and "RGBA" are PREMULTIPLIED first -- converted to "La" and
    //      "RGBa", resized, and converted back. Without it, a fully
    //      transparent pixel's colour (often black, or garbage) is averaged
    //      into its visible neighbours and leaves a dark fringe.
    //   3. mode "I" rounds its accumulator instead of shifting a fixed-point
    //      one, so it takes its own entry point.
    def resize($size, $filter, $box, $reducing_gap) {
        if (type($size) != "list" || len($size) != 2) {
            throw "Image.resize: size must be [width, height]";
        }
        $w = floor($size[0]);
        $h = floor($size[1]);
        if ($w < 1 || $h < 1) {
            throw "Image.resize: width and height must be at least 1 (got " +
                  str($w) + "x" + str($h) + ")";
        }
        _checkPixels($w, $h, "Image.resize");

        $f = $filter;
        if ($f == null) { $f = resample.BICUBIC; }
        if (resample.filter_name($f) == null) {
            throw "Image.resize: unknown resampling filter " + str($f) +
                  ". Use NEAREST, LANCZOS, BILINEAR, BICUBIC, BOX or HAMMING (0..5)";
        }
        if ($reducing_gap != null) {
            throw "Image.resize: reducing_gap is not implemented yet (it needs reduce())";
        }

        // 1. a palette or bilevel image can only be sampled, never averaged
        if (this.mode == "1" || this.mode == "P") { $f = resample.NEAREST; }

        // 2. premultiply, resize, unpremultiply
        if ((this.mode == "LA" || this.mode == "RGBA") && $f != resample.NEAREST) {
            $premul = "La";
            if (this.mode == "RGBA") { $premul = "RGBa"; }
            $tmp = this.convert($premul, null, null);
            $r = $tmp.resize($size, $f, $box, null);
            return $r.convert(this.mode, null, null);
        }

        if (mode.is_float(this.mode)) {
            $out = resample.resize(this.px, $w, $h, $f, $box);
        } else {
            if (this.mode == "I") {
                $out = resample.resize_i32(this.px, $w, $h, $f, $box);
            } else {
                $out = resample.resize(this.px, $w, $h, $f, $box);
            }
        }
        return _wrap(this.mode, $out, _copyPalette(this.palette),
                     _copyInfo(this.info), null);
    }

    // ─── geometry: transpose, rotate, reduce, thumbnail, transform ──────

    // transpose($op) -> one of the eight flips and right-angle rotations.
    // Every one is an axis permutation or a flip, so all eight are exact and
    // effectively free -- no resampling is involved at any point.
    def transpose($op) {
        $out = xform.transpose(this.px, $op);
        return _wrap(this.mode, $out, _copyPalette(this.palette),
                     _copyInfo(this.info), null);
    }

    // reduce($factor, $box) -> integer box averaging.
    //
    // $factor is a number or [xfactor, yfactor]. The output size rounds UP,
    // and a partial block at the right or bottom edge is averaged over its
    // own smaller area rather than over zeros.
    //
    // LA and RGBA are premultiplied first, for the same reason resize does
    // it: averaging a transparent pixel's colour into its visible neighbours
    // leaves a fringe.
    def reduce($factor, $box) {
        if (type($factor) == "list") {
            if (len($factor) != 2) { throw "Image.reduce: factor must be a number or [x, y]"; }
            $fx = floor($factor[0]);
            $fy = floor($factor[1]);
        } else {
            $fx = floor($factor);
            $fy = $fx;
        }
        if ($fx < 1 || $fy < 1) {
            throw "Image.reduce: the factor must be at least 1 (got " +
                  str($fx) + "x" + str($fy) + ")";
        }
        if ($fx == 1 && $fy == 1) {
            if ($box == null) { return this.copy(); }
            return this.crop($box);
        }
        // ImagingReduce refuses these outright: a palette index and a
        // bilevel value cannot be averaged, and the I;16 path is absent
        // upstream. Raising is better than returning a blend of indices.
        if (this.mode == "P" || this.mode == "1" ||
            this.mode == "I;16" || this.mode == "I;16L" ||
            this.mode == "I;16B" || this.mode == "I;16N") {
            throw "Image.reduce: mode " + this.mode + " cannot be reduced " +
                  "(averaging an index or a bilevel value is meaningless). " +
                  "Convert to L or RGB first.";
        }
        if (this.mode == "LA" || this.mode == "RGBA") {
            $premul = "La";
            if (this.mode == "RGBA") { $premul = "RGBa"; }
            $tmp = this.convert($premul, null, null);
            return $tmp.reduce($factor, $box).convert(this.mode, null, null);
        }
        $kind = "u8";
        if (this.mode == "I") { $kind = "i32"; }
        if (mode.is_float(this.mode)) { $kind = "f32"; }
        $out = xform.reduce(this.px, $fx, $fy, $box, $kind);
        return _wrap(this.mode, $out, _copyPalette(this.palette),
                     _copyInfo(this.info), null);
    }

    // transform($size, $method, $data, $filter, $fill) -> a general remap.
    //
    // $method is AFFINE, EXTENT, PERSPECTIVE, QUAD or MESH, and $data is
    // that method's coefficients. A source coordinate outside the image
    // takes $fill (a colour or null for zero) rather than a clamped edge
    // pixel, which is what stops a rotated image growing a smeared border.
    def transform($size, $method, $data, $filter, $fill) {
        if (type($size) != "list" || len($size) != 2) {
            throw "Image.transform: size must be [width, height]";
        }
        $w = floor($size[0]);
        $h = floor($size[1]);
        if ($w < 1 || $h < 1) {
            throw "Image.transform: width and height must be at least 1";
        }
        _checkPixels($w, $h, "Image.transform");
        $f = $filter;
        if ($f == null) { $f = resample.NEAREST; }
        if ($f != resample.NEAREST && $f != resample.BILINEAR && $f != resample.BICUBIC) {
            throw "Image.transform: " + str(resample.filter_name($f)) +
                  " cannot be used. Use NEAREST, BILINEAR or BICUBIC.";
        }
        // LA and RGBA are PREMULTIPLIED for any interpolating filter, the
        // same dance resize does and for the same reason: averaging a
        // transparent pixel's colour into its visible neighbours leaves a
        // fringe. Pillow's Python layer does this before reaching the C.
        if ((this.mode == "LA" || this.mode == "RGBA") && $f != resample.NEAREST) {
            $premul = "La";
            if (this.mode == "RGBA") { $premul = "RGBa"; }
            $tmp = this.convert($premul, null, null);
            $r = $tmp.transform($size, $method, $data, $f, $fill);
            return $r.convert(this.mode, null, null);
        }
        // A palette index or a bilevel value cannot be interpolated, so the
        // filter degrades to NEAREST exactly as resize does.
        if (this.mode == "1" || this.mode == "P") { $f = resample.NEAREST; }
        $fillList = null;
        if ($fill != null) { $fillList = _colourList($fill, this.mode); }
        $out = xform.apply(this.px, $w, $h, $method, $data, $f, $fillList);
        return _wrap(this.mode, $out, _copyPalette(this.palette),
                     _copyInfo(this.info), null);
    }

    // rotate($angle, $filter, $expand, $center, $translate, $fill)
    //
    // Counter-clockwise, in degrees. 0, 90, 180 and 270 take the exact
    // transpose path whenever there is no custom centre or translation --
    // and for 90/270 only when the result fits, i.e. expand is asked for or
    // the image is square. Everything else is an affine transform.
    def rotate($angle, $filter, $expand, $center, $translate, $fill) {
        $a = $angle % 360.0;
        if ($a < 0) { $a = $a + 360.0; }
        $exp = false;
        if ($expand != null) { $exp = $expand; }

        if ($center == null && $translate == null) {
            if ($a == 0) { return this.copy(); }
            if ($a == 180) { return this.transpose(xform.ROTATE_180); }
            if (($a == 90 || $a == 270) && ($exp || this.width == this.height)) {
                if ($a == 90) { return this.transpose(xform.ROTATE_90); }
                return this.transpose(xform.ROTATE_270);
            }
        }

        $w = this.width;
        $h = this.height;
        $tx = 0;
        $ty = 0;
        if ($translate != null) { $tx = $translate[0]; $ty = $translate[1]; }
        $cx = $w / 2.0;
        $cy = $h / 2.0;
        if ($center != null) { $cx = $center[0]; $cy = $center[1]; }

        // The matrix is the REVERSE transform -- destination to source --
        // because each discrete output pixel is interpolated from the area
        // around its (floating) source position. Pillow rounds the four
        // trigonometric entries to 15 decimal places so that a right angle
        // comes out exactly 0 and 1 rather than 6e-17.
        $rad = 0.0 - ($a * PI / 180.0);
        $ca = _round15(cos($rad));
        $sa = _round15(sin($rad));
        $m = [$ca, $sa, 0.0, 0.0 - $sa, $ca, 0.0];

        $p = _affinePoint(0.0 - $cx - $tx, 0.0 - $cy - $ty, $m);
        $m[2] = $p[0] + $cx;
        $m[5] = $p[1] + $cy;

        if ($exp) {
            $corners = [[0, 0], [$w, 0], [$w, $h], [0, $h]];
            $xs = [];
            $ys = [];
            $i = 0;
            while ($i < 4) {
                $q = _affinePoint($corners[$i][0], $corners[$i][1], $m);
                push($xs, $q[0]);
                push($ys, $q[1]);
                $i = $i + 1;
            }
            $nw = ceil(max($xs)) - floor(min($xs));
            $nh = ceil(max($ys)) - floor(min($ys));
            $p2 = _affinePoint(0.0 - ($nw - $w) / 2.0, 0.0 - ($nh - $h) / 2.0, $m);
            $m[2] = $p2[0];
            $m[5] = $p2[1];
            $w = $nw;
            $h = $nh;
        }

        $f = $filter;
        if ($f == null) { $f = resample.NEAREST; }
        return this.transform([$w, $h], xform.AFFINE, $m, $f, $fill);
    }

    // thumbnail($size, $filter, $reducing_gap) -> shrink in place, keeping
    // the aspect ratio. Returns null, as Pillow's does.
    //
    // It only ever shrinks: if the image already fits inside $size, nothing
    // happens. The preserved size is computed with Pillow's round_aspect,
    // which picks whichever of floor and ceil lands closer to the true
    // aspect -- not a plain round, which can be a pixel out.
    def thumbnail($size, $filter, $reducing_gap) {
        if (type($size) != "list" || len($size) != 2) {
            throw "Image.thumbnail: size must be [width, height]";
        }
        $x = floor($size[0]);
        $y = floor($size[1]);
        if ($x < 1 || $y < 1) { throw "Image.thumbnail: size must be at least 1x1"; }
        if ($x >= this.width && $y >= this.height) { return null; }

        $aspect = this.width / this.height;
        if ($x / $y >= $aspect) {
            $x = _roundAspectW($y * $aspect, $y, $aspect);
        } else {
            $y = _roundAspectH($x / $aspect, $x, $aspect);
        }
        if ($x < 1) { $x = 1; }
        if ($y < 1) { $y = 1; }

        $f = $filter;
        if ($f == null) { $f = resample.BICUBIC; }
        if (this.width == $x && this.height == $y) { return null; }

        $r = this.resize([$x, $y], $f, null, null);
        this.px = $r.px;
        this.size = [$x, $y];
        this.width = $x;
        this.height = $y;
        this.mode = $r.mode;
        return null;
    }

    // ─── bytes ──────────────────────────────────────────────────────────
    //
    // Mode "1" is PACKED: one bit per pixel, most significant bit first, with
    // each row padded to a whole byte. A 13-wide bilevel image is two bytes
    // per row, not thirteen. Every other mode is its wire element, one value
    // per band per pixel.
    def tobytes() {
        if (this.mode == "1") { return _packBilevel(this.px); }
        return nd_tobytes(this.px, mode.wire_elem(this.mode));
    }

    // ─── output ─────────────────────────────────────────────────────────
    def save($path, $format) {
        $id = $format;
        if ($id == null) { $id = registry.format_for_path($path); }
        if ($id == null) {
            throw "Image.save: cannot tell the format from '" + str($path) +
                  "'. Pass it explicitly.";
        }
        $spec = {"mode": this.mode, "size": this.size, "px": this.px,
                 "palette": this.palette, "info": this.info};
        $bytes = registry.encode($spec, $id);
        writefile($path, $bytes);
        return null;
    }

    // show() writes the file and prints its path. Bantu has no GUI binding
    // and no shell builtin, so there is nothing to hand a file to --
    // bplot.show() makes the same accommodation for the same reason.
    def show($title) {
        $p = "/tmp/pillow-show.ppm";
        this.save($p, "PPM");
        print("pillow: wrote " + $p + " (" + str(this.width) + "x" +
              str(this.height) + " " + this.mode + ")");
        return $p;
    }
}


// ════════════════════════════════════════════════════════════════════════
//  Factories and helpers
// ════════════════════════════════════════════════════════════════════════

def _wrap($m, $px, $palette, $info, $format) {
    $d = pxbuf.dims($px);
    return new PillowImage($m, $d[1], $d[0], $px, $palette, $info, $format);
}

def _copyInfo($info) {
    $out = {};
    if ($info == null) { return $out; }
    for $k, $v in $info.items() { $out[$k] = $v; }
    return $out;
}

def _copyPalette($p) {
    if ($p == null) { return null; }
    return nd_copy($p);
}

// _colourList($colour, $m) -> a list with one value per band of mode m.
// Accepts a colour string, a number, or a list.
def _colourList($colour, $m) {
    $nb = mode.bands($m);
    if ($colour == null) { return mode.default_colour($m); }
    if (type($colour) == "string") {
        $c = color.getcolor($colour, $m);
        if (type($c) != "list") { $c = [$c]; }
        // getcolor switches on basemode, so CMYK comes back with three
        // values; pad to the band count with zero, and keep alpha last.
        while (len($c) < $nb) { push($c, 0); }
        if (len($c) > $nb) {
            $t = [];
            $i = 0;
            while ($i < $nb) { push($t, $c[$i]); $i = $i + 1; }
            $c = $t;
        }
        return $c;
    }
    if (type($colour) == "list") {
        if (len($colour) == $nb) { return $colour; }
        if (len($colour) == 1) {
            $out = [];
            $i = 0;
            while ($i < $nb) { push($out, $colour[0]); $i = $i + 1; }
            return $out;
        }
        if (len($colour) < $nb) {
            $out = [];
            $i = 0;
            while ($i < len($colour)) { push($out, $colour[$i]); $i = $i + 1; }
            while (len($out) < $nb) { push($out, 255); }
            return $out;
        }
        $out = [];
        $i = 0;
        while ($i < $nb) { push($out, $colour[$i]); $i = $i + 1; }
        return $out;
    }
    // a single number: every band gets it
    $out = [];
    $i = 0;
    while ($i < $nb) { push($out, $colour); $i = $i + 1; }
    return $out;
}

// new_image($mode, $size, $colour) -> Pillow's Image.new.
//
// NAMED new_image, NOT new, because `new` is Bantu's instantiation keyword
// and `def new(...)` does not parse. Pillow's spelling is restored at the
// bottom of this file by binding an anonymous function to $new: a MEMBER
// called `new` parses fine, so callers still write pillow.new(...). The same
// trick restores `open` (a builtin) and `eval`.
//
// The default fill is ZERO in every band, alpha included, so a new RGBA image
// is transparent black -- verified against Pillow, where guessing 255 for
// alpha is the obvious mistake.
def new_image($m, $size, $colour) {
    if (!mode.is_valid($m)) {
        throw "Image.new: unknown mode " + str($m);
    }
    if (type($size) != "list" || len($size) != 2) {
        throw "Image.new: size must be [width, height]";
    }
    $w = floor($size[0]);
    $h = floor($size[1]);
    if ($w < 0 || $h < 0) {
        throw "Image.new: negative size " + str($w) + "x" + str($h);
    }
    _checkPixels($w, $h, "Image.new");
    $px = pxbuf.make($h, $w, mode.bands($m), mode.dtype($m));
    if ($colour != null) { pxbuf.fill($px, _colourList($colour, $m)); }
    return _wrap($m, $px, null, {}, null);
}

// _checkPixels -> the decompression-bomb guard, before any allocation.
def _checkPixels($w, $h, $who) {
    if ($w == 0 || $h == 0) { return true; }
    if ($w > 2147483647 / max($h, 1)) {
        throw $who + ": " + str($w) + "x" + str($h) + " overflows";
    }
    $n = $w * $h;
    if ($n > $CFG.MAX_IMAGE_PIXELS) {
        throw $who + ": " + str($w) + "x" + str($h) + " is " + str($n) +
              " pixels, over the MAX_IMAGE_PIXELS limit of " +
              str($CFG.MAX_IMAGE_PIXELS) + ". Raise it deliberately if you trust the source.";
    }
    return true;
}

// open_image($path) -> Pillow's Image.open. Reads, sniffs, decodes.
// Named open_image because `open` is a Bantu builtin; see new_image above.
def open_image($path) {
    if (type($path) != "string") { throw "Image.open: expected a file path"; }
    if (!file_exists($path)) { throw "Image.open: no such file: " + $path; }
    $data = readfile($path);
    return _fromSpec(registry.decode($data, $path), $path);
}

// frombytes_file($data, $path) -> decode bytes already in memory.
def open_bytes($data, $hintPath) {
    return _fromSpec(registry.decode($data, $hintPath), null);
}

def _fromSpec($spec, $path) {
    $w = $spec.size[0];
    $h = $spec.size[1];
    _checkPixels($w, $h, "Image.open");
    $info = $spec.info;
    if ($info == null) { $info = {}; }
    $fmt = $spec.format;
    return _wrap($spec.mode, $spec.px, $spec.palette, $info, $fmt);
}

// _packBilevel($px) -> mode "1" as one bit per pixel, MSB first, rows padded
// to a byte boundary.
//
// Vectorised: the row is padded to a multiple of 8, the eight bit planes are
// eight strided slices, and each is shifted into place and summed. The loop
// is over BITS (8), never over pixels.
def _packBilevel($px) {
    $d = pxbuf.dims($px);
    $h = $d[0];
    $w = $d[1];
    if ($h == 0 || $w == 0) { return ""; }
    $rowBytes = floor(($w + 7) / 8);
    $padded = nd_zeros([$h, $rowBytes * 8, 1], "i64");
    // a stored 255 becomes a set bit
    $bits = nd_floor_divide(nd_clip($px, 0, 255), 255);
    nd_add($bits, 0, nd_slice($padded, [null, [0, $w], null]));
    $acc = nd_zeros([$h, $rowBytes], "i64");
    $bit = 0;
    while ($bit < 8) {
        $plane = nd_reshape(nd_slice($padded, [null, [$bit, $rowBytes * 8, 8], null]),
                            [$h, $rowBytes]);
        $acc = nd_add($acc, nd_multiply($plane, pow(2, 7 - $bit)));
        $bit = $bit + 1;
    }
    return nd_tobytes($acc, "u8");
}

// _unpackBilevel($data, $w, $h) -> the inverse.
def _unpackBilevel($data, $w, $h) {
    if ($h == 0 || $w == 0) { return pxbuf.make($h, $w, 1, "i64"); }
    $rowBytes = floor(($w + 7) / 8);
    $need = $rowBytes * $h;
    if (len($data) < $need) {
        throw "Image.frombytes: a " + str($w) + "x" + str($h) +
              " bilevel image needs " + str($need) + " packed bytes, got " + str(len($data));
    }
    $raw = nd_frombytes(substr($data, 0, $need), "u8", [$h, $rowBytes]);
    $out = nd_zeros([$h, $rowBytes * 8, 1], "i64");
    $bit = 0;
    while ($bit < 8) {
        $onoff = nd_mod(nd_floor_divide($raw, pow(2, 7 - $bit)), 2);
        nd_add(nd_reshape(nd_multiply($onoff, 255), [$h, $rowBytes, 1]), 0,
               nd_slice($out, [null, [$bit, $rowBytes * 8, 8], null]));
        $bit = $bit + 1;
    }
    if ($rowBytes * 8 == $w) { return $out; }
    return pxbuf.materialise(nd_slice($out, [null, [0, $w], null]), "i64");
}

// frombytes($mode, $size, $data) -> Image.frombytes
def frombytes($m, $size, $data) {
    if (!mode.is_valid($m)) { throw "Image.frombytes: unknown mode " + str($m); }
    $w = floor($size[0]);
    $h = floor($size[1]);
    _checkPixels($w, $h, "Image.frombytes");
    if ($m == "1") { return _wrap($m, _unpackBilevel($data, $w, $h), null, {}, null); }
    $nb = mode.bands($m);
    $elem = mode.wire_elem($m);
    $px = nd_frombytes($data, $elem, [$h, $w, $nb]);
    return _wrap($m, $px, null, {}, null);
}

// fromarray($array, $mode) -> an image over a numba array.
//
// A 2-D array becomes a single-band image; a 3-D array's last axis is bands.
// The data is COPIED, so later writes to the array do not reach the image.
def fromarray($a, $m) {
    $s = nd_shape($a);
    if (len($s) == 2) {
        $h = $s[0];
        $w = $s[1];
        $nb = 1;
        $src = nd_reshape(nd_ascontiguous($a), [$h, $w, 1]);
    } else {
        if (len($s) != 3) {
            throw "Image.fromarray: expected a 2- or 3-dimensional array, got " + str($s);
        }
        $h = $s[0];
        $w = $s[1];
        $nb = $s[2];
        $src = $a;
    }
    $guess = $m;
    if ($guess == null) {
        if (nd_dtype($a) == "f64") { $guess = "F"; }
        else {
            if ($nb == 1) { $guess = "L"; }
            else {
                if ($nb == 2) { $guess = "LA"; }
                else {
                    if ($nb == 3) { $guess = "RGB"; }
                    else {
                        if ($nb == 4) { $guess = "RGBA"; }
                        else { throw "Image.fromarray: cannot guess a mode for " + str($nb) + " bands"; }
                    }
                }
            }
        }
    }
    if (mode.bands($guess) != $nb) {
        throw "Image.fromarray: mode " + $guess + " needs " + str(mode.bands($guess)) +
              " bands, the array has " + str($nb);
    }
    _checkPixels($w, $h, "Image.fromarray");
    return _wrap($guess, pxbuf.materialise($src, mode.dtype($guess)), null, {}, null);
}

// merge($mode, $bands) -> Image.merge
def merge($m, $bands) {
    if (!mode.is_valid($m)) { throw "Image.merge: unknown mode " + str($m); }
    $n = mode.bands($m);
    if (type($bands) != "list" || len($bands) != $n) {
        throw "Image.merge: mode " + $m + " needs " + str($n) + " bands, got " +
              str(len($bands));
    }
    $w = $bands[0].width;
    $h = $bands[0].height;
    $parts = [];
    $i = 0;
    while ($i < $n) {
        if ($bands[$i].width != $w || $bands[$i].height != $h) {
            throw "Image.merge: band " + str($i) + " is " + str($bands[$i].width) + "x" +
                  str($bands[$i].height) + ", band 0 is " + str($w) + "x" + str($h);
        }
        if (mode.bands($bands[$i].mode) != 1) {
            throw "Image.merge: band " + str($i) + " must be a single-band image, got " +
                  $bands[$i].mode;
        }
        push($parts, $bands[$i].px);
        $i = $i + 1;
    }
    return _wrap($m, pxbuf.stack($parts, mode.dtype($m)), null, {}, null);
}

// blend($im1, $im2, $alpha) -> im1 * (1 - alpha) + im2 * alpha
def blend($im1, $im2, $alpha) {
    _sameShape($im1, $im2, "Image.blend");
    if ($im1.mode != $im2.mode) {
        throw "Image.blend: modes differ (" + $im1.mode + " and " + $im2.mode + ")";
    }
    $a = $alpha;
    $f1 = nd_astype($im1.px, "f64");
    $f2 = nd_astype($im2.px, "f64");
    $out = nd_add(nd_multiply($f1, 1.0 - $a), nd_multiply($f2, $a));
    if (mode.is_float($im1.mode)) {
        return _wrap($im1.mode, pxbuf.to_f32($out), null, {}, null);
    }
    $r = nd_astype(nd_floor(nd_add($out, 0.5)), "i64");
    return _wrap($im1.mode, pxbuf.clamp($r, $im1.mode), null, {}, null);
}

// composite($im1, $im2, $mask) -> im2 where mask is 0, im1 where mask is 255
def composite($im1, $im2, $mask) {
    _sameShape($im1, $im2, "Image.composite");
    $out = $im2.copy();
    $out.paste($im1, null, $mask);
    return $out;
}

// alpha_composite($dst, $src) -> src laid over dst, both RGBA.
//
// The standard Porter-Duff "over", in integers:
//   aOut = aSrc + aDst * (255 - aSrc) / 255
//   cOut = (cSrc * aSrc + cDst * aDst * (255 - aSrc) / 255) / aOut
def alpha_composite($dst, $src) {
    if ($dst.mode != "RGBA" || $src.mode != "RGBA") {
        throw "Image.alpha_composite: both images must be RGBA (got " +
              $dst.mode + " and " + $src.mode + ")";
    }
    _sameShape($dst, $src, "Image.alpha_composite");
    $as = nd_astype(pxbuf.band($src.px, 3), "f64");
    $ad = nd_astype(pxbuf.band($dst.px, 3), "f64");
    $inv = nd_divide(nd_multiply($ad, nd_subtract(255.0, $as)), 255.0);
    $ao = nd_add($as, $inv);
    $safe = nd_maximum($ao, 1.0);
    $parts = [];
    $i = 0;
    while ($i < 3) {
        $cs = nd_astype(pxbuf.band($src.px, $i), "f64");
        $cd = nd_astype(pxbuf.band($dst.px, $i), "f64");
        $num = nd_add(nd_multiply($cs, $as), nd_multiply($cd, $inv));
        $c = nd_divide($num, $safe);
        $c = nd_where(nd_less_equal($ao, 0), 0, $c);
        push($parts, nd_astype(nd_clip(nd_floor(nd_add($c, 0.5)), 0, 255), "i64"));
        $i = $i + 1;
    }
    push($parts, nd_astype(nd_clip(nd_floor(nd_add($ao, 0.5)), 0, 255), "i64"));
    return _wrap("RGBA", pxbuf.stack($parts, "i64"), null, {}, null);
}

// eval_image($im, $fn) -> Pillow's Image.eval: fn applied to every value
// through a lookup table, so fn is called at most 256 times, never per pixel.
def eval_image($im, $fn) {
    return $im.point($fn, null);
}

// ─── Pillow's spellings, restored ───────────────────────────────────────
//
// `new`, `open` and `eval` cannot be function NAMES here -- the first is a
// keyword, the second a builtin -- but they can be module variables holding
// a function, and a member access spells them exactly as Pillow does. So
// pillow.new(...), pillow.open(...) and pillow.eval(...) all work.
$new = def($m, $size, $colour) { return new_image($m, $size, $colour); };
$open = def($path) { return open_image($path); };
$eval = def($im, $fn) { return eval_image($im, $fn); };

// _round15($v) -> rounded to 15 decimal places, as Pillow's rotate does to
// its trigonometric entries, so cos(90 degrees) is exactly 0 rather than
// 6.1e-17 and a right angle lands on whole pixels.
def _round15($v) {
    $s = 1000000000000000.0;
    if ($v < 0) { return 0.0 - floor(0.5 - $v * $s) / $s; }
    return floor(0.5 + $v * $s) / $s;
}

// _affinePoint($x, $y, $m) -> the 6-element affine applied to one point.
def _affinePoint($x, $y, $m) {
    return [$m[0] * $x + $m[1] * $y + $m[2], $m[3] * $x + $m[4] * $y + $m[5]];
}

// Pillow's round_aspect: of floor(n) and ceil(n), take whichever leaves the
// aspect ratio closer to the original, never below 1. A plain round can be a
// pixel out, and for a thumbnail that is a visible stretch.
def _roundAspectW($n, $y, $aspect) {
    $lo = floor($n);
    $hi = ceil($n);
    if ($lo < 1) { $lo = 1; }
    if ($hi < 1) { $hi = 1; }
    $dlo = $aspect - $lo / $y;
    if ($dlo < 0) { $dlo = 0.0 - $dlo; }
    $dhi = $aspect - $hi / $y;
    if ($dhi < 0) { $dhi = 0.0 - $dhi; }
    if ($dlo <= $dhi) { return $lo; }
    return $hi;
}

def _roundAspectH($n, $x, $aspect) {
    $lo = floor($n);
    $hi = ceil($n);
    if ($lo < 1) { $lo = 1; }
    if ($hi < 1) { $hi = 1; }
    // Pillow's key is 0 when n == 0, which cannot arise once clamped to 1.
    $dlo = $aspect - $x / $lo;
    if ($dlo < 0) { $dlo = 0.0 - $dlo; }
    $dhi = $aspect - $x / $hi;
    if ($dhi < 0) { $dhi = 0.0 - $dhi; }
    if ($dlo <= $dhi) { return $lo; }
    return $hi;
}

def _sameShape($a, $b, $who) {
    if ($a.width != $b.width || $a.height != $b.height) {
        throw $who + ": sizes differ (" + str($a.width) + "x" + str($a.height) +
              " and " + str($b.width) + "x" + str($b.height) + ")";
    }
    return true;
}
