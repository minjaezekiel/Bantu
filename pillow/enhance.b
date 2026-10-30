// ════════════════════════════════════════════════════════════════════════
//  pillow/enhance.b — ImageEnhance: four enhancements, all of them the
//  same two lines.
//
//  Each class builds a DEGENERATE image -- the zero-enhancement extreme --
//  and `enhance(factor)` is `Image.blend(degenerate, image, factor)`.
//  Extrapolation past 1.0 falls out of blend's own arithmetic, which is
//  why a factor above 1 sharpens rather than being clamped.
//
//      Color       the image converted to L (or LA) and back
//      Contrast    a flat image of the mean grey level
//      Brightness  black
//      Sharpness   the image through ImageFilter.SMOOTH
//
//  Each degenerate image gets the ORIGINAL's alpha band put back, so an
//  RGBA image keeps its transparency at every factor.
//
//      $e = enhance.Contrast($im);
//      $out = enhance.enhance($e, 1.6);
//
//  docs/pillow-architecture.md §7.
// ════════════════════════════════════════════════════════════════════════

include "./mode.b" as mode;
include "./image.b" as image;
include "./filter.b" as flt;
include "./stat.b" as stat;

def _withAlpha($deg, $im) {
    $bands = mode.band_names($im.mode);
    $i = 0;
    while ($i < len($bands)) {
        if ($bands[$i] == "A") { $deg.putalpha($im.getchannel("A")); return $deg; }
        $i = $i + 1;
    }
    return $deg;
}

// Color($im) -> the colour-balance enhancement.
def Color($im) {
    $inter = "L";
    $bands = mode.band_names($im.mode);
    $i = 0;
    while ($i < len($bands)) {
        if ($bands[$i] == "A") { $inter = "LA"; }
        $i = $i + 1;
    }
    $deg = $im;
    if ($inter != $im.mode) {
        $deg = $im.convert($inter, null, null).convert($im.mode, null, null);
    }
    return {"name": "Color", "image": $im, "degenerate": $deg};
}

// Contrast($im) -> the contrast enhancement. The degenerate image is a
// flat grey at the MEAN of the L-converted image, rounded half up, and the
// mean comes from the 256-bin histogram (ImageStat) rather than from the
// pixels -- so for I and F images it is a mean of bin indices, as Pillow's is.
def Contrast($im) {
    $grey = $im;
    if ($im.mode != "L") { $grey = $im.convert("L", null, null); }
    $st = stat.Stat($grey, null);
    $mean = floor($st.mean[0] + 0.5);
    $deg = image.new_image("L", $im.size, $mean);
    if ($deg.mode != $im.mode) { $deg = $deg.convert($im.mode, null, null); }
    return {"name": "Contrast", "image": $im, "degenerate": _withAlpha($deg, $im)};
}

// Brightness($im) -> the brightness enhancement; the degenerate image is
// black, which for an RGBA image means transparent black until the alpha
// band is put back.
def Brightness($im) {
    $deg = image.new_image($im.mode, $im.size, 0);
    return {"name": "Brightness", "image": $im, "degenerate": _withAlpha($deg, $im)};
}

// Sharpness($im) -> the sharpness enhancement; the degenerate image is the
// image smoothed, so a factor above 1 extrapolates away from the blur.
def Sharpness($im) {
    $deg = image.wrap($im.mode, flt.apply($im.px, $im.mode, flt.SMOOTH),
                      null, {}, null);
    return {"name": "Sharpness", "image": $im, "degenerate": _withAlpha($deg, $im)};
}

// enhance($e, $factor) -> the enhanced image. 1.0 is the original.
def enhance($e, $factor) {
    return image.blend($e.degenerate, $e.image, $factor);
}
