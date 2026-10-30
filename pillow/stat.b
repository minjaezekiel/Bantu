// ════════════════════════════════════════════════════════════════════════
//  pillow/stat.b — ImageStat: the histogram, and nothing but.
//
//  Every property is derived from histogram() with 256 bins per band --
//  INCLUDING extrema, which therefore reports bin indices and so cannot
//  exceed 255 even for an I or F image. That is a documented Pillow wart
//  (its own docstring says to use getextrema() instead), so it is
//  reproduced rather than quietly improved: a caller who ports code from
//  Pillow gets Pillow's numbers.
//
//      $st = stat.Stat($im, null);
//      $st.mean          // one value per band
//      $st.extrema $st.count $st.sum $st.sum2 $st.median $st.rms
//      $st.var $st.stddev
//
//  `var` is the computational form -- (sum2 - sum^2/count)/count -- with
//  its cancellation, not the numerically better two-pass form, because
//  that is the number Pillow prints.
//
//  docs/pillow-architecture.md §7.4.
// ════════════════════════════════════════════════════════════════════════

// Stat($imageOrHistogram, $mask) -> a dict of every statistic.
//
// Pillow's Stat is a class with cached properties; here it is a dict,
// computed once, because the cost is 256 numbers per band either way.
def Stat($src, $mask) {
    if (type($src) == "list") { $h = $src; }
    else if (type($src) == "instance") { $h = $src.histogram($mask); }
    else { throw "ImageStat.Stat: first argument must be an image or a histogram"; }
    if (len($h) % 256 != 0 || len($h) == 0) {
        throw "ImageStat.Stat: a histogram is 256 bins per band, got " + str(len($h));
    }
    $nb = floor(len($h) / 256);

    $extrema = [];
    $count = [];
    $sum = [];
    $sum2 = [];
    $mean = [];
    $median = [];
    $rms = [];
    $var = [];
    $stddev = [];

    $b = 0;
    while ($b < $nb) {
        $base = $b * 256;
        // extrema: the lowest and highest non-empty BIN. Pillow's minmax
        // starts at (255, 0), so an all-empty band reports (255, 0).
        $lo = 255;
        $hi = 0;
        $j = 0;
        while ($j < 256) {
            if ($h[$base + $j] != 0) { $lo = $j; break; }
            $j = $j + 1;
        }
        $j = 255;
        while ($j >= 0) {
            if ($h[$base + $j] != 0) { $hi = $j; break; }
            $j = $j - 1;
        }
        push($extrema, [$lo, $hi]);

        $c = 0;
        $s = 0;
        $s2 = 0;
        $j = 0;
        while ($j < 256) {
            $n = $h[$base + $j];
            $c = $c + $n;
            $s = $s + $j * $n;
            $s2 = $s2 + $j * $j * $n;
            $j = $j + 1;
        }
        push($count, $c);
        push($sum, $s);
        push($sum2, $s2);

        if ($c == 0) {
            push($mean, 0);
            push($rms, 0);
            push($var, 0);
            push($stddev, 0);
            push($median, 0);
        } else {
            push($mean, $s / $c);
            push($rms, sqrt($s2 / $c));
            $v = ($s2 - ($s * $s) / $c) / $c;
            push($var, $v);
            if ($v < 0) { push($stddev, 0); } else { push($stddev, sqrt($v)); }
            // median: the first bin at which the running count EXCEEDS half.
            // Pillow leaves `j` at 255 if that never happens, because the
            // loop variable survives the for.
            $half = floor($c / 2);
            $run = 0;
            $at = 255;
            $j = 0;
            while ($j < 256) {
                $run = $run + $h[$base + $j];
                if ($run > $half) { $at = $j; break; }
                $j = $j + 1;
            }
            push($median, $at);
        }
        $b = $b + 1;
    }

    return {"bands": $nb, "h": $h, "extrema": $extrema, "count": $count,
            "sum": $sum, "sum2": $sum2, "mean": $mean, "median": $median,
            "rms": $rms, "var": $var, "stddev": $stddev};
}

// Global is Pillow's own alias.
def Global($src, $mask) { return Stat($src, $mask); }
