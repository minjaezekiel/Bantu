// ════════════════════════════════════════════════════════════════════════
//  pillow/path.b — ImagePath.Path.
//
//      include "./path.b" as path;
//      $p = path.Path([0, 0, 10, 0, 10, 10]);
//      path.getbbox($p)     -> [0, 0, 10, 10]
//      path.tolist($p)      -> [[0,0], [10,0], [10,10]]
//
//  A Path is a flat sequence of x, y pairs. Pillow's version exists so that
//  ImageDraw can take a long polygon without rebuilding a Python list for
//  every call, and so a sequence of points can be mapped or transformed in
//  one step. The same two reasons apply here.
//
//  State is a dict rather than a class, because `new alias.Class()` does not
//  parse across a module boundary in Bantu -- so Path() is a function.
// ════════════════════════════════════════════════════════════════════════

// Path($points) -> a path from a flat [x, y, x, y, ...] list, or from a list
// of [x, y] pairs. Both forms are what Pillow accepts.
def Path($points) {
    if (type($points) != "list") { throw "ImagePath.Path: expected a list of points"; }
    $flat = [];
    if (len($points) > 0 && type($points[0]) == "list") {
        $i = 0;
        while ($i < len($points)) {
            if (len($points[$i]) < 2) {
                throw "ImagePath.Path: point " + str($i) + " needs an x and a y";
            }
            push($flat, $points[$i][0]);
            push($flat, $points[$i][1]);
            $i = $i + 1;
        }
    } else {
        if (len($points) % 2 != 0) {
            throw "ImagePath.Path: a flat point list must have an even length, got " +
                  str(len($points));
        }
        $i = 0;
        while ($i < len($points)) { push($flat, $points[$i]); $i = $i + 1; }
    }
    return {"xy": $flat};
}

// len_points($p) -> how many points (not how many numbers).
def len_points($p) { return floor(len($p.xy) / 2); }

// tolist($p, $flat) -> the points, as pairs by default or flat when asked.
def tolist($p, $flat) {
    if ($flat != null && $flat) {
        $out = [];
        $i = 0;
        while ($i < len($p.xy)) { push($out, $p.xy[$i]); $i = $i + 1; }
        return $out;
    }
    $out = [];
    $i = 0;
    while ($i + 1 < len($p.xy)) { push($out, [$p.xy[$i], $p.xy[$i + 1]]); $i = $i + 2; }
    return $out;
}

// getbbox($p) -> [xmin, ymin, xmax, ymax], or null for an empty path.
def getbbox($p) {
    if (len($p.xy) < 2) { return null; }
    $x0 = $p.xy[0];  $x1 = $p.xy[0];
    $y0 = $p.xy[1];  $y1 = $p.xy[1];
    $i = 2;
    while ($i + 1 < len($p.xy)) {
        if ($p.xy[$i] < $x0) { $x0 = $p.xy[$i]; }
        if ($p.xy[$i] > $x1) { $x1 = $p.xy[$i]; }
        if ($p.xy[$i + 1] < $y0) { $y0 = $p.xy[$i + 1]; }
        if ($p.xy[$i + 1] > $y1) { $y1 = $p.xy[$i + 1]; }
        $i = $i + 2;
    }
    return [$x0, $y0, $x1, $y1];
}

// transform($p, $matrix) -> every point through the 6-element affine
// [a, b, c, d, e, f]:  x' = a*x + b*y + c,  y' = d*x + e*y + f.
// Mutates the path and returns it, as Pillow's does.
def transform($p, $m) {
    if (len($m) < 6) { throw "ImagePath.transform: needs 6 coefficients"; }
    $i = 0;
    while ($i + 1 < len($p.xy)) {
        $x = $p.xy[$i];
        $y = $p.xy[$i + 1];
        $p.xy[$i]     = $m[0] * $x + $m[1] * $y + $m[2];
        $p.xy[$i + 1] = $m[3] * $x + $m[4] * $y + $m[5];
        $i = $i + 2;
    }
    return $p;
}

// map($p, $fn) -> every point through fn(x, y) -> [x, y]. Mutates.
def map_points($p, $fn) {
    $i = 0;
    while ($i + 1 < len($p.xy)) {
        $r = $fn($p.xy[$i], $p.xy[$i + 1]);
        $p.xy[$i] = $r[0];
        $p.xy[$i + 1] = $r[1];
        $i = $i + 2;
    }
    return $p;
}

// compact($p, $distance) -> drop points closer than `distance` to the one
// kept before them, using the CITY-BLOCK distance Pillow uses (|dx| + |dy|,
// not Euclidean) so no square root is needed. Returns how many were removed.
//
// The first and last points are always kept: dropping an endpoint changes
// the shape rather than simplifying it.
def compact($p, $distance) {
    $limit = 2;
    if ($distance != null) { $limit = $distance; }
    $n = len_points($p);
    if ($n < 3) { return 0; }
    $keep = [$p.xy[0], $p.xy[1]];
    $lx = $p.xy[0];
    $ly = $p.xy[1];
    $i = 1;
    while ($i < $n - 1) {
        $x = $p.xy[$i * 2];
        $y = $p.xy[$i * 2 + 1];
        $dx = $x - $lx;
        if ($dx < 0) { $dx = 0 - $dx; }
        $dy = $y - $ly;
        if ($dy < 0) { $dy = 0 - $dy; }
        if ($dx + $dy >= $limit) {
            push($keep, $x);
            push($keep, $y);
            $lx = $x;
            $ly = $y;
        }
        $i = $i + 1;
    }
    push($keep, $p.xy[($n - 1) * 2]);
    push($keep, $p.xy[($n - 1) * 2 + 1]);
    $removed = $n - floor(len($keep) / 2);
    $p.xy = $keep;
    return $removed;
}
