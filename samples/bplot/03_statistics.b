// ════════════════════════════════════════════════════════════════════════
//  bplot — the statistical chart types.
//
//  hist, boxplot, violin and errorbar are not drawing routines. Each is an
//  algorithm with a published definition, and bplot matches the definition
//  the rest of this tree already uses: histogram bins follow numba's
//  nd_histogram exactly, and quantiles follow nd_quantile, so the same data
//  summarised as a Bantu list and as a numba array gives the same chart.
//
//  Run:  bantu run samples/bplot/03_statistics.b
// ════════════════════════════════════════════════════════════════════════

include "./bplot/bplot.b" as plt;

// A reproducible sample, so this program draws the same chart every time.
// A chart in a report that moves between runs is a chart nobody can cite.
def sample($n, $seed, $centre, $spread) {
    $out = [];
    $s = $seed;
    $i = 0;
    while ($i < $n) {
        // Two uniforms averaged twelve times would be a normal; four is
        // enough for a shape that reads as bell-like at this scale.
        $acc = 0;
        $k = 0;
        while ($k < 4) {
            $s = ($s * 1103515245 + 12345) - floor(($s * 1103515245 + 12345) / 2147483648) * 2147483648;
            $acc = $acc + $s / 2147483648;
            $k = $k + 1;
        }
        push($out, $centre + ($acc / 4 - 0.5) * $spread);
        $i = $i + 1;
    }
    return $out;
}

$a = sample(4000, 7, 50, 40);
$b = sample(4000, 99, 62, 22);
$c = sample(4000, 4242, 44, 60);

$fig = plt.figure(1000, 720);
$axes = $fig.subplots(2, 2);

// ── A histogram ──────────────────────────────────────────────────────────
$axes[0].hist($a, {"bins": 30, "color": "#1f77b4"});
$axes[0].setTitle("hist — 4,000 samples in 30 bins");
$axes[0].setXLabel("value");
$axes[0].setYLabel("count");

// ── A box plot: five order statistics and the outliers ───────────────────
// Whiskers reach the most extreme OBSERVED value within 1.5 x IQR, which is
// Tukey's rule and matplotlib's. Drawing them at the fence itself — the
// frequent error — puts the whisker end where no observation exists.
$axes[1].boxplot([$a, $b, $c], {"labels": ["alpha", "beta", "gamma"]});
$axes[1].setTitle("boxplot — Tukey, 1.5 x IQR");
$axes[1].setYLabel("value");

// ── A violin: the whole distribution, not five numbers of it ─────────────
// The density is estimated from a 512-bin histogram rather than from every
// point, so the cost does not grow with the sample size.
$axes[2].violin([$a, $b, $c], {"labels": ["alpha", "beta", "gamma"]});
$axes[2].setTitle("violin — binned KDE, Scott's bandwidth");
$axes[2].setYLabel("value");

// ── Error bars: asymmetric, because real uncertainty usually is ──────────
$x = [1, 2, 3, 4, 5];
$y = [12, 19, 14, 23, 21];
$lower = [2, 3, 1, 4, 2];
$upper = [5, 2, 3, 6, 4];
$axes[3].errorbar($x, $y, {"yerr": [$lower, $upper], "line": true, "color": "#d62728"});
$axes[3].setTitle("errorbar — asymmetric [lower, upper]");
$axes[3].setXLabel("run");
$axes[3].setYLabel("measurement");

$fig.tight_layout(true);
$path = $fig.savefig("/tmp/bplot_statistics.svg");
print("wrote " + $path);

// ── A log axis, and the error it exists to produce ───────────────────────
// Non-positive data on a log scale is the classic silent-garbage case:
// log10(0) is -inf, which reaches the document as a NaN coordinate, which
// every browser renders as NOTHING AT ALL with nothing in any log. bplot
// raises instead, names the value, and points at symlog.
$fig2 = plt.figure(720, 420);
$ax2 = $fig2.addAxes();
$ax2.setScale("y", "log", null);
$ax2.plot([1, 2, 3, 4, 5], [1, 30, 900, 27000, 810000], {"label": "3^n"});
$ax2.setGrid(true);
$ax2.setLegend(true);
$ax2.setTitle("a log axis — decade ticks, 2..9 minors");
$fig2.tight_layout(true);
print("wrote " + $fig2.savefig("/tmp/bplot_log.svg"));

try {
    $bad = plt.figure(400, 300);
    $bax = $bad.addAxes();
    $bax.setScale("y", "log", null);
    $bax.plot([1, 2, 3], [10, 0, 30], null);
    $bad.to_svg();
    print("UNREACHABLE: a zero on a log axis should have raised");
} catch ($e) {
    print("");
    print("and a zero on a log axis is refused rather than drawn as nothing:");
    if (type($e) == "string") { print("  " + $e); } else { print("  " + str($e["message"])); }
}
