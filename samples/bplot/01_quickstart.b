// ────────────────────────────────────────────────────────────────────────
//  01_quickstart.b — three lines to a chart.
//
//  Run:  bantu run samples/bplot/01_quickstart.b
// ────────────────────────────────────────────────────────────────────────
include "./bplot/bplot.b" as plt;

plt.plot([1, 2, 3, 4, 5], [2, 4, 9, 3, 7]);
plt.savefig("/tmp/bplot_quickstart.svg");

print("wrote /tmp/bplot_quickstart.svg");

// That is the whole API for a first chart. Everything below is optional.
plt.plot([1, 2, 3, 4, 5], [1, 3, 2, 5, 4], {"label": "measured"});
plt.plot([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], {"label": "expected", "color": "orange"});
plt.title("Measured against expected");
plt.xlabel("trial");
plt.ylabel("score");
plt.grid(true);
plt.legend(true);
plt.savefig("/tmp/bplot_quickstart2.svg");
print("wrote /tmp/bplot_quickstart2.svg");
