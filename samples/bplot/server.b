// ════════════════════════════════════════════════════════════════════════
//  bplot — charts served over HTTP by sua.
//
//  A chart is drawn per request and sent as image/svg+xml -- or as
//  image/png from /chart.png -- with a title taken from the query string,
//  which is to say, from a stranger.
//
//  Two rules make that safe, and both are shown here:
//
//    1. Use the OBJECT API in a handler: $fig = plt.figure(...). The plt.*
//       functions share one current figure for the whole process, so two
//       requests using them at once would draw into each other's chart.
//
//    2. SVG is an executable document format, not an image in the way PNG
//       is. bplot escapes every piece of text it emits, so a title cannot
//       inject markup through bplot -- and the Content-Security-Policy
//       header below means a browser opening the chart directly would run
//       nothing anyway. Keep both: defence in depth is the point.
//
//  Run:  bantu run samples/bplot/server.b      then open http://127.0.0.1:8080/
// ════════════════════════════════════════════════════════════════════════

include "bplot" as plt;

$port = 8080;
if (env("PORT") != null && env("PORT") != "") { $port = num(env("PORT")); }

// What a real dashboard would read from its database.
$months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
$rain   = [66, 61, 118, 290, 197, 33, 24, 22, 30, 41, 125, 118];

def chartFor($title) {
    $fig = plt.figure(760, 420);
    $ax = $fig.addAxes();
    $ax.bar($months, $rain, {"color": "#1f77b4"});
    $ax.setTitle($title);
    $ax.setYLabel("rainfall (mm)");
    $ax.setGrid(true);
    // Measured by the backend that draws it, so the SVG and the PNG each get
    // gutters for their own font -- and concurrent renders cannot interfere.
    $fig.tight_layout(true);
    return $fig;
}

sua.server.get("/chart.svg", def($req, $res) {
    $title = $req.query["title"];
    if ($title == null || $title == "") { $title = "Monthly rainfall, Dar es Salaam"; }
    $res.set("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'");
    $res.set("X-Content-Type-Options", "nosniff");
    $res.type("image/svg+xml; charset=utf-8");
    $res.send(chartFor($title).to_svg());
});

// The same chart as a PNG: pixels, not a document, so there is nothing in it
// to execute -- but it costs a render on the server rather than the browser.
sua.server.get("/chart.png", def($req, $res) {
    $title = $req.query["title"];
    if ($title == null || $title == "") { $title = "Monthly rainfall, Dar es Salaam"; }
    $res.set("X-Content-Type-Options", "nosniff");
    $res.type("image/png");
    $res.send(chartFor($title).to_png(null));
});

sua.server.get("/", def($req, $res) {
    $res.type("text/html; charset=utf-8");
    $res.send("<!doctype html><meta charset=\"utf-8\"><title>bplot on sua</title>" +
              "<h1>Rainfall</h1><img src=\"/chart.svg\" alt=\"Monthly rainfall\" width=\"760\" height=\"420\">" +
              "<h2>The same chart as a PNG</h2><img src=\"/chart.png\" alt=\"Monthly rainfall\" width=\"760\" height=\"420\">");
});

print("bplot charts on http://127.0.0.1:" + str($port) + "/");
sua.server.listen($port);
