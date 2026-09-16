// ════════════════════════════════════════════════════════════════════════
//  lang_file_test.b — file modes, and binary data that survives the trip.
//
//  open() accepted "r", "w" and "a" and sent ANY OTHER MODE to read mode, so
//  open($path, "wb") silently opened the file for reading and every write to
//  it failed without a word. And nothing in the file builtins set
//  std::ios::binary, so on Windows a binary file written through them had
//  every \n byte turned into \r\n, and reading stopped at a 0x1A byte.
//
//  That is fatal for a PNG, whose eight-byte signature is exactly the bytes a
//  text-mode stream mangles: 0x89 P N G \r \n 0x1A \n. It is why bplot's raster
//  backend (B6) starts here.
//
//  Design: docs/bplot-raster-architecture.md §8
// ════════════════════════════════════════════════════════════════════════

$R = {"pass": 0, "fail": 0};
def ok($cond, $what) {
    if ($cond) { $R["pass"] = $R["pass"] + 1; }
    else { $R["fail"] = $R["fail"] + 1; print("  FAIL: " + $what); }
}
def eq($got, $want, $what) {
    if ($got == $want) { $R["pass"] = $R["pass"] + 1; }
    else { $R["fail"] = $R["fail"] + 1; print("  FAIL: " + $what + " -- got " + str($got) + ", wanted " + str($want)); }
}
// Raises, and the message contains `fragment`.
def raises($fn, $fragment, $what) {
    $msg = null;
    try { $fn(); } catch ($e) { $msg = str($e); }
    if ($msg == null) { ok(false, $what + " (did not raise)"); return null; }
    if (!contains($msg, $fragment)) { print("    message was: " + $msg); }
    ok(contains($msg, $fragment), $what);
    return null;
}

$base = "/tmp/bantu_lang_file_" + str(clock()) + "_";

// Every byte value, once, in order.
$all = [];
$i = 0;
while ($i < 256) { push($all, chr($i)); $i = $i + 1; }
$bytes = join($all, "");
eq(len($bytes), 256, "(setup) a string can hold all 256 byte values");

print("── binary modes round-trip every byte ────────────────────────────");

writefile($base + "all.bin", $bytes, "wb");
$back = readfile($base + "all.bin", "rb");
eq(len($back), 256, "writefile wb + readfile rb keeps all 256 bytes");
ok($back == $bytes, "and every one of them is the byte that was written");

// The PNG signature is the canonical text-mode casualty.
$sig = chr(137) + "PNG" + chr(13) + chr(10) + chr(26) + chr(10);
writefile($base + "sig.png", $sig, "wb");
ok(readfile($base + "sig.png", "rb") == $sig, "a PNG signature survives: 0x89 PNG CR LF 0x1A LF");

// Runs of the bytes text mode treats specially.
$nul = "";
$i = 0;
while ($i < 64) { $nul = $nul + chr(0); $i = $i + 1; }
$tricky = $nul + "x" + chr(13) + chr(10) + chr(10) + chr(13) + chr(26) + chr(26) + $nul + chr(255);
writefile($base + "tricky.bin", $tricky, "wb");
$tb = readfile($base + "tricky.bin", "rb");
eq(len($tb), len($tricky), "runs of NUL, CR, LF and 0x1A keep their length");
ok($tb == $tricky, "and their bytes");

// A megabyte, so the round trip is not only correct for small buffers.
$big = $bytes;
while (len($big) < 1000000) { $big = $big + $big; }
$t0 = clock();
writefile($base + "big.bin", $big, "wb");
$bigBack = readfile($base + "big.bin", "rb");
$ms = clock() - $t0;
eq(len($bigBack), len($big), "a 1 MB binary file keeps its length");
ok($bigBack == $big, "and its bytes");
ok($ms < 2000, "in well under two seconds");

print("── open() with the binary modes ──────────────────────────────────");

// "wb" used to open the file for READING, so this write failed silently.
$f = open($base + "open.bin", "wb");
eq(write($f, $bytes), 256, "open wb: write reports every byte");
close($f);
ok(readfile($base + "open.bin", "rb") == $bytes, "and they are really in the file");

$f = open($base + "open.bin", "ab");
write($f, $bytes);
close($f);
ok(readfile($base + "open.bin", "rb") == $bytes + $bytes, "open ab appends, byte for byte");

$f = open($base + "open.bin", "rb");
ok(read($f) == $bytes + $bytes, "open rb reads it back");
close($f);

appendfile($base + "open.bin", $sig, "ab");
ok(readfile($base + "open.bin", "rb") == $bytes + $bytes + $sig, "appendfile ab appends binary");

print("── text modes are unchanged ──────────────────────────────────────");

writefile($base + "text.txt", "line one\nline two\n");
eq(readfile($base + "text.txt"), "line one\nline two\n", "writefile and readfile default to text, as before");
appendfile($base + "text.txt", "line three\n");
eq(readfile($base + "text.txt"), "line one\nline two\nline three\n", "appendfile defaults to text, as before");
$f = open($base + "text.txt", "w");
write($f, "fresh\n");
close($f);
$f = open($base + "text.txt", "r");
eq(read($f), "fresh\n", "open w and open r, as before");
close($f);
$f = open($base + "text.txt", null);
eq(read($f), "fresh\n", "a null mode means the default");
close($f);
writefile($base + "text2.txt", "t", "wt");
appendfile($base + "text2.txt", "u", "at");
eq(readfile($base + "text2.txt", "rt"), "tu", "\"rt\", \"wt\" and \"at\" are accepted as text modes");
writefile($base + "text3.txt", "n", null);
eq(readfile($base + "text3.txt", null), "n", "and a null mode on the helpers means the default too");

print("── an unknown mode raises, and touches nothing ───────────────────");

writefile($base + "keep.txt", "keep me");
raises(def() { open($base + "keep.txt", "x"); }, "unknown mode 'x'", "open with mode x raises (it used to open for reading)");
raises(def() { open($base + "keep.txt", "w+"); }, "unknown mode 'w+'", "open with mode w+ raises");
raises(def() { open($base + "keep.txt", "r+"); }, "unknown mode 'r+'", "open with mode r+ raises");
raises(def() { open($base + "keep.txt", "rw"); }, "\"rb\"", "and the message names the modes that exist");
// A read helper handed a write mode must refuse before opening -- opening
// with trunc would empty the file it was asked to READ.
raises(def() { readfile($base + "keep.txt", "wb"); }, "unknown mode 'wb'", "readfile refuses a write mode");
eq(readfile($base + "keep.txt"), "keep me", "and the file it refused is untouched");
raises(def() { writefile($base + "keep.txt", "gone", "rb"); }, "unknown mode 'rb'", "writefile refuses a read mode");
raises(def() { writefile($base + "keep.txt", "gone", "a"); }, "unknown mode 'a'", "writefile refuses an append mode");
raises(def() { appendfile($base + "keep.txt", "gone", "w"); }, "unknown mode 'w'", "appendfile refuses a truncating mode");
eq(readfile($base + "keep.txt"), "keep me", "none of the refused calls changed the file");

print("── a write that fails says so ────────────────────────────────────");

$f = open($base + "keep.txt", "r");
raises(def() { write($f, "nope"); }, "mode 'r'", "write() to a file opened for reading raises, naming the mode");
close($f);
eq(readfile($base + "keep.txt"), "keep me", "and nothing was written");
raises(def() { readfile($base + "no_such_file.txt"); }, "Cannot read file", "a missing file still raises, as before");

print("");
print("Passed: " + str($R["pass"]) + "   Failed: " + str($R["fail"]));
if ($R["fail"] == 0) { print("RESULT: ALL GREEN"); } else { print("RESULT: FAILURES"); }
