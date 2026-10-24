# Bantu language features (v1.3.0)

New and fixed language features. Everything follows Bantu style — `$`-variables, `def`,
`//` comments, and **curly braces are mandatory** for every block.

## Control flow

### Compound assignment (fixed)
```bantu
$x = 10;
$x += 5;   // 15
$x *= 2;   // 30
$x -= 3;   // 27
$x /= 2;   // 13.5
```

### break / continue (fixed)
```bantu
$i = 0;
while ($i < 100) {
    $i += 1;
    if ($i == 3) { continue; }   // skip 3
    if ($i == 6) { break; }      // stop at 6
    print(str($i));
}
```

### switch / case / default (new)
Braces required, **no fallthrough** — the first matching case runs and control leaves.
```bantu
switch ($n) {
    case 1 { print("one"); }
    case 2 { print("two"); }
    default { print("many"); }
}
```

### throw / try / catch (fixed + new)
```bantu
try {
    throw {"code": 42, "msg": "boom"};   // throw any value
} catch ($e) {
    print("caught " + str($e.code) + ": " + $e.msg);
}

// Runtime errors are catchable too — $e is {message, type, line}:
try { $x = 1 / 0; } catch ($e) { print($e.message); }   // "Division by zero"
```

## const is truly constant (fixed)
A `const` binding cannot be reassigned (like Java `final`). Enforced at runtime and flagged by
the linter. The referenced object may still be mutated.
```bantu
const $PI = 3.14159;
$PI = 3;   // error: cannot reassign constant 'PI'
```

## Functions

### Anonymous functions (new)
`def` is a first-class value — use it inline.
```bantu
$handlers = {
    "double": def($x) { return $x * 2; },
    "square": def($x) { return $x * $x; }
};
print(str($handlers.double(5)));   // 10

$apply = def($f, $v) { return $f($v); };
print(str($apply($handlers.square, 9)));   // 81
```

### Bare return (fixed)
```bantu
def maybe($x) {
    if ($x < 0) { return; }   // returns null
    return $x * 2;
}
```

### Calls are limited to 1,000 deep (fixed)
```bantu
def countdown($n) {
    if ($n <= 0) { return 0; }
    return 1 + countdown($n - 1);
}

countdown(900)          // 900 — fine

try { countdown(100000); }
catch ($e) { print($e.message); }
// Maximum call depth of 1000 exceeded -- this is usually infinite recursion.
// If the recursion is genuine, rewrite it as a loop with an explicit stack.
```
Every Bantu call is a real C++ recursion, so an unbounded one used to walk off the native
stack: measured on macOS, 3,000 frames survived and 3,200 died with **SIGSEGV** — no message,
and no `try` that could catch it. From a sua handler it took the worker process with it, and
Windows was worse still, since its default thread stack is 1 MB against 8 MB here.

The limit is therefore **a number we choose, not one the stack imposes** — low enough to hold
on the smallest platform stack, high enough that no reasonable recursive algorithm reaches it.
Functions, methods, constructors and mutually recursive pairs all count against it, and
exceeding it raises an ordinary catchable error rather than ending the process.

If you hit it with genuine recursion rather than a missing base case, rewrite the function as a
loop with an explicit stack — a list you `push` onto and `pop` from does the same job with no
depth limit at all.

## Globals, as seen from inside a function

A function **reads** a global freely. **Assigning to its name inside a function does not change
the global** — it creates a local that is discarded when the call returns:

```bantu
$peers = {};

def addBroken($id, $name) {
    $next = {};
    each ($k in keys($peers)) { $next[$k] = $peers[$k]; }
    $next[$id] = $name;
    $peers = $next;              // writes a LOCAL. The global is untouched.
}
```

Writing a **field** of a global does propagate, because a dict is a reference type and this is an
index-set on it rather than a rebinding:

```bantu
$peers = {};
def add($id, $name) { $peers[$id] = $name; }     // works
```

So state a function must replace wholesale belongs in a **field of a container**:

```bantu
$S = { "peers": {} };

def dropPeer($id) {
    $next = {};
    each ($k in keys($S["peers"])) {
        if ($k != $id) { $next[$k] = $S["peers"][$k]; }
    }
    $S["peers"] = $next;         // an index-set on $S — this one propagates
}
```

This matters most in callbacks — `sua.ws.on("disconnect", ...)`, a route handler — where the
whole point is to mutate shared state and there is no return value to carry it out. The failure is
silent: the write appears to succeed and the global simply never changes. In
`samples/webrtc-chat` it meant a peer who disconnected was never removed, so "online" never went
down and every browser kept a ghost of them.

Two related points, since they come up in the same code:

- **Lists are values, dicts are references.** `$b = $a;` copies a list, so passing a list to a
  function copies it; a dict or class instance is shared. A registry that functions mutate should
  be a dict for that reason alone.
- **There is no key-removal builtin.** `remove(list, index)` is for lists, and `delete` is a
  reserved word (it is `sua.server.delete`). Drop a key by rebuilding the dict without it and
  replacing the field, as above.

## Collections

### Dict iteration (new — Python style)
```bantu
$d = {"a": 1, "b": 2, "c": 3};

for $key, $value in $d.items() {
    print($key + " = " + str($value));
}

for $x in [10, 20, 30] { print(str($x)); }   // for-in over lists

// each also takes two vars:
each ($k, $v in $d) { print($k); }
```

### Dict methods & builtins (new)
```bantu
$d.keys()     // list of keys
$d.values()   // list of values
$d.items()    // list of [key, value] pairs
$d.size()     // number of entries
keys($d)   values($d)   entries($d)   // builtin equivalents
```

### In-place list mutators (new)
```bantu
$l = [1, 2, 3];
append($l, 4);        // [1,2,3,4]  (mutates in place)
$last = pop($l);      // 4, list → [1,2,3]
insert($l, 0, 0);     // [0,1,2,3]
remove($l, 2);        // removes index 2 → [0,1,3]
extend($l, [7, 8]);   // [0,1,3,7,8]
$l.size()             // 5
```

**What each form returns.** `append`, `insert`, `extend` and `$l.push(x)` all return the **new
length**, as in JavaScript. The bare `push($l, x)` returns the **list** when its value is used, so
the older `$x = push($x, v)` idiom still works:

```bantu
$l = [1, 2];
$n = $l.push(3);      // $n = 3 (the new length), $l = [1,2,3]
$x = [1];
$x = push($x, 2);     // $x = [1,2]  -- the value IS used, so the list comes back
push($x, 3);          // a statement: the result is discarded, nothing is copied
```

That distinction is a performance fix, not a style choice. A Bantu list is copied by value, so
returning the mutated list from every `push` deep-copied every element — an O(1) append became
O(n), and building a list in a loop became **O(n²)**. It measured 9,491 ms for 20,000 pushes against
69 ms for the identical `append`, and 100,000 pushes took about four minutes. Both forms are now
linear (100,000 pushes ≈ 0.3 s). The gate is [`tests/lang_list_test.b`](../tests/lang_list_test.b).

**`len($var)` no longer copies its argument either.** Bantu lists have value semantics, so passing
one to a function copies the whole list — and `len` is the one builtin routinely called on the very
container being built:

```bantu
while (...) { $out[len($out)] = $v; }    // 20,000 items: 7,027 ms before, 58 ms now
```

That idiom is used throughout `hash.b` and `crypto.b`, so their pure-Bantu paths were quadratic in
input length. `len($var)` now reads the length from the real storage. The answer was identical in
every case — including a dict (then `0`) and a non-container (`0`) — and a user-defined `len()` still
shadows the builtin. A dict has since learned to report its real length; see *`len` and `contains`
on every container* below.

**Passing a big list to any other function still copies it.** This is inherent to the value
semantics and has not changed: `col($big, "f64")`, `sum($big)` and any user function taking a list
copy on the way in. For large data, prefer native containers (arctic columns) over Bantu lists.

### `sort` and `reverse` (new)

The language had `push`, `pop`, `insert`, `remove`, `extend` and `slice`, and **no way to order a
list**. Every median, quantile, boxplot, ranking and "top N" in every Bantu program was an
interpreted sort. Both return a **new** list; the argument is untouched, as value semantics require.

```bantu
sort([3, 1, 2])                  // [1, 2, 3]
sort([3, 1, 2], "desc")          // [3, 2, 1]
sort(["pear", "apple", "fig"])   // ["apple", "fig", "pear"]
reverse([1, 2, 3])               // [3, 2, 1]
reverse("abc")                   // "cba"

def byLength($a, $b) { return len($a) - len($b); }
sort(["aaa", "b", "cc"], byLength)      // ["b", "cc", "aaa"]
```

For anything bigger than a toy, sort by a **key** rather than a comparator:

```bantu
def byAge($r) { return $r["age"]; }

sort($rows, {"key": byAge})                  // youngest first
sort($rows, {"key": byAge, "desc": true})    // oldest first
sort($rows, {"desc": true})                  // no key, just the direction
```

A comparator is called **O(n log n)** times; a key is called **n** times, and the ordering itself
then happens in C++ on the keys alone. On a 100,000-row list that is the difference between about
1.7 million calls and 100,000. Measured on the same rows, with the cost of building the list
subtracted from both: **13,331 ms with a comparator against 764 ms with a key — 17× faster**, for an
identical result. Python replaced `cmp=` with `key=` in 3.0 for exactly this reason. The key function must return the same type for every element, and `NaN` keys sort last in
both directions, just as elements do.

Five things worth knowing, each with a reason:

- **`NaN` sorts last, in both directions.** This is a correctness requirement, not a preference:
  every comparison with `NaN` is false, so `a < b` is *not* a strict weak ordering when one is
  present, and `std::sort` given such a comparator reads past the end of its range — a genuine
  out-of-bounds access, not merely a wrong order. numba's `nd_sort` already orders `NaN` last, so
  the two agree on the same data.
- **`"desc"` reverses the comparison, not the finished list.** Reversing the list afterwards would
  reverse *ties* too, destroying stability, and would drag `NaN` to the front.
- **A mixed list raises.** Ordering a number against a string has no correct answer, and picking one
  silently is how a sort quietly produces garbage that still looks sorted.
- **A comparator gets a merge sort.** A comparator written in Bantu can be non-transitive and no
  validation catches that; a merge sort cannot leave its range whatever the comparator answers, so
  the worst case is a strangely ordered list rather than memory corruption. Return a negative number
  if the first argument comes first, positive if the second does, zero if they tie.

### `len` and `contains` on every container (fixed)

Both answered silently wrong for the containers they did not know about:

| call | before | now |
|---|---|---|
| `len({"a": 1, "b": 2})` | `0` | `2` — a dict counts its entries |
| `len($ndarray)` | `0` | the first axis, as in NumPy; a 0-d array raises |
| `len($column)` | `0` | the row count |
| `contains([1, 2], 1)` | `false` | `true` — list membership, with the same equality as `==` |

`while ($i < len($a))` over an array or a dict never ran, and `if (contains($seen, $x))` never
fired, with nothing to say why. `len` of anything else — a number, `null` — still answers `0`,
because `len(null)` is a common guard and changing it would break working programs.

### `+` joins two lists (fixed)

```bantu
$l = [1, 2] + [3];        // [1, 2, 3]
$p = [0] + $point;        // a new list; neither operand changes
```

`[1, 2] + [3]` used to read the numeric value of both lists and silently answer `0` — which then
indexed as `null`, so the mistake surfaced far from where it was made. Two lists now concatenate,
as in Python, into a new list (value semantics: neither operand is touched). Only list `+` list
changed; `+` with a list on one side and anything else on the other behaves as before.

### `&&` and `||` now short-circuit (fixed)

They did not, and that was a real defect rather than a quirk. The universal guard idiom

```bantu
if ($i < len($a) && $a[$i] == 9) { ... }     // died with "Index out of bounds"
if ($d != null && $d["k"] == 1) { ... }      // evaluated $d["k"] on a null
```

evaluated the right operand unconditionally and failed on exactly the boundary the guard was written
to prevent. Every programmer arriving from any other language writes that line.

The result is still a boolean, so nothing that already worked changes value — only the point at
which the right side stops being evaluated, and with it any side effect it carries. A/B'd on a
1M-iteration arithmetic loop containing no logical operators at all, best-of-5 in both orderings:
**540/541 ms without the guard against 535/533 ms with it**, so the cost is below the noise floor.

### Objects are freed (fixed — this was a leak)

`new ClassName()` allocated an instance that **nothing ever deleted**, so every object a Bantu
program created leaked for the life of the process. Measured before the fix: ~300 bytes per
instance, and a program building 20,000 plotting figures reached **372 MB** of resident memory and
climbing. A `sua` handler creating objects per request grew without bound until the worker was
killed.

Instances are now refcounted, like every other Bantu value. Measured after: 20,000 instances is
**flat** at 4.9 MB, and 6,000 figures built and dropped move resident memory from 10.5 MB to 12.0 MB.

### Cycles are collected too (fixed — this was the other half of the leak)

Reference counting cannot free a cycle: two objects that point at each other each hold the other's
count at one. That left three ordinary things leaking without bound, and **two of them the user never
wrote** — the interpreter built the cycle itself:

```bantu
$a.peer = $b; $b.peer = $a;     // a tree node and its parent
$a.callback = $a.someMethod;    // a handler stored on its own object
def outer() { def helper() { … } }   // a private helper inside a function
```

The second is a bound method: binding one builds a scope holding `this` and a function closing over
that scope. The third is a nested `def`: the call frame holds the function and the function closes
over the call frame, so **every call leaked its whole frame**. Measured before the collector, 400,000
iterations of each: the mutual pair reached **502 MB**, the stored method **558 MB**, and 20,000
nested-`def` calls held **20,001 scopes**.

Bantu now runs a **cycle collector** alongside reference counting, the way CPython and PHP do.
Refcounting still frees the overwhelming majority promptly; the collector handles only what it
provably cannot, running at a statement boundary once enough garbage has built up. After it, the
same 400,000-iteration runs are **flat at 17–20 MB**, and ten times the run length costs under twice
the memory.

**A program that has no cycles never pays for it**: one million acyclic objects trigger **zero**
collections. The interpreter benchmark is **−0.85%** against the build before the collector existed —
inside the ±2% gate, and below the noise floor.

You can watch it, and turn it off:

```bantu
gc_stats()      // {"live", "collections", "freed", "threshold", "enabled"}
gc_collect()    // collect now; returns how many objects were freed
gc_enable(false)   // stop the automatic one; returns the previous setting
```

`BANTU_GC=0` in the environment disables automatic collection for the process. `gc_collect()` still
works when it is off, so the switch is a diagnostic and a latency escape hatch, never a way to lose
the fix.

Two limits worth knowing. A cycle is freed at the **next collection**, not at the statement that
dropped it — only non-cyclic garbage is freed promptly. And a cycle that runs through a **native
closure's captures** cannot be traced, so it is retained rather than freed; the one place in the tree
that ever built one (sua's `$res` object) captures weakly instead.

Design, and the alternatives rejected: [`docs/object-lifetime-architecture.md`](object-lifetime-architecture.md).


## Scalar maths (new)

The language shipped with `abs ceil cos floor log max min pow round sin sqrt tan random`
and nothing else — no `exp`, no `atan2`, no `asin`/`acos`, no `log10`, no `PI`.

### Constants
```bantu
PI      // 3.141592653589793     — also reachable as $PI
TAU     // 6.283185307179586     — 2*PI
E       // 2.718281828459045
INF     // positive infinity
NAN     // a quiet NaN
```
These live in the same namespace as everything else, so `$PI = 3;` replaces the constant
for the rest of your program — the same rule that lets `$len = 3;` destroy `len()`.
`E` is the likeliest to be shadowed by accident.

### Functions
```bantu
exp($x)      expm1($x)    log1p($x)    log2($x)     log10($x)    cbrt($x)
asin($x)     acos($x)     atan($x)     atan2($y,$x) hypot($x,$y)
sinh($x)     cosh($x)     tanh($x)     asinh($x)    acosh($x)    atanh($x)
trunc($x)    sign($x)     fmod($x,$y)  copysign($x,$y)
degrees($x)  radians($x)  clamp($x,$lo,$hi)
isnan($x)    isinf($x)    isfinite($x)
```

Three that are not just conveniences:

- **`atan2($y, $x)`** knows which quadrant the point is in, which `atan($y/$x)` cannot,
  and it is defined at `$x == 0`.
- **`hypot($x, $y)`** does not overflow: `hypot(1e200, 1e200)` is finite where
  `sqrt($x*$x + $y*$y)` is `inf`.
- **`isnan` / `isinf` / `isfinite`** — the language could always *produce* these values
  (`log(0)` is `-inf`, `sqrt(-1)` is `nan`, `pow(10,400)` is `inf`) but until now there
  was no way to test for one.

Domain and range behaviour is IEEE 754's: `acos(2)` is `NaN` rather than an error, `log(0)`
is `-inf`, and NaN propagates. A **non-number argument raises** a catchable error naming the
argument and its type — unlike the older maths builtins, where `sqrt("hello")` quietly
answers `0`.

### max / min are variadic and list-aware (fixed)
```bantu
max(1, 2, 9)        // 9   — answered 2 before this fix
min(5, 4, 1)        // 1   — answered 4
max([3, 17, 5])     // 17  — a single list argument is reduced over
```
`max` and `min` read only their first two arguments and silently ignored the rest.
A single list argument now walks the list natively, which is how you take the range of a
100,000-point series in 13 ms instead of a 100,000-iteration loop.

**NaN propagates** — `max(1, NAN, 3)` is `NaN`, matching NumPy's `max` (as opposed to its
separate `nanmax`). A primitive should not silently discard a value it was handed; filter
first if you want NaN skipped.

### Bit operations are 32-bit

Bantu has no `&`, `|`, `^`, `<<` or `>>` operators. The work is done by builtins, and they
operate on **unsigned 32-bit integers**, like JavaScript's bitwise operators:

```bantu
band(12, 10)        // 8
bor(12, 10)         // 14
bxor(12, 10)        // 6
shl(1, 31)          // 2147483648
shr(256, 4)         // 16

shl(1, 32)          // 1   — NOT 4294967296
shl(1, 40)          // 256 — NOT 1099511627776
```

Both operands are truncated to `uint32`, and the shift count is masked to its low 5 bits, so a
shift of 32 is a shift of 0 and a shift of 40 is a shift of 8. This is deliberate and matches
JavaScript, but it means **64-bit bit manipulation is not expressible in Bantu** — 64-bit
hashes, checksums, packed flags and bitboards have to be split into two 32-bit halves by hand,
or done natively. Anything in the standard library that needs 64-bit words (SHA-512, for
instance) computes them in C++ for exactly this reason.

Note also that numbers are `float64` throughout, so integers are exact only to 2^53:
`9007199254740993` prints as `9007199254740992`.

## Strings

### join (new)
```bantu
join(["a", "b", "c"], "-")   // "a-b-c"
join([1, 2, 3], ",")         // "1,2,3"   — non-strings stringify as print would
join(["a", "b"])             // "ab"      — a missing separator means ""
```

`split()` has existed since v1.0 and its inverse never did, which mattered more than it
sounds: without `join`, the only way to build a string in a loop was `$s = $s + $part`,
and that is **O(n²)** — every `+` copies the whole accumulated string.

```bantu
// Don't: 40,000 appends take 6,752 ms
$s = "";
$i = 0;
while ($i < 40000) { $s = $s + $part; $i = $i + 1; }

// Do: the same 40,000 parts take 147 ms to collect and 13 ms to join
$parts = [];
$i = 0;
while ($i < 40000) { push($parts, $part); $i = $i + 1; }
$s = join($parts, "");
```

### str() past 2^63 (fixed)
`str(1e21)` answered `"-9223372036854775808"`. Any integral double was cast to `long long`,
and converting a floating-point value outside the destination integer range is undefined
behaviour — it saturates to `INT64_MIN` on x86-64 and ARM64. It now prints `1e+21`, and
`num(str($x))` round-trips at every magnitude.

### `num()` reads the whole string (fixed)

`num()` used `std::stod`, which reads the longest *prefix* that looks like a number and throws on
overflow — and every exception was caught and turned into `0`. So it produced plausible wrong
numbers in exactly the code that parses input it did not write:

| call | before | now |
|---|---|---|
| `num("12abc")` | `12` | `0` — trailing junk means it is not a number |
| `num("0x10")` | `16` | `0` — decimal only; `str()` never writes hex |
| `num("1e999")` | `0` | `inf` |
| `num(true)` | `0` | `1` |

Surrounding whitespace is fine, and `"inf"` and `"nan"` are read, so `num(str($x))` still
round-trips. **Unparsable input still reads `0`**, so a handler written as
`num($req.query["page"])` keeps working. To tell a real zero from no number at all, pass a second
argument — it is returned instead:

```bantu
$page = num("12abc", null);     // null, not 12
if ($page == null) { $page = 1; }
```

## Performance

### List indexing is no longer O(n) (fixed)
`$a[$i]` copied the **entire list** to read one element out of the copy, because a `Value`
holding a list owns its elements inline and evaluation returns values by value. Every loop
over a list was therefore quadratic.

| | before | after |
|---|---|---|
| 10,000 reads | 1,919 ms | **26 ms** |
| 20,000 reads | 8,093 ms | **52 ms** |
| 100,000 reads | — | **282 ms** |

Measured on an i7-9750H. The old cost grew 4.2× for a 2× workload; the new one is linear.
Dicts and class instances never had this problem — they hold a `shared_ptr`, which is also
why they have reference semantics and lists do not. Nothing about list semantics changed:
`$b = $a;` is still a copy.

## Comparison

`==` and `!=` are type-aware. `<`, `<=`, `>` and `>=` compare two **numbers**
numerically and two **strings** lexicographically, byte by byte — which for UTF-8 is also
code-point order. A comparison between a string and a number still coerces to numbers.

**Strings used to compare as numbers (fixed).** Both sides were read as `numberVal`, which is
`0` for any string, so **every** `<` and `>` between strings answered `false` and every `<=`
and `>=` answered `true`:

```bantu
print("a" < "b");        // was false — now true
print("b" > "a");        // was false — now true
print("apple" < "banana");  // was false — now true
$ch = substr($s, $i, 1);
if ($ch >= " ") { ... }  // was ALWAYS taken, so this kept control characters
```

Nothing could be sorted alphabetically, and a range check on a character silently admitted
everything. `==` and `!=` were never affected — they go through a type-aware path — which is why
this survived as long as it did. `tests/lang_test.b` now asserts the ordering in both directions.

## JSON

`json.parse` and `json.stringify`, the HTTP request-body parser and a WebSocket frame's
`$msg.json` all share one implementation.

**`\uXXXX` escapes used to be dropped, and they desynchronised the parser (fixed).** The
escape was skipped rather than decoded, and the position advanced seven characters for a
six-character escape — so it also ate the character *after* it:

```bantu
json.parse("{\"s\":\"caf\\u00e9\"}")["s"]       // was "caf}"  — now "café"
json.parse("{\"s\":\"\\u0041\\u0042\"}")["s"]     // was "u0042" — now "AB"
json.parse("{\"a\":\"caf\\u00e9\",\"b\":\"x\"}")  // 'a' swallowed its closing quote
```

That second effect is the dangerous one: a string could corrupt the structure around it, not
only itself. It mattered well beyond Bantu's own code, because **an ASCII-safe JSON encoder
escapes every non-ASCII character** — Python's `json.dumps` does so by default — so any client
posting `{"name": "café"}` had the character silently deleted and the object potentially
misparsed. Escapes are now decoded to UTF-8, surrogate pairs included, so an escaped emoji
survives a round trip; a lone surrogate becomes U+FFFD, which keeps the result valid UTF-8.

## Modules

```bantu
include "./routes.b";                 // brings the module's symbols into scope
include "./controller.b" as ctrl;     // binds a namespace object
include "numba" as np;                // a BARE name: an installed package
```

A bare name (no `./`, no slash) searches `bantu_modules/<name>/` beside the importing file and under
the working directory — where `bantu add <pkg>` installs — honouring the package's `package.json`
`"main"`, then falling back to `<name>.b`, `index.b`, `main.b`. A path with `./` or `../` always
means exactly that path. `$BANTU_PATH` is searched last.

A bare name that names a **directory** beside the importing file or under the working directory is
resolved *inside* it, the way `bantu_modules/<name>/` is — `package.json` `"main"`, then `<name>.b`,
then `index.b` — which is how `include "bplot"` works from a checkout that has `bplot/bplot.b`, and is
Node's convention for `require('./dir')`. An installed package still wins over a folder that happens
to share its name.

**A bare name is searched up the whole tree (fixed).** Both bare-name rules looked only at the
importing file's own directory and the working directory, and nowhere else — so a bare name meant
different things depending on where the process happened to start. `include "bplot" as plt;`
resolved from the repo root and raised `Module not found: bplot` from `samples/bplot/`: the same
file, run from its own folder, and the message named the module rather than the cwd that caused it.
Every bare name is now looked for in the importing file's directory **and in every directory above
it**, nearest first, then the same walk up from the cwd — the way `node_modules` is searched up the
tree, and Python's parent packages, and Cargo's workspace root. An explicit `./` or `../` path is
untouched, precedence is unchanged (an installed `bantu_modules/` package still beats a same-named
folder), and a name that is genuinely absent is still reported. This is what makes
`include "bplot" as plt;` work anywhere inside a project, which is what the examples and
`docs/bplot.md` have always told you to write.

`tests/lang_module_cwd_test.sh` runs the whole module suite from the repo root, from `tests/`, from
a sample directory, from `$HOME` and from `/`, and requires the same result each time.

**A module is a regular file (fixed).** The resolver's existence check used to accept a directory,
and a directory opened as a stream reads as empty — so `include "bplot" as plt;` run from a folder
containing a `bplot/` directory parsed an empty file and bound `plt` to a module with nothing in it.
No error was raised; the first sign was `Cannot call 'figure': it holds null` at the first call.

**One file, one module.** A module executes once; every later `include` of it binds that same
namespace object, so two of your files can both `include "arctic" as arctic;` and both get a working
`arctic`. (Before v1.3.0 the second one silently bound *nothing*.) A genuine circular include —
where the file is still mid-execution and has exported nothing yet — is reported by name and
skipped.

**Classes cannot be namespaced**: `new ctrl.Thing()` does not parse. Packages expose factory
functions instead (`arctic.series(...)`, `np.array(...)`). Class names are a flat global registry,
so a package should prefix its class names to avoid collisions.

## File I/O (new — Python style)
```bantu
$f = open("data.txt", "w");    // text: "r" "w" "a"   binary: "rb" "wb" "ab"
write($f, "hello\n");
close($f);

$f2 = open("data.txt", "r");
$firstLine = readline($f2);
$rest = read($f2);
close($f2);

// one-shot helpers:
$all = readfile("data.txt");
writefile("out.txt", "hi");
appendfile("log.txt", "line\n");
$lines = readlines(open("data.txt", "r"));

// the helpers take the same modes, as an optional last argument:
$png = readfile("chart.png", "rb");
writefile("copy.png", $png, "wb");
appendfile("frames.bin", $png, "ab");
```

### Binary files, and modes that do not exist (fixed)

Two defects, both silent, both fatal to any binary file:

- **An unknown mode opened the file for reading.** `open()` recognised `"r"`, `"w"` and `"a"` and sent
  everything else to read mode — so `open($path, "wb")`, `"w+"`, `"r+"` and `"x"` all opened the file
  for *reading*, and every `write()` to it failed without a word. An unknown mode now **raises**, naming
  the modes that exist, and `readfile()` refuses a write mode *before* opening, so it can never truncate
  the file it was asked to read.
- **Nothing set `std::ios::binary`.** On Windows a text-mode stream turns every `\n` written into
  `\r\n` and treats byte `0x1A` as the end of the file — which destroys a PNG at its eighth byte. The
  `"rb"`, `"wb"` and `"ab"` modes are binary on every platform.

**The defaults are still text**, so no existing program changes behaviour anywhere. `"rt"`, `"wt"` and
`"at"` are accepted as spellings of the text modes. `write()` to a file opened for reading, and a
`writefile()` that fails part-way — a full disk — now raise instead of reporting success.

## FFI — call C libraries (new, via libffi)
Type names: `"int"`, `"double"`, `"string"`, `"pointer"`, `"void"`.
```bantu
$m = loadlib("libm.dylib");           // "libm.so.6" on Linux
$sqrt = func($m, "sqrt", "double", ["double"]);
print(str($sqrt(2.0)));               // 1.41421356

$c = loadlib("libc.dylib");
$strlen = func($c, "strlen", "int", ["string"]);
print(str($strlen("hello")));         // 5
```

## Parameterized SQL (new — injection-safe)
```bantu
sua.sqlite.exec("INSERT INTO users(name, age) VALUES(?, ?)", ["Ada", 36]);
$rows = sua.sqlite.query("SELECT * FROM users WHERE age > ?", [18]);
```

## Reserved words as variables (fixed)
The `$` sigil means "variable", so reserved words are usable as variable names:
```bantu
$db = sua.sqlite;   $list = [1, 2, 3];   $create = "ok";
```

**Seven words were reserved for nothing (fixed).** `read`, `await`, `private`, `public`, `calc`,
`import` and `export` were lexed as keywords that no grammar rule accepted. The cost was concrete:
`read($f)`, a documented builtin, **could not be called at all** — even `$rest = read($f);` failed
with `Unexpected token: 'read'` — and no function could be named `calc` or `export`. They are
ordinary names now. Nothing that parsed before parses differently, because no rule ever accepted
those tokens.

## Linter & compile gate (new)
```sh
bantu lint app.b            # human-readable diagnostics
bantu lint app.b --json     # machine-readable (used by the VS Code extension)
bantu run app.b             # refuses to run if there are errors
bantu run app.b --no-lint   # bypass the gate (const is still enforced at runtime)
```
In VS Code, errors show a **red** squiggle and warnings a **yellow** one, live as you type.
