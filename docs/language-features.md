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
$f = open("data.txt", "w");    // modes: "r" "w" "a"
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
```

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

## Linter & compile gate (new)
```sh
bantu lint app.b            # human-readable diagnostics
bantu lint app.b --json     # machine-readable (used by the VS Code extension)
bantu run app.b             # refuses to run if there are errors
bantu run app.b --no-lint   # bypass the gate (const is still enforced at runtime)
```
In VS Code, errors show a **red** squiggle and warnings a **yellow** one, live as you type.
