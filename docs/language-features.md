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
input length. `len($var)` now reads the length from the real storage. The answer is identical in
every case, including `len` of a string, a dict (`0`) and a non-container (`0`), and a user-defined
`len()` still shadows the builtin.

**Passing a big list to any other function still copies it.** This is inherent to the value
semantics and has not changed: `col($big, "f64")`, `sum($big)` and any user function taking a list
copy on the way in. For large data, prefer native containers (arctic columns) over Bantu lists.


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
