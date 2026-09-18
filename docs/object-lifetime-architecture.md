# Object lifetime in Bantu — reference counting, and the cycle collector

**Status:** implemented and measured. §8 records the results against the gates set out below.
**Scope:** how a Bantu object is freed, why reference counting alone is not enough, and the
collector that closes the gap.
**Related:** [`docs/bantu-architecture.md`](bantu-architecture.md), decision BP30 in
[`bplot-suite/DECISIONS.md`](../bplot-suite/DECISIONS.md).

---

## 1. What is already true

Every Bantu value that owns heap memory is owned by a `std::shared_ptr`:

| value kind | owner |
|---|---|
| class instance | `Value::classInstancePtr` → `shared_ptr<ClassInstance>` |
| function / closure | `Value::functionPtr` → `shared_ptr<BantuFunction>` |
| dict (object) | `Value::objectVal` → `shared_ptr<ObjectMap>` |
| native handle (arctic column, numba array) | `Value::handle` → `shared_ptr<void>` |
| scope | `shared_ptr<Environment>`, with `Environment::parent` the chain |

Lists are different and deliberately so: `Value::listVal` is a `std::vector<Value>` held **by value**,
so a list is copied when it is assigned or passed. A list therefore has no identity of its own and
cannot be the *node* in a reference cycle — though it can carry the *edges* of one, because the
values inside it may be instances or dicts.

Reference counting was the right primary mechanism and stays the primary mechanism. It is prompt
(an object is freed at the statement that drops it, not at some later pause), it is already how
every native handle works, and it needs no knowledge of a root set — which matters a great deal in a
tree-walking evaluator where thousands of live `Value` temporaries sit in C++ stack frames that no
tracing collector could see without a shadow stack.

## 2. What is not true: reference counting cannot free a cycle

If two objects refer to each other, each keeps the other's count at one and neither is ever freed.
This is not hypothetical, and it is not rare. Measured on the current build, 400,000 iterations of
each pattern:

| pattern | live objects at exit | peak RSS |
|---|---|---|
| `$a = new Node();` — create and drop | **1** | 5.1 MB |
| `$a = new Node(); $a.self = $a;` | **400,000** | **246 MB** |
| `$a = new Node(); $b = new Node(); $a.p = $b; $b.p = $a;` | **800,000** | **502 MB** |
| `$a = new Node(); $a.cb = $a.m;` | **400,000** | **558 MB** |
| `$d = {}; $d["self"] = $d;` (20,000 iterations) | — | 13.0 MB vs 5.1 MB baseline |
| `def outer() { def inner() { … } }`, 20,000 calls | **20,001 scopes** | 34.0 MB |

Plain creation is flat, which is the refcounting fix working. Everything else grows without bound.

Read those rows again, because they are worse than "cycles leak", which every refcounted runtime
can say:

- **Row 3 is a tree with parent pointers.** A node that knows its parent and a parent that knows its
  children is the most ordinary data structure there is. In Bantu it leaks the entire tree.
- **Row 4 is the interpreter's own doing.** `$a.cb = $a.m` stores a method on its own object — a
  callback, an event handler, a strategy slot. The user wrote no back-reference at all. The cycle is
  manufactured by `evalDotAccess`, which binds a method by building a fresh `Environment` holding
  `this` and a fresh `BantuFunction` closing over it. The user cannot see that cycle in their code
  and cannot avoid it except by not storing methods.
- **Row 6 is a nested function.** `def` inside `def` is not an exotic technique; it is how anyone
  writes a private helper. `evalFuncDecl` stores the function into the scope it closes over, so the
  scope holds the function and the function holds the scope. Every call leaks its whole frame —
  1.7 KB a call here, and more as the frame grows. A `sua` handler with one nested helper leaks per
  request until the worker is killed.

This is also the reason bplot's `Axes` holds no pointer back to its `Figure` (BP30). That was a
library contorting itself around a runtime limitation, and every future Bantu library author would
have had to rediscover the same contortion. Removing the need for it is the point.

## 3. Options considered

**Do nothing, document it.** Rejected. Rows 4 and 6 are not something a user can be told to avoid,
because the user did not write them.

**Weak references only (`weakref($x)`).** Rejected as a complete answer. It puts the burden on
every library author, it cannot help rows 4 and 6 at all without the interpreter also using it
internally, and languages that have weak references (Python, C#, Swift) *also* ship a cycle
collector. Worth having later; not a fix.

**Replace refcounting with a tracing garbage collector.** Rejected. A precise tracing collector
needs a complete root set, and in this evaluator a large share of live references are C++ locals —
`Value obj = evalNode(...)` in some 200 places. Making those visible means a shadow stack touched on
every expression: an invasive, hot-path change to a 9,000-line file, with a use-after-free for every
site that gets missed. It would also fight the `shared_ptr` ownership that arctic's columns and
numba's buffers already depend on.

**Break the two interpreter-made cycles as special cases.** Rejected as insufficient, and one of
them is not fixable this way. The nested-`def` cycle looks like it has an exact fix — make the
scope's reference to its own function the strong one and the closure's back-reference weak — but
that breaks `return inner;`, where the returned function is the *only* thing keeping the frame
alive. That pattern works correctly today (measured: the returned-lambda row is flat) and must keep
working. And neither special case touches row 3, the general one.

**Add a cycle collector alongside reference counting.** Chosen. This is what every mature
refcounted runtime does — CPython since 2.0, PHP since 5.3 — for the same reason: refcounting
handles the overwhelming majority promptly and cheaply, and a periodic collector handles only what
refcounting provably cannot.

## 4. The design

### 4.1 What is tracked

Four node kinds can be part of a cycle, because each holds `Value`s and each has an identity shared
by `shared_ptr`:

| kind | holds | cycle it makes possible |
|---|---|---|
| `ClassInstance` | `properties` | `$a.p = $b; $b.p = $a` |
| `ObjectMap` (dict) | entries | `$d["self"] = $d` |
| `Environment` | `variables`, `parent` | a closure and the scope that names it |
| `BantuFunction` | `closure` | the edge that closes the two above |

Lists are traversed but never tracked, because a list has no shared identity — it is copied, not
referenced. Native handles are traversed not at all: they hold no `Value`.

### 4.2 Registration costs nothing

Each tracked type embeds a `bantu_gc::Head` — two list pointers, an owner pointer, a kind byte, and
two scratch fields — and links itself into one global intrusive doubly-linked list in its
constructor, unlinking in its destructor. That is four pointer writes on create and four on
destroy: **no allocation, no atomic operation, no hash insert.** This matters because `Environment`
is on the hottest path in the interpreter; the measurements above show roughly six scopes created
per interpreted loop iteration.

The refcount itself is read through `std::enable_shared_from_this::weak_from_this()`, which costs
nothing to keep and does not inflate the count.

### 4.3 The algorithm

The collector never computes a root set by scanning the C++ stack. It infers roots from the
reference counts, which is the scan phase of Bacon–Rajan trial deletion:

```
reset      every node: internal = 0, marked = false
count      every node: for each outgoing edge to a tracked node t: t->internal++
roots      every node where use_count() > internal  →  mark, push to worklist
mark       drain the worklist, marking everything reachable
sweep      every unmarked node is garbage in a cycle
```

A node's `use_count()` counts *all* strong references. `internal` counts only those we can see
inside other tracked containers. Anything left over is a reference from somewhere we did not trace:
the evaluator's own members, a C++ stack temporary, an argument vector. Those are exactly the
references that make a node a root.

**Every source of imprecision is conservative in the safe direction.** A reference we fail to see
makes `use_count() > internal`, which makes the node a root, which retains it. The failure mode of a
bug in the edge walker is a cycle that survives one more collection — never a live object being
freed. That property is why this design is acceptable in a production language and a tracing
collector is not.

Two invariants make the count exact rather than merely safe:

1. **Each edge is enumerated exactly once.** When the walker meets a `Value` that points at a
   tracked node it records one edge and stops; it does not recurse. Recursion happens by iterating
   the registry instead. Without this rule a `BantuFunction` referenced from two places would have
   its closure edge counted twice, `internal` could exceed `use_count()`, and a reachable node could
   be swept. This is the one rule in the whole design whose violation is unsafe, so it is asserted
   in the debug build.
2. **A node not owned by a `shared_ptr` is never swept.** `weak_from_this()` returns empty for a
   stack-allocated object, giving `use_count() == 0`; such nodes are skipped entirely.

### 4.4 The sweep frees nothing itself

The collector does not call `delete`. For each unmarked node it takes a strong reference, pins the
whole victim set, clears each victim's container (`properties`, `variables`, the dict's entries, the
function's closure), then drops the pins. Clearing removes the internal references; **reference
counting then does all the freeing**, in its usual order, running any future destructor logic
normally.

Pinning before clearing is required: clearing one victim can free another, which unlinks a `Head`
from the list being walked.

So a logic error in the collector nulls some fields of an object that should have lived. That is a
bug, and a bad one — but it is not memory corruption and not undefined behaviour, and it is
diagnosable from Bantu. This is a deliberate trade against a design that frees memory directly.

### 4.5 When it runs

A counter tracks tracked-node creations minus destructions. When the surplus crosses a threshold the
collector runs **at the next statement boundary** — never in the middle of evaluating an expression.

Statement boundaries are chosen not because mid-expression would be unsafe by the reachability
argument (it would not be: a temporary inflates the count and roots the node) but because parts of
the evaluator hold a raw `Value*` into a container across a few statements — `resolveLValue` returns
one — and a statement boundary is the point where provably none is in flight. It is also cheaper: one
integer compare per statement rather than per allocation.

The threshold starts at 10,000 and, after each collection, is reset to twice the surviving tracked
population (floor 10,000). A program holding a large live set therefore does not pay an O(live) scan
repeatedly; collection cost is amortised constant per allocated object.

### 4.6 The Bantu-facing surface

Modelled on the one every Python programmer already knows, because the rule is that this has to be
adoptable without a manual:

| call | does |
|---|---|
| `gc_collect()` | run a collection now; returns the number of objects freed |
| `gc_stats()` | `{"live", "collections", "freed", "threshold", "enabled"}` |
| `gc_enable($on)` | turn automatic collection on or off; returns the previous state |

`BANTU_GC=0` in the environment disables automatic collection for the process. Nothing else changes:
`gc_collect()` still works when called explicitly, so the switch is a diagnostic tool and a latency
escape hatch, not a way to lose the fix.

`gc_stats()["live"]` is also the measurement that makes the leak gates in `tests/` assertions about
a number rather than inspections of RSS — RSS was too coarse to show the dict cycle at all.

### 4.7 Threading

Bantu code runs on one thread at a time. `sua`'s suspendable handlers use real threads, but
`bantu_co::Scheduler` hands a baton between them under a mutex and condition variable: `spawn()`
returns only once the task has parked or finished, so no two threads ever evaluate Bantu
concurrently, and the hand-off establishes the happens-before edge that makes a plain global
registry correct. The registry is therefore a plain global and takes no lock — the same reasoning,
and the same invariant, that the event loop already relies on.

(`SuaServer` in `server.hpp` does spawn a thread per connection, but nothing constructs it; it is
unreferenced legacy and runs no interpreter code.)

## 5. The six questions

**Scalable?** Yes. Registration is O(1) with no allocation. Collection is O(live tracked nodes) and
the threshold scales with the survivors, so the amortised cost per allocated object is constant.
The gate is the interpreter benchmark within ±2%.

**Maintainable, long-term?** Yes. No new dependency; ~500 lines in two headers, split so that
`gc.hpp` knows nothing about Bantu's types and `gc_collect.hpp` is the only file that walks them.
The one rule a future edit could break — enumerate each edge once — is asserted rather than trusted.

**Easy, and the Bantu way?** Yes, and mostly by being invisible: a Bantu program that never looked
at memory behaves identically and simply stops leaking. The three builtins mirror Python's `gc`
module.

**Documentable and testable?** Yes, and better than by RSS. `gc_stats()["live"]` makes every leak
gate an exact assertion: build a cycle N times, collect, assert live returns to baseline.

**Efficient?** The thing being fixed is unbounded memory growth, so the honest comparison is against
a process that dies. Against a program with no cycles, the cost is the intrusive link/unlink and one
compare per statement, and the gate is ±2% on the existing benchmark.

**Secure?** This is the security fix. Unbounded memory growth driven by request traffic is a
denial-of-service: a `sua` handler that stores a callback on an object, or uses a nested helper,
grows until the worker is OOM-killed, with no error and no log line. That is reachable by anyone who
can send requests. The collector's own failure mode is deliberately bounded to nulled fields rather
than freed memory (§4.4), and the conservative direction of every approximation (§4.3) means the
collector cannot free something reachable.

## 6. What this does not do

- **It does not make collection prompt.** A cycle is freed at the next collection, not at the
  statement that dropped it. Non-cyclic garbage is still freed immediately by refcounting, which is
  the overwhelming majority.
- **It does not collect native handles.** An arctic column or numba buffer held only by a cycle is
  freed when the cycle is, since the cycle's clearing drops the handle. But a cycle *through* native
  memory — one native object holding a `Value` holding it back — is not traced. No native type does
  this today, and none should.
- **It does not add weak references.** They remain worth having and are a separate design.
- **It does not trace a native closure's captures.** A `std::function`'s captures cannot be
  enumerated, so a cycle running through one is *retained* rather than freed — the safe direction,
  because the capture still shows in `use_count()` and roots the node. The only place in the tree
  that ever built such a cycle was sua's `$res` object, whose chaining methods owned the map that
  owned them; they capture it weakly now. Any future native closure capturing a `Value` must do the
  same.

## 8. Results

Measured on the same machine, same build flags, against the build immediately before the collector.

**Memory — the defect itself.** 400,000 iterations of each shape, peak RSS:

| shape | before | after |
|---|---|---|
| create and drop (no cycle) | 5.1 MB | 5.2 MB |
| `$a.self = $a` | 246 MB | 18.9 MB |
| `$a.p = $b; $b.p = $a` | 502 MB | 17.9 MB |
| `$a.cb = $a.m` | 558 MB | 17.0 MB |
| nested `def` | 34 MB (20k calls) | 19.8 MB (400k calls) |
| `$d["self"] = $d` | ~260 MB (extrapolated) | 16.7 MB |

**Bounded, not merely smaller.** Ten times the run length — 400,000 to 4,000,000 mutual pairs —
costs 17.3 MB against 20.7 MB. Every shape in `tests/gc_stress.sh` passes the same ratio gate.

**The gate measures the collector and not something else.** The identical 200,000-iteration run with
`BANTU_GC=0` uses **135 MB** against **17.6 MB** with the collector on.

**Cost when there are no cycles.** One million acyclic objects trigger **zero** collections. The
interpreter benchmark (3M arithmetic iterations, 600k function calls, 400k object creations),
best-of-5 in both A/B orderings: **7,898 ms before, 7,830 ms after — −0.85%**, inside the ±2% gate.

That last number only holds because the registry is a namespace-scope `inline` variable rather than a
function-local `static`. The first implementation used a function-local static and measured
**+3.85%**: every call to `link`, `unlink` and `due` paid a thread-safe-initialisation guard, and
those run several times per interpreted loop iteration. The guard was the entire cost.

**Correctness.** `tests/lang_gc_test.b` — 55 assertions, covering each cycle shape, the reachable
cycles that must survive, returned closures, and the automatic trigger. `tests/gc_stress.sh` — 19
checks. Full regression: 37 `.b` suites and 13 `.sh` suites green. **ASan + UBSan clean across all 37
`.b` suites**, which is the tier that matters for a change to object ownership.

**`sort` with a key.** 100,000 rows, list-construction cost subtracted from both: **13,331 ms with a
comparator against 764 ms with a key — 17× faster**, identical result.

## 7. Related change: `sort` with a key function

Not a lifetime matter, but the same complaint — the language forcing users into the slow version.

`sort($list, $cmp)` calls a Bantu comparator O(n log n) times. Each interpreted call costs 1–3 µs, so
100,000 rows is about 1.7 million calls. The scalable form is to compute a key once per element —
O(n) calls — and sort natively on the keys. Python settled this the same way, and for the same
reason: `key=` replaced `cmp=` in Python 3.

`sort` gains an options dict alongside the existing spellings:

```bantu
sort($xs)                                  // ascending
sort($xs, "desc")                          // descending
sort($xs, def($a, $b) { return $a - $b; }) // comparator, still supported
sort($xs, {"key": def($r) { return $r["age"]; }, "desc": true})
```

The key path decorates, sorts natively with `std::stable_sort`, and undecorates. Keys must be all
numbers or all strings — a mixed key set raises and names both types, exactly as the no-comparator
path already does. NaN keys sort last, in both directions, for the reason the existing code
documents: `a < b` with NaN present is not a strict weak ordering, and `std::sort` given one reads
past its range.
