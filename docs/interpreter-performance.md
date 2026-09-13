# Interpreter performance — what actually costs, and the fix

Bantu is a tree-walking interpreter. Its speed has been described in this repository for a while as
"~1 µs per interpreted operation", with `Value`'s 192-byte size named as the likely cause. **That
attribution was wrong**, and this document replaces it with a profile.

> Measurements: **Intel Core i7-9750H @ 2.60 GHz, macOS 15.7.9, Apple clang 17**, `bantu` built by
> `build-mac.sh` (`-O2`), commit `7cac1d4`. Profiles taken with `sample(1)` over six seconds of a
> 20-million-iteration arithmetic loop.

---

## 1. The measurement

```bantu
$i = 0; $s = 0;
while ($i < 20000000) { $s = $s + $i * 2 - 1; $i = $i + 1; }
```

Self-time attribution, main thread, 4,716 attributed samples:

| | samples | share |
|---|---|---|
| **`dynamic_cast` machinery** | **3,753** | **79.6 %** |
| everything else | 963 | 20.4 % |

Broken out, the `dynamic_cast` half:

| symbol | samples |
|---|---|
| `dyn_cast_slow` | 794 |
| `__dynamic_cast` | 732 |
| `dyn_cast_try_downcast` | 545 |
| `__si_class_type_info::search_below_dst` | 436 |
| `__si_class_type_info::search_above_dst` | 382 |
| `__class_type_info::search_below_dst` | 336 |
| `__class_type_info::search_above_dst` | 234 |
| `dyn_cast_get_derived_info` | 164 |
| `is_equal` | 67 |
| `dyn_cast_to_derived` | 63 |

And the other half, every entry above 0.9 %:

| symbol | samples | share |
|---|---|---|
| hash-map `find` internals | 197 | 4.2 % |
| `Evaluator::evalNode` itself | 173 | 3.7 % |
| `bzero` — zeroing a 192-byte `Value` | 87 | 1.8 % |
| `evalBinaryOp` | 75 | 1.6 % |
| `murmur2/cityhash` — hashing a variable name | 73 | 1.5 % |
| `evalWhile` | 69 | 1.5 % |
| `Value::~Value` | 53 | 1.1 % |
| `Value::Value(const Value&)` | 44 | 0.9 % |

**Every cost that has previously been blamed for Bantu's speed — the 192-byte `Value`, its string
and vector and `std::function` members, the allocator, the environment's string hashing — adds up to
20 %.** The dispatcher is four times all of them combined.

### Why it is so expensive

`Evaluator::evalNode` dispatches by trying each node type in turn:

```cpp
if (auto n = dynamic_cast<NumberNode*>(node.get()))   return evalNumber(n);
if (auto n = dynamic_cast<StringNode*>(node.get()))   return evalString(n);
…                                                     // 38 of these
if (auto n = dynamic_cast<IncludeNode*>(node.get()))  return evalInclude(n);
```

Three things compound:

1. **`dynamic_cast` is not a comparison, it is a search.** It calls into the runtime
   (`__dynamic_cast` in `libc++abi`), which walks the class hierarchy comparing `type_info` records.
   The profile shows `search_above_dst` and `search_below_dst` doing real work on every call — a
   *failing* cast is the expensive case, and 38-way sequential dispatch is mostly failing casts.
2. **Position is cost.** The chain is ordered by when each node type was added, not by frequency.
   `VariableNode` is 7th, `AssignNode` 9th, `BinaryOpNode` 12th, `CallNode` **20th**,
   `IndexAccessNode` **22nd**. Every function call in every Bantu program pays for nineteen failed
   searches first.
3. **It runs per node, per visit.** The loop above evaluates ~10 nodes per iteration; at 20 million
   iterations that is 200 million dispatches, each averaging ~19 failed casts.

There are 39 node types and 57 `dynamic_cast`s in `evaluator.hpp` — 38 in the dispatch chain, the
rest in `resolveLValue`, `borrowLValue`, `evalCall` and the assignment paths, several of which are
themselves hot.

---

## 2. The fix: a type tag and a switch

**Give `ASTNode` a `NodeKind` tag set by its own constructor, and dispatch with `switch`.**

```cpp
enum class NodeKind : uint8_t { Number, String, Bool, Null, List, Dict, Variable, … , Include };

struct ASTNode {
    NodeKind kind;
    int line, col;
    virtual ~ASTNode() = default;
    ASTNode(NodeKind k, int l = 0, int c = 0) : kind(k), line(l), col(c) {}
};

// The tag comes from the base-class name, so the struct's declaration line is
// the single place it is written and there is nothing to keep in sync.
template <NodeKind K>
struct ASTNodeK : ASTNode {
    static constexpr NodeKind kKind = K;
    ASTNodeK(int l = 0, int c = 0) : ASTNode(K, l, c) {}
};

struct NumberNode : ASTNodeK<NodeKind::Number> {
    double value;
    NumberNode(double v, int l, int c) : ASTNodeK(l, c), value(v) {}
};
```

```cpp
#define NODE(K, T) nodeExact<NodeKind::K, T>(node.get())
switch (node->kind) {
    case NodeKind::Number:   return evalNumber(NODE(Number, NumberNode));
    case NodeKind::Variable: return evalVariable(NODE(Variable, VariableNode));
    …
}
```

One load of a byte already in the same cache line as the vtable pointer, and one jump through a
table the compiler builds. **O(1), and identical for the 39th node type and the 1st.**

### Why this and not the alternatives

**Virtual dispatch / the Visitor pattern** — `virtual Value accept(Visitor&)` on every node. Also
O(1), also correct, and type-safe without a tag to maintain. Rejected for two reasons specific to
this codebase. First, it requires 39 `accept` overrides in `ast.hpp` plus a 39-method visitor
interface, and every future node type must be added in three places instead of two. Second, and
decisively, **it would move evaluation out of `evaluator.hpp`** or force every `evalX` through an
indirection layer; the whole evaluator is deliberately one file today, and that is worth more to
long-term maintenance than the few nanoseconds between a jump table and a vtable. The switch keeps
every `evalX(XNode*)` function byte-for-byte as it is.

**A bytecode compiler and VM.** The real long-term answer for another order of magnitude, and out of
scope here: it is a rewrite of evaluation, not a change to it, and it would have to reproduce the
semantics of 39 node types before anything ran at all. This fix is a prerequisite for it rather
than a competitor — the `NodeKind` tag is exactly what a compiler would switch on to emit opcodes.
Recorded as deferred, with its own justification required.

**Reordering the existing chain by frequency.** A five-line change that would help — putting
`Variable`, `BinaryOp` and `Call` first — and it is a patch, not a fix: still O(n) in the number of
node types, still slow for whatever is last, and it silently degrades every time someone adds a node
type. Rejected as exactly the "easiest solution" this project does not take.

**Caching the resolved type on the node.** Adds a field and a branch and still needs the first cast.
Strictly worse than a tag set at construction, which is free.

### Making a wrong tag impossible rather than unlikely

A tag plus `static_cast` is fast precisely because it skips the runtime check. If a tag ever
disagreed with its type, the `static_cast` would be undefined behaviour — a silent, memory-corrupting
failure mode, and one worth engineering against rather than hoping about.

**The two ways to express a mismatch are both compile errors**, which is better than the test-based
plan this document originally proposed:

1. **A node cannot mis-tag itself.** The tag is not written in the constructor where it could be
   copy-pasted wrong; it comes from the base class, `ASTNodeK<NodeKind::Number>`, which appears on
   the struct's own declaration line and passes `K` up itself.
2. **A case cannot cast to the wrong type.** `nodeExact<K, T>` takes the kind from the switch's own
   case label and `static_assert`s `T::kKind == K`. Labelling `case NodeKind::Bool` with a cast to
   `StringNode` does not compile — verified by deliberately breaking one and watching it fail:
   *"evalNode: this case label does not match the node type it casts to"*.

Two further defences cover the parts a compiler cannot see — `nodeIf<T>` at the twenty call sites
outside the dispatcher, and any node type added later:

3. **`-DBANTU_CHECK_NODEKIND` makes every cast a checked one**, aborting with both type names on a
   mismatch. The **whole test suite runs that way in CI**, so this holds for node types nobody has
   written yet.
4. **The `dynamic_cast` chain is kept as `default:`.** A node whose kind is not in the switch still
   evaluates correctly, just slowly. The change cannot break a node type; at worst it fails to
   accelerate one.

---

## 3. The second fix: `$s = $s + $part` is still O(n²)

`join()` gave Bantu a linear way to build a string, but it did not make the *quadratic* way stop
being quadratic. Existing code, and anyone who reaches for the obvious idiom, still pays
**6,752 ms for 40,000 appends** because every `+` copies the whole accumulated string.

**The fix is the one CPython uses** (`unicode_concatenate` in `ceval.c`, since 2.4, and the reason
the "string concatenation is quadratic in Python" folklore is false in practice): when the result of
`x + y` is being assigned straight back to `x`, the old value of `x` is dead the instant the
assignment completes, so it can be **appended to in place instead of copied**.

In Bantu this is a peephole in `evalAssign`, and the parser has already arranged for it to cover
both spellings — `$s += $x` desugars to exactly `AssignNode(s, BinaryOp(PLUS, Variable(s), x))`:

```
assignment to X, whose value expression is a '+' chain
whose left-most leaf is Variable(X), and X currently holds a STRING
   → evaluate the rest of the chain left to right, appending each piece
     into X's own buffer; never materialise the intermediate result
```

Walking the left spine means `$s = $s + $a + $b + $c` is handled too, not just the two-operand case:
`+` is left-associative, so the leftmost leaf of the chain is the accumulator.

**Correctness, case by case:**

| case | why it is safe |
|---|---|
| `$s = $s + $s` | the right-hand side is evaluated into a temporary *before* the append begins |
| `$t = $s; $s = $s + "x"` | `Value` holds `std::string` **by value**, so `$t` already owns a separate buffer |
| `$s = $s + f()` where `f` mutates `$s` | the right operand is fully evaluated first, then the borrow is taken |
| `$s = $x + $s` | left-most leaf is not the target — the peephole does not fire, ordinary path |
| `$s` is a number | the peephole checks the live type and declines; numeric `+` is untouched |
| `$s` is `const` | assignment raises as it always did, before any append |

### Fields and elements, not just locals

`$fig.parts += $x` and `$a[$i] += $x` are different node types (`DictAssignNode`, `IndexAssignNode`)
and were not covered by the first version of this. That left the language 64× apart on two spellings
of the same operation — 40,000 appends into a local took 22 ms and into a dict field 1,413 ms — which
is not a defensible thing for a language to do, especially since accumulating into a field is what
object-oriented code does. Both now take the same path, at 18 ms.

They are held to a **stricter** rule than a local, and the difference is the interesting part. For
`$s = $s + f()`, no `f` can touch `$s`: `Environment::assign` stops at the nearest function boundary,
so a callee assigning to `$s` creates its own local. A *field* has no such protection — dicts, lists
and class instances are reachable through references, so a callee holding the same object can replace
the very string being appended to. For those targets the bar is therefore absolute: **no operand may
run any code at all** (`isPureExpr` — literals, variables, arithmetic and field reads, nothing else).
An impure operand declines the fast path and takes the ordinary one, evaluated exactly once, which a
test asserts by counting the calls.

Identity is decided syntactically (`sameLValue`): `$o.a.b`, `$a[3]` and `$a[$i]` qualify, a computed
index like `$a[f()]` does not, because it may not name the same element twice.

**Why not a rope, or a copy-on-write string?** A rope (Boehm–Atkinson–Plass) makes concatenation
O(1) but makes indexing, comparison and every C-string boundary slower and more complicated, and
there are 94 sites in the tree that touch `stringVal` directly. Copy-on-write with an append-when-
unique check is closer, but it changes `Value`'s representation — the one thing this document's
profile says is **not** where the time goes — and would still not help, because evaluating the left
operand hands back a copy that makes the buffer non-unique anyway. The assignment-level peephole
needs no representation change, touches one function, and fixes the idiom people actually write.

---

## 4. What this deliberately does not fix

The profile names the next two costs precisely, and neither is addressed here, because each deserves
its own measurement after the dispatcher stops dominating:

- **Variable lookup by string hash — 4.2 % + 1.5 % + `Environment::get` 0.4 %.** Every `$x` hashes a
  `std::string` and walks a scope chain of `unordered_map`s. The real fix is lexical addressing —
  resolving names to (depth, slot) at parse time and indexing a vector. That is a change to scoping,
  with closures and `sua`'s per-request environments to get right, and it is a phase of its own.
  After the dispatcher fix it becomes proportionally ~30 % of what remains, which is when it will be
  worth doing and not before.
- **`Value` is 192 bytes** — `bzero` 1.8 %, copy 0.9 %, destruct 1.1 %, plus allocator traffic. A
  tagged union or NaN-boxed representation would shrink it to 16, but it touches 94 `.stringVal` and
  67 `.listVal` sites and every native library in the tree. **At ~5 % it is not worth that risk
  today**, and this document exists partly to record that the intuition naming it as the bottleneck
  was measured and found wrong.

Both stay on the roadmap with the same rule numba's Phase 7 has: only on a concrete measured wall.

---

## 5. The six questions

**Scalable?** Yes, in the dimension that was failing. Dispatch stops being O(number of node types),
so the 40th node type costs what the 1st does — the current design gets *slower for every program*
each time the language grows. String building stops being O(n²).

**Maintainable, long-term?** Yes, and better than what it replaces. Adding a node type is: write the
struct, add an enum member, add a case — and forgetting the case is caught by the compiler's
`-Wswitch` warning on an unhandled enumerator, where forgetting a line in the old chain was silent.
No new dependency, no new file, and every `evalX` function unchanged.

**Easy, and the Bantu way?** It is invisible to users: no syntax, no builtin, no behaviour change.
That is the right shape for a performance fix.

**Documentable and testable?** Yes: a checking build that aborts on any tag mismatch, a test that
verifies all 39 tags against their real types, the full suite green under both builds, and an
order-controlled before/after benchmark.

**Efficient?** The profile bounds it: with dispatch at 79.6 %, removing nearly all of it has a
**ceiling of 4.9×**. Real gains will be lower — the switch is not free, and the long tail grows in
proportion. The phase is gated on the measurement, not on this estimate.

**Secure?** The one new risk is the unchecked `static_cast`, and §2 answers it with four layers:
constructor-set tags, a checking build that the whole suite must pass, a test over every node type,
and the old chain retained as `default:` so an unmapped node degrades to slow rather than to
undefined. The string peephole reduces memory traffic rather than adding any, and mutates a buffer
only after proving it is the assignment target's own and that every operand has already been
evaluated.

---

## 6. Results

Order-controlled: best-of-two per ordering, then best across both orderings, because this harness
was previously shown to carry a ~2.7 % position bias — larger than the gate it is measured against.

### `benchmarks/hotpath.b`

| | before | after | |
|---|---|---|---|
| 1M arithmetic while loop | 2,574 ms | **448 ms** | **5.7×** |
| 1M comparison loop | 2,436 ms | **539 ms** | **4.5×** |
| 500k unary negation | 1,171 ms | **200 ms** | **5.9×** |
| `fib(24)` recursive | 1,910 ms | **1,535 ms** | 1.24× |
| 200k list index read | 575 ms | **101 ms** | **5.7×** |
| 100k list index write | 213 ms | **41 ms** | **5.2×** |
| 50k dict set | 178 ms | **45 ms** | **4.0×** |
| 100k string concat (short) | 181 ms | **35 ms** | **5.2×** |

`fib(24)` is the honest outlier and the informative one: a recursive call is dominated by building
an `Environment` (a `shared_ptr` allocation) and copying arguments, not by dispatch, so it gains
1.24× where straight-line code gains 5–6×. That is the shape of the next phase, not a disappointment
in this one.

### `benchmarks/bench.b`, per iteration

| | before | after | |
|---|---|---|---|
| 1M-iteration arithmetic loop | 2,119 ms | **383 ms** | **5.5×** |
| `fib(28)` recursive | 13,353 ms | **10,741 ms** | 1.24× |
| list push 100k | 226 ms | **59.8 ms** | 3.8× |
| dict set 100k | 274 ms | **103 ms** | 2.7× |
| string concat 10k | 26.7 ms | **3.0 ms** | 8.9× |

### String building, which was the other half of the job

| appends | before | after | |
|---|---|---|---|
| 40,000 | 6,752 ms | **23 ms** | **294×** |
| 100,000 | ~42 s (extrapolated) | **53 ms** | |
| 200,000 | — | **112 ms** | |

Linear, where it was quadratic: 50k → 100k → 200k costs 26 → 44 → 112 ms.

### The profile afterwards

`dynamic_cast` **79.6 % → 0.0 %**. It does not appear in the profile at all. What replaced it at the
top is exactly what §4 predicted:

| | share |
|---|---|
| hash-map `find` (variable lookup) | 19.8 % |
| `murmur2/cityhash` (hashing variable names) | 8.5 % |
| allocator (`tiny_malloc` + `free_tiny`) | 12.9 % |
| `Value` copy / destroy / assign | 14.3 % |
| `evalWhile` | 7.2 % |
| `evalNode` (the switch itself) | 7.2 % |
| `evalBinaryOp` | 6.7 % |
| the append peephole's per-assignment test | 1.9 % |

**Variable lookup is now ~28 % and `Value` plus the allocator ~27 %** — the two items §4 named as
deferred, in the order it named them. They are the next phase, when there is a reason for one.

### What it cost to get the peephole to ~0 %

Two rounds of measurement, both worth recording:

- **The first version reserved `size() + extra` on every append**, which forces an exact-size
  reallocation and a full copy every time. The optimisation *was* the quadratic term: 40,000 appends
  still took 893 ms and 200,000 took 58 s. `std::string` already grows geometrically; the fix is to
  reserve only past what it would do on its own.
- **The second version allocated a `std::vector` per assignment** to hold the `+` chain's operands,
  which cost 25 % on assignment-heavy loops — paid by every `$i = $i + 1` in the language, not just
  by string building. The memo on the node now carries the *piece count*, so the single-piece case
  (nearly all of them) runs with no container at all, and a non-string target costs one scope lookup
  that replaces the one `Environment::assign` would have done anyway.

### Verification

- Full suite green: 34 `.b` suites, 12 `.sh` suites, and the `const_bad` negative fixture.
- **The entire suite green again under `-DBANTU_CHECK_NODEKIND`**, where every dispatch cast is
  verified against RTTI — now a CI step, so it holds for every node type added later.
- A deliberately mislabelled case was confirmed to fail at **compile** time, via `nodeExact`'s
  `static_assert`.
- ASan + UBSan + tag checking together, across the language, numba, arctic, crypto and webpush
  suites: no report.
- `tests/lang_perf_test.b`, 59 assertions, most of them about the semantics the peephole must not
  have changed — scoping (an assignment inside a function still creates a local rather than mutating
  a global), aliasing (`$t = $s` keeps its own copy, `$s = $s + $s` reads the old value), `const`
  still raising, and numbers, lists, dicts and handles still taking the ordinary operator path.

### A defect found while doing this

**`benchmarks/bench.b` called `sua.clock()`, which does not exist**, so it raised on its first
benchmark. `benchmarks/run.sh` runs it, which means the benchmark suite has been failing, and the
numbers published in `benchmarks/results.md` and `benchmarks/README.md` came from a script that no
longer runs. Fixed to the global `clock()`, which already returns milliseconds, so the `* 1000.0`
scaling it applied was wrong too. Both tables in this section come from the repaired script, with
the *same* script run against both binaries.

---

## Sources

- Itanium C++ ABI §2.9.5 and §15.3, *RTTI Layout* and *`__dynamic_cast`* — why a failing cast is a
  hierarchy walk rather than a comparison. <https://itanium-cxx-abi.github.io/cxx-abi/abi.html>
- LLVM `libcxxabi/src/private_typeinfo.cpp` — `search_above_dst` / `search_below_dst`, the functions
  the profile names.
- CPython, `Python/ceval.c`, `unicode_concatenate` and `PyUnicode_Append` — in-place append when the
  target is about to be overwritten; the design taken here.
- Lua 5.4 `lvm.c`, CPython `ceval.c`, Ruby `vm_insnhelper.c` — production interpreters dispatching on
  an integer tag, not on RTTI.
- Boehm, Atkinson and Plass, *Ropes: an Alternative to Strings*, Software — Practice and Experience
  25(12), 1995 — the alternative considered and rejected in §3.
- Nystrom, *Crafting Interpreters*, ch. 14 — "A Virtual Machine"; the bytecode step recorded as
  deferred in §2.
