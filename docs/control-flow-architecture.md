# Control flow without exceptions — `return`, `break` and `continue`

**Status:** implemented. **Tests:** `tests/lang_control_flow_test.b`, and every existing suite.

## The defect

`return`, `break` and `continue` were C++ exceptions: `evalReturn` threw `ReturnSignal`, and every
function call, loop and module caught it. A throw is not a jump. The runtime allocates the exception,
walks the stack with the unwinder, and — on macOS — asks dyld which image each frame belongs to. It
costs the same whether the `return` is one frame from its call or twenty.

Measured on the `bantu` that `build-mac.sh` produces (Intel Core i7-9750H, macOS), 100,000 of each:

| | before | per operation |
|---|---|---|
| a call to `def f($i) { return $i; }` | 891 ms | 8.9 µs |
| a call to `def f($i) { $i; }` (implicit result) | 117 ms | 1.2 µs |
| a loop iteration that runs `continue` | 887 ms | 8.9 µs |
| a bare loop iteration | 32 ms | 0.3 µs |

So **a `return` cost six times the rest of the call**, and nearly every Bantu function returns. It was
found by bplot's 4000×3000 stress figure: a heatmap drawn through small helper functions took 15.8 s,
and the profile was dominated by the unwinder, not by drawing.

## The design: a pending signal, checked between statements

The evaluator holds one field, `flow_`, that is `Normal` almost always, plus the value a `return`
carries:

```cpp
enum class Flow : uint8_t { Normal, Break, Continue, Return };
Flow  flow_ = Flow::Normal;
Value flowValue_;
```

- `return <expr>` evaluates the expression, stores it in `flowValue_`, sets `flow_ = Return`, and
  returns normally. `break` and `continue` set `flow_`.
- **Every statement list runs through one helper**, which stops at the first statement that leaves
  `flow_` set. Blocks of `if`, `switch`, `try` and `catch`, loop bodies, function bodies, method
  bodies, modules and the program itself all use it — so there is exactly one loop that has to be
  right.
- **Loops consume `Break` and `Continue`.** `Continue` still runs a `for` loop's update, as before.
  A loop leaves `Return` set and exits, so the function around it sees it.
- **Calls consume `Return`**, taking `flowValue_` as the result. A function that finishes without
  `return` still returns the value of its last statement, exactly as before.
- **The program and each file** consume a top-level `Return` (it ends the program, as before) and
  turn a stray `Break` or `Continue` into the same error as before.

Nothing that is evaluated as an *expression* can set `flow_`: `return`, `break` and `continue` are
statements, and any call inside an expression consumes its own `Return` before the expression goes
on. So the only code that can observe a pending signal is a statement list, and every statement list
goes through the helper.

**One behaviour is kept deliberately, through the old path.** A `break` or `continue` inside a
function but outside any loop in it used to escape the function and act on the *caller's* loop. That
is odd, but it is behaviour, and this change is additive. A call that ends with `Break` or `Continue`
pending therefore rethrows the old `BreakSignal`/`ContinueSignal`, and loops still catch them. That
path costs what it always did; it is simply no longer the common one.

**Errors stay exceptions.** `throw`, runtime errors and `try`/`catch` are unchanged: they are
exceptional, and unwinding is the right tool for them.

## Rejected alternatives

- **Catch closer to the throw** (a try block per statement): catching sooner does not make throwing
  cheaper; the cost is in the throw.
- **Special-case a `return` in tail position**: faster for one shape of function and still slow for a
  `return` inside an `if` — the common guard clause — and two mechanisms for one statement.
- **`setjmp`/`longjmp`**: skips destructors, so every `shared_ptr` environment on the way would leak.
- **A bytecode compiler**: the real long-term answer to interpreter speed, and a project of its own.
  This change is what a bytecode VM would do too — a jump, not an unwind — at the scale of one field.

## The six questions

- **Scalable?** One byte compared after each statement, against a throw per `return`.
- **Maintainable?** One helper runs every statement list; one field holds the state.
- **Easy, and the Bantu way?** No syntax and no semantics change.
- **Documentable and testable?** This document, and a suite that runs `return`, `break` and `continue`
  through every construct that can contain them, beside the whole existing regression.
- **Efficient?** See the numbers below.
- **Secure?** No new input surface. The failure mode to guard against is a signal left pending, which
  would stop later statements silently — so the tests check the statements *after* each construct run.

## Why the flag is safe with sua's threads

sua runs a suspendable handler on its own thread, but under a **baton**: one thread executes Bantu at a
time, and a handler hands the baton over only inside a native call (`bantuOffBaton`). `flow_` is set
only while statement lists are unwinding, and no call is made while they unwind, so at every handover
it is `Normal`. It needs no per-thread copy and no lock.

## Results

Same machine, same build flags, three runs each, before and after:

| | before | after | |
|---|---|---|---|
| 100,000 calls that `return` | 891 ms | **145 ms** | 6.1× |
| 100,000 loop iterations that `continue` | 887 ms | **71 ms** | 12.5× |
| `benchmarks/bench.b` | 35.7 s | **6.7 s** | 5.4× |
| `benchmarks/hotpath.b` | 13.0 s | **8.9 s** | 1.5× |
| bplot's 1100×800 dashboard, SVG | 78 ms | **37 ms** | 2.1× |

A call that returns now costs what a call that does not return costs — the return itself is free.
Semantics are unchanged: `tests/lang_control_flow_test.b` passes 35 of 35 on the binary before the
change and the binary after it, and so does every existing suite, server test, sample and package test.
