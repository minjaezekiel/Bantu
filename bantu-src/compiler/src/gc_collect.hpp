#pragma once
// ════════════════════════════════════════════════════════════════════════════
//  gc_collect.hpp — the half of the cycle collector that knows Bantu's types.
//
//  gc.hpp holds the registry and knows nothing about Value, because it is
//  included by the very headers that define the tracked types. This file sits
//  above all of them and is the ONLY place that walks a Bantu object graph.
//
//  THE ALGORITHM (the scan phase of Bacon-Rajan trial deletion)
//
//      reset   every node: internal = 0, marked = false
//      count   every node: for each outgoing edge to a tracked node t, t->internal++
//      roots   every node with use_count() > internal  ->  mark and enqueue
//      mark    drain the queue, marking everything reachable
//      sweep   every unmarked node is garbage inside a cycle
//
//  use_count() counts ALL strong references. `internal` counts only those we
//  can see inside other tracked containers. Whatever is left over is a
//  reference from somewhere we did not trace -- the evaluator's own members, a
//  C++ stack temporary, an argument vector -- and those are exactly the
//  references that make a node a root. So the collector never needs a root
//  set, which is the whole reason this design is possible in a tree-walking
//  evaluator where most live references are C++ locals.
//
//  WHY IT IS SAFE
//
//  Every source of imprecision points the same way. A reference we FAIL to see
//  leaves use_count() > internal, which makes the node a root, which retains
//  it. A bug in the edge walker costs a cycle one more collection to notice --
//  it can never free something reachable.
//
//  Two invariants make the count exact rather than merely safe:
//
//    1. EACH EDGE IS ENUMERATED EXACTLY ONCE. When the walker meets a Value
//       pointing at a tracked node it records one edge and STOPS -- it does not
//       recurse through it. Recursion happens by iterating the registry
//       instead. Break this and a shared node's edges get counted twice,
//       internal can exceed use_count, and a reachable node can be swept. This
//       is the one rule here whose violation is unsafe, so the debug build
//       asserts it.
//    2. A NODE NOT OWNED BY A shared_ptr IS NEVER SWEPT. weak_from_this()
//       returns empty for one, giving use_count() == 0; those are treated as
//       roots, because whatever does own them is outside our view.
//
//  And the sweep frees nothing itself: it clears containers and lets reference
//  counting do the freeing, in its usual order. A logic error here nulls fields
//  of an object that should have lived -- a bug, but not memory corruption.
//
//  Full design and the rejected alternatives:
//  docs/object-lifetime-architecture.md
// ════════════════════════════════════════════════════════════════════════════

#include "gc.hpp"
#include "types.hpp"
#include "class.hpp"
#include "environment.hpp"
#include "function.hpp"

#include <cassert>
#include <memory>
#include <vector>

namespace bantu_gc {

// `owner` holds the Tracked<K> subobject's address; Tracked<K> is a unique,
// non-virtual base of exactly one type, so this downcast is well defined.
template <typename T, Kind K>
inline T* ownerAs(Head* h) {
    return static_cast<T*>(static_cast<Tracked<K>*>(h->owner));
}

// Clears the collecting flag however the collection leaves -- including by an
// exception out of a destructor during the sweep. Without it one throw would
// latch the flag and disable the collector for the life of the process, which
// is a leak that only shows up after something else has already gone wrong.
struct CollectingGuard {
    CollectingGuard()  { registry().collecting = true; }
    ~CollectingGuard() { registry().collecting = false; }
};

// ── Edges out of one Value ──────────────────────────────────────────────────
//
// Calls sink(Head*) once per strong reference to a tracked node, and does NOT
// recurse into it (invariant 1). Lists are walked inline because a Bantu list
// is held by value -- it belongs to this Value alone, so its contents are this
// Value's edges and are counted here exactly once.
template <typename F>
inline void edgesOfValue(const Value& v, F&& sink) {
    switch (v.type) {
        case Value::CLASS_INSTANCE:
            // Only an OWNING reference is an edge. A Value carrying just the
            // raw pointer holds no count, so it contributes nothing.
            if (v.classInstancePtr) sink(&v.classInstancePtr->gcHead());
            return;
        case Value::OBJECT:
            if (v.objectVal) sink(&v.objectVal->gcHead());
            return;
        case Value::FUNCTION:
            if (v.functionPtr) sink(&v.functionPtr->gcHead());
            return;
        case Value::LIST:
            for (const Value& e : v.listVal) edgesOfValue(e, sink);
            return;
        default:
            // Numbers, strings, bools, null, class definitions (never freed,
            // and reached through a raw pointer that holds no count) and native
            // handles (which hold no Value) have no edges.
            //
            // NATIVE_FN is deliberately here too, and it is the one real blind
            // spot: a std::function's captures cannot be enumerated, so a cycle
            // running through one is retained rather than collected. That is the
            // safe direction -- the capture still shows in use_count, so the
            // node reads as externally referenced and is treated as a root --
            // but it is not freed. The only place in the tree that ever built
            // such a cycle was sua's $res object, whose six chaining methods
            // owned the map that owned them; they capture it WEAKLY now
            // (bantuBuildResObject), which is the right fix regardless of the
            // collector. Any future native closure that captures a Value must
            // do the same.
            return;
    }
}

// ── Edges out of one tracked node ───────────────────────────────────────────
template <typename F>
inline void edgesOfNode(Head* h, F&& sink) {
    switch (h->kind) {
        case Kind::Instance: {
            auto* o = ownerAs<ClassInstance, Kind::Instance>(h);
            for (auto& kv : o->properties) edgesOfValue(kv.second, sink);
            return;
        }
        case Kind::Dict: {
            auto* o = ownerAs<ObjectMap, Kind::Dict>(h);
            for (auto& kv : *o) edgesOfValue(kv.second, sink);
            return;
        }
        case Kind::Env: {
            auto* o = ownerAs<Environment, Kind::Env>(h);
            for (auto& kv : o->variables) edgesOfValue(kv.second, sink);
            // The scope chain is an edge too: a closure's captured scope keeps
            // its parents alive, and a cycle can run through them.
            if (o->parent) sink(&o->parent->gcHead());
            return;
        }
        case Kind::Func: {
            auto* o = ownerAs<BantuFunction, Kind::Func>(h);
            if (o->closure) sink(&o->closure->gcHead());
            return;
        }
    }
}

// ── The strong reference count of one tracked node ──────────────────────────
// Read through weak_from_this(), which does not inflate the count. Zero means
// the object is not owned by a shared_ptr at all (a stack or `new` allocation),
// and such a node is treated as a root.
inline long useCountOf(Head* h) {
    switch (h->kind) {
        case Kind::Instance: return ownerAs<ClassInstance,  Kind::Instance>(h)->weak_from_this().use_count();
        case Kind::Dict:     return ownerAs<ObjectMap,      Kind::Dict    >(h)->weak_from_this().use_count();
        case Kind::Env:      return ownerAs<Environment,    Kind::Env     >(h)->weak_from_this().use_count();
        case Kind::Func:     return ownerAs<BantuFunction,  Kind::Func    >(h)->weak_from_this().use_count();
    }
    return 0;
}

// A strong reference to a tracked node, used to pin victims across the sweep.
// Returns an empty shared_ptr<void> for a node no shared_ptr owns.
inline std::shared_ptr<void> pin(Head* h) {
    switch (h->kind) {
        case Kind::Instance: return ownerAs<ClassInstance,  Kind::Instance>(h)->weak_from_this().lock();
        case Kind::Dict:     return ownerAs<ObjectMap,      Kind::Dict    >(h)->weak_from_this().lock();
        case Kind::Env:      return ownerAs<Environment,    Kind::Env     >(h)->weak_from_this().lock();
        case Kind::Func:     return ownerAs<BantuFunction,  Kind::Func    >(h)->weak_from_this().lock();
    }
    return {};
}

// Drop a victim's outgoing references. This is all the collector does to free
// anything: removing these references takes the cycle's counts to zero and
// reference counting performs the actual destruction.
inline void clearNode(Head* h) {
    switch (h->kind) {
        case Kind::Instance:
            ownerAs<ClassInstance, Kind::Instance>(h)->properties.clear();
            return;
        case Kind::Dict:
            ownerAs<ObjectMap, Kind::Dict>(h)->clear();
            return;
        case Kind::Env: {
            auto* o = ownerAs<Environment, Kind::Env>(h);
            o->variables.clear();
            o->constNames.clear();
            o->parent.reset();
            return;
        }
        case Kind::Func:
            // Only the closure. `body` is shared AST, which holds no Values and
            // is owned by the parse tree regardless.
            ownerAs<BantuFunction, Kind::Func>(h)->closure.reset();
            return;
    }
}

// ── The collection ──────────────────────────────────────────────────────────
// Returns the number of tracked nodes swept.
inline long collect() {
    Registry& r = registry();
    if (r.collecting) return 0;
    CollectingGuard guard;

    // Phase 1 — reset the scratch fields.
    for (Head* h = r.head; h; h = h->next) { h->internal = 0; h->marked = false; }

    // Phase 2 — count references held by other tracked nodes.
    for (Head* h = r.head; h; h = h->next)
        edgesOfNode(h, [](Head* t) { t->internal++; });

    // Phase 3 — anything with a reference we could not account for is a root,
    // and so is anything no shared_ptr owns.
    std::vector<Head*> work;
    for (Head* h = r.head; h; h = h->next) {
        const long uc = useCountOf(h);
        // Invariant 1, checked where checking is free. If this ever fires some
        // edge is being enumerated twice.
        assert(uc == 0 || h->internal <= uc);
        // A node is garbage ONLY when every reference to it was accounted for
        // from inside another tracked container. Written as an equality rather
        // than `internal < uc` on purpose: if a future edit ever did enumerate
        // an edge twice, internal would exceed uc, and the `<` form would then
        // treat a reachable node as garbage and sweep it. This form treats the
        // same mistake as "reachable" and retains it. The release build must not
        // depend on an assert that NDEBUG can remove.
        const bool garbage = (uc > 0 && h->internal == uc);
        if (!garbage) { h->marked = true; work.push_back(h); }
    }

    // Phase 4 — mark everything reachable from a root.
    while (!work.empty()) {
        Head* h = work.back();
        work.pop_back();
        edgesOfNode(h, [&work](Head* t) {
            if (!t->marked) { t->marked = true; work.push_back(t); }
        });
    }

    // Phase 5 — sweep. Pin every victim FIRST: clearing one victim can drop the
    // last reference to another and free it, which would unlink a Head from the
    // list being walked.
    std::vector<std::shared_ptr<void>> victims;
    std::vector<Head*> heads;
    for (Head* h = r.head; h; h = h->next) {
        if (h->marked) continue;
        // A node with no shared_ptr owner was marked in phase 3, so pin()
        // succeeds for everything reaching here.
        if (auto p = pin(h)) { victims.push_back(std::move(p)); heads.push_back(h); }
    }
    const long swept = (long)heads.size();
    for (Head* h : heads) clearNode(h);
    victims.clear();   // reference counting frees everything here

    r.collections++;
    r.freed += swept;
    r.surplus = 0;
    // Scale the next threshold with the surviving population, so a program
    // holding a large live set does not pay an O(live) scan over and over.
    r.threshold = r.live * 2;
    if (r.threshold < 10000) r.threshold = 10000;

    return swept;
}

// Called at statement boundaries; see gc.hpp::due().
inline void maybeCollect() { if (due()) collect(); }

} // namespace bantu_gc
