#pragma once
// ════════════════════════════════════════════════════════════════════════════
//  gc.hpp — the cycle collector's registry.
//
//  Reference counting frees a Bantu object the moment the last reference to it
//  drops, which is prompt, cheap and needs no root set. What it cannot do is
//  free a cycle: two objects that refer to each other each hold the other's
//  count at one, and neither is ever freed. Measured on the build before this
//  landed, 400,000 iterations each:
//
//      $a = new Node();                          ->        1 live,   5.1 MB
//      $a = new Node(); $a.self = $a;            ->  400,000 live,   246 MB
//      $a = new Node(); $b = ...; $a.p = $b;
//                       $b.p = $a;               ->  800,000 live,   502 MB
//      $a = new Node(); $a.cb = $a.m;            ->  400,000 live,   558 MB
//      def outer() { def inner() {...} }         ->   20,001 scopes,  34 MB
//                                                     (20,000 calls)
//
//  The last two are the interpreter's own doing, not the user's: binding a
//  method builds a scope holding `this` and a function closing over it, and
//  defining a function stores it into the very scope it captured. A user who
//  writes a nested helper, or stores a handler on an object, leaks without
//  having written a back-reference at all.
//
//  So refcounting stays primary and a collector handles only what it provably
//  cannot -- which is what CPython (since 2.0) and PHP (since 5.3) both do.
//
//  THIS HEADER KNOWS NOTHING ABOUT BANTU'S TYPES. It is included by
//  class.hpp, environment.hpp, function.hpp and ordered_map.hpp, all of which
//  are below Value in the include order, so it must not mention Value. The
//  half that walks Bantu objects lives in gc_collect.hpp, which sits above all
//  of them. That split is what keeps the include graph acyclic.
//
//  Design, the alternatives rejected, and the safety argument:
//  docs/object-lifetime-architecture.md
//
//  THREADING: Bantu code runs on one thread at a time. sua's suspendable
//  handlers are real threads, but bantu_co::Scheduler hands a baton between
//  them under a mutex -- spawn() returns only once the task has parked or
//  finished -- so no two threads evaluate Bantu concurrently, and the hand-off
//  publishes the writes. Hence a plain global list and no lock, which is the
//  same invariant the event loop already relies on.
// ════════════════════════════════════════════════════════════════════════════

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace bantu_gc {

// The tracked kinds. A node kind is anything that both holds Bantu values and
// has a shared identity, so it can be one end of a cycle. Lists are NOT here:
// a Bantu list is held by value and copied on assignment, so it has no
// identity to point back with -- it carries a cycle's edges, never its nodes.
enum class Kind : uint8_t { Instance, Dict, Env, Func };

// Embedded in every tracked object. Two list pointers, the owner, a kind and
// two scratch fields used only during a collection: 32 bytes, no allocation.
struct Head {
    Head*  prev  = nullptr;
    Head*  next  = nullptr;
    void*  owner = nullptr;   // the object this Head is embedded in
    long   internal = 0;      // scratch: references seen from other tracked nodes
    Kind   kind = Kind::Instance;
    bool   marked = false;    // scratch: reachable from a root
};

// One global registry. A doubly-linked list rather than a vector or a hash set
// because unlink must be O(1) and allocation-free: Environment is the hottest
// allocation in the interpreter (roughly six scopes per interpreted loop
// iteration), and a registry that allocated would tax every function call.
struct Registry {
    Head*    head = nullptr;
    long     live = 0;          // tracked nodes currently alive
    long     surplus = 0;       // creations - destructions since the last collection
    long     threshold = 10000; // collect when surplus crosses this
    long     collections = 0;
    long     freed = 0;         // tracked nodes swept, cumulative
    bool     enabled = true;
    bool     collecting = false; // re-entrancy guard: the sweep runs destructors
};

// A namespace-scope inline variable, NOT a function-local static, and that is
// a measured decision rather than a style one. Every Registry member has a
// constant initialiser, so this is constant-initialised: the accessor compiles
// to a plain address. A function-local static would instead need a thread-safe
// initialisation guard -- an atomic load on EVERY call -- and link(), unlink()
// and due() are called several times per interpreted loop iteration. With the
// guard the interpreter benchmark measured +3.85% against the pre-collector
// build -- over the +/-2% gate. Without it, -0.85%: the collector's remaining
// cost is below the noise floor, and the guard was the whole of it. Both
// figures are best-of-5, run in both A/B orderings to cancel position bias.
inline Registry g_registry;
inline Registry& registry() { return g_registry; }

// BANTU_GC=0 disables the AUTOMATIC collection for the process. gc_collect()
// still works when called explicitly, so this is a diagnostic tool and a
// latency escape hatch, never a way to lose the fix. Called once at startup,
// from where the gc_* builtins are registered, so nothing is read per call.
inline void applyEnvironmentOverride() {
    const char* e = std::getenv("BANTU_GC");
    if (e && e[0] == '0' && e[1] == '\0') g_registry.enabled = false;
}

// Link on construction. Four pointer writes, no allocation, no atomics.
inline void link(Head* h, Kind k, void* owner) {
    Registry& r = registry();
    h->kind  = k;
    h->owner = owner;
    h->prev  = nullptr;
    h->next  = r.head;
    if (r.head) r.head->prev = h;
    r.head = h;
    r.live++;
    r.surplus++;
}

// Unlink on destruction. Also runs during the sweep, when clearing one victim
// frees another -- which is why the sweep pins its victims with strong
// references before it clears any of them.
inline void unlink(Head* h) {
    Registry& r = registry();
    if (h->prev) h->prev->next = h->next;
    else if (r.head == h) r.head = h->next;
    if (h->next) h->next->prev = h->prev;
    h->prev = h->next = nullptr;
    r.live--;
    r.surplus--;
}

// Called at statement boundaries. Deliberately tiny and header-inline: this is
// the only cost a program with no cycles pays beyond link/unlink.
inline bool due() {
    const Registry& r = registry();
    return r.enabled && !r.collecting && r.surplus >= r.threshold;
}

// ── The base every tracked type derives from ────────────────────────────────
//
// Kind is a template parameter rather than a constructor argument so that
// gc_collect.hpp can recover the derived pointer with a static_cast: `owner`
// holds the Tracked<K> subobject's address, and Tracked<K> is a unique,
// non-virtual base of exactly one type. Storing `this` from the derived
// constructor instead would not work -- the base runs first.
//
// Copying a tracked object gives the copy its OWN registry entry and leaves
// both heads where they are; a Head is an identity, not a value. ObjectMap is
// the one tracked type that is genuinely copied (every dict literal), so this
// is load-bearing rather than defensive.
template <Kind K>
class Tracked {
public:
    Tracked()                          { link(&gcHead_, K, this); }
    Tracked(const Tracked&)            { link(&gcHead_, K, this); }
    Tracked(Tracked&&) noexcept        { link(&gcHead_, K, this); }
    Tracked& operator=(const Tracked&) noexcept { return *this; }
    Tracked& operator=(Tracked&&)      noexcept { return *this; }
    ~Tracked()                         { unlink(&gcHead_); }

    Head&       gcHead()       { return gcHead_; }
    const Head& gcHead() const { return gcHead_; }

private:
    Head gcHead_;
};

} // namespace bantu_gc
