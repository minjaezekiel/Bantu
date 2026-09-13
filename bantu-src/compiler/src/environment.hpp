#pragma once
/**
 * Bantu Language - Variable Environment (Scope Chain)
 */

#include "types.hpp"
#include <memory>
#include <unordered_map>
#include <unordered_set>

class Environment {
public:
    std::unordered_map<std::string, Value> variables;
    std::unordered_set<std::string> constNames;  // names declared `const` in THIS scope
    std::shared_ptr<Environment> parent;
    // Marks a function-call scope (or the global root). Plain-variable assignment
    // (`$x = v`) resolves only up to and including the nearest such boundary; it
    // never reaches past it into the caller/closure. This gives functions proper
    // LOCAL variables (Python-style) so a callee reusing a caller's variable name
    // — e.g. a loop counter `$i` — cannot clobber the caller's binding. Reads
    // (`get`) still fall through all scopes, so functions can read globals.
    bool functionScope = false;

    explicit Environment(std::shared_ptr<Environment> parentEnv = nullptr)
        : parent(std::move(parentEnv)) {}

    // define() creates/overwrites a binding in this scope. When isConst is true
    // the binding becomes final — a later assignment to it raises an error
    // (like Java's `final`; the referenced object may still be mutated).
    void define(const std::string& name, const Value& value, bool isConst = false) {
        variables[name] = value;
        if (isConst) constNames.insert(name);
    }

    Value get(const std::string& name) {
        auto it = variables.find(name);
        if (it != variables.end()) return it->second;
        if (parent) return parent->get(name);
        ErrorHandler::throwReferenceError("Undefined variable: " + name);
        return Value();
    }

    // The slot `assign()` would write to -- but only if it ALREADY exists
    // within the window assign() searches, which is this scope up to and
    // including the nearest function boundary. Never creates a binding, and
    // refuses a const so the caller falls back and assign() raises as usual.
    //
    // Exists so that `$s = $s + …` can append into the string in place without
    // guessing where assign() would have put the result. Resolving it any other
    // way would be subtly wrong: getRef() walks the WHOLE chain, so inside a
    // function it would find and mutate a global that assign() would instead
    // have shadowed with a new local.
    Value* existingAssignSlot(const std::string& name) {
        Environment* e = this;
        while (true) {
            auto it = e->variables.find(name);
            if (it != e->variables.end())
                return e->constNames.count(name) ? nullptr : &it->second;
            if (e->functionScope || !e->parent) return nullptr;
            e = e->parent.get();
        }
    }

    // Like getRef, but reports a missing name instead of raising. One walk of
    // the scope chain answers both "is it there?" and "where is it?" —
    // has() followed by getRef() walks it twice, which is measurable on a path
    // as hot as `$a[$i]`.
    Value* tryGetRef(const std::string& name) {
        for (Environment* e = this; e; e = e->parent.get()) {
            auto it = e->variables.find(name);
            if (it != e->variables.end()) return &it->second;
        }
        return nullptr;
    }

    Value& getRef(const std::string& name) {
        auto it = variables.find(name);
        if (it != variables.end()) return it->second;
        if (parent) return parent->getRef(name);
        ErrorHandler::throwReferenceError("Undefined variable: " + name);
        static Value nullVal;
        return nullVal;
    }

    void set(const std::string& name, const Value& value) {
        auto it = variables.find(name);
        if (it != variables.end()) {
            if (constNames.count(name)) {
                ErrorHandler::throwError("Cannot reassign constant '" + name + "'",
                                         0, 0, ErrorHandler::TYPE_ERROR);
            }
            it->second = value;
            return;
        }
        if (parent) { parent->set(name, value); return; }
        ErrorHandler::throwReferenceError("Undefined variable: " + name);
    }

    bool has(const std::string& name) const {
        auto it = variables.find(name);
        if (it != variables.end()) return true;
        if (parent) return parent->has(name);
        return false;
    }

    // assign() implements `$x = value`. It updates the nearest existing binding
    // found from this scope up to AND INCLUDING the enclosing function boundary;
    // if none exists in that range, it creates the binding at the function
    // boundary (a function-local). It never climbs past a functionScope env, so
    // assignment cannot mutate a caller's or a closure's/global's variable of the
    // same name. (Field/index assignment on shared dicts is unaffected — that
    // mutates the object, not a variable binding.)
    void assign(const std::string& name, const Value& value) {
        Environment* e = this;
        while (true) {
            auto it = e->variables.find(name);
            if (it != e->variables.end()) {
                if (e->constNames.count(name)) {
                    ErrorHandler::throwError("Cannot reassign constant '" + name + "'",
                                             0, 0, ErrorHandler::TYPE_ERROR);
                }
                it->second = value;
                return;
            }
            // Not found in this scope. Stop at the function boundary (or the
            // global root) and define the variable there as a local.
            if (e->functionScope || !e->parent) {
                e->variables[name] = value;
                return;
            }
            e = e->parent.get();
        }
    }
};
