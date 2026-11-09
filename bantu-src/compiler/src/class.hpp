#pragma once
/**
 * Bantu Language - Class System
 * Supports: single inheritance (extends), multiple inheritance (implements)
 */

#include "types.hpp"
#include "environment.hpp"
#include "gc.hpp"
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>

class ClassDefinition {
public:
    std::string name;
    std::unordered_map<std::string, Value> methods;
    std::unordered_map<std::string, Value> staticMethods;
    ClassDefinition* parentClass = nullptr;                    // extends (single)
    std::vector<ClassDefinition*> implementsClasses;           // implements (multiple)

    ClassDefinition(const std::string& n) : name(n) {}

    void addMethod(const std::string& methodName, const Value& method) {
        methods[methodName] = method;
    }

    void addStaticMethod(const std::string& methodName, const Value& method) {
        staticMethods[methodName] = method;
    }

    // How deep an inheritance graph may be walked before we declare it
    // malformed. Real hierarchies are a handful of levels; this only has to be
    // smaller than the C++ stack and larger than anything sane.
    static const int kMaxInheritWalk = 256;

    // Look up method: own methods -> parent chain -> implements chain
    //
    // `depth` bounds the walk. These are raw pointers with no owner and the
    // only termination condition is a null parent, so a cycle anywhere in the
    // graph used to mean unbounded recursion and a stack overflow -- a SIGSEGV
    // with no diagnostic. evalClassDecl now makes a cycle unreachable AND
    // rejects one explicitly, so this is defence in depth: it turns the crash
    // into a value the caller can reason about, for the cost of one integer
    // compare per level.
    Value getMethod(const std::string& methodName, int depth = 0) {
        if (depth > kMaxInheritWalk) return Value();

        // 1. Check own methods
        auto it = methods.find(methodName);
        if (it != methods.end()) return it->second;

        // 2. Check extends (parent) — single inheritance chain
        if (parentClass) {
            Value parentMethod = parentClass->getMethod(methodName, depth + 1);
            if (!parentMethod.isNull()) return parentMethod;
        }

        // 3. Check implements — multiple inheritance (depth-first, left-to-right)
        for (auto* impl : implementsClasses) {
            Value implMethod = impl->getMethod(methodName, depth + 1);
            if (!implMethod.isNull()) return implMethod;
        }

        return Value();
    }

    // Check if this class is or extends a given class (for isinstance).
    // Depth-bounded for the same reason as getMethod.
    bool isInstanceOf(const std::string& className, int depth = 0) {
        if (depth > kMaxInheritWalk) return false;
        if (name == className) return true;
        if (parentClass && parentClass->isInstanceOf(className, depth + 1)) return true;
        for (auto* impl : implementsClasses) {
            if (impl->isInstanceOf(className, depth + 1)) return true;
        }
        return false;
    }
};

// Tracked by the cycle collector: `properties` holds Values, so two instances
// referring to each other -- a tree node and its parent, the commonest shape
// there is -- keep each other's reference count at one forever. See gc.hpp.
class ClassInstance : public bantu_gc::Tracked<bantu_gc::Kind::Instance>,
                      public std::enable_shared_from_this<ClassInstance> {
public:
    ClassDefinition* classDef;
    std::unordered_map<std::string, Value> properties;

    ClassInstance(ClassDefinition* cd) : classDef(cd) {}

    Value getProperty(const std::string& propName) {
        // 1. Check instance properties (set at runtime)
        auto it = properties.find(propName);
        if (it != properties.end()) return it->second;

        // 2. Check class methods
        Value method = classDef->getMethod(propName);
        if (!method.isNull()) return method;

        return Value();
    }

    void setProperty(const std::string& propName, const Value& val) {
        properties[propName] = val;
    }
};
