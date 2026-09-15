#pragma once
/**
 * Bantu Language v1.2.2 — Module Resolver
 *
 * Resolves `include` paths relative to the importing file, reads, lexes,
 * and parses the target `.b` file into an AST. The Evaluator is responsible
 * for executing the AST in the proper scope.
 *
 * Resolution order (v1.2.2):
 *   1. If `path` is absolute and exists, use it directly.
 *   2. Resolve relative to the *importing* file's directory (preferred).
 *   3. Resolve relative to the current working directory.
 *   4. Try with `.b` extension appended if missing.
 *   5. For a BARE name (`include "numba"`, not `"./numba.b"`), look in
 *      `bantu_modules/<name>/` next to the importing file and under the cwd,
 *      honouring the package's `package.json` "main". This is where
 *      `bantu add <pkg>` installs, and without it an installed package could
 *      only be reached by spelling out its full path.
 *   6. (v1.2.2) For each directory in $BANTU_PATH (':' on POSIX, ';' on Windows),
 *      try `<dir>/<path>` and `<dir>/<path>.b`. Lets users install shared
 *      module libraries outside their project tree.
 *
 * The resolved path is canonicalized via realpath() so that the same file
 * reached via different relative paths (./x.b, ../pkg/x.b) produces the
 * same canonical key — preventing duplicate execution.
 *
 * The resolver never throws — on failure it returns an empty AST and
 * records a diagnostic string in `errOut`.
 */

#include "types.hpp"
#include "lexer.hpp"
#include "parser.hpp"

#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <cstdlib>
#include <cctype>

#ifdef _WIN32
    #include <direct.h>
    #include <windows.h>
    #define GETCWD _getcwd
#else
    #include <unistd.h>
    #define GETCWD getcwd
#endif

namespace bantu {

inline bool pathExists(const std::string& p) {
    struct stat buf;
    return stat(p.c_str(), &buf) == 0;
}

// A module must be a REGULAR FILE. pathExists() also says yes to a directory,
// and a directory opened as a stream reads as empty -- so `include "bplot"`
// run from a folder that contains a bplot/ directory parsed an empty file and
// bound the alias to a module with nothing in it. No error anywhere: the first
// sign was "figure holds null" at the first call. That is exactly the layout
// of this repository, and of any project that vendors a package.
inline bool isRegularFile(const std::string& p) {
    struct stat buf;
    if (stat(p.c_str(), &buf) != 0) return false;
    return (buf.st_mode & S_IFMT) == S_IFREG;
}

inline bool isDirectory(const std::string& p) {
    struct stat buf;
    if (stat(p.c_str(), &buf) != 0) return false;
    return (buf.st_mode & S_IFMT) == S_IFDIR;
}

inline std::string dirOf(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    return (slash == std::string::npos) ? "." : path.substr(0, slash);
}

inline std::string getCwd() {
    char buf[4096];
    if (GETCWD(buf, sizeof(buf)) != nullptr) return std::string(buf);
    return ".";
}

// v1.2.2: Canonicalize a path so that the same file reached via different
// relative paths produces the same key. Falls back to the input path on
// platforms where realpath() is unavailable or fails.
inline std::string canonicalize(const std::string& path) {
#ifdef _WIN32
    char full[MAX_PATH];
    DWORD len = GetFullPathNameA(path.c_str(), MAX_PATH, full, nullptr);
    if (len > 0 && len < MAX_PATH) return std::string(full);
    return path;
#else
    char full[4096];
    if (realpath(path.c_str(), full) != nullptr) return std::string(full);
    return path;
#endif
}

inline std::string joinPath(const std::string& dir, const std::string& rel) {
    if (rel.empty()) return dir;
    if (rel[0] == '/' || rel[0] == '\\' ||
        (rel.size() >= 2 && rel[1] == ':')) return rel; // absolute (POSIX or Windows drive)
    if (dir.empty() || dir == ".") return rel;
    char sep =
#ifdef _WIN32
        '\\';
#else
        '/';
#endif
    if (dir.back() == '/' || dir.back() == '\\') return dir + rel;
    return dir + std::string(1, sep) + rel;
}

// v1.2.2: Split a PATH-style env var. On Windows the separator is ';',
// on POSIX it is ':'.
inline std::vector<std::string> splitPathEnv(const std::string& env) {
    std::vector<std::string> out;
    if (env.empty()) return out;
    char sep =
#ifdef _WIN32
        ';';
#else
        ':';
#endif
    std::string cur;
    for (char c : env) {
        if (c == sep) {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// v1.2.2: Read $BANTU_PATH and return a list of search dirs.
inline std::vector<std::string> bantuPathDirs() {
    const char* env = std::getenv("BANTU_PATH");
    if (!env) return {};
    return splitPathEnv(env);
}

// ── Installed packages: ./bantu_modules/<name>/ ──────────────────────────────
// `bantu add <pkg>` installs into ./bantu_modules/<pkg>/, but nothing taught
// the resolver about that directory, so `include "numba" as np;` did not
// resolve and every user had to write the full
// `include "./bantu_modules/numba/numba.b" as np;` instead. This closes that
// for every installed package at once.

// Read one top-level string field out of a package.json. Deliberately a
// minimal scanner rather than a JSON parser: these manifests are written by
// `bantu init` / `bantu publish` and are always a flat object of strings, and
// package_manager.hpp already reads them the same way. Requiring the key to
// sit right after '{' or ',' stops a value like "the main entry" in
// "description" from being mistaken for the "main" key.
inline std::string readManifestField(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    size_t from = 0;
    while (true) {
        size_t k = json.find(needle, from);
        if (k == std::string::npos) return "";
        from = k + needle.size();
        // The key must be a key, not part of some other string's value.
        size_t p = k;
        while (p > 0 && std::isspace((unsigned char)json[p - 1])) p--;
        if (p == 0 || (json[p - 1] != '{' && json[p - 1] != ',')) continue;

        size_t colon = json.find(':', from);
        if (colon == std::string::npos) return "";
        size_t q1 = json.find('"', colon + 1);
        if (q1 == std::string::npos) return "";
        size_t q2 = json.find('"', q1 + 1);
        if (q2 == std::string::npos) return "";
        return json.substr(q1 + 1, q2 - q1 - 1);
    }
}

inline std::string readWholeFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// A bare module name: "numba", not "./numba.b" or "../pkg/numba.b". Only bare
// names look in bantu_modules, so an explicit relative path always means
// exactly what it says.
inline bool isBareModuleName(const std::string& p) {
    if (p.empty()) return false;
    if (p[0] == '.' || p[0] == '/' || p[0] == '\\') return false;
    if (p.find('/') != std::string::npos || p.find('\\') != std::string::npos) return false;
    if (p.size() >= 2 && p[1] == ':') return false;   // Windows drive letter
    return true;
}

// Append the candidate entry points for <baseDir>/bantu_modules/<name>/.
// package.json's "main" wins; the conventional names are tried after it so a
// package with no manifest still resolves.
inline void addPackageCandidates(std::vector<std::string>& candidates,
                                 const std::string& baseDir,
                                 const std::string& name) {
    const std::string pkgDir = joinPath(joinPath(baseDir, "bantu_modules"), name);
    if (!pathExists(pkgDir)) return;

    const std::string manifest = joinPath(pkgDir, "package.json");
    if (pathExists(manifest)) {
        const std::string main = readManifestField(readWholeFile(manifest), "main");
        if (!main.empty()) candidates.push_back(joinPath(pkgDir, main));
    }
    candidates.push_back(joinPath(pkgDir, name + ".b"));
    candidates.push_back(joinPath(pkgDir, "index.b"));
    candidates.push_back(joinPath(pkgDir, "main.b"));
}

struct ResolvedModule {
    std::string resolvedPath;   // absolute-ish path that was opened
    std::string source;         // file contents
    std::vector<std::shared_ptr<ASTNode>> ast;
    bool ok = false;
    std::string err;
};

inline ResolvedModule resolveAndParse(const std::string& rawPath,
                                      const std::string& importingFilePath) {
    ResolvedModule out;

    std::vector<std::string> candidates;
    // 1. Absolute (already-normalized by joinPath)
    candidates.push_back(rawPath);
    // 2. Relative to importing file's directory
    if (!importingFilePath.empty()) {
        candidates.push_back(joinPath(dirOf(importingFilePath), rawPath));
    }
    // 3. Relative to cwd
    candidates.push_back(joinPath(getCwd(), rawPath));
    // 4. Append .b if missing
    {
        std::string withExt = rawPath;
        if (withExt.size() < 2 || withExt.substr(withExt.size() - 2) != ".b") {
            withExt += ".b";
            candidates.push_back(withExt);
            if (!importingFilePath.empty()) {
                candidates.push_back(joinPath(dirOf(importingFilePath), withExt));
            }
            candidates.push_back(joinPath(getCwd(), withExt));
        }
    }
    // 5. Installed packages: ./bantu_modules/<name>/ for a BARE name.
    //    Placed after the relative forms so an explicit path always wins, and
    //    before $BANTU_PATH so a project's own dependency beats a global
    //    search dir.
    if (isBareModuleName(rawPath)) {
        if (!importingFilePath.empty()) {
            addPackageCandidates(candidates, dirOf(importingFilePath), rawPath);
        }
        addPackageCandidates(candidates, getCwd(), rawPath);
    }
    // 5b. A bare name that names a DIRECTORY is a package directory: resolve
    //     inside it exactly as bantu_modules/<name>/ is resolved. This is how
    //     `include "bplot"` works from a checkout that has bplot/bplot.b, and
    //     it is Node's convention for require('./dir'). Placed after
    //     bantu_modules so an installed package still wins over a folder that
    //     happens to share its name.
    if (isBareModuleName(rawPath)) {
        auto addDirPackage = [&](const std::string& baseDir) {
            const std::string dir = joinPath(baseDir, rawPath);
            if (!isDirectory(dir)) return;
            const std::string manifest = joinPath(dir, "package.json");
            if (isRegularFile(manifest)) {
                const std::string main = readManifestField(readWholeFile(manifest), "main");
                if (!main.empty()) candidates.push_back(joinPath(dir, main));
            }
            candidates.push_back(joinPath(dir, rawPath + ".b"));
            candidates.push_back(joinPath(dir, "index.b"));
        };
        if (!importingFilePath.empty()) addDirPackage(dirOf(importingFilePath));
        addDirPackage(getCwd());
    }
    // 6. (v1.2.2) $BANTU_PATH lookup — for shared module libraries
    for (const auto& dir : bantuPathDirs()) {
        candidates.push_back(joinPath(dir, rawPath));
        std::string withExt = rawPath;
        if (withExt.size() < 2 || withExt.substr(withExt.size() - 2) != ".b") {
            withExt += ".b";
            candidates.push_back(joinPath(dir, withExt));
        }
    }

    std::string chosen;
    for (const auto& c : candidates) {
        if (isRegularFile(c)) { chosen = c; break; }
    }

    if (chosen.empty()) {
        out.err = "Module not found: " + rawPath;
        if (!importingFilePath.empty()) {
            out.err += " (imported from " + importingFilePath + ")";
        }
        return out;
    }

    std::ifstream f(chosen, std::ios::binary);
    if (!f.is_open()) {
        out.err = "Cannot open module: " + chosen;
        return out;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    out.source = ss.str();
    // v1.2.2: canonicalize so duplicate relative paths collapse to one key
    out.resolvedPath = canonicalize(chosen);

    try {
        Lexer lex(out.source);
        auto tokens = lex.tokenize();
        Parser parser(std::move(tokens));
        out.ast = parser.parse();
        out.ok = true;
    } catch (const std::exception& e) {
        out.err = std::string("Parse error in '") + chosen + "': " + e.what();
    } catch (...) {
        out.err = std::string("Unknown parse error in '") + chosen + "'";
    }
    return out;
}

} // namespace bantu
