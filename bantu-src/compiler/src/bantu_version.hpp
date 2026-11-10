#pragma once
// ═══════════════════════════════════════════════════════════════════════════
//  The one place the language's version number lives.
//
//  It used to live in main.cpp alone, which was survivable while nothing else
//  needed it. The engine field (ecosystem stage 2) needs it in the package
//  manager too, and a version number that exists twice is a version number
//  that will disagree with itself.
//
//  WHEN TO BUMP, which is the part that was not being done:
//
//    * a new NATIVE FAMILY or a new native builtin  -> bump the MINOR.
//      A package can now call something the previous binary does not have, so
//      the floor table needs a version to point at, and the version it points
//      at has to exist.
//    * a behaviour change to an existing builtin    -> bump the MINOR.
//    * a fix with no new surface                    -> bump the PATCH.
//
//  The history that made this rule explicit: 1.3.2 was released 2026-09-06 and
//  the version string never moved again, while ndarray (2026-09-13), raster
//  (2026-09-16), the crypto families (2026-08-28) and the threading builtins
//  (2026-11-10) all landed afterwards. Nine packages in the registry therefore
//  required a binary that had never been released, and the only symptom a user
//  got was `Undefined variable: nd_zeros`. See docs/ecosystem-architecture.md
//  §7.2 L0.
// ═══════════════════════════════════════════════════════════════════════════

#define BANTU_VERSION_STR "1.4.0"

namespace bantu {
inline const char* version() { return BANTU_VERSION_STR; }
}
