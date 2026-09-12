# Fixture packages

Not real dependencies. `bantu add <pkg>` installs into `bantu_modules/<pkg>/`, and these two
packages exist so `tests/lang_module_test.b` can prove that `include "greeter"` resolves there:

- **`greeter/`** has a `package.json` whose `"main"` points at `src/hello.b` — somewhere the
  conventional `<name>.b` rule would never find. It also carries a `"note": "main"` decoy, so the
  manifest scanner is proven to distinguish a key from a value that happens to read the same.
- **`plainpkg/`** has no manifest at all, proving the `<name>.b` fallback.

The `tests/*.b` glob CI runs is shallow, so nothing in here is executed as a test.
