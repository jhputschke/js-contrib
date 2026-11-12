# Contributing to js-contrib

## Interface contract

Each contrib must:

1. **Inherit** from a JETSCAPE base class
   (`FluidDynamics`, `InitialState`, `JetEnergyLossModule<T>`, etc.)
2. **Implement** `InitTask()` and `ExecuteTask()` (or `DoEnergyLoss()` etc.)
3. **Register** with the static factory:
   ```cpp
   // In MyModule.h (private section)
   static RegisterJetScapeModule<MyModule> reg;

   // In MyModule.cc
   RegisterJetScapeModule<MyModule> MyModule::reg("MyModule");
   ```
4. **Provide** a per-contrib `CMakeLists.txt` that:
   - Uses `${JetScape_INCLUDE_DIRS}` for include paths (set by the top-level js-contrib CMakeLists)
   - Links against `JetScape`
   - Defaults `BUILD_<CONTRIB>=OFF`

## Directory layout

```
contribs/
  MyContrib/
    CMakeLists.txt
    src/
      MyModule.cc
      MyModule.h
    config/
      example.xml
    example/
      CMakeLists.txt
      myExample.cc
    README.md
```

## Registering the module in XML

Once built and linked (via `add_subdirectory` or `LD_PRELOAD`), the module is
available in any JETSCAPE config XML under its registered name, e.g.:

```xml
<Hydro>
  <MyModule>
    <name>MyModule</name>
  </MyModule>
</Hydro>
```

No core JETSCAPE/X-SCAPE changes required.

## Releases and versions

A js-contrib release is a git tag `vX.Y` on `main` (with a GitHub release). pyjetscape's
version comes from that tag (setuptools-scm, `contribs/PyJetscape/pyproject.toml`): there
is no version number to edit.

- At the tag the version is `X.Y`; later commits give `X.(Y+1).devN+g<commit>`, and
  uncommitted changes add `.d<date>`. Only tags `v<digits>...` count (`tutorials-v*` does not).
- `pip install -e contribs/PyJetscape` keeps the version of the moment it ran; reinstall
  after a new tag.
- A build from a copy without js-contrib's `.git` (the macOS wheel's `build_wheel.sh`) sets
  `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYJETSCAPE=X.Y`.
