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
