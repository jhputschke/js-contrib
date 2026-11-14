# C++ bindings

`jetscape.pyjetscape_core`: the pybind11 bindings of X-SCAPE, with the C++ class and method names
and argument order. Every module base class (`JetEnergyLoss`, `HardProcess`, `FluidDynamics`,
...) can be subclassed in Python; [Writing modules in Python](../modules.md) shows how. The
methods and properties of the Pythonic layer ([`jetscape.pythonic`](run.md)) appear here under the class they
are installed on.

This page is generated from the type stubs `pyjetscape_core.pyi`, which
`utils/gen_pyjetscape_stubs.py` writes from a build (see [Building this site](../../reference/site.md)).

::: jetscape.pyjetscape_core
    options:
      heading_level: 2
      show_if_no_docstring: true
      members_order: alphabetical
