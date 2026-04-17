cmake_minimum_required(VERSION 3.18)

# ---------------------------------------------------------------------------
# Python3 must be found with Development.Module BEFORE pybind11 is loaded.
# pybind11 >= 2.6 "new tools" mode delegates to python3_add_library(), which
# is only registered when FindPython3 has been invoked with that component.
# (Development.Module was introduced in CMake 3.18.)
# ---------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
message(STATUS "Python3: ${Python3_EXECUTABLE} (${Python3_VERSION})")

# ---------------------------------------------------------------------------
# Locate pybind11.
# Priority:
#   1. User-supplied pybind11_DIR / CMAKE_PREFIX_PATH (system install, conda)
#   2. pip-installed pybind11 — ask the active Python interpreter for its
#      cmake dir (works with `pip install pybind11` or conda pybind11).
# ---------------------------------------------------------------------------
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _pybind11_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pybind11_cmd_result
  )
  if(_pybind11_cmd_result EQUAL 0 AND _pybind11_cmake_dir)
    list(APPEND CMAKE_PREFIX_PATH "${_pybind11_cmake_dir}")
    find_package(pybind11 CONFIG QUIET)
  endif()
endif()

if(NOT pybind11_FOUND)
  message(FATAL_ERROR
    "Could not find pybind11. Install it with one of:\n"
    "  pip install pybind11\n"
    "  conda install -c conda-forge pybind11\n"
    "or set pybind11_DIR to the directory containing pybind11Config.cmake."
  )
endif()

message(STATUS "pybind11 found: ${pybind11_DIR}")

set(PYJETSCAPE_SOURCES
  src/pyjetscape_core.cc
  src/bind_framework.cc
  src/bind_evolution.cc
  src/bind_initial_state.cc
  src/bind_fluid_dynamics.cc
  src/bind_music.cc
  src/bind_signal_manager.cc
)

pybind11_add_module(pyjetscape_core ${PYJETSCAPE_SOURCES})

target_include_directories(pyjetscape_core PRIVATE
  ${JetScape_INCLUDE_DIRS}
)

target_link_libraries(pyjetscape_core PRIVATE JetScape)

# The compiled extension lives inside the contrib source tree so that the
# jetscape Python package works directly via sys.path or PYTHONPATH without
# any installation step.
set_target_properties(pyjetscape_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/jetscape"
)

# Optional: install the jetscape package in editable mode so that
# `import jetscape` works without any PYTHONPATH / sys.path setup.
# Enable with -DJS_PIP_INSTALL_PYJETSCAPE=ON at cmake configure time.
option(JS_PIP_INSTALL_PYJETSCAPE
  "Run 'pip install -e' on the PyJetscape package after every build (requires pip)" OFF)

if(JS_PIP_INSTALL_PYJETSCAPE)
  add_custom_command(TARGET pyjetscape_core POST_BUILD
    COMMAND "${Python3_EXECUTABLE}" -m pip install -e
            "${CMAKE_CURRENT_SOURCE_DIR}" --quiet --no-build-isolation
    COMMENT "JS_PIP_INSTALL_PYJETSCAPE=ON: installing jetscape package in editable mode"
  )
  message(STATUS "pyjetscape_core: pip install -e enabled (JS_PIP_INSTALL_PYJETSCAPE=ON)")
else()
  message(STATUS "pyjetscape_core: output -> python/jetscape/ (set JS_PIP_INSTALL_PYJETSCAPE=ON to auto pip-install)")
endif()
