cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(js-contrib VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Route all executables built by js-contrib to the top-level build directory
# so they sit next to runJetscape etc.
# NOTE: CMAKE_LIBRARY_OUTPUT_DIRECTORY is intentionally NOT set globally here
# because ROOT_GENERATE_DICTIONARY computes install() DESTINATION as the
# relative path from CMAKE_BINARY_DIR to CMAKE_LIBRARY_OUTPUT_DIRECTORY;
# setting both to the same path yields an empty string → CMake error.
# Shared library destinations are set per-target instead.
# ---------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# ---------------------------------------------------------------------------
# Locate JetScape.  When included via add_subdirectory from X-SCAPE/JETSCAPE
# the JetScape target already exists in scope; otherwise locate it via the
# build-tree export written to JETSCAPE_DIR (= path to the build directory).
# ---------------------------------------------------------------------------
if(NOT TARGET JetScape)
  find_package(JetScape REQUIRED
    HINTS $ENV{JETSCAPE_DIR} ${JETSCAPE_DIR}
    PATH_SUFFIXES . build
  )
  message(STATUS "js-contrib: found JetScape package in ${JetScape_DIR}")
else()
  message(STATUS "js-contrib: using in-tree JetScape target")
  # When built in-tree the include dirs come from the parent CMake scope.
  # Replicate them as JetScape_INCLUDE_DIRS so all contribs use the same variable.
  if(NOT DEFINED JetScape_INCLUDE_DIRS)
    get_target_property(_js_iface JetScape INTERFACE_INCLUDE_DIRECTORIES)
    if(_js_iface)
      set(JetScape_INCLUDE_DIRS ${_js_iface})
    else()
      # Fall back to standard X-SCAPE/JETSCAPE-FNO source layout
      set(JetScape_INCLUDE_DIRS
        ${CMAKE_SOURCE_DIR}/src/framework
        ${CMAKE_SOURCE_DIR}/src/initialstate
        ${CMAKE_SOURCE_DIR}/src/preequilibrium
        ${CMAKE_SOURCE_DIR}/src/hydro
        ${CMAKE_SOURCE_DIR}/src/liquefier
        ${CMAKE_SOURCE_DIR}/src/jet
        ${CMAKE_SOURCE_DIR}/src/hadronization
        ${CMAKE_SOURCE_DIR}/src/afterburner
        ${CMAKE_SOURCE_DIR}/external_packages
        ${CMAKE_SOURCE_DIR}/external_packages/gtl/include
        ${CMAKE_SOURCE_DIR}/external_packages/trento/src
      )
    endif()
  endif()
endif()

# ---------------------------------------------------------------------------
# Per-contrib options (all OFF by default)
# ---------------------------------------------------------------------------
option(BUILD_FNO_HYDRO
  "Build FnoHydro contrib (requires libtorch ~2 GB and ROOT)" OFF)
option(BUILD_PYJETSCAPE
  "Build PyJetscape pybind11 Python bindings (compile from source, requires pybind11)" OFF)

# ---------------------------------------------------------------------------
# Contribs
# ---------------------------------------------------------------------------
if(BUILD_FNO_HYDRO)
  add_subdirectory(contribs/FnoHydro)
endif()

if(BUILD_PYJETSCAPE)
  add_subdirectory(contribs/PyJetscape)
endif()
