cmake_minimum_required(VERSION 3.15)

# pybind11 can be found via cmake or pip-installed pybind11
find_package(pybind11 REQUIRED)

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

# Output the .so directly into python/jetscape/ for easy import
set_target_properties(pyjetscape_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/jetscape"
)

message(STATUS "pyjetscape_core will be built to ${CMAKE_CURRENT_SOURCE_DIR}/python/jetscape/")
