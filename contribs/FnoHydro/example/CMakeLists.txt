cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# Place all executables directly in the top-level build directory so they are
# visible alongside runJetscape etc. regardless of nesting depth.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include(${ROOT_USE_FILE})

foreach(_exe fnoHydroTest fnoRooInTest BulkFnoRooIn JetFnoRooIn)
  add_executable(${_exe} ${CMAKE_CURRENT_SOURCE_DIR}/${_exe}.cc)
  target_link_libraries(${_exe} FnoModule ${ROOT_LIBRARIES})
  target_include_directories(${_exe} PRIVATE ${JetScape_INCLUDE_DIRS})
endforeach()

foreach(_exe simpleReaderAna bulkAna jetAna)
  add_executable(${_exe} ${CMAKE_CURRENT_SOURCE_DIR}/${_exe}.cc)
  target_link_libraries(${_exe} JetScape ${ROOT_LIBRARIES})
  target_include_directories(${_exe} PRIVATE ${JetScape_INCLUDE_DIRS})
endforeach()
