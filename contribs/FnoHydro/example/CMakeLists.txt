cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

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
