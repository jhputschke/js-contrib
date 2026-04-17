cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# Place executables in the top-level build directory alongside runJetscape etc.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include(${ROOT_USE_FILE})

foreach(_exe bulkTest bulkRootWriter bulkRootWriterFull)
  add_executable(${_exe} ${CMAKE_CURRENT_SOURCE_DIR}/${_exe}.cc)
  target_link_libraries(${_exe} JetScape ${ROOT_LIBRARIES})
  target_include_directories(${_exe} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${JetScape_INCLUDE_DIRS}
  )
  ROOT_GENERATE_DICTIONARY(${_exe}_dict MODULE ${_exe}
    LINKDEF JetScapeRootBulkLinkDef.h)
endforeach()
