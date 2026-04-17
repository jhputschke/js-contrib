cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

# --- Torch ---
if(NOT TORCH_FOUND)
  find_package(Torch REQUIRED)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# --- ROOT ---
if(NOT ROOT_FOUND)
  find_package(ROOT REQUIRED)
endif()
include(${ROOT_USE_FILE})

# --- Sources ---
set(FNO_SOURCES
  src/FnoHydro.cc
  src/FnoHydro_beta.cc
  src/FnoRooIn.cc
  src/FnoRooIn_beta.cc
  src/PGunFno.cc
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${JetScape_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)

ROOT_GENERATE_DICTIONARY(FnoModuleDict LINKDEF src/FnoModuleLinkDef.h)

add_library(FnoModule SHARED ${FNO_SOURCES} FnoModuleDict.cxx)
target_link_libraries(FnoModule
  JetScape
  "${TORCH_LIBRARIES}"
  ${ROOT_LIBRARIES}
)
set_property(TARGET FnoModule PROPERTY CXX_STANDARD 17)

message(STATUS "FnoHydro: building FnoModule shared library")

# --- root_bulk utilities ---
add_subdirectory(root_bulk)

# --- Optional: examples ---
option(BUILD_FNO_HYDRO_EXAMPLES "Build FnoHydro examples" ON)
if(BUILD_FNO_HYDRO_EXAMPLES)
  add_subdirectory(example)
endif()
