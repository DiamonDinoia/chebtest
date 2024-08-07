cmake_minimum_required(VERSION 3.14)
project(
  chebtest
  LANGUAGES CXX
)

add_compile_options(-march=native)
set(CMAKE_CXX_STANDARD 23)

set (default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message (STATUS "No build type specified. Setting build type to Release.")
  set (CMAKE_BUILD_TYPE "Release" CACHE STRING "Valid options: Debug, RelWithDebInfo, Release" FORCE)
endif()

include(FetchContent)
FetchContent_Declare(
  nanobench
  GIT_REPOSITORY https://github.com/martinus/nanobench.git
  GIT_TAG v4.3.9
  GIT_SHALLOW TRUE)

FetchContent_Declare(
        xsimd
        GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
        GIT_TAG 13.0.0
        GIT_SHALLOW TRUE)


# Add vectorclass library
FetchContent_Declare(
  vectorclass
  GIT_REPOSITORY https://github.com/vectorclass/version2.git
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(nanobench)
FetchContent_GetProperties(vectorclass)

if(NOT vectorclass_POPULATED)
  FetchContent_Populate(vectorclass)
endif()


FetchContent_MakeAvailable(xsimd)

add_library(vectorclass INTERFACE)
target_include_directories(vectorclass INTERFACE ${vectorclass_SOURCE_DIR})

add_executable(chebtest src/benchmarks.cpp src/cheb.cpp)
target_link_libraries(chebtest PUBLIC nanobench xsimd vectorclass)
target_include_directories(chebtest PUBLIC)

add_executable(chebtest_ipo src/benchmarks.cpp src/cheb.cpp)
target_link_libraries(chebtest_ipo PUBLIC nanobench xsimd vectorclass)
target_include_directories(chebtest_ipo PUBLIC)
set_property(TARGET chebtest_ipo PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)