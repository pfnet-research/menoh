cmake_minimum_required(VERSION 3.1)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(MENOH_MAJOR_VERSION 1)
set(MENOH_MINOR_VERSION 1)
set(MENOH_PATCH_VERSION 0)

# Options
option(ENABLE_MKLDNN    "Use MKLDNN"      OFF)
option(ENABLE_TENSORRT  "Use TensorRT"    ON)

option(BUILD_SHARED_LIBS "Build shared libs" ON)

option(USE_OLD_GLIBCXX_ABI "Generate binaries for the old libstdc++ ABI" OFF)
option(LINK_STATIC_LIBPROTOBUF "Link static libprotobuf to libmenoh" OFF)

option(ENABLE_TEST      "Build test"      OFF)
option(ENABLE_BENCHMARK "Build benchmark" ON)
option(ENABLE_EXAMPLE   "Build example"   ON)

option(SHOW_ALL_VARIABLES "Debug: show all variables" OFF)

# C++ setup
set(CMAKE_CXX_STANDARD 14)
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE is unset, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang") # check Clang and AppleClang
    # using Clang
    set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -pg -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # using GCC
  if(ENABLE_TENSORRT)
    set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -pg -Wall -Wextra -DTENSORRT_DEBUG")
  else()
    set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -pg -Wall -Wextra")
  endif()
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -s -DNDEBUG")
endif()

# Configure to use the new `libstdc++` ABI
if(USE_OLD_GLIBCXX_ABI)
    message(STATUS "Set _GLIBCXX_USE_CXX11_ABI macro to 0")
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
endif()

include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/include")

set(EXTERNAL_DIR ${CMAKE_SOURCE_DIR}/external)
set(DOWNLOAD_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/downloads CACHE PATH "A directory to save the downloaded files.")
mark_as_advanced(DOWNLOAD_LOCATION)

# Enable ExternalProject_Add
include(ExternalProject)

# Setup protobuf
include(SetupProtobuf)

# Generate source codes from ONNX protobuf schema
include(GenerateOnnxSrc)

if(ENABLE_MKLDNN)
    # Setup MKLDNN
    find_package(MKLDNN "0.14")
    if (NOT MKLDNN_FOUND)
        message(FATAL_ERROR "MKLDNN is not found")
    endif()

    include_directories("${MKLDNN_INCLUDE_DIR}")

    if(${ENABLE_TEST})
        message(STATUS "# add_subdirectory\(test\)")
        add_subdirectory(test)
    endif()

    if(${ENABLE_BENCHMARK})
        message(STATUS "# add_subdirectory\(benchmark\)")
        add_subdirectory(benchmark)
    endif()

    if(${ENABLE_EXAMPLE})
        message(STATUS "# add_subdirectory\(example\)")
        add_subdirectory(example)
    endif()
endif()

if (ENABLE_TENSORRT)
    add_definitions(-DENABLE_TENSORRT -I /usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)

    if(ENABLE_BENCHMARK)
        message(STATUS "# add_subdirectory\(benchmark\)")
        add_subdirectory(benchmark)
    endif()

    if(${ENABLE_EXAMPLE})
        message(STATUS "# add_subdirectory\(example\)")
        add_subdirectory(example)
    endif()
endif()

message(STATUS "# add_subdirectory\(menoh\)")
add_subdirectory(menoh)

message(STATUS "# add_subdirectory\(include\)")
add_subdirectory(include)

if(SHOW_ALL_VARIABLES)
    get_cmake_property(_variableNames VARIABLES)
    list(SORT _variableNames)
    foreach(_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
endif()
