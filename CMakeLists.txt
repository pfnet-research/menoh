cmake_minimum_required(VERSION 3.1)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(MENOH_MAJOR_VERSION 1)
set(MENOH_MINOR_VERSION 0)
set(MENOH_PATCH_VERSION 1)

# Options
option(BUILD_SHARED_LIBS "Build shared libs" ON)
option(ENABLE_TEST "Build test" OFF)
option(ENABLE_BENCHMARK "Build benchmark" ON)
option(ENABLE_EXAMPLE "Build example" ON)
option(ENABLE_TOOL "Build tool" ON)

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
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -march=native")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # using GCC
    set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -pg -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -s -DNDEBUG -march=native")
endif()

include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/include")

set(EXTERNAL_DIR ${CMAKE_SOURCE_DIR}/external)

# ONNX setup
set(ONNX_DIR ${EXTERNAL_DIR}/onnx)
set(Protobuf_PROTOC_EXECUTABLE protoc CACHE STRING "protoc path")
include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})
execute_process(COMMAND git submodule update --init -- ${ONNX_DIR} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
execute_process(COMMAND ${Protobuf_PROTOC_EXECUTABLE} -I=${ONNX_DIR} --cpp_out=${ONNX_DIR} ${ONNX_DIR}/onnx/onnx.proto WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set(ONNX_PROTO_HEADER ${ONNX_DIR}/onnx/onnx.pb.h)
set(ONNX_PROTO_SRC ${ONNX_DIR}/onnx/onnx.pb.cc)
include_directories(${EXTERNAL_DIR})
include_directories(${ONNX_DIR})

# MKLDNN setup
find_package(MKLDNN "0.14")
if (NOT MKLDNN_FOUND)
    message(FATAL_ERROR "MKLDNN is not found")
endif()

include_directories("${MKLDNN_INCLUDE_DIR}")

if(${ENABLE_TEST})
    enable_testing()
    # GTest setup
    set(GTEST_DIR test/lib/googletest)
    execute_process(COMMAND git submodule update --init -- ${GTEST_DIR} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    add_subdirectory(${GTEST_DIR})
    add_subdirectory(test)
endif()

if(${ENABLE_BENCHMARK})
    add_subdirectory(benchmark)
endif()

if(${ENABLE_EXAMPLE})
    add_subdirectory(example)
endif()

if(${ENABLE_TOOL})
    add_subdirectory(tool)
endif()

add_subdirectory(menoh)
add_subdirectory(include)
