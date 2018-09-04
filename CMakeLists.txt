cmake_minimum_required(VERSION 3.1)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(MENOH_MAJOR_VERSION 1)
set(MENOH_MINOR_VERSION 0)
set(MENOH_PATCH_VERSION 2)

# Options
option(ENABLE_MKLDNN          "Use MKLDNN"           OFF)
option(ENABLE_ARMNN           "Use ARM NN"           ON)
option(ENABLE_ARMNN_X86_64    "Use ARM NN on X64_64" ON)
option(ENABLE_ANDROID_AARCH64 "Make Android AArch64" OFF)
option(ENABLE_OPENCL          "Use OpenCL"           OFF)

option(BUILD_SHARED_LIBS      "Build shared libs" ON)
option(ENABLE_TEST            "Build test"        OFF)
option(ENABLE_BENCHMARK       "Build benchmark"   ON)
option(ENABLE_EXAMPLE         "Build example"     ON)
option(ENABLE_TOOL            "Build tool"        OFF)

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
    if (ENABLE_ANDROID_AARCH64)
      if (ENABLE_OPENCL) 
        set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -fPIE -fPIC -DANDROID_ARM -DENABLE_OPENCL")
      else()
        set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -fPIE -fPIC -DANDROID_ARM")
      endif()
      set(CMAKE_EXE_LINKER_FLAGS "-pie -llog")
    else()
      set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -march=native")
    endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # using GCC
    set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -pg -Wall -Wextra -DARM_DEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -s -DNDEBUG -march=native")
endif()

include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/include")

set(EXTERNAL_DIR ${CMAKE_SOURCE_DIR}/external)

# for ARM Environment
if (ENABLE_ARMNN)
    set(ARM_HOME   ~/ARM_HOME)
    set(ACL_HOME   ${ARM_HOME}/ComputeLibrary/v18.08)
    set(ARMNN_HOME ${ARM_HOME}/armnn/v18.08)
endif()

# for protobuf
if (ENABLE_ANDROID_AARCH64)
    message("-- Build for Android AArch64")
    include_directories(${ARM_HOME}/armnn/google/arm64_pb_install/include)
    link_directories(${ARM_HOME}/armnn/google/arm64_pb_install/lib)
    set(PROTOBUF_LIBRARY protobuf)
else()
    set(Protobuf_PROTOC_EXECUTABLE protoc CACHE STRING "protoc path")
    include(FindProtobuf)
    find_package(Protobuf 2.6.1 REQUIRED)
    include_directories(${PROTOBUF_INCLUDE_DIR})
endif()

# ONNX setup
set(ONNX_DIR ${EXTERNAL_DIR}/onnx)

execute_process(COMMAND git submodule update --init -- ${ONNX_DIR} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
execute_process(COMMAND ${Protobuf_PROTOC_EXECUTABLE} -I=${ONNX_DIR} --cpp_out=${ONNX_DIR} ${ONNX_DIR}/onnx/onnx.proto WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set(ONNX_PROTO_HEADER ${ONNX_DIR}/onnx/onnx.pb.h)
set(ONNX_PROTO_SRC ${ONNX_DIR}/onnx/onnx.pb.cc)
include_directories(${EXTERNAL_DIR})
include_directories(${ONNX_DIR})

if (ENABLE_MKLDNN)
    # MKLDNN setup

    find_package(MKLDNN "0.14")
    if (NOT MKLDNN_FOUND)
        message(FATAL_ERROR "MKLDNN is not found")
    endif()

    add_definitions(-DENABLE_MKLDNN)
    include_directories("${MKLDNN_INCLUDE_DIR}")

    if(ENABLE_TEST)
        enable_testing()
        # GTest setup
        set(GTEST_DIR test/lib/googletest)
        execute_process(COMMAND git submodule update --init -- ${GTEST_DIR} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
        add_subdirectory(${GTEST_DIR})
        add_subdirectory(test)
    endif()

    if(ENABLE_BENCHMARK)
        add_subdirectory(benchmark)
    endif()

    if(ENABLE_EXAMPLE)
        add_subdirectory(example)
    endif()

    if(ENABLE_TOOL)
        add_subdirectory(tool)
    endif()
endif()

if (ENABLE_ARMNN)

    add_definitions(-DENABLE_ARMNN)

    include_directories(${Boost_INCLUDE_DIR})
    link_directories(${Boost_LIBRARY_DIR})
    
    include_directories(${ACL_HOME}/arm_compute)
    include_directories(${ARMNN_HOME}/include)
    include_directories(${ARMNN_HOME}/src)
    
    if(ENABLE_ANDROID_AARCH64)
        include_directories(${ARM_HOME}/armnn/boost/install/include)
        link_directories(${ARM_HOME}/armnn/boost/install/lib)
	
        if(ENABLE_OPENCL)
            set(ACL_LIBRARY_DIR ${ACL_HOME}/build_android_aarch64_opencl)
            set(ARMNN_LIBRARY_DIR ${ARMNN_HOME}/build_android_aarch64_opencl)
        else()
            set(ACL_LIBRARY_DIR ${ACL_HOME}/build_android_aarch64)
            set(ARMNN_LIBRARY_DIR ${ARMNN_HOME}/build_android_aarch64)
        endif()
    else()
        find_package(Boost 1.60 REQUIRED)
        if (NOT Boost_FOUND)
            message(FATAL_ERROR "Boost is not found")
        endif()

        if (ENABLE_ARMNN_X86_64) 
            set(ACL_LIBRARY_DIR ${ACL_HOME}/build_x86_64)
            set(ARMNN_LIBRARY_DIR ${ARMNN_HOME}/build_x86_64)
        else()
            set(ACL_LIBRARY_DIR ${ACL_HOME}/build)
            set(ARMNN_LIBRARY_DIR ${ARMNN_HOME}/build)
        endif()
    endif()

    link_directories(${ACL_LIBRARY_DIR})
    link_directories(${ARMNN_LIBRARY_DIR})

    set(ARMNN_LIBRARY armnn)
    set(ACL_LIBRARY   arm_compute-static arm_compute_core-static arm_compute_graph-static)

    if(ENABLE_BENCHMARK)
        add_subdirectory(benchmark)
    endif()

    if(ENABLE_EXAMPLE AND !ENABLE_ANDROID_AARCH64)
#    if(ENABLE_EXAMPLE)
        add_subdirectory(example_arm)
    endif()
endif()

add_subdirectory(menoh)
add_subdirectory(include)
