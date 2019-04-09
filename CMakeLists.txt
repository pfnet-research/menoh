cmake_minimum_required(VERSION 3.1)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(MENOH_MAJOR_VERSION 1)
set(MENOH_MINOR_VERSION 2)
set(MENOH_PATCH_VERSION 0)

# Options

option(ENABLE_MKLDNN          "Use MKLDNN"           ON)
option(ENABLE_ARMNN           "Use ARM NN"           OFF)
option(ENABLE_ARMNN_X86_64    "Use ARM NN on X64_64" OFF)
option(ENABLE_ANDROID_AARCH64 "Make Android AArch64" OFF)
option(ENABLE_OPENCL          "Use OpenCL"           OFF)

option(BUILD_SHARED_LIBS      "Build shared libs" ON)
option(ENABLE_TEST            "Build test"        OFF)
option(ENABLE_BENCHMARK       "Build benchmark"   ON)
option(ENABLE_EXAMPLE         "Build example"     ON)
option(ENABLE_TOOL            "Build tool"        OFF)

option(USE_OLD_GLIBCXX_ABI "Generate binaries for the old libstdc++ ABI" OFF)
option(LINK_STATIC_LIBPROTOBUF "Link static libprotobuf to libmenoh" OFF)

option(ENABLE_TEST "Build test" OFF)
option(ENABLE_BENCHMARK "Build benchmark" ON)
option(ENABLE_EXAMPLE "Build example" ON)

option(BUILD_SHARED_LIBS "Build shared libs" ON)

set(INSTALL_PKGCONFIG_DIR "${CMAKE_INSTALL_PREFIX}/share/pkgconfig" CACHE PATH "installation path for pkg-config(.pc) file")

option(SHOW_ALL_VARIABLES "Debug: show all variables" OFF)

## C++ setup
set(CMAKE_CXX_STANDARD 14)

## Build type
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE is unset, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

## Compiler dependent options
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang") # check Clang and AppleClang
    # using Clang
    set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -Wall -Wextra -DARM_DEBUG")
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
    set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -Wall -Wextra -DARM_DEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -s -DNDEBUG -march=native")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # using GCC
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
        message(FATAL_ERROR "GCC is too old. Please use GCC version >= 4.9")
    endif()
    set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -pg -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -s -DNDEBUG")
endif()

## Configure to use the new `libstdc++` ABI
if(USE_OLD_GLIBCXX_ABI)
    message(STATUS "Set _GLIBCXX_USE_CXX11_ABI macro to 0")
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
endif()

set(EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)
set(DOWNLOAD_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/downloads CACHE PATH "A directory to save the downloaded files.")
mark_as_advanced(DOWNLOAD_LOCATION)

## Setup protobuf (it is used in ONNX and Menoh)
if(LINK_STATIC_LIBPROTOBUF)
    # Note: We can't use `set(PROTOBUF_BUILD_SHARED_LIBS OFF)` in `FindProtobuf` module
    # because `libprotobuf.a` produced by the package manager is not PIC. So we need to
    # build it by ourselves.
    if(NOT BUILD_SHARED_LIBS)
        message(FATAL_ERROR "LINK_STATIC_LIBPROTOBUF=ON is supported only when BUILD_SHARED_LIBS=ON")
    endif()
    if(UNIX OR MINGW)
        include(BuildProtobuf)
    else()
        message(FATAL_ERROR "LINK_STATIC_LIBPROTOBUF is supported only in UNIX-like environments")
    endif()
else()
    # Note: It may conflict with the loading mechanism in onnx's CMake configuration.
    # See external/onnx/CMakeLists.txt for more details.
    include(FindProtobuf)
    find_package(Protobuf ${PROTOBUF_VERSION} REQUIRED)

    # Note: `protobuf::protoc` is not supported in CMake 3.9.4 or older
    if(NOT TARGET protobuf::protoc)
        find_program(Protobuf_PROTOC_EXECUTABLE
            NAMES protoc
            DOC "The Google Protocol Buffers Compiler")
        add_executable(protobuf::protoc IMPORTED)
        if(EXISTS "${Protobuf_PROTOC_EXECUTABLE}")
            set_target_properties(protobuf::protoc PROPERTIES
                IMPORTED_LOCATION "${Protobuf_PROTOC_EXECUTABLE}")
        endif()
    endif()
endif()

## Setup git
find_package(Git)
if(NOT GIT_FOUND)
    message(FATAL_ERROR "git not found")
endif()

## Build libonnx.a
message(STATUS "Adding external/onnx")

# for ARM Environment
if (ENABLE_ARMNN)
    set(ARM_HOME   ~/ARM_HOME)
    set(ACL_HOME   ${ARM_HOME}/ComputeLibrary/v18.11)
    set(ARMNN_HOME ${ARM_HOME}/armnn/v18.11)
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

set(ONNX_SRC_DIR ${EXTERNAL_DIR}/onnx)
execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init -- ${ONNX_SRC_DIR} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# TODO: enable the following option when it is ready for migrating to onnx-ml
#set(ONNX_ML 1)

if (ENABLE_MKLDNN)
    set(BUILD_SHARED_LIBS_SAVED "${BUILD_SHARED_LIBS}")
    set(BUILD_SHARED_LIBS OFF)
    set(ONNX_NAMESPACE "menoh_onnx" CACHE INTERNAL "onnx namespace")
    add_subdirectory(${EXTERNAL_DIR}/onnx EXCLUDE_FROM_ALL) # Note: BUILD_SHARED_LIBS must be OFF in this place
    if(NOT ${BUILD_SHARED_LIBS_SAVED}) # install libmenoh_onnx and lib_menoh_onnx_proto
        set_target_properties(onnx_proto PROPERTIES OUTPUT_NAME "menoh_onnx_proto")
        set_target_properties(onnx PROPERTIES OUTPUT_NAME "menoh_onnx")
        find_library(MENOH_ONNX_PROTO_LIB "menoh_onnx_proto" "${CMAKE_CURRENT_BINARY_DIR}/external/onnx")
        find_library(MENOH_ONNX_LIB "menoh_onnx" "${CMAKE_CURRENT_BINARY_DIR}/external/onnx")
        install(FILES "${MENOH_ONNX_LIB}" "${MENOH_ONNX_PROTO_LIB}" DESTINATION "lib")
    endif()
    set(BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS_SAVED}")

    if(${ENABLE_TEST})
        message(STATUS "Adding test")
        enable_testing()
        add_subdirectory(test)
    endif()

    if(ENABLE_BENCHMARK)
        add_subdirectory(benchmark)
    endif()

    if(ENABLE_EXAMPLE)
        add_subdirectory(example)
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

    if(ENABLE_EXAMPLE)
        add_subdirectory(example_arm)
    endif()
endif()

## Build libmenoh
message(STATUS "Adding menoh")
add_subdirectory(menoh)

message(STATUS "Adding include")
add_subdirectory(include)

if(SHOW_ALL_VARIABLES)
    get_cmake_property(_variableNames VARIABLES)
    list(SORT _variableNames)
    foreach(_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
endif()
