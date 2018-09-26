# Options
option(LINK_STATIC_LIBGCC "Link static libgcc to libmenoh" OFF)
option(LINK_STATIC_LIBSTDCXX "Link static libstdc++ to libmenoh" OFF)

if(NOT DEFINED ONNX_PROTO_SRC)
    message(FATAL_ERROR "ONNX_PROTO_SRC is not found")
endif()
if(NOT DEFINED ONNX_PROTO_HEADER)
    message(FATAL_ERROR "ONNX_PROTO_HEADER is not found")
endif()

# Note: The libraries can be static (.a) or shared (.so)
if (ENABLE_MKLDNN)
    if(NOT DEFINED MKLDNN_LIBRARIES)
        message(FATAL_ERROR "MKLDNN_LIBRARIES is not found")
    endif()
endif()

if(NOT DEFINED PROTOBUF_LIBRARIES)
    message(FATAL_ERROR "PROTOBUF_LIBRARIES is not found")
endif()

if (ENABLE_MKLDNN)
    file(GLOB MKLDNN_SOURCES "mkldnn/*.cpp")
endif()

if (ENABLE_TENSORRT)
    add_definitions(-DENABLE_TENSORRT -I /usr/local/cuda/include -I .)
    file(GLOB TENSORRT_SOURCES "tensorrt/*.cpp")
endif()

file(GLOB MENOH_SOURCES "./*.cpp")

set(SOURCES ${ONNX_PROTO_SRC} ${ONNX_PROTO_HEADER} ${MKLDNN_SOURCES} ${TENSORRT_SOURCES} ${MENOH_SOURCES})

# Create a object library for generating shared library
add_library(menoh_objlib OBJECT ${SOURCES})

add_dependencies(menoh_objlib gen_onnx_outputs)
target_sources(menoh_objlib PRIVATE ${ONNX_PROTO_SRC})

set_target_properties(menoh_objlib PROPERTIES POSITION_INDEPENDENT_CODE ON)

include(ConfigureMenoh)

# menoh
add_library(menoh SHARED $<TARGET_OBJECTS:menoh_objlib>)
if(NOT APPLE AND NOT MSVC)
    # Remove private symbols (Note: it works in MINGW but not in MSVC)
    set_property(
        TARGET menoh APPEND_STRING PROPERTY
            LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/menoh.map")
endif()
menoh_link_libraries(menoh PRIVATE)

# menoh_test_target: only used in `test` subdirectory
add_library(menoh_test_target SHARED $<TARGET_OBJECTS:menoh_objlib>)
menoh_link_libraries(menoh_test_target PRIVATE)

set_source_files_properties(${ONNX_PROTO_SRC} PROPERTIES GENERATED TRUE)
set_source_files_properties(${ONNX_PROTO_HEADER} PROPERTIES GENERATED TRUE)

install(TARGETS menoh
    RUNTIME DESTINATION "bin"
    LIBRARY DESTINATION "lib"
    ARCHIVE DESTINATION "lib")

configure_file("version.h.in" "version.h")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/version.h" DESTINATION "include/menoh")
