option(LINK_STATIC_LIBGCC OFF)
option(LINK_STATIC_LIBSTDCXX OFF)

## Setup MKLDNN
find_package(MKLDNN "0.14")
if(NOT MKLDNN_FOUND)
    message(FATAL_ERROR "MKLDNN is not found")
endif()

# Create a object library for generating shared library
add_library(menoh_objlib OBJECT
    dtype.cpp
    array.cpp
    onnx.cpp
    composite_backend/backend/mkldnn/memory_cache.cpp
    composite_backend/backend/mkldnn/mkldnn_context.cpp
    composite_backend/backend/mkldnn/memory_conversion.cpp
    composite_backend/backend/generic/generic_context.cpp
    composite_backend/model_core.cpp
    model_core_factory.cpp
    dims.cpp
    node.cpp
    graph.cpp
    mkldnn/utility.cpp
    mkldnn/operator/conv_transpose.cpp
    mkldnn/operator/softmax.cpp
    mkldnn/operator/gemm.cpp
    mkldnn/operator/eltwise.cpp
    mkldnn/operator/lrn.cpp
    mkldnn/operator/conv.cpp
    mkldnn/operator/sum.cpp
    mkldnn/operator/add.cpp
    mkldnn/operator/concat.cpp
    mkldnn/operator/pool.cpp
    mkldnn/operator/fc.cpp
    mkldnn/operator/batch_norm.cpp
    mkldnn/model_core.cpp
    menoh.cpp
    model_core.cpp
)
set_target_properties(menoh_objlib PROPERTIES POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS})
add_dependencies(menoh_objlib gen_onnx_proto) # custom target defined in onnx
target_include_directories(menoh_objlib PUBLIC
    $<BUILD_INTERFACE:${PROTOBUF_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${ONNX_INCLUDE_DIRS}>
)

# menoh
add_library(menoh $<TARGET_OBJECTS:menoh_objlib>)
if(NOT APPLE AND NOT MSVC)
    # Remove private symbols (Note: it works in MINGW but not in MSVC)
    set_property(
        TARGET menoh APPEND_STRING PROPERTY
            LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/menoh.map")
endif()
target_link_libraries(menoh PUBLIC ${MKLDNN_LIBRARIES} onnx)
if(LINK_STATIC_LIBGCC)
    target_link_libraries(menoh PRIVATE -static-libgcc)
endif()
if(LINK_STATIC_LIBSTDCXX)
    target_link_libraries(menoh PRIVATE -static-libstdc++)
endif()

# menoh_test_target: only used in `test` subdirectory
add_library(menoh_test_target $<TARGET_OBJECTS:menoh_objlib>)
target_link_libraries(menoh_test_target PUBLIC ${MKLDNN_LIBRARIES} onnx)

install(TARGETS menoh
    RUNTIME DESTINATION "bin"
    LIBRARY DESTINATION "lib"
    ARCHIVE DESTINATION "lib")

configure_file("version.h.in" "version.h")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/version.h" DESTINATION "include/menoh")
