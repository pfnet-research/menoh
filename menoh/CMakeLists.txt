option(LINK_STATIC_LIBGCC OFF)
option(LINK_STATIC_LIBSTDCXX OFF)

# Create a object library for generating shared library
add_library(menoh_objlib OBJECT)
set_target_properties(menoh_objlib PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_dependencies(menoh_objlib gen_onnx_proto) # custom target defined in onnx
target_sources(menoh_objlib PRIVATE
    array.cpp
    dims.cpp
    dtype.cpp
    graph.cpp
    menoh.cpp
    model_core.cpp
    model_core_factory.cpp
    node.cpp
    onnx.cpp
)
target_include_directories(menoh_objlib PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROTOBUF_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${ONNX_INCLUDE_DIRS}>
)

add_library(menoh $<TARGET_OBJECTS:menoh_objlib>)
if(NOT APPLE AND NOT MSVC)
    # Remove private symbols (Note: it works in MINGW but not in MSVC)
    set_target_properties(
        menoh PROPERTIES
            LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/menoh.map")
endif()
if(LINK_STATIC_LIBGCC)
    target_link_libraries(menoh PRIVATE -static-libgcc)
endif()
if(LINK_STATIC_LIBSTDCXX)
    target_link_libraries(menoh PRIVATE -static-libstdc++)
endif()
target_link_libraries(menoh PRIVATE onnx)

add_library(menoh_test_target $<TARGET_OBJECTS:menoh_objlib>)
target_link_libraries(menoh_test_target PRIVATE onnx)

if(ENABLE_MKLDNN)
    find_package(MKLDNN "0.14")
    if(NOT MKLDNN_FOUND)
        message(FATAL_ERROR "MKLDNN is not found")
    endif()
    target_sources(menoh_objlib PRIVATE
        composite_backend/model_core.cpp
        composite_backend/backend/generic/generic_context.cpp
        composite_backend/backend/mkldnn/memory_cache.cpp
        composite_backend/backend/mkldnn/mkldnn_context.cpp
        composite_backend/backend/mkldnn/memory_conversion.cpp
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
    )
    target_include_directories(menoh_objlib PUBLIC
        $<BUILD_INTERFACE:${MKLDNN_INCLUDE_DIR}>)

    target_link_libraries(menoh PRIVATE ${MKLDNN_LIBRARIES})
    target_link_libraries(menoh_test_target PRIVATE ${MKLDNN_LIBRARIES})
endif()

if(ENABLE_TENSORRT)
    find_package(TensorRT REQUIRED)
    if(NOT TENSORRT_FOUND)
        message(FATAL_ERROR "TensorRT is not found")
    endif()
    find_package(CUDA REQUIRED)
    if(NOT CUDA_FOUND)
        message(FATAL_ERROR "CUDA is not found")
    endif()
    add_definitions(-DXXH_NAMESPACE=menoh_)
    target_sources(menoh_objlib PRIVATE
        hash/xxhash.c
        hash/hasher.cpp
        tensorrt/cuda_memory.cpp
        tensorrt/Exception.cpp
        tensorrt/Parser.cpp
        tensorrt/Inference.cpp
        tensorrt/model_core.cpp
    )
    target_include_directories(menoh_objlib PRIVATE
        "${CUDA_INCLUDE_DIRS}"
        "${TENSORRT_INCLUDE_DIRS}")

    target_link_libraries(menoh PRIVATE
        "${CUDA_LIBRARIES}"
        "${TENSORRT_LIBRARIES}")
    target_link_libraries(menoh_test_target PRIVATE
        "${CUDA_LIBRARIES}"
        "${TENSORRT_LIBRARIES}")
endif()

install(TARGETS menoh
    RUNTIME DESTINATION "bin"
    LIBRARY DESTINATION "lib"
    ARCHIVE DESTINATION "lib")

configure_file("version.h.in" "version.h")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/version.h" DESTINATION "include/menoh")
