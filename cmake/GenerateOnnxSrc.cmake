set(ONNX_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/onnx)

if(NOT EXISTS "${ONNX_OUTPUT_DIR}")
    file(MAKE_DIRECTORY "${ONNX_OUTPUT_DIR}")
endif()

set(ONNX_SRC_DIR ${EXTERNAL_DIR}/onnx)
execute_process(COMMAND git submodule update --init -- ${ONNX_SRC_DIR} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} RESULT_VARIABLE TEST_ERROR)
if(TEST_ERROR)
    message(FATAL_ERROR "`git submodule update --init ${ONNX_SRC_DIR}` failed, returned ${TEST_ERROR}")
endif()

set(ONNX_PROTO_HEADER ${ONNX_OUTPUT_DIR}/onnx/onnx.pb.h)
set(ONNX_PROTO_SRC ${ONNX_OUTPUT_DIR}/onnx/onnx.pb.cc)

set(ONNX_GENERATED_OUTPUTS ${ONNX_PROTO_HEADER} ${ONNX_PROTO_SRC})

add_custom_target(gen_onnx_outputs DEPENDS ${ONNX_GENERATED_OUTPUTS})
add_custom_command(
    OUTPUT ${ONNX_GENERATED_OUTPUTS}
    COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
    ARGS -I ${ONNX_SRC_DIR} --cpp_out . ${ONNX_SRC_DIR}/onnx/onnx.proto
    DEPENDS ${PROTOBUF_PROTOC_EXECUTABLE} ${ONNX_SRC_DIR}/onnx/onnx.proto
    COMMENT "Generating ONNX source files"
    WORKING_DIRECTORY ${ONNX_OUTPUT_DIR}
    VERBATIM)

include_directories(${ONNX_OUTPUT_DIR}) # for ONNX_PROTO_HEADER
