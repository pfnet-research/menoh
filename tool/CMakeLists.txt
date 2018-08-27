# Note: The library can be static (.a) or shared (.so)
if(NOT DEFINED PROTOBUF_LIBRARIES)
    message(FATAL_ERROR "PROTOBUF_LIBRARIES is not found")
endif()

add_executable(onnx_viewer onnx_viewer.cpp)

# ONNX requires protobuf regardless of whether it is static and shared
target_link_libraries(onnx_viewer ${PROTOBUF_LIBRARIES} menoh)

set_target_properties(onnx_viewer PROPERTIES OUTPUT_NAME "menoh_onnx_viewer")
install(TARGETS onnx_viewer RUNTIME DESTINATION bin)
