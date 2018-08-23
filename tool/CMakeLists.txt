if(NOT DEFINED Protobuf_LIBRARY)
    message(FATAL_ERROR "Protobuf_LIBRARY is not found")
endif()

add_executable(onnx_viewer onnx_viewer.cpp)
target_link_libraries(onnx_viewer ${Protobuf_LIBRARY} menoh)
set_target_properties(onnx_viewer PROPERTIES OUTPUT_NAME "menoh_onnx_viewer")
install(TARGETS onnx_viewer RUNTIME DESTINATION bin)
