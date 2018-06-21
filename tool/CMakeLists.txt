add_executable(onnx_viewer onnx_viewer.cpp)
target_link_libraries(onnx_viewer menoh)
set_target_properties(onnx_viewer PROPERTIES OUTPUT_NAME "menoh_onnx_viewer")
