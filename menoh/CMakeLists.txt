file(GLOB_RECURSE SOURCES "." "*.cpp")
add_library(menoh ${SOURCES} ${ONNX_PROTO_SRC} ${ONNX_PROTO_HEADER})
target_link_libraries(menoh ${MKLDNN_LIBRARY} ${PROTOBUF_LIBRARY})
set_source_files_properties(${ONNX_PROTO_SRC} PROPERTIES GENERATED TRUE)
set_source_files_properties(${ONNX_PROTO_HEADER} PROPERTIES GENERATED TRUE)

install(TARGETS menoh
    RUNTIME DESTINATION "bin"
    LIBRARY DESTINATION "lib"
    ARCHIVE DESTINATION "lib")

configure_file("version.h.in" "version.h")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/version.h" DESTINATION "include/menoh")
