
if (ENABLE_MKLDNN)
    add_subdirectory(mkldnn)
    file(GLOB MKLDNN_SOURCES "mkldnn/*.cpp")
endif()  

if(ENABLE_ARMNN)
    add_subdirectory(armnn)
    file(GLOB ARM_SOURCES "armnn/*.cpp")
endif()

file(GLOB SOURCES "./*.cpp")

add_library(menoh ${SOURCES} ${ONNX_PROTO_SRC} ${ONNX_PROTO_HEADER} ${MKLDNN_SOURCES} ${ARM_SOURCES})

if(ENABLE_ARMNN)
  target_link_libraries(menoh ${ACL_LIBRARY})
  target_link_libraries(menoh ${ARMNN_LIBRARY})
  target_link_libraries(menoh boost_system boost_thread)
endif()

if (ENABLE_MKLDNN)
  target_link_libraries(menoh ${MKLDNN_LIBRARY})
endif()

target_link_libraries(menoh ${PROTOBUF_LIBRARY})

set_source_files_properties(${ONNX_PROTO_SRC}    PROPERTIES GENERATED TRUE)
set_source_files_properties(${ONNX_PROTO_HEADER} PROPERTIES GENERATED TRUE)

install(TARGETS menoh
    RUNTIME DESTINATION "bin"
    LIBRARY DESTINATION "lib"
    ARCHIVE DESTINATION "lib")

configure_file("version.h.in" "version.h")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/version.h" DESTINATION "include/menoh")

