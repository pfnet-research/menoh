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
if(NOT DEFINED MKLDNN_LIBRARIES)
    message(FATAL_ERROR "MKLDNN_LIBRARIES is not found")
endif()
if(NOT DEFINED PROTOBUF_LIBRARIES)
    message(FATAL_ERROR "PROTOBUF_LIBRARIES is not found")
endif()

file(GLOB_RECURSE SOURCES "." "*.cpp")
add_library(menoh SHARED ${SOURCES} ${ONNX_PROTO_SRC} ${ONNX_PROTO_HEADER})

# Link libraries to menoh
if(LINK_STATIC_LIBGCC)
    target_link_libraries(menoh PRIVATE -static-libgcc)
endif()
if(LINK_STATIC_LIBSTDCXX)
    target_link_libraries(menoh PRIVATE -static-libstdc++)
endif()
# PUBLIC will add transitive dependencies (`mklml_intel` and `iomp5`) to the link interface
# Note: change it to PRIVATE after building mkldnn itself
target_link_libraries(menoh PUBLIC ${MKLDNN_LIBRARIES})
target_link_libraries(menoh PRIVATE ${PROTOBUF_LIBRARIES})

set_source_files_properties(${ONNX_PROTO_SRC} PROPERTIES GENERATED TRUE)
set_source_files_properties(${ONNX_PROTO_HEADER} PROPERTIES GENERATED TRUE)

# remove private symbols
if(NOT APPLE)
    set_property(
        TARGET menoh APPEND_STRING PROPERTY
            LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/menoh.map")
endif()

install(TARGETS menoh
    RUNTIME DESTINATION "bin"
    LIBRARY DESTINATION "lib"
    ARCHIVE DESTINATION "lib")

configure_file("version.h.in" "version.h")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/version.h" DESTINATION "include/menoh")
