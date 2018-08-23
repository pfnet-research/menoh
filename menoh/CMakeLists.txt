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
if(NOT DEFINED MKLDNN_LIBRARY)
    message(FATAL_ERROR "MKLDNN_LIBRARY is not found")
endif()
if(NOT DEFINED Protobuf_LIBRARY)
    message(FATAL_ERROR "Protobuf_LIBRARY is not found")
endif()

file(GLOB_RECURSE SOURCES "." "*.cpp")
add_library(menoh SHARED ${SOURCES} ${ONNX_PROTO_SRC} ${ONNX_PROTO_HEADER})

# configure MENOH_LINK_LIBRARIES
if(LINK_STATIC_LIBGCC)
    set(MENOH_LINK_LIBRARIES ${MENOH_LINK_LIBRARIES} -static-libgcc)
endif()
if(LINK_STATIC_LIBSTDCXX)
    set(MENOH_LINK_LIBRARIES ${MENOH_LINK_LIBRARIES} -static-libstdc++)
endif()
set(MENOH_LINK_LIBRARIES ${MENOH_LINK_LIBRARIES} ${MKLDNN_LIBRARY} ${Protobuf_LIBRARY})

message(STATUS "MENOH_LINK_LIBRARIES: ${MENOH_LINK_LIBRARIES}")
target_link_libraries(menoh PRIVATE ${MENOH_LINK_LIBRARIES})

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
