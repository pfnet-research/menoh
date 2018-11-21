include(ExternalProject)
set(PROTOBUF_VERSION_STATIC "3.6.1")
set(PROTOBUF_HASH MD5=406d5b8636576b1c86730ca5cbd1e576)

set(PROTOBUF_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf-${PROTOBUF_VERSION_STATIC})
set(PROTOBUF_URL "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION_STATIC}/protobuf-cpp-${PROTOBUF_VERSION_STATIC}.tar.gz")

# Requires `-fPIC` for linking with a shared library
set(PROTOBUF_CFLAGS "-g -O2 -fPIC")
set(PROTOBUF_CXXFLAGS "-g -O2 -fPIC")
if(USE_OLD_GLIBCXX_ABI)
    set(PROTOBUF_CXXFLAGS "${PROTOBUF_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

ExternalProject_Add(Protobuf
    PREFIX ${PROTOBUF_DIR}
    URL ${PROTOBUF_URL}
    URL_HASH ${PROTOBUF_HASH}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND bash -ex ${CMAKE_MODULE_PATH}/configure-helper.sh --CC ${CMAKE_C_COMPILER} --CXX ${CMAKE_CXX_COMPILER} -- "--prefix=${PROTOBUF_DIR}" "CFLAGS=${PROTOBUF_CFLAGS}" "CXXFLAGS=${PROTOBUF_CXXFLAGS}"
    BUILD_COMMAND make -j4
    INSTALL_COMMAND make install
)

set(PROTOBUF_LIBRARY_STATIC ${PROTOBUF_DIR}/lib/libprotobuf.a)
set(PROTOBUF_LIBRARY_SHARED ${PROTOBUF_DIR}/lib/libprotobuf.so)

# Mimic the behavior of `FindProtobuf` module
# Use the old variable names to ensure backward compatibility
set(PROTOBUF_INCLUDE_DIR ${PROTOBUF_DIR}/include)
set(PROTOBUF_INCLUDE_DIRS ${PROTOBUF_INCLUDE_DIR})
set(PROTOBUF_LIBRARY ${PROTOBUF_LIBRARY_STATIC}) # use the static library
set(PROTOBUF_LIBRARIES ${PROTOBUF_LIBRARY})
set(PROTOBUF_PROTOC_EXECUTABLE ${PROTOBUF_DIR}/bin/protoc)
set(PROTOBUF_FOUND TRUE)

# configure protobuf::libprotobuf
add_library(protobuf::libprotobuf UNKNOWN IMPORTED)
# Note: INTERFACE_INCLUDE_DIRECTORIES can't set in this place because include/ is
# not installed during executing `cmake`
set_target_properties(protobuf::libprotobuf PROPERTIES
    IMPORTED_LOCATION "${PROTOBUF_LIBRARY_STATIC}")
add_dependencies(protobuf::libprotobuf Protobuf)

# configure protobuf::protoc
add_executable(protobuf::protoc IMPORTED)
set_target_properties(protobuf::protoc PROPERTIES
    IMPORTED_LOCATION "${PROTOBUF_PROTOC_EXECUTABLE}")
add_dependencies(protobuf::protoc Protobuf)
