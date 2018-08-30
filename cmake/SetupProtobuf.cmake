set(PROTOBUF_VERSION "2.6.1")

if(LINK_STATIC_LIBPROTOBUF)
    # Note: We can't use `set(PROTOBUF_BUILD_SHARED_LIBS OFF)` in `FindProtobuf` module
    # because `libprotobuf.a` produced by the package manager is not PIC. So we need to
    # build it by ourselves.

    if(UNIX)
        set(PROTOBUF_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf-${PROTOBUF_VERSION})
        set(PROTOBUF_URL "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-${PROTOBUF_VERSION}.tar.gz")
        set(PROTOBUF_HASH MD5=f3916ce13b7fcb3072a1fa8cf02b2423)

        # Requires `-fPIC` for linking with a shared library
        set(PROTOBUF_CFLAGS -fPIC)
        set(PROTOBUF_CXXFLAGS -fPIC)
        if(USE_OLD_GLIBCXX_ABI)
            set(PROTOBUF_CXXFLAGS "${PROTOBUF_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        endif()

        ExternalProject_Add(Protobuf
            PREFIX ${PROTOBUF_DIR}
            URL ${PROTOBUF_URL}
            URL_HASH ${PROTOBUF_HASH}
            DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
            BUILD_IN_SOURCE 1
            CONFIGURE_COMMAND ./configure --prefix=${PROTOBUF_DIR} CFLAGS=${PROTOBUF_CFLAGS} CXXFLAGS=${PROTOBUF_CXXFLAGS}
            BUILD_COMMAND make
            INSTALL_COMMAND make install
        )

        set(PROTOBUF_LIBRARY_STATIC ${PROTOBUF_DIR}/lib/libprotobuf.a)
        set(PROTOBUF_LIBRARY_SHARED ${PROTOBUF_DIR}/lib/libprotobuf.so)

        # Mimic the behavior of `FindProtobuf` module
        # Use the old variable names to ensure backward compatibility
        set(PROTOBUF_INCLUDE_DIR ${PROTOBUF_DIR}/include)
        set(PROTOBUF_LIBRARY ${PROTOBUF_LIBRARY_STATIC}) # use the static library
        set(PROTOBUF_LIBRARIES ${PROTOBUF_LIBRARY})
        set(PROTOBUF_PROTOC_EXECUTABLE ${PROTOBUF_DIR}/bin/protoc)
        set(PROTOBUF_FOUND TRUE)

        add_library(protobuf::libprotobuf UNKNOWN IMPORTED)
        # Note: INTERFACE_INCLUDE_DIRECTORIES can't set in this place because include/ is
        # not installed during executing `cmake`
        set_target_properties(protobuf::libprotobuf PROPERTIES
            IMPORTED_LOCATION "${PROTOBUF_LIBRARY_STATIC}")
    else()
        message(FATAL_ERROR "LINK_STATIC_LIBPROTOBUF is supported only in UNIX-like environments")
    endif()
else()
    include(FindProtobuf)
    find_package(Protobuf ${PROTOBUF_VERSION} REQUIRED)
endif()

include_directories(${PROTOBUF_INCLUDE_DIR})
