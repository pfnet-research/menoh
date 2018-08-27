option(LINK_STATIC_LIBPROTOBUF "Link static libprotobuf to libmenoh" OFF)

add_executable(vgg16_benchmark vgg16_benchmark.cpp)
target_link_libraries(vgg16_benchmark menoh)

if(NOT LINK_STATIC_LIBPROTOBUF)
    if(DEFINED PROTOBUF_LIBRARIES)
        target_link_libraries(vgg16_benchmark ${PROTOBUF_LIBRARIES})
    else()
        message(FATAL_ERROR "PROTOBUF_LIBRARIES is not found")
    endif()
endif()
