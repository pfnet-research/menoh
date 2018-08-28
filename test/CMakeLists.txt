option(LINK_STATIC_LIBPROTOBUF "Link static libprotobuf to libmenoh" OFF)

add_executable(menoh_test
    np_io.cpp
    array.cpp
    node.cpp
    graph.cpp
    onnx.cpp
    mkldnn.cpp

    operator.cpp
    model.cpp
    #vgg16.cpp

)
target_link_libraries(menoh_test gtest_main menoh_test_target)

if(NOT LINK_STATIC_LIBPROTOBUF)
    if(DEFINED PROTOBUF_LIBRARIES)
        target_link_libraries(menoh_test ${PROTOBUF_LIBRARIES})
    else()
        message(FATAL_ERROR "PROTOBUF_LIBRARIES is not found")
    endif()
endif()

add_test(NAME menoh_test COMMAND menoh_test)
