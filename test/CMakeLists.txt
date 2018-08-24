# Note: The libraries can be static (.a) or shared (.so)
if(NOT DEFINED PROTOBUF_LIBRARIES)
    message(FATAL_ERROR "PROTOBUF_LIBRARIES is not found")
endif()

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
target_link_libraries(menoh_test gtest_main ${PROTOBUF_LIBRARIES} menoh)
add_test(NAME menoh_test COMMAND menoh_test)
