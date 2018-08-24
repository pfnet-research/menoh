# Note: The libraries can be static (.a) or shared (.so)
if(NOT DEFINED MKLDNN_LIBRARIES)
    message(FATAL_ERROR "MKLDNN_LIBRARIES is not found")
endif()
if(NOT DEFINED Protobuf_LIBRARIES)
    message(FATAL_ERROR "Protobuf_LIBRARIES is not found")
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
target_link_libraries(menoh_test PRIVATE gtest_main ${MKLDNN_LIBRARIES} ${Protobuf_LIBRARIES} menoh)
add_test(NAME menoh_test COMMAND menoh_test)
