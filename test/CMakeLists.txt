add_executable(menoh_test
    np_io.cpp
    array.cpp
    node.cpp
    graph.cpp
    onnx.cpp
    mkldnn.cpp

    operator.cpp
    model.cpp
    model_data.cpp
    #vgg16.cpp

)
target_link_libraries(menoh_test gtest_main menoh_test_target)

add_test(NAME menoh_test COMMAND menoh_test)
