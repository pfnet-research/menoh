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
if(UNIX AND NOT APPLE)
    set_property(
        TARGET menoh_test APPEND_STRING PROPERTY
            LINK_FLAGS "-Wl,--disable-new-dtags")
endif()

add_test(NAME menoh_test COMMAND menoh_test)
