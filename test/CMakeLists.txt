enable_testing()

# GTest setup
set(GTEST_DIR "lib/googletest")
execute_process(COMMAND git submodule update --init -- test/${GTEST_DIR} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
message(STATUS "Adding ${GTEST_DIR}")
add_subdirectory(${GTEST_DIR} EXCLUDE_FROM_ALL)

# filesystem setup
set(FILESYSTEM_DIR "lib/filesystem")
execute_process(COMMAND git submodule update --init -- test/${FILESYSTEM_DIR} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
include_directories("${FILESYSTEM_DIR}")

add_executable(menoh_test
    np_io.cpp
    array.cpp
    node.cpp
    graph.cpp
    onnx.cpp

    operator.cpp
    mkldnn_with_generic_fallback_backend.cpp
    model_data.cpp
    attribute_completion_and_shape_inference.cpp
    #vgg16.cpp
)

target_include_directories(menoh_test PUBLIC $<BUILD_INTERFACE:${PROTOBUF_INCLUDE_DIRS}>)

if(LINK_STATIC_LIBPROTOBUF)
    target_link_libraries(menoh_test gtest_main menoh_test_target)
else()
    target_link_libraries(menoh_test gtest_main menoh_test_target ${PROTOBUF_LIBRARIES})
endif()

if(UNIX AND NOT APPLE)
    set_property(
        TARGET menoh_test APPEND_STRING PROPERTY
            LINK_FLAGS "-Wl,--disable-new-dtags")
endif()

add_test(NAME menoh_test COMMAND menoh_test)
