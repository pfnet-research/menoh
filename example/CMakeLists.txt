# Setup OpenCV
find_package(OpenCV)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV is not found. OpenCV is needed to build example.")
endif()

# Build vgg16_example_in_c
add_executable(mkldnn_vgg16_example_in_c mkldnn_vgg16_example_in_c.c)
target_include_directories(mkldnn_vgg16_example_in_c PUBLIC "${PROJECT_SOURCE_DIR}/include" "${OpenCV_INCLUDE_DIRS}")
target_link_libraries(mkldnn_vgg16_example_in_c menoh)

# Build vgg16_example_in_cpp
add_executable(mkldnn_vgg16_example_in_cpp mkldnn_vgg16_example_in_cpp.cpp)
target_include_directories(mkldnn_vgg16_example_in_cpp PUBLIC "${PROJECT_SOURCE_DIR}/include" "${OpenCV_INCLUDE_DIRS}")
target_link_libraries(mkldnn_vgg16_example_in_cpp menoh ${OpenCV_LIBS})

# Buld general_cnn_example_in_cpp
add_executable(general_cnn_example_in_cpp general_cnn_example_in_cpp.cpp)
target_include_directories(general_cnn_example_in_cpp PUBLIC "${PROJECT_SOURCE_DIR}/include" "${OpenCV_INCLUDE_DIRS}")
target_link_libraries(general_cnn_example_in_cpp menoh ${OpenCV_LIBS})

# Build vgg19_onnx_example_in_cpp (vgg19, inceptionv1)
add_executable(vgg19_example_in_cpp vgg19_example_in_cpp.cpp)
target_include_directories(vgg19_example_in_cpp PUBLIC "${PROJECT_SOURCE_DIR}/include" "${OpenCV_INCLUDE_DIRS}")
target_link_libraries(vgg19_example_in_cpp menoh ${OpenCV_LIBS})

# Build resnet50_example_in_cpp (resnet50, shufflenet)
add_executable(resnet50_example_in_cpp resnet50_example_in_cpp.cpp)
target_include_directories(resnet50_example_in_cpp PUBLIC "${PROJECT_SOURCE_DIR}/include" "${OpenCV_INCLUDE_DIRS}")
target_link_libraries(resnet50_example_in_cpp menoh ${OpenCV_LIBS})

# Build squeezenet_example_in_cpp
add_executable(squeezenet_example_in_cpp squeezenet_example_in_cpp.cpp)
target_include_directories(squeezenet_example_in_cpp PUBLIC "${PROJECT_SOURCE_DIR}/include" "${OpenCV_INCLUDE_DIRS}")
target_link_libraries(squeezenet_example_in_cpp menoh ${OpenCV_LIBS})

