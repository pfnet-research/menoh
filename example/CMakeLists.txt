# Setup OpenCV
find_package(OpenCV)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV is not found. OpenCV is needed to build example.")
endif()
include_directories("../inlcude" ${OpenCV_INCLUDE_DIRS})

# Build vgg16_example_in_c
add_executable(vgg16_example_in_c vgg16_example_in_c.c)
target_link_libraries(vgg16_example_in_c menoh)

# Build vgg16_example_in_cpp
add_executable(vgg16_example_in_cpp vgg16_example_in_cpp.cpp)
target_link_libraries(vgg16_example_in_cpp menoh ${OpenCV_LIBS})

# Buld general_cnn_example_in_cpp
add_executable(general_cnn_example_in_cpp general_cnn_example_in_cpp.cpp)
target_link_libraries(general_cnn_example_in_cpp menoh ${OpenCV_LIBS})

# Build vgg19_example_in_cpp
add_executable(vgg19_example_in_cpp vgg19_example_in_cpp.cpp)
target_link_libraries(vgg19_example_in_cpp menoh ${OpenCV_LIBS})

# Build resnet50_example_in_cpp
add_executable(resnet50_example_in_cpp resnet50_example_in_cpp.cpp)
target_link_libraries(resnet50_example_in_cpp menoh ${OpenCV_LIBS})

# Build inception_v2_example_in_cpp
add_executable(inception_v2_example_in_cpp inception_v2_example_in_cpp.cpp)
target_link_libraries(inception_v2_example_in_cpp menoh ${OpenCV_LIBS})
