# Setup OpenCV
find_package(OpenCV)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV is not found. OpenCV is needed to build example.")
endif()

# Build vgg16_example_in_c
add_executable(vgg16_example_in_c vgg16_example_in_c.c)
target_include_directories(vgg16_example_in_c PUBLIC "${PROJECT_SOURCE_DIR}/include" "${OpenCV_INCLUDE_DIRS}")
target_link_libraries(vgg16_example_in_c menoh)

# Build vgg16_example_in_cpp
add_executable(vgg16_example_in_cpp vgg16_example_in_cpp.cpp)
target_include_directories(vgg16_example_in_cpp PUBLIC "${PROJECT_SOURCE_DIR}/include" "${OpenCV_INCLUDE_DIRS}")
target_link_libraries(vgg16_example_in_cpp menoh ${OpenCV_LIBS})

# Buld general_cnn_example_in_cpp
add_executable(general_cnn_example_in_cpp general_cnn_example_in_cpp.cpp)
target_include_directories(general_cnn_example_in_cpp PUBLIC "${PROJECT_SOURCE_DIR}/include" "${OpenCV_INCLUDE_DIRS}")
target_link_libraries(general_cnn_example_in_cpp menoh ${OpenCV_LIBS})
