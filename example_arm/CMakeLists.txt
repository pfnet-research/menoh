# Setup OpenCV
find_package(OpenCV)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV is not found. OpenCV is needed to build example.")
endif()
include_directories("../inlcude" ${OpenCV_INCLUDE_DIRS})

# Build vgg16_example_in_c
add_executable(vgg16_example_in_c_arm vgg16_example_in_c_arm.c)
target_link_libraries(vgg16_example_in_c_arm menoh ${OpenCV_LIBS})

# Build vgg16_example_in_cpp
add_executable(vgg16_example_in_cpp_arm vgg16_example_in_cpp_arm.cpp)
target_link_libraries(vgg16_example_in_cpp_arm menoh ${OpenCV_LIBS})


