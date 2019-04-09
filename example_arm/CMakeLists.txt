# Setup OpenCV
if(ENABLE_ANDROID_AARCH64)
    set(OpenCV_INCLUDE_DIRS ${ARM_HOME}/armnn/OpenCV-2.4.9/include)
    set(OpenCV_LIBRARY_DIRS ${ARM_HOME}/armnn/OpenCV-2.4.9/lib/arm64-v8a)
else()
    find_package(OpenCV)
    if (NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV is not found. OpenCV is needed to build example.")
    endif()
endif()

include_directories("../inlcude")
include_directories(${OpenCV_INCLUDE_DIRS})

#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -I ${OpenCV_INCLUDE_DIRS}")

# Build vgg16_example_in_c
add_executable(vgg16_example_in_c_arm vgg16_example_in_c_arm.c)
target_link_libraries(vgg16_example_in_c_arm menoh ${OpenCV_LIBS})

# Build vgg16_example_in_cpp
add_executable(vgg16_example_in_cpp_arm vgg16_example_in_cpp_arm.cpp)
target_link_libraries(vgg16_example_in_cpp_arm menoh ${OpenCV_LIBS})

# Build vgg19_example_in_cpp
add_executable(vgg19_example_in_cpp vgg19_example_in_cpp.cpp)
target_link_libraries(vgg19_example_in_cpp menoh ${OpenCV_LIBS})

# Build resnet50_example_in_cpp
#add_executable(resnet50_example_in_cpp resnet50_example_in_cpp.cpp)
#target_link_libraries(resnet50_example_in_cpp menoh ${OpenCV_LIBS})

# Build squeezenet_example_in_cpp
#add_executable(squeezenet_example_in_cpp squeezenet_example_in_cpp.cpp)
#target_link_libraries(squeezenet_example_in_cpp menoh ${OpenCV_LIBS})


