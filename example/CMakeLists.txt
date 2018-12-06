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

# Build vgg19_example_in_cpp
add_executable(vgg19_example_in_cpp vgg19_example_in_cpp.cpp)
target_include_directories(vgg19_example_in_cpp PUBLIC "${PROJECT_SOURCE_DIR}/include" "${OpenCV_INCLUDE_DIRS}")
target_link_libraries(vgg19_example_in_cpp menoh ${OpenCV_LIBS})

# Build resnet50_example_in_cpp
add_executable(resnet50_example_in_cpp resnet50_example_in_cpp.cpp)
target_include_directories(resnet50_example_in_cpp PUBLIC "${PROJECT_SOURCE_DIR}/include" "${OpenCV_INCLUDE_DIRS}")
target_link_libraries(resnet50_example_in_cpp menoh ${OpenCV_LIBS})

# Build squeezenet_example_in_cpp
add_executable(squeezenet_example_in_cpp squeezenet_example_in_cpp.cpp)
target_include_directories(squeezenet_example_in_cpp PUBLIC "${PROJECT_SOURCE_DIR}/include" "${OpenCV_INCLUDE_DIRS}")
target_link_libraries(squeezenet_example_in_cpp menoh ${OpenCV_LIBS})

# Build inception_v1_example_in_cpp
#add_executable(inception_v1_example_in_cpp inception_v1_example_in_cpp.cpp)
#target_link_libraries(inception_v1_example_in_cpp menoh ${OpenCV_LIBS})

# Build inception_v2_example_in_cpp
#add_executable(inception_v2_example_in_cpp inception_v2_example_in_cpp.cpp)
#target_link_libraries(inception_v2_example_in_cpp menoh ${OpenCV_LIBS})

# Build densnet121_example_in_cpp
#add_executable(densnet121_example_in_cpp densnet121_example_in_cpp.cpp)
#target_link_libraries(densnet121_example_in_cpp menoh ${OpenCV_LIBS})

# Build shufflenet_example_in_cpp
#add_executable(shufflenet_example_in_cpp shufflenet_example_in_cpp.cpp)
#target_link_libraries(shufflenet_example_in_cpp menoh ${OpenCV_LIBS})
