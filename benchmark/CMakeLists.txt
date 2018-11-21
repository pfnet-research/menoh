add_executable(vgg16_benchmark vgg16_benchmark.cpp)
target_include_directories(vgg16_benchmark PUBLIC "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(vgg16_benchmark menoh)
