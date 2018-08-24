# Note: The libraries can be static (.a) or shared (.so)
if(NOT DEFINED Protobuf_LIBRARIES)
    message(FATAL_ERROR "Protobuf_LIBRARIES is not found")
endif()

add_executable(vgg16_benchmark vgg16_benchmark.cpp)
target_link_libraries(vgg16_benchmark ${Protobuf_LIBRARIES} menoh)
