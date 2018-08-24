# Note: The libraries can be static (.a) or shared (.so)
if(NOT DEFINED PROTOBUF_LIBRARIES)
    message(FATAL_ERROR "PROTOBUF_LIBRARIES is not found")
endif()

add_executable(vgg16_benchmark vgg16_benchmark.cpp)
target_link_libraries(vgg16_benchmark ${PROTOBUF_LIBRARIES} menoh)
