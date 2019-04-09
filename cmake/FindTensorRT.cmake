find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS
        /usr/include
        /usr/local/include
        /usr/local/TensorRT-5.0.2.6/include)
find_library(TENSORRT_LIBRARY
    NAMES nvinfer
    PATHS
        /usr/lib
        /usr/local/lib
        /usr/local/TensorRT-5.0.2.6/lib)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TENSORRT DEFAULT_MSG
	TENSORRT_LIBRARY TENSORRT_INCLUDE_DIR)
set(TENSORRT_INCLUDE_DIRS ${TENSORRT_INCLUDE_DIR})
set(TENSORRT_LIBRARIES ${TENSORRT_LIBRARY})
