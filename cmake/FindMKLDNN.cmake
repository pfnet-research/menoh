find_path(MKLDNN_INCLUDE_DIR
    NAMES mkldnn.hpp PATH_SUFFIXES mkldnn
    PATHS
        /usr/include
        /usr/local/include)
find_library(MKLDNN_LIBRARY
    NAMES mkldnn
    PATHS
        /usr/lib
        /usr/local/lib)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKLDNN DEFAULT_MSG
    MKLDNN_LIBRARY MKLDNN_INCLUDE_DIR)
set(MKLDNN_INCLUDE_DIRS ${MKLDNN_INCLUDE_DIR})
set(MKLDNN_LIBRARIES ${MKLDNN_LIBRARY})
