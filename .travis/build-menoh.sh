#!/bin/bash

# retrieve arguments
while [[ $# != 0 ]]; do
    case $1 in
        --)
            shift
            break
            ;;
        --build-type)
            readonly ARG_BUILD_TYPE="$2"
            shift 2
            ;;
        --source-dir)
            readonly ARG_SOURCE_DIR="$2"
            shift 2
            ;;
        --build-dir)
            readonly ARG_BUILD_DIR="$2"
            shift 2
            ;;
        --install-dir)
            readonly ARG_INSTALL_DIR="$2"
            shift 2
            ;;
        --mkldnn-dir)
            readonly ARG_MKLDNN_DIR="$2"
            shift 2
            ;;
        --link-static-libgcc)
            readonly ARG_LINK_STATIC_LIBGCC="$2"
            shift 2
            ;;
        --link-static-libstdcxx)
            readonly ARG_LINK_STATIC_LIBSTDCXX="$2"
            shift 2
            ;;
        --link-static-libprotobuf)
            readonly ARG_LINK_STATIC_LIBPROTOBUF="$2"
            shift 2
            ;;
        -*)
            echo Unknown option \"$1\" 1>&2
            exit
            ;;
        *)
            break
            ;;

    esac
done

# validate the arguments
test -n "${ARG_SOURCE_DIR}" || { echo "--source-dir is not specified" 1>&2; exit 1; }

# options that have default value
test -n "${ARG_BUILD_DIR}" || readonly ARG_BUILD_DIR=${ARG_SOURCE_DIR}/build
test -n "${ARG_BUILD_TYPE}" || readonly ARG_BUILD_TYPE=Debug
test -n "${ARG_LINK_STATIC_LIBGCC}" || readonly ARG_LINK_STATIC_LIBGCC='OFF'
test -n "${ARG_LINK_STATIC_LIBSTDCXX}" || readonly ARG_LINK_STATIC_LIBSTDCXX='OFF'
test -n "${ARG_LINK_STATIC_LIBPROTOBUF}" || readonly ARG_LINK_STATIC_LIBPROTOBUF='OFF'

echo -e "\e[33;1mBuilding Menoh\e[0m"

[ -d "${ARG_BUILD_DIR}" ] || mkdir -p ${ARG_BUILD_DIR}

cd ${ARG_BUILD_DIR}
if [ -n "${ARG_INSTALL_DIR}" ]; then
    OPT_CMAKE_INSTALL_PREFIX=-DCMAKE_INSTALL_PREFIX=${ARG_INSTALL_DIR}
fi
if [ -n "${ARG_MKLDNN_DIR}" ]; then
    OPT_MKLDNN_INCLUDE_DIR=-DMKLDNN_INCLUDE_DIR=${ARG_MKLDNN_DIR}/include
    OPT_MKLDNN_LIBRARY=-DMKLDNN_LIBRARY=${ARG_MKLDNN_DIR}/lib/libmkldnn.so
fi
cmake \
    -DCMAKE_BUILD_TYPE=${ARG_BUILD_TYPE} \
    ${OPT_CMAKE_INSTALL_PREFIX} \
    ${OPT_MKLDNN_INCLUDE_DIR} \
    ${OPT_MKLDNN_LIBRARY} \
    -DLINK_STATIC_LIBGCC=${ARG_LINK_STATIC_LIBGCC} \
    -DLINK_STATIC_LIBSTDCXX=${ARG_LINK_STATIC_LIBSTDCXX} \
    -DLINK_STATIC_LIBPROTOBUF=${ARG_LINK_STATIC_LIBPROTOBUF} \
    -DENABLE_TEST=ON \
    ${ARG_SOURCE_DIR}

make
