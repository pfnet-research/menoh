#!/bin/bash -e

BASE_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)

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

# options that have default value
test -n "${ARG_BUILD_TYPE}" || readonly ARG_BUILD_TYPE=Debug
test -n "${ARG_SOURCE_DIR}" || readonly ARG_SOURCE_DIR="${BASE_DIR}/.."
test -n "${ARG_BUILD_DIR}" || readonly ARG_BUILD_DIR="${ARG_SOURCE_DIR}/build"
test -n "${ARG_INSTALL_DIR}" || readonly ARG_INSTALL_DIR=/usr/local

if [ -n "${ARG_MKLDNN_DIR}" ]; then
    OPT_MKLDNN_INCLUDE_DIR=-DMKLDNN_INCLUDE_DIR=${ARG_MKLDNN_DIR}/include
    OPT_MKLDNN_LIBRARY=-DMKLDNN_LIBRARY=${ARG_MKLDNN_DIR}/lib/libmkldnn.so
fi

test -n "${ARG_LINK_STATIC_LIBGCC}" || readonly ARG_LINK_STATIC_LIBGCC='OFF'
test -n "${ARG_LINK_STATIC_LIBSTDCXX}" || readonly ARG_LINK_STATIC_LIBSTDCXX='OFF'
test -n "${ARG_LINK_STATIC_LIBPROTOBUF}" || readonly ARG_LINK_STATIC_LIBPROTOBUF='OFF'

echo -e "\e[33;1mBuilding Menoh\e[0m"

[ -d "${ARG_BUILD_DIR}" ] || mkdir -p "${ARG_BUILD_DIR}"

cd "${ARG_BUILD_DIR}"
cmake \
    -DCMAKE_BUILD_TYPE=${ARG_BUILD_TYPE} \
    "-DCMAKE_INSTALL_PREFIX=${ARG_INSTALL_DIR}" \
    "${OPT_MKLDNN_INCLUDE_DIR}" \
    "${OPT_MKLDNN_LIBRARY}" \
    -DLINK_STATIC_LIBGCC=${ARG_LINK_STATIC_LIBGCC} \
    -DLINK_STATIC_LIBSTDCXX=${ARG_LINK_STATIC_LIBSTDCXX} \
    -DLINK_STATIC_LIBPROTOBUF=${ARG_LINK_STATIC_LIBPROTOBUF} \
    -DENABLE_TEST=ON \
    "${ARG_SOURCE_DIR}"

make
