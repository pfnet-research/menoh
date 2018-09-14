#!/bin/bash

# retrieve arguments
while [[ $# != 0 ]]; do
    case $1 in
        --)
            shift
            break
            ;;
        --source-dir)
            ARG_SOURCE_DIR="$2"
            shift 2
            ;;
        --install-dir)
            ARG_INSTALL_DIR="$2"
            shift 2
            ;;
        --link-static-libgcc)
            ARG_LINK_STATIC_LIBGCC="$2"
            shift 2
            ;;
        --link-static-libstdcxx)
            ARG_LINK_STATIC_LIBSTDCXX="$2"
            shift 2
            ;;
        --link-static-libprotobuf)
            ARG_LINK_STATIC_LIBPROTOBUF="$2"
            shift 2
            ;;
        -*)
            err Unknown option \"$1\"
            exit
            ;;
        *)
            break
            ;;

    esac
done

# validate the arguments
test -n "${ARG_SOURCE_DIR}" || { echo "--source-dir is not specified" 1>&2; exit 1; }

test -n "${ARG_LINK_STATIC_LIBGCC}" || ARG_LINK_STATIC_LIBGCC='OFF'
test -n "${ARG_LINK_STATIC_LIBSTDCXX}" || ARG_LINK_STATIC_LIBSTDCXX='OFF'
test -n "${ARG_LINK_STATIC_LIBPROTOBUF}" || ARG_LINK_STATIC_LIBPROTOBUF='OFF'

echo -e "\e[33;1mBuilding Menoh\e[0m"

cd ${ARG_SOURCE_DIR}
[ -d "build" ] || mkdir -p build

cd build
if [ -n "${ARG_INSTALL_DIR}" ]; then
    CMAKE_INSTALL_PREFIX="-DCMAKE_INSTALL_PREFIX=${ARG_INSTALL_DIR}"
fi
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    ${CMAKE_INSTALL_PREFIX} \
    -DLINK_STATIC_LIBGCC=${ARG_LINK_STATIC_LIBGCC} \
    -DLINK_STATIC_LIBSTDCXX=${ARG_LINK_STATIC_LIBSTDCXX} \
    -DLINK_STATIC_LIBPROTOBUF=${ARG_LINK_STATIC_LIBPROTOBUF} \
    -DENABLE_TEST=ON \
    ..

make
