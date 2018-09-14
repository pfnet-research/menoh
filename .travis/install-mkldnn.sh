#!/bin/bash

# retrieve arguments
while [[ $# != 0 ]]; do
    case $1 in
        --)
            shift
            break
            ;;
        --version)
            ARG_VERSION="$2"
            shift 2
            ;;
        --download-dir)
            ARG_DOWNLOAD_DIR="$2"
            shift 2
            ;;
        --build-dir)
            ARG_BUILD_DIR="$2"
            shift 2
            ;;
        --install-dir)
            ARG_INSTALL_DIR="$2"
            shift 2
            ;;
        --parallel)
            ARG_PARALLEL="$2"
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
test -n "${ARG_VERSION}" || { echo "--version is not specified"; exit 1; }
test -n "${ARG_DOWNLOAD_DIR}" || { echo "--download-dir is not specified"; exit 1; }
test -n "${ARG_BUILD_DIR}" || { echo "--build-dir is not specified"; exit 1; }
test -n "${ARG_INSTALL_DIR}" || { echo "--install-dir is not specified"; exit 1; }
test -n "${ARG_PARALLEL}" || ARG_PARALLEL=1

# download (if it isn't cached)
if [ ! -e "${ARG_BUILD_DIR}/mkl-dnn-${ARG_VERSION}/LICENSE" ]; then
    echo -e "\e[33;1mDownloading libmkldnn\e[0m"

    [ -d "${ARG_DOWNLOAD_DIR}" ] || mkdir -p ${ARG_DOWNLOAD_DIR}

    cd ${ARG_DOWNLOAD_DIR}
    if [ ! -e "mkl-dnn-${ARG_VERSION}.tar.gz" ]; then
        download_url="https://github.com/intel/mkl-dnn/archive/v${ARG_VERSION}.tar.gz"
        wget -O mkl-dnn-${ARG_VERSION}.tar.gz ${download_url}
    fi
    tar -zxf mkl-dnn-${ARG_VERSION}.tar.gz -C ${ARG_BUILD_DIR}

    echo -e "\e[32;1mlibmkldnn was successfully downloaded.\e[0m"
else
    echo -e "\e[32;1mlibmkldnn has been downloaded.\e[0m"
fi

# build (if it isn't cached)
if [ ! -e "${ARG_BUILD_DIR}/mkl-dnn-${ARG_VERSION}/build/src/libmkldnn.so" ]; then
    echo -e "\e[33;1mBuilding libmkldnn\e[0m"

    cd ${ARG_BUILD_DIR}/mkl-dnn-${ARG_VERSION}/scripts
    ./prepare_mkl.sh

    cd ${ARG_BUILD_DIR}/mkl-dnn-${ARG_VERSION}
    [ -d "build" ] || mkdir -p build

    cd build
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=${ARG_INSTALL_DIR} \
        -DWITH_TEST=OFF \
        -DWITH_EXAMPLE=OFF \
        -DARCH_OPT_FLAGS='' \
        -Wno-error=unused-result \
        ..
    make -j${ARG_PARALLEL}

    echo -e "\e[32;1mlibmkldnn was successfully built.\e[0m"
else
    echo -e "\e[32;1mlibmkldnn has been built.\e[0m"
fi

# install (always)
echo -e "\e[33;1mInstalling libmkldnn\e[0m"

cd ${ARG_BUILD_DIR}/mkl-dnn-${ARG_VERSION}/build
make install/strip
