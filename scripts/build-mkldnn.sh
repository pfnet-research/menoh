#!/bin/bash -e

# retrieve arguments
while [[ $# != 0 ]]; do
    case $1 in
        --)
            shift
            break
            ;;
        --version)
            readonly ARG_VERSION="$2"
            shift 2
            ;;
        --download-dir)
            readonly ARG_DOWNLOAD_DIR="$2"
            shift 2
            ;;
        --extract-dir)
            readonly ARG_EXTRACT_DIR="$2"
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
        --parallel)
            readonly ARG_PARALLEL="$2"
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
test -n "${ARG_DOWNLOAD_DIR}" || { echo "--download-dir is not specified" 1>&2; exit 1; }
test -n "${ARG_EXTRACT_DIR}" || { echo "--extract-dir is not specified" 1>&2; exit 1; }

# options that have default value
test -n "${ARG_VERSION}" || readonly ARG_VERSION=0.16

readonly LIBRARY_NAME=mkl-dnn-${ARG_VERSION}
readonly SOURCE_DIR="${ARG_EXTRACT_DIR}/${LIBRARY_NAME}"

test -n "${ARG_BUILD_DIR}" || readonly ARG_BUILD_DIR="${SOURCE_DIR}/build"
test -n "${ARG_INSTALL_DIR}" || readonly ARG_INSTALL_DIR=/usr/local
test -n "${ARG_PARALLEL}" || readonly ARG_PARALLEL=1

# download (if it isn't cached)
if [ ! -e "${SOURCE_DIR}/LICENSE" ]; then
    echo -e "\e[33;1mDownloading libmkldnn\e[0m"

    [ -d "${ARG_DOWNLOAD_DIR}" ] || mkdir -p "${ARG_DOWNLOAD_DIR}"

    cd "${ARG_DOWNLOAD_DIR}"
    if [ ! -e "${LIBRARY_NAME}.tar.gz" ]; then
        download_url="https://github.com/intel/mkl-dnn/archive/v${ARG_VERSION}.tar.gz"
        wget -O ${LIBRARY_NAME}.tar.gz ${download_url}
    fi
    tar -zxf ${LIBRARY_NAME}.tar.gz -C "${ARG_EXTRACT_DIR}"

    echo -e "\e[32;1mlibmkldnn was successfully downloaded.\e[0m"
else
    echo -e "\e[32;1mlibmkldnn has been downloaded.\e[0m"
fi

# build (if it isn't cached)
if [ ! -e "${ARG_BUILD_DIR}/src/libmkldnn.so" ]; then
    echo -e "\e[33;1mBuilding libmkldnn\e[0m"

    cd "${SOURCE_DIR}/scripts"
    ./prepare_mkl.sh

    [ -d "${ARG_BUILD_DIR}" ] || mkdir -p "${ARG_BUILD_DIR}"

    cd "${ARG_BUILD_DIR}"
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        "-DCMAKE_INSTALL_PREFIX=${ARG_INSTALL_DIR}" \
        -DWITH_TEST=OFF \
        -DWITH_EXAMPLE=OFF \
        -DARCH_OPT_FLAGS='' \
        -Wno-error=unused-result \
        "${SOURCE_DIR}"
    make -j${ARG_PARALLEL}

    echo -e "\e[32;1mlibmkldnn was successfully built.\e[0m"
else
    echo -e "\e[32;1mlibmkldnn has been built.\e[0m"
fi
