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
test -n "${ARG_VERSION}" || readonly ARG_VERSION=3.6.1

readonly LIBRARY_NAME=protobuf-${ARG_VERSION}
readonly SOURCE_DIR="${ARG_EXTRACT_DIR}/${LIBRARY_NAME}"

test -n "${ARG_BUILD_DIR}" || readonly ARG_BUILD_DIR="${SOURCE_DIR}"
test -n "${ARG_INSTALL_DIR}" || readonly ARG_INSTALL_DIR=/usr/local
test -n "${ARG_PARALLEL}" || readonly ARG_PARALLEL=1

# download (if it isn't cached)
if [ ! -e "${SOURCE_DIR}/LICENSE" ]; then
    echo -e "\e[33;1mDownloading libprotobuf\e[0m"

    [ -d "${ARG_DOWNLOAD_DIR}" ] || mkdir -p "${ARG_DOWNLOAD_DIR}"

    cd "${ARG_DOWNLOAD_DIR}"
    if [ ! -e "protobuf-cpp-${ARG_VERSION}.tar.gz" ]; then
        download_dir="https://github.com/protocolbuffers/protobuf/releases/download/v${ARG_VERSION}/protobuf-cpp-${ARG_VERSION}.tar.gz"
        wget ${download_dir}
    fi
    tar -zxf protobuf-cpp-${ARG_VERSION}.tar.gz -C "${ARG_EXTRACT_DIR}"

    echo -e "\e[32;1mlibprotobuf was successfully downloaded.\e[0m"
else
    echo -e "\e[32;1mlibprotobuf has been downloaded.\e[0m"
fi

# build (if it isn't cached)
if [ ! -e "${ARG_BUILD_DIR}/src/libprotobuf.la" ]; then
    echo -e "\e[33;1mBuilding libprotobuf\e[0m"

    [ -d "${ARG_BUILD_DIR}" ] || mkdir -p "${ARG_BUILD_DIR}"

    cd "${ARG_BUILD_DIR}"
    "${SOURCE_DIR}/configure" --prefix="${ARG_INSTALL_DIR}" CFLAGS="-g -O2 -fPIC" CXXFLAGS="-g -O2 -fPIC"
    make -j${ARG_PARALLEL}

    echo -e "\e[32;1mlibprotobuf was successfully built.\e[0m"
else
    echo -e "\e[32;1mlibprotobuf has been built.\e[0m"
fi
