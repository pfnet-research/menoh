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
if [ ! -e "${ARG_BUILD_DIR}/protobuf-${ARG_VERSION}/LICENSE" ]; then
    echo -e "\e[33;1mDownloading libprotobuf\e[0m"

    [ -d "${ARG_DOWNLOAD_DIR}" ] || mkdir -p ${ARG_DOWNLOAD_DIR}

    cd ${ARG_DOWNLOAD_DIR}
    if [ ! -e "protobuf-cpp-${ARG_VERSION}.tar.gz" ]; then
        download_dir="https://github.com/protocolbuffers/protobuf/releases/download/v${ARG_VERSION}/protobuf-cpp-${ARG_VERSION}.tar.gz"
        wget ${download_dir}
    fi
    tar -zxf protobuf-cpp-${ARG_VERSION}.tar.gz -C ${ARG_BUILD_DIR}

    echo -e "\e[32;1mlibprotobuf was successfully downloaded.\e[0m"
else
    echo -e "\e[32;1mlibprotobuf has been downloaded.\e[0m"
fi

# build (if it isn't cached)
if [ ! -e "${ARG_BUILD_DIR}/protobuf-${ARG_VERSION}/src/libprotobuf.la" ]; then
    echo -e "\e[33;1mBuilding libprotobuf\e[0m"

    cd ${ARG_BUILD_DIR}/protobuf-${ARG_VERSION}
    ./configure --prefix=${ARG_INSTALL_DIR} CFLAGS=-fPIC CXXFLAGS=-fPIC
    make -j${ARG_PARALLEL}

    echo -e "\e[32;1mlibprotobuf was successfully built.\e[0m"
else
    echo -e "\e[32;1mlibprotobuf has been built.\e[0m"
fi

# install (always)
echo -e "\e[33;1mInstalling libprotobuf\e[0m"

cd ${ARG_BUILD_DIR}/protobuf-${ARG_VERSION}
make install
