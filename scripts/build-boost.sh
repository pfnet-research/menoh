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
        --stage-dir)
            readonly ARG_STAGE_DIR="$2"
            shift 2
            ;;
        --install-dir)
            readonly ARG_INSTALL_DIR="$2"
            shift 2
            ;;
        --toolset)
            readonly ARG_TOOLSET="$2"
            shift 2
            ;;
        --cxx)
            readonly ARG_CXX_CMD="$2"
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
test -n "${ARG_VERSION}" || readonly ARG_VERSION=1.64.0

readonly LIBRARY_NAME=boost_$(echo ${ARG_VERSION} | sed -e 's/\./_/g') # e.g. boost_1_64_0
readonly SOURCE_DIR="${ARG_EXTRACT_DIR}/${LIBRARY_NAME}"

test -n "${ARG_BUILD_DIR}" || readonly ARG_BUILD_DIR="${SOURCE_DIR}/build"
test -n "${ARG_STAGE_DIR}" || readonly ARG_STAGE_DIR="${SOURCE_DIR}/stage"
test -n "${ARG_INSTALL_DIR}" || readonly ARG_INSTALL_DIR=/usr/local
test -n "${ARG_PARALLEL}" || readonly ARG_PARALLEL=1

# options for cross compiling
if [ -n "${ARG_TOOLSET}" ]; then # e.g. gcc-arm
    # requires a location of g++ command
    test -n "${ARG_CXX_CMD}" || { echo "--cxx is not specified" 1>&2; exit 1; }

    readonly USER_CONFIG_JAM="using $(echo ${ARG_TOOLSET} | sed -e 's/-/ : /') : ${ARG_CXX_CMD} ;"
    readonly OPT_TOOLSET=toolset=${ARG_TOOLSET}
fi

# download (if it isn't cached)
if [ ! -e "${SOURCE_DIR}/INSTALL" ]; then
    echo -e "\e[33;1mDownloading libboost\e[0m"

    [ -d "${ARG_DOWNLOAD_DIR}" ] || mkdir -p "${ARG_DOWNLOAD_DIR}"

    cd "${ARG_DOWNLOAD_DIR}"
    if [ ! -e "${LIBRARY_NAME}.tar.gz" ]; then
        download_dir="https://dl.bintray.com/boostorg/release/${ARG_VERSION}/source/${LIBRARY_NAME}.tar.gz"
        curl -LO ${download_dir} # wget doesn't work for bintray with 403 errpr
    fi
    tar -zxf ${LIBRARY_NAME}.tar.gz -C "${ARG_EXTRACT_DIR}"

    echo -e "\e[32;1mlibboost was successfully downloaded.\e[0m"
else
    echo -e "\e[32;1mlibboost has been downloaded.\e[0m"
fi

# build (if it isn't cached)
if [ ! -e "${ARG_STAGE_DIR}/lib/libboost_system.a" ]; then
    echo -e "\e[33;1mBuilding libprotobuf\e[0m"

    cd "${SOURCE_DIR}"

    # bootstrap
    ./bootstrap.sh --prefix="${ARG_INSTALL_DIR}"

    # configure options
    if [ -n "${USER_CONFIG_JAM}" ]; then
        echo "${USER_CONFIG_JAM}" > ${SOURCE_DIR}/user_config.jam
        readonly OPT_USER_CONFIG_JAM=--user-config=${SOURCE_DIR}/user_config.jam
    fi
    echo "-j${ARG_PARALLEL} ${OPT_TOOLSET} ${OPT_USER_CONFIG_JAM} link=static cflags=-fPIC cxxflags=-fPIC --with-filesystem --with-test --with-log --with-program_options --build-dir=${ARG_BUILD_DIR} --stage-dir=${ARG_STAGE_DIR}" > b2_opts.txt

    # run a build
    ./b2 stage $(cat b2_opts.txt)

    echo -e "\e[32;1mlibboost was successfully built.\e[0m"
else
    echo -e "\e[32;1mlibboost has been built.\e[0m"
fi
