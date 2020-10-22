# check if variables are set
test -n "${MAKE_JOBS}" || { echo "MAKE_JOBS can't be empty" 1>&2; exit 1; }

test -n "${BUILD_STATIC_LIBS}" || BUILD_STATIC_LIBS=false

# TODO: make them configurable for outside Travis
export WORK_DIR=${HOME}
export PROJ_DIR=${TRAVIS_BUILD_DIR} # = ${HOME}/build/${TRAVIS_REPO_SLUG}

export MKLDNN_INSTALL_DIR=/usr/local

## define shared functions for macOS (OSX) platforms

function build_mkldnn() {
    bash -ex "${PROJ_DIR}/scripts/build-mkldnn.sh" \
        --version ${MKLDNN_VERSION} \
        --download-dir "${WORK_DIR}/downloads" \
        --extract-dir "${WORK_DIR}/build" \
        --install-dir "${MKLDNN_INSTALL_DIR}" \
        --parallel ${MAKE_JOBS}
}

function install_mkldnn() {
    bash -ex "${PROJ_DIR}/scripts/install-mkldnn.sh" \
        --build-dir "${WORK_DIR}/build/oneDNN-${MKLDNN_VERSION}/build"
}

function prepare_menoh_data() {
    bash -ex "${PROJ_DIR}/scripts/prepare-menoh-data.sh" \
        --source-dir "${PROJ_DIR}" \
        --python-executable python
}

function build_menoh() {
    if [ "${BUILD_STATIC_LIBS}" = "true" ]; then
        bash -ex "${PROJ_DIR}/scripts/build-menoh.sh" \
            --build-type Release \
            --source-dir "${PROJ_DIR}" \
            --python-executable python \
            --build-shared-libs OFF
    elif [ "${LINK_STATIC}" != "true" ]; then
        bash -ex "${PROJ_DIR}/scripts/build-menoh.sh" \
            --build-type Release \
            --source-dir "${PROJ_DIR}" \
            --python-executable python
    else
        # Does not set --link-static-libgcc and --link-static-libstdcxx in macOS
        bash -ex "${PROJ_DIR}/scripts/build-menoh.sh" \
            --build-type Release \
            --source-dir "${PROJ_DIR}" \
            --python-executable python \
            --link-static-libprotobuf ON
    fi
}

function test_menoh() {
    cd "${PROJ_DIR}/build"
    ./test/menoh_test
}

function check_menoh_artifact() {
    if [ "${BUILD_STATIC_LIBS}" != "true" ]; then
      otool -L "${PROJ_DIR}/build/menoh/libmenoh.dylib"
    fi
}
