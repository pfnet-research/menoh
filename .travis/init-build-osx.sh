# check if variables are set
test -n "${MAKE_JOBS}" || { echo "MAKE_JOBS can't be empty" 1>&2; exit 1; }

# TODO: make them configurable for outside Travis
export WORK_DIR=${HOME}
export PROJ_DIR=${TRAVIS_BUILD_DIR} # = ${HOME}/build/${TRAVIS_REPO_SLUG}

## define shared functions for macOS (OSX) platforms

function prepare_menoh_data() {
    bash -ex ${PROJ_DIR}/.travis/prepare-menoh-data.sh \
        --source-dir ${PROJ_DIR} \
        --python-executable python
}

function build_menoh() {
    if [ "${LINK_STATIC}" != "true" ]; then
        bash -ex ${PROJ_DIR}/.travis/build-menoh.sh \
            --source-dir ${PROJ_DIR}
    else
        # Does not set --link-static-libgcc and --link-static-libstdcxx in macOS
        bash -ex ${PROJ_DIR}/.travis/build-menoh.sh \
            --source-dir ${PROJ_DIR} \
            --link-static-libprotobuf ON
    fi
}

function test_menoh() {
    cd ${PROJ_DIR}/build
    ./test/menoh_test
}

function check_menoh_artifact() {
    otool -L ${PROJ_DIR}/build/menoh/libmenoh.dylib
}
