# check if variables are set
test -n "${MAKE_JOBS}" || { echo "MAKE_JOBS does not exist"; exit 1; }

# TODO: make them configurable for outside Travis
export WORK_DIR=${HOME}
export PROJ_DIR=${TRAVIS_BUILD_DIR} # = ${HOME}/build/${TRAVIS_REPO_SLUG}

function prepare_menoh_data() {
    echo -e "\e[33;1mPreparing data/ for Menoh\e[0m"

    cd ${PROJ_DIR}
    [ -d "data" ] || mkdir -p data

    python retrieve_data.py
    python gen_test_data.py
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
    cd ${PROJ_DIR}/build/menoh && ./test/menoh_test
}

function check_menoh_artifact() {
    otool -L ${PROJ_DIR}/build/menoh/libmenoh.dylib
}
