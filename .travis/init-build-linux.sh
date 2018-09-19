# check if variables are set
test -n "${PROTOBUF_VERSION}" || { echo "PROTOBUF_VERSION can't be empty" 1>&2; exit 1; }
test -n "${MKLDNN_VERSION}" || { echo "MKLDNN_VERSION can't be empty" 1>&2; exit 1; }
test -n "${MAKE_JOBS}" || { echo "MAKE_JOBS can't be empty" 1>&2; exit 1; }

test -n "${LINK_STATIC}" || LINK_STATIC=false

# TODO: make them configurable for outside Travis
export WORK_DIR=${HOME}
export PROJ_DIR=${TRAVIS_BUILD_DIR} # = ${HOME}/build/${TRAVIS_REPO_SLUG}

export PROTOBUF_INSTALL_DIR=/usr/local
export MKLDNN_INSTALL_DIR=/usr/local

# Run a docker container and map Travis's $HOME to the container's $HOME
# $HOME:$HOME = /home/travis                     : /home/travis
#               /home/travis/build               : /home/travis/build
#               /home/travis/build/<user>/<repo> : /home/travis/build/<user>/<repo> (= ${TRAVIS_BUILD_DIR})
SHARED_SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
source ${SHARED_SCRIPT_DIR}/run-container.sh --image ${BUILDENV_IMAGE} --work-dir ${WORK_DIR}
test -n "${BUILDENV_CONTAINER_ID}" || { echo "BUILDENV_CONTAINER_ID can't be empty" 1>&2; exit 1; }

## define shared functions for Linux-based platforms

# Run the specified string as command in the container
function docker_exec() {
    docker exec -it ${BUILDENV_CONTAINER_ID} /bin/bash -xec "$1"
    return $?
}

# Run the specified shell script in the container
function docker_exec_script() {
    docker exec -it ${BUILDENV_CONTAINER_ID} /bin/bash -xe $@
    return $?
}

function install_protobuf() {
    docker_exec_script \
        ${PROJ_DIR}/.travis/install-protobuf.sh \
            --version ${PROTOBUF_VERSION} \
            --download-dir ${WORK_DIR}/downloads \
            --build-dir ${WORK_DIR}/build \
            --install-dir ${PROTOBUF_INSTALL_DIR} \
            --parallel ${MAKE_JOBS}
}

function install_mkldnn() {
    docker_exec_script \
        ${PROJ_DIR}/.travis/install-mkldnn.sh \
            --version ${MKLDNN_VERSION} \
            --download-dir ${WORK_DIR}/downloads \
            --build-dir ${WORK_DIR}/build \
            --install-dir ${MKLDNN_INSTALL_DIR} \
            --parallel ${MAKE_JOBS}
}

function prepare_menoh_data() {
    docker_exec_script \
        ${PROJ_DIR}/.travis/prepare-menoh-data.sh \
            --source-dir ${PROJ_DIR} \
            --python-executable python3
}

function build_menoh() {
    if [ "${LINK_STATIC}" != "true" ]; then
        docker_exec_script \
            ${PROJ_DIR}/.travis/build-menoh.sh \
                --source-dir ${PROJ_DIR}
    else
        docker_exec_script \
            ${PROJ_DIR}/.travis/build-menoh.sh \
                --source-dir ${PROJ_DIR} \
                --link-static-libgcc ON \
                --link-static-libstdcxx ON \
                --link-static-libprotobuf ON
    fi
}

function test_menoh() {
    docker_exec "cd ${PROJ_DIR}/build && ./test/menoh_test"
}

function check_menoh_artifact() {
    docker_exec "ldd ${PROJ_DIR}/build/menoh/libmenoh.so"
}
