# check if variables are set
test -n "${DOCKER_CONTAINER_ID}" || { echo "DOCKER_CONTAINER_ID does not exist"; exit 1; }
test -n "${PROTOBUF_VERSION}" || { echo "PROTOBUF_VERSION does not exist"; exit 1; }
test -n "${MKLDNN_VERSION}" || { echo "MKLDNN_VERSION does not exist"; exit 1; }
test -n "${MAKE_JOBS}" || { echo "MAKE_JOBS does not exist"; exit 1; }

test -n "${LINK_STATIC}" || LINK_STATIC=false

# TODO: make them configurable for outside Travis
export WORK_DIR=${HOME}
export PROJ_DIR=${TRAVIS_BUILD_DIR} # = ${HOME}/build/${TRAVIS_REPO_SLUG}

export PROTOBUF_INSTALL_DIR=/usr/local
export MKLDNN_INSTALL_DIR=/usr/local

# define shared functions for Linux-based platforms
function docker_exec() {
    docker exec -it ${DOCKER_CONTAINER_ID} /bin/bash -xec "$1"
}

function docker_exec_cmd() {
    docker exec -it ${DOCKER_CONTAINER_ID} /bin/bash -xe $@
}

function install_protobuf() {
    docker_exec_cmd \
        ${PROJ_DIR}/.travis/install-protobuf.sh \
            --version ${PROTOBUF_VERSION} \
            --download-dir ${WORK_DIR}/downloads \
            --build-dir ${WORK_DIR}/build \
            --install-dir ${PROTOBUF_INSTALL_DIR} \
            --parallel ${MAKE_JOBS}
}

function install_mkldnn() {
    docker_exec_cmd \
        ${PROJ_DIR}/.travis/install-mkldnn.sh \
            --version ${MKLDNN_VERSION} \
            --download-dir ${WORK_DIR}/downloads \
            --build-dir ${WORK_DIR}/build \
            --install-dir ${MKLDNN_INSTALL_DIR} \
            --parallel ${MAKE_JOBS}
}

function prepare_menoh_data() {
    echo -e "\e[33;1mPreparing data/ for Menoh\e[0m"
    docker_exec "$(cat << EOS
        cd ${PROJ_DIR}/menoh && \
        ([ -d "data" ] || mkdir -p data) && \
        python3 retrieve_data.py && \
        python3 gen_test_data.py
EOS
)"
}

function build_menoh() {
    # CMakeCache.txt generated for coverity_scan build hinders out-of-source build
    docker_exec "if [ -f \"CMakeCache.txt\" ]; then cd ${PROJ_DIR} && rm CMakeCache.txt; fi"
    if [ "${LINK_STATIC}" != "true" ]; then
        docker_exec_cmd \
            ${PROJ_DIR}/.travis/build-menoh.sh \
                --source-dir ${PROJ_DIR}
    else
        docker_exec_cmd \
            ${PROJ_DIR}/.travis/build-menoh.sh \
                --source-dir ${PROJ_DIR} \
                --link-static-libgcc ON \
                --link-static-libstdcxx ON \
                --link-static-libprotobuf ON
    fi
}

function test_menoh() {
    docker_exec "cd ${PROJ_DIR}/menoh/build && ./test/menoh_test"
}

function check_menoh_artifact() {
    ldd ${PROJ_DIR}/menoh/build/menoh/libmenoh.so
}
