#!/bin/bash -ex

BASE_DIR=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)

# initialize this script
source ${BASE_DIR}/../init-build-linux.sh

# check the environment
docker_exec "ls -l ${WORK_DIR}"
docker_exec "ls -l ${WORK_DIR}/build"
docker_exec "ls -l ${WORK_DIR}/build/${TRAVIS_REPO_SLUG}"

docker_exec "(printenv | grep PATH) && make --version && cmake --version && g++ --version && ldd --version"

# build and install prerequisites
install_protobuf
install_mkldnn

docker_exec "pip3 install --user chainer" # for generating test data

docker_exec "rpm -qa"
docker_exec "pip3 list"

# build and test menoh
build_menoh
prepare_menoh_data
test_menoh

# check the artifact and release
check_menoh_artifact

# release the artifact
# TODO
