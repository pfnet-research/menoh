#!/bin/bash

BASE_DIR=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)

# initialize this script
source ${BASE_DIR}/../init-build-osx.sh

# check the environment
ls -l ${WORK_DIR}
ls -l ${WORK_DIR}/build
ls -l ${WORK_DIR}/build/${TRAVIS_REPO_SLUG}

printenv | grep PATH
make --version
cmake --version
g++ --version

# install prerequisites
brew update
brew upgrade python

export PATH=/usr/local/opt/python/libexec/bin:$PATH

brew install numpy || true
brew install opencv mkl-dnn

if [ "$LINK_STATIC" != "true" ]; then brew install protobuf; fi

pip install --user chainer # for generating test data

brew list --versions
pip list

# build and test menoh
build_menoh
prepare_menoh_data
test_menoh

# check the artifact and release
check_menoh_artifact
