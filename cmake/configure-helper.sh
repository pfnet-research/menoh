#!/bin/bash -ex

c_compiler=$1
cxx_compiler=$2
prefix=$3
cflags=$4
cxxflags=$5

export CC=${c_compiler}
export CPP="${c_compiler} -E"
export CXX=${cxx_compiler}
export CXXCPP="${cxx_compiler} -E"

./configure --prefix=${prefix} CFLAGS=${cflags} CXXFLAGS=${cxxflags}
