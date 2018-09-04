#!/bin/sh

CXX=aarch64-linux-android-clang++ \
 CC=aarch64-linux-android-clang \
 LD=aarch64-linux-android-ld \
cmake -DENABLE_ANDROID_AARCH64=ON -DENABLE_ARMNN_X86_64=OFF ..

make -j8


