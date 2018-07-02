if [ ! -d "/mkl-dnn-86b7129-mingw" ]; then
    git clone https://github.com/intel/mkl-dnn.git
    cd mkl-dnn
    git checkout 86b712989c82dffdd8742aa49ee1c7d883fc838b
    cd scripts && ./prepare_mkl.bat && cd ..
    sed -i 's/add_subdirectory(examples)//g' CMakeLists.txt
    sed -i 's/add_subdirectory(tests)//g' CMakeLists.txt
    mkdir -p build && cd build
    export MSYS2_ARG_CONV_EXCL="-DCMAKE_INSTALL_PREFIX="
    cmake -G "MSYS Makefiles" -DCMAKE_INSTALL_PREFIX=/mkl-dnn-86b7129-mingw ..
    make
    make DESTDIR=/ install
    cd ../..
else
    echo "Using cached directory."
fi
