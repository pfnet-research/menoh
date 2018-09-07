if [ ! -d "$HOME/mkl-dnn${MKL_DNN_INSTALL_SUFFIX}/lib" ]; then
    git clone https://github.com/intel/mkl-dnn.git
    cd mkl-dnn
    git checkout $MKL_DNN_REV
    cd scripts && bash ./prepare_mkl.sh && cd ..
    sed -i 's/add_subdirectory(examples)//g' CMakeLists.txt
    sed -i 's/add_subdirectory(tests)//g' CMakeLists.txt
    mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX=$HOME/mkl-dnn${MKL_DNN_INSTALL_SUFFIX} .. && make
    make install
    cd ..
else
    echo "Using cached directory."
fi
