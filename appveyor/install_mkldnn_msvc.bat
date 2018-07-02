if exist "c:\mkl-dnn-v0.14-msvc" (
    echo "Using cached directory."
) else (
    git clone https://github.com/intel/mkl-dnn.git
    cd mkl-dnn
    git checkout v0.14
    cd scripts
    call prepare_mkl.bat
    cd ..
    mkdir build
    cd build
    cmake -G "Visual Studio 14 Win64" -DCMAKE_INSTALL_PREFIX=c:\mkl-dnn-v0.14-msvc ..
    cmake --build . --config Release --target install
    cd ..\..
)
