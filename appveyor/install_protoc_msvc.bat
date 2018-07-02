if exist "c:\protobuf-3.6.0-msvc" (
    echo "Using cached directory."
) else (
    curl -oprotobuf-3.6.0.zip -L --insecure https://github.com/google/protobuf/archive/v3.6.0.zip
    7z x protobuf-3.6.0.zip
    cd protobuf-3.6.0
    cd cmake
    mkdir build
    cd build
    cmake .. -G "Visual Studio 14 Win64" -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=c:\protobuf-3.6.0-msvc
    cmake --build . --config Release --target install
    cd ..\..\..
)
