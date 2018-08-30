# Building Menoh

## Prerequisites
To build Menoh, you require the following toolchains:

Unix:
- CMake 3.1 or later
- GCC 4.9 or later

macOS (OSX):
- TODO

Windows:
- git
- Visual Studio 2015

You also need to install the dependent libraries on your system:

- [Protocol Buffers](https://developers.google.com/protocol-buffers/) 2.6.1 or later
    - Building instructions are [here](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md)
- [MKL-DNN](https://github.com/intel/mkl-dnn) 0.14 or later (for `mkldnn` backend)
    - Building instructions are [here](https://github.com/intel/mkl-dnn/blob/master/README.md#installation)

You can install `protobuf` through the package manager instead of building it yourself. `mkl-dnn` package, unfortunatelly, is not available in many environments at the moment (except for `brew` in macOS).

### Debian/Ubuntu
```
apt-get install gcc g++ cmake-data cmake libopencv-dev libprotobuf-dev protobuf-compiler

# See the MKL-DNN's instructions for details
git clone https://github.com/intel/mkl-dnn.git
cd mkl-dnn
cd scripts && ./prepare_mkl.sh && cd ..
mkdir -p build && cd build && cmake .. && make
make install # as root
```

### macOS
```
brew update
brew install protobuf mkl-dnn
```

### Windows

None

## Building

### Unix
Run the following command to build the source code:

```
git clone https://github.com/pfnet-research/menoh.git
cd menoh
mkdir -p build && cd build
cmake ..
make
```

To install Menoh into your system:

```
make install # as root
```

To run the example, you also need to download model data:

```
python retrieve_data.py
```

#### Static linking
Menoh depends on several other libraries like `protobuf`. In Unix like systems, there is a chance to fail if you take the binary you built to other system because sometimes the dependent libraries installed on the system are not compatible with it.

To improve the portability, you can statically link Menoh with its dependencies. There is the following options for `cmake` command:

- `LINK_STATIC_LIBGCC` for `libgcc`
- `LINK_STATIC_LIBSTDCXX` for `libstdc++`
- `LINK_STATIC_LIBPROTOBUF` for `libprotobuf`
- `LINK_STATIC_MKLDNN` for `libmkldnn` (NOT supported in this version)

If you use `LINK_STATIC_LIBPROTOBUF` and `LINK_STATIC_MKLDNN`, you don't need to install the libraries on your system because they build static library from the source.

All options are disabled by default, and you can turn them on as below:

```
cmake \
  -DLINK_STATIC_LIBGCC=OFF \
  -DLINK_STATIC_LIBSTDCXX=OFF \
  -DLINK_STATIC_LIBPROTOBUF=ON \
  ..
```

Note that static linking is great for binary portability, but it increases the binary size and potentially introduces further weird problems depending on the combination. We strongly recommend to avoid using `LINK_STATIC_*` options and consider building the binary against the specific system where you run your Menoh application.

#### Old `libstdc++` ABI support
Menoh has `USE_OLD_GLIBCXX_ABI` option to build its C++ source codes against the old `libstdc++` ABI to improve [backward compatibility](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html) for the systems that depend on older GCC (`< 5.2`).

Note that enabling this option may cause a problem if `protobuf` in your environment is compiled against the new ABI because its API includes `std::` data types. We recommend to use it along with `-DLINK_STATIC_LIBPROTOBUF=ON`:

```
cmake -DUSE_OLD_GLIBCXX_ABI=ON -DLINK_STATIC_LIBPROTOBUF=ON ..
```

### macOS (OS X)
TODO

### Windows

Please specify your directory into (CMake_Install_Dir).

#### mkl-dnn

```
git clone https://github.com/intel/mkl-dnn.git
cd mkl-dnn/scripts
.\prepare_mkl.bat
cd ..
mdir build
cd build
cmake .. -G "Visual Studio 14 Win64"  -DCMAKE_INSTALL_PREFIX=(CMake_Install_Dir)
cmake --build . --config Debug --target install
cmake --build . --config Release --target install
cd ../..
```

#### protobuf

Download and unzip https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protobuf-cpp-3.6.1.zip

```
cd protobuf-3.6.1/cmake
mdir build
cd build
cmake .. -G "Visual Studio 14" -A x64 -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=(CMake_Install_Dir)
cmake --build . --config Debug --target install
cmake --build . --config Release --target install
cd ../../..
```

#### menoh

```
git clone https://github.com/pfnet-research/menoh.git
cd menoh
mdir build
cd build
cmake .. -G "Visual Studio 14 Win64" -DCMAKE_PREFIX_PATH=CMake_Install_Dir) -DCMAKE_INSTALL_PREFIX=CMake_Install_Dir) -DENABLE_TEST=OFF -DENABLE_BENCHMARK=OFF -DENABLE_EXAMPLE=OFF -DENABLE_TOOL=OFF
cmake --build . --config Debug --target install
cmake --build . --config Release --target install
```
