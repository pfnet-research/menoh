# Building Menoh
You need to install [prerequisites](#prerequisites) for your platform before [building](#building) Menoh.

## Prerequisites
To build Menoh, you require the following toolchains:

Unix:
- CMake 3.1 or later
- GCC 4.9 or later

macOS (OSX):
- XCode
- [Homebrew](https://brew.sh/)

Windows:
- Visual Studio 2015

Windows (MINGW):
- [MSYS2](http://www.msys2.org/)

You also need to install the dependent libraries on your system:

- [Protocol Buffers](https://developers.google.com/protocol-buffers/) 2.6.1 or later (building instructions are [here](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md))
- [MKL-DNN](https://github.com/intel/mkl-dnn) 0.14 or later (for `mkldnn` backend) (building instructions are [here](https://github.com/intel/mkl-dnn/blob/master/README.md#installation))

`protobuf` can be installed through most package managers instead of building it yourself. `mkl-dnn` package, unfortunatelly, is not available in many environments at the moment (except for `brew` in macOS).

Note that you can use ProtoBuf either version 2 or 3, but, for example, if you build Menoh with `protoc` ver 3 you should use the binary with runtime ver 3.

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
Please replace `(CMake_Install_Dir)` in the following with your working directory.

#### ProtoBuf
Download and unzip https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protobuf-cpp-3.6.1.zip

```
cd protobuf-3.6.1/cmake
mkdir build
cd build
cmake .. -G "Visual Studio 14" -A x64 -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=(CMake_Install_Dir)
cmake --build . --config Release --target install
cd ../../..
```

#### MKL-DNN
```
git clone https://github.com/intel/mkl-dnn.git
cd mkl-dnn/scripts
.\prepare_mkl.bat
cd ..
mkdir build
cd build
cmake .. -G "Visual Studio 14 Win64"  -DCMAKE_INSTALL_PREFIX=(CMake_Install_Dir)
cmake --build . --config Release --target install
cd ../..
```

### Windows (MINGW)
```
pacman -S mingw-w64-x86_64-toolchain
pacman -S git
pacman -S mingw-w64-x86_64-cmake
pacman -S mingw-w64-x86_64-protobuf mingw-w64-x86_64-protobuf-c
```

#### Installing MKL-DNN from binary package
```
curl -omingw-w64-x86_64-mkl-dnn-0.15-1-x86_64.pkg.tar.xz -L https://github.com/pfnet-research/menoh/releases/download/v1.0.3/mingw-w64-x86_64-mkl-dnn-0.15-1-x86_64.pkg.tar.xz
pacman -S --noconfirm mingw-w64-x86_64-mkl-dnn-0.15-1-x86_64.pkg.tar.xz
```

#### Installing MKL-DNN from source
```
git clone https://github.com/intel/mkl-dnn.git
cd mkl-dnn
cd scripts && ./prepare_mkl.sh && cd ..
mkdir -p build
cd build
MSYS2_ARG_CONV_EXCL="-DCMAKE_INSTALL_PREFIX=" \
  cmake -G "MSYS Makefiles" -DCMAKE_INSTALL_PREFIX=/mingw64
make
make install
```

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
python scripts/retrieve_data.py
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

```
git clone https://github.com/pfnet-research/menoh.git
cd menoh
mkdir -p build && cd build
cmake ..
make
make install
```

### Windows
Please replace `(CMake_Install_Dir)` in the following with your working directory.

```
git clone https://github.com/pfnet-research/menoh.git
cd menoh
mkdir build
cd build
cmake .. -G "Visual Studio 14 Win64" -DCMAKE_PREFIX_PATH=(CMake_Install_Dir) -DCMAKE_INSTALL_PREFIX=(CMake_Install_Dir) -DENABLE_TEST=OFF -DENABLE_BENCHMARK=OFF -DENABLE_EXAMPLE=OFF
cmake --build . --config Release --target install
```

### Windows (MINGW)

```
git clone https://github.com/pfnet-research/menoh.git
cd menoh
mkdir -p build && cd build
MSYS2_ARG_CONV_EXCL="-DCMAKE_INSTALL_PREFIX=" \
  cmake -G "MSYS Makefiles" -DCMAKE_INSTALL_PREFIX=/mingw64
make
```

### Note

#### Python command name
Menoh requires `python` command to generate source codes at build time. Add `PYTHON_EXECUTABLE` option to `cmake` if you want to use `python` command with non-standard name (e.g. `python3`).

```bash
cmake -DPYTHON_EXECUTABLE=python3 ..
```
