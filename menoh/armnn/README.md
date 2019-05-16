
# Menoh on ARM

We checked the operation with the following software.

## Arm Compute Library : v19.02
  - https://github.com/ARM-software/ComputeLibrary/tree/v19.02

  % git clone -b v19.02 https://github.com/ARM-software/ComputeLibrary.git v19.02
  % cd v19.02

  # native compile 
  # for ARM64 CPU(NEON) only
  % scons Werror=1 -j4 debug=0 neon=1 opencl=0 os=linux arch=arm64-v8a build=native openmp=yes 

  # for ARM64 CPU(NEON) and GPU(OpenCL)
  % scons Werror=1 -j4 debug=0 neon=1 opencl=1 os=linux arch=arm64-v8a build=native openmp=yes 

  # for Raspberry Pi 2/3 (ARM32 CPU:NEON)
  $ sudo apt install scons
  
  $ scons Werror=1 -j4 debug=0 neon=1 opencl=0 os=linux arch=armv7abuild=native openmp=yes

  # for X86_64 (Cpu Referece)
  $ scons Werror=1 -j8 debug=0 neon=0 opencl=0 os=linux arch=x86_64 openmp=yes

  # cross compile 
  # for (ARM64 CPU:NEON) on X86_64 linux
  
  $ sudo apt install scons
  
  $ sudo apt install gcc-aarh64-linux-gnu g++aarch64-linux-gnu
  
  $ scons Werror=1 -j4 debug=0 neon=1 opencl=0 os=linux arch=arm64-v8a openmp=yes

  # cross compile 
  # for Raspberry Pi 2/3 (ARM32 CPU:NEON) on X86_64 linux
  $ sudo apt install scons
  
  $ sudo apt install gcc-arm-linux-gnueabihf
  
  $ sudo apt install g++-arm-linux-gnueabihf
  
  $ sudo apt install binutils-arm-linux-gnueabihf
  
  $ scons Werror=1 -j4 debug=0 neon=1 opencl=0 os=linux arch=armv7a openmp=yes extra_cxx_flags=-fPIC

## Arm NN SDK          : v19.02
  - https://github.com/ARM-software/armnn/tree/v19.02

  $ git clone -b v19.02 https://github.com/ARM-software/armnn v19.02

  $ cd v19.02

  $ mkdir build

  $ cd build
