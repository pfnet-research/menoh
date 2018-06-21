# Getting started

## Requirement

The following softwares are required to install Menoh.

- [MKL-DNN](https://github.com/intel/mkl-dnn)
- [ProtocolBuffers](https://developers.google.com/protocol-buffers/)
- [OpenCV](https://opencv.org/)(optional, for building examples)
- [Chainer](https://chainer.org/)(optional, for building tests)

## Build and Install Menoh

Git clone Menoh:

```
git clone ... # Menoh repo
```

Then, on root directory of Menoh:

```
mkdir build
cd build
cmake ..
make # or make -j2
```

It builds library, examples and tests.

To install Menoh, type below command **with root privileges**:

```
make install
```

## Run VGG16 Example

Execute below command in root directory:

```
python retrieve_data.py
cd build
./example/vgg16_example
```

Result is below

```
vgg16 example
-22.3708 -34.4082 -10.218 24.2962 -0.252342 -8.004 -27.0804 -23.0728 -7.05607 16.1343
top 5 categories are
8 0.96132 n01514859 hen
7 0.0369939 n01514668 cock
86 0.00122795 n01807496 partridge
82 0.000225824 n01797886 ruffed grouse, partridge, Bonasa umbellus
97 3.83677e-05 n01847000 drake
```

Please give `--help` option for details

```
./example/vgg16_example.cpp --help
```

## Run Tests

Execute below commands in root directory:

```
python gen_test_data.py
cd build
cmake -DENABLE_TEST=ON ..
make
./test/menoh_test
```
