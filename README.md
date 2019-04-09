# Menoh

[![travis](https://img.shields.io/travis/pfnet-research/menoh/master.svg)](https://travis-ci.org/pfnet-research/menoh) [![Build status](https://ci.appveyor.com/api/projects/status/luo2m9p5fg9jxjsh/branch/master?svg=true)](https://ci.appveyor.com/project/pfnet-research/menoh/branch/master)
[![Coverity Scan Build Status](https://scan.coverity.com/projects/16151/badge.svg)](https://scan.coverity.com/projects/pfnet-research-menoh)

Menoh is DNN inference library with C API.

Menoh is released under MIT License.

DISCLAIMER: Menoh is still experimental. Use it at your own risk.
In particular not all operators in ONNX are supported, so please check whether the operators used in your model are supported. We have checked that VGG16 and ResNet50 models converted by onnx-chainer work fine.

[Document](https://pfnet-research.github.io/menoh/)

This codebase contains C API and C++ API.

## Goal

- DNN Inference with CPU
- ONNX support
- Easy to use.

## Related Projects

- Chainer model to ONNX : [onnx-chainer](https://github.com/chainer/onnx-chainer)
- C# wrapper : [menoh-sharp](https://github.com/pfnet-research/menoh-sharp)
- Go wrapper : [go-menoh](https://github.com/pfnet-research/go-menoh)
  - (unofficial wrapper [gomenoh](https://github.com/kou-m/gomenoh) by kou-m san has been merged)
- Haskell wrapper : [menoh-haskell](https://github.com/pfnet-research/menoh-haskell)
- Node.js wrapper : [node-menoh](https://github.com/pfnet-research/node-menoh)
- Ruby wrapper : [menoh-ruby](https://github.com/pfnet-research/menoh-ruby)
- Rust wrapper : [menoh-rs](https://github.com/pfnet-research/menoh-rs)
  - There is also [unofficial Rust wrapper by Y-Nak san](https://github.com/Y-Nak/menoh-rs)
- Java wrapper : [menoh-java](https://github.com/pfnet-research/menoh-java)
- [Unofficial] ROS interface by Akio Ochiai san : [menoh_ros](https://github.com/akio/menoh_ros)
- [Unofficial] OCaml wrapper by wkwkes san : [Menohcaml](https://github.com/wkwkes/Menohcaml)

# Installation using package manager or binary packages

- For Windows users, prebuild libraries are available (see [release](https://github.com/pfnet-research/menoh/releases)) and [Nuget package](https://www.nuget.org/packages/Menoh/) is available.
- For macOS user, [Homebrew tap repository](https://github.com/pfnet-research/homebrew-menoh) is available.
- For Ubuntu user, binary packages are available.
    ```
    $ curl -LO https://github.com/pfnet-research/menoh/releases/download/v1.1.1/ubuntu1604_mkl-dnn_0.16-1_amd64.deb
    $ curl -LO https://github.com/pfnet-research/menoh/releases/download/v1.1.1/ubuntu1604_menoh_1.1.1-1_amd64.deb
    $ curl -LO https://github.com/pfnet-research/menoh/releases/download/v1.1.1/ubuntu1604_menoh-dev_1.1.1-1_amd64.deb
    $ sudo apt install ./ubuntu1604_*_amd64.deb
    ```
    If you are using Ubuntu 18.04, please replace `1604` with `1804`.

# Installation from source

## Requirements

- MKL-DNN Library (0.14 or later)
- Protocol Buffers (2.6.1 or later)

## Build

Execute following commands in root directory.

```
python scripts/retrieve_data.py
mkdir build && cd build
cmake .. #-DENABLE_TENSORRT=ON # to enable TensorRT
make
```

See [BUILDING.md](BUILDING.md) for details.

## Installation

Execute following command in build directory created at Build section.

```
make install
```

# Run VGG16 example (it can run ResNet-50 as well)

Execute following command in root directory.

```
./example/mkldnn_vgg16_example_in_cpp # for tensorrt, tensorrt_vgg16_example_in_cpp is available
```

Result is here

```
vgg16 example
-18.1883 -26.5022 -20.0474 13.5325 -0.107129 0.76102 -23.9688 -24.218 -21.6314 14.2164 
top 5 categories are
8 0.885836 n01514859 hen
7 0.104591 n01514668 cock
86 0.00313584 n01807496 partridge
82 0.000934658 n01797886 ruffed grouse, partridge, Bonasa umbellus
97 0.000839487 n01847000 drake

```

You can also run ResNet-50

```
./example/mkldnn_vgg16_example_in_cpp -m ../data/resnet50.onnx
```

Please give `--help` option for details

```
./example/mkldnn_vgg16_example_in_cpp --help
```


# Run test

Setup chainer

Then, execute following commands in root directory.

```
python scripts/gen_test_data.py
cd build
cmake -DENABLE_TEST=ON ..
make
./test/menoh_test.out
```

# Current supported backends

Users can specify backend and backend_config.

backend_config options are specified with JSON format.

e.g. For `tensorrt`, backend_config can be specified `{"device_id":2, "allow_fp16_mode":true, "enable_model_caching":false, "cached_model_dir":"/tmp"}`.

For more information, see `example/tensorrt_vgg16_example_in_cpp.cpp`.


## MKLDNN (Intel CPU)

backend name: `mkldnn_with_generic_fallback`

backend_config options:
- `cpu_id`: The specification of device id where computation is processed. The default value is 0.

## TensorRT (NVIDIA GPU)

backend name: `tensorrt`

backend_config options:
- `batch_size`: The specification of batch size of input data. This is optional and when it is omitted, the first dimension size of the first input specified add_input_profile() is set as batch size.
- `device_id`: The specification of device id where computation is processed. The default value is 0. 
- `enable_profiler`: Enabling profiling feature or not.
- `allow_fp16_mode`: Allowing to use FP16 kernel. It doesn’t guarantee FP16 kernel to be used. The default value is false. 
- `force_fp16_mode`: Forcing to use FP16 kernel. The default value is false. This option override allow_fp16_mode when this option is true.
- `enable_model_caching`: Enabling model caching or not. The default value is true.
- `cached_model_dir`: Specification of the directory where cached models placed and loaded. The default value is “.”. This option is ignored when enable_model_caching is false.

# Current supported operators

|                            |                    | MKLDNN (with generic) | TensorRT |
|----------------------------|--------------------|:---------------------:|:--------:|
| Constants                  | Const              |                       |     ○    |
|                            | Identity           |                       |     ○    |
| Activations                | Elu                |           ○           |          |
|                            | LeakyRelu          |           ○           |          |
|                            | Relu               |           ○           |     ○    |
|                            | Sigmoid            |           ○           |     ○    |
|                            | Softmax            |           ○           |     ○    |
|                            | Tanh               |           ○           |     ○    |
| Array manipurations        | Concat             |                       |     ○    |
|                            | Reshape            |           ○           |          |
|                            | Transpose          |           ○           |          |
|                            | Unsqueeze          |                       |     ○    |
| Neural network connections | Conv               |           ○           |     ○    |
|                            | ConvTranspose      |                       |          |
|                            | Gemm               |           ○           |     ○    |
| Mathematical functions     | Abs                |           ○           |          |
|                            | Add                |           ○           |     ○    |
|                            | Mul                |           ○           |     ○    |
|                            | Sqrt               |           ○           |          |
|                            | Sum                |           ○           |     ○    |
| Normalization functions    | BatchNormalization |           ○           |     ○    |
|                            | LRN                |                       |          |
| Spatial pooling            | AveragePool        |           ○           |     ○    |
|                            | GlobalAveragePool  |                       |     ○    |
|                            | GlobalMaxPool      |                       |     ○    |
|                            | MaxPool            |           ○           |     ○    |

# License

Menoh is released under MIT License. Please see the LICENSE file for details.

Pre-trained models downloaded via `retrieve_data.py` were converted by onnx-chainer. The original models were downloaded via [ChainerCV](https://github.com/chainer/chainercv).
Check `scripts/generate_vgg16_onnx.py` and `scripts/generate_resnet50_onnx.py` and see [the LICENSE of ChainerCV](https://chainercv.readthedocs.io/en/stable/license.html) about each terms of use of the pre-trained models. 
