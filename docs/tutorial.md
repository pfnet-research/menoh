# Tutorial

In this tutorial, we are going to make a CNN model inference software.

That software loads `data/VGG16.onnx` and takes input image, then outputs classification result.

We start simple hello world code:

```cpp
#include <iostream>

int main(int argc, char** argv) {
    std::cout << "Hello Menoh" << std::endl;
}
```

## Preprocessing input

First of all, preprocessing input is required. `data/VGG16.onnx` takes 3 channels 224 x 224 sized image but input image
is not always sized 224x224. So we use `resize()` function in OpenCV :

```cpp
cv::resize(image_mat, image_mat, cv::Size(224, 224));
```

VGG16 supposes that the input image is subtracted the mean values of images included in imagenet.

```cpp
image_mat.convertTo(image_mat, CV_32FC3); // change data type to float
image_mat -= cv::Scalar(103.939, 116.779, 123.68); // subtract BGR mean
```

Menoh takes images as NCHW format(N x Channels x Height x Width), but `Mat` of OpenCV holds image as HWC format(Height x Width x Channels).

So next we define *reorder_to_chw*.

```cpp
auto reorder_to_chw(cv::Mat const&mat) {
    assert(mat.channels() == 3);
    std::vector<float> data(mat.channels() * mat.rows * mat.cols);
    for(int y = 0; y < mat.rows; ++y) {
        for(int x = 0; x < mat.cols; ++x) {
            for(int c = 0; c < mat.channels(); ++c) {
                data[c * (mat.rows * mat.cols) + y * mat.cols + x] =
                  mat.at<cv::Vec3f>(y, x)[c];
            }
        }
    }
    return data;
}
```

The main code of preprocessing input is:

```cpp
// define input dims
const int batch_size = 1;
const int channels_num = 3;
const int height = 224;
const int  width = 224;

// Preprocessing input image
cv::resize(image_mat, image_mat, cv::Size(width, height));
image_mat.convertTo(image_mat, CV_32FC3);
image_mat -= cv::Scalar(123.68, 116.779, 103.939);
auto image_data = reorder_to_chw(image_mat);
```

## Setup model
ONNX model has some named variables. To build model, we have to set names of input variables and output variables.

We can checks them with [Netron](https://github.com/lutzroeder/Netron):

```
netron ../data/VGG16.onnx
```

Then you can see the content by accessing *localhost:8080* with browser like below.

\image html vgg16_view.png

VGG16 has one input and one output. So now we can check that the input name is *140326425860192* (input of 0:Conv) and the output name is *140326200803680* (output of 39:Softmax).

Some of we are interested in the feature vector of input image. So in addition, we are going to take the output of 32:FC(fc6, which is the first FC layer after CNNs) named *140326200777584*.

We define name aliases for convenience:

```cpp
std::string input_name = "140326425860192";
std::string fc6_output_name = "140326200777584";
std::string softmax_output_name = "140326200803680";
```

We load model data from ONNX file:

```cpp
menoh::model_data model_data = menoh::load_onnx("../data/VGG16.onnx");
```

To build the model, we have to build variable_profile.

To build variable_profile, we make variable_profile_builder.

```cpp
menoh::variable_profile_table_builder vpt_builder;
```

We add information of variables.

```cpp
vpt_builder.add_input_profile(conv1_1_in_name, menoh::dtype_t::float_,
                              {batch_size, channel_num, height, width});
vpt_builder.add_output_profile(fc6_out_name, menoh::dtype_t::float_);
vpt_builder.add_output_profile(softmax_out_name, menoh::dtype_t::float_);
```

Then build variable_profile_table.

```cpp
auto vpt = vpt_builder.build_variable_profile_table(model_data);
```

*variable_profile_table* has the dimensions of output variables calculated from the dimensions of input variable.
So we can get output dims from variable_profile_table.

```cpp
auto fc6_dims = vpt.get_variable_profile(fc6_out_name).dims;
```

Let's prepare fc6 data buffer.

```cpp
std::vector<float> fc6_out_data(std::accumulate(
  fc6_dims.begin(), fc6_dims.end(), 1, std::multiplies<int32_t>()));
```

We now can make model_builder.

```cpp
menoh::model_builder model_builder(vpt);
```

We can specify which data buffer is used for target variable by attaching.

```cpp
model_builder.attach_external_buffer(conv1_1_in_name,
                                     static_cast<void*>(image_data.data()));
model_builder.attach_external_buffer(
    fc6_out_name, static_cast<void*>(fc6_out_data.data()));
```

For softmax_out_name variable, no buffer is attached here. Don't worry.

An intenal buffer is attached to softmax_out_name variable automatically.

And we can get that buffer handle later.

Let's build the model.

```cpp
auto model = model_builder.build_model(model_data, "mkldnn");
model_data
    .reset(); // you can delete model_data explicitly after model building
```


## Run inference and get result

Now we can run inference.

```cpp
// Run inference
model.run();
```

We can take output variable by calling get_variable.

```cpp
// Get buffer pointer of output
auto softmax_output_var = model.get_variable(softmax_out_name);
float* softmax_output_buff =
    static_cast<float*>(softmax_output_var.buffer_handle);
```

That's it.

You can see the full code in `example/vgg16_example_in_cpp.cpp`.
