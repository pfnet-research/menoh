#if defined(_WIN32) || defined(__WIN32__)
#include <Windows.h>
#endif

#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <menoh/menoh.hpp>

#include "../external/cmdline.h"

auto reorder_bgr_hwc_to_rgb_chw(cv::Mat const& mat) {
    assert(mat.channels() == 3);
    std::vector<float> data(mat.channels() * mat.rows * mat.cols);
    for(int y = 0; y < mat.rows; ++y) {
        for(int x = 0; x < mat.cols; ++x) {
            for(int c = 0; c < mat.channels(); ++c) {
                data[c * (mat.rows * mat.cols) + y * mat.cols + x] =
                  mat.at<cv::Vec3f>(y, x)[2 - c];
            }
        }
    }
    return data;
}

template <typename InIter>
auto extract_top_k_index_list(
  InIter first, InIter last,
  typename std::iterator_traits<InIter>::difference_type k) {
    using diff_t = typename std::iterator_traits<InIter>::difference_type;
    std::priority_queue<
      std::pair<typename std::iterator_traits<InIter>::value_type, diff_t>>
      q;
    for(diff_t i = 0; first != last; ++first, ++i) {
        q.push({*first, i});
    }
    std::vector<diff_t> indices;
    for(diff_t i = 0; i < k; ++i) {
        indices.push_back(q.top().second);
        q.pop();
    }
    return indices;
}

auto load_category_list(std::string const& synset_words_path) {
    std::ifstream ifs(synset_words_path);
    if(!ifs) {
        throw std::runtime_error("File open error: " + synset_words_path);
    }
    std::vector<std::string> categories;
    std::string line;
    while(std::getline(ifs, line)) {
        categories.push_back(std::move(line));
    }
    return categories;
}

int main(int argc, char** argv) {
    std::cout << "vgg16 example" << std::endl;

    // Aliases to onnx's node input and output tensor name
    // Please use [Netron](https://github.com/lutzroeder/Netron)
    // See Menoh tutorial for more information.
    const std::string conv1_1_in_name = "Input_0";
    const std::string fc6_out_name = "Gemm_0";
    const std::string softmax_out_name = "Softmax_0";

    const int batch_size = 1;
    const int channel_num = 3;
    const int height = 224;
    const int width = 224;

    cmdline::parser a;
    a.add<std::string>("input_image", 'i', "input image path", false,
                       "../data/Light_sussex_hen.jpg");
    a.add<std::string>("model", 'm', "onnx model path", false,
                       "../data/vgg16.onnx");
    a.add<std::string>("synset_words", 's', "synset words path", false,
                       "../data/synset_words.txt");
    a.parse_check(argc, argv);

    auto input_image_path = a.get<std::string>("input_image");
    auto onnx_model_path = a.get<std::string>("model");
    auto synset_words_path = a.get<std::string>("synset_words");

    cv::Mat image_mat =
      cv::imread(input_image_path.c_str(), CV_LOAD_IMAGE_COLOR);
    if(!image_mat.data) {
        throw std::runtime_error("Invalid input image path: " +
                                 input_image_path);
    }

    // Preprocess
    cv::resize(image_mat, image_mat, cv::Size(width, height));
    image_mat.convertTo(image_mat, CV_32FC3);
    image_mat -= cv::Scalar(103.939, 116.779, 123.68); // subtract BGR mean
    auto image_data = reorder_bgr_hwc_to_rgb_chw(image_mat);

    // Load ONNX model data
    auto model_data = menoh::make_model_data_from_onnx(onnx_model_path);

    // Define input profile (name, dtype, dims) and output profile (name, dtype)
    // dims of output is automatically calculated later
    menoh::variable_profile_table_builder vpt_builder;
    vpt_builder.add_input_profile(conv1_1_in_name, menoh::dtype_t::float_,
                                  {batch_size, channel_num, height, width});
    vpt_builder.add_output_name(fc6_out_name);
    vpt_builder.add_output_name(softmax_out_name);

    // Build variable_profile_table and get variable dims (if needed)
    auto vpt = vpt_builder.build_variable_profile_table(model_data);
    auto fc6_dims = vpt.get_variable_profile(fc6_out_name).dims;
    std::vector<float> fc6_out_data(std::accumulate(
      fc6_dims.begin(), fc6_dims.end(), 1, std::multiplies<int32_t>()));

    model_data.optimize(vpt);

    // Make model_builder and attach extenal memory buffer
    // Variables which are not attached external memory buffer here are attached
    // internal memory buffers which are automatically allocated
    menoh::model_builder model_builder(vpt);
    model_builder.attach_external_buffer(conv1_1_in_name,
                                         static_cast<void*>(image_data.data()));
    model_builder.attach_external_buffer(
      fc6_out_name, static_cast<void*>(fc6_out_data.data()));

    // Build model
    auto model = model_builder.build_model(
      model_data, "composite_backend",
      R"({"backends":[{"type":"mkldnn"}, {"type":"generic"}], "log_output":"stdout"})");
    model_data
      .reset(); // you can delete model_data explicitly after model building

    // Get buffer pointer of output
    auto softmax_output_var = model.get_variable(softmax_out_name);
    float* softmax_output_buff =
      static_cast<float*>(softmax_output_var.buffer_handle);

    // Run inference
    try {
        model.run();
    } catch(std::exception const& e) { std::cout << e.what() << std::endl; }

    // Get output
    for(int i = 0; i < 10; ++i) {
        std::cout << fc6_out_data.at(i) << " ";
    }
    std::cout << "\n";
    auto categories = load_category_list(synset_words_path);
    auto top_k = 5;
    auto top_k_indices = extract_top_k_index_list(
      softmax_output_buff, softmax_output_buff + softmax_output_var.dims.at(1),
      top_k);
    std::cout << "top " << top_k << " categories are\n";
    for(auto ki : top_k_indices) {
        std::cout << ki << " " << *(softmax_output_buff + ki) << " "
                  << categories.at(ki) << std::endl;
    }
}
