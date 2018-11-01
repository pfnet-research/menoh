#include <chrono>
#include <iostream>
#include <queue>
#include <vector>

#include <menoh/menoh.hpp>

#include "../external/cmdline.h"
#include "../test/np_io.hpp"

int main(int argc, char** argv) {
    using clock = std::chrono::high_resolution_clock;
    cmdline::parser a;
    a.add<std::string>("input", '\0', "input_data");
    a.add<std::string>("model", '\0', "onnx model path", false,
                       "../data/vgg16.onnx");
    a.add<int>("iteration", '\0', "number of iterations", false, 1);
    a.parse_check(argc, argv);

    constexpr auto category_num = 1000;
    auto input_data = menoh_impl::load_np_array_as_array(a.get<std::string>("input"));
    auto batch_size = input_data.dims().at(0);
    menoh_impl::array output_data(menoh_impl::dtype_t::float_, {batch_size, category_num});

    auto input_image_path = a.get<std::string>("input");
    auto onnx_model_path = a.get<std::string>("model");

    // Aliases to onnx's node input and output tensor name
    auto conv1_1_in_name = "Input_0";
    auto softmax_out_name = "Softmax_0";

    // Load ONNX model data
    auto model_data = menoh::make_model_data_from_onnx(onnx_model_path);

    // Make variable_profile_table
    menoh::variable_profile_table_builder vpt_builder;
    vpt_builder.add_input_profile(conv1_1_in_name, menoh::dtype_t::float_,
                                  {batch_size, 3, 224, 224});
    vpt_builder.add_output_name(softmax_out_name);
    auto vpt = vpt_builder.build_variable_profile_table(model_data);

    // Build model
    menoh::model_builder model_builder(vpt);
    auto model = model_builder.build_model(model_data, "mkldnn_with_generic_fallback");
    model_data
      .reset(); // you can delete model_data explicitly after model building

    /*
    // Get buffer pointer of output
    float* softmax_output_buff =
      static_cast<float*>(model.get_buffer_handle(softmax_out_name));
    */

    std::vector<std::chrono::microseconds> usecs;

    for (int i = 0; i < a.get<int>("iteration"); ++i) {
        auto start = clock::now();

        // Run inference
        model.run();

        auto end = clock::now();
        usecs.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
    }

    auto total_usec = std::accumulate(usecs.begin(), usecs.end(), std::chrono::microseconds::zero());
    std::cout << total_usec.count() / usecs.size() * 0.001 << " msec" << std::endl;
}
