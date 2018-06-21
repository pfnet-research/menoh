#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <vector>

#include <menoh/array.hpp>
#include <menoh/dtype.hpp>
#include <menoh/model_data.hpp>
#include <menoh/model_factory.hpp>
#include <menoh/onnx.hpp>

namespace menoh_impl {
    namespace {
        class VGG16Test : public ::testing::Test {};

        TEST_F(VGG16Test, make_model) {
            std::vector<std::string> required_output_name_list{
              "140326201104648", // conv1_1 without relu
              "140326201105432", // conv1_1
              "140326201105600", // conv1_2
              "140326429223512", // max_pool1
            };
            auto model_data =
              menoh_impl::load_onnx("../data/VGG16.onnx", required_output_name_list);
            auto input_name_list = extract_model_input_name_list(model_data);
            auto const& input_name = input_name_list.front();
            constexpr auto batch_size = 1;
            constexpr auto channel_num = 3;
            constexpr auto height = 224;
            constexpr auto width = 224;
            std::vector<int> input_dims{batch_size, channel_num, height, width};
            auto output_dims_table = make_output_dims_table(
              model_data, {
                            {input_name, input_array.dims()},
                          });
            auto input_array = uniforms(dtype_t::float_, input_dims, 1.);
            auto model = make_model(
              {
                {input_name, input_array},
              },
              {
                {softmax_out_name, input_array},
              },
              model_data, "mkldnn");
            auto output_table = model->run();
            for(auto const& name : required_output_name_list) {
                auto const& output_arr = find_value(output_table, name);
                for(int i = 0; i < 2000; ++i) {
                    std::cout << fat(output_arr, i) << " ";
                }
                std::cout << "\n";
            }
        }
    } // namespace
} // namespace menoh_impl
