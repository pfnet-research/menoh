#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <tuple>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <menoh/mkldnn/model_core.hpp>
#include <menoh/model.hpp>
#include <menoh/onnx.hpp>
#include <menoh/utility.hpp>

namespace menoh_impl {
    namespace {

        class MKLDNNTest : public ::testing::Test {};

        TEST_F(MKLDNNTest, run_onnx_model) {
            static mkldnn::engine engine(mkldnn::engine::cpu, 0);
            auto model_data = load_onnx("../data/VGG16.onnx");
            // model_data = trim_redundant_nodes(model_data);

            auto input_name_list = extract_model_input_name_list(model_data);
            if(input_name_list.size() != 1) {
                throw std::runtime_error(
                  "VGG16 data is invalid: input name list size is " +
                  std::to_string(input_name_list.size()));
            }
            auto const& input_name = input_name_list.front();
            constexpr auto batch_size = 1;
            constexpr auto channel_num = 3;
            constexpr auto height = 224;
            constexpr auto width = 224;
            std::vector<int> input_dims{batch_size, channel_num, height, width};
            array input_arr(dtype_t::float_, input_dims);

            auto output_name_list = extract_model_output_name_list(model_data);
            if(output_name_list.size() != 1) {
                throw std::runtime_error(
                  "VGG16 data is invalid: input name list size is " +
                  std::to_string(output_name_list.size()));
            }
            auto const& output_name = output_name_list.front();
            auto output_dims_table =
              make_output_dims_table(model_data, {{input_name, input_dims}});
            array output_arr(dtype_t::float_,
                             find_value(output_dims_table, output_name));
            auto model = menoh_impl::mkldnn_backend::model_core(
              {
                {input_name, input_arr},
              },
              {
                {output_name, output_arr},
              },
              model_data, engine);
            model.run();
            auto max_i =
              std::max_element(fbegin(output_arr), fend(output_arr)) -
              fbegin(output_arr);
            std::cout << "max_i " << max_i << std::endl;
        }

        TEST_F(MKLDNNTest, make_mkldnn_model_with_invalid_backend_config) {
            // Aliases to onnx's node input and output tensor name
            auto conv1_1_in_name = "140326425860192";
            auto softmax_out_name = "140326200803680";

            constexpr auto batch_size = 1;
            constexpr auto channel_num = 3;
            constexpr auto height = 224;
            constexpr auto width = 224;
            // category_num is 1000

            std::vector<int> input_dims{batch_size, channel_num, height, width};

            // Load ONNX model data
            auto model_data = menoh_impl::load_onnx("../data/VGG16.onnx");

            auto cpu_count =
              mkldnn::engine::get_count(mkldnn::engine::kind::cpu);

            // Construct computation primitive list and memories
            ASSERT_THROW(
              menoh_impl::model(
                {{conv1_1_in_name, menoh_impl::dtype_t::float_, input_dims,
                  nullptr}}, // table of input_name, dtype, input_dims and
                             // data_handle
                {{softmax_out_name, menoh_impl::dtype_t::float_,
                  nullptr}}, // list of output names, dtypes and data_handles
                model_data, "mkldnn",
                "{\"cpu_id\":" + std::to_string(cpu_count + 2) + "}"),
              menoh_impl::backend_error);
        }

    } // namespace
} // namespace menoh_impl
