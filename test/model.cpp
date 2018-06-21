#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <vector>

#include <mkldnn.hpp>

#include <menoh/model.hpp>
#include <menoh/onnx.hpp>

namespace menoh_impl {
    namespace {
        class ModelTest : public ::testing::Test {};

        TEST_F(ModelTest, make_model) {
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
            auto model_data = std::make_unique<menoh_impl::model_data>(
              menoh_impl::load_onnx("../data/VGG16.onnx"));

            // Construct computation primitive list and memories
            auto model = menoh_impl::model(
              {{conv1_1_in_name, dtype_t::float_, input_dims, nullptr}},
              {{softmax_out_name, dtype_t::float_, nullptr}}, *model_data,
              "mkldnn");
            model_data.reset(); // delete model_data

            model.run();
        }

    } // namespace
} // namespace menoh_impl
