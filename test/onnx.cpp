#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <numeric>
#include <string>

#include <menoh/model_data.hpp>
#include <menoh/onnx.hpp>

namespace menoh_impl {
    namespace {
        class ONNXTest : public ::testing::Test {};

        TEST_F(ONNXTest, load_onnx_model) {
            auto model_data = menoh_impl::load_onnx("../data/VGG16.onnx");
            std::cout << "param table" << std::endl;
            std::cout << "node list size: " << model_data.node_list.size()
                      << std::endl;
            for(auto name_and_arr : model_data.parameter_name_and_array_list) {
                std::string name;
                array arr;
                std::tie(name, arr) = name_and_arr;
                std::cout << name << " ";
                for(auto d : arr.dims()) {
                    std::cout << d << " ";
                }
                std::cout << "\n";
            }
            std::cout << "graph" << std::endl;
            for(auto const& node : model_data.node_list) {
                std::cout << node.op_type << " ";
            }
            std::cout << "\n";

            auto input_name_list = extract_model_input_name_list(model_data);
            std::cout << "input_name_set" << std::endl;
            for(auto const& input_name : input_name_list) {
                std::cout << input_name << std::endl;
            }

            auto output_name_list = extract_model_output_name_list(model_data);
            std::cout << "output_name_set" << std::endl;
            for(auto const& output_name : output_name_list) {
                std::cout << output_name << std::endl;
            }
        }

    } // namespace
} // namespace menoh_impl
