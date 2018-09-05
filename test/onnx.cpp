#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <numeric>
#include <string>

#include <menoh/model_data.hpp>
#include <menoh/onnx.hpp>

#include "common.hpp"

namespace menoh_impl {
    namespace {
        class ONNXTest : public ::testing::Test {
        public:
            std::string model_filename{"../data/VGG16.onnx"};
        };

        TEST_F(ONNXTest, make_model_data_from_onnx_file) {
            auto model_data =
              menoh_impl::make_model_data_from_onnx_file(model_filename);
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

        TEST_F(ONNXTest, make_model_data_from_onnx_data_on_memory) {
            std::ifstream input(model_filename, std::ios::binary);
            std::istreambuf_iterator<char> begin(input);
            std::istreambuf_iterator<char> end;
            std::vector<char> buffer(begin, end);
            auto model_data_from_memory =
              make_model_data_from_onnx_data_on_memory(
                static_cast<uint8_t*>(static_cast<void*>(buffer.data())),
                buffer.size());

            auto model_data_from_file =
              menoh_impl::make_model_data_from_onnx_file(model_filename);

            ASSERT_EQ(model_data_from_memory.node_list.size(),
                      model_data_from_file.node_list.size());

            std::sort(model_data_from_memory.node_list.begin(),
                      model_data_from_memory.node_list.end());
            std::sort(model_data_from_file.node_list.begin(),
                      model_data_from_file.node_list.end());
            ASSERT_EQ(model_data_from_memory.node_list,
                      model_data_from_file.node_list);

            ASSERT_EQ(
              model_data_from_memory.parameter_name_and_array_list.size(),
              model_data_from_file.parameter_name_and_array_list.size());

            for(auto const& p :
                model_data_from_memory.parameter_name_and_array_list) {
                auto param_iter = std::find_if(
                  model_data_from_file.parameter_name_and_array_list.begin(),
                  model_data_from_file.parameter_name_and_array_list.end(),
                  [&p](auto e) { return e.first == p.first; });

                // check if param which has same name is found
                ASSERT_NE(
                  param_iter,
                  model_data_from_file.parameter_name_and_array_list.end());

                // check if parameter has same values
                ASSERT_TRUE(is_near_array(p.second, param_iter->second));
            }
        }

    } // namespace
} // namespace menoh_impl
