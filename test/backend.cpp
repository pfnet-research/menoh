#include <gtest/gtest.h>

#include <unordered_map>

#include <menoh/menoh.hpp>

#include "common.hpp"
#include "np_io.hpp"

namespace menoh {
    class BackendTest : public ::testing::Test {
    protected:
        BackendTest() = default;
        virtual void SetUp() {}

        auto
        operator_test(std::string const& backend_name,
                      std::string const& backend_config,
                      menoh::model_data& model_data,
                      std::vector<std::pair<std::string, std::string>> const&
                        input_filename_table,
                      std::unordered_map<std::string, std::string> const&
                        true_output_filename_table,
                      float eps = 10.e-4) {

            menoh::variable_profile_table_builder vpt_builder;

            std::unordered_map<std::string,
                               std::tuple<std::vector<int>, std::vector<float>>>
              input_table;
            for(auto const& p : input_filename_table) {
                auto const& input_name = p.first;
                auto const& input_filename = p.second;

                model_data.add_input_name_to_current_node(input_name);

                std::vector<int32_t> input_dims;
                std::vector<float> data;
                std::tie(std::ignore, input_dims, data) =
                  menoh_impl::load_np_array(input_filename);
                dtype_t dtype = dtype_t::float_; // TODO other dtype
                input_table.insert({input_name, {input_dims, data}});
                // vpt_builder.add_input_profile(input_name, dtype, input_dims);
            }

            for(auto const& p : input_filename_table) {
                auto const& input_name = p.first;
                // auto const& input_filename = p.second;
                dtype_t dtype = dtype_t::float_; // TODO other dtype
                model_data.add_initializer(
                  input_name, dtype, std::get<0>(input_table.at(input_name)),
                  std::get<1>(input_table.at(input_name)).data());
            }

            std::unordered_map<std::string, std::vector<float>>
              true_output_table;
            for(auto const& p : true_output_filename_table) {
                auto const& output_name = p.first;
                auto const& output_filename = p.second;
                model_data.add_output_name_to_current_node(output_name);

                std::vector<int32_t> output_dims;
                std::vector<float> data;
                std::tie(std::ignore, output_dims, data) =
                  menoh_impl::load_np_array(output_filename);
                dtype_t dtype = dtype_t::float_; // TODO other dtype
                true_output_table.insert({output_name, data});
                vpt_builder.add_output_profile(output_name, dtype);
            }

            auto vpt = vpt_builder.build_variable_profile_table(model_data);
            model_builder model_builder(vpt);
            auto model = model_builder.build_model(model_data, backend_name,
                                                   backend_config);
            model.run();

            for(auto const& p : true_output_table) {
                auto const& output_name = p.first;
                auto const& true_output = p.second;
                auto output_var = model.get_variable(output_name);
                menoh_impl::assert_near_list(
                  static_cast<float*>(output_var.buffer_handle),
                  static_cast<float*>(output_var.buffer_handle) +
                    std::accumulate(output_var.dims.begin(),
                                    output_var.dims.end(), 1,
                                    std::multiplies<>()),
                  true_output.begin(), true_output.end(), eps);
            }
        }

        auto add_test(std::vector<std::string> const& input_filename_list,
                      std::string const& true_output_filename) {
            menoh::model_data model_data;
            model_data.add_new_node("Add");
            std::vector<std::pair<std::string, std::string>> inputs;
            for(int32_t i = 0; i < input_filename_list.size(); ++i) {
                auto const& filename = input_filename_list.at(i);
                inputs.push_back({"input" + std::to_string(i), filename});
            }
            operator_test("naive", "", model_data, inputs,
                          {{"output", true_output_filename}});
        }
        auto concat_test(float axis,
                         std::vector<std::string> const& input_filename_list,
                         std::string const& true_output_filename) {
            menoh::model_data model_data;
            model_data.add_new_node("Concat");
            model_data.add_attribute_int_to_current_node("axis", axis);
            std::vector<std::pair<std::string, std::string>> inputs;
            for(int32_t i = 0; i < input_filename_list.size(); ++i) {
                auto const& filename = input_filename_list.at(i);
                inputs.push_back({"input" + std::to_string(i), filename});
            }
            operator_test("naive", "", model_data, inputs,
                          {{"output", true_output_filename}});
        }
        auto elu_test(float alpha, std::string const& input_filename,
                      std::string const& true_output_filename) {
            menoh::model_data model_data;
            model_data.add_new_node("Elu");
            model_data.add_attribute_float_to_current_node("alpha", alpha);
            operator_test("naive", "", model_data, {{"input", input_filename}},
                          {{"output", true_output_filename}});
        }
        auto gemm_test(std::string const& input_filename,
                       std::string const& weight_filename,
                       std::string const& bias_filename,
                       std::string const& true_output_filename) {
            menoh::model_data model_data;
            model_data.add_new_node("Gemm");
            model_data.add_attribute_int_to_current_node("transB", 1);
            operator_test("naive", "", model_data,
                          {{"input", input_filename},
                           {"weight", weight_filename},
                           {"bias", bias_filename}},
                          {{"output", true_output_filename}});
        }
        auto leaky_relu_test(float alpha, std::string const& input_filename,
                             std::string const& true_output_filename) {
            menoh::model_data model_data;
            model_data.add_new_node("LeakyRelu");
            model_data.add_attribute_float_to_current_node("alpha", alpha);
            operator_test("naive", "", model_data, {{"input", input_filename}},
                          {{"output", true_output_filename}});
        }
        auto relu_test(std::string const& input_filename,
                       std::string const& true_output_filename) {
            menoh::model_data model_data;
            model_data.add_new_node("Relu");
            operator_test("naive", "", model_data, {{"input", input_filename}},
                          {{"output", true_output_filename}});
        }
        auto sum_test(std::vector<std::string> const& input_filename_list,
                      std::string const& true_output_filename) {
            menoh::model_data model_data;
            model_data.add_new_node("Sum");
            std::vector<std::pair<std::string, std::string>> inputs;
            for(int32_t i = 0; i < input_filename_list.size(); ++i) {
                auto const& filename = input_filename_list.at(i);
                inputs.push_back({"input" + std::to_string(i), filename});
            }
            operator_test("naive", "", model_data, inputs,
                          {{"output", true_output_filename}});
        }
        auto tanh_test(std::string const& input_filename,
                       std::string const& true_output_filename) {
            menoh::model_data model_data;
            model_data.add_new_node("Tanh");
            operator_test("naive", "", model_data, {{"input", input_filename}},
                          {{"output", true_output_filename}});
        }
    };

    TEST_F(BackendTest, add_1d_test) {
        add_test({"../data/random_input_3_4096.txt",
                  "../data/random_input_3_4096.txt"},
                 "../data/add_1d.txt");
    }
    TEST_F(BackendTest, add_2d_test) {
        add_test({"../data/random_input_3_4_32_32.txt",
                  "../data/random_input_3_4_32_32.txt"},
                 "../data/add_2d.txt");
    }
    TEST_F(BackendTest, concat_test_axis0_2) {
        concat_test(0,
                    {"../data/random_input_3_4096.txt",
                     "../data/random_input_3_4096.txt"},
                    "../data/concat_1d_6_4096.txt");
    }
    TEST_F(BackendTest, concat_test_axis1_2) {
        concat_test(1,
                    {"../data/random_input_3_4096.txt",
                     "../data/random_input_3_4096.txt"},
                    "../data/concat_1d_3_8192.txt");
    }
    TEST_F(BackendTest, concat_test_axis0_3) {
        concat_test(0,
                    {"../data/random_input_3_4096.txt",
                     "../data/random_input_3_4096.txt",
                     "../data/random_input_3_4096.txt"},
                    "../data/concat_1d_9_4096.txt");
    }
    TEST_F(BackendTest, concat_test_axis1_3) {
        concat_test(1,
                    {"../data/random_input_3_4096.txt",
                     "../data/random_input_3_4096.txt",
                     "../data/random_input_3_4096.txt"},
                    "../data/concat_1d_3_12288.txt");
    }
    TEST_F(BackendTest, elu_1d_test) {
        elu_test(1.1, "../data/random_input_3_4096.txt", "../data/elu_1d.txt");
    }
    TEST_F(BackendTest, elu_2d_test) {
        elu_test(1.1, "../data/random_input_3_4_32_32.txt",
                 "../data/elu_2d.txt");
    }
    TEST_F(BackendTest, gemm_1d_test) {
        gemm_test("../data/random_input_3_4096.txt",
                  "../data/random_weight_256_4096.txt",
                  "../data/random_bias_256.txt",
                  "../data/linear_1d_w256_4096_b_256.txt"); //, 1, 1, 0, 1);
    }
    TEST_F(BackendTest, gemm_2d_test) {
        gemm_test("../data/random_input_3_4_32_32.txt",
                  "../data/random_weight_256_4096.txt",
                  "../data/random_bias_256.txt",
                  "../data/linear_2d_w256_4096_b_256.txt"); //, 1, 1, 0, 1);
    }
    TEST_F(BackendTest, leaky_relu_1d_test) {
        leaky_relu_test(0.001, "../data/random_input_3_4096.txt",
                        "../data/leaky_relu_1d.txt");
    }
    TEST_F(BackendTest, leaky_relu_2d_test) {
        leaky_relu_test(0.001, "../data/random_input_3_4_32_32.txt",
                        "../data/leaky_relu_2d.txt");
    }
    TEST_F(BackendTest, relu_1d_test) {
        relu_test("../data/random_input_3_4096.txt", "../data/relu_1d.txt");
    }
    TEST_F(BackendTest, relu_2d_test) {
        relu_test("../data/random_input_3_4_32_32.txt", "../data/relu_2d.txt");
    }
    TEST_F(BackendTest, sum_1d_test) {
        sum_test({"../data/random_input_3_4096.txt",
                  "../data/random_input_3_4096.txt"},
                 "../data/add_1d.txt"); // TODO change from add
    }
    TEST_F(BackendTest, sum_2d_test) {
        sum_test({"../data/random_input_3_4_32_32.txt",
                  "../data/random_input_3_4_32_32.txt"},
                 "../data/add_2d.txt"); // TODO change from add
    }
    TEST_F(BackendTest, tanh_1d_test) {
        tanh_test("../data/random_input_3_4096.txt", "../data/tanh_1d.txt");
    }
    TEST_F(BackendTest, tanh_2d_test) {
        tanh_test("../data/random_input_3_4_32_32.txt", "../data/tanh_2d.txt");
    }
} // namespace menoh
