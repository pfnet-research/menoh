#ifndef MENOH_TEST_BACKEND_HPP
#define MENOH_TEST_BACKEND_HPP

#include <unordered_map>

#include <menoh/menoh.hpp>

#include "common.hpp"
#include "np_io.hpp"

namespace menoh {
    inline auto
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
            if(input_dims.size() == 2 ||
               input_dims.size() == 4) { // FIXME dirty hack
                vpt_builder.add_input_profile(input_name, dtype, input_dims);
            }
        }

        for(auto const& p : input_filename_table) {
            auto const& input_name = p.first;
            // auto const& input_filename = p.second;
            dtype_t dtype = dtype_t::float_; // TODO other dtype
            model_data.add_parameter(
              input_name, dtype, std::get<0>(input_table.at(input_name)),
              std::get<1>(input_table.at(input_name)).data());
        }

        std::unordered_map<std::string, std::vector<float>> true_output_table;
        for(auto const& p : true_output_filename_table) {
            auto const& output_name = p.first;
            auto const& output_filename = p.second;
            model_data.add_output_name_to_current_node(output_name);

            std::vector<int32_t> output_dims;
            std::vector<float> data;
            std::tie(std::ignore, output_dims, data) =
              menoh_impl::load_np_array(output_filename);
            true_output_table.insert({output_name, data});
            vpt_builder.add_output_name(output_name);
        }

        auto vpt = vpt_builder.build_variable_profile_table(model_data);
        model_builder model_builder(vpt);
        auto model =
          model_builder.build_model(model_data, backend_name, backend_config);
        model.run();

        for(auto const& p : true_output_table) {
            auto const& output_name = p.first;
            auto const& true_output = p.second;
            auto output_var = model.get_variable(output_name);
            menoh_impl::assert_near_list(
              static_cast<float*>(output_var.buffer_handle),
              static_cast<float*>(output_var.buffer_handle) +
                std::accumulate(output_var.dims.begin(), output_var.dims.end(),
                                1, std::multiplies<>()),
              true_output.begin(), true_output.end(), eps);
        }
    }

    inline auto add_test(std::string const& backend_name,
                         std::string const& config,
                         std::vector<std::string> const& input_filename_list,
                         std::string const& true_output_filename) {
        menoh::model_data model_data;
        model_data.add_new_node("Add");
        std::vector<std::pair<std::string, std::string>> inputs;
        for(int32_t i = 0; i < input_filename_list.size(); ++i) {
            auto const& filename = input_filename_list.at(i);
            inputs.push_back({"input" + std::to_string(i), filename});
        }
        operator_test(backend_name, config, model_data, inputs,
                      {{"output", true_output_filename}});
    }
    inline auto average_pool_test(std::string const& backend_name,
                                  std::string const& config,
                                  std::vector<int32_t> const& kernel_shape,
                                  std::vector<int32_t> const& pads,
                                  std::vector<int32_t> const& strides,
                                  std::string const& input_filename,
                                  std::string const& true_output_filename) {
        menoh::model_data model_data;
        model_data.add_new_node("AveragePool");
        model_data.add_attribute_ints_to_current_node("kernel_shape",
                                                      kernel_shape);
        model_data.add_attribute_ints_to_current_node("pads", pads);
        model_data.add_attribute_ints_to_current_node("strides", strides);
        operator_test(backend_name, config, model_data,
                      {{"input", input_filename}},
                      {{"output", true_output_filename}});
    }
    inline auto concat_test(std::string const& backend_name,
                            std::string const& config, float axis,
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
        operator_test(backend_name, config, model_data, inputs,
                      {{"output", true_output_filename}});
    }
    inline auto elu_test(std::string const& backend_name,
                         std::string const& config, float alpha,
                         std::string const& input_filename,
                         std::string const& true_output_filename) {
        menoh::model_data model_data;
        model_data.add_new_node("Elu");
        model_data.add_attribute_float_to_current_node("alpha", alpha);
        operator_test(backend_name, config, model_data,
                      {{"input", input_filename}},
                      {{"output", true_output_filename}});
    }
    inline auto gemm_test(std::string const& backend_name,
                          std::string const& config,
                          std::string const& input_filename,
                          std::string const& weight_filename,
                          std::string const& bias_filename,
                          std::string const& true_output_filename) {
        menoh::model_data model_data;
        model_data.add_new_node("Gemm");
        model_data.add_attribute_int_to_current_node("transB", 1);
        operator_test(backend_name, config, model_data,
                      {{"input", input_filename},
                       {"weight", weight_filename},
                       {"bias", bias_filename}},
                      {{"output", true_output_filename}});
    }
    inline auto gemm_relu_test(std::string const& backend_name,
                               std::string const& config,
                               std::string const& input_filename,
                               std::string const& weight_filename,
                               std::string const& bias_filename,
                               std::string const& true_output_filename) {
        menoh::model_data model_data;
        model_data.add_new_node("Gemm");
        model_data.add_attribute_int_to_current_node("transB", 1);
        model_data.add_new_node("Relu");
        operator_test(backend_name, config, model_data,
                      {{"input", input_filename},
                       {"weight", weight_filename},
                       {"bias", bias_filename}},
                      {{"output", true_output_filename}});
    }
    inline auto leaky_relu_test(std::string const& backend_name,
                                std::string const& config, float alpha,
                                std::string const& input_filename,
                                std::string const& true_output_filename) {
        menoh::model_data model_data;
        model_data.add_new_node("LeakyRelu");
        model_data.add_attribute_float_to_current_node("alpha", alpha);
        operator_test(backend_name, config, model_data,
                      {{"input", input_filename}},
                      {{"output", true_output_filename}});
    }
    inline auto max_pool_test(std::string const& backend_name,
                              std::string const& config,
                              std::vector<int32_t> const& kernel_shape,
                              std::vector<int32_t> const& pads,
                              std::vector<int32_t> const& strides,
                              std::string const& input_filename,
                              std::string const& true_output_filename) {
        menoh::model_data model_data;
        model_data.add_new_node("MaxPool");
        model_data.add_attribute_ints_to_current_node("kernel_shape",
                                                      kernel_shape);
        model_data.add_attribute_ints_to_current_node("pads", pads);
        model_data.add_attribute_ints_to_current_node("strides", strides);
        operator_test(backend_name, config, model_data,
                      {{"input", input_filename}},
                      {{"output", true_output_filename}});
    }
    inline auto relu_test(std::string const& backend_name,
                          std::string const& config,
                          std::string const& input_filename,
                          std::string const& true_output_filename) {
        menoh::model_data model_data;
        model_data.add_new_node("Relu");
        operator_test(backend_name, config, model_data,
                      {{"input", input_filename}},
                      {{"output", true_output_filename}});
    }
    inline auto softmax_test(std::string const& backend_name,
                             std::string const& config, int32_t axis,
                             std::string const& input_filename,
                             std::string const& true_output_filename) {
        menoh::model_data model_data;
        model_data.add_new_node("Softmax");
        model_data.add_attribute_int_to_current_node("axis", axis);
        operator_test(backend_name, config, model_data,
                      {{"input", input_filename}},
                      {{"output", true_output_filename}});
    }
    inline auto sum_test(std::string const& backend_name,
                         std::string const& config,
                         std::vector<std::string> const& input_filename_list,
                         std::string const& true_output_filename) {
        menoh::model_data model_data;
        model_data.add_new_node("Sum");
        std::vector<std::pair<std::string, std::string>> inputs;
        for(int32_t i = 0; i < input_filename_list.size(); ++i) {
            auto const& filename = input_filename_list.at(i);
            inputs.push_back({"input" + std::to_string(i), filename});
        }
        operator_test(backend_name, config, model_data, inputs,
                      {{"output", true_output_filename}});
    }
    inline auto tanh_test(std::string const& backend_name,
                          std::string const& config,
                          std::string const& input_filename,
                          std::string const& true_output_filename) {
        menoh::model_data model_data;
        model_data.add_new_node("Tanh");
        operator_test(backend_name, config, model_data,
                      {{"input", input_filename}},
                      {{"output", true_output_filename}});
    }

} // namespace menoh

#endif // MENOH_TEST_BACKEND_HPP
