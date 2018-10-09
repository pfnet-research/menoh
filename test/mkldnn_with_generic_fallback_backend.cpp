#include <gtest/gtest.h>

#include "backend.hpp"

namespace menoh {
    class MkldnnWithGenericFallbackBackendTest : public ::testing::Test {
    protected:
        MkldnnWithGenericFallbackBackendTest() = default;
        virtual void SetUp() {}
    };

    TEST_F(MkldnnWithGenericFallbackBackendTest, gemm_1d_test) {
        gemm_test("mkldnn_with_generic_fallback", R"({"log_output": "stdout"})",
                  "../data/random_input_3_4096.txt",
                  "../data/random_weight_256_4096.txt",
                  "../data/random_bias_256.txt",
                  "../data/linear_1d_w256_4096_b_256.txt"); //, 1, 1, 0, 1);
    }
    TEST_F(MkldnnWithGenericFallbackBackendTest, gemm_2d_test) {
        gemm_test("mkldnn_with_generic_fallback", R"({"log_output": "stdout"})",
                  "../data/random_input_3_4_32_32.txt",
                  "../data/random_weight_256_4096.txt",
                  "../data/random_bias_256.txt",
                  "../data/linear_2d_w256_4096_b_256.txt"); //, 1, 1, 0, 1);
    }

    TEST_F(MkldnnWithGenericFallbackBackendTest, gemm_1d_relu_test) {
        std::string backend_name = "mkldnn_with_generic_fallback";
        std::string backend_config = R"({"log_output": "stdout"})";
        menoh::model_data model_data;
        menoh::variable_profile_table_builder vpt_builder;
        std::pair<std::string, std::string> gemm_true_output_filename = {
          "gemm_out", "../data/linear_2d_w256_4096_b_256.txt"};
        std::pair<std::string, std::string> relu_true_output_filename = {
          "relu_out", "../data/linear_2d_w256_4096_b_256.txt"};
        float eps = 10.e-4;
        std::unordered_map<std::string, std::vector<float>> input_table;

        {
            model_data.add_new_node("Gemm");
            model_data.add_attribute_int_to_current_node("transB", 1);

            std::vector<std::pair<std::string, std::string>>
              input_filename_table = {
                {"input", "../data/random_input_3_4_32_32.txt"},
                {"weight", "../data/random_weight_256_4096.txt"},
                {"bias", "../data/random_bias_256.txt"}};

            for(auto const& p : input_filename_table) {
                std::string input_name;
                std::string input_filename;
                std::tie(input_name, input_filename) = p;
                model_data.add_input_name_to_current_node(input_name);
                std::vector<int32_t> input_dims;
                std::vector<float> input_data;
                std::tie(std::ignore, input_dims, input_data) =
                  menoh_impl::load_np_array(input_filename);
                input_table.emplace(input_name, input_data);
                dtype_t dtype = dtype_t::float_; // TODO other dtype
                model_data.add_parameter(input_name, dtype, input_dims,
                                         input_table.at(input_name).data());
                if(input_name !=
                   "bias") { // FIXME bias is 1d so vpt_builder can't take it.
                    vpt_builder.add_input_profile(input_name, dtype,
                                                  input_dims);
                }
            }

            std::string output_name;
            std::tie(output_name, std::ignore) = gemm_true_output_filename;
            model_data.add_output_name_to_current_node(output_name);
            dtype_t dtype = dtype_t::float_; // TODO other dtype
            vpt_builder.add_output_profile(output_name, dtype);
        }

        {
            model_data.add_new_node("Relu");
            model_data.add_input_name_to_current_node("gemm_out");
            model_data.add_output_name_to_current_node("relu_out");
            vpt_builder.add_output_profile("relu_out", dtype_t::float_);
        }

        auto vpt = vpt_builder.build_variable_profile_table(model_data);
        model_builder model_builder(vpt);
        auto model =
          model_builder.build_model(model_data, backend_name, backend_config);
        model.run();

        {
            auto output_var = model.get_variable("gemm_out");
            std::cout << "gemm out "
                      << *static_cast<float*>(output_var.buffer_handle)
                      << std::endl;
        }
        {
            auto output_var = model.get_variable("relu_out");
            std::cout << "relu out "
                      << *static_cast<float*>(output_var.buffer_handle)
                      << std::endl;
        }

        std::unordered_map<std::string, std::string> true_output_filename_table{
          {gemm_true_output_filename, relu_true_output_filename}};
        for(auto const& p : true_output_filename_table) {
            auto const& output_name = p.first;
            std::cout << output_name << std::endl;

            auto const& true_output_filename = p.second;
            std::vector<int32_t> output_dims;
            std::vector<float> true_output_data;
            std::tie(std::ignore, output_dims, true_output_data) =
              menoh_impl::load_np_array(true_output_filename);
            dtype_t dtype = dtype_t::float_; // TODO other dtype
            if(output_name == "relu_out") {
                std::transform(true_output_data.begin(), true_output_data.end(),
                               true_output_data.begin(),
                               [](auto e) { return std::max(e, 0.f); });
            }

            auto output_var = model.get_variable(output_name);
            menoh_impl::assert_near_list(
              static_cast<float*>(output_var.buffer_handle),
              static_cast<float*>(output_var.buffer_handle) +
                std::accumulate(output_var.dims.begin(), output_var.dims.end(),
                                1, std::multiplies<>()),
              true_output_data.begin(), true_output_data.end(), eps);
        }
    }

    TEST_F(MkldnnWithGenericFallbackBackendTest, relu_1d_test) {
        relu_test("mkldnn_with_generic_fallback", R"({"log_output": "stdout"})",
                  "../data/random_input_3_4096.txt", "../data/relu_1d.txt");
    }
} // namespace menoh
