#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "./common.hpp"
#include "np_io.hpp"

#include <menoh/dims.hpp>
#include <menoh/mkldnn/operator.hpp>
#include <menoh/mkldnn/utility.hpp>
#include <menoh/model_core.hpp>

namespace menoh_impl {
    namespace mkldnn_backend {

        auto load_np_array_as_memory(std::string const& filename,
                                     mkldnn::engine const& engine) {
            auto data = menoh_impl::load_np_array_as_array(filename);
            assert(data.dims().size() == 2 || data.dims().size() == 4);
            mkldnn::memory::format format = data.dims().size() == 2
                                              ? mkldnn::memory::format::nc
                                              : mkldnn::memory::format::nchw;
            mkldnn::memory memory(
              {{{data.dims()},
                dtype_to_mkldnn_memory_data_type(data.dtype()),
                format},
               engine});
            std::copy(fbegin(data), fend(data),
                      static_cast<float*>(memory.get_data_handle()));
            return memory;
        }

        class OperatorTest : public ::testing::Test {
        protected:
            OperatorTest() = default;
            virtual void SetUp() {}

            auto relu_test(std::string const& input_filename,
                           std::string const& true_output_filename) const {
                auto input_memory =
                  load_np_array_as_memory(input_filename, engine_);
                auto output_dims = extract_dims(input_memory);
                auto output = array(dtype_t::float_, output_dims);
                auto node = menoh_impl::node{"", {"input"}, {"output"}, {}};
                auto factory_return = make_relu_primitive(
                  node, 0, {node}, {}, {{"input", input_memory}},
                  {{"output", output}}, engine_);
                auto& net = std::get<0>(factory_return);
                mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();

                auto true_output =
                  menoh_impl::load_np_array_as_array(true_output_filename);

                assert_near_list(fbegin(output), fend(output),
                                 fbegin(true_output), fend(true_output),
                                 10.e-4);
            }

            auto
            leaky_relu_test(std::string const& input_filename,
                            std::string const& true_output_filename) const {
                auto input_memory =
                  load_np_array_as_memory(input_filename, engine_);
                auto output_dims = extract_dims(input_memory);
                auto output = array(dtype_t::float_, output_dims);
                auto node = menoh_impl::node{
                  "", {"input"}, {"output"}, {{"alpha", 0.001f}}};
                auto factory_return = make_leaky_relu_primitive(
                  node, 0, {node}, {}, {{"input", input_memory}},
                  {{"output", output}}, engine_);
                auto& net = std::get<0>(factory_return);
                mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();

                auto true_output =
                  menoh_impl::load_np_array_as_array(true_output_filename);

                assert_near_list(fbegin(output), fend(output),
                                 fbegin(true_output), fend(true_output),
                                 10.e-4);
            }

            auto elu_test(std::string const& input_filename,
                          std::string const& true_output_filename) const {
                auto input_memory =
                  load_np_array_as_memory(input_filename, engine_);
                auto output_dims = extract_dims(input_memory);
                auto output = array(dtype_t::float_, output_dims);
                auto node = menoh_impl::node{
                  "", {"input"}, {"output"}, {{"alpha", 1.1f}}};
                auto factory_return = make_elu_primitive(
                  node, 0, {node}, {}, {{"input", input_memory}},
                  {{"output", output}}, engine_);
                auto& net = std::get<0>(factory_return);
                mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();

                auto true_output =
                  menoh_impl::load_np_array_as_array(true_output_filename);

                assert_near_list(fbegin(output), fend(output),
                                 fbegin(true_output), fend(true_output),
                                 10.e-4);
            }

            auto abs_test(std::string const& input_filename,
                          std::string const& true_output_filename) const {
                auto input_memory =
                  load_np_array_as_memory(input_filename, engine_);
                auto output_dims = extract_dims(input_memory);
                auto output = array(dtype_t::float_, output_dims);
                auto node = menoh_impl::node{"", {"input"}, {"output"}, {}};
                auto factory_return = make_abs_primitive(
                  node, 0, {node}, {}, {{"input", input_memory}},
                  {{"output", output}}, engine_);
                auto& net = std::get<0>(factory_return);
                mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();

                auto true_output =
                  menoh_impl::load_np_array_as_array(true_output_filename);

                assert_near_list(fbegin(output), fend(output),
                                 fbegin(true_output), fend(true_output),
                                 10.e-4);
            }

            auto sqrt_test(std::string const& input_filename,
                           std::string const& true_output_filename) const {
                auto input_memory =
                  load_np_array_as_memory(input_filename, engine_);
                auto output_dims = extract_dims(input_memory);
                auto output = array(dtype_t::float_, output_dims);
                auto node = menoh_impl::node{"", {"input"}, {"output"}, {}};
                auto factory_return = make_sqrt_primitive(
                  node, 0, {node}, {}, {{"input", input_memory}},
                  {{"output", output}}, engine_);
                auto& net = std::get<0>(factory_return);
                mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();

                auto true_output =
                  menoh_impl::load_np_array_as_array(true_output_filename);

                assert_near_list(fbegin(output), fend(output),
                                 fbegin(true_output), fend(true_output),
                                 10.e-4);
            }

            auto tanh_test(std::string const& input_filename,
                           std::string const& true_output_filename) const {
                auto input_memory =
                  load_np_array_as_memory(input_filename, engine_);
                auto output_dims = extract_dims(input_memory);
                auto output = array(dtype_t::float_, output_dims);
                auto node = menoh_impl::node{"", {"input"}, {"output"}, {}};
                auto factory_return = make_tanh_primitive(
                  node, 0, {node}, {}, {{"input", input_memory}},
                  {{"output", output}}, engine_);
                auto& net = std::get<0>(factory_return);
                mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();

                auto true_output =
                  menoh_impl::load_np_array_as_array(true_output_filename);

                assert_near_list(fbegin(output), fend(output),
                                 fbegin(true_output), fend(true_output),
                                 10.e-4);
            }

            auto softmax_test(std::string const& input_filename,
                              std::string const& true_output_filename) const {
                auto input_memory =
                  load_np_array_as_memory(input_filename, engine_);
                auto output_dims = extract_dims(input_memory);
                auto output = array(dtype_t::float_, output_dims);
                auto factory_return = make_softmax_primitive(
                  menoh_impl::node{"", {"input"}, {"output"}, {}}, {},
                  {{"input", input_memory}}, {{"output", output}}, engine_);
                auto& net = std::get<0>(factory_return);
                mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();

                auto true_output =
                  menoh_impl::load_np_array_as_array(true_output_filename);

                assert_near_list(fbegin(output), fend(output),
                                 fbegin(true_output), fend(true_output),
                                 10.e-4);
            }

            auto fc_test(std::string const& input_filename,
                         std::string const& weight_filename,
                         std::string const& bias_filename,
                         std::string const& true_output_filename) const {
                auto input_memory =
                  load_np_array_as_memory(input_filename, engine_);
                auto weight =
                  menoh_impl::load_np_array_as_array(weight_filename);
                auto bias = menoh_impl::load_np_array_as_array(bias_filename);
                std::vector<int> output_dims{extract_dims(input_memory).at(0),
                                             weight.dims().at(0)};
                auto output = array(dtype_t::float_, output_dims);
                auto factory_return = make_fc_primitive(
                  menoh_impl::node{"",
                                   {"input", "weight", "bias"},
                                   {"output"},
                                   {{"axis", 1}, {"axis_w", 1}}},
                  {{"weight", weight}, {"bias", bias}},
                  {{"input", input_memory}}, {{"output", output}}, engine_);
                auto& net = std::get<0>(factory_return);
                mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();

                auto true_output =
                  menoh_impl::load_np_array_as_array(true_output_filename);

                assert_near_list(fbegin(output), fend(output),
                                 fbegin(true_output), fend(true_output),
                                 10.e-4);
            }

            auto max_pool_test(std::string const& input_filename, int k, int s,
                               int p) const {
                auto input_memory =
                  load_np_array_as_memory(input_filename, engine_);
                std::vector<int> strides{{s, s}};
                std::vector<int> kernel_shape{{k, k}};
                std::vector<int> pads{{p, p, p, p}};
                auto input_dims = extract_dims(input_memory);
                auto output_dims = calc_2d_output_dims(
                  input_dims, input_dims.at(1), kernel_shape, strides, pads);
                auto output = array(dtype_t::float_, output_dims);
                auto factory_return = make_max_pool_primitive(
                  menoh_impl::node{"",
                                   {"input"},
                                   {"output"},
                                   {{"strides", strides},
                                    {"kernel_shape", kernel_shape},
                                    {"pads", pads}}},
                  {}, {{"input", input_memory}}, {{"output", output}}, engine_);
                auto& net = std::get<0>(factory_return);
                mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();

                auto true_output = menoh_impl::load_np_array_as_array(
                  "../data/max_pooling_2d_k" + std::to_string(k) + "_s" +
                  std::to_string(s) + "_p" + std::to_string(p) + ".txt");

                assert_near_list(fbegin(output), fend(output),
                                 fbegin(true_output), fend(true_output),
                                 10.e-4);
            }

            auto global_max_pool_test(
              std::string const& input_filename,
              std::string const& true_output_filename) const {
                auto input_memory =
                  load_np_array_as_memory(input_filename, engine_);
                auto input_dims = extract_dims(input_memory);
                std::vector<int> strides{1, 1};
                std::vector<int> kernel_shape{input_dims.at(2),
                                              input_dims.at(3)};
                std::vector<int> pads{0, 0, 0, 0};
                auto output_dims = calc_2d_output_dims(
                  input_dims, input_dims.at(1), kernel_shape, strides, pads);
                auto output = array(dtype_t::float_, output_dims);
                auto factory_return = make_global_max_pool_primitive(
                  menoh_impl::node{"", {"input"}, {"output"}, {}}, {},
                  {{"input", input_memory}}, {{"output", output}}, engine_);
                auto& net = std::get<0>(factory_return);
                mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();

                auto true_output =
                  menoh_impl::load_np_array_as_array(true_output_filename);

                assert_near_list(fbegin(output), fend(output),
                                 fbegin(true_output), fend(true_output),
                                 10.e-4);
            }

            auto average_pool_test(std::string const& input_filename, int k,
                                   int s, int p) const {
                auto input_memory =
                  load_np_array_as_memory(input_filename, engine_);
                std::vector<int> strides{{s, s}};
                std::vector<int> kernel_shape{{k, k}};
                std::vector<int> pads{{p, p, p, p}};
                auto input_dims = extract_dims(input_memory);
                auto output_dims = calc_2d_output_dims(
                  input_dims, input_dims.at(1), kernel_shape, strides, pads);
                auto output = array(dtype_t::float_, output_dims);
                auto factory_return = make_average_pool_primitive(
                  menoh_impl::node{"",
                                   {"input"},
                                   {"output"},
                                   {{"strides", strides},
                                    {"kernel_shape", kernel_shape},
                                    {"pads", pads}}},
                  {}, {{"input", input_memory}}, {{"output", output}}, engine_);
                auto& net = std::get<0>(factory_return);
                mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();

                auto true_output = menoh_impl::load_np_array_as_array(
                  "../data/average_pooling_2d_k" + std::to_string(k) + "_s" +
                  std::to_string(s) + "_p" + std::to_string(p) + ".txt");

                assert_near_list(fbegin(output), fend(output),
                                 fbegin(true_output), fend(true_output),
                                 10.e-4);
            }

            auto global_average_pool_test(
              std::string const& input_filename,
              std::string const& true_output_filename) const {
                auto input_memory =
                  load_np_array_as_memory(input_filename, engine_);
                auto input_dims = extract_dims(input_memory);
                std::vector<int> strides{1, 1};
                std::vector<int> kernel_shape{input_dims.at(2),
                                              input_dims.at(3)};
                std::vector<int> pads{0, 0, 0, 0};
                auto output_dims = calc_2d_output_dims(
                  input_dims, input_dims.at(1), kernel_shape, strides, pads);
                auto output = array(dtype_t::float_, output_dims);
                auto factory_return = make_global_average_pool_primitive(
                  menoh_impl::node{"", {"input"}, {"output"}, {}}, {},
                  {{"input", input_memory}}, {{"output", output}}, engine_);
                auto& net = std::get<0>(factory_return);
                mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();

                auto true_output =
                  menoh_impl::load_np_array_as_array(true_output_filename);

                assert_near_list(fbegin(output), fend(output),
                                 fbegin(true_output), fend(true_output),
                                 10.e-4);
            }

            auto conv_test(std::string const& input_filename, int k, int s,
                           int p) const {
                auto input_memory =
                  load_np_array_as_memory(input_filename, engine_);
                auto weight = menoh_impl::load_np_array_as_array(
                  "../data/random_weight_5_4_" + std::to_string(k) + "_" +
                  std::to_string(k) + ".txt");
                std::vector<int> strides{{s, s}};
                std::vector<int> kernel_shape{{k, k}};
                std::vector<int> pads{{p, p, p, p}};
                auto input_dims = extract_dims(input_memory);
                auto output_dims = calc_2d_output_dims(
                  input_dims, weight.dims().at(0), kernel_shape, strides, pads);
                auto output = array(dtype_t::float_, output_dims);
                auto factory_return = make_conv_primitive(
                  menoh_impl::node{"",
                                   {"input", "weight"},
                                   {"output"},
                                   {{"strides", strides},
                                    {"kernel_shape", kernel_shape},
                                    {"pads", pads}}},
                  {{"weight", weight}}, {{"input", input_memory}},
                  {{"output", output}}, engine_);
                auto& net = std::get<0>(factory_return);
                mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();

                auto true_output = menoh_impl::load_np_array_as_array(
                  "../data/convolution_2d_w5_4_" + std::to_string(k) + "_" +
                  std::to_string(k) + "_k" + std::to_string(k) + "_s" +
                  std::to_string(s) + "_p" + std::to_string(p) + ".txt");

                assert_near_list(fbegin(output), fend(output),
                                 fbegin(true_output), fend(true_output),
                                 10.e-4);
            }

            auto conv_with_bias_test(std::string const& input_filename, int k,
                                     int s, int p) const {
                auto input_memory =
                  load_np_array_as_memory(input_filename, engine_);
                auto weight = menoh_impl::load_np_array_as_array(
                  "../data/random_weight_5_4_" + std::to_string(k) + "_" +
                  std::to_string(k) + ".txt");
                auto bias = menoh_impl::load_np_array_as_array(
                  "../data/random_bias_5.txt");
                std::vector<int> strides{{s, s}};
                std::vector<int> kernel_shape{{k, k}};
                std::vector<int> pads{{p, p, p, p}};
                auto input_dims = extract_dims(input_memory);
                auto output_dims = calc_2d_output_dims(
                  input_dims, weight.dims().at(0), kernel_shape, strides, pads);
                auto output = array(dtype_t::float_, output_dims);
                auto factory_return = make_conv_primitive(
                  menoh_impl::node{"",
                                   {"input", "weight", "bias"},
                                   {"output"},
                                   {{"strides", strides},
                                    {"kernel_shape", kernel_shape},
                                    {"pads", pads}}},
                  {{"weight", weight}, {"bias", bias}},
                  {{"input", input_memory}}, {{"output", output}}, engine_);
                auto& net = std::get<0>(factory_return);
                mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();

                auto true_output = menoh_impl::load_np_array_as_array(
                  "../data/convolution_2d_w5_4_" + std::to_string(k) + "_" +
                  std::to_string(k) + "_k" + std::to_string(k) + "_s" +
                  std::to_string(s) + "_p" + std::to_string(p) +
                  "_with_bias.txt");

                assert_near_list(fbegin(output), fend(output),
                                 fbegin(true_output), fend(true_output),
                                 10.e-4);
            }

            auto conv_transpose_test(std::string const& input_filename, int k,
                                     int s, int p) const {
                auto input_memory =
                  load_np_array_as_memory(input_filename, engine_);
                auto weight = menoh_impl::load_np_array_as_array(
                  "../data/random_weight_4_5_" + std::to_string(k) + "_" +
                  std::to_string(k) + ".txt");
                std::vector<int> strides{{s, s}};
                std::vector<int> kernel_shape{{k, k}};
                std::vector<int> pads{{p, p, p, p}};
                auto input_dims = extract_dims(input_memory);
                auto output_dims = calc_2d_output_dims_for_conv_transpose(
                  input_dims, weight.dims().at(1), kernel_shape, strides, pads);
                auto output = array(dtype_t::float_, output_dims);
                auto factory_return = make_conv_transpose_primitive(
                  menoh_impl::node{"",
                                   {"input", "weight"},
                                   {"output"},
                                   {{"strides", strides},
                                    {"kernel_shape", kernel_shape},
                                    {"pads", pads}}},
                  {{"weight", weight}}, {{"input", input_memory}},
                  {{"output", output}}, engine_);
                auto& net = std::get<0>(factory_return);
                mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();

                auto true_output = menoh_impl::load_np_array_as_array(
                  "../data/deconvolution_2d_w4_5_" + std::to_string(k) + "_" +
                  std::to_string(k) + "_k" + std::to_string(k) + "_s" +
                  std::to_string(s) + "_p" + std::to_string(p) + ".txt");

                assert_near_list(fbegin(output), fend(output),
                                 fbegin(true_output), fend(true_output),
                                 10.e-4);
            }

            auto
            conv_transpose_with_bias_test(std::string const& input_filename,
                                          int k, int s, int p) const {
                auto input_memory =
                  load_np_array_as_memory(input_filename, engine_);
                auto weight = menoh_impl::load_np_array_as_array(
                  "../data/random_weight_4_5_" + std::to_string(k) + "_" +
                  std::to_string(k) + ".txt");
                auto bias = menoh_impl::load_np_array_as_array(
                  "../data/random_bias_4.txt");
                std::vector<int> strides{{s, s}};
                std::vector<int> kernel_shape{{k, k}};
                std::vector<int> pads{{p, p, p, p}};
                auto input_dims = extract_dims(input_memory);
                auto output_dims = calc_2d_output_dims_for_conv_transpose(
                  input_dims, weight.dims().at(1), kernel_shape, strides, pads);
                auto output = array(dtype_t::float_, output_dims);
                auto factory_return = make_conv_transpose_primitive(
                  menoh_impl::node{"",
                                   {"input", "weight", "bias"},
                                   {"output"},
                                   {{"strides", strides},
                                    {"kernel_shape", kernel_shape},
                                    {"pads", pads}}},
                  {{"weight", weight}, {"bias", bias}},
                  {{"input", input_memory}}, {{"output", output}}, engine_);
                auto& net = std::get<0>(factory_return);
                mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();

                auto true_output = menoh_impl::load_np_array_as_array(
                  "../data/deconvolution_2d_w4_5_" + std::to_string(k) + "_" +
                  std::to_string(k) + "_k" + std::to_string(k) + "_s" +
                  std::to_string(s) + "_p" + std::to_string(p) +
                  "_with_bias.txt");

                assert_near_list(fbegin(output), fend(output),
                                 fbegin(true_output), fend(true_output),
                                 10.e-4);
            }

            auto
            batch_norm_test(std::string const& input_filename,
                            std::string const& mean_filename,
                            std::string const& var_fileanme,
                            std::string const& gamma_filename,
                            std::string const& beta_filename,
                            std::string const& true_output_filename) const {
                auto input_memory =
                  load_np_array_as_memory(input_filename, engine_);
                auto mean = menoh_impl::load_np_array_as_array(mean_filename);
                auto var = menoh_impl::load_np_array_as_array(var_fileanme);
                auto gamma = menoh_impl::load_np_array_as_array(gamma_filename);
                auto beta = menoh_impl::load_np_array_as_array(beta_filename);
                auto output_dims = extract_dims(input_memory);
                auto output = array(dtype_t::float_, output_dims);
                auto factory_return = make_batch_norm_primitive(
                  menoh_impl::node{"",
                                   {"input", "gamma", "beta", "mean", "var"},
                                   {"output"},
                                   {{"epsilon", 1e-5f}, {"is_test", 1}}},
                  {{"gamma", gamma},
                   {"beta", beta},
                   {"mean", mean},
                   {"var", var}},
                  {{"input", input_memory}}, {{"output", output}}, engine_);
                auto& net = std::get<0>(factory_return);
                mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();

                auto true_output =
                  menoh_impl::load_np_array_as_array(true_output_filename);

                assert_near_list(fbegin(output), fend(output),
                                 fbegin(true_output), fend(true_output),
                                 10.e-4);
            }

            auto add_test(std::string const& input_a_filename,
                          std::string const& input_b_filename,
                          std::string const& true_output_filename) const {
                auto input_a_memory =
                  load_np_array_as_memory(input_a_filename, engine_);
                auto input_b_memory =
                  load_np_array_as_memory(input_b_filename, engine_);
                auto output_dims = extract_dims(input_a_memory);
                auto output = array(dtype_t::float_, output_dims);
                auto factory_return = make_add_primitive(
                  menoh_impl::node{
                    "", {"input_a", "input_b"}, {"output"}, {{"broadcast", 0}}},
                  {},
                  {{"input_a", input_a_memory}, {"input_b", input_b_memory}},
                  {{"output", output}}, engine_);
                auto& net = std::get<0>(factory_return);
                mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();
                auto true_output =
                  menoh_impl::load_np_array_as_array(true_output_filename);

                assert_near_list(fbegin(output), fend(output),
                                 fbegin(true_output), fend(true_output),
                                 10.e-4);
            }

            auto
            concat_test(std::vector<std::string> const& input_filename_list,
                        int axis,
                        std::string const& true_output_filename) const {
                std::vector<std::string> input_name_list;
                std::unordered_map<std::string, mkldnn::memory>
                  input_memory_table;
                auto i = 0;
                for(auto const& input_filename : input_filename_list) {
                    auto input_name = std::to_string(i);
                    input_name_list.push_back(input_name);
                    input_memory_table.insert(
                      {input_name,
                       load_np_array_as_memory(input_filename, engine_)});
                    ++i;
                }
                auto output_dims =
                  extract_dims(input_memory_table.begin()->second);
                output_dims.at(axis) = 0;
                for(auto const& name_and_memory : input_memory_table) {
                    output_dims.at(axis) +=
                      extract_dims(name_and_memory.second).at(axis);
                }
                auto output = array(dtype_t::float_, output_dims);
                auto factory_return = make_concat_primitive(
                  menoh_impl::node{
                    "", input_name_list, {"output"}, {{"axis", axis}}},
                  {}, input_memory_table, {{"output", output}}, engine_);
                auto& net = std::get<0>(factory_return);
                mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();
                auto true_output =
                  menoh_impl::load_np_array_as_array(true_output_filename);

                assert_near_list(fbegin(output), fend(output),
                                 fbegin(true_output), fend(true_output),
                                 10.e-4);
            }

            auto lrn_test(std::string const& input_filename, float alpha,
                          float beta, float bias, int size,
                          std::string const& true_output_filename) {
                auto input_memory =
                  load_np_array_as_memory(input_filename, engine_);
                auto output_dims = extract_dims(input_memory);
                auto output = array(dtype_t::float_, output_dims);
                auto factory_return = make_lrn_primitive(
                  menoh_impl::node{"",
                                   {"input"},
                                   {"output"},
                                   {{"alpha", alpha},
                                    {"beta", beta},
                                    {"bias", bias},
                                    {"size", size}}},
                  {}, {{"input", input_memory}}, {{"output", output}}, engine_);
                auto& net = std::get<0>(factory_return);
                mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();
                /*
                auto true_output = menoh_impl::load_np_array_as_array(
                  "../data/lrn_alpha" + std::to_string(alpha) + "_beta" +
                  std::to_string(beta) + "_bias" + std::to_string(bias) +
                  "_size" + std::to_string(size) + ".txt");
                */
                auto true_output =
                  menoh_impl::load_np_array_as_array(true_output_filename);
                assert_near_list(fbegin(output), fend(output),
                                 fbegin(true_output), fend(true_output),
                                 10.e-4);
            }

            auto gemm_test(std::string const& input_filename,
                           std::string const& weight_filename,
                           std::string const& bias_filename,
                           std::string const& true_output_filename, float alpha,
                           float beta, int trans_a, int trans_b) const {
                auto input_memory =
                  load_np_array_as_memory(input_filename, engine_);
                auto weight =
                  menoh_impl::load_np_array_as_array(weight_filename);
                auto bias = menoh_impl::load_np_array_as_array(bias_filename);
                std::vector<int> output_dims{extract_dims(input_memory).at(0),
                                             weight.dims().at(0)};
                auto output = array(dtype_t::float_, output_dims);
                auto factory_return = make_gemm_primitive(
                  menoh_impl::node{"",
                                   {"input", "weight", "bias"},
                                   {"output"},
                                   {{"alpha", alpha},
                                    {"beta", beta},
                                    {"transA", trans_a},
                                    {"transB", trans_b}}},
                  {{"weight", weight}, {"bias", bias}},
                  {{"input", input_memory}}, {{"output", output}}, engine_);
                auto& net = std::get<0>(factory_return);
                mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();

                auto true_output =
                  menoh_impl::load_np_array_as_array(true_output_filename);

                assert_near_list(fbegin(output), fend(output),
                                 fbegin(true_output), fend(true_output),
                                 10.e-4);
            }

            mkldnn::engine engine_{mkldnn::engine::cpu, 0};
        };

        TEST_F(OperatorTest, relu_1d_test) {
            relu_test("../data/random_input_3_4096.txt", "../data/relu_1d.txt");
        }
        TEST_F(OperatorTest, relu_2d_test) {
            relu_test("../data/random_input_3_4_32_32.txt",
                      "../data/relu_2d.txt");
        }

        TEST_F(OperatorTest, leaky_relu_1d_test) {
            leaky_relu_test("../data/random_input_3_4096.txt",
                            "../data/leaky_relu_1d.txt");
        }
        TEST_F(OperatorTest, leaky_relu_2d_test) {
            leaky_relu_test("../data/random_input_3_4_32_32.txt",
                            "../data/leaky_relu_2d.txt");
        }

        TEST_F(OperatorTest, elu_1d_test) {
            elu_test("../data/random_input_3_4096.txt", "../data/elu_1d.txt");
        }
        TEST_F(OperatorTest, elu_2d_test) {
            elu_test("../data/random_input_3_4_32_32.txt",
                     "../data/elu_2d.txt");
        }

        TEST_F(OperatorTest, abs_1d_test) {
            abs_test("../data/random_input_3_4096.txt", "../data/abs_1d.txt");
        }
        TEST_F(OperatorTest, abs_2d_test) {
            abs_test("../data/random_input_3_4_32_32.txt",
                     "../data/abs_2d.txt");
        }

        TEST_F(OperatorTest, sqrt_1d_test) {
            sqrt_test("../data/random_positive_input_3_4096.txt",
                      "../data/sqrt_1d.txt");
        }
        TEST_F(OperatorTest, sqrt_2d_test) {
            sqrt_test("../data/random_positive_input_3_4_32_32.txt",
                      "../data/sqrt_2d.txt");
        }

        TEST_F(OperatorTest, tanh_1d_test) {
            tanh_test("../data/random_input_3_4096.txt", "../data/tanh_1d.txt");
        }
        TEST_F(OperatorTest, tanh_2d_test) {
            tanh_test("../data/random_input_3_4_32_32.txt",
                      "../data/tanh_2d.txt");
        }

        TEST_F(OperatorTest, softmax_1d_test) {
            softmax_test("../data/random_input_3_4096.txt",
                         "../data/softmax_1d.txt");
        }
        TEST_F(OperatorTest, softmax_2d_test) {
            softmax_test("../data/random_input_3_4_32_32.txt",
                         "../data/softmax_2d.txt");
        }

        TEST_F(OperatorTest, fc_1d_test) {
            fc_test("../data/random_input_3_4096.txt",
                    "../data/random_weight_256_4096.txt",
                    "../data/random_bias_256.txt",
                    "../data/linear_1d_w256_4096_b_256.txt");
        }
        TEST_F(OperatorTest, fc_2d_test) {
            fc_test("../data/random_input_3_4_32_32.txt",
                    "../data/random_weight_256_4096.txt",
                    "../data/random_bias_256.txt",
                    "../data/linear_2d_w256_4096_b_256.txt");
        }

        TEST_F(OperatorTest, max_pool_2_2_0_test) {
            max_pool_test("../data/random_input_3_4_32_32.txt", 2, 2, 0);
        }
        TEST_F(OperatorTest, max_pool_3_2_0_test) {
            max_pool_test("../data/random_input_3_4_32_32.txt", 3, 2, 0);
        }
        TEST_F(OperatorTest, max_pool_3_2_1_test) {
            max_pool_test("../data/random_input_3_4_32_32.txt", 3, 2, 1);
        }

        TEST_F(OperatorTest, average_pool_2_2_0_test) {
            average_pool_test("../data/random_input_3_4_32_32.txt", 2, 2, 0);
        }
        TEST_F(OperatorTest, average_pool_3_2_0_test) {
            average_pool_test("../data/random_input_3_4_32_32.txt", 3, 2, 0);
        }
        TEST_F(OperatorTest, average_pool_3_2_1_test) {
            average_pool_test("../data/random_input_3_4_32_32.txt", 3, 2, 1);
        }

        TEST_F(OperatorTest, global_max_pool_test) {
            global_max_pool_test("../data/random_input_3_4_32_32.txt",
                                 "../data/global_max_pooling_2d.txt");
        }

        TEST_F(OperatorTest, global_average_pool_test) {
            global_average_pool_test("../data/random_input_3_4_32_32.txt",
                                     "../data/global_average_pooling_2d.txt");
        }

        TEST_F(OperatorTest, conv_1_1_0_test) {
            conv_test("../data/random_input_3_4_32_32.txt", 1, 1, 0);
        }
        TEST_F(OperatorTest, conv_2_1_0_test) {
            conv_test("../data/random_input_3_4_32_32.txt", 2, 1, 0);
        }
        TEST_F(OperatorTest, conv_2_1_1_test) {
            conv_test("../data/random_input_3_4_32_32.txt", 2, 1, 1);
        }
        TEST_F(OperatorTest, conv_2_2_0_test) {
            conv_test("../data/random_input_3_4_32_32.txt", 2, 2, 0);
        }
        TEST_F(OperatorTest, conv_2_2_1_test) {
            conv_test("../data/random_input_3_4_32_32.txt", 2, 2, 1);
        }
        TEST_F(OperatorTest, conv_3_1_1_test) {
            conv_test("../data/random_input_3_4_32_32.txt", 3, 1, 1);
        }
        TEST_F(OperatorTest, conv_3_2_0_test) {
            conv_test("../data/random_input_3_4_32_32.txt", 3, 2, 0);
        }
        TEST_F(OperatorTest, conv_3_2_1_test) {
            conv_test("../data/random_input_3_4_32_32.txt", 3, 2, 1);
        }
        TEST_F(OperatorTest, conv_with_bias_1_1_0_test) {
            conv_with_bias_test("../data/random_input_3_4_32_32.txt", 1, 1, 0);
        }
        TEST_F(OperatorTest, conv_with_bias_2_1_0_test) {
            conv_test("../data/random_input_3_4_32_32.txt", 2, 1, 0);
        }
        TEST_F(OperatorTest, conv_with_bias_2_1_1_test) {
            conv_with_bias_test("../data/random_input_3_4_32_32.txt", 2, 1, 1);
        }
        TEST_F(OperatorTest, conv_with_bias_2_2_0_test) {
            conv_with_bias_test("../data/random_input_3_4_32_32.txt", 2, 2, 0);
        }
        TEST_F(OperatorTest, conv_with_bias_2_2_1_test) {
            conv_with_bias_test("../data/random_input_3_4_32_32.txt", 2, 2, 1);
        }
        TEST_F(OperatorTest, conv_with_bias_3_1_1_test) {
            conv_with_bias_test("../data/random_input_3_4_32_32.txt", 3, 1, 1);
        }
        TEST_F(OperatorTest, conv_with_bias_3_2_0_test) {
            conv_with_bias_test("../data/random_input_3_4_32_32.txt", 3, 2, 0);
        }
        TEST_F(OperatorTest, conv_with_bias_3_2_1_test) {
            conv_with_bias_test("../data/random_input_3_4_32_32.txt", 3, 2, 1);
        }

        TEST_F(OperatorTest, conv_transpose_1_1_0_test) {
            conv_transpose_test("../data/random_input_3_4_32_32.txt", 1, 1, 0);
        }
        TEST_F(OperatorTest, conv_transpose_2_1_0_test) {
            conv_transpose_test("../data/random_input_3_4_32_32.txt", 2, 1, 0);
        }
        TEST_F(OperatorTest, conv_transpose_2_1_1_test) {
            conv_transpose_test("../data/random_input_3_4_32_32.txt", 2, 1, 1);
        }
        TEST_F(OperatorTest, conv_transpose_2_2_0_test) {
            conv_transpose_test("../data/random_input_3_4_32_32.txt", 2, 2, 0);
        }
        TEST_F(OperatorTest, conv_transpose_2_2_1_test) {
            conv_transpose_test("../data/random_input_3_4_32_32.txt", 2, 2, 1);
        }
        TEST_F(OperatorTest, conv_transpose_3_1_1_test) {
            conv_transpose_test("../data/random_input_3_4_32_32.txt", 3, 1, 1);
        }
        TEST_F(OperatorTest, conv_transpose_3_2_0_test) {
            conv_transpose_test("../data/random_input_3_4_32_32.txt", 3, 2, 0);
        }
        TEST_F(OperatorTest, conv_transpose_3_2_1_test) {
            conv_transpose_test("../data/random_input_3_4_32_32.txt", 3, 2, 1);
        }

        TEST_F(OperatorTest, conv_transpose_with_bias_1_1_0_test) {
            conv_transpose_with_bias_test("../data/random_input_3_4_32_32.txt",
                                          1, 1, 0);
        }
        TEST_F(OperatorTest, conv_transpose_with_bias_2_1_0_test) {
            conv_transpose_with_bias_test("../data/random_input_3_4_32_32.txt",
                                          2, 1, 0);
        }
        TEST_F(OperatorTest, conv_transpose_with_bias_2_1_1_test) {
            conv_transpose_with_bias_test("../data/random_input_3_4_32_32.txt",
                                          2, 1, 1);
        }
        TEST_F(OperatorTest, conv_transpose_with_bias_2_2_0_test) {
            conv_transpose_with_bias_test("../data/random_input_3_4_32_32.txt",
                                          2, 2, 0);
        }
        TEST_F(OperatorTest, conv_transpose_with_bias_2_2_1_test) {
            conv_transpose_with_bias_test("../data/random_input_3_4_32_32.txt",
                                          2, 2, 1);
        }
        TEST_F(OperatorTest, conv_transpose_with_bias_3_1_1_test) {
            conv_transpose_with_bias_test("../data/random_input_3_4_32_32.txt",
                                          3, 1, 1);
        }
        TEST_F(OperatorTest, conv_transpose_with_bias_3_2_0_test) {
            conv_transpose_with_bias_test("../data/random_input_3_4_32_32.txt",
                                          3, 2, 0);
        }
        TEST_F(OperatorTest, conv_transpose_with_bias_3_2_1_test) {
            conv_transpose_with_bias_test("../data/random_input_3_4_32_32.txt",
                                          3, 2, 1);
        }

        TEST_F(OperatorTest, batch_norm_test) {
            batch_norm_test(
              "../data/random_input_3_4_32_32.txt", "../data/random_mean_4.txt",
              "../data/random_var_4.txt", "../data/random_gamma_4.txt",
              "../data/random_beta_4.txt", "../data/batch_normalization.txt");
        }

        TEST_F(OperatorTest, add_1d_test) {
            add_test("../data/random_input_3_4096.txt",
                     "../data/random_input_3_4096.txt", "../data/add_1d.txt");
        }
        TEST_F(OperatorTest, add_2d_test) {
            add_test("../data/random_input_3_4_32_32.txt",
                     "../data/random_input_3_4_32_32.txt",
                     "../data/add_2d.txt");
        }

        TEST_F(OperatorTest, concat_1d_2_inputs_axis_0_test) {
            concat_test({"../data/random_input_3_4096.txt",
                         "../data/random_input_3_4096.txt"},
                        0, "../data/concat_1d_6_4096.txt");
        }
        TEST_F(OperatorTest, concat_1d_2_inputs_axis_1_test) {
            concat_test({"../data/random_input_3_4096.txt",
                         "../data/random_input_3_4096.txt"},
                        1, "../data/concat_1d_3_8192.txt");
        }
        TEST_F(OperatorTest, concat_1d_3_inputs_axis_0_test) {
            concat_test({"../data/random_input_3_4096.txt",
                         "../data/random_input_3_4096.txt",
                         "../data/random_input_3_4096.txt"},
                        0, "../data/concat_1d_9_4096.txt");
        }
        TEST_F(OperatorTest, concat_1d_3_inputs_axis_1_test) {
            concat_test({"../data/random_input_3_4096.txt",
                         "../data/random_input_3_4096.txt",
                         "../data/random_input_3_4096.txt"},
                        1, "../data/concat_1d_3_12288.txt");
        }

        TEST_F(OperatorTest, lrn_alpha_00004_beta_075_bias_1_size_1_test) {
            lrn_test("../data/random_input_3_4_32_32.txt", 0.0001, 0.75, 1, 1,
                     "../data/lrn_alpha0.0001_beta0.75_bias1_size1.txt");
        }
        TEST_F(OperatorTest, lrn_alpha_00004_beta_075_bias_1_size_2_test) {
            lrn_test("../data/random_input_3_4_32_32.txt", 0.0001, 0.75, 1, 2,
                     "../data/lrn_alpha0.0001_beta0.75_bias1_size2.txt");
        }
        TEST_F(OperatorTest, lrn_alpha_00004_beta_075_bias_1_size_3_test) {
            lrn_test("../data/random_input_3_4_32_32.txt", 0.0001, 0.75, 1, 3,
                     "../data/lrn_alpha0.0001_beta0.75_bias1_size3.txt");
        }
        TEST_F(OperatorTest, lrn_alpha_00004_beta_075_bias_1_size_4_test) {
            lrn_test("../data/random_input_3_4_32_32.txt", 0.0001, 0.75, 1, 4,
                     "../data/lrn_alpha0.0001_beta0.75_bias1_size4.txt");
        }
        TEST_F(OperatorTest, lrn_alpha_00004_beta_075_bias_2_size_1_test) {
            lrn_test("../data/random_input_3_4_32_32.txt", 0.0001, 0.75, 2, 1,
                     "../data/lrn_alpha0.0001_beta0.75_bias2_size1.txt");
        }
        TEST_F(OperatorTest, lrn_alpha_00004_beta_075_bias_2_size_2_test) {
            lrn_test("../data/random_input_3_4_32_32.txt", 0.0001, 0.75, 2, 2,
                     "../data/lrn_alpha0.0001_beta0.75_bias2_size2.txt");
        }
        TEST_F(OperatorTest, lrn_alpha_00004_beta_075_bias_2_size_3_test) {
            lrn_test("../data/random_input_3_4_32_32.txt", 0.0001, 0.75, 2, 3,
                     "../data/lrn_alpha0.0001_beta0.75_bias2_size3.txt");
        }
        TEST_F(OperatorTest, lrn_alpha_00004_beta_075_bias_2_size_4_test) {
            lrn_test("../data/random_input_3_4_32_32.txt", 0.0001, 0.75, 2, 4,
                     "../data/lrn_alpha0.0001_beta0.75_bias2_size4.txt");
        }

        TEST_F(OperatorTest, gemm_1d_test) {
            gemm_test("../data/random_input_3_4096.txt",
                      "../data/random_weight_256_4096.txt",
                      "../data/random_bias_256.txt",
                      "../data/linear_1d_w256_4096_b_256.txt", 1, 1, 0, 1);
        }
        TEST_F(OperatorTest, gemm_2d_test) {
            gemm_test("../data/random_input_3_4_32_32.txt",
                      "../data/random_weight_256_4096.txt",
                      "../data/random_bias_256.txt",
                      "../data/linear_2d_w256_4096_b_256.txt", 1, 1, 0, 1);
        }
        TEST_F(OperatorTest, gemm_1d_test_invalid_alpha) {
            EXPECT_THROW(
              {
                  gemm_test("../data/random_input_3_4096.txt",
                            "../data/random_weight_256_4096.txt",
                            "../data/random_bias_256.txt",
                            "../data/linear_1d_w256_4096_b_256.txt", 2, 1, 0,
                            1);
              },
              failed_to_configure_operator);
        }
        TEST_F(OperatorTest, gemm_2d_test_invalid_alpha) {
            EXPECT_THROW(
              {
                  gemm_test("../data/random_input_3_4_32_32.txt",
                            "../data/random_weight_256_4096.txt",
                            "../data/random_bias_256.txt",
                            "../data/linear_2d_w256_4096_b_256.txt", 2, 1, 0,
                            1);
              },
              failed_to_configure_operator);
        }
        TEST_F(OperatorTest, gemm_1d_test_invalid_beta) {
            EXPECT_THROW(
              {
                  gemm_test("../data/random_input_3_4096.txt",
                            "../data/random_weight_256_4096.txt",
                            "../data/random_bias_256.txt",
                            "../data/linear_1d_w256_4096_b_256.txt", 1, 2, 0,
                            1);
              },
              failed_to_configure_operator);
        }
        TEST_F(OperatorTest, gemm_2d_test_invalid_beta) {
            EXPECT_THROW(
              {
                  gemm_test("../data/random_input_3_4_32_32.txt",
                            "../data/random_weight_256_4096.txt",
                            "../data/random_bias_256.txt",
                            "../data/linear_2d_w256_4096_b_256.txt", 1, 2, 0,
                            1);
              },
              failed_to_configure_operator);
        }
        TEST_F(OperatorTest, gemm_1d_test_invalid_transA) {
            EXPECT_THROW(
              {
                  gemm_test("../data/random_input_3_4096.txt",
                            "../data/random_weight_256_4096.txt",
                            "../data/random_bias_256.txt",
                            "../data/linear_1d_w256_4096_b_256.txt", 1, 1, 1,
                            1);
              },
              failed_to_configure_operator);
        }
        TEST_F(OperatorTest, gemm_2d_test_invalid_transA) {
            EXPECT_THROW(
              {
                  gemm_test("../data/random_input_3_4_32_32.txt",
                            "../data/random_weight_256_4096.txt",
                            "../data/random_bias_256.txt",
                            "../data/linear_2d_w256_4096_b_256.txt", 1, 1, 1,
                            1);
              },
              failed_to_configure_operator);
        }
        TEST_F(OperatorTest, gemm_1d_test_invalid_transB) {
            EXPECT_THROW(
              {
                  gemm_test("../data/random_input_3_4096.txt",
                            "../data/random_weight_256_4096.txt",
                            "../data/random_bias_256.txt",
                            "../data/linear_1d_w256_4096_b_256.txt", 1, 1, 0,
                            0);
              },
              failed_to_configure_operator);
        }
        TEST_F(OperatorTest, gemm_2d_test_invalid_transB) {
            EXPECT_THROW(
              {
                  gemm_test("../data/random_input_3_4_32_32.txt",
                            "../data/random_weight_256_4096.txt",
                            "../data/random_bias_256.txt",
                            "../data/linear_2d_w256_4096_b_256.txt", 1, 1, 0,
                            0);
              },
              failed_to_configure_operator);
        }

    } // namespace mkldnn_backend
} // namespace menoh_impl
