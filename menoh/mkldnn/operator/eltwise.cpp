#include <menoh/mkldnn/operator/eltwise.hpp>

#include <tuple>
#include <vector>

#include <mkldnn.hpp>

#include <menoh/utility.hpp>

#include <menoh/mkldnn/operator/common.hpp>
#include <menoh/mkldnn/utility.hpp>

namespace menoh_impl {
    namespace mkldnn_backend {

        template <mkldnn::algorithm eltwise_alg>
        auto make_eltwise_primitive(
          float alpha, float beta, menoh_impl::node const& node, int node_index,
          std::vector<menoh_impl::node> const& node_list,
          std::unordered_map<std::string, array> const& /*parameter_table*/,
          std::unordered_map<std::string, mkldnn::memory> const&
            variable_memory_table,
          std::unordered_map<std::string, array> const& required_output_table,
          mkldnn::engine const& engine) {

            std::vector<mkldnn::primitive> net;
            std::unordered_map<std::string, mkldnn::memory> output_memory_table;
            std::vector<mkldnn::memory> temp_memory_list;

            auto const& input_memory =
              find_value(variable_memory_table, node.input_name_list.at(0));
            auto input_dims = extract_dims(input_memory);

            auto const& output_name = node.output_name_list.at(0);
            auto output_dims = input_dims;

            auto op_desc = mkldnn::eltwise_forward::desc(
              mkldnn::prop_kind::forward_inference, eltwise_alg,
              input_memory.get_primitive_desc().desc(), alpha, beta);
            auto op_pd =
              mkldnn::eltwise_forward::primitive_desc(op_desc, engine);

            auto output_format = input_dims.size() == 2
                                   ? mkldnn::memory::format::nc
                                   : mkldnn::memory::format::nchw;

            manage_output_memory_inplace_if_possible(
              net, node.input_name_list.at(0), input_memory, output_name,
              output_format, op_pd.dst_primitive_desc(), output_memory_table,
              required_output_table, temp_memory_list, node_index, node_list,
              engine,
              [&net, &input_memory, &node, &op_pd](auto& op_output_memory) {
                  net.push_back(mkldnn::eltwise_forward(op_pd, input_memory,
                                                        op_output_memory));
              });

            return std::make_tuple(net, output_memory_table, temp_memory_list,
                                   std::vector<array>());
        }

        primitive_factory_return_type make_relu_primitive(
          menoh_impl::node const& node, int node_index,
          std::vector<menoh_impl::node> const& node_list,
          std::unordered_map<std::string, array> const& parameter_table,
          std::unordered_map<std::string, mkldnn::memory> const&
            variable_memory_table,
          std::unordered_map<std::string, array> const& required_output_table,
          mkldnn::engine const& engine) {
            float alpha = 0.;
            float beta = 0.;
            return make_eltwise_primitive<mkldnn::algorithm::eltwise_relu>(
              alpha, beta, node, node_index, node_list, parameter_table,
              variable_memory_table, required_output_table, engine);
        }

        primitive_factory_return_type make_leaky_relu_primitive(
          menoh_impl::node const& node, int node_index,
          std::vector<menoh_impl::node> const& node_list,
          std::unordered_map<std::string, array> const& parameter_table,
          std::unordered_map<std::string, mkldnn::memory> const&
            variable_memory_table,
          std::unordered_map<std::string, array> const& required_output_table,
          mkldnn::engine const& engine) {
            float alpha = optional_attribute_float(node, "alpha", 0.01f);
            float beta = 0.;
            return make_eltwise_primitive<mkldnn::algorithm::eltwise_relu>(
              alpha, beta, node, node_index, node_list, parameter_table,
              variable_memory_table, required_output_table, engine);
        }

        primitive_factory_return_type make_elu_primitive(
          menoh_impl::node const& node, int node_index,
          std::vector<menoh_impl::node> const& node_list,
          std::unordered_map<std::string, array> const& parameter_table,
          std::unordered_map<std::string, mkldnn::memory> const&
            variable_memory_table,
          std::unordered_map<std::string, array> const& required_output_table,
          mkldnn::engine const& engine) {
            float alpha = optional_attribute_float(node, "alpha", 1.0f);
            float beta = 0.;
            return make_eltwise_primitive<mkldnn::algorithm::eltwise_elu>(
              alpha, beta, node, node_index, node_list, parameter_table,
              variable_memory_table, required_output_table, engine);
        }

        primitive_factory_return_type make_abs_primitive(
          menoh_impl::node const& node, int node_index,
          std::vector<menoh_impl::node> const& node_list,
          std::unordered_map<std::string, array> const& parameter_table,
          std::unordered_map<std::string, mkldnn::memory> const&
            variable_memory_table,
          std::unordered_map<std::string, array> const& required_output_table,
          mkldnn::engine const& engine) {
            float alpha = 0.;
            float beta = 0.;
            return make_eltwise_primitive<mkldnn::algorithm::eltwise_abs>(
              alpha, beta, node, node_index, node_list, parameter_table,
              variable_memory_table, required_output_table, engine);
        }

        primitive_factory_return_type make_sqrt_primitive(
          menoh_impl::node const& node, int node_index,
          std::vector<menoh_impl::node> const& node_list,
          std::unordered_map<std::string, array> const& parameter_table,
          std::unordered_map<std::string, mkldnn::memory> const&
            variable_memory_table,
          std::unordered_map<std::string, array> const& required_output_table,
          mkldnn::engine const& engine) {
            float alpha = 0.;
            float beta = 0.;
            return make_eltwise_primitive<mkldnn::algorithm::eltwise_sqrt>(
              alpha, beta, node, node_index, node_list, parameter_table,
              variable_memory_table, required_output_table, engine);
        }

        primitive_factory_return_type make_tanh_primitive(
          menoh_impl::node const& node, int node_index,
          std::vector<menoh_impl::node> const& node_list,
          std::unordered_map<std::string, array> const& parameter_table,
          std::unordered_map<std::string, mkldnn::memory> const&
            variable_memory_table,
          std::unordered_map<std::string, array> const& required_output_table,
          mkldnn::engine const& engine) {
            float alpha = 0.;
            float beta = 0.;
            return make_eltwise_primitive<mkldnn::algorithm::eltwise_tanh>(
              alpha, beta, node, node_index, node_list, parameter_table,
              variable_memory_table, required_output_table, engine);
        }

    } // namespace mkldnn_backend
} // namespace menoh_impl
