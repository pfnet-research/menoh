#include <menoh/mkldnn/operator/pool.hpp>

#include <tuple>

#include <mkldnn.hpp>

#include <menoh/utility.hpp>

#include <menoh/mkldnn/operator/common.hpp>
#include <menoh/mkldnn/utility.hpp>

#include <menoh/graph.hpp> // for unsupported_operator error

namespace menoh_impl {
    namespace mkldnn_backend {

        template <mkldnn::algorithm pooling_alg>
        auto make_pool_primitive_impl(
          menoh_impl::node const& node,
          std::unordered_map<std::string, mkldnn::memory> const&
            variable_memory_table,
          std::unordered_map<std::string, array> const& required_output_table,
          mkldnn::engine const& engine, std::vector<int> const& strides,
          std::vector<int> const& kernel_shape, std::vector<int> const& pads) {

            std::vector<mkldnn::primitive> net;
            std::unordered_map<std::string, mkldnn::memory> output_memory_table;
            std::unordered_map<std::string, array> output_table;
            std::vector<mkldnn::memory> temp_memory_list;

            std::vector<int> padding_l{pads[0], pads[1]};
            std::vector<int> padding_r{pads[2], pads[3]};

            auto const& input_memory =
              find_value(variable_memory_table, node.input_name_list.at(0));

            auto input_dims = extract_dims(input_memory);
            auto output_channel_num = input_dims[1];
            auto output_dims = calc_2d_output_dims(
              input_dims, output_channel_num, kernel_shape, strides, pads);

            auto const& output_name = node.output_name_list.at(0);

            auto pool_output_md = mkldnn::memory::desc(
              {output_dims}, mkldnn::memory::data_type::f32,
              mkldnn::memory::format::any);
            auto pool_desc = mkldnn::pooling_forward::desc(
              mkldnn::prop_kind::forward, pooling_alg,
              input_memory.get_primitive_desc().desc(), pool_output_md, strides,
              kernel_shape, padding_l, padding_r, mkldnn::padding_kind::zero);
            auto pool_pd =
              mkldnn::pooling_forward::primitive_desc(pool_desc, engine);

            manage_output_memory(
              net, output_name, mkldnn::memory::format::nchw,
              pool_pd.dst_primitive_desc(), output_memory_table,
              required_output_table, temp_memory_list, engine,
              [pa = pooling_alg, &net, &input_memory, &temp_memory_list,
               &pool_pd](auto& op_output_memory) {
                  if(pa == mkldnn::pooling_max) {
                      auto pool_indices_memory =
                        mkldnn::memory(pool_pd.workspace_primitive_desc());
                      temp_memory_list.push_back(pool_indices_memory);
                      net.push_back(mkldnn::pooling_forward(
                        pool_pd, input_memory, op_output_memory,
                        pool_indices_memory));
                  } else {
                      net.push_back(mkldnn::pooling_forward(
                        pool_pd, input_memory, op_output_memory));
                  }
              });

            return std::make_tuple(net, output_memory_table, temp_memory_list,
                                   std::vector<array>());
        }

        template <mkldnn::algorithm pooling_alg>
        auto make_pool_primitive(
          menoh_impl::node const& node,
          std::unordered_map<std::string, mkldnn::memory> const&
            variable_memory_table,
          std::unordered_map<std::string, array> const& required_output_table,
          mkldnn::engine const& engine) {
            std::vector<int> strides, kernel_shape, pads;
            std::tie(strides, kernel_shape, pads) =
              attributes_for_2d_data_processing(node);
            return make_pool_primitive_impl<pooling_alg>(
              node, variable_memory_table, required_output_table, engine,
              strides, kernel_shape, pads);
        }

        primitive_factory_return_type make_max_pool_primitive(
          menoh_impl::node const& node,
          std::unordered_map<std::string, array> const&,
          std::unordered_map<std::string, mkldnn::memory> const&
            variable_memory_table,
          std::unordered_map<std::string, array> const& required_output_table,
          mkldnn::engine const& engine) {
            if(node.output_name_list.size() != 1) {
                throw unsupported_operator("MaxPool issuing multiple outputs");
            }
            return make_pool_primitive<mkldnn::pooling_max>(
              node, variable_memory_table, required_output_table, engine);
        }

        primitive_factory_return_type make_average_pool_primitive(
          menoh_impl::node const& node,
          std::unordered_map<std::string, array> const&,
          std::unordered_map<std::string, mkldnn::memory> const&
            variable_memory_table,
          std::unordered_map<std::string, array> const& required_output_table,
          mkldnn::engine const& engine) {
            return make_pool_primitive<mkldnn::pooling_avg_include_padding>(
              node, variable_memory_table, required_output_table, engine);
        }

        template <mkldnn::algorithm pooling_alg>
        auto make_global_pool_primitive(
          menoh_impl::node const& node,
          std::unordered_map<std::string, mkldnn::memory> const&
            variable_memory_table,
          std::unordered_map<std::string, array> const& required_output_table,
          mkldnn::engine const& engine) {
            auto const& input_memory =
              find_value(variable_memory_table, node.input_name_list.at(0));
            auto input_dims = extract_dims(input_memory);
            std::vector<int> strides{1, 1};
            std::vector<int> kernel_shape{input_dims.at(2), input_dims.at(3)};
            std::vector<int> pads{0, 0, 0, 0};
            return make_pool_primitive_impl<pooling_alg>(
              node, variable_memory_table, required_output_table, engine,
              strides, kernel_shape, pads);
        }

        primitive_factory_return_type make_global_max_pool_primitive(
          menoh_impl::node const& node,
          std::unordered_map<std::string, array> const&,
          std::unordered_map<std::string, mkldnn::memory> const&
            variable_memory_table,
          std::unordered_map<std::string, array> const& required_output_table,
          mkldnn::engine const& engine) {
            return make_global_pool_primitive<mkldnn::pooling_max>(
              node, variable_memory_table, required_output_table, engine);
        }

        primitive_factory_return_type make_global_average_pool_primitive(
          menoh_impl::node const& node,
          std::unordered_map<std::string, array> const&,
          std::unordered_map<std::string, mkldnn::memory> const&
            variable_memory_table,
          std::unordered_map<std::string, array> const& required_output_table,
          mkldnn::engine const& engine) {
            return make_global_pool_primitive<
              mkldnn::pooling_avg_include_padding>(
              node, variable_memory_table, required_output_table, engine);
        }

    } // namespace mkldnn_backend
} // namespace menoh_impl
