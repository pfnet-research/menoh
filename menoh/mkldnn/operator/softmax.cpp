#include <menoh/mkldnn/operator/softmax.hpp>

#include <tuple>

#include <mkldnn.hpp>

#include <menoh/utility.hpp>

#include <menoh/mkldnn/operator/common.hpp>
#include <menoh/mkldnn/utility.hpp>

namespace menoh_impl {
    namespace mkldnn_backend {

        primitive_factory_return_type make_softmax_primitive(
          menoh_impl::node const& node,
          std::unordered_map<std::string, array> const&,
          std::unordered_map<std::string, mkldnn::memory> const&
            variable_memory_table,
          std::unordered_map<std::string, array> const& required_output_table,
          mkldnn::engine const& engine) {

            auto softmax_axis = optional_attribute_int(node, "axis", 1);

            std::vector<mkldnn::primitive> net;
            std::unordered_map<std::string, mkldnn::memory> output_memory_table;
            std::unordered_map<std::string, array> output_table;
            std::vector<mkldnn::memory> temp_memory_list;

            auto const& input_memory =
              find_value(variable_memory_table, node.input_name_list.at(0));

            auto input_dims = extract_dims(input_memory);
            auto output_dims = input_dims;

            auto const& output_name = node.output_name_list.at(0);

            auto op_desc = mkldnn::softmax_forward::desc(
              mkldnn::prop_kind::forward_inference,
              input_memory.get_primitive_desc().desc(), softmax_axis);
            auto op_pd =
              mkldnn::softmax_forward::primitive_desc(op_desc, engine);

            auto output_format = input_dims.size() == 2
                                   ? mkldnn::memory::format::nc
                                   : mkldnn::memory::format::nchw;

            manage_output_memory(
              net, output_name, output_format,
              input_memory.get_primitive_desc(), output_memory_table,
              required_output_table, temp_memory_list, engine,
              [&net, &input_memory, &op_pd](mkldnn::memory& op_output_memory) {
                  net.push_back(mkldnn::softmax_forward(op_pd, input_memory,
                                                        op_output_memory));
              });

            return std::make_tuple(net, output_memory_table, temp_memory_list,
                                   std::vector<array>());
        }

    } // namespace mkldnn_backend
} // namespace menoh_impl
