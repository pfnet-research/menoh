#include <menoh/mkldnn/operator/lrn.hpp>

#include <tuple>

#include <menoh/optional.hpp>
#include <menoh/utility.hpp>

#include <menoh/mkldnn/operator/common.hpp>
#include <menoh/mkldnn/utility.hpp>

namespace menoh_impl {
    namespace mkldnn_backend {

        primitive_factory_return_type make_lrn_primitive(
          menoh_impl::node const& node,
          std::unordered_map<std::string, array> const& /*parameter_table*/,
          std::unordered_map<std::string, mkldnn::memory> const&
            variable_memory_table,
          std::unordered_map<std::string, array> const& required_output_table,
          mkldnn::engine const& engine) {

            std::vector<mkldnn::primitive> net;
            std::unordered_map<std::string, mkldnn::memory> output_memory_table;
            std::vector<mkldnn::memory> temp_memory_list;
            std::vector<array> owned_array_list;

            float alpha = optional_attribute_float(node, "alpha", 1e-4f);
            float beta = optional_attribute_float(node, "beta", 0.75f);
            float bias = optional_attribute_float(node, "bias", 1.f);
            float size = attribute_int(node, "size");

            auto const& input_memory =
              find_value(variable_memory_table, node.input_name_list.at(0));

            auto input_dims = extract_dims(input_memory);
            auto output_dims = input_dims;

            auto const& output_name = node.output_name_list.at(0);

            auto lrn_input_md = input_memory.get_primitive_desc().desc();

            auto lrn_desc =
              mkldnn::lrn_forward::desc(mkldnn::prop_kind::forward_scoring,
                                        mkldnn::algorithm::lrn_across_channels,
                                        lrn_input_md, size, alpha, beta, bias);
            auto lrn_pd = mkldnn::lrn_forward::primitive_desc(lrn_desc, engine);

            auto lrn_input_memory = input_memory;
            if(mkldnn::memory::primitive_desc(lrn_pd.src_primitive_desc()) !=
               input_memory.get_primitive_desc()) {
                lrn_input_memory = mkldnn::memory(lrn_pd.src_primitive_desc());
                temp_memory_list.push_back(lrn_input_memory);
                net.push_back(mkldnn::reorder(input_memory, lrn_input_memory));
            }

            manage_output_memory(
              net, output_name, mkldnn::memory::format::nchw,
              lrn_pd.dst_primitive_desc(), output_memory_table,
              required_output_table, temp_memory_list, engine,
              [&net, &lrn_input_memory, &lrn_pd](mkldnn::memory& op_output_memory) {
                  net.push_back(mkldnn::lrn_forward(lrn_pd, lrn_input_memory,
                                                    op_output_memory));
              });

            return std::make_tuple(net, output_memory_table, temp_memory_list,
                                   owned_array_list);
        }

    } // namespace mkldnn_backend
} // namespace menoh_impl
