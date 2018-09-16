#ifndef MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_GEMM_HPP
#define MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_GEMM_HPP

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {
        namespace mkldnn_backend {

            inline std::tuple<
              std::vector<mkldnn::primitive>,
              std::vector<std::pair<std::string, mkldnn::memory>>,
              std::vector<mkldnn::memory>>
            make_gemm(int node_index, std::vector<node> const& node_list,
                      std::vector<array> const& input_list,
                      std::unordered_map<std::string, array> const&
                        required_output_table,
                      mkldnn::engine const& engine) {

                using namespace menoh_impl::mkldnn_backend;
                std::vector<mkldnn::primitive> primitives;
                std::unordered_map<std::string, array> output_table;
                std::vector<mkldnn::memory> temp_memory_list;
                std::vector<array> owned_array_list;

                auto node = node_list.at(node_index);

                auto alpha = optional_attribute_float(node, "alpha", 1.f);
                if(alpha != 1) {
                    throw failed_to_configure_operator(
                      node.op_type, node.output_name_list.at(0),
                      "alpha of Gemm must be 1 but given: " +
                        std::to_string(alpha));
                }
                auto beta = optional_attribute_float(node, "beta", 1.f);
                if(beta != 1) {
                    throw failed_to_configure_operator(
                      node.op_type, node.output_name_list.at(0),
                      "beta of Gemm must be 1 but given: " +
                        std::to_string(alpha));
                }

                auto trans_a = optional_attribute_int(node, "transA", 0);
                if(trans_a) {
                    throw failed_to_configure_operator(
                      node.op_type, node.output_name_list.at(0),
                      "transA of Gemm must be 0 but given: " +
                        std::to_string(alpha));
                }
                auto trans_b = optional_attribute_int(node, "transB", 0);
                if(!trans_b) {
                    throw failed_to_configure_operator(
                      node.op_type, node.output_name_list.at(0),
                      "transB of Gemm must be 0 but given: " +
                        std::to_string(alpha));
                }

                auto input_dims = input_list.at(0).dims();
                auto input_format =
                  (input_dims.size() == 2 ? mkldnn::memory::format::nc
                                          : mkldnn::memory::format::nchw);
                auto input_memory =
                  array_to_memory(input_list.at(0), input_format, engine);
                temp_memory_list.push_back(input_memory);

                auto const& weight_arr = input_list.at(1);
                auto weight_dims = weight_arr.dims();
                auto weight_format =
                  (input_dims.size() == 2 ? mkldnn::memory::format::oi
                                          : mkldnn::memory::format::oihw);
                if(weight_format == mkldnn::memory::format::oihw) {
                    weight_dims = std::vector<int>{weight_dims.front()};
                    weight_dims.insert(weight_dims.end(),
                                       input_dims.begin() + 1,
                                       input_dims.end());
                }
                auto weight_memory = array_to_memory(weight_arr, weight_dims,
                                                     weight_format, engine);
                temp_memory_list.push_back(weight_memory);

                auto bias_arr = input_list.at(2);
                auto bias_memory =
                  array_to_memory(bias_arr, mkldnn::memory::format::x, engine);
                temp_memory_list.push_back(bias_memory);

                auto bias_dims = input_list.at(2).dims();
                int output_size = weight_arr.dims()[0];
                if(output_size != bias_dims[0]) {
                    throw failed_to_configure_operator(
                      node.op_type, node.output_name_list.at(0),
                      "dims[0] of input C must be equal to dims[0] of "
                      "input B: "
                      "broadcast is not supported yet");
                }
                mkldnn::memory::dims output_dims{input_dims[0], output_size};

                auto const& output_name = node.output_name_list.at(0);

                auto gemm_input_md = mkldnn::memory::desc(
                  {input_dims}, mkldnn::memory::data_type::f32,
                  mkldnn::memory::format::any);
                auto gemm_weight_md = mkldnn::memory::desc(
                  {weight_dims}, mkldnn::memory::data_type::f32,
                  mkldnn::memory::format::any);
                auto gemm_output_md = mkldnn::memory::desc(
                  {output_dims}, mkldnn::memory::data_type::f32,
                  mkldnn::memory::format::any);

                mkldnn::inner_product_forward::desc gemm_desc(
                  mkldnn::prop_kind::forward_inference, gemm_input_md,
                  gemm_weight_md, bias_memory.get_primitive_desc().desc(),
                  gemm_output_md);
                auto gemm_pd = mkldnn::inner_product_forward::primitive_desc(
                  gemm_desc, engine);

                auto gemm_input_memory = input_memory;
                if(mkldnn::memory::primitive_desc(
                     gemm_pd.src_primitive_desc()) !=
                   input_memory.get_primitive_desc()) {
                    gemm_input_memory =
                      mkldnn::memory(gemm_pd.src_primitive_desc());
                    temp_memory_list.push_back(gemm_input_memory);
                    primitives.push_back(
                      mkldnn::reorder(input_memory, gemm_input_memory));
                }

                auto gemm_weight_memory = weight_memory;
                if(mkldnn::memory::primitive_desc(
                     gemm_pd.weights_primitive_desc()) !=
                   weight_memory.get_primitive_desc()) {
                    gemm_weight_memory =
                      mkldnn::memory(gemm_pd.weights_primitive_desc());
                    temp_memory_list.push_back(gemm_weight_memory);
                    primitives.push_back(
                      mkldnn::reorder(weight_memory, gemm_weight_memory));
                }

                auto output_format = mkldnn::memory::format::nc;
                std::vector<std::pair<std::string, array>>
                  output_name_and_arr_list;
                menoh_impl::optional<mkldnn::memory> output_memory_opt;
                auto found = required_output_table.find(output_name);
                if(found != required_output_table.end()) {
                    std::string name;
                    array output_array;
                    std::tie(name, output_array) = *found;
                    output_memory_opt =
                      array_to_memory(output_array, output_format, engine);
                }

                auto output_pd = gemm_pd.dst_primitive_desc();
                auto op_output_memory =
                  output_memory_opt.value_or(mkldnn::memory(output_pd));
                if(output_memory_opt &&
                   mkldnn::memory::primitive_desc(output_pd) !=
                     output_memory_opt->get_primitive_desc()) {
                    op_output_memory = mkldnn::memory(output_pd);
                    temp_memory_list.push_back(*output_memory_opt);
                }

                primitives.push_back(mkldnn::inner_product_forward(
                  gemm_pd, gemm_input_memory, gemm_weight_memory, bias_memory,
                  op_output_memory));

                if(output_memory_opt &&
                   op_output_memory != *output_memory_opt) {
                    primitives.push_back(
                      mkldnn::reorder(op_output_memory, *output_memory_opt));
                }

                std::vector<std::pair<std::string, mkldnn::memory>>
                  output_memory_list;
                output_memory_list.emplace_back(output_name, op_output_memory);

                return std::make_tuple(primitives, output_memory_list,
                                       temp_memory_list);
            }

        } // namespace mkldnn_backend
    } // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_GEMM_HPP
