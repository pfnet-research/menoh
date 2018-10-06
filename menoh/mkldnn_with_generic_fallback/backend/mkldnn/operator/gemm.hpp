#ifndef MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_GEMM_HPP
#define MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_GEMM_HPP

#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/memory_cache.hpp>

#include <mkldnn.hpp>

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {
        namespace mkldnn_backend {

            inline std::tuple<std::vector<mkldnn::primitive>,
                              std::vector<std::pair<std::string, memory_cache>>>
            make_gemm(node const& node,
                      std::vector<std::reference_wrapper<memory_cache>> const&
                        input_memory_cache_list,
                      std::vector<mkldnn::memory> const& output_memory_list,
                      mkldnn::engine const& engine) {

                std::vector<mkldnn::primitive> primitives;
                std::vector<mkldnn::memory> temp_memory_list;
                std::vector<array> owned_array_list;

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

                memory_cache& input_memory_cache =
                  input_memory_cache_list.at(0);
                auto input_dims = input_memory_cache.dims();

                memory_cache& weight_memory_cache =
                  input_memory_cache_list.at(1);
                auto weight_dims = weight_memory_cache.dims(); // mutable
                assert(weight_dims.size() == 2);
                if(input_dims.size() != 2) {
                    weight_dims = std::vector<int>{weight_dims.front()};
                    weight_dims.insert(weight_dims.end(),
                                       input_dims.begin() + 1,
                                       input_dims.end());
                }

                memory_cache& bias_memory_cache = input_memory_cache_list.at(2);
                auto bias_dims = bias_memory_cache.dims();
                int output_size = weight_dims.at(0);
                if(output_size != bias_dims.at(0)) {
                    throw failed_to_configure_operator(
                      node.op_type, node.output_name_list.at(0),
                      "dims[0] of input C must be equal to dims[0] of "
                      "input B: "
                      "broadcast is not supported yet");
                }

                auto const& output_memory = output_memory_list.at(0);
                auto output_dims = extract_dims(output_memory);
                assert(output_dims.at(0) == input_dims.at(0) &&
                       "invalid shape inference");
                assert(output_dims.at(1) == output_size &&
                       "invalid shape inference");
                auto gemm_input_md =
                  mkldnn::memory::desc({input_dims}, input_memory_cache.dtype(),
                                       mkldnn::memory::format::any);
                auto gemm_weight_md = mkldnn::memory::desc(
                  {weight_dims}, weight_memory_cache.dtype(),
                  mkldnn::memory::format::any);
                auto gemm_output_md = mkldnn::memory::desc(
                  {output_dims}, extract_data_type(output_memory),
                  mkldnn::memory::format::any);

                auto bias_memory = get_memory(
                  bias_memory_cache, mkldnn::memory::format::x, primitives);

                mkldnn::inner_product_forward::desc gemm_desc(
                  mkldnn::prop_kind::forward_inference, gemm_input_md,
                  gemm_weight_md, bias_memory.get_primitive_desc().desc(),
                  gemm_output_md);
                auto gemm_pd = mkldnn::inner_product_forward::primitive_desc(
                  gemm_desc, engine);

                auto input_memory = get_memory(
                  input_memory_cache,
                  extract_format(gemm_pd.src_primitive_desc()), primitives);
                auto weight_memory = get_memory(
                  weight_memory_cache, weight_dims,
                  extract_format(gemm_pd.weights_primitive_desc()), primitives);

                memory_cache output_memory_cache;
                optional<mkldnn::memory> op_output_memory;
                if(extract_format(output_memory) ==
                     mkldnn::memory::format::any ||
                   extract_format(output_memory) ==
                     extract_format(gemm_pd.dst_primitive_desc())) {
                    op_output_memory = mkldnn::memory(
                      {{{extract_dims(output_memory)},
                        extract_data_type(output_memory),
                        extract_format(gemm_pd.dst_primitive_desc())},
                       engine},
                      output_memory.get_data_handle());
                    output_memory_cache.add_cached_memory(*op_output_memory);
                } else {
                    op_output_memory = mkldnn::memory(
                      {{{extract_dims(output_memory)},
                        extract_data_type(output_memory),
                        extract_format(gemm_pd.dst_primitive_desc())},
                       engine});
                    output_memory_cache.add_cached_memory(*op_output_memory);
                    output_memory_cache.add_cached_memory(output_memory);
                }

                primitives.push_back(mkldnn::inner_product_forward(
                  gemm_pd, input_memory, weight_memory, bias_memory,
                  *op_output_memory));

                if(extract_format(output_memory) !=
                     mkldnn::memory::format::any &&
                   extract_format(output_memory) !=
                     extract_format(gemm_pd.dst_primitive_desc())) {
                    primitives.push_back(
                      mkldnn::reorder(*op_output_memory, output_memory));
                }

                std::vector<std::pair<std::string, memory_cache>>
                  output_memory_cache_list({std::make_pair(
                    node.output_name_list.at(0), output_memory_cache)});
                return std::make_tuple(primitives, output_memory_cache_list);
            }

        } // namespace mkldnn_backend
    }     // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_GEMM_HPP
