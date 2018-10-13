#ifndef MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_ELTWISE_HPP
#define MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_ELTWISE_HPP

#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/memory_cache.hpp>

#include <mkldnn.hpp>

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {
        namespace mkldnn_backend {

            inline std::tuple<std::vector<mkldnn::primitive>,
                              std::vector<std::pair<std::string, memory_cache>>>
            make_eltwise(
              mkldnn::algorithm eltwise_alg, float alpha, float beta,
              node const& node,
              std::vector<std::reference_wrapper<memory_cache>> const&
                input_memory_cache_list,
              std::vector<mkldnn::memory> const& output_memory_list,
              mkldnn::engine const& engine) {

                std::vector<mkldnn::primitive> primitives;
                std::vector<mkldnn::memory> temp_memory_list;
                std::vector<array> owned_array_list;

                memory_cache& input_memory_cache =
                  input_memory_cache_list.at(0);
                auto input_dims = input_memory_cache.dims();
                auto input_memory = input_memory_cache.get_data_memory();

                auto const& output_memory = output_memory_list.at(0);
                auto output_dims = extract_dims(output_memory);

                mkldnn::eltwise_forward::desc eltwise_desc(
                  mkldnn::prop_kind::forward_inference, eltwise_alg,
                  input_memory.get_primitive_desc().desc(), alpha, beta);
                auto eltwise_pd =
                  mkldnn::eltwise_forward::primitive_desc(eltwise_desc, engine);

                memory_cache output_memory_cache;
                optional<mkldnn::memory> op_output_memory;
                if(extract_format(output_memory) ==
                     mkldnn::memory::format::any ||
                   extract_format(output_memory) ==
                     extract_format(eltwise_pd.dst_primitive_desc())) {
                    op_output_memory = mkldnn::memory(
                      {{{extract_dims(output_memory)},
                        extract_data_type(output_memory),
                        extract_format(eltwise_pd.dst_primitive_desc())},
                       engine},
                      output_memory.get_data_handle());
                    output_memory_cache.add_cached_memory(*op_output_memory);
                } else {
                    op_output_memory = mkldnn::memory(
                      {{{extract_dims(output_memory)},
                        extract_data_type(output_memory),
                        extract_format(eltwise_pd.dst_primitive_desc())},
                       engine});
                    output_memory_cache.add_cached_memory(*op_output_memory);
                    output_memory_cache.add_cached_memory(output_memory);
                }

                primitives.push_back(mkldnn::eltwise_forward(
                  eltwise_pd, input_memory, *op_output_memory));

                if(extract_format(output_memory) !=
                     mkldnn::memory::format::any &&
                   extract_format(output_memory) !=
                     extract_format(eltwise_pd.dst_primitive_desc())) {
                    primitives.push_back(
                      mkldnn::reorder(*op_output_memory, output_memory));
                }

                std::vector<std::pair<std::string, memory_cache>>
                  output_memory_cache_list({std::make_pair(
                    node.output_name_list.at(0), output_memory_cache)});
                return std::make_tuple(primitives, output_memory_cache_list);
            }

            inline std::tuple<std::vector<mkldnn::primitive>,
                              std::vector<std::pair<std::string, memory_cache>>>
            make_abs(node const& node,
                     std::vector<std::reference_wrapper<memory_cache>> const&
                       input_memory_cache_list,
                     std::vector<mkldnn::memory> const& output_memory_list,
                     mkldnn::engine const& engine) {
                auto alpha = 0.f;
                auto beta = 0.f;
                return make_eltwise(mkldnn::algorithm::eltwise_abs, alpha, beta,
                                    node, input_memory_cache_list,
                                    output_memory_list, engine);
            }

            inline std::tuple<std::vector<mkldnn::primitive>,
                              std::vector<std::pair<std::string, memory_cache>>>
            make_elu(node const& node,
                     std::vector<std::reference_wrapper<memory_cache>> const&
                       input_memory_cache_list,
                     std::vector<mkldnn::memory> const& output_memory_list,
                     mkldnn::engine const& engine) {
                auto alpha = attribute_float(node, "alpha");
                auto beta = 0.f;
                return make_eltwise(mkldnn::algorithm::eltwise_elu, alpha, beta,
                                    node, input_memory_cache_list,
                                    output_memory_list, engine);
            }

            inline std::tuple<std::vector<mkldnn::primitive>,
                              std::vector<std::pair<std::string, memory_cache>>>
            make_leaky_relu(
              node const& node,
              std::vector<std::reference_wrapper<memory_cache>> const&
                input_memory_cache_list,
              std::vector<mkldnn::memory> const& output_memory_list,
              mkldnn::engine const& engine) {
                auto alpha = attribute_float(node, "alpha");
                auto beta = 0.f;
                return make_eltwise(mkldnn::algorithm::eltwise_relu,
                                    alpha, beta, node, input_memory_cache_list,
                                    output_memory_list, engine);
            }

            inline std::tuple<std::vector<mkldnn::primitive>,
                              std::vector<std::pair<std::string, memory_cache>>>
            make_relu(node const& node,
                      std::vector<std::reference_wrapper<memory_cache>> const&
                        input_memory_cache_list,
                      std::vector<mkldnn::memory> const& output_memory_list,
                      mkldnn::engine const& engine) {
                auto alpha = 0.f;
                auto beta = 0.f;
                return make_eltwise(mkldnn::algorithm::eltwise_relu, alpha,
                                    beta, node, input_memory_cache_list,
                                    output_memory_list, engine);
            }

            inline std::tuple<std::vector<mkldnn::primitive>,
                              std::vector<std::pair<std::string, memory_cache>>>
            make_sqrt(node const& node,
                      std::vector<std::reference_wrapper<memory_cache>> const&
                        input_memory_cache_list,
                      std::vector<mkldnn::memory> const& output_memory_list,
                      mkldnn::engine const& engine) {
                auto alpha = 0.f;
                auto beta = 0.f;
                return make_eltwise(mkldnn::algorithm::eltwise_sqrt, alpha,
                                    beta, node, input_memory_cache_list,
                                    output_memory_list, engine);
            }

            inline std::tuple<std::vector<mkldnn::primitive>,
                              std::vector<std::pair<std::string, memory_cache>>>
            make_tanh(node const& node,
                      std::vector<std::reference_wrapper<memory_cache>> const&
                        input_memory_cache_list,
                      std::vector<mkldnn::memory> const& output_memory_list,
                      mkldnn::engine const& engine) {
                auto alpha = 0.f;
                auto beta = 0.f;
                return make_eltwise(mkldnn::algorithm::eltwise_tanh, alpha,
                                    beta, node, input_memory_cache_list,
                                    output_memory_list, engine);
            }

        } // namespace mkldnn_backend
    }     // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_ELTWISE_HPP
