#ifndef MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_POOL_HPP
#define MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_POOL_HPP

#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/memory_cache.hpp>

#include <mkldnn.hpp>

#include <iostream>

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {
        namespace mkldnn_backend {

            inline std::tuple<std::vector<mkldnn::primitive>,
                              std::vector<std::pair<std::string, memory_cache>>>
            make_pool(mkldnn::algorithm pooling_alg, node const& node,
                      std::vector<std::reference_wrapper<memory_cache>> const&
                        input_memory_cache_list,
                      std::vector<mkldnn::memory> const& output_memory_list,
                      mkldnn::engine const& engine) {

                assert(pooling_alg == mkldnn::pooling_max ||
                       pooling_alg == mkldnn::pooling_avg_include_padding ||
                       pooling_alg == mkldnn::pooling_avg_exclude_padding);

                std::vector<mkldnn::primitive> primitives;
                std::vector<std::pair<std::string, memory_cache>>
                  output_memory_cache_list;

                std::vector<int> strides, kernel_shape, pads;
                std::tie(strides, kernel_shape, pads) =
                  attributes_for_2d_data_processing(node);
                std::vector<int> padding_l{pads[0], pads[1]};
                std::vector<int> padding_r{pads[2], pads[3]};

                memory_cache& input_memory_cache =
                  input_memory_cache_list.at(0);
                auto input_dims = input_memory_cache.dims();
                auto input_memory = input_memory_cache.get_data_memory();

                auto const& output_memory = output_memory_list.at(0);
                auto output_dims = extract_dims(output_memory);
                assert(output_dims.at(0) == input_dims.at(0) &&
                       "invalid shape inference");
                auto pool_output_md = mkldnn::memory::desc(
                  {output_dims}, extract_data_type(output_memory),
                  mkldnn::memory::format::any);

                mkldnn::pooling_forward::desc pool_desc(
                  mkldnn::prop_kind::forward_inference, pooling_alg,
                  input_memory.get_primitive_desc().desc(), pool_output_md,
                  strides, kernel_shape, padding_l, padding_r,
                  mkldnn::padding_kind::zero);
                mkldnn::pooling_forward::primitive_desc pool_pd(pool_desc,
                                                                engine);

                memory_cache output_memory_cache;
                optional<mkldnn::memory> op_output_memory;
                if(extract_format(output_memory) ==
                     mkldnn::memory::format::any ||
                   extract_format(output_memory) ==
                     extract_format(pool_pd.dst_primitive_desc())) {
                    op_output_memory = mkldnn::memory(
                      {{{extract_dims(output_memory)},
                        extract_data_type(output_memory),
                        extract_format(pool_pd.dst_primitive_desc())},
                       engine},
                      output_memory.get_data_handle());
                    output_memory_cache.add_cached_memory(*op_output_memory);
                } else {
                    op_output_memory = mkldnn::memory(
                      {{{extract_dims(output_memory)},
                        extract_data_type(output_memory),
                        extract_format(pool_pd.dst_primitive_desc())},
                       engine});
                    output_memory_cache.add_cached_memory(*op_output_memory);
                    output_memory_cache.add_cached_memory(output_memory);
                }

                primitives.push_back(mkldnn::pooling_forward(
                  pool_pd, input_memory, *op_output_memory));

                if(extract_format(output_memory) !=
                     mkldnn::memory::format::any &&
                   extract_format(output_memory) !=
                     extract_format(pool_pd.dst_primitive_desc())) {
                    primitives.push_back(
                      mkldnn::reorder(*op_output_memory, output_memory));
                }

                output_memory_cache_list.emplace_back(
                  node.output_name_list.at(0), output_memory_cache);
                return std::make_tuple(primitives, output_memory_cache_list);
            }

            inline std::tuple<std::vector<mkldnn::primitive>,
                              std::vector<std::pair<std::string, memory_cache>>>
            make_average_pool(
              node const& node,
              std::vector<std::reference_wrapper<memory_cache>> const&
                input_memory_cache_list,
              std::vector<mkldnn::memory> const& output_memory_list,
              mkldnn::engine const& engine) {
                auto pooling_alg =
                  attribute_int(node, "count_include_pad")
                    ? mkldnn::algorithm::pooling_avg_include_padding
                    : mkldnn::algorithm::pooling_avg_exclude_padding;
                return make_pool(pooling_alg, node, input_memory_cache_list,
                                 output_memory_list, engine);
            }

            inline std::tuple<std::vector<mkldnn::primitive>,
                              std::vector<std::pair<std::string, memory_cache>>>
            make_max_pool(
              node const& node,
              std::vector<std::reference_wrapper<memory_cache>> const&
                input_memory_cache_list,
              std::vector<mkldnn::memory> const& output_memory_list,
              mkldnn::engine const& engine) {
                if(node.output_name_list.size() != 1) {
                    throw std::runtime_error(
                      "MaxPool issuing multiple outputs");
                }
                return make_pool(mkldnn::algorithm::pooling_max, node,
                                 input_memory_cache_list, output_memory_list,
                                 engine);
            }

        } // namespace mkldnn_backend
    }     // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_POOL_HPP
