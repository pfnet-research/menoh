#ifndef MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_BATCH_NORM_HPP
#define MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_BATCH_NORM_HPP

#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/memory_cache.hpp>

#include <mkldnn.hpp>

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {
        namespace mkldnn_backend {

            inline std::tuple<std::vector<mkldnn::primitive>,
                              std::vector<std::pair<std::string, memory_cache>>>
            make_batch_norm(
              node const& node,
              std::vector<std::reference_wrapper<memory_cache>> const&
                input_memory_cache_list,
              std::vector<mkldnn::memory> const& output_memory_list,
              mkldnn::engine const& engine) {

                std::vector<mkldnn::primitive> primitives;
                std::vector<std::pair<std::string, memory_cache>>
                  output_memory_cache_list;

                auto epsilon = attribute_float(node, "epsilon");
                auto spatial = attribute_int(node, "spatial");
                if(!spatial) {
                    throw std::runtime_error(
                      "Non spacial BatchNorm is not supported");
                }

                memory_cache& input_memory_cache =
                  input_memory_cache_list.at(0);
                auto input_dims = input_memory_cache.dims();
                auto input_memory = input_memory_cache.get_data_memory();

                auto mean_memory =
                  get_memory(input_memory_cache_list.at(3),
                             mkldnn::memory::format::x, primitives);
                auto var_memory =
                  get_memory(input_memory_cache_list.at(4),
                             mkldnn::memory::format::x, primitives);

                memory_cache& scale_memory_cache =
                  input_memory_cache_list.at(1);
                auto scale_dims = scale_memory_cache.dims();
                memory_cache& b_memory_cache = input_memory_cache_list.at(2);
                std::vector<int> weight_dims({2});
                weight_dims.insert(weight_dims.end(), scale_dims.begin(),
                                   scale_dims.end());
                auto scale_memory = get_memory(scale_memory_cache, scale_dims,
                                               mkldnn::memory::x, primitives);
                auto b_memory = get_memory(b_memory_cache, scale_dims,
                                           mkldnn::memory::x, primitives);
                mkldnn::memory weight_memory({{{weight_dims},
                                               extract_data_type(scale_memory),
                                               mkldnn::memory::format::nc},
                                              engine});
                auto scale_total_size =
                  std::accumulate(scale_dims.begin(), scale_dims.end(), 1,
                                  std::multiplies<int>());
                // TODO other data_type
                if(extract_data_type(scale_memory) ==
                   mkldnn::memory::data_type::f32) {
                    std::copy(
                      static_cast<float*>(scale_memory.get_data_handle()),
                      static_cast<float*>(scale_memory.get_data_handle()) +
                        scale_total_size,
                      static_cast<float*>(weight_memory.get_data_handle()));
                    std::copy(
                      static_cast<float*>(b_memory.get_data_handle()),
                      static_cast<float*>(b_memory.get_data_handle()) +
                        scale_total_size,
                      static_cast<float*>(weight_memory.get_data_handle()) +
                        scale_total_size);
                } else {
                    throw std::runtime_error(
                      "BatchNormalization with invalid data_type");
                }
                memory_cache weight_memory_cache(weight_memory);
                output_memory_cache_list.emplace_back(
                  "menoh_mkldnn_temp_memory_" + node.op_type + "_" +
                    node.output_name_list.front() + "_weight_memory",
                  weight_memory_cache);

                auto const& output_memory = output_memory_list.at(0);
                auto output_dims = extract_dims(output_memory);
                assert(output_dims.at(0) == input_dims.at(0) &&
                       "invalid shape inference");

                mkldnn::batch_normalization_forward::desc batch_norm_desc(
                  mkldnn::prop_kind::forward_inference,
                  input_memory.get_primitive_desc().desc(), epsilon,
                  mkldnn::use_global_stats | mkldnn::use_scale_shift |
                    omit_stats);
                mkldnn::batch_normalization_forward::primitive_desc
                  batch_norm_pd(batch_norm_desc, engine);

                memory_cache output_memory_cache;
                optional<mkldnn::memory> op_output_memory;
                if(extract_format(output_memory) ==
                     mkldnn::memory::format::any ||
                   extract_format(output_memory) ==
                     extract_format(batch_norm_pd.dst_primitive_desc())) {
                    op_output_memory = mkldnn::memory(
                      {{{extract_dims(output_memory)},
                        extract_data_type(output_memory),
                        extract_format(batch_norm_pd.dst_primitive_desc())},
                       engine},
                      output_memory.get_data_handle());
                    output_memory_cache.add_cached_memory(*op_output_memory);
                } else {
                    op_output_memory = mkldnn::memory(
                      {{{extract_dims(output_memory)},
                        extract_data_type(output_memory),
                        extract_format(batch_norm_pd.dst_primitive_desc())},
                       engine});
                    output_memory_cache.add_cached_memory(*op_output_memory);
                    output_memory_cache.add_cached_memory(output_memory);
                }

                primitives.push_back(mkldnn::batch_normalization_forward(
                  batch_norm_pd,
                  static_cast<mkldnn::primitive::at>(input_memory),
                  static_cast<mkldnn::primitive::at>(mean_memory),
                  static_cast<mkldnn::primitive::at>(var_memory),
                  static_cast<mkldnn::primitive::at>(weight_memory),
                  *op_output_memory));

                if(extract_format(output_memory) !=
                     mkldnn::memory::format::any &&
                   extract_format(output_memory) !=
                     extract_format(batch_norm_pd.dst_primitive_desc())) {
                    primitives.push_back(
                      mkldnn::reorder(*op_output_memory, output_memory));
                }

                output_memory_cache_list.emplace_back(
                  node.output_name_list.at(0), output_memory_cache);
                return std::make_tuple(primitives, output_memory_cache_list);
            }

        } // namespace mkldnn_backend
    }     // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_BATCH_NORM_HPP
