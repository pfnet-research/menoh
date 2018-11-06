#ifndef MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_BATCH_NORM_HPP
#define MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_BATCH_NORM_HPP

#include <menoh/combinated_backends/backend/mkldnn/formatted_array.hpp>
#include <menoh/combinated_backends/backend/mkldnn/memory_cache.hpp>
#include <menoh/combinated_backends/backend/mkldnn/operator/output_management.hpp>
#include <menoh/combinated_backends/backend/mkldnn/procedure_factory.hpp>

#include <mkldnn.hpp>

namespace menoh_impl {
    namespace combinated_backends {
        namespace mkldnn_backend {

            inline procedure_factory_return_type make_batch_norm(
              MENOH_MKLDNN_CONTEXT_PROCEDURE_FACTORY_PARAMETER_LIST) {

                std::vector<mkldnn::primitive> primitives;
                std::vector<std::pair<std::string, memory_cache>>
                  temp_memory_cache_list;

                auto epsilon = attribute_float(node, "epsilon");
                auto spatial = attribute_int(node, "spatial");
                if(!spatial) {
                    throw std::runtime_error(
                      "Non spacial BatchNormalization is not supported");
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
                temp_memory_cache_list.emplace_back(
                  "menoh_mkldnn_temp_memory_" + node.op_type + "_" +
                    node.output_name_list.front() + "_weight_memory",
                  weight_memory_cache);

                auto output_dims =
                  output_formatted_array_list.at(0).array().dims();
                assert(output_dims.at(0) == input_dims.at(0) &&
                       "invalid shape inference");

                mkldnn::batch_normalization_forward::desc batch_norm_desc(
                  mkldnn::prop_kind::forward_inference,
                  input_memory.get_primitive_desc().desc(), epsilon,
                  mkldnn::use_global_stats | mkldnn::use_scale_shift |
                    mkldnn::omit_stats);
                mkldnn::batch_normalization_forward::primitive_desc
                  batch_norm_pd(batch_norm_desc, engine);

                auto output_memory_cache = manage_output(
                  output_formatted_array_list.at(0),
                  batch_norm_pd.dst_primitive_desc(), engine, primitives,
                  [&batch_norm_pd, &input_memory, &mean_memory, &var_memory,
                   &weight_memory](mkldnn::memory const& output_memory) {
                      return mkldnn::batch_normalization_forward(
                        batch_norm_pd,
                        static_cast<mkldnn::primitive::at>(input_memory),
                        static_cast<mkldnn::primitive::at>(mean_memory),
                        static_cast<mkldnn::primitive::at>(var_memory),
                        static_cast<mkldnn::primitive::at>(weight_memory),
                        output_memory);
                  });

                return procedure_factory_return_type{
                  primitives, {output_memory_cache}, temp_memory_cache_list};
            }

        } // namespace mkldnn_backend
    }     // namespace combinated_backends
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_BATCH_NORM_HPP
