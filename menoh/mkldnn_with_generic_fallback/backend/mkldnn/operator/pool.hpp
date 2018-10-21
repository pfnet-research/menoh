#ifndef MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_POOL_HPP
#define MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_POOL_HPP

#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/formatted_array.hpp>
#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/memory_cache.hpp>
#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/operator/output_management.hpp>
#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/procedure_factory.hpp>

#include <mkldnn.hpp>

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {
        namespace mkldnn_backend {

            inline procedure_factory_return_type make_pool_impl(
              mkldnn::algorithm pooling_alg,
              MENOH_MKLDNN_CONTEXT_PROCEDURE_FACTORY_PARAMETER_LIST) {

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

                auto output_dims =
                  output_formatted_array_list.at(0).array().dims();
                assert(output_dims.at(0) == input_dims.at(0) &&
                       "invalid shape inference");
                auto pool_output_md = mkldnn::memory::desc(
                  {output_dims}, input_memory_cache.data_type(),
                  mkldnn::memory::format::any);

                mkldnn::pooling_forward::desc pool_desc(
                  mkldnn::prop_kind::forward_inference, pooling_alg,
                  input_memory.get_primitive_desc().desc(), pool_output_md,
                  strides, kernel_shape, padding_l, padding_r,
                  mkldnn::padding_kind::zero);
                mkldnn::pooling_forward::primitive_desc pool_pd(pool_desc,
                                                                engine);

                auto output_memory_cache = manage_output(
                  output_formatted_array_list.at(0),
                  pool_pd.dst_primitive_desc(), engine, primitives,
                  [&pool_pd,
                   &input_memory](mkldnn::memory const& output_memory) {
                      return mkldnn::pooling_forward(pool_pd, input_memory,
                                                     output_memory);
                  });

                return procedure_factory_return_type{
                  primitives, {output_memory_cache}, {}};
            }

            inline procedure_factory_return_type make_average_pool(
              MENOH_MKLDNN_CONTEXT_PROCEDURE_FACTORY_PARAMETER_LIST) {
                auto pooling_alg =
                  attribute_int(node, "count_include_pad")
                    ? mkldnn::algorithm::pooling_avg_include_padding
                    : mkldnn::algorithm::pooling_avg_exclude_padding;
                return make_pool_impl(
                  pooling_alg,
                  MENOH_MKLDNN_CONTEXT_PROCEDURE_FACTORY_ARGUMENT_LIST);
            }

            inline procedure_factory_return_type make_max_pool(
              MENOH_MKLDNN_CONTEXT_PROCEDURE_FACTORY_PARAMETER_LIST) {
                if(node.output_name_list.size() != 1) {
                    throw std::runtime_error(
                      "MaxPool issuing multiple outputs");
                }
                return make_pool_impl(
                  mkldnn::algorithm::pooling_max,
                  MENOH_MKLDNN_CONTEXT_PROCEDURE_FACTORY_ARGUMENT_LIST);
            }

        } // namespace mkldnn_backend
    }     // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_POOL_HPP
