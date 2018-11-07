#ifndef MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_SOFTMAX_HPP
#define MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_SOFTMAX_HPP

#include <menoh/composite_backend/backend/mkldnn/formatted_array.hpp>
#include <menoh/composite_backend/backend/mkldnn/memory_cache.hpp>
#include <menoh/composite_backend/backend/mkldnn/operator/output_management.hpp>
#include <menoh/composite_backend/backend/mkldnn/procedure_factory.hpp>

#include <mkldnn.hpp>

namespace menoh_impl {
    namespace composite_backend {
        namespace mkldnn_backend {

            inline procedure_factory_return_type make_softmax(
              MENOH_MKLDNN_CONTEXT_PROCEDURE_FACTORY_PARAMETER_LIST) {

                std::vector<mkldnn::primitive> primitives;

                auto axis = attribute_int(node, "axis");

                memory_cache& input_memory_cache =
                  input_memory_cache_list.at(0);
                auto input_dims = input_memory_cache.dims();
                auto input_memory = input_memory_cache.get_data_memory();

                auto output_dims =
                  output_formatted_array_list.at(0).array().dims();
                assert(output_dims.at(0) == input_dims.at(0) &&
                       "invalid shape inference");

                mkldnn::softmax_forward::desc softmax_desc(
                  mkldnn::prop_kind::forward_inference,
                  input_memory.get_primitive_desc().desc(), axis);
                auto softmax_pd =
                  mkldnn::softmax_forward::primitive_desc(softmax_desc, engine);

                auto output_memory_cache = manage_output(
                  output_formatted_array_list.at(0),
                  input_memory.get_primitive_desc(), engine, primitives,
                  [&softmax_pd,
                   &input_memory](mkldnn::memory const& output_memory) {
                      return mkldnn::softmax_forward(softmax_pd, input_memory,
                                                     output_memory);
                  });

                return procedure_factory_return_type{
                  primitives, {output_memory_cache}, {}};
            }

        } // namespace mkldnn_backend
    }     // namespace composite_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_SOFTMAX_HPP
