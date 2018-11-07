#ifndef MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_ELTWISE_HPP
#define MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_ELTWISE_HPP

#include <menoh/composite_backend/backend/mkldnn/formatted_array.hpp>
#include <menoh/composite_backend/backend/mkldnn/memory_cache.hpp>
#include <menoh/composite_backend/backend/mkldnn/operator/output_management.hpp>
#include <menoh/composite_backend/backend/mkldnn/procedure_factory.hpp>

#include <mkldnn.hpp>

namespace menoh_impl {
    namespace composite_backend {
        namespace mkldnn_backend {

            inline procedure_factory_return_type make_eltwise(
              mkldnn::algorithm eltwise_alg, float alpha, float beta,
              MENOH_MKLDNN_CONTEXT_PROCEDURE_FACTORY_PARAMETER_LIST) {

                std::vector<mkldnn::primitive> primitives;

                memory_cache& input_memory_cache =
                  input_memory_cache_list.at(0);
                auto input_dims = input_memory_cache.dims();
                auto input_memory = input_memory_cache.get_data_memory();

                auto output_dims =
                  output_formatted_array_list.at(0).array().dims();

                mkldnn::eltwise_forward::desc eltwise_desc(
                  mkldnn::prop_kind::forward_inference, eltwise_alg,
                  input_memory.get_primitive_desc().desc(), alpha, beta);
                auto eltwise_pd =
                  mkldnn::eltwise_forward::primitive_desc(eltwise_desc, engine);

                auto output_memory_cache = manage_output(
                  output_formatted_array_list.at(0),
                  eltwise_pd.dst_primitive_desc(), engine, primitives,
                  [&eltwise_pd,
                   &input_memory](mkldnn::memory const& output_memory) {
                      return mkldnn::eltwise_forward(eltwise_pd, input_memory,
                                                     output_memory);
                  });

                return procedure_factory_return_type{
                  primitives, {output_memory_cache}, {}};
            }

#define MENOH_MKLDNN_CONTEXT_ELTWISE_FACTORY(factory_name, eltwise_alg, alpha, \
                                             beta)                             \
    inline procedure_factory_return_type factory_name(                         \
      MENOH_MKLDNN_CONTEXT_PROCEDURE_FACTORY_PARAMETER_LIST) {                 \
        return make_eltwise(                                                   \
          eltwise_alg, alpha, beta,                                            \
          MENOH_MKLDNN_CONTEXT_PROCEDURE_FACTORY_ARGUMENT_LIST);               \
    }
#define MENOH_MKLDNN_CONTEXT_ELTWISE_FACTORY_NO_PARAM(factory_name, \
                                                      eltwise_alg)  \
    MENOH_MKLDNN_CONTEXT_ELTWISE_FACTORY(factory_name, eltwise_alg, 0.f, 0.f)

            MENOH_MKLDNN_CONTEXT_ELTWISE_FACTORY_NO_PARAM(
              make_abs, mkldnn::algorithm::eltwise_abs);
            MENOH_MKLDNN_CONTEXT_ELTWISE_FACTORY_NO_PARAM(
              make_sqrt, mkldnn::algorithm::eltwise_sqrt);
            MENOH_MKLDNN_CONTEXT_ELTWISE_FACTORY_NO_PARAM(
              make_tanh, mkldnn::algorithm::eltwise_tanh);

            MENOH_MKLDNN_CONTEXT_ELTWISE_FACTORY(make_elu,
                                                 mkldnn::algorithm::eltwise_elu,
                                                 attribute_float(node, "alpha"),
                                                 0.f);
            MENOH_MKLDNN_CONTEXT_ELTWISE_FACTORY(
              make_leaky_relu, mkldnn::algorithm::eltwise_relu,
              attribute_float(node, "alpha"), 0.f);
            MENOH_MKLDNN_CONTEXT_ELTWISE_FACTORY_NO_PARAM(
              make_relu, mkldnn::algorithm::eltwise_relu);

#undef MENOH_MKLDNN_CONTEXT_ELTWISE_FACTORY_NO_PARAM
#undef MENOH_MKLDNN_CONTEXT_ELTWISE_FACTORY

        } // namespace mkldnn_backend
    }     // namespace composite_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_ELTWISE_HPP
