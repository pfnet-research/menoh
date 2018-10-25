#ifndef MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_GEMM_HPP
#define MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_GEMM_HPP

#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/formatted_array.hpp>
#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/memory_cache.hpp>
#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/operator/output_management.hpp>
#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/procedure_factory.hpp>

#include <mkldnn.hpp>

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {
        namespace mkldnn_backend {

            inline procedure_factory_return_type
            make_gemm(MENOH_MKLDNN_CONTEXT_PROCEDURE_FACTORY_PARAMETER_LIST) {

                std::vector<mkldnn::primitive> primitives;

                auto alpha = attribute_float(node, "alpha");
                if(alpha != 1.f) {
                    throw failed_to_configure_operator(
                      node.op_type, node.output_name_list.at(0),
                      "alpha of Gemm must be 1 but given: " +
                        std::to_string(alpha));
                }
                auto beta = attribute_float(node, "beta");
                if(beta != 1.f) {
                    throw failed_to_configure_operator(
                      node.op_type, node.output_name_list.at(0),
                      "beta of Gemm must be 1 but given: " +
                        std::to_string(beta));
                }

                auto trans_a = attribute_int(node, "transA");
                if(trans_a) {
                    throw failed_to_configure_operator(
                      node.op_type, node.output_name_list.at(0),
                      "transA of Gemm must be 0 but given: " +
                        std::to_string(trans_a));
                }
                auto trans_b = attribute_int(node, "transB");
                if(!trans_b) {
                    throw failed_to_configure_operator(
                      node.op_type, node.output_name_list.at(0),
                      "transB of Gemm must be 0 but given: " +
                        std::to_string(trans_b));
                }

                memory_cache& input_memory_cache =
                  input_memory_cache_list.at(0);
                auto input_dims = input_memory_cache.dims();

                memory_cache& weight_memory_cache =
                  input_memory_cache_list.at(1);
                auto weight_dims = weight_memory_cache.dims(); // mutable
                assert(weight_dims.size() == 2);
                if(input_dims.size() != 2) {
                    weight_dims = std::vector<int>({weight_dims.front()});
                    weight_dims.insert(weight_dims.end(),
                                       input_dims.begin() + 1,
                                       input_dims.end());
                }

                memory_cache& bias_memory_cache = input_memory_cache_list.at(2);
                auto bias_dims = bias_memory_cache.dims();
                int output_size = weight_dims.at(0);
                if(output_size != bias_dims.at(0)) {
                    throw std::runtime_error("broadcast is not supported yet");
                }

                auto output_dims =
                  output_formatted_array_list.at(0).array().dims();
                assert(output_dims.at(0) == input_dims.at(0) &&
                       "invalid shape inference");
                assert(output_dims.at(1) == output_size &&
                       "invalid shape inference");
                auto gemm_input_md = mkldnn::memory::desc(
                  {input_dims}, input_memory_cache.data_type(),
                  mkldnn::memory::format::any);
                auto gemm_weight_md = mkldnn::memory::desc(
                  {weight_dims}, weight_memory_cache.data_type(),
                  mkldnn::memory::format::any);
                auto gemm_output_md = mkldnn::memory::desc(
                  {output_dims}, input_memory_cache.data_type(),
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

                auto output_memory_cache = manage_output(
                  output_formatted_array_list.at(0),
                  gemm_pd.dst_primitive_desc(), engine, primitives,
                  [&gemm_pd, &input_memory, &weight_memory,
                   &bias_memory](mkldnn::memory const& output_memory) {
                      return mkldnn::inner_product_forward(
                        gemm_pd, input_memory, weight_memory, bias_memory,
                        output_memory);
                  });

                return procedure_factory_return_type{
                  primitives, {output_memory_cache}, {}};
            }

        } // namespace mkldnn_backend
    }     // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_GEMM_HPP
