#ifndef MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_CONV_HPP
#define MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_CONV_HPP

#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/formatted_array.hpp>
#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/memory_cache.hpp>
#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/operator/output_management.hpp>
#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/procedure_factory.hpp>

#include <mkldnn.hpp>

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {
        namespace mkldnn_backend {

            inline procedure_factory_return_type
            make_conv(MENOH_MKLDNN_CONTEXT_PROCEDURE_FACTORY_PARAMETER_LIST) {

                std::vector<mkldnn::primitive> primitives;

                std::vector<int> strides, kernel_shape, pads;
                std::tie(strides, kernel_shape, pads) =
                  attributes_for_2d_data_processing(node);
                std::vector<int> padding_l{pads[0], pads[1]};
                std::vector<int> padding_r{pads[2], pads[3]};

                memory_cache& input_memory_cache =
                  input_memory_cache_list.at(0);
                auto input_dims = input_memory_cache.dims();

                memory_cache& weight_memory_cache =
                  input_memory_cache_list.at(1);
                auto weight_dims = weight_memory_cache.dims();

                auto output_dims =
                  output_formatted_array_list.at(0).array().dims();
                assert(output_dims.at(0) == input_dims.at(0) &&
                       "invalid shape inference");
                auto conv_input_md = mkldnn::memory::desc(
                  {input_dims}, input_memory_cache.data_type(),
                  mkldnn::memory::format::any);
                auto conv_weight_md = mkldnn::memory::desc(
                  {weight_dims}, weight_memory_cache.data_type(),
                  mkldnn::memory::format::any);
                auto conv_output_md = mkldnn::memory::desc(
                  {output_dims}, input_memory_cache.data_type(),
                  mkldnn::memory::format::any);

                optional<mkldnn::memory> bias_memory_opt;
                menoh_impl::optional<mkldnn::convolution_forward::desc>
                  conv_desc_opt;
                auto dilations = attribute_ints(node, "dilations");
                auto is_no_dilations =
                  std::all_of(dilations.begin(), dilations.end(),
                              [](auto e) { return e == 1; });
                if(node.input_name_list.size() == 2) {
                    if(is_no_dilations) {
                        conv_desc_opt = mkldnn::convolution_forward::desc(
                          mkldnn::prop_kind::forward_inference,
                          mkldnn::algorithm::convolution_direct, conv_input_md,
                          conv_weight_md, conv_output_md, strides, padding_l,
                          padding_r, mkldnn::padding_kind::zero);
                    } else {
                        conv_desc_opt = mkldnn::convolution_forward::desc(
                          mkldnn::prop_kind::forward_inference,
                          mkldnn::algorithm::convolution_direct, conv_input_md,
                          conv_weight_md, conv_output_md, strides, dilations,
                          padding_l, padding_r, mkldnn::padding_kind::zero);
                    }
                } else {
                    assert(node.input_name_list.size() == 3);

                    memory_cache& bias_memory_cache =
                      input_memory_cache_list.at(2);
                    bias_memory_opt = get_memory(
                      bias_memory_cache, mkldnn::memory::format::x, primitives);

                    if(is_no_dilations) {
                        conv_desc_opt = mkldnn::convolution_forward::desc(
                          mkldnn::prop_kind::forward_inference,
                          mkldnn::algorithm::convolution_direct, conv_input_md,
                          conv_weight_md,
                          bias_memory_opt->get_primitive_desc().desc(),
                          conv_output_md, strides, padding_l, padding_r,
                          mkldnn::padding_kind::zero);
                    } else {
                        conv_desc_opt = mkldnn::convolution_forward::desc(
                          mkldnn::prop_kind::forward_inference,
                          mkldnn::algorithm::convolution_direct, conv_input_md,
                          conv_weight_md,
                          bias_memory_opt->get_primitive_desc().desc(),
                          conv_output_md, strides, dilations, padding_l,
                          padding_r, mkldnn::padding_kind::zero);
                    }
                }
                auto conv_desc = *conv_desc_opt;
                auto conv_pd = mkldnn::convolution_forward::primitive_desc(
                  conv_desc, engine);

                auto input_memory = get_memory(
                  input_memory_cache,
                  extract_format(conv_pd.src_primitive_desc()), primitives);
                auto weight_memory = get_memory(
                  weight_memory_cache, weight_dims,
                  extract_format(conv_pd.weights_primitive_desc()), primitives);

                auto output_memory_cache = manage_output(
                  output_formatted_array_list.at(0),
                  conv_pd.dst_primitive_desc(), engine, primitives,
                  [&conv_pd, &input_memory, &weight_memory,
                   &bias_memory_opt](mkldnn::memory const& output_memory) {
                      if(bias_memory_opt) {
                          return mkldnn::convolution_forward(
                            conv_pd, input_memory, weight_memory,
                            *bias_memory_opt, output_memory);
                      } else {
                          return mkldnn::convolution_forward(
                            conv_pd, input_memory, weight_memory,
                            output_memory);
                      }
                  });

                return procedure_factory_return_type{
                  primitives, {output_memory_cache}, {}};
            }

        } // namespace mkldnn_backend
    }     // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_CONV_HPP
