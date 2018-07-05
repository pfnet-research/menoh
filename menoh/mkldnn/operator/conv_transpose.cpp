#include <menoh/mkldnn/operator/conv_transpose.hpp>

#include <tuple>

#include <menoh/optional.hpp>
#include <menoh/utility.hpp>

#include <menoh/mkldnn/operator/common.hpp>
#include <menoh/mkldnn/utility.hpp>

namespace menoh_impl {
    namespace mkldnn_backend {

        primitive_factory_return_type make_conv_transpose_primitive(
          menoh_impl::node const& node,
          std::unordered_map<std::string, array> const& parameter_table,
          std::unordered_map<std::string, mkldnn::memory> const&
            variable_memory_table,
          std::unordered_map<std::string, array> const& required_output_table,
          mkldnn::engine const& engine) {

            std::vector<mkldnn::primitive> net;
            std::unordered_map<std::string, mkldnn::memory> output_memory_table;
            std::vector<mkldnn::memory> temp_memory_list;
            std::vector<array> owned_array_list;

            std::vector<int32_t> strides, kernel_shape, pads;
            std::tie(strides, kernel_shape, pads) =
              attributes_for_2d_data_processing(node);

            std::vector<int32_t> padding_l{pads[0], pads[1]};
            std::vector<int32_t> padding_r{pads[2], pads[3]};

            auto const& input_memory =
              find_value(variable_memory_table, node.input_name_list.at(0));
            auto weight_arr =
              find_value(parameter_table, node.input_name_list.at(1));
            auto weight_tr_dims = weight_arr.dims();
            std::swap(weight_tr_dims[0], weight_tr_dims[1]);
            menoh_impl::array weight_tr_arr(weight_arr.dtype(), weight_tr_dims);
            assert(weight_arr.dtype() == dtype_t::float_);
            for(int32_t o = 0; o < weight_arr.dims()[0]; ++o) {
                for(int32_t i = 0; i < weight_arr.dims()[1]; ++i) {
                    for(int32_t h = 0; h < weight_arr.dims()[2]; ++h) {
                        for(int32_t w = 0; w < weight_arr.dims()[3]; ++w) {
                            auto weight_i =
                              o * weight_arr.dims()[1] * weight_arr.dims()[2] *
                                weight_arr.dims()[3] +
                              i * weight_arr.dims()[2] * weight_arr.dims()[3] +
                              h * weight_arr.dims()[3] + w;
                            auto weight_tr_i =
                              i * weight_arr.dims()[0] * weight_arr.dims()[2] *
                                weight_arr.dims()[3] +
                              o * weight_arr.dims()[2] * weight_arr.dims()[3] +
                              h * weight_arr.dims()[3] + w;
                            *(static_cast<float*>(weight_tr_arr.data()) +
                              weight_tr_i) =
                              *(static_cast<float*>(weight_arr.data()) +
                                weight_i);
                        }
                    }
                }
            }
            auto weight_memory = array_to_memory_and_deal_ownership(
              weight_tr_arr, mkldnn::memory::format::oihw, engine,
              temp_memory_list, owned_array_list);

            menoh_impl::optional<mkldnn::memory> bias_memory_opt;
            if(node.input_name_list.size() == 3) {
                bias_memory_opt = array_to_memory_and_deal_ownership(
                  find_value(parameter_table, node.input_name_list.at(2)),
                  mkldnn::memory::format::x, engine, temp_memory_list,
                  owned_array_list);
            }

            auto input_dims = extract_dims(input_memory);
            auto weight_dims = extract_dims(weight_memory);
            auto output_dims = calc_2d_output_dims_for_conv_transpose(
              input_dims, weight_dims[0], kernel_shape, strides, pads);

            auto const& output_name = node.output_name_list.at(0);

            auto deconv_input_md =
              mkldnn::memory::desc({input_dims}, mkldnn::memory::data_type::f32,
                                   mkldnn::memory::format::any);
            auto deconv_weight_md = mkldnn::memory::desc(
              {weight_dims}, mkldnn::memory::data_type::f32,
              mkldnn::memory::format::any);
            auto deconv_output_md = mkldnn::memory::desc(
              {output_dims}, mkldnn::memory::data_type::f32,
              mkldnn::memory::format::any);

            menoh_impl::optional<mkldnn::deconvolution_forward::desc> deconv_desc_opt;
            if(bias_memory_opt) {
                deconv_desc_opt = mkldnn::deconvolution_forward::desc(
                  mkldnn::prop_kind::forward_inference,
                  mkldnn::algorithm::deconvolution_direct, deconv_input_md,
                  deconv_weight_md,
                  bias_memory_opt->get_primitive_desc().desc(),
                  deconv_output_md, strides, padding_l, padding_r,
                  mkldnn::padding_kind::zero);
            } else {
                deconv_desc_opt = mkldnn::deconvolution_forward::desc(
                  mkldnn::prop_kind::forward_inference,
                  mkldnn::algorithm::deconvolution_direct, deconv_input_md,
                  deconv_weight_md, deconv_output_md, strides, padding_l,
                  padding_r, mkldnn::padding_kind::zero);
            }
            auto deconv_pd = mkldnn::deconvolution_forward::primitive_desc(
              *deconv_desc_opt, engine);

            auto deconv_input_memory = input_memory;
            if(mkldnn::memory::primitive_desc(deconv_pd.dst_primitive_desc()) !=
               input_memory.get_primitive_desc()) {
                deconv_input_memory =
                  mkldnn::memory(deconv_pd.src_primitive_desc());
                temp_memory_list.push_back(deconv_input_memory);
                net.push_back(
                  mkldnn::reorder(input_memory, deconv_input_memory));
            }

            auto deconv_weight_memory = weight_memory;
            if(mkldnn::memory::primitive_desc(
                 deconv_pd.weights_primitive_desc()) !=
               weight_memory.get_primitive_desc()) {
                deconv_weight_memory =
                  mkldnn::memory(deconv_pd.weights_primitive_desc());
                temp_memory_list.push_back(deconv_weight_memory);
                net.push_back(
                  mkldnn::reorder(weight_memory, deconv_weight_memory));
            }

            manage_output_memory(
              net, output_name, mkldnn::memory::format::nchw,
              deconv_pd.dst_primitive_desc(), output_memory_table,
              required_output_table, temp_memory_list, engine,
              [&net, &deconv_input_memory, &deconv_weight_memory, &deconv_pd,
               &bias_memory_opt](auto& op_output_memory) {
                  if(bias_memory_opt) {
                      net.push_back(mkldnn::deconvolution_forward(
                        deconv_pd, deconv_input_memory, deconv_weight_memory,
                        *bias_memory_opt, op_output_memory));
                  } else {
                      net.push_back(mkldnn::deconvolution_forward(
                        deconv_pd, deconv_input_memory, deconv_weight_memory,
                        op_output_memory));
                  }
              });

            return std::make_tuple(net, output_memory_table, temp_memory_list,
                                   owned_array_list);
        }

    } // namespace mkldnn_backend
} // namespace menoh_impl
