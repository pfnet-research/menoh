#include <menoh/mkldnn/operator/gemm.hpp>

#include <tuple>

#include <mkldnn.hpp>

#include <menoh/utility.hpp>
#include <menoh/model_core.hpp>

#include <menoh/mkldnn/operator/common.hpp>
#include <menoh/mkldnn/utility.hpp>

namespace menoh_impl {
    namespace mkldnn_backend {

        primitive_factory_return_type make_gemm_primitive(
          menoh_impl::node const& node,
          std::unordered_map<std::string, array> const& parameter_table,
          std::unordered_map<std::string, mkldnn::memory> const&
            variable_memory_table,
          std::unordered_map<std::string, array> const& required_output_table,
          mkldnn::engine const& engine) {

            std::vector<mkldnn::primitive> net;
            std::unordered_map<std::string, mkldnn::memory> output_memory_table;
            std::unordered_map<std::string, array> output_table;
            std::vector<mkldnn::memory> temp_memory_list;
            std::vector<array> owned_array_list;

            auto alpha = optional_attribute_float(node, "alpha", 1.f);
            if(alpha != 1) {
                throw failed_to_configure_operator(
                  node.op_type, node.output_name_list.at(0),
                  "alpha of Gemm must be 1 but given: " +
                    std::to_string(alpha));
            }
            auto beta = optional_attribute_float(node, "beta", 1.f);
            if(beta != 1) {
                throw failed_to_configure_operator(
                  node.op_type, node.output_name_list.at(0),
                  "beta of Gemm must be 1 but given: " + std::to_string(alpha));
            }

            auto trans_a = optional_attribute_int(node, "transA", 0);
            if(trans_a) {
                throw failed_to_configure_operator(
                  node.op_type, node.output_name_list.at(0),
                  "transA of Gemm must be 0 but given: " +
                    std::to_string(alpha));
            }
            auto trans_b = optional_attribute_int(node, "transB", 0);
            if(!trans_b) {
                throw failed_to_configure_operator(
                  node.op_type, node.output_name_list.at(0),
                  "transB of Gemm must be 0 but given: " +
                    std::to_string(alpha));
            }

            auto const& input_memory =
              find_value(variable_memory_table, node.input_name_list.at(0));
            auto input_dims = extract_dims(input_memory);

            auto weight_format = input_dims.size() == 2
                                   ? mkldnn::memory::format::oi
                                   : mkldnn::memory::format::oihw;
            auto weight_arr =
              find_value(parameter_table, node.input_name_list.at(1));
            auto weight_dims = weight_arr.dims();
            if(weight_format == mkldnn::memory::format::oihw) {
                weight_dims = std::vector<int>{weight_dims.front()};
                weight_dims.insert(weight_dims.end(), input_dims.begin() + 1,
                                   input_dims.end());
            }
            auto weight_memory = array_to_memory_and_deal_ownership(
              weight_arr, weight_dims, weight_format, engine, temp_memory_list,
              owned_array_list);

            auto bias_memory = array_to_memory_and_deal_ownership(
              parameter_table.at(node.input_name_list.at(2)),
              mkldnn::memory::format::x, engine, temp_memory_list,
              owned_array_list);

            auto bias_dims = extract_dims(bias_memory);
            int output_size = weight_arr.dims()[0];
            if(output_size != bias_dims[0]) {
                throw failed_to_configure_operator(
                  node.op_type, node.output_name_list.at(0),
                  "dims[0] of input C must be equal to dims[0] of input B: "
                  "broadcast is not supported yet");
            }
            mkldnn::memory::dims output_dims{input_dims[0], output_size};

            auto const& output_name = node.output_name_list.at(0);

            auto gemm_input_md =
              mkldnn::memory::desc({input_dims}, mkldnn::memory::data_type::f32,
                                   mkldnn::memory::format::any);
            auto gemm_weight_md = mkldnn::memory::desc(
              {weight_dims}, mkldnn::memory::data_type::f32,
              mkldnn::memory::format::any);
            auto gemm_output_md = mkldnn::memory::desc(
              {output_dims}, mkldnn::memory::data_type::f32,
              mkldnn::memory::format::any);

            mkldnn::inner_product_forward::desc gemm_desc(
              mkldnn::prop_kind::forward_inference, gemm_input_md,
              gemm_weight_md, bias_memory.get_primitive_desc().desc(),
              gemm_output_md);
            auto gemm_pd =
              mkldnn::inner_product_forward::primitive_desc(gemm_desc, engine);

            auto gemm_input_memory = input_memory;
            if(mkldnn::memory::primitive_desc(gemm_pd.src_primitive_desc()) !=
               input_memory.get_primitive_desc()) {
                gemm_input_memory =
                  mkldnn::memory(gemm_pd.src_primitive_desc());
                temp_memory_list.push_back(gemm_input_memory);
                net.push_back(mkldnn::reorder(input_memory, gemm_input_memory));
            }

            auto gemm_weight_memory = weight_memory;
            if(mkldnn::memory::primitive_desc(
                 gemm_pd.weights_primitive_desc()) !=
               weight_memory.get_primitive_desc()) {
                gemm_weight_memory =
                  mkldnn::memory(gemm_pd.weights_primitive_desc());
                temp_memory_list.push_back(gemm_weight_memory);
                net.push_back(
                  mkldnn::reorder(weight_memory, gemm_weight_memory));
            }

            std::vector<std::pair<
              std::string, std::tuple<mkldnn::memory, mkldnn::memory::format>>>
              variable_memory_list;
            std::vector<std::pair<std::string, array>> output_name_and_arr_list;

            manage_output_memory(
              net, output_name, mkldnn::memory::format::nc,
              gemm_pd.dst_primitive_desc(), output_memory_table,
              required_output_table, temp_memory_list, engine,
              [&net, &gemm_input_memory, &gemm_weight_memory, &gemm_pd,
               &bias_memory](auto& op_output_memory) {
                  net.push_back(mkldnn::inner_product_forward(
                    gemm_pd, gemm_input_memory, gemm_weight_memory, bias_memory,
                    op_output_memory));
              });

            return std::make_tuple(net, output_memory_table, temp_memory_list,
                                   owned_array_list);
        }

    } // namespace mkldnn_backend
} // namespace menoh_impl
