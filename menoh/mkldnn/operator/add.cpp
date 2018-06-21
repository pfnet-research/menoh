#include <menoh/mkldnn/operator/add.hpp>

#include <tuple>

#include <mkldnn.hpp>

#include <menoh/utility.hpp>
#include <menoh/model_core.hpp>

#include <menoh/mkldnn/operator/common.hpp>
#include <menoh/mkldnn/utility.hpp>

namespace menoh_impl {
    namespace mkldnn_backend {

        primitive_factory_return_type make_add_primitive(
          menoh_impl::node const& node,
          std::unordered_map<std::string, array> const& /*parameter_table*/,
          std::unordered_map<std::string, mkldnn::memory> const&
            variable_memory_table,
          std::unordered_map<std::string, array> const& required_output_table,
          mkldnn::engine const& engine) {

            std::vector<mkldnn::primitive> net;
            std::unordered_map<std::string, mkldnn::memory> output_memory_table;
            std::unordered_map<std::string, array> output_table;
            std::vector<mkldnn::memory> temp_memory_list;

            auto const& input_a_memory =
              find_value(variable_memory_table, node.input_name_list.at(0));
            auto const& input_b_memory =
              find_value(variable_memory_table, node.input_name_list.at(1));
            auto input_a_dims = extract_dims(input_a_memory);
            auto input_b_dims = extract_dims(input_b_memory);
            if(input_a_dims != input_b_dims) {
                throw failed_to_configure_operator(
                  node.op_type, node.output_name_list.at(0),
                  "input A and input B has different dims: broadcast is not supported yet");
            }

            auto const& output_name = node.output_name_list.at(0);
            auto output_dims = input_a_dims;

            auto add_output_md = mkldnn::memory::desc(
              {output_dims}, mkldnn::memory::data_type::f32,
              mkldnn::memory::format::any);

            mkldnn::sum::primitive_desc add_pd(
              add_output_md, std::vector<float>{1.0f, 1.0f},
              {input_a_memory.get_primitive_desc(),
               input_b_memory.get_primitive_desc()});

            auto output_format = input_a_dims.size() == 2
                                   ? mkldnn::memory::format::nc
                                   : mkldnn::memory::format::nchw;

            std::vector<std::pair<
              std::string, std::tuple<mkldnn::memory, mkldnn::memory::format>>>
              variable_memory_list;
            std::vector<std::pair<std::string, array>> output_name_and_arr_list;

            manage_output_memory(
              net, output_name, output_format, add_pd.dst_primitive_desc(),
              output_memory_table, required_output_table, temp_memory_list,
              engine,
              [&net, &input_a_memory, &input_b_memory,
               &add_pd](auto& op_output_memory) {
                  std::vector<mkldnn::primitive::at> inputs{
                    static_cast<mkldnn::primitive::at>(input_a_memory),
                    static_cast<mkldnn::primitive::at>(input_b_memory)};
                  net.push_back(mkldnn::sum(add_pd, inputs, op_output_memory));
              });

            return std::make_tuple(net, output_memory_table, temp_memory_list,
                                   std::vector<array>());
        }

    } // namespace mkldnn_backend
} // namespace menoh_impl
