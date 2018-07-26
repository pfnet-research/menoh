#include <menoh/mkldnn/operator/concat.hpp>

#include <tuple>

#include <mkldnn.hpp>

#include <menoh/utility.hpp>

#include <menoh/mkldnn/operator/common.hpp>
#include <menoh/mkldnn/utility.hpp>

namespace menoh_impl {
    namespace mkldnn_backend {

        primitive_factory_return_type make_concat_primitive(
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

            auto axis = attribute_int(node, "axis");

            std::vector<mkldnn::memory> input_memories;
            for(auto const& input_name : node.input_name_list) {
                input_memories.push_back(
                  find_value(variable_memory_table, input_name));
            }
            std::vector<mkldnn::memory::primitive_desc> input_memory_pds;
            for(auto const& input_memory : input_memories) {
                input_memory_pds.push_back(input_memory.get_primitive_desc());
            }

            auto const& output_name = node.output_name_list.at(0);
            auto output_dims = extract_dims(input_memories.front());
            output_dims[axis] = 0;
            for(auto const& input_memory : input_memories) {
                output_dims.at(axis) += extract_dims(input_memory).at(axis);
            }

            auto concat_output_md = mkldnn::memory::desc(
              {output_dims}, mkldnn::memory::data_type::f32,
              mkldnn::memory::format::any);

            mkldnn::concat::primitive_desc concat_pd(concat_output_md, axis,
                                                     input_memory_pds);

            auto output_format =
              extract_dims(input_memories.front()).size() == 2
                ? mkldnn::memory::format::nc
                : mkldnn::memory::format::nchw;

            std::vector<std::pair<
              std::string, std::tuple<mkldnn::memory, mkldnn::memory::format>>>
              variable_memory_list;
            std::vector<std::pair<std::string, array>> output_name_and_arr_list;

            manage_output_memory(
              net, output_name, output_format, concat_pd.dst_primitive_desc(),
              output_memory_table, required_output_table, temp_memory_list,
              engine,
              [&net, &input_memories,
               &concat_pd](mkldnn::memory& op_output_memory) {
                  std::vector<mkldnn::primitive::at> inputs;
                  for(auto const& input_memory : input_memories) {
                      inputs.push_back(
                        static_cast<mkldnn::primitive::at>(input_memory));
                  }
                  net.push_back(
                    mkldnn::concat(concat_pd, inputs, op_output_memory));
              });

            return std::make_tuple(net, output_memory_table, temp_memory_list,
                                   std::vector<array>());
        }

    } // namespace mkldnn_backend
} // namespace menoh_impl
