#include <menoh/mkldnn/operator/sum.hpp>

#include <tuple>

#include <mkldnn.hpp>

#include <menoh/utility.hpp>
#include <menoh/model_core.hpp>

#include <menoh/mkldnn/operator/common.hpp>
#include <menoh/mkldnn/utility.hpp>

namespace menoh_impl {
    namespace mkldnn_backend {

        primitive_factory_return_type make_sum_primitive(
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

            std::vector<mkldnn::memory> input_memories;
            std::vector<mkldnn::memory::primitive_desc> input_pds;
            std::vector<std::vector<int>> input_dims;
            for(auto const& name : node.input_name_list) {
                input_memories.push_back(
                  find_value(variable_memory_table, name));
                input_pds.push_back(input_memories.back().get_primitive_desc());
                input_dims.push_back(extract_dims(input_memories.back()));
            }
            if(!std::all_of(input_dims.begin() + 1, input_dims.end(),
                            [&input_dims](auto const& e) {
                                return input_dims.front() == e;
                            })) {
                throw failed_to_configure_operator(
                  node.op_type, node.output_name_list.at(0),
                  "at least one of input has different dims: broadcast is not "
                  "supported yet");
            }

            auto const& output_name = node.output_name_list.at(0);
            auto output_dims = input_dims.front();

            auto sum_output_md = mkldnn::memory::desc(
              {output_dims}, mkldnn::memory::data_type::f32,
              mkldnn::memory::format::any);

            mkldnn::sum::primitive_desc sum_pd(
              sum_output_md, std::vector<float>(input_pds.size(), 1.f),
              input_pds);

            auto output_format = output_dims.size() == 2
                                   ? mkldnn::memory::format::nc
                                   : mkldnn::memory::format::nchw;

            std::vector<std::pair<
              std::string, std::tuple<mkldnn::memory, mkldnn::memory::format>>>
              variable_memory_list;
            std::vector<std::pair<std::string, array>> output_name_and_arr_list;

            manage_output_memory(
              net, output_name, output_format, sum_pd.dst_primitive_desc(),
              output_memory_table, required_output_table, temp_memory_list,
              engine, [&net, &input_memories, &sum_pd](auto& op_output_memory) {
                  std::vector<mkldnn::primitive::at> inputs;
                  inputs.reserve(input_memories.size());
                  std::transform(input_memories.begin(), input_memories.end(),
                                 std::back_inserter(inputs), [](auto& e) {
                                     return static_cast<mkldnn::primitive::at>(
                                       e);
                                 });
                  net.push_back(mkldnn::sum(sum_pd, inputs, op_output_memory));
              });

            return std::make_tuple(net, output_memory_table, temp_memory_list,
                                   std::vector<array>());
        }

    } // namespace mkldnn_backend
} // namespace menoh_impl
