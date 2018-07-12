#include <menoh/naive/operator/concat.hpp>

#include <cmath>
#include <numeric>

#include <menoh/model_core.hpp>
#include <menoh/optional.hpp>
#include <menoh/utility.hpp>

#include <menoh/naive/operator/index_conversion.hpp>

namespace menoh_impl {
    namespace naive_backend {

        computation_node_factory_return_type make_concat(
          int32_t i, std::vector<node> const& node_list,
          std::unordered_map<std::string, array> const& variable_table) {
            auto const& node = node_list.at(i);

            auto axis = attribute_int(node, "axis");

            std::vector<array> inputs;
            for(int32_t i = 0; i < node.input_name_list.size(); ++i) {
                inputs.push_back(
                  find_value(variable_table, node.input_name_list.at(i)));
            }

            auto found = variable_table.find(node.output_name_list.front());
            optional<array> output_opt;
            if(found == variable_table.end()) {
                std::vector<int32_t> output_dims = inputs.front().dims();
                for(auto const& input : inputs) {
                    output_dims.at(axis) += input.dims().at(axis);
                }
                output_opt = array(dtype_t::float_, output_dims);
            } else {
                output_opt = found->second;
            }

            index_converter dst_cvt(output_opt->dims());

            auto computation_node = [axis, inputs, output = *output_opt,
                                     dst_cvt]() {
                int32_t offset = 0;
                for(auto const& input : inputs) {
                    index_converter cvt(input.dims());
                    for(int32_t i = 0; i < total_size(input); ++i) {
                        auto indices = cvt.flat_index_to_indices(i);
                        auto dst_indices = indices;
                        dst_indices[axis] += offset;
                        fat(output,
                            dst_cvt.indices_to_flat_index(dst_indices)) =
                          fat(input, i);
                    }
                    offset += input.dims()[axis];
                }
            };

            std::vector<std::pair<std::string, array>> outputs;
            if(found == variable_table.end()) {
                outputs.push_back(std::pair<std::string, array>(
                  node.output_name_list.at(0), *output_opt));
            }
            return std::make_tuple(computation_node, outputs);
        }

    } // namespace naive_backend
} // namespace menoh_impl
