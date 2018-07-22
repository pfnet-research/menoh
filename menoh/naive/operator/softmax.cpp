#include <menoh/naive/operator/softmax.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

#include <menoh/model_core.hpp>
#include <menoh/optional.hpp>
#include <menoh/utility.hpp>

#include <menoh/naive/operator/index_conversion.hpp>

namespace menoh_impl {
    namespace naive_backend {

        computation_node_factory_return_type make_softmax(
          int32_t i, std::vector<node> const& node_list,
          std::unordered_map<std::string, array> const& variable_table) {
            auto const& node = node_list.at(i);

            auto axis = optional_attribute_int(node, "axis", 1);

            auto const& x_arr =
              find_value(variable_table, node.input_name_list.at(0));
            auto batch_size =
              std::accumulate(x_arr.dims().begin(), x_arr.dims().begin() + axis,
                              1, std::multiplies<>());
            auto vec_size =
              std::accumulate(x_arr.dims().begin() + axis, x_arr.dims().end(),
                              1, std::multiplies<>());
            array reshaped_x_arr(
              x_arr.dtype(), {batch_size, vec_size}); // TODO check inplace-able

            auto found = variable_table.find(node.output_name_list.at(0));
            optional<array> output_opt;
            if(found == variable_table.end()) {
                output_opt = array(dtype_t::float_,
                                   x_arr.dims()); // TODO check inplace-able
            } else {
                output_opt =
                  found->second; // output is required so not inplace-able
            }

            index_converter in_cvt(reshaped_x_arr.dims());
            auto computation_node = [in_cvt, batch_size, vec_size, x_arr,
                                     reshaped_x_arr, output = *output_opt]() {
                for(decltype(batch_size) b = 0; b < batch_size; ++b) {
                    float exp_sum = 0.;
                    auto max_elem =
                      *std::max_element(fbegin(x_arr) + b * vec_size,
                                        fbegin(x_arr) + (b + 1) * vec_size);
                    for(decltype(vec_size) v = 0; v < vec_size; ++v) {
                        auto i = in_cvt.indices_to_flat_index({b, v});
                        auto e = std::exp(fat(x_arr, i) - max_elem);
                        fat(reshaped_x_arr, i) = e;
                        exp_sum += e;
                    }
                    for(decltype(vec_size) v = 0; v < vec_size; ++v) {
                        auto i = in_cvt.indices_to_flat_index({b, v});
                        fat(output, i) = fat(reshaped_x_arr, i) / exp_sum;
                    }
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
