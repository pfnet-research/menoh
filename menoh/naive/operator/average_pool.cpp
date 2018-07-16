#include <menoh/naive/operator/max_pool.hpp>

#include <numeric>

#include <menoh/model_core.hpp>
#include <menoh/optional.hpp>
#include <menoh/utility.hpp>

#include <menoh/naive/operator/index_conversion.hpp>

namespace menoh_impl {
    namespace naive_backend {

        computation_node_factory_return_type make_average_pool(
          int32_t i, std::vector<node> const& node_list,
          std::unordered_map<std::string, array> const& variable_table) {
            auto const& node = node_list.at(i);

            std::vector<int32_t> strides, kernel_shape, pads;
            std::tie(strides, kernel_shape, pads) =
              attributes_for_2d_data_processing(node);

            auto input =
              find_value(variable_table, node.input_name_list.front());
            auto batch_size = input.dims().at(0);
            auto channel_num = input.dims().at(1);
            auto input_height = input.dims().at(2);
            auto input_width = input.dims().at(3);

            auto output_dims = calc_2d_output_dims(input.dims(), channel_num,
                                                   kernel_shape, strides, pads);
            auto output_height = output_dims.at(2);
            auto output_width = output_dims.at(3);

            auto found = variable_table.find(node.output_name_list.at(0));
            optional<array> output_opt;
            if(found == variable_table.end()) {
                output_opt = array(dtype_t::float_, output_dims);
            } else {
                output_opt = found->second;
            }

            index_converter in_cvt(input.dims());
            index_converter out_cvt(output_dims);
            auto computation_node = [in_cvt, out_cvt, batch_size, channel_num,
                                     input_height, input_width, output_height,
                                     output_width, kernel_shape, strides, pads,
                                     input, output = *output_opt]() {
                for(int n = 0; n < batch_size; ++n) {
                    for(int c = 0; c < channel_num; ++c) {
                        for(int oy = 0; oy < output_height; ++oy) {
                            for(int ox = 0; ox < output_width; ++ox) {
                                float sum = 0.;
                                auto ky_begin =
                                  std::max(pads[0], oy * strides[0]);
                                auto ky_end =
                                  std::min(input_height + pads[0],
                                           oy * strides[0] + kernel_shape[0]);
                                for(auto ky = ky_begin; ky < ky_end; ++ky) {
                                    auto kx_begin =
                                      std::max(pads[1], ox * strides[1]);
                                    auto kx_end = std::min(
                                      input_width + pads[1],
                                      ox * strides[1] + kernel_shape[1]);
                                    for(auto kx = kx_begin; kx < kx_end; ++kx) {
                                        sum += fat(input,
                                                   in_cvt.indices_to_flat_index(
                                                     {n, c, ky - pads[0],
                                                      kx - pads[1]}));
                                    }
                                }
                                fat(output, out_cvt.indices_to_flat_index(
                                              {n, c, oy, ox})) =
                                  sum / (kernel_shape[0] * kernel_shape[1]);
                            }
                        }
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
