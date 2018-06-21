#ifndef MENOH_DIMS_HPP
#define MENOH_DIMS_HPP

#include <string>
#include <vector>

namespace menoh_impl {

    std::size_t calc_total_size(std::vector<int> const& dims);

    std::vector<int> calc_2d_output_dims(std::vector<int> const& input_dims,
                                         int output_channel_num,
                                         std::vector<int> const& kernel_shape,
                                         std::vector<int> const& strides,
                                         std::vector<int> const& pads);

    std::vector<int> calc_2d_output_dims_for_conv_transpose(
      std::vector<int> const& input_dims, int output_channel_num,
      std::vector<int> const& kernel_shape, std::vector<int> const& strides,
      std::vector<int> const& pads);

    int
    get_batch_size_from_variable_dims(std::vector<int> const& variable_dims);

    int
    get_channel_num_from_variable_dims(std::vector<int> const& variable_dims);

    int get_output_channel_num_from_parameter_dims(
      std::vector<int> const& parameter_dims);

} // namespace menoh_impl

#endif // MENOH_DIMS_HPP
