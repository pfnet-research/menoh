#ifndef MENOH_DIMS_HPP
#define MENOH_DIMS_HPP

#include <string>
#include <vector>

namespace menoh_impl {

    std::size_t calc_total_size(std::vector<int32_t> const& dims);

    std::vector<int32_t> calc_2d_output_dims(std::vector<int32_t> const& input_dims,
                                         int32_t output_channel_num,
                                         std::vector<int32_t> const& kernel_shape,
                                         std::vector<int32_t> const& strides,
                                         std::vector<int32_t> const& pads);

    std::vector<int32_t> calc_2d_output_dims_for_conv_transpose(
      std::vector<int32_t> const& input_dims, int32_t output_channel_num,
      std::vector<int32_t> const& kernel_shape, std::vector<int32_t> const& strides,
      std::vector<int32_t> const& pads);

    int32_t
    get_batch_size_from_variable_dims(std::vector<int32_t> const& variable_dims);

    int32_t
    get_channel_num_from_variable_dims(std::vector<int32_t> const& variable_dims);

    int32_t get_output_channel_num_from_parameter_dims(
      std::vector<int32_t> const& parameter_dims);

} // namespace menoh_impl

#endif // MENOH_DIMS_HPP
