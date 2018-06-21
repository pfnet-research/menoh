#include <menoh/dims.hpp>

#include <cassert>
#include <functional>
#include <numeric>

#include <menoh/exception.hpp>

namespace menoh_impl {

    std::size_t calc_total_size(std::vector<int> const& dims) {
        return std::accumulate(dims.begin(), dims.end(),
                               static_cast<std::size_t>(1),
                               std::multiplies<>());
    }

    template <typename CalcLength>
    std::vector<int> calc_2d_output_dims_impl(
      std::vector<int> const& input_dims, int output_channel_num,
      std::vector<int> const& kernel_shape, std::vector<int> const& strides,
      std::vector<int> const& pads, CalcLength calc_length) {
        assert(kernel_shape.size() == 2);
        assert(strides.size() == 2);
        assert(pads.size() == 4);
        auto batch_size = input_dims[0];
        auto ih = input_dims[2];
        auto iw = input_dims[3];
        auto kh = kernel_shape[0];
        auto kw = kernel_shape[1];
        return std::vector<int>(
          {batch_size, output_channel_num,
           calc_length(ih, kh, pads[0], pads[2], strides[0]),
           calc_length(iw, kw, pads[1], pads[3], strides[1])});
    }

    std::vector<int> calc_2d_output_dims(std::vector<int> const& input_dims,
                                         int output_channel_num,
                                         std::vector<int> const& kernel_shape,
                                         std::vector<int> const& strides,
                                         std::vector<int> const& pads) {
        assert(kernel_shape.size() == 2);
        assert(strides.size() == 2);
        assert(pads.size() == 4);
        auto calc_length = [](int il, int kl, int p_begin, int p_end, int s) {
            return (il - kl + p_begin + p_end) / s + 1;
        };
        return calc_2d_output_dims_impl(input_dims, output_channel_num,
                                        kernel_shape, strides, pads,
                                        calc_length);
    }

    std::vector<int> calc_2d_output_dims_for_conv_transpose(
      std::vector<int> const& input_dims, int output_channel_num,
      std::vector<int> const& kernel_shape, std::vector<int> const& strides,
      std::vector<int> const& pads) {
        assert(kernel_shape.size() == 2);
        assert(strides.size() == 2);
        assert(pads.size() == 4);
        auto calc_length = [](int il, int kl, int p_begin, int p_end, int s) {
            return s * (il - 1) + kl - p_begin - p_end;
        };
        return calc_2d_output_dims_impl(input_dims, output_channel_num,
                                        kernel_shape, strides, pads,
                                        calc_length);
    }

    int
    get_batch_size_from_variable_dims(std::vector<int> const& variable_dims) {
        return variable_dims.at(0); // n of nchw
    }

    int
    get_channel_num_from_variable_dims(std::vector<int> const& variable_dims) {
        return variable_dims.at(1); // c of nchw
    }

    int get_output_channel_num_from_parameter_dims(
      std::vector<int> const& parameter_dims) {
        return parameter_dims.at(0); // o of oihw
    }

} // namespace menoh_impl
