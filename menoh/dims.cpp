#include <menoh/dims.hpp>

#include <cassert>
#include <functional>
#include <numeric>

#include <menoh/exception.hpp>

namespace menoh_impl {

    std::size_t calc_total_size(std::vector<int32_t> const& dims) {
        return std::accumulate(dims.begin(), dims.end(),
                               static_cast<std::size_t>(1),
                               std::multiplies<>());
    }

    template <typename CalcLength>
    std::vector<int32_t> calc_2d_output_dims_impl(
      std::vector<int32_t> const& input_dims, int32_t output_channel_num,
      std::vector<int32_t> const& kernel_shape, std::vector<int32_t> const& strides,
      std::vector<int32_t> const& pads, CalcLength calc_length) {
        assert(kernel_shape.size() == 2);
        assert(strides.size() == 2);
        assert(pads.size() == 4);
        auto batch_size = input_dims[0];
        auto ih = input_dims[2];
        auto iw = input_dims[3];
        auto kh = kernel_shape[0];
        auto kw = kernel_shape[1];
        return std::vector<int32_t>(
          {batch_size, output_channel_num,
           calc_length(ih, kh, pads[0], pads[2], strides[0]),
           calc_length(iw, kw, pads[1], pads[3], strides[1])});
    }

    std::vector<int32_t> calc_2d_output_dims(std::vector<int32_t> const& input_dims,
                                         int32_t output_channel_num,
                                         std::vector<int32_t> const& kernel_shape,
                                         std::vector<int32_t> const& strides,
                                         std::vector<int32_t> const& pads) {
        assert(kernel_shape.size() == 2);
        assert(strides.size() == 2);
        assert(pads.size() == 4);
        auto calc_length = [](int32_t il, int32_t kl, int32_t p_begin, int32_t p_end, int32_t s) {
            return (il - kl + p_begin + p_end) / s + 1;
        };
        return calc_2d_output_dims_impl(input_dims, output_channel_num,
                                        kernel_shape, strides, pads,
                                        calc_length);
    }

    std::vector<int32_t> calc_2d_output_dims_for_conv_transpose(
      std::vector<int32_t> const& input_dims, int32_t output_channel_num,
      std::vector<int32_t> const& kernel_shape, std::vector<int32_t> const& strides,
      std::vector<int32_t> const& pads) {
        assert(kernel_shape.size() == 2);
        assert(strides.size() == 2);
        assert(pads.size() == 4);
        auto calc_length = [](int32_t il, int32_t kl, int32_t p_begin, int32_t p_end, int32_t s) {
            return s * (il - 1) + kl - p_begin - p_end;
        };
        return calc_2d_output_dims_impl(input_dims, output_channel_num,
                                        kernel_shape, strides, pads,
                                        calc_length);
    }

    int32_t
    get_batch_size_from_variable_dims(std::vector<int32_t> const& variable_dims) {
        return variable_dims.at(0); // n of nchw
    }

    int32_t
    get_channel_num_from_variable_dims(std::vector<int32_t> const& variable_dims) {
        return variable_dims.at(1); // c of nchw
    }

    int32_t get_output_channel_num_from_parameter_dims(
      std::vector<int32_t> const& parameter_dims) {
        return parameter_dims.at(0); // o of oihw
    }

} // namespace menoh_impl
