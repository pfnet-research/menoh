#include <menoh/node.hpp>

#include <functional>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <menoh/dims.hpp>
#include <menoh/utility.hpp>
#include <menoh/variant.hpp>

namespace menoh_impl {

    bool operator==(node const& a, node const& b) {
        return a.op_type == b.op_type &&
               a.input_name_list == b.input_name_list &&
               a.output_name_list == b.output_name_list &&
               a.attribute_table == b.attribute_table;
    }

    bool operator<(node const& a, node const& b) {
        return a.output_name_list < b.output_name_list;
    }

    template <typename T>
    T optional_attribute(node const& n, std::string const& attr_name,
                         T default_value) {
        auto found = n.attribute_table.find(attr_name);
        if(found == n.attribute_table.end()) {
            return default_value;
        }
        return menoh_impl::get<T>(found->second);
    }

    int optional_attribute_int(node const& n, std::string const& attr_name,
                               int default_value) {
        return optional_attribute<int>(n, attr_name, default_value);
    }

    float optional_attribute_float(node const& n, std::string const& attr_name,
                                   float default_value) {
        return optional_attribute<float>(n, attr_name, default_value);
    }

    int attribute_int(node const& n, std::string const& attr_name) {
        return menoh_impl::get<int>(find_value(n.attribute_table, attr_name));
    }

    float attribute_float(node const& n, std::string const& attr_name) {
        return menoh_impl::get<float>(find_value(n.attribute_table, attr_name));
    }

    array const& attribute_tensor(node const& n, std::string const& attr_name) {
        return menoh_impl::get<array>(find_value(n.attribute_table, attr_name));
    }

    std::vector<int>
    optional_attribute_ints(node const& n, std::string const& attr_name,
                            std::vector<int> const& default_value) {
        return optional_attribute<std::vector<int>>(n, attr_name,
                                                    default_value);
    }
    std::vector<int> const& attribute_ints(node const& n,
                                           std::string const& attr_name) {
        return menoh_impl::get<std::vector<int>>(
          find_value(n.attribute_table, attr_name));
    }
    std::vector<float>
    optional_attribute_floats(node const& n, std::string const& attr_name,
                              std::vector<float> const& default_value) {
        return optional_attribute<std::vector<float>>(n, attr_name,
                                                      default_value);
    }
    std::vector<float> const& attribute_floats(node const& n,
                                               std::string const& attr_name) {
        return menoh_impl::get<std::vector<float>>(
          find_value(n.attribute_table, attr_name));
    }
    std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
    attributes_for_2d_data_processing(node const& n) {
        // Workaround for onnx-chainer. see
        // https://github.com/chainer/onnx-chainer/pull/11
        auto pads = optional_attribute_ints(n, "pads", {0, 0});
        if(pads.size() == 2) {
            pads = std::vector<int>{pads[0], pads[1], pads[0], pads[1]};
        }
        return std::make_tuple(optional_attribute_ints(n, "strides", {1, 1}),
                               attribute_ints(n, "kernel_shape"), pads);
    }

    std::vector<int>
    calc_2d_output_dims(menoh_impl::node const& node, int output_channel_num,
                        std::unordered_map<std::string, std::vector<int>> const&
                          variable_dims_table) {
        std::vector<int> strides, kernel_shape, pads;
        std::tie(strides, kernel_shape, pads) =
          attributes_for_2d_data_processing(node);
        return calc_2d_output_dims(
          find_value(variable_dims_table, node.input_name_list.at(0)),
          output_channel_num, kernel_shape, strides, pads);
    }

    std::vector<int> calc_2d_output_dims_for_conv_transpose(
      menoh_impl::node const& node, int output_channel_num,
      std::unordered_map<std::string, std::vector<int>> const&
        variable_dims_table) {
        std::vector<int> strides, kernel_shape, pads;
        std::tie(strides, kernel_shape, pads) =
          attributes_for_2d_data_processing(node);
        return calc_2d_output_dims_for_conv_transpose(
          find_value(variable_dims_table, node.input_name_list.at(0)),
          output_channel_num, kernel_shape, strides, pads);
    }

} // namespace menoh_impl
