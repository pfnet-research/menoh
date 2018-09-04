#ifndef MENOH_NODE_HPP
#define MENOH_NODE_HPP

#include <string>
#include <unordered_map>
#include <vector>

#include <menoh/variant.hpp>

namespace menoh_impl {

    using attribute =
      menoh_impl::variant<int, float, std::vector<int>, std::vector<float>>;

    struct node {
        std::string op_type;
        std::vector<std::string> input_name_list;
        std::vector<std::string> output_name_list;
        std::unordered_map<std::string, attribute> attribute_table;
    };

    bool operator==(node const& a, node const& b);
    inline bool operator!=(node const& a, node const& b) { return !(a == b); }
    bool operator<(node const& a, node const& b);

    int optional_attribute_int(node const& n, std::string const& attr_name,
                               int default_value);
    int attribute_int(node const& n, std::string const& attr_name);
    float optional_attribute_float(node const& n, std::string const& attr_name,
                                   float default_value);
    float attribute_float(node const& n, std::string const& attr_name);
    std::vector<int>
    optional_attribute_ints(node const& n, std::string const& attr_name,
                            std::vector<int> const& default_value);
    std::vector<int> const& attribute_ints(node const& n,
                                           std::string const& attr_name);
    std::vector<float>
    optional_attribute_floats(node const& n, std::string const& attr_name,
                              std::vector<float> const& default_value);
    std::vector<float> const& attribute_floats(node const& n,
                                               std::string const& attr_name);

    std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
    attributes_for_2d_data_processing(node const& n);

    std::vector<int>
    calc_2d_output_dims(menoh_impl::node const& node, int output_channel_num,
                        std::unordered_map<std::string, std::vector<int>> const&
                          variable_dims_table);

    std::vector<int> calc_2d_output_dims_for_conv_transpose(
      menoh_impl::node const& node, int output_channel_num,
      std::unordered_map<std::string, std::vector<int>> const&
        variable_dims_table);

} // namespace menoh_impl

#endif // MENOH_NODE_HPP
