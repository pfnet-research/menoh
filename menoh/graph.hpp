#ifndef MENOH_GRAPH_HPP
#define MENOH_GRAPH_HPP

#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include <menoh/exception.hpp>
#include <menoh/node.hpp>

namespace menoh_impl {

    struct model_data;

    std::vector<node> extract_needed_node_list(
      std::vector<node> const& node_list,
      std::vector<std::string> const& required_output_name_list);

    std::set<std::string>
    extract_all_input_name_set(std::vector<node> const& node_list);

    std::set<std::string>
    extract_all_output_name_set(std::vector<node> const& node_list);

    class graph {
    public:
        graph() = default;
        explicit graph(std::vector<node>&& node_list);
        explicit graph(std::vector<node> const& node_list);

        auto const& node_list() const { return node_list_; }

    private:
        std::vector<node> node_list_;
    };

    graph make_graph(std::vector<node> node_list);

    template <typename Visitor>
    auto reconstruct_node_list(std::vector<node>& node_list, Visitor visitor) {
        auto node_list_for_loop = node_list;
        for(auto& node : node_list_for_loop) {
            visitor(node_list, node);
        }
    }

    void trim_node(std::vector<node>& node_list, menoh_impl::node const& node);

    void trim_dropout(std::vector<node>& node_list);
    void trim_reshape(std::vector<node>& node_list);

    class unsupported_operator_attribute : public exception {
    public:
        explicit unsupported_operator_attribute(
          std::string const& op_type, std::string const& first_output_name,
          std::string const& attribute_name, std::string const& actual_value,
          std::string const& valid_value)
          : exception(menoh_error_code_unsupported_operator_attribute,
                      "menoh unsupported operator attribute error: " + op_type +
                        " issuing \"" + first_output_name + "\": " +
                        attribute_name + " actual value: " + actual_value +
                        " valid value: " + valid_value) {}
    };
    class unsupported_operator : public exception {
    public:
        explicit unsupported_operator(std::string const& op_type)
          : exception(menoh_error_code_unsupported_operator,
                      "menoh unsupported operator error: " + op_type) {}
    };
    class dimension_mismatch : public exception {
    public:
        explicit dimension_mismatch(std::string const& op_type,
                                    std::string const& first_output_name,
                                    std::string const& message,
                                    std::string const& actual_value,
                                    std::string const& valid_value)
          : exception(menoh_error_code_dimension_mismatch,
                      "menoh dimension mismatch error: " + op_type +
                        " issuing \"" + first_output_name + "\": " + message +
                        " actual value: " + actual_value +
                        " valid value: " + valid_value) {}
    };

    std::unordered_map<std::string, std::vector<int32_t>> make_output_dims_table(
      menoh_impl::model_data const& model_data,
      std::vector<std::pair<std::string, std::vector<int32_t>>> const&
        input_dims_table);

} // namespace menoh_impl

#endif // MENOH_GRAPH_HPP
