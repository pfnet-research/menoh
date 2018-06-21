#ifndef MENOH_MKLDNN_OPERATOR_COMMON_HPP
#define MENOH_MKLDNN_OPERATOR_COMMON_HPP

#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <mkldnn.hpp>

#include <menoh/array.hpp>
#include <menoh/optional.hpp>

#include <menoh/mkldnn/utility.hpp>

namespace menoh_impl {
    namespace mkldnn_backend {

        inline auto array_to_memory_and_deal_ownership(
          array const& arr, std::vector<int> const& dims,
          mkldnn::memory::format format, mkldnn::engine const& engine,
          std::vector<mkldnn::memory>& temp_memory_list,
          std::vector<array>& owned_array_list) {
            if(arr.has_ownership()) {
                owned_array_list.push_back(arr);
            }
            auto mem = array_to_memory(arr, dims, format, engine);
            temp_memory_list.push_back(mem);
            return mem;
        }

        inline auto array_to_memory_and_deal_ownership(
          array const& arr, mkldnn::memory::format format,
          mkldnn::engine const& engine,
          std::vector<mkldnn::memory>& temp_memory_list,
          std::vector<array>& owned_array_list) {
            return array_to_memory_and_deal_ownership(arr, arr.dims(), format,
                                                      engine, temp_memory_list,
                                                      owned_array_list);
        }

        template <typename OpPrimitiveGenerator>
        auto manage_output_memory(
          std::vector<mkldnn::primitive>& net, std::string const& output_name,
          mkldnn::memory::format output_format,
          mkldnn::memory::primitive_desc const& output_pd,
          std::unordered_map<std::string, mkldnn::memory>& output_memory_table,
          std::unordered_map<std::string, array> const& output_table,
          std::vector<mkldnn::memory>& temp_memory_list,
          mkldnn::engine const& engine,
          OpPrimitiveGenerator op_primitive_generator) {

            menoh_impl::optional<mkldnn::memory> output_memory_opt;
            auto found = output_table.find(output_name);
            if(found != output_table.end()) {
                std::string name;
                array output_array;
                std::tie(name, output_array) = *found;
                (void)name;
                {
                    auto const& output_dims = output_array.dims();
                    for(decltype(output_dims.size()) i = 0;
                        i < output_dims.size(); ++i) {
                        assert(
                          output_dims[i] ==
                          const_cast<mkldnn::memory::primitive_desc&>(output_pd)
                            .desc()
                            .data.dims[i]);
                    }
                }
                output_memory_opt =
                  array_to_memory(output_array, output_format, engine);
            }

            auto op_output_memory =
              output_memory_opt.value_or(mkldnn::memory(output_pd));
            if(output_memory_opt && mkldnn::memory::primitive_desc(output_pd) !=
                                      output_memory_opt->get_primitive_desc()) {
                op_output_memory = mkldnn::memory(output_pd);
                temp_memory_list.push_back(*output_memory_opt);
            }

            op_primitive_generator(op_output_memory);

            if(output_memory_opt && op_output_memory != *output_memory_opt) {
                net.push_back(
                  mkldnn::reorder(op_output_memory, *output_memory_opt));
            }

            output_memory_table.insert({output_name, op_output_memory});
        }

        template <typename OpPrimitiveGenerator>
        auto manage_output_memory_inplace_if_possible(
          std::vector<mkldnn::primitive>& net, std::string const& input_name,
          mkldnn::memory const& input_memory, std::string const& output_name,
          mkldnn::memory::format output_format,
          mkldnn::memory::primitive_desc const& output_pd,
          std::unordered_map<std::string, mkldnn::memory>& output_memory_table,
          std::unordered_map<std::string, array> const& output_table,
          std::vector<mkldnn::memory>& temp_memory_list, int node_index,
          std::vector<menoh_impl::node> const& node_list,
          mkldnn::engine const& engine,
          OpPrimitiveGenerator op_primitive_generator) {
            assert(node_index < static_cast<int>(node_list.size()));

            menoh_impl::optional<mkldnn::memory> output_memory_opt;
            auto found = output_table.find(output_name);

            if(found != output_table.end()) { // if output is required
                std::string name;
                array output_array;
                std::tie(name, output_array) = *found;
                (void)name;
                {
                    auto const& output_dims = output_array.dims();
                    for(decltype(output_dims.size()) i = 0;
                        i < output_dims.size(); ++i) {
                        assert(
                          output_dims[i] ==
                          const_cast<mkldnn::memory::primitive_desc&>(output_pd)
                            .desc()
                            .data.dims[i]);
                    }
                }
                output_memory_opt =
                  array_to_memory(output_array, output_format, engine);
            } else if(output_table.find(input_name) == output_table.end() &&
                      all_of(
                        node_list.begin() + node_index + 1, node_list.end(),
                        [&input_name](auto const& following_node) {
                            return std::find(
                                     following_node.input_name_list.begin(),
                                     following_node.input_name_list.end(),
                                     input_name) ==
                                   following_node.input_name_list.end();
                        })) { // if input is not required output and not
                              // input of other nodes
                // TODO more check. There are more innplace-able input.
                assert(input_memory.get_primitive_desc() ==
                       mkldnn::memory::primitive_desc(output_pd));
                output_memory_opt = input_memory;
            }

            auto op_output_memory =
              output_memory_opt.value_or(mkldnn::memory(output_pd));
            if(output_memory_opt && mkldnn::memory::primitive_desc(output_pd) !=
                                      output_memory_opt->get_primitive_desc()) {
                op_output_memory = mkldnn::memory(output_pd);
                temp_memory_list.push_back(*output_memory_opt);
            }

            op_primitive_generator(op_output_memory);

            if(output_memory_opt && op_output_memory != *output_memory_opt) {
                net.push_back(
                  mkldnn::reorder(op_output_memory, *output_memory_opt));
            }

            output_memory_table.insert({output_name, op_output_memory});
        }

    } // namespace mkldnn_backend
} // namespace menoh_impl

#endif // MENOH_MKLDNN_OPERATOR_COMMON_HPP
