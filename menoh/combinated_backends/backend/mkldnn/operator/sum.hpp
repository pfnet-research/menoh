#ifndef MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_SUM_HPP
#define MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_SUM_HPP

#include <algorithm>

#include <menoh/graph.hpp> // for dimension_mismatch error

#include <menoh/combinated_backends/backend/mkldnn/formatted_array.hpp>
#include <menoh/combinated_backends/backend/mkldnn/memory_cache.hpp>
#include <menoh/combinated_backends/backend/mkldnn/operator/output_management.hpp>
#include <menoh/combinated_backends/backend/mkldnn/procedure_factory.hpp>

#include <mkldnn.hpp>

#include <sstream>

namespace menoh_impl {
    namespace combinated_backends {
        namespace mkldnn_backend {

            inline procedure_factory_return_type
            make_sum(MENOH_MKLDNN_CONTEXT_PROCEDURE_FACTORY_PARAMETER_LIST) {

                std::vector<mkldnn::primitive> primitives;

                auto output_dims =
                  output_formatted_array_list.at(0).array().dims();

                auto output_md = mkldnn::memory::desc(
                  {output_dims},
                  dtype_to_mkldnn_memory_data_type(
                    output_formatted_array_list.at(0).array().dtype()),
                  mkldnn::memory::format::any);

                std::vector<mkldnn::memory> input_memory_list;
                for(memory_cache& input_memory_cache :
                    input_memory_cache_list) {
                    input_memory_list.push_back(
                      input_memory_cache.get_data_memory());
                }
                assert(std::all_of(
                         input_memory_list.begin() + 1, input_memory_list.end(),
                         [&output_dims](auto const& e) {
                             return extract_dims(e).at(0) == output_dims.at(0);
                         }) &&
                       "invalid shape inference");
                if(!std::all_of(
                     input_memory_list.begin() + 1, input_memory_list.end(),
                     [first = input_memory_list.at(0)](auto const& e) {
                         return extract_dims(e) == extract_dims(first);
                     })) {

                    auto different_input_memory = std::find_if(
                      input_memory_list.begin(), input_memory_list.end(),
                      [first = input_memory_list.at(0)](auto const& e) {
                          return extract_dims(e) != extract_dims(first);
                      });
                    auto index = static_cast<int>(different_input_memory -
                                                  input_memory_list.begin());
                    auto different_dims = extract_dims(*different_input_memory);
                    auto dims_to_string = [](std::vector<int> const& dims) {
                        std::stringstream ss;
                        ss << "(";
                        for(auto d : dims) {
                            ss << d << " ";
                        }
                        ss << ")";
                        return ss.str();
                    };
                    throw dimension_mismatch(
                      node.op_type, node.output_name_list.front(),
                      "input[" + std::to_string(index) +
                        "] has different shape from "
                        "the input[0]'s. broadcast "
                        "is not supported yet",
                      dims_to_string(different_dims),
                      dims_to_string(extract_dims(input_memory_list.at(0))));
                }
                std::vector<mkldnn::memory::primitive_desc>
                  input_memory_pd_list;
                for(auto const& input_memory : input_memory_list) {
                    input_memory_pd_list.push_back(
                      input_memory.get_primitive_desc());
                }

                auto sum_pd = mkldnn::sum::primitive_desc(
                  output_md,
                  std::vector<float>(node.input_name_list.size(), 1.f),
                  input_memory_pd_list);

                auto output_memory_cache = manage_output(
                  output_formatted_array_list.at(0),
                  input_memory_list.at(0).get_primitive_desc(), engine,
                  primitives,
                  [&sum_pd,
                   &input_memory_list](mkldnn::memory const& output_memory) {
                      std::vector<mkldnn::primitive::at> inputs(
                        input_memory_list.begin(), input_memory_list.end());
                      return mkldnn::sum(sum_pd, inputs, output_memory);
                  });

                return procedure_factory_return_type{
                  primitives, {output_memory_cache}, {}};
            }

        } // namespace mkldnn_backend
    }     // namespace combinated_backends
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_SUM_HPP
