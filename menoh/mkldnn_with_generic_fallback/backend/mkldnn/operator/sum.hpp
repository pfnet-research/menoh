#ifndef MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_SUM_HPP
#define MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_SUM_HPP

#include <algorithm>

#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/formatted_array.hpp>
#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/memory_cache.hpp>
#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/operator/output_management.hpp>
#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/procedure_factory.hpp>

#include <mkldnn.hpp>

#include <iostream>

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {
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
                    throw std::runtime_error(
                      "broadcasting is not supported yet");
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
    }     // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_SUM_HPP
