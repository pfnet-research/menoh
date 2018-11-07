#ifndef MENOH_IMPL_MKLDNN_WITH_FALLBACK_BACKEND_BACKEND_MKLDNN_MKLDNN_CONTEXT_HPP
#define MENOH_IMPL_MKLDNN_WITH_FALLBACK_BACKEND_BACKEND_MKLDNN_MKLDNN_CONTEXT_HPP

#include <string>
#include <unordered_map>
#include <vector>

#include <mkldnn.hpp>

#include <menoh/any.hpp>
#include <menoh/array.hpp>
#include <menoh/mkldnn/utility.hpp>
#include <menoh/model_core.hpp>

#include <menoh/mkldnn_with_generic_fallback/context.hpp>

#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/formatted_array.hpp>
#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/memory_cache.hpp>
#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/procedure_factory.hpp>

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {
        namespace mkldnn_backend {

            class mkldnn_context final : public context {
            public:
                mkldnn_context();

            private:
                virtual optional<std::tuple<procedure, array>>
                do_try_to_get_variable(std::string const& name) override {
                    auto found = variable_memory_cache_table_.find(name);
                    if(found == variable_memory_cache_table_.end()) {
                        return nullopt;
                    }
                    auto& variable_memory_cache = found->second;
                    std::vector<mkldnn::primitive> primitives;
                    auto variable_memory =
                      get_memory(variable_memory_cache,
                                 ndims_to_data_memory_format(
                                   variable_memory_cache.dims().size()),
                                 primitives);
                    assert(extract_format(variable_memory) ==
                             mkldnn::memory::format::nchw ||
                           extract_format(variable_memory) ==
                             mkldnn::memory::format::nc);
                    procedure copy_proc =
                      primitives.empty()
                        ? procedure(nullptr)
                        : procedure([primitives] {
                              mkldnn::stream(mkldnn::stream::kind::eager)
                                .submit(primitives)
                                .wait();
                          });
                    return std::make_tuple(
                      copy_proc, array(mkldnn_memory_data_type_to_dtype(
                                         extract_data_type(variable_memory)),
                                       extract_dims(variable_memory),
                                       variable_memory.get_data_handle()));
                }

                virtual optional<std::tuple<std::vector<procedure>, int>>
                do_process_node_list(
                  std::string const& context_name, int current_index,
                  std::vector<node> const& node_list,
                  std::unordered_map<std::string, array> const&
                    common_parameter_table,
                  std::unordered_map<std::string, array> const&
                    common_input_table,
                  std::unordered_map<std::string, array> const&
                    required_output_table,
                  std::unordered_map<std::string, array_profile> const&
                    output_profile_table,
                  std::vector<
                    std::pair<std::string, std::unique_ptr<context>>> const&
                    context_list,
                  logger_handle logger) override;

                // for specialized optimization across backends
                virtual any
                do_take_variable_handle(std::string const& name) override {
                    return variable_memory_cache_table_.at(name);
                }

                mkldnn::engine engine_{mkldnn::engine::kind::cpu, 0}; // TODO
                std::vector<array> allocated_array_list_;
                std::unordered_map<std::string, memory_cache>
                  variable_memory_cache_table_;
                std::unordered_map<std::string, memory_cache>
                  temp_memory_cache_table_;
                std::unordered_map<std::string, procedure_factory>
                  procedure_factory_table_;
            };

        } // namespace mkldnn_backend
    }     // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_FALLBACK_BACKEND_BACKEND_MKLDNN_MKLDNN_CONTEXT_HPP
