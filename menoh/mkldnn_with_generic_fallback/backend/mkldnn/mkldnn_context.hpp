#ifndef MENOH_IMPL_MKLDNN_WITH_FALLBACK_BACKEND_BACKEND_MKLDNN_MKLDNN_CONTEXT_HPP
#define MENOH_IMPL_MKLDNN_WITH_FALLBACK_BACKEND_BACKEND_MKLDNN_MKLDNN_CONTEXT_HPP

#include <menoh/array.hpp>
#include <menoh/mkldnn/utility.hpp>
#include <menoh/model_core.hpp>

#include <menoh/mkldnn_with_generic_fallback/context.hpp>

#include <mkldnn.hpp>

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {
        namespace mkldnn_backend {

            class mkldnn_context final : public context {
            public:
                mkldnn_context();

            private:
                virtual optional<std::tuple<procedure, array>>
                do_try_to_get_variable(std::string const& name) override {
                    auto found = variable_memory_table_.find(name);
                    if(found == variable_memory_table_.end()) {
                        return nullopt;
                    }
                    auto dims =
                      ::menoh_impl::mkldnn_backend::extract_dims(found->second);
                    auto variable_memory = found->second; // mutable
                    procedure copy_proc(nullptr);         // mutable
                    if(dims.size() == 4) {
                        auto format = static_cast<mkldnn::memory::format>(
                          found->second.get_primitive_desc()
                            .desc()
                            .data.format);
                        if(format != mkldnn::memory::nchw) {
                            auto data_type =
                              static_cast<mkldnn::memory::data_type>(
                                found->second.get_primitive_desc()
                                  .desc()
                                  .data.data_type);
                            variable_memory = mkldnn::memory(
                              {{dims, data_type, mkldnn::memory::format::nchw},
                               engine_});
                            temp_memory_list_.push_back(variable_memory);
                            std::vector<mkldnn::primitive> primitives(
                              {mkldnn::reorder(found->second,
                                               variable_memory)});
                            copy_proc = [primitives]() {
                                mkldnn::stream(mkldnn::stream::kind::eager)
                                  .submit(primitives)
                                  .wait();
                            };
                        }
                    }
                    return std::make_tuple(
                      copy_proc, array(dtype_t::float_, dims,
                                       found->second.get_data_handle()));
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
                    return variable_memory_table_.at(name);
                }

                using procedure_factory = std::function<std::tuple<
                  std::vector<mkldnn::primitive>,
                  std::vector<std::pair<std::string, mkldnn::memory>>,
                  std::vector<mkldnn::memory>>(
                  int, std::vector<node> const&, std::vector<array> const&,
                  std::unordered_map<std::string, array> const&,
                  mkldnn::engine const&)>;

                mkldnn::engine engine_{mkldnn::engine::kind::cpu, 0}; // TODO
                std::unordered_map<std::string, mkldnn::memory>
                  variable_memory_table_;
                std::vector<mkldnn::memory> temp_memory_list_;
                std::unordered_map<std::string, procedure_factory>
                  procedure_factory_table_;
            };

        } // namespace mkldnn_backend
    }     // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_FALLBACK_BACKEND_BACKEND_MKLDNN_MKLDNN_CONTEXT_HPP
