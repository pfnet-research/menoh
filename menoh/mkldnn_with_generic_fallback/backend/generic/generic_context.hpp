#ifndef MENOH_MKLDNN_WITH_FALLBACK_BACKEND_GENERIC_GENERIC_CONTEXT_HPP
#define MENOH_MKLDNN_WITH_FALLBACK_BACKEND_GENERIC_GENERIC_CONTEXT_HPP

#include <menoh/mkldnn_with_generic_fallback/context.hpp>

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {
        namespace generic_backend {

            class generic_context final : public context {
            public:
                generic_context();

            private:
                virtual optional<std::tuple<procedure, array>>
                do_try_to_get_variable(std::string const& name) override {
                    auto varriable_iter = variable_table_.find(name);
                    if(varriable_iter == variable_table_.end()) {
                        return nullopt;
                    }
                    return std::make_tuple(procedure(nullptr),
                                           variable_table_.at(name));
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
                    return variable_table_.at(name);
                }

                using procedure_factory = std::function<procedure(
                  node const&, // node
                  std::vector<array> const&, // input list
                  std::vector<array> const&  // output list
                  )>;
                optional<std::function<void()>>
                try_to_get_input_from_common_table(
                  std::string const& input_name,
                  std::unordered_map<std::string, array> const& common_table);

                std::unordered_map<std::string, array> variable_table_;
                std::unordered_map<std::string, procedure_factory>
                  procedure_factory_table_;
            };

        } // namespace generic_backend
    }     // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl

#endif // MENOH_MKLDNN_WITH_FALLBACK_BACKEND_GENERIC_GENERIC_CONTEXT_HPP
