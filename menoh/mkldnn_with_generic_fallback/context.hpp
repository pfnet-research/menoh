#ifndef MENOH_MKLDNN_WITH_FALLBACK_CONTEXT_HPP
#define MENOH_MKLDNN_WITH_FALLBACK_CONTEXT_HPP

#include <iosfwd>

#include <menoh/any.hpp>
#include <menoh/array.hpp>
#include <menoh/node.hpp>
#include <menoh/optional.hpp>

#include <menoh/mkldnn_with_generic_fallback/logger.hpp>
#include <menoh/mkldnn_with_generic_fallback/procedure.hpp>

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {

        class context {
        public:
            optional<std::tuple<procedure, array>>
            try_to_get_variable(std::string const& name) {
                return do_try_to_get_variable(name);
            }

            optional<std::tuple<std::vector<procedure>, int>> process_node_list(
              std::string const& context_name, int current_index,
              std::vector<node> const& node_list,
              std::unordered_map<std::string, array> const&
                common_parameter_table,
              std::unordered_map<std::string, array> const& common_input_table,
              std::unordered_map<std::string, array> const&
                required_output_table,
              std::unordered_map<std::string, array_profile> const&
                output_profile_table,
              std::vector<
                std::pair<std::string, std::unique_ptr<context>>> const&
                context_list,
              logger_handle logger) {
                return do_process_node_list(
                  context_name, current_index, node_list,
                  common_parameter_table, common_input_table,
                  required_output_table, output_profile_table, context_list,
                  logger);
            }

            // for specialized optimization across backends
            any take_variable_handle(std::string const& name) {
                return do_take_variable_handle(name);
            }

        private:
            virtual optional<std::tuple<procedure, array>>
            do_try_to_get_variable(std::string const& name) = 0;

            virtual optional<std::tuple<std::vector<procedure>, int>>
            do_process_node_list(
              std::string const& context_name, int current_index,
              std::vector<node> const& node_list,
              std::unordered_map<std::string, array> const&
                common_parameter_table,
              std::unordered_map<std::string, array> const& common_input_table,
              std::unordered_map<std::string, array> const&
                required_output_table,
              std::unordered_map<std::string, array_profile> const&
                output_profile_table,
              std::vector<
                std::pair<std::string, std::unique_ptr<context>>> const&
                context_list,
              logger_handle logger) = 0;

            // for specialized optimization across backends
            virtual any do_take_variable_handle(std::string const& name) = 0;
        };

    } // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl

#endif // MENOH_MKLDNN_WITH_FALLBACK_CONTEXT_HPP
