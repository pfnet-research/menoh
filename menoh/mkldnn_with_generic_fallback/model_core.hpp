#ifndef MENOH_MKLDNN_WITH_FALLBACK_MODEL_CORE_HPP
#define MENOH_MKLDNN_WITH_FALLBACK_MODEL_CORE_HPP

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include <menoh/array.hpp>
#include <menoh/model_core.hpp>
#include <menoh/model_data.hpp>
#include <menoh/backend_config.hpp>

#include <menoh/mkldnn_with_generic_fallback/context.hpp>
#include <menoh/mkldnn_with_generic_fallback/logger.hpp>

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {

        class model_core final : public menoh_impl::model_core {
        public:
            model_core(
              std::vector<std::pair<std::string, std::unique_ptr<context>>>
                context_list,
              std::unordered_map<std::string, array> const& input_table,
              std::unordered_map<std::string, array> const& output_table,
              std::unordered_map<std::string, array_profile> const&
                output_profile_table,
              menoh_impl::model_data const& model_data, backend_config const& config);

        private:
            virtual void do_run() override;

            std::unordered_map<std::string, array> common_parameter_table_;
            std::unordered_map<std::string, array> common_input_table_;
            std::unordered_map<std::string, array> required_output_table_;
            std::vector<std::pair<std::string, std::unique_ptr<context>>>
              context_list_;
            logger logger_;

            std::vector<procedure> procedure_list_;
        };

    } // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl

#endif // MENOH_MKLDNN_WITH_FALLBACK_MODEL_CORE_HPP
