#ifndef MENOH_TENSORRT_MODEL_CORE_HPP
#define MENOH_TENSORRT_MODEL_CORE_HPP

#include <menoh/backend_config.hpp>

#include <menoh/tensorrt/Inference.hpp>

namespace menoh_impl {
    namespace tensorrt_backend {

        class model_core final : public menoh_impl::model_core {
        public:
            model_core(
              std::unordered_map<std::string, array> const& input_table,
              std::unordered_map<std::string, array> const& output_table,
              menoh_impl::model_data const& model_data, config const& config);

        private:
            virtual void do_run() override;

            Inference m_inference;
        };

        model_core make_model_core(
          std::unordered_map<std::string, array> const& input_table,
          std::unordered_map<std::string, array> const& output_table,
          menoh_impl::model_data const& model_data,
          backend_config const& config = backend_config());

    } // namespace tensorrt_backend
} // namespace menoh_impl
#endif // MENOH_TENSORRT_MODEL_CORE_HPP
