#ifndef MENOH_MKLDNN_MODEL_CORE_HPP
#define MENOH_MKLDNN_MODEL_CORE_HPP

#include <string>
#include <unordered_map>
#include <vector>

#include <mkldnn.hpp>

#include <menoh/array.hpp>
#include <menoh/backend_config.hpp>
#include <menoh/model_core.hpp>
#include <menoh/model_data.hpp>

namespace menoh_impl {
    namespace mkldnn_backend {

        class model_core final : public menoh_impl::model_core {
        public:
            model_core(
              std::unordered_map<std::string, array> const& input_table,
              std::unordered_map<std::string, array> const& output_table,
              menoh_impl::model_data const& model_data, mkldnn::engine const& engine);

        private:
            virtual void do_run() override;

            mkldnn::engine engine_;
            std::vector<mkldnn::primitive> nets_;
            std::unordered_map<std::string, mkldnn::memory>
              variable_memory_table_;
            std::vector<mkldnn::memory> temp_memory_list_;
            std::vector<array> owned_array_list_;
        };

        model_core make_model_core(
          std::unordered_map<std::string, array> const& input_table,
          std::unordered_map<std::string, array> const& output_table,
          menoh_impl::model_data const& model_data,
          backend_config const& config = backend_config());

    } // namespace mkldnn_backend
} // namespace menoh_impl
#endif // MENOH_MKLDNN_MODEL_CORE_HPP
