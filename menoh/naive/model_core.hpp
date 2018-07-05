#ifndef MENOH_NAIVE_MODEL_CORE_HPP
#define MENOH_NAIVE_MODEL_CORE_HPP

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include <menoh/array.hpp>
#include <menoh/model_core.hpp>
#include <menoh/model_data.hpp>

namespace menoh_impl {
    namespace naive_backend {

        class computation_node;

        class model_core final : public menoh_impl::model_core {
        public:
            model_core(
              std::unordered_map<std::string, array> const& input_table,
              std::unordered_map<std::string, array> const& output_table,
              menoh_impl::model_data const& model_data);

        private:
            virtual void do_run() override;

            std::vector<std::function<void()>> computation_node_list_;
            std::unordered_map<std::string, array> variable_table_;
        };

    } // namespace naive_backend
} // namespace menoh_impl

#endif // MENOH_NAIVE_MODEL_CORE_HPP
