#ifndef MENOH_MODEL_HPP
#define MENOH_MODEL_HPP

#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <menoh/array.hpp>
#include <menoh/backend_config.hpp>
#include <menoh/model_core.hpp>
#include <menoh/model_core_factory.hpp>
#include <menoh/model_data.hpp>

namespace menoh_impl {

    class model {
    public:
        model(
          std::vector<
            std::tuple<std::string, dtype_t, std::vector<int32_t>, void*>> const&
            input_name_and_dtype_and_dims_and_data_handle_list,
          std::vector<std::tuple<std::string, dtype_t, void*>> const&
            required_output_name_and_dtype_and_data_handle_list,
          menoh_impl::model_data const& model_data, std::string const& backend_name,
          backend_config const& config = backend_config());

        array const& input(std::string const& name) const;

        array const& output(std::string const& name) const;

        void run();

    private:
        std::unordered_map<std::string, array> input_table_;
        std::unordered_map<std::string, array> output_table_;
        std::unique_ptr<model_core> model_;
    };

} // namespace menoh_impl
#endif // MENOH_MODEL_HPP
