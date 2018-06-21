#include <menoh/model_core.hpp>
#include <menoh/model_core_factory.hpp>

#include <menoh/mkldnn/model_core.hpp>

namespace menoh_impl {

    std::unique_ptr<menoh_impl::model_core>
    make_model_core(std::unordered_map<std::string, array> const& input_table,
                    std::unordered_map<std::string, array> const& output_table,
                    menoh_impl::model_data const& model_data,
                    std::string const& backend_name,
                    backend_config const& config) {
        if(backend_name == "mkldnn") {
            return std::make_unique<mkldnn_backend::model_core>(
              mkldnn_backend::make_model_core(input_table, output_table,
                                              model_data, config));
        }
        throw invalid_backend_name(backend_name);
    }

} // namespace menoh_impl
