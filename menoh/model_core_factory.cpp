#include <menoh/model_core.hpp>
#include <menoh/model_core_factory.hpp>

#if ENABLE_MKLDNN
#include <menoh/mkldnn/model_core.hpp>
#endif

#if ENABLE_ARMNN
#include <menoh/arm/model_core.hpp>
#endif

namespace menoh_impl {

    std::unique_ptr<menoh_impl::model_core>
    make_model_core(std::unordered_map<std::string, array> const& input_table,
                    std::unordered_map<std::string, array> const& output_table,
                    menoh_impl::model_data const& model_data,
                    std::string const& backend_name,
                    backend_config const& config) {

#if ENABLE_MKLDNN
        if(backend_name == "mkldnn") {
            return std::make_unique<mkldnn_backend::model_core>(
              mkldnn_backend::make_model_core(input_table, output_table,
                                              model_data, config));
        }
#endif

#if ENABLE_ARMNN
        if(backend_name == "armnn") {
            return std::make_unique<armnn_backend::model_core>(
              armnn_backend::make_model_core(input_table, output_table,
                                              model_data, config));
        }
#endif

        throw invalid_backend_name(backend_name);
    }

} // namespace menoh_impl
