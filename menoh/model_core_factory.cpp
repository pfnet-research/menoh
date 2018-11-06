#include <menoh/model_core.hpp>
#include <menoh/model_core_factory.hpp>

#include <menoh/mkldnn/model_core.hpp>
#include <menoh/combinated_backends/model_core.hpp>

#include <menoh/combinated_backends/backend/generic/generic_context.hpp>
#include <menoh/combinated_backends/backend/mkldnn/mkldnn_context.hpp>

namespace menoh_impl {

    std::unique_ptr<menoh_impl::model_core>
    make_model_core(std::unordered_map<std::string, array> const& input_table,
                    std::unordered_map<std::string, array> const& required_output_table,
                    std::unordered_map<std::string, array_profile> const&
                      output_profile_table,
                    menoh_impl::model_data const& model_data,
                    std::string const& backend_name,
                    backend_config const& config) {
        if(backend_name == "mkldnn") {
            return std::make_unique<mkldnn_backend::model_core>(
              mkldnn_backend::make_model_core(input_table, required_output_table,
                                              model_data, config));
        } else if(backend_name == "combinated_backends") {
            using namespace combinated_backends;
            std::vector<std::pair<std::string, std::unique_ptr<context>>>
              context_list;
            context_list.emplace_back(
              "mkldnn", std::make_unique<combinated_backends::
                                           mkldnn_backend::mkldnn_context>());
            context_list.emplace_back(
              "generic",
              std::make_unique<combinated_backends::
                                 generic_backend::generic_context>());
            return std::make_unique<
              combinated_backends::model_core>(
              std::move(context_list), input_table, required_output_table,
              output_profile_table, model_data, config);
        }

        throw invalid_backend_name(backend_name);
    }

} // namespace menoh_impl
