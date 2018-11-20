#ifndef MENOH_MODEL_CORE_FACTORY_HPP
#define MENOH_MODEL_CORE_FACTORY_HPP

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <menoh/array.hpp>
#include <menoh/backend_config.hpp>
#include <menoh/exception.hpp>
#include <menoh/model_core.hpp>

namespace menoh_impl {

    class invalid_backend_name : public exception {
    public:
        explicit invalid_backend_name(std::string const& name)
          : exception(menoh_error_code_invalid_backend_name,
                      "menoh invalid backend name error: " + name) {}
    };

    struct model_data;

    std::unique_ptr<menoh_impl::model_core> make_model_core(
      std::unordered_map<std::string, array> const& input_table,
      std::unordered_map<std::string, array> const& required_output_table,
      std::unordered_map<std::string, array_profile> const&
        output_profile_table,
      menoh_impl::model_data const& model_data, std::string const& backend_name,
      backend_config const& config = backend_config());

} // namespace menoh_impl

#endif // MENOH_MODEL_CORE_FACTORY_HPP
