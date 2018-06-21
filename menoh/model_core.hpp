#ifndef MENOH_MODEL_CORE_HPP
#define MENOH_MODEL_CORE_HPP

#include <stdexcept>
#include <string>
#include <unordered_map>

#include <menoh/exception.hpp>

namespace menoh_impl {

    class failed_to_configure_operator : public exception {
    public:
        explicit failed_to_configure_operator(
          std::string const& op_type, std::string const& first_output_name,
          std::string const& message)
          : exception(menoh_error_code_failed_to_configure_operator,
                      "menoh failed to configure operator error: " + op_type +
                        " which issues " + first_output_name + ": " + message) {
        }
    };

    class backend_error : public exception {
    public:
        explicit backend_error(std::string const& backend_name,
                               std::string const& message)
          : exception(menoh_error_code_backend_error,
                      "menoh backend error: " + backend_name + ": " + message) {
        }
    };

    class model_core {
    public:
        virtual ~model_core() = 0;

        void run() { do_run(); }

    private:
        virtual void do_run() = 0;
    };

} // namespace menoh_impl
#endif // MENOH_MODEL_CORE_HPP
