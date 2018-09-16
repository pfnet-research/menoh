#ifndef MENOH_EXCEPTION
#define MENOH_EXCEPTION

#include <stdexcept>
#include <string>

#include <menoh/menoh.h>

namespace menoh_impl {

    class exception : public std::runtime_error {
    public:
        exception(menoh_error_code ec, std::string const& message)
          : std::runtime_error(message), ec_(ec) {}

        menoh_error_code error_code() const noexcept { return ec_; }

    private:
        menoh_error_code ec_;
    };

    class invalid_filename : public exception {
    public:
        explicit invalid_filename(std::string const& filename)
          : exception(menoh_error_code_invalid_filename,
                      "menoh invalid filename error: " + filename) {}
    };

    class json_parse_error : public exception {
    public:
        explicit json_parse_error(std::string const& message)
          : exception(menoh_error_code_json_parse_error,
                      "menoh json parse error: " + message) {}
    };

    class invalid_backend_config_error : public exception {
    public:
        explicit invalid_backend_config_error(std::string const& message)
          : exception(menoh_error_code_invalid_backend_config_error,
                      "menoh invalid backend config error: " + message) {}
    };

} // namespace menoh_impl

#endif // MENOH_EXCEPTION
