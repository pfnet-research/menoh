#ifndef MENOH_UTILITY_HPP
#define MENOH_UTILITY_HPP

#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <menoh/exception.hpp>

namespace menoh_impl {

    class variable_not_found : public exception {
    public:
        variable_not_found(std::string const& variable_name)
          : exception(menoh_error_code_variable_not_found,
                      "menoh variable not found error: " + variable_name) {}
    };

    template <typename T>
    auto const& find_value(std::unordered_map<std::string, T> const& m,
                           std::string const& key) {
        auto found = m.find(key);
        if(found == m.end()) {
            throw variable_not_found(key);
        }
        return found->second;
    }

    template <typename T>
    auto& find_value(std::unordered_map<std::string, T>& m,
                     std::string const& key) {
        auto found = m.find(key);
        if(found == m.end()) {
            throw variable_not_found(key);
        }
        return found->second;
    }

} // namespace menoh_impl

#endif // MENOH_UTILITY_HPP
