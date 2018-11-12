#ifndef MENOH_COMPOSITE_BACKEND_LOGGER_HPP
#define MENOH_COMPOSITE_BACKEND_LOGGER_HPP

#include <iostream>

namespace menoh_impl {
    namespace composite_backend {
        using logger = std::unique_ptr<std::ostream>;
        using logger_handle = std::ostream*;
    } // namespace composite_backend
} // namespace menoh_impl

#endif // MENOH_COMPOSITE_BACKEND_LOGGER_HPP
