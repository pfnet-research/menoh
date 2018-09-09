#ifndef MENOH_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_LOGGER_HPP
#define MENOH_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_LOGGER_HPP

#include <iostream>

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {
        using logger = std::unique_ptr<std::ostream>;
        using logger_handle = std::ostream*;
    } // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl

#endif // MENOH_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_LOGGER_HPP
