#ifndef MENOH_IMPL_MKLDNN_WITH_FALLBACK_BACKEND_PROCEDURE_HPP
#define MENOH_IMPL_MKLDNN_WITH_FALLBACK_BACKEND_PROCEDURE_HPP

#include <functional>

namespace menoh_impl {
namespace mkldnn_with_generic_fallback_backend {
    using procedure = std::function<void()>;
} // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_FALLBACK_BACKEND_PROCEDURE_HPP
