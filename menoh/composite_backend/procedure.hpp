#ifndef MENOH_IMPL_COMPOSITE_BACKEND_PROCEDURE_HPP
#define MENOH_IMPL_COMPOSITE_BACKEND_PROCEDURE_HPP

#include <functional>

namespace menoh_impl {
namespace composite_backend {
    using procedure = std::function<void()>;
} // namespace composite_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_COMPOSITE_BACKEND_PROCEDURE_HPP
