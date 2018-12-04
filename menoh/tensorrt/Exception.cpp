#include <menoh/tensorrt/Exception.hpp>

#include <string>

namespace menoh_impl {
namespace tensorrt_backend {

Exception::Exception(const std::string& message)
: msg(message)
{
}

const char* Exception::what() const noexcept
{
    return msg.c_str();
}

} // namespace tensorrt_backend
} // namespace menoh_impl
