#ifndef EXCEPTION_HPP
#define EXCEPTION_HPP

#include <string>
#include <sstream>

namespace menoh_impl {
namespace tensorrt_backend {

// base class for all ArmNN exceptions so that users can filter to just those
class Exception : public std::exception
{
public:
    explicit Exception(const std::string& message);

    virtual const char* what() const noexcept override;

private:
    std::string m_Message;
};

class InvalidArgumentException : public Exception
{
public:
    using Exception::Exception;
};

class ParseException : public Exception
{
public:
    using Exception::Exception;
};

} // namespace tensorrt_backend
} // menoh_impl

#endif // EXCEPTION_HPP

