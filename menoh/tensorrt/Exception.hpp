#ifndef EXCEPTION_HPP
#define EXCEPTION_HPP

#include <string>

namespace menoh_impl {
namespace tensorrt_backend {

class Exception : public std::exception
{
public:
    explicit Exception(const std::string& message);

    virtual const char* what() const noexcept override;

private:
    std::string msg;
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

