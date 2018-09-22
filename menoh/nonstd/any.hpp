//
// Copyright (c) 2016-2018 Martin Moene
//
// https://github.com/martinmoene/any-lite
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#ifndef NONSTD_ANY_LITE_HPP
#define NONSTD_ANY_LITE_HPP

#define any_lite_MAJOR  0
#define any_lite_MINOR  1
#define any_lite_PATCH  0

#define any_lite_VERSION  any_STRINGIFY(any_lite_MAJOR) "." any_STRINGIFY(any_lite_MINOR) "." any_STRINGIFY(any_lite_PATCH)

#define any_STRINGIFY(  x )  any_STRINGIFY_( x )
#define any_STRINGIFY_( x )  #x

// any-lite configuration:

#define any_ANY_DEFAULT  0
#define any_ANY_LITE     1
#define any_ANY_STD      2

#if !defined( any_CONFIG_SELECT_ANY )
# define any_CONFIG_SELECT_ANY  ( any_HAVE_STD_ANY ? any_ANY_STD : any_ANY_LITE )
#endif

// C++ language version detection (C++20 is speculative):
// Note: VC14.0/1900 (VS2015) lacks too much from C++14.

#ifndef   any_CPLUSPLUS
# ifdef  _MSVC_LANG
#  define any_CPLUSPLUS  (_MSC_VER == 1900 ? 201103L : _MSVC_LANG )
# else
#  define any_CPLUSPLUS  __cplusplus
# endif
#endif

#define any_CPP98_OR_GREATER  ( any_CPLUSPLUS >= 199711L )
#define any_CPP11_OR_GREATER  ( any_CPLUSPLUS >= 201103L )
#define any_CPP14_OR_GREATER  ( any_CPLUSPLUS >= 201402L )
#define any_CPP17_OR_GREATER  ( any_CPLUSPLUS >= 201703L )
#define any_CPP20_OR_GREATER  ( any_CPLUSPLUS >= 202000L )

// use C++17 std::any if available and requested:

#if any_CPP17_OR_GREATER && defined(__has_include ) && __has_include( <any> )
# define any_HAVE_STD_ANY  1
#else
# define any_HAVE_STD_ANY  0
#endif

#define any_USES_STD_ANY  ( (any_CONFIG_SELECT_ANY == any_ANY_STD) || ((any_CONFIG_SELECT_ANY == any_ANY_DEFAULT) && any_HAVE_STD_ANY) )

// Using std::any:

#if any_USES_STD_ANY

#include <any>

namespace nonstd {

    using std::any;
    using std::any_cast;
    using std::make_any;
    using std::swap;
    using std::bad_any_cast;

    using std::in_place;
    using std::in_place_type;
    using std::in_place_t;
    using std::in_place_type_t;
}

#else // C++17 std::any

#include <typeinfo>
#include <utility>

// Compiler versions:
//
// MSVC++ 6.0  _MSC_VER == 1200 (Visual Studio 6.0)
// MSVC++ 7.0  _MSC_VER == 1300 (Visual Studio .NET 2002)
// MSVC++ 7.1  _MSC_VER == 1310 (Visual Studio .NET 2003)
// MSVC++ 8.0  _MSC_VER == 1400 (Visual Studio 2005)
// MSVC++ 9.0  _MSC_VER == 1500 (Visual Studio 2008)
// MSVC++ 10.0 _MSC_VER == 1600 (Visual Studio 2010)
// MSVC++ 11.0 _MSC_VER == 1700 (Visual Studio 2012)
// MSVC++ 12.0 _MSC_VER == 1800 (Visual Studio 2013)
// MSVC++ 14.0 _MSC_VER == 1900 (Visual Studio 2015)
// MSVC++ 14.1 _MSC_VER >= 1910 (Visual Studio 2017)

#if defined( _MSC_VER ) && !defined( __clang__ )
# define any_COMPILER_MSVC_VERSION  (_MSC_VER / 10 - 10 * ( 5 + (_MSC_VER < 1900 ) ) )
#else
# define any_COMPILER_MSVC_VERSION  0
#endif

#define any_COMPILER_VERSION( major, minor, patch )  ( 10 * ( 10 * (major) + (minor) ) + (patch) )

#if defined __clang__
# define any_COMPILER_CLANG_VERSION  any_COMPILER_VERSION(__clang_major__, __clang_minor__, __clang_patchlevel__)
#else
# define any_COMPILER_CLANG_VERSION  0
#endif

#if defined __GNUC__
# define any_COMPILER_GNUC_VERSION  any_COMPILER_VERSION(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
#else
# define any_COMPILER_GNUC_VERSION  0
#endif

// half-open range [lo..hi):
//#define any_BETWEEN( v, lo, hi ) ( lo <= v && v < hi )

// Presence of language and library features:

#define any_HAVE( feature )  ( any_HAVE_##feature )

#ifdef _HAS_CPP0X
# define any_HAS_CPP0X  _HAS_CPP0X
#else
# define any_HAS_CPP0X  0
#endif

#define any_CPP11_90   (any_CPP11_OR_GREATER || any_COMPILER_MSVC_VERSION >= 90)
#define any_CPP11_100  (any_CPP11_OR_GREATER || any_COMPILER_MSVC_VERSION >= 100)
#define any_CPP11_120  (any_CPP11_OR_GREATER || any_COMPILER_MSVC_VERSION >= 120)
#define any_CPP11_140  (any_CPP11_OR_GREATER || any_COMPILER_MSVC_VERSION >= 140)

#define any_CPP14_000  (any_CPP14_OR_GREATER)
#define any_CPP17_000  (any_CPP17_OR_GREATER)

// Presence of C++11 language features:

#define any_HAVE_CONSTEXPR_11           any_CPP11_140
#define any_HAVE_DEFAULT_FUNCTION_TEMPLATE_ARG \
                                        any_CPP11_120
#define any_HAVE_INITIALIZER_LIST       any_CPP11_120
#define any_HAVE_NOEXCEPT               any_CPP11_140
#define any_HAVE_NULLPTR                any_CPP11_100
#define any_HAVE_TYPE_TRAITS            any_CPP11_90
#define any_HAVE_STATIC_ASSERT          any_CPP11_100
#define any_HAVE_ADD_CONST              any_CPP11_90
#define any_HAVE_REMOVE_REFERENCE       any_CPP11_90

#define any_HAVE_TR1_ADD_CONST          (!! any_COMPILER_GNUC_VERSION )
#define any_HAVE_TR1_REMOVE_REFERENCE   (!! any_COMPILER_GNUC_VERSION )
#define any_HAVE_TR1_TYPE_TRAITS        (!! any_COMPILER_GNUC_VERSION )

// Presence of C++14 language features:

#define any_HAVE_CONSTEXPR_14           any_CPP14_000

// Presence of C++17 language features:

#define any_HAVE_NODISCARD              any_CPP17_000

// Presence of C++ library features:

#if any_HAVE_CONSTEXPR_11
# define any_constexpr constexpr
#else
# define any_constexpr /*constexpr*/
#endif

#if any_HAVE_CONSTEXPR_14
# define any_constexpr14 constexpr
#else
# define any_constexpr14 /*constexpr*/
#endif

#if any_HAVE_NOEXCEPT
# define any_noexcept noexcept
#else
# define any_noexcept /*noexcept*/
#endif

#if any_HAVE_NULLPTR
# define any_nullptr nullptr
#else
# define any_nullptr NULL
#endif

#if any_HAVE_NODISCARD
# define any_nodiscard [[nodiscard]]
#else
# define any_nodiscard /*[[nodiscard]]*/
#endif

// additional includes:

#if ! any_HAVE_NULLPTR
# include <cstddef>
#endif

#if any_HAVE_INITIALIZER_LIST
# include <initializer_list>
#endif

#if any_HAVE_TYPE_TRAITS
# include <type_traits>
#elif any_HAVE_TR1_TYPE_TRAITS
# include <tr1/type_traits>
#endif

//
// in_place: code duplicated in any-lite, optional-lite, variant-lite:
//

#if ! defined nonstd_lite_HAVE_IN_PLACE_TYPES

namespace nonstd {

namespace detail {

template< class T >
struct in_place_type_tag {};

template< std::size_t I >
struct in_place_index_tag {};

} // namespace detail

struct in_place_t {};

template< class T >
inline in_place_t in_place( detail::in_place_type_tag<T> = detail::in_place_type_tag<T>() )
{
    return in_place_t();
}

template< std::size_t I >
inline in_place_t in_place( detail::in_place_index_tag<I> = detail::in_place_index_tag<I>() )
{
    return in_place_t();
}

template< class T >
inline in_place_t in_place_type( detail::in_place_type_tag<T> = detail::in_place_type_tag<T>() )
{
    return in_place_t();
}

template< std::size_t I >
inline in_place_t in_place_index( detail::in_place_index_tag<I> = detail::in_place_index_tag<I>() )
{
    return in_place_t();
}

// mimic templated typedef:

#define nonstd_lite_in_place_type_t( T)  nonstd::in_place_t(&)( nonstd::detail::in_place_type_tag<T>  )
#define nonstd_lite_in_place_index_t(T)  nonstd::in_place_t(&)( nonstd::detail::in_place_index_tag<I> )

#define nonstd_lite_HAVE_IN_PLACE_TYPES  1

} // namespace nonstd

#endif // nonstd_lite_HAVE_IN_PLACE_TYPES

//
// any:
//

namespace nonstd {  namespace any_lite {

namespace detail {

// C++11 emulation:

#if any_HAVE_ADD_CONST

using std::add_const;

#elif any_HAVE_TR1_ADD_CONST

using std::tr1::add_const;

#else

template< class T > struct add_const { typedef const T type; };

#endif // any_HAVE_ADD_CONST

#if any_HAVE_REMOVE_REFERENCE

using std::remove_reference;

#elif any_HAVE_TR1_REMOVE_REFERENCE

using std::tr1::remove_reference;

#else

template< class T > struct remove_reference     { typedef T type; };
template< class T > struct remove_reference<T&> { typedef T type; };

#endif // any_HAVE_REMOVE_REFERENCE

} // namespace detail

class bad_any_cast : public std::bad_cast
{
public:
#if any_CPP11_OR_GREATER
    virtual const char* what() const any_noexcept
#else
    virtual const char* what() const throw()
#endif
   {
      return "any-lite: bad any_cast";
   }
};

class any
{
public:
    any_constexpr any() any_noexcept
    : content( any_nullptr )
    {}

    any( any const & rhs )
    : content( rhs.content ? rhs.content->clone() : any_nullptr )
    {}

#if any_CPP11_OR_GREATER

    any( any && rhs ) any_noexcept
    : content( std::move( rhs.content ) )
    {
        rhs.content = any_nullptr;
    }

    template<
        class ValueType, class T = typename std::decay<ValueType>::type
        , typename = typename std::enable_if< ! std::is_same<T, any>::value >::type
    >
    any( ValueType && value ) any_noexcept
    : content( new holder<T>( std::move( value ) ) )
    {}

    template<
        class T, class... Args
        , typename = typename std::enable_if< std::is_constructible<T, Args...>::value >::type
    >
    explicit any( nonstd_lite_in_place_type_t(T), Args&&... args )
    : content( new holder<T>( T( std::forward<Args>(args)... ) ) )
    {}

    template<
        class T, class U, class... Args
        , typename = typename std::enable_if< std::is_constructible<T, std::initializer_list<U>&, Args...>::value >::type
    >
    explicit any( nonstd_lite_in_place_type_t(T), std::initializer_list<U> il, Args&&... args )
    : content( new holder<T>( T( il, std::forward<Args>(args)... ) ) )
    {}

#else

    template< class ValueType >
    any( ValueType const & value )
    : content( new holder<ValueType>( value ) )
    {}

#endif // any_CPP11_OR_GREATER

    ~any()
    {
        reset();
    }

    any & operator=( any const & rhs )
    {
        any( rhs ).swap( *this );
        return *this;
    }

#if any_CPP11_OR_GREATER

    any & operator=( any && rhs ) any_noexcept
    {
        any( std::move( rhs ) ).swap( *this );
        return *this;
    }

    template<
        class ValueType, class T = typename std::decay<ValueType>::type
        , typename = typename std::enable_if< ! std::is_same<T, any>::value >::type
    >
    any & operator=( ValueType && rhs )
    {
        any( std::move( rhs ) ).swap( *this );
        return *this;
    }

    template< class T, class... Args >
    void emplace( Args && ... args )
    {
        any( T( std::forward<Args>(args)... ) ).swap( *this );
    }

    template<
        class T, class U, class... Args
        , typename = typename std::enable_if< std::is_constructible<T, std::initializer_list<U>&, Args...>::value >::type
    >
    void emplace( std::initializer_list<U> il, Args&&... args )
    {
        any( T( il, std::forward<Args>(args)... ) ).swap( *this );
    }

#else

    template< class ValueType >
    any & operator=( ValueType const & rhs )
    {
        any( rhs ).swap( *this );
        return *this;
    }

#endif // any_CPP11_OR_GREATER

    void reset() any_noexcept
    {
        delete content; content = any_nullptr;
    }

    void swap( any & rhs ) any_noexcept
    {
        std::swap( content, rhs.content );
    }

    bool has_value() const any_noexcept
    {
        return content != any_nullptr;
    }

    const std::type_info & type() const any_noexcept
    {
        return has_value() ? content->type() : typeid( void );
    }

    //
    // non-standard:
    //

    template< class ValueType >
    const ValueType * to_ptr() const
    {
        return &( static_cast<holder<ValueType> *>( content )->held );
    }

    template< class ValueType >
    ValueType * to_ptr()
    {
        return &( static_cast<holder<ValueType> *>( content )->held );
    }

private:
    class placeholder
    {
    public:
        virtual ~placeholder()
        {
        }

        virtual std::type_info const & type() const = 0;

        virtual placeholder * clone() const = 0;
    };

    template< typename ValueType >
    class holder : public placeholder
    {
    public:
        holder( ValueType const & value )
        : held( value )
        {}

#if any_CPP11_OR_GREATER
        holder( ValueType && value )
        : held( std::move( value ) )
        {}
#endif

        virtual std::type_info const & type() const
        {
            return typeid( ValueType );
        }

        virtual placeholder * clone() const
        {
            return new holder( held );
        }

        ValueType held;
    };

    placeholder * content;
};

inline void swap( any & x, any & y ) any_noexcept
{
    x.swap( y );
}

#if any_CPP11_OR_GREATER

template< class T, class ...Args >
inline any make_any( Args&& ...args )
{
    return any( in_place<T>, std::forward<Args>(args)...);
}

template< class T, class U, class ...Args >
inline any make_any( std::initializer_list<U> il, Args&& ...args )
{
    return any( in_place<T>, il, std::forward<Args>(args)...);
}

#endif // any_CPP11_OR_GREATER

template<
    class ValueType
#if any_HAVE_DEFAULT_FUNCTION_TEMPLATE_ARG
    , typename = typename std::enable_if< std::is_reference<ValueType>::value || std::is_copy_constructible<ValueType>::value >::type
#endif
>
any_nodiscard inline ValueType any_cast( any const & operand )
{
   const ValueType * result = any_cast< typename detail::add_const< typename detail::remove_reference<ValueType>::type >::type >( &operand );

   if ( ! result )
   {
      throw bad_any_cast();
   }

   return *result;
}

template<
    class ValueType
#if any_HAVE_DEFAULT_FUNCTION_TEMPLATE_ARG
    , typename = typename std::enable_if< std::is_reference<ValueType>::value || std::is_copy_constructible<ValueType>::value >::type
#endif
>
any_nodiscard inline ValueType any_cast( any & operand )
{
   const ValueType * result = any_cast< typename detail::remove_reference<ValueType>::type >( &operand );

   if ( ! result )
   {
      throw bad_any_cast();
   }

   return *result;
}

#if any_CPP11_OR_GREATER

template<
    class ValueType
#if any_HAVE_DEFAULT_FUNCTION_TEMPLATE_ARG
    , typename = typename std::enable_if< std::is_reference<ValueType>::value || std::is_copy_constructible<ValueType>::value >::type
#endif
>
any_nodiscard inline ValueType any_cast( any && operand )
{
   const ValueType * result = any_cast< typename detail::remove_reference<ValueType>::type >( &operand );

   if ( ! result )
   {
      throw bad_any_cast();
   }

   return *result;
}

#endif // any_CPP11_OR_GREATER

template< class ValueType >
any_nodiscard inline ValueType const * any_cast( any const * operand ) any_noexcept
{
    return operand != any_nullptr && operand->type() == typeid(ValueType) ? operand->to_ptr<ValueType>() : any_nullptr;
}

template<class ValueType >
any_nodiscard inline ValueType * any_cast( any * operand ) any_noexcept
{
    return operand != any_nullptr && operand->type() == typeid(ValueType) ? operand->to_ptr<ValueType>() : any_nullptr;
}

} // namespace any_lite

using namespace any_lite;

} // namespace nonstd

#endif // have C++17 std::any

#endif // NONSTD_ANY_LITE_HPP
