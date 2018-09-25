#ifndef TYPESUTILS_HPP
#define TYPESUTILS_HPP

#include <NvInfer.h>

using namespace nvinfer1;

template<typename T, typename U=T>
struct GetDataTypeImpl;

template<>
struct GetDataTypeImpl<float>
{
    static constexpr DataType Value = DataType::kFLOAT;
};

constexpr unsigned int GetDataTypeSize(DataType dataType)
{
    switch (dataType)
    {
        case DataType::kFLOAT:     return 4U;
        default:                   return 0U;
    }
}

template <typename T>
constexpr DataType GetDataType()
{
    return GetDataTypeImpl<T>::Value;
}

#endif // TYPESUTILS_HPP
