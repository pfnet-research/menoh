#include <gtest/gtest.h>

#include <vector>

#include <menoh/array.hpp>

namespace menoh_impl {
    namespace {

        class ArrayTest : public ::testing::Test {};

        TEST_F(ArrayTest, test_array_construct_from_int32_data_handle) {
            std::vector<int32_t> data(1 * 2 * 3);
            array arr(dtype_t::int32, {1, 2, 3}, data.data());
            ASSERT_EQ(arr.data(), data.data());
        }

        TEST_F(ArrayTest, test_array_construct_from_int64_data_handle) {
            std::vector<int64_t> data(1 * 2 * 3);
            array arr(dtype_t::int32, {1, 2, 3}, data.data());
            ASSERT_EQ(arr.data(), data.data());
        }

        TEST_F(ArrayTest, test_array_construct_from_float_data_handle) {
            std::vector<float> data(1 * 2 * 3);
            array arr(dtype_t::int32, {1, 2, 3}, data.data());
            ASSERT_EQ(arr.data(), data.data());
        }

        TEST_F(ArrayTest, test_array_construct_int32) {
            volatile array arr(dtype_t::int32, {1, 2, 3});
        }

        TEST_F(ArrayTest, test_array_construct_int64) {
            volatile array arr(dtype_t::int64, {1, 2, 3});
        }

        TEST_F(ArrayTest, test_array_construct_float) {
            volatile array arr(dtype_t::float_, {1, 2, 3});
        }

        TEST_F(ArrayTest, test_array_copy) {
            array arr(dtype_t::float_, {1, 2, 3});
            auto arr2 = arr;
            ASSERT_EQ(arr2.data(), arr.data());
        }

        TEST_F(ArrayTest, test_array_move_copy) {
            std::vector<float> data(1 * 2 * 3);
            array arr(dtype_t::float_, {1, 2, 3}, data.data());
            auto arr2 = std::move(arr);
            ASSERT_EQ(arr2.data(), data.data());
        }

    } // namespace
} // namespace menoh_impl
