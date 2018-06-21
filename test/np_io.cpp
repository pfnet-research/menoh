#include <gtest/gtest.h>

#include "./common.hpp"
#include "np_io.hpp"

#include <array>

namespace menoh_impl {

    namespace {

        class NumpyIOTest : public ::testing::Test {};

        TEST_F(NumpyIOTest, load_test) {
            int dims_num;
            std::vector<int> shape;
            std::vector<float> data;
            std::tie(dims_num, shape, data) =
              load_np_array("../data/random_input_3_4_32_32.txt");
            (void)data;
            ASSERT_EQ(dims_num, 4);
            assert_eq_list(shape, std::array<int, 4>{{3, 4, 32, 32}});
        }

        TEST_F(NumpyIOTest, load_as_arr_test) {
            auto arr = load_np_array_as_array("../data/random_input_3_4_32_32.txt");
            ASSERT_EQ(arr.dims().size(), 4);
            assert_eq_list(arr.dims(), std::array<int, 4>{{3, 4, 32, 32}});
        }

    } // namespace

} // namespace menoh_impl
