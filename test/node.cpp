#include <gtest/gtest.h>

#include <unordered_set>
#include <iostream>

#include <menoh/node.hpp>

#include "./common.hpp"

namespace menoh_impl {

    namespace {

        class NodeTest : public ::testing::Test {
        public:
            void SetUp() {}
        };

        TEST_F(NodeTest, optional_attribute_float_access_test) {
            EXPECT_NEAR(optional_attribute_float(
                        node{"", {}, {}, {{"attr", 3.14f}}}, "attr", 4.2),
                      3.14, 1e-4);
            EXPECT_NEAR(optional_attribute_float(
                        node{"", {}, {}, {}}, "attr", 4.2),
                      4.2, 1e-4);
        }

    } // namespace

} // namespace menoh_impl
