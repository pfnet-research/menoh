#include <gtest/gtest.h>

#include <iostream>
#include <unordered_set>

#include <menoh/menoh.cpp>
#include <menoh/menoh.hpp>

#include "common.hpp"

namespace {
    class ModelDataTest : public ::testing::Test {};

    TEST_F(ModelDataTest, make_model_data_with_default_constructor) {
        menoh::model_data model_data;
        EXPECT_EQ(model_data.get()->model_data.node_list.size(), 0);
    }

    TEST_F(ModelDataTest, add_new_node) {
        menoh::model_data model_data;
        model_data.add_new_node("op1");
        EXPECT_EQ(model_data.get()->model_data.node_list.size(), 1);
        EXPECT_EQ(model_data.get()->model_data.node_list.at(0).op_type, "op1");
        model_data.add_new_node("op2");
        EXPECT_EQ(model_data.get()->model_data.node_list.size(), 2);
        EXPECT_EQ(model_data.get()->model_data.node_list.at(1).op_type, "op2");
    }

    TEST_F(ModelDataTest, add_attribute) {
        menoh::model_data model_data;
        model_data.add_new_node("");
        auto const& attribute_table =
          model_data.get()->model_data.node_list.front().attribute_table;

        model_data.add_attribute_int_to_current_node("attr1", 42);
        EXPECT_EQ(menoh_impl::get<int>(attribute_table.at("attr1")), 42);

        model_data.add_attribute_int_to_current_node("attr2", 53);
        EXPECT_EQ(menoh_impl::get<int>(attribute_table.at("attr1")), 42);
        EXPECT_EQ(menoh_impl::get<int>(attribute_table.at("attr2")), 53);

        // attribute duplication is not allowed
        EXPECT_THROW(model_data.add_attribute_int_to_current_node("attr1", 123),
                     menoh::error);
    }

    TEST_F(ModelDataTest, add_parameter) {
        menoh::model_data model_data;
        auto const& parameter_name_and_array_list =
          model_data.get()->model_data.parameter_name_and_array_list;
        std::vector<float> param1_buffer({1, 2, 3, 4});
        model_data.add_parameter("param1", menoh::dtype_t::float_, {2, 2},
                                 param1_buffer.data());
        EXPECT_EQ(parameter_name_and_array_list.front().first, "param1");
        EXPECT_EQ(parameter_name_and_array_list.front().second.data(),
                  param1_buffer.data());
        menoh_impl::array param1_clone_arr(
          menoh_impl::dtype_t::float_, std::vector<int>({2, 2}),
          static_cast<void*>(param1_buffer.data()));
        EXPECT_TRUE(is_near_array(parameter_name_and_array_list.front().second,
                                  param1_clone_arr));

        std::vector<float> param2_buffer({11, 12, 13, 14});
        model_data.add_parameter("param2", menoh::dtype_t::float_, {2, 2},
                                 param2_buffer.data());

        EXPECT_EQ(parameter_name_and_array_list.front().first, "param1");
        EXPECT_EQ(parameter_name_and_array_list.front().second.data(),
                  param1_buffer.data());
        EXPECT_TRUE(is_near_array(parameter_name_and_array_list.front().second,
                                  param1_clone_arr));

        EXPECT_EQ(parameter_name_and_array_list.at(1).first, "param2");
        EXPECT_EQ(parameter_name_and_array_list.at(1).second.data(),
                  param2_buffer.data());
        menoh_impl::array param2_clone_arr(
          menoh_impl::dtype_t::float_, std::vector<int>({2, 2}),
          static_cast<void*>(param2_buffer.data()));
        EXPECT_TRUE(is_near_array(parameter_name_and_array_list.at(1).second,
                                  param2_clone_arr));

        // named parameter duplication is not allowed
        EXPECT_THROW(model_data.add_parameter("param1", menoh::dtype_t::float_,
                                              {2, 2}, param1_buffer.data()),
                     menoh::error);
    }
} // namespace
