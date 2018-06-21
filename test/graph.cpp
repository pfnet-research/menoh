#include <gtest/gtest.h>

#include <unordered_set>

#include <menoh/graph.hpp>

#include "./common.hpp"

namespace menoh_impl {

    namespace {

        auto print_node_list(std::vector<node> const& node_list) {
            for(auto const& node : node_list) {
                for(auto const& input : node.input_name_list) {
                    std::cout << input << " ";
                }
                std::cout << "-> ";
                for(auto const& output : node.output_name_list) {
                    std::cout << output << " ";
                }
                std::cout << "\n";
            }
        }

        class GraphTest : public ::testing::Test {
        public:
            void SetUp() {
                stright_node_list.push_back(node{"", {"a"}, {"b"}, {}});
                stright_node_list.push_back(node{"", {"b"}, {"c"}, {}});
                stright_node_list.push_back(node{"", {"c"}, {"d"}, {}});

                complex_node_list.push_back(node{"", {"a"}, {"b"}, {}});
                complex_node_list.push_back(node{"", {"b"}, {"c"}, {}});
                complex_node_list.push_back(node{"", {"b"}, {"d"}, {}});
                complex_node_list.push_back(node{"", {"a", "c"}, {"e"}, {}});
            }

            std::vector<node> stright_node_list;
            std::vector<node> complex_node_list;
        };

        TEST_F(GraphTest, construct_test) {
            node n{"",
                   {"input"},
                   {"output"},
                   {{"strides", std::vector<int>{2, 2}},
                    {"pads", std::vector<int>{0, 0, 0, 0}},
                    {"ksize", std::vector<int>{3, 3}}}};
        }

        TEST_F(GraphTest, attribute_access_test) {
            node n{"",
                   {"input"},
                   {"output"},
                   {{"strides", std::vector<int>{2, 2}},
                    {"pads", std::vector<int>{0, 0, 0, 0}},
                    {"ksize", std::vector<int>{3, 3}}}};
            assert_eq_list(attribute_ints(n, "strides"), std::vector<int>{2, 2});
            assert_eq_list(attribute_ints(n, "pads"), std::vector<int>{0, 0, 0, 0});
            assert_eq_list(attribute_ints(n, "ksize"), std::vector<int>{3, 3});
        }

        TEST_F(GraphTest, extract_needed_node_list_test1) {
            auto needed_node_list =
              extract_needed_node_list(stright_node_list, {"d"});
            print_node_list(needed_node_list);
            ASSERT_EQ(needed_node_list.size(), 3);
        }

        TEST_F(GraphTest, extract_needed_node_list_test2) {
            auto needed_node_list =
              extract_needed_node_list(stright_node_list, {"c"});
            print_node_list(needed_node_list);
            ASSERT_EQ(needed_node_list.size(), 2);
        }

        TEST_F(GraphTest, extract_needed_node_list_test3) {
            auto needed_node_list =
              extract_needed_node_list(complex_node_list, {"d"});
            print_node_list(needed_node_list);
            ASSERT_EQ(needed_node_list.size(), 2);
        }

        TEST_F(GraphTest, extract_needed_node_list_test4) {
            auto needed_node_list =
              extract_needed_node_list(complex_node_list, {"c", "d"});
            print_node_list(needed_node_list);
            ASSERT_EQ(needed_node_list.size(), 3);
        }

        TEST_F(GraphTest, extract_needed_node_list_test5) {
            auto needed_node_list =
              extract_needed_node_list(complex_node_list, {"e"});
            print_node_list(needed_node_list);
            ASSERT_EQ(needed_node_list.size(), 3);
        }

    } // namespace

} // namespace menoh_impl
