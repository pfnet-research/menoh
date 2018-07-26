#ifndef MENOH_TEST_COMMON_HPP
#define MENOH_TEST_COMMON_HPP

namespace menoh_impl {

    template <typename Iter1, typename Iter2>
    void assert_eq_list(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2) {
        ASSERT_EQ(std::distance(first1, last1), std::distance(first2, last2))
          << "size is different";
        while(first1 != last1) {
            ASSERT_EQ(*first1, *first2)
              << *first1 << " and " << *first2 << " are different";
            ++first1;
            ++first2;
        }
    }

    template <typename List1, typename List2>
    void assert_eq_list(List1 const& list1, List2 const& list2) {
        using std::begin;
        using std::end;
        assert_eq_list(begin(list1), end(list1), begin(list2), end(list2));
    }

    template <typename Iter1, typename Iter2>
    void assert_near_list(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2, float eps) {
        ASSERT_EQ(std::distance(first1, last1), std::distance(first2, last2))
          << "size is different";
        int i = 0;
        while(first1 != last1) {
            ASSERT_NEAR(*first1, *first2, eps)
              << i << ": " << *first1 << " and " << *first2 << " are different";
            ++first1;
            ++first2;
            ++i;
        }
    }

    template <typename List1, typename List2>
    void assert_near_list(List1 const& list1, List2 const& list2, float eps) {
        using std::begin;
        using std::end;
        assert_near_list(begin(list1), end(list1), begin(list2), end(list2), eps);
    }
}

#endif // MENOH_TEST_COMMON_HPP
