set(CMAKE_CXX_STANDARD 14)

include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
        hashing_test
        hashing_test.cpp
)
target_link_libraries(hashing_test PUBLIC GTest::gtest_main string_hashing hashing)

add_executable(
        bst_test
        bst_test.cpp
)
target_link_libraries(
        bst_test PUBLIC GTest::gtest_main GTest::gmock GTest::gtest bst
)

add_executable(
        avl_tree_test
        avl_tree_test.cpp
)
target_link_libraries(avl_tree_test PUBLIC GTest::gtest_main avl_tree)

include(GoogleTest)
gtest_discover_tests(hashing_test)
gtest_discover_tests(bst_test)
gtest_discover_tests(avl_tree_test)
