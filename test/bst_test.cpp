//
// Created by Petio Petrov on 2023-04-07.
//
#include <iostream>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "BST.h"

using namespace std;
using ::testing::ElementsAre;

TEST(BSTNode, BasicBuild) {
    int a = 1;
    BSTNode<int> node(a);

    EXPECT_EQ(a, node.value);
}

TEST(BST, isEmpty) {
    BST<int> tree;

    EXPECT_TRUE(tree.isEmpty());
}

TEST(BST, insert) {
    BST<int> tree;
    tree.insert(3);
    tree.insert(1);
    tree.insert(2);

    EXPECT_FALSE(tree.isEmpty());
    ASSERT_THAT(tree.traverse(), ElementsAre(1, 2, 3));
}

TEST(BST, remove) {
    BST<int> tree;
    tree.insert(10);
    tree.insert(5);
    tree.insert(2);
    tree.insert(1);
    tree.insert(4);
    tree.insert(3);

    ASSERT_THAT(tree.traverse(), ElementsAre(1, 2, 3, 4, 5, 10));

    tree.remove(5);

    ASSERT_THAT(tree.traverse(), ElementsAre(1, 2, 3, 4, 10));
}
