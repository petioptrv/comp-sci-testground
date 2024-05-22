//
// Created by Petio Petrov on 2023-04-07.
//
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "AVLTree.h"

using namespace std;
using ::testing::ElementsAre;

TEST(AVLNode, BasicBuild) {
    int a = 1;
    AVLNode<int> node(a);

    EXPECT_EQ(a, node.value);
    EXPECT_EQ(1, node.balance);
}

TEST(AVLTree, insert) {
    AVLTree<int> tree;
    tree.insert(1);
    tree.insert(2);
    tree.insert(3);

    EXPECT_FALSE(tree.isEmpty());
    ASSERT_THAT(tree.traversePostOrder(), ElementsAre(1, 3, 2));
}