//
// Created by Petio Petrov on 2023-04-07.
//
#include <iostream>
#include <gtest/gtest.h>

using namespace std;

TEST(FirstBSTTest, HelloWorld) {
    cout << "Hello BST Test World!" << endl;
    int target = 1;
    EXPECT_EQ(target, 1);
}

TEST(AnotherTest, ThisisaTest) {
    cout << "This is a test" << endl;
}
