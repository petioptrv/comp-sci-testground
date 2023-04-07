//
// Created by Petio Petrov on 2022-10-23.
//
#include <iostream>
#include <string>
#include <gtest/gtest.h>
#include <HashMap.h>
#include <StringHashFunctions.h>

using namespace std;

int hashFunctionCaller(const string &target, int (*func)(const string&)) {
    return func(target);
}

TEST(StringHashFunctionsTest, BasicAssertions) {
    string target = "some target";
    int result = hashFunctionCaller(target, &StringHashFunctions::basicStringHash);
    cout << result << endl;
}

TEST(StringKeyValuePairTest, BasicAssertions) {
    string target = "some target";
    int value = 10;
    StringKeyValuePair<int> pair = StringKeyValuePair<int>(target, &value);

    EXPECT_EQ(target, pair.key);
    EXPECT_EQ(value, *pair.value);
}

//TEST(HashingTest, BasicAssertions) {
//    HashMap<string> map;
//    string key = "test key";
//
//    EXPECT_FALSE(map.exists(key));
//
//    string value = "test value";
//    map.add(key, &value);
//
//    EXPECT_TRUE(map.exists(key));
//
//    string another_key = "other test key";
//    map.add(key, &value);
//}
