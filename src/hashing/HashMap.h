//
// Created by Petio Petrov on 2022-10-23.
//

#ifndef CODING_HASH_MAP_H
#define CODING_HASH_MAP_H

#include <string>
#include <utility>
#include <StringHashFunctions.h>

template <typename T>
class StringKeyValuePair {
public:
    std::string key;
    const T* value;

    StringKeyValuePair(std::string  key, const T* value) : key(std::move(key)), value(value) {};
    bool empty() {return key.empty();};
    void clear() {
        value = nullptr;
        key.clear();
    };
};

template <typename T>
class HashMap {
public:
    HashMap() {
        table_size_ = 10;
        table_ = new StringKeyValuePair<T>*[table_size_]();
        strHashFunc = &StringHashFunctions::basicStringHash;
    };
    ~HashMap() {
        delete[] table_;
    };
    T* add(const std::string& key, T* value) {
        int hashValue = strHashFunc(key);
        while (table_[hashValue] != nullptr)
        table_[hashValue] = new StringKeyValuePair<T>(key, value);
        return table_[hashValue]->value;
    };
    bool exists(const std::string& key) {
        int hashValue = strHashFunc(key);
        return table_[hashValue] != nullptr;
    };
    T* get(const std::string& key) {
        int hashValue = strHashFunc(key);
        return table_[hashValue]->value;
    };
    T* remove(const std::string& key) {
        int hashValue = strHashFunc(key);
        StringKeyValuePair<T>* ptr = table_[hashValue];
        table_[hashValue] = nullptr;
        return ptr->value;
    };

private:
    int table_size_;
    StringKeyValuePair<T>** table_;
    int (*strHashFunc)(const std::string&);
};


#endif //CODING_HASH_MAP_H
