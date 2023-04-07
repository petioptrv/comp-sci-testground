//
// Created by Petio Petrov on 2022-10-23.
//
#include <iostream>
#include <string>

using namespace std;

void fn(string* strPntr) {
    string* list = new string[10];
    list[0] = *strPntr;
    cout << &list[0] << endl;
    cout << sizeof(&list[0]) << endl;
    cout << sizeof(list[0]) << endl;
    cout << strPntr << endl;
    cout << sizeof(strPntr) << endl;
    cout << sizeof(*strPntr) << endl;
}

int main() {
    string str = "Hello hashing World!";

    fn(&str);

    string* list = new string[10];
    list[1] = str;

//    cout << sizeof(list[0]) << endl;
//    cout << sizeof(list[1]) << endl;
//    cout << list[0] << endl;
//    cout << list[1] << endl;
//    cout << list[2] << endl;
}
