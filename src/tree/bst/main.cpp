//
// Created by Petio Petrov on 2023-04-07.
//

#include <fstream>
#include <cstdlib>
#include "BST.h"

using namespace std;

int main() {
    int element;
    BST<int> tree;

    for (int i = 0; i != 30; ++i) {
        element = rand() % 1000;
        tree.insert(element);
    }

    buildViz(&tree, "bst");

    cout << "depth: " << tree.getDepth() << endl;

    for (auto v: tree.traversePostOrder()) {
        cout << v << " ";
    }

    cout << endl;

    return 0;
}