//
// Created by Petio Petrov on 2023-04-07.
//

#include <fstream>
#include "BST.h"

using namespace std;

int main() {
    BST<int> tree;
    tree.insert(1);
    tree.insert(2);
    tree.insert(3);
    tree.insert(4);

    buildViz(&tree, "bst");

    return 0;
}