//
// Created by Petio Petrov on 2023-04-07.
//

#include <fstream>
#include <cstdlib>
#include "AVLTree.h"

using namespace std;

int main() {
    int element;
    AVLTree<int> tree;

    for (int i = 0; i != 30; ++i) {
        element = rand() % 1000;
        tree.insert(element);
    }

    buildViz(&tree, "avl");

    cout << "depth: " << tree.getDepth() << endl;

    return 0;
}