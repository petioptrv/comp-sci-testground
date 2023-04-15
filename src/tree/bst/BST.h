//
// Created by Petio Petrov on 2023-04-07.
//

#ifndef CODING_INTERVIEW_UNIVERSITY_BST_H
#define CODING_INTERVIEW_UNIVERSITY_BST_H

#include <stack>
#include <string>
#include <sstream>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <functional>

template<class T>
class BSTNode {
public:
    T value;
    BSTNode<T> *leftChild;
    BSTNode<T> *rightChild;

    explicit BSTNode(const T &value);
};

template<class T>
BSTNode<T>::BSTNode(const T& value):
value(value), leftChild(nullptr), rightChild(nullptr) {}

template<class T>
class BST {
public:
    BSTNode<T> *root;

    BST();
    BST<T>* insert(T value);
    void remove(T value);
    std::vector<T> traverse();
    std::vector<T> traverseInOrder();
    void traverseInOrder(const std::function<void(T)>& function);
    T getMin();
    BSTNode<T> *getMin(BSTNode<T> *node);
    int getDepth();
    bool isEmpty();
};

template<class T>
BST<T>::BST(): root(nullptr) {};

template<class T>
BST<T> *BST<T>::insert(T value) { // [ALGO CHALLENGE]
    if (isEmpty()) {
        root = new BSTNode<T>(value);
    } else {
        auto currentNode = root;
        while (currentNode->value != value) {
            if (value < currentNode->value) {
                if (currentNode->leftChild == nullptr) {
                    currentNode->leftChild = new BSTNode<T>(value);
                }
                currentNode = currentNode->leftChild;
            } else {
                if (currentNode->rightChild == nullptr) {
                    currentNode->rightChild = new BSTNode<T>(value);
                }
                currentNode = currentNode->rightChild;
            }
        }
    }
}

template<class T>
void BST<T>::remove(T value) { // [ALGO CHALLENGE]
    BSTNode<T> *currentNode = root;
    BSTNode<T> *parent = nullptr;
    bool isLeftChild = false;

    while (currentNode != nullptr && currentNode->value != value) {
        parent = currentNode;
        if (currentNode->value > value) {
            currentNode = currentNode->leftChild;
            isLeftChild = true;
        } else {
            currentNode = currentNode->rightChild;
            isLeftChild = false;
        }
    }

    if (currentNode != nullptr) {
        if (
            currentNode->leftChild == nullptr && currentNode->rightChild == nullptr
        ) { // case 1: the node is a leaf
            if (currentNode == root) {
                root = nullptr;
            } else if (isLeftChild) {
                parent->leftChild = nullptr;
            } else {
                parent->rightChild = nullptr;
            }
        } else if (
            currentNode->rightChild == nullptr
        ) { // case 2: the node has only a left child
            if (isLeftChild) {
                parent->leftChild = currentNode->leftChild;
            } else {
                parent->rightChild = currentNode->leftChild;
            }
            delete currentNode;
        } else if (
            currentNode->leftChild == nullptr
        ) { // case 3: the node has only a right child
            if (isLeftChild) {
                parent->leftChild = currentNode->rightChild;
            } else {
                parent->rightChild = currentNode->rightChild;
            }
            delete currentNode;
        } else { // case 4: the node has both children
            parent = currentNode;
            BSTNode<T>* temp = currentNode->leftChild;

            while (temp->rightChild != nullptr) {
                parent = temp;
                temp = temp->rightChild;
            }

            if (parent == currentNode) {
                currentNode->leftChild = temp->leftChild;
            } else {
                parent->rightChild = temp->leftChild;
            }

            currentNode->value = temp->value;
            delete temp;
        }
    }
}

template<class T>
std::vector<T> BST<T>::traverse() {
    return traverseInOrder();
}

template<class T>
std::vector<T> BST<T>::traverseInOrder() { // [ALGO CHALLENGE]
    std::vector<T> traversalVector;
    std::stack<BSTNode<T>*> nodeStack;
    BSTNode<T>* currentNode = root;

    while (currentNode != nullptr || !nodeStack.empty()) {
        while (currentNode != nullptr) {
            nodeStack.push(currentNode);
            currentNode = currentNode->leftChild;
        }

        currentNode = nodeStack.top();
        nodeStack.pop();
        traversalVector.push_back(currentNode->value);
        currentNode = currentNode->rightChild;
    }

    return traversalVector;
}

template<class T>
T BST<T>::getMin() {
    return getMin(root)->value;
}

template<class T>
BSTNode<T> *BST<T>::getMin(BSTNode<T> *node) {
    BSTNode<T> *currentNode = node;
    while (currentNode != nullptr && currentNode->leftChild != nullptr) {
        currentNode = currentNode->leftChild;
    }
    return currentNode;
}

template<class T>
int BST<T>::getDepth() { // [ALGO CHALLENGE]
    std::stack<BSTNode<T> *> nodeStack;
    std::stack<int> heightStack;
    BSTNode<T> *currentNode = root;
    int depth = 0;
    int maxDepth = 0;

    while (currentNode != nullptr || !nodeStack.empty()) {
        while (currentNode != nullptr) {
            nodeStack.push(currentNode);
            heightStack.push(depth);

            currentNode = currentNode->leftChild;
            depth += 1;
        }

        currentNode = nodeStack.top();
        nodeStack.pop();
        depth = heightStack.top();
        heightStack.pop();
        maxDepth = depth > maxDepth ? depth : maxDepth;

        currentNode = currentNode->rightChild;
        depth += 1;
    }

    return maxDepth;
}

template<class T>
bool BST<T>::isEmpty() {
    return root == nullptr;
}

//  ======= VISUALIZATION =======

template<class T>
void buildViz(BST<T> *tree, std::string fileName) {
    // Generate the DOT representation and save it to a file
    std::string dot = toDot(tree);
    std::ofstream dotFile(fileName + ".dot");
    dotFile << dot;
    dotFile.close();
    system(("dot -Tpng " + fileName + ".dot -o " + fileName + ".png").c_str());
    remove((fileName + ".dot").c_str());
}

template<class T>
std::string toDot(BST<T> *const tree) {
    std::stringstream ss;
    ss << "digraph BST {" << std::endl;
    ss << "  node [fontname=\"Arial\"];" << std::endl;
    if (tree->root) {
        toDotHelper(ss, tree->root);
    }
    ss << "}" << std::endl;
    return ss.str();
}

template<class T>
void toDotHelper(std::stringstream &ss, BSTNode<T> *node) {
    if (node->leftChild) {
        ss << "  " << node->value << " -> " << node->leftChild->value << ";"
           << std::endl;
        toDotHelper(ss, node->leftChild);
    }
    if (node->rightChild) {
        ss << "  " << node->value << " -> " << node->rightChild->value << ";"
           << std::endl;
        toDotHelper(ss, node->rightChild);
    }
}

#endif //CODING_INTERVIEW_UNIVERSITY_BST_H
