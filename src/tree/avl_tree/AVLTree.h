//
// Created by Petio Petrov on 2023-04-07.
//

#ifndef CODING_INTERVIEW_UNIVERSITY_AVLTREE_H
#define CODING_INTERVIEW_UNIVERSITY_AVLTREE_H

#include "BST.h"

template<class T>
class AVLNode: public BSTNode<T> {
public:
    unsigned int balance: 2; // 2-bit unsigned int -- 0-1-2
    AVLNode<T> *parent;

    explicit AVLNode(const T &value);
};

template<class T>
AVLNode<T>::AVLNode(const T& value): BSTNode<T>(value), balance(1) {}

template<class T>
class AVLTree : public BST<T> {
public:
    void insert(T value) override;
};

template<class T>
void AVLTree<T>::insert(T value) { // [ALGO CHALLENGE]
    auto currentNode = this->root;
    if (this->isEmpty()) {
        this->root = new AVLNode<T>(value);
    } else {
        while (currentNode->value != value) {
            if (value < currentNode->value) {
                if (currentNode->leftChild == nullptr) {
                    currentNode->leftChild = new AVLNode<T>(value);
                }
                currentNode = currentNode->leftChild;
            } else {
                if (currentNode->rightChild == nullptr) {
                    currentNode->rightChild = new AVLNode<T>(value);
                }
                currentNode = currentNode->rightChild;
            }
        }
    }

    // ==== NEXT ====
    // Currently this method copies the BST method. Now it needs to be adapted to
    // update the balance information of the nodes and then to re-balance the tree.
    // This should make the single test in the AVL tests pass.
};

#endif //CODING_INTERVIEW_UNIVERSITY_AVLTREE_H
