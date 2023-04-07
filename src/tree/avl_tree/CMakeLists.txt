add_executable(avl_main main.cpp)

add_library(avl_tree STATIC AVLTree.cpp AVLTree.h)
target_include_directories(avl_tree PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
