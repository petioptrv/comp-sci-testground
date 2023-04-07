add_executable(bst_main main.cpp)

add_library(bst STATIC BST.cpp BST.h)
target_include_directories(bst PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
